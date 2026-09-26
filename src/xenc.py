"""Neural pair scorers (Tier 2): a multilingual cross-encoder (mDeBERTa-v3-base, full fine-tune) and a small
LLM (Qwen3-4B-Base, LoRA) with a binary classification head. Input is the RAW text of both records, so
the pretrained multilingual knowledge covers words the GBDTs have never seen (French).

The GPU budget is fixed, so every command is safe to kill and re-run:
  train  checkpoints every --ckpt_minutes and resumes from the last checkpoint; --max_minutes stops it
         cleanly and saves the model (meta.json says whether the schedule finished)
  score  writes one parquet part per --chunk rows, skips parts that exist, stops at --max_minutes;
         rows are scored in `prio` order (see scripts/export_pairs_text.py)

  python src/xenc.py probe   --model mdeberta --data data/xenc/train.parquet
  python src/xenc.py train   --model mdeberta --data data/xenc/train.parquet --out data/xenc/m_mdeb
  python src/xenc.py train   --model qwen3-4b --data data/xenc/train.parquet --subset llm --out data/xenc/m_qwen
  python src/xenc.py score   --model_dir data/xenc/m_mdeb --data data/xenc/score_train.parquet,data/xenc/score_test.parquet \
                             --out data/xenc/p_mdeb
  python src/xenc.py collect --parts data/xenc/p_mdeb --tag xmdeb      # CPU side -> data/pred/{split}_xmdeb.parquet

--tiny builds a small random model of the same architecture (tokenizer only is downloaded): CPU smoke tests.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
MODELS = {"mdeberta": "microsoft/mdeberta-v3-base",     # MIT, 86M backbone + 190M embeddings
          "qwen3-4b": "Qwen/Qwen3-4B-Base",              # Apache-2.0, 4.0B total
          "gemma4-e2b": "google/gemma-4-E2B"}            # Apache-2.0, 5.1B total (fallback, untested)
DECODER = {"qwen3-4b", "gemma4-e2b"}                     # LoRA + single-string prompt
DEFAULTS = {  # batch, lr, score batch
    "mdeberta": (128, 3e-5, 1024),
    "qwen3-4b": (32, 1e-4, 256),
    "gemma4-e2b": (32, 1e-4, 256),
}
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- model
def tiny_config(hf: str):
    cfg = AutoConfig.from_pretrained(hf)
    for k, v in dict(num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=16, pooler_hidden_size=64).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def build(name: str, tiny: bool = False, lora_r: int = 16):
    hf = MODELS.get(name, name)
    tok = AutoTokenizer.from_pretrained(hf)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    decoder = name in DECODER
    if tiny:
        torch.manual_seed(0)
        cfg = tiny_config(hf)
        cfg.num_labels = 1
        model = AutoModelForSequenceClassification.from_config(cfg, dtype=torch.float32)
    else:
        dtype = torch.bfloat16 if (decoder and DEV == "cuda") else torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(hf, num_labels=1, dtype=dtype)
    model.config.pad_token_id = tok.pad_token_id
    if decoder:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", r=lora_r, lora_alpha=2 * lora_r,
                                                 lora_dropout=0.05, target_modules=LORA_TARGETS))
        for p in model.parameters():                      # LoRA + head in fp32, frozen base in bf16
            if p.requires_grad:
                p.data = p.data.float()
    return model.to(DEV), tok


def trainable_state(model) -> dict:
    return {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}


def save_weights(model, path: str, full: bool):
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()} if full else trainable_state(model), path)


def load_weights(model, path: str):
    sd = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    trainable = {k for k, v in model.named_parameters() if v.requires_grad}
    assert not (trainable & set(missing)), f"trainable weights missing from {path}"


def load_for_scoring(model_dir: str):
    meta = json.load(open(f"{model_dir}/meta.json"))
    model, tok = build(meta["model"], meta["tiny"], meta["lora_r"])
    load_weights(model, f"{model_dir}/weights.pt")
    model.eval()
    return model, tok, meta


# ---------------------------------------------------------------- data
def read(paths: str, subset: str = "", country: str = "") -> pl.DataFrame:
    """One or more comma-separated parquet files; `split` = test for files named *test*, else train."""
    df = pl.concat([pl.read_parquet(p).with_columns(
        pl.lit("test" if "test" in os.path.basename(p) else "train").alias("split")) for p in paths.split(",")],
        how="vertical_relaxed")
    if subset:
        df = df.filter(pl.col(subset) == 1)
    if country:
        df = df.filter(pl.col("country").is_in(country.split(",")))
    return df


def encode(tok, a: list, b: list, decoder: bool, max_len: int) -> list:
    if decoder:
        text = [f"Business A: {x}\nBusiness B: {y}\nSame business:" for x, y in zip(a, b)]
        return tok(text, truncation=True, max_length=max_len)["input_ids"]
    return tok(a, b, truncation="longest_first", max_length=max_len)["input_ids"]


def collate(tok, seqs: list) -> dict:
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    return {"input_ids": ids.to(DEV, non_blocking=True), "attention_mask": att.to(DEV, non_blocking=True)}


def length_batches(lengths: np.ndarray, bs: int, seed: int) -> list:
    """Shuffle, sort by length inside windows of 64 batches (little padding), then shuffle the batches."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(lengths))
    win = bs * 64
    out = []
    for i in range(0, len(idx), win):
        w = idx[i:i + win]
        w = w[np.argsort(lengths[w], kind="stable")]
        out += [w[j:j + bs] for j in range(0, len(w), bs)]
    return [out[i] for i in rng.permutation(len(out))]


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16) if DEV == "cuda" else torch.autocast("cpu", enabled=False)


def logits(model, batch) -> torch.Tensor:
    with autocast():
        return model(**batch).logits[:, 0].float()


# ---------------------------------------------------------------- commands
def cmd_train(a):
    t0 = time.time()
    bs, lr, _ = DEFAULTS.get(a.model, (32, 1e-4, 256))
    bs, lr = a.batch or bs, a.lr or lr
    os.makedirs(a.out, exist_ok=True)
    model, tok = build(a.model, a.tiny, a.lora_r)
    if a.init:                                            # continue from a trained model (e.g. round 2)
        load_weights(model, f"{a.init}/weights.pt")
    decoder = a.model in DECODER
    df = read(a.data, a.subset, a.country)
    rates = json.load(open(a.rates)) if os.path.isfile(a.rates) else {}
    if a.fit_minutes and a.model in rates:                # size the set so the full schedule fits the time
        rate = rates[a.model]["train"]
        a.limit = min(a.limit or df.height, int(rate * a.fit_minutes * 60 / a.epochs))
        log(f"fit {a.fit_minutes} min at {rate:,.0f} pairs/s -> {a.limit:,} of {df.height:,} pairs")
    elif a.fit_minutes:
        log(f"no measured rate for {a.model} in {a.rates}: training on all pairs, --max_minutes caps it")
    if a.limit:
        df = df.head(a.limit)
    ids = encode(tok, df["a"].to_list(), df["b"].to_list(), decoder, a.max_len)
    y = torch.tensor(df["y"].to_numpy(), dtype=torch.float32)
    lengths = np.array([len(s) for s in ids])
    log(f"train {a.model}: {len(ids):,} pairs (pos {y.mean():.3f}), mean {lengths.mean():.0f} tokens, "
        f"batch {bs}, lr {lr}, {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M trainable")
    batches = [b for e in range(a.epochs) for b in length_batches(lengths, bs, a.seed + e)]
    total = len(batches)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01, fused=DEV == "cuda")
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    ckpt, step = f"{a.out}/ckpt.pt", 0
    if os.path.isfile(ckpt):
        st = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(st["model"], strict=False)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step = st["step"]
        log(f"resumed at step {step}/{total}")

    def checkpoint():
        torch.save({"model": trainable_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step}, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)

    model.train()
    t_ck, t_log, seen, loss_ema = time.time(), time.time(), 0, None
    bar = tqdm(total=total, initial=step, desc=f"train {a.model}", unit="step", mininterval=10)
    while step < total:
        if (time.time() - t0) / 60 > a.max_minutes:
            log(f"time cap {a.max_minutes} min reached at step {step}/{total}")
            break
        b = batches[step]
        out = logits(model, collate(tok, [ids[i] for i in b]))
        loss = F.binary_cross_entropy_with_logits(out, y[b].to(DEV))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        seen += len(b)
        loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()
        bar.update(1)
        bar.set_postfix(loss=f"{loss_ema:.4f}", refresh=False)
        if time.time() - t_log > 60 or step == total:
            rate = seen / (time.time() - t_log)
            mem = torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else 0
            log(f"step {step}/{total} loss {loss_ema:.4f} {rate:,.0f} pairs/s, "
                f"ETA {(total - step) * bs / max(rate, 1e-9) / 60:.0f} min, peak mem {mem:.1f} GiB")
            t_log, seen = time.time(), 0
        if time.time() - t_ck > a.ckpt_minutes * 60:
            checkpoint()
            t_ck = time.time()
    bar.close()
    checkpoint()
    save_weights(model, f"{a.out}/weights.pt", full=a.tiny or a.model not in DECODER)
    json.dump({"model": a.model, "tiny": a.tiny, "lora_r": a.lora_r, "max_len": a.max_len, "steps": step,
               "total_steps": total, "complete": step >= total, "n_pairs": len(ids), "subset": a.subset,
               "country": a.country}, open(f"{a.out}/meta.json", "w"), indent=1)
    log(f"saved {a.out} ({step}/{total} steps) in {(time.time() - t0) / 60:.1f} min")


@torch.inference_mode()
def predict(model, tok, ids: list, bs: int) -> np.ndarray:
    order = np.argsort([len(s) for s in ids], kind="stable")
    p = np.empty(len(ids), dtype=np.float32)
    for i in range(0, len(order), bs):
        b = order[i:i + bs]
        p[b] = torch.sigmoid(logits(model, collate(tok, [ids[j] for j in b]))).cpu().numpy()
    return p


def cmd_score(a):
    t0 = time.time()
    model, tok, meta = load_for_scoring(a.model_dir)
    decoder = meta["model"] in DECODER
    bs = a.batch or DEFAULTS.get(meta["model"], (0, 0, 256))[2]
    df = read(a.data, a.subset, a.country)
    df = df.with_columns((pl.col("a").str.len_chars() + pl.col("b").str.len_chars()).alias("_len")) \
           .sort(["prio", "_len"]).drop("_len")
    os.makedirs(a.out, exist_ok=True)
    n_parts = math.ceil(df.height / a.chunk)
    log(f"score {meta['model']} ({a.model_dir}): {df.height:,} pairs in {n_parts} parts of {a.chunk:,}")
    done = 0
    for k in tqdm(range(n_parts), desc=f"score {meta['model']}", unit="part", mininterval=10):
        part = f"{a.out}/part-{k:05d}.parquet"
        if os.path.isfile(part):
            continue
        if (time.time() - t0) / 60 > a.max_minutes:
            log(f"time cap {a.max_minutes} min reached; {k}/{n_parts} parts done")
            break
        c = df.slice(k * a.chunk, a.chunk)
        t1 = time.time()
        ids = encode(tok, c["a"].to_list(), c["b"].to_list(), decoder, meta["max_len"])
        p = predict(model, tok, ids, bs)
        c.select(["s1_id", "cand_id", "split", "fold", "y", "country"]).with_columns(pl.Series("p", p)) \
         .write_parquet(part + ".tmp")
        os.replace(part + ".tmp", part)
        done += c.height
        log(f"part {k + 1}/{n_parts}: {c.height / (time.time() - t1):,.0f} pairs/s")
    log(f"scored {done:,} pairs in {(time.time() - t0) / 60:.1f} min")


def cmd_probe(a):
    """Measure train and score throughput (and peak memory) on a sample; project hours for the planned sizes."""
    bs, lr, sbs = DEFAULTS.get(a.model, (32, 1e-4, 256))
    bs, sbs = a.batch or bs, a.score_batch or sbs
    model, tok = build(a.model, a.tiny, a.lora_r)
    decoder = a.model in DECODER
    df = read(a.data, a.subset)
    df = df.sample(min(a.n, df.height), seed=0)
    ids = encode(tok, df["a"].to_list(), df["b"].to_list(), decoder, a.max_len)
    y = torch.tensor(df["y"].to_numpy(), dtype=torch.float32)
    lengths = np.array([len(s) for s in ids])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    batches = length_batches(lengths, bs, 0)[:a.steps + 3]
    model.train()
    for k, b in enumerate(batches):
        if k == 3:                                       # warm-up steps excluded from timing
            sync()
            t, n = time.time(), 0
        loss = F.binary_cross_entropy_with_logits(logits(model, collate(tok, [ids[i] for i in b])), y[b].to(DEV))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if k >= 3:
            n += len(b)
    sync()
    tr_rate = n / (time.time() - t)
    model.eval()
    predict(model, tok, ids[:sbs], sbs)
    sync()
    t = time.time()
    predict(model, tok, ids, sbs)
    sync()
    sc_rate = len(ids) / (time.time() - t)
    mem = torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else 0
    log(f"probe {a.model}: mean {lengths.mean():.0f} tokens; train {tr_rate:,.0f} pairs/s (batch {bs}), "
        f"score {sc_rate:,.0f} pairs/s (batch {sbs}); peak mem {mem:.1f} GiB")
    rates = json.load(open(a.rates)) if os.path.isfile(a.rates) else {}
    rates[a.model] = {"train": tr_rate, "score": sc_rate, "tokens": float(lengths.mean()), "peak_gib": mem}
    json.dump(rates, open(a.rates, "w"), indent=1)
    log(f"projection: train {a.plan_train:,} pairs = {a.plan_train / tr_rate / 3600:.2f} h; "
        f"score {a.plan_score:,} pairs = {a.plan_score / sc_rate / 3600:.2f} h")


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def cmd_collect(a):
    """CPU side: merge score parts into data/pred/{split}_{tag}.parquet (the schema train.py writes);
    print AUC / log-loss per country wherever labels exist."""
    parts = sorted(glob.glob(f"{a.parts}/**/part-*.parquet", recursive=True))
    df = pl.concat([pl.read_parquet(x) for x in parts]).unique(["split", "s1_id", "cand_id"])
    for (split,), g in df.group_by(["split"]):
        log(f"{split}: {g.height:,} scored pairs from {len(parts)} parts")
        if split == "train":
            report(g)
        if not a.report_only:
            cols = ["s1_id", "cand_id", "fold", "y", "p"] if split == "train" else ["s1_id", "cand_id", "p"]
            os.makedirs(f"{DATA}/pred", exist_ok=True)
            g.select(cols).write_parquet(f"{DATA}/pred/{split}_{a.tag}.parquet")
            log(f"  -> data/pred/{split}_{a.tag}.parquet")


def report(df: pl.DataFrame):
    from sklearn.metrics import log_loss, roc_auc_score
    for (c,), g in df.group_by(["country"]):
        y, p = g["y"].to_numpy(), g["p"].to_numpy()
        if 0 < y.mean() < 1:
            log(f"  {c}: n {len(y):,} AUC {roc_auc_score(y, p):.4f} logloss {log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)):.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data")
    common.add_argument("--subset", default="", help="column that must be 1 (e.g. llm)")
    common.add_argument("--country", default="", help="comma-separated countries to keep")
    common.add_argument("--batch", type=int, default=0)
    common.add_argument("--max_len", type=int, default=128)
    common.add_argument("--lora_r", type=int, default=16)
    common.add_argument("--tiny", action="store_true", help="small random model (CPU smoke tests)")
    common.add_argument("--max_minutes", type=float, default=1e9)
    t = sub.add_parser("train", parents=[common])
    t.add_argument("--model", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--lr", type=float, default=0.0)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--ckpt_minutes", type=float, default=15)
    t.add_argument("--init", default="", help="model dir to start from (weights.pt)")
    t.add_argument("--fit_minutes", type=float, default=0, help="limit pairs so training takes about this long")
    t.add_argument("--rates", default=f"{DATA}/xenc/rates.json", help="throughput measured by probe")
    s = sub.add_parser("score", parents=[common])
    s.add_argument("--model_dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--chunk", type=int, default=200_000)
    p = sub.add_parser("probe", parents=[common])
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--score_batch", type=int, default=0)
    p.add_argument("--plan_train", type=int, default=1_500_000)
    p.add_argument("--plan_score", type=int, default=11_000_000)
    p.add_argument("--rates", default=f"{DATA}/xenc/rates.json", help="where the measured throughput is stored")
    c = sub.add_parser("collect")
    c.add_argument("--parts", required=True, help="dir with train/ and test/ part folders")
    c.add_argument("--tag", required=True)
    c.add_argument("--report_only", action="store_true", help="metrics only (e.g. the transfer check)")
    a = ap.parse_args()
    {"train": cmd_train, "score": cmd_score, "probe": cmd_probe, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
