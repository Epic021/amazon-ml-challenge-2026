"""E1: can a fine-tuned multilingual bi-encoder retrieve the true pairs that ALL our retrievers miss (India)?

Missed set: fold-5 India true pairs absent from the stage-2 pair universe (our union + teammate's candidates).
Control: a sample of fold-5 India true pairs that we do retrieve.
Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0), fine-tuned with in-batch
contrastive loss (MNRL) on India true pairs of folds 2-4 (record text -> S1 text). Text = "name | address".
Search: record -> S1 over ALL India S1 (cosine, FAISS flat). Reports recall@k, pretrained vs fine-tuned.

  /work/venv_nn/bin/python scripts/e1_biencoder.py --n_train 150000
"""
import argparse
import multiprocessing as mp
import os
import time

import faiss
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
KEY = ["s1_id", "cand_id"]
NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
KS = (1, 5, 10, 20, 50)


def texts(df):
    return (df["business_name"].fill_null("") + " | " + df["business_address"].fill_null("")).to_list()


def _enc_worker(args):
    path, chunk, threads = args
    torch.set_num_threads(threads)
    m = SentenceTransformer(path, device="cpu")
    m.max_seq_length = 64
    with torch.no_grad():
        return m.encode(chunk, batch_size=256, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def encode_parallel(path, txt, n_proc=8, threads=8):
    """Encode with n_proc spawned processes x threads (one process cannot use 64 cores on a small model)."""
    step = len(txt) // n_proc + 1
    chunks = [(path, txt[i:i + step], threads) for i in range(0, len(txt), step)]
    with mp.get_context("spawn").Pool(n_proc) as pool:
        return np.vstack(pool.map(_enc_worker, chunks))


def recall(path, s1_txt, s1_ids, q_txt, q_true, tag, e_s1=None):
    t0 = time.time()
    if e_s1 is None:
        e_s1 = encode_parallel(path, s1_txt)
    e_q = encode_parallel(path, q_txt, n_proc=4, threads=16)
    index = faiss.IndexFlatIP(e_s1.shape[1])
    index.add(e_s1)
    _, nn = index.search(e_q, max(KS))
    pos = {s: i for i, s in enumerate(s1_ids)}
    tgt = np.array([pos.get(s, -1) for s in q_true])
    hit = {k: float(np.mean([(tgt[i] in nn[i, :k]) for i in range(len(tgt))])) for k in KS}
    print(f"  [{tag}] recall@k {hit}  ({time.time() - t0:.0f}s)", flush=True)
    return nn, tgt, e_s1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--n_control", type=int, default=5000)
    ap.add_argument("--skip_pretrained", action="store_true", help="skip the pretrained baseline (already measured)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    fold = pl.col("s1_id").str.slice(3).cast(pl.Int64) % 10
    s1 = pl.read_parquet(f"{DATA}/parquet/train_s1.parquet", columns=["entity_id", "business_name", "business_address", "country"])
    s1 = s1.filter(pl.col("country") == "India").rename({"entity_id": "s1_id"})
    rec = pl.concat([pl.read_parquet(f"{DATA}/parquet/train_s{k}.parquet", columns=["entity_id", "business_name", "business_address"])
                     for k in (2, 3)]).rename({"entity_id": "cand_id"})
    truth = pl.read_parquet(f"{DATA}/parquet/train_pairs.parquet").join(s1.select("s1_id"), on="s1_id")
    universe = pl.read_parquet(f"{DATA}/pred/train_b5sb3.parquet", columns=KEY).filter(fold == 5).with_columns(pl.lit(1).alias("found"))
    h = truth.filter(fold == 5).join(universe, on=KEY, how="left").with_columns(pl.col("found").fill_null(0))
    missed = h.filter(pl.col("found") == 0).join(rec, on="cand_id")
    control = h.filter(pl.col("found") == 1).sample(a.n_control, seed=1).join(rec, on="cand_id")
    print(f"India fold-5 true pairs {h.height:,}; MISSED by all retrievers {missed.height:,} "
          f"({missed.height / h.height:.4f}); control {control.height:,} ({time.time() - t0:.0f}s)", flush=True)
    print("missed examples:", flush=True)
    ex = missed.head(8).join(s1.select("s1_id", pl.col("business_name").alias("s1_name"), pl.col("business_address").alias("s1_addr")), on="s1_id")
    for r in ex.iter_rows(named=True):
        print(f"   S1 [{r['s1_name']} | {r['s1_addr']}]  <-  rec [{r['business_name']} | {r['business_address']}]", flush=True)

    s1_ids, s1_txt = s1["s1_id"].to_list(), texts(s1)
    model = SentenceTransformer(NAME, device="cpu")
    model.max_seq_length = 64
    if not a.skip_pretrained:
      print("PRETRAINED:", flush=True)
      _, _, e_pre = recall(NAME, s1_txt, s1_ids, texts(missed), missed["s1_id"].to_list(), "missed")
      recall(NAME, s1_txt, s1_ids, texts(control), control["s1_id"].to_list(), "control", e_s1=e_pre)
      del e_pre

    tr = truth.filter(fold.is_in([2, 3, 4])).sample(a.n_train, seed=2, shuffle=True).join(rec, on="cand_id") \
              .join(s1.select("s1_id", pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1")), on="s1_id")
    q = texts(tr)
    d = (tr["n1"].fill_null("") + " | " + tr["a1"].fill_null("")).to_list()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    steps = len(q) // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    model.train()
    t1 = time.time()
    for s in range(steps):
        qa, da = q[s * a.batch:(s + 1) * a.batch], d[s * a.batch:(s + 1) * a.batch]
        ea = F.normalize(model(model.tokenize(qa))["sentence_embedding"], dim=-1)
        eb = F.normalize(model(model.tokenize(da))["sentence_embedding"], dim=-1)
        loss = F.cross_entropy(ea @ eb.T * 20.0, torch.arange(len(qa)))
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad()
        if s % 50 == 0:
            print(f"  step {s}/{steps} loss {loss.item():.4f} ({time.time() - t1:.0f}s)", flush=True)
    model.eval()
    ft = f"{DATA}/models/e1_biencoder_india"
    model.save(ft)
    print("FINE-TUNED:", flush=True)
    nn, tgt, e_ft = recall(ft, s1_txt, s1_ids, texts(missed), missed["s1_id"].to_list(), "missed")
    recall(ft, s1_txt, s1_ids, texts(control), control["s1_id"].to_list(), "control", e_s1=e_ft)
    got = [i for i in range(len(tgt)) if tgt[i] in nn[i, :20]]
    print(f"recovered@20 examples ({len(got):,} of {len(tgt):,}):", flush=True)
    for i in got[:8]:
        r = missed.row(i, named=True)
        print(f"   S1 {r['s1_id']}  <-  rec [{r['business_name']} | {r['business_address']}]", flush=True)
    print(f"E1 DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
