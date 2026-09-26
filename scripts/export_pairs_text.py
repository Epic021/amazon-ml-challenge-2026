"""Export pair text for the neural pair scorers (src/xenc.py). Runs on the CPU VM; only these parquet
files go to the GPU box.

Writes data/xenc/:
  train.parquet        training pairs from folds 0-4 (out-of-sample for the stacker's folds 5-9), hard-mined:
                         swap class (same number, near-identical address, one word added + one dropped),
                         grey pairs (teammate OOF p or our p in [0.02, 0.98]), ours-vs-teammate disagreement > 0.5,
                         plus a sample of easy pairs. `llm` = 1 marks the hardest subset for the LLM.
  score_train.parquet  shortlist of folds 5-9 (stacker train / calibration / holdout)
  score_test.parquet   shortlist of test
  train_feats.parquet  the pair features of train.parquet's rows (scripts/transfer_check.py runs on the pod)
Columns: s1_id, cand_id, fold, y (-1 on test), country, prio, llm, a (S1 text), b (record text).
`prio` orders scoring so a time-capped run covers what the stacker needs first:
  0 folds 5-7 · 1 test France · 2 folds 8-9 · 3 test India/US.
Text is the RAW name | address (accents and scripts kept; the tokenizer handles them).

  python scripts/export_pairs_text.py --tag b5 --feat_dir data/feat_b5
"""
import argparse
import os

import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
PQ, PRED, OUT = f"{DATA}/parquet", f"{DATA}/pred", f"{DATA}/xenc"
KEY = ["s1_id", "cand_id"]
FOLD = (pl.col("s1_id").str.slice(3).cast(pl.Int64) % 10).cast(pl.Int8)
SWAP_COLS = ["num_rel", "addr_tset", "n_added", "n_dropped", "is_c_best", "twin_name"]
# same definition as scripts/france_shap.py, minus the twin/best-S1 conditions (keep all swap pairs)
SWAP = ((pl.col("num_rel") == 1) & (pl.col("addr_tset") >= 90) & (pl.col("n_added") >= 1)
        & (pl.col("n_dropped") >= 1))


def friend_path(split: str) -> str:
    return f"{DATA}/friend/{'train_oof_probs' if split == 'train' else 'test_probs'}.parquet"


def load_pairs(split: str, tag: str, feat_dir: str) -> pl.DataFrame:
    """Our p + teammate p (outer) + swap-class columns + country; train also fold and y."""
    o = pl.scan_parquet(f"{PRED}/{split}_{tag}.parquet").select(KEY + ["p"])
    if os.path.isfile(friend_path(split)):
        t = pl.scan_parquet(friend_path(split)).select(KEY + [pl.col("p").alias("p_t")])
        df = o.join(t, on=KEY, how="full", coalesce=True)
    else:
        df = o.with_columns(pl.lit(None, pl.Float32).alias("p_t"))
    f = pl.scan_parquet(f"{feat_dir}/{split}.parquet/*.parquet").select(KEY + SWAP_COLS)
    df = df.join(f, on=KEY, how="left").with_columns(FOLD.alias("fold"), SWAP.fill_null(False).alias("swap"))
    s1 = pl.scan_parquet(f"{PQ}/{split}_s1.parquet").select(pl.col("entity_id").alias("s1_id"), "country")
    df = df.join(s1, on="s1_id", how="left")
    if split == "train":
        tr = pl.scan_parquet(f"{PQ}/train_pairs.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
        df = df.join(tr, on=KEY, how="left").with_columns(pl.col("y").fill_null(0).cast(pl.Int8))
    else:
        df = df.with_columns(pl.lit(-1, pl.Int8).alias("y"), pl.lit(-1, pl.Int8).alias("fold"))
    return df.collect()


def grey(col: str, lo: float, hi: float) -> pl.Expr:
    return pl.col(col).is_between(lo, hi).fill_null(False)


def disagree(thr: float) -> pl.Expr:
    return ((pl.col("p").fill_null(0) - pl.col("p_t").fill_null(0)).abs() > thr) & pl.col("p_t").is_not_null()


def attach_text(df: pl.DataFrame, split: str) -> pl.DataFrame:
    rec = pl.concat([pl.scan_parquet(f"{PQ}/{split}_s{k}.parquet").select(
        "entity_id", "src", pl.concat_str([pl.col("business_name"), pl.col("business_address")], separator=" | ").alias("t"))
        for k in (1, 2, 3)])
    a = rec.filter(pl.col("src") == "S1").select(pl.col("entity_id").alias("s1_id"), pl.col("t").alias("a"))
    b = rec.filter(pl.col("src") != "S1").select(
        pl.col("entity_id").alias("cand_id"), pl.concat_str([pl.col("src"), pl.col("t")], separator=": ").alias("b"))
    return df.lazy().join(a, on="s1_id", how="left").join(b, on="cand_id", how="left").collect()


COLS = ["s1_id", "cand_id", "fold", "y", "country", "prio", "llm", "a", "b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="our stage-1 predictions data/pred/{split}_{tag}.parquet")
    ap.add_argument("--feat_dir", required=True, help="feature dir with the swap-class columns")
    ap.add_argument("--n_train", type=int, default=1_500_000, help="training pairs for the cross-encoder")
    ap.add_argument("--n_llm", type=int, default=350_000, help="hardest training pairs for the LLM")
    ap.add_argument("--easy_frac", type=float, default=0.1, help="minimum share of easy pairs in the training set")
    ap.add_argument("--p_min", type=float, default=0.005, help="shortlist: keep pairs with p >= p_min ...")
    ap.add_argument("--top", type=int, default=2, help="... or ranked <= top in their S1")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # ---- training pairs (folds 0-4). Our p is in-sample there (folds 2-4 train it, 0-1 build its word
    # tables), so hardness leans on the teammate's out-of-fold p and on the label-free swap class.
    tr = load_pairs("train", a.tag, a.feat_dir)
    tr_all = tr
    tr = tr.filter(pl.col("fold") <= 4)
    hard = pl.col("swap") | grey("p_t", 0.02, 0.98) | grey("p", 0.02, 0.98) | disagree(0.5)
    hardest = pl.col("swap") | grey("p_t", 0.1, 0.9) | disagree(0.5)
    tr = tr.with_columns(hard.alias("_hard"), hardest.alias("_hardest"))
    h, e = tr.filter(pl.col("_hard")), tr.filter(~pl.col("_hard"))
    n_easy = min(e.height, max(int(a.easy_frac * a.n_train), a.n_train - h.height))   # >= easy_frac of the set
    h = h.sample(min(h.height, a.n_train - n_easy), seed=a.seed)
    tr = pl.concat([h, e.sample(n_easy, seed=a.seed)])
    llm_ids = tr.filter(pl.col("_hardest")).select(KEY)
    llm_ids = llm_ids.sample(min(llm_ids.height, a.n_llm), seed=a.seed).with_columns(pl.lit(1, pl.Int8).alias("llm"))
    tr = (tr.join(llm_ids, on=KEY, how="left").with_columns(pl.col("llm").fill_null(0), pl.lit(0, pl.Int8).alias("prio"))
            .sample(fraction=1.0, shuffle=True, seed=a.seed))
    tr = attach_text(tr, "train").select(COLS)
    tr.write_parquet(f"{OUT}/train.parquet")
    pl.scan_parquet(f"{a.feat_dir}/train.parquet/*.parquet").join(tr.lazy().select(KEY), on=KEY, how="semi") \
      .collect().write_parquet(f"{OUT}/train_feats.parquet")
    print(f"train: {tr.height:,} pairs (pos {tr['y'].mean():.3f}); llm subset {tr['llm'].sum():,} "
          f"(pos {tr.filter(pl.col('llm') == 1)['y'].mean():.3f}); by country "
          f"{dict(tr.group_by('country').len().iter_rows())}", flush=True)

    # ---- shortlists to score
    llm_flag = (pl.col("swap") | grey("p", 0.05, 0.95) | grey("p_t", 0.05, 0.95) | disagree(0.3)).cast(pl.Int8)
    for split in ("train", "test"):
        df = tr_all.filter(pl.col("fold") >= 5) if split == "train" else load_pairs("test", a.tag, a.feat_dir)
        rank = pl.col("p").fill_null(0).rank("ordinal", descending=True).over("s1_id")
        keep = (pl.col("p") >= a.p_min).fill_null(False) | (pl.col("p_t") >= a.p_min).fill_null(False) | (rank <= a.top)
        df = df.filter(keep).with_columns(llm_flag.alias("llm"))
        if split == "train":
            prio = pl.when(pl.col("fold") <= 7).then(0).otherwise(2)
        else:
            prio = pl.when(pl.col("country") == "France").then(1).otherwise(3)
        df = attach_text(df.with_columns(prio.cast(pl.Int8).alias("prio")), split).select(COLS)
        df.write_parquet(f"{OUT}/score_{split}.parquet")
        n_llm = df.filter(pl.col("llm") == 1).group_by("prio").len().sort("prio")
        print(f"score_{split}: {df.height:,} pairs ({df.height / df['s1_id'].n_unique():.2f} per S1); "
              f"llm-flagged by prio {dict(n_llm.iter_rows())}", flush=True)


if __name__ == "__main__":
    main()
