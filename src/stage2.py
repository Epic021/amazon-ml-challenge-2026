"""Stage 2: re-score every pair using the stage-1 probabilities of its competitors (stacking).

Stage 1 judges each (S1, record) pair alone. The decisive information is relational:
  - record side: the best stage-1 p this record has with ANY OTHER S1, the margin to it, the sum of
    p over its S1 candidates (soft exclusivity), how many S1 claim it with p > 0.5
  - S1 side: the pair's p rank within the S1, gap to the S1's best p, the S1's sum of p (expected set
    size), how many candidates exceed 0.5
  - twins in probability space: best p among OTHER-source candidates of the same S1 that share this
    record's number set / core name
Stage-1 predictions are out-of-sample on folds 0-1 and 5-9 (stage 1 trains on 2-4), so stage 2 trains
on folds 6-7, isotonic-calibrates on 8-9 and is scored on 5 (the holdout). On test, stage-1 p is
out-of-sample by construction.

  python src/stage2.py --tag b4 --feat_dir data/feat_b4      # writes data/pred/{train,test}_{tag}s2.parquet
Then:  python src/decode.py --tag b4s2 --calib_folds 8,9
"""
import argparse
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
PRED = os.path.join(DATA, "pred")
NORM = os.path.join(DATA, "norm")
S2_TRAIN, HOLD = (6, 7), 5
# stage-1 features carried into stage 2 (strongest per gain; keeps stage 2 fast)
BASE = ["score", "rank_q", "rank_s", "gap_c", "gap_s1", "c_margin", "is_c_best", "mutual_best", "n_c_s1",
        "n_s1_cands", "num_rel", "num_min_lev", "num_first_logdiff", "add_lo_min", "add_lo_sum", "padd_min",
        "addr_tset", "core_tset", "name_tset", "c_addr_empty", "c_nonlatin", "c_src_s3",
        "s1_core_n_s1", "c_core_n_s1", "n_retrievers"]


def relational(df: pl.DataFrame, split: str) -> pl.DataFrame:
    rec = pl.concat([pl.read_parquet(f"{NORM}/{split}_s{k}.parquet", columns=["id", "src", "nums", "name_core"])
                     for k in (1, 2, 3)]).rename({"id": "cand_id"})
    df = df.join(rec, on="cand_id", how="left")
    df = df.with_columns(
        # record side
        pl.col("p").max().over("cand_id").alias("r_best"),
        pl.col("p").sum().over("cand_id").alias("r_sum"),
        (pl.col("p") > 0.5).sum().over("cand_id").alias("r_n05"),
        pl.col("p").top_k(2).min().over("cand_id").alias("_r_top2min"),
        pl.len().over("cand_id").alias("_r_n"),
        # S1 side
        pl.col("p").rank("ordinal", descending=True).over("s1_id").cast(pl.Int16).alias("s_rank"),
        pl.col("p").max().over("s1_id").alias("s_best"),
        pl.col("p").sum().over("s1_id").alias("s_sum"),
        (pl.col("p") > 0.5).sum().over("s1_id").alias("s_n05"),
    )
    df = df.with_columns(
        # best p among the record's OTHER S1 candidates
        pl.when(pl.col("p") >= pl.col("r_best"))
          .then(pl.when(pl.col("_r_n") >= 2).then(pl.col("_r_top2min")).otherwise(0.0))
          .otherwise(pl.col("r_best")).alias("r_other_best"),
        (pl.col("s_best") - pl.col("p")).alias("s_gap"),
        (pl.col("p") / pl.col("s_sum").clip(1e-6)).alias("p_share_s"),
        (pl.col("p") / pl.col("r_sum").clip(1e-6)).alias("p_share_r"),
    ).with_columns((pl.col("p") - pl.col("r_other_best")).alias("r_margin"))
    # twins in probability space: other-source candidates of the same S1 with the same numbers / name
    for key, name in (("nums", "twin_p_num"), ("name_core", "twin_p_name")):
        g = (df.filter(pl.col(key) != "").group_by(["s1_id", key, "src"])
               .agg(pl.col("p").max().alias("_mx")))
        other = (g.join(g, on=["s1_id", key], suffix="_o").filter(pl.col("src") != pl.col("src_o"))
                  .group_by(["s1_id", key, "src"]).agg(pl.col("_mx_o").max().alias(name)))
        df = df.join(other, on=["s1_id", key, "src"], how="left").with_columns(pl.col(name).fill_null(-1.0))
    return df.drop(["_r_top2min", "_r_n", "src", "nums", "name_core"])


def load(split: str, tag: str, feat_dir: str) -> pl.DataFrame:
    p = pl.read_parquet(f"{PRED}/{split}_{tag}.parquet")
    cols = ["s1_id", "cand_id"] + BASE
    f = pl.from_pandas(pd.read_parquet(f"{feat_dir}/{split}.parquet", columns=cols))
    return relational(p.join(f, on=["s1_id", "cand_id"], how="left"), split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--rounds", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()
    tr = load("train", a.tag, a.feat_dir).to_pandas()
    feats = [c for c in tr.columns if c not in ("s1_id", "cand_id", "fold", "y")]
    print(f"stage 2: {len(feats)} features, {len(tr):,} train pairs ({time.time() - t0:.0f}s)", flush=True)
    m_tr, m_ho = tr.fold.isin(S2_TRAIN).values, (tr.fold == HOLD).values
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=os.cpu_count(), verbose=-1, seed=7)
    model = lgb.train(params, lgb.Dataset(tr.loc[m_tr, feats], tr.y[m_tr]), a.rounds,
                      valid_sets=[lgb.Dataset(tr.loc[m_ho, feats], tr.y[m_ho])], valid_names=["hold"],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("stage-2 top features:\n", (imp / imp.sum()).head(15).round(4).to_string(), flush=True)
    out = tr[["s1_id", "cand_id", "fold", "y"]].copy()
    out["p"] = model.predict(tr[feats], num_threads=os.cpu_count()).astype(np.float32)
    out.to_parquet(f"{PRED}/train_{a.tag}s2.parquet", index=False)
    del tr
    te = load("test", a.tag, a.feat_dir).to_pandas()
    te_out = te[["s1_id", "cand_id"]].copy()
    te_out["p"] = model.predict(te[feats], num_threads=os.cpu_count()).astype(np.float32)
    te_out.to_parquet(f"{PRED}/test_{a.tag}s2.parquet", index=False)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
