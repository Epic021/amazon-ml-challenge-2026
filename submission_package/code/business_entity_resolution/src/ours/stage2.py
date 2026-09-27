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

--friend: stacked blend. Adds the teammate's probability (data/friend/, OOF on train) as a second input,
with the same relational features computed on it, so stage 2 learns when to trust which model. The
pair set is the union of both candidate sets (missing p -> 0). Writes {tag}sb instead of {tag}s2.
"""
import argparse
import glob
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
HOLD = 5
KEY = ["s1_id", "cand_id"]
# stage-1 features carried into stage 2 (strongest per gain; keeps stage 2 fast)
BASE = ["score", "rank_q", "rank_s", "gap_c", "gap_s1", "c_margin", "is_c_best", "mutual_best", "n_c_s1",
        "n_s1_cands", "num_rel", "num_min_lev", "num_first_logdiff", "add_lo_min", "add_lo_sum", "padd_min",
        "addr_tset", "core_tset", "name_tset", "c_addr_empty", "c_nonlatin", "c_src_s3",
        "s1_core_n_s1", "c_core_n_s1", "n_retrievers"]
# extra pair features for the stacked blend: which kind of name difference, sound key, twins
PAIR = ["n_added", "n_dropped", "n_added_unknown", "n_dropped_unknown", "padd_max", "pdrop_max",
        "legal_conflict", "snd_tset", "twin_num", "twin_name", "addr_diff_explained", "name_diff_explained"]


REL = ["r_best", "r_sum", "r_n05", "s_rank", "s_best", "s_sum", "s_n05", "r_other_best", "s_gap", "p_share_s",
       "p_share_r", "r_margin", "twin_p_num", "twin_p_name"]


def relational(df: pl.DataFrame, split: str, p: str = "p", pre: str = "") -> pl.DataFrame:
    """Competition features of probability column `p`, named with prefix `pre`."""
    df = df.rename({p: "_p"})
    rec = pl.concat([pl.read_parquet(f"{NORM}/{split}_s{k}.parquet", columns=["id", "src", "nums", "name_core"])
                     for k in (1, 2, 3)]).rename({"id": "cand_id"})
    df = df.join(rec, on="cand_id", how="left")
    df = df.with_columns(
        # record side
        pl.col("_p").max().over("cand_id").alias("r_best"),
        pl.col("_p").sum().over("cand_id").alias("r_sum"),
        (pl.col("_p") > 0.5).sum().over("cand_id").alias("r_n05"),
        pl.col("_p").top_k(2).min().over("cand_id").alias("_r_top2min"),
        pl.len().over("cand_id").alias("_r_n"),
        # S1 side
        pl.col("_p").rank("ordinal", descending=True).over("s1_id").cast(pl.Int16).alias("s_rank"),
        pl.col("_p").max().over("s1_id").alias("s_best"),
        pl.col("_p").sum().over("s1_id").alias("s_sum"),
        (pl.col("_p") > 0.5).sum().over("s1_id").alias("s_n05"),
    )
    df = df.with_columns(
        # best p among the record's OTHER S1 candidates
        pl.when(pl.col("_p") >= pl.col("r_best"))
          .then(pl.when(pl.col("_r_n") >= 2).then(pl.col("_r_top2min")).otherwise(0.0))
          .otherwise(pl.col("r_best")).alias("r_other_best"),
        (pl.col("s_best") - pl.col("_p")).alias("s_gap"),
        (pl.col("_p") / pl.col("s_sum").clip(1e-6)).alias("p_share_s"),
        (pl.col("_p") / pl.col("r_sum").clip(1e-6)).alias("p_share_r"),
    ).with_columns((pl.col("_p") - pl.col("r_other_best")).alias("r_margin"))
    # twins in probability space: other-source candidates of the same S1 with the same numbers / name
    for key, name in (("nums", "twin_p_num"), ("name_core", "twin_p_name")):
        g = (df.filter(pl.col(key) != "").group_by(["s1_id", key, "src"])
               .agg(pl.col("_p").max().alias("_mx")))
        other = (g.join(g, on=["s1_id", key], suffix="_o").filter(pl.col("src") != pl.col("src_o"))
                  .group_by(["s1_id", key, "src"]).agg(pl.col("_mx_o").max().alias(name)))
        df = df.join(other, on=["s1_id", key, "src"], how="left").with_columns(pl.col(name).fill_null(-1.0))
    df = df.drop(["_r_top2min", "_r_n", "src", "nums", "name_core"]).rename({"_p": p})
    return df.rename({c: pre + c for c in REL}) if pre else df


def load(split: str, tag: str, feat_dir: str, friend: bool = False, friend_dir: str = "friend",
         all_feats: bool = False, extra: tuple = (), context_folds: tuple = (5, 6, 7, 8, 9),
         third: str = "") -> pl.DataFrame:
    p = pl.read_parquet(f"{PRED}/{split}_{tag}.parquet", columns=KEY + ["p"])
    fold = (pl.col("s1_id").str.slice(3).cast(pl.Int64) % 10).cast(pl.Int8)
    if split == "train":
        p = p.filter(fold.is_in(context_folds))           # rows that give competition context (and predictions)
    if all_feats:                                         # every stage-1 feature (minus the dropped numx_)
        first = sorted(glob.glob(f"{feat_dir}/{split}.parquet/*.parquet"))[0]
        cols = KEY + [c for c in pl.read_parquet_schema(first)
                      if c not in KEY + ["y", "fold"] and not c.startswith("numx_")]
    else:
        cols = KEY + BASE + (PAIR if friend else [])
    if friend:
        t = pl.read_parquet(f"{DATA}/{friend_dir}/{'train_oof_probs' if split == 'train' else 'test_probs'}.parquet",
                            columns=KEY + ["p"]).rename({"p": "p_t"})
        if split == "train":
            t = t.filter(fold.is_in(context_folds))
        p = p.join(t, on=KEY, how="full", coalesce=True).with_columns(
            pl.col("p").is_null().cast(pl.Int8).alias("o_missing"), pl.col("p_t").is_null().cast(pl.Int8).alias("t_missing"))
        p = p.with_columns(pl.col("p").fill_null(0.0), pl.col("p_t").fill_null(0.0))
        p = p.with_columns(((pl.col("p") + pl.col("p_t")) / 2).alias("p_m"), (pl.col("p") - pl.col("p_t")).alias("p_diff"))
    f = pl.scan_parquet(f"{feat_dir}/{split}.parquet/*.parquet").select(cols)
    df = p.join(f.collect(), on=KEY, how="left")
    if split == "train":
        tr = pl.read_parquet(f"{DATA}/parquet/train_pairs.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
        df = df.join(tr, on=KEY, how="left").with_columns(pl.col("y").fill_null(0), fold.alias("fold"))
    if third:                                             # a third base model's probability (e.g. XGBoost)
        x = pl.read_parquet(f"{PRED}/{split}_{third}.parquet", columns=KEY + ["p"]).rename({"p": "p_x"})
        df = df.join(x, on=KEY, how="left").with_columns(pl.col("p_x").fill_null(0.0))
    for name in extra:                                    # extra pair features, e.g. data/extra/{split}_diffsup.parquet
        df = df.join(pl.read_parquet(f"{DATA}/extra/{split}_{name}.parquet"), on=KEY, how="left")
    df = relational(df, split)
    if friend:
        df = relational(relational(df, split, "p_t", "t_"), split, "p_m", "m_")
    if third:
        df = relational(df, split, "p_x", "x_")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--friend", action="store_true", help="stacked blend with the teammate's probabilities")
    ap.add_argument("--friend_dir", default="friend", help="teammate probabilities under data/")
    ap.add_argument("--train_folds", default="6,7", help="stage-2 training folds (5 = holdout; calibrate on the rest)")
    ap.add_argument("--all_feats", action="store_true", help="carry every stage-1 feature, not just BASE/PAIR")
    ap.add_argument("--out_tag", default=None, help="output tag (default {tag}sb / {tag}s2)")
    ap.add_argument("--extra", default="", help="comma-separated extra feature sets under data/extra/{split}_<name>.parquet")
    ap.add_argument("--third", default="", help="tag of a third base model's predictions (data/pred/{split}_<tag>.parquet)")
    ap.add_argument("--context_folds", default="5,6,7,8,9",
                    help="train folds whose pairs enter the competition features (all S1 are present on test). "
                         "Stage-1 p is out-of-fold on 0,1,5-9; 2-4 are stage-1 training folds")
    a = ap.parse_args()
    t0 = time.time()
    out_tag = a.out_tag or a.tag + ("sb" if a.friend else "s2")
    train_folds = tuple(int(x) for x in a.train_folds.split(","))
    kw = dict(friend=a.friend, friend_dir=a.friend_dir, all_feats=a.all_feats,
              extra=tuple(x for x in a.extra.split(",") if x),
              context_folds=tuple(int(x) for x in a.context_folds.split(",")), third=a.third)
    tr = load("train", a.tag, a.feat_dir, **kw).to_pandas()
    feats = [c for c in tr.columns if c not in ("s1_id", "cand_id", "fold", "y")]
    print(f"stage 2: {len(feats)} features, {len(tr):,} train pairs ({time.time() - t0:.0f}s)", flush=True)
    m_tr, m_ho = tr.fold.isin(train_folds).values, (tr.fold == HOLD).values
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
    out.to_parquet(f"{PRED}/train_{out_tag}.parquet", index=False)
    del tr
    te = load("test", a.tag, a.feat_dir, **kw)
    step = 10_000_000                                     # predict in slices to bound pandas memory
    ps = [model.predict(te.slice(i, step).select(feats).to_pandas(), num_threads=os.cpu_count())
          for i in range(0, te.height, step)]
    te_out = te.select(["s1_id", "cand_id"]).to_pandas()
    te_out["p"] = np.concatenate(ps).astype(np.float32)
    te_out.to_parquet(f"{PRED}/test_{out_tag}.parquet", index=False)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
