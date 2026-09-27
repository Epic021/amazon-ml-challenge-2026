"""Do segment specialists beat the global stacker on their own segment? (mixture-of-experts check)

Segments (where the holdout loss concentrates): empty-address candidates, same-address word swaps,
non-Latin (Indic-script) candidates. For each: LightGBM on segment rows of folds 6-8 (the stacker's
training folds) with the 95 stage-1 features + the three base probabilities (ours, teammate, XGBoost);
compared with the global stacker (--ref) on the segment's fold-5 rows: logloss and AUC.

  python scripts/specialist_test.py --ref b5sb3
"""
import argparse
import os

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
KEY = ["s1_id", "cand_id"]
SEG = {
    "empty_address": pl.col("c_addr_empty") == 1,
    "same_addr_swap": (pl.col("num_rel") == 1) & (pl.col("addr_tset") >= 90) & (pl.col("n_added") >= 1) & (pl.col("n_dropped") >= 1),
    "non_latin": pl.col("c_nonlatin") == 1,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="b5sb3")
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    fold = (pl.col("s1_id").str.slice(3).cast(pl.Int64) % 10).cast(pl.Int8)
    parts = sorted(__import__("glob").glob(f"{DATA}/feat_b5/train.parquet/*.parquet"))
    cols = [c for c in pl.read_parquet_schema(parts[0]) if c not in ("s1_id", "cand_id", "y", "fold") and not c.startswith("numx_")]
    f = pl.scan_parquet(parts).filter(fold.is_in([5, 6, 7, 8])).select(KEY + ["y"] + cols).collect()
    for tag, name in (("b5", "p_o"), ("x5", "p_x")):
        f = f.join(pl.read_parquet(f"{DATA}/pred/train_{tag}.parquet", columns=KEY + ["p"]).rename({"p": name}), on=KEY, how="left")
    f = f.join(pl.read_parquet(f"{DATA}/friend_v2/train_oof_probs.parquet", columns=KEY + ["p"]).rename({"p": "p_t"}), on=KEY, how="left")
    f = f.join(pl.read_parquet(f"{DATA}/pred/train_{a.ref}.parquet", columns=KEY + ["p"]).rename({"p": "p_ref"}), on=KEY, how="left")
    f = f.with_columns(pl.col("p_t").fill_null(0.0), fold.alias("fold"))
    feats = cols + ["p_o", "p_x", "p_t"]
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=a.threads, verbose=-1, seed=3)
    for seg, cond in SEG.items():
        d = f.filter(cond)
        tr, ho = d.filter(pl.col("fold") != 5), d.filter(pl.col("fold") == 5)
        m = lgb.train(params, lgb.Dataset(tr.select(feats).to_pandas(), tr["y"].to_numpy()), 2000,
                      valid_sets=[lgb.Dataset(ho.select(feats).to_pandas(), ho["y"].to_numpy())],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        ps = m.predict(ho.select(feats).to_pandas(), num_threads=a.threads)
        y = ho["y"].to_numpy()
        pr = np.clip(ho["p_ref"].fill_null(0.0).to_numpy(), 1e-6, 1 - 1e-6)
        ps = np.clip(ps, 1e-6, 1 - 1e-6)
        print(f"{seg:15s} holdout rows {len(y):>9,} pos {y.mean():.3f} | global stacker logloss {log_loss(y, pr):.5f} "
              f"AUC {roc_auc_score(y, pr):.5f} | SPECIALIST logloss {log_loss(y, ps):.5f} AUC {roc_auc_score(y, ps):.5f} "
              f"(iters {m.best_iteration})", flush=True)


if __name__ == "__main__":
    main()
