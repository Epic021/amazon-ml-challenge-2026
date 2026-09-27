"""Third base model for the stacker: XGBoost (depth-wise trees, column subsampling) on the stage-1 features.

A different inductive bias from the leaf-wise LightGBM stage 1, trained on folds --train_folds and
early-stopped on fold 5. Predicts every train pair (out-of-fold outside the training folds) and every test pair,
part file by part file, with the same schema as train.py:
  data/pred/train_{tag}.parquet (s1_id, cand_id, fold, y, p), data/pred/test_{tag}.parquet (s1_id, cand_id, p)

  python src/train_xgb.py --tag x5 --feat_dir data/feat_b5 --train_folds 2,3
"""
import argparse
import glob
import os
import time

import numpy as np
import polars as pl
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
PRED = os.path.join(DATA, "pred")
MODELS = os.path.join(DATA, "models")
NON_FEATURES = {"s1_id", "cand_id", "y", "fold"}
HOLD = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="x5")
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--train_folds", default="2,3")
    ap.add_argument("--drop", default="numx_")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(PRED, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    t0 = time.time()
    folds = [int(x) for x in a.train_folds.split(",")]
    parts = sorted(glob.glob(f"{a.feat_dir}/train.parquet/*.parquet"))
    drop = tuple(x for x in a.drop.split(",") if x)
    cols = [c for c in pl.read_parquet_schema(parts[0]) if c not in NON_FEATURES and not (drop and c.startswith(drop))]
    lf = pl.scan_parquet(parts)

    def matrix(fold_list):
        d = lf.filter(pl.col("fold").is_in(fold_list)).select(cols + ["y"]).collect()
        return d.select(cols).to_numpy().astype(np.float32), d["y"].to_numpy()

    Xtr, ytr = matrix(folds)
    Xho, yho = matrix([HOLD])
    print(f"xgb: {len(cols)} features; train {len(ytr):,} (pos {ytr.mean():.3f}); holdout {len(yho):,} "
          f"({time.time() - t0:.0f}s)", flush=True)
    dtr = xgb.QuantileDMatrix(Xtr, ytr, max_bin=256, nthread=a.threads)
    dho = xgb.QuantileDMatrix(Xho, yho, ref=dtr, nthread=a.threads)
    del Xtr, Xho
    params = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", max_depth=10,
                  eta=0.1, subsample=0.8, colsample_bytree=0.6, colsample_bylevel=0.8, min_child_weight=20,
                  reg_lambda=2.0, max_bin=256, nthread=a.threads, seed=11)
    bst = xgb.train(params, dtr, a.rounds, evals=[(dho, "hold")], early_stopping_rounds=50, verbose_eval=50)
    bst.save_model(f"{MODELS}/xgb_{a.tag}.json")
    print(f"best iteration {bst.best_iteration} ({time.time() - t0:.0f}s)", flush=True)
    del dtr, dho
    rng = (0, bst.best_iteration + 1)
    for split in ("train", "test"):
        out = []
        for f in sorted(glob.glob(f"{a.feat_dir}/{split}.parquet/*.parquet")):
            keep = ["s1_id", "cand_id"] + (["fold", "y"] if split == "train" else [])
            d = pl.read_parquet(f, columns=keep + cols)
            p = bst.inplace_predict(d.select(cols).to_numpy().astype(np.float32), iteration_range=rng)
            out.append(d.select(keep).with_columns(pl.Series("p", p.astype(np.float32))))
        pl.concat(out).write_parquet(f"{PRED}/{split}_{a.tag}.parquet")
        print(f"{split} predictions written ({time.time() - t0:.0f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
