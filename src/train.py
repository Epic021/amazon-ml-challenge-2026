"""Train LightGBM pair model on fold 2-4, early-stop on fold 5 (holdout), predict train + test.

  python src/train.py [--tag v1]
"""
import argparse
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(ROOT, "data", "feat")
PRED = os.path.join(ROOT, "data", "pred")
MODELS = os.path.join(ROOT, "models")

NON_FEATURES = {"s1_id", "cand_id", "y", "fold"}
TRAIN_FOLDS, HOLD_FOLD = (2, 3, 4), 5


def feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=255)
    a = ap.parse_args()
    os.makedirs(PRED, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    t0 = time.time()

    tr = pd.read_parquet(f"{FEAT}/train.parquet")
    cols = feature_cols(tr)
    m_tr, m_ho = tr.fold.isin(TRAIN_FOLDS).values, (tr.fold == HOLD_FOLD).values
    print(f"{len(cols)} features; train pairs {m_tr.sum():,} (pos {tr.y[m_tr].mean():.3f}), "
          f"holdout pairs {m_ho.sum():,}", flush=True)

    params = dict(objective="binary", learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=os.cpu_count(), verbose=-1, seed=42)
    dtr = lgb.Dataset(tr.loc[m_tr, cols], tr.y[m_tr], free_raw_data=True)
    dho = lgb.Dataset(tr.loc[m_ho, cols], tr.y[m_ho], reference=dtr)
    model = lgb.train(params, dtr, a.rounds, valid_sets=[dho], valid_names=["hold"],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    model.save_model(f"{MODELS}/lgb_{a.tag}.txt")
    imp = pd.Series(model.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print("top features by gain:\n", (imp / imp.sum()).head(25).round(4).to_string(), flush=True)

    tr["p"] = model.predict(tr[cols], num_threads=os.cpu_count()).astype(np.float32)
    tr[["s1_id", "cand_id", "fold", "y", "p"]].to_parquet(f"{PRED}/train_{a.tag}.parquet", index=False)
    del tr
    if os.path.isfile(f"{FEAT}/test.parquet"):
        te = pd.read_parquet(f"{FEAT}/test.parquet")
        te["p"] = model.predict(te[cols], num_threads=os.cpu_count()).astype(np.float32)
        te[["s1_id", "cand_id", "p"]].to_parquet(f"{PRED}/test_{a.tag}.parquet", index=False)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
