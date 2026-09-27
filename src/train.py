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
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))   # override for smoke runs
FEAT = os.path.join(DATA, "feat")
PRED = os.path.join(DATA, "pred")
MODELS = os.path.join(DATA, "models")

NON_FEATURES = {"s1_id", "cand_id", "y", "fold"}
TRAIN_FOLDS, HOLD_FOLD = (2, 3, 4), 5


def feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURES]


LABEL_EVIDENCE = {"add_lo_": "n_added_unknown", "drop_lo_": "n_dropped_unknown",
                  "aadd_lo_": "aadd_unknown", "adrop_lo_": "adrop_unknown"}
EVIDENCE_TOTAL = {"n_added_unknown": "n_added", "n_dropped_unknown": "n_dropped"}


def evidence_dropout(X: pd.DataFrame, p: float, seed: int = 42) -> pd.DataFrame:
    """On a fraction p of training rows, pretend every name/address word is unseen by the label-based
    log-odds (as French words are): zero those features and set the 'unknown' counts to all words."""
    if p <= 0:
        return X
    m = np.random.default_rng(seed).random(len(X)) < p
    for pref, unk in LABEL_EVIDENCE.items():
        for c in [c for c in X.columns if c.startswith(pref)]:
            X.loc[m, c] = 0.0
        if unk in X.columns and unk in EVIDENCE_TOTAL and EVIDENCE_TOTAL[unk] in X.columns:
            X.loc[m, unk] = X.loc[m, EVIDENCE_TOTAL[unk]]
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=255)
    ap.add_argument("--drop", default="", help="comma-separated feature-name prefixes to exclude")
    ap.add_argument("--train_folds", default="2,3,4", help="S1 folds used for training (holdout is fold 5)")
    ap.add_argument("--feat_dir", default=FEAT, help="directory with train.parquet / test.parquet")
    ap.add_argument("--threads", type=int, default=os.cpu_count(), help="LightGBM threads (two trainings at once)")
    ap.add_argument("--evidence_dropout", type=float, default=0.0, help="fraction of training rows (see evidence_dropout)")
    a = ap.parse_args()
    os.makedirs(PRED, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    t0 = time.time()

    train_folds = tuple(int(x) for x in a.train_folds.split(","))
    assert HOLD_FOLD not in train_folds, "fold 5 is the holdout"
    tr = pd.read_parquet(f"{a.feat_dir}/train.parquet")
    drop = tuple(x for x in a.drop.split(",") if x)
    cols = [c for c in feature_cols(tr) if not (drop and c.startswith(drop))]
    m_tr, m_ho = tr.fold.isin(train_folds).values, (tr.fold == HOLD_FOLD).values
    print(f"{len(cols)} features; train pairs {m_tr.sum():,} (pos {tr.y[m_tr].mean():.3f}), "
          f"holdout pairs {m_ho.sum():,}", flush=True)

    params = dict(objective="binary", learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=a.threads, verbose=-1, seed=42)
    dtr = lgb.Dataset(evidence_dropout(tr.loc[m_tr, cols].copy(), a.evidence_dropout), tr.y[m_tr], free_raw_data=True)
    dho = lgb.Dataset(tr.loc[m_ho, cols], tr.y[m_ho], reference=dtr)
    model = lgb.train(params, dtr, a.rounds, valid_sets=[dho], valid_names=["hold"],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)])
    model.save_model(f"{MODELS}/lgb_{a.tag}.txt")
    imp = pd.Series(model.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print("top features by gain:\n", (imp / imp.sum()).head(25).round(4).to_string(), flush=True)

    tr["p"] = model.predict(tr[cols], num_threads=a.threads).astype(np.float32)
    tr[["s1_id", "cand_id", "fold", "y", "p"]].to_parquet(f"{PRED}/train_{a.tag}.parquet", index=False)
    del tr
    if os.path.exists(f"{a.feat_dir}/test.parquet"):      # file or directory of part files
        te = pd.read_parquet(f"{a.feat_dir}/test.parquet")
        te["p"] = model.predict(te[cols], num_threads=a.threads).astype(np.float32)
        te[["s1_id", "cand_id", "p"]].to_parquet(f"{PRED}/test_{a.tag}.parquet", index=False)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
