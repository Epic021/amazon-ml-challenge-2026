"""Evidence gate for "more training data": small LightGBM on a given set of training folds, scored on
the holdout (fold 5) with the production decode rule (isotonic on folds 8-9, soft exclusivity,
threshold 0.55). Run several fold sets in parallel and compare; only if the score still rises
with more folds is a full larger training run worth it.

  python scripts/pilot_curve.py --folds 2,3 --threads 16
"""
import argparse
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from decode import soft_excl  # noqa: E402
from metric import per_entity_f05  # noqa: E402
from train import feature_cols, evidence_dropout  # noqa: E402

DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--feat_dir", default=os.path.join(DATA, "feat"))
    ap.add_argument("--drop", default="", help="feature-name prefixes to exclude")
    ap.add_argument("--evidence_dropout", type=float, default=0.0)
    ap.add_argument("--sample", type=float, default=1.0, help="score only S1 with id%%1000 < sample*1000")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    t0 = time.time()
    folds = tuple(int(x) for x in a.folds.split(","))
    tr = pd.read_parquet(f"{a.feat_dir}/train.parquet")
    drop = tuple(x for x in a.drop.split(",") if x)
    cols = [c for c in feature_cols(tr) if not (drop and c.startswith(drop))]
    m_tr = tr.fold.isin(folds).values
    params = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=a.threads, verbose=-1, seed=42)
    model = lgb.train(params, lgb.Dataset(evidence_dropout(tr.loc[m_tr, cols].copy(), a.evidence_dropout),
                                          tr.y[m_tr]), a.rounds)
    ev = tr[tr.fold.isin((5, 8, 9))][["s1_id", "cand_id", "fold", "y"]].copy()
    ev["p"] = model.predict(tr.loc[ev.index, cols], num_threads=a.threads)
    cal = ev[ev.fold.isin((8, 9))]
    ev["p"] = IsotonicRegression(out_of_bounds="clip").fit(cal.p, cal.y).predict(ev.p)
    ho = soft_excl(ev[ev.fold == 5])
    truth = pd.read_parquet(f"{DATA}/parquet/train_pairs.parquet")
    s1 = pd.read_parquet(f"{DATA}/parquet/train_s1.parquet", columns=["entity_id"]).entity_id
    num = s1.str[3:].astype(np.int64)
    hold = s1[((num % 10) == 5) & ((num % 1000) < a.sample * 1000)]
    f = per_entity_f05(ho[ho.p >= 0.55], truth, hold).mean()
    print(f"PILOT {a.name} folds={a.folds} drop={a.drop or '-'} ed={a.evidence_dropout} hold_S1={len(hold):,} train_pairs={m_tr.sum():,} holdout_macroF05={f:.5f} ({time.time() - t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
