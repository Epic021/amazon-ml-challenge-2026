"""Leave-one-country-out pilot: an offline proxy for France (a country with no labels).

Train a small LightGBM on ONE country (folds 2-4, 10% of S1), calibrate on the same country's folds 8-9,
and score the OTHER country's holdout (fold 5) with the production decode (soft exclusivity, thr 0.55).
Also reports the in-country holdout. A feature group whose removal raises the cross-country score
does not transfer, so it is a suspect for France.

  python scripts/loco_pilot.py --train_country US --drop add_lo_,drop_lo_ --name noLO
"""
import argparse
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from decode import soft_excl  # noqa: E402
from metric import per_entity_f05  # noqa: E402
from train import feature_cols  # noqa: E402

DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
SUB = pl.col("s1_id").str.slice(3).cast(pl.Int64) % 1000 < 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_country", required=True, choices=["US", "India"])
    ap.add_argument("--drop", default="", help="comma-separated feature-name prefixes to exclude")
    ap.add_argument("--name", default="")
    ap.add_argument("--feat_dir", default=f"{DATA}/feat_b5")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=400)
    a = ap.parse_args()
    t0 = time.time()
    s1 = pl.read_parquet(f"{DATA}/parquet/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    tr = pl.scan_parquet(f"{a.feat_dir}/train.parquet/*.parquet").filter(SUB).collect().join(s1, on="s1_id")
    tr = tr.to_pandas()
    drop = tuple(x for x in a.drop.split(",") if x) + ("numx_",)
    cols = [c for c in feature_cols(tr) if c != "country" and not c.startswith(drop)]
    own = (tr.country == a.train_country).values
    m_tr = own & tr.fold.isin((2, 3, 4)).values
    params = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=a.threads, verbose=-1, seed=42)
    model = lgb.train(params, lgb.Dataset(tr.loc[m_tr, cols], tr.y[m_tr]), a.rounds)
    ev = tr[tr.fold.isin((5, 8, 9))][["s1_id", "cand_id", "fold", "y", "country"]].copy()
    ev["p"] = model.predict(tr.loc[ev.index, cols], num_threads=a.threads)
    cal = ev[ev.fold.isin((8, 9)) & (ev.country == a.train_country)]          # labels of the TRAIN country only
    ev["p"] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(cal.p, cal.y).predict(ev.p)
    truth = pd.read_parquet(f"{DATA}/parquet/train_pairs.parquet")
    res = {}
    for c in ("India", "US"):
        ho = soft_excl(ev[(ev.fold == 5) & (ev.country == c)][["s1_id", "cand_id", "p"]])
        ids = s1.filter((pl.col("country") == c) & SUB & ((pl.col("s1_id").str.slice(3).cast(pl.Int64) % 10) == 5))["s1_id"].to_pandas()
        res[c] = per_entity_f05(ho[ho.p >= 0.55], truth, ids).mean()
    other = "India" if a.train_country == "US" else "US"
    print(f"LOCO {a.name or '-':12s} train={a.train_country:5s} drop={a.drop or '-'} features={len(cols)} "
          f"in-country={res[a.train_country]:.5f} CROSS({other})={res[other]:.5f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
