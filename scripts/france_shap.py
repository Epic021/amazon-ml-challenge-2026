"""Which features push France's same-address word-swap pairs down? Mean SHAP per feature, France vs India/US.

Class: same house number, near-identical address, one name word added and one dropped, this S1 is the
record's best S1, no other-source twin with the same core name (in train such pairs are ~70% true).

  python scripts/france_shap.py --tag b5 --feat_dir data/feat_b5
"""
import argparse
import os

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
SWAP = ((pl.col("num_rel") == 1) & (pl.col("addr_tset") >= 90) & (pl.col("n_added") >= 1)
        & (pl.col("n_dropped") >= 1) & (pl.col("is_c_best") == 1) & (pl.col("twin_name") <= 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b5")
    ap.add_argument("--feat_dir", default=f"{DATA}/feat_b5")
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    m = lgb.Booster(model_file=f"{DATA}/models/lgb_{a.tag}.txt")
    F = m.feature_name()
    te = pl.scan_parquet(f"{a.feat_dir}/test.parquet/*.parquet").filter(SWAP).collect()
    s1 = pl.read_parquet(f"{DATA}/parquet/test_s1.parquet", columns=["entity_id", "country"])
    te = te.join(s1.rename({"entity_id": "s1_id"}), on="s1_id")
    sh, val, raw = {}, {}, {}
    for c in ("France", "India", "US"):
        X = te.filter(pl.col("country") == c).sample(a.n, seed=0).select(F).to_pandas()
        s = m.predict(X, pred_contrib=True, num_threads=16)
        sh[c], val[c], raw[c] = s[:, :-1].mean(0), X.mean().values, s.sum(1).mean()
    print("mean raw score (log-odds):", {c: round(v, 2) for c, v in raw.items()})
    d = sh["France"] - (sh["India"] + sh["US"]) / 2
    o = np.argsort(d)
    row = lambda i: f"  {F[i]:22s} {d[i]:+.3f}   value France {val['France'][i]:.3g} | India {val['India'][i]:.3g} | US {val['US'][i]:.3g}"
    print("features pushing France DOWN vs India/US (SHAP difference):")
    print("\n".join(row(i) for i in o[:18]))
    print("pushing France UP:")
    print("\n".join(row(i) for i in o[-8:][::-1]))


if __name__ == "__main__":
    main()
