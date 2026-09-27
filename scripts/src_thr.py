"""Per-source thresholds (S2 vs S3) on a stacker's calibrated, soft-exclusive probabilities.

Grid over (thr_S2, thr_S3) on the holdout (fold 5); compares with the single best threshold and reports
a paired bootstrap CI. --write applies the best pair to test.

  python scripts/src_thr.py --tag b5sb2 --calib_folds 9 [--write --out output_srcthr]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from decode import soft_excl, write, bootstrap_gain  # noqa: E402
from metric import per_entity_f05  # noqa: E402

DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
GRID = np.round(np.arange(0.45, 0.91, 0.05), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--calib_folds", default="9")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default="output_srcthr")
    a = ap.parse_args()
    tr = pd.read_parquet(f"{DATA}/pred/train_{a.tag}.parquet")
    cal = tr[tr.fold.isin(tuple(int(x) for x in a.calib_folds.split(",")))]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(cal.p.values, cal.y.values)
    tr["p"] = iso.predict(tr.p.values).astype(np.float32)
    ex = soft_excl(tr[["s1_id", "cand_id", "p"]])
    s1 = pd.read_parquet(f"{DATA}/parquet/train_s1.parquet", columns=["entity_id"]).entity_id
    hold = s1[(s1.str[3:].astype(np.int64) % 10) == 5]
    ho = ex[ex.s1_id.isin(set(hold))]
    ho = ho[ho.p >= GRID.min()]
    is3 = (ho.cand_id.str[:2] == "S3").values
    truth = pd.read_parquet(f"{DATA}/parquet/train_pairs.parquet")
    res, per = {}, {}
    for t2 in GRID:
        for t3 in GRID:
            keep = np.where(is3, ho.p.values >= t3, ho.p.values >= t2)
            f = per_entity_f05(ho[keep], truth, hold)
            res[(t2, t3)], per[(t2, t3)] = f.mean(), f
    res = pd.Series(res)
    best = res.idxmax()
    same = max(((t, t) for t in GRID), key=lambda k: res[k])
    g, lo, hi = bootstrap_gain(per[best], per[same])
    print(f"single best thr {same[0]}: {res[same]:.5f} | per-source best S2 {best[0]} S3 {best[1]}: {res[best]:.5f} "
          f"gain {g:+.5f} [95% CI {lo:+.5f}, {hi:+.5f}]", flush=True)
    print(res.sort_values(ascending=False).head(8).round(5).to_string(), flush=True)
    if a.write:
        te = pd.read_parquet(f"{DATA}/pred/test_{a.tag}.parquet")
        te["p"] = iso.predict(te.p.values).astype(np.float32)
        ext = soft_excl(te[["s1_id", "cand_id", "p"]])
        is3t = (ext.cand_id.str[:2] == "S3").values
        pred = ext[np.where(is3t, ext.p.values >= best[1], ext.p.values >= best[0])]
        s1t = pd.read_parquet(f"{DATA}/parquet/test_s1.parquet", columns=["entity_id"]).entity_id
        out = os.path.join(ROOT, a.out)
        os.makedirs(out, exist_ok=True)
        write(pred, s1t, f"{out}/matching_results.tsv", "matched_entity_ids")
        write(te[["s1_id", "cand_id"]], s1t, f"{out}/candidate_pairs.tsv", "candidate_entity_ids")
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
