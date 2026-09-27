"""Per-country thresholds chosen on the labelled holdout (India, US); unseen countries (France) keep the
global best. Isotonic on --calib_folds, soft exclusivity over all rows, grid on fold 5, bootstrap CI vs the
single global threshold. --write applies it to test.

  python scripts/country_hold_thr.py --tag b5sb3 --calib_folds 9 [--write --out output_ctry]
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
    ap.add_argument("--out", default="output_ctry")
    a = ap.parse_args()
    tr = pd.read_parquet(f"{DATA}/pred/train_{a.tag}.parquet")
    cal = tr[tr.fold.isin(tuple(int(x) for x in a.calib_folds.split(",")))]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(cal.p.values, cal.y.values)
    tr["p"] = iso.predict(tr.p.values).astype(np.float32)
    ex = soft_excl(tr[["s1_id", "cand_id", "p"]])
    s1 = pd.read_parquet(f"{DATA}/parquet/train_s1.parquet", columns=["entity_id", "country"])
    hold = s1[(s1.entity_id.str[3:].astype(np.int64) % 10) == 5]
    ho = ex[ex.s1_id.isin(set(hold.entity_id))]
    ho = ho[ho.p >= GRID.min()]
    truth = pd.read_parquet(f"{DATA}/parquet/train_pairs.parquet")
    per = {t: per_entity_f05(ho[ho.p >= t], truth, hold.entity_id) for t in GRID}
    ctry = hold.set_index("entity_id").country
    glob_t = max(GRID, key=lambda t: per[t].mean())
    best = {}
    for c in ("India", "US"):
        ids = ctry.index[ctry.values == c]
        best[c] = max(GRID, key=lambda t: per[t].reindex(ids).mean())
        print(f"{c}: best thr {best[c]} ({per[best[c]].reindex(ids).mean():.5f}) vs global {glob_t} "
              f"({per[glob_t].reindex(ids).mean():.5f})", flush=True)
    mixed = pd.concat([per[best[c]].reindex(ctry.index[ctry.values == c]) for c in ("India", "US")])
    g, lo, hi = bootstrap_gain(mixed, per[glob_t])
    print(f"global thr {glob_t}: {per[glob_t].mean():.5f} | per-country {best}: {mixed.mean():.5f} "
          f"gain {g:+.5f} [95% CI {lo:+.5f}, {hi:+.5f}]", flush=True)
    if a.write:
        te = pd.read_parquet(f"{DATA}/pred/test_{a.tag}.parquet")
        te["p"] = iso.predict(te.p.values).astype(np.float32)
        ext = soft_excl(te[["s1_id", "cand_id", "p"]])
        s1t = pd.read_parquet(f"{DATA}/parquet/test_s1.parquet", columns=["entity_id", "country"])
        cmap = s1t.set_index("entity_id").country
        thr = ext.s1_id.map(cmap).map(lambda c: best.get(c, glob_t)).values
        pred = ext[ext.p.values >= thr]
        out = os.path.join(ROOT, a.out)
        os.makedirs(out, exist_ok=True)
        write(pred, s1t.entity_id, f"{out}/matching_results.tsv", "matched_entity_ids")
        write(te[["s1_id", "cand_id"]], s1t.entity_id, f"{out}/candidate_pairs.tsv", "candidate_entity_ids")
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
