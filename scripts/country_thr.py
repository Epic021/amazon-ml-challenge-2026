"""DIAGNOSTIC ONLY: re-decode a model's test predictions with a different threshold for ONE country.

Used as an LB probe: India/US rows are byte-identical to the base decode, so the LB change is 0.15 x the
France F0.5 change. The result tells us the direction and size of France's error; it is NOT a tuned
setting to ship (the organizers re-run the code; nothing may be fitted to the test file).

  python scripts/country_thr.py --tag b5sb --calib_folds 8,9 --thr 0.65 --country France --country_thr 0.75 \
      --out output_probe_fr075
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from decode import soft_excl, write  # noqa: E402

DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--calib_folds", default="8,9")
    ap.add_argument("--thr", type=float, required=True, help="threshold for all other countries (the base decode's)")
    ap.add_argument("--country", default="France")
    ap.add_argument("--country_thr", type=float, nargs="+", required=True)
    ap.add_argument("--out", required=True, help="output dir prefix; one dir per country_thr")
    a = ap.parse_args()
    tr = pd.read_parquet(f"{DATA}/pred/train_{a.tag}.parquet")
    cal = tr[tr.fold.isin(tuple(int(x) for x in a.calib_folds.split(",")))]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(cal.p, cal.y)
    del tr
    te = pd.read_parquet(f"{DATA}/pred/test_{a.tag}.parquet")
    te["p"] = iso.predict(te.p).astype(np.float32)
    ex = soft_excl(te[["s1_id", "cand_id", "p"]])
    s1 = pd.read_parquet(f"{DATA}/parquet/test_s1.parquet", columns=["entity_id", "country"])
    ex["country"] = ex.s1_id.map(s1.set_index("entity_id").country)
    for ct in a.country_thr:
        thr = np.where(ex.country.values == a.country, ct, a.thr)
        pred = ex[ex.p.values >= thr]
        out = os.path.join(ROOT, f"{a.out}_{int(round(ct * 100)):03d}")
        os.makedirs(out, exist_ok=True)
        write(pred, s1.entity_id, f"{out}/matching_results.tsv", "matched_entity_ids")
        write(te[["s1_id", "cand_id"]], s1.entity_id, f"{out}/candidate_pairs.tsv", "candidate_entity_ids")
        per = pred.groupby("s1_id").size().reindex(s1.entity_id, fill_value=0)
        print(f"{a.country} thr {ct} (others {a.thr}) -> {out}: preds/S1 "
              f"{per.groupby(s1.country.values).mean().round(3).to_dict()}", flush=True)


if __name__ == "__main__":
    main()
