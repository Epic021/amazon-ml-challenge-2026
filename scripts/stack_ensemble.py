"""S2 of the clutch plan: average of calibrated stacker predictions (e.g. v1 b5sb + v2 b5sb2) on the holdout.

Each model is isotonic-calibrated on fold 9 (the only fold neither stacker trained on besides 8 for v1),
averaged, soft exclusivity, threshold grid on fold 5. --write produces the test TSVs.

  python scripts/stack_ensemble.py --tags b5sb,b5sb2 [--write --out output_ens]
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
from metric import per_entity_f05  # noqa: E402

DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
KEY = ["s1_id", "cand_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="b5sb,b5sb2")
    ap.add_argument("--calib_fold", type=int, default=9)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default="output_ens")
    a = ap.parse_args()
    tags = a.tags.split(",")
    tr, isos = None, {}
    for t in tags:
        d = pd.read_parquet(f"{DATA}/pred/train_{t}.parquet")
        c = d[d.fold == a.calib_fold]
        isos[t] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(c.p, c.y)
        d = d[d.fold == 5].assign(**{f"p_{t}": lambda z, t=t: isos[t].predict(z.p)}).drop(columns="p")
        tr = d if tr is None else tr.merge(d.drop(columns=["fold", "y"]), on=KEY, how="outer")
    tr = tr.fillna({f"p_{t}": 0.0 for t in tags})
    truth = pd.read_parquet(f"{DATA}/parquet/train_pairs.parquet")
    s1 = pd.read_parquet(f"{DATA}/parquet/train_s1.parquet", columns=["entity_id", "country"])
    hold = s1[(s1.entity_id.str[3:].astype(np.int64) % 10) == 5]
    res = {}
    for name, p in [(t, tr[f"p_{t}"]) for t in tags] + [("MEAN", tr[[f"p_{t}" for t in tags]].mean(axis=1))]:
        ex = soft_excl(tr[KEY].assign(p=p.values))
        for thr in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75):
            f = per_entity_f05(ex[ex.p >= thr], truth, hold.entity_id)
            res[(name, thr)] = f.mean()
        best = max((k for k in res if k[0] == name), key=res.get)
        print(f"{name:8s} best thr {best[1]}: holdout {res[best]:.5f}", flush=True)
    if a.write:
        thr = max((k for k in res if k[0] == "MEAN"), key=res.get)[1]
        te = None
        for t in tags:
            d = pd.read_parquet(f"{DATA}/pred/test_{t}.parquet")
            d[f"p_{t}"] = isos[t].predict(d.p)
            d = d.drop(columns="p")
            te = d if te is None else te.merge(d, on=KEY, how="outer")
        te = te.fillna({f"p_{t}": 0.0 for t in tags})
        ex = soft_excl(te[KEY].assign(p=te[[f"p_{t}" for t in tags]].mean(axis=1).values))
        s1t = pd.read_parquet(f"{DATA}/parquet/test_s1.parquet", columns=["entity_id"])
        out = os.path.join(ROOT, a.out)
        os.makedirs(out, exist_ok=True)
        write(ex[ex.p >= thr], s1t.entity_id, f"{out}/matching_results.tsv", "matched_entity_ids")
        write(te[KEY], s1t.entity_id, f"{out}/candidate_pairs.tsv", "candidate_entity_ids")
        print(f"wrote {out} (thr {thr})", flush=True)


if __name__ == "__main__":
    main()
