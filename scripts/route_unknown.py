"""Unknown-word routing (France fix, country-agnostic): a pair whose changed name words are mostly unknown to
the label-based word log-odds tables gets the probability of the model trained WITHOUT those features;
every other pair keeps the full model's. Both models are isotonic-calibrated on folds 6-9 first.

  coverage = (n_added - n_added_unknown + n_dropped - n_dropped_unknown) / (n_added + n_dropped)
  routed   = at least one changed name word and coverage < --min_cov

Writes data/pred/{train,test}_{out}.parquet (the schema src/train.py writes, p already calibrated), reports
the routed share per country (test: France should be far higher than India/US) and the holdout cost.

  python scripts/route_unknown.py --full b5 --nolo b5nolo --feat_dir data/feat_b5 --out b5r
  python scripts/blend.py --tag b5r --write                 # then blend with the teammate as usual
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from holdout import DATA, HOLD, KEY, TUNE, Holdout, bootstrap_gain, iso_fit  # noqa: E402

COV = ["n_added", "n_dropped", "n_added_unknown", "n_dropped_unknown"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="b5")
    ap.add_argument("--nolo", default="b5nolo")
    ap.add_argument("--feat_dir", default=f"{DATA}/feat_b5")
    ap.add_argument("--out", default="b5r")
    ap.add_argument("--min_cov", type=float, default=0.5)
    a = ap.parse_args()

    tr_f = pd.read_parquet(f"{DATA}/pred/train_{a.full}.parquet")
    tr_n = pd.read_parquet(f"{DATA}/pred/train_{a.nolo}.parquet", columns=KEY + ["p"])
    cal = tr_f.fold.isin(TUNE).values
    iso_f = iso_fit(tr_f.p.values[cal], tr_f.y.values[cal])
    tr_n = tr_f[KEY + ["fold", "y"]].merge(tr_n, on=KEY, how="left")
    assert tr_n.p.notna().all(), "the two models must score the same pairs (same feature dir)"
    iso_n = iso_fit(tr_n.p.values[cal], tr_n.y.values[cal])

    s1c = {}
    for split in ("train", "test"):
        s1c[split] = pd.read_parquet(f"{DATA}/parquet/{split}_s1.parquet", columns=["entity_id", "country"]) \
                       .set_index("entity_id").country
    out = {}
    for split in ("train", "test"):
        if split == "train":
            full, nolo = tr_f, tr_n
        else:
            full = pd.read_parquet(f"{DATA}/pred/test_{a.full}.parquet")
            nolo = full[KEY].merge(pd.read_parquet(f"{DATA}/pred/test_{a.nolo}.parquet", columns=KEY + ["p"]),
                                   on=KEY, how="left")
            assert nolo.p.notna().all(), "test pairs differ between the two models"
        cov = full[KEY].merge(pl.scan_parquet(f"{a.feat_dir}/{split}.parquet/*.parquet").select(KEY + COV)
                                .collect().to_pandas(), on=KEY, how="left")
        n = (cov.n_added + cov.n_dropped).fillna(0).values
        known = (cov.n_added - cov.n_added_unknown + cov.n_dropped - cov.n_dropped_unknown).fillna(0).values
        route = (n > 0) & (known < a.min_cov * n)
        d = full.copy()
        d["p_full"] = iso_f.predict(full.p.values).astype(np.float32)
        d["p_nolo"] = iso_n.predict(nolo.p.values).astype(np.float32)
        d["p"] = np.where(route, d.p_nolo, d.p_full).astype(np.float32)
        d["routed"] = route
        ctry = s1c[split].reindex(d.s1_id.values).values
        share = pd.Series(route).groupby(ctry).mean().round(3).to_dict()
        chg = pd.Series(route[n > 0]).groupby(ctry[n > 0]).mean().round(3).to_dict()
        print(f"[{split}] routed share of pairs {share}; of pairs with a changed name word {chg}", flush=True)
        out[split] = d
        cols = KEY + (["fold", "y", "p"] if split == "train" else ["p"])
        d[cols].to_parquet(f"{DATA}/pred/{split}_{a.out}.parquet", index=False)

    tr = out["train"]
    tr = tr[tr.fold >= HOLD]
    H = Holdout(tr.s1_id)
    t_f, f_f = H.evaluate(tr.assign(p=tr.p_full))
    t_r, f_r = H.evaluate(tr)
    g, lo, hi = bootstrap_gain(f_r, f_f)
    print(f"holdout {a.full} (thr {t_f}): {H.by_country(f_f)}")
    print(f"holdout {a.out} routed (thr {t_r}): {H.by_country(f_r)}")
    print(f"  routed - full {g:+.5f} [95% CI {lo:+.5f}, {hi:+.5f}]")
    print(f"GATE 3 (routing costs <= 0.0005 on India/US): {'PASS' if g >= -0.0005 else 'FAIL'}")
    print(f"wrote data/pred/{{train,test}}_{a.out}.parquet (calibrated p)")


if __name__ == "__main__":
    main()
