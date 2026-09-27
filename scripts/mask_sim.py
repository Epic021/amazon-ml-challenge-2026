"""France simulation on the holdout: how much does a model lose when every changed word is unknown to the
label-based word log-odds tables, as nearly all French words are? (gates 1-2 of the France routing plan)

Predicts folds 5-9 with the model's word evidence masked exactly as src/train.py's evidence_dropout does
(log-odds -> 0, unknown counts -> all changed words; address words too), then decodes with the model's
UNMASKED calibration and threshold, as France is decoded in production. Fold-5 F0.5 by country.
With --other, also scores a second model (e.g. the one trained without the log-odds) the normal way.

  python scripts/mask_sim.py --tag b5 --other b5nolo --feat_dir data/feat_b5
"""
import argparse
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from holdout import DATA, HOLD, KEY, TUNE, Holdout, bootstrap_gain, iso_fit  # noqa: E402
from train import evidence_dropout  # noqa: E402

ADDR_UNKNOWN = {"aadd_unknown": "addr_c_only", "adrop_unknown": "addr_s1_only"}   # unknown -> all changed


def load_pred(tag: str) -> pd.DataFrame:
    d = pd.read_parquet(f"{DATA}/pred/train_{tag}.parquet")
    return d[d.fold >= HOLD].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b5")
    ap.add_argument("--other", default="", help="second model scored unmasked (e.g. b5nolo)")
    ap.add_argument("--feat_dir", default=f"{DATA}/feat_b5")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()

    base = load_pred(a.tag)
    H = Holdout(base.s1_id)
    iso = iso_fit(base.p[base.fold.isin(TUNE)].values, base.y[base.fold.isin(TUNE)].values)
    base["p"] = iso.predict(base.p.values)
    thr, f_un = H.evaluate(base)
    print(f"[{a.tag}] unmasked  thr {thr}: {H.by_country(f_un)}", flush=True)

    model = lgb.Booster(model_file=f"{DATA}/models/lgb_{a.tag}.txt")
    cols = model.feature_name()
    need = sorted(set(cols) | {c for c in ADDR_UNKNOWN.values()} | {"n_added", "n_dropped"})
    X = (pl.scan_parquet(f"{a.feat_dir}/train.parquet/*.parquet")
           .filter(pl.col("fold") >= HOLD).select(KEY + need).collect().to_pandas())
    X = base[KEY].merge(X, on=KEY, how="left")
    Xm = evidence_dropout(X[cols].copy(), 1.0)
    for unk, total in ADDR_UNKNOWN.items():
        if unk in Xm.columns:
            Xm[unk] = X[total].values
    changed = (X.n_added.fillna(0) + X.n_dropped.fillna(0)) > 0
    print(f"pairs {len(X):,}; with a changed name word {changed.mean():.3f}", flush=True)
    masked = base[KEY + ["fold", "y"]].copy()
    masked["p"] = iso.predict(model.predict(Xm, num_threads=a.threads))
    _, f_m = H.evaluate(masked, thr)
    g, lo, hi = bootstrap_gain(f_m, f_un)
    print(f"[{a.tag}] MASKED    thr {thr}: {H.by_country(f_m)}   vs unmasked {g:+.5f} [{lo:+.5f}, {hi:+.5f}]")
    gate1 = -g >= 0.005
    print(f"GATE 1 (masking costs >= 0.005): {'PASS' if gate1 else 'FAIL'}", flush=True)

    if a.other:
        o = load_pred(a.other)
        io = iso_fit(o.p[o.fold.isin(TUNE)].values, o.y[o.fold.isin(TUNE)].values)
        o["p"] = io.predict(o.p.values)
        t_o, f_o = H.evaluate(o)
        g_m, lo_m, hi_m = bootstrap_gain(f_o, f_m)
        g_u, lo_u, hi_u = bootstrap_gain(f_o, f_un)
        print(f"[{a.other}] thr {t_o}: {H.by_country(f_o)}")
        print(f"  vs {a.tag} masked   {g_m:+.5f} [{lo_m:+.5f}, {hi_m:+.5f}]")
        print(f"  vs {a.tag} unmasked {g_u:+.5f} [{lo_u:+.5f}, {hi_u:+.5f}]")
        gate2 = g_m >= 0.005 and g_u >= -0.002
        print(f"GATE 2 ({a.other} >= masked + 0.005 and >= unmasked - 0.002): {'PASS' if gate2 else 'FAIL'}")


if __name__ == "__main__":
    main()
