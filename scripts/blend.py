"""Blend our model with the teammate's model (probabilities only; their pipeline is not run here).

Inputs:  data/pred/{train,test}_{tag}.parquet  (ours: s1_id, cand_id, fold, y, p)
         data/friend/train_oof_probs.parquet, data/friend/test_probs.parquet (theirs: s1_id, cand_id, p; OOF on train)
Both are out-of-sample on folds 5-9 (ours trains on 2-4). Missing pair on one side -> p = 0 there.
Each model is isotonic-calibrated on folds 6-9, blended p = w*ours + (1-w)*theirs, then decoded with soft
exclusivity + threshold. (w, threshold) are chosen on folds 6-9; the score is reported on fold 5.

  python scripts/blend.py --tag b4 [--write]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from decode import soft_excl, bootstrap_gain, write  # noqa: E402
from metric import per_entity_f05  # noqa: E402

DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
TUNE, HOLD = (6, 7, 8, 9), 5
WS = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0)
THRS = np.round(np.arange(0.40, 0.91, 0.05), 2)


def s1_fold(ids: pd.Series) -> np.ndarray:
    return (ids.str[3:].astype(np.int64) % 10).values


def iso(p_fit, y_fit):
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_fit, y_fit)


def score(df, thr, truth, ids, excluded=False):
    ex = df if excluded else soft_excl(df[["s1_id", "cand_id", "p"]])
    return per_entity_f05(ex[ex.p >= thr], truth, ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--write", action="store_true", help="write the test TSVs")
    ap.add_argument("--out", default="output_blend", help="output dir (relative to repo root)")
    ap.add_argument("--friend_dir", default="friend", help="teammate probabilities under data/")
    a = ap.parse_args()
    truth = pd.read_parquet(f"{DATA}/parquet/train_pairs.parquet")
    s1 = pd.read_parquet(f"{DATA}/parquet/train_s1.parquet", columns=["entity_id", "country"])
    s1["fold"] = s1_fold(s1.entity_id)

    ours = pd.read_parquet(f"{DATA}/pred/train_{a.tag}.parquet")
    ours = ours[ours.fold.isin(TUNE + (HOLD,))][["s1_id", "cand_id", "p"]].rename(columns={"p": "p_o"})
    theirs = pd.read_parquet(f"{DATA}/{a.friend_dir}/train_oof_probs.parquet", columns=["s1_id", "cand_id", "p"])
    theirs = theirs[np.isin(s1_fold(theirs.s1_id), TUNE + (HOLD,))].rename(columns={"p": "p_t"})
    df = ours.merge(theirs, on=["s1_id", "cand_id"], how="outer")
    df["p_o"], df["p_t"] = df.p_o.fillna(0.0), df.p_t.fillna(0.0)
    df["fold"] = s1_fold(df.s1_id)
    df = df.merge(truth.assign(y=1), on=["s1_id", "cand_id"], how="left")
    df["y"] = df.y.fillna(0).astype(np.int8)
    print(f"pairs folds 5-9: {len(df):,} (ours {len(ours):,}, theirs {len(theirs):,}); "
          f"true pairs covered: ours {df[df.y == 1].p_o.gt(0).mean():.4f}, theirs {df[df.y == 1].p_t.gt(0).mean():.4f}, "
          f"union {df[df.y == 1][['p_o', 'p_t']].max(axis=1).gt(0).mean():.4f}", flush=True)

    cal = df[df.fold.isin(TUNE)]
    iso_o, iso_t = iso(cal.p_o, cal.y), iso(cal.p_t, cal.y)
    df["c_o"], df["c_t"] = iso_o.predict(df.p_o), iso_t.predict(df.p_t)

    tune_ids = s1.entity_id[s1.fold.isin(TUNE)]
    hold_ids = s1.entity_id[s1.fold == HOLD]
    tu, ho = df[df.fold.isin(TUNE)], df[df.fold == HOLD]
    res = {}
    for w in WS:                                         # soft exclusivity once per w, thresholds reuse it
        ex = soft_excl(tu.assign(p=w * tu.c_o + (1 - w) * tu.c_t)[["s1_id", "cand_id", "p"]])
        ex = ex[ex.p >= THRS.min()]
        for t in THRS:
            res[(w, t)] = score(ex, t, truth, tune_ids, excluded=True).mean()
        print(f"  w={w}: best {max(res[(w, t)] for t in THRS):.5f}", flush=True)
    res = pd.Series(res)
    best = res.idxmax()
    best_o = res.loc[1.0].idxmax()
    best_t = res.loc[0.0].idxmax()
    print(f"tuning folds 6-9: best blend w={best[0]} thr={best[1]} ({res[best]:.5f}); "
          f"ours alone thr={best_o} ({res[(1.0, best_o)]:.5f}); theirs alone thr={best_t} ({res[(0.0, best_t)]:.5f})")
    print("  by w (best thr):", {w: round(res.loc[w].max(), 5) for w in WS})

    f_b = score(ho.assign(p=best[0] * ho.c_o + (1 - best[0]) * ho.c_t), best[1], truth, hold_ids)
    f_o = score(ho.assign(p=ho.c_o), best_o, truth, hold_ids)
    f_t = score(ho.assign(p=ho.c_t), best_t, truth, hold_ids)
    ctry = s1.set_index("entity_id").country.reindex(hold_ids).values
    for name, f in (("ours", f_o), ("theirs", f_t), ("BLEND", f_b)):
        print(f"HOLDOUT fold 5 {name:7s}: {f.mean():.5f}  by country {f.groupby(ctry).mean().round(4).to_dict()}")
    for name, f in (("ours", f_o), ("theirs", f_t)):
        g, lo, hi = bootstrap_gain(f_b, f)
        print(f"  blend - {name}: {g:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]")

    if a.write:
        te_o = pd.read_parquet(f"{DATA}/pred/test_{a.tag}.parquet").rename(columns={"p": "p_o"})
        te_t = pd.read_parquet(f"{DATA}/{a.friend_dir}/test_probs.parquet").rename(columns={"p": "p_t"})
        te = te_o.merge(te_t, on=["s1_id", "cand_id"], how="outer")
        te["p_o"], te["p_t"] = te.p_o.fillna(0.0), te.p_t.fillna(0.0)
        te["p"] = best[0] * iso_o.predict(te.p_o) + (1 - best[0]) * iso_t.predict(te.p_t)
        ex = soft_excl(te[["s1_id", "cand_id", "p"]])
        pred = ex[ex.p >= best[1]]
        s1t = pd.read_parquet(f"{DATA}/parquet/test_s1.parquet", columns=["entity_id", "country"])
        out = os.path.join(ROOT, a.out)
        os.makedirs(out, exist_ok=True)
        write(pred, s1t.entity_id, f"{out}/matching_results.tsv", "matched_entity_ids")
        write(te[["s1_id", "cand_id"]], s1t.entity_id, f"{out}/candidate_pairs.tsv", "candidate_entity_ids")
        per = pred.groupby("s1_id").size().reindex(s1t.entity_id, fill_value=0)
        print("[test] predicted matches/S1:", per.groupby(s1t.country.values).mean().round(3).to_dict(),
              " empty share:", (per == 0).groupby(s1t.country.values).mean().round(3).to_dict())


if __name__ == "__main__":
    main()
