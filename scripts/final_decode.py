"""Final decode from the stackers' probabilities (v3 b5sb3, v4 b5sb4): pick the best of each alone and
their average on the clean holdout, then write the submission.

Stacker folds 6-8 are in-sample, so calibration AND the threshold use fold 9 only; fold 5 is the holdout.
Pipeline per candidate: isotonic (fold 9) -> soft one-owner p' = o / (1 + sum o) -> threshold.
candidate_pairs.tsv = every test pair the stackers scored, so matches are a subset by construction.

  python scripts/final_decode.py --tags b5sb3,b5sb4 --out output_final [--france_thr 0.85]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from decode import bootstrap_gain, soft_excl, write  # noqa: E402
from metric import per_entity_f05, s1_universe  # noqa: E402

DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
KEY = ["s1_id", "cand_id"]
CAL, HOLD = 9, 5
THRS = np.round(np.arange(0.40, 0.951, 0.025), 3)


def select(ex: pd.DataFrame, thr, t2: float) -> pd.DataFrame:
    """Pairs with p >= thr (thr: scalar or per-row array); S1s left empty get their best pair if p >= t2."""
    t = np.broadcast_to(np.asarray(thr, dtype=np.float64), (len(ex),))
    sel = ex[ex.p.values >= t]
    if t2 >= 1.0:
        return sel
    rest = ex[~ex.s1_id.isin(set(sel.s1_id))]
    top = rest.sort_values("p", ascending=False).drop_duplicates("s1_id")
    return pd.concat([sel, top[top.p >= t2]])


def fold(ids: pd.Series) -> np.ndarray:
    return (ids.str[3:].astype(np.int64) % 10).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="b5sb3,b5sb4")
    ap.add_argument("--out", default="output_final")
    ap.add_argument("--france_thr", type=float, default=0.0, help="France-only threshold (0 = same as the others)")
    ap.add_argument("--no_fallback", action="store_true", help="disable the empty-S1 fallback")
    ap.add_argument("--france_t2", type=float, default=0.0, help="France-only fallback bar (e.g. t2 + France shift)")
    ap.add_argument("--force", default="", help="use this candidate (a tag or 'avg') instead of the best")
    a = ap.parse_args()
    tags = [t for t in a.tags.split(",") if os.path.isfile(f"{DATA}/pred/train_{t}.parquet")]
    print("stackers found:", tags, flush=True)
    truth = pd.read_parquet(f"{DATA}/parquet/train_pairs.parquet")
    s1 = pd.read_parquet(f"{DATA}/parquet/train_s1.parquet", columns=["entity_id", "country"])
    s1["fold"] = fold(s1.entity_id)
    cal_ids, hold_ids = s1.entity_id[s1.fold == CAL], s1.entity_id[s1.fold == HOLD]
    ctry = s1.set_index("entity_id").country

    tr, te, isos = None, None, {}
    for t in tags:
        d = pd.read_parquet(f"{DATA}/pred/train_{t}.parquet", columns=KEY + ["p"])
        d = d[np.isin(fold(d.s1_id), (CAL, HOLD))].rename(columns={"p": t})
        tr = d if tr is None else tr.merge(d, on=KEY, how="outer")
        e = pd.read_parquet(f"{DATA}/pred/test_{t}.parquet", columns=KEY + ["p"]).rename(columns={"p": t})
        te = e if te is None else te.merge(e, on=KEY, how="outer")
    tr, te = tr.fillna(0.0), te.fillna(0.0)
    cal_ids, hold_ids = s1_universe(cal_ids, tr.s1_id), s1_universe(hold_ids, tr.s1_id)   # covered S1s only
    tr = tr.merge(truth.assign(y=1), on=KEY, how="left").fillna({"y": 0})
    tr["fold"] = fold(tr.s1_id)
    c = tr.fold.values == CAL
    for t in tags:
        isos[t] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(tr[t].values[c], tr.y.values[c])
        tr[t], te[t] = isos[t].predict(tr[t].values), isos[t].predict(te[t].values)
    cands = {t: [t] for t in tags}
    if len(tags) > 1:
        cands["avg"] = tags

    res = {}
    for name, members in cands.items():
        ex = soft_excl(tr[KEY].assign(p=tr[members].mean(axis=1).values))
        ex = ex[ex.p >= 0.05]
        cal_ex = ex[ex.s1_id.isin(set(cal_ids))]
        thr = max(THRS, key=lambda t: per_entity_f05(cal_ex[cal_ex.p >= t], truth, cal_ids).mean())
        f0 = per_entity_f05(ex[ex.p >= thr], truth, hold_ids)
        print(f"HOLDOUT fold 5 {name:6s} thr {thr}: {f0.mean():.5f} "
              f"{f0.groupby(ctry.reindex(f0.index).values).mean().round(4).to_dict()}", flush=True)
        # empty-S1 fallback: tuned on fold 9, scored on fold 5
        t2s = [1.0] + [round(x, 3) for x in np.arange(0.10, thr, 0.05)]
        sc = {t2: per_entity_f05(select(cal_ex, thr, t2), truth, cal_ids).mean() for t2 in t2s}
        print(f"  fallback on fold 9: {({k: round(v, 5) for k, v in sc.items()})}", flush=True)
        t2 = max(sc, key=sc.get) if not a.no_fallback else 1.0
        f = per_entity_f05(select(ex, thr, t2), truth, hold_ids)
        g, lo, hi = bootstrap_gain(f, f0)
        print(f"HOLDOUT fold 5 {name:6s} thr {thr} + fallback {t2}: {f.mean():.5f} "
              f"{f.groupby(ctry.reindex(f.index).values).mean().round(4).to_dict()}  gain {g:+.5f} [{lo:+.5f}, {hi:+.5f}]",
              flush=True)
        res[name] = ((thr, t2), f)
    best = a.force or max(res, key=lambda k: res[k][1].mean())
    for k in res:
        if k != best:
            g, lo, hi = bootstrap_gain(res[best][1], res[k][1])
            print(f"  {best} - {k}: {g:+.5f} [95% CI {lo:+.5f}, {hi:+.5f}]")
    thr, t2 = res[best][0]
    print(f"CHOSEN: {best} (threshold {thr}, empty-S1 fallback {t2})", flush=True)

    s1t = pd.read_parquet(f"{DATA}/parquet/test_s1.parquet", columns=["entity_id", "country"])
    ex = soft_excl(te[KEY].assign(p=te[cands[best]].mean(axis=1).values))
    fr = ex.s1_id.map(s1t.set_index("entity_id").country).eq("France").values
    t_row = np.where(fr, a.france_thr if a.france_thr else thr, thr)
    pred = select(ex, t_row, t2)
    if a.france_t2:                                  # French fallback pairs must also clear the France bar
        pfr = pred.s1_id.map(s1t.set_index("entity_id").country).eq("France").values
        n0 = len(pred)
        pred = pred[~(pfr & (pred.p.values < a.france_thr) & (pred.p.values < a.france_t2))]
        print(f"France fallback bar {a.france_t2}: removed {n0 - len(pred):,} French fallback pairs")
    fb = (pred.p.values < pred.s1_id.map(s1t.set_index("entity_id").country).eq("France").map(
        {True: a.france_thr or thr, False: thr}).values).sum()
    print(f"[test] fallback pairs added (below the threshold): {fb:,}")
    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    write(pred, s1t.entity_id, f"{out}/matching_results.tsv", "matched_entity_ids")
    write(te[KEY], s1t.entity_id, f"{out}/candidate_pairs.tsv", "candidate_entity_ids")
    per = pred.groupby("s1_id").size().reindex(s1t.entity_id, fill_value=0)
    print(f"[test] France threshold {a.france_thr or thr}; predicted matches/S1:",
          per.groupby(s1t.country.values).mean().round(3).to_dict(), " empty share:",
          (per == 0).groupby(s1t.country.values).mean().round(3).to_dict())
    print(f"wrote {out}/matching_results.tsv and candidate_pairs.tsv")


if __name__ == "__main__":
    main()
