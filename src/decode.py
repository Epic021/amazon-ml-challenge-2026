"""Turn pair probabilities into per-S1 match sets, tuned for macro F0.5 on the holdout.

1. Calibration: isotonic regression fitted on folds 6-9 (out-of-sample, disjoint from the holdout).
2. Exclusivity (each S2/S3 record belongs to <= 1 S1):
     hard  keep a record only for its highest-p S1
     soft  p' = o / (1 + sum_k o_k), o = p / (1 - p) over all S1 competing for the record
3. Set choice per S1:
     thr   plain global threshold
     dp    exact expected-F0.5 top-k (src/efdp.py) with M ~ Poisson(lam) missed true matches,
           lam = c * (1 - R) / R * sum(p)   (R = holdout pair recall of the candidate set)
All combinations are scored on the holdout (fold 5); the best one is applied to test and a
bootstrap 95% CI of its gain over the best plain threshold is printed.

  python src/decode.py --tag b1
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import per_entity_f05  # noqa: E402
from efdp import best_k  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))   # override for smoke runs
PRED = os.path.join(DATA, "pred")
PQ = os.path.join(DATA, "parquet")
OUT = os.environ.get("BER_OUT", os.path.join(ROOT, "output"))
HOLD_FOLD = 5
CALIB_FOLDS = (6, 7, 8, 9)       # out-of-sample for the model, disjoint from holdout
N_MAX, P_MIN = 24, 0.01


def hard_excl(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["cand_id", "p"], ascending=[True, False])
    return df[~df.cand_id.duplicated()]


def soft_excl(df: pd.DataFrame) -> pd.DataFrame:
    q = df.p.clip(upper=0.999)
    o = q / (1 - q)
    return df.assign(p=(o / (1 + o.groupby(df.cand_id).transform("sum"))).astype(np.float32))


def threshold(df: pd.DataFrame, t: float) -> pd.DataFrame:
    return df[df.p >= t][["s1_id", "cand_id", "p"]]


def dp_decode(df: pd.DataFrame, c: float, recall: float) -> pd.DataFrame:
    df = df.sort_values(["s1_id", "p"], ascending=[True, False])
    df = df.assign(r=df.groupby("s1_id", sort=False).cumcount(),
                   tot=df.groupby("s1_id", sort=False).p.transform("sum"))
    keep = (df.p >= P_MIN) & (df.r < N_MAX)
    small = df[~keep].groupby("s1_id", sort=False).p.sum()             # mass we don't decode -> lam
    d = df[keep]
    if d.empty:
        return d[["s1_id", "cand_id", "p"]]
    ids = pd.unique(d.s1_id)
    g = pd.Series(np.arange(len(ids)), index=ids).reindex(d.s1_id.values).values
    P = np.zeros((len(ids), N_MAX), dtype=np.float64)
    P[g, d.r.values] = d.p.values
    lam = (c * (1 - recall) / recall * d.groupby("s1_id", sort=False).tot.first().reindex(ids).values
           + small.reindex(ids).fillna(0).values)
    ks = np.concatenate([best_k(P[i:i + 50_000], lam[i:i + 50_000]) for i in range(0, len(ids), 50_000)])
    kk = pd.Series(ks, index=ids)
    return d[d.r.values < kk.reindex(d.s1_id.values).values][["s1_id", "cand_id", "p"]]


def write(pred: pd.DataFrame, s1_ids, path: str, col: str):
    lists = pred.groupby("s1_id").cand_id.agg(",".join)
    out = pd.DataFrame({"source1_entity_id": np.asarray(s1_ids)})
    out[col] = out.source1_entity_id.map(lists).fillna("")
    out.to_csv(path, sep="\t", index=False, encoding="utf-8")


def bootstrap_gain(f_a: pd.Series, f_b: pd.Series, n: int = 1000) -> tuple:
    d = (f_a - f_b.reindex(f_a.index)).values
    rng = np.random.default_rng(0)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--calib_folds", default="6,7,8,9", help="folds for isotonic calibration (never trained on)")
    ap.add_argument("--dp", action="store_true", help="also evaluate the exact expected-F DP (slow; never beat the threshold)")
    a = ap.parse_args()
    calib_folds = tuple(int(x) for x in a.calib_folds.split(","))

    tr = pd.read_parquet(f"{PRED}/train_{a.tag}.parquet")
    cal = tr[tr.fold.isin(calib_folds)]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(cal.p.values, cal.y.values)
    tr["p"] = iso.predict(tr.p.values).astype(np.float32)
    truth = pd.read_parquet(f"{PQ}/train_pairs.parquet")
    s1 = pd.read_parquet(f"{PQ}/train_s1.parquet", columns=["entity_id", "country"])
    hold_ids = s1.entity_id[(s1.entity_id.str[3:].astype(np.int64) % 10) == HOLD_FOLD]
    hold_set = set(hold_ids)
    t_ho = truth[truth.s1_id.isin(hold_set)]
    ho_all = tr[tr.s1_id.isin(hold_set)]
    recall = max(ho_all.y.sum() / max(len(t_ho), 1), 1e-3)
    print(f"holdout S1 {len(hold_ids):,}; pair recall of candidates R={recall:.4f}; oracle F0.5="
          f"{per_entity_f05(ho_all[ho_all.y == 1], truth, hold_ids).mean():.4f}", flush=True)

    views = {"hard": hard_excl(tr), "soft": soft_excl(tr)}
    views = {k: v[v.s1_id.isin(hold_set)] for k, v in views.items()}
    rules = {}
    for ex in views:
        for t in np.round(np.arange(0.3, 0.96, 0.05), 2):
            rules[(ex, "thr", float(t))] = lambda d, t=t: threshold(d, t)
        for c in ((0.0, 0.5, 1.0, 2.0) if a.dp else ()):
            rules[(ex, "dp", c)] = lambda d, c=c: dp_decode(d, c, recall)
    scores, per = {}, {}
    for key, fn in rules.items():
        f = per_entity_f05(fn(views[key[0]]), truth, hold_ids)
        scores[key], per[key] = f.mean(), f
    res = pd.Series(scores).sort_values(ascending=False)
    print(res.head(12).round(5).to_string(), flush=True)
    best = res.index[0]
    best_thr = max((k for k in scores if k[1] == "thr"), key=scores.get)
    g, lo, hi = bootstrap_gain(per[best], per[best_thr])
    print(f"BEST {best}: holdout macro F0.5 = {scores[best]:.5f}; vs best plain threshold {best_thr} "
          f"({scores[best_thr]:.5f}): gain {g:+.5f} [95% CI {lo:+.5f}, {hi:+.5f}]")
    f = per[best]
    ctry = s1.set_index("entity_id").country.reindex(f.index).values
    n_true = t_ho.groupby("s1_id").size().reindex(f.index, fill_value=0).values
    print("  by country:", f.groupby(ctry).mean().round(4).to_dict(),
          " singletons:", round(f[n_true == 0].mean(), 4), " non-singletons:", round(f[n_true > 0].mean(), 4))

    path = f"{PRED}/test_{a.tag}.parquet"
    if os.path.isfile(path):
        te = pd.read_parquet(path)
        te["p"] = iso.predict(te.p.values).astype(np.float32)
        s1t = pd.read_parquet(f"{PQ}/test_s1.parquet", columns=["entity_id", "country"])
        pred = rules[best]((hard_excl if best[0] == "hard" else soft_excl)(te))
        os.makedirs(OUT, exist_ok=True)
        write(pred, s1t.entity_id, f"{OUT}/matching_results.tsv", "matched_entity_ids")
        # candidate_pairs.tsv = exactly the pairs the model scored (post-pruning), per README
        write(te[["s1_id", "cand_id"]], s1t.entity_id, f"{OUT}/candidate_pairs.tsv", "candidate_entity_ids")
        per_s1 = pred.groupby("s1_id").size().reindex(s1t.entity_id, fill_value=0)
        sum_p = te.groupby("s1_id").p.sum().reindex(s1t.entity_id, fill_value=0)
        print("[test] predicted matches/S1:", per_s1.groupby(s1t.country.values).mean().round(3).to_dict(),
              " empty share:", (per_s1 == 0).groupby(s1t.country.values).mean().round(3).to_dict())
        print("[label-shift check] mean sum p per S1  test:",
              sum_p.groupby(s1t.country.values).mean().round(3).to_dict(), " holdout:",
              round(ho_all.groupby("s1_id").p.sum().reindex(hold_ids, fill_value=0).mean(), 3))


if __name__ == "__main__":
    main()
