"""Turn pair probabilities into per-S1 match sets, tuned for macro F0.5 on the holdout.

1. Exclusive assignment: each S2/S3 record is kept only for its highest-probability S1.
2. Per S1, candidates sorted by p; choose m (0..n) maximising the approximate expected F0.5
      E[F(m)] ~ 1.25*sum_{i<=m} p_i / (m + 0.25*(sum_all p + miss))      (m >= 1)
      E[F(0)] =  prod_i (1 - p_i) * exp(-miss)                              (predict "no match")
   with a floor: never include a candidate with p < pmin.
   `miss` = expected true matches outside the candidate set (blocking misses), tuned on holdout.
3. Compared against a plain global threshold; the better rule (on holdout) is used for test.

  python src/decode.py --tag v1
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import per_entity_f05  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, "data", "pred")
PQ = os.path.join(ROOT, "data", "parquet")
CAND = os.path.join(ROOT, "data", "cand")
OUT = os.path.join(ROOT, "output")
HOLD_FOLD = 5
CALIB_FOLDS = (6, 7, 8, 9)       # out-of-sample for the model, disjoint from holdout


def exclusive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["cand_id", "p"], ascending=[True, False])
    return df[~df.cand_id.duplicated()]


def expected_f(df: pd.DataFrame, miss: float, pmin: float) -> pd.DataFrame:
    df = df.sort_values(["s1_id", "p"], ascending=[True, False]).copy()
    g = df.groupby("s1_id", sort=False)
    df["m"] = g.cumcount() + 1
    df["cum"] = g.p.cumsum()
    tot = g.p.transform("sum")
    df["ef"] = 1.25 * df.cum / (df.m + 0.25 * (tot + miss))
    df.loc[df.p < pmin, "ef"] = -1.0
    best_ef = df.groupby("s1_id", sort=False).ef.max()
    best_m = df[df.ef.values == df.s1_id.map(best_ef).values].groupby("s1_id", sort=False).m.min()
    log1m = np.log1p(-df.p.clip(upper=0.999999))
    ef0 = np.exp(log1m.groupby(df.s1_id, sort=False).sum() - miss)       # P(no true match at all)
    keep_m = best_m.where(best_ef.reindex(best_m.index) > ef0.reindex(best_m.index), 0)
    return df[df.m.values <= df.s1_id.map(keep_m).values][["s1_id", "cand_id", "p"]]


def threshold(df: pd.DataFrame, t: float) -> pd.DataFrame:
    return df[df.p >= t][["s1_id", "cand_id", "p"]]


def evaluate(pred, truth, s1_ids):
    return per_entity_f05(pred, truth, s1_ids).mean()


def write(pred: pd.DataFrame, s1_ids, path: str, col: str):
    lists = pred.groupby("s1_id").cand_id.agg(",".join)
    out = pd.DataFrame({"source1_entity_id": np.asarray(s1_ids)})
    out[col] = out.source1_entity_id.map(lists).fillna("")
    out.to_csv(path, sep="\t", index=False, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()

    tr = pd.read_parquet(f"{PRED}/train_{a.tag}.parquet")
    cal = tr[tr.fold.isin(CALIB_FOLDS)]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(cal.p.values, cal.y.values)
    tr["p_raw"] = tr.p
    tr["p"] = iso.predict(tr.p.values).astype(np.float32)
    ho_raw = tr[tr.fold == HOLD_FOLD]
    print(f"calibration: holdout mean p raw={ho_raw.p_raw.mean():.4f} iso={ho_raw.p.mean():.4f} y={ho_raw.y.mean():.4f}")
    truth = pd.read_parquet(f"{PQ}/train_pairs.parquet")
    s1 = pd.read_parquet(f"{PQ}/train_s1.parquet", columns=["entity_id", "country"])
    hold_ids = s1.entity_id[(s1.entity_id.str[3:].astype(np.int64) % 10) == HOLD_FOLD]
    ex = exclusive(tr)
    ho = ex[ex.s1_id.isin(set(hold_ids))]
    ho_all = tr[tr.s1_id.isin(set(hold_ids))]

    oracle = ho_all[ho_all.y == 1]
    print(f"holdout S1: {len(hold_ids):,}; oracle (perfect matcher on candidates) F0.5 = "
          f"{evaluate(oracle, truth, hold_ids):.4f}")

    res = []
    for t in np.arange(0.2, 0.95, 0.05):
        res.append(("thr", round(t, 2), 0.0, evaluate(threshold(ho, t), truth, hold_ids)))
    t_no_excl = max(res, key=lambda r: r[3])[1]
    print(f"  threshold WITHOUT exclusivity (t={t_no_excl}): "
          f"{evaluate(threshold(ho_all, t_no_excl), truth, hold_ids):.4f}")
    for miss in (0.0, 0.1, 0.25, 0.5):
        for pmin in (0.05, 0.1, 0.2, 0.3, 0.4):
            res.append(("ef", miss, pmin, evaluate(expected_f(ho, miss, pmin), truth, hold_ids)))
    res = pd.DataFrame(res, columns=["rule", "a", "b", "f05"]).sort_values("f05", ascending=False)
    print(res.head(10).to_string(index=False))
    best = res.iloc[0]
    rule = (lambda d: threshold(d, best.a)) if best.rule == "thr" else (lambda d: expected_f(d, best.a, best.b))
    hp = rule(ho)
    f = per_entity_f05(hp, truth, hold_ids)
    ctry = s1.set_index("entity_id").country.reindex(f.index).values
    print(f"BEST {best.rule} a={best.a} b={best.b}: holdout macro F0.5 = {f.mean():.4f}")
    print("  by country:", f.groupby(ctry).mean().round(4).to_dict())
    n_true = truth[truth.s1_id.isin(set(hold_ids))].groupby("s1_id").size().reindex(f.index, fill_value=0)
    print("  singletons:", round(f[n_true.values == 0].mean(), 4), " non-singletons:", round(f[n_true.values > 0].mean(), 4))

    if os.path.isfile(f"{PRED}/test_{a.tag}.parquet"):
        te = pd.read_parquet(f"{PRED}/test_{a.tag}.parquet")
        te["p"] = iso.predict(te.p.values).astype(np.float32)
        s1t = pd.read_parquet(f"{PQ}/test_s1.parquet", columns=["entity_id", "country"])
        pred = rule(exclusive(te))
        cand = pd.read_parquet(f"{CAND}/test.parquet", columns=["s1_id", "cand_id"])
        os.makedirs(OUT, exist_ok=True)
        write(pred, s1t.entity_id, f"{OUT}/matching_results.tsv", "matched_entity_ids")
        write(cand, s1t.entity_id, f"{OUT}/candidate_pairs.tsv", "candidate_entity_ids")
        per = pred.groupby("s1_id").size().reindex(s1t.entity_id, fill_value=0)
        print("[test] predicted matches/S1:", per.groupby(s1t.country.values).mean().round(3).to_dict(),
              " empty share:", (per == 0).groupby(s1t.country.values).mean().round(3).to_dict())


if __name__ == "__main__":
    main()
