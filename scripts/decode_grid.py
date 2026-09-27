"""Grid of small decoding strategies on stacker probabilities, all scored on the holdout (fold 5).

Sources: v3, v4, mean(v3, v4) (each isotonic-calibrated on fold 9).
Soft exclusivity p' = o^a / (c + sum o^a) over each record's S1 candidates, with c in {0.5,1,2,4}
(weight of the "no owner" option) and odds temperature a in {0.8,1,1.25}; optional cap of k matches per S1;
thresholds 0.40-0.90 step 0.025. Prints the top configurations and the gain of the best over the
production rule (v3, c=1, a=1, no cap, best threshold) with a paired bootstrap CI.

  python scripts/decode_grid.py --workers 8 [--write_best --out output_grid]
"""
import argparse
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from decode import write, bootstrap_gain  # noqa: E402
from metric import per_entity_f05  # noqa: E402

DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
THRS = np.round(np.arange(0.40, 0.901, 0.025), 3)
G = {}


def load():
    s1 = pd.read_parquet(f"{DATA}/parquet/train_s1.parquet", columns=["entity_id"]).entity_id
    hold = s1[(s1.str[3:].astype(np.int64) % 10) == 5]
    df, isos = None, {}
    for t in ("b5sb3", "b5sb4"):
        d = pd.read_parquet(f"{DATA}/pred/train_{t}.parquet")
        c = d[d.fold == 9]
        isos[t] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(c.p.values, c.y.values)
        d[t] = isos[t].predict(d.p.values).astype(np.float32)
        d = d.drop(columns="p")
        df = d if df is None else df.merge(d[["s1_id", "cand_id", t]], on=["s1_id", "cand_id"], how="outer")
    df = df.fillna({"b5sb3": 0.0, "b5sb4": 0.0})
    df["mean"] = (df.b5sb3 + df.b5sb4) / 2
    hc = set(df.cand_id[df.s1_id.isin(set(hold))])
    df = df[df.cand_id.isin(hc)].reset_index(drop=True)          # rows that matter for holdout exclusivity
    return df, hold, isos


def excl(p, cand, c, a):
    q = np.clip(p, 1e-6, 0.999)
    o = (q / (1 - q)) ** a
    s = pd.Series(o).groupby(cand.values).transform("sum").values
    return o / (c + s)


def run(cfg):
    src, c, a, cap = cfg
    df, hold, truth = G["df"], G["hold"], G["truth"]
    px = excl(df[src].values, df.cand_id, c, a)
    ho = df[["s1_id", "cand_id"]].assign(p=px)
    ho = ho[ho.s1_id.isin(G["hold_set"]) & (ho.p >= THRS.min())]
    if cap:
        ho = ho.sort_values("p", ascending=False)
        ho = ho[ho.groupby("s1_id").cumcount() < cap]
    out = []
    for t in THRS:
        f = per_entity_f05(ho[ho.p >= t], truth, hold)
        out.append((src, c, a, cap, float(t), f.mean()))
    return out


def init(df, hold, truth):
    G.update(df=df, hold=hold, truth=truth, hold_set=set(hold))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--write_best", action="store_true")
    ap.add_argument("--out", default="output_grid")
    a = ap.parse_args()
    df, hold, isos = load()
    truth = pd.read_parquet(f"{DATA}/parquet/train_pairs.parquet")
    truth = truth[truth.s1_id.isin(set(hold))]
    cfgs = list(itertools.product(["b5sb3", "b5sb4", "mean"], [0.5, 1.0, 2.0, 4.0], [0.8, 1.0, 1.25], [0, 6]))
    print(f"{len(cfgs)} configs x {len(THRS)} thresholds = {len(cfgs) * len(THRS)} decoders on {len(df):,} rows", flush=True)
    with Pool(a.workers, initializer=init, initargs=(df, hold, truth)) as p:
        res = [r for part in p.imap_unordered(run, cfgs) for r in part]
    r = pd.DataFrame(res, columns=["src", "c", "a", "cap", "thr", "F"]).sort_values("F", ascending=False)
    base = r[(r.src == "b5sb3") & (r.c == 1.0) & (r.a == 1.0) & (r.cap == 0)].iloc[0]
    print(f"BASELINE (v3, c=1, a=1, no cap, thr {base.thr}): {base.F:.5f}", flush=True)
    print(r.head(15).to_string(index=False), flush=True)
    best = r.iloc[0]
    init(df, hold, truth)
    def per(row):
        px = excl(df[row.src].values, df.cand_id, row.c, row.a)
        ho = df[["s1_id", "cand_id"]].assign(p=px)
        ho = ho[ho.s1_id.isin(set(hold)) & (ho.p >= row.thr)]
        if row.cap:
            ho = ho.sort_values("p", ascending=False)
            ho = ho[ho.groupby("s1_id").cumcount() < row.cap]
        return per_entity_f05(ho, truth, hold)
    g, lo, hi = bootstrap_gain(per(best), per(base))
    print(f"BEST {dict(best)} vs BASELINE: gain {g:+.5f} [95% CI {lo:+.5f}, {hi:+.5f}]", flush=True)
    if a.write_best:
        te = None
        for t in ("b5sb3", "b5sb4"):
            d = pd.read_parquet(f"{DATA}/pred/test_{t}.parquet")
            d[t] = isos[t].predict(d.p.values).astype(np.float32)
            d = d.drop(columns="p")
            te = d if te is None else te.merge(d, on=["s1_id", "cand_id"], how="outer")
        te = te.fillna({"b5sb3": 0.0, "b5sb4": 0.0})
        te["mean"] = (te.b5sb3 + te.b5sb4) / 2
        te["p"] = excl(te[best.src].values, te.cand_id, best.c, best.a)
        pred = te[te.p >= best.thr][["s1_id", "cand_id", "p"]]
        if best.cap:
            pred = pred.sort_values("p", ascending=False)
            pred = pred[pred.groupby("s1_id").cumcount() < best.cap]
        s1t = pd.read_parquet(f"{DATA}/parquet/test_s1.parquet", columns=["entity_id"]).entity_id
        out = os.path.join(ROOT, a.out)
        os.makedirs(out, exist_ok=True)
        write(pred, s1t, f"{out}/matching_results.tsv", "matched_entity_ids")
        write(te[["s1_id", "cand_id"]], s1t, f"{out}/candidate_pairs.tsv", "candidate_entity_ids")
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
