"""Learn token equivalences (st <-> street, tx <-> texas, praivet <-> private, r <-> rue ...) from data
instead of hand-written maps.

For pairs believed to be the same business, look at the tokens that differ between the two sides.
When a variant token y (candidate side) is repeatedly "explained" by the same S1-side token or
2-token phrase x, emit y -> x (S1 is the clean reference, so its spelling is canonical).

  labeled  (train): pairs = ground-truth matches
  pseudo   (test) : pairs = mutual-best candidates with identical non-empty number sets and a high
                    blocking score. Unlabeled, recomputed on whatever test data is given, so it also
                    covers France or any unseen country.

  python src/mine_equiv.py --split train
  python src/mine_equiv.py --split test
Writes data/equiv/{split}.parquet (field, variant, canonical, n, share).
"""
import argparse
import os
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NORM = os.path.join(ROOT, "data", "norm")
CAND = os.path.join(ROOT, "data", "cand")
PQ = os.path.join(ROOT, "data", "parquet")
EQ = os.path.join(ROOT, "data", "equiv")


def count_subs(args):
    s1_strs, c_strs = args
    pair, var = Counter(), Counter()
    for a, b in zip(s1_strs, c_strs):
        A = [t for t in a.split() if not t.isdigit()]
        B = [t for t in b.split() if not t.isdigit()]
        sa, sb = set(A), set(B)
        ua, ub = [t for t in A if t not in sb], [t for t in B if t not in sa]
        if not ub or len(ua) > 3 or len(ub) > 3:
            continue
        var.update(set(ub))
        xs = set(ua)
        xs |= {f"{A[i]} {A[i + 1]}" for i in range(len(A) - 1) if A[i] in ua and A[i + 1] in ua}
        for y in set(ub):
            for x in xs:
                pair[(x, y)] += 1
    return pair, var


def mine(s1_strs, c_strs, procs, min_n, min_share):
    n = 200_000
    chunks = [(s1_strs[i:i + n], c_strs[i:i + n]) for i in range(0, len(s1_strs), n)]
    pair, var = Counter(), Counter()
    with Pool(procs) as p:
        for pc, vc in p.imap_unordered(count_subs, chunks):
            pair.update(pc)
            var.update(vc)
    best = {}
    for (x, y), k in pair.items():
        if k >= min_n and k / var[y] >= min_share and (y not in best or k > best[y][1]):
            best[y] = (x, k)
    rows = [(y, x, k, k / var[y]) for y, (x, k) in best.items()]
    return pd.DataFrame(rows, columns=["variant", "canonical", "n", "share"]).sort_values("n", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--max_pairs", type=int, default=3_000_000)
    ap.add_argument("--min_n", type=int, default=50)
    ap.add_argument("--min_share", type=float, default=0.3)
    a = ap.parse_args()
    os.makedirs(EQ, exist_ok=True)
    procs = os.cpu_count() or 4
    t0 = time.time()

    rec = pd.concat([pd.read_parquet(f"{NORM}/{a.split}_s{k}.parquet",
                                     columns=["id", "country", "name_norm", "addr_norm", "nums"])
                     for k in (1, 2, 3)], ignore_index=True).set_index("id")
    if a.split == "train":
        pairs = pd.read_parquet(f"{PQ}/train_pairs.parquet")
    else:
        c = pd.read_parquet(f"{CAND}/test.parquet")
        c = c[(c.rank_q == 1) & (c.rank_s == 1) & (c.score >= 0.5)]
        pairs = c[["s1_id", "cand_id"]]
    if len(pairs) > a.max_pairs:
        pairs = pairs.sample(a.max_pairs, random_state=0)
    s1r, cr = rec.reindex(pairs.s1_id.values), rec.reindex(pairs.cand_id.values)
    if a.split == "test":                           # pseudo-labels: identical, non-empty number sets
        keep = (s1r.nums.values == cr.nums.values) & (s1r.nums.values != "")
        s1r, cr = s1r[keep], cr[keep]
    print(f"[{a.split}] mining from {len(s1r):,} pairs", flush=True)

    out = []
    for field in ("addr_norm", "name_norm"):
        for country in sorted(pd.unique(s1r.country)):
            m = (s1r.country.values == country)
            df = mine(s1r[field].values[m], cr[field].values[m], procs, a.min_n, a.min_share)
            df.insert(0, "field", field)
            df.insert(1, "country", country)
            out.append(df)
            print(f"  {field:9s} {country:7s}: {len(df):4d} rules  e.g. "
                  f"{list(zip(df.variant.head(12), df.canonical.head(12)))}", flush=True)
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(f"{EQ}/{a.split}.parquet", index=False)
    print(f"done in {time.time() - t0:.0f}s -> {EQ}/{a.split}.parquet ({len(res)} rules)")


if __name__ == "__main__":
    main()
