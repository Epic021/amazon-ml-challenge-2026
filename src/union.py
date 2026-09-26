"""Merge the candidate sets of all retrievers into data/cand/{split}.parquet.

Keeps every retriever's score and ranks as columns (score_word, rank_q_word, ...), which flow
through features.py as model features, plus the combined columns the pipeline uses:
  score  = max score over retrievers
  rank_q = best (min) rank_q over retrievers,  rank_s = best rank_s

  python src/union.py --split train      # also reports marginal recall + oracle F0.5 ceiling
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import per_entity_f05  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))   # override for smoke runs
CAND = os.path.join(DATA, "cand")
PQ = os.path.join(DATA, "parquet")
RETRIEVERS = ("word", "namechar", "addr", "numaddr", "theirs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--max_rank_q", type=int, default=10)
    ap.add_argument("--max_rank_s", type=int, default=15)
    ap.add_argument("--retrievers", default=",".join(RETRIEVERS), help="comma-separated subset to merge")
    ap.add_argument("--out", default=None, help="output path (default data/cand/{split}.parquet)")
    a = ap.parse_args()

    u, used = None, []
    for r in a.retrievers.split(","):
        path = f"{CAND}/{a.split}_{r}.parquet"
        if not os.path.isfile(path):
            continue
        d = pd.read_parquet(path, columns=["s1_id", "cand_id", "score", "rank_q", "rank_s"])
        d = d.rename(columns={"score": f"score_{r}", "rank_q": f"rank_q_{r}", "rank_s": f"rank_s_{r}"})
        u = d if u is None else u.merge(d, on=["s1_id", "cand_id"], how="outer")
        used.append(r)
        print(f"  + {r}: {len(d):,} pairs -> union {len(u):,}", flush=True)
    for r in used:
        u[f"score_{r}"] = u[f"score_{r}"].fillna(0).astype(np.float32)
        u[f"rank_q_{r}"] = u[f"rank_q_{r}"].fillna(99).astype(np.int16)
        u[f"rank_s_{r}"] = u[f"rank_s_{r}"].fillna(99).astype(np.int16)
    u["score"] = u[[f"score_{r}" for r in used]].max(axis=1)
    u["rank_q"] = u[[f"rank_q_{r}" for r in used]].min(axis=1).astype(np.int16)
    u["rank_s"] = u[[f"rank_s_{r}" for r in used]].min(axis=1).astype(np.int16)
    u["n_retrievers"] = sum((u[f"score_{r}"] > 0).astype(np.int8) for r in used)
    u.to_parquet(a.out or f"{CAND}/{a.split}.parquet", index=False)
    keep = (u.rank_q <= a.max_rank_q) | (u.rank_s <= a.max_rank_s)
    print(f"[{a.split}] union of {used}: {len(u):,} pairs; after rank pruning {keep.sum():,}", flush=True)

    if a.split == "train":
        truth = pd.read_parquet(f"{PQ}/train_pairs.parquet")
        s1 = pd.read_parquet(f"{PQ}/train_s1.parquet", columns=["entity_id", "country"])
        ctry = s1.set_index("entity_id").country
        hit = truth.merge(u, on=["s1_id", "cand_id"], how="left")
        for r in used:
            only = (hit[f"score_{r}"] > 0) & (hit.n_retrievers == 1)
            print(f"  recall {r:9s} alone {(hit[f'score_{r}'] > 0).mean():.4f}   found ONLY by {r}: {only.mean():.4f}")
        print(f"  recall union {hit.score.notna().mean():.4f}; after pruning "
              f"{((hit.rank_q <= a.max_rank_q) | (hit.rank_s <= a.max_rank_s)).mean():.4f}")
        for name, m in (("union", np.ones(len(u), bool)), ("pruned", keep.values)):
            ok = u.loc[m, ["s1_id", "cand_id"]].merge(truth, on=["s1_id", "cand_id"])
            f = per_entity_f05(ok, truth, s1.entity_id)
            print(f"  oracle macro F0.5 [{name}] = {f.mean():.4f}  by country "
                  f"{f.groupby(ctry.reindex(f.index).values).mean().round(4).to_dict()}")


if __name__ == "__main__":
    main()
