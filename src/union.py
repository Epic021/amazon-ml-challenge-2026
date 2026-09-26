"""Merge the candidate sets of all retrievers into data/cand/{split}.parquet (polars, multi-threaded).

Keeps every retriever's score and ranks as columns (score_word, rank_q_word, ...), which flow
through features.py as model features, plus the combined columns the pipeline uses:
  score  = max score over retrievers
  rank_q = best (min) rank_q over retrievers,  rank_s = best rank_s
  n_retrievers = number of retrievers that proposed the pair

  python src/union.py --split train      # also reports marginal recall + oracle F0.5 ceiling
"""
import argparse
import os
import sys
import time

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import per_entity_f05  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))   # override for smoke runs
CAND = os.path.join(DATA, "cand")
PQ = os.path.join(DATA, "parquet")
RETRIEVERS = ("word", "namechar", "addr", "numaddr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--max_rank_q", type=int, default=10)
    ap.add_argument("--max_rank_s", type=int, default=15)
    ap.add_argument("--retrievers", default=",".join(RETRIEVERS), help="comma-separated subset to merge")
    ap.add_argument("--out", default=None, help="output path (default data/cand/{split}.parquet)")
    a = ap.parse_args()
    t0 = time.time()

    frames, used = [], []
    for r in a.retrievers.split(","):
        path = f"{CAND}/{a.split}_{r}.parquet"
        if not os.path.isfile(path):
            continue
        d = pl.read_parquet(path, columns=["s1_id", "cand_id", "score", "rank_q", "rank_s"])
        frames.append(d.with_columns(pl.lit(r).alias("ret")))
        used.append(r)
        print(f"  + {r}: {len(d):,} pairs", flush=True)
    long = pl.concat(frames, how="vertical_relaxed")
    aggs = []
    for r in used:
        m = pl.col("ret") == r
        aggs += [pl.col("score").filter(m).max().alias(f"score_{r}"),
                 pl.col("rank_q").filter(m).min().alias(f"rank_q_{r}"),
                 pl.col("rank_s").filter(m).min().alias(f"rank_s_{r}")]
    u = long.group_by(["s1_id", "cand_id"]).agg(aggs)
    u = u.with_columns([pl.col(f"score_{r}").fill_null(0).cast(pl.Float32) for r in used]
                       + [pl.col(f"rank_{d}_{r}").fill_null(99).cast(pl.Int16) for r in used for d in ("q", "s")])
    u = u.with_columns(
        pl.max_horizontal([f"score_{r}" for r in used]).alias("score"),
        pl.min_horizontal([f"rank_q_{r}" for r in used]).cast(pl.Int16).alias("rank_q"),
        pl.min_horizontal([f"rank_s_{r}" for r in used]).cast(pl.Int16).alias("rank_s"),
        pl.sum_horizontal([(pl.col(f"score_{r}") > 0).cast(pl.Int8) for r in used]).cast(pl.Int8).alias("n_retrievers"),
    )
    u.write_parquet(a.out or f"{CAND}/{a.split}.parquet")
    keep = (pl.col("rank_q") <= a.max_rank_q) | (pl.col("rank_s") <= a.max_rank_s)
    print(f"[{a.split}] union of {used}: {len(u):,} pairs; after rank pruning {u.filter(keep).height:,} "
          f"({time.time() - t0:.0f}s)", flush=True)

    if a.split == "train":
        truth = pl.read_parquet(f"{PQ}/train_pairs.parquet")
        s1 = pl.read_parquet(f"{PQ}/train_s1.parquet", columns=["entity_id", "country"])
        hit = truth.join(u, on=["s1_id", "cand_id"], how="left")
        found = hit["score"].is_not_null()
        for r in used:
            alone = hit[f"score_{r}"].fill_null(0) > 0
            only = alone & (hit["n_retrievers"].fill_null(0) == 1)
            print(f"  recall {r:9s} alone {alone.mean():.4f}   found ONLY by {r}: {only.mean():.4f}")
        pruned = found & ((hit["rank_q"] <= a.max_rank_q) | (hit["rank_s"] <= a.max_rank_s)).fill_null(False)
        print(f"  recall union {found.mean():.4f}; after pruning {pruned.mean():.4f}")
        ctry = s1.to_pandas().set_index("entity_id").country
        tr_pd = truth.to_pandas()
        for name, mask in (("union", found), ("pruned", pruned)):
            ok = hit.filter(mask).select(["s1_id", "cand_id"]).to_pandas()
            f = per_entity_f05(ok, tr_pd, ctry.index)
            print(f"  oracle macro F0.5 [{name}] = {f.mean():.4f}  by country "
                  f"{f.groupby(ctry.reindex(f.index).values).mean().round(4).to_dict()}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
