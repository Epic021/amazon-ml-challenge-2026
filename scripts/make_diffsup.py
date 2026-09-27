"""Collective-ER features for stage 2: is a pair's name difference repeated by an independent record?

For every stage-2 pair (union of our stage-1 pairs and the teammate's pairs):
  diff_support = # other-source candidates of the same S1 with the identical (added | dropped) word signature
  add_support  = # other-source candidates of the same S1 with the identical added words
(-1 when there is no difference / no added word). Noise is independent per source, so a difference repeated
in another source points to a different business (a neighbour/orphan entity), not a noisy copy.
Writes data/extra/{split}_diffsup.parquet (s1_id, cand_id, diff_support, add_support).

  python scripts/make_diffsup.py --tag b5 --friend_dir friend_v2
"""
import argparse
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diff_support import with_support, KEY, DATA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b5")
    ap.add_argument("--friend_dir", default="friend_v2")
    a = ap.parse_args()
    os.makedirs(f"{DATA}/extra", exist_ok=True)
    fold = pl.col("s1_id").str.slice(3).cast(pl.Int64) % 10
    for split in ("train", "test"):
        p = pl.read_parquet(f"{DATA}/pred/{split}_{a.tag}.parquet", columns=KEY)
        t = pl.read_parquet(f"{DATA}/{a.friend_dir}/{'train_oof_probs' if split == 'train' else 'test_probs'}.parquet", columns=KEY)
        if split == "train":
            p, t = p.filter(fold >= 5), t.filter(fold >= 5)
        pairs = pl.concat([p, t]).unique()
        d = with_support(pairs, split).select(KEY + ["diff_support", "add_support"])
        d = d.with_columns(pl.col("diff_support").cast(pl.Int16), pl.col("add_support").cast(pl.Int16))
        d.write_parquet(f"{DATA}/extra/{split}_diffsup.parquet")
        print(f"{split}: {d.height:,} pairs; diff_support>=1 share {(d['diff_support'] >= 1).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
