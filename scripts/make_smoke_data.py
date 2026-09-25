"""Build a tiny but structurally faithful dataset for end-to-end smoke tests.

Keeps ~1% of S1 entities (spread over every fold), all their true matches, and a random ~1% of the
remaining S2/S3 records as distractors. Reads the full Parquet under ROOT/data, writes to $BER_DATA.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL = os.path.join(ROOT, "data", "parquet")
OUT = os.path.join(os.environ["BER_DATA"], "parquet")


def keep_s1(ids: pd.Series) -> np.ndarray:
    return ((ids.str[3:].astype(np.int64) // 10) % 100 == 3).values   # independent of fold (id % 10)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    pairs = pd.read_parquet(f"{FULL}/train_pairs.parquet")
    for split in ("train", "test"):
        s1 = pd.read_parquet(f"{FULL}/{split}_s1.parquet")
        s1 = s1[keep_s1(s1.entity_id)]
        s1.to_parquet(f"{OUT}/{split}_s1.parquet", index=False)
        linked = set(pairs.cand_id[pairs.s1_id.isin(set(s1.entity_id))]) if split == "train" else set()
        for k in (2, 3):
            df = pd.read_parquet(f"{FULL}/{split}_s{k}.parquet")
            m = df.entity_id.isin(linked).values | (rng.random(len(df)) < 0.012)
            df[m].to_parquet(f"{OUT}/{split}_s{k}.parquet", index=False)
        print(f"{split}: S1={len(s1):,}")
    p = pairs[pairs.s1_id.isin(set(pd.read_parquet(f"{OUT}/train_s1.parquet").entity_id))]
    p.to_parquet(f"{OUT}/train_pairs.parquet", index=False)
    nm = pd.read_parquet(f"{FULL}/train_s1_nmatch.parquet")
    nm[nm.s1_id.isin(set(p.s1_id)) | keep_s1(nm.s1_id)].to_parquet(f"{OUT}/train_s1_nmatch.parquet", index=False)
    print(f"train pairs: {len(p):,}")


if __name__ == "__main__":
    main()
