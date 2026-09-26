"""Convert the teammate retriever's output (code/business_entity_resolution: WORK/<split>/cands.npz,
row numbers into s1.arrow / rec.arrow) into our candidate format data/cand/{split}_theirs.parquet.

  python scripts/theirs_to_cand.py --split train --work /work/theirs_work
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))


def ids(path):
    return pa.ipc.open_file(pa.memory_map(path)).read_all().column("entity_id").to_numpy(zero_copy_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    d = os.path.join(a.work, a.split)
    c = np.load(os.path.join(d, "cands.npz"))
    s1_ids, rec_ids = ids(os.path.join(d, "s1.arrow")), ids(os.path.join(d, "rec.arrow"))
    k1, k2 = int(c["rank_r"].max()), int(c["rank_s"].max())      # "absent" rank = K + 1
    out = pd.DataFrame({
        "s1_id": s1_ids[c["s1"]], "cand_id": rec_ids[c["rec"]], "score": c["score"].astype(np.float32),
        "rank_q": np.where(c["rank_r"] >= k1, 99, c["rank_r"]).astype(np.int16),
        "rank_s": np.where(c["rank_s"] >= k2, 99, c["rank_s"]).astype(np.int16)})
    path = os.path.join(DATA, "cand", f"{a.split}_theirs.parquet")
    out.to_parquet(path, index=False)
    print(f"{path}: {len(out):,} pairs, {out.s1_id.nunique():,} S1")


if __name__ == "__main__":
    main()
