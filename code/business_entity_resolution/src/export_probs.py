#!/usr/bin/env python3
"""Export one model's probabilities for blending with another team model.

  python3 export_probs.py NAME OUTDIR
    OUTDIR/test_probs.parquet       s1_id, cand_id, p        (test pairs, p >= 0.01)
    OUTDIR/train_oof_probs.parquet  s1_id, cand_id, p, label (training pairs, out-of-fold, p >= 0.01)
p is the final (stage-2) probability before exclusive assignment and set decoding.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import WORK, log, split_dir


def write(split, p, out, label=None):
    d = split_dir(split)
    c = np.load(d / "cands.npz")
    keep = np.flatnonzero(p >= 0.01)
    s1 = pa.ipc.open_file(pa.memory_map(str(d / "s1.arrow"))).read_all()["entity_id"].combine_chunks()
    rec = pa.ipc.open_file(pa.memory_map(str(d / "rec.arrow"))).read_all()["entity_id"].combine_chunks()
    cols = {"s1_id": s1.take(pa.array(c["s1"][keep])), "cand_id": rec.take(pa.array(c["rec"][keep])),
            "p": pa.array(p[keep].astype(np.float32))}
    if label is not None:
        cols["label"] = pa.array(label[keep].astype(np.int8))
    pq.write_table(pa.table(cols), out, compression="zstd")
    log(f"{out}: {len(keep):,} rows")


if __name__ == "__main__":
    name, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    write("test", np.load(split_dir("test") / f"pred_{name}.npz")["p2"], out / "test_probs.parquet")
    d = split_dir("train")
    c, lab = np.load(d / "cands.npz"), np.load(d / "labels.npz")
    y = lab["true_row"][c["rec"]] == c["s1"]
    write("train", np.load(WORK / "models" / name / "oof_p2.npy"), out / "train_oof_probs.parquet", y)
