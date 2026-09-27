"""5th retriever: fine-tuned multilingual bi-encoder (E1), record -> S1, top-k within a country.

Queries: the country's S2/S3 records WITHOUT a confident owner (best stacker-v3 p < --owner_p, or absent from
the stage-2 pair universe). Records already matched confidently cannot gain an owner (one owner per record).
Keys: all S1 of the country. Writes data/cand_bienc/{split}_{country}.parquet with
(s1_id, cand_id, bienc_sim, bienc_rank, new) where new = 1 if the pair is not already in the stage-2 universe.

  /work/venv_nn/bin/python scripts/bienc_retrieve.py --split train --country India
"""
import argparse
import os
import sys
import time

import faiss
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_biencoder import encode_parallel, texts  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
KEY = ["s1_id", "cand_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--country", default="India")
    ap.add_argument("--model", default=f"{DATA}/models/e1_biencoder_india")
    ap.add_argument("--stack_tag", default="b5sb3")
    ap.add_argument("--owner_p", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=20)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(f"{DATA}/cand_bienc", exist_ok=True)
    s1 = pl.read_parquet(f"{DATA}/parquet/{a.split}_s1.parquet", columns=["entity_id", "business_name", "business_address", "country"]) \
           .filter(pl.col("country") == a.country).rename({"entity_id": "s1_id"})
    rec = pl.concat([pl.read_parquet(f"{DATA}/parquet/{a.split}_s{k}.parquet", columns=["entity_id", "business_name", "business_address", "country"])
                     for k in (2, 3)]).filter(pl.col("country") == a.country).rename({"entity_id": "cand_id"})
    uni = pl.read_parquet(f"{DATA}/pred/{a.split}_{a.stack_tag}.parquet", columns=KEY + ["p"])
    best = uni.group_by("cand_id").agg(pl.col("p").max().alias("best_p"))
    q = rec.join(best, on="cand_id", how="left").filter(pl.col("best_p").is_null() | (pl.col("best_p") < a.owner_p))
    print(f"[{a.split}/{a.country}] S1 {s1.height:,}; records {rec.height:,}; queries without a confident owner "
          f"{q.height:,} ({time.time() - t0:.0f}s)", flush=True)
    e_s1 = encode_parallel(a.model, texts(s1))
    print(f"  S1 encoded ({time.time() - t0:.0f}s)", flush=True)
    e_q = encode_parallel(a.model, texts(q))
    print(f"  queries encoded ({time.time() - t0:.0f}s)", flush=True)
    index = faiss.IndexFlatIP(e_s1.shape[1])
    index.add(e_s1)
    faiss.omp_set_num_threads(64)
    sim, nn = index.search(e_q, a.k)
    s1_ids = np.array(s1["s1_id"].to_list())
    out = pl.DataFrame({"cand_id": np.repeat(np.array(q["cand_id"].to_list()), a.k),
                        "s1_id": s1_ids[nn.ravel()],
                        "bienc_sim": sim.ravel().astype(np.float32),
                        "bienc_rank": np.tile(np.arange(1, a.k + 1, dtype=np.int16), len(q))})
    out = out.join(uni.select(KEY).with_columns(pl.lit(0, pl.Int8).alias("new")), on=KEY, how="left") \
             .with_columns(pl.col("new").fill_null(1).cast(pl.Int8))
    out.write_parquet(f"{DATA}/cand_bienc/{a.split}_{a.country}.parquet")
    msg = f"  pairs {out.height:,}; new (not in universe) {int(out['new'].sum()):,}"
    if a.split == "train":
        truth = pl.read_parquet(f"{DATA}/parquet/train_pairs.parquet").with_columns(pl.lit(1).alias("y"))
        o = out.join(truth, on=KEY, how="left").with_columns(pl.col("y").fill_null(0))
        fold = pl.col("s1_id").str.slice(3).cast(pl.Int64) % 10
        n5 = o.filter((fold == 5) & (pl.col("new") == 1))
        msg += f"; true NEW pairs: {int(o.filter(pl.col('new') == 1)['y'].sum()):,} (fold 5: {int(n5['y'].sum()):,} of {n5.height:,})"
    print(msg + f" ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
