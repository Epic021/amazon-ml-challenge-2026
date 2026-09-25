"""Convert the challenge TSVs to Parquet once; everything downstream reads Parquet.

Outputs (in data/parquet/):
  {train,test}_s{1,2,3}.parquet   entity_id, business_name, business_address, country, src
  train_pairs.parquet             s1_id, cand_id  (one row per true match)
  train_s1_nmatch.parquet         s1_id, n_match  (0 = singleton)
"""
import csv
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("DATA_DIR", os.path.join(ROOT, "student_resource", "dataset"))   # raw TSVs
OUT = os.path.join(os.environ.get("BER_DATA", os.path.join(ROOT, "data")), "parquet")


def read_tsv(path):
    # QUOTE_NONE: names contain quotes; keep_default_na=False: "NA"/"NULL" stay strings
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE, engine="c")


def main():
    os.makedirs(OUT, exist_ok=True)
    for split in ("train", "test"):
        for k in (1, 2, 3):
            src = os.path.join(SRC, split, f"{split}_source{k}.tsv")
            if not os.path.isfile(src):
                sys.exit(f"missing {src}")
            df = read_tsv(src)
            df["src"] = f"S{k}"
            dst = os.path.join(OUT, f"{split}_s{k}.parquet")
            df.to_parquet(dst, index=False)
            print(f"{dst}: {len(df):,} rows, countries={df.country.value_counts().to_dict()}")

    gt = read_tsv(os.path.join(SRC, "train", "train_ground_truth.tsv"))
    lists = gt.matched_entity_ids.str.split(",")
    gt.assign(n_match=lists.map(lambda l: 0 if l == [""] else len(l))) \
      .rename(columns={"source1_entity_id": "s1_id"})[["s1_id", "n_match"]] \
      .to_parquet(os.path.join(OUT, "train_s1_nmatch.parquet"), index=False)
    pairs = gt.assign(cand_id=lists).explode("cand_id")
    pairs = pairs[pairs.cand_id != ""].rename(columns={"source1_entity_id": "s1_id"})[["s1_id", "cand_id"]]
    pairs.to_parquet(os.path.join(OUT, "train_pairs.parquet"), index=False)
    print(f"train_pairs: {len(pairs):,} true pairs; singletons: {(lists.map(len).eq(1) & gt.matched_entity_ids.eq('')).sum():,}")


if __name__ == "__main__":
    main()
