"""Add the G5 name-uniqueness columns to existing feature files without recomputing everything.
Uses exactly the functions features.py uses, so a full regeneration gives identical columns.

  python scripts/add_name_counts.py --split train
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from features import add_name_counts, name_count_cols, NAME_COUNT_COLS, NORM, FEAT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    a = ap.parse_args()
    rec = pd.concat([pd.read_parquet(f"{NORM}/{a.split}_s{k}.parquet", columns=["id", "src", "country", "name_core"])
                     for k in (1, 2, 3)], ignore_index=True)
    r = add_name_counts(rec).set_index("id")
    path = f"{FEAT}/{a.split}.parquet"
    f = pd.read_parquet(path)
    f = f.drop(columns=[c for c in NAME_COUNT_COLS if c in f.columns])
    f[NAME_COUNT_COLS] = name_count_cols(f, r).values
    f.to_parquet(path, index=False)
    print(f"{path}: +{NAME_COUNT_COLS} -> {f.shape}")
    print(f[NAME_COUNT_COLS].describe().T[["mean", "50%", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
