"""Baseline B0: exact-key matching (high precision, low recall).

key = country | sorted name tokens (legal forms / honorifics removed) | first house number.
An S2/S3 record is linked to an S1 only if exactly one S1 in the same split has that key.

  python src/b0_exact.py            # score on train, write test outputs to output/
"""
import os
import re
import sys
import time
import unicodedata
from multiprocessing import Pool

import pandas as pd
import ftfy
from anyascii import anyascii

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import per_entity_f05  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PQ = os.path.join(ROOT, "data", "parquet")
OUT = os.path.join(ROOT, "output")

STOP = set("""private limited pvt ltd llc llp inc incorporated corp corporation co company lp pllc pc plc
gmbh sarl sas sasu eurl sci sa ei the and m s smt sri shri mr mrs ms dr""".split())
NULLS = {"null", "n/a", "nan", "na", "none"}
NUM = re.compile(r"\d+")


def to_ascii(s: str) -> str:
    if not s.isascii():
        s = anyascii(unicodedata.normalize("NFKC", ftfy.fix_text(s)))
    return s.lower()


def name_key(s: str) -> str:
    toks = re.sub(r"[^0-9a-z]+", " ", to_ascii(s)).split()
    return " ".join(sorted(t for t in toks if t not in STOP))


def first_num(s: str) -> str:
    s = to_ascii(s)
    if s.strip() in NULLS:
        return ""
    m = NUM.search(s)
    return m.group(0).lstrip("0") or "0" if m else ""


def keys_chunk(df: pd.DataFrame) -> pd.Series:
    nk = df.business_name.map(name_key)
    fn = df.business_address.map(first_num)
    k = df.country + "|" + nk + "|" + fn
    return k.where((nk != "") & (fn != ""), "")


def add_keys(df: pd.DataFrame, procs: int) -> pd.DataFrame:
    chunks = [df.iloc[i:i + 200_000] for i in range(0, len(df), 200_000)]
    with Pool(procs) as p:
        df = df.assign(key=pd.concat(p.map(keys_chunk, chunks)))
    return df


def match(split: str, procs: int) -> pd.DataFrame:
    s1 = add_keys(pd.read_parquet(f"{PQ}/{split}_s1.parquet"), procs)
    cand = add_keys(pd.concat([pd.read_parquet(f"{PQ}/{split}_s{k}.parquet") for k in (2, 3)]), procs)
    s1 = s1[s1.key != ""]
    uniq = s1.drop_duplicates("key", keep=False)[["key", "entity_id"]].rename(columns={"entity_id": "s1_id"})
    m = cand[cand.key != ""].merge(uniq, on="key")[["s1_id", "entity_id"]].rename(columns={"entity_id": "cand_id"})
    return m


def write_outputs(pred: pd.DataFrame, s1_ids: pd.Series, path: str, col: str):
    lists = pred.groupby("s1_id").cand_id.agg(",".join)
    out = pd.DataFrame({"source1_entity_id": s1_ids.values})
    out[col] = out.source1_entity_id.map(lists).fillna("")
    out.to_csv(path, sep="\t", index=False, encoding="utf-8")


def main():
    procs = os.cpu_count() or 4
    t0 = time.time()
    pred = match("train", procs)
    truth = pd.read_parquet(f"{PQ}/train_pairs.parquet")
    s1 = pd.read_parquet(f"{PQ}/train_s1.parquet", columns=["entity_id", "country"])
    f = per_entity_f05(pred, truth, s1.entity_id)
    tp = pred.merge(truth, on=["s1_id", "cand_id"])
    print(f"[train] links={len(pred):,} precision={len(tp)/max(len(pred),1):.4f} "
          f"recall={len(tp)/len(truth):.4f} macroF0.5={f.mean():.4f}  ({time.time()-t0:.0f}s)")
    print("[train] macroF0.5 by country:", f.groupby(s1.set_index("entity_id").country.reindex(f.index).values).mean().round(4).to_dict())

    pred = match("test", procs)
    s1t = pd.read_parquet(f"{PQ}/test_s1.parquet", columns=["entity_id", "country"])
    os.makedirs(OUT, exist_ok=True)
    write_outputs(pred, s1t.entity_id, f"{OUT}/matching_results.tsv", "matched_entity_ids")
    write_outputs(pred, s1t.entity_id, f"{OUT}/candidate_pairs.tsv", "candidate_entity_ids")
    per = pred.groupby("s1_id").size().reindex(s1t.entity_id, fill_value=0)
    print(f"[test] links={len(pred):,}; share of S1 with >=1 link by country:",
          (per > 0).groupby(s1t.country.values).mean().round(3).to_dict())
    print(f"done in {time.time()-t0:.0f}s -> {OUT}/")


if __name__ == "__main__":
    main()
