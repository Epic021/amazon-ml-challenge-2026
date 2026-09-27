#!/usr/bin/env python3
"""Stage 0: read one split's TSVs, normalise every record, write memory-mappable Arrow tables.

  WORK/<split>/s1.arrow   one row per S1 record
  WORK/<split>/rec.arrow  one row per S2/S3 record (S2 first, then S3)
  WORK/<split>/labels.npz (train only) true S1 row of every record, true set size of every S1
  WORK/<split>/areas.json  address components learned as "area" (too common to identify a
                           business) for every country, from this split's own records, no labels

Usage: python3 prep.py train|test
"""
import csv
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa

import normalize
from common import DATA, FRAC, NPROC, id_hash, log, split_dir
from normalize import addr_components, addr_tokens, content, name_tokens, retrieval_tokens, romanize

COLS = ["ntoks", "nind", "awords", "anums", "aempty", "nrom", "arom", "rt_n", "rt_a", "ckey"]
AREA_FRAC = float(os.environ.get("BER_AREA_FRAC", "0.001"))   # component in >= 0.1% of a country's records
AREA_MIN = 20


def count_chunk(args):
    """Pass 1: how often each digit-free address component occurs, per country (records, not tokens)."""
    addrs, countries = args
    cnt = Counter()
    for addr, c in zip(addrs, countries):
        for comp in set(addr_components(addr)):
            if not any(ch.isdigit() for ch in comp):
                cnt[(c, comp)] += 1
    return cnt


def learn_areas(addrs, countries, pool):
    step = 50_000
    chunks = [(addrs[i:i + step], countries[i:i + step]) for i in range(0, len(addrs), step)]
    total = Counter()
    for part in pool.imap_unordered(count_chunk, chunks):
        total.update(part)
    n_c = Counter(countries)
    areas = {}
    for (c, comp), k in total.items():
        if k >= max(AREA_MIN, AREA_FRAC * n_c[c]):
            areas.setdefault(c, set()).add(comp)
    return areas


def _init_areas(areas):
    normalize.AREAS = areas


def read_tsv(path):
    df = pd.read_csv(path, sep="\t", dtype=str, na_filter=False, quoting=csv.QUOTE_NONE)
    df.columns = [c.strip() for c in df.columns]
    return df


def norm_chunk(args):
    names, addrs, countries = args
    out = {c: [] for c in COLS}
    for name, addr, country in zip(names, addrs, countries):
        nt, ind = name_tokens(name, country)
        af = addr_tokens(addr, country)
        rn, ra = retrieval_tokens(nt, af)
        out["ntoks"].append(" ".join(nt))
        out["nind"].append(int(ind))
        out["awords"].append(" ".join(af[0]) if af else "")
        out["anums"].append(" ".join(af[2]) if af else "")
        out["aempty"].append(int(af is None or (not af[0] and not af[2])))   # nothing that identifies
        out["nrom"].append(romanize(name))
        out["arom"].append(romanize(addr))
        out["rt_n"].append(rn)
        out["rt_a"].append(ra)
        out["ckey"].append(" ".join(sorted(content(nt))))
    return out


def normalise(df, pool):
    names, addrs = df["business_name"].tolist(), df["business_address"].tolist()
    countries = df["country"].tolist()
    step = 20_000
    chunks = [(names[i:i + step], addrs[i:i + step], countries[i:i + step]) for i in range(0, len(names), step)]
    res = {c: [] for c in COLS}
    for part in pool.imap(norm_chunk, chunks):
        for c in COLS:
            res[c].extend(part[c])
    for c in COLS:
        df[c] = res[c]
    return df


def main(split):
    d = split_dir(split)
    src = DATA / split
    s1 = read_tsv(src / f"{split}_source1.tsv")
    recs = []
    for k in (2, 3):
        r = read_tsv(src / f"{split}_source{k}.tsv")
        r["src"] = k
        recs.append(r)
    rec = pd.concat(recs, ignore_index=True)
    s1["id"] = s1["entity_id"].str.slice(3).astype(np.int64)
    rec["id"] = rec["entity_id"].str.slice(3).astype(np.int64)
    log(f"{split}: {len(s1):,} S1, {len(rec):,} S2/S3 records read")

    true_s1_id = None
    if split == "train":
        gt = read_tsv(src / "train_ground_truth.tsv")
        m = gt.assign(mid=gt["matched_entity_ids"].str.split(",")).explode("mid")
        m = m[m["mid"].fillna("") != ""]
        key = m["mid"].str.slice(0, 2).map({"S2": 2, "S3": 3}).astype(np.int64) * 10**10 + \
            m["mid"].str.slice(3).astype(np.int64)
        lut = pd.Series(m["source1_entity_id"].str.slice(3).astype(np.int64).to_numpy(), index=key.to_numpy())
        true_s1_id = lut.reindex(rec["src"].to_numpy() * 10**10 + rec["id"].to_numpy()).fillna(-1).astype(np.int64).to_numpy()

    if FRAC > 0:   # smoke test: a small slice of S1 businesses with their records
        keep1 = id_hash(s1["id"].to_numpy(), 1000) < int(FRAC * 1000)
        kept = set(s1["id"].to_numpy()[keep1].tolist())
        if true_s1_id is not None:
            keepr = np.array([t in kept for t in true_s1_id]) | \
                ((true_s1_id < 0) & (id_hash(rec["id"].to_numpy(), 1000) < int(FRAC * 1000)))
            true_s1_id = true_s1_id[keepr]
        else:
            keepr = id_hash(rec["id"].to_numpy(), 1000) < int(FRAC * 1000)
        s1, rec = s1[keep1].reset_index(drop=True), rec[keepr].reset_index(drop=True)
        log(f"smoke slice: {len(s1):,} S1, {len(rec):,} records")

    # areas: learned per country from this split's records (S1, S2, S3), no labels
    with Pool(NPROC) as pool:
        areas = learn_areas(s1["business_address"].tolist() + rec["business_address"].tolist(),
                            s1["country"].tolist() + rec["country"].tolist(), pool)
    (d / "areas.json").write_text(json.dumps({c: sorted(v) for c, v in areas.items()}, indent=0))
    log("areas learned: " + ", ".join(f"{c} {len(v)}" for c, v in sorted(areas.items())))
    with Pool(NPROC, initializer=_init_areas, initargs=(areas,)) as pool:
        s1 = normalise(s1, pool)
        log("S1 normalised")
        rec = normalise(rec, pool)
        log("records normalised")

    # how common a business name / an address is among this split's S1 businesses of the same country
    name_cnt = s1.groupby(["country", "ckey"]).size()
    addr_cnt = s1[s1["awords"] != ""].groupby(["country", "awords"]).size()
    for df in (s1, rec):
        df["name_freq"] = name_cnt.reindex(pd.MultiIndex.from_frame(df[["country", "ckey"]])).fillna(0).to_numpy(np.int32)
        df["addr_freq"] = addr_cnt.reindex(pd.MultiIndex.from_frame(df[["country", "awords"]])).fillna(0).to_numpy(np.int32)
        df.loc[df["ckey"] == "", "name_freq"] = 0
        df.loc[df["awords"] == "", "addr_freq"] = 0

    keep_cols = ["id", "entity_id", "country", "business_name", "business_address"] + COLS + ["name_freq", "addr_freq"]
    for name, df, extra in (("s1", s1, []), ("rec", rec, ["src"])):
        t = pa.Table.from_pandas(df[keep_cols + extra], preserve_index=False)
        with pa.OSFile(str(d / f"{name}.arrow"), "wb") as f, pa.ipc.new_file(f, t.schema) as w:
            w.write_table(t)
    if true_s1_id is not None:
        s1_row = pd.Series(np.arange(len(s1)), index=s1["id"].to_numpy())
        true_row = s1_row.reindex(true_s1_id).fillna(-1).astype(np.int64).to_numpy()
        true_size = np.bincount(true_row[true_row >= 0], minlength=len(s1)).astype(np.int32)
        np.savez(d / "labels.npz", true_row=true_row, true_size=true_size)
        log(f"labels: {int((true_row >= 0).sum()):,} matched records, {int((true_size == 0).sum()):,} singletons")
    log(f"{split} prep written to {d}")


if __name__ == "__main__":
    main(sys.argv[1])
