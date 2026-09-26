#!/usr/bin/env python3
"""Pick real examples for every insight in the report, and gather what the audit knows about
each case that was checked by hand. Random sampling uses a fixed seed (21).

Needs the outputs of build_pairs.py and analyze.py. Writes eda_output/label_audit/examples.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
W = HERE / "work"
rng = np.random.default_rng(21)
NAME_DIFF = ["identical", "legal form / filler only", "typo only", "typo + legal form / filler",
             "one word added", "2+ words added", "one word dropped", "2+ words dropped",
             "words replaced", "nothing in common",
             "other script: same words", "other script: legal form / filler only", "other script: words added",
             "other script: words dropped", "other script: words replaced", "other script: nothing in common"]
ADDR_DIFF = ["identical", "only numbers differ", "typo in words", "typo in words + numbers differ",
             "words added", "words dropped", "words replaced", "nothing in common",
             "record address empty", "S1 address empty"]
NUM_REL = ["same numbers", "number added", "number dropped", "number cut short", "number: one character changed",
           "number nearby (<=50 apart)", "number far-off", "n/a"]
COUNTRY = ["US", "India"]
TEXT = ["s1_id", "rec_id", "s1_name", "rec_name", "s1_addr", "rec_addr",
        "name_added", "name_dropped", "addr_added", "addr_dropped"]

# ---------------------------------------------------------------- load
P = pq.read_table(W / "pairs.parquet", columns=["country", "src", "label", "reach", "nd", "ad", "rel"]).to_pandas()
country, src, y = P["country"].to_numpy(), P["src"].to_numpy(), P["label"].to_numpy()
comp = P["reach"].to_numpy() > 0
nd, ad, rel = P["nd"].to_numpy(), P["ad"].to_numpy(), P["rel"].to_numpy()
p = np.load(W / "oof_p.npy")
pflag = np.load(W / "pattern_flag.npy")
mflag = np.load(W / "model_flag.npy")
PS = pd.read_csv(HERE / "pattern_summary.tsv", sep="\t")
PS_KEY = {(r.country, r.source, r.pattern): (int(r.pairs), float(r.match_rate_pct), r.majority)
          for r in PS.itertuples(index=False)}
S1 = pq.read_table(W / "s1.parquet", columns=["s1", "s1_id", "name", "addr", "true_size"]).to_pandas().set_index("s1")


def take_rows(path, columns, rows):
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    out, start = [], 0
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=500_000):
        n = batch.num_rows
        lo, hi = np.searchsorted(rows, [start, start + n])
        if hi > lo:
            out.append(batch.take(rows[lo:hi] - start).to_pandas())
        start += n
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=columns)


def pattern_text(i):
    return f"name: {NAME_DIFF[nd[i]]} | address: {ADDR_DIFF[ad[i]]} | numbers: {NUM_REL[rel[i]]}"


def describe(rows):
    rows = [int(i) for i in rows]
    if not rows:
        return []
    order = sorted(set(rows))
    txt = take_rows(W / "pairs.parquet", TEXT, order)
    txt.index = order
    out = []
    for i in rows:
        key = (COUNTRY[country[i]], f"S{src[i]}", pattern_text(i))
        ps = PS_KEY.get(key) if comp[i] else None
        out.append({**{k: (v if v == v else "") for k, v in txt.loc[i].to_dict().items()},
                    "label": "match" if y[i] else "no match", "pattern": pattern_text(i),
                    "country": key[0], "source": key[1], "comparable": bool(comp[i]),
                    "pattern_pairs": ps[0] if ps else None, "pattern_match_rate": ps[1] if ps else None,
                    "pattern_majority": ps[2] if ps else None, "model_p": round(float(p[i]), 4),
                    "flag_pattern": bool(pflag[i]), "flag_model": bool(mflag[i])})
    return out


def pick(mask, label, k):
    idx = np.flatnonzero(mask & (y == label))
    return list(rng.choice(idx, min(k, len(idx)), replace=False)) if len(idx) else []


def group(caption, rows):
    return {"caption": caption, "rows": describe(rows)}


def s1_text(s1_num):
    if s1_num is None or s1_num < 0 or s1_num not in S1.index:
        return None
    r = S1.loc[s1_num]
    return {"s1_id": r["s1_id"], "name": r["name"], "addr": r["addr"], "true_size": int(r["true_size"])}


insights = {}

# identical inputs
R = pq.read_table(W / "records.parquet", columns=["country", "label", "true_s1", "sig_raw", "paired"]).to_pandas()
dup = R.duplicated(["country", "sig_raw"], keep=False) & (R["label"] == 1)
anchor = int(rng.choice(np.flatnonzero(dup.to_numpy())))
members = np.flatnonzero(((R["country"] == R.at[anchor, "country"]) & (R["sig_raw"] == R.at[anchor, "sig_raw"])).to_numpy())
recs = take_rows(W / "records.parquet", ["rec_id", "name", "addr", "true_s1"], members)
raw_dup = {"records": recs[["rec_id", "name", "addr"]].to_dict("records"),
           "assigned": [s1_text(int(t)) for t in recs["true_s1"]]}
insights["identical"] = {
    "raw_duplicate_group": raw_dup,
    "groups": [group("Name and address identical after normalisation: match", pick(comp & (nd == 0) & (ad == 0), 1, 3)),
               group("The only identical pair that is not a match", pick(comp & (nd == 0) & (ad == 0), 0, 5))],
}

# only the numbers differ, by name change
insights["numbers_only"] = {"groups": [
    group("Name identical, only numbers differ: match", pick(comp & (ad == 1) & (nd == 0), 1, 2)),
    group("Name identical, only numbers differ: no match", pick(comp & (ad == 1) & (nd == 0), 0, 2)),
    group("One word added to the name, only numbers differ: no match", pick(comp & (ad == 1) & (nd == 4), 0, 2)),
    group("One word added to the name, only numbers differ: match", pick(comp & (ad == 1) & (nd == 4), 1, 2)),
    group("Only a legal form changed, only numbers differ: match", pick(comp & (ad == 1) & (nd == 1), 1, 1)),
    group("Only a legal form changed, only numbers differ: no match", pick(comp & (ad == 1) & (nd == 1), 0, 1)),
]}

# name identical, by number change
g = []
for code in [1, 2, 3, 4, 5, 6]:
    m = comp & (ad == 1) & (nd == 0) & (rel == code)
    g.append(group(f"Name identical, {NUM_REL[code]}: match", pick(m, 1, 1)))
    g.append(group(f"Name identical, {NUM_REL[code]}: no match", pick(m, 0, 1)))
insights["number_change"] = {"groups": [x for x in g if x["rows"]]}

# address identical, by name change
g = []
for code in [1, 2, 4, 8, 9]:
    m = comp & (ad == 0) & (nd == code)
    g.append(group(f"Address identical, name {NAME_DIFF[code]}: match", pick(m, 1, 1)))
    g.append(group(f"Address identical, name {NAME_DIFF[code]}: no match", pick(m, 0, 1)))
insights["address_identical"] = {"groups": [x for x in g if x["rows"]]}

# same name, empty address
dc = pd.read_csv(HERE / "duplicate_conflicts.tsv", sep="\t", dtype=str, keep_default_na=False)
dc = dc[(dc["fingerprint"] == "identical after normalisation") & (dc["addr"].str.strip() == "")]
empty_groups = []
for gid in rng.choice(dc["gid"].unique(), min(2, dc["gid"].nunique()), replace=False):
    rows = dc[dc["gid"] == gid]
    empty_groups.append({"conflict": rows["conflict"].iloc[0], "records": [
        {"rec_id": r.rec_id, "name": r.name, "assigned_s1": r.assigned_s1,
         "s1": s1_text(int(r.assigned_s1[3:])) if r.assigned_s1.startswith("S1-") else None}
        for r in rows.itertuples(index=False)]})
insights["empty_address"] = {"conflict_groups": empty_groups}

# the pattern with the most pairs against its majority, where the model still gets both labels right
PS["minority"] = np.minimum(PS["matches"], PS["non_matches"]) / PS["pairs"]
mixed = PS.sort_values("against_majority", ascending=False).iloc[0]   # the most contradicted pattern
mnd, mad, mrel = (NAME_DIFF.index(mixed["pattern"].split(" | ")[0][6:]),
                  ADDR_DIFF.index(mixed["pattern"].split(" | ")[1][9:]),
                  NUM_REL.index(mixed["pattern"].split(" | ")[2][9:]))
m = (comp & (country == COUNTRY.index(mixed["country"])) & (src == int(mixed["source"][1:]))
     & (nd == mnd) & (ad == mad) & (rel == mrel))
insights["finer_level"] = {
    "pattern": {"country": mixed["country"], "source": mixed["source"], "pattern": mixed["pattern"],
                "pairs": int(mixed["pairs"]), "match_rate": float(mixed["match_rate_pct"])},
    "groups": [group("Same pattern, match the model gets right (p > 0.9)", pick(m & (p > 0.9), 1, 2)),
               group("Same pattern, no match the model gets right (p < 0.1)", pick(m & (p < 0.1), 0, 2))]}

# the model's strongest contradictions
insights["model_flags"] = {"groups": [
    group("Labelled match, model p < 0.02", pick(comp & mflag, 1, 3)),
    group("Labelled no match, model p > 0.98", pick(comp & mflag, 0, 3))]}

# names in another script
g = []
for code in [10, 11, 12, 13, 14, 15]:
    m = comp & (nd == code)
    g.append(group(f"Name {NAME_DIFF[code]}: match", pick(m, 1, 1)))
    g.append(group(f"Name {NAME_DIFF[code]}: no match", pick(m, 0, 1)))
insights["other_script"] = {"groups": [x for x in g if x["rows"]]}

# coverage gaps
unpaired = np.flatnonzero(((R["label"] == 0) & (R["paired"] == 0)).to_numpy())
up = take_rows(W / "records.parquet", ["rec_id", "name", "addr"], rng.choice(unpaired, 3, replace=False))
insights["coverage"] = {
    "groups": [group("Match that shares no blocking key with its S1", pick(~comp, 1, 3))],
    "unpaired_unmatched": up.to_dict("records")}

# ---------------------------------------------------------------- the cases checked by hand
HAND = [  # (S1 business, records looked at), in the order they came up
    ("S1-302869473", ["S2-521228093", "S3-765861618"]),
    ("S1-730719211", ["S2-467798366", "S2-61695240", "S2-241456014", "S3-951830396"]),
    ("S1-262997549", ["S2-879297655", "S3-910786226"]),
    ("S1-989924267", ["S2-941206792", "S2-973482655", "S3-262595741"]),
    ("S1-783518091", ["S3-188760616", "S3-44406563"]),
    ("S1-14448673", ["S3-219141874"]),
    ("S1-115182842", ["S2-464452285", "S2-440934584", "S3-176968062", "S2-716420516"]),
    ("S1-733655541", ["S2-229535269"]),
    ("S1-570477188", ["S2-783520744"]),
    ("S1-963588162", ["S3-481869771"]),
    ("S1-250404567", ["S2-32283955"]),
    ("S1-71142843", ["S2-964683665", "S3-665555605"]),
]
want = [r for _, rs in HAND for r in rs]
pa_arr = pa.array(want)
rec_col = pq.read_table(W / "pairs.parquet", columns=["rec_id"])["rec_id"].combine_chunks()
pos = pc.indices_nonzero(pc.is_in(rec_col, value_set=pa_arr)).to_numpy()
ids_at = rec_col.take(pos).to_pylist()
row_of = dict(zip(ids_at, pos.tolist()))
rec_all = pq.read_table(W / "records.parquet", columns=["rec_id"])["rec_id"].combine_chunks()
rpos = pc.indices_nonzero(pc.is_in(rec_all, value_set=pa_arr)).to_numpy()
rrows = take_rows(W / "records.parquet", ["rec_id", "name", "addr", "label", "true_s1", "paired"], rpos).set_index("rec_id")
hand = []
for s1_id, rec_ids in HAND:
    s1 = s1_text(int(s1_id[3:]))
    rows = []
    for rid in rec_ids:
        if rid in row_of:
            d = describe([row_of[rid]])[0]
        else:
            r = rrows.loc[rid]
            d = {"rec_id": rid, "rec_name": r["name"], "rec_addr": r["addr"],
                 "label": "match" if r["label"] == 1 else "no match", "paired": False}
        d["expected_s1"] = s1_id
        rows.append(d)
    hand.append({"s1": s1, "records": rows})

out = {"seed": 21, "insights": insights, "hand_cases": hand}
(HERE / "examples.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
print(f"examples.json written: {sum(len(g['rows']) for v in insights.values() for g in v.get('groups', []))} "
      f"insight examples, {sum(len(h['records']) for h in hand)} hand-checked records")
