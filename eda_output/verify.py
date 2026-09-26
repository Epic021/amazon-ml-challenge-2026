#!/usr/bin/env python3
"""Verify the load-bearing claims in the review against the TRAIN data.
Joins ground truth to sources on a sample to characterize TRUE pairs."""
import re, gc, json, unicodedata, random
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path("/home/darkslayer/Dev_Mode/Python/amazon_challenge/dataset/student_resource/dataset")
random.seed(0)
out = {}

def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

LEGAL = {"pvt","private","ltd","limited","llc","inc","incorporated","corp","corporation",
         "llp","co","company","and","the","sarl","sas","sasu","eurl","sci","sa","ei","plc"}
def core_tokens(s):
    return frozenset(t for t in norm(s).split() if t not in LEGAL)

def first_num(addr):
    m = re.search(r"\d+", addr)
    return m.group(0).lstrip("0") or "0" if m else None

# ---------- FACT 1: each S2/S3 record matches at most one S1 (uniqueness) ----------
gt = pd.read_csv(BASE/"train"/"train_ground_truth.tsv", sep="\t", dtype=str, na_filter=False)
total_ids = 0
seen = set()
dups = 0
for s in gt["matched_entity_ids"].values:
    if not s:
        continue
    for mid in s.split(","):
        total_ids += 1
        if mid in seen:
            dups += 1
        else:
            seen.add(mid)
out["fact1_total_matched_ids"] = total_ids
out["fact1_distinct_matched_ids"] = len(seen)
out["fact1_duplicate_assignments"] = dups
out["fact1_each_record_at_most_one_S1"] = (dups == 0)
del seen; gc.collect()

# ---------- sample S1 entities for the GT-join analyses ----------
gt_nonsing = gt[gt["matched_entity_ids"].str.strip() != ""]
samp = gt_nonsing.sample(60000, random_state=1)
need = {}
for s1, ms in zip(samp["source1_entity_id"], samp["matched_entity_ids"]):
    need[s1] = ms.split(",")
needed_ids = set()
for v in need.values():
    needed_ids.update(v)
out["sample_S1"] = len(need)
out["sample_matched_ids"] = len(needed_ids)
del gt, gt_nonsing; gc.collect()

# ---------- load S1 sample records ----------
s1_ids = set(need.keys())
rec = {}
for chunk in pd.read_csv(BASE/"train"/"train_source1.tsv", sep="\t", dtype=str,
                         na_filter=False, chunksize=500_000):
    hit = chunk[chunk["entity_id"].isin(s1_ids)]
    for r in hit.itertuples(index=False):
        rec[r.entity_id] = (r.business_name, r.business_address, r.country)
gc.collect()

# ---------- stream S2/S3, keep needed matched records ----------
for f in ["train_source2.tsv","train_source3.tsv"]:
    for chunk in pd.read_csv(BASE/"train"/f, sep="\t", dtype=str,
                             na_filter=False, chunksize=500_000):
        hit = chunk[chunk["entity_id"].isin(needed_ids)]
        for r in hit.itertuples(index=False):
            rec[r.entity_id] = (r.business_name, r.business_address, r.country)
    gc.collect()

# ---------- FACTS 2,5,6: iterate true pairs ----------
cross_country = 0; pairs = 0
name_tokens_ident = 0
both_sim80 = 0
hn_equal = hn_diff = hn_one_missing = hn_both_missing = 0
name_sim_lt50 = 0
for s1, mids in need.items():
    if s1 not in rec:
        continue
    n1, a1, c1 = rec[s1]
    ct1 = core_tokens(n1); nn1 = norm(n1); na1 = norm(a1); hn1 = first_num(a1)
    for mid in mids:
        if mid not in rec:
            continue
        n2, a2, c2 = rec[mid]
        pairs += 1
        if c1 != c2:
            cross_country += 1
        # name token identity (core tokens equal)
        if ct1 == core_tokens(n2) and ct1:
            name_tokens_ident += 1
        # similarity proxies (difflib ratio *100)
        nsim = SequenceMatcher(None, nn1, norm(n2)).ratio()*100
        asim = SequenceMatcher(None, na1, norm(a2)).ratio()*100
        if nsim >= 80 and asim >= 80:
            both_sim80 += 1
        if nsim < 50:
            name_sim_lt50 += 1
        # house number relationship
        hn2 = first_num(a2)
        if hn1 is None and hn2 is None:
            hn_both_missing += 1
        elif hn1 is None or hn2 is None:
            hn_one_missing += 1
        elif hn1 == hn2:
            hn_equal += 1
        else:
            hn_diff += 1

def p(x): return round(100.0*x/pairs, 1) if pairs else 0.0
out["fact2_true_pairs_evaluated"] = pairs
out["fact2_cross_country_pct"] = p(cross_country)
out["fact5_name_core_tokens_identical_pct"] = p(name_tokens_ident)
out["fact5_name_AND_addr_sim>=80_pct"] = p(both_sim80)
out["fact5_renamed_name_sim<50_pct"] = p(name_sim_lt50)
hn_both = hn_equal + hn_diff
out["fact6_house_number_equal_pct_of_both_present"] = round(100.0*hn_equal/hn_both,1) if hn_both else None
out["fact6_house_number_differ_pct_of_both_present"] = round(100.0*hn_diff/hn_both,1) if hn_both else None
out["fact6_one_missing_pct"] = p(hn_one_missing)
del rec; gc.collect()

# ---------- FACT 3: non-Latin script share (excluding accented Latin), S2 & S3 ----------
SCRIPTS = re.compile(r"[ऀ-ॿ஀-௿ఀ-౿ಀ-೿઀-૿ঀ-৿ഀ-ൿ਀-੿଀-୿]")  # Indic blocks
for f, key in [("train_source2.tsv","S2"),("train_source3.tsv","S3")]:
    df = pd.read_csv(BASE/"train"/f, sep="\t", dtype=str, usecols=["business_name"],
                     na_filter=False)
    nm = df["business_name"].sample(300000, random_state=2)
    out[f"fact3_{key}_nonlatin_script_pct"] = round(100.0*nm.str.contains(SCRIPTS).mean(),2)
    out[f"fact3_{key}_accented_or_nonascii_pct"] = round(100.0*nm.str.contains(r"[^\x00-\x7F]").mean(),2)
    del df, nm; gc.collect()

# ---------- FACT 4: postcode/PIN presence by country (S1) ----------
s1 = pd.read_csv(BASE/"train"/"train_source1.tsv", sep="\t", dtype=str, na_filter=False)
for cc in ["US","India"]:
    sub = s1[s1["country"]==cc]["business_address"]
    if cc == "US":
        has = sub.str.contains(r"\b\d{5}(?:-\d{4})?\b")
    else:
        has = sub.str.contains(r"\b\d{6}\b")
    out[f"fact4_{cc}_postcode_present_pct"] = round(100.0*has.mean(),2)
del s1; gc.collect()

# ---------- record-per-S1 ratio (test vs train) ----------
out["ratio_train_S2S3_per_S1"] = round((5034616+5285603)/2206821,2)
out["ratio_test_S2S3_per_S1"] = round((4887273+5082316)/1732544,2)

print(json.dumps(out, indent=2))
Path("/home/darkslayer/Dev_Mode/Python/amazon_challenge/eda_output/verify.json").write_text(json.dumps(out, indent=2))
