#!/usr/bin/env python3
"""Supplementary checks: script breakdown, numeric-only names, France samples,
match integrity, and exact-name blocking signal."""
import json, re, gc
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path("/home/darkslayer/Dev_Mode/Python/amazon_challenge/dataset/student_resource/dataset")
OUT = Path("/home/darkslayer/Dev_Mode/Python/amazon_challenge/eda_output")

SCRIPTS = {
    "devanagari": r"[ऀ-ॿ]",
    "tamil": r"[஀-௿]",
    "telugu": r"[ఀ-౿]",
    "kannada": r"[ಀ-೿]",
    "gujarati": r"[઀-૿]",
    "bengali": r"[ঀ-৿]",
    "malayalam": r"[ഀ-ൿ]",
    "gurmukhi": r"[਀-੿]",
    "oriya": r"[଀-୿]",
    "latin_accented": r"[À-ɏ]",
    "cjk": r"[一-鿿]",
    "arabic": r"[؀-ۿ]",
}
SCRIPTS = {k: re.compile(v) for k, v in SCRIPTS.items()}

def read(path, split):
    return pd.read_csv(BASE / split / path, sep="\t", dtype=str,
                       usecols=["entity_id","business_name","business_address","country"],
                       na_filter=False, engine="c")

out = {}

# ---- script breakdown on a big sample of S2 train names ----
df = read("train_source2.tsv", "train")
name = df["business_name"]
samp = name.sample(500_000, random_state=0)
sb = {}
for k, rx in SCRIPTS.items():
    sb[k] = round(100.0 * samp.str.contains(rx).sum() / len(samp), 3)
out["S2_script_breakdown_pct"] = sb

# ---- numeric-only investigation ----
mask_num = name.str.fullmatch(r"[\d\W]+")
num_samples = df[mask_num].sample(15, random_state=3)["business_name"].tolist()
out["numeric_only_examples"] = num_samples
out["numeric_only_count_S2"] = int(mask_num.sum())
# how many are purely digits vs contain non-ascii-word punctuation
pure_digit = name.str.fullmatch(r"\d+").sum()
out["pure_digit_names_S2"] = int(pure_digit)
del df, name, samp, mask_num; gc.collect()

# ---- France samples from test ----
dt = read("test_source1.tsv", "test")
fr = dt[dt["country"] == "France"].sample(8, random_state=5)[
    ["business_name","business_address"]].to_dict("records")
out["france_test_S1_samples"] = fr
# France name script check
frn = dt[dt["country"]=="France"]["business_name"]
out["france_name_accented_pct"] = round(100.0*frn.str.contains(SCRIPTS["latin_accented"]).sum()/len(frn),2)
out["france_name_nonascii_pct"] = round(100.0*frn.str.contains(r"[^\x00-\x7F]").sum()/len(frn),2)
del dt, frn; gc.collect()

# ---- match integrity: do matched ids ever reference S1? do they exist? ----
gt = pd.read_csv(BASE/"train"/"train_ground_truth.tsv", sep="\t", dtype=str, na_filter=False)
# sample 200k rows to check prefixes referenced
samp_gt = gt.sample(200_000, random_state=7)
all_ids = ",".join(samp_gt["matched_entity_ids"][samp_gt["matched_entity_ids"].str.strip()!=""])
ids = [i for i in all_ids.split(",") if i]
pref = pd.Series([i[:3] for i in ids]).value_counts().to_dict()
out["matched_id_prefixes_sampled"] = pref
# does every S1 in source1 appear in GT exactly once?
s1 = read("train_source1.tsv","train")
s1_ids = set(s1["entity_id"])
gt_ids = gt["source1_entity_id"]
out["gt_rows"] = len(gt)
out["gt_unique_s1"] = int(gt_ids.nunique())
out["s1_records"] = len(s1_ids)
out["s1_in_gt_all_match"] = bool(set(gt_ids) == s1_ids)
del gt, samp_gt; gc.collect()

# ---- exact normalized-name blocking signal (train) ----
def norm(s):
    return (s.str.lower()
             .str.replace(r"[^\w\s]", " ", regex=True)
             .str.replace(r"\s+", " ", regex=True)
             .str.strip())
s1n = set(norm(s1["business_name"]).tolist())
del s1; gc.collect()
s2 = read("train_source2.tsv","train")
s2n = norm(s2["business_name"])
overlap2 = s2n.isin(s1n).mean()
del s2, s2n; gc.collect()
s3 = read("train_source3.tsv","train")
s3n = norm(s3["business_name"])
overlap3 = s3n.isin(s1n).mean()
del s3, s3n; gc.collect()
out["exact_normname_S2_in_S1_pct"] = round(100.0*float(overlap2),2)
out["exact_normname_S3_in_S1_pct"] = round(100.0*float(overlap3),2)

(OUT/"supplementary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps(out, indent=2, ensure_ascii=False))
