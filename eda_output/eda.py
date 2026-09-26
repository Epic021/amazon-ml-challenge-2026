#!/usr/bin/env python3
"""EDA for the Amazon ML Challenge 2026 — Business Entity Resolution.

Processes each large TSV one at a time to stay within memory, computes
distribution / anomaly / outlier stats, writes charts to charts/ and a
machine-readable summary to summary.json.
"""
import gc
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path("/home/darkslayer/Dev_Mode/Python/amazon_challenge/dataset/student_resource/dataset")
OUT = Path("/home/darkslayer/Dev_Mode/Python/amazon_challenge/eda_output")
CHARTS = OUT / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
PAL = {"S1": "#4C72B0", "S2": "#DD8452", "S3": "#55A868",
       "US": "#4C72B0", "India": "#DD8452", "France": "#C44E52"}

SUMMARY = {}

# ---- helpers ---------------------------------------------------------------
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
NONLATIN = re.compile(r"[^\x00-\x7F]")
DOTCOM = re.compile(r"\.(com|net|org|in|co)\b", re.I)
LEGAL = re.compile(r"\b(inc|corp|corporation|llc|ltd|limited|pvt|private|co|company|"
                   r"llp|plc|gmbh|group|enterprises|industries)\b", re.I)
NOISE_LEAD = re.compile(r"^\s*[-\.\*#/]{2,}")  # leading junk like "--", ".."

def script_of(s):
    if not s:
        return "empty"
    if DEVANAGARI.search(s):
        return "devanagari"
    if NONLATIN.search(s):
        return "other_nonlatin"
    return "latin"

def read_source(path):
    """Read a source file with only the columns we need, as strings."""
    df = pd.read_csv(path, sep="\t", dtype=str,
                     usecols=["entity_id", "business_name", "business_address", "country"],
                     na_filter=False, engine="c")
    return df

def pct(n, d):
    return round(100.0 * n / d, 3) if d else 0.0

# ---- per-source analysis ---------------------------------------------------
def analyze_source(tag, path, collect_samples=False):
    print(f"\n=== {tag}: {path.name} ===", flush=True)
    df = read_source(path)
    n = len(df)
    rec = {"file": path.name, "n_records": n}

    # id prefix sanity
    prefixes = df["entity_id"].str.slice(0, 3).value_counts().to_dict()
    rec["id_prefixes"] = prefixes

    # country distribution
    rec["country_counts"] = df["country"].value_counts().to_dict()

    name = df["business_name"]
    addr = df["business_address"]

    # emptiness / missing (na_filter=False so empties are "")
    empty_name = (name.str.strip() == "").sum()
    empty_addr = (addr.str.strip() == "").sum()
    rec["empty_name"] = int(empty_name); rec["empty_name_pct"] = pct(empty_name, n)
    rec["empty_addr"] = int(empty_addr); rec["empty_addr_pct"] = pct(empty_addr, n)

    # lengths
    nlen = name.str.len().to_numpy()
    alen = addr.str.len().to_numpy()
    def stats(a):
        a = a[~np.isnan(a)] if a.dtype.kind == "f" else a
        return {"min": int(a.min()), "p25": int(np.percentile(a, 25)),
                "median": int(np.median(a)), "mean": round(float(a.mean()), 1),
                "p75": int(np.percentile(a, 75)), "p95": int(np.percentile(a, 95)),
                "p99": int(np.percentile(a, 99)), "max": int(a.max())}
    rec["name_len"] = stats(nlen)
    rec["addr_len"] = stats(alen)

    # duplicates
    dup_id = df["entity_id"].duplicated().sum()
    nonempty_names = name[name.str.strip() != ""]
    dup_name = nonempty_names.duplicated().sum()
    rec["dup_entity_id"] = int(dup_id)
    rec["dup_business_name"] = int(dup_name)
    rec["dup_business_name_pct"] = pct(dup_name, len(nonempty_names))

    # script / language mix on a sample for speed
    samp = name.sample(min(200_000, n), random_state=0) if n else name
    scr = samp.map(script_of).value_counts()
    total_s = scr.sum()
    rec["name_script_pct"] = {k: pct(v, total_s) for k, v in scr.items()}

    # noise flags (vectorized regex) on full set
    rec["name_has_legal_suffix_pct"] = pct(name.str.contains(LEGAL, na=False).sum(), n)
    rec["name_is_domain_pct"] = pct(name.str.contains(DOTCOM, na=False).sum(), n)
    rec["name_leading_junk_pct"] = pct(name.str.contains(NOISE_LEAD, na=False).sum(), n)
    rec["addr_has_near_landmark_pct"] = pct(
        addr.str.contains(r"\b(near|opp|opposite|behind|beside)\b", case=False, na=False).sum(), n)
    rec["addr_has_digit_pct"] = pct(addr.str.contains(r"\d", na=False).sum(), n)
    # very short / anomalous names
    rec["name_len_le2_pct"] = pct((name.str.len() <= 2).sum(), n)
    rec["name_numeric_only"] = int(name.str.fullmatch(r"[\d\W]+").sum())

    samples = {}
    if collect_samples:
        def take(mask, k=6):
            sub = df[mask]
            if len(sub) == 0:
                return []
            return sub.sample(min(k, len(sub)), random_state=1)[
                ["entity_id", "business_name", "business_address", "country"]
            ].to_dict("records")
        samples["devanagari_name"] = take(name.str.contains(DEVANAGARI, na=False))
        samples["domain_name"] = take(name.str.contains(DOTCOM, na=False))
        samples["leading_junk"] = take(name.str.contains(NOISE_LEAD, na=False))
        samples["empty_address"] = take(addr.str.strip() == "")
        samples["near_landmark_addr"] = take(addr.str.contains(r"\bnear\b", case=False, na=False))
        samples["long_name"] = take(name.str.len() > rec["name_len"]["p99"])
        samples["typical"] = take(pd.Series(True, index=df.index) & (name.str.len().between(5, 40)))

    # keep small arrays for plotting, then free df
    keep = {
        "nlen_hist": np.histogram(nlen, bins=np.arange(0, 121, 4))[0].tolist(),
        "alen_hist": np.histogram(alen, bins=np.arange(0, 201, 8))[0].tolist(),
        "country_counts": rec["country_counts"],
    }
    del df, name, addr, nlen, alen, samp
    gc.collect()
    return rec, keep, samples

# ---- ground truth analysis -------------------------------------------------
def analyze_gt(path):
    print(f"\n=== ground truth: {path.name} ===", flush=True)
    gt = pd.read_csv(path, sep="\t", dtype=str, na_filter=False, engine="c")
    n = len(gt)
    matched = gt["matched_entity_ids"]
    # split counts
    def count_ids(s):
        s = s.strip()
        return 0 if s == "" else s.count(",") + 1
    counts = matched.map(count_ids).to_numpy()
    singleton = int((counts == 0).sum())
    rec = {
        "n_source1_entities": n,
        "singletons": singleton,
        "singleton_pct": pct(singleton, n),
        "with_matches": int((counts > 0).sum()),
        "match_count_stats": {
            "mean": round(float(counts.mean()), 3),
            "median": int(np.median(counts)),
            "p95": int(np.percentile(counts, 95)),
            "p99": int(np.percentile(counts, 99)),
            "max": int(counts.max()),
        },
    }
    # S2 vs S3 composition among matched ids
    joined = ",".join(matched[counts > 0].tolist())
    s2 = joined.count("S2-")
    s3 = joined.count("S3-")
    tot = s2 + s3
    rec["matched_id_source_mix"] = {"S2": s2, "S3": s3,
                                    "S2_pct": pct(s2, tot), "S3_pct": pct(s3, tot),
                                    "total_matched_ids": tot}
    # per-entity: has S2? has S3? both?
    has_s2 = matched.str.contains("S2-")
    has_s3 = matched.str.contains("S3-")
    rec["entities_with_S2"] = int(has_s2.sum())
    rec["entities_with_S3"] = int(has_s3.sum())
    rec["entities_with_both"] = int((has_s2 & has_s3).sum())
    rec["entities_S2_only"] = int((has_s2 & ~has_s3).sum())
    rec["entities_S3_only"] = int((~has_s2 & has_s3).sum())

    hist = np.histogram(counts, bins=np.arange(0, 16))[0].tolist()
    del gt, matched
    gc.collect()
    return rec, counts, hist

# ============================ RUN ==========================================
print("Loading & analyzing sources...", flush=True)

results = {"train": {}, "test": {}}
plotdata = {}
allsamples = {}

for tag, split, fname, samp in [
    ("train_S1", "train", "train_source1.tsv", True),
    ("train_S2", "train", "train_source2.tsv", True),
    ("train_S3", "train", "train_source3.tsv", True),
    ("test_S1", "test", "test_source1.tsv", False),
    ("test_S2", "test", "test_source2.tsv", False),
    ("test_S3", "test", "test_source3.tsv", False),
]:
    rec, keep, samples = analyze_source(tag, BASE / split / fname, collect_samples=samp)
    results[split][tag] = rec
    plotdata[tag] = keep
    if samples:
        allsamples[tag] = samples

gt_rec, gt_counts, gt_hist = analyze_gt(BASE / "train" / "train_ground_truth.tsv")
results["train"]["ground_truth"] = gt_rec

SUMMARY = results
(OUT / "summary.json").write_text(json.dumps(SUMMARY, indent=2, ensure_ascii=False))
(OUT / "samples.json").write_text(json.dumps(allsamples, indent=2, ensure_ascii=False))
print("\nSaved summary.json and samples.json", flush=True)

# ============================ CHARTS =======================================
print("Rendering charts...", flush=True)

# 1. Records per source (train vs test)
fig, ax = plt.subplots(figsize=(8, 4.5))
srcs = ["S1", "S2", "S3"]
train_n = [results["train"][f"train_{s}"]["n_records"] for s in srcs]
test_n = [results["test"][f"test_{s}"]["n_records"] for s in srcs]
x = np.arange(3); w = 0.38
b1 = ax.bar(x - w/2, train_n, w, label="Train", color="#4C72B0")
b2 = ax.bar(x + w/2, test_n, w, label="Test", color="#DD8452")
ax.set_xticks(x); ax.set_xticklabels(["Source 1\n(reference)", "Source 2", "Source 3"])
ax.set_ylabel("Records"); ax.set_title("Record counts per source (Train vs Test)")
ax.legend()
for b in list(b1) + list(b2):
    ax.annotate(f"{b.get_height()/1e6:.2f}M", (b.get_x()+b.get_width()/2, b.get_height()),
                ha="center", va="bottom", fontsize=8)
ax.yaxis.set_major_formatter(lambda v, _: f"{v/1e6:.0f}M")
fig.tight_layout(); fig.savefig(CHARTS / "01_records_per_source.png"); plt.close(fig)

# 2. Country distribution per source (train) + test S1 to show France
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
# train stacked
countries = ["US", "India", "France"]
srcnames = ["train_S1", "train_S2", "train_S3"]
bottom = np.zeros(3)
for c in countries:
    vals = [results["train"][s]["country_counts"].get(c, 0) for s in srcnames]
    axes[0].bar(["S1", "S2", "S3"], vals, bottom=bottom, label=c, color=PAL.get(c))
    bottom += np.array(vals, dtype=float)
axes[0].set_title("TRAIN — country composition per source")
axes[0].set_ylabel("Records"); axes[0].legend()
axes[0].yaxis.set_major_formatter(lambda v, _: f"{v/1e6:.1f}M")
# test stacked
bottom = np.zeros(3)
srcnames_t = ["test_S1", "test_S2", "test_S3"]
for c in countries:
    vals = [results["test"][s]["country_counts"].get(c, 0) for s in srcnames_t]
    axes[1].bar(["S1", "S2", "S3"], vals, bottom=bottom, label=c, color=PAL.get(c))
    bottom += np.array(vals, dtype=float)
axes[1].set_title("TEST — country composition (note: France is new!)")
axes[1].set_ylabel("Records"); axes[1].legend()
axes[1].yaxis.set_major_formatter(lambda v, _: f"{v/1e6:.1f}M")
fig.tight_layout(); fig.savefig(CHARTS / "02_country_distribution.png"); plt.close(fig)

# 3. Business name length distribution (train sources)
fig, ax = plt.subplots(figsize=(9, 4.8))
edges = np.arange(0, 121, 4)
centers = (edges[:-1] + edges[1:]) / 2
for s in srcs:
    h = np.array(plotdata[f"train_{s}"]["nlen_hist"], dtype=float)
    h = h / h.sum()
    ax.plot(centers, h, label=f"Source {s[-1]}", color=PAL[s], lw=2)
ax.set_xlabel("Business name length (chars)"); ax.set_ylabel("Density")
ax.set_title("Business-name length distribution (train)")
ax.legend()
fig.tight_layout(); fig.savefig(CHARTS / "03_name_length_dist.png"); plt.close(fig)

# 4. Address length distribution + empty-address rate
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
edges = np.arange(0, 201, 8); centers = (edges[:-1] + edges[1:]) / 2
for s in srcs:
    h = np.array(plotdata[f"train_{s}"]["alen_hist"], dtype=float)
    h = h / h.sum()
    axes[0].plot(centers, h, label=f"Source {s[-1]}", color=PAL[s], lw=2)
axes[0].set_xlabel("Address length (chars)"); axes[0].set_ylabel("Density")
axes[0].set_title("Address length distribution (train)"); axes[0].legend()
# empty rates
labels = ["S1", "S2", "S3"]
en = [results["train"][f"train_{s}"]["empty_name_pct"] for s in srcs]
ea = [results["train"][f"train_{s}"]["empty_addr_pct"] for s in srcs]
x = np.arange(3); w = 0.38
axes[1].bar(x - w/2, en, w, label="empty name %", color="#C44E52")
axes[1].bar(x + w/2, ea, w, label="empty address %", color="#8172B3")
axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
axes[1].set_ylabel("% of records"); axes[1].set_title("Missing name / address rate (train)")
axes[1].legend()
for i, v in enumerate(en): axes[1].annotate(f"{v:.1f}", (i-w/2, v), ha="center", va="bottom", fontsize=8)
for i, v in enumerate(ea): axes[1].annotate(f"{v:.1f}", (i+w/2, v), ha="center", va="bottom", fontsize=8)
fig.tight_layout(); fig.savefig(CHARTS / "04_address_len_and_missing.png"); plt.close(fig)

# 5. Match-count distribution per S1 entity (ground truth)
fig, ax = plt.subplots(figsize=(9, 4.8))
vals, cnts = np.unique(np.clip(gt_counts, 0, 12), return_counts=True)
bars = ax.bar(vals, cnts, color="#4C72B0")
ax.set_xlabel("# matched records per Source-1 entity (12 = 12+)")
ax.set_ylabel("Number of S1 entities")
ax.set_title("Match-count distribution per Source-1 entity (train ground truth)")
ax.yaxis.set_major_formatter(lambda v, _: f"{v/1e3:.0f}K")
for b in bars:
    ax.annotate(f"{b.get_height()/1e3:.0f}K", (b.get_x()+b.get_width()/2, b.get_height()),
                ha="center", va="bottom", fontsize=7)
fig.tight_layout(); fig.savefig(CHARTS / "05_match_count_dist.png"); plt.close(fig)

# 6. Match composition: singleton vs matched, S2/S3 breakdown
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
g = results["train"]["ground_truth"]
axes[0].pie([g["singletons"], g["with_matches"]],
            labels=[f"Singletons\n{g['singleton_pct']:.1f}%",
                    f"Has ≥1 match\n{100-g['singleton_pct']:.1f}%"],
            colors=["#C44E52", "#55A868"], autopct=lambda p: f"{p:.1f}%", startangle=90)
axes[0].set_title("Source-1 entities: singleton vs matched (train)")
# entity match-source composition
comp = [g["entities_S2_only"], g["entities_S3_only"], g["entities_with_both"]]
axes[1].bar(["S2 only", "S3 only", "S2 & S3"], comp,
            color=["#DD8452", "#55A868", "#4C72B0"])
axes[1].set_title("Among matched entities: which sources they link to")
axes[1].set_ylabel("Number of S1 entities")
axes[1].yaxis.set_major_formatter(lambda v, _: f"{v/1e3:.0f}K")
for i, v in enumerate(comp):
    axes[1].annotate(f"{v/1e3:.0f}K", (i, v), ha="center", va="bottom", fontsize=8)
fig.tight_layout(); fig.savefig(CHARTS / "06_match_composition.png"); plt.close(fig)

# 7. Name script mix per source (Latin / Devanagari / other)
fig, ax = plt.subplots(figsize=(9, 4.8))
kinds = ["latin", "devanagari", "other_nonlatin", "empty"]
colors = {"latin": "#4C72B0", "devanagari": "#DD8452",
          "other_nonlatin": "#C44E52", "empty": "#999999"}
bottom = np.zeros(3)
for k in kinds:
    vals = [results["train"][f"train_{s}"]["name_script_pct"].get(k, 0) for s in srcs]
    ax.bar(labels, vals, bottom=bottom, label=k, color=colors[k])
    bottom += np.array(vals, dtype=float)
ax.set_ylabel("% of names (sampled)"); ax.set_title("Business-name script mix per source (train)")
ax.legend()
fig.tight_layout(); fig.savefig(CHARTS / "07_name_script_mix.png"); plt.close(fig)

# 8. Noise-pattern prevalence per source
fig, ax = plt.subplots(figsize=(10, 5))
metrics = [
    ("name_has_legal_suffix_pct", "Legal suffix in name"),
    ("name_is_domain_pct", "Name is a web domain"),
    ("name_leading_junk_pct", "Leading junk (--, ..)"),
    ("addr_has_near_landmark_pct", "Landmark-based address"),
    ("addr_has_digit_pct", "Address has digits"),
]
x = np.arange(len(metrics)); w = 0.26
for i, s in enumerate(srcs):
    vals = [results["train"][f"train_{s}"][m] for m, _ in metrics]
    ax.bar(x + (i-1)*w, vals, w, label=f"Source {s[-1]}", color=PAL[s])
ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in metrics], rotation=20, ha="right")
ax.set_ylabel("% of records"); ax.set_title("Noise-pattern prevalence per source (train)")
ax.legend()
fig.tight_layout(); fig.savefig(CHARTS / "08_noise_patterns.png"); plt.close(fig)

print("Charts written to", CHARTS, flush=True)
print("\nDONE.")
