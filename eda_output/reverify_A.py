#!/usr/bin/env python3
"""RE-VERIFY part A: script mix, true-pair similarity AFTER romanization,
house-number relationship (4-way), test shift. Addresses corrections 1, 2, 5."""
import re, gc, json, unicodedata, sys
from pathlib import Path
sys.path.insert(0, "/tmp/claude-1000/-home-darkslayer-Dev-Mode-Python-amazon-challenge/bbeb9d98-4f58-4951-8476-c902f17e9d2b/scratchpad/libs")
import numpy as np, pandas as pd
from anyascii import anyascii
from rapidfuzz import fuzz

BASE = Path("/home/darkslayer/Dev_Mode/Python/amazon_challenge/dataset/student_resource/dataset")
OUT = Path("/home/darkslayer/Dev_Mode/Python/amazon_challenge/eda_output")
out = {}

# ---------------- normalization (STRATEGY-style) ----------------
LEGAL = {"pvt","private","ltd","limited","llc","inc","incorporated","corp","corporation",
         "llp","co","company","and","the","sarl","sas","sasu","eurl","sci","sa","ei","plc",
         "pvtltd","group"}  # 'group' deliberately NOT removed later; see note
LEGAL_CORE = LEGAL - {"group"}
HON = {"smt","sri","shri","mr","mrs","ms","dr","m/s","ms."}

def romanize(s):
    if not s: return ""
    s = unicodedata.normalize("NFKC", s)
    if not s.isascii():
        s = anyascii(s)
    return s.lower()

def norm_txt(s):
    s = romanize(s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def name_core(s):
    toks = [t for t in norm_txt(s).split() if t not in LEGAL_CORE and t not in HON]
    return " ".join(toks)

def core_tokset(s):
    return frozenset(name_core(s).split())

NUM = re.compile(r"\d+")
def nums_of(addr):
    return [n.lstrip("0") or "0" for n in NUM.findall(romanize(addr))]
def first_num(addr):
    ns = nums_of(addr)
    return ns[0] if ns else None

# ================= PART A: script mix (correction 1) =================
INDIC = re.compile(r"[ऀ-ॿ஀-௿ఀ-౿ಀ-೿"
                   r"઀-૿ঀ-৿ഀ-ൿ਀-੿଀-୿]")
DEVA = re.compile(r"[ऀ-ॿ]")
partA = {}
for f, key in [("train_source1.tsv","S1"),("train_source2.tsv","S2"),("train_source3.tsv","S3")]:
    df = pd.read_csv(BASE/"train"/f, sep="\t", dtype=str, usecols=["business_name"], na_filter=False)
    nm = df["business_name"].sample(400_000, random_state=2)
    n = len(nm)
    is_ascii = nm.map(lambda s: s.isascii())
    has_indic = nm.str.contains(INDIC)
    has_deva  = nm.str.contains(DEVA)
    # accented Latin = non-ascii but NOT indic (and not other exotic scripts)
    accented_only = (~is_ascii) & (~has_indic)
    partA[key] = {
        "pure_ascii_pct":        round(100.0*is_ascii.mean(), 2),
        "nonlatin_indic_pct":    round(100.0*has_indic.mean(), 2),
        "  devanagari_pct":      round(100.0*has_deva.mean(), 2),
        "  other_indic_pct":     round(100.0*(has_indic & ~has_deva).mean(), 2),
        "accented_latin_only_pct": round(100.0*accented_only.mean(), 2),
        "any_nonascii_pct":      round(100.0*(~is_ascii).mean(), 2),
    }
    del df, nm; gc.collect()
out["A_script_mix_train"] = partA
print("PART A done:", json.dumps(partA, indent=2))

# ================= PART B: TRUE PAIR analysis, romanized (corrections 2,3) =======
gt = pd.read_csv(BASE/"train"/"train_ground_truth.tsv", sep="\t", dtype=str, na_filter=False)
nonsing = gt[gt["matched_entity_ids"].str.strip()!=""]
samp = nonsing.sample(50_000, random_state=1)
need = {r.source1_entity_id: r.matched_entity_ids.split(",") for r in samp.itertuples(index=False)}
needed = set()
for v in need.values(): needed.update(v)
del gt, nonsing, samp; gc.collect()

rec = {}
s1_ids = set(need.keys())
for chunk in pd.read_csv(BASE/"train"/"train_source1.tsv", sep="\t", dtype=str,
                         na_filter=False, chunksize=500_000):
    h = chunk[chunk["entity_id"].isin(s1_ids)]
    for r in h.itertuples(index=False):
        rec[r.entity_id] = (r.business_name, r.business_address, r.country)
for f in ["train_source2.tsv","train_source3.tsv"]:
    for chunk in pd.read_csv(BASE/"train"/f, sep="\t", dtype=str, na_filter=False, chunksize=500_000):
        h = chunk[chunk["entity_id"].isin(needed)]
        for r in h.itertuples(index=False):
            rec[r.entity_id] = (r.business_name, r.business_address, r.country)
    gc.collect()

pairs=0; tok_ident=0; both80=0; renamed=0; cross=0
hn_equal=hn_diff=hn_added=hn_dropped=hn_none=0
addr_empty=0
added_tok = {}; dropped_tok = {}
for s1, mids in need.items():
    if s1 not in rec: continue
    n1,a1,c1 = rec[s1]
    ct1 = core_tokset(n1); nc1 = name_core(n1); na1 = norm_txt(a1); h1 = first_num(a1)
    for mid in mids:
        if mid not in rec: continue
        n2,a2,c2 = rec[mid]
        pairs += 1
        if c1 != c2: cross += 1
        if not a2.strip(): addr_empty += 1
        ct2 = core_tokset(n2)
        if ct1 and ct1 == ct2: tok_ident += 1
        ns = fuzz.token_set_ratio(nc1, name_core(n2))
        as_ = fuzz.token_set_ratio(na1, norm_txt(a2))
        if ns >= 80 and as_ >= 80: both80 += 1
        if ns < 50: renamed += 1
        for t in (ct2 - ct1): added_tok[t] = added_tok.get(t,0)+1
        for t in (ct1 - ct2): dropped_tok[t] = dropped_tok.get(t,0)+1
        h2 = first_num(a2)
        if h1 is None and h2 is None: hn_none += 1
        elif h1 is not None and h2 is None: hn_dropped += 1
        elif h1 is None and h2 is not None: hn_added += 1
        elif h1 == h2: hn_equal += 1
        else: hn_diff += 1

P = lambda x: round(100.0*x/pairs,1)
out["B_true_pairs"] = {
    "n_pairs": pairs,
    "cross_country_pct": P(cross),
    "name_core_tokens_IDENTICAL_pct": P(tok_ident),
    "name_AND_addr_token_set_ratio>=80_pct": P(both80),
    "renamed_name_sim<50_pct": P(renamed),
    "candidate_addr_empty_pct": P(addr_empty),
    "house_number": {
        "equal_pct": P(hn_equal), "differs_pct": P(hn_diff),
        "candidate_dropped_pct": P(hn_dropped), "candidate_added_pct": P(hn_added),
        "neither_has_number_pct": P(hn_none),
        "equal_pct_of_both_present": round(100.0*hn_equal/(hn_equal+hn_diff),1) if (hn_equal+hn_diff) else None,
    },
}
out["B_top_added_tokens_in_TRUE_pairs"] = sorted(added_tok.items(), key=lambda x:-x[1])[:25]
out["B_top_dropped_tokens_in_TRUE_pairs"] = sorted(dropped_tok.items(), key=lambda x:-x[1])[:25]
print("PART B done:", json.dumps(out["B_true_pairs"], indent=2))
del rec; gc.collect()

# ================= PART D: test shift (correction 5) =================
out["D_shift"] = {
    "train_S2S3_per_S1": round((5034616+5285603)/2206821,3),
    "test_S2S3_per_S1":  round((4887273+5082316)/1732544,3),
    "train_matched_ids": 7638365,
    "train_S2S3_total": 5034616+5285603,
    "train_unmatched_S2S3_pct": round(100.0*(1-(7638365/(5034616+5285603))),1),
    "train_mean_matches_per_S1": round(7638365/2206821,3),
}
r = out["D_shift"]
for mr in (0.71, 0.74):
    r[f"test_implied_matches_per_S1_at_{int(mr*100)}pct_matchrate"] = round(
        (4887273+5082316)*mr/1732544, 2)
print("PART D:", json.dumps(out["D_shift"], indent=2))

(OUT/"reverify_A.json").write_text(json.dumps(out, indent=2))
print("\nsaved reverify_A.json")
