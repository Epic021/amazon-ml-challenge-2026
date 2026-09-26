#!/usr/bin/env python3
"""RE-VERIFY part C — the NEIGHBOUR-NEGATIVE mechanism (correction 3).

Tests the generator hypothesis directly: a decoy copies an S1 address, CHANGES the
house number, and ADDS a word. Method: strip all digits from the romanized address to
get an `addr_key`. If an unmatched S2/S3 record shares an addr_key with an S1 record,
its address is a near-copy. Then compare the house-number relationship and the added
name tokens for DECOYS vs TRUE pairs on the same footing.
"""
import re, gc, json, unicodedata, sys, math
from pathlib import Path
sys.path.insert(0, "/tmp/claude-1000/-home-darkslayer-Dev-Mode-Python-amazon-challenge/bbeb9d98-4f58-4951-8476-c902f17e9d2b/scratchpad/libs")
import numpy as np, pandas as pd
from anyascii import anyascii
from rapidfuzz import fuzz

BASE = Path("/home/darkslayer/Dev_Mode/Python/amazon_challenge/dataset/student_resource/dataset")
OUT = Path("/home/darkslayer/Dev_Mode/Python/amazon_challenge/eda_output")
out = {}

LEGAL_CORE = {"pvt","private","ltd","limited","llc","inc","incorporated","corp","corporation",
              "llp","co","company","and","the","sarl","sas","sasu","eurl","sci","sa","ei","plc"}
HON = {"smt","sri","shri","mr","mrs","ms","dr"}

def romanize(s):
    if not s: return ""
    s = unicodedata.normalize("NFKC", s)
    return (anyascii(s) if not s.isascii() else s).lower()
def norm_txt(s):
    s = re.sub(r"[^\w\s]", " ", romanize(s))
    return re.sub(r"\s+", " ", s).strip()
def core_toks(s):
    return frozenset(t for t in norm_txt(s).split() if t not in LEGAL_CORE and t not in HON)
NUM = re.compile(r"\d+")
def addr_key(a):
    """normalized address with ALL digits removed -> the 'street/locality signature'"""
    t = re.sub(r"\d+", " ", norm_txt(a))
    return re.sub(r"\s+", " ", t).strip()
def first_num(a):
    m = NUM.search(romanize(a))
    return (m.group(0).lstrip("0") or "0") if m else None

# ---------- matched-ID membership as sorted int arrays (memory-light) ----------
gt = pd.read_csv(BASE/"train"/"train_ground_truth.tsv", sep="\t", dtype=str, na_filter=False)
m2, m3 = [], []
for s in gt["matched_entity_ids"].values:
    if not s: continue
    for mid in s.split(","):
        (m2 if mid[1]=="2" else m3).append(int(mid[3:]))
M2 = np.sort(np.array(m2, dtype=np.int64)); M3 = np.sort(np.array(m3, dtype=np.int64))
del m2, m3; gc.collect()
print(f"matched: S2={len(M2)} S3={len(M3)}")

# true pairs (for the apples-to-apples contrast), sampled
nonsing = gt[gt["matched_entity_ids"].str.strip()!=""].sample(40_000, random_state=11)
true_map = {r.source1_entity_id: r.matched_entity_ids.split(",") for r in nonsing.itertuples(index=False)}
true_needed = set()
for v in true_map.values(): true_needed.update(v)
del gt, nonsing; gc.collect()

def in_matched(arr_ids, which):
    M = M2 if which==2 else M3
    idx = np.searchsorted(M, arr_ids)
    idx[idx>=len(M)] = len(M)-1
    return M[idx] == arr_ids

results = {}
added_decoy, added_true = {}, {}
GLOBAL = {"decoy_hits":0,"decoy_total":0,
          "d_eq":0,"d_diff":0,"d_add":0,"d_drop":0,
          "t_eq":0,"t_diff":0,"t_add":0,"t_drop":0,"true_keyhits":0}

for COUNTRY in ["US","India"]:
    print(f"\n===== {COUNTRY} =====")
    # ---- S1 index for this country ----
    s1 = pd.read_csv(BASE/"train"/"train_source1.tsv", sep="\t", dtype=str, na_filter=False)
    s1 = s1[s1["country"]==COUNTRY].reset_index(drop=True)
    s1["k"]  = s1["business_address"].map(addr_key)
    s1["hn"] = s1["business_address"].map(first_num)
    s1 = s1[s1["k"].str.len()>=8]                       # drop degenerate keys
    kc = s1["k"].value_counts()
    keep = set(kc[kc<=25].index)                        # drop hyper-generic keys
    s1 = s1[s1["k"].isin(keep)].reset_index(drop=True)
    s1_small = s1[["entity_id","business_name","k","hn"]]
    print(f"  S1 rows indexed: {len(s1_small)}")
    del s1, kc, keep; gc.collect()

    # ---- sample UNMATCHED S2/S3 for this country ----
    un_rows = []
    for f, which in [("train_source2.tsv",2),("train_source3.tsv",3)]:
        for chunk in pd.read_csv(BASE/"train"/f, sep="\t", dtype=str,
                                 na_filter=False, chunksize=400_000):
            c = chunk[chunk["country"]==COUNTRY]
            if not len(c): continue
            ids = c["entity_id"].str.slice(3).astype(np.int64).to_numpy()
            unmatched = ~in_matched(ids, which)
            c = c[unmatched]
            if len(c):
                un_rows.append(c.sample(min(len(c), 9000), random_state=3))
        gc.collect()
    un = pd.concat(un_rows, ignore_index=True).sample(frac=1, random_state=4).head(120_000)
    del un_rows; gc.collect()
    un["k"]  = un["business_address"].map(addr_key)
    un["hn"] = un["business_address"].map(first_num)
    un = un[un["k"].str.len()>=8].reset_index(drop=True)
    print(f"  unmatched sampled: {len(un)}")

    # ---- merge on addr_key: does an S1 share this de-numbered address? ----
    mg = un.merge(s1_small, on="k", suffixes=("_u","_s"))
    hit_ids = set(mg["entity_id_u"])
    denom = len(un)
    GLOBAL["decoy_total"] += denom; GLOBAL["decoy_hits"] += len(hit_ids)
    print(f"  unmatched with an S1 address-key hit: {len(hit_ids)} / {denom} = {100*len(hit_ids)/denom:.1f}%")

    # best S1 per unmatched record, by name similarity
    eq=diff=add=drop=0
    best = {}
    for r in mg.itertuples(index=False):
        s = fuzz.token_set_ratio(norm_txt(r.business_name_u), norm_txt(r.business_name_s))
        p = best.get(r.entity_id_u)
        if p is None or s > p[0]:
            best[r.entity_id_u] = (s, r.hn_u, r.hn_s, r.business_name_u, r.business_name_s)
    for s, hu, hs, nu, ns in best.values():
        if hu is None and hs is None: continue
        elif hs is not None and hu is None: drop += 1
        elif hs is None and hu is not None: add += 1
        elif hu == hs: eq += 1
        else: diff += 1
        for t in (core_toks(nu) - core_toks(ns)):
            added_decoy[t] = added_decoy.get(t,0)+1
    tot = eq+diff+add+drop
    GLOBAL["d_eq"]+=eq; GLOBAL["d_diff"]+=diff; GLOBAL["d_add"]+=add; GLOBAL["d_drop"]+=drop
    results[COUNTRY] = {
        "unmatched_sampled": denom,
        "pct_with_S1_addresskey_hit": round(100.0*len(hit_ids)/denom,1),
        "house_number_among_addresskey_decoys": {
            "differs_pct": round(100.0*diff/tot,1), "equal_pct": round(100.0*eq/tot,1),
            "decoy_added_number_pct": round(100.0*add/tot,1),
            "decoy_dropped_number_pct": round(100.0*drop/tot,1), "n": tot},
    }
    print("  ", json.dumps(results[COUNTRY]["house_number_among_addresskey_decoys"]))
    del mg, un, s1_small, best; gc.collect()

out["C_by_country"] = results
out["C_overall"] = {
    "pct_unmatched_with_S1_addresskey_hit": round(100.0*GLOBAL["decoy_hits"]/GLOBAL["decoy_total"],1),
    "house_number_among_addresskey_decoys": {
        "differs_pct": round(100.0*GLOBAL["d_diff"]/(GLOBAL["d_eq"]+GLOBAL["d_diff"]+GLOBAL["d_add"]+GLOBAL["d_drop"]),1),
        "equal_pct":   round(100.0*GLOBAL["d_eq"]/(GLOBAL["d_eq"]+GLOBAL["d_diff"]+GLOBAL["d_add"]+GLOBAL["d_drop"]),1),
    },
}

# ---------- TRUE pairs restricted to address-key hits (apples-to-apples) ----------
rec = {}
s1ids = set(true_map.keys())
for chunk in pd.read_csv(BASE/"train"/"train_source1.tsv", sep="\t", dtype=str,
                         na_filter=False, chunksize=500_000):
    h = chunk[chunk["entity_id"].isin(s1ids)]
    for r in h.itertuples(index=False): rec[r.entity_id]=(r.business_name,r.business_address)
for f in ["train_source2.tsv","train_source3.tsv"]:
    for chunk in pd.read_csv(BASE/"train"/f, sep="\t", dtype=str, na_filter=False, chunksize=500_000):
        h = chunk[chunk["entity_id"].isin(true_needed)]
        for r in h.itertuples(index=False): rec[r.entity_id]=(r.business_name,r.business_address)
    gc.collect()
teq=tdiff=tadd=tdrop=0; tkey=0; tpairs=0
for s1, mids in true_map.items():
    if s1 not in rec: continue
    n1,a1 = rec[s1]; k1 = addr_key(a1); h1 = first_num(a1); c1 = core_toks(n1)
    for mid in mids:
        if mid not in rec: continue
        n2,a2 = rec[mid]; tpairs += 1
        if addr_key(a2) != k1 or len(k1) < 8: continue
        tkey += 1
        h2 = first_num(a2)
        if h1 is None and h2 is None: pass
        elif h1 is not None and h2 is None: tdrop += 1
        elif h1 is None and h2 is not None: tadd += 1
        elif h1 == h2: teq += 1
        else: tdiff += 1
        for t in (core_toks(n2) - c1): added_true[t] = added_true.get(t,0)+1
tt = teq+tdiff+tadd+tdrop
out["C_true_pairs_with_same_addresskey"] = {
    "true_pairs_evaluated": tpairs, "with_matching_addr_key": tkey,
    "pct_of_true_pairs_sharing_addr_key": round(100.0*tkey/tpairs,1),
    "house_number": {"equal_pct": round(100.0*teq/tt,1), "differs_pct": round(100.0*tdiff/tt,1),
                     "dropped_pct": round(100.0*tdrop/tt,1), "added_pct": round(100.0*tadd/tt,1), "n": tt},
}

# ---------- added-token log-odds: decoy vs true ----------
Dt = sum(added_decoy.values()); Tt = sum(added_true.values())
rows = []
for t in set(added_decoy) | set(added_true):
    d, tr = added_decoy.get(t,0), added_true.get(t,0)
    if d + tr < 150: continue
    lo = math.log(((d+1)/(Dt+1)) / ((tr+1)/(Tt+1)))
    rows.append((t, d, tr, round(lo,2)))
rows.sort(key=lambda x:-x[3])
out["C_added_token_logodds_DECOY_positive"] = rows[:22]
out["C_added_token_logodds_TRUEMATCH_negative"] = rows[-22:][::-1]

(OUT/"reverify_C.json").write_text(json.dumps(out, indent=2))
print("\n"+json.dumps({k:v for k,v in out.items() if not k.startswith("C_added")}, indent=2))
print("\nDECOY-indicating added tokens:", rows[:15])
print("\nTRUE-MATCH-indicating added tokens:", rows[-15:][::-1])
