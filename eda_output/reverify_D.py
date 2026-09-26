#!/usr/bin/env python3
"""Sensitivity check: is the low 8.1% 'address near-copy' rate just key strictness?
Relax the address key in 4 levels and watch the hit-rate climb. US only."""
import re, gc, json, unicodedata, sys
from pathlib import Path
sys.path.insert(0, "/tmp/claude-1000/-home-darkslayer-Dev-Mode-Python-amazon-challenge/bbeb9d98-4f58-4951-8476-c902f17e9d2b/scratchpad/libs")
import numpy as np, pandas as pd
from anyascii import anyascii
from rapidfuzz import fuzz, process

BASE = Path("/home/darkslayer/Dev_Mode/Python/amazon_challenge/dataset/student_resource/dataset")
OUT  = Path("/home/darkslayer/Dev_Mode/Python/amazon_challenge/eda_output")

ABBR = {"rd":"road","st":"street","ave":"avenue","av":"avenue","blvd":"boulevard","dr":"drive",
        "ln":"lane","ct":"court","pl":"place","hwy":"highway","pkwy":"parkway","cir":"circle",
        "ste":"suite","apt":"apartment","bldg":"building","fl":"floor","rm":"room","n":"north",
        "s":"south","e":"east","w":"west","ne":"northeast","nw":"northwest","se":"southeast",
        "sw":"southwest","twp":"township","hts":"heights","jct":"junction","sq":"square",
        "trl":"trail","ter":"terrace","pkw":"parkway","expy":"expressway"}
STATES = {"al":"alabama","ak":"alaska","az":"arizona","ar":"arkansas","ca":"california",
 "co":"colorado","ct":"connecticut","de":"delaware","fl":"florida","ga":"georgia","hi":"hawaii",
 "id":"idaho","il":"illinois","in":"indiana","ia":"iowa","ks":"kansas","ky":"kentucky",
 "la":"louisiana","me":"maine","md":"maryland","ma":"massachusetts","mi":"michigan",
 "mn":"minnesota","ms":"mississippi","mo":"missouri","mt":"montana","ne":"nebraska",
 "nv":"nevada","nh":"new hampshire","nj":"new jersey","nm":"new mexico","ny":"new york",
 "nc":"north carolina","nd":"north dakota","oh":"ohio","ok":"oklahoma","or":"oregon",
 "pa":"pennsylvania","ri":"rhode island","sc":"south carolina","sd":"south dakota",
 "tn":"tennessee","tx":"texas","ut":"utah","vt":"vermont","va":"virginia","wa":"washington",
 "wv":"west virginia","wi":"wisconsin","wy":"wyoming"}

def romanize(s):
    if not s: return ""
    s = unicodedata.normalize("NFKC", s)
    return (anyascii(s) if not s.isascii() else s).lower()
def toks_nonum(a):
    t = re.sub(r"[^\w\s]", " ", romanize(a))
    t = re.sub(r"\d+", " ", t)
    return [x for x in t.split() if x]

def k1(a):  # L1 exact de-numbered string
    return " ".join(toks_nonum(a))
def k2(a):  # L2 order-invariant token set
    return " ".join(sorted(set(toks_nonum(a))))
def k3(a):  # L3 + abbreviation & state expansion
    out=[]
    for t in toks_nonum(a):
        t = ABBR.get(t,t); t = STATES.get(t,t)
        out.extend(t.split())
    return " ".join(sorted(set(out)))

# matched ids
gt = pd.read_csv(BASE/"train"/"train_ground_truth.tsv", sep="\t", dtype=str, na_filter=False)
m2,m3=[],[]
for s in gt["matched_entity_ids"].values:
    if not s: continue
    for mid in s.split(","): (m2 if mid[1]=="2" else m3).append(int(mid[3:]))
M2=np.sort(np.array(m2,dtype=np.int64)); M3=np.sort(np.array(m3,dtype=np.int64))
del gt,m2,m3; gc.collect()
def in_matched(ids, which):
    M = M2 if which==2 else M3
    i = np.searchsorted(M, ids); i[i>=len(M)]=len(M)-1
    return M[i]==ids

# S1 US
s1 = pd.read_csv(BASE/"train"/"train_source1.tsv", sep="\t", dtype=str, na_filter=False)
s1 = s1[s1["country"]=="US"].reset_index(drop=True)
print("S1 US:", len(s1))

# unmatched US sample
rows=[]
for f,w in [("train_source2.tsv",2),("train_source3.tsv",3)]:
    for ch in pd.read_csv(BASE/"train"/f, sep="\t", dtype=str, na_filter=False, chunksize=400_000):
        c = ch[ch["country"]=="US"]
        if not len(c): continue
        ids = c["entity_id"].str.slice(3).astype(np.int64).to_numpy()
        c = c[~in_matched(ids,w)]
        if len(c): rows.append(c.sample(min(len(c),4000), random_state=3))
    gc.collect()
un = pd.concat(rows, ignore_index=True).sample(n=40000, random_state=4).reset_index(drop=True)
del rows; gc.collect()
print("unmatched US sampled:", len(un))

res={}
for name, fn in [("L1_exact_denumbered", k1), ("L2_token_set", k2), ("L3_abbrev_state_expanded", k3)]:
    sk = set(s1["business_address"].map(fn))
    uk = un["business_address"].map(fn)
    hit = uk.map(lambda x: len(x)>=8 and x in sk)
    res[name] = round(100.0*hit.mean(),1)
    print(f"  {name}: {res[name]}%")
    del sk, uk, hit; gc.collect()

# L4: fuzzy >=90 against S1 addresses sharing >=1 rare-ish token block (approximate)
# block on the L3 key's rarest token to keep it cheap
from collections import defaultdict
s1["_k3"] = s1["business_address"].map(k3)
blocks = defaultdict(list)
for addr, k in zip(s1["business_address"].values, s1["_k3"].values):
    ts = k.split()
    if not ts: continue
    blocks[max(ts, key=len)].append(addr)   # longest token as a cheap block key
sub = un.sample(6000, random_state=9)
hits = 0; n = 0
for a in sub["business_address"].values:
    ts = k3(a).split()
    if not ts: continue
    n += 1
    cand = blocks.get(max(ts, key=len), [])
    if not cand: continue
    na = " ".join(sorted(set(toks_nonum(a))))
    best = process.extractOne(na, [" ".join(sorted(set(toks_nonum(c)))) for c in cand[:400]],
                              scorer=fuzz.token_set_ratio, score_cutoff=90)
    if best: hits += 1
res["L4_fuzzy_ge90_blocked"] = round(100.0*hits/n,1)
print(f"  L4_fuzzy_ge90_blocked: {res['L4_fuzzy_ge90_blocked']}%  (n={n})")

(OUT/"reverify_D.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
