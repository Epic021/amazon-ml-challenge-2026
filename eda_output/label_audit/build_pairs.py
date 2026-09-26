#!/usr/bin/env python3
"""Label audit, stage 1: pair every train S2/S3 record with an S1 record and describe how
the two differ.

  matched record   -> its ground-truth S1                                   (label 1)
  unmatched record -> the most similar S1 among those sharing a blocking key (label 0)
                      keys: same address words; same core name; and 4-letter-prefix versions
                      of both (so single typos still meet)

A matched pair is "comparable" when it shares at least one of those keys with its S1, i.e.
when the same rule could have paired it. Only comparable pairs enter the pattern and model
analyses, so both labels are measured the same way.

Writes eda_output/label_audit/work/:
  pairs.parquet    one row per pair: difference pattern, similarity features, added/dropped
                   words, raw text
  records.parquet  one row per S2/S3 record: raw and normalised fingerprints, label, assigned
                   S1, S1 records it is an exact (normalised) copy of
  s1.parquet       one row per S1 record: fingerprints, ground-truth match count
Run:  python3 eda_output/label_audit/build_pairs.py      (needs: pip install anyascii)
"""
import os
import re
import sys
import time
import unicodedata
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

try:
    from anyascii import anyascii
except ImportError:
    sys.exit("anyascii is required: pip install anyascii")

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "dataset" / "student_resource" / "dataset" / "train"
WORK = Path(os.environ.get("AUDIT_WORK", Path(__file__).resolve().parent / "work"))
LIMIT_CHUNKS = int(os.environ.get("AUDIT_LIMIT_CHUNKS", "0"))  # >0: smoke test on the first chunks only
WORK.mkdir(parents=True, exist_ok=True)
MAX_BLOCK = 30  # skip blocking keys shared by more S1 records than this (generic names/streets)

# ---------------------------------------------------------------- vocabulary for normalisation
US_STATES = {"al":"alabama","ak":"alaska","az":"arizona","ar":"arkansas","ca":"california","co":"colorado",
 "ct":"connecticut","de":"delaware","fl":"florida","ga":"georgia","hi":"hawaii","id":"idaho","il":"illinois",
 "in":"indiana","ia":"iowa","ks":"kansas","ky":"kentucky","la":"louisiana","me":"maine","md":"maryland",
 "ma":"massachusetts","mi":"michigan","mn":"minnesota","ms":"mississippi","mo":"missouri","mt":"montana",
 "ne":"nebraska","nv":"nevada","nh":"new hampshire","nj":"new jersey","nm":"new mexico","ny":"new york",
 "nc":"north carolina","nd":"north dakota","oh":"ohio","ok":"oklahoma","or":"oregon","pa":"pennsylvania",
 "ri":"rhode island","sc":"south carolina","sd":"south dakota","tn":"tennessee","tx":"texas","ut":"utah",
 "vt":"vermont","va":"virginia","wa":"washington","wv":"west virginia","wi":"wisconsin","wy":"wyoming",
 "dc":"district of columbia"}
IN_STATES = {"an":"andaman and nicobar islands","ap":"andhra pradesh","ar":"arunachal pradesh","as":"assam",
 "br":"bihar","ch":"chandigarh","cg":"chhattisgarh","ct":"chhattisgarh","dn":"dadra and nagar haveli",
 "dd":"daman and diu","dl":"delhi","ga":"goa","gj":"gujarat","hr":"haryana","hp":"himachal pradesh",
 "jk":"jammu and kashmir","jh":"jharkhand","ka":"karnataka","kl":"kerala","la":"ladakh","ld":"lakshadweep",
 "mp":"madhya pradesh","mh":"maharashtra","mn":"manipur","ml":"meghalaya","mz":"mizoram","nl":"nagaland",
 "od":"odisha","or":"odisha","py":"puducherry","pb":"punjab","rj":"rajasthan","sk":"sikkim","tn":"tamil nadu",
 "tg":"telangana","ts":"telangana","tr":"tripura","up":"uttar pradesh","uk":"uttarakhand","ut":"uttarakhand",
 "wb":"west bengal"}
EXTRA_STATE_NAMES = ["orissa", "pondicherry", "uttaranchal", "jammu kashmir", "nct of delhi", "india",
                     "united states", "usa"]
STATE_CODES = set(US_STATES) | set(IN_STATES)
_phr = sorted(set(US_STATES.values()) | set(IN_STATES.values()) | set(EXTRA_STATE_NAMES), key=len, reverse=True)
STATE_PHRASES = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in _phr) + r")\b")
ABBR = {"rd":"road","st":"street","str":"street","ave":"avenue","av":"avenue","blvd":"boulevard","dr":"drive",
        "ln":"lane","ct":"court","pl":"place","hwy":"highway","pkwy":"parkway","cir":"circle","n":"north",
        "s":"south","e":"east","w":"west","ne":"northeast","nw":"northwest","se":"southeast","sw":"southwest",
        "twp":"township","hts":"heights","sq":"square","trl":"trail","ter":"terrace","expy":"expressway",
        "mt":"mount","ft":"fort","nr":"near","opp":"opposite","sec":"sector","ph":"phase","ind":"industrial",
        "indl":"industrial","estt":"estate","mkt":"market","ngr":"nagar","cplx":"complex","soc":"society",
        "r":"rue","bd":"boulevard","imp":"impasse","chem":"chemin","all":"allee"}
DROP = {"unit","apt","apartment","apartments","suite","ste","fl","floor","flr","room","rm","bldg","building",
        "po","box","pmb","no","nos","number","h","hno","house","plot","flat","shop","office","ofc","door",
        "null","na","nan","none","n","a"}
# ordinal words -> digit form, so "Fifth Street" and "5th Street" compare equal but "6th Street" does not
ORD_WORDS = {w: f"{i}{'st' if i % 10 == 1 and i != 11 else 'nd' if i % 10 == 2 and i != 12 else 'rd' if i % 10 == 3 and i != 13 else 'th'}"
             for i, w in enumerate(["first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth",
                                    "eleventh","twelfth","thirteenth","fourteenth","fifteenth","sixteenth","seventeenth",
                                    "eighteenth","nineteenth","twentieth"], start=1)}
ORD_RE = re.compile(r"^0*(\d+)(st|nd|rd|th)$")
LEAD_INT = re.compile(r"\d+")
# words that do not name the business: legal forms, honorifics, fillers
MINOR = {"inc","incorporated","llc","llp","lp","ltd","limited","corp","corporation","co","company","plc",
         "pllc","pc","pvt","private","opc","sarl","sas","sasu","eurl","sci","sa","ei","snc","gmbh",
         "smt","sri","shri","shree","mr","mrs","ms","dr","the","and","of","de","du","des","la","le","les","et"}
INDIC = re.compile(r"[ऀ-෿]+")
TOK = re.compile(r"[a-z0-9]+")

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


def romanize(s):
    s = unicodedata.normalize("NFKC", s or "")
    return (s if s.isascii() else anyascii(s)).lower()


def name_feats(name):
    """(sorted tuple of name words, keeping repeats, has Indic script)."""
    return tuple(sorted(TOK.findall(romanize(name)))), bool(INDIC.search(name or ""))


def addr_feats(addr):
    """(frozenset of address words without numbers, tuple of numbers), or None if empty."""
    if not addr or not addr.strip():
        return None
    s = INDIC.sub(" ", unicodedata.normalize("NFKC", addr))  # Indic words in addresses are state names
    s = STATE_PHRASES.sub(" ", re.sub(r"[^a-z0-9]+", " ", romanize(s)))
    words, nums = [], []
    for t in s.split():
        if t.isdigit():
            nums.append(t.lstrip("0") or "0")
        elif t in ORD_WORDS:
            words.append(ORD_WORDS[t])
        elif ORD_RE.match(t):                      # 5th, 1st, 59rd: part of a street name, not a house number
            m = ORD_RE.match(t)
            words.append(m.group(1) + m.group(2))
        elif any(c.isdigit() for c in t):          # 25a, 10199b, 005a: alphanumeric house / plot numbers
            nums.append(t.lstrip("0") or "0")
        elif t in STATE_CODES:
            continue
        else:
            t = ABBR.get(t, t)
            if t not in DROP:
                words.append(t)
    return frozenset(words), tuple(nums)


def lead_int(t):
    m = LEAD_INT.match(t)
    return int(m.group(0)[:15]) if m else None


def content(tokens):
    return frozenset(t for t in tokens if t not in MINOR and len(t) > 1)


def pair_typos(added, dropped):
    """Pair each added word with a dropped word 1-2 character edits away."""
    added, dropped = set(added), set(dropped)
    pairs = []
    for a in sorted(added, key=len, reverse=True):
        best, bd = None, 99
        for d in dropped:
            if min(len(a), len(d)) < 3:
                continue
            thr = 1 if max(len(a), len(d)) <= 4 else 2
            if abs(len(a) - len(d)) > thr:
                continue
            dist = Levenshtein.distance(a, d, score_cutoff=thr)
            if dist <= thr and dist < bd:
                best, bd = d, dist
        if best is not None:
            pairs.append((a, best))
            dropped.discard(best)
    for a, _ in pairs:
        added.discard(a)
    return pairs, added, dropped


# Sound-alike skeleton for comparing a romanised Indic name with a Latin one: "keyr" ~ "care",
# "laujistiks" ~ "logistics", "praivet" ~ "private". Hand-written rules, applied to both sides.
DIGRAPHS = (("ph", "f"), ("sh", "s"), ("ch", "c"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"),
            ("bh", "b"), ("ck", "k"))
SOUND = str.maketrans({"c": "k", "q": "k", "z": "s", "x": "s", "j": "g", "w": "v", "y": "a"})
VOWEL_RUN = re.compile(r"[aeiou]+")
REPEAT = re.compile(r"(.)\1+")


NASAL_M = re.compile(r"m(?=[bcdfgjklnpqrstvwxz])")   # anyascii writes the Hindi nasal mark as m


def skeleton(t):
    if t.isdigit():
        return t
    for a, b in DIGRAPHS:
        t = t.replace(a, b)
    t = NASAL_M.sub("n", t).translate(SOUND)
    sk = REPEAT.sub(r"\1", VOWEL_RUN.sub("", t))   # drop vowels, collapse doubled letters
    return sk or t[:1]


MINOR_SK = {skeleton(w) for w in MINOR}


def cross_content(toks):
    """Content words for a cross-script comparison: not a minor word, not one letter, and not
    sounding like a minor word (catches romanised forms such as "praivet" for private)."""
    return [t for t in toks if t not in MINOR and len(t) > 1 and skeleton(t) not in MINOR_SK]


def name_compare_cross(S_toks, R_toks):
    """Name comparison when exactly one side is in an Indic script, on sound-alike skeletons."""
    s_c, r_c = cross_content(S_toks), cross_content(R_toks)
    s_map = {skeleton(t): t for t in s_c}
    r_map = {skeleton(t): t for t in r_c}
    S, R = Counter(skeleton(t) for t in s_c), Counter(skeleton(t) for t in r_c)
    s_minor = {skeleton(t) for t in S_toks if t not in s_c}
    r_minor = {skeleton(t) for t in R_toks if t not in r_c}
    cs, cr = set(S), set(R)
    union = cs | cr
    jac = len(cs & cr) / len(union) if union else 1.0
    minor = int(s_minor != r_minor)
    if S == R:
        return (11 if minor else 10), 0, 0, 0, minor, (), (), jac
    pairs, a, d = pair_typos(list((R - S).elements()), list((S - R).elements()))
    if not (cs & cr) and not pairs:
        code = 15
    elif not a and not d:
        code = 11 if minor else 10
    elif a and not d:
        code = 12
    elif d and not a:
        code = 13
    else:
        code = 14
    added = sorted(r_map.get(t, t) for t in a)
    dropped = sorted(s_map.get(t, t) for t in d)
    return code, len(a), len(d), len(pairs), minor, added, dropped, jac


def name_compare(S_toks, s_ind, R_toks, r_ind):
    if s_ind != r_ind:
        return name_compare_cross(S_toks, R_toks)
    S, R = Counter(S_toks), Counter(R_toks)          # word counts, so "Partners Partners" differs from "Partners"
    cs, cr = content(S), content(R)
    union = cs | cr
    jac = len(cs & cr) / len(union) if union else 1.0
    if S == R:
        return 0, 0, 0, 0, 0, (), (), jac
    pairs, a, d = pair_typos(list((R - S).elements()), list((S - R).elements()))
    ca, cd = content(a), content(d)
    minor = int(bool((a - ca) | (d - cd)))
    typo_shared = any(p[0] in cr and p[1] in cs for p in pairs)
    if cs and cr and not (cs & cr) and not typo_shared:
        code = 9
    elif not ca and not cd:
        code = 3 if (pairs and minor) else 2 if pairs else 1
    elif ca and not cd:
        code = 4 if len(ca) == 1 else 5
    elif cd and not ca:
        code = 6 if len(cd) == 1 else 7
    else:
        code = 8
    return code, len(ca), len(cd), len(pairs), minor, sorted(ca), sorted(cd), jac


def pair_rel(a, b):
    if a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a):
        return 3
    if Levenshtein.distance(a, b) == 1:
        return 4
    ia, ib = lead_int(a), lead_int(b)
    if ia is not None and ib is not None and abs(ia - ib) <= 50:
        return 5
    return 6


def num_compare(sn, rn):
    """Compare the address numbers as multisets (components come in any order).
    -> (relation code, log gap of the closest changed pair, its edit distance, n added, n dropped)"""
    nan = float("nan")
    if sorted(sn) == sorted(rn) or "".join(sn) == "".join(rn):   # reordered, or written 7-15 vs 715
        return 0, 0.0, 0.0, 0, 0
    cs, cr = Counter(sn), Counter(rn)
    dropped, added = list((cs - cr).elements()), list((cr - cs).elements())
    if not dropped:
        return 1, nan, nan, len(added), 0
    if not added:
        return 2, nan, nan, 0, len(dropped)
    best = None
    for a in dropped:                                          # classify the closest changed pair
        for b in added:
            key = (pair_rel(a, b), Levenshtein.distance(a, b), a, b)
            if best is None or key < best:
                best = key
    rel, lev, a, b = best
    ia, ib = lead_int(a), lead_int(b)
    gap = float(np.log1p(abs(ia - ib))) if ia is not None and ib is not None else nan
    return rel, gap, float(lev), len(added), len(dropped)


def addr_compare(sa, ra):
    """-> code, rel, n_added, n_dropped, n_typos, added, dropped, jaccard, gap, numlev, n_num_add, n_num_drop"""
    nan = float("nan")
    if ra is None:
        return 8, 7, 0, 0, 0, (), (), nan, nan, nan, 0, 0
    if sa is None:
        return 9, 7, 0, 0, 0, (), (), nan, nan, nan, 0, 0
    (sw, sn), (rw, rn) = sa, ra
    rel, logdiff, numlev, nadd, ndrop = num_compare(sn, rn)
    union = sw | rw
    jac = len(sw & rw) / len(union) if union else 1.0
    if sw == rw:
        return (0 if rel == 0 else 1), rel, 0, 0, 0, (), (), jac, logdiff, numlev, nadd, ndrop
    pairs, a, d = pair_typos(rw - sw, sw - rw)
    if not a and not d:
        code = 2 if rel == 0 else 3
    elif not (sw & rw) and not pairs:
        code = 7
    elif a and not d:
        code = 4
    elif d and not a:
        code = 5
    else:
        code = 6
    return code, rel, len(a), len(d), len(pairs), sorted(a), sorted(d), jac, logdiff, numlev, nadd, ndrop


N_KEYS = 6


def block_keys(ntoks, af):
    """The six blocking keys (hashes) of a record, or None where the text is too short:
    address words; core name words; both cut to 4 letters; address words of 4+ letters only
    (ignores stray letters like the C of "#C-242"); sound-alike skeletons of the core name words
    (meets a romanised Indic name with its Latin spelling)."""
    c = content(ntoks)
    w = af[0] if af else frozenset()
    joined_w, joined_c = "".join(sorted(w)), "".join(sorted(c))
    ka = hash(("a", w)) if len(joined_w) >= 6 else None
    kn = hash(("n", c)) if len(joined_c) >= 4 else None
    ka4 = hash(("a4", frozenset(x[:4] for x in w))) if len(joined_w) >= 6 else None
    kn4 = hash(("n4", frozenset(x[:4] for x in c))) if len(joined_c) >= 4 else None
    long_w = frozenset(x for x in w if len(x) >= 4)
    kal = hash(("al", long_w)) if len("".join(sorted(long_w))) >= 6 else None
    sk = frozenset(skeleton(t) for t in cross_content(ntoks))
    kns = hash(("ns", sk)) if len("".join(sorted(sk))) >= 4 else None
    return ka, kn, ka4, kn4, kal, kns


def madd(d, k, v):
    cur = d.get(k)
    if cur is None:
        d[k] = v
    elif type(cur) is list:
        cur.append(v)
    else:
        d[k] = [cur, v]


def mget(d, k):
    cur = d.get(k)
    if cur is None:
        return ()
    return cur if type(cur) is list else (cur,)


# ---------------------------------------------------------------- ground truth
t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:6.0f}s] {msg}", flush=True)

gt = pd.read_csv(TRAIN / "train_ground_truth.tsv", sep="\t", dtype=str, na_filter=False)
true_size = {}
rec = {2: (array("q"), array("q")), 3: (array("q"), array("q"))}
for s1id, ms in zip(gt["source1_entity_id"].values, gt["matched_entity_ids"].values):
    n1 = int(s1id[3:])
    ids = ms.split(",") if ms else []
    true_size[n1] = len(ids)
    for m in ids:
        a, b = rec[int(m[1])]
        a.append(int(m[3:])); b.append(n1)
REV = {}
for src, (a, b) in rec.items():
    a = np.frombuffer(a, dtype=np.int64); b = np.frombuffer(b, dtype=np.int64)
    o = np.argsort(a, kind="stable"); REV[src] = (a[o], b[o])
del gt, rec
log("ground truth loaded")

# ---------------------------------------------------------------- writers
PAIR_SCHEMA = pa.schema([
    ("country", pa.int8()), ("src", pa.int8()), ("label", pa.int8()), ("reach", pa.int8()),
    ("nd", pa.int8()), ("ad", pa.int8()), ("rel", pa.int8()),
    ("s1", pa.int64()), ("rec", pa.int64()),
    ("name_tsr", pa.float32()), ("name_ratio", pa.float32()), ("name_jac", pa.float32()),
    ("n_name_add", pa.int8()), ("n_name_drop", pa.int8()), ("n_name_typo", pa.int8()), ("name_minor", pa.int8()),
    ("name_len_s1", pa.int8()), ("name_len_rec", pa.int8()),
    ("addr_tsr", pa.float32()), ("addr_ratio", pa.float32()), ("addr_jac", pa.float32()),
    ("n_addr_add", pa.int8()), ("n_addr_drop", pa.int8()), ("n_addr_typo", pa.int8()),
    ("n_num_s1", pa.int8()), ("n_num_rec", pa.int8()), ("num_logdiff", pa.float32()), ("num_lev", pa.float32()),
    ("n_num_add", pa.int8()), ("n_num_drop", pa.int8()),
    ("name_added", pa.string()), ("name_dropped", pa.string()),
    ("addr_added", pa.string()), ("addr_dropped", pa.string()),
    ("s1_id", pa.string()), ("rec_id", pa.string()),
    ("s1_name", pa.string()), ("rec_name", pa.string()), ("s1_addr", pa.string()), ("rec_addr", pa.string()),
])
REC_SCHEMA = pa.schema([
    ("country", pa.int8()), ("src", pa.int8()), ("label", pa.int8()), ("rec", pa.int64()), ("true_s1", pa.int64()),
    ("sig_norm", pa.int64()), ("sig_raw", pa.int64()), ("n_ident_s1", pa.int16()), ("true_is_ident", pa.int8()),
    ("ident_s1", pa.int64()), ("paired", pa.int8()),
    ("rec_id", pa.string()), ("name", pa.string()), ("addr", pa.string()),
])
S1_SCHEMA = pa.schema([
    ("country", pa.int8()), ("s1", pa.int64()), ("sig_norm", pa.int64()), ("sig_raw", pa.int64()),
    ("true_size", pa.int16()), ("s1_id", pa.string()), ("name", pa.string()), ("addr", pa.string()),
])
pair_w = pq.ParquetWriter(WORK / "pairs.parquet", PAIR_SCHEMA, compression="zstd")
rec_w = pq.ParquetWriter(WORK / "records.parquet", REC_SCHEMA, compression="zstd")
s1_w = pq.ParquetWriter(WORK / "s1.parquet", S1_SCHEMA, compression="zstd")


def cols(schema):
    return {f.name: [] for f in schema}


def flush(writer, schema, buf):
    if buf[schema.names[0]]:
        writer.write_table(pa.table(buf, schema=schema))
    return cols(schema)


# ---------------------------------------------------------------- main pass
s1_all = pd.read_csv(TRAIN / "train_source1.tsv", sep="\t", dtype=str, na_filter=False)
NAN = float("nan")

for ci, C in enumerate(["US", "India"]):
    s1 = s1_all[s1_all["country"] == C]
    s1_ids, s1_names, s1_addrs = s1["entity_id"].values, s1["business_name"].values, s1["business_address"].values
    info = {}
    blocks = tuple({} for _ in range(N_KEYS))
    ident = {}
    sbuf = cols(S1_SCHEMA)
    for pos, (eid, name, addr) in enumerate(zip(s1_ids, s1_names, s1_addrs)):
        n = int(eid[3:])
        nf, ind = name_feats(name)
        af = addr_feats(addr)
        info[n] = (pos, nf, ind, af)
        for d, k in zip(blocks, block_keys(nf, af)):
            if k is not None:
                madd(d, k, n)
        sig = hash((nf, af))
        madd(ident, sig, n)
        for key, val in (("country", ci), ("s1", n), ("sig_norm", sig), ("sig_raw", hash((name, addr))),
                         ("true_size", true_size.get(n, 0)), ("s1_id", eid), ("name", name), ("addr", addr)):
            sbuf[key].append(val)
    flush(s1_w, S1_SCHEMA, sbuf)
    log(f"{C}: {len(info):,} S1 records indexed")

    stats = {"pairs": 0, "unmatched_no_lookalike": 0, "matched_not_comparable": 0}
    for fname, src in [("train_source2.tsv", 2), ("train_source3.tsv", 3)]:
        recs, s1s = REV[src]
        for k, ch in enumerate(pd.read_csv(TRAIN / fname, sep="\t", dtype=str, na_filter=False, chunksize=250_000)):
            if LIMIT_CHUNKS and k >= LIMIT_CHUNKS:
                break
            c = ch[ch["country"] == C]
            if not len(c):
                continue
            ids = c["entity_id"].str.slice(3).astype(np.int64).to_numpy()
            i = np.minimum(np.searchsorted(recs, ids), len(recs) - 1)
            true_s1 = np.where(recs[i] == ids, s1s[i], -1)
            pbuf, rbuf = cols(PAIR_SCHEMA), cols(REC_SCHEMA)
            for eid, rnum, name, addr, t1 in zip(c["entity_id"].values, ids, c["business_name"].values,
                                                c["business_address"].values, true_s1):
                t1 = int(t1)
                nf, ind = name_feats(name)
                af = addr_feats(addr)
                keys = block_keys(nf, af)
                sig = hash((nf, af))
                idents = mget(ident, sig)
                label = 1 if t1 >= 0 else 0
                partner, reach = None, 0
                if label:
                    s = info.get(t1)
                    if s is not None:
                        partner = t1
                        skeys = block_keys(s[1], s[3])
                        reach = sum(1 << j for j in range(N_KEYS) if keys[j] is not None and keys[j] == skeys[j])
                        if not reach:
                            stats["matched_not_comparable"] += 1
                else:
                    cand = set()
                    for d, k in zip(blocks, keys):
                        if k is not None:
                            g = mget(d, k)
                            if len(g) <= MAX_BLOCK:
                                cand.update(g)
                    if cand:
                        rn, ra = romanize(name), romanize(addr)
                        best, bsc = None, -1
                        for n in cand:
                            p = info[n][0]
                            sc = fuzz.token_set_ratio(rn, romanize(s1_names[p])) + fuzz.token_set_ratio(ra, romanize(s1_addrs[p]))
                            if sc > bsc:
                                best, bsc = n, sc
                        partner = best
                        skeys = block_keys(info[best][1], info[best][3])
                        reach = sum(1 << j for j in range(N_KEYS) if keys[j] is not None and keys[j] == skeys[j])
                    else:
                        stats["unmatched_no_lookalike"] += 1
                # record row
                for key, val in (("country", ci), ("src", src), ("label", label), ("rec", int(rnum)),
                                 ("true_s1", t1), ("sig_norm", sig), ("sig_raw", hash((name, addr))),
                                 ("n_ident_s1", min(len(idents), 32000)), ("true_is_ident", int(t1 in idents)),
                                 ("ident_s1", idents[0] if idents else -1), ("paired", int(partner is not None)),
                                 ("rec_id", eid), ("name", name), ("addr", addr)):
                    rbuf[key].append(val)
                if partner is None:
                    continue
                pos, snf, sind, saf = info[partner]
                sname, saddr = s1_names[pos], s1_addrs[pos]
                nd, nadd, ndrop, ntypo, nminor, nadded, ndropped, njac = name_compare(snf, sind, nf, ind)
                ad, rel, aadd, adrop, atypo, aadded, adropped, ajac, logdiff, numlev, nnadd, nndrop = addr_compare(saf, af)
                rs_name, rr_name, rs_addr, rr_addr = romanize(sname), romanize(name), romanize(saddr), romanize(addr)
                row = (("country", ci), ("src", src), ("label", label), ("reach", reach),
                       ("nd", nd), ("ad", ad), ("rel", rel), ("s1", partner), ("rec", int(rnum)),
                       ("name_tsr", fuzz.token_set_ratio(rs_name, rr_name)), ("name_ratio", fuzz.ratio(rs_name, rr_name)),
                       ("name_jac", njac), ("n_name_add", min(nadd, 100)), ("n_name_drop", min(ndrop, 100)),
                       ("n_name_typo", min(ntypo, 100)), ("name_minor", nminor),
                       ("name_len_s1", min(len(snf), 100)), ("name_len_rec", min(len(nf), 100)),
                       ("addr_tsr", fuzz.token_set_ratio(rs_addr, rr_addr) if af and saf else NAN),
                       ("addr_ratio", fuzz.ratio(rs_addr, rr_addr) if af and saf else NAN),
                       ("addr_jac", ajac), ("n_addr_add", min(aadd, 100)), ("n_addr_drop", min(adrop, 100)),
                       ("n_addr_typo", min(atypo, 100)),
                       ("n_num_s1", min(len(saf[1]), 100) if saf else 0), ("n_num_rec", min(len(af[1]), 100) if af else 0),
                       ("num_logdiff", logdiff), ("num_lev", numlev),
                       ("n_num_add", min(nnadd, 100)), ("n_num_drop", min(nndrop, 100)),
                       ("name_added", " ".join(nadded)), ("name_dropped", " ".join(ndropped)),
                       ("addr_added", " ".join(aadded)), ("addr_dropped", " ".join(adropped)),
                       ("s1_id", f"S1-{partner}"), ("rec_id", eid),
                       ("s1_name", sname), ("rec_name", name), ("s1_addr", saddr), ("rec_addr", addr))
                for key, val in row:
                    pbuf[key].append(val)
                stats["pairs"] += 1
            flush(pair_w, PAIR_SCHEMA, pbuf)
            flush(rec_w, REC_SCHEMA, rbuf)
            log(f"{C} {fname}: {stats}")
    del info, blocks, ident

pair_w.close(); rec_w.close(); s1_w.close()
log("done")
