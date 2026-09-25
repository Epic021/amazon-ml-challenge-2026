import pandas as pd, numpy as np, re, csv, collections, unicodedata
import os
D = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "student_resource", "dataset"))
def load(p):
    return pd.read_csv(f"{D}/{p}", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE, engine="c")
def script_of(s):
    for ch in s:
        if ord(ch) > 0x24F and ch.isalpha():
            try: return unicodedata.name(ch).split()[0]
            except ValueError: return "?"
    return "LATIN"

gt = load("train/train_ground_truth.tsv")
s1 = load("train/train_source1.tsv"); s2 = load("train/train_source2.tsv"); s3 = load("train/train_source3.tsv")
lists = gt.matched_entity_ids.map(lambda x: x.split(",") if x else [])
print("S2", s2.shape, "unique", s2.entity_id.is_unique)
matched = set(i for l in lists for i in l)
print("S2 never matched", (~s2.entity_id.isin(matched)).mean(), "S3 never matched", (~s3.entity_id.isin(matched)).mean())
print("S2 countries", s2.country.value_counts().to_dict())
print("S2 addr empty", (s2.business_address == "").mean(), "name empty", (s2.business_name == "").mean())

rng = np.random.default_rng(1)
for nm, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    smp = df.sample(200000, random_state=0)
    print(f"--- {nm} name script", smp.business_name.map(script_of).value_counts(normalize=True).round(4).head(8).to_dict())
    print(f"    addr script", smp.business_address.map(script_of).value_counts(normalize=True).round(4).head(8).to_dict())
    print("    upper-case addr frac", (smp.business_address.str.upper() == smp.business_address).mean().round(3),
          "NULL/N/A tokens", smp.business_address.str.contains(r"\bNULL\b|\bN/A\b|\bnan\b", regex=True).mean().round(4))
    ind = smp[smp.country == "India"].business_address
    print("    India addr w/ 6-digit PIN", ind.str.contains(r"\b\d{6}\b|\b\d{3} \d{3}\b", regex=True).mean().round(3))
    us = smp[smp.country == "US"].business_address
    print("    US addr w/ 5-digit ZIP", us.str.contains(r"\b\d{5}\b", regex=True).mean().round(3))

# row-order / id leakage checks
pos1 = pd.Series(np.arange(len(s1)), index=s1.entity_id)
pos3 = pd.Series(np.arange(len(s3)), index=s3.entity_id)
posg = np.arange(len(gt))
pairs = [(g, i) for g, l in zip(gt.source1_entity_id.values[:200000], lists.values[:200000]) for i in l if i.startswith("S3-")]
a = pos1.loc[[p[0] for p in pairs]].values; b = pos3.loc[[p[1] for p in pairs]].values
print("corr S1 row pos vs matched S3 row pos", np.corrcoef(a, b)[0, 1])
print("corr GT row vs S1 row pos", np.corrcoef(pos1.loc[gt.source1_entity_id.values[:200000]].values, posg[:200000])[0, 1])
ida = np.array([int(p[0][3:]) for p in pairs]); idb = np.array([int(p[1][3:]) for p in pairs])
print("corr S1 id vs matched S3 id", np.corrcoef(ida, idb)[0, 1])
# within-list S3 row adjacency
adj = []
for l in lists.values[:100000]:
    p = sorted(pos3.loc[[i for i in l if i.startswith("S3-")]].values) if any(i.startswith("S3-") for i in l) else []
    if len(p) > 1: adj.append(np.median(np.diff(p)))
print("median row gap between S3 records of same entity", np.median(adj), "(random expect ~", len(s3) // 3, ")")

# pairwise stats on matched pairs: exact name / address equality after light normalization
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()
s1i = s1.set_index("entity_id"); s2i = s2.set_index("entity_id"); s3i = s3.set_index("entity_id")
samp = gt.sample(30000, random_state=2)
rows = []
for sid, ml in zip(samp.source1_entity_id, samp.matched_entity_ids):
    if not ml: continue
    r = s1i.loc[sid]
    n1 = set(norm(r.business_name)); a1 = set(norm(r.business_address))
    for m in ml.split(","):
        q = (s2i if m.startswith("S2") else s3i).loc[m]
        n2 = set(norm(q.business_name)); a2 = set(norm(q.business_address))
        jn = len(n1 & n2) / max(1, len(n1 | n2)); ja = len(a1 & a2) / max(1, len(a1 | a2))
        nums1 = {t for t in a1 if t.isdigit()}; nums2 = {t for t in a2 if t.isdigit()}
        rows.append((m[:2], r.country, jn, ja, len(nums1 & nums2) > 0, len(nums2) == 0))
st = pd.DataFrame(rows, columns=["src", "country", "jname", "jaddr", "share_num", "no_num"])
print("matched-pair name jaccard by src/country\n", st.groupby(["src", "country"]).jname.describe()[["mean", "25%", "50%"]].round(3))
print("matched-pair addr jaccard\n", st.groupby(["src", "country"]).jaddr.describe()[["mean", "25%", "50%"]].round(3))
print("frac name jaccard==0", st.groupby(["src", "country"]).jname.apply(lambda x: (x == 0).mean()).round(3).to_dict())
print("frac share a number", st.groupby(["src", "country"]).share_num.mean().round(3).to_dict())

# hard negatives: unmatched S3 records -- what do they look like? nearest S1 by exact normalized address
key = lambda s: " ".join(sorted(norm(s)))
s1_addr_key = collections.Counter(s1.business_address.map(key))
print("S1 address exact-dup (norm) frac", np.mean([v > 1 for v in s1_addr_key.values()]))
un3 = s3[~s3.entity_id.isin(matched)].sample(8, random_state=3)
print("\nUnmatched S3 samples:"); print(un3.to_string())
un2 = s2[~s2.entity_id.isin(matched)].sample(8, random_state=3)
print("\nUnmatched S2 samples:"); print(un2.to_string())

# examples with S2
for idx in rng.choice(len(gt), 8, replace=False):
    sid = gt.source1_entity_id.iat[idx]; r = s1i.loc[sid]
    print("\n==", sid, "|", r.business_name, "|", r.business_address, "|", r.country)
    for mid in lists.iat[idx]:
        q = (s2i if mid.startswith("S2") else s3i).loc[mid]
        print("   ", mid, "|", q.business_name, "|", q.business_address)
# singleton examples
print("\nSingletons:")
print(s1[s1.entity_id.isin(gt.source1_entity_id[gt.matched_entity_ids == ""].sample(6, random_state=5))].to_string())
