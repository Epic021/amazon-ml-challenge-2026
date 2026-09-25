"""Resolve open data questions:
Q5 test distractor shift, Q6 sibling-vs-orphan negatives, Q7/Q8 noise ops in pos vs neg,
Q9 address-only / name-only share, Q10 France structure.
Retrieval: IDF-weighted inverted index (query S2/S3 -> S1), then rapidfuzz sims on top-5."""
import pandas as pd, numpy as np, re, csv, unicodedata, collections, gc
from anyascii import anyascii
import ftfy
from rapidfuzz import fuzz
import os
D = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "student_resource", "dataset"))
def load(p):
    return pd.read_csv(f"{D}/{p}", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE, engine="c")
ABBR = {"st": "street", "rd": "road", "dr": "drive", "ave": "avenue", "av": "avenue", "ln": "lane", "ct": "court",
        "blvd": "boulevard", "bd": "boulevard", "r": "rue", "ltd": "limited", "pvt": "private", "corp": "corporation",
        "co": "company", "inc": "incorporated", "no": "", "h": "", "null": "", "nan": "", "n": "", "a": ""}
def clean(s):
    if not s.isascii():
        s = anyascii(unicodedata.normalize("NFKC", ftfy.fix_text(s)))
    s = re.sub(r"[^0-9a-z]+", " ", s.lower())
    return " ".join(t for t in (ABBR.get(t, t) for t in s.split()) if t)
def explode(ids, names, addrs):
    r_id, r_tok = [], []
    for i, n, a in zip(ids, names, addrs):
        for t in set(n.split()): r_id.append(i); r_tok.append("n:" + t)
        for t in set(a.split()): r_id.append(i); r_tok.append("a:" + t)
    return pd.DataFrame({"id": r_id, "tok": r_tok})

def retrieve(S, Q, k=5, maxdf=1500, chunk=4000):
    """S, Q have cols nm, ad (cleaned). returns df qi, si, score, rank (top-k)."""
    E = explode(np.arange(len(S)), S.nm.values, S.ad.values)
    df = E.tok.value_counts(); idf = np.log(len(S) / df)
    keep = df[df <= maxdf].index
    E = E[E.tok.isin(keep)]
    E["tok"] = pd.Categorical(E.tok, categories=keep)
    out = []
    for c0 in range(0, len(Q), chunk):
        q = Q.iloc[c0:c0 + chunk]
        QE = explode(np.arange(c0, c0 + len(q)), q.nm.values, q.ad.values)
        QE = QE[QE.tok.isin(keep)]
        QE["w"] = QE.tok.map(idf).astype("float32")
        QE["tok"] = pd.Categorical(QE.tok, categories=keep)
        M = QE.merge(E, on="tok", suffixes=("q", "s"))
        sc = M.groupby(["idq", "ids"], observed=True).w.sum().reset_index()
        sc = sc.sort_values(["idq", "w"], ascending=[True, False])
        sc["rank"] = sc.groupby("idq").cumcount() + 1
        out.append(sc[sc["rank"] <= k])
    R = pd.concat(out); R.columns = ["qi", "si", "score", "rank"]
    return R

def sims(R, S, Q):
    R = R.copy()
    qn, qa = Q.nm.values[R.qi.values], Q.ad.values[R.qi.values]
    sn, sa = S.nm.values[R.si.values], S.ad.values[R.si.values]
    R["nsim"] = [fuzz.token_set_ratio(a, b) for a, b in zip(qn, sn)]
    R["asim"] = [fuzz.token_set_ratio(a, b) if a else -1 for a, b in zip(qa, sa)]
    return R

def prep(df):
    df = df.reset_index(drop=True)
    df["nm"] = df.business_name.map(clean); df["ad"] = df.business_address.map(clean)
    return df

# ---------------- TRAIN ----------------
gt = load("train/train_ground_truth.tsv")
pairs = gt[gt.matched_entity_ids != ""].assign(m=lambda d: d.matched_entity_ids.str.split(",")).explode("m")
true_s1 = dict(zip(pairs.m, pairs.source1_entity_id)); del pairs, gt; gc.collect()
s1 = load("train/train_source1.tsv")
Qs = []
for k in (2, 3):
    s = load(f"train/train_source{k}.tsv"); Qs.append(s.sample(24000, random_state=k)); del s
Qtr = pd.concat(Qs, ignore_index=True); Qtr["truth"] = Qtr.entity_id.map(true_s1)
allres = {}
for country in ["US", "India"]:
    S = prep(s1[s1.country == country]); Q = prep(Qtr[Qtr.country == country])
    R = sims(retrieve(S, Q), S, Q)
    R["q_id"] = Q.entity_id.values[R.qi.values]; R["s_id"] = S.entity_id.values[R.si.values]
    R["truth"] = Q.truth.values[R.qi.values]; R["is_true"] = R.s_id == R.truth
    R["country"] = country; R["src"] = R.q_id.str[:2]
    allres[country] = (S, Q, R)

R = pd.concat([v[2] for v in allres.values()])
R["comb"] = (R.nsim + R.asim.clip(lower=0)) / 2
top = R.sort_values(["country", "qi", "comb"], ascending=[True, True, False]).groupby(["country", "qi"]).head(1)
top["matched"] = top.truth.notna()
bins = [-1, 40, 55, 65, 75, 85, 92, 101]
top["bin"] = pd.cut(top.comb, bins)
tab = top.groupby("bin", observed=False).agg(n=("matched", "size"), p_matched=("matched", "mean"),
                                             p_top_is_true=("is_true", "mean"))
print("TRAIN: best-candidate similarity bin -> P(record has a match), P(best cand is the true S1)\n", tab.round(3))

# Q6: unmatched records -- sibling of an S1 (constructed negative) or orphan?
u = top[~top.matched]
print("\nQ6 unmatched records: best-S1 addr_sim>=90 & name_sim>=70 (sibling-like):",
      round(((u.asim >= 90) & (u.nsim >= 70)).mean(), 3), "| addr>=90 only:", round((u.asim >= 90).mean(), 3),
      "| comb<55 (orphan-like):", round((u.comb < 55).mean(), 3))

# Q7/Q8: positives vs hard negatives with near-identical address
def nums(s): return set(re.findall(r"\d+", s))
def rel(a, b):
    A, B = nums(a), nums(b)
    if not A and not B: return "none"
    if A == B: return "equal"
    if B > A: return "cand_adds_number"
    if B < A: return "cand_drops_number"
    return "differ"
pos = R[R.is_true].copy()
neg = top[(~top.matched) & (top.asim >= 85)].copy()
for nm_, d in [("positives", pos), ("hard negatives (unmatched, addr_sim>=85)", neg)]:
    S_all = {c: allres[c] for c in allres}
    rels, add_tok, drop_tok = [], collections.Counter(), collections.Counter()
    for c, qi, si in zip(d.country, d.qi, d.si):
        S, Q, _ = S_all[c]
        rels.append(rel(S.ad.values[si], Q.ad.values[qi]))
        qa, sa = set(Q.nm.values[qi].split()), set(S.nm.values[si].split())
        add_tok.update(qa - sa); drop_tok.update(sa - qa)
    d["numrel"] = rels
    print(f"\nQ8 {nm_} (n={len(d)}): number relation", d.numrel.value_counts(normalize=True).round(3).to_dict())
    print(f"   name-sim median {d.nsim.median()}, frac name_sim==100: {(d.nsim == 100).mean():.3f}")
    d.attrs["add"] = add_tok; d.attrs["drop"] = drop_tok
    print("   top ADDED name tokens:", add_tok.most_common(25))
    print("   top DROPPED name tokens:", drop_tok.most_common(15))
# log-odds of added tokens
pa, na = pos.attrs["add"], neg.attrs["add"]
tp, tn = sum(pa.values()), sum(na.values())
lo = {t: np.log((na[t] + 1) / tn) - np.log((pa[t] + 1) / tp) for t in set(pa) | set(na) if pa[t] + na[t] >= 30}
print("\nQ7 added tokens most indicative of NEGATIVE:", sorted(lo, key=lo.get, reverse=True)[:30])
print("Q7 added tokens most indicative of POSITIVE:", sorted(lo, key=lo.get)[:30])

# Q9: positives by what carries the signal
print("\nQ9 positives: name_sim<50 (renamed):", round((pos.nsim < 50).mean(), 3),
      "| addr empty:", round((pos.asim < 0).mean(), 3), "| addr_sim<50:", round(((pos.asim >= 0) & (pos.asim < 50)).mean(), 3),
      "| both >=80:", round(((pos.nsim >= 80) & (pos.asim >= 80)).mean(), 3))
print("   by country/src renamed frac:", pos.assign(r=pos.nsim < 50).groupby(["country", "src"]).r.mean().round(3).to_dict())
tr_bin_rate = tab.p_matched
del allres, R, s1, Qtr; gc.collect()

# ---------------- TEST (Q5, Q10) ----------------
t1 = load("test/test_source1.tsv")
Qs = []
for k in (2, 3):
    s = load(f"test/test_source{k}.tsv"); Qs.append(s.sample(36000, random_state=k)); del s
Qte = pd.concat(Qs, ignore_index=True)
tops = []
for country in ["US", "India", "France"]:
    S = prep(t1[t1.country == country]); Q = prep(Qte[Qte.country == country])
    Rt = sims(retrieve(S, Q), S, Q); Rt["comb"] = (Rt.nsim + Rt.asim.clip(lower=0)) / 2
    tt = Rt.sort_values(["qi", "comb"], ascending=[True, False]).groupby("qi").head(1).copy()
    tt["country"] = country; tops.append(tt)
    if country == "France":
        print("\nQ10 France: sample S2/S3 query -> best test-S1")
        for _, r in tt.sample(12, random_state=1).iterrows():
            print(f"  [{r.comb:.0f}] {Q.business_name.values[r.qi]} | {Q.business_address.values[r.qi]}\n       -> {S.business_name.values[r.si]} | {S.business_address.values[r.si]}")
        FS = S; FQ = Q
TT = pd.concat(tops); TT["bin"] = pd.cut(TT.comb, bins)
dist = TT.groupby(["country", "bin"], observed=False).size().unstack(0)
distn = dist / dist.sum()
trd = top.groupby("bin", observed=False).size(); trd = trd / trd.sum()
print("\nQ5 best-candidate similarity distribution (share of S2/S3 records):")
print(pd.concat([trd.rename("train_US+IN"), distn], axis=1).round(3))
for c in ["US", "India", "France"]:
    est = (distn[c] * tr_bin_rate.values).sum()
    print(f"Q5 est. share of test S2/S3 records that are true matches ({c}): {est:.3f}  (train actual ~0.74)")
# France vocabulary
ft = collections.Counter(t for n in FS.nm for t in n.split())
print("\nQ10 France S1 top name tokens:", ft.most_common(40))
fa = collections.Counter(t for n in FQ.ad for t in n.split() if not t.isdigit())
print("Q10 France S2/S3 top address tokens:", fa.most_common(50))
