"""Blocking-recall probe: IDF-weighted inverted-index retrieval (S2/S3 query -> S1), full-scale S1 index."""
import pandas as pd, numpy as np, re, csv, unicodedata
import os
D = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "student_resource", "dataset"))
def load(p):
    return pd.read_csv(f"{D}/{p}", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE, engine="c")
ABBR = {"st": "street", "rd": "road", "dr": "drive", "ave": "avenue", "ln": "lane", "ct": "court", "blvd": "boulevard",
        "ltd": "limited", "pvt": "private", "corp": "corporation", "co": "company", "inc": "incorporated"}
def toks(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w]+", " ", s)
    return {ABBR.get(t, t) for t in s.split() if t not in ("null", "n", "a", "nan")}
gt = load("train/train_ground_truth.tsv")
s1 = load("train/train_source1.tsv")
pairs = gt[gt.matched_entity_ids != ""].assign(m=lambda d: d.matched_entity_ids.str.split(",")).explode("m")[["source1_entity_id", "m"]]
true_s1 = dict(zip(pairs.m, pairs.source1_entity_id))
rng = np.random.default_rng(0)
qs = []
for k in (2, 3):
    s = load(f"train/train_source{k}.tsv")
    matched = s.entity_id.isin(true_s1)
    qs.append(s[matched].sample(15000, random_state=k)); qs.append(s[~matched].sample(5000, random_state=k))
    del s
q = pd.concat(qs, ignore_index=True)
q["truth"] = q.entity_id.map(true_s1)

def explode(df, idcol):
    rows = []
    for i, n, a in zip(df[idcol].values, df.business_name.values, df.business_address.values):
        for t in toks(n): rows.append((i, "n:" + t))
        for t in toks(a): rows.append((i, "a:" + t))
    return pd.DataFrame(rows, columns=["id", "tok"])

res = []
for country in ["US", "India"]:
    S = s1[s1.country == country].reset_index(drop=True)
    S["idx"] = np.arange(len(S))
    E = explode(S, "idx")
    df = E.tok.value_counts()
    idf = np.log(len(S) / df)
    Q = q[q.country == country].reset_index(drop=True); Q["qi"] = np.arange(len(Q))
    QE = explode(Q, "qi")
    QE = QE[QE.tok.map(df).fillna(10**9) <= 3000]
    QE["w"] = QE.tok.map(idf)
    M = QE.merge(E[E.tok.isin(set(QE.tok))], on="tok")
    sc = M.groupby(["id_x", "id_y"]).w.sum().reset_index()
    sc.columns = ["qi", "idx", "score"]
    sc["rank"] = sc.groupby("qi").score.rank(ascending=False, method="first")
    sc = sc[sc["rank"] <= 50]
    sc["s1id"] = S.entity_id.values[sc.idx.values]
    Q = Q.merge(sc[sc["rank"] == 1][["qi", "s1id", "score"]].rename(columns={"s1id": "top1", "score": "top1score"}), on="qi", how="left")
    hit = sc.merge(Q[["qi", "truth"]], on="qi")
    hit = hit[hit.s1id == hit.truth][["qi", "rank", "score"]]
    Q = Q.merge(hit, on="qi", how="left")
    Q["src"] = Q.entity_id.str[:2]; Q["nonlatin"] = Q.business_name.map(lambda s: any(ord(c) > 0x24F and c.isalpha() for c in s))
    res.append(Q)
R = pd.concat(res)
m = R[R.truth.notna()]
for (c, s), g in m.groupby(["country", "src"]):
    print(c, s, "n", len(g), {f"R@{k}": round((g["rank"] <= k).mean(), 4) for k in (1, 5, 10, 20, 50)},
          "| nonlatin-name R@50", round((g[g.nonlatin]["rank"] <= 50).mean(), 3), "nonlatin frac", round(g.nonlatin.mean(), 3))
u = R[R.truth.isna()]
print("\nUnmatched queries: top1 score quantiles", u.top1score.quantile([.1, .5, .9]).round(2).to_dict())
print("Matched queries: true score quantiles", m.score.quantile([.1, .5, .9]).round(2).to_dict())
print("Matched queries where top1 != truth: frac", (m.top1 != m.truth).mean().round(4))
s1i = s1.set_index("entity_id")
print("\nHard-negative examples (unmatched query with highest top1 score):")
for _, r in u.sort_values("top1score", ascending=False).head(8).iterrows():
    t = s1i.loc[r.top1]
    print(f"  Q {r.entity_id} | {r.business_name} | {r.business_address}\n    -> {r.top1} | {t.business_name} | {t.business_address}  score={r.top1score:.1f}")
print("\nMatched but top1 wrong examples:")
for _, r in m[(m.top1 != m.truth)].sample(6, random_state=0).iterrows():
    t = s1i.loc[r.top1]; tt = s1i.loc[r.truth]
    print(f"  Q {r.entity_id} | {r.business_name} | {r.business_address}\n    top1 {t.business_name} | {t.business_address}\n    TRUE {tt.business_name} | {tt.business_address} (rank {r['rank']})")
