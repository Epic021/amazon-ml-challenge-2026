import pandas as pd, numpy as np, re, csv, collections, sys
import os
D = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "student_resource", "dataset"))
def load(p):
    return pd.read_csv(f"{D}/{p}", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE, engine="c")

gt = load("train/train_ground_truth.tsv")
s1 = load("train/train_source1.tsv")
print("GT", gt.shape, "S1", s1.shape)
print("S1 id unique", s1.entity_id.is_unique, "GT s1 unique", gt.source1_entity_id.is_unique,
      "GT==S1 set", set(gt.source1_entity_id) == set(s1.entity_id))
lists = gt.matched_entity_ids.map(lambda x: x.split(",") if x else [])
n = lists.map(len)
n2 = lists.map(lambda l: sum(i.startswith("S2-") for i in l))
n3 = lists.map(lambda l: sum(i.startswith("S3-") for i in l))
print("singleton frac", (n == 0).mean())
print("match count dist\n", n.value_counts().sort_index().head(20))
print("S2 count dist\n", n2.value_counts().sort_index().head(12))
print("S3 count dist\n", n3.value_counts().sort_index().head(12))
allm = [i for l in lists for i in l]
c = collections.Counter(allm)
print("total matched ids", len(allm), "unique", len(c), "ids matched to >1 S1:", sum(v > 1 for v in c.values()))
s2ids = {i for i in c if i.startswith("S2-")}; s3ids = {i for i in c if i.startswith("S3-")}
print("unique S2 in GT", len(s2ids), "unique S3 in GT", len(s3ids))

s3 = load("train/train_source3.tsv")
print("S3", s3.shape, "id unique", s3.entity_id.is_unique)
s3set = set(s3.entity_id)
print("GT S3 ids present in train_s3:", len(s3ids & s3set), "/", len(s3ids), "; S3 rows never matched:", len(s3set - s3ids))

for name, df in [("S1", s1), ("S3", s3)]:
    print(f"--- {name}")
    print("country\n", df.country.value_counts().head(10))
    for col in ["business_name", "business_address", "country"]:
        print(col, "empty frac", (df[col].str.strip() == "").mean())
    print("name len", df.business_name.str.len().describe().round(1).to_dict())
    print("addr len", df.business_address.str.len().describe().round(1).to_dict())
    print("dup (name,addr,country)", df.duplicated(["business_name", "business_address", "country"]).mean())
    print("mojibake frac (Ã|Â|à²)", df.business_name.str.contains("Ã|Â|à¤|à²", regex=True).mean(),
          df.business_address.str.contains("Ã|Â|à¤|à²", regex=True).mean())
    ids = df.entity_id.str[3:].astype(np.int64)
    print("id num range", ids.min(), ids.max(), "digits dist", df.entity_id.str.len().value_counts().head(5).to_dict())

# id semantics: does S1 id numeric correlate with matched ids / singleton?
s1n = gt.source1_entity_id.str[3:].astype(np.int64)
print("corr(id, n_matches)", np.corrcoef(s1n, n)[0, 1], "singleton frac by id decile",
      pd.Series(n.values == 0).groupby(pd.qcut(s1n, 10, labels=False).values).mean().round(3).tolist())
# country of S1 vs singleton / match counts
m = s1.set_index("entity_id").country
gc = gt.source1_entity_id.map(m)
print("singleton by country", pd.Series(n.values == 0).groupby(gc.values).mean().to_dict())
print("mean n by country", n.groupby(gc.values).mean().to_dict())
# is S3 in match set same country?
s3c = s3.set_index("entity_id").country
rows = [(g, i) for g, l in zip(gc.values[:300000], lists.values[:300000]) for i in l if i.startswith("S3-")]
same = np.mean([s3c.get(i) == g for g, i in rows])
print("S3 match same country frac", same)

# sample examples
rng = np.random.default_rng(0)
s1i = s1.set_index("entity_id"); s3i = s3.set_index("entity_id")
for idx in rng.choice(len(gt), 12, replace=False):
    sid = gt.source1_entity_id.iat[idx]; r = s1i.loc[sid]
    print("\n==", sid, "|", r.business_name, "|", r.business_address, "|", r.country)
    for mid in lists.iat[idx]:
        if mid in s3i.index:
            q = s3i.loc[mid]; print("   ", mid, "|", q.business_name, "|", q.business_address, "|", q.country)
        else:
            print("   ", mid, "(not in train files)")
