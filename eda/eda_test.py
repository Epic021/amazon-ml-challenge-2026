import pandas as pd, numpy as np, csv
import os
D = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "student_resource", "dataset"))
def load(p):
    return pd.read_csv(f"{D}/{p}", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE, engine="c")
tr = {k: load(f"train/train_source{k}.tsv") for k in (1, 2, 3)}
te = {k: load(f"test/test_source{k}.tsv") for k in (1, 2, 3)}
for k in (1, 2, 3):
    print(f"S{k} train {len(tr[k])} test {len(te[k])}  test countries {te[k].country.value_counts().to_dict()}")
    print("   id overlap train/test", len(set(tr[k].entity_id) & set(te[k].entity_id)))
    print("   name overlap train/test (exact)", te[k].business_name.isin(set(tr[k].business_name)).mean().round(3),
          " exact (name,addr) overlap", te[k].set_index(["business_name", "business_address"]).index.isin(
              tr[k].set_index(["business_name", "business_address"]).index).mean().round(4))
print("ratio (S2+S3)/S1 train", (len(tr[2]) + len(tr[3])) / len(tr[1]), "test", (len(te[2]) + len(te[3])) / len(te[1]))
for c in ["US", "India", "France"]:
    n1 = (te[1].country == c).sum(); n23 = (te[2].country == c).sum() + (te[3].country == c).sum()
    print(c, "test S1", n1, "S2+S3", n23, "ratio", round(n23 / max(n1, 1), 2))
fr = {k: te[k][te[k].country == "France"] for k in (1, 2, 3)}
for k in (1, 2, 3):
    print(f"\nFrance S{k} sample"); print(fr[k].sample(8, random_state=0).to_string())
# France: exact normalized address overlap S1 vs S2/S3 to see structure
