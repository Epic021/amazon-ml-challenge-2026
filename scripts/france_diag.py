"""Label-free per-country check of a test submission.

For the predicted matches, how often do they look like the organisers' decoys
(near-identical address, different house number, extra word in the name)? France has no labels,
so this is our proxy: a lower decoy-signature share for France = fewer false merges.

  python scripts/france_diag.py output/matching_results.tsv
"""
import csv
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))


def main(path):
    f = pd.read_parquet(f"{DATA}/feat/test.parquet",
                        columns=["s1_id", "cand_id", "num_rel", "addr_tset", "n_added"])
    m = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    m = m.assign(cand_id=m.matched_entity_ids.str.split(",")).explode("cand_id")
    m = m[m.cand_id != ""].rename(columns={"source1_entity_id": "s1_id"})[["s1_id", "cand_id"]]
    s1 = pd.read_parquet(f"{DATA}/parquet/test_s1.parquet", columns=["entity_id", "country"])
    ctry = s1.set_index("entity_id").country
    p = m.merge(f, on=["s1_id", "cand_id"], how="left")
    p["country"] = p.s1_id.map(ctry)
    decoy = (p.num_rel == 5) & (p.addr_tset >= 90) & (p.n_added > 0)
    per = m.groupby("s1_id").size().reindex(s1.entity_id, fill_value=0)
    out = pd.DataFrame({
        "pred_per_s1": per.groupby(s1.country.values).mean(),
        "empty_share": (per == 0).groupby(s1.country.values).mean(),
        "num_differ_share": (p.num_rel == 5).groupby(p.country).mean(),
        "decoy_signature_share": decoy.groupby(p.country).mean(),
    }).round(4)
    print(f"{path}\n{out.to_string()}")


if __name__ == "__main__":
    main(sys.argv[1])
