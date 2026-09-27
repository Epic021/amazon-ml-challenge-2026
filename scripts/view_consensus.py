"""Does a record's FULL name agree with the S1's other confident matches ("views")?

For ambiguous empty-address records (identical core name shared by 2-3 S1), the S1 name cannot decide
the owner. The owner's other records (stage-1 p >= 0.8, any source) carry the entity's full name incl.
legal form. view_sim(S1, c) = max rapidfuzz ratio between name_norm(c) and name_norm(view), view != c.

Reports, over records whose candidate S1 set has >= 2 members (all train folds as candidates, truth in
fold 5): how often the S1 with the highest view_sim is the owner, vs stage-1 top pick and chance.
Also P(match) by view_sim bucket over all holdout empty-address / swap pairs.

  python scripts/view_consensus.py --tag b5
"""
import argparse
import os

import numpy as np
import polars as pl
from rapidfuzz import fuzz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
KEY = ["s1_id", "cand_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b5")
    ap.add_argument("--feat_dir", default=f"{DATA}/feat_b5")
    a = ap.parse_args()
    pl.Config.set_tbl_rows(40).set_tbl_width_chars(200)
    fold = (pl.col("s1_id").str.slice(3).cast(pl.Int64) % 10).cast(pl.Int8)
    p = pl.read_parquet(f"{DATA}/pred/train_{a.tag}.parquet", columns=KEY + ["p"]).with_columns(fold.alias("fold"))
    p = p.filter(pl.col("fold").is_in([0, 1, 5, 6, 7, 8, 9]))               # out-of-fold stage-1 p only
    f = pl.scan_parquet(f"{a.feat_dir}/train.parquet/*.parquet").select(
        KEY + ["c_addr_empty", "core_eq", "c_core_n_s1", "num_rel", "addr_tset", "n_added", "n_dropped"]).collect()
    tr = pl.read_parquet(f"{DATA}/parquet/train_pairs.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
    d = p.join(f, on=KEY, how="left").join(tr, on=KEY, how="left").with_columns(pl.col("y").fill_null(0))
    names = pl.concat([pl.read_parquet(f"{DATA}/norm/train_s{k}.parquet", columns=["id", "name_norm"]) for k in (2, 3)])
    nm = dict(zip(names["id"].to_list(), names["name_norm"].to_list()))

    views = d.filter(pl.col("p") >= 0.8).select("s1_id", pl.col("cand_id").alias("v_id"))
    amb = d.filter((pl.col("c_addr_empty") == 1) & (pl.col("core_eq") == 1) & pl.col("c_core_n_s1").is_between(2, 3))
    swap = d.filter((pl.col("fold") == 5) & (pl.col("num_rel") == 1) & (pl.col("addr_tset") >= 90)
                    & (pl.col("n_added") >= 1) & (pl.col("n_dropped") >= 1))
    for name, x in (("ambiguous empty-address", amb), ("same-address word swap (fold 5)", swap)):
        t = x.select(KEY + ["p", "y", "fold"]).join(views, on="s1_id").filter(pl.col("v_id") != pl.col("cand_id"))
        c_names = [nm.get(c, "") for c in t["cand_id"].to_list()]
        v_names = [nm.get(v, "") for v in t["v_id"].to_list()]
        t = t.with_columns(pl.Series("sim", np.array([fuzz.ratio(a_, b_) for a_, b_ in zip(c_names, v_names)], np.float32)),
                           pl.Series("exact", np.array([a_ == b_ and a_ != "" for a_, b_ in zip(c_names, v_names)], np.int8)))
        vs = t.group_by(KEY).agg(pl.col("sim").max().alias("view_sim"), pl.col("exact").max().alias("view_exact"))
        x = x.join(vs, on=KEY, how="left").with_columns(pl.col("view_sim").fill_null(-1), pl.col("view_exact").fill_null(-1))
        b = x.filter(pl.col("fold") == 5) if name.startswith("amb") else x
        print(f"\n== {name}: P(match) by view agreement (fold-5 truth) ==")
        print(b.with_columns(pl.when(pl.col("view_sim") < 0).then(pl.lit("no views")).when(pl.col("view_exact") == 1)
                             .then(pl.lit("exact")).when(pl.col("view_sim") >= 90).then(pl.lit(">=90"))
                             .when(pl.col("view_sim") >= 75).then(pl.lit("75-90")).otherwise(pl.lit("<75")).alias("agree"))
               .group_by("agree").agg(pl.len(), pl.col("y").mean().round(3).alias("P_match"), pl.col("p").mean().round(3).alias("stage1_p"))
               .sort("agree"))
        if name.startswith("amb"):
            g = x.filter(pl.col("cand_id").is_in(x.filter((pl.col("y") == 1) & (pl.col("fold") == 5))["cand_id"]))
            g = g.with_columns(pl.len().over("cand_id").alias("k"),
                               pl.col("view_sim").rank("ordinal", descending=True).over("cand_id").alias("r_view"),
                               pl.col("p").rank("ordinal", descending=True).over("cand_id").alias("r_p"))
            tt = g.filter((pl.col("y") == 1) & (pl.col("k") >= 2))
            print(f"records with >=2 candidate S1: {tt.height:,}; owner is top by view_sim {(tt['r_view'] == 1).mean():.3f}, "
                  f"top by stage-1 p {(tt['r_p'] == 1).mean():.3f}, chance {(1 / tt['k']).mean():.3f}")


if __name__ == "__main__":
    main()
