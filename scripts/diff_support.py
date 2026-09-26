"""Collective-ER signal test: is a name difference "real" (a different business) or "noise"?

For a pair (S1, record c): sig = sorted(words of c not in S1) | sorted(words of S1 not in c), on name_core.
diff_support = number of OTHER-source candidates of the same S1 with the identical non-empty sig.
Noise (typos, abbreviations) is independent per source, so an identical difference seen in an independent
record suggests the difference is the name of a different business (a neighbour). Label-free.

Reports P(match) by diff_support on the holdout (fold 5), within the same-address word-swap class and
overall, and how common the signal is on test per country.

  python scripts/diff_support.py --feat_dir data/feat_b5
"""
import argparse
import os

import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
KEY = ["s1_id", "cand_id"]
SWAP = ((pl.col("num_rel") == 1) & (pl.col("addr_tset") >= 90) & (pl.col("n_added") >= 1)
        & (pl.col("n_dropped") >= 1))


def with_support(pairs: pl.DataFrame, split: str) -> pl.DataFrame:
    rec = pl.concat([pl.read_parquet(f"{DATA}/norm/{split}_s{k}.parquet", columns=["id", "src", "name_core"])
                     for k in (1, 2, 3)])
    tok = rec.select("id", "src", pl.col("name_core").str.split(" ").list.unique().alias("t"))
    d = pairs.join(tok.select(pl.col("id").alias("s1_id"), pl.col("t").alias("t1")), on="s1_id") \
             .join(tok.select(pl.col("id").alias("cand_id"), "src", pl.col("t").alias("tc")), on="cand_id")
    d = d.with_columns(pl.col("tc").list.set_difference("t1").list.sort().list.join(" ").alias("_add"),
                       pl.col("t1").list.set_difference("tc").list.sort().list.join(" ").alias("_drop"))
    d = d.with_columns((pl.col("_add") + "|" + pl.col("_drop")).alias("sig"))
    d = d.with_columns(
        (pl.len().over(["s1_id", "sig"]) - pl.len().over(["s1_id", "sig", "src"])).alias("diff_support"),
        (pl.len().over(["s1_id", "_add"]) - pl.len().over(["s1_id", "_add", "src"])).alias("add_support"))
    empty = (pl.col("_add") == "") & (pl.col("_drop") == "")
    return d.with_columns(pl.when(empty).then(-1).otherwise(pl.col("diff_support")).alias("diff_support"),
                          pl.when(pl.col("_add") == "").then(-1).otherwise(pl.col("add_support")).alias("add_support")
                          ).drop(["t1", "tc", "_add", "_drop"])


def bucket(c):
    return pl.when(pl.col(c) < 0).then(pl.lit("n/a")).when(pl.col(c) == 0).then(pl.lit("0")).otherwise(pl.lit(">=1")).alias(c + "_b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", default=f"{DATA}/feat_b5")
    a = ap.parse_args()
    pl.Config.set_tbl_rows(40).set_tbl_width_chars(200)
    cls = ["num_rel", "addr_tset", "n_added", "n_dropped", "is_c_best"]
    fold = pl.col("s1_id").str.slice(3).cast(pl.Int64) % 10
    tr = pl.scan_parquet(f"{a.feat_dir}/train.parquet/*.parquet").filter(fold == 5).select(KEY + ["y"] + cls).collect()
    tr = with_support(tr, "train").with_columns(SWAP.alias("swap"), bucket("diff_support"), bucket("add_support"))
    s1 = pl.read_parquet(f"{DATA}/parquet/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    tr = tr.join(s1, on="s1_id")
    print("HOLDOUT P(match) by diff_support (identical add|drop signature in an other-source candidate):")
    print(tr.group_by(["country", "swap", "diff_support_b"]).agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("P_match"))
            .sort(["country", "swap", "diff_support_b"]))
    print("HOLDOUT P(match) by add_support (same added words in an other-source candidate), swap class only:")
    print(tr.filter("swap").group_by(["country", "add_support_b"]).agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("P_match"))
            .sort(["country", "add_support_b"]))
    te = pl.scan_parquet(f"{a.feat_dir}/test.parquet/*.parquet").select(KEY + cls).collect()
    te = with_support(te, "test").with_columns(SWAP.alias("swap"), bucket("diff_support"))
    s1t = pl.read_parquet(f"{DATA}/parquet/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    te = te.join(s1t, on="s1_id")
    n = s1t.group_by("country").len("n_s1")
    print("TEST swap-class pairs per S1 by diff_support:")
    print(te.filter("swap").group_by(["country", "diff_support_b"]).len().join(n, on="country")
            .with_columns((pl.col("len") / pl.col("n_s1")).round(3).alias("per_S1")).sort(["country", "diff_support_b"]))


if __name__ == "__main__":
    main()
