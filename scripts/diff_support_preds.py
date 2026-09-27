"""How many PREDICTED matches fall in the "difference repeated by an independent record" bucket, per country,
and what is the holdout precision of that bucket? (France false-match hypothesis.)

Predictions: stage-1 p for --tag, isotonic on folds 6-9, soft exclusivity, threshold --thr.

  python scripts/diff_support_preds.py --tag b5 --thr 0.6
"""
import argparse
import os
import sys

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diff_support import with_support, SWAP, KEY, DATA  # noqa: E402


def decode(d: pl.DataFrame, iso, thr: float) -> pl.DataFrame:
    d = d.with_columns(pl.Series("q", iso.predict(d["p"].to_numpy())).clip(upper_bound=0.999))
    o = pl.col("q") / (1 - pl.col("q"))
    return d.with_columns((o / (1 + o.sum().over("cand_id"))).alias("px")).with_columns((pl.col("px") >= thr).alias("pred"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b5")
    ap.add_argument("--feat_dir", default=f"{DATA}/feat_b5")
    ap.add_argument("--thr", type=float, default=0.6)
    a = ap.parse_args()
    pl.Config.set_tbl_rows(60).set_tbl_width_chars(220)
    fold = pl.col("s1_id").str.slice(3).cast(pl.Int64) % 10
    cls = ["num_rel", "addr_tset", "n_added", "n_dropped", "is_c_best"]
    tr = pl.read_parquet(f"{DATA}/pred/train_{a.tag}.parquet", columns=KEY + ["p", "fold", "y"])
    cal = tr.filter(pl.col("fold").is_in([6, 7, 8, 9]))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(cal["p"].to_numpy(), cal["y"].to_numpy())
    for split in ("train", "test"):
        s1 = pl.read_parquet(f"{DATA}/parquet/{split}_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
        if split == "train":
            s1 = s1.filter(fold == 5)
            p = tr.filter(pl.col("fold") == 5)
        else:                                  # all of France, 20% of India/US S1 (memory)
            s1 = s1.filter((pl.col("country") == "France") | (pl.col("s1_id").str.slice(3).cast(pl.Int64) % 5 == 0))
            p = pl.read_parquet(f"{DATA}/pred/test_{a.tag}.parquet", columns=KEY + ["p"]).join(s1.select("s1_id"), on="s1_id")
        d = decode(p, iso, a.thr)
        f = pl.scan_parquet(f"{a.feat_dir}/{split}.parquet/*.parquet")
        f = (f.filter(fold == 5) if split == "train" else f).select(KEY + cls).collect()
        d = with_support(d.join(f, on=KEY, how="left"), split)
        d = d.join(s1, on="s1_id").filter("pred")
        d = d.with_columns(pl.when(SWAP).then(pl.lit("same-num swap")).when((pl.col("n_added") >= 1) & (pl.col("n_dropped") >= 1))
                           .then(pl.lit("other swap")).otherwise(pl.lit("no swap")).alias("kind"),
                           pl.when(pl.col("diff_support") >= 1).then(pl.lit("repeated")).otherwise(pl.lit("-")).alias("diff"))
        n = s1.group_by("country").len("n_s1")
        agg = [pl.len().alias("n"), pl.col("px").mean().round(3).alias("px")]
        if split == "train":
            agg.append(pl.col("y").mean().round(3).alias("PRECISION"))
        out = d.group_by(["country", "kind", "diff"]).agg(agg).join(n, on="country") \
               .with_columns((pl.col("n") / pl.col("n_s1")).round(4).alias("pred_per_S1")).drop("n_s1")
        print(f"{split.upper()} predicted pairs (thr {a.thr}) by kind x repeated-difference:")
        print(out.sort(["kind", "diff", "country"]))


if __name__ == "__main__":
    main()
