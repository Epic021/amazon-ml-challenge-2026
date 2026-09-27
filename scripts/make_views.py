"""View-consensus features for stage 2 (see scripts/view_consensus.py).

For each stage-2 pair (S1, c) with max(our p, teammate p) >= 0.01:
  view_sim   = max rapidfuzz ratio between name_norm(c) and name_norm(v) over the S1's confident other
               matches v (our stage-1 p >= 0.8, v != c); -1 if the S1 has none
  view_exact = 1 if some view has exactly the same normalised full name
  view_n     = number of confident views
Writes data/extra/{split}_views.parquet. Train uses the out-of-fold folds only (context folds).

  python scripts/make_views.py --tag b5 --friend_dir friend_v2
"""
import argparse
import os

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
KEY = ["s1_id", "cand_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b5")
    ap.add_argument("--friend_dir", default="friend_v2")
    ap.add_argument("--context_folds", default="0,1,5,6,7,8,9")
    a = ap.parse_args()
    os.makedirs(f"{DATA}/extra", exist_ok=True)
    fold = pl.col("s1_id").str.slice(3).cast(pl.Int64) % 10
    ctx = [int(x) for x in a.context_folds.split(",")]
    for split in ("train", "test"):
        p = pl.read_parquet(f"{DATA}/pred/{split}_{a.tag}.parquet", columns=KEY + ["p"])
        t = pl.read_parquet(f"{DATA}/{a.friend_dir}/{'train_oof_probs' if split == 'train' else 'test_probs'}.parquet",
                            columns=KEY + ["p"]).rename({"p": "p_t"})
        if split == "train":
            p, t = p.filter(fold.is_in(ctx)), t.filter(fold.is_in(ctx))
        pairs = p.join(t, on=KEY, how="full", coalesce=True).fill_null(0.0)
        views = pairs.filter(pl.col("p") >= 0.8).select("s1_id", pl.col("cand_id").alias("v_id"))
        nv = views.group_by("s1_id").len("view_n")
        tgt = pairs.filter(pl.max_horizontal("p", "p_t") >= 0.01).select(KEY)
        tri = tgt.join(views, on="s1_id").filter(pl.col("v_id") != pl.col("cand_id"))
        names = pl.concat([pl.read_parquet(f"{DATA}/norm/{split}_s{k}.parquet", columns=["id", "name_norm"]) for k in (2, 3)])
        tri = tri.join(names.rename({"id": "cand_id", "name_norm": "a"}), on="cand_id", how="left") \
                 .join(names.rename({"id": "v_id", "name_norm": "b"}), on="v_id", how="left").fill_null("")
        sim = cpdist(tri["a"].to_list(), tri["b"].to_list(), scorer=fuzz.ratio, workers=-1).astype(np.float32)
        tri = tri.with_columns(pl.Series("sim", sim), ((pl.col("a") == pl.col("b")) & (pl.col("a") != "")).cast(pl.Int8).alias("ex"))
        agg = tri.group_by(KEY).agg(pl.col("sim").max().alias("view_sim"), pl.col("ex").max().alias("view_exact"))
        out = tgt.join(agg, on=KEY, how="left").join(pl.concat([tgt.select("s1_id").unique()]).join(nv, on="s1_id", how="left"),
                                                   on="s1_id", how="left")
        out = out.with_columns(pl.col("view_sim").fill_null(-1.0), pl.col("view_exact").fill_null(-1).cast(pl.Int8),
                               pl.col("view_n").fill_null(0).cast(pl.Int16))
        out.write_parquet(f"{DATA}/extra/{split}_views.parquet")
        print(f"{split}: {out.height:,} pairs with p>=0.01, {tri.height:,} (pair, view) comparisons", flush=True)


if __name__ == "__main__":
    main()
