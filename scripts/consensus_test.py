"""Collective-ER signal test 2: does a word-swap candidate agree with the S1's CONFIDENT matches?

For a same-address word-swap pair (S1, c): added = words of c not in S1 (name_core). Among the S1's other
candidates from the OTHER source with stage-1 p >= 0.8 ("confident matches"):
  add_confirmed = 1 if some confident match also contains all of c's added words, 0 if none does,
                  -1 if the S1 has no confident other-source match.
Stage-1 p is out-of-sample on fold 5 and on test, so this is usable as a stage-2 feature.

  python scripts/consensus_test.py --tag b5 --feat_dir data/feat_b5
"""
import argparse
import os

import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
KEY = ["s1_id", "cand_id"]
SWAP = ((pl.col("num_rel") == 1) & (pl.col("addr_tset") >= 90) & (pl.col("n_added") >= 1)
        & (pl.col("n_dropped") >= 1))


def consensus(pairs: pl.DataFrame, split: str) -> pl.DataFrame:
    """pairs: s1_id, cand_id, p, swap (+ anything). Returns pairs with add_confirmed."""
    rec = pl.concat([pl.read_parquet(f"{DATA}/norm/{split}_s{k}.parquet", columns=["id", "src", "name_core"])
                     for k in (1, 2, 3)])
    tok = rec.select("id", "src", pl.col("name_core").str.split(" ").list.unique().alias("t"))
    d = pairs.join(tok.select(pl.col("id").alias("s1_id"), pl.col("t").alias("t1")), on="s1_id") \
             .join(tok.select(pl.col("id").alias("cand_id"), "src", pl.col("t").alias("tc")), on="cand_id")
    d = d.with_columns(pl.col("tc").list.set_difference("t1").alias("added"))
    conf = d.filter(pl.col("p") >= 0.8).select("s1_id", pl.col("cand_id").alias("o_id"), pl.col("src").alias("o_src"),
                                               pl.col("tc").alias("t_o"))
    sw = d.filter(pl.col("swap")).select(KEY + ["src", "added"])
    j = sw.join(conf, on="s1_id").filter((pl.col("o_src") != pl.col("src")) & (pl.col("o_id") != pl.col("cand_id")))
    j = j.with_columns((pl.col("added").list.set_difference("t_o").list.len() == 0).alias("_has"))
    agg = j.group_by(KEY).agg(pl.col("_has").any().cast(pl.Int8).alias("add_confirmed"))
    return d.drop(["t1", "tc", "added"]).join(agg, on=KEY, how="left").with_columns(
        pl.when(pl.col("swap")).then(pl.col("add_confirmed").fill_null(-1)).otherwise(None).alias("add_confirmed"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b5")
    ap.add_argument("--feat_dir", default=f"{DATA}/feat_b5")
    a = ap.parse_args()
    pl.Config.set_tbl_rows(40).set_tbl_width_chars(200)
    cls = ["num_rel", "addr_tset", "n_added", "n_dropped"]
    fold = pl.col("s1_id").str.slice(3).cast(pl.Int64) % 10
    for split in ("train", "test"):
        f = pl.scan_parquet(f"{a.feat_dir}/{split}.parquet/*.parquet")
        f = f.filter(fold == 5) if split == "train" else f
        f = f.select(KEY + cls + (["y"] if split == "train" else [])).collect().with_columns(SWAP.alias("swap"))
        p = pl.read_parquet(f"{DATA}/pred/{split}_{a.tag}.parquet", columns=KEY + ["p"])
        d = consensus(f.join(p, on=KEY), split)
        s1 = pl.read_parquet(f"{DATA}/parquet/{split}_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
        d = d.filter("swap").join(s1, on="s1_id")
        agg = [pl.len().alias("n"), pl.col("p").mean().round(3).alias("model_p")]
        if split == "train":
            agg.append(pl.col("y").mean().round(3).alias("P_match"))
        n = s1.group_by("country").len("n_s1")
        out = d.group_by(["country", "add_confirmed"]).agg(agg).join(n, on="country") \
               .with_columns((pl.col("n") / pl.col("n_s1")).round(3).alias("per_S1")).drop("n_s1")
        print(f"{split.upper()} swap class by add_confirmed (1 = a confident other-source match has the added words, "
              f"0 = none does, -1 = no confident match):")
        print(out.sort(["country", "add_confirmed"]))


if __name__ == "__main__":
    main()
