"""S1 of the clutch plan: can file row order or ID numbers break ties between S1 that share a name?

On train (labels): for each S2/S3 record, row/ID distance to its true S1 vs to other candidate S1.
  1) corr of relative row positions (S1 file vs record file) over true pairs
  2) adjacency of one S1's records within a source file (row gaps)
  3) the ambiguous class (empty address, identical core name, 2-3 S1 share the name, fold 5): how often
     the true S1 is the closest by row / by ID, vs chance and vs the stacker's top pick

  python scripts/order_signal.py
"""
import os

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
K = ["s1_id", "cand_id"]


def rows(k: int) -> pl.DataFrame:
    return (pl.read_parquet(f"{DATA}/parquet/train_s{k}.parquet", columns=["entity_id"]).with_row_index("row")
              .with_columns(pl.col("row").cast(pl.Int64), pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("num"),
                            pl.lit(k).alias("src")))


def main():
    s1 = rows(1).rename({"entity_id": "s1_id", "row": "s_row", "num": "s_num"}).drop("src")
    rec = pl.concat([rows(2), rows(3)]).rename({"entity_id": "cand_id", "row": "c_row", "num": "c_num"})
    n1 = s1.height
    nsrc = rec.group_by("src").len("n_src")
    tp = pl.read_parquet(f"{DATA}/parquet/train_pairs.parquet").join(s1, on="s1_id").join(rec, on="cand_id").join(nsrc, on="src")
    tp = tp.with_columns((pl.col("s_row") / n1).alias("r1"), (pl.col("c_row") / pl.col("n_src")).alias("rc"))
    print(f"1) true pairs {tp.height:,}: corr(relative row S1, relative row record) = "
          f"{np.corrcoef(tp['r1'], tp['rc'])[0, 1]:.4f}; corr(S1 id, record id) = {np.corrcoef(tp['s_num'], tp['c_num'])[0, 1]:.4f}")
    g = tp.sort(["s1_id", "src", "c_row"]).with_columns(pl.col("c_row").diff().over(["s1_id", "src"]).alias("gap")).drop_nulls("gap")
    print(f"2) same-S1 same-source row gaps: ==1 {(g['gap'] == 1).mean():.4f}, <=10 {(g['gap'] <= 10).mean():.4f}, "
          f"median {g['gap'].median():,.0f} (random ~{int(nsrc['n_src'].mean() / 3):,})")

    fold = pl.col("s1_id").str.slice(3).cast(pl.Int64) % 10
    d = pl.read_parquet(f"{DATA}/pred/train_b5sb2.parquet").filter(pl.col("fold") == 5)
    f = pl.scan_parquet(f"{DATA}/feat_b5/train.parquet/*.parquet").filter(fold == 5) \
          .select(K + ["c_addr_empty", "core_eq", "c_core_n_s1"]).collect()
    x = d.join(f, on=K).filter((pl.col("c_addr_empty") == 1) & (pl.col("core_eq") == 1)
                               & (pl.col("c_core_n_s1") >= 2) & (pl.col("c_core_n_s1") <= 3))
    x = x.join(s1, on="s1_id").join(rec, on="cand_id")
    x = x.with_columns((pl.col("c_num") - pl.col("s_num")).abs().alias("dnum"),
                       (pl.col("c_row") - pl.col("s_row")).abs().alias("drow"))
    has_true = x.group_by("cand_id").agg(pl.col("y").max().alias("_t"), pl.len().alias("k")).filter((pl.col("_t") == 1) & (pl.col("k") >= 2))
    x = x.join(has_true.select("cand_id", "k"), on="cand_id")
    x = x.with_columns(pl.col("dnum").rank("ordinal").over("cand_id").alias("r_num"),
                       pl.col("drow").rank("ordinal").over("cand_id").alias("r_row"),
                       pl.col("p").rank("ordinal", descending=True).over("cand_id").alias("r_p"))
    t = x.filter(pl.col("y") == 1)
    print(f"3) ambiguous empty-address records with >=2 candidate S1 (holdout): {t.height:,}; true S1 is closest by "
          f"ID {(t['r_num'] == 1).mean():.3f}, by row {(t['r_row'] == 1).mean():.3f}; stacker top pick "
          f"{(t['r_p'] == 1).mean():.3f}; chance {(1 / t['k']).mean():.3f}")


if __name__ == "__main__":
    main()
