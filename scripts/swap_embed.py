"""Word-meaning correction for same-slot word swaps (France's weak spot), zero-shot from a multilingual
embedding model (model2vec potion-multilingual-128M, MIT).

For uncertain pairs (calibrated p in [LO, HI]) whose core name differs by exactly one word added and one
dropped ("amis" -> "club"), the cosine similarity of the two words says whether the swap keeps the meaning
(typo, synonym, translation) or brings a new word. A small logistic model on fold 9 (India/US labels)
learns how to adjust p from it; fold 5 stays clean for scripts/final_decode.py. The embedding is
language-independent, so the same correction applies to French pairs.

  python scripts/swap_embed.py --tag b5sb3 --out b5sb3e
  python scripts/final_decode.py --tags b5sb3,b5sb3e --out output_e --france_thr 0.85 --france_t2 0.70
"""
import argparse
import os
import sys

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
KEY = ["s1_id", "cand_id"]
FOLD = (pl.col("s1_id").str.slice(3).cast(pl.Int64) % 10).cast(pl.Int8)
MODEL = "minishlab/potion-multilingual-128M"


def names(split: str) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(f"{DATA}/norm/{split}_s{k}.parquet", columns=["id", "name_core"])
                      for k in (1, 2, 3)])


def swap_features(df: pl.DataFrame, nm: pl.DataFrame) -> pl.DataFrame:
    df = (df.join(nm.rename({"id": "s1_id", "name_core": "a"}), on="s1_id", how="left")
            .join(nm.rename({"id": "cand_id", "name_core": "b"}), on="cand_id", how="left"))
    df = df.with_columns(pl.col("a").fill_null("").str.split(" ").list.unique().alias("A"),
                         pl.col("b").fill_null("").str.split(" ").list.unique().alias("B"))
    df = df.with_columns(pl.col("B").list.set_difference("A").list.eval(pl.element().filter(pl.element() != "")).alias("add"),
                         pl.col("A").list.set_difference("B").list.eval(pl.element().filter(pl.element() != "")).alias("drop"))
    df = df.with_columns(((pl.col("add").list.len() == 1) & (pl.col("drop").list.len() == 1)).alias("swap"))
    return df.with_columns(pl.when(pl.col("swap")).then(pl.col("add").list.first()).alias("w_add"),
                           pl.when(pl.col("swap")).then(pl.col("drop").list.first()).alias("w_drop")) \
             .drop(["a", "b", "A", "B", "add", "drop"])


def add_cos(df: pl.DataFrame, model) -> pl.DataFrame:
    vocab = sorted(set(df["w_add"].drop_nulls().to_list()) | set(df["w_drop"].drop_nulls().to_list()))
    if not vocab:
        return df.with_columns(pl.lit(0.0).alias("cos"))
    e = model.encode(vocab, batch_size=4096).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True).clip(1e-9)
    idx = {w: i for i, w in enumerate(vocab)}
    ia = np.array([idx.get(w, -1) if w is not None else -1 for w in df["w_add"].to_list()])
    idr = np.array([idx.get(w, -1) if w is not None else -1 for w in df["w_drop"].to_list()])
    ok = (ia >= 0) & (idr >= 0)
    cos = np.zeros(len(df), dtype=np.float32)
    cos[ok] = (e[ia[ok]] * e[idr[ok]]).sum(1)
    return df.with_columns(pl.Series("cos", cos))


def design(df: pl.DataFrame) -> np.ndarray:
    pc = df["pc"].to_numpy().clip(1e-4, 1 - 1e-4)
    lg = np.log(pc / (1 - pc))
    sw = df["swap"].to_numpy().astype(np.float32)
    cs = df["cos"].to_numpy()
    return np.column_stack([lg, sw, sw * cs, sw * lg, sw * cs * lg])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b5sb3")
    ap.add_argument("--out", default="b5sb3e")
    ap.add_argument("--lo", type=float, default=0.05)
    ap.add_argument("--hi", type=float, default=0.97)
    a = ap.parse_args()
    from model2vec import StaticModel
    model = StaticModel.from_pretrained(MODEL)

    tr = pl.read_parquet(f"{DATA}/pred/train_{a.tag}.parquet", columns=KEY + ["p"]).with_columns(FOLD.alias("fold"))
    tr = tr.filter(pl.col("fold").is_in([5, 9]))
    truth = pl.read_parquet(f"{DATA}/parquet/train_pairs.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
    tr = tr.join(truth, on=KEY, how="left").with_columns(pl.col("y").fill_null(0))
    c9 = tr.filter(pl.col("fold") == 9)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(c9["p"].to_numpy(), c9["y"].to_numpy())
    tr = tr.with_columns(pl.Series("pc", iso.predict(tr["p"].to_numpy()).astype(np.float32)))
    band = (pl.col("pc") >= a.lo) & (pl.col("pc") <= a.hi)
    tb = add_cos(swap_features(tr.filter(band), names("train")), model)
    fit = tb.filter(pl.col("fold") == 9)
    lr = LogisticRegression(C=1.0, max_iter=500).fit(design(fit), fit["y"].to_numpy())
    print(f"train band pairs {tb.height:,} (swap share {tb['swap'].mean():.3f}); LR coef "
          f"{np.round(lr.coef_[0], 3).tolist()} (logit, swap, swap*cos, swap*logit, swap*cos*logit)", flush=True)
    for name, d in (("fold 9", fit), ("fold 5", tb.filter(pl.col("fold") == 5))):
        s = d.filter(pl.col("swap"))
        if s.height:
            y = s["y"].to_numpy()
            print(f"  {name} swap pairs {s.height:,}: true rate {y.mean():.3f}; mean cos | true {s.filter(pl.col('y') == 1)['cos'].mean():.3f} "
                  f"| false {s.filter(pl.col('y') == 0)['cos'].mean():.3f}", flush=True)

    def adjust(d: pl.DataFrame, band_d: pl.DataFrame) -> pl.DataFrame:
        pn = band_d.select(KEY).with_columns(pl.Series("pn", lr.predict_proba(design(band_d))[:, 1].astype(np.float32)))
        return d.join(pn, on=KEY, how="left").with_columns(pl.coalesce("pn", "pc").alias("p")).select(KEY + ["p"])

    adjust(tr, tb).write_parquet(f"{DATA}/pred/train_{a.out}.parquet")
    te = pl.read_parquet(f"{DATA}/pred/test_{a.tag}.parquet", columns=KEY + ["p"])
    te = te.with_columns(pl.Series("pc", iso.predict(te["p"].to_numpy()).astype(np.float32)))
    teb = add_cos(swap_features(te.filter(band), names("test")), model)
    s1 = pl.read_parquet(f"{DATA}/parquet/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    st = teb.join(s1, on="s1_id").group_by("country").agg(pl.len(), pl.col("swap").mean(), pl.col("cos").filter(pl.col("swap")).mean())
    print("test band pairs by country:", st.sort("country").rows(), flush=True)
    adjust(te, teb).write_parquet(f"{DATA}/pred/test_{a.out}.parquet")
    print(f"wrote data/pred/{{train,test}}_{a.out}.parquet (p calibrated; final_decode re-calibrates on fold 9)")


if __name__ == "__main__":
    main()
