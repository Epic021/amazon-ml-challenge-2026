"""Where does France lose? Label-free analysis of the blend (ours + teammate) on test, checked on the holdout.

A) Expected F0.5 per country from calibrated probabilities (Monte Carlo over Bernoulli labels).
   Validated on holdout fold 5 (expected vs actual, India/US). On test it gives the model's own
   belief about France; if the LB-implied France is far below it, France is confidently wrong
   (distribution shift), not just uncertain.
B) Uncertainty profile: grey pairs (0.2 <= r <= 0.8) per S1, expected matches per S1, per country.
C) Ours vs teammate disagreement rates per country; on the holdout, who is right when they disagree.
D) Feature profile (house-number relation, empty address, source, ...) of predicted and grey pairs,
   France vs India/US, plus France examples written to logs/france_examples.tsv.

  POLARS_MAX_THREADS=16 python scripts/france_gap.py --tag b4 --feat_dir data/feat_b4
"""
import argparse
import os
import sys

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
TUNE, HOLD = [6, 7, 8, 9], 5
THR_O, THR_T = 0.55, 0.5
FOLD = (pl.col("s1_id").str.slice(3).cast(pl.Int64) % 10).cast(pl.Int8)
KEY = ["s1_id", "cand_id"]


def iso(x, y):
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(x, y)


def soft(df: pl.DataFrame, col: str, out: str) -> pl.DataFrame:
    o = pl.col(col).clip(upper_bound=0.999) / (1 - pl.col(col).clip(upper_bound=0.999))
    return df.with_columns((o / (1 + o.sum().over("cand_id"))).cast(pl.Float32).alias(out))


def pairs(split: str, tag: str) -> pl.DataFrame:
    o = pl.read_parquet(f"{DATA}/pred/{split}_{tag}.parquet", columns=KEY + ["p"]).rename({"p": "p_o"})
    t = pl.read_parquet(f"{DATA}/friend/{'train_oof_probs' if split == 'train' else 'test_probs'}.parquet",
                        columns=KEY + ["p"]).rename({"p": "p_t"})
    if split == "train":
        o, t = (d.filter(FOLD >= HOLD) for d in (o, t))
    df = o.join(t, on=KEY, how="full", coalesce=True).with_columns(pl.col("p_o").fill_null(0.0),
                                                                   pl.col("p_t").fill_null(0.0))
    s1 = pl.read_parquet(f"{DATA}/parquet/{split}_s1.parquet", columns=["entity_id", "country"])
    df = df.join(s1.rename({"entity_id": "s1_id"}), on="s1_id", how="left").with_columns(FOLD.alias("fold"))
    if split == "train":
        tr = pl.read_parquet(f"{DATA}/parquet/train_pairs.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
        df = df.join(tr, on=KEY, how="left").with_columns(pl.col("y").fill_null(0))
    return df


def expected_f(df: pl.DataFrame, s1: pl.DataFrame, draws=16, seed=0) -> pl.DataFrame:
    """Per-S1 expected F0.5 under independent Bernoulli(r) labels over the candidate list."""
    s1 = s1.with_row_index("_i")
    d = df.filter((pl.col("r") > 1e-3) | pl.col("pred")).select(["s1_id", "r", "pred"]) \
          .join(s1.select(["entity_id", "_i"]).rename({"entity_id": "s1_id"}), on="s1_id", how="inner")
    idx = d["_i"].to_numpy().astype(np.int64)
    r, pred = d["r"].to_numpy().astype(np.float64), d["pred"].to_numpy()
    n = s1.height
    s1 = s1.drop("_i")
    S = np.bincount(idx, pred, n)
    rng = np.random.default_rng(seed)
    acc = np.zeros(n)
    for _ in range(draws):
        y = rng.random(len(r)) < r
        tp, T = np.bincount(idx, y & pred, n), np.bincount(idx, y, n)
        den = S + 0.25 * T
        acc += np.where(den == 0, 1.0, 1.25 * tp / np.maximum(den, 1e-9))
    er = np.bincount(idx, r, n)
    return s1.with_columns(pl.Series("ef", acc / draws), pl.Series("exp_matches", er), pl.Series("n_pred", S))


def actual_f(df: pl.DataFrame, s1: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    tp = df.filter(pl.col("pred") & (pl.col("y") == 1)).group_by("s1_id").len("tp")
    np_ = df.filter(pl.col("pred")).group_by("s1_id").len("np")
    nt = truth.group_by("s1_id").len("nt")
    a = s1.rename({"entity_id": "s1_id"}).join(tp, on="s1_id", how="left").join(np_, on="s1_id", how="left") \
          .join(nt, on="s1_id", how="left").fill_null(0)
    den = pl.col("np") + 0.25 * pl.col("nt")
    return a.with_columns(pl.when(den == 0).then(1.0).otherwise(1.25 * pl.col("tp") / den).alias("f"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b4")
    ap.add_argument("--feat_dir", default=f"{DATA}/feat_b4")
    ap.add_argument("--w", type=float, default=0.5, help="weight on ours (1.0 = ours alone)")
    ap.add_argument("--thr", type=float, default=0.45)
    ap.add_argument("--only_a", action="store_true", help="expected-vs-actual F only")
    a = ap.parse_args()
    W, THR = a.w, a.thr
    pl.Config.set_tbl_rows(40).set_tbl_cols(20).set_tbl_width_chars(220)

    tr = pairs("train", a.tag)
    cal = tr.filter(pl.col("fold").is_in(TUNE))
    y = cal["y"].to_numpy()
    iso_o, iso_t = iso(cal["p_o"].to_numpy(), y), iso(cal["p_t"].to_numpy(), y)

    def prep(df):
        df = df.with_columns(pl.Series("c_o", iso_o.predict(df["p_o"].to_numpy()).astype(np.float32)),
                             pl.Series("c_t", iso_t.predict(df["p_t"].to_numpy()).astype(np.float32)))
        df = df.with_columns((W * pl.col("c_o") + (1 - W) * pl.col("c_t")).alias("b"))
        df = soft(soft(soft(df, "b", "bx"), "c_o", "ox"), "c_t", "tx")
        return df.with_columns((pl.col("bx") >= THR).alias("pred"), (pl.col("ox") >= THR_O).alias("pred_o"),
                               (pl.col("tx") >= THR_T).alias("pred_t"))

    tr = prep(tr)
    cal = tr.filter(pl.col("fold").is_in(TUNE))
    iso_r = iso(cal["bx"].to_numpy(), cal["y"].to_numpy())          # p' after exclusivity -> probability
    tr = tr.with_columns(pl.Series("r", iso_r.predict(tr["bx"].to_numpy()).astype(np.float32)))
    ho = tr.filter(pl.col("fold") == HOLD)
    truth = pl.read_parquet(f"{DATA}/parquet/train_pairs.parquet").filter(FOLD == HOLD)
    s1h = pl.read_parquet(f"{DATA}/parquet/train_s1.parquet", columns=["entity_id", "country"]) \
            .filter((pl.col("entity_id").str.slice(3).cast(pl.Int64) % 10) == HOLD)

    te = prep(pairs("test", a.tag))
    te = te.with_columns(pl.Series("r", iso_r.predict(te["bx"].to_numpy()).astype(np.float32)))
    s1t = pl.read_parquet(f"{DATA}/parquet/test_s1.parquet", columns=["entity_id", "country"])

    # A) expected vs actual F0.5
    eh = expected_f(ho, s1h).join(actual_f(ho, s1h, truth).select(["s1_id", "f"]).rename({"s1_id": "entity_id"}),
                                  on="entity_id")
    et = expected_f(te, s1t)
    print("\n== A) expected F0.5 (model's belief) vs actual ==")
    print(eh.group_by("country").agg(pl.col("ef").mean().round(5).alias("expected"),
                                     pl.col("f").mean().round(5).alias("actual"),
                                     pl.col("exp_matches").mean().round(3), pl.col("n_pred").mean().round(3))
            .sort("country"))
    print("test:")
    print(et.group_by("country").agg(pl.col("ef").mean().round(5).alias("expected"),
                                     pl.col("exp_matches").mean().round(3), pl.col("n_pred").mean().round(3),
                                     (pl.col("ef") < 0.9).mean().round(4).alias("share_ef<0.9"))
            .sort("country"))
    if a.only_a:
        return

    # B) uncertainty profile
    print("\n== B) grey pairs (0.2<=r<=0.8) per S1; near-threshold predicted pairs ==")
    for name, d, s in (("holdout", ho, s1h), ("test", te, s1t)):
        g = d.group_by("country").agg(
            ((pl.col("r") >= 0.2) & (pl.col("r") <= 0.8)).sum().alias("grey"),
            (pl.col("pred") & (pl.col("r") < 0.8)).sum().alias("pred_r<0.8"),
            (~pl.col("pred") & (pl.col("r") >= 0.2)).sum().alias("rej_r>=0.2"),
            pl.len().alias("pairs"))
        g = g.join(s.group_by("country").len("n_s1"), on="country").with_columns(
            [(pl.col(c) / pl.col("n_s1")).round(4).alias(c + "/S1") for c in ("grey", "pred_r<0.8", "rej_r>=0.2", "pairs")])
        print(name)
        print(g.select(["country", "n_s1", "grey/S1", "pred_r<0.8/S1", "rej_r>=0.2/S1", "pairs/S1"]).sort("country"))

    # C) disagreement ours vs teammate
    print("\n== C) ours vs teammate: disagreements per S1 (and who is right on holdout) ==")
    for name, d, s in (("holdout", ho, s1h), ("test", te, s1t)):
        agg = [(pl.col("pred_o") & ~pl.col("pred_t")).sum().alias("ours_only"),
               (~pl.col("pred_o") & pl.col("pred_t")).sum().alias("theirs_only"),
               (pl.col("pred") & ~(pl.col("pred_o") & pl.col("pred_t"))).sum().alias("blend_took_disputed")]
        if name == "holdout":
            agg += [(pl.col("pred_o") & ~pl.col("pred_t") & (pl.col("y") == 1)).sum().alias("ours_only_true"),
                    (~pl.col("pred_o") & pl.col("pred_t") & (pl.col("y") == 1)).sum().alias("theirs_only_true")]
        g = d.group_by("country").agg(agg).join(s.group_by("country").len("n_s1"), on="country")
        g = g.with_columns([(pl.col(c) / pl.col("n_s1")).round(4).alias(c + "/S1")
                            for c in ("ours_only", "theirs_only", "blend_took_disputed")])
        if name == "holdout":
            g = g.with_columns((pl.col("ours_only_true") / pl.col("ours_only")).round(3).alias("prec_ours_only"),
                               (pl.col("theirs_only_true") / pl.col("theirs_only")).round(3).alias("prec_theirs_only"))
        print(name)
        print(g.drop(["ours_only", "theirs_only", "blend_took_disputed"]).sort("country"))

    # D) feature profile of predicted / grey pairs
    cols = ["num_rel", "c_addr_empty", "c_src_s3", "c_nonlatin", "n_s1_cands", "n_added", "addr_tset", "core_tset"]
    feat = pl.scan_parquet(f"{a.feat_dir}/test.parquet/*.parquet").select(KEY + cols)
    te_i = te.filter(pl.col("pred") | (pl.col("r") >= 0.2)).with_columns(
        pl.when(pl.col("pred") & (pl.col("r") >= 0.8)).then(pl.lit("sure_pred"))
          .when(pl.col("pred")).then(pl.lit("weak_pred")).otherwise(pl.lit("weak_rej")).alias("kind"))
    te_i = te_i.join(feat.collect(), on=KEY, how="left")
    print("\n== D) feature profile (test) by country x kind ==")
    print(te_i.group_by(["country", "kind"]).agg(
        pl.len().alias("n"), pl.col("r").mean().round(3).alias("r_mean"),
        *[(pl.col("num_rel") == v).mean().round(3).alias(f"num_rel={v}") for v in range(7)],
        pl.col("c_addr_empty").mean().round(3), pl.col("c_src_s3").mean().round(3),
        pl.col("n_added").mean().round(2), pl.col("addr_tset").mean().round(1), pl.col("core_tset").mean().round(1))
        .sort(["kind", "country"]))
    print("num_rel values present:", te_i["num_rel"].unique().sort().to_list())

    # examples: France weak predictions, weak rejections, and ours/theirs disagreements
    rec = pl.concat([pl.read_parquet(f"{DATA}/parquet/test_s{k}.parquet",
                                     columns=["entity_id", "business_name", "business_address"]) for k in (1, 2, 3)])
    fr = te_i.filter(pl.col("country") == "France")
    fr = fr.with_columns(pl.when(pl.col("pred_o") & ~pl.col("pred_t")).then(pl.lit("ours_only"))
                           .when(~pl.col("pred_o") & pl.col("pred_t")).then(pl.lit("theirs_only"))
                           .otherwise(pl.col("kind")).alias("kind"))
    ex = pl.concat([fr.filter(pl.col("kind") == k).sample(min(40, fr.filter(pl.col("kind") == k).height), seed=1)
                    for k in ("weak_pred", "weak_rej", "ours_only", "theirs_only")])
    ex = ex.join(rec.rename({"entity_id": "s1_id", "business_name": "s1_name", "business_address": "s1_addr"}),
                 on="s1_id", how="left") \
           .join(rec.rename({"entity_id": "cand_id", "business_name": "c_name", "business_address": "c_addr"}),
                 on="cand_id", how="left")
    ex.select(["kind", "r", "c_o", "c_t", "num_rel", "s1_name", "c_name", "s1_addr", "c_addr", "s1_id", "cand_id"]) \
      .sort(["kind", "r"]).write_csv(os.path.join(ROOT, "logs", "france_examples.tsv"), separator="\t")
    print("examples -> logs/france_examples.tsv", fr.group_by("kind").len().sort("kind").to_dicts())


if __name__ == "__main__":
    main()
