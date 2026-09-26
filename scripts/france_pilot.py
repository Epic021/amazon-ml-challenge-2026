"""Pilot: do retrieval-rank features break France? (france_shap.py: rank_q_word, score_numaddr, rank_q,
n_retrievers push France's same-address word-swap pairs to log-odds -4.4 vs +2.4 / +4.2 for India / US.
France's dense streets and generic names crowd the retrievers, so "found late / not found by retriever X"
means something else there.)

Small LightGBM on 10% of S1 per variant; reports holdout macro F0.5 (isotonic 8-9, soft excl, 0.55) and on
test the mean p of the swap class + predictions per S1 by country.
  V0 all features | V1 minus per-retriever scores/ranks | V2 V1 minus absolute scores (relative context only)

  python scripts/france_pilot.py --variant V1 --feat_dir data/feat_b5
"""
import argparse
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from decode import soft_excl  # noqa: E402
from metric import per_entity_f05  # noqa: E402
from train import feature_cols  # noqa: E402

DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
RET = ["word", "namechar", "addr", "numaddr"]
PER_RET = [f"{k}_{r}" for r in RET for k in ("score", "rank_q", "rank_s")]
DROP = {"V0": [],
        "V1": PER_RET + ["n_retrievers", "rank_q", "rank_s", "rank_in_s1"],
        "V2": PER_RET + ["n_retrievers", "rank_q", "rank_s", "rank_in_s1", "score", "s1_best", "c_best"]}
SUB = pl.col("s1_id").str.slice(3).cast(pl.Int64) % 1000 < 100      # 10% of S1
SWAP = ((pl.col("num_rel") == 1) & (pl.col("addr_tset") >= 90) & (pl.col("n_added") >= 1)
        & (pl.col("n_dropped") >= 1) & (pl.col("is_c_best") == 1) & (pl.col("twin_name") <= 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(DROP))
    ap.add_argument("--feat_dir", default=f"{DATA}/feat_b5")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=400)
    a = ap.parse_args()
    t0 = time.time()
    tr = pl.scan_parquet(f"{a.feat_dir}/train.parquet/*.parquet").filter(SUB).collect().to_pandas()
    cols = [c for c in feature_cols(tr) if c not in DROP[a.variant] and not c.startswith("numx_")]
    m_tr = tr.fold.isin((2, 3, 4)).values
    params = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=a.threads, verbose=-1, seed=42)
    model = lgb.train(params, lgb.Dataset(tr.loc[m_tr, cols], tr.y[m_tr]), a.rounds)
    ev = tr[tr.fold.isin((5, 8, 9))][["s1_id", "cand_id", "fold", "y"]].copy()
    ev["p"] = model.predict(tr.loc[ev.index, cols], num_threads=a.threads)
    cal = ev[ev.fold.isin((8, 9))]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(cal.p, cal.y)
    ev["p"] = iso.predict(ev.p)
    ho = soft_excl(ev[ev.fold == 5][["s1_id", "cand_id", "p"]])
    truth = pd.read_parquet(f"{DATA}/parquet/train_pairs.parquet")
    s1 = pd.read_parquet(f"{DATA}/parquet/train_s1.parquet", columns=["entity_id", "country"])
    num = s1.entity_id.str[3:].astype(np.int64)
    hold = s1[((num % 10) == 5) & ((num % 1000) < 100)]
    f = per_entity_f05(ho[ho.p >= 0.55], truth, hold.entity_id)
    by = f.groupby(hold.set_index("entity_id").country.reindex(f.index).values).mean().round(5).to_dict()
    del tr

    te = pl.scan_parquet(f"{a.feat_dir}/test.parquet/*.parquet").filter(SUB).collect()
    s1t = pl.read_parquet(f"{DATA}/parquet/test_s1.parquet", columns=["entity_id", "country"])
    te = te.join(s1t.rename({"entity_id": "s1_id"}), on="s1_id")
    te = te.with_columns(pl.Series("p", iso.predict(model.predict(te.select(cols).to_pandas(),
                                                                  num_threads=a.threads)).astype(np.float32)))
    ex = pl.from_pandas(soft_excl(te.select(["s1_id", "cand_id", "p"]).to_pandas())).rename({"p": "px"})
    te = te.join(ex, on=["s1_id", "cand_id"])
    n_s1 = s1t.filter(SUB).group_by("country").len("n_s1")
    st = te.group_by("country").agg(pl.col("p").filter(SWAP).mean().round(3).alias("swap_p"),
                                    (pl.col("px") >= 0.55).sum().alias("n_pred"),
                                    pl.col("px").sum().alias("exp")).join(n_s1, on="country")
    st = st.with_columns((pl.col("n_pred") / pl.col("n_s1")).round(3).alias("pred/S1"),
                         (pl.col("exp") / pl.col("n_s1")).round(3).alias("exp/S1")).sort("country")
    print(f"PILOT {a.variant}: {len(cols)} features, holdout macroF05={f.mean():.5f} {by} | test "
          + "; ".join(f"{r['country']}: swap_p {r['swap_p']}, pred/S1 {r['pred/S1']}, exp/S1 {r['exp/S1']}"
                      for r in st.to_dicts()) + f" ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
