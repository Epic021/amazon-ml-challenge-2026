"""Gate for GPU session 2: does the neural scorer transfer across countries better than LightGBM?

France has no labels, so India stands in for it. The US-only mDeBERTa (gpu_session.sh s1, data/xenc/m_mdeb_us)
scored India's hardest training pairs (data/xenc/p_mdeb_us). Here LightGBM is trained on exactly the same
US rows and scores the same India rows. Two LightGBM versions:
  full    every feature. Optimistic: the word log-odds tables were built from folds 0-1 of BOTH countries,
          so India's words are known to it (France's are not)
  no_lo   without the label-based word log-odds: the France-like case, where the words are unseen
Proceed with the LLM session if the neural AUC beats no_lo on the swap class (where France loses).

  python scripts/transfer_check.py --feat_dir data/feat_b5      # CPU VM
  python scripts/transfer_check.py                               # GPU pod: data/xenc/train_feats.parquet
Exit code 0 = PASS, 1 = FAIL (gpu_session.sh auto uses it to decide on the LLM session).
"""
import sys
import argparse
import glob
import json
import os

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
X = f"{DATA}/xenc"
KEY = ["s1_id", "cand_id"]
LABEL_LO = ("add_lo_", "drop_lo_", "aadd_lo_", "adrop_lo_")
SWAP = ((pl.col("num_rel") == 1) & (pl.col("addr_tset") >= 90) & (pl.col("n_added") >= 1) & (pl.col("n_dropped") >= 1))


def metrics(y, p) -> str:
    if not 0 < y.mean() < 1:
        return "n/a"
    return f"AUC {roc_auc_score(y, p):.4f}  logloss {log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)):.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", default="", help="full feature dir; default: data/xenc/train_feats.parquet")
    ap.add_argument("--model_dir", default=f"{X}/m_mdeb_us")
    ap.add_argument("--threads", type=int, default=os.cpu_count(), help="LightGBM threads (the pod shares CPU with the GPU job)")
    ap.add_argument("--parts", default=f"{X}/p_mdeb_us")
    a = ap.parse_args()
    meta = json.load(open(f"{a.model_dir}/meta.json"))
    tr = pl.read_parquet(f"{X}/train.parquet")
    us = tr.filter(pl.col("country") == "US").head(meta["n_pairs"])         # the rows the neural model saw
    nn = pl.concat([pl.read_parquet(p) for p in glob.glob(f"{a.parts}/part-*.parquet")]).select(KEY + ["p"])
    feats = pl.scan_parquet(f"{a.feat_dir}/train.parquet/*.parquet" if a.feat_dir else f"{X}/train_feats.parquet")
    cols = [c for c in feats.collect_schema().names() if c not in KEY + ["y", "fold"]]
    us = us.select(KEY + ["y"]).join(feats.collect(), on=KEY, how="left")
    ind = nn.join(tr.select(KEY + ["y"]), on=KEY).join(feats.collect(), on=KEY, how="left")
    print(f"US training rows {us.height:,} (pos {us['y'].mean():.3f}); India scored rows {ind.height:,} "
          f"(pos {ind['y'].mean():.3f}; swap class {ind.filter(SWAP).height:,})")

    res = {"neural": ind["p"].to_numpy()}
    rng = np.random.default_rng(0)
    va = rng.random(us.height) < 0.1
    for name, use in (("full", cols), ("no_lo", [c for c in cols if not c.startswith(LABEL_LO)])):
        Xu, yu = us.select(use).to_pandas(), us["y"].to_numpy()
        m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200,
                           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1,
                           num_threads=a.threads, seed=42),
                      lgb.Dataset(Xu[~va], yu[~va]), 3000, valid_sets=[lgb.Dataset(Xu[va], yu[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        res[f"lgbm_{name}"] = m.predict(ind.select(use).to_pandas(), num_threads=a.threads)
    y, swap = ind["y"].to_numpy(), ind.select(SWAP.fill_null(False).alias("swap"))["swap"].to_numpy()
    for k, p in res.items():
        print(f"  {k:11s} all: {metrics(y, p)}   swap class: {metrics(y[swap], p[swap])}")
    better = roc_auc_score(y[swap], res["neural"][swap]) > roc_auc_score(y[swap], res["lgbm_no_lo"][swap])
    print("GATE:", "PASS - run GPU session 2" if better else "FAIL - skip session 2, keep the money")
    sys.exit(0 if better else 1)


if __name__ == "__main__":
    main()
