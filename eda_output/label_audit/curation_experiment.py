#!/usr/bin/env python3
"""Does cleaning the training labels help? A holdout experiment on the audit's pair table.

Holdout: every pair of 20% of the S1 businesses (grouped by S1), labels untouched, as in test.
Each variant removes or re-weights a different set of training pairs, trains the same LightGBM,
and is scored on the same holdout by macro F0.5 over the holdout's S1 businesses.

  A  keep everything                                              (baseline)
  B  drop pairs labelled against their pattern's majority
  C  drop every pair in a pattern whose minority label exceeds 5%   ("clean sample")
  D  drop pairs a cross-validated model contradicts (confident learning)
  E  all hard pairs + 10% of easy ones re-weighted x10              (smaller sample, same mix)

Writes eda_output/label_audit/curation_experiment.json
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import log_loss, roc_auc_score

from wordfeats import NAMES as WF_NAMES, encode_train_and_test, load_tokens

HERE = Path(__file__).resolve().parent
W = HERE / "work"
t0 = time.time()
rng = np.random.default_rng(11)
CATS = ["country", "src", "nd", "ad", "rel"]
FEATS = ["name_tsr", "name_ratio", "name_jac", "n_name_add", "n_name_drop", "n_name_typo", "name_minor",
         "name_len_s1", "name_len_rec", "addr_tsr", "addr_ratio", "addr_jac", "n_addr_add", "n_addr_drop",
         "n_addr_typo", "n_num_s1", "n_num_rec", "num_logdiff", "num_lev", "n_num_add", "n_num_drop"]
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
              num_threads=16, verbose=-1, max_cat_to_onehot=12, seed=7)
MAX_TRAIN = 3_000_000
FEAT_NAMES = CATS + FEATS + WF_NAMES


def log(msg):
    print(f"[{time.time()-t0:6.0f}s] {msg}", flush=True)


def s1_fold(ids):
    return (((ids.astype(np.uint64) * np.uint64(2654435761)) >> np.uint64(16)) % np.uint64(5)).astype(np.int8)


# ---------------------------------------------------------------- data
P = pq.read_table(W / "pairs.parquet", columns=CATS + ["label", "reach", "s1"] + FEATS).to_pandas()
N = len(P)
y = P["label"].to_numpy(np.int8)
comp = P["reach"].to_numpy() > 0
fold = s1_fold(P["s1"].to_numpy())
hold = fold == 0                       # every pair of the holdout businesses, comparable or not
trn = comp & ~hold                     # training candidates
X = np.column_stack([P[c].to_numpy(np.float32) for c in CATS + FEATS])

# patterns, measured on training rows only
gid = P.loc[trn, CATS].groupby(CATS, sort=False).ngroup().to_numpy()
size = np.bincount(gid)
m = np.bincount(gid, weights=y[trn])
maj = (m * 2 >= size).astype(np.int8)
against = np.zeros(N, bool)
against[trn] = y[trn] != maj[gid]
mixed = np.zeros(N, bool)
mixed[trn] = (np.minimum(m, size - m) / size)[gid] > 0.05

S1 = pq.read_table(W / "s1.parquet", columns=["s1", "true_size"]).to_pandas()
H = S1[s1_fold(S1["s1"].to_numpy()) == 0].reset_index(drop=True)
T = H["true_size"].to_numpy()
pair_h = pd.Index(H["s1"]).get_indexer(P["s1"].to_numpy())
s1_ids = P["s1"].to_numpy()
del P
toks = load_tokens(W / "pairs.parquet", N)
hold_idx = np.flatnonzero(hold)
log(f"{N:,} pairs; train candidates {trn.sum():,}; holdout pairs {hold.sum():,} over {len(H):,} S1 businesses")


def macro_f05(p_hold, t):
    """Macro F0.5 over every holdout S1 business, predicting pairs with p >= t as matches."""
    sel = p_hold >= t
    ph_, yh = pair_h[hold_idx], y[hold_idx]
    tp = np.bincount(ph_[sel & (yh == 1)], minlength=len(H))
    pp = np.bincount(ph_[sel], minlength=len(H))
    f = np.where(T == 0, (pp == 0).astype(float), 1.25 * tp / np.maximum(pp + 0.25 * T, 1e-9))
    return float(f.mean())


def fit_predict(train_mask, test_mask, weights=None):
    """Leak-free word features, LightGBM on up to MAX_TRAIN training rows, predictions for test rows."""
    WF = encode_train_and_test(toks, y, fold, train_mask, test_mask, N)
    idx = np.flatnonzero(train_mask)
    if len(idx) > MAX_TRAIN:
        idx = np.sort(rng.choice(idx, MAX_TRAIN, replace=False))
    ds = lgb.Dataset(np.hstack([X[idx], WF[idx]]), label=y[idx], weight=None if weights is None else weights[idx],
                     feature_name=FEAT_NAMES, categorical_feature=list(range(len(CATS))), free_raw_data=True)
    booster = lgb.train(PARAMS, ds, num_boost_round=400)
    te = np.flatnonzero(test_mask)
    return te, booster.predict(np.hstack([X[te], WF[te]]))


# ---------------------------------------------------------------- confident learning inside the training set
p_inner = np.full(N, np.nan)
half1 = trn & np.isin(fold, [1, 2])
half2 = trn & np.isin(fold, [3, 4])
for a, b in ((half1, half2), (half2, half1)):
    te, pr = fit_predict(a, b)
    p_inner[te] = pr
log("inner cross-validated predictions done")
pt, yt = p_inner[trn], y[trn]
t1, t0_ = pt[yt == 1].mean(), (1 - pt[yt == 0]).mean()
pick1 = (pt >= t1) & (((1 - pt) < t0_) | ((pt - t1) >= ((1 - pt) - t0_)))
pick0 = ((1 - pt) >= t0_) & ~pick1
issue = np.zeros(N, bool)
issue[trn] = ((yt == 0) & pick1) | ((yt == 1) & pick0)
easy = trn & ((p_inner <= 0.002) | (p_inner >= 0.998))
keep_easy = easy & (rng.random(N) < 0.10)
w_e = np.ones(N, np.float32)
w_e[keep_easy] = 10.0

variants = [
    ("A  keep everything", trn, None),
    ("B  drop pairs against their pattern's majority", trn & ~against, None),
    ("C  drop every pattern with >5% minority label", trn & ~mixed, None),
    ("D  drop pairs the model contradicts", trn & ~issue, None),
    ("E  hard pairs + 10% of easy pairs, weighted x10", (trn & ~easy) | keep_easy, w_e),
]
results = []
yh, comp_h = y[hold_idx], comp[hold_idx]
for name, keep, weights in variants:
    te, ph = fit_predict(keep, hold, weights)
    assert np.array_equal(te, hold_idx)
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    scores = [macro_f05(ph, t) for t in grid]
    band = comp_h & (ph > 0.2) & (ph < 0.8)
    res = {
        "variant": name,
        "training_pairs": int(keep.sum()),
        "removed_from_training": int(trn.sum() - keep.sum()),
        "holdout_auc": round(float(roc_auc_score(yh[comp_h], ph[comp_h])), 5),
        "holdout_log_loss": round(float(log_loss(yh[comp_h], np.clip(ph[comp_h], 1e-6, 1 - 1e-6))), 5),
        "macro_f05_at_0.5": round(macro_f05(ph, 0.5), 5),
        "macro_f05_best_threshold": round(max(scores), 5),
        "best_threshold": float(grid[int(np.argmax(scores))]),
        "uncertain_band_predicted_vs_actual_match_rate": [round(float(ph[band].mean()), 4), round(float(yh[band].mean()), 4)]
        if band.any() else None,
    }
    results.append(res)
    log(json.dumps(res))

out = {"holdout_s1_businesses": int(len(H)), "holdout_pairs": int(hold.sum()),
       "confident_learning_issues_in_training": int(issue.sum()), "results": results}
(HERE / "curation_experiment.json").write_text(json.dumps(out, indent=2))
print("\n" + pd.DataFrame(results).drop(columns=["uncertain_band_predicted_vs_actual_match_rate"]).to_string(index=False))
log("done")
