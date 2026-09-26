#!/usr/bin/env python3
"""Label audit, stage 2: find every kind of label inconsistency in the pairs from build_pairs.py.

Four lenses, from the most assumption-free to the most general:
  1 exact duplicates     identical S2/S3 records (raw text, and after normalisation) that carry
                         different labels or different S1s; duplicate S1 records
  2 difference patterns  every (country, source, name change, address change, number change)
                         pattern; pairs whose label disagrees with their pattern's majority
  3 model                LightGBM on all pair features plus evidence from the added and dropped
                         words (no row sees its own label), scored out-of-fold; labels it finds implausible
  4 assignment anomalies records that are an exact copy of a different S1 than their own, or of
                         an S1 while unmatched; matches that share nothing with their S1

Writes eda_output/label_audit/: pattern_summary.tsv, flagged_pairs.tsv, duplicate_conflicts.tsv,
s1_duplicates.tsv, assignment_anomalies.tsv, audit.json
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import log_loss, roc_auc_score

from wordfeats import NAMES as WF_NAMES, encode_train_and_test, load_tokens, top_words as _top_words

HERE = Path(__file__).resolve().parent
W = HERE / "work"
rng = np.random.default_rng(7)
t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:6.0f}s] {msg}", flush=True)


# keep in sync with build_pairs.py
NAME_DIFF = ["identical", "legal form / filler only", "typo only", "typo + legal form / filler",
             "one word added", "2+ words added", "one word dropped", "2+ words dropped",
             "words replaced", "nothing in common",
             "other script: same words", "other script: legal form / filler only", "other script: words added",
             "other script: words dropped", "other script: words replaced", "other script: nothing in common"]
ADDR_DIFF = ["identical", "only numbers differ", "typo in words", "typo in words + numbers differ",
             "words added", "words dropped", "words replaced", "nothing in common",
             "record address empty", "S1 address empty"]
NUM_REL = ["same numbers", "number added", "number dropped", "number cut short", "number: one character changed",
           "number nearby (<=50 apart)", "number far-off", "n/a"]
COUNTRY = ["US", "India"]
FEATS = ["name_tsr", "name_ratio", "name_jac", "n_name_add", "n_name_drop", "n_name_typo", "name_minor",
         "name_len_s1", "name_len_rec", "addr_tsr", "addr_ratio", "addr_jac", "n_addr_add", "n_addr_drop",
         "n_addr_typo", "n_num_s1", "n_num_rec", "num_logdiff", "num_lev", "n_num_add", "n_num_drop"]
CATS = ["country", "src", "nd", "ad", "rel"]
WORDCOLS = ["name_added", "name_dropped", "addr_added", "addr_dropped"]
TEXT = ["s1_id", "rec_id", "s1_name", "rec_name", "s1_addr", "rec_addr"] + WORDCOLS
audit = {}


def pct(a, b):
    return round(100.0 * a / b, 2) if b else None


def take_rows(path, columns, rows):
    """Rows (by position) from a parquet file, read batch by batch to stay light on memory."""
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    out, start = [], 0
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=500_000):
        n = batch.num_rows
        lo, hi = np.searchsorted(rows, [start, start + n])
        if hi > lo:
            out.append(batch.take(rows[lo:hi] - start).to_pandas())
        start += n
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=columns)


def pattern_name(nd, ad, rel):
    return f"name: {NAME_DIFF[nd]} | address: {ADDR_DIFF[ad]} | numbers: {NUM_REL[rel]}"


def f05_macro(true_size, fn, fp):
    """Macro F0.5 over all S1 when fn[s1] true matches are missed and fp[s1] wrong records added."""
    df = true_size.to_frame("T").join(fn.rename("fn")).join(fp.rename("fp")).fillna(0)
    T, FN, FP = df["T"].values, df["fn"].values, df["fp"].values
    TP = T - FN
    pred = TP + FP
    denom = pred + 0.25 * T
    score = np.where(T == 0, (pred == 0).astype(float), np.where(denom > 0, 1.25 * TP / np.maximum(denom, 1e-9), 0.0))
    touched = (FN + FP) > 0
    return float(score.mean()), int(touched.sum()), float(score[touched].mean()) if touched.any() else 1.0


# ================================================================ load
P = pq.read_table(W / "pairs.parquet", columns=CATS + ["label", "reach", "s1", "rec"] + FEATS).to_pandas()
N = len(P)
y = P["label"].to_numpy(np.int8)
comp = P["reach"].to_numpy() > 0
S1 = pq.read_table(W / "s1.parquet").to_pandas()
true_size = S1.set_index("s1")["true_size"].astype(np.int64)
R = pq.read_table(W / "records.parquet", columns=["country", "src", "label", "rec", "true_s1", "sig_norm", "sig_raw",
                                                   "n_ident_s1", "true_is_ident", "ident_s1", "paired"]).to_pandas()
log(f"loaded {N:,} pairs, {len(R):,} records, {len(S1):,} S1 records")

audit["overview"] = {
    "s2_s3_records": int(len(R)),
    "matched_records": int((R.label == 1).sum()),
    "unmatched_records": int((R.label == 0).sum()),
    "unmatched_without_lookalike": int(((R.label == 0) & (R.paired == 0)).sum()),
    "pairs": N,
    "comparable_pairs": int(comp.sum()),
    "comparable_matches": int((comp & (y == 1)).sum()),
    "comparable_non_matches": int((comp & (y == 0)).sum()),
    "matches_sharing_no_key_with_their_s1": int((~comp & (y == 1)).sum()),
    "s1_records": int(len(S1)),
    "s1_singletons": int((S1.true_size == 0).sum()),
}

# ================================================================ lens 1: exact duplicates
L1 = {}
dup_rows = []
for sig, label_name in (("sig_raw", "identical raw text"), ("sig_norm", "identical after normalisation")):
    keys = ["country", sig]
    D = R.loc[R.duplicated(keys, keep=False), keys + ["label", "true_s1", "rec", "src"]].copy()
    g = D.groupby(keys, sort=False)
    D["gid"] = g.ngroup()
    lo, hi = g["true_s1"].transform("min"), g["true_s1"].transform("max")
    distinct = D[D.true_s1 >= 0].groupby("gid")["true_s1"].nunique()
    D["n_s1"] = D["gid"].map(distinct).fillna(0).astype(int)
    D["mixed"] = (lo < 0) & (hi >= 0)          # some copies matched, some unmatched
    D["multi"] = D["n_s1"] >= 2                # copies matched to different S1 records
    groups = D.groupby("gid").agg(mixed=("mixed", "first"), multi=("multi", "first"), size=("rec", "size"))
    L1[label_name] = {
        "duplicate_groups": int(len(groups)),
        "records_in_duplicate_groups": int(len(D)),
        "groups_some_matched_some_unmatched": int(groups.mixed.sum()),
        "records_in_mixed_groups": int(groups.loc[groups.mixed, "size"].sum()),
        "groups_matched_to_different_s1": int(groups.multi.sum()),
        "records_in_multi_s1_groups": int(groups.loc[groups.multi, "size"].sum()),
    }
    bad = D[D.mixed | D.multi].copy()
    bad["fingerprint"] = label_name
    bad["conflict"] = np.where(bad.mixed & bad.multi, "matched to different S1s and unmatched",
                               np.where(bad.mixed, "some copies matched, some unmatched", "matched to different S1s"))
    dup_rows.append(bad)
    log(f"lens 1 {label_name}: {L1[label_name]}")

# duplicate S1 records
S1dup = S1[S1.duplicated(["country", "sig_norm"], keep=False)].copy()
S1dup["gid"] = S1dup.groupby(["country", "sig_norm"], sort=False).ngroup()
sg = S1dup.groupby("gid").agg(size=("s1", "size"), singles=("true_size", lambda s: int((s == 0).sum())),
                              owners=("true_size", lambda s: int((s > 0).sum())))
L1["duplicate_s1_records"] = {
    "groups_identical_after_normalisation": int(len(sg)),
    "s1_records_in_them": int(sg["size"].sum()),
    "groups_with_raw_identical_text": int(S1[S1.duplicated(["country", "sig_raw"], keep=False)]
                                          .groupby(["country", "sig_raw"]).ngroups),
    "groups_where_one_copy_is_a_singleton_and_another_has_matches": int(((sg.singles > 0) & (sg.owners > 0)).sum()),
    "groups_where_several_copies_have_matches": int((sg.owners >= 2).sum()),
}
S1dup.drop(columns=["sig_norm", "sig_raw"]).sort_values(["gid", "s1"]).to_csv(HERE / "s1_duplicates.tsv", sep="\t", index=False)
audit["lens1_exact_duplicates"] = L1

dup = pd.concat(dup_rows, ignore_index=True)
if len(dup):
    rows = R.index[R.rec.isin(set(dup.rec))].to_numpy()
    txt = take_rows(W / "records.parquet", ["rec", "rec_id", "name", "addr"], rows)
    dup = dup.merge(txt, on="rec", how="left")
    # conflicts where every copy has an empty address are identical names with nothing else to tell them apart
    dup["addr_empty"] = dup["addr"].fillna("").str.strip() == ""
    ge = dup.groupby(["fingerprint", "conflict", "gid"])["addr_empty"].all().reset_index()
    for fp_name, sub in ge.groupby("fingerprint"):
        L1[fp_name]["conflict_groups_by_address"] = {
            conf: {"address empty": int(s2["addr_empty"].sum()), "address present": int((~s2["addr_empty"]).sum())}
            for conf, s2 in sub.groupby("conflict")}
    s1map = S1.set_index("s1")["s1_id"]
    dup["assigned_s1"] = dup.true_s1.map(s1map).fillna("unmatched")
    dup[["fingerprint", "conflict", "gid", "rec_id", "src", "label", "assigned_s1", "name", "addr"]] \
        .sort_values(["fingerprint", "gid", "assigned_s1"]).to_csv(HERE / "duplicate_conflicts.tsv", sep="\t", index=False)
examples = {"lens1": []}
if len(dup):
    # random conflict groups, mostly ones whose copies have an address
    picks = []
    for want_empty, k in ((False, 12), (True, 4)):
        sub = dup[dup.addr_empty == want_empty]
        gids = sub[["fingerprint", "gid"]].drop_duplicates()
        gids = gids.sample(min(k, len(gids)), random_state=7)
        picks.append(sub.merge(gids, on=["fingerprint", "gid"]))
    examples["lens1"] = pd.concat(picks).sort_values(["addr_empty", "fingerprint", "gid"])[
        ["fingerprint", "conflict", "gid", "rec_id", "assigned_s1", "name", "addr"]].to_dict("records")
del dup, dup_rows

# ================================================================ lens 4: assignment anomalies
wrong_twin = (R.label == 1) & (R.n_ident_s1 > 0) & (R.true_is_ident == 0)
unmatched_twin = (R.label == 0) & (R.n_ident_s1 > 0)
ut = R.loc[unmatched_twin, "ident_s1"].map(true_size)
nd, ad, rel = P["nd"].to_numpy(), P["ad"].to_numpy(), P["rel"].to_numpy()
nothing = (y == 1) & np.isin(nd, [9, 15]) & ((P["addr_jac"].to_numpy() < 0.25) | np.isin(ad, [7, 8]))
L4 = {
    "matched_record_is_exact_copy_of_a_different_s1": int(wrong_twin.sum()),
    "unmatched_record_is_exact_copy_of_an_s1": int(unmatched_twin.sum()),
    "...where_that_s1_is_a_singleton": int((ut == 0).sum()),
    "...where_that_s1_has_other_matches": int((ut > 0).sum()),
    "matches_sharing_no_name_word_and_little_address": int(nothing.sum()),
}
audit["lens4_assignment"] = L4
anom = []
for label, mask in (("record is an exact copy of a different S1 than its own", wrong_twin),
                    ("unmatched record is an exact copy of an S1", unmatched_twin)):
    sub = R.loc[mask, ["rec", "label", "true_s1", "ident_s1"]].copy()
    sub["anomaly"] = label
    anom.append(sub)
anom = pd.concat(anom, ignore_index=True)
if len(anom):
    rows = R.index[R.rec.isin(set(anom.rec))].to_numpy()
    anom = anom.merge(take_rows(W / "records.parquet", ["rec", "rec_id", "name", "addr"], rows), on="rec", how="left")
    s1t = S1.set_index("s1")[["s1_id", "name", "addr"]]
    anom = anom.join(s1t.add_prefix("assigned_s1_"), on="true_s1").join(s1t.add_prefix("lookalike_s1_"), on="ident_s1")
    anom.drop(columns=["rec", "true_s1", "ident_s1"]).to_csv(HERE / "assignment_anomalies.tsv", sep="\t", index=False)
examples["lens4"] = anom.groupby("anomaly").head(8).drop(columns=["rec"], errors="ignore").to_dict("records") if len(anom) else []
log(f"lens 4: {L4}")

del R, anom

# ================================================================ lens 2: difference patterns
Pc = P.loc[comp, CATS + ["label"]]
gid = Pc.groupby(CATS, sort=False).ngroup().to_numpy()
size = np.bincount(gid)
matches = np.bincount(gid, weights=Pc["label"].to_numpy()).astype(np.int64)
majority = (matches * 2 >= size).astype(np.int8)
against = np.where(majority == 1, size - matches, matches)
_, first = np.unique(gid, return_index=True)
pat = Pc.iloc[first][CATS].reset_index(drop=True)
pat["pairs"], pat["matches"] = size, matches
pat["non_matches"] = size - matches
pat["match_rate_pct"] = np.round(100 * matches / size, 2)
pat["majority"] = np.where(majority == 1, "match", "non-match")
pat["against_majority"] = against
pat["minority_share_pct"] = np.round(100 * against / size, 2)
pat["pattern"] = [pattern_name(a, b, c) for a, b, c in zip(pat.nd, pat.ad, pat.rel)]
pat["country"] = pat.country.map(dict(enumerate(COUNTRY)))
pat["source"] = "S" + pat.src.astype(str)

row_major = majority[gid]
pattern_flag = np.zeros(N, dtype=bool)
pattern_flag[comp] = Pc["label"].to_numpy() != row_major
row_rate = np.full(N, np.nan, dtype=np.float32)
row_rate[comp] = (matches / size)[gid]
row_gid = np.full(N, -1, dtype=np.int64)
row_gid[comp] = gid
pat.sort_values("against_majority", ascending=False)[
    ["country", "source", "pattern", "pairs", "matches", "non_matches", "match_rate_pct", "majority",
     "against_majority", "minority_share_pct"]].to_csv(HERE / "pattern_summary.tsv", sep="\t", index=False)

tot = int(size.sum())
share = against / size
L2 = {
    "patterns": int(len(size)),
    "pairs_against_their_pattern_majority": int(against.sum()),
    "pct_of_comparable_pairs": pct(int(against.sum()), tot),
    "pct_pairs_in_patterns_with_minority_over_1pct": pct(int(size[share > 0.01].sum()), tot),
    "pct_pairs_in_patterns_with_minority_over_5pct": pct(int(size[share > 0.05].sum()), tot),
    "pct_pairs_in_patterns_with_minority_over_20pct": pct(int(size[share > 0.20].sum()), tot),
    "patterns_with_both_labels": int(((matches > 0) & (matches < size)).sum()),
}
# the same patterns, pooled over country and source
coarse = pat.groupby("pattern", sort=False)[["pairs", "matches", "non_matches"]].sum().reset_index()
coarse["match_rate_pct"] = np.round(100 * coarse.matches / coarse.pairs, 2)
coarse["against_majority"] = np.minimum(coarse.matches, coarse.non_matches)
L2["top_patterns_pooled"] = coarse.sort_values("against_majority", ascending=False).head(40).to_dict("records")


def view(mask):
    m = mask & comp
    return {"pairs": int(m.sum()), "matches": int((m & (y == 1)).sum()), "non_matches": int((m & (y == 0)).sum()),
            "against_pattern_majority": int((m & pattern_flag).sum())}


typo_like = (np.isin(nd, [0, 2]) & (np.isin(ad, [0, 2]) | (np.isin(ad, [1, 3]) & np.isin(rel, [3, 4])))
             & ~((nd == 0) & (ad == 0)))
L2["reported_cases"] = {
    "only the house/plot number differs": view(ad == 1),
    "...and the name is identical": view((ad == 1) & (nd == 0)),
    "only the name differs (address identical or a word typo)": view((nd != 0) & np.isin(ad, [0, 2])),
    "...and the address is fully identical": view((nd != 0) & (ad == 0)),
    "only typo-like differences": view(typo_like),
    "name and address identical": view((nd == 0) & (ad == 0)),
}
by_view = {}
for label, mask in (("only numbers differ, by name change", ad == 1), ("address identical, by name change", ad == 0)):
    rows = []
    for k, nm in enumerate(NAME_DIFF):
        v = view(mask & (nd == k))
        if v["pairs"]:
            rows.append({"name change": nm, **v, "match_rate_pct": pct(v["matches"], v["pairs"])})
    by_view[label] = rows
rows = []
for k, nm in enumerate(NUM_REL):
    v = view((ad == 1) & (nd == 0) & (rel == k))
    if v["pairs"]:
        rows.append({"number change": nm, **v, "match_rate_pct": pct(v["matches"], v["pairs"])})
by_view["name identical, only numbers differ, by number change"] = rows
L2["breakdowns"] = by_view
audit["lens2_patterns"] = L2
log(f"lens 2: {len(size)} patterns, {against.sum():,} pairs against majority")

# ================================================================ lens 3: model, out-of-fold
fold = ((P["s1"].to_numpy().astype(np.uint64) * np.uint64(2654435761)) >> np.uint64(16)) % np.uint64(5)
fold = fold.astype(np.int8)
X_base = np.column_stack([P[c].to_numpy(np.float32) for c in CATS + FEATS])
feat_names = CATS + FEATS + WF_NAMES
cat_idx = list(range(len(CATS)))

toks = load_tokens(W / "pairs.parquet", N)
log("word lists encoded")


params = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
              num_threads=16, verbose=-1, max_cat_to_onehot=12, seed=7)
p = np.full(N, np.nan, dtype=np.float32)
importance = np.zeros(len(feat_names))
for k in range(5):
    train_mask = comp & (fold != k)
    WF = encode_train_and_test(toks, y, fold, train_mask, fold == k, N)   # no row sees its own label
    tr_idx = np.flatnonzero(train_mask)
    if len(tr_idx) > 3_000_000:
        tr_idx = np.sort(rng.choice(tr_idx, 3_000_000, replace=False))
    ds = lgb.Dataset(np.hstack([X_base[tr_idx], WF[tr_idx]]), label=y[tr_idx], feature_name=feat_names,
                     categorical_feature=cat_idx, free_raw_data=True)
    booster = lgb.train(params, ds, num_boost_round=400)
    te_idx = np.flatnonzero(fold == k)
    p[te_idx] = booster.predict(np.hstack([X_base[te_idx], WF[te_idx]]))
    importance += booster.feature_importance("gain")
    del WF, ds, booster
    log(f"fold {k} done")

pcomp, ycomp = p[comp], y[comp]
t1 = float(pcomp[ycomp == 1].mean())
t0_ = float((1 - pcomp[ycomp == 0]).mean())
conf1 = pcomp >= t1
conf0 = (1 - pcomp) >= t0_
pick1 = conf1 & (~conf0 | ((pcomp - t1) >= ((1 - pcomp) - t0_)))
pick0 = conf0 & ~pick1
cl_issue = ((ycomp == 0) & pick1) | ((ycomp == 1) & pick0)
model_flag = np.zeros(N, dtype=bool)
model_flag[comp] = ((ycomp == 1) & (pcomp < 0.02)) | ((ycomp == 0) & (pcomp > 0.98))
# keep every pair's out-of-fold probability and flags, so any single pair can be explained later
np.save(W / "oof_p.npy", p)
np.save(W / "pattern_flag.npy", pattern_flag)
np.save(W / "pattern_rate.npy", row_rate)
np.save(W / "model_flag.npy", model_flag)
L3 = {
    "auc": round(float(roc_auc_score(ycomp, pcomp)), 5),
    "log_loss": round(float(log_loss(ycomp, np.clip(pcomp, 1e-6, 1 - 1e-6))), 5),
    "brier": round(float(np.mean((pcomp - ycomp) ** 2)), 5),
    "pct_pairs_uncertain_0.2_to_0.8": pct(int(((pcomp > 0.2) & (pcomp < 0.8)).sum()), len(pcomp)),
    "pct_pairs_uncertain_0.05_to_0.95": pct(int(((pcomp > 0.05) & (pcomp < 0.95)).sum()), len(pcomp)),
    "matches_model_rejects_p_below_0.02": int(((ycomp == 1) & (pcomp < 0.02)).sum()),
    "non_matches_model_accepts_p_above_0.98": int(((ycomp == 0) & (pcomp > 0.98)).sum()),
    "confident_learning_label_issues": int(cl_issue.sum()),
    "confident_learning_pct": pct(int(cl_issue.sum()), len(pcomp)),
    "confident_learning_thresholds": {"match": round(t1, 4), "non_match": round(t0_, 4)},
    "feature_importance_gain_pct": {n: round(100 * v / importance.sum(), 2)
                                    for n, v in sorted(zip(feat_names, importance), key=lambda kv: -kv[1])[:15]},
}
# which patterns and which words the model flags
fl = pd.DataFrame({"nd": nd[model_flag], "ad": ad[model_flag], "rel": rel[model_flag], "label": y[model_flag]})
fl["pattern"] = [pattern_name(a, b, c) for a, b, c in zip(fl.nd, fl.ad, fl.rel)]
L3["top_patterns_matches_rejected"] = fl[fl.label == 1].pattern.value_counts().head(15).to_dict()
L3["top_patterns_non_matches_accepted"] = fl[fl.label == 0].pattern.value_counts().head(15).to_dict()


def top_words(col, mask, k=15):
    return _top_words(toks, col, mask, k)


L3["top_added_name_words"] = {
    "matches the model rejects": top_words("name_added", model_flag & (y == 1)),
    "non-matches the model accepts": top_words("name_added", model_flag & (y == 0)),
}
L3["top_dropped_name_words"] = {
    "matches the model rejects": top_words("name_dropped", model_flag & (y == 1)),
    "non-matches the model accepts": top_words("name_dropped", model_flag & (y == 0)),
}
audit["lens3_model"] = L3
log(f"lens 3: {L3['auc']=}, flags {int(model_flag.sum()):,}")

# ================================================================ score impact
s1_arr = P["s1"].to_numpy()


def per_s1(mask):
    return pd.Series(s1_arr[mask]).value_counts()
maj_row = np.zeros(N, dtype=np.int8)
maj_row[comp] = row_major
imp = {}
imp["pattern_majority_rule"] = f05_macro(true_size, per_s1(comp & (y == 1) & (maj_row == 0)),
                                         per_s1(comp & (y == 0) & (maj_row == 1)))
imp["model_threshold_0.5"] = f05_macro(true_size, per_s1(comp & (y == 1) & (p < 0.5)),
                                       per_s1(comp & (y == 0) & (p >= 0.5)))
audit["score_impact"] = {k: {"macro_f05_all_s1": round(v[0], 4), "s1_touched": v[1],
                             "mean_f05_on_touched_s1": round(v[2], 4)} for k, v in imp.items()}
log(f"impact: {audit['score_impact']}")

# ================================================================ flagged pairs + examples
flag = pattern_flag | model_flag
audit["flagged_pairs_total"] = int(flag.sum())
out_path = HERE / "flagged_pairs.tsv"
header = True
start = 0
for batch in pq.ParquetFile(W / "pairs.parquet").iter_batches(columns=CATS + ["label"] + TEXT, batch_size=500_000):
    n = batch.num_rows
    sel = np.flatnonzero(flag[start:start + n])
    if len(sel):
        b = batch.take(sel).to_pandas()
        g = start + sel
        b.insert(0, "flagged_by", np.where(pattern_flag[g] & model_flag[g], "pattern + model",
                                           np.where(pattern_flag[g], "pattern", "model")))
        b.insert(1, "label", np.where(b.pop("label") == 1, "match", "non-match"))
        b.insert(2, "model_p_match", np.round(p[g], 4))
        b.insert(3, "pattern_match_rate_pct", np.round(100 * row_rate[g], 2))
        b.insert(4, "pattern", [pattern_name(a, c, d) for a, c, d in zip(b.nd, b.ad, b.rel)])
        b["country"] = b.country.map(dict(enumerate(COUNTRY)))
        b["src"] = "S" + b.src.astype(str)
        b.drop(columns=["nd", "ad", "rel"]).to_csv(out_path, sep="\t", index=False, header=header, mode="w" if header else "a")
        header = False
    start += n
log(f"flagged pairs written: {int(flag.sum()):,}")

# side-by-side examples for the top patterns: majority-label and against-majority pairs
top_g = np.argsort(-against)[:14]
ex = []
for g_ in top_g:
    idx = np.flatnonzero(row_gid == g_)
    maj_lab = majority[g_]
    agree = idx[y[idx] == maj_lab]
    disagree = idx[y[idx] != maj_lab]
    pick = list(rng.choice(agree, min(3, len(agree)), replace=False)) + list(rng.choice(disagree, min(3, len(disagree)), replace=False))
    ex.append((g_, pick))
all_rows = sorted({int(i) for _, pk in ex for i in pk})
txt = take_rows(W / "pairs.parquet", ["label"] + TEXT, all_rows)
txt.index = all_rows
examples["lens2"] = [{
    "pattern": pat.loc[g_, "pattern"], "country": pat.loc[g_, "country"], "source": pat.loc[g_, "source"],
    "pairs": int(size[g_]), "match_rate_pct": float(pat.loc[g_, "match_rate_pct"]),
    "rows": [{**txt.loc[int(i)].to_dict(), "label": "match" if txt.loc[int(i), "label"] == 1 else "non-match",
              "against_majority": bool(pattern_flag[int(i)]), "model_p": round(float(p[int(i)]), 3)} for i in pk],
} for g_, pk in ex]
m_idx = np.flatnonzero(model_flag)
pick = rng.choice(m_idx, min(40, len(m_idx)), replace=False) if len(m_idx) else []
txt = take_rows(W / "pairs.parquet", ["label"] + TEXT, pick)
examples["lens3"] = [{**r, "label": "match" if r["label"] == 1 else "non-match"} for r in txt.to_dict("records")]
audit["examples"] = examples
(HERE / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str))
log("done")
