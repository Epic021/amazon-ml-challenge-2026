#!/usr/bin/env python3
"""Stage 3: two-stage LightGBM, cross-fitted on two halves of the training S1 businesses.

  stage 1  pair features + word evidence                       -> p1 (out-of-fold on train)
  context  self-correction features from p1: competition for the record, the S1's other candidates,
           twins (a record in the other source with the same numbers / name / full text, and what
           p1 says about them), identical copies                                -> ctx
  stage 2  pair features + word evidence + p1 + ctx, on pairs with p1 >= PRUNE -> p2 (out-of-fold)
  decode   exclusive assignment + per-S1 expected-F0.5 (or threshold) sets, tuned on train OOF

Every training label is kept by default (variant A). Other variants change only which training
rows a model learns from (or their weights); predictions are always made for untouched rows of the
other half, so every variant is scored on the same untouched labels:
  B drop pairs labelled against their pattern's majority      C drop patterns whose minority > 5%
  D drop pairs contradicted by out-of-fold p1 (stage 2)       E easy pairs 10% with weight 10 (stage 2)
  F drop empty-address records in identical-name groups with conflicting labels
  G weight 0.2 for pairs contradicted by out-of-fold p1 (stage 2)

Usage: python3 model.py train A[,B,C,...]     python3 model.py test A
       BER_TAG=_fast BER_MAX_ROWS=4000000 ... for quick comparisons (models/<variant>_fast)
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.metrics import log_loss, roc_auc_score

from common import NPROC, WORK, id_hash, log, split_dir
from decode import decode, decode_kept, exclusive, macro_f05
from features import NUM, WORDS

RET = ["score", "rank_r", "rank_s", "rec_ncand", "rec_best", "gap_r", "s1_ncand", "s1_best", "gap_s"]
BASE = NUM + RET
CATS = ["nd", "ad", "rel", "src"]
WF_NAMES = [f"{c}_{s}" for c in WORDS for s in ("lo_max", "lo_min", "lo_mean", "unseen")]
CTX = ["p1", "p1_rec_max_other", "p1_rec_sum_other", "p1_rec_nhi_other", "p1_rec_rank", "p1_margin",
       "p1_s1_max_other", "p1_s1_sum_other", "p1_s1_nhi_other", "p1_s1_rank",
       "tw_num_p1", "tw_name_p1", "dup_p1", "rec_ncand_hi"]
F1 = BASE + WF_NAMES
F2 = BASE + WF_NAMES + CTX
PRUNE = float(os.environ.get("BER_PRUNE", "0.005"))
MAX_ROWS = int(os.environ.get("BER_MAX_ROWS", "10000000"))
ROUNDS = int(os.environ.get("BER_ROUNDS", "1500"))
LR = float(os.environ.get("BER_LR", "0.1"))
DROPOUT = float(os.environ.get("BER_EVIDENCE_DROPOUT", "0.15"))
TAG = os.environ.get("BER_TAG", "")          # e.g. "_fast": model folder models/<variant><TAG>
BATCH = 2_000_000
PARAMS = dict(objective="binary", learning_rate=LR, num_leaves=255, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255,
              num_threads=NPROC, verbose=-1, max_cat_to_onehot=16, seed=7)
N_TAB = 11          # word-evidence tables: k*5+j (half k without inner fold j), k*5+4 (all of half k), 10 unseen


# ------------------------------------------------------------------ data
def load_split(split):
    d = split_dir(split)
    c = np.load(d / "cands.npz")
    s1, rec = c["s1"].astype(np.int64), c["rec"].astype(np.int64)
    n = len(s1)
    X = np.empty((n, len(BASE)), np.float32)
    start = 0
    for b in pq.ParquetFile(d / "feats.parquet").iter_batches(columns=NUM, batch_size=BATCH):
        m = b.num_rows
        for j, col in enumerate(NUM):
            X[start:start + m, j] = b.column(col).to_numpy(zero_copy_only=False)
        start += m
    assert start == n
    s1_tab = pa.ipc.open_file(pa.memory_map(str(d / "s1.arrow"))).read_all()
    rec_tab = pa.ipc.open_file(pa.memory_map(str(d / "rec.arrow"))).read_all()
    n_s1, n_rec = s1_tab.num_rows, rec_tab.num_rows
    score = c["score"].astype(np.float32)
    rec_best = np.zeros(n_rec, np.float32)
    np.maximum.at(rec_best, rec, score)
    s1_best = np.zeros(n_s1, np.float32)
    np.maximum.at(s1_best, s1, score)
    ret = [score, c["rank_r"], c["rank_s"], np.bincount(rec, minlength=n_rec)[rec], rec_best[rec],
           rec_best[rec] - score, np.bincount(s1, minlength=n_s1)[s1], s1_best[s1], s1_best[s1] - score]
    for j, v in enumerate(ret):
        X[:, len(NUM) + j] = v
    codes = {}
    for col in ("anums", "ntoks", "awords"):
        codes[col] = pd.factorize(rec_tab[col].to_numpy(zero_copy_only=False))[0].astype(np.int64)
    empty_num = rec_tab["anums"].to_numpy(zero_copy_only=False) == ""
    codes["anums"][empty_num] = -1
    full = pd.factorize(pd.Series(codes["ntoks"]).astype(str) + "|" + pd.Series(codes["awords"]).astype(str)
                        + "|" + pd.Series(codes["anums"]).astype(str))[0].astype(np.int64)
    meta = dict(
        d=d, n=n, s1=s1, rec=rec, n_s1=n_s1, n_rec=n_rec,
        s1_id=s1_tab["id"].to_numpy(), rec_src=rec_tab["src"].to_numpy().astype(np.int64),
        rec_country=pd.factorize(rec_tab["country"].to_numpy(zero_copy_only=False))[0],
        country_names=pd.factorize(rec_tab["country"].to_numpy(zero_copy_only=False))[1],
        rec_aempty=rec_tab["aempty"].to_numpy(), code_num=codes["anums"], code_name=codes["ntoks"],
        code_full=full, s1_country=s1_tab["country"].to_numpy(zero_copy_only=False),
    )
    if (d / "labels.npz").exists():
        lab = np.load(d / "labels.npz")
        meta["true_row"], meta["true_size"] = lab["true_row"], lab["true_size"]
        meta["y"] = (lab["true_row"][rec] == s1).astype(np.int8)
    g = id_hash(meta["s1_id"], 8).astype(np.int64)
    meta["half"] = (g % 2)[s1]
    meta["inner"] = (g // 2)[s1]
    meta["es"] = (id_hash(meta["s1_id"] // 7, 50) == 0)[s1]      # early-stopping S1 groups
    log(f"{split}: {n:,} pairs, {n_s1:,} S1, {n_rec:,} records loaded")
    return X, meta


# ------------------------------------------------------------------ word evidence
def _tokens(arr):
    lists = pc.split_pattern(arr, pattern=" ")
    lens = pc.list_value_length(lists).to_numpy(zero_copy_only=False)
    flat = pc.list_flatten(lists)
    parents = np.repeat(np.arange(len(lens)), lens)
    keep = pc.not_equal(flat, "").to_numpy(zero_copy_only=False)
    return pc.filter(flat, pa.array(keep)), parents[keep]


def _batches(d, rows=None):
    """(start row, record batch) over feats.parquet; optionally only batches touching `rows`."""
    start = 0
    for b in pq.ParquetFile(d / "feats.parquet").iter_batches(columns=WORDS, batch_size=BATCH):
        yield start, b
        start += b.num_rows


class WordEvidence:
    """Smoothed log-odds of each added / dropped word being in a match, from chosen label groups."""

    def __init__(self, vocab=None):
        self.vocab = vocab or {}
        self.LO, self.UNS = {}, {}

    def fit(self, meta, count_mask):
        d, y = meta["d"], meta["y"]
        grp = (meta["half"] + 2 * meta["inner"]).astype(np.int64)      # 8 groups: half + 2*inner
        if not self.vocab:
            for c in WORDS:
                u = [pc.unique(_tokens(b.column(c))[0]) for _, b in _batches(d)]
                self.vocab[c] = pc.unique(pa.concat_arrays(u))
        cnt = {c: np.zeros((8, 2, len(self.vocab[c]) + 1), np.int64) for c in WORDS}
        for start, b in _batches(d):
            rows = np.arange(start, start + b.num_rows)
            for c in WORDS:
                flat, par = _tokens(b.column(c))
                ids = self._ids(flat, c)
                r = rows[par]
                m = count_mask[r]
                V1 = len(self.vocab[c]) + 1
                idx = (grp[r[m]] * 2 + y[r[m]]) * V1 + ids[m]
                cnt[c] += np.bincount(idx, minlength=16 * V1).reshape(8, 2, V1)
        for c in WORDS:
            V1 = len(self.vocab[c]) + 1
            LO = np.zeros((N_TAB, V1), np.float32)
            UNS = np.ones((N_TAB, V1), bool)
            for k in (0, 1):
                for j in range(5):
                    sel = [g for g in range(8) if g % 2 == k and (j == 4 or g // 2 != j)]
                    pos = cnt[c][sel, 1].sum(0).astype(np.float64)
                    neg = cnt[c][sel, 0].sum(0).astype(np.float64)
                    pos[-1] = neg[-1] = 0
                    V = V1 - 1
                    lo = np.log((pos + 1) / (pos.sum() + V)) - np.log((neg + 1) / (neg.sum() + V))
                    uns = (pos + neg) == 0
                    lo[uns] = 0
                    LO[k * 5 + j], UNS[k * 5 + j] = lo, uns
            self.LO[c], self.UNS[c] = LO, UNS
        return self

    def _ids(self, flat, c):
        V = len(self.vocab[c])
        return pc.fill_null(pc.index_in(flat, value_set=self.vocab[c]), V).to_numpy(zero_copy_only=False).astype(np.int64)

    def encode(self, batch, tbl):
        """Features for the rows of a record batch; tbl = table index per row."""
        nb_ = batch.num_rows
        out = np.empty((nb_, len(WF_NAMES)), np.float32)
        for ci, c in enumerate(WORDS):
            flat, par = _tokens(batch.column(c))
            ids = self._ids(flat, c)
            tt = tbl[par]
            v = self.LO[c][tt, ids]
            u = self.UNS[c][tt, ids].astype(np.float32)
            cntr = np.bincount(par, minlength=nb_)
            mx, mn, me = (np.full(nb_, np.nan, np.float32) for _ in range(3))
            nu = np.zeros(nb_, np.float32)
            nz = cntr > 0
            if len(v):
                st = np.concatenate([[0], np.cumsum(cntr)[:-1]])[nz]
                mx[nz] = np.maximum.reduceat(v, st)
                mn[nz] = np.minimum.reduceat(v, st)
                me[nz] = np.add.reduceat(v, st) / cntr[nz]
                nu[nz] = np.add.reduceat(u, st)
            out[:, ci * 4:ci * 4 + 4] = np.column_stack([mx, mn, me, nu])
        return out

    def save(self, path):
        arrs = {f"LO_{c}": self.LO[c][[4, 9, 10]] for c in WORDS}
        arrs.update({f"UNS_{c}": self.UNS[c][[4, 9, 10]] for c in WORDS})
        np.savez(path, **arrs)
        for c in WORDS:
            with pa.OSFile(f"{path}.vocab_{c}.arrow", "wb") as f, pa.ipc.new_file(f, pa.schema([("w", pa.string())])) as w:
                w.write_table(pa.table({"w": self.vocab[c]}))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        vocab = {c: pa.ipc.open_file(pa.memory_map(f"{path}.vocab_{c}.arrow")).read_all()["w"].combine_chunks()
                 for c in WORDS}
        we = cls(vocab)
        for c in WORDS:     # saved tables 4, 9, 10 -> put them back at those indices
            LO = np.zeros((N_TAB,) + z[f"LO_{c}"].shape[1:], np.float32)
            UNS = np.ones((N_TAB,) + z[f"UNS_{c}"].shape[1:], bool)
            LO[[4, 9, 10]], UNS[[4, 9, 10]] = z[f"LO_{c}"], z[f"UNS_{c}"]
            we.LO[c], we.UNS[c] = LO, UNS
        return we


def gather_wf(we, meta, rows, tbl):
    """Word-evidence features for sorted rows (tbl = table index per row), one pass over the parquet."""
    out = np.empty((len(rows), len(WF_NAMES)), np.float32)
    for start, b in _batches(meta["d"]):
        lo, hi = np.searchsorted(rows, [start, start + b.num_rows])
        if hi == lo:
            continue
        out[lo:hi] = we.encode(b.take(pa.array(rows[lo:hi] - start)), tbl[lo:hi])
    return out


# ------------------------------------------------------------------ context (self-correction) features
def _group_rank_stats(key, p, n_key):
    order = np.lexsort((-p, key))
    ks, ps = key[order], p[order]
    n = len(p)
    first = np.ones(n, bool)
    first[1:] = ks[1:] != ks[:-1]
    gstart = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    rank_sorted = np.arange(n) - gstart
    top1 = ps[gstart]
    nxt = np.minimum(gstart + 1, n - 1)
    top2 = np.where((gstart + 1 < n) & (ks[nxt] == ks), ps[nxt], 0.0)
    max_other_s = np.where(rank_sorted == 0, top2, top1)
    rank = np.empty(n, np.float32)
    rank[order] = rank_sorted
    max_other = np.empty(n, np.float32)
    max_other[order] = max_other_s
    sum_all = np.bincount(key, weights=p, minlength=n_key)
    hi_all = np.bincount(key, weights=(p > 0.5).astype(np.float64), minlength=n_key)
    return rank, max_other, (sum_all[key] - p).astype(np.float32), (hi_all[key] - (p > 0.5)).astype(np.float32)


def _other_source_max(s1, src_bit, code, p, valid):
    """Max p over records of the OTHER source with the same code, under the same S1 (NaN if none)."""
    k_self = (s1 << 33) | (src_bit << 32) | code
    k_other = (s1 << 33) | ((1 - src_bit) << 32) | code
    ser = pd.Series(p[valid]).groupby(k_self[valid]).max()
    out = np.full(len(p), np.nan, np.float32)
    out[valid] = ser.reindex(k_other[valid]).to_numpy(np.float32)
    return out


def _dup_max(s1, code, p):
    """Max p over OTHER records with an identical (name, address, numbers) under the same S1."""
    key = (s1 << 32) | code
    uk, inv = np.unique(key, return_inverse=True)
    rank, max_other, _, _ = _group_rank_stats(inv.astype(np.int64), p, len(uk))
    size = np.bincount(inv)[inv]
    return np.where(size > 1, max_other, np.nan).astype(np.float32)


def context(meta, p1):
    s1, rec = meta["s1"], meta["rec"]
    r_rank, r_mo, r_sum, r_hi = _group_rank_stats(rec, p1, meta["n_rec"])
    s_rank, s_mo, s_sum, s_hi = _group_rank_stats(s1, p1, meta["n_s1"])
    src_bit = (meta["rec_src"][rec] == 3).astype(np.int64)
    cnum = meta["code_num"][rec]
    tw_num = _other_source_max(s1, src_bit, np.maximum(cnum, 0), p1, cnum >= 0)
    tw_name = _other_source_max(s1, src_bit, meta["code_name"][rec], p1, np.ones(len(p1), bool))
    dup = _dup_max(s1, meta["code_full"][rec], p1)
    ncand_hi = np.bincount(rec, weights=(p1 > 0.1).astype(np.float64), minlength=meta["n_rec"])[rec]
    return np.column_stack([p1, r_mo, r_sum, r_hi, r_rank, p1 - r_mo, s_mo, s_sum, s_hi, s_rank,
                            tw_num, tw_name, dup, ncand_hi]).astype(np.float32)


# ------------------------------------------------------------------ variants
def variant_rows(variant, meta, X, stage, p1=None):
    """(keep mask, weight) for training rows under a data-handling variant."""
    n = meta["n"]
    y = meta["y"]
    keep = np.ones(n, bool)
    w = np.ones(n, np.float32)
    if variant in ("B", "C"):
        cols = [BASE.index(c) for c in ("nd", "ad", "rel", "src")]
        key = pd.DataFrame({"c": meta["rec_country"][meta["rec"]], "h": meta["half"],
                            **{f"f{j}": X[:, j].astype(np.int16) for j in cols}})
        gid = key.groupby(list(key.columns), sort=False).ngroup().to_numpy()   # patterns per half
        size = np.bincount(gid)
        pos = np.bincount(gid, weights=y)
        if variant == "B":
            maj = (pos * 2 >= size).astype(np.int8)
            keep &= y == maj[gid]
        else:
            keep &= (np.minimum(pos, size - pos) / size)[gid] <= 0.05
    if variant == "F":
        rec = meta["rec"]
        empty = meta["rec_aempty"] == 1
        codes = meta["code_name"][empty]
        truth = meta["true_row"][empty]
        df = pd.DataFrame({"c": codes, "t": truth})
        nuniq = df.groupby("c")["t"].nunique()
        bad_codes = set(nuniq[nuniq > 1].index.tolist())
        bad_rec = np.zeros(meta["n_rec"], bool)
        bad_rec[np.flatnonzero(empty)[np.isin(codes, list(bad_codes))]] = True
        keep &= ~bad_rec[rec]
    if stage == 2 and p1 is not None:
        contra = ((y == 1) & (p1 < 0.02)) | ((y == 0) & (p1 > 0.98))
        if variant == "D":
            keep &= ~contra
        if variant == "G":
            w[contra] = 0.2
        if variant == "E":
            easy = (p1 < 0.002) | (p1 > 0.998)
            rng = np.random.default_rng(11)
            sel = rng.random(n) < 0.10
            keep &= ~easy | sel
            w[easy & sel] = 10.0
    return keep, w


# ------------------------------------------------------------------ training
HARD_COLS = [BASE.index(c) for c in ("score", "rank_r", "rank_s", "name_tsr", "addr_tsr")]


def sample_rows(cand, y, w, rng, X):
    """All positives, plus negatives up to MAX_ROWS: 70% of the room for hard-looking negatives
    (the neighbour decoys: high retrieval score, top ranks, similar name or address), 30% for easy
    ones. Each stratum is re-weighted by population / sample, so the model stays unbiased."""
    pos = cand[y[cand] == 1]
    neg = cand[y[cand] == 0]
    room = max(MAX_ROWS - len(pos), 1)
    ww = w.copy()
    if len(neg) > room:
        sc, rr, rs, nt, at = (X[neg, j] for j in HARD_COLS)
        hard = (sc >= 0.4) | (rr <= 2) | (rs <= 2) | (nt >= 80) | (np.nan_to_num(at) >= 80)
        hn, en = neg[hard], neg[~hard]
        r_h = min(len(hn), int(0.7 * room))
        r_e = min(len(en), room - r_h)
        hs = np.sort(rng.choice(hn, r_h, replace=False)) if r_h < len(hn) else hn
        es = np.sort(rng.choice(en, r_e, replace=False)) if r_e < len(en) else en
        ww[hs] = ww[hs] * (len(hn) / max(len(hs), 1))
        ww[es] = ww[es] * (len(en) / max(len(es), 1))
        log(f"  negatives: {len(hn):,} hard -> {len(hs):,}, {len(en):,} easy -> {len(es):,}")
        neg = np.concatenate([hs, es])
    rows = np.sort(np.concatenate([pos, neg]))
    return rows, ww[rows]


def fit(Xtr, ytr, wtr, Xva, yva, names, params=None, rounds=None, patience=50):
    cat = [names.index(c) for c in CATS]
    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, feature_name=names, categorical_feature=cat, free_raw_data=True)
    dva = lgb.Dataset(Xva, label=yva, feature_name=names, categorical_feature=cat, reference=dtr)
    booster = lgb.train(params or PARAMS, dtr, num_boost_round=rounds or ROUNDS, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(100)])
    log(f"  best iteration {booster.best_iteration}, valid logloss {booster.best_score['valid_0']['binary_logloss']:.5f}")
    return booster


def train_stage(stage, meta, X, we, keep, w, prune_mask, ctx, rng, params=None, rounds=None, patience=50):
    """Two models (one per half); returns models and out-of-fold predictions for all pruned rows."""
    y = meta["y"]
    names = F1 if stage == 1 else F2
    models = []
    oof = np.zeros(meta["n"], np.float32)
    for k in (0, 1):
        cand = np.flatnonzero((meta["half"] == k) & keep & prune_mask & ~meta["es"])
        rows, ww = sample_rows(cand, y, w, rng, X)
        va = np.flatnonzero((meta["half"] == k) & prune_mask & meta["es"])
        if len(va) > 1_000_000:
            va = np.sort(rng.choice(va, 1_000_000, replace=False))
        # training rows: evidence from the other inner folds of this half (no row sees its own label);
        # a DROPOUT share of rows sees every word as unseen, as French test words will be
        drop = rng.random(len(rows)) < DROPOUT
        WFtr = gather_wf(we, meta, rows, np.where(drop, 10, k * 5 + meta["inner"][rows]).astype(np.int64))
        WFva = gather_wf(we, meta, va, np.full(len(va), k * 5 + 4, np.int64))
        Xtr = np.hstack([X[rows], WFtr] + ([ctx[rows]] if stage == 2 else []))
        Xva = np.hstack([X[va], WFva] + ([ctx[va]] if stage == 2 else []))
        log(f"stage {stage} half {k}: training on {len(rows):,} rows ({int(y[rows].sum()):,} matches)")
        models.append(fit(Xtr, y[rows], ww, Xva, y[va], names, params, rounds, patience))
        del Xtr, WFtr
    oof[:] = predict(models, meta, X, we, prune_mask, ctx, stage, oof_mode=True)
    return models, oof


def predict(models, meta, X, we, mask, ctx, stage, oof_mode):
    """oof_mode: rows of half h are predicted by the model of the other half (word evidence from it);
    otherwise (test) every row gets the average of both models."""
    p = np.zeros(meta["n"], np.float32)
    for start, b in _batches(meta["d"]):
        local = np.flatnonzero(mask[start:start + b.num_rows])
        if not len(local):
            continue
        rows = start + local
        bb = b.take(pa.array(local))
        extra = [ctx[rows]] if stage == 2 else []
        if oof_mode:
            h = meta["half"][rows]
            Xb = np.hstack([X[rows], we.encode(bb, (1 - h) * 5 + 4)] + extra)
            pr = np.empty(len(rows), np.float32)
            for k in (0, 1):
                sel = h == 1 - k                  # rows of the other half
                if sel.any():
                    pr[sel] = models[k].predict(Xb[sel], num_threads=NPROC)
        else:
            X0 = np.hstack([X[rows], we.encode(bb, np.full(len(rows), 4))] + extra)
            X1 = np.hstack([X[rows], we.encode(bb, np.full(len(rows), 9))] + extra)
            pr = 0.5 * (models[0].predict(X0, num_threads=NPROC) + models[1].predict(X1, num_threads=NPROC))
        p[rows] = pr
    return p


# ------------------------------------------------------------------ evaluation
def evaluate(meta, p, tag):
    s1, rec, y = meta["s1"], meta["rec"], meta["y"].astype(bool)
    ts, n_s1 = meta["true_size"], meta["n_s1"]
    res = {"tag": tag}
    half_s1 = id_hash(meta["s1_id"], 8) % 2
    best = None
    pk = np.where(exclusive(rec, p), p, 0.0)
    for method, params in ([("threshold", dict(t=t)) for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)] +
                           [("expected", dict(lam=l)) for l in (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0)]):
        pred = decode_kept(s1, pk, n_s1, method=method, **params)
        f = macro_f05(s1, pred, y, ts, n_s1)
        key = f"{method} {params}"
        res[key] = {"all": round(float(f.mean()), 5), "half0": round(float(f[half_s1 == 0].mean()), 5),
                    "half1": round(float(f[half_s1 == 1].mean()), 5)}
        if best is None or f.mean() > best[0]:
            best = (float(f.mean()), method, params, f)
    res["best"] = {"method": best[1], "params": best[2], "macro_f05": round(best[0], 5)}
    f = best[3]
    for ci, cname in enumerate(sorted(set(meta["s1_country"].tolist()))):
        m = meta["s1_country"] == cname
        res["best"][f"macro_f05_{cname}"] = round(float(f[m].mean()), 5)
    res["best"]["singletons"] = round(float(f[ts == 0].mean()), 5)
    res["best"]["non_singletons"] = round(float(f[ts > 0].mean()), 5)
    log(f"{tag}: best {best[1]} {best[2]} macro F0.5 = {best[0]:.5f}")
    return res, best


def ceiling(meta, mask):
    y = meta["y"].astype(bool)
    tp = np.bincount(meta["s1"][mask & y], minlength=meta["n_s1"])
    ts = meta["true_size"]
    f = np.where(ts == 0, 1.0, 1.25 * tp / np.maximum(tp + 0.25 * ts, 1e-9))
    return round(float(f.mean()), 5), int((mask & y).sum()), int((meta["true_row"] >= 0).sum())


# ------------------------------------------------------------------ main
STAGE2_ONLY = {"D", "E", "G"}      # variants that change only stage-2 training rows: reuse A's stage 1
if os.environ.get("BER_REUSE_STAGE1") == "1":     # fast comparisons: every variant cleans stage 2 only
    STAGE2_ONLY = {"B", "C", "D", "E", "F", "G"}


def run_train(variants):
    """Train one or more variants in one process (the pair features are loaded once)."""
    X, meta = load_split("train")
    base_ceiling = ceiling(meta, np.ones(meta["n"], bool))
    cache = {}
    for v in variants.split(","):
        train_variant(v.strip(), X, meta, base_ceiling, cache)


def train_variant(variant, X, meta, base_ceiling, cache):
    t0 = time.time()
    rng = np.random.default_rng(7)
    y = meta["y"]
    mdir = WORK / "models" / f"{variant}{TAG}"
    mdir.mkdir(parents=True, exist_ok=True)
    report = {"variant": variant, "tag": TAG, "max_rows": MAX_ROWS, "rounds": ROUNDS, "lr": LR,
              "pairs": meta["n"], "retrieval_ceiling": base_ceiling}
    s1key = "A" if variant in STAGE2_ONLY else variant
    if s1key in cache:                                  # same stage 1 as A (same rows, same word evidence)
        we, m1, p1, rep1 = cache[s1key]
        log(f"{variant}: reusing stage 1 of {s1key}")
    else:
        keep1, w1 = variant_rows(s1key, meta, X, stage=1)
        we = WordEvidence().fit(meta, keep1)
        log("word evidence fitted")
        m1, p1 = train_stage(1, meta, X, we, keep1, w1, np.ones(meta["n"], bool), None, rng)
        rep1 = {"stage1_auc": round(float(roc_auc_score(y, p1)), 6),
                "stage1_logloss": round(float(log_loss(y, np.clip(p1, 1e-6, 1 - 1e-6))), 5),
                "stage1_decoded": evaluate(meta, p1, f"{s1key} stage 1")[0]["best"]}
        cache[s1key] = (we, m1, p1, rep1)
    report.update(rep1)
    we.save(str(mdir / "wordev.npz"))
    np.save(mdir / "oof_p1.npy", p1)
    prune = p1 >= PRUNE
    report["prune_ceiling"] = ceiling(meta, prune)
    report["pairs_after_prune"] = int(prune.sum())
    log(f"pruned to {prune.sum():,} pairs; ceiling {report['prune_ceiling']}")

    ctx = context(meta, p1)
    keep2, w2 = variant_rows(variant, meta, X, stage=2, p1=p1)
    report["stage2_training_rows_removed"] = int((~keep2).sum())
    report["stage2_training_rows_downweighted"] = int((w2 < 1).sum())
    m2, p2 = train_stage(2, meta, X, we, keep2, w2, prune, ctx, rng)
    p2[~prune] = 0.0
    report["stage2_auc_pruned"] = round(float(roc_auc_score(y[prune], p2[prune])), 6)
    res2, best = evaluate(meta, p2, f"{variant} stage 2")
    report["stage2_decoded"] = res2
    for k, m in enumerate(m1):
        m.save_model(str(mdir / f"stage1_{k}.txt"))
    for k, m in enumerate(m2):
        m.save_model(str(mdir / f"stage2_{k}.txt"))
    imp = pd.Series(m2[0].feature_importance("gain"), index=F2)
    report["stage2_top_features"] = (imp / imp.sum()).sort_values(ascending=False).head(25).round(4).to_dict()
    report["decoder"] = {"method": best[1], "params": best[2]}
    report["minutes"] = round((time.time() - t0) / 60, 1)
    (mdir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    np.save(mdir / "oof_p2.npy", p2)
    log(f"variant {variant}{TAG} done: {json.dumps(report['stage2_decoded']['best'])}")


S2_GRID = [   # stage-2 settings tried by "tune" (stage 1 is trained once, at full size)
    dict(learning_rate=0.1, num_leaves=255, min_data_in_leaf=100),
    dict(learning_rate=0.05, num_leaves=255, min_data_in_leaf=100),
    dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=300),
    dict(learning_rate=0.05, num_leaves=511, min_data_in_leaf=50),
    dict(learning_rate=0.03, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.6),
    dict(learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, lambda_l2=10.0),
]


def run_tune(variant):
    """Stage 1 once (full-size settings from the environment), then every stage-2 setting in S2_GRID
    for this variant's stage-2 rows. Each result goes to models/<variant>_tune<i>; the best index is
    written to models/<variant>_tune_best.txt."""
    X, meta = load_split("train")
    y = meta["y"]
    base_ceiling = ceiling(meta, np.ones(meta["n"], bool))
    rng = np.random.default_rng(7)
    s1key = "A" if variant in STAGE2_ONLY else variant
    keep1, w1 = variant_rows(s1key, meta, X, stage=1)
    we = WordEvidence().fit(meta, keep1)
    log("word evidence fitted")
    m1, p1 = train_stage(1, meta, X, we, keep1, w1, np.ones(meta["n"], bool), None, rng)
    rep1 = {"stage1_max_rows": MAX_ROWS, "stage1_lr": LR,
            "stage1_auc": round(float(roc_auc_score(y, p1)), 6),
            "stage1_decoded": evaluate(meta, p1, f"{variant} full stage 1")[0]["best"]}
    prune = p1 >= PRUNE
    ctx = context(meta, p1)
    keep2, w2 = variant_rows(variant, meta, X, stage=2, p1=p1)
    results = []
    for i, extra in enumerate(S2_GRID):
        t0 = time.time()
        mdir = WORK / "models" / f"{variant}_tune{i}"
        mdir.mkdir(parents=True, exist_ok=True)
        params = {**PARAMS, **extra}
        m2, p2 = train_stage(2, meta, X, we, keep2, w2, prune, ctx, np.random.default_rng(7), params, 3000, 100)
        p2[~prune] = 0.0
        res2, best = evaluate(meta, p2, f"{variant} tune{i} {extra}")
        report = {"variant": variant, "tag": f"_tune{i}", "stage2_params": extra, "retrieval_ceiling": base_ceiling,
                  "prune_ceiling": ceiling(meta, prune), **rep1, "stage2_decoded": res2,
                  "decoder": {"method": best[1], "params": best[2]}, "minutes": round((time.time() - t0) / 60, 1)}
        we.save(str(mdir / "wordev.npz"))
        for k, m in enumerate(m1):
            m.save_model(str(mdir / f"stage1_{k}.txt"))
        for k, m in enumerate(m2):
            m.save_model(str(mdir / f"stage2_{k}.txt"))
        np.save(mdir / "oof_p2.npy", p2)
        (mdir / "report.json").write_text(json.dumps(report, indent=2, default=str))
        results.append(best[0])
        log(f"tune{i} {extra}: macro F0.5 = {best[0]:.5f}")
    bi = int(np.argmax(results))
    (WORK / "models" / f"{variant}_tune_best.txt").write_text(str(bi))
    log(f"best stage-2 setting: tune{bi} {S2_GRID[bi]} = {results[bi]:.5f}")


def run_test(variant):
    X, meta = load_split("test")
    mdir = WORK / "models" / f"{variant}{TAG}"
    rep = json.loads((mdir / "report.json").read_text())
    we = WordEvidence.load(str(mdir / "wordev.npz"))
    m1 = [lgb.Booster(model_file=str(mdir / f"stage1_{k}.txt")) for k in (0, 1)]
    m2 = [lgb.Booster(model_file=str(mdir / f"stage2_{k}.txt")) for k in (0, 1)]
    all_rows = np.ones(meta["n"], bool)
    p1 = predict(m1, meta, X, we, all_rows, None, 1, oof_mode=False)
    prune = p1 >= PRUNE
    ctx = context(meta, p1)
    p2 = predict(m2, meta, X, we, prune, ctx, 2, oof_mode=False)
    p2[~prune] = 0.0
    dec = rep["decoder"]
    pred = decode(meta["s1"], meta["rec"], p2, meta["n_s1"], method=dec["method"], **dec["params"])
    np.savez(meta["d"] / f"pred_{variant}{TAG}.npz", p1=p1, p2=p2, prune=prune, pred=pred)
    log(f"test: {prune.sum():,} pairs scored by stage 2, {pred.sum():,} predicted matches")


if __name__ == "__main__":
    mode = sys.argv[1]
    variant = sys.argv[2] if len(sys.argv) > 2 else "A"
    {"train": run_train, "test": run_test, "tune": run_tune}[mode](variant)
