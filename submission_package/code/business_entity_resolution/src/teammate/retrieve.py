#!/usr/bin/env python3
"""Stage 1: candidate generation. Per country, TF-IDF cosine over typed tokens
(name words, 4-letter prefixes, sound-alike skeletons; address words, prefixes, house numbers),
top-K1 S1 for every S2/S3 record and top-K2 records for every S1, unioned.

Name and address are weighted separately (each part L2-normalised, then mixed 50/50), so a long
address cannot drown the name. Tokens shared by more than MAXP_FRAC of the searched side are skipped
(they carry almost no identity and dominate the cost).

  WORK/<split>/cands.npz : s1, rec (row numbers), score, rank_r (rank of the S1 among the record's
                           candidates, K1+1 if absent), rank_s (rank of the record among the S1's)
Usage: python3 retrieve.py train|test
"""
import os
import sys

import numba as nb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import scipy.sparse as sp

from common import log, split_dir

K1 = int(os.environ.get("BER_K1", "10"))     # S1 candidates per record
K2 = int(os.environ.get("BER_K2", "10"))     # record candidates per S1
K1_EMPTY = int(os.environ.get("BER_K1_EMPTY", "25"))   # records without an address: name-only, more ties
MAXP_FRAC = float(os.environ.get("BER_MAXP", "0.02"))  # record -> S1: skip tokens in >2% of S1
MAXP_REV = float(os.environ.get("BER_MAXP_REV", "0.005"))  # S1 -> record: skip tokens in >0.5% of records
W_NAME = 0.5


def load(path, cols):
    return pa.ipc.open_file(pa.memory_map(str(path))).read_all().select(cols)


def tfidf(strings_a, strings_b):
    """Shared-vocabulary binary TF-IDF, rows L2-normalised; returns (A, B) CSR matrices."""
    both = pa.chunked_array(strings_a.chunks + strings_b.chunks).combine_chunks()
    lists = pc.split_pattern(both, pattern=" ")
    lens = pc.list_value_length(lists).to_numpy(zero_copy_only=False)
    flat = pc.list_flatten(lists)
    keep = pc.not_equal(flat, "").to_numpy(zero_copy_only=False)
    enc = pc.dictionary_encode(flat)
    ids = enc.indices.to_numpy(zero_copy_only=False)
    rows = np.repeat(np.arange(len(lens)), lens)
    rows, ids = rows[keep], ids[keep]
    V = len(enc.dictionary)
    n = len(lens)
    df = np.bincount(ids, minlength=V).astype(np.float64)
    idf = np.log((n + 1) / (df + 1)).astype(np.float32)
    M = sp.csr_matrix((idf[ids], (rows, ids)), shape=(n, V), dtype=np.float32)
    norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
    norms[norms == 0] = 1
    M = sp.diags((1 / norms).astype(np.float32)) @ M
    na = len(strings_a)
    return M[:na].tocsr(), M[na:].tocsr()


def combine(n_part, a_part, has_n, has_a):
    """Mix the name and address parts so cos = W_NAME*cos_name + (1-W_NAME)*cos_addr when both exist."""
    wn = np.where(has_a, np.sqrt(W_NAME), 1.0).astype(np.float32)
    wa = np.where(has_n, np.sqrt(1 - W_NAME), 1.0).astype(np.float32)
    return sp.hstack([sp.diags(wn) @ n_part, sp.diags(wa) @ a_part]).tocsr()


@nb.njit(cache=True)
def _all_common(q_ptr, q_idx, d_ptr, i, maxp):
    for j in range(q_ptr[i], q_ptr[i + 1]):
        tok = q_idx[j]
        ln = d_ptr[tok + 1] - d_ptr[tok]
        if 0 < ln <= maxp:
            return False
    return True


@nb.njit(parallel=True, fastmath=True, cache=True)
def search(q_ptr, q_idx, q_val, d_ptr, d_idx, d_val, n_d, K, maxp):
    nq = q_ptr.shape[0] - 1
    out_i = np.full((nq, K), -1, np.int32)
    out_s = np.zeros((nq, K), np.float32)
    nth = nb.get_num_threads()
    for t in nb.prange(nth):
        acc = np.zeros(n_d, np.float32)
        touched = np.empty(n_d, np.int32)
        ks = np.empty(K, np.float32)
        ki = np.empty(K, np.int32)
        for i in range(t, nq, nth):
            nt = 0
            rare = -1
            rare_len = 1 << 62
            for j in range(q_ptr[i], q_ptr[i + 1]):
                tok = q_idx[j]
                ln = d_ptr[tok + 1] - d_ptr[tok]
                if ln > 0 and ln < rare_len:
                    rare_len = ln
                    rare = j
            for j in range(q_ptr[i], q_ptr[i + 1]):
                tok = q_idx[j]
                w = q_val[j]
                s = d_ptr[tok]
                e = d_ptr[tok + 1]
                if e - s > maxp and not (j == rare and rare_len > maxp and rare_len <= 50 * maxp and
                                         _all_common(q_ptr, q_idx, d_ptr, i, maxp)):
                    continue
                for k in range(s, e):
                    r = d_idx[k]
                    if acc[r] == 0.0:
                        touched[nt] = r
                        nt += 1
                    acc[r] += w * d_val[k]
            cnt = 0
            minpos = 0
            for u in range(nt):
                r = touched[u]
                sc = acc[r]
                acc[r] = 0.0
                if cnt < K:
                    ks[cnt] = sc
                    ki[cnt] = r
                    cnt += 1
                    if cnt == K:
                        minpos = 0
                        for v in range(1, K):
                            if ks[v] < ks[minpos]:
                                minpos = v
                elif sc > ks[minpos]:
                    ks[minpos] = sc
                    ki[minpos] = r
                    minpos = 0
                    for v in range(1, K):
                        if ks[v] < ks[minpos]:
                            minpos = v
            order = np.argsort(-ks[:cnt])
            for v in range(cnt):
                out_s[i, v] = ks[order[v]]
                out_i[i, v] = ki[order[v]]
    return out_i, out_s


def topk(Q, D, K, frac):
    Dc = D.tocsc()
    maxp = max(2000, int(frac * D.shape[0]))
    return search(Q.indptr.astype(np.int64), Q.indices.astype(np.int32), Q.data.astype(np.float32),
                  Dc.indptr.astype(np.int64), Dc.indices.astype(np.int32), Dc.data.astype(np.float32),
                  D.shape[0], K, maxp)


def main(split):
    d = split_dir(split)
    s1 = load(d / "s1.arrow", ["country", "rt_n", "rt_a"])
    rec = load(d / "rec.arrow", ["country", "rt_n", "rt_a"])
    s1_c = s1["country"].to_numpy()
    rec_c = rec["country"].to_numpy()
    parts = []
    for C in sorted(set(s1_c.tolist())):
        si = np.flatnonzero(s1_c == C).astype(np.int32)
        ri = np.flatnonzero(rec_c == C).astype(np.int32)
        if not len(si) or not len(ri):
            continue
        take = lambda tab, idx, col: tab[col].take(pa.array(idx))
        Sn, Rn = tfidf(take(s1, si, "rt_n"), take(rec, ri, "rt_n"))
        Sa, Ra = tfidf(take(s1, si, "rt_a"), take(rec, ri, "rt_a"))
        S = combine(Sn, Sa, Sn.getnnz(axis=1) > 0, Sa.getnnz(axis=1) > 0)
        R = combine(Rn, Ra, Rn.getnnz(axis=1) > 0, Ra.getnnz(axis=1) > 0)
        log(f"{C}: {len(si):,} S1 x {len(ri):,} records, {S.shape[1]:,} tokens")
        ri_top, rs_top = topk(R, S, K1, MAXP_FRAC)          # record -> S1
        emp = np.flatnonzero(Ra.getnnz(axis=1) == 0)         # no address: name-only, wider list
        if len(emp) and K1_EMPTY > K1:
            ei, es = topk(R[emp], S, K1_EMPTY, MAXP_FRAC)
            emp_pairs = (emp, ei, es)
        else:
            emp_pairs = None
        log(f"{C}: record -> S1 done")
        si_top, ss_top = topk(S, R, K2, MAXP_REV)          # S1 -> record
        log(f"{C}: S1 -> record done")
        a_s1 = np.repeat(np.arange(len(ri)), K1).reshape(len(ri), K1)
        m1 = ri_top >= 0
        p1 = np.column_stack([si[ri_top[m1]], ri[a_s1[m1]], rs_top[m1],
                              (np.tile(np.arange(1, K1 + 1), (len(ri), 1)))[m1]])
        if emp_pairs is not None:                      # ranks beyond K1 for address-less records
            emp, ei, es = emp_pairs
            a_e = np.repeat(emp, K1_EMPTY).reshape(len(emp), K1_EMPTY)
            rk = np.tile(np.arange(1, K1_EMPTY + 1), (len(emp), 1))
            me = (ei >= 0) & (rk > K1)
            p1 = np.vstack([p1, np.column_stack([si[ei[me]], ri[a_e[me]], es[me], np.minimum(rk[me], 127)])])
        a_r = np.repeat(np.arange(len(si)), K2).reshape(len(si), K2)
        m2 = si_top >= 0
        p2 = np.column_stack([si[a_r[m2]], ri[si_top[m2]], ss_top[m2],
                              (np.tile(np.arange(1, K2 + 1), (len(si), 1)))[m2]])
        # union: key = s1 row * 2^32 + rec row
        k1 = p1[:, 0].astype(np.int64) << 32 | p1[:, 1].astype(np.int64)
        k2 = p2[:, 0].astype(np.int64) << 32 | p2[:, 1].astype(np.int64)
        keys = np.union1d(k1, k2)
        score = np.zeros(len(keys), np.float32)
        rank_r = np.full(len(keys), K1 + 1, np.int8)
        rank_s = np.full(len(keys), K2 + 1, np.int8)
        i1 = np.searchsorted(keys, k1)
        i2 = np.searchsorted(keys, k2)
        score[i1] = p1[:, 2]
        np.maximum.at(score, i2, p2[:, 2].astype(np.float32))
        rank_r[i1] = p1[:, 3]
        rank_s[i2] = p2[:, 3]
        parts.append((keys, score, rank_r, rank_s))
        log(f"{C}: {len(keys):,} candidate pairs ({len(keys) / len(si):.1f} per S1)")
    keys = np.concatenate([p[0] for p in parts])
    out = dict(s1=(keys >> 32).astype(np.int32), rec=(keys & 0xFFFFFFFF).astype(np.int32),
               score=np.concatenate([p[1] for p in parts]), rank_r=np.concatenate([p[2] for p in parts]),
               rank_s=np.concatenate([p[3] for p in parts]))
    np.savez(d / "cands.npz", **out)
    log(f"{split}: {len(keys):,} candidate pairs written")
    if (d / "labels.npz").exists():
        report_recall(d, out)


def report_recall(d, c):
    lab = np.load(d / "labels.npz")
    true_row, true_size = lab["true_row"], lab["true_size"]
    hit = true_row[c["rec"]] == c["s1"]
    n_true = int((true_row >= 0).sum())
    tp = np.bincount(c["s1"][hit], minlength=len(true_size))
    f = np.where(true_size == 0, 1.0, 1.25 * tp / np.maximum(tp + 0.25 * true_size, 1e-9))
    log(f"recall: {hit.sum():,} of {n_true:,} true pairs retrieved ({hit.sum() / n_true:.4f}); "
        f"ceiling macro F0.5 = {f.mean():.5f}; {len(c['s1']) / len(true_size):.1f} candidates per S1")
    for k in (1, 2, 3, 5, K1):
        log(f"  true S1 within the record's top {k}: {(hit & (c['rank_r'] <= k)).sum() / n_true:.4f}")
    for k in (1, 3, 5, K2):
        log(f"  record within the S1's top {k}: {(hit & (c['rank_s'] <= k)).sum() / n_true:.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
