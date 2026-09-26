"""Turning pair probabilities into per-S1 match sets.

1. Exclusive assignment: an S2/S3 record matches at most one S1 (true in every training label), so
   each record keeps only its most probable S1.
2. Per S1, either a plain threshold, or the set size m that maximises the expected F0.5 given the
   (kept) probabilities: E[1.25 X / (m + 0.25 (X + Y + Z))], X / Y = true matches among the chosen /
   unchosen candidates (Poisson-binomial), Z ~ Poisson(lam) = true matches the candidates missed.
   m = 0 ("no match") scores P(no true match at all), which is how singletons are protected.
"""
import numba as nb
import numpy as np


def exclusive(rec, p):
    """Mask keeping, for every record, only its highest-probability pair (ties: first)."""
    order = np.lexsort((-p, rec))
    first = np.ones(len(order), bool)
    first[1:] = rec[order][1:] != rec[order][:-1]
    keep = np.zeros(len(p), bool)
    keep[order[first]] = True
    return keep


def group_sorted(s1, p, n_s1, min_p, cap):
    """Kept pairs with p >= min_p, grouped by S1 and sorted by p descending (at most cap per S1)."""
    idx = np.flatnonzero(p >= min_p)
    order = idx[np.lexsort((-p[idx], s1[idx]))]
    g = s1[order]
    start = np.searchsorted(g, np.arange(n_s1), "left")
    end = np.searchsorted(g, np.arange(n_s1), "right")
    end = np.minimum(end, start + cap)
    return order, start, end


@nb.njit(parallel=True, cache=True)
def _expected_f(pv, start, end, lam, beta2):
    n_s1 = len(start)
    m_best = np.zeros(n_s1, np.int32)
    zmax = 4
    pz = np.zeros(zmax + 1)
    pz[0] = np.exp(-lam)
    for z in range(1, zmax + 1):
        pz[z] = pz[z - 1] * lam / z
    for s in nb.prange(n_s1):
        n = end[s] - start[s]
        if n == 0:
            continue
        q = pv[start[s]:end[s]]
        # suffix Poisson-binomial distributions: suf[m] = distribution of Y over q[m:]
        suf = np.zeros((n + 1, n + 1))
        suf[n, 0] = 1.0
        for m in range(n - 1, -1, -1):
            for k in range(0, n - m + 1):
                v = suf[m + 1, k] * (1 - q[m])
                if k > 0:
                    v += suf[m + 1, k - 1] * q[m]
                suf[m, k] = v
        best = suf[0, 0] * pz[0]          # m = 0: F = 1 only if there is no true match at all
        bm = 0
        pre = np.zeros(n + 1)
        pre[0] = 1.0
        for m in range(1, n + 1):
            for k in range(m, 0, -1):     # add q[m-1] to the chosen set
                pre[k] = pre[k] * (1 - q[m - 1]) + pre[k - 1] * q[m - 1]
            pre[0] = pre[0] * (1 - q[m - 1])
            e = 0.0
            for x in range(1, m + 1):
                if pre[x] < 1e-12:
                    continue
                acc = 0.0
                for y in range(0, n - m + 1):
                    if suf[m, y] < 1e-12:
                        continue
                    for z in range(zmax + 1):
                        acc += suf[m, y] * pz[z] * (1 + beta2) * x / (m + beta2 * (x + y + z))
                e += pre[x] * acc
            if e > best:
                best = e
                bm = m
        m_best[s] = bm
    return m_best


def decode(s1, rec, p, n_s1, method="expected", t=0.5, lam=0.05, min_p=0.02, cap=30):
    """-> boolean mask over pairs: predicted matches."""
    return decode_kept(s1, np.where(exclusive(rec, p), p, 0.0), n_s1, method, t, lam, min_p, cap)


def decode_kept(s1, pk, n_s1, method="expected", t=0.5, lam=0.05, min_p=0.02, cap=30):
    """Same, on probabilities already reduced to each record's best S1 (others 0)."""
    if method == "threshold":
        return pk >= t
    order, start, end = group_sorted(s1, pk, n_s1, min_p, cap)
    pv = pk[order].astype(np.float64)
    m_best = _expected_f(pv, start.astype(np.int64), end.astype(np.int64), lam, 0.25)
    pred = np.zeros(len(pk), bool)
    s_of = s1[order]                                  # the first m_best[s] of each S1's sorted block
    rank = np.arange(len(order)) - start[s_of]
    pred[order[rank < m_best[s_of]]] = True
    return pred


def macro_f05(s1, pred, is_true, true_size, n_s1):
    """Macro F0.5 over all S1 (singletons included), true sets = full ground truth."""
    tp = np.bincount(s1[pred & is_true], minlength=n_s1)
    pp = np.bincount(s1[pred], minlength=n_s1)
    t = true_size
    f = np.where((t == 0) & (pp == 0), 1.0, 1.25 * tp / np.maximum(pp + 0.25 * t, 1e-9))
    return f
