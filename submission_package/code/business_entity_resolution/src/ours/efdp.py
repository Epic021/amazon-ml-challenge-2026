"""Exact expected-F0.5 top-k decoding (research_sota.md §4).

Per S1: candidates with (calibrated, independent) probabilities p, sorted descending, plus
M ~ Poisson(lam) true matches the blocker missed. F = 1.25*TP / (k + 0.25*T) with T = TP + rest + M;
empty prediction scores 1 iff T = 0. By the top-k theorem only the n+1 prefixes need scoring:

  E[0] = P(T = 0) = prod(1 - p) * exp(-lam)
  E[k] = sum_a sum_b A_k[a] * B_k[b] * 1.25 a / (k + 0.25 (a + b))
         A_k = Poisson-binomial of the first k,  B_k = Poisson-binomial of the rest (*) Poisson(lam)

Vectorised over batches of S1 with numpy (batched matmuls), no numba needed.
"""
import numpy as np
from math import factorial


def _poisson_pmf(lam: np.ndarray, m: int) -> np.ndarray:
    ks = np.arange(m)
    fact = np.array([factorial(int(k)) for k in ks], dtype=np.float64)
    pmf = np.exp(-lam[:, None]) * lam[:, None] ** ks[None, :] / fact[None, :]
    pmf[:, -1] += np.clip(1 - pmf.sum(1), 0, None)          # fold the tail into the last bucket
    return pmf


def best_k(P: np.ndarray, lam: np.ndarray, m_pois: int = 8) -> np.ndarray:
    """P: (G, N) probabilities sorted descending per row, zero-padded. lam: (G,). Returns k* per row."""
    G, N = P.shape
    P = P.astype(np.float64)
    # prefix distributions A[g, k, a], k = 0..N
    A = np.zeros((G, N + 1, N + 1))
    A[:, 0, 0] = 1.0
    for k in range(1, N + 1):
        p = P[:, k - 1:k]
        A[:, k, :] = A[:, k - 1, :] * (1 - p)
        A[:, k, 1:] += A[:, k - 1, :-1] * p
    # suffix distributions S[g, k, b] over items k+1..N
    S = np.zeros((G, N + 1, N + 1))
    S[:, N, 0] = 1.0
    for k in range(N - 1, -1, -1):
        p = P[:, k:k + 1]
        S[:, k, :] = S[:, k + 1, :] * (1 - p)
        S[:, k, 1:] += S[:, k + 1, :-1] * p
    # convolve suffix with Poisson(lam) -> B[g, k, b], b = 0..N+m-1
    pois = _poisson_pmf(lam.astype(np.float64), m_pois)                  # (G, m)
    NB = N + m_pois
    B = np.zeros((G, N + 1, NB))
    for j in range(m_pois):
        B[:, :, j:j + N + 1] += S * pois[:, None, j:j + 1]
    # weight tensor W[k, a, b] = 1.25 a / (k + 0.25 (a + b)), k >= 1
    k_ = np.arange(N + 1)[:, None, None]
    a_ = np.arange(N + 1)[None, :, None]
    b_ = np.arange(NB)[None, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.where(k_ >= 1, 1.25 * a_ / (k_ + 0.25 * (a_ + b_)), 0.0)
    # C[k, g, a] = sum_b B[g, k, b] W[k, a, b]  (batched matmul over k)
    C = np.matmul(B.transpose(1, 0, 2), W.transpose(0, 2, 1))           # (K, G, A)
    E = np.einsum("gka,kga->gk", A, C)                                   # (G, K)
    E[:, 0] = B[:, 0, 0]                                                 # P(T = 0)
    n_real = (P > 0).sum(1)
    E[np.arange(N + 1)[None, :] > n_real[:, None]] = -1.0                # never pick padding
    return E.argmax(1)


def brute_force(p: np.ndarray, lam: float, m_pois: int = 8) -> int:
    """Enumerate all label vectors (small n) — for testing best_k."""
    import itertools
    n = len(p)
    pois = _poisson_pmf(np.array([lam]), m_pois)[0]
    best, arg = -1, 0
    for k in range(n + 1):
        e = 0.0
        for ys in itertools.product([0, 1], repeat=n):
            pr = np.prod([pi if y else 1 - pi for pi, y in zip(p, ys)])
            tp, T0 = sum(ys[:k]), sum(ys)
            for mm in range(m_pois):
                T = T0 + mm
                f = (1.0 if T == 0 else 0.0) if k == 0 else 1.25 * tp / (k + 0.25 * T)
                e += pr * pois[mm] * f
        if e > best + 1e-12:
            best, arg = e, k
    return arg


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(300):
        n = rng.integers(1, 7)
        p = np.sort(rng.random(n) ** rng.uniform(0.3, 3))[::-1]
        lam = float(rng.choice([0.0, 0.05, 0.3, 1.0]))
        Pm = np.zeros((1, 8)); Pm[0, :n] = p
        k1, k2 = best_k(Pm, np.array([lam]))[0], brute_force(p, lam)
        bad += k1 != k2
    print("mismatches vs brute force:", bad, "/ 300")
