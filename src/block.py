"""Candidate generation: token TF-IDF cosine, top-k in both directions, per country.

  S2/S3 -> S1 : each record keeps its top-`topq` S1 (each record has <= 1 true S1, so this is sharp)
  S1 -> S2/S3 : each S1 keeps its top-`tops` records (covers S1 with many matches)
Union, then cap each S1 at `cap` candidates by score.

  python src/block.py --split train      # also reports recall ceiling against ground truth
  python src/block.py --split test
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NORM = os.path.join(ROOT, "data", "norm")
CAND = os.path.join(ROOT, "data", "cand")
PQ = os.path.join(ROOT, "data", "parquet")

_VOW = re.compile(r"(?<=.)[aeiouyh]")
_REP = re.compile(r"(.)\1+")


def skel(t: str) -> str:
    """Consonant skeleton: typo/transliteration-tolerant key (marketing -> mrktng)."""
    return _REP.sub(r"\1", _VOW.sub("", t))


def doc(name_core: str, addr: str) -> str:
    n = name_core.split()
    toks = ["n_" + t for t in n]
    toks += ["k_" + s for s in map(skel, n) if len(s) >= 3]
    toks += ["a_" + t for t in addr.split()]
    return " ".join(toks)


def to_long(C: sparse.csr_matrix, row_ids: np.ndarray, col_ids: np.ndarray, rank_col: str) -> pd.DataFrame:
    C = C.tocsr()
    counts = np.diff(C.indptr)
    rows = np.repeat(np.arange(C.shape[0]), counts)
    # rank inside each row by descending score (1 = best)
    order = np.lexsort((-C.data, rows))
    pos = np.empty(len(order), dtype=np.int64)
    pos[order] = np.arange(len(order))
    rank = (pos - np.repeat(C.indptr[:-1], counts) + 1).astype(np.int16)
    return pd.DataFrame({"r": row_ids[rows], "c": col_ids[C.indices], "score": C.data.astype(np.float32),
                         rank_col: rank})


def block_country(s1: pd.DataFrame, q: pd.DataFrame, topq: int, tops: int, maxdf: int, cap: int,
                  threads: int) -> pd.DataFrame:
    docs = pd.concat([s1.doc, q.doc], ignore_index=True)
    vec = TfidfVectorizer(analyzer=str.split, lowercase=False, max_df=maxdf, min_df=1,
                          sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(docs).tocsr()
    B, A = X[:len(s1)], X[len(s1):]
    s1_ids, q_ids = s1.id.values, q.id.values

    t = time.time()
    Cq = sp_matmul_topn(A, B, top_n=topq, threshold=0.02, sort=True, n_threads=threads)
    dq = to_long(Cq, q_ids, s1_ids, "rank_q").rename(columns={"r": "cand_id", "c": "s1_id"})
    t1 = time.time()
    Cs = sp_matmul_topn(B, A, top_n=tops, threshold=0.02, sort=True, n_threads=threads)
    ds = to_long(Cs, s1_ids, q_ids, "rank_s").rename(columns={"r": "s1_id", "c": "cand_id"})
    t2 = time.time()
    print(f"    vocab={len(vec.vocabulary_):,} q->s1 {t1 - t:.0f}s  s1->q {t2 - t1:.0f}s", flush=True)

    d = dq.merge(ds, on=["s1_id", "cand_id"], how="outer", suffixes=("", "_s"))
    d["score"] = d.score.fillna(d.score_s)
    d = d.drop(columns="score_s")
    d["rank_q"] = d.rank_q.fillna(99).astype(np.int16)
    d["rank_s"] = d.rank_s.fillna(99).astype(np.int16)
    d = d.sort_values(["s1_id", "score"], ascending=[True, False])
    d = d[d.groupby("s1_id").cumcount() < cap]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--topq", type=int, default=10)
    ap.add_argument("--tops", type=int, default=40)
    ap.add_argument("--maxdf", type=int, default=30000)
    ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    os.makedirs(CAND, exist_ok=True)

    cols = ["id", "country", "name_core", "addr_norm"]
    s1 = pd.read_parquet(f"{NORM}/{a.split}_s1.parquet", columns=cols)
    q = pd.concat([pd.read_parquet(f"{NORM}/{a.split}_s{k}.parquet", columns=cols) for k in (2, 3)],
                  ignore_index=True)
    for df in (s1, q):
        df["doc"] = [doc(n, ad) for n, ad in zip(df.name_core.values, df.addr_norm.values)]

    parts = []
    for country in sorted(s1.country.unique()):      # open set: whatever countries the file has
        t0 = time.time()
        cs, cq = s1[s1.country == country], q[q.country == country]
        print(f"[{country}] S1={len(cs):,} S2/S3={len(cq):,}", flush=True)
        parts.append(block_country(cs, cq, a.topq, a.tops, a.maxdf, a.cap, a.threads))
        print(f"    -> {len(parts[-1]):,} pairs in {time.time() - t0:.0f}s", flush=True)
    cand = pd.concat(parts, ignore_index=True)
    cand.to_parquet(f"{CAND}/{a.split}.parquet", index=False)
    per = cand.groupby("s1_id").size()
    print(f"total pairs {len(cand):,}; per S1 mean {per.mean():.1f}, S1 with none: {len(s1) - len(per):,}")

    if a.split == "train":
        truth = pd.read_parquet(f"{PQ}/train_pairs.parquet")
        hit = truth.merge(cand[["s1_id", "cand_id", "rank_q", "rank_s"]], on=["s1_id", "cand_id"], how="left")
        found = hit.rank_q.notna()
        print(f"pair recall (ceiling) = {found.mean():.4f}")
        hit["country"] = hit.s1_id.map(s1.set_index("id").country)
        hit["src"] = hit.cand_id.str[:2]
        print(hit.assign(found=found).groupby(["country", "src"]).found.mean().round(4).to_string())
        for k in (1, 3, 5, 10):
            print(f"  true pairs with rank_q<={k}: {(hit.rank_q <= k).mean():.4f}")


if __name__ == "__main__":
    main()
