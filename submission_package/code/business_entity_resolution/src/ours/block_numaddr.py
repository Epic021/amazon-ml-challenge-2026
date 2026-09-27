"""RET-BLOCK: address-structure blocking, then name re-ranking (roadmap: India recall).

B2 error analysis: blocking misses 6.1% of India's true pairs, mostly records whose NAME is in an
Indic script (or renamed) while the ADDRESS is still close to S1's ("Shop No-#40 ... Krishna Market
Kalkaji"). Name-first retrievers fail on those; a global name embedding cannot pick one name out of
~900k. So: narrow by address first, choose by name second.

  1. Each record -> composite keys "number|token" (house/unit number x each address word).
     These are rare by construction, so sharing one is strong evidence of the same premises.
  2. TF-IDF over the keys, S2/S3 -> S1 top-`wide` by cosine (per country).
  3. Re-rank those by  0.5 * key-cosine + 0.5 * name similarity  and keep the top `topq`.

  python src/block_numaddr.py --split train                    # full run -> data/cand/{split}_numaddr.parquet
  python src/block_numaddr.py --split train --eval_sample 5000 # realistic check on current union misses
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block import to_long  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))   # override for smoke runs
NORM = os.path.join(DATA, "norm")
CAND = os.path.join(DATA, "cand")
PQ = os.path.join(DATA, "parquet")


def keys(nums: str, addr: str) -> str:
    ns = nums.split()[:3]
    ts = {t for t in addr.split() if not t.isdigit() and len(t) > 1}
    return " ".join(f"{n}|{t}" for n in ns for t in ts)


def name_sim(a: list, b: list) -> np.ndarray:
    ts = process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    ns = process.cpdist([x.replace(" ", "") for x in a], [x.replace(" ", "") for x in b],
                        scorer=fuzz.ratio, workers=-1, dtype=np.float32)
    return np.maximum(ts, ns) / 100.0


def retrieve(s1: pd.DataFrame, q: pd.DataFrame, wide: int, topq: int, maxdf: int, threads: int) -> pd.DataFrame:
    vec = TfidfVectorizer(analyzer=str.split, lowercase=False, max_df=maxdf, min_df=1,
                          sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(pd.concat([s1.doc, q.doc], ignore_index=True)).tocsr()
    B, A = X[:len(s1)], X[len(s1):]
    C = sp_matmul_topn(A, B, top_n=wide, threshold=0.05, sort=True, n_threads=threads)
    d = to_long(C, q.id.values, s1.id.values, "rank_key").rename(columns={"r": "cand_id", "c": "s1_id"})
    core = pd.concat([s1.set_index("id").name_core, q.set_index("id").name_core])
    d["name"] = name_sim(core.reindex(d.cand_id.values).tolist(), core.reindex(d.s1_id.values).tolist())
    d["comb"] = 0.5 * d.score + 0.5 * d.name
    d = d.sort_values(["cand_id", "comb"], ascending=[True, False])
    d["rank_q"] = (d.groupby("cand_id").cumcount() + 1).astype(np.int16)
    d = d[d.rank_q <= topq]
    return pd.DataFrame({"s1_id": d.s1_id.values, "cand_id": d.cand_id.values,
                         "score": d.comb.values.astype(np.float32), "rank_q": d.rank_q.values,
                         "rank_s": np.int16(99)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--wide", type=int, default=50)
    ap.add_argument("--topq", type=int, default=5)
    ap.add_argument("--maxdf", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--eval_sample", type=int, default=0,
                    help="train only: score N true pairs the current union misses + N random queries")
    a = ap.parse_args()
    cols = ["id", "src", "country", "name_core", "addr_norm", "nums"]
    rec = pd.concat([pd.read_parquet(f"{NORM}/{a.split}_s{k}.parquet", columns=cols) for k in (1, 2, 3)],
                    ignore_index=True)
    rec["doc"] = [keys(n, ad) for n, ad in zip(rec.nums.values, rec.addr_norm.values)]
    rec = rec[rec.doc.str.len() > 0]
    s1_all, q_all = rec[rec.src == "S1"], rec[rec.src != "S1"]

    if a.eval_sample:
        truth = pd.read_parquet(f"{PQ}/train_pairs.parquet")
        cand = pd.read_parquet(f"{CAND}/train.parquet", columns=["s1_id", "cand_id"])
        miss = truth.merge(cand, on=["s1_id", "cand_id"], how="left", indicator=True)
        miss = miss[miss._merge == "left_only"][["s1_id", "cand_id"]]
        ctry = rec.set_index("id").country
        miss["country"] = miss.s1_id.map(ctry)
        miss = miss[miss.cand_id.isin(set(q_all.id))]           # needs a number in the address
        for country, g in miss.groupby("country"):
            g = g.sample(min(a.eval_sample, len(g)), random_state=0)
            rnd = q_all[q_all.country == country].sample(a.eval_sample, random_state=0)
            q = q_all[q_all.id.isin(set(g.cand_id) | set(rnd.id))]
            t = time.time()
            d = retrieve(s1_all[s1_all.country == country], q, a.wide, a.topq, a.maxdf, a.threads)
            hit = g.merge(d, on=["s1_id", "cand_id"], how="left")
            print(f"[{country}] union misses with a numbered address: {len(miss[miss.country == country]):,}; "
                  f"sample {len(g):,}: recall@{a.topq} = {hit.score.notna().mean():.3f} "
                  f"(@1 {(hit.rank_q == 1).mean():.3f}); pairs/query {len(d) / len(q):.2f}; {time.time() - t:.0f}s")
        return

    parts = []
    for country in sorted(rec.country.unique()):
        s1, q = s1_all[s1_all.country == country], q_all[q_all.country == country]
        if len(s1) == 0 or len(q) == 0:
            continue
        t = time.time()
        parts.append(retrieve(s1, q, a.wide, a.topq, a.maxdf, a.threads))
        print(f"[numaddr][{country}] S1={len(s1):,} queries={len(q):,} -> {len(parts[-1]):,} pairs in "
              f"{time.time() - t:.0f}s", flush=True)
    os.makedirs(CAND, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(f"{CAND}/{a.split}_numaddr.parquet", index=False)


if __name__ == "__main__":
    main()
