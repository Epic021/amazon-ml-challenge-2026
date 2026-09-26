"""Dense cross-script name retriever (roadmap: India recall).

Error analysis on B2: blocking misses 6.1% of India's true pairs, mostly S2/S3 names written in an
Indic script (सिल्वर इन्वेस्टमेंट प्राइवेट लिमिटेड vs Silver Investment Private Limited) whose
transliteration shares few character n-grams with the English S1 name.

A multilingual static embedding model (model2vec potion-multilingual-128M, MIT) embeds the RAW names
of (a) every S2/S3 record whose name contains non-Latin letters and (b) every S1 record of the same
country; each such record keeps its top-k S1 by cosine. Countries without non-Latin queries are skipped.

  python src/block_emb.py --split train        # writes data/cand/{split}_emb.parquet
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))   # override for smoke runs
NORM = os.path.join(DATA, "norm")
PQ = os.path.join(DATA, "parquet")
CAND = os.path.join(DATA, "cand")
MODEL = "minishlab/potion-multilingual-128M"    # MIT licence


def embed(model, texts: list) -> np.ndarray:
    e = model.encode(texts, batch_size=4096, show_progress_bar=False).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True).clip(min=1e-9)
    return e


def topk_cosine(Q: np.ndarray, S: np.ndarray, k: int, chunk: int = 8192):
    idx = np.empty((len(Q), k), dtype=np.int64)
    sc = np.empty((len(Q), k), dtype=np.float32)
    for i in range(0, len(Q), chunk):
        M = Q[i:i + chunk] @ S.T
        part = np.argpartition(-M, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(M, part, axis=1)
        o = np.argsort(-ps, axis=1)
        idx[i:i + chunk] = np.take_along_axis(part, o, axis=1)
        sc[i:i + chunk] = np.take_along_axis(ps, o, axis=1)
    return idx, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--topq", type=int, default=10)
    a = ap.parse_args()
    from model2vec import StaticModel
    model = StaticModel.from_pretrained(MODEL)
    os.makedirs(CAND, exist_ok=True)

    raw = pd.concat([pd.read_parquet(f"{PQ}/{a.split}_s{k}.parquet", columns=["entity_id", "business_name"])
                     for k in (1, 2, 3)], ignore_index=True).set_index("entity_id").business_name
    norm = pd.concat([pd.read_parquet(f"{NORM}/{a.split}_s{k}.parquet", columns=["id", "src", "country", "nonlatin"])
                      for k in (1, 2, 3)], ignore_index=True)
    parts = []
    for country in sorted(norm.country.unique()):           # open set of countries
        q = norm[(norm.country == country) & (norm.src != "S1") & (norm.nonlatin == 1)]
        if len(q) == 0:
            continue
        s1 = norm[(norm.country == country) & (norm.src == "S1")]
        t = time.time()
        S = embed(model, raw.reindex(s1.id.values).tolist())
        Q = embed(model, raw.reindex(q.id.values).tolist())
        k = min(a.topq, len(s1))
        idx, sc = topk_cosine(Q, S, k)
        d = pd.DataFrame({"cand_id": np.repeat(q.id.values, k), "s1_id": s1.id.values[idx.ravel()],
                          "score": sc.ravel(), "rank_q": np.tile(np.arange(1, k + 1, dtype=np.int16), len(q)),
                          "rank_s": np.int16(99)})
        parts.append(d)
        print(f"[emb][{country}] S1={len(s1):,} non-Latin queries={len(q):,} -> {len(d):,} pairs in {time.time() - t:.0f}s",
              flush=True)
    cols = ["s1_id", "cand_id", "score", "rank_q", "rank_s"]
    cand = pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)
    cand.to_parquet(f"{CAND}/{a.split}_emb.parquet", index=False)
    if a.split == "train" and len(cand):
        truth = pd.read_parquet(f"{PQ}/train_pairs.parquet")
        nl = truth[truth.cand_id.isin(set(norm.id[norm.nonlatin == 1]))]
        hit = nl.merge(cand, on=["s1_id", "cand_id"], how="left")
        print(f"recall on true pairs with a non-Latin candidate name: {hit.score.notna().mean():.4f} "
              f"(@1 {(hit.rank_q == 1).mean():.4f}, @5 {(hit.rank_q <= 5).mean():.4f})")


if __name__ == "__main__":
    main()
