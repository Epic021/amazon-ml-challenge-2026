"""Pair features for candidate pairs.

Train S1 entities are split by id into folds (stable, id-based):
  fold 0-1  STATS  -> learns name-token log-odds table (never used to train/evaluate the model)
  fold 2-4  TRAIN  -> LightGBM training pairs
  fold 5    HOLD   -> holdout for early stopping, thresholds and the score we trust
  fold 6-9  REST   -> scored only so exclusive assignment sees all competitors

  python src/features.py --split train     # builds log-odds table + features for all train pairs
  python src/features.py --split test
"""
import argparse
import os
import re
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NORM = os.path.join(ROOT, "data", "norm")
CAND = os.path.join(ROOT, "data", "cand")
FEAT = os.path.join(ROOT, "data", "feat")
PQ = os.path.join(ROOT, "data", "parquet")

LEGAL = set("""private limited llc llp incorporated corporation company lp pllc pc plc gmbh sarl sas sasu eurl
sci sa ei selarl scp snc""".split())
_VOW = re.compile(r"(?<=.)[aeiouyh]")
_REP = re.compile(r"(.)\1+")
LO_ADD, LO_DROP = {}, {}
EQ_ADDR, EQ_NAME = {}, {}          # variant -> set of canonical forms (mined, see mine_equiv.py)


def fold_of(ids: pd.Series) -> np.ndarray:
    return (ids.str[3:].astype(np.int64) % 10).values


def skel_str(s: str) -> str:
    return " ".join(_REP.sub(r"\1", _VOW.sub("", t)) for t in s.split())


# ---------------------------------------------------------------- python-loop features
def _num_feats(a: str, b: str):
    A, B = a.split(), b.split()
    sa, sb = set(A), set(B)
    if not sa and not sb:
        rel = 0
    elif sa == sb:
        rel = 1
    elif sa < sb:
        rel = 2                        # candidate adds numbers
    elif sb < sa:
        rel = 3                        # candidate drops numbers
    elif sa & sb:
        rel = 4                        # partial overlap, some differ
    else:
        rel = 5                        # all numbers differ
    first_eq = -1 if not (A and B) else int(A[0] == B[0])
    s1first_in = -1 if not A else int(A[0] in sb)
    ua, ub = list(sa - sb)[:6], list(sb - sa)[:6]
    if ua and ub:
        min_lev = min(Levenshtein.distance(x, y) for x in ua for y in ub)
        prefix = int(any(x.startswith(y) or y.startswith(x) for x in ua for y in ub))
    else:
        min_lev, prefix = -1, -1
    if A and B:
        fa, fb = int(A[0][:9]), int(B[0][:9])
        fdiff = float(np.log1p(abs(fa - fb)))
    else:
        fdiff = -1.0
    return rel, first_eq, s1first_in, len(sa & sb), len(sa - sb), len(sb - sa), min_lev, prefix, fdiff


def _name_diff(a: str, b: str):
    ta, tb = set(a.split()), set(b.split())
    return tb - ta, ta - tb, ta, tb


def _explained(extra, other_side: set, table: dict) -> int:
    """How many of the tokens only on one side are a mined variant of something on the other side."""
    n = 0
    for y in extra:
        for x in table.get(y, ()):
            if x in other_side or (" " in x and set(x.split()) <= other_side):
                n += 1
                break
    return n


def _lo_stats(vals):
    if not vals:
        return 0.0, 0.0, 0.0
    return float(sum(vals)), float(max(vals)), float(min(vals))


def loop_feats(args):
    s1n, cn, s1c, cc, s1num, cnum, s1a, ca = args
    out = np.zeros((len(s1n), 29), dtype=np.float32)
    for i in range(len(s1n)):
        added, dropped, ta, tb = _name_diff(s1n[i], cn[i])
        la, lb = ta & LEGAL, tb & LEGAL
        lo_a = [LO_ADD[t] for t in added if t in LO_ADD]
        lo_d = [LO_DROP[t] for t in dropped if t in LO_DROP]
        sa, sb = set(s1c[i].split()), set(cc[i].split())
        aa = {t for t in s1a[i].split() if not t.isdigit()}
        ab = {t for t in ca[i].split() if not t.isdigit()}
        out[i, :9] = _num_feats(s1num[i], cnum[i])
        out[i, 9:12] = _lo_stats(lo_a)
        out[i, 12:15] = _lo_stats(lo_d)
        out[i, 15] = len(added)
        out[i, 16] = len(dropped)
        out[i, 17] = len(added) - len(lo_a)                     # added tokens never seen in stats
        out[i, 18] = len(dropped) - len(lo_d)
        out[i, 19] = int(bool(la) and bool(lb) and not (la & lb))  # legal form changed
        out[i, 20] = len(lb - la)
        out[i, 21] = int(s1c[i] == cc[i])
        out[i, 22] = int(bool(sa) and sa <= sb)
        out[i, 23] = int(bool(sb) and sb <= sa)
        out[i, 24] = len(aa & ab) / max(1, len(aa | ab))
        out[i, 25] = len(ab - aa)
        out[i, 26] = len(aa - ab)
        out[i, 27] = _explained(ab - aa, aa, EQ_ADDR)
        out[i, 28] = _explained(added, ta, EQ_NAME)
    return out


LOOP_COLS = ["num_rel", "num_first_eq", "num_s1first_in", "num_common", "num_s1_only", "num_c_only",
             "num_min_lev", "num_prefix", "num_first_logdiff",
             "add_lo_sum", "add_lo_max", "add_lo_min", "drop_lo_sum", "drop_lo_max", "drop_lo_min",
             "n_added", "n_dropped", "n_added_unknown", "n_dropped_unknown", "legal_conflict", "legal_added",
             "core_eq", "core_s1_in_c", "core_c_in_s1", "addr_tok_jacc", "addr_c_only", "addr_s1_only",
             "addr_diff_explained", "name_diff_explained"]


def diff_tokens(args):
    s1n, cn, y = args
    ca_p, ca_n, cd_p, cd_n = Counter(), Counter(), Counter(), Counter()
    for a, b, lab in zip(s1n, cn, y):
        added, dropped, _, _ = _name_diff(a, b)
        (ca_p if lab else ca_n).update(added)
        (cd_p if lab else cd_n).update(dropped)
    return ca_p, ca_n, cd_p, cd_n


def build_logodds(df: pd.DataFrame, procs: int, min_count: int = 30):
    chunks = [(df.s1_name.values[i:i + 500_000], df.c_name.values[i:i + 500_000], df.y.values[i:i + 500_000])
              for i in range(0, len(df), 500_000)]
    tot = [Counter(), Counter(), Counter(), Counter()]
    with Pool(procs) as p:
        for parts in p.imap_unordered(diff_tokens, chunks):
            for t, c in zip(tot, parts):
                t.update(c)
    npos, nneg = max(1, int(df.y.sum())), max(1, int((1 - df.y).sum()))

    def table(pos, neg):
        out = {}
        for t in set(pos) | set(neg):
            if pos[t] + neg[t] >= min_count:
                out[t] = float(np.log((pos[t] + 1) / npos) - np.log((neg[t] + 1) / nneg))
        return out
    return table(tot[0], tot[1]), table(tot[2], tot[3])


# ---------------------------------------------------------------- main
def context_feats(c: pd.DataFrame) -> pd.DataFrame:
    g1, gc = c.groupby("s1_id").score, c.groupby("cand_id").score
    c["s1_best"] = g1.transform("max")
    c["gap_s1"] = c.s1_best - c.score
    c["rank_in_s1"] = g1.rank(ascending=False, method="first").astype(np.int16)
    c["n_s1_cands"] = g1.transform("size").astype(np.int16)
    c["c_best"] = gc.transform("max")
    c["gap_c"] = c.c_best - c.score
    c["n_c_s1"] = gc.transform("size").astype(np.int16)
    srt = c[["cand_id", "score"]].sort_values(["cand_id", "score"], ascending=[True, False])
    srt["k"] = srt.groupby("cand_id").cumcount()
    c["c_second"] = c.cand_id.map(srt[srt.k == 1].set_index("cand_id").score).fillna(0)
    c["c_margin"] = c.c_best - c.c_second
    c["is_c_best"] = (c.gap_c == 0).astype(np.int8)
    c["mutual_best"] = ((c.gap_c == 0) & (c.gap_s1 == 0)).astype(np.int8)
    return c.drop(columns=["c_second"])


def attach_records(c: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    r = rec.set_index("id")
    for side, key in (("s1", "s1_id"), ("c", "cand_id")):
        sub = r.reindex(c[key].values)
        c[f"{side}_name"] = sub.name_norm.values
        c[f"{side}_core"] = sub.name_core.values
        c[f"{side}_addr"] = sub.addr_norm.values
        c[f"{side}_nums"] = sub.nums.values
        c[f"{side}_skel"] = sub.skel.values
        if side == "c":
            c["c_src_s3"] = (sub.src.values == "S3").astype(np.int8)
            c["c_nonlatin"] = sub.nonlatin.values.astype(np.int8)
            c["c_addr_empty"] = sub.addr_empty.values.astype(np.int8)
    return c


def string_feats(c: pd.DataFrame) -> pd.DataFrame:
    def cp(a, b, scorer):
        return process.cpdist(c[a].tolist(), c[b].tolist(), scorer=scorer, workers=-1, dtype=np.float32)
    f = {}
    f["core_ratio"] = cp("s1_core", "c_core", fuzz.ratio)
    f["core_tset"] = cp("s1_core", "c_core", fuzz.token_set_ratio)
    f["core_tsort"] = cp("s1_core", "c_core", fuzz.token_sort_ratio)
    f["core_partial"] = cp("s1_core", "c_core", fuzz.partial_ratio)
    f["core_jw"] = cp("s1_core", "c_core", JaroWinkler.normalized_similarity)
    f["name_ratio"] = cp("s1_name", "c_name", fuzz.ratio)
    f["name_tset"] = cp("s1_name", "c_name", fuzz.token_set_ratio)
    f["skel_ratio"] = cp("s1_skel", "c_skel", fuzz.ratio)
    f["skel_tset"] = cp("s1_skel", "c_skel", fuzz.token_set_ratio)
    f["addr_ratio"] = cp("s1_addr", "c_addr", fuzz.ratio)
    f["addr_tset"] = cp("s1_addr", "c_addr", fuzz.token_set_ratio)
    f["addr_tsort"] = cp("s1_addr", "c_addr", fuzz.token_sort_ratio)
    f["addr_partial"] = cp("s1_addr", "c_addr", fuzz.partial_ratio)
    f["core_len_ratio"] = (c.c_core.str.len() / c.s1_core.str.len().clip(lower=1)).astype(np.float32).values
    f["addr_len_ratio"] = (c.c_addr.str.len() / c.s1_addr.str.len().clip(lower=1)).astype(np.float32).values
    return pd.DataFrame(f, index=c.index)


def run_loop(c: pd.DataFrame, procs: int) -> pd.DataFrame:
    n = 200_000
    cols = ["s1_name", "c_name", "s1_core", "c_core", "s1_nums", "c_nums", "s1_addr", "c_addr"]
    chunks = [tuple(c[k].values[i:i + n] for k in cols) for i in range(0, len(c), n)]
    with Pool(procs) as p:
        arr = np.vstack(p.map(loop_feats, chunks))
    return pd.DataFrame(arr, columns=LOOP_COLS, index=c.index)


def load_equiv(split: str):
    tabs = [f"{ROOT}/data/equiv/train.parquet"] + ([f"{ROOT}/data/equiv/test.parquet"] if split == "test" else [])
    eq = pd.concat([pd.read_parquet(t) for t in tabs if os.path.isfile(t)], ignore_index=True)
    out = {"addr_norm": {}, "name_norm": {}}
    for f, v, c in eq[["field", "variant", "canonical"]].itertuples(index=False):
        out[f].setdefault(v, set()).add(c)
    return out["addr_norm"], out["name_norm"]


def main():
    global LO_ADD, LO_DROP, EQ_ADDR, EQ_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--chunk", type=int, default=6_000_000)
    ap.add_argument("--sample", type=float, default=1.0, help="fraction of S1 ids (smoke tests)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_rank_q", type=int, default=10)
    ap.add_argument("--max_rank_s", type=int, default=15)
    a = ap.parse_args()
    procs = os.cpu_count() or 4
    os.makedirs(FEAT, exist_ok=True)
    t0 = time.time()
    EQ_ADDR, EQ_NAME = load_equiv(a.split)
    print(f"equivalence rules: addr={len(EQ_ADDR)} name={len(EQ_NAME)}", flush=True)

    cand = pd.read_parquet(f"{CAND}/{a.split}.parquet")
    if a.sample < 1.0:
        cand = cand[(cand.s1_id.str[3:].astype(np.int64) % 1000) < a.sample * 1000]
    cand = context_feats(cand)                       # context over the full candidate list
    cand = cand[(cand.rank_q <= a.max_rank_q) | (cand.rank_s <= a.max_rank_s)].reset_index(drop=True)
    rec = pd.concat([pd.read_parquet(f"{NORM}/{a.split}_s{k}.parquet") for k in (1, 2, 3)], ignore_index=True)
    rec["skel"] = [skel_str(s) for s in rec.name_core.values]
    print(f"[{a.split}] {len(cand):,} pairs; context done {time.time() - t0:.0f}s", flush=True)

    if a.split == "train":
        truth = pd.read_parquet(f"{PQ}/train_pairs.parquet")
        truth["y"] = np.int8(1)
        cand = cand.merge(truth, on=["s1_id", "cand_id"], how="left")
        cand["y"] = cand.y.fillna(0).astype(np.int8)
        cand["fold"] = fold_of(cand.s1_id)
        stats = attach_records(cand[cand.fold <= 1][["s1_id", "cand_id", "y"]].copy(), rec)
        LO_ADD, LO_DROP = build_logodds(stats, procs)
        pd.DataFrame({"token": list(LO_ADD), "lo": list(LO_ADD.values())}).to_parquet(f"{FEAT}/lo_add.parquet")
        pd.DataFrame({"token": list(LO_DROP), "lo": list(LO_DROP.values())}).to_parquet(f"{FEAT}/lo_drop.parquet")
        top = sorted(LO_ADD.items(), key=lambda kv: kv[1])
        print(f"log-odds: add={len(LO_ADD):,} drop={len(LO_DROP):,}; most negative added: "
              f"{[t for t, _ in top[:15]]}; most positive added: {[t for t, _ in top[-15:]]}", flush=True)
        del stats
    else:
        LO_ADD = dict(pd.read_parquet(f"{FEAT}/lo_add.parquet").itertuples(index=False))
        LO_DROP = dict(pd.read_parquet(f"{FEAT}/lo_drop.parquet").itertuples(index=False))

    parts = []
    for i in range(0, len(cand), a.chunk):
        t1 = time.time()
        c = attach_records(cand.iloc[i:i + a.chunk].copy(), rec)
        c = pd.concat([c, string_feats(c), run_loop(c, procs)], axis=1)
        c = c.drop(columns=[f"{side}_{k}" for side in ("s1", "c")
                            for k in ("name", "core", "addr", "nums", "skel")])
        parts.append(c)
        print(f"  chunk {i // a.chunk}: {len(c):,} pairs in {time.time() - t1:.0f}s", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(a.out or f"{FEAT}/{a.split}.parquet", index=False)
    print(f"[{a.split}] features {out.shape} written in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
