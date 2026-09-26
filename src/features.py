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
import shutil
import threading
from multiprocessing import Pool

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))   # override for smoke runs
NORM = os.path.join(DATA, "norm")
CAND = os.path.join(DATA, "cand")
FEAT = os.path.join(DATA, "feat")
PQ = os.path.join(DATA, "parquet")

LEGAL = set("""private limited llc llp incorporated corporation company lp pllc pc plc gmbh sarl sas sasu eurl
sci sa ei selarl scp snc""".split())
_VOW = re.compile(r"(?<=.)[aeiouyh]")
_REP = re.compile(r"(.)\1+")
LO_ADD, LO_DROP = {}, {}
EQ_ADDR, EQ_NAME = {}, {}          # variant -> set of canonical forms (mined, see mine_equiv.py)
PLO_ADD, PLO_DROP, S1DF = {}, {}, {}  # per-country, label-free word roles (see build_role_tables)
ALO_ADD, ALO_DROP = {}, {}            # label log-odds of added / dropped ADDRESS words (folds 0-1)
_DIGRAPH = [("ph", "f"), ("sh", "s"), ("ch", "c"), ("kh", "k"), ("gh", "g"), ("th", "t"),
            ("dh", "d"), ("bh", "b"), ("ck", "k")]
_LETTER = str.maketrans({"c": "k", "q": "k", "z": "s", "x": "s", "j": "g", "w": "v", "y": "a"})
_NASAL = re.compile(r"m(?=[bcdfghjklnpqrstvwxz])")


def fold_of(ids: pd.Series) -> np.ndarray:
    return (ids.str[3:].astype(np.int64) % 10).values


def sound_token(t: str) -> str:
    """Sound-alike key: digraphs, nasal m before a consonant -> n, c/q->k, z/x->s, j->g, w->v, y->a,
    then drop vowels after the first letter and collapse repeats (praivet -> prvt = private)."""
    for a_, b_ in _DIGRAPH:
        t = t.replace(a_, b_)
    t = _NASAL.sub("n", t).translate(_LETTER)
    return _REP.sub(r"\1", t[:1] + re.sub(r"[aeiou]", "", t[1:]))


def sound_str(s: str) -> str:
    return " ".join(sound_token(t) for t in s.split())


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


def _numx_feats(a: str, b: str):
    """Finer number relations (EDA: truncation 516->51 is typo-like; a gap <= 10 is decoy-like;
    7-15 vs 715 is the same number)."""
    A, B = a.split(), b.split()
    sa, sb = set(A), set(B)
    ua, ub = list(sa - sb)[:6], list(sb - sa)[:6]
    suffix, gap = -1, -1.0
    if ua and ub:
        suffix = int(any(x.endswith(y) or y.endswith(x) for x in ua for y in ub))
        gap = float(min(abs(int(x[:9]) - int(y[:9])) for x in ua for y in ub))
    concat = int(bool(A) and bool(B) and A != B and "".join(A) == "".join(B))
    return suffix, concat, (np.log1p(gap) if gap >= 0 else -1.0), (int(0 <= gap <= 10) if gap >= 0 else -1)


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
    s1n, cn, s1c, cc, s1num, cnum, s1a, ca, ctry = args
    out = np.zeros((len(s1n), 48), dtype=np.float32)
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
        # per-country, label-free roles of the added / dropped words
        pa_t, pd_t, dfm = PLO_ADD.get(ctry[i], {}), PLO_DROP.get(ctry[i], {}), S1DF.get(ctry[i], {})
        out[i, 29:32] = _lo_stats([pa_t[t] for t in added if t in pa_t])
        out[i, 32:34] = _lo_stats([pd_t[t] for t in dropped if t in pd_t])[:2]
        gen = [dfm.get(t, -20.0) for t in added]
        out[i, 34] = max(gen) if gen else -30.0                   # most generic added word (log S1-name share)
        out[i, 35] = sum(t not in dfm for t in added)             # added words never seen in this country's S1 names
        out[i, 36:40] = _numx_feats(s1num[i], cnum[i])
        a_add, a_drop = ab - aa, aa - ab                          # address words (no digits) added / dropped
        lo_aa = [ALO_ADD[t] for t in a_add if t in ALO_ADD]
        lo_ad = [ALO_DROP[t] for t in a_drop if t in ALO_DROP]
        out[i, 40:43] = _lo_stats(lo_aa)
        out[i, 43:46] = _lo_stats(lo_ad)
        out[i, 46] = len(a_add) - len(lo_aa)
        out[i, 47] = len(a_drop) - len(lo_ad)
    return out


LOOP_COLS = ["num_rel", "num_first_eq", "num_s1first_in", "num_common", "num_s1_only", "num_c_only",
             "num_min_lev", "num_prefix", "num_first_logdiff",
             "add_lo_sum", "add_lo_max", "add_lo_min", "drop_lo_sum", "drop_lo_max", "drop_lo_min",
             "n_added", "n_dropped", "n_added_unknown", "n_dropped_unknown", "legal_conflict", "legal_added",
             "core_eq", "core_s1_in_c", "core_c_in_s1", "addr_tok_jacc", "addr_c_only", "addr_s1_only",
             "addr_diff_explained", "name_diff_explained",
             "padd_sum", "padd_max", "padd_min", "pdrop_sum", "pdrop_max", "add_gen_max", "add_novel",
             "numx_suffix", "numx_concat_eq", "numx_min_logdiff", "numx_gap_le10",
             "aadd_lo_sum", "aadd_lo_max", "aadd_lo_min", "adrop_lo_sum", "adrop_lo_max", "adrop_lo_min",
             "aadd_unknown", "adrop_unknown"]


def diff_tokens(args):
    s1n, cn, y = args
    ca_p, ca_n, cd_p, cd_n = Counter(), Counter(), Counter(), Counter()
    for a, b, lab in zip(s1n, cn, y):
        added, dropped, _, _ = _name_diff(a, b)
        (ca_p if lab else ca_n).update(added)
        (cd_p if lab else cd_n).update(dropped)
    return ca_p, ca_n, cd_p, cd_n


def _lo_table(pos: Counter, neg: Counter, npos: int, nneg: int, min_count: int) -> dict:
    return {t: float(np.log((pos[t] + 1) / npos) - np.log((neg[t] + 1) / nneg))
            for t in set(pos) | set(neg) if pos[t] + neg[t] >= min_count}


def _numrel_chunk(args):
    a, b = args
    return np.array([_num_feats(x, y)[0] for x, y in zip(a, b)], dtype=np.int8)


def build_role_tables(cand: pd.DataFrame, rec: pd.DataFrame, procs: int, per_country: int = 3_000_000):
    """Label-free, per-country word roles, recomputed on whatever split is being processed.

    Among best-ranked candidate pairs with near-identical addresses (token-set >= 90), a DIFFERENT
    house number is almost always a decoy (train: 88% of such negatives) and an EQUAL one mostly a
    match. Log-odds of each word being ADDED/DROPPED in equal-number vs different-number pairs
    therefore learns the decoy vocabulary of any language (holdings / developpement / participations)
    without labels. Also: how common each word is in the country's S1 names.
    """
    r = rec.set_index("id")
    pairs = cand.loc[cand.rank_q == 1, ["s1_id", "cand_id"]]
    s1r, cr = r.reindex(pairs.s1_id.values), r.reindex(pairs.cand_id.values)
    df = pd.DataFrame({"country": s1r.country.values, "s1n": s1r.name_norm.values, "cn": cr.name_norm.values,
                       "s1a": s1r.addr_norm.values, "ca": cr.addr_norm.values,
                       "s1u": s1r.nums.values, "cu": cr.nums.values})
    df = pd.concat([g.sample(min(len(g), per_country), random_state=0) for _, g in df.groupby("country")])
    df["at"] = process.cpdist(df.s1a.tolist(), df.ca.tolist(), scorer=fuzz.token_set_ratio, workers=-1)
    n = 200_000
    with Pool(procs) as p:
        df["rel"] = np.concatenate(p.map(_numrel_chunk, [(df.s1u.values[i:i + n], df.cu.values[i:i + n])
                                                          for i in range(0, len(df), n)]))
    lab = df[(df["at"] >= 90) & df.rel.isin([1, 5])].assign(y=lambda d: (d.rel == 1).astype(np.int8))
    plo_add, plo_drop = {}, {}
    for ctry, g in lab.groupby("country"):
        chunks = [(g.s1n.values[i:i + n], g.cn.values[i:i + n], g.y.values[i:i + n]) for i in range(0, len(g), n)]
        tot = [Counter() for _ in range(4)]
        with Pool(procs) as p:
            for parts in p.imap_unordered(diff_tokens, chunks):
                for t, c in zip(tot, parts):
                    t.update(c)
        npos, nneg = max(1, int(g.y.sum())), max(1, int((1 - g.y).sum()))
        plo_add[ctry] = _lo_table(tot[0], tot[1], npos, nneg, 20)
        plo_drop[ctry] = _lo_table(tot[2], tot[3], npos, nneg, 20)
        top = sorted(plo_add[ctry].items(), key=lambda kv: kv[1])
        print(f"  roles[{ctry}]: {npos:,} equal-number vs {nneg:,} different-number near-address pairs; "
              f"decoy-like added words: {[t for t, _ in top[:12]]}", flush=True)
    s1df = {}
    for ctry, g in rec[rec.src == "S1"].groupby("country"):
        cnt = Counter(t for s_ in g.name_norm.values for t in set(s_.split()))
        s1df[ctry] = {t: float(np.log(c / len(g))) for t, c in cnt.items() if c >= 3}
    return plo_add, plo_drop, s1df


def build_logodds(df: pd.DataFrame, procs: int, min_count: int = 30):
    chunks = [(df.s1_name.values[i:i + 500_000], df.c_name.values[i:i + 500_000], df.y.values[i:i + 500_000])
              for i in range(0, len(df), 500_000)]
    tot = [Counter(), Counter(), Counter(), Counter()]
    with Pool(procs) as p:
        for parts in p.imap_unordered(diff_tokens, chunks):
            for t, c in zip(tot, parts):
                t.update(c)
    npos, nneg = max(1, int(df.y.sum())), max(1, int((1 - df.y).sum()))
    return _lo_table(tot[0], tot[1], npos, nneg, min_count), _lo_table(tot[2], tot[3], npos, nneg, min_count)


# ---------------------------------------------------------------- main
def context_feats(c: pl.DataFrame) -> pl.DataFrame:
    """Rank context over the full candidate list (polars windows, all cores)."""
    c = c.with_columns(
        pl.col("score").max().over("s1_id").alias("s1_best"),
        pl.col("score").rank("ordinal", descending=True).over("s1_id").cast(pl.Int16).alias("rank_in_s1"),
        pl.len().over("s1_id").cast(pl.Int16).alias("n_s1_cands"),
        pl.col("score").max().over("cand_id").alias("c_best"),
        pl.len().over("cand_id").cast(pl.Int16).alias("n_c_s1"),
        pl.col("score").top_k(2).min().over("cand_id").alias("_top2min"),
    )
    c = c.with_columns(
        (pl.col("s1_best") - pl.col("score")).alias("gap_s1"),
        (pl.col("c_best") - pl.col("score")).alias("gap_c"),
        pl.when(pl.col("n_c_s1") >= 2).then(pl.col("_top2min")).otherwise(0.0).alias("_c_second"),
    )
    return c.with_columns(
        (pl.col("c_best") - pl.col("_c_second")).alias("c_margin"),
        (pl.col("gap_c") == 0).cast(pl.Int8).alias("is_c_best"),
        ((pl.col("gap_c") == 0) & (pl.col("gap_s1") == 0)).cast(pl.Int8).alias("mutual_best"),
    ).drop(["_top2min", "_c_second"])


def add_name_counts(rec: pd.DataFrame) -> pd.DataFrame:
    """Name uniqueness within the country (G5), computed from the split's own records:
    n_s1 = how many S1 records share this record's name core, n_q = how many S2/S3 records do.
    A unique name makes an empty-address candidate a safe match; a common one does not."""
    k = rec.country + "|" + rec.name_core
    is_s1 = (rec.src == "S1").values
    cs1, cq = k[is_s1].value_counts(), k[~is_s1].value_counts()
    empty = (rec.name_core == "").values
    rec["n_s1"] = np.where(empty, -1, k.map(cs1).fillna(0).values).astype(np.float32)
    rec["n_q"] = np.where(empty, -1, k.map(cq).fillna(0).values).astype(np.float32)
    return rec


NAME_COUNT_COLS = ["s1_core_n_s1", "c_core_n_s1", "c_core_n_q"]


def name_count_cols(c: pd.DataFrame, rec_indexed: pd.DataFrame) -> pd.DataFrame:
    s1 = rec_indexed.reindex(c.s1_id.values)
    cd = rec_indexed.reindex(c.cand_id.values)
    return pd.DataFrame({"s1_core_n_s1": s1.n_s1.values, "c_core_n_s1": cd.n_s1.values,
                         "c_core_n_q": cd.n_q.values}, index=c.index)


def twin_feats(cand: pl.DataFrame, rec: pd.DataFrame) -> pl.DataFrame:
    """Twin agreement (teammate EDA twin_signal: when the number differs from S1, an other-source record
    under the same S1 with the same number lifts P(true) 40% -> 90% for far-off numbers).
    twin_num  = other-source candidates of the same S1 with the identical number set
    twin_name = other-source candidates of the same S1 with the identical core name"""
    r = pl.from_pandas(rec[["id", "src", "nums", "name_core"]]).rename(
        {"id": "cand_id", "src": "_src", "nums": "_nums", "name_core": "_core"})
    c = cand.join(r, on="cand_id", how="left")
    c = c.with_columns(
        (pl.len().over(["s1_id", "_nums"]) - pl.len().over(["s1_id", "_nums", "_src"])).alias("_tn"),
        (pl.len().over(["s1_id", "_core"]) - pl.len().over(["s1_id", "_core", "_src"])).alias("_tc"))
    return c.with_columns(
        pl.when(pl.col("_nums") == "").then(-1).otherwise(pl.col("_tn")).cast(pl.Int16).alias("twin_num"),
        pl.when(pl.col("_core") == "").then(-1).otherwise(pl.col("_tc")).cast(pl.Int16).alias("twin_name"),
    ).drop(["_src", "_nums", "_core", "_tn", "_tc"])


class RecStore:
    """Records as numpy arrays + an id -> row index, built once (lookups by get_indexer + take)."""

    def __init__(self, rec: pd.DataFrame):
        self.idx = pd.Index(rec.id.values)
        self.a = {k: rec[k].values for k in rec.columns if k != "id"}

    def take(self, ids, col):
        return self.a[col][self.idx.get_indexer(ids)]


def attach_records(c: pd.DataFrame, store: "RecStore") -> pd.DataFrame:
    for side, key in (("s1", "s1_id"), ("c", "cand_id")):
        pos = store.idx.get_indexer(c[key].values)
        a = store.a
        c[f"{side}_name"] = a["name_norm"][pos]
        c[f"{side}_core"] = a["name_core"][pos]
        c[f"{side}_addr"] = a["addr_norm"][pos]
        c[f"{side}_nums"] = a["nums"][pos]
        c[f"{side}_skel"] = a["skel"][pos]
        c[f"{side}_snd"] = a["snd"][pos]
        if side == "s1":
            c["country"] = a["country"][pos]
            if "n_s1" in a:
                c["s1_core_n_s1"] = a["n_s1"][pos]
        if side == "c":
            c["c_src_s3"] = (a["src"][pos] == "S3").astype(np.int8)
            c["c_nonlatin"] = a["nonlatin"][pos].astype(np.int8)
            c["c_addr_empty"] = a["addr_empty"][pos].astype(np.int8)
            if "n_s1" in a:
                c["c_core_n_s1"] = a["n_s1"][pos]
                c["c_core_n_q"] = a["n_q"][pos]
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
    f["snd_ratio"] = cp("s1_snd", "c_snd", fuzz.ratio)
    f["snd_tset"] = cp("s1_snd", "c_snd", fuzz.token_set_ratio)
    f["addr_ratio"] = cp("s1_addr", "c_addr", fuzz.ratio)
    f["addr_tset"] = cp("s1_addr", "c_addr", fuzz.token_set_ratio)
    f["addr_tsort"] = cp("s1_addr", "c_addr", fuzz.token_sort_ratio)
    f["addr_partial"] = cp("s1_addr", "c_addr", fuzz.partial_ratio)
    f["core_len_ratio"] = (c.c_core.str.len() / c.s1_core.str.len().clip(lower=1)).astype(np.float32).values
    f["addr_len_ratio"] = (c.c_addr.str.len() / c.s1_addr.str.len().clip(lower=1)).astype(np.float32).values
    return pd.DataFrame(f, index=c.index)


def run_loop(c: pd.DataFrame, pool, procs: int) -> pd.DataFrame:
    n = max(20_000, len(c) // (procs * 3) + 1)
    cols = ["s1_name", "c_name", "s1_core", "c_core", "s1_nums", "c_nums", "s1_addr", "c_addr", "country"]
    chunks = [tuple(c[k].values[i:i + n] for k in cols) for i in range(0, len(c), n)]
    arr = np.vstack(pool.map(loop_feats, chunks))
    return pd.DataFrame(arr, columns=LOOP_COLS, index=c.index)


def _skel_chunk(names):
    return [skel_str(x) for x in names]


def _snd_chunk(names):
    return [sound_str(x) for x in names]


def _addr_words(strs):
    return [" ".join(t for t in x.split() if not t.isdigit()) for x in strs]


def load_equiv(split: str):
    tabs = [f"{DATA}/equiv/train.parquet"] + ([f"{DATA}/equiv/test.parquet"] if split == "test" else [])
    eq = pd.concat([pd.read_parquet(t) for t in tabs if os.path.isfile(t)], ignore_index=True)
    out = {"addr_norm": {}, "name_norm": {}}
    for f, v, c in eq[["field", "variant", "canonical"]].itertuples(index=False):
        out[f].setdefault(v, set()).add(c)
    return out["addr_norm"], out["name_norm"]


def main():
    global LO_ADD, LO_DROP, EQ_ADDR, EQ_NAME, PLO_ADD, PLO_DROP, S1DF, ALO_ADD, ALO_DROP
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--chunk", type=int, default=6_000_000)
    ap.add_argument("--sample", type=float, default=1.0, help="fraction of S1 ids (smoke tests)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_rank_q", type=int, default=10)
    ap.add_argument("--max_rank_s", type=int, default=15)
    ap.add_argument("--no_label_lo", action="store_true",
                    help="drop the label-based word log-odds (keep only language-independent roles)")
    a = ap.parse_args()
    procs = os.cpu_count() or 4
    os.makedirs(FEAT, exist_ok=True)
    t0 = time.time()
    EQ_ADDR, EQ_NAME = load_equiv(a.split)
    print(f"equivalence rules: addr={len(EQ_ADDR)} name={len(EQ_NAME)}", flush=True)

    cand = pl.read_parquet(f"{CAND}/{a.split}.parquet")
    if a.sample < 1.0:
        cand = cand.filter(pl.col("s1_id").str.slice(3).cast(pl.Int64) % 1000 < a.sample * 1000)
    cand = context_feats(cand)                       # context over the full candidate list
    cand = cand.filter((pl.col("rank_q") <= a.max_rank_q) | (pl.col("rank_s") <= a.max_rank_s))
    rec = pd.concat([pd.read_parquet(f"{NORM}/{a.split}_s{k}.parquet") for k in (1, 2, 3)], ignore_index=True)
    with Pool(procs) as p:
        step = len(rec) // (procs * 4) + 1
        rec["skel"] = [x for part in p.map(_skel_chunk, [rec.name_core.values[i:i + step]
                                                          for i in range(0, len(rec), step)]) for x in part]
        rec["snd"] = [x for part in p.map(_snd_chunk, [rec.name_core.values[i:i + step]
                                                        for i in range(0, len(rec), step)]) for x in part]
    rec = add_name_counts(rec)
    store = RecStore(rec)
    cand = twin_feats(cand, rec)
    print(f"[{a.split}] {len(cand):,} pairs; context done {time.time() - t0:.0f}s", flush=True)

    if a.split == "train":
        truth = pl.read_parquet(f"{PQ}/train_pairs.parquet").with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
        cand = cand.join(truth, on=["s1_id", "cand_id"], how="left").with_columns(
            pl.col("y").fill_null(0).cast(pl.Int8),
            (pl.col("s1_id").str.slice(3).cast(pl.Int64) % 10).cast(pl.Int8).alias("fold"))
        stats = attach_records(cand.filter(pl.col("fold") <= 1).select(["s1_id", "cand_id", "y"]).to_pandas(), store)
        LO_ADD, LO_DROP = build_logodds(stats, procs)
        pd.DataFrame({"token": list(LO_ADD), "lo": list(LO_ADD.values())}).to_parquet(f"{FEAT}/lo_add.parquet")
        pd.DataFrame({"token": list(LO_DROP), "lo": list(LO_DROP.values())}).to_parquet(f"{FEAT}/lo_drop.parquet")
        with Pool(procs) as p:                       # address-word evidence (digits removed), same folds
            n_ = len(stats) // (procs * 4) + 1
            aw = lambda col: [x for part in p.map(_addr_words, [stats[col].values[i:i + n_]
                                                                for i in range(0, len(stats), n_)]) for x in part]
            astats = pd.DataFrame({"s1_name": aw("s1_addr"), "c_name": aw("c_addr"), "y": stats.y.values})
        ALO_ADD, ALO_DROP = build_logodds(astats, procs)
        pd.DataFrame({"token": list(ALO_ADD), "lo": list(ALO_ADD.values())}).to_parquet(f"{FEAT}/alo_add.parquet")
        pd.DataFrame({"token": list(ALO_DROP), "lo": list(ALO_DROP.values())}).to_parquet(f"{FEAT}/alo_drop.parquet")
        del astats
        top = sorted(LO_ADD.items(), key=lambda kv: kv[1])
        print(f"log-odds: add={len(LO_ADD):,} drop={len(LO_DROP):,}; most negative added: "
              f"{[t for t, _ in top[:15]]}; most positive added: {[t for t, _ in top[-15:]]}", flush=True)
        del stats
    else:
        LO_ADD = dict(pd.read_parquet(f"{FEAT}/lo_add.parquet").itertuples(index=False))
        LO_DROP = dict(pd.read_parquet(f"{FEAT}/lo_drop.parquet").itertuples(index=False))
        ALO_ADD = dict(pd.read_parquet(f"{FEAT}/alo_add.parquet").itertuples(index=False))
        ALO_DROP = dict(pd.read_parquet(f"{FEAT}/alo_drop.parquet").itertuples(index=False))

    if a.no_label_lo:
        LO_ADD, LO_DROP = {}, {}
    PLO_ADD, PLO_DROP, S1DF = build_role_tables(
        cand.filter(pl.col("rank_q") == 1).select(["s1_id", "cand_id", "rank_q"]).to_pandas(), rec, procs)

    out_dir = a.out or f"{FEAT}/{a.split}.parquet"
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    elif os.path.exists(out_dir):
        os.remove(out_dir)
    os.makedirs(out_dir)
    writer, n_rows, n_cols = None, 0, 0
    drop_cols = [f"{side}_{k}" for side in ("s1", "c") for k in ("name", "core", "addr", "nums", "skel", "snd")] + ["country"]
    with Pool(procs) as pool:                        # forked once, after all lookup tables are set
        starts = range(0, len(cand), a.chunk)
        for part, i in enumerate(tqdm(starts, desc=f"features {a.split}", unit="chunk", mininterval=5)):
            t1 = time.time()
            c = attach_records(cand.slice(i, a.chunk).to_pandas(), store)
            c = pd.concat([c, string_feats(c), run_loop(c, pool, procs)], axis=1).drop(columns=drop_cols)
            if writer is not None:
                writer.join()
            writer = threading.Thread(target=c.to_parquet, args=(f"{out_dir}/part-{part:04d}.parquet",),
                                      kwargs={"index": False})
            writer.start()
            n_rows, n_cols = n_rows + len(c), c.shape[1]
            print(f"  chunk {part}: {len(c):,} pairs in {time.time() - t1:.0f}s", flush=True)
    if writer is not None:
        writer.join()
    print(f"[{a.split}] features ({n_rows:,}, {n_cols}) written to {out_dir} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
