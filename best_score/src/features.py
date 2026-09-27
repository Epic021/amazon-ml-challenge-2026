#!/usr/bin/env python3
"""Stage 2: pair features for every candidate pair (parallel over slices of the candidate list).

Features follow the label audit's findings (see docs/METHODOLOGY.md):
  * how the name changed: 16 classes incl. cross-script ones compared by sound; counts of added,
    dropped and typo words; legal-form change; similarity scores (raw, normalised, core, sound-alike)
  * how the address changed: 10 classes; word counts; similarities
  * how the house / plot numbers changed: 8 classes, log gap and edit distance of the closest changed
    pair, numbers added / dropped, whether the changed one is the first (house) number
  * context that needs no labels: how common the name and address are among S1 businesses, empty
    address, script, source, retrieval score and ranks
The added / dropped words are kept as text for the label-based word evidence computed in model.py.

  WORK/<split>/feats.parquet  (same row order as cands.npz)
Usage: python3 features.py train|test
"""
import sys
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from common import NPROC, log, split_dir
from normalize import addr_compare, akey, content, cross_content, metaphone, name_compare, nysiis, soundex

REC_COLS = ["ntoks", "nind", "awords", "anums", "aempty", "nrom", "arom", "name_freq", "addr_freq"]
NUM = ["nd", "ad", "rel", "src", "name_tsr", "name_ratio", "name_partial", "name_sort", "core_ratio", "core_jw",
       "name_jac", "sk_jac", "ph_sdx_jac", "ph_mph_jac", "ph_nys_jac", "addr_ph_jac", "n_name_add", "n_name_drop",
       "n_name_typo", "name_minor", "name_same_form", "n_addr_abbr", "name_len_s1",
       "name_len_rec", "nind_rec", "addr_tsr", "addr_ratio", "addr_jac", "n_addr_add", "n_addr_drop",
       "n_addr_typo", "n_aw_s1", "n_aw_rec", "aempty_rec", "aempty_s1", "n_num_s1", "n_num_rec", "num_logdiff",
       "num_lev", "n_num_add", "n_num_drop", "num_first", "name_freq_s1", "name_freq_rec", "addr_freq_s1",
       "addr_freq_rec"]
WORDS = ["name_added", "name_dropped", "addr_added", "addr_dropped"]
_S1 = _REC = None


def _init(dirpath):
    global _S1, _REC
    _S1 = pa.ipc.open_file(pa.memory_map(f"{dirpath}/s1.arrow")).read_all().select(REC_COLS)
    _REC = pa.ipc.open_file(pa.memory_map(f"{dirpath}/rec.arrow")).read_all().select(REC_COLS + ["src"])


def _rows(tab, idx, cols):
    u, inv = np.unique(idx, return_inverse=True)
    t = tab.take(pa.array(u))
    return {c: t[c].to_pylist() for c in cols}, inv


def _parse(r, i):
    nt = tuple(r["ntoks"][i].split())
    af = None if r["aempty"][i] else (frozenset(r["awords"][i].split()), tuple(r["anums"][i].split()))
    cc = cross_content(nt)
    ph = ({akey(t) for t in cc}, {soundex(t) for t in cc}, {metaphone(t) for t in cc}, {nysiis(t) for t in cc},
          {metaphone(w) for w in af[0] if len(w) > 2} if af else set())
    return nt, bool(r["nind"][i]), af, ph


def _jac(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def _chunk(args):
    s1_idx, rec_idx = args
    S, si = _rows(_S1, s1_idx, REC_COLS)
    R, ri = _rows(_REC, rec_idx, REC_COLS + ["src"])
    out = {c: [] for c in NUM + WORDS}
    cache_s, cache_r = {}, {}
    for a, b in zip(si.tolist(), ri.tolist()):
        ps = cache_s.get(a)
        if ps is None:
            ps = cache_s[a] = _parse(S, a)
        pr = cache_r.get(b)
        if pr is None:
            pr = cache_r[b] = _parse(R, b)
        snt, sind, saf, sph = ps
        rnt, rind, raf, rph = pr
        nd, nadd, ndrop, ntypo, nminor, nadded, ndropped, njac, same_form = name_compare(snt, sind, rnt, rind)
        (ad, rel, aadd, adrop, atypo, aadded, adropped, ajac, logdiff, numlev, nnadd, nndrop,
         first, n_abbr) = addr_compare(saf, raf)
        sn, rn = S["nrom"][a], R["nrom"][b]
        sc, rc = " ".join(sorted(content(snt))), " ".join(sorted(content(rnt)))
        both_addr = saf is not None and raf is not None
        sa, ra = S["arom"][a], R["arom"][b]
        vals = (nd, ad, rel, R["src"][b],
                fuzz.token_set_ratio(sn, rn), fuzz.ratio(sn, rn), fuzz.partial_ratio(sn, rn),
                fuzz.ratio(" ".join(snt), " ".join(rnt)), fuzz.ratio(sc, rc), JaroWinkler.similarity(sc, rc),
                njac, _jac(sph[0], rph[0]), _jac(sph[1], rph[1]), _jac(sph[2], rph[2]), _jac(sph[3], rph[3]),
                _jac(sph[4], rph[4]) if saf is not None and raf is not None else np.nan,
                min(nadd, 50), min(ndrop, 50), min(ntypo, 50), nminor, same_form, min(n_abbr, 50),
                min(len(snt), 50), min(len(rnt), 50),
                int(rind),
                fuzz.token_set_ratio(sa, ra) if both_addr else np.nan,
                fuzz.ratio(sa, ra) if both_addr else np.nan,
                ajac, min(aadd, 50), min(adrop, 50), min(atypo, 50),
                len(saf[0]) if saf else 0, len(raf[0]) if raf else 0, int(raf is None), int(saf is None),
                min(len(saf[1]), 50) if saf else 0, min(len(raf[1]), 50) if raf else 0,
                logdiff, numlev, min(nnadd, 50), min(nndrop, 50), first,
                S["name_freq"][a], R["name_freq"][b], S["addr_freq"][a], R["addr_freq"][b])
        for c, v in zip(NUM, vals):
            out[c].append(v)
        out["name_added"].append(" ".join(nadded))
        out["name_dropped"].append(" ".join(ndropped))
        out["addr_added"].append(" ".join(aadded))
        out["addr_dropped"].append(" ".join(adropped))
    cols = {c: np.asarray(out[c], dtype=np.float32) for c in NUM}
    cols.update({c: pa.array(out[c], pa.string()) for c in WORDS})
    return pa.table(cols)


def main(split):
    d = split_dir(split)
    c = np.load(d / "cands.npz")
    s1, rec = c["s1"], c["rec"]
    n = len(s1)
    step = 50_000
    tasks = ((s1[i:i + step], rec[i:i + step]) for i in range(0, n, step))
    writer = None
    done = 0
    with Pool(NPROC, initializer=_init, initargs=(str(d),)) as pool:
        for t in pool.imap(_chunk, tasks, chunksize=1):
            if writer is None:
                writer = pq.ParquetWriter(d / "feats.parquet", t.schema, compression="zstd")
            writer.write_table(t)
            done += t.num_rows
            if done // step % 100 == 0:
                log(f"{done:,} / {n:,} pairs")
    writer.close()
    log(f"{split}: features for {n:,} pairs written")


if __name__ == "__main__":
    main(sys.argv[1])
