#!/usr/bin/env python3
"""Stage 4: write output/matching_results.tsv and output/candidate_pairs.tsv, then validate.

candidate_pairs.tsv holds exactly the pairs the final (stage-2) model scored, i.e. the retrieval
candidates kept by the stage-1 pruning model; every predicted match is one of them.
Per-country sanity statistics (predicted singleton rate, matches per S1, share of records assigned)
are printed so France can be compared with US / India.

Usage: python3 submit.py [VARIANT]
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa

from common import DATA, OUT, log, split_dir


def lists(s1_rows, rec_rows, n_s1, rec_ids):
    order = np.lexsort((rec_rows, s1_rows))
    s1_sorted, rec_sorted = s1_rows[order], rec_rows[order]
    bounds = np.searchsorted(s1_sorted, np.arange(n_s1 + 1))
    ids = rec_ids[rec_sorted]
    return [",".join(ids[bounds[i]:bounds[i + 1]]) for i in range(n_s1)]


def main(variant):
    d = split_dir("test")
    z = np.load(d / f"pred_{variant}{os.environ.get('BER_TAG', '')}.npz")
    c = np.load(d / "cands.npz")
    s1_tab = pa.ipc.open_file(pa.memory_map(str(d / "s1.arrow"))).read_all()
    rec_tab = pa.ipc.open_file(pa.memory_map(str(d / "rec.arrow"))).read_all()
    s1_ids = np.asarray(s1_tab["entity_id"].to_pylist(), dtype=object)
    rec_ids = np.asarray(rec_tab["entity_id"].to_pylist(), dtype=object)
    s1, rec = c["s1"].astype(np.int64), c["rec"].astype(np.int64)
    n_s1 = len(s1_ids)
    pred, prune = z["pred"], z["prune"]
    OUT.mkdir(parents=True, exist_ok=True)
    match = lists(s1[pred], rec[pred], n_s1, rec_ids)
    cand = lists(s1[prune], rec[prune], n_s1, rec_ids)
    pd.DataFrame({"source1_entity_id": s1_ids, "matched_entity_ids": match}).to_csv(
        OUT / "matching_results.tsv", sep="\t", index=False)
    pd.DataFrame({"source1_entity_id": s1_ids, "candidate_entity_ids": cand}).to_csv(
        OUT / "candidate_pairs.tsv", sep="\t", index=False)
    log(f"wrote {OUT / 'matching_results.tsv'} and candidate_pairs.tsv")

    # label-free sanity checks per country
    country = np.asarray(s1_tab["country"].to_pylist())
    rec_country = np.asarray(rec_tab["country"].to_pylist())
    n_match = np.bincount(s1[pred], minlength=n_s1)
    n_cand = np.bincount(s1[prune], minlength=n_s1)
    assigned = np.zeros(len(rec_ids), bool)
    assigned[rec[pred]] = True
    rows = []
    for C in sorted(set(country.tolist())):
        m, mr = country == C, rec_country == C
        rows.append({"country": C, "S1": int(m.sum()), "predicted singletons %": round(100 * (n_match[m] == 0).mean(), 2),
                     "matches per S1": round(n_match[m].mean(), 3), "candidates per S1": round(n_cand[m].mean(), 2),
                     "records assigned %": round(100 * assigned[mr].mean(), 2)})
    print(pd.DataFrame(rows).to_string(index=False))
    here = DATA.parent
    subprocess.run([sys.executable, str(here / "utils" / "validate_submission.py"),
                    "--matching", str(OUT / "matching_results.tsv"),
                    "--candidate", str(OUT / "candidate_pairs.tsv"),
                    "--test-dir", str(DATA / "test")], check=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "A")
