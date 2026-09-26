#!/usr/bin/env python3
"""Post-hoc steps on saved probabilities (no model training).

  redecode NAME            re-pick the decoder on NAME's out-of-fold probabilities with the wider
                           grid, then decode NAME's test probabilities with it
                           -> test/pred_<NAME>_dec.npz  (+ models/<NAME>/redecode.json)
  ensemble OUT M1,M2,...   average the members' probabilities (out-of-fold on train, and on test),
                           pick the decoder on the out-of-fold average, decode the test average
                           -> test/pred_<OUT>.npz  (+ models/<OUT>/report.json)

NAME / members are model folders such as C_fast or C_tune2 (their test predictions must exist as
test/pred_<NAME>.npz). Write the TSVs afterwards with:  BER_TAG=<suffix> python3 submit.py <variant>
"""
import json
import sys

import numpy as np
import pyarrow as pa

from common import WORK, id_hash, log, split_dir
from decode import decode
from model import evaluate


def train_meta():
    d = split_dir("train")
    c = np.load(d / "cands.npz")
    lab = np.load(d / "labels.npz")
    s1_tab = pa.ipc.open_file(pa.memory_map(str(d / "s1.arrow"))).read_all()
    s1, rec = c["s1"].astype(np.int64), c["rec"].astype(np.int64)
    return dict(s1=s1, rec=rec, y=(lab["true_row"][rec] == s1).astype(np.int8), true_size=lab["true_size"],
                n_s1=s1_tab.num_rows, s1_id=s1_tab["id"].to_numpy(),
                s1_country=s1_tab["country"].to_numpy(zero_copy_only=False))


def test_meta():
    d = split_dir("test")
    c = np.load(d / "cands.npz")
    n_s1 = pa.ipc.open_file(pa.memory_map(str(d / "s1.arrow"))).read_all().num_rows
    return d, c["s1"].astype(np.int64), c["rec"].astype(np.int64), n_s1


def pick_and_decode(p_oof, p_test, prune_test, out_name, extra):
    meta = train_meta()
    res, best = evaluate(meta, p_oof, out_name)
    d, s1, rec, n_s1 = test_meta()
    pred = decode(s1, rec, p_test, n_s1, method=best[1], **best[2])
    np.savez(d / f"pred_{out_name}.npz", p2=p_test, prune=prune_test, pred=pred)
    mdir = WORK / "models" / out_name
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "report.json").write_text(json.dumps({**extra, "decoded": res,
                                                  "decoder": {"method": best[1], "params": best[2]}},
                                                 indent=2, default=str))
    log(f"{out_name}: out-of-fold macro F0.5 {best[0]:.5f} ({best[1]} {best[2]}); "
        f"test: {int(pred.sum()):,} predicted matches")


def redecode(name):
    d = split_dir("test")
    z = np.load(d / f"pred_{name}.npz")
    p_oof = np.load(WORK / "models" / name / "oof_p2.npy")
    pick_and_decode(p_oof, z["p2"], z["prune"], f"{name}_dec", {"source": name})


def ensemble(out, members):
    d = split_dir("test")
    p_oof = np.mean([np.load(WORK / "models" / m / "oof_p2.npy") for m in members], axis=0)
    zs = [np.load(d / f"pred_{m}.npz") for m in members]
    p_test = np.mean([z["p2"] for z in zs], axis=0)
    prune = np.any([z["prune"] for z in zs], axis=0)
    pick_and_decode(p_oof, p_test, prune, out, {"members": members, "weights": "equal"})


if __name__ == "__main__":
    if sys.argv[1] == "redecode":
        redecode(sys.argv[2])
    else:
        ensemble(sys.argv[2], sys.argv[3].split(","))
