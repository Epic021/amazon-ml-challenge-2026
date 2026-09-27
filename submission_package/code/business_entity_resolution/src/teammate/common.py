"""Paths, logging and small helpers shared by every stage."""
import os
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("BER_DATA", REPO / "dataset" / "student_resource" / "dataset"))
WORK = Path(os.environ.get("BER_WORK", REPO / "work"))
OUT = Path(os.environ.get("BER_OUT", REPO / "output"))
NPROC = int(os.environ.get("BER_NPROC", os.cpu_count() or 8))
# smoke-test mode: keep only this fraction of S1 businesses (0 = everything)
FRAC = float(os.environ.get("BER_FRAC", "0"))

_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:7.0f}s] {msg}", flush=True)


def id_hash(ids, mod):
    """Deterministic bucket of numeric entity ids (the audit's fold hash)."""
    ids = np.asarray(ids).astype(np.uint64)
    return ((ids * np.uint64(2654435761)) >> np.uint64(16)) % np.uint64(mod)


def split_dir(split):
    d = WORK / split
    d.mkdir(parents=True, exist_ok=True)
    return d
