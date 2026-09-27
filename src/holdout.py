"""Holdout evaluation shared by scripts/mask_sim.py and scripts/route_unknown.py: the production decode
(isotonic fitted on folds 6-9, soft exclusivity, one threshold tuned on folds 6-9), scored on fold 5 by country.
"""
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from decode import bootstrap_gain, soft_excl  # noqa: F401  (re-exported for the scripts)
from metric import per_entity_f05, s1_universe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))
TUNE, HOLD = (6, 7, 8, 9), 5
THRS = np.round(np.arange(0.30, 0.91, 0.05), 2)
KEY = ["s1_id", "cand_id"]


def fold_of(ids: pd.Series) -> np.ndarray:
    return (ids.str[3:].astype(np.int64) % 10).values


def iso_fit(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)


class Holdout:
    """Truth, S1 universe (sampled train features aware) and country of the train S1s."""

    def __init__(self, pred_s1: pd.Series):
        self.truth = pd.read_parquet(f"{DATA}/parquet/train_pairs.parquet")
        s1 = pd.read_parquet(f"{DATA}/parquet/train_s1.parquet", columns=["entity_id", "country"])
        s1 = s1[s1.entity_id.isin(set(s1_universe(s1.entity_id, pred_s1)))]
        s1["fold"] = fold_of(s1.entity_id)
        self.tune_ids = s1.entity_id[s1.fold.isin(TUNE)]
        self.hold_ids = s1.entity_id[s1.fold == HOLD]
        self.country = s1.set_index("entity_id").country

    def evaluate(self, df: pd.DataFrame, thr: float = None):
        """df: s1_id, cand_id, p (calibrated) over folds 5-9. Returns (thr, per-S1 fold-5 F0.5).
        thr None -> tuned on folds 6-9."""
        ex = soft_excl(df[KEY + ["p"]])
        ex = ex[ex.p >= THRS.min()]
        if thr is None:
            tu = ex[ex.s1_id.isin(set(self.tune_ids))]
            thr = max(THRS, key=lambda t: per_entity_f05(tu[tu.p >= t], self.truth, self.tune_ids).mean())
        return thr, per_entity_f05(ex[ex.p >= thr], self.truth, self.hold_ids)

    def by_country(self, f: pd.Series) -> str:
        g = f.groupby(self.country.reindex(f.index).values).mean().round(4).to_dict()
        return f"{f.mean():.5f}  {g}"
