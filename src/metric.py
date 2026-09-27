"""Macro F0.5 exactly as the challenge defines it.

Per S1 entity: F0.5 = 1.25*TP / (|pred| + 0.25*|true|); both empty -> 1.0; otherwise TP=0 -> 0.
Averaged over ALL S1 entities in `s1_ids` (singletons included).
"""
import numpy as np
import pandas as pd


def macro_f05(pred: pd.DataFrame, truth: pd.DataFrame, s1_ids) -> float:
    """pred / truth: long DataFrames with columns s1_id, cand_id (one row per predicted / true link)."""
    return per_entity_f05(pred, truth, s1_ids).mean()


def s1_universe(s1_ids: pd.Series, pred_s1: pd.Series) -> pd.Series:
    """The S1 ids a prediction file covers. Train features may be computed for a sample of S1 ids
    (features.py --sample keeps id % 1000 < sample * 1000); S1s outside the sample have no predictions
    and must not be scored as empty sets. Full coverage returns s1_ids unchanged."""
    m = int(pd.Series(pd.unique(np.asarray(pred_s1))).str[3:].astype(np.int64).mod(1000).max()) + 1
    return s1_ids if m >= 1000 else s1_ids[(s1_ids.str[3:].astype(np.int64) % 1000 < m).values]


def per_entity_f05(pred: pd.DataFrame, truth: pd.DataFrame, s1_ids) -> pd.Series:
    idx = pd.Index(pd.unique(np.asarray(s1_ids)), name="s1_id")
    pred = pred[pred.s1_id.isin(idx)].drop_duplicates(["s1_id", "cand_id"])
    truth = truth[truth.s1_id.isin(idx)]
    n_pred = pred.groupby("s1_id").size().reindex(idx, fill_value=0)
    n_true = truth.groupby("s1_id").size().reindex(idx, fill_value=0)
    tp = pred.merge(truth, on=["s1_id", "cand_id"]).groupby("s1_id").size().reindex(idx, fill_value=0)
    denom = n_pred + 0.25 * n_true
    f = np.where(denom > 0, 1.25 * tp / denom.where(denom > 0, 1), 1.0)  # both empty -> 1
    return pd.Series(f, index=idx, name="f05")


if __name__ == "__main__":
    # README worked example: pred 3 (2 correct), truth 2 -> 0.714
    p = pd.DataFrame({"s1_id": ["a"] * 3 + ["b"], "cand_id": ["x", "y", "z", "q"]})
    t = pd.DataFrame({"s1_id": ["a", "a"], "cand_id": ["x", "z"]})
    s = per_entity_f05(p, t, ["a", "b", "c"])
    assert abs(s["a"] - 0.7142857) < 1e-6 and s["b"] == 0 and s["c"] == 1, s
    print("metric self-test ok:", s.round(4).to_dict())
