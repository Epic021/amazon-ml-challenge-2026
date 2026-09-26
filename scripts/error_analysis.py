"""Where does the holdout macro F0.5 go? Re-applies the chosen decode rule to a model's train predictions
and attributes the loss on the holdout (fold 5) to: blocking misses, rejected true pairs, accepted false pairs.

  python scripts/error_analysis.py --tag b2 --excl soft --thr 0.55
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from decode import hard_excl, soft_excl, CALIB_FOLDS, HOLD_FOLD  # noqa: E402
from metric import per_entity_f05  # noqa: E402

DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "data"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b2")
    ap.add_argument("--excl", default="soft")
    ap.add_argument("--thr", type=float, default=0.55)
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()

    tr = pd.read_parquet(f"{DATA}/pred/train_{a.tag}.parquet")
    cal = tr[tr.fold.isin(CALIB_FOLDS)]
    tr["p"] = IsotonicRegression(out_of_bounds="clip").fit(cal.p.values, cal.y.values).predict(tr.p.values)
    tr = (soft_excl if a.excl == "soft" else hard_excl)(tr)
    truth = pd.read_parquet(f"{DATA}/parquet/train_pairs.parquet")
    s1 = pd.read_parquet(f"{DATA}/parquet/train_s1.parquet", columns=["entity_id", "country"])
    ctry = s1.set_index("entity_id").country
    hold = s1.entity_id[(s1.entity_id.str[3:].astype(np.int64) % 10) == HOLD_FOLD]
    hs = set(hold)
    ho = tr[tr.s1_id.isin(hs)]
    pred = ho[ho.p >= a.thr][["s1_id", "cand_id", "p"]]
    th = truth[truth.s1_id.isin(hs)]
    f = per_entity_f05(pred, truth, hold)
    print(f"holdout macro F0.5 = {f.mean():.5f}; loss = {1 - f.mean():.5f}")

    # classify every true pair and every predicted pair
    tp = th.merge(pred, on=["s1_id", "cand_id"], how="left", indicator=True)
    in_cand = th.merge(ho[["s1_id", "cand_id", "p"]], on=["s1_id", "cand_id"], how="left")
    fn_block = in_cand.p.isna()
    fn_model = in_cand.p.notna() & (in_cand.p < a.thr)
    fp = pred.merge(th, on=["s1_id", "cand_id"], how="left", indicator=True)
    fp = fp[fp._merge == "left_only"]
    th = th.assign(country=th.s1_id.map(ctry).values)
    print("\ntrue pairs: missed by BLOCKING", round(fn_block.mean(), 4), "| rejected by MODEL", round(fn_model.mean(), 4))
    print("  by country (blocking / model):",
          {c: (round(fn_block.values[(th.country == c).values].mean(), 4),
               round(fn_model.values[(th.country == c).values].mean(), 4)) for c in th.country.unique()})
    print("predicted pairs that are WRONG:", round(len(fp) / max(len(pred), 1), 4),
          " by country:", fp.s1_id.map(ctry).value_counts().div(pred.s1_id.map(ctry).value_counts()).round(4).to_dict())
    n_true = th.groupby("s1_id").size().reindex(f.index, fill_value=0)
    print("singleton S1 with any prediction (score 0):", round((f[n_true.values == 0] < 1).mean(), 4),
          f"of {int((n_true == 0).sum()):,} singletons")

    # context for examples
    feat = pd.read_parquet(f"{DATA}/feat/train.parquet", columns=["s1_id", "cand_id", "num_rel", "addr_tset", "core_tset", "n_added"])
    rel = {0: "none", 1: "equal", 2: "c_adds", 3: "c_drops", 4: "partial", 5: "differ"}
    fpx = fp.merge(feat, on=["s1_id", "cand_id"], how="left")
    print("\nFALSE POSITIVES by house-number relation:", fpx.num_rel.map(rel).value_counts(normalize=True).round(3).to_dict())
    print("  with near-identical address (>=90):", round((fpx.addr_tset >= 90).mean(), 3),
          " with an added name word:", round((fpx.n_added > 0).mean(), 3))
    fnm = in_cand[fn_model].merge(feat, on=["s1_id", "cand_id"], how="left")
    print("MODEL-REJECTED true pairs by house-number relation:", fnm.num_rel.map(rel).value_counts(normalize=True).round(3).to_dict())
    print("  p quantiles:", np.percentile(fnm.p.fillna(0), [25, 50, 75, 90]).round(3))

    raw = pd.concat([pd.read_parquet(f"{DATA}/parquet/train_s{k}.parquet") for k in (1, 2, 3)]).set_index("entity_id")

    def show(title, df):
        print(f"\n## {title}")
        for _, r in df.sample(min(a.n, len(df)), random_state=1).iterrows():
            print(f"  [{ctry.get(r.s1_id)} p={r.p:.2f}] S1: {raw.loc[r.s1_id, 'business_name']} | {raw.loc[r.s1_id, 'business_address']}")
            print(f"  {'':18s}  C : {raw.loc[r.cand_id, 'business_name']} | {raw.loc[r.cand_id, 'business_address']}")
    show("FALSE POSITIVES (accepted, wrong)", fpx)
    show("MODEL-REJECTED true pairs", fnm)
    bl = in_cand[fn_block].assign(p=np.nan)
    show("BLOCKING misses (never became candidates)", bl[bl.s1_id.map(ctry) == "India"])


if __name__ == "__main__":
    main()
