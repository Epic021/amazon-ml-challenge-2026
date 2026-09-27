# Amazon ML Challenge 2026: Business Entity Resolution

Team repo. Start with **[STRATEGY.md](STRATEGY.md)**: the problem, data findings, validation, pipeline, roles and the 72-hour plan.

## Setup
Place the organizer's `student_resource/` folder (dataset + validator) at the repo root. It is git-ignored and must never be committed.

```
student_resource/dataset/{train,test}/*.tsv
```

## EDA
Open **[eda/report.html](eda/report.html)** in a browser for the visual EDA report: structure, scripts, noise, match structure, hard negatives, test shift and France.

Scripts in `eda/` read from `student_resource/dataset`, or from `$DATA_DIR` if set:
- `eda1.py`: label structure, match counts, basic column stats
- `eda2.py`: source styles, scripts, leakage checks, matched-pair similarity
- `eda3.py`: blocking-recall probe (IDF inverted index), hard-negative examples
- `eda4.py`: test distractor shift, sibling-vs-orphan negatives, number and name-difference patterns, France
- `eda_test.py`: test-set sizes, country mix, train/test overlap

## Fast rebuild + France routing (branch `fast-france`)
On a fresh 56+ vCPU VM, in tmux:
```
git clone <repo> && cd amazon-ml-challenge-2026 && git checkout fast-france
bash scripts/setup_cpu_vm.sh          # uv env, Drive dataset, retrievers, features (train 50%), b5 + b5nolo, decode
source .venv/bin/activate
```
Then (teammate probabilities from their `export_probs.py` in `data/friend/`: `train_oof_probs.parquet`, `test_probs.parquet`):
```
python scripts/mask_sim.py --tag b5 --other b5nolo --feat_dir data/feat_b5                   # gates 1-2
python scripts/route_unknown.py --full b5 --nolo b5nolo --feat_dir data/feat_b5 --out b5r     # gate 3
python scripts/blend.py --tag b5                                                            # reference
python scripts/blend.py --tag b5r --write --out output_final                                # gate 4 + TSVs
python student_resource/utils/validate_submission.py --test-dir student_resource/dataset/test --check-ids \
  -m output_final/matching_results.tsv -c output_final/candidate_pairs.tsv
```
Submit `output_final/` only if gates 1-4 pass (plan: word log-odds unknown on French pairs -> route those pairs to b5nolo).
