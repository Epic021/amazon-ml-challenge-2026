# Business Entity Resolution: reproducible pipeline

This folder regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the challenge TSVs, using only the files provided by the organisers. No external data or lookups are used. The only model is LightGBM (MIT).

## Setup

The two pair models pin different library versions, so there are two environments.

```bash
python3 -m venv .venv    && .venv/bin/pip install -r requirements.txt              # our pipeline + stacker (Python 3.10)
python3 -m venv .venv_tm && .venv_tm/bin/pip install -r requirements_teammate.txt  # teammate pipeline (Python 3.12)
```

## Run (one command)

```bash
bash run_all.sh /path/to/student_resource/dataset C
```

- The first argument is the folder that contains `train/` and `test/`.
- The second argument is the teammate model's data-handling variant. `C` drops mixed-label pattern groups; this is the configuration used for the submission.
- Intermediate files go to `work/` (about 120 GB). Override the location with `BER_WORK_ROOT`.
- Reference machine: 64 vCPU and 500 GB RAM. Run time is about 9 h.

## Stages

| # | Step | Code | Output |
|---|---|---|---|
| 1 | Teammate model: normalise, retrieve, features, two-stage LightGBM (cross-fitted), export probabilities (out-of-fold on train) | `src/teammate/{prep,retrieve,features,model,export_probs}.py` | `work/data/friend/{train_oof_probs,test_probs}.parquet` |
| 2a | TSV → parquet; normalisation (ftfy, NFKC, anyascii romanisation, abbreviations, numbers, core name) | `src/ours/tsv_to_parquet.py`, `normalize.py` | `work/data/{parquet,norm}/` |
| 2b | Candidate generation: four per-country TF-IDF retrievers (word, char-3-gram name, address, house-number\|street keys), both directions | `block.py`, `block_numaddr.py` | `work/data/cand/` |
| 2c | Union + rank pruning (S1 top-10 or record top-15); mined token equivalences | `union.py`, `mine_equiv.py` | `work/data/cand/{split}.parquet` |
| 2d | ~95 pair features (house-number relation, word evidence, label-free word roles, sound key, twins, similarities, rank context) | `features.py` | `work/data/feat_b5/` |
| 2e | Stage-1 LightGBM (train folds 2–4, early stop on fold 5) | `train.py` | `work/data/pred/{train,test}_b5.parquet` |
| 3 | Stage-2 stacker: both models' probabilities + competition features + stage-1 features (train folds 6–8) | `stage2.py` | `work/data/pred/*_b5sb2.parquet` |
| 4 | Isotonic calibration (fold 9) → soft one-owner renormalisation → threshold (best on holdout fold 5) | `decode.py` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |

**Folds:** S1 ids mod 10.
- 0–1: word statistics.
- 2–4: stage 1.
- 5: holdout.
- 6–8: stage 2.
- 9: calibration.

**Holdout macro F0.5** (fold 5, 220,810 S1): 0.98825 (India 0.987, US 0.989).
