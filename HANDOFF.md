# HANDOFF: Amazon ML Challenge 2026, Business Entity Resolution (state at Sep 27, ~21:05 IST)

Read this first if you are picking up the work. The detailed history and every experiment are in [STRATEGY.md](STRATEGY.md).

## 1. Where we stand

| Item | Value |
|---|---|
| Deadline | Sep 27, 23:59 IST (leaderboard uploads + final zip) |
| Submissions left | **2** (the user's count at ~19:00) |
| Best public LB | **0.983**: STACK-B5 v1 (`submissions/stacker_v1_LB0983.tsv`) |
| Best holdout | **0.98842**: stacker v3 (`submissions/stacker_v3_holdout098842.tsv`), checker PASS. **Recommended submission #1** |
| Running | stacker v4 (v3 + XGBoost third model), holdout expected ~21:38 IST on the VM |
| Top-100 public cutoff | 0.988672. Only the top 100 get evaluated |
| Final ranking | **Private** leaderboard (the rest of the test set); the public LB is a subset |
| Holdout → LB drop (history) | −0.014 (B1), −0.007 (B2/B3), −0.006 (blend), **−0.005 (stack v1)** |

## 2. The model that is on the LB (and the family we submit)

```
raw TSV → normalize → 4 TF-IDF retrievers (word, namechar, addr, numaddr) → union + prune
       → 95 features → stage 1 LightGBM "B5" (train folds 2-4)                       ┐
teammate pipeline (code/business_entity_resolution, variant C) → OOF + test probs    ├→ stage-2 stacker
[v4 only] XGBoost third model on the same 95 features (train folds 2-3)              ┘   (LightGBM)
       → isotonic calibration (fold 9) → soft one-owner p' = o/(1+Σo) → threshold 0.65 → TSVs
```

- **Folds** (S1 id mod 10): 0–1 word statistics · 2–4 stage-1 training · **5 holdout** · 6–8 stacker training · 9 calibration.
- **Stacker versions:**

  | Version | What | Holdout | LB |
  |---|---|---|---|
  | v1 `b5sb` | B5 + teammate v1 probs, 37 features, trained on folds 6–7 | 0.98792 | **0.983** |
  | v2 `b5sb2` | + teammate **v2** probs, 3 training folds, all 131 features | 0.98825 | not submitted |
  | **v3 `b5sb3`** | v2 + **`--context_folds 0,1,5,6,7,8,9`**: competition features over all out-of-fold train folds (v1/v2 saw only half of each record's competing S1; test sees all) | **0.98842** | – |
  | v4 `b5sb4` | v3 + XGBoost `x5` probabilities (`--third x5`) | ⏳ | – |

## 3. Files (laptop: `C:\Users\namja\Downloads\Amazon_ML_challenge`)

| Path | What |
|---|---|
| `submissions/stacker_v3_holdout098842.tsv` | **best file**: submit #1 |
| `submissions/stacker_v3_france085.tsv` | v3 with French rows at threshold 0.85 (India/US identical). Higher-variance option for #2 |
| `submissions/stacker_v2_holdout098830.tsv`, `stacker_v2_france085.tsv` | older v2 variants |
| `submissions/stacker_v1_LB0983.tsv` | the file that scored 0.983 |
| `submissions/PROBE_stackv1_france075.tsv`, `PROBE_stackv1_france085.tsv` | diagnostic France probes (never submitted) |
| `submissions/probs_v3/{test_probs,train_probs}.parquet` | v3 stacker raw probabilities, p ≥ 0.01 (train folds 6–8 are in-sample, flagged `stacker_train_fold`) |
| `submissions/weights/lgb_b5.txt` | stage-1 LightGBM weights (1,184 trees; the text file *is* the model) |
| `submission_package/` | **final zip contents**: `code/business_entity_resolution/{src/ours, src/teammate, run_all.sh, README.md, requirements*.txt}` + `Documentation_template.md` (methodology, filled) |

The v3 stacker weights were **not saved**: `stage2.py` only started saving models on Sep 27 at 20:50. v4 saves `data/models/stage2_b5sb4.txt`. A v3 retrain was cancelled for lack of time; its probabilities are exported instead.

## 4. VM (GCP n2-highmem-64, 64 vCPU / 503 GB)

- `ssh -i ~/.ssh/id_ed25519 namja@34.47.199.217`. The IP is ephemeral and changes after a stop/start. The VM has a 1 h idle history; check it is running.
- Repo: `/work/amazon-ml-challenge-2026`. venv: `source /work/venv/bin/activate` (py3.10; xgboost 2.1.4 installed Sep 27).
- The repo is private and not cloned on the VM. Sync code with `git archive HEAD src scripts | ssh … "tar -x -C /work/amazon-ml-challenge-2026"`.
- Key data: `data/pred/{train,test}_{b5,b5sb,b5sb2,b5sb3,b5sb4,x5}.parquet`, `data/feat_b5/` (36 GB), `data/friend_v2/` (teammate probs), `data/models/`, `output_*` (decoded TSVs), `logs/*_summary.txt`.
- **Never** `pkill -f`/`pgrep -f` in an ssh one-liner that contains the same name: it kills your own shell. Launch jobs from scp'd `.sh` files with `setsid nohup … &`.
- The teammate VM (`namja@34.57.107.54`, user `sh`, n2-highmem-32) has their B5 re-run and neural-scorer inputs (`data/xenc`). The neural scorer was **never run**.

## 5. How to reproduce / run pieces

- One command, full pipeline: `submission_package/code/business_entity_resolution/run_all.sh <dataset_dir> C` (~9 h; two venvs).
- Stacker only (on the VM, from existing stage-1 + teammate probs):
  `bash scripts/run_stack.sh b5 data/feat_b5 friend_v2 6,7,8 9 <out_tag> --all_feats --context_folds 0,1,5,6,7,8,9 [--third x5]`
  This writes `output_<out_tag>/`, `logs/<out_tag>_summary.txt` (holdout, validator, France proxy), and `data/models/stage2_<out_tag>.txt`.
- One-country re-threshold (diagnostic): `scripts/country_thr.py --tag <tag> --calib_folds 9 --thr 0.65 --country France --country_thr 0.85 --out output_x`.
- Validator: `python student_resource/utils/validate_submission.py --test-dir student_resource/dataset/test --check-ids --matching … --candidate …`.

## 6. What we learned (do not repeat)

- **Where the holdout loss is (0.0117):**
  - ~45% true pairs rejected: 63–74% of them are empty-address records whose name is shared by 2–3 S1. That is undecidable: an exact full-name match with the S1's other records gives only 50%.
  - ~25% true pairs never retrieved (India 1.5%).
  - ~20% wrong merges: same-address neighbours with one word swapped.
- **France** (15% of test, no labels) ≈ 0.95 implied. It over-accepts same-address word swaps: stacker v1's extra caution on French swaps gave France +0.009.
- **Tested and found no gain:**
  - row/ID order signals (chance level);
  - "repeated difference" collective features;
  - view-consensus of full names;
  - dropping retrieval-rank features (France unchanged);
  - French pseudo-label word statistics;
  - per-source thresholds;
  - expected-F DP (+0.0001);
  - a v1+v2 stacker ensemble;
  - a stricter threshold for unseen countries (the LOCO curves show no consistent shift);
  - B5 features at full scale (+0.0002 vs the pilot's +0.001).
- Pilots overstate gains. Keep a change only for ≥ +0.001 on the holdout.

## 7. Next steps (in order)

1. When v4's holdout lands (`logs/b5sb4_summary.txt`): if it is > 0.98842, download `output_b5sb4/matching_results.tsv` (validator already run in the summary) and use it as the best file.
2. Submissions: **#1 = best holdout file** (v3 or v4). **#2** = v4 if it beats #1, else `stacker_v3_france085.tsv` as the higher-variance France bet.
3. Final zip `<team>_submission.zip`:
   - `output/{matching_results.tsv, candidate_pairs.tsv}`: the final file **and its candidate file** from the VM (`output_b5sb3/` or `output_b5sb4/`, 1.7 GB);
   - `code/business_entity_resolution/`: from `submission_package/`. If v4 is final, add the XGBoost step (`python train_xgb.py --tag x5 --feat_dir … --train_folds 2,3` before stage 2, plus `--third x5`) to `run_all.sh`;
   - `Documentation_template.md`: fill in team name and members; update the final holdout/LB numbers.
4. Upload the zip at least 30 min before the deadline.
