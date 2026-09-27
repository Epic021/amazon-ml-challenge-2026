# HANDOFF: Amazon ML Challenge 2026, Business Entity Resolution

**State as of Sep 27, 2026, ~21:10 IST.** This is the complete handoff. The experiment-by-experiment history is also in [STRATEGY.md](STRATEGY.md) (§0 is the final-evening plan, §4–§6 the log).

---

## 0. TL;DR: do this now

1. **Submit `submissions/stacker_v3_holdout098842.tsv`** (holdout 0.98842, validator PASS). It is the best checked file we have.
2. **Wait for stacker v4** (on the VM; holdout ≈ 21:40 IST).
   - Check with `ssh -i ~/.ssh/id_ed25519 namja@34.47.199.217 'cat /work/amazon-ml-challenge-2026/logs/b5sb4_summary.txt'`.
   - If its `BEST` holdout is > 0.98842, download `output_b5sb4/matching_results.tsv` and use it as **submission #2**.
   - Otherwise submission #2 = `submissions/stacker_v3_france085.tsv` (the higher-variance France bet).
3. **Build and upload the final zip** (§11) by **~23:30 IST**. The package is mandatory for every team.

---

## 1. Competition facts that drive every decision

- **Task.** For each Source-1 business (S1), output the S2/S3 records that are the same business (possibly none).
- **Metric.** F0.5 per S1, macro-averaged: `F = 1.25·TP / (|pred| + 0.25·|true|)`.
  - An empty prediction for a singleton scores 1; any prediction for a singleton scores 0.
  - A wrong match costs ~2.5× a missed one (≈0.188 vs ≈0.075 per affected S1 at 3.46 matches).
- **Data.**

  | | S1 | S2 + S3 records |
  |---|---|---|
  | Train | 2.21M (US 1.32M, India 0.88M) | 10.3M |
  | Test | 1.73M (India 46.8%, US 38.3%, **France 15%, no French labels anywhere**) | 9.97M |

  - Train has 3.46 matches per S1; 5.6% of S1 are singletons.
  - Test has 5.75 records per S1 vs 4.68 in train. Our model expects ~3.4 matches per S1, so the extra records are distractors.
- **Leaderboard.**
  - The public LB is a subset of test. **Final ranking = private LB** (the rest of the test).
  - **Only the top 100 of the public LB get evaluated**; the cutoff was **0.988672** at ~20:00 IST Sep 27.
  - Rank 50 ≈ 0.9896; top ≈ 0.9906 (Sep 26).
- **Submissions.** The user reported **2 left** at ~19:00 IST Sep 27. Each upload is `matching_results.tsv` (tab-separated; header `source1_entity_id matched_entity_ids`).
- **Rules.**
  - No external data or lookups.
  - Final model MIT/Apache-2.0 and ≤ 8B parameters. All ours: LightGBM MIT, XGBoost Apache-2.0.
  - The organizers re-run the code, so nothing may be fitted to the test file.
  - The final zip must contain `output/{matching_results.tsv, candidate_pairs.tsv}`, `code/business_entity_resolution/{src/, README.md, requirements.txt}` and `Documentation_template.md`.
- **Validator.** `python student_resource/utils/validate_submission.py --test-dir student_resource/dataset/test --check-ids --matching <m.tsv> --candidate <c.tsv>`.

## 2. Scoreboard (all our LB submissions + key holdouts)

| # | Model | Holdout macro F0.5 (India / US) | Public LB | Holdout → LB |
|---|---|---|---|---|
| 1 | B1: word retriever + 60 features + LightGBM | 0.9662 (0.9546 / 0.9739) | 0.952 | −0.014 |
| 2 | B2: + namechar/addr retrievers + label-free word roles | 0.9772 (0.966 / 0.9847) | 0.970 | −0.007 |
| 3 | B3: + name uniqueness | 0.9777 | 0.970 | −0.008 |
| – | Teammate model alone (v1) | 0.98239 (fold 5) | 0.975 | −0.007 |
| 4 | BLEND-B4: B4 + teammate, fixed 50/50 | 0.98487 (0.9833 / 0.9859) | 0.979 | −0.006 |
| 5 | **STACK v1 (`b5sb`)**: stage-2 stacker, B5 + teammate v1 | 0.98792 (0.9866 / 0.9888) | **0.983** | −0.005 |
| – | STACK v2 (`b5sb2`) | 0.98825 (0.987 / 0.9892) | not submitted | – |
| – | **STACK v3 (`b5sb3`)** | **0.98842 (0.9872 / 0.9892)** | ⏳ submit | – |
| – | STACK v4 (`b5sb4`, + XGBoost) | ⏳ ~21:40 IST | – | – |

- **Implied France** = (LB − 0.468·India − 0.383·US) / 0.15, taking India/US from the holdout:
  - B1 0.88 → B2 0.94 → BLEND 0.94 → **STACK v1 0.950**.
  - The holdout → LB drop is mostly France.
- **Expected LB for v3/v4:** ≈ 0.9835–0.9850.

## 3. Pipeline in detail

```
raw TSV ─ tsv_to_parquet ─ normalize ─┬─ block.py word ───────┐
                                      ├─ block.py namechar ───┤
                                      ├─ block.py addr ───────┼─ union.py (+prune) ─ mine_equiv ─ features.py (95 feats)
                                      └─ block_numaddr.py ────┘                                        │
      stage 1: train.py LightGBM "b5" (train folds 2-4) ──────────────────────────────────────────────┤
      teammate pipeline (code/business_entity_resolution/src, variant C) → friend_v2 OOF + test probs ─┤
      [v4] train_xgb.py XGBoost "x5" (train folds 2-3) ───────────────────────────────────────────────┤
      stage 2: stage2.py stacker (train folds 6-8, context folds 0,1,5-9) ─ decode.py (isotonic fold 9 → soft excl → thr 0.65)
```

### 3.1 Normalisation (`src/normalize.py`)
- ftfy + NFKC, then anyascii (Indian scripts → Latin).
- Abbreviation and state maps (US/India hand maps, still in place: the planned "G1" rewrite never happened).
- Ordinals, number extraction (`nums`).
- `name_core` = name minus legal forms (incl. French SARL/SAS/SASU/EURL/SA/SCI) and honorifics/"the"/"and".
- Flags `nonlatin` and `addr_empty`.

### 3.2 Candidate generation
- **Library:** TF-IDF + `sparse_dot_topn`, per country, top-k in both directions.
- **Retrievers** (`block.py`, `block_numaddr.py`):

  | Retriever | Input | Settings |
  |---|---|---|
  | word | name words + consonant skeleton + address words | `(10, 40, 30000)` |
  | namechar | char 3-grams of `name_core`, record → S1 | `(5, 0, 50000)` |
  | addr | address words | `(5, 15, 30000)` |
  | numaddr | `number\|token` keys | top-50 by key cosine, re-ranked by 0.5·key + 0.5·name similarity, top-5 |

- **`union.py`:**
  - keeps each retriever's score/rank_q/rank_s as features, plus `score` = max, `rank_q`/`rank_s` = min and `n_retrievers`;
  - pruning keeps rank_q ≤ 10 or rank_s ≤ 15.
- **Recall:**
  - own candidates: pair recall 0.9769, oracle F0.5 0.9917;
  - **with the teammate's candidates in the union: recall 0.9899, oracle ceiling 0.9968.**
- **Test:** ≈ 128M scored pairs (≈ 74 per S1).

### 3.3 Features (`src/features.py`, 95 used by B5; `numx_*` dropped)
- **Retrieval/rank context:**
  - per-retriever scores and ranks;
  - s1_best, rank_in_s1, n_s1_cands, c_best, n_c_s1, gap_s1, gap_c, c_margin, is_c_best, mutual_best.
- **House numbers:** num_rel (0 none, 1 equal, 2 cand adds, 3 cand drops, 4 partial, 5 all differ), num_first_eq, num_s1first_in, num_common, num_s1_only, num_c_only, num_min_lev, num_prefix, num_first_logdiff.
- **Name/word evidence:**
  - add/drop label log-odds (`add_lo_*`, `drop_lo_*`; learned on folds 0–1);
  - n_added, n_dropped, *_unknown;
  - legal_conflict, legal_added;
  - core_eq, core_s1_in_c, core_c_in_s1, name_diff_explained;
  - **label-free per-country word roles** (`padd_*`, `pdrop_*`, add_gen_max, add_novel);
  - name uniqueness (s1_core_n_s1, c_core_n_s1, c_core_n_q).
- **Address:** addr_tok_jacc, addr_c_only, addr_s1_only, addr_diff_explained, address-word log-odds (`aadd_lo_*`, `adrop_lo_*`, *_unknown).
- **Similarities** (rapidfuzz on name/core/skeleton/sound key/address):
  - ratio, token_set, token_sort, partial, Jaro-Winkler;
  - length ratios.
- **Other:**
  - twins (twin_num, twin_name: other-source candidates of the same S1 with the same number set / core);
  - flags c_src_s3, c_nonlatin, c_addr_empty.

### 3.4 Stage 1 (`src/train.py`)
- LightGBM binary: lr 0.05, 255 leaves, min_data_in_leaf 200, feature_fraction 0.8, bagging 0.8, λ2 = 1.
- Trained on folds 2–4 (42.7M pairs); early stopping on fold 5; 1,184 trees.
- Alone: holdout 0.98288.
- Weights: `data/models/lgb_b5.txt` (also in `submissions/weights/`).

### 3.5 Teammate model (`code/business_entity_resolution/`, their own pipeline)
- Steps: `prep.py → retrieve.py → features.py → model.py train/test <VARIANT> → submit.py`, then `export_probs.py NAME OUTDIR`.
- It writes `train_oof_probs.parquet` (s1_id, cand_id, p, label; out-of-fold; p ≥ 0.01) and `test_probs.parquet`.
- Variants: A keep all · B drop pattern minority · **C drop mixed-label pattern groups (>5% disagree)**, which is what the user described · D–G other label cleaning.
- **The exact variant behind `friend_v2` should be confirmed with the teammate.**
- **v1 vs v2 files:** v2 = refreshed model, corr 0.99 with v1, 1.5% France flips. Blend with B4: 0.98519 (v2) vs 0.98487 (v1).
- The teammate pins numpy 2.4.6 / pandas 3.0.3, hence a separate venv.

### 3.6 XGBoost third model (`src/train_xgb.py`, tag `x5`)
- hist, max_depth 10, eta 0.1, subsample 0.8, colsample_bytree 0.6, colsample_bylevel 0.8, min_child_weight 20, λ = 2, max_bin 256.
- Trained on folds 2–3 (28.5M pairs), early stopping on fold 5; best iteration 890, holdout logloss 0.00326.
- **Alone: holdout 0.98249** (India 0.977, US 0.9862).
- Weights: `data/models/xgb_x5.json`.

### 3.7 Stage-2 stacker (`src/stage2.py`)
- **Inputs:**
  - p (ours), p_t (teammate), p_m = mean, p_diff, o_missing/t_missing;
  - [v4] p_x (XGBoost).
- **Competition features on each of p, p_t, p_m (prefixes none / `t_` / `m_`), and `x_` for v4:**
  - r_best, r_sum, r_n05 (record side);
  - r_other_best, r_margin, p_share_r;
  - s_rank, s_best, s_sum, s_n05, s_gap, p_share_s (S1 side);
  - twin_p_num, twin_p_name.
- **Plus stage-1 features:** 37 in v1 (BASE + PAIR), all 131 in v2+ (`--all_feats`).
- **LightGBM:** lr 0.05, 127 leaves, min_data 200, ff 0.9, bagging 0.8, λ2 = 1, seed 7.
- **Folds:** trained on folds 6–8 (v2+), early stop on fold 5.
- **`--context_folds 0,1,5,6,7,8,9` (v3+).** Competition features are computed over every out-of-fold train fold.
  - v1/v2 used folds 5–9 only, so each record showed half its competing S1. Test shows all.
  - The fix raised "margin to the record's best other S1" from 3% to 24% of the stacker's gain.
- Pair universe = union of our candidates and the teammate's.
- The model is saved to `data/models/stage2_<tag>.txt` since Sep 27 20:50, so **v3's stacker weights do not exist** and v4's will.
- Top v3 features: p_m 62%, m_r_margin 24%, p_t 12%.

### 3.8 Decode (`src/decode.py`)
- Isotonic calibration on fold 9 (v2+; v1 used 8–9).
- Soft one-owner renormalisation: `p' = o/(1+Σo)` over each record's S1 candidates, with `o` = odds. This beat hard exclusivity by +0.0008.
- Threshold grid on fold 5 → **0.65**.
- The exact expected-F DP (`efdp.py`, `--dp`) is only +0.0001 on the stacker.
- `decode.py` also writes `candidate_pairs.tsv` = every scored pair.

## 4. Validation protocol
- S1 folds = `int(id[3:]) % 10`: 0–1 word statistics · 2–4 stage 1 · **5 holdout** · 6–8 stacker · 9 calibration.
- Blocking always runs on the full record universe.
- Keep a change only if the holdout gains ≥ +0.001 (strict).
- Pilots (10% of S1) overstate gains. The feature-pilot bar was raised to +0.002 after B5 (pilot +0.001 → full +0.0002).
- France has no labels:
  - proxy tools are `scripts/france_gap.py`, `france_diag.py`, `france_shap.py`, `loco_pilot.py` (leave-one-country-out);
  - otherwise use implied France from the LB.

## 5. Where the remaining holdout loss is (STACK v1 error analysis, loss 0.0121)

| Share | What | Detail |
|---|---|---|
| ~45% | True pairs rejected (1.9% of true pairs) | **63–74% have an empty address**. Records with empty addresses belong to *some* S1 97.7% of the time. When the core name is shared by 2–3 S1 the owner is a calibrated coin flip (P = 0.41, model 0.40). Row/ID order, "source quota" and full-name agreement with the S1's other records (P = 0.50 even on exact match) do **not** break the tie. The stacker picks the right owner 72% of the time (chance 46–50%) |
| ~25% | True pairs never retrieved | India 1.5%, US 0.7%; mostly Indian-script / renamed |
| ~20% | Wrong merges (0.31% of predictions) | Same address + same house number, one generic word swapped |
| ~3% | Singletons given a match | – |

**France:**
- 4.3 same-address word-swap candidate pairs per S1, against 0.95 for the US.
- The model believes France ≈ 0.967, but the LB implies 0.95.
- Stacker v1 dropped 0.135 French predictions per S1, mostly swaps, and France rose +0.009, so the removed ones were mostly wrong.
- v3 still predicts 0.63 French swaps per S1 at ~0.94 confidence.

## 6. Everything we tried (verdicts)

**Retrieval:**
- ✓ word, namechar, addr;
- ✓ numaddr (ceiling 0.9867 → 0.9917);
- ✗ model2vec embedding retriever (India ceiling +0.002);
- ✗ uroman.

**Features:**
- ✓ house numbers, similarities, label log-odds (+0.001), label-free roles (France 0.88 → 0.94), name uniqueness (+0.0005);
- ✓ sound key / address log-odds / twins (B5, +0.0002 at full scale);
- ✗ finer number relations (0);
- ✗ evidence dropout (−0.0004);
- ✗ more stage-1 training data (flat curve).

**Ensembling:**
- ✓ fixed blend with teammate (+0.0029);
- ✓ **stacker (+0.0031, LB 0.983)**;
- ✓ v2 (+0.0003);
- ✓ v3 context fix (+0.0002 under a harder evaluation);
- ✗ v1+v2 average (0);
- ⏳ v4 + XGBoost.

**Decoding:**
- ✓ soft exclusivity;
- ✗ expected-F DP (+0.0001);
- ✗ per-source (S2/S3) thresholds (0);
- ✗ stricter threshold for unseen countries (LOCO curves: no consistent shift).

**France:**
- ✗ dropping retrieval-rank features (France swap p 0.335 → 0.331; −0.001 holdout);
- ✗ pseudo-label word log-odds (word statistics barely matter even on swaps: AUC 0.9998 vs 0.9997);
- ✗ "repeated difference" collective features (only 0.05 French predictions per S1 in that bucket; holdout precision 0.99+);
- ✗ consensus / view-consensus (already known to the models).

**Leakage:** none. Row/ID correlation 0.0001, record adjacency 0%, and the ID/row tie-break is at chance.

## 7. Files

### Laptop (`C:\Users\namja\Downloads\Amazon_ML_challenge`, git repo `Epic021/amazon-ml-challenge-2026`, private, branch `main`)

| Path | Content |
|---|---|
| `src/` | our pipeline: normalize, block, block_numaddr, union, mine_equiv, features, train, train_xgb, stage2, decode, efdp, metric |
| `scripts/` | tsv_to_parquet, run_stack.sh, blend, country_thr, france_*, loco_pilot, diff_support*, consensus_test, view_consensus, make_diffsup/make_views, order_signal, stack_ensemble, src_thr, smoke.sh |
| `code/business_entity_resolution/` | teammate pipeline (as in the repo) |
| `submission_package/` | **final zip contents** (see §11): `code/business_entity_resolution/{src/ours, src/teammate, run_all.sh, README.md, requirements.txt, requirements_teammate.txt}`, `Documentation_template.md` |
| `submissions/` (gitignored) | all candidate TSVs, `probs_v3/`, `weights/lgb_b5.txt`, `vm_artifacts.tgz` |
| `offrepo_handoff_20260927.zip` (gitignored, 525 MB) | a copy of `submissions/` + this file |
| `STRATEGY.md`, `HANDOFF.md`, `eda/report.html`, `research_sota.md`, `bottle_necks.md`, `docs/` | docs |

### VM (`namja@34.47.199.217`, `/work/amazon-ml-challenge-2026`)
- `data/parquet` (1.7G), `data/norm` (1.8G), `data/cand` (14G), `data/feat_b5` (36G, part files), `data/pred` (40G).
- `data/friend`, `data/friend_v2`, `data/models` (lgb_b1..b5, xgb_x5, stage2_b5sb4 once written), `data/extra`.
- `output_<tag>/` for each decoded run: `output_b5sb3` = v3; `output_v3_fr_085` = v3 France-0.85; `output_b5sb4` = v4 when done.
- `logs/` (every run; `<tag>_summary.txt` has holdout + validator + France proxy).
- `probs_v3/` (exported v3 probabilities).
- Launcher scripts live in `/work/*.sh`.

### Teammate VM
- `namja@34.57.107.54`, n2-highmem-32; files owned by user `sh` under `/data/amazon-ml-challenge-2026`.
- It holds their re-run of our B5, plus inputs for a GPU neural scorer (`data/xenc`). **The neural scorer was never run.**
- Our read-only analysis venv is in `/tmp/ana`.

## 8. Operating the VM (hard-won lessons)
- **Connect:** `ssh -i ~/.ssh/id_ed25519 namja@34.47.199.217`. The external IP is ephemeral: after a stop/start, get the new one from the GCP console. The public key is `~/.ssh/id_ed25519.pub`.
- **`/tmp` is wiped on restart.** The smoke data (`/tmp/ber_smoke`) was rebuilt Sep 27 with `bash scripts/smoke.sh` in `/work/smoke-repo`.
- **Code sync:** the repo is private, so there's no git on the VM.
  - `git archive HEAD src scripts | ssh … "tar -x -C /work/amazon-ml-challenge-2026"`
  - Use one ssh per tar: two tars from one stdin fail.
- **Launch long jobs** from a scp'd `.sh`: `setsid nohup bash /work/x.sh > /dev/null 2>&1 < /dev/null & disown`. Always log to `logs/*.log`.
- **Never `pkill -f X` / `kill $(pgrep -f X)` in an ssh one-liner that also contains X.** It matched and killed our own shell several times. Find the PID in one call and kill it in another.
- **LightGBM with 64 threads collapses when another CPU job runs** (seen with v3 + XGBoost: load average 124).
  - Pause the other job with `kill -STOP <pid>`, resume with `kill -CONT <pid>`.
  - Or give the second job fewer threads.
- **Env:** `source /work/venv/bin/activate` (Python 3.10, polars 1.44.2, lightgbm 4.6.0, xgboost 2.1.4, rapidfuzz 3.14.3). Unset `BER_DATA`/`BER_OUT` for full runs; smoke runs set them to `/tmp/ber_smoke/...`.

## 9. How to check the running job (v4)
- **Training:** `ssh … 'grep -E "^\[|Early|best" /work/amazon-ml-challenge-2026/logs/b5sb4_stage2.log | tail -3'`. Round 200 logloss was 0.0031492 vs v3's 0.0031677 at round 200; v3's best was 0.0031424.
- **Result:** `ssh … 'cat /work/amazon-ml-challenge-2026/logs/b5sb4_summary.txt'`. It contains the BEST holdout line, per-country scores, test predictions per S1, validator PASS/FAIL, the France proxy and `B5SB4 DONE`.
- **Files:** `output_b5sb4/{matching_results.tsv, candidate_pairs.tsv}`, `data/models/stage2_b5sb4.txt`.

## 10. Reproduce
- **Everything:** `bash submission_package/code/business_entity_resolution/run_all.sh /path/to/student_resource/dataset C`. Needs two venvs; ~9 h on 64 vCPU / 500 GB.
- **Stacker from existing stage-1 + teammate probabilities** (repo root on the VM):
  `bash scripts/run_stack.sh b5 data/feat_b5 friend_v2 6,7,8 9 <tag> --all_feats --context_folds 0,1,5,6,7,8,9 [--third x5]` (~45–60 min).
- **XGBoost:** `python src/train_xgb.py --tag x5 --feat_dir data/feat_b5 --train_folds 2,3` (~45 min with contention).
- **Approximate stage timings (full data):**

  | Stage | Time |
  |---|---|
  | normalize | 3 min |
  | retrievers | 15–25 min each |
  | union | 5 min |
  | features train / test | 85 / 60 min |
  | stage 1 | 80 min |
  | stacker | 45–60 min |
  | decode + validator | 20 min |

## 11. Final zip: exact procedure (do on the VM, then download)

```bash
# on the VM (choose FINAL = b5sb3 or b5sb4)
FINAL=b5sb3
cd /work && rm -rf final && mkdir -p final/output
cp amazon-ml-challenge-2026/output_$FINAL/matching_results.tsv amazon-ml-challenge-2026/output_$FINAL/candidate_pairs.tsv final/output/
# from the laptop: scp -r submission_package/code submission_package/Documentation_template.md namja@34.47.199.217:/work/final/
cd /work/final && zip -r ../TEAMNAME_submission.zip output code Documentation_template.md
# laptop: scp -i ~/.ssh/id_ed25519 namja@34.47.199.217:/work/TEAMNAME_submission.zip .
```

Before zipping:
1. Fill in team name, members and final numbers in `Documentation_template.md`.
2. If v4 is final: add the XGBoost step to `run_all.sh` (`python train_xgb.py --tag x5 --feat_dir "$BER_DATA/feat_b5" --train_folds 2,3` before stage 2, and `--third x5 --out_tag b5sb4` on stage2/decode), and set the README holdout.
3. Confirm the teammate variant letter in `run_all.sh` (default `C`).
4. `output/matching_results.tsv` must be **the same file uploaded to the leaderboard**.

## 12. Open risks / known issues
- v3's stacker weights were not saved; its probabilities are exported (`submissions/probs_v3/`). Retraining reproduces it up to parallel row-order nondeterminism.
- The package's teammate variant letter and their exact run are unverified.
- The stacker depends on the teammate model, so reproduction requires both pipelines (two environments).
- The top-100 cutoff (0.988672) is above our expected LB (~0.984–0.985). Evaluation may not reach us, but the package is still required.

## 13. People / roles
- **P1:** validation, decode, experiment log, submission sign-off.
- **P2:** VM, retrievers, full runs, final zip.
- **P3:** features, stacker.
- **P4:** normalisation, methodology doc, teammate model/probabilities.
- The teammate model's author owns `code/business_entity_resolution/` and the `friend*` probability files.
