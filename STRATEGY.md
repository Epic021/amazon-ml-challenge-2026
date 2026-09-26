# Amazon ML Challenge 2026: Business Entity Resolution. Strategy

**Status (Sep 26, ~11:15 IST):**
- **Best submission:** B2. Holdout 0.9772, **LB 0.970**.
- **Target:** top 50 (≥ 0.979 on Sep 26), so we need **+0.009**.
- **Running:** B3 (name-uniqueness features): training, holdout log-loss 0.0043 at round 450.
- **Also running:** the EMB retriever (Indian-script names), relaunched after an out-of-memory fix.
- **Repo is private:** code reaches the VM only via `git archive HEAD src scripts requirements.txt | ssh vm tar -x` (the team decided on no deploy key).
- **Deadline:** around early Sep 28 IST. Verify on the portal.

Supporting docs:
- [eda/report.html](eda/report.html): EDA.
- [bottle_necks.md](bottle_necks.md): design options per stage.
- [research_sota.md](research_sota.md): literature and sources per component.

---

## 1. The problem

For each **Source-1 (S1)** business, output the set of **S2/S3** records that are the same business. The set may be empty.

| | Train | Test |
|---|---|---|
| S1 | 2.21M (US 1.32M, India 0.88M) | 1.73M (India 47%, US 38%, **France 15%, unseen in train**) |
| S2 + S3 | 10.3M | 9.97M |

**Metric:** F0.5 per S1, macro-averaged over all S1.
- Per entity: `F = 1.25·TP / (|pred| + 0.25·|true|)`.
- A singleton predicted empty scores 1.
- **A wrong match costs ~2.7× a missed one.**

**Rules that shape the solution:**
- No external data or lookups.
- The final model must be MIT or Apache-2.0 and ≤ 8B parameters.
- `candidate_pairs.tsv` must be exactly the set the model scores; it is audited for recall and reduction ratio.
- The code is re-run by the organizers, so nothing may be fitted to this particular test file.

## 2. What the data tells us (the facts the design is built on)

1. **The hard negatives were constructed: "neighbour businesses".**
   - 44% of unmatched records copy an S1 address almost exactly.
   - They change the house number and add a business-type or place word (Holdings, Group, North, Midtown; in French: Développement, Participations).
   - Among near-identical-address negatives, the **house number differs 88%** of the time. In true matches it is **equal 65%** of the time, with typos in 10%.
2. **Each S2/S3 record belongs to at most one S1.** Matches never cross countries. 5.6% of S1 have no match; S1 averages 3.46 matches.
3. **The sources are noisy in systematic ways:**
   - S2 is uppercase registry style, with 9% of names in Indian scripts.
   - S3 has typos, full or native-script state names, and `NULL` tokens.
   - About 3% of addresses are empty.
   - 3% of true matches are renamed ("Ectokor", "@1800").
4. **France follows the same generator:**
   - French words and legal forms;
   - region ↔ department swaps;
   - about 15 cities, so very dense streets.
5. **There is no leakage.** IDs and row order carry no signal, and train and test share no entities.
6. **There is no label shift.** Σp per S1 on test ≈ the holdout's.

## 3. The approach (as built)

```
raw TSV → normalize → 4 retrievers (per country) → union + prune → features → LightGBM
        → isotonic calibration → soft exclusivity → threshold → matching_results.tsv
```

| Stage | What | Why |
|---|---|---|
| Normalize (`normalize.py`) | ftfy, NFKC, anyascii (Indian scripts → Latin), abbreviations, ordinals, number extraction, `name_core` (legal forms and honorifics removed) | Make variants comparable. *Still holds hand-written US/India maps; replacing them is G1* |
| Retrievers (`block.py`, `block_emb.py`) | **word**: name + skeleton + address TF-IDF, both directions<br>**namechar**: char 3-grams of the name, S2/S3 → S1<br>**addr**: address only<br>**emb**: model2vec on raw non-Latin names | Each covers a different miss type: typos and domains, renamed businesses, empty addresses, Indian scripts |
| Union + prune (`union.py`, `features.py`) | Per-retriever scores and ranks kept as features; keep pairs with rank_q ≤ 10 or rank_s ≤ 15 | Recall with a bounded candidate set |
| Features (`features.py`, ~80) | **House-number relation** (equal / added / dropped / differs, edit distance, prefix); **added/dropped name words**: label log-odds + **per-country label-free roles** (G3) + name frequency; rapidfuzz similarities; **rank context** (gap to best, mutual best, competing S1); mined equivalences; **name uniqueness** (G5) | Separates decoys from typo-matches. The label-free roles carry the decoy vocabulary to France |
| Model (`train.py`) | LightGBM binary, 255 leaves, early stopping on the holdout | GBDT is near-SOTA for short structured records |
| Decode (`decode.py`) | Isotonic calibration (folds 6–9) → soft exclusivity p′ = o/(1+Σo) → threshold chosen on the holdout (the exact expected-F DP is also evaluated) | Enforces one S1 per record; optimizes set-level F0.5 |

**Not doing:** LLMs, geocoding or libpostal (external data), uroman (no gain, licence unclear), 5-fold CV, big hyperparameter searches.

## 4. How we validate and decide

**Folds** (S1 split by `int(id[3:]) % 10`):

| Folds | Use |
|---|---|
| 0–1 | Word statistics and mined rules |
| 2–4 | Training |
| **5** | **Holdout: the score we trust** |
| 6–9 | Calibration |

Blocking always runs on the full universe.

**Every run reports:**
- **holdout macro F0.5** after decoding, split by country and singleton;
- the **ceiling** (the F0.5 of a perfect model on our candidates);
- the **France proxy** (`france_diag.py`): the share of predicted matches that look like decoys, plus predictions per S1.

**Reading the LB:** test is 46.8% India, 38.3% US, 15% France. So `LB ≈ 0.468·India + 0.383·US + 0.15·France`, with India and US taken from the holdout, which gives the implied France score.

**Keep a change** only if the holdout macro F0.5 improves by **≥ +0.001** and the France proxy does not get worse. Submit whenever that holds and the validator passes.

## 5. Results

| Run | Change | Holdout (India / US) | Ceiling | LB | Decision |
|---|---|---|---|---|---|
| B0 | Exact key: name tokens + first number | 0.587 | – | – | Baseline |
| B1 | word retriever + 60 features + LightGBM + calibration + soft exclusivity + threshold 0.55 | 0.9662 (0.9546 / 0.9739) | 0.9768 | 0.952 | Implied France ≈ 0.88 |
| **B2** | + namechar & addr retrievers + G3 label-free word roles | **0.9772** (0.966 / 0.9847) | 0.9867 | **0.970** | **Best.** Implied France ≈ 0.94; France decoy share 2.08% → 0.16% |
| B2-noLO | B2 without the English word log-odds | 0.9762 | 0.9867 | – | Dropped |
| B3 | B2 + G5 name uniqueness | ⏳ | 0.9867 | – | Running |

**What we learned:**
- **The model reaches ~99% of its ceiling. Blocking recall and France are the levers, not the model.**
- **Decoding:** soft exclusivity beats hard by +0.0008. The exact expected-F DP is correct (it matches brute force) but does not beat a threshold, because the candidates are not independent.
- **G3 learned each country's decoy words without labels:**
  - US: southside, midtown, holdings.
  - France: participations, développement, groupe.
- **Where B2 loses its 0.0228 on the holdout:**

  | Source | Size |
  |---|---|
  | Blocking misses | 3.45% of true pairs (**India 6.1%**, US 1.7%), mostly Indian-script names |
  | Model rejections | 2.0% of true pairs, **74% of them empty-address candidates** |
  | Wrong accepts | 0.33% of predictions |

## 6. Plan to +0.009 (in order)

| # | Item | Targets | Est | Keep if |
|---|---|---|---|---|
| 1 | **G5 name uniqueness** (B3, running) | Empty-address rejections (74% of model misses) | running | Holdout ≥ +0.001 |
| 2 | **RET-EMB** (running): model2vec retriever for non-Latin names; the correct English name ranks first in 5/5 real misses | India blocking (6.1% of India's true pairs) | running | India ceiling ≥ +0.003 |
| 3 | **B4 = union with EMB → features (G5 built in) → retrain** | Both of the above | ~4 h | Holdout > B3 |
| 4 | **G4 house-number roles** (locality cardinality; match levels) | France's dense streets, remaining decoys | 2 h | France proxy ↓, holdout ≥ +0.001 |
| 5 | **F1 synthetic French decoys** (number ±1–2, added frequent word) | Measure France rejection directly | 1.5 h | Diagnostic |
| 6 | **RET-B reranker**: wide union → top 40 | Recall at a fixed candidate budget | 3 h | Recall@40 ≥ union recall − 0.001 |
| 7 | **G1 normalize v2**: structural rules only, drop the hand-written maps | Generality, code audit | 2 h | Holdout ≥ B-current − 0.001 |
| 8 | Stretch: S3 larger training set (60% of S1); S2 S2↔S3 twins; cross-encoder feature (laptop GPU → ONNX) | +0.001–0.003 each (est.) | 1–6 h | Holdout ≥ +0.001 |

**Final 12 h, no new features:**
1. Freeze.
2. One-command regeneration from raw data (output must match the submitted file).
3. Validator with `--check-ids`.
4. Zip: `output/` + `code/business_entity_resolution/{src,README.md,requirements.txt}` + methodology doc.
5. Final upload ≥ 3 h before the deadline.

## 7. How to run (VM, repo root)

Setup:
```
source /work/venv/bin/activate
export PYTHONUNBUFFERED=1
```

**Always `bash scripts/smoke.sh` first** (~6 min, ~1% of the data, isolated in `/tmp`). Full runs use `env -u BER_DATA -u BER_OUT`.

| # | Step | Command | Output | Time (full) |
|---|---|---|---|---|
| 0 | TSV → Parquet | `python scripts/tsv_to_parquet.py` | `data/parquet/` | 2 min |
| 1 | Normalize | `python src/normalize.py` | `data/norm/` | 3 min |
| 2 | Retrievers (per split) | `python src/block.py --split S --retriever {word,namechar,addr}`; `python src/block_emb.py --split S` | `data/cand/{S}_{r}.parquet` | ~25 / 20 / 12 min, emb tbd |
| 3 | Union (+ ceiling on train) | `python src/union.py --split S` | `data/cand/{S}.parquet` | ~20 min |
| 4 | Equivalence mining | `python src/mine_equiv.py --split {train,test}` | `data/equiv/` | 2 min |
| 5 | Features | `python src/features.py --split {train,test}` | `data/feat/` | ~60 / 50 min |
| 6 | Train | `python src/train.py --tag T` | `data/models/`, `data/pred/` | ~1.5–2 h |
| 7 | Decode | `BER_OUT=output_T python src/decode.py --tag T` | `output_T/*.tsv` | ~15 min |
| 8 | Checks | `validate_submission.py --check-ids`; `scripts/france_diag.py <file>`; `scripts/error_analysis.py --tag T` | logs | ~10 min |

## 8. Rules

- **Process:**
  - Smoke test before every full run.
  - One change per run.
  - Log every run in §5 and update §6.
  - Unbuffered logs.
  - Stop the VM when idle.
- **Compliance:**
  - No external lookups.
  - Models MIT/Apache only: LightGBM, model2vec potion-multilingual-128M. Not allowed: jina-v3 (CC BY-NC), Unidecode (GPL).
  - Every per-country table (IDF, word roles, name counts, equivalences) is **recomputed from the input data at run time**.
  - Everything learned from unlabeled test inputs is disclosed in the methodology doc.
- **Git:** no Claude attribution in commits.

## 9. Team

| Person | Owns |
|---|---|
| P1 | Validation, decoding, the experiment log, submission sign-off |
| P2 | VM, retrievers, full runs, final reproduction + zip |
| P3 | Features (G4, G5), training-set size |
| P4 | Normalization v2 (G1), France (F1), methodology doc; cross-encoder on the laptop GPU if time allows |
