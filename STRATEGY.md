# Amazon ML Challenge 2026: Business Entity Resolution. Strategy

**Status (Sep 26, ~19:00 IST)**

| | |
|---|---|
| **Best LB** | **0.979**, BLEND-B4 (our B4 + the teammate's model), holdout 0.98487 |
| **Where the LB is lost** | **France ≈ 0.94** (implied), unchanged since B2. India and the US are ≈ 0.983 / 0.986 |
| **Target** | 0.98+ (top 10 ≈ 0.985). Each +0.01 on France is +0.0015 on the LB |
| **Running** | B5 decode (B5 trained; features: sound key, address-word log-odds, twins); B5 + teammate blend on the holdout |
| **Next** | 3g stacked blend → 3h France pseudo-labels (§6) |
| **Deadline** | around early Sep 28 IST (verify on the portal); freeze ~Sep 27 evening |

The repo is private, so code reaches the VM only via `git archive HEAD src scripts requirements.txt | ssh vm tar -x`.

Supporting docs:
- [eda/report.html](eda/report.html): EDA.
- [bottle_necks.md](bottle_necks.md): design options per stage.
- [research_sota.md](research_sota.md): literature per component.

---

## 1. The problem

For each **Source-1 (S1)** business, output the set of **S2/S3** records that are the same business. The set may be empty.

| | Train | Test |
|---|---|---|
| S1 | 2.21M (US 1.32M, India 0.88M) | 1.73M (India 47%, US 38%, **France 15%, no French labels in train**) |
| S2 + S3 | 10.3M | 9.97M |

**Metric:** F0.5 per S1, macro-averaged.
- Per entity: `F = 1.25·TP / (|pred| + 0.25·|true|)`.
- An empty prediction for an S1 with no match scores 1.
- **A wrong match costs ~2.7× a missed one.**

**Rules that shape the solution:**
- No external data or lookups.
- Models must be MIT/Apache and ≤ 8B parameters.
- `candidate_pairs.tsv` must be exactly the scored set; it is audited for recall and reduction.
- The organizers re-run the code, so nothing may be fitted to this particular test file.

## 2. Data facts the design rests on

1. **Hard negatives are constructed "neighbour businesses".**
   - They copy an S1 address, change the house number (88% of the time) and add a business-type or place word (Holdings, Midtown; in French Groupe, Développement).
   - In true matches the house number is equal 65% of the time, with typos in 10%.
2. **Each S2/S3 record belongs to at most one S1.**
   - Matches never cross countries.
   - An S1 has 3.46 matches on average; 5.6% have none.
3. **Source noise:**
   - S2 is uppercase registry style, with 9% of names in Indian scripts.
   - S3 has typos, native-script state names and `NULL` tokens.
   - About 3% of addresses are empty.
   - About 3% of matches are renamed ("Ectokor").
4. **France follows the same generator**, with French words and legal forms, region ↔ department swaps, and about 15 cities (so very dense streets).
5. **The hardest class is the same address and house number with one name word swapped** ("Horizon Amis SARL" vs "Horizon Club SARL").
   - Train: only 43% (India) / 48% (US) of these are true matches.
   - Test pairs per S1: France **4.3**, India 2.1, US 0.95.
6. **No leakage** (IDs and row order carry no signal) and **no label shift** (Σp per S1 on test ≈ the holdout's).

## 3. Pipeline (as built)

```
raw TSV → normalize → 4 retrievers → union + prune → features (~95) → LightGBM
        → isotonic → [blend with teammate] → soft exclusivity → threshold → TSVs
```

| Stage | What |
|---|---|
| Normalize (`normalize.py`) | ftfy, NFKC, anyascii, abbreviations, numbers, `name_core` (legal forms removed). *Still has hand-written US/India maps (G1)* |
| Retrievers (`block.py`, `block_numaddr.py`) | **word** (name + skeleton + address TF-IDF, both directions); **namechar** (char 3-grams); **addr**; **numaddr** (`number\|token` address keys, re-ranked by name) |
| Union + prune (`union.py`) | Per-retriever scores and ranks kept as features; keep rank_q ≤ 10 or rank_s ≤ 15. **Ceiling 0.9917** |
| Features (`features.py`) | House-number relation; added/dropped words (label log-odds, per-country label-free roles, name uniqueness); sound-alike key; address-word log-odds; twins; rapidfuzz similarities; rank context |
| Model (`train.py`) | LightGBM binary, 255 leaves, early stopping on the holdout |
| Decode (`decode.py`, `scripts/blend.py`) | Isotonic (folds 6–9) → blend w = 0.5 with the teammate → soft exclusivity p′ = o/(1+Σo) → threshold |
| Stacking (`stage2.py`) | Re-score with competitors' probabilities (record/S1 competition, twins in probability space). Smoke passed |

**Not doing:** LLMs, geocoding or libpostal, uroman, 5-fold CV, big hyperparameter searches, running the teammate's code on our VM (we use only their probabilities and EDA).

## 4. Validation and decision rules

**Folds** (`int(id[3:]) % 10`): 0–1 word statistics · 2–4 train · **5 holdout** · 6–9 calibration and blend tuning.

**Reading the LB:** `LB ≈ 0.468·India + 0.383·US + 0.15·France`. India and the US come from the holdout, which gives the implied France score. The LB shows 3 decimals, so a change needs about +0.001 to be visible.

**Keep a change only if** the holdout improves by **≥ +0.001** (strict; B3 lesson). **Prioritize changes worth ≥ +0.003.** France can only be checked on the LB, so spend LB submissions on France-targeted changes.

**Evidence gates: nothing runs at full scale without a cheap measurement first.**

| Change | Cheap evidence | Gate |
|---|---|---|
| Retriever | Realistic sample → train union ceiling | Ceiling ≥ +0.003 |
| Features | 10% pilot, small LightGBM, with vs without | **≥ +0.002** (B5: pilot +0.001 → full +0.0002; pilots overstate) |
| More data | Learning curve on 1 / 2 / 3 folds | Still rising |
| France-only change | Offline simulation on India (below) | Passes the gate, then an LB submission |

**France tool:** `scripts/france_gap.py --tag T [--w W --thr T --only_a]`. It reports:
- the model's own expected F0.5 per country (Monte Carlo from calibrated p; within 0.002 of the true score for the blend);
- borderline-pair counts per country;
- how often ours and the teammate's disagree, and which one is right on the holdout;
- a feature profile of borderline pairs;
- France examples written to `logs/france_examples.tsv`.

## 5. Results

| Run | Change | Holdout (India / US) | Ceiling | LB | Implied France | Decision |
|---|---|---|---|---|---|---|
| B0 | Exact key | 0.587 | – | – | – | Baseline |
| B1 | word retriever + 60 features + LightGBM + calibration + soft exclusivity | 0.9662 (0.9546 / 0.9739) | 0.9768 | 0.952 | 0.88 | – |
| B2 | + namechar, addr + label-free word roles | 0.9772 (0.966 / 0.9847) | 0.9867 | 0.970 | 0.94 | Kept |
| B2-noLO | B2 without word log-odds | 0.9762 | 0.9867 | – | – | Dropped (log-odds worth +0.001) |
| B3 | + name uniqueness | 0.9777 (0.9664 / 0.9853) | 0.9867 | 0.970 | 0.94 | +0.0005, below the bar; invisible on the LB |
| B4 | + numaddr retriever; polars pipeline | 0.98265 (0.9772 / 0.9863) | 0.9917 | not submitted | – | Best single model; 99.1% of the ceiling |
| **BLEND-B4** | B4 + teammate (isotonic each, w = 0.5, thr 0.45) | **0.98487** (0.9833 / 0.9859) | – | **0.979** | **0.94** | **Current best.** +0.0029 vs B4, +0.0025 vs the teammate (CIs > 0) |
| **STACK-B5** | Stage 2 on B5 + teammate p (both models' p, relational features on each and on their mean, pair features); trained folds 6–7, early stop 5, isotonic 8–9, soft excl, thr 0.65 | **0.98792** (0.9866 / 0.9888) | – | ⏳ | **+0.0031 vs BLEND-B4.** Top features: mean p 62%, teammate p 23%, margin vs record's other S1 12% |
| B5 | B4 + sound key + address-word log-odds + twins | 0.98288 (0.9776 / 0.9864) | 0.9917 | – | – | **+0.0002 only** (pilot said +0.001). Blend with teammate = BLEND-B4 (0.98434 vs 0.98433 at w=0.3); not submitted |

**What we learned:**
- **India and the US are near their ceiling; France is the gap.** Every LB gain since B2 came from India and the US. France has sat at ≈ 0.94 across B2, B3 and BLEND-B4.
- **France errors are confident, not just uncertain.**
  - The blend believes France ≈ 0.967 and the LB says ≈ 0.94.
  - France has 3× the borderline pairs of India or the US, and 2× the disagreements between our model and the teammate's, which point in opposite directions (0.95 vs 0.001).
  - The losses sit in the same-address word-swap class (fact 5). Our models learned which swapped words matter from English and Indian labels.
- **Blending two different models is worth +0.0025–0.003** on the holdout and was visible on the LB (+0.009 incl. B4).
- **Decoding:** soft exclusivity beats hard by +0.0008. The exact expected-F DP does not beat a threshold.
- **Label-free word roles work:** US southside, midtown, holdings; France participations, développement, groupe. France's decoy-signature share is only 0.2%, so classic decoys are handled.
- **Dead ends:** embedding retriever (India ceiling +0.002), more training data (+0.0002), evidence dropout (−0.0004), finer number relations (0).

## 6. Plan (in order)

| # | Item | Why | Time | Keep if |
|---|---|---|---|---|
| 1 | ~~B5 + BLEND-B5~~ **done, no gain** | – | – | – |
| 2 | **3g Stacked blend**: stage 2 with both models' p, relational features (does the record fit another S1 better?) and pair features (number relation, word swap) | The models disagree confidently on France swaps, and a fixed 0.5 average puts those at the threshold. The stacker learns when to trust which | 1 h | ≥ blend + 0.001 → submit |
| 3 | **3h France co-training pseudo-labels**: French pairs where both models agree confidently → French added/dropped-word log-odds → recompute France features → re-predict (no retrain) | France's vocabulary is unseen. Two independent models agreeing limits confirmation bias | 2–3 h | **Gate:** on India, log-odds from pseudo-labels in place of true ones keep ≥ half the log-odds gain. Then one LB submission |
| 3i | **France diagnosis (Sep 26 eve):** SHAP on France swap pairs (same address + number, one word swapped, sole claimant) blamed retrieval-rank features (rank_q_word France 53 vs 15/10). **Pilot FAILED:** dropping them costs −0.001 holdout and leaves France swap p unchanged (0.335 → 0.331); the model rejects via name similarity instead (French names are ~3 words, so one swap = 1/3 of the name). Retrain not done | – | done | ✗ |
| 3j | Teammate v2 probabilities (`data/friend_v2`): corr 0.99 with v1, 1.5% France flips, France swap p unchanged. Blend B4+v2 running | – | 25 min | ≥ BLEND-B4 + 0.001 |
| 3k | **France direction probe (needs team OK, costs 1 LB submission):** blend with France-only threshold 0.35; LB delta = 0.15·ΔF(France) → tells FP-heavy vs FN-heavy. Diagnostic only, never shipped as a tuned France threshold | Which way France errs | 10 min | – |
| 4 | **G4 house-number roles** (locality cardinality, match levels) | France's dense streets | 2 h | ≥ +0.001 or France LB ↑ |
| 5 | F1 synthetic French decoys | Measure France rejection directly | 1.5 h | Diagnostic |
| 6 | G1 normalize v2 (structural rules, no hand maps) | Code audit / generality | 2 h | Holdout ≥ current − 0.001 |

**Final 12 h (from ~Sep 27 evening), no new features:**
1. Freeze.
2. One-command regeneration from raw data (output must match the submitted file).
3. Validator with `--check-ids`.
4. Zip: `output/` + `code/business_entity_resolution/{src,README.md,requirements.txt}` + methodology doc. Resolve the folder clash with the teammate.
5. Upload ≥ 3 h before the deadline.

## 7. How to run (VM, repo root)

Setup:
```
source /work/venv/bin/activate
export PYTHONUNBUFFERED=1
```

**Always `bash scripts/smoke.sh` first** (~6 min, ~1% of the data, in `/tmp`). Full runs use `env -u BER_DATA -u BER_OUT`.

| # | Step | Command | Time (full) |
|---|---|---|---|
| 0 | TSV → Parquet | `python scripts/tsv_to_parquet.py` | 2 min |
| 1 | Normalize | `python src/normalize.py` | 3 min |
| 2 | Retrievers | `python src/block.py --split S --retriever {word,namechar,addr}`; `python src/block_numaddr.py --split S` | ~25 / 20 / 12 / 15 min |
| 3 | Union (+ ceiling on train) | `python src/union.py --split S` | ~5 min |
| 4 | Equivalences | `python src/mine_equiv.py --split S` | 2 min |
| 5 | Features | `python src/features.py --split S --out data/feat_T` | ~85 min train |
| 6 | Train | `python src/train.py --tag T --feat_dir data/feat_T` | ~80 min |
| 7 | Decode | `BER_OUT=output_T python src/decode.py --tag T` | ~15 min |
| 8 | Blend | `python scripts/blend.py --tag T --write` → `output_blend/` | ~15 min |
| 9 | Checks | `validate_submission.py --check-ids`; `scripts/france_diag.py`; `scripts/france_gap.py` | ~10 min |

## 8. Rules

- **Process:**
  - Smoke test before full runs.
  - Pass the evidence gate before scaling up.
  - One change per run.
  - Log every run in §5 and update §6 **before** reordering.
  - Stop the VM when idle.
- **Compliance:**
  - No external lookups.
  - MIT/Apache models only (LightGBM, model2vec). Not allowed: jina-v3 (CC BY-NC), Unidecode (GPL).
  - Per-country tables are recomputed from the input data at run time.
  - Anything learned from unlabeled test inputs (word roles, pseudo-labels) is disclosed in the methodology doc.
- **Team:**
  - Don't run the teammate's pipeline on our VM.
  - No Claude attribution in commits.

## 9. Team

| Person | Owns |
|---|---|
| P1 | Validation, decoding, experiment log, submission sign-off |
| P2 | VM, retrievers, full runs, final reproduction + zip |
| P3 | Features (G4), France pseudo-labels |
| P4 | G1, F1, methodology doc; teammate model / probabilities for blending |
