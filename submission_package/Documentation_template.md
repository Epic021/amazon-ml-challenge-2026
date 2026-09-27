# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Team name]
**Team Members:** [Members]
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

The pipeline has four stages:
1. **Retrieval:** four complementary per-country TF-IDF retrievers generate candidates.
2. **Stage 1:** a LightGBM model scores every candidate pair on ~95 structural features. These are dominated by house-number relations, added/dropped-word evidence and rank context.
3. **Stage 2 (stacker):** a second LightGBM re-scores each pair using two independent pair models (ours and a teammate model with a different design). It also uses **competition features computed from both models' probabilities**: does the record fit another business better, the business's total probability mass, and twins across sources.
4. **Decoding:** calibrated probabilities are decoded with a soft one-owner constraint and an F0.5-optimal threshold.

The key ideas:
- **structural decoy features**: neighbour businesses are built by changing the house number and adding a type word;
- **label-free, per-country word roles**, which carry decoy vocabulary to the unseen country (France);
- **stacking two diverse models with relational features.** This was worth +0.003 over a fixed average, and most of it came from France.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Constructed hard negatives.** 44% of unmatched records copy an S1 address almost exactly.
  - They change the house number (88% of the time among near-identical-address negatives) and add a business-type or place word (Holdings, Group, North, Midtown; in French Groupe, Développement, Participations).
  - In true matches the house number is equal 65% of the time, with typos in 10%.
- **One owner per record.** Each S2/S3 record belongs to at most one S1, and matches never cross countries.
  - An S1 has 3.46 matches on average; 5.6% of S1 have none (singletons).
- **Source noise:**
  - S2 is uppercase registry style, and 9% of its names are in Indian scripts.
  - S3 has typos, native-script state names and `NULL` tokens.
  - About 3% of addresses are empty, and about 3% of true matches are renamed.
- **Empty-address records are almost never decoys.** 97.7% of them belong to some S1, against 73% of records with an address. When 2–3 S1 share the same core name, which one owns the record is essentially undetermined by the data. Our calibrated probability is 0.40 against an empirical 0.41.
- **France (test only)** follows the same generator:
  - French legal forms, region ↔ department swaps, abbreviations (R, AV, BD, N°, Bis);
  - about 15 cities, so streets are very dense;
  - short template-like names ("{token} {generic word} {legal form}").
  - Its hardest class is **same address + same house number + one generic word swapped**: 4.3 such candidate pairs per S1, against 0.95 in the US.
- **No leakage.** Row order, ID numbers and adjacency carry no signal: the correlation is 0.0001, and the tie-break accuracy by ID or row is at chance.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage classifier (stacked ensemble of two independent pair models) + constrained decoding.

**Core Innovation:**
- **Structural, transferable decoy evidence:** house-number relation classes, and per-country word roles learned without labels from the unlabeled data itself.
- **A relational stacker** that learns *when* to trust which of two diverse models, using competition features computed from both models' probabilities.

**Validation protocol (no leakage):**
- S1 entities are split into 10 folds by `int(id) mod 10`:
  - folds 0–1: word statistics;
  - folds 2–4: stage-1 training;
  - **fold 5: holdout**;
  - folds 6–8: stage-2 training;
  - fold 9: calibration.
- Blocking always runs against the full record universe.
- Every change is kept only if the holdout macro F0.5 improves by ≥ 0.001.
- France is tracked with a label-free proxy and a leave-one-country-out stress test (train on US, score India, and the reverse).

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used** (all per country, TF-IDF + sparse top-k via `sparse_dot_topn`, in both directions):
  1. **word**: name words + consonant skeletons + address words;
  2. **namechar**: character 3-grams of the core name (legal forms removed), record → S1;
  3. **addr**: address words only (catches renamed businesses);
  4. **numaddr**: `house-number|street-token` keys, taking the top-50 by key cosine and re-ranking by 0.5·key + 0.5·name similarity (catches noisy names at the right address).
- **Union and pruning:**
  - Every retriever's score and ranks are kept as features.
  - A pair is kept if it is in the S1's top 10 or the record's top 15.
  - The stage-2 pair set is the union of our candidates and the teammate model's candidates.
- **Candidate pairs generated (test):** ≈ 128 M scored pairs (≈ 74 per S1), out of ~1.7 M × ~10 M possible.
- **How we ensured true matches were not lost:**
  - Four retrievers with different failure modes, run in both directions.
  - Both the recall and the oracle F0.5 ceiling are measured on every run.
  - Final pair recall on the holdout is **0.990**, with an oracle ceiling F0.5 of **0.9968**.
  - An embedding retriever (model2vec) was evaluated and dropped: it recovered only 0.16% extra true pairs.

---

## 4. Matching Model

**Features used (stage 1, ~95):**
- **Name features:**
  - rapidfuzz ratio, token-set, token-sort, partial and Jaro-Winkler on the full name, core name, consonant skeleton and a sound-alike key (praivet ≈ private);
  - counts of added and dropped words;
  - label log-odds of added/dropped words (learned on folds 0–1 only);
  - **per-country label-free word roles** (how often a word is "added" relative to its presence in S1 names);
  - legal-form conflicts;
  - name uniqueness (how many S1 share the core name).
- **Address features:**
  - token overlap and edit similarities;
  - **house-number relation** (equal / added / dropped / partial / all differ), number edit distance, prefix match and first-number log difference;
  - address-word log-odds;
  - empty-address flag.
- **Other:**
  - every retriever's score and ranks, and the number of retrievers that proposed the pair;
  - rank context (gap to the S1's best candidate, gap to the record's best S1, mutual best);
  - twins (other-source candidates of the same S1 with the same number set or core name);
  - source flags; non-Latin script flag.

**Stage 2 (stacker) features:**
- both models' probabilities, their mean and their difference;
- for each of the three probabilities: the record's best probability with another S1, the margin to it, the S1's probability mass, the rank within the S1, and twins in probability space;
- all stage-1 features.

**Model type:** LightGBM (MIT) for both stages; 255 / 127 leaves, learning rate 0.05, early stopping on the holdout fold. No neural models and no external data.

**Threshold selection method:**
1. isotonic calibration on a fold never trained on;
2. soft one-owner renormalisation p′ = o/(1+Σo) over each record's S1 candidates (o = odds), which beat hard exclusivity by +0.0008;
3. a global threshold chosen to maximise holdout macro F0.5.

An exact expected-F0.5 set decoder was also implemented and evaluated.

---

## 5. Results & Error Analysis

| Stage | Holdout macro F0.5 (India / US) | Public LB |
|---|---|---|
| Stage 1, word retriever only (B1) | 0.9662 | 0.952 |
| + two more retrievers + label-free word roles (B2) | 0.9772 | 0.970 |
| + numaddr retriever (B4) | 0.98265 | – |
| + fixed 50/50 blend with teammate model | 0.98487 (0.9833 / 0.9859) | 0.979 |
| **+ relational stacker (final family)** | **0.9879 – 0.9883 (0.987 / 0.989)** | **0.983** |

- **Where the remaining holdout loss is (0.012):**
  - two-thirds is missed matches: 1.0% of true pairs never retrieved (India 1.5%), 1.9% rejected by the model;
  - one-third is wrong merges: 0.31% of predictions.
- **Common false positives (wrong merges):**
  - neighbour businesses at the same street address with the same house number where one generic word differs;
  - legal-form-only variants of another business at the same address.
- **Common false negatives (missed matches):**
  - **empty-address records whose name is shared by 2–3 S1.** These are about 60–70% of model rejections, and they are calibrated coin flips given the available fields;
  - heavily renamed businesses;
  - Indian-script names that no retriever surfaces.
- **France:**
  - The implied France F0.5 rose from 0.88 (B1) to 0.94 (label-free word roles), then to ≈ 0.95 (stacker).
  - The stacker removed 0.135 predictions per French S1, mostly same-number word swaps, and the leaderboard confirmed those were predominantly wrong merges.
  - A leave-one-country-out stress test (US → India loses 0.04, similar to France's gap) showed that no feature group harms transfer, and that no stricter threshold is justified for unseen countries.

---

## 6. Conclusion

- Precision-first entity resolution here is mostly about **structure**: house-number relations, one-owner constraints and the evidence of competing businesses. That structure transfers to an unseen country far better than word identity does.
- Stacking two independently designed models with relational features was the single biggest late gain.
- The main open problem is the unseen-country gap (France), where the same-address word-swap class behaves differently from the training countries.

---

## Appendix

### A. Code Artefacts

The runnable pipeline is under `code/business_entity_resolution/`: `src/`, `README.md` (exact commands), and `requirements.txt` (pinned).

1. **Teammate model:**
   - `src/teammate/`: `prep.py → retrieve.py → features.py → model.py → export_probs.py`;
   - it writes out-of-fold train probabilities and test probabilities.
2. **Our pipeline:**
   - `src/`: `tsv_to_parquet.py → normalize.py → block.py ×3 + block_numaddr.py → union.py → mine_equiv.py → features.py → train.py` (stage 1);
   - then `stage2.py --friend` (stacker) → `decode.py` → `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
3. **Checks:** the organisers' `validate_submission.py --check-ids`.

### B. Additional Results

- Decoding: soft exclusivity +0.0008 vs hard. The expected-F DP matched brute force but did not beat the threshold on stage-1 probabilities.
- Dead ends:
  - embedding retriever: India ceiling +0.002;
  - more stage-1 training data: flat learning curve;
  - evidence dropout: −0.0004;
  - dropping retrieval-rank features: France unchanged, holdout −0.001;
  - collective "repeated difference" features: already captured by the model.
