# Data-handling strategies: experiment tracker

**Status: all 7 strategies scored, and test TSVs exist for all of them (validator PASS). Strategy C is chosen and tuned. **Best model: tuned C, 0.98331.** The run is on GCP VM `ber-train` (`c2d-standard-56`, Mumbai). It started 2026-09-26 at 08:21 UTC.

| Milestone | ETA (UTC) |
|---|---|
| First submission file (strategy A) | ~09:50 |
| All seven strategies scored | ~10:45 |

This file is updated as results come in.

## What is being tested

Each strategy changes only **which training rows the model learns from**, or their weights.

- The pair features are computed once and shared by all strategies.
- Every strategy is scored the same way: out-of-fold macro F0.5 over **all** training S1 businesses, after exclusive assignment and set decoding.
- Scores use **untouched** labels: the model trained on one half of the S1s predicts the other half.

| Strategy | What changes in training | Why try it | Status | Macro F0.5 | vs A |
|---|---|---|---|---|---|
| **A** | nothing: keep every label (baseline, used for the first submission) | the label audit found cleaning hurts | DONE | 0.98217 | – |
| **D** | drop pairs the first-stage model contradicts strongly (match with p < 0.02, no-match with p > 0.98) | audit: −0.01 (idealised) | DONE | 0.98231 | +0.00014 |
| **E** | easy pairs (p < 0.002 or > 0.998): keep 10%, weight ×10 | audit: −0.05, with 30% of the rows | DONE | 0.98247 | +0.00030 |
| **G** | down-weight (×0.2) the pairs in D instead of dropping them | soft version of D | DONE | 0.98251 | +0.00034 |
| **B** | drop pairs labelled against their pattern's majority | audit: −1.59 (idealised) | DONE | **0.94760** | **−0.03457** |
| **C** | drop whole patterns whose minority label exceeds 5% | audit: −0.46 | DONE | **0.98265** | **+0.00048** |
| **F** | drop empty-address records whose identical name is labelled inconsistently | the one exact-duplicate conflict in the audit | DONE | 0.98222 | +0.00005 |

**Pattern** = (country, source, name-change type, address-change type, number-change type).

**Fast-mode settings (same for all strategies):**
- LightGBM with up to 12M training rows per model.
- Stratified negatives: hard-looking decoys first, easy ones sampled, weighted so nothing is biased.
- Learning rate 0.15, at most 600 rounds, early stopping.
- B–G reuse A's first stage and apply their change to the final (second-stage) model.

**Decision rule:** submit the strategy with the highest untouched macro F0.5. A stays unless another beats it by more than noise, about 0.0005.

## Reading the results
- **B collapses (−3.5 points).** Making every pattern pure teaches the model that patterns decide the label outright, so it fails on the mixed cases.
- **A, D, E, G, C and F are all within 0.0005 of each other**, which is about the noise level of one run. C (+0.00048) is the best, then G and E. Here C cleans only the second-stage model's training rows; the first stage still learns from every label.

## Strategy C improvements

| Step | Status | Result |
|---|---|---|
| Wider decoder grid (λ up to 3.0) on C | DONE | λ = 0.4 is still the best: 0.98265, no change |
| Full-size stage 1 (16M rows per half, learning rate 0.1) | DONE | stage-1 macro F0.5 0.98098 (fast mode: 0.98048) |
| Stage-2 tuning grid (6 settings: learning rate 0.1 / 0.05 / 0.03, leaves 127–511, regularization) | DONE | 0.98316 / 0.98320 / 0.98327 / 0.98322 / 0.98330 / **0.98331** (best: learning rate 0.05, 255 leaves, L2 10) |
| Tuned C test TSV + probability export for blending | DONE | **0.98331** (+0.00066 vs C fast, +0.00114 vs A); validator PASS |
| Our own ensemble: C + G + E + tuned C | DONE | 0.98320, below tuned C alone, so the ensemble isn't used |

Scripts: `model.py tune C`, `posthoc.py redecode|ensemble`, `export_probs.py`, and `queue_c.sh`, the unattended queue that stops the VM at the end.

## Pipeline progress

| Stage | Status | Result |
|---|---|---|
| Train prep (normalization, learned areas) | DONE | 12.5M records; areas learned: US 189, India 221 |
| Train retrieval | DONE | 116.1M candidate pairs (52.6 per S1); **98.55%** of true pairs retrieved; ceiling macro F0.5 0.9953 |
| Train pair features | DONE | 116.1M pairs |
| Test prep + retrieval | DONE | 108.8M candidate pairs; areas learned for France 24, India 223, US 186 |
| Test pair features | DONE | 108.8M pairs |
| Model training (A, then D, E, G, B, C, F) | DONE | A: stage 1 macro F0.5 0.98048, after self-correction (stage 2) **0.98217**; ceiling after pruning 0.99519 (9.96M pairs) |
| Test predictions + `matching_results.tsv` (A) | DONE | validator PASS; predicted singletons ~5.3–5.8% and ~3.3 matches per S1 in every country (France included) |

## Reproduce

```bash
cd code/business_entity_resolution/src
BER_TAG=_fast BER_MAX_ROWS=12000000 BER_LR=0.15 BER_ROUNDS=600 BER_REUSE_STAGE1=1 \
  python3 model.py train A,D,E,G,B,C,F
BER_TAG=_fast python3 model.py test A && BER_TAG=_fast python3 submit.py A
```

Background: [`DATASET_STRATEGY.md`](DATASET_STRATEGY.md) explains each case in plain words. The audit and cleaning experiment are in [`../eda_output/label_audit/REPORT.md`](../eda_output/label_audit/REPORT.md).
