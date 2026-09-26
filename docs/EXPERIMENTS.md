# Data-handling strategies: experiment tracker

**Status: RUNNING.** The run is on GCP VM `ber-train` (`c2d-standard-56`, Mumbai). It started 2026-09-26 at 08:21 UTC.

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
| **G** | down-weight (×0.2) the pairs in D instead of dropping them | soft version of D | RUNNING | – | – |
| **B** | drop pairs labelled against their pattern's majority | audit: −1.59 (idealised) | queued | – | – |
| **C** | drop whole patterns whose minority label exceeds 5% | audit: −0.46 | queued | – | – |
| **F** | drop empty-address records whose identical name is labelled inconsistently | the one exact-duplicate conflict in the audit | queued | – | – |

**Pattern** = (country, source, name-change type, address-change type, number-change type).

**Fast-mode settings (same for all strategies):**
- LightGBM with up to 12M training rows per model.
- Stratified negatives: hard-looking decoys first, easy ones sampled, weighted so nothing is biased.
- Learning rate 0.15, at most 600 rounds, early stopping.
- B–G reuse A's first stage and apply their change to the final (second-stage) model.

**Decision rule:** submit the strategy with the highest untouched macro F0.5. A stays unless another beats it by more than noise, about 0.0005.

## Pipeline progress

| Stage | Status | Result |
|---|---|---|
| Train prep (normalization, learned areas) | DONE | 12.5M records; areas learned: US 189, India 221 |
| Train retrieval | DONE | 116.1M candidate pairs (52.6 per S1); **98.55%** of true pairs retrieved; ceiling macro F0.5 0.9953 |
| Train pair features | DONE | 116.1M pairs |
| Test prep + retrieval | DONE | 108.8M candidate pairs; areas learned for France 24, India 223, US 186 |
| Test pair features | DONE | 108.8M pairs |
| Model training (A, then D, E, G, B, C, F) | RUNNING | A: stage 1 macro F0.5 0.98048, after self-correction (stage 2) **0.98217**; ceiling after pruning 0.99519 (9.96M pairs) |
| Test predictions + `matching_results.tsv` (A) | DONE | validator PASS; predicted singletons ~5.3–5.8% and ~3.3 matches per S1 in every country (France included) |

## Reproduce

```bash
cd code/business_entity_resolution/src
BER_TAG=_fast BER_MAX_ROWS=12000000 BER_LR=0.15 BER_ROUNDS=600 BER_REUSE_STAGE1=1 \
  python3 model.py train A,D,E,G,B,C,F
BER_TAG=_fast python3 model.py test A && BER_TAG=_fast python3 submit.py A
```

Background: [`DATASET_STRATEGY.md`](DATASET_STRATEGY.md) explains each case in plain words. The audit and cleaning experiment are in [`../eda_output/label_audit/REPORT.md`](../eda_output/label_audit/REPORT.md).
