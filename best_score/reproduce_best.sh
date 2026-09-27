#!/usr/bin/env bash
# Reproduces the best model ("tuned C", out-of-fold macro F0.5 0.98331) from the raw challenge files:
#   data -> normalisation -> candidates -> pair features -> stage 1 (full size, all labels)
#   -> stage 2 with strategy C (drop mixed patterns) and setting tune5 -> decoder -> output TSVs.
# Environment: BER_DATA (folder with train/ and test/), BER_WORK (scratch, ~60 GB), BER_OUT (outputs),
#              BER_NPROC (processes / threads; default all cores). Tested on 56 vCPU / 224 GB.
set -euo pipefail
cd "$(dirname "$0")/src"
for split in train test; do
  python3 prep.py "$split"
  python3 retrieve.py "$split"
  python3 features.py "$split"
done
# stage 1 once on every label (16M rows / half, lr 0.1), stage 2 = strategy C with S2_GRID[5]
BER_MAX_ROWS=16000000 BER_LR=0.1 BER_ROUNDS=1500 BER_REUSE_STAGE1=1 BER_S2_GRID_IDX=5 python3 model.py tune C
BER_TAG=_tune5 python3 model.py test C
BER_TAG=_tune5 python3 submit.py C          # -> $BER_OUT/matching_results.tsv + candidate_pairs.tsv
# optional: probabilities for blending with another model
python3 export_probs.py C_tune5 "${BER_WORK:-../../work}/ensemble_C_tuned" || true
