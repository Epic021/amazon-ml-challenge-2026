#!/usr/bin/env bash
# End to end: data -> candidates -> features -> two-stage model -> output/*.tsv
# Usage: bash run_all.sh [VARIANT]        (VARIANT defaults to A = keep every training label)
# Environment: BER_DATA (dataset dir with train/ and test/), BER_WORK (scratch), BER_OUT (output dir),
#              BER_NPROC (processes / threads; default: all cores)
set -euo pipefail
V="${1:-A}"
cd "$(dirname "$0")/src"
for split in train test; do
  python3 prep.py "$split"
  python3 retrieve.py "$split"
  python3 features.py "$split"
done
python3 model.py train "$V"
python3 model.py test "$V"
python3 submit.py "$V"
