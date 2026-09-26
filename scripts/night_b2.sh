#!/usr/bin/env bash
# Overnight follow-up to the B2 chain (runs on the VM, independent of the laptop session).
#  1. wait for B2 decode -> save + validate + France diagnostic
#  2. B2-noLO: retrain without the label-based (English-only) word log-odds -> decode -> validate + diag
#  3. write logs/night_summary.txt
set -uo pipefail
unset BER_DATA BER_OUT   # never inherit smoke-test paths from the launching shell
cd /work/amazon-ml-challenge-2026
source /work/venv/bin/activate
export PYTHONUNBUFFERED=1
VAL="python student_resource/utils/validate_submission.py --test-dir student_resource/dataset/test --check-ids"
S=logs/night_summary.txt

until grep -q "label-shift check" logs/b2_decode.log 2>/dev/null; do sleep 60; done

mkdir -p output_b2 && cp output/matching_results.tsv output/candidate_pairs.tsv output_b2/
{
  echo "===== B2 (union candidates + G3 word roles) ====="
  grep -E "holdout S1|BEST|by country|\[test\]|label-shift" logs/b2_decode.log
  $VAL --matching output_b2/matching_results.tsv --candidate output_b2/candidate_pairs.tsv 2>&1 | tail -2
  python scripts/france_diag.py output_b2/matching_results.tsv
  echo "B1 for comparison:"
  echo "  (B1 holdout 0.9662, LB 0.952)"
} >> "$S" 2>&1

python src/train.py --tag b2nolo --drop add_lo_,drop_lo_,n_added_unknown,n_dropped_unknown > logs/b2nolo_train.log 2>&1 \
  && BER_OUT=/work/amazon-ml-challenge-2026/output_b2nolo python src/decode.py --tag b2nolo > logs/b2nolo_decode.log 2>&1
{
  echo; echo "===== B2-noLO (same, without label-based English word log-odds) ====="
  grep -E "holdout S1|BEST|by country|\[test\]|label-shift" logs/b2nolo_decode.log
  $VAL --matching output_b2nolo/matching_results.tsv --candidate output_b2nolo/candidate_pairs.tsv 2>&1 | tail -2
  python scripts/france_diag.py output_b2nolo/matching_results.tsv
  echo; echo "NIGHT DONE $(date)"
} >> "$S" 2>&1
