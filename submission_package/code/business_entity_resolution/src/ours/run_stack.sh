#!/usr/bin/env bash
# Stacked blend: stage 2 on our stage-1 predictions + the teammate's probabilities -> decode -> checks.
#   bash scripts/run_stack.sh TAG FEAT_DIR FRIEND_DIR TRAIN_FOLDS CALIB_FOLDS OUT_TAG [extra stage2 args]
# LB 0.983 (STACK-B5):  bash scripts/run_stack.sh b5 data/feat_b5 friend 6,7 8,9 b5sb
# Stacker v2:           bash scripts/run_stack.sh b5 data/feat_b5 friend_v2 6,7,8 9 b5sb2 --all_feats
set -uo pipefail
cd "$(dirname "$0")/.."
TAG=$1 FEAT=$2 FRIEND=$3 FOLDS=$4 CALIB=$5 OUT=$6; shift 6
export PYTHONUNBUFFERED=1
unset BER_DATA BER_OUT
python src/stage2.py --tag "$TAG" --feat_dir "$FEAT" --friend --friend_dir "$FRIEND" --train_folds "$FOLDS" \
    --out_tag "$OUT" "$@" > "logs/${OUT}_stage2.log" 2>&1 || { echo "STAGE2 FAILED $(date)" > "logs/${OUT}_summary.txt"; exit 1; }
BER_OUT="output_$OUT" python src/decode.py --tag "$OUT" --calib_folds "$CALIB" > "logs/${OUT}_decode.log" 2>&1
{ grep -E "BEST|by country|\[test\]" "logs/${OUT}_decode.log"
  python student_resource/utils/validate_submission.py --test-dir student_resource/dataset/test --check-ids \
      --matching "output_$OUT/matching_results.tsv" --candidate "output_$OUT/candidate_pairs.tsv" | tail -1
  python scripts/france_diag.py "output_$OUT/matching_results.tsv" "$FEAT"
  echo "${OUT^^} DONE $(date)"; } > "logs/${OUT}_summary.txt" 2>&1
