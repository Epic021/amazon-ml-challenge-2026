#!/usr/bin/env bash
# End to end: raw TSVs -> teammate model probabilities + our stage-1 model -> stage-2 stacker -> output/*.tsv
#
#   bash run_all.sh /path/to/student_resource/dataset  [TEAMMATE_VARIANT]
#
# Two Python environments (the two models pin different library versions):
#   .venv      <- requirements.txt            (our pipeline + stacker)
#   .venv_tm   <- requirements_teammate.txt   (teammate pipeline)
# Scale: 64 vCPU / 500 GB RAM; ~9 h end to end.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RAW="${1:?path to the dataset folder containing train/ and test/}"
VARIANT="${2:-C}"
WORK="${BER_WORK_ROOT:-$HERE/work}"
mkdir -p "$WORK/data" "$HERE/output" "$HERE/logs"
export PYTHONUNBUFFERED=1

# ---- 1. Teammate model: out-of-fold train probabilities + test probabilities --------------------
(
  source "$HERE/.venv_tm/bin/activate"
  export BER_DATA="$RAW" BER_WORK="$WORK/teammate" BER_OUT="$WORK/teammate_out"
  cd "$HERE/src/teammate"
  for s in train test; do python3 prep.py $s; python3 retrieve.py $s; python3 features.py $s; done
  python3 model.py train "$VARIANT"
  python3 model.py test "$VARIANT"
  python3 export_probs.py "$VARIANT" "$WORK/data/friend"
) 2>&1 | tee "$HERE/logs/teammate.log"

# ---- 2. Our pipeline: stage 1 (B5) ------------------------------------------------------------------
source "$HERE/.venv/bin/activate"
export BER_DATA="$WORK/data" DATA_DIR="$RAW"
cd "$HERE/src/ours"
python tsv_to_parquet.py
python normalize.py
for s in train test; do
  for r in word namechar addr; do python block.py --split $s --retriever $r; done
  python block_numaddr.py --split $s
  python union.py --split $s
  python mine_equiv.py --split $s
  python features.py --split $s --out "$BER_DATA/feat_b5/$s.parquet"
done
python train.py --tag b5 --feat_dir "$BER_DATA/feat_b5" --drop numx_

# ---- 3. Stage 2 stacker over both models, decode -> output/ -------------------------------------------
python stage2.py --tag b5 --feat_dir "$BER_DATA/feat_b5" --friend --friend_dir friend \
    --train_folds 6,7,8 --all_feats --context_folds 0,1,5,6,7,8,9 --out_tag b5sb3
BER_OUT="$HERE/output" python decode.py --tag b5sb3 --calib_folds 9
echo "done: $HERE/output/matching_results.tsv, $HERE/output/candidate_pairs.tsv"
