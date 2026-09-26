#!/usr/bin/env bash
# Fresh CPU VM -> environment, dataset (Google Drive), full CPU pipeline (B5), validated test submission,
# and the 4-file bundle for the GPU pod. Resumable: each finished step leaves logs/done/<step>; a re-run
# skips those. Step logs: logs/<step>.log.
#
#   git clone <repo> && cd amazon-ml-challenge-2026 && git checkout neural-members
#   tmux new -s setup
#   bash scripts/setup_cpu_vm.sh            (output live + logs/<step>.log; re-run after any failure)
# Environment: uv (installed if missing) -> .venv with Python 3.12 (numpy 1.26.4 has no 3.13 wheels).
#
# VM: Ubuntu 22.04/24.04, >= 32 vCPU, >= 256 GB RAM, >= 200 GB disk (the previous VM's spec). ~7 h on 32 vCPU.
# Result:
#   output/{matching_results,candidate_pairs}.tsv   B5 single-model submission (validator run at the end)
#   data/xenc_for_pod.tar                           copy to the pod, `tar -xf` in the repo root (README: GPU pod)
# Env: TAG (default b5), DRIVE_URL, SKIP_APT=1 (no sudo apt), SKIP_SMOKE=1, STOP_AFTER=<step> (testing).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
DRIVE_URL=${DRIVE_URL:-https://drive.google.com/drive/folders/1VcPLFqR8wTt_ihxWOjpvSQyDZTRixZso}
TAG=${TAG:-b5}
FEAT=data/feat_$TAG
unset BER_DATA BER_OUT DATA_DIR               # never inherit smoke-test paths
export PYTHONUNBUFFERED=1
mkdir -p logs/done
T0=$SECONDS

N_STEPS=23; [ -z "${SKIP_SMOKE:-}" ] || N_STEPS=22
K=0
run() {   # run <step> <command...>: skip if done, output live + logs/<step>.log, stop the chain on failure
  local name=$1; shift
  K=$((K + 1))
  if [ -f "logs/done/$name" ]; then echo "=== [$K/$N_STEPS] skip $name (done)"; return 0; fi
  echo; echo "=== [$K/$N_STEPS] $name  ($(( (SECONDS - T0) / 60 )) min elapsed, $(date +%H:%M))"
  "$@" 2>&1 | tee "logs/$name.log"
  if [ "${PIPESTATUS[0]}" -eq 0 ]; then
    touch "logs/done/$name"
  else
    echo "!!! [$K/$N_STEPS] $name FAILED (full log: logs/$name.log). Fix, then re-run this script."; exit 1
  fi
  [ "${STOP_AFTER:-}" != "$name" ] || { echo "=== STOP_AFTER=$name"; exit 0; }
}

env_setup() {
  if [ -z "${SKIP_APT:-}" ]; then
    sudo apt-get update -y && sudo apt-get install -y build-essential git tmux htop unzip curl
  fi
  if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  [ -d .venv ] || uv venv --python 3.12 .venv
  source .venv/bin/activate
  uv pip install -r requirements.txt gdown
  uv pip install torch --index-url https://download.pytorch.org/whl/cpu      # xenc collect + smoke (CPU)
  uv pip install -r requirements-gpu.txt
  python -c "import polars, lightgbm, torch, transformers, gdown, tqdm; print('env ok')"
}

download() {
  mkdir -p student_resource
  gdown --folder "$DRIVE_URL" -O student_resource
  # gdown may nest the Drive folder: move dataset/ and utils/ to student_resource/ if so
  for d in dataset utils; do
    if [ ! -d "student_resource/$d" ]; then
      src=$(find student_resource -mindepth 2 -maxdepth 3 -type d -name "$d" | head -n 1)
      [ -n "$src" ] && mv "$src" "student_resource/$d"
    fi
  done
  for f in train/train_source1.tsv train/train_source2.tsv train/train_source3.tsv train/train_ground_truth.tsv \
           test/test_source1.tsv test/test_source2.tsv test/test_source3.tsv; do
    [ -s "student_resource/dataset/$f" ] || { echo "missing student_resource/dataset/$f (Drive quota? download by hand)"; return 1; }
  done
  [ -s student_resource/utils/validate_submission.py ] || { echo "missing validate_submission.py"; return 1; }
  du -sh student_resource/dataset
}

validate() {
  python student_resource/utils/validate_submission.py --test-dir student_resource/dataset/test --check-ids \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv
}

bundle() {
  python scripts/export_pairs_text.py --tag "$TAG" --feat_dir "$FEAT"
  tar -cf data/xenc_for_pod.tar data/xenc/train.parquet data/xenc/score_train.parquet \
      data/xenc/score_test.parquet data/xenc/train_feats.parquet
  ls -la data/xenc_for_pod.tar
}

export PATH="$HOME/.local/bin:$PATH"          # uv's default install dir
run env        env_setup
source .venv/bin/activate
run download   download
run parquet    python scripts/tsv_to_parquet.py
[ -n "${SKIP_SMOKE:-}" ] || run smoke bash scripts/smoke.sh
run normalize  python src/normalize.py
for S in train test; do
  for R in word namechar addr; do
    run "block_${S}_$R" python src/block.py --split "$S" --retriever "$R"
  done
  run "block_${S}_numaddr" python src/block_numaddr.py --split "$S"
  run "union_$S"           python src/union.py --split "$S"
  run "equiv_$S"           python src/mine_equiv.py --split "$S"
done
run features_train python src/features.py --split train --out "$FEAT/train.parquet"
run features_test  python src/features.py --split test --out "$FEAT/test.parquet"
run "train_$TAG"   python src/train.py --tag "$TAG" --feat_dir "$FEAT"
run "decode_$TAG"  python src/decode.py --tag "$TAG"
run validate       validate
run bundle         bundle
echo "=== ALL DONE in $(( (SECONDS - T0) / 60 )) min"
echo "    submission: output/   (holdout score: grep -E 'holdout|BEST' logs/decode_$TAG.log)"
echo "    GPU pod:    copy data/xenc_for_pod.tar to the pod repo root, tar -xf it, then README 'GPU pod'"
