#!/usr/bin/env bash
# Fresh CPU VM (56+ vCPU) -> uv env, Drive dataset, fast B5 rebuild + the no-log-odds model for France routing.
# Resumable: each finished step leaves logs/done/<step>; a re-run skips those. Step logs: logs/<step>.log.
#
#   git clone <repo> && cd amazon-ml-challenge-2026 && git checkout fast-france
#   tmux new -s run
#   bash scripts/setup_cpu_vm.sh
#
# Fast mode (4-hour deadline): no smoke test (RUN_SMOKE=1 to add it), train features on 50% of train S1s
# (SAMPLE, context/twin features still see every candidate), b5 and b5nolo trained at the same time.
# Ends with data/pred/{train,test}_{b5,b5nolo}.parquet, data/models/lgb_{b5,b5nolo}.txt, output/ (b5 alone).
# Then the France steps by hand (README "France routing").
# Env: SAMPLE (default 0.5), DRIVE_URL, SKIP_APT=1, RUN_SMOKE=1, STOP_AFTER=<step> (testing).
set -euo pipefail
cd "$(dirname "$0")/.."
DRIVE_URL=${DRIVE_URL:-https://drive.google.com/drive/folders/1VcPLFqR8wTt_ihxWOjpvSQyDZTRixZso}
SAMPLE=${SAMPLE:-0.5}
FEAT=data/feat_b5
NOLO="add_lo_,drop_lo_,aadd_lo_,adrop_lo_,n_added_unknown,n_dropped_unknown,aadd_unknown,adrop_unknown"
unset BER_DATA BER_OUT DATA_DIR               # never inherit smoke-test paths
export PYTHONUNBUFFERED=1
mkdir -p logs/done
T0=$SECONDS
N_STEPS=20; [ -z "${RUN_SMOKE:-}" ] || N_STEPS=21
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
  python -c "import polars, lightgbm, rapidfuzz, sparse_dot_topn, gdown, tqdm; print('env ok')"
}

download() {
  mkdir -p student_resource
  gdown --folder "$DRIVE_URL" -O student_resource
  for d in dataset utils; do                    # gdown may nest the Drive folder
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

train_both() {   # full model and the no-label-log-odds model at the same time, half the cores each
  local th=$(( $(nproc) / 2 ))
  python src/train.py --tag b5 --feat_dir "$FEAT" --threads "$th" > logs/train_b5.log 2>&1 &
  local p1=$!
  python src/train.py --tag b5nolo --feat_dir "$FEAT" --threads "$th" --drop "$NOLO" > logs/train_b5nolo.log 2>&1 &
  local p2=$!
  local rc1=0 rc2=0
  wait $p1 || rc1=$?
  wait $p2 || rc2=$?
  grep -E "features;|Early stopping|best iteration|done in" logs/train_b5.log logs/train_b5nolo.log || true
  [ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] || { echo "train b5 rc=$rc1 (logs/train_b5.log), b5nolo rc=$rc2 (logs/train_b5nolo.log)"; return 1; }
}

export PATH="$HOME/.local/bin:$PATH"          # uv's default install dir
run env        env_setup
source .venv/bin/activate
run download   download
run parquet    python scripts/tsv_to_parquet.py
[ -z "${RUN_SMOKE:-}" ] || run smoke bash scripts/smoke.sh
run normalize  python src/normalize.py
retrievers() {   # one split's chain: 4 retrievers, union, equivalences
  local S=$1
  for R in word namechar addr; do
    run "block_${S}_$R" python src/block.py --split "$S" --retriever "$R"
  done
  run "block_${S}_numaddr" python src/block_numaddr.py --split "$S"
  run "union_$S"           python src/union.py --split "$S"
  run "equiv_$S"           python src/mine_equiv.py --split "$S"
}
# train and test chains at the same time: most retriever time is single-threaded pandas, cores sit idle
( retrievers train ) & P_TR=$!
( retrievers test ) & P_TE=$!
RC=0; wait $P_TR || RC=1; wait $P_TE || RC=1
[ $RC -eq 0 ] || { echo "!!! a retriever chain failed: see the FAILED line above"; exit 1; }
K=$((K + 12))
run features_train python src/features.py --split train --sample "$SAMPLE" --out "$FEAT/train.parquet"
run features_test  python src/features.py --split test --out "$FEAT/test.parquet"
run train_both     train_both
run decode_b5      python src/decode.py --tag b5
echo "=== ALL DONE in $(( (SECONDS - T0) / 60 )) min. Next: README 'France routing'."
