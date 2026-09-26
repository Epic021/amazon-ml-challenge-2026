#!/usr/bin/env bash
# Unattended GPU session for the neural pair scorers (src/xenc.py) on a RunPod A100 80 GB.
# GPU budget: $10 total at $1.59/h (~6.2 h), no refill. Every step has a time cap, each session has a
# hard cap, and the pod stops itself at the end, even on failure.
#
# On the pod, after cloning the repo and copying the 4 files from the CPU VM's data/xenc/ into data/xenc/
# (train, score_train, score_test, train_feats .parquet; made by scripts/export_pairs_text.py):
#   bash scripts/gpu_session.sh setup                                    (once, ~5 min, checks everything)
#   nohup bash scripts/gpu_session.sh auto > /dev/null 2>&1 &            (then log out; tail -f logs/gpu_auto.log)
#
# Sessions (minutes are caps):
#   auto  s1, then s2 only if the transfer gate (run on the pod during s1) passed          ~4.5 h (~$7)
#   s1    probe both models · US-only mDeBERTa + India scoring + transfer gate · mDeBERTa on the training
#         set · mDeBERTa scores the full shortlist (folds 5-9 + test, prio order)         ~2.0 h
#   s2    Qwen3-4B LoRA on the hardest pairs, sized to fit 65 min · scores the llm-flagged shortlist
#                                                                                         ~2.5 h
#   r     round 2: continue Qwen3-4B on data/xenc/pseudo.parquet · re-score France's llm pairs  ~0.8 h
# Outputs stay on /workspace (kept when a RunPod pod stops): data/xenc/m_*/, data/xenc/p_*/, logs/
# Env: STOP_CMD (default: runpodctl stop pod $RUNPOD_POD_ID), SYNC_CMD (optional: copy results off first),
#      HARD_MIN (override the session cap), XENC_EXTRA (e.g. --tiny for CPU tests), X (override the runner).
set -uo pipefail
S=${1:?"session: setup | auto | s1 | s2 | r"}
cd "$(dirname "$0")/.."
mkdir -p logs data/xenc
exec > >(tee -a "logs/gpu_$S.log") 2>&1
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=true
X=${X:-python src/xenc.py}
E=${XENC_EXTRA:-}
D=data/xenc
echo "=== session $S start $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

if [ "$S" = setup ]; then
  ok=1
  pip install -q -r requirements-gpu.txt || ok=0
  python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, torch.cuda.get_device_name())" || ok=0
  for f in train score_train score_test train_feats; do
    [ -f "$D/$f.parquet" ] || { echo "!!! missing $D/$f.parquet: copy it from the CPU VM's data/xenc/"; ok=0; }
  done
  hf download microsoft/mdeberta-v3-base >/dev/null && hf download Qwen/Qwen3-4B-Base >/dev/null && echo "models cached" || ok=0
  command -v runpodctl >/dev/null || [ -n "${STOP_CMD:-}" ] || echo "!!! runpodctl not found and STOP_CMD unset: the pod will NOT stop itself"
  [ $ok = 1 ] && echo "=== SETUP OK: nohup bash scripts/gpu_session.sh auto > /dev/null 2>&1 &" || echo "=== SETUP FAILED (see above)"
  exit 0
fi

if [ -z "${STOP_CMD:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then STOP_CMD="runpodctl stop pod $RUNPOD_POD_ID"; fi
finish() {
  echo "=== session $S end $(date) (${SECONDS}s)"
  [ -n "${SYNC_CMD:-}" ] && { eval "$SYNC_CMD" || echo "SYNC FAILED"; }
  if [ -n "${STOP_CMD:-}" ]; then eval "$STOP_CMD"; else echo "!!! STOP_CMD not set: STOP THE INSTANCE BY HAND !!!"; fi
}
trap finish EXIT
STEP_PID=
trap '[ -n "$STEP_PID" ] && kill -TERM "$STEP_PID" 2>/dev/null; exit 143' TERM INT   # kill the step (not tee), then finish()
case $S in auto) HARD=290 ;; s1) HARD=130 ;; s2) HARD=160 ;; r) HARD=55 ;; *) echo "unknown session $S"; exit 2 ;; esac
HARD=${HARD_MIN:-$HARD}
( sleep "$(( HARD * 60 ))"; echo "!!! hard cap ${HARD} min: stopping"; kill -TERM $$ ) &
WATCHDOG=$!

# steps run in the background + wait, so the hard-cap signal interrupts them at once (a foreground child
# would delay the trap until it finished)
step() {
  echo; echo "=== [$(( SECONDS / 60 )) min] $1"; shift
  timeout "$1" "${@:2}" &
  STEP_PID=$!
  wait $STEP_PID
  local rc=$?
  [ $rc -eq 0 ] || echo "step exit code $rc"
  return 0
}

s1() {
  step "probe mdeberta" 8m  $X probe --model mdeberta --data $D/train.parquet --plan_train 1500000 $E
  step "probe qwen3-4b" 12m $X probe --model qwen3-4b --data $D/train.parquet --subset llm --plan_train 350000 --plan_score 800000 $E
  step "train mdeberta US-only (transfer gate)" 12m \
       $X train --model mdeberta --data $D/train.parquet --country US --out $D/m_mdeb_us --fit_minutes 8 --max_minutes 10 $E
  step "score India hardest pairs (transfer gate)" 6m \
       $X score --model_dir $D/m_mdeb_us --data $D/train.parquet --country India --subset llm --out $D/p_mdeb_us --max_minutes 5
  # the gate is CPU work: run it alongside the GPU steps; its verdict is read after s1
  rm -f logs/gate.txt
  ( timeout 20m python scripts/transfer_check.py --threads 8 > logs/gate.txt 2>&1; echo; echo "=== transfer gate:"; cat logs/gate.txt ) &
  GATE_PID=$!
  step "train mdeberta" 45m $X train --model mdeberta --data $D/train.parquet --out $D/m_mdeb --fit_minutes 38 --max_minutes 42 $E
  step "score shortlist" 50m \
       $X score --model_dir $D/m_mdeb --data $D/score_train.parquet,$D/score_test.parquet --out $D/p_mdeb --max_minutes 47
  wait $GATE_PID 2>/dev/null
}
s2() {
  step "train qwen3-4b" 80m $X train --model qwen3-4b --data $D/train.parquet --subset llm --out $D/m_qwen --fit_minutes 65 --max_minutes 72 $E
  step "score llm shortlist" 75m \
       $X score --model_dir $D/m_qwen --data $D/score_train.parquet,$D/score_test.parquet --subset llm --out $D/p_qwen --max_minutes 70
}
r() {
  step "round 2 train (pseudo-labels)" 25m \
       $X train --model qwen3-4b --init $D/m_qwen --data $D/pseudo.parquet --out $D/m_qwen_r2 --lr 5e-5 --fit_minutes 18 --max_minutes 20 $E
  step "round 2 score France" 30m \
       $X score --model_dir $D/m_qwen_r2 --data $D/score_test.parquet --country France --subset llm --out $D/p_qwen_r2 --max_minutes 27
}

case $S in
  s1) s1 ;;
  s2) s2 ;;
  r) r ;;
  auto)
    s1
    if grep -q "GATE: PASS" logs/gate.txt 2>/dev/null; then
      echo "=== gate passed: LLM session"; s2
    else
      echo "=== gate did not pass (logs/gate.txt): no LLM session, money kept"
    fi ;;
esac
kill $WATCHDOG 2>/dev/null
