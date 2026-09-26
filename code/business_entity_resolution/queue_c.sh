#!/usr/bin/env bash
# Unattended queue on the VM: test TSVs for E D F B, then strategy C improvements, then stop the VM.
# Every job logs to ~/q_<job>.log; progress lines go to ~/queue.log. A failed job is logged and skipped.
source ~/venv/bin/activate
cd ~/ber/code/business_entity_resolution/src
Q=~/queue.log
log() { echo "[$(date -u +%H:%M) UTC] $*" >> $Q; }
run() { local name=$1; shift; log "START $name"; if bash -c "$*" > ~/q_$name.log 2>&1; then log "OK    $name"; else log "FAIL  $name (see ~/q_$name.log)"; fi; }

log "QUEUE START"
# 1. test TSVs for the remaining strategies (models already trained, fast mode)
for V in E D F B; do
  run test_$V "BER_TAG=_fast BER_OUT=~/ber/output_$V python3 model.py test $V && BER_TAG=_fast BER_OUT=~/ber/output_$V python3 submit.py $V"
done
# 2. wider decoder grid on C's saved probabilities -> output_C_dec
run redecode_C "python3 posthoc.py redecode C_fast && BER_TAG=_fast_dec BER_OUT=~/ber/output_C_dec python3 submit.py C"
# 3. full-size stage 1 + stage-2 tuning grid for C -> models/C_tune*, best -> output_C_tuned
run tune_C "BER_MAX_ROWS=16000000 BER_LR=0.1 BER_ROUNDS=1500 BER_REUSE_STAGE1=1 python3 model.py tune C"
BI=$(cat ~/ber/work/models/C_tune_best.txt 2>/dev/null)
MEMBERS=C_fast,G_fast,E_fast
if [ -n "$BI" ]; then
  run test_C_tuned "BER_TAG=_tune$BI python3 model.py test C && BER_TAG=_tune$BI BER_OUT=~/ber/output_C_tuned python3 submit.py C"
  run export_C_tuned "python3 export_probs.py C_tune$BI ~/ber/ensemble_C_tuned"
  [ -f ~/ber/work/test/pred_C_tune$BI.npz ] && MEMBERS=$MEMBERS,C_tune$BI
fi
# 4. ensemble of our own models (equal average, decoder picked out-of-fold) -> output_ENS
run ensemble "python3 posthoc.py ensemble ENS_mix $MEMBERS && BER_TAG=_mix BER_OUT=~/ber/output_ENS python3 submit.py ENS"
log "QUEUE DONE (members: $MEMBERS, best tune: ${BI:-none}) -- stopping the VM"
sudo shutdown -h now
