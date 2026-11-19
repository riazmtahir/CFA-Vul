#!/bin/bash
# Gate 1 RE-RUN, trained to convergence.
# Every run in the fixed-3-epoch Gate 1 had validation PR-AUC still RISING at
# epoch 3 (21,854 rows / batch 16 = only 4,098 gradient steps). An undertrained
# model has had less opportunity to memorise clone families, which biases the
# leakage measurement TOWARD ZERO -- so the original null may be an artifact.
# Max 15 epochs, early stop after 2 epochs without validation improvement.
set -u
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"   # the original runs used the mirror https://hf-mirror.com
export HF_HOME="${HF_HOME:-$CFA_DATA_DIR/hf}"
D="${CFA_DATA_DIR:?set CFA_DATA_DIR, see README}"; PY="${CFA_PYTHON:-python}"
DATA=$D/data/devign/devign_all.jsonl
FAM=$D/devign_audit_families_theta0.7.json
FD=$D/devign_audit_splitfile_theta0.7.json
RC=$D/devign_audit_splitfile_randomctrl_theta0.7.json
OUT=$D/rq2_devign_converged
mkdir -p $OUT
for MODEL in microsoft/codebert-base microsoft/graphcodebert-base; do
  for SEED in 1 2 3; do
    for ARM in published random_control family_disjoint; do
      TAG="$(basename $MODEL)_${ARM}_seed${SEED}"
      [ -f "$OUT/${TAG}.json" ] && { echo "SKIP $TAG"; continue; }
      case $ARM in
        published)       SF="" ;;
        random_control)  SF="--splitfile $RC" ;;
        family_disjoint) SF="--splitfile $FD" ;;
      esac
      echo "=========== $TAG  $(date +%H:%M:%S) ==========="
      $PY $CFA_CODE_DIR/finetune_vd.py --data $DATA --model $MODEL --arm $ARM \
          --id-field id --famfile $FAM $SF --seed $SEED \
          --epochs 15 --patience 2 --batch-size 16 --out $OUT 2>&1 \
        | grep -viE "warn|it/s\]$"
    done
  done
done
echo "=========== GATE 1 CONVERGED COMPLETE $(date) ==========="
