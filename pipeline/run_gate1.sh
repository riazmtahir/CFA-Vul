#!/bin/bash
# Gate 1: CodeBERT + GraphCodeBERT x {published, random_control, family_disjoint} x 3 seeds.
# Each run writes its own JSON, so an interruption costs at most the run in flight.
set -u
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"   # the original runs used the mirror https://hf-mirror.com
export HF_HOME="${HF_HOME:-$CFA_DATA_DIR/hf}"

DATA=$CFA_DATA_DIR/data/devign/devign_all.jsonl
FAM=$CFA_DATA_DIR/devign_audit_families_theta0.7.json
FD="${CFA_DATA_DIR:?set CFA_DATA_DIR, see README}"/devign_audit_splitfile_theta0.7.json
RC=$CFA_DATA_DIR/devign_audit_splitfile_randomctrl_theta0.7.json
OUT=$CFA_DATA_DIR/rq2_devign
PY="${CFA_PYTHON:-python}"

mkdir -p $OUT
for MODEL in microsoft/codebert-base microsoft/graphcodebert-base; do
  for SEED in 1 2 3; do
    for ARM in published random_control family_disjoint; do
      TAG="$(basename $MODEL)_${ARM}_seed${SEED}"
      if [ -f "$OUT/${TAG}.json" ]; then echo "SKIP $TAG (done)"; continue; fi
      case $ARM in
        published)       SF="" ;;
        random_control)  SF="--splitfile $RC" ;;
        family_disjoint) SF="--splitfile $FD" ;;
      esac
      echo "=========== $TAG  $(date +%H:%M:%S) ==========="
      $PY $CFA_CODE_DIR/finetune_vd.py --data $DATA --model $MODEL --arm $ARM \
          --id-field id --famfile $FAM $SF --seed $SEED --epochs 3 \
          --batch-size 16 --out $OUT 2>&1 | grep -viE 'warn|it/s\]$'
    done
  done
done
echo "=========== GATE 1 COMPLETE $(date +%H:%M:%S) ==========="
