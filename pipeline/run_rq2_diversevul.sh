#!/bin/bash
# RQ2 on DiverseVul. CodeBERT only x 3 arms x 3 seeds = 9 runs, ~74 min each (~11 h, ~Y22).
# CodeBERT-only is deliberate: DiverseVul answers the cross-project/leakage question,
# not an architecture comparison. Each run writes its own JSON and finetune_vd.py
# resumes mid-run from per-epoch checkpoints, so this survives interruption.
set -u
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"   # the original runs used the mirror https://hf-mirror.com
export HF_HOME="${HF_HOME:-$CFA_DATA_DIR/hf}"
D="${CFA_DATA_DIR:?set CFA_DATA_DIR, see README}"
DATA=$D/data/diversevul/diversevul.jsonl
FAM=$D/diversevul_audit_families_theta0.7.json
FD=$D/diversevul_audit_splitfile_theta0.7.json
RC=$D/diversevul_audit_splitfile_randomctrl_theta0.7.json
OUT=$D/rq2_diversevul
PY="${CFA_PYTHON:-python}"
mkdir -p $OUT
for SEED in 1 2 3; do
  for ARM in published random_control family_disjoint; do
    TAG="codebert-base_${ARM}_seed${SEED}"
    [ -f "$OUT/${TAG}.json" ] && { echo "SKIP $TAG (done)"; continue; }
    case $ARM in
      published)       SF="" ;;
      random_control)  SF="--splitfile $RC" ;;
      family_disjoint) SF="--splitfile $FD" ;;
    esac
    echo "=========== $TAG  $(date +%m-%d\ %H:%M:%S) ==========="
    $PY $CFA_CODE_DIR/finetune_vd.py --data $DATA --model microsoft/codebert-base \
        --arm $ARM --id-field idx --famfile $FAM $SF --seed $SEED --epochs 3 \
        --batch-size 16 --out $OUT 2>&1 | grep -viE "warn|it/s\]$"
  done
done
echo "=========== RQ2 DIVERSEVUL COMPLETE $(date) ==========="
