#!/bin/bash
# STAGE A -- fix the instrument, then re-measure Devign; sweep DiverseVul's LR.
#
# Three defects are being closed here, all of which biased the Gate-1 null:
#   1. plain CrossEntropyLoss on an imbalanced corpus  -> --class-weight balanced
#   2. no hyperparameter search at all                 -> LR swept on VALIDATION
#   3. fixed 3 epochs with validation PR-AUC still rising -> --epochs 15 --patience 2
#
# The LR is swept ONCE per corpus on the published arm's validation set and then
# FROZEN across every arm and seed. Per-arm tuning would confound RQ2.
#
# Stage A deliberately stops before the expensive DiverseVul final runs so the
# sweep can be read before ~22 GPU-hours are committed.
set -u
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"   # the original runs used the mirror https://hf-mirror.com
export HF_HOME="${HF_HOME:-$CFA_DATA_DIR/hf}"
D="${CFA_DATA_DIR:?set CFA_DATA_DIR, see README}"; PY="${CFA_PYTHON:-python}"; VD="${CFA_CODE_DIR:-$(cd "$(dirname "$0")" && pwd)}"
LRS="1e-5 2e-5 3e-5 5e-5"

DEV=$D/data/devign/devign_all.jsonl
DEV_FAM=$D/devign_audit_families_theta0.7.json
DEV_FD=$D/devign_audit_splitfile_theta0.7.json
DEV_RC=$D/devign_audit_splitfile_randomctrl_theta0.7.json

DV=$D/data/diversevul/diversevul.jsonl
DV_FAM=$D/diversevul_audit_families_theta0.7.json
DV_FD=$D/diversevul_audit_splitfile_theta0.7.json
DV_RC=$D/diversevul_audit_splitfile_randomctrl_theta0.7.json

mkdir -p $D/sweep_devign $D/sweep_diversevul $D/rq2_devign_final

echo "############ A1  Devign LR sweep   $(date +%m-%d\ %H:%M:%S) ############"
for LR in $LRS; do
  T=codebert-base_published_seed1
  [ -f "$D/sweep_devign/lr${LR}/${T}.json" ] && { echo "SKIP lr=$LR"; continue; }
  echo "--- devign sweep lr=$LR $(date +%H:%M:%S) ---"
  $PY $VD/finetune_vd.py --data $DEV --model microsoft/codebert-base \
      --arm published --id-field id --famfile $DEV_FAM --seed 1 \
      --lr $LR --class-weight balanced --epochs 15 --patience 2 \
      --batch-size 16 --out $D/sweep_devign/lr${LR} 2>&1 | grep -viE "warn|it/s\]$"
done
echo "--- Devign LR selection (validation PR-AUC) ---"
$PY $VD/pick_lr.py "$D/sweep_devign" "$D/lr_devign.txt" || exit 1
LR_DEV=$(cat $D/lr_devign.txt)

echo "############ A2  Devign final  lr=$LR_DEV   $(date +%m-%d\ %H:%M:%S) ############"
# Fresh output dir: the 6 runs in rq2_devign_converged used UNWEIGHTED CE and an
# unswept LR, so they are a different protocol and must not be mixed in.
for MODEL in microsoft/codebert-base microsoft/graphcodebert-base; do
  for SEED in 1 2 3; do
    for ARM in published random_control family_disjoint; do
      T="$(basename $MODEL)_${ARM}_seed${SEED}"
      [ -f "$D/rq2_devign_final/${T}.json" ] && { echo "SKIP $T"; continue; }
      case $ARM in
        published)       SF="" ;;
        random_control)  SF="--splitfile $DEV_RC" ;;
        family_disjoint) SF="--splitfile $DEV_FD" ;;
      esac
      echo "--- $T $(date +%H:%M:%S) ---"
      $PY $VD/finetune_vd.py --data $DEV --model $MODEL --arm $ARM \
          --id-field id --famfile $DEV_FAM $SF --seed $SEED \
          --lr $LR_DEV --class-weight balanced --epochs 15 --patience 2 \
          --batch-size 16 --out $D/rq2_devign_final 2>&1 | grep -viE "warn|it/s\]$"
    done
  done
done

echo "############ A3  DiverseVul LR sweep   $(date +%m-%d\ %H:%M:%S) ############"
# Subsampled to 60k train rows to keep the sweep affordable (~35 min/run rather
# than ~2.5 h). LIMITATION, to be stated in the paper: a smaller train set means
# fewer optimiser steps, so the selected LR is a proxy for the full-data optimum,
# not a measurement of it.
for LR in $LRS; do
  T=codebert-base_published_seed1
  [ -f "$D/sweep_diversevul/lr${LR}/${T}.json" ] && { echo "SKIP lr=$LR"; continue; }
  echo "--- diversevul sweep lr=$LR $(date +%H:%M:%S) ---"
  $PY $VD/finetune_vd.py --data $DV --model microsoft/codebert-base \
      --arm published --id-field idx --famfile $DV_FAM --seed 1 \
      --lr $LR --class-weight balanced --epochs 8 --patience 2 \
      --limit-train 60000 --batch-size 16 \
      --out $D/sweep_diversevul/lr${LR} 2>&1 | grep -viE "warn|it/s\]$"
done
echo "--- DiverseVul LR selection (validation PR-AUC) ---"
$PY $VD/pick_lr.py "$D/sweep_diversevul" "$D/lr_diversevul.txt" || exit 1

echo "############ STAGE A COMPLETE  $(date +%m-%d\ %H:%M:%S) ############"
echo "Devign lr     = $(cat $D/lr_devign.txt)"
echo "DiverseVul lr = $(cat $D/lr_diversevul.txt)"
echo "NEXT: run_stage_b.sh (DiverseVul final, 9 runs, ~22 GPU-h) -- not started on purpose."
