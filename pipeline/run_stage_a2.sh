#!/bin/bash
# STAGE A, fault-tolerant. Drop-in replacement for run_stage_a.sh: it skips every
# run whose result JSON already exists, so it can take over mid-pipeline.
#
# FAILURE POLICY (this is the point of the rewrite):
#   * A single run that crashes is logged to failed_runs.txt and the queue MOVES ON.
#     One bad run must never cost the other 25.
#   * A2 depends on A1 (it needs A1's learning rate). If A1's selection fails, A2 is
#     SKIPPED -- but A3 still runs, because A3 is a DiverseVul sweep and is entirely
#     independent of anything Devign.
#   * Only a failure with no independent work left behind it ends the script.
set -u
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"   # the original runs used the mirror https://hf-mirror.com
export HF_HOME="${HF_HOME:-$CFA_DATA_DIR/hf}"
D="${CFA_DATA_DIR:?set CFA_DATA_DIR, see README}"; PY="${CFA_PYTHON:-python}"; VD="${CFA_CODE_DIR:-$(cd "$(dirname "$0")" && pwd)}"
LRS="1e-5 2e-5 3e-5 5e-5"
FAILED=$D/failed_runs.txt

DEV=$D/data/devign/devign_all.jsonl
DEV_FAM=$D/devign_audit_families_theta0.7.json
DEV_FD=$D/devign_audit_splitfile_theta0.7.json
DEV_RC=$D/devign_audit_splitfile_randomctrl_theta0.7.json
DV=$D/data/diversevul/diversevul.jsonl
DV_FAM=$D/diversevul_audit_families_theta0.7.json
DV_FD=$D/diversevul_audit_splitfile_theta0.7.json
DV_RC=$D/diversevul_audit_splitfile_randomctrl_theta0.7.json
mkdir -p $D/sweep_devign $D/sweep_diversevul $D/rq2_devign_final

# run_one <outdir> <tag> <args...>  -- never propagates a failure to the caller.
run_one() {
  local out=$1 tag=$2; shift 2
  if [ -f "$out/$tag.json" ]; then echo "SKIP $tag (done)"; return 0; fi
  echo "--- $tag  $(date +%m-%d\ %H:%M:%S) ---"
  $PY $VD/finetune_vd.py "$@" --out "$out" 2>&1 | grep -viE "warn|it/s\]$"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "!!! FAILED $tag (exit $rc) -- continuing with the next run"
    echo "$(date +%F\ %T) $tag exit=$rc" >> "$FAILED"
    return 1
  fi
  [ -f "$out/$tag.json" ] || { echo "!!! $tag exited 0 but wrote no JSON"; \
    echo "$(date +%F\ %T) $tag no-json" >> "$FAILED"; return 1; }
  return 0
}
arm_split() { case $1 in published) echo "";; random_control) echo "--splitfile $2";;
                        family_disjoint) echo "--splitfile $3";; esac; }

echo "############ A1  Devign LR sweep  $(date +%m-%d\ %H:%M:%S) ############"
for LR in $LRS; do
  run_one "$D/sweep_devign/lr${LR}" codebert-base_published_seed1 \
    --data $DEV --model microsoft/codebert-base --arm published --id-field id \
    --famfile $DEV_FAM --seed 1 --lr $LR --class-weight balanced \
    --epochs 15 --patience 2 --batch-size 16 || true
done
A1_OK=1
$PY $VD/pick_lr.py "$D/sweep_devign" "$D/lr_devign.txt" || A1_OK=0

if [ "$A1_OK" = "1" ] && [ -s $D/lr_devign.txt ]; then
  LR_DEV=$(cat $D/lr_devign.txt)
  echo "############ A2  Devign final  lr=$LR_DEV  $(date +%m-%d\ %H:%M:%S) ############"
  for MODEL in microsoft/codebert-base microsoft/graphcodebert-base; do
    for SEED in 1 2 3; do
      for ARM in published random_control family_disjoint; do
        run_one "$D/rq2_devign_final" "$(basename $MODEL)_${ARM}_seed${SEED}" \
          --data $DEV --model $MODEL --arm $ARM --id-field id --famfile $DEV_FAM \
          $(arm_split $ARM $DEV_RC $DEV_FD) --seed $SEED --lr $LR_DEV \
          --class-weight balanced --epochs 15 --patience 2 --batch-size 16 || true
      done
    done
  done
else
  echo "!!! A1 selection failed -- SKIPPING A2 (it needs A1's lr)."
  echo "!!! Proceeding to A3, which is independent of every Devign result."
fi

echo "############ A3  DiverseVul LR sweep  $(date +%m-%d\ %H:%M:%S) ############"
# 60k train subsample keeps each run ~35 min instead of ~2.5 h. LIMITATION for the
# paper: fewer optimiser steps, so this is a proxy for the full-data LR optimum.
for LR in $LRS; do
  run_one "$D/sweep_diversevul/lr${LR}" codebert-base_published_seed1 \
    --data $DV --model microsoft/codebert-base --arm published --id-field idx \
    --famfile $DV_FAM --seed 1 --lr $LR --class-weight balanced \
    --epochs 8 --patience 2 --limit-train 60000 --batch-size 16 || true
done
$PY $VD/pick_lr.py "$D/sweep_diversevul" "$D/lr_diversevul.txt" || \
  echo "!!! DiverseVul LR selection failed -- Stage B will need a manual lr"

touch $D/.stage_a_done
echo "############ STAGE A COMPLETE  $(date +%m-%d\ %H:%M:%S) ############"
[ -s "$FAILED" ] && { echo "runs that failed:"; cat "$FAILED"; } || echo "no failed runs"
echo "Devign lr     = $(cat $D/lr_devign.txt 2>/dev/null || echo UNSET)"
echo "DiverseVul lr = $(cat $D/lr_diversevul.txt 2>/dev/null || echo UNSET)"
echo "NEXT: Stage B (DiverseVul final, 9 runs, ~22 GPU-h, ~Y44) -- GATED, needs approval."
