#!/bin/bash
# STAGE B -- DiverseVul final. GATED: resume.sh will not start this unless
# $CFA_DATA_DIR/.stage_b_approved exists.
#
# CodeBERT x {published, random_control, family_disjoint} x 3 seeds = 9 runs.
# Architecture comparison is not this corpus's job; leakage is.
# Protocol frozen to Stage A: class-weighted CE, LR from the DiverseVul sweep
# (validation only), epochs 15, patience 2. Full train set -- no --limit-train.
#
# Do not write into rq2_diversevul/: those 9 JSONs used unweighted 3-epoch CE.
set -u
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"   # the original runs used the mirror https://hf-mirror.com
export HF_HOME="${HF_HOME:-$CFA_DATA_DIR/hf}"
D="${CFA_DATA_DIR:?set CFA_DATA_DIR, see README}"; PY="${CFA_PYTHON:-python}"; VD="${CFA_CODE_DIR:-$(cd "$(dirname "$0")" && pwd)}"
FAILED=$D/failed_runs.txt

DV=$D/data/diversevul/diversevul.jsonl
DV_FAM=$D/diversevul_audit_families_theta0.7.json
DV_FD=$D/diversevul_audit_splitfile_theta0.7.json
DV_RC=$D/diversevul_audit_splitfile_randomctrl_theta0.7.json
OUT=$D/rq2_diversevul_final
mkdir -p "$OUT"

if [ ! -s "$D/lr_diversevul.txt" ]; then
  echo "!!! no $D/lr_diversevul.txt -- Stage A3 must finish first"
  exit 1
fi
LR=$(cat "$D/lr_diversevul.txt")
echo "############ STAGE B  DiverseVul final  lr=$LR  $(date +%m-%d\ %H:%M:%S) ############"

run_one() {
  local tag=$1; shift
  if [ -f "$OUT/$tag.json" ]; then echo "SKIP $tag (done)"; return 0; fi
  echo "--- $tag  $(date +%m-%d\ %H:%M:%S) ---"
  $PY $VD/finetune_vd.py "$@" --out "$OUT" 2>&1 | grep --line-buffered -viE "warn|it/s\]$"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "!!! FAILED $tag (exit $rc) -- continuing"
    echo "$(date +%F\ %T) $tag exit=$rc" >> "$FAILED"
    return 1
  fi
  [ -f "$OUT/$tag.json" ] || { echo "!!! $tag no JSON"; \
    echo "$(date +%F\ %T) $tag no-json" >> "$FAILED"; return 1; }
}

arm_split() { case $1 in published) echo "";; random_control) echo "--splitfile $DV_RC";;
                        family_disjoint) echo "--splitfile $DV_FD";; esac; }

for SEED in 1 2 3; do
  for ARM in published random_control family_disjoint; do
    TAG="codebert-base_${ARM}_seed${SEED}"
    run_one "$TAG" \
      --data $DV --model microsoft/codebert-base --arm $ARM --id-field idx \
      --famfile $DV_FAM $(arm_split $ARM) --seed $SEED --lr $LR \
      --class-weight balanced --epochs 15 --patience 2 --batch-size 16 || true
  done
done

touch $D/.stage_b_done
echo "############ STAGE B COMPLETE  $(date +%m-%d\ %H:%M:%S) ############"
[ -s "$FAILED" ] && { echo "runs that failed:"; cat "$FAILED"; } || echo "no failed runs"
echo "NEXT: read Δ(random→family) on $OUT. Large → more baselines. Small → outcome C, cut RQ4."
