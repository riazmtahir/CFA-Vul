#!/bin/bash
# PHASE 3 -- commit-disjoint arm, RESEARCH_PLAN §13.6 (pre-registered H3,
# analysis fixed in analyze_phase3.py before any run). GATED on
# $CFA_DATA_DIR/.phase3_approved; started by hand, not by resume.sh:
#   touch $CFA_DATA_DIR/.phase3_approved
#   setsid nohup bash $CFA_CODE_DIR/run_phase3.sh > $CFA_DATA_DIR/phase3.log 2>&1 &
#   bash $CFA_CODE_DIR/resume.sh          # starts the watchdog (关机 ~15 min after the end)
#
# 3 CodeBERT runs on DiverseVul, protocol frozen to Stage B / Phase 2 (class-weighted
# CE, lr from lr_diversevul.txt, <=15 epochs, patience 2, batch 16), seeds 1-3,
# saving predictions and best weights. Re-running skips finished runs; an interrupted
# run resumes from its checkpoint. Output: $CFA_DATA_DIR/rq2_diversevul_phase3/.
set -u
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"   # the original runs used the mirror https://hf-mirror.com
export HF_HOME="${HF_HOME:-$CFA_DATA_DIR/hf}"
D="${CFA_DATA_DIR:?set CFA_DATA_DIR, see README}"; PY="${CFA_PYTHON:-python}"; VD="${CFA_CODE_DIR:-$(cd "$(dirname "$0")" && pwd)}"
FAILED=$D/failed_runs.txt
[ -f "$D/.phase3_approved" ] || { echo "!!! not approved: touch $D/.phase3_approved"; exit 1; }

DV=$D/data/diversevul/diversevul.jsonl
DV_FAM=$D/diversevul_audit_families_theta0.7.json
DV_CD=$D/diversevul_audit_splitfile_commitdisjoint_seed42.json
OUT=$D/rq2_diversevul_phase3
mkdir -p "$OUT"
for f in "$DV" "$DV_FAM" "$DV_CD" "$D/lr_diversevul.txt"; do
  [ -s "$f" ] || { echo "!!! missing input: $f -- run run_commit_leak.sh first"; exit 1; }
done
LR=$(cat "$D/lr_diversevul.txt")

if [ ! -f "$OUT/provenance.txt" ]; then
  {
    echo "phase 3 started $(date '+%F %T %Z')"
    echo "code: $(git -C "$VD" log --oneline -1 2>/dev/null)"
    echo "gpu:  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)"
    echo "lr:   $LR"
    md5sum "$DV_FAM" "$DV_CD" "$DV"
  } > "$OUT/provenance.txt"
fi
echo "############ PHASE 3  commit_disjoint  lr=$LR  $(date +%m-%d\ %H:%M:%S) ############"

for SEED in 1 2 3; do
  TAG="codebert-base_commit_disjoint_seed${SEED}"
  if [ -f "$OUT/$TAG.json" ]; then echo "SKIP $TAG (done)"; continue; fi
  echo "--- $TAG  $(date +%m-%d\ %H:%M:%S) ---"
  $PY $VD/finetune_vd.py --data "$DV" --model microsoft/codebert-base \
    --arm commit_disjoint --id-field idx --famfile "$DV_FAM" --splitfile "$DV_CD" \
    --seed "$SEED" --lr "$LR" --class-weight balanced --epochs 15 --patience 2 \
    --batch-size 16 --save-preds --save-best --out "$OUT" 2>&1 \
    | grep --line-buffered -viE "warn|it/s\]$"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ] || [ ! -f "$OUT/$TAG.json" ]; then
    echo "!!! FAILED $TAG (exit $rc) -- continuing"
    echo "$(date +%F\ %T) phase3 $TAG exit=$rc" >> "$FAILED"
  fi
done

touch $D/.phase3_done
echo "############ PHASE 3 RUNS COMPLETE  $(date +%m-%d\ %H:%M:%S) ############"
$PY $VD/analyze_phase3.py --disk "$D"
