#!/bin/bash
# PHASE 4 -- commit_hrfamily_disjoint arm, RESEARCH_PLAN §13.7 (pre-registered H4,
# analysis fixed in analyze_phase4.py before any run). GATED on
# $CFA_DATA_DIR/.phase4_approved AND on a passing pre-check (h4_precheck.json verdict
# PASS from run_h4_precheck.sh); started by hand, not by resume.sh:
#   touch $CFA_DATA_DIR/.phase4_approved
#   setsid nohup bash $CFA_CODE_DIR/run_phase4.sh > $CFA_DATA_DIR/phase4.log 2>&1 &
#   bash $CFA_CODE_DIR/resume.sh          # starts the watchdog (关机 ~15 min after the end)
#
# 3 CodeBERT runs on DiverseVul, protocol frozen to Phase 3 (class-weighted CE, lr from
# lr_diversevul.txt, <=15 epochs, patience 2, batch 16), seeds 1-3, saving predictions and
# best weights. Re-running skips finished runs; an interrupted run resumes from its
# checkpoint. Output: $CFA_DATA_DIR/rq2_diversevul_phase4/.
set -u
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"   # the original runs used the mirror https://hf-mirror.com
export HF_HOME="${HF_HOME:-$CFA_DATA_DIR/hf}"
D="${CFA_DATA_DIR:?set CFA_DATA_DIR, see README}"; PY="${CFA_PYTHON:-python}"; VD="${CFA_CODE_DIR:-$(cd "$(dirname "$0")" && pwd)}"
FAILED=$D/failed_runs.txt
[ -f "$D/.phase4_approved" ] || { echo "!!! not approved: touch $D/.phase4_approved"; exit 1; }
grep -q '"verdict": "PASS"' "$D/h4_precheck.json" 2>/dev/null \
  || { echo "!!! H4 pre-check has not passed ($D/h4_precheck.json) -- run run_h4_precheck.sh; H4 is not run"; exit 1; }
grep -q '"design": "H4"' "$D/h4_design.json" 2>/dev/null \
  || { echo "!!! §13.8 precedence does not select H4 ($D/h4_design.json)"; exit 1; }

DV=$D/data/diversevul/diversevul.jsonl
DV_FAM=$D/diversevul_audit_families_theta0.7.json
DV_HF=$D/diversevul_audit_splitfile_commithrfamily_seed42.json
HR_FAM=$D/audit_highrecall/diversevul_hr_families_theta0.7.json
OUT=$D/rq2_diversevul_phase4
mkdir -p "$OUT"
for f in "$DV" "$DV_FAM" "$DV_HF" "$HR_FAM" "$D/lr_diversevul.txt"; do
  [ -s "$f" ] || { echo "!!! missing input: $f -- run run_h4_precheck.sh first"; exit 1; }
done
LR=$(cat "$D/lr_diversevul.txt")

if [ ! -f "$OUT/provenance.txt" ]; then
  {
    echo "phase 4 started $(date '+%F %T %Z')"
    echo "code: $(git -C "$VD" log --oneline -1 2>/dev/null)"
    echo "gpu:  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)"
    echo "lr:   $LR"
    md5sum "$DV_FAM" "$DV_HF" "$HR_FAM" "$DV" "$D/h4_precheck.json"
  } > "$OUT/provenance.txt"
fi
echo "############ PHASE 4  commit_hrfamily_disjoint  lr=$LR  $(date +%m-%d\ %H:%M:%S) ############"

for SEED in 1 2 3; do
  TAG="codebert-base_commit_hrfamily_disjoint_seed${SEED}"
  if [ -f "$OUT/$TAG.json" ]; then echo "SKIP $TAG (done)"; continue; fi
  echo "--- $TAG  $(date +%m-%d\ %H:%M:%S) ---"
  $PY $VD/finetune_vd.py --data "$DV" --model microsoft/codebert-base \
    --arm commit_hrfamily_disjoint --id-field idx --famfile "$DV_FAM" --splitfile "$DV_HF" \
    --seed "$SEED" --lr "$LR" --class-weight balanced --epochs 15 --patience 2 \
    --batch-size 16 --save-preds --save-best --out "$OUT" 2>&1 \
    | grep --line-buffered -viE "warn|it/s\]$"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ] || [ ! -f "$OUT/$TAG.json" ]; then
    echo "!!! FAILED $TAG (exit $rc) -- continuing"
    echo "$(date +%F\ %T) phase4 $TAG exit=$rc" >> "$FAILED"
  fi
done

touch $D/.phase4_done
echo "############ PHASE 4 RUNS COMPLETE  $(date +%m-%d\ %H:%M:%S) ############"
$PY $VD/analyze_phase4.py --disk "$D"
