#!/bin/bash
# PHASE 4b -- commit_decon vs commit_sizectrl, RESEARCH_PLAN §13.8 (pre-registered H4b,
# analysis fixed in analyze_phase4b.py before any run). GATED on
# $CFA_DATA_DIR/.phase4b_approved, a PASS in h4b_precheck.json AND the §13.8 precedence
# selecting H4b in h4_design.json (H4 is run instead when its own gate passes). Started by
# hand, not by resume.sh:
#   touch $CFA_DATA_DIR/.phase4b_approved
#   setsid nohup bash $CFA_CODE_DIR/run_phase4b.sh > $CFA_DATA_DIR/phase4b.log 2>&1 &
#   bash $CFA_CODE_DIR/resume.sh          # starts the watchdog (关机 ~15 min after the end)
#
# 6 CodeBERT runs on DiverseVul (2 arms x seeds 1-3, interleaved by seed), protocol frozen to
# Phase 3 (class-weighted CE, lr from lr_diversevul.txt, <=15 epochs, patience 2, batch 16),
# saving predictions and best weights. Re-running skips finished runs; an interrupted run
# resumes from its checkpoint. Output: $CFA_DATA_DIR/rq2_diversevul_phase4b/.
set -u
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"   # the original runs used the mirror https://hf-mirror.com
export HF_HOME="${HF_HOME:-$CFA_DATA_DIR/hf}"
D="${CFA_DATA_DIR:?set CFA_DATA_DIR, see README}"; PY="${CFA_PYTHON:-python}"; VD="${CFA_CODE_DIR:-$(cd "$(dirname "$0")" && pwd)}"
FAILED=$D/failed_runs.txt
[ -f "$D/.phase4b_approved" ] || { echo "!!! not approved: touch $D/.phase4b_approved"; exit 1; }
grep -q '"verdict": "PASS"' "$D/h4b_precheck.json" 2>/dev/null \
  || { echo "!!! H4b pre-check has not passed ($D/h4b_precheck.json) -- H4b is not run"; exit 1; }
grep -q '"design": "H4b"' "$D/h4_design.json" 2>/dev/null \
  || { echo "!!! §13.8 precedence does not select H4b ($D/h4_design.json) -- run what it selects"; exit 1; }

DV=$D/data/diversevul/diversevul.jsonl
DV_FAM=$D/diversevul_audit_families_theta0.7.json
SP_DEC=$D/diversevul_audit_splitfile_commitdecon_seed42.json
SP_CTL=$D/diversevul_audit_splitfile_commitsizectrl_seed42.json
HITS=$D/direct_edges/diversevul_hr_direct_commit_disjoint_train_hits.json
OUT=$D/rq2_diversevul_phase4b
mkdir -p "$OUT"
for f in "$DV" "$DV_FAM" "$SP_DEC" "$SP_CTL" "$HITS" "$D/lr_diversevul.txt"; do
  [ -s "$f" ] || { echo "!!! missing input: $f -- run run_h4_precheck.sh first"; exit 1; }
done
LR=$(cat "$D/lr_diversevul.txt")

if [ ! -f "$OUT/provenance.txt" ]; then
  {
    echo "phase 4b started $(date '+%F %T %Z')"
    echo "code: $(git -C "$VD" log --oneline -1 2>/dev/null)"
    echo "gpu:  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)"
    echo "lr:   $LR"
    md5sum "$DV_FAM" "$SP_DEC" "$SP_CTL" "$HITS" "$DV" "$D/h4b_precheck.json" "$D/h4_design.json"
  } > "$OUT/provenance.txt"
fi
echo "############ PHASE 4b  commit_sizectrl / commit_decon  lr=$LR  $(date +%m-%d\ %H:%M:%S) ############"

for SEED in 1 2 3; do
  for ARM in commit_sizectrl commit_decon; do
    SP=$SP_CTL; [ "$ARM" = commit_decon ] && SP=$SP_DEC
    TAG="codebert-base_${ARM}_seed${SEED}"
    if [ -f "$OUT/$TAG.json" ]; then echo "SKIP $TAG (done)"; continue; fi
    echo "--- $TAG  $(date +%m-%d\ %H:%M:%S) ---"
    $PY $VD/finetune_vd.py --data "$DV" --model microsoft/codebert-base \
      --arm "$ARM" --id-field idx --famfile "$DV_FAM" --splitfile "$SP" \
      --seed "$SEED" --lr "$LR" --class-weight balanced --epochs 15 --patience 2 \
      --batch-size 16 --save-preds --save-best --out "$OUT" 2>&1 \
      | grep --line-buffered -viE "warn|it/s\]$"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ] || [ ! -f "$OUT/$TAG.json" ]; then
      echo "!!! FAILED $TAG (exit $rc) -- continuing"
      echo "$(date +%F\ %T) phase4b $TAG exit=$rc" >> "$FAILED"
    fi
  done
done

touch $D/.phase4b_done
echo "############ PHASE 4b RUNS COMPLETE  $(date +%m-%d\ %H:%M:%S) ############"
$PY $VD/analyze_phase4b.py --disk "$D"
