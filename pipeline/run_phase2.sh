#!/bin/bash
# PHASE 2 -- RESEARCH_PLAN §13.4. GATED: resume.sh starts this only when
# $CFA_DATA_DIR/.phase2_approved exists AND Stage B is complete (9/9).
#
# 7 CodeBERT runs on DiverseVul, protocol frozen to Stage B (class-weighted CE,
# lr from lr_diversevul.txt, <=15 epochs, patience 2, batch 16, full train set),
# every run saving test+valid predictions and best-valid weights:
#   project_disjoint  seeds 1-3   -> pre-registered H2
#   random_control    seeds 4-5   -> n=5 for the main Δ, predictions for H1
#   family_disjoint   seeds 4-5
# Order interleaves arms so a partial batch still yields usable pairs.
# Output: $CFA_DATA_DIR/rq2_diversevul_phase2/ (never rq2_diversevul_final/).
set -u
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"   # the original runs used the mirror https://hf-mirror.com
export HF_HOME="${HF_HOME:-$CFA_DATA_DIR/hf}"
D="${CFA_DATA_DIR:?set CFA_DATA_DIR, see README}"; PY="${CFA_PYTHON:-python}"; VD="${CFA_CODE_DIR:-$(cd "$(dirname "$0")" && pwd)}"
FAILED=$D/failed_runs.txt

DV=$D/data/diversevul/diversevul.jsonl
DV_FAM=$D/diversevul_audit_families_theta0.7.json
DV_FD=$D/diversevul_audit_splitfile_theta0.7.json
DV_RC=$D/diversevul_audit_splitfile_randomctrl_theta0.7.json
DV_PD=$D/diversevul_audit_splitfile_projectdisjoint_seed42.json
OUT=$D/rq2_diversevul_phase2
mkdir -p "$OUT"

for f in "$DV" "$DV_FAM" "$DV_FD" "$DV_RC" "$DV_PD" "$D/lr_diversevul.txt"; do
  [ -s "$f" ] || { echo "!!! missing input: $f -- run run_phase1.sh / Stage A first"; exit 1; }
done
LR=$(cat "$D/lr_diversevul.txt")

# Freeze what this batch ran on: split-file fingerprints, code version, GPU.
if [ ! -f "$OUT/provenance.txt" ]; then
  {
    echo "phase 2 started $(date '+%F %T %Z')"
    echo "code: $(git -C "$VD" log --oneline -1 2>/dev/null)"
    echo "gpu:  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)"
    echo "lr:   $LR"
    md5sum "$DV_FAM" "$DV_FD" "$DV_RC" "$DV_PD" "$DV"
  } > "$OUT/provenance.txt"
fi
echo "############ PHASE 2  DiverseVul  lr=$LR  $(date +%m-%d\ %H:%M:%S) ############"

run_one() {
  local tag=$1; shift
  if [ -f "$OUT/$tag.json" ]; then echo "SKIP $tag (done)"; return 0; fi
  echo "--- $tag  $(date +%m-%d\ %H:%M:%S) ---"
  $PY $VD/finetune_vd.py "$@" --out "$OUT" 2>&1 | grep --line-buffered -viE "warn|it/s\]$"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "!!! FAILED $tag (exit $rc) -- continuing"
    echo "$(date +%F\ %T) phase2 $tag exit=$rc" >> "$FAILED"
    return 1
  fi
  [ -f "$OUT/$tag.json" ] || { echo "!!! $tag no JSON"; \
    echo "$(date +%F\ %T) phase2 $tag no-json" >> "$FAILED"; return 1; }
}

split_of() { case $1 in random_control) echo "$DV_RC";; family_disjoint) echo "$DV_FD";;
                        project_disjoint) echo "$DV_PD";; esac; }

for RUN in project_disjoint:1 random_control:4 family_disjoint:4 \
           project_disjoint:2 random_control:5 family_disjoint:5 \
           project_disjoint:3; do
  ARM=${RUN%%:*}; SEED=${RUN##*:}
  TAG="codebert-base_${ARM}_seed${SEED}"
  run_one "$TAG" \
    --data "$DV" --model microsoft/codebert-base --arm "$ARM" --id-field idx \
    --famfile "$DV_FAM" --splitfile "$(split_of "$ARM")" --seed "$SEED" --lr "$LR" \
    --class-weight balanced --epochs 15 --patience 2 --batch-size 16 \
    --save-preds --save-best || true
done

touch $D/.phase2_done
echo "############ PHASE 2 COMPLETE  $(date +%m-%d\ %H:%M:%S) ############"
grep -s phase2 "$FAILED" && echo "(phase 2 failures above)" || echo "no failed phase 2 runs"
