#!/bin/bash
# M1 SENSITIVITY (referee review 9 Nov 2026, item 13). GATED: runs only when
# $CFA_DATA_DIR/.m1_approved exists. Not part of any pre-specified analysis; reported as a sensitivity.
#
# Re-runs the three project-disjoint CodeBERT runs of Phase 2 with the SAME split file, seeds 1-3 and
# hyperparameters (class-weighted CE, lr from lr_diversevul.txt, <= 15 epochs, patience 2, batch 16,
# full train set), changing only the model-selection rule (finetune_vd.py --select-after-warmup):
#   - only epochs that end after the 1.5-epoch warm-up are eligible (first eligible epoch: 2);
#   - patience is counted from the first eligible epoch;
#   - test metrics are logged at every epoch, for reporting only.
# Output: $CFA_DATA_DIR/rq2_diversevul_m1_postwarmup/ (tags *_postwarmup; primary files untouched).
# Then the fourth run that selected epoch 1: GraphCodeBERT, Devign family-disjoint, seed 2 (Stage A2
# protocol), into $CFA_DATA_DIR/rq2_devign_m1_postwarmup/, with a content check of its split file.
# Resumable: finished runs are skipped, an interrupted run resumes from its checkpoint.
set -u
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"   # the original runs used the mirror https://hf-mirror.com
export HF_HOME="${HF_HOME:-$CFA_DATA_DIR/hf}"
D="${CFA_DATA_DIR:?set CFA_DATA_DIR, see README}"; PY="${CFA_PYTHON:-python}"; VD="${CFA_CODE_DIR:-$(cd "$(dirname "$0")" && pwd)}"
FAILED=$D/failed_runs.txt
[ -f "$D/.m1_approved" ] || { echo "!!! not approved: touch $D/.m1_approved first"; exit 2; }

DV=$D/data/diversevul/diversevul.jsonl
DV_FAM=$D/diversevul_audit_families_theta0.7.json
DV_PD=$D/diversevul_audit_splitfile_projectdisjoint_seed42.json
OUT=$D/rq2_diversevul_m1_postwarmup
mkdir -p "$OUT"
for f in "$DV" "$DV_FAM" "$DV_PD" "$D/lr_diversevul.txt"; do
  [ -s "$f" ] || { echo "!!! missing input: $f"; exit 1; }
done
# The split file must be the one Phase 2 used (md5 recorded in rq2_diversevul_phase2/provenance.txt).
WANT=aae66bcb79411a8fb67b810116a2f982
GOT=$(md5sum "$DV_PD" | cut -d' ' -f1)
[ "$GOT" = "$WANT" ] || { echo "!!! project split md5 $GOT != Phase 2's $WANT -- stop"; exit 1; }
grep -q -- "--select-after-warmup" "$VD/finetune_vd.py" || { echo "!!! finetune_vd.py lacks --select-after-warmup: sync the code"; exit 1; }
LR=$(cat "$D/lr_diversevul.txt")

if [ ! -f "$OUT/provenance.txt" ]; then
  {
    echo "m1 sensitivity started $(date '+%F %T %Z')"
    echo "code: $(git -C "$VD" log --oneline -1 2>/dev/null) (md5 finetune_vd.py $(md5sum "$VD/finetune_vd.py" | cut -d' ' -f1))"
    echo "gpu:  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)"
    echo "lr:   $LR"
    echo "rule: --select-after-warmup (first eligible epoch after the 10% warm-up; patience from there)"
    md5sum "$DV_FAM" "$DV_PD" "$DV"
  } > "$OUT/provenance.txt"
fi
echo "############ M1 SENSITIVITY  project-disjoint  lr=$LR  $(date +%m-%d\ %H:%M:%S) ############"

for SEED in 1 2 3; do
  TAG="codebert-base_project_disjoint_seed${SEED}_postwarmup"
  if [ -f "$OUT/$TAG.json" ]; then echo "SKIP $TAG (done)"; continue; fi
  echo "--- $TAG  $(date +%m-%d\ %H:%M:%S) ---"
  $PY $VD/finetune_vd.py \
    --data "$DV" --model microsoft/codebert-base --arm project_disjoint --id-field idx \
    --famfile "$DV_FAM" --splitfile "$DV_PD" --seed "$SEED" --lr "$LR" \
    --class-weight balanced --epochs 15 --patience 2 --batch-size 16 \
    --save-preds --select-after-warmup --out "$OUT" 2>&1 | grep --line-buffered -viE "warn|it/s\]$"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ] || [ ! -f "$OUT/$TAG.json" ]; then
    echo "!!! FAILED $TAG (exit $rc)"; echo "$(date +%F\ %T) m1 $TAG exit=$rc" >> "$FAILED"
  fi
done

# ---- Devign: GraphCodeBERT, family-disjoint, seed 2 (the run that collapsed after epoch 1) --------------
# Same protocol as Stage A2 (run_stage_a2.sh): balanced CE, Devign lr, <= 15 epochs, patience 2, batch 16;
# only the selection rule changes. No md5 of the Devign split was recorded in Stage A2, so the split is
# checked by content: the rows finetune_vd.py would load must give exactly the partition sizes logged in
# the original run's JSON, and the split path must be the one that JSON names. Its md5 is recorded here.
DEV=$D/data/devign/devign_all.jsonl
DEV_FAM=$D/devign_audit_families_theta0.7.json
DEV_FD=$D/devign_audit_splitfile_theta0.7.json
ORIG=$D/rq2_devign_final/graphcodebert-base_family_disjoint_seed2.json
DOUT=$D/rq2_devign_m1_postwarmup
mkdir -p "$DOUT"
DTAG="graphcodebert-base_family_disjoint_seed2_postwarmup"
for f in "$DEV" "$DEV_FAM" "$DEV_FD" "$ORIG"; do
  [ -s "$f" ] || { echo "!!! missing input: $f"; exit 1; }
done
DLR=$($PY - "$ORIG" "$DEV" "$DEV_FD" <<'PYEOF'
import json, sys
orig, data, split = sys.argv[1:4]
o = json.load(open(orig, encoding="utf-8"))
assert o["splitfile"] == split, f"split path {o['splitfile']} != {split}"
assert (o["model"], o["arm"], o["seed"], o["epochs"], o["patience"], o["batch_size"], o["class_weight"]) == \
    ("microsoft/graphcodebert-base", "family_disjoint", 2, 15, 2, 16, "balanced"), "original run settings differ"
assign = json.load(open(split, encoding="utf-8"))
n = {"train": 0, "valid": 0, "test": 0}
for i, line in enumerate(open(data, encoding="utf-8", errors="replace")):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if not isinstance(r.get("func"), str) or not r["func"].strip():
        continue
    sp = assign.get(str(r.get("id", i)))
    if sp is None:
        continue
    sp = "valid" if sp in ("valid", "validation") else sp
    if sp in n:
        n[sp] += 1
want = {"train": o["n_train"], "valid": o["n_valid"], "test": o["n_test"]}
assert n == want, f"partition sizes {n} != original run {want}"
print(o["lr"])
PYEOF
) || { echo "!!! Devign split check failed -- stop"; exit 1; }
if [ ! -f "$DOUT/provenance.txt" ]; then
  {
    echo "m1 sensitivity (Devign) started $(date '+%F %T %Z')"
    echo "code: $(git -C "$VD" log --oneline -1 2>/dev/null) (md5 finetune_vd.py $(md5sum "$VD/finetune_vd.py" | cut -d' ' -f1))"
    echo "gpu:  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)"
    echo "lr:   $DLR (read from $ORIG)"
    echo "split check: partition sizes equal the original run's n_train/n_valid/n_test"
    md5sum "$DEV_FAM" "$DEV_FD" "$DEV"
  } > "$DOUT/provenance.txt"
fi
if [ -f "$DOUT/$DTAG.json" ]; then echo "SKIP $DTAG (done)"; else
  echo "--- $DTAG  $(date +%m-%d\ %H:%M:%S) ---"
  $PY $VD/finetune_vd.py \
    --data "$DEV" --model microsoft/graphcodebert-base --arm family_disjoint --id-field id \
    --famfile "$DEV_FAM" --splitfile "$DEV_FD" --seed 2 --lr "$DLR" \
    --class-weight balanced --epochs 15 --patience 2 --batch-size 16 \
    --save-preds --select-after-warmup --out "$DOUT" 2>&1 | grep --line-buffered -viE "warn|it/s\]$"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ] || [ ! -f "$DOUT/$DTAG.json" ]; then
    echo "!!! FAILED $DTAG (exit $rc)"; echo "$(date +%F\ %T) m1 $DTAG exit=$rc" >> "$FAILED"
  fi
fi
touch $D/.m1_done
echo "############ M1 SENSITIVITY DONE  $(date +%m-%d\ %H:%M:%S) ############"
