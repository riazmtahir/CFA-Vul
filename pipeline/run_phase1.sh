#!/bin/bash
# PHASE 1 (RESEARCH_PLAN §13.3 items 1, 2, 6) -- CPU only, safe in 无卡模式.
# Reads the existing θ=0.7 audit outputs; trains nothing; never touches a GPU,
# Stage A/B results, checkpoints or the watchdog. Re-running overwrites only the
# Phase 1 outputs listed below.
#
#   bash $CFA_CODE_DIR/run_phase1.sh
#
# Writes:
#   $D/diversevul_audit_splitfile_projectdisjoint_seed42.json (+ _report.json)
#   $D/family_label/{diversevul,devign}_theta0.7.{json,md}  (+ _leakcat_<arm>.json)
set -eu
D="${CFA_DATA_DIR:?set CFA_DATA_DIR, see README}"; PY="${CFA_PYTHON:-python}"; VD="${CFA_CODE_DIR:-$(cd "$(dirname "$0")" && pwd)}"
OUT=$D/family_label; mkdir -p "$OUT"

DV=$D/data/diversevul/diversevul.jsonl
DV_FAM=$D/diversevul_audit_families_theta0.7.json
DV_FD=$D/diversevul_audit_splitfile_theta0.7.json
DV_RC=$D/diversevul_audit_splitfile_randomctrl_theta0.7.json
DV_PD=$D/diversevul_audit_splitfile_projectdisjoint_seed42

DEV=$D/data/devign/devign_all.jsonl
DEV_FAM=$D/devign_audit_families_theta0.7.json
DEV_FD=$D/devign_audit_splitfile_theta0.7.json
DEV_RC=$D/devign_audit_splitfile_randomctrl_theta0.7.json

for f in "$DV" "$DV_FAM" "$DV_FD" "$DV_RC" "$DEV" "$DEV_FAM" "$DEV_FD" "$DEV_RC"; do
  [ -s "$f" ] || { echo "!!! missing input: $f"; exit 1; }
done

echo "############ 1/3  DiverseVul project-disjoint split (seed 42)  $(date +%T) ############"
$PY $VD/make_project_split.py --data "$DV" --ref-splitfile "$DV_FD" \
    --famfile "$DV_FAM" --seed 42 --out "$DV_PD"

echo "############ 2/3  DiverseVul family label audit  $(date +%T) ############"
$PY $VD/family_label_audit.py --data "$DV" --famfile "$DV_FAM" --id-field idx \
    --splitfile random_control="$DV_RC" --splitfile family_disjoint="$DV_FD" \
    --splitfile project_disjoint="$DV_PD.json" \
    --out "$OUT/diversevul_theta0.7"

echo "############ 3/3  Devign family label audit  $(date +%T) ############"
$PY $VD/family_label_audit.py --data "$DEV" --famfile "$DEV_FAM" --id-field id \
    --splitfile random_control="$DEV_RC" --splitfile family_disjoint="$DEV_FD" \
    --out "$OUT/devign_theta0.7"

echo
echo "############ RESULTS -- send these back ############"
cat "$DV_PD"_report.json
echo
cat "$OUT/diversevul_theta0.7.md"
echo
cat "$OUT/devign_theta0.7.md"
