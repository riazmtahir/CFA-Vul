#!/bin/bash
# PRIMEVUL AUDIT + THETA SENSITIVITY (RESEARCH_PLAN §13.3 items 3, 4) -- CPU only,
# safe in 无卡模式. Trains nothing; never touches Stage B / Phase 2 outputs or the
# θ=0.7 audit files the training runs used. Resumable: each step is skipped when
# its output exists, so re-running after a stop continues where it left off.
#
#   setsid nohup bash $CFA_CODE_DIR/run_primevul_audit.sh > $CFA_DATA_DIR/primevul_audit.log 2>&1 &
#   tail -f $CFA_DATA_DIR/primevul_audit.log          # Ctrl+C stops watching only
#
# Needs: $CFA_DATA_DIR/data/primevul/primevul_{train,valid,test}.jsonl (+ _paired)
# Writes:
#   data/primevul/primevul_all.jsonl, primevul_prep_report.json
#   primevul_audit_{report.json,report.md,families_theta*.json,splitfile*_theta*.json}
#   family_label/primevul_theta0.7.{json,md} (+ _leakcat_<arm>.json)
#   family_label/primevul_pairs.{json,md}
#   theta_sens/{devign,diversevul}_audit_report.{json,md}   (θ 0.8, 0.9)
set -u
D="${CFA_DATA_DIR:?set CFA_DATA_DIR, see README}"; PY="${CFA_PYTHON:-python}"; VD="${CFA_CODE_DIR:-$(cd "$(dirname "$0")" && pwd)}"
P=$D/data/primevul
PV=$P/primevul_all.jsonl
PA=$D/primevul_audit
mkdir -p "$D/family_label" "$D/theta_sens"

echo "############ PrimeVul audit  $(date '+%F %T')  code: $(git -C $VD log --oneline -1 2>/dev/null) ############"
lim=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)
echo "cpus: $(nproc)   memory limit: $( [[ "$lim" =~ ^[0-9]+$ ]] && echo "$((lim/1024/1024)) MB" || echo "${lim:-unknown}")"

step() { echo; echo "############ $1  $(date +%T) ############"; }
die()  { echo "!!! $1"; exit 1; }

step "1/6  merge PrimeVul splits + check against the paper"
if [ -s "$PV" ] && grep -q '"matches_paper": true' "$P/primevul_prep_report.json" 2>/dev/null; then
  echo "SKIP (done)"
else
  $PY $VD/primevul_tools.py prep --dir "$P" || die "prep failed or counts do not match the paper"
fi

step "2/6  PrimeVul clone-family audit, θ 0.7/0.8/0.9 + 1-NN baseline"
if [ -s "${PA}_report.json" ]; then
  echo "SKIP (done)"
else
  $PY $VD/clone_family_audit.py --data "$PV" --split-field split --id-field idx \
      --theta 0.7 0.8 0.9 --knn-baseline --out "$PA" || die "PrimeVul audit failed"
fi

step "3/6  PrimeVul family label audit (published / random_control / family_disjoint)"
if [ -s "$D/family_label/primevul_theta0.7.json" ]; then
  echo "SKIP (done)"
else
  $PY $VD/family_label_audit.py --data "$PV" --famfile "${PA}_families_theta0.7.json" \
      --id-field idx \
      --splitfile random_control="${PA}_splitfile_randomctrl_theta0.7.json" \
      --splitfile family_disjoint="${PA}_splitfile_theta0.7.json" \
      --out "$D/family_label/primevul_theta0.7" || die "label audit failed"
fi

step "4/6  PrimeVul vulnerable/patch pairs vs families"
$PY $VD/primevul_tools.py pairs --dir "$P" \
    --famfile 0.7="${PA}_families_theta0.7.json" \
    --famfile 0.8="${PA}_families_theta0.8.json" \
    --famfile 0.9="${PA}_families_theta0.9.json" \
    --out "$D/family_label/primevul_pairs" || die "pairs check failed"

# θ sensitivity: separate output prefix, so the θ=0.7 files used by training stay untouched.
step "5/6  Devign θ 0.8/0.9 sensitivity"
if [ -s "$D/theta_sens/devign_audit_report.json" ]; then
  echo "SKIP (done)"
else
  $PY $VD/clone_family_audit.py --data "$D/data/devign/devign_all.jsonl" \
      --split-field split --id-field id --theta 0.8 0.9 \
      --out "$D/theta_sens/devign_audit" || die "Devign sensitivity failed"
fi

step "6/6  DiverseVul θ 0.8/0.9 sensitivity"
if [ -s "$D/theta_sens/diversevul_audit_report.json" ]; then
  echo "SKIP (done)"
else
  $PY $VD/clone_family_audit.py --data "$D/data/diversevul/diversevul.jsonl" \
      --split-field split --id-field idx --theta 0.8 0.9 \
      --out "$D/theta_sens/diversevul_audit" || die "DiverseVul sensitivity failed"
fi

touch "$D/.primevul_audit_done"
echo
echo "############ RESULTS -- send these back  $(date '+%F %T') ############"
cat "$P/primevul_prep_report.json"; echo
cat "${PA}_report.md"; echo
cat "$D/family_label/primevul_theta0.7.md"; echo
cat "$D/family_label/primevul_pairs.md"; echo
cat "$D/theta_sens/devign_audit_report.md"; echo
cat "$D/theta_sens/diversevul_audit_report.md"
echo "############ PRIMEVUL AUDIT COMPLETE ############"
