#!/bin/bash
# EXPLORATORY HIGH-RECALL FAMILY AUDIT (action A14, 18 Oct 2026) -- CPU only. Sensitivity
# analysis of the RQ1 audit under a looser clone definition: blind identifiers (every
# identifier -> ID) and 3-token shingles. On BigCloneBench this variant captures 88.2% of
# Strong Type-3 pairs and 0.05% of non-clone pairs at theta=0.7, against 30.0% / 0.0% for
# the audit's default (results/rq0_calibration_ours/highrecall_blind_k3/). It does NOT
# re-define any training arm and re-tests no hypothesis (action_list.md, NOT ALLOWED).
#
#   setsid nohup bash $CFA_CODE_DIR/run_audit_highrecall.sh > $CFA_DATA_DIR/audit_highrecall.log 2>&1 &
#   tail -f $CFA_DATA_DIR/audit_highrecall.log
#
# Per corpus (Devign, DiverseVul, PrimeVul):
#   a) clone_family_audit.py --rename blind --shingle-k 3 --theta 0.7 0.8 0.9
#      -> published-split leakage + its own random control, per theta
#   b) family_label_audit.py with the high-recall theta=0.7 families on the published split
#      and on the random-control / family-disjoint split files used for training
#      -> how much high-recall family leakage those training arms keep
# Writes $D/audit_highrecall/{devign,diversevul,primevul}_hr_* and *_label_theta0.7.{json,md}.
# Resumable: each step is skipped when its output exists. Never writes outside audit_highrecall/.
set -u
D=${D:-$CFA_DATA_DIR}; PY=${PY:-$PY}; VD=${VD:-$CFA_CODE_DIR}
O=$D/audit_highrecall
mkdir -p "$O"
HR="--rename blind --shingle-k 3"

echo "############ high-recall audit (EXPLORATORY)  $(date '+%F %T')  code: $(git -C "$VD" log --oneline -1 2>/dev/null) ############"
lim=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)
echo "cpus: $(nproc)   memory limit: $( [[ "$lim" =~ ^[0-9]+$ ]] && echo "$((lim/1024/1024)) MB" || echo "${lim:-unknown}")"

step() { echo; echo "############ $1  $(date +%T) ############"; }
die()  { echo "!!! $1"; exit 1; }
need() { for f in "$@"; do [ -s "$f" ] || die "missing input: $f"; done; }

DEV=$D/data/devign/devign_all.jsonl
DV=$D/data/diversevul/diversevul.jsonl
PV=$D/data/primevul/primevul_all.jsonl
need "$DEV" "$DV" "$PV" \
     "$D/devign_audit_splitfile_randomctrl_theta0.7.json" "$D/devign_audit_splitfile_theta0.7.json" \
     "$D/diversevul_audit_splitfile_randomctrl_theta0.7.json" "$D/diversevul_audit_splitfile_theta0.7.json" \
     "$D/primevul_audit_splitfile_randomctrl_theta0.7.json" "$D/primevul_audit_splitfile_theta0.7.json"

# corpus  data  id-field  prefix of the default (theta=0.7) split files on $D
for spec in "devign|$DEV|id|$D/devign_audit" "diversevul|$DV|idx|$D/diversevul_audit" \
            "primevul|$PV|idx|$D/primevul_audit"; do
  IFS='|' read -r C DATA IDF REF <<< "$spec"

  step "$C  a) audit, blind identifiers, k=3, theta 0.7/0.8/0.9"
  if [ -s "$O/${C}_hr_report.json" ]; then
    echo "SKIP (done)"
  else
    # shellcheck disable=SC2086
    $PY "$VD/clone_family_audit.py" --data "$DATA" --split-field split --id-field "$IDF" \
        $HR --theta 0.7 0.8 0.9 --out "$O/${C}_hr" || die "$C high-recall audit failed"
  fi

  step "$C  b) label audit of the high-recall families on the training split files"
  if [ -s "$O/${C}_hr_label_theta0.7.json" ]; then
    echo "SKIP (done)"
  else
    $PY "$VD/family_label_audit.py" --data "$DATA" --famfile "$O/${C}_hr_families_theta0.7.json" \
        --id-field "$IDF" \
        --splitfile random_control="${REF}_splitfile_randomctrl_theta0.7.json" \
        --splitfile family_disjoint="${REF}_splitfile_theta0.7.json" \
        --out "$O/${C}_hr_label_theta0.7" || die "$C label audit failed"
  fi
done

touch "$D/.audit_highrecall_done"
echo
echo "############ RESULTS (EXPLORATORY) -- send these back  $(date '+%F %T') ############"
for C in devign diversevul primevul; do
  echo "=== $C"; cat "$O/${C}_hr_report.md"; echo; cat "$O/${C}_hr_label_theta0.7.md"; echo
done
echo "############ HIGH-RECALL AUDIT COMPLETE ############"
