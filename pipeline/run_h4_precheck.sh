#!/bin/bash
# H4 / H4b PRE-CHECKS (RESEARCH_PLAN §13.7 and §13.8, 21 Oct 2026) -- CPU only, no GPU,
# trains nothing. ONE run prepares both pre-registered designs and applies the §13.8 precedence.
#   setsid nohup bash $CFA_CODE_DIR/run_h4_precheck.sh > $CFA_DATA_DIR/h4_precheck.log 2>&1 &
#   tail -f $CFA_DATA_DIR/h4_precheck.log
# Steps (built files are skipped when they exist, so re-running resumes; pre-checks always re-run):
#   1  scripts/make_commit_family_split.py -> diversevul_audit_splitfile_commithrfamily_seed42{.json,_report.json}
#   2  scripts/precheck_h4.py              -> h4_precheck.{json,md}       (exit 3 = H4 gate fails)
#   3  scripts/direct_clone_edges.py       -> direct_edges/diversevul_hr_direct_* (5 splits)
#   4  scripts/make_decon_splits.py        -> diversevul_audit_splitfile_commit{decon,sizectrl}_seed42.json
#   5  scripts/precheck_h4b.py             -> h4b_precheck.{json,md}      (exit 3 = H4b gate fails)
#   6  scripts/croft_rescore.py (lift CI)  -> rq2_diversevul_phase2/croft_rescore_lift.{json,md} (EXPLORATORY)
#   7  scripts/select_h4_design.py         -> h4_design.json: H4, H4b or none (§13.8 precedence)
# Fails loudly (exit 1) on a missing input or a crashed step. Exit 0 if a design is selected,
# 3 if neither gate passes (run neither; report both pre-checks).
set -u
D=${D:-$CFA_DATA_DIR}; PY=${PY:-$PY}; VD=${VD:-$CFA_CODE_DIR}
P2=$D/rq2_diversevul_phase2; P3=$D/rq2_diversevul_phase3

step() { echo; echo "############ $1  $(date +%T) ############"; }
die()  { echo "!!! $1"; exit 1; }
need() { for f in "$@"; do [ -s "$f" ] || die "missing input: $f"; done; }

echo "############ H4 / H4b pre-checks  $(date '+%F %T')  code: $(git -C "$VD" log --oneline -1 2>/dev/null) ############"
lim=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)
echo "cpus: $(nproc)   memory limit: $( [[ "$lim" =~ ^[0-9]+$ ]] && echo "$((lim/1024/1024)) MB" || echo "${lim:-unknown}")"

DV=$D/data/diversevul/diversevul.jsonl
HR=$D/audit_highrecall/diversevul_hr_families_theta0.7.json
FAM=$D/diversevul_audit_families_theta0.7.json
RC=$D/diversevul_audit_splitfile_randomctrl_theta0.7.json
FD=$D/diversevul_audit_splitfile_theta0.7.json
PD=$D/diversevul_audit_splitfile_projectdisjoint_seed42.json
CD=$D/diversevul_audit_splitfile_commitdisjoint_seed42.json
NEW=$D/diversevul_audit_splitfile_commithrfamily_seed42
EDG=$D/direct_edges/diversevul_hr_direct
PREDS=""
for s in 4 5; do PREDS="$PREDS $P2/codebert-base_random_control_seed$s.preds.json.gz $P2/codebert-base_family_disjoint_seed$s.preds.json.gz"; done
for s in 1 2 3; do PREDS="$PREDS $P2/codebert-base_project_disjoint_seed$s.preds.json.gz $P3/codebert-base_commit_disjoint_seed$s.preds.json.gz"; done
# shellcheck disable=SC2086
need "$DV" "$HR" "$FAM" "$CD" "$RC" "$FD" "$PD" $PREDS
mkdir -p "$D/direct_edges"

step "1/7  H4: build the commit + high-recall-family split (seed 42)"
if [ -s "$NEW.json" ] && [ -s "${NEW}_report.json" ]; then
  echo "SKIP (done)"
else
  $PY "$VD/scripts/make_commit_family_split.py" --data "$DV" --hr-famfile "$HR" --ref-splitfile "$CD" \
      --strict-famfile "$FAM" --seed 42 --out "$NEW" || die "split builder failed"
fi

step "2/7  H4 pre-check (§13.7)"
$PY "$VD/scripts/precheck_h4.py" --disk "$D"; pc4=$?
[ "$pc4" -eq 0 ] || [ "$pc4" -eq 3 ] || die "precheck_h4.py crashed (exit $pc4)"

step "3/7  direct high-recall clone edges, test -> train, five splits"
if [ -s "${EDG}_summary.json" ]; then
  echo "SKIP (done)"
else
  $PY "$VD/scripts/direct_clone_edges.py" --data "$DV" \
      --splitfile random_control="$RC" --splitfile family_disjoint="$FD" \
      --splitfile project_disjoint="$PD" --splitfile commit_disjoint="$CD" \
      --out "$EDG" || die "direct_clone_edges failed"
fi

step "4/7  H4b: build commit_decon and commit_sizectrl (seed 42)"
if [ -s "$D/diversevul_audit_splitfile_commitdecon_seed42.json" ] \
   && [ -s "$D/diversevul_audit_splitfile_commitsizectrl_seed42.json" ]; then
  echo "SKIP (done)"
else
  $PY "$VD/scripts/make_decon_splits.py" --data "$DV" --ref-splitfile "$CD" \
      --train-hits "${EDG}_commit_disjoint_train_hits.json" --seed 42 --out-dir "$D" || die "make_decon_splits failed"
fi

step "5/7  H4b pre-check (§13.8)"
$PY "$VD/scripts/precheck_h4b.py" --disk "$D"; pc4b=$?
[ "$pc4b" -eq 0 ] || [ "$pc4b" -eq 3 ] || die "precheck_h4b.py crashed (exit $pc4b)"

step "6/7  Croft re-scoring with the lift CI (EXPLORATORY)"
if [ -s "$P2/croft_rescore_lift.json" ]; then
  echo "SKIP (done)"
else
  $PY "$VD/scripts/croft_rescore.py" --disk "$D" --out "$P2/croft_rescore_lift" || die "croft_rescore failed"
fi

step "7/7  §13.8 precedence"
$PY "$VD/scripts/select_h4_design.py" --disk "$D" || die "select_h4_design failed"

echo
echo "############ SUMMARY -- send these back  $(date '+%F %T') ############"
cat "${NEW}_report.json"; echo
cat "$D/h4_precheck.md"; echo
cat "${EDG}_summary.md"; echo
cat "$D/diversevul_h4b_splits_report.json"; echo
cat "$D/h4b_precheck.md"; echo
cat "$P2/croft_rescore_lift.md"; echo
cat "$D/h4_design.json"; echo
design=$(grep -o '"design": "[A-Za-z0-9]*"' "$D/h4_design.json" | sed 's/.*: "//; s/"//')
touch "$D/.h4_prechecks_done"
case "$design" in
  H4)  echo "############ SELECTED: H4 (§13.7 gate passed) -- run_phase4.sh after supervisor OK ############"; exit 0 ;;
  H4b) echo "############ SELECTED: H4b (§13.7 failed, §13.8 passed) -- run_phase4b.sh after supervisor OK ############"; exit 0 ;;
  *)   echo "############ SELECTED: none -- both gates failed; run neither, report both pre-checks ############"; exit 3 ;;
esac
