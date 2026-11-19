#!/bin/bash
# CPU CHECKS A3 / A4 / A5 (18 Oct 2026) -- CPU only, safe in 无卡模式 for items 1 and 3;
# item 2 reads both corpora (~2 GB RAM, see the project log "Next process"). Trains nothing and
# never touches a pre-registered output. Resumable: each step is skipped when its output
# exists. Fails loudly (exit 1) if an input is missing or a step fails.
#
#   setsid nohup bash $CFA_CODE_DIR/run_cpu_checks.sh > $CFA_DATA_DIR/cpu_checks.log 2>&1 &
#   tail -f $CFA_DATA_DIR/cpu_checks.log
#
# Writes
#   rq2_diversevul_phase2/pair_identity_check.{json,md}       A4 counts (descriptive)
#   rq2_diversevul_phase2/paired_eval_noidentical.{json,md}   A4 paired eval without them
#   corpus_overlap/primevul_diversevul_overlap.{json,md}      A3 (descriptive)
#   rq2_diversevul_phase2/croft_rescore.{json,md}             A5 (EXPLORATORY)
set -u
D=${D:-$CFA_DATA_DIR}; PY=${PY:-$PY}; VD=${VD:-$CFA_CODE_DIR}
P2=$D/rq2_diversevul_phase2; P3=$D/rq2_diversevul_phase3

step() { echo; echo "############ $1  $(date +%T) ############"; }
die()  { echo "!!! $1"; exit 1; }
need() { for f in "$@"; do [ -s "$f" ] || die "missing input: $f"; done; }

echo "############ CPU checks A3/A4/A5  $(date '+%F %T')  code: $(git -C "$VD" log --oneline -1 2>/dev/null) ############"

DV=$D/data/diversevul/diversevul.jsonl
PV=$D/data/primevul/primevul_all.jsonl
RC=$D/diversevul_audit_splitfile_randomctrl_theta0.7.json
FD=$D/diversevul_audit_splitfile_theta0.7.json
PD=$D/diversevul_audit_splitfile_projectdisjoint_seed42.json
CD=$D/diversevul_audit_splitfile_commitdisjoint_seed42.json
FAM=$D/diversevul_audit_families_theta0.7.json
PREDS=""
for s in 4 5; do PREDS="$PREDS $P2/codebert-base_random_control_seed$s.preds.json.gz $P2/codebert-base_family_disjoint_seed$s.preds.json.gz"; done
for s in 1 2 3; do PREDS="$PREDS $P2/codebert-base_project_disjoint_seed$s.preds.json.gz $P3/codebert-base_commit_disjoint_seed$s.preds.json.gz"; done
# shellcheck disable=SC2086
need "$DV" "$PV" "$RC" "$FD" "$PD" "$CD" "$FAM" $PREDS

step "1/3  A4 identical vulnerable/patched pairs + paired eval without them"
if [ -s "$P2/pair_identity_check.json" ] && [ -s "$P2/paired_eval_noidentical.json" ]; then
  echo "SKIP (done)"
else
  $PY "$VD/scripts/pair_identity_check.py" --disk "$D" || die "pair_identity_check failed"
fi

step "2/3  A3 PrimeVul ∩ DiverseVul overlap"
if [ -s "$D/corpus_overlap/primevul_diversevul_overlap.json" ]; then
  echo "SKIP (done)"
else
  $PY "$VD/scripts/corpus_overlap.py" --disk "$D" || die "corpus_overlap failed"
fi

step "3/3  A5 Croft-style re-scoring (EXPLORATORY)"
if [ -s "$P2/croft_rescore.json" ]; then
  echo "SKIP (done)"
else
  $PY "$VD/scripts/croft_rescore.py" --disk "$D" || die "croft_rescore failed"
fi

for f in "$P2/pair_identity_check.md" "$P2/paired_eval_noidentical.md" \
         "$D/corpus_overlap/primevul_diversevul_overlap.md" "$P2/croft_rescore.md"; do
  [ -s "$f" ] || die "expected output missing: $f"
done
touch "$D/.cpu_checks_done"
echo
echo "############ SUMMARY -- send these back  $(date '+%F %T') ############"
cat "$P2/pair_identity_check.md"; echo
cat "$P2/paired_eval_noidentical.md"; echo
cat "$D/corpus_overlap/primevul_diversevul_overlap.md"; echo
cat "$P2/croft_rescore.md"
echo "############ CPU CHECKS COMPLETE ############"
