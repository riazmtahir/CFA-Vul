#!/bin/bash
# COMMIT-LEVEL TWIN LEAKAGE -- CPU only, light (ids/labels/commits; fine in 无卡模式).
# For each corpus and split arm: share of test functions whose opposite-label
# same-commit twin (patched / pre-patch version) is in train, and how much of that
# lies OUTSIDE the θ=0.7 family, i.e. survives family-disjoint splitting.
#   bash $CFA_CODE_DIR/run_commit_leak.sh 2>&1 | tee $CFA_DATA_DIR/commit_leak.log
set -eu
D="${CFA_DATA_DIR:?set CFA_DATA_DIR, see README}"; PY="${CFA_PYTHON:-python}"; VD="${CFA_CODE_DIR:-$(cd "$(dirname "$0")" && pwd)}"
OUT=$D/family_label; mkdir -p "$OUT"

# commit-disjoint arm (RESEARCH_PLAN §13.6): whole commits, size/prevalence matched to
# the family split, same two-key greedy + refinement as the project split.
CD=$D/diversevul_audit_splitfile_commitdisjoint_seed42
echo "############ 0/3  DiverseVul commit-disjoint split  $(date +%T) ############"
if [ -s "$CD.json" ]; then echo "SKIP (exists)"; else
  $PY $VD/make_project_split.py --data $D/data/diversevul/diversevul.jsonl \
      --ref-splitfile $D/diversevul_audit_splitfile_theta0.7.json \
      --famfile $D/diversevul_audit_families_theta0.7.json \
      --group-field commit_id --seed 42 --out "$CD"
fi

echo "############ 1/3  DiverseVul  $(date +%T) ############"
$PY $VD/commit_leak_audit.py --data $D/data/diversevul/diversevul.jsonl \
    --famfile $D/diversevul_audit_families_theta0.7.json --id-field idx \
    --splitfile random_control=$D/diversevul_audit_splitfile_randomctrl_theta0.7.json \
    --splitfile family_disjoint=$D/diversevul_audit_splitfile_theta0.7.json \
    --splitfile project_disjoint=$D/diversevul_audit_splitfile_projectdisjoint_seed42.json \
    --splitfile commit_disjoint="$CD.json" \
    --out $OUT/diversevul_commit_leak

echo "############ 2/3  PrimeVul  $(date +%T) ############"
$PY $VD/commit_leak_audit.py --data $D/data/primevul/primevul_all.jsonl \
    --famfile $D/primevul_audit_families_theta0.7.json --id-field idx \
    --splitfile random_control=$D/primevul_audit_splitfile_randomctrl_theta0.7.json \
    --splitfile family_disjoint=$D/primevul_audit_splitfile_theta0.7.json \
    --out $OUT/primevul_commit_leak

echo "############ 3/3  Devign  $(date +%T) ############"
$PY $VD/commit_leak_audit.py --data $D/data/devign/devign_all.jsonl \
    --famfile $D/devign_audit_families_theta0.7.json --id-field id \
    --splitfile random_control=$D/devign_audit_splitfile_randomctrl_theta0.7.json \
    --splitfile family_disjoint=$D/devign_audit_splitfile_theta0.7.json \
    --out $OUT/devign_commit_leak
echo
echo "commit-disjoint split report:"; cat "${CD}_report.json"; echo
echo "############ COMMIT LEAK AUDIT COMPLETE -- send everything above ############"
