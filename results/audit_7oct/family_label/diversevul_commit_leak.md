# Commit-level twin leakage -- `diversevul.jsonl`

opp. label = a train function from the same commit with the opposite label (any function of that commit). own twin = same commit AND same function name, opposite label: the function's own patched / pre-patch version. outside family = not in the test function's θ family, so family-disjoint splitting cannot remove it.

| arm | test subset | n | same commit in train % | opp. label same commit % | own twin in train % | own twin outside family % |
|---|---|---|---|---|---|---|
| published | all_test | 33,050 | 99.75 | 90.84 | 9.36 | 6.21 |
| published | test_positives | 1,931 | 96.94 | 95.03 | 70.9 | 42.52 |
| random_control | all_test | 33,172 | 99.72 | 91.63 | 9.44 | 6.24 |
| random_control | test_positives | 1,901 | 95.79 | 92.79 | 69.02 | 42.29 |
| family_disjoint | all_test | 33,172 | 99.54 | 90.01 | 6.52 | 6.52 |
| family_disjoint | test_positives | 1,891 | 94.82 | 92.38 | 46.01 | 46.01 |
| project_disjoint | all_test | 33,165 | 0.0 | 0.0 | 0.0 | 0.0 |
| project_disjoint | test_positives | 1,891 | 0.0 | 0.0 | 0.0 | 0.0 |
| commit_disjoint | all_test | 33,172 | 0.0 | 0.0 | 0.0 | 0.0 |
| commit_disjoint | test_positives | 1,891 | 0.0 | 0.0 | 0.0 | 0.0 |
