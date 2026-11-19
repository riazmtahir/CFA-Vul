# Commit-level twin leakage -- `primevul_all.jsonl`

opp. label = a train function from the same commit with the opposite label (any function of that commit). own twin = same commit AND same function name, opposite label: the function's own patched / pre-patch version. outside family = not in the test function's θ family, so family-disjoint splitting cannot remove it.

| arm | test subset | n | same commit in train % | opp. label same commit % | own twin in train % | own twin outside family % |
|---|---|---|---|---|---|---|
| published | all_test | 25,911 | 0.66 | 0.66 | 0.06 | 0.03 |
| published | test_positives | 695 | 0.86 | 0.86 | 0.86 | 0.58 |
| random_control | all_test | 24,649 | 99.51 | 77.87 | 4.23 | 2.54 |
| random_control | test_positives | 729 | 86.42 | 85.73 | 62.41 | 35.12 |
| family_disjoint | all_test | 24,649 | 99.38 | 77.42 | 2.5 | 2.5 |
| family_disjoint | test_positives | 681 | 85.17 | 84.29 | 32.45 | 32.45 |
