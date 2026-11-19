# Commit-level twin leakage -- `devign_all.jsonl`

opp. label = a train function from the same commit with the opposite label (any function of that commit). own twin = same commit AND same function name, opposite label: the function's own patched / pre-patch version. outside family = not in the test function's θ family, so family-disjoint splitting cannot remove it.

| arm | test subset | n | same commit in train % | opp. label same commit % | own twin in train % | own twin outside family % |
|---|---|---|---|---|---|---|
| published | all_test | 2,732 | 65.04 | 0.0 | 0.0 | 0.0 |
| published | test_positives | 1,255 | 58.17 | 0.0 | 0.0 | 0.0 |
| random_control | all_test | 2,738 | 66.18 | 0.0 | 0.0 | 0.0 |
| random_control | test_positives | 1,249 | 58.77 | 0.0 | 0.0 | 0.0 |
| family_disjoint | all_test | 2,738 | 65.08 | 0.0 | 0.0 | 0.0 |
| family_disjoint | test_positives | 1,245 | 57.51 | 0.0 | 0.0 | 0.0 |
