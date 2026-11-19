# Croft-style re-scoring: test set with family-leaked functions removed (EXPLORATORY, A5)

Leaked = a training function of the arm's split is in the test function's θ=0.7 family. Mean over seeds; 95% bootstrap CI over test functions (paired for the difference). Removal changes the positive rate, so PR/pos is shown too.

| arm | seeds | test n | removed (leaked %) | PR-AUC full | PR-AUC without leaked | Δ (without − full) | pos rate full → without | PR/pos full → without |
|---|---|---|---|---|---|---|---|---|
| random_control | 4,5 | 33,172 | 9,748 (29.39%) | 0.2572 [0.2401, 0.2766] | 0.2046 [0.1853, 0.2286] | -0.0525 [-0.0675, -0.0361] | 0.0573 → 0.0460 | 4.49 → 4.45 |
| family_disjoint | 4,5 | 33,172 | 0 (0.0%) | 0.2621 [0.2455, 0.2806] | 0.2621 [0.2455, 0.2806] | +0.0000 [+0.0000, +0.0000] | 0.0570 → 0.0570 | 4.60 → 4.60 |
| project_disjoint | 1,2,3 | 33,165 | 1,934 (5.83%) | 0.1472 [0.1367, 0.1595] | 0.1476 [0.1369, 0.1601] | +0.0004 [-0.0013, +0.0018] | 0.0570 → 0.0594 | 2.58 → 2.48 |
| commit_disjoint | 1,2,3 | 33,172 | 5,680 (17.12%) | 0.1723 [0.1599, 0.1863] | 0.1594 [0.1476, 0.1732] | -0.0129 [-0.0205, -0.0060] | 0.0570 → 0.0600 | 3.02 → 2.66 |
