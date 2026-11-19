# M1: post-warm-up selection replayed on the logged validation curves

Final-protocol runs (max 15 epochs, patience 2): 53. Same selected epoch: 49. Different or not determinable: 4.

Check: default mode equals the original rule on 20,000 random curves (mismatches: 0); first eligible epoch for 15 epochs: 2.

| run | logged best / run | post-warm-up best within the log | status | validation PR-AUC by epoch |
|---|---|---|---|---|
| from_backup/rq2_devign_final/graphcodebert-base_family_disjoint_seed2.json | 1 / 3 | 3 | differs: would train past the logged epochs (not determinable from the log) | 0.6401, 0.5008, 0.5067 |
| rq2_diversevul_phase2/codebert-base_project_disjoint_seed1.json | 1 / 3 | 3 | differs: would train past the logged epochs (not determinable from the log) | 0.1917, 0.1680, 0.1683 |
| rq2_diversevul_phase2/codebert-base_project_disjoint_seed2.json | 1 / 3 | 3 | differs: would train past the logged epochs (not determinable from the log) | 0.2016, 0.1574, 0.1742 |
| rq2_diversevul_phase2/codebert-base_project_disjoint_seed3.json | 1 / 3 | 2 | differs: would train past the logged epochs (not determinable from the log) | 0.1957, 0.1852, 0.1399 |
