# Phase 3 -- commit-disjoint arm (pre-registered H3)

| arm | seed | PR-AUC | headroom | MCC | ROC-AUC |
|---|---|---|---|---|---|
| random_control | 1 | 0.2467 | 0.0967 | 0.2639 | 0.8076 |
| random_control | 2 | 0.2647 | 0.1147 | 0.2774 | 0.8226 |
| random_control | 3 | 0.2502 | 0.1002 | 0.2610 | 0.8093 |
| project_disjoint | 1 | 0.1484 | 0.0040 | 0.1813 | 0.7393 |
| project_disjoint | 2 | 0.1531 | 0.0087 | 0.1884 | 0.7434 |
| project_disjoint | 3 | 0.1401 | -0.0043 | 0.1682 | 0.7336 |
| commit_disjoint | 1 | 0.1651 | 0.0238 | 0.2041 | 0.7403 |
| commit_disjoint | 2 | 0.1770 | 0.0357 | 0.2064 | 0.7521 |
| commit_disjoint | 3 | 0.1750 | 0.0337 | 0.2093 | 0.7591 |

| paired Δ (seeds 1–3) | pr_auc | headroom | mcc | roc_auc |
|---|---|---|---|---|
| random to commit | -0.0815 ± 0.0063 [-0.0970, -0.0660] (n=3) | -0.0728 ± 0.0063 [-0.0883, -0.0573] (n=3) | -0.0608 ± 0.0097 [-0.0849, -0.0368] (n=3) | -0.0627 ± 0.0109 [-0.0898, -0.0356] (n=3) |
| random to project | -0.1067 ± 0.0073 [-0.1248, -0.0886] (n=3) | -0.1011 ± 0.0073 [-0.1192, -0.0830] (n=3) | -0.0881 ± 0.0052 [-0.1009, -0.0753] (n=3) | -0.0744 ± 0.0056 [-0.0882, -0.0606] (n=3) |
| commit to project | -0.0252 ± 0.0092 [-0.0479, -0.0024] (n=3) | -0.0283 ± 0.0092 [-0.0510, -0.0055] (n=3) | -0.0273 ± 0.0122 [-0.0576, +0.0030] (n=3) | -0.0117 ± 0.0125 [-0.0429, +0.0194] (n=3) |

**H3 SUPPORTED: same-commit sharing inflates random-split PR-AUC.**
Share of the random→project gap accounted for by commit sharing: 0.76.
