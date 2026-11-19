# Phase 4b -- commit_decon vs commit_sizectrl (pre-registered H4b, RESEARCH_PLAN §13.8)

Pre-check: **PASS**; selected design: **H4b**

| arm | seed | PR-AUC | headroom | MCC | ROC-AUC |
|---|---|---|---|---|---|
| commit_sizectrl | 1 | 0.1843 | 0.0431 | 0.2157 | 0.7584 |
| commit_sizectrl | 2 | 0.1901 | 0.0489 | 0.2125 | 0.7602 |
| commit_sizectrl | 3 | 0.1735 | 0.0323 | 0.2093 | 0.7538 |
| commit_decon | 1 | 0.1575 | 0.0163 | 0.1882 | 0.7434 |
| commit_decon | 2 | 0.1597 | 0.0185 | 0.1880 | 0.7385 |
| commit_decon | 3 | 0.1513 | 0.0101 | 0.1740 | 0.7376 |

| paired Δ sizectrl → decon (seeds 1–3) | pr_auc | headroom | mcc | roc_auc |
|---|---|---|---|---|
| Δ [95% t-CI] | -0.0265 ± 0.0041 [-0.0367, -0.0163] (n=3) | -0.0265 ± 0.0041 [-0.0367, -0.0163] (n=3) | -0.0291 ± 0.0056 [-0.0430, -0.0152] (n=3) | -0.0176 ± 0.0036 [-0.0265, -0.0088] (n=3) |

**H4b SUPPORTED: direct high-recall Type-3 clones in training inflate PR-AUC beyond commit sharing.**
