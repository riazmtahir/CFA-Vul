# Paired evaluation (PrimeVul-style) on DiverseVul test pairs -- identical pairs EXCLUDED (level: any)

Pair = vulnerable + patched version of the same function (same commit, same name), both in test. Validation max-F1 threshold per run. P-C correct pair · P-V both flagged · P-B neither · P-R reversed · rank-acc = P(score vuln > score patched), chance 0.5. Mean over seeds; 95% bootstrap CI over pairs (1000).

| arm | seeds | pairs (ambiguous skipped) | P-C | P-V | P-B | P-R | rank-acc |
|---|---|---|---|---|---|---|---|
| random_control | 4,5 | 136 (31) | 0.059 [0.033, 0.088] | 0.404 [0.331, 0.478] | 0.500 [0.423, 0.574] | 0.037 [0.015, 0.066] | 0.570 [0.507, 0.632] |
| family_disjoint | 4,5 | 550 (36) | 0.032 [0.020, 0.046] | 0.368 [0.331, 0.405] | 0.591 [0.555, 0.626] | 0.009 [0.004, 0.015] | 0.503 [0.475, 0.529] |
| project_disjoint | 1,2,3 | 989 (106) | 0.005 [0.002, 0.009] | 0.220 [0.197, 0.244] | 0.764 [0.740, 0.786] | 0.011 [0.007, 0.016] | 0.388 [0.366, 0.412] |
| commit_disjoint | 1,2,3 | 1,394 (75) | 0.013 [0.010, 0.017] | 0.256 [0.236, 0.276] | 0.711 [0.691, 0.732] | 0.019 [0.015, 0.024] | 0.436 [0.419, 0.452] |

Length check -- a length-only scorer ranks the LONGER function as vulnerable; rank-acc split by whether the patched version is longer than the vulnerable one.

| arm | patched longer | length-only rank-acc | rank-acc, patched longer | rank-acc, patched not longer | length-balanced rank-acc |
|---|---|---|---|---|---|
| random_control | 0.699 | 0.294 | 0.484 (n=95) | 0.768 (n=41) | 0.626 [0.565, 0.684] |
| family_disjoint | 0.795 | 0.186 | 0.471 (n=437) | 0.624 (n=113) | 0.548 [0.512, 0.579] |
| project_disjoint | 0.792 | 0.197 | 0.321 (n=783) | 0.646 (n=206) | 0.484 [0.460, 0.509] |
| commit_disjoint | 0.800 | 0.194 | 0.391 (n=1115) | 0.615 (n=279) | 0.503 [0.482, 0.522] |

Identical pairs excluded per arm: random_control 8, family_disjoint 138, project_disjoint 248, commit_disjoint 121
