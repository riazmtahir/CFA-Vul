# Paired evaluation (PrimeVul-style) on DiverseVul test pairs -- identical pairs EXCLUDED (level: ws)

Pair = vulnerable + patched version of the same function (same commit, same name), both in test. Validation max-F1 threshold per run. P-C correct pair · P-V both flagged · P-B neither · P-R reversed · rank-acc = P(score vuln > score patched), chance 0.5. Mean over seeds; 95% bootstrap CI over pairs (1000).

| arm | seeds | pairs (ambiguous skipped) | P-C | P-V | P-B | P-R | rank-acc |
|---|---|---|---|---|---|---|---|
| random_control | 4,5 | 142 (31) | 0.056 [0.032, 0.088] | 0.394 [0.313, 0.461] | 0.514 [0.440, 0.595] | 0.035 [0.014, 0.060] | 0.556 [0.493, 0.620] |
| family_disjoint | 4,5 | 637 (36) | 0.035 [0.024, 0.048] | 0.357 [0.326, 0.391] | 0.600 [0.564, 0.631] | 0.008 [0.003, 0.013] | 0.509 [0.486, 0.532] |
| project_disjoint | 1,2,3 | 1,036 (106) | 0.005 [0.002, 0.009] | 0.217 [0.195, 0.238] | 0.767 [0.746, 0.790] | 0.010 [0.006, 0.015] | 0.396 [0.374, 0.417] |
| commit_disjoint | 1,2,3 | 1,483 (75) | 0.013 [0.009, 0.017] | 0.253 [0.234, 0.273] | 0.715 [0.695, 0.734] | 0.019 [0.015, 0.024] | 0.436 [0.421, 0.452] |

Length check -- a length-only scorer ranks the LONGER function as vulnerable; rank-acc split by whether the patched version is longer than the vulnerable one.

| arm | patched longer | length-only rank-acc | rank-acc, patched longer | rank-acc, patched not longer | length-balanced rank-acc |
|---|---|---|---|---|---|
| random_control | 0.690 | 0.303 | 0.480 (n=98) | 0.727 (n=44) | 0.603 [0.540, 0.663] |
| family_disjoint | 0.768 | 0.206 | 0.475 (n=489) | 0.618 (n=148) | 0.547 [0.518, 0.576] |
| project_disjoint | 0.782 | 0.199 | 0.330 (n=810) | 0.629 (n=226) | 0.480 [0.455, 0.503] |
| commit_disjoint | 0.796 | 0.196 | 0.394 (n=1180) | 0.600 (n=303) | 0.497 [0.477, 0.516] |

Identical pairs excluded per arm: random_control 2, family_disjoint 51, project_disjoint 201, commit_disjoint 32
