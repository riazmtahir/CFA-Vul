# Paired evaluation (PrimeVul-style) on DiverseVul test pairs

Pair = vulnerable + patched version of the same function (same commit, same name), both in test. Validation max-F1 threshold per run. P-C correct pair · P-V both flagged · P-B neither · P-R reversed · rank-acc = P(score vuln > score patched), chance 0.5. Mean over seeds; 95% bootstrap CI over pairs (1000).

| arm | seeds | pairs (ambiguous skipped) | P-C | P-V | P-B | P-R | rank-acc |
|---|---|---|---|---|---|---|---|
| random_control | 4,5 | 144 (31) | 0.056 [0.031, 0.087] | 0.389 [0.312, 0.458] | 0.521 [0.448, 0.597] | 0.035 [0.014, 0.059] | 0.556 [0.497, 0.618] |
| family_disjoint | 4,5 | 688 (36) | 0.036 [0.025, 0.049] | 0.368 [0.339, 0.400] | 0.588 [0.553, 0.618] | 0.008 [0.004, 0.013] | 0.515 [0.491, 0.537] |
| project_disjoint | 1,2,3 | 1,237 (106) | 0.005 [0.002, 0.008] | 0.196 [0.176, 0.215] | 0.790 [0.770, 0.811] | 0.009 [0.005, 0.012] | 0.416 [0.396, 0.435] |
| commit_disjoint | 1,2,3 | 1,515 (75) | 0.013 [0.010, 0.017] | 0.255 [0.237, 0.273] | 0.713 [0.692, 0.732] | 0.019 [0.015, 0.024] | 0.438 [0.424, 0.454] |

Length check -- a length-only scorer ranks the LONGER function as vulnerable; rank-acc split by whether the patched version is longer than the vulnerable one.

| arm | patched longer | length-only rank-acc | rank-acc, patched longer | rank-acc, patched not longer | length-balanced rank-acc |
|---|---|---|---|---|---|
| random_control | 0.688 | 0.306 | 0.480 (n=99) | 0.722 (n=45) | 0.601 [0.542, 0.668] |
| family_disjoint | 0.727 | 0.249 | 0.476 (n=500) | 0.617 (n=188) | 0.547 [0.521, 0.573] |
| project_disjoint | 0.656 | 0.329 | 0.331 (n=811) | 0.577 (n=426) | 0.454 [0.434, 0.473] |
| commit_disjoint | 0.781 | 0.211 | 0.394 (n=1183) | 0.593 (n=332) | 0.494 [0.474, 0.512] |
