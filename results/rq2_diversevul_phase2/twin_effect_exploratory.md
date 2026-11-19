# Own-twin effect on CodeBERT predictions (DiverseVul) -- EXPLORATORY v2

Twin = same commit + same function name, opposite label. Groups by where the twin is: train / elsewhere (valid or test) / none. Threshold = validation max-F1 per run. 'std' = length-standardised (within length quintiles, weighted by the exposed group). CIs: percentile bootstrap, 1000 resamples, mean over seeds.

## random_control (seeds 4, 5)

### positives (recall)

| twin location | n | recall | mean prob | mean length (chars) |
|---|---|---|---|---|
| twin_in_train | 1,312 | 0.3815 | 0.3122 | 3330 |
| twin_elsewhere | 270 | 0.3722 | 0.3043 | 3648 |
| no_twin | 319 | 0.4937 | 0.4033 | 4119 |
| twin_in_train & famleak | 635 | 0.4197 | 0.3352 | 3890 |
| twin_in_train & no famleak | 677 | 0.3456 | 0.2907 | 2805 |

- recall train vs elsewhere: raw **+0.0093** [-0.0468, +0.0685] · length-std **+0.0135** [-0.0409, +0.0691]
- recall train vs none: raw **-0.1123** [-0.1661, -0.0566] · length-std **-0.1085** [-0.1581, -0.0572]

### negatives (FPR)

| twin location | n | FPR | mean prob | mean length (chars) |
|---|---|---|---|---|
| twin_in_train | 1,819 | 0.4258 | 0.3337 | 2913 |
| twin_elsewhere | 290 | 0.3638 | 0.2929 | 3629 |
| no_twin | 29,162 | 0.0483 | 0.0741 | 1038 |

- FPR train vs elsewhere: raw **+0.0620** [+0.0094, +0.1174] · length-std **+0.0966** [+0.0524, +0.1458]
- FPR train vs none: raw **+0.3775** [+0.3549, +0.3978] · length-std **+0.3298** [+0.3090, +0.3495]

## family_disjoint (seeds 4, 5)

### positives (recall)

| twin location | n | recall | mean prob | mean length (chars) |
|---|---|---|---|---|
| twin_in_train | 870 | 0.3471 | 0.2289 | 2975 |
| twin_elsewhere | 750 | 0.3807 | 0.2297 | 3722 |
| no_twin | 271 | 0.5424 | 0.3572 | 3231 |

- recall train vs elsewhere: raw **-0.0335** [-0.0751, +0.0077] · length-std **-0.0003** [-0.0387, +0.0385]
- recall train vs none: raw **-0.1953** [-0.2551, -0.1413] · length-std **-0.1635** [-0.2219, -0.1076]

### negatives (FPR)

| twin location | n | FPR | mean prob | mean length (chars) |
|---|---|---|---|---|
| twin_in_train | 1,292 | 0.3924 | 0.2502 | 2161 |
| twin_elsewhere | 778 | 0.3535 | 0.2077 | 3772 |
| no_twin | 29,211 | 0.0439 | 0.0305 | 1048 |

- FPR train vs elsewhere: raw **+0.0389** [+0.0033, +0.0790] · length-std **+0.1119** [+0.0798, +0.1468]
- FPR train vs none: raw **+0.3486** [+0.3235, +0.3734] · length-std **+0.3216** [+0.2981, +0.3461]

