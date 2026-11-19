# Clone-family contamination audit

Input: {'train': 184427, 'valid': 25430, 'test': 25911}  |  representatives: 199610

| theta | families (>1) | span train+test | leaked test fns | leaked test % | contradictory labels | cross-project fams |
|---|---|---|---|---|---|---|
| 0.7 | 28331 | 2464 | 4069 | 15.7 | 2935 | 5161 |
| 0.8 | 26823 | 2328 | 3758 | 14.5 | 2578 | 4851 |
| 0.9 | 24719 | 2097 | 3378 | 13.04 | 2018 | 4480 |

## 1-NN clone-retrieval baseline (no learning)

| theta | coverage % | PR-AUC | test positive rate |
|---|---|---|---|
| 0.7 | 40.95 | 0.0944 | 0.0268 |
| 0.8 | 40.95 | 0.0944 | 0.0268 |
| 0.9 | 40.95 | 0.0944 | 0.0268 |

## Split arms -- RQ2 must train on all three

The random control holds partition sizes and label balance fixed while
ignoring families, so it retains the leakage. Comparing published vs
family-disjoint alone would confound removed leakage with a different
test set; the control separates them.

| theta | arm | train | valid | test | test pos rate | leaked test % |
|---|---|---|---|---|---|---|
| 0.7 | published | 184427 | 25430 | 25911 | — | 15.7 |
| 0.7 | random control | 187285 | 23834 | 24649 | 0.0296 | 30.05 |
| 0.7 | family-disjoint | 187285 | 23834 | 24649 | 0.0276 | 0.0 |
| 0.8 | published | 184427 | 25430 | 25911 | — | 14.5 |
| 0.8 | random control | 186791 | 23527 | 25450 | 0.0295 | 27.86 |
| 0.8 | family-disjoint | 186791 | 23527 | 25450 | 0.0263 | 0.0 |
| 0.9 | published | 184427 | 25430 | 25911 | — | 13.04 |
| 0.9 | random control | 190074 | 22961 | 22733 | 0.0295 | 25.5 |
| 0.9 | family-disjoint | 190074 | 22961 | 22733 | 0.0312 | 0.0 |
