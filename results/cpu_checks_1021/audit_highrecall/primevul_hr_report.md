# Clone-family contamination audit

Input: {'train': 184427, 'valid': 25430, 'test': 25911}  |  representatives: 197952

| theta | families (>1) | span train+test | leaked test fns | leaked test % | contradictory labels | cross-project fams |
|---|---|---|---|---|---|---|
| 0.7 | 34364 | 3338 | 8086 | 31.21 | 4832 | 6612 |
| 0.8 | 35149 | 3465 | 6694 | 25.83 | 4767 | 6847 |
| 0.9 | 32570 | 3110 | 5178 | 19.98 | 4187 | 6142 |

## Split arms -- RQ2 must train on all three

The random control holds partition sizes and label balance fixed while
ignoring families, so it retains the leakage. Comparing published vs
family-disjoint alone would confound removed leakage with a different
test set; the control separates them.

| theta | arm | train | valid | test | test pos rate | leaked test % |
|---|---|---|---|---|---|---|
| 0.7 | published | 184427 | 25430 | 25911 | — | 31.21 |
| 0.7 | random control | 186949 | 24511 | 24308 | 0.0296 | 51.61 |
| 0.7 | family-disjoint | 186949 | 24511 | 24308 | 0.0282 | 0.0 |
| 0.8 | published | 184427 | 25430 | 25911 | — | 25.83 |
| 0.8 | random control | 188506 | 23225 | 24037 | 0.0296 | 44.81 |
| 0.8 | family-disjoint | 188506 | 23225 | 24037 | 0.0287 | 0.0 |
| 0.9 | published | 184427 | 25430 | 25911 | — | 19.98 |
| 0.9 | random control | 188064 | 24010 | 23694 | 0.0295 | 36.5 |
| 0.9 | family-disjoint | 188064 | 24010 | 23694 | 0.0293 | 0.0 |
