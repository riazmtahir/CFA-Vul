# Clone-family contamination audit

Input: {'train': 264392, 'valid': 33049, 'test': 33050}  |  representatives: 284746

| theta | families (>1) | span train+test | leaked test fns | leaked test % | contradictory labels | cross-project fams |
|---|---|---|---|---|---|---|
| 0.7 | 49904 | 11905 | 18688 | 56.54 | 12027 | 9116 |
| 0.8 | 51068 | 11801 | 16084 | 48.67 | 12133 | 9247 |
| 0.9 | 46788 | 10408 | 12812 | 38.77 | 10968 | 7950 |

## Split arms -- RQ2 must train on all three

The random control holds partition sizes and label balance fixed while
ignoring families, so it retains the leakage. Comparing published vs
family-disjoint alone would confound removed leakage with a different
test set; the control separates them.

| theta | arm | train | valid | test | test pos rate | leaked test % |
|---|---|---|---|---|---|---|
| 0.7 | published | 264392 | 33049 | 33050 | — | 56.54 |
| 0.7 | random control | 264218 | 33834 | 32439 | 0.0573 | 56.29 |
| 0.7 | family-disjoint | 264218 | 33834 | 32439 | 0.0589 | 0.0 |
| 0.8 | published | 264392 | 33049 | 33050 | — | 48.67 |
| 0.8 | random control | 264059 | 33727 | 32705 | 0.0573 | 48.72 |
| 0.8 | family-disjoint | 264059 | 33727 | 32705 | 0.0582 | 0.0 |
| 0.9 | published | 264392 | 33049 | 33050 | — | 38.77 |
| 0.9 | random control | 265247 | 32423 | 32821 | 0.0573 | 38.73 |
| 0.9 | family-disjoint | 265247 | 32423 | 32821 | 0.0579 | 0.0 |
