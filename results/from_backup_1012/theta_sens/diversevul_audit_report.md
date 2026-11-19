# Clone-family contamination audit

Input: {'train': 264392, 'valid': 33049, 'test': 33050}  |  representatives: 287485

| theta | families (>1) | span train+test | leaked test fns | leaked test % | contradictory labels | cross-project fams |
|---|---|---|---|---|---|---|
| 0.8 | 35017 | 7369 | 8797 | 26.62 | 6633 | 5509 |
| 0.9 | 30787 | 6473 | 7695 | 23.28 | 5348 | 4789 |

## Split arms -- RQ2 must train on all three

The random control holds partition sizes and label balance fixed while
ignoring families, so it retains the leakage. Comparing published vs
family-disjoint alone would confound removed leakage with a different
test set; the control separates them.

| theta | arm | train | valid | test | test pos rate | leaked test % |
|---|---|---|---|---|---|---|
| 0.8 | published | 264392 | 33049 | 33050 | — | 26.62 |
| 0.8 | random control | 264441 | 33011 | 33039 | 0.0573 | 26.54 |
| 0.8 | family-disjoint | 264441 | 33011 | 33039 | 0.0574 | 0.0 |
| 0.9 | published | 264392 | 33049 | 33050 | — | 23.28 |
| 0.9 | random control | 264990 | 32843 | 32658 | 0.0573 | 23.16 |
| 0.9 | family-disjoint | 264990 | 32843 | 32658 | 0.0584 | 0.0 |
