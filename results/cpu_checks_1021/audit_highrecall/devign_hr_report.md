# Clone-family contamination audit

Input: {'train': 21854, 'valid': 2732, 'test': 2732}  |  representatives: 25689

| theta | families (>1) | span train+test | leaked test fns | leaked test % | contradictory labels | cross-project fams |
|---|---|---|---|---|---|---|
| 0.7 | 4262 | 956 | 1190 | 43.56 | 2662 | 31 |
| 0.8 | 3906 | 833 | 943 | 34.52 | 2384 | 26 |
| 0.9 | 2939 | 589 | 642 | 23.5 | 1667 | 12 |

## Split arms -- RQ2 must train on all three

The random control holds partition sizes and label balance fixed while
ignoring families, so it retains the leakage. Comparing published vs
family-disjoint alone would confound removed leakage with a different
test set; the control separates them.

| theta | arm | train | valid | test | test pos rate | leaked test % |
|---|---|---|---|---|---|---|
| 0.7 | published | 21854 | 2732 | 2732 | — | 43.56 |
| 0.7 | random control | 21867 | 2705 | 2746 | 0.4559 | 44.06 |
| 0.7 | family-disjoint | 21867 | 2705 | 2746 | 0.4527 | 0.0 |
| 0.8 | published | 21854 | 2732 | 2732 | — | 34.52 |
| 0.8 | random control | 21857 | 2788 | 2673 | 0.456 | 35.43 |
| 0.8 | family-disjoint | 21857 | 2788 | 2673 | 0.4714 | 0.0 |
| 0.9 | published | 21854 | 2732 | 2732 | — | 23.5 |
| 0.9 | random control | 21859 | 2698 | 2761 | 0.456 | 23.54 |
| 0.9 | family-disjoint | 21859 | 2698 | 2761 | 0.4491 | 0.0 |
