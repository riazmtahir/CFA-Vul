# Clone-family contamination audit

Input: {'train': 21854, 'valid': 2732, 'test': 2732}  |  representatives: 25793

| theta | families (>1) | span train+test | leaked test fns | leaked test % | contradictory labels | cross-project fams |
|---|---|---|---|---|---|---|
| 0.8 | 1803 | 343 | 368 | 13.47 | 846 | 8 |
| 0.9 | 1495 | 278 | 298 | 10.91 | 622 | 8 |

## Split arms -- RQ2 must train on all three

The random control holds partition sizes and label balance fixed while
ignoring families, so it retains the leakage. Comparing published vs
family-disjoint alone would confound removed leakage with a different
test set; the control separates them.

| theta | arm | train | valid | test | test pos rate | leaked test % |
|---|---|---|---|---|---|---|
| 0.8 | published | 21854 | 2732 | 2732 | — | 13.47 |
| 0.8 | random control | 21904 | 2706 | 2708 | 0.4561 | 13.04 |
| 0.8 | family-disjoint | 21904 | 2706 | 2708 | 0.4623 | 0.0 |
| 0.9 | published | 21854 | 2732 | 2732 | — | 10.91 |
| 0.9 | random control | 21893 | 2658 | 2767 | 0.4561 | 10.99 |
| 0.9 | family-disjoint | 21893 | 2658 | 2767 | 0.4478 | 0.0 |
