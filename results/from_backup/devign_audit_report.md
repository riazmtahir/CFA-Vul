# Clone-family contamination audit

Input: {'train': 21854, 'valid': 2732, 'test': 2732}  |  representatives: 25793

| theta | families (>1) | span train+test | leaked test fns | leaked test % | contradictory labels | cross-project fams |
|---|---|---|---|---|---|---|
| 0.7 | 2031 | 384 | 419 | 15.34 | 1001 | 9 |

## 1-NN clone-retrieval baseline (no learning)

| theta | coverage % | PR-AUC | test positive rate |
|---|---|---|---|
| 0.7 | 40.26 | 0.5234 | 0.4594 |

## Split arms -- RQ2 must train on all three

The random control holds partition sizes and label balance fixed while
ignoring families, so it retains the leakage. Comparing published vs
family-disjoint alone would confound removed leakage with a different
test set; the control separates them.

| theta | arm | train | valid | test | test pos rate | leaked test % |
|---|---|---|---|---|---|---|
| 0.7 | published | 21854 | 2732 | 2732 | — | 15.34 |
| 0.7 | random control | 21807 | 2773 | 2738 | 0.4562 | 15.01 |
| 0.7 | family-disjoint | 21807 | 2773 | 2738 | 0.4547 | 0.0 |
