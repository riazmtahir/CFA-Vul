# Clone-family contamination audit

Input: {'train': 264392, 'valid': 33049, 'test': 33050}  |  representatives: 287485

| theta | families (>1) | span train+test | leaked test fns | leaked test % | contradictory labels | cross-project fams |
|---|---|---|---|---|---|---|
| 0.7 | 37954 | 8095 | 9787 | 29.61 | 7457 | 6064 |

## 1-NN clone-retrieval baseline (no learning)

| theta | coverage % | PR-AUC | test positive rate |
|---|---|---|---|
| 0.7 | 50.59 | 0.1553 | 0.0584 |

## Split arms -- RQ2 must train on all three

The random control holds partition sizes and label balance fixed while
ignoring families, so it retains the leakage. Comparing published vs
family-disjoint alone would confound removed leakage with a different
test set; the control separates them.

| theta | arm | train | valid | test | test pos rate | leaked test % |
|---|---|---|---|---|---|---|
| 0.7 | published | 264392 | 33049 | 33050 | — | 29.61 |
| 0.7 | random control | 264775 | 32544 | 33172 | 0.0573 | 29.39 |
| 0.7 | family-disjoint | 264775 | 32544 | 33172 | 0.057 | 0.0 |
