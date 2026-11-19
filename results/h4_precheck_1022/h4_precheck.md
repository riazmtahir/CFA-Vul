# H4 pre-check (commit_hrfamily_disjoint vs commit_disjoint)

Functions 330,491; groups 753; largest group 321,171 (97.18% of corpus); top 10 [321171, 223, 139, 120, 93, 92, 80, 78, 78, 74]

| quantity | commit_hrfamily_disjoint | commit_disjoint |
|---|---|---|
| train functions / pos rate / mean chars | 321,171 / 0.0543 / 1287 | 264,775 / 0.0572 / 1325 |
| valid functions / pos rate / mean chars | 4,885 / 0.1529 / 2036 | 32,544 / 0.0587 / 1292 |
| test functions / pos rate / mean chars | 4,435 / 0.1693 / 2472 | 33,172 / 0.0570 / 1245 |
| test sharing a commit with train | 0 | 0 |
| test with own twin in train | 0 | 0 |
| high-recall family leak: any / same-label % | 0.00 / 0.00 | 38.38 / 37.01 |
| strict family leak: any / same-label % | 0.07 / 0.05 | 17.12 / 16.42 |
| length-only floor PR-AUC (test) | 0.2825 | 0.1413 |

**FAIL**: train size 321,171 vs 264,775 (21.30% > 2%); valid size 4,885 vs 32,544 (84.99% > 2%); valid positive rate 0.1529 vs 0.0587 (> 0.003); test size 4,435 vs 33,172 (86.63% > 2%); test positive rate 0.1693 vs 0.0570 (> 0.003); strict family leak: 3 test functions (must be 0); largest group 321,171 functions = 97.18% of corpus (> 15%)
