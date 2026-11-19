# RQ0 — Clone-threshold calibration on BigCloneBench

Source: `$CFA_TOUSIMCC_SIM_DIR` (column `t1_sim` = Jaccard; confirmed by Type-1 median = 1.000).

| BCB category | n | mean | median | @θ=0.6 | @θ=0.7 | @θ=0.8 | @θ=0.9 |
|---|---|---|---|---|---|---|---|
| Type-1 exact | 48,111 | 1.000 | 1.000 | 100.0% | 100.0% | 100.0% | 100.0% |
| Type-2 renamed | 4,232 | 0.993 | 1.000 | 100.0% | 100.0% | 100.0% | 99.9% |
| Strong Type-3 (ST3) | 21,385 | 0.895 | 0.897 | 100.0% | 99.0% | 90.1% | 49.3% |
| Moderate Type-3 (MT3) | 86,125 | 0.736 | 0.740 | 92.8% | 67.2% | 24.3% | 2.1% |
| Weak Type-3 / Type-4 | 109,773 | 0.423 | 0.432 | 22.3% | 6.6% | 0.5% | 0.0% |
| non-clone | 278,846 | 0.353 | 0.342 | 12.9% | 3.2% | 0.3% | 0.0% |

**Every contamination figure computed at these thresholds is a strict lower bound:** weak Type-3 and Type-4 (semantic) clones fall below any usable threshold and are therefore not counted.

## Parsing diagnostics

| file | rows read | unparseable | out of range | usable |
|---|---|---|---|---|
| Type-1 exact | 48,115 | 4 | 0 | 48,111 |
| Type-2 renamed | 4,233 | 1 | 0 | 4,232 |
| Strong Type-3 (ST3) | 21,394 | 9 | 0 | 21,385 |
| Moderate Type-3 (MT3) | 86,340 | 215 | 0 | 86,125 |
| Weak Type-3 / Type-4 | 109,913 | 140 | 0 | 109,773 |
| non-clone | 279,032 | 186 | 0 | 278,846 |
