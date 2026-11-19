# RQ0 — Clone-threshold calibration on BigCloneBench, our pipeline

Source: `$CFA_BCB_DIR` (function source `id2sourcecode.zip`, pair files type-1..5, nonclone). Score = exact Jaccard of 5-token shingles after `normalize_tokens` (`clone_family_audit.py`); Type-1 median = 1.000.

| BCB category | n | mean | median | @θ=0.6 | @θ=0.7 | @θ=0.8 | @θ=0.9 |
|---|---|---|---|---|---|---|---|
| Type-1 exact | 48,116 | 1.000 | 1.000 | 100.0% | 100.0% | 100.0% | 100.0% |
| Type-2 renamed | 4,234 | 0.997 | 1.000 | 99.8% | 99.6% | 99.5% | 99.1% |
| Strong Type-3 (ST3) | 21,395 | 0.422 | 0.326 | 34.5% | 30.0% | 22.5% | 15.6% |
|   of which VST3 (BCB sim >= 0.9) | 4,577 | 0.638 | 0.803 | 59.5% | 57.3% | 50.1% | 43.9% |
|   of which ST3 proper (BCB sim 0.7-0.9) | 16,818 | 0.363 | 0.217 | 27.7% | 22.6% | 15.0% | 7.9% |
| Moderate Type-3 (MT3) | 86,341 | 0.087 | 0.050 | 0.7% | 0.3% | 0.0% | 0.0% |
| Weak Type-3 / Type-4 | 109,914 | 0.014 | 0.008 | 0.0% | 0.0% | 0.0% | 0.0% |
| non-clone | 279,032 | 0.008 | 0.003 | 0.0% | 0.0% | 0.0% | 0.0% |

## Difference from the TouSimCC calibration (`results/rq0_calibration/`)

Capture in percentage points, ours − TouSimCC.

| BCB category | Δ@θ=0.6 | Δ@θ=0.7 | Δ@θ=0.8 | Δ@θ=0.9 | matched pairs | median Δ Jaccard | r | same decision @0.7 |
|---|---|---|---|---|---|---|---|---|
| Type-1 exact | +0.0 | +0.0 | +0.0 | +0.0 | 48,111 | +0.000 | 0.9996 | 100.0% |
| Type-2 renamed | -0.2 | -0.4 | -0.5 | -0.8 | 4,232 | +0.000 | 0.1135 | 99.6% |
| Strong Type-3 (ST3) | -65.5 | -69.0 | -67.6 | -33.6 | 21,385 | -0.559 | 0.4193 | 30.7% |
|   of which VST3 (BCB sim >= 0.9) | -40.5 | -42.7 | -49.9 | -53.2 | n/a | n/a | n/a | n/a |
|   of which ST3 proper (BCB sim 0.7-0.9) | -72.3 | -76.2 | -72.4 | -28.3 | n/a | n/a | n/a | n/a |
| Moderate Type-3 (MT3) | -92.1 | -66.8 | -24.3 | -2.1 | 86,125 | -0.666 | 0.2034 | 33.2% |
| Weak Type-3 / Type-4 | -22.3 | -6.6 | -0.5 | -0.0 | 109,773 | -0.418 | 0.2583 | 93.4% |
| non-clone | -12.9 | -3.2 | -0.3 | -0.0 | 278,846 | -0.333 | 0.2651 | 96.8% |

Sub-rows: TouSimCC capture recomputed on the matched pairs of that band (`results/rq0_calibration/` has no band split).

Capture here is the family-verification criterion (exact Jaccard ≥ θ). LSH candidate recall is not modelled, so these are upper bounds on what the audit links.

## Mapping check (BCB similarity bands, min(sim_line, sim_token))

| category | band | min | max | share inside |
|---|---|---|---|---|
| Strong Type-3 (ST3) | [0.7, 1.0] | 0.7 | 1.0 | 100.00% |
| Moderate Type-3 (MT3) | [0.5, 0.7) | 0.5 | 0.6992 | 100.00% |
| Weak Type-3 / Type-4 | [0.0, 0.5) | 0.0014 | 0.4953 | 100.00% |

## Parsing diagnostics

| file | pairs in file | bad rows | source missing | empty shingles | usable |
|---|---|---|---|---|---|
| Type-1 exact | 48,116 | 0 | 0 | 0 | 48,116 |
| Type-2 renamed | 4,234 | 0 | 0 | 0 | 4,234 |
| Strong Type-3 (ST3) | 21,395 | 0 | 0 | 0 | 21,395 |
| Moderate Type-3 (MT3) | 86,341 | 0 | 0 | 0 | 86,341 |
| Weak Type-3 / Type-4 | 109,914 | 0 | 0 | 15 | 109,914 |
| non-clone | 279,033 | 1 | 0 | 14 | 279,032 |
