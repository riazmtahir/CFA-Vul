# RQ0 — Clone-threshold calibration on BigCloneBench, our pipeline

Source: `$CFA_BCB_DIR` (function source `id2sourcecode.zip`, pair files type-1..5, nonclone). Score = exact Jaccard of 3-token shingles after `blind_tokens` (every identifier -> ID; **EXPLORATORY high-recall variant**) (`clone_family_audit.py`); Type-1 median = 1.000.

| BCB category | n | mean | median | @θ=0.6 | @θ=0.7 | @θ=0.8 | @θ=0.9 |
|---|---|---|---|---|---|---|---|
| Type-1 exact | 48,116 | 1.000 | 1.000 | 100.0% | 100.0% | 100.0% | 100.0% |
| Type-2 renamed | 4,234 | 0.998 | 1.000 | 100.0% | 99.8% | 99.8% | 99.7% |
| Strong Type-3 (ST3) | 21,395 | 0.844 | 0.855 | 96.8% | 88.2% | 69.6% | 37.8% |
|   of which VST3 (BCB sim >= 0.9) | 4,577 | 0.956 | 0.969 | 99.9% | 99.9% | 98.6% | 93.7% |
|   of which ST3 proper (BCB sim 0.7-0.9) | 16,818 | 0.814 | 0.831 | 96.0% | 85.0% | 61.8% | 22.6% |
| Moderate Type-3 (MT3) | 86,341 | 0.552 | 0.542 | 33.4% | 12.2% | 3.2% | 0.7% |
| Weak Type-3 / Type-4 | 109,914 | 0.300 | 0.292 | 0.5% | 0.1% | 0.0% | 0.0% |
| non-clone | 279,032 | 0.210 | 0.208 | 0.2% | 0.1% | 0.0% | 0.0% |

## Difference from the TouSimCC calibration (`results/rq0_calibration/`)

Capture in percentage points, ours − TouSimCC.

| BCB category | Δ@θ=0.6 | Δ@θ=0.7 | Δ@θ=0.8 | Δ@θ=0.9 | matched pairs | median Δ Jaccard | r | same decision @0.7 |
|---|---|---|---|---|---|---|---|---|
| Type-1 exact | +0.0 | +0.0 | +0.0 | +0.0 | 48,111 | +0.000 | 0.9996 | 100.0% |
| Type-2 renamed | -0.0 | -0.2 | -0.2 | -0.2 | 4,232 | +0.000 | 0.1496 | 99.8% |
| Strong Type-3 (ST3) | -3.2 | -10.8 | -20.5 | -11.4 | 21,385 | -0.030 | 0.5751 | 87.8% |
|   of which VST3 (BCB sim >= 0.9) | -0.1 | -0.1 | -1.4 | -3.3 | n/a | n/a | n/a | n/a |
|   of which ST3 proper (BCB sim 0.7-0.9) | -4.0 | -13.7 | -25.6 | -13.7 | n/a | n/a | n/a | n/a |
| Moderate Type-3 (MT3) | -59.5 | -55.0 | -21.1 | -1.4 | 86,125 | -0.193 | 0.4253 | 40.9% |
| Weak Type-3 / Type-4 | -21.9 | -6.5 | -0.5 | -0.0 | 109,773 | -0.121 | 0.5353 | 93.4% |
| non-clone | -12.7 | -3.2 | -0.3 | -0.0 | 278,846 | -0.132 | 0.5493 | 96.8% |

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
