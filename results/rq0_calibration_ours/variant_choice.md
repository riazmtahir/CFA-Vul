# Choice of the high-recall variant for the exploratory audit (EXPLORATORY, 18 Oct 2026)

Rule (set in the 18 Oct request, before the full-BCB check below): highest Strong Type-3 capture
with non-clone capture ≤ 1% at its θ. Source: `decomposition_sample.json` (1,500 random pairs per
file, seed 1; the "type-3 file" is BCB's Strong Type-3 = VST3 + ST3). Cells: Strong Type-3 / MT3 /
non-clone capture, %.

| variant | θ = 0.7 | θ = 0.8 | θ = 0.9 |
|---|---|---|---|
| positional renaming, k=5 (audit default) | 30.3 / 0.3 / 0.0 | 22.7 / 0.0 / 0.0 | 16.1 / 0.0 / 0.0 |
| positional renaming, k=1 | 99.8 / 87.5 / **11.1** | 95.8 / 50.9 / **1.4** | 69.9 / 12.7 / 0.1 |
| raw identifiers, k=5 | 34.3 / 0.1 / 0.0 | 23.1 / 0.0 / 0.0 | 13.2 / 0.0 / 0.0 |
| raw identifiers, k=1 | 71.2 / 6.4 / 0.0 | 45.5 / 1.6 / 0.0 | 25.3 / 0.3 / 0.0 |
| blind identifiers, k=5 | 73.1 / 1.9 / 0.0 | 46.2 / 0.3 / 0.0 | 28.5 / 0.0 / 0.0 |
| **blind identifiers, k=3** | **90.0 / 12.0 / 0.1** | 71.7 / 3.2 / 0.1 | 39.0 / 0.6 / 0.1 |

Bold non-clone values break the ≤ 1% rule. Eligible cells ranked by Strong Type-3 capture: blind
k=3 @ 0.7 (90.0), blind k=5 @ 0.7 (73.1), blind k=3 @ 0.8 (71.7), raw k=1 @ 0.7 (71.2), positional
k=1 @ 0.9 (69.9). **Chosen: blind identifiers, k = 3, θ = 0.7.** It also admits 12.0% of Moderate
Type-3 pairs, so its families are looser than "near-duplicate".

Full-BCB check of the chosen variant (all pairs, `calibrate_theta_ours.py --rename blind
--shingle-k 3`, output `highrecall_blind_k3/`): at θ = 0.7 Strong Type-3 88.2% (VST3 99.9%,
ST3 proper 85.0%), MT3 12.2%, weak Type-3/Type-4 0.05%, non-clone 0.05% (n = 279,032); Type-1
100.0%, Type-2 99.8%. The sample estimate holds. A re-run of the default setting on all pairs
reproduced `rq0_theta_calibration_ours.{json,md}` exactly.

This variant is used only for the exploratory family-definition sensitivity audit
(`run_audit_highrecall.sh`). It does not define any training arm and re-tests no hypothesis.
