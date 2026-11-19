# Split files and clone-family maps

The training runs read the split and family files listed below. Their MD5 checksums are those recorded at
run time in the `provenance.txt` files under `results/` (Phase 2, Phase 3, Phase 4b and the M1 reruns).
After you place a file here, check it with `md5sum splits/<file>` (Linux) or
`Get-FileHash -Algorithm MD5 splits\<file>` (Windows).

| File | MD5 at run time | Used by |
|---|---|---|
| `diversevul_audit_splitfile_theta0.7.json` | `8b68b18284f4dce553a582112ff6d643` | family-disjoint split (DiverseVul) |
| `diversevul_audit_splitfile_randomctrl_theta0.7.json` | `7a936fd6139a33605f08afa506af321d` | random control (DiverseVul) |
| `diversevul_audit_splitfile_projectdisjoint_seed42.json` | `aae66bcb79411a8fb67b810116a2f982` | project-disjoint split |
| `diversevul_audit_splitfile_commitdisjoint_seed42.json` | `824ef7843bfaa2786210ea7725b15ba7` | commit-disjoint split |
| `diversevul_audit_splitfile_commitdecon_seed42.json` | `55662742499879fd17bd18ce48062c34` | H4b decontaminated training set |
| `diversevul_audit_splitfile_commitsizectrl_seed42.json` | `85c7cc23e1c32713c63cf92518ef1208` | H4b matched stratified control |
| `diversevul_audit_families_theta0.7.json` | `ba84d1def26d89f1b61cdd3b203fb8b7` | near-duplicate family map (DiverseVul) |
| `devign_audit_splitfile_theta0.7.json` | `db0e586918d8977c4c58dd59beffcba1` | Devign family-disjoint split |

The published splits need no split file: `finetune_vd.py --arm published` uses the `split` field of the
corpus file written by `pipeline/setup.sh` (Devign) or `pipeline/prep_diversevul_hf.py` (DiverseVul).
| `devign_audit_families_theta0.7.json` | `c7500a1c53786550f049ea49639736d1` | near-duplicate family map (Devign) |

The Devign random-control split file, the PrimeVul files and the high-recall families were not read by a
run that wrote a provenance file, so no run-time checksum is recorded for them.

All files are deterministic outputs of the pipeline (seed 42) and can be rebuilt from the corpora:
`pipeline/clone_family_audit.py` (families, published-split audit, random control and family-disjoint
split), `pipeline/make_project_split.py` (project-disjoint, and commit-disjoint with
`--group-field commit_id`, run by `pipeline/run_commit_leak.sh`) and `pipeline/scripts/make_decon_splits.py`
(the two H4b training sets, run by `pipeline/run_h4_precheck.sh`). The `*_report.json` files under
`results/` record the partition sizes and positive counts each rebuilt file must match.

The function identifiers in these files refer to the corpora as distributed by their sources (see the
main README); the corpora themselves are not redistributed.
