# CFA-Vul: replication package

Replication package for the paper *Clone families, commits and projects: decomposing train–test leakage
in C/C++ vulnerability detection benchmarks* (M. T. Riaz, W. Kong, V. H. Balusu, R. M. Yaqoob).

It contains the code that audits clone-family leakage in Devign, DiverseVul and PrimeVul, builds the
leakage-controlled splits, fine-tunes CodeBERT (and GraphCodeBERT on Devign), and analyzes the results;
the result files of every run reported in the paper; and the scripts that recompute every reported number
from those files and regenerate the tables and figures.

## What you can reproduce, and how

| Level | What | Needs | Time |
|---|---|---|---|
| 1. Quick check | every number, table and figure of the paper and the supplement, recomputed from the shipped result files | Python 3.10, `requirements.txt`, CPU | 1–2 min |
| 2. Audits and splits | clone families, leak rates, split files, commit exposure, pre-checks | the corpora (below), CPU; the largest step (H4/H4b pre-check) needed about 4–6 GB RAM | minutes to about 1.5 h per step |
| 3. Training | the 25 DiverseVul runs and 18 Devign runs | the corpora, one 24 GB GPU (the runs used an RTX 4090 / 4090 D) | 2–4 GPU-hours per DiverseVul run (the nine runs of `run_stage_b.sh` took about 25 GPU-hours) |

### 1. Quick check (no corpus, no GPU)

```bash
python -m pip install -r requirements.txt
python analysis/run_all.py
```

`run_all.py` first runs the checking scripts (`analysis/scripts/*_numbers.py`, `effect_sizes.py`). Each one
recomputes its numbers from `results/` and exits with an error if any value differs from the value
reported in the paper. It then regenerates the tables (`analysis/tables/`, LaTeX), the figures
(`analysis/figures/*.pdf`; SVG and PNG at 300 and 600 dpi in `analysis/figures/export/`) and the
supplementary tables and figures (`analysis/supplementary/`). The last line reads `ALL CHECKS PASSED`.
Figures use the Arial font; without it matplotlib falls back to another sans-serif font.

Two checks need material that is not in this package and are skipped with a message: the checks of the
study plan's text in `design_numbers.py` (the plan stays in the development repository; commit times
come from `PROTOCOL_HISTORY.md`) and the checks of the wording of the paper's answer boxes in
`revision_numbers.py` (the manuscript is not part of the package).

### 2–3. Rerunning the pipeline

Set the environment variables, obtain the corpora and run the scripts in `pipeline/` in the order of the
table further below.

| Variable | Meaning | Default |
|---|---|---|
| `CFA_DATA_DIR` | data and output folder (corpora under `data/`, split files, run outputs) | required by the `run_*.sh` scripts; `data` in the Python scripts |
| `CFA_CODE_DIR` | folder holding the pipeline code | the folder of the run script (`pipeline/`) |
| `CFA_PYTHON` | Python interpreter | `python` |
| `HF_ENDPOINT`, `HF_HOME` | Hugging Face endpoint and cache | `https://huggingface.co`, `$CFA_DATA_DIR/hf` |
| `CFA_BCB_DIR`, `CFA_TOUSIMCC_SIM_DIR` | BigCloneBench files and TouSimCC similarity scores (θ characterization only) | `data/BCB`, `data/TouSimCC_output_16_sim` |

The `run_*.sh` scripts were written for a single Linux GPU server. Some wait for an approval file in
`$CFA_DATA_DIR` before they start (for example `.stage_b_approved`); this is how the project made sure each
batch started only after its decision rule was fixed. Create the file to start a batch. Every run resumes
from its last checkpoint if interrupted, and finished runs are skipped.

```bash
export CFA_DATA_DIR=/path/to/data
python -m pip install -r requirements-pipeline.txt
bash pipeline/setup.sh                       # Devign from Hugging Face; caches CodeBERT, GraphCodeBERT, UniXcoder
python pipeline/prep_diversevul_hf.py        # DiverseVul from the Hugging Face mirror (pin the revision below)
```

## Obtaining the corpora

The corpora are not redistributed. Obtain them from their original sources, under their terms:

| Corpus | Source | Revision used | Files expected in `$CFA_DATA_DIR/data/` |
|---|---|---|---|
| Devign (CodeXGLUE defect detection) | https://huggingface.co/datasets/google/code_x_glue_cc_defect_detection | `69bd48c03223c2104342acd9a807caf61ac3efb8` | `devign/devign_all.jsonl` (27,318 functions; written by `setup.sh`) |
| DiverseVul (unofficial Hugging Face mirror, whose 80/10/10 split is the "published" split) | https://huggingface.co/datasets/bstee615/diversevul | `3ed5dae8fdf3f6026f1e260f789656c4bb6d98e6` | `diversevul/diversevul.jsonl` (330,492 functions; written by `prep_diversevul_hf.py`) |
| PrimeVul (original release; audit only) | https://github.com/DLVulDet/PrimeVul (data linked from there) | release of the paper | `primevul/` with the six JSONL files (train/valid/test and the `_paired` files); `primevul_tools.py prep` merges and checks the counts |
| BigCloneBench (θ characterization only) | Svajlenko et al., ICSME 2014 (BigCloneEval distribution) | – | `$CFA_BCB_DIR`: `id2sourcecode.zip`, `type-1.csv` … `type-5.csv`, `nonclone.csv` |

`prep_diversevul_hf.py` numbers functions sequentially over train, validation and test; one empty function
is dropped from training by `finetune_vd.py`, which gives 264,392 training functions.

## Split files

The split files and clone-family maps used by the training runs are listed in `splits/README.md` with the
MD5 checksums recorded at run time. They are deterministic outputs of the pipeline (seed 42) and can be
rebuilt from the corpora; the `*_report.json` files in `results/` record the partition sizes each rebuilt
file must reproduce.

## Pipeline steps and where their results are

| Step | Script(s) in `pipeline/` | Results in `results/` |
|---|---|---|
| θ characterization on BigCloneBench | `../analysis/scripts/calibrate_theta_ours.py` | `rq0_calibration_ours/` (TouSimCC comparison input: `rq0_calibration/`) |
| Clone-family audit, Devign and DiverseVul (θ = 0.7) | `clone_family_audit.py` | `from_backup/*_audit_report.json` |
| Pilot protocol (3 epochs, unweighted loss; reported only as a pilot) and discarded Gate 1 on Devign | `run_rq2_diversevul.sh`, `run_gate1.sh`, `run_gate1_converged.sh` | `from_backup/rq2_diversevul/`, `from_backup/rq2_devign/`, `from_backup_1012/rq2_devign_converged/` |
| Learning-rate selection | `run_stage_a.sh`, `pick_lr.py` | `from_backup/sweep_*` |
| Devign final runs (CodeBERT, GraphCodeBERT) | `run_stage_a2.sh` | `from_backup/rq2_devign_final/` |
| DiverseVul published / random control / family-disjoint, seeds 1–3 | `run_stage_b.sh`, `summarize_stage_b.py` | `rq2_diversevul_final/` |
| Family-label audit, project-disjoint split | `run_phase1.sh` (`family_label_audit.py`, `make_project_split.py`) | `from_backup_1012/family_label/` |
| Project-disjoint s1–3, random control and family-disjoint s4–5; H1, H2 | `run_phase2.sh`, `analyze_phase2.py` | `rq2_diversevul_phase2/` (incl. predictions `*.preds.json.gz`) |
| Commit exposure, counterparts, commit-disjoint split | `run_commit_leak.sh` (`commit_leak_audit.py`, `make_project_split.py --group-field commit_id`) | `audit_7oct/` |
| PrimeVul audit, θ sensitivity | `run_primevul_audit.sh`, `primevul_tools.py` | `audit_7oct/`, `from_backup_1012/theta_sens/` |
| Counterpart effect, paired evaluation | `twin_effect_exploratory.py`, `paired_eval.py` | `rq2_diversevul_phase2/` |
| Commit-disjoint s1–3; H3 | `run_phase3.sh`, `analyze_phase3.py` | `rq2_diversevul_phase3/` |
| CPU checks (corpus overlap, whitespace pairs, re-scoring) and high-recall audit | `run_cpu_checks.sh`, `run_audit_highrecall.sh` (`scripts/corpus_overlap.py`, `scripts/pair_identity_check.py`, `scripts/croft_rescore.py`) | `cpu_checks_1021/` |
| H4 and H4b pre-checks, direct clone edges, design selection | `run_h4_precheck.sh` (`scripts/make_commit_family_split.py`, `scripts/precheck_h4.py`, `scripts/direct_clone_edges.py`, `scripts/make_decon_splits.py`, `scripts/precheck_h4b.py`, `scripts/select_h4_design.py`) | `h4_precheck_1022/` |
| H4b runs (decontaminated vs matched stratified control) | `run_phase4b.sh`, `analyze_phase4b.py` (H4: `run_phase4.sh`, `analyze_phase4.py`; not run, gate failed) | `rq2_diversevul_phase4b/` |
| Post hoc selection sensitivity (M1) and same-commit analysis (Note S1) | `run_m1_sensitivity.sh`, `scripts/m1_selection_replay.py`, `scripts/item57_commit_mates.py` | `m1_postwarmup_1112/`, `m1_selection_replay.*` |

Tests written before the real runs (synthetic data with known outcomes): `pipeline/scripts/tests/`
(`test_h4.py`, `test_h4b.py`, `test_cpu_checks.py`; `test_audit_default_identical.py` needs BigCloneBench).

## Table and figure map

Paper numbering. "Checked by" scripts recompute the numbers in the text around the table or figure as well.

| Item | Produced by (`analysis/scripts/`) | Reads | Checked by |
|---|---|---|---|
| Fig. 1 Study overview | `fig_overview.py` | – (method diagram) | – |
| Table 1 Related work | – (literature; manuscript only) | – | – |
| Table 2 Hypotheses and outcomes | `hypotheses_table.py` | `rq2_diversevul_phase{2,3,4b}/*_analysis.json`, `out/effect_sizes.json`, `out/rq2_numbers.json`, `out/h4b_numbers.json` | `effect_sizes.py`, `rq2_numbers.py`, `h4b_numbers.py` |
| Table 3 Corpora | `corpora_table.py` | audit reports in `results/` | `corpora_table.py` (built-in checks) |
| Table 4 Splits of DiverseVul | – (set in the manuscript) | per-run JSONs | `design_numbers.py` |
| Algorithms 1–2 | – | – | implemented in `pipeline/clone_family_audit.py`, `pipeline/make_project_split.py` |
| Table 5 Clone-family audit | `rq1_table_figure.py` | `out/rq1_numbers.json` | `rq1_numbers.py`, `cpu_checks_numbers.py` |
| Fig. 2 Leakage against θ | `rq1_table_figure.py` | `out/rq1_numbers.json` | `rq1_numbers.py` |
| Table 6 Test PR-AUC per split | `rq2_tables_figure.py` | `out/rq2_numbers.json`, `out/h4b_numbers.json` | `rq2_numbers.py`, `h4b_numbers.py` |
| Table 7 Paired Δ | `rq2_tables_figure.py` | same | same |
| Fig. 3 Paired Δ | `rq2_tables_figure.py` | same | same |
| Fig. 4 Leakage profile | `fig_leakage_profile.py` | audit, exposure and direct-edge results | `fig_leakage_profile.py` (built-in check) |
| Table 8 Commit-level exposure | `rq3_tables_figure.py` | `out/rq3_numbers.json` | `rq3_numbers.py` |
| Table 9 Counterpart effect | `rq3_tables_figure.py` | `out/rq3_numbers.json` | `rq3_numbers.py` |
| Fig. 5 Rank accuracy on vulnerable/patched pairs | `rq3_tables_figure.py` | `out/rq3_numbers.json` | `rq3_numbers.py`, `cpu_checks_numbers.py` |
| Supp. Table S1 Per-seed results | `per_seed_table.py` | per-run JSONs | `per_seed_table.py` (means against `out/rq2_numbers.json`, `out/h4b_numbers.json`) |
| Supp. Table S2 Decision-rule record | – (from `PROTOCOL_HISTORY.md`) | `PROTOCOL_HISTORY.md`, `provenance.txt` files | `design_numbers.py` |
| Supp. Table S3 Selection after warm-up | `m1_numbers.py`, `supp_m1_item57.py` | `m1_postwarmup_1112/` | `m1_numbers.py`, `revision_numbers.py` |
| Supp. Table S4 Pair outcomes | `supp_m1_item57.py` | `rq2_diversevul_phase2/paired_eval.json` | `revision_numbers.py` |
| Supp. Note S1 | `supp_m1_item57.py` | `m1_postwarmup_1112/item57/item57.json` | `revision_numbers.py` |
| Supp. Fig. S1 Validation curves | `supp_validation_curves.py` | per-run JSONs (epoch histories) | – |
| Supp. Figs. S2–S4 | `supp_figures.py` | per-run JSONs, `cpu_checks_1021/`, `h4_precheck_1022/` | `supp_figures.py` (built-in checks) |

Numbers that appear only in the text (backup archives, CPU checks, the revision round) are checked by
`backup_numbers.py`, `cpu_checks_numbers.py`, `revision_numbers.py` and `design_numbers.py`.

## Hardware, software and non-determinism

Training used one NVIDIA RTX 4090 (the first nine DiverseVul runs and the Devign runs) or RTX 4090 D (the
other 16 DiverseVul runs and the reruns), with fp16 automatic mixed precision. The reruns of Supp. Table S3
used PyTorch 2.12.1+cu130; the versions of the other training libraries were not recorded. No deterministic
cuDNN setting was used, so GPU training is not bit-reproducible: rerunning with the same code, split, seed
and settings gives slightly different numbers (in the M1 rerun, up to 0.0081 test PR-AUC under the
original selection rule). The audits, split construction and all analyses are deterministic.

## Pre-specified decision rules

`PROTOCOL_HISTORY.md` records when each pre-specified decision rule (H1–H4, H4b) was fixed, exported from
the development repository's git log; the rules themselves are in the docstrings of
`pipeline/analyze_phase{2,3,4,4b}.py`.

## Contents and sizes

`pipeline/` (code, run scripts, tests), `analysis/` (checking, table and figure scripts; outputs are
regenerated), `results/` (result files, including the saved predictions of the Phase 2, Phase 3 and M1 runs; 16 MB in
all, no file above 1 MB), `splits/` (manifest), `PROTOCOL_HISTORY.md`, `LICENSE`, `requirements*.txt`.

Result files are copied unchanged from the study's server and keep the server's data paths
(`/root/autodl-tmp/...`) in their metadata fields; read these as `$CFA_DATA_DIR/...`. One exception: in the six
files of `results/rq0_calibration*/`, the local input folders of the θ characterization were replaced by
`$CFA_BCB_DIR` and `$CFA_TOUSIMCC_SIM_DIR`; no value changed.

## Licence

Code and documentation: MIT (see `LICENSE`). The corpora and BigCloneBench remain under the licences and
terms of their original sources.
