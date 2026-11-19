# Protocol history: when each pre-specified decision rule was fixed

Exported on 20 November 2026 from the git log of the study's development repository. That repository is
not published in full: it also holds the manuscript, planning notes and copyrighted literature. This
package therefore does not contain the commits themselves. The table gives, for every commit that fixed
or amended a pre-specified decision rule, the committer time (China Standard Time, UTC+8), the
abbreviated commit hash, the protocol files it changed and what it fixed. It is the record behind
Supplementary Table S2 of the paper. `analysis/scripts/design_numbers.py` reads this file and checks the
times against the run start times in the `provenance.txt` files under `results/`.

The decision rules are written out in the docstrings of the analysis scripts in `pipeline/`
(`analyze_phase2.py` for H1 and H2, `analyze_phase3.py` for H3, `analyze_phase4.py` for H4,
`analyze_phase4b.py` for H4b). `RESEARCH_PLAN.md` is the study plan of the development repository and is
not distributed. Paths are given as they were in the development repository; `analyze_phase*.py` are now
in `pipeline/` and `scripts/*.py` in `pipeline/scripts/`.

## Commits that fixed a decision rule

| Committer time (CST) | Commit | Protocol files changed | What was fixed |
|---|---|---|---|
| 2026-10-05 22:20:31 +0800 | `108db2c` | `RESEARCH_PLAN.md` | H1 and H2 written into the study plan, with the rule that a paired interval including zero is a tie |
| 2026-10-06 02:50:07 +0800 | `3ee3b4b` | `RESEARCH_PLAN.md`, `analyze_phase2.py` | Operational form of H1 (recall and false-positive rate at a validation threshold) and H2 (joint condition on two intervals) in the analysis script; family contrast with n = 5 |
| 2026-10-07 19:39:23 +0800 | `c58f02e` | `RESEARCH_PLAN.md`, `analyze_phase3.py`, `make_project_split.py`, `run_phase3.sh` | H3 (commit-disjoint split) and its analysis script |
| 2026-10-07 20:14:01 +0800 | `ec6523a` | `RESEARCH_PLAN.md`, `analyze_phase3.py` | H3 amendment before any commit-disjoint run: both tails of the contrast named |
| 2026-10-21 04:40:54 +0800 | `90de531` | `RESEARCH_PLAN.md`, `analyze_phase4.py`, `scripts/make_commit_family_split.py`, `scripts/precheck_h4.py`, `run_h4_precheck.sh`, `run_phase4.sh` | H4 with its feasibility gate, analysis script and pre-check |
| 2026-10-21 05:17:42 +0800 | `586236e` | `RESEARCH_PLAN.md`, `analyze_phase4b.py`, `scripts/direct_clone_edges.py`, `scripts/make_decon_splits.py`, `scripts/precheck_h4b.py`, `scripts/select_h4_design.py`, `run_h4_precheck.sh`, `run_phase4b.sh` | H4b (training-side decontamination against a matched stratified control) with amendment 1, its gate and analysis script |
| 2026-10-21 05:28:54 +0800 | `0936287` | `RESEARCH_PLAN.md`, `scripts/make_decon_splits.py`, `scripts/precheck_h4b.py` | H4b amendment 2 before any pre-check or run: control drawn from non-clone training functions; borrowing limit in the gate |

Runs and gates decided by these rules started at: Phase 2 (H1, H2) 2026-10-06 02:33; Phase 3 (H3)
2026-10-08 01:59; H4/H4b gate 2026-10-22 06:09; Phase 4b (H4b) 2026-10-22 06:46 (CST; from the
`provenance.txt` files and `results/h4_precheck_1022/h4_precheck.log`).

## Complete history of each analysis script

No analysis script changed after the runs it decides. The development repository's full `git log` of
each script:

| Script | Commits (all) |
|---|---|
| `analyze_phase2.py` | `3ee3b4b` |
| `analyze_phase3.py` | `ec6523a`, `c58f02e` |
| `analyze_phase4.py` | `90de531` |
| `analyze_phase4b.py` | `586236e` |
