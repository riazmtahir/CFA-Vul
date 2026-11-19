#!/usr/bin/env python3
"""run_all.py -- quick check of the replication package: recompute every reported number from the shipped
result files, then regenerate the tables and figures.

Usage (from the repository root):  python analysis/run_all.py
Each checking script exits non-zero on any mismatch with the values reported in the paper; this script
stops at the first failure. Outputs: analysis/scripts/out/*.json (recomputed numbers), analysis/tables/,
analysis/figures/ (PDF) and analysis/figures/export/ (SVG, PNG 300 and 600 dpi), analysis/supplementary/.
CPU only; about one to two minutes.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STEPS = [
    # 1. numbers: recompute from results/ and compare with the reported values
    "rq1_numbers.py", "rq2_numbers.py", "rq3_numbers.py", "h4b_numbers.py", "backup_numbers.py",
    "cpu_checks_numbers.py", "m1_numbers.py", "effect_sizes.py", "design_numbers.py", "revision_numbers.py",
    # 2. tables and figures (read the JSONs written in step 1)
    "corpora_table.py", "hypotheses_table.py", "rq1_table_figure.py", "rq2_tables_figure.py",
    "rq3_tables_figure.py", "fig_leakage_profile.py", "fig_overview.py",
    "per_seed_table.py", "supp_validation_curves.py", "supp_figures.py", "supp_m1_item57.py",
]
ok = True
for s in STEPS:
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, "scripts", s)], cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    last = [l for l in (r.stdout + r.stderr).strip().splitlines() if l.strip()][-1:] or [""]
    print(f"{'OK ' if r.returncode == 0 else 'FAIL'} {s:28s} {time.time() - t0:5.1f} s  {last[0][:90]}")
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        ok = False
        break
print("ALL CHECKS PASSED" if ok else "STOPPED AT A FAILURE")
sys.exit(0 if ok else 1)
