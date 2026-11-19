#!/usr/bin/env python3
"""corpora_table.py -- Table "Corpora", generated from the checked outputs and result files.

Reads
  analysis/scripts/out/rq1_numbers.json   functions, positives, split sizes, PrimeVul projects per partition
                                       (rq1_numbers.py recomputes them from the audit reports)
  results/from_backup_1012/diversevul_audit_splitfile_projectdisjoint_seed42_report.json   n_projects
Writes paper/tables/corpora.tex. Every value is asserted against paper/numbers.md (exit 1 on mismatch).
Projects are given only where a result file records them (Devign: not recorded).
Usage (repository root): python analysis/scripts/rq1_numbers.py && python analysis/scripts/corpora_table.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
R = os.path.join(os.path.dirname(PAPER), "results")
N = json.load(open(os.path.join(HERE, "out", "rq1_numbers.json"), encoding="utf-8"))
proj_dv = json.load(open(os.path.join(R, "from_backup_1012",
                                      "diversevul_audit_splitfile_projectdisjoint_seed42_report.json"),
                         encoding="utf-8"))["n_projects"]
pv = N["primevul_prep"]["projects"]

rows, fails = [], []
SPEC = {
    "devign": ("Devign", r"2~\cite{zhou2019devign}", r"CodeXGLUE release~\cite{codexglue_defect_hf}",
               "audit and training"),
    "diversevul": ("DiverseVul", f"{proj_dv:,}",
                   r"unofficial Hugging Face mirror, revision \texttt{3ed5dae}~\cite{bstee615diversevul}",
                   "audit and training"),
    "primevul": ("PrimeVul", f"{pv['train']:,} / {pv['valid']:,} / {pv['test']:,}",
                 r"released split~\cite{primevul_data}", "audit only"),
}
EXPECT = {"devign": (27318, 12460, (21854, 2732, 2732)), "diversevul": (330491, 18945, (264392, 33049, 33050)),
          "primevul": (235768, 6968, (184427, 25430, 25911))}
for c, (name, projects, source, role) in SPEC.items():
    n = N["family_label"][c]["n_functions"]
    pos = N["family_label"][c]["positives_total"]
    sp = N["inputs"][c]
    sizes = (sp["train"], sp["valid"], sp["test"])
    if (n, pos, sizes) != EXPECT[c]:
        fails.append((c, n, pos, sizes))
    rows.append(f"{name} & {n:,} & {100 * pos / n:.2f} & {projects} & "
                f"{sizes[0]:,} / {sizes[1]:,} / {sizes[2]:,} & {source} & {role} \\\\")
if proj_dv != 800 or (pv["train"], pv["valid"], pv["test"]) != (633, 218, 222):
    fails.append(("projects", proj_dv, pv))

tex = r"""\begin{table}[!tbp]
\centering
\caption{Corpora. Vulnerable: share of functions labeled vulnerable. Projects: distinct projects;
for PrimeVul per partition (train / validation / test); for Devign, its public part as reported
in the cited source.
Published split: functions in train / validation / test, as released (DiverseVul without one empty
training function). Training uses CodeBERT on all DiverseVul splits and CodeBERT and GraphCodeBERT
on three Devign splits (Section~\ref{sec:design-training}).}
\label{tab:corpora}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{tabularx}{\linewidth}{@{}lrr>{\raggedright\arraybackslash}p{1.6cm}>{\raggedright\arraybackslash}p{2.2cm}LL@{}}
\toprule
Corpus & Functions & Vulnerable (\%) & Projects & Published split (train / validation / test) & Split source & Use in this study \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
open(os.path.join(PAPER, "tables", "corpora.tex"), "w", encoding="utf-8", newline="\n").write(tex)
print("wrote tables/corpora.tex;", "checks passed" if not fails else f"FAILED {fails}")
sys.exit(1 if fails else 0)
