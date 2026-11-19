#!/usr/bin/env python3
"""hypotheses_table.py -- Table "Hypotheses and outcomes at a glance", generated from the analysis JSONs.

Reads
  results/rq2_diversevul_phase2/phase2_analysis.json   H1 (C_H1), H2 (B_H2), written by analyze_phase2.py
  results/rq2_diversevul_phase3/phase3_analysis.json   H3, written by analyze_phase3.py
  results/h4_precheck_1022/h4_precheck.json            H4 feasibility gate
  results/rq2_diversevul_phase4b/phase4b_analysis.json H4b, written by analyze_phase4b.py
  analysis/scripts/out/effect_sizes.json, h4b_numbers.json  relative changes (secondary)
Writes paper/tables/hypotheses.tex. Verdicts are copied verbatim from the JSONs; every number is
re-checked against the values printed elsewhere in the paper (exit 1 on mismatch).
Usage (repository root): python analysis/scripts/hypotheses_table.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
R = os.path.join(os.path.dirname(PAPER), "results")


def load(*p):
    return json.load(open(os.path.join(*p), encoding="utf-8"))


p2 = load(R, "rq2_diversevul_phase2", "phase2_analysis.json")
p3 = load(R, "rq2_diversevul_phase3", "phase3_analysis.json")
p4b = load(R, "rq2_diversevul_phase4b", "phase4b_analysis.json")
h4 = load(R, "h4_precheck_1022", "h4_precheck.json")
eff = load(HERE, "out", "effect_sizes.json")["contrasts"]
h4bn = load(HERE, "out", "h4b_numbers.json")
rq2 = load(HERE, "out", "rq2_numbers.json")["deltas"]


def s4(x):
    s = f"{x:+.4f}"
    return s.replace("-", "$-$").replace("+", "$+$")


def pct(x):
    return f"{100 * x:+.1f}".replace("-", "$-$").replace("+", "$+$") + "\%"


def ci(m, lo, hi):
    return f"{s4(m)} [{s4(lo)}, {s4(hi)}]"


fails = []


def expect(name, got, want, nd=4):
    if abs(got - want) > 0.5 * 10 ** (-nd) + 1e-9:
        fails.append((name, got, want))


h1a = p2["C_H1"]["random_control"]["contrasts"]["H1a recall leaked_opposite pos - clean pos"]
expect("H1a diff", h1a["difference"], 0.0276)
proj = p2["B_H2"]["random_to_project"]["pr_auc"]
fam3 = p2["B_H2"]["random_to_family_same_seeds"]["pr_auc"]
expect("H2 project", proj["mean"], -0.1067)
expect("H2 family s1-3", fam3["mean"], -0.0005)
com = rq2["random_to_commit"]["pr_auc"]
expect("H3", com["mean"], -0.0815)
hb = h4bn["deltas"]["pr_auc"]
expect("H4b", hb["mean"], -0.0265)

# Outcome in full words, derived from (and asserted against) the verdict strings of the analysis JSONs.
OUTCOME = {
    "NOT SUPPORTED (tie)": "Not supported (H1a is a tie)",
    "SUPPORTED": "Supported",
    "H3 SUPPORTED: same-commit sharing inflates random-split PR-AUC":
        "Supported: same-commit sharing inflates random-split PR-AUC",
    "H4b SUPPORTED: direct high-recall Type-3 clones in training inflate PR-AUC beyond commit sharing":
        "Supported: direct high-recall Type-3 clones in training inflate PR-AUC beyond commit sharing",
}
assert h4["verdict"] == "FAIL"
rows = [
    ("H1 (RQ3)", "leak category minus clean, random control: recall and false-positive rate",
     f"not used (H1a: $\\Delta$recall {s4(h1a['difference'])} [{s4(h1a['ci95'][0])}, {s4(h1a['ci95'][1])}])",
     "not used", "4--5", OUTCOME[p2["C_H1"]["random_control"]["verdict_H1"]]),
    ("H2 (RQ2)", "random control $\\to$ project-disjoint; family condition on seeds 1--3",
     f"{ci(proj['mean'], *proj['ci'])}; family {ci(fam3['mean'], *fam3['ci'])}",
     pct(eff["random_to_project"]["relative_change_pr_auc"]["relative"]), "1--3",
     OUTCOME[p2["B_H2"]["verdict_H2"]]),
    ("H3 (RQ2)", "random control $\\to$ commit-disjoint",
     ci(com["mean"], com["lo"], com["hi"]),
     pct(eff["random_to_commit"]["relative_change_pr_auc"]["relative"]), "1--3", OUTCOME[p3["verdict"]]),
    ("H4 (RQ2)", "commits and high-recall families kept together", "not run", "not run", "none",
     "Feasibility gate failed; not run"),
    ("H4b (RQ2)", "matched stratified control $\\to$ decontaminated (commit-disjoint test set)",
     ci(hb["mean"], *hb["ci"]), pct(h4bn["relative_change_pr_auc"]), "1--3", OUTCOME[p4b["verdict"]]),
]
expect("H2 rel", 100 * eff["random_to_project"]["relative_change_pr_auc"]["relative"], -42.0, 1)
expect("H3 rel", 100 * eff["random_to_commit"]["relative_change_pr_auc"]["relative"], -32.1, 1)
expect("H4b rel", 100 * h4bn["relative_change_pr_auc"], -14.5, 1)

body = "\n".join(" & ".join(r[:5]) + " & " + r[5] + r" \\" for r in rows)
tex = r"""\begin{table}[!tbp]
\centering
\caption{Hypotheses and outcomes at a glance. H1--H4b: pre-specified hypotheses, whose decision rules
were committed before the results they decide (Section~\ref{sec:design-prereg}). $\Delta$PR-AUC: paired change in test PR-AUC
(later minus earlier split) with its 95\% $t$-interval (Eq.~\eqref{eq:paired}). Relative change:
$\Delta$PR-AUC divided by the earlier split's mean PR-AUC on the same seeds. H1 is decided on error
rates, not on PR-AUC (Section~\ref{sec:rq3-h1}). Outcomes restate the analysis scripts' verdicts.}
\label{tab:hypotheses}
\scriptsize
\setlength{\tabcolsep}{2pt}
\begin{tabularx}{\linewidth}{@{}l>{\raggedright\arraybackslash}p{2.7cm}>{\raggedright\arraybackslash}p{3.0cm}>{\raggedright\arraybackslash}p{1.1cm}lL@{}}
\toprule
Hypothesis & Contrast & $\Delta$PR-AUC [95\% CI] & Relative change & Seeds & Outcome \\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
open(os.path.join(PAPER, "tables", "hypotheses.tex"), "w", encoding="utf-8", newline="\n").write(tex)
print("wrote tables/hypotheses.tex;", "checks passed" if not fails else f"FAILED {fails}")
sys.exit(1 if fails else 0)
