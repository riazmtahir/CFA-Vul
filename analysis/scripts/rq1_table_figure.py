#!/usr/bin/env python3
"""rq1_table_figure.py -- RQ1 table and figure, generated from data only.

Reads analysis/scripts/out/rq1_numbers.json (written by rq1_numbers.py) and, for the EXPLORATORY
high-recall rows, analysis/scripts/out/cpu_checks_numbers.json (cpu_checks_numbers.py). No value
typed in.

Writes
  paper/tables/rq1_audit.tex     per-corpus audit at theta = 0.7
  paper/figures/rq1_theta.pdf    leaked test % vs theta, published (solid) and random control (dashed)

Figure spec: same rules as Figures 1-2 (IST Guide for Authors pp. 11-13): vector PDF,
pdf.fonttype 42, Arial 8 pt, 3.5 x 2.3 in (single column), black only, one marker shape per
corpus, solid vs dashed per split, so it reads in grayscale.

Usage (repository root):  python analysis/scripts/rq1_numbers.py && python analysis/scripts/rq1_table_figure.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
NUM = json.load(open(os.path.join(HERE, "out", "rq1_numbers.json"), encoding="utf-8"))
AU, FL = NUM["audit"], NUM["family_label"]
HRN = json.load(open(os.path.join(HERE, "out", "cpu_checks_numbers.json"), encoding="utf-8"))["values"]["highrecall"]
CORPORA = [("devign", "Devign"), ("diversevul", "DiverseVul"), ("primevul", "PrimeVul")]

# ------------------------------------------------------------------ Table
ROWS = [
    ("Functions", lambda c: f"{AU[c]['0.7']['n_functions']:,}"),
    ("Test functions (published split)", lambda c: f"{AU[c]['0.7']['n_test']:,}"),
    ("Type-1/2 redundant functions (\\%)", lambda c: f"{AU[c]['0.7']['type12_pct']:.2f}"),
    ("Families of size $>1$", lambda c: f"{AU[c]['0.7']['families_gt1']:,}"),
    ("Largest family (functions)", lambda c: f"{AU[c]['0.7']['largest_family']:,}"),
    ("Leaked test functions, published (\\%)", lambda c: f"{AU[c]['0.7']['leaked_pct']:.2f}"),
    ("Leaked test functions, random control (\\%)", lambda c: f"{AU[c]['0.7']['rc_leaked_pct']:.2f}"),
    ("Positives in mixed-label families (\\%)", lambda c: f"{FL[c]['pos_mixed_pct']:.2f}"),
    ("Cross-project families (\\% of families $>1$)", lambda c: f"{FL[c]['xp_pct']:.2f}"),
]
# EXPLORATORY high-recall criterion (blind identifiers, k = 3, theta = 0.7): any / same-label (Croft-def)
HR_ROWS = [
    ("\\quad published", lambda c: f"{HRN[c]['published']['any_pct']:.2f} / {HRN[c]['published']['croft_pct']:.2f}"),
    ("\\quad random control", lambda c: f"{HRN[c]['random_control']['any_pct']:.2f} / "
                                        f"{HRN[c]['random_control']['croft_pct']:.2f}"),
    ("\\quad family-disjoint (strict)", lambda c: f"{HRN[c]['family_disjoint']['any_pct']:.2f} / "
                                                  f"{HRN[c]['family_disjoint']['croft_pct']:.2f}"),
    ("\\quad largest family (\\% of corpus)", lambda c: f"{HRN[c]['largest_family_pct']:.2f}"),
]
body = "\n".join(f"{lab} & " + " & ".join(fn(c) for c, _ in CORPORA) + r" \\" for lab, fn in ROWS)
body += "\n\\midrule\n\\multicolumn{4}{l}{\\emph{High-recall criterion (exploratory): leaked any / same-label (\\%)}} \\\\\n"
body += "\n".join(f"{lab} & " + " & ".join(fn(c) for c, _ in CORPORA) + r" \\" for lab, fn in HR_ROWS)
tab = r"""\begin{table}[!tbp]
\centering
\caption{Clone-family audit at $\theta=0.7$ (Algorithm~\ref{alg:families}). Type-1/2 redundant:
functions collapsed onto another function with the same Type-2 normal form (group size
minus one, summed over groups). Leaked: test functions
with a training family-mate (Eq.~\eqref{eq:leak}). Random control: label-stratified random
split with the family-disjoint split's partition sizes. Mixed-label: families holding both
vulnerable and non-vulnerable functions. Lower block (exploratory, Section~\ref{sec:rq1-highrecall}):
blind identifiers, 3-token shingles, $\theta=0.7$, on the published split and on the random-control
and strict family-disjoint splits of RQ2; same-label: a training family-mate with the test
function's label (Croft et al.'s definition).}
\label{tab:rq1-audit}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lrrr}
\toprule
 & """ + " & ".join(n for _, n in CORPORA) + r""" \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""
os.makedirs(os.path.join(PAPER, "tables"), exist_ok=True)
open(os.path.join(PAPER, "tables", "rq1_audit.tex"), "w", encoding="utf-8", newline="\n").write(tab)

# ------------------------------------------------------------------ Figure
import figstyle as fs  # noqa: E402  shared style (13 Nov 2026): print width, fonts >= 7 pt
fs.apply()
# Two panels (published | random control), direct line labels; corpora are not splits, so black with one
# marker per corpus (review item 6.4, 24 Oct 2026). Drawn at print width (\linewidth).
fig, axes = plt.subplots(1, 2, figsize=(fs.MAIN, 2.55), sharey=True)
TH = ["0.7", "0.8", "0.9"]
for ax, field, title, ls in ((axes[0], "leaked_pct", "Published split", "-"),
                             (axes[1], "rc_leaked_pct", "Random control", "--")):
    x = [float(t) for t in TH]
    ys = {c: [AU[c][t][field] for t in TH] for c, _ in CORPORA}
    for c, name in CORPORA:
        ax.plot(x, ys[c], ls, marker=fs.CORPUS[c]["marker"], color=fs.CORPUS[c]["color"],
                mfc=fs.CORPUS[c]["color"], mec="white", mew=0.6,
                ms=fs.MARKER + 0.8, zorder=3)
    # direct labels at the right end; nudge apart labels closer than 2.5 points
    order = sorted(CORPORA, key=lambda cn: ys[cn[0]][-1])
    last = None
    for c, name in order:
        y = ys[c][-1]
        if last is not None and y - last < 2.5:
            y = last + 2.5
        ax.text(0.912, y, name, fontsize=fs.SMALL, va="center", ha="left", color=fs.CORPUS[c]["color"])
        last = y
    ax.set_title(title)
    ax.set_xticks([0.7, 0.8, 0.9])
    ax.set_xlim(0.67, 0.98)
    ax.set_ylim(0, 33)
    ax.set_xlabel("Clone threshold θ")
    fs.grid(ax)
axes[0].set_ylabel("Leaked test functions (%)")
fig.tight_layout(pad=0.3, w_pad=1.0)
os.makedirs(os.path.join(PAPER, "figures"), exist_ok=True)
fs.save(fig, os.path.join(PAPER, "figures", "rq1_theta.pdf"))
print("wrote tables/rq1_audit.tex, figures/rq1_theta.pdf")
