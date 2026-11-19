#!/usr/bin/env python3
"""rq2_tables_figure.py -- RQ2 Tables 1-2 and Figure 1, generated from data only.

Reads analysis/scripts/out/rq2_numbers.json and h4b_numbers.json (rq2_numbers.py, h4b_numbers.py; rq2 recomputes
everything from the per-seed result JSONs and checks it against the project log). No value
is typed in by hand.

Writes
  paper/tables/rq2_arms.tex      Table 1  mean test PR-AUC (sd) per arm
  paper/tables/rq2_deltas.tex    Table 2  paired deltas with 95% t-CI
  paper/figures/rq2_deltas.pdf   Figure 1 dot-and-whisker of paired delta PR-AUC

Figure spec (IST Guide for Authors pp. 11-13: vector PDF, fonts embedded, caption
in the manuscript, readable for colour-vision deficiencies): pdf.fonttype 42
(TrueType, embedded), Arial 8 pt, 3.5 x 1.9 in (single column, ~89 mm), black only,
distinct marker shapes, so it reads in grayscale.

Usage (repository root):  python analysis/scripts/rq2_numbers.py && python analysis/scripts/rq2_tables_figure.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
NUM = json.load(open(os.path.join(HERE, "out", "rq2_numbers.json"), encoding="utf-8"))
A, D = NUM["per_arm"], NUM["deltas"]
# H4b row of Table 2 (pre-registered, RESEARCH_PLAN §13.8), from h4b_numbers.py
H4B = json.load(open(os.path.join(HERE, "out", "h4b_numbers.json"), encoding="utf-8"))["deltas"]
D = {**D, "sizectrl_to_decon": {k: {"mean": v["mean"], "lo": v["ci"][0], "hi": v["ci"][1], "n": v["n"]}
                                for k, v in H4B.items()}}


def f4(x):
    """4-decimal number with a typographic minus sign for LaTeX."""
    s = f"{x:.4f}"
    return s.replace("-", "$-$") if x < 0 else s


def sgn(x):
    s = f"{x:+.4f}"
    return s.replace("-", "$-$").replace("+", "$+$")


def ci(d):
    return f"[{sgn(d['lo'])}, {sgn(d['hi'])}]"


# ------------------------------------------------------------------ Table 1
ARMS = [("published", "Published"),
        ("random_control", "Random control"),
        ("family_disjoint", "Family-disjoint"),
        ("project_disjoint", "Project-disjoint"),
        ("commit_disjoint", "Commit-disjoint")]

# Family leak % of test (Eq. leak) per arm, read from the audit outputs:
#  results/audit_7oct/family_label/diversevul_theta0.7.md, table "leaked any %"
#  results/audit_7oct/diversevul_audit_splitfile_commitdisjoint_seed42_report.json
ROOT = os.path.dirname(PAPER)
LEAK = {}
with open(os.path.join(ROOT, "results", "audit_7oct", "family_label", "diversevul_theta0.7.md"),
          encoding="utf-8") as fh:
    for line in fh:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] in ("published", "random_control", "family_disjoint",
                                              "project_disjoint") and cells[0] not in LEAK:
            LEAK[cells[0]] = float(cells[2])
with open(os.path.join(ROOT, "results", "audit_7oct",
                       "diversevul_audit_splitfile_commitdisjoint_seed42_report.json"), encoding="utf-8") as fh:
    LEAK["commit_disjoint"] = json.load(fh)["family_leak_under_this_split"]["leaked_test_pct"]
assert set(LEAK) == {k for k, _ in ARMS}, LEAK

rows = []
for key, label in ARMS:
    a = A[key]
    rows.append(
        f"{label} & {a['pr_auc']['n']} & {a['n_test'][0]:,} & {a['pos_rate']['mean']:.4f} & "
        f"{LEAK[key]:.2f} & {f4(a['floor']['mean'])} & "
        f"{f4(a['pr_auc']['mean'])} ({a['pr_auc']['sd']:.4f}) \\\\")
# H4b training variants of the commit-disjoint arm (same test set), from h4b_numbers.py per-run values
H4B_RUNS = json.load(open(os.path.join(HERE, "out", "h4b_numbers.json"), encoding="utf-8"))["per_run"]
for key, label in (("commit_decon", r"\quad decontaminated (H4b)"), ("commit_sizectrl", r"\quad matched stratified control (H4b)")):
    runs = list(H4B_RUNS[key].values())
    vals = [r["pr_auc"] for r in runs]
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    assert len({r["n_test"] for r in runs}) == 1 and len({r["floor"] for r in runs}) == 1
    rows.append(f"{label} & {len(runs)} & {runs[0]['n_test']:,} & {runs[0]['pos_rate']:.4f} & n.c. & "
                f"{f4(runs[0]['floor'])} & {f4(m)} ({sd:.4f}) \\\\")
H4B_NTRAIN = {r["n_train"] for k in H4B_RUNS for r in H4B_RUNS[k].values()}
assert len(H4B_NTRAIN) == 1
# 1-NN clone-retrieval baseline on DiverseVul's published split (same source as rq3_numbers.py)
_rep = json.load(open(os.path.join(ROOT, "results", "from_backup", "diversevul_audit_report.json"), encoding="utf-8"))
KNN_DV = [x for x in _rep["thresholds"] if abs(x["theta"] - 0.7) < 1e-9][0]["knn_retrieval_baseline"]["pr_auc"]
rc13 = A["random_control_s1_3"]["pr_auc"]
fd13 = A["family_disjoint_s1_3"]["pr_auc"]
t1 = r"""\begin{table}[!tbp]
\centering
\caption{Test PR-AUC of CodeBERT on DiverseVul per split: mean (sample sd) over the training
seeds. Published: the split released with the corpus (called the original split
in~\cite{ullah2026vulnscout}), here that of the unofficial mirror~\cite{bstee615diversevul}.
Leak: test functions with a training function in the same clone family at $\theta=0.7$
(Eq.~\eqref{eq:leak}). The two H4b rows (pre-specified hypothesis H4b, Table~\ref{tab:hypotheses}) are training variants of the commit-disjoint split
(""" + f"{next(iter(H4B_NTRAIN)):,}" + r""" training functions each, same test set); n.c.\ = not computed: the family audit was not re-run on the H4b training sets. Reference
value: 1-NN clone retrieval on the published split reaches PR-AUC """ + f"{KNN_DV:.4f}" + r""" (Section~\ref{sec:rq3-floors}).}
\label{tab:rq2-arms}
\footnotesize
\setlength{\tabcolsep}{2.8pt}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{2.75cm}rrrrrl@{}}
\toprule
Split & Seeds & \shortstack[r]{Test\\functions} & \shortstack[r]{Positive\\rate} & \shortstack[r]{Leak\\(\%)} & \shortstack[r]{Length-only\\baseline} & PR-AUC (sd) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\smallskip
\parbox{\linewidth}{\footnotesize Seeds 1--3 only: random control """ + \
    f"{f4(rc13['mean'])} ({rc13['sd']:.4f}), family-disjoint {f4(fd13['mean'])} ({fd13['sd']:.4f})." + \
    r""" Published PR-AUC is on a different test set and is not paired with the others.}
\end{table}
"""

# ------------------------------------------------------------------ Table 2
CONTRASTS = [("random_to_family_n5", "Random $\\to$ family", "Family contrast"),
             ("random_to_project", "Random $\\to$ project", "H2"),
             ("random_to_commit", "Random $\\to$ commit", "H3"),
             ("commit_to_project", "Commit $\\to$ project", "H3, reported"),
             ("sizectrl_to_decon", "Matched stratified control $\\to$ decontaminated", "H4b")]
METRICS = [("pr_auc", "$\\Delta$PR-AUC"), ("headroom", "$\\Delta$Headroom"), ("mcc", "$\\Delta$MCC"),
           ("roc_auc", "$\\Delta$ROC-AUC")]
lines = []
for key, label, hyp in CONTRASTS:
    d = D[key]
    n = d["pr_auc"]["n"]
    lines.append(f"{label} & " + " & ".join(sgn(d[m]['mean']) for m, _ in METRICS) + r" \\")
    lines.append(f"\\quad {hyp}, $n={n}$ & " + " & ".join(ci(d[m]) for m, _ in METRICS) + r" \\[2pt]")
t2 = r"""\begin{table}[!tbp]
\centering
\caption{Paired $\Delta$ (Eq.~\eqref{eq:paired}) with 95\% $t$-intervals below each mean.
Second line: the analysis the contrast belongs to, the family contrast or a pre-specified hypothesis H
(Table~\ref{tab:hypotheses}; ``reported'': fixed under H3 to be reported whatever its sign), and the number of seeds $n$. In H4b both arms share the test set and length-only baseline,
so the headroom $\Delta$ equals the PR-AUC $\Delta$.}
\label{tab:rq2-deltas}
\scriptsize
\setlength{\tabcolsep}{1.5pt}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{2.75cm}cccc@{}}
\toprule
Contrast & """ + " & ".join(m for _, m in METRICS) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
os.makedirs(os.path.join(PAPER, "tables"), exist_ok=True)
open(os.path.join(PAPER, "tables", "rq2_arms.tex"), "w", encoding="utf-8", newline="\n").write(t1)
open(os.path.join(PAPER, "tables", "rq2_deltas.tex"), "w", encoding="utf-8", newline="\n").write(t2)

# ------------------------------------------------------------------ Figure 1
import figstyle as fs  # noqa: E402  shared style (13 Nov 2026): print width, fonts >= 7 pt, split colours
fs.apply()
fig, ax = plt.subplots(figsize=(fs.MAIN, 2.75))
# two-level row labels (19 Nov 2026 design round): the contrast, then the analysis and n in grey
labels = {"random_to_family_n5": ("random → family", "family contrast, n = 5"),
          "random_to_project": ("random → project", "H2, n = 3"),
          "random_to_commit": ("random → commit", "H3, n = 3"),
          "commit_to_project": ("commit → project", "H3, reported, n = 3"),
          "sizectrl_to_decon": ("matched stratified control →\ndecontaminated", "H4b, n = 3")}
# each contrast takes the colour and marker of its later split (figstyle.SPLIT)
later = {"random_to_family_n5": "family_disjoint", "random_to_project": "project_disjoint",
         "random_to_commit": "commit_disjoint", "commit_to_project": "project_disjoint",
         "sizectrl_to_decon": "decontaminated"}
markers = {k: fs.SPLIT[v]["marker"] for k, v in later.items()}
keys = [k for k, _, _ in CONTRASTS if k in markers]          # Figure: all five contrasts of Table rq2-deltas
for i, k in enumerate(keys):
    y = len(keys) - 1 - i
    d = D[k]["pr_auc"]
    st = fs.SPLIT[later[k]]
    ax.errorbar(d["mean"], y, xerr=[[d["mean"] - d["lo"]], [d["hi"] - d["mean"]]],
                fmt=st["marker"], color=st["color"], mec=fs.INK, mew=0.4, capsize=2.5, elinewidth=fs.ERR_LW,
                ms=fs.MARKER + 0.6, zorder=3)
ax.axvline(0, **fs.REF)
fs.bands(ax, len(keys))
fs.row_labels(ax, [labels[k] for k in keys])
ax.tick_params(axis="y", length=0)
ax.set_xlabel("Paired Δ test PR-AUC (95% t-CI)")
ax.set_ylim(-0.5, len(keys) - 0.5)
fs.grid(ax, "x")
fig.tight_layout(pad=0.3)
fig.subplots_adjust(left=0.30)
os.makedirs(os.path.join(PAPER, "figures"), exist_ok=True)
out = os.path.join(PAPER, "figures", "rq2_deltas.pdf")
fs.save(fig, out)
print("wrote tables/rq2_arms.tex, tables/rq2_deltas.tex, figures/rq2_deltas.pdf")
