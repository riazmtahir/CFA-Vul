#!/usr/bin/env python3
"""rq3_tables_figure.py -- RQ3 Tables 3-4 and Figure 2, generated from data only.

Reads analysis/scripts/out/rq3_numbers.json (written by rq3_numbers.py, which reads the
result files and cross-checks them). No value is typed in by hand.

Writes
  paper/tables/rq3_exposure.tex   Table 3  commit-level exposure per arm (descriptive)
  paper/tables/rq3_twin.tex       Table 4  twin effect by twin location (exploratory)
  paper/figures/rq3_rank.pdf      Figure 2 rank accuracy per arm (descriptive)

Figure spec: same rules as Figure 1 (IST Guide for Authors pp. 11-13): vector PDF,
pdf.fonttype 42 (TrueType, embedded), Arial 8 pt, 3.5 x 2.2 in (single column), black and
one grey, distinct marker shapes and fills, so it reads in grayscale.

Usage (repository root):  python analysis/scripts/rq3_numbers.py && python analysis/scripts/rq3_tables_figure.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
NUM = json.load(open(os.path.join(HERE, "out", "rq3_numbers.json"), encoding="utf-8"))
E, TW, PE = NUM["exposure"], NUM["twin"], NUM["paired"]


def sgn(x):
    s = f"{x:+.4f}"
    return s.replace("-", "$-$").replace("+", "$+$")


def f4(x):
    return f"{x:.4f}"


def pct(x):
    return f"{x:.2f}"


# ------------------------------------------------------------------ Table 3
ARMS = {"diversevul": [("published", "Published"), ("random_control", "Random control"),
                       ("family_disjoint", "Family-disjoint"), ("project_disjoint", "Project-disjoint"),
                       ("commit_disjoint", "Commit-disjoint")],
        "primevul": [("published", "Published"), ("random_control", "Random control"),
                     ("family_disjoint", "Family-disjoint")],
        "devign": [("published", "Published"), ("random_control", "Random control"),
                   ("family_disjoint", "Family-disjoint")]}
NAMES = {"diversevul": "DiverseVul", "primevul": "PrimeVul", "devign": "Devign"}
rows = []
for corpus, arms in ARMS.items():
    rows.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{NAMES[corpus]}}}}} \\\\")
    for key, label in arms:
        p, a = E[corpus][key]["test_positives"], E[corpus][key]["all_test"]
        rows.append(f"\\quad {label} & {int(p['n']):,} & {pct(p['same_commit'])} & {pct(p['opp_label'])} & "
                    f"{pct(p['own_twin'])} & {pct(p['own_twin_outside_family'])} & "
                    f"{pct(a['same_commit'])} & {pct(a['own_twin'])} \\\\")
t3 = r"""\begin{table}[tbp]
\centering
\caption{Commit-level exposure of test functions to training (descriptive audit, \%).
$n$: test positives. Commit: a training function from the same commit. Opposite label: one with the opposite
label. Counterpart: the function's patched or vulnerable counterpart
(Eq.~\eqref{eq:twin}). Outside family: counterpart not in the function's clone family at
$\theta=0.7$. Published: the split released with the corpus, for DiverseVul the mirror's split~\cite{bstee615diversevul}, the
released splits for PrimeVul and Devign.}
\label{tab:rq3-exposure}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrrr}
\toprule
 & \multicolumn{5}{c}{Test positives} & \multicolumn{2}{c}{All test} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-8}
Split & $n$ & Commit & \shortstack{Opposite\\label} & \shortstack{Counter-\\part} & \shortstack{Outside\\family} & Commit & \shortstack{Counter-\\part} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

# ------------------------------------------------------------------ Table 4
LOCS = ("twin_in_train", "twin_elsewhere", "no_twin")
lines = []
for arm, label in (("random_control", "Random control"), ("family_disjoint", "Family-disjoint")):
    lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{label}}}, seeds "
                 f"{'--'.join(map(str, (TW[arm]['seeds'][0], TW[arm]['seeds'][-1])))}}} \\\\")
    for cls, metric in (("negatives", "FPR"), ("positives", "Recall")):
        g, c = TW[arm][cls]["groups"], TW[arm][cls]["contrasts"]["train_vs_elsewhere"]
        lines.append(f"\\quad {metric} & " + " & ".join(f"{f4(g[k]['rate'])}" for k in LOCS) +
                     f" & {sgn(c['raw'])} & {sgn(c['std'])} \\\\")
        lines.append("\\quad\\quad $n$ / 95\\% CI & " + " & ".join(f"{g[k]['n']:,}" for k in LOCS) +
                     f" & [{sgn(c['raw_ci95'][0])}, {sgn(c['raw_ci95'][1])}]"
                     f" & [{sgn(c['std_ci95'][0])}, {sgn(c['std_ci95'][1])}] \\\\[2pt]")
t4 = r"""\begin{table}[tbp]
\centering
\caption{Exploratory counterpart effect on DiverseVul (not pre-specified). Test functions grouped
by where their counterpart (Eq.~\eqref{eq:twin}) is: in training, only in validation or
test (elsewhere), or nowhere. Rates at $\tau^*(\mathrm{Va})$, mean over two seeds.
Train $-$ elsewhere is the exposure contrast; length-standardized: within length quintiles,
weighted by the in-training group. 95\% percentile bootstrap CIs over test functions
(1,000 resamples).}
\label{tab:rq3-twin}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrcc}
\toprule
 & \multicolumn{3}{c}{Counterpart} & \multicolumn{2}{c}{Train $-$ elsewhere} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
 & In train & Elsewhere & None & Raw & Length-standardized \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
os.makedirs(os.path.join(PAPER, "tables"), exist_ok=True)
open(os.path.join(PAPER, "tables", "rq3_exposure.tex"), "w", encoding="utf-8", newline="\n").write(t3)
open(os.path.join(PAPER, "tables", "rq3_twin.tex"), "w", encoding="utf-8", newline="\n").write(t4)

# ------------------------------------------------------------------ Figure 2
import figstyle as fs  # noqa: E402  shared style (13 Nov 2026): print width, fonts >= 7 pt, split colours
from matplotlib.lines import Line2D  # noqa: E402
fs.apply()
fig, ax = plt.subplots(figsize=(fs.MAIN, 2.75))
import matplotlib.transforms as mtrans  # noqa: E402
RIGHT = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
ORDER = [("random_control", "random control"), ("family_disjoint", "family-disjoint"),
         ("commit_disjoint", "commit-disjoint"), ("project_disjoint", "project-disjoint")]
for i, (k, name) in enumerate(ORDER):
    y = len(ORDER) - 1 - i
    p = PE[k]
    raw, rlo, rhi = p["mean"]["rank_acc"], *p["ci95"]["rank_acc"]
    bal, blo, bhi = p["balanced_rank_acc"], *p["balanced_rank_acc_ci95"]
    st = fs.SPLIT[k]
    # raw rank accuracy: open marker; length-balanced: filled marker; both in the split's colour and shape
    ax.errorbar(raw, y + 0.17, xerr=[[raw - rlo], [rhi - raw]], fmt=st["marker"], mfc="white", mec=st["color"],
                mew=1.0, color=st["color"], capsize=2.5, elinewidth=fs.ERR_LW, zorder=3)
    ax.errorbar(bal, y - 0.17, xerr=[[bal - blo], [bhi - bal]], fmt=st["marker"], mfc=st["color"], mec=fs.INK,
                mew=0.4, color=st["color"], capsize=2.5, elinewidth=fs.ERR_LW, zorder=3)
    # review suggestion 54: the length-balanced value and its interval printed at the right margin
    ax.text(1.03, y - 0.17, f"{bal:.4f} [{blo:.4f}, {bhi:.4f}]", transform=RIGHT, ha="left", va="center",
            fontsize=fs.SMALL, clip_on=False)
    ax.plot(p["length_only_rank_acc"], y, fs.LENGTH_ONLY["marker"], color=fs.LENGTH_ONLY["color"],
            markersize=5, mew=1.0, zorder=3)
# column header for the printed values (19 Nov 2026 design round)
ax.text(1.03, len(ORDER) - 0.5, "Length-balanced\n[95% CI]", transform=RIGHT, ha="left", va="bottom",
        fontsize=fs.SMALL, color=fs.MUTED, clip_on=False, linespacing=1.05)
ax.axvline(0.5, **fs.REF)
fs.grid(ax, "x")
fs.bands(ax, len(ORDER))
seeds = lambda k: "s" + "–".join(map(str, (PE[k]["seeds"][0], PE[k]["seeds"][-1])))
fs.row_labels(ax, [(name, f"{PE[k]['pairs']:,} pairs, {seeds(k)}") for k, name in ORDER])
ax.tick_params(axis="y", length=0)
ax.set_xlabel("P(score vulnerable > score patched), 95% CI")
ax.set_xlim(0.15, 0.72)
ax.set_ylim(-0.5, len(ORDER) - 0.5)
proxies = [Line2D([], [], ls="", marker="o", mfc="white", mec=fs.INK, mew=1.0, ms=fs.MARKER),
           Line2D([], [], ls="", marker="o", mfc=fs.INK, mec=fs.INK, ms=fs.MARKER),
           Line2D([], [], ls="", marker=fs.LENGTH_ONLY["marker"], color=fs.LENGTH_ONLY["color"], mew=1.0, ms=5)]
fig.legend(proxies, ["rank accuracy (open)", "length-balanced (filled)", "length only"], loc="upper center",
           ncol=3, frameon=False, fontsize=fs.SMALL, handletextpad=0.3, columnspacing=0.9,
           bbox_to_anchor=(0.48, 1.0))
fig.tight_layout(pad=0.3, rect=(0, 0, 0.815, 0.88))
fig.subplots_adjust(left=0.175)
os.makedirs(os.path.join(PAPER, "figures"), exist_ok=True)
fs.save(fig, os.path.join(PAPER, "figures", "rq3_rank.pdf"))
print("wrote tables/rq3_exposure.tex, tables/rq3_twin.tex, figures/rq3_rank.pdf")
