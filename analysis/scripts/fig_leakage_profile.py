#!/usr/bin/env python3
"""fig_leakage_profile.py -- Figure "Leakage profile of each split" (DiverseVul), from the result files.

Four measures per split (published, random control, family-, project-, commit-disjoint):
  near-duplicate family leakage, % of test functions (Eq. leak, strict criterion)
      results/audit_7oct/family_label/diversevul_theta0.7.md ("leaked any %"),
      results/audit_7oct/diversevul_audit_splitfile_commitdisjoint_seed42_report.json (commit-disjoint)
  direct high-recall clone in training, % of test functions (no chaining)
      results/h4_precheck_1022/direct_edges/diversevul_hr_direct_summary.json
  commit sharing, % of test functions (a training function from the same commit)
  patched counterpart in training, % of test POSITIVES (Eq. twin)
      both from analysis/scripts/out/rq3_numbers.json "exposure" (commit_leak_audit.py outputs, checked there)
Every bar is checked against the value in paper/numbers.md (exit 1 on mismatch).
Figure spec as Figures 1-3: vector PDF, fonts embedded (pdf.fonttype 42), Arial, no text below 7 pt at print size, black/grey with
hatching so it reads in grayscale; two panels, one per denominator; value labels on every bar.
Usage (repository root): python analysis/scripts/rq3_numbers.py && python analysis/scripts/fig_leakage_profile.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
R = os.path.join(os.path.dirname(PAPER), "results")
SPLITS = [("published", "Published"), ("random_control", "Random\ncontrol"), ("family_disjoint", "Family-\ndisjoint"),
          ("project_disjoint", "Project-\ndisjoint"), ("commit_disjoint", "Commit-\ndisjoint")]

leak = {}
with open(os.path.join(R, "audit_7oct", "family_label", "diversevul_theta0.7.md"), encoding="utf-8") as fh:
    for line in fh:
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) >= 3 and c[0] in ("published", "random_control", "family_disjoint", "project_disjoint") \
                and c[0] not in leak:
            leak[c[0]] = float(c[2])
leak["commit_disjoint"] = json.load(open(os.path.join(
    R, "audit_7oct", "diversevul_audit_splitfile_commitdisjoint_seed42_report.json"),
    encoding="utf-8"))["family_leak_under_this_split"]["leaked_test_pct"]
direct = {k: v["direct_leaked_test_pct"] for k, v in json.load(open(os.path.join(
    R, "h4_precheck_1022", "direct_edges", "diversevul_hr_direct_summary.json"), encoding="utf-8"))["splits"].items()}
ex = json.load(open(os.path.join(HERE, "out", "rq3_numbers.json"), encoding="utf-8"))["exposure"]["diversevul"]
commit = {k: ex[k]["all_test"]["same_commit"] for k, _ in SPLITS}
counterpart = {k: ex[k]["test_positives"]["own_twin"] for k, _ in SPLITS}

import figstyle as fs  # noqa: E402  shared style (13 Nov 2026): print width, fonts >= 7 pt, split colours
# colour round (20 Nov 2026): a slate ramp light -> dark for the three shares of all test functions and a hatched
# bar for the counterpart share; slate is outside the split palette, so a bar colour never means a split
MEASURES = [("Near-duplicate family leakage (% of test functions)", leak, fs.MEASURE_FILLS[0], ""),
            ("Direct high-recall clone in training (% of test functions)", direct, fs.MEASURE_FILLS[1], ""),
            ("Commit sharing (% of test functions)", commit, fs.MEASURE_FILLS[2], ""),
            ("Patched counterpart in training (% of test positives)", counterpart, fs.MEASURE_FILLS[3], "////")]
EXPECT = {"Near": (29.61, 29.39, 0.00, 5.83, 17.12), "Direct": (56.45, 56.23, 44.87, 16.53, 37.15),
          "Commit": (99.75, 99.72, 99.54, 0.00, 0.00), "Patched": (70.90, 69.02, 46.01, 0.00, 0.00)}
fails = []
for (name, vals, _, _), (key, want) in zip(MEASURES, EXPECT.items()):
    for (s, _), w in zip(SPLITS, want):
        if abs(vals[s] - w) > 0.005 + 1e-9:
            fails.append((key, s, vals[s], w))

fs.apply()
# Two panels so the two denominators never share an axis (review item 50); every bar carries its value
# (item 49). Drawn at print size (0.75 linewidth). Bars are measures, not splits: grey fills and a hatch;
# each split on the x axis carries its figstyle marker in its colour under the label.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fs.MAIN_075, 3.9), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1.6]})
def bars(ax, measures, w):
    for j, (name, vals, color, hatch) in enumerate(measures):
        off = (j - (len(measures) - 1) / 2) * w
        xs = [i + off for i in range(len(SPLITS))]
        ys = [vals[s] for s, _ in SPLITS]
        ax.bar(xs, ys, width=w, color=color, edgecolor=fs.INK, linewidth=fs.EDGE_LW, hatch=hatch, label=name, zorder=2)
        for x, y in zip(xs, ys):
            ax.text(x, y + 2, f"{y:.2f}", ha="center", va="bottom", fontsize=fs.SMALL, rotation=90)
    ax.set_ylim(0, 135)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("%")
    fs.grid(ax)
    ax.legend(loc="upper right", fontsize=fs.SMALL, handlelength=1.4, borderaxespad=0.1)
bars(ax1, [(n.replace(" (% of test functions)", ""), v, c, h) for n, v, c, h in MEASURES[:3]], 0.26)
ax1.set_title("(a) Share of all test functions", loc="left", pad=2)
ax1.set_ylim(0, 170)
bars(ax2, [(n.replace(" (% of test positives)", ""), v, c, h) for n, v, c, h in MEASURES[3:]], 0.4)
ax2.set_title("(b) Share of test positives", loc="left", pad=2)
ax2.set_ylim(0, 150)
ax2.set_xticks(range(len(SPLITS)))
ax2.set_xticklabels([lab for _, lab in SPLITS], fontsize=fs.SMALL)
ax2.tick_params(axis="x", pad=9)
for i, (s, _) in enumerate(SPLITS):
    ax2.plot(i, -0.06, marker=fs.SPLIT[s]["marker"], color=fs.SPLIT[s]["color"], mec=fs.INK, mew=0.4,
             ms=fs.MARKER, transform=ax2.get_xaxis_transform(), clip_on=False)
fig.tight_layout(pad=0.3, h_pad=0.6)
fs.save(fig, os.path.join(PAPER, "figures", "leakage_profile.pdf"))
print("wrote figures/leakage_profile.pdf;", "checks passed" if not fails else f"FAILED {fails}")
sys.exit(1 if fails else 0)
