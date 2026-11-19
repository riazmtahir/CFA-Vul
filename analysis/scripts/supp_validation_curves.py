#!/usr/bin/env python3
"""supp_validation_curves.py -- Supplementary Fig. S1: epoch-wise validation PR-AUC and validation
best-F1 per split (mean over seeds, min-max band), from the epoch histories in the result JSONs.

Reads the 25 DiverseVul run JSONs (same file list as per_seed_table.py) and writes
  paper/supplementary/fig_s1_validation_curves.pdf
  paper/supplementary/fig_s1_selected_epochs.json   selected (best) epoch and epochs run per seed
At epochs where some runs had already stopped early, mean and band use the runs still training;
the number of runs per epoch is in the JSON. Vector PDF, fonts embedded, black/grey line styles.
Usage (repository root): python analysis/scripts/supp_validation_curves.py
"""
from __future__ import annotations

import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
R = os.path.join(os.path.dirname(PAPER), "results")
RUNS = [("rq2_diversevul_final", "published", (1, 2, 3)),
        ("rq2_diversevul_final", "random_control", (1, 2, 3)), ("rq2_diversevul_phase2", "random_control", (4, 5)),
        ("rq2_diversevul_final", "family_disjoint", (1, 2, 3)), ("rq2_diversevul_phase2", "family_disjoint", (4, 5)),
        ("rq2_diversevul_phase2", "project_disjoint", (1, 2, 3)),
        ("rq2_diversevul_phase3", "commit_disjoint", (1, 2, 3)),
        ("rq2_diversevul_phase4b", "commit_decon", (1, 2, 3)),
        ("rq2_diversevul_phase4b", "commit_sizectrl", (1, 2, 3))]
LABEL = {"published": "Published", "random_control": "Random control", "family_disjoint": "Family-disjoint",
         "project_disjoint": "Project-disjoint", "commit_disjoint": "Commit-disjoint",
         "commit_decon": "Decontaminated (H4b)", "commit_sizectrl": "Matched stratified control (H4b)"}
STYLE = {"published": ("black", "-"), "random_control": ("black", "--"), "family_disjoint": ("0.45", "-"),
         "project_disjoint": ("0.45", "--"), "commit_disjoint": ("black", ":"), "commit_decon": ("0.45", ":"),
         "commit_sizectrl": ("0.65", "-.")}

hist, sel = {}, {}
for d, arm, seeds in RUNS:
    for s in seeds:
        r = json.load(open(os.path.join(R, d, f"codebert-base_{arm}_seed{s}.json"), encoding="utf-8"))
        assert (r["arm"], r["seed"]) == (arm, s)
        hist.setdefault(arm, []).append({e["epoch"]: (e["valid"]["pr_auc"], e["valid"]["best_f1"]) for e in r["history"]})
        sel.setdefault(arm, []).append({"seed": s, "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"]})
assert sum(len(v) for v in hist.values()) == 25
# values cited in rq2.tex (Section 5.1); numbers.md "Supplementary Fig. S1"
EXPECT_BEST = {"published": [4, 9, 5], "random_control": [3, 4, 3, 3, 4], "family_disjoint": [5, 5, 9, 4, 5],
               "project_disjoint": [1, 1, 1], "commit_disjoint": [7, 3, 3], "commit_decon": [4, 4, 5],
               "commit_sizectrl": [4, 3, 4]}
for arm, want in EXPECT_BEST.items():
    assert [x["best_epoch"] for x in sel[arm]] == want, arm
runs_all = [x["epochs_run"] for v in sel.values() for x in v]
assert (min(runs_all), max(runs_all)) == (3, 11) and all(x["epochs_run"] < 15 for v in sel.values() for x in v)

import figstyle as fs  # noqa: E402  shared style (13 Nov 2026): supplement print width, fonts >= 7 pt
fs.apply()
STYLE = {arm: (fs.SPLIT[arm]["color"], fs.SPLIT[arm]["ls"]) for arm in LABEL}   # one colour/line per split
fig, axes = plt.subplots(1, 2, figsize=(fs.SUPP, 2.9))
curves = {}
for k, (metric, title) in enumerate(((0, "Validation PR-AUC"), (1, "Validation best-F1"))):
    ax = axes[k]
    for arm in LABEL:
        runs = hist[arm]
        epochs = sorted({e for h in runs for e in h})
        xs, mean, lo, hi, n = [], [], [], [], []
        for e in epochs:
            v = [h[e][metric] for h in runs if e in h]
            xs.append(e); mean.append(st.mean(v)); lo.append(min(v)); hi.append(max(v)); n.append(len(v))
        c, ls = STYLE[arm]
        ax.fill_between(xs, lo, hi, color=c, alpha=0.12, linewidth=0)
        ax.plot(xs, mean, color=c, linestyle=ls, marker=fs.SPLIT[arm]["marker"], markersize=3,
                markevery=2, label=LABEL[arm])
        curves.setdefault(arm, {})[title] = {"epochs": xs, "mean": mean, "min": lo, "max": hi, "runs": n}
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    fs.grid(ax)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=fs.SMALL,
           bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(pad=0.4, rect=(0, 0.17, 1, 1))
out = os.path.join(PAPER, "supplementary")
os.makedirs(out, exist_ok=True)
fs.save(fig, os.path.join(out, "fig_s1_validation_curves.pdf"))
json.dump({"selected_epochs": sel, "curves": curves},
          open(os.path.join(out, "fig_s1_selected_epochs.json"), "w", encoding="utf-8"), indent=1)
for arm in LABEL:
    print(f"{LABEL[arm]:22s} best epochs {[x['best_epoch'] for x in sel[arm]]}  epochs run {[x['epochs_run'] for x in sel[arm]]}")
