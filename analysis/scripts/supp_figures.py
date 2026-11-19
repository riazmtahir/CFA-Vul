#!/usr/bin/env python3
"""supp_figures.py -- Supplementary Figs. S2-S4, every value read from the result files.

S2  per-seed test PR-AUC of every DiverseVul split (dots), split mean (bar) and length-only baseline
    (tick); source: the 25 run JSONs (same list as per_seed_table.py).
S4  H4b composition from results/h4_precheck_1022/h4b_precheck.json:
    (a) length-decile distribution of the removed functions (strata counts; identical for the
        decontamination and the size-control removal by construction; the kept set's decile
        counts are not in the file, so it is not plotted);
    (b) top-10 project shares and (c) top-10 CWE shares of removed-by-decontamination,
        removed-by-size-control and kept training functions.
S3  exploratory Croft re-scoring: PR-AUC full vs without family-leaked test functions with 95%
    bootstrap CIs (results/cpu_checks_1021/rq2_diversevul_phase2/croft_rescore.json) and lift
    (PR-AUC / positive rate) with the paired-bootstrap CI of its change
    (results/h4_precheck_1022/h4_precheck.log, step 6/7; no JSON copy of the lift CI exists).
Values cited in the paper are asserted (exit 1 on mismatch). Vector PDF, fonts embedded, black/grey.
Usage (repository root): python analysis/scripts/supp_figures.py
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
R = os.path.join(os.path.dirname(PAPER), "results")
OUT = os.path.join(PAPER, "supplementary")
sys.path.insert(0, HERE)
import figstyle as fs  # noqa: E402  shared style (13 Nov 2026): supplement print width, fonts >= 7 pt
fs.apply()
fails = []


def expect(name, got, want, tol):
    if abs(got - want) > tol:
        fails.append((name, got, want))


def despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# ------------------------------------------------------------------ S2
RUNS = [("rq2_diversevul_final", "published", (1, 2, 3)),
        ("rq2_diversevul_final", "random_control", (1, 2, 3)), ("rq2_diversevul_phase2", "random_control", (4, 5)),
        ("rq2_diversevul_final", "family_disjoint", (1, 2, 3)), ("rq2_diversevul_phase2", "family_disjoint", (4, 5)),
        ("rq2_diversevul_phase2", "project_disjoint", (1, 2, 3)),
        ("rq2_diversevul_phase3", "commit_disjoint", (1, 2, 3)),
        ("rq2_diversevul_phase4b", "commit_decon", (1, 2, 3)),
        ("rq2_diversevul_phase4b", "commit_sizectrl", (1, 2, 3))]
LABEL = {"published": "Published", "random_control": "Random control", "family_disjoint": "Family-disjoint",
         "project_disjoint": "Project-disjoint", "commit_disjoint": "Commit-disjoint",
         "commit_decon": "Decontaminated\n(H4b)", "commit_sizectrl": "Matched stratified\ncontrol (H4b)"}
pr, floor = {}, {}
for d, arm, seeds in RUNS:
    for s in seeds:
        r = json.load(open(os.path.join(R, d, f"codebert-base_{arm}_seed{s}.json"), encoding="utf-8"))
        pr.setdefault(arm, []).append((s, r["test"]["pr_auc"]))
        floor.setdefault(arm, set()).add(r["length_only_floor"]["pr_auc"])
MEANS = {"published": 0.2492, "random_control": 0.2552, "family_disjoint": 0.2569, "project_disjoint": 0.1472,
         "commit_disjoint": 0.1724, "commit_decon": 0.1562, "commit_sizectrl": 0.1826}
for arm, want in MEANS.items():
    expect(f"S2 mean {arm}", st.mean(v for _, v in pr[arm]), want, 5e-5)
    assert len(floor[arm]) == 1, arm
fig, ax = plt.subplots(figsize=(fs.SUPP, 2.6))
for i, arm in enumerate(LABEL):
    vals = [v for _, v in pr[arm]]
    ax.bar(i, st.mean(vals), width=0.55, color="0.88", edgecolor=fs.INK, linewidth=fs.EDGE_LW)
    ax.plot([i - 0.32, i + 0.32], [next(iter(floor[arm]))] * 2, color="black", linewidth=1.2, linestyle="--")
    xs = [i + (k - (len(vals) - 1) / 2) * 0.08 for k in range(len(vals))]
    ax.scatter(xs, vals, s=14, color=fs.SPLIT[arm]["color"], marker=fs.SPLIT[arm]["marker"], edgecolors=fs.INK,
               linewidths=0.4, zorder=3)
ax.set_xticks(range(len(LABEL)))
ax.set_xticklabels(list(LABEL.values()), fontsize=fs.SMALL)
ax.set_ylabel("Test PR-AUC")
fs.grid(ax)
ax.set_ylim(0, 0.3)
despine(ax)
fig.tight_layout(pad=0.4)
fs.save(fig, os.path.join(OUT, "fig_s2_per_seed_pr_auc.pdf"))

# ------------------------------------------------------------------ S3
pb = json.load(open(os.path.join(R, "h4_precheck_1022", "h4b_precheck.json"), encoding="utf-8"))
comp, strata = pb["composition"], pb["strata"]
expect("H4b removed", comp["removed_by_decon"]["n"], 35192, 0)
expect("H4b kept", comp["kept"]["n"], 229583, 0)
expect("H4b linux decon", dict(comp["removed_by_decon"]["top10_projects"])["linux"], 27.49, 0.005)
expect("H4b linux control", dict(comp["removed_by_control"]["top10_projects"])["linux"], 17.32, 0.005)
dec = [sum(strata[f"label{lab}_decile{d}"]["decon"] for lab in (0, 1)) for d in range(10)]
ctl = [sum(strata[f"label{lab}_decile{d}"]["control"] for lab in (0, 1)) for d in range(10)]
assert dec == ctl and sum(dec) == comp["removed_by_decon"]["n"]
SETS = [("removed_by_decon", "Removed: decontamination", fs.SPLIT["decontaminated"]["color"], ""),
        ("removed_by_control", "Removed: matched stratified control", fs.SPLIT["matched_control"]["color"], "...."),
        ("kept", "Kept (decontaminated training set)", "white", "////")]
fig, axes = plt.subplots(1, 3, figsize=(fs.SUPP, 3.3), gridspec_kw={"width_ratios": [1.0, 1.25, 1.25]})
ax = axes[0]
ax.bar(range(1, 11), [100 * x / sum(dec) for x in dec], color="0.75", edgecolor=fs.INK, linewidth=fs.EDGE_LW)
ax.axhline(10, **fs.REF)
ax.set_xticks(range(1, 11))
ax.set_xlabel("Length decile of the\ncommit-disjoint training set")
ax.set_ylabel("% of removed functions")
ax.set_title("(a) Length deciles, both removals")
despine(ax)
for ax, key, title in ((axes[1], "top10_projects", "(b) Top-10 projects"), (axes[2], "top10_cwes", "(c) Top-10 CWEs")):
    names = []
    for k, _, _, _ in SETS:
        for n, _ in comp[k][key]:
            if n not in names:
                names.append(n)
    h = 0.27
    for j, (k, lab, c, hat) in enumerate(SETS):
        share = dict(comp[k][key])
        ys = [i + (j - 1) * h for i in range(len(names))]
        ax.barh(ys, [share.get(n, 0) for n in names], height=h, color=c, edgecolor=fs.INK, linewidth=0.4,
                hatch=hat, label=lab)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=fs.SMALL)
    ax.invert_yaxis()
    ax.set_xlabel("% of the set (0 = not in its top 10)")
    ax.set_title(title)
    despine(ax)
handles, labels = axes[2].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=fs.SMALL, ncol=3, loc="lower center")
fig.tight_layout(pad=0.4, rect=(0, 0.06, 1, 1))
fs.save(fig, os.path.join(OUT, "fig_s4_h4b_composition.pdf"))

# ------------------------------------------------------------------ S4
cr = json.load(open(os.path.join(R, "cpu_checks_1021", "rq2_diversevul_phase2", "croft_rescore.json"),
                    encoding="utf-8"))["arms"]
log = open(os.path.join(R, "h4_precheck_1022", "h4_precheck.log"), encoding="utf-8").read()
step6 = log[log.index("6/7  Croft re-scoring"):log.index("7/7")]
lift_ci = {}
for line in step6.splitlines():
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    if len(c) == 10 and c[0] in cr:
        m = c[9].replace("+", "")
        lift_ci[c[0]] = (float(m.split("[")[0]), *map(float, m.split("[")[1].rstrip("]").split(",")))
ARMS = [("random_control", "Random control"), ("family_disjoint", "Family-disjoint"),
        ("project_disjoint", "Project-disjoint"), ("commit_disjoint", "Commit-disjoint")]
expect("Croft random full", cr["random_control"]["pr_auc_full"], 0.2572, 5e-5)
expect("Croft random without", cr["random_control"]["pr_auc_without_leaked"], 0.2046, 5e-5)
expect("Croft commit lift full", cr["commit_disjoint"]["pr_over_pos_full"], 3.02, 0.005)
expect("Croft commit lift without", cr["commit_disjoint"]["pr_over_pos_without_leaked"], 2.66, 0.005)
expect("Croft commit dlift", lift_ci["commit_disjoint"][0], -0.37, 0.005)
fig, axes = plt.subplots(1, 3, figsize=(fs.SUPP, 2.5))
for i, (arm, lab) in enumerate(ARMS):
    a = cr[arm]
    for j, (key, ci_key, c) in enumerate((("pr_auc_full", "pr_auc_full_ci95", "white"),
                                          ("pr_auc_without_leaked", "pr_auc_without_leaked_ci95", "0.6"))):
        v, (lo, hi) = a[key], a[ci_key]
        axes[0].bar(i + (j - 0.5) * 0.36, v, width=0.36, color=c, edgecolor=fs.INK, linewidth=fs.EDGE_LW,
                    yerr=[[v - lo], [hi - v]], capsize=2, error_kw={"elinewidth": 0.6},
                    label=("full test set" if j == 0 else "without family-leaked") if i == 0 else None)
        lift = a["pr_over_pos_full"] if j == 0 else a["pr_over_pos_without_leaked"]
        axes[1].bar(i + (j - 0.5) * 0.36, lift, width=0.36, color=c, edgecolor=fs.INK, linewidth=fs.EDGE_LW)
    m, lo, hi = lift_ci[arm]
    axes[2].errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt=fs.SPLIT[arm]["marker"], color=fs.SPLIT[arm]["color"],
                     mec=fs.INK, mew=0.4, capsize=2, elinewidth=fs.ERR_LW)
for ax, t in ((axes[0], "(a) PR-AUC, 95% CI"), (axes[1], "(b) Lift = PR-AUC / positive rate")):
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([lab.replace("-", "-\n", 1) if "disjoint" in lab else lab.replace(" ", "\n") for _, lab in ARMS],
                       fontsize=fs.SMALL)
    ax.set_title(t)
    despine(ax)
axes[0].legend(frameon=False, fontsize=fs.SMALL, loc="upper right", bbox_to_anchor=(1.0, 1.0))
axes[0].set_ylim(0, 0.36)
axes[2].axvline(0, **fs.REF)
axes[2].set_yticks(range(len(ARMS)))
axes[2].set_yticklabels([lab for _, lab in ARMS], fontsize=fs.SMALL)
axes[2].invert_yaxis()
axes[2].set_title("(c) Change in lift, 95% CI")
despine(axes[2])
fig.tight_layout(pad=0.4)
fs.save(fig, os.path.join(OUT, "fig_s3_croft_rescoring.pdf"))

print("wrote supplementary/fig_s2..s4;", "checks passed" if not fails else f"FAILED {fails}")
sys.exit(1 if fails else 0)
