#!/usr/bin/env python3
"""h4b_numbers.py -- recompute and check every H4b number (RESEARCH_PLAN §13.8), independently of
analyze_phase4b.py (no import from it or from analyze_phase2.py).

Reads
  results/rq2_diversevul_phase4b/codebert-base_commit_{sizectrl,decon}_seed{1,2,3}.json   per-seed runs
  results/rq2_diversevul_phase4b/phase4b_analysis.json                                     server analysis
  results/rq2_diversevul_phase3/codebert-base_commit_disjoint_seed{1,2,3}.json             Phase 3 (descriptive)
  results/h4_precheck_1022/{h4_precheck,h4b_precheck,h4_design,diversevul_h4b_splits_report}.json
  results/h4_precheck_1022/direct_edges/diversevul_hr_direct_summary.json
Recomputes per-seed Δ(commit_sizectrl → commit_decon) = decon − sizectrl for PR-AUC, headroom (PR-AUC −
length-only floor), MCC, ROC-AUC; mean, sd (ddof 1), 95% t-interval with t(0.975, 2) = 4.303; applies the
§13.8 rule verbatim; checks the server analysis and the values printed in the paper. Any mismatch -> exit 1.
Output: analysis/scripts/out/h4b_numbers.json.
Usage (repository root): python analysis/scripts/h4b_numbers.py
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
P4B = os.path.join(ROOT, "results", "rq2_diversevul_phase4b")
P3 = os.path.join(ROOT, "results", "rq2_diversevul_phase3")
PC = os.path.join(ROOT, "results", "h4_precheck_1022")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
T975_DF2 = 4.303
SEEDS = (1, 2, 3)
checks, fails = [], []


def chk(name, got, want, nd, src):
    ok = (got == want) if nd is None else abs(got - want) <= 0.5 * 10 ** (-nd) + 1e-9
    checks.append({"name": name, "got": got, "want": want, "decimals": nd, "src": src, "ok": ok})
    if not ok:
        fails.append(checks[-1])


def load(*p):
    return json.load(open(os.path.join(*p), encoding="utf-8"))


def run(d, arm, s):
    r = load(d, f"codebert-base_{arm}_seed{s}.json")
    t, fl = r["test"], r["length_only_floor"]["pr_auc"]
    return {"pr_auc": t["pr_auc"], "headroom": t["pr_auc"] - fl, "mcc": t["mcc"], "roc_auc": t["roc_auc"],
            "floor": fl, "n_test": t["n"], "pos_rate": t["positive_rate"], "n_train": r["n_train"],
            "arm": r["arm"], "seed": r["seed"]}


R = {arm: {s: run(P4B, arm, s) for s in SEEDS} for arm in ("commit_sizectrl", "commit_decon")}
for arm, rs in R.items():
    for s, m in rs.items():
        chk(f"{arm} s{s}: file arm/seed", (m["arm"], m["seed"]) == (arm, s), True, None, "run json")
# per-seed PR-AUC as printed by the server (phase4b_analysis.md)
PR = {"commit_sizectrl": (0.1843, 0.1901, 0.1735), "commit_decon": (0.1575, 0.1597, 0.1513)}
for arm, vals in PR.items():
    for s, v in zip(SEEDS, vals):
        chk(f"{arm} s{s} PR-AUC", R[arm][s]["pr_auc"], v, 4, "run json")
# same test set in every run -> same length-only floor; so Δheadroom == ΔPR-AUC by construction
floors = {R[a][s]["floor"] for a in R for s in SEEDS}
chk("one length-only floor across all six runs", len(floors), 1, 0, "run json")
chk("length-only floor value", floors.pop(), 0.1412, 4, "run json")
chk("one test set size across all six runs", len({R[a][s]["n_test"] for a in R for s in SEEDS}), 1, 0, "run json")
chk("test n == commit_disjoint test (33,172)", R["commit_decon"][1]["n_test"], 33172, 0, "run json")
chk("train n equal in both arms", len({R[a][s]["n_train"] for a in R for s in SEEDS}), 1, 0, "run json")


def t_ci(xs):
    m, sd = st.mean(xs), st.stdev(xs)
    h = T975_DF2 * sd / math.sqrt(len(xs))
    return {"n": len(xs), "mean": m, "sd": sd, "ci": [m - h, m + h]}


D = {}
for k in ("pr_auc", "headroom", "mcc", "roc_auc"):
    per = [R["commit_decon"][s][k] - R["commit_sizectrl"][s][k] for s in SEEDS]
    D[k] = {**t_ci(per), "per_seed": dict(zip(map(str, SEEDS), per))}
EXP = {"pr_auc": (-0.0265, -0.0367, -0.0163), "headroom": (-0.0265, -0.0367, -0.0163),
       "mcc": (-0.0291, -0.0430, -0.0152), "roc_auc": (-0.0176, -0.0265, -0.0088)}
for k, (m, lo, hi) in EXP.items():
    chk(f"Δ {k} mean", D[k]["mean"], m, 4, "recomputed")
    chk(f"Δ {k} CI lo", D[k]["ci"][0], lo, 4, "recomputed")
    chk(f"Δ {k} CI hi", D[k]["ci"][1], hi, 4, "recomputed")
for s in SEEDS:
    chk(f"Δheadroom == ΔPR-AUC, seed {s} (same floor)", D["headroom"]["per_seed"][str(s)],
        D["pr_auc"]["per_seed"][str(s)], 12, "recomputed")

# §13.8 rule, verbatim cases
lo, hi = D["pr_auc"]["ci"]
case = "below 0" if hi < 0 else ("above 0" if lo > 0 else "includes 0")
RULE = {"below 0": "direct high-recall Type-3 clones in training inflate PR-AUC beyond commit sharing",
        "includes 0": "no evidence of inflation by direct high-recall clones beyond commit sharing",
        "above 0": "reversed: removing the direct clones raises PR-AUC"}
chk("§13.8 case from the interval", case, "below 0", None, "rule")
negligible = abs(D["pr_auc"]["mean"]) <= 0.01
chk("|Δ| <= 0.01 (practically negligible)?", negligible, False, None, "rule")
chk("CI upper bound below -0.01", hi < -0.01, True, None, "rule")

# server analysis agrees
an = load(P4B, "phase4b_analysis.json")
chk("server verdict case", an["verdict_pr_auc"], case, None, "phase4b_analysis.json")
chk("server verdict text carries the rule's words", RULE[case] in an["verdict"], True, None, "phase4b_analysis.json")
chk("server: pre-check PASS", an["precheck_verdict"], "PASS", None, "phase4b_analysis.json")
chk("server: design H4b", an["selected_design"], "H4b", None, "phase4b_analysis.json")
for k in D:
    sv = an["deltas_sizectrl_to_decon"][k]
    chk(f"server Δ {k} mean == recomputed", sv["mean"], D[k]["mean"], 10, "phase4b_analysis.json")
    chk(f"server Δ {k} CI == recomputed", max(abs(a - b) for a, b in zip(sv["ci"], D[k]["ci"])), 0.0, 3, "phase4b_analysis.json")
chk("server: no missing runs", an["missing_runs"], [], None, "phase4b_analysis.json")

# derived values cited in the paper
mean = {a: {k: st.mean(R[a][s][k] for s in SEEDS) for k in ("pr_auc", "headroom", "mcc", "roc_auc")} for a in R}
rel = D["pr_auc"]["mean"] / mean["commit_sizectrl"]["pr_auc"]
chk("size-control mean PR-AUC", mean["commit_sizectrl"]["pr_auc"], 0.1826, 4, "recomputed")
chk("decon mean PR-AUC", mean["commit_decon"]["pr_auc"], 0.1562, 4, "recomputed")
chk("relative change (%)", round(100 * rel, 1), -14.5, 1, "recomputed")
chk("mean headroom, size control", mean["commit_sizectrl"]["headroom"], 0.041, 3, "recomputed")
chk("mean headroom, decon", mean["commit_decon"]["headroom"], 0.015, 3, "recomputed")
for a, (mc, roc) in {"commit_sizectrl": (0.2125, 0.7575), "commit_decon": (0.1834, 0.7398)}.items():
    chk(f"{a} mean MCC", mean[a]["mcc"], mc, 4, "recomputed")
    chk(f"{a} mean ROC-AUC", mean[a]["roc_auc"], roc, 4, "recomputed")
p3 = st.mean(load(P3, f"codebert-base_commit_disjoint_seed{s}.json")["test"]["pr_auc"] for s in SEEDS)
chk("Phase 3 commit_disjoint mean PR-AUC (descriptive)", p3, 0.1724, 4, "phase3 run json")

# pre-checks and removal
h4 = load(PC, "h4_precheck.json")
chk("H4 pre-check FAIL", h4["verdict"], "FAIL", None, "h4_precheck.json")
chk("H4 largest group % of corpus", h4["largest_group_pct"], 97.18, 2, "h4_precheck.json")
chk("H4 largest group functions", h4["largest_group"], 321171, 0, "h4_precheck.json")
chk("design file selects H4b", load(PC, "h4_design.json")["design"], "H4b", None, "h4_design.json")
pb = load(PC, "h4b_precheck.json")
chk("H4b pre-check PASS", pb["verdict"], "PASS", None, "h4b_precheck.json")
chk("removed functions", pb["removed"], 35192, 0, "h4b_precheck.json")
chk("removed % of commit_disjoint train", pb["removed_pct_of_train"], 13.29, 2, "h4b_precheck.json")
chk("removed % recomputed", 100 * pb["removed"] / pb["commit_disjoint_train"], 13.29, 2, "h4b_precheck.json")
chk("removed positives", pb["removed_pos"], 1646, 0, "h4b_precheck.json")
chk("borrowed", pb["borrowed"], 0, 0, "h4b_precheck.json")
proj = lambda part: dict(pb["composition"][part]["top10_projects"])
for p, d, c in (("linux", 27.49, 17.32), ("linux-2.6", 5.52, 2.87)):
    chk(f"decon removal share {p}", proj("removed_by_decon")[p], d, 2, "h4b_precheck.json")
    chk(f"control removal share {p}", proj("removed_by_control")[p], c, 2, "h4b_precheck.json")
for p, d in (("gpac", 6.75), ("openssl", 3.72)):
    chk(f"decon removal share {p}", proj("removed_by_decon")[p], d, 2, "h4b_precheck.json")
    chk(f"{p} not in the control removal's top 10", p in proj("removed_by_control"), False, None, "h4b_precheck.json")
    chk(f"{p} share of kept train (top 10)", p in proj("kept"), True, None, "h4b_precheck.json")
chk("kept share gpac", proj("kept")["gpac"], 1.44, 2, "h4b_precheck.json")
chk("kept share openssl", proj("kept")["openssl"], 1.53, 2, "h4b_precheck.json")
chk("kept share linux", proj("kept")["linux"], 20.69, 2, "h4b_precheck.json")
for part, med in (("removed_by_decon", 182), ("removed_by_control", 186)):
    chk(f"{part} median chars", pb["composition"][part]["median_chars"], med, 0, "h4b_precheck.json")
chk("direct leak, published test (%) (rq1)",
    load(PC, "direct_edges", "diversevul_hr_direct_summary.json")["splits"]["published"]["direct_leaked_test_pct"],
    56.45, 2, "direct summary")
de = load(PC, "direct_edges", "diversevul_hr_direct_summary.json")["splits"]["commit_disjoint"]
chk("direct leak, commit_disjoint test (%)", de["direct_leaked_test_pct"], 37.15, 2, "direct summary")
chk("direct leak, commit_disjoint test positives (%)", de["direct_leaked_test_pos_pct"], 40.4, 2, "direct summary")
chk("train hit == removal", de["train_hit"], pb["removed"], 0, "direct summary")

# EXPLORATORY Croft re-scoring with the lift CI: step 6/7 of h4_precheck.log (22 Oct 2026); the log is the
# only copy of the table (no JSON in the backup). Lift = PR-AUC / positive rate; Δ lift = without - full.
LIFT = {"random_control": (4.49, 4.45, -0.04, -0.31, 0.30), "project_disjoint": (2.58, 2.48, -0.10, -0.13, -0.07),
        "commit_disjoint": (3.02, 2.66, -0.37, -0.48, -0.26)}
log = open(os.path.join(PC, "h4_precheck.log"), encoding="utf-8").read()
step6 = log[log.index("6/7  Croft re-scoring"):log.index("7/7")]
lift = {}
for line in step6.splitlines():
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    if len(c) == 10 and c[0] in LIFT:
        full, wo = (float(x) for x in c[8].split("→"))
        mean_ci = c[9].replace("+", "")
        m, lo, hi = (float(x) for x in [mean_ci.split("[")[0]] + mean_ci.split("[")[1].rstrip("]").split(","))
        lift[c[0]] = (full, wo, m, lo, hi)
for arm, want in LIFT.items():
    for name, g, w in zip(("lift full", "lift without leaked", "Δ lift", "Δ lift CI lo", "Δ lift CI hi"),
                          lift[arm], want):
        chk(f"Croft lift {arm}: {name}", g, w, 2, "h4_precheck.log step 6/7")

os.makedirs(OUT, exist_ok=True)
json.dump({"per_run": {a: {f"s{s}": m for s, m in rs.items()} for a, rs in R.items()}, "deltas": D,
           "means": mean, "relative_change_pr_auc": rel, "case": case, "rule_text": RULE[case],
           "practically_negligible": negligible, "phase3_commit_mean_pr_auc": p3, "checks": checks},
          open(os.path.join(OUT, "h4b_numbers.json"), "w", encoding="utf-8"), indent=1)
print(f"{len(checks)} checks; {len(fails)} failed")
for f in fails:
    print("  FAIL", f["name"], "got", f["got"], "want", f["want"], f["src"])
sys.exit(1 if fails else 0)
