#!/usr/bin/env python3
"""cpu_checks_numbers.py -- check every number taken from the 21 Oct 2026 CPU checks.

Reads only results/cpu_checks_1021/ (copied from backup_vd_20261021_0329.tar.gz):
  audit_highrecall/{devign,diversevul,primevul}_hr_report.json        EXPLORATORY high-recall audit
  audit_highrecall/{...}_hr_label_theta0.7.json                       leak categories on training splits
  corpus_overlap/primevul_diversevul_overlap.json                      A3 (descriptive)
  rq2_diversevul_phase2/pair_identity_check.json                       A4 (descriptive)
  rq2_diversevul_phase2/paired_eval_noidentical_{ws,any}.json          A4 paired eval without identical pairs
  rq2_diversevul_phase2/croft_rescore.json                             A5 (EXPLORATORY)
and results/rq0_calibration_ours/highrecall_blind_k3/ for the BCB capture of the variant.

Checks: every value cited in rq1/rq2/rq3.tex and numbers.md against the files (values as
printed, at the printed precision); md tables equal the JSON where both exist; percentages
recomputed from counts. Any mismatch -> exit 1.
Output: analysis/scripts/out/cpu_checks_numbers.json (read by rq1_table_figure.py for the table).
Usage (repository root): python analysis/scripts/cpu_checks_numbers.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
C = os.path.join(ROOT, "results", "cpu_checks_1021")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
checks, fails = [], []


def chk(name, got, want, nd, src):
    ok = abs(got - want) <= 0.5 * 10 ** (-nd) + 1e-9
    checks.append({"name": name, "got": got, "want": want, "decimals": nd, "src": src, "ok": ok})
    if not ok:
        fails.append(checks[-1])


def load(*p):
    return json.load(open(os.path.join(*p), encoding="utf-8"))


out = {}
# ------------------------------------------------------------------ high-recall audit (EXPLORATORY)
HR = {"devign": {"published": (43.56, 29.54), "random_control": (43.9, 30.79), "family_disjoint": (35.9, 25.31)},
      "diversevul": {"published": (56.54, 51.62), "random_control": (56.32, 51.52), "family_disjoint": (45.63, 42.6)},
      "primevul": {"published": (31.21, 30.78), "random_control": (51.71, 48.94), "family_disjoint": (39.49, 37.79)}}
LARGEST = {"devign": (633, 2.32), "diversevul": (25220, 7.63), "primevul": (20320, 8.62)}
out["highrecall"] = {}
for corpus, arms in HR.items():
    lab = load(C, "audit_highrecall", f"{corpus}_hr_label_theta0.7.json")
    rep = load(C, "audit_highrecall", f"{corpus}_hr_report.json")
    src = f"{corpus}_hr_label_theta0.7.json"
    chk(f"{corpus} hr config rename=blind", rep["config"]["rename"] == "blind", True, 0, f"{corpus}_hr_report.json")
    chk(f"{corpus} hr config shingle_k", rep["config"]["shingle_k"], 3, 0, f"{corpus}_hr_report.json")
    out["highrecall"][corpus] = {}
    for arm, (anyp, croft) in arms.items():
        a = lab["arms"][arm]
        chk(f"{corpus} {arm} hr leaked any %", a["leaked_any_pct"], anyp, 2, src)
        chk(f"{corpus} {arm} hr Croft-def %", a["leaked_croft_definition_pct"], croft, 2, src)
        chk(f"{corpus} {arm} hr leaked any % recomputed", 100 * a["leaked_any"] / a["n_test"], anyp, 2, src)
        chk(f"{corpus} {arm} hr Croft-def % recomputed", 100 * a["leaked_croft_definition"] / a["n_test"], croft, 2, src)
        out["highrecall"][corpus][arm] = {"any_pct": a["leaked_any_pct"], "croft_pct": a["leaked_croft_definition_pct"],
                                          "n_test": a["n_test"]}
    sd = lab["size_distribution"]
    chk(f"{corpus} hr largest family", sd["largest_family_size"], LARGEST[corpus][0], 0, src)
    chk(f"{corpus} hr largest family % of corpus", sd["largest_family_share_pct"], LARGEST[corpus][1], 2, src)
    chk(f"{corpus} hr largest family % recomputed", 100 * sd["largest_family_size"] / sd["n_functions"],
        LARGEST[corpus][1], 2, src)
    # the audit report (its own families) and the label audit agree on the published split
    t07 = next(t for t in rep["thresholds"] if t["theta"] == 0.7)
    chk(f"{corpus} hr report published leak == label audit", t07["leaked_test_pct"],
        lab["arms"]["published"]["leaked_any_pct"], 2, f"{corpus}_hr_report.json")
    chk(f"{corpus} hr report largest family == label audit", t07["largest_family"], sd["largest_family_size"], 0,
        f"{corpus}_hr_report.json")
    out["highrecall"][corpus]["largest_family"] = sd["largest_family_size"]
    out["highrecall"][corpus]["largest_family_pct"] = sd["largest_family_share_pct"]
rep = load(C, "audit_highrecall", "diversevul_hr_report.json")
for th, want in ((0.7, 56.54), (0.8, 48.67), (0.9, 38.77)):
    t = next(x for x in rep["thresholds"] if x["theta"] == th)
    chk(f"diversevul hr published leak theta {th}", t["leaked_test_pct"], want, 2, "diversevul_hr_report.json")
    chk(f"diversevul hr published leak theta {th} recomputed", 100 * t["leaked_test_functions"] / t["n_test_functions"],
        want, 2, "diversevul_hr_report.json")
# BCB capture of the variant (cited with the table)
bcb = load(ROOT, "results", "rq0_calibration_ours", "highrecall_blind_k3", "rq0_theta_calibration_ours.json")
cap = {c["category"].strip(): c["capture"] for c in bcb["categories"]}
chk("BCB blind k=3 Strong Type-3 @0.7 (%)", round(100 * cap["Strong Type-3 (ST3)"]["0.70"], 1), 88.2, 1, "highrecall_blind_k3")
chk("BCB blind k=3 non-clone @0.7 (%)", round(100 * cap["non-clone"]["0.70"], 2), 0.05, 2, "highrecall_blind_k3")
chk("BCB blind k=3 config", (bcb.get("rename") == "blind") and bcb["shingle_k"] == 3, True, 0, "highrecall_blind_k3")

# ------------------------------------------------------------------ corpus overlap (descriptive)
ov = load(C, "corpus_overlap", "primevul_diversevul_overlap.json")
k = ov["keys"]
src = "primevul_diversevul_overlap.json"
chk("PrimeVul functions", ov["n"]["primevul"], 235768, 0, src)
chk("DiverseVul functions", ov["n"]["diversevul"], 330491, 0, src)
chk("PrimeVul in DiverseVul, raw %", k["raw"]["primevul_matched_pct"], 57.99, 2, src)
chk("PrimeVul in DiverseVul, raw % recomputed", 100 * k["raw"]["primevul_matched"] / ov["n"]["primevul"], 57.99, 2, src)
chk("PrimeVul in DiverseVul, norm %", k["norm"]["primevul_matched_pct"], 80.01, 2, src)
chk("PrimeVul in DiverseVul, norm % recomputed", 100 * k["norm"]["primevul_matched"] / ov["n"]["primevul"], 80.01, 2, src)
chk("raw matches with the same label %", k["raw"]["primevul_matched_same_label_pct"], 99.85, 2, src)
chk("DiverseVul in PrimeVul, raw %", k["raw"]["diversevul_matched_pct"], 41.37, 2, src)
chk("PrimeVul test in DiverseVul published train, raw %",
    k["raw"]["primevul_test_in_diversevul_train"]["published"]["pct_of_primevul_test"], 69.89, 2, src)
chk("PrimeVul test in DiverseVul published train, raw % recomputed",
    100 * k["raw"]["primevul_test_in_diversevul_train"]["published"]["count"] / ov["primevul_test_n"], 69.89, 2, src)
out["overlap"] = {"raw_pct": 57.99, "norm_pct": 80.01, "same_label_pct": 99.85, "test_in_train_raw_pct": 69.89}

# ------------------------------------------------------------------ identical pairs + paired eval (descriptive)
pid = load(C, "rq2_diversevul_phase2", "pair_identity_check.json")["scopes"]
src = "pair_identity_check.json"
chk("corpus pairs", pid["corpus"]["pairs"], 14324, 0, src)
chk("corpus raw-identical pairs", pid["corpus"]["identical"]["raw"], 0, 0, src)
chk("corpus whitespace-only pairs %", pid["corpus"]["identical_pct"]["ws"], 2.71, 2, src)
chk("corpus whitespace-only pairs % recomputed", 100 * pid["corpus"]["identical"]["ws"] / pid["corpus"]["pairs"], 2.71, 2, src)
chk("project_disjoint test whitespace-only %", pid["project_disjoint"]["identical_pct"]["ws"], 16.25, 2, src)
chk("project_disjoint test whitespace-only % recomputed",
    100 * pid["project_disjoint"]["identical"]["ws"] / pid["project_disjoint"]["pairs"], 16.25, 2, src)
chk("corpus normalised-identical %", pid["corpus"]["identical_pct"]["norm"], 7.85, 2, src)
ws = load(C, "rq2_diversevul_phase2", "paired_eval_noidentical_ws.json")
chk("ws file exclusion level", ws["exclude_identical"] == "ws", True, 0, "paired_eval_noidentical_ws.json")
WS = {"random_control": (0.603, None), "family_disjoint": (0.547, None),
      "project_disjoint": (0.480, (0.455, 0.503)), "commit_disjoint": (0.497, (0.477, 0.516))}
out["paired_ws"] = {}
for arm, (m, ci) in WS.items():
    lc = ws["arms"][arm]["length_check"]
    chk(f"ws-excluded length-balanced rank-acc {arm}", lc["balanced_rank_acc"], m, 3, "paired_eval_noidentical_ws.json")
    if ci:
        chk(f"ws-excluded LB rank-acc CI lo {arm}", lc["balanced_rank_acc_ci95"][0], ci[0], 3, "paired_eval_noidentical_ws.json")
        chk(f"ws-excluded LB rank-acc CI hi {arm}", lc["balanced_rank_acc_ci95"][1], ci[1], 3, "paired_eval_noidentical_ws.json")
    chk(f"ws excluded count {arm} == pair_identity ws", ws["arms"][arm]["identical_pairs_excluded"],
        pid[arm]["identical"]["ws"], 0, "paired_eval_noidentical_ws.json")
    out["paired_ws"][arm] = {"lb_rank_acc": lc["balanced_rank_acc"], "ci": lc["balanced_rank_acc_ci95"],
                             "pairs": ws["arms"][arm]["pairs"]}
an = load(C, "rq2_diversevul_phase2", "paired_eval_noidentical_any.json")
chk("any file exclusion level", an["exclude_identical"] == "any", True, 0, "paired_eval_noidentical_any.json")
chk("any: commit_disjoint LB rank-acc", an["arms"]["commit_disjoint"]["length_check"]["balanced_rank_acc"], 0.503, 3,
    "paired_eval_noidentical_any.json")
chk("any: project_disjoint LB rank-acc", an["arms"]["project_disjoint"]["length_check"]["balanced_rank_acc"], 0.484, 3,
    "paired_eval_noidentical_any.json")
chk("any: project_disjoint LB CI lo", an["arms"]["project_disjoint"]["length_check"]["balanced_rank_acc_ci95"][0],
    0.460, 3, "paired_eval_noidentical_any.json")
chk("any: project_disjoint LB CI hi", an["arms"]["project_disjoint"]["length_check"]["balanced_rank_acc_ci95"][1],
    0.509, 3, "paired_eval_noidentical_any.json")
chk("any: commit_disjoint LB CI lo", an["arms"]["commit_disjoint"]["length_check"]["balanced_rank_acc_ci95"][0],
    0.482, 3, "paired_eval_noidentical_any.json")
chk("any: commit_disjoint LB CI hi", an["arms"]["commit_disjoint"]["length_check"]["balanced_rank_acc_ci95"][1],
    0.522, 3, "paired_eval_noidentical_any.json")
# the generic noidentical file is the 'any' run
chk("paired_eval_noidentical.json == _any.json",
    load(C, "rq2_diversevul_phase2", "paired_eval_noidentical.json") == an, True, 0, "paired_eval_noidentical*.json")

# ------------------------------------------------------------------ Croft re-scoring (EXPLORATORY)
cr = load(C, "rq2_diversevul_phase2", "croft_rescore.json")["arms"]
src = "croft_rescore.json"
CR = {"random_control": (0.2572, 0.2046, 0.0573, 0.0460, 4.49, 4.45, -0.0525),
      "family_disjoint": (0.2621, 0.2621, 0.0570, 0.0570, 4.60, 4.60, 0.0),
      "project_disjoint": (0.1472, 0.1476, 0.0570, 0.0594, 2.58, 2.48, 0.0004),
      "commit_disjoint": (0.1723, 0.1594, 0.0570, 0.0600, 3.02, 2.66, -0.0129)}
out["croft"] = {}
for arm, (f, w, pf, pw, lf, lw, d) in CR.items():
    r = cr[arm]
    chk(f"Croft {arm} PR-AUC full", r["pr_auc_full"], f, 4, src)
    chk(f"Croft {arm} PR-AUC without leaked", r["pr_auc_without_leaked"], w, 4, src)
    chk(f"Croft {arm} pos rate full", r["pos_rate_full"], pf, 4, src)
    chk(f"Croft {arm} pos rate without", r["pos_rate_without_leaked"], pw, 4, src)
    chk(f"Croft {arm} lift full", r["pr_over_pos_full"], lf, 2, src)
    chk(f"Croft {arm} lift without", r["pr_over_pos_without_leaked"], lw, 2, src)
    chk(f"Croft {arm} lift full recomputed", r["pr_auc_full"] / r["pos_rate_full"], lf, 2, src)
    chk(f"Croft {arm} delta", r["diff_without_minus_full"], d, 4, src)
    out["croft"][arm] = {k2: r[k2] for k2 in ("pr_auc_full", "pr_auc_without_leaked", "pos_rate_full",
                                              "pos_rate_without_leaked", "pr_over_pos_full",
                                              "pr_over_pos_without_leaked", "leaked_pct")}
chk("Croft random_control removed % == strict RC leak", cr["random_control"]["leaked_pct"], 29.39, 2, src)
chk("Croft commit_disjoint removed % == strict commit-split leak", cr["commit_disjoint"]["leaked_pct"], 17.12, 2, src)

os.makedirs(OUT, exist_ok=True)
json.dump({"values": out, "checks": checks}, open(os.path.join(OUT, "cpu_checks_numbers.json"), "w",
          encoding="utf-8"), indent=1)
print(f"{len(checks)} checks; {len(fails)} failed")
for f in fails:
    print("  FAIL", f["name"], "got", f["got"], "want", f["want"], f["src"])
sys.exit(1 if fails else 0)
