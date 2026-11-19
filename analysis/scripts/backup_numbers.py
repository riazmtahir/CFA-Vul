#!/usr/bin/env python3
"""backup_numbers.py -- recompute the ablation/sensitivity numbers that live only in the
backup archive (backup_vd_20261007_2030.tar.gz, extracted to results/from_backup/) and
check them against the project log.

Reads only results/from_backup/:
  rq2_devign_final/      Stage A2, Devign, converged protocol (CodeBERT, GraphCodeBERT)
  rq2_devign/            Gate 1, Devign, 3 epochs, unweighted CE (discarded protocol)
  rq2_diversevul/        DiverseVul, 3 epochs, unweighted CE (discarded protocol)
  sweep_devign/, sweep_diversevul/, lr_devign.txt, lr_diversevul.txt   Stage A LR sweeps
  devign_audit_report.json, diversevul_audit_report.json   theta=0.7 audits (config, 1-NN)

Paired deltas and 95% t-CIs use the same rule as rq2_numbers.py / analyze_phase2.py
(df = n-1). Any mismatch with the project log -> exit 1, nothing written.

Output: analysis/scripts/out/backup_numbers.json
Usage (repository root):  python analysis/scripts/backup_numbers.py
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BK = os.path.join(ROOT, "results", "from_backup")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
ARMS = ("published", "random_control", "family_disjoint")


def load(rel):
    with open(os.path.join(BK, rel), encoding="utf-8") as fh:
        return json.load(fh)


def t_ci(xs):
    n, m = len(xs), st.mean(xs)
    sd = st.stdev(xs)
    h = T975[n - 1] * sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "lo": m - h, "hi": m + h}


def block(d, model):
    """per-arm test PR-AUC over seeds 1-3, and paired deltas."""
    runs = {a: {s: load(f"{d}/{model}_{a}_seed{s}.json") for s in (1, 2, 3)} for a in ARMS}
    pr = {a: {s: runs[a][s]["test"]["pr_auc"] for s in (1, 2, 3)} for a in ARMS}
    arms = {a: {"mean": st.mean(pr[a].values()), "sd": st.stdev(pr[a].values()), "per_seed": pr[a],
                "lr": sorted({runs[a][s]["lr"] for s in (1, 2, 3)}),
                "class_weight": sorted({str(runs[a][s].get("class_weight")) for s in (1, 2, 3)}),
                "epochs": sorted({runs[a][s]["epochs"] for s in (1, 2, 3)}),
                "best_epoch": {s: runs[a][s].get("best_epoch") for s in (1, 2, 3)}} for a in ARMS}
    deltas = {}
    for name, lo, hi in (("published_to_random", "published", "random_control"),
                         ("random_to_family", "random_control", "family_disjoint")):
        d_s = {s: pr[hi][s] - pr[lo][s] for s in (1, 2, 3)}
        deltas[name] = {**t_ci(list(d_s.values())), "per_seed": d_s}
    return {"arms": arms, "deltas": deltas}


def sweep(d):
    out = {}
    for lr in ("1e-5", "2e-5", "3e-5", "5e-5"):
        r = load(f"{d}/lr{lr}/codebert-base_published_seed1.json")
        out[lr] = {"valid_best": max(h["valid"]["pr_auc"] for h in r["history"]),
                   "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
                   "max_epochs": r["max_epochs"], "n_train": r["n_train"], "test_pr_auc": r["test"]["pr_auc"]}
    return out


def audit(name):
    r = load(f"{name}_audit_report.json")
    t = r["thresholds"][0]
    keep = ("theta", "shingle_k", "perms", "bands", "exhaustive_bucket", "seed", "knn_probe")
    return {"config": {k: r["config"][k] for k in keep},
            "n_functions": t["n_functions"], "n_representatives": r["n_representatives"],
            "leaked_test_pct": t["leaked_test_pct"],
            "random_control_leaked_pct": t["random_control_split"]["leaked_test_pct"],
            "buckets_exhaustive": t["buckets_exhaustive"],
            "buckets_pivot_approximated": t["buckets_pivot_approximated"],
            "lsh_scurve_midpoint": t["lsh_scurve_midpoint"],
            "knn": t["knn_retrieval_baseline"]}


def main() -> int:
    res = {
        "devign_stage_a2_codebert": block("rq2_devign_final", "codebert-base"),
        "devign_stage_a2_graphcodebert": block("rq2_devign_final", "graphcodebert-base"),
        "devign_gate1_codebert": block("rq2_devign", "codebert-base"),
        "devign_gate1_graphcodebert": block("rq2_devign", "graphcodebert-base"),
        "diversevul_3epoch_codebert": block("rq2_diversevul", "codebert-base"),
        "sweep_devign": sweep("sweep_devign"),
        "sweep_diversevul": sweep("sweep_diversevul"),
        "lr_selected": {c: open(os.path.join(BK, f"lr_{c}.txt")).read().strip() for c in ("devign", "diversevul")},
        "audit_devign": audit("devign"),
        "audit_diversevul": audit("diversevul"),
    }
    # family leak recounted per test id from the leak-category files (rq2.tex inference
    # sentence: published vs random control)
    leak = {}
    for arm in ("published", "random_control"):
        cats = load(f"family_label/diversevul_theta0.7_leakcat_{arm}.json")
        leak[arm] = {"n_test": len(cats), "leaked_pct": 100.0 * sum(c != "clean" for c in cats.values()) / len(cats)}
    res["diversevul_leakcat_recount"] = leak
    A2, A2g = res["devign_stage_a2_codebert"], res["devign_stage_a2_graphcodebert"]
    E3 = res["diversevul_3epoch_codebert"]
    checks = [  # (name, recomputed, the project log value, decimals, the project log section)
        ("Devign A2 CodeBERT published mean", A2["arms"]["published"]["mean"], 0.7283, 4, "Stage A2"),
        ("Devign A2 CodeBERT published sd", A2["arms"]["published"]["sd"], 0.0112, 4, "Stage A2"),
        ("Devign A2 CodeBERT random mean", A2["arms"]["random_control"]["mean"], 0.7286, 4, "Stage A2"),
        ("Devign A2 CodeBERT random sd", A2["arms"]["random_control"]["sd"], 0.0019, 4, "Stage A2"),
        ("Devign A2 CodeBERT family mean", A2["arms"]["family_disjoint"]["mean"], 0.7252, 4, "Stage A2"),
        ("Devign A2 CodeBERT family sd", A2["arms"]["family_disjoint"]["sd"], 0.0014, 4, "Stage A2"),
        ("Devign A2 CodeBERT pub->rand", A2["deltas"]["published_to_random"]["mean"], 0.0003, 4, "Stage A2"),
        ("Devign A2 CodeBERT rand->fam", A2["deltas"]["random_to_family"]["mean"], -0.0034, 4, "Stage A2"),
        ("Devign A2 GraphCodeBERT published mean", A2g["arms"]["published"]["mean"], 0.7401, 4, "Stage A2"),
        ("Devign A2 GraphCodeBERT published sd", A2g["arms"]["published"]["sd"], 0.0044, 4, "Stage A2"),
        ("Devign A2 GraphCodeBERT random mean", A2g["arms"]["random_control"]["mean"], 0.7458, 4, "Stage A2"),
        ("Devign A2 GraphCodeBERT random sd", A2g["arms"]["random_control"]["sd"], 0.0025, 4, "Stage A2"),
        ("Devign A2 GraphCodeBERT family mean", A2g["arms"]["family_disjoint"]["mean"], 0.7062, 4, "Stage A2"),
        ("Devign A2 GraphCodeBERT family sd", A2g["arms"]["family_disjoint"]["sd"], 0.0583, 4, "Stage A2"),
        ("Devign A2 GraphCodeBERT pub->rand", A2g["deltas"]["published_to_random"]["mean"], 0.0057, 4, "Stage A2"),
        ("Devign A2 GraphCodeBERT rand->fam", A2g["deltas"]["random_to_family"]["mean"], -0.0396, 4, "Stage A2"),
        ("Devign A2 GraphCodeBERT family s2", A2g["arms"]["family_disjoint"]["per_seed"][2], 0.6391, 4, "Stage A2"),
        # added 24 Oct 2026 for the GraphCodeBERT sentence in rq2.tex (review item 3.5)
        ("Devign A2 GraphCodeBERT rand->fam CI lo", A2g["deltas"]["random_to_family"]["lo"], -0.1784, 4, "Stage A2"),
        ("Devign A2 GraphCodeBERT rand->fam CI hi", A2g["deltas"]["random_to_family"]["hi"], 0.0993, 4, "Stage A2"),
        ("Devign A2 GraphCodeBERT family s2 best epoch", A2g["arms"]["family_disjoint"]["best_epoch"][2], 1, 0,
         "Stage A2"),
        ("Devign A2 GraphCodeBERT family s1", A2g["arms"]["family_disjoint"]["per_seed"][1], 0.736, 4, "Stage A2"),
        ("Devign A2 GraphCodeBERT family s3", A2g["arms"]["family_disjoint"]["per_seed"][3], 0.7436, 4, "Stage A2"),
        # "validation PR-AUC peaked in epoch 1 and then collapsed": epoch-2 value below epoch 1 by > 0.1
        ("Devign A2 GraphCodeBERT family s2 valid ep1 - ep2 > 0.1",
         int(load("rq2_devign_final/graphcodebert-base_family_disjoint_seed2.json")["history"][0]["valid"]["pr_auc"]
             - load("rq2_devign_final/graphcodebert-base_family_disjoint_seed2.json")["history"][1]["valid"]["pr_auc"]
             > 0.1), 1, 0, "Stage A2 (valid 0.6401 -> 0.5008)"),
        ("DiverseVul 3ep published mean", E3["arms"]["published"]["mean"], 0.2563, 4, "Old DiverseVul 3-epoch"),
        ("DiverseVul 3ep published sd", E3["arms"]["published"]["sd"], 0.0074, 4, "Old DiverseVul 3-epoch"),
        ("DiverseVul 3ep random mean", E3["arms"]["random_control"]["mean"], 0.2158, 4, "Old DiverseVul 3-epoch"),
        ("DiverseVul 3ep random sd", E3["arms"]["random_control"]["sd"], 0.0700, 4, "Old DiverseVul 3-epoch"),
        ("DiverseVul 3ep family mean", E3["arms"]["family_disjoint"]["mean"], 0.1824, 4, "Old DiverseVul 3-epoch"),
        ("DiverseVul 3ep family sd", E3["arms"]["family_disjoint"]["sd"], 0.0465, 4, "Old DiverseVul 3-epoch"),
        ("DiverseVul 3ep family - published", E3["arms"]["family_disjoint"]["mean"] - E3["arms"]["published"]["mean"],
         -0.074, 3, "Old DiverseVul 3-epoch"),
        ("sweep Devign 5e-5 valid", res["sweep_devign"]["5e-5"]["valid_best"], 0.7194, 4, "Why Stage A existed"),
        ("sweep DiverseVul 1e-5 valid", res["sweep_diversevul"]["1e-5"]["valid_best"], 0.2131, 4, "Why Stage A existed"),
        ("Devign leaked test %", res["audit_devign"]["leaked_test_pct"], 15.34, 2, "RQ1 numbers"),
        ("DiverseVul leaked test %", res["audit_diversevul"]["leaked_test_pct"], 29.61, 2, "RQ1 numbers"),
        ("Devign random-control leaked %", res["audit_devign"]["random_control_leaked_pct"], 15.01, 2, "RQ1 numbers"),
        ("DiverseVul random-control leaked %", res["audit_diversevul"]["random_control_leaked_pct"], 29.39, 2, "RQ1 numbers"),
        ("Devign 1-NN PR-AUC", res["audit_devign"]["knn"]["pr_auc"], 0.5234, 4, "Where we are"),
        ("DiverseVul 1-NN PR-AUC", res["audit_diversevul"]["knn"]["pr_auc"], 0.1553, 4, "Where we are"),
        ("DiverseVul published leak, recount", leak["published"]["leaked_pct"], 29.61, 2, "RQ1 numbers"),
        ("DiverseVul random leak, recount", leak["random_control"]["leaked_pct"], 29.39, 2, "RQ1 numbers"),
        ("Devign A2 CodeBERT rand->fam lo", A2["deltas"]["random_to_family"]["lo"], -0.0080, 4, "Stage A2 (CI added 12 Oct)"),
        ("Devign A2 CodeBERT rand->fam hi", A2["deltas"]["random_to_family"]["hi"], 0.0012, 4, "Stage A2 (CI added 12 Oct)"),
    ]
    bad = [f"{n}: recomputed {g:.6f} vs the project log {w} ({s})" for n, g, w, d, s in checks
           if abs(g - w) > 0.5 * 10 ** (-d) + 1e-9]
    # selections must be the argmax of validation PR-AUC
    for c in ("devign", "diversevul"):
        sw = res[f"sweep_{c}"]
        best = max(sw, key=lambda k: sw[k]["valid_best"])
        if float(best) != float(res["lr_selected"][c]):
            bad.append(f"{c}: lr file {res['lr_selected'][c]} is not the valid argmax {best}")
    print(f"checked {len(checks)} values + 2 LR selections against the project log")
    if bad:
        print("MISMATCH -- not writing output:\n  " + "\n  ".join(bad))
        return 1
    print("all match")
    res["checks"] = [{"name": n, "recomputed": g, "reference": w, "decimals": d, "section": s}
                     for n, g, w, d, s in checks]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "backup_numbers.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("wrote analysis/scripts/out/backup_numbers.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
