#!/usr/bin/env python3
"""rq2_numbers.py -- recompute every RQ2 number from the per-seed result JSONs.

Reads only results/rq2_diversevul_{final,phase2,phase3}/codebert-base_<arm>_seed<s>.json
(test metrics and length-only floor as written by finetune_vd.py), recomputes
per-arm means and the paired deltas with 95% t-intervals (df = n-1, same critical
values as analyze_phase2.T975), and checks every value against the numbers
recorded in the project log. Any mismatch -> exit code 1 and nothing is written.

Output: analysis/scripts/out/rq2_numbers.json (read by rq2_tables.py and rq2_figure.py).

Usage (from the repository root):  python analysis/scripts/rq2_numbers.py
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Same two-sided 95% t critical values as analyze_phase2.py (df = 1..5).
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}

# arm -> {seed: result file}
DIRS = {
    "published": {s: "rq2_diversevul_final" for s in (1, 2, 3)},
    "random_control": {**{s: "rq2_diversevul_final" for s in (1, 2, 3)},
                       **{s: "rq2_diversevul_phase2" for s in (4, 5)}},
    "family_disjoint": {**{s: "rq2_diversevul_final" for s in (1, 2, 3)},
                        **{s: "rq2_diversevul_phase2" for s in (4, 5)}},
    "project_disjoint": {s: "rq2_diversevul_phase2" for s in (1, 2, 3)},
    "commit_disjoint": {s: "rq2_diversevul_phase3" for s in (1, 2, 3)},
}
KEYS = ("pr_auc", "headroom", "mcc", "roc_auc")


def load_run(arm, seed):
    path = os.path.join(RES, DIRS[arm][seed], f"codebert-base_{arm}_seed{seed}.json")
    with open(path, encoding="utf-8") as fh:
        r = json.load(fh)
    t, fl = r["test"], r["length_only_floor"]
    return {"file": os.path.relpath(path, ROOT).replace("\\", "/"),
            "pr_auc": t["pr_auc"], "roc_auc": t["roc_auc"], "mcc": t["mcc"],
            "floor": fl["pr_auc"], "headroom": t["pr_auc"] - fl["pr_auc"],
            "pos_rate": t["positive_rate"], "n_test": t["n"],
            "n_train": r["n_train"], "n_valid": r["n_valid"],
            "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
            "lr": r["lr"], "max_len": r["max_len"], "batch_size": r["batch_size"],
            "class_weight": r["class_weight"], "patience": r["patience"],
            "max_epochs": r["max_epochs"], "selected_on": r["selected_on"]}


def t_ci(xs):
    n = len(xs)
    m = st.mean(xs)
    sd = st.stdev(xs)
    h = T975[n - 1] * sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "lo": m - h, "hi": m + h, "t": m / (sd / math.sqrt(n)),
            "t_crit": T975[n - 1]}


def summary(xs):
    return {"n": len(xs), "mean": st.mean(xs), "sd": st.stdev(xs) if len(xs) > 1 else None}


def main() -> int:
    runs = {arm: {s: load_run(arm, s) for s in seeds} for arm, seeds in DIRS.items()}

    per_arm = {}
    for arm, rs in runs.items():
        seeds = sorted(rs)
        per_arm[arm] = {"seeds": seeds,
                        **{k: summary([rs[s][k] for s in seeds]) for k in ("pr_auc", "floor", "headroom",
                                                                             "mcc", "roc_auc", "pos_rate")},
                        "per_seed_pr_auc": {s: rs[s]["pr_auc"] for s in seeds},
                        "n_test": sorted({rs[s]["n_test"] for s in seeds}),
                        "n_train": sorted({rs[s]["n_train"] for s in seeds}),
                        "n_valid": sorted({rs[s]["n_valid"] for s in seeds}),
                        "best_epoch": {s: rs[s]["best_epoch"] for s in seeds},
                        "files": [rs[s]["file"] for s in seeds]}
    # random_control restricted to seeds 1-3 (the comparison set for H2/H3)
    rc13 = {s: runs["random_control"][s] for s in (1, 2, 3)}
    per_arm["random_control_s1_3"] = {"seeds": [1, 2, 3],
                                      **{k: summary([rc13[s][k] for s in (1, 2, 3)])
                                         for k in ("pr_auc", "floor", "headroom", "mcc", "roc_auc", "pos_rate")}}
    fd13 = {s: runs["family_disjoint"][s] for s in (1, 2, 3)}
    per_arm["family_disjoint_s1_3"] = {"seeds": [1, 2, 3],
                                       **{k: summary([fd13[s][k] for s in (1, 2, 3)])
                                          for k in ("pr_auc", "floor", "headroom", "mcc", "roc_auc", "pos_rate")}}

    def paired(a, b, seeds):
        out = {}
        for k in KEYS:
            d = {s: runs[b][s][k] - runs[a][s][k] for s in seeds}
            out[k] = {**t_ci(list(d.values())), "per_seed": d}
        return out

    contrasts = {
        "random_to_family_n5": ("random_control", "family_disjoint", [1, 2, 3, 4, 5]),
        "random_to_family_n3": ("random_control", "family_disjoint", [1, 2, 3]),
        "published_to_random_n3": ("published", "random_control", [1, 2, 3]),
        "published_to_family_n3": ("published", "family_disjoint", [1, 2, 3]),
        "random_to_project": ("random_control", "project_disjoint", [1, 2, 3]),
        "random_to_commit": ("random_control", "commit_disjoint", [1, 2, 3]),
        "commit_to_project": ("commit_disjoint", "project_disjoint", [1, 2, 3]),
    }
    deltas = {name: {"from": a, "to": b, "seeds": s, **paired(a, b, s)}
              for name, (a, b, s) in contrasts.items()}
    share = deltas["random_to_commit"]["pr_auc"]["mean"] / deltas["random_to_project"]["pr_auc"]["mean"]

    # ---- check against the project log (value, decimals shown in the project log, the project log section)
    D = deltas
    checks = [
        # Stage B per-arm (the project log "Stage B -- DiverseVul final", per-arm table)
        ("published mean PR-AUC", per_arm["published"]["pr_auc"]["mean"], 0.2492, 4, "Stage B"),
        ("published sd PR-AUC", per_arm["published"]["pr_auc"]["sd"], 0.0242, 4, "Stage B"),
        ("random s1-3 mean PR-AUC", per_arm["random_control_s1_3"]["pr_auc"]["mean"], 0.2539, 4, "Stage B"),
        ("random s1-3 sd PR-AUC", per_arm["random_control_s1_3"]["pr_auc"]["sd"], 0.0095, 4, "Stage B"),
        ("family s1-3 mean PR-AUC", per_arm["family_disjoint_s1_3"]["pr_auc"]["mean"], 0.2534, 4, "Stage B"),
        ("family s1-3 sd PR-AUC", per_arm["family_disjoint_s1_3"]["pr_auc"]["sd"], 0.0081, 4, "Stage B"),
        # Stage B deltas
        ("pub->rand mean", D["published_to_random_n3"]["pr_auc"]["mean"], 0.0047, 4, "Stage B"),
        ("pub->rand sd", D["published_to_random_n3"]["pr_auc"]["sd"], 0.0337, 4, "Stage B"),
        ("pub->rand lo", D["published_to_random_n3"]["pr_auc"]["lo"], -0.079, 3, "Stage B"),
        ("pub->rand hi", D["published_to_random_n3"]["pr_auc"]["hi"], 0.088, 3, "Stage B (corrected 8 Oct 2026)"),
        ("rand->fam n3 mean", D["random_to_family_n3"]["pr_auc"]["mean"], -0.0005, 4, "Stage B / Phase 2 B"),
        ("rand->fam n3 sd", D["random_to_family_n3"]["pr_auc"]["sd"], 0.0110, 4, "Stage B"),
        ("rand->fam n3 lo", D["random_to_family_n3"]["pr_auc"]["lo"], -0.0278, 4, "Phase 2 B"),
        ("rand->fam n3 hi", D["random_to_family_n3"]["pr_auc"]["hi"], 0.0268, 4, "Phase 2 B"),
        ("pub->fam mean", D["published_to_family_n3"]["pr_auc"]["mean"], 0.0042, 4, "Stage B"),
        ("pub->fam sd", D["published_to_family_n3"]["pr_auc"]["sd"], 0.0268, 4, "Stage B"),
        ("rand->fam n3 headroom", D["random_to_family_n3"]["headroom"]["mean"], 0.0031, 4, "Stage B"),
        ("rand->fam n3 MCC", D["random_to_family_n3"]["mcc"]["mean"], 0.0058, 4, "Stage B"),
        ("rand->fam n3 ROC", D["random_to_family_n3"]["roc_auc"]["mean"], 0.0021, 4, "Stage B"),
        # Phase 2 A (n=5)
        ("rand->fam n5 PR mean", D["random_to_family_n5"]["pr_auc"]["mean"], 0.0017, 4, "Phase 2 A"),
        ("rand->fam n5 PR sd", D["random_to_family_n5"]["pr_auc"]["sd"], 0.0090, 4, "Phase 2 A"),
        ("rand->fam n5 PR lo", D["random_to_family_n5"]["pr_auc"]["lo"], -0.0095, 4, "Phase 2 A"),
        ("rand->fam n5 PR hi", D["random_to_family_n5"]["pr_auc"]["hi"], 0.0129, 4, "Phase 2 A"),
        ("rand->fam n5 headroom mean", D["random_to_family_n5"]["headroom"]["mean"], 0.0053, 4, "Phase 2 A"),
        ("rand->fam n5 headroom lo", D["random_to_family_n5"]["headroom"]["lo"], -0.0059, 4, "Phase 2 A"),
        ("rand->fam n5 headroom hi", D["random_to_family_n5"]["headroom"]["hi"], 0.0165, 4, "Phase 2 A"),
        ("rand->fam n5 MCC mean", D["random_to_family_n5"]["mcc"]["mean"], 0.0089, 4, "Phase 2 A"),
        ("rand->fam n5 MCC lo", D["random_to_family_n5"]["mcc"]["lo"], -0.0012, 4, "Phase 2 A"),
        ("rand->fam n5 MCC hi", D["random_to_family_n5"]["mcc"]["hi"], 0.0190, 4, "Phase 2 A"),
        ("rand->fam n5 ROC mean", D["random_to_family_n5"]["roc_auc"]["mean"], 0.0047, 4, "Phase 2 A"),
        ("rand->fam n5 ROC lo", D["random_to_family_n5"]["roc_auc"]["lo"], -0.0055, 4, "Phase 2 A"),
        ("rand->fam n5 ROC hi", D["random_to_family_n5"]["roc_auc"]["hi"], 0.0149, 4, "Phase 2 A"),
        # Phase 2 B (H2)
        ("rand->proj PR mean", D["random_to_project"]["pr_auc"]["mean"], -0.1067, 4, "Phase 2 B"),
        ("rand->proj PR sd", D["random_to_project"]["pr_auc"]["sd"], 0.0073, 4, "Phase 2 B"),
        ("rand->proj PR lo", D["random_to_project"]["pr_auc"]["lo"], -0.1248, 4, "Phase 2 B"),
        ("rand->proj PR hi", D["random_to_project"]["pr_auc"]["hi"], -0.0886, 4, "Phase 2 B"),
        ("rand->proj headroom mean", D["random_to_project"]["headroom"]["mean"], -0.1011, 4, "Phase 2 B"),
        ("rand->proj headroom lo", D["random_to_project"]["headroom"]["lo"], -0.1192, 4, "Phase 2 B"),
        ("rand->proj headroom hi", D["random_to_project"]["headroom"]["hi"], -0.0830, 4, "Phase 2 B"),
        ("rand->proj MCC mean", D["random_to_project"]["mcc"]["mean"], -0.0881, 4, "Phase 2 B"),
        ("rand->proj MCC lo", D["random_to_project"]["mcc"]["lo"], -0.1009, 4, "Phase 2 B"),
        ("rand->proj MCC hi", D["random_to_project"]["mcc"]["hi"], -0.0753, 4, "Phase 2 B"),
        ("rand->proj ROC mean", D["random_to_project"]["roc_auc"]["mean"], -0.0744, 4, "Phase 2 B"),
        ("rand->proj ROC lo", D["random_to_project"]["roc_auc"]["lo"], -0.0882, 4, "Phase 2 B"),
        ("rand->proj ROC hi", D["random_to_project"]["roc_auc"]["hi"], -0.0606, 4, "Phase 2 B"),
        # Phase 3 (H3)
        ("commit mean PR-AUC", per_arm["commit_disjoint"]["pr_auc"]["mean"], 0.1724, 4, "Phase 3"),
        ("rand->commit PR mean", D["random_to_commit"]["pr_auc"]["mean"], -0.0815, 4, "Phase 3"),
        ("rand->commit PR lo", D["random_to_commit"]["pr_auc"]["lo"], -0.0970, 4, "Phase 3"),
        ("rand->commit PR hi", D["random_to_commit"]["pr_auc"]["hi"], -0.0660, 4, "Phase 3"),
        ("rand->commit headroom mean", D["random_to_commit"]["headroom"]["mean"], -0.0728, 4, "Phase 3"),
        ("rand->commit headroom lo", D["random_to_commit"]["headroom"]["lo"], -0.0883, 4, "Phase 3"),
        ("rand->commit headroom hi", D["random_to_commit"]["headroom"]["hi"], -0.0573, 4, "Phase 3"),
        ("rand->commit MCC mean", D["random_to_commit"]["mcc"]["mean"], -0.0608, 4, "Phase 3"),
        ("rand->commit MCC lo", D["random_to_commit"]["mcc"]["lo"], -0.0849, 4, "Phase 3"),
        ("rand->commit MCC hi", D["random_to_commit"]["mcc"]["hi"], -0.0368, 4, "Phase 3"),
        ("rand->commit ROC mean", D["random_to_commit"]["roc_auc"]["mean"], -0.0627, 4, "Phase 3"),
        ("rand->commit ROC lo", D["random_to_commit"]["roc_auc"]["lo"], -0.0898, 4, "Phase 3"),
        ("rand->commit ROC hi", D["random_to_commit"]["roc_auc"]["hi"], -0.0356, 4, "Phase 3"),
        ("commit->proj PR mean", D["commit_to_project"]["pr_auc"]["mean"], -0.0252, 4, "Phase 3"),
        ("commit->proj PR lo", D["commit_to_project"]["pr_auc"]["lo"], -0.0479, 4, "Phase 3"),
        ("commit->proj PR hi", D["commit_to_project"]["pr_auc"]["hi"], -0.0024, 4, "Phase 3"),
        ("commit->proj headroom mean", D["commit_to_project"]["headroom"]["mean"], -0.0283, 4, "Phase 3"),
        ("commit->proj MCC mean", D["commit_to_project"]["mcc"]["mean"], -0.0273, 4, "Phase 3"),
        ("commit->proj MCC lo", D["commit_to_project"]["mcc"]["lo"], -0.0576, 4, "Phase 3"),
        ("commit->proj MCC hi", D["commit_to_project"]["mcc"]["hi"], 0.0030, 4, "Phase 3"),
        ("commit->proj ROC mean", D["commit_to_project"]["roc_auc"]["mean"], -0.0117, 4, "Phase 3"),
        ("commit->proj ROC lo", D["commit_to_project"]["roc_auc"]["lo"], -0.0429, 4, "Phase 3"),
        ("commit->proj ROC hi", D["commit_to_project"]["roc_auc"]["hi"], 0.0194, 4, "Phase 3"),
        ("share of project gap", share, 0.76, 2, "Phase 3"),
        # CI half-widths of the family contrast, n = 3 and n = 5 (rq2.tex §5.2, discussion.tex §7.2)
        ("rand->fam n3 half-width", (D["random_to_family_n3"]["pr_auc"]["hi"] - D["random_to_family_n3"]["pr_auc"]["lo"]) / 2,
         0.0273, 4, "rq2.tex / discussion.tex"),
        ("rand->fam n5 half-width", (D["random_to_family_n5"]["pr_auc"]["hi"] - D["random_to_family_n5"]["pr_auc"]["lo"]) / 2,
         0.0112, 4, "rq2.tex / discussion.tex"),
    ]
    # per-seed values listed in the project log
    for s, v in zip((1, 2, 3, 4, 5), (0.0122, -0.0075, -0.0061, 0.0000, 0.0099)):
        checks.append((f"rand->fam PR s{s}", D["random_to_family_n5"]["pr_auc"]["per_seed"][s], v, 4, "Phase 2 A"))
    for s, v in zip((1, 2, 3), (0.1484, 0.1531, 0.1401)):
        checks.append((f"project PR-AUC s{s}", runs["project_disjoint"][s]["pr_auc"], v, 4, "Phase 2 B"))
    for s, v in zip((1, 2, 3), (0.1651, 0.1770, 0.1750)):
        checks.append((f"commit PR-AUC s{s}", runs["commit_disjoint"][s]["pr_auc"], v, 4, "Phase 3"))
    # remaining hand-computed Stage B values in the project log (CIs at 3 decimals, t at 2)
    checks += [
        ("pub->fam lo", D["published_to_family_n3"]["pr_auc"]["lo"], -0.062, 3, "Stage B"),
        ("pub->fam hi", D["published_to_family_n3"]["pr_auc"]["hi"], 0.071, 3, "Stage B"),
        ("rand->fam n3 lo (3dp)", D["random_to_family_n3"]["pr_auc"]["lo"], -0.028, 3, "Stage B"),
        ("rand->fam n3 hi (3dp)", D["random_to_family_n3"]["pr_auc"]["hi"], 0.027, 3, "Stage B"),
        ("pub->rand t", D["published_to_random_n3"]["pr_auc"]["t"], 0.24, 2, "Stage B"),
        ("rand->fam n3 t", D["random_to_family_n3"]["pr_auc"]["t"], -0.07, 2, "Stage B"),
        ("pub->fam t", D["published_to_family_n3"]["pr_auc"]["t"], 0.27, 2, "Stage B"),
        ("rand->fam n3 headroom sd", D["random_to_family_n3"]["headroom"]["sd"], 0.0110, 4, "Stage B"),
        ("rand->fam n3 MCC sd", D["random_to_family_n3"]["mcc"]["sd"], 0.0097, 4, "Stage B"),
        ("rand->fam n3 ROC sd", D["random_to_family_n3"]["roc_auc"]["sd"], 0.0067, 4, "Stage B"),
        ("published mean ROC", per_arm["published"]["roc_auc"]["mean"], 0.8049, 4, "Stage B"),
        ("random s1-3 mean ROC", per_arm["random_control_s1_3"]["roc_auc"]["mean"], 0.8132, 4, "Stage B"),
        ("family s1-3 mean ROC", per_arm["family_disjoint_s1_3"]["roc_auc"]["mean"], 0.8152, 4, "Stage B"),
        ("random s1-3 test pos rate", per_arm["random_control_s1_3"]["pos_rate"]["mean"], 0.0573, 4, "Phase 2 B"),
        ("project test pos rate", per_arm["project_disjoint"]["pos_rate"]["mean"], 0.0570, 4, "Phase 2 B"),
        ("commit test pos rate", per_arm["commit_disjoint"]["pos_rate"]["mean"], 0.0570, 4, "Phase 3"),
    ]
    for s, v in zip((1, 2, 3), (-0.0220, 0.0426, -0.0065)):
        checks.append((f"pub->rand s{s}", D["published_to_random_n3"]["pr_auc"]["per_seed"][s], v, 4, "Stage B"))
    for s, v in zip((1, 2, 3), (-0.0098, 0.0351, -0.0126)):
        checks.append((f"pub->fam s{s}", D["published_to_family_n3"]["pr_auc"]["per_seed"][s], v, 4, "Stage B"))
    for s, v in zip((1, 2, 3), (0.0158, -0.0039, -0.0025)):
        checks.append((f"rand->fam headroom s{s}", D["random_to_family_n3"]["headroom"]["per_seed"][s], v, 4, "Stage B"))
    for s, v in zip((1, 2, 3), (0.0165, -0.0023, 0.0031)):
        checks.append((f"rand->fam MCC s{s}", D["random_to_family_n3"]["mcc"]["per_seed"][s], v, 4, "Stage B"))
    checks.append(("project floor", per_arm["project_disjoint"]["floor"]["mean"], 0.1444, 4, "Phase 2 B"))
    checks.append(("commit floor", per_arm["commit_disjoint"]["floor"]["mean"], 0.1413, 4, "Phase 3"))

    bad = []
    for name, got, want, nd, sec in checks:
        # match = the recomputed value rounds to the the project log value at the project log's precision
        ok = abs(got - want) <= 0.5 * 10 ** (-nd) + 1e-9
        if not ok:
            bad.append(f"{name}: recomputed {got:.6f} -> {round(got, nd)} vs the project log {want} ({sec})")
    print(f"checked {len(checks)} values against the project log")
    if bad:
        print("MISMATCH -- not writing output:")
        print("\n".join("  " + b for b in bad))
        return 1
    print("all match")

    os.makedirs(OUT, exist_ok=True)
    out = {"per_arm": per_arm, "deltas": deltas, "share_of_project_gap": share,
           "checks": [{"name": n, "recomputed": g, "reference": w, "decimals": d, "section": s}
                      for n, g, w, d, s in checks]}
    with open(os.path.join(OUT, "rq2_numbers.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"wrote {os.path.relpath(os.path.join(OUT, 'rq2_numbers.json'), ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
