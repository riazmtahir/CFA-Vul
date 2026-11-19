#!/usr/bin/env python3
"""effect_sizes.py -- relative change and paired d_z for the pre-registered RQ2 contrasts (action A6).

Relative change = mean(delta) / mean baseline PR-AUC, the baseline being the earlier arm of the
contrast over the same seeds (e.g. random -> commit: delta / mean random-control PR-AUC, seeds
1-3). This is the effect size reported in rq2.tex, as "(X% relative)" where each contrast's delta
is first given; the printed values are checked here (TEXT_PCT).

d_z = mean(delta) / sd(delta), delta = per-seed difference (later arm minus earlier arm),
sd with ddof = 1. Equivalently d_z = t / sqrt(n). Replication package only (not in Table 2).
With n = 3 seeds d_z is imprecise (its sd is a 2-df estimate), and it is large whenever the
seeds agree, whatever the size of the delta in PR-AUC units.

Both are secondary statistics: the pre-registered decision rules (analyze_phase2.py,
analyze_phase3.py) use the t-interval, and no verdict depends on them.

Reads only results/rq2_diversevul_{final,phase2,phase3}/codebert-base_<arm>_seed<s>.json
(the same per-seed files and the same arm/seed map as rq2_numbers.py), then checks that the
per-seed deltas, their means and the baseline means equal analysis/scripts/out/rq2_numbers.json,
and that the relative changes equal the values printed in rq2.tex. Any mismatch -> exit 1,
nothing written.

Output: analysis/scripts/out/effect_sizes.{json,md}.
Usage (repository root): python analysis/scripts/rq2_numbers.py && python analysis/scripts/effect_sizes.py
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

DIRS = {
    "random_control": {**{s: "rq2_diversevul_final" for s in (1, 2, 3)},
                       **{s: "rq2_diversevul_phase2" for s in (4, 5)}},
    "family_disjoint": {**{s: "rq2_diversevul_final" for s in (1, 2, 3)},
                        **{s: "rq2_diversevul_phase2" for s in (4, 5)}},
    "project_disjoint": {s: "rq2_diversevul_phase2" for s in (1, 2, 3)},
    "commit_disjoint": {s: "rq2_diversevul_phase3" for s in (1, 2, 3)},
}
# name -> (from arm, to arm, seeds, pre-registered analysis)
CONTRASTS = {
    "random_to_family_n5": ("random_control", "family_disjoint", [1, 2, 3, 4, 5], "main delta"),
    "random_to_project": ("random_control", "project_disjoint", [1, 2, 3], "H2"),
    "random_to_commit": ("random_control", "commit_disjoint", [1, 2, 3], "H3"),
    "commit_to_project": ("commit_disjoint", "project_disjoint", [1, 2, 3], "H3, reported"),
}
KEYS = ("pr_auc", "headroom", "mcc", "roc_auc")
# rq2_numbers.json per_arm entry that holds the baseline mean for (arm, number of seeds)
BASE_KEY = {("random_control", 5): "random_control", ("random_control", 3): "random_control_s1_3",
            ("commit_disjoint", 3): "commit_disjoint"}
# "(X% relative)" as printed in rq2.tex next to each contrast's first delta (1 decimal)
TEXT_PCT = {"random_to_family_n5": 0.7, "random_to_project": -42.0, "random_to_commit": -32.1,
            "commit_to_project": -14.6}


def metrics(arm, seed):
    path = os.path.join(RES, DIRS[arm][seed], f"codebert-base_{arm}_seed{seed}.json")
    r = json.load(open(path, encoding="utf-8"))
    t = r["test"]
    return {"pr_auc": t["pr_auc"], "roc_auc": t["roc_auc"], "mcc": t["mcc"],
            "headroom": t["pr_auc"] - r["length_only_floor"]["pr_auc"]}


def d_z(xs):
    m, sd = st.mean(xs), st.stdev(xs)
    return {"n": len(xs), "mean": m, "sd": sd, "d_z": m / sd if sd > 0 else math.nan,
            "t": m / (sd / math.sqrt(len(xs))) if sd > 0 else math.nan}


def main() -> int:
    runs = {arm: {s: metrics(arm, s) for s in seeds} for arm, seeds in DIRS.items()}
    out, fails = {}, []
    num = json.load(open(os.path.join(OUT, "rq2_numbers.json"), encoding="utf-8"))
    ref, pa = num["deltas"], num["per_arm"]
    for name, (a, b, seeds, hyp) in CONTRASTS.items():
        out[name] = {"from": a, "to": b, "seeds": seeds, "analysis": hyp}
        for k in KEYS:
            d = [runs[b][s][k] - runs[a][s][k] for s in seeds]
            out[name][k] = {**d_z(d), "per_seed": dict(zip(map(str, seeds), d))}
            r = ref[name][k]
            if abs(r["mean"] - out[name][k]["mean"]) > 1e-12 or abs(r["sd"] - out[name][k]["sd"]) > 1e-12:
                fails.append(f"{name} {k}: mean/sd differ from rq2_numbers.json")
            for s in seeds:
                if abs(r["per_seed"][str(s)] - out[name][k]["per_seed"][str(s)]) > 1e-12:
                    fails.append(f"{name} {k} seed {s}: per-seed delta differs")
        base = st.mean(runs[a][s]["pr_auc"] for s in seeds)
        if abs(base - pa[BASE_KEY[(a, len(seeds))]]["pr_auc"]["mean"]) > 1e-12:
            fails.append(f"{name}: baseline mean differs from rq2_numbers.json per_arm")
        rel = out[name]["pr_auc"]["mean"] / base
        out[name]["relative_change_pr_auc"] = {"baseline_arm": a, "baseline_mean_pr_auc": base,
                                               "relative": rel, "relative_pct_1dp": round(100 * rel, 1)}
        if round(100 * rel, 1) != TEXT_PCT[name]:
            fails.append(f"{name}: relative change {100 * rel:.3f}% != {TEXT_PCT[name]}% printed in rq2.tex")
    if fails:
        print("\n".join(fails), file=sys.stderr)
        return 1
    os.makedirs(OUT, exist_ok=True)
    note = ("relative = mean paired delta PR-AUC / mean PR-AUC of the earlier (baseline) arm over the same "
            "seeds; reported in rq2.tex. d_z = mean/sd of per-seed paired deltas (sd ddof=1) = t/sqrt(n); "
            "replication package only. Both secondary; verdicts use the pre-registered t-intervals. "
            "d_z is imprecise at n=3 (sd from 2 df).")
    json.dump({"note": note, "contrasts": out}, open(os.path.join(OUT, "effect_sizes.json"), "w",
              encoding="utf-8"), indent=2)
    with open(os.path.join(OUT, "effect_sizes.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Paired effect sizes (secondary, action A6)\n\n" + note + "\n\n")
        fh.write("| contrast | analysis | n | mean Δ PR-AUC | baseline PR-AUC | relative | "
                 + " | ".join(f"d_z {k}" for k in KEYS) + " | sd Δ PR-AUC |\n"
                 + "|---|---|---|---|---|---|" + "---|" * (len(KEYS) + 1) + "\n")
        for name, c in out.items():
            rc = c["relative_change_pr_auc"]
            fh.write(f"| {name} | {c['analysis']} | {c['pr_auc']['n']} | {c['pr_auc']['mean']:+.4f} | "
                     f"{rc['baseline_mean_pr_auc']:.4f} | {100 * rc['relative']:+.1f}% | "
                     + " | ".join(f"{c[k]['d_z']:+.2f}" for k in KEYS) + f" | {c['pr_auc']['sd']:.4f} |\n")
    for name, c in out.items():
        rc = c["relative_change_pr_auc"]
        print(f"{name:22s} n={c['pr_auc']['n']}  relative {100 * rc['relative']:+.2f}% "
              f"(baseline {rc['baseline_mean_pr_auc']:.4f})  d_z PR-AUC {c['pr_auc']['d_z']:+.2f}")
    print("checks vs rq2_numbers.json and rq2.tex values passed; wrote out/effect_sizes.{json,md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
