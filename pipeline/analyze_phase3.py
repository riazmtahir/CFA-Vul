#!/usr/bin/env python3
"""
analyze_phase3.py -- PRE-REGISTERED analysis of the commit-disjoint arm
(RESEARCH_PLAN §13.6, written 7 Oct 2026 BEFORE any commit-disjoint run).

Why: on DiverseVul, 99.7% of random-control test functions share a commit with a
train function (commit_leak_audit.py), and project-disjoint splitting removes that
sharing as a side effect (commits live inside one project). The H2 drop
Δ(random→project) = −0.107 PR-AUC could therefore be project shift, same-commit
sharing, or both. commit_disjoint keeps every commit whole but spreads projects,
so it removes same-commit sharing without project shift.

H3, decision rule (fixed before results; paired by seed, t-CI df = n−1, n = 3):
  * Δ(random→commit) t-CI entirely below 0  -> same-commit sharing inflates
    random-split scores. Report share = mean Δ(r→c) / mean Δ(r→p) as the part
    of the project gap it accounts for.
  * Δ(random→commit) t-CI includes 0        -> commit sharing does not explain the
    project gap; H2's project-shift reading stands.
  * Δ(random→commit) t-CI entirely above 0  -> [amended 7 Oct 2026, before any
    commit-disjoint run, after the own-twin audit] opposite-label twins in train
    suppress detection on random splits (the function's own patched version
    teaches "benign"); commit-disjoint is the EASIER split.
  * Δ(commit→project) is reported either way (the gap left after commit sharing).
  * Caveat fixed in advance: commit-disjoint also lowers Type-3 family leakage
    (DiverseVul 29.4% -> 17.1% of test). Since removing ALL family leakage moved
    PR-AUC by +0.0017 [−0.0095, +0.0129] (Phase 2, n=5), a Δ(random→commit) well
    outside that interval is attributed to commit sharing, not to families.
Primary metric PR-AUC; headroom over the length-only floor, MCC and ROC-AUC are
secondary and reported with the same intervals.

Inputs: random_control seeds 1–3 (Stage B dir), project_disjoint seeds 1–3
(Phase 2 dir), commit_disjoint seeds 1–3 (Phase 3 dir). Same protocol for all.
Writes <phase3>/phase3_analysis.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_phase2 import (MODEL, fmt, fmt_ci, load_json, paired_delta,  # noqa: E402
                            run_metrics, verdict_zero)

KEYS = ("pr_auc", "headroom", "mcc", "roc_auc")
SEEDS = [1, 2, 3]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disk", default=os.environ.get("CFA_DATA_DIR", "data"))
    args = ap.parse_args()
    dirs = {"random_control": os.path.join(args.disk, "rq2_diversevul_final"),
            "project_disjoint": os.path.join(args.disk, "rq2_diversevul_phase2"),
            "commit_disjoint": os.path.join(args.disk, "rq2_diversevul_phase3")}

    runs, missing = {}, []
    for arm, d in dirs.items():
        for s in SEEDS:
            res = load_json(os.path.join(d, f"{MODEL}_{arm}_seed{s}.json"))
            if res is None:
                missing.append(f"{arm} seed{s}")
                continue
            runs.setdefault(arm, {})[s] = run_metrics(res)
    rc, pdj, cd = (runs.get(a, {}) for a in dirs)

    D = {name: {k: paired_delta(lo, hi, SEEDS, k) for k in KEYS}
         for name, lo, hi in (("random_to_commit", rc, cd),
                              ("random_to_project", rc, pdj),
                              ("commit_to_project", cd, pdj))}
    v = verdict_zero(D["random_to_commit"]["pr_auc"])
    rp = D["random_to_project"]["pr_auc"].get("mean")
    rcm = D["random_to_commit"]["pr_auc"].get("mean")
    share = (rcm / rp) if (rp and rcm is not None and v == "below 0") else None
    verdict = ("H3 SUPPORTED: same-commit sharing inflates random-split PR-AUC"
               if v == "below 0" else
               "H3 NOT SUPPORTED: commit sharing does not explain the project gap"
               if v == "includes 0" else
               "H3 REVERSED: commit-disjoint is easier -- opposite-label twins in train "
               "suppress detection on random splits"
               if v == "above 0" else
               f"Δ(random→commit) CI {v} -- outside the pre-registered cases, report as is")
    rep = {"inputs": dirs, "missing_runs": missing,
           "per_arm": {a: {f"s{s}": m for s, m in sorted(r.items())} for a, r in runs.items()},
           "deltas": D, "verdict_random_to_commit_pr_auc": v,
           "share_of_project_gap": share, "verdict": verdict}

    out = dirs["commit_disjoint"]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "phase3_analysis.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    L = ["# Phase 3 -- commit-disjoint arm (pre-registered H3)", ""]
    if missing:
        L += [f"**Missing runs:** {', '.join(missing)}", ""]
    L += ["| arm | seed | PR-AUC | headroom | MCC | ROC-AUC |", "|---|---|---|---|---|---|"]
    for a, r in runs.items():
        for s, m in sorted(r.items()):
            L.append(f"| {a} | {s} | {fmt(m['pr_auc'])} | {fmt(m['headroom'])} | "
                     f"{fmt(m['mcc'])} | {fmt(m['roc_auc'])} |")
    L += ["", "| paired Δ (seeds 1–3) | " + " | ".join(KEYS) + " |",
          "|---|" + "---|" * len(KEYS)]
    for name, dk in D.items():
        L.append(f"| {name.replace('_', ' ')} | " + " | ".join(fmt_ci(dk[k]) for k in KEYS) + " |")
    L += ["", f"**{verdict}.**"]
    if share is not None:
        L.append(f"Share of the random→project gap accounted for by commit sharing: {share:.2f}.")
    open(os.path.join(out, "phase3_analysis.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
