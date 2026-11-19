#!/usr/bin/env python3
"""
analyze_phase4.py -- PRE-REGISTERED analysis of the commit_hrfamily_disjoint arm
(RESEARCH_PLAN §13.7, written 21 Oct 2026 BEFORE any H4 run).

H4 decision rule (quoted from §13.7):
  Contrast Δ(commit_disjoint → commit_hrfamily_disjoint), PR-AUC primary, paired by
  seeds 1–3, 95% t-CI, df = 2:
  * CI entirely below 0 → high-recall Type-3 leakage inflates PR-AUC beyond commit sharing.
  * CI includes 0 → no evidence of inflation beyond commit sharing.
  * CI entirely above 0 → reversed: the commit_hrfamily_disjoint split is the easier split.
  * Secondary: headroom over the length-only floor, MCC, ROC-AUC, same intervals.
  * Caveat fixed in advance: the new split changes test composition (length, project mix),
    so headroom is reported next to PR-AUC, and any Δ within ±0.01 is described as
    practically negligible regardless of CI.
  * Not changed after results. A result outside these cases is reported as is.
Feasibility gate: the H4 runs are valid only if precheck_h4.py passed; its verdict is read
from <disk>/h4_precheck.json and printed with the result.

Inputs: commit_disjoint seeds 1–3 (<disk>/rq2_diversevul_phase3), commit_hrfamily_disjoint
seeds 1–3 (<disk>/rq2_diversevul_phase4). Same protocol for both.
Writes <phase4>/phase4_analysis.{json,md}.
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
NEGLIGIBLE = 0.01


def decide(ci_pr_auc):
    """The §13.7 rule. Returns (case, verdict text)."""
    v = verdict_zero(ci_pr_auc)
    text = ("H4 SUPPORTED: high-recall Type-3 leakage inflates PR-AUC beyond commit sharing"
            if v == "below 0" else
            "H4 NOT SUPPORTED: no evidence of inflation beyond commit sharing"
            if v == "includes 0" else
            "H4 REVERSED: the commit_hrfamily_disjoint split is the easier split"
            if v == "above 0" else
            f"Δ(commit→commit_hrfamily) CI {v} -- outside the pre-registered cases, report as is")
    m = (ci_pr_auc or {}).get("mean")
    if m is not None and abs(m) <= NEGLIGIBLE:
        text += f"; the Δ ({m:+.4f}) is within ±{NEGLIGIBLE}, so practically negligible regardless of CI"
    return v, text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disk", default=os.environ.get("CFA_DATA_DIR", "data"))
    args = ap.parse_args()
    dirs = {"commit_disjoint": os.path.join(args.disk, "rq2_diversevul_phase3"),
            "commit_hrfamily_disjoint": os.path.join(args.disk, "rq2_diversevul_phase4")}
    runs, missing = {}, []
    for arm, d in dirs.items():
        for s in SEEDS:
            res = load_json(os.path.join(d, f"{MODEL}_{arm}_seed{s}.json"))
            if res is None:
                missing.append(f"{arm} seed{s}")
                continue
            runs.setdefault(arm, {})[s] = run_metrics(res)
    cd, hf = runs.get("commit_disjoint", {}), runs.get("commit_hrfamily_disjoint", {})
    D = {k: paired_delta(cd, hf, SEEDS, k) for k in KEYS}
    case, verdict = decide(D["pr_auc"])
    pre = load_json(os.path.join(args.disk, "h4_precheck.json")) or {}
    rep = {"inputs": dirs, "missing_runs": missing, "precheck_verdict": pre.get("verdict", "missing"),
           "per_arm": {a: {f"s{s}": m for s, m in sorted(r.items())} for a, r in runs.items()},
           "deltas_commit_to_commit_hrfamily": D, "verdict_pr_auc": case, "verdict": verdict}
    out = dirs["commit_hrfamily_disjoint"]
    os.makedirs(out, exist_ok=True)
    json.dump(rep, open(os.path.join(out, "phase4_analysis.json"), "w", encoding="utf-8"), indent=2)
    L = ["# Phase 4 -- commit_hrfamily_disjoint arm (pre-registered H4, RESEARCH_PLAN §13.7)", "",
         f"Pre-check: **{rep['precheck_verdict']}**", ""]
    if missing:
        L += [f"**Missing runs:** {', '.join(missing)} (partial: n < 3)", ""]
    L += ["| arm | seed | PR-AUC | headroom | MCC | ROC-AUC |", "|---|---|---|---|---|---|"]
    for a, r in runs.items():
        for s, m in sorted(r.items()):
            L.append(f"| {a} | {s} | {fmt(m['pr_auc'])} | {fmt(m['headroom'])} | "
                     f"{fmt(m['mcc'])} | {fmt(m['roc_auc'])} |")
    L += ["", "| paired Δ commit → commit_hrfamily (seeds 1–3) | " + " | ".join(KEYS) + " |",
          "|---|" + "---|" * len(KEYS),
          "| Δ [95% t-CI] | " + " | ".join(fmt_ci(D[k]) for k in KEYS) + " |", "", f"**{verdict}.**"]
    open(os.path.join(out, "phase4_analysis.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
