#!/usr/bin/env python3
"""
analyze_phase4b.py -- PRE-REGISTERED analysis of the H4b arms (RESEARCH_PLAN §13.8, written
21 Oct 2026 BEFORE any H4 or H4b pre-check or run).

H4b decision rule (quoted from §13.8):
  Contrast Δ(commit_sizectrl → commit_decon), PR-AUC primary, paired by seeds 1–3, 95% t-CI,
  df = 2:
  * CI entirely below 0 → direct high-recall Type-3 clones in training inflate PR-AUC beyond
    commit sharing.
  * CI includes 0 → no evidence of inflation by direct high-recall clones beyond commit sharing.
  * CI entirely above 0 → reversed: removing the direct clones raises PR-AUC.
  * Secondary: headroom over the length-only floor, MCC, ROC-AUC, same intervals.
  * Any |Δ| ≤ 0.01 is called practically negligible regardless of CI.
  * Not changed after results. A result outside these cases is reported as is.
The pre-check verdict (<disk>/h4b_precheck.json) and the selected design (<disk>/h4_design.json)
are printed with the result.

Inputs: commit_sizectrl and commit_decon seeds 1–3 (<disk>/rq2_diversevul_phase4b); both arms
share the commit_disjoint test set. Writes <phase4b>/phase4b_analysis.{json,md}.
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
    """The §13.8 rule. Returns (case, verdict text)."""
    v = verdict_zero(ci_pr_auc)
    text = ("H4b SUPPORTED: direct high-recall Type-3 clones in training inflate PR-AUC beyond commit sharing"
            if v == "below 0" else
            "H4b NOT SUPPORTED: no evidence of inflation by direct high-recall clones beyond commit sharing"
            if v == "includes 0" else
            "H4b REVERSED: removing the direct clones raises PR-AUC"
            if v == "above 0" else
            f"Δ(sizectrl→decon) CI {v} -- outside the pre-registered cases, report as is")
    m = (ci_pr_auc or {}).get("mean")
    if m is not None and abs(m) <= NEGLIGIBLE:
        text += f"; |Δ| = {abs(m):.4f} ≤ {NEGLIGIBLE}, so practically negligible regardless of CI"
    return v, text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disk", default=os.environ.get("CFA_DATA_DIR", "data"))
    args = ap.parse_args()
    out = os.path.join(args.disk, "rq2_diversevul_phase4b")
    runs, missing = {}, []
    for arm in ("commit_sizectrl", "commit_decon"):
        for s in SEEDS:
            res = load_json(os.path.join(out, f"{MODEL}_{arm}_seed{s}.json"))
            if res is None:
                missing.append(f"{arm} seed{s}")
                continue
            runs.setdefault(arm, {})[s] = run_metrics(res)
    ctl, dec = runs.get("commit_sizectrl", {}), runs.get("commit_decon", {})
    D = {k: paired_delta(ctl, dec, SEEDS, k) for k in KEYS}
    case, verdict = decide(D["pr_auc"])
    pre = load_json(os.path.join(args.disk, "h4b_precheck.json")) or {}
    des = load_json(os.path.join(args.disk, "h4_design.json")) or {}
    rep = {"input": out, "missing_runs": missing, "precheck_verdict": pre.get("verdict", "missing"),
           "selected_design": des.get("design", "missing"),
           "per_arm": {a: {f"s{s}": m for s, m in sorted(r.items())} for a, r in runs.items()},
           "deltas_sizectrl_to_decon": D, "verdict_pr_auc": case, "verdict": verdict}
    os.makedirs(out, exist_ok=True)
    json.dump(rep, open(os.path.join(out, "phase4b_analysis.json"), "w", encoding="utf-8"), indent=2)
    L = ["# Phase 4b -- commit_decon vs commit_sizectrl (pre-registered H4b, RESEARCH_PLAN §13.8)", "",
         f"Pre-check: **{rep['precheck_verdict']}**; selected design: **{rep['selected_design']}**", ""]
    if missing:
        L += [f"**Missing runs:** {', '.join(missing)} (partial: n < 3)", ""]
    L += ["| arm | seed | PR-AUC | headroom | MCC | ROC-AUC |", "|---|---|---|---|---|---|"]
    for a, r in runs.items():
        for s, m in sorted(r.items()):
            L.append(f"| {a} | {s} | {fmt(m['pr_auc'])} | {fmt(m['headroom'])} | "
                     f"{fmt(m['mcc'])} | {fmt(m['roc_auc'])} |")
    L += ["", "| paired Δ sizectrl → decon (seeds 1–3) | " + " | ".join(KEYS) + " |",
          "|---|" + "---|" * len(KEYS),
          "| Δ [95% t-CI] | " + " | ".join(fmt_ci(D[k]) for k in KEYS) + " |", "", f"**{verdict}.**"]
    open(os.path.join(out, "phase4b_analysis.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
