#!/usr/bin/env python3
"""
select_h4_design.py -- the precedence of RESEARCH_PLAN §13.8 (fixed 21 Oct 2026):
  H4 pre-check PASS                      -> run H4   (run_phase4.sh)
  H4 not PASS and H4b pre-check PASS     -> run H4b  (run_phase4b.sh)
  neither PASS                           -> run neither; report both pre-checks
A missing or unreadable pre-check file counts as not PASS.
Reads <disk>/h4_precheck.json and <disk>/h4b_precheck.json; writes <disk>/h4_design.json.
Usage: python scripts/select_h4_design.py [--disk $CFA_DATA_DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def verdict(path):
    try:
        return json.load(open(path, encoding="utf-8")).get("verdict", "missing")
    except (OSError, ValueError):
        return "missing"


def select(v4, v4b):
    if v4 == "PASS":
        return "H4"
    if v4b == "PASS":
        return "H4b"
    return "none"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disk", default=os.environ.get("CFA_DATA_DIR", "data"))
    a = ap.parse_args()
    v4 = verdict(os.path.join(a.disk, "h4_precheck.json"))
    v4b = verdict(os.path.join(a.disk, "h4b_precheck.json"))
    d = select(v4, v4b)
    rep = {"rule": "RESEARCH_PLAN §13.8 precedence", "h4_precheck": v4, "h4b_precheck": v4b, "design": d,
           "action": {"H4": "run_phase4.sh (after .phase4_approved)",
                      "H4b": "run_phase4b.sh (after .phase4b_approved)",
                      "none": "run neither; report h4_precheck.md and h4b_precheck.md"}[d]}
    json.dump(rep, open(os.path.join(a.disk, "h4_design.json"), "w", encoding="utf-8"), indent=2)
    print(f"H4 pre-check: {v4}   H4b pre-check: {v4b}   -> selected design: {d} ({rep['action']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
