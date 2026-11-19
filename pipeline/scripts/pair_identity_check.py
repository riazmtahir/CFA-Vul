#!/usr/bin/env python3
"""
pair_identity_check.py -- how many DiverseVul vulnerable/patched pairs are textually
identical (action A4, 18 Oct 2026). DESCRIPTIVE; changes no pre-registered result.

Pairs are those of paired_eval.py (same commit_id + function name, opposite labels,
clean 1:1 groups only). Two scopes:
  corpus   every clean pair in diversevul.jsonl, wherever its members are (the scope
           of Li et al. 2026, Table 1 p3: 3.47% self-identical after whitespace removal)
  <arm>    the test pairs of each split arm used by paired_eval.py
Levels: raw (same string), ws (same after removing all whitespace), norm (same
normalize_tokens stream of clone_family_audit.py: comments stripped, identifiers
renamed in order of use, literals collapsed), any (any of the three).

Then re-runs the paired evaluation without the identical pairs:
  paired_eval.py --exclude-identical any  ->  <phase2>/paired_eval_noidentical.{json,md}
(the default paired_eval.{json,md} is not touched). --no-rerun skips that step.

Writes <disk>/rq2_diversevul_phase2/pair_identity_check.{json,md}.
Usage: python scripts/pair_identity_check.py [--disk $CFA_DATA_DIR] [--no-rerun]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

VD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VD)
from paired_eval import LEVELS, identical_levels, load_texts, test_pairs  # noqa: E402
from twin_effect_exploratory import load_corpus  # noqa: E402

ARMS = [("random_control", "diversevul_audit_splitfile_randomctrl_theta0.7.json"),
        ("family_disjoint", "diversevul_audit_splitfile_theta0.7.json"),
        ("project_disjoint", "diversevul_audit_splitfile_projectdisjoint_seed42.json"),
        ("commit_disjoint", "diversevul_audit_splitfile_commitdisjoint_seed42.json")]


def count(pairs, texts):
    c = {k: 0 for k in LEVELS + ("any",)}
    for v, p in pairs:
        lv = identical_levels(texts[v], texts[p])
        for k in LEVELS:
            c[k] += lv[k]
        c["any"] += any(lv.values())
    n = len(pairs)
    return {"pairs": n, "identical": c,
            "identical_pct": {k: round(100 * x / n, 2) if n else None for k, x in c.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disk", default=os.environ.get("CFA_DATA_DIR", "data"))
    ap.add_argument("--no-rerun", action="store_true", help="counts only; do not run paired_eval.py")
    args = ap.parse_args()
    D = args.disk
    corpus = os.path.join(D, "data/diversevul/diversevul.jsonl")
    if not os.path.exists(corpus):
        print(f"!!! missing input: {corpus}", file=sys.stderr)
        return 2
    info = load_corpus(corpus, "idx")
    scopes = {"corpus": test_pairs(info, {i: "test" for i in info})}
    for arm, sf in ARMS:
        path = os.path.join(D, sf)
        if not os.path.exists(path):
            print(f"  (no split file for {arm}: {path})")
            continue
        scopes[arm] = test_pairs(info, {str(k): v for k, v in json.load(open(path)).items()})
    need = {x for pairs, _ in scopes.values() for pr in pairs for x in pr}
    texts = load_texts(corpus, need)
    rep = {"note": "DESCRIPTIVE (action A4); pairs as in paired_eval.py", "scopes": {}}
    for name, (pairs, amb) in scopes.items():
        rep["scopes"][name] = {**count(pairs, texts), "ambiguous_groups": amb}

    L = ["# Identical vulnerable/patched pairs on DiverseVul (descriptive, A4)", "",
         "Pair = same commit + function name, opposite labels, clean 1:1 group. `corpus` = all such "
         "pairs in the corpus; arm rows = test pairs of that split (paired_eval.py). raw = same string; "
         "ws = same after removing whitespace (Li et al. 2026); norm = same `normalize_tokens` stream; "
         "any = any of the three.", "",
         "| scope | pairs | raw | ws | norm | any |", "|---|---|---|---|---|---|"]
    for name, r in rep["scopes"].items():
        L.append(f"| {name} | {r['pairs']:,} | " + " | ".join(
            f"{r['identical'][k]:,} ({r['identical_pct'][k]}%)" for k in LEVELS + ("any",)) + " |")
    out = os.path.join(D, "rq2_diversevul_phase2", "pair_identity_check")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rep, open(out + ".json", "w", encoding="utf-8"), indent=2)
    open(out + ".md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))

    if not args.no_rerun:
        r = subprocess.run([sys.executable, os.path.join(VD, "paired_eval.py"), "--disk", D,
                            "--exclude-identical", "any"])
        if r.returncode:
            print("!!! paired_eval.py --exclude-identical failed", file=sys.stderr)
            return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
