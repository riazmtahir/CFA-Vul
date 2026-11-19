#!/usr/bin/env python3
"""
commit_leak_audit.py -- CPU only, reads ids/labels/commits, no code.

Clone families (θ-Jaccard) are not the only route by which a test function can
have a near-twin in train. A vulnerability-fixing commit yields a vulnerable
function (target 1) and its patched version (target 0) that differ by the fix
itself; on PrimeVul's _paired files only ~47% of such pairs fall in the same
θ=0.7 family (run_primevul_audit.sh step 4). A family-disjoint split therefore
does NOT guarantee that a test function's own patch/pre-patch twin is absent
from train.

For every arm, per test function:
  same_commit_in_train        a train function comes from the same commit
  opposite_same_commit        ... with the opposite label (ANY function of the commit;
                              on DiverseVul most commits also contribute unchanged
                              non-vulnerable functions, so this is an upper bound)
  own_twin                    same commit AND same function name, opposite label:
                              the function's own patched / pre-patch version
  ... outside_family          that train function is NOT in the test function's
                              family, i.e. leakage a family-disjoint split cannot see
Reported for all test functions and for test positives separately.

USAGE
  python commit_leak_audit.py --data CORPUS.jsonl --famfile FAM_theta0.7.json \
      --id-field idx --splitfile random_control=RC.json --splitfile family_disjoint=FD.json \
      --out OUT_PREFIX
The published arm comes from the JSONL `split` field (skip with --no-published).
Writes OUT_PREFIX.json and OUT_PREFIX.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict


# name of a C/C++ function = the identifier right before the first "(" of its text
NAME_RE = re.compile(r"([A-Za-z_~][A-Za-z0-9_:~]*)\s*\(")


def func_name(code):
    m = NAME_RE.search(code)
    return m.group(1) if m else ""


def load(path, id_field):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(o.get("func"), str) or not o["func"].strip():
                continue   # same filter as the other audit tools
            sp = str(o.get("split", "all"))
            rows.append({"id": str(o.get(id_field, n)),
                         "label": 1 if o.get("target") == 1 else 0,
                         "commit": str(o.get("commit_id") or ""),
                         "name": func_name(o["func"]),
                         "split": "valid" if sp in ("valid", "validation") else sp})
    return rows


def pct(a, b):
    return round(100.0 * a / b, 2) if b else None


def arm_stats(rows, split_of, fam):
    # commit -> labels present in train, and train families per (commit, label)
    tr_lab = defaultdict(set)
    tr_fam = defaultdict(set)
    tr_twin = defaultdict(set)    # (commit, name, label) -> families of those train fns
    for r, s in zip(rows, split_of):
        if s == "train" and r["commit"]:
            tr_lab[r["commit"]].add(r["label"])
            tr_fam[(r["commit"], r["label"])].add(fam.get(r["id"]))
            if r["name"]:
                tr_twin[(r["commit"], r["name"], r["label"])].add(fam.get(r["id"]))
    out = {}
    for name, keep in (("all_test", lambda r: True), ("test_positives", lambda r: r["label"] == 1)):
        n = same = opp = opp_out = no_commit = own = own_out = 0
        for r, s in zip(rows, split_of):
            if s != "test" or not keep(r):
                continue
            n += 1
            c = r["commit"]
            if not c:
                no_commit += 1
                continue
            if c in tr_lab:
                same += 1
            other = 1 - r["label"]
            if other in tr_lab.get(c, ()):
                opp += 1
                if fam.get(r["id"]) not in tr_fam[(c, other)]:
                    opp_out += 1
            tw = tr_twin.get((c, r["name"], other)) if r["name"] else None
            if tw:
                own += 1
                if fam.get(r["id"]) not in tw:
                    own_out += 1
        out[name] = {"n": n, "no_commit_id": no_commit,
                     "same_commit_in_train": same, "same_commit_pct": pct(same, n),
                     "opposite_same_commit": opp, "opposite_same_commit_pct": pct(opp, n),
                     "opposite_outside_family": opp_out,
                     "opposite_outside_family_pct": pct(opp_out, n),
                     "own_twin_in_train": own, "own_twin_pct": pct(own, n),
                     "own_twin_outside_family": own_out,
                     "own_twin_outside_family_pct": pct(own_out, n)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--famfile", required=True)
    ap.add_argument("--id-field", default="idx")
    ap.add_argument("--splitfile", action="append", default=[], metavar="ARM=PATH")
    ap.add_argument("--no-published", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load(args.data, args.id_field)
    fam = {str(k): v for k, v in json.load(open(args.famfile)).items()}
    missing = sum(1 for r in rows if r["id"] not in fam)
    if missing:
        print(f"  ! {missing} functions without a family id (treated as singletons)",
              file=sys.stderr)
    arms = []
    if not args.no_published:
        arms.append(("published", [r["split"] for r in rows]))
    for spec in args.splitfile:
        if "=" not in spec:
            print(f"--splitfile wants ARM=PATH, got {spec!r}", file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        a = json.load(open(path))
        arms.append((name, [{"validation": "valid"}.get(a.get(r["id"]), a.get(r["id"]))
                            for r in rows]))

    res = {"data": args.data, "famfile": args.famfile,
           "functions": len(rows),
           "functions_without_commit_id": sum(1 for r in rows if not r["commit"]),
           "arms": {name: arm_stats(rows, sp, fam) for name, sp in arms}}
    with open(args.out + ".json", "w") as fh:
        json.dump(res, fh, indent=2)
    with open(args.out + ".md", "w") as fh:
        fh.write(f"# Commit-level twin leakage -- `{args.data.split('/')[-1]}`\n\n")
        fh.write("opp. label = a train function from the same commit with the opposite "
                 "label (any function of that commit). own twin = same commit AND same "
                 "function name, opposite label: the function's own patched / pre-patch "
                 "version. outside family = not in the test function's θ family, so "
                 "family-disjoint splitting cannot remove it.\n\n")
        fh.write("| arm | test subset | n | same commit in train % | opp. label same commit % | "
                 "own twin in train % | own twin outside family % |\n"
                 "|---|---|---|---|---|---|---|\n")
        for name, st in res["arms"].items():
            for sub, s in st.items():
                fh.write(f"| {name} | {sub} | {s['n']:,} | {s['same_commit_pct']} | "
                         f"{s['opposite_same_commit_pct']} | {s['own_twin_pct']} | "
                         f"{s['own_twin_outside_family_pct']} |\n")
    print(open(args.out + ".md").read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
