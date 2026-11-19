#!/usr/bin/env python3
"""
make_commit_family_split.py -- the commit_hrfamily_disjoint arm for pre-registered H4
(RESEARCH_PLAN §13.7, written 21 Oct 2026 before any H4 run). CPU only.

Groups = connected components of the union of two relations on DiverseVul functions:
  (a) same commit_id (functions with an empty commit_id are not linked by (a)), and
  (b) same HIGH-RECALL theta=0.7 clone family (blind identifiers, 3-token shingles;
      audit_highrecall/diversevul_hr_families_theta0.7.json).
Whole groups are assigned to train / valid / test with the SAME two-key greedy and
whole-group refinement as make_project_split.py (imported, not copied), with targets =
the partition sizes and positive counts of the existing commit-disjoint split
(diversevul_audit_splitfile_commitdisjoint_seed42.json), seed 42. So no commit and no
high-recall family spans two partitions.

Because (b) is transitive (union-find), groups can be very large; the report gives the
largest group and the ten largest, and precheck_h4.py refuses the split (exit 3) if the
largest group exceeds 15% of the corpus or the partitions miss their targets.

USAGE (server)
  D=$CFA_DATA_DIR
  python scripts/make_commit_family_split.py --data $D/data/diversevul/diversevul.jsonl \
      --hr-famfile $D/audit_highrecall/diversevul_hr_families_theta0.7.json \
      --ref-splitfile $D/diversevul_audit_splitfile_commitdisjoint_seed42.json \
      --strict-famfile $D/diversevul_audit_families_theta0.7.json \
      --seed 42 --out $D/diversevul_audit_splitfile_commithrfamily_seed42
OUTPUT
  <out>.json         id -> train/valid/test  (finetune_vd.py --splitfile)
  <out>_report.json  sizes, positive rates, group sizes, invariants, residual leaks
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

VD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VD)
from make_project_split import PARTS, greedy_assign, objective, refine  # noqa: E402


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def load_rows(path, id_field):
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
                continue
            rows.append({"id": str(o.get(id_field, n)), "label": 1 if o.get("target") == 1 else 0,
                         "commit": str(o.get("commit_id") or "")})
    return rows


def build_groups(rows, fam):
    """Connected components of same-commit OR same-family; returns {('g', root): [row idx]}."""
    uf = UF(len(rows))
    first = {}
    for i, r in enumerate(rows):
        for key in (("c", r["commit"]) if r["commit"] else None,
                    ("f", fam.get(r["id"])) if fam.get(r["id"]) is not None else None):
            if key is None:
                continue
            j = first.setdefault(key, i)
            if j != i:
                uf.union(i, j)
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[("g", uf.find(i))].append(i)
    return groups


def leak(rows, part_of, fam):
    tr = {fam.get(rows[i]["id"]) for i in range(len(rows)) if part_of[i] == 0} - {None}
    te = [i for i in range(len(rows)) if part_of[i] == 2]
    k = sum(1 for i in te if fam.get(rows[i]["id"]) in tr)
    return {"leaked_test_functions": k, "leaked_test_pct": round(100 * k / len(te), 2) if te else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--hr-famfile", required=True)
    ap.add_argument("--ref-splitfile", required=True, help="commit-disjoint split: size/positive targets")
    ap.add_argument("--strict-famfile", default=None, help="strict theta=0.7 families, for the report")
    ap.add_argument("--id-field", default="idx")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for p in (a.data, a.hr_famfile, a.ref_splitfile) + ((a.strict_famfile,) if a.strict_famfile else ()):
        if not os.path.exists(p):
            print(f"!!! missing input: {p}", file=sys.stderr)
            return 2

    rows = load_rows(a.data, a.id_field)
    n = len(rows)
    fam = {str(k): v for k, v in json.load(open(a.hr_famfile, encoding="utf-8")).items()}
    ref = {str(k): ("valid" if v in ("valid", "validation") else v)
           for k, v in json.load(open(a.ref_splitfile, encoding="utf-8")).items()}
    missing_fam = sum(1 for r in rows if r["id"] not in fam)
    missing_ref = sum(1 for r in rows if r["id"] not in ref)
    if missing_ref:
        print(f"!!! {missing_ref} functions have no partition in the reference split", file=sys.stderr)
        return 2
    size_target = [sum(1 for r in rows if ref[r["id"]] == p) for p in PARTS]
    pos_target = [sum(r["label"] for r in rows if ref[r["id"]] == p) for p in PARTS]

    groups = build_groups(rows, fam)
    pog = greedy_assign(groups, rows, size_target, pos_target, a.seed)

    def j_of(m):
        sz, ps = [0, 0, 0], [0, 0, 0]
        for g, t in m.items():
            sz[t] += len(groups[g])
            ps[t] += sum(rows[i]["label"] for i in groups[g])
        return objective(sz, ps, size_target, pos_target)
    j_greedy = j_of(pog)
    pog, rstats = refine(groups, rows, pog, size_target, pos_target, a.seed)
    j_final = j_of(pog)
    part_of = [None] * n
    for g, idxs in groups.items():
        for i in idxs:
            part_of[i] = pog[g]

    # invariants: no commit and no high-recall family in two partitions
    span = {"commit": defaultdict(set), "hr_family": defaultdict(set)}
    for i, r in enumerate(rows):
        if r["commit"]:
            span["commit"][r["commit"]].add(part_of[i])
        if fam.get(r["id"]) is not None:
            span["hr_family"][fam[r["id"]]].add(part_of[i])
    bad = {k: sum(1 for s in v.values() if len(s) > 1) for k, v in span.items()}
    assert not any(bad.values()), f"groups split across partitions: {bad}"

    assign = {rows[i]["id"]: PARTS[part_of[i]] for i in range(n)}
    json.dump(assign, open(f"{a.out}.json", "w", encoding="utf-8"))

    size = [sum(1 for t in part_of if t == k) for k in range(3)]
    pos = [sum(rows[i]["label"] for i in range(n) if part_of[i] == k) for k in range(3)]
    gs = sorted((len(v) for v in groups.values()), reverse=True)
    big = max(groups, key=lambda g: len(groups[g]))
    rep = {
        "data": a.data, "hr_famfile": a.hr_famfile, "ref_splitfile": a.ref_splitfile, "seed": a.seed,
        "relations": "same commit_id OR same high-recall theta=0.7 family (connected components)",
        "n_functions": n, "functions_without_hr_family": missing_fam,
        "functions_without_commit": sum(1 for r in rows if not r["commit"]),
        "n_groups": len(groups), "n_groups_gt1": sum(1 for x in gs if x > 1),
        "largest_group": {"functions": gs[0], "share_pct": round(100 * gs[0] / n, 2),
                          "partition": PARTS[pog[big]],
                          "positives": sum(rows[i]["label"] for i in groups[big]),
                          "commits": len({rows[i]["commit"] for i in groups[big] if rows[i]["commit"]})},
        "top10_group_sizes": gs[:10],
        "split_across_partitions": bad,
        "balance_objective": {"greedy": round(j_greedy, 6), "final": round(j_final, 6), "refine": rstats},
        "partitions": {PARTS[k]: {"functions": size[k], "target_functions": size_target[k],
                                  "pos_rate": round(pos[k] / size[k], 4) if size[k] else None,
                                  "target_pos_rate": round(pos_target[k] / size_target[k], 4)
                                  if size_target[k] else None} for k in range(3)},
        "hr_family_leak_under_this_split": leak(rows, part_of, fam),
    }
    if a.strict_famfile:
        sf = {str(k): v for k, v in json.load(open(a.strict_famfile, encoding="utf-8")).items()}
        rep["strict_family_leak_under_this_split"] = leak(rows, part_of, sf)
    json.dump(rep, open(f"{a.out}_report.json", "w", encoding="utf-8"), indent=2)

    for k in range(3):
        p = rep["partitions"][PARTS[k]]
        print(f"  {PARTS[k]:5s} {p['functions']:>8,} (target {p['target_functions']:,})  "
              f"pos {p['pos_rate']} (target {p['target_pos_rate']})")
    lg = rep["largest_group"]
    print(f"  groups {len(groups):,}; largest {lg['functions']:,} fns ({lg['share_pct']}% of corpus, "
          f"{lg['commits']:,} commits) -> {lg['partition']}; top10 {gs[:10]}")
    print(f"  balance objective J: greedy {j_greedy:.5f} -> final {j_final:.5f} ({rstats['moves']} moves)")
    if lg["share_pct"] > 15:
        print(f"  ! largest group is {lg['share_pct']}% of the corpus (> 15%): precheck_h4.py will refuse "
              f"this split", file=sys.stderr)
    print(f"wrote {a.out}.json and {a.out}_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
