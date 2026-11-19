#!/usr/bin/env python3
"""
make_project_split.py -- RESEARCH_PLAN §13.3 item 6 (CPU only): the
project-disjoint arm for pre-registered hypothesis H2 (plan §13.2).

Assigns WHOLE PROJECTS to train / valid / test so no project appears in more
than one partition, while matching the partition sizes and positive rates of a
reference split (normally the family-disjoint split), so that
Δ(random_control -> project_disjoint) can be read against
Δ(random_control -> family_disjoint) on equal-size, equal-prevalence test sets.

Placement starts from the same two-key greedy as
clone_family_audit.family_disjoint_split (projects shuffled with --seed, then
placed largest first into the partition furthest behind on size and positives).
A few very large projects (linux is 21.9% of DiverseVul) can leave the greedy far
off target, so a REFINEMENT pass follows (on by default, --no-refine to skip):
whole projects are moved between partitions, one at a time, whenever the move
lowers  J = sum over valid,test of (size/target - 1)^2 + (positives/target - 1)^2
-- train absorbs the remainder. J uses only partition sizes and label counts,
never model output, so it is balancing, not tuning. Both the greedy and the
refined J are reported.

Reported alongside the split, because a reviewer will ask:
  * realised sizes and positive rates vs the reference;
  * number of projects per partition, and the largest project's share;
  * the Type-3 family leak rate UNDER this split -- clone families cross
    projects (6,064 on DiverseVul at θ=0.7), so project-disjoint is not
    family-disjoint, and the residual leak is part of the H2 reading.
Functions with an empty project field are grouped one per function and
counted in the report.

USAGE (DiverseVul, on the server)
  D=$CFA_DATA_DIR
  python make_project_split.py --data $D/data/diversevul/diversevul.jsonl \
      --ref-splitfile $D/diversevul_audit_splitfile_theta0.7.json \
      --famfile $D/diversevul_audit_families_theta0.7.json \
      --seed 42 --out $D/diversevul_audit_splitfile_projectdisjoint_seed42

OUTPUT
  <out>.json         id -> train/valid/test  (finetune_vd.py --splitfile)
  <out>_report.json  the numbers above
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict

PARTS = ("train", "valid", "test")


def load_corpus(path, id_field, group_field="project"):
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
            rows.append({"id": str(o.get(id_field, n)),
                         "label": 1 if o.get("target") == 1 else 0,
                         "project": str(o.get(group_field, "") or "")})
    return rows


def greedy_assign(groups, rows, size_target, pos_target, seed):
    """Two-key greedy, identical in form to family_disjoint_split."""
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: -len(kv[1]))          # stable: ties keep shuffled order
    have_size = [0, 0, 0]
    have_pos = [0, 0, 0]
    part_of_group = {}
    for g, idxs in items:
        sz = len(idxs)
        pz = sum(rows[i]["label"] for i in idxs)
        best_t, best_cost = 0, None
        for t in range(3):
            if size_target[t] <= 0:
                continue
            size_def = (size_target[t] - have_size[t]) / size_target[t]
            pos_def = ((pos_target[t] - have_pos[t]) / pos_target[t]
                       if pos_target[t] > 0 else 0.0)
            cost = -(size_def + 2.0 * pos_def)
            if best_cost is None or cost < best_cost:
                best_t, best_cost = t, cost
        part_of_group[g] = best_t
        have_size[best_t] += sz
        have_pos[best_t] += pz
    return part_of_group


def objective(size, pos, size_target, pos_target):
    j = 0.0
    for t in (1, 2):                       # valid, test; train takes the rest
        if size_target[t] > 0:
            j += (size[t] / size_target[t] - 1.0) ** 2
        if pos_target[t] > 0:
            j += (pos[t] / pos_target[t] - 1.0) ** 2
    return j


def refine(groups, rows, part_of_group, size_target, pos_target, seed, max_passes=200):
    """First-improvement local search over single whole-project moves."""
    gsize = {g: len(v) for g, v in groups.items()}
    gpos = {g: sum(rows[i]["label"] for i in v) for g, v in groups.items()}
    size = [0, 0, 0]
    pos = [0, 0, 0]
    for g, t in part_of_group.items():
        size[t] += gsize[g]
        pos[t] += gpos[g]
    j = objective(size, pos, size_target, pos_target)
    order = sorted(groups, key=lambda g: (str(g[0]), str(g[1])))
    random.Random(seed + 7).shuffle(order)
    moves = passes = 0
    for passes in range(1, max_passes + 1):
        moved = False
        for g in order:
            src = part_of_group[g]
            for dst in range(3):
                if dst == src:
                    continue
                size[src] -= gsize[g]; pos[src] -= gpos[g]
                size[dst] += gsize[g]; pos[dst] += gpos[g]
                jn = objective(size, pos, size_target, pos_target)
                if jn < j - 1e-12:
                    part_of_group[g] = dst
                    j = jn
                    moves += 1
                    moved = True
                    break
                size[dst] -= gsize[g]; pos[dst] -= gpos[g]
                size[src] += gsize[g]; pos[src] += gpos[g]
        if not moved:
            break
    return part_of_group, {"moves": moves, "passes": passes}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ref-splitfile", default=None,
                    help="split whose partition sizes and positive rates to match "
                         "(default: 80/10/10 of the corpus at its overall prevalence)")
    ap.add_argument("--famfile", default=None, help="families file, for the residual leak rate")
    ap.add_argument("--id-field", default="idx")
    ap.add_argument("--group-field", default="project",
                    help="field whose values are kept whole (default project; "
                         "commit_id gives the commit-disjoint arm)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-refine", action="store_true",
                    help="keep the plain greedy placement (no local-search refinement)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_corpus(args.data, args.id_field, args.group_field)
    if not rows:
        print("no rows loaded", file=sys.stderr)
        return 2
    n, n_pos = len(rows), sum(r["label"] for r in rows)

    if args.ref_splitfile:
        with open(args.ref_splitfile, encoding="utf-8") as fh:
            ref = json.load(fh)
        ref_of = [ref.get(r["id"]) for r in rows]
        ref_of = ["valid" if s in ("valid", "validation") else s for s in ref_of]
        size_target = [sum(1 for s in ref_of if s == p) for p in PARTS]
        pos_target = [sum(r["label"] for r, s in zip(rows, ref_of) if s == p) for p in PARTS]
        if sum(size_target) != n:
            print(f"  ! reference covers {sum(size_target):,} of {n:,} functions; "
                  f"targets are taken from the covered part", file=sys.stderr)
    else:
        size_target = [0.8 * n, 0.1 * n, 0.1 * n]
        pos_target = [0.8 * n_pos, 0.1 * n_pos, 0.1 * n_pos]

    groups = defaultdict(list)
    no_project = 0
    for i, r in enumerate(rows):
        if r["project"]:
            groups[("p", r["project"])].append(i)
        else:
            no_project += 1
            groups[("single", i)].append(i)

    part_of_group = greedy_assign(groups, rows, size_target, pos_target, args.seed)

    def j_of(pog):
        sz, ps = [0, 0, 0], [0, 0, 0]
        for g, t in pog.items():
            sz[t] += len(groups[g])
            ps[t] += sum(rows[i]["label"] for i in groups[g])
        return objective(sz, ps, size_target, pos_target)
    j_greedy = j_of(part_of_group)
    refine_stats = None
    if not args.no_refine:
        part_of_group, refine_stats = refine(groups, rows, part_of_group,
                                             size_target, pos_target, args.seed)
    j_final = j_of(part_of_group)
    part_of = [None] * n
    for g, idxs in groups.items():
        for i in idxs:
            part_of[i] = part_of_group[g]

    # --- invariant: no project in more than one partition ---------------------
    parts_of_project = defaultdict(set)
    for i, r in enumerate(rows):
        if r["project"]:
            parts_of_project[r["project"]].add(part_of[i])
    overlap = [p for p, s in parts_of_project.items() if len(s) > 1]
    assert not overlap, f"{args.group_field} overlap across partitions: {overlap[:5]}"

    assign = {rows[i]["id"]: PARTS[part_of[i]] for i in range(n)}
    with open(f"{args.out}.json", "w", encoding="utf-8") as fh:
        json.dump(assign, fh)

    size = [sum(1 for t in part_of if t == k) for k in range(3)]
    pos = [sum(rows[i]["label"] for i in range(n) if part_of[i] == k) for k in range(3)]
    proj_count = Counter(part_of_group[g] for g in groups if g[0] == "p")
    proj_size = Counter({g[1]: len(v) for g, v in groups.items() if g[0] == "p"})
    biggest = proj_size.most_common(1)[0] if proj_size else ("", 0)
    rep = {
        "data": args.data, "ref_splitfile": args.ref_splitfile, "seed": args.seed,
        "group_field": args.group_field,
        "n_functions": n, "n_projects": len(proj_size),
        "functions_without_project": no_project,
        "largest_project": {"name": biggest[0], "functions": biggest[1],
                            "share_pct": round(100 * biggest[1] / n, 2),
                            "partition": PARTS[part_of_group[("p", biggest[0])]] if biggest[1] else None},
        "projects_in_more_than_one_partition": 0,
        "balance_objective": {"greedy": round(j_greedy, 6), "final": round(j_final, 6),
                              "refined": not args.no_refine, "refine": refine_stats},
        "partitions": {
            PARTS[k]: {"functions": size[k], "target_functions": round(size_target[k]),
                       "pos_rate": round(pos[k] / size[k], 4) if size[k] else None,
                       "target_pos_rate": round(pos_target[k] / size_target[k], 4) if size_target[k] else None,
                       "projects": proj_count.get(k, 0)}
            for k in range(3)
        },
    }

    if args.famfile:
        with open(args.famfile, encoding="utf-8") as fh:
            fam = json.load(fh)
        train_fams = {fam.get(rows[i]["id"]) for i in range(n) if part_of[i] == 0}
        train_fams.discard(None)
        test_idx = [i for i in range(n) if part_of[i] == 2]
        leaked = sum(1 for i in test_idx if fam.get(rows[i]["id"]) in train_fams)
        rep["family_leak_under_this_split"] = {
            "famfile": args.famfile, "leaked_test_functions": leaked,
            "leaked_test_pct": round(100 * leaked / len(test_idx), 2) if test_idx else None}

    with open(f"{args.out}_report.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)

    for k in range(3):
        p = rep["partitions"][PARTS[k]]
        print(f"  {PARTS[k]:5s} {p['functions']:>8,} (target {p['target_functions']:,})  "
              f"pos {p['pos_rate']} (target {p['target_pos_rate']})  projects {p['projects']}")
    print(f"  balance objective J: greedy {j_greedy:.5f} -> final {j_final:.5f}"
          + (f"  ({refine_stats['moves']} project moves)" if refine_stats else ""))
    lp = rep["largest_project"]
    print(f"  largest project {lp['name']!r}: {lp['functions']:,} fns ({lp['share_pct']}%) -> {lp['partition']}")
    if "family_leak_under_this_split" in rep:
        print(f"  Type-3 family leak under this split: "
              f"{rep['family_leak_under_this_split']['leaked_test_pct']}% of test")
    for k in range(3):
        if size_target[k] and abs(size[k] - size_target[k]) / size_target[k] > 0.05:
            print(f"  ! {PARTS[k]} is {100 * (size[k] - size_target[k]) / size_target[k]:+.1f}% "
                  f"off its size target -- report it, or try another --seed", file=sys.stderr)
        tp = pos_target[k] / size_target[k] if size_target[k] else 0
        if size[k] and tp and abs(pos[k] / size[k] - tp) / tp > 0.10:
            print(f"  ! {PARTS[k]} positive rate {pos[k] / size[k]:.4f} is more than 10% "
                  f"off its target {tp:.4f} -- report it, or try another --seed", file=sys.stderr)
    print(f"wrote {args.out}.json and {args.out}_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
