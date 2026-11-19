#!/usr/bin/env python3
"""
primevul_tools.py -- RESEARCH_PLAN §13.3 item 3 (CPU only, no GPU).

PrimeVul (Ding et al., arXiv:2403.18624) ships its split as three files
(primevul_{train,valid,test}.jsonl) plus `_paired` subsets of explicit
vulnerable/patched function pairs. Two subcommands:

  prep   Merge the three split files into one JSONL with a `split` field -- the
         format clone_family_audit.py and family_label_audit.py already read --
         and check it against the paper before anything is computed on it
         (RESEARCH_PLAN §9.1 standing rule): total 235,768 functions, 6,968
         vulnerable. Also reports, for later interpretation:
           * whether `idx` is unique across the three files (if not, ids become
             "<split>:<idx>" so no function is silently merged with another)
           * PrimeVul's own `hash` field shared across partitions (its dedup
             claim; expected 0)
           * how many `_paired` functions are present in the main split files

  pairs  For every vulnerable function in a `_paired` file, is a patched
         (target 0) function from the SAME commit in the SAME clone family?
         Reported per theta. This is a direct, model-free check that the
         family detector places vulnerability/patch pairs together -- i.e. that
         the "same family, opposite label" mechanism behind H1 is real on a
         dataset whose pairs were built by someone else.

USAGE (on the server; run_primevul_audit.sh calls both)
  python primevul_tools.py prep  --dir $CFA_DATA_DIR/data/primevul
  python primevul_tools.py pairs --dir $CFA_DATA_DIR/data/primevul \
      --famfile 0.7=$CFA_DATA_DIR/primevul_audit_families_theta0.7.json \
      --out $CFA_DATA_DIR/family_label/primevul_pairs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

SPLITS = ("train", "valid", "test")
PAPER_TOTAL, PAPER_POS = 235768, 6968   # Ding et al., PrimeVul Table 2 / §3


def read_jsonl(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                yield n, json.loads(line)
            except json.JSONDecodeError:
                print(f"  ! {os.path.basename(path)} line {n + 1}: bad JSON, skipped",
                      file=sys.stderr)


def make_id(split, idx, prefixed):
    return f"{split}:{idx}" if prefixed else str(idx)


# ---------------------------------------------------------------------------
def cmd_prep(args):
    src = {s: os.path.join(args.dir, f"primevul_{s}.jsonl") for s in SPLITS}
    for s, p in src.items():
        if not os.path.isfile(p):
            print(f"!!! missing {p}", file=sys.stderr)
            return 1

    # pass 1: is idx unique across the three files?
    seen = Counter()
    for s in SPLITS:
        for _, o in read_jsonl(src[s]):
            seen[str(o.get("idx"))] += 1
    dup_idx = sum(1 for c in seen.values() if c > 1)
    prefixed = dup_idx > 0
    if prefixed:
        print(f"  ! {dup_idx} idx values repeat across files -- ids become '<split>:<idx>'")

    out = os.path.join(args.dir, "primevul_all.jsonl")
    tmp = out + ".tmp"
    cnt, pos, empty = Counter(), Counter(), Counter()
    hash_splits = defaultdict(set)
    projects = defaultdict(set)
    ids = {}
    with open(tmp, "w", encoding="utf-8") as fo:
        for s in SPLITS:
            for _, o in read_jsonl(src[s]):
                func = o.get("func")
                if not isinstance(func, str) or not func.strip():
                    empty[s] += 1          # same filter as the audit tools
                    continue
                lab = 1 if o.get("target") == 1 else 0
                fid = make_id(s, o.get("idx"), prefixed)
                ids[(s, str(o.get("idx")))] = fid
                cnt[s] += 1
                pos[s] += lab
                if o.get("hash") is not None:
                    hash_splits[str(o["hash"])].add(s)
                projects[s].add(str(o.get("project") or ""))
                fo.write(json.dumps({
                    "idx": fid,
                    "func": func,
                    "target": lab,
                    "project": o.get("project") or "",
                    "commit_id": o.get("commit_id") or "",
                    "cwe": o.get("cwe") or [],
                    "big_vul_idx": o.get("big_vul_idx"),
                    "hash": o.get("hash"),
                    "split": s,
                }) + "\n")
    # moved into place only after the paper check below passes, so a mismatching
    # corpus can never be picked up by a later step

    total, total_pos = sum(cnt.values()), sum(pos.values())
    cross_hash = Counter()
    for sp in hash_splits.values():
        if len(sp) > 1:
            cross_hash["+".join(x for x in SPLITS if x in sp)] += 1

    # paired subsets: present in the main file of the same split?
    paired = {}
    for s in SPLITS:
        p = os.path.join(args.dir, f"primevul_{s}_paired.jsonl")
        if not os.path.isfile(p):
            continue
        n = found = npos = 0
        for _, o in read_jsonl(p):
            n += 1
            npos += 1 if o.get("target") == 1 else 0
            found += (s, str(o.get("idx"))) in ids
        paired[s] = {"functions": n, "vulnerable": npos, "in_main_split_file": found}

    rep = {
        "output": out,
        "ids_prefixed_with_split": prefixed,
        "duplicate_idx_values": dup_idx,
        "counts": dict(cnt), "positives": dict(pos),
        "pos_rate": {s: round(pos[s] / cnt[s], 4) for s in SPLITS if cnt[s]},
        "dropped_empty_func": dict(empty),
        "total": total, "total_positive": total_pos,
        "paper_total": args.expect_total, "paper_positive": args.expect_pos,
        "matches_paper": total == args.expect_total and total_pos == args.expect_pos,
        "hashes_shared_across_splits": dict(cross_hash),
        "projects": {s: len(projects[s]) for s in SPLITS},
        "projects_in_train_and_test": len(projects["train"] & projects["test"]),
        "paired": paired,
    }
    with open(os.path.join(args.dir, "primevul_prep_report.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps(rep, indent=2))
    if rep["matches_paper"]:
        os.replace(tmp, out)
    else:
        os.remove(tmp)
        print(f"!!! counts {total}/{total_pos} != paper {args.expect_total}/{args.expect_pos} "
              "-- §9.1 rule: do not use until explained", file=sys.stderr)
        return 3
    return 0


# ---------------------------------------------------------------------------
def cmd_pairs(args):
    rep = json.load(open(os.path.join(args.dir, "primevul_prep_report.json")))
    prefixed = rep["ids_prefixed_with_split"]
    fams = []
    for spec in args.famfile:
        theta, path = spec.split("=", 1)
        fams.append((theta, json.load(open(path))))

    result = {"note": "per vulnerable function in a _paired file: is a target-0 "
                      "function of the same commit in the same family?",
              "splits": {}}
    for s in SPLITS:
        p = os.path.join(args.dir, f"primevul_{s}_paired.jsonl")
        if not os.path.isfile(p):
            continue
        by_commit = defaultdict(lambda: ([], []))   # commit -> (vuln ids, patched ids)
        for _, o in read_jsonl(p):
            fid = make_id(s, o.get("idx"), prefixed)
            by_commit[str(o.get("commit_id") or "")][0 if o.get("target") == 1 else 1].append(fid)
        shape = Counter(f"{len(v)}v+{len(b)}p" for v, b in by_commit.values())
        row = {"commits": len(by_commit),
               "vulnerable": sum(len(v) for v, _ in by_commit.values()),
               "commit_shapes_top": dict(shape.most_common(5)),
               "theta": {}}
        for theta, fam in fams:
            scored = together = missing = 0
            for vs, ps in by_commit.values():
                pfams = {fam[x] for x in ps if x in fam}
                for v in vs:
                    if v not in fam:
                        missing += 1
                        continue
                    if not ps:
                        continue
                    scored += 1
                    together += fam[v] in pfams
            row["theta"][theta] = {
                "vulnerable_with_a_same_commit_patch": scored,
                "patch_in_same_family": together,
                "pct": round(100.0 * together / scored, 2) if scored else None,
                "vulnerable_not_in_family_file": missing,
            }
        result["splits"][s] = row

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as fh:
        json.dump(result, fh, indent=2)
    with open(args.out + ".md", "w") as fh:
        fh.write("# PrimeVul vulnerable/patch pairs -- same clone family?\n\n")
        fh.write("| split | theta | vulnerable fns with a same-commit patch | "
                 "patch in same family | % |\n|---|---|---|---|---|\n")
        for s, row in result["splits"].items():
            for theta, t in row["theta"].items():
                fh.write(f"| {s} | {theta} | {t['vulnerable_with_a_same_commit_patch']} | "
                         f"{t['patch_in_same_family']} | {t['pct']} |\n")
    print(open(args.out + ".md").read())
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prep")
    a.add_argument("--dir", required=True)
    a.add_argument("--expect-total", type=int, default=PAPER_TOTAL,
                   help="paper count to check against (override only for tests)")
    a.add_argument("--expect-pos", type=int, default=PAPER_POS)
    b = sub.add_parser("pairs")
    b.add_argument("--dir", required=True)
    b.add_argument("--famfile", action="append", required=True, metavar="THETA=PATH")
    b.add_argument("--out", required=True)
    args = ap.parse_args()
    return cmd_prep(args) if args.cmd == "prep" else cmd_pairs(args)


if __name__ == "__main__":
    sys.exit(main())
