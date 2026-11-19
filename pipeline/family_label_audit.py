#!/usr/bin/env python3
"""
family_label_audit.py -- RESEARCH_PLAN §13.3 items 1-2 (CPU only, no GPU).

Reads the corpus JSONL and a families file written by clone_family_audit.py and
reports, without retraining anything:

  1. FAMILY SIZE DISTRIBUTION  -- how many families, how big, how much of the
     corpus the largest one covers (answers the giant-component objection from
     Tashenova et al. 2026, whose plugin+family components covered 59-72%).
  2. LABEL CONSISTENCY         -- per family (pure-negative / pure-positive /
     mixed) AND per function (share of functions, and of vulnerable functions,
     that sit in a mixed-label family). The function-level number is the one
     comparable with Croft et al.'s Type-1 consistency (ICSE 2023, p8).
  3. LEAK CATEGORIES per arm   -- every test function is put in exactly one of
         clean            no family member in train
         leaked_same      all train family-mates carry the function's own label
         leaked_opposite  all train family-mates carry the opposite label
         leaked_both      train family-mates carry both labels
     Croft et al. count a test item as a duplicate only when its training clone
     has the same label, i.e. leaked_same + leaked_both. Both definitions are
     reported. The per-id categories are written to a file so the Phase 2
     prediction analysis (pre-registered H1, plan §13.2) can join on them
     without recomputing anything.

The published arm uses the JSONL's own `split` field; every --splitfile adds
one more arm (id -> train/valid/test, the format clone_family_audit.py and
make_project_split.py write). Families are taken as given; nothing is
re-clustered here.

USAGE (DiverseVul, on the server)
  D=$CFA_DATA_DIR
  python family_label_audit.py --data $D/data/diversevul/diversevul.jsonl \
      --famfile $D/diversevul_audit_families_theta0.7.json \
      --splitfile random_control=$D/diversevul_audit_splitfile_randomctrl_theta0.7.json \
      --splitfile family_disjoint=$D/diversevul_audit_splitfile_theta0.7.json \
      --out $D/family_label/diversevul_theta0.7

OUTPUT
  <out>.json               every number below, machine-readable
  <out>.md                 the same as tables
  <out>_leakcat_<arm>.json test id -> leak category, one file per arm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

CATS = ("clean", "leaked_same", "leaked_opposite", "leaked_both")
SIZE_BINS = ((1, 1), (2, 2), (3, 5), (6, 10), (11, 50), (51, 200), (201, None))


def load_corpus(path, id_field, label_field):
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
            # same filter as finetune_vd.py / clone_family_audit.py
            if not isinstance(o.get("func"), str) or not o["func"].strip():
                continue
            sp = str(o.get("split", "all"))
            rows.append({
                "id": str(o.get(id_field, n)),
                "label": 1 if o.get(label_field) == 1 else 0,
                "split": "valid" if sp in ("valid", "validation") else sp,
                "project": str(o.get("project", "") or ""),
            })
    return rows


def pct(a, b):
    return round(100.0 * a / b, 2) if b else None


def size_distribution(members, n_rows, n_unassigned):
    sizes = sorted((len(v) for v in members.values()), reverse=True)
    multi = [s for s in sizes if s > 1]
    in_multi = sum(multi)
    bins = {}
    for lo, hi in SIZE_BINS:
        name = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi else f">{lo - 1}")
        sel = [s for s in sizes if s >= lo and (hi is None or s <= hi)]
        bins[name] = {"families": len(sel), "functions": sum(sel)}

    def q(p):
        return multi[int(p * (len(multi) - 1))] if multi else None
    return {
        "n_functions": n_rows,
        "n_functions_without_family": n_unassigned,
        "n_families_total": len(sizes),
        "n_families_size_gt1": len(multi),
        "functions_in_families_gt1": in_multi,
        "functions_in_families_gt1_pct": pct(in_multi, n_rows),
        "largest_family_size": sizes[0] if sizes else 0,
        "largest_family_share_pct": pct(sizes[0], n_rows) if sizes else None,
        "top10_family_sizes": sizes[:10],
        # sizes is sorted descending, so q(0.5) is the median and q(0.1) the 90th pct
        "size_gt1_median": q(0.5),
        "size_gt1_p90": q(0.1),
        "size_gt1_p99": q(0.01),
        "size_bins": bins,
    }


def label_consistency(rows, members):
    fam_pure_neg = fam_pure_pos = fam_mixed = 0
    fn_in_mixed = pos_in_mixed = 0
    n_pos = sum(r["label"] for r in rows)
    for idxs in members.values():
        if len(idxs) < 2:
            continue
        labs = [rows[i]["label"] for i in idxs]
        s = sum(labs)
        if s == 0:
            fam_pure_neg += 1
        elif s == len(labs):
            fam_pure_pos += 1
        else:
            fam_mixed += 1
            fn_in_mixed += len(idxs)
            pos_in_mixed += s
    n_multi = fam_pure_neg + fam_pure_pos + fam_mixed
    return {
        "families_gt1": n_multi,
        "families_pure_negative": fam_pure_neg,
        "families_pure_positive": fam_pure_pos,
        "families_mixed_label": fam_mixed,
        "families_mixed_label_pct": pct(fam_mixed, n_multi),
        "functions_in_mixed_families": fn_in_mixed,
        "functions_in_mixed_families_pct": pct(fn_in_mixed, len(rows)),
        "positives_total": n_pos,
        "positives_in_mixed_families": pos_in_mixed,
        "positives_in_mixed_families_pct": pct(pos_in_mixed, n_pos),
    }


def cross_project(rows, members):
    n_multi = cross = 0
    for idxs in members.values():
        if len(idxs) < 2:
            continue
        n_multi += 1
        projs = {rows[i]["project"] for i in idxs if rows[i]["project"]}
        if len(projs) > 1:
            cross += 1
    return {"families_gt1": n_multi, "cross_project_families": cross,
            "cross_project_families_pct": pct(cross, n_multi)}


def leak_categories(rows, fam_of, split_of):
    """split_of: list aligned with rows (None = not in this arm)."""
    train_labels = defaultdict(set)
    for i, r in enumerate(rows):
        if split_of[i] == "train" and fam_of[i] is not None:
            train_labels[fam_of[i]].add(r["label"])

    cat_of = {}
    by_cat = Counter()
    by_cat_label = Counter()
    n_test = 0
    for i, r in enumerate(rows):
        if split_of[i] != "test":
            continue
        n_test += 1
        labs = train_labels.get(fam_of[i], set()) if fam_of[i] is not None else set()
        if not labs:
            c = "clean"
        elif labs == {r["label"]}:
            c = "leaked_same"
        elif r["label"] not in labs:
            c = "leaked_opposite"
        else:
            c = "leaked_both"
        cat_of[r["id"]] = c
        by_cat[c] += 1
        by_cat_label[(c, r["label"])] += 1

    leaked = n_test - by_cat["clean"]
    croft = by_cat["leaked_same"] + by_cat["leaked_both"]
    out = {
        "n_test": n_test,
        "n_test_positive": sum(by_cat_label[(c, 1)] for c in CATS),
        "leaked_any": leaked,
        "leaked_any_pct": pct(leaked, n_test),
        "leaked_croft_definition": croft,
        "leaked_croft_definition_pct": pct(croft, n_test),
        "categories": {
            c: {"n": by_cat[c], "pct_of_test": pct(by_cat[c], n_test),
                "negatives": by_cat_label[(c, 0)], "positives": by_cat_label[(c, 1)],
                "pos_rate": round(by_cat_label[(c, 1)] / by_cat[c], 4) if by_cat[c] else None}
            for c in CATS
        },
    }
    return out, cat_of


def write_md(path, rep):
    L = [f"# Family label audit — `{os.path.basename(rep['famfile'])}`", "",
         "_Generated by family_label_audit.py from the corpus and families file; "
         "do not edit by hand._", ""]
    s = rep["size_distribution"]
    L += ["## 1. Family size distribution", "",
          f"- functions: {s['n_functions']:,} (without a family id: {s['n_functions_without_family']:,})",
          f"- families: {s['n_families_total']:,} total, {s['n_families_size_gt1']:,} with >1 member",
          f"- functions in families >1: {s['functions_in_families_gt1']:,} "
          f"({s['functions_in_families_gt1_pct']}%)",
          f"- largest family: {s['largest_family_size']:,} functions "
          f"= **{s['largest_family_share_pct']}% of the corpus**",
          f"- size of families >1: median {s['size_gt1_median']}, p90 {s['size_gt1_p90']}, "
          f"p99 {s['size_gt1_p99']}; top-10 {s['top10_family_sizes']}", "",
          "| family size | families | functions |", "|---|---|---|"]
    L += [f"| {k} | {v['families']:,} | {v['functions']:,} |" for k, v in s["size_bins"].items()]
    c = rep["label_consistency"]
    L += ["", "## 2. Label consistency (families >1)", "",
          "| | count | % |", "|---|---|---|",
          f"| pure negative families | {c['families_pure_negative']:,} | |",
          f"| pure positive families | {c['families_pure_positive']:,} | |",
          f"| mixed-label families | {c['families_mixed_label']:,} | {c['families_mixed_label_pct']} |",
          f"| functions in mixed families | {c['functions_in_mixed_families']:,} | "
          f"{c['functions_in_mixed_families_pct']} (of all functions) |",
          f"| positives in mixed families | {c['positives_in_mixed_families']:,} | "
          f"{c['positives_in_mixed_families_pct']} (of all positives) |"]
    x = rep["cross_project"]
    L += ["", f"Cross-project families: {x['cross_project_families']:,} of "
          f"{x['families_gt1']:,} families >1 ({x['cross_project_families_pct']}%).", "",
          "## 3. Test-set leak categories per arm", "",
          "clean = no family member in train · same/opposite/both = labels of the train "
          "family-mates relative to the test function. Croft-style duplicate = same + both.", "",
          "| arm | n test | leaked any % | Croft-def % | clean | same | opposite | both |",
          "|---|---|---|---|---|---|---|---|"]
    for arm, a in rep["arms"].items():
        cc = a["categories"]
        cell = lambda k: f"{cc[k]['n']:,} ({cc[k]['pct_of_test']}%, pos {cc[k]['pos_rate']})"
        L.append(f"| {arm} | {a['n_test']:,} | {a['leaked_any_pct']} | "
                 f"{a['leaked_croft_definition_pct']} | {cell('clean')} | {cell('leaked_same')} | "
                 f"{cell('leaked_opposite')} | {cell('leaked_both')} |")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--famfile", required=True)
    ap.add_argument("--splitfile", action="append", default=[], metavar="ARM=PATH",
                    help="extra arm as name=path (repeatable)")
    ap.add_argument("--no-published", action="store_true",
                    help="skip the arm defined by the JSONL's own split field")
    ap.add_argument("--id-field", default="idx")
    ap.add_argument("--label-field", default="target")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_corpus(args.data, args.id_field, args.label_field)
    if not rows:
        print("no rows loaded", file=sys.stderr)
        return 2
    with open(args.famfile, encoding="utf-8") as fh:
        fam = json.load(fh)
    fam_of = [fam.get(r["id"]) for r in rows]
    missing = sum(1 for f in fam_of if f is None)
    # A function with no family id is treated as its own singleton family:
    # it can never be leaked, which is the conservative direction.
    members = defaultdict(list)
    for i, f in enumerate(fam_of):
        members[f if f is not None else ("__single__", i)].append(i)
    print(f"loaded {len(rows):,} functions; {len(fam):,} family ids; "
          f"{missing:,} functions without one")
    if missing > 0.01 * len(rows):
        print(f"  ! {missing:,} functions have no family id -- check --id-field and "
              f"that --famfile was built from this --data", file=sys.stderr)

    rep = {
        "data": args.data, "famfile": args.famfile, "id_field": args.id_field,
        "size_distribution": size_distribution(members, len(rows), missing),
        "label_consistency": label_consistency(rows, members),
        "cross_project": cross_project(rows, members),
        "arms": {},
    }

    arms = []
    if not args.no_published:
        arms.append(("published", [r["split"] for r in rows]))
    for spec in args.splitfile:
        if "=" not in spec:
            print(f"--splitfile wants ARM=PATH, got {spec!r}", file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        with open(path, encoding="utf-8") as fh:
            assign = json.load(fh)
        split_of = [assign.get(r["id"]) for r in rows]
        absent = sum(1 for s in split_of if s is None)
        if absent:
            print(f"  ! arm {name}: {absent:,} corpus ids absent from {path}", file=sys.stderr)
        arms.append((name, ["valid" if s in ("valid", "validation") else s for s in split_of]))

    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    for name, split_of in arms:
        stats, cat_of = leak_categories(rows, fam_of, split_of)
        rep["arms"][name] = stats
        with open(f"{args.out}_leakcat_{name}.json", "w", encoding="utf-8") as fh:
            json.dump(cat_of, fh)
        print(f"  {name:16s} test {stats['n_test']:,}  leaked {stats['leaked_any_pct']}%  "
              f"(Croft-def {stats['leaked_croft_definition_pct']}%)")

    with open(f"{args.out}.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    write_md(f"{args.out}.md", rep)
    s, c = rep["size_distribution"], rep["label_consistency"]
    print(f"largest family {s['largest_family_size']:,} = {s['largest_family_share_pct']}% of corpus; "
          f"mixed-label families {c['families_mixed_label_pct']}%; "
          f"positives in mixed families {c['positives_in_mixed_families_pct']}%")
    print(f"wrote {args.out}.json / .md / _leakcat_*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
