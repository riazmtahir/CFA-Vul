#!/usr/bin/env python3
"""
precheck_h4.py -- feasibility check of the commit_hrfamily_disjoint split BEFORE any H4
run (RESEARCH_PLAN §13.7). CPU only. Recomputes everything from the corpus and the split
files; it does not read the builder's report.

Compared with the commit-disjoint split (the H4 baseline arm):
  * partition sizes and positive rates;
  * largest group (connected component of same-commit OR same high-recall family), % of corpus;
  * same-commit sharing: test functions with a train function of the same commit;
  * own twins: test functions whose same-commit, same-name, opposite-label version is in train;
  * high-recall theta=0.7 family leak: any, and same-label (Croft-def = same + both);
  * strict theta=0.7 family leak (5-token shingles, positional renaming);
  * mean function length (characters) per partition, and the length-only floor PR-AUC
    (logistic regression on log1p(characters, whitespace tokens) fit on train, scored on test;
    the same features and model as finetune_vd.length_floor).
FAILS with exit 3 if any of: a partition size differs from the commit-disjoint split by > 2%,
a positive rate differs by > 0.003, any leak or sharing above is > 0, or the largest group is
> 15% of the corpus. Exit 2 on a missing input. Exit 0 = H4 may be run (after .phase4_approved).

Writes <out>.json / <out>.md (default <disk>/h4_precheck).
Usage: python scripts/precheck_h4.py [--disk $CFA_DATA_DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

VD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VD)
from commit_leak_audit import func_name  # noqa: E402

PARTS = ("train", "valid", "test")
LIMITS = {"size_rel": 0.02, "pos_rate_abs": 0.003, "largest_group_pct": 15.0}


def load_rows(path):
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
            f = o.get("func")
            if not isinstance(f, str) or not f.strip():
                continue
            rows.append({"id": str(o.get("idx", n)), "label": 1 if o.get("target") == 1 else 0,
                         "commit": str(o.get("commit_id") or ""), "name": func_name(f),
                         "chars": len(f), "words": len(f.split())})
    return rows


def components(rows, fam):
    p = list(range(len(rows)))

    def find(x):
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x
    first = {}
    for i, r in enumerate(rows):
        keys = []
        if r["commit"]:
            keys.append(("c", r["commit"]))
        if fam.get(r["id"]) is not None:
            keys.append(("f", fam[r["id"]]))
        for k in keys:
            j = first.setdefault(k, i)
            a, b = find(i), find(j)
            if a != b:
                p[max(a, b)] = min(a, b)
    return Counter(find(i) for i in range(len(rows)))


def split_stats(rows, split, hr, strict):
    part = [split.get(r["id"]) for r in rows]
    idx = {p: [i for i, s in enumerate(part) if s == p] for p in PARTS}
    out = {"partitions": {}}
    for p in PARTS:
        ii = idx[p]
        out["partitions"][p] = {"functions": len(ii),
                                "pos_rate": sum(rows[i]["label"] for i in ii) / len(ii) if ii else None,
                                "mean_chars": float(np.mean([rows[i]["chars"] for i in ii])) if ii else None}
    tr, te = idx["train"], idx["test"]
    tr_commits = {rows[i]["commit"] for i in tr if rows[i]["commit"]}
    tr_twin = {(rows[i]["commit"], rows[i]["name"], rows[i]["label"]) for i in tr if rows[i]["commit"] and rows[i]["name"]}
    nte = len(te)
    share = sum(1 for i in te if rows[i]["commit"] in tr_commits)
    twin = sum(1 for i in te if rows[i]["commit"] and rows[i]["name"]
               and (rows[i]["commit"], rows[i]["name"], 1 - rows[i]["label"]) in tr_twin)

    def fam_leak(fam):
        labs = defaultdict(set)
        for i in tr:
            f = fam.get(rows[i]["id"])
            if f is not None:
                labs[f].add(rows[i]["label"])
        anyk = croft = 0
        for i in te:
            ls = labs.get(fam.get(rows[i]["id"]), set())
            if ls:
                anyk += 1
                croft += rows[i]["label"] in ls
        return {"any": anyk, "any_pct": 100 * anyk / nte, "croft_def": croft, "croft_def_pct": 100 * croft / nte}
    out["test_sharing_a_commit_with_train"] = share
    out["test_with_own_twin_in_train"] = twin
    out["hr_family_leak"] = fam_leak(hr)
    out["strict_family_leak"] = fam_leak(strict)
    X = lambda ii: np.log1p(np.array([[rows[i]["chars"], rows[i]["words"]] for i in ii], dtype=float))
    ytr = np.array([rows[i]["label"] for i in tr])
    yte = np.array([rows[i]["label"] for i in te])
    if len(set(ytr)) == 2 and 0 < yte.sum() < len(yte):
        clf = LogisticRegression(max_iter=1000).fit(X(tr), ytr)
        out["length_floor_pr_auc"] = float(average_precision_score(yte, clf.predict_proba(X(te))[:, 1]))
    else:
        out["length_floor_pr_auc"] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disk", default=os.environ.get("CFA_DATA_DIR", "data"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    D = a.disk
    paths = {"data": os.path.join(D, "data/diversevul/diversevul.jsonl"),
             "new": os.path.join(D, "diversevul_audit_splitfile_commithrfamily_seed42.json"),
             "ref": os.path.join(D, "diversevul_audit_splitfile_commitdisjoint_seed42.json"),
             "hr": os.path.join(D, "audit_highrecall/diversevul_hr_families_theta0.7.json"),
             "strict": os.path.join(D, "diversevul_audit_families_theta0.7.json")}
    for p in paths.values():
        if not os.path.exists(p):
            print(f"!!! missing input: {p}", file=sys.stderr)
            return 2
    rows = load_rows(paths["data"])
    ld = lambda k: {str(x): y for x, y in json.load(open(paths[k], encoding="utf-8")).items()}
    new, ref, hr, strict = ld("new"), ld("ref"), ld("hr"), ld("strict")
    new = {k: ("valid" if v == "validation" else v) for k, v in new.items()}
    ref = {k: ("valid" if v == "validation" else v) for k, v in ref.items()}
    uncovered = sum(1 for r in rows if r["id"] not in new)

    comp = components(rows, hr)
    sizes = sorted(comp.values(), reverse=True)
    S = {"commit_hrfamily_disjoint": split_stats(rows, new, hr, strict),
         "commit_disjoint": split_stats(rows, ref, hr, strict)}
    N, R = S["commit_hrfamily_disjoint"], S["commit_disjoint"]

    fails = []
    if uncovered:
        fails.append(f"{uncovered} functions not in the new split")
    for p in PARTS:
        a1, b1 = N["partitions"][p], R["partitions"][p]
        rel = abs(a1["functions"] - b1["functions"]) / b1["functions"]
        if rel > LIMITS["size_rel"]:
            fails.append(f"{p} size {a1['functions']:,} vs {b1['functions']:,} ({100 * rel:.2f}% > 2%)")
        if abs(a1["pos_rate"] - b1["pos_rate"]) > LIMITS["pos_rate_abs"]:
            fails.append(f"{p} positive rate {a1['pos_rate']:.4f} vs {b1['pos_rate']:.4f} (> 0.003)")
    for k, v in (("same-commit sharing", N["test_sharing_a_commit_with_train"]),
                 ("own twins", N["test_with_own_twin_in_train"]),
                 ("high-recall family leak (any)", N["hr_family_leak"]["any"]),
                 ("high-recall family leak (same-label)", N["hr_family_leak"]["croft_def"]),
                 ("strict family leak", N["strict_family_leak"]["any"])):
        if v:
            fails.append(f"{k}: {v} test functions (must be 0)")
    largest_pct = 100 * sizes[0] / len(rows)
    if largest_pct > LIMITS["largest_group_pct"]:
        fails.append(f"largest group {sizes[0]:,} functions = {largest_pct:.2f}% of corpus (> 15%)")

    rep = {"note": "H4 pre-check (RESEARCH_PLAN §13.7); written before any H4 run", "limits": LIMITS,
           "n_functions": len(rows), "groups": len(sizes), "largest_group": sizes[0],
           "largest_group_pct": largest_pct, "top10_group_sizes": sizes[:10], "splits": S,
           "fails": fails, "verdict": "PASS" if not fails else "FAIL"}
    out = a.out or os.path.join(D, "h4_precheck")
    json.dump(rep, open(out + ".json", "w", encoding="utf-8"), indent=2)
    f2 = lambda x: "—" if x is None else f"{x:.4f}"
    L = ["# H4 pre-check (commit_hrfamily_disjoint vs commit_disjoint)", "",
         f"Functions {len(rows):,}; groups {len(sizes):,}; largest group {sizes[0]:,} "
         f"({largest_pct:.2f}% of corpus); top 10 {sizes[:10]}", "",
         "| quantity | commit_hrfamily_disjoint | commit_disjoint |", "|---|---|---|"]
    for p in PARTS:
        L.append(f"| {p} functions / pos rate / mean chars | {N['partitions'][p]['functions']:,} / "
                 f"{f2(N['partitions'][p]['pos_rate'])} / {N['partitions'][p]['mean_chars']:.0f} | "
                 f"{R['partitions'][p]['functions']:,} / {f2(R['partitions'][p]['pos_rate'])} / "
                 f"{R['partitions'][p]['mean_chars']:.0f} |")
    for lab, k in (("test sharing a commit with train", "test_sharing_a_commit_with_train"),
                   ("test with own twin in train", "test_with_own_twin_in_train")):
        L.append(f"| {lab} | {N[k]:,} | {R[k]:,} |")
    for lab, k in (("high-recall family leak", "hr_family_leak"), ("strict family leak", "strict_family_leak")):
        L.append(f"| {lab}: any / same-label % | {N[k]['any_pct']:.2f} / {N[k]['croft_def_pct']:.2f} | "
                 f"{R[k]['any_pct']:.2f} / {R[k]['croft_def_pct']:.2f} |")
    L.append(f"| length-only floor PR-AUC (test) | {f2(N['length_floor_pr_auc'])} | {f2(R['length_floor_pr_auc'])} |")
    L += ["", f"**{rep['verdict']}**" + ("" if not fails else ": " + "; ".join(fails))]
    open(out + ".md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0 if not fails else 3


if __name__ == "__main__":
    sys.exit(main())
