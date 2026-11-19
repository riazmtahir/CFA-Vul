#!/usr/bin/env python3
"""
precheck_h4b.py -- the §13.8 feasibility gate for H4b (RESEARCH_PLAN, 21 Oct 2026; size control
amended the same day, before any pre-check or run). CPU only. Recomputes everything from the
split files, the train-hit list and the corpus.

Gate (exit 3 on fail):
  * removal <= 40% of the commit_disjoint train set;
  * >= 1,000 positives left in train;
  * the size control holds no decon-listed function (registered pool, second §13.8 amendment);
  * borrowing <= 5% of the removal, where borrowed = sum over label x length-decile strata of
    max(0, decon removals - control removals) (deciles of character length on the
    commit_disjoint train set), recounted here independently of make_decon_splits.py;
  * integrity: valid and test of both arms identical to commit_disjoint; size control removes
    exactly the same number of functions and of positives; no listed train function left in
    commit_decon train; every removed function was a commit_disjoint train function.
Descriptive (no gate), for removed-by-decon / removed-by-control / kept (= commit_disjoint train
minus the decon removal): mean and median length (characters), top-10 projects and top-10 CWEs by
share, train positive rate per arm, and the direct leak rates from direct_clone_edges.py.
Exit 2 on a missing input. Writes <disk>/h4b_precheck.{json,md}.
Usage: python scripts/precheck_h4b.py [--disk $CFA_DATA_DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from collections import Counter

import numpy as np

LIMITS = {"max_removed_pct_of_train": 40.0, "min_train_positives_left": 1000, "max_borrowed_pct_of_removal": 5.0}


def load_corpus(path):
    info = {}
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
            if isinstance(f, str) and f.strip():
                cwe = o.get("cwe") or []
                cwe = [str(c) for c in (cwe if isinstance(cwe, list) else [cwe])] or ["(none)"]
                info[str(o.get("idx", n))] = {"label": 1 if o.get("target") == 1 else 0, "len": len(f),
                                              "project": str(o.get("project") or "(none)"), "cwe": cwe}
    return info


def composition(ids, info):
    if not ids:
        return {"n": 0}
    ln = [info[i]["len"] for i in ids]
    proj = Counter(info[i]["project"] for i in ids)
    cwe = Counter(c for i in ids for c in info[i]["cwe"])
    n = len(ids)
    return {"n": n, "mean_chars": st.mean(ln), "median_chars": st.median(ln),
            "pos_rate": sum(info[i]["label"] for i in ids) / n,
            "top10_projects": [[p, round(100 * c / n, 2)] for p, c in proj.most_common(10)],
            "top10_cwes": [[w, round(100 * c / n, 2)] for w, c in cwe.most_common(10)]}


def evaluate(ref, decon, size, hits, info):
    fails = []
    part = lambda sp, p: {i for i, v in sp.items() if v == p}
    tr = part(ref, "train")
    lab = lambda i: info[i]["label"]
    for name, sp in (("commit_decon", decon), ("commit_sizectrl", size)):
        for p in ("valid", "test"):
            if part(sp, p) != part(ref, p):
                fails.append(f"{name}: {p} set differs from commit_disjoint")
        if not part(sp, "train") <= tr:
            fails.append(f"{name}: train holds functions outside commit_disjoint train")
    rem_d, rem_s = tr - part(decon, "train"), tr - part(size, "train")
    if rem_d != hits:
        fails.append(f"commit_decon removed {len(rem_d):,} functions, the hit list has {len(hits):,}")
    if not hits <= tr:
        fails.append("hit list holds functions outside commit_disjoint train")
    if rem_s & rem_d:
        fails.append(f"size control removes {len(rem_s & rem_d):,} decon-listed functions (pool must exclude them)")
    if len(rem_s) != len(rem_d) or sum(lab(i) for i in rem_s) != sum(lab(i) for i in rem_d):
        fails.append("size control does not match the removal (functions or positives)")
    # stratum recount: label x length decile on commit_disjoint train
    trl = sorted(tr)
    edges = np.quantile(np.array([info[i]["len"] for i in trl], dtype=float), np.linspace(0.1, 0.9, 9)) \
        if trl else np.zeros(9)
    stratum = lambda i: (lab(i), int(np.searchsorted(edges, info[i]["len"], side="right")))
    cd, cs = Counter(stratum(i) for i in rem_d), Counter(stratum(i) for i in rem_s)
    # control removals in excess of the decon count, per stratum = functions borrowed into it
    borrowed = sum(max(0, cs[k] - cd.get(k, 0)) for k in cs)
    bpct = 100 * borrowed / len(rem_d) if rem_d else 0.0
    n_tr = len(tr)
    pct = 100 * len(rem_d) / n_tr if n_tr else 0.0
    pos_left = sum(lab(i) for i in tr - rem_d)
    if pct > LIMITS["max_removed_pct_of_train"]:
        fails.append(f"removal {pct:.2f}% of commit_disjoint train (> 40%)")
    if pos_left < LIMITS["min_train_positives_left"]:
        fails.append(f"{pos_left:,} positives left in train (< 1,000)")
    if bpct > LIMITS["max_borrowed_pct_of_removal"]:
        fails.append(f"borrowing {borrowed:,} functions = {bpct:.2f}% of the removal (> 5%)")
    desc = {"commit_disjoint_train": n_tr, "removed": len(rem_d), "removed_pct_of_train": pct,
            "removed_pos": sum(lab(i) for i in rem_d), "train_pos_left": pos_left,
            "borrowed": borrowed, "borrowed_pct_of_removal": bpct,
            "strata": {f"label{k[0]}_decile{k[1]}": {"decon": cd[k], "control": cs.get(k, 0)} for k in sorted(cd)},
            "train_pos_rate": {"commit_disjoint": sum(lab(i) for i in tr) / n_tr if n_tr else None,
                               "commit_decon": pos_left / (n_tr - len(rem_d)) if n_tr > len(rem_d) else None,
                               "commit_sizectrl": sum(lab(i) for i in tr - rem_s) / (n_tr - len(rem_s))
                               if n_tr > len(rem_s) else None},
            "composition": {"removed_by_decon": composition(sorted(rem_d), info),
                            "removed_by_control": composition(sorted(rem_s), info),
                            "kept": composition(sorted(tr - rem_d), info)},
            "sizectrl_overlap_with_decon_removal": len(rem_d & rem_s)}
    return fails, desc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disk", default=os.environ.get("CFA_DATA_DIR", "data"))
    ap.add_argument("--edges-prefix", default=None, help="default <disk>/direct_edges/diversevul_hr_direct")
    a = ap.parse_args()
    D = a.disk
    ep = a.edges_prefix or os.path.join(D, "direct_edges", "diversevul_hr_direct")
    P = {"data": os.path.join(D, "data/diversevul/diversevul.jsonl"),
         "ref": os.path.join(D, "diversevul_audit_splitfile_commitdisjoint_seed42.json"),
         "decon": os.path.join(D, "diversevul_audit_splitfile_commitdecon_seed42.json"),
         "size": os.path.join(D, "diversevul_audit_splitfile_commitsizectrl_seed42.json"),
         "hits": f"{ep}_commit_disjoint_train_hits.json", "summary": f"{ep}_summary.json"}
    for p in P.values():
        if not os.path.exists(p):
            print(f"!!! missing input: {p}", file=sys.stderr)
            return 2
    ld = lambda k: {str(x): ("valid" if y == "validation" else y) for x, y in
                    json.load(open(P[k], encoding="utf-8")).items()}
    ref, decon, size = ld("ref"), ld("decon"), ld("size")
    hits = {str(x) for x in json.load(open(P["hits"], encoding="utf-8"))}
    info = load_corpus(P["data"])
    fails, desc = evaluate(ref, decon, size, hits, info)
    summ = json.load(open(P["summary"], encoding="utf-8"))
    rep = {"note": "H4b pre-check (RESEARCH_PLAN §13.8, size control amended 21 Oct 2026)", "limits": LIMITS,
           **desc,
           "direct_leak_by_split": {s: {k: v for k, v in r.items() if k.startswith("direct_")}
                                    for s, r in summ["splits"].items()},
           "fails": fails, "verdict": "PASS" if not fails else "FAIL"}
    json.dump(rep, open(os.path.join(D, "h4b_precheck.json"), "w", encoding="utf-8"), indent=2)
    f = lambda x: "—" if x is None else f"{x:.4f}"
    L = ["# H4b pre-check (commit_decon / commit_sizectrl vs commit_disjoint)", "",
         f"commit_disjoint train {desc['commit_disjoint_train']:,}; removed {desc['removed']:,} "
         f"({desc['removed_pct_of_train']:.2f}%), of which {desc['removed_pos']:,} positives; "
         f"positives left {desc['train_pos_left']:,}; borrowed {desc['borrowed']:,} "
         f"({desc['borrowed_pct_of_removal']:.2f}% of the removal)", "",
         f"train positive rate: commit_disjoint {f(desc['train_pos_rate']['commit_disjoint'])}, decon "
         f"{f(desc['train_pos_rate']['commit_decon'])}, sizectrl {f(desc['train_pos_rate']['commit_sizectrl'])}", "",
         "| set | n | mean chars | median chars | pos rate | top-10 projects (% of set) | top-10 CWEs (% of set) |",
         "|---|---|---|---|---|---|---|"]
    for name, c in desc["composition"].items():
        if not c.get("n"):
            L.append(f"| {name} | 0 | — | — | — | — | — |")
            continue
        L.append(f"| {name} | {c['n']:,} | {c['mean_chars']:.0f} | {c['median_chars']:.0f} | {c['pos_rate']:.4f} | "
                 + ", ".join(f"{p} {s}" for p, s in c["top10_projects"]) + " | "
                 + ", ".join(f"{w} {s}" for w, s in c["top10_cwes"]) + " |")
    L += ["", "| split | direct leak % (test) | positives: direct leak % |", "|---|---|---|"]
    for s, r in summ["splits"].items():
        L.append(f"| {s} | {r['direct_leaked_test_pct']} | {r['direct_leaked_test_pos_pct']} |")
    L += ["", f"**{rep['verdict']}**" + ("" if not fails else ": " + "; ".join(fails))]
    open(os.path.join(D, "h4b_precheck.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0 if not fails else 3


if __name__ == "__main__":
    sys.exit(main())
