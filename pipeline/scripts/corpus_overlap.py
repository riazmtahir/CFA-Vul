#!/usr/bin/env python3
"""
corpus_overlap.py -- overlap between PrimeVul and DiverseVul on our server copies
(action A3, 18 Oct 2026). DESCRIPTIVE. Li et al. 2026 (Figure 1 p3) report that the
two corpora share 69.8% of samples; this measures it on the files we audited.

Three match keys, each computed per function:
  raw      md5 of the exact function text
  norm     md5 of the normalize_tokens stream (clone_family_audit.py: comments and
           whitespace ignored, identifiers renamed in order of use, literals collapsed)
  commit   (commit_id, function name), name = identifier before the first "(" as in
           commit_leak_audit.py; functions without a commit id or a name get no key
For each key: functions of each corpus with a match in the other (count, % of that
corpus), label agreement among matched PrimeVul functions, and the share of PrimeVul
TEST functions that appear in DiverseVul TRAIN under the published (mirror) split and
under every DiverseVul split file present (random control, family-, project-,
commit-disjoint).

Inputs: <disk>/data/primevul/primevul_all.jsonl (primevul_tools.py prep; field split),
        <disk>/data/diversevul/diversevul.jsonl (field split = published mirror split).
Writes <disk>/corpus_overlap/primevul_diversevul_overlap.{json,md}.
Usage: python scripts/corpus_overlap.py [--disk $CFA_DATA_DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

VD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VD)
from clone_family_audit import normalize_tokens  # noqa: E402
from commit_leak_audit import func_name  # noqa: E402

KEYS = ("raw", "norm", "commit")
DV_SPLITS = {"random_control": "diversevul_audit_splitfile_randomctrl_theta0.7.json",
             "family_disjoint": "diversevul_audit_splitfile_theta0.7.json",
             "project_disjoint": "diversevul_audit_splitfile_projectdisjoint_seed42.json",
             "commit_disjoint": "diversevul_audit_splitfile_commitdisjoint_seed42.json"}


def keys_of(o):
    f = o["func"]
    nm = func_name(f)
    c = str(o.get("commit_id") or "")
    return {"raw": hashlib.md5(f.encode("utf-8", "replace")).digest(),
            "norm": hashlib.md5(" ".join(normalize_tokens(f)).encode("utf-8")).digest(),
            "commit": (c, nm) if c and nm else None}


def read(path, id_field, report_every=50000):
    """id -> (label, split, {key: value}); same empty-function filter as the audit tools."""
    out = {}
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
            out[str(o.get(id_field, n))] = (1 if o.get("target") == 1 else 0, str(o.get("split", "")), keys_of(o))
            if report_every and len(out) % report_every == 0:
                print(f"    {os.path.basename(path)}: {len(out):,}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disk", default=os.environ.get("CFA_DATA_DIR", "data"))
    ap.add_argument("--out", default=None, help="default <disk>/corpus_overlap/primevul_diversevul_overlap")
    args = ap.parse_args()
    D = args.disk
    pv_path = os.path.join(D, "data/primevul/primevul_all.jsonl")
    dv_path = os.path.join(D, "data/diversevul/diversevul.jsonl")
    for p in (pv_path, dv_path):
        if not os.path.exists(p):
            print(f"!!! missing input: {p}", file=sys.stderr)
            return 2
    print("reading PrimeVul ...", flush=True)
    pv = read(pv_path, "idx")
    print("reading DiverseVul ...", flush=True)
    dv = read(dv_path, "idx")

    dv_splits = {"published": {i: r[1] for i, r in dv.items()}}
    for arm, f in DV_SPLITS.items():
        p = os.path.join(D, f)
        if os.path.exists(p):
            dv_splits[arm] = {str(k): v for k, v in json.load(open(p)).items()}

    rep = {"note": "DESCRIPTIVE (action A3)", "n": {"primevul": len(pv), "diversevul": len(dv)},
           "primevul_test_n": sum(r[1] == "test" for r in pv.values()), "keys": {}}
    for k in KEYS:
        dv_all = defaultdict(set)                       # key value -> DiverseVul ids
        for i, r in dv.items():
            if r[2][k] is not None:
                dv_all[r[2][k]].add(i)
        pv_vals = {r[2][k] for r in pv.values() if r[2][k] is not None}
        pv_hit = [i for i, r in pv.items() if r[2][k] is not None and r[2][k] in dv_all]
        dv_hit = sum(1 for r in dv.values() if r[2][k] is not None and r[2][k] in pv_vals)
        # label agreement: a matched PrimeVul function agrees if some matching DiverseVul function has its label
        agree = sum(1 for i in pv_hit if any(dv[j][0] == pv[i][0] for j in dv_all[pv[i][2][k]]))
        test_ids = [i for i, r in pv.items() if r[1] == "test"]
        in_train = {}
        for arm, split in dv_splits.items():
            tr_vals = {dv[j][2][k] for j, s in split.items() if s == "train" and j in dv and dv[j][2][k] is not None}
            hits = sum(1 for i in test_ids if pv[i][2][k] is not None and pv[i][2][k] in tr_vals)
            in_train[arm] = {"count": hits, "pct_of_primevul_test": round(100 * hits / len(test_ids), 2)
                             if test_ids else None}
        rep["keys"][k] = {
            "primevul_with_key": sum(r[2][k] is not None for r in pv.values()),
            "diversevul_with_key": sum(r[2][k] is not None for r in dv.values()),
            "primevul_matched": len(pv_hit), "primevul_matched_pct": round(100 * len(pv_hit) / len(pv), 2),
            "diversevul_matched": dv_hit, "diversevul_matched_pct": round(100 * dv_hit / len(dv), 2),
            "primevul_matched_same_label_pct": round(100 * agree / len(pv_hit), 2) if pv_hit else None,
            "primevul_test_in_diversevul_train": in_train}
        print(f"  {k}: PrimeVul matched {len(pv_hit):,} ({rep['keys'][k]['primevul_matched_pct']}%), "
              f"DiverseVul matched {dv_hit:,} ({rep['keys'][k]['diversevul_matched_pct']}%)", flush=True)

    L = ["# PrimeVul ∩ DiverseVul on our server copies (descriptive, A3)", "",
         f"PrimeVul {len(pv):,} functions (test {rep['primevul_test_n']:,}); DiverseVul {len(dv):,}. "
         "raw = exact text; norm = `normalize_tokens` stream; commit = (commit_id, function name).", "",
         "| key | PrimeVul matched | % of PrimeVul | DiverseVul matched | % of DiverseVul | same label (PrimeVul matched) |",
         "|---|---|---|---|---|---|"]
    for k, r in rep["keys"].items():
        L.append(f"| {k} | {r['primevul_matched']:,} | {r['primevul_matched_pct']} | {r['diversevul_matched']:,} | "
                 f"{r['diversevul_matched_pct']} | {r['primevul_matched_same_label_pct']} |")
    L += ["", "PrimeVul test functions found in DiverseVul train (count, % of PrimeVul test):", "",
          "| key | " + " | ".join(dv_splits) + " |", "|---|" + "---|" * len(dv_splits)]
    for k, r in rep["keys"].items():
        L.append(f"| {k} | " + " | ".join(f"{v['count']:,} ({v['pct_of_primevul_test']}%)"
                                         for v in r["primevul_test_in_diversevul_train"].values()) + " |")
    out = args.out or os.path.join(D, "corpus_overlap", "primevul_diversevul_overlap")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rep, open(out + ".json", "w", encoding="utf-8"), indent=2)
    open(out + ".md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
