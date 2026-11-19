#!/usr/bin/env python3
"""rq1_numbers.py -- recompute and cross-check every RQ1 number from the result files.

Reads
  results/rq0_calibration/rq0_theta_calibration.json   BCB capture rates (calibrate_theta_bcb.py,
                                                        re-run 12 Oct 2026; no longer cited)
  results/rq0_calibration_ours/rq0_theta_calibration_ours.{json,md}   BCB capture with the
                                                        audit's own scorer (cited in rq1.tex)
  results/from_backup/{devign,diversevul}_audit_report.json          theta 0.7 audit
  results/from_backup_1012/theta_sens/{devign,diversevul}_audit_report.json   theta 0.8 / 0.9
  results/from_backup_1012/primevul_audit_report.json                 theta 0.7 / 0.8 / 0.9
  results/from_backup_1012/family_label/{devign,diversevul,primevul}_theta0.7.json  Phase 1
  results/from_backup_1012/family_label/primevul_pairs.json          vulnerable/patch pairs
  results/audit_7oct/primevul_audit.log                 PrimeVul prep report (counts, hashes, projects)
  results/audit_7oct/{primevul_audit_report.md, theta_sens/*_audit_report.md}  generated reports

Checks: percentages recomputed from counts; report JSON vs generated .md; family-label JSON vs
audit report (same families); every value against the project log. Any mismatch -> exit 1.
Output: analysis/scripts/out/rq1_numbers.json (read by rq1_table_figure.py).
Usage (repository root): python analysis/scripts/rq1_numbers.py
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
B, B12, A7 = (os.path.join(RES, d) for d in ("from_backup", "from_backup_1012", "audit_7oct"))
checks, fails = [], []


def chk(name, got, want, nd, src):
    ok = abs(got - want) <= 0.5 * 10 ** (-nd) + 1e-9
    checks.append({"name": name, "got": got, "want": want, "decimals": nd, "src": src, "ok": ok})
    if not ok:
        fails.append(checks[-1])


def load(*p):
    return json.load(open(os.path.join(*p), encoding="utf-8"))


def md_rows(path):
    with open(path, encoding="utf-8") as fh:
        return [[c.strip() for c in l.strip().strip("|").split("|")] for l in fh
                if l.startswith("|") and not l.startswith("|---")]


# ------------------------------------------------------------------ calibration (RQ0)
cal = load(RES, "rq0_calibration", "rq0_theta_calibration.json")
chk("BCB mapping check: Type-1 median Jaccard", cal["mapping_check_type1_median"], 1.0, 4, "calibration")
capt = {c["category"]: c["capture"] for c in cal["categories"]}
ncat = {c["category"]: c["n"] for c in cal["categories"]}
ST3, MT3, WK, NC = "Strong Type-3 (ST3)", "Moderate Type-3 (MT3)", "Weak Type-3 / Type-4", "non-clone"
# prompt / clone_family_audit.py docstring values (in %, 1 decimal)
for cat, th, want in ((ST3, "0.70", 99.0), (MT3, "0.70", 67.2), (NC, "0.70", 3.2), (ST3, "0.80", 90.1),
                      (MT3, "0.80", 24.3), (NC, "0.80", 0.3), (ST3, "0.90", 49.3), (MT3, "0.90", 2.1),
                      (WK, "0.70", 6.6)):
    # the JSON stores capture rounded to 4 decimals; MT3@0.7 = 0.6715 is printed 67.2% by the script
    chk(f"BCB {cat} @ {th} (%)", round(100 * capt[cat][th] + 1e-9, 1), want, 1, "rq0_theta_calibration.json")
for r in md_rows(os.path.join(RES, "rq0_calibration", "rq0_theta_calibration.md")):
    if r[0] in capt and len(r) == 8:
        for th, cell in zip(("0.60", "0.70", "0.80", "0.90"), r[4:8]):
            chk(f"BCB md == json {r[0]} {th}", float(cell.rstrip("%")), round(100 * capt[r[0]][th] + 1e-9, 1), 1,
                "rq0_theta_calibration.md")
# The TouSimCC values above are kept as a check of results/rq0_calibration/ only; since 18 Oct 2026
# (action A1) rq1.tex cites the audit's own scorer, checked here.
own = load(RES, "rq0_calibration_ours", "rq0_theta_calibration_ours.json")
chk("BCB (own scorer) Type-1 median Jaccard", own["mapping_check_type1_median"], 1.0, 4, "calibration_ours")
chk("BCB (own scorer) shingle k", own["shingle_k"], 5, 0, "calibration_ours")
ocap = {c["category"].strip(): c["capture"] for c in own["categories"]}
VST, ST3P = "of which VST3 (BCB sim >= 0.9)", "of which ST3 proper (BCB sim 0.7-0.9)"
for cat, vals in ((ST3, (30.0, 22.5, 15.6)), (VST, (57.3, 50.1, 43.9)), (ST3P, (22.6, 15.0, 7.9)),
                  (MT3, (0.3, 0.0, 0.0)), (WK, (0.0, 0.0, 0.0)), (NC, (0.0, 0.0, 0.0))):
    for th, want in zip(("0.70", "0.80", "0.90"), vals):
        chk(f"BCB own scorer {cat} @ {th} (%)", round(100 * ocap[cat][th] + 1e-9, 1), want, 1,
            "rq0_theta_calibration_ours.json")
for r in md_rows(os.path.join(RES, "rq0_calibration_ours", "rq0_theta_calibration_ours.md")):
    if r[0].strip() in ocap and len(r) == 8 and r[4].endswith("%"):
        for th, cell in zip(("0.60", "0.70", "0.80", "0.90"), r[4:8]):
            chk(f"BCB own md == json {r[0].strip()} {th}", float(cell.rstrip("%")),
                round(100 * ocap[r[0].strip()][th] + 1e-9, 1), 1, "rq0_theta_calibration_ours.md")
calib = {"capture": capt, "n": ncat, "capture_own_scorer": ocap}

# ------------------------------------------------------------------ audit reports per theta
REP = {"devign": {0.7: (B, "devign_audit_report.json"), 0.8: (B12, "theta_sens", "devign_audit_report.json"),
                  0.9: (B12, "theta_sens", "devign_audit_report.json")},
       "diversevul": {0.7: (B, "diversevul_audit_report.json"),
                      0.8: (B12, "theta_sens", "diversevul_audit_report.json"),
                      0.9: (B12, "theta_sens", "diversevul_audit_report.json")},
       "primevul": {t: (B12, "primevul_audit_report.json") for t in (0.7, 0.8, 0.9)}}
audit, inputs = {}, {}
for corpus, by in REP.items():
    audit[corpus] = {}
    for th, parts in by.items():
        rep = load(*parts)
        t = [x for x in rep["thresholds"] if abs(x["theta"] - th) < 1e-9][0]
        rc = t["random_control_split"]
        fd = t["family_disjoint_split"]
        src = os.path.relpath(os.path.join(*parts), ROOT).replace("\\", "/")
        chk(f"{corpus} {th} leaked% = leaked/test", 100 * t["leaked_test_functions"] / t["n_test_functions"],
            t["leaked_test_pct"], 2, src)
        chk(f"{corpus} {th} redundancy% = redundant/n", 100 * t["type12_redundant_functions"] / t["n_functions"],
            t["type12_redundancy_pct"], 2, src)
        chk(f"{corpus} {th} RC leaked% = leaked/test", 100 * rc["leaked_test_functions"] / rc["test"],
            rc["leaked_test_pct"], 2, src)
        chk(f"{corpus} {th} family-disjoint leak == 0", fd["leaked_test_pct"], 0.0, 2, src)
        audit[corpus][str(th)] = {
            "src": src, "n_functions": t["n_functions"], "n_test": t["n_test_functions"],
            "type12_groups": t["type12_exact_groups"], "type12_redundant": t["type12_redundant_functions"],
            "type12_pct": t["type12_redundancy_pct"], "families_total": t["n_families_total"],
            "families_gt1": t["n_families_size_gt1"], "largest_family": t["largest_family"],
            "span_train_test": t["families_spanning_train_and_test"],
            "leaked_test": t["leaked_test_functions"], "leaked_pct": t["leaked_test_pct"],
            "contradictory": t["families_with_contradictory_labels"], "cross_project": t["cross_project_families"],
            "rc_test": rc["test"], "rc_leaked_pct": rc["leaked_test_pct"], "rc_leaked": rc["leaked_test_functions"]}
    inputs[corpus] = load(*by[0.7])["input_counts"]
# the project log theta-sensitivity table and RQ1 numbers
THETA = {"devign": ((15.34, 15.01), (13.47, 13.04), (10.91, 10.99)),
         "diversevul": ((29.61, 29.39), (26.62, 26.54), (23.28, 23.16)),
         "primevul": ((15.70, 30.05), (14.50, 27.86), (13.04, 25.50))}
for corpus, rows in THETA.items():
    for th, (pub, rc) in zip(("0.7", "0.8", "0.9"), rows):
        chk(f"{corpus} {th} published leak vs the project log", audit[corpus][th]["leaked_pct"], pub, 2, "the project log theta table")
        chk(f"{corpus} {th} random-control leak vs the project log", audit[corpus][th]["rc_leaked_pct"], rc, 2,
            "the project log theta table")
for corpus, (fam, span, contra, xp) in {"devign": (2031, 384, 1001, 9), "diversevul": (37954, 8095, 7457, 6064),
                                        "primevul": (28331, 2464, 2935, 5161)}.items():
    a = audit[corpus]["0.7"]
    for k, w in (("families_gt1", fam), ("span_train_test", span), ("contradictory", contra), ("cross_project", xp)):
        chk(f"{corpus} {k} vs the project log", a[k], w, 0, a["src"])
for th, (fam, span, contra, xp) in {"0.8": (26823, 2328, 2578, 4851), "0.9": (24719, 2097, 2018, 4480)}.items():
    a = audit["primevul"][th]
    for k, w in (("families_gt1", fam), ("span_train_test", span), ("contradictory", contra), ("cross_project", xp)):
        chk(f"primevul {th} {k} vs the project log", a[k], w, 0, a["src"])
# generated md reports agree (theta table rows)
for corpus in ("devign", "diversevul"):
    for r in md_rows(os.path.join(A7, "theta_sens", f"{corpus}_audit_report.md")):
        if r and r[0] in ("0.8", "0.9") and len(r) == 7 and r[1].isdigit():
            chk(f"{corpus} theta_sens md leaked% {r[0]}", float(r[4]), audit[corpus][r[0]]["leaked_pct"], 2,
                "audit_7oct theta_sens md")
for r in md_rows(os.path.join(A7, "primevul_audit_report.md")):
    if r and r[0] in ("0.7", "0.8", "0.9") and len(r) == 7 and r[1].isdigit():
        chk(f"primevul md leaked% {r[0]}", float(r[4]), audit["primevul"][r[0]]["leaked_pct"], 2,
            "primevul_audit_report.md")

# ------------------------------------------------------------------ Phase 1 family-label audit
fl = {}
CL = {"devign": dict(gt1_fns_pct=18.23, largest=36, largest_share=0.13, mixed_pct=49.29, mixed_fns_pct=8.53,
                     pos_mixed_pct=9.21, xp_pct=0.44, median=2, p90=3, p99=9),
      "diversevul": dict(gt1_fns_pct=33.05, largest=353, largest_share=0.11, mixed_pct=19.65, mixed_fns_pct=6.37,
                         pos_mixed_pct=44.56, xp_pct=15.98, median=2, p90=4, p99=15),
      "primevul": dict(gt1_fns_pct=33.97, largest=409, largest_share=0.17, mixed_pct=10.36, mixed_fns_pct=3.25,
                       pos_mixed_pct=47.22, xp_pct=18.22, median=2, p90=4, p99=15)}
for corpus in ("devign", "diversevul", "primevul"):
    d = load(B12, "family_label", f"{corpus}_theta0.7.json")
    sz, lc, xp = d["size_distribution"], d["label_consistency"], d["cross_project"]
    src = f"results/from_backup_1012/family_label/{corpus}_theta0.7.json"
    rec = {"gt1_fns_pct": 100 * sz["functions_in_families_gt1"] / sz["n_functions"],
           "largest": sz["largest_family_size"],
           "largest_share": 100 * sz["largest_family_size"] / sz["n_functions"],
           "mixed_pct": 100 * lc["families_mixed_label"] / lc["families_gt1"],
           "mixed_fns_pct": 100 * lc["functions_in_mixed_families"] / sz["n_functions"],
           "pos_mixed_pct": 100 * lc["positives_in_mixed_families"] / lc["positives_total"],
           "xp_pct": 100 * xp["cross_project_families"] / xp["families_gt1"],
           "median": sz["size_gt1_median"], "p90": sz["size_gt1_p90"], "p99": sz["size_gt1_p99"]}
    for k, w in CL[corpus].items():
        chk(f"{corpus} {k} recomputed vs the project log", rec[k], w, 2, src)
    # same families as the audit report
    a = audit[corpus]["0.7"]
    chk(f"{corpus} family-label families>1 == audit", lc["families_gt1"], a["families_gt1"], 0, src)
    chk(f"{corpus} family-label mixed == audit contradictory", lc["families_mixed_label"], a["contradictory"], 0, src)
    chk(f"{corpus} family-label largest == audit", sz["largest_family_size"], a["largest_family"], 0, src)
    chk(f"{corpus} family-label published leak == audit", d["arms"]["published"]["leaked_any_pct"], a["leaked_pct"], 2, src)
    arms = {}
    for arm, v in d["arms"].items():
        cats = v["categories"]
        n_pos = v["n_test_positive"]
        n_neg = v["n_test"] - n_pos
        chk(f"{corpus} {arm} category n sum", sum(c["n"] for c in cats.values()), v["n_test"], 0, src)
        arms[arm] = {"n_test": v["n_test"], "n_pos": n_pos, "leaked_pct": v["leaked_any_pct"],
                     "croft_pct": v["leaked_croft_definition_pct"],
                     "cat": {k: {"pct_of_test": 100 * c["n"] / v["n_test"], "pos_rate": c["pos_rate"],
                                 "share_of_pos": 100 * c["positives"] / n_pos if n_pos else None,
                                 "share_of_neg": 100 * c["negatives"] / n_neg if n_neg else None,
                                 "positives": c["positives"], "negatives": c["negatives"], "n": c["n"]}
                             for k, c in cats.items()},
                     "leaked_pos": sum(cats[k]["positives"] for k in ("leaked_same", "leaked_opposite", "leaked_both"))}
    rec["arms"] = arms
    rec["families_gt1"] = lc["families_gt1"]
    rec["families_mixed"] = lc["families_mixed_label"]
    rec["positives_total"] = lc["positives_total"]
    rec["n_functions"] = sz["n_functions"]
    fl[corpus] = rec
# Phase 1 readings in the project log
dv = fl["diversevul"]["arms"]["random_control"]
for k, (pct, pr) in {"clean": (70.61, 0.046), "leaked_same": (24.10, 0.015), "leaked_opposite": (3.47, 0.508),
                     "leaked_both": (1.81, 0.199)}.items():
    chk(f"DiverseVul RC {k} % of test", dv["cat"][k]["pct_of_test"], pct, 2, "diversevul_theta0.7.json")
    chk(f"DiverseVul RC {k} pos rate", dv["cat"][k]["pos_rate"], pr, 3, "diversevul_theta0.7.json")
chk("DiverseVul RC Croft-def leak", dv["croft_pct"], 25.92, 2, "diversevul_theta0.7.json")
for k, w in {"clean": 57, "leaked_opposite": 31, "leaked_same": 6, "leaked_both": 6}.items():
    chk(f"DiverseVul RC share of positives {k} (the project log approx %)", dv["cat"][k]["share_of_pos"], w, 0,
        "diversevul_theta0.7.json")
chk("DiverseVul RC share of negatives with same-label mate (the project log ~25%)", dv["cat"]["leaked_same"]["share_of_neg"],
    25, 0, "diversevul_theta0.7.json")
dvr = [fl["devign"]["arms"]["random_control"]["cat"][k]["pos_rate"] for k in fl["devign"]["arms"]["random_control"]["cat"]]
chk("Devign RC category pos rates min (the project log 0.40)", min(dvr), 0.40, 2, "devign_theta0.7.json")
chk("Devign RC category pos rates max (the project log 0.51)", max(dvr), 0.51, 2, "devign_theta0.7.json")
chk("Devign RC Croft-def leak", fl["devign"]["arms"]["random_control"]["croft_pct"], 10.08, 2, "devign_theta0.7.json")
pv = fl["primevul"]["arms"]["published"]
chk("PrimeVul published leaked positives (the project log ~78)", pv["leaked_pos"], 78, 0, "primevul_theta0.7.json")
chk("PrimeVul published test positives", pv["n_pos"], 695, 0, "primevul_theta0.7.json")
for k, w in {"clean": 84.30, "leaked_same": 15.09, "leaked_opposite": 0.21, "leaked_both": 0.40}.items():
    chk(f"PrimeVul published {k} % of test", pv["cat"][k]["pct_of_test"], w, 2, "primevul_theta0.7.json")

# ------------------------------------------------------------------ PrimeVul prep report + pairs
log = open(os.path.join(A7, "primevul_audit.log"), encoding="utf-8").read()
m = re.search(r"merge PrimeVul splits.*?\n(\{.*?\n\})\n", log, re.S)
prep = json.loads(m.group(1))
chk("PrimeVul matches paper", int(prep["matches_paper"]), 1, 0, "primevul_audit.log")
chk("PrimeVul total", prep["total"], 235768, 0, "primevul_audit.log")
chk("PrimeVul positives", prep["total_positive"], 6968, 0, "primevul_audit.log")
chk("PrimeVul hashes shared across splits", len(prep["hashes_shared_across_splits"]), 0, 0, "primevul_audit.log")
chk("PrimeVul test projects", prep["projects"]["test"], 222, 0, "primevul_audit.log")
chk("PrimeVul projects in train and test", prep["projects_in_train_and_test"], 149, 0, "primevul_audit.log")
for s, n in prep["counts"].items():
    chk(f"PrimeVul {s} count == audit input", n, inputs["primevul"][s], 0, "primevul_audit.log vs report")
pp = load(B12, "family_label", "primevul_pairs.json")
pairs = {}
for s, v in pp["splits"].items():
    pairs[s] = {}
    for th, r in v["theta"].items():
        pct = 100 * r["patch_in_same_family"] / r["vulnerable_with_a_same_commit_patch"]
        chk(f"PrimeVul pairs {s} {th} recomputed", pct, r["pct"], 2, "primevul_pairs.json")
        pairs[s][th] = {"pct": pct, "n": r["vulnerable_with_a_same_commit_patch"], "same": r["patch_in_same_family"]}
for th, w in (("0.7", 47.0), ("0.8", 40.8), ("0.9", 30.9)):
    chk(f"PrimeVul train pairs {th} vs the project log", pairs["train"][th]["pct"], w, 1, "primevul_pairs.json")
chk("PrimeVul valid/test pairs 0.7 min (the project log 44%)", min(pairs["valid"]["0.7"]["pct"], pairs["test"]["0.7"]["pct"]),
    44, 0, "primevul_pairs.json")
chk("PrimeVul valid/test pairs 0.7 max (the project log 48%)", max(pairs["valid"]["0.7"]["pct"], pairs["test"]["0.7"]["pct"]),
    48, 0, "primevul_pairs.json")

# ------------------------------------------------------------------ derived
derived = {"primevul_pub_over_rc": audit["primevul"]["0.7"]["leaked_pct"] / audit["primevul"]["0.7"]["rc_leaked_pct"],
           "min_leak_at_0_9": min(audit[c]["0.9"][k] for c in audit for k in ("leaked_pct", "rc_leaked_pct")),
           "primevul_pub_leaked_pos_pct": 100 * pv["leaked_pos"] / pv["n_pos"]}
print(f"{len(checks)} checks; {len(fails)} failed")
for f in fails:
    print("  FAIL:", f)
if fails:
    sys.exit(1)
os.makedirs(OUT, exist_ok=True)
json.dump({"calibration": calib, "audit": audit, "inputs": inputs, "family_label": fl, "primevul_prep": prep,
           "primevul_pairs": pairs, "derived": derived, "checks": checks},
          open(os.path.join(OUT, "rq1_numbers.json"), "w", encoding="utf-8"), indent=1)
print("wrote out/rq1_numbers.json")
