#!/usr/bin/env python3
"""rq3_numbers.py -- recompute and cross-check every RQ3 number from the result files.

Reads (nothing typed in by hand except the the project log reference values checked against):
  results/from_backup_1012/family_label/{diversevul,primevul,devign}_commit_leak.json
      commit-level exposure per arm (commit_leak_audit.py output; counts), checked against
  results/audit_7oct/family_label/*_commit_leak.md (the generated reports)
  results/from_backup_1012/family_label/primevul_pairs.json   PrimeVul pair share
  results/rq2_diversevul_phase2/phase2_analysis.{json,md}      pre-registered H1 (section C)
  results/rq2_diversevul_phase2/twin_effect_exploratory.{json,md}   exploratory twin effect
  results/rq2_diversevul_phase2/paired_eval.{json,md}          descriptive paired evaluation
  results/from_backup/{devign,diversevul}_audit_report.json    1-NN baseline (theta 0.7 entry)
  results/from_backup_1012/primevul_audit_report.json         1-NN baseline, PrimeVul (+ .md check)
  results/rq2_diversevul_{final,phase2,phase3}/*.json          length-only floor, headroom
  results/from_backup/rq2_devign_final/*.json                  Devign CodeBERT published PR-AUC

Checks
  * JSON against the generated .md report of the same script (4 decimals as printed);
  * recomputations: paired means = mean of per-seed values, length-balanced rank accuracy =
    mean of the two strata, H1 verdicts re-derived from the CIs and the fixed rule, twin
    group sizes against the own-twin audit counts, group sizes summing to the test classes;
  * every value against the project log (3 decimals where the project log gives 3, else 4 / 2 for %).
Any mismatch of a recomputation -> exit 1, nothing written. Differences between the project log
and a result file are listed as "superseded" (the result file wins) and do not stop.

Output: analysis/scripts/out/rq3_numbers.json (read by rq3_tables_figure.py).
Usage (repository root): python analysis/scripts/rq3_numbers.py
"""
from __future__ import annotations

import glob
from decimal import ROUND_HALF_UP, Decimal
import json
import os
import re
import statistics as st
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
P2 = os.path.join(RES, "rq2_diversevul_phase2")

checks, superseded, fails = [], [], []


def half_up(x, nd):
    return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def chk(name, got, want, nd, src, hard=True):
    # the project log values were rounded half-up from the 4-decimal .md reports (e.g. 0.1805 -> 0.181);
    # that double rounding is accepted, nothing wider.
    ok = (abs(got - want) <= 0.5 * 10 ** (-nd) + 1e-12
          or (nd < 4 and half_up(half_up(got, 4), nd) == round(want, nd)))
    checks.append({"name": name, "got": got, "want": want, "decimals": nd, "src": src, "ok": ok})
    if not ok:
        (fails if hard else superseded).append(checks[-1])


def md_rows(path):
    """All table rows of a markdown file as lists of stripped cells."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("|") and not line.startswith("|---"):
                out.append([c.strip() for c in line.strip().strip("|").split("|")])
    return out


def num(s):
    return float(s.replace(",", "").replace("**", "").strip())


# ------------------------------------------------------------------ 1. commit-level exposure
COLS = ("n", "same_commit", "opp_label", "own_twin", "own_twin_outside_family")
JKEYS = {"same_commit": ("same_commit_in_train", "same_commit_pct"),
         "opp_label": ("opposite_same_commit", "opposite_same_commit_pct"),
         "own_twin": ("own_twin_in_train", "own_twin_pct"),
         "own_twin_outside_family": ("own_twin_outside_family", "own_twin_outside_family_pct")}
B12 = os.path.join(RES, "from_backup_1012")
exposure = {}
for corpus in ("diversevul", "primevul", "devign"):
    # primary: the audit JSON (counts); percentages recomputed from counts, then checked
    # against the JSON's own pct and the generated .md report
    cj = json.load(open(os.path.join(B12, "family_label", f"{corpus}_commit_leak.json"), encoding="utf-8"))
    for arm, subs in cj["arms"].items():
        for sub in ("all_test", "test_positives"):
            s = subs[sub]
            row = {"n": s["n"]}
            for col, (ck, pk) in JKEYS.items():
                row[col] = 100.0 * s[ck] / s["n"]
                chk(f"{corpus} {arm} {sub} {col}: 100*count/n == JSON pct", row[col], s[pk], 2, "recomputed")
            exposure.setdefault(corpus, {}).setdefault(arm, {})[sub] = row
    path = os.path.join(RES, "audit_7oct", "family_label", f"{corpus}_commit_leak.md")
    for r in md_rows(path):
        if len(r) == 7 and r[1] in ("all_test", "test_positives"):
            for col, v in zip(COLS, map(num, r[2:])):
                chk(f"{corpus} {r[0]} {r[1]} {col}: md == JSON", exposure[corpus][r[0]][r[1]][col], v,
                    0 if col == "n" else 2, f"audit_7oct/family_label/{corpus}_commit_leak.md")
# the project log "Commit-level sharing" and "Own twins" tables
CL = {("diversevul", "published"): (99.75, 90.84, 95.03, 70.90, 42.52, 9.36),
      ("diversevul", "random_control"): (99.72, 91.63, 92.79, 69.02, 42.29, 9.44),
      ("diversevul", "family_disjoint"): (99.54, 90.01, 92.38, 46.01, 46.01, 6.52),
      ("diversevul", "project_disjoint"): (0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
      ("diversevul", "commit_disjoint"): (None, None, None, 0.00, 0.00, 0.00),
      ("primevul", "published"): (0.66, 0.66, 0.86, 0.86, 0.58, 0.06),
      ("primevul", "random_control"): (99.51, 77.87, 85.73, 62.41, 35.12, 4.23),
      ("primevul", "family_disjoint"): (99.38, 77.42, 84.29, 32.45, 32.45, 2.50),
      ("devign", "published"): (65.04, 0.0, 0.0, 0.0, 0.0, 0.0),
      ("devign", "random_control"): (66.18, 0.0, 0.0, 0.0, 0.0, 0.0),
      ("devign", "family_disjoint"): (65.08, 0.0, 0.0, 0.0, 0.0, 0.0)}
for (corpus, arm), (sc, opp_all, opp_pos, tw_pos, tw_out_pos, tw_all) in CL.items():
    e = exposure[corpus][arm]
    src = f"audit_7oct/family_label/{corpus}_commit_leak.md"
    if sc is not None:
        chk(f"{corpus} {arm} same commit % (all)", e["all_test"]["same_commit"], sc, 2, src)
        chk(f"{corpus} {arm} opp. label same commit % (all)", e["all_test"]["opp_label"], opp_all, 2, src)
        chk(f"{corpus} {arm} opp. label same commit % (pos)", e["test_positives"]["opp_label"], opp_pos, 2, src)
    chk(f"{corpus} {arm} own twin % (pos)", e["test_positives"]["own_twin"], tw_pos, 2, src)
    chk(f"{corpus} {arm} own twin outside family % (pos)", e["test_positives"]["own_twin_outside_family"], tw_out_pos, 2, src)
    chk(f"{corpus} {arm} own twin % (all)", e["all_test"]["own_twin"], tw_all, 2, src)
# family-disjoint: every remaining own twin is outside the family (by construction)
for corpus in ("diversevul", "primevul"):
    for sub in ("all_test", "test_positives"):
        e = exposure[corpus]["family_disjoint"][sub]
        chk(f"{corpus} family_disjoint {sub}: own twin == outside family", e["own_twin"], e["own_twin_outside_family"], 2,
            "recomputed")

# ------------------------------------------------------------------ 2. H1 (pre-registered)
p2a = json.load(open(os.path.join(P2, "phase2_analysis.json"), encoding="utf-8"))
h1 = p2a["C_H1"]["random_control"]
PRED = {"positive": 1, "negative": -1}
h1_out = {"seeds": h1["seeds"], "valid_thresholds": h1["valid_thresholds"], "contrasts": {}, "categories": {}}
CL_H1 = {"H1a": (0.028, -0.019, 0.074, "TIE"), "H1b": (0.516, 0.480, 0.553, "SUPPORTED"),
         "H1c": (0.262, 0.181, 0.345, "SUPPORTED"), "H1d": (-0.039, -0.044, -0.035, "SUPPORTED")}
for name, c in h1["contrasts"].items():
    key = name.split()[0]
    lo, hi = c["ci95"]
    s = PRED[c["predicted_sign"]]
    v = "SUPPORTED" if (lo > 0 and s > 0) or (hi < 0 and s < 0) else \
        "CONTRADICTED" if (lo > 0 and s < 0) or (hi < 0 and s > 0) else "TIE"
    if v != c["verdict"]:
        fails.append({"name": f"{key} verdict re-derived", "got": v, "want": c["verdict"]})
    checks.append({"name": f"{key} verdict re-derived", "got": v, "want": c["verdict"], "ok": v == c["verdict"]})
    d0, l0, h0, v0 = CL_H1[key]
    chk(f"{key} difference vs the project log", c["difference"], d0, 3, "phase2_analysis.json")
    chk(f"{key} CI lo vs the project log", lo, l0, 3, "phase2_analysis.json")
    chk(f"{key} CI hi vs the project log", hi, h0, 3, "phase2_analysis.json")
    h1_out["contrasts"][key] = {"difference": c["difference"], "ci95": c["ci95"], "verdict": c["verdict"],
                                "n_category": c["n_category"], "n_clean": c["n_clean"]}
whole = "SUPPORTED" if all(h1_out["contrasts"][k]["verdict"] == "SUPPORTED" for k in ("H1a", "H1b")) else "NOT SUPPORTED"
checks.append({"name": "H1 overall re-derived (needs H1a and H1b)", "got": whole,
               "want": h1["verdict_H1"], "ok": h1["verdict_H1"].startswith(whole)})
if not h1["verdict_H1"].startswith(whole):
    fails.append(checks[-1])
for cat, c in ((k, v) for k, v in h1["categories"].items() if isinstance(v, dict)):
    m = c["mean_over_seeds"]
    h1_out["categories"][cat] = {"n": c["n"], "pos_rate": c["pos_rate"], "recall": m["recall_pos"],
                                 "fpr": m["fpr_neg"], "roc_auc": m["roc_auc"]}
CL_CAT = {"clean": (23424, 0.046, 0.355, 0.072, 0.781), "leaked_same": (7996, 0.015, 0.617, 0.033, 0.920),
          "leaked_opposite": (1150, 0.508, 0.383, 0.588, 0.363), "leaked_both": (602, 0.199, 0.654, 0.202, 0.827)}
for cat, (n, pr, rc, fp, ra) in CL_CAT.items():
    o = h1_out["categories"][cat]
    chk(f"H1 {cat} n", o["n"], n, 0, "phase2_analysis.json")
    chk(f"H1 {cat} pos rate", o["pos_rate"], pr, 3, "phase2_analysis.json")
    chk(f"H1 {cat} recall", o["recall"], rc, 3, "phase2_analysis.json")
    chk(f"H1 {cat} FPR", o["fpr"], fp, 3, "phase2_analysis.json")
    chk(f"H1 {cat} ROC-AUC", o["roc_auc"], ra, 3, "phase2_analysis.json")
# md report agrees with JSON (4 decimals as printed)
md = open(os.path.join(P2, "phase2_analysis.md"), encoding="utf-8").read()
for key, c in h1_out["contrasts"].items():
    m = re.search(rf"\| {key} [^|]*\| ([+-][0-9.]+) \| \[([+-][0-9.]+), ([+-][0-9.]+)\]", md)
    chk(f"{key} md == json", float(m.group(1)), c["difference"], 4, "phase2_analysis.md")
    chk(f"{key} md CI lo == json", float(m.group(2)), c["ci95"][0], 4, "phase2_analysis.md")

# ------------------------------------------------------------------ 3. twin effect (exploratory)
tw = json.load(open(os.path.join(P2, "twin_effect_exploratory.json"), encoding="utf-8"))
twin = {}
for arm in ("random_control", "family_disjoint"):
    a = tw["arms"][arm]
    twin[arm] = {"seeds": a["seeds"]}
    for cls, metric in (("positives", "recall"), ("negatives", "FPR")):
        g, c = a[cls]["groups"], a[cls]["contrasts"]
        twin[arm][cls] = {"groups": {k: {"n": v["n"], "rate": v[metric], "mean_len": v["mean_len_chars"]}
                                     for k, v in g.items()},
                          "contrasts": c}
# group sizes: positives sum to test positives; twin_in_train == own-twin audit count
POS = {"random_control": 1901, "family_disjoint": 1891}
for arm in POS:
    gp = twin[arm]["positives"]["groups"]
    gn = twin[arm]["negatives"]["groups"]
    npos = sum(gp[k]["n"] for k in ("twin_in_train", "twin_elsewhere", "no_twin"))
    nneg = sum(gn[k]["n"] for k in ("twin_in_train", "twin_elsewhere", "no_twin"))
    chk(f"twin {arm} positives sum to test positives", npos, exposure["diversevul"][arm]["test_positives"]["n"], 0,
        "recomputed")
    chk(f"twin {arm} all groups sum to test n", npos + nneg, exposure["diversevul"][arm]["all_test"]["n"], 0,
        "recomputed")
    chk(f"twin {arm} twin_in_train / positives == own-twin audit %", 100 * gp["twin_in_train"]["n"] / npos,
        exposure["diversevul"][arm]["test_positives"]["own_twin"], 2, "recomputed")
    chk(f"twin {arm} twin_in_train (all) / test n == own-twin audit % (all)",
        100 * (gp["twin_in_train"]["n"] + gn["twin_in_train"]["n"]) / (npos + nneg),
        exposure["diversevul"][arm]["all_test"]["own_twin"], 2, "recomputed")
gp = twin["random_control"]["positives"]["groups"]
chk("famleak split sums to twin_in_train", gp["twin_in_train & famleak"]["n"] + gp["twin_in_train & no famleak"]["n"],
    gp["twin_in_train"]["n"], 0, "recomputed")
CL_TW = {("random_control", "negatives"): ((0.426, 0.364, 0.048), (0.062, 0.009, 0.117), (0.097, 0.052, 0.146)),
         ("family_disjoint", "negatives"): ((0.392, 0.354, 0.044), (0.039, 0.003, 0.079), (0.112, 0.080, 0.147)),
         ("random_control", "positives"): ((0.382, 0.372, 0.494), (0.009, -0.047, 0.069), (0.014, None, None)),
         ("family_disjoint", "positives"): ((0.347, 0.381, 0.542), (-0.034, -0.075, 0.008), (-0.000, None, None))}
for (arm, cls), (rates, raw, std) in CL_TW.items():
    g = twin[arm][cls]["groups"]
    for k, want in zip(("twin_in_train", "twin_elsewhere", "no_twin"), rates):
        chk(f"twin {arm} {cls} {k} rate vs the project log", g[k]["rate"], want, 3, "twin_effect_exploratory.json")
    c = twin[arm][cls]["contrasts"]["train_vs_elsewhere"]
    chk(f"twin {arm} {cls} train-elsewhere raw", c["raw"], raw[0], 3, "twin_effect_exploratory.json")
    chk(f"twin {arm} {cls} train-elsewhere raw lo", c["raw_ci95"][0], raw[1], 3, "twin_effect_exploratory.json")
    chk(f"twin {arm} {cls} train-elsewhere raw hi", c["raw_ci95"][1], raw[2], 3, "twin_effect_exploratory.json")
    chk(f"twin {arm} {cls} train-elsewhere std", c["std"], std[0], 3, "twin_effect_exploratory.json")
    if std[1] is not None:
        chk(f"twin {arm} {cls} train-elsewhere std lo", c["std_ci95"][0], std[1], 3, "twin_effect_exploratory.json")
        chk(f"twin {arm} {cls} train-elsewhere std hi", c["std_ci95"][1], std[2], 3, "twin_effect_exploratory.json")
# user-supplied values (12 Oct message)
chk("recall train-vs-none raw (random)", twin["random_control"]["positives"]["contrasts"]["train_vs_none"]["raw"],
    -0.1123, 4, "twin_effect_exploratory.json")
chk("recall twin_in_train & famleak (random)", gp["twin_in_train & famleak"]["rate"], 0.4197, 4,
    "twin_effect_exploratory.json")
chk("recall twin_in_train & no famleak (random)", gp["twin_in_train & no famleak"]["rate"], 0.3456, 4,
    "twin_effect_exploratory.json")
# md report agrees with JSON
md = open(os.path.join(P2, "twin_effect_exploratory.md"), encoding="utf-8").read()
for arm in ("random_control", "family_disjoint"):
    sec = md.split(f"\n## {arm}")[1].split("\n## ")[0]
    for cls, metric, lab in (("positives", "recall", "recall"), ("negatives", "FPR", "FPR")):
        for cname in ("train_vs_elsewhere", "train_vs_none"):
            c = twin[arm][cls]["contrasts"][cname]
            pat = rf"- {lab} {cname.replace('_vs_', ' vs ')}: raw \*\*([+-][0-9.]+)\*\* \[([+-][0-9.]+), ([+-][0-9.]+)\] · length-std \*\*([+-][0-9.]+)\*\* \[([+-][0-9.]+), ([+-][0-9.]+)\]"
            m = re.search(pat, sec)
            vals = [float(x) for x in m.groups()]
            want = [c["raw"], *c["raw_ci95"], c["std"], *c["std_ci95"]]
            for i, (a, b) in enumerate(zip(vals, want)):
                chk(f"twin md {arm} {cls} {cname} [{i}]", a, b, 4, "twin_effect_exploratory.md")

# ------------------------------------------------------------------ 4. paired evaluation (descriptive)
pe = json.load(open(os.path.join(P2, "paired_eval.json"), encoding="utf-8"))
paired = {}
for arm, a in pe["arms"].items():
    for k in ("P-C", "P-V", "P-B", "P-R", "rank_acc"):
        chk(f"paired {arm} {k} mean == mean of seeds", a["mean"][k], st.mean(v[k] for v in a["per_seed"].values()), 10,
            "recomputed")
    chk(f"paired {arm} P-C+P-V+P-B+P-R == 1", sum(a["mean"][k] for k in ("P-C", "P-V", "P-B", "P-R")), 1.0, 10,
        "recomputed")
    L = a["length_check"]
    chk(f"paired {arm} balanced == mean of strata", L["balanced_rank_acc"],
        0.5 * (L["rank_acc_patched_longer"] + L["rank_acc_patched_not_longer"]), 10, "recomputed")
    chk(f"paired {arm} strata n sum to pairs", L["n_patched_longer"] + L["n_patched_not_longer"], a["pairs"], 0,
        "recomputed")
    chk(f"paired {arm} patched-longer share == n ratio", L["patched_longer_frac"], L["n_patched_longer"] / a["pairs"], 10,
        "recomputed")
    paired[arm] = {"seeds": a["seeds"], "pairs": a["pairs"], "ambiguous": a["ambiguous_groups"],
                   "mean": a["mean"], "ci95": a["ci95"], **L}
CL_PE = {"random_control": (144, 0.056, 0.389, 0.521, 0.035, 0.556, (0.496, 0.622), 0.688, 0.306, 0.601, (0.542, 0.668)),
         "family_disjoint": (688, 0.036, 0.368, 0.588, 0.008, 0.515, (0.491, 0.538), 0.727, 0.249, 0.547, (0.521, 0.573)),
         "project_disjoint": (1237, 0.005, 0.196, 0.790, 0.009, 0.416, (0.397, 0.435), 0.656, 0.329, 0.454, (0.434, 0.473)),
         "commit_disjoint": (1515, 0.013, 0.255, 0.713, 0.019, 0.438, (0.424, 0.454), 0.781, 0.211, 0.494, (0.474, 0.512))}
for arm, (n, pc, pv, pb, prr, ra, raci, longer, lonly, bal, balci) in CL_PE.items():
    p = paired[arm]
    src = "paired_eval.json"
    chk(f"paired {arm} pairs", p["pairs"], n, 0, src)
    for k, w in (("P-C", pc), ("P-V", pv), ("P-B", pb), ("P-R", prr), ("rank_acc", ra)):
        chk(f"paired {arm} {k} vs the project log", p["mean"][k], w, 3, src)
    # the project log "first result" CIs for random/family/project predate the 8 Oct re-run -> soft
    chk(f"paired {arm} rank_acc CI lo vs the project log", p["ci95"]["rank_acc"][0], raci[0], 3, src, hard=False)
    chk(f"paired {arm} rank_acc CI hi vs the project log", p["ci95"]["rank_acc"][1], raci[1], 3, src, hard=False)
    chk(f"paired {arm} patched longer vs the project log", p["patched_longer_frac"], longer, 3, src)
    chk(f"paired {arm} length-only RA vs the project log", p["length_only_rank_acc"], lonly, 3, src)
    chk(f"paired {arm} balanced RA vs the project log", p["balanced_rank_acc"], bal, 3, src)
    chk(f"paired {arm} balanced CI lo vs the project log", p["balanced_rank_acc_ci95"][0], balci[0], 3, src)
    chk(f"paired {arm} balanced CI hi vs the project log", p["balanced_rank_acc_ci95"][1], balci[1], 3, src)
# md report agrees with JSON
for r in md_rows(os.path.join(P2, "paired_eval.md")):
    if r[0] in paired and len(r) == 9:
        m = re.match(r"([0-9.]+) \[([0-9.]+), ([0-9.]+)\]", r[8])
        chk(f"paired md {r[0]} rank_acc", float(m.group(1)), paired[r[0]]["mean"]["rank_acc"], 3, "paired_eval.md")
        chk(f"paired md {r[0]} rank_acc lo", float(m.group(2)), paired[r[0]]["ci95"]["rank_acc"][0], 3, "paired_eval.md")
    if r[0] in paired and len(r) == 6:
        m = re.match(r"([0-9.]+) \[([0-9.]+), ([0-9.]+)\]", r[5])
        chk(f"paired md {r[0]} balanced", float(m.group(1)), paired[r[0]]["balanced_rank_acc"], 3, "paired_eval.md")

# ------------------------------------------------------------------ 5. floors and 1-NN
rq2 = json.load(open(os.path.join(OUT, "rq2_numbers.json"), encoding="utf-8"))
floors = {}
for arm, a in rq2["per_arm"].items():
    if arm.endswith("_s1_3"):
        continue
    runs = [json.load(open(os.path.join(ROOT, f), encoding="utf-8")) for f in a["files"]]
    hr = [r["test"]["pr_auc"] - r["length_only_floor"]["pr_auc"] for r in runs]
    fl = [r["length_only_floor"]["pr_auc"] for r in runs]
    chk(f"floor {arm} recomputed == rq2_numbers", st.mean(fl), a["floor"]["mean"], 4, "per-seed JSONs")
    chk(f"headroom {arm} recomputed == rq2_numbers", st.mean(hr), a["headroom"]["mean"], 4, "per-seed JSONs")
    floors[arm] = {"n": len(runs), "floor": st.mean(fl), "pr_auc": a["pr_auc"]["mean"], "headroom": st.mean(hr),
                   "headroom_sd": st.stdev(hr)}
chk("headroom project mean (rq2.tex 0.0028)", floors["project_disjoint"]["headroom"], 0.0028, 4, "per-seed JSONs")
chk("headroom random mean n=5 (rq2.tex 0.1052)", floors["random_control"]["headroom"], 0.1052, 4, "per-seed JSONs")
# seeds 1-3 headroom for every arm (rq2.tex/rq3.tex use seeds 1-3 throughout since 24 Oct 2026, review item 2.6)
for arm, want in (("random_control_s1_3", 0.1039), ("family_disjoint_s1_3", 0.1070), ("published", 0.0933),
                  ("commit_disjoint", 0.0311), ("project_disjoint", 0.0028)):
    a = rq2["per_arm"][arm.replace("_s1_3", "")]
    runs = [json.load(open(os.path.join(ROOT, f), encoding="utf-8")) for f in a["files"]]
    runs = [r for r in runs if r["seed"] in (1, 2, 3)]
    chk(f"{arm} seeds 1-3 present", len(sorted(r["seed"] for r in runs)), 3, 0, "per-seed JSONs")
    chk(f"headroom {arm} seeds 1-3", st.mean(r["test"]["pr_auc"] - r["length_only_floor"]["pr_auc"] for r in runs),
        want, 4, "per-seed JSONs")

knn = {}
for corpus in ("devign", "diversevul"):
    rep = json.load(open(os.path.join(RES, "from_backup", f"{corpus}_audit_report.json"), encoding="utf-8"))
    t = [x for x in rep["thresholds"] if abs(x["theta"] - 0.7) < 1e-9][0]["knn_retrieval_baseline"]
    knn[corpus] = {"pr_auc": t["pr_auc"], "coverage_pct": t["coverage_pct"], "prevalence": t["positive_rate"],
                   "test_functions": t["test_functions"], "src": f"results/from_backup/{corpus}_audit_report.json"}
rep = json.load(open(os.path.join(B12, "primevul_audit_report.json"), encoding="utf-8"))
t = [x for x in rep["thresholds"] if abs(x["theta"] - 0.7) < 1e-9][0]["knn_retrieval_baseline"]
knn["primevul"] = {"pr_auc": t["pr_auc"], "coverage_pct": t["coverage_pct"], "prevalence": t["positive_rate"],
                   "test_functions": t["test_functions"], "src": "results/from_backup_1012/primevul_audit_report.json"}
for corpus in knn:
    k = knn[corpus]
    chk(f"1-NN {corpus} coverage: 100*covered/n == coverage_pct",
        100.0 * [x for x in json.load(open(os.path.join(ROOT, k["src"]), encoding="utf-8"))["thresholds"]
                 if abs(x["theta"] - 0.7) < 1e-9][0]["knn_retrieval_baseline"]["test_with_a_clone_in_train"]
        / k["test_functions"], k["coverage_pct"], 2, "recomputed")
for r in md_rows(os.path.join(RES, "audit_7oct", "primevul_audit_report.md")):
    if r[0] == "0.7" and len(r) == 4:
        chk("1-NN primevul md coverage == JSON", num(r[1]), knn["primevul"]["coverage_pct"], 2, "primevul_audit_report.md")
        chk("1-NN primevul md PR-AUC == JSON", num(r[2]), knn["primevul"]["pr_auc"], 4, "primevul_audit_report.md")
        break

# PrimeVul vulnerable/patch pairs in one family (theta 0.7, paired training file)
pp = json.load(open(os.path.join(B12, "family_label", "primevul_pairs.json"), encoding="utf-8"))
pt = pp["splits"]["train"]["theta"]["0.7"]
pair_share = {"pct": 100.0 * pt["patch_in_same_family"] / pt["vulnerable_with_a_same_commit_patch"],
              "n": pt["vulnerable_with_a_same_commit_patch"], "same_family": pt["patch_in_same_family"],
              "src": "results/from_backup_1012/family_label/primevul_pairs.json"}
chk("PrimeVul pair share recomputed == JSON pct", pair_share["pct"], pt["pct"], 2, "recomputed")
chk("PrimeVul pair share (the project log 47.0%)", pair_share["pct"], 47.0, 1, pair_share["src"])
for r in md_rows(os.path.join(RES, "audit_7oct", "family_label", "primevul_pairs.md")):
    if r[:2] == ["train", "0.7"]:
        chk("PrimeVul pair share md == JSON", num(r[4]), pt["pct"], 2, "primevul_pairs.md")

# LSH candidate probability at J = theta (Eq. lsh), from the run config b, r
cfg = json.load(open(os.path.join(RES, "from_backup", "diversevul_audit_report.json"), encoding="utf-8"))["config"]
b, K = cfg["bands"], cfg["perms"]
lsh_p07 = 1 - (1 - 0.7 ** (K // b)) ** b
chk("LSH candidate probability at J=0.7 (the project log ~0.61)", lsh_p07, 0.61, 2, "Eq. lsh with config bands/perms")
for corpus, (pa, cov, prev) in {"devign": (0.5234, 40.26, 0.4594), "diversevul": (0.1553, 50.59, 0.0584),
                                "primevul": (0.0944, 40.95, 0.0268)}.items():
    chk(f"1-NN {corpus} PR-AUC", knn[corpus]["pr_auc"], pa, 4, knn[corpus]["src"])
    chk(f"1-NN {corpus} coverage", knn[corpus]["coverage_pct"], cov, 2, knn[corpus]["src"])
    chk(f"1-NN {corpus} prevalence", knn[corpus]["prevalence"], prev, 4, knn[corpus]["src"])
    knn[corpus]["ratio_to_prevalence"] = knn[corpus]["pr_auc"] / knn[corpus]["prevalence"]
# CodeBERT on the same published splits (for the comparison in the text)
dev = [json.load(open(f, encoding="utf-8"))["test"]["pr_auc"]
       for f in sorted(glob.glob(os.path.join(RES, "from_backup", "rq2_devign_final", "codebert-base_published_seed*.json")))]
chk("Devign CodeBERT published n", len(dev), 3, 0, "from_backup/rq2_devign_final")
chk("Devign CodeBERT published mean (the project log 0.7283)", st.mean(dev), 0.7283, 4, "from_backup/rq2_devign_final")
codebert_published = {"devign": st.mean(dev), "diversevul": floors["published"]["pr_auc"]}

# ------------------------------------------------------------------ write
n_hard = sum(1 for c in checks if c not in superseded)
print(f"{len(checks)} checks; {len(fails)} failed; {len(superseded)} the project log values superseded by the result file")
for s in superseded:
    print("  superseded:", s["name"], "the project log", s["want"], "-> file", round(s["got"], 4))
if fails:
    for f in fails:
        print("  FAIL:", f)
    sys.exit(1)
os.makedirs(OUT, exist_ok=True)
json.dump({"exposure": exposure, "h1": h1_out, "twin": twin, "paired": paired, "floors": floors, "knn": knn,
           "codebert_published": codebert_published, "primevul_pair_share": pair_share,
           "lsh_candidate_prob_at_0_7": lsh_p07, "superseded": superseded,
           "checks": checks}, open(os.path.join(OUT, "rq3_numbers.json"), "w", encoding="utf-8"), indent=1)
print("wrote out/rq3_numbers.json")
