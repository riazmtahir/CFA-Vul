#!/usr/bin/env python3
"""design_numbers.py -- check every number in sections/study_design.tex against the files.

Reads
  results/from_backup_1012/family_label/{devign,diversevul}_theta0.7.json, primevul_theta0.7.json
                                                       corpus sizes and positives
  results/from_backup/sweep_{devign,diversevul}/lr*/codebert-base_published_seed1.json
                                                       learning-rate grid and selection
  results/from_backup/rq2_diversevul/*.json            pilot protocol (3 epochs, unweighted, 2e-5)
  results/from_backup/rq2_devign_final/*.json          Devign final runs (count)
  results/rq2_diversevul_{final,phase2,phase3,phase4b}/codebert-base_*_seed*.json
                                                       final protocol, run counts, seeds
  results/rq2_diversevul_{phase2,phase3,phase4b}/provenance.txt   GPU and start times
  results/h4_precheck_1022/{h4b_precheck,h4_design}.json, h4_precheck.log   H4b sizes, gate time
  git log                                              pre-registration commits, committer times (UTC+8)
Any mismatch -> exit 1. Output: analysis/scripts/out/design_numbers.json.
Usage (repository root): python analysis/scripts/design_numbers.py
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import re
import statistics as st
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
checks, fails = [], []


def chk(name, got, want, nd=None, src=""):
    ok = (got == want) if nd is None else abs(got - want) <= 0.5 * 10 ** (-nd) + 1e-9
    checks.append({"name": name, "got": got, "want": want, "src": src, "ok": ok})
    if not ok:
        fails.append(checks[-1])


def load(*p):
    return json.load(open(os.path.join(*p), encoding="utf-8"))


# ---- corpora
for c, path, n, pos, pct in (("Devign", "from_backup_1012/family_label/devign_theta0.7.json", 27318, 12460, 45.61),
                             ("DiverseVul", "from_backup_1012/family_label/diversevul_theta0.7.json", 330491, 18945, 5.73),
                             ("PrimeVul", "from_backup_1012/family_label/primevul_theta0.7.json", 235768, 6968, 2.96)):
    d = load(R, path)
    chk(f"{c} functions", d["size_distribution"]["n_functions"], n, src=path)
    chk(f"{c} positives", d["label_consistency"]["positives_total"], pos, src=path)
    chk(f"{c} % vulnerable", 100 * d["label_consistency"]["positives_total"] / d["size_distribution"]["n_functions"],
        pct, 2, path)
pub = load(R, "rq2_diversevul_final", "codebert-base_published_seed1.json")
chk("DiverseVul published train / valid / test", (pub["n_train"], pub["n_valid"], pub["test"]["n"]),
    (264392, 33049, 33050), src="published run json")

# ---- learning-rate selection
for corpus, want_lr, want_va, epochs, ntr in (("diversevul", 1e-5, 0.2131, 8, 60000), ("devign", 5e-5, 0.7194, 15, 21854)):
    runs = [load(f) for f in glob.glob(os.path.join(R, "from_backup", f"sweep_{corpus}", "lr*", "*.json"))]
    grid = sorted(r["lr"] for r in runs)
    chk(f"{corpus} lr grid", grid, [1e-5, 2e-5, 3e-5, 5e-5], src="sweep jsons")
    va = {r["lr"]: max(h["valid"]["pr_auc"] for h in r["history"] if "valid" in h) for r in runs}
    best = max(va, key=va.get)
    chk(f"{corpus} selected lr", best, want_lr, src="sweep jsons")
    chk(f"{corpus} selected valid PR-AUC", va[best], want_va, 4, "sweep jsons")
    chk(f"{corpus} sweep max epochs", {r["max_epochs"] for r in runs}, {epochs}, src="sweep jsons")
    chk(f"{corpus} sweep train size", {r["n_train"] for r in runs}, {ntr}, src="sweep jsons")
    chk(f"{corpus} sweep arm / seed", {(r["arm"], r["seed"]) for r in runs}, {("published", 1)}, src="sweep jsons")
    chk(f"{corpus} lr file", float(open(os.path.join(R, "from_backup", f"lr_{corpus}.txt")).read()), want_lr,
        src=f"lr_{corpus}.txt")

# ---- final protocol, runs, seeds
dirs = ["rq2_diversevul_final", "rq2_diversevul_phase2", "rq2_diversevul_phase3", "rq2_diversevul_phase4b"]
runs = {}
for d in dirs:
    for f in glob.glob(os.path.join(R, d, "codebert-base_*_seed*.json")):
        r = load(f)
        runs[(r["arm"], r["seed"])] = (d, r)
chk("DiverseVul final-protocol runs", len(runs), 25, src="run jsons")
chk("protocol fields (all 25)", {(r["lr"], r["class_weight"], r["max_epochs"], r["patience"], r["batch_size"],
                                  r["max_len"]) for _, r in runs.values()},
    {(1e-5, "balanced", 15, 2, 16, 512)}, src="run jsons")
seeds = {}
for (arm, s) in runs:
    seeds.setdefault(arm, set()).add(s)
chk("seeds per arm", {a: sorted(v) for a, v in seeds.items()},
    {"published": [1, 2, 3], "random_control": [1, 2, 3, 4, 5], "family_disjoint": [1, 2, 3, 4, 5],
     "project_disjoint": [1, 2, 3], "commit_disjoint": [1, 2, 3], "commit_decon": [1, 2, 3],
     "commit_sizectrl": [1, 2, 3]}, src="run jsons")
chk("first 9 runs (Stage B) in rq2_diversevul_final", sum(1 for d, _ in runs.values() if d == dirs[0]), 9,
    src="run jsons")
dev = glob.glob(os.path.join(R, "from_backup", "rq2_devign_final", "*.json"))
chk("Devign final runs", len(dev), 18, src="rq2_devign_final")
chk("Devign models x arms x seeds", sorted({(load(f)["model"].split("/")[-1], load(f)["arm"], load(f)["seed"]) for f in dev})
    == sorted({(m, a, s) for m in ("codebert-base", "graphcodebert-base")
               for a in ("published", "random_control", "family_disjoint") for s in (1, 2, 3)}), True, src="rq2_devign_final")
for d in dirs[1:]:
    prov = open(os.path.join(R, d, "provenance.txt"), encoding="utf-8").read()
    chk(f"{d} GPU", "RTX 4090 D" in prov, True, src="provenance.txt")
pilot = [load(f) for f in glob.glob(os.path.join(R, "from_backup", "rq2_diversevul", "*.json"))]
chk("pilot lr", {p["lr"] for p in pilot}, {2e-5}, src="rq2_diversevul pilot")
chk("pilot epochs", {p["epochs"] for p in pilot}, {3}, src="rq2_diversevul pilot")
chk("pilot class weight", {p.get("class_weight") for p in pilot}, {None}, src="rq2_diversevul pilot")
pm = {a: st.mean(p["test"]["pr_auc"] for p in pilot if p["arm"] == a) for a in ("published", "family_disjoint")}
chk("pilot family - published mean PR-AUC", pm["family_disjoint"] - pm["published"], -0.074, 3, "pilot jsons")

# ---- H4b sizes and gate
pb = load(R, "h4_precheck_1022", "h4b_precheck.json")
chk("H4b removed", pb["removed"], 35192, src="h4b_precheck.json")
chk("H4b removed % of train", pb["removed_pct_of_train"], 13.29, 2, "h4b_precheck.json")
chk("H4b train per arm", pb["commit_disjoint_train"] - pb["removed"], 229583, src="h4b_precheck.json")
chk("H4b train per arm == kept composition n", pb["composition"]["kept"]["n"], 229583, src="h4b_precheck.json")
log = open(os.path.join(R, "h4_precheck_1022", "h4_precheck.log"), encoding="utf-8").read()
chk("gate run started 22 Oct 06:09", "2026-10-22 06:09" in log, True, src="h4_precheck.log")

# ---- partition sizes of the split arms (study_design.tex, Section 3.4; added 24 Oct 2026)
SIZES = {("rq2_diversevul_final", "random_control"): (264775, 32544, 33172),
         ("rq2_diversevul_final", "family_disjoint"): (264775, 32544, 33172),
         ("rq2_diversevul_phase3", "commit_disjoint"): (264775, 32544, 33172),
         ("rq2_diversevul_phase2", "project_disjoint"): (264793, 32533, 33165),
         ("rq2_diversevul_phase4b", "commit_decon"): (229583, 32544, 33172),
         ("rq2_diversevul_phase4b", "commit_sizectrl"): (229583, 32544, 33172)}
for (d, arm), want in SIZES.items():
    for s in (1, 2, 3):
        rj = json.load(open(os.path.join(R, d, f"codebert-base_{arm}_seed{s}.json"), encoding="utf-8"))
        chk(f"{arm} s{s} train/valid/test sizes", (rj["n_train"], rj["n_valid"], rj["test"]["n"]), want,
            src=f"{d} run json")

# ---- pre-registration timeline (git, shown in UTC+8)
CST = dt.timezone(dt.timedelta(hours=8))


# Replication package: without the development repository's history, commit times and changed files are
# read from PROTOCOL_HISTORY.md (exported from that history); the study plan itself is not distributed.
GIT = subprocess.run(["git", "cat-file", "-e", "108db2c^{commit}"], cwd=ROOT, capture_output=True).returncode == 0
PH, PH_HIST = {}, {}
if not GIT:
    for line in open(os.path.join(ROOT, "PROTOCOL_HISTORY.md"), encoding="utf-8"):
        m = re.match(r"\|\s*(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \+0800\s*\|\s*`([0-9a-f]{7})`\s*\|([^|]*)\|", line)
        if m:
            PH[m.group(2)] = (dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=CST),
                              re.findall(r"`([^`]+)`", m.group(3)))
        m = re.match(r"\|\s*`(analyze_phase[0-9b]+\.py)`\s*\|([^|]*)\|", line)
        if m:
            PH_HIST[m.group(1)] = re.findall(r"`([0-9a-f]{7})`", m.group(2))


def ctime(h):
    if not GIT:
        return PH[h][0]
    s = subprocess.run(["git", "log", "-1", "--format=%cI", h], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return dt.datetime.fromisoformat(s).astimezone(CST)


def touches(h, path):
    if not GIT:
        return path in [os.path.basename(p) for p in PH[h][1]] or path in PH[h][1]
    out = subprocess.run(["git", "show", "--name-only", "--format=", h], cwd=ROOT, capture_output=True, text=True).stdout
    return path in out.split()


TL = {"108db2c": ("2026-10-05 22:20", "RESEARCH_PLAN.md"), "3ee3b4b": ("2026-10-06 02:50", "analyze_phase2.py"), "c58f02e": ("2026-10-07 19:39", "analyze_phase3.py"),
      "ec6523a": ("2026-10-07 20:14", "RESEARCH_PLAN.md"), "90de531": ("2026-10-21 04:40", "analyze_phase4.py"),
      "586236e": ("2026-10-21 05:17", "analyze_phase4b.py"), "0936287": ("2026-10-21 05:28", "RESEARCH_PLAN.md")}
for h, (when, path) in TL.items():
    chk(f"commit {h} time (CST)", ctime(h).strftime("%Y-%m-%d %H:%M"), when, src="git log")
    chk(f"commit {h} touches {path}", touches(h, path), True, src="git show")
starts = {}
for d in dirs[1:]:
    m = re.search(r"started (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) CST", open(os.path.join(R, d, "provenance.txt")).read())
    starts[d] = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=CST)
chk("Phase 2 start", starts["rq2_diversevul_phase2"].strftime("%m-%d %H:%M"), "10-06 02:33", src="provenance.txt")
chk("Phase 3 start", starts["rq2_diversevul_phase3"].strftime("%m-%d %H:%M"), "10-08 01:59", src="provenance.txt")
chk("Phase 4b start", starts["rq2_diversevul_phase4b"].strftime("%m-%d %H:%M"), "10-22 06:46", src="provenance.txt")
# Table 3 columns "Amendment" and "Analysis script" (added 25 Oct 2026)
for script, want in (("analyze_phase2.py", ["3ee3b4b"]), ("analyze_phase3.py", ["ec6523a", "c58f02e"]),
                     ("analyze_phase4.py", ["90de531"]), ("analyze_phase4b.py", ["586236e"])):
    hist = (subprocess.run(["git", "log", "--format=%h", "--", script], cwd=ROOT, capture_output=True,
                           text=True).stdout.split() if GIT else PH_HIST.get(script))
    chk(f"{script} commit history", hist, want, src="git log -- " + script)
chk("0936287 does not touch analyze_phase4b.py", touches("0936287", "analyze_phase4b.py"), False, src="git show")
if GIT:  # the study plan is checked only in the development repository
    chk("586236e plan holds H4b amendment 1", "Amended 21 Oct 2026, before any H4/H4b pre-check or run" in
        subprocess.run(["git", "show", "586236e:RESEARCH_PLAN.md"], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8").stdout, True, src="git show 586236e:RESEARCH_PLAN.md")
    plan_txt = subprocess.run(["git", "show", "108db2c:RESEARCH_PLAN.md"], cwd=ROOT, capture_output=True,
                              text=True, encoding="utf-8").stdout
    chk("108db2c holds the H1/H2 section", "Pre-registered hypotheses" in plan_txt and "is a tie" in plan_txt, True,
        src="git show 108db2c:RESEARCH_PLAN.md")
else:
    print("  plan-text checks skipped: the study plan is in the development repository only")
lead = (starts["rq2_diversevul_phase2"] - ctime("108db2c")).total_seconds() / 60
chk("H1/H2 rules minutes before Phase 2 start", round(lead), 253, src="git + provenance")
lag = (ctime("3ee3b4b") - starts["rq2_diversevul_phase2"]).total_seconds() / 60
chk("H1/H2 commit minutes after Phase 2 start", round(lag), 17, src="git + provenance")
first_done = dt.datetime(2026, 10, 6, 3, 48, 54, tzinfo=CST)   # run_phase2.log: next run starts 10-06 03:48:54
# whole minutes (floor), as a reader computes them from Table 5's 02:50 and 03:48 (58 min 47 s; review item 19)
chk("H1/H2 commit minutes before first Phase 2 run finished", int((first_done - ctime("3ee3b4b")).total_seconds() // 60),
    58, src="results/h4_precheck_1022/logs/run_phase2.log")
for h, d in (("ec6523a", "rq2_diversevul_phase3"), ("0936287", "rq2_diversevul_phase4b")):
    chk(f"{h} before {d} start", ctime(h) < starts[d], True, src="git + provenance")
chk("H4/H4b commits before gate run", ctime("0936287") < dt.datetime(2026, 10, 22, 6, 9, tzinfo=CST), True,
    src="git + h4_precheck.log")

os.makedirs(OUT, exist_ok=True)
json.dump({"checks": checks}, open(os.path.join(OUT, "design_numbers.json"), "w", encoding="utf-8"), indent=1, default=str)
print(f"{len(checks)} checks; {len(fails)} failed")
for f in fails:
    print("  FAIL", f["name"], "got", f["got"], "want", f["want"], f["src"])
sys.exit(1 if fails else 0)
