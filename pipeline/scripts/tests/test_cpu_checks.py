#!/usr/bin/env python3
"""test_cpu_checks.py -- synthetic end-to-end test of run_cpu_checks.sh (A3/A4/A5) and
run_audit_highrecall.sh (A14), laptop only.

Builds a tiny server-like data disk in a temporary directory:
  data/diversevul/diversevul.jsonl   C-like functions in fix commits: each changed function
                                     has a vulnerable and a patched version (same name). The
                                     patched version is planted as raw-identical, whitespace-only,
                                     rename-only, or a real change, in known numbers.
  data/primevul/primevul_all.jsonl   functions copied from DiverseVul raw / with whitespace
                                     changes / with a renamed variable, functions that share only
                                     (commit, name), and new ones, in known numbers
  data/devign/devign_all.jsonl       small corpus for the high-recall runner
  <corpus>_audit_*                   default audit outputs (clone_family_audit.py, real run)
  *_projectdisjoint_seed42.json, *_commitdisjoint_seed42.json   grouped split files
  rq2_diversevul_phase{2,3}/*.preds.json.gz   fake predictions for every arm/seed
Then checks:
  1. run_cpu_checks.sh fails loudly when an input is missing;
  2. a full run writes every output, and a re-run skips every step;
  3. A4 counts equal the planted numbers and an independent recount; paired_eval.py's default
     output is byte-identical to the HEAD version's; the no-identical run drops exactly the
     identical pairs and its rank accuracy equals a brute-force recomputation;
  4. A3 counts equal the planted numbers and a brute-force recount;
  5. A5 PR-AUC full / without leaked equal a brute-force recomputation, and the family-disjoint
     arm removes nothing;
  6. run_audit_highrecall.sh writes its outputs for all three corpora and resumes (re-run skips).
Usage (repository root): python scripts/tests/test_cpu_checks.py [--keep DIR]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score

VD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, VD)
from clone_family_audit import normalize_tokens  # noqa: E402

R = random.Random(11)
VOC = [f"{a}{b}" for a in ("buf", "len", "ptr", "idx", "cnt", "tmp", "ctx", "pos", "sz", "val")
       for b in ("", "_a", "_b", "_in", "_out", "2", "_x", "_hdr")]
PLANT = {"identical": 12, "ws": 9, "rename": 7, "real": 52}     # vulnerable/patched pairs
FAILS = []


def check(name, got, want):
    ok = got == want
    print(f"  {'ok ' if ok else 'FAIL'} {name}: {got}" + ("" if ok else f" (want {want})"))
    if not ok:
        FAILS.append(name)


def stmt():
    a, b, c = R.sample(VOC, 3)
    k = R.randrange(7)
    return [f"{a} = {b} + {R.randrange(100)};", f"if ({a} > {b}) {{ {c} = {a}; }}",
            f"memcpy({a}, {b}, {c});", f"for (i = 0; i < {a}; i++) {{ {b}[i] = {c}; }}",
            f"{a} = foo_{R.randrange(500)}({b}, {c});", f"while ({a}--) {{ {b} += {R.randrange(9)}; }}",
            f"return {a} * {b};"][k]


def func(name, n=None):
    body = "\n".join("    " + stmt() for _ in range(n or R.randrange(8, 16)))
    return f"int {name}(int {R.choice(VOC)}, char *{R.choice(VOC)})\n{{\n{body}\n}}\n"


def rename_one(code):
    """Consistently rename one identifier of the body to a fresh name (Type-2 change)."""
    ids = [t for t in re.findall(r"[A-Za-z_]\w*", code) if t in VOC]
    old = ids[0]
    return re.sub(rf"\b{old}\b", old + "_renamed", code)


def ws_change(code):
    return code.replace("\n    ", "\n\t\t").replace(" = ", "  =  ")


def real_change(code):
    lines = code.split("\n")
    lines.insert(3, f"    if ({R.choice(VOC)} == NULL) {{ return -{R.randrange(1, 9)}; }}")
    return "\n".join(lines)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def build(D):
    # ---------------- DiverseVul: fix commits with planted pair types
    kinds = [k for k, n in PLANT.items() for _ in range(n)]
    R.shuffle(kinds)
    dv, idx, fn = [], 0, 0
    commits = [f"c{c:04d}" for c in range(40)]
    for kind in kinds:
        c = R.choice(commits)
        proj = f"proj{int(c[1:]) % 8}"
        name = f"fn_{fn}"; fn += 1
        v = func(name)
        p = {"identical": v, "ws": ws_change(v), "rename": rename_one(v), "real": real_change(v)}[kind]
        for code, lab in ((v, 1), (p, 0)):
            dv.append({"idx": idx, "func": code, "target": lab, "project": proj, "commit_id": c, "kind": kind})
            idx += 1
    for _ in range(320):                                   # unchanged functions of the same commits
        c = R.choice(commits)
        dv.append({"idx": idx, "func": func(f"fn_{fn}"), "target": 0, "project": f"proj{int(c[1:]) % 8}",
                   "commit_id": c, "kind": "other"}); idx += 1; fn += 1
    for _ in range(60):                                    # near-duplicate clones across commits -> families
        src = R.choice(dv[:300])
        c = R.choice(commits)
        code = re.sub(r"int (\w+)\(", r"int \1_clone(", src["func"], count=1)   # new name: forms no pair
        dv.append({"idx": idx, "func": code.replace("return", "return  ", 1) + "\n", "target": R.randrange(2),
                   "project": f"proj{int(c[1:]) % 8}", "commit_id": c, "kind": "clone"}); idx += 1
    for r in dv:
        u = R.random()
        r["split"] = "train" if u < 0.7 else ("valid" if u < 0.8 else "test")
    write_jsonl(os.path.join(D, "data/diversevul/diversevul.jsonl"), dv)

    # ---------------- PrimeVul: planted overlap with DiverseVul
    pv, plan = [], {"raw": 40, "ws": 25, "rename": 20, "commit_only": 15, "new": 200}
    pool = R.sample([r for r in dv if r["kind"] in ("other", "real")], 100)
    k = 0
    for kind, n in plan.items():
        for _ in range(n):
            if kind == "new":
                code, c, lab = func(f"pv_new_{k}"), f"pc{k}", R.randrange(2)
            else:
                s = pool.pop()
                code = {"raw": s["func"], "ws": ws_change(s["func"]), "rename": rename_one(s["func"]),
                        "commit_only": func(re.search(r"int (\w+)\(", s["func"]).group(1))}[kind]
                c, lab = s["commit_id"], s["target"]
            u = R.random()
            pv.append({"idx": f"pv{k}", "func": code, "target": lab, "project": "pp", "commit_id": c,
                       "split": "train" if u < 0.6 else ("valid" if u < 0.7 else "test"), "kind": kind})
            k += 1
    write_jsonl(os.path.join(D, "data/primevul/primevul_all.jsonl"), pv)

    # ---------------- Devign: small corpus with a few clones
    dev = []
    for i in range(300):
        code = func(f"d_{i}") if i < 260 else dev[i - 260]["func"].replace(";", " ;", 2)
        u = R.random()
        dev.append({"id": i, "func": code, "target": R.randrange(2), "project": f"q{i % 2}",
                    "split": "train" if u < 0.8 else ("valid" if u < 0.9 else "test")})
    write_jsonl(os.path.join(D, "data/devign/devign_all.jsonl"), dev)

    # ---------------- default audit outputs (real runs of clone_family_audit.py)
    for c, path, idf in (("diversevul", "data/diversevul/diversevul.jsonl", "idx"),
                         ("primevul", "data/primevul/primevul_all.jsonl", "idx"),
                         ("devign", "data/devign/devign_all.jsonl", "id")):
        subprocess.run([sys.executable, os.path.join(VD, "clone_family_audit.py"), "--data", os.path.join(D, path),
                        "--id-field", idf, "--report-every", "0", "--out", os.path.join(D, f"{c}_audit")],
                       check=True, capture_output=True)
    ids = [str(r["idx"]) for r in dv]
    by = {str(r["idx"]): r for r in dv}
    for name, key in (("projectdisjoint", "project"), ("commitdisjoint", "commit_id")):
        groups = sorted({r[key] for r in dv})
        R.shuffle(groups)
        part = {g: ("test" if i % 5 == 0 else ("valid" if i % 5 == 1 else "train")) for i, g in enumerate(groups)}
        json.dump({i: part[by[i][key]] for i in ids},
                  open(os.path.join(D, f"diversevul_audit_splitfile_{name}_seed42.json"), "w"))

    # ---------------- fake predictions per arm and seed
    arms = {"random_control": ("diversevul_audit_splitfile_randomctrl_theta0.7.json", "rq2_diversevul_phase2", (4, 5)),
            "family_disjoint": ("diversevul_audit_splitfile_theta0.7.json", "rq2_diversevul_phase2", (4, 5)),
            "project_disjoint": ("diversevul_audit_splitfile_projectdisjoint_seed42.json", "rq2_diversevul_phase2",
                                 (1, 2, 3)),
            "commit_disjoint": ("diversevul_audit_splitfile_commitdisjoint_seed42.json", "rq2_diversevul_phase3",
                                (1, 2, 3))}
    for arm, (sf, sub, seeds) in arms.items():
        split = json.load(open(os.path.join(D, sf)))
        for s in seeds:
            rr = random.Random(hash((arm, s)) % 1000)
            pay = {}
            for part in ("test", "valid"):
                pid = [i for i in ids if split.get(i) == part]
                tg = [by[i]["target"] for i in pid]
                pay[part] = {"id": pid, "target": tg,
                             "prob": [round(min(1, max(0, 0.3 * t + rr.random() * 0.7)), 6) for t in tg]}
            os.makedirs(os.path.join(D, sub), exist_ok=True)
            with gzip.open(os.path.join(D, sub, f"codebert-base_{arm}_seed{s}.preds.json.gz"), "wt") as fh:
                json.dump(pay, fh)
    return dv, pv, arms


def sh(script, D, ok=True):
    env = {**os.environ, "D": D, "PY": sys.executable.replace("\\", "/"), "VD": VD.replace("\\", "/"),
           "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(["bash", os.path.join(VD, script)], env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if ok and r.returncode:
        sys.exit(f"{script} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r


def pairs_of(rows, split):
    g = defaultdict(lambda: ([], []))
    for r in rows:
        i = str(r["idx"])
        if split.get(i) == "test":
            m = re.search(r"([A-Za-z_~][A-Za-z0-9_:~]*)\s*\(", r["func"])
            if r["commit_id"] and m:
                g[(r["commit_id"], m.group(1))][0 if r["target"] == 1 else 1].append(i)
    return [(v[0], p[0]) for v, p in g.values() if len(v) == 1 and len(p) == 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep")
    a = ap.parse_args()
    D = (a.keep or tempfile.mkdtemp(prefix="cpu_checks_")).replace("\\", "/")
    if os.path.exists(D):
        shutil.rmtree(D)
    os.makedirs(D)
    print(f"building synthetic disk in {D}")
    dv, pv, arms = build(D)
    by = {str(r["idx"]): r for r in dv}
    P2 = os.path.join(D, "rq2_diversevul_phase2")

    print("1. missing input -> loud failure")
    keep = os.path.join(D, "x.tmp")
    os.replace(os.path.join(D, "diversevul_audit_families_theta0.7.json"), keep)
    r = sh("run_cpu_checks.sh", D, ok=False)
    check("exit code with missing input", r.returncode, 1)
    check("message names the missing file", "missing input" in r.stdout and "families_theta0.7" in r.stdout, True)
    os.replace(keep, os.path.join(D, "diversevul_audit_families_theta0.7.json"))

    print("2. full run, then re-run skips")
    sh("run_cpu_checks.sh", D)
    r = sh("run_cpu_checks.sh", D)
    check("re-run: steps skipped", r.stdout.count("SKIP (done)"), 3)

    print("3. A4 identical pairs")
    rep = json.load(open(os.path.join(P2, "pair_identity_check.json")))["scopes"]
    corp = rep["corpus"]
    check("corpus pairs", corp["pairs"], sum(PLANT.values()))
    check("corpus raw-identical", corp["identical"]["raw"], PLANT["identical"])
    check("corpus ws-identical", corp["identical"]["ws"], PLANT["identical"] + PLANT["ws"])
    check("corpus norm-identical", corp["identical"]["norm"], PLANT["identical"] + PLANT["ws"] + PLANT["rename"])
    check("corpus any", corp["identical"]["any"], PLANT["identical"] + PLANT["ws"] + PLANT["rename"])
    noid = json.load(open(os.path.join(P2, "paired_eval_noidentical.json")))
    env = {**os.environ, "PYTHONPATH": VD, "PYTHONIOENCODING": "utf-8"}
    subprocess.run([sys.executable, os.path.join(VD, "paired_eval.py"), "--disk", D], check=True,
                   capture_output=True, env=env)                 # default run of the current version
    default = json.load(open(os.path.join(P2, "paired_eval.json")))
    for arm, (sf, sub, seeds) in arms.items():
        split = json.load(open(os.path.join(D, sf)))
        prs = pairs_of(dv, split)
        ident = [pr for pr in prs if normalize_tokens(by[pr[0]]["func"]) == normalize_tokens(by[pr[1]]["func"])
                 or by[pr[0]]["func"] == by[pr[1]]["func"]
                 or re.sub(r"\s+", "", by[pr[0]]["func"]) == re.sub(r"\s+", "", by[pr[1]]["func"])]
        check(f"{arm} test pairs", rep[arm]["pairs"], len(prs))
        check(f"{arm} identical (any)", rep[arm]["identical"]["any"], len(ident))
        if arm not in noid["arms"]:
            continue
        check(f"{arm} excluded in paired_eval_noidentical", noid["arms"][arm]["identical_pairs_excluded"], len(ident))
        check(f"{arm} pairs kept", noid["arms"][arm]["pairs"], len(prs) - len(ident))
        check(f"{arm} default paired_eval pairs unchanged", default["arms"][arm]["pairs"], len(prs))
        kept = [pr for pr in prs if pr not in ident]
        ra = []
        for s in seeds:
            with gzip.open(os.path.join(D, sub, f"codebert-base_{arm}_seed{s}.preds.json.gz"), "rt") as fh:
                pr = json.load(fh)
            sc = dict(zip(pr["test"]["id"], pr["test"]["prob"]))
            ra.append(np.mean([(sc[v] > sc[p]) + 0.5 * (sc[v] == sc[p]) for v, p in kept]))
        check(f"{arm} rank-acc without identical (brute force)", round(noid["arms"][arm]["mean"]["rank_acc"], 10),
              round(float(np.mean(ra)), 10))

    print("   paired_eval.py default output vs HEAD version")
    head = os.path.join(D, "paired_eval_head.py")
    open(head, "wb").write(subprocess.run(["git", "show", "HEAD:paired_eval.py"], cwd=VD, capture_output=True,
                                          check=True).stdout)
    shutil.copy(os.path.join(P2, "paired_eval.json"), os.path.join(D, "pe_new.json"))
    shutil.copy(os.path.join(P2, "paired_eval.md"), os.path.join(D, "pe_new.md"))
    subprocess.run([sys.executable, head, "--disk", D], check=True, capture_output=True, env=env)
    check("paired_eval.json byte-identical to HEAD", open(os.path.join(P2, "paired_eval.json"), "rb").read()
          == open(os.path.join(D, "pe_new.json"), "rb").read(), True)
    check("paired_eval.md byte-identical to HEAD", open(os.path.join(P2, "paired_eval.md"), "rb").read()
          == open(os.path.join(D, "pe_new.md"), "rb").read(), True)

    print("4. A3 corpus overlap")
    ov = json.load(open(os.path.join(D, "corpus_overlap", "primevul_diversevul_overlap.json")))["keys"]
    kinds = defaultdict(int)
    for r in pv:
        kinds[r["kind"]] += 1
    check("raw matches (planted raw)", ov["raw"]["primevul_matched"], kinds["raw"])
    check("norm matches (planted raw+ws+rename)", ov["norm"]["primevul_matched"],
          kinds["raw"] + kinds["ws"] + kinds["rename"])
    check("commit+name matches (planted raw+ws+rename+commit_only)", ov["commit"]["primevul_matched"],
          kinds["raw"] + kinds["ws"] + kinds["rename"] + kinds["commit_only"])
    dv_tr = {r["func"] for r in dv if r["split"] == "train"}
    pv_te = [r for r in pv if r["split"] == "test"]
    check("PrimeVul test in DiverseVul published train (raw, brute force)",
          ov["raw"]["primevul_test_in_diversevul_train"]["published"]["count"],
          sum(r["func"] in dv_tr for r in pv_te))
    dv_func = {r["func"] for r in dv}
    check("DiverseVul functions matched (raw, brute force)", ov["raw"]["diversevul_matched"],
          sum(r["func"] in {p["func"] for p in pv} for r in dv) if dv_func else 0)

    print("5. A5 Croft re-scoring")
    cr = json.load(open(os.path.join(P2, "croft_rescore.json")))["arms"]
    fam = json.load(open(os.path.join(D, "diversevul_audit_families_theta0.7.json")))
    for arm, (sf, sub, seeds) in arms.items():
        split = json.load(open(os.path.join(D, sf)))
        trf = {fam[i] for i, s in split.items() if s == "train"}
        full, clean = [], []
        for s in seeds:
            with gzip.open(os.path.join(D, sub, f"codebert-base_{arm}_seed{s}.preds.json.gz"), "rt") as fh:
                pr = json.load(fh)
            y, p = np.array(pr["test"]["target"]), np.array(pr["test"]["prob"])
            k = np.array([fam[i] not in trf for i in pr["test"]["id"]])
            full.append(average_precision_score(y, p)); clean.append(average_precision_score(y[k], p[k]))
        check(f"{arm} removed", cr[arm]["n_leaked_removed"], int((~k).sum()))
        check(f"{arm} PR-AUC full", round(cr[arm]["pr_auc_full"], 10), round(float(np.mean(full)), 10))
        check(f"{arm} PR-AUC without leaked", round(cr[arm]["pr_auc_without_leaked"], 10),
              round(float(np.mean(clean)), 10))
    check("family_disjoint removes nothing", cr["family_disjoint"]["n_leaked_removed"], 0)

    print("6. high-recall audit runner")
    sh("run_audit_highrecall.sh", D)
    O = os.path.join(D, "audit_highrecall")
    for c in ("devign", "diversevul", "primevul"):
        for suf in ("_hr_report.json", "_hr_report.md", "_hr_families_theta0.7.json", "_hr_label_theta0.7.json",
                    "_hr_label_theta0.7.md"):
            check(f"{c}{suf} written", os.path.exists(os.path.join(O, c + suf)), True)
        hr = json.load(open(os.path.join(O, f"{c}_hr_report.json")))
        df = json.load(open(os.path.join(D, f"{c}_audit_report.json")))
        check(f"{c} config.rename = blind, shingle_k = 3", (hr["config"].get("rename"), hr["config"]["shingle_k"]),
              ("blind", 3))
        check(f"{c} default report has no rename key", "rename" in df["config"], False)
    r = sh("run_audit_highrecall.sh", D)
    check("high-recall re-run: steps skipped", r.stdout.count("SKIP (done)"), 6)
    print(f"\nwork dir: {D}")
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): {FAILS}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
