#!/usr/bin/env python3
"""test_h4.py -- synthetic tests of the H4 preparation (laptop only; no GPU, no real data).

1. Split builder + pre-check on a corpus WITHOUT a giant group (20,000 functions in 4,000
   fix commits, 300 planted cross-commit near-duplicates). High-recall families come from a
   real run of clone_family_audit.py --rename blind --shingle-k 3; the commit-disjoint
   reference split from make_project_split.py --group-field commit_id (as on the server).
   Checks: no commit and no high-recall family spans two partitions (brute force); every
   function assigned; precheck_h4.py exits 0 and reports zero sharing / twins / leaks; its
   length-only floor equals finetune_vd.length_floor on the same partitions.
2. The same with a planted giant chained family (a short template copied into 25% of the
   commits) -> the largest group exceeds 15% and precheck_h4.py exits 3.
3. Missing input -> precheck_h4.py exits 2; run_h4_precheck.sh exits 1 naming the file.
4. analyze_phase4.py on planted run JSONs: CI below 0 / includes 0 / above 0 give the three
   §13.7 verdicts; |Δ| <= 0.01 adds "practically negligible".
5. croft_rescore.py: every key of the HEAD version's JSON is unchanged; the new lift difference
   equals PR-AUC/pos recomputed by brute force, and its CI contains the point estimate.
6. run_h4_precheck.sh end to end (7 steps): on case 1's disk H4 passes and is selected (exit 0); a
   re-run skips the four build steps; on case 2's disk neither gate passes (fewer than 1,000 train
   positives for H4b) and the runner selects none (exit 3).
Usage (repository root): python scripts/tests/test_h4.py [--keep DIR]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score

VD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, VD)
sys.path.insert(0, os.path.join(VD, "scripts", "tests"))
FAILS = []
PY = sys.executable


def check(name, got, want):
    ok = got == want
    print(f"  {'ok ' if ok else 'FAIL'} {name}: {got}" + ("" if ok else f" (want {want})"))
    if not ok:
        FAILS.append(name)


def run(args, ok=(0,), env=None):
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8", **(env or {})})
    if r.returncode not in ok:
        sys.exit(f"FAILED ({r.returncode}): {' '.join(map(str, args))}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r


VOC = [f"v{i}" for i in range(400)]
from clone_family_audit import C_KEYWORDS, PRESERVED_CALLS  # noqa: E402
# tokens that survive blind normalisation, so random functions differ in their 3-shingles
KEEP = sorted(C_KEYWORDS) + sorted(PRESERVED_CALLS) + ["+", "-", "*", "/", "%", "==", "!=", "<", ">",
                                                      "&", "|", "^", "<<", ">>", "->", "[", "]", "!"]


def func(rng, name):
    st = []
    for _ in range(rng.randrange(8, 18)):
        toks = [rng.choice(KEEP) if rng.random() < 0.6 else rng.choice(VOC) for _ in range(rng.randrange(4, 9))]
        st.append(" ".join(toks) + ";")
    return f"int {name}(int {rng.choice(VOC)})\n{{\n    " + "\n    ".join(st) + "\n}\n"


def corpus(D, giant):
    rng = random.Random(5)
    rows, idx = [], 0
    pool = []
    for c in range(4000):
        cid = f"{c:05x}aa"
        proj = f"p{c % 60}"
        for k in range(5):
            nm = f"f{c}_{k}"
            if giant and c % 4 == 0 and k == 0:          # chained template: tiny edits, many commits
                code = f"int {nm}(int a)\n{{\n    return a + {rng.randrange(3)};\n}}\n"
                lab = 0
            elif k == 1 and rng.random() < 0.3 and pool:  # cross-commit near-duplicate
                code = rng.choice(pool).replace(";", " ;", 1)
                lab = 0
            else:
                code = func(rng, nm)
                lab = 1 if k == 0 and rng.random() < 0.3 else 0
            pool.append(code) if rng.random() < 0.02 else None
            rows.append({"idx": idx, "func": code, "target": lab, "project": proj, "commit_id": cid,
                         "split": "train"})
            idx += 1
    os.makedirs(os.path.join(D, "data", "diversevul"), exist_ok=True)
    with open(os.path.join(D, "data", "diversevul", "diversevul.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    dv = os.path.join(D, "data", "diversevul", "diversevul.jsonl")
    run([PY, os.path.join(VD, "clone_family_audit.py"), "--data", dv, "--report-every", "0",
         "--out", os.path.join(D, "diversevul_audit")])
    os.makedirs(os.path.join(D, "audit_highrecall"), exist_ok=True)
    run([PY, os.path.join(VD, "clone_family_audit.py"), "--data", dv, "--report-every", "0", "--rename", "blind",
         "--shingle-k", "3", "--out", os.path.join(D, "audit_highrecall", "diversevul_hr")])
    run([PY, os.path.join(VD, "make_project_split.py"), "--data", dv, "--group-field", "commit_id",
         "--ref-splitfile", os.path.join(D, "diversevul_audit_splitfile_theta0.7.json"), "--seed", "42",
         "--out", os.path.join(D, "diversevul_audit_splitfile_commitdisjoint_seed42")])
    return rows


def build_and_check(D, rows):
    out = os.path.join(D, "diversevul_audit_splitfile_commithrfamily_seed42")
    run([PY, os.path.join(VD, "scripts", "make_commit_family_split.py"),
         "--data", os.path.join(D, "data", "diversevul", "diversevul.jsonl"),
         "--hr-famfile", os.path.join(D, "audit_highrecall", "diversevul_hr_families_theta0.7.json"),
         "--ref-splitfile", os.path.join(D, "diversevul_audit_splitfile_commitdisjoint_seed42.json"),
         "--strict-famfile", os.path.join(D, "diversevul_audit_families_theta0.7.json"), "--seed", "42",
         "--out", out])
    sp = json.load(open(out + ".json"))
    hr = json.load(open(os.path.join(D, "audit_highrecall", "diversevul_hr_families_theta0.7.json")))
    check("every function assigned", len(sp), len(rows))
    by_c, by_f = defaultdict(set), defaultdict(set)
    for r in rows:
        i = str(r["idx"])
        by_c[r["commit_id"]].add(sp[i])
        by_f[hr[i]].add(sp[i])
    check("no commit in two partitions (brute force)", sum(len(v) > 1 for v in by_c.values()), 0)
    check("no high-recall family in two partitions (brute force)", sum(len(v) > 1 for v in by_f.values()), 0)
    return sp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep")
    a = ap.parse_args()
    base = (a.keep or tempfile.mkdtemp(prefix="h4_")).replace("\\", "/")
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(base)

    print("1. corpus without a giant group")
    D1 = os.path.join(base, "ok").replace("\\", "/")
    rows = corpus(D1, giant=False)
    build_and_check(D1, rows)
    r = run([PY, os.path.join(VD, "scripts", "precheck_h4.py"), "--disk", D1], ok=(0, 3))
    rep = json.load(open(os.path.join(D1, "h4_precheck.json")))
    check("precheck exit 0", r.returncode, 0)
    print("     fails:", rep["fails"])
    N = rep["splits"]["commit_hrfamily_disjoint"]
    check("same-commit sharing 0", N["test_sharing_a_commit_with_train"], 0)
    check("own twins 0", N["test_with_own_twin_in_train"], 0)
    check("hr leak any 0", N["hr_family_leak"]["any"], 0)
    check("strict leak 0", N["strict_family_leak"]["any"], 0)
    R = rep["splits"]["commit_disjoint"]
    check("reference split has some hr leak (so the check can fail)", R["hr_family_leak"]["any"] > 0, True)
    try:
        # finetune_vd needs torch/transformers; take length_floor and its helpers from its source
        import ast
        src = open(os.path.join(VD, "finetune_vd.py"), encoding="utf-8").read()
        keep = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)
                and n.name in ("length_floor", "evaluate", "best_f1", "vd_score")]
        ns = {}
        header = ("import numpy as np\n"
                  "from sklearn.linear_model import LogisticRegression\n"
                  "from sklearn.metrics import (average_precision_score, matthews_corrcoef, "
                  "precision_recall_curve, roc_auc_score, roc_curve)\n")
        exec(header + ast.unparse(ast.Module(keep, [])), ns)

        class finetune_vd:  # noqa: N801
            length_floor = staticmethod(ns["length_floor"])
        sp = json.load(open(os.path.join(D1, "diversevul_audit_splitfile_commithrfamily_seed42.json")))
        tr = [{"func": x["func"], "target": x["target"]} for x in rows if sp[str(x["idx"])] == "train"]
        te = [{"func": x["func"], "target": x["target"]} for x in rows if sp[str(x["idx"])] == "test"]
        check("length floor == finetune_vd.length_floor (4 dp)", round(N["length_floor_pr_auc"], 4),
              finetune_vd.length_floor(tr, te)["pr_auc"])
    except ImportError as e:
        print(f"  (skipped length-floor comparison: {e})")

    print("2. corpus with a planted giant chained family")
    D2 = os.path.join(base, "giant").replace("\\", "/")
    rows2 = corpus(D2, giant=True)
    build_and_check(D2, rows2)
    r = run([PY, os.path.join(VD, "scripts", "precheck_h4.py"), "--disk", D2], ok=(0, 3))
    rep2 = json.load(open(os.path.join(D2, "h4_precheck.json")))
    check("precheck exit 3", r.returncode, 3)
    check("largest group > 15% flagged", any("largest group" in f for f in rep2["fails"]), True)
    print(f"     largest group {rep2['largest_group_pct']:.1f}% of corpus")

    print("3. missing input")
    r = run([PY, os.path.join(VD, "scripts", "precheck_h4.py"), "--disk", os.path.join(base, "nowhere")], ok=(2,))
    check("precheck exit 2", r.returncode, 2)
    r = run(["bash", os.path.join(VD, "run_h4_precheck.sh")], ok=(1,),
            env={"D": os.path.join(base, "nowhere").replace("\\", "/"), "PY": PY.replace("\\", "/"),
                 "VD": VD.replace("\\", "/")})
    check("runner exit 1 with 'missing input'", (r.returncode, "missing input" in r.stdout), (1, True))

    print("4. analyze_phase4.py decision rule")
    import analyze_phase4
    for name, deltas, want in (("below", [-0.03, -0.035, -0.032], "H4 SUPPORTED"),
                               ("includes", [-0.01, 0.012, 0.003], "H4 NOT SUPPORTED"),
                               ("above", [0.03, 0.028, 0.035], "H4 REVERSED"),
                               ("negligible", [-0.006, -0.0065, -0.0062], "practically negligible")):
        Dz = os.path.join(base, f"a4_{name}")
        for arm, sub, add in (("commit_disjoint", "rq2_diversevul_phase3", [0, 0, 0]),
                              ("commit_hrfamily_disjoint", "rq2_diversevul_phase4", deltas)):
            os.makedirs(os.path.join(Dz, sub), exist_ok=True)
            for s, dd in zip((1, 2, 3), add):
                pr = 0.17 + 0.001 * s + dd
                json.dump({"test": {"pr_auc": pr, "roc_auc": 0.7 + dd, "mcc": 0.2 + dd, "positive_rate": 0.057},
                           "length_only_floor": {"pr_auc": 0.14}},
                          open(os.path.join(Dz, sub, f"codebert-base_{arm}_seed{s}.json"), "w"))
        r = run([PY, os.path.join(VD, "analyze_phase4.py"), "--disk", Dz])
        v = json.load(open(os.path.join(Dz, "rq2_diversevul_phase4", "phase4_analysis.json")))["verdict"]
        check(f"rule '{name}' -> {want}", want in v, True)
    check("decide() on the 'includes' case without negligible", "negligible" in analyze_phase4.decide(
        {"mean": 0.02, "ci": [-0.01, 0.05]})[1], False)

    print("5. croft_rescore.py lift CI")
    from test_cpu_checks import build
    D5 = os.path.join(base, "croft").replace("\\", "/")
    os.makedirs(D5)
    build(D5)
    head = os.path.join(base, "croft_head.py")
    open(head, "wb").write(subprocess.run(["git", "show", "HEAD:scripts/croft_rescore.py"], cwd=VD,
                                          capture_output=True, check=True).stdout)
    shutil.copy(head, os.path.join(VD, "scripts", "_croft_head_tmp.py"))   # HEAD copy must import from VD
    try:
        run([PY, os.path.join(VD, "scripts", "_croft_head_tmp.py"), "--disk", D5])
    finally:
        os.remove(os.path.join(VD, "scripts", "_croft_head_tmp.py"))
    old = json.load(open(os.path.join(D5, "rq2_diversevul_phase2", "croft_rescore.json")))
    run([PY, os.path.join(VD, "scripts", "croft_rescore.py"), "--disk", D5,
         "--out", os.path.join(D5, "rq2_diversevul_phase2", "croft_rescore_lift")])
    new = json.load(open(os.path.join(D5, "rq2_diversevul_phase2", "croft_rescore_lift.json")))
    same = all(new["arms"][arm][k] == v for arm, r in old["arms"].items() for k, v in r.items())
    check("every HEAD key unchanged", same, True)
    for arm, r in new["arms"].items():
        lf, lw = r["pr_auc_full"] / r["pos_rate_full"], r["pr_auc_without_leaked"] / r["pos_rate_without_leaked"]
        check(f"{arm} lift diff == brute force", round(r["lift_diff_without_minus_full"], 10), round(lw - lf, 10))
        lo, hi = r["lift_diff_ci95"]
        check(f"{arm} lift CI contains the point", lo - 1e-9 <= r["lift_diff_without_minus_full"] <= hi + 1e-9, True)
    check("default output path still croft_rescore.json (no _lift written by default)",
          os.path.exists(os.path.join(D5, "rq2_diversevul_phase2", "croft_rescore.json")), True)

    print("6. run_h4_precheck.sh end to end (case-1 disk + case-5 predictions)")
    for sub in ("rq2_diversevul_phase2", "rq2_diversevul_phase3"):
        shutil.copytree(os.path.join(D5, sub), os.path.join(D1, sub), dirs_exist_ok=True)
    # predictions of case 5 refer to case 5's ids; give case 1 its own split-consistent fake predictions
    sps = {"random_control": "diversevul_audit_splitfile_randomctrl_theta0.7.json",
           "family_disjoint": "diversevul_audit_splitfile_theta0.7.json",
           "project_disjoint": "diversevul_audit_splitfile_projectdisjoint_seed42.json",
           "commit_disjoint": "diversevul_audit_splitfile_commitdisjoint_seed42.json"}
    json.dump(json.load(open(os.path.join(D1, sps["random_control"]))),
              open(os.path.join(D1, sps["project_disjoint"]), "w"))         # any valid split will do here
    lab = {str(x["idx"]): x["target"] for x in rows}
    rng = random.Random(3)
    for arm, seeds, sub in (("random_control", (4, 5), "rq2_diversevul_phase2"),
                            ("family_disjoint", (4, 5), "rq2_diversevul_phase2"),
                            ("project_disjoint", (1, 2, 3), "rq2_diversevul_phase2"),
                            ("commit_disjoint", (1, 2, 3), "rq2_diversevul_phase3")):
        sp = json.load(open(os.path.join(D1, sps[arm])))
        for s in seeds:
            pay = {}
            for part in ("test", "valid"):
                ids = [i for i, v in sp.items() if v == part]
                pay[part] = {"id": ids, "target": [lab[i] for i in ids],
                             "prob": [round(0.3 * lab[i] + 0.7 * rng.random(), 6) for i in ids]}
            with gzip.open(os.path.join(D1, sub, f"codebert-base_{arm}_seed{s}.preds.json.gz"), "wt") as fh:
                json.dump(pay, fh)
    env = {"D": D1, "PY": PY.replace("\\", "/"), "VD": VD.replace("\\", "/")}
    r = run(["bash", os.path.join(VD, "run_h4_precheck.sh")], ok=(0, 3), env=env)
    check("runner exit 0 and selects H4", (r.returncode, "SELECTED: H4 (" in r.stdout), (0, True))
    for f in ("diversevul_audit_splitfile_commithrfamily_seed42_report.json", "h4_precheck.md",
              "direct_edges/diversevul_hr_direct_summary.md", "diversevul_audit_splitfile_commitdecon_seed42.json",
              "diversevul_audit_splitfile_commitsizectrl_seed42.json", "h4b_precheck.md",
              "rq2_diversevul_phase2/croft_rescore_lift.md", "h4_design.json", ".h4_prechecks_done"):
        check(f"{f} written", os.path.exists(os.path.join(D1, f)), True)
    summ = json.load(open(os.path.join(D1, "direct_edges", "diversevul_hr_direct_summary.json")))
    check("direct edges computed for five splits", sorted(summ["splits"]),
          sorted(["published", "random_control", "family_disjoint", "project_disjoint", "commit_disjoint"]))
    r = run(["bash", os.path.join(VD, "run_h4_precheck.sh")], ok=(0, 3), env=env)
    check("re-run skips the four build steps", r.stdout.count("SKIP (done)"), 4)
    env2 = {"D": D2, "PY": PY.replace("\\", "/"), "VD": VD.replace("\\", "/")}
    for sub in ("rq2_diversevul_phase2", "rq2_diversevul_phase3"):
        shutil.copytree(os.path.join(D1, sub), os.path.join(D2, sub), dirs_exist_ok=True)
    shutil.copy(os.path.join(D1, sps["project_disjoint"]), os.path.join(D2, sps["project_disjoint"]))
    r = run(["bash", os.path.join(VD, "run_h4_precheck.sh")], ok=(0, 1, 3), env=env2)
    des = json.load(open(os.path.join(D2, "h4_design.json")))
    check("giant case: H4 pre-check FAIL", des["h4_precheck"], "FAIL")
    check("giant case: H4b pre-check FAIL", des["h4b_precheck"], "FAIL")
    check("giant case: runner exit 3, selects none",
          (r.returncode, "SELECTED: none" in r.stdout), (3, True))
    print(f"\nwork dir: {base}")
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): {FAILS}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
