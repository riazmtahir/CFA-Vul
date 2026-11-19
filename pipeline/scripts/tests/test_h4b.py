#!/usr/bin/env python3
"""test_h4b.py -- synthetic tests of the H4b preparation (laptop only; no GPU, no real data).

1. direct_clone_edges.py vs brute force. A 3,000-function corpus with planted near-duplicates at
   several similarity levels across train and test. Brute force = every test x train pair, Jaccard
   of the SETS of blind-identifier 3-grams (tuples, no hashing; min(k, n)-grams for short
   functions, as shingle_ids does), computed independently of the script.
   * --exhaustive: the leaked test functions, the per-test train lists and the train-hit list equal
     brute force exactly.
   * LSH (default): no false positives (every listed pair has brute J >= 0.7); every brute pair with
     J >= 0.9 is found; recall in [0.7, 0.9) is printed.
   * the summary's leak rates (all / positives) equal a recount from the lists.
2. make_decon_splits.py: decon removes exactly the hit list; the size control matches the removal
   in every label x length-decile stratum (recounted independently), also for a skewed removal of
   short functions; the control (default pool = train minus the decon list) holds no decon-listed
   function; a stratum short of candidates borrows exactly the shortfall from the nearest decile of
   the same label; valid and test unchanged; seed 42 reproducible.
3. precheck_h4b.py on a 20,000-function label-only corpus: PASS at a 10% removal; FAIL (exit 3) at
   45% removal, at < 1,000 positives left, at borrowing > 5% of the removal, and when the size
   control is tampered with; composition (length, projects, CWEs) reported; exit 2 on a missing input.
4. select_h4_design.py: all nine combinations of PASS / FAIL / missing give the §13.8 precedence.
5. analyze_phase4b.py: the three §13.8 verdicts and the "practically negligible" qualifier.
Usage (repository root): python scripts/tests/test_h4b.py [--keep DIR]
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
from collections import Counter

VD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, VD)
sys.path.insert(0, os.path.join(VD, "scripts"))
from clone_family_audit import C_KEYWORDS, PRESERVED_CALLS, blind_tokens  # noqa: E402

PY = sys.executable
FAILS = []


def check(name, got, want):
    ok = got == want
    show = (lambda x: f"<set of {len(x)}>" if isinstance(x, set) and len(x) > 8 else x)
    print(f"  {'ok ' if ok else 'FAIL'} {name}: {show(got)}" + ("" if ok else f" (want {show(want)})"))
    if not ok:
        FAILS.append(name)


def run(args, ok=(0,)):
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if r.returncode not in ok:
        sys.exit(f"FAILED ({r.returncode}): {' '.join(map(str, args))}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r


KEEP = sorted(C_KEYWORDS) + sorted(PRESERVED_CALLS) + ["+", "-", "*", "==", "<", ">", "&", "|", "->", "!"]


def func(rng, name):
    st = [" ".join(rng.choice(KEEP) if rng.random() < 0.6 else f"v{rng.randrange(300)}"
                   for _ in range(rng.randrange(4, 9))) + ";" for _ in range(rng.randrange(8, 16))]
    return f"int {name}(int a)\n{{\n    " + "\n    ".join(st) + "\n}\n"


def mutate(rng, code, n_edits):
    lines = code.split("\n")
    for _ in range(n_edits):
        j = rng.randrange(2, len(lines) - 2)
        lines[j] = "    " + " ".join(rng.choice(KEEP) for _ in range(rng.randrange(3, 7))) + ";"
    return "\n".join(lines)


def grams(code, k=3):
    t = blind_tokens(code)
    if not t:
        return set()
    kk = min(k, len(t))
    return {tuple(t[i:i + kk]) for i in range(len(t) - kk + 1)}


def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def edges_case(base):
    rng = random.Random(9)
    rows = []
    for i in range(3000):
        if i >= 2400 and rng.random() < 0.5:                  # test-side near-duplicate of a train function
            src = rows[rng.randrange(2000)]["func"]
            code = mutate(rng, src, rng.choice([0, 1, 2, 3, 5]))
        else:
            code = func(rng, f"f{i}")
        rows.append({"idx": i, "func": code, "target": int(rng.random() < 0.2),
                     "split": "train" if i < 2000 else ("valid" if i < 2400 else "test")})
    D = os.path.join(base, "edges")
    os.makedirs(D)
    data = os.path.join(D, "c.jsonl")
    with open(data, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    alt = {str(r["idx"]): ("test" if r["idx"] % 7 == 0 else "train") for r in rows}   # a second split
    json.dump(alt, open(os.path.join(D, "alt.json"), "w"))
    G = {r["idx"]: grams(r["func"]) for r in rows}
    brute = {}
    for name, part in (("published", {r["idx"]: r["split"] for r in rows}),
                       ("alt", {r["idx"]: alt[str(r["idx"])] for r in rows})):
        te = [i for i, p in part.items() if p == "test"]
        tr = [i for i, p in part.items() if p == "train"]
        brute[name] = {t: {x: jac(G[t], G[x]) for x in tr if jac(G[t], G[x]) >= 0.7} for t in te}
    for mode in ("exhaustive", "lsh"):
        out = os.path.join(D, mode)
        run([PY, os.path.join(VD, "scripts", "direct_clone_edges.py"), "--data", data, "--splitfile",
             f"alt={os.path.join(D, 'alt.json')}", "--report-every", "0", "--out", out]
            + (["--exhaustive"] if mode == "exhaustive" else []))
        summ = json.load(open(out + "_summary.json"))
        for name in ("published", "alt"):
            lists = {}
            with gzip.open(f"{out}_{name}_edges.jsonl.gz", "rt", encoding="utf-8") as fh:
                for line in fh:
                    o = json.loads(line)
                    lists[int(o["test"])] = {int(x) for x in o["train"]}
            b = {t: set(v) for t, v in brute[name].items() if v}
            hits = {int(x) for x in json.load(open(f"{out}_{name}_train_hits.json"))}
            if mode == "exhaustive":
                check(f"[{mode}] {name}: leaked test functions == brute force", set(lists), set(b))
                check(f"[{mode}] {name}: per-test train lists == brute force", lists == b, True)
                check(f"[{mode}] {name}: train-hit list == brute force", hits, set().union(*b.values()) if b else set())
            else:
                fp = sum(1 for t, xs in lists.items() for x in xs if brute[name][t].get(x, 0) < 0.7)
                check(f"[{mode}] {name}: false positives", fp, 0)
                strong = {(t, x) for t, v in brute[name].items() for x, j in v.items() if j >= 0.9}
                found = {(t, x) for t, xs in lists.items() for x in xs}
                check(f"[{mode}] {name}: pairs with J >= 0.9 all found", len(strong - found), 0)
                mid = {(t, x) for t, v in brute[name].items() for x, j in v.items() if j < 0.9}
                print(f"     recall for 0.7 <= J < 0.9: {len(mid & found)}/{len(mid)}")
            s = summ["splits"][name]
            check(f"[{mode}] {name}: summary leak count == lists", s["direct_leaked_test"], len(lists))
            lab = {r["idx"]: r["target"] for r in rows}
            check(f"[{mode}] {name}: summary positive leak count == lists", s["direct_leaked_test_pos"],
                  sum(lab[t] for t in lists))
    return D


def labels_corpus(D, n=20000, pos_rate=0.12, seed=4):
    rng = random.Random(seed)
    rows = [{"idx": i, "func": "x" * rng.randrange(10, 400), "target": int(rng.random() < pos_rate),
             "project": f"p{rng.randrange(30)}", "cwe": [f"CWE-{rng.choice([119, 125, 787, 476, 416])}"]}
            for i in range(n)]
    os.makedirs(os.path.join(D, "data", "diversevul"), exist_ok=True)
    with open(os.path.join(D, "data", "diversevul", "diversevul.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    ref = {str(r["idx"]): ("train" if r["idx"] % 10 < 8 else ("valid" if r["idx"] % 10 == 8 else "test")) for r in rows}
    json.dump(ref, open(os.path.join(D, "diversevul_audit_splitfile_commitdisjoint_seed42.json"), "w"))
    return rows, ref


def write_hits(D, hits):
    os.makedirs(os.path.join(D, "direct_edges"), exist_ok=True)
    pre = os.path.join(D, "direct_edges", "diversevul_hr_direct")
    json.dump(sorted(hits), open(pre + "_commit_disjoint_train_hits.json", "w"))
    json.dump({"splits": {"commit_disjoint": {"direct_leaked_test_pct": 1.0, "direct_leaked_test_pos_pct": 1.0}}},
              open(pre + "_summary.json", "w"))
    return pre + "_commit_disjoint_train_hits.json"


def decon(D, hits_file, seed=42, pool="train_minus_decon"):
    run([PY, os.path.join(VD, "scripts", "make_decon_splits.py"), "--data",
         os.path.join(D, "data", "diversevul", "diversevul.jsonl"), "--ref-splitfile",
         os.path.join(D, "diversevul_audit_splitfile_commitdisjoint_seed42.json"), "--train-hits", hits_file,
         "--seed", str(seed), "--control-pool", pool, "--out-dir", D])
    ld = lambda n: json.load(open(os.path.join(D, f"diversevul_audit_splitfile_{n}_seed{seed}.json")))
    return ld("commitdecon"), ld("commitsizectrl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep")
    a = ap.parse_args()
    base = (a.keep or tempfile.mkdtemp(prefix="h4b_")).replace("\\", "/")
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(base)

    print("1. direct edges vs brute force")
    edges_case(base)

    print("2. make_decon_splits.py")
    D = os.path.join(base, "gate").replace("\\", "/")
    rows, ref = labels_corpus(D)
    lab = {str(r["idx"]): r["target"] for r in rows}
    train = sorted(i for i, p in ref.items() if p == "train")
    rng = random.Random(1)
    hits = rng.sample(train, len(train) // 10)
    hf = write_hits(D, hits)
    dec, ctl = decon(D, hf)
    rem_d = {i for i in train if i not in dec}
    rem_s = {i for i in train if i not in ctl}
    check("decon removes exactly the hit list", rem_d, set(hits))
    check("size control: same number removed", len(rem_s), len(rem_d))
    check("size control: same positives removed", sum(lab[i] for i in rem_s), sum(lab[i] for i in rem_d))
    check("size control removes train functions only", all(ref[i] == "train" for i in set(ref) - set(ctl)), True)
    # strata: label x character-length decile on the commit_disjoint train set (independent recount)
    import numpy as np
    ln = {str(r["idx"]): len(r["func"]) for r in rows}
    edges = np.quantile(np.array([ln[i] for i in train], dtype=float), np.linspace(0.1, 0.9, 9))
    strat = lambda i: (lab[i], int(np.searchsorted(edges, ln[i], side="right")))
    cnt = lambda ids: Counter(strat(i) for i in ids)
    check("stratum counts match exactly (label x length decile)", cnt(rem_s) == cnt(rem_d), True)
    rep = json.load(open(os.path.join(D, "diversevul_h4b_splits_report.json")))
    check("registered pool: control holds no decon-listed function", len(rem_s & rem_d), 0)
    check("registered pool: report says train_minus_decon", rep["control_pool"], "train_minus_decon")
    check("10% removal: no borrowing needed", rep["borrowed"], 0)
    # skewed removal (short functions over-removed) is still matched stratum by stratum
    short = sorted((i for i in train if strat(i)[1] <= 1), key=lambda i: ln[i])
    hs_skew = rng.sample(short, int(0.4 * len(short)))   # < 50%, so the non-clone pool suffices
    dec_k, ctl_k = decon(D, write_hits(D, hs_skew))
    rd = {i for i in train if i not in dec_k}
    rs = {i for i in train if i not in ctl_k}
    check("skewed removal: stratum counts match", cnt(rs) == cnt(rd), True)
    check("skewed removal: control holds no decon-listed function", len(rs & rd), 0)
    m = lambda ids: sum(ln[i] for i in ids) / len(ids)
    print(f"     mean chars: decon-removed {m(rd):.0f}, control-removed {m(rs):.0f}, train {m(train):.0f}")
    # borrowing: control pool without the decon list, 80% of the shortest positive decile removed
    s0 = [i for i in train if strat(i) == (1, 0)]
    hs_b = sorted(rng.sample(s0, int(0.8 * len(s0))))
    dec_b, ctl_b = decon(D, write_hits(D, hs_b))
    rep_b = json.load(open(os.path.join(D, "diversevul_h4b_splits_report.json")))
    rd_b = {i for i in train if i not in dec_b}
    rs_b = {i for i in train if i not in ctl_b}
    need = len(hs_b) - (len(s0) - len(hs_b))
    check("borrowing: count = shortfall in the stratum", rep_b["borrowed"], need)
    check("borrowing: control avoids decon-listed functions", len(rs_b & rd_b), 0)
    check("borrowing: same label only", sum(lab[i] for i in rs_b), sum(lab[i] for i in rd_b))
    check("borrowing: taken from the nearest decile",
          {k: v for k, v in (cnt(rs_b) - cnt(rd_b)).items()}, {(1, 1): need})
    check("borrowing: whole stratum used before borrowing", cnt(rs_b)[(1, 0)], len(s0) - len(hs_b))
    for p in ("valid", "test"):
        part = {i for i, v in ref.items() if v == p}
        check(f"{p} unchanged in both arms", ({i for i, v in dec.items() if v == p} == part,
                                              {i for i, v in ctl.items() if v == p} == part), (True, True))
    hf = write_hits(D, hits)                       # the skew / borrow cases above overwrote it
    dec2, ctl2 = decon(D, hf)
    check("seed 42 reproducible", ctl2 == ctl, True)
    _, ctl7 = decon(D, hf, seed=7)
    check("another seed draws another control", ctl7 != ctl, True)
    os.remove(os.path.join(D, "diversevul_audit_splitfile_commitdecon_seed7.json"))
    os.remove(os.path.join(D, "diversevul_audit_splitfile_commitsizectrl_seed7.json"))

    print("3. precheck_h4b.py gate")
    pc = lambda ok=(0, 3): run([PY, os.path.join(VD, "scripts", "precheck_h4b.py"), "--disk", D], ok=ok)
    r = pc()
    check("10% removal -> PASS (exit 0)", r.returncode, 0)
    print("     ", json.load(open(os.path.join(D, "h4b_precheck.json")))["fails"])
    pos_train = [i for i in train if lab[i] == 1]
    for name, hs in (("45% removal", rng.sample(train, int(0.45 * len(train)))),
                     ("< 1,000 positives left", pos_train[:len(pos_train) - 900])):
        decon(D, write_hits(D, hs))
        r = pc()
        js = json.load(open(os.path.join(D, "h4b_precheck.json")))
        check(f"{name} -> FAIL (exit 3)", (r.returncode, js["verdict"]), (3, "FAIL"))
        print("     ", js["fails"])
    decon(D, write_hits(D, hs_b))
    r = pc()
    js = json.load(open(os.path.join(D, "h4b_precheck.json")))
    check("borrowing > 5% of removal -> FAIL (exit 3)",
          (r.returncode, any("borrowing" in f for f in js["fails"])), (3, True))
    check("precheck recounts the same borrowing", js["borrowed"], need)
    print("     ", js["fails"])
    decon(D, write_hits(D, hits))
    r = pc()
    js = json.load(open(os.path.join(D, "h4b_precheck.json")))
    comp = js["composition"]
    check("composition reported for decon / control / kept",
          all(k in comp and "top10_projects" in comp[k] and "top10_cwes" in comp[k] and "median_chars" in comp[k]
              for k in ("removed_by_decon", "removed_by_control", "kept")), True)
    sp = os.path.join(D, "diversevul_audit_splitfile_commitsizectrl_seed42.json")
    bad = json.load(open(sp))
    drop = next(i for i, v in bad.items() if v == "train" and lab[i] == 1)
    del bad[drop]
    json.dump(bad, open(sp, "w"))
    r = pc()
    check("tampered size control -> FAIL (exit 3)", r.returncode, 3)
    decon(D, write_hits(D, hits), pool="train")          # first-amendment pool: may overlap the list
    r = pc()
    js = json.load(open(os.path.join(D, "h4b_precheck.json")))
    check("control overlapping the decon list -> FAIL (exit 3)",
          (r.returncode, any("decon-listed" in f for f in js["fails"])), (3, True))
    r = run([PY, os.path.join(VD, "scripts", "precheck_h4b.py"), "--disk", os.path.join(base, "nowhere")], ok=(2,))
    check("missing input -> exit 2", r.returncode, 2)

    print("4. §13.8 precedence")
    import select_h4_design as S
    for v4 in ("PASS", "FAIL", "missing"):
        for v4b in ("PASS", "FAIL", "missing"):
            want = "H4" if v4 == "PASS" else ("H4b" if v4b == "PASS" else "none")
            check(f"select({v4}, {v4b})", S.select(v4, v4b), want)
    Dp = os.path.join(base, "prec")
    os.makedirs(Dp)
    json.dump({"verdict": "FAIL"}, open(os.path.join(Dp, "h4_precheck.json"), "w"))
    json.dump({"verdict": "PASS"}, open(os.path.join(Dp, "h4b_precheck.json"), "w"))
    run([PY, os.path.join(VD, "scripts", "select_h4_design.py"), "--disk", Dp])
    check("file-based: H4 FAIL + H4b PASS -> H4b",
          json.load(open(os.path.join(Dp, "h4_design.json")))["design"], "H4b")
    os.remove(os.path.join(Dp, "h4b_precheck.json"))
    run([PY, os.path.join(VD, "scripts", "select_h4_design.py"), "--disk", Dp])
    check("file-based: H4 FAIL + H4b missing -> none",
          json.load(open(os.path.join(Dp, "h4_design.json")))["design"], "none")

    print("5. analyze_phase4b.py rule")
    for name, deltas, want in (("below", [-0.03, -0.035, -0.032], "H4b SUPPORTED"),
                               ("includes", [-0.01, 0.012, 0.003], "H4b NOT SUPPORTED"),
                               ("above", [0.03, 0.028, 0.035], "H4b REVERSED"),
                               ("negligible", [-0.006, -0.0065, -0.0062], "practically negligible")):
        Dz = os.path.join(base, f"a4b_{name}")
        os.makedirs(os.path.join(Dz, "rq2_diversevul_phase4b"))
        for arm, add in (("commit_sizectrl", [0, 0, 0]), ("commit_decon", deltas)):
            for s, dd in zip((1, 2, 3), add):
                json.dump({"test": {"pr_auc": 0.17 + 0.001 * s + dd, "roc_auc": 0.7 + dd, "mcc": 0.2 + dd,
                                    "positive_rate": 0.057}, "length_only_floor": {"pr_auc": 0.14}},
                          open(os.path.join(Dz, "rq2_diversevul_phase4b", f"codebert-base_{arm}_seed{s}.json"), "w"))
        run([PY, os.path.join(VD, "analyze_phase4b.py"), "--disk", Dz])
        v = json.load(open(os.path.join(Dz, "rq2_diversevul_phase4b", "phase4b_analysis.json")))["verdict"]
        check(f"rule '{name}' -> {want}", want in v, True)
    print(f"\nwork dir: {base}")
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): {FAILS}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
