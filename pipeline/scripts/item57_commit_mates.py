#!/usr/bin/env python3
"""item57_commit_mates.py -- referee item 57: what does a fix commit share beyond the counterpart pair?

DESCRIPTIVE / EXPLORATORY, written 9 Nov 2026 after all H3 results (never pre-specified). CPU only.
Reads the corpus, split files and saved Phase 2 predictions; writes ONLY to --out (default
$CFA_DATA_DIR/item57/) and refuses any output directory that holds primary results.

For each split (published, random control, family-disjoint) and each test function t with commit c:
  counterpart_in_train   a training function with the same commit, the same name and the opposite
                         label (Eq. twin; same rule and name extraction as commit_leak_audit.py)
  vul_mate_in_train      a training function with the same commit, label 1 and a DIFFERENT name:
                         another vulnerable function changed by the same fix commit
  nonvul_mate_in_train   the same with label 0 (unchanged or patched functions of that commit)
Reported as a 2 x 2 table (counterpart x vulnerable commit-mate) for test positives and negatives.

Commit structure over the whole corpus: vulnerable functions per commit; share of commits and of
vulnerable functions in commits with >= 2 vulnerable functions.

With the saved predictions (random control and family-disjoint, seeds 4-5; test AND valid
probabilities of the selected model): recall of test positives and FPR of test negatives at the
validation max-F1 threshold, by vul_mate_in_train, within the stratum WITHOUT a counterpart in
training (so the commit-mate contrast is not the counterpart contrast of Table 10), mean over seeds,
95% percentile bootstrap over test functions (1,000 resamples, seed 0).

Qualitative example: the random-control commit with the most vulnerable functions that has at least
one in test and one in training; prints names, partitions, labels and lengths (no code).

USAGE (server)
  python pipeline/scripts/item57_commit_mates.py            # defaults below
  python scripts/item57_commit_mates.py --self-test                               # synthetic check
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

D = os.environ.get("CFA_DATA_DIR", "data")
PRIMARY = {"rq2_diversevul_final", "rq2_diversevul_phase2", "rq2_diversevul_phase3", "rq2_diversevul_phase4b",
           "rq2_devign_final", "family_label", "commit_leak", "theta_sens", "audit_highrecall", "direct_edges"}
NAME_RE = re.compile(r"([A-Za-z_~][A-Za-z0-9_:~]*)\s*\(")   # identical to commit_leak_audit.py


def func_name(code):
    m = NAME_RE.search(code)
    return m.group(1) if m else ""


def load(path, id_field):
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
            if not isinstance(o.get("func"), str) or not o["func"].strip():
                continue
            sp = str(o.get("split", "all"))
            rows.append({"id": str(o.get(id_field, n)), "label": 1 if o.get("target") == 1 else 0,
                         "commit": str(o.get("commit_id") or ""), "project": str(o.get("project") or ""),
                         "name": func_name(o["func"]), "chars": len(o["func"]),
                         "split": "valid" if sp in ("valid", "validation") else sp})
    return rows


def flags(rows, split_of):
    """Per test index: (counterpart_in_train, vul_mate_in_train, nonvul_mate_in_train)."""
    tr = defaultdict(list)                     # commit -> [(name, label)] of training functions
    for r, s in zip(rows, split_of):
        if s == "train" and r["commit"]:
            tr[r["commit"]].append((r["name"], r["label"]))
    out = {}
    for i, (r, s) in enumerate(zip(rows, split_of)):
        if s != "test":
            continue
        mates = tr.get(r["commit"], []) if r["commit"] else []
        cp = bool(r["name"]) and any(n == r["name"] and l == 1 - r["label"] for n, l in mates)
        vm = any(l == 1 and n != r["name"] for n, l in mates)
        nm = any(l == 0 and n != r["name"] for n, l in mates)
        out[i] = (cp, vm, nm)
    return out


def table(rows, fl):
    res = {}
    for lab, key in ((1, "test_positives"), (0, "test_negatives")):
        idx = [i for i in fl if rows[i]["label"] == lab]
        c = Counter((fl[i][0], fl[i][1]) for i in idx)
        n = len(idx)
        res[key] = {"n": n,
                    "counterpart_and_vul_mate": c[(True, True)], "counterpart_only": c[(True, False)],
                    "vul_mate_only": c[(False, True)], "neither": c[(False, False)],
                    "vul_mate_any_pct": round(100 * (c[(True, True)] + c[(False, True)]) / n, 2) if n else None,
                    "vul_mate_without_counterpart_pct": round(100 * c[(False, True)] / n, 2) if n else None,
                    "nonvul_mate_any_pct": round(100 * sum(fl[i][2] for i in idx) / n, 2) if n else None}
    return res


def max_f1_threshold(y, p):
    order = sorted(range(len(p)), key=lambda i: -p[i])
    P = sum(y); tp = fp = 0; best, thr = -1.0, 1.0
    for k, i in enumerate(order):
        tp += y[i]; fp += 1 - y[i]
        if k + 1 < len(order) and p[order[k + 1]] == p[i]:
            continue                            # ties: evaluate after the last equal score
        f1 = 2 * tp / (P + tp + fp) if P else 0.0
        if f1 > best:
            best, thr = f1, p[i]
    return thr


try:   # the rule used by every earlier analysis (finetune_vd.best_f1 via analyze_phase2)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analyze_phase2 import best_f1_threshold as THRESHOLD  # noqa: E402
except Exception:   # pragma: no cover - fallback without numpy/sklearn
    THRESHOLD = max_f1_threshold


def rate(vals):
    return sum(vals) / len(vals) if vals else None


def boot(groups_by_seed, B=1000, seed=0):
    """groups_by_seed: list over seeds of (a, b) lists of 0/1 outcomes; returns mean diff b - a and CI."""
    rng = random.Random(seed)
    def stat(gs):
        d = [rate(b) - rate(a) for a, b in gs if a and b]
        return sum(d) / len(d) if d else None
    point = stat(groups_by_seed)
    sims = []
    for _ in range(B):
        gs = [([rng.choice(a) for _ in a], [rng.choice(b) for _ in b]) for a, b in groups_by_seed]
        v = stat(gs)
        if v is not None:
            sims.append(v)
    sims.sort()
    ci = [sims[int(0.025 * len(sims))], sims[int(0.975 * len(sims)) - 1]] if sims else None
    return point, ci


def prediction_contrast(rows, fl, preds):
    """Within the no-counterpart stratum: recall (positives) and FPR (negatives) by vul_mate_in_train."""
    id2i = {rows[i]["id"]: i for i in fl}
    out = {}
    for lab, key in ((1, "recall_positives"), (0, "fpr_negatives")):
        per_seed, rates = [], []
        for pr in preds:
            thr = float(THRESHOLD(pr["valid"]["target"], pr["valid"]["prob"]))
            g = {False: [], True: []}
            for tid, y, p in zip(pr["test"]["id"], pr["test"]["target"], pr["test"]["prob"]):
                i = id2i.get(str(tid))
                if i is None or y != lab or fl[i][0]:
                    continue
                g[fl[i][1]].append(1 if p >= thr else 0)
            per_seed.append((g[False], g[True]))
            rates.append({"threshold": thr, "without_vul_mate": [rate(g[False]), len(g[False])],
                          "with_vul_mate": [rate(g[True]), len(g[True])]})
        diff, ci = boot(per_seed)
        out[key] = {"per_seed": rates, "with_minus_without": diff, "ci95": ci}
    return out


def commit_structure(rows):
    vul = Counter(r["commit"] for r in rows if r["label"] == 1 and r["commit"])
    n_c = len(vul); multi = [c for c, k in vul.items() if k >= 2]
    return {"commits_with_vulnerable_functions": n_c,
            "commits_with_2plus_vulnerable_pct": round(100 * len(multi) / n_c, 2) if n_c else None,
            "vulnerable_functions_in_2plus_commits_pct":
                round(100 * sum(vul[c] for c in multi) / sum(vul.values()), 2) if vul else None,
            "vulnerable_per_commit_max": max(vul.values()) if vul else 0,
            "vulnerable_per_commit_hist": dict(sorted(Counter(min(k, 10) for k in vul.values()).items()))}


def example(rows, split_of):
    by = defaultdict(list)
    for r, s in zip(rows, split_of):
        if r["commit"] and r["label"] == 1:
            by[r["commit"]].append((r, s))
    cands = [(len(v), c) for c, v in by.items() if any(s == "test" for _, s in v) and any(s == "train" for _, s in v)]
    if not cands:
        return None
    _, c = max(cands)
    allc = [(r, s) for r, s in zip(rows, split_of) if r["commit"] == c]
    return {"commit": c, "project": allc[0][0]["project"],
            "functions": sorted(({"name": r["name"], "label": r["label"], "split": s, "chars": r["chars"]}
                                 for r, s in allc), key=lambda d: (-d["label"], d["split"], d["name"]))}


def run(data, splits, preds_dir, out, id_field="idx"):
    if os.path.basename(os.path.normpath(out)) in PRIMARY or os.path.normpath(out) == os.path.normpath(D):
        sys.exit(f"refusing to write into {out}: choose a separate directory")
    os.makedirs(out, exist_ok=True)
    rows = load(data, id_field)
    arms = [("published", [r["split"] for r in rows])]
    for name, path in splits:
        a = json.load(open(path, encoding="utf-8"))
        arms.append((name, [{"validation": "valid"}.get(a.get(r["id"]), a.get(r["id"])) for r in rows]))
    res = {"label": "descriptive/exploratory (referee item 57), not pre-specified", "data": data,
           "functions": len(rows), "commit_structure": commit_structure(rows), "arms": {}}
    for name, sp in arms:
        fl = flags(rows, sp)
        entry = {"table": table(rows, fl)}
        files = sorted(f for f in os.listdir(preds_dir) if f.startswith(f"codebert-base_{name}_seed")
                       and f.endswith(".preds.json.gz")) if preds_dir and os.path.isdir(preds_dir) else []
        if files:
            preds = [json.load(gzip.open(os.path.join(preds_dir, f), "rt", encoding="utf-8")) for f in files]
            entry["prediction_files"] = files
            entry["predictions_no_counterpart_stratum"] = prediction_contrast(rows, fl, preds)
        if name == "random_control":
            entry["example_commit"] = example(rows, sp)
        res["arms"][name] = entry
    json.dump(res, open(os.path.join(out, "item57.json"), "w", encoding="utf-8"), indent=1)
    L = ["# Item 57: same-commit vulnerable functions beyond the counterpart (descriptive/exploratory)", "",
         f"Corpus: {data} ({len(rows):,} functions).", "",
         "## Commit structure", "", "```", json.dumps(res["commit_structure"], indent=1), "```", "",
         "## Test functions by counterpart and vulnerable commit-mate in training", "",
         "| split | subset | n | counterpart + vul. mate | counterpart only | vul. mate only | neither | any vul. mate % | vul. mate, no counterpart % | any non-vul. mate % |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, e in res["arms"].items():
        for sub, t in e["table"].items():
            L.append(f"| {name} | {sub} | {t['n']:,} | {t['counterpart_and_vul_mate']:,} | {t['counterpart_only']:,} | "
                     f"{t['vul_mate_only']:,} | {t['neither']:,} | {t['vul_mate_any_pct']} | "
                     f"{t['vul_mate_without_counterpart_pct']} | {t['nonvul_mate_any_pct']} |")
    L += ["", "## Predictions, stratum without a counterpart in training (validation max-F1 threshold)", ""]
    for name, e in res["arms"].items():
        if "predictions_no_counterpart_stratum" in e:
            for k, v in e["predictions_no_counterpart_stratum"].items():
                L.append(f"- {name}, {k}: with minus without vulnerable commit-mate = {v['with_minus_without']}, "
                         f"95% CI {v['ci95']}; per seed {v['per_seed']}")
    ex = res["arms"].get("random_control", {}).get("example_commit")
    if ex:
        L += ["", "## Example commit (random control)", "", f"commit `{ex['commit']}`, project `{ex['project']}`", "",
              "| function | label | split | chars |", "|---|---|---|---|"]
        L += [f"| {f['name']} | {f['label']} | {f['split']} | {f['chars']} |" for f in ex["functions"]]
    open(os.path.join(out, "item57.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    return res


def self_test():
    """Synthetic corpus: planted counts must equal a brute-force recount; output-dir guard must hold."""
    import tempfile
    T = tempfile.mkdtemp()
    rng = random.Random(1)
    rows, split = [], {}
    for c in range(60):
        k = rng.randint(1, 4)
        for j in range(k):
            for lab in (1, 0):
                i = len(rows)
                rows.append({"idx": i, "func": f"int f{c}_{j}(int a) {{ return {lab}; }}", "target": lab,
                             "commit_id": f"c{c}", "project": f"p{c % 5}", "split": "train"})
                split[str(i)] = rng.choice(["train", "train", "valid", "test"])
    data = os.path.join(T, "d.jsonl"); open(data, "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
    sp = os.path.join(T, "rc.json"); json.dump(split, open(sp, "w"))
    res = run(data, [("random_control", sp)], None, os.path.join(T, "out"))
    tab = res["arms"]["random_control"]["table"]["test_positives"]
    # brute force
    want = Counter()
    for i, r in enumerate(rows):
        if split[str(i)] != "test" or r["target"] != 1:
            continue
        nm = func_name(r["func"])
        tr = [(func_name(o["func"]), o["target"]) for k, o in enumerate(rows)
              if split[str(k)] == "train" and o["commit_id"] == r["commit_id"]]
        want[(any(n == nm and l == 0 for n, l in tr), any(l == 1 and n != nm for n, l in tr))] += 1
    got = Counter({(True, True): tab["counterpart_and_vul_mate"], (True, False): tab["counterpart_only"],
                   (False, True): tab["vul_mate_only"], (False, False): tab["neither"]})
    assert +got == +want, (got, want)
    # prediction path: two synthetic seeds; every test positive WITH a vulnerable commit-mate scores 0.9,
    # every other test function 0.05, so recall(with) - recall(without) must be exactly 1 in the stratum
    pd_dir = os.path.join(T, "preds"); os.makedirs(pd_dir)
    fl = flags([dict(r, label=r["target"], commit=r["commit_id"], name=func_name(r["func"])) for r in rows],
               [split[str(i)] for i in range(len(rows))])
    for s in (4, 5):
        te = [i for i in range(len(rows)) if split[str(i)] == "test"]
        va = [i for i in range(len(rows)) if split[str(i)] == "valid"]
        prob = [0.9 if rows[i]["target"] == 1 and fl[i][1] else 0.05 for i in te]
        json.dump({"test": {"id": [str(i) for i in te], "target": [rows[i]["target"] for i in te], "prob": prob},
                   "valid": {"id": [str(i) for i in va], "target": [rows[i]["target"] for i in va],
                             "prob": [0.9 if rows[i]["target"] == 1 else 0.05 for i in va]}},
                  gzip.open(os.path.join(pd_dir, f"codebert-base_random_control_seed{s}.preds.json.gz"), "wt"))
    res = run(data, [("random_control", sp)], pd_dir, os.path.join(T, "out2"))
    rc = res["arms"]["random_control"]["predictions_no_counterpart_stratum"]["recall_positives"]
    assert abs(rc["with_minus_without"] - 1.0) < 1e-9, rc
    try:
        run(data, [("random_control", sp)], None, os.path.join(T, "rq2_diversevul_phase2"))
        raise AssertionError("guard did not fire")
    except SystemExit:
        pass
    y = [1, 1, 0, 0, 1]; p = [0.9, 0.8, 0.7, 0.2, 0.1]          # F1: 0.8 at 0.8 > 0.75 at 0.1
    assert max_f1_threshold(y, p) == 0.8, max_f1_threshold(y, p)
    assert abs(float(THRESHOLD(y, p)) - 0.8) < 1e-9, THRESHOLD(y, p)
    print("SELF-TEST PASSED: 2x2 counts equal a brute-force recount; output guard works; max-F1 threshold ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=f"{D}/data/diversevul/diversevul.jsonl")
    ap.add_argument("--rc", default=f"{D}/diversevul_audit_splitfile_randomctrl_theta0.7.json")
    ap.add_argument("--fd", default=f"{D}/diversevul_audit_splitfile_theta0.7.json")
    ap.add_argument("--preds-dir", default=f"{D}/rq2_diversevul_phase2")
    ap.add_argument("--out", default=f"{D}/item57")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return 0
    for f in (a.data, a.rc, a.fd):
        if not os.path.exists(f):
            sys.exit(f"!!! missing input: {f}")
    run(a.data, [("random_control", a.rc), ("family_disjoint", a.fd)], a.preds_dir, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
