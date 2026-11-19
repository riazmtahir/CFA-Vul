#!/usr/bin/env python3
"""test_audit_default_identical.py -- the --rename option leaves the default audit unchanged.

Builds a 2,000-function JSONL from real BigCloneBench Java functions (laptop only; BCB is at
$CFA_BCB_DIR): 600 functions taken as clone pairs from the
Type-1/2/3 pair files, so families exist, plus random others. Labels, splits (80/10/10),
projects and ids are assigned from a fixed seed. Then runs
  (a) the audit as it was before the option (git HEAD copy of clone_family_audit.py), and
  (b) the current clone_family_audit.py with no new flag,
both with --theta 0.7 0.8 0.9 --knn-baseline, and compares every output file byte for byte.
report.json carries wall-clock "runtime_seconds"; that one field is removed before comparing
it, and the test shows that two runs of (a) also differ only there.
Finally runs (c) --rename blind --shingle-k 3 to check the option works and differs.

Usage (repository root): python scripts/tests/test_audit_default_identical.py [--keep DIR]
"""
from __future__ import annotations

import argparse
import csv
import filecmp
import json
import os
import random
import subprocess
import sys
import tempfile
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BCB = os.environ.get("CFA_BCB_DIR", "data/BCB")
N, N_PAIRED, SEED = 2000, 600, 7


def make_sample(path):
    rnd = random.Random(SEED)
    pairs = []
    for f in ("type-1.csv", "type-2.csv", "type-3.csv"):
        rows = [r for r in csv.reader(open(os.path.join(BCB, f), encoding="utf-8")) if r and r[0].isdigit()]
        pairs += rnd.sample(rows, 150)
    want = []
    for a, b, *_ in pairs:
        for x in (int(a), int(b)):
            if x not in want:
                want.append(x)
    want = want[:N_PAIRED]
    with zipfile.ZipFile(os.path.join(BCB, "id2sourcecode.zip")) as z:
        names = {int(os.path.basename(n)[:-5]): n for n in z.namelist() if n.endswith(".java")}
        others = rnd.sample(sorted(set(names) - set(want)), N - len(want))
        ids = want + others
        rnd.shuffle(ids)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for i, x in enumerate(ids):
                u = rnd.random()
                fh.write(json.dumps({"idx": i, "func": z.read(names[x]).decode("utf-8", "replace"),
                                     "target": int(rnd.random() < 0.2),
                                     "split": "train" if u < 0.8 else ("valid" if u < 0.9 else "test"),
                                     "project": f"p{rnd.randrange(25)}"}) + "\n")


def run(script, data, out, extra=()):
    cmd = [sys.executable, script, "--data", data, "--theta", "0.7", "0.8", "0.9", "--knn-baseline",
           "--report-every", "0", "--out", out, *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(script))
    if r.returncode:
        sys.exit(f"FAILED: {' '.join(cmd)}\n{r.stderr}")


def outputs(prefix):
    d, b = os.path.dirname(prefix), os.path.basename(prefix)
    return sorted(f[len(b):] for f in os.listdir(d) if f.startswith(b))


def strip_runtime(path):
    rep = json.load(open(path, encoding="utf-8"))
    for t in rep["thresholds"]:
        t.pop("runtime_seconds", None)
    return rep


def compare(p1, p2):
    o1, o2 = outputs(p1), outputs(p2)
    assert o1 == o2, (o1, o2)
    diff = []
    for suf in o1:
        if not filecmp.cmp(p1 + suf, p2 + suf, shallow=False):
            diff.append(suf)
    return o1, diff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", help="keep outputs in this directory")
    a = ap.parse_args()
    work = a.keep or tempfile.mkdtemp(prefix="audit_identical_")
    os.makedirs(work, exist_ok=True)
    old = os.path.join(work, "clone_family_audit_head.py")
    src = subprocess.run(["git", "show", "HEAD:clone_family_audit.py"], cwd=ROOT, capture_output=True,
                         check=True).stdout
    open(old, "wb").write(src)
    data = os.path.join(work, "sample.jsonl")
    make_sample(data)
    new = os.path.join(ROOT, "clone_family_audit.py")
    run(old, data, os.path.join(work, "A1", "x"))
    run(old, data, os.path.join(work, "A2", "x"))
    run(new, data, os.path.join(work, "B", "x"))
    run(new, data, os.path.join(work, "C", "x"), ("--rename", "blind", "--shingle-k", "3"))

    files, d_aa = compare(os.path.join(work, "A1", "x"), os.path.join(work, "A2", "x"))
    _, d_ab = compare(os.path.join(work, "A1", "x"), os.path.join(work, "B", "x"))
    print(f"{len(files)} output files: {', '.join(files)}")
    print(f"HEAD vs HEAD, files that differ byte-wise: {d_aa}")
    print(f"HEAD vs current default, files that differ byte-wise: {d_ab}")
    # only report.json may differ (wall-clock runtime_seconds); whether it does depends on timing
    ok = set(d_ab) <= {"_report.json"} and set(d_aa) <= {"_report.json"}
    same = strip_runtime(os.path.join(work, "A1", "x_report.json")) == \
        strip_runtime(os.path.join(work, "B", "x_report.json"))
    print(f"report.json equal after removing runtime_seconds: {same}")
    rc = json.load(open(os.path.join(work, "C", "x_report.json"), encoding="utf-8"))
    rb = json.load(open(os.path.join(work, "B", "x_report.json"), encoding="utf-8"))
    print("default vs blind k=3: representatives "
          f"{rb['n_representatives']} vs {rc['n_representatives']}; families>1 at 0.7 "
          f"{rb['thresholds'][0]['n_families_size_gt1']} vs {rc['thresholds'][0]['n_families_size_gt1']}; "
          f"config.rename {rb['config'].get('rename')} vs {rc['config'].get('rename')}")
    blind_ok = rc["config"].get("rename") == "blind" and "rename" not in rb["config"] and \
        rc["thresholds"][0]["n_families_size_gt1"] >= rb["thresholds"][0]["n_families_size_gt1"]
    print(f"work dir: {work}")
    if ok and same and blind_ok:
        print("PASS: default output identical to HEAD (all files byte-identical except report.json's "
              "wall-clock runtime_seconds, which also differs between two HEAD runs)")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
