#!/usr/bin/env python3
"""m1_numbers.py -- referee item 13 (M1): post hoc sensitivity of the project-disjoint results to the
selection rule, from the rerun with finetune_vd.py --select-after-warmup (run_m1_sensitivity.sh).

POST HOC sensitivity (written after the review, 9-12 Nov 2026); the primary pre-specified results are unchanged.
Inputs (read only):
  M/rq2_diversevul_m1_postwarmup/codebert-base_project_disjoint_seed{1,2,3}_postwarmup.json   rerun (history[].valid,
      history[].test, best_epoch = post-warm-up choice, test = metrics of that choice)
  M/rq2_devign_m1_postwarmup/graphcodebert-base_family_disjoint_seed2_postwarmup.json         Devign rerun
  results/rq2_diversevul_phase2/codebert-base_project_disjoint_seed{1,2,3}.json                 Phase 2 originals
  results/rq2_diversevul_final/codebert-base_random_control_seed{1,2,3}.json                    Stage B random control
  results/rq2_diversevul_phase3/codebert-base_commit_disjoint_seed{1,2,3}.json                  Phase 3 commit-disjoint
  results/from_backup/rq2_devign_final/graphcodebert-base_family_disjoint_seed2.json             Devign original
  M = results/m1_postwarmup_1112 (default; --m1-dir to change)
a) per seed: epoch and test PR-AUC under the original rule (all epochs; replayed on the rerun's own validation
   curve with the selection functions of finetune_vd.py) and under the post-warm-up rule; paired difference
   post - original, mean and 95% t-interval (t_{0.975,2} = 4.303, as everywhere in the paper).
b) reproduction: rerun vs Phase 2 per-epoch validation PR-AUC (epochs both logged) and original-rule test PR-AUC;
   maximum absolute difference.
c) H2 and the commit -> project share with post-warm-up project values: Delta(random -> project) and
   Delta(commit -> project) paired by seed 1-3; rho = mean Delta(random -> commit) / mean Delta(random -> project).
   Random control: Stage B; commit-disjoint: Phase 3; project-disjoint: the rerun.
d) Devign GraphCodeBERT family-disjoint seed 2: both rules on the rerun, against the original run.
Writes analysis/scripts/out/m1_numbers.json. --self-test runs on a synthetic copy of this layout.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import shutil
import statistics as st
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = open(os.path.join(ROOT, "pipeline", "finetune_vd.py"), encoding="utf-8").read()
_ns: dict = {}
_mod = ast.parse(SRC)
exec(compile(ast.Module(body=[n for n in _mod.body if isinstance(n, ast.FunctionDef) and n.name in
                              {"first_eligible_epoch", "is_new_best", "should_stop", "replay_selection"}],
                        type_ignores=[]), "finetune_vd.py", "exec"), _ns)
replay = _ns["replay_selection"]
T975_2 = 4.303
# run settings recorded in every result JSON; the rerun must match the original on all of them
SETTINGS = ("arm", "model", "seed", "data", "splitfile", "epochs", "lr", "max_len", "class_weight", "batch_size",
            "n_train", "n_valid", "n_test", "patience", "max_epochs", "selected_on", "length_only_floor")


def tci(x):
    m = st.mean(x)
    h = T975_2 * st.stdev(x) / math.sqrt(len(x))
    return {"mean": m, "lo": m - h, "hi": m + h, "per_seed": x}


def load(*p):
    return json.load(open(os.path.join(*p), encoding="utf-8"))


def compute(root, m1):
    R = os.path.join(root, "results")
    out = {"label": "post hoc sensitivity (referee item 13); primary pre-specified results unchanged",
           "sources": {"random_control": "Stage B (rq2_diversevul_final), seeds 1-3",
                       "commit_disjoint": "Phase 3 (rq2_diversevul_phase3), seeds 1-3",
                       "project_disjoint_post_warmup": "M1 rerun (rq2_diversevul_m1_postwarmup), seeds 1-3",
                       "project_disjoint_original": "Phase 2 (rq2_diversevul_phase2), seeds 1-3"}}
    seeds = (1, 2, 3)
    a, b = {}, {"valid_abs_diff": [], "test_abs_diff_original_rule": []}
    post, orig_rerun, settings_diff = [], [], []
    for s in seeds:
        r = load(m1, "rq2_diversevul_m1_postwarmup", f"codebert-base_project_disjoint_seed{s}_postwarmup.json")
        p2 = load(R, "rq2_diversevul_phase2", f"codebert-base_project_disjoint_seed{s}.json")
        h = r["history"]
        va = [e["valid"]["pr_auc"] for e in h]
        te = [e["test"]["pr_auc"] for e in h]
        fe = r["first_eligible_epoch"]
        eo, _, complete = replay(va, r["patience"], r["max_epochs"], 1)
        assert complete, f"seed {s}: original rule not determinable on the rerun curve"
        ep = r["best_epoch"]
        assert replay(va, r["patience"], r["max_epochs"], fe)[0] == ep, f"seed {s}: post-warm-up replay != logged"
        assert abs(te[ep - 1] - r["test"]["pr_auc"]) < 1e-4, f"seed {s}: logged test != per-epoch test at best epoch"
        a[s] = {"epochs_run": len(h), "first_eligible_epoch": fe, "valid": va, "test": te,
                "original_rule_epoch": eo, "original_rule_test": te[eo - 1],
                "post_warmup_epoch": ep, "post_warmup_test": r["test"]["pr_auc"],
                "diff_post_minus_original": r["test"]["pr_auc"] - te[eo - 1]}
        post.append(r["test"]["pr_auc"]); orig_rerun.append(te[eo - 1])
        pv = [e["valid"]["pr_auc"] for e in p2["history"]]
        k = min(len(pv), len(va))
        b["valid_abs_diff"] += [abs(pv[i] - va[i]) for i in range(k)]
        b["test_abs_diff_original_rule"].append(abs(p2["test"]["pr_auc"] - te[eo - 1]) if eo == p2["best_epoch"] else None)
        a[s]["phase2_best_epoch"] = p2["best_epoch"]; a[s]["phase2_test"] = p2["test"]["pr_auc"]
        settings_diff += [{"run": f"project s{s}", "key": k, "phase2": p2.get(k), "rerun": r.get(k)}
                          for k in SETTINGS if p2.get(k) != r.get(k)]
    out["a_per_seed"] = a
    out["a_paired_post_minus_original"] = tci([a[s]["diff_post_minus_original"] for s in seeds])
    out["b_max_abs_valid_diff"] = max(b["valid_abs_diff"])
    td = [x for x in b["test_abs_diff_original_rule"] if x is not None]
    out["b_max_abs_test_diff_original_rule"] = max(td) if td else None
    out["b_settings_compared"] = list(SETTINGS)
    out["b_settings_differences"] = settings_diff
    out["b_note"] = ("rerun uses the same code path, split file, seeds and hyperparameters (b_settings_differences is "
                     "empty); differences come from non-deterministic GPU kernels under fp16 AMP")
    rc = [load(R, "rq2_diversevul_final", f"codebert-base_random_control_seed{s}.json")["test"]["pr_auc"] for s in seeds]
    cd = [load(R, "rq2_diversevul_phase3", f"codebert-base_commit_disjoint_seed{s}.json")["test"]["pr_auc"] for s in seeds]
    pj = [load(R, "rq2_diversevul_phase2", f"codebert-base_project_disjoint_seed{s}.json")["test"]["pr_auc"] for s in seeds]
    out["c_primary"] = {"random_to_project": tci([pj[i] - rc[i] for i in range(3)]),
                        "commit_to_project": tci([pj[i] - cd[i] for i in range(3)]),
                        "random_to_commit": tci([cd[i] - rc[i] for i in range(3)])}
    out["c_primary"]["rho"] = out["c_primary"]["random_to_commit"]["mean"] / out["c_primary"]["random_to_project"]["mean"]
    out["c_post_warmup"] = {"random_to_project": tci([post[i] - rc[i] for i in range(3)]),
                            "commit_to_project": tci([post[i] - cd[i] for i in range(3)])}
    out["c_post_warmup"]["rho"] = out["c_primary"]["random_to_commit"]["mean"] / out["c_post_warmup"]["random_to_project"]["mean"]
    out["c_post_warmup"]["relative_random_to_project"] = out["c_post_warmup"]["random_to_project"]["mean"] / st.mean(rc)
    dv = load(m1, "rq2_devign_m1_postwarmup", "graphcodebert-base_family_disjoint_seed2_postwarmup.json")
    do = load(R, "from_backup", "rq2_devign_final", "graphcodebert-base_family_disjoint_seed2.json")
    va = [e["valid"]["pr_auc"] for e in dv["history"]]; te = [e["test"]["pr_auc"] for e in dv["history"]]
    eo, _, comp = replay(va, dv["patience"], dv["max_epochs"], 1)
    out["d_settings_differences"] = [{"key": k, "original": do.get(k), "rerun": dv.get(k)}
                                     for k in SETTINGS if do.get(k) != dv.get(k)]
    out["d_devign"] = {"valid": va, "test": te, "original_rule_epoch": eo if comp else None,
                       "original_rule_test": te[eo - 1] if comp else None,
                       "post_warmup_epoch": dv["best_epoch"], "post_warmup_test": dv["test"]["pr_auc"],
                       "original_run": {"best_epoch": do["best_epoch"], "test": do["test"]["pr_auc"],
                                        "valid": [e["valid"]["pr_auc"] for e in do["history"]]}}
    return out


def self_test():
    """Synthetic layout: planted curves; checks the rules, the pairing and the H2 recomputation."""
    T = tempfile.mkdtemp()
    R = os.path.join(T, "results"); M = os.path.join(R, "m1")
    for d in ("rq2_diversevul_phase2", "rq2_diversevul_final", "rq2_diversevul_phase3",
              os.path.join("from_backup", "rq2_devign_final")):
        os.makedirs(os.path.join(R, d))
    for d in ("rq2_diversevul_m1_postwarmup", "rq2_devign_m1_postwarmup"):
        os.makedirs(os.path.join(M, d))
    shutil.copy(os.path.join(ROOT, "pipeline", "finetune_vd.py"), os.path.join(T, "finetune_vd.py"))
    curve = [0.20, 0.17, 0.18, 0.19, 0.185, 0.18]       # original rule: epoch 1; post-warm-up: epoch 4
    test = [0.15, 0.14, 0.16, 0.17, 0.16, 0.15]
    for s in (1, 2, 3):
        h = [{"epoch": i + 1, "valid": {"pr_auc": curve[i]}, "test": {"pr_auc": test[i] + 0.001 * s}} for i in range(6)]
        json.dump({"history": h, "best_epoch": 4, "test": {"pr_auc": test[3] + 0.001 * s}, "patience": 2, "max_epochs": 15,
                   "first_eligible_epoch": 2},
                  open(os.path.join(M, "rq2_diversevul_m1_postwarmup", f"codebert-base_project_disjoint_seed{s}_postwarmup.json"), "w"))
        json.dump({"history": [{"valid": {"pr_auc": c}} for c in curve[:3]], "best_epoch": 1, "test": {"pr_auc": test[0] + 0.001 * s}},
                  open(os.path.join(R, "rq2_diversevul_phase2", f"codebert-base_project_disjoint_seed{s}.json"), "w"))
        json.dump({"test": {"pr_auc": 0.25}}, open(os.path.join(R, "rq2_diversevul_final", f"codebert-base_random_control_seed{s}.json"), "w"))
        json.dump({"test": {"pr_auc": 0.17}}, open(os.path.join(R, "rq2_diversevul_phase3", f"codebert-base_commit_disjoint_seed{s}.json"), "w"))
    json.dump({"history": [{"valid": {"pr_auc": v}, "test": {"pr_auc": v + 0.01}} for v in (0.64, 0.70, 0.72, 0.71, 0.70)],
               "best_epoch": 3, "test": {"pr_auc": 0.73}, "patience": 2, "max_epochs": 15},
              open(os.path.join(M, "rq2_devign_m1_postwarmup", "graphcodebert-base_family_disjoint_seed2_postwarmup.json"), "w"))
    json.dump({"history": [{"valid": {"pr_auc": v}} for v in (0.64, 0.50, 0.51)], "best_epoch": 1, "test": {"pr_auc": 0.6391}},
              open(os.path.join(R, "from_backup", "rq2_devign_final", "graphcodebert-base_family_disjoint_seed2.json"), "w"))
    o = compute(T, M)
    assert all(o["a_per_seed"][s]["original_rule_epoch"] == 1 and o["a_per_seed"][s]["post_warmup_epoch"] == 4 for s in (1, 2, 3))
    assert abs(o["a_paired_post_minus_original"]["mean"] - 0.02) < 1e-9
    assert o["b_max_abs_valid_diff"] == 0 and o["b_max_abs_test_diff_original_rule"] < 1e-12
    assert abs(o["c_post_warmup"]["random_to_project"]["mean"] - (0.172 - 0.25)) < 1e-9
    assert abs(o["c_primary"]["rho"] - (0.17 - 0.25) / (0.152 - 0.25)) < 1e-9
    assert o["d_devign"]["post_warmup_epoch"] == 3 and o["d_devign"]["original_rule_epoch"] == 3
    print("SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m1-dir", default=os.path.join(ROOT, "results", "m1_postwarmup_1112"))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return 0
    if not os.path.isdir(a.m1_dir):
        sys.exit(f"!!! missing {a.m1_dir}: extract m1_item57_full.tgz there first")
    out = compute(ROOT, a.m1_dir)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "out", "m1_numbers.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "a_per_seed"}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
