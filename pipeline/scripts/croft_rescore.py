#!/usr/bin/env python3
"""
croft_rescore.py -- Croft-style re-scoring of saved DiverseVul predictions (action A5,
18 Oct 2026). EXPLORATORY: written after every pre-registered result was known; it
changes no H1-H3 verdict and has no decision rule.

Croft et al. (2023, Table V p7) removed cross-set duplicates from the TEST set and
re-scored. Our RQ2 instead re-partitions and compares against a random control.
This script applies Croft's design to our own predictions: for every arm with saved
predictions, PR-AUC on the full test set vs on the test set with family-leaked
functions removed (a test function is leaked if a training function of that arm's
split is in its theta=0.7 clone family, Eq. leak). Removing items changes the
positive rate (leaked DiverseVul test functions are mostly negatives), so the
positive rate and PR-AUC / positive rate are reported next to PR-AUC.

Per arm: mean over the arm's seeds; 95% bootstrap CI over test functions (the same
resampled indices for both subsets, so the difference is paired), 1,000 resamples.
Arms and seeds as in paired_eval.py: random_control / family_disjoint (Phase 2 seeds
4-5), project_disjoint (Phase 2 seeds 1-3), commit_disjoint (Phase 3 seeds 1-3).
The family-disjoint arm has no leaked test functions by construction (difference 0).

Lift = PR-AUC / positive rate; its change has a paired bootstrap CI over the same resamples
(added 21 Oct 2026; earlier keys and their values are unchanged).

Writes <disk>/rq2_diversevul_phase2/croft_rescore.{json,md}.
Usage: python scripts/croft_rescore.py [--disk $CFA_DATA_DIR] [--n-boot 1000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score

VD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VD)
from analyze_phase2 import load_preds  # noqa: E402

MODEL = "codebert-base"
FAM = "diversevul_audit_families_theta0.7.json"
ARMS = [("random_control", "diversevul_audit_splitfile_randomctrl_theta0.7.json", "rq2_diversevul_phase2", (4, 5)),
        ("family_disjoint", "diversevul_audit_splitfile_theta0.7.json", "rq2_diversevul_phase2", (4, 5)),
        ("project_disjoint", "diversevul_audit_splitfile_projectdisjoint_seed42.json", "rq2_diversevul_phase2",
         (1, 2, 3)),
        ("commit_disjoint", "diversevul_audit_splitfile_commitdisjoint_seed42.json", "rq2_diversevul_phase3",
         (1, 2, 3))]


def leaked_ids(split, fam):
    tr = {fam[i] for i, s in split.items() if s == "train" and i in fam}
    return {i for i, s in split.items() if s == "test" and i in fam and fam[i] in tr}


def ap(y, p):
    return float(average_precision_score(y, p)) if 0 < y.sum() < y.size else float("nan")


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--disk", default=os.environ.get("CFA_DATA_DIR", "data"))
    a.add_argument("--n-boot", type=int, default=1000)
    a.add_argument("--seed", type=int, default=12345)
    a.add_argument("--out", default=None,
                   help="output prefix; default <disk>/rq2_diversevul_phase2/croft_rescore")
    args = a.parse_args()
    D = args.disk
    famp = os.path.join(D, FAM)
    if not os.path.exists(famp):
        print(f"!!! missing input: {famp}", file=sys.stderr)
        return 2
    fam = {str(k): v for k, v in json.load(open(famp)).items()}
    rng = np.random.default_rng(args.seed)
    rep = {"note": "EXPLORATORY (action A5), not pre-registered", "arms": {}}
    for arm, sf, sub, seeds in ARMS:
        sp = os.path.join(D, sf)
        if not os.path.exists(sp):
            print(f"  (no split file for {arm})")
            continue
        split = {str(k): v for k, v in json.load(open(sp)).items()}
        leak = leaked_ids(split, fam)
        runs, ids0 = [], None
        for s in seeds:
            pr = load_preds(os.path.join(D, sub, f"{MODEL}_{arm}_seed{s}.preds.json.gz"))
            if pr is None:
                continue
            ids = [str(x) for x in pr["test"]["id"]]
            if ids0 is None:
                ids0 = ids
            elif ids != ids0:
                print(f"!!! {arm} seed {s}: test ids differ from the first seed", file=sys.stderr)
                return 2
            runs.append((s, np.asarray(pr["test"]["target"]), np.asarray(pr["test"]["prob"], dtype=float)))
        if not runs:
            print(f"  (no predictions for {arm})")
            continue
        y = runs[0][1]
        if any((r[1] != y).any() for r in runs):
            print(f"!!! {arm}: test labels differ between seeds", file=sys.stderr)
            return 2
        keep = np.asarray([i not in leak for i in ids0])

        def stats(idx):
            yy, kk = y[idx], keep[idx]
            full = np.mean([ap(yy, p[idx]) for _, _, p in runs])
            clean = np.mean([ap(yy[kk], p[idx][kk]) for _, _, p in runs])
            # lift = PR-AUC / positive rate of the same (resampled) set; computed from the same
            # indices, so the bootstrap draws and the first three statistics are unchanged
            lf = full / yy.mean() if yy.any() else float("nan")
            lc = clean / yy[kk].mean() if yy[kk].any() else float("nan")
            return full, clean, clean - full, lf, lc, lc - lf

        full, clean, diff, *_ = stats(np.arange(y.size))
        boots = np.array([stats(rng.integers(0, y.size, y.size)) for _ in range(args.n_boot)])
        ci = lambda c: [float(np.nanpercentile(boots[:, c], 2.5)), float(np.nanpercentile(boots[:, c], 97.5))]
        pos_full, pos_clean = float(y.mean()), float(y[keep].mean()) if keep.any() else float("nan")
        rep["arms"][arm] = {
            "seeds": [r[0] for r in runs], "n_test": int(y.size), "n_leaked_removed": int((~keep).sum()),
            "leaked_pct": round(100 * float((~keep).mean()), 2),
            "pos_rate_full": pos_full, "pos_rate_without_leaked": pos_clean,
            "pr_auc_full": full, "pr_auc_full_ci95": ci(0),
            "pr_auc_without_leaked": clean, "pr_auc_without_leaked_ci95": ci(1),
            "diff_without_minus_full": diff, "diff_ci95": ci(2),
            "pr_over_pos_full": full / pos_full, "pr_over_pos_without_leaked": clean / pos_clean,
            "per_seed": {f"s{s}": {"full": ap(y, p), "without_leaked": ap(y[keep], p[keep])} for s, _, p in runs},
            # added 21 Oct 2026 (H4 prep, B.5): paired bootstrap CI of the change in lift
            "pr_over_pos_full_ci95": ci(3), "pr_over_pos_without_leaked_ci95": ci(4),
            "lift_diff_without_minus_full": clean / pos_clean - full / pos_full, "lift_diff_ci95": ci(5)}
        print(f"  {arm}: removed {(~keep).sum():,} leaked of {y.size:,}; PR-AUC {full:.4f} -> {clean:.4f}", flush=True)

    L = ["# Croft-style re-scoring: test set with family-leaked functions removed (EXPLORATORY, A5)", "",
         "Leaked = a training function of the arm's split is in the test function's θ=0.7 family. "
         "Mean over seeds; 95% bootstrap CI over test functions (paired for the difference). "
         "Removal changes the positive rate, so PR/pos is shown too; Δ lift = change in PR/pos with its "
         "paired bootstrap CI (same resamples).", "",
         "| arm | seeds | test n | removed (leaked %) | PR-AUC full | PR-AUC without leaked | Δ (without − full) | "
         "pos rate full → without | PR/pos full → without | Δ lift [95% CI] |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda m, c: f"{m:.4f} [{c[0]:.4f}, {c[1]:.4f}]"
    for arm, r in rep["arms"].items():
        L.append(f"| {arm} | {','.join(map(str, r['seeds']))} | {r['n_test']:,} | {r['n_leaked_removed']:,} "
                 f"({r['leaked_pct']}%) | {f(r['pr_auc_full'], r['pr_auc_full_ci95'])} | "
                 f"{f(r['pr_auc_without_leaked'], r['pr_auc_without_leaked_ci95'])} | "
                 f"{r['diff_without_minus_full']:+.4f} [{r['diff_ci95'][0]:+.4f}, {r['diff_ci95'][1]:+.4f}] | "
                 f"{r['pos_rate_full']:.4f} → {r['pos_rate_without_leaked']:.4f} | "
                 f"{r['pr_over_pos_full']:.2f} → {r['pr_over_pos_without_leaked']:.2f} | "
                 f"{r['lift_diff_without_minus_full']:+.2f} [{r['lift_diff_ci95'][0]:+.2f}, "
                 f"{r['lift_diff_ci95'][1]:+.2f}] |")
    out = args.out or os.path.join(D, "rq2_diversevul_phase2", "croft_rescore")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rep, open(out + ".json", "w", encoding="utf-8"), indent=2)
    open(out + ".md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
