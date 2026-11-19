#!/usr/bin/env python3
"""
analyze_phase2.py -- the pre-registered Phase 2 analysis (RESEARCH_PLAN §13.2).
Written and committed BEFORE any Phase 2 result existed; the decision rules
below are fixed here and must not be changed after looking at the numbers.

INPUTS (all on the data disk; CPU only)
  Stage B JSONs   rq2_diversevul_final/codebert-base_<arm>_seed{1,2,3}.json
  Phase 2 JSONs   rq2_diversevul_phase2/codebert-base_<arm>_seed<s>.json
  Phase 2 preds   rq2_diversevul_phase2/codebert-base_<arm>_seed<s>.preds.json.gz
  leak categories family_label/diversevul_theta0.7_leakcat_<arm>.json

A. MAIN Δ, n = 5.  Δ(random_control -> family_disjoint), paired by seed over
   seeds 1-5 (1-3 from Stage B, 4-5 from Phase 2): test PR-AUC, headroom
   (PR-AUC - length floor), MCC, ROC-AUC. Mean ± sample sd, 95% t-CI.
   Test-set bootstrap CI for the seeds that saved predictions (4-5).

B. H2 -- distribution shift, not memorisation.
   Δ(random_control -> project_disjoint), paired by seed over seeds 1-3, beside
   Δ(random_control -> family_disjoint) over the same seeds.
   DECISION RULE (fixed): H2 is SUPPORTED iff the 95% t-CI of the mean
   Δ(random -> project) PR-AUC lies entirely below 0 AND the 95% t-CI of
   Δ(random -> family) includes 0. If both include 0: "no detectable project
   effect either" (H2 not supported). Any other pattern is reported as found.
   Secondary: bootstrap CI of the difference in mean test PR-AUC between the
   project arm (seeds 1-3, predictions) and the random arm (seeds 4-5,
   predictions) -- unpaired seeds, so it is a robustness check only.
   The project arm keeps a residual 5.83% family leak; its per-category table
   (section C layout) is reported descriptively.

C. H1 -- label-consistent memorisation, on the random_control TEST set
   (identical test functions for every seed, only training randomness differs).
   Every test function carries one category from family_label_audit.py:
   clean / leaked_same / leaked_opposite / leaked_both.
   Categories differ hugely in prevalence (DiverseVul: ~0.015 vs ~0.51), so
   PR-AUC across categories is NOT the test statistic (it is reported, marked
   prevalence-confounded). The test statistics are error rates by true label at
   a threshold chosen on that run's VALIDATION predictions (max-F1 on valid,
   never on test):
     recall    on positives  (TPR)
     false-pos on negatives  (FPR)
   plus ROC-AUC within a category when both classes are present.
   Memorisation predicts, relative to clean functions of the same label:
     H1a  recall(leaked_opposite positives) < recall(clean positives)
     H1b  FPR(leaked_opposite negatives)    > FPR(clean negatives)
     H1c  recall(leaked_same positives)     > recall(clean positives)
     H1d  FPR(leaked_same negatives)        < FPR(clean negatives)
   Each contrast: averaged over the seeds with predictions, with a 95% paired
   bootstrap CI (test functions resampled within category, the same resample
   applied to every seed). DECISION RULE (fixed): a contrast is SUPPORTED iff
   its CI excludes 0 in the predicted direction; CONTRADICTED iff it excludes 0
   in the opposite direction; otherwise a TIE. H1 as a whole is supported iff
   H1a and H1b are both supported (they carry the mechanism; H1c/H1d have few
   positives). Known confound, to be stated with the result: leaked_opposite
   positives are pre-fix versions whose post-fix twin is in train; clean
   positives may differ in other ways (length, CWE). The category design
   cannot remove that; the result is an association, not a causal estimate.

USAGE (server)
  python pipeline/analyze_phase2.py
  # partial results are fine: whatever exists is analysed, the rest is listed
OUTPUT
  rq2_diversevul_phase2/phase2_analysis.{json,md}
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import statistics as st
import sys

import numpy as np
from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score)

# two-sided 95% t critical values, df = 1..30 (avoids a scipy dependency)
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]
CATS = ("clean", "leaked_same", "leaked_opposite", "leaked_both")
MODEL = "codebert-base"


# ------------------------------------------------------------------ helpers

def t_ci(xs):
    """mean, sd, (lo, hi) 95% t-interval; None-safe for n < 2."""
    n = len(xs)
    if n == 0:
        return None
    m = st.mean(xs)
    if n < 2:
        return {"n": 1, "mean": m, "sd": None, "ci": None}
    sd = st.stdev(xs)
    h = T975[min(n - 1, len(T975)) - 1] * sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "ci": [m - h, m + h]}


def verdict_zero(ci):
    if not ci or ci.get("ci") is None:
        return "n/a"
    lo, hi = ci["ci"]
    return "below 0" if hi < 0 else ("above 0" if lo > 0 else "includes 0")


def best_f1_threshold(y, p):
    """Same rule as finetune_vd.best_f1 (max-F1 over the PR curve)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec),
                   where=(prec + rec) > 0)
    i = int(np.argmax(f1))
    return float(thr[i]) if i < len(thr) else 1.0


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_preds(path):
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def run_metrics(res):
    t = res["test"]
    floor = (res.get("length_only_floor") or {}).get("pr_auc")
    return {"pr_auc": t["pr_auc"], "roc_auc": t["roc_auc"], "mcc": t["mcc"],
            "pos_rate": t["positive_rate"],
            "headroom": t["pr_auc"] - floor if floor is not None else None}


def paired_delta(lo_runs, hi_runs, seeds, key):
    d = {s: hi_runs[s][key] - lo_runs[s][key] for s in seeds
         if s in lo_runs and s in hi_runs
         and lo_runs[s][key] is not None and hi_runs[s][key] is not None}
    out = t_ci(list(d.values())) or {"n": 0}
    out["per_seed"] = {f"s{s}": round(v, 4) for s, v in sorted(d.items())}
    return out


def boot_mean_ap(pred_sets, rng, n_boot):
    """Bootstrap distribution of the mean (over seeds) test PR-AUC. All seeds of
    one arm share a test set, so one index resample is applied to every seed."""
    y = np.asarray(pred_sets[0]["target"])
    probs = [np.asarray(p["prob"]) for p in pred_sets]
    n = len(y)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        out[b] = np.mean([average_precision_score(yb, pr[idx]) for pr in probs]) \
            if 0 < yb.sum() < n else np.nan
    return out


def pct_ci(a):
    a = a[~np.isnan(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else None


# ------------------------------------------------------------------ H1 core

def category_table(pred_sets, thresholds, cat_of):
    """Per-category metrics for each seed. pred_sets/thresholds aligned."""
    ids = pred_sets[0]["id"]
    for p in pred_sets[1:]:
        assert p["id"] == ids, "seeds must share the same test ids in the same order"
    y = np.asarray(pred_sets[0]["target"])
    cat = np.array([cat_of.get(i, "missing") for i in ids])
    rows = {}
    for c in CATS:
        m = cat == c
        yc = y[m]
        per_seed = []
        for p, thr in zip(pred_sets, thresholds):
            pc = np.asarray(p["prob"])[m]
            pos, neg = yc == 1, yc == 0
            hit = pc >= thr
            per_seed.append({
                "recall_pos": float(hit[pos].mean()) if pos.any() else None,
                "fpr_neg": float(hit[neg].mean()) if neg.any() else None,
                "mean_prob_pos": float(pc[pos].mean()) if pos.any() else None,
                "mean_prob_neg": float(pc[neg].mean()) if neg.any() else None,
                "roc_auc": float(roc_auc_score(yc, pc)) if pos.any() and neg.any() else None,
                "pr_auc_prevalence_confounded":
                    float(average_precision_score(yc, pc)) if pos.any() and neg.any() else None,
            })
        avg = {k: (float(np.mean([s[k] for s in per_seed])) if all(s[k] is not None for s in per_seed) else None)
               for k in per_seed[0]}
        rows[c] = {"n": int(m.sum()), "n_pos": int(yc.sum()), "n_neg": int((yc == 0).sum()),
                   "pos_rate": float(yc.mean()) if m.any() else None,
                   "mean_over_seeds": avg}
    rows["_missing_category"] = int((cat == "missing").sum())
    return rows, y, cat


def h1_contrasts(pred_sets, thresholds, y, cat, rng, n_boot):
    """Paired bootstrap of the four pre-registered contrasts."""
    probs = [np.asarray(p["prob"]) for p in pred_sets]
    hits = [pr >= t for pr, t in zip(probs, thresholds)]

    def rate(c, label):
        idx = np.where((cat == c) & (y == label))[0]
        return idx

    groups = {(c, l): rate(c, l) for c in CATS for l in (0, 1)}

    def stat(sel):          # mean over seeds of the positive-prediction rate in sel
        return float(np.mean([h[sel].mean() for h in hits])) if len(sel) else float("nan")

    spec = {  # name: (category, label, predicted sign of (category - clean))
        "H1a recall leaked_opposite pos - clean pos": ("leaked_opposite", 1, -1),
        "H1b FPR leaked_opposite neg - clean neg": ("leaked_opposite", 0, +1),
        "H1c recall leaked_same pos - clean pos": ("leaked_same", 1, +1),
        "H1d FPR leaked_same neg - clean neg": ("leaked_same", 0, -1),
    }
    out = {}
    for name, (c, lab, sign) in spec.items():
        a, b = groups[(c, lab)], groups[("clean", lab)]
        if len(a) == 0 or len(b) == 0:
            out[name] = {"n_category": int(len(a)), "n_clean": int(len(b)), "verdict": "n/a"}
            continue
        point = stat(a) - stat(b)
        bs = np.empty(n_boot)
        for k in range(n_boot):
            ra = a[rng.integers(0, len(a), len(a))]
            rb = b[rng.integers(0, len(b), len(b))]
            bs[k] = stat(ra) - stat(rb)
        lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        if (sign < 0 and hi < 0) or (sign > 0 and lo > 0):
            v = "SUPPORTED"
        elif (sign < 0 and lo > 0) or (sign > 0 and hi < 0):
            v = "CONTRADICTED"
        else:
            v = "TIE"
        out[name] = {"n_category": int(len(a)), "n_clean": int(len(b)),
                     "difference": point, "ci95": [float(lo), float(hi)],
                     "predicted_sign": "negative" if sign < 0 else "positive",
                     "verdict": v}
    return out


# ------------------------------------------------------------------ report

def fmt(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def fmt_ci(d):
    if not d or d.get("ci") is None:
        return "—"
    return f"{d['mean']:+.4f} ± {d['sd']:.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] (n={d['n']})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disk", default=os.environ.get("CFA_DATA_DIR", "data"))
    ap.add_argument("--stage-b", default=None, help="default <disk>/rq2_diversevul_final")
    ap.add_argument("--phase2", default=None, help="default <disk>/rq2_diversevul_phase2")
    ap.add_argument("--leakcat-prefix", default=None,
                    help="default <disk>/family_label/diversevul_theta0.7_leakcat_")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345, help="bootstrap RNG seed")
    args = ap.parse_args()
    sb = args.stage_b or os.path.join(args.disk, "rq2_diversevul_final")
    p2 = args.phase2 or os.path.join(args.disk, "rq2_diversevul_phase2")
    lp = args.leakcat_prefix or os.path.join(args.disk, "family_label", "diversevul_theta0.7_leakcat_")
    rng = np.random.default_rng(args.seed)

    runs = {}            # arm -> seed -> metrics
    preds = {}           # arm -> seed -> preds payload
    missing = []
    plan = {"random_control": [1, 2, 3, 4, 5], "family_disjoint": [1, 2, 3, 4, 5],
            "project_disjoint": [1, 2, 3]}
    for arm, seeds in plan.items():
        for s in seeds:
            d = sb if s <= 3 and arm != "project_disjoint" else p2
            res = load_json(os.path.join(d, f"{MODEL}_{arm}_seed{s}.json"))
            if res is None:
                missing.append(f"{arm} seed{s}")
                continue
            runs.setdefault(arm, {})[s] = run_metrics(res)
            pr = load_preds(os.path.join(d, f"{MODEL}_{arm}_seed{s}.preds.json.gz"))
            if pr is not None:
                preds.setdefault(arm, {})[s] = pr

    rep = {"inputs": {"stage_b": sb, "phase2": p2, "leakcat_prefix": lp,
                      "n_boot": args.n_boot, "bootstrap_seed": args.seed},
           "missing_runs": missing}
    rc, fd, pd = runs.get("random_control", {}), runs.get("family_disjoint", {}), runs.get("project_disjoint", {})

    # ---- A. main Δ, n=5
    A = {k: paired_delta(rc, fd, [1, 2, 3, 4, 5], k) for k in ("pr_auc", "headroom", "mcc", "roc_auc")}
    if preds.get("random_control") and preds.get("family_disjoint"):
        s_ok = sorted(set(preds["random_control"]) & set(preds["family_disjoint"]))
        if s_ok:
            br = boot_mean_ap([preds["random_control"][s]["test"] for s in s_ok], rng, args.n_boot)
            bf = boot_mean_ap([preds["family_disjoint"][s]["test"] for s in s_ok], rng, args.n_boot)
            A["bootstrap_pr_auc_family_minus_random"] = {"seeds": s_ok, "ci95": pct_ci(bf - br)}
    A["verdict_pr_auc"] = verdict_zero(A["pr_auc"])
    rep["A_main_delta_n5"] = A

    # ---- B. H2
    B = {"random_to_project": {k: paired_delta(rc, pd, [1, 2, 3], k)
                               for k in ("pr_auc", "headroom", "mcc", "roc_auc")},
         "random_to_family_same_seeds": {k: paired_delta(rc, fd, [1, 2, 3], k)
                                         for k in ("pr_auc", "headroom", "mcc", "roc_auc")},
         "test_pos_rates": {arm: {f"s{s}": r["pos_rate"] for s, r in sorted(runs.get(arm, {}).items())}
                            for arm in plan}}
    vp = verdict_zero(B["random_to_project"]["pr_auc"])
    vf = verdict_zero(B["random_to_family_same_seeds"]["pr_auc"])
    if "n/a" in (vp, vf):
        h2 = "INCOMPLETE"
    elif vp == "below 0" and vf == "includes 0":
        h2 = "SUPPORTED"
    elif vp == "includes 0" and vf == "includes 0":
        h2 = "NOT SUPPORTED (no detectable project effect either)"
    else:
        h2 = f"OTHER (random->project {vp}; random->family {vf})"
    B["verdict_H2"] = h2
    if preds.get("project_disjoint") and preds.get("random_control"):
        bp = boot_mean_ap([preds["project_disjoint"][s]["test"] for s in sorted(preds["project_disjoint"])],
                          rng, args.n_boot)
        brr = boot_mean_ap([preds["random_control"][s]["test"] for s in sorted(preds["random_control"])],
                           rng, args.n_boot)
        B["bootstrap_pr_auc_project_minus_random_unpaired"] = {
            "project_seeds": sorted(preds["project_disjoint"]), "random_seeds": sorted(preds["random_control"]),
            "ci95": pct_ci(bp - brr)}
    rep["B_H2"] = B

    # ---- C. H1 (random_control test set) + descriptive project-arm table
    C = {}
    for arm in ("random_control", "project_disjoint"):
        if not preds.get(arm):
            continue
        cat_of = load_json(f"{lp}{arm}.json")
        if cat_of is None:
            C[arm] = {"error": f"missing {lp}{arm}.json -- run run_phase1.sh"}
            continue
        seeds = sorted(preds[arm])
        test_sets = [preds[arm][s]["test"] for s in seeds]
        thr = [best_f1_threshold(preds[arm][s]["valid"]["target"], preds[arm][s]["valid"]["prob"])
               for s in seeds]
        table, y, cat = category_table(test_sets, thr, cat_of)
        entry = {"seeds": seeds, "valid_thresholds": [round(t, 4) for t in thr], "categories": table}
        if arm == "random_control":
            con = h1_contrasts(test_sets, thr, y, cat, rng, args.n_boot)
            entry["contrasts"] = con
            a = con.get("H1a recall leaked_opposite pos - clean pos", {}).get("verdict")
            b = con.get("H1b FPR leaked_opposite neg - clean neg", {}).get("verdict")
            entry["verdict_H1"] = ("SUPPORTED" if a == b == "SUPPORTED" else
                                   "CONTRADICTED" if "CONTRADICTED" in (a, b) else
                                   "NOT SUPPORTED (tie)" if a and b else "INCOMPLETE")
        C[arm] = entry
    rep["C_H1"] = C

    out_dir = p2 if os.path.isdir(p2) else "."
    with open(os.path.join(out_dir, "phase2_analysis.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)

    # ---- markdown
    L = ["# Phase 2 analysis (pre-registered, RESEARCH_PLAN §13.2)", "",
         "_Generated by analyze_phase2.py; decision rules fixed in its docstring before the data existed._", ""]
    if missing:
        L += [f"**Missing runs ({len(missing)}):** " + ", ".join(missing), ""]
    L += ["## A. Main Δ random → family, paired by seed (target n = 5)", "",
          "| metric | Δ mean ± sd [95% t-CI] | per seed |", "|---|---|---|"]
    for k in ("pr_auc", "headroom", "mcc", "roc_auc"):
        L.append(f"| {k} | {fmt_ci(A[k])} | {A[k].get('per_seed')} |")
    if "bootstrap_pr_auc_family_minus_random" in A:
        b = A["bootstrap_pr_auc_family_minus_random"]
        L.append(f"\nTest-set bootstrap (seeds {b['seeds']}), family − random PR-AUC: "
                 f"[{fmt(b['ci95'][0])}, {fmt(b['ci95'][1])}]")
    L += [f"\nPR-AUC Δ CI {A['verdict_pr_auc']}.", "",
          "## B. H2 — random → project vs random → family (seeds 1–3)", "",
          "| metric | random → project | random → family |", "|---|---|---|"]
    for k in ("pr_auc", "headroom", "mcc", "roc_auc"):
        L.append(f"| {k} | {fmt_ci(B['random_to_project'][k])} | {fmt_ci(B['random_to_family_same_seeds'][k])} |")
    L.append(f"\nTest positive rates: {B['test_pos_rates']}")
    if "bootstrap_pr_auc_project_minus_random_unpaired" in B:
        b = B["bootstrap_pr_auc_project_minus_random_unpaired"]
        L.append(f"\nUnpaired bootstrap, project (s{b['project_seeds']}) − random (s{b['random_seeds']}) "
                 f"PR-AUC: [{fmt(b['ci95'][0])}, {fmt(b['ci95'][1])}] (robustness only)")
    L += [f"\n**H2: {h2}**", "", "## C. H1 — error direction by leak category", ""]
    for arm, e in C.items():
        if "error" in e:
            L += [f"{arm}: {e['error']}", ""]
            continue
        L += [f"### {arm} test set — seeds {e['seeds']}, valid-chosen thresholds {e['valid_thresholds']}", "",
              "| category | n | pos | pos rate | recall (pos) | FPR (neg) | ROC-AUC | PR-AUC (prev.-confounded) |",
              "|---|---|---|---|---|---|---|---|"]
        for c in CATS:
            r = e["categories"][c]
            m = r["mean_over_seeds"]
            L.append(f"| {c} | {r['n']:,} | {r['n_pos']:,} | {fmt(r['pos_rate'])} | {fmt(m['recall_pos'])} | "
                     f"{fmt(m['fpr_neg'])} | {fmt(m['roc_auc'])} | {fmt(m['pr_auc_prevalence_confounded'])} |")
        if e["categories"]["_missing_category"]:
            L.append(f"\n! {e['categories']['_missing_category']} test ids had no category")
        if "contrasts" in e:
            L += ["", "| contrast (category − clean) | difference | 95% bootstrap CI | predicted | verdict |",
                  "|---|---|---|---|---|"]
            for name, v in e["contrasts"].items():
                if v["verdict"] == "n/a":
                    L.append(f"| {name} | — | — | — | n/a |")
                    continue
                L.append(f"| {name} | {v['difference']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | "
                         f"{v['predicted_sign']} | **{v['verdict']}** |")
            L += ["", f"**H1: {e['verdict_H1']}** (needs H1a and H1b). Association only — see the "
                  "confound note in the script docstring.", ""]
    with open(os.path.join(out_dir, "phase2_analysis.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out_dir}/phase2_analysis.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
