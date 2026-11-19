#!/usr/bin/env python3
"""revision_numbers.py -- check every number added in the revision of 9 Nov 2026 against the result files.

Referee review `paper/review/referee_review_2026-11-09.md`; items in brackets.
  [M1/13]  project-disjoint runs: selected epoch 1, validation PR-AUC at epochs 1-3, warm-up 1.5 epochs
           results/rq2_diversevul_phase2/codebert-base_project_disjoint_seed{1,2,3}.json (history[].valid)
  [M3/14]  published -> family-disjoint under the final protocol, seeds 1-3: +0.0042 [-0.0623, +0.0707]
           results/rq2_diversevul_final/codebert-base_{published,family_disjoint}_seed{1,2,3}.json
           pilot gap -0.074: results/from_backup/rq2_diversevul/ (3-epoch pilot)
  [M4/15]  H4b descriptive: commit-disjoint -> matched stratified control +0.0103 [-0.0162, +0.0367];
           commit-disjoint -> decontaminated -0.0162 [-0.0363, +0.0039]; shares 0.39 / 0.61 of -0.0265;
           headroom 0.0414 / 0.0150; direct leak of commit-disjoint test positives 40.40%
  [M5/16]  length-balanced rank accuracy with CIs, random control and family-disjoint
           results/rq2_diversevul_phase2/paired_eval.json
  [32]     exclusion levels at 4 decimals: results/cpu_checks_1021/rq2_diversevul_phase2/paired_eval_noidentical_{ws,any}.json
  [21]     pivot-approximated buckets (strict criterion, theta 0.7): Devign 3, DiverseVul 163, PrimeVul 141
  [32]     GraphCodeBERT Devign family-disjoint seed 1: 0.7360
  [M1/13]  post hoc rerun with selection after the warm-up (final round, 12 Nov 2026), via m1_numbers.compute:
           results/m1_postwarmup_1112/rq2_{diversevul,devign}_m1_postwarmup/ (settings identical, epochs,
           paired difference, H2 / commit->project / rho recomputed, reproduction differences, Devign rerun)
  [57]     same-commit vulnerable functions (exploratory, post hoc): results/m1_postwarmup_1112/item57/item57.json
t-intervals: t_{0.975,2} = 4.303, as in rq2_numbers.py. Exit 1 on any mismatch.
Usage (repository root): python analysis/scripts/revision_numbers.py
"""
from __future__ import annotations

import json
import math
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results")
checks, fails = [], []


def chk(name, got, want, nd, src):
    ok = round(got, nd) == round(want, nd) if isinstance(want, float) else got == want
    checks.append({"name": name, "got": got, "want": want, "src": src, "ok": ok})
    if not ok:
        fails.append(checks[-1])


def run(d, arm, s):
    return json.load(open(os.path.join(R, d, f"codebert-base_{arm}_seed{s}.json"), encoding="utf-8"))


def tci(x):
    m = st.mean(x)
    h = 4.303 * st.stdev(x) / math.sqrt(len(x))
    return m, m - h, m + h


S3 = (1, 2, 3)
# [M1] project-disjoint selection and warm-up
va = {}
for s in S3:
    r = run("rq2_diversevul_phase2", "project_disjoint", s)
    chk(f"project s{s} best epoch", r["best_epoch"], 1, 0, "run json")
    chk(f"project s{s} epochs run", r["epochs_run"], 3, 0, "run json")
    va[s] = [h["valid"]["pr_auc"] for h in r["history"]]
    chk(f"project s{s} no test metric in history", any("test" in h for h in r["history"]), False, 0, "run json")
ep1 = [va[s][0] for s in S3]
later = [v for s in S3 for v in va[s][1:]]
chk("valid PR-AUC epoch 1, min", min(ep1), 0.1917, 4, "history")
chk("valid PR-AUC epoch 1, max", max(ep1), 0.2016, 4, "history")
chk("valid PR-AUC epochs 2-3, min", min(later), 0.1399, 4, "history")
chk("valid PR-AUC epochs 2-3 of the best post-warm-up epoch, min", min(max(va[s][1:]) for s in S3), 0.1683, 4, "history")
chk("valid PR-AUC epochs 2-3, all below epoch 1", all(max(va[s][1:]) < va[s][0] for s in S3), True, 0, "history")
post = sorted(v for s in S3 for v in va[s][1:])
chk("valid PR-AUC epochs 2-3, max", max(later), 0.1852, 4, "history")
src = open(os.path.join(os.path.dirname(os.path.dirname(HERE)), "pipeline", "finetune_vd.py"), encoding="utf-8").read()
chk("warm-up = 10% of 15-epoch schedule", "get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)" in src, True, 0,
    "finetune_vd.py")
chk("warm-up epochs", 0.1 * 15, 1.5, 1, "finetune_vd.py")
# [M3] published -> family under the final protocol, and the pilot gap
pub = [run("rq2_diversevul_final", "published", s)["test"]["pr_auc"] for s in S3]
fam = [run("rq2_diversevul_final", "family_disjoint", s)["test"]["pr_auc"] for s in S3]
m, lo, hi = tci([f - p for f, p in zip(fam, pub)])
chk("published->family mean", m, 0.0042, 4, "rq2_diversevul_final")
chk("published->family lo", lo, -0.0623, 4, "rq2_diversevul_final")
chk("published->family hi", hi, 0.0707, 4, "rq2_diversevul_final")
pp = [run(os.path.join("from_backup", "rq2_diversevul"), "published", s)["test"]["pr_auc"] for s in S3]
pf = [run(os.path.join("from_backup", "rq2_diversevul"), "family_disjoint", s)["test"]["pr_auc"] for s in S3]
chk("pilot family minus published", st.mean(pf) - st.mean(pp), -0.074, 3, "from_backup/rq2_diversevul")
# [M4] H4b descriptive comparisons with the unmodified commit-disjoint split
cd = [run("rq2_diversevul_phase3", "commit_disjoint", s)["test"]["pr_auc"] for s in S3]
sz = [run("rq2_diversevul_phase4b", "commit_sizectrl", s) for s in S3]
dc = [run("rq2_diversevul_phase4b", "commit_decon", s) for s in S3]
szp = [r["test"]["pr_auc"] for r in sz]
dcp = [r["test"]["pr_auc"] for r in dc]
m, lo, hi = tci([a - b for a, b in zip(szp, cd)])
chk("commit->control mean", m, 0.0103, 4, "phase3 + phase4b")
chk("commit->control lo", lo, -0.0162, 4, "phase3 + phase4b")
chk("commit->control hi", hi, 0.0367, 4, "phase3 + phase4b")
m2, lo2, hi2 = tci([a - b for a, b in zip(dcp, cd)])
chk("commit->decon mean", m2, -0.0162, 4, "phase3 + phase4b")
chk("commit->decon lo", lo2, -0.0363, 4, "phase3 + phase4b")
chk("commit->decon hi", hi2, 0.0039, 4, "phase3 + phase4b")
h4b = st.mean(dcp) - st.mean(szp)
chk("H4b difference", h4b, -0.0265, 4, "phase4b")
chk("share from control rise", m / -h4b, 0.39, 2, "derived")
chk("share from decon fall", -m2 / -h4b, 0.61, 2, "derived")
chk("headroom control", st.mean(r["test"]["pr_auc"] - r["length_only_floor"]["pr_auc"] for r in sz), 0.0414, 4, "phase4b")
chk("headroom decon", st.mean(r["test"]["pr_auc"] - r["length_only_floor"]["pr_auc"] for r in dc), 0.0150, 4, "phase4b")
de = json.load(open(os.path.join(R, "h4_precheck_1022", "direct_edges", "diversevul_hr_direct_summary.json"),
                    encoding="utf-8"))["splits"]["commit_disjoint"]
chk("direct leak of commit-disjoint test positives (%)", 100 * de["direct_leaked_test_pos"] / de["n_test_pos"], 40.40, 2,
    "direct summary")
# [M5] rank accuracy intervals
pe = json.load(open(os.path.join(R, "rq2_diversevul_phase2", "paired_eval.json"), encoding="utf-8"))["arms"]
for arm, (v, lo, hi) in {"random_control": (0.6010, 0.5419, 0.6677), "family_disjoint": (0.5465, 0.5205, 0.5734),
                         "commit_disjoint": (0.4936, 0.4741, 0.5117), "project_disjoint": (0.4540, 0.4342, 0.4731)}.items():
    lc = pe[arm]["length_check"]
    chk(f"{arm} balanced rank acc", lc["balanced_rank_acc"], v, 4, "paired_eval.json")
    chk(f"{arm} balanced lo", lc["balanced_rank_acc_ci95"][0], lo, 4, "paired_eval.json")
    chk(f"{arm} balanced hi", lc["balanced_rank_acc_ci95"][1], hi, 4, "paired_eval.json")
    if arm in ("random_control", "family_disjoint"):
        chk(f"{arm} CI above 0.5", lc["balanced_rank_acc_ci95"][0] > 0.5, True, 0, "paired_eval.json")
# [32] exclusion levels
P2 = os.path.join(R, "cpu_checks_1021", "rq2_diversevul_phase2")
for fn, want in (("paired_eval_noidentical_ws.json", {"random_control": (0.6034,), "family_disjoint": (0.5469,),
                                                     "commit_disjoint": (0.4970, 0.4775, 0.5157),
                                                     "project_disjoint": (0.4798, 0.4553, 0.5029)}),
                 ("paired_eval_noidentical_any.json", {"commit_disjoint": (0.5034, 0.4825, 0.5221),
                                                      "project_disjoint": (0.4835, 0.4602, 0.5086)})):
    arms = json.load(open(os.path.join(P2, fn), encoding="utf-8"))["arms"]
    for arm, w in want.items():
        lc = arms[arm]["length_check"]
        chk(f"{fn} {arm}", lc["balanced_rank_acc"], w[0], 4, fn)
        if len(w) == 3:
            chk(f"{fn} {arm} lo", lc["balanced_rank_acc_ci95"][0], w[1], 4, fn)
            chk(f"{fn} {arm} hi", lc["balanced_rank_acc_ci95"][1], w[2], 4, fn)
# [21] pivot-approximated buckets, strict criterion theta 0.7
for name, path, want in (("Devign", os.path.join("from_backup", "devign_audit_report.json"), 3),
                         ("DiverseVul", os.path.join("from_backup", "diversevul_audit_report.json"), 163)):
    rep = json.load(open(os.path.join(R, path), encoding="utf-8"))
    vals = []
    def walk(o):
        if isinstance(o, dict):
            if "buckets_pivot_approximated" in o and o.get("theta", 0.7) in (0.7, None):
                vals.append(o["buckets_pivot_approximated"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(rep)
    chk(f"{name} pivot buckets", vals[0] if vals else None, want, 0, path)
log = open(os.path.join(R, "audit_7oct", "primevul_audit.log"), encoding="utf-8").read()
chk("PrimeVul pivot buckets", int(re.search(r"buckets_pivot_approximated: (\d+)", log).group(1)), 141, 0,
    "audit_7oct/primevul_audit.log")
# [32] GraphCodeBERT Devign family-disjoint seed 1
g = json.load(open(os.path.join(R, "from_backup", "rq2_devign_final", "graphcodebert-base_family_disjoint_seed1.json"),
                   encoding="utf-8"))["test"]["pr_auc"]
chk("GraphCodeBERT Devign family s1", g, 0.7360, 4, "rq2_devign_final")

# [M1/13, final round 12 Nov 2026] post hoc selection after the warm-up (rerun results/m1_postwarmup_1112)
sys.path.insert(0, HERE)
import m1_numbers  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
M1 = m1_numbers.compute(ROOT, os.path.join(R, "m1_postwarmup_1112"))
SRC = "m1_postwarmup_1112 via m1_numbers.py"
chk("M1 settings identical to Phase 2", M1["b_settings_differences"], [], 0, SRC)
chk("M1 Devign settings identical", M1["d_settings_differences"], [], 0, SRC)
for s, (eo, ep) in {1: (1, 3), 2: (1, 3), 3: (1, 2)}.items():
    chk(f"M1 s{s} original-rule epoch", M1["a_per_seed"][s]["original_rule_epoch"], eo, 0, SRC)
    chk(f"M1 s{s} post-warm-up epoch", M1["a_per_seed"][s]["post_warmup_epoch"], ep, 0, SRC)
a = M1["a_paired_post_minus_original"]
for k, w in (("mean", 0.0077), ("lo", -0.0059), ("hi", 0.0212)):
    chk(f"M1 post minus original {k}", a[k], w, 4, SRC)
cw = M1["c_post_warmup"]
for key, want in (("random_to_project", (-0.1013, -0.1155, -0.0872)), ("commit_to_project", (-0.0198, -0.0284, -0.0112))):
    for k, w in zip(("mean", "lo", "hi"), want):
        chk(f"M1 post-warm-up {key} {k}", cw[key][k], w, 4, SRC)
chk("M1 post-warm-up rho", cw["rho"], 0.80, 2, SRC)
chk("M1 primary rho unchanged", M1["c_primary"]["rho"], 0.76, 2, SRC)
chk("M1 H2 rule holds (interval below zero)", cw["random_to_project"]["hi"] < 0, True, 0, SRC)
chk("M1 max abs valid diff", M1["b_max_abs_valid_diff"], 0.0179, 4, SRC)
chk("M1 max abs test diff, original rule", M1["b_max_abs_test_diff_original_rule"], 0.0081, 4, SRC)
d = M1["d_devign"]
chk("M1 Devign both rules epoch 5", (d["original_rule_epoch"], d["post_warmup_epoch"]), (5, 5), 0, SRC)
chk("M1 Devign rerun test", d["post_warmup_test"], 0.7178, 4, SRC)
chk("M1 Devign rerun best valid", max(d["valid"]), 0.7375, 4, SRC)
chk("M1 Devign original valid", d["original_run"]["valid"], [0.6401, 0.5008, 0.5067], 0, SRC)
# [57] same-commit vulnerable functions (exploratory, post hoc)
I57 = json.load(open(os.path.join(R, "m1_postwarmup_1112", "item57", "item57.json"), encoding="utf-8"))
cs = I57["commit_structure"]
chk("item57 commits with vulnerable functions", cs["commits_with_vulnerable_functions"], 7514, 0, "item57.json")
chk("item57 commits with 2+ (%)", cs["commits_with_2plus_vulnerable_pct"], 37.62, 2, "item57.json")
chk("item57 vulnerable functions in 2+ commits (%)", cs["vulnerable_functions_in_2plus_commits_pct"], 75.26, 2, "item57.json")
chk("item57 max per commit", cs["vulnerable_per_commit_max"], 245, 0, "item57.json")
for arm, key, w in (("random_control", "recall_positives", (-0.0941, -0.1622, -0.0371)),
                    ("random_control", "fpr_negatives", (-0.0099, -0.0171, -0.0036)),
                    ("family_disjoint", "recall_positives", (0.0182, -0.0254, 0.0645)),
                    ("family_disjoint", "fpr_negatives", (-0.0219, -0.0286, -0.0154))):
    v = I57["arms"][arm]["predictions_no_counterpart_stratum"][key]
    for name, got, want in zip(("diff", "lo", "hi"), [v["with_minus_without"]] + v["ci95"], w):
        chk(f"item57 {arm} {key} {name}", got, want, 4, "item57.json")
# [R39, 13 Nov 2026] Supplementary Table S3: four pair outcomes per split (descriptive), paired_eval.json
for arm, (n, pc) in {"random_control": (144, 0.0556), "family_disjoint": (688, 0.0363),
                     "commit_disjoint": (1515, 0.0132), "project_disjoint": (1237, 0.0049)}.items():
    a = pe[arm]
    chk(f"Table S3 {arm} pairs", a["pairs"], n, 0, "paired_eval.json")
    chk(f"Table S3 {arm} P-C", a["mean"]["P-C"], pc, 4, "paired_eval.json")
    chk(f"Table S3 {arm} outcomes sum to 1", sum(a["mean"][m] for m in ("P-C", "P-V", "P-B", "P-R")), 1.0, 9, "paired_eval.json")
chk("at most 5.56% of pairs correct on both sides", 100 * max(pe[a]["mean"]["P-C"] for a in pe), 5.56, 2, "paired_eval.json")
# [13 Nov 2026] RQ answer boxes: every number in a box traced to the result files; box length <= 40 words
R1N = json.load(open(os.path.join(HERE, "out", "rq1_numbers.json"), encoding="utf-8"))
SEC = os.path.join(os.path.dirname(HERE), "sections")
# Replication package: the manuscript text is not distributed, so the checks that read the RQ
# answer boxes run only when a sections/ folder is present; the numbers themselves are checked below.
HAVE_TEXT = os.path.isdir(SEC)


def box(sec):
    t = open(os.path.join(SEC, sec), encoding="utf-8").read()
    return re.search(r"\\begin\{rqanswer\}(.*?)\\end\{rqanswer\}", t, re.S).group(1)


def box_words(b):
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", b)
    return len([x for x in s.split() if re.search(r"[A-Za-z0-9]", x)])  # math tokens count as words


b1 = box("rq1.tex") if HAVE_TEXT else ""
for corpus, want in (("devign", 15.34), ("diversevul", 29.61), ("primevul", 15.70)):
    chk(f"RQ1 box {corpus} leak at 0.7", R1N["audit"][corpus]["0.7"]["leaked_pct"], want, 2, "rq1_numbers.json")
    if HAVE_TEXT: chk(f"RQ1 box shows {want}", f"{want:.2f}\\%" in b1, True, 0, "rq1.tex box")
chk("RQ1 box min leak at 0.9", R1N["derived"]["min_leak_at_0_9"], 10.91, 2, "rq1_numbers.json")
if HAVE_TEXT: chk("RQ1 box shows 10.91", "10.91\\%" in b1, True, 0, "rq1.tex box")
b2 = box("rq2.tex") if HAVE_TEXT else ""
chk("RQ2 box rho (primary)", round(M1["c_primary"]["rho"], 2), 0.76, 2, SRC)
if HAVE_TEXT: chk("RQ2 box shows 0.76", "0.76 of the project-disjoint drop" in b2, True, 0, "rq2.tex box")
for sec in (("rq1.tex", "rq2.tex", "rq3.tex") if HAVE_TEXT else ()):
    if HAVE_TEXT: chk(f"{sec} answer box <= 40 words", box_words(box(sec)) <= 40, True, 0, sec)
rp = {a: I57["arms"][a]["predictions_no_counterpart_stratum"]["recall_positives"] for a in ("random_control", "family_disjoint")}
chk("item57 recall lower with mate on random control (CI below 0)", rp["random_control"]["ci95"][1] < 0, True, 0, "item57.json")
chk("item57 recall CI includes 0 on family-disjoint", rp["family_disjoint"]["ci95"][0] < 0 < rp["family_disjoint"]["ci95"][1], True, 0,
    "item57.json")

os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
json.dump({"checks": checks}, open(os.path.join(HERE, "out", "revision_numbers.json"), "w", encoding="utf-8"), indent=1,
          default=str)
print(f"{len(checks)} checks; {len(fails)} failed")
for f in fails:
    print("  FAIL", f["name"], "got", f["got"], "want", f["want"], f["src"])
sys.exit(1 if fails else 0)
