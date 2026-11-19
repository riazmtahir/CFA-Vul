#!/usr/bin/env python3
"""supp_m1_item57.py -- Supplementary Table S2 (post hoc selection sensitivity, M1) and Supplementary Note S1
(item 57, same-commit vulnerable functions; exploratory, post hoc).

Inputs (read only):
  analysis/scripts/out/m1_numbers.json                       written by m1_numbers.py (run that first)
  results/m1_postwarmup_1112/item57/item57.json           server output of scripts/item57_commit_mates.py
Outputs:
  paper/supplementary/m1_epochs_fragment.tex              Table S2
  paper/supplementary/item57_fragment.tex                 Note S1 tables
Usage (repository root): python analysis/scripts/supp_m1_item57.py
"""
from __future__ import annotations

import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SUPP = os.path.join(ROOT, "analysis", "supplementary")
M = json.load(open(os.path.join(HERE, "out", "m1_numbers.json"), encoding="utf-8"))
I57 = json.load(open(os.path.join(ROOT, "results", "m1_postwarmup_1112", "item57", "item57.json"), encoding="utf-8"))


def f4(x):
    return f"{x:.4f}"


def s4(x):
    return f"${'+' if x >= 0 else '-'}{abs(x):.4f}$"


def ci(d):
    return f"{s4(d['mean'])} [{s4(d['lo'])}, {s4(d['hi'])}]"


def cell(v, ep, orig, post):
    t = f4(v)
    if ep == post:
        t = r"\textbf{" + t + "}"
    if ep == orig:
        t = r"\underline{" + t + "}"
    return t


def table_s2():
    runs = [(f"DiverseVul project-disj., seed {s}", M["a_per_seed"][s], f4(M["a_per_seed"][s]["phase2_test"]))
            for s in ("1", "2", "3")]
    d = M["d_devign"]
    runs.append(("Devign GraphCodeBERT family-disj., seed 2",
                 {"valid": d["valid"], "test": d["test"], "original_rule_epoch": d["original_rule_epoch"],
                  "post_warmup_epoch": d["post_warmup_epoch"]}, f4(d["original_run"]["test"])))
    ne = max(len(r["valid"]) for _, r, _ in runs)
    L = [r"\begin{tabular}{ll" + "r" * ne + "r}", r"\toprule",
         "Run & PR-AUC & " + " & ".join(f"Ep.~{e}" for e in range(1, ne + 1)) + r" & Reported \\", r"\midrule"]
    for name, r, rep in runs:
        o, p = r["original_rule_epoch"], r["post_warmup_epoch"]
        for i, key in enumerate(("valid", "test")):
            vals = [cell(v, e + 1, o, p) for e, v in enumerate(r[key])] + [""] * (ne - len(r[key]))
            L.append((name if i == 0 else "") + f" & {'validation' if key == 'valid' else 'test'} & "
                     + " & ".join(vals) + f" & {rep if key == 'test' else ''} " + r"\\")
        L.append(r"\addlinespace")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    a, b = M["a_paired_post_minus_original"], M
    cp, cw = M["c_primary"], M["c_post_warmup"]
    summary = [r"\begin{tabular}{lll}", r"\toprule",
               r"Quantity (seeds 1--3, paired by seed) & Reported runs & Post hoc rerun \\", r"\midrule",
               r"Selected epoch, original rule & 1, 1, 1 & " + ", ".join(str(M["a_per_seed"][s]["original_rule_epoch"]) for s in "123") + r" \\",
               r"Selected epoch, after warm-up & -- & " + ", ".join(str(M["a_per_seed"][s]["post_warmup_epoch"]) for s in "123") + r" \\",
               r"Test PR-AUC, after warm-up minus original rule & -- & " + ci(a) + r" \\",
               r"$\Delta(\text{random}\to\text{project})$ & " + ci(cp["random_to_project"]) + " & " + ci(cw["random_to_project"]) + r" \\",
               r"$\Delta(\text{commit}\to\text{project})$ & " + ci(cp["commit_to_project"]) + " & " + ci(cw["commit_to_project"]) + r" \\",
               r"Share $\rho$ & " + f"{cp['rho']:.2f} & {cw['rho']:.2f}" + r" \\",
               r"\bottomrule", r"\end{tabular}"]
    repro = (f"Reproduction under the original rule: the rerun's per-epoch validation PR-AUC differs from the reported "
             f"runs by at most {f4(b['b_max_abs_valid_diff'])} and its test PR-AUC by at most "
             f"{f4(b['b_max_abs_test_diff_original_rule'])}.")
    return "\n".join(L) + "\n\n\\medskip\n" + "\n".join(summary) + "\n\n\\medskip\\noindent " + repro + "\n"


ARMS = (("published", "Published"), ("random_control", "Random control"), ("family_disjoint", "Family-disjoint"))


def table_item57():
    L = [r"\begin{tabular}{lllrrr}", r"\toprule",
         r"Split & Test functions & Counterpart in training & \multicolumn{2}{c}{Vulnerable commit-mate in training} & Total \\",
         r"\cmidrule(lr){4-5}", r" & & & yes & no & \\", r"\midrule"]
    for k, name in ARMS:
        for j, (cls, lab) in enumerate((("test_positives", "positives"), ("test_negatives", "negatives"))):
            t = I57["arms"][k]["table"][cls]
            yes = (t["counterpart_and_vul_mate"], t["counterpart_only"])
            no = (t["vul_mate_only"], t["neither"])
            L.append(f"{name if j == 0 else ''} & {lab} (n = {t['n']:,}) & yes & {yes[0]:,} & {yes[1]:,} & {sum(yes):,}" + r" \\")
            L.append(f" & & no & {no[0]:,} & {no[1]:,} & {sum(no):,}" + r" \\")
        L.append(r"\addlinespace")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    P = [r"\begin{tabular}{lllll}", r"\toprule",
         r"Split (seeds 4--5) & Rate & Without mate (n) & With mate (n) & With minus without [95\% CI] \\", r"\midrule"]
    for k, name in ARMS[1:]:
        pr = I57["arms"][k]["predictions_no_counterpart_stratum"]
        for key, lab in (("recall_positives", "recall of positives"), ("fpr_negatives", "FPR of negatives")):
            v = pr[key]
            wo = st.mean(x["without_vul_mate"][0] for x in v["per_seed"])
            wi = st.mean(x["with_vul_mate"][0] for x in v["per_seed"])
            nwo, nwi = v["per_seed"][0]["without_vul_mate"][1], v["per_seed"][0]["with_vul_mate"][1]
            assert abs((wi - wo) - v["with_minus_without"]) < 1e-9
            P.append(f"{name} & {lab} & {f4(wo)} ({nwo:,}) & {f4(wi)} ({nwi:,}) & {s4(v['with_minus_without'])} "
                     f"[{s4(v['ci95'][0])}, {s4(v['ci95'][1])}]" + r" \\")
    P += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n\n\\medskip\n" + "\n".join(P) + "\n"


PE = json.load(open(os.path.join(ROOT, "results", "rq2_diversevul_phase2", "paired_eval.json"), encoding="utf-8"))
PAIR_ARMS = (("random_control", "Random control"), ("family_disjoint", "Family-disjoint"),
             ("commit_disjoint", "Commit-disjoint"), ("project_disjoint", "Project-disjoint"))


def table_pairs():
    """Table S3: the four pair outcomes of Ding et al. per split (descriptive), from paired_eval.json."""
    L = [r"\begin{tabular}{llrllll}", r"\toprule",
         r"Split & Seeds & Pairs & P-C & P-V & P-B & P-R \\", r"\midrule"]
    for k, name in PAIR_ARMS:
        a = PE["arms"][k]
        cells = [f"{f4(a['mean'][m])} [{f4(a['ci95'][m][0])}, {f4(a['ci95'][m][1])}]" for m in ("P-C", "P-V", "P-B", "P-R")]
        assert abs(sum(a["mean"][m] for m in ("P-C", "P-V", "P-B", "P-R")) - 1) < 1e-9
        seeds = "--".join(str(s) for s in (a["seeds"][0], a["seeds"][-1]))
        L.append(f"{name} & {seeds} & {a['pairs']:,} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def main():
    hdr = "% generated by analysis/scripts/supp_m1_item57.py -- do not edit by hand\n"
    open(os.path.join(SUPP, "pair_outcomes_fragment.tex"), "w", encoding="utf-8", newline="\n").write(hdr + table_pairs())
    open(os.path.join(SUPP, "m1_epochs_fragment.tex"), "w", encoding="utf-8", newline="\n").write(hdr + table_s2())
    open(os.path.join(SUPP, "item57_fragment.tex"), "w", encoding="utf-8", newline="\n").write(hdr + table_item57())
    cs = I57["commit_structure"]
    print("commit structure:", cs["commits_with_vulnerable_functions"], cs["commits_with_2plus_vulnerable_pct"],
          cs["vulnerable_functions_in_2plus_commits_pct"], cs["vulnerable_per_commit_max"])
    print("wrote m1_epochs_fragment.tex, item57_fragment.tex")


if __name__ == "__main__":
    main()
