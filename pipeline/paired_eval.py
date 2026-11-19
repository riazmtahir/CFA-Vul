#!/usr/bin/env python3
"""
paired_eval.py -- PrimeVul-style paired evaluation on DiverseVul from saved
predictions (CPU only, no retraining). DESCRIPTIVE: not a pre-registered test;
written 7 Oct 2026 before looking at any pairwise prediction.

A pair = the vulnerable function and its patched version: same commit_id, same
function name, opposite labels, BOTH in the test split of the arm. Only clean
groups (exactly one vulnerable and one patched function for that commit+name in
test) are used; ambiguous groups are counted and skipped. Functions whose name
cannot be parsed are excluded (same rule as commit_leak_audit.py's own-twin column).

Per run, at the validation max-F1 threshold (same rule as H1), each pair is
  P-C  vulnerable flagged, patched not      (correct pair)
  P-V  both flagged
  P-B  neither flagged
  P-R  patched flagged, vulnerable not      (reversed)
following Ding et al. (PrimeVul), plus the threshold-free
  rank-acc  P(score(vulnerable) > score(patched)), ties = 0.5; chance = 0.5.
Statistics are averaged over the arm's seeds; 95% CIs by bootstrap over pairs.

Arms (whichever have prediction files): random_control / family_disjoint
(Phase 2 seeds 4-5), project_disjoint (Phase 2 seeds 1-3), commit_disjoint
(Phase 3 seeds 1-3). Writes <phase2>/paired_eval.{json,md}.

--exclude-identical LEVEL (added 18 Oct 2026, action A4; default off, so the default
run is unchanged) drops pairs whose two versions are textually identical, where
LEVEL is raw (same string), ws (same after removing all whitespace, the rule of
Li et al. 2026), norm (same normalize_tokens stream of clone_family_audit.py) or
any (any of the three). Such pairs force a tie in rank accuracy. Output then goes
to <phase2>/paired_eval_noidentical.{json,md}. Counts per level:
scripts/pair_identity_check.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_phase2 import best_f1_threshold, load_preds  # noqa: E402
from twin_effect_exploratory import load_corpus  # noqa: E402

MODEL = "codebert-base"


def test_pairs(info, split):
    """Clean (vulnerable id, patched id) pairs with both members in test."""
    groups = defaultdict(lambda: ([], []))
    for i, s in split.items():
        if s != "test" or i not in info:
            continue
        lab, c, nm, _ = info[i]
        if c and nm:
            groups[(c, nm)][0 if lab == 1 else 1].append(i)
    pairs, ambiguous = [], 0
    for v, p in groups.values():
        if v and p:
            if len(v) == 1 and len(p) == 1:
                pairs.append((v[0], p[0]))
            else:
                ambiguous += 1
    return pairs, ambiguous


_WS = re.compile(r"\s+")
LEVELS = ("raw", "ws", "norm")


def load_texts(path, ids, id_field="idx"):
    """id -> function text, for the given ids only (same filter as load_corpus)."""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            i = str(o.get(id_field, n))
            if i in ids and isinstance(o.get("func"), str) and o["func"].strip():
                out[i] = o["func"]
    return out


def identical_levels(a, b):
    """Which identity levels hold for two function texts."""
    from clone_family_audit import normalize_tokens
    return {"raw": a == b,
            "ws": _WS.sub("", a) == _WS.sub("", b),
            "norm": normalize_tokens(a) == normalize_tokens(b)}


def pair_stats(fv, fp, sv, sp):
    """fv/fp: flags (0/1) of vulnerable/patched members; sv/sp: scores."""
    return {"P-C": float(np.mean((fv == 1) & (fp == 0))),
            "P-V": float(np.mean((fv == 1) & (fp == 1))),
            "P-B": float(np.mean((fv == 0) & (fp == 0))),
            "P-R": float(np.mean((fv == 0) & (fp == 1))),
            "rank_acc": float(np.mean((sv > sp) + 0.5 * (sv == sp)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disk", default=os.environ.get("CFA_DATA_DIR", "data"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--exclude-identical", choices=LEVELS + ("any",), default=None,
                    help="drop pairs identical at this level; writes paired_eval_noidentical.*")
    args = ap.parse_args()
    D = args.disk
    p2, p3 = os.path.join(D, "rq2_diversevul_phase2"), os.path.join(D, "rq2_diversevul_phase3")
    arms = [("random_control", "diversevul_audit_splitfile_randomctrl_theta0.7.json", p2, (4, 5)),
            ("family_disjoint", "diversevul_audit_splitfile_theta0.7.json", p2, (4, 5)),
            ("project_disjoint", "diversevul_audit_splitfile_projectdisjoint_seed42.json", p2, (1, 2, 3)),
            ("commit_disjoint", "diversevul_audit_splitfile_commitdisjoint_seed42.json", p3, (1, 2, 3))]
    corpus = os.path.join(D, "data/diversevul/diversevul.jsonl")
    info = load_corpus(corpus, "idx")
    texts = None
    rng = np.random.default_rng(args.seed)
    KEYS = ("P-C", "P-V", "P-B", "P-R", "rank_acc")

    rep = {"note": "DESCRIPTIVE, not pre-registered", "arms": {}}
    L = ["# Paired evaluation (PrimeVul-style) on DiverseVul test pairs", "",
         "Pair = vulnerable + patched version of the same function (same commit, same name), "
         "both in test. Validation max-F1 threshold per run. P-C correct pair · P-V both "
         "flagged · P-B neither · P-R reversed · rank-acc = P(score vuln > score patched), "
         f"chance 0.5. Mean over seeds; 95% bootstrap CI over pairs ({args.n_boot}).", "",
         "| arm | seeds | pairs (ambiguous skipped) | P-C | P-V | P-B | P-R | rank-acc |",
         "|---|---|---|---|---|---|---|---|"]
    for arm, sf, d, seeds in arms:
        path = os.path.join(D, sf)
        if not os.path.exists(path):
            continue
        split = {str(k): v for k, v in json.load(open(path)).items()}
        pairs, amb = test_pairs(info, split)
        n_excluded = None
        if args.exclude_identical:
            if texts is None:     # one pass over the corpus for every arm's pair members
                need = set()
                for _, sf2, _, _ in arms:
                    if os.path.exists(os.path.join(D, sf2)):
                        sp2 = {str(k): v for k, v in json.load(open(os.path.join(D, sf2))).items()}
                        need.update(x for pr2 in test_pairs(info, sp2)[0] for x in pr2)
                texts = load_texts(corpus, need)
            keep = []
            for v, p in pairs:
                lv = identical_levels(texts[v], texts[p])
                if not (any(lv.values()) if args.exclude_identical == "any" else lv[args.exclude_identical]):
                    keep.append((v, p))
            n_excluded, pairs = len(pairs) - len(keep), keep
        runs = []
        for s in seeds:
            pr = load_preds(os.path.join(d, f"{MODEL}_{arm}_seed{s}.preds.json.gz"))
            if pr is None:
                continue
            thr = best_f1_threshold(pr["valid"]["target"], pr["valid"]["prob"])
            score = dict(zip((str(x) for x in pr["test"]["id"]), pr["test"]["prob"]))
            if any(v not in score or p not in score for v, p in pairs):
                print(f"!!! {arm} seed{s}: pair member missing from predictions", file=sys.stderr)
                return 2
            sv = np.asarray([score[v] for v, _ in pairs]); sp = np.asarray([score[p] for _, p in pairs])
            runs.append((s, (sv >= thr).astype(int), (sp >= thr).astype(int), sv, sp))
        if not runs or not pairs:
            continue
        # length check: fixes often ADD code, and the model's score tracks length
        lv = np.asarray([info[v][3] for v, _ in pairs]); lp = np.asarray([info[p][3] for _, p in pairs])
        longer = lp > lv
        def ra(mask):
            if not mask.any():
                return None
            return float(np.mean([np.mean((sv[mask] > sp[mask]) + 0.5 * (sv[mask] == sp[mask]))
                                  for _, _, _, sv, sp in runs]))
        # length-balanced rank accuracy: mean of the two length strata, so the
        # length bias cancels; bootstrap over pairs within each stratum
        def bal(idx_l, idx_n):
            return float(np.mean([0.5 * (np.mean((sv[idx_l] > sp[idx_l]) + 0.5 * (sv[idx_l] == sp[idx_l]))
                                         + np.mean((sv[idx_n] > sp[idx_n]) + 0.5 * (sv[idx_n] == sp[idx_n])))
                                  for _, _, _, sv, sp in runs]))
        il, inl = np.flatnonzero(longer), np.flatnonzero(~longer)
        bal_mean = bal_ci = None
        if len(il) and len(inl):
            bal_mean = bal(il, inl)
            bb = [bal(rng.choice(il, len(il)), rng.choice(inl, len(inl))) for _ in range(args.n_boot)]
            bal_ci = [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))]
        length_check = {"balanced_rank_acc": bal_mean, "balanced_rank_acc_ci95": bal_ci,
                        "patched_longer_frac": float(longer.mean()),
                        "length_only_rank_acc": float(np.mean((lv > lp) + 0.5 * (lv == lp))),
                        "rank_acc_patched_longer": ra(longer),
                        "rank_acc_patched_not_longer": ra(~longer),
                        "n_patched_longer": int(longer.sum()), "n_patched_not_longer": int((~longer).sum())}
        per_seed = {f"s{s}": pair_stats(fv, fp, sv, sp) for s, fv, fp, sv, sp in runs}
        mean = {k: float(np.mean([v[k] for v in per_seed.values()])) for k in KEYS}
        boots = {k: [] for k in KEYS}
        n = len(pairs)
        for _ in range(args.n_boot):
            idx = rng.integers(0, n, n)
            bs = [pair_stats(fv[idx], fp[idx], sv[idx], sp[idx]) for _, fv, fp, sv, sp in runs]
            for k in KEYS:
                boots[k].append(np.mean([b[k] for b in bs]))
        ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
        rep["arms"][arm] = {"seeds": [r[0] for r in runs], "pairs": n, "ambiguous_groups": amb,
                            "mean": mean, "ci95": ci, "per_seed": per_seed,
                            "length_check": length_check}
        if n_excluded is not None:
            rep["arms"][arm]["identical_pairs_excluded"] = n_excluded
        cell = lambda k: f"{mean[k]:.3f} [{ci[k][0]:.3f}, {ci[k][1]:.3f}]"
        L.append(f"| {arm} | {','.join(str(r[0]) for r in runs)} | {n:,} ({amb}) | "
                 + " | ".join(cell(k) for k in KEYS) + " |")

    L += ["", "Length check -- a length-only scorer ranks the LONGER function as vulnerable; "
          "rank-acc split by whether the patched version is longer than the vulnerable one.", "",
          "| arm | patched longer | length-only rank-acc | rank-acc, patched longer | rank-acc, patched not longer | length-balanced rank-acc |",
          "|---|---|---|---|---|---|"]
    g = lambda x: "—" if x is None else f"{x:.3f}"
    for arm, r in rep["arms"].items():
        c = r["length_check"]
        L.append(f"| {arm} | {c['patched_longer_frac']:.3f} | {c['length_only_rank_acc']:.3f} | "
                 f"{g(c['rank_acc_patched_longer'])} (n={c['n_patched_longer']}) | "
                 f"{g(c['rank_acc_patched_not_longer'])} (n={c['n_patched_not_longer']}) | "
                 + ("—" if c["balanced_rank_acc"] is None else
                    f"{c['balanced_rank_acc']:.3f} [{c['balanced_rank_acc_ci95'][0]:.3f}, "
                    f"{c['balanced_rank_acc_ci95'][1]:.3f}]") + " |")

    if args.exclude_identical:
        rep["exclude_identical"] = args.exclude_identical
        L[0] += f" -- identical pairs EXCLUDED (level: {args.exclude_identical})"
        L.append("")
        L.append("Identical pairs excluded per arm: " + ", ".join(
            f"{a} {r['identical_pairs_excluded']}" for a, r in rep["arms"].items()))
    out = os.path.join(p2, "paired_eval_noidentical" if args.exclude_identical else "paired_eval")
    json.dump(rep, open(out + ".json", "w"), indent=2)
    open(out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
