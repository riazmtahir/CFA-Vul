#!/usr/bin/env python3
"""
twin_effect_exploratory.py -- EXPLORATORY (not pre-registered; written 7 Oct 2026
after the own-twin audit). v2 adds the two controls the v1 result called for.

Question: does a test function whose OWN opposite-label twin (same commit, same
function name -- its patched / pre-patch version) is in TRAIN get scored
differently? Uses the Phase 2 predictions already on disk (random_control and
family_disjoint seeds 4-5; no GPU).

Each test function is put in one of three groups by where its twin is:
  twin_in_train   its twin is in train                       (exposed)
  twin_elsewhere  it has a twin, but in valid/test only      (same kind of
                  function -- a changed function of a fix commit -- not exposed)
  no_twin         no opposite-label same-commit same-name function at all
twin_in_train vs twin_elsewhere is the memorisation contrast: both groups are
functions touched by a fix; only exposure differs. twin_in_train vs no_twin
mixes exposure with "being a changed function".

Twin-having functions are much longer than ordinary ones and the model's score
tracks length, so every contrast is also reported LENGTH-STANDARDISED: rate
differences within length quintiles (bins from that class's test functions),
weighted by the exposed group's length distribution.

At the validation max-F1 threshold per run (same rule as H1):
  positives -> recall, negatives -> FPR; mean predicted probability; mean length.
CIs: percentile bootstrap over test functions within group, statistic averaged
over seeds. Writes <phase2>/twin_effect_exploratory.{json,md}. Report as exploratory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_phase2 import best_f1_threshold, load_preds  # noqa: E402
from commit_leak_audit import func_name  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)   # empty strata in tiny groups -> nan, reported as such
MODEL = "codebert-base"
LOCS = ("twin_in_train", "twin_elsewhere", "no_twin")


def load_corpus(path, id_field):
    info = {}
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
            info[str(o.get(id_field, n))] = (1 if o.get("target") == 1 else 0,
                                             str(o.get("commit_id") or ""),
                                             func_name(o["func"]), len(o["func"]))
    return info


def twin_location(info, split, fam):
    """test id -> (twin location, has any train family-mate)."""
    where = defaultdict(set)          # (commit, name, label) -> partitions holding it
    tr_fam = set()
    for i, s in split.items():
        if i not in info:
            continue
        lab, c, nm, _ = info[i]
        if c and nm:
            where[(c, nm, lab)].add(s)
        if s == "train" and fam.get(i) is not None:
            tr_fam.add(fam[i])
    out = {}
    for i, s in split.items():
        if s != "test" or i not in info:
            continue
        lab, c, nm, _ = info[i]
        parts = where.get((c, nm, 1 - lab), set()) if (c and nm) else set()
        loc = "twin_in_train" if "train" in parts else ("twin_elsewhere" if parts else "no_twin")
        out[i] = (loc, fam.get(i) in tr_fam)
    return out


def std_diff(pred, a, b, qbin):
    """Length-standardised rate difference a-b: within-quintile differences weighted
    by a's quintile distribution; quintiles with no b are dropped and weights renormalised."""
    num = den = 0.0
    for k in np.unique(qbin[a]):
        ak, bk = a & (qbin == k), b & (qbin == k)
        if not bk.any():
            continue
        w = ak.sum()
        num += w * (pred[ak].mean() - pred[bk].mean())
        den += w
    return num / den if den else None


def boot(runs, a, b, qbin, rng, n_boot):
    """CIs for raw and length-standardised differences (mean over seeds)."""
    ia, ib = np.flatnonzero(a), np.flatnonzero(b)
    if len(ia) == 0 or len(ib) == 0:
        return None, None
    K = int(qbin.max()) + 1
    raw, std = [], []
    for _ in range(n_boot):
        ra, rb = rng.choice(ia, len(ia)), rng.choice(ib, len(ib))
        qa, qb = qbin[ra], qbin[rb]
        ca, cb = np.bincount(qa, minlength=K), np.bincount(qb, minlength=K)
        ok = (ca > 0) & (cb > 0)
        r_vals, s_vals = [], []
        for p in runs:
            r_vals.append(p[ra].mean() - p[rb].mean())
            sa = np.bincount(qa, weights=p[ra], minlength=K)
            sb = np.bincount(qb, weights=p[rb], minlength=K)
            d = sa[ok] / ca[ok] - sb[ok] / cb[ok]
            s_vals.append((ca[ok] * d).sum() / ca[ok].sum() if ok.any() else np.nan)
        raw.append(np.mean(r_vals)); std.append(np.nanmean(s_vals))
    pc = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return pc(raw), pc(std)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disk", default=os.environ.get("CFA_DATA_DIR", "data"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    D = args.disk
    p2 = os.path.join(D, "rq2_diversevul_phase2")
    arms = {"random_control": os.path.join(D, "diversevul_audit_splitfile_randomctrl_theta0.7.json"),
            "family_disjoint": os.path.join(D, "diversevul_audit_splitfile_theta0.7.json")}
    info = load_corpus(os.path.join(D, "data/diversevul/diversevul.jsonl"), "idx")
    fam = {str(k): v for k, v in
           json.load(open(os.path.join(D, "diversevul_audit_families_theta0.7.json"))).items()}
    rng = np.random.default_rng(args.seed)
    f4 = lambda x: "—" if x is None else f"{x:+.4f}"
    f_ = lambda x: "—" if x is None else f"{x:.4f}"
    ci = lambda c: "—" if c is None else f"[{c[0]:+.4f}, {c[1]:+.4f}]"

    rep = {"note": "EXPLORATORY, not pre-registered (v2: twin location + length standardisation)",
           "arms": {}}
    L = ["# Own-twin effect on CodeBERT predictions (DiverseVul) -- EXPLORATORY v2", "",
         "Twin = same commit + same function name, opposite label. Groups by where the "
         "twin is: train / elsewhere (valid or test) / none. Threshold = validation max-F1 "
         "per run. 'std' = length-standardised (within length quintiles, weighted by the "
         f"exposed group). CIs: percentile bootstrap, {args.n_boot} resamples, mean over seeds.", ""]
    for arm, sf in arms.items():
        flags = twin_location(info, {str(k): v for k, v in json.load(open(sf)).items()}, fam)
        runs, seeds, ids0 = [], [], None
        for s in (4, 5):
            pr = load_preds(os.path.join(p2, f"{MODEL}_{arm}_seed{s}.preds.json.gz"))
            if pr is None:
                continue
            ids = [str(x) for x in pr["test"]["id"]]
            if ids0 is None:
                ids0 = ids
            elif ids != ids0:
                print(f"!!! {arm}: test ids differ between seeds", file=sys.stderr)
                return 2
            thr = best_f1_threshold(pr["valid"]["target"], pr["valid"]["prob"])
            prob = np.asarray(pr["test"]["prob"])
            runs.append(((prob >= thr).astype(float), prob))
            seeds.append(s)
        if not runs:
            L += [f"## {arm}: no prediction files found", ""]
            continue
        y = np.asarray([info[i][0] for i in ids0])
        loc = np.asarray([flags.get(i, ("no_twin", False))[0] for i in ids0])
        famleak = np.asarray([flags.get(i, ("no_twin", False))[1] for i in ids0])
        length = np.asarray([info[i][3] for i in ids0])
        preds = [p for p, _ in runs]

        arm_rep = {"seeds": seeds}
        L += [f"## {arm} (seeds {', '.join(map(str, seeds))})", ""]
        for cls, lab, metric in (("positives", 1, "recall"), ("negatives", 0, "FPR")):
            c = y == lab
            edges = np.quantile(length[c], [0.2, 0.4, 0.6, 0.8])
            qbin = np.digitize(length, edges)
            m = {g: c & (loc == g) for g in LOCS}
            if lab == 1 and arm == "random_control":
                m["twin_in_train & famleak"] = m["twin_in_train"] & famleak
                m["twin_in_train & no famleak"] = m["twin_in_train"] & ~famleak
            rows = {}
            for g, mk in m.items():
                rows[g] = {"n": int(mk.sum()),
                           metric: None if not mk.any() else float(np.mean([p[mk].mean() for p in preds])),
                           "mean_prob": None if not mk.any() else float(np.mean([pb[mk].mean() for _, pb in runs])),
                           "mean_len_chars": None if not mk.any() else float(length[mk].mean())}
            contrasts = {}
            for name, a, b in (("train_vs_elsewhere", "twin_in_train", "twin_elsewhere"),
                               ("train_vs_none", "twin_in_train", "no_twin")):
                if not m[a].any() or not m[b].any():
                    continue
                raw = float(np.mean([p[m[a]].mean() - p[m[b]].mean() for p in preds]))
                sd = [std_diff(p, m[a], m[b], qbin) for p in preds]
                sd = None if any(v is None for v in sd) else float(np.mean(sd))
                cr, cs = boot(preds, m[a], m[b], qbin, rng, args.n_boot)
                contrasts[name] = {"raw": raw, "raw_ci95": cr, "std": sd, "std_ci95": cs}
            arm_rep[cls] = {"groups": rows, "contrasts": contrasts}
            L += [f"### {cls} ({metric})", "",
                  f"| twin location | n | {metric} | mean prob | mean length (chars) |",
                  "|---|---|---|---|---|"]
            for g, v in rows.items():
                ln = "—" if v["mean_len_chars"] is None else str(round(v["mean_len_chars"]))
                L.append(f"| {g} | {v['n']:,} | {f_(v[metric])} | {f_(v['mean_prob'])} | {ln} |")
            L.append("")
            for name, v in contrasts.items():
                L.append(f"- {metric} {name.replace('_', ' ')}: raw **{f4(v['raw'])}** {ci(v['raw_ci95'])}"
                         f" · length-std **{f4(v['std'])}** {ci(v['std_ci95'])}")
            L.append("")
        rep["arms"][arm] = arm_rep

    out = os.path.join(p2, "twin_effect_exploratory")
    json.dump(rep, open(out + ".json", "w"), indent=2)
    open(out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
