#!/usr/bin/env python3
"""
make_decon_splits.py -- the two H4b arms (RESEARCH_PLAN §13.8, 21 Oct 2026; size control amended
the same day, before any H4/H4b pre-check or run). CPU only.

  commit_decon     commit_disjoint split minus every TRAIN function listed in the direct-edge
                   train-hit file (scripts/direct_clone_edges.py, split commit_disjoint)
  commit_sizectrl  commit_disjoint split minus the same number of TRAIN functions drawn at random
                   within strata of label x length decile, matching the decon removal's count in
                   every stratum exactly, seed 42. Length = characters; decile edges are computed
                   on the commit_disjoint train set. If a stratum lacks candidates, the remainder is
                   borrowed from the nearest length decile of the same label (nearer first, lower
                   decile first on ties) and counted in the report ("borrowed").
Candidate pool (--control-pool): "train_minus_decon" (default, as registered by the second §13.8
amendment of 21 Oct 2026: only commit_disjoint train functions that are NOT in the decon list, so
the control keeps every direct clone) or "train" (every train function; the first amendment's pool,
kept for comparison only). The pre-check refuses the split if borrowing exceeds 5% of the removal.
Valid and test are copied unchanged. A removed function is simply absent from the split file
(finetune_vd.py skips ids without a partition).

USAGE (server)
  D=$CFA_DATA_DIR
  python scripts/make_decon_splits.py --data $D/data/diversevul/diversevul.jsonl \
     --ref-splitfile $D/diversevul_audit_splitfile_commitdisjoint_seed42.json \
     --train-hits $D/direct_edges/diversevul_hr_direct_commit_disjoint_train_hits.json \
     --seed 42 --out-dir $D
OUTPUT
  <out-dir>/diversevul_audit_splitfile_commitdecon_seed42.json
  <out-dir>/diversevul_audit_splitfile_commitsizectrl_seed42.json
  <out-dir>/diversevul_h4b_splits_report.json   (incl. per-stratum counts and borrowing)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np


def load(path, id_field="idx"):
    """id -> (label, character length)."""
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
            f = o.get("func")
            if isinstance(f, str) and f.strip():
                out[str(o.get(id_field, n))] = (1 if o.get("target") == 1 else 0, len(f))
    return out


def decile_edges(lengths):
    """Nine inner edges of the character-length deciles (commit_disjoint train)."""
    return np.quantile(np.asarray(lengths, dtype=float), np.linspace(0.1, 0.9, 9))


def decile_of(length, edges):
    return int(np.searchsorted(edges, length, side="right"))          # 0..9


def build(ref, hits, info, seed, pool="train_minus_decon"):
    """Returns (decon, sizectrl, report). ref: id -> partition; hits: train ids to remove;
    info: id -> (label, length)."""
    train = sorted(i for i, p in ref.items() if p == "train")
    hits = sorted(set(hits))
    bad = [i for i in hits if ref.get(i) != "train"]
    if bad:
        raise ValueError(f"{len(bad)} listed ids are not in the reference train set, e.g. {bad[:3]}")
    hs = set(hits)
    edges = decile_edges([info[i][1] for i in train])
    stratum = {i: (info[i][0], decile_of(info[i][1], edges)) for i in train}
    need = {}
    for i in hits:
        need[stratum[i]] = need.get(stratum[i], 0) + 1
    cand = {}
    for i in train:
        if pool == "train_minus_decon" and i in hs:
            continue
        cand.setdefault(stratum[i], []).append(i)            # train is sorted, so lists are sorted
    rng = random.Random(seed)
    chosen, borrowed, unfilled, per = set(), 0, 0, {}
    for (lab, d) in sorted(need):
        k = need[(lab, d)]
        avail = [i for i in cand.get((lab, d), []) if i not in chosen]
        take = rng.sample(avail, min(k, len(avail)))
        short = k - len(take)
        b_here = 0
        for dd in sorted((x for x in range(10) if x != d), key=lambda x: (abs(x - d), x)):
            if short == 0:
                break
            tk = set(take)
            more = [i for i in cand.get((lab, dd), []) if i not in chosen and i not in tk]
            extra = rng.sample(more, min(short, len(more)))
            take += extra
            short -= len(extra)
            b_here += len(extra)
        # if even the whole label is exhausted, the control is left short; the pre-check then
        # refuses the split (size control does not match), so H4b is not run
        unfilled += short
        chosen.update(take)
        borrowed += b_here
        per[f"label{lab}_decile{d}"] = {"decon": k, "control_same_stratum": k - b_here - short,
                                        "borrowed": b_here, "unfilled": short}
    n_pos = sum(info[i][0] for i in hits)
    decon = {i: p for i, p in ref.items() if i not in hs}
    size = {i: p for i, p in ref.items() if i not in chosen}
    rep = {"seed": seed, "control_pool": pool, "length_decile_edges_chars": [float(x) for x in edges],
           "reference_train": len(train), "reference_train_pos": sum(info[i][0] for i in train),
           "removed": len(hits), "removed_pos": n_pos, "removed_neg": len(hits) - n_pos,
           "removed_pct_of_train": round(100 * len(hits) / len(train), 2) if train else None,
           "train_after": len(train) - len(hits),
           "train_pos_after": sum(info[i][0] for i in train) - n_pos,
           "borrowed": borrowed, "unfilled": unfilled, "borrowed_pct_of_removal": round(100 * borrowed / len(hits), 2) if hits else 0.0,
           "strata": per, "sizectrl_overlap_with_decon_removal": len(chosen & hs)}
    return decon, size, rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ref-splitfile", required=True)
    ap.add_argument("--train-hits", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--control-pool", choices=("train_minus_decon", "train"), default="train_minus_decon")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    for p in (a.data, a.ref_splitfile, a.train_hits):
        if not os.path.exists(p):
            print(f"!!! missing input: {p}", file=sys.stderr)
            return 2
    ref = {str(k): ("valid" if v == "validation" else v)
           for k, v in json.load(open(a.ref_splitfile, encoding="utf-8")).items()}
    hits = [str(x) for x in json.load(open(a.train_hits, encoding="utf-8"))]
    info = load(a.data)
    try:
        decon, size, rep = build(ref, hits, info, a.seed, a.control_pool)
    except (ValueError, KeyError) as e:
        print(f"!!! {e}", file=sys.stderr)
        return 2
    rep.update({"ref_splitfile": a.ref_splitfile, "train_hits": a.train_hits})
    for name, sp in (("commitdecon", decon), ("commitsizectrl", size)):
        json.dump(sp, open(os.path.join(a.out_dir, f"diversevul_audit_splitfile_{name}_seed{a.seed}.json"), "w",
                           encoding="utf-8"))
    json.dump(rep, open(os.path.join(a.out_dir, "diversevul_h4b_splits_report.json"), "w", encoding="utf-8"),
              indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k != "strata"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
