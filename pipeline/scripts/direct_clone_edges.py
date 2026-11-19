#!/usr/bin/env python3
"""
direct_clone_edges.py -- DIRECT high-recall clone edges between test and training functions
(no transitive closure), for H4b (RESEARCH_PLAN §13.8, 21 Oct 2026). CPU only.

Criterion: the high-recall variant of the audit (blind identifiers, 3-token shingles,
theta = 0.7), with clone_family_audit.py's own preprocessing (imported, unchanged):
normalise -> collapse functions with identical normal forms onto one representative ->
shingle -> MinHash (K = 128) -> LSH (16 bands x 8 rows). Every test x train candidate pair
that shares an LSH bucket in any band is verified by EXACT Jaccard; a pair is an edge iff
J >= theta. There is no union-find: an edge means the two functions themselves are similar.
Functions with the same normal form (same representative) have J = 1 and are always edges.
Unlike build_families(), large buckets are not pivot-approximated: all test x train pairs in
a bucket are verified (vectorised sparse products for big blocks).

Recall caveat: a pair at J = 0.7 shares at least one bucket with probability 0.61
(1 - (1 - 0.7^8)^16), rising to 0.99 at J = 0.85, so the direct leak rate is a lower bound
with respect to LSH misses. --exhaustive verifies ALL test x train pairs (tests, small data).

Splits: "published" = the corpus's own split field; every --splitfile NAME=PATH adds one.
Outputs (prefix --out):
  <out>_summary.{json,md}           per split: direct leak % (all / positives), train functions hit
  <out>_<split>_train_hits.json     sorted ids of train functions with >= 1 direct edge to test
  <out>_<split>_edges.jsonl.gz      one line per leaked test function: {"test", "label", "train": [ids]}
                                     (omit with --no-lists)
USAGE (server)
  D=$CFA_DATA_DIR
  python scripts/direct_clone_edges.py --data $D/data/diversevul/diversevul.jsonl \
     --splitfile random_control=$D/diversevul_audit_splitfile_randomctrl_theta0.7.json \
     --splitfile family_disjoint=$D/diversevul_audit_splitfile_theta0.7.json \
     --splitfile project_disjoint=$D/diversevul_audit_splitfile_projectdisjoint_seed42.json \
     --splitfile commit_disjoint=$D/diversevul_audit_splitfile_commitdisjoint_seed42.json \
     --out $D/direct_edges/diversevul_hr_direct
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix

VD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VD)
from clone_family_audit import jaccard_sorted, load_jsonl, preprocess  # noqa: E402

SMALL_BLOCK = 256          # test x train pairs below this are verified one by one


def verify_block(T, R, store, theta):
    """All (t, r) with t in T, r in R, t != r, exact Jaccard >= theta. T, R: rep positions."""
    if len(T) * len(R) <= SMALL_BLOCK:
        out = []
        for t in T:
            st = store.get(t)
            for r in R:
                if r != t and jaccard_sorted(st, store.get(r)) >= theta:
                    out.append((t, r))
        return out
    allp = list(T) + list(R)
    sh = [store.get(p) for p in allp]
    lens = np.fromiter((x.size for x in sh), dtype=np.int64, count=len(sh))
    flat = np.concatenate(sh) if lens.sum() else np.empty(0, dtype=np.uint32)
    _, inv = np.unique(flat, return_inverse=True)
    indptr = np.concatenate([[0], np.cumsum(lens)])
    M = csr_matrix((np.ones(flat.size, dtype=np.int32), inv.ravel(), indptr),
                   shape=(len(allp), int(inv.max()) + 1 if flat.size else 1))
    A, B = M[:len(T)], M[len(T):].T.tocsc()
    la, lb = lens[:len(T)], lens[len(T):]
    Ta, Ra = np.asarray(T), np.asarray(R)
    out = []
    for s in range(0, len(T), 512):
        inter = (A[s:s + 512] @ B).toarray()
        den = la[s:s + 512, None] + lb[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            J = np.where(den > 0, inter / np.maximum(den, 1), 0.0)
        ii, jj = np.nonzero(J >= theta)
        keep = Ta[s + ii] != Ra[jj]
        out.extend(zip(Ta[s + ii][keep].tolist(), Ra[jj][keep].tolist()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--splitfile", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--no-published", action="store_true")
    ap.add_argument("--id-field", default="idx")
    ap.add_argument("--theta", type=float, default=0.7)
    ap.add_argument("--shingle-k", type=int, default=3)
    ap.add_argument("--rename", default="blind", choices=("blind", "positional"))
    ap.add_argument("--perms", type=int, default=128)
    ap.add_argument("--bands", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exhaustive", action="store_true", help="verify all test x train pairs (no LSH)")
    ap.add_argument("--no-lists", action="store_true")
    ap.add_argument("--report-every", type=int, default=50000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    for spec in a.splitfile:
        if "=" not in spec or not os.path.exists(spec.split("=", 1)[1]):
            print(f"!!! missing input: {spec}", file=sys.stderr)
            return 2
    if not os.path.exists(a.data):
        print(f"!!! missing input: {a.data}", file=sys.stderr)
        return 2

    rows = load_jsonl(a.data, "func", "target", "split", a.id_field)
    ids = [r["id"] for r in rows]
    norm = lambda s: "valid" if s in ("valid", "validation") else s
    splits = {} if a.no_published else {"published": [norm(r["split"]) for r in rows]}
    for spec in a.splitfile:
        name, path = spec.split("=", 1)
        m = {str(k): norm(v) for k, v in json.load(open(path, encoding="utf-8")).items()}
        splits[name] = [m.get(i) for i in ids]
    print(f"{len(rows):,} functions; splits {list(splits)}", flush=True)

    reps, member_of_rep, store, sigs, _ = preprocess(rows, a.shingle_k, a.perms, a.seed, a.report_every,
                                                     rename=a.rename)
    m = len(reps)
    members = [member_of_rep[r] for r in reps]
    rep_of = np.empty(len(rows), dtype=np.int64)
    for pos, mem in enumerate(members):
        rep_of[mem] = pos
    role = {}
    for s, part in splits.items():
        te = np.zeros(m, dtype=bool)
        tr = np.zeros(m, dtype=bool)
        for i, p in enumerate(part):
            if p == "test":
                te[rep_of[i]] = True
            elif p == "train":
                tr[rep_of[i]] = True
        role[s] = (te, tr)

    edges = {s: [] for s in splits}            # int64 keys t * m + r, deduplicated per band
    if a.exhaustive:
        for s, (te, tr) in role.items():
            got = verify_block(np.flatnonzero(te).tolist(), np.flatnonzero(tr).tolist(), store, a.theta)
            if got:
                edges[s].append(np.unique(np.array([t * m + r for t, r in got], dtype=np.int64)))
    else:
        rpb = a.perms // a.bands
        for bi in range(a.bands):
            seg = np.ascontiguousarray(sigs[:, bi * rpb:(bi + 1) * rpb])
            buckets = defaultdict(list)
            for i in range(m):
                buckets[seg[i].tobytes()].append(i)
            found = {s: [] for s in splits}
            for mem in buckets.values():
                if len(mem) < 2:
                    continue
                for s, (te, tr) in role.items():
                    T = [p for p in mem if te[p]]
                    if not T:
                        continue
                    R = [p for p in mem if tr[p]]
                    if R and not (len(T) == 1 and R == T):
                        found[s].extend(verify_block(T, R, store, a.theta))
            for s in splits:
                if found[s]:
                    k = np.array([t * m + r for t, r in found[s]], dtype=np.int64)
                    edges[s] = [np.unique(np.concatenate(edges[s] + [k]))]
            print(f"  band {bi + 1}/{a.bands}: " + ", ".join(
                f"{s} {edges[s][0].size if edges[s] else 0:,}" for s in splits) + f" rep edges ({time.time() - t0:.0f}s)",
                flush=True)

    summary = {"criterion": {"rename": a.rename, "shingle_k": a.shingle_k, "theta": a.theta, "perms": a.perms,
                             "bands": a.bands, "exhaustive": a.exhaustive,
                             "lsh_pair_prob_at_theta": round(1 - (1 - a.theta ** (a.perms // a.bands)) ** a.bands, 3)},
               "n_functions": len(rows), "n_representatives": m, "splits": {}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for s, part in splits.items():
        te, tr = role[s]
        key = edges[s][0] if edges[s] else np.empty(0, dtype=np.int64)
        tpos, rpos = key // m, key % m
        nbr = defaultdict(set)
        for t, r in zip(tpos.tolist(), rpos.tolist()):
            nbr[t].add(r)
        train_mem = lambda p: [i for i in members[p] if part[i] == "train"]
        hit_train = set()
        test_rows = [i for i, p in enumerate(part) if p == "test"]
        leaked = leaked_pos = 0
        lines = []
        for i in test_rows:
            p = rep_of[i]
            cl = set(train_mem(p))                       # same normal form: J = 1
            for r in nbr.get(p, ()):
                cl.update(train_mem(r))
            if cl:
                leaked += 1
                leaked_pos += rows[i]["label"]
                hit_train |= cl
                if not a.no_lists:
                    lines.append(json.dumps({"test": ids[i], "label": rows[i]["label"],
                                             "train": sorted(ids[j] for j in cl)}))
        n_pos = sum(rows[i]["label"] for i in test_rows)
        tr_rows = [i for i, p in enumerate(part) if p == "train"]
        summary["splits"][s] = {
            "n_test": len(test_rows), "n_test_pos": n_pos,
            "direct_leaked_test": leaked, "direct_leaked_test_pct": round(100 * leaked / len(test_rows), 2)
            if test_rows else None,
            "direct_leaked_test_pos": leaked_pos, "direct_leaked_test_pos_pct": round(100 * leaked_pos / n_pos, 2)
            if n_pos else None,
            "n_train": len(tr_rows), "train_hit": len(hit_train),
            "train_hit_pct": round(100 * len(hit_train) / len(tr_rows), 2) if tr_rows else None,
            "train_hit_pos": sum(rows[j]["label"] for j in hit_train),
            "train_pos": sum(rows[j]["label"] for j in tr_rows),
            "rep_edges": int(key.size), "mixed_representatives": int((te & tr).sum())}
        json.dump(sorted(ids[j] for j in hit_train), open(f"{a.out}_{s}_train_hits.json", "w", encoding="utf-8"))
        if not a.no_lists:
            with gzip.open(f"{a.out}_{s}_edges.jsonl.gz", "wt", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    json.dump(summary, open(f"{a.out}_summary.json", "w", encoding="utf-8"), indent=2)
    L = ["# Direct high-recall clone edges, test -> train (no transitive closure)", "",
         f"Criterion: {a.rename} identifiers, {a.shingle_k}-token shingles, exact Jaccard >= {a.theta}; "
         + ("all test x train pairs verified." if a.exhaustive else
            f"LSH candidates ({a.bands} bands x {a.perms // a.bands} rows; P(candidate | J = {a.theta}) = "
            f"{summary['criterion']['lsh_pair_prob_at_theta']}), so the rates are lower bounds."), "",
         "| split | test n | direct leak % | positives: direct leak % | train functions hit (% of train) | "
         "positives hit / train positives |", "|---|---|---|---|---|---|"]
    for s, r in summary["splits"].items():
        L.append(f"| {s} | {r['n_test']:,} | {r['direct_leaked_test_pct']} | {r['direct_leaked_test_pos_pct']} | "
                 f"{r['train_hit']:,} ({r['train_hit_pct']}%) | {r['train_hit_pos']:,} / {r['train_pos']:,} |")
    open(f"{a.out}_summary.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
