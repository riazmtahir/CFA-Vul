#!/usr/bin/env python3
"""
clone_family_audit.py  (v2)
===========================
Type-3 clone-family contamination audit for C/C++ vulnerability-detection
benchmarks, plus the 1-nearest-neighbour clone-retrieval baseline.

WHAT IT ANSWERS
---------------
RQ1  After a benchmark deduplicates at the exact / near-exact level, how many
     Type-3 clone families still span its train and test partitions, and what
     fraction of test functions have a near-duplicate ancestor in train?

RQ3  The 1-NN retrieval baseline: for each test function, copy the label of its
     most similar *training* function. Contains no learning whatsoever. If it
     rivals a fine-tuned transformer, the benchmark rewards family memorisation.

WHY v2 EXISTS (defects fixed from v1 — all found by testing, not by inspection)
-------------------------------------------------------------------------------
 1. CORRECTNESS: v1's MinHash did `a*x % (2**61-1)` in uint64. With a,x up to
    2^61 the product silently overflowed, so it was not a valid universal hash
    and Jaccard estimates were biased. v2 constrains a,b < 2^31 and x < 2^32 so
    every product fits in uint64. Verified against Python big-int arithmetic.
 2. MEMORY: v1 held one Python `set` of ints per function (~3 GB at 180k
    functions) and one global candidate-pair `set` (unbounded). v2 stores
    shingles in a single flat uint32 array with CSR-style offsets (~200 MB at
    180k) and never materialises a global pair set.
 3. SPEED: v1 hashed every shingle with a pure-Python FNV-1a loop over bytes.
    v2 maps tokens to integer ids once, then computes k-gram polynomial hashes
    with a vectorised sliding window. Roughly two orders of magnitude faster.
 4. SILENT DATA LOSS: v1 skipped LSH buckets larger than 3000 with only a
    printed warning, dropping real families from the counts. v2 processes every
    bucket (exact pairwise below --exhaustive-bucket, pivot-based above) and
    reports exactly how many buckets took which path.
 5. Added --limit for trial runs, --report-every, peak-memory reporting, and a
    deterministic --seed so runs are reproducible across machines.

METHOD (and its documented approximations)
------------------------------------------
  normalise (Type-2)  ->  collapse exact duplicates to representatives
  ->  k-shingle  ->  MinHash (K perms)  ->  LSH banding  ->  candidate pairs
  ->  EXACT Jaccard verification  ->  union-find  ->  families

  * Candidate generation is recall-oriented and approximate; every reported
    clone pair is confirmed by an exact Jaccard computation, so the *precision*
    of the family assignment is exact. Recall is bounded by the LSH S-curve.
  * For buckets larger than --exhaustive-bucket (default 200) we verify each
    member against a pivot rather than all pairs, to keep the cost linear. With
    --bands independent bucketings, connectivity is recovered in practice. The
    report states how many buckets used this path so the approximation is
    visible in the paper rather than hidden.

CHOOSING THETA (characterised on BigCloneBench)
-----------------------------------------------
`analysis/scripts/calibrate_theta_ours.py` scores every BigCloneBench pair with
this file's own scorer (normalize_tokens, 5-token shingles, exact Jaccard;
results/rq0_calibration_ours/). At theta=0.7 the criterion captures 30.0% of
Strong Type-3 pairs (57.3% of the very-strong ones, BCB sim >= 0.9), 0.3% of
Moderate Type-3 and 0.0% of Weak Type-3/Type-4 and non-clone pairs; at 0.8 /
0.9 Strong Type-3 falls to 22.5% / 15.6%. The older figures from
`calibrate_theta_bcb.py` (99.0% Strong Type-3 at 0.7) come from TouSimCC's
unigram similarity and do not describe this scorer. Report 0.7 as primary and
0.8/0.9 as stricter bounds.

Consequence worth stating in any paper that uses this tool: on 5-token shingles
theta=0.7 is a high-precision, low-recall near-duplicate criterion, and weak
Type-3 and Type-4 (semantic) clones fall below the cut entirely.
**Contamination measured here is therefore a conservative LOWER BOUND on true
clone-family overlap**, which is the right direction for an audit.

DEPENDENCIES
------------
numpy only (pandas not required). Nothing that touches your `concord` env.

INPUT
-----
JSONL, one object per line. Defaults match the VulnScout release and
PrimeVul/BigVul/Devign dumps:  idx / func / target  (+ optional split, project)

USAGE
-----
  # decisive Week-1 run, single file carrying a split column
  python clone_family_audit.py --data bigvul_all.jsonl \
      --theta 0.7 0.8 0.9 --knn-baseline --out results/bigvul

  # separate split files (VulnScout / PrimeVul layout)
  python clone_family_audit.py \
      --train train.jsonl --valid valid.jsonl --test test.jsonl \
      --theta 0.8 --knn-baseline --out results/primevul

  # 20k-function trial first, to check runtime on your machine
  python clone_family_audit.py --data bigvul_all.jsonl --limit 20000 --theta 0.8

OUTPUT
------
  <out>_families_theta<T>.json   function id -> family id (the citable artifact)
  <out>_splitfile_theta<T>.json  family-disjoint train/valid/test assignment
  <out>_splitfile_randomctrl_theta<T>.json
                                 control arm: same partition sizes and label
                                 balance, families ignored, leakage retained
  <out>_report.md  /  _report.json

Every number printed is computed from the input. Nothing is estimated.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

# ===========================================================================
# 1. C/C++ normalisation  (Type-2 canonical form)
# ===========================================================================

C_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if", "inline",
    "int", "long", "register", "restrict", "return", "short", "signed",
    "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned",
    "void", "volatile", "while", "_Bool", "_Complex", "bool", "true", "false",
    "NULL", "nullptr", "class", "public", "private", "protected", "virtual",
    "template", "typename", "namespace", "using", "new", "delete", "try",
    "catch", "throw", "this", "operator", "friend", "explicit", "export",
    "mutable", "constexpr", "decltype", "noexcept", "static_cast",
    "dynamic_cast", "const_cast", "reinterpret_cast",
}

# Library calls that carry vulnerability signal are preserved verbatim;
# renaming them would destroy exactly the information under study.
PRESERVED_CALLS = {
    "malloc", "calloc", "realloc", "free", "memcpy", "memmove", "memset",
    "strcpy", "strncpy", "strcat", "strncat", "strlen", "strcmp", "strncmp",
    "sprintf", "snprintf", "vsprintf", "vsnprintf", "printf", "fprintf",
    "scanf", "sscanf", "fscanf", "gets", "fgets", "puts", "fputs", "open",
    "close", "read", "write", "fopen", "fclose", "fread", "fwrite", "system",
    "exec", "execl", "execve", "popen", "alloca", "kmalloc", "kfree", "kzalloc",
    "vmalloc", "vfree", "copy_from_user", "copy_to_user", "get_user",
    "put_user", "assert", "abort", "exit", "atoi", "atol", "strtol", "strtoul",
}

_TOKEN_RE = re.compile(
    r'"(?:\\.|[^"\\])*"'          # 1 string literal
    r"|'(?:\\.|[^'\\])*'"         # 2 char literal
    r"|\b0[xX][0-9a-fA-F]+[uUlL]*\b"
    r"|\b\d+\.?\d*(?:[eE][+-]?\d+)?[uUlLfF]*\b"
    r"|[A-Za-z_]\w*"              # identifier / keyword
    r"|[-+*/%=<>!&|^~?:;,.\[\]{}()#]"
)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n]*")
_IDENT_START = re.compile(r"[A-Za-z_]")


def normalize_tokens(code: str) -> list[str]:
    """Type-2 normalised token stream: consistent identifier renaming,
    literals collapsed, comments stripped, keywords and security-relevant
    library calls preserved."""
    code = _BLOCK_COMMENT.sub(" ", code)
    code = _LINE_COMMENT.sub(" ", code)

    out: list[str] = []
    rename: dict[str, str] = {}
    ap = out.append
    for val in _TOKEN_RE.findall(code):
        c = val[0]
        if c == '"':
            ap("STR")
        elif c == "'":
            ap("CHR")
        elif c.isdigit():
            ap("NUM")
        elif _IDENT_START.match(c):
            if val in C_KEYWORDS or val in PRESERVED_CALLS:
                ap(val)
            else:
                r = rename.get(val)
                if r is None:
                    r = f"V{len(rename)}"
                    rename[val] = r
                ap(r)
        else:
            ap(val)
    return out


def blind_tokens(code: str) -> list[str]:
    """normalize_tokens with every renamed identifier mapped to one symbol "ID"
    (BigCloneBench-style blind renaming; keywords and PRESERVED_CALLS kept).
    Same definition as blind() in analysis/scripts/calibrate_theta_decompose.py.
    Used only with --rename blind (exploratory high-recall audit)."""
    return ["ID" if t[0] == "V" and t[1:].isdigit() else t for t in normalize_tokens(code)]


TOKENIZERS = {"positional": normalize_tokens, "blind": blind_tokens}


# ===========================================================================
# 2. Shingling  (vectorised k-gram polynomial hashing)
# ===========================================================================

_B = np.uint64(1099511628211)          # FNV prime, used as polynomial base
_MASK32 = np.uint64(0xFFFFFFFF)


def shingle_ids(token_ids: np.ndarray, k: int) -> np.ndarray:
    """Unique 32-bit hashes of the k-grams of a token-id sequence.

    Overflow in uint64 multiplication is intentional here: wraparound modulo
    2**64 is a standard polynomial hash. The result is masked to 32 bits so it
    can be fed to the (overflow-free) universal hash in minhash_block().
    """
    n = token_ids.size
    if n == 0:
        return np.empty(0, dtype=np.uint32)
    kk = min(k, n)
    win = np.lib.stride_tricks.sliding_window_view(token_ids, kk)  # (n-kk+1, kk)
    powers = np.empty(kk, dtype=np.uint64)
    p = 1
    for i in range(kk - 1, -1, -1):
        powers[i] = np.uint64(p)
        p = (p * 1099511628211) & 0xFFFFFFFFFFFFFFFF   # explicit wraparound
    with np.errstate(over="ignore"):
        h = (win.astype(np.uint64) * powers[None, :]).sum(axis=1, dtype=np.uint64)
        h ^= h >> np.uint64(29)
        h = h * np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(32)
    return np.unique((h & _MASK32).astype(np.uint32))


# ===========================================================================
# 3. MinHash  (overflow-free universal hashing)
# ===========================================================================

_P = np.uint64((1 << 61) - 1)          # Mersenne prime
# a,b < 2**31 and x < 2**32  =>  a*x + b  <  2**63  =>  fits in uint64.
_A_MAX = 1 << 31


def make_perms(K: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _A_MAX, size=K, dtype=np.uint64)
    b = rng.integers(0, _A_MAX, size=K, dtype=np.uint64)
    return a, b


def minhash_block(shingles: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if shingles.size == 0:
        return np.full(a.shape[0], _P, dtype=np.uint64)
    x = shingles.astype(np.uint64)
    return ((a[:, None] * x[None, :] + b[:, None]) % _P).min(axis=1)


def _selftest_minhash() -> None:
    """Assert the universal hash matches Python big-int arithmetic exactly.
    This is the bug that silently corrupted v1; it must never regress."""
    a, b = make_perms(8, 12345)
    x = np.array([0, 1, 2**32 - 1, 987654321], dtype=np.uint32)
    got = ((a[:, None] * x[None, :].astype(np.uint64) + b[:, None]) % _P)
    P = int(_P)
    for i in range(a.size):
        for j in range(x.size):
            want = (int(a[i]) * int(x[j]) + int(b[i])) % P
            assert int(got[i, j]) == want, (
                f"universal hash overflow regression: {int(got[i,j])} != {want}")


# ===========================================================================
# 4. Union-find
# ===========================================================================

class UnionFind:
    __slots__ = ("p", "r")

    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = bytearray(n)

    def find(self, x: int) -> int:
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.r[rx] < self.r[ry]:
            rx, ry = ry, rx
        self.p[ry] = rx
        if self.r[rx] == self.r[ry]:
            self.r[rx] += 1
        return True


# ===========================================================================
# 5. Shingle store (CSR-style flat arrays — the memory fix)
# ===========================================================================

class ShingleStore:
    """All shingles in one flat uint32 array + per-item offsets."""

    def __init__(self):
        self._chunks: list[np.ndarray] = []
        self._offs: list[int] = [0]
        self.flat: np.ndarray | None = None

    def add(self, arr: np.ndarray) -> None:
        self._chunks.append(arr)
        self._offs.append(self._offs[-1] + arr.size)

    def finalize(self) -> None:
        self.flat = (np.concatenate(self._chunks) if self._chunks
                     else np.empty(0, dtype=np.uint32))
        self.offs = np.asarray(self._offs, dtype=np.int64)
        self._chunks = []

    def get(self, i: int) -> np.ndarray:
        return self.flat[self.offs[i]:self.offs[i + 1]]

    @property
    def nbytes(self) -> int:
        return (0 if self.flat is None else self.flat.nbytes) + self.offs.nbytes


def jaccard_sorted(x: np.ndarray, y: np.ndarray) -> float:
    """Exact Jaccard of two sorted unique uint32 arrays."""
    if x.size == 0 or y.size == 0:
        return 0.0
    inter = np.intersect1d(x, y, assume_unique=True).size
    return inter / (x.size + y.size - inter)


# ===========================================================================
# 6. Loading
# ===========================================================================

def load_jsonl(path, code_f, label_f, split_f, id_f, forced_split=None, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh):
            if limit is not None and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            code = o.get(code_f)
            if not isinstance(code, str) or not code.strip():
                continue
            split = forced_split or (str(o.get(split_f, "all")) if split_f else "all")
            rows.append({
                "id": str(o.get(id_f, f"{os.path.basename(path)}:{n}")),
                "code": code,
                "label": int(o.get(label_f, 0) or 0),
                "split": split,
                "project": str(o.get("project", o.get("repo", "")) or ""),
            })
    return rows


# ===========================================================================
# 7. Pipeline
# ===========================================================================

def preprocess(rows, k, K, seed, report_every, verbose=True, rename="positional"):
    """Normalise, collapse exact duplicates, shingle and MinHash the
    representatives. Returns (rep_idx, member_of_rep, store, sigs, exact_groups).
    rename="blind" also collapses functions that differ only in identifier names
    (not only by a consistent renaming) in the exact-duplicate step."""
    tokenize = TOKENIZERS[rename]
    n = len(rows)
    t0 = time.time()
    if verbose:
        print(f"  [1/3] normalising {n} functions ...", flush=True)

    vocab: dict[str, int] = {}
    exact: dict[bytes, list[int]] = defaultdict(list)
    token_ids: list[np.ndarray] = []

    for i, r in enumerate(rows):
        toks = tokenize(r["code"])
        ids = np.empty(len(toks), dtype=np.uint64)
        for j, t in enumerate(toks):
            v = vocab.get(t)
            if v is None:
                v = len(vocab) + 1
                vocab[t] = v
            ids[j] = v
        token_ids.append(ids)
        exact[ids.tobytes()].append(i)
        if verbose and report_every and (i + 1) % report_every == 0:
            print(f"        {i+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)

    # One representative per exact-normalised-token group (Type-1/2 collapse).
    reps = [g[0] for g in exact.values()]
    member_of_rep = {g[0]: g for g in exact.values()}
    if verbose:
        print(f"  [2/3] {len(reps)} representatives after Type-1/2 collapse "
              f"({n - len(reps)} exact duplicates removed); shingling ...",
              flush=True)

    a, b = make_perms(K, seed)
    store = ShingleStore()
    sigs = np.empty((len(reps), K), dtype=np.uint64)
    for pos, i in enumerate(reps):
        sh = shingle_ids(token_ids[i], k)
        store.add(sh)
        sigs[pos] = minhash_block(sh, a, b)
        if verbose and report_every and (pos + 1) % report_every == 0:
            print(f"        {pos+1}/{len(reps)}  ({time.time()-t0:.0f}s)", flush=True)
    store.finalize()
    token_ids.clear()

    if verbose:
        print(f"  [3/3] shingle store {store.nbytes/1e6:.0f} MB, "
              f"signatures {sigs.nbytes/1e6:.0f} MB "
              f"({time.time()-t0:.0f}s total)", flush=True)
    return reps, member_of_rep, store, sigs, exact


def build_families(reps, store, sigs, theta, bands, exhaustive_bucket,
                   seed, verbose=True):
    """LSH -> exact-Jaccard verification -> union-find over representatives."""
    m, K = sigs.shape
    rows_per_band = K // bands
    uf = UnionFind(m)
    rng = random.Random(seed)

    verified = attempted = 0
    n_exhaustive = n_pivot = 0
    t0 = time.time()

    for bi in range(bands):
        seg = sigs[:, bi * rows_per_band:(bi + 1) * rows_per_band]
        buckets: dict[bytes, list[int]] = defaultdict(list)
        for i in range(m):
            buckets[seg[i].tobytes()].append(i)

        for members in buckets.values():
            sz = len(members)
            if sz < 2:
                continue
            if sz <= exhaustive_bucket:
                n_exhaustive += 1
                for x in range(sz):
                    ix = members[x]
                    sx = store.get(ix)
                    for y in range(x + 1, sz):
                        iy = members[y]
                        if uf.find(ix) == uf.find(iy):
                            continue          # already same family: skip work
                        attempted += 1
                        if jaccard_sorted(sx, store.get(iy)) >= theta:
                            uf.union(ix, iy)
                            verified += 1
            else:
                # Linear pivot pass. Documented approximation: connectivity for
                # very large buckets relies on the other bands' bucketings.
                n_pivot += 1
                pivot = members[rng.randrange(sz)]
                sp = store.get(pivot)
                for iy in members:
                    if iy == pivot or uf.find(pivot) == uf.find(iy):
                        continue
                    attempted += 1
                    if jaccard_sorted(sp, store.get(iy)) >= theta:
                        uf.union(pivot, iy)
                        verified += 1
        if verbose:
            print(f"      band {bi+1}/{bands}: {verified} verified pairs "
                  f"({time.time()-t0:.0f}s)", flush=True)

    return uf, {
        "verified_clone_pairs": verified,
        "jaccard_computations": attempted,
        "buckets_exhaustive": n_exhaustive,
        "buckets_pivot_approximated": n_pivot,
        "lsh_bands": bands,
        "lsh_rows_per_band": rows_per_band,
        "lsh_scurve_midpoint": round((1.0 / bands) ** (1.0 / rows_per_band), 3),
    }


def expand_families(rows, reps, member_of_rep, uf):
    """Map every function (not just representatives) to a family id."""
    fam = [0] * len(rows)
    for pos, rep in enumerate(reps):
        root = uf.find(pos)
        for member in member_of_rep[rep]:
            fam[member] = root
    return fam


# ===========================================================================
# 8. Analysis
# ===========================================================================

def analyse(rows, fam, exact, theta):
    n = len(rows)
    members = defaultdict(list)
    for i, f in enumerate(fam):
        members[f].append(i)

    multi = {f: v for f, v in members.items() if len(v) > 1}
    sizes = Counter(len(v) for v in members.values())

    spanning = tt_span = contradictory = 0
    leaked_test: set[int] = set()
    cross_project_families = 0
    for idxs in multi.values():
        splits = {rows[i]["split"] for i in idxs}
        if len(splits) > 1:
            spanning += 1
        if "train" in splits and "test" in splits:
            tt_span += 1
            leaked_test.update(i for i in idxs if rows[i]["split"] == "test")
        if len({rows[i]["label"] for i in idxs}) > 1:
            contradictory += 1
        projs = {rows[i]["project"] for i in idxs if rows[i]["project"]}
        if len(projs) > 1:
            cross_project_families += 1

    n_test = sum(1 for r in rows if r["split"] == "test")
    exact_multi = sum(1 for v in exact.values() if len(v) > 1)
    exact_red = sum(len(v) - 1 for v in exact.values() if len(v) > 1)

    return {
        "theta": theta,
        "n_functions": n,
        "n_test_functions": n_test,
        "type12_exact_groups": exact_multi,
        "type12_redundant_functions": exact_red,
        "type12_redundancy_pct": round(100.0 * exact_red / n, 2) if n else None,
        "n_families_total": len(members),
        "n_families_size_gt1": len(multi),
        "largest_family": max((len(v) for v in members.values()), default=0),
        "family_size_head": dict(sorted(sizes.items())[:10]),
        "families_spanning_multiple_splits": spanning,
        "families_spanning_train_and_test": tt_span,
        "leaked_test_functions": len(leaked_test),
        "leaked_test_pct": round(100.0 * len(leaked_test) / n_test, 2) if n_test else None,
        "families_with_contradictory_labels": contradictory,
        "cross_project_families": cross_project_families,
    }


def family_disjoint_split(rows, fam, seed, ratios=(0.8, 0.1, 0.1)):
    """Assign whole clone families to exactly one partition (grouped split),
    greedily balancing the positive rate across partitions."""
    members = defaultdict(list)
    for i, f in enumerate(fam):
        members[f].append(i)
    rng = random.Random(seed)
    fams = list(members.items())
    rng.shuffle(fams)
    fams.sort(key=lambda kv: -len(kv[1]))       # big families placed first

    n = len(rows)
    n_pos = sum(r["label"] for r in rows)
    size_target = [ratios[t] * n for t in range(3)]
    pos_target = [ratios[t] * n_pos for t in range(3)]
    buckets: list[list[int]] = [[], [], []]
    have_size = [0, 0, 0]
    have_pos = [0, 0, 0]

    # Two-key greedy: place each family where it least overshoots BOTH the size
    # quota and the positive-label quota. Balancing on size alone leaves a
    # label-shift between partitions (families carry correlated labels), which
    # would confound every downstream comparison.
    for _f, idxs in fams:
        sz = len(idxs)
        pz = sum(rows[i]["label"] for i in idxs)
        best_t, best_cost = 0, None
        for t in range(3):
            if size_target[t] <= 0:
                continue
            size_deficit = (size_target[t] - have_size[t]) / size_target[t]
            pos_deficit = ((pos_target[t] - have_pos[t]) / pos_target[t]
                           if pos_target[t] > 0 else 0.0)
            # prefer the partition that is furthest behind on both axes
            cost = -(size_deficit + 2.0 * pos_deficit)
            if best_cost is None or cost < best_cost:
                best_t, best_cost = t, cost
        buckets[best_t].extend(idxs)
        have_size[best_t] += sz
        have_pos[best_t] += pz

    assign = {}
    for name, idxs in zip(("train", "valid", "test"), buckets):
        for i in idxs:
            assign[rows[i]["id"]] = name
    return assign, {
        "train": len(buckets[0]), "valid": len(buckets[1]), "test": len(buckets[2]),
        "train_pos_rate": _pos_rate(rows, buckets[0]),
        "valid_pos_rate": _pos_rate(rows, buckets[1]),
        "test_pos_rate": _pos_rate(rows, buckets[2]),
    }


def _pos_rate(rows, idxs):
    return round(sum(rows[i]["label"] for i in idxs) / len(idxs), 4) if idxs else None


def random_control_split(rows, sizes, seed):
    """Label-stratified random split at the SAME partition sizes as the
    family-disjoint split, ignoring clone families.

    This is the control arm. Without it, a published-vs-family-disjoint delta
    conflates two effects: leakage removed, and a simply different test set.
    With it the delta decomposes -- published->random isolates re-partitioning,
    random->family-disjoint isolates family-awareness. Per-label rounding means
    partition sizes match `sizes` to within a few functions, not exactly; the
    realised sizes are returned rather than the requested ones.
    """
    rng = random.Random(seed + 1)
    by_label = defaultdict(list)
    for i, r in enumerate(rows):
        by_label[r["label"]].append(i)

    total = sum(sizes)
    buckets: list[list[int]] = [[], [], []]
    for idxs in by_label.values():
        rng.shuffle(idxs)
        cut = 0
        for t in range(3):
            k = len(idxs) - cut if t == 2 else round(len(idxs) * sizes[t] / total)
            buckets[t].extend(idxs[cut:cut + k])
            cut += k

    assign = {}
    for name, idxs in zip(("train", "valid", "test"), buckets):
        for i in idxs:
            assign[rows[i]["id"]] = name
    return assign, {
        "train": len(buckets[0]), "valid": len(buckets[1]), "test": len(buckets[2]),
        "train_pos_rate": _pos_rate(rows, buckets[0]),
        "valid_pos_rate": _pos_rate(rows, buckets[1]),
        "test_pos_rate": _pos_rate(rows, buckets[2]),
    }


def leakage_under(rows, fam, assign):
    """Leaked test functions under an arbitrary split assignment, so the
    control arm can be shown to retain the leakage the family split removes."""
    fam_splits = defaultdict(set)
    for i, f in enumerate(fam):
        fam_splits[f].add(assign[rows[i]["id"]])
    leaked = sum(1 for i, f in enumerate(fam)
                 if assign[rows[i]["id"]] == "test" and "train" in fam_splits[f])
    n_test = sum(1 for r in rows if assign[r["id"]] == "test")
    return {"leaked_test_functions": leaked,
            "leaked_test_pct": round(100 * leaked / n_test, 2) if n_test else None}


# ===========================================================================
# 9. 1-NN clone-retrieval baseline
# ===========================================================================

def knn_baseline(rows, reps, member_of_rep, store, sigs, bands, probe, seed):
    """For each TEST function, find its most similar TRAIN function among LSH
    co-bucket candidates and copy that label. No learning of any kind."""
    m, K = sigs.shape
    rows_per_band = K // bands
    rep_split = []
    for pos, rep in enumerate(reps):
        splits = {rows[i]["split"] for i in member_of_rep[rep]}
        rep_split.append(splits)

    # candidate lists per representative, restricted to cross-split pairs
    cand: dict[int, set[int]] = defaultdict(set)
    for bi in range(bands):
        seg = sigs[:, bi * rows_per_band:(bi + 1) * rows_per_band]
        buckets = defaultdict(list)
        for i in range(m):
            buckets[seg[i].tobytes()].append(i)
        for mem in buckets.values():
            if len(mem) < 2:
                continue
            tests = [i for i in mem if "test" in rep_split[i]]
            trains = [i for i in mem if "train" in rep_split[i]]
            if not tests or not trains:
                continue
            for t in tests:
                s = cand[t]
                for tr in trains:
                    if tr != t and len(s) < probe:
                        s.add(tr)

    best: dict[int, tuple[float, int]] = {}
    # A representative whose exact-duplicate group contains BOTH train and test
    # members is the strongest possible leak: the test function's nearest
    # training neighbour is a byte-identical copy of itself (Jaccard = 1.0).
    # v2 fix — an earlier version skipped these and reported 0% coverage.
    for pos in range(m):
        if "test" in rep_split[pos] and "train" in rep_split[pos]:
            best[pos] = (1.0, pos)

    for t, cands in cand.items():
        if t in best:
            continue
        st = store.get(t)
        bs, bj = 0.0, -1
        for c in cands:
            s = jaccard_sorted(st, store.get(c))
            if s > bs:
                bs, bj = s, c
        if bj >= 0:
            best[t] = (bs, bj)

    # score every test function
    rep_of = {}
    for pos, rep in enumerate(reps):
        for mem in member_of_rep[rep]:
            rep_of[mem] = pos

    y_true, y_score, covered = [], [], 0
    for i, r in enumerate(rows):
        if r["split"] != "test":
            continue
        y_true.append(r["label"])
        hit = best.get(rep_of[i])
        if hit is None:
            y_score.append(0.0)
        else:
            sim, j = hit
            covered += 1
            nb_label = max(rows[x]["label"] for x in member_of_rep[reps[j]])
            y_score.append(sim if nb_label == 1 else (1.0 - sim) * 0.5)

    if not y_true:
        return {"error": "no test split found"}
    return {
        "test_functions": len(y_true),
        "test_with_a_clone_in_train": covered,
        "coverage_pct": round(100.0 * covered / len(y_true), 2),
        "pr_auc": _pr_auc(y_true, y_score),
        "positive_rate": round(sum(y_true) / len(y_true), 4),
        "note": "approximate 1-NN over LSH co-bucket candidates; "
                "similarities are exact Jaccard",
    }


def _pr_auc(y_true, y_score) -> float:
    """Average precision, no sklearn dependency."""
    P = sum(y_true)
    if P == 0:
        return 0.0
    order = sorted(range(len(y_score)), key=lambda i: -y_score[i])
    tp = 0
    ap = 0.0
    for rank, i in enumerate(order, 1):
        if y_true[i] == 1:
            tp += 1
            ap += tp / rank
    return round(ap / P, 4)


# ===========================================================================
# 10. Main
# ===========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data")
    ap.add_argument("--train"); ap.add_argument("--valid"); ap.add_argument("--test")
    ap.add_argument("--code-field", default="func")
    ap.add_argument("--label-field", default="target")
    ap.add_argument("--split-field", default="split")
    ap.add_argument("--id-field", default="idx")
    ap.add_argument("--theta", type=float, nargs="+", default=[0.7],
                    help="Jaccard thresholds defining a Type-3 clone. Default 0.7 "
                         "is characterised on BigCloneBench ground-truth clone "
                         "types (see analysis/scripts/calibrate_theta_ours.py). Pass '0.7 0.8 0.9' "
                         "to report the conservative bounds alongside it.")
    ap.add_argument("--shingle-k", type=int, default=5)
    ap.add_argument("--rename", choices=sorted(TOKENIZERS), default="positional",
                    help="identifier handling: 'positional' (default, the audit: V0, V1, ... "
                         "in order of first use) or 'blind' (every identifier -> ID; "
                         "exploratory high-recall audit, use with --shingle-k 3)")
    ap.add_argument("--perms", type=int, default=128)
    ap.add_argument("--bands", type=int, default=16)
    ap.add_argument("--exhaustive-bucket", type=int, default=200,
                    help="buckets up to this size are verified pairwise; "
                         "larger ones use a linear pivot pass")
    ap.add_argument("--knn-baseline", action="store_true")
    ap.add_argument("--knn-probe", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None, help="trial run on first N")
    ap.add_argument("--report-every", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="clone_audit")
    args = ap.parse_args()

    if args.perms % args.bands:
        print("--perms must be divisible by --bands", file=sys.stderr)
        return 2

    _selftest_minhash()
    print("minhash self-test passed (no uint64 overflow)")

    rows = []
    if args.data:
        rows += load_jsonl(args.data, args.code_field, args.label_field,
                           args.split_field, args.id_field, limit=args.limit)
    for path, name in ((args.train, "train"), (args.valid, "valid"), (args.test, "test")):
        if path:
            rows += load_jsonl(path, args.code_field, args.label_field, "",
                               args.id_field, forced_split=name, limit=args.limit)
    if not rows:
        print("No rows loaded — check --code-field and paths.", file=sys.stderr)
        return 2

    counts = Counter(r["split"] for r in rows)
    print(f"Loaded {len(rows)} functions: {dict(counts)}")
    if "test" not in counts:
        print("  ! no rows with split=='test'; cross-split numbers will be empty",
              file=sys.stderr)

    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    print("\nPreprocessing (shared across all theta) ...")
    reps, member_of_rep, store, sigs, exact = preprocess(
        rows, args.shingle_k, args.perms, args.seed, args.report_every, rename=args.rename)

    report = {
        "input_counts": dict(counts),
        # "rename" is recorded only when it is not the default, so the default report is
        # byte-identical to the one written before the option existed.
        "config": {k: v for k, v in vars(args).items()
                   if k != "out" and not (k == "rename" and v == "positional")},
        "n_representatives": len(reps),
        "thresholds": [],
    }

    for theta in args.theta:
        print(f"\n=== theta = {theta} ===")
        t0 = time.time()
        uf, lsh_stats = build_families(
            reps, store, sigs, theta, args.bands, args.exhaustive_bucket, args.seed)
        fam = expand_families(rows, reps, member_of_rep, uf)
        stats = analyse(rows, fam, exact, theta)
        stats.update(lsh_stats)
        stats["runtime_seconds"] = round(time.time() - t0, 1)

        assign, split_stats = family_disjoint_split(rows, fam, args.seed)
        stats["family_disjoint_split"] = split_stats
        split_stats.update(leakage_under(rows, fam, assign))

        ctrl_assign, ctrl_stats = random_control_split(
            rows, (split_stats["train"], split_stats["valid"], split_stats["test"]),
            args.seed)
        ctrl_stats.update(leakage_under(rows, fam, ctrl_assign))
        stats["random_control_split"] = ctrl_stats

        if args.knn_baseline:
            stats["knn_retrieval_baseline"] = knn_baseline(
                rows, reps, member_of_rep, store, sigs, args.bands,
                args.knn_probe, args.seed)

        report["thresholds"].append(stats)

        with open(f"{args.out}_families_theta{theta}.json", "w", encoding="utf-8") as fh:
            json.dump({rows[i]["id"]: int(f) for i, f in enumerate(fam)}, fh)
        with open(f"{args.out}_splitfile_theta{theta}.json", "w", encoding="utf-8") as fh:
            json.dump(assign, fh)
        with open(f"{args.out}_splitfile_randomctrl_theta{theta}.json", "w",
                  encoding="utf-8") as fh:
            json.dump(ctrl_assign, fh)

        for kk, vv in stats.items():
            print(f"  {kk}: {vv}")

    with open(f"{args.out}_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    with open(f"{args.out}_report.md", "w", encoding="utf-8") as fh:
        fh.write("# Clone-family contamination audit\n\n")
        fh.write(f"Input: {dict(counts)}  |  representatives: {len(reps)}\n\n")
        fh.write("| theta | families (>1) | span train+test | leaked test fns | "
                 "leaked test % | contradictory labels | cross-project fams |\n")
        fh.write("|---|---|---|---|---|---|---|\n")
        for s in report["thresholds"]:
            fh.write(f"| {s['theta']} | {s['n_families_size_gt1']} | "
                     f"{s['families_spanning_train_and_test']} | "
                     f"{s['leaked_test_functions']} | {s['leaked_test_pct']} | "
                     f"{s['families_with_contradictory_labels']} | "
                     f"{s['cross_project_families']} |\n")
        if args.knn_baseline:
            fh.write("\n## 1-NN clone-retrieval baseline (no learning)\n\n")
            fh.write("| theta | coverage % | PR-AUC | test positive rate |\n|---|---|---|---|\n")
            for s in report["thresholds"]:
                kb = s.get("knn_retrieval_baseline", {})
                fh.write(f"| {s['theta']} | {kb.get('coverage_pct')} | "
                         f"{kb.get('pr_auc')} | {kb.get('positive_rate')} |\n")

        fh.write("\n## Split arms -- RQ2 must train on all three\n\n")
        fh.write("The random control holds partition sizes and label balance fixed while\n"
                 "ignoring families, so it retains the leakage. Comparing published vs\n"
                 "family-disjoint alone would confound removed leakage with a different\n"
                 "test set; the control separates them.\n\n")
        fh.write("| theta | arm | train | valid | test | test pos rate | leaked test % |\n")
        fh.write("|---|---|---|---|---|---|---|\n")
        for s in report["thresholds"]:
            fd, rc = s["family_disjoint_split"], s["random_control_split"]
            fh.write(f"| {s['theta']} | published | {counts.get('train')} | "
                     f"{counts.get('valid')} | {counts.get('test')} | — | "
                     f"{s['leaked_test_pct']} |\n")
            for arm, a in (("random control", rc), ("family-disjoint", fd)):
                fh.write(f"| {s['theta']} | {arm} | {a['train']} | {a['valid']} | "
                         f"{a['test']} | {a['test_pos_rate']} | {a['leaked_test_pct']} |\n")
    print(f"\nWrote {args.out}_report.md / _report.json / _families_*.json / _splitfile_*.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
