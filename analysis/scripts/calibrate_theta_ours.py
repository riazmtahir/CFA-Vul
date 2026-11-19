#!/usr/bin/env python3
"""calibrate_theta_ours.py -- RQ0 re-run with OUR family pipeline instead of TouSimCC.

results/rq0_calibration/ (calibrate_theta_bcb.py) reads TouSimCC's precomputed Jaccard
column t1_sim. That is not the measure the audit uses. This script scores the same
BigCloneBench labelled pairs with the audit's own functions, imported unchanged from
clone_family_audit.py:

    normalize_tokens(code) -> token ids (one global vocabulary, as in preprocess())
    -> shingle_ids(ids, k=5) -> jaccard_sorted(a, b)

That exact Jaccard is the criterion an LSH candidate pair must pass to join a family
(build_families), and identical normalised token streams collapse to one representative
(Jaccard 1 here as well). LSH candidate recall is NOT modelled: the numbers are the
capture of the verification criterion, i.e. an upper bound on what the audit finds.

Reads (read-only; the zip is read in memory, nothing is extracted)
  <bcb>/id2sourcecode.zip       id2sourcecode/<id>.java, 73,319 functions
  <bcb>/type-1.csv .. type-5.csv  id1,id2,type,sim_line,sim_token (no header)
  <bcb>/nonclone.csv            FUNCTION_ID_ONE,FUNCTION_ID_TWO
  <sim>/type-*_sim.csv, nonclone_sim.csv   TouSimCC scores, for the pair-level comparison
  results/rq0_calibration/rq0_theta_calibration.json   TouSimCC calibration (same format)
Default <bcb> = $CFA_BCB_DIR, <sim> = $CFA_TOUSIMCC_SIM_DIR.

Mapping check (refuses to emit numbers if it fails): files type-3/4/5 must hold the
BCB similarity bands on min(sim_line, sim_token) (SourcererCC, ICSE 2016, p8), and Type-1
pairs must have median Jaccard 1.000. Measured 12 Oct 2026: type-4 = MT3 [0.5, 0.7) and
type-5 = WT3/T4 [0, 0.5) exactly, but type-3 spans [0.7, 1.0], i.e. VST3 (>= 0.9) PLUS ST3.
The row keeps the label "Strong Type-3 (ST3)" of results/rq0_calibration/ for comparison,
and two sub-rows split it into VST3 and ST3 proper [0.7, 0.9).

Caveat: normalize_tokens() was written for C/C++. On Java, keywords absent from its
C/C++ list (e.g. import, final, extends, throws, boolean, synchronized) are renamed like
identifiers, and '@' (annotations) is not a token. Renaming is consistent within a
function, so this affects the score only through positional identifier numbering.

Output: results/rq0_calibration_ours/rq0_theta_calibration_ours.{md,json}
Usage (repository root): python analysis/scripts/calibrate_theta_ours.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import zipfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from clone_family_audit import TOKENIZERS, jaccard_sorted, normalize_tokens, shingle_ids  # noqa: E402

BCB = os.environ.get("CFA_BCB_DIR", "data/BCB")
SIM = os.environ.get("CFA_TOUSIMCC_SIM_DIR", "data/TouSimCC_output_16_sim")
CATEGORIES = [  # (pair file, TouSimCC score file, label, BCB band on min(sim_line, sim_token))
    ("type-1.csv", "type-1_sim.csv", "Type-1 exact", None),
    ("type-2.csv", "type-2_sim.csv", "Type-2 renamed", None),
    ("type-3.csv", "type-3_sim.csv", "Strong Type-3 (ST3)", (0.7, 1.0)),  # = VST3 + ST3, see check
    ("type-4.csv", "type-4_sim.csv", "Moderate Type-3 (MT3)", (0.5, 0.7)),
    ("type-5.csv", "type-5_sim.csv", "Weak Type-3 / Type-4", (0.0, 0.5)),
    ("nonclone.csv", "nonclone_sim.csv", "non-clone", None),
]


def read_pairs(path):
    """(id1, id2, band_value or None) per row; header rows and bad rows are counted."""
    pairs, bad = [], 0
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            try:
                a, b = int(row[0]), int(row[1])
            except (ValueError, IndexError):
                bad += 1
                continue
            band = None
            if len(row) >= 5:
                try:
                    band = min(float(row[3]), float(row[4]))
                except ValueError:
                    pass
            pairs.append((a, b, band))
    return pairs, bad


def read_tousimcc(path):
    """{(id1, id2): t1_sim} with the same defensive parsing as calibrate_theta_bcb.py."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rd = csv.reader(fh)
        next(rd, None)
        for row in rd:
            try:
                a, b, v = int(row[0]), int(row[1]), float(row[2])
            except (ValueError, IndexError):
                continue
            if 0.0 <= v <= 1.0:
                out[(a, b)] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bcb", default=BCB)
    ap.add_argument("--sim", default=SIM)
    ap.add_argument("--shingle-k", type=int, default=5)
    ap.add_argument("--rename", choices=sorted(TOKENIZERS), default="positional",
                    help="identifier handling, as clone_family_audit.py --rename. 'blind' is the "
                         "EXPLORATORY high-recall variant (added 18 Oct 2026); write it to another --out")
    ap.add_argument("--theta", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "rq0_calibration_ours"))
    args = ap.parse_args()
    t0 = time.time()

    zpath = os.path.join(args.bcb, "id2sourcecode.zip")
    if not os.path.isfile(zpath):
        print(f"ERROR: BCB source not found: {zpath}", file=sys.stderr)
        return 2

    # 1. Read every pair file first, so only needed functions are tokenised.
    cats = []
    for pf, sf, label, band in CATEGORIES:
        pairs, bad = read_pairs(os.path.join(args.bcb, pf))
        cats.append({"file": pf, "sim_file": sf, "label": label, "band": band, "pairs": pairs, "bad_rows": bad})
    needed = {i for c in cats for a, b, _ in c["pairs"] for i in (a, b)}

    # 2. Mapping check on BCB's own similarity columns.
    band_check = {}
    for c in cats:
        if c["band"] is None:
            continue
        vals = np.array([v for _, _, v in c["pairs"] if v is not None])
        lo, hi = c["band"]
        inside = float(((vals >= lo - 1e-9) & ((vals < hi) | (hi >= 1.0))).mean())
        band_check[c["label"]] = {"band": [lo, hi], "min": round(float(vals.min()), 4),
                                  "max": round(float(vals.max()), 4), "share_inside": round(inside, 4)}
        if inside < 0.99:
            print(f"ERROR: {c['file']} is not the {c['label']} band ({inside:.3f} inside "
                  f"[{lo}, {hi})). Refusing to emit numbers.", file=sys.stderr)
            return 3

    # 3. Tokenise and shingle each needed function once (global vocabulary as in preprocess()).
    vocab: dict[str, int] = {}
    shingles: dict[int, np.ndarray] = {}
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.endswith(".java"):
                continue
            try:
                fid = int(os.path.basename(name)[:-5])
            except ValueError:
                continue
            if fid not in needed:
                continue
            code = z.read(name).decode("utf-8", errors="replace")
            toks = TOKENIZERS[args.rename](code)
            ids = np.empty(len(toks), dtype=np.uint64)
            for j, t in enumerate(toks):
                v = vocab.get(t)
                if v is None:
                    v = len(vocab) + 1
                    vocab[t] = v
                ids[j] = v
            shingles[fid] = shingle_ids(ids, args.shingle_k)
    print(f"tokenised {len(shingles):,} of {len(needed):,} needed functions ({time.time()-t0:.0f}s)")

    # 4. Score pairs.
    ref = {}
    rp = os.path.join(ROOT, "results", "rq0_calibration", "rq0_theta_calibration.json")
    if os.path.isfile(rp):
        ref = {r["category"]: r for r in json.load(open(rp, encoding="utf-8"))["categories"]}
    rows, diags = [], {}
    for c in cats:
        js, missing, empty, bands, tvals = [], 0, 0, [], []
        tsc = read_tousimcc(os.path.join(args.sim, c["sim_file"]))
        paired = []
        for a, b, band in c["pairs"]:
            if a not in shingles or b not in shingles:
                missing += 1
                continue
            sa, sb = shingles[a], shingles[b]
            if sa.size == 0 or sb.size == 0:
                empty += 1
            j = jaccard_sorted(sa, sb)
            js.append(j)
            bands.append(-1.0 if band is None else band)
            t = tsc.get((a, b), tsc.get((b, a)))
            tvals.append(np.nan if t is None else t)
            if t is not None:
                paired.append((j, t))
        s = np.array(js)
        diags[c["label"]] = {"pairs_in_file": len(c["pairs"]) + c["bad_rows"], "bad_rows": c["bad_rows"],
                             "missing_source": missing, "empty_shingles": empty, "usable": int(s.size)}
        r = {"category": c["label"], "n": int(s.size), "mean": round(float(s.mean()), 3),
             "p10": round(float(np.quantile(s, 0.10)), 3), "p25": round(float(np.quantile(s, 0.25)), 3),
             "median": round(float(np.median(s)), 3), "p75": round(float(np.quantile(s, 0.75)), 3),
             "capture": {f"{t:.2f}": round(float((s >= t).mean()), 4) for t in args.theta}}
        if c["label"] in ref:
            r["tousimcc_capture"] = ref[c["label"]]["capture"]
            r["tousimcc_n"] = ref[c["label"]]["n"]
        if paired:
            p = np.array(paired)
            d = p[:, 0] - p[:, 1]
            r["pairwise_vs_tousimcc"] = {
                "n_matched": int(len(p)),
                "median_diff_ours_minus_tousimcc": round(float(np.median(d)), 4),
                "mean_diff": round(float(d.mean()), 4),
                "pearson_r": round(float(np.corrcoef(p[:, 0], p[:, 1])[0, 1]), 4) if len(p) > 2 else None,
                "decision_agreement": {f"{t:.2f}": round(float(((p[:, 0] >= t) == (p[:, 1] >= t)).mean()), 4)
                                       for t in args.theta}}
        rows.append(r)
        if c["file"] == "type-3.csv":  # split VST3 / ST3 proper by BCB's own band
            bv, tv = np.array(bands), np.array(tvals)
            for sub, m in (("  of which VST3 (BCB sim >= 0.9)", bv >= 0.9),
                           ("  of which ST3 proper (BCB sim 0.7-0.9)", (bv >= 0.7) & (bv < 0.9))):
                ss, tt = s[m], tv[m]
                tt = tt[~np.isnan(tt)]
                rows.append({"category": sub, "n": int(ss.size), "mean": round(float(ss.mean()), 3),
                             "p10": round(float(np.quantile(ss, 0.10)), 3),
                             "p25": round(float(np.quantile(ss, 0.25)), 3),
                             "median": round(float(np.median(ss)), 3),
                             "p75": round(float(np.quantile(ss, 0.75)), 3),
                             "capture": {f"{t:.2f}": round(float((ss >= t).mean()), 4) for t in args.theta},
                             "tousimcc_capture_matched_pairs": {f"{t:.2f}": round(float((tt >= t).mean()), 4)
                                                                for t in args.theta},
                             "tousimcc_n_matched": int(tt.size)})

    t1 = next(r for r in rows if r["category"] == "Type-1 exact")
    if abs(t1["median"] - 1.0) > 1e-6:
        print(f"ERROR: Type-1 median Jaccard {t1['median']} != 1.000; pipeline or files changed.", file=sys.stderr)
        return 3

    # 5. Write.
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, "rq0_theta_calibration_ours")
    payload = {"bcb_root": args.bcb, "tousimcc_root": args.sim,
               "pipeline": "clone_family_audit.normalize_tokens -> shingle_ids(k) -> jaccard_sorted",
               **({} if args.rename == "positional" else
                  {"rename": args.rename, "label": "EXPLORATORY high-recall variant (blind identifiers)"}),
               "shingle_k": args.shingle_k, "vocabulary_size": len(vocab), "thresholds": args.theta,
               "mapping_check_type1_median": t1["median"], "band_check": band_check,
               "categories": rows, "parse_diagnostics": diags}
    json.dump(payload, open(stem + ".json", "w", encoding="utf-8"), indent=2)

    th = args.theta
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write("# RQ0 — Clone-threshold calibration on BigCloneBench, our pipeline\n\n")
        fh.write(f"Source: `{args.bcb}` (function source `id2sourcecode.zip`, pair files type-1..5, nonclone). "
                 f"Score = exact Jaccard of {args.shingle_k}-token shingles after "
                 + ("`normalize_tokens` " if args.rename == "positional" else
                    "`blind_tokens` (every identifier -> ID; **EXPLORATORY high-recall variant**) ") +
                 f"(`clone_family_audit.py`); Type-1 median = {t1['median']:.3f}.\n\n")
        fh.write("| BCB category | n | mean | median | " + " | ".join(f"@θ={t:.1f}" for t in th) + " |\n")
        fh.write("|---|---|---|---|" + "---|" * len(th) + "\n")
        for r in rows:
            caps = " | ".join(f"{r['capture'][f'{t:.2f}']*100:.1f}%" for t in th)
            fh.write(f"| {r['category']} | {r['n']:,} | {r['mean']:.3f} | {r['median']:.3f} | {caps} |\n")
        fh.write("\n## Difference from the TouSimCC calibration (`results/rq0_calibration/`)\n\n")
        fh.write("Capture in percentage points, ours − TouSimCC.\n\n")
        fh.write("| BCB category | " + " | ".join(f"Δ@θ={t:.1f}" for t in th) +
                 " | matched pairs | median Δ Jaccard | r | same decision @0.7 |\n")
        fh.write("|---|" + "---|" * len(th) + "---|---|---|---|\n")
        for r in rows:
            tc = r.get("tousimcc_capture") or r.get("tousimcc_capture_matched_pairs")
            ds = " | ".join(f"{(r['capture'][f'{t:.2f}'] - tc[f'{t:.2f}'])*100:+.1f}" if tc and f"{t:.2f}" in tc
                            else "n/a" for t in th)
            pw = r.get("pairwise_vs_tousimcc")
            pws = (f"{pw['n_matched']:,} | {pw['median_diff_ours_minus_tousimcc']:+.3f} | {pw['pearson_r']} | "
                   f"{pw['decision_agreement'].get('0.70', float('nan'))*100:.1f}%") if pw else "n/a | n/a | n/a | n/a"
            fh.write(f"| {r['category']} | {ds} | {pws} |\n")
        fh.write("\nSub-rows: TouSimCC capture recomputed on the matched pairs of that band "
                 "(`results/rq0_calibration/` has no band split).\n")
        fh.write("\nCapture here is the family-verification criterion (exact Jaccard ≥ θ). LSH candidate "
                 "recall is not modelled, so these are upper bounds on what the audit links.\n")
        fh.write("\n## Mapping check (BCB similarity bands, min(sim_line, sim_token))\n\n")
        fh.write("| category | band | min | max | share inside |\n|---|---|---|---|---|\n")
        for k, v in band_check.items():
            fh.write(f"| {k} | [{v['band'][0]}, {v['band'][1]}{']' if v['band'][1] >= 1.0 else ')'} | {v['min']} | {v['max']} | {v['share_inside']*100:.2f}% |\n")
        fh.write("\n## Parsing diagnostics\n\n| file | pairs in file | bad rows | source missing | empty shingles | usable |\n"
                 "|---|---|---|---|---|---|\n")
        for k, d in diags.items():
            fh.write(f"| {k} | {d['pairs_in_file']:,} | {d['bad_rows']} | {d['missing_source']} | "
                     f"{d['empty_shingles']} | {d['usable']:,} |\n")
    print(f"wrote {stem}.md / .json ({time.time()-t0:.0f}s)")
    for r in rows:
        print(f"  {r['category']:<24} n={r['n']:>7,}  " +
              "  ".join(f"{t:.1f}:{r['capture'][f'{t:.2f}']*100:5.1f}%" for t in th))
    return 0


if __name__ == "__main__":
    sys.exit(main())
