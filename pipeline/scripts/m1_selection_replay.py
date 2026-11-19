#!/usr/bin/env python3
"""m1_selection_replay.py -- referee item 13 (M1): the post-warm-up selection rule, replayed on the logs.

No model, no GPU. The selection functions are taken from finetune_vd.py itself (parsed with ast and
executed alone, so torch/transformers are not needed) and
  1. checked to reduce EXACTLY to the original rule when the option is off (first_eligible = 1):
     20,000 random validation curves, compared with a verbatim copy of the original loop logic;
  2. replayed on every final-protocol run (max 15 epochs, patience 2) under the post-warm-up rule
     (first eligible epoch = 2, because the warm-up is 10% of 15 epochs = 1.5 epochs), listing every
     run whose selected epoch would change or whose outcome cannot be determined from the log.
Writes results/m1_selection_replay.{json,md}. Exit 1 if check 1 fails.
Usage (repository root): python scripts/m1_selection_replay.py
"""
from __future__ import annotations

import ast
import glob
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = open(os.path.join(ROOT, "finetune_vd.py"), encoding="utf-8").read()
WANT = {"first_eligible_epoch", "is_new_best", "should_stop", "replay_selection"}
ns: dict = {}
mod = ast.parse(SRC)
exec(compile(ast.Module(body=[n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name in WANT],
                        type_ignores=[]), "finetune_vd.py", "exec"), ns)
assert WANT <= ns.keys(), WANT - ns.keys()
first_eligible_epoch, replay = ns["first_eligible_epoch"], ns["replay_selection"]


def original_rule(curve, patience, max_epochs):
    """Verbatim logic of the primary protocol (finetune_vd.py before the option existed)."""
    best_va, best_ep = -1.0, 0
    for ep in range(1, max_epochs + 1):
        if ep > len(curve):
            return best_ep, None, False
        if curve[ep - 1] > best_va:
            best_va, best_ep = curve[ep - 1], ep
        if patience and ep - best_ep >= patience:
            return best_ep, ep, True
    return best_ep, max_epochs, True


# 1. default mode == original rule
rng = random.Random(0)
mismatch = 0
for _ in range(20000):
    n = rng.randint(1, 15)
    curve = [round(rng.random(), 2) for _ in range(n)]
    pat = rng.choice([0, 1, 2, 3])
    if replay(curve, pat, 15, 1) != original_rule(curve, pat, 15):
        mismatch += 1
spe = 16549
fe = first_eligible_epoch(int(0.1 * spe * 15), spe)
checks = {"default_equals_original_mismatches": mismatch, "first_eligible_epoch_15_epochs": fe}
ok = mismatch == 0 and fe == 2

# 2. replay on every final-protocol run
rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "results", "**", "*_seed*.json"), recursive=True)):
    if f.endswith(".preds.json"):
        continue
    d = json.load(open(f, encoding="utf-8"))
    if d.get("max_epochs") != 15 or d.get("patience") != 2 or "history" not in d:
        continue                                    # pilot / 3-epoch / sweep runs are not final protocol
    curve = [h["valid"]["pr_auc"] for h in d["history"]]
    prim = replay(curve, 2, 15, 1)
    post = replay(curve, 2, 15, fe)
    rel = os.path.relpath(f, os.path.join(ROOT, "results")).replace(os.sep, "/")
    assert prim[0] == d["best_epoch"], (rel, prim, d["best_epoch"])   # replay reproduces the logged choice
    if not post[2]:
        status = "differs: would train past the logged epochs (not determinable from the log)"
    elif post[0] == d["best_epoch"]:
        status = "same"
    else:
        status = f"differs: epoch {post[0]}"
    rows.append({"run": rel, "logged_best_epoch": d["best_epoch"], "epochs_run": d["epochs_run"],
                 "post_warmup_best_epoch_within_log": post[0], "complete": post[2], "status": status,
                 "valid_pr_auc": curve})
diff = [r for r in rows if r["status"] != "same"]
out = {"checks": checks, "n_runs": len(rows), "n_same": len(rows) - len(diff), "differs": diff, "runs": rows}
json.dump(out, open(os.path.join(ROOT, "results", "m1_selection_replay.json"), "w", encoding="utf-8"), indent=1)
md = ["# M1: post-warm-up selection replayed on the logged validation curves", "",
      f"Final-protocol runs (max 15 epochs, patience 2): {len(rows)}. Same selected epoch: {len(rows) - len(diff)}. "
      f"Different or not determinable: {len(diff)}.", "",
      f"Check: default mode equals the original rule on 20,000 random curves (mismatches: {mismatch}); "
      f"first eligible epoch for 15 epochs: {fe}.", "",
      "| run | logged best / run | post-warm-up best within the log | status | validation PR-AUC by epoch |",
      "|---|---|---|---|---|"]
for r in diff:
    md.append(f"| {r['run']} | {r['logged_best_epoch']} / {r['epochs_run']} | {r['post_warmup_best_epoch_within_log']} | "
              f"{r['status']} | {', '.join(f'{v:.4f}' for v in r['valid_pr_auc'])} |")
open(os.path.join(ROOT, "results", "m1_selection_replay.md"), "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
print("\n".join(md))
sys.exit(0 if ok else 1)
