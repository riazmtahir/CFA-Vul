#!/usr/bin/env python
"""Pick the learning rate with the best VALIDATION PR-AUC from a sweep dir.

Selection is on validation only and the winner is then frozen across every arm
and seed. Tuning per-arm would confound the very comparison RQ2 makes: an arm
that got a luckier LR would look better for a reason that has nothing to do
with clone-family leakage.
"""
import glob, json, os, sys

sweep_dir, out_file = sys.argv[1], sys.argv[2]
best = None
# Results live one level down, in <sweep_dir>/lr<LR>/<tag>.json -- recursive.
for f in sorted(glob.glob(os.path.join(sweep_dir, "**", "*.json"), recursive=True)):
    d = json.load(open(f))
    va = max((h["valid"]["pr_auc"] for h in d["history"] if "valid" in h),
             default=None)
    if va is None:          # no validation split -> unusable for selection
        continue
    print(f"  lr {d['lr']:<8g} valid PR-AUC {va:.4f}  "
          f"(test {d['test']['pr_auc']:.4f}, best_ep {d['best_epoch']})")
    if best is None or va > best[1]:
        best = (d["lr"], va)

if best is None:
    sys.exit("no sweep results with a validation split")
print(f"  --> chose lr={best[0]:g} on validation PR-AUC {best[1]:.4f}")
open(out_file, "w").write(f"{best[0]:g}\n")
