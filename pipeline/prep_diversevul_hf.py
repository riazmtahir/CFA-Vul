"""DiverseVul -> audit JSONL, from the bstee615/diversevul mirror.

Verified against Chen et al. RAID 2023 before use: 330,492 functions,
18,945 vulnerable, 150 CWEs -- exact match on all three, and identical to the
official Google Drive release checked independently on 2026-09-17.

The mirror ships an 80/10/10 split with reproducible .indices files, so the
published-protocol arm is a real community-standard partition rather than a
synthetic one.
"""
import json
from datasets import load_dataset

d = load_dataset("bstee615/diversevul")
DST = __import__("os").path.join(__import__("os").environ.get("CFA_DATA_DIR", "data"), "data", "diversevul", "diversevul.jsonl")
import os; os.makedirs(os.path.dirname(DST), exist_ok=True)

n = 0
counts = {}
pos = {}
with open(DST, "w", encoding="utf-8") as out:
    for split, name in [("train","train"),("validation","valid"),("test","test")]:
        counts[name] = 0; pos[name] = 0
        for r in d[split]:
            lab = 1 if r["target"] == 1 else 0
            out.write(json.dumps({
                "idx": n, "func": r["func"], "target": lab,
                "project": r.get("project") or "",
                "commit_id": r.get("commit_id") or "",
                "cwe": r.get("cwe") or [], "split": name,
            }) + "\n")
            n += 1; counts[name] += 1; pos[name] += lab
print("rows:", n)
print("splits:", counts)
print("pos rates:", {k: round(pos[k]/counts[k], 4) for k in counts})
