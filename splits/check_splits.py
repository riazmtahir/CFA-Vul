#!/usr/bin/env python3
"""check_splits.py -- check the split files and family maps in this folder against the MD5 checksums recorded
at run time (splits/README.md). Usage (from the repository root): python splits/check_splits.py
Exit code 0 only if every expected file is present, matches its checksum and is below GitHub's 100 MB limit."""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED = {
    "diversevul_audit_splitfile_theta0.7.json": "8b68b18284f4dce553a582112ff6d643",
    "diversevul_audit_splitfile_randomctrl_theta0.7.json": "7a936fd6139a33605f08afa506af321d",
    "diversevul_audit_splitfile_projectdisjoint_seed42.json": "aae66bcb79411a8fb67b810116a2f982",
    "diversevul_audit_splitfile_commitdisjoint_seed42.json": "824ef7843bfaa2786210ea7725b15ba7",
    "diversevul_audit_splitfile_commitdecon_seed42.json": "55662742499879fd17bd18ce48062c34",
    "diversevul_audit_splitfile_commitsizectrl_seed42.json": "85c7cc23e1c32713c63cf92518ef1208",
    "diversevul_audit_families_theta0.7.json": "ba84d1def26d89f1b61cdd3b203fb8b7",
    "devign_audit_splitfile_theta0.7.json": "db0e586918d8977c4c58dd59beffcba1",
    "devign_audit_families_theta0.7.json": "c7500a1c53786550f049ea49639736d1",
}
ok = True
for name, want in EXPECTED.items():
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        print(f"MISSING  {name}")
        ok = False
        continue
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    mb = os.path.getsize(p) / 2**20
    good = h.hexdigest() == want and mb < 100
    ok &= good
    print(f"{'OK      ' if good else 'MISMATCH'} {name}  {mb:.1f} MB  md5 {h.hexdigest()}"
          + ("" if h.hexdigest() == want else f" (expected {want})") + ("" if mb < 100 else "  > 100 MB: too large for GitHub"))
print("ALL SPLIT FILES OK" if ok else "NOT READY: fix the lines above before committing")
sys.exit(0 if ok else 1)
