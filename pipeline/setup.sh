#!/bin/bash
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"   # the original runs used the mirror https://hf-mirror.com
export HF_HOME="${HF_HOME:-$CFA_DATA_DIR/hf}"
PY="${CFA_PYTHON:-python}"
echo "=== packages ==="
$PY -m pip install -q transformers datasets scikit-learn tqdm 2>&1 | tail -3
echo "=== devign ==="
mkdir -p $CFA_DATA_DIR/data/devign
$PY - <<PYEOF
from datasets import load_dataset
import json
d = load_dataset("google/code_x_glue_cc_defect_detection")
out = open("$CFA_DATA_DIR/data/devign/devign_all.jsonl","w",encoding="utf-8")
n=0
for s,name in [("train","train"),("validation","valid"),("test","test")]:
    for r in d[s]:
        r["split"]=name; out.write(json.dumps(r)+"\n"); n+=1
out.close()
print("devign rows:", n, {s:d[s].num_rows for s in d})
PYEOF
echo "=== encoders ==="
$PY - <<PYEOF
from transformers import AutoTokenizer, AutoModel
for n in ["microsoft/codebert-base","microsoft/graphcodebert-base","microsoft/unixcoder-base"]:
    AutoTokenizer.from_pretrained(n); AutoModel.from_pretrained(n); print(n,"OK")
PYEOF
echo "=== SETUP DONE ==="
