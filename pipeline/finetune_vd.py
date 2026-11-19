#!/usr/bin/env python3
"""
finetune_vd.py — RQ2: fine-tune an encoder for vulnerability detection under a
specified split arm, and report the metrics the manuscript actually needs.

WHY THIS EXISTS
---------------
RQ2 compares a model across three split arms -- published, random control, and
family-disjoint -- on identical data. The comparison is only interpretable if
every arm is trained and scored by the same code with the same seeds, so this
script takes the split as an INPUT (a split file emitted by
clone_family_audit.py) rather than baking one in.

THE FLOOR IS NOT OPTIONAL
-------------------------
Every run also fits a length-only logistic regression on the same split and
reports its PR-AUC. A transformer that does not clear the length floor has not
demonstrated anything about vulnerability, and on PrimeVul the published
headroom over that floor is roughly 1.3 points. Reporting the floor next to the
model is what keeps the comparison honest.

METRICS
-------
  PR-AUC          primary; ROC-AUC is reported but is a negative exhibit only
  best-F1         threshold-swept, never fixed at 0.5 (0.5 is an artifact under
                  heavy imbalance)
  MCC             at the best-F1 threshold
  VD-S            FNR at FPR <= 0.5%, for VulnScout comparability
  worst-family    lowest per-family PR-AUC over families with >=10 test members;
                  the quantity RQ4's Group DRO is supposed to improve

USAGE
-----
  python finetune_vd.py --data devign.jsonl --model microsoft/codebert-base \\
      --splitfile devign_audit_splitfile_theta0.7.json --arm family_disjoint \\
      --seed 1 --out results/

  # published arm = use the split field already in the jsonl
  python finetune_vd.py --data devign.jsonl --model microsoft/codebert-base \\
      --arm published --seed 1 --out results/

Every number written is computed from the run. Nothing is estimated.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


# ---------------------------------------------------------------- data

class FuncDataset(Dataset):
    def __init__(self, rows, tok, max_len):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["func"], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        return (enc["input_ids"][0], enc["attention_mask"][0],
                torch.tensor(r["target"], dtype=torch.long))


def load_rows(path, splitfile, arm, id_field):
    """Load the corpus and attach the split for the requested arm."""
    assign = None
    if splitfile:
        with open(splitfile, encoding="utf-8") as fh:
            assign = json.load(fh)

    rows = []
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
            rid = str(o.get(id_field, n))
            if assign is not None:
                sp = assign.get(rid)
                if sp is None:            # id absent from the split file
                    continue
            else:
                sp = str(o.get("split", "train"))
            rows.append({"id": rid, "func": o["func"],
                         "target": 1 if o.get("target") == 1 else 0,
                         "split": "valid" if sp in ("valid", "validation") else sp})

    by = defaultdict(list)
    for r in rows:
        by[r["split"]].append(r)
    if not by.get("test"):
        raise SystemExit(f"arm '{arm}': no test rows found -- check --splitfile/--id-field")
    return by


# ---------------------------------------------------------------- model

class Classifier(nn.Module):
    """Encoder + linear head over the CLS token of last_hidden_state.

    Deliberately NOT pooler_output: GraphCodeBERT's checkpoint ships no pooler
    weights, so pooler_output is a randomly initialised projection. That is
    survivable when fine-tuning end to end but silently destroys any frozen
    encoder variant, so the raw CLS token is used everywhere for consistency.
    """

    def __init__(self, name, dropout=0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(h, 2)

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.out(self.drop(h))


# ---------------------------------------------------------------- metrics

def vd_score(y, p, max_fpr=0.005):
    """FNR at FPR <= max_fpr. Lower is better."""
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= max_fpr
    return float(1.0 - tpr[ok].max()) if ok.any() else 1.0


def best_f1(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(prec), where=(prec + rec) > 0)
    i = int(np.argmax(f1))
    t = float(thr[i]) if i < len(thr) else 1.0
    return float(f1[i]), t


def evaluate(y, p):
    y, p = np.asarray(y), np.asarray(p)
    f1, thr = best_f1(y, p)
    return {
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "best_f1": round(f1, 4),
        "best_f1_threshold": round(thr, 4),
        "mcc": round(float(matthews_corrcoef(y, (p >= thr).astype(int))), 4),
        "vd_s": round(vd_score(y, p), 4),
        "positive_rate": round(float(y.mean()), 4),
        "n": int(len(y)),
    }


def length_floor(train, test):
    """Length-only logistic regression: the floor that every table needs."""
    xf = lambda rs: np.array([[len(r["func"]), len(r["func"].split())] for r in rs],
                             dtype=float)
    ytr = np.array([r["target"] for r in train])
    yte = np.array([r["target"] for r in test])
    if len(set(ytr)) < 2:
        return None
    clf = LogisticRegression(max_iter=1000).fit(np.log1p(xf(train)), ytr)
    p = clf.predict_proba(np.log1p(xf(test)))[:, 1]
    return evaluate(yte, p)


def worst_family(test, probs, famfile, min_members=5):
    """Lowest per-family PR-AUC -- the quantity Group DRO targets in RQ4."""
    if not famfile or not os.path.exists(famfile):
        return None
    with open(famfile, encoding="utf-8") as fh:
        fam = json.load(fh)
    groups = defaultdict(list)
    for r, p in zip(test, probs):
        f = fam.get(r["id"])
        if f is not None:
            groups[f].append((r["target"], p))
    scored = []
    for f, items in groups.items():
        y = [a for a, _ in items]
        if len(items) >= min_members and 0 < sum(y) < len(y):
            scored.append((float(average_precision_score(y, [b for _, b in items])), f))
    if not scored:
        return None
    scored.sort()
    return {"worst_family_pr_auc": round(scored[0][0], 4),
            "worst_family_id": scored[0][1],
            "families_scored": len(scored),
            "median_family_pr_auc": round(scored[len(scored) // 2][0], 4)}


# ---------------------------------------------------------------- run

# ---- model-selection rule (referee item 13 / M1) ---------------------------------------------
# Primary protocol: first_eligible = 1, which makes both functions reduce exactly to the original
# rule (new best if valid PR-AUC > best so far; stop when ep - best_epoch >= patience).
# Sensitivity (--select-after-warmup): only epochs that END after the warm-up are eligible, and
# patience is counted from the first eligible epoch.
PRIMARY_OUT_DIRS = {"rq2_diversevul_final", "rq2_diversevul_phase2", "rq2_diversevul_phase3",
                    "rq2_diversevul_phase4b", "rq2_devign_final", "rq2_devign_converged", "rq2_devign",
                    "rq2_diversevul"}


def first_eligible_epoch(warmup_steps: int, steps_per_epoch: int) -> int:
    """Smallest epoch whose last step lies strictly after the warm-up (15 epochs: 1.5 -> 2)."""
    ep = 1
    while ep * steps_per_epoch <= warmup_steps:
        ep += 1
    return ep


def is_new_best(ep: int, va: float, best_va: float, first_eligible: int = 1) -> bool:
    return ep >= first_eligible and va > best_va


def should_stop(ep: int, best_epoch: int, patience: int, first_eligible: int = 1) -> bool:
    return ep - max(best_epoch, first_eligible - 1) >= patience


def replay_selection(valid_pr_auc: list, patience: int, max_epochs: int, first_eligible: int = 1):
    """Apply the rule to a logged validation curve. Returns (best_epoch, stop_epoch, complete); complete is
    False when the rule would have trained past the last logged epoch (outcome then not determinable)."""
    best_va, best_ep = -1.0, 0
    for ep in range(1, max_epochs + 1):
        if ep > len(valid_pr_auc):
            return best_ep, None, False
        if is_new_best(ep, valid_pr_auc[ep - 1], best_va, first_eligible):
            best_va, best_ep = valid_pr_auc[ep - 1], ep
        if patience and should_stop(ep, best_ep, patience, first_eligible):
            return best_ep, ep, True
    return best_ep, max_epochs, True


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for ids, mask, _ in loader:
        logits = model(ids.to(device), mask.to(device))
        out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="microsoft/codebert-base")
    ap.add_argument("--splitfile", default=None,
                    help="id -> split json from clone_family_audit.py; "
                         "omit to use the jsonl's own split field")
    ap.add_argument("--arm", required=True,
                    choices=["published", "random_control", "family_disjoint",
                             "project_disjoint", "commit_disjoint",
                             "commit_hrfamily_disjoint", "commit_decon", "commit_sizectrl"])
    ap.add_argument("--famfile", default=None, help="id -> family json, for worst-family")
    ap.add_argument("--id-field", default="idx")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3,
                    help="MAXIMUM epochs; training stops early once validation "
                         "PR-AUC has not improved for --patience epochs")
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stopping patience in epochs (0 = disabled). "
                         "Set this rather than guessing an epoch count: every "
                         "Gate-1 run had validation PR-AUC still rising at "
                         "epoch 3, and an undertrained model has had less "
                         "opportunity to memorise clone families, which "
                         "biases the leakage measurement toward zero.")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                    help="'balanced' weights the cross-entropy by inverse class "
                         "frequency (n/(2*n_c)). Unweighted CE on DiverseVul's "
                         "5.84%% positives left seeds at wildly different points "
                         "on the learning curve (control-arm sd 0.0700 > the "
                         "0.0334 effect being measured), because the cheapest "
                         "way to cut the loss is to predict the majority class. "
                         "Default stays 'none' so earlier runs remain "
                         "reproducible; pass 'balanced' for imbalanced corpora.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit-train", type=int, default=None,
                    help="cap training rows (for a cheap smoke run)")
    ap.add_argument("--out", default="rq2_results")
    ap.add_argument("--ckpt-dir", default=None,
                    help="checkpoints for resume (default <out>/ckpt); lives on "
                         "the data disk so a 迁移 can pick up mid-run")
    ap.add_argument("--ckpt-every", type=int, default=1000,
                    help="save a resumable checkpoint every N training steps "
                         "inside an epoch (0 = epoch boundary only). DiverseVul "
                         "is ~16k steps/epoch; a crash then loses at most N steps.")
    ap.add_argument("--save-preds", action="store_true",
                    help="also write <out>/<tag>.preds.json.gz: test AND valid ids, "
                         "labels and probabilities from the selected (best-valid) "
                         "model, for bootstrap CIs and per-subset analyses "
                         "(RESEARCH_PLAN §13.4) without retraining")
    ap.add_argument("--save-best", action="store_true",
                    help="also write <out>/<tag>.best.pt: the selected model's "
                         "state_dict (~500 MB for a 125M encoder), so it can be "
                         "re-scored later without retraining")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore an existing checkpoint and start over")
    ap.add_argument("--select-after-warmup", action="store_true",
                    help="SENSITIVITY ONLY (referee item 13 / M1; default OFF, so every primary "
                         "result is reproduced unchanged). Model selection considers only epochs "
                         "that end after the linear warm-up (10%% of all steps), patience is "
                         "counted from the first such epoch, and test metrics are logged at every "
                         "epoch in history[].test for reporting only (never used for selection). "
                         "The run tag gets the suffix '_postwarmup' and --out must not be a primary "
                         "results directory, so primary files are never overwritten.")
    args = ap.parse_args()
    if args.select_after_warmup and os.path.basename(os.path.normpath(args.out)) in PRIMARY_OUT_DIRS:
        ap.error(f"--select-after-warmup writes to a separate directory; --out {args.out} "
                 f"is a primary results directory")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    by = load_rows(args.data, args.splitfile, args.arm, args.id_field)
    train, valid, test = by.get("train", []), by.get("valid", []), by["test"]
    if args.limit_train:
        random.Random(args.seed).shuffle(train)
        train = train[:args.limit_train]

    print(f"arm={args.arm} model={args.model} seed={args.seed} device={device}")
    print(f"  train {len(train)}  valid {len(valid)}  test {len(test)}")
    print(f"  train pos rate {np.mean([r['target'] for r in train]):.4f} | "
          f"test pos rate {np.mean([r['target'] for r in test]):.4f}")

    floor = length_floor(train, test)
    print(f"  length-only floor PR-AUC: {floor['pr_auc'] if floor else 'n/a'}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = Classifier(args.model).to(device)

    def make_loader(rows, bs, shuffle=False, ep=0, skip_steps=0):
        """Deterministic shuffle so a mid-epoch resume can skip completed batches
        without replaying them or desynchronising the optimiser."""
        ds = FuncDataset(rows, tok, args.max_len)
        if not shuffle:
            return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2,
                              pin_memory=True, drop_last=False)
        g = torch.Generator()
        g.manual_seed(args.seed * 1_000_003 + ep)
        order = torch.randperm(len(ds), generator=g).tolist()
        if skip_steps:
            order = order[skip_steps * bs:]
        return DataLoader(Subset(ds, order), batch_size=bs,
                          shuffle=False, num_workers=2, pin_memory=True,
                          drop_last=False)

    te_loader = make_loader(test, args.eval_batch_size)
    va_loader = make_loader(valid, args.eval_batch_size) if valid else None

    steps_per_epoch = max(1, (len(train) + args.batch_size - 1) // args.batch_size)
    steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    if args.class_weight == "balanced":
        cnt = np.bincount([r["target"] for r in train], minlength=2).astype(float)
        # n / (2 * n_c): the sklearn 'balanced' convention. Mean weight stays 1,
        # so the effective learning rate is unchanged and lr is still comparable
        # across weighted and unweighted runs.
        w = len(train) / (2.0 * np.maximum(cnt, 1.0))
        print(f"  class weights (balanced): neg {w[0]:.3f}  pos {w[1]:.3f}")
        lossf = nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float32, device=device))
    else:
        lossf = nn.CrossEntropyLoss()

    tag = f"{os.path.basename(args.model)}_{args.arm}_seed{args.seed}"
    warmup_steps = int(0.1 * steps)
    first_eligible = first_eligible_epoch(warmup_steps, steps_per_epoch) if args.select_after_warmup else 1
    if args.select_after_warmup:
        tag += "_postwarmup"
        print(f"  SENSITIVITY: select after warm-up ({warmup_steps} warm-up steps, "
              f"{steps_per_epoch} steps/epoch): first eligible epoch {first_eligible}; "
              f"test metrics logged every epoch for reporting only", flush=True)
    ckpt_dir = args.ckpt_dir or os.path.join(args.out, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{tag}.pt")
    meta_path = os.path.join(ckpt_dir, f"{tag}.meta.json")

    best_va, best_state, history = -1.0, None, []
    best_epoch = 0
    start_ep, start_step, elapsed_before = 1, 0, 0.0
    early_stopped = False
    resumed = False
    tot_carry = 0.0

    def write_meta(ep, step, epoch_done, extra=None):
        meta = {
            "tag": tag, "epoch": ep, "step": step, "epoch_done": epoch_done,
            "steps_per_epoch": steps_per_epoch, "best_va": best_va,
            "best_epoch": best_epoch, "early_stopped": early_stopped,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra:
            meta.update(extra)
        tmp = meta_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        os.replace(tmp, meta_path)

    def save_ckpt(ep, step, epoch_done):
        payload = {
            "epoch": ep, "step": step, "epoch_done": epoch_done,
            "model": model.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "scaler": scaler.state_dict(),
            "best_va": best_va, "best_state": best_state, "history": history,
            "best_epoch": best_epoch, "early_stopped": early_stopped,
            "tot": tot_carry,
            "elapsed": elapsed_before + (time.time() - t0),
            "py_rng": random.getstate(), "np_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_rng"] = torch.cuda.get_rng_state().cpu()
        torch.save(payload, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
        write_meta(ep, step, epoch_done)
        kind = "epoch" if epoch_done else f"ep{ep} step {step}"
        print(f"    checkpoint saved ({kind})", flush=True)

    t0 = time.time()
    if os.path.exists(ckpt_path) and not args.no_resume:
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            scaler.load_state_dict(ck["scaler"])
            best_va, best_state = ck["best_va"], ck["best_state"]
            best_epoch = ck.get("best_epoch", 0)
            history = ck["history"]
            elapsed_before = ck.get("elapsed", 0.0)
            tot_carry = ck.get("tot", 0.0)
            early_stopped = bool(ck.get("early_stopped", False))
            random.setstate(ck["py_rng"])
            np.random.set_state(ck["np_rng"])
            trng = ck["torch_rng"]
            torch.set_rng_state(trng.cpu() if hasattr(trng, "cpu") else trng)
            if "cuda_rng" in ck and torch.cuda.is_available():
                torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
            if ck.get("epoch_done") or ck.get("step", 0) == 0:
                start_ep, start_step = ck["epoch"] + 1, 0
                tot_carry = 0.0
            else:
                start_ep, start_step = ck["epoch"], ck.get("step", 0)
            resumed = True
            print(f"  RESUMED from {ckpt_path}: next ep{start_ep} step {start_step}+, "
                  f"best valid pr_auc {best_va}", flush=True)
        except Exception as e:
            print(f"  checkpoint unreadable ({type(e).__name__}: {e}); starting fresh")
            start_ep, start_step = 1, 0

    if early_stopped:
        print(f"  checkpoint already early-stopped at best epoch {best_epoch}; "
              f"skipping to test")
    elif start_ep > args.epochs:
        print(f"  all {args.epochs} epochs already complete in checkpoint")

    t0 = time.time()
    for ep in range(start_ep, args.epochs + 1):
        if early_stopped:
            break
        skip = start_step if ep == start_ep else 0
        tr_loader = make_loader(train, args.batch_size, shuffle=True,
                                ep=ep, skip_steps=skip)
        model.train()
        tot = tot_carry if (ep == start_ep and skip) else 0.0
        tot_carry = 0.0
        n_done = skip
        for i, (ids, mask, y) in enumerate(tr_loader, skip + 1):
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                loss = lossf(model(ids, mask), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            n_done = i
            if i % 200 == 0:
                print(f"    ep{ep} step {i}/{steps_per_epoch} loss {tot/i:.4f} "
                      f"({elapsed_before + time.time()-t0:.0f}s)", flush=True)
            if args.ckpt_every and i % args.ckpt_every == 0 and i < steps_per_epoch:
                tot_carry = tot
                save_ckpt(ep, i, epoch_done=False)
                tot_carry = 0.0

        rec = {"epoch": ep, "train_loss": round(tot / max(n_done, 1), 4)}
        if va_loader:
            vm = evaluate([r["target"] for r in valid], predict(model, va_loader, device))
            rec["valid"] = vm
            if is_new_best(ep, vm["pr_auc"], best_va, first_eligible):
                best_va = vm["pr_auc"]
                best_epoch = ep
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
        if args.select_after_warmup:
            # reporting only: never read by the selection or stopping rule
            rec["test"] = evaluate([r["target"] for r in test], predict(model, te_loader, device))
            rec["eligible_for_selection"] = ep >= first_eligible
        history.append(rec)
        print(f"  epoch {ep}: {rec}", flush=True)

        if args.patience and va_loader and should_stop(ep, best_epoch, args.patience, first_eligible):
            early_stopped = True
            print(f"  early stop: no valid improvement for {args.patience} epoch(s) "
                  f"since epoch {best_epoch} (best valid pr_auc {best_va:.4f})")
        save_ckpt(ep, n_done, epoch_done=True)
        start_step = 0
        if early_stopped:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    probs = predict(model, te_loader, device)
    metrics = evaluate([r["target"] for r in test], probs)

    # Optional artefacts for the Phase 2 analyses. Written atomically and BEFORE
    # the result JSON, so a run counts as done (JSON present) only once they
    # exist; resume.sh therefore never skips a run whose artefacts are missing.
    preds_file = best_file = None
    os.makedirs(args.out, exist_ok=True)
    if args.save_preds:
        va_probs = predict(model, va_loader, device) if va_loader else np.array([])
        preds_file = os.path.join(args.out, f"{tag}.preds.json.gz")
        payload = {
            "tag": tag, "arm": args.arm, "seed": args.seed, "best_epoch": best_epoch,
            "note": "probabilities from the model selected on valid PR-AUC",
            "test": {"id": [r["id"] for r in test],
                     "target": [int(r["target"]) for r in test],
                     "prob": [round(float(x), 6) for x in probs]},
            "valid": {"id": [r["id"] for r in valid],
                      "target": [int(r["target"]) for r in valid],
                      "prob": [round(float(x), 6) for x in va_probs]},
        }
        with gzip.open(preds_file + ".tmp", "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(preds_file + ".tmp", preds_file)
        print(f"  wrote {preds_file}")
    if args.save_best:
        best_file = os.path.join(args.out, f"{tag}.best.pt")
        state = best_state if best_state is not None else {
            k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save({"model": args.model, "best_epoch": best_epoch, "best_va": best_va,
                    "state_dict": state}, best_file + ".tmp")
        os.replace(best_file + ".tmp", best_file)
        print(f"  wrote {best_file}")
    wf = worst_family(test, probs, args.famfile)

    result = {
        "arm": args.arm, "model": args.model, "seed": args.seed,
        "data": args.data, "splitfile": args.splitfile,
        "epochs": args.epochs, "lr": args.lr, "max_len": args.max_len,
        "class_weight": args.class_weight,
        "batch_size": args.batch_size,
        "n_train": len(train), "n_valid": len(valid), "n_test": len(test),
        "test": metrics, "length_only_floor": floor, "worst_family": wf,
        "history": history,
        "runtime_seconds": round(elapsed_before + (time.time() - t0), 1),
        "resumed": resumed,
        "epochs_run": history[-1]["epoch"] if history else 0,
        "best_epoch": best_epoch,
        "max_epochs": args.epochs, "patience": args.patience,
        "converged": bool(args.patience and history
                          and history[-1]["epoch"] - best_epoch >= args.patience),
        "selected_on": "valid pr_auc" if va_loader else "last epoch",
        "preds_file": os.path.basename(preds_file) if preds_file else None,
        "best_weights_file": os.path.basename(best_file) if best_file else None,
    }
    if args.select_after_warmup:   # keys added only in the sensitivity mode; primary JSONs unchanged
        result.update({"selection_rule": "best valid pr_auc among epochs ending after the warm-up; "
                                         "patience counted from the first such epoch",
                       "warmup_steps": warmup_steps, "steps_per_epoch": steps_per_epoch,
                       "first_eligible_epoch": first_eligible,
                       "test_per_epoch_note": "history[].test is for reporting only; not used for selection"})

    out_json = os.path.join(args.out, f"{tag}.json")
    tmp_json = out_json + ".tmp"
    with open(tmp_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_json, out_json)

    for p in (ckpt_path, ckpt_path + ".tmp", meta_path, meta_path + ".tmp"):
        if os.path.exists(p):
            os.remove(p)

    print("\n=== TEST ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    if floor:
        print(f"  --- length-only floor PR-AUC {floor['pr_auc']} "
              f"(headroom {round(metrics['pr_auc'] - floor['pr_auc'], 4)}) ---")
    if wf:
        print(f"  worst-family PR-AUC {wf['worst_family_pr_auc']} "
              f"over {wf['families_scored']} families")
    print(f"\nWrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
