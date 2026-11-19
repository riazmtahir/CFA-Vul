#!/usr/bin/env python3
"""fig_overview.py -- Figure "Study overview" (method diagram), vector PDF; navy/slate colours outside the split palette (20 Nov 2026).

Content follows sections/study_design.tex and sections/method_core.tex: three corpora; the clone-family
audit (Algorithm 1; strict criterion theta = 0.7 with 0.8/0.9 bounds; high-recall criterion, exploratory);
the split arms (Algorithm 2) and the two H4b training variants (Eq. decon); CodeBERT under the frozen
protocol (Section 3.5; fine-tuned end-to-end, linear head); the analyses per RQ with their labels. Drawn at print size (5.42 in), no text below 7 pt. No number in the figure is new: all are in
study_design.tex (learning rate, epochs, patience, seeds, runs).

Design round (19 Nov 2026, author comment 25): every box has a shaded title band, corpus names are set in
bold with their sources indented beneath, thin rules separate the sub-blocks of a box, the training box has a
column rule, and the suspended compounds ("family-, project- and commit-disjoint", "H2 project-, H3
commit-disjoint") are written out in full (author comment 2). Text content is otherwise unchanged.
Usage (repository root): python analysis/scripts/fig_overview.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
import figstyle as fs  # noqa: E402  shared style: fonts, sizes, line widths, export
fs.apply()

# Drawn at print size: the figure is included at \linewidth = 390 pt = 5.42 in, so every font size below is
# the size on paper. Minimum 7 pt (review item 47). Three rows: pipeline, training and evaluation, analyses.
W, H = 5.42, 3.92
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
TITLE, BODY = fs.TITLE, fs.SMALL   # pt on paper (8 and 7)
LH = BODY * 1.3 / 72               # line height in inches
BAND_H = 0.20                      # title band height (in)
EDGE = fs.SLATE_EDGE
# colour round (20 Nov 2026): navy title bands with white titles for the pipeline boxes, light slate bands for
# the analysis (RQ) boxes; the slate/navy family is not used for any split
PIPE_BAND, RQ_BAND, RQ_FILL = fs.NAVY, fs.SLATE_LIGHT, "#F3F6F9"
TITLE_ON = {fs.NAVY: "white"}


def panel(x, y, w, h, title, band=PIPE_BAND, fill="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                linewidth=0.7, edgecolor=EDGE, facecolor=fill, zorder=1))
    # title band: a rectangle clipped to the rounded box, with a rule under it
    clip = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", transform=ax.transData,
                          facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    band_rect = Rectangle((x, y + h - BAND_H), w, BAND_H, facecolor=band, edgecolor="none", zorder=1.5)
    ax.add_patch(band_rect)
    band_rect.set_clip_path(clip)
    ax.plot([x, x + w], [y + h - BAND_H] * 2, color=EDGE, lw=0.5, zorder=2)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                linewidth=0.7, edgecolor=EDGE, facecolor="none", zorder=3))
    ax.text(x + w / 2, y + h - BAND_H / 2, title, ha="center", va="center", fontsize=TITLE,
            fontweight="bold", zorder=4, color=TITLE_ON.get(band, fs.INK))


def lines(x, y, items):
    """items: (text, style) with style in {"", "b", "i"} (b = bold, i = indented); "--" draws a rule.
    Returns the y below the last line."""
    for text, style in items:
        if text == "--":
            y -= LH * 0.35
            ax.plot([x, x + RULE_W], [y, y], color="0.70", lw=0.45, zorder=4)
            y -= LH * 0.45
            continue
        ax.text(x + (0.10 if style == "i" else 0), y, text, ha="left", va="top", fontsize=BODY,
                fontweight="bold" if style == "b" else "normal", zorder=4)
        y -= LH
    return y


def box(x, y, w, h, title, items, band=PIPE_BAND, fill="white"):
    global RULE_W
    panel(x, y, w, h, title, band, fill)
    RULE_W = w - 0.14
    lines(x + 0.07, y + h - BAND_H - 0.06, items)


def arrow(x0, y0, x1, y1, style="arc3"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                 linewidth=0.8, color=fs.NAVY, shrinkA=0, shrinkB=0, connectionstyle=style,
                                 zorder=5))


RULE_W = 1.0
# row 1: corpora -> clone-family audit -> splits
R1, H1 = 2.42, 1.46
box(0.04, R1, 1.44, H1, "Corpora (C/C++)",
    [("Devign", "b"), ("CodeXGLUE release", "i"), ("DiverseVul", "b"), ("unofficial Hugging", "i"),
     ("Face mirror", "i"), ("PrimeVul", "b"), ("released split", "i")])
box(1.66, R1, 1.70, H1, "Clone-family audit",
    [("Type-2 normal form, token", ""), ("shingles, MinHash + LSH,", ""), ("exact Jaccard, union-find", ""),
     ("--", ""), ("strict: k = 5, θ = 0.7", ""), ("(bounds 0.8, 0.9)", "i"), ("--", ""),
     ("high-recall: blind ids,", ""), ("k = 3 (exploratory)", "i")])
box(3.54, R1, 1.84, H1, "Splits (DiverseVul)",
    [("published, random control,", ""), ("family-disjoint, project-disjoint", ""), ("and commit-disjoint", ""),
     ("--", ""), ("H4b variants of the", ""), ("commit-disjoint training set:", ""),
     ("decontaminated, matched", "i"), ("stratified control", "i")])
# row 2: RQ1 (model-free) beside training and evaluation
R2, H2 = 1.24, 1.02
box(0.04, R2, 1.62, H2, "RQ1: leakage audit",
    [("no model; all corpora;", ""), ("published split and", ""), ("random control;", ""), ("θ = 0.7–0.9", "")],
    band=RQ_BAND, fill=RQ_FILL)
panel(1.84, R2, 3.54, H2, "Training and evaluation (every split)")
RULE_W = 1.6
lines(1.91, R2 + H2 - BAND_H - 0.06,
      [("CodeBERT, fine-tuned end-to-end:", "b"), ("[CLS] → dropout → linear head", ""),
       ("class-weighted cross-entropy,", ""), ("lr 0.00001, ≤ 15 epochs, early", ""),
       ("stopping on validation PR-AUC", ""), ("(patience 2), best epoch kept", "")])
ax.plot([3.60, 3.60], [R2 + 0.08, R2 + H2 - BAND_H - 0.06], color="0.70", lw=0.45, zorder=4)
lines(3.66, R2 + H2 - BAND_H - 0.06,
      [("PR-AUC (primary), headroom over", ""), ("length-only baseline, MCC, ROC-AUC", ""),
       ("thresholds: validation max-F1;", ""), ("test max-F1 for MCC (paired only)", ""),
       ("seeds 1–3; 1–5 for random control", ""), ("and family-disjoint (25 runs)", "")])
# row 3: RQ2 and RQ3 below training (shaded)
R3, H3 = 0.04, 1.02
box(1.84, R3, 1.68, H3, "RQ2: removing leakage",
    [("family contrast (n = 5)", ""), ("H2 project-disjoint;", ""), ("H3 commit-disjoint;", ""),
     ("H4 gate failed;", ""), ("H4b decontaminated", "")], band=RQ_BAND, fill=RQ_FILL)
box(3.68, R3, 1.70, H3, "RQ3: leaked material",
    [("paired evaluation (descr.)", ""), ("H1 error rates by leak", ""), ("category; counterparts,", ""),
     ("re-scoring (exploratory)", "")], band=RQ_BAND, fill=RQ_FILL)

arrow(1.48, R1 + 0.80, 1.66, R1 + 0.80)     # corpora -> audit
arrow(3.36, R1 + 1.02, 3.54, R1 + 1.02)     # strict families -> family-disjoint
arrow(3.36, R1 + 0.28, 3.54, R1 + 0.28)     # direct high-recall pairs -> H4b variants
arrow(4.46, R1, 4.46, R2 + H2)              # splits -> training and evaluation
arrow(1.80, R1, 1.30, R2 + H2, "angle,angleA=-90,angleB=180,rad=0")   # audit -> RQ1 (no model)
arrow(2.68, R2, 2.68, R3 + H3)              # training -> RQ2
arrow(4.53, R2, 4.53, R3 + H3)              # training -> RQ3
fs.save(fig, os.path.join(PAPER, "figures", "overview.pdf"))
print("wrote figures/overview.pdf")
