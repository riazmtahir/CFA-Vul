"""figstyle.py -- one shared style for every figure of the paper and the supplement (13 Nov 2026).

Import it in each figure script (``import figstyle as fs; fs.apply()``) and draw every figure at its PRINT
width (``fs.MAIN``, ``fs.MAIN_075`` or ``fs.SUPP``), so the font sizes below are the sizes on paper:
no text is set below 7 pt.

Palette: Okabe and Ito (2008), designed for colour-vision deficiencies. Every split has one colour AND one
marker shape, used in every figure; grey-only measures (bars that are not splits) use grey fills and hatches,
so each figure also reads in grayscale (marker shapes, hatches and line styles carry the distinction).

Design round (19 Nov 2026, author comment 25): near-black ink instead of pure black for axes and text,
solid light grid lines, bold left-aligned panel titles, alternating light row bands and two-level row labels
(primary label in ink, secondary detail in grey) for the forest plots, and one export routine ``save`` that
writes every figure as PDF (vector, TrueType fonts embedded), SVG (vector, text as paths) and PNG at 300
and 600 dpi. Data, labels and the split palette are unchanged.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt

# print widths (inches): manuscript \linewidth = 390 pt = 5.42 in; supplement text width = 17 cm = 6.69 in
MAIN = 5.42
MAIN_075 = 0.75 * MAIN
SUPP = 6.69

FONT = "Arial"
BASE = 8.0      # axis labels, tick labels
SMALL = 7.0     # legends, value labels, direct labels: the minimum anywhere
TITLE = 8.0     # panel titles

AXES_LW = 0.6
LINE_LW = 1.1
ERR_LW = 0.9
EDGE_LW = 0.5
MARKER = 4.6
INK = "#1F2328"       # axes, ticks and text (near-black; prints as black)
MUTED = "0.38"        # secondary text: analysis labels, n, pair counts
BAND = "0.955"        # alternating row bands of the forest plots (light enough for grayscale print)
GRID = {"color": "0.87", "linewidth": 0.45, "linestyle": "-"}
REF = {"color": "0.30", "linewidth": 0.75, "linestyle": "--"}    # reference lines (zero, chance, 10%)

# Okabe-Ito colours
BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERMILLION, PURPLE = (
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

# one colour, one marker and one line style per split, everywhere
SPLIT = {
    "published":        {"color": BLACK,      "marker": "D", "ls": "-",  "label": "Published"},
    "random_control":   {"color": BLUE,       "marker": "o", "ls": "-",  "label": "Random control"},
    "family_disjoint":  {"color": ORANGE,     "marker": "s", "ls": "--", "label": "Family-disjoint"},
    "project_disjoint": {"color": VERMILLION, "marker": "^", "ls": "-.", "label": "Project-disjoint"},
    "commit_disjoint":  {"color": GREEN,      "marker": "v", "ls": ":",  "label": "Commit-disjoint"},
    "decontaminated":   {"color": PURPLE,     "marker": "P", "ls": "--", "label": "Decontaminated (H4b)"},
    "matched_control":  {"color": SKY,        "marker": "X", "ls": "-.", "label": "Matched stratified control (H4b)"},
}
LENGTH_ONLY = {"color": "0.45", "marker": "x"}          # the length-only baseline / scorer
# aliases used by result files
SPLIT["commit_decon"] = SPLIT["decontaminated"]
SPLIT["commit_sizectrl"] = SPLIT["matched_control"]

# Colour round (20 Nov 2026, author request after comment 25): colour for the elements that are NOT splits,
# taken from a slate/navy family that is not used for any split, so no colour has two meanings.
NAVY = "#1F3A5F"          # title bands of the overview diagram, arrows
SLATE_DARK = "#334E68"
SLATE_MID = "#829AB1"
SLATE_LIGHT = "#D9E2EC"
SLATE_EDGE = "#486581"
# corpora (Fig. 2 only; not splits): muted colours outside the split palette, plus one marker and the
# panel's line style per corpus, so the figure still reads in grayscale
CORPUS = {"devign": {"marker": "o", "color": "#3E6B66"}, "diversevul": {"marker": "s", "color": NAVY},
          "primevul": {"marker": "^", "color": "#9A4A1E"}}

# fills and hatches for measures that are not splits (Fig. 4): a slate ramp, light to dark, plus a hatch
GREYS = ["white", "0.75", "0.45", "0.25"]          # kept for the supplementary figures
MEASURE_FILLS = [SLATE_LIGHT, SLATE_MID, SLATE_DARK, "#EEF2F6"]
HATCHES = ["", "", "", "////"]


def apply():
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "font.family": "sans-serif", "font.sans-serif": [FONT],
        "font.size": BASE, "axes.titlesize": TITLE, "axes.labelsize": BASE,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 4.0,
        "xtick.labelsize": BASE, "ytick.labelsize": BASE, "legend.fontsize": SMALL,
        "axes.linewidth": AXES_LW, "xtick.major.width": AXES_LW, "ytick.major.width": AXES_LW,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "lines.linewidth": LINE_LW, "lines.markersize": MARKER, "hatch.linewidth": EDGE_LW,
        "hatch.color": INK, "patch.edgecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "savefig.format": "pdf",
    })


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, **GRID)
    ax.set_axisbelow(True)


def bands(ax, n):
    """Light bands behind every second row (rows at y = 0..n-1) of a forest plot."""
    for y in range(n):
        if y % 2 == 1:
            ax.axhspan(y - 0.5, y + 0.5, color=BAND, lw=0, zorder=0)


def row_labels(ax, rows, pad=0.02):
    """Two-level row labels at y = 0..n-1: rows = [(primary, secondary), ...] from the top row down.

    Primary text in ink at the base size, secondary text in grey at the minimum size, right-aligned at
    the left edge of the axes (replaces the tick labels)."""
    import matplotlib.transforms as mtrans
    tr = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
    n = len(rows)
    ax.set_yticks(range(n))
    ax.set_yticklabels([""] * n)
    for i, (primary, secondary) in enumerate(rows):
        y = n - 1 - i
        ax.text(-pad, y + 0.02, primary, transform=tr, ha="right", va="bottom", fontsize=BASE, color=INK,
                linespacing=1.1)
        ax.text(-pad, y - 0.04, secondary, transform=tr, ha="right", va="top", fontsize=SMALL, color=MUTED)


def save(fig, pdf_path):
    """Write the figure as PDF (the file LaTeX uses) and, under paper/figures/export/, as SVG and PNG at
    300 and 600 dpi, all from the same drawing. Creation dates are left out so reruns are byte-stable."""
    here = os.path.dirname(os.path.abspath(__file__))
    export = os.path.join(os.path.dirname(here), "figures", "export")
    os.makedirs(export, exist_ok=True)
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    fig.savefig(pdf_path, format="pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(os.path.join(export, stem + ".svg"), format="svg", metadata={"Date": None})
    for dpi in (300, 600):
        fig.savefig(os.path.join(export, f"{stem}_{dpi}dpi.png"), format="png", dpi=dpi,
                    metadata={"Software": None})
    return pdf_path
