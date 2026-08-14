from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns

FONT_DIR = Path(__file__).resolve().parents[2] / "fonts"

for font_path in FONT_DIR.glob("*.ttf"):
    fm.fontManager.addfont(font_path)

CREAM = "#FBF8F3"
BROWN = "#4A4036"

Q_BLUE = "#7EA6C6"
MC_ROSE = "#D28F92"

Q_BLUE_LIGHT = "#BFD3E2"
MC_ROSE_LIGHT = "#E7C2C3"

SAGE = "#A9BE9A"
LAVENDER = "#B9ADD2"
PEACH = "#E3B18D"

GRID = "#D8D0C6"


def setup():
    sns.set_theme(
        style="white",
        context="paper",
    )

    plt.rcParams.update({
        # Typography
        "font.family": "CMU Serif",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 11.5,
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",

        # Figure
        "figure.facecolor": CREAM,
        "axes.facecolor": CREAM,

        # Text
        "text.color": BROWN,
        "axes.labelcolor": BROWN,
        "axes.titlecolor": BROWN,
        "xtick.color": BROWN,
        "ytick.color": BROWN,

        # Axes
        "axes.edgecolor": BROWN,
        "axes.linewidth": 0.8,

        # Grid
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.32,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",

        # Legend
        "legend.frameon": False,
        "legend.fontsize": 10,

        # Lines
        "lines.linewidth": 2.4,

        # Saving
        "savefig.facecolor": CREAM,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,

        # PDF/PS text remains actual text
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def finish(ax):
    """
    Final cleanup for a paper-style Decoy axis.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)

    ax.grid(
        axis="y",
        color=GRID,
        alpha=0.32,
        linewidth=0.7,
    )

    ax.grid(axis="x", visible=False)

    ax.tick_params(
        length=3,
        width=0.7,
    )

    return ax