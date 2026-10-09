"""Shared plot style: thin marks, recessive axes, a fixed categorical colour order."""
import matplotlib as mpl

BLUE = "#2a78d6"    # series 1
ORANGE = "#eb6834"  # series 2
AQUA = "#1baf7a"    # series 3
INK = "#333333"
MUTED = "#8a8a8a"


def setup():
    mpl.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
    })
