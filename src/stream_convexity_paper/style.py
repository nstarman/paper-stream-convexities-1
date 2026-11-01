"""The manuscript's matplotlib style.

These are the rcParams the single-column figures are drawn with: 8 pt serif
body text, 10 pt axis labels, hairline spines and inward ticks on all four
sides. Sizes are absolute, so a figure must be sized in inches to the column
width rather than scaled in LaTeX.
"""

__all__ = ("PAPER_RCPARAMS", "use_paper_style")

import matplotlib as mpl

PAPER_RCPARAMS: dict[str, object] = {
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.fontsize": 7,
}


def use_paper_style() -> None:
    """Apply the manuscript style to the global matplotlib state."""
    mpl.rcParams.update(PAPER_RCPARAMS)
