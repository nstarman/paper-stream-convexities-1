"""Unit tests for the shared figure style."""

import matplotlib

matplotlib.use("Agg")

from stream_convexity_paper.style import PAPER_RCPARAMS, use_paper_style


def test_style_matches_the_values_the_figures_used() -> None:
    assert PAPER_RCPARAMS == {
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


def test_use_paper_style_applies_every_key() -> None:
    matplotlib.rcdefaults()

    use_paper_style()

    for key, value in PAPER_RCPARAMS.items():
        actual = matplotlib.rcParams[key]
        # matplotlib normalizes a handful of rcParams (e.g. "font.family")
        # to a single-element list internally, even when set from a str.
        if isinstance(actual, list) and not isinstance(value, list):
            actual = actual[0] if len(actual) == 1 else actual
        assert actual == value

    matplotlib.rcdefaults()
