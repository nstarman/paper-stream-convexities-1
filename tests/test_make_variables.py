"""Unit tests for the exporter's number formatting."""

import importlib.util

from stream_convexity_paper.paths import ROOT

_spec = importlib.util.spec_from_file_location(
    "make_variables", ROOT / "src" / "scripts" / "make_variables.py",
)
make_variables = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_variables)


def test_sigfig_keeps_trailing_zeros() -> None:
    """0.90 is two significant figures; 0.9 would claim one."""
    assert make_variables.tex_sigfig(0.9031222616773218) == "0.90"
    assert make_variables.tex_sigfig(0.8253703249205742) == "0.83"


def test_sigfig_rounds_half_up() -> None:
    """Half-up, not Python's round-half-even."""
    assert make_variables.tex_sigfig(0.665) == "0.67"
    assert make_variables.tex_sigfig(7.784287165675167) == "7.8"
