r"""Physical inputs for the paper, loaded from ``variables.yml``.

``variables.yml`` is the single source of truth for numbers that appear both in
the analysis code and in the manuscript. Figure scripts read them from
``PARAMS``; ``src/scripts/make_variables.py`` renders them into
``output/**/*.tex``, which ``ms.tex`` reads through ``\variable{}``.

This module deliberately imports nothing heavier than PyYAML, so the exporter
does not have to pay for jax.
"""

import math
import pathlib

from typing import Any, Final

import yaml

REPO_ROOT: Final = pathlib.Path(__file__).resolve().parents[2]
VARIABLES_FILE: Final = REPO_ROOT / "variables.yml"
OUTPUT_DIR: Final = REPO_ROOT / "src" / "tex" / "output"

if not VARIABLES_FILE.is_file():  # pragma: no cover
    msg = (
        f"cannot find {VARIABLES_FILE}. `stream_convexity_paper` must be "
        "installed from the repository (an editable install) for the physical "
        "inputs to be locatable."
    )
    raise FileNotFoundError(msg)

def _coerce_numbers(node: Any) -> Any:  # noqa: ANN401
    """Recursively turn numeric-looking strings into floats.

    PyYAML implements YAML 1.1, whose float pattern demands a signed exponent,
    so ``1.0e12`` loads as the string ``"1.0e12"`` while ``1.0e+12`` loads as a
    float. Normalising here means neither the figure scripts nor the exporter
    have to know about the quirk. Genuinely non-numeric strings are untouched.
    """
    if isinstance(node, dict):
        return {k: _coerce_numbers(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_coerce_numbers(v) for v in node]
    if isinstance(node, str):
        try:
            return float(node)
        except ValueError:
            return node
    return node


with VARIABLES_FILE.open(encoding="utf-8") as f:
    PARAMS: Final[dict[str, Any]] = _coerce_numbers(yaml.safe_load(f))

#: Physical unit (as the analysis code spells it) -> siunitx unit macro.
SI_UNITS: Final[dict[str, str]] = {
    "Msun": r"\Msun",
    "kpc": r"\kilo\parsec",
    "Gyr": r"\giga\year",
    "deg": r"\degree",
}

#: Powers of ten at or above this exponent render as ``eNN`` for siunitx,
#: so 1e12 becomes ``\qty{e12}{...}`` rather than ``\qty{1000000000000}{...}``.
_EXPONENT_THRESHOLD: Final = 4


def value_unit(entry: dict[str, Any]) -> tuple[float, str]:
    """Return ``(value, unit)``, ready to splat into a Quantity constructor.

    Keeps the analysis code free of the entry-dict layout, and keeps unxt out
    of this module.

    >>> value_unit({"value": 14.0, "unit": "kpc"})
    (14.0, 'kpc')

    """
    return entry["value"], entry["unit"]


def tex_number(x: float) -> str:
    """Render a number the way siunitx wants to receive it.

    Exact powers of ten at or above ``_EXPONENT_THRESHOLD`` become ``eNN``;
    everything else uses the shortest round-tripping decimal form.

    >>> tex_number(1.0e12)
    'e12'
    >>> tex_number(14.0)
    '14'
    >>> tex_number(0.85)
    '0.85'

    """
    if x == 0:
        return "0"
    exponent = math.log10(abs(x))
    if exponent.is_integer() and abs(exponent) >= _EXPONENT_THRESHOLD:
        sign = "-" if x < 0 else ""
        return f"{sign}e{int(exponent)}"
    return f"{x:g}"


def tex_quantity(entry: dict[str, Any]) -> str:
    r"""Render one ``variables.yml`` entry as complete TeX.

    A dimensionless entry renders as a bare number, one with a unit as a
    full ``\qty{}{}`` call. The result is complete TeX rather than a bare
    number because ``\input`` is not expandable, so a ``\variable{}`` can
    never be nested inside a siunitx argument.

    >>> tex_quantity({"value": 14.0, "unit": "kpc"})
    '\\qty{14}{\\kilo\\parsec}'
    >>> tex_quantity({"value": 0.85})
    '0.85'

    """
    number = entry.get("tex") or tex_number(entry["value"])
    unit = entry.get("unit")
    if unit is None:
        return number
    if unit not in SI_UNITS:
        msg = f"no siunitx spelling for unit {unit!r}; add one to SI_UNITS"
        raise KeyError(msg)
    return rf"\qty{{{number}}}{{{SI_UNITS[unit]}}}"


def tex_list(values: list[float]) -> str:
    """Render a sequence as a comma-separated list.

    >>> tex_list([0.3, 0.5, 0.7, 0.9])
    '0.3, 0.5, 0.7, 0.9'

    """
    return ", ".join(tex_number(v) for v in values)


def tex_qtyrange(entry: dict[str, Any]) -> str:
    r"""Render a ranged entry with units as a siunitx range.

    >>> tex_qtyrange({"value": [35.0, 100.0], "unit": "kpc"})
    '\\qtyrange{35}{100}{\\kilo\\parsec}'

    """
    lo, hi = entry["value"]
    unit = entry["unit"]
    if unit not in SI_UNITS:
        msg = f"no siunitx spelling for unit {unit!r}; add one to SI_UNITS"
        raise KeyError(msg)
    return rf"\qtyrange{{{tex_number(lo)}}}{{{tex_number(hi)}}}{{{SI_UNITS[unit]}}}"


def tex_interval(bounds: list[float]) -> str:
    """Render a two-element range as a closed interval.

    >>> tex_interval([0.05, 0.10])
    '[0.05, 0.1]'

    """
    lo, hi = bounds
    return f"[{tex_number(lo)}, {tex_number(hi)}]"
