"""Writer for the `.values.json` files the manuscript quotes.

Each figure script exports the constants and measurements the text needs into
`src/data/values/<script>.values.json`. `src/scripts/make_variables.py` renders
those into `src/tex/output/**/*.tex`, which `ms.tex` reads through `\\variable{}`.

The format is fixed by that consumer: indented JSON with a trailing newline.
"""

__all__ = ("write_values",)

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def write_values(script: str | Path, values: Mapping[str, Any]) -> Path:
    """Write ``values`` to ``src/data/values/<script stem>.values.json``.

    Args:
        script: The calling script, normally ``__file__``.
        values: Derived values to export. Key order is preserved, so group
            related entries together.

    Returns:
        The path written.

    Note:
        This does not pass a `default=` hook to `json.dumps`, so a numpy
        scalar or array raises `TypeError` rather than being silently
        coerced. Wrap such a value in `float(...)` (or `.tolist()`) at the
        call site before passing it here.

    """
    from stream_convexity_paper.paths import VALUES

    VALUES.mkdir(parents=True, exist_ok=True)
    path = VALUES / (Path(script).stem + ".values.json")
    path.write_text(json.dumps(dict(values), indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}  (a build propagates it to the manuscript)")  # noqa: T201
    return path
