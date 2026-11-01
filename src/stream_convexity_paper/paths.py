"""Where the showyourwork layout keeps things; scripts build every path from here."""

import pathlib

from typing import Final

ROOT: Final = pathlib.Path(__file__).resolve().parents[2]
DATA: Final = ROOT / "src" / "data"  # generated, gitignored
STATIC: Final = ROOT / "src" / "static"  # inputs not made by this paper's code, committed
FIGURES: Final = ROOT / "src" / "tex" / "figures"
OUTPUT: Final = ROOT / "src" / "tex" / "output"
MANUSCRIPT: Final = ROOT / "src" / "tex" / "ms.tex"
#: `<script stem>.values.json` files: the numbers each script measures or chooses.
VALUES: Final = DATA / "values"
