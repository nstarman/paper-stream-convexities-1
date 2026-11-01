"""Export the full convexity explorer's dataset: Fig. 4's streams and Fig. 3's test orbits.

    uv run --locked --with "figpress-apps==2026.10.28" \
        python webapps/convexity-full/export.py --out build/data

The same dataset as webapps/convexity (same name, so the two apps share their code) plus the orbits.
"""

import argparse
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("convexity_export", Path(__file__).resolve().parents[1] / "convexity" / "export.py")
convexity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convexity)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("build/data"))
    print(convexity.main(ap.parse_args().out, orbits=True))
