"""Export Paper Figure 5's dataset (its interactive version: boxes drawn by the reader) → <out>/uniform-box-rates/.

    uv run --locked --with "figpress-apps==2026.10.29" \
        python webapps/uniform-box-rates/export.py --out build/data

step_2_plot_triaxiality.py --simple is run (its file writes switched off) and everything comes from
it: the figure, with the right panel's CDM and SIDM labels saying they are population-weighted, and,
on a fine grid of shapes (c/a, b/a) over the left panel, each shape's mean projected flattening
(qbar_of) and the two rates the paper reads off its curves at that flattening (rate_at on the median
lambda_det^(10) curve and on the upper lambda_arc envelope, as for its CDM and SIDM diamonds). A box
the reader draws averages the grid cells inside it, each with the same weight: the same computation
as the paper's diamonds, with a uniform distribution of shapes in place of the halos'.
"""

import argparse
import io
import runpy
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

import figpress_apps

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src" / "scripts" / "cosmology" / "step_2_plot_triaxiality.py"
PAPER = "https://ui.adsabs.harvard.edu/abs/2026arXiv260940057S/abstract"
DPI = 200  # the figure is 13 in wide: ~2300 px, sharp at 2× in a 1100 px column
STEP_CA, STEP_BA = 0.005, 0.0025  # grid cells; the left panel is ~4× taller per unit b/a
LABELS = {"CDM": "CDM\nAuriga-\nweighted", "SIDM": "SIDM\npopulation-\nweighted"}


def run_script() -> dict:
    """step_2_plot_triaxiality.py --simple's namespace, its figure drawn and unsaved."""
    import stream_convexity_paper.values as values
    from matplotlib.figure import Figure

    mpl.use("agg")
    sys.path.insert(0, str(SCRIPT.parent))
    argv, write_values, savefig = sys.argv, values.write_values, Figure.savefig
    sys.argv = [str(SCRIPT), "--simple"]
    values.write_values, Figure.savefig = (lambda *a, **k: None), (lambda *a, **k: None)
    try:
        return runpy.run_path(str(SCRIPT), run_name="__main__")
    finally:
        sys.argv, values.write_values, Figure.savefig = argv, write_values, savefig


def box_mean(grid: dict, ca: tuple, ba: tuple) -> dict:
    """A box's uniform mean over the grid cells whose centres it holds (science.js's boxMean)."""
    inside = ((grid["ca"][None, :] >= ca[0]) & (grid["ca"][None, :] <= ca[1])
              & (grid["ba"][:, None] >= ba[0]) & (grid["ba"][:, None] <= ba[1]) & np.isfinite(grid["qbar"]))
    return {"n": int(inside.sum()), **{k: float(grid[k][inside].mean()) for k in ("qbar", "lam10", "lamarc")}}


def main(out: Path) -> Path:
    ns = run_script()
    fig, left, right = ns["fig"], ns["ax2"], ns["ax_top"]
    for t in right.texts:  # the paper's population labels: say how they are weighted
        name = t.get_text().split("\n")[0]
        if name in LABELS:
            t.set_text(LABELS[name])

    # Cell centres over the left panel's view; b/a < c/a is excluded by definition (NaN).
    (ca1, ca0), (ba1, ba0) = sorted(left.get_xlim())[::-1], sorted(left.get_ylim())[::-1]
    ca = np.arange(ca0 + STEP_CA / 2, ca1, STEP_CA)
    ba = np.arange(ba0 + STEP_BA / 2, ba1, STEP_BA)
    BA, CA = np.meshgrid(ba, ca, indexing="ij")
    ok = BA >= CA
    qbar = np.full(BA.shape, np.nan)
    qbar[ok] = ns["qbar_of"](BA[ok], CA[ok])
    rate_at = ns["rate_at"]
    grid = {"ca": ca, "ba": ba, "qbar": qbar,
            "lam10": np.where(ok, rate_at(np.nan_to_num(qbar, nan=1.0), ns["m3_mid"]), np.nan),
            "lamarc": np.where(ok, rate_at(np.nan_to_num(qbar, nan=1.0), ns["m2_hi"]), np.nan)}

    fig.set_dpi(DPI)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    box = fig.get_tightbbox(r).padded(0.1)  # as the script's savefig(bbox_inches="tight")

    def frac(ax):  # the axes box → [left, top, width, height] as fractions of the image
        e = ax.get_window_extent(r)
        return [(e.x0 / fig.dpi - box.x0) / box.width, (box.y1 - e.y1 / fig.dpi) / box.height,
                e.width / fig.dpi / box.width, e.height / fig.dpi / box.height]

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches=box)
    cdm = ((ns["CDM_CA_MIN"], ns["CDM_CA_MAX"]), (ns["CDM_BA_MIN"], ns["CDM_BA_MAX"]))
    sidm = ((ns["SIDM_CA_MIN"], ns["SIDM_CA_MAX"]), (ns["SIDM_BA_MIN"], ns["SIDM_BA_MAX"]))
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in grid.items()}
    arrays["paper"] = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    descriptions = {
        "ca": "c/a at each grid column's centre", "ba": "b/a at each grid row's centre",
        "qbar": "mean projected flattening of the shape (qbar_of); NaN where b/a < c/a",
        "lam10": "lambda_det^(10) read off the median curve at qbar (rate_at)",
        "lamarc": "lambda_arc read off the upper envelope at qbar (rate_at)",
        "paper": "PNG: Paper Figure 5 (step_2_plot_triaxiality.py --simple), its population labels saying how they are weighted",
    }
    axes = lambda ax: {"axes": frac(ax), "xlim": list(map(float, ax.get_xlim())), "ylim": list(map(float, ax.get_ylim()))}
    meta = {
        "title": "Paper Figure 5: convexity rate and halo shape", "paper": PAPER,
        "layout": {"size_pt": [box.width * 72, box.height * 72], "left": axes(left), "right": axes(right)},
        # The paper's populations, weighted by their halos, for comparison; and its interquartile boxes.
        "populations": {
            "cdm": {"qbar": float(ns["QBAR_CDM"]), "lam10": float(ns["_y_m3_marks"][1]), "lamarc": float(ns["_y_m2_marks"][1]),
                    "box": {"ca": list(map(float, cdm[0])), "ba": list(map(float, cdm[1]))}},
            "sidm": {"qbar": float(ns["QBAR_SIDM"]), "lam10": float(ns["_y_m3_marks"][0]), "lamarc": float(ns["_y_m2_marks"][0]),
                     "box": {"ca": list(map(float, sidm[0])), "ba": list(map(float, sidm[1]))}},
        },
        # What science.test.js checks boxMean against: the CDM box, uniform.
        "check": {"cdm_box_uniform": box_mean(grid, *cdm)},
    }
    return figpress_apps.export(out, "uniform-box-rates", arrays, meta=meta, descriptions=descriptions,
                                licence="MIT", overwrite=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("build/data"))
    print(main(ap.parse_args().out))
