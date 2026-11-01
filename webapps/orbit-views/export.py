"""Export Paper Figure 3's dataset: the two test orbits and their viewing-angle maps → <out>/orbit-views/.

    uv run --locked --with "figpress-apps==2026.10.29" \
        python webapps/orbit-views/export.py --out build/data

The orbits come from step_1_compute_orbits.py's triaxial_data.npz: their tracks (position, velocity,
acceleration) for the sky panels, their maps on the 3° grid, and the paper's two views. The look comes
from the paper's own figure: step_2_plot_mollweide.py is run (its file writes switched off), what
moves with a view is hidden — the panels' orbits, ticks, grid and (l, b) boxes, and the map's View
dots — and the rest is kept as an image, with where each axes sits, so the app draws only what moves.
"""

import argparse
import io
import runpy
from pathlib import Path

import matplotlib as mpl
import numpy as np

import figpress_apps

from stream_convexity_paper import paths

PAPER = "https://ui.adsabs.harvard.edu/abs/2026arXiv260940057S/abstract"
R_NEAR_KPC = 25.0  # step_2's R_NEAR_KPC: orbit 2 is dotted inside it
# id, key in the npz, panel title, colour map of its map layer, the panel's rotation (step_2 turns
# orbit 2's sky by −45°)
ORBITS = [("orbit1", "circ", "Orbit 1 View", "viridis", 0.0), ("orbit2", "ellip", "Orbit 2 View", "cividis", -45.0)]


def lut(name: str) -> np.ndarray:
    """A matplotlib colour map as 256 RGB bytes."""
    return np.rint(255 * mpl.colormaps[name](np.linspace(0, 1, 256))[:, :3]).astype(np.uint8)


DPI = 160  # the static layer: ~2400 px across, sharp at 2× in a 1200 px column
STEP_2 = Path(__file__).resolve().parents[2] / "src" / "scripts" / "triaxial" / "step_2_plot_mollweide.py"


def paper_layer() -> tuple[np.ndarray, dict]:
    """Run step_2, hide what moves, and return the rest as PNG bytes, with the layout: each axes'
    box and the map's ellipse as fractions of the image (from its top left), and its size in pt."""
    import matplotlib.pyplot as plt
    import stream_convexity_paper.values as values
    from matplotlib.transforms import Bbox

    mpl.use("agg")
    write_values, savefig = values.write_values, plt.savefig
    values.write_values, plt.savefig = (lambda *a, **k: None), (lambda *a, **k: None)
    try:
        runpy.run_path(str(STEP_2), run_name="__main__")
    finally:
        values.write_values, plt.savefig = write_values, savefig
    fig = plt.gcf()
    top, bot, ax_map = fig.axes[:3]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    box = fig.get_tightbbox(r).padded(0.1)  # as savefig(bbox_inches="tight") frames it

    def frac(b):  # display bbox → [left, top, width, height] as fractions of the image
        return [(b.x0 / fig.dpi - box.x0) / box.width, (box.y1 - b.y1 / fig.dpi) / box.height,
                b.width / fig.dpi / box.width, b.height / fig.dpi / box.height]

    def ellipse_px(x, y):
        return ax_map.transData.transform((x, y))
    (xl, yc), (xr, _), (_, yt) = ellipse_px(-np.pi, 0), ellipse_px(np.pi, 0), ellipse_px(0, np.pi / 2)
    layout = {
        "size_pt": [box.width * 72, box.height * 72],
        "panels": {"orbit1": frac(top.get_window_extent(r)), "orbit2": frac(bot.get_window_extent(r))},
        "map": {"cx": ((xl + xr) / 2 / fig.dpi - box.x0) / box.width, "cy": (box.y1 - yc / fig.dpi) / box.height,
                "rx": (xr - xl) / 2 / fig.dpi / box.width, "ry": (yt - yc) / fig.dpi / box.height},
        "paper_lims": {"orbit1": [list(top.get_xlim()), list(top.get_ylim())],
                       "orbit2": [list(bot.get_xlim()), list(bot.get_ylim())]},
    }
    for ax in (top, bot):  # the orbits, the CoM, the boxes (drawn over the orbits), ticks and grid; frame and labels stay
        for a in [*ax.lines, *ax.collections, *ax.texts]:
            a.set_visible(False)
        ax.grid(False)
        ax.tick_params(which="both", colors="none")  # transparent, not removed: the axis labels keep their places
    for a in [*(l for l in ax_map.lines if l.get_marker() == "."), *(t for t in ax_map.texts if t.get_text().startswith("View"))]:
        a.set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches=Bbox([[box.x0, box.y0], [box.x1, box.y1]]))
    plt.close(fig)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8), layout


def main(out: Path) -> Path:
    tri = np.load(paths.DATA / "triaxial" / "triaxial_data.npz")
    arrays, descriptions, orbits = {}, {}, []
    for i, key, title, cmap, rot in ORBITS:
        pos = tri[f"track_q_{key}"]
        arrays |= {
            f"{i}_pos": pos, f"{i}_d1": tri[f"track_dq_{key}"], f"{i}_d2": tri[f"track_d2q_{key}"],
            f"{i}_map": tri[f"nonconcave_frac_{key}"].astype("<f4"), f"{i}_lut": lut(cmap),
        }
        descriptions |= {
            f"{i}_pos": "orbit, potential frame", f"{i}_d1": "d(orbit)/dt", f"{i}_d2": "d²(orbit)/dt²",
            f"{i}_map": "non-concave fraction per viewing angle; rows latitude −90°…90°, columns longitude −180°…180°, 3° cells",
            f"{i}_lut": f"{cmap} colour map, 256 RGB",
        }
        r = np.linalg.norm(pos, axis=1)
        k = key[0]
        orbits.append({
            "id": i, "title": title, "color": mpl.colors.to_hex(mpl.colormaps[cmap](0.5)), "panel_rot_deg": rot,
            "view": [float(tri[f"lon_{k}_deg"]), float(tri[f"lat_{k}_deg"])],
            "paper_fraction": float(tri[f"nonconcave_mask_{key}"].mean()),
            "r_range": [float(r.min()), float(r.max())], "lhat": tri[f"Lhat_{key}"].tolist(),
            "near_dotted": key == "ellip",
        })
    png, layout = paper_layer()
    arrays["paper_layer"] = png
    descriptions["paper_layer"] = "PNG: Paper Figure 3 without what moves with a view (step_2_plot_mollweide.py)"
    meta = {
        "layout": layout,
        "title": "Paper Figure 3: viewing angles of two orbits",
        "paper": PAPER,
        "orbits": orbits,
        "r_near_kpc": R_NEAR_KPC,
        "overlay": {"below": 0.05, "alpha": 0.8},  # step_2: orbit 2's layer hidden at or below 5 %, at 0.8
    }
    units = {f"{o['id']}_pos": "kpc" for o in orbits}
    return figpress_apps.export(out, "orbit-views", arrays, meta=meta, units=units, descriptions=descriptions,
                                licence="MIT", overwrite=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("build/data"))
    print(main(ap.parse_args().out))
