"""Export Paper Figure 4's dataset: the two simulated streams and the paper's figure → <out>/stream-views/.

    uv run --locked --with "figpress-apps==2026.10.29" \
        python webapps/stream-views/export.py --out build/data

plot_mollweide.py is run (its savefig switched off) and everything comes from it: each stream's
ridge-line track (position, tangent, curvature) and the particles its sky panel bins (the same
sample), its view, panel limits, density and track styles; and the figure itself. What moves with a
view is hidden — the panels' densities, tracks, centres, boxes, ticks and grid, and the map's two
markers — and the rest kept as an image, the panels' title boxes as a second image drawn over the
panels, and each marker with its label as a sprite, with where each sits, so the app draws only
what moves.
"""

import argparse
import io
import runpy
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np

import figpress_apps

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src" / "scripts" / "streams" / "plot_mollweide.py"
PAPER = "https://ui.adsabs.harvard.edu/abs/2026arXiv260940057S/abstract"
DPI = 160  # the static layers: ~2400 px across, sharp at 2× in a 1200 px column
N_GAMMA, GAMMA_RANGE = 500, (-0.95, 0.95)  # ridgeline_sky_stats' defaults
MAX_PARTICLES, RNG_SEED = 100_000, 42  # plot_sky_density_panel's sample
# id, the script's view, density and track styles, badge, info box corner, flip (the map shows the
# particle-spray stream mirrored about the equator, and its panel's b with it)
STREAMS = [("spray", "A", "lower left", True), ("nbody", "B", "lower right", False)]


def run_script() -> dict:
    """plot_mollweide.py's namespace, its figure open and unsaved."""
    from matplotlib.figure import Figure

    mpl.use("agg")
    sys.path.insert(0, str(SCRIPT.parent))
    savefig = Figure.savefig
    Figure.savefig = lambda *a, **k: None
    try:
        return runpy.run_path(str(SCRIPT), run_name="__main__")
    finally:
        Figure.savefig = savefig


def png(fig, box, **kw) -> np.ndarray:
    from matplotlib.transforms import Bbox

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches=Bbox([[box.x0, box.y0], [box.x1, box.y1]]), **kw)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


def only(fig, keep):
    """Hide every artist but `keep` (and what they draw); returns a function restoring them."""
    keep_all = {a for k in keep for a in k.findobj()}
    containers = {fig, *fig.axes}
    changed = [a for a in fig.findobj() if a not in keep_all and a not in containers and a.get_visible()]
    for a in changed:
        a.set_visible(False)
    return lambda: [a.set_visible(True) for a in changed]


def lut(cmap) -> np.ndarray:
    """A colour map as 256 RGBA bytes."""
    cm = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    return np.rint(255 * cm(np.linspace(0, 1, 256))).astype(np.uint8)


def main(out: Path) -> Path:
    jax.config.update("jax_enable_x64", True)
    ns = run_script()
    from matplotlib.transforms import Bbox
    from ridgeline_data import ridgeline_sky_stats  # the script's directory, on the path from run_script
    fig, top, bot, ax_map = ns["fig"], ns["ax_top"], ns["ax_bot"], ns["ax_map"]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    box = fig.get_tightbbox(r).padded(0.1)  # as savefig(bbox_inches="tight") frames it

    def frac(b):  # display bbox → [left, top, width, height] as fractions of the image
        return [(b.x0 / fig.dpi - box.x0) / box.width, (box.y1 - b.y1 / fig.dpi) / box.height,
                b.width / fig.dpi / box.width, b.height / fig.dpi / box.height]

    def at(xy):  # display point → fractions of the image
        return [(xy[0] / fig.dpi - box.x0) / box.width, (box.y1 - xy[1] / fig.dpi) / box.height]

    (xl, yc), (xr, _), (_, yt) = (ax_map.transData.transform(p) for p in ((-np.pi, 0), (np.pi, 0), (0, np.pi / 2)))
    gamma = jnp.linspace(*GAMMA_RANGE, N_GAMMA)
    arrays, descriptions, streams = {}, {}, []
    markers = [l for l in ax_map.lines if l.get_marker() in ("*", "P")]
    labels = [t for t in ax_map.texts if t.get_text() in (ns["MARKER_A"].text_template, ns["MARKER_B"].text_template)]
    for (i, key, corner, flip), ax, marker, label in zip(STREAMS, (top, bot), markers, labels):
        view, dens, track = ns[f"VIEW_{key}"], ns[f"DENSITY_{key}"], ns[f"TRACK_{key}"]
        parts = np.asarray(view.case.particles)
        if len(parts) > MAX_PARTICLES:
            rng = np.random.default_rng(RNG_SEED)
            parts = parts[np.sort(rng.choice(len(parts), MAX_PARTICLES, replace=False))]
        arrays |= {
            f"{i}_pos": np.asarray(view.case.track.positions(gamma)),
            f"{i}_d1": np.asarray(jax.vmap(view.case.track.tangent)(gamma)),
            f"{i}_d2": np.asarray(jax.vmap(view.case.track.curvature)(gamma)),
            f"{i}_particles": parts.astype("<f4"),
            f"{i}_density_lut": lut(dens.cmap),
        }
        descriptions |= {
            f"{i}_pos": "ridge-line track, stream frame", f"{i}_d1": "d(track)/dγ",
            f"{i}_d2": "Frenet curvature vector of the track", f"{i}_particles": "the particles the paper's panel bins",
            f"{i}_density_lut": "the panel's density colour map, 256 RGBA",
        }
        stats = ridgeline_sky_stats(view.case.track, view.theta, view.phi)
        lon, lat = np.degrees((view.phi + np.pi) % (2 * np.pi) - np.pi), 90 - np.degrees(view.theta)
        # The marker's sprite: it and its label, alone, on a transparent ground; its anchor the marker.
        restore = only(fig, [marker, label])
        b = Bbox.union([marker.get_window_extent(r), label.get_tightbbox(r)]).padded(3)
        sbox = Bbox.from_extents(b.x0 / fig.dpi, b.y0 / fig.dpi, b.x1 / fig.dpi, b.y1 / fig.dpi)
        sprite = png(fig, sbox, transparent=True)
        restore()
        mx, my = ax_map.transData.transform((np.radians(lon), np.radians(-lat if flip else lat)))
        arrays[f"{i}_marker"] = sprite
        descriptions[f"{i}_marker"] = "PNG: the map's marker for this view, with its label"
        streams.append({
            "id": i, "flip": flip, "view": [float(lon), float(-lat if flip else lat)],
            "paper_fraction": float(stats["pct_bad"]) / 100,
            "lims": [list(view.xlim), list(view.ylim)], "info": corner,
            "density": {"cmin": dens.cmin, "alpha": dens.alpha, "vmax_percentile": dens.vmax_percentile,
                        "bin_width": dens.adaptive_bin_width_kpc, "bins": dens.bins,
                        "log": dens.use_lognorm, "gamma": dens.norm_gamma},
            "track": {"good": mpl.colors.to_hex(track.track_color_good), "bad": mpl.colors.to_hex(track.track_color_bad),
                      "lw_good": track.track_lw_good, "lw_bad": track.track_lw_bad,
                      "alpha_good": track.track_alpha_good, "alpha_bad": track.track_alpha_bad},
            "panel": frac(ax.get_window_extent(r)),
            # The sprite: its box, and the marker within it, as fractions of the figure image.
            "marker": {"box": [(sbox.x0 - box.x0) / box.width, (box.y1 - sbox.y1) / box.height,
                               sbox.width / box.width, sbox.height / box.height], "at": at((mx, my))},
        })

    fs = ns["fstyle"]
    layout = {
        "size_pt": [box.width * 72, box.height * 72],
        "map": {"cx": ((xl + xr) / 2 / fig.dpi - box.x0) / box.width, "cy": (box.y1 - yc / fig.dpi) / box.height,
                "rx": (xr - xl) / 2 / fig.dpi / box.width, "ry": (yt - yc) / fig.dpi / box.height},
        "fonts": {"tick": fs.tick_label_fontsize, "info": fs.info_fontsize},
    }
    # Over the panels: their title boxes alone (zorder 20 in the paper, above the tracks).
    badges = [ax.texts[0] for ax in (top, bot)]
    restore = only(fig, badges)
    arrays["overlay"] = png(fig, box, transparent=True)
    restore()
    # Under: the figure without what moves. Ticks transparent, not removed, so the labels keep their places.
    for ax in (top, bot):
        for a in [*ax.collections, *ax.lines, *ax.texts, *ax.images]:
            a.set_visible(False)
        ax.grid(False)
        ax.tick_params(which="both", colors="none")
    for a in [*markers, *labels]:
        a.set_visible(False)
    arrays["paper_layer"] = png(fig, box)
    descriptions |= {"paper_layer": "PNG: Paper Figure 4 without what moves with a view (plot_mollweide.py)",
                     "overlay": "PNG: the panels' title boxes, drawn over the panels"}
    meta = {"title": "Paper Figure 4: viewing angles of two simulated streams", "paper": PAPER,
            "streams": streams, "layout": layout}
    units = {f"{s['id']}_{k}": "kpc" for s in streams for k in ("pos", "particles")}
    return figpress_apps.export(out, "stream-views", arrays, meta=meta, units=units, descriptions=descriptions,
                                licence="MIT", overwrite=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("build/data"))
    print(main(ap.parse_args().out))
