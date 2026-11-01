"""Export Paper Figure 6's dataset: its Z_LR curves and the paper's figure → <out>/likelihood-ratio/.

    uv run --locked --with "figpress-apps==2026.10.29" \
        python webapps/likelihood-ratio/export.py --out build/data

streams_required_pcdm.py is run (its file writes switched off) and everything comes from it: each
curve's Z_LR on the script's own grid of observed rates, with its catalogue size and stroke, and the
figure itself in layers, so the app can draw the curves and switch them on and off: the figure
without the curves and their labels; what the paper draws over the curves (the rate labels, the
model plates); each curve's label; and a legend in the top left, which the paper does not have, in
the paper's style, with where each entry sits. Every layer is the whole figure, so they stack.
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
SCRIPT = ROOT / "src" / "scripts" / "cosmology" / "streams_required_pcdm.py"
PAPER = "https://ui.adsabs.harvard.edu/abs/2026arXiv260940057S/abstract"
CUSTOM = "#d62728"  # the reader's own curve
DPI = 430  # one column: ~1400 px across, sharp at 2× in a 700 px column


def run_script() -> dict:
    """streams_required_pcdm.py's namespace, its figures drawn and unsaved."""
    import stream_convexity_paper.values as values
    from matplotlib.figure import Figure

    mpl.use("agg")
    sys.path.insert(0, str(SCRIPT.parent))
    write_values, savefig = values.write_values, Figure.savefig
    values.write_values, Figure.savefig = (lambda *a, **k: None), (lambda *a, **k: None)
    try:
        return runpy.run_path(str(SCRIPT), run_name="__main__")
    finally:
        values.write_values, Figure.savefig = write_values, savefig


def only(fig, keep):
    """Hide every artist but `keep` (and what they draw); returns a function restoring them."""
    keep_all = {a for k in keep for a in k.findobj()}
    containers = {fig, *fig.axes}
    changed = [a for a in fig.findobj() if a not in keep_all and a not in containers and a.get_visible()]
    for a in changed:
        a.set_visible(False)
    return lambda: [a.set_visible(True) for a in changed]


def main(out: Path) -> Path:
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Bbox

    ns = run_script()
    fig, ax = ns["fig_lr"], ns["ax_lr"]
    specs = ns["_lr_line_specs"]
    lines = [c for c in ax.collections if c.get_zorder() == 3]  # the curves, in the order of specs
    labels = [t for t, _x, _n in ns["_curve_labels"]]
    over = [a for a in ax.get_children() if a.get_zorder() > 3 and a not in labels]  # drawn over the curves
    # The web version's legend: top left, the paper's 7 pt and frame, a line per curve as the paper strokes it.
    handles = [Line2D([], [], color=l.get_color()[0], lw=l.get_linewidth()[0], ls="-" if ls == "-" else (0, (2, 1.2)))
               for l, (_n, ls, *_r) in zip(lines, specs)]
    # The Euclid entry by its N: the paper's "S+26-Euclid-minimum" would reach the CDM plate.
    names = [lbl if ls == "-" else rf"$N={n}$ (Euclid)" for n, ls, *_r, lbl in specs]
    # And a row for the reader's own N, in red: "N =" and room for the app's input field after it.
    handles.append(Line2D([], [], color=CUSTOM, lw=lines[1].get_linewidth()[0]))
    names.append(r"$N=$" + "\u2007" * 9)
    legend = ax.legend(handles, names, loc="upper left", fontsize=7, handlelength=1.6,
                       handletextpad=0.5, borderaxespad=0.4)
    legend.set_zorder(7)
    fig.set_dpi(DPI)  # measured at the dpi the layers are saved at: text spacing differs between dpis
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    box = fig.get_tightbbox(r).padded(0.02)  # as the script's savefig(bbox_inches="tight", pad_inches=0.02)
    b = ax.get_window_extent(r)

    def frac(e):  # display bbox → [left, top, width, height] as fractions of the image
        return [(e.x0 / fig.dpi - box.x0) / box.width, (box.y1 - e.y1 / fig.dpi) / box.height,
                e.width / fig.dpi / box.width, e.height / fig.dpi / box.height]

    # Each legend entry's row, as what a click on it hits: the frame's width, rows meeting halfway.
    rows = [Bbox.union([h.get_window_extent(r), t.get_window_extent(r)]) for h, t in zip(legend.legend_handles, legend.get_texts())]
    frame = legend.get_frame().get_window_extent(r)
    cuts = [frame.y1, *[(a.y0 + b.y1) / 2 for a, b in zip(rows, rows[1:])], frame.y0]
    entries = [frac(Bbox.from_extents(frame.x0, lo, frame.x1, hi)) for hi, lo in zip(cuts, cuts[1:])]
    # The input field: the last row's label after its "N =".
    text = legend.get_texts()[-1].get_window_extent(r)
    probe = ax.text(0, 0, r"$N=$", fontsize=7)
    n_eq = probe.get_window_extent(r).width
    probe.remove()
    field = frac(Bbox.from_extents(text.x0 + n_eq + 0.15 * DPI * 7 / 72, text.y0, text.x1, text.y1))

    def png(keep=None):
        restore = only(fig, keep) if keep is not None else (lambda: None)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, bbox_inches=box, transparent=keep is not None)
        restore()
        return np.frombuffer(buf.getvalue(), dtype=np.uint8)

    x = ns["P_OBS_GRID"]
    arrays = {"rate": x, "over": png(over), "legend": png([legend])}
    descriptions = {"rate": "observed rate K/N, the script's grid", "over": "PNG: what the paper draws over the curves",
                    "legend": "PNG: the web version's legend"}
    curves = []
    for i, ((n, ls, *_r), line, label) in enumerate(zip(specs, lines, labels)):
        arrays[f"z{i}"] = ns["signed_lr_sigma"](x, ns["P_CDM_LR"], ns["P_SIDM_LR"], n)
        arrays[f"label{i}"] = png([label])
        descriptions |= {f"z{i}": f"Z_LR for N = {n}", f"label{i}": f"PNG: the curve label for N = {n}"}
        curves.append({"n": int(n), "euclid": ls == "--", "color": mpl.colors.to_hex(line.get_color()[0]),
                       "lw": float(line.get_linewidth()[0]), "dash": None if ls == "-" else [14, 10],  # in grid segments
                       "entry": entries[i]})
    custom = {"color": CUSTOM, "lw": float(lines[1].get_linewidth()[0]), "entry": entries[-1], "field": field}
    for a in [*lines, *labels, *over, legend]:
        a.set_visible(False)
    arrays["paper"] = png()
    descriptions["paper"] = "PNG: Paper Figure 6 without its curves (streams_required_pcdm.py)"
    meta = {
        "title": "Paper Figure 6: likelihood ratio of CDM and SIDM", "paper": PAPER, "curves": curves, "custom": custom,
        "rates": {"cdm": float(ns["P_CDM_LR"]), "sidm": float(ns["P_SIDM_LR"]), "equal": float(ns["RATE_EQUAL"])},
        "layout": {
            "size_pt": [box.width * 72, box.height * 72],
            # The axes box, [left, top, width, height] as fractions of the image, and its limits.
            "axes": frac(b),
            "xlim": list(map(float, ax.get_xlim())), "ylim": list(map(float, ax.get_ylim())),
        },
    }
    return figpress_apps.export(out, "likelihood-ratio", arrays, meta=meta, descriptions=descriptions,
                                licence="MIT", overwrite=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("build/data"))
    print(main(ap.parse_args().out))
