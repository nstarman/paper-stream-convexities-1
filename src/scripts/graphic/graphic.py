"""Main Graphic: Stellar Stream Curvature Schematic.

Generates a TikZ schematic of a curvilinear stellar stream path around a center
of mass (COM), with:
  - Exact tangent (t_hat) and inward normal (n_hat) vectors at three sample points
  - Right-angle symbols between tangent and normal vectors
  - Decomposition of position vector into tangential and normal components
  - Coordinate axes triad showing observer perspective
  - Green/red shading for concave/convex regions

Uses scipy.interpolate.CubicSpline for smooth curve geometry and a minimal
TikZ helper class for LaTeX code generation.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from stream_convexity_paper import paths


# ════════════════════════════════════════════════════════════════════════════════
# MINIMAL TIKZ PICTURE HELPER CLASS
# ════════════════════════════════════════════════════════════════════════════════


class _PlotCoordinates:
    r"""Helper for rendering \draw plot coordinates in TikZ."""

    def __init__(self, pts: list, options: str = "") -> None:
        self._pts = pts
        self._options = options
        self.plot_options = ""

    def render(self) -> str:
        r"""Generate TikZ \draw plot command."""
        coords = " ".join(f"({x:.4f}, {y:.4f})" for x, y in self._pts)
        opts = self._options
        if self.plot_options:
            opts = f"{self.plot_options}, {opts}" if opts else self.plot_options
        opt_str = f"[{opts}]" if opts else ""
        return rf"\draw{opt_str} plot coordinates {{{coords}}};"


class TikzPicture:
    """Minimal TikZ picture builder (no external tikz library required)."""

    def __init__(self, options: str = "") -> None:
        """Initialize a tikzpicture environment.

        Args:
            options: Global options string for the tikzpicture.

        """
        self._options = options
        self._commands: list = []
        self.extra_files: list = []

    def add_command(self, cmd: str) -> None:
        """Add a raw TikZ command."""
        self._commands.append(cmd)

    def plot_coordinates(self, pts: list, options: str = "") -> _PlotCoordinates:
        """Add a plot coordinates object and return it for further configuration."""
        obj = _PlotCoordinates(pts, options)
        self._commands.append(obj)
        return obj

    def node(self, pos: tuple, options: str = "", text: str = "") -> None:
        """Add a text node at position (x, y)."""
        x, y = pos
        opt = f"[{options}]" if options else ""
        self._commands.append(rf"\node{opt} at ({x:.4f}, {y:.4f}) {{{text}}};")

    def circle(self, pos: tuple, radius: float, options: str = "") -> None:
        """Add a filled/drawn circle at position (x, y) with given radius."""
        x, y = pos
        opt = f"[{options}]" if options else ""
        self._commands.append(rf"\filldraw{opt} ({x:.4f}, {y:.4f}) circle ({radius});")

    def line(self, start: tuple, end: tuple, options: str = "") -> None:
        """Add a line from start to end position."""
        x1, y1 = start
        x2, y2 = end
        opt = f"[{options}]" if options else ""
        self._commands.append(
            rf"\draw{opt} ({x1:.4f}, {y1:.4f}) -- ({x2:.4f}, {y2:.4f});",
        )

    def code(self) -> str:
        """Generate complete TikZ code as a string."""
        lines = [rf"\begin{{tikzpicture}}[{self._options}]"]
        for cmd in self._commands:
            if isinstance(cmd, _PlotCoordinates):
                lines.append(cmd.render())
            else:
                lines.append(str(cmd))
        lines.append(r"\end{tikzpicture}")
        return "\n".join(lines)

    def compile(self, pdf_destination: str, quiet: bool = True) -> None:
        """Compile TikZ code to PDF using pdflatex.

        Args:
            pdf_destination: Output PDF file path.
            quiet: If True, suppress pdflatex output.

        """
        doc = (
            r"\documentclass[tikz,border=2mm]{standalone}"
            "\n"
            r"\usepackage{tikz}"
            "\n"
            r"\usepackage{graphicx}"
            "\n"
            r"\usetikzlibrary{arrows.meta,calc}"
            "\n"
            r"\begin{document}"
            "\n" + self.code() + "\n" + r"\end{document}"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_file = Path(tmpdir) / "figure.tex"
            tex_file.write_text(doc)
            for ef in self.extra_files:
                if Path(ef).exists():
                    shutil.copy(ef, tmpdir)
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "figure.tex"],
                cwd=tmpdir,
                capture_output=quiet,
                check=False,
            )
            pdf_src = Path(tmpdir) / "figure.pdf"
            if pdf_src.exists():
                shutil.copy(pdf_src, pdf_destination)
                print(f"PDF written to {pdf_destination}")
            else:
                print("pdflatex not available or failed; skipping PDF compilation")


# ════════════════════════════════════════════════════════════════════════════════
# SPLINE GEOMETRY: DEFINE THE STREAM CURVE
# ════════════════════════════════════════════════════════════════════════════════

# Waypoints defining the stream path:
#   Left end low, broad arc, gentle inflection just past COM (x=0)
wx = np.array([-2.8, -1.8, -0.6, 0.4, 1.1, 1.8, 2.8])
wy = np.array([-0.3, 0.7, 1.35, 1.35, 1.05, 0.85, 1.2])
cs = CubicSpline(wx, wy, bc_type="natural")

# Dense sample points for smooth curve rendering
x_dense = np.linspace(wx[0], wx[-1], 28)
y_dense = cs(x_dense)
curve_pts = list(zip(x_dense.tolist(), y_dense.tolist(), strict=False))

# X-positions where vector annotations will be placed
sample_xs = [-2.3, -0.2, 2.1]

# Location of inflection point (where y'' = 0)
x_infl = 0.9367
y_infl = cs(x_infl)


# ────────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTION: Compute Curve Tangent and Normal Vectors
# ────────────────────────────────────────────────────────────────────────────────


def get_vectors(x: float) -> tuple:
    """Compute position, tangent, and normal vectors at a given x-coordinate.

    The tangent vector t_hat points rightward (positive x-direction).
    The normal vector n_hat points toward the center of curvature, determined by
    the sign of the second derivative: concave up (y'' > 0) → normal points upward.

    Args:
        x: X-coordinate on the curve.

    Returns:
        Tuple of (pos, t_hat, n_hat) where:
            - pos: [x, y] position on curve
            - t_hat: unit tangent vector pointing rightward
            - n_hat: unit normal vector pointing toward curvature center

    """
    pos = np.array([x, cs(x)])
    dy = cs(x, 1)
    d2y = cs(x, 2)

    t = np.array([1.0, dy])
    t /= np.linalg.norm(t)

    # Two candidate normals: rotate t by ±90°
    n1 = np.array([-t[1], t[0]])  # left of t (points "up-left" for rightward t)
    n2 = np.array([t[1], -t[0]])  # right of t (points "down-right")

    n = n1 if d2y > 0 else n2

    return pos, t, n


# ════════════════════════════════════════════════════════════════════════════════
# TIKZ PICTURE SETUP AND BACKGROUND ELEMENTS
# ════════════════════════════════════════════════════════════════════════════════

# Convert galaxy SVG → PDF for LaTeX inclusion (used as background image)
_galaxy_pdf = paths.DATA / "graphic" / "galaxy.pdf"
_galaxy_pdf.parent.mkdir(parents=True, exist_ok=True)
try:
    subprocess.run(
        [
            "inkscape",
            "--export-type=pdf",
            f"--export-filename={_galaxy_pdf}",
            str(paths.STATIC / "graphic" / "galaxy.svg"),
        ],
        check=False,
        capture_output=True,
    )
except FileNotFoundError:  # no inkscape at all
    pass
if not _galaxy_pdf.exists():  # no inkscape: use the committed conversion
    shutil.copy(paths.STATIC / "graphic" / "galaxy.pdf", _galaxy_pdf)

tikz = TikzPicture(options="scale=2.5")

tikz.add_command(r"""\tikzset{
  dot/.style={circle, fill=black, inner sep=1.8pt},
}""")

# ────────────────────────────────────────────────────────────────────────────────
# CANVAS AND CLIPPING
# ────────────────────────────────────────────────────────────────────────────────

# Define canvas bounds (used for bounding box and clipping)
x_left = -3.2
x_right = 3.5
y_bottom = -1.0
y_top = 2.0

# Set explicit bounding box to prevent TikZ auto-sizing
tikz.add_command(
    rf"\useasboundingbox ({x_left}, {y_bottom}) rectangle ({x_right}, {y_top});",
)

tikz.add_command(
    rf"\clip ({x_left}, {y_bottom}) rectangle ({x_right}, {y_top});",
)

# ────────────────────────────────────────────────────────────────────────────────
# BACKGROUND SHADING: CONCAVE (GREEN) AND CONVEX (RED) REGIONS
# ────────────────────────────────────────────────────────────────────────────────

# Green shading: left side (concave, where y'' > 0) up to inflection point
tikz.add_command(
    rf"\fill[green!8] ({x_left}, {y_bottom}) rectangle ({x_infl:.4f}, {y_top});",
)

# Red shading: right side (convex, where y'' < 0) from inflection point onward
tikz.add_command(
    rf"\fill[red!5] ({x_infl:.4f}, {y_bottom}) rectangle ({x_right}, {y_top});",
)

# ────────────────────────────────────────────────────────────────────────────────
# GALAXY IMAGE (CENTER OF MASS)
# ────────────────────────────────────────────────────────────────────────────────

tikz.add_command(
    r"\node[opacity=0.45, rotate=18, xscale=1.0, yscale=0.8] at (-0.4000, -0.3000) "
    r"{\includegraphics[width=3.5cm]{galaxy}};",
)
tikz.extra_files.append(_galaxy_pdf)

# ────────────────────────────────────────────────────────────────────────────────
# THE STREAM CURVE
# ────────────────────────────────────────────────────────────────────────────────

# Draw the main smooth curve (drawn on top of shading so it's visible)
curve = tikz.plot_coordinates(curve_pts, options="line width=2.5pt")
curve.plot_options = "smooth"

# ────────────────────────────────────────────────────────────────────────────────
# CONCAVE/CONVEX REGION LABELS
# ────────────────────────────────────────────────────────────────────────────────

tikz.node(
    (x_left + 0.1, y_top - 0.05),
    options="below right",
    text=r"{\LARGE {Concave}}",
)
tikz.node(
    (x_right - 0.1, y_top - 0.05),
    options="below left",
    text=r"{\LARGE {Convex}}",
)

# ────────────────────────────────────────────────────────────────────────────────
# STREAM LABEL s_p AND SURROUNDING ARCS
# ────────────────────────────────────────────────────────────────────────────────

# Position and label in the concave section between the first two sample points
_sp_x = -1.0
_sp_y = cs(_sp_x)

# Compute unit normal at label point (points toward center of curvature = upward)
_sp_dy = cs(_sp_x, 1)
_sp_t = np.array([1.0, _sp_dy])
_sp_t /= np.linalg.norm(_sp_t)
_sp_n = np.array([-_sp_t[1], _sp_t[0]])  # left-of-t normal (upward in concave)
_label_offset = 0.20

tikz.node(
    (_sp_x + _label_offset * _sp_n[0], _sp_y + _label_offset * _sp_n[1]),
    options="",
    text=r"{\LARGE $\mathrm{s}_p$}",
)

# Draw curved arcs on either side of the label (constant normal offset)
_arc_half_width = 0.45
_arc_n_pts = 10
for _x_start, _x_end in [
    (_sp_x - _arc_half_width, _sp_x - 0.12),
    (_sp_x + 0.12, _sp_x + _arc_half_width),
]:
    _arc_xs = np.linspace(_x_start, _x_end, _arc_n_pts)
    _arc_pts = []
    for _ax in _arc_xs:
        _ay = cs(_ax)
        _ady = cs(_ax, 1)
        _at = np.array([1.0, _ady])
        _at /= np.linalg.norm(_at)
        _an = np.array([-_at[1], _at[0]])
        _arc_pts.append(
            (_ax + _label_offset * _an[0], _ay + _label_offset * _an[1]),
        )
    _arc_obj = tikz.plot_coordinates(
        _arc_pts,
        options="line width=1.5pt, gray!60, dashed",
    )
    _arc_obj.plot_options = "smooth"


# ════════════════════════════════════════════════════════════════════════════════
# VECTOR ANNOTATIONS AT SAMPLE POINTS
# ════════════════════════════════════════════════════════════════════════════════

# Vector arrow lengths
L = 1.0  # t_hat length
Ln = 0.8  # n_hat length (shorter than t_hat)
for sx in sample_xs:
    pos, t, n = get_vectors(sx)
    px, py = pos[0], pos[1]
    tx, ty = t[0], t[1]
    nx, ny = n[0], n[1]

    # Right-angle symbol (drawn first so arrows render on top)
    s = 0.22
    tikz.add_command(
        rf"\draw[gray, dotted, line width=2pt] "
        rf"({px + s * tx:.4f}, {py + s * ty:.4f}) -- "
        rf"({px + s * tx + s * nx:.4f}, {py + s * ty + s * ny:.4f}) -- "
        rf"({px + s * nx:.4f}, {py + s * ny:.4f});",
    )

    tikz.circle((px, py), 0.07, options="fill=gray")

    # Tangent vector arrow (blue)
    tikz.line(
        (px, py),
        (px + L * tx, py + L * ty),
        options="-{Stealth[scale=1.5]}, line width=2pt, blue",
    )

    # Normal vector arrow (cyan)
    tikz.line(
        (px, py),
        (px + Ln * nx, py + Ln * ny),
        options="-{Stealth[scale=1.5]}, line width=2pt, cyan",
    )

    # Tangent vector label: placed at midpoint, offset perpendicular to vector
    perp_x, perp_y = -ty, tx
    offset = 0.2
    tikz.node(
        (px + 0.5 * L * tx + offset * perp_x, py + 0.5 * L * ty + offset * perp_y),
        options="",
        text=r"{\LARGE $\mathbf{\hat{t}}_p$}",
    )

    # Normal vector label: placed at midpoint, offset perpendicular to vector
    n_perp_x, n_perp_y = -ny, nx
    tikz.node(
        (
            px + 0.5 * Ln * nx + offset * n_perp_x,
            py + 0.5 * Ln * ny + offset * n_perp_y,
        ),
        options="",
        text=r"{\LARGE $\mathbf{\hat{n}}_p$}",
    )


# ════════════════════════════════════════════════════════════════════════════════
# POSITION VECTOR DECOMPOSITION (COM TO CURVE POINT)
# ════════════════════════════════════════════════════════════════════════════════

com = np.array([-0.4, -0.3])

x_target = 1.8
target_pos = np.array([x_target, cs(x_target)])

r_vec = target_pos - com
r_len = np.linalg.norm(r_vec)
r_hat = r_vec / r_len

dy_t = cs(x_target, 1)
d2y_t = cs(x_target, 2)
t_tgt = np.array([1.0, dy_t])
t_tgt /= np.linalg.norm(t_tgt)
n1_t = np.array([-t_tgt[1], t_tgt[0]])
n2_t = np.array([t_tgt[1], -t_tgt[0]])
n_tgt = n1_t if d2y_t > 0 else n2_t

r_t = np.dot(r_hat, t_tgt) * r_len  # physical length along t
r_n = np.dot(r_hat, n_tgt) * r_len  # physical length along n

# Parallelogram vertices for decomposition diagram
C = com + r_t * t_tgt  # endpoint of tangential component
D = com + r_n * n_tgt  # endpoint of normal component

tikz.circle((target_pos[0], target_pos[1]), 0.07, options="fill=gray")

# Main position vector (gray arrow from COM to curve point)
shorten = 0.12
tikz.line(
    (com[0], com[1]),
    (
        target_pos[0] - shorten * r_hat[0],
        target_pos[1] - shorten * r_hat[1],
    ),
    options="-{Stealth[scale=1.5]}, line width=2pt, gray!80",
)

# Decomposition: dashed arrow COM → C (tangential component)
shorten_d = 0.10
t_dir = t_tgt
n_dir = n_tgt

tikz.line(
    (com[0], com[1]),
    (C[0], C[1]),
    options="-{Stealth[scale=1.2]}, line width=1.5pt, gray!80, dashed",
)

# Decomposition: dashed arrow C → target (normal component)
tikz.line(
    (C[0], C[1]),
    (
        target_pos[0] - shorten_d * n_dir[0],
        target_pos[1] - shorten_d * n_dir[1],
    ),
    options="-{Stealth[scale=1.2]}, line width=1.5pt, gray!80, dashed",
)

mid_frac = 0.5
lx = com[0] + mid_frac * r_vec[0]
ly = com[1] + mid_frac * r_vec[1]
perp_rx, perp_ry = r_hat[1], -r_hat[0]
tikz.node(
    (lx - 0.13 * perp_rx, ly + 0.3 * perp_ry),
    options="",
    text=r"{\LARGE $\mathbf{x}_p$}",
)

tc_mx = (com[0] + C[0]) / 2
tc_my = (com[1] + C[1]) / 2
t_angle = np.degrees(np.arctan2(t_tgt[1], t_tgt[0])) - 1
tikz.node(
    (
        tc_mx + 0.1 * t_tgt[0] - 0.25 * t_tgt[1],
        tc_my + 0.1 * t_tgt[1] + 0.1 * t_tgt[0],
    ),
    options=f"rotate={t_angle:.2f}",
    text=(
        r"{\large $\mathbf{x}_{p\perp} = "
        r"(\mathbf{x}_p\cdot\mathbf{\hat{t}}_p)\mathbf{\hat{t}}_p$}"
    ),
)

nc_mx = (C[0] + target_pos[0]) / 2
nc_my = (C[1] + target_pos[1]) / 2
n_angle = np.degrees(np.arctan2(n_tgt[1], n_tgt[0])) - 1
tikz.node(
    (nc_mx - 0.15 * n_tgt[1], nc_my - 0.13),
    options=f"rotate={n_angle:.2f}",
    text=r"{\large $(\mathbf{x}_p\cdot\mathbf{\hat{n}}_p)\mathbf{\hat{n}}_p$}",
)


# ════════════════════════════════════════════════════════════════════════════════
# COORDINATE AXES INSET (OBSERVER PERSPECTIVE)
# ════════════════════════════════════════════════════════════════════════════════

inset_ll = np.array([2.25, -0.97])
inset_ur = np.array([3.47, 0.20])

# Observer position (⊙ symbol)
observer_pos = inset_ll + np.array([0.23, 0.27])

# Galaxy center in the inset (where axes origin is located)
galaxy_center = inset_ll + np.array([0.60, 0.65])

axis_len = 0.4
z_vec = observer_pos - galaxy_center
z_axis_hat = z_vec / np.linalg.norm(z_vec)
z_tip = galaxy_center + axis_len * z_axis_hat

tikz.add_command(
    rf"\filldraw[fill=white, fill opacity=0.88, draw=gray!35, line width=0.5pt, "
    rf"rounded corners=7pt] ({inset_ll[0]:.4f}, {inset_ll[1]:.4f}) "
    rf"rectangle ({inset_ur[0]:.4f}, {inset_ur[1]:.4f});",
)

tikz.add_command(
    rf"\node[opacity=0.55, rotate=18, xscale=1.0, yscale=0.8] "
    rf"at ({galaxy_center[0]:.4f}, {galaxy_center[1]:.4f}) "
    r"{\includegraphics[width=1.5cm]{galaxy}};",
)

tikz.line(
    tuple(galaxy_center),
    tuple(galaxy_center + np.array([axis_len, 0.0])),
    options="-{Stealth[scale=1.1]}, line width=1.1pt, black",
)
tikz.node(
    tuple(galaxy_center + np.array([axis_len + 0.075, 0.0])),
    text=r"{\large $\hat{x}$}",
)

tikz.line(
    tuple(galaxy_center),
    tuple(galaxy_center + np.array([0.0, axis_len])),
    options="-{Stealth[scale=1.1]}, line width=1.1pt, black",
)
tikz.node(
    tuple(galaxy_center + np.array([0.10, axis_len - 0.05])),
    text=r"{\large $\hat{y}$}",
)

# Z-axis (points toward observer)
tikz.line(
    tuple(galaxy_center),
    tuple(z_tip),
    options="-{Stealth[scale=1.1]}, line width=1.1pt, black",
)

tikz.node(
    tuple(observer_pos),
    text=r"{\Large $\odot$}",
)

tikz.node(
    tuple(z_tip + np.array([0.15, 0.0])),
    options="center center",
    text=r"{\large $\hat{z}$}",
)

tikz.node(
    tuple(observer_pos + np.array([0.18, -0.135])),
    options="lower right",
    text=r"{\large observer}",
)


# ════════════════════════════════════════════════════════════════════════════════
# CENTER OF MASS (COM) MARKER (DRAWN LAST SO IT'S ON TOP)
# ════════════════════════════════════════════════════════════════════════════════

tikz.circle((-0.4, -0.3), 0.07, options="fill=black")
tikz.node((-0.4, -0.3), options="left=4pt", text=r"{\LARGE \textbf{COM}}")


# ════════════════════════════════════════════════════════════════════════════════
# COMPILE TIKZ TO PDF
# ════════════════════════════════════════════════════════════════════════════════

tikz_code = tikz.code()
# To src/data, not the figures folder: see the `graphic` rule in the Snakefile.
tikz.compile(pdf_destination=str(paths.DATA / "graphic" / "graphic.pdf"), quiet=True)

# ────────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ────────────────────────────────────────────────────────────────────────────────

print(
    "Done. Generated files:\n"
    "  graphic.pdf           -- main output figure (compiled from TikZ)\n"
    "\n"
    "LaTeX preamble requirements:\n"
    r"  \usepackage{tikz}"
    "\n"
    r"  \usetikzlibrary{arrows.meta, calc}"
    "\n",
)
