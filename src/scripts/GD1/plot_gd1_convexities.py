r"""GD-1 at the worst viewing angle.

Same projection panel as ``GD_1_worest_result.png`` but with the 3D→2D
curvature variant removed.  Shows the \gamma-coloured track, the 2D-projected
curvature normals, the inward-radial vectors, the bad-normal markers, and
the projected origin (COM).

Output: figures/GD1/gd1_convexities.png
"""

import json
from pathlib import Path

from jaxtyping import Array, Float

import astropy.units as apyu
import interpax as ipx
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import QTable

import unxt as u
from stream_convexity_paper.curvature import (
    count_bad_normals,
    projected_curvature_normal_and_kappa,
)
from stream_convexity_paper.data import rotation_matrix_from_viewing_angles
from stream_convexity_paper.track import Track3D
from stream_convexity_paper.utils import quiver_vec
from stream_convexity_paper.values import write_values
from stream_convexity_paper.variables import PARAMS

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

from stream_convexity_paper import paths

# Track file is the Price-Whelan & Bonaca (2018) GD-1 track shipped with the
# galstreams database (Mateu 2023, MNRAS 520, 5225).
# See: https://github.com/cmateu/galstreams under galstreams/tracks/
DATA_PATH = paths.STATIC / "GD1" / "track.st.GD-1.pricewhelan2018.ecsv"
OUT_PATH = paths.FIGURES / "GD1" / "gd1_convexities.pdf"

# Curve fitting parameters
N_CONTROL = 200  # Control points after subsampling the raw track
N_GAMMA_DENSE = 500  # Dense sample points for smooth curve
#: Analyse gamma in [-GAMMA_HALF_SPAN, +GAMMA_HALF_SPAN] rather than the full
#: [-1, 1], keeping clear of the spline ends.  Every rate below refers to
#: this span, which is shorter than the full track.
GAMMA_HALF_SPAN = 0.95
N_VIEWING_ANGLES = int(PARAMS["gd1"]["n_viewing_angles"]["value"])  # variables.yml
KEY = jax.random.key(0)  # JAX random key for reproducibility

# Plot parameters
FIG_SIZE = (6.0, 4.0)  # Figure dimensions in inches
DPI = 300  # Resolution for output image
#: Minimum convex run length counted as a detection.  Read from the
#: cosmology run rather than re-typed, so the threshold cannot drift away
#: from the ell_min the manuscript quotes.  step_2_plot_triaxiality runs
#: before this script in the figure registry.
ELL_MIN_KPC = float(
    json.loads(
        (
            paths.VALUES / "step_2_plot_triaxiality.values.json"
        ).read_text(encoding="utf-8"),
    )["ell_min_kpc"],
)
#: Tolerance for treating viewing angles as attaining the same maximum.
PCT_TIE_TOL = 1e-6
#: Keep the CoM at least this fraction of the track's on-sky extent clear
#: of the nearest point of the track.
MIN_COM_CLEARANCE = 0.18
#: Accept a projection whose width-to-height ratio is within this factor of
#: the figure frame's own aspect, so the track fills the panel.
ASPECT_TOL = 1.35
N_ARROWS = 25  # Number of vector arrows to display
ARROW_SCALE = 1.0  # Scale factor for arrow lengths
ARROW_ALPHA = 0.5  # Alpha transparency for arrows
TRAJ_LW = 3.0  # Line width for track trajectory
TRAJ_ALPHA = 0.8  # Alpha transparency for track
MARGIN = 0.2  # Margin around data as fraction of range
CURV_COLOR = "#00adef"  # 2D curvature-normal color (TikZ cyan as rendered in Fig. 2)


# ════════════════════════════════════════════════════════════════════════════════
# CURVE CONSTRUCTION: INTERPOLATION AND ARC-LENGTH PARAMETERIZATION
# ════════════════════════════════════════════════════════════════════════════════


def principal_axis(points: Float[Array, "n 3"]) -> Float[Array, "3"]:
    """Compute the first principal axis via SVD of the centred point cloud.

    Args:
        points: Point cloud of shape (n, 3).

    Returns:
        Unit vector along the first principal axis.

    """
    centred = points - points.mean(axis=0)
    _, _, vt = jnp.linalg.svd(centred, full_matrices=False)
    return vt[0]


def chord_param(points: Float[Array, "n 3"]) -> Float[Array, " n"]:
    """Compute normalized cumulative chord length in [0, 1].

    Args:
        points: Point cloud of shape (n, 3).

    Returns:
        Cumulative chord length normalized to [0, 1].

    """
    seg = jnp.linalg.norm(jnp.diff(points, axis=0), axis=-1)
    s = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg)])
    return s / s[-1]


def build_track(
    points: Float[Array, "n 3"],
    *,
    n_control: int = N_CONTROL,
    n_arc: int = 5000,
    n_knots: int = 1000,
) -> tuple[Track3D, float]:
    """Fit a twice-differentiable track through a noisy 3D point cloud.

    1. Sort the cloud along its first principal axis.
    2. Bin-average down to ``n_control`` control points, a low-pass filter.
    3. Spline those in chord parameter and integrate to get arc length.
    4. Resample on a uniform arc-length grid and fit ONE ``cubic2`` spline in
       gamma, which is what the returned track wraps.

    Step 4 matters. Differentiating the composition of a piecewise-LINEAR
    gamma-to-chord map with the chord splines makes d2q/dgamma2 jump at every
    interpolation knot, and the curvature this figure is about would be partly
    an artefact of those jumps. A single cubic2 spline in gamma has continuous
    curvature.

    Args:
        points: Point cloud, shape (n, 3), in kpc.
        n_control: Control points after binning.
        n_arc: Samples used to integrate arc length.
        n_knots: Knots of the final spline.

    Returns:
        The track, parameterised by gamma in [-1, 1], and its total arc length
        in kpc.

    """
    order = jnp.argsort(points @ principal_axis(points))
    sorted_pts = points[order]
    chord = chord_param(sorted_pts)

    # Bin-average to suppress the scatter in the raw track
    bin_idx = jnp.clip((chord * n_control).astype(int), 0, n_control - 1)
    sums = jax.ops.segment_sum(sorted_pts, bin_idx, num_segments=n_control)
    counts = jax.ops.segment_sum(
        jnp.ones(sorted_pts.shape[0]),
        bin_idx,
        num_segments=n_control,
    )
    pts = sums / counts[:, None]
    p = chord_param(pts)

    # Chord-parameter splines, used only to measure arc length and to resample
    sx = ipx.Interpolator1D(p, pts[:, 0], method="cubic2")
    sy = ipx.Interpolator1D(p, pts[:, 1], method="cubic2")
    sz = ipx.Interpolator1D(p, pts[:, 2], method="cubic2")

    p_dense = jnp.linspace(0.0, 1.0, n_arc)
    dense = jnp.stack([sx(p_dense), sy(p_dense), sz(p_dense)], axis=-1)
    ds = jnp.linalg.norm(jnp.diff(dense, axis=0), axis=-1)
    s = jnp.concatenate([jnp.zeros(1), jnp.cumsum(ds)])
    total_length = s[-1]
    gamma_of_p = 2.0 * s / s[-1] - 1.0

    # Resample uniformly in gamma, then fit the single spline the track wraps
    gamma_knots = jnp.linspace(-1.0, 1.0, n_knots)
    p_at_gamma = ipx.interp1d(gamma_knots, gamma_of_p, p_dense, method="linear")
    knots = jnp.stack([sx(p_at_gamma), sy(p_at_gamma), sz(p_at_gamma)], axis=-1)

    ridge = ipx.Interpolator1D(gamma_knots, knots, method="cubic2")
    return Track3D(ridge_line=ridge), float(total_length)


# ════════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS: PROJECTIONS AND TRANSFORMATIONS
# ════════════════════════════════════════════════════════════════════════════════


def projected_curvature_normal(
    tangent_3d: Float[Array, "n 3"],
    curvature_3d: Float[Array, "n 3"],
    R: jax.scipy.spatial.transform.Rotation,
    eps: float = 1e-12,
) -> Float[Array, "n 2"]:
    """Compute unit 2D curvature normal in the rotated sky plane.

    Thin adapter over
    :func:`stream_convexity_paper.curvature.projected_curvature_normal_and_kappa`
    that takes the rotation as a ``Rotation`` object and drops the curvature.

    Args:
        tangent_3d: 3D tangent vectors, shape (n, 3).
        curvature_3d: 3D curvature vectors, shape (n, 3).
        R: 3D rotation transformation.
        eps: Small regularization to avoid division by zero.

    Returns:
        Unit 2D curvature normal in sky plane, shape (n, 2).

    """
    nhat, _ = projected_curvature_normal_and_kappa(
        tangent_3d,
        curvature_3d,
        R.as_matrix(),
        eps=eps,
    )
    return nhat


def uniform_sphere_angles(
    key: jax.Array,
    n: int,
) -> tuple[Float[u.Q, " n"], Float[u.Q, " n"]]:
    """Sample viewing angles uniformly on the sphere.

    Uses the inverse CDF method: cos(θ) ~ U(-1, 1) and φ ~ U(0, 2π).

    Args:
        key: JAX random key for reproducibility.
        n: Number of angle pairs to sample.

    Returns:
        Tuple of (theta, phi) angles in radians with units.

    """
    key_phi, key_u = jax.random.split(key)
    phi = jax.random.uniform(key_phi, (n,), minval=0.0, maxval=2.0 * jnp.pi)
    cos_theta = jax.random.uniform(key_u, (n,), minval=-1.0, maxval=1.0)
    return u.Q(jnp.arccos(cos_theta), "rad"), u.Q(phi, "rad")


# ════════════════════════════════════════════════════════════════════════════════
# WORST-ANGLE SEARCH: FIND VIEWING DIRECTION WITH WORST CONVEXITY
# ════════════════════════════════════════════════════════════════════════════════


def percentage_bad_per_angle(
    positions_3d: Float[Array, "n 3"],
    tangent_3d: Float[Array, "n 3"],
    curvature_3d: Float[Array, "n 3"],
    all_theta: Float[u.Q, " m"],
    all_phi: Float[u.Q, " m"],
) -> Float[Array, " m"]:
    """Compute per-direction fraction of points with bad curvature normal.

    Counts points where the 2D projected curvature normal points outward
    (κ̂_2D · r̂_in < 0) for each viewing angle.

    Args:
        positions_3d: 3D positions of curve points, shape (n, 3).
        tangent_3d: 3D tangent vectors, shape (n, 3).
        curvature_3d: 3D curvature vectors, shape (n, 3).
        all_theta: Polar angles in radians.
        all_phi: Azimuthal angles in radians.

    Returns:
        Fraction (%) of bad points for each viewing angle, shape (m,).

    """
    n_pts = positions_3d.shape[0]

    def _one(theta: Float[u.Q, ""], phi: Float[u.Q, ""]) -> Float[Array, ""]:
        R = rotation_matrix_from_viewing_angles(theta, phi)
        pos_2d = R.apply(positions_3d)[:, :2]
        nhat = projected_curvature_normal(tangent_3d, curvature_3d, R)
        return 100.0 * count_bad_normals(pos_2d, nhat) / n_pts

    return jax.vmap(_one)(all_theta, all_phi)


# ════════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════════════════════════

_tbl = QTable.read(DATA_PATH, format="ascii.ecsv")
_coords = SkyCoord(
    ra=_tbl["ra"],
    dec=_tbl["dec"],
    distance=_tbl["distance"],
    frame="icrs",
).transform_to("galactocentric")
pts_3d: Float[Array, "n 3"] = jnp.stack(
    _coords.cartesian.xyz.to_value(apyu.kpc),
    axis=-1,
)

track, total_length = build_track(pts_3d)
print(f"  Total arc length: {total_length:.3f} kpc")

gamma_dense = jnp.linspace(-GAMMA_HALF_SPAN, GAMMA_HALF_SPAN, N_GAMMA_DENSE)
positions_3d = track.positions(gamma_dense)
tangent_3d = track.tangent(gamma_dense)
curvature_3d = track.curvature(gamma_dense)

print(f"Sampling {N_VIEWING_ANGLES} uniform-sphere viewing angles…")
all_theta, all_phi = uniform_sphere_angles(KEY, N_VIEWING_ANGLES)

print("Computing per-angle bad-normal fraction…")
pct_bad = percentage_bad_per_angle(
    positions_3d,
    tangent_3d,
    curvature_3d,
    all_theta,
    all_phi,
)
# Several viewing directions reach the maximum convex fraction (within
# PCT_TIE_TOL). Among them, keep those with the CoM clear of the track and the
# track filling the landscape frame, and show the one that bends most.
_pct_max = jnp.max(pct_bad)
_tied = jnp.flatnonzero(pct_bad >= _pct_max - PCT_TIE_TOL)


def _illustrative_quality(idx: int) -> tuple[float, float]:
    """Rank one tied viewing direction for use as the illustration.

    Every direction in the tied set is equally correct, so the choice among
    them is a presentation decision. Three things make this figure readable,
    and they are applied as two constraints on a single objective:

    * the track must bend visibly, since a projection that flattens it toward
      a straight line illustrates convexity poorly however correct it is.
      This is the objective, maximised.
    * the centre-of-mass marker must stand clear of the track, or the
      inward-radial arrows collapse into an unreadable fan where they meet it.
    * the track must fill a landscape frame, because the figure is drawn in
      one; a portrait-shaped projection wastes half the panel on white space.

    Returns ``(is_legible, turning_deg)`` so that ``max`` prefers a legible
    view, and the most visibly bent among those.

    One tempting alternative is worth naming so it is not retried: ranking by
    how far the curvature normals turn from the radial direction is
    ANTI-correlated with visible bend (r = -0.54 over the tied set) and selects
    the straightest views in it, and denser sampling does not rescue it since
    its maximum is already saturated.
    """
    rot = rotation_matrix_from_viewing_angles(all_theta[idx], all_phi[idx])
    sky = rot.apply(positions_3d)[:, :2]

    width = jnp.max(sky[:, 0]) - jnp.min(sky[:, 0])
    height = jnp.max(sky[:, 1]) - jnp.min(sky[:, 1])
    extent = jnp.maximum(width, height)
    clearance = jnp.min(jnp.linalg.norm(sky, axis=-1)) / extent
    aspect = width / height

    seg = jnp.diff(sky, axis=0)
    angle = jnp.unwrap(jnp.arctan2(seg[:, 1], seg[:, 0]))
    turning = jnp.sum(jnp.abs(jnp.diff(angle))) * 180.0 / jnp.pi

    frame_aspect = FIG_SIZE[0] / FIG_SIZE[1]
    legible = (
        float(clearance) >= MIN_COM_CLEARANCE
        and frame_aspect / ASPECT_TOL <= float(aspect) <= frame_aspect * ASPECT_TOL
    )
    return (1.0 if legible else 0.0, float(turning))


worst = int(max((int(i) for i in _tied), key=_illustrative_quality))
print(
    f"  {_tied.size} viewing angles tie at {float(_pct_max):.2f}%; "
    f"drawing the clearest of them",
)
theta_w = all_theta[worst]
phi_w = all_phi[worst]
theta_deg = np.degrees(theta_w.ustrip("rad"))
phi_deg = np.degrees(phi_w.ustrip("rad"))
pct_worst = float(pct_bad[worst])
print(
    f"  Worst: idx={worst}  θ={theta_deg:.2f}°  φ={phi_deg:.2f}°  pct={pct_worst:.2f}%",
)

# Values this script owns that the manuscript quotes. Committed alongside the
# figure so `showyourwork build` works without re-running the computation.
# ════════════════════════════════════════════════════════════════════════════════
# ISOTROPIC CONVEXITY RATES
# ════════════════════════════════════════════════════════════════════════════════
# The single most convex view above says nothing about how often convexity is
# visible at all.  These are the same quantities the population forecast uses
# (see sec:cosmology:rates), measured on one real stream over the same
# isotropic directions: the mean convex arc FRACTION, and the fraction of
# directions showing a convex run longer than a threshold.  Both are measured
# in PROJECTED on-sky length, which is what an observer measures.
#
# These are mock-observation rates for the whole track, defined differently
# from lambda_det:
#   1. no on-sky CoM exclusion (lambda_det drops epochs inside r_exclude);
#   2. a plain contiguous run (lambda_det may bridge an inner-disc passage);
#   3. `>= ell_min` (lambda_det uses `> ell_min`).

_R_all = jax.vmap(
    lambda t, p: rotation_matrix_from_viewing_angles(t, p).as_matrix(),
)(all_theta, all_phi)
_sky = jnp.einsum("vij,nj->vni", _R_all, positions_3d)[..., :2]
_v2 = jnp.einsum("vij,nj->vni", _R_all, tangent_3d)[..., :2]
_a2 = jnp.einsum("vij,nj->vni", _R_all, curvature_3d)[..., :2]
_sp2 = jnp.maximum(jnp.sum(_v2 * _v2, axis=-1, keepdims=True), 1e-12)
_perp = _a2 - (jnp.sum(_a2 * _v2, axis=-1, keepdims=True) / _sp2) * _v2
_nhat_all = _perp / jnp.maximum(jnp.linalg.norm(_perp, axis=-1, keepdims=True), 1e-12)
_inward = -_sky / jnp.linalg.norm(_sky, axis=-1, keepdims=True)
_convex = jnp.sum(_nhat_all * _inward, axis=-1) <= 0  # (n_views, n_gamma)

# Lengths are measured ON THE SKY, per viewing direction, because that is what
# an observer sees.  The 3D arc is longer under foreshortening, and using it
# would overstate how often a run clears a projected-length threshold.
_segp = jnp.linalg.norm(jnp.diff(_sky, axis=1), axis=-1)
_ds = jnp.concatenate(
    [_segp[:, :1] / 2, (_segp[:, :-1] + _segp[:, 1:]) / 2, _segp[:, -1:] / 2],
    axis=1,
)
_proj_len = jnp.sum(_ds, axis=1)

_arc_frac = jnp.sum(jnp.where(_convex, _ds, 0.0), axis=1) / _proj_len

# Longest contiguous convex run per view, by a cumulative sum reset at each
# concave sample.  Arc lengths are non-negative, so the running total is
# monotone and the most recent reset is a running maximum.
_cum = jnp.cumsum(jnp.where(_convex, _ds, 0.0), axis=1)
_at_reset = jnp.where(_convex, -jnp.inf, _cum)
_base = jax.lax.cummax(_at_reset, axis=1)
_base = jnp.where(jnp.isneginf(_base), 0.0, _base)
_longest = jnp.max(_cum - _base, axis=1)

FRAC_ANY = float(jnp.mean(_arc_frac > 0.0))
FRAC_DET10 = float(jnp.mean(_longest >= ELL_MIN_KPC))
FRAC_FULL = float(jnp.mean(jnp.all(_convex, axis=1)))
LAMBDA_ARC = float(jnp.mean(_arc_frac))
print(
    f"  isotropic rates: any {100 * FRAC_ANY:.2f}%  "
    f">={ELL_MIN_KPC:.0f}kpc {100 * FRAC_DET10:.2f}%  "
    f"full {100 * FRAC_FULL:.2f}%  lambda_arc {100 * LAMBDA_ARC:.2f}%",
)

write_values(
    __file__,
    {
        "theta_deg": float(theta_deg),
        "phi_deg": float(phi_deg),
        "convex_frac": pct_worst / 100.0,
        # Isotropic rates over the same views, comparable with the
        # population rates of sec:cosmology:rates.
        "frac_any_convex": FRAC_ANY,
        "frac_det_ell_min": FRAC_DET10,
        "frac_fully_convex": FRAC_FULL,
        "lambda_arc": LAMBDA_ARC,
        # Three different lengths, because they are easy to confuse.  The
        # rates above are measured on the analysed span and in projection.
        "arc_length_kpc": float(total_length),
        "arc_length_analysed_kpc": float(total_length * GAMMA_HALF_SPAN),
        "proj_length_median_kpc": float(jnp.median(_proj_len)),
    },
)

# Project to 2D and compute geometry in the worst-angle view
R = rotation_matrix_from_viewing_angles(theta_w, phi_w)
pos_2d = np.asarray(R.apply(positions_3d)[:, :2])
raw_2d = np.asarray(R.apply(pts_3d)[:, :2])
nhat = np.asarray(projected_curvature_normal(tangent_3d, curvature_3d, R))

radial_in = -pos_2d / (np.linalg.norm(pos_2d, axis=-1, keepdims=True) + 1e-10)

# ────────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ────────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

ax.plot(
    pos_2d[:, 0],
    pos_2d[:, 1],
    color="black",
    linewidth=TRAJ_LW,
    alpha=TRAJ_ALPHA,
    zorder=2,
    label="GD-1 track",
)

step = max(1, len(pos_2d) // N_ARROWS)
from_pts = pos_2d[::step]

# 2D projected curvature normal (cyan, matching the schematic figure's TikZ cyan)
quiver_vec(
    from_pts,
    from_pts + nhat[::step] * ARROW_SCALE,
    ax=ax,
    color=CURV_COLOR,
    alpha=ARROW_ALPHA,
    angles="xy",
    scale_units="xy",
    scale=1,
    width=0.005,
)

quiver_vec(
    from_pts,
    from_pts + radial_in[::step] * ARROW_SCALE,
    ax=ax,
    color="gray",
    alpha=ARROW_ALPHA,
    angles="xy",
    scale_units="xy",
    scale=1,
    width=0.005,
)

ax.plot(
    [],
    [],
    color=CURV_COLOR,
    marker=">",
    label="on-sky curvature",
    linewidth=2,
    alpha=ARROW_ALPHA,
)
ax.plot(
    [],
    [],
    color="gray",
    marker=">",
    label="inward radial",
    linewidth=2,
    alpha=ARROW_ALPHA,
)

ax.plot(
    0,
    0,
    marker="o",
    color="black",
    markersize=18,
    linestyle="None",
    zorder=10,
)
ax.plot(
    [],
    [],
    marker="o",
    color="black",
    markersize=8,
    linestyle="None",
    label="Galaxy CoM",
)

xs = np.concatenate([raw_2d[:, 0], pos_2d[:, 0], [0.0]])
ys = np.concatenate([raw_2d[:, 1], pos_2d[:, 1], [0.0]])
x_min, x_max = xs.min(), xs.max()
y_min, y_max = ys.min(), ys.max()

x_pad = MARGIN * (x_max - x_min)
y_pad = MARGIN / 2 * (y_max - y_min)
ax.set_xlim(x_min - x_pad, x_max + x_pad)
ax.set_ylim(y_min - y_pad, y_max + y_pad)

# Equal aspect ratio: 1 kpc x == 1 kpc y in display
ax.set_aspect("equal", adjustable="box")

ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)

ax.text(
    0.025,
    0.04,
    f"θ = {theta_deg:.1f}°\nφ = {phi_deg:.1f}°",
    transform=ax.transAxes,
    va="bottom",
    ha="left",
    fontsize=13,
    bbox={
        "boxstyle": "round",
        "facecolor": "white",
        "edgecolor": "gray",
        "alpha": 0.9,
    },
)

ax.set_xlabel("X' [kpc]", fontsize=14, fontweight="bold")
ax.set_ylabel("Y' [kpc]", fontsize=14, fontweight="bold")
ax.tick_params(labelsize=16)
ax.grid(visible=True, alpha=0.3, linewidth=0.8)
ax.legend(loc="best", fontsize=10)

fig.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight")
print(f"Saved {OUT_PATH}")
