"""Integrate orbits and compute sliding-window non-concavity fractions.

For a triaxial NFW potential grid.

Single-pass pipeline: for each (q1, q2) potential on the grid, sample
N_ORBITS random orbits, integrate each one, and for each orbit apply a
2 Gyr sliding window (centred at 0, 0.25, …, 5 Gyr) to test whether any
time step within the window yields a bad curvature normal.

A dwarf-galaxy stream occupies ~2 Gyr of orbital arc.  The Y/N test asks:
"would this stream segment appear non-concave from viewing angle v?"

Edge handling: windows are clipped to [0, T_END] so the centre sits on
either endpoint, giving shorter (but not zero-length) edge windows:

  centre (Gyr):  0    0.5   1.0  ...  4.0   4.5   5.0
  window:       [0,1] [0,1.5] [0,2]  [3,5] [3.5,5] [4,5]

This weights every orbital epoch equally (no middle-time bias).

Inner-disc exclusion: every metric ignores epochs whose *projected* (on-sky)
galactocentric distance is inside R_EXCLUDE_KPC (10 kpc) — that inner region is
not a clean stream-tracing regime.  The exclusion is applied per viewing angle
(a 2-D disc in each projection, not a 3-D sphere), so validity is per (epoch,
view), and excluded pairs are dropped from BOTH the numerator and denominator of
every fraction.

Three per-orbit event-rate metrics are recorded, all derived from the same
per-epoch bad-normal table bad_per_t[t, v] (does viewing angle v see a bad
curvature normal at epoch t?), restricted to non-excluded epochs:

  1. window_bad_frac  — fraction of valid (window, viewing-angle) pairs where
     the 2 Gyr stream segment is non-concave (ANY non-excluded bad epoch in the
     window).  A (window, view) counts in the denominator only if it contains ≥1
     non-excluded epoch for that view.

  2. arclen_bad_frac  — fraction of non-excluded (epoch, viewing-angle) pairs
     that are non-concave, i.e. the non-concave arc-length (time-step) fraction.
     Algebraically jnp.mean(bad_per_t) over non-excluded (epoch, view) pairs.

  3. minlen_bad_frac  — same window Y/N test as metric 1, but a window only
     counts as non-concave if it contains a *contiguous* non-concave run whose
     projected on-sky arc length exceeds a threshold X kpc.  A non-concave run
     may bridge an inner-disc passage (by continuity the convexity persists
     through the unobserved core), but only if the visible convex arc on BOTH
     sides of the passage is ≥ X/2 — equivalently the bridge passes threshold X
     iff 2·min(L_before, L_after) > X, so a tiny convex flick cannot license a
     spurious bridge.  Recorded for X = 1, 2, …, 10 kpc (MINLEN_THRESHOLDS_KPC)
     so any threshold can be plotted without re-running the integration.

Output saved to src/data/cosmology/rates/:
    avg_bad_fracs.npy          (N_POTS, N_ORBITS)      metric 1 window-bad fraction
    arclen_bad_fracs.npy       (N_POTS, N_ORBITS)      metric 2 arc-length fraction
    minlen_bad_fracs.npy       (N_POTS, N_ORBITS, 10)  metric 3, per X threshold
    minlen_thresholds_kpc.npy  (10,)                   the X thresholds [kpc]
    eccentricities.npy         (N_POTS, N_ORBITS)      orbital eccentricities
    projected_flattenings.npy  (N_POTS,)               mean projected axis ratio q̄
    q1_flat.npy, q2_flat.npy   (N_POTS,)               potential-grid parameters
"""

import functools as ft
import pathlib
import sys

from jaxtyping import Array, Bool, Float, PRNGKeyArray
from typing import Any, Final

import coordinax as cx
import diffrax as dfx
import equinox as eqx
import galax.dynamics as gd
import galax.potential as gp
import jax
import jax.random as jr
import numpy as np
from jax_tqdm import scan_tqdm
from jaxmore import vmap
from numpy.typing import NDArray
from plum import convert
from tqdm.auto import trange

import quaxed.numpy as jnp
import unxt as u
from stream_convexity_paper.data import (
    generate_orbit_ic,
    rotation_matrix_from_viewing_angles,
)
from stream_convexity_paper.variables import PARAMS, value_unit

jax.config.update("jax_enable_x64", val=True)

# Monte-Carlo configuration, declared in variables.yml so the manuscript can
# quote it without re-running this (multi-hour) calculation.
_CFG: Final = PARAMS["cosmology_rates"]
# Fiducial host mass and scale radius, shared with the demonstration figure.
_HOST: Final = PARAMS["host_potential"]

# ════════════════════════════════════════════════════════════════════════════════
# TYPE ALIASES (PEP 695) — Named as [Type][Shape] where F=Float, B=Bool, I=Int
# Shape abbreviations: View/Window/Epoch/Orbit/Minlen/Potential
#                     Sz0=scalar, Sz2=2D, Sz3=3D for unnamed shapes
# ════════════════════════════════════════════════════════════════════════════════
# Scalar and basic shapes
type FSz0 = Float[Array, ""]
type FSz3 = Float[Array, "3"]
type FSz2 = Float[Array, "2"]

# Per-dimension data
type FView = Float[Array, "n_views"]
type BView = Bool[Array, "n_views"]
type FWindow = Float[Array, "n_windows"]
type BWindow = Bool[Array, "n_windows"]
type FEpoch3 = Float[Array, "n_epochs 3"]
type FView2 = Float[Array, "n_views 2"]
type FOrbit = Float[Array, "n_orbits"]
type FMinlen = Float[Array, "n_minlen"]

# 2D and higher-order data
type FWindowView = Float[Array, "n_windows n_views"]
type BWindowView = Bool[Array, "n_windows n_views"]
type FOrbitMinlen = Float[Array, "n_orbits n_minlen"]
type FPotentialOrbit = Float[Array, "n_pots n_orbits"]

# Tuple return types
type StepCarry = tuple[
    BWindowView,
    FWindowView,
    FWindowView,
    BWindowView,
    FWindowView,
    FView2,
    FSz0,
    FSz0,
    BWindowView,
]
"""Carry state for the epoch-scan step: (win_any, run, best, prev_convex, left, prev_q2d, bad_total, n_valid_pairs, window_has_valid)."""

type StepInputs = tuple[FSz3, FSz3, FSz3, Array]
"""Input tuple for the epoch-scan step: (q3, dq3, d2q3, mask_col)."""

type OrbitMetrics = tuple[FSz0, FSz0, FMinlen, FSz0]
"""Return type of compute_orbit_stats: (window_bad_frac, arclen_bad_frac, minlen_bad_frac, eccentricity)."""

type BatchMetrics = tuple[
    FOrbit,
    FOrbit,
    FOrbitMinlen,
    FOrbit,
]
"""Vmapped batch results: (window_fracs, arclen_fracs, minlen_fracs, eccentricities)."""

type PotentialScanResult = tuple[
    None,
    tuple[
        FOrbit,
        FOrbit,
        FOrbitMinlen,
        FOrbit,
        FView,
    ],
]
"""Return type from potential grid scan step: (carry=None, results)."""

from stream_convexity_paper import paths
DATA: Final = paths.DATA / "cosmology" / "rates"  # the rule's output (cached on Zenodo)
CHECKPOINTS: Final = paths.DATA / "cosmology" / "rates_checkpoints"  # not cached
DATA.mkdir(parents=True, exist_ok=True)
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

# ── Settings ──────────────────────────────────────────────────────────────────

usys: Final = u.unitsystems.galactic

# Equal-area sphere grid: uniform in (phi, cos theta) → no reweighting needed.
N_PHI: Final = int(_CFG["n_phi"]["value"])  # azimuth bins
N_THETA: Final = int(_CFG["n_theta"]["value"])  # colatitude bins

N_ORBITS: Final = int(_CFG["n_orbits"]["value"])
# Orbits per vmapped batch. CPU benchmark at the full (7200-view, 500-epoch)
# grid: per-orbit time bottoms out near B≈20-25 (~0.127 s/orbit) and *regresses*
# for larger B (B=40→0.168 s, B=100→0.160 s) as the (B, N_WINDOWS, N_VIEWS) scan
# carries spill the CPU cache and become bandwidth-bound.  25 sits at the knee
# and divides N_ORBITS evenly (200 batches); raise only if running where more
# parallelism pays for the memory (e.g. a GPU).
N_ORBITS_BATCH: Final = 25
assert N_ORBITS % N_ORBITS_BATCH == 0, "N_ORBITS_BATCH must divide N_ORBITS"

def _q_range(entry: dict[str, Any]) -> tuple[u.Q, u.Q]:
    """Build a (lo, hi) Quantity pair from a ranged variables.yml entry."""
    lo, hi = entry["value"]
    return u.Q(lo, entry["unit"]), u.Q(hi, entry["unit"])


E_RANGE: Final = tuple(_CFG["e_range"])
R_RANGE: Final = _q_range(_CFG["r_range"])
# Launch-direction azimuth over the FULL circle (0-360 deg), not the 0-90
# default. `inc` here is the angle of the (purely tangential) launch velocity
# within the local tangent plane. Over a fixed integration window the orbit arc
# depends on the actual launch direction, not just the orbital plane (reversing
# the velocity samples a different arc), so only the full 0-360 deg sweep makes
# the velocity direction uniform in the tangent plane. With the
# already-isotropic launch positions this renders the whole orbit ensemble
# isotropic, so the non-concavity rate is *exactly* symmetric under q1↔q2
# (relabelling the y, z axes) — which is what makes the swap-mirror below exact
# rather than approximate.
I_RANGE: Final = _q_range(_CFG["i_range"])

# ── Integration time ──────────────────────────────────────────────────────────

t_end: Final = u.Q(*value_unit(_CFG["t_integrate"]))
t_orbit: Final = u.Q(jnp.array([0.0, t_end.value]), t_end.unit)
T_END_GYR: Final = t_end.ustrip("Gyr")  # 5.0  (plain float for arithmetic)

# ── Fine time grid (used as the discrete epoch set) ──────────────────────────
#
# 1 sample per 10 Myr over the full 5 Gyr orbit → 500 evaluation points.
# This grid is used in two ways:
#   • Step 1: evaluate bad_indicator[t, v] at each of N_T_KAPPA epochs.
#   • Step 2: select which epochs belong to each sliding window via
#             the boolean mask WINDOW_MASKS (precomputed below).

N_T_KAPPA: Final = int(t_end.ustrip("Myr") / 10)  # 500
T_KAPPA_GYR: Final = jnp.linspace(0.0, T_END_GYR, N_T_KAPPA)  # (500,) plain Gyr

# ── Sliding window configuration ──────────────────────────────────────────────
#
# Window half-width: 1 Gyr  →  full window spans 2 Gyr.
# Centre step: 0.25 Gyr  →  21 centres at 0, 0.25, 0.5, …, 5.0 Gyr.
# Each window is clipped to [0, T_END]:
#
#   centre  lo     hi    duration
#   0.00    0.00   1.00  1.00 Gyr   ← centre at left edge
#   0.25    0.00   1.25  1.25 Gyr
#   0.50    0.00   1.50  1.50 Gyr
#   0.75    0.00   1.75  1.75 Gyr
#   1.00    0.00   2.00  2.00 Gyr   ← first full window
#   1.25    0.25   2.25  2.00 Gyr
#   …
#   3.75    2.75   4.75  2.00 Gyr
#   4.00    3.00   5.00  2.00 Gyr   ← last full window
#   4.25    3.25   5.00  1.75 Gyr
#   4.50    3.50   5.00  1.50 Gyr
#   4.75    3.75   5.00  1.25 Gyr
#   5.00    4.00   5.00  1.00 Gyr   ← centre at right edge

WINDOW_HALF_GYR: Final = float(_CFG["window_half"]["value"])  # [Gyr]
WINDOW_STEP_GYR: Final = float(_CFG["window_step"]["value"])  # [Gyr]

_wc_np: NDArray = np.arange(0.0, T_END_GYR + 1e-9, WINDOW_STEP_GYR)  # (21,)
N_WINDOWS: Final = len(_wc_np)  # 21

# WINDOW_MASKS[w, t] = True iff T_KAPPA_GYR[t] falls inside window w.
# Pre-built at Python time as a plain bool array so it is a JAX constant
# (never traced) inside compute_orbit_stats.
# Shape: (N_WINDOWS, N_T_KAPPA) = (21, 500).
_lo = np.maximum(0.0, _wc_np - WINDOW_HALF_GYR)[:, None]  # (21, 1)
_hi = np.minimum(T_END_GYR, _wc_np + WINDOW_HALF_GYR)[:, None]  # (21, 1)
_t = np.asarray(T_KAPPA_GYR)[None, :]  # (1, 500)
WINDOW_MASKS: Final = jnp.array((_t >= _lo) & (_t <= _hi))  # (21, 500) bool

# ── Metric 3: minimum-length thresholds ───────────────────────────────────────
#
# A window counts as non-concave (metric 3) only if it contains a contiguous
# non-concave run whose projected on-sky arc length exceeds the threshold.
# Recorded for every X below so any threshold can be selected at plot time.
MINLEN_THRESHOLDS_KPC: Final = jnp.arange(1.0, 11.0)  # [1, 2, …, 10] kpc, (10,)
N_MINLEN: Final = len(MINLEN_THRESHOLDS_KPC)  # 10

# ── Sphere grid (viewing angles) ──────────────────────────────────────────────

phi_edges = np.linspace(0.0, 2.0 * np.pi, N_PHI + 1)
u_edges = np.linspace(-1.0, 1.0, N_THETA + 1)
phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
theta_centers = np.arccos(0.5 * (u_edges[:-1] + u_edges[1:]))

N_VIEWS: Final = N_PHI * N_THETA  # 7200

_THETA_GRID, _PHI_GRID = np.meshgrid(theta_centers, phi_centers, indexing="ij")
THETA_FLAT: Final = u.Q(jnp.array(_THETA_GRID.ravel()), "rad")
PHI_FLAT: Final = u.Q(jnp.array(_PHI_GRID.ravel()), "rad")  # (N_VIEWS,)

R_FLAT: Final = jax.vmap(rotation_matrix_from_viewing_angles)(THETA_FLAT, PHI_FLAT)

# Rotation matrices (R_MAT_FLAT[v] @ x == R_FLAT[v].apply(x)).  Precomputed so a
# single epoch's 3-D orbit derivatives can be projected into every observer frame
# with a plain matmul — no per-view orbit interpolation.
R_MAT_FLAT: Final = R_FLAT.as_matrix()  # (N_VIEWS, 3, 3)

# Regularisation for the Frenet-Serret normal (matches data.py's eps).
EPS: Final = 1e-12

# ── Inner-disc exclusion ──────────────────────────────────────────────────────
#
# The inner Galaxy is not a clean stream-tracing regime (baryonic disc, rapid
# phase mixing, departures from the NFW form), so every metric ignores epochs
# that fall inside this radius *in projection* (on the sky). Because the
# projected distance depends on the viewing angle, the exclusion is a
# per-(epoch, view) boolean (|q2d| ≥ R_EXCLUDE_KPC), i.e. a 2-D disc in each
# projection rather than a 3-D sphere. Excluded (epoch, view) pairs are dropped
# from BOTH the numerator and denominator of every fraction (they are not "good"
# observations).
R_EXCLUDE_KPC: Final = float(_CFG["r_exclude"]["value"])  # [kpc] on-sky exclusion

# ── Potential grid ────────────────────────────────────────────────────────────

# Non-uniform axis, shared by q1 and q2: 0.03 steps over [0.60, 0.90), 0.02 over
# [0.90, 1.00].  The floor reaches 0.60 so the measured CDM region (c/a 0.65-0.79,
# Auriga MHD interquartile range; Prada+:2019:DarkMatterHalo) is fully sampled
# rather than truncated at 0.70.  The step halves above 0.90 because the SIDM
# region (b/a 0.92-0.98, c/a 0.78-0.90) is narrow: uniform 0.03 spacing would
# sample it with ~8 cells, this grid gives ~20, matching the CDM region's ~20.
# linspace with explicit counts, not arange: arange(0.60, 0.90, 0.03) float-drifts
# and emits a duplicate 0.90, which would desynchronise N_POTS from the unique
# axis values step_2 recovers.
# --ca-floor extends the grid below 0.60 at the same 0.03 spacing, for the flat
# tail of the Auriga sample (c/a down to 0.48, b/a to 0.55) that the default
# grid does not reach.  The paper's grid uses --ca-floor=0.45 (21 x 21; see the
# Snakefile's rate_dependence rule).
_CA_FLOOR: Final = float(
    next(
        (a.split("=", 1)[1] for a in sys.argv if a.startswith("--ca-floor=")),
        0.60,
    ),
)
_STEP: Final = (0.87 - 0.60) / 9
_Q_KNOTS: Final = np.concatenate(
    [
        np.arange(_CA_FLOOR, 0.60 - 1e-9, _STEP) if _CA_FLOOR < 0.60 else np.empty(0),
        np.linspace(0.60, 0.87, 10),
        np.linspace(0.90, 1.00, 6),
    ],
).round(4)

N_Q1: Final = len(_Q_KNOTS)
N_Q2: Final = N_Q1

_q1_vals = _Q_KNOTS
_q2_vals = _Q_KNOTS
_Q1_GRID, _Q2_GRID = np.meshgrid(_q1_vals, _q2_vals)
Q1_FLAT: Final = jnp.array(_Q1_GRID.ravel())  # (N_POTS,)
Q2_FLAT: Final = jnp.array(_Q2_GRID.ravel())  # (N_POTS,)
N_POTS: Final = N_Q1 * N_Q2

# Swap symmetry: non-concavity(q1, q2) == non-concavity(q2, q1) (relabelling the
# y, z axes; with the isotropic 0-360 deg inclination above the equality is
# exact).  Only the lower triangle + diagonal of the grid is therefore
# independent, so we integrate N_UNIQ canonical (q1 ≤ q2) potentials and mirror
# onto the full grid — ~46% fewer integrations.  Requires the q1, q2 axes to
# share sample values.
assert np.allclose(_q1_vals, _q2_vals), "swap symmetry needs identical q1/q2 axes"
N_UNIQ: Final = N_Q1 * (N_Q1 + 1) // 2  # 136 canonical (q1 ≤ q2) potentials
# Checkpoint granularity.  Must divide N_UNIQ, which changes with --ca-floor,
# so pick the divisor nearest the 17 the published grid used.
CHUNK_SIZE: Final = min(
    (d for d in range(8, 40) if N_UNIQ % d == 0),
    key=lambda d: abs(d - 17),
)
assert N_UNIQ % CHUNK_SIZE == 0


@jax.jit
def rotation_to_observer(theta: u.Q, phi: u.Q) -> Array:
    """Rotation matrix whose rows are the observer-frame basis vectors.

    After applying R, the z-axis points along the line-of-sight direction
    n = (sin θ cos φ, sin θ sin φ, cos θ).
    """
    st, ct = jnp.sin(theta), jnp.cos(theta)
    sp, cp = jnp.sin(phi), jnp.cos(phi)
    ex = jnp.stack([cp * ct, sp * ct, -st])
    ey = jnp.stack([-sp, cp, jnp.zeros_like(sp)])
    ez = jnp.stack([st * cp, st * sp, ct])
    return jnp.stack([ex, ey, ez], axis=0).ustrip("")


@ft.partial(vmap, in_axes=(0, 0), in_kw={"q1": None, "q2": None})
def proj_axis_ratio(theta: u.Q, phi: u.Q, *, q1: Array, q2: Array) -> Array:
    """Projected minor/major axis ratio q_proj for one viewing angle.

    Parameters q1 = b/a and q2 = c/a are the intrinsic axis ratios (≤ 1 when
    sorted so that a ≥ b ≥ c).  Uses the Schur-complement of the intrinsic
    quadratic form to obtain the 2-D projected ellipse, then returns
    sqrt(λ_min / λ_max).  The common 1/a² factor cancels in the ratio.
    """
    # Inverse-square shape matrix of the intrinsic ellipsoid (1/a² factor
    # dropped; it cancels in the final eigenvalue ratio).
    S = jnp.diag(jnp.array([1.0, 1.0 / q1**2, 1.0 / q2**2]))
    # Rotation that takes the intrinsic frame to the observer frame.
    R = rotation_to_observer(theta, phi)
    # S expressed in the observer frame: S' = R S Rᵀ.
    Sp = R @ S @ R.T
    # Partition S' into the sky-plane block (A2), the los-coupling vector (bv),
    # and the line-of-sight scalar (cv).
    A2 = Sp[:2, :2]  # (2, 2)  sky-plane block
    bv = Sp[:2, 2]  # (2,)    sky-los coupling
    cv = Sp[2, 2]  # ()      line-of-sight entry
    # Schur complement: projected 2-D ellipse matrix after marginalising over los.
    P = A2 - jnp.outer(bv, bv) / cv
    # Minor/major axis ratio = sqrt of the eigenvalue ratio of P.
    evals = jnp.linalg.eigvalsh(P)  # sorted ascending
    return jnp.sqrt(evals[0] / evals[1])


@jax.jit
def compute_projected_density_axis_ratios(q1: Array, q2: Array) -> Array:
    """Projected axis-ratio vector over the Mollweide grid for a (q1, q2) density.

    Returns
    -------
    q_flat : Array, shape (N_LAT * N_LON,)
        Per-viewing-angle projected minor/major axis ratio.

    """
    axes = jnp.flip(jnp.sort(jnp.array([1.0, q1, q2])))  # descending: [a, b, c]
    b_a, c_a = axes[1] / axes[0], axes[2] / axes[0]
    return proj_axis_ratio(THETA_FLAT, PHI_FLAT, q1=b_a, q2=c_a)


# ── Per-orbit computation ─────────────────────────────────────────────────────

N_BATCHES: Final = N_ORBITS // N_ORBITS_BATCH
keys: Array = jr.split(jr.key(0), N_ORBITS).reshape(N_BATCHES, N_ORBITS_BATCH)

solver = gd.OrbitSolver(stepsize_controller=dfx.PIDController(rtol=1e-07, atol=1e-07))


@eqx.filter_jit
def compute_orbit_stats(
    pot: gp.TriaxialNFWPotential,
    key: PRNGKeyArray,
    /,
) -> OrbitMetrics:
    """Integrate one orbit; return the three event-rate metrics + eccentricity.

    Performance design
    ------------------
    The orbit's 3-D position q(t) and its first two t-derivatives are
    independent of viewing angle; only the sky-plane projection R is per-view,
    and R is linear, so q_sky = R q, q'_sky = R q', q''_sky = R q''.  We
    therefore evaluate the (expensive) dense interpolant ONCE per epoch to get
    (q, q', q'') in 3-D, then project into all N_VIEWS observer frames with
    cheap matmuls — instead of re-differentiating the interpolant inside the
    per-view loop (~N_VIEWSx redundant before).

    All three metrics are accumulated in a single epoch-ordered lax.scan, so the
    per-epoch projected positions are consumed immediately and never stored as a
    big (N_T_KAPPA, N_VIEWS, 2) array; only small per-(window, view) carries
    persist, keeping the batched memory footprint low.

    Inner-disc exclusion
    --------------------
    Epochs whose projected distance |q2d| < R_EXCLUDE_KPC are excluded, per
    viewing angle (a 2-D on-sky disc, not a 3-D sphere), so validity is per
    (epoch, view).  Excluded pairs are dropped from both numerator and
    denominator: the fraction-1/3 denominator counts only valid (window, view)
    pairs (≥1 non-excluded epoch for that view), and the fraction-2 denominator
    counts only non-excluded (epoch, view) pairs.

    Metrics
    -------
    1. window_bad_frac : sliding-window Y/N — ANY non-excluded non-concave epoch
       in window, over valid (window, view) pairs.
    2. arclen_bad_frac : non-concave (epoch x view) fraction = mean(bad_per_t)
       over non-excluded (epoch, view) pairs.
    3. minlen_bad_frac : like 1, but a window counts only if it contains a
       contiguous non-concave run whose projected on-sky arc length exceeds X
       kpc, for every X in MINLEN_THRESHOLDS_KPC.  A run may bridge an inner-
       disc passage only when the visible convex arc on both sides is ≥ X/2;
       since that is equivalent to 2·min(L_before, L_after) > X, the per-(window,
       view) summary `best` = max(longest plain run, 2·min over bridges) is X-
       independent and thresholded against every X at the end.

    Returns
    -------
    window_bad_frac : Array, shape ()
        Metric 1, in [0, 1].
    arclen_bad_frac : Array, shape ()
        Metric 2, in [0, 1].
    minlen_bad_frac : Array, shape (N_MINLEN,)
        Metric 3, one fraction per length threshold, each in [0, 1].
    e : Array, shape ()
        Orbital eccentricity, in [0, 1).

    """
    w0, info = generate_orbit_ic(
        key,
        pot,
        e_range=E_RANGE,
        r_range=R_RANGE,
        i_range=I_RANGE,
    )
    orbit = gd.compute_orbit(pot, w0, t_orbit, dense=True, solver=solver)

    # ── Step 1: per-epoch 3-D position + first two t-derivatives ──────────
    # View-independent, so evaluated once per epoch.  lax.map keeps this
    # sequential (one epoch's interpolant work resident at a time).
    def _q3(s: Array) -> Array:
        w = jax.lax.stop_gradient(orbit)(u.Q(s, "Gyr")).vconvert(cx.CartesianPos3D)
        return convert(w.q, u.Q).ustrip(usys)  # (3,)

    def _derivs(t_gyr: Array) -> tuple[Array, Array, Array]:
        # Nested JVP yields ((q, q'), (q', q'')) in a single pass.
        (q3, dq3), (_, d2q3) = jax.jvp(
            lambda s: jax.jvp(_q3, (s,), (1.0,)),
            (t_gyr,),
            (1.0,),
        )
        return q3, dq3, d2q3

    q3_all, dq3_all, d2q3_all = jax.lax.map(_derivs, T_KAPPA_GYR)
    # shapes: (N_T_KAPPA, 3) each

    # ── Per-epoch, per-view projection + non-concavity test ──────────────
    # Project the shared 3-D derivatives into every observer frame and run the
    # 2-D Frenet-Serret normal test.  Equivalent to the library's per-view JVP
    # (rotation linearity), without re-evaluating the interpolant.
    def _project_views(q3: FSz3, dq3: FSz3, d2q3: FSz3) -> tuple[BView, FView2]:
        def _pv(rmat: Array) -> tuple[Array, Array]:
            q2 = (rmat @ q3)[:2]
            dq2 = (rmat @ dq3)[:2]
            d2q2 = (rmat @ d2q3)[:2]
            speed2 = jnp.maximum(jnp.dot(dq2, dq2), EPS)
            a_perp = d2q2 - (jnp.dot(d2q2, dq2) / speed2) * dq2
            nhat = a_perp / jnp.maximum(jnp.linalg.norm(a_perp), EPS)
            dir_to_center = -q2 / jnp.maximum(jnp.linalg.norm(q2), EPS)
            bad = jnp.sum(nhat * dir_to_center) <= 0.0
            return bad, q2

        return jax.vmap(_pv)(R_MAT_FLAT)  # (N_VIEWS,) bool, (N_VIEWS, 2)

    # ── Step 2: single epoch-scan accumulating all three metrics ─────────
    def _step(carry: StepCarry, xs: StepInputs) -> tuple[StepCarry, None]:
        (
            win_any,  # (N_WINDOWS, N_VIEWS) bool : metric 1, ANY bad-in-window
            run,  # (N_WINDOWS, N_VIEWS)      : current contiguous convex run [kpc]
            best,  # (N_WINDOWS, N_VIEWS)     : metric-3 summary (max of plain run
            #                                   and bridged 2·min over passages)
            prev_convex,  # (N_WINDOWS, N_VIEWS) bool : was prev epoch convex?
            left,  # (N_WINDOWS, N_VIEWS)     : captured L_before for a pending
            #                                   inner-disc bridge [kpc]
            prev_q2d,  # (N_VIEWS, 2)         : previous projected position
            bad_total,  # ()                  : metric-2 numerator
            n_valid_pairs,  # ()             : metric-2 denominator (epoch,view pairs)
            window_has_valid,  # (N_WINDOWS, N_VIEWS) bool : valid (window, view) pairs
        ) = carry
        q3, dq3, d2q3, mask_col = xs  # mask_col: (N_WINDOWS,) windows incl. epoch
        bad, q2d = _project_views(q3, dq3, d2q3)  # (N_VIEWS,), (N_VIEWS, 2)

        # Inner-disc exclusion: per viewing angle, drop epochs whose *projected*
        # (on-sky) galactocentric distance is inside R_EXCLUDE_KPC.  This is a
        # 2-D disc in each projection, so validity depends on the viewing angle.
        valid_row = (  # (1, N_VIEWS) bool : per-view on-sky validity
            jnp.linalg.norm(q2d, axis=-1) >= R_EXCLUDE_KPC
        )[None, :]

        in_win = mask_col[:, None]  # (N_WINDOWS, 1) bool : epoch lies in window w
        bad_row = bad[None, :]  # (1, N_VIEWS) bool
        seg = jnp.linalg.norm(q2d - prev_q2d, axis=-1)[None, :]  # (1, N_VIEWS) [kpc]

        # Per-(window, view) epoch class (all gated to "in this window"):
        #   convex  — non-excluded non-concave point (extends a run)
        #   gap     — excluded (inner-disc) point (transparent; bridge escrow)
        #   break   — non-excluded concave point, or out-of-window (kills a run)
        convex = in_win & valid_row & bad_row  # (N_WINDOWS, N_VIEWS)
        gap = in_win & ~valid_row  # (N_WINDOWS, N_VIEWS)
        brk = (in_win & valid_row & ~bad_row) | ~in_win  # (N_WINDOWS, N_VIEWS)

        # Metric 3 — contiguous convex run length, gaps bridged when the convex
        # arc on BOTH sides is ≥ X/2 (encoded X-independently as 2·min(L, run)).
        #
        # `left` is the run length captured the instant we entered the current
        # inner-disc passage (L_before); it persists across the passage, is
        # overwritten when a fresh run ends into the next passage (chained
        # passages: each L_after becomes the next L_before), and is zeroed by
        # any genuine break.
        left = jnp.where(
            gap,
            jnp.where(prev_convex, run, left),  # capture on entry, hold through gap
            jnp.where(brk, 0.0, left),  # break clears; convex leaves unchanged
        )
        # Plain run grows only on consecutive convex epochs; anything else resets.
        run = jnp.where(convex, jnp.where(prev_convex, run + seg, 0.0), 0.0)
        # `best` takes the larger of the plain run and a licensed bridge value.
        bridge = jnp.where(left > 0.0, 2.0 * jnp.minimum(left, run), 0.0)
        best = jnp.where(convex, jnp.maximum(best, jnp.maximum(run, bridge)), best)

        # Metric 1 — ANY non-excluded non-concave epoch per (window, view).
        win_any = win_any | convex
        # Metric 2 — count non-concave (epoch, view) pairs over valid pairs only.
        bad_total = bad_total + jnp.sum(bad & valid_row[0])
        n_valid_pairs = n_valid_pairs + jnp.sum(valid_row.astype(jnp.float32))
        # Valid (window, view) bookkeeping for the metric-1/3 denominators.
        window_has_valid = window_has_valid | (in_win & valid_row)

        return (
            win_any,
            run,
            best,
            convex,
            left,
            q2d,
            bad_total,
            n_valid_pairs,
            window_has_valid,
        ), None

    zeros_wv = jnp.zeros((N_WINDOWS, N_VIEWS))
    false_wv = jnp.zeros((N_WINDOWS, N_VIEWS), bool)
    init = (
        false_wv,  # win_any
        zeros_wv,  # run length
        zeros_wv,  # best (metric-3 summary)
        false_wv,  # prev_convex
        zeros_wv,  # left (pending-bridge L_before)
        jnp.zeros((N_VIEWS, 2)),  # prev_q2d (unused at t=0: prev_convex is False)
        jnp.array(0.0),  # bad_total
        jnp.array(0.0),  # n_valid_pairs
        false_wv,  # window_has_valid (N_WINDOWS, N_VIEWS)
    )
    (win_any, _, best_run, _, _, _, bad_total, n_valid_pairs, window_has_valid), _ = (
        jax.lax.scan(
            _step,
            init,
            (q3_all, dq3_all, d2q3_all, WINDOW_MASKS.T),
        )
    )

    # ── Reduce to the three metrics ──────────────────────────────────────
    # Denominators drop inner-disc-excluded (epoch, view) pairs: metrics 1/3
    # count only valid (window, view) pairs (≥1 non-excluded epoch for that
    # view), metric 2 counts only non-excluded (epoch, view) pairs.  Each is
    # guarded so an orbit projecting entirely inside R_EXCLUDE_KPC yields 0, not
    # NaN.
    valid_wv = jnp.maximum(jnp.sum(window_has_valid.astype(jnp.float32)), 1.0)

    window_bad_frac = jnp.sum(win_any.astype(jnp.float32)) / valid_wv
    arclen_bad_frac = bad_total / jnp.maximum(n_valid_pairs, 1.0)
    # A (window, view) is non-concave at threshold X iff its (bridge-aware) best
    # run exceeds X.  Windows with no valid epoch never anchor a run, so best=0
    # there and they contribute only to the (excluded) denominator.
    minlen_results = best_run[None] > MINLEN_THRESHOLDS_KPC[:, None, None]
    # shape: (N_MINLEN, N_WINDOWS, N_VIEWS)
    minlen_counts = jnp.sum(minlen_results.astype(jnp.float32), axis=(1, 2))
    minlen_bad_frac = minlen_counts / valid_wv
    # shape: (N_MINLEN,)

    return window_bad_frac, arclen_bad_frac, minlen_bad_frac, info.e


# ── Scan over the potential grid ──────────────────────────────────────────────


@scan_tqdm(CHUNK_SIZE)
def _scan_pots_body(
    carry: None,
    x: tuple[Array, Array, Array],
) -> PotentialScanResult:
    """Outer scan step: integrate N_ORBITS orbits in one (q1, q2) potential."""
    # scan_tqdm prepends an integer step index; unpack and discard it.
    _, q1_i, q2_i = x

    # Build the potential for this grid point. galax potentials are equinox
    # Modules (pytrees), so traced JAX scalars can be passed directly.
    pot_i = gp.TriaxialNFWPotential(
        m=u.Q(*value_unit(_HOST["m_tot"])),
        r_s=u.Q(*value_unit(_HOST["r_s"])),
        q1=q1_i,
        q2=q2_i,
        units="galactic",
    )

    # Inner scan: step through N_BATCHES vmapped batches sequentially so that
    # only N_ORBITS_BATCH orbits are resident in memory at a time.
    def _inner_body(
        carry: None,
        x: tuple[Array, Array],
    ) -> tuple[None, BatchMetrics]:
        # Discard the batch-index (used only to satisfy scan's xs structure).
        _, batch_keys = x
        # vmap over the batch of keys; pot_i is broadcast (not mapped).
        win_b, arclen_b, minlen_b, eccs_b = vmap(
            compute_orbit_stats,
            in_axes=(None, 0),
        )(pot_i, batch_keys)
        # win_b, arclen_b, eccs_b: (N_ORBITS_BATCH,)
        # minlen_b: (N_ORBITS_BATCH, N_MINLEN)
        return carry, (win_b, arclen_b, minlen_b, eccs_b)

    _inner_body = scan_tqdm(N_BATCHES, position=1, leave=False)(_inner_body)

    _, (win_b, arclen_b, minlen_b, eccs_b) = jax.lax.scan(
        _inner_body,
        None,
        (jnp.arange(N_BATCHES), keys),  # keys shape: (N_BATCHES, N_ORBITS_BATCH)
    )
    # Per-viewing-angle projected axis ratios for this potential; shape (N_VIEWS,).
    q_flat_vec_i = compute_projected_density_axis_ratios(q1_i, q2_i)
    # Flatten the N_BATCHES sub-results: scalars → (N_ORBITS,),
    # minlen → (N_ORBITS, N_MINLEN).
    return carry, (
        win_b.ravel(),
        arclen_b.ravel(),
        minlen_b.reshape(N_ORBITS, N_MINLEN),
        eccs_b.ravel(),
        q_flat_vec_i,
    )  # q_flat_vec_i weighted in step_2


# ── Run (chunked with checkpointing), exploiting q1↔q2 swap symmetry ──────────
# Integrate only the N_UNIQ canonical (q1 ≤ q2) potentials, in N_UNIQ/CHUNK_SIZE
# chunks with a per-chunk .npz checkpoint, then mirror the results back onto the
# full N_POTS grid.  The grid point with axis indices (c, r) [q1 = axis[c],
# q2 = axis[r]] takes the canonical result for the sorted pair (min, max).

# Canonical (q1 ≤ q2) potentials: sorted axis-index pairs (a ≤ b).
_uniq_ab: list = [(a, b) for a in range(N_Q1) for b in range(a, N_Q1)]  # len N_UNIQ
uniq_q1: NDArray[np.float64] = np.array([_q1_vals[a] for a, _ in _uniq_ab])
uniq_q2: NDArray[np.float64] = np.array([_q1_vals[b] for _, b in _uniq_ab])
_ab_to_uniq: dict = {ab: i for i, ab in enumerate(_uniq_ab)}
# Full-grid → canonical index.  meshgrid("xy").ravel(): k = r * N_Q1 + c, with
# q1 = _q1_vals[c], q2 = _q2_vals[r]; the canonical pair is (min(c, r), max(c, r)).
inv_index: NDArray[np.int64] = np.array(
    [_ab_to_uniq[(min(c, r), max(c, r))] for r in range(N_Q2) for c in range(N_Q1)],
)  # (N_POTS,)

uq1_chunks = uniq_q1.reshape(-1, CHUNK_SIZE)
uq2_chunks = uniq_q2.reshape(-1, CHUNK_SIZE)
N_CHUNKS: Final = uq1_chunks.shape[0]  # N_UNIQ // CHUNK_SIZE

chunk_bad_fracs: list = []
chunk_arclen_fracs: list = []
chunk_minlen_fracs: list = []
chunk_eccentricities: list = []
chunk_q_flat_vecs: list = []

for i in trange(N_CHUNKS, desc="potential chunks", position=0):
    ckpt = CHECKPOINTS / f"checkpoint_{i:03d}.npz"
    # Only reuse a checkpoint that holds all three metrics, matches the current
    # chunk size, AND was computed for these same potentials.  Without the last
    # check, changing _Q_KNOTS silently reuses rates measured at other shapes
    # whenever the chunk size happens to line up.
    if ckpt.exists():
        _d = np.load(ckpt)
        _same_pots = (
            "q1" in _d.files
            and np.array_equal(_d["q1"], uq1_chunks[i])
            and np.array_equal(_d["q2"], uq2_chunks[i])
        )
        if "minlen_fracs" in _d.files and _d["bad_fracs"].shape[0] == CHUNK_SIZE and _same_pots:
            chunk_bad_fracs.append(_d["bad_fracs"])
            chunk_arclen_fracs.append(_d["arclen_fracs"])
            chunk_minlen_fracs.append(_d["minlen_fracs"])
            chunk_eccentricities.append(_d["eccentricities"])
            chunk_q_flat_vecs.append(_d["q_flat_vecs"])
            continue

    _, (bf, af, mf, ec, qf) = jax.lax.scan(
        _scan_pots_body,
        None,
        (jnp.arange(CHUNK_SIZE), jnp.array(uq1_chunks[i]), jnp.array(uq2_chunks[i])),
    )

    bf_np = np.array(bf)
    af_np = np.array(af)
    mf_np = np.array(mf)
    ec_np = np.array(ec)
    qf_np = np.array(qf)

    np.savez(
        ckpt,
        bad_fracs=bf_np,
        arclen_fracs=af_np,
        minlen_fracs=mf_np,
        eccentricities=ec_np,
        q_flat_vecs=qf_np,
        q1=uq1_chunks[i],
        q2=uq2_chunks[i],
    )

    chunk_bad_fracs.append(bf_np)
    chunk_arclen_fracs.append(af_np)
    chunk_minlen_fracs.append(mf_np)
    chunk_eccentricities.append(ec_np)
    chunk_q_flat_vecs.append(qf_np)

# Stack the canonical results, then mirror onto the full grid via inv_index.
uniq_bad_fracs = np.concat(chunk_bad_fracs, axis=0)  # (N_UNIQ, N_ORBITS)
uniq_arclen_fracs = np.concat(chunk_arclen_fracs, axis=0)  # (N_UNIQ, N_ORBITS)
uniq_minlen_fracs = np.concat(
    chunk_minlen_fracs,
    axis=0,
)  # (N_UNIQ, N_ORBITS, N_MINLEN)
uniq_eccentricities = np.concat(chunk_eccentricities, axis=0)  # (N_UNIQ, N_ORBITS)
uniq_q_flat_vecs = np.concat(chunk_q_flat_vecs, axis=0)  # (N_UNIQ, N_VIEWS)

all_bad_fracs = uniq_bad_fracs[inv_index]  # (N_POTS, N_ORBITS)
all_arclen_fracs = uniq_arclen_fracs[inv_index]  # (N_POTS, N_ORBITS)
all_minlen_fracs = uniq_minlen_fracs[inv_index]  # (N_POTS, N_ORBITS, N_MINLEN)
all_eccentricities = uniq_eccentricities[inv_index]  # (N_POTS, N_ORBITS)
all_q_flat_vecs = uniq_q_flat_vecs[inv_index]  # (N_POTS, N_VIEWS)

# Pixels are equal-area on the sphere, so a plain mean is the solid-angle
# integral of q_proj divided by 4π.
mean_q_flat: NDArray[np.float64] = all_q_flat_vecs.mean(axis=1)  # (N_POTS,)

print(
    f"\nN_pots={N_POTS}  N_orbits={N_ORBITS}  "
    f"mean window_bad_frac={all_bad_fracs.mean():.4f}  "
    f"mean arclen_bad_frac={all_arclen_fracs.mean():.4f}  "
    f"mean minlen_bad_frac (X=1,5,10 kpc)="
    f"[{all_minlen_fracs[..., 0].mean():.4f}, "
    f"{all_minlen_fracs[..., 4].mean():.4f}, "
    f"{all_minlen_fracs[..., 9].mean():.4f}]  "
    f"q range: [{mean_q_flat.min():.3f}, {mean_q_flat.max():.3f}]",
)

np.save(DATA / "avg_bad_fracs.npy", all_bad_fracs)  # metric 1
np.save(DATA / "arclen_bad_fracs.npy", all_arclen_fracs)  # metric 2
np.save(DATA / "minlen_bad_fracs.npy", all_minlen_fracs)  # metric 3 (per X)
np.save(DATA / "minlen_thresholds_kpc.npy", np.array(MINLEN_THRESHOLDS_KPC))
np.save(DATA / "eccentricities.npy", all_eccentricities)
np.save(DATA / "projected_flattenings.npy", mean_q_flat)
np.save(DATA / "q1_flat.npy", np.array(Q1_FLAT))
np.save(DATA / "q2_flat.npy", np.array(Q2_FLAT))
print(f"Saved results to {DATA}")
