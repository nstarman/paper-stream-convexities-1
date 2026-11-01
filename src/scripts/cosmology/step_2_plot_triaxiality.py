"""Diagnostic plot: non-concave event-rate metrics vs triaxiality.

Loads arrays saved by step_1_rate_dependence.py and produces a two-panel figure:
  Left:  mean metric-3 (min-length >= 10 kpc) non-concave rate on a q1xq2 grid,
         with SIDM/CDM regions annotated.
  Right: per-orbit quantile bands of metric 3 (>= 10 kpc) vs mean projected
         flattening, with the metric 1 (window Y/N) and metric 2 (arc-length
         fraction) mean curves overlaid.
"""

import csv
import pathlib
import sys

from typing import Any, Final

import jax
import jax.numpy as jnp
import matplotlib

# The layout below measures text in pixels, and some offsets do not scale with
# the DPI, so the tight bbox depends on it. Pin Agg at 200 dpi (what a Retina
# Mac's default backend gives) so every machine, CI included, saves the same size.
matplotlib.use("agg")

import matplotlib.patheffects as mpatheffects
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from matplotlib.path import Path as _MplPath
from matplotlib.text import Text
from numpy.typing import NDArray
from scipy.optimize import linprog

from stream_convexity_paper.values import write_values

# Two versions of this figure, from one code path so they cannot drift: the
# in-text figure drops detail that the appendix version shows in full.
SIMPLE: Final = "--simple" in sys.argv

from stream_convexity_paper import paths
DATA: Final = paths.DATA / "cosmology" / "rates"


# ── Load ──────────────────────────────────────────────────────────────────────

# metric 1 (N_POTS, N_ORBITS)
all_bad_fracs: NDArray = np.load(DATA / "avg_bad_fracs.npy")
# metric 2 (N_POTS, N_ORBITS)
arclen_fracs: NDArray = np.load(
    DATA / "arclen_bad_fracs.npy",
)
# metric 3 (N_POTS, N_ORBITS, N_MINLEN)
minlen_fracs: NDArray = np.load(
    DATA / "minlen_bad_fracs.npy",
)
minlen_thresholds: NDArray = np.load(
    DATA / "minlen_thresholds_kpc.npy",
)  # (N_MINLEN,)
Q_FLAT: NDArray = np.load(DATA / "projected_flattenings.npy")  # (N_POTS,)
q1_flat: NDArray = np.load(DATA / "q1_flat.npy")  # (N_POTS,)
q2_flat: NDArray = np.load(DATA / "q2_flat.npy")  # (N_POTS,)

N_POTS, N_ORBITS = all_bad_fracs.shape

# Metric 3 at the 10 kpc threshold is the "headline" metric: it colours the left
# heatmap and supplies the quantile bands on the right panel.
X_SELECT_KPC: Final = 10.0
_ix_select = int(np.argmin(np.abs(minlen_thresholds - X_SELECT_KPC)))
m3_fracs: np.ndarray = minlen_fracs[:, :, _ix_select]  # (N_POTS, N_ORBITS)

# ── Derived quantities ────────────────────────────────────────────────────────

# Sort potentials by projected flattening for clean fill_between bands.
sort_idx = np.argsort(Q_FLAT)
q_sorted = Q_FLAT[sort_idx]  # (N_POTS,)

m1_sorted = all_bad_fracs[sort_idx]  # metric 1, sorted by q̄
m2_sorted = arclen_fracs[sort_idx]  # metric 2, sorted
m3_sorted = m3_fracs[sort_idx]  # metric 3 (10 kpc), sorted

# Right panel: smooth quantile bands of the headline metric (M3) are fit below
# (after the envelope machinery) by quantile regression on the pooled per-orbit
# cloud.  LEVELS sets the percentile band edges; LEGEND_LEVELS is the subset the
# figure actually draws, which the simplified version narrows to 1 sigma.
LEVELS = [0.05, 0.16, 0.25, 0.50, 0.75, 0.84, 0.95]  # 16/84 = 1 sigma
LEGEND_LEVELS = [0.16, 0.50, 0.84] if SIMPLE else LEVELS
# The band key's handle is sized from its row count, so a band is the same
# height in both figures however many levels each one shows.  The median row
# is a rule, not a band, so it takes a fraction of a band's height.
MEDIAN_ROW_W: Final = 0.62
BAND_ROW_H: Final = 2.27
BAND_HANDLE_H: Final = BAND_ROW_H * (len(LEGEND_LEVELS) - 1 + MEDIAN_ROW_W)

# Per-metric mean curves vs q̄.
mean_m1 = m1_sorted.mean(axis=1)  # (N_POTS,)
mean_m2 = m2_sorted.mean(axis=1)  # (N_POTS,)
# ... and the same metric in potential order, for the region integral below.
mean_m2_pot = arclen_fracs.mean(axis=1)  # (N_POTS,), in potential-grid order
mean_m3 = m3_sorted.mean(axis=1)  # (N_POTS,)


def fit_monotone_envelope(
    qbar: jax.Array,
    rates: jax.Array,
    quantile: jax.Array,
    *,
    degree: int = 3,
) -> jax.Array:
    r"""Fit a monotone rate(q̄) curve, anchored at (q̄=1, 0), to a data quantile.

    Model, with :math:`x \equiv 1 - \bar q \ge 0`:

    .. math::  \mathrm{rate}(x) = \sum_{k=1}^{D} c_k\, x^k, \quad c_k \ge 0,

    where ``D = degree``.

    Dropping the constant term pins ``rate(q̄=1) = 0`` exactly, and ``c_k >= 0``
    makes the curve monotonically increasing as q̄ falls.  The coefficients
    minimise the pinball (quantile) loss, so ``quantile`` -> 1 gives an upper
    envelope and ``quantile`` -> 0 a lower one.

    The pinball loss is piecewise linear and convex in ``c``, and the only
    constraint is ``c >= 0``, so this is a small linear program and is solved
    exactly.  A smooth-objective optimiser is the wrong tool: the loss is
    non-differentiable exactly at the optimum's active set.

    Introducing ``u_i >= rho_q(y_i - B_i c)`` as epigraph variables,

        minimise    mean(u)
        subject to  -q B c - u <= -q y,  (1-q) B c - u <= (1-q) y,  c >= 0.

    Returns the nonnegative coefficients ``c_k``, shape ``(degree,)`` -- evaluate
    with :func:`envelope_curve`.
    """
    x = np.clip(1.0 - np.asarray(qbar), 0.0, None)
    basis = x[:, None] ** np.arange(1, degree + 1)  # (N, degree)
    y = np.asarray(rates)
    q = float(quantile)
    n = len(y)

    eye = np.eye(n)
    a_ub = np.block([[-q * basis, -eye], [(1.0 - q) * basis, -eye]])
    b_ub = np.concatenate([-q * y, (1.0 - q) * y])
    cost = np.concatenate([np.zeros(degree), np.full(n, 1.0 / n)])

    res = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=(0.0, None), method="highs")
    if not res.success:  # pragma: no cover - would mean a malformed problem
        msg = f"envelope LP failed at quantile {q}: {res.message}"
        raise RuntimeError(msg)
    return jnp.asarray(res.x[:degree])


def envelope_curve(coeffs: jax.Array, qbar: jax.Array) -> jax.Array:
    """Evaluate a :func:`fit_monotone_envelope` fit at the given q̄."""
    x = jnp.clip(1.0 - qbar, 0.0, None)
    return (x[:, None] ** jnp.arange(1, coeffs.shape[0] + 1)) @ coeffs


# Fit monotone lower/upper envelopes (each anchored at (q̄=1, 0)) to the spread of
# per-potential means for every metric, then sample on a fine q̄ grid for plotting.
# Fill between each metric's envelope pair.  The key's fill is only a few
# points tall, so it needs more alpha than the panel's to read at print size.
ENV_FILL_ALPHA: Final = 0.12
ENV_KEY_FILL_ALPHA: Final = 0.32
ENV_LO_Q: Final = 0.05
ENV_HI_Q: Final = 0.95
q_env = np.linspace(float(q_sorted.min()), 1.0, 200)
_qbar_j = jnp.asarray(q_sorted)
_q_env_j = jnp.asarray(q_env)


def _metric_envelopes(rates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lower/upper monotone-anchored envelope curves on `q_env` for one metric."""
    rates_j = jnp.asarray(rates)
    lo_c = fit_monotone_envelope(_qbar_j, rates_j, jnp.asarray(ENV_LO_Q))
    hi_c = fit_monotone_envelope(_qbar_j, rates_j, jnp.asarray(ENV_HI_Q))
    lo = np.asarray(envelope_curve(lo_c, _q_env_j))
    hi = np.asarray(envelope_curve(hi_c, _q_env_j))
    return lo, hi


m3_lo, m3_hi = _metric_envelopes(mean_m3)
m1_lo, m1_hi = _metric_envelopes(mean_m1)
m2_lo, m2_hi = _metric_envelopes(mean_m2)

# Smooth M3 quantile bands.  The raw per-potential np.quantile band edges spike
# because q̄→shape is many-to-one (adjacent-in-q̄ potentials are different shapes)
# and each potential has only N_ORBITS draws.  We keep the within-potential band
# width: estimate every level per potential, then fit a smooth monotone-anchored
# curve (median regression, the same machinery as the envelopes) through those
# estimates — averaging over the q̄→shape scatter and orbit noise.  Sort across
# levels afterwards to forbid quantile crossing.
_q_levels = np.quantile(m3_sorted, LEVELS, axis=1)  # (len(LEVELS), N_POTS)
# One small LP per level; the fit is a scipy solve, so it is not vmappable.
_band_coeffs = jnp.stack(
    [fit_monotone_envelope(_qbar_j, jnp.asarray(yv), jnp.asarray(0.5)) for yv in _q_levels],
)
_bands = np.sort(
    np.asarray(jax.vmap(lambda c: envelope_curve(c, _q_env_j))(_band_coeffs)),
    axis=0,
)  # (len(LEVELS), len(q_env)), ascending in level
q05, q16, q25, q50, q75, q84, q95 = _bands
# The colored median curve is the 50% band level itself, so it cannot disagree
# with the ladder it is drawn inside.
m3_mid = q50

# Left panel: recover unique axis values and reshape the headline metric (M3 @
# 10 kpc) to a (N_Q1, N_Q2) grid.
q1_vals = np.unique(q1_flat)
q2_vals = np.unique(q2_flat)
# mean_m3 is in q̄-sorted order (it feeds the right panel); q1_flat/q2_flat are
# not.  Reshape the UNSORTED per-potential mean, or every cell gets the rate of
# whichever potential happens to share its rank in q̄.
mean_m3_pot = m3_fracs.mean(axis=1)  # (N_POTS,), in potential-grid order
bad_grid = mean_m3_pot.reshape(len(q1_vals), len(q2_vals))
# lambda(q1, q2) = lambda(q2, q1) exactly: the grid is the swap-mirror of the
# N_UNIQ canonical potentials, so any ordering slip shows up as asymmetry.
assert np.abs(bad_grid - bad_grid.T).max() == 0.0, "bad_grid is not in grid order"

# Literature-motivated regions, as boxes in (b/a, c/a).  With x the major axis,
# b/a = max(q1,q2) and c/a = min(q1,q2).  The polygons drawn on the left panel,
# the strips on the right panel, and the region-mean masks are all derived from
# these, so that the figure and the quoted eta_region_mean always describe the
# same region.
# CDM: measured.  Interquartile range of the 30 Milky-Way-mass Auriga MHD
# (with-baryons) haloes of Prada+:2019:DarkMatterHalo, pooled over R200/8, R200/4
# and R200/2 (29-115 kpc), which brackets the stream band.  Their shape profiles
# are flat over that range, so the band edges barely matter.
# CDM: the interquartile range of the Auriga MHD haloes in
# literature/prada2019_auriga_mhd.csv, pooled over R200/8, R200/4 and
# R200/2 (~26-131 kpc, 30 haloes at each, n = 90).
# The Auriga haloes themselves, pooled over R200/8, R200/4, R200/2 (n = 90).
# The box below is their per-axis interquartile range; the rates are integrated
# over the full sample, not read at the box.
_AURIGA = paths.STATIC / "cosmology" / "literature" / "prada2019_auriga_mhd.csv"
_rows = list(
    csv.DictReader(
        line for line in _AURIGA.read_text().splitlines() if not line.startswith("#")
    ),
)
AURIGA_BA: Final = np.array(
    [float(r[f"ba_R200o{k}"]) for k in (8, 4, 2) for r in _rows],
)
AURIGA_CA: Final = np.array(
    [float(r[f"ca_R200o{k}"]) for k in (8, 4, 2) for r in _rows],
)

# The two panels cannot use the same coverage and both stay readable.  A box is
# a rectangle in (b/a, c/a); a strip is a percentile of qbar, which averages
# over both axes.  The b/a marginal has wide tails that qbar largely cancels,
# so at 5-95% the CDM box spans b/a 0.72-0.99 and swallows the panel while its
# strip stays tight.  Boxes therefore show the interquartile range and strips
# the 5-95%, and the caption says which is which.
#
# The box is a JOINT interquartile region: it holds half the haloes.  Taking
# the 25-75% range on each axis separately does not do that -- it holds only
# ~23% of them -- so the per-axis fraction is solved for instead.
BOX_JOINT_COVERAGE: Final = 0.50
# Wash of each region's own colour.  Applied to the FACE colour, not as the
# patch's alpha: a patch-level alpha fades the dotted edge with the fill and
# the box stops reading as a boundary.
REGION_FILL_ALPHA: Final = 0.13
BAND_PCTL: Final = (5.0, 95.0)


def _solve_box_fraction(
    x: np.ndarray,
    y: np.ndarray,
    target: float = BOX_JOINT_COVERAGE,
) -> float:
    """Central per-axis fraction whose marginal box encloses `target` of the sample.

    1/sqrt(target) is the independent-axes answer and is wrong here: b/a and
    c/a correlate at ~+0.5 in Auriga, so 1/sqrt(2) per axis encloses 60% rather
    than 50%.  Bisect on the measured joint fraction instead.  Monotone maps of
    each axis (the SIDM relaxation is one) preserve ranks, so the same fraction
    serves both populations.
    """
    lo, hi = 0.05, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        xl, xh = np.percentile(x, [50 * (1 - mid), 50 * (1 + mid)])
        yl, yh = np.percentile(y, [50 * (1 - mid), 50 * (1 + mid)])
        inside = np.mean((x >= xl) & (x <= xh) & (y >= yl) & (y <= yh))
        if inside < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


_BOX_F: Final = _solve_box_fraction(AURIGA_BA, AURIGA_CA)
BOX_PCTL: Final = (50.0 * (1.0 - _BOX_F), 50.0 * (1.0 + _BOX_F))


def _pctl_box(vals: np.ndarray, knots: np.ndarray) -> tuple[float, float]:
    """BOX_PCTL range of `vals`, clipped to the shapes the grid samples."""
    lo, hi = np.percentile(vals, BOX_PCTL).round(2)
    return float(max(lo, knots.min())), float(min(hi, knots.max()))


_BA_KNOTS = np.unique(np.maximum(q1_flat, q2_flat))
_CA_KNOTS = np.unique(np.minimum(q1_flat, q2_flat))
CDM_BA_MIN, CDM_BA_MAX = _pctl_box(AURIGA_BA, _BA_KNOTS)
CDM_CA_MIN, CDM_CA_MAX = _pctl_box(AURIGA_CA, _CA_KNOTS)

# SIDM: derived, for the sigma/m >~ a few cm^2/g regime of sec:cosmology:microphysics.
# No hydrodynamic, Milky-Way-mass, multi-halo suite exists above sigma/m = 1, so
# the region is extrapolated from three anchors:
#   1. Giocoli+:2026:AIDATNGProject3D Fig. 3 gives the fractional c/a offset over
#      CDM for sigma/m = 1: +17% at 0.15 R200 falling to +6% at 0.5 R200 with
#      baryons, against +36% -> +17% dark-matter-only.  The ratio of those two is
#      a steady factor ~2.4 across the band: the baryon suppression of the signal.
#   2. Chua+:2021:ImpactInelasticSelfinteracting simulates a Milky-Way-mass halo
#      whose elastic cross-section is "a few cm^2 g^-1", and gives a ~25% larger
#      fractional offset than sigma/m = 1 inside 0.2 R200.  It is dark-matter-only
#      and a single halo, so it sets the scaling, not the normalisation.
#   3. Applying (2) then dividing by (1) gives a with-baryons offset of +19% at
#      0.15 R200 falling to +9% at 0.5 R200.  On the Auriga CDM median of 0.73
#      that is c/a 0.87 -> 0.80 across 30-100 kpc, i.e. a band mean of ~0.83.
#
# The c/a CEILING is measured rather than extrapolated: Despali+:2022 Fig. 4
# reaches c/a ~ 0.92 in the dark matter of the with-baryons SIDM run for 2 of its
# 5 haloes, at sigma/m = 1.  Since this region is scoped to sigma/m >~ a few, a
# ceiling below that measured value would be hard to defend.  The transfer is not
# free: those haloes are 1e13 Msun early types and the 0.92 occurs at r ~ 15-40
# kpc, inside this band in scaled radius, so it leans on the same r_1(sigma/m)
# argument as the rest of the region.
#
# The c/a FLOOR is the weakest edge: the band-mean median of ~0.83 less an
# interquartile half-width of ~0.06.  Propagating the Auriga CDM interquartile
# range multiplicatively instead would put it at 0.74, which is more faithful but
# admits enough flat haloes to lower qbar.  Flagged rather than hidden.
#
# b/a is not taken from the Giocoli+ Table 2 b/a-c/a locus: that locus runs ~0.035
# low against Auriga's own measured b/a, and mixing it with the Auriga-measured
# CDM box would compress the b/a gap between the regions artificially.  It is
# instead the interquartile range of the Auriga haloes that actually populate
# c/a over this region's range, which is [0.920, 0.960] (n = 31).  Note that
# extending it to b/a = 1 would LOWER qbar, not raise it: at fixed c/a an oblate
# halo shows its full c/a flattening along every equatorial sightline, whereas a
# triaxial one shows at most c/b there, so qbar falls as b/a -> 1.
# Self-interaction relaxes each halo toward the shape set by the local
# potential, so it is a pull toward an attractor, not a fixed fractional boost:
# a halo already near c/a = C_ATTRACTOR barely moves, a flat one moves far.  A
# multiplicative offset has this backwards, and needs a hard clip at the ceiling
# that this form does not.
#   C_ATTRACTOR  Scattering isotropizes the dispersion, so the endpoint is the
#                shape of the potential the dark matter sits in.  Over 26-105 kpc
#                the disc contributes little to that potential, leaving a sphere.
#                Despali+:2022 Fig. 4 measures c/a ~ 0.92 in the with-baryons
#                SIDM run; that is a well-relaxed halo at sigma/m = 1, not an
#                asymptote, and taking it as the endpoint makes it unreachable
#                (no halo passes its own attractor).
#   CA_OFFSET    Fractional c/a offset of the band median, for sigma/m >~ a few.
#                Giocoli+:2026 Fig. 6 gives the MW-mass with-baryons offset at
#                sigma/m = 1 directly (+12.3% at R200/10).  Their Fig. 3 gives
#                the baryon suppression of that signal, 2.47 at 0.1 R200 falling
#                to 2.16 over the 0.22-0.37 R200 band that r_1(sigma/m >~ few)
#                covers.  Taking the band to saturate at the dark-matter-only
#                peak, scaled to MW mass, gives 35.2% / 2.16 = 16.3%.
# RELAXATION is then fixed by the median, not tuned: no free parameter is left.
C_ATTRACTOR: Final = 1.0
CA_OFFSET: Final = 0.163
_ca_med: Final = float(np.median(AURIGA_CA))
RELAXATION: Final = _ca_med * CA_OFFSET / (C_ATTRACTOR - _ca_med)

# A convex combination of the halo's shape and the attractor's, so it cannot
# overshoot: no clip is needed, and none would ever fire.  b/a relaxes the same
# way toward the same shape, which for a sphere is 1 on both axes.
BA_ATTRACTOR: Final = 1.0
SIDM_CA: Final = AURIGA_CA + RELAXATION * (C_ATTRACTOR - AURIGA_CA)
SIDM_BA: Final = AURIGA_BA + RELAXATION * (BA_ATTRACTOR - AURIGA_BA)

SIDM_BA_MIN, SIDM_BA_MAX = _pctl_box(SIDM_BA, _BA_KNOTS)
SIDM_CA_MIN, SIDM_CA_MAX = _pctl_box(SIDM_CA, _CA_KNOTS)

_q1, _q2 = q1_flat, q2_flat
_ba, _ca = np.maximum(_q1, _q2), np.minimum(_q1, _q2)
sidm_mask: np.ndarray = (
    (_ba >= SIDM_BA_MIN)
    & (_ba <= SIDM_BA_MAX)
    & (_ca >= SIDM_CA_MIN)
    & (_ca <= SIDM_CA_MAX)
)
cdm_mask: np.ndarray = (
    (_ba >= CDM_BA_MIN) & (_ba <= CDM_BA_MAX) & (_ca >= CDM_CA_MIN) & (_ca <= CDM_CA_MAX)
)

# The regions live here but the (q1, q2) grid they are sampled on is set in
# step_1_rate_dependence.py.  An empty mask means the two have drifted apart.
for _name, _m in (("SIDM", sidm_mask), ("CDM", cdm_mask)):
    if not _m.any():
        msg = (
            f"{_name} region selects no grid potential; the box in this file and "
            f"the _Q_KNOTS grid in step_1_rate_dependence.py have diverged"
        )
        raise ValueError(msg)

# Margin past the CDM band edge on the right panel's x-axis.
XLIM_PAD: Final = 0.015


# Each halo carries its own projected flattening: qbar is read from the grid
# potential nearest its (b/a, c/a).  Rates are then averaged over the haloes,
# which is the integral over P_M(S) the framework asks for.  Reading one rate at
# one representative qbar instead discards the spread, and lambda(qbar) is steep
# and convex near the spherical limit, so the flat tail carries the rate.
_GRID_BA, _GRID_CA = np.maximum(q1_flat, q2_flat), np.minimum(q1_flat, q2_flat)
_QBAR_VIEWS: Final = 40_000
_QBAR_SEED: Final = 7


def qbar_of(ba: np.ndarray, ca: np.ndarray) -> np.ndarray:
    """Viewing-averaged projected flattening of the ellipsoid (1, b/a, c/a).

    Purely geometric, so it needs no rate grid.  Along each sightline the
    projected isodensity contour is an ellipse whose axis ratio follows from
    the 2x2 restriction of diag(1, (b/a)^2, (c/a)^2) to the plane
    perpendicular to it, and qbar is the mean of that ratio over isotropic
    sightlines.

    The restriction is the projected SECOND-MOMENT ellipse, which is the
    standard projected-isodensity shape and the quantity the grid measures.
    It is not the silhouette of the solid ellipsoid: that would subtract
    (P^T m u)(u^T m P) / (u^T m u), and runs rounder.

    Computing it directly rather than snapping to the nearest sampled shape
    matters for the flat tail of the Auriga sample, which falls off the grid.
    """
    rng = np.random.default_rng(_QBAR_SEED)
    u = rng.normal(size=(_QBAR_VIEWS, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    e1 = np.cross(u, [0.0, 0.0, 1.0])
    _n = np.linalg.norm(e1, axis=1)
    _degenerate = _n < 1e-8
    e1[_degenerate] = np.cross(u[_degenerate], [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(u, e1)
    out = np.empty(len(np.atleast_1d(ba)))
    for i, (b, c) in enumerate(zip(np.atleast_1d(ba), np.atleast_1d(ca), strict=True)):
        m = np.diag([1.0, b * b, c * c])
        s11 = np.einsum("ij,jk,ik->i", e1, m, e1)
        s22 = np.einsum("ij,jk,ik->i", e2, m, e2)
        s12 = np.einsum("ij,jk,ik->i", e1, m, e2)
        half_tr = 0.5 * (s11 + s22)
        disc = np.sqrt(np.maximum(half_tr**2 - (s11 * s22 - s12**2), 0.0))
        out[i] = np.sqrt(np.maximum(half_tr - disc, 0.0) / (half_tr + disc)).mean()
    return out


# The grid's own Monte-Carlo qbar is the reference; disagreement means the
# geometry here and the potentials the rates were measured in have drifted.
_chk = np.random.default_rng(0).choice(len(Q_FLAT), 40, replace=False)
assert np.abs(qbar_of(_GRID_BA[_chk], _GRID_CA[_chk]) - Q_FLAT[_chk]).max() < 2e-3, (
    "analytic qbar disagrees with the grid"
)


def rate_at(qbar: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """A fitted rate curve, extended linearly below the range the grid spans.

    The flattest Auriga haloes sit below q_env, where np.interp would clamp and
    silently assign them the edge rate.  lambda(qbar) is convex there, so a
    linear extension is a floor on their true rate, not a flattering guess.
    """
    slope = (curve[1] - curve[0]) / (q_env[1] - q_env[0])
    below = np.minimum(np.asarray(qbar) - q_env[0], 0.0)
    return np.interp(qbar, q_env, curve) + slope * below


QBAR_CDM_DIST: Final = qbar_of(AURIGA_BA, AURIGA_CA)
QBAR_SIDM_DIST: Final = qbar_of(SIDM_BA, SIDM_CA)
# Population means, for the dot-dashed reference lines and the quoted rates.
QBAR_SIDM: Final = float(QBAR_SIDM_DIST.mean())
QBAR_CDM: Final = float(QBAR_CDM_DIST.mean())

SIDM_Q: Final = tuple(np.percentile(QBAR_SIDM_DIST, BAND_PCTL))
CDM_Q: Final = tuple(np.percentile(QBAR_CDM_DIST, BAND_PCTL))

# Colormap scale for the heatmap + scatter (headline metric M3 @ 10 kpc).
CMAP: Final = "magma"
VMIN: Final = 0.0
# The right panel's y-axis has to reach the top of the ideal-rate envelope, but
# no potential is coloured above `bad_grid.max()`, so the colour scale stops
# there: the ramp then spends its whole range on rates that occur.  The ribbon
# beside the axis keeps that axis's scale and simply ends at the same value.
RATE_MAX: Final = 0.18
VMAX: Final = float(bad_grid.max())

SIDM_COLOR: Final = "#7DCEA0"
CDM_COLOR: Final = "deepskyblue"

# ── Legend handler ───────────────────────────────────────────────────────────


class BandedQuantileLegendHandler(HandlerBase):
    """Custom legend handler that renders quantile bands as stacked rectangles."""

    def create_artists(
        self,
        legend: Any,  # noqa: ARG002
        orig_handle: Any,  # noqa: ARG002
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: mtransforms.Transform,
    ) -> list[Any]:
        # Percentile increases UPWARD, matching the panel: the 84% band is drawn
        # above the median, the 16% below.  Every row is a quantile of the
        # per-orbit rate within a potential; the dashed envelope is a quantile
        # across potentials, so it is keyed separately and stays out of this
        # ladder.  Built from LEGEND_LEVELS so the key cannot disagree with the
        # bands drawn.
        color = "gray"
        _lv = LEGEND_LEVELS
        _n = len(_lv)
        _depth = [min(i, _n - 1 - i) for i in range(_n)]
        rows: list[tuple[str, str, float]] = []
        for v, d in zip(reversed(_lv), reversed(_depth), strict=True):
            if abs(v - 0.50) < 1e-9:
                rows.append(("line", "50%", 0.6))
            else:
                alpha = 0.30 if SIMPLE else (0.20, 0.40, 0.60)[min(d, 2)]
                rows.append(("band", f"{round(100 * v):>2d}%", alpha))

        weights = [MEDIAN_ROW_W if kind == "line" else 1.0 for kind, _, _ in rows]
        unit = height / sum(weights)
        artists: list[Any] = []

        y_top = height - ydescent
        for (kind, label, alpha), w in zip(rows, weights, strict=True):
            band_height = w * unit
            y0 = y_top - band_height
            y_top = y0
            if kind == "line":
                # The 50% level is the colored median curve, so its key carries the
                # same gradient, over the range of rates the curve actually spans.
                _x = np.linspace(xdescent, xdescent + width, 64)
                _pts = np.column_stack([_x, np.full_like(_x, y0 + band_height / 2)])
                _lc = LineCollection(
                    np.concatenate([_pts[:-1, None], _pts[1:, None]], axis=1),
                    cmap=CMAP,
                    norm=plt.Normalize(VMIN, VMAX),
                    linewidth=3,
                    transform=trans,
                )
                _lc.set_array(np.linspace(m3_mid.min(), m3_mid.max(), len(_x) - 1))
                artists.append(_lc)
            else:
                artists.append(
                    Rectangle(
                        (xdescent, y0),
                        width,
                        band_height,
                        facecolor=color,
                        alpha=alpha,
                        edgecolor="none",
                        transform=trans,
                    ),
                )

            if label:
                artists.append(
                    Text(
                        x=xdescent + width + fontsize * 0.3,
                        y=y0 + band_height / 2,
                        text=label,
                        va="center",
                        fontsize=fontsize * 0.75,
                        transform=trans,
                        fontfamily="monospace",
                    ),
                )

        return artists


_banded_quantile_handle = object()


class EnvelopePairHandler(HandlerBase):
    """Render a metric's key as the two lines of its envelope, as drawn."""

    def create_artists(
        self,
        legend: Any,  # noqa: ARG002
        orig_handle: Any,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,  # noqa: ARG002
        trans: mtransforms.Transform,
    ) -> list[Any]:
        y_hi, y_lo = height * 0.72 - ydescent, height * 0.28 - ydescent
        color = orig_handle.get_color()
        return [
            Rectangle(
                (xdescent, y_lo),
                width,
                y_hi - y_lo,
                facecolor=color,
                alpha=ENV_KEY_FILL_ALPHA,
                edgecolor="none",
                transform=trans,
            ),
            *(
                Line2D(
                    [xdescent, xdescent + width],
                    [y] * 2,
                    color=color,
                    linestyle=orig_handle.get_linestyle(),
                    linewidth=orig_handle.get_linewidth(),
                    transform=trans,
                )
                for y in (y_hi, y_lo)
            ),
        ]


# ── Combined figure ───────────────────────────────────────────────────────────

fig = plt.figure(figsize=(13, 5), dpi=200)
# wspace holds only the left panel's b/a ticks and label now that the ribbon has
# moved outside the left panel.  At 0.20 they touch the right panel's frame.
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.26)

# Left panel
ax2 = fig.add_subplot(gs[0, 0])

# Right panel: a single axes over the full 0-18% range, which compresses the
# sub-1% end against the axis.
ax_top = fig.add_subplot(gs[0, 1])
ax = ax_top

# ── Panel 1: quantile bands ───────────────────────────────────────────────────


def _plot_quantile_panel(ax_panel):
    """Plot quantile bands and metrics on the given axes."""
    COLOR = "gray"
    # Quantile bands of the headline metric (M3, min-length >= 10 kpc).
    if SIMPLE:
        ax_panel.fill_between(q_env, q16, q84, alpha=0.30, color=COLOR, zorder=2)
    else:
        ax_panel.fill_between(q_env, q05, q95, alpha=0.10, color=COLOR, zorder=2)
        ax_panel.fill_between(q_env, q16, q84, alpha=0.20, color=COLOR, zorder=2)
        ax_panel.fill_between(q_env, q25, q75, alpha=0.40, color=COLOR, zorder=2)
    # The 50% level is drawn below as the colored median line, not as a gray one.
    # Two dashed envelopes bracket the per-potential means at each q̄.  Their spread
    # is physical: q̄ fixes only the mean projected elongation, while the rate also
    # depends on the intrinsic shape (b/a, c/a), so a single mean curve would hide a
    # real band (an oblate halo is more non-concave than a prolate one of equal q̄).
    ax_panel.fill_between(
        q_env,
        m3_lo,
        m3_hi,
        color="#333333",
        alpha=ENV_FILL_ALPHA,
        lw=0,
        zorder=3,
    )
    ax_panel.plot(q_env, m3_lo, color="#333333", lw=2.0, ls="--", zorder=4)
    (m3_artist,) = ax_panel.plot(
        q_env,
        m3_hi,
        color="#333333",
        lw=2.0,
        ls="--",
        # Matches \lamdadet{10} in ms.tex: \lambda_{\rm det}^{(10)}.
        label=r"$\lambda_{\mathrm{det}}^{(10)}$",
        zorder=4,
    )
    # The 50% level of the orbit-to-orbit ladder, colored along its length by the
    # same scale as the left panel's heatmap, so the color ties the two panels
    # together.
    _seg_xy = np.column_stack([q_env, m3_mid]).reshape(-1, 1, 2)
    _median_line = LineCollection(
        np.concatenate([_seg_xy[:-1], _seg_xy[1:]], axis=1),
        cmap=CMAP,
        norm=plt.Normalize(VMIN, VMAX),
        linewidth=3.2,
        zorder=5,
    )
    _median_line.set_array(0.5 * (m3_mid[:-1] + m3_mid[1:]))
    ax_panel.add_collection(_median_line)
    # Metric 1 (window Y/N) and metric 2 (arc-length fraction) envelopes overlaid,
    # fitted the same way as M3 (solid here to distinguish from M3's dashed pair),
    # with a light fill between each pair.
    if not SIMPLE:
        ax_panel.fill_between(
            q_env,
            m1_lo,
            m1_hi,
            color="#8B1538",
            alpha=ENV_FILL_ALPHA,
            lw=0,
            zorder=3,
        )
    ax_panel.fill_between(
        q_env,
        m2_lo,
        m2_hi,
        color="#8B0000",
        alpha=ENV_FILL_ALPHA,
        lw=0,
        zorder=3,
    )
    m1_artist = None
    if not SIMPLE:
        ax_panel.plot(q_env, m1_lo, color="#8B1538", lw=1.8, ls=":", zorder=4)
        (m1_artist,) = ax_panel.plot(
            q_env,
            m1_hi,
            color="#8B1538",
            lw=1.8,
            ls=":",
            zorder=4,
            # Matches \lamdadet{0} in ms.tex: \lambda_{\rm det}^{(0)}.
            label=r"$\lambda_{\mathrm{det}}^{(0)}$",
        )
    ax_panel.plot(q_env, m2_lo, color="#8B0000", lw=1.8, zorder=4)
    (m2_artist,) = ax_panel.plot(
        q_env,
        m2_hi,
        color="#8B0000",
        lw=1.8,
        zorder=4,
        label=r"$\lambda_{\mathrm{arc}}$",
    )

    # Vertical reference lines at fixed q̄ values.
    for _xref in (QBAR_SIDM, QBAR_CDM):
        ax_panel.axvline(_xref, color="black", lw=0.9, ls="-.", zorder=2.5)

    # Major grid only.  The minor y grid put a rule between every labelled
    # tick, which read as clutter against the bands in both variants.
    ax_panel.grid(visible=True, which="major", color="0.85", linewidth=0.6, zorder=0)
    ax_panel.set_axisbelow(True)
    ax_panel.invert_xaxis()
    # Run past the widest band's low-qbar edge rather than stopping on it, so
    # the edge is visible instead of coinciding with the panel border.  Follows
    # what is DRAWN: the distributions have flat tails far past their bands, and
    # chasing those leaves most of the panel empty.
    ax_panel.set_xlim(
        left=1.001,
        right=max(q_env[0], min(CDM_Q[0], SIDM_Q[0]) - XLIM_PAD),
    )
    ax_panel.tick_params(labelsize=13)

    return m1_artist, m3_artist, m2_artist


# Plot on both axes (they'll show different y-ranges)
m1_artist, m3_artist, m2_artist = _plot_quantile_panel(ax_top)

ax_top.set_ylim(VMIN, RATE_MAX)
ax_top.set_xlabel(
    r"Mean projected flattening $\boldsymbol{\bar{q}}$",
    fontsize=14,
    fontweight="bold",
)
ax_top.yaxis.tick_right()
ax_top.yaxis.set_label_position("right")
ax_top.set_ylabel("Convexity rate (%)", fontsize=14, fontweight="bold", labelpad=10)
ax_top.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.3g}"))
# Percent sign lives in the axis label, not on every tick.
ax_top.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{100 * v:.0f}"))
ax_top.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
ax_top.minorticks_on()

# Quantile-band legend: separate box with a tall handle so all 7 band levels
# are readable.  Added as an extra artist so the mean gets its own legend.
banded_legend = ax_top.legend(
    handles=[_banded_quantile_handle],
    labels=[""],
    handler_map={_banded_quantile_handle: BandedQuantileLegendHandler()},
    title="$\\lambda_{\\mathrm{det}}^{(10)}$ orbit-to-\norbit scatter",
    title_fontsize=8,
    # Anchored below the metric legend after both are measured, just below.
    bbox_to_anchor=(0.0, 0.0),
    loc="upper left",
    bbox_transform=ax_top.transAxes,
    fontsize=9,
    handlelength=4,
    handleheight=BAND_HANDLE_H,
    handletextpad=2,
    borderpad=0.4,
)
ax_top.add_artist(banded_legend)

# Legend for the three metric mean curves (separate from the band legend).
_metric_handles = [a for a in (m1_artist, m3_artist, m2_artist) if a is not None]
metric_legend = ax_top.legend(
    handles=_metric_handles,
    handler_map=dict.fromkeys(_metric_handles, EnvelopePairHandler()),
    # Names the other axis of scatter: at fixed q̄ the rate still depends on the
    # intrinsic shape, and these are the quantiles of that spread.
    title=f"Shape-to-shape {round(100 * ENV_LO_Q)}-{round(100 * ENV_HI_Q)}% envelope",
    title_fontsize=8,
    alignment="left",
    bbox_to_anchor=(0.0, 1.0),
    loc="upper left",
    fontsize=9,
    ncols=len(_metric_handles),
    columnspacing=1.2,
    handleheight=1.3,
    framealpha=0.85,
)
ax_top.add_artist(metric_legend)

# The two keys differ in height between the simple and full figures, so a fixed
# anchor leaves a different gap in each.  Measure the metric legend and hang the
# band key a fixed number of points below it, giving both figures the same gap.
LEGEND_GAP_PT: Final = 6.0
fig.canvas.draw()
_ax_px = ax_top.get_window_extent().height
_gap_ax = (LEGEND_GAP_PT / 72.0 * fig.dpi) / _ax_px
_metric_bot = ax_top.transAxes.inverted().transform(
    (0.0, metric_legend.get_window_extent().y0),
)[1]
banded_legend.set_bbox_to_anchor(
    (0.0, _metric_bot - _gap_ax),
    transform=ax_top.transAxes,
)

# Shade the Q̄ ranges predicted by each DM model.
# Where the two ranges overlap, replace solid fills with complementary hatching
# so the two colors don't mix.  Outside the overlap: plain semi-transparent fills.
# Apply to both axes to show regions across the break
_overlap = (max(SIDM_Q[0], CDM_Q[0]), min(SIDM_Q[1], CDM_Q[1]))
_has_overlap = _overlap[0] < _overlap[1]

# Shade the Q̄ ranges predicted by each DM model.
# Where the two ranges overlap, replace solid fills with complementary hatching
# so the two colors don't mix.  Outside the overlap: plain semi-transparent fills.
# The simple figure drops the model bands; the appendix version keeps them.
SHOW_MODEL_BANDS: Final = not SIMPLE
for ax_shading in [ax_top] if SHOW_MODEL_BANDS else []:
    if _has_overlap:
        for _lo, _hi, _col in [
            (SIDM_Q[0], _overlap[0], SIDM_COLOR),
            (_overlap[1], SIDM_Q[1], SIDM_COLOR),
            (CDM_Q[0], _overlap[0], CDM_COLOR),
            (_overlap[1], CDM_Q[1], CDM_COLOR),
        ]:
            if _lo < _hi:
                ax_shading.axvspan(_lo, _hi, alpha=0.20, color=_col, lw=0, zorder=0)
    else:
        ax_shading.axvspan(*SIDM_Q, alpha=0.20, color=SIDM_COLOR, lw=0, zorder=0)
        ax_shading.axvspan(*CDM_Q, alpha=0.20, color=CDM_COLOR, lw=0, zorder=0)

# Labels for the fiducial-model vertical lines.  The model name rides with its
# q̄ symbol rather than sitting separately at the top of the panel, so one label
# names the line instead of two labels splitting the job.
# Sit them clear of everything grey at that q̄: the outermost quantile band and
# the dashed per-potential envelope both reach higher than the median curve, and
# which of the two is on top changes across the panel.
_blend = mtransforms.blended_transform_factory(ax_top.transData, ax_top.transAxes)
_GREY_TOP = q84 if SIMPLE else q95
_LABEL_PAD = 0.005  # data units (0.5 percentage points) above the grey
# The envelope is a wedge that climbs steeply toward low q̄, so clearing it only
# at the line's own q̄ still leaves the label looking embedded: the grey beside
# it is higher.  Clear it across the label's full width instead.
_LABEL_HALF_W = 0.012  # half the label width, in q̄
_MODEL_LABELS = [
    (QBAR_SIDM, "SIDM", r"$\bar{q}_{\rm SI}^{\rm sim}$", SIDM_COLOR),
    (QBAR_CDM, "CDM", r"$\bar{q}_{\rm C}^{\rm sim}$", CDM_COLOR),
]


def _clear_of_grey(xref: float) -> float:
    """Lowest y that clears every grey artist across the label's width."""
    span = np.linspace(xref - _LABEL_HALF_W, xref + _LABEL_HALF_W, 9)
    return _LABEL_PAD + max(
        float(np.max(rate_at(span, _GREY_TOP))),
        float(np.max(rate_at(span, m3_hi))),
    )


# One shared height, set by whichever label needs the most clearance.  Placing
# each at its own clearance staggers them by ~5 percentage points, which reads
# as an accident rather than as data.
_Y_LABELS = max(_clear_of_grey(x) for x, *_ in _MODEL_LABELS)


for _xref, _name, _sym, _col in _MODEL_LABELS:
    if SHOW_MODEL_BANDS:
        # The appendix envelope runs to 5-95%, which reaches the top of the
        # panel, so there is no room above it.  Pin to the top instead; the
        # coloured bands anchor the label there.  The label has to stay on its
        # own line, so where a legend already occupies the top it drops below
        # that legend rather than sliding sideways out of the way.
        _ypos, _va, _tf = 0.97, "top", _blend
        _x_disp = ax_top.transData.transform((_xref, 0.0))[0]
        for _lg in (metric_legend, banded_legend):
            _bb = _lg.get_window_extent()
            if _bb.x0 <= _x_disp <= _bb.x1:
                _below = ax_top.transAxes.inverted().transform((0.0, _bb.y0))[1]
                _ypos = min(_ypos, _below - 0.02)
    else:
        _ypos, _va, _tf = _Y_LABELS, "bottom", ax_top.transData
    ax_top.text(
        _xref,
        _ypos,
        f"{_name}\n{_sym}",
        color=_col,
        fontsize=10,
        fontweight="bold",
        ha="center",
        va=_va,
        linespacing=1.25,
        transform=_tf,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2),
        zorder=6,
    )

# Markers on the fiducial-model vertical lines.  Each is the POPULATION MEAN of
# the rate over that model's halo distribution, which is the integral over
# P_M(S), plotted at the population mean qbar.  In principle that is not the
# rate at the mean shape, but lambda(qbar) is nearly straight over this range
# (d2lambda/dqbar2 = +0.14 against a slope of 0.63), so the two agree to under
# 0.3% and the markers land on the curve.  Averaging is still the honest
# computation: it is what eq. p_M defines, and it does not assume the curve
# stays straight for a shape distribution wider than these two.
_xref_marks = np.array([QBAR_SIDM, QBAR_CDM])
_qbar_dists = (QBAR_SIDM_DIST, QBAR_CDM_DIST)
_y_m2_marks = np.array([rate_at(q, m2_hi).mean() for q in _qbar_dists])
_y_m3_marks = np.array([rate_at(q, m3_mid).mean() for q in _qbar_dists])

# Sensitivity of the forecast to the single-qbar approximation.  The alternative
# is to take P_M(S) uniform over each region, which makes lambda the MEAN OF THE
# RATE over the grid potentials inside the region -- not the rate at the mean
# qbar.  The two differ by ~1% because lambda(qbar) is curved, and the manuscript
# describes this as an integral, so compute it that way.  Ordered (SIDM, CDM),
# like `_xref_marks`.
_in_sidm = sidm_mask
_in_cdm = cdm_mask
_xref_region_means = np.array([Q_FLAT[_in_sidm].mean(), Q_FLAT[_in_cdm].mean()])
# Average the region's own potentials.  Averaging the fitted envelope over the
# same q̄ values instead would vary only where the curve is sampled, so it would
# not register the envelope sitting ~3% above the halos it describes.
_y_m2_region_means = np.array(
    [
        float(mean_m2_pot[_in_sidm].mean()),
        float(mean_m2_pot[_in_cdm].mean()),
    ],
)
for _ax_mark in (ax_top,):
    _ax_mark.scatter(
        np.concatenate([_xref_marks, _xref_marks]),
        np.concatenate([_y_m2_marks, _y_m3_marks]),
        marker="D",
        s=45,
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        zorder=7,
    )

# The two lower (arc-length) diamonds sit close together near the axis, so
# label them with their values.
for _xm, _ym in zip(_xref_marks, _y_m2_marks, strict=True):
    _ax_mark.annotate(
        f"{100 * _ym:.2f}%",
        xy=(_xm, _ym),
        xytext=(6, 7),
        textcoords="offset points",
        fontsize=9,
        ha="left",
        va="bottom",
        zorder=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1),
    )

# ── Panel 2: q1 x q2 heatmap ─────────────────────────────────────────────────

# Only the q1 ≥ q2 (a ≥ b ≥ c) triangle is independently computed; the rest of
# the grid is its exact swap-mirror.  Show just that triangle, rather than imply
# N_POTS independent evaluations.  Dropping whole cells would step the boundary
# along cell edges, so the full mesh is drawn and clipped to the q1 = q2 line
# instead: the cells it straddles are cut, and what survives of them carries the
# value at their own centre, which the mirror makes exact on either side.
# Derived, not hard-coded: a fixed 0.5 sat outside the view only while the grid
# floor was 0.60, and silently clipped the mesh short of the right spine once
# --ca-floor extended the grid to 0.45.  The vertex stays on the b/a = c/a
# diagonal either way, so the cut line is unchanged.
_TRI_CLIP_LO: Final = float(min(q1_vals.min(), q2_vals.min())) - 0.1
_TRI_CLIP_HI: Final = 1.1
_tri_clip_path = _MplPath(
    [
        (_TRI_CLIP_LO, _TRI_CLIP_LO),
        (_TRI_CLIP_HI, _TRI_CLIP_HI),
        (_TRI_CLIP_LO, _TRI_CLIP_HI),
        (_TRI_CLIP_LO, _TRI_CLIP_LO),
    ],
    [_MplPath.MOVETO, _MplPath.LINETO, _MplPath.LINETO, _MplPath.CLOSEPOLY],
)


# ── q̄ turnover in b/a (quoted in sec:cosmology:rates) ────────────────────────
# At fixed c/a, q̄ peaks short of the oblate limit and falls to b/a = 1:
# equatorial sightlines span projected ratios from c/a to c/b, and b -> a
# collapses that spread onto the flattest of them.
#
# Work in (b/a, c/a) off the flat arrays rather than off a reshaped grid.  Both
# q̄ and the rate are exactly swap-symmetric, so a grid would work too, but only
# by accident: the flat layout varies q1 along the fast axis, which is the
# transpose of what the pcolormesh call below assumes.
_BA_FLAT = np.maximum(q1_flat, q2_flat)
_CA_FLAT = np.minimum(q1_flat, q2_flat)


def _ridge_ba(ca: float) -> float:
    """b/a at which q̄ peaks, at this c/a."""
    sel = np.isclose(_CA_FLAT, ca)
    return float(_BA_FLAT[sel][np.argmax(Q_FLAT[sel])])


def _at_shape(ba: float, ca: float) -> tuple[float, float]:
    """(q̄, rate) at one grid shape."""
    sel = np.isclose(_BA_FLAT, ba) & np.isclose(_CA_FLAT, ca)
    return float(Q_FLAT[sel][0]), float(mean_m3_pot[sel][0])


# Quote the ridge over the c/a range the two regions actually span, not the
# whole grid: outside it the peak runs into the b/a = c/a edge.
_ridge_cas = q2_vals[(q2_vals >= CDM_CA_MIN) & (q2_vals <= SIDM_CA_MAX)]
_ridge_bas = [_ridge_ba(float(c)) for c in _ridge_cas]

# Worked example at the SIDM region's lower c/a edge, snapped to the nearest
# grid knot: the edge is a quartile of the halo sample, not a sampled shape.
_TURN_CA: Final = float(q2_vals[np.argmin(np.abs(q2_vals - SIDM_CA_MIN))])
_TURN_BA: Final = _ridge_ba(_TURN_CA)
_turn_qbar_peak, _turn_rate_peak = _at_shape(_TURN_BA, _TURN_CA)
_turn_qbar_obl, _turn_rate_obl = _at_shape(1.0, _TURN_CA)


# The b/a axis accounts for almost none of the CDM/SIDM separation in q̄: the
# two regions' b/a ranges overlap and the SIDM one sits near the turnover, so
# holding c/a fixed and sweeping b/a across every knot the two regions cover
# moves q̄ by _QBAR_BA_SPREAD at most, against the _QBAR_SEPARATION between the
# two representative flattenings.
_reg_ba_knots = q1_vals[
    (q1_vals >= min(CDM_BA_MIN, SIDM_BA_MIN))
    & (q1_vals <= max(CDM_BA_MAX, SIDM_BA_MAX))
]
_reg_ca_knots = q2_vals[
    (q2_vals >= min(CDM_CA_MIN, SIDM_CA_MIN))
    & (q2_vals <= max(CDM_CA_MAX, SIDM_CA_MAX))
]


def _qbar_ba_spread(ca: float) -> float:
    """Range of q̄ over the regions' b/a knots, at this c/a."""
    vals = [
        _at_shape(float(b), ca)[0] for b in _reg_ba_knots if b >= ca
    ]
    return max(vals) - min(vals) if len(vals) > 1 else 0.0


_QBAR_BA_SPREAD: Final = max(_qbar_ba_spread(float(c)) for c in _reg_ca_knots)
_QBAR_SEPARATION: Final = QBAR_SIDM - QBAR_CDM


def _clip_upper_right(verts: np.ndarray) -> np.ndarray:
    """Clip a polygon to the b/a >= c/a half-plane (Sutherland-Hodgman, one edge).

    Vertices are in panel coordinates, (x, y) = (c/a, b/a), so this checks
    p[1] >= p[0].
    """

    def _inside(p: np.ndarray) -> bool:
        return p[1] >= p[0]

    def _inter(s: np.ndarray, e: np.ndarray) -> tuple[float, float]:
        dx, dy = e[0] - s[0], e[1] - s[1]
        denom = dy - dx
        t = 0.0 if denom == 0 else (s[0] - s[1]) / denom
        return (s[0] + t * dx, s[1] + t * dy)

    out: list = []
    for i in range(len(verts)):
        s, e = verts[i - 1], verts[i]
        if _inside(e):
            if not _inside(s):
                out.append(_inter(s, e))
            out.append(tuple(e))
        elif _inside(s):
            out.append(_inter(s, e))
    return np.array(out)


def _region_vertices(
    ba_min: float,
    ba_max: float,
    ca_min: float,
    ca_max: float,
) -> np.ndarray:
    """Outline of a (b/a, c/a) box, in the panel's (x, y) = (q2, q1) coordinates.

    b/a is max(q1, q2) and c/a is min(q1, q2), so a box in those ratios is *two*
    rectangles in the (q1, q2) square, mirrored about the diagonal.  Both boxes
    here have ba_min > ca_max, which leaves the two disjoint, and the panel is
    clipped to q1 >= q2, so exactly the upper-left one survives and the region
    draws as a plain rectangle.

    When ba_min <= ca_max the two rectangles instead overlap across the
    diagonal and their union is a single connected stepped octagon, which is
    what the eight-vertex branch returns.  Drawing *that* outline for a disjoint
    box would bridge the gap and enclose a wedge of the (q1, q2) plane that is
    not in the region, and drawing the rectangle for an overlapping box would
    drop the part that crosses the diagonal, so the two cases are kept apart.
    """
    if ba_min > ca_max:  # disjoint: only the upper-left rectangle is in view
        return np.array(
            [(ca_min, ba_min), (ca_max, ba_min), (ca_max, ba_max), (ca_min, ba_max)],
        )
    return np.array(  # connected: symmetric stepped octagon about the diagonal
        [
            (ca_min, ba_max),  # top-left
            (ca_max, ba_max),  # before top-right cutout
            (ca_max, ca_max),  # inner corner (top-right cutout)
            (ba_max, ca_max),  # right side of top-right cutout
            (ba_max, ca_min),  # bottom-right
            (ba_min, ca_min),  # before bottom-left cutout
            (ba_min, ba_min),  # inner corner (bottom-left cutout)
            (ca_min, ba_min),  # left side of bottom-left cutout
        ],
    )


pm = ax2.pcolormesh(
    q2_vals,
    q1_vals,
    bad_grid,
    cmap=CMAP,
    vmin=VMIN,
    vmax=VMAX,
    shading="gouraud",
)
pm.set_clip_path(_tri_clip_path, ax2.transData)
# The b/a dependence is real but an order of magnitude weaker than the c/a one
# (mean |dlambda/d(c/a)| = 0.30 against 0.10), and it is non-monotonic: at fixed
# c/a the rate dips where q̄ peaks, near b/a ~ 0.92-0.94.  That is under 1 pp
# against an 18 pp ramp, so the colour map cannot show it.  Contours can.
CONTOUR_LEVELS: Final = np.arange(0.02, VMAX, 0.02)
_cs = ax2.contour(
    q2_vals,
    q1_vals,
    bad_grid,
    levels=CONTOUR_LEVELS,
    colors="white",
    linewidths=0.7,
    alpha=0.35,
    zorder=2,
)
_cs.set_clip_path(_tri_clip_path, ax2.transData)


# The panel is clipped to q1 >= q2, so within the plotted triangle the axes are
# ordered (r_x >= r_y >= r_z) and q1 IS b/a, q2 IS c/a.  Label them that way:
# it matches the literature the regions are drawn from, and it is what the
# region bounds above are expressed in.
ax2.set_xlabel(r"$\mathbf{c/a}$", fontsize=14, fontweight="bold")
ax2.set_ylabel(r"$\mathbf{b/a}$", fontsize=14, fontweight="bold")
# Both ratios descend, so the spherical limit (1, 1) lands in the lower-left
# corner and the blank b/a < c/a half is the upper-left wedge.  That leaves the
# left edge bordering blank space, so the b/a axis is drawn on the right.
# shading='gouraud' colours between knot *centres*, so the mesh ends on the
# first and last knot and the limits have to end there too.  (Under
# shading='nearest' each knot owns a cell and the limits need a half-cell pad
# instead, taken from the edge cell itself: the grid is non-uniform, 0.03 below
# 0.90 and 0.02 above, so the mean diff would gap one end and clip the other.)
# b/a is cropped to [BA_VIEW_MIN, 1]: the b/a >= c/a clip leaves only a thin
# diagonal sliver populated below ~0.85, and both regions sit above 0.89.
BA_VIEW_MIN: Final = 0.85
ax2.set_xlim(q2_vals.max(), q2_vals.min())  # descending: c/a = 1 at the left
ax2.set_ylim(q1_vals.max(), BA_VIEW_MIN)  # descending: b/a = 1 at the bottom
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.3g}"))
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.3g}"))
ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.025))
# Drop the left/top frame edges; they border the blank b/a < c/a half.  The b/a
# ticks and label move to the right edge, which is fully populated.
ax2.spines["left"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
# The spherical limit sits at (c/a, b/a) = (1, 1), the corner cell where the
# convexity rate goes to zero.  Mark the point itself, not just the region.
ax2.plot(
    1.0 - 0.009,
    1.0 - 0.0025,
    marker="o",
    markersize=7,
    markerfacecolor="white",
    markeredgecolor="black",
    markeredgewidth=1.0,
    clip_on=False,
    zorder=6,
)
# Offset in points, not data: (1, 1) is the corner, so a data-space offset that
# clears the frame at one aspect ratio sits on it at another.  The rise also has
# to carry the whole label inside the b/a >= c/a wedge, which at this panel's
# aspect buys 2.6 pt of width per pt of rise.
ax2.annotate(
    "spherical",
    xy=(1.0, 1.0),
    xytext=(10, 8),
    textcoords="offset points",
    color="white",
    fontsize=12,
    weight="bold",
    ha="left",
    va="bottom",
    zorder=6,
    path_effects=[mpatheffects.withStroke(linewidth=2.5, foreground="black")],
)


# SIDM + baryons expectation: b/a in [SIDM_BA_MIN, SIDM_BA_MAX], c/a in
# [SIDM_CA_MIN, SIDM_CA_MAX].  Bounds are set above; keep them there.
sidm_vertices = _region_vertices(
    SIDM_BA_MIN,
    SIDM_BA_MAX,
    SIDM_CA_MIN,
    SIDM_CA_MAX,
)
ax2.add_patch(
    MplPolygon(
        _clip_upper_right(sidm_vertices),
        closed=True,
        linewidth=3.5,
        edgecolor=SIDM_COLOR,
        facecolor=to_rgba(SIDM_COLOR, REGION_FILL_ALPHA),
        linestyle=":",
        # Above the CDM box: the SIDM region is the smaller of the two and
        # sits inside its corner, so it has to win the overlap to stay legible.
        zorder=8,
    ),
)
# Centred on the region box itself, so the label follows the bounds rather than
# needing to be re-placed by hand whenever they move.
ax2.text(
    0.5 * (SIDM_CA_MIN + SIDM_CA_MAX),
    0.5 * (SIDM_BA_MIN + SIDM_BA_MAX),
    # Sized to the box, which spans only 0.07 in c/a.  Keep "few", not ">1":
    # CA_OFFSET is calibrated to sigma/m >~ a few, so a ">1" label would claim
    # the box covers rounder haloes than that range gives.
    "SIDM\n($\\sigma/m\\gtrsim$few\nIQR)",
    color=SIDM_COLOR,
    fontsize=8,
    fontweight="bold",
    ha="center",
    va="center",
    zorder=9,  # with its box, above the CDM one
)
# CDM + baryons expectation: b/a in [CDM_BA_MIN, CDM_BA_MAX], c/a in
# [CDM_CA_MIN, CDM_CA_MAX].  Bounds are set above; keep them there.
cdm_vertices = _region_vertices(CDM_BA_MIN, CDM_BA_MAX, CDM_CA_MIN, CDM_CA_MAX)
ax2.add_patch(
    MplPolygon(
        _clip_upper_right(cdm_vertices),
        closed=True,
        linewidth=3.5,
        edgecolor=CDM_COLOR,
        facecolor=to_rgba(CDM_COLOR, REGION_FILL_ALPHA),
        linestyle=":",
        zorder=7,  # over the contours, under the SIDM box
    ),
)
# CDM region clipped to q1>=q2 triangle - visual center of clipped polygon
# larger q1 = further down; c/a ascends to the right
ax2.text(
    0.5 * (CDM_CA_MIN + CDM_CA_MAX),
    0.5 * (CDM_BA_MIN + CDM_BA_MAX),
    # 12 pt to clear the dotted outline: at 13 pt the second line is 125 px
    # against the box's 133 px.
    "CDM\n(Auriga IQR)",
    color=CDM_COLOR,
    fontsize=12,
    fontweight="bold",
    ha="center",
    va="center",
    zorder=7,
)
ax_top.tick_params(labelsize=13)
ax2.tick_params(labelsize=13)

# ── Overlap-zone hatching (data coords, so it survives bbox_inches='tight') ──
# IdentityTransform() pixel coordinates go stale when savefig re-renders.
# Instead, compute lines in data coordinates using the axes physical dimensions
# (from ax_top.get_position() after tight_layout) to derive the 45° slope.
if SHOW_MODEL_BANDS and _has_overlap:
    # Hatching for top axis (1% - 17.5%)
    _pos = ax_top.get_position()
    _ax_w = _pos.width * fig.get_figwidth()  # axes width in inches
    _ax_h = _pos.height * fig.get_figheight()  # axes height in inches
    _xlim_range = abs(ax_top.get_xlim()[1] - ax_top.get_xlim()[0])
    _ylim_bot, _ylim_top = ax_top.get_ylim()  # use actual axes limits
    _y_range = _ylim_top - _ylim_bot
    # dy_data per unit |dx_data| such that "/" appears at 45° in display space
    _slope = (_y_range / _xlim_range) * (_ax_w / _ax_h)

    _w = _overlap[1] - _overlap[0]
    _n = 5  # lines per colour across _w
    _sp = _w / _n  # x-spacing between same-colour lines
    # k must run high enough so lines cover the full y-height of the axes
    _k_max = int((_w + _y_range / _slope) / _sp) + 2
    # Perpendicular gap between adjacent interleaved lines (alternating colours
    # are spaced _sp/2 apart in data-x; 45° lines → multiply by cos 45°).
    _lw = 1.05 * (_sp / 2) / _xlim_range * _ax_w * 72 / (2**0.5)

    def _make_segs(phase: float) -> list:
        segs = []
        for k in range(_k_max):
            x0 = _overlap[0] + (k + phase) * _sp
            # Full-height "/" line; the clip path handles all four boundaries.
            segs.append(
                [
                    (x0, _ylim_bot),
                    (x0 - _y_range / _slope, _ylim_top),
                ],
            )
        return segs

    # Clip both collections to the overlap rectangle so thick lines don't
    # bleed past the vertical boundaries.  Using an explicit Path + transData
    # avoids the Rectangle-patch internals that misbehave on an inverted x-axis.
    _clip_verts = [
        (_overlap[0], _ylim_bot),
        (_overlap[1], _ylim_bot),
        (_overlap[1], _ylim_top),
        (_overlap[0], _ylim_top),
        (_overlap[0], _ylim_bot),
    ]
    _clip_codes = [
        _MplPath.MOVETO,
        _MplPath.LINETO,
        _MplPath.LINETO,
        _MplPath.LINETO,
        _MplPath.CLOSEPOLY,
    ]
    _clip_mplpath = _MplPath(_clip_verts, _clip_codes)

    _lc_sidm = LineCollection(
        _make_segs(0.0),
        colors=SIDM_COLOR,
        linewidths=_lw,
        alpha=0.20,
        transform=ax_top.transData,
        zorder=0,
    )
    _lc_cdm = LineCollection(
        _make_segs(0.5),
        colors=CDM_COLOR,
        linewidths=_lw,
        alpha=0.20,
        transform=ax_top.transData,
        zorder=0,
    )
    _lc_sidm.set_clip_path(_clip_mplpath, ax_top.transData)
    _lc_cdm.set_clip_path(_clip_mplpath, ax_top.transData)
    ax_top.add_collection(_lc_sidm)
    ax_top.add_collection(_lc_cdm)


# ======================================================

# The colorbar and the right panel's y-axis encode the same quantity, so the
# scale rides that axis instead of standing alone beside the heatmap: the panel's
# own tick labels double as the colorbar ticks over its full height.
# ── Colour bar, lying along the b/a = c/a diagonal ───────────────────────────
# The blank b/a < c/a wedge is the only empty space in the left panel, and it is
# bounded by the diagonal, so a bar parallel to that line is the one orientation
# that fits.  Build it in data coordinates: a display-space strip would go stale
# when `bbox_inches="tight"` re-renders, and `floating_axes` fits its rotated
# frame to a bbox, which shears the angle unless the bbox aspect happens to
# match.  The axes are linear, so mapping the strip's corners through
# `transData` once is exact.
_t2 = ax2.transData
_inv2 = _t2.inverted()
_p0 = _t2.transform((1.0, 1.0))  # the spherical corner
_p1 = _t2.transform((BA_VIEW_MIN, BA_VIEW_MIN))  # far end of the diagonal
_d = _p1 - _p0
_len = float(np.hypot(*_d))
_u = _d / _len  # along the diagonal
_n = np.array([-_u[1], _u[0]])  # perpendicular, into the wedge
if np.dot(_n, _t2.transform((1.0, BA_VIEW_MIN)) - _p0) < 0:
    _n = -_n

CB_OFFSET_PX: Final = 30.0  # perpendicular displacement of the bar's centre line
CB_WIDTH_PX: Final = 22.0
CB_T0: Final = 0.07  # fraction of the diagonal where the bar starts
CB_T1: Final = 0.98


def _bar_xy(t: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Strip coordinates in data space; `t` along the bar, `w` across it."""
    disp = (
        _p0
        + t[:, None, None] * _len * _u
        + (CB_OFFSET_PX + w[None, :, None] * CB_WIDTH_PX) * _n
    )
    out = _inv2.transform(disp.reshape(-1, 2)).reshape((*disp.shape[:2], 2))
    return out[..., 0], out[..., 1]


_ts = np.linspace(CB_T0, CB_T1, 257)
_bx, _by = _bar_xy(_ts, np.array([-0.5, 0.5]))
# Gouraud, not flat: 256 abutting flat quads leave hairline seams between them
# in vector output, which print as white striping across the bar.  Gouraud
# interpolates one mesh, so the ramp is continuous and needs no rasterizing.
# It colours by VERTEX, so the value array spans 257 x 2, not 256 x 1.
_cb = ax2.pcolormesh(
    _bx,
    _by,
    np.repeat(np.linspace(VMIN, VMAX, 257)[:, None], 2, axis=1),
    cmap=CMAP,
    vmin=VMIN,
    vmax=VMAX,
    shading="gouraud",
    zorder=7,
)
_cb.set_clip_on(False)
# The left panel's contour levels, drawn across the bar in the same style, so a
# contour out on the wedge can be read straight off the scale.
for _lvl in CONTOUR_LEVELS:
    _lf = CB_T0 + (CB_T1 - CB_T0) * (_lvl - VMIN) / (VMAX - VMIN)
    _rx, _ry = _bar_xy(np.array([_lf]), np.array([-0.5, 0.5]))
    ax2.plot(
        _rx[0],
        _ry[0],
        color="white",
        lw=0.7,
        alpha=0.35,
        clip_on=False,
        zorder=7.5,
    )
# Outline
_ox, _oy = _bar_xy(np.array([CB_T0, CB_T1]), np.array([-0.5, 0.5]))
ax2.add_patch(
    MplPolygon(
        np.column_stack(
            [
                [_ox[0, 0], _ox[1, 0], _ox[1, 1], _ox[0, 1]],
                [_oy[0, 0], _oy[1, 0], _oy[1, 1], _oy[0, 1]],
            ],
        ),
        closed=True,
        facecolor="none",
        edgecolor="black",
        linewidth=0.5,
        clip_on=False,
        zorder=8,
    ),
)

# Ticks: short stubs on the wedge side, labels rotated to lie along the bar.
_CB_ANGLE: Final = float(np.degrees(np.arctan2(_d[1], _d[0])))
for _val in [t for t in ax_top.get_yticks() if VMIN <= t <= VMAX]:
    _f = CB_T0 + (CB_T1 - CB_T0) * (_val - VMIN) / (VMAX - VMIN)
    _tick = np.array([_f])
    _tx, _ty = _bar_xy(_tick, np.array([0.5, 0.95]))
    ax2.plot(_tx[0], _ty[0], color="black", lw=1.0, clip_on=False, zorder=8)
    # ha="center" centres the label on its tick along the bar; va="bottom" then
    # sets it just outside the stub, since `_n` is 90 deg CCW of `_u`.
    _lx, _ly = _bar_xy(_tick, np.array([1.05]))
    ax2.text(
        _lx[0, 0],
        _ly[0, 0],
        f"{100 * _val:.0f}",
        ha="center",
        va="bottom",
        fontsize=11,
        rotation=_CB_ANGLE,
        rotation_mode="anchor",
        clip_on=False,
        zorder=8,
    )

# Title runs along the bar, alongside its midpoint.  `_n` is 90 deg CCW of `_u`,
# so va="bottom" puts the text on the far side of the anchor from the bar.
_ttl = ax2.text(
    0.0,
    0.0,
    r"$\lambda_{\mathrm{det}}^{(10)}$ convexity rate (%)",
    ha="center",
    va="bottom",
    rotation=_CB_ANGLE,
    rotation_mode="anchor",
    fontsize=12,
    fontweight="bold",
    clip_on=False,
    zorder=8,
)
# The value labels reach out to w = 2.0 (measured by projecting their corners
# onto `_n`), so the title clears them just past that.  A bbox overlap test is
# no use here: the axis-aligned box of a 70-degree rotated title spans w = -0.8
# to 5.2, six times the text's real thickness.
CB_TITLE_W: Final = 2.65
_ttl_x, _ttl_y = _bar_xy(np.array([0.5]), np.array([CB_TITLE_W]))
_ttl.set_position((_ttl_x[0, 0], _ttl_y[0, 0]))

# ==========================================================

out = paths.FIGURES / "cosmology" / ("convexity_rate_grid.pdf" if SIMPLE else "convexity_rate_grid_full.pdf")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")

# Values this script owns that the manuscript quotes: the plot-time minimum
# segment length, the percentile band edges, the two fiducial flattenings, and
# the rates read off the curves at those flattenings (the white diamonds).
# `_xref_marks` is ordered (SIDM, CDM). See src/scripts/make_variables.py.
# Variant-independent, so only the full pass writes them.
if not SIMPLE:
    write_values(
        __file__,
        {
            "ell_min_kpc": X_SELECT_KPC,
            "qbar_sidm": QBAR_SIDM,
            "qbar_cdm": QBAR_CDM,
            # The SIDM relaxation, so eq:cosmology:sidm_relaxation in the
            # manuscript cannot drift from the map actually applied here.
            "sidm_ca_offset": CA_OFFSET,
            "sidm_relaxation": RELAXATION,
            "auriga_ca_median": _ca_med,
            # The halo-shape regions.  These fix qbar_C, qbar_SI and so eta, i.e.
            # the whole forecast, and are quoted in sec:cosmology:rates; export them
            # so the manuscript and the figure cannot state different boxes.
            "region_bounds": {
                "cdm_ba": [CDM_BA_MIN, CDM_BA_MAX],
                "cdm_ca": [CDM_CA_MIN, CDM_CA_MAX],
                "sidm_ba": [SIDM_BA_MIN, SIDM_BA_MAX],
                "sidm_ca": [SIDM_CA_MIN, SIDM_CA_MAX],
            },
            # q̄ turnover in b/a; see the block above _clip_upper_right.
            "qbar_ba_spread": float(_QBAR_BA_SPREAD),
            "qbar_region_separation": float(_QBAR_SEPARATION),
            "qbar_turnover": {
                "ba_lo": min(_ridge_bas),
                "ba_hi": max(_ridge_bas),
                "ca": float(_TURN_CA),
                "ba": float(_TURN_BA),
                "qbar_peak": _turn_qbar_peak,
                "qbar_oblate": _turn_qbar_obl,
                "rate_peak": _turn_rate_peak,
                "rate_oblate": _turn_rate_obl,
            },
            "percentile_levels": [round(100 * v) for v in LEVELS],
            # Coverage of the qbar strips, quoted in the caption and in
            # sec:cosmology:rates.
            "band_pctl": [round(v) for v in BAND_PCTL],
            # Sensitivity of the SIDM rate to the representative flattening.
            # The quoted qbar is printed to two significant figures; the shift
            # that rounding hides already moves the rate by several per cent,
            # which is the systematic quoted in sec:discussion:caveats.
            "qbar_sidm_display_shift": float(
                abs(QBAR_SIDM - float(f"{QBAR_SIDM:.2g}")),
            ),
            "lambda_arc_sidm_shift_frac": float(
                np.interp(float(f"{QBAR_SIDM:.2g}"), q_env, m2_hi)
                / np.interp(QBAR_SIDM, q_env, m2_hi)
                - 1.0,
            ),
            "lambda_arc_sidm": float(_y_m2_marks[0]),
            "lambda_arc_cdm": float(_y_m2_marks[1]),
            # Region-mean alternative to the selected flattenings, quoted in
            # sec:cosmology:rates as the sensitivity of the forecast.
            "qbar_region_mean_sidm": float(_xref_region_means[0]),
            "qbar_region_mean_cdm": float(_xref_region_means[1]),
            "lambda_arc_sidm_region_mean": float(_y_m2_region_means[0]),
            "lambda_arc_cdm_region_mean": float(_y_m2_region_means[1]),
            "lambda_det_sidm": float(_y_m3_marks[0]),
            "lambda_det_cdm": float(_y_m3_marks[1]),
        },
    )
