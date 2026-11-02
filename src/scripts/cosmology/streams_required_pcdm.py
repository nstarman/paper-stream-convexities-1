"""Required stream sample size and observed-catalogue inference.

Implements eq:cosmology:N_required_full (balanced-power one-sided z-test):

    N  >=  z^2 * (sqrt(p_CDM*(1-p_CDM)) + sqrt(p_SIDM*(1-p_SIDM)))^2
                / (p_CDM - p_SIDM)^2

Left panel:
    x = p_CDM                  (log scale)
    y = N_streams required      (log scale)

Right panel:
    x = observed non-concavity rate K/N
    y = signed likelihood-ratio scale Z_LR for a fixed catalogue size

In the left panel each sigma curve is drawn as a *band* spanning the
simulation-motivated range of eta = p_SIDM/p_CDM ∈ [ETA_LO, ETA_HI], with a
fiducial line at
ETA_FID.  For a given (p_CDM, eta) pair, p_SIDM = eta * p_CDM, which is
always valid for 0 < eta < 1 and p_CDM ∈ (0, 1) — no NaN fallback needed.

The normal-approximation formula above is what the left panel plots.  It is
validated against the exact binomial two-hypothesis test by n_required_exact()
below.  At the measured rates the two agree to well under 1% for Nsigma >= 2,
while the normal form underestimates N by ~20% at 1sigma (discreteness dominates
when the critical count K* is only a few).  The exact requirement is a sawtooth
in N, so the 1sigma figure moves noticeably for small changes in the rates; the
exported values are recomputed with them.  Treat the 1sigma curve as indicative
only.

Calibration notes:
  - The paper text (sec:cosmology:sample_size) quotes eta in [0.10, 0.70], with
    a fiducial eta = ETA_FID; eta = 0.70 requires N ~ 3.3e4 at 3sigma for
    lambda_CDM = 0.01.
  - SIDM sphericalises halos for sigma/m ~ 1-10 cm^2/g (Rocha+2013,
    Vogelsberger+2014), driving eta << 1 for strong cross-sections.
  - Survey stream counts are order-of-magnitude estimates; refine with actual
    catalogue projections before publication.
"""

import json
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.textpath import TextPath
from matplotlib.text import Text as _AnnText
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.stats import binom, norm

from stream_convexity_paper.style import use_paper_style
from stream_convexity_paper import paths
from stream_convexity_paper.values import write_values

# ── Matplotlib style (matches paper conventions) ──────────────────────────────
use_paper_style()


# ── Formula (eq:cosmology:N_required_full) ────────────────────────────────────


def n_required(z, p_cdm, p_sidm):
    """Full balanced-power sample-size formula.

    Parameters
    ----------
    z : float
        Sigma level (z_alpha = z_beta = z).
    p_cdm, p_sidm : array_like
        Per-stream convexity probabilities; must satisfy 0 < p_sidm < p_cdm < 1.

    Returns
    -------
    ndarray
        N_required; NaN where inputs are out of range.

    """
    p_c = np.asarray(p_cdm, dtype=float)
    p_s = np.asarray(p_sidm, dtype=float)
    valid = (p_s > 0) & (p_c < 1) & (p_s < p_c)
    with np.errstate(invalid="ignore"):
        num = (np.sqrt(p_c * (1 - p_c)) + np.sqrt(p_s * (1 - p_s))) ** 2
        return np.where(valid, z**2 * num / (p_c - p_s) ** 2, np.nan)


# Consecutive passes required before the exact requirement is called settled.
MIN_STABLE_RUN = 64


def _exact_test_holds(n, alpha, p_cdm, p_sidm):
    """Whether N=n admits an integer critical count meeting both error rates."""
    k_star = int(binom.isf(alpha, n, p_sidm)) + 1
    while binom.sf(k_star - 1, n, p_sidm) > alpha:
        k_star += 1
    return binom.cdf(k_star - 1, n, p_cdm) <= alpha, k_star


def n_required_exact(z, p_cdm, p_sidm, n_max=400_000):
    """Exact binomial counterpart of n_required, for validation.

    Smallest N admitting an integer critical count K* with
        P(K >= K* | N, p_sidm) <= alpha   (do not favour CDM if SIDM is true)
        P(K <  K* | N, p_cdm)  <= beta    (do not miss CDM if CDM is true)
    at alpha = beta = Phi(-z), matching the balanced-power convention of
    n_required().  Returns (N, K*), or (None, None) if n_max is reached.

    The condition is a sawtooth in N: because K* is an integer, isolated N below
    the requirement happen to satisfy it while their neighbours do not.  We
    return the smallest N beyond which it never fails again, which is the
    quantity a survey can actually plan against.  Returning the first N to
    satisfy it instead makes the answer land on whichever tooth the rates
    happen to select, and it then moves by hundreds of streams under changes in
    the rates too small to see.

    Termination: failures cluster below the requirement and the teeth shorten
    as N grows, so we stop once the run of consecutive passes is longer than
    the span from the first pass to the last failure seen.
    """
    alpha = norm.sf(z)
    first = None
    last_fail = None
    run = 0
    for n in range(1, n_max):
        ok, k_star = _exact_test_holds(n, alpha, p_cdm, p_sidm)
        if not ok:
            if first is not None:
                last_fail = n
            run = 0
            continue
        if first is None:
            first = n
        run += 1
        if run > max(MIN_STABLE_RUN, (last_fail or first) - first):
            n_req = (last_fail + 1) if last_fail is not None else first
            return n_req, _exact_test_holds(n_req, alpha, p_cdm, p_sidm)[1]
    return None, None


def p_reject_sidm(k, n_streams, p_sidm):
    """Exact upper tail, eq:cosmology:p_reject_sidm.

    P(K' >= k | n_streams, p_sidm): the probability that SIDM alone produces at
    least the observed number of non-concavities.  Small values reject SIDM.
    """
    return binom.sf(k - 1, n_streams, p_sidm)


def p_reject_cdm(k, n_streams, p_cdm):
    """Exact lower tail, eq:cosmology:p_reject_cdm.

    P(K' <= k | n_streams, p_cdm): the probability that CDM produces no more
    than the observed number of non-concavities.  Small values reject CDM.
    """
    return binom.cdf(k, n_streams, p_cdm)


def z_from_p(p):
    """One-sided Gaussian significance, Z = Phi^-1(1 - p)."""
    return norm.isf(p)


def z_wald(k, n_streams, lam):
    """Normal (Wald) approximation, eq:cosmology:one_sided_rejection."""
    return abs(k / n_streams - lam) / np.sqrt(lam * (1 - lam) / n_streams)


def report_exact_significances(p_cdm=None, p_sidm=None, n_streams=1000):
    """Median-expected exact significances, and the Wald values they replace.

    Quoted in sec:cosmology:likelihood.  For n_streams=1000 at the conservative
    rates this reproduces Z_SI = 3.5 (Wald 4.8) and Z_rej,C = 2.8 (Wald 2.5).
    """
    p_cdm = P_CDM_LR if p_cdm is None else p_cdm
    p_sidm = P_SIDM_LR if p_sidm is None else p_sidm

    k_cdm = int(binom.median(n_streams, p_cdm))       # median catalogue if CDM
    p_si = p_reject_sidm(k_cdm, n_streams, p_sidm)

    k_sidm = int(binom.median(n_streams, p_sidm))     # median catalogue if SIDM
    p_c = p_reject_cdm(k_sidm, n_streams, p_cdm)

    return {
        "n_streams": n_streams,
        "k_if_cdm": k_cdm,
        "p_reject_sidm": p_si,
        "z_reject_sidm": z_from_p(p_si),
        "z_reject_sidm_wald": z_wald(k_cdm, n_streams, p_sidm),
        "k_if_sidm": k_sidm,
        "p_reject_cdm": p_c,
        "z_reject_cdm": z_from_p(p_c),
        "z_reject_cdm_wald": z_wald(k_sidm, n_streams, p_cdm),
    }


def delta_log_likelihood(obs_rate, p_cdm, p_sidm, n_streams):
    """Log P(K|CDM) / P(K|SIDM) with K = N * obs_rate."""
    obs = np.asarray(obs_rate, dtype=float)
    valid = (obs >= 0) & (obs <= 1) & (p_sidm > 0) & (p_sidm < p_cdm) & (p_cdm < 1)
    with np.errstate(invalid="ignore"):
        dll = n_streams * (
            obs * np.log(p_cdm / p_sidm)
            + (1 - obs) * np.log((1 - p_cdm) / (1 - p_sidm))
        )
    return np.where(valid, dll, np.nan)


def signed_lr_sigma(obs_rate, p_cdm, p_sidm, n_streams):
    """Signed likelihood-ratio scale; positive favours CDM."""
    dll = delta_log_likelihood(obs_rate, p_cdm, p_sidm, n_streams)
    return np.sign(dll) * np.sqrt(2 * np.abs(dll))


# ── Parameters ────────────────────────────────────────────────────────────────

# Measured convexity rates, from the script that reads them off the rate grid.
# The manuscript rounds these to ~1% and ~1/4% when speaking, but every number
# here and in the text is computed from the measured values themselves.
_RATES = json.loads(
    (paths.VALUES / "step_2_plot_triaxiality.values.json").read_text(
        encoding="utf-8",
    ),
)
_lam_arc_cdm = _RATES["lambda_arc_cdm"]
_lam_arc_sidm = _RATES["lambda_arc_sidm"]
_lam_det_cdm = _RATES["lambda_det_cdm"]
_lam_det_sidm = _RATES["lambda_det_sidm"]

# SIDM suppression ratio eta = p_SIDM / p_CDM.
# The band spans [ETA_LO, ETA_HI], a theoretical range; the fiducial solid line
# uses ETA_FID, which is measured rather than declared.
# Lower eta = stronger SIDM suppression = fewer streams needed.
# Higher eta = weaker suppression (models harder to tell apart).
ETA_LO = 0.10  # strong suppression (lower N bound)
ETA_FID = _lam_arc_sidm / _lam_arc_cdm  # measured
ETA_HI = 0.70  # weak suppression (upper N bound); eta=0.9 -> N >> 10^4 at 3sigma

SIGMAS = [1, 2, 3, 5]
# Largest present-day Local-Group stream catalogue, as quoted in
# sec:cosmology:sample_size ("N ~ 30-100").
N_LOCAL_GROUP = 100

# p_CDM grid (x-axis).
N_PTS = 800
P_CDM_MIN = 0.001
P_CDM_MAX = 0.05
P_CDM_GRID = np.logspace(np.log10(P_CDM_MIN), np.log10(P_CDM_MAX), N_PTS)

# The Euclid projection appears twice: as a survey line in the N-required panel
# and as its own curve in the Z_LR panel.  One constant, so they cannot drift.
N_EUCLID_MIN = 2_300

# Projected stream-catalogue sizes (order-of-magnitude; see calibration notes).
# Format: (label, N_streams, gray_level, lower_bound)
# lower_bound=True draws up arrows along the line (the count is a minimum).
# Ordered largest to smallest for readability.
SURVEYS = [
    # (r"Roman HLWA", 10_000, "0.15", False),
    (r"Starkman et al. 2026 Euclid Projection", N_EUCLID_MIN, "0.25", True),
    # (r"Rubin/LSST Y1", 500, "0.35", False),
    (r"Current (DES, SSLS)", 100, "0.45", False),
]

# Simulation-calibrated CDM convexity rate.
P_CDM_SIM = _lam_arc_cdm

# Fiducial observed-catalogue inference panel.
N_LR_VALUES = (100, 1_000, 10_000)
P_CDM_LR = P_CDM_SIM
P_SIDM_LR = _lam_arc_sidm

# Catalogue size for the worked example in sec:cosmology:excluded.
N_WORKED = 3_500
P_OBS_GRID = np.linspace(0.0, 0.02, N_PTS)

# Equal-favour observed rate: where Delta log L = 0 and the two models tie.
# Solving obs*log(p_C/p_S) + (1-obs)*log((1-p_C)/(1-p_S)) = 0 (independent of N).
_A_EQ = np.log(P_CDM_LR / P_SIDM_LR)
_B_EQ = np.log((1 - P_CDM_LR) / (1 - P_SIDM_LR))
RATE_EQUAL = float(-_B_EQ / (_A_EQ - _B_EQ))  # ~0.5%

LR_Y_MIN, LR_Y_MAX = -10.0, 10.0
LR_X_PAD = 0.35 * (P_CDM_LR - P_SIDM_LR)
LR_X_MIN = max(P_OBS_GRID[0], P_SIDM_LR - LR_X_PAD)
LR_X_MAX = min(P_OBS_GRID[-1], P_CDM_LR + LR_X_PAD)

# Model-region colours (match step_2_plot_triaxiality.py).
SIDM_COLOR = "#7DCEA0"
CDM_COLOR = "deepskyblue"


# Set True to print the exact-tail and sample-size checks quoted in the text.
PRINT_CHECKS = False

if PRINT_CHECKS:
    for _z in SIGMAS:
        _n_ex, _k = n_required_exact(_z, P_CDM_SIM, ETA_FID * P_CDM_SIM)
        _n_no = n_required(_z, P_CDM_SIM, ETA_FID * P_CDM_SIM)
        print(f"{_z}sigma: N_normal={_n_no:.0f} N_exact={_n_ex} K*={_k}")
    for _n in (100, 1000, 10000):
        print(report_exact_significances(n_streams=_n))


# ── Tick formatter: plain log labels (reused from ba_constraint.py) ───────────


def plain_log_labels(x, _pos):
    """Format log-axis ticks as plain decimal values."""
    if x >= 0.1:
        return f"{x:.1f}".rstrip("0").rstrip(".")
    return f"{x:.3f}".rstrip("0")


# ── Colors: gnuplot_r, darker = higher sigma (shared by both panels below) ───
_cmap = mpl.colormaps["gnuplot_r"]
# 1 sigma is a reference level rather than a result anyone would claim, so it
# is drawn in faint gray and the colormap is spent on the levels that matter.
# The ramp therefore spans the COLORED sigmas only, which puts the light end on
# 2 sigma.
SIGMA_GRAY_LEVEL = 1
GRAY_COLOR = "0.45"
GRAY_ALPHA = 0.30
_COLORED = [z for z in SIGMAS if z != SIGMA_GRAY_LEVEL]
# Map the sigma *value* (not its index) onto the colormap, so spacing in color
# reflects spacing in sigma: e.g. 5sigma sits further from 3sigma than 3 from 2.
_sigma_norm = mpl.colors.Normalize(vmin=min(_COLORED), vmax=max(_COLORED))


def sigma_color(z: float) -> tuple[float, float, float, float] | str:
    """Colormap color for sigma level ``z``; scales with the sigma value."""
    if z == SIGMA_GRAY_LEVEL:
        return GRAY_COLOR
    return _cmap(0.12 + 0.70 * _sigma_norm(z))


def sigma_alpha(z: float) -> float:
    """Opacity for sigma level ``z``; the gray reference level is faint."""
    return GRAY_ALPHA if z == SIGMA_GRAY_LEVEL else 1.0


COLORS = list(map(sigma_color, SIGMAS))


class BandLineHandle:
    """Proxy handle for a legend entry showing band + upper edge + fiducial line."""

    def __init__(self, color, band_alpha=0.15, line_lw=1.1, edge_lw=0.6, line_alpha=1.0) -> None:
        self.color = color
        self.band_alpha = band_alpha
        self.line_alpha = line_alpha
        self.line_lw = line_lw
        self.edge_lw = edge_lw


class BandLineLegendHandler(HandlerBase):
    """Draw a translucent band with dotted upper edge and solid center line."""

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        band_y = ydescent + 0.1 * height
        band_h = 0.8 * height
        edge_y = band_y + band_h
        line_y = ydescent + 0.5 * height

        band = Rectangle(
            (xdescent, band_y),
            width,
            band_h,
            facecolor=orig_handle.color,
            edgecolor="none",
            alpha=orig_handle.band_alpha,
            transform=trans,
        )
        upper_edge = Line2D(
            [xdescent, xdescent + width],
            [edge_y, edge_y],
            color=orig_handle.color,
            lw=orig_handle.edge_lw,
            ls=":",
            alpha=orig_handle.line_alpha,
            transform=trans,
        )
        fiducial = Line2D(
            [xdescent, xdescent + width],
            [line_y, line_y],
            color=orig_handle.color,
            lw=orig_handle.line_lw,
            alpha=orig_handle.line_alpha,
            transform=trans,
        )
        return [band, upper_edge, fiducial]


# ── Figures ───────────────────────────────────────────────────────────────────
# Two separate single-column figures rather than one two-panel figure: the
# required-N plot is a pre-survey forecast (sec:cosmology:sample_size) and the
# Z_LR plot is post-survey inference (sec:cosmology:favoured), used tens of
# paragraphs apart for different purposes.
fig_n, ax = plt.subplots(figsize=(3.7, 3.25))
fig_lr, ax_lr = plt.subplots(figsize=(3.4, 3.25))
legend_handles = []
legend_labels = []

for z, color in zip(SIGMAS, COLORS, strict=False):
    # p_SIDM = eta * p_CDM; valid for all p_CDM in grid and eta in (0, 1).
    N_lo = n_required(z, P_CDM_GRID, ETA_LO * P_CDM_GRID)
    N_fid = n_required(z, P_CDM_GRID, ETA_FID * P_CDM_GRID)
    N_hi = n_required(z, P_CDM_GRID, ETA_HI * P_CDM_GRID)

    _a = sigma_alpha(z)
    # N_lo < N_fid < N_hi (smaller eta -> larger Delta_p -> fewer streams).
    ax.fill_between(
        P_CDM_GRID,
        N_lo,
        N_hi,
        color=color,
        alpha=0.15 * _a,
        linewidth=0,
        zorder=max(SIGMAS) - z,
    )

    ax.plot(
        P_CDM_GRID,
        N_hi,
        color=color,
        lw=0.8,
        ls=":",
        alpha=_a,
        zorder=max(SIGMAS) - z,
    )

    ax.plot(
        P_CDM_GRID,
        N_fid,
        color=color,
        lw=1.6,
        alpha=_a,
    )
    legend_handles.append(
        BandLineHandle(color=color, band_alpha=0.15 * _a, line_alpha=_a),
    )
    legend_labels.append(rf"${z}\sigma$")

# ── Survey horizontal lines (all dashed) ─────────────────────────────────────
x_label_right = P_CDM_MAX * 0.97  # slightly inside right edge
x_label_left = P_CDM_MIN * 1.03  # slightly inside left edge
lower_bound_lines = []  # (label_artist, N_surv, gray) for up-arrow annotation
for name, N_surv, gray, lower_bound in SURVEYS:
    ax.axhline(N_surv, color=gray, lw=0.65, ls="--", zorder=10)
    # Lower-bound lines label on the left and carry up arrows; others on right.
    label = ax.text(
        x_label_left if lower_bound else x_label_right,
        N_surv * 1.12,
        name,
        fontsize=5.5,
        fontweight="semibold",
        color=f"{float(gray) * 0.75}",
        ha="left" if lower_bound else "right",
        va="bottom",
        zorder=10,
    )
    if lower_bound:
        lower_bound_lines.append((label, N_surv, gray))

# ── Simulated p_CDM vertical line ─────────────────────────────────────────────
ax.axvline(P_CDM_SIM, color="0.3", lw=0.9, ls="-.", zorder=5)
ax.text(
    P_CDM_SIM / 1.08,
    0.97,
    r"$\lambda_{\rm C}^{\rm sim}$",
    fontsize=7,
    color="0.3",
    ha="right",
    va="top",
    transform=ax.get_xaxis_transform(),
)

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(P_CDM_MIN, P_CDM_MAX)
ax.set_ylim(1e2, 5e5)
ax.set_xlabel(r"Model rate $\lambda_{\rm C}$", labelpad=3)
ax.set_ylabel(r"Number of Streams", labelpad=3)

ax.xaxis.set_major_locator(LogLocator(subs=[1, 2, 5], numticks=20))
ax.xaxis.set_major_formatter(
    mpl.ticker.FuncFormatter(lambda x, _: f"{x * 100:g}%"),
)
ax.xaxis.set_minor_locator(LogLocator(subs=np.arange(2, 10) * 0.1, numticks=20))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_locator(LogLocator(subs=np.arange(2, 10) * 0.1, numticks=20))
ax.yaxis.set_minor_formatter(NullFormatter())

ax.grid(True, which="major", ls=":", lw=0.35, alpha=0.40, color="0.5")

# ── Up arrows on lower-bound survey lines (allowed region is above) ───────────
# Drawn after the axis scales/limits are final so the text extent is measured in
# the correct (log) coordinates; arrows start just past the label's right edge.
fig_n.canvas.draw()
for label, N_surv, gray in lower_bound_lines:
    bbox = label.get_window_extent()
    x_text_right = ax.transData.inverted().transform((bbox.x1, bbox.y0))[0]
    x_arrow_start = x_text_right * 1.25
    x_arrow_end = P_CDM_MAX * 0.95
    if x_arrow_start < x_arrow_end:
        for x_arrow in np.logspace(
            np.log10(x_arrow_start),
            np.log10(x_arrow_end),
            5,
        ):
            ax.annotate(
                "",
                xy=(x_arrow, N_surv * 1.65),
                xytext=(x_arrow, N_surv),
                arrowprops={
                    "arrowstyle": "-|>,head_width=0.25,head_length=0.5",
                    "color": gray,
                    "lw": 0.45,
                    "alpha": 0.5,
                },
                zorder=10,
            )

# ── Legend ────────────────────────────────────────────────────────────────────
leg = ax.legend(
    legend_handles,
    legend_labels,
    title=r"2 Hyp. Test",
    loc="upper right",
    ncol=2,
    framealpha=0.90,
    edgecolor="0.7",
    handlelength=1.2,
    handletextpad=0.4,
    borderpad=0.3,
    labelspacing=0.3,
    columnspacing=0.7,
    handler_map={BandLineHandle: BandLineLegendHandler()},
)
leg.get_title().set_fontsize(7)

# ── Annotation: band definition ───────────────────────────────────────────────
# Build a composite widget: fraction text | band drawing | stacked labels.
# The band mirrors BandLineLegendHandler: shaded fill + dotted top edge +
# solid centre line (at the visual midpoint, labelled 0.4 = (0.1+0.7)/2).
_ann_s = 0.7  # global scale factor for the annotation widget
_ann_band_w = 22 * _ann_s  # width of the drawn band graphic
_ann_da_w = 44 * _ann_s  # total DA width: band + gap + label text
_ann_da_h = 44 * _ann_s
_ann_by0 = 0.10 * _ann_da_h  # band bottom y in DrawingArea coords
_ann_bh = 0.80 * _ann_da_h  # band height
_ann_col = "0.40"

_ann_da = DrawingArea(_ann_da_w, _ann_da_h)
_ann_da.add_artist(
    Rectangle(
        (0, _ann_by0),
        _ann_band_w,
        _ann_bh,
        facecolor=_ann_col,
        alpha=0.15,
        edgecolor="none",
    ),
)
_ann_da.add_artist(
    Line2D(
        [0, _ann_band_w],
        [_ann_by0 + _ann_bh] * 2,
        color=_ann_col,
        lw=0.8 * _ann_s,
        ls=":",
    ),
)
_ann_da.add_artist(
    Line2D(
        [0, _ann_band_w],
        [_ann_by0 + (ETA_FID - ETA_LO) / (ETA_HI - ETA_LO) * _ann_bh] * 2,
        color=_ann_col,
        lw=1.6 * _ann_s,
    ),
)

# Labels at exact band y-positions, inside the (wider) DrawingArea.
_ann_x_lbl = _ann_band_w + 2 * _ann_s
for _y_lbl, _lbl_txt in [
    (_ann_by0 + _ann_bh, f"{ETA_HI:g}"),
    (_ann_by0 + (ETA_FID - ETA_LO) / (ETA_HI - ETA_LO) * _ann_bh, f"{ETA_FID:.2f}"),
    (_ann_by0, f"{ETA_LO:g}"),
]:
    _ann_da.add_artist(
        _AnnText(
            x=_ann_x_lbl,
            y=_y_lbl,
            text=_lbl_txt,
            va="center",
            fontsize=7.5 * _ann_s,
        ),
    )

_ann_hbox = HPacker(
    children=[
        TextArea(
            r"$\dfrac{\lambda_{\rm SI}}{\lambda_{\rm C}}\!\in$",
            textprops=dict(size=10 * _ann_s),
        ),
        _ann_da,
    ],
    pad=0,
    sep=4 * _ann_s,
    align="center",
)

_ann_box = AnchoredOffsetbox(
    loc="lower left",
    child=_ann_hbox,
    pad=0.3,
    frameon=True,
    bbox_to_anchor=(0.03, 0.03),
    bbox_transform=ax.transAxes,
    borderpad=0.3,
)
_ann_box.patch.set(facecolor="white", edgecolor="none", alpha=0.7)
ax.add_artist(_ann_box)

# ── Model-favour background: SIDM lower-left, CDM upper-right ────────────────
# Quadrants, not bands.  Each model is favoured only where a LOW (high) rate
# actually produces a negative (positive) Z_LR, so the shading is the
# intersection of the two conditions, meeting at RATE_EQUAL on the zero line.
for _x0, _x1, _y0, _y1, _c in [
    (LR_X_MIN, RATE_EQUAL, LR_Y_MIN, 0.0, SIDM_COLOR),
    (RATE_EQUAL, LR_X_MAX, 0.0, LR_Y_MAX, CDM_COLOR),
]:
    ax_lr.add_patch(
        Rectangle(
            (_x0, _y0),
            _x1 - _x0,
            _y1 - _y0,
            facecolor=_c,
            edgecolor="none",
            alpha=0.20,
            zorder=0,
        ),
    )

ax_lr.axhline(0, color="0.25", lw=0.8, zorder=1)
for _z in [1, 2, 3, 5, 10]:
    _alpha = 0.18 if _z < 5 else 0.28
    ax_lr.axhline(_z, color="0.55", lw=0.55, ls=":", alpha=_alpha, zorder=1)
    ax_lr.axhline(-_z, color="0.55", lw=0.55, ls=":", alpha=_alpha, zorder=1)

# Format: (N_streams, linestyle, linewidth, x_label, label).
_lr_line_specs = [
    (100, "-", 2.0, 0.0140, r"$N=10^2$"),
    (1_000, "-", 2.0, 0.0125, r"$N=10^3$"),
    (N_EUCLID_MIN, "--", 2.0, 0.0103, r"S+26-Euclid-minimum"),
    (10_000, "-", 2.0, 0.0113, r"$N=10^4$"),
]

_lr_log_n = np.log10([spec[0] for spec in _lr_line_specs])
_lr_log_n_min = float(_lr_log_n.min())
_lr_log_n_span = float(_lr_log_n.max() - _lr_log_n_min)


def _lr_log_n_frac(n_streams: int) -> float:
    """Normalized log10(N_streams) coordinate on [0, 1]."""
    if _lr_log_n_span == 0:
        return 1.0
    return float((np.log10(n_streams) - _lr_log_n_min) / _lr_log_n_span)


def lr_curve_color(n_streams: int) -> str:
    """Gray-to-black ramp keyed to log10(N_streams)."""
    frac = _lr_log_n_frac(n_streams)
    gray = 0.42 * (1 - frac)
    return f"{gray:.3f}"


def lr_curve_linewidth(base_lw: float, n_streams: int) -> float:
    """Slightly widen larger-N curves using log10(N_streams)."""
    frac = _lr_log_n_frac(n_streams)
    return base_lw * (1.0 + 0.22 * frac)


def plot_lr_curve(ax, x, y, lw, ls, color, zorder, dash=None):
    """Plot a likelihood-ratio curve with a uniform grayscale stroke.

    For dashed curves, drop alternating short segments so the dash pattern is
    preserved across the line collection.
    """
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    if dash is not None:
        on, off = dash
        keep = (np.arange(len(segments)) % (on + off)) < on
        segments = segments[keep]
    lc = LineCollection(
        segments,
        colors=[color],
        linewidths=lw,
        linestyles="-" if dash is not None else ls,
        zorder=zorder,
    )
    ax.add_collection(lc)
    return lc


# Draw the likelihood-ratio curves as plain black lines.
for _n_streams, _ls, _lw, _x_label, _lbl in _lr_line_specs:
    _z_lr = signed_lr_sigma(P_OBS_GRID, P_CDM_LR, P_SIDM_LR, _n_streams)
    _dash = None if _ls == "-" else (14, 10)
    _curve_color = lr_curve_color(_n_streams)
    _curve_lw = lr_curve_linewidth(_lw, _n_streams)
    plot_lr_curve(
        ax_lr,
        P_OBS_GRID,
        _z_lr,
        _curve_lw,
        _ls,
        _curve_color,
        zorder=3,
        dash=_dash,
    )

ax_lr.axvline(P_SIDM_LR, color="0.40", lw=0.9, ls="--", zorder=2)
ax_lr.axvline(P_CDM_LR, color="0.20", lw=0.9, ls="-.", zorder=2)
for _xref, _txt, _col in [
    (P_SIDM_LR, r"$\lambda_{\rm SI}^{\rm sim}$", "0.35"),
    (P_CDM_LR, r"$\lambda_{\rm C}^{\rm sim}$", "0.25"),
]:
    ax_lr.text(
        _xref,
        0.06,
        _txt,
        fontsize=7,
        color=_col,
        ha="right",
        va="bottom",
        rotation=90,
        transform=ax_lr.get_xaxis_transform(),
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
        zorder=4,
    )

ax_lr.set_xlim(LR_X_MIN, LR_X_MAX)
ax_lr.set_ylim(LR_Y_MIN, LR_Y_MAX)
ax_lr.set_xlabel(r"Observed rate $\hat{\lambda}_{\rm obs}=K/N$", labelpad=3)
# Not "Significance": Wilks does not apply to these two non-nested,
# a-priori-fixed models, so Z_LR is a relabeling of Dln(L) for the axis
# range and not a sigma count.  See the text around eq:cosmology:Z_LR.
ax_lr.set_ylabel(r"Likelihood ratio $Z_{\rm LR}$", labelpad=0)

# ── Model labels, one inside each shaded quadrant ────────────────────────────
# A Text bbox= wraps the font's LAYOUT box, which reserves ascent and descent
# the glyphs do not use: these labels are all caps, so the ink is ~6.7pt inside
# a ~9.4pt box and rides visibly high.  get_text_width_height_descent reports
# descent 0 here and cannot correct it, so measure the glyph outline with
# TextPath and build the plate around that.  Everything is in axes fractions,
# which survive savefig's bbox_inches='tight' re-render.
_QUAD_FS = 9.0
_QUAD_PAD_PT = 4.0
_QUAD_GAP = 0.02  # axes fractions, from the quadrant corner

_ax_bb = ax_lr.get_window_extent()
_pt_to_ax_x = (fig_lr.dpi / 72.0) / _ax_bb.width
_pt_to_ax_y = (fig_lr.dpi / 72.0) / _ax_bb.height
_x_equal_ax = ax_lr.transAxes.inverted().transform(
    ax_lr.transData.transform((RATE_EQUAL, 0.0)),
)[0]

for _txt, _col, _side in [("SIDM", SIDM_COLOR, "lower"), ("CDM", CDM_COLOR, "upper")]:
    _prop = FontProperties(size=_QUAD_FS, weight="bold")
    _ink = TextPath((0, 0), _txt, size=_QUAD_FS, prop=_prop).get_extents()
    _bw = (_ink.width + 2 * _QUAD_PAD_PT) * _pt_to_ax_x
    _bh = (_ink.height + 2 * _QUAD_PAD_PT) * _pt_to_ax_y
    # SIDM occupies the lower-left quadrant, labelled at its lower right;
    # CDM the upper-right quadrant, labelled at its upper left.
    if _side == "lower":
        _bx, _by = _x_equal_ax - _QUAD_GAP - _bw, _QUAD_GAP
    else:
        _bx, _by = _x_equal_ax + _QUAD_GAP, 1.0 - _QUAD_GAP - _bh
    ax_lr.add_patch(
        FancyBboxPatch(
            (_bx, _by),
            _bw,
            _bh,
            boxstyle="round,pad=0,rounding_size=0.018",
            transform=ax_lr.transAxes,
            facecolor="white",
            edgecolor="none",
            alpha=0.55,
            zorder=5,
        ),
    )
    # Baseline-left anchored, so the glyph origin is unambiguous and the ink
    # lands exactly one pad inside the plate on every side.
    ax_lr.text(
        _bx + _QUAD_PAD_PT * _pt_to_ax_x - _ink.x0 * _pt_to_ax_x,
        _by + _QUAD_PAD_PT * _pt_to_ax_y - _ink.y0 * _pt_to_ax_y,
        _txt,
        transform=ax_lr.transAxes,
        ha="left",
        va="baseline",
        fontsize=_QUAD_FS,
        fontweight="bold",
        color=_col,
        zorder=6,
    )

ax_lr.xaxis.set_major_formatter(
    mpl.ticker.FuncFormatter(lambda x, _: f"{x * 100:g}%"),
)
ax_lr.grid(True, which="major", ls=":", lw=0.35, alpha=0.40, color="0.5")


def _curve_chord_angle(x_start: float, x_end: float, n_streams: int) -> float:
    """Angle (deg, display coords) of the curve chord from ``x_start`` to ``x_end``.

    Using the chord across the label's full horizontal extent averages the
    curve orientation over the whole label, rather than the slope at one point.
    """
    y0 = signed_lr_sigma(x_start, P_CDM_LR, P_SIDM_LR, n_streams)
    y1 = signed_lr_sigma(x_end, P_CDM_LR, P_SIDM_LR, n_streams)
    p0 = ax_lr.transData.transform((x_start, y0))
    p1 = ax_lr.transData.transform((x_end, y1))
    return float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))


# Place the curve labels horizontally first, then rotate each to the chord angle
# across its own measured width so the orientation is averaged over the label.
_curve_labels = []
for _n_streams, _ls, _lw, _x_label, _lbl in _lr_line_specs:
    _y_label = signed_lr_sigma(_x_label, P_CDM_LR, P_SIDM_LR, _n_streams)
    _curve_color = lr_curve_color(_n_streams)
    _txt = ax_lr.text(
        _x_label,
        _y_label,
        _lbl,
        fontsize=7,
        color=_curve_color,
        ha="left",
        va="center",
        rotation=0,
        rotation_mode="default",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=0.3),
        zorder=4,
    )
    _curve_labels.append((_txt, _x_label, _n_streams))

fig_lr.canvas.draw()
for _txt, _x_label, _n_streams in _curve_labels:
    _bb = _txt.get_window_extent()  # horizontal extent (rotation still 0)
    _x_end = ax_lr.transData.inverted().transform((_bb.x1, _bb.y0))[0]
    _x_end = min(P_OBS_GRID[-1], _x_end)
    # Anchor at the label's centre on the curve so it sits balanced (symmetric
    # about its midpoint), and orient by the chord across its full width.
    _x_c = 0.5 * (_x_label + _x_end)
    _y_c = signed_lr_sigma(_x_c, P_CDM_LR, P_SIDM_LR, _n_streams)
    _txt.set_ha("center")
    _txt.set_position((_x_c, _y_c))
    _txt.set_rotation(_curve_chord_angle(_x_label, _x_end, _n_streams))

# ── Save ──────────────────────────────────────────────────────────────────────
# Panel labels are gone with the split: each figure stands alone.
out_dir = paths.FIGURES / "cosmology"
out_dir.mkdir(parents=True, exist_ok=True)

for _fig, _name in ((fig_n, "streams_required"), (fig_lr, "likelihood_ratio")):
    _path = out_dir / f"{_name}.pdf"
    _fig.savefig(_path, dpi=200, bbox_inches="tight", pad_inches=0.02, backend="pdf")
    print(f"Saved {_path}")  # noqa: T201
    plt.close(_fig)

# ── Numbers the manuscript quotes ─────────────────────────────────────────────
# The optimistic forecast uses the measured detection rates rather than the
# fiducial pair, so we read them from the script that measures them.

# Worked catalogue of sec:cosmology:excluded, at the measured arc-length floor
# rather than the fiducial pair: the text quotes it as the conservative case.
_worked = report_exact_significances(
    p_cdm=_lam_arc_cdm,
    p_sidm=_lam_arc_sidm,
    n_streams=N_WORKED,
)
# The N_sigma that eq:cosmology:N_required_full assigns to the same catalogue,
# by inverting N >= N_sigma^2 * coefficient.
_coeff = n_required(1.0, _lam_arc_cdm, _lam_arc_sidm)
_worked["n_sigma_balanced"] = float(np.sqrt(N_WORKED / _coeff))

# Values this script owns that the manuscript quotes; see
# src/scripts/make_variables.py.
write_values(
    __file__,
    {
        "n_euclid_min": N_EUCLID_MIN,
        "eta_lo": ETA_LO,
        "eta_fid": ETA_FID,
        "eta_hi": ETA_HI,
        # The SIDM region is drawn for sigma/m >~ a few cm^2/g, so ETA_FID *is*
        # that regime's rate ratio.  The manuscript quotes it as an upper bound
        # on eta ("the eta <~ X part of the figure"), so round UP to the next
        # 0.1, which keeps the bound true for any ETA_FID.
        "eta_sidm_regime": float(np.ceil(ETA_FID * 10.0) / 10.0),
        "lambda_cdm_sim": P_CDM_SIM,
        "lambda_sidm_sim": ETA_FID * P_CDM_SIM,
        "catalogue_sizes": [100, 1000, 10000],
        "sigmas": SIGMAS,
        # Required catalogue size at each significance, fiducial rates.
        "n_required_fiducial": [
            float(n_required(z, P_CDM_LR, P_SIDM_LR)) for z in SIGMAS
        ],
        # ... and if the measured detection rates are achievable instead.
        "n_required_optimistic": [
            float(n_required(z, _lam_det_cdm, _lam_det_sidm)) for z in SIGMAS
        ],
        "lambda_det_over_fiducial": _lam_det_cdm / P_CDM_LR,
        # The detection rates have their own suppression ratio, close
        # to but not equal to the arc-length one.
        "eta_det": _lam_det_sidm / _lam_det_cdm,
        # Exact binomial sample size, for validating the normal form.  This is
        # the settled requirement, not the first tooth (see n_required_exact).
        "n_required_exact": [
            n_required_exact(z, P_CDM_LR, P_SIDM_LR)[0] for z in SIGMAS
        ],
        # Significance a Local-Group-sized catalogue reaches today.  N scales
        # as N_sigma^2, so N_sigma = sqrt(N / N_1sigma).  N_LOCAL_GROUP is the
        # top of the "N ~ 30-100" range the manuscript quotes; keep the two in
        # step if either moves.
        "n_sigma_local_group": float(
            np.sqrt(N_LOCAL_GROUP / n_required(1, P_CDM_LR, P_SIDM_LR)),
        ),
        # Spread of the 3-sigma requirement across the eta band.
        "n_required_3sigma_eta_lo": float(n_required(3, P_CDM_SIM, ETA_LO * P_CDM_SIM)),
        "n_required_3sigma_eta_hi": float(n_required(3, P_CDM_SIM, ETA_HI * P_CDM_SIM)),
        "worked_example": _worked,
    },
)
