"""Presentation vocabulary for the ridgeline sky figures.

Constants, literal option types, and the style dataclasses that the plotting
functions in ``viz_ridgeline`` take as arguments. Nothing here draws.
"""

from dataclasses import dataclass

from typing import Literal

from ridgeline_data import RidgelineCase

CB_LABEL = "Non-concave / total arclength"
MOLL_TITLE = "Viewing angles"
MOLLWEIDE_DATA_ASPECT = 2.0  # lon span / lat span in projection data coords
MollweideLayerOrder = Literal["default", "reversed"]
MollweideDisplayMode = Literal["overlay", "hemisphere_split"]
MollweideGCPlane = Literal["mean_L", "pca"]
SkyPanelInfoPosition = Literal[
    "upper right",
    "lower right",
    "upper left",
    "lower left",
]


@dataclass
class MollweideLayer:
    """One galaxy / case layer on the Mollweide map."""

    case: RidgelineCase | None
    cmap: str = "viridis"
    badge: str = "1"
    overlay_alpha: float = 1.0
    mask_threshold: float = 0.05  # hide overlay where fraction ≤ this
    flip_y: bool = False  # mirror heatmap + great circle: lat → -lat (equator flip)


@dataclass
class SkyViewSpec:
    """One sky-density panel: case + viewing direction.

    Set ``xlim`` / ``ylim`` to ``(lo, hi)`` in kpc for manual axis ranges; leave
    both ``None`` for the default origin-centred square limits from particles + track.

    When the x and y spans differ, the sky panel stays a square on screen but uses
    independent kpc-per-pixel scaling on each axis (see ``apply_sky_panel_aspect``).
    """

    case: RidgelineCase
    theta: float
    phi: float
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None


@dataclass
class SkyDensityStyle:
    """Hyperparameters for sky-plane particle-density ``hist2d``."""

    cmap: str = "Blues"
    alpha: float = 1.0
    cmin: float = 1.0
    # Fixed bin count (used when adaptive_bin_width_kpc is None)
    bins: int = 90
    # If set, each panel picks n_bins ≈ span / width from its own x/y limits [kpc]
    adaptive_bin_width_kpc: float | None = None
    vmax_percentile: float = 96.0
    norm_gamma: float | None = 0.5
    use_lognorm: bool = False


@dataclass
class SkyTrackStyle:
    """Ridgeline styling for one sky-density panel (line only, no scatter markers)."""

    track_color_good: str = "crimson"
    track_color_bad: str = "red"
    track_lw_good: float = 2.5
    track_lw_bad: float = 7.0
    track_alpha_good: float = 0.95
    track_alpha_bad: float = 0.55


@dataclass
class MollweideViewMarkerStyle:
    """Marker on the Mollweide map for one sky-panel viewing direction."""

    color: str = "white"
    marker: str = "*"
    size: float = 14.0
    alpha: float = 1.0
    edgecolor: str = "black"
    edgewidth: float = 0.6
    # None → ``{index}: $(l,b)=(l°, b°)$``; supports {index},{l},{b},{theta},{phi}
    text_template: str | None = None
    text_bbox_alpha: float = 0.85
    text_bbox_facecolor: str | None = None  # None → use marker color
    text_offset: tuple[float, float] | None = None  # None → auto per index
    # Anchor of the label relative to the offset point.  Centered, so the
    # offset translates the label directly rather than the anchor flipping.
    text_ha: Literal["left", "center", "right"] = "center"
    text_va: Literal["top", "center", "bottom", "baseline"] = "center"
    zorder_marker: int = 30
    zorder_text: int = 32


@dataclass
class FigureStyle:
    """Central registry for every font-size and padding knob.

    Pass an instance to plotting functions; individual plotting helpers accept
    it via the ``fstyle`` keyword.
    """

    # Sky panel
    axis_label_fontsize: float = 15.0
    tick_label_fontsize: float = 15.0
    badge_fontsize: float = 13.0
    info_fontsize: float = 11.0
    grid_alpha: float = 0.3

    # Mollweide
    moll_title_fontsize: float = 13.0
    moll_lon_tick_fontsize: float = 15.0
    moll_lat_tick_fontsize: float = 15.0
    moll_grid_alpha: float = 0.35

    # Colorbars
    cb_tick_fontsize: float = 14.0
    cb_label_fontsize: float = 14.0
    cb_badge_fontsize: float = 13.0

    # View-marker on Mollweide
    marker_text_fontsize: float = 8.0


@dataclass
class SkyPanelInfo:
    """Content for the step_2-style info box on a sky panel.

    Set *r_min* / *r_max* to ``None`` to suppress the *r* line.
    *(l, b)* is auto-filled from the panel viewing angles when omitted.
    Leave *pct_bad* as ``None`` (default) to suppress the bad-curvature line;
    set it explicitly (e.g. from ``ridgeline_sky_stats``) to show
    ``κ_{<0}=...%``.
    """

    r_min: float | None = None
    r_max: float | None = None
    l_deg: float | None = None
    b_deg: float | None = None
    pct_bad: float | None = None


