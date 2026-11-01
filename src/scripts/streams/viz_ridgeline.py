"""Sky-density and Mollweide plotting functions for the ridgeline figures.

Draws on the style vocabulary in ``sky_style`` and the geometry/data helpers
in ``sky_geometry`` and ``ridgeline_data``; this module only draws.
"""

from typing import Any, Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LogNorm, Normalize, PowerNorm
from ridgeline_data import (
    RidgelineCase,
    load_data,
    mollweide_fraction_grid,
    ridgeline_sky_stats,
    worst_viewing_angle,
)
from sky_geometry import (
    format_lb_latex,
    normal_to_plane_basis,
    principal_plane_great_circle,
    project_to_sky,
    resolve_sky_panel_limits,
    split_mollweide_segments,
    symmetric_limits_around_origin,
    to_mollweide_coords,
    viewing_angles_lb_deg,
)
from sky_style import (
    CB_LABEL,
    MOLL_TITLE,
    FigureStyle,
    MollweideLayer,
    MollweideViewMarkerStyle,
    SkyDensityStyle,
    SkyPanelInfo,
    SkyPanelInfoPosition,
    SkyTrackStyle,
    SkyViewSpec,
)

from stream_convexity_paper.track import Track3D

__all__ = [
    "FigureStyle",
    "MollweideLayer",
    "MollweideViewMarkerStyle",
    "SkyDensityStyle",
    "SkyPanelInfo",
    "SkyTrackStyle",
    "SkyViewSpec",
    "add_dual_colorbars",
    "load_data",
    "plot_mollweide_map",
    "plot_sky_density_panel",
    "viewing_angles_lb_deg",
    "worst_viewing_angle",
]

# ════════════════════════════════════════════════════════════════════════════
# Sky Panel Annotation & Styling
# ════════════════════════════════════════════════════════════════════════════


_SKY_PANEL_INFO_LOC: dict[
    SkyPanelInfoPosition,
    tuple[float, float, Literal["left", "right"], Literal["top", "bottom"]],
] = {
    "upper right": (0.95, 0.95, "right", "top"),
    "lower right": (0.95, 0.05, "right", "bottom"),
    "upper left": (0.05, 0.95, "left", "top"),
    "lower left": (0.05, 0.05, "left", "bottom"),
}


def _annotate_sky_panel_info(
    ax: plt.Axes,
    info: SkyPanelInfo,
    *,
    position: SkyPanelInfoPosition = "lower right",
    fstyle: FigureStyle,
) -> None:
    """Info box with optional *r* range, *(l, b)*, and bad-curvature %.

    *position* uses matplotlib-style corner names (``upper right``, etc.);
    coordinates are fixed fractions in axes space via :data:`_SKY_PANEL_INFO_LOC`.
    """
    lines = [
        rf"$(l,b)=({info.l_deg:.0f}°, {info.b_deg:.0f}°)$",
    ]
    if info.pct_bad is not None:
        lines.append(rf"$\kappa_{{<0}}={info.pct_bad:.1f}\%$")
    text = "\n".join(lines)

    x, y, ha, va = _SKY_PANEL_INFO_LOC[position]

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fstyle.info_fontsize,
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor="gray",
            alpha=0.8,
        ),
        zorder=20,
    )


# ════════════════════════════════════════════════════════════════════════════
# Plotting Primitives
# ════════════════════════════════════════════════════════════════════════════


def _plot_track_with_bad_segments(
    ax: plt.Axes,
    xy: np.ndarray,
    is_bad: np.ndarray,
    *,
    style: SkyTrackStyle | None = None,
) -> None:
    """Draw ridgeline; bad-normal segments as separate-colour overlay."""
    st = style or SkyTrackStyle()
    x, y = xy[:, 0], xy[:, 1]
    ax.plot(
        np.where(~is_bad, x, np.nan),
        np.where(~is_bad, y, np.nan),
        lw=st.track_lw_good,
        color=st.track_color_good,
        alpha=st.track_alpha_good,
        solid_capstyle="round",
        zorder=9,
    )
    ax.plot(
        np.where(is_bad, x, np.nan),
        np.where(is_bad, y, np.nan),
        lw=st.track_lw_bad,
        color=st.track_color_bad,
        alpha=st.track_alpha_bad,
        solid_capstyle="round",
        zorder=11,
    )


def plot_galaxy_center(ax: plt.Axes) -> None:
    """Mark galactic centre at the sky-plane origin."""
    ax.plot(
        0.0,
        0.0,
        marker="x",
        color="black",
        markersize=9,
        markeredgewidth=2.0,
        zorder=15,
        clip_on=False,
    )


def _resolve_hist_bins(
    style: SkyDensityStyle,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> int:
    """Bin count: fixed ``bins`` or adaptive from panel spatial span."""
    if style.adaptive_bin_width_kpc is not None:
        span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        return max(int(np.ceil(span / style.adaptive_bin_width_kpc)), 10)
    return style.bins


def _density_norm(
    counts: np.ndarray,
    style: SkyDensityStyle,
) -> Normalize | LogNorm | PowerNorm:
    """Build a colour normalizer from histogram counts."""
    positive = counts[counts > 0]
    if positive.size == 0:
        vmax = style.cmin + 1.0
    else:
        vmax = float(np.percentile(positive, style.vmax_percentile))
        vmax = max(vmax, style.cmin + 1.0)

    if style.use_lognorm:
        return LogNorm(vmin=style.cmin, vmax=vmax)
    if style.norm_gamma is not None:
        return PowerNorm(gamma=style.norm_gamma, vmin=style.cmin, vmax=vmax)
    return Normalize(vmin=style.cmin, vmax=vmax)


def plot_sky_density_panel(
    ax: plt.Axes,
    particles_3d: np.ndarray,
    track: Track3D,
    theta: float,
    phi: float,
    *,
    panel: str = "top",
    panel_badge: str = "Case A",
    density_style: SkyDensityStyle | None = None,
    track_style: SkyTrackStyle | None = None,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    max_particles: int | None = 100_000,
    rng_seed: int = 42,
    panel_info: SkyPanelInfo | None = None,
    info_position: SkyPanelInfoPosition = "lower right",
    fstyle: FigureStyle,
) -> dict[str, Any]:
    """Sky-plane density + ridgeline, annotated in step_2 style.

    Draws a particle-density heatmap (``density_style.cmap``) with the
    ridgeline overplotted on top: ``track_color_good`` where the projected
    curvature normal is well-behaved, switching to the thicker
    ``track_color_bad`` where it flags as non-concave. A black "x" marks the
    galactic centre, a bold panel-badge label sits upper-left, and an info
    box (default lower-right) shows *(l, b)* from the viewing angles; optional
    *r* range and ``κ_{<0}`` come from *panel_info* when set (``pct_bad`` is
    never auto-filled).

    Returns a dict with ``sky_stats``, ``xlim``, ``ylim``, etc.
    """
    style = density_style or SkyDensityStyle()
    idx = np.arange(len(particles_3d))
    if max_particles is not None and len(idx) > max_particles:
        rng = np.random.default_rng(rng_seed)
        idx = np.sort(rng.choice(idx, max_particles, replace=False))

    sky = project_to_sky(particles_3d[idx], theta, phi)
    stats = ridgeline_sky_stats(track, theta, phi)

    all_x = np.concatenate([sky[:, 0], stats["positions_2d"][:, 0]])
    all_y = np.concatenate([sky[:, 1], stats["positions_2d"][:, 1]])
    auto_xlim, auto_ylim = symmetric_limits_around_origin(all_x, all_y)
    xlim, ylim = resolve_sky_panel_limits(xlim, ylim, auto_xlim, auto_ylim)
    n_bins = _resolve_hist_bins(style, xlim, ylim)

    counts, _, _ = np.histogram2d(
        sky[:, 0],
        sky[:, 1],
        bins=n_bins,
        range=[xlim, ylim],
    )
    norm = _density_norm(counts, style)

    _, _, _, im = ax.hist2d(
        sky[:, 0],
        sky[:, 1],
        bins=n_bins,
        range=[xlim, ylim],
        cmap=style.cmap,
        norm=norm,
        cmin=style.cmin,
        alpha=style.alpha,
        zorder=1,
    )
    im.set_rasterized(True)

    track_st = track_style or SkyTrackStyle()
    _plot_track_with_bad_segments(
        ax,
        stats["positions_2d"],
        stats["is_bad_normal"],
        style=track_st,
    )

    plot_galaxy_center(ax)

    # ── step_2 style annotations ──────────────────────────────────────────
    ax.text(
        0.05,
        0.95,
        panel_badge,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fstyle.badge_fontsize,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "gray"},
        zorder=20,
    )

    # Info box: auto-fill (l, b); pct_bad only when caller sets it
    if panel_info is None:
        l_deg, b_deg = viewing_angles_lb_deg(theta, phi)
        panel_info = SkyPanelInfo(l_deg=l_deg, b_deg=b_deg)
    else:
        l_deg = panel_info.l_deg
        b_deg = panel_info.b_deg
        if l_deg is None or b_deg is None:
            auto_l, auto_b = viewing_angles_lb_deg(theta, phi)
            if l_deg is None:
                l_deg = auto_l
            if b_deg is None:
                b_deg = auto_b
        panel_info = SkyPanelInfo(
            r_min=panel_info.r_min,
            r_max=panel_info.r_max,
            l_deg=l_deg,
            b_deg=b_deg,
            pct_bad=panel_info.pct_bad,
        )
    _annotate_sky_panel_info(ax, panel_info, position=info_position, fstyle=fstyle)

    # ── Axis styling ──────────────────────────────────────────────────────

    ax.set(xlim=xlim, ylim=ylim)
    ax.set_box_aspect(1)
    ax.set_aspect("auto")
    ax.grid(True, alpha=fstyle.grid_alpha)

    if panel == "top":
        ax.set_xlabel("")
        ax.set_ylabel(r"$y_\mathrm{sky}$ [kpc]", fontsize=fstyle.axis_label_fontsize)
        ax.tick_params(axis="both", labelsize=fstyle.tick_label_fontsize)
        ax.tick_params(
            axis="x",
            bottom=False,
            top=True,
            labeltop=True,
            labelbottom=False,
        )

    else:
        ax.set_xlabel(r"$x_\mathrm{sky}$ [kpc]", fontsize=fstyle.axis_label_fontsize)
        ax.set_ylabel(r"$y_\mathrm{sky}$ [kpc]", fontsize=fstyle.axis_label_fontsize)
        ax.tick_params(axis="both", labelsize=fstyle.tick_label_fontsize)

    return {
        "sky_stats": stats,
        "sky_particles": sky,
        "xlim": xlim,
        "ylim": ylim,
        "auto_xlim": auto_xlim,
        "auto_ylim": auto_ylim,
    }


# ════════════════════════════════════════════════════════════════════════════
# Mollweide Map
# ════════════════════════════════════════════════════════════════════════════


def _plot_great_circle(
    ax: plt.Axes,
    case: RidgelineCase,
    *,
    color: str = "white",
    ls: str = ":",
    lw: float = 1.4,
    alpha: float = 0.85,
    zorder: int = 20,
    flip_y: bool = False,
) -> None:
    """Orbital-plane great circle for one case.

    Uses the case's mean angular-momentum axis to define the plane when it is
    stored in ``case.meta``, else falls back to the saved PCA principal-plane
    basis.
    """
    if "L_mean_hat" in case.meta:
        L_hat = np.asarray(case.meta["L_mean_hat"], dtype=float)
        v1, v2 = normal_to_plane_basis(L_hat)
    else:
        v1, v2 = case.arrays["v1_pca"], case.arrays["v2_pca"]
    theta_gc, phi_gc = principal_plane_great_circle(v1, v2)
    lon_gc, lat_gc = to_mollweide_coords(theta_gc, phi_gc)
    if flip_y:
        lat_gc = -lat_gc
    for lseg, bseg in split_mollweide_segments(lon_gc, lat_gc):
        ax.plot(
            lseg,
            bseg,
            color=color,
            lw=lw,
            ls=ls,
            alpha=alpha,
            zorder=zorder,
        )


def _marker_label_text(
    style: MollweideViewMarkerStyle,
    index: int,
    theta: float,
    phi: float,
) -> str:
    l_deg, b_deg = viewing_angles_lb_deg(theta, phi)
    if style.text_template is not None:
        return style.text_template.format(
            index=index,
            l=l_deg,
            b=b_deg,
            theta=float(np.degrees(theta)),
            phi=float(np.degrees(phi)),
        )
    return format_lb_latex(theta, phi, index=index)


def _plot_view_markers(
    ax: plt.Axes,
    sky_views: list[SkyViewSpec],
    marker_styles: list[MollweideViewMarkerStyle],
    *,
    view_flip_y: list[bool] | None = None,
    fstyle: FigureStyle | None = None,
) -> None:
    """Mark each sky-panel viewing direction with a star and an (l, b) label.

    Each marker gets a leader-line annotation; without an explicit
    ``text_offset`` the default alternates the label above the first marker
    and below the second, since two view markers are often close together on
    the map and would otherwise overlap.
    """
    fs = fstyle or FigureStyle()
    for i, (view, mstyle) in enumerate(zip(sky_views, marker_styles)):
        flip_y = bool(view_flip_y[i]) if view_flip_y is not None else False
        lon, lat = to_mollweide_coords(
            np.array([view.theta]),
            np.array([view.phi]),
        )
        theta_label = view.theta
        if flip_y:
            lat = -lat
            theta_label = np.pi - view.theta
        idx = i + 1
        if mstyle.text_offset is not None:
            offset = mstyle.text_offset
        else:
            offset = (10, 8) if i == 0 else (10, -12)
        face = mstyle.text_bbox_facecolor or mstyle.color
        ax.plot(
            lon[0],
            lat[0],
            marker=mstyle.marker,
            markersize=mstyle.size,
            color=mstyle.color,
            markeredgecolor=mstyle.edgecolor,
            markeredgewidth=mstyle.edgewidth,
            alpha=mstyle.alpha,
            zorder=mstyle.zorder_marker + i,
        )
        ax.annotate(
            _marker_label_text(mstyle, idx, theta_label, view.phi),
            xy=(lon[0], lat[0]),
            xytext=offset,
            textcoords="offset points",
            fontsize=fs.marker_text_fontsize,
            fontweight="bold",
            color="black",
            ha=mstyle.text_ha,
            va=mstyle.text_va,
            zorder=mstyle.zorder_text + i,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": face,
                "edgecolor": "black",
                "alpha": mstyle.text_bbox_alpha,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "black",
                "lw": 0.8,
                "shrinkA": 2,
                "shrinkB": 2,
            },
        )


def plot_mollweide_layer(
    ax: plt.Axes,
    layer: MollweideLayer,
    lon_msh: np.ndarray,
    lat_msh: np.ndarray,
    grid: np.ndarray,
    *,
    is_overlay: bool = False,
) -> plt.cm.ScalarMappable | None:
    """Draw one pcolormesh layer; return mappable (or None if placeholder).

    As an overlay, pixels at or below ``layer.mask_threshold`` are masked
    transparent so only the region with non-trivial non-concave fraction
    tints the layer beneath.
    """
    if layer.case is None:
        return None

    arr = np.asarray(grid, dtype=float)
    data = np.ma.masked_invalid(arr)
    if is_overlay:
        data = np.ma.masked_where(
            (arr <= layer.mask_threshold) | np.isnan(arr),
            arr,
        )

    mesh = ax.pcolormesh(
        lon_msh,
        lat_msh,
        data,
        cmap=layer.cmap,
        shading="gouraud",
        rasterized=True,
        vmin=0.0,
        vmax=1.0,
        zorder=2 if is_overlay else 1,
        alpha=layer.overlay_alpha if is_overlay else 1.0,
    )
    return mesh


def plot_mollweide_map(
    ax: plt.Axes,
    primary: MollweideLayer,
    secondary: MollweideLayer,
    *,
    sky_views: list[SkyViewSpec] | None = None,
    view_marker_styles: list[MollweideViewMarkerStyle] | None = None,
    fstyle: FigureStyle | None = None,
) -> tuple[plt.cm.ScalarMappable, plt.cm.ScalarMappable]:
    """Dual-layer Mollweide map with orbital-plane great circles and view markers.

    ``secondary`` is drawn first as a full opaque heatmap; ``primary`` is
    drawn on top as a masked, semi-transparent overlay, so the two cases'
    non-concave-fraction fields are visible simultaneously without one fully
    occluding the other. Each case's great circle is overplotted in white,
    and each entry in *sky_views* adds a starred marker with an (l, b) label.
    """
    fs = fstyle or FigureStyle()

    mesh_by_badge = {
        primary.badge: None,
        secondary.badge: None,
    }
    gc_zorder = 20

    def _draw_layer(layer: MollweideLayer, *, is_overlay: bool) -> None:
        nonlocal gc_zorder
        if layer.case is None:
            return
        lon_msh, lat_msh, grid = mollweide_fraction_grid(
            layer.case,
            flip_y=layer.flip_y,
        )
        mesh = plot_mollweide_layer(
            ax,
            layer,
            lon_msh,
            lat_msh,
            grid,
            is_overlay=is_overlay,
        )
        mesh_by_badge[layer.badge] = mesh

        _plot_great_circle(
            ax,
            layer.case,
            color="white",
            ls=":",
            lw=1.8,
            zorder=gc_zorder,
            flip_y=layer.flip_y,
        )
        gc_zorder += 1

    _draw_layer(primary, is_overlay=True)
    _draw_layer(secondary, is_overlay=False)

    if sky_views:
        defaults = [
            MollweideViewMarkerStyle(color="white"),
            MollweideViewMarkerStyle(color="salmon"),
        ]
        styles = view_marker_styles or defaults[: len(sky_views)]
        if len(styles) < len(sky_views):
            styles = list(styles) + defaults[len(styles) : len(sky_views)]
        marker_flip_y = [primary.flip_y]
        if len(sky_views) > 1:
            marker_flip_y.append(secondary.flip_y)
        _plot_view_markers(
            ax,
            sky_views,
            styles,
            view_flip_y=marker_flip_y,
            fstyle=fs,
        )

    ax.grid(True, ls=":", lw=0.8, alpha=fs.moll_grid_alpha, color="gray", zorder=5)
    ax.set_title(MOLL_TITLE, fontsize=fs.moll_title_fontsize, fontweight="bold", pad=10)

    for label in ax.get_xticklabels():
        label.set_fontsize(fstyle.moll_lon_tick_fontsize)
        label.set_color("white")
    ax.yaxis.tick_right()
    for label in ax.get_yticklabels():
        label.set_fontsize(fstyle.moll_lat_tick_fontsize)

    return mesh_by_badge[primary.badge], mesh_by_badge[secondary.badge]


# ════════════════════════════════════════════════════════════════════════════
# Colorbars
# ════════════════════════════════════════════════════════════════════════════


colorbar_fmt = mticker.FuncFormatter(lambda x, _: f"{x * 100:.0f}%")

colorbar_bbox = {
    "boxstyle": "round,pad=0.2",
    "facecolor": "white",
    "edgecolor": "black",
    "alpha": 0.6,
}


def add_dual_colorbars(
    fig: plt.Figure,
    mesh_primary: plt.cm.ScalarMappable,
    mesh_secondary: plt.cm.ScalarMappable,
    cbax_left: plt.Axes,
    cbax_right: plt.Axes,
    *,
    badge_left: str = "1",
    badge_right: str = "2",
    fstyle: FigureStyle,
) -> tuple[Any, Any]:
    """Paired vertical colorbars; ticks left / right; shared label on right."""
    cb_left = fig.colorbar(mesh_primary, cax=cbax_left, format=colorbar_fmt)
    cb_left.ax.yaxis.set_ticks_position("left")
    cb_left.ax.yaxis.set_label_position("left")
    cb_left.ax.tick_params(
        labelleft=True,
        left=True,
        labelright=False,
        right=False,
        labelsize=fstyle.cb_tick_fontsize,
    )
    cbax_left.text(
        0.5,
        0.5,
        badge_left,
        transform=cbax_left.transAxes,
        ha="center",
        va="center",
        fontsize=fstyle.cb_badge_fontsize,
        fontweight="bold",
        bbox=colorbar_bbox,
    )

    cb_right = fig.colorbar(
        mesh_secondary,
        cax=cbax_right,
        format=colorbar_fmt,
        label=CB_LABEL,
    )
    cb_right.ax.tick_params(
        labelright=True,
        right=True,
        labelleft=False,
        left=False,
        labelsize=fstyle.cb_tick_fontsize,
    )
    cb_right.ax.yaxis.label.set_size(fstyle.cb_label_fontsize)
    cbax_right.text(
        0.5,
        0.5,
        badge_right,
        transform=cbax_right.transAxes,
        ha="center",
        va="center",
        fontsize=fstyle.cb_badge_fontsize,
        fontweight="bold",
        bbox=colorbar_bbox,
    )

    return cb_left, cb_right
