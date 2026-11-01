"""Numerical-confirmation figure: sky density + Mollweide + orientation inset.

Two stacked sky-plane density panels (particle-spray stream, N-body stream)
on the left, a shared dual-layer Mollweide map of non-concave fraction in the
middle with an orientation-inset schematic, and paired colorbars on the
right.

Output: src/tex/figures/streams/numerical_confirmation.pdf
"""

import pathlib

import jax
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from viz_ridgeline import (
    FigureStyle,
    MollweideLayer,
    MollweideViewMarkerStyle,
    SkyDensityStyle,
    SkyPanelInfo,
    SkyTrackStyle,
    SkyViewSpec,
    add_dual_colorbars,
    load_data,
    plot_mollweide_map,
    plot_sky_density_panel,
    viewing_angles_lb_deg,
    worst_viewing_angle,
)

from stream_convexity_paper.orientation_inset import add_orientation_inset

jax.config.update("jax_enable_x64", True)

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150})


from stream_convexity_paper import paths

# =========================================================

# Particle Spray Example
DATA_PARTICLESPRAY = load_data(
    paths.STATIC / "streams" / "triaxial08507_rs14_65Gyr_init2_Mprog1e6_ridgeline",
)

# N-Body Example
DATA_NBODY = load_data(paths.STATIC / "streams" / "nbody_q15_ridgeline")

# =========================================================

# The Mollweide viewing-direction labels ("1: Particle Spray", "2: N-Body")
# are small against the map at the default 8 pt.
fstyle = FigureStyle(marker_text_fontsize=11.0)

fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 3.5, 0.15], hspace=0.02, wspace=0.08)

# Left column — two stacked orbit panels (independent axes)
ax_top = fig.add_subplot(gs[0, 0])
ax_bot = fig.add_subplot(gs[1, 0])

# Middle column — Mollweide map spanning both rows
ax_map = fig.add_subplot(gs[:, 1], projection="mollweide")

# Right column — split horizontally into two equal colorbar axes with no gap;
# wspace=0 makes them flush so the pair looks like a single two-tone bar
gs_cb = gs[:, 2].subgridspec(1, 2, wspace=0)
cbax1 = fig.add_subplot(gs_cb[0, 0])  # left half  → viridis  (circular orbit)
cbax2 = fig.add_subplot(gs_cb[0, 1])  # right half → cividis  (elliptical orbit)

# Nudge the Mollweide left (closer to orbit panels) and shift the colorbars
# right by the same amount so the latitude tick labels have room to breathe.
for _ax, delta_x in ((ax_map, -0.010), (cbax1, 0.035), (cbax2, 0.035)):
    _p = _ax.get_position()
    _ax.set_position([_p.x0 + delta_x, _p.y0, _p.width, _p.height])

# ========================================================
# Top left panel

cmap_a = plt.get_cmap("cividis")
color_a = cmap_a(0.5)
cmap_density_a = LinearSegmentedColormap.from_list("density_a", [(1, 1, 1, 0), color_a])

DENSITY_A = SkyDensityStyle(
    cmap=cmap_density_a,
    vmax_percentile=92.0,
    norm_gamma=0.65,
    adaptive_bin_width_kpc=0.4,
    use_lognorm=True,
)

TRACK_A = SkyTrackStyle(
    track_color_good=cmap_a(0.5),
    track_lw_good=2.5,
    track_alpha_good=0.95,
    track_color_bad="#FF7F7F",
    track_lw_bad=2.5,
    track_alpha_bad=0.75,
)

# Sky panel limits [kpc]
THETA_A, PHI_A = worst_viewing_angle(DATA_PARTICLESPRAY)
VIEW_A = SkyViewSpec(DATA_PARTICLESPRAY, THETA_A, PHI_A, xlim=(-15, 15), ylim=(-25, 40))

# step_2 style info boxes: r range + (l,b); no κ_<0 line unless pct_bad set ─
# Flip the viewing angle
_, b_A = viewing_angles_lb_deg(THETA_A, PHI_A)
INFO_A = SkyPanelInfo(r_min=5, r_max=50, b_deg=-b_A)

top_info = plot_sky_density_panel(
    ax_top,
    VIEW_A.case.particles,
    VIEW_A.case.track,
    VIEW_A.theta,
    VIEW_A.phi,
    panel="top",
    panel_badge="1: Particle Spray",
    density_style=DENSITY_A,
    track_style=TRACK_A,
    xlim=VIEW_A.xlim,
    ylim=VIEW_A.ylim,
    panel_info=INFO_A,
    info_position="lower left",
    fstyle=fstyle,
)

# ========================================================
# Bottom left panel

cmap_b = plt.get_cmap("viridis")
color_b = cmap_b(0.4)
cmap_density_b = LinearSegmentedColormap.from_list("density_b", [(1, 1, 1, 0), color_b])

DENSITY_B = SkyDensityStyle(
    cmap=cmap_density_b,
    vmax_percentile=92.0,
    norm_gamma=0.45,
    adaptive_bin_width_kpc=0.8,
)

TRACK_B = SkyTrackStyle(
    track_color_good=cmap_b(0.5),
    track_lw_good=2.5,
    track_alpha_good=0.95,
    track_color_bad="#FF7F7F",
    track_lw_bad=2.5,
    track_alpha_bad=0.9,
)

# Sky panel limits [kpc]
THETA_B, PHI_B = worst_viewing_angle(DATA_NBODY)
VIEW_B = SkyViewSpec(DATA_NBODY, THETA_B, PHI_B, xlim=(-20, 30), ylim=(-25, 45))

bot_info = plot_sky_density_panel(
    ax_bot,
    VIEW_B.case.particles,
    VIEW_B.case.track,
    VIEW_B.theta,
    VIEW_B.phi,
    panel="bottom",
    panel_badge="2: N-body",
    density_style=DENSITY_B,
    track_style=TRACK_B,
    xlim=VIEW_B.xlim,
    ylim=VIEW_B.ylim,
    panel_info=SkyPanelInfo(r_min=10, r_max=60),
    info_position="lower right",
    fstyle=fstyle,
)

# ========================================================
# Mollweide plot

primary_layer = MollweideLayer(
    case=DATA_PARTICLESPRAY,
    cmap="cividis",
    badge="1: Particle Spray",
    overlay_alpha=0.82,
    mask_threshold=0.10,
    flip_y=True,
)
secondary_layer = MollweideLayer(
    case=DATA_NBODY,
    cmap="viridis",
    badge="2: N-body",
    overlay_alpha=0.82,
    mask_threshold=0.10,
    flip_y=False,
)

# Mollweide view markers
MARKER_A = MollweideViewMarkerStyle(
    color="white",
    marker="*",
    size=15,
    text_bbox_alpha=0.9,
    text_template="1: Particle Spray",
    text_offset=(-12, -26),
)
MARKER_B = MollweideViewMarkerStyle(
    color="salmon",
    marker="P",
    size=12,
    text_bbox_alpha=0.75,
    text_template="2: N-Body",
    text_offset=(-45, 3),
)

mesh_a, mesh_b = plot_mollweide_map(
    ax_map,
    primary_layer,
    secondary_layer,
    sky_views=[VIEW_A, VIEW_B],
    view_marker_styles=[MARKER_A, MARKER_B],
    fstyle=fstyle,
)

# ── Orientation inset ─────────────────────────────────────
# Wireframe globe of viewing directions with the particle-spray stream inside
# it, an external observer, and the sky plane carrying the stream's shadow —
# what a point on the Mollweide map above means.

_rng = np.random.default_rng(0)
_spray = np.asarray(DATA_PARTICLESPRAY.particles)
_spray = _spray[_rng.choice(len(_spray), 2500, replace=False)]
add_orientation_inset(
    fig,
    ax_map,
    _spray,
    # Darker end of the same cividis ramp: cmap_a(0.5) is mid-grey and would
    # vanish into the grey wireframe and projection plane.
    color=cmap_a(0.15),
    # Same box size as the Fig. 3 inset so the two schematics read as one pair
    rect=(-0.052, 0.746, 0.20, 0.35),
    linestyle="none",
    marker=".",
    markersize=0.8,
    alpha=0.55,
)

# ── Colorbars ─────────────────────────────────────────────

add_dual_colorbars(
    fig,
    mesh_a,
    mesh_b,
    cbax1,
    cbax2,
    fstyle=fstyle,
)

# ── Legend (step_2 style) ─────────────────────────────────────────────

_h_com = mlines.Line2D(
    [],
    [],
    marker="x",
    color="black",
    linestyle="None",
    markersize=9,
    markeredgewidth=2,
    label="CoM",
)
_h_bad = mlines.Line2D(
    [],
    [],
    color=TRACK_B.track_color_bad,
    lw=5,
    alpha=TRACK_B.track_alpha_bad,
    label=r"$\mathcal{A} \leq 0$",
)
ax_bot.legend(
    handles=[_h_com, _h_bad],
    loc="center left",
    bbox_to_anchor=(1.0, 0.12),
    bbox_transform=ax_bot.transAxes,
    fontsize=12,
    framealpha=0.9,
)

# ── Aspects  ─────────────────────────────────────────────

ax_top.set_box_aspect(1)
ax_top.set_aspect("auto")
ax_bot.set_box_aspect(1)
ax_bot.set_aspect("auto")


ax_map.set_axisbelow(False)
for lbl in ax_map.get_xticklabels():
    lbl.set_zorder(1000)
    lbl.set_clip_on(False)
    lbl.set_color("white")


out_path = paths.FIGURES / "streams" / "numerical_confirmation.pdf"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved → {out_path}")
