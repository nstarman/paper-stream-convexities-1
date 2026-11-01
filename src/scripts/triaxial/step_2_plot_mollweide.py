"""Triaxial NFW Orbit Convexity: plot Mollweide maps.

Loads the cached arrays produced by step_1_compute_orbits.py and produces a
three-column figure:
  Left   — two stacked sky-plane orbit panels
  Middle — Mollweide non-concave-fraction map (two overlaid colourmaps)
  Right  — paired colorbars (viridis | cividis)
"""

import pathlib

from typing import Final

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from stream_convexity_paper.orientation_inset import add_orientation_inset
from stream_convexity_paper.values import write_values

from stream_convexity_paper import paths

#: Orbit segments inside this radius are drawn dotted, marking where baryonic
#: contributions to the potential are no longer negligible. Quoted in the
#: manuscript, so it is exported below rather than repeated there by hand.
R_NEAR_KPC: Final = 25.0

# ── Load data ─────────────────────────────────────────────────────────────────

data = np.load(paths.DATA / "triaxial" / "triaxial_data.npz")

nonconcave_frac_circ: np.ndarray = data["nonconcave_frac_circ"]
nonconcave_frac_ellip: np.ndarray = data["nonconcave_frac_ellip"]
nonconcave_mask_circ: np.ndarray = data["nonconcave_mask_circ"]
nonconcave_mask_ellip: np.ndarray = data["nonconcave_mask_ellip"]
lat_c_deg: float = float(data["lat_c_deg"])
lon_e_deg: float = float(data["lon_e_deg"])
lat_e_deg: float = float(data["lat_e_deg"])
r_ellip_kpc: np.ndarray = data["r_ellip_kpc"]
lon_edges: np.ndarray = data["lon_edges"]
lat_edges: np.ndarray = data["lat_edges"]
Lhat_circ: np.ndarray = data["Lhat_circ"]
Lhat_ellip: np.ndarray = data["Lhat_ellip"]

# ── Figure layout ─────────────────────────────────────────────────────────────
# Three-column GridSpec:
#   col 0 (narrow)  — two stacked sky-plane orbit panels
#   col 1 (wide)    — Mollweide non-concave-fraction map
#   col 2 (thin)    — pair of colorbars (viridis | cividis), flush together
fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 3.5, 0.15], hspace=0.02, wspace=0.08)

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

# ── Top-left panel: near-circular orbit sky projection ────────────────────────

orbit1_cmap = plt.get_cmap("viridis")
ax_top.scatter(0, 0, marker="x", color="black", label="CoM", zorder=5)

_x_nc = np.where(nonconcave_mask_circ, data["sky_c_x"], np.nan)
_y_nc = np.where(nonconcave_mask_circ, data["sky_c_y"], np.nan)
ax_top.plot(
    _x_nc,
    _y_nc,
    lw=7,
    color="red",
    alpha=0.5,
    zorder=1,
    solid_capstyle="round",
)
ax_top.plot(
    np.where(~nonconcave_mask_circ, data["sky_c_x"], np.nan),
    np.where(~nonconcave_mask_circ, data["sky_c_y"], np.nan),
    lw=2.5,
    color=orbit1_cmap(0.5),
    alpha=0.8,
    zorder=2,
)
ax_top.plot(
    np.where(nonconcave_mask_circ, data["sky_c_x"], np.nan),
    np.where(nonconcave_mask_circ, data["sky_c_y"], np.nan),
    lw=2.5,
    color=orbit1_cmap(0.5),
    alpha=1.0,
    zorder=2,
)
ax_top.text(
    0.05,
    0.95,
    "Orbit 1 View",
    transform=ax_top.transAxes,
    ha="left",
    va="top",
    fontsize=13,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray"),
)
ax_top.text(
    0.53,
    0.19,
    (
        rf"$r \in ({float(data['r_min_circ']):.0f}, {float(data['r_max_circ']):.0f})\,\mathrm{{kpc}}$"
        "\n"
        rf"$(l, b) = ({float(data['lon_c_deg']):.0f}^{{\!\circ}}\!\!, {float(data['lat_c_deg']):.0f}^{{\!\circ}}\!)$"
    ),
    transform=ax_top.transAxes,
    ha="right",
    va="top",
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
)
ax_top.set_xlabel("")
ax_top.set_ylabel("$y_\\mathrm{sky}$ [kpc]", fontsize=15)
ax_top.tick_params(axis="both", labelsize=15)
ax_top.tick_params(axis="x", bottom=False, top=True, labeltop=True, labelbottom=False)
ax_top.grid(True, alpha=0.3)

# ── Bottom-left panel: elliptical orbit sky projection ────────────────────────
# Rotate orbit 2 sky-plane coordinates by 45° around the z-axis.
_rot45 = np.deg2rad(-45)
_cos45, _sin45 = np.cos(_rot45), np.sin(_rot45)
sky_e_x_r = _cos45 * data["sky_e_x"] - _sin45 * data["sky_e_y"]
sky_e_y_r = _sin45 * data["sky_e_x"] + _cos45 * data["sky_e_y"]

orbit2_cmap = plt.get_cmap("cividis")
ax_bot.scatter(0, 0, marker="x", color="black", label="CoM", zorder=5)
_x_nc = np.where(nonconcave_mask_ellip, sky_e_x_r, np.nan)
_y_nc = np.where(nonconcave_mask_ellip, sky_e_y_r, np.nan)
ax_bot.plot(
    _x_nc,
    _y_nc,
    lw=7,
    color="red",
    alpha=0.5,
    zorder=1,
    solid_capstyle="round",
)
_near_mask_ellip = r_ellip_kpc < R_NEAR_KPC
ax_bot.plot(
    np.where(~_near_mask_ellip, sky_e_x_r, np.nan),
    np.where(~_near_mask_ellip, sky_e_y_r, np.nan),
    lw=2.5,
    color=orbit2_cmap(0.5),
    alpha=0.8,
    zorder=2,
)
ax_bot.plot(
    np.where(_near_mask_ellip, sky_e_x_r, np.nan),
    np.where(_near_mask_ellip, sky_e_y_r, np.nan),
    lw=2.5,
    color=orbit2_cmap(0.5),
    alpha=0.8,
    linestyle="dotted",
    zorder=2,
)
ax_bot.text(
    0.05,
    0.95,
    "Orbit 2 View",
    transform=ax_bot.transAxes,
    ha="left",
    va="top",
    fontsize=13,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray"),
)
ax_bot.text(
    0.54,
    0.05,
    (
        rf"$r \in ({float(data['r_min_ellip']):.0f}, {float(data['r_max_ellip']):.0f})\,\mathrm{{kpc}}$"
        "\n"
        rf"$(l, b) = ({float(data['lon_e_deg']):.0f}^{{\!\circ}}\!\!, {float(data['lat_e_deg']):.0f}^{{\!\circ}}\!)$"
    ),
    transform=ax_bot.transAxes,
    ha="right",
    va="bottom",
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
)
ax_bot.set_xlabel("$x_\\mathrm{sky}$ [kpc]", fontsize=15)
ax_bot.set_ylabel("$y_\\mathrm{sky}$ [kpc]", fontsize=15)
ax_bot.tick_params(axis="both", labelsize=15)
ax_bot.grid(True, alpha=0.3)

# ── Single legend straddling both orbit panels ────────────────────────────────
# Anchored at centre-right of the ax_top/ax_bot boundary so CoM entry sits in
# the bottom-right of Orbit 1 and the non-concave entry in the top-right of
# Orbit 2 — one legend object.
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
_h_nc = mlines.Line2D(
    [],
    [],
    color="red",
    lw=5,
    alpha=0.5,
    label=r"$\mathcal{A} \leq 0$",
)
ax_bot.legend(
    handles=[_h_com, _h_nc],
    loc="center left",
    bbox_to_anchor=(1.0, 0.12),
    bbox_transform=ax_bot.transAxes,
    fontsize=12,
    framealpha=0.9,
)


# Tight per-axis limits with 5 % margin on each side for both panels.
def _tight_lims(arr: np.ndarray, margin: float = 0.05) -> tuple[float, float]:
    span = arr.max() - arr.min()
    pad = margin * span
    return float(arr.min() - pad), float(arr.max() + pad)


ax_top.set_xlim(*_tight_lims(data["sky_c_x"]))
ax_top.set_ylim(*_tight_lims(data["sky_c_y"]))
ax_bot.set_xlim(*_tight_lims(sky_e_x_r))
ax_bot.set_ylim(-12, 13)

# ── Mollweide non-concave-fraction map ────────────────────────────────────────────

# Circular-orbit layer (viridis) — shown everywhere, including zero
mesh_circ = ax_map.pcolormesh(
    lon_edges,
    lat_edges,
    nonconcave_frac_circ,
    cmap=orbit1_cmap,
    vmin=0.0,
    shading="flat",
    rasterized=True,
)

# Elliptical-orbit layer (cividis)
# Mask cells with fraction ≤ 0.05 so the viridis layer shows through where
# the elliptical orbit has low non-concavity
nonconcave_frac_ellip_masked = np.ma.masked_where(
    nonconcave_frac_ellip <= 0.05,
    nonconcave_frac_ellip,
)
mesh_ellip = ax_map.pcolormesh(
    lon_edges,
    lat_edges,
    nonconcave_frac_ellip_masked,
    cmap=orbit2_cmap,
    vmin=0.0,
    shading="flat",
    rasterized=True,
    alpha=0.8,
)
ax_map.grid(True, alpha=0.3)
ax_map.text(
    0.5,
    1.06,
    "Viewing Angles",
    transform=ax_map.transAxes,
    ha="center",
    va="top",
    fontsize=13,
    fontweight="bold",
)

# ── Osculating-plane great circles ────────────────────────────────────────────


def _great_circle(
    Lhat: np.ndarray,
    n_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """(lon, lat) in radians of the great circle perpendicular to Lhat.

    Uses the viewing-angle convention: the unit vector toward the observer at
    (lon, lat) is (-cos b cos l, -cos b sin l, sin b).  The great circle is
    the locus of viewing directions that lie in the orbital plane.
    NaN breaks are inserted at antimeridian jumps to avoid spurious lines.
    """
    tmp = np.array([1.0, 0.0, 0.0]) if abs(Lhat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u_vec = np.cross(Lhat, tmp)
    u_vec /= np.linalg.norm(u_vec)
    v_vec = np.cross(Lhat, u_vec)  # already unit length

    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    pts = np.outer(np.cos(t), u_vec) + np.outer(np.sin(t), v_vec)  # (n, 3)

    lat = np.arcsin(np.clip(pts[:, 2], -1.0, 1.0))
    lon = np.arctan2(-pts[:, 1], -pts[:, 0])

    jump_idx = np.where(np.abs(np.diff(lon)) > np.pi)[0] + 1
    lon = np.insert(lon.astype(float), jump_idx, np.nan)
    lat = np.insert(lat.astype(float), jump_idx, np.nan)
    return lon, lat


for _Lhat in (Lhat_circ, Lhat_ellip):
    _gc_lon, _gc_lat = _great_circle(_Lhat)
    ax_map.plot(
        _gc_lon,
        _gc_lat,
        color="white",
        linestyle="dotted",
        lw=1.5,
        zorder=4,
    )

# ── Stars ────────────────────────────────────────────

# Stars marking the worst viewing angle for each orbit
_lon_c_rad = np.deg2rad(data["lon_c_deg"])
_lat_c_rad = np.deg2rad(lat_c_deg)
ax_map.plot(
    _lon_c_rad,
    _lat_c_rad,
    marker=".",
    markersize=16,
    color="salmon",
    zorder=5,
)
ax_map.text(
    _lon_c_rad + 0.25,
    _lat_c_rad + 0.15,
    "View 1",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.2",
        facecolor="white",
        edgecolor="black",
        alpha=0.6,
    ),
)

_lon_e_rad = np.deg2rad(lon_e_deg)
_lat_e_rad = np.deg2rad(lat_e_deg)
ax_map.plot(
    _lon_e_rad,
    _lat_e_rad,
    marker=".",
    markersize=16,
    color="salmon",
    zorder=5,
)
ax_map.text(
    _lon_e_rad + 0,
    _lat_e_rad - 0.15,
    "View 2",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.2",
        facecolor="white",
        edgecolor="black",
        alpha=0.6,
    ),
)

# Make longitude tick labels bigger; the ones drawn inside the map are white
for label in ax_map.get_xticklabels():
    label.set_fontsize(15)
    label.set_color("white")
ax_map.yaxis.tick_right()
for label in ax_map.get_yticklabels():
    label.set_fontsize(15)

# ── Orientation inset: wireframe globe + projection plane + external observer ─
# Small schematic in the blank upper-left corner of the Mollweide axes,
# explaining what a point on the map means: a direction from which an external
# observer views the orbit, and the sky plane they see it projected onto.

add_orientation_inset(
    fig,
    ax_map,
    data["xyz_circ_kpc"],
    color=orbit1_cmap(0.5),
    lw=1.3,
    rect=(-0.046, 0.746, 0.20, 0.35),
)

# ── Colorbars ─────────────────────────────────────────────────────────────────
# Left bar (viridis, circular orbit): suppress all ticks and labels — it acts
# only as a colour reference; the shared scale is labelled on the right bar.

_pct_fmt = mticker.FuncFormatter(lambda x, _: f"{x * 100:.0f}%")

cb1 = fig.colorbar(mesh_circ, cax=cbax1, format=_pct_fmt)
cb1.ax.yaxis.set_ticks_position("left")
cb1.ax.yaxis.set_label_position("left")
cb1.ax.tick_params(
    labelleft=True,
    left=True,
    labelright=False,
    right=False,
    labelsize=14,
)
cbax1.text(
    0.5,
    0.5,
    "1",
    transform=cbax1.transAxes,
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.2",
        facecolor="white",
        edgecolor="black",
        alpha=0.6,
    ),
)

# Right bar (cividis, elliptical orbit): carries the axis label and tick values
cb2 = fig.colorbar(
    mesh_ellip,
    cax=cbax2,
    label="Non-concave / total arclength",
    format=_pct_fmt,
)
cb2.ax.tick_params(labelsize=14)
cb2.ax.yaxis.label.set_size(14)
cbax2.text(
    0.5,
    0.5,
    "2",
    transform=cbax2.transAxes,
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.2",
        facecolor="white",
        edgecolor="black",
        alpha=0.6,
    ),
)

# ── Save ──────────────────────────────────────────────────────────────────────

out = paths.FIGURES / "triaxial" / "triaxial_diagnostic.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")

# Values this script owns that the manuscript quotes; see
# src/scripts/make_variables.py.
write_values(__file__, {"r_near_kpc": R_NEAR_KPC})
