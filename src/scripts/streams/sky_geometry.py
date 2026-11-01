"""Sky-plane and Mollweide coordinate transforms.

Rotations from a viewing direction, the projections that put a 3D track on the
sky, and the wrapping, segment-splitting and gridding needed to draw fields in
a Mollweide projection without artefacts at the antimeridian.
"""

import numpy as np
from scipy.interpolate import griddata

import unxt as u
from stream_convexity_paper.curvature import (
    rotation_matrix_from_viewing_angles as _rotation_from_angles,
)


def rotation_matrix_from_viewing_angles(theta: float, phi: float) -> np.ndarray:
    """Rotation matrix mapping the viewing direction to +z (numpy).

    Thin adapter over
    :func:`stream_convexity_paper.curvature.rotation_matrix_from_viewing_angles`
    that takes bare radians and returns a plain matrix.
    """
    rotation = _rotation_from_angles(u.Q(theta, "rad"), u.Q(phi, "rad"))
    return np.asarray(rotation.as_matrix())


def normal_to_plane_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors spanning the plane whose normal is *n*."""
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(n, ref)) > 0.9:
        # n nearly parallel to ref would make n × ref near zero; switch axis
        ref = np.array([0.0, 1.0, 0.0])
    v1 = np.cross(n, ref)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(n, v1)
    v2 /= np.linalg.norm(v2)
    return v1, v2


def principal_plane_great_circle(
    v1: np.ndarray,
    v2: np.ndarray,
    n: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Great circle on the sphere for the PCA principal (edge-on) plane."""
    alpha = np.linspace(0, 2 * np.pi, n)
    gc_3d = np.outer(np.cos(alpha), v1 / np.linalg.norm(v1)) + np.outer(
        np.sin(alpha),
        v2 / np.linalg.norm(v2),
    )
    gc_3d /= np.linalg.norm(gc_3d, axis=1, keepdims=True)
    theta_gc = np.arccos(np.clip(gc_3d[:, 2], -1, 1))
    phi_gc = np.arctan2(gc_3d[:, 1], gc_3d[:, 0])
    phi_gc = np.where(phi_gc < 0, phi_gc + 2 * np.pi, phi_gc)
    return theta_gc, phi_gc


def flip_mollweide_grid_y(
    lon_msh: np.ndarray,
    lat_msh: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror a Mollweide field about the equator (lat → -lat).

    Keep ``grid[i]`` at row ``i`` and only negate latitude coordinates so each
    sample moves from ``lat`` to ``-lat``. Reversing grid rows cancels this
    move and leaves the heatmap unchanged.
    """
    return lon_msh, -lat_msh, grid


def project_to_sky(positions_3d: np.ndarray, theta: float, phi: float) -> np.ndarray:
    """Project 3D positions onto the sky plane for viewing angles (θ, φ)."""
    r = rotation_matrix_from_viewing_angles(theta, phi)
    return positions_3d @ r.T[:, :2]


def _phi_deg_wrapped(phi: float | np.ndarray) -> float | np.ndarray:
    """Galactocentric φ [rad] → degrees in [-180, 180], with φ=0 at map centre."""
    lon = (np.asarray(phi) + np.pi) % (2 * np.pi) - np.pi
    return np.degrees(lon)


def viewing_angles_lb_deg(theta: float, phi: float) -> tuple[float, float]:
    """Viewing direction as Galactic (l, b) in degrees (θ colatitude, φ azimuth)."""
    l_deg = float(_phi_deg_wrapped(phi))
    b_deg = 90.0 - float(np.degrees(theta))
    return l_deg, b_deg


def format_lb_latex(
    theta: float,
    phi: float,
    *,
    index: int | str | None = None,
) -> str:
    """LaTeX string ``(l,b)=(l°, b°)`` with optional leading index."""
    l_deg, b_deg = viewing_angles_lb_deg(theta, phi)
    body = rf"$(l,b)=({l_deg:.0f}^\circ,\,{b_deg:.0f}^\circ)$"
    if index is None:
        return body
    return rf"{index}: {body}"


def to_mollweide_coords(
    theta_arr: np.ndarray,
    phi_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Spherical (θ, φ) [rad] → Mollweide (lon, lat) [rad].

    Horizontal axis: φ ∈ [-π, π], **φ = 0 at map centre**.
    """
    lon_rad = (phi_arr + np.pi) % (2 * np.pi) - np.pi
    lat_rad = np.pi / 2 - theta_arr
    return lon_rad, lat_rad


def split_mollweide_segments(
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    jump_threshold: float = np.pi,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a closed curve at Mollweide longitude jumps.

    A curve crossing the antimeridian would otherwise be drawn as a spurious
    line straight across the map; splitting into per-segment pieces at each
    jump and plotting them separately avoids that artefact.
    """
    lon_diff = np.diff(lon)
    breaks = np.where(np.abs(lon_diff) > jump_threshold)[0]
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for b in breaks:
        segments.append((lon[start : b + 1], lat[start : b + 1]))
        start = b + 1
    segments.append((lon[start:], lat[start:]))
    return segments


def interpolate_mollweide_field(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    *,
    n_lon: int = 800,
    n_lat: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate scattered Mollweide samples onto a regular grid.

    Linear interpolation leaves grid points outside the samples' convex hull
    as NaN; those are filled in with nearest-neighbour values so the map has
    no holes near the edges.
    """
    pts = np.c_[lon, lat]
    lon_grd = np.linspace(-np.pi, np.pi, n_lon)
    lat_grd = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lon_msh, lat_msh = np.meshgrid(lon_grd, lat_grd)

    grid = griddata(pts, values, (lon_msh, lat_msh), method="linear", fill_value=np.nan)
    mask = np.isnan(grid)
    if np.any(mask):
        grid[mask] = griddata(pts, values, (lon_msh, lat_msh), method="nearest")[mask]
    return lon_msh, lat_msh, grid


def symmetric_limits_around_origin(
    x: np.ndarray,
    y: np.ndarray,
    *,
    pad_frac: float = 0.06,
    min_half: float = 1.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Square limits centred on the origin."""
    half = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), min_half * 0.5)
    half *= 1.0 + pad_frac
    return (-half, half), (-half, half)


def resolve_sky_panel_limits(
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    auto_xlim: tuple[float, float],
    auto_ylim: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Merge optional manual limits with auto-computed defaults."""
    return (
        auto_xlim if xlim is None else xlim,
        auto_ylim if ylim is None else ylim,
    )
