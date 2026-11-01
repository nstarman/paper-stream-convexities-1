"""The saved-case loader and the convexity statistics for the ridgeline figures.

Loads one saved analysis case from disk, rebuilds its
``stream_convexity_paper.track.Track3D``, and reduces it to the per-gamma
non-concavity flags the figures draw.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import interpax
import jax.numpy as jnp
import numpy as np

from stream_convexity_paper.track import Track3D
from stream_convexity_paper.curvature import (
    is_nonconcave,
    projected_curvature_normal_and_kappa,
)

from sky_geometry import (
    flip_mollweide_grid_y,
    interpolate_mollweide_field,
    rotation_matrix_from_viewing_angles,
    to_mollweide_coords,
)


@dataclass
class RidgelineCase:
    """In-memory bundle for one saved ridgeline analysis case."""

    case_id: str
    meta: dict[str, Any]
    arrays: dict[str, np.ndarray]
    track: Track3D
    particles: np.ndarray  # (N, 3) positions [kpc]


def load_data(case_dir: Path, /) -> RidgelineCase:
    """Load metadata + npz arrays and rebuild ``Track3D``."""
    meta_path = next(case_dir.glob("metadata_*.json"))
    npz_path = next(case_dir.glob("viz_data_*.npz"))

    meta = json.loads(meta_path.read_text())
    raw = np.load(npz_path)
    arrays = {k: raw[k] for k in raw.files}

    gamma_knots = jnp.asarray(arrays["gamma_knots"], dtype=float)
    knots_xyz = jnp.asarray(arrays["knots_xyz"], dtype=float)
    track = Track3D(interpax.Interpolator1D(gamma_knots, knots_xyz, method="cubic2"))

    particles = np.asarray(arrays["mem_pos"], dtype=float)

    return RidgelineCase(
        case_id=str(meta.get("case_id", case_dir.name)),
        meta=meta,
        arrays=arrays,
        track=track,
        particles=particles,
    )


def worst_viewing_angle(case: RidgelineCase) -> tuple[float, float]:
    """Return (θ, φ) with the highest sphere-scan bad-κ percentage."""
    i = int(np.argmax(case.arrays["pct_neg"]))
    return (
        float(case.arrays["theta_sphere"][i]),
        float(case.arrays["phi_sphere"][i]),
    )


def _compute_projected_curvature_normal(
    track: Track3D,
    gamma: jnp.ndarray,
    r: np.ndarray,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Unit normal and scalar curvature in the projected sky plane."""
    nhat, kappa = projected_curvature_normal_and_kappa(
        track.tangent(gamma),
        track.curvature(gamma),
        jnp.asarray(r),
        eps=eps,
    )
    return np.asarray(nhat), np.asarray(kappa)


def ridgeline_sky_stats(
    track: Track3D,
    theta: float,
    phi: float,
    *,
    n_gamma: int = 500,
    gamma_range: tuple[float, float] = (-0.95, 0.95),
) -> dict[str, np.ndarray]:
    """Ridgeline sky projection plus bad-normal flags along γ."""
    gamma = jnp.linspace(gamma_range[0], gamma_range[1], n_gamma)
    r = rotation_matrix_from_viewing_angles(theta, phi)
    pos_3d = np.asarray(track.positions(gamma))
    pos_2d = pos_3d @ r.T[:, :2]

    nhat, _ = _compute_projected_curvature_normal(track, gamma, r)
    is_bad = np.asarray(is_nonconcave(pos_2d, nhat))

    return {
        "gamma": np.asarray(gamma),
        "positions_2d": pos_2d,
        "is_bad_normal": is_bad,
        "pct_bad": 100.0 * is_bad.mean(),
    }


def mollweide_fraction_grid(
    case: RidgelineCase,
    *,
    n_lon: int = 800,
    n_lat: int = 400,
    flip_y: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolated non-concave fraction grid for one case."""
    theta = case.arrays["theta_sphere"]
    phi = case.arrays["phi_sphere"]
    lon, lat = to_mollweide_coords(theta, phi)
    frac = np.asarray(case.arrays["pct_neg"], dtype=float) / 100.0
    lon_msh, lat_msh, grid = interpolate_mollweide_field(
        lon,
        lat,
        frac,
        n_lon=n_lon,
        n_lat=n_lat,
    )
    if flip_y:
        lon_msh, lat_msh, grid = flip_mollweide_grid_y(lon_msh, lat_msh, grid)
    return lon_msh, lat_msh, grid
