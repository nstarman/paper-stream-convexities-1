"""Triaxial NFW Orbit Convexity: compute and cache results.

Two orbits in a triaxial NFW potential are projected onto a regular
longitude/latitude grid of viewing angles.  For each angle the sky-plane spline
is fit and the non-concave arc-length fraction (κ < 0 segments / total arc) is
computed with `compute_bad_normals_for_angle` from `data.py`.

Outputs (saved to data/triaxial_data.npz):
  nonconcave_frac_circ  — (N_LAT, N_LON) non-concave fraction, near-circular orbit
  nonconcave_frac_ellip — (N_LAT, N_LON) non-concave fraction, elliptical orbit
  nonconcave_mask_circ  — (N_T_KAPPA,) bool, non-concave points at worst angle (Orbit 1)
  nonconcave_mask_ellip — (N_T_KAPPA,) bool, non-concave points at worst angle (Orbit 2)
  sky_c_x, sky_c_y — sky-plane orbit positions at the worst viewing angle (Orbit 1)
  sky_e_x, sky_e_y — sky-plane orbit positions at the worst viewing angle (Orbit 2)
  lon_edges, lat_edges — grid edges used for pcolormesh
"""
# ruff: noqa: N803

import pathlib

from typing import Final

import coordinax as cx
import coordinax.vecs as cxv
import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
import jax
import jax.scipy.spatial.transform
import numpy as np
import plum
import quax
from numpy.typing import NDArray

import quaxed.numpy as jnp
import unxt as u
from stream_convexity_paper.data import (
    _orbit_derivatives,
    compute_bad_normals_for_angle,
    compute_max_nonconcave_tangential_offset_for_angle,
    compute_projected_curvature_normal,
    generate_orbit,
    rotation_matrix_from_viewing_angles,
)
from stream_convexity_paper.values import write_values
from stream_convexity_paper.variables import PARAMS, value_unit

jax.config.update("jax_enable_x64", True)


# ── Settings ──────────────────────────────────────────────────────────────────

from stream_convexity_paper import paths

usys = u.unitsystems.galactic

N_LON = 120
N_LAT = 60
N_ORBIT_STEPS = 20_000

lon_edges = np.linspace(-np.pi, np.pi, N_LON + 1)
lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
LON: NDArray[np.float64]
LAT: NDArray[np.float64]
LON, LAT = np.meshgrid(lon_centers, lat_centers)  # (N_LAT, N_LON)

# ── Potential ─────────────────────────────────────────────────────────────────
# Parameters come from variables.yml, the single source of truth shared with the
# manuscript: src/scripts/make_variables.py renders the same values into
# output/triaxial/*.tex, which ms.tex reads through \variable{}. Change them
# there, not here, then run `showyourwork build`.

TRI: Final = PARAMS["triaxial_orbits"]
POT: Final = TRI["potential"]
HOST: Final = PARAMS["host_potential"]

pot = gp.TriaxialNFWPotential(
    m=u.Q(*value_unit(HOST["m_tot"])),
    r_s=u.Q(*value_unit(HOST["r_s"])),
    q1=POT["q1"]["value"],
    q2=POT["q2"]["value"],
    units="galactic",
)


# ── Orbits ────────────────────────────────────────────────────────────────────

w0_circ = gc.PhaseSpaceCoordinate(
    q=u.Q([4.949159151423, -61.627532288559, 47.814619651884], "kpc"),
    p=u.Q([0.193106108931, 0.015507887964, 0.023936867692], "kpc / Myr"),
    t=u.Q(0.0, "Myr"),
)
w0_ellip = gc.PhaseSpaceCoordinate(
    q=u.Q([10.747525628097, -19.315636156702, 43.846361129027], "kpc"),
    p=u.Q([0.096638620438, 0.053771257721, 0.14290570994], "kpc / Myr"),
    t=u.Q(0.0, "Myr"),
)

t_end = u.Q(*value_unit(TRI["t_integrate"]))
t_orbit = u.Q(jnp.linspace(0.0, t_end.value, N_ORBIT_STEPS), "Gyr")

orbit_circ: gd.Orbit = generate_orbit(pot, w0_circ, t_orbit, dense=True)
orbit_ellip: gd.Orbit = generate_orbit(
    pot,
    w0_ellip,
    t_orbit,
    dense=True,
    max_steps=16384,
)

# ── Non-concave fraction grid ─────────────────────────────────────────────────

N_T_KAPPA = 500  # number of time steps at which κ is evaluated per orbit

t_kappa = u.Q(jnp.linspace(0.0, t_end.value, N_T_KAPPA), "Gyr")


def compute_nonconcave_fracs(
    orbit: gd.Orbit,
    lon_2d: NDArray[np.float64],
    lat_2d: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return (N_LAT, N_LON) array of non-concave arc-length fractions.

    For each (lon, lat) viewing direction the orbit is projected onto the
    sky plane, a spline is fit, and the fraction of arc length with κ < 0
    (non-concave) is returned.
    """
    # Grid convention: colatitude θ = π/2 − lat, azimuth φ = lon + π
    theta_q = u.Q(jnp.pi / 2 - lat_2d.ravel(), "rad")
    phi_q = u.Q(lon_2d.ravel() + jnp.pi, "rad")
    counts = compute_bad_normals_for_angle(theta_q, phi_q, t_kappa, orbit, usys=usys)
    return np.array(counts / N_T_KAPPA).reshape(lat_2d.shape)


def compute_max_nonconcave_tangential_offset_grid(
    orbit: gd.Orbit,
    lon_2d: NDArray[np.float64],
    lat_2d: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return (N_LAT, N_LON) array of max |sin(tangent-radial angle)| at non-concave points."""
    theta_q = u.Q(jnp.pi / 2 - lat_2d.ravel(), "rad")
    phi_q = u.Q(lon_2d.ravel() + jnp.pi, "rad")
    offsets = compute_max_nonconcave_tangential_offset_for_angle(
        theta_q,
        phi_q,
        t_kappa,
        orbit,
        usys=usys,
    )
    return np.array(offsets).reshape(lat_2d.shape)


nonconcave_frac_circ: NDArray[np.float64] = compute_nonconcave_fracs(
    orbit_circ,
    LON,
    LAT,
)
nonconcave_frac_ellip: NDArray[np.float64] = compute_nonconcave_fracs(
    orbit_ellip,
    LON,
    LAT,
)

# ── Viewing angle ────────────────────────────────────


theta_c = u.Q(0.863937979737, unit="rad")
phi_c = u.Q(4.581489286485, unit="rad")
lon_c_deg = 82.5
lat_c_deg = 40.5

theta_e = u.Q(1.073377489977, unit="rad")
phi_e = u.Q(3.848451000647, unit="rad")
lon_e_deg = 40.5
lat_e_deg = 28.5

R_circ = rotation_matrix_from_viewing_angles(theta_c, phi_c)
R_ellip = rotation_matrix_from_viewing_angles(theta_e, phi_e)


def _sky_xy(
    orbit: gd.Orbit,
    R: jax.scipy.spatial.transform.Rotation,
) -> cxv.CartesianPos2D:
    """Project orbit positions onto the sky plane (x-y after rotation).

    Evaluates the dense orbit at the t_kappa time grid, converts positions to
    Cartesian, applies the viewing rotation, and returns a CartesianPos2D with
    the sky-plane x and y coordinates.
    """
    q: u.Q = plum.convert(orbit(t_kappa).q.vconvert(cx.CartesianPos3D), u.Q)
    xyz_rot: u.Q = quax.quaxify(R.apply)(q)
    return cxv.CartesianPos2D(x=xyz_rot[:, 0], y=xyz_rot[:, 1])


sky_c: cxv.CartesianPos2D = _sky_xy(orbit_circ, R_circ)
sky_e: cxv.CartesianPos2D = _sky_xy(orbit_ellip, R_ellip)
# ── Per-point non-concave masks at the worst viewing angles ───────────────────────


def _nonconcave_mask(
    orbit: gd.Orbit,
    R: jax.scipy.spatial.transform.Rotation,
) -> NDArray[np.bool_]:
    """Boolean mask (N_T_KAPPA,): True where the sky-plane orbit is non-concave.

    A point is non-concave when the curvature normal points away from the
    projected orbit centre (dot(nhat, –q/|q|) ≤ 0).
    """
    # Curvature normals at every time step (vmapped over t_kappa)
    nhat_all = jax.vmap(
        lambda t: compute_projected_curvature_normal(t, orbit, R, usys=usys),
    )(t_kappa)  # (N_T_KAPPA, 2)

    # Sky-plane positions (x, y) at each time step
    q_all = quax.quaxify(R.apply)(
        plum.convert(orbit(t_kappa).q.vconvert(cx.CartesianPos3D), u.Q),
    )[:, :2]  # (N_T_KAPPA, 2)

    # A normal is non-concave when it points away from the orbit centre
    dir_to_center = -q_all / jnp.linalg.norm(q_all, axis=-1, keepdims=True)
    dot_products = jnp.sum(nhat_all * dir_to_center, axis=-1)
    return np.array(dot_products <= 0)  # (N_T_KAPPA,) bool


nonconcave_mask_circ: NDArray[np.bool_] = _nonconcave_mask(orbit_circ, R_circ)
nonconcave_mask_ellip: NDArray[np.bool_] = _nonconcave_mask(orbit_ellip, R_ellip)

# ── Radial extents (3-D) ─────────────────────────────────────────────────────
_q_circ_3d = plum.convert(orbit_circ(t_kappa).q.vconvert(cx.CartesianPos3D), u.Q)
_r_circ_kpc = np.array(jnp.linalg.norm(_q_circ_3d, axis=-1).ustrip(usys))
r_min_circ: float = float(_r_circ_kpc.min())
r_max_circ: float = float(_r_circ_kpc.max())

_q_ellip_3d = plum.convert(orbit_ellip(t_kappa).q.vconvert(cx.CartesianPos3D), u.Q)
_r_ellip_kpc = np.array(jnp.linalg.norm(_q_ellip_3d, axis=-1).ustrip(usys))
r_min_ellip: float = float(_r_ellip_kpc.min())
r_max_ellip: float = float(_r_ellip_kpc.max())

# ── Orbital eccentricities ───────────────────────────────────────────────────
# e = (r_apo - r_peri) / (r_apo + r_peri), the convention `generate_orbit_ic`
# uses to set initial conditions. A triaxial potential does not close the orbit,
# so each successive pericentre/apocentre pair has its own e and we record the
# range. Written to step_1_compute_orbits.values.json, which
# src/scripts/make_variables.py renders into output/triaxial/orbit{1,2}_ecc.tex
# for the manuscript.
#
# The `e_range` handed to `generate_orbit_ic` sets the launch, and the map to the
# integrated orbit is an O(triaxiality) approximation, so e is measured here.


def _eccentricity_range(r_kpc: NDArray[np.float64]) -> tuple[float, float]:
    """Return the (min, max) eccentricity over successive peri/apo pairs."""
    dr = np.diff(r_kpc)
    peri = np.where((dr[:-1] < 0) & (dr[1:] >= 0))[0] + 1
    apo = np.where((dr[:-1] > 0) & (dr[1:] <= 0))[0] + 1
    turns = np.sort(np.concatenate([peri, apo]))
    if turns.size < 2:  # noqa: PLR2004
        msg = f"only {turns.size} turning points; cannot measure an eccentricity"
        raise ValueError(msg)
    lo = np.minimum(r_kpc[turns[:-1]], r_kpc[turns[1:]])
    hi = np.maximum(r_kpc[turns[:-1]], r_kpc[turns[1:]])
    e = (hi - lo) / (hi + lo)
    return float(e.min()), float(e.max())


ecc_circ: Final = _eccentricity_range(_r_circ_kpc)
ecc_ellip: Final = _eccentricity_range(_r_ellip_kpc)

# ── Mean orbital-plane normals (angular-momentum directions) ──────────────────


def _mean_Lhat(q: u.Q) -> NDArray[np.float64]:
    """Mean angular-momentum unit vector from consecutive position cross products."""
    xyz = np.array(q.ustrip(usys))  # (N_T_KAPPA, 3) in kpc
    L = np.cross(xyz[:-1], xyz[1:])  # (N_T_KAPPA-1, 3)
    Lmean = L.mean(axis=0)
    return Lmean / np.linalg.norm(Lmean)


Lhat_circ: NDArray[np.float64] = _mean_Lhat(_q_circ_3d)
Lhat_ellip: NDArray[np.float64] = _mean_Lhat(_q_ellip_3d)

# ── Tracks for the docs site's interactive figure ───────────────────────────
# Position and its first two time derivatives at t_kappa, what compute_bad_normals_for_angle
# projects for every view: the webapp re-projects them in the browser (webapps/convexity-full).


def _track(orbit: gd.Orbit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q, dq, d2q = jax.vmap(lambda t: _orbit_derivatives(t, orbit, usys))(t_kappa)
    return np.asarray(q), np.asarray(dq), np.asarray(d2q)


q_circ, dq_circ, d2q_circ = _track(orbit_circ)
q_ellip, dq_ellip, d2q_ellip = _track(orbit_ellip)

# ── Save ──────────────────────────────────────────────────────────────────────

out_dir = paths.DATA / "triaxial"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "triaxial_data.npz"

np.savez(
    out,
    nonconcave_frac_circ=nonconcave_frac_circ,
    nonconcave_frac_ellip=nonconcave_frac_ellip,
    nonconcave_mask_circ=nonconcave_mask_circ,
    nonconcave_mask_ellip=nonconcave_mask_ellip,
    xyz_circ_kpc=np.array(_q_circ_3d.ustrip(usys)),
    sky_c_x=np.array(sky_c.x.ustrip(usys)),
    sky_c_y=np.array(sky_c.y.ustrip(usys)),
    sky_e_x=np.array(sky_e.x.ustrip(usys)),
    sky_e_y=np.array(sky_e.y.ustrip(usys)),
    lon_edges=lon_edges,
    lat_edges=lat_edges,
    r_min_circ=r_min_circ,
    r_max_circ=r_max_circ,
    r_min_ellip=r_min_ellip,
    r_max_ellip=r_max_ellip,
    lon_c_deg=lon_c_deg,
    lat_c_deg=lat_c_deg,
    lon_e_deg=lon_e_deg,
    lat_e_deg=lat_e_deg,
    r_ellip_kpc=_r_ellip_kpc,
    Lhat_circ=Lhat_circ,
    Lhat_ellip=Lhat_ellip,
    track_q_circ=q_circ,
    track_dq_circ=dq_circ,
    track_d2q_circ=d2q_circ,
    track_q_ellip=q_ellip,
    track_dq_ellip=dq_ellip,
    track_d2q_ellip=d2q_ellip,
)
print(f"Saved {out}")

# Values this script owns that the manuscript quotes. Unlike data/, this file is
# committed, so `showyourwork build` works without re-running the computation.
write_values(
    __file__,
    {
        "orbit1_ecc": list(ecc_circ),
        "orbit2_ecc": list(ecc_ellip),
        "orbit1_peak_convex_frac": float(nonconcave_frac_circ.max()),
        "orbit2_peak_convex_frac": float(nonconcave_frac_ellip.max()),
    },
)
