"""Data generation utilities for stellar orbits and streams."""

__all__ = (
    "OrbitICInfo",
    "compute_bad_normals_for_angle",
    "compute_bad_normals_for_rotation",
    "compute_max_nonconcave_kappa_for_angle",
    "compute_max_nonconcave_tangential_offset_for_angle",
    "compute_projected_curvature_normal",
    "compute_projected_curvature_normal_and_kappa",
    "count_bad_normals",
    "generate_orbit",
    "generate_orbit_ic",
    "rotation_matrix_from_viewing_angles",
)

import functools as ft

from jaxtyping import Array, Float, PRNGKeyArray
from typing import NamedTuple

import coordinax as cx
import equinox as eqx
import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
import jax
import jax.random as jr
import quax
from jaxmore import vmap
from plum import convert

import quaxed.numpy as jnp
import unxt as u

from .curvature import (
    count_bad_normals,
    is_nonconcave,
    project_to_sky_plane,
    projected_curvature_normal_and_kappa as _nhat_kappa,
    rotation_matrix_from_viewing_angles,
)

usys = u.unitsystems.galactic

R_RANGE_DEFAULT = (u.Q(20.0, "kpc"), u.Q(100.0, "kpc"))
ECCENTRICITY_RANGE_DEFAULT = (0.0, 0.3)
INCLINATION_RANGE_DEFAULT = (u.Q(0.0, "deg"), u.Q(90.0, "deg"))
T0_DEFAULT = u.Q(0.0, "Myr")

##############################################################################
# Orbit generation


class OrbitICInfo(NamedTuple):
    """Sampled scalar properties of a set of initial conditions."""

    r: u.Q  # Galactocentric radius at t0
    e: Array  # orbital eccentricity in [0, 1)
    inc: u.Q  # inclination angle


@eqx.filter_jit
def generate_orbit_ic(
    key: PRNGKeyArray,
    potential: gp.AbstractPotential,
    *,
    r_range: tuple[u.Q, u.Q] = R_RANGE_DEFAULT,
    e_range: tuple[float, float] = ECCENTRICITY_RANGE_DEFAULT,
    i_range: tuple[u.Q, u.Q] = INCLINATION_RANGE_DEFAULT,
    t0: u.Q = T0_DEFAULT,
) -> tuple[gc.PhaseSpaceCoordinate, OrbitICInfo]:
    """Generate initial conditions for an orbit with specified constraints.

    Args:
        key: JAX random key for reproducible generation.
        potential: Gravitational potential in which the orbit will evolve.
        r_range: Radial range (min, max) for initial position in kpc.
            Default is (20 kpc, 100 kpc).
        e_range: Range of orbital eccentricity (min, max).
            Values must satisfy 0 ≤ min < max < 1 for bound orbits.
            Default is (0.0, 0.3).
        i_range: Range of inclination angles (min, max) in degrees.
            Default is (0°, 90°).
        t0: Initial time for the phase space coordinate.
            Default is 0 Myr.

    Returns:
        Tuple of (initial phase-space coordinates, :class:`OrbitICInfo`).

    Examples:
    --------
    >>> import jax.random as jr
    >>> import galax.potential as gp
    >>> import unxt as u
    >>> from stream_convexity_paper.data import generate_orbit_ic

    >>> key = jr.key(42)
    >>> pot = gp.BovyMWPotential2014()
    >>> w0, info = generate_orbit_ic(key, pot)
    >>> w0.q.x.shape
    ()

    >>> # The sampled properties come back alongside the phase-space point
    >>> bool(0.0 <= info.e < 0.3)
    True

    >>> # With custom constraints
    >>> w0, info = generate_orbit_ic(
    ...     key,
    ...     pot,
    ...     r_range=(u.Q(50.0, "kpc"), u.Q(80.0, "kpc")),
    ...     e_range=(0.05, 0.15),
    ... )
    >>> w0.q.x.shape
    ()

    """
    key_r, key_t, key_p, key_e, key_inc = jr.split(key, 5)

    r_min, r_max = r_range
    r = jr.uniform(key_r, minval=r_min.value, maxval=r_max.value)
    r = u.Q(r, r_min.unit)

    # Generate spherical angles (uniform on sphere) t=azimuthal, p=polar
    theta = u.Q(jr.uniform(key_t, minval=0.0, maxval=2.0 * jnp.pi), "rad")
    phi = u.Q(jnp.arccos(jr.uniform(key_p, minval=-1.0, maxval=1.0)), "rad")

    x = r * jnp.sin(phi) * jnp.cos(theta)
    y = r * jnp.sin(phi) * jnp.sin(theta)
    z = r * jnp.cos(phi)
    q = jnp.stack([x, y, z], axis=-1)  # shape (3,), units of kpc

    e_min, e_max = e_range
    e = jr.uniform(key_e, minval=e_min, maxval=e_max)

    inc_min, inc_max = i_range
    inc = jr.uniform(key_inc, minval=inc_min.value, maxval=inc_max.value)
    inc = u.Q(inc, inc_min.unit)

    # ── Launch speed ────────────────────────────────────────────────────────
    # Solve for the apocentric tangential speed that yields the requested
    # eccentricity in the *actual* potential.  The Keplerian v_circ·sqrt(1-e²)
    # relation is wrong for extended mass profiles (e.g. NFW) even in the
    # spherical limit, mapping the target e to a badly mismatched orbit.
    #
    # Treat the sampled radius as the apocentre r_apo and put the pericentre at
    # r_peri = r_apo (1-e)/(1+e).  Both radial turning points have zero radial
    # velocity, so energy conservation between them fixes the specific angular
    # momentum L:
    #     Φ(r_apo) + L²/(2 r_apo²) = Φ(r_peri) + L²/(2 r_peri²)
    #  ⇒  L² = 2 [Φ(r_peri) − Φ(r_apo)] / (1/r_apo² − 1/r_peri²),
    # and the apocentric speed is purely tangential, v = L / r_apo.  Exact for any
    # spherical potential; for a triaxial potential Φ is sampled along the launch
    # ray, an O(triaxiality) approximation but far better than Keplerian.
    r_apo = r
    ratio = (1.0 - e) / (1.0 + e)  # r_peri / r_apo
    r_peri = r_apo * ratio
    phi_apo = potential.potential(q, t=u.Q(0.0, "Myr"))
    phi_peri = potential.potential(q * ratio, t=u.Q(0.0, "Myr"))
    l_sq = 2.0 * (phi_peri - phi_apo) / (1.0 / r_apo**2 - 1.0 / r_peri**2)
    v_mag = jnp.sqrt(l_sq) / r_apo

    # Circular limit (e→0 is 0/0 above): fall back to the local circular speed.
    acc = potential.acceleration(q, t=u.Q(0.0, "Myr"))
    v_circ = jnp.sqrt(r * jnp.linalg.norm(acc))
    v_mag = jnp.where(e < 1e-4, v_circ, v_mag)

    # Cap at 95% of escape speed where Φ < 0 (e.g. NFW) to guarantee a bound
    # orbit.  Confining potentials with Φ >= 0 are already bound; skip the cap.
    v_esc_sq = -2.0 * phi_apo
    v_esc_safe = jnp.sqrt(jnp.maximum(v_esc_sq, 0.0 * v_esc_sq))
    v_mag = jnp.where(v_esc_sq > 0.0, jnp.minimum(v_mag, 0.95 * v_esc_safe), v_mag)

    # ── Launch direction ────────────────────────────────────────────────────
    # Apocentre ⇒ purely tangential velocity (v·n̂ = 0).  Build an orthonormal
    # pair perpendicular to the radial unit vector n̂ and rotate between them by
    # the inclination, so the speed is exactly v_mag and the radial component is
    # exactly zero for every inclination.  (The previous construction leaked a
    # latitude-dependent radial component and shrank the speed by up to ~30×.)
    n_hat = (q / r).ustrip("")  # (3,) radial unit vector
    horiz = jnp.sqrt(n_hat[0] ** 2 + n_hat[1] ** 2)  # sin(colatitude)
    # Azimuthal (eastward) unit vector ẑ×n̂/|·|; fall back to x̂ at the poles.
    e_az = jnp.where(
        horiz > 1e-8,
        jnp.stack([-n_hat[1], n_hat[0], jnp.zeros_like(n_hat[0])]) / jnp.maximum(horiz, 1e-8),
        jnp.array([1.0, 0.0, 0.0]),
    )
    e_mer = jnp.cross(n_hat, e_az)  # meridional (northward) unit vector
    d_hat = jnp.cos(inc) * e_az + jnp.sin(inc) * e_mer  # inclined tangential direction
    p = v_mag * d_hat  # (3,) velocity

    w0 = gc.PhaseSpaceCoordinate(q=q, p=p, t=t0)
    return w0, OrbitICInfo(r=r, e=e, inc=inc)


@eqx.filter_jit
def generate_orbit(
    potential: gp.AbstractPotential,
    w0: gc.PhaseSpaceCoordinate,
    t: u.Q,
    *,
    dense: bool = False,
    max_steps: int | None = None,
) -> gd.Orbit:
    """Generate orbit by integrating initial conditions in a potential.

    Args:
        potential: Gravitational potential in which the orbit evolves.
        w0: Initial phase-space coordinates.
        t: Times at which to evaluate the orbit. Can be a single time or array.
        dense: Whether to return a dense orbit object.
        max_steps: Maximum number of integration steps.

    Returns:
        Dense orbit object with positions and velocities at requested times.

    Examples:
    --------
    >>> import jax.random as jr
    >>> import galax.potential as gp
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> from stream_convexity_paper.data import generate_orbit_ic, generate_orbit

    >>> key = jr.key(42)
    >>> pot = gp.BovyMWPotential2014()
    >>> w0, _ = generate_orbit_ic(key, pot)
    >>> t = u.Q(jnp.linspace(0, 1, 100), "Gyr")
    >>> orbit = generate_orbit(pot, w0, t)
    >>> orbit.q.x.shape
    (100,)

    """
    solver = None if max_steps is None else gd.OrbitSolver(max_steps=max_steps)
    return gd.compute_orbit(potential, w0, t, dense=dense, solver=solver)


##############################################################################
# Observer


def generate_sphere_angles(n: int, /) -> tuple[Float[u.Q, "n"], Float[u.Q, "n"]]:
    """Generate evenly distributed angles on a unit sphere using Fibonacci lattice.

    Args:
        n: Number of points to generate on the sphere.

    Returns:
        Tuple containing:
            - theta: Polar angles (colatitude) in radians, shape (n,).
            - phi: Azimuthal angles in radians, shape (n,).

    """
    indices = jnp.arange(n)

    golden_ratio = (1 + jnp.sqrt(5)) / 2

    # Theta (polar angle): arccos of evenly spaced z from -1 to 1
    z = 1 - (2 * indices + 1) / n
    theta = jnp.arccos(z)

    # Phi (azimuthal angle): golden angle spiral
    phi = (2 * jnp.pi * indices) / golden_ratio
    phi = jnp.mod(phi, 2 * jnp.pi)

    return u.Q(theta, "rad"), u.Q(phi, "rad")


##############################################################################
# Sky-plane projection and curvature


def _orbit_derivatives(
    t: Float[u.Q, ""],
    orbit: gd.Orbit,
    usys: u.AbstractUnitSystem,
) -> tuple[Float[Array, " 3"], Float[Array, " 3"], Float[Array, " 3"]]:
    """Return (q, dq/dt, d2q/dt2) of the 3D orbit track at time ``t``.

    The position is differentiated in 3D rather than in the sky plane; the two
    agree because the viewing rotation is linear and constant, and this keeps
    the projection in one place (:mod:`~stream_convexity_paper.curvature`).

    Args:
        t: Scalar time at which to evaluate the orbit.
        orbit: Dense orbit interpolant.
        usys: Unit system for stripping units.

    Returns:
        Tuple of position and its first two time derivatives, each shape (3,),
        as plain arrays in ``usys``.

    """
    # Cache the unit of t so we can strip/re-attach units inside the closure
    # without carrying the full Quantity object through JAX tracing.
    t_unit = t.unit

    def _q3(s: float) -> Array:
        """3D position as a plain float array. s is a scalar in t_unit.

        This is the function we differentiate w.r.t. s.  Everything inside must
        be JAX-traceable; units are stripped so only plain arrays are returned.

        stop_gradient on `orbit` prevents JAX from trying to differentiate
        through the underlying ODE solve / interpolant coefficients.  We only
        ever want d(position)/dt, treating the orbit as a fixed parametric
        curve.
        """
        # Evaluate the dense orbit interpolant at time s (re-attach unit for
        # galax's API), then convert the phase-space point to Cartesian.
        w = jax.lax.stop_gradient(orbit)(u.Q(s, t_unit)).vconvert(cx.CartesianPos3D)
        # Strip units to a plain (3,) float array in the chosen unit system
        return convert(w.q, u.Q).ustrip(usys)

    # Work with a dimensionless scalar; jvp requires plain arrays
    t_raw = t.ustrip(t_unit)

    # Fused first + second derivative via a single nested JVP.
    # jvp(f, (t,), (1,)) returns (f(t), df/dt).  Wrapping in another jvp gives
    # ((f, f'), (f', f'')) in one pass: 4 orbit evaluations vs 2+4=6 separately.
    (q, dq), (_, d2q) = jax.jvp(
        lambda s: jax.jvp(_q3, (s,), (1.0,)),
        (t_raw,),
        (1.0,),
    )
    return q, dq, d2q


@eqx.filter_jit
def compute_projected_curvature_normal(
    t: Float[u.Q, ""],
    /,
    orbit: gd.Orbit,
    R: jax.scipy.spatial.transform.Rotation,
    *,
    usys: u.AbstractUnitSystem,
    eps: float = 1e-12,
) -> Float[Array, " 2"]:
    """Compute 2D planar curvature normal from orbit projection.

    Differentiates the projected position track q(t) twice using forward-mode
    autodiff to obtain the Frenet-Serret normal of the sky-plane curve.

        nhat = a_perp / |a_perp|,  a_perp = q'' - (q''·q' / |q'|²) q'

    where q' = dq/dt is the positional tangent in the sky plane.

    Notes:
    -----
    - stop_gradient on orbit prevents tracing back through the ODE solver.
    - When the curvature is zero the normal is undefined; regularize with `eps`.

    Args:
        t: Scalar time at which to evaluate the curvature.
        orbit: Dense orbit interpolant.
        R: Rotation that aligns the viewing direction with +z.
        usys: Unit system for stripping units.
        eps: Regularization to avoid division by zero.

    Returns:
        nhat_xy: Unit normal vector in the rotated x-y plane, shape (2,).

    """
    _, dq, d2q = _orbit_derivatives(t, orbit, usys)
    nhat_xy, _ = _nhat_kappa(dq, d2q, R.as_matrix(), eps=eps)
    return nhat_xy  # (2,) unit normal in the sky plane


@eqx.filter_jit
def compute_projected_curvature_normal_and_kappa(
    t: Float[u.Q, ""],
    /,
    orbit: gd.Orbit,
    R: jax.scipy.spatial.transform.Rotation,
    *,
    usys: u.AbstractUnitSystem,
    eps: float = 1e-12,
) -> tuple[Float[Array, " 2"], Float[Array, ""]]:
    """Compute 2D planar curvature normal and curvature magnitude from orbit projection.

    Differentiates the projected position track q(t) twice using forward-mode
    autodiff to obtain the Frenet-Serret normal and curvature of the sky-plane
    curve.

        nhat = a_perp / |a_perp|,   kappa = |a_perp| / |q'|²
        a_perp = q'' - (q''·q' / |q'|²) q'

    Notes:
    -----
    - stop_gradient on orbit prevents tracing back through the ODE solver.
    - When the curvature is zero the normal is undefined; regularize with `eps`.

    Args:
        t: Scalar time at which to evaluate the curvature.
        orbit: Dense orbit interpolant.
        R: Rotation that aligns the viewing direction with +z.
        usys: Unit system for stripping units.
        eps: Regularization to avoid division by zero.

    Returns:
        nhat_xy: Unit normal vector in the rotated x-y plane, shape (2,).
        kappa_sky: Curvature magnitude (1/length in usys), scalar.

    """
    _, dq, d2q = _orbit_derivatives(t, orbit, usys)
    # κ is in inverse length of usys: [length/time²] / [length/time]² = 1/length
    return _nhat_kappa(dq, d2q, R.as_matrix(), eps=eps)


@ft.partial(vmap, in_axes=(0, 0, None, None), in_kw={"usys": None})
def compute_bad_normals_for_angle(
    theta_i: Float[u.Q, ""],
    phi_i: Float[u.Q, ""],
    t: Float[u.Q, "n"],
    orbit: gd.Orbit,
    *,
    usys: u.AbstractUnitSystem,
) -> Array:
    """Compute the number of bad normals for a single viewing angle (vmapped).

    Args:
        theta_i: Scalar polar viewing angle (colatitude).
        phi_i: Scalar azimuthal viewing angle.
        t: Array of times at which to evaluate the orbit.
        orbit: Orbit to project.
        usys: Unit system for internal conversions.

    Returns:
        Number of normal vectors pointing away from the orbit center.

    """
    R_i = rotation_matrix_from_viewing_angles(theta_i, phi_i)

    nhat_fn = vmap(
        compute_projected_curvature_normal,
        in_axes=(0, None, None),
        static_kw={"usys": usys},
    )
    nhat_geom = nhat_fn(t, orbit, R_i)

    q_2d = quax.quaxify(R_i.apply)(convert(orbit(t).q, u.Q))[:, :2]
    return count_bad_normals(q_2d, nhat_geom)


@ft.partial(vmap, in_axes=(0, None, None), in_kw={"usys": None})
def compute_bad_normals_for_rotation(
    R_i: jax.scipy.spatial.transform.Rotation,
    t: Float[u.Q, "n"],
    orbit: gd.Orbit,
    *,
    usys: u.AbstractUnitSystem,
) -> Array:
    """Compute bad normals for a single precomputed rotation (vmapped over rotations).

    Like ``compute_bad_normals_for_angle`` but accepts a precomputed
    ``Rotation`` instead of raw angles.  Use this when the same viewing-angle
    grid is reused across many orbits: compute the rotations once with
    ``jax.vmap(rotation_matrix_from_viewing_angles)(theta, phi)`` and pass the
    result here to avoid recomputing trigonometry for every orbit.

    Args:
        R_i: Precomputed rotation aligning the viewing direction with +z.
        t: Array of times at which to evaluate the orbit.
        orbit: Orbit to project.
        usys: Unit system for internal conversions.

    Returns:
        Number of normal vectors pointing away from the orbit center.

    """
    nhat_fn = vmap(
        compute_projected_curvature_normal,
        in_axes=(0, None, None),
        static_kw={"usys": usys},
    )
    nhat_geom = nhat_fn(t, orbit, R_i)

    q_2d = quax.quaxify(R_i.apply)(convert(orbit(t).q, u.Q))[:, :2]
    return count_bad_normals(q_2d, nhat_geom)


@ft.partial(vmap, in_axes=(0, 0, None, None), in_kw={"usys": None})
def compute_max_nonconcave_kappa_for_angle(
    theta_i: Float[u.Q, ""],
    phi_i: Float[u.Q, ""],
    t: Float[u.Q, "n"],
    orbit: gd.Orbit,
    *,
    usys: u.AbstractUnitSystem,
) -> Array:
    """Max non-concave curvature κ for a single viewing angle (vmapped).

    Returns the largest κ among arc points where the orbit is non-concave
    (normal pointing away from the projected centre), or 0.0 if none.
    """
    R_i = rotation_matrix_from_viewing_angles(theta_i, phi_i)

    nhat_kappa_fn = vmap(
        compute_projected_curvature_normal_and_kappa,
        in_axes=(0, None, None),
        static_kw={"usys": usys},
    )
    nhats, kappas = nhat_kappa_fn(t, orbit, R_i)

    q_2d = quax.quaxify(R_i.apply)(convert(orbit(t).q, u.Q))[:, :2]
    return jnp.max(jnp.where(is_nonconcave(q_2d, nhats), kappas, 0.0))


@eqx.filter_jit
def compute_sky_tangent_and_nhat(
    t: Float[u.Q, ""],
    /,
    orbit: gd.Orbit,
    R: jax.scipy.spatial.transform.Rotation,
    *,
    usys: u.AbstractUnitSystem,
    eps: float = 1e-12,
) -> tuple[Float[Array, " 2"], Float[Array, " 2"], Float[Array, " 2"]]:
    """Return (q_sky, tangent_hat, nhat) at time t via a single nested JVP."""
    q, dq, d2q = _orbit_derivatives(t, orbit, usys)
    R_mat = R.as_matrix()

    q2d = project_to_sky_plane(q, R_mat)
    dq2d = project_to_sky_plane(dq, R_mat)
    t_hat = dq2d / jnp.maximum(jnp.linalg.norm(dq2d), eps)
    nhat, _ = _nhat_kappa(dq, d2q, R_mat, eps=eps)
    return q2d, t_hat, nhat


@ft.partial(vmap, in_axes=(0, 0, None, None), in_kw={"usys": None})
def compute_max_nonconcave_tangential_offset_for_angle(
    theta_i: Float[u.Q, ""],
    phi_i: Float[u.Q, ""],
    t: Float[u.Q, "n"],
    orbit: gd.Orbit,
    *,
    usys: u.AbstractUnitSystem,
) -> Array:
    """Max |sin(angle(tangent, radial))| at non-concave points (vmapped).

    For each viewing direction, among points where the orbit is non-concave,
    returns the largest |sin| of the angle between the orbit tangent and the
    radial direction to the CoM (i.e. how non-radially the stream is pointing).
    Returns 0.0 if no non-concave points exist.
    """
    R_i = rotation_matrix_from_viewing_angles(theta_i, phi_i)
    fn = vmap(
        compute_sky_tangent_and_nhat,
        in_axes=(0, None, None),
        static_kw={"usys": usys},
    )
    q_all, t_hats, nhats = fn(t, orbit, R_i)  # (N, 2) each

    # Non-concave mask: curvature normal points away from CoM
    q_norms = jnp.linalg.norm(q_all, axis=-1, keepdims=True)
    q_hats = q_all / jnp.maximum(q_norms, 1e-12)
    nonconcave = is_nonconcave(q_all, nhats)

    # |sin(angle between tangent and radial)| via the 2D cross-product magnitude
    sin_angles = jnp.abs(q_hats[:, 0] * t_hats[:, 1] - q_hats[:, 1] * t_hats[:, 0])
    return jnp.max(jnp.where(nonconcave, sin_angles, 0.0))
