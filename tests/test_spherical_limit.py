"""The paper's central geometric claim, tested two ways.

An orbit in a spherical potential is concave from every viewing direction: its
projected curvature normal always has a positive component toward the projected
centre of mass.  Convexity in a stream therefore cannot come from a spherical
halo.

The claim is checked first pointwise from the definition of a central force
(no integration, no potential), then end to end on an orbit integrated in the
spherical limit of the triaxial potential the paper uses, with the triaxial
case as a control so the assertion cannot pass vacuously.
"""

import coordinax as cx
import galax.coordinates as gc
import galax.potential as gp
import jax.random as jr
import numpy as np
import pytest
from plum import convert
from scipy.spatial.transform import Rotation as ScipyRotation

import quaxed.numpy as jnp
import unxt as u
from stream_convexity_paper.curvature import (
    count_bad_normals,
    project_to_sky_plane,
    projected_curvature_normal_and_kappa,
)
from stream_convexity_paper.data import (
    compute_bad_normals_for_angle,
    compute_max_nonconcave_kappa_for_angle,
    compute_max_nonconcave_tangential_offset_for_angle,
    generate_orbit,
    generate_orbit_ic,
    generate_sphere_angles,
    usys,
)

RNG = np.random.default_rng(20260915)

#: Viewing directions swept in the orbit-level tests.
N_VIEWS = 200

#: Reference orbit used for the orbit-level tests.  Fixed rather than sampled so
#: a failure is reproducible, and deliberately inclined so that no viewing
#: direction in the lattice lands edge-on to the orbital plane, where the
#: projected track degenerates to a segment and the normal is undefined.
W0 = gc.PhaseSpaceCoordinate(
    q=u.Q([30.0, 10.0, 15.0], "kpc"),
    p=u.Q([0.02, 0.15, 0.06], "kpc/Myr"),
    t=u.Q(0.0, "Myr"),
)
T_EVAL = u.Q(jnp.linspace(0.0, 3000.0, 200), "Myr")


def nfw(q1: float, q2: float) -> gp.TriaxialNFWPotential:
    """Triaxial NFW halo; ``q1 = q2 = 1`` is the spherical limit."""
    return gp.TriaxialNFWPotential(
        m=u.Q(1e12, "Msun"),
        r_s=u.Q(15.0, "kpc"),
        q1=q1,
        q2=q2,
        units="galactic",
    )


@pytest.fixture(scope="module")
def spherical_orbit() -> object:
    """Dense orbit in the spherical limit."""
    return generate_orbit(nfw(1.0, 1.0), W0, T_EVAL, dense=True)


@pytest.fixture(scope="module")
def triaxial_orbit() -> object:
    """Dense orbit in a markedly triaxial halo (the control)."""
    return generate_orbit(nfw(0.8, 0.6), W0, T_EVAL, dense=True)


@pytest.fixture(scope="module")
def views() -> tuple[u.Q, u.Q]:
    """Fibonacci lattice of viewing angles."""
    return generate_sphere_angles(N_VIEWS)


# ════════════════════════════════════════════════════════════════════════════
# Pointwise: the claim follows from the direction of the acceleration alone
# ════════════════════════════════════════════════════════════════════════════


def central_force_sample(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random positions and velocities with acceleration pointing at the origin."""
    q = RNG.normal(size=(n, 3)) * 20.0
    v = RNG.normal(size=(n, 3))
    strength = RNG.uniform(0.1, 5.0, size=(n, 1))
    a = -strength * q / np.linalg.norm(q, axis=-1, keepdims=True)
    return q, v, a


@pytest.mark.parametrize("seed", range(4))
def test_central_force_is_concave_from_every_direction(seed: int) -> None:
    """A central force gives a concave projection for any viewing rotation.

    Writing ``a = -A q̂`` and projecting with the linear map ``P``, the normal
    component is ``a_perp = -A (P q̂)_perp``, so

        n̂ · (-q̂_2d) ∝ (P q̂)_perp · (P q̂) = |(P q̂)_perp|² >= 0.

    No planarity or integration is needed: it holds instant by instant.
    """
    q, v, a = central_force_sample(256)
    R = ScipyRotation.random(1, rng=seed).as_matrix()[0]

    nhat, _ = projected_curvature_normal_and_kappa(v, a, R)
    q_2d = project_to_sky_plane(q, R)

    assert int(count_bad_normals(q_2d, nhat)) == 0


def test_central_force_concavity_has_margin() -> None:
    """The inward dot product is strictly positive, not merely non-negative.

    Equality requires the projected radius and velocity to be parallel, which
    random samples do not hit.
    """
    q, v, a = central_force_sample(512)
    R = ScipyRotation.random(1, rng=7).as_matrix()[0]

    nhat, _ = projected_curvature_normal_and_kappa(v, a, R)
    q_2d = np.asarray(project_to_sky_plane(q, R))
    q_hat = q_2d / np.linalg.norm(q_2d, axis=-1, keepdims=True)
    dots = np.sum(np.asarray(nhat) * -q_hat, axis=-1)

    assert dots.min() > 0.0


def test_noncentral_force_can_be_convex() -> None:
    """The control for the pointwise test: a non-radial force need not be concave.

    Guards against the assertion passing because of how the normal is built
    rather than because the force points at the centre.
    """
    q, v, _ = central_force_sample(512)
    a_outward = +q / np.linalg.norm(q, axis=-1, keepdims=True)
    R = ScipyRotation.random(1, rng=7).as_matrix()[0]

    nhat, _ = projected_curvature_normal_and_kappa(v, a_outward, R)
    q_2d = project_to_sky_plane(q, R)

    assert int(count_bad_normals(q_2d, nhat)) == q.shape[0]


# ════════════════════════════════════════════════════════════════════════════
# End to end: an integrated orbit, swept over viewing directions
# ════════════════════════════════════════════════════════════════════════════


def test_spherical_orbit_has_no_convexity_from_any_view(
    spherical_orbit: object,
    views: tuple[u.Q, u.Q],
) -> None:
    """The headline claim: zero non-concave points over the whole sphere."""
    theta, phi = views
    counts = compute_bad_normals_for_angle(
        theta,
        phi,
        T_EVAL,
        spherical_orbit,
        usys=usys,
    )

    assert int(jnp.max(counts)) == 0


def test_triaxial_orbit_does_have_convexity(
    triaxial_orbit: object,
    views: tuple[u.Q, u.Q],
) -> None:
    """Control: the same orbit in a triaxial halo is convex from many views.

    Without this, the test above would also pass if the counting were broken.
    """
    theta, phi = views
    counts = compute_bad_normals_for_angle(
        theta,
        phi,
        T_EVAL,
        triaxial_orbit,
        usys=usys,
    )

    assert int(jnp.max(counts)) > 0
    assert int(jnp.sum(counts > 0)) > N_VIEWS // 10


def test_max_nonconcave_kappa_vanishes_in_the_spherical_limit(
    spherical_orbit: object,
    views: tuple[u.Q, u.Q],
) -> None:
    """With no non-concave points, the largest non-concave κ is exactly zero."""
    theta, phi = views
    kappa_max = compute_max_nonconcave_kappa_for_angle(
        theta,
        phi,
        T_EVAL,
        spherical_orbit,
        usys=usys,
    )

    np.testing.assert_array_equal(np.asarray(kappa_max), 0.0)


def test_max_nonconcave_kappa_is_positive_when_triaxial(
    triaxial_orbit: object,
    views: tuple[u.Q, u.Q],
) -> None:
    """Control for the reduction above."""
    theta, phi = views
    kappa_max = compute_max_nonconcave_kappa_for_angle(
        theta,
        phi,
        T_EVAL,
        triaxial_orbit,
        usys=usys,
    )

    assert float(jnp.max(kappa_max)) > 0.0


def test_max_tangential_offset_vanishes_in_the_spherical_limit(
    spherical_orbit: object,
    views: tuple[u.Q, u.Q],
) -> None:
    """No non-concave points means no tangential offset to report."""
    theta, phi = views
    offsets = compute_max_nonconcave_tangential_offset_for_angle(
        theta,
        phi,
        T_EVAL,
        spherical_orbit,
        usys=usys,
    )

    np.testing.assert_array_equal(np.asarray(offsets), 0.0)


def test_max_tangential_offset_is_a_sine(
    triaxial_orbit: object,
    views: tuple[u.Q, u.Q],
) -> None:
    """The offset is |sin| of an angle, so it lies in [0, 1]."""
    theta, phi = views
    offsets = np.asarray(
        compute_max_nonconcave_tangential_offset_for_angle(
            theta,
            phi,
            T_EVAL,
            triaxial_orbit,
            usys=usys,
        ),
    )

    assert offsets.min() >= 0.0
    assert offsets.max() <= 1.0
    assert offsets.max() > 0.0


# ════════════════════════════════════════════════════════════════════════════
# Viewing-angle lattice
# ════════════════════════════════════════════════════════════════════════════


def test_sphere_angles_cover_the_sphere() -> None:
    """Fibonacci lattice: angles in range, evenly spaced in z, centroid at origin."""
    n = 4096
    theta, phi = generate_sphere_angles(n)
    th = np.asarray(theta.ustrip("rad"))
    ph = np.asarray(phi.ustrip("rad"))

    assert th.min() >= 0.0
    assert th.max() <= np.pi
    assert ph.min() >= 0.0
    assert ph.max() < 2 * np.pi

    # z is the evenly spaced midpoint grid on (-1, 1)
    z = np.cos(th)
    np.testing.assert_allclose(np.sort(z), np.sort(1 - (2 * np.arange(n) + 1) / n))

    # A uniform cover has its centroid at the origin.
    xyz = np.stack(
        [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), z],
        axis=-1,
    )
    assert np.linalg.norm(xyz.mean(axis=0)) < 1e-3


# ════════════════════════════════════════════════════════════════════════════
# Initial conditions: the launch solution quoted in generate_orbit_ic
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("seed", range(4))
def test_launch_velocity_is_purely_tangential(seed: int) -> None:
    """Apocentric launch means the radial velocity is zero, exactly."""
    pot = gp.NFWPotential(m=u.Q(1e12, "Msun"), r_s=u.Q(15.0, "kpc"), units="galactic")
    w0, _ = generate_orbit_ic(jr.key(seed), pot, e_range=(0.05, 0.6))

    q = np.asarray(convert(w0.q, u.Q).ustrip("kpc"))
    p = np.asarray(convert(w0.p, u.Q).ustrip("kpc/Myr"))

    radial_speed = np.dot(p, q / np.linalg.norm(q))
    assert abs(radial_speed) < 1e-12 * np.linalg.norm(p)


@pytest.mark.parametrize("seed", [0, 3])
def test_requested_eccentricity_is_realised(seed: int) -> None:
    """The energy-conservation solution for L² reproduces the requested e.

    ``generate_orbit_ic`` solves Φ(r_apo) + L²/2r_apo² = Φ(r_peri) + L²/2r_peri²
    instead of using the Keplerian relation, which is wrong for an extended
    profile.  Integrating the result must return the eccentricity asked for,
    and the sampled radius must be the apocentre.
    """
    pot = gp.NFWPotential(m=u.Q(1e12, "Msun"), r_s=u.Q(15.0, "kpc"), units="galactic")
    w0, info = generate_orbit_ic(jr.key(seed), pot, e_range=(0.05, 0.6))

    t = u.Q(jnp.linspace(0.0, 6000.0, 3000), "Myr")
    orbit = generate_orbit(pot, w0, t)
    q3 = convert(orbit.q.vconvert(cx.CartesianPos3D), u.Q).ustrip("kpc")
    r = np.linalg.norm(np.asarray(q3), axis=-1)

    r_peri, r_apo = r.min(), r.max()
    e_measured = (r_apo - r_peri) / (r_apo + r_peri)

    assert e_measured == pytest.approx(float(info.e), abs=1e-3)
    assert r_apo == pytest.approx(float(info.r.ustrip("kpc")), rel=1e-3)


def test_circular_limit_keeps_the_radius_constant() -> None:
    """E → 0 falls back to the local circular speed, giving a constant radius."""
    pot = gp.NFWPotential(m=u.Q(1e12, "Msun"), r_s=u.Q(15.0, "kpc"), units="galactic")
    w0, info = generate_orbit_ic(jr.key(11), pot, e_range=(0.0, 0.0))

    t = u.Q(jnp.linspace(0.0, 6000.0, 2000), "Myr")
    orbit = generate_orbit(pot, w0, t)
    q3 = convert(orbit.q.vconvert(cx.CartesianPos3D), u.Q).ustrip("kpc")
    r = np.linalg.norm(np.asarray(q3), axis=-1)

    assert float(info.e) == 0.0
    np.testing.assert_allclose(r, r[0], rtol=1e-6)
