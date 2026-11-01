"""Unit tests for the sky-plane curvature geometry.

These check the kernel against closed forms and against the invariances the
construction is supposed to have.  Nothing here integrates an orbit, so the
whole module runs in about a second.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as ScipyRotation

import unxt as u
from stream_convexity_paper.curvature import (
    count_bad_normals,
    is_nonconcave,
    project_to_sky_plane,
    projected_curvature_normal_and_kappa,
    rotation_matrix_from_viewing_angles,
)

RNG = np.random.default_rng(20260915)


def random_rotation(n: int = 8) -> np.ndarray:
    """Return ``n`` random rotation matrices."""
    return ScipyRotation.random(n, rng=42).as_matrix()


def circle_derivatives(
    t: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Position and first two derivatives of a radius-``radius`` circle in z=0."""
    q = radius * np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=-1)
    dq = radius * np.stack([-np.sin(t), np.cos(t), np.zeros_like(t)], axis=-1)
    d2q = -q
    return q, dq, d2q


# ════════════════════════════════════════════════════════════════════════════
# Closed forms
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("radius", [0.5, 1.0, 3.0, 42.0])
def test_kappa_of_circle_is_inverse_radius(radius: float) -> None:
    """A circle of radius R seen face-on has curvature 1/R everywhere."""
    t = np.linspace(0.0, 2 * np.pi, 17)
    _, dq, d2q = circle_derivatives(t, radius)

    _, kappa = projected_curvature_normal_and_kappa(dq, d2q, np.eye(3))

    np.testing.assert_allclose(kappa, 1.0 / radius, rtol=1e-12)


def test_kappa_matches_the_planar_cross_product_formula() -> None:
    """κ agrees with |x'y'' - y'x''| / |q'|³ computed independently."""
    dq3 = RNG.normal(size=(64, 3))
    d2q3 = RNG.normal(size=(64, 3))
    R = random_rotation(1)[0]

    _, kappa = projected_curvature_normal_and_kappa(dq3, d2q3, R)

    v = np.asarray(project_to_sky_plane(dq3, R))
    a = np.asarray(project_to_sky_plane(d2q3, R))
    expected = np.abs(v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0]) / np.linalg.norm(
        v,
        axis=-1,
    ) ** 3

    np.testing.assert_allclose(kappa, expected, rtol=1e-10)


def test_normal_is_unit_length_and_orthogonal_to_the_tangent() -> None:
    """n̂ is a unit vector perpendicular to the projected tangent."""
    dq3 = RNG.normal(size=(128, 3))
    d2q3 = RNG.normal(size=(128, 3))
    R = random_rotation(1)[0]

    nhat, _ = projected_curvature_normal_and_kappa(dq3, d2q3, R)
    v = np.asarray(project_to_sky_plane(dq3, R))
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)

    np.testing.assert_allclose(np.linalg.norm(nhat, axis=-1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(np.asarray(nhat) * v, axis=-1), 0.0, atol=1e-12)


def test_normal_points_toward_the_centre_of_curvature() -> None:
    """On a circle, n̂ points at the centre: n̂ = -q̂."""
    t = np.linspace(0.0, 2 * np.pi, 17)
    q, dq, d2q = circle_derivatives(t, 2.5)

    nhat, _ = projected_curvature_normal_and_kappa(dq, d2q, np.eye(3))

    q_hat = q[:, :2] / np.linalg.norm(q[:, :2], axis=-1, keepdims=True)
    np.testing.assert_allclose(nhat, -q_hat, atol=1e-12)


# ════════════════════════════════════════════════════════════════════════════
# Invariances
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("scale", [0.1, 2.0, 17.0])
def test_invariant_under_reparameterisation(scale: float) -> None:
    """Rescaling the parameter (dq→c·dq, d2q→c²·d2q) changes nothing.

    Curvature is a property of the curve, not of how it is traversed.
    """
    dq3 = RNG.normal(size=(32, 3))
    d2q3 = RNG.normal(size=(32, 3))
    R = random_rotation(1)[0]

    nhat, kappa = projected_curvature_normal_and_kappa(dq3, d2q3, R)
    nhat_s, kappa_s = projected_curvature_normal_and_kappa(
        scale * dq3,
        scale**2 * d2q3,
        R,
    )

    np.testing.assert_allclose(nhat_s, nhat, atol=1e-12)
    np.testing.assert_allclose(kappa_s, kappa, rtol=1e-10)


@pytest.mark.parametrize("lam", [-3.0, 0.25, 4.0])
def test_invariant_under_tangential_contamination(lam: float) -> None:
    """Adding any multiple of the tangent to q'' leaves n̂ and κ alone.

    This is the whole point of subtracting the parallel component: a
    non-uniform traversal speed adds exactly such a term.
    """
    dq3 = RNG.normal(size=(32, 3))
    d2q3 = RNG.normal(size=(32, 3))
    R = random_rotation(1)[0]

    nhat, kappa = projected_curvature_normal_and_kappa(dq3, d2q3, R)
    nhat_c, kappa_c = projected_curvature_normal_and_kappa(
        dq3,
        d2q3 + lam * dq3,
        R,
    )

    np.testing.assert_allclose(nhat_c, nhat, atol=1e-10)
    np.testing.assert_allclose(kappa_c, kappa, rtol=1e-10)


def test_equivariant_under_rotation_about_the_line_of_sight() -> None:
    """Spinning the sky plane by α rotates n̂ by α and leaves κ unchanged."""
    dq3 = RNG.normal(size=(32, 3))
    d2q3 = RNG.normal(size=(32, 3))
    alpha = 0.7
    spin = ScipyRotation.from_rotvec([0.0, 0.0, alpha]).as_matrix()

    nhat, kappa = projected_curvature_normal_and_kappa(dq3, d2q3, np.eye(3))
    nhat_spun, kappa_spun = projected_curvature_normal_and_kappa(dq3, d2q3, spin)

    c, s = np.cos(alpha), np.sin(alpha)
    expected = np.stack(
        [c * nhat[:, 0] - s * nhat[:, 1], s * nhat[:, 0] + c * nhat[:, 1]],
        axis=-1,
    )
    np.testing.assert_allclose(nhat_spun, expected, atol=1e-12)
    np.testing.assert_allclose(kappa_spun, kappa, rtol=1e-10)


# ════════════════════════════════════════════════════════════════════════════
# Degenerate cases
# ════════════════════════════════════════════════════════════════════════════


def test_straight_line_has_zero_curvature_and_no_nans() -> None:
    """A track with q'' parallel to q' is straight: κ = 0, n̂ finite."""
    dq3 = np.tile(np.array([1.0, 2.0, -0.5]), (5, 1))
    d2q3 = 3.0 * dq3

    nhat, kappa = projected_curvature_normal_and_kappa(dq3, d2q3, np.eye(3))

    np.testing.assert_allclose(kappa, 0.0, atol=1e-10)
    assert np.all(np.isfinite(np.asarray(nhat)))


def test_edge_on_projection_of_a_planar_track_is_straight_with_a_cusp() -> None:
    """Viewing a planar circle edge-on collapses it to a traversed segment.

    Away from the fold the projected track is straight, so κ = 0.  At the fold
    the projected speed vanishes and the curve genuinely has a cusp: κ is
    unbounded there, and the ``eps`` floor is all that keeps it finite.  This
    is why consumers that maximise κ over viewing angles must exclude
    near-edge-on views rather than trusting the maximum.
    """
    t = np.linspace(0.2, 2 * np.pi - 0.2, 21)
    _, dq, d2q = circle_derivatives(t, 3.0)
    edge_on = ScipyRotation.from_rotvec([np.pi / 2, 0.0, 0.0]).as_matrix()

    nhat, kappa = projected_curvature_normal_and_kappa(dq, d2q, edge_on)
    kappa = np.asarray(kappa)

    assert np.all(np.isfinite(np.asarray(nhat)))
    assert np.all(np.isfinite(kappa))

    # One sample sits on the fold (t = π, where the tangent is along the line
    # of sight); every other sample lies on the straight part.
    fold = np.argmax(kappa)
    assert np.isclose(t[fold], np.pi, atol=1e-9)
    np.testing.assert_allclose(np.delete(kappa, fold), 0.0, atol=1e-10)
    assert kappa[fold] > 1.0 / 3.0  # far above the face-on value 1/R


# ════════════════════════════════════════════════════════════════════════════
# Concavity criterion
# ════════════════════════════════════════════════════════════════════════════


def test_concave_and_nonconcave_normals_are_classified() -> None:
    """A normal pointing at the origin is concave; its negation is not."""
    q_2d = np.array([[3.0, 0.0], [0.0, -2.0]])
    inward = np.array([[-1.0, 0.0], [0.0, 1.0]])

    assert not np.any(np.asarray(is_nonconcave(q_2d, inward)))
    assert np.all(np.asarray(is_nonconcave(q_2d, -inward)))
    assert int(count_bad_normals(q_2d, inward)) == 0
    assert int(count_bad_normals(q_2d, -inward)) == q_2d.shape[0]


def test_tangential_normal_counts_as_nonconcave() -> None:
    """A normal exactly perpendicular to the radius is the boundary case.

    The criterion is ``dot <= 0``, so the boundary is counted as non-concave.
    This pins the convention for every call site.
    """
    q_2d = np.array([[5.0, 0.0]])
    tangential = np.array([[0.0, 1.0]])

    assert bool(is_nonconcave(q_2d, tangential)[0])


def test_criterion_accepts_positions_with_units() -> None:
    """``q_2d`` may be a Quantity: the orbit code passes kpc."""
    q_2d = u.Q(np.array([[3.0, 0.0], [0.0, -2.0]]), "kpc")
    inward = np.array([[-1.0, 0.0], [0.0, 1.0]])

    assert int(count_bad_normals(q_2d, inward)) == 0


def test_criterion_is_invariant_under_reflection() -> None:
    """Mirroring the sky plane flips q and n̂ together, so the verdict holds."""
    q_2d = RNG.normal(size=(64, 2))
    nhat = RNG.normal(size=(64, 2))
    nhat /= np.linalg.norm(nhat, axis=-1, keepdims=True)
    mirror = np.array([1.0, -1.0])

    np.testing.assert_array_equal(
        np.asarray(is_nonconcave(q_2d, nhat)),
        np.asarray(is_nonconcave(q_2d * mirror, nhat * mirror)),
    )


# ════════════════════════════════════════════════════════════════════════════
# Viewing rotations
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("theta", "phi"),
    [
        (0.0, 0.0),  # already +z
        (np.pi, 0.0),  # antipode: needs a half turn, not the identity
        (np.pi, 1.3),
        (np.pi / 2, 0.0),
        (0.7, 2.0),
        (2.9, 5.5),
    ],
)
def test_rotation_puts_the_viewing_direction_on_z(theta: float, phi: float) -> None:
    """R maps the viewing direction to +z, for every angle including the poles."""
    R = rotation_matrix_from_viewing_angles(u.Q(theta, "rad"), u.Q(phi, "rad"))
    view_dir = np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
    )

    np.testing.assert_allclose(
        np.asarray(R.apply(view_dir)),
        [0.0, 0.0, 1.0],
        atol=1e-8,
    )


def test_rotation_is_a_proper_rotation() -> None:
    """R is orthogonal with unit determinant: no reflections, no scaling."""
    theta = RNG.uniform(0.0, np.pi, size=16)
    phi = RNG.uniform(0.0, 2 * np.pi, size=16)

    for th, ph in zip(theta, phi, strict=True):
        mat = np.asarray(
            rotation_matrix_from_viewing_angles(
                u.Q(th, "rad"),
                u.Q(ph, "rad"),
            ).as_matrix(),
        )
        np.testing.assert_allclose(mat @ mat.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(mat), 1.0, atol=1e-12)


def test_project_to_sky_plane_matches_rotation_apply() -> None:
    """The kernel's matrix projection agrees with ``Rotation.apply``."""
    v3 = RNG.normal(size=(32, 3))
    R = rotation_matrix_from_viewing_angles(u.Q(1.1, "rad"), u.Q(0.4, "rad"))

    np.testing.assert_allclose(
        np.asarray(project_to_sky_plane(v3, R.as_matrix())),
        np.asarray(R.apply(v3))[:, :2],
        atol=1e-12,
    )
