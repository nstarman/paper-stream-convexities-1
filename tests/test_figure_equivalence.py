"""The shared kernel reproduces the implementations it replaced.

Before this suite existed the projected-curvature construction was written out
four times: once per figure script and twice in ``data.py``.  The copies are
reproduced verbatim below as references, so that collapsing them into
:mod:`stream_convexity_paper.curvature` can be shown not to have changed any
number the figures depend on.

Two deliberate differences are asserted rather than hidden:

* the jax rotation now handles the antipode like the numpy one always did;
* the non-concavity test is ``dot <= 0`` everywhere, where one call site used
  ``dot < 0`` with a different regularisation.

Both are covered at the end of this module.  Delete this file once the old
implementations stop being interesting.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as ScipyRotation

import unxt as u
from stream_convexity_paper.curvature import (
    count_bad_normals,
    is_nonconcave,
    projected_curvature_normal_and_kappa,
    rotation_matrix_from_viewing_angles,
)

RNG = np.random.default_rng(31415)


# ════════════════════════════════════════════════════════════════════════════
# The replaced implementations, verbatim
# ════════════════════════════════════════════════════════════════════════════


def ref_gd1_projected_curvature_normal(
    tangent_3d: np.ndarray,
    curvature_3d: np.ndarray,
    R: ScipyRotation,
    eps: float = 1e-12,
) -> np.ndarray:
    """Was ``figures/GD1/plot_gd1_convexities.py::projected_curvature_normal``."""
    v_xy = R.apply(tangent_3d)[:, :2]
    a_xy = R.apply(curvature_3d)[:, :2]
    speed2 = jnp.maximum(jnp.sum(v_xy * v_xy, axis=-1, keepdims=True), eps)
    a_para = (jnp.sum(a_xy * v_xy, axis=-1, keepdims=True) / speed2) * v_xy
    a_perp = a_xy - a_para
    norm = jnp.maximum(jnp.linalg.norm(a_perp, axis=-1, keepdims=True), eps)
    return a_perp / norm


def ref_ridgeline_projected_curvature_normal(
    tangent_3d: np.ndarray,
    curvature_3d: np.ndarray,
    r: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Was ``figures/streams/viz_ridgeline.py::_compute_projected_curvature_normal``."""
    r_j = jnp.asarray(r)
    v2 = jnp.dot(tangent_3d, r_j.T)[:, :2]
    a2 = jnp.dot(curvature_3d, r_j.T)[:, :2]
    speed2 = jnp.maximum(jnp.sum(v2 * v2, axis=-1, keepdims=True), eps)
    av = jnp.sum(a2 * v2, axis=-1, keepdims=True)
    a_perp = a2 - (av / speed2) * v2
    a_perp_norm = jnp.maximum(jnp.linalg.norm(a_perp, axis=-1, keepdims=True), eps)
    nhat = a_perp / a_perp_norm
    kappa = a_perp_norm[..., 0] / speed2[..., 0]
    return np.asarray(nhat), np.asarray(kappa)


def ref_data_frenet_on_projected(
    dq: np.ndarray,
    d2q: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Was the inline algebra in ``data.py``, acting on already-projected 2D."""
    speed2 = jnp.maximum(jnp.dot(dq, dq), eps)
    a_para = (jnp.dot(d2q, dq) / speed2) * dq
    a_perp = d2q - a_para
    a_perp_norm = jnp.maximum(jnp.linalg.norm(a_perp), eps)
    return a_perp / a_perp_norm, a_perp_norm / speed2


def ref_numpy_rotation(theta: float, phi: float) -> np.ndarray:
    """Was ``figures/streams/viz_ridgeline.py::rotation_matrix_from_viewing_angles``."""
    view_dir = np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
    )
    z_axis = np.array([0.0, 0.0, 1.0])
    rot_axis = np.cross(view_dir, z_axis)
    rot_axis_norm = np.linalg.norm(rot_axis)

    if rot_axis_norm < 1e-8:
        if np.dot(view_dir, z_axis) > 0:
            return np.eye(3)
        return np.diag([1.0, -1.0, -1.0])

    k = rot_axis / rot_axis_norm
    cos_angle = np.dot(view_dir, z_axis)
    sin_angle = rot_axis_norm
    kx, ky, kz = k
    k_mat = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    return np.eye(3) + sin_angle * k_mat + (1 - cos_angle) * (k_mat @ k_mat)


def ref_count_bad_normals(q_2d: np.ndarray, nhat_2d: np.ndarray) -> int:
    """Was ``data.py::count_bad_normals``."""
    dir_to_center = -q_2d / jnp.linalg.norm(q_2d, axis=-1, keepdims=True)
    dot_products = jnp.sum(nhat_2d * dir_to_center, axis=-1)
    return jnp.sum(dot_products <= 0)


# ════════════════════════════════════════════════════════════════════════════
# Equivalence
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("seed", range(4))
def test_matches_the_gd1_implementation(seed: int) -> None:
    """Same normals as the GD-1 figure computed before the refactor."""
    tangent = RNG.normal(size=(128, 3))
    curvature = RNG.normal(size=(128, 3))
    R = ScipyRotation.random(1, rng=seed)[0]

    nhat, _ = projected_curvature_normal_and_kappa(
        tangent,
        curvature,
        R.as_matrix(),
    )

    np.testing.assert_allclose(
        np.asarray(nhat),
        np.asarray(ref_gd1_projected_curvature_normal(tangent, curvature, R)),
        atol=1e-14,
    )


@pytest.mark.parametrize("seed", range(4))
def test_matches_the_ridgeline_implementation(seed: int) -> None:
    """Same normals and curvatures as the ridgeline figures computed before."""
    tangent = RNG.normal(size=(128, 3))
    curvature = RNG.normal(size=(128, 3))
    r = ScipyRotation.random(1, rng=seed).as_matrix()[0]

    nhat, kappa = projected_curvature_normal_and_kappa(tangent, curvature, r)
    ref_nhat, ref_kappa = ref_ridgeline_projected_curvature_normal(
        tangent,
        curvature,
        r,
    )

    np.testing.assert_allclose(np.asarray(nhat), ref_nhat, atol=1e-14)
    np.testing.assert_allclose(np.asarray(kappa), ref_kappa, rtol=1e-13)


def test_matches_the_inline_orbit_algebra() -> None:
    """Projecting then decomposing equals decomposing the projected derivatives.

    Decomposing the projected derivatives agrees with differentiating the
    sky-plane position directly, because the viewing rotation is linear and
    constant.
    """
    R = ScipyRotation.random(1, rng=99).as_matrix()[0]

    for _ in range(32):
        dq3 = RNG.normal(size=3)
        d2q3 = RNG.normal(size=3)

        nhat, kappa = projected_curvature_normal_and_kappa(dq3, d2q3, R)
        ref_nhat, ref_kappa = ref_data_frenet_on_projected(
            (dq3 @ R.T)[:2],
            (d2q3 @ R.T)[:2],
        )

        np.testing.assert_allclose(np.asarray(nhat), np.asarray(ref_nhat), atol=1e-14)
        np.testing.assert_allclose(float(kappa), float(ref_kappa), rtol=1e-13)


def test_matches_the_old_bad_normal_count() -> None:
    """Counting is unchanged for ordinary (non-degenerate) inputs."""
    q_2d = RNG.normal(size=(256, 2)) * 10.0
    nhat = RNG.normal(size=(256, 2))
    nhat = nhat / np.linalg.norm(nhat, axis=-1, keepdims=True)

    assert int(count_bad_normals(q_2d, nhat)) == int(ref_count_bad_normals(q_2d, nhat))


@pytest.mark.parametrize(
    ("theta", "phi"),
    [(0.0, 0.0), (np.pi, 0.0), (np.pi, 2.2), (0.3, 1.0), (1.9, 4.4), (3.0, 0.1)],
)
def test_matches_the_numpy_rotation_including_the_antipode(
    theta: float,
    phi: float,
) -> None:
    """The jax rotation agrees with the numpy one everywhere.

    ``theta = pi`` is the case to watch: returning the identity there would
    leave the line of sight on -z, a half turn from correct, which mirrors the
    sky plane.  The concavity test is reflection invariant, so figure verdicts
    would survive that, but plotted orientations would not.
    """
    mat = np.asarray(
        rotation_matrix_from_viewing_angles(
            u.Q(theta, "rad"),
            u.Q(phi, "rad"),
        ).as_matrix(),
    )

    np.testing.assert_allclose(mat, ref_numpy_rotation(theta, phi), atol=1e-8)


def test_boundary_convention_is_the_only_counting_difference() -> None:
    """``dot <= 0`` vs ``dot < 0`` differ only exactly on the boundary.

    ``ridgeline_sky_stats`` used the strict form.  The two disagree only for a
    normal exactly perpendicular to the radius, which no sampled track hits.
    """
    q_2d = np.array([[4.0, 0.0], [4.0, 0.0], [4.0, 0.0]])
    nhat = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    dot = np.sum(nhat * (-q_2d / np.linalg.norm(q_2d, axis=-1, keepdims=True)), axis=-1)
    strict = dot < 0
    inclusive = np.asarray(is_nonconcave(q_2d, nhat))

    np.testing.assert_array_equal(inclusive, dot <= 0)
    assert list(strict) == [False, False, True]
    assert list(inclusive) == [False, True, True]
