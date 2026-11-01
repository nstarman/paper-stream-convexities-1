"""Sky-plane curvature geometry.

The single implementation of the projected Frenet-Serret construction used
throughout the paper: given the first and second derivatives of a 3D track and
a rotation that puts the line of sight on +z, return the unit curvature normal
and the curvature of the projected sky-plane curve, and decide whether that
normal points away from the projected centre of mass.

Every consumer (orbit projections in :mod:`~stream_convexity_paper.data`, the
GD-1 figure, the ridgeline figures) shares these functions so that the
convexity criterion is defined in exactly one place.
"""

__all__ = (
    "count_bad_normals",
    "is_nonconcave",
    "project_to_sky_plane",
    "projected_curvature_normal_and_kappa",
    "rotation_matrix_from_viewing_angles",
)

from jaxtyping import Array, Float

import equinox as eqx
import jax
import jax.scipy.spatial.transform  # jax.scipy submodules load lazily

import quaxed.numpy as jnp
import unxt as u

#: Regularisation floor for the denominators below.  The normal is undefined
#: where the projected curve is locally straight or stationary; clamping keeps
#: the result finite (and the direction arbitrary) instead of NaN.
EPS_DEFAULT = 1e-12


@eqx.filter_jit
def rotation_matrix_from_viewing_angles(
    theta: Float[u.Q, ""],
    phi: Float[u.Q, ""],
) -> jax.scipy.spatial.transform.Rotation:
    """Compute rotation matrix to align viewing direction with z-axis.

    Args:
        theta: Polar viewing angle (colatitude).
        phi: Azimuthal viewing angle.

    Returns:
        Rotation matrix that rotates the viewing direction to [0, 0, 1].

    """
    view_dir = jnp.array(
        [
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta),
        ],
    )

    z_axis = jnp.array([0.0, 0.0, 1.0])

    axis = jnp.cross(view_dir, z_axis)
    axis_norm = jnp.linalg.norm(axis)

    cos_angle = jnp.dot(view_dir, z_axis)
    angle = jnp.arccos(cos_angle)

    # The cross product vanishes both when the viewing direction already is +z
    # (no rotation) and when it is -z (a half turn about any perpendicular axis,
    # here x).  Treating the antipode as "no rotation" would leave the line of
    # sight on -z.
    degenerate = u.Q(
        jnp.where(cos_angle > 0, 0.0, jnp.pi) * jnp.array([1.0, 0.0, 0.0]),
        "rad",
    )

    # Rotation vector: axis * angle (avoid division by zero for aligned vectors)
    rotvec = jnp.where(
        axis_norm > 1e-10,
        (axis / jnp.maximum(axis_norm, 1e-10)) * angle,
        degenerate,
    ).ustrip("rad")

    return jax.scipy.spatial.transform.Rotation.from_rotvec(rotvec)


def project_to_sky_plane(
    v3: Float[Array, "*batch 3"],
    R: Float[Array, "3 3"],
) -> Float[Array, "*batch 2"]:
    """Rotate 3D vectors so the line of sight is +z and keep the sky plane.

    Args:
        v3: Vectors to project, shape ``(..., 3)``.
        R: Rotation matrix aligning the viewing direction with +z.  Equivalent
            to ``jax.scipy.spatial.transform.Rotation.as_matrix()``.

    Returns:
        The rotated ``(x, y)`` components, shape ``(..., 2)``.

    """
    return (v3 @ jnp.asarray(R).T)[..., :2]


def projected_curvature_normal_and_kappa(
    dq3: Float[Array, "*batch 3"],
    d2q3: Float[Array, "*batch 3"],
    R: Float[Array, "3 3"],
    *,
    eps: float = EPS_DEFAULT,
) -> tuple[Float[Array, "*batch 2"], Float[Array, " *batch"]]:
    r"""Unit curvature normal and curvature of the projected track.

    Projection commutes with differentiation because ``R`` is linear and
    constant, so the sky-plane derivatives are the projections of the 3D ones.
    Decomposing the second derivative about the tangent gives

    .. math::

        a_\perp = q'' - \frac{q'' \cdot q'}{|q'|^2} q',
        \qquad
        \hat{n} = \frac{a_\perp}{|a_\perp|},
        \qquad
        \kappa = \frac{|a_\perp|}{|q'|^2},

    all evaluated on the projected ``(x, y)`` components.  :math:`\kappa`
    follows from :math:`\mathrm{d}s/\mathrm{d}t = |q'|`, which makes
    :math:`\mathrm{d}^2 q/\mathrm{d}s^2 = a_\perp / |q'|^2`.

    The result is invariant to reparameterisation of the track: scaling the
    parameter leaves both outputs unchanged, as does adding any multiple of
    the tangent to ``d2q3``.

    Args:
        dq3: First derivative of the 3D track, shape ``(..., 3)``.
        d2q3: Second derivative of the 3D track, shape ``(..., 3)``.
        R: Rotation matrix aligning the viewing direction with +z.
        eps: Regularisation floor for the denominators.

    Returns:
        Tuple of the unit normal in the sky plane, shape ``(..., 2)``, and the
        curvature in inverse units of the track, shape ``(...)``.

    """
    v = project_to_sky_plane(dq3, R)
    a = project_to_sky_plane(d2q3, R)

    # |q'|^2, the squared parametric speed of the projected curve.
    speed2 = jnp.maximum(jnp.sum(v * v, axis=-1), eps)

    # Component of q'' perpendicular to the tangent: points to the centre of
    # curvature and is what survives reparameterisation.
    a_perp = a - (jnp.sum(a * v, axis=-1) / speed2)[..., None] * v
    a_perp_norm = jnp.maximum(jnp.linalg.norm(a_perp, axis=-1), eps)

    return a_perp / a_perp_norm[..., None], a_perp_norm / speed2


def is_nonconcave(
    q_2d: Float[Array, "*batch 2"],
    nhat_2d: Float[Array, "*batch 2"],
) -> Float[Array, " *batch"]:
    """Flag points whose curvature normal does not point toward the centre.

    The convexity criterion of the paper: a track is concave (as every
    projection of an orbit in a spherical potential must be) where its
    curvature normal has a positive component along the direction from the
    point to the projected centre of mass at the origin.

    Args:
        q_2d: Sky-plane positions relative to the projected centre of mass,
            shape ``(..., 2)``.  May carry units.
        nhat_2d: Unit curvature normals, shape ``(..., 2)``.

    Returns:
        Boolean array, ``True`` where the point is non-concave, shape ``(...)``.

    """
    # No regularisation here: q_2d may be a Quantity, and the comparison would
    # need a matching unit.  A point exactly at the projected centre has no
    # defined radial direction and is the caller's problem.
    dir_to_center = -q_2d / jnp.linalg.norm(q_2d, axis=-1, keepdims=True)
    return jnp.sum(nhat_2d * dir_to_center, axis=-1) <= 0


def count_bad_normals(
    q_2d: Float[Array, "*batch 2"],
    nhat_2d: Float[Array, "*batch 2"],
) -> Float[Array, ""]:
    """Count normal vectors pointing away from the projected orbit centre.

    Args:
        q_2d: Sky-plane positions, shape ``(..., 2)``.
        nhat_2d: Unit curvature normals, shape ``(..., 2)``.

    Returns:
        Number of non-concave points.

    """
    return jnp.sum(is_nonconcave(q_2d, nhat_2d))
