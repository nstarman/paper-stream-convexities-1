"""A parametric track in 3D.

``potamides`` models a stream ridge line in 2D: its ``Track.positions`` returns
``(N, 2)`` and its curvature is the scalar ``kappa``.  The convexity argument in
this paper needs the full 3D curvature vector before projection, so
:class:`Track3D` overrides those three methods to return 3D quantities and
inherits everything else from ``potamides.AbstractTrack``.

The spline must be twice differentiable, hence ``method="cubic2"``: the whole
analysis rests on the second derivative, and a curve whose curvature jumps at
the knots would put those jumps into the figures.
"""

__all__ = ("Track3D",)

import functools as ft
from dataclasses import dataclass

from jaxtyping import Array, Real
from typing import final

import interpax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from potamides import AbstractTrack, splinelib

type Sz0 = Real[Array, ""]
type Sz3 = Real[Array, "3"]
type SzN = Real[Array, "N"]
type SzN3 = Real[Array, "N 3"]


@final
@ft.partial(jtu.register_dataclass, data_fields=["ridge_line"], meta_fields=[])
@dataclass(frozen=True, slots=True, eq=False)
class Track3D(AbstractTrack):
    r"""A twice-differentiable parametric curve in 3D.

    Examples
    --------
    Build a track from an interpax spline through 3D knots:

    >>> import jax.numpy as jnp
    >>> import interpax
    >>> from stream_convexity_paper.track import Track3D

    >>> gamma = jnp.linspace(0, 1, 50)
    >>> knots = jnp.stack(
    ...     [gamma, gamma**2, jnp.sin(3 * jnp.pi * gamma)], axis=-1
    ... )
    >>> spline = interpax.Interpolator1D(gamma, knots, method="cubic2")
    >>> track = Track3D(ridge_line=spline)

    Positions, tangents and curvature vectors are 3D; kappa stays scalar:

    >>> gamma_test = jnp.array([0.0, 0.5, 1.0])
    >>> track.positions(gamma_test).shape
    (3, 3)
    >>> track.tangent(gamma_test).shape
    (3, 3)
    >>> track.curvature(gamma_test).shape
    (3, 3)
    >>> track.kappa(gamma_test).shape
    (3,)

    Inherited track properties still work:

    >>> len(track.knots)
    50

    """

    #: [x, y, z](gamma) spline.  Must be twice differentiable (``cubic2``) for
    #: the curvature vector to exist.
    ridge_line: interpax.Interpolator1D

    def positions(self, gamma: SzN) -> SzN3:
        """Return the 3D position at each gamma."""
        return self.ridge_line(gamma)

    @ft.partial(jnp.vectorize, signature="()->(3)", excluded=(0,))
    @ft.partial(jax.jit)
    def tangent(self, gamma: Sz0, /) -> Sz3:
        r"""Return $\mathrm{d}q/\mathrm{d}\gamma$, the 3D tangent vector."""
        return splinelib.tangent(self.ridge_line, gamma)

    @ft.partial(jnp.vectorize, signature="()->(3)", excluded=(0,))
    @ft.partial(jax.jit)
    def curvature(self, gamma: Sz0, /) -> Sz3:
        r"""Return the 3D Frenet curvature vector $\kappa\hat{N}$.

        This is $\mathrm{d}\hat{T}/\mathrm{d}s$, NOT $\mathrm{d}^2q/\mathrm{d}\gamma^2$.
        The two are parallel only when gamma is arc length, and differ by
        $(\mathrm{d}s/\mathrm{d}\gamma)^2$ in magnitude otherwise, a factor
        of order 50 for the tracks in this paper. The distinction does not
        reach the convexity test, which normalises the projected normal, but
        it does reach any curvature MAGNITUDE computed from this vector.
        """
        return splinelib.curvature(self.ridge_line, gamma)
