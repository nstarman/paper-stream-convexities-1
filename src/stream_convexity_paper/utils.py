"""Utility functions for curvature analysis."""

from jaxtyping import ArrayLike, Real

import coordinax.vecs as cxv
import matplotlib.pyplot as plt
import numpy as np
from plum import convert, dispatch

import unxt as u


@dispatch
def plot_vec(vec: Real[ArrayLike, "n F"], *, ax: plt.Axes, **kwargs: object):
    ax.plot(*vec.T, **kwargs)


@dispatch
def plot_vec(vec: Real[u.Quantity, "n F"], *, ax: plt.Axes, **kwargs: object):
    plot_vec(np.array(vec.value), ax=ax, **kwargs)


@dispatch
def plot_vec(vec: cxv.AbstractVector, *, ax: plt.Axes, **kwargs: object):
    q = convert(vec, u.Quantity)
    plot_vec(q, ax=ax, **kwargs)


@dispatch
def scatter_vec(vec: Real[ArrayLike, "n F"], *, ax: plt.Axes, **kwargs: object):
    return ax.scatter(*vec.T, **kwargs)


@dispatch
def scatter_vec(vec: Real[u.Quantity, "n F"], *, ax: plt.Axes, **kwargs: object):
    return scatter_vec(np.array(vec.value), ax=ax, **kwargs)


@dispatch
def scatter_vec(vec: cxv.AbstractVector, *, ax: plt.Axes, **kwargs: object):
    q = convert(vec, u.Quantity)
    return scatter_vec(q, ax=ax, **kwargs)


@dispatch
def quiver_vec(
    from_: Real[ArrayLike | u.Quantity, " F"] | Real[ArrayLike | u.Quantity, "n F"],
    to_: Real[ArrayLike | u.Quantity, " F"] | Real[ArrayLike | u.Quantity, "n F"],
    /,
    *,
    ax: plt.Axes,
    **kwargs: object,
):
    if "c" in kwargs:
        kwargs["color"] = kwargs.pop("c")
    ax.quiver(*np.array(from_).T, *np.array(to_ - from_).T, **kwargs)
