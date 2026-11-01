"""Orientation inset for the Mollweide viewing-angle figures.

A small schematic that explains what a point on a Mollweide viewing-angle map
means: a wireframe globe of viewing directions, the track drawn in its true 3-D
orientation inside it, an external observer, and the projection (sky) plane the
observer sees, carrying the track's shadow cast along the line of sight.
"""

__all__ = ["add_orientation_inset"]

from typing import Any

import matplotlib.patches as mpatches
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

#: U+23FF OBSERVER EYE SYMBOL.  STIX Two Math is the only commonly installed
#: font carrying that glyph; the subset is embedded in the saved PDF.
_EYE_GLYPH = "⏿"
_EYE_FONT = "STIX Two Math"


def add_orientation_inset(  # noqa: PLR0913
    fig: Figure,
    ax_map: Axes,
    xyz: np.ndarray,
    *,
    color: Any = "tab:blue",
    rect: tuple[float, float, float, float] = (-0.032, 0.77, 0.16, 0.28),
    elev_deg: float = 20.0,
    azim_deg: float = -160.5,
    eye_ang_deg: float = 24.0,
    foreshortening: float = 0.55,
    plane_side: float = 2.2,
    plane_at: float = 2.05,
    **plot_kw: Any,
) -> Axes:
    """Draw the globe/observer/projection-plane schematic beside a Mollweide map.

    Parameters
    ----------
    fig, ax_map
        Figure to draw on and the Mollweide axes to position against.  ``rect``
        is ``(dx, fy, width, fheight)``: ``dx`` and ``width`` are figure
        fractions, while ``fy`` and ``fheight`` are fractions of the map's
        height, measured from its bottom.
    xyz
        ``(N, 3)`` positions in any length unit; rescaled to fit the globe.
    color
        Colour for the track inside the globe and for its shadow.
    elev_deg, azim_deg
        Camera onto the globe.  Elevation is how far the globe tips towards the
        reader, so keep it low for the polar axis to read as upright; azimuth
        only spins the globe and does not move the observer.
    eye_ang_deg, foreshortening
        The observer, specified by how it should look rather than by an ``(l,
        b)``: the eye sits ``eye_ang_deg`` above horizontal on the page, and the
        plane it looks down on is turned ``foreshortening`` of the way to
        face-on.  These are independent of the camera.
    plane_side, plane_at
        Size of the square projection plane and its distance out along the line
        of sight, in globe radii.
    **plot_kw
        Passed to ``Axes.plot`` for the track and its shadow — pass e.g.
        ``linestyle="none", marker=".", markersize=0.6`` to draw a particle
        cloud instead of a curve.

    """
    p = ax_map.get_position()
    ax = fig.add_axes(
        (p.x0 + rect[0], p.y0 + rect[1] * p.height, rect[2], rect[3] * p.height),
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 4.3)
    ax.set_ylim(-1.5, 2.9)
    ax.axis("off")

    # Orthographic camera, shared by the wireframe and everything drawn in it,
    # so the whole schematic is one consistent projection.  The polar axis stays
    # vertical on the page because `right` is built horizontal.
    elev, azim = np.deg2rad(elev_deg), np.deg2rad(azim_deg)
    cam = np.array(
        [np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)],
    )
    right = np.cross([0.0, 0.0, 1.0], cam)
    right /= np.linalg.norm(right)
    up = np.cross(cam, right)

    def project(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Orthographic screen (x, y), in units of the globe radius."""
        return pts @ right, pts @ up

    # ── Wireframe globe ──────────────────────────────────────────────────────
    # Limb (exactly the unit circle under orthographic projection), then
    # parallels and meridians as projected 3-D circles.  No hidden-line removal:
    # the far side shows through, which is what reads as a wireframe.
    t = np.linspace(0.0, 2.0 * np.pi, 200)
    wire = {"color": "0.55", "lw": 0.7, "zorder": 1}
    ax.add_patch(
        mpatches.Circle(
            (0, 0), 1.0, edgecolor="0.55", facecolor="none", lw=1.1, zorder=1,
        ),
    )
    for phi in np.deg2rad([-60.0, -30.0, 0.0, 30.0, 60.0]):
        c = np.cos(phi)
        circle = np.column_stack(
            [c * np.cos(t), c * np.sin(t), np.full_like(t, np.sin(phi))],
        )
        ax.plot(*project(circle), **wire)
    for lam in np.deg2rad([0.0, 45.0, 90.0, 135.0]):
        circle = np.column_stack(
            [np.cos(t) * np.cos(lam), np.cos(t) * np.sin(lam), np.sin(t)],
        )
        ax.plot(*project(circle), **wire)

    # ── Track, in its true 3-D orientation, scaled to sit inside the globe ───
    pts = np.asarray(xyz, dtype=float)
    pts = 0.92 * pts / np.linalg.norm(pts, axis=-1).max()
    ax.plot(*project(pts), color=color, zorder=2, **plot_kw)

    # ── Observer ─────────────────────────────────────────────────────────────
    # Solving for the line of sight in the camera basis places the eye exactly
    # where we want it on the page without having to tip the globe to get there.
    eye_ang = np.deg2rad(eye_ang_deg)
    rho = np.sqrt(1.0 - foreshortening**2)
    nhat = (
        rho * np.cos(eye_ang) * right + rho * np.sin(eye_ang) * up
        + foreshortening * cam
    )
    # Sky basis spanning the plane normal to the line of sight (north up)
    e1 = np.cross([0.0, 0.0, 1.0], nhat)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nhat, e1)
    if max(abs(e1 @ nhat), abs(e2 @ nhat)) > 1e-12:  # noqa: PLR2004
        msg = "projection plane is not perpendicular to the line of sight"
        raise RuntimeError(msg)

    # Projection (sky) plane: a square normal to the line of sight, set out
    # beyond the globe so it does not collide with it.  Its corners are placed
    # in 3-D and projected like everything else, so the foreshortening of the
    # square comes out on its own.
    corners = np.array(
        [
            plane_at * nhat + s1 * (plane_side / 2) * e1 + s2 * (plane_side / 2) * e2
            for s1, s2 in ((1, 1), (1, -1), (-1, -1), (-1, 1))
        ],
    )
    ax.add_patch(
        mpatches.Polygon(
            np.column_stack(project(corners)),
            closed=True,
            facecolor="0.6",
            edgecolor="0.3",
            linestyle="dotted",
            lw=1.2,
            alpha=0.45,
            zorder=3,
        ),
    )

    # The track's shadow, cast along the line of sight onto that plane — what
    # the observer actually sees.
    shadow = plane_at * nhat + np.outer(pts @ e1, e1) + np.outer(pts @ e2, e2)
    ax.plot(*project(shadow), color=color, zorder=4, **plot_kw)

    # Eye, drawn as a path rather than text so it can be mirrored and rotated:
    # the glyph gazes along +x, so mirroring in x and then rotating by the
    # line-of-sight angle turns the gaze back down it towards the globe.
    los = np.array(project(nhat))
    los /= np.linalg.norm(los)
    eye_path = TextPath((0, 0), _EYE_GLYPH, size=1.0, prop=FontProperties(family=_EYE_FONT))
    bb = eye_path.get_extents()
    scale = 1.3 / bb.width  # glyph width in globe radii
    transform = (
        Affine2D()
        .translate(-bb.x0 - bb.width / 2, -bb.y0 - bb.height / 2)
        .scale(-scale, scale)
        .rotate_deg(np.degrees(np.arctan2(los[1], los[0])))
        .translate(*(3.7 * los))
    )
    ax.add_patch(
        mpatches.PathPatch(
            transform.transform_path(eye_path), color="black", lw=0, zorder=5,
        ),
    )

    return ax


def _self_check() -> None:
    """Check the drawn geometry satisfies the invariants the schematic claims.

    Plane normal on the observer, plane centred on the ray out of the globe
    centre, and eye beyond the plane on the projection of that same ray.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 6))
    ax_map = fig.add_subplot(projection="mollweide")
    s = np.linspace(0, 4 * np.pi, 300)
    helix = np.column_stack([30 * np.cos(s), 30 * np.sin(s), 4 * s])
    ax = add_orientation_inset(fig, ax_map, helix, color="tab:blue", lw=1.3)

    (poly,) = [p for p in ax.patches if isinstance(p, mpatches.Polygon)]
    quad = np.asarray(poly.get_xy())[:4]
    eye = next(p for p in ax.patches if isinstance(p, mpatches.PathPatch))

    # Reconstruct the default camera to check the screen-space claims
    elev, azim = np.deg2rad(20.0), np.deg2rad(-160.5)
    cam = np.array(
        [np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)],
    )
    right = np.cross([0.0, 0.0, 1.0], cam)
    right /= np.linalg.norm(right)
    up = np.cross(cam, right)
    nhat = 0.55 * cam + np.sqrt(1 - 0.55**2) * (
        np.cos(np.deg2rad(24.0)) * right + np.sin(np.deg2rad(24.0)) * up
    )
    los = np.array([nhat @ right, nhat @ up])
    los /= np.linalg.norm(los)

    centre = quad.mean(axis=0)
    assert np.allclose(centre / np.linalg.norm(centre), los, atol=1e-9), centre
    eye_xy = np.asarray(eye.get_path().vertices).mean(axis=0)
    assert np.linalg.norm(eye_xy) > np.linalg.norm(centre), "eye must sit beyond plane"

    # The square's projected edges stay parallel in pairs (it is a parallelogram)
    edges = np.diff(np.vstack([quad, quad[:1]]), axis=0)

    def cross2(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    assert abs(cross2(edges[0], -edges[2])) < 1e-9, edges
    assert abs(cross2(edges[1], -edges[3])) < 1e-9, edges
    print("orientation_inset self-check OK")


if __name__ == "__main__":
    _self_check()
