"""Export the convexity explorer's dataset: the streams of the paper's Fig. 4 → <out>/convexity/.

    uv run --locked --with "figpress-apps==2026.10.28" \
        python webapps/convexity/export.py --out build/data

The globe frame is the N-body stream's frame, as on Fig. 4's Mollweide map; the particle-spray
stream's map is drawn mirrored about the equator there (``flip_y``), so its frame is the globe
frame with z reversed. Each stream starts at the viewing direction Fig. 4 shows.
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np

import figpress_apps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "scripts" / "streams"))

from ridgeline_data import load_data, worst_viewing_angle  # noqa: E402
from sky_geometry import interpolate_mollweide_field  # noqa: E402

from stream_convexity_paper import paths  # noqa: E402

jax.config.update("jax_enable_x64", True)

PAPER = "https://ui.adsabs.harvard.edu/abs/2026arXiv260940057S/abstract"
N_GAMMA, GAMMA_RANGE = 500, (-0.95, 0.95)  # as ridgeline_sky_stats
N_POINTS = 3000  # particles per stream shown in the sky panels
R_GLOBE, RADIAL_POWER = 90.0, 0.5  # kpc at the globe's surface; radial squash on the globe (√: meta.note,
# and the captions in docs/src/pages/index.mdx and docs/syw-site.yml)
MASK_THRESHOLD = 0.10  # Fig. 4's overlay layer: transparent at or below this fraction
GLOBE_ALPHA = 0.35  # lighter than Fig. 4's 0.82, so the globe stays see-through

FLIP_Z = np.diag([1.0, 1.0, -1.0])
# Fig. 3's test orbits, integrated in the potential the particle-spray stream was run in, so in its
# frame. id, key in triaxial_data.npz, name, colour.
ORBITS = [("orbit1", "circ", "Orbit 1", "#0072B2"), ("orbit2", "ellip", "Orbit 2", "#E69F00")]

# id, case folder, name, colour map of its Fig. 4 layer, globe frame → stream frame
STREAMS = [
    ("spray", "triaxial08507_rs14_65Gyr_init2_Mprog1e6", "1: Particle spray", "cividis", FLIP_Z),
    ("nbody", "nbody_q15", "2: N-body", "viridis", np.eye(3)),
]


def direction(theta: float, phi: float) -> np.ndarray:
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def squash(p: np.ndarray) -> np.ndarray:
    """Radii r → (r / R_GLOBE)^RADIAL_POWER, in units of the globe radius; directions unchanged."""
    r = np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-9)
    return p / r * (r / R_GLOBE) ** RADIAL_POWER


def field_on_globe(case, T: np.ndarray, colat: np.ndarray, az: np.ndarray) -> np.ndarray:
    """The case's non-concave fraction (0–1) at each globe direction, gridded as Fig. 4 does."""
    lon = (case.arrays["phi_sphere"] + np.pi) % (2 * np.pi) - np.pi
    lat = np.pi / 2 - case.arrays["theta_sphere"]
    _, _, grid = interpolate_mollweide_field(lon, lat, case.arrays["pct_neg"] / 100.0, n_lon=721, n_lat=361)
    v = np.stack([np.sin(colat) * np.cos(az), np.sin(colat) * np.sin(az), np.cos(colat)], -1) @ T.T
    i = np.rint((np.pi / 2 - np.arcsin(np.clip(v[..., 2], -1, 1))) / np.pi * 360).astype(int)
    j = np.rint((np.arctan2(v[..., 1], v[..., 0]) + np.pi) / (2 * np.pi) * 720).astype(int)
    return grid[360 - i, j]  # grid rows run from lat −π/2 up


def main(out: Path, *, orbits: bool = False) -> Path:
    rng = np.random.default_rng(0)
    colat, az = np.meshgrid(np.linspace(0, np.pi, 181), np.linspace(0, 2 * np.pi, 361), indexing="ij")
    gamma = jnp.linspace(*GAMMA_RANGE, N_GAMMA)
    arrays, descriptions, streams, layers = {}, {}, [], []
    for i, folder, name, cmap, T in STREAMS:
        case = load_data(paths.STATIC / "streams" / f"{folder}_ridgeline")
        pos = np.asarray(case.track.positions(gamma))
        pts = case.particles[rng.choice(len(case.particles), min(N_POINTS, len(case.particles)), replace=False)]
        arrays |= {
            f"{i}_pos": pos,
            f"{i}_d1": np.asarray(jax.vmap(case.track.tangent)(gamma)),
            f"{i}_d2": np.asarray(jax.vmap(case.track.curvature)(gamma)),
            f"{i}_points": pts,
            f"{i}_t": T,
            f"{i}_globe_line": squash(pos @ T).astype("<f4"),  # T is its own inverse
            f"{i}_globe_points": squash(pts[::2] @ T).astype("<f4"),
        }
        descriptions |= {
            f"{i}_pos": "ridge-line track, stream frame", f"{i}_d1": "d(track)/dγ",
            f"{i}_d2": "Frenet curvature vector of the track", f"{i}_points": "particles, stream frame",
            f"{i}_t": "matrix taking the globe frame to the stream frame",
            f"{i}_globe_line": "track in the globe frame, radially squashed, globe radius = 1",
            f"{i}_globe_points": "particles in the globe frame, radially squashed",
        }
        theta, phi = worst_viewing_angle(case)
        streams.append({"id": i, "name": name, "color": mpl.colors.to_hex(mpl.colormaps[cmap](0.5)),
                        "view": (T @ direction(theta, phi)).tolist(), "panel_rot_deg": 0.0,
                        "paper_fraction": float(case.arrays["pct_neg"].max()) / 100})
        layers.append((mpl.colormaps[cmap], field_on_globe(case, T, colat, az)))

    if orbits:  # each at its Fig. 3 view, with the fraction the paper finds there
        tri = np.load(paths.DATA / "triaxial" / "triaxial_data.npz")
        for i, key, name, color in ORBITS:
            pos = tri[f"track_q_{key}"]
            arrays |= {
                f"{i}_pos": pos, f"{i}_d1": tri[f"track_dq_{key}"], f"{i}_d2": tri[f"track_d2q_{key}"],
                f"{i}_points": np.zeros((0, 3)), f"{i}_t": FLIP_Z,
                f"{i}_globe_line": squash(pos @ FLIP_Z).astype("<f4"),
                f"{i}_globe_points": np.zeros((0, 3), "<f4"),
            }
            descriptions |= {f"{i}_pos": "orbit, potential frame", f"{i}_d1": "d(orbit)/dt",
                             f"{i}_d2": "d²(orbit)/dt²", f"{i}_t": "matrix taking the globe frame to the potential frame"}
            lon, lat = float(tri[f"lon_{key[0]}_deg"]), float(tri[f"lat_{key[0]}_deg"])
            view = direction(np.radians(90 - lat), np.radians(lon + 180))  # step_1's grid convention
            streams.append({"id": i, "name": name, "color": color, "view": (FLIP_Z @ view).tolist(),
                            "panel_rot_deg": 0.0, "paper_fraction": float(tri[f"nonconcave_mask_{key}"].mean())})

    # Fig. 4's map, on a see-through globe: each layer only where its fraction > MASK_THRESHOLD (as
    # Fig. 4 draws its overlay), the second over the first; elsewhere transparent.
    rgb, alpha = np.ones(colat.shape + (3,)), np.zeros(colat.shape + (1,))
    for cmap, f in layers:
        a = np.where(f > MASK_THRESHOLD, GLOBE_ALPHA, 0.0)[..., None]
        out_a = a + alpha * (1 - a)
        rgb = np.where(out_a > 0, (a * cmap(f)[..., :3] + alpha * (1 - a) * rgb) / np.maximum(out_a, 1e-9), rgb)
        alpha = out_a
    arrays["texture"] = np.rint(255 * np.concatenate([rgb, alpha], -1)).astype(np.uint8)
    descriptions["texture"] = ("RGBA non-concave fraction (Fig. 4's two layers, transparent below the threshold); rows colatitude 0..pi, "
                               "columns azimuth 0..2pi inclusive, in the globe frame")

    # Camera: looking at Fig. 4's two views from outside (the orbits, if any, are a turn away), z up.
    toward = np.sum([s["view"] for s in streams[: len(STREAMS)]], axis=0)
    toward /= np.linalg.norm(toward)
    right = np.cross([0.0, 0.0, 1.0], toward)
    right /= np.linalg.norm(right)
    meta = {
        "title": "Stream convexities: streams and test orbits" if orbits else "Explore a Stream Convexity",
        "paper": PAPER,
        "streams": streams,
        "camera": np.stack([right, np.cross(toward, right), toward]).ravel().tolist(),
        "cap_deg": 5.5,
        # How the globe draws the streams (squash), said under the app's instructions.
        "note": f"Inside the globe, distances from the centre are on a square-root scale: a point r from the "
                f"centre is drawn at √(r / {R_GLOBE:g} kpc) of the globe's radius. The sky panels are to scale.",
    }
    units = {f"{s['id']}_{k}": "kpc" for s in streams for k in ("pos", "points")}
    return figpress_apps.export(out, "convexity", arrays, meta=meta, units=units, descriptions=descriptions,
                         licence="MIT", overwrite=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("build/data"))
    print(main(ap.parse_args().out))
