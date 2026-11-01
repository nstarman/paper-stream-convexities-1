// Paper Figure 3's science, with no DOM: the Mollweide map of viewing angles, and an orbit seen from
// any of them (step_1_compute_orbits.py and step_2_plot_mollweide.py). A point of the sky track is
// non-concave when its projected curvature normal has no component toward the projected centre.

import { rotationToZ, mat3 } from "figpress-apps/geom.js";

const RAD = Math.PI / 180;

/** The observer's direction at map position (l, b), in degrees: (−cos b cos l, −cos b sin l, sin b). */
export function direction([l, b]) {
  return [-Math.cos(b * RAD) * Math.cos(l * RAD), -Math.cos(b * RAD) * Math.sin(l * RAD), Math.sin(b * RAD)];
}

/** Mollweide: (l, b) in degrees → (x, y) with x ∈ [−2√2, 2√2], y ∈ [−√2, √2] (y up). */
export function mollweide([l, b]) {
  const phi = b * RAD;
  let t = phi; // solve 2t + sin 2t = π sin φ
  if (Math.abs(Math.abs(phi) - Math.PI / 2) < 1e-9) t = phi;
  else for (let k = 0; k < 50; k++) {
    const d = (2 * t + Math.sin(2 * t) - Math.PI * Math.sin(phi)) / (2 + 2 * Math.cos(2 * t));
    t -= d;
    if (Math.abs(d) < 1e-12) break;
  }
  return [(2 * Math.SQRT2 / Math.PI) * l * RAD * Math.cos(t), Math.SQRT2 * Math.sin(t)];
}

/** Inverse Mollweide: (x, y) → (l, b) in degrees, or null outside the ellipse. */
export function inverseMollweide([x, y]) {
  if ((x * x) / 8 + (y * y) / 2 > 1) return null;
  const t = Math.asin(Math.max(-1, Math.min(1, y / Math.SQRT2)));
  const b = Math.asin(Math.max(-1, Math.min(1, (2 * t + Math.sin(2 * t)) / Math.PI)));
  const c = Math.cos(t);
  const l = c < 1e-12 ? 0 : (Math.PI * x) / (2 * Math.SQRT2 * c);
  return [Math.max(-180, Math.min(180, l / RAD)), b / RAD];
}

/** The map's value at (l, b): its cell on the 3° grid (rows b from −90°, columns l from −180°). */
export function mapValue(map, [l, b]) {
  const [rows, cols] = map.shape;
  const i = Math.min(rows - 1, Math.floor(((b + 90) / 180) * rows));
  const j = Math.min(cols - 1, Math.floor(((l + 180) / 360) * cols));
  return map.data[i * cols + j];
}

/** The orbit's points on the great circle of views that see its mean orbital plane edge-on, as (l, b). */
export function edgeOnCircle(lhat, n = 720) {
  const L = lhat, tmp = Math.abs(L[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const cross = (a, c) => [a[1] * c[2] - a[2] * c[1], a[2] * c[0] - a[0] * c[2], a[0] * c[1] - a[1] * c[0]];
  let u = cross(L, tmp);
  const nu = Math.hypot(...u);
  u = u.map((c) => c / nu);
  const v = cross(L, u);
  const out = [];
  for (let k = 0; k < n; k++) {
    const a = (2 * Math.PI * k) / n, p = u.map((c, i) => Math.cos(a) * c + Math.sin(a) * v[i]);
    out.push([Math.atan2(-p[1], -p[0]) / RAD, Math.asin(Math.max(-1, Math.min(1, p[2]))) / RAD]);
  }
  return out;
}

/** One orbit's arrays and settings from the loaded dataset. */
export function orbitFromData(data, id) {
  const o = data.meta.orbits.find((x) => x.id === id), A = data.arrays;
  return { ...o, pos: A[`${id}_pos`].data, d1: A[`${id}_d1`].data, d2: A[`${id}_d2`].data,
           map: A[`${id}_map`], lut: A[`${id}_lut`].data };
}

/** The orbit seen from (l, b): {xy (flat n×2, the panel's axes), mask, near, fraction, readout}. */
export function classify(orbit, view, rNear = Infinity) {
  const R = rotationToZ(direction(view));
  const a = orbit.panel_rot_deg * RAD, c = Math.cos(a), s = Math.sin(a);
  const r0 = mat3.row(R, 0), r1 = mat3.row(R, 1);
  const ex = r0.map((x, i) => c * x - s * r1[i]), ey = r0.map((x, i) => s * x + c * r1[i]);
  const n = orbit.pos.length / 3, xy = new Float32Array(2 * n), mask = new Uint8Array(n), near = new Uint8Array(n);
  const P = (A, i) => [ex[0] * A[3 * i] + ex[1] * A[3 * i + 1] + ex[2] * A[3 * i + 2],
    ey[0] * A[3 * i] + ey[1] * A[3 * i + 1] + ey[2] * A[3 * i + 2]];
  let bad = 0;
  for (let i = 0; i < n; i++) {
    const q = P(orbit.pos, i), t = P(orbit.d1, i), acc = P(orbit.d2, i);
    const k = (acc[0] * t[0] + acc[1] * t[1]) / (t[0] * t[0] + t[1] * t[1]);
    const nx = acc[0] - k * t[0], ny = acc[1] - k * t[1]; // the curvature normal, unnormalised
    xy[2 * i] = q[0]; xy[2 * i + 1] = q[1];
    if (-(nx * q[0] + ny * q[1]) <= 0) { mask[i] = 1; bad++; }
    near[i] = Math.hypot(orbit.pos[3 * i], orbit.pos[3 * i + 1], orbit.pos[3 * i + 2]) < rNear ? 1 : 0;
  }
  const fraction = bad / n;
  return { xy, mask, near, fraction,
           readout: `non-concave ${(100 * fraction).toFixed(1)}%` };
}
