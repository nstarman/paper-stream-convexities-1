// Paper Figure 4's science, with no DOM (plot_mollweide.py, viz_ridgeline.py, sky_geometry.py): a
// stream seen from a map position, its ridge-line track classified as Fig. 4 does, and its particles
// binned as its sky panel does.

import { rotationToZ, mat3 } from "figpress-apps/geom.js";

const RAD = Math.PI / 180;

/** The stream-frame viewing direction for map position (l, b) in degrees: θ = 90° − b, φ = l; the
 *  particle-spray stream's map is mirrored about the equator (flip), so there θ = 90° + b. */
export function direction([l, b], flip = false) {
  const th = (flip ? 90 + b : 90 - b) * RAD, ph = l * RAD;
  return [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)];
}

/** Mollweide: (l, b) in degrees → (x, y) with x ∈ [−2√2, 2√2], y ∈ [−√2, √2] (y up). */
export function mollweide([l, b]) {
  const phi = b * RAD;
  let t = phi;
  if (Math.abs(Math.abs(phi) - Math.PI / 2) > 1e-9) {
    for (let k = 0; k < 50; k++) {
      const d = (2 * t + Math.sin(2 * t) - Math.PI * Math.sin(phi)) / (2 + 2 * Math.cos(2 * t));
      t -= d;
      if (Math.abs(d) < 1e-12) break;
    }
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

/** One stream's arrays and settings from the loaded dataset. */
export function streamFromData(data, id) {
  const s = data.meta.streams.find((x) => x.id === id), A = data.arrays;
  return { ...s, pos: A[`${id}_pos`].data, d1: A[`${id}_d1`].data, d2: A[`${id}_d2`].data,
           particles: A[`${id}_particles`].data, lut: A[`${id}_density_lut`].data };
}

/** The sky axes for a map position (rotation_matrix_from_viewing_angles' first two rows). */
function axes(stream, view) {
  const R = rotationToZ(direction(view, stream.flip));
  return [mat3.row(R, 0), mat3.row(R, 1)];
}

/** The track seen from (l, b): {xy (flat n×2), mask (non-concave), fraction} — ridgeline_sky_stats. */
export function classify(stream, view) {
  const [ex, ey] = axes(stream, view);
  const n = stream.pos.length / 3, xy = new Float32Array(2 * n), mask = new Uint8Array(n);
  const P = (A, i) => [ex[0] * A[3 * i] + ex[1] * A[3 * i + 1] + ex[2] * A[3 * i + 2],
    ey[0] * A[3 * i] + ey[1] * A[3 * i + 1] + ey[2] * A[3 * i + 2]];
  let bad = 0;
  for (let i = 0; i < n; i++) {
    const q = P(stream.pos, i), t = P(stream.d1, i), a = P(stream.d2, i);
    const k = (a[0] * t[0] + a[1] * t[1]) / (t[0] * t[0] + t[1] * t[1]);
    const nx = a[0] - k * t[0], ny = a[1] - k * t[1];
    xy[2 * i] = q[0]; xy[2 * i + 1] = q[1];
    if (-(nx * q[0] + ny * q[1]) <= 0) { mask[i] = 1; bad++; }
  }
  return { xy, mask, fraction: bad / n };
}

/** The particles on the sky, flat n×2 (project_to_sky). */
export function project(stream, view) {
  const [ex, ey] = axes(stream, view), p = stream.particles, n = p.length / 3, out = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) {
    out[2 * i] = ex[0] * p[3 * i] + ex[1] * p[3 * i + 1] + ex[2] * p[3 * i + 2];
    out[2 * i + 1] = ey[0] * p[3 * i] + ey[1] * p[3 * i + 1] + ey[2] * p[3 * i + 2];
  }
  return out;
}

/** Square limits centred on the origin around the particles and track (symmetric_limits_around_origin). */
export function autoLimits(sky, xy) {
  let half = 0.5;
  for (const a of [sky, xy]) for (let i = 0; i < a.length; i++) half = Math.max(half, Math.abs(a[i]));
  half *= 1.06;
  return [[-half, half], [-half, half]];
}

/** The panel's histogram (np.histogram2d over the limits) and its colours (hist2d with the panel's
 *  norm, cmin and colour map): {n, rgba} with rgba n×n rows from the top. */
export function density(stream, sky, [[x0, x1], [y0, y1]]) {
  const d = stream.density;
  const n = d.bin_width ? Math.max(Math.ceil(Math.max(x1 - x0, y1 - y0) / d.bin_width), 10) : d.bins;
  const counts = new Float64Array(n * n);
  for (let i = 0; i < sky.length; i += 2) {
    const x = sky[i], y = sky[i + 1];
    if (x < x0 || x > x1 || y < y0 || y > y1) continue;
    const ix = Math.min(n - 1, Math.floor(((x - x0) / (x1 - x0)) * n)), iy = Math.min(n - 1, Math.floor(((y - y0) / (y1 - y0)) * n));
    counts[iy * n + ix]++;
  }
  const positive = Array.from(counts).filter((c) => c > 0).sort((a, b) => a - b);
  let vmax = d.cmin + 1;
  if (positive.length) { // np.percentile, linear
    const r = (d.vmax_percentile / 100) * (positive.length - 1), lo = Math.floor(r);
    vmax = Math.max(vmax, positive[lo] + (r - lo) * ((positive[lo + 1] ?? positive[lo]) - positive[lo]));
  }
  const norm = d.log ? (c) => (Math.log(c) - Math.log(d.cmin)) / (Math.log(vmax) - Math.log(d.cmin))
    : d.gamma ? (c) => Math.max(0, (c - d.cmin) / (vmax - d.cmin)) ** d.gamma : (c) => (c - d.cmin) / (vmax - d.cmin);
  const rgba = new Uint8ClampedArray(4 * n * n);
  for (let iy = 0; iy < n; iy++) {
    for (let ix = 0; ix < n; ix++) {
      const c = counts[iy * n + ix];
      if (c < d.cmin) continue; // hist2d's cmin: left out
      const k = 4 * Math.min(255, Math.max(0, Math.floor(norm(c) * 256))), o = 4 * ((n - 1 - iy) * n + ix);
      rgba[o] = stream.lut[k]; rgba[o + 1] = stream.lut[k + 1]; rgba[o + 2] = stream.lut[k + 2];
      rgba[o + 3] = stream.lut[k + 3] * d.alpha;
    }
  }
  return { n, rgba, counts };
}
