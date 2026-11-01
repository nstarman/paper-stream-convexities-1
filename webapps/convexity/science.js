// The paper's science step, with no DOM (figpress-apps' globe example, which ports this paper): project a
// stream onto the sky for a viewing direction and classify each track point. A point is non-concave
// when its projected curvature normal has no component toward the projected centre (the paper's
// criterion, stream_convexity_paper.curvature.is_nonconcave).

import { unit, mat3, rotationToZ } from "figpress-apps/geom.js";

/** One stream's arrays and frame from the loaded dataset. */
export function streamFromData(data, id) {
  const s = data.meta.streams.find((x) => x.id === id);
  const A = data.arrays;
  return {
    id, panelRotDeg: s.panel_rot_deg, T: Array.from(A[`${id}_t`].data), // globe frame → simulation frame
    pos: A[`${id}_pos`].data, d1: A[`${id}_d1`].data, d2: A[`${id}_d2`].data, points: A[`${id}_points`].data,
  };
}

/** Sky axes (simulation frame) of the panel for a globe-frame view, with the panel's rotation. */
function skyAxes(st, view) {
  const R = rotationToZ(unit(mat3.apply(st.T, view)));
  const a = st.panelRotDeg * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
  const r0 = mat3.row(R, 0), r1 = mat3.row(R, 1);
  return { ex: r0.map((x, i) => c * x - s * r1[i]), ey: r0.map((x, i) => s * x + c * r1[i]) };
}

/** Tangent axes of the stream's marker square, in the globe frame. */
export function markerAxes(st, view) {
  const { ex, ey } = skyAxes(st, view);
  return [mat3.applyT(st.T, ex), mat3.applyT(st.T, ey)];
}

/** Project and classify: {xy, points, mask, fraction, readout}; xy and points are flat n×2. */
export function classify(st, view) {
  const { ex, ey } = skyAxes(st, view);
  const n = st.pos.length / 3, xy = new Float32Array(2 * n), mask = new Uint8Array(n);
  const P = (a, i) => [ex[0] * a[3 * i] + ex[1] * a[3 * i + 1] + ex[2] * a[3 * i + 2],
    ey[0] * a[3 * i] + ey[1] * a[3 * i + 1] + ey[2] * a[3 * i + 2]];
  let bad = 0;
  for (let i = 0; i < n; i++) {
    const q = P(st.pos, i), t = P(st.d1, i), a = P(st.d2, i);
    const k = (a[0] * t[0] + a[1] * t[1]) / (t[0] * t[0] + t[1] * t[1]);
    const nx = a[0] - k * t[0], ny = a[1] - k * t[1];
    xy[2 * i] = q[0]; xy[2 * i + 1] = q[1];
    if (-(nx * q[0] + ny * q[1]) <= 0) { mask[i] = 1; bad++; }
  }
  const m = st.points.length / 3, points = new Float32Array(2 * m);
  for (let i = 0; i < m; i++) [points[2 * i], points[2 * i + 1]] = P(st.points, i);
  const fraction = bad / n;
  return { xy, points, mask, fraction, readout: `non-concave ${(100 * fraction).toFixed(1)}%` };
}
