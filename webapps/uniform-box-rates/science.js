// A box of halo shapes, weighted uniformly: the mean of qbar and of the two rates over the grid cells
// whose centres lie in it. Cells with b/a < c/a hold NaN (excluded by definition) and are skipped.
// export.py's box_mean is the same computation, and science.test.js holds the two to each other.

/** `box` = { ca: [lo, hi], ba: [lo, hi] }; `grid` = { ca, ba, qbar, lam10, lamarc } (row-major: b/a rows, c/a columns). */
export function boxMean(grid, box) {
  const { ca, ba } = grid, nc = ca.length;
  let n = 0, qbar = 0, lam10 = 0, lamarc = 0;
  for (let i = 0; i < ba.length; i++) {
    if (ba[i] < box.ba[0] || ba[i] > box.ba[1]) continue;
    for (let j = 0; j < nc; j++) {
      const k = i * nc + j, q = grid.qbar[k];
      if (ca[j] < box.ca[0] || ca[j] > box.ca[1] || !Number.isFinite(q)) continue;
      n++; qbar += q; lam10 += grid.lam10[k]; lamarc += grid.lamarc[k];
    }
  }
  return n ? { n, qbar: qbar / n, lam10: lam10 / n, lamarc: lamarc / n } : { n: 0 };
}

/** A box from two corners in data units, its bounds in increasing order. */
export const boxFrom = ([ca0, ba0], [ca1, ba1]) =>
  ({ ca: [Math.min(ca0, ca1), Math.max(ca0, ca1)], ba: [Math.min(ba0, ba1), Math.max(ba0, ba1)] });
