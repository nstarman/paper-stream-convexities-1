// A box's uniform mean is export.py's (meta.check, the CDM box), and the b/a < c/a wedge adds
// nothing to a box. Run after the export: uv run figpress-apps test webapps/uniform-box-rates
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { loadData } from "figpress-apps/data.js";
import { boxMean, boxFrom } from "app/science.js";

const fileFetch = async (url) => {
  const buf = await readFile(new URL(url, `file://${process.cwd()}/`));
  return { ok: true, json: async () => JSON.parse(buf), arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
};
const NAME = "uniform-box-rates";
const load = async () => {
  const { meta, arrays } = await loadData(NAME, { fetch: fileFetch, base: `docs/.syw-site/webapps/${NAME}/data/` });
  return { meta, grid: Object.fromEntries(["ca", "ba", "qbar", "lam10", "lamarc"].map((k) => [k, arrays[k].data])) };
};

test("the CDM box, uniform, is export.py's, and not the paper's Auriga-weighted CDM", async () => {
  const { meta, grid } = await load();
  const got = boxMean(grid, meta.populations.cdm.box), want = meta.check.cdm_box_uniform;
  assert.equal(got.n, want.n);
  for (const k of ["qbar", "lam10", "lamarc"]) assert.ok(Math.abs(got[k] - want[k]) < 1e-12, k);
  assert.ok(Math.abs(got.lam10 - meta.populations.cdm.lam10) > 1e-3); // the weighting shows
});

test("the b/a < c/a wedge adds nothing", async () => {
  const { grid } = await load();
  assert.equal(boxMean(grid, boxFrom([0.9, 0.85], [1.0, 0.88])).n, 0); // c/a above b/a throughout
  // A box across the diagonal holds only its cells with b/a >= c/a.
  let allowed = 0;
  for (const b of grid.ba) for (const c of grid.ca) if (b >= 0.85 && c >= 0.85 && b >= c) allowed++;
  const straddle = boxMean(grid, boxFrom([1.0, 1.0], [0.85, 0.85]));
  assert.equal(straddle.n, allowed);
  assert.ok(allowed > 0);
});
