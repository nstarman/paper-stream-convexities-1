// The app reproduces Paper Figure 4: at each stream's paper view, the non-concave fraction is the
// paper's (meta.paper_fraction, ridgeline_sky_stats' pct_bad); the map position inverts.
// Run after the export: uv run figpress-apps test webapps/stream-views
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { loadData } from "figpress-apps/data.js";
import { classify, streamFromData, project, density, mollweide, inverseMollweide } from "app/science.js";

const fileFetch = async (url) => {
  const buf = await readFile(new URL(url, `file://${process.cwd()}/`));
  return { ok: true, json: async () => JSON.parse(buf), arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
};
const NAME = "stream-views";
const load = () => loadData(NAME, { fetch: fileFetch, base: `docs/.syw-site/webapps/${NAME}/data/` });

test("at the paper's views, the fraction is the paper's", async () => {
  const data = await load();
  for (const s of data.meta.streams) {
    const got = classify(streamFromData(data, s.id), s.view).fraction;
    assert.ok(Math.abs(got - s.paper_fraction) < 1e-9, `${s.id}: ${got} vs paper ${s.paper_fraction}`);
  }
});

test("the panel's histogram keeps every particle inside its limits", async () => {
  const data = await load();
  for (const s of data.meta.streams) {
    const st = streamFromData(data, s.id), sky = project(st, s.view), { n, counts } = density(st, sky, s.lims);
    const [[x0, x1], [y0, y1]] = s.lims;
    let inside = 0;
    for (let i = 0; i < sky.length; i += 2) if (sky[i] >= x0 && sky[i] <= x1 && sky[i + 1] >= y0 && sky[i + 1] <= y1) inside++;
    assert.equal(counts.reduce((a, b) => a + b, 0), inside, s.id);
    assert.equal(n, Math.max(Math.ceil(Math.max(x1 - x0, y1 - y0) / s.density.bin_width), 10), s.id);
  }
});

test("Mollweide inverts", () => {
  for (const lb of [[0, 0], [11.97, 58.11], [138.5, 68.5], [-150, -75]]) {
    const back = inverseMollweide(mollweide(lb));
    assert.ok(Math.hypot(back[0] - lb[0], back[1] - lb[1]) < 1e-6, `${lb} → ${back}`);
  }
});
