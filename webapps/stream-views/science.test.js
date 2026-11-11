// The app reproduces Paper Figure 4: at each stream's paper view, the non-concave fraction is the
// paper's (meta.paper_fraction, ridgeline_sky_stats' pct_bad); the map position inverts.
// Run after the export: uv run figpress-apps test webapps/stream-views
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { loadData } from "figpress-apps/data.js";
import { classify, streamFromData, project, density, mollweide, inverseMollweide } from "app/science.js";
import { routes, routeAt } from "app/autopilot.js";

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

test("the autopilot's routes follow the bright arcs of the maps", async () => {
  // The map is classify's fraction on a 6 degree grid. Every point on a route is brighter than the map's median,
  // and the route's median brighter than its 75th percentile: the waypoints still trace the bright cells.
  const data = await load();
  for (const [id, r] of Object.entries(routes())) {
    const s = streamFromData(data, id), grid = [];
    for (let b = -87; b <= 87; b += 6) for (let l = -177; l <= 177; l += 6) grid.push(classify(s, [l, b]).fraction);
    grid.sort((a, b) => a - b);
    const q = (p) => grid[Math.floor(p * (grid.length - 1))], L = r.d[r.d.length - 1];
    const vals = Array.from({ length: 120 }, (_, k) => classify(s, routeAt(r, (k / 120) * L)).fraction).sort((a, b) => a - b);
    assert.ok(vals[0] > q(0.5), `${id}: a route point at ${vals[0]} is no brighter than the map's median ${q(0.5)}`);
    assert.ok(vals[60] > q(0.75), `${id}: route median ${vals[60]} vs the map's 75th percentile ${q(0.75)}`);
  }
});

test("the autopilot's views move smoothly: no jump between frames", () => {
  for (const [id, r] of Object.entries(routes())) {
    const L = r.d[r.d.length - 1];
    let prev = routeAt(r, 0);
    for (let s = 0.5; s <= 3 * L; s += 0.5) {
      const p = routeAt(r, s), step = Math.hypot(p[0] - prev[0], p[1] - prev[1]);
      assert.ok(step <= 0.5 + 1e-9, `${id}: jump of ${step.toFixed(2)} deg at s = ${s}`);
      prev = p;
    }
  }
});
