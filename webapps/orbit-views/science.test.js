// The app reproduces Paper Figure 3: at each orbit's paper view, the non-concave fraction is the
// paper's (meta.paper_fraction, from step_1's mask) and the map's value there; the map projection
// inverts. Run after the export: uv run figpress-apps test webapps/orbit-views
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { loadData } from "figpress-apps/data.js";
import { classify, orbitFromData, mapValue, mollweide, inverseMollweide, direction, edgeOnCircle } from "app/science.js";
import { routes, routeAt } from "app/autopilot.js";

const fileFetch = async (url) => {
  const buf = await readFile(new URL(url, `file://${process.cwd()}/`));
  return { ok: true, json: async () => JSON.parse(buf), arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
};
const NAME = "orbit-views";
const load = () => loadData(NAME, { fetch: fileFetch, base: `docs/.syw-site/webapps/${NAME}/data/` });

test("at the paper's views, the fraction is the paper's and the map's", async () => {
  const data = await load();
  for (const o of data.meta.orbits) {
    const orbit = orbitFromData(data, o.id), got = classify(orbit, o.view).fraction;
    assert.ok(Math.abs(got - o.paper_fraction) < 1e-9, `${o.id}: ${got} vs paper ${o.paper_fraction}`);
    assert.ok(Math.abs(got - mapValue(orbit.map, o.view)) < 1e-6, `${o.id}: ${got} vs map ${mapValue(orbit.map, o.view)}`);
  }
});

test("Mollweide inverts, and stays inside its ellipse", () => {
  for (const lb of [[0, 0], [82.5, 40.5], [-150, -75], [179, 10], [-30, 89]]) {
    const back = inverseMollweide(mollweide(lb));
    assert.ok(Math.hypot(back[0] - lb[0], back[1] - lb[1]) < 1e-6, `${lb} → ${back}`);
  }
  assert.equal(inverseMollweide([2.9, 0]), null);
  assert.deepEqual(mollweide([0, 90]).map((v) => +v.toFixed(9)), [0, +Math.SQRT2.toFixed(9)]);
});

test("the edge-on circle's views see the orbital plane edge-on", async () => {
  const data = await load();
  for (const o of data.meta.orbits) {
    for (const lb of edgeOnCircle(o.lhat, 36)) {
      const d = direction(lb);
      assert.ok(Math.abs(d[0] * o.lhat[0] + d[1] * o.lhat[1] + d[2] * o.lhat[2]) < 1e-9, `${o.id} ${lb}`);
    }
  }
});

test("the autopilot's routes follow the bright bands of the maps", async () => {
  // Every point on a route is brighter than its map's median, and the route's median brighter than the map's
  // 75th percentile: the waypoints still trace the bright cells if the maps change.
  const data = await load();
  for (const [id, r] of Object.entries(routes())) {
    const map = orbitFromData(data, id).map, all = Array.from(map.data).sort((a, b) => a - b);
    const q = (p) => all[Math.floor(p * (all.length - 1))], L = r.d[r.d.length - 1];
    const vals = Array.from({ length: 400 }, (_, k) => mapValue(map, routeAt(r, (k / 400) * L))).sort((a, b) => a - b);
    assert.ok(vals[0] > q(0.5), `${id}: a route point at ${vals[0]} is no brighter than the map's median ${q(0.5)}`);
    assert.ok(vals[200] > q(0.75), `${id}: route median ${vals[200]} vs the map's 75th percentile ${q(0.75)}`);
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
