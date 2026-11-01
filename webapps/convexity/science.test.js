// The app reproduces the paper: at each stream's Fig. 4 view, the non-concave fraction is the one the
// paper's Python finds (meta.paper_fraction, the sphere scan's maximum, written by export.py).
// Run after the export: uv run figpress-apps test webapps/convexity
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { loadData } from "figpress-apps/data.js";
import { streamFromData, classify } from "app/science.js";

const fileFetch = async (url) => {
  const buf = await readFile(new URL(url, `file://${process.cwd()}/`));
  return { ok: true, json: async () => JSON.parse(buf), arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
};
const NAME = "convexity";
const load = () => loadData(NAME, { fetch: fileFetch, base: `docs/.syw-site/webapps/${NAME}/data/` });

test("non-concave fractions at the paper's views match the paper", async () => {
  const data = await load();
  for (const s of data.meta.streams) {
    const got = classify(streamFromData(data, s.id), s.view).fraction;
    assert.ok(Math.abs(got - s.paper_fraction) < 1e-6, // the paper stores float32 percentages
      `${s.id}: ${got} vs ${s.paper_fraction}`);
  }
});

test("the antipodal view gives the same fraction (mirrored sky)", async () => {
  const data = await load();
  for (const s of data.meta.streams) {
    const st = streamFromData(data, s.id);
    assert.ok(Math.abs(classify(st, s.view).fraction - classify(st, s.view.map((c) => -c)).fraction) < 1e-12, s.id);
  }
});
