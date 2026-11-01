// The convexity explorer: the globe of viewing directions and one sky panel per stream (figpress-apps' globe
// example, on this paper's data). An anywidget module, so the same file runs on the web page (through
// figpress-apps' `run`) and in Jupyter/marimo (through figpress_apps.anywidget.Widget).

import { loadData } from "figpress-apps/data.js";
import { mount as mountGlobe } from "figpress-apps/globe.js";
import { mount as mountPanel } from "figpress-apps/skypanel.js";
import { streamFromData, classify, markerAxes } from "app/science.js";

export default {
  async render({ model, el, signal }) {
    const source = model.get("manifest") ? model : (model.get("data_source") ?? "convexity");
    const data = await loadData(source, { base: model.get("data_base") ?? "data/" });
    const { meta, arrays } = data;
    const streams = Object.fromEntries(meta.streams.map((s) => [s.id, streamFromData(data, s.id)]));
    const paperViews = () => Object.fromEntries(meta.streams.map((s) => [s.id, s.view.slice()]));
    const colors = Object.fromEntries(meta.streams.map((s) => [s.id, s.color]));
    if (!(model.get("camera")?.length === 9)) model.set("camera", meta.camera.slice());
    if (!Object.keys(model.get("views") ?? {}).length) model.set("views", paperViews());
    model.save_changes();

    el.classList.add("fk-app");
    el.dataset.figpressApp = "";
    const toolbar = document.createElement("div");
    toolbar.className = "fk-toolbar";
    const button = (label, fn) => {
      const b = document.createElement("button");
      b.textContent = label;
      b.onclick = () => { fn(); model.save_changes(); };
      return b;
    };
    // Embedded in the paper's site, the caption under the frame says how to use it; on its own page,
    // in the single file or in a notebook, the app says it here.
    const embedded = globalThis.self !== globalThis.top;
    toolbar.append(
      button("Reset camera", () => model.set("camera", meta.camera.slice())),
      button("Reset views", () => model.set("views", paperViews())),
      ...(embedded ? [] : ["Drag the globe to turn it; drag a square to see that stream from there. Red: where the stream curves away from the galaxy's centre (convex)."]),
    );
    const note = document.createElement("p");
    note.className = "fk-note";
    note.textContent = meta.note ?? "";
    const grid = document.createElement("div");
    grid.className = "fk-split ov-split"; // globe beside the panels; stacked when the app is narrow (index.html)
    const globeEl = document.createElement("div");
    const panels = document.createElement("div");
    panels.className = "fk-grid-2";
    grid.append(globeEl, panels);
    el.replaceChildren(toolbar, ...(meta.note && !embedded ? [note] : []), grid);

    const disposers = [mountGlobe(globeEl, model, {
      texture: arrays.texture,
      lines: Object.fromEntries(meta.streams.map((s) => [s.id, arrays[`${s.id}_globe_line`]])),
      points: Object.fromEntries(meta.streams.map((s) => [s.id, arrays[`${s.id}_globe_points`]])),
      colors,
      markerAxes: (id, view) => markerAxes(streams[id], view),
      markerSizeDeg: meta.cap_deg,
      mollweide: true, // the whole sky, as Paper Figure 4 draws it (same frame), in the globe's corner
    })];
    for (const s of meta.streams) {
      const panel = document.createElement("div");
      panels.appendChild(panel);
      disposers.push(mountPanel(panel, model, {
        id: s.id, title: s.name, color: s.color, compute: (view) => classify(streams[s.id], view),
      }));
    }
    signal?.addEventListener("abort", () => disposers.forEach((d) => d()));
  },
};
