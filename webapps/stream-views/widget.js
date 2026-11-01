// Paper Figure 4, interactive: the paper's own figure (plot_mollweide.py, without what moves), with
// its two map markers and each stream's sky panel drawn over it, in the paper's units and styles.
// Drag a marker and its stream's panel shows the stream from there: its particles binned as the
// paper bins them, and its track, red where it is convex. An anywidget module, so the same file runs
// on the web page (figpress-apps' `run`) and in Jupyter/marimo.

import { loadData } from "figpress-apps/data.js";
import { createStage, dragHelper } from "figpress-apps/stage.js";
import { classify, streamFromData, project, density, autoLimits, mollweide, inverseMollweide } from "app/science.js";

const X_MAX = 2 * Math.SQRT2, Y_MAX = Math.SQRT2;
const FONT = '"DejaVu Sans", Verdana, "Bitstream Vera Sans", sans-serif'; // matplotlib's default face
const MINUS = "−";

/** matplotlib's MaxNLocator ("auto": at most 9 bins): the smallest of 1, 2, 2.5, 5, 10 × 10^k. */
function ticks(lo, hi, bins) {
  const raw = (hi - lo) / Math.max(1, Math.min(9, bins)), p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw * (1 - 1e-9));
  const out = [];
  for (let t = Math.ceil(lo / step - 1e-9) * step; t <= hi + 1e-9; t += step) out.push(+t.toFixed(10));
  return out;
}
const num = (t) => (t < 0 ? MINUS : "") + String(Math.abs(t));
const round = (v) => { const r = Math.round(v); return Math.abs(v % 1) === 0.5 && r % 2 ? r - 1 : r; }; // Python's .0f

/** A text box as matplotlib's bbox=dict(boxstyle="round,pad=0.2", ...): one line of [text, italic?] runs. */
function box(ctx, runs, x, y, { size, ha, va, k }) {
  const px = size * k, pad = 0.2 * px, h = 1.2 * px;
  const font = (it) => `${it ? "italic " : ""}${px}px ${FONT}`;
  const w = runs.reduce((s, [t, it]) => { ctx.font = font(it); return s + ctx.measureText(t).width; }, 0);
  const x0 = ha === "left" ? x : x - w, y0 = va === "top" ? y : y - h;
  ctx.save();
  ctx.globalAlpha = 0.8;
  ctx.beginPath(); ctx.roundRect(x0 - pad, y0 - pad, w + 2 * pad, h + 2 * pad, pad);
  ctx.fillStyle = "#fff"; ctx.fill(); ctx.lineWidth = Math.max(1, 0.8 * k); ctx.strokeStyle = "gray"; ctx.stroke();
  ctx.restore();
  ctx.fillStyle = "#000"; ctx.textBaseline = "middle"; ctx.textAlign = "left";
  let cx = x0;
  for (const [t, it] of runs) { ctx.font = font(it); ctx.fillText(t, cx, y0 + h / 2); cx += ctx.measureText(t).width; }
}

const bitmap = (bytes) => createImageBitmap(new Blob([bytes], { type: "image/png" }));

function mountFigure(el, model, { streams, layout, images }) {
  const [Wpt, Hpt] = layout.size_pt, F = layout.fonts;
  el.style.aspectRatio = `${Wpt} / ${Hpt}`;
  let img = null;
  const cache = {};
  const stage = createStage(el, { draw });
  Promise.all([bitmap(images.layer), bitmap(images.overlay), ...streams.map((s) => bitmap(images.markers[s.id]))])
    .then(([layer, overlay, ...markers]) => { img = { layer, overlay, markers }; stage.requestDraw(); });

  const geo = () => {
    const { width: W, height: H } = stage, m = layout.map;
    return { W, H, k: W / Wpt, cx: m.cx * W, cy: m.cy * H, rx: m.rx * W, ry: m.ry * H };
  };
  const toPx = (g, lb) => { const [x, y] = mollweide(lb); return [g.cx + (x / X_MAX) * g.rx, g.cy - (y / Y_MAX) * g.ry]; };
  const fromPx = (g, px, py) => inverseMollweide([((px - g.cx) / g.rx) * X_MAX, (-(py - g.cy) / g.ry) * Y_MAX]);

  /** What a view shows, computed once per view: the track, the particles' limits and density image. */
  function compute(s, view) {
    const c = cache[s.id];
    if (c && c.view === view) return c;
    const track = classify(s, view), sky = project(s, view);
    const paper = view[0] === s.view[0] && view[1] === s.view[1];
    const lims = paper ? s.lims : autoLimits(sky, track.xy); // the paper's limits only at its view
    const { n, rgba } = density(s, sky, lims);
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = n;
    canvas.getContext("2d").putImageData(new ImageData(rgba, n, n), 0, 0);
    return (cache[s.id] = { view, track, lims, canvas });
  }

  function panel(ctx, g, s, view) {
    const [fx, fy, fw, fh] = s.panel, L = fx * g.W, T = fy * g.H, Wp = fw * g.W, Hp = fh * g.H, k = g.k;
    const { track, lims: [[x0, x1], [y0, y1]], canvas } = compute(s, view);
    const X = (x) => L + ((x - x0) / (x1 - x0)) * Wp, Y = (y) => T + ((y1 - y) / (y1 - y0)) * Hp;
    ctx.save();
    ctx.beginPath(); ctx.rect(L, T, Wp, Hp); ctx.clip();
    ctx.imageSmoothingEnabled = false; // hist2d's cells, crisp
    ctx.drawImage(canvas, L, T, Wp, Hp);
    ctx.restore();
    const xt = ticks(x0, x1, Math.floor(fw * Wpt / (3 * F.tick))), yt = ticks(y0, y1, Math.floor(fh * Hpt / (2 * F.tick)));
    ctx.strokeStyle = "rgba(176,176,176,0.3)"; ctx.lineWidth = 0.8 * k; // ax.grid(True, alpha=0.3)
    ctx.beginPath();
    for (const t of xt) { ctx.moveTo(X(t), T); ctx.lineTo(X(t), T + Hp); }
    for (const t of yt) { ctx.moveTo(L, Y(t)); ctx.lineTo(L + Wp, Y(t)); }
    ctx.stroke();
    // Ticks outward, labels 15 pt: the top panel's x along the top, the bottom's along the bottom.
    const top = s.id === "spray", tl = 3.5 * k, pad = 3.5 * k;
    ctx.strokeStyle = "#000"; ctx.lineWidth = 0.8 * k; ctx.fillStyle = "#000"; ctx.font = `${F.tick * k}px ${FONT}`;
    ctx.beginPath();
    ctx.textAlign = "center"; ctx.textBaseline = top ? "bottom" : "top";
    for (const t of xt) {
      const yy = top ? T : T + Hp;
      ctx.moveTo(X(t), yy); ctx.lineTo(X(t), top ? yy - tl : yy + tl);
      ctx.fillText(num(t), X(t), top ? yy - tl - pad : yy + tl + pad);
    }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const t of yt) { ctx.moveTo(L, Y(t)); ctx.lineTo(L - tl, Y(t)); ctx.fillText(num(t), L - tl - pad, Y(t)); }
    ctx.stroke();

    ctx.save();
    ctx.beginPath(); ctx.rect(L, T, Wp, Hp); ctx.clip();
    const { xy, mask } = track, n = mask.length, st = s.track;
    const segs = (bad) => { // the good track, then the convex parts over it (zorder 9, 11)
      ctx.beginPath();
      for (let i = 1; i < n; i++) if (!!mask[i] === bad && !!mask[i - 1] === bad) { ctx.moveTo(X(xy[2 * i - 2]), Y(xy[2 * i - 1])); ctx.lineTo(X(xy[2 * i]), Y(xy[2 * i + 1])); }
      ctx.strokeStyle = bad ? st.bad : st.good; ctx.lineWidth = (bad ? st.lw_bad : st.lw_good) * k;
      ctx.globalAlpha = bad ? st.alpha_bad : st.alpha_good; ctx.stroke(); ctx.globalAlpha = 1;
    };
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    segs(false); segs(true);
    ctx.restore();
    const r = 4.5 * k; // the galactic centre: "x", markersize 9, edge 2, not clipped
    ctx.strokeStyle = "#000"; ctx.lineWidth = 2 * k; ctx.lineCap = "butt"; ctx.beginPath();
    ctx.moveTo(X(0) - r, Y(0) - r); ctx.lineTo(X(0) + r, Y(0) + r); ctx.moveTo(X(0) - r, Y(0) + r); ctx.lineTo(X(0) + r, Y(0) - r); ctx.stroke();

    const runs = [["(", false], ["l", true], [",", false], ["b", true], [`) = (${round(view[0])}°, ${round(view[1])}°)`, false]];
    const right = s.info.endsWith("right");
    box(ctx, runs, L + (right ? 0.95 : 0.05) * Wp, T + 0.95 * Hp, { size: F.info, ha: right ? "right" : "left", va: "bottom", k });
  }

  function draw({ ctx, width, height }) {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, width, height); // the paper's figure is on white
    if (!img) return;
    ctx.drawImage(img.layer, 0, 0, width, height);
    const g = geo(), views = model.get("views");
    for (const s of streams) panel(ctx, g, s, views[s.id]);
    ctx.drawImage(img.overlay, 0, 0, width, height); // the panels' title boxes, over the panels
    streams.forEach((s, i) => { // each marker with its label, as the paper draws them, at its view
      const [px, py] = toPx(g, views[s.id]), { box: [bx, by, bw, bh], at: [ax, ay] } = s.marker;
      ctx.drawImage(img.markers[i], px - (ax - bx) * g.W, py - (ay - by) * g.H, bw * g.W, bh * g.H);
    });
  }

  const nearest = (x, y) => {
    const g = geo(), views = model.get("views");
    let best = null, dmin = Math.max(12, 10 * g.k);
    for (const s of streams) { const [px, py] = toPx(g, views[s.id]), d = Math.hypot(px - x, py - y); if (d < dmin) { dmin = d; best = s.id; } }
    return best;
  };
  const undrag = dragHelper(stage.canvas, {
    hitTest: nearest,
    onMove: (id, _dx, _dy, x, y) => {
      if (!id) return;
      const lb = fromPx(geo(), x, y);
      if (lb) { model.set("views", { ...model.get("views"), [id]: lb }); model.save_changes(); }
    },
  });
  const hover = (e) => { const b = stage.canvas.getBoundingClientRect(); stage.canvas.style.cursor = nearest(e.clientX - b.left, e.clientY - b.top) ? "grab" : ""; };
  stage.canvas.addEventListener("pointermove", hover);
  model.on("change:views", stage.requestDraw);
  return () => { undrag(); stage.canvas.removeEventListener("pointermove", hover); model.off("change:views", stage.requestDraw); stage.dispose(); };
}

export default {
  async render({ model, el, signal }) {
    const source = model.get("manifest") ? model : (model.get("data_source") ?? "stream-views");
    const data = await loadData(source, { base: model.get("data_base") ?? "data/" });
    const { meta, arrays } = data;
    const streams = meta.streams.map((s) => streamFromData(data, s.id));
    const paperViews = () => Object.fromEntries(meta.streams.map((s) => [s.id, s.view.slice()]));
    if (!Object.keys(model.get("views") ?? {}).length) model.set("views", paperViews());
    model.save_changes();

    el.classList.add("fk-app");
    el.dataset.figpressApp = "";
    const toolbar = document.createElement("div");
    toolbar.className = "fk-toolbar";
    const reset = document.createElement("button");
    reset.textContent = "Reset views";
    reset.onclick = () => { model.set("views", paperViews()); model.save_changes(); };
    const readout = document.createElement("span");
    readout.className = "fk-readout";
    const names = { spray: "1: Particle spray", nbody: "2: N-body" };
    const update = () => {
      const v = model.get("views");
      readout.textContent = streams.map((s) => `${names[s.id]}: ${(100 * classify(s, v[s.id]).fraction).toFixed(1)}% convex`).join(" · ");
    };
    model.on("change:views", update);
    update();
    toolbar.append(reset, "Drag a marker on the map.", readout);
    const scroller = document.createElement("div");
    scroller.className = "ov-scroll";
    const fig = document.createElement("div");
    fig.className = "ov-figure";
    scroller.append(fig);
    el.replaceChildren(toolbar, scroller);
    const images = { layer: arrays.paper_layer.data, overlay: arrays.overlay.data,
                     markers: Object.fromEntries(streams.map((s) => [s.id, arrays[`${s.id}_marker`].data])) };
    const dispose = mountFigure(fig, model, { streams, layout: meta.layout, images });
    signal?.addEventListener("abort", () => { dispose(); model.off("change:views", update); });
  },
};
