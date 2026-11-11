// Paper Figure 3, interactive: the paper's own figure (step_2_plot_mollweide.py, without what moves),
// with the View dots on the map and each orbit's sky panel drawn over it, in the paper's units and
// styles. Drag a dot and its orbit's panel shows the orbit from there. An anywidget module, so the
// same file runs on the web page (figpress-apps' `run`) and in Jupyter/marimo.

import { loadData } from "figpress-apps/data.js";
import { createStage, dragHelper } from "figpress-apps/stage.js";
import { classify, orbitFromData, mollweide, inverseMollweide } from "app/science.js";
import { startAutopilot } from "app/autopilot.js";

const RAD = Math.PI / 180, X_MAX = 2 * Math.SQRT2, Y_MAX = Math.SQRT2;
const FONT = '"DejaVu Sans", Verdana, "Bitstream Vera Sans", sans-serif'; // matplotlib's default face
const MINUS = "−";
// step_2's sizes, in pt: tick labels, panel titles, (l, b) boxes, View labels; lines; the View dots.
const PT = { tick: 15, title: 13, info: 11, view: 13, line: 2.5, highlight: 7, ticklen: 3.5, tickw: 0.8, grid: 0.8, dot: 8, com: 6 };

/** matplotlib's MaxNLocator: the smallest of 1, 2, 2.5, 5, 10 × 10^k giving at most `bins` intervals. */
function ticks(lo, hi, bins) {
  const raw = (hi - lo) / Math.max(1, bins), p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw * (1 - 1e-9));
  const out = [];
  for (let t = Math.ceil(lo / step - 1e-9) * step; t <= hi + 1e-9; t += step) out.push(+t.toFixed(10));
  return out;
}
const num = (t) => (t < 0 ? MINUS : "") + String(Math.abs(t));
const round = (v) => { const r = Math.round(v); return Math.abs(v % 1) === 0.5 && r % 2 ? r - 1 : r; }; // Python's .0f

/** A text box as matplotlib's bbox=dict(boxstyle="round,pad=0.2", ...): lines of [text, italic?] runs. */
function box(ctx, lines, x, y, { size, bold = false, ha = "left", va = "top", face = "#fff", edge = "gray", alpha = 1, k }) {
  const px = size * k, pad = 0.2 * px, lh = 1.2 * px;
  const font = (it) => `${it ? "italic " : ""}${bold ? "bold " : ""}${px}px ${FONT}`;
  const width = (runs) => runs.reduce((w, [t, it]) => { ctx.font = font(it); return w + ctx.measureText(t).width; }, 0);
  const w = Math.max(...lines.map(width)), h = lh * lines.length;
  const x0 = ha === "left" ? x : ha === "right" ? x - w : x - w / 2;
  const y0 = va === "top" ? y : va === "bottom" ? y - h : y - h / 2;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.beginPath(); ctx.roundRect(x0 - pad, y0 - pad, w + 2 * pad, h + 2 * pad, pad);
  ctx.fillStyle = face; ctx.fill(); ctx.lineWidth = Math.max(1, 0.8 * k); ctx.strokeStyle = edge; ctx.stroke();
  ctx.restore();
  ctx.fillStyle = "#000"; ctx.textBaseline = "middle"; ctx.textAlign = "left";
  lines.forEach((runs, i) => {
    let cx = x0 + (w - width(runs)) * (ha === "right" ? 1 : ha === "center" ? 0.5 : 0);
    for (const [t, it] of runs) { ctx.font = font(it); ctx.fillText(t, cx, y0 + lh * (i + 0.5)); cx += ctx.measureText(t).width; }
  });
}

function mountFigure(el, model, { orbits, layout, layer, rNear }) {
  const [Wpt, Hpt] = layout.size_pt;
  el.style.aspectRatio = `${Wpt} / ${Hpt}`;
  let image = null;
  const results = {}, shown = {};
  const stage = createStage(el, { draw });
  createImageBitmap(new Blob([layer], { type: "image/png" })).then((b) => { image = b; stage.requestDraw(); });

  const geo = () => {
    const { width: W, height: H } = stage, m = layout.map;
    return { W, H, k: W / Wpt, cx: m.cx * W, cy: m.cy * H, rx: m.rx * W, ry: m.ry * H };
  };
  const toPx = (g, lb) => { const [x, y] = mollweide(lb); return [g.cx + (x / X_MAX) * g.rx, g.cy - (y / Y_MAX) * g.ry]; };
  const fromPx = (g, px, py) => inverseMollweide([((px - g.cx) / g.rx) * X_MAX, (-(py - g.cy) / g.ry) * Y_MAX]);
  const isPaper = (o, v) => v[0] === o.view[0] && v[1] === o.view[1];

  function panel(ctx, g, o, view) {
    const [fx, fy, fw, fh] = layout.panels[o.id], L = fx * g.W, T = fy * g.H, Wp = fw * g.W, Hp = fh * g.H, k = g.k;
    const { xy, mask, near } = results[o.id], n = mask.length;
    let [[x0, x1], [y0, y1]] = layout.paper_lims[o.id];
    if (!isPaper(o, view)) { // the paper's limits (step_2's _tight_lims, or orbit 2's fixed y) only at its view
      const lim = (c) => {
        let lo = Infinity, hi = -Infinity;
        for (let i = c; i < xy.length; i += 2) { lo = Math.min(lo, xy[i]); hi = Math.max(hi, xy[i]); }
        const m = 0.05 * (hi - lo); return [lo - m, hi + m];
      };
      [x0, x1] = lim(0); [y0, y1] = lim(1);
    }
    const X = (x) => L + ((x - x0) / (x1 - x0)) * Wp, Y = (y) => T + ((y1 - y) / (y1 - y0)) * Hp;
    const xt = ticks(x0, x1, Math.floor(fw * Wpt / (3 * PT.tick))), yt = ticks(y0, y1, Math.floor(fh * Hpt / (2 * PT.tick)));
    ctx.strokeStyle = "rgba(176,176,176,0.3)"; ctx.lineWidth = PT.grid * k; // ax.grid(True, alpha=0.3)
    ctx.beginPath();
    for (const t of xt) { ctx.moveTo(X(t), T); ctx.lineTo(X(t), T + Hp); }
    for (const t of yt) { ctx.moveTo(L, Y(t)); ctx.lineTo(L + Wp, Y(t)); }
    ctx.stroke();
    // Ticks outward, labels 15 pt: orbit 1's x along the top, orbit 2's along the bottom; y on the left.
    const top = o.id === "orbit1", tl = PT.ticklen * k, pad = 3.5 * k;
    ctx.strokeStyle = "#000"; ctx.lineWidth = PT.tickw * k; ctx.fillStyle = "#000"; ctx.font = `${PT.tick * k}px ${FONT}`;
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
    const segs = (keep, style, dash = []) => {
      ctx.beginPath();
      for (let i = 1; i < n; i++) if (keep(i) && keep(i - 1)) { ctx.moveTo(X(xy[2 * i - 2]), Y(xy[2 * i - 1])); ctx.lineTo(X(xy[2 * i]), Y(xy[2 * i + 1])); }
      Object.assign(ctx, style); ctx.setLineDash(dash); ctx.stroke(); ctx.setLineDash([]);
    };
    ctx.lineCap = "round";
    segs((i) => mask[i], { strokeStyle: "rgba(255,0,0,0.5)", lineWidth: PT.highlight * k }); // 𝒜 ≤ 0
    ctx.lineCap = "butt";
    const lw = PT.line * k;
    if (o.near_dotted) { // orbit 2: dotted within r_near of the centre
      ctx.globalAlpha = 0.8;
      segs((i) => !near[i], { strokeStyle: o.color, lineWidth: lw });
      segs((i) => near[i], { strokeStyle: o.color, lineWidth: lw }, [1 * lw, 1.65 * lw]);
    } else { // orbit 1: at 0.8 where concave, opaque where not
      ctx.globalAlpha = 0.8; segs((i) => !mask[i], { strokeStyle: o.color, lineWidth: lw });
      ctx.globalAlpha = 1; segs((i) => mask[i], { strokeStyle: o.color, lineWidth: lw });
    }
    ctx.globalAlpha = 1;
    const r = (PT.com / 2) * k; // the CoM, scatter's "x"
    ctx.strokeStyle = "#000"; ctx.lineWidth = 1.5 * k; ctx.beginPath();
    ctx.moveTo(X(0) - r, Y(0) - r); ctx.lineTo(X(0) + r, Y(0) + r); ctx.moveTo(X(0) - r, Y(0) + r); ctx.lineTo(X(0) + r, Y(0) - r); ctx.stroke();
    ctx.restore();

    box(ctx, [[[o.title, false]]], L + 0.05 * Wp, T + 0.05 * Hp, { size: PT.title, bold: true, k });
    const info = [[["r", true], [` ∈ (${o.r_range.map(Math.round).join(", ")}) kpc`, false]],
      [["(", false], ["l", true], [", ", false], ["b", true], [`) = (${round(view[0])}°, ${round(view[1])}°)`, false]]];
    if (top) box(ctx, info, L + 0.53 * Wp, T + 0.81 * Hp, { size: PT.info, ha: "right", va: "top", alpha: 0.8, k });
    else box(ctx, info, L + 0.54 * Wp, T + 0.95 * Hp, { size: PT.info, ha: "right", va: "bottom", alpha: 0.8, k });
  }

  function draw({ ctx, width, height }) {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, width, height); // the paper's figure is on white
    if (!image) return;
    ctx.drawImage(image, 0, 0, width, height);
    const g = geo(), views = model.get("views");
    for (const o of orbits) {
      if (shown[o.id] !== views[o.id]) { shown[o.id] = views[o.id]; results[o.id] = classify(o, views[o.id], o.near_dotted ? rNear : Infinity); }
      panel(ctx, g, o, views[o.id]);
    }
    orbits.forEach((o, n) => { // step_2's View dots (salmon ".", markersize 16) and their labels
      const v = views[o.id], [x, y] = toPx(g, v);
      ctx.beginPath(); ctx.arc(x, y, (PT.dot / 2) * g.k, 0, 2 * Math.PI); ctx.fillStyle = "#FA8072"; ctx.fill();
      const at = n === 0 ? [v[0] + 0.25 / RAD, v[1] + 0.15 / RAD] : [v[0], v[1] - 0.15 / RAD];
      const [lx, ly] = toPx(g, [Math.max(-180, Math.min(180, at[0])), Math.max(-90, Math.min(90, at[1]))]);
      box(ctx, [[[`View ${n + 1}`, false]]], lx, ly, { size: PT.view, bold: true, ha: "center", va: "center", edge: "#000", alpha: 0.6, k: g.k });
    });
  }

  const nearest = (x, y) => {
    const g = geo(), views = model.get("views");
    let best = null, dmin = Math.max(12, 10 * g.k);
    for (const o of orbits) { const [px, py] = toPx(g, views[o.id]), d = Math.hypot(px - x, py - y); if (d < dmin) { dmin = d; best = o.id; } }
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
    const source = model.get("manifest") ? model : (model.get("data_source") ?? "orbit-views");
    const data = await loadData(source, { base: model.get("data_base") ?? "data/" });
    const { meta, arrays } = data;
    const orbits = meta.orbits.map((o) => orbitFromData(data, o.id));
    const paperViews = () => Object.fromEntries(meta.orbits.map((o) => [o.id, o.view.slice()]));
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
    const update = () => {
      const v = model.get("views");
      readout.textContent = orbits.map((o, n) => `Orbit ${n + 1}: ${(100 * classify(o, v[o.id]).fraction).toFixed(1)}% convex`).join(" · ");
    };
    model.on("change:views", update);
    update();
    toolbar.append(reset, "Drag a View dot on the map.", readout);
    const scroller = document.createElement("div");
    scroller.className = "ov-scroll";
    const fig = document.createElement("div");
    fig.className = "ov-figure";
    scroller.append(fig);
    el.replaceChildren(toolbar, scroller);
    const dispose = mountFigure(fig, model, { orbits, layout: meta.layout, layer: arrays.paper_layer.data, rNear: meta.r_near_kpc });
    const stopAutopilot = startAutopilot(model, el);
    signal?.addEventListener("abort", () => { stopAutopilot(); dispose(); model.off("change:views", update); });
  },
};
