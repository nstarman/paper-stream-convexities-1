// Paper Figure 6, interactive: the paper's own figure (streams_required_pcdm.py) with its curves drawn
// here, so each can be switched off from the legend (top left; the paper has none), whose last row
// takes the reader's own N and draws its curve in red (off until one is entered). Near a curve, a
// dot marks the nearest point on it and a box gives its catalogue size N, observed rate K/N, K, and
// Z_LR. An anywidget module, so the same file runs on the web page (figpress-apps' `run`) and in
// Jupyter/marimo.

import { loadData } from "figpress-apps/data.js";
import { createStage } from "figpress-apps/stage.js";

const FONT = '"DejaVu Sans", Verdana, "Bitstream Vera Sans", sans-serif'; // matplotlib's default face
const MINUS = "−";
const signed = (v, d) => (v < 0 ? MINUS : "+") + Math.abs(v).toFixed(d);
/** Z_LR at observed rate x for N streams (signed_lr_sigma): positive favours CDM. */
export function zLR(x, pc, ps, n) {
  const dll = n * (x * Math.log(pc / ps) + (1 - x) * Math.log((1 - pc) / (1 - ps)));
  return Math.sign(dll) * Math.sqrt(2 * Math.abs(dll));
}
/** The next round N above (dir 1) or below (dir −1) n, on 1, 1.5, 2, 3, 5, 7 × 10^k; 1000 from nothing. */
export function logStep(n, dir) {
  if (!(n > 0)) return 1000;
  const k = Math.floor(Math.log10(n)), nice = [];
  for (let e = k - 1; e <= k + 1; e++) for (const m of [1, 1.5, 2, 3, 5, 7]) nice.push(Math.round(m * 10 ** e));
  const next = dir > 0 ? nice.find((v) => v > n) : nice.findLast((v) => v < n);
  return Math.max(1, next ?? (dir > 0 ? nice.at(-1) : 1));
}
const bitmap = (bytes) => createImageBitmap(new Blob([bytes], { type: "image/png" }));

function mountFigure(el, model, { curves, rate, rates, layout, images }) {
  const [Wpt, Hpt] = layout.size_pt, [ax, ay, aw, ah] = layout.axes;
  const [[x0, x1], [y0, y1]] = [layout.xlim, layout.ylim];
  el.style.aspectRatio = `${Wpt} / ${Hpt}`;
  let img = null, hit = null;
  const own = curves.at(-1), J = curves.length - 1; // the reader's curve, last
  const input = document.createElement("input");
  Object.assign(input, { type: "number", min: "1", step: "any", placeholder: "your N", ariaLabel: "Your catalogue size N" });
  const [fl, ft, fw, fh] = own.field;
  input.style.cssText = `position:absolute;left:${100 * fl}%;top:${100 * ft}%;width:${100 * fw}%;height:${100 * fh}%;` +
    `box-sizing:border-box;padding:0 0.2em;border:1px solid #bbb;border-radius:2px;background:#fff;color:#000;font-family:${FONT}`;
  input.value = model.get("custom_n") ?? "";
  el.append(input);
  const setOwn = () => { const n = model.get("custom_n"); own.n = n; own.z = n > 0 ? rate.map((x) => zLR(x, rates.cdm, rates.sidm, n)) : null; };
  setOwn();
  const stage = createStage(el, { draw, onResize: (s) => { input.style.fontSize = `${(7 * s.width) / Wpt}px`; } });
  Promise.all([images.paper, images.over, images.legend, ...images.labels].map(bitmap))
    .then(([paper, over, legend, ...labels]) => { img = { paper, over, legend, labels }; stage.requestDraw(); });
  const shown = (j) => !model.get("hidden").includes(j);

  const geo = () => {
    const { width: W, height: H } = stage;
    const L = ax * W, T = ay * H, Wp = aw * W, Hp = ah * H;
    return { W, H, k: W / Wpt, L, T, Wp, Hp, X: (x) => L + ((x - x0) / (x1 - x0)) * Wp, Y: (y) => T + ((y1 - y) / (y1 - y0)) * Hp };
  };

  /** The legend entry at (px, py), or −1. */
  function entryAt(px, py) {
    const { W, H } = geo();
    return curves.findIndex(({ entry: [l, t, w, h] }) => px >= l * W && px <= (l + w) * W && py >= t * H && py <= (t + h) * H);
  }

  /** The point of a shown curve nearest (px, py) on screen, within reach, or null. */
  function nearest(px, py) {
    const g = geo();
    if (px < g.L || px > g.L + g.Wp || py < g.T || py > g.T + g.Hp) return null;
    let best = null, dmin = Math.max(10, 8 * g.k);
    curves.forEach((c, j) => {
      if (!shown(j) || !c.z) return;
      for (let i = 0; i < rate.length; i++) {
        const x = rate[i], z = c.z[i];
        if (x < x0 || x > x1 || !(z >= y0 && z <= y1)) continue;
        const d = Math.hypot(g.X(x) - px, g.Y(z) - py);
        if (d < dmin) { dmin = d; best = { j, i }; }
      }
    });
    return best;
  }

  /** A curve as the paper strokes it: its grid's segments, a dashed curve keeping `on` of every on + off. */
  function curve(ctx, g, c) {
    const [on, off] = c.dash ?? [1, 0];
    ctx.beginPath();
    for (let i = 1; i < rate.length; i++) {
      if ((i - 1) % (on + off) >= on) continue;
      if ((i - 1) % (on + off) === 0 || i === 1) ctx.moveTo(g.X(rate[i - 1]), g.Y(c.z[i - 1]));
      ctx.lineTo(g.X(rate[i]), g.Y(c.z[i]));
    }
    ctx.strokeStyle = c.color; ctx.lineWidth = c.lw * g.k; ctx.lineJoin = "round"; ctx.lineCap = "butt";
    ctx.stroke();
  }

  function draw({ ctx, width, height }) {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, width, height); // the paper's figure is on white
    if (!img) return;
    const g = geo(), k = g.k;
    ctx.drawImage(img.paper, 0, 0, width, height);
    ctx.save();
    ctx.beginPath(); ctx.rect(g.L, g.T, g.Wp, g.Hp); ctx.clip();
    curves.forEach((c, j) => shown(j) && c.z && curve(ctx, g, c));
    ctx.restore();
    curves.forEach((_c, j) => shown(j) && img.labels[j] && ctx.drawImage(img.labels[j], 0, 0, width, height));
    ctx.drawImage(img.over, 0, 0, width, height);
    ctx.drawImage(img.legend, 0, 0, width, height);
    curves.forEach(({ entry: [l, t, w, h] }, j) => { // a switched-off entry, faded
      if (shown(j)) return;
      ctx.fillStyle = "rgba(255,255,255,0.7)"; ctx.fillRect(l * width, t * height, w * width, h * height);
    });
    if (!hit) return;
    const c = curves[hit.j], x = rate[hit.i], z = c.z[hit.i], px = g.X(x), py = g.Y(z);
    ctx.beginPath(); ctx.arc(px, py, 3 * k, 0, 2 * Math.PI);
    ctx.fillStyle = c.color; ctx.fill(); ctx.lineWidth = 1.2 * k; ctx.strokeStyle = "#fff"; ctx.stroke();

    const favours = z > 0 ? "favours CDM" : z < 0 ? "favours SIDM" : "neither";
    // Each line as [text, bold?, subscript?] runs; Z's "LR" as a subscript, as the paper sets it.
    const size = 7 * k, pad = 3 * k, lh = 1.3 * size;
    const lines = [[[`N = ${c.n.toLocaleString("en-US")}${c.euclid ? " (Euclid minimum)" : c === own ? " (yours)" : ""}`, true]],
      [[`K/N = ${(100 * x).toFixed(2)}%  (K ≈ ${Math.round(c.n * x).toLocaleString("en-US")})`]],
      [["Z"], ["LR", false, true], [` = ${signed(z, 2)}, ${favours}`]]];
    const font = ([, bold, sub]) => `${bold ? "bold " : ""}${sub ? 0.7 * size : size}px ${FONT}`;
    const measure = (runs) => runs.reduce((s, r) => { ctx.font = font(r); return s + ctx.measureText(r[0]).width; }, 0);
    const w = Math.max(...lines.map(measure)) + 2 * pad, h = lines.length * lh + 2 * pad;
    // Beside the dot, on whichever side keeps the box inside the axes.
    const bx = px + 8 * k + w > g.L + g.Wp ? px - 8 * k - w : px + 8 * k;
    const by = Math.min(Math.max(py - h / 2, g.T + 2 * k), g.T + g.Hp - h - 2 * k);
    ctx.beginPath(); ctx.roundRect(bx, by, w, h, 2 * k);
    ctx.fillStyle = "rgba(255,255,255,0.95)"; ctx.fill(); ctx.lineWidth = Math.max(1, 0.6 * k); ctx.strokeStyle = "gray"; ctx.stroke();
    ctx.fillStyle = "#000"; ctx.textBaseline = "middle"; ctx.textAlign = "left";
    lines.forEach((runs, n) => {
      let cx = bx + pad;
      for (const r of runs) {
        ctx.font = font(r);
        ctx.fillText(r[0], cx, by + pad + (n + 0.5) * lh + (r[2] ? 0.3 * size : 0));
        cx += ctx.measureText(r[0]).width;
      }
    });
  }

  const at = (e) => { const b = stage.canvas.getBoundingClientRect(); return [e.clientX - b.left, e.clientY - b.top]; };
  const move = (e) => {
    const [px, py] = at(e), onEntry = entryAt(px, py) >= 0, h = onEntry ? null : nearest(px, py);
    stage.canvas.style.cursor = onEntry ? "pointer" : h ? "crosshair" : "";
    if (h?.j === hit?.j && h?.i === hit?.i) return;
    hit = h; stage.requestDraw();
  };
  const leave = () => { if (hit) { hit = null; stage.requestDraw(); } };
  const click = (e) => {
    const j = entryAt(...at(e));
    if (j < 0) return;
    if (j === J && !own.z) { input.focus(); return; } // nothing to show yet: ask for an N
    const hidden = model.get("hidden");
    model.set("hidden", hidden.includes(j) ? hidden.filter((i) => i !== j) : [...hidden, j]);
    model.save_changes();
  };
  const redraw = () => { if (hit && (!shown(hit.j) || !curves[hit.j].z)) hit = null; stage.requestDraw(); };
  const typed = () => { // a valid N shows the curve; clearing it hides it
    const n = Number(input.value), ok = input.value !== "" && n > 0 && Number.isFinite(n);
    const hidden = model.get("hidden").filter((i) => i !== J);
    model.set("custom_n", ok ? n : null);
    model.set("hidden", ok ? hidden : [...hidden, J]);
    model.save_changes();
  };
  // Stepping (the spinner, wheel or ↑/↓) goes up or down a log scale of round N, not by 1. A step
  // fires an input event with no inputType; typing and pasting have one.
  const step = (dir) => { input.value = logStep(Number(input.value), dir); typed(); };
  let prev = Number(input.value) || null;
  const onInput = (e) => {
    if (!e.inputType && input.value !== "") {
      const dir = prev ? Math.sign(Number(input.value) - prev) : 0;
      input.value = dir ? logStep(prev, dir) : logStep(null, 1);
    }
    typed();
    prev = Number(input.value) > 0 ? Number(input.value) : null;
  };
  const onKey = (e) => {
    if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
    e.preventDefault();
    step(e.key === "ArrowUp" ? 1 : -1);
    prev = Number(input.value);
  };
  const changed = () => { setOwn(); redraw(); };
  input.addEventListener("input", onInput);
  input.addEventListener("keydown", onKey);
  model.on("change:custom_n", changed);
  stage.canvas.addEventListener("pointermove", move);
  stage.canvas.addEventListener("pointerleave", leave);
  stage.canvas.addEventListener("click", click);
  model.on("change:hidden", redraw);
  return () => {
    stage.canvas.removeEventListener("pointermove", move);
    stage.canvas.removeEventListener("pointerleave", leave);
    stage.canvas.removeEventListener("click", click);
    model.off("change:hidden", redraw);
    model.off("change:custom_n", changed);
    input.removeEventListener("keydown", onKey);
    input.remove();
    stage.dispose();
  };
}

export default {
  async render({ model, el, signal }) {
    const source = model.get("manifest") ? model : (model.get("data_source") ?? "likelihood-ratio");
    const { meta, arrays } = await loadData(source, { base: model.get("data_base") ?? "data/" });
    const curves = meta.curves.map((c, j) => ({ ...c, z: arrays[`z${j}`].data }));
    curves.push({ ...meta.custom, n: null, z: null }); // the reader's own N
    if (!Array.isArray(model.get("hidden"))) model.set("hidden", [curves.length - 1]); // theirs off at first
    model.save_changes();

    el.classList.add("fk-app");
    el.dataset.figpressApp = "";
    const toolbar = document.createElement("div");
    toolbar.className = "fk-toolbar";
    toolbar.append("Point at a curve for its values; click a legend entry to show or hide its curve, or enter your own N in its last row.");
    const scroller = document.createElement("div");
    scroller.className = "ov-scroll";
    const fig = document.createElement("div");
    fig.className = "ov-figure";
    scroller.append(fig);
    el.replaceChildren(toolbar, scroller);
    const images = { paper: arrays.paper.data, over: arrays.over.data, legend: arrays.legend.data,
                     labels: meta.curves.map((_c, j) => arrays[`label${j}`].data) };
    const dispose = mountFigure(fig, model, { curves, rate: arrays.rate.data, rates: meta.rates, layout: meta.layout, images });
    signal?.addEventListener("abort", dispose);
  },
};
