// Uniform boxes on Paper Figure 5: the paper's own figure (step_2_plot_triaxiality.py --simple), on
// whose left panel the reader draws boxes of halo shapes (c/a, b/a). Each box weights every shape in
// it equally, and shows up on the right panel as the paper's CDM and SIDM do: a dot-dashed line at
// its mean projected flattening and diamonds at its mean lambda_det^(10) and lambda_arc. The paper's
// CDM and SIDM are weighted by their halos instead, which the table under the figure sets beside the
// boxes. An anywidget module, so the same file runs on the web page (figpress-apps' `run`) and in
// Jupyter/marimo.

import { loadData } from "figpress-apps/data.js";
import { createStage, dragHelper } from "figpress-apps/stage.js";
import { boxMean, boxFrom } from "app/science.js";

const FONT = '"DejaVu Sans", Verdana, "Bitstream Vera Sans", sans-serif'; // matplotlib's default face
const COLORS = ["#ff7f0e", "#9467bd", "#e377c2", "#bcbd22", "#8c564b"]; // clear of the CDM and SIDM colours
const pct = (v, d = 2) => `${(100 * v).toFixed(d)}%`;
const bitmap = (bytes) => createImageBitmap(new Blob([bytes], { type: "image/png" }));

/** An axes' data ↔ pixel maps, from its [left, top, width, height] fractions and limits. */
function axesMap({ axes: [l, t, w, h], xlim: [x0, x1], ylim: [y0, y1] }, W, H) {
  const L = l * W, T = t * H, Wp = w * W, Hp = h * H;
  return {
    L, T, Wp, Hp,
    X: (x) => L + ((x - x0) / (x1 - x0)) * Wp, Y: (y) => T + ((y1 - y) / (y1 - y0)) * Hp,
    x: (px) => x0 + ((px - L) / Wp) * (x1 - x0), y: (py) => y1 - ((py - T) / Hp) * (y1 - y0),
    inside: (px, py) => px >= L && px <= L + Wp && py >= T && py <= T + Hp,
    clampX: (px) => Math.min(Math.max(px, L), L + Wp), clampY: (py) => Math.min(Math.max(py, T), T + Hp),
    xlim: [x0, x1],
  };
}

function mountFigure(el, model, { grid, layout, paper }) {
  const [Wpt, Hpt] = layout.size_pt;
  el.style.aspectRatio = `${Wpt} / ${Hpt}`;
  let img = null, draft = null;
  const stage = createStage(el, { draw });
  bitmap(paper).then((b) => { img = b; stage.requestDraw(); });
  const geo = () => {
    const { width: W, height: H } = stage;
    return { k: W / Wpt, left: axesMap(layout.left, W, H), right: axesMap(layout.right, W, H) };
  };

  function draw({ ctx, width, height }) {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, width, height); // the paper's figure is on white
    if (!img) return;
    ctx.drawImage(img, 0, 0, width, height);
    const { k, left, right } = geo();
    const boxes = model.get("boxes");
    // Left: each box, white under its colour so it reads on any part of the colour map.
    ctx.save();
    ctx.beginPath(); ctx.rect(left.L, left.T, left.Wp, left.Hp); ctx.clip();
    for (const [i, b] of [...boxes, ...(draft ? [draft] : [])].entries()) {
      const x = left.X(b.ca[1]), y = left.Y(b.ba[0]), w = left.X(b.ca[0]) - x, h = left.Y(b.ba[1]) - y;
      for (const [color, lw] of [["#fff", 4], [b.color, 2]]) {
        ctx.setLineDash(b === draft ? [4 * k, 3 * k] : []);
        ctx.lineWidth = lw * k; ctx.strokeStyle = color; ctx.strokeRect(x, y, w, h);
      }
      if (b !== draft) label(ctx, String(i + 1), x + 3 * k, y + 3 * k, b.color, k, "left", "top");
    }
    ctx.restore();
    // Right: a line at each box's mean q̄, diamonds at its mean rates, as the paper marks CDM and SIDM.
    ctx.save();
    ctx.beginPath(); ctx.rect(right.L, right.T, right.Wp, right.Hp); ctx.clip();
    boxes.forEach((b, i) => {
      const m = boxMean(grid, b);
      if (!m.n) return;
      const px = right.X(m.qbar), edge = right.L + right.Wp;
      if (px > edge) { // flatter than the panel reaches: an arrow at its edge, at the box's lambda_det^(10)
        const y = Math.min(right.Y(m.lam10), right.T + right.Hp - 8 * k), r = 6 * k;
        ctx.beginPath(); ctx.moveTo(edge, y); ctx.lineTo(edge - r, y - r); ctx.lineTo(edge - r, y + r); ctx.closePath();
        ctx.fillStyle = b.color; ctx.fill(); ctx.lineWidth = 0.8 * k; ctx.strokeStyle = "#000"; ctx.stroke();
        label(ctx, `${i + 1} uniform →`, edge - r - 3 * k, y + 4 * k, b.color, k, "right", "bottom");
        return;
      }
      ctx.setLineDash([6 * k, 2 * k, 1 * k, 2 * k]);
      ctx.lineWidth = 1.4 * k; ctx.strokeStyle = b.color;
      ctx.beginPath(); ctx.moveTo(px, right.T); ctx.lineTo(px, right.T + right.Hp); ctx.stroke();
      ctx.setLineDash([]);
      for (const v of [m.lam10, m.lamarc]) diamond(ctx, px, right.Y(v), 3.6 * k, b.color, k);
      const side = px > right.L + right.Wp - 90 * k ? "right" : "left";
      label(ctx, `${i + 1} uniform`, px + (side === "left" ? 5 : -5) * k, right.Y(m.lam10) - 8 * k, b.color, k, side, "bottom");
    });
    ctx.restore();
  }

  // Drag on the left panel to draw a box; a click on a box (no drag) removes it.
  let start = null;
  const undrag = dragHelper(stage.canvas, {
    hitTest: (x, y) => (geo().left.inside(x, y) ? [x, y] : null),
    onStart: (s) => { start = s; },
    onMove: (s, _dx, _dy, x, y) => {
      if (!s) return;
      const { left } = geo(), [x0, y0] = s, [x1, y1] = [left.clampX(x), left.clampY(y)];
      draft = { ...boxFrom([left.x(x0), left.y(y0)], [left.x(x1), left.y(y1)]), color: nextColor(),
                big: Math.abs(x1 - x0) > 4 && Math.abs(y1 - y0) > 4 };
      stage.requestDraw();
    },
    onEnd: (s) => {
      const boxes = model.get("boxes");
      if (draft?.big) {
        const { ca, ba, color } = draft;
        model.set("boxes", [...boxes, { ca, ba, color }]);
      } else if (s) { // a click: remove the last-drawn box under it
        const { left } = geo(), ca = left.x(s[0]), ba = left.y(s[1]);
        const j = boxes.findLastIndex((b) => ca >= b.ca[0] && ca <= b.ca[1] && ba >= b.ba[0] && ba <= b.ba[1]);
        if (j >= 0) model.set("boxes", boxes.filter((_b, i) => i !== j));
      }
      draft = null; start = null;
      model.save_changes();
      stage.requestDraw();
    },
  });
  const nextColor = () => {
    const used = model.get("boxes").map((b) => b.color);
    return COLORS.find((c) => !used.includes(c)) ?? COLORS[model.get("boxes").length % COLORS.length];
  };
  const hover = (e) => {
    if (start) return;
    const b = stage.canvas.getBoundingClientRect();
    stage.canvas.style.cursor = geo().left.inside(e.clientX - b.left, e.clientY - b.top) ? "crosshair" : "";
  };
  stage.canvas.addEventListener("pointermove", hover);
  model.on("change:boxes", stage.requestDraw);
  return () => { undrag(); stage.canvas.removeEventListener("pointermove", hover); model.off("change:boxes", stage.requestDraw); stage.dispose(); };
}

function diamond(ctx, x, y, r, fill, k) {
  ctx.beginPath(); ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath();
  ctx.fillStyle = fill; ctx.fill(); ctx.lineWidth = 0.8 * k; ctx.strokeStyle = "#000"; ctx.stroke();
}

function label(ctx, text, x, y, color, k, align, baseline) {
  ctx.font = `bold ${9 * k}px ${FONT}`; ctx.textAlign = align; ctx.textBaseline = baseline;
  const w = ctx.measureText(text).width, h = 11 * k, pad = 1.5 * k;
  const bx = align === "left" ? x - pad : x - w - pad, by = baseline === "top" ? y - pad : y - h - pad;
  ctx.fillStyle = "rgba(255,255,255,0.85)"; ctx.fillRect(bx, by, w + 2 * pad, h + 2 * pad);
  ctx.fillStyle = color; ctx.fillText(text, x, y);
}

/** The table under the figure: each box, uniform, beside the paper's CDM and SIDM, halo-weighted. */
function table(grid, meta, boxes) {
  const off = (q) => (q < meta.layout.right.xlim[1] ? " (off the panel)" : "");
  const row = (swatch, name, weighting, m) => `<tr><td>${swatch}</td><td>${name}</td><td>${weighting}</td>` +
    (m.n === 0 ? `<td colspan="3">no shapes with b/a ≥ c/a</td>` :
      `<td>${m.qbar.toFixed(3)}${off(m.qbar)}</td><td>${pct(m.lam10)}</td><td>${pct(m.lamarc)}</td>`) + "</tr>";
  const dot = (c) => `<span class="ub-swatch" style="background:${c}"></span>`;
  const range = ([lo, hi]) => `${lo.toFixed(3)}–${hi.toFixed(3)}`;
  const { cdm, sidm } = meta.populations;
  return `<table class="ub-table"><thead><tr><th></th><th>Shapes</th><th>Weighting</th><th>q̄</th>` +
    `<th>λ<sub>det</sub><sup>(10)</sup></th><th>λ<sub>arc</sub></th></tr></thead><tbody>` +
    row(dot("deepskyblue"), "CDM (paper)", "the Auriga halos", cdm) +
    row(dot("#7DCEA0"), "SIDM (paper)", "the rounded Auriga halos", sidm) +
    boxes.map((b, i) => row(dot(b.color), `Box ${i + 1}: c/a ${range(b.ca)}, b/a ${range(b.ba)}`,
                            "<strong>uniform over the box</strong>", boxMean(grid, b))).join("") +
    `</tbody></table>`;
}

export default {
  async render({ model, el, signal }) {
    const source = model.get("manifest") ? model : (model.get("data_source") ?? "uniform-box-rates");
    const { meta, arrays } = await loadData(source, { base: model.get("data_base") ?? "data/" });
    const grid = Object.fromEntries(["ca", "ba", "qbar", "lam10", "lamarc"].map((k) => [k, arrays[k].data]));
    if (!Array.isArray(model.get("boxes"))) model.set("boxes", []);
    model.save_changes();

    el.classList.add("fk-app");
    el.dataset.figpressApp = "";
    const toolbar = document.createElement("div");
    toolbar.className = "fk-toolbar";
    const clear = document.createElement("button");
    clear.textContent = "Clear boxes";
    clear.onclick = () => { model.set("boxes", []); model.save_changes(); };
    toolbar.append(clear, "Drag on the left panel to draw a box of halo shapes; click a box to remove it.");
    const note = document.createElement("p");
    note.className = "fk-note";
    note.innerHTML = "A box weights <strong>every shape in it equally</strong>. The paper's CDM and SIDM are not " +
      "boxes: they average over the halos' own shapes, which crowd some parts of their dotted boxes and leave " +
      "others empty. So a box drawn over a dotted box need not land on the paper's line.";
    const scroller = document.createElement("div");
    scroller.className = "ov-scroll";
    const fig = document.createElement("div");
    fig.className = "ov-figure";
    scroller.append(fig);
    const rows = document.createElement("div");
    rows.className = "ov-scroll";
    const update = () => { rows.innerHTML = table(grid, meta, model.get("boxes")); };
    model.on("change:boxes", update);
    update();
    el.replaceChildren(toolbar, note, scroller, rows);
    const dispose = mountFigure(fig, model, { grid, layout: meta.layout, paper: arrays.paper.data });
    signal?.addEventListener("abort", () => { dispose(); model.off("change:boxes", update); });
  },
};
