// The autopilot: when the app opens, both markers glide along the bright arcs of their streams' maps (the
// fraction of the track that is non-concave from each view), back and forth, until the first click or key press.
// The waypoints, (l, b) in degrees, were traced through the brightest cells of each stream's map (from classify on
// a 3 degree grid); science.test.js checks that they still are bright.

export const SPEED = 22; // degrees per second
const ROUTES = { spray: { closed: false }, nbody: { closed: false } };
const PATHS = {
  spray: [[-175.5,-59.0],[-172.5,-58.9],[-169.5,-58.9],[-166.5,-58.8],[-163.5,-58.6],[-160.5,-58.2],[-157.5,-57.7],[-154.5,-57.1],[-151.5,-56.4],[-148.5,-55.5],[-145.5,-54.6],[-142.5,-53.5],[-139.5,-52.4],[-136.5,-51.1],[-133.5,-49.6],[-130.5,-48.1],[-127.5,-46.3],[-124.5,-44.3],[-121.5,-42.1],[-118.5,-39.6],[-115.5,-36.9],[-112.5,-33.9],[-109.5,-30.6],[-106.5,-27.1],[-103.5,-23.2],[-100.5,-19.1],[-97.5,-14.6],[-94.5,-10.2],[-91.5,-5.6],[-88.5,-0.9],[-85.5,4.6],[-82.5,10.8],[-79.5,17.3],[-76.5,23.5],[-73.5,29.3],[-70.5,34.6],[-67.5,39.5],[-64.5,43.0],[-61.5,45.6],[-58.5,47.2],[-55.5,48.6],[-52.5,49.9],[-49.5,51.2],[-46.5,52.3],[-43.5,53.5],[-40.5,54.5],[-37.5,55.5],[-34.5,56.3],[-31.5,57.1],[-28.5,57.7],[-25.5,58.1],[-22.5,58.5],[-19.5,58.8],[-16.5,59.1],[-13.5,59.5],[-10.5,59.8],[-7.5,59.9],[-4.5,59.9],[-1.5,59.9],[1.5,59.8],[4.5,59.6],[7.5,59.3],[10.5,59.0],[13.5,58.8],[16.5,58.6],[19.5,58.2],[22.5,57.7],[25.5,57.1],[28.5,56.4],[31.5,55.5],[34.5,54.6],[37.5,53.5],[40.5,52.4],[43.5,51.1],[46.5,49.6],[49.5,48.1],[52.5,46.3],[55.5,44.3],[58.5,42.1],[61.5,39.6],[64.5,36.9],[67.5,33.9],[70.5,30.6],[73.5,27.1],[76.5,23.2],[79.5,19.1],[82.5,14.6],[85.5,10.2],[88.5,5.6],[91.5,0.9],[94.5,-4.6],[97.5,-10.8],[100.5,-17.3],[103.5,-23.5],[106.5,-29.3],[109.5,-34.6],[112.5,-39.5],[115.5,-43.0],[118.5,-45.6],[121.5,-47.2],[124.5,-48.6],[127.5,-49.9],[130.5,-51.2],[133.5,-52.3],[136.5,-53.5],[139.5,-54.5],[142.5,-55.5],[145.5,-56.3],[148.5,-57.1],[151.5,-57.7],[154.5,-58.1],[157.5,-58.5],[160.5,-58.8],[163.5,-59.1],[166.5,-59.5],[169.5,-59.9],[172.5,-60.2],[175.5,-60.6]],
  nbody: [[-54.0,-64.5],[-52.7,-64.7],[-50.9,-65.2],[-48.7,-65.8],[-46.5,-66.4],[-44.4,-67.0],[-42.1,-67.4],[-39.7,-67.5],[-37.3,-67.5],[-34.8,-67.5],[-32.4,-67.5],[-29.9,-67.5],[-27.5,-67.5],[-25.1,-67.5],[-22.6,-67.5],[-20.3,-67.3],[-18.0,-66.8],[-16.0,-66.0],[-14.0,-64.8],[-12.2,-63.2],[-10.5,-61.5],[-8.8,-59.8],[-7.1,-58.1],[-5.3,-56.3],[-3.6,-54.6],[-1.9,-52.9],[-0.2,-51.2],[1.6,-49.4],[3.3,-47.7],[5.0,-46.0],[6.7,-44.3],[8.4,-42.6],[10.2,-40.8],[11.9,-39.1],[13.6,-37.4],[15.3,-35.7],[17.1,-33.9],[18.8,-32.2],[20.5,-30.5],[22.2,-28.8],[24.0,-27.0],[25.7,-25.3],[27.4,-23.6],[29.1,-21.9],[30.8,-20.2],[32.6,-18.4],[34.3,-16.7],[36.0,-15.0],[37.7,-13.3],[39.5,-11.5],[41.2,-9.8],[42.9,-8.1],[44.6,-6.4],[46.3,-4.7],[48.1,-2.9],[49.8,-1.2],[51.5,0.5],[53.2,2.2],[55.0,4.0],[56.7,5.7],[58.4,7.3],[60.3,8.6],[62.4,9.6],[64.5,10.2],[66.9,10.5],[69.3,10.5],[71.7,10.5],[74.2,10.5],[76.5,10.8],[78.7,11.4],[80.7,12.4],[82.6,13.7],[84.4,15.4],[86.1,17.1],[87.8,18.8],[89.5,20.5],[91.2,22.2],[93.0,24.0],[94.7,25.7],[96.4,27.4],[97.9,29.2],[99.2,31.1],[100.5,33.0],[101.6,35.0],[102.3,37.1],[103.0,39.3],[103.9,41.4],[104.7,43.5],[105.8,45.4],[107.3,47.3],[109.0,49.0],[110.7,50.7],[112.4,52.4],[114.2,54.2],[115.9,55.9],[117.6,57.6],[119.3,59.3],[121.1,61.1],[122.4,62.4],[123.5,63.5]],
};

// Arc length along the waypoints, and the point at arc length s (ping-pong on an open path, a loop on a closed one).
function arcLengths(pts, closed) {
  const d = [0], n = pts.length + (closed ? 1 : 0);
  for (let i = 1; i < n; i++) {
    const a = pts[i - 1], b = pts[i % pts.length];
    d.push(d[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1]));
  }
  return d;
}

/** The view (l, b) at distance s travelled along a route: a closed route loops, an open one goes back and forth. */
export function routeAt({ pts, closed, d }, s) {
  const L = d[d.length - 1];
  s = closed ? ((s % L) + L) % L : (() => { const u = ((s % (2 * L)) + 2 * L) % (2 * L); return u < L ? u : 2 * L - u; })();
  let i = 1;
  while (i < d.length - 1 && d[i] < s) i++;
  const t = (s - d[i - 1]) / ((d[i] - d[i - 1]) || 1), a = pts[i - 1], b = pts[i % pts.length];
  return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
}

/** Each route with its arc lengths, keyed by view id. */
export function routes() {
  return Object.fromEntries(Object.entries(ROUTES).map(([id, { closed }]) => [id, { pts: PATHS[id], closed, d: arcLengths(PATHS[id], closed) }]));
}

/**
 * Start the autopilot: every view in ROUTES glides along its route from the waypoint nearest where it is, at
 * SPEED degrees a second, until the first pointer-down or key press in `el`. Not when the reader asks for
 * reduced motion. Returns a function that stops it.
 */
export function startAutopilot(model, el) {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return () => {};
  const rs = routes(), v0 = model.get("views"), s = {};
  for (const [id, r] of Object.entries(rs)) {
    let best = 0;
    r.pts.forEach((p, i) => { if (Math.hypot(p[0] - v0[id][0], p[1] - v0[id][1]) < Math.hypot(r.pts[best][0] - v0[id][0], r.pts[best][1] - v0[id][1])) best = i; });
    s[id] = r.d[best];
  }
  let raf = 0, last = performance.now();
  const step = (now) => {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    const views = { ...model.get("views") };
    for (const [id, r] of Object.entries(rs)) views[id] = routeAt(r, (s[id] += SPEED * dt));
    model.set("views", views);
    raf = requestAnimationFrame(step);
  };
  raf = requestAnimationFrame(step);
  const evs = ["pointerdown", "keydown"];
  const stop = () => { cancelAnimationFrame(raf); evs.forEach((ev) => el.removeEventListener(ev, onUser, true)); };
  const onUser = () => { stop(); model.save_changes(); };
  evs.forEach((ev) => el.addEventListener(ev, onUser, true));
  return stop;
}
