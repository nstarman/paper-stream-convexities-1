"""Paper Figure 7 (fig:cosmology:Nreq) with figpress-plotly: the web figure on the site, with hover,
σ toggles and λ_SI/λ_C as a control. Drawn as src/scripts/cosmology/streams_required_pcdm.py draws
the paper's figure, from the values that script writes; its PDF matches the paper's within 1 pt:

    uv run --no-project --with figpress-plotly==2026.10.29.2 --with matplotlib \
        figpress-plotly build webapps/streams-required/figure.py:figure --out build/ --target pdf
    ... figpress-plotly compare build/figure.pdf src/tex/figures/cosmology/streams_required.pdf
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from matplotlib import colormaps, colors

import figpress_plotly as fp

ROOT = Path(__file__).resolve().parents[2]
V = json.loads((ROOT / "src/data/values/streams_required_pcdm.values.json").read_text(encoding="utf-8"))

ETA_LO, ETA_FID, ETA_HI = V["eta_lo"], V["eta_fid"], V["eta_hi"]
LAM_C = V["lambda_cdm_sim"]
N_EUCLID = V["n_euclid_min"]
SIGMAS = [1, 2, 3, 5]
P_MIN, P_MAX = 0.001, 0.05
P = np.logspace(np.log10(P_MIN), np.log10(P_MAX), 800)
N_MIN, N_MAX = 1e2, 5e5
SURVEYS = [("Starkman et al. 2026 Euclid Projection", N_EUCLID, 0.25, True), ("Current (DES, SSLS)", 100, 0.45, False)]

# The paper's figure, measured from its PDF (pt from the page's top left): the axes frame at its
# spines' centres, the band widget's box and band, and where labels sit that matplotlib placed.
PAGE = (246.86, 207.31)
FRAME = (32.58, 1.38, 206.52, 180.42)
SPINE = 0.7
WIDGET = dict(box=(34.6, 140.6, 90.5, 175.4), band=(63.6, 143.6, 79.0, 168.6), labels=81.0, baseline=2.06)


def n_required(z, p_c, p_s):
    num = (np.sqrt(p_c * (1 - p_c)) + np.sqrt(p_s * (1 - p_s))) ** 2
    return z**2 * num / (p_c - p_s) ** 2


def sigma_color(z):
    """The level's colour and alpha (1σ in faint grey), as the paper's script."""
    if z == 1:
        return (0.45, 0.45, 0.45), 0.30
    return colormaps["gnuplot_r"](0.12 + 0.70 * colors.Normalize(2, 5)(z))[:3], 1.0


def rgba(rgb, a=1.0):
    r, g, b = (round(255 * c) for c in rgb)
    return f"rgba({r},{g},{b},{a:g})"


def gray(level, a=1.0):
    return rgba((float(level),) * 3, a)


def lx(p):  # a rate's position across the frame (0–1)
    return (np.log10(p) - np.log10(P_MIN)) / (np.log10(P_MAX) - np.log10(P_MIN))


# The control's settings step by 0.01 through the paper's measured η, so the paper's value is one.
_LO = ETA_FID - round((ETA_FID - 0.11) / 0.01) * 0.01


@fp.figure(page=PAGE, frame=FRAME, fonts=fp.dejavu(), spine=SPINE, name="streams-required",
           controls=[fp.Control("eta", _LO, _LO + 59 * 0.01, 0.01, value=ETA_FID,
                                label=r"$\lambda_{\rm SI}/\lambda_{\rm C}$")])
def figure(target, eta=ETA_FID):
    fig = go.Figure()
    # matplotlib's stacking: each band and its edge at zorder max(SIGMAS) − z, the lines at 2.
    layers = []
    for n, z in enumerate(SIGMAS):
        color, alpha = sigma_color(z)
        lo, mid, hi = (n_required(z, P, k * P) for k in (ETA_LO, eta, ETA_HI))
        g = f"s{z}"
        band = [go.Scatter(x=P, y=lo, line_width=0, hoverinfo="skip", showlegend=False, legendgroup=g),
                go.Scatter(x=P, y=hi, fill="tonexty", fillcolor=rgba(color, 0.15 * alpha), hoverinfo="skip",
                           showlegend=False, legendgroup=g,
                           line=dict(color=rgba(color, alpha), width=fp.pt(0.8), dash=fp.dash(":", 0.8)))]
        line = [go.Scatter(x=P, y=mid, legendgroup=g, showlegend=False, name=f"{z}σ",
                           line=dict(color=rgba(color, alpha), width=fp.pt(1.6)), customdata=np.c_[lo, hi],
                           hovertemplate=f"<b>{z}σ</b><br>λ<sub>C</sub> = %{{x:.2%}}<br>"
                                         f"N = %{{y:,.0f}} at λ<sub>SI</sub>/λ<sub>C</sub> = {eta:.2f}<br>"
                                         f"over [{ETA_LO:g}, {ETA_HI:g}]: %{{customdata[0]:,.0f}} – %{{customdata[1]:,.0f}}"
                                         "<extra></extra>")]
        layers += [(max(SIGMAS) - z, 2 * n, band), (2, 2 * n + 1, line)]
    for _, _, traces in sorted(layers, key=lambda t: t[:2]):
        fig.add_traces(traces)

    for name, n, level, lower in SURVEYS:
        fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y", y0=n, y1=n, layer="above",
                      line=dict(color=gray(level), width=fp.pt(0.65), dash=fp.dash("--", 0.65)))
        x = P_MIN * 1.03 if lower else P_MAX * 0.97
        fp.text(fig, name, np.log10(x), np.log10(n * 1.12), 5.5, role="bold", color=gray(level * 0.75),
                ha="left" if lower else "right", va="bottom", data=True)
    # Up arrows from past the Euclid label's right end (154.1 pt), as the script spaces them.
    x_text = 10 ** (np.log10(P_MIN) + (154.1 - FRAME[0]) / FRAME[2] * (np.log10(P_MAX) - np.log10(P_MIN)))
    for x in np.logspace(np.log10(x_text * 1.25), np.log10(P_MAX * 0.95), 5):
        fig.add_annotation(x=np.log10(x), y=np.log10(N_EUCLID * 1.65), ax=np.log10(x), ay=np.log10(N_EUCLID),
                           axref="x", ayref="y", text="", showarrow=True, arrowhead=2, arrowsize=2.2,
                           arrowwidth=fp.pt(0.45), arrowcolor=gray(0.25, 0.5), standoff=fp.pt(2), startstandoff=fp.pt(2))

    fig.add_shape(type="line", x0=LAM_C, x1=LAM_C, yref="paper", y0=0, y1=1, layer="above",
                  line=dict(color=gray(0.3), width=fp.pt(0.9), dash=fp.dash("-.", 0.9)))
    fp.stack(fig, r"$\lambda$", r"$\rm sim$", r"$\rm C$", 151.6, 13.3, 7, color=gray(0.3))

    fp.legend(fig, [fp.Entry(f"s{z}", f"${z}\\sigma$", band=rgba(c, 0.15 * a),
                             edge=dict(color=rgba(c, a), width=0.6, dash=":"), line=dict(color=rgba(c, a), width=1.1))
                    for z in SIGMAS for c, a in [sigma_color(z)]],
              at=_legend_at(), size=7, title="2 Hyp. Test", **LEGEND)

    # The band's definition (the script's AnchoredOffsetbox): λ_SI/λ_C ∈ a band from ETA_LO to ETA_HI.
    ctx = fp.context()
    (bl, bt, br, bb), (x0, y0, x1, y1) = WIDGET["box"], WIDGET["band"]
    page = dict(xref="paper", yref="paper", layer="above")
    fig.add_shape(type="rect", x0=ctx.x(bl), x1=ctx.x(br), y0=ctx.y(bb), y1=ctx.y(bt), fillcolor="rgba(255,255,255,0.7)",
                  line_width=0, **page)
    fig.add_shape(type="rect", x0=ctx.x(x0), x1=ctx.x(x1), y0=ctx.y(y1), y1=ctx.y(y0), fillcolor=gray(0.4, 0.15),
                  line_width=0, **page)
    fig.add_shape(type="line", x0=ctx.x(x0), x1=ctx.x(x1), y0=ctx.y(y0), y1=ctx.y(y0),
                  line=dict(color=gray(0.4), width=fp.pt(0.56), dash=fp.dash(":", 0.56)), **page)
    y_eta = y1 - (eta - ETA_LO) / (ETA_HI - ETA_LO) * (y1 - y0)
    fig.add_shape(type="line", x0=ctx.x(x0), x1=ctx.x(x1), y0=ctx.y(y_eta), y1=ctx.y(y_eta),
                  line=dict(color=gray(0.4), width=fp.pt(1.12)), **page)
    for y, s in ((y0, f"{ETA_HI:g}"), (y_eta, f"{eta:.2f}"), (y1, f"{ETA_LO:g}")):
        fp.text(fig, s, WIDGET["labels"], y + WIDGET["baseline"], 5.25)
    fp.frac(fig, r"$\lambda_{\rm SI}$", r"$\lambda_{\rm C}$", 47.4, 155.2, 7)
    fp.text(fig, r"$\in$", 53.3, 157.6, 7)

    major = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    minor_x = [m * 10.0**k for k in (-3, -2) for m in range(2, 10) if not np.any(np.isclose(m * 10.0**k, major))]
    minor_x = [v for v in minor_x if v <= P_MAX]
    minor_y = [m * 10.0**k for k in range(2, 6) for m in range(2, 10) if m * 10.0**k <= N_MAX]
    axis = dict(type="log", showline=True, linecolor="black", linewidth=fp.pt(SPINE), mirror="ticks", ticks="inside",
                ticklen=fp.pt(3.5), tickwidth=fp.pt(0.7), tickcolor="black", zeroline=False,
                showgrid=True, gridcolor=gray(0.5, 0.4), gridwidth=fp.pt(0.35), griddash=fp.dash(":", 0.35),
                tickfont=dict(family="fp-text", size=fp.pt(8), color="black"),
                title=dict(font=dict(family="fp-text", size=fp.pt(10), color="black")))
    minor = dict(ticks="inside", ticklen=fp.pt(2), tickwidth=fp.pt(0.5), tickcolor="black", showgrid=False)
    fig.update_xaxes(axis, range=[np.log10(P_MIN), np.log10(P_MAX)], tickvals=major,
                     ticktext=["0.1%", "0.2%", "0.5%", "1%", "2%", "5%"], minor=dict(minor, tickvals=minor_x),
                     title_text=fp.tex(r"Model rate $\lambda_{\rm C}$"), title_standoff=fp.pt(4),
                     ticklabelstandoff=round(fp.pt(-1.45)))
    fig.update_yaxes(axis, range=[np.log10(N_MIN), np.log10(N_MAX)], tickvals=[1e2, 1e3, 1e4, 1e5],
                     ticktext=[f"10<sup>{k}</sup>" for k in range(2, 6)], minor=dict(minor, tickvals=minor_y),
                     title_text="Number of Streams", title_standoff=fp.pt(4), ticklabelstandoff=round(fp.pt(2.7)))
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                      hovermode="closest" if target == "web" else False)
    return fig


# The script's legend: upper right, borderaxespad 0.5 of its 7 pt font.
LEGEND = dict(ncol=2, handlelength=1.2, handletextpad=0.4, borderpad=0.3, labelspacing=0.3, columnspacing=0.7,
              fill="rgba(255,255,255,0.9)", edgecolor=gray(0.7))


def _legend_at():
    """The legend's top left, from its width: drawn once off-figure to measure it."""
    left, _, right, _ = fp.legend(go.Figure(), [fp.Entry(f"s{z}", f"${z}\\sigma$") for z in SIGMAS], at=(0, 0), size=7,
                                  title="2 Hyp. Test", **LEGEND)
    return FRAME[0] + FRAME[2] - 3.5 - (right - left), FRAME[1] + 3.5
