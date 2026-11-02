from functools import lru_cache
from pathlib import Path
import re

import showyourwork.config as _syw_config

# Work around a ShowYourWork performance bug: after the `syw__dag`
# checkpoint it walks every output's upstream dependencies, one `git rev-parse` per
# edge, which with ~80 \variable{} outputs looks like a hang at "Updating checkpoint
# dependencies". Memoizing the per-edge helper removes the subprocesses.
_syw_config.expand_dependency_directories = lru_cache(maxsize=None)(
    _syw_config.expand_dependency_directories
)

PY = "uv run --locked python"
PKG = "src/stream_convexity_paper/"
SETTINGS = ["variables.yml", "pyproject.toml", "uv.lock", "environment.yml"]
# The Zenodo-cached rules leave out the lock and pyproject: a dependency bump would
# otherwise change their cache keys and send CI looking for a cache that is not there.
# After a bump that could change the numbers (numpy, jax, ...), rebuild them by hand.
CACHED_SETTINGS = ["variables.yml", "environment.yml"]


def pkg(*modules):
    """The package modules a script imports (plus __init__ and paths, always)."""
    return [PKG + m + ".py" for m in ("__init__", "paths", *modules)]


# Every \variable{} the manuscript cites (comments stripped), as SYW discovers them.
_ms = "\n".join(
    re.sub(r"(?<!\\)%.*", "", line)
    for line in Path("src/tex/ms.tex").read_text(encoding="utf-8").splitlines()
)
VARIABLE_OUTPUTS = sorted({"src/tex/" + p for p in re.findall(r"\\variable\{([^}]*)\}", _ms)})
VALUES = "src/data/values/{}.values.json".format


# ── Cosmology ────────────────────────────────────────────────────────────────
# The potential-grid integration: ~21 h. Its one output is a folder (SYW tars it)
# cached on Zenodo; checkpoints go to src/data/cosmology/rates_checkpoints, outside it.
rule rate_dependence:
    input:
        script="src/scripts/cosmology/step_1_rate_dependence.py",
        code=pkg("curvature", "data", "variables"),
        settings=CACHED_SETTINGS,
    output:
        directory("src/data/cosmology/rates"),
    conda:
        "environment.yml"
    cache:
        True
    shell:
        # The published grid starts at q = 0.45 (21 x 21), not the script's default 0.60.
        f"{PY} {{input.script}} --ca-floor=0.45"


# `local_cache: [rate_dependence]` in showyourwork.yml: copy the results from the
# gitignored .local_cache/ instead, so a code edit never starts the 21-h run or
# re-uploads to Zenodo. A plain copy with no code inputs, on purpose.
if "rate_dependence" in set(config.get("local_cache") or []):

    rule rate_dependence_from_local_cache:
        input:
            ".local_cache/rate_dependence",
        output:
            directory("src/data/cosmology/rates"),
        shell:
            "rm -rf {output} && cp -R {input} {output}"

    ruleorder: rate_dependence_from_local_cache > rate_dependence


rule triaxiality:
    input:
        script="src/scripts/cosmology/step_2_plot_triaxiality.py",
        rates="src/data/cosmology/rates",
        auriga="src/static/cosmology/literature/prada2019_auriga_mhd.csv",
        code=pkg("values"),
        settings=SETTINGS,
    output:
        simple="src/tex/figures/cosmology/convexity_rate_grid.pdf",
        full="src/tex/figures/cosmology/convexity_rate_grid_full.pdf",
        values=VALUES("step_2_plot_triaxiality"),
    conda:
        "environment.yml"
    shell:
        f"{PY} {{input.script}} && {PY} {{input.script}} --simple"


rule streams_required:
    input:
        script="src/scripts/cosmology/streams_required_pcdm.py",
        rates=VALUES("step_2_plot_triaxiality"),
        code=pkg("style", "values"),
        settings=SETTINGS,
    output:
        "src/tex/figures/cosmology/streams_required.pdf",
        "src/tex/figures/cosmology/likelihood_ratio.pdf",
        VALUES("streams_required_pcdm"),
    conda:
        "environment.yml"
    shell:
        f"{PY} {{input.script}}"


# ── GD-1 ─────────────────────────────────────────────────────────────────────
rule gd1:
    input:
        script="src/scripts/GD1/plot_gd1_convexities.py",
        track="src/static/GD1/track.st.GD-1.pricewhelan2018.ecsv",
        rates=VALUES("step_2_plot_triaxiality"),
        code=pkg("curvature", "data", "track", "utils", "values", "variables"),
        settings=SETTINGS,
    output:
        "src/tex/figures/GD1/gd1_convexities.pdf",
        VALUES("plot_gd1_convexities"),
    conda:
        "environment.yml"
    shell:
        f"{PY} {{input.script}}"


# ── Graphic ──────────────────────────────────────────────────────────────────
# Needs Inkscape and pdflatex, which CI does not have: cached on Zenodo, so CI
# restores it instead of running it. A cached output is a symlink into
# .showyourwork/cache; in src/tex/figures it would reach the GitHub Actions article
# cache dangling and crash `showyourwork cache restore`. So it lands in src/data and
# graphic_figure copies the real file into the figures folder.
rule graphic:
    input:
        script="src/scripts/graphic/graphic.py",
        svg="src/static/graphic/galaxy.svg",
        pdf="src/static/graphic/galaxy.pdf",
        code=pkg(),
        settings=CACHED_SETTINGS,
    output:
        "src/data/graphic/graphic.pdf",
    conda:
        "environment.yml"
    cache:
        True
    shell:
        f"{PY} {{input.script}}"


rule graphic_figure:
    input:
        "src/data/graphic/graphic.pdf",
    output:
        "src/tex/figures/graphic/graphic.pdf",
    shell:
        "cp {input} {output}"


# ── Streams ──────────────────────────────────────────────────────────────────
rule numerical_confirmation:
    input:
        script="src/scripts/streams/plot_mollweide.py",
        helpers=[
            "src/scripts/streams/ridgeline_data.py",
            "src/scripts/streams/sky_geometry.py",
            "src/scripts/streams/sky_style.py",
            "src/scripts/streams/viz_ridgeline.py",
        ],
        data=[
            "src/static/streams/nbody_q15_ridgeline",
            "src/static/streams/triaxial08507_rs14_65Gyr_init2_Mprog1e6_ridgeline",
        ],
        code=pkg("curvature", "orientation_inset", "track"),
        settings=SETTINGS,
    output:
        "src/tex/figures/streams/numerical_confirmation.pdf",
    conda:
        "environment.yml"
    shell:
        f"{PY} {{input.script}}"


# ── Triaxial ─────────────────────────────────────────────────────────────────
rule triaxial_orbits:
    input:
        script="src/scripts/triaxial/step_1_compute_orbits.py",
        code=pkg("curvature", "data", "values", "variables"),
        settings=SETTINGS,
    output:
        "src/data/triaxial/triaxial_data.npz",
        VALUES("step_1_compute_orbits"),
    conda:
        "environment.yml"
    shell:
        f"{PY} {{input.script}}"


rule triaxial_diagnostic:
    input:
        script="src/scripts/triaxial/step_2_plot_mollweide.py",
        data="src/data/triaxial/triaxial_data.npz",
        code=pkg("orientation_inset", "values"),
        settings=SETTINGS,
    output:
        "src/tex/figures/triaxial/triaxial_diagnostic.pdf",
        VALUES("step_2_plot_mollweide"),
    conda:
        "environment.yml"
    shell:
        f"{PY} {{input.script}}"


# ── Numbers quoted in the text ───────────────────────────────────────────────
rule variables:
    input:
        script="src/scripts/make_variables.py",
        values=[
            VALUES(s)
            for s in (
                "step_1_compute_orbits",
                "step_2_plot_mollweide",
                "step_2_plot_triaxiality",
                "streams_required_pcdm",
                "plot_gd1_convexities",
            )
        ],
        code=pkg("variables"),
        settings=SETTINGS,
    output:
        VARIABLE_OUTPUTS,
    conda:
        "environment.yml"
    shell:
        f"{PY} {{input.script}}"

# The paper's docs site (syw-site).
try:
    import syw_site
except ImportError:
    syw_site = None
    print("syw-site is not installed in showyourwork's environment: the docs rules are off.")
if syw_site is not None:
    include: syw_site.rules_path()
