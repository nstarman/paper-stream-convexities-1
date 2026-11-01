"""Compare the working tree's step_1 against HEAD's at a reduced grid.

`src/scripts/cosmology/step_1_rate_dependence.py` takes about 21 hours at full
size, so it is never re-run to verify a refactor. Instead both versions are
copied into a temporary directory, shrunk identically, run, and their outputs
compared array by array. This proves the code path is unchanged; it does not
re-derive the cached arrays the manuscript uses.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "src/scripts/cosmology/step_1_rate_dependence.py"

#: Identical shrink applied to both copies: a 4x4 potential grid (10 canonical
#: potentials in 2 chunks), one batch of orbits, and a 12x6 view grid. That is
#: ~156,000x fewer potential-orbit-view triples than the real run (78 x 5000 x
#: 7200 vs. 10 x 25 x 72), so the cost is dominated by JAX compilation rather
#: than by the integration.
#:
#: N_Q1 must equal N_Q2 (swap-symmetry assertion) and CHUNK_SIZE must divide
#: N_UNIQ = N_Q1 * (N_Q1 + 1) // 2 = 10 (checkpoint-granularity assertion).
PATCHES: tuple[tuple[str, str], ...] = (
    # _Q_KNOTS spans several lines, so match through its closing paren: a
    # single-line pattern leaves the continuation lines behind and the shrunken
    # twin will not parse.
    (
        r"(?s)^_Q_KNOTS: Final = np\.concatenate\(.*?\n\)(?:\.round\(\d+\))?$",
        "_Q_KNOTS: Final = np.linspace(0.6, 1.0, 4)",
    ),
    # CHUNK_SIZE is a multi-line `min(...)` over the divisors of N_UNIQ.
    (r"(?s)^CHUNK_SIZE: Final = min\(.*?\n\)$", "CHUNK_SIZE: Final = 5"),
    (r"^N_ORBITS: Final = .*$", "N_ORBITS: Final = 25"),
    (r"^N_PHI: Final = .*$", "N_PHI: Final = 12"),
    (r"^N_THETA: Final = .*$", "N_THETA: Final = 6"),
    # Outputs go beside each staged copy, never to src/data: that holds the
    # paper's real arrays, which this run would overwrite.
    (r"^DATA: Final = .*$", 'DATA: Final = pathlib.Path(__file__).parent / "data"'),
    (
        r"^CHECKPOINTS: Final = .*$",
        'CHECKPOINTS: Final = pathlib.Path(__file__).parent / "checkpoints"',
    ),
)


def shrink(source: str) -> str:
    """Apply every shrink patch, failing loudly if one does not match."""
    for pattern, replacement in PATCHES:
        source, n = re.subn(pattern, replacement, source, flags=re.MULTILINE)
        if n != 1:
            msg = f"shrink pattern {pattern!r} matched {n} times, expected 1"
            raise RuntimeError(msg)
    return source


def stage(where: Path, source: str) -> Path:
    """Write a shrunken copy of the script into its own directory."""
    target = where
    target.mkdir(parents=True)
    script = target / Path(SCRIPT).name
    script.write_text(shrink(source), encoding="utf-8")
    return script


#: Generous ceiling for the reduced-grid run (JAX compilation dominates it).
RUN_TIMEOUT = 1800


def run(script: Path) -> None:
    """Execute one staged copy."""
    try:
        result = subprocess.run(
            ["uv", "run", "python", str(script)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=RUN_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"{script} exceeded the {RUN_TIMEOUT}s timeout"
        raise RuntimeError(msg) from exc
    if result.returncode != 0:
        msg = f"{script} exited {result.returncode}:\n{result.stderr[-2000:]}"
        raise RuntimeError(msg)


def compare_outputs(old_dir: Path, new_dir: Path) -> list[str]:
    """Compare every .npy and .npz the two runs produced."""
    problems: list[str] = []
    old_files = {p.name for p in old_dir.iterdir() if p.suffix in {".npy", ".npz"}}
    new_files = {p.name for p in new_dir.iterdir() if p.suffix in {".npy", ".npz"}}

    if not old_files and not new_files:
        return ["no .npy or .npz files found in either run's data directory"]

    for missing in sorted(old_files - new_files):
        problems.append(f"{missing}: written by HEAD but not by the working tree")
    for extra in sorted(new_files - old_files):
        problems.append(f"{extra}: written by the working tree but not by HEAD")

    for name in sorted(old_files & new_files):
        old, new = np.load(old_dir / name), np.load(new_dir / name)
        if name.endswith(".npz"):
            old_keys, new_keys = set(old.files), set(new.files)
            if old_keys != new_keys:
                problems.append(
                    f"{name}: keys {sorted(old_keys)} -> {sorted(new_keys)}"
                )
                continue
            keys = sorted(old_keys)
        else:
            keys = [None]
        for key in keys:
            a = old[key] if key else old
            b = new[key] if key else new
            label = f"{name}[{key}]" if key else name
            if a.shape != b.shape:
                problems.append(f"{label}: shape {a.shape} -> {b.shape}")
            elif not np.array_equal(a, b, equal_nan=True):
                problems.append(f"{label}: values differ")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Run both versions at reduced size and report any difference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "rev",
        nargs="?",
        default="HEAD",
        help=(
            "git revision to compare the working tree against (default: "
            "HEAD; pass an earlier commit once the change is committed)"
        ),
    )
    args = parser.parse_args(argv)

    head = subprocess.run(
        ["git", "show", f"{args.rev}:{SCRIPT}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    working = (ROOT / SCRIPT).read_text(encoding="utf-8")

    if head == working:
        print(f"step_1 is unchanged against {args.rev}; nothing to compare")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        old_script = stage(base / "head", head)
        new_script = stage(base / "working", working)

        print("running HEAD's step_1 at reduced size ...")
        run(old_script)
        print("running the working tree's step_1 at reduced size ...")
        run(new_script)

        problems = compare_outputs(
            old_script.parent / "data",
            new_script.parent / "data",
        )

    if problems:
        print("\nstep_1 outputs changed:")
        for line in problems:
            print(f"  {line}")
        return 1
    print("\nstep_1 outputs identical at reduced size")
    return 0


if __name__ == "__main__":
    sys.exit(main())
