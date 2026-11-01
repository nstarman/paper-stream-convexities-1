"""Unit tests for the step_1 twin-run comparison primitives.

These exercise only the pure helpers (`shrink`, `compare_outputs`); they never
invoke a subprocess, never run step_1 itself, and never touch `src/data/`.
"""

from pathlib import Path

import numpy as np
import pytest

from scripts.check_step1 import compare_outputs, shrink

#: A representative snippet carrying exactly what `PATCHES` targets, in the same
#: form as the real script -- including the multi-line `_Q_KNOTS` assignment,
#: which a single-line pattern would shrink into unparseable source.
SOURCE = """\
_Q_KNOTS: Final = np.concatenate(
    [np.linspace(0.60, 0.87, 10), np.linspace(0.90, 1.00, 6)],
)
N_Q1: Final = len(_Q_KNOTS)
CHUNK_SIZE: Final = min(
    (d for d in range(8, 40) if N_UNIQ % d == 0),
    key=lambda d: abs(d - 17),
)
DATA: Final = paths.DATA / "cosmology" / "rates"  # the rule's output
CHECKPOINTS: Final = paths.DATA / "cosmology" / "rates_checkpoints"
N_ORBITS: Final = int(_CFG["n_orbits"]["value"])
N_PHI: Final = int(_CFG["n_phi"]["value"])  # azimuth bins
N_THETA: Final = int(_CFG["n_theta"]["value"])  # colatitude bins
"""


def _dirs(tmp_path: Path) -> tuple[Path, Path]:
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    return old_dir, new_dir


def test_shrink_applies_every_patch() -> None:
    shrunk = shrink(SOURCE)

    assert "_Q_KNOTS: Final = np.linspace(0.6, 1.0, 4)" in shrunk
    assert "np.concatenate" not in shrunk  # the continuation lines go too
    assert "CHUNK_SIZE: Final = 5" in shrunk
    assert "N_ORBITS: Final = 25" in shrunk
    assert "N_PHI: Final = 12" in shrunk
    assert "N_THETA: Final = 6" in shrunk
    assert "paths.DATA" not in shrunk  # outputs never reach the real src/data


def test_shrink_raises_when_a_pattern_matches_zero_times() -> None:
    missing_constant = SOURCE.replace("N_PHI: Final = int(_CFG[\"n_phi\"][\"value\"])  # azimuth bins\n", "")

    with pytest.raises(RuntimeError, match="matched 0 times"):
        shrink(missing_constant)


def test_shrink_raises_when_a_pattern_matches_twice() -> None:
    duplicated_constant = SOURCE.replace(
        "N_ORBITS: Final = int(_CFG[\"n_orbits\"][\"value\"])\n",
        "N_ORBITS: Final = int(_CFG[\"n_orbits\"][\"value\"])\n"
        "N_ORBITS: Final = 7  # duplicate\n",
    )

    with pytest.raises(RuntimeError, match="matched 2 times"):
        shrink(duplicated_constant)


def test_compare_outputs_identical_npy_and_npz(tmp_path: Path) -> None:
    old_dir, new_dir = _dirs(tmp_path)
    arr = np.arange(6.0).reshape(2, 3)
    np.save(old_dir / "a.npy", arr)
    np.save(new_dir / "a.npy", arr.copy())
    np.savez(old_dir / "b.npz", x=arr, y=arr * 2)
    np.savez(new_dir / "b.npz", x=arr.copy(), y=arr.copy() * 2)

    assert compare_outputs(old_dir, new_dir) == []


def test_compare_outputs_reports_value_difference(tmp_path: Path) -> None:
    old_dir, new_dir = _dirs(tmp_path)
    np.save(old_dir / "a.npy", np.array([1.0, 2.0]))
    np.save(new_dir / "a.npy", np.array([1.0, 3.0]))

    problems = compare_outputs(old_dir, new_dir)

    assert len(problems) == 1
    assert "a.npy" in problems[0]
    assert "values differ" in problems[0]


def test_compare_outputs_reports_shape_mismatch(tmp_path: Path) -> None:
    old_dir, new_dir = _dirs(tmp_path)
    np.save(old_dir / "a.npy", np.zeros((2, 3)))
    np.save(new_dir / "a.npy", np.zeros((3, 2)))

    problems = compare_outputs(old_dir, new_dir)

    assert len(problems) == 1
    assert "shape" in problems[0]


def test_compare_outputs_reports_file_present_in_one_dir_only(
    tmp_path: Path,
) -> None:
    old_dir, new_dir = _dirs(tmp_path)
    np.save(old_dir / "only_old.npy", np.array([1]))
    np.save(new_dir / "only_new.npy", np.array([1]))

    problems = compare_outputs(old_dir, new_dir)

    assert any(
        "only_old.npy" in p and "not by the working tree" in p for p in problems
    )
    assert any("only_new.npy" in p and "not by HEAD" in p for p in problems)


def test_compare_outputs_reports_differing_npz_keys(tmp_path: Path) -> None:
    old_dir, new_dir = _dirs(tmp_path)
    np.savez(old_dir / "b.npz", x=np.array([1]))
    np.savez(new_dir / "b.npz", y=np.array([1]))

    # Must report a problem, not raise KeyError.
    problems = compare_outputs(old_dir, new_dir)

    assert len(problems) == 1
    assert "b.npz" in problems[0]
    assert "keys" in problems[0]


def test_compare_outputs_treats_matching_nan_as_equal(tmp_path: Path) -> None:
    old_dir, new_dir = _dirs(tmp_path)
    arr = np.array([1.0, np.nan, 3.0])
    np.save(old_dir / "a.npy", arr)
    np.save(new_dir / "a.npy", arr.copy())

    assert compare_outputs(old_dir, new_dir) == []


def test_compare_outputs_reports_problem_when_nothing_compared(
    tmp_path: Path,
) -> None:
    old_dir, new_dir = _dirs(tmp_path)

    problems = compare_outputs(old_dir, new_dir)

    assert len(problems) == 1
    assert "no .npy or .npz files" in problems[0]
