"""Unit tests for the derived-values writer."""

import json
from pathlib import Path

import pytest

from stream_convexity_paper import paths
from stream_convexity_paper.values import write_values


@pytest.fixture(autouse=True)
def values_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write to a temporary directory, never to the real src/data/values."""
    monkeypatch.setattr(paths, "VALUES", tmp_path / "values")
    return tmp_path / "values"


def test_writes_to_the_values_dir_with_the_values_suffix(
    tmp_path: Path, values_dir: Path
) -> None:
    script = tmp_path / "plot_something.py"

    path = write_values(script, {"a": 1})

    assert path == values_dir / "plot_something.values.json"
    assert path.exists()


def test_content_is_indented_json_with_a_trailing_newline(tmp_path: Path) -> None:
    script = tmp_path / "plot_something.py"

    path = write_values(script, {"theta_deg": 117.29, "convex_frac": 1.0})

    text = path.read_text(encoding="utf-8")
    assert text == json.dumps({"theta_deg": 117.29, "convex_frac": 1.0}, indent=2) + "\n"
    assert text.endswith("}\n")


def test_key_order_is_preserved(tmp_path: Path) -> None:
    script = tmp_path / "plot_something.py"

    path = write_values(script, {"z": 1, "a": 2, "m": 3})

    assert list(json.loads(path.read_text(encoding="utf-8"))) == ["z", "a", "m"]


def test_reports_where_it_wrote(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    script = tmp_path / "plot_something.py"

    write_values(script, {"a": 1})

    out = capsys.readouterr().out
    assert "plot_something.values.json" in out
    assert "a build propagates it" in out
