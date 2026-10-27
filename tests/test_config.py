"""Behavior tests for configuration files."""

from pathlib import Path

import pytest

from smpy import Config

@pytest.mark.parametrize("text, iterations", [
    ("methods:\n  ks_plus:\n    inpainting_iterations: 5\n", 5),
    ("", None),
    ("plotting:\n", None),
], ids=["nested override", "empty file", "empty section"])
def test_yaml_merges_over_defaults(tmp_path, text, iterations):
    """A YAML file changes only the settings it names."""
    path = tmp_path / "run.yaml"
    path.write_text(text)
    expected = Config().to_dict()
    if iterations is not None:
        expected["methods"]["ks_plus"]["inpainting_iterations"] = iterations
    assert Config.from_file(path).to_dict() == expected

def test_relative_paths_follow_the_file(tmp_path):
    """Relative paths resolve from the YAML file and survive a move."""
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "run.yaml").write_text(
        "general:\n  input_path: ../data/cat.fits\n  output_directory: out\n")
    config = Config.from_file(tmp_path / "a" / "run.yaml")
    resolved = config.resolved()["general"]
    assert Path(resolved["input_path"]) == (tmp_path / "data" / "cat.fits").resolve()

    (tmp_path / "b" / "c").mkdir(parents=True)
    config.save(tmp_path / "b" / "c" / "copy.yaml")
    moved = Config.from_file(tmp_path / "b" / "c" / "copy.yaml").resolved()["general"]
    assert moved["input_path"] == resolved["input_path"]
    assert moved["output_directory"] == resolved["output_directory"]
