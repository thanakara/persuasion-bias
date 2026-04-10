from pathlib import Path

from persuasion_bias.utils import path
from persuasion_bias.utils.path import ROOT_DIR, CONFIG_DIR


def test_root_dir_is_path_instance():
    assert isinstance(ROOT_DIR, Path)


def test_config_dir_is_path_instance():
    assert isinstance(CONFIG_DIR, Path)


def test_root_dir_is_four_levels_up():
    """ROOT_DIR should be 4 parents above path.py"""
    expected = Path(path.__file__).parents[3]
    assert ROOT_DIR == expected


def test_config_dir_is_relative_to_root():
    assert CONFIG_DIR == ROOT_DIR / "conf"


def test_root_dir_is_absolute():
    assert ROOT_DIR.is_absolute()


def test_config_dir_is_absolute():
    assert CONFIG_DIR.is_absolute()
