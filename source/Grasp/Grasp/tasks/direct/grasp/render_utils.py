from __future__ import annotations

import random
from pathlib import Path

import yaml

from .utils import COLORS_DICT


def load_object_display_names(asset_root: str, object_name_list: str, num_envs: int) -> list[str]:
    names = yaml.safe_load((Path(asset_root) / object_name_list).read_text())
    if not isinstance(names, list):
        raise ValueError(f"object_name_list must contain a YAML list: {object_name_list}")
    return [str(names[index % len(names)]) for index in range(num_envs)]


def build_instruction(
    use_advanced_instruction: bool,
    instruction_template: str,
    object_name: str,
    color_name: str,
) -> str:
    if not use_advanced_instruction:
        return instruction_template
    return instruction_template.replace("{COLOR}", color_name).replace("{OBJ}", object_name)


def sample_color_name(color_choices: tuple[str, ...]) -> str:
    if len(color_choices) == 0:
        raise ValueError("render.randomization_params.object_color_choices must not be empty")
    return random.choice(tuple(color_choices))


def color_rgb(color_name: str) -> tuple[float, float, float]:
    if color_name not in COLORS_DICT:
        raise ValueError(f"Unknown color name: {color_name}")
    rgb = COLORS_DICT[color_name]
    return float(rgb[0]), float(rgb[1]), float(rgb[2])