from __future__ import annotations

from pathlib import Path


NEW_TRAINING_SET_ROOT = Path(
    "/home/windsky/project/Grasp/assets/new_training_set"
)

TEACHER_OBJECT_NAMES: tuple[str, ...] = (
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "025_mug",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "051_large_clamp",
    "052_extra_large_clamp",
    "061_foam_brick",
    "big_tape",
    "blue_pitcher",
    "brush_functional",
    "car_down",
    "cracker_box_oriented",
    "fan_small_head",
    "gun_functional",
    "hammer",
    "loopy_head_side",
    "mouse",
    "off_water_body",
    "small_block",
    "small_tape",
    "solder_iron_head",
    "suger_box_oriented",
    "wood_block_oriented",
)

TEACHER_WEIGHTED_OBJECT_NAMES: tuple[str, ...] = (
    *TEACHER_OBJECT_NAMES,
    "037_scissors",
    "037_scissors",
    "off_water_body",
    "off_water_body",
    "019_pitcher_base",
    "011_banana",
    "mouse",
    "hammer",
    "small_block",
)

TEACHER_OBJECT_TOP_USD_PATHS: tuple[str, ...] = tuple(
    str(NEW_TRAINING_SET_ROOT / name / f"{name}_top.usd")
    for name in TEACHER_OBJECT_NAMES
)
TEACHER_OBJECT_BOTTOM_USD_PATHS: tuple[str, ...] = tuple(
    str(NEW_TRAINING_SET_ROOT / name / f"{name}_bottom.usd")
    for name in TEACHER_OBJECT_NAMES
)

TEACHER_WEIGHTED_OBJECT_TOP_USD_PATHS: tuple[str, ...] = tuple(
    str(NEW_TRAINING_SET_ROOT / name / f"{name}_top.usd")
    for name in TEACHER_WEIGHTED_OBJECT_NAMES
)
TEACHER_WEIGHTED_OBJECT_BOTTOM_USD_PATHS: tuple[str, ...] = tuple(
    str(NEW_TRAINING_SET_ROOT / name / f"{name}_bottom.usd")
    for name in TEACHER_WEIGHTED_OBJECT_NAMES
)

TEACHER_OBJECT_INDEX: dict[str, int] = {
    object_name: object_index
    for object_index, object_name in enumerate(
        TEACHER_OBJECT_NAMES
    )
}

TEACHER_WEIGHTED_OBJECT_INDICES: tuple[int, ...] = tuple(
    TEACHER_OBJECT_INDEX[object_name]
    for object_name in TEACHER_WEIGHTED_OBJECT_NAMES
)

TEACHER_REPEAT_PER_OBJECT = 2
TEACHER_FULL_ENV_COUNT = (
    len(TEACHER_WEIGHTED_OBJECT_NAMES)
    * TEACHER_REPEAT_PER_OBJECT
)

__all__ = [
    "NEW_TRAINING_SET_ROOT",
    "TEACHER_FULL_ENV_COUNT",
    "TEACHER_OBJECT_BOTTOM_USD_PATHS",
    "TEACHER_OBJECT_INDEX",
    "TEACHER_OBJECT_NAMES",
    "TEACHER_OBJECT_TOP_USD_PATHS",
    "TEACHER_REPEAT_PER_OBJECT",
    "TEACHER_WEIGHTED_OBJECT_BOTTOM_USD_PATHS",
    "TEACHER_WEIGHTED_OBJECT_INDICES",
    "TEACHER_WEIGHTED_OBJECT_NAMES",
    "TEACHER_WEIGHTED_OBJECT_TOP_USD_PATHS",
]
