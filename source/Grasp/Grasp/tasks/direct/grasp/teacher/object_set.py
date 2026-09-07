"""Teacher 训练对象集合配置。

本文件只负责：
1. 定义唯一对象名称。
2. 定义 weighted object sampling 顺序。
3. 生成 top / bottom USD 文件路径。
4. 建立 weighted entry -> unique object index 的映射。
5. 计算默认并行环境数量。

GPU 说明：
    这里保存的是 Python 字符串、Path、tuple、dict 等静态配置，
    不属于 RL 数值训练数据。

    因此本文件不创建 torch.Tensor，也不需要 CUDA 化。
    真正参与训练的 object index / point cloud / normal / pose 会在
    affordance_data.py 和 env.py 中直接创建为 CUDA Tensor。

当前数量关系：
    35 个 unique objects
        ↓
    + 9 个额外 weighted entries
        ↓
    44 个 weighted entries
        ↓
    每个 entry 重复 2 个环境
        ↓
    默认 88 个并行环境
"""

from __future__ import annotations

from pathlib import Path


# =============================================================================
# 数据集根目录
# =============================================================================
NEW_TRAINING_SET_ROOT = Path(
    "/home/windsky/project/Grasp/assets/new_training_set"
)


# =============================================================================
# 1. 唯一 Teacher 对象
# =============================================================================
# 这里每个名字只表示一种唯一几何对象。
#
# 注意：
#     "suger_box_oriented" 保留当前数据集中的原始目录名称，
#     不在这里擅自改成 "sugar_box_oriented"，否则会改变资产路径。
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

TEACHER_UNIQUE_OBJECT_COUNT = len(
    TEACHER_OBJECT_NAMES
)


# =============================================================================
# 2. Weighted sampling
# =============================================================================
# RobustDexGrasp Teacher 对部分难抓或希望提高出现频率的对象重复放入
# sampling list。
#
# 每出现一次，就额外增加一个 weighted entry。
#
# 当前额外权重：
#     scissors       +2
#     off_water_body +2
#     pitcher_base   +1
#     banana         +1
#     mouse          +1
#     hammer         +1
#     small_block    +1
#
# 总计 9 个额外 entries。
TEACHER_WEIGHT_EXTRA_NAMES: tuple[str, ...] = (
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

TEACHER_WEIGHTED_OBJECT_NAMES: tuple[str, ...] = (
    *TEACHER_OBJECT_NAMES,
    *TEACHER_WEIGHT_EXTRA_NAMES,
)

TEACHER_WEIGHTED_OBJECT_COUNT = len(
    TEACHER_WEIGHTED_OBJECT_NAMES
)


# =============================================================================
# 3. USD 路径
# =============================================================================
def _make_object_usd_paths(
    object_names: tuple[str, ...],
    part_name: str,
) -> tuple[str, ...]:
    """根据对象顺序生成 top / bottom USD 路径。

    这里仅生成静态文件路径字符串，不读取 mesh，也不生成训练数值。
    """

    return tuple(
        str(
            NEW_TRAINING_SET_ROOT
            / object_name
            / f"{object_name}_{part_name}.usd"
        )
        for object_name in object_names
    )


# 每种 unique object 各一份路径。
TEACHER_OBJECT_TOP_USD_PATHS: tuple[str, ...] = (
    _make_object_usd_paths(
        TEACHER_OBJECT_NAMES,
        "top",
    )
)

TEACHER_OBJECT_BOTTOM_USD_PATHS: tuple[str, ...] = (
    _make_object_usd_paths(
        TEACHER_OBJECT_NAMES,
        "bottom",
    )
)

# weighted list 保留重复对象。
#
# object_pair_spawner.py 会按这个顺序把不同 object pair 分配给各环境，
# 因此这里的重复条目直接决定某些对象出现得更频繁。
TEACHER_WEIGHTED_OBJECT_TOP_USD_PATHS: tuple[str, ...] = (
    _make_object_usd_paths(
        TEACHER_WEIGHTED_OBJECT_NAMES,
        "top",
    )
)

TEACHER_WEIGHTED_OBJECT_BOTTOM_USD_PATHS: tuple[str, ...] = (
    _make_object_usd_paths(
        TEACHER_WEIGHTED_OBJECT_NAMES,
        "bottom",
    )
)


# =============================================================================
# 4. unique object index
# =============================================================================
# 例如：
#     "002_master_chef_can" -> 0
#     "003_cracker_box"     -> 1
#     ...
#
# affordance_data.py 中真正用于 batch gather 的 env_object_indices
# 会根据这个映射直接生成在 CUDA 上。
TEACHER_OBJECT_INDEX: dict[str, int] = {
    object_name: object_index
    for object_index, object_name in enumerate(
        TEACHER_OBJECT_NAMES
    )
}


# =============================================================================
# 5. weighted entry -> unique object index
# =============================================================================
# 例如 weighted list 中：
#
#     "037_scissors"
#     "037_scissors"
#
# 会映射成：
#
#     scissors_unique_index
#     scissors_unique_index
#
# 因此不会为重复权重再加载一份独立 mesh；
# affordance_data.py 可以复用同一个 unique object geometry。
TEACHER_WEIGHTED_OBJECT_INDICES: tuple[int, ...] = tuple(
    TEACHER_OBJECT_INDEX[object_name]
    for object_name in TEACHER_WEIGHTED_OBJECT_NAMES
)


# =============================================================================
# 6. 默认并行环境数量
# =============================================================================
# 每个 weighted entry 使用 2 个并行环境。
TEACHER_REPEAT_PER_OBJECT = 2

# 44 weighted entries * 2 = 88 environments。
TEACHER_FULL_ENV_COUNT = (
    TEACHER_WEIGHTED_OBJECT_COUNT
    * TEACHER_REPEAT_PER_OBJECT
)


__all__ = [
    "NEW_TRAINING_SET_ROOT",
    "TEACHER_OBJECT_NAMES",
    "TEACHER_UNIQUE_OBJECT_COUNT",
    "TEACHER_WEIGHT_EXTRA_NAMES",
    "TEACHER_WEIGHTED_OBJECT_NAMES",
    "TEACHER_WEIGHTED_OBJECT_COUNT",
    "TEACHER_OBJECT_TOP_USD_PATHS",
    "TEACHER_OBJECT_BOTTOM_USD_PATHS",
    "TEACHER_WEIGHTED_OBJECT_TOP_USD_PATHS",
    "TEACHER_WEIGHTED_OBJECT_BOTTOM_USD_PATHS",
    "TEACHER_OBJECT_INDEX",
    "TEACHER_WEIGHTED_OBJECT_INDICES",
    "TEACHER_REPEAT_PER_OBJECT",
    "TEACHER_FULL_ENV_COUNT",
]
