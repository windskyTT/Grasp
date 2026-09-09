from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path("/home/windsky/project/Grasp")
RSC_ROOT = PROJECT_ROOT / "rsc"
TRAIN_ROOT = RSC_ROOT / "new_training_set"
EVAL_ROOT = RSC_ROOT / "shapenet-30obj"

TRAIN_OBJECT_NAMES = tuple(sorted(p.name for p in TRAIN_ROOT.iterdir() if p.is_dir()))
EVAL_OBJECT_NAMES = tuple(sorted(p.name for p in EVAL_ROOT.iterdir() if p.is_dir()))

# This is the exact source train.py list construction: all directories, then
# the nine additional entries, followed by two repetitions of that list.
SOURCE_TRAIN_OBJECT_ORDER = (
    *TRAIN_OBJECT_NAMES,
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
TRAIN_OBJECT_ORDER = tuple(
    item for _ in range(2) for item in SOURCE_TRAIN_OBJECT_ORDER
)

def object_urdf_path(dataset_root: Path, object_name: str) -> str:
    return str(dataset_root / object_name / f"{object_name}.urdf")

def top_mesh_path(dataset_root: Path, object_name: str) -> str:
    return str(dataset_root / object_name / "top_watertight_tiny.obj")

def bottom_mesh_path(dataset_root: Path, object_name: str) -> str:
    return str(dataset_root / object_name / "bottom_watertight_tiny.obj")

def lowest_point_path(dataset_root: Path, object_name: str) -> str:
    return str(dataset_root / object_name / "lowest_point_new.txt")

TRAIN_URDF_PATHS = tuple(object_urdf_path(TRAIN_ROOT, name) for name in TRAIN_OBJECT_ORDER)
TRAIN_TOP_MESH_PATHS = tuple(top_mesh_path(TRAIN_ROOT, name) for name in TRAIN_OBJECT_ORDER)
TRAIN_BOTTOM_MESH_PATHS = tuple(bottom_mesh_path(TRAIN_ROOT, name) for name in TRAIN_OBJECT_ORDER)
TRAIN_LOWEST_POINT_PATHS = tuple(lowest_point_path(TRAIN_ROOT, name) for name in TRAIN_OBJECT_ORDER)

EVAL_URDF_PATHS = tuple(object_urdf_path(EVAL_ROOT, name) for name in EVAL_OBJECT_NAMES)
EVAL_TOP_MESH_PATHS = tuple(top_mesh_path(EVAL_ROOT, name) for name in EVAL_OBJECT_NAMES)
EVAL_BOTTOM_MESH_PATHS = tuple(bottom_mesh_path(EVAL_ROOT, name) for name in EVAL_OBJECT_NAMES)
EVAL_LOWEST_POINT_PATHS = tuple(lowest_point_path(EVAL_ROOT, name) for name in EVAL_OBJECT_NAMES)

__all__ = [
    "PROJECT_ROOT", "RSC_ROOT", "TRAIN_ROOT", "EVAL_ROOT",
    "TRAIN_OBJECT_NAMES", "EVAL_OBJECT_NAMES", "SOURCE_TRAIN_OBJECT_ORDER",
    "TRAIN_OBJECT_ORDER", "TRAIN_URDF_PATHS", "TRAIN_TOP_MESH_PATHS",
    "TRAIN_BOTTOM_MESH_PATHS", "TRAIN_LOWEST_POINT_PATHS", "EVAL_URDF_PATHS",
    "EVAL_TOP_MESH_PATHS", "EVAL_BOTTOM_MESH_PATHS", "EVAL_LOWEST_POINT_PATHS",
]
