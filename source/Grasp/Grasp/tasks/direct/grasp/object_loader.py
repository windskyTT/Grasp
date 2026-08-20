from __future__ import annotations

from pathlib import Path

import yaml
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg


def load_object_urdf_file_list(asset_root: str, multi_object_list: str) -> list[str]:
    list_path = Path(asset_root) / multi_object_list
    object_dir = multi_object_list.split("/")[0]
    names = sorted(yaml.safe_load(list_path.read_text()))
    return [f"{object_dir}/urdf/{name}" for name in names]


def object_urdf_to_generated_usd(object_file: str) -> str:
    object_dir, _, urdf_name = object_file.split("/")
    usd_name = Path(urdf_name).stem.replace("-", "_") + "_rigid_mesh_v1.usd"
    return f"{object_dir}/generated/mesh_usd/{usd_name}"


def load_multi_object_urdf_cfgs(
    asset_root: str,
    multi_object_list: str,
    use_object_vhacd: bool,
    object_friction: float,
) -> MultiAssetSpawnerCfg:
    list_path = Path(asset_root) / multi_object_list
    object_files = load_object_urdf_file_list(asset_root, multi_object_list)
    collider_type = "convex_decomposition" if use_object_vhacd else "convex_hull"
    assets_cfg = []
    for object_file in object_files:
        assets_cfg.append(
            sim_utils.UrdfFileCfg(
                asset_path=str(Path(asset_root) / object_file),
                fix_base=False,
                merge_fixed_joints=True,
                joint_drive=None,
                collider_type=collider_type,
                self_collision=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    max_depenetration_velocity=1000.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002,
                    rest_offset=0.0,
                ),
                semantic_tags=[("class", "object")],
            )
        )
    return MultiAssetSpawnerCfg(assets_cfg=assets_cfg, random_choice=False)

def load_multi_object_usd_cfgs(
    asset_root: str,
    multi_object_list: str,
) -> MultiAssetSpawnerCfg:
    object_files = load_object_urdf_file_list(asset_root, multi_object_list)
    assets_cfg = []
    for object_file in object_files:
        usd_file = object_urdf_to_generated_usd(object_file)
        assets_cfg.append(
            sim_utils.UsdFileCfg(
                usd_path=str(Path(asset_root) / usd_file),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    max_depenetration_velocity=1000.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002,
                    rest_offset=0.0,
                ),
                semantic_tags=[("class", "object")],
            )
        )
    return MultiAssetSpawnerCfg(assets_cfg=assets_cfg, random_choice=False)