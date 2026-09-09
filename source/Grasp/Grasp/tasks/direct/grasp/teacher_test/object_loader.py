from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg

def make_object_spawner(urdf_paths: tuple[str, ...]) -> MultiAssetSpawnerCfg:
    assets_cfg = []
    for path in urdf_paths:
        assets_cfg.append(
            sim_utils.UrdfFileCfg(
                asset_path=str(Path(path)),
                fix_base=False,
                merge_fixed_joints=True,
                joint_drive=None,
                collider_type="convex_hull",
                self_collision=False,
                activate_contact_sensors=True,
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

__all__ = ["make_object_spawner"]
