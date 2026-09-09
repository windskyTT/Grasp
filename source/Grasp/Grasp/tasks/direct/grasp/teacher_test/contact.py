from __future__ import annotations

import torch

from isaaclab.sensors import ContactSensorCfg

from .robot_cfg import ARM_CONTACT_BODY_NAMES, HAND_CONTACT_BODY_NAMES

FILTER_PATHS = (
    "{ENV_REGEX_NS}/Object/top",
    "{ENV_REGEX_NS}/Object/bottom",
    "{ENV_REGEX_NS}/Table",
)

def make_contact_sensor_cfg(body_name: str, filters: tuple[str, ...] = FILTER_PATHS) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/{body_name}",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
        force_threshold=0.001,
        filter_prim_paths_expr=list(filters),
    )

def make_contact_sensor_cfgs() -> dict[str, ContactSensorCfg]:
    return {
        f"hand_{name}": make_contact_sensor_cfg(name)
        for name in HAND_CONTACT_BODY_NAMES
    } | {
        f"arm_{name}": make_contact_sensor_cfg(name)
        for name in ARM_CONTACT_BODY_NAMES
    }

def make_self_collision_sensor_cfgs() -> dict[str, ContactSensorCfg]:
    return {
        f"self_{name}": make_contact_sensor_cfg(name, filters=())
        for name in ARM_CONTACT_BODY_NAMES
    }

def collect_contact_data(env):
    hand_forces = []
    arm_forces = []
    self_forces = []
    for name in HAND_CONTACT_BODY_NAMES:
        hand_forces.append(env.contact_sensors[f"hand_{name}"].data.force_matrix_w[:, 0])
    for name in ARM_CONTACT_BODY_NAMES:
        arm_forces.append(env.contact_sensors[f"arm_{name}"].data.force_matrix_w[:, 0])
        self_forces.append(env.contact_sensors[f"self_{name}"].data.net_forces_w[:, 0])
    hand_forces = torch.stack(hand_forces, dim=1) * env.cfg.sim.dt
    arm_forces = torch.stack(arm_forces, dim=1) * env.cfg.sim.dt
    self_forces = torch.stack(self_forces, dim=1) * env.cfg.sim.dt

    object_impulse = hand_forces[:, :, 0]
    non_affordance_impulse = hand_forces[:, :, 1]
    table_impulse = hand_forces[:, :, 2]
    arm_table_impulse = arm_forces[:, :, 2]
    arm_self_impulse = self_forces.norm(dim=-1).squeeze(-1)
    contacts_object = object_impulse.norm(dim=-1) > 0.01
    contacts_non_affordance = non_affordance_impulse.norm(dim=-1) > 0.01
    contacts_table = table_impulse.norm(dim=-1) > 0.01
    contacts_arm_table = arm_table_impulse.norm(dim=-1) > 0.01
    contacts_arm_all = arm_self_impulse > 0.01
    return object_impulse, non_affordance_impulse, table_impulse, arm_table_impulse, contacts_object, contacts_non_affordance, contacts_table, contacts_arm_table, contacts_arm_all

__all__ = ["make_contact_sensor_cfgs", "make_self_collision_sensor_cfgs", "collect_contact_data"]
