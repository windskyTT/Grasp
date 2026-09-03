from __future__ import annotations

import re
from dataclasses import MISSING

import carb
from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim import schemas
from isaaclab.sim.spawners.from_files.from_files_cfg import FileCfg
from isaaclab.utils import configclass


def _spawn_teacher_object_pairs_entry(
    prim_path: str,
    cfg: "TeacherObjectPairSpawnerCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    clone_in_fabric: bool = False,
    replicate_physics: bool = False,
) -> Usd.Prim:
    return spawn_teacher_object_pairs(
        prim_path=prim_path,
        cfg=cfg,
        translation=translation,
        orientation=orientation,
        clone_in_fabric=clone_in_fabric,
        replicate_physics=replicate_physics,
    )


@configclass
class TeacherObjectPairSpawnerCfg(FileCfg):
    func = _spawn_teacher_object_pairs_entry

    top_usd_paths: list[str] = MISSING
    bottom_usd_paths: list[str] = MISSING


def _spawn_teacher_object_pair_prototype(
    prototype_path: str,
    top_usd_path: str,
    bottom_usd_path: str,
    cfg: TeacherObjectPairSpawnerCfg,
    translation: tuple[float, float, float] | None,
    orientation: tuple[float, float, float, float] | None,
) -> Usd.Prim:
    stage = sim_utils.get_current_stage()
    root = sim_utils.create_prim(
        prototype_path,
        prim_type="Xform",
        translation=translation,
        orientation=orientation,
        scale=cfg.scale,
        stage=stage,
    )

    bottom_path = f"{prototype_path}/bottom"
    top_path = f"{prototype_path}/top"
    bottom = stage.DefinePrim(bottom_path, "Xform")
    top = stage.DefinePrim(top_path, "Xform")
    bottom.GetReferences().AddReference(
        bottom_usd_path,
        Sdf.Path("/TeacherObject/bottom"),
    )
    top.GetReferences().AddReference(
        top_usd_path,
        Sdf.Path("/TeacherObject/top"),
    )

    top.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    top.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    if not bottom.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(bottom)

    joint = UsdPhysics.RevoluteJoint.Define(
        stage,
        f"{prototype_path}/rotation",
    )
    joint.CreateAxisAttr().Set(UsdPhysics.Tokens.z)
    joint.CreateLowerLimitAttr().Set(0.0)
    joint.CreateUpperLimitAttr().Set(0.001)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(bottom_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(top_path)])

    if cfg.rigid_props is not None:
        schemas.modify_rigid_body_properties(
            prototype_path,
            cfg.rigid_props,
            stage=stage,
        )
    if cfg.collision_props is not None:
        schemas.modify_collision_properties(
            prototype_path,
            cfg.collision_props,
            stage=stage,
        )
    if cfg.articulation_props is not None:
        schemas.modify_articulation_root_properties(
            bottom_path,
            cfg.articulation_props,
            stage=stage,
        )
    if cfg.activate_contact_sensors:
        schemas.activate_contact_sensors(
            prototype_path,
            stage=stage,
        )
    return root


def spawn_teacher_object_pairs(
    prim_path: str,
    cfg: TeacherObjectPairSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    clone_in_fabric: bool = False,
    replicate_physics: bool = False,
) -> Usd.Prim:
    del clone_in_fabric, replicate_physics
    stage = sim_utils.get_current_stage()
    root_path, asset_name = prim_path.rsplit("/", 1)
    source_paths = (
        sim_utils.find_matching_prim_paths(root_path)
        if re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None
        else [root_path]
    )

    template_path = sim_utils.get_next_free_prim_path(
        "/World/TeacherObjectPairTemplate",
        stage=stage,
    )
    sim_utils.create_prim(template_path, "Scope", stage=stage)
    prototype_paths: list[str] = []
    for index, (top_path, bottom_path) in enumerate(
        zip(cfg.top_usd_paths, cfg.bottom_usd_paths, strict=True)
    ):
        prototype_path = f"{template_path}/Pair_{index:04d}"
        _spawn_teacher_object_pair_prototype(
            prototype_path=prototype_path,
            top_usd_path=top_path,
            bottom_usd_path=bottom_path,
            cfg=cfg,
            translation=translation,
            orientation=orientation,
        )
        prototype_paths.append(prototype_path)

    object_paths = [f"{source_path}/{asset_name}" for source_path in source_paths]
    with Sdf.ChangeBlock():
        for index, object_path in enumerate(object_paths):
            destination = Sdf.CreatePrimInLayer(
                stage.GetRootLayer(),
                object_path,
            )
            Sdf.CopySpec(
                destination.layer,
                Sdf.Path(prototype_paths[index % len(prototype_paths)]),
                destination.layer,
                Sdf.Path(object_path),
            )

    if cfg.semantic_tags is not None:
        for object_path in object_paths:
            object_prim = stage.GetPrimAtPath(object_path)
            for semantic_type, semantic_value in cfg.semantic_tags:
                sim_utils.add_labels(
                    object_prim,
                    labels=[semantic_value.replace(" ", "_")],
                    instance_name=semantic_type.replace(" ", "_"),
                    overwrite=False,
                )

    sim_utils.delete_prim(template_path, stage=stage)
    carb.settings.get_settings().set_bool(
        "/isaaclab/spawn/multi_assets",
        True,
    )
    return stage.GetPrimAtPath(object_paths[0])


__all__ = [
    "TeacherObjectPairSpawnerCfg",
    "spawn_teacher_object_pairs",
]
