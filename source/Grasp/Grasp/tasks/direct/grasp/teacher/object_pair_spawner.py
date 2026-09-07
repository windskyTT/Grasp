"""Teacher top/bottom object pair spawner.

本文件负责 Isaac Sim / USD 场景初始化阶段的资产构建。

重要：
    这里处理的是 USD Prim、Sdf Path、PhysX Schema 和文件引用，
    不属于 Teacher RL 的数值训练链路。

    因此：
        - 不需要，也不应该把本文件改成 Torch CUDA 代码。
        - 不创建 CPU torch.Tensor。
        - 不存在 NumPy -> Torch -> CUDA 的数值搬运。
        - 真正参与 reset / observation / reward / PPO 的数值数据，
          在 affordance_data.py / env.py / pregrasp.py 中保持 CUDA。

场景结构：

    Object
    ├── bottom      <- articulation root
    ├── top
    └── rotation    <- 极小范围 revolute joint

top / bottom 两部分继续保持 RobustDexGrasp 的物体语义：
    top    = affordance part
    bottom = non-affordance part
"""

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
    """IsaacLab FileCfg 的统一入口。

    clone_in_fabric / replicate_physics 由 IsaacLab spawner 接口传入。
    当前 Teacher 使用 heterogeneous object pairs，因此实际复制逻辑由
    spawn_teacher_object_pairs() 自己控制。
    """

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
    """Teacher top/bottom object pair 的资产配置。

    top_usd_paths 与 bottom_usd_paths 按同一索引一一对应：

        top_usd_paths[i]
            +
        bottom_usd_paths[i]
            ↓
        Teacher object pair i

    weighted object 的重复关系由 object_set.py 中路径列表的顺序决定。
    """

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
    """创建一个 top + bottom Teacher object prototype。

    这里完成的都是 USD authoring：
        1. 建立 prototype 根 Xform。
        2. 引用 bottom USD。
        3. 引用 top USD。
        4. 把 articulation root 放在 bottom。
        5. 用一个极小活动范围的 revolute joint 连接 top/bottom。
        6. 应用 rigid/collision/contact 配置。

    该函数只在场景创建阶段调用，不位于 RL step 热路径。
    """

    stage = sim_utils.get_current_stage()

    # 整个 prototype 的根节点。
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

    bottom = stage.DefinePrim(
        bottom_path,
        "Xform",
    )
    top = stage.DefinePrim(
        top_path,
        "Xform",
    )

    # -------------------------------------------------------------
    # 引用已经离线准备好的 top / bottom USD。
    #
    # 这里只是 USD 文件引用，不会把训练数值数据生成在 CPU。
    # -------------------------------------------------------------
    bottom.GetReferences().AddReference(
        bottom_usd_path,
        Sdf.Path("/TeacherObject/bottom"),
    )
    top.GetReferences().AddReference(
        top_usd_path,
        Sdf.Path("/TeacherObject/top"),
    )

    # -------------------------------------------------------------
    # bottom 作为整个 object pair 的 articulation root。
    # top 不能同时保留 articulation root API。
    # -------------------------------------------------------------
    top.RemoveAPI(
        UsdPhysics.ArticulationRootAPI
    )
    top.RemoveAPI(
        PhysxSchema.PhysxArticulationAPI
    )

    if not bottom.HasAPI(
        UsdPhysics.ArticulationRootAPI
    ):
        UsdPhysics.ArticulationRootAPI.Apply(
            bottom
        )

    # -------------------------------------------------------------
    # top / bottom 之间保持一个极小范围的 revolute joint。
    #
    # 这保留当前迁移代码的 object-pair articulation 结构：
    #     lower = 0
    #     upper = 0.001
    #
    # 不在这里改变原算法的物体结构。
    # -------------------------------------------------------------
    joint = UsdPhysics.RevoluteJoint.Define(
        stage,
        f"{prototype_path}/rotation",
    )
    joint.CreateAxisAttr().Set(
        UsdPhysics.Tokens.z
    )
    joint.CreateLowerLimitAttr().Set(
        0.0
    )
    joint.CreateUpperLimitAttr().Set(
        0.001
    )
    joint.CreateBody0Rel().SetTargets(
        [Sdf.Path(bottom_path)]
    )
    joint.CreateBody1Rel().SetTargets(
        [Sdf.Path(top_path)]
    )

    # -------------------------------------------------------------
    # IsaacLab / PhysX 资产属性。
    # -------------------------------------------------------------
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
    """为所有 Teacher 并行环境分配 top/bottom object pair。

    分配方式：

        prototype 0 -> env 0
        prototype 1 -> env 1
        ...
        prototype N -> env N
        超出 prototype 数量后按 modulo 循环。

    object_set.py 已经提前把 weighted object 顺序编码进：
        cfg.top_usd_paths
        cfg.bottom_usd_paths

    因此这里保持路径列表顺序，不重新随机，也不在 CPU 生成任何
    后续需要送入 GPU 的训练数值。
    """

    # IsaacLab spawner 接口要求保留这两个参数。
    # 当前 Teacher 的 heterogeneous asset 复制由下面的 Sdf.CopySpec 实现。
    del clone_in_fabric, replicate_physics

    stage = sim_utils.get_current_stage()

    # prim_path 例如：
    #     /World/envs/env_.*/Object
    #
    # root_path:
    #     /World/envs/env_.*
    #
    # asset_name:
    #     Object
    root_path, asset_name = prim_path.rsplit(
        "/",
        1,
    )

    # 如果 root_path 是 regex，则解析出所有环境 prim。
    source_paths = (
        sim_utils.find_matching_prim_paths(
            root_path
        )
        if re.match(
            r"^[a-zA-Z0-9/_]+$",
            root_path,
        ) is None
        else [root_path]
    )

    # -------------------------------------------------------------
    # 先创建临时 prototypes。
    # -------------------------------------------------------------
    template_path = (
        sim_utils.get_next_free_prim_path(
            "/World/TeacherObjectPairTemplate",
            stage=stage,
        )
    )
    sim_utils.create_prim(
        template_path,
        "Scope",
        stage=stage,
    )

    prototype_paths: list[str] = []

    for index, (
        top_usd_path,
        bottom_usd_path,
    ) in enumerate(
        zip(
            cfg.top_usd_paths,
            cfg.bottom_usd_paths,
            strict=True,
        )
    ):
        prototype_path = (
            f"{template_path}/Pair_{index:04d}"
        )

        _spawn_teacher_object_pair_prototype(
            prototype_path=prototype_path,
            top_usd_path=top_usd_path,
            bottom_usd_path=bottom_usd_path,
            cfg=cfg,
            translation=translation,
            orientation=orientation,
        )

        prototype_paths.append(
            prototype_path
        )

    # -------------------------------------------------------------
    # 给每个 env 创建 Object 路径。
    # -------------------------------------------------------------
    object_paths = [
        f"{source_path}/{asset_name}"
        for source_path in source_paths
    ]

    # -------------------------------------------------------------
    # 把 prototype spec 复制到每个环境。
    #
    # 注意：
    # 这是 USD layer copy，不是 runtime Tensor copy。
    # 所以与 Teacher 的“训练数值禁止 CPU -> GPU”要求没有冲突。
    # -------------------------------------------------------------
    with Sdf.ChangeBlock():
        for index, object_path in enumerate(
            object_paths
        ):
            destination = Sdf.CreatePrimInLayer(
                stage.GetRootLayer(),
                object_path,
            )

            prototype_path = prototype_paths[
                index % len(prototype_paths)
            ]

            Sdf.CopySpec(
                destination.layer,
                Sdf.Path(prototype_path),
                destination.layer,
                Sdf.Path(object_path),
            )

    # -------------------------------------------------------------
    # 语义标签。
    # -------------------------------------------------------------
    if cfg.semantic_tags is not None:
        for object_path in object_paths:
            object_prim = stage.GetPrimAtPath(
                object_path
            )

            for (
                semantic_type,
                semantic_value,
            ) in cfg.semantic_tags:
                sim_utils.add_labels(
                    object_prim,
                    labels=[
                        semantic_value.replace(
                            " ",
                            "_",
                        )
                    ],
                    instance_name=(
                        semantic_type.replace(
                            " ",
                            "_",
                        )
                    ),
                    overwrite=False,
                )

    # prototypes 只用于构建最终 env Object，复制完成后删除临时 Scope。
    sim_utils.delete_prim(
        template_path,
        stage=stage,
    )

    # 告诉 IsaacLab 当前 spawn 使用了多个不同资产。
    carb.settings.get_settings().set_bool(
        "/isaaclab/spawn/multi_assets",
        True,
    )

    # IsaacLab spawner API 需要返回一个有效 Prim。
    return stage.GetPrimAtPath(
        object_paths[0]
    )


__all__ = [
    "TeacherObjectPairSpawnerCfg",
    "spawn_teacher_object_pairs",
]
