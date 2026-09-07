"""Teacher reset 采样：纯 Torch CUDA 实现。

本文件负责：
1. 采样 FR3 工作空间中的 object XY。
2. 采样 object yaw / pose。
3. 使用 stable state 构造 object pose。
4. 反复生成 pregrasp candidate，并剔除 IK 无效与自碰撞状态。
5. reset 统计量清零。

GPU 约定：
    - 不使用 NumPy。
    - 不创建 CPU torch.Tensor。
    - 所有随机数通过 torch 在指定 CUDA device 上直接生成。
    - Beta distribution 的 concentration Tensor 直接建立在 CUDA。
    - 不执行 GPU -> CPU 数值转换。
    - reset 阶段不需要梯度，因此采样函数统一使用 @torch.no_grad()。

说明：
    本文件中的 Python float / tuple / cfg 参数只是静态配置。
    CUDA Tensor 与 Python scalar 做乘加时，真正的数值 kernel 仍在 GPU 上执行。
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .env_cfg import TeacherResetCfg


TeacherResetCandidateSampler = Callable[
    [torch.Tensor],
    torch.Tensor,
]

TeacherSelfCollisionChecker = Callable[
    [torch.Tensor],
    torch.Tensor,
]


@torch.no_grad()
def _sample_angle_and_distance_unit_interval(
    count: int,
    cfg: TeacherResetCfg,
    device: torch.device | str,
    uniform_only: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """在 GPU 上采样 angle / distance 的 [0, 1] 随机变量。

    RobustDexGrasp 风格的 reset distribution：

        uniform_only=True:
            angle ~ Uniform(0, 1)
            distance ~ Uniform(0, 1)

        non_uniform_sampling=True:
            先分别生成：
                Uniform(0, 1)
                Beta(alpha, beta)

            然后每个样本以：
                uniform_sampling_probability
            选择 Uniform，否则选择 Beta。

    当前默认：
        alpha = beta = 0.5

    Beta(0.5, 0.5) 会更偏向区间两端，
    因此 object 的 angle / distance 会比纯均匀采样更多地出现在边界附近。
    """

    # -------------------------------------------------------------
    # Uniform samples：直接由 CUDA RNG 生成。
    # -------------------------------------------------------------
    uniform_angle = torch.rand(
        (count,),
        dtype=torch.float32,
        device=device,
    )

    uniform_distance = torch.rand(
        (count,),
        dtype=torch.float32,
        device=device,
    )

    if (
        uniform_only
        or not cfg.non_uniform_sampling
    ):
        return (
            uniform_angle,
            uniform_distance,
        )

    # -------------------------------------------------------------
    # Beta concentration 直接创建在 CUDA。
    #
    # 这里没有：
    #     CPU tensor
    #         -> .to(cuda)
    #
    # torch.distributions.Beta.sample() 会继承 concentration 的 device。
    # -------------------------------------------------------------
    beta_alpha = torch.tensor(
        cfg.beta_alpha,
        dtype=torch.float32,
        device=device,
    )

    beta_beta = torch.tensor(
        cfg.beta_beta,
        dtype=torch.float32,
        device=device,
    )

    beta_distribution = (
        torch.distributions.Beta(
            beta_alpha,
            beta_beta,
        )
    )

    beta_angle = beta_distribution.sample(
        (count,)
    )

    beta_distance = beta_distribution.sample(
        (count,)
    )

    # 每个 sample 独立决定使用 uniform 还是 beta。
    use_uniform = torch.rand(
        (count,),
        dtype=torch.float32,
        device=device,
    ) < cfg.uniform_sampling_probability

    angle_unit = torch.where(
        use_uniform,
        uniform_angle,
        beta_angle,
    )

    distance_unit = torch.where(
        use_uniform,
        uniform_distance,
        beta_distance,
    )

    return (
        angle_unit,
        distance_unit,
    )


@torch.no_grad()
def sample_teacher_object_xy(
    count: int,
    cfg: TeacherResetCfg,
    workspace_center_y: float,
    device: torch.device | str,
    uniform_only: bool = False,
) -> torch.Tensor:
    """在 FR3 工作空间内采样 object XY。

    原 RobustDexGrasp 的 source workspace 使用极坐标：
        source_angle
        distance

    当前迁移到 FR3 后再增加：
        source_to_fr3_angle_offset

    因此：

        fr3_angle
            = source_angle
            + source_to_fr3_angle_offset

        local_x
            = distance * cos(fr3_angle)

        local_y
            = distance * sin(fr3_angle)

    然后保留：
        local_y_range[0]
        <
        local_y
        <
        local_y_range[1]

    rejection sampling 中的 candidate_xy 始终是 CUDA Tensor。
    Python list 只保存这些 CUDA Tensor 的引用，
    最终 torch.cat() 仍然在 GPU 上执行。
    """

    accepted_xy: list[torch.Tensor] = []
    accepted_count = 0

    while accepted_count < count:
        remaining = (
            count
            - accepted_count
        )

        # 一次多生成一些 candidate，减少 rejection loop 次数。
        draw_count = max(
            remaining * 2,
            64,
        )

        (
            angle_unit,
            distance_unit,
        ) = (
            _sample_angle_and_distance_unit_interval(
                count=draw_count,
                cfg=cfg,
                device=device,
                uniform_only=uniform_only,
            )
        )

        # ---------------------------------------------------------
        # [0,1] -> source angle range
        # ---------------------------------------------------------
        source_angle = (
            cfg.angle_range[0]
            + angle_unit
            * (
                cfg.angle_range[1]
                - cfg.angle_range[0]
            )
        )

        # ---------------------------------------------------------
        # [0,1] -> radial distance range
        # ---------------------------------------------------------
        distance = (
            cfg.distance_range[0]
            + distance_unit
            * (
                cfg.distance_range[1]
                - cfg.distance_range[0]
            )
        )

        # ---------------------------------------------------------
        # UR5/source workspace -> FR3 workspace。
        # 当前默认 offset = +pi/2。
        # ---------------------------------------------------------
        fr3_angle = (
            source_angle
            + cfg.source_to_fr3_angle_offset
        )

        local_x = (
            distance
            * torch.cos(fr3_angle)
        )

        local_y = (
            distance
            * torch.sin(fr3_angle)
        )

        # 只接受 FR3 workspace 中允许的 local y。
        accepted = (
            (
                local_y
                > cfg.local_y_range[0]
            )
            & (
                local_y
                < cfg.local_y_range[1]
            )
        )

        candidate_xy = torch.stack(
            (
                local_x,
                (
                    local_y
                    + workspace_center_y
                ),
            ),
            dim=-1,
        )[accepted]

        # shape[0] / numel() 是 Tensor metadata，
        # 不把 candidate 数值搬回 CPU。
        take_count = min(
            remaining,
            candidate_xy.shape[0],
        )

        if take_count > 0:
            accepted_xy.append(
                candidate_xy[
                    :take_count
                ]
            )

            accepted_count += (
                take_count
            )

    # 所有 list 元素都是 CUDA Tensor；
    # torch.cat() 输出仍然位于同一 CUDA device。
    return torch.cat(
        accepted_xy,
        dim=0,
    )


@torch.no_grad()
def sample_teacher_object_pose(
    env_origins: torch.Tensor,
    lowest_points: torch.Tensor,
    support_height: float,
    workspace_center_y: float,
    cfg: TeacherResetCfg,
    uniform_only: bool = False,
) -> torch.Tensor:
    """直接在 GPU 上生成普通 Teacher object root pose。

    输出：
        [B, 7]

        xyz + quaternion(wxyz)

    object z：
        env_origin_z
        + support_height
        - lowest_point

    因此物体最低点落在 table / support surface 上。
    """

    count = env_origins.shape[0]
    device = env_origins.device
    dtype = env_origins.dtype

    object_xy = sample_teacher_object_xy(
        count=count,
        cfg=cfg,
        workspace_center_y=(
            workspace_center_y
        ),
        device=device,
        uniform_only=uniform_only,
    )

    # yaw 直接在 CUDA 上 uniform sampling。
    yaw = torch.empty(
        (count,),
        dtype=dtype,
        device=device,
    ).uniform_(
        cfg.yaw_range[0],
        cfg.yaw_range[1],
    )

    # xyz + quaternion(wxyz)
    object_pose_w = torch.zeros(
        (count, 7),
        dtype=dtype,
        device=device,
    )

    # local XY -> 每个 IsaacLab environment 的 world XY。
    object_pose_w[:, 0:2] = (
        object_xy
        + env_origins[:, 0:2]
    )

    # 让 mesh 最低点落在 support height。
    object_pose_w[:, 2] = (
        env_origins[:, 2]
        + support_height
        - lowest_points.reshape(count)
    )

    # 纯 yaw quaternion：
    #
    #     q = [cos(yaw/2), 0, 0, sin(yaw/2)]
    #
    # IsaacLab quaternion 顺序为 wxyz。
    object_pose_w[:, 3] = (
        torch.cos(
            0.5 * yaw
        )
    )

    object_pose_w[:, 6] = (
        torch.sin(
            0.5 * yaw
        )
    )

    return object_pose_w


@torch.no_grad()
def sample_teacher_stable_object_pose(
    env_origins: torch.Tensor,
    stable_states: torch.Tensor,
    support_height: float,
    source_support_height: float,
    workspace_center_y: float,
    cfg: TeacherResetCfg,
) -> torch.Tensor:
    """使用数据集 stable state 在 GPU 上生成 object root pose。

    stable_states:
        [B, 7]
        已经由 affordance_data.py 直接创建在 CUDA。

    stable state 中保留：
        z
        quaternion

    XY 仍然重新在当前 FR3 workspace 中采样。
    """

    count = env_origins.shape[0]
    device = env_origins.device
    dtype = env_origins.dtype

    # stable-state evaluation 使用 uniform workspace sampling。
    object_xy = sample_teacher_object_xy(
        count=count,
        cfg=cfg,
        workspace_center_y=(
            workspace_center_y
        ),
        device=device,
        uniform_only=True,
    )

    object_pose_w = torch.zeros(
        (count, 7),
        dtype=dtype,
        device=device,
    )

    object_pose_w[:, 0:2] = (
        object_xy
        + env_origins[:, 0:2]
    )

    # 把 source dataset 的 support height
    # 平移到当前 IsaacLab support height。
    object_pose_w[:, 2] = (
        env_origins[:, 2]
        + support_height
        + stable_states[:, 2]
        - source_support_height
    )

    # stable orientation 直接保留。
    object_pose_w[:, 3:7] = (
        stable_states[:, 3:7]
    )

    return object_pose_w


@torch.no_grad()
def sample_collision_free_teacher_resets(
    env_ids: torch.Tensor,
    max_reset_rounds: int,
    sample_candidates: TeacherResetCandidateSampler,
    check_self_collision: TeacherSelfCollisionChecker,
) -> None:
    """生成 Teacher pregrasp，并只对失败环境做一次修复。

    旧迁移版：
        pending envs
            ↓
        pregrasp + raycast + 10 candidate + 64-step DLS IK
            ↓
        collision screening
            ↓
        失败后重新执行整套流程
            ↓
        最多 max_reset_rounds=32 次

    这会把一次 reset 最坏放大为：
        32 × 完整 pregrasp pipeline

    当前版本改为：

        第 1 阶段：
            所有 env 只生成一次 candidate
            + 一次 self-collision screening

        第 2 阶段：
            只对第 1 阶段失败/碰撞 env
            再做一次 candidate 修复
            + 一次 self-collision screening

    因此完整 pregrasp pipeline 最多执行两轮，而不是 32 轮。

    为什么不直接照搬原 Raisim 的“复制同物体正常环境状态”：
        当前 resets.py 的回调接口只能访问：
            sample_candidates(env_ids)
            check_self_collision(env_ids)

        它拿不到 env.py 内部的：
            reset_robot_joint_pos
            reset_object_root_state
            object -> env 分组

        为了遵守本次“只改 pregrasp.py / resets.py / train_teacher.py”
        的要求，这里不修改 env.py 接口，而采用一次局部重采样作为 repair。

    max_reset_rounds 参数仅保留现有 env.py 调用兼容性；
    不再控制 32 轮循环。
    """

    # =================================================================
    # Phase 1: 所有环境只做一次完整 pregrasp sampling。
    # =================================================================
    candidate_valid = sample_candidates(
        env_ids
    )

    invalid_env_ids = env_ids[
        ~candidate_valid
    ]

    valid_env_ids = env_ids[
        candidate_valid
    ]

    if valid_env_ids.numel() > 0:
        self_collision = (
            check_self_collision(
                valid_env_ids
            )
        )

        collided_env_ids = (
            valid_env_ids[
                self_collision
            ]
        )
    else:
        collided_env_ids = (
            valid_env_ids
        )

    repair_env_ids = torch.cat(
        (
            invalid_env_ids,
            collided_env_ids,
        ),
        dim=0,
    )

    if repair_env_ids.numel() == 0:
        return

    # =================================================================
    # Phase 2: 只修复第 1 阶段失败的环境。
    #
    # 不再：
    #     for _ in range(32)
    #
    # 所以不会因为少量 IK / collision failure 把启动 reset 放大几十倍。
    # =================================================================
    repair_valid = sample_candidates(
        repair_env_ids
    )

    unresolved_env_ids = (
        repair_env_ids[
            ~repair_valid
        ]
    )

    repaired_valid_env_ids = (
        repair_env_ids[
            repair_valid
        ]
    )

    if repaired_valid_env_ids.numel() > 0:
        repaired_self_collision = (
            check_self_collision(
                repaired_valid_env_ids
            )
        )

        repaired_collision_env_ids = (
            repaired_valid_env_ids[
                repaired_self_collision
            ]
        )
    else:
        repaired_collision_env_ids = (
            repaired_valid_env_ids
        )

    remaining_failed_env_ids = torch.cat(
        (
            unresolved_env_ids,
            repaired_collision_env_ids,
        ),
        dim=0,
    )

    if remaining_failed_env_ids.numel() == 0:
        return

    # 姿态修正后，如果这里仍然失败，应该把它当作真实的
    # pregrasp/IK 或 collision 问题处理，而不是静默再跑 30 轮。
    raise RuntimeError(
        "Teacher reset still contains invalid or self-colliding "
        "FR3 + Inspire pregrasp states after one repair pass."
    )


@torch.no_grad()
def reset_teacher_statistics(
    buffers: dict[
        str,
        torch.Tensor,
    ],
    env_ids: torch.Tensor,
) -> None:
    """把指定环境的统计 buffer 在原 CUDA device 上清零。"""

    for value in buffers.values():
        value[env_ids] = 0.0


__all__ = [
    "TeacherResetCandidateSampler",
    "TeacherSelfCollisionChecker",
    "sample_teacher_object_xy",
    "sample_teacher_object_pose",
    "sample_teacher_stable_object_pose",
    "sample_collision_free_teacher_resets",
    "reset_teacher_statistics",
]
