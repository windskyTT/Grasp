"""FR3 + Inspire Teacher reward：纯 Torch CUDA 计算。

本文件只负责把 env.py 已经计算好的 CUDA Tensor 转成 reward term。

保持当前 RobustDexGrasp Teacher 迁移语义：
    - top    = affordance part
    - bottom = non-affordance part
    - bottom contact / impulse 目前只作为 raw diagnostic，
      不单独加入 weighted reward。
    - support = Table + Mat + WoodenTable
    - Teacher 最终 reward 仍由 env.py 负责 terminal penalty 和 min_reward clamp。

GPU 约定：
    - 不使用 NumPy。
    - 不创建 CPU torch.Tensor。
    - 不执行 GPU <-> CPU 数值转换。
    - reward weight Tensor 由环境初始化一次，后续复用 CUDA Tensor。
    - Python float / tuple 仅作为静态 reward 配置，不属于运行期采样数据。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .observations import TEACHER_HAND_BODY_COUNT

if TYPE_CHECKING:
    from .env_cfg import TeacherRewardCfg


# 非法/终止状态在 env.py 中额外叠加该惩罚。
TEACHER_TERMINAL_REWARD = -10.0


# env.py 会按这些名字建立日志 buffer。
# 顺序同时决定 base_reward 中各 weighted term 的堆叠顺序。
TEACHER_REWARD_TERM_NAMES = (
    "affordance_reward",
    "affordance_contact_reward",
    "affordance_impulse_reward",
    "table_reward",
    "table_contact_reward",
    "table_impulse_reward",
    "arm_height_reward",
    "arm_contact_reward",
    "arm_impulse_reward",
    "arm_collision_reward",
    "push_reward",
    "wrist_vel_reward",
    "wrist_qvel_reward",
    "obj_vel_reward",
    "obj_qvel_reward",
    "obj_displacement_reward",
    "arm_joint_vel_reward",
)


@torch.no_grad()
def compute_teacher_reward_terms(
    *,
    nearest_affordance_distance: torch.Tensor,
    hand_body_height: torch.Tensor,
    arm_body_height: torch.Tensor,
    top_total_impulse_vector_w: torch.Tensor,
    top_normal_impulse_vector_w: torch.Tensor,
    top_friction_impulse_vector_w: torch.Tensor,
    bottom_total_impulse_vector_w: torch.Tensor,
    support_total_impulse_vector_w: torch.Tensor,
    arm_interaction_total_impulse_w: torch.Tensor,
    arm_all_contact: torch.Tensor,
    wrist_linear_velocity_w: torch.Tensor,
    wrist_angular_velocity_w: torch.Tensor,
    object_top_linear_velocity_w: torch.Tensor,
    object_top_angular_velocity_w: torch.Tensor,
    object_position_w: torch.Tensor,
    object_initial_position_w: torch.Tensor,
    arm_joint_velocity: torch.Tensor,
    arm_height_penalty_indices: torch.Tensor,
    arm_collision_indices: torch.Tensor,
    geometry_weights: torch.Tensor,
    contact_weights: torch.Tensor,
    impulse_upper: torch.Tensor,
    cfg: TeacherRewardCfg,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    """计算一个 RL step 的所有 Teacher reward term。

    输入主要 shape：
        nearest_affordance_distance      [B, 13]
        hand_body_height                 [B, 13]
        arm_body_height                  [B, 6]

        top_*_impulse_vector_w           [B, 13, 3]
        bottom_total_impulse_vector_w    [B, 13, 3]
        support_total_impulse_vector_w   [B, 13, 3]

        arm_interaction_total_impulse_w  [B, 6, 3]
        arm_all_contact                  [B, 6]

        wrist linear/angular velocity    [B, 3]
        object top velocity              [B, 3]
        object root position             [B, 3]
        arm_joint_velocity               [B, 7]

    输出：
        base_reward:
            [B]

        weighted_terms:
            每一项已经乘 cfg 中的 reward scale。

        raw_terms:
            未乘 reward scale 的原始诊断量。
    """

    dtype = nearest_affordance_distance.dtype

    # 13 = 当前 Inspire Teacher hand geometry/contact body 数量。
    # 不再把 13.0 作为散落的 magic number。
    hand_contact_normalizer = float(
        TEACHER_HAND_BODY_COUNT
    )

    # =========================================================================
    # 1. Affordance geometry reward
    # =========================================================================
    # 每个 hand body 到最近 top-affordance point 的距离：
    #
    #     r_aff_raw
    #       = - sum_i(
    #             distance_i
    #             * geometry_weight_i
    #         )
    #
    # 距离越小，raw reward 越接近 0，经过正 scale 后惩罚越小。
    affordance_raw = -(
        nearest_affordance_distance
        * geometry_weights
    ).sum(
        dim=-1
    )

    # =========================================================================
    # 2. Hand height / table proximity
    # =========================================================================
    # 原迁移公式：
    #
    #     -log(
    #         50 * clamp(height, 0.002, 0.02)
    #     )
    #
    # height 越接近支撑面，该 raw term 越大；
    # 最终 cfg.table_reward_scale 为负，因此表现为 table proximity penalty。
    table_raw = -(
        torch.log(
            50.0
            * torch.clamp(
                hand_body_height,
                min=0.002,
                max=0.02,
            )
        )
        * geometry_weights
    ).sum(
        dim=-1
    )

    # 只对 cfg 指定的 FR3 link 计算 arm height penalty。
    selected_arm_height = (
        arm_body_height.index_select(
            1,
            arm_height_penalty_indices,
        )
    )

    arm_height_raw = -torch.log(
        50.0
        * torch.clamp(
            selected_arm_height,
            min=0.002,
            max=0.02,
        )
    ).sum(
        dim=-1
    )

    # =========================================================================
    # 3. Top affordance contact / impulse
    # =========================================================================
    # 当前 env.py 给这里的是当前 Teacher contact step 的 impulse vector。
    #
    # total     = normal + friction
    # tangential = friction
    # normal     = normal
    top_total_norm = (
        torch.linalg.vector_norm(
            top_total_impulse_vector_w,
            dim=-1,
        )
    )

    top_tangential_norm = (
        torch.linalg.vector_norm(
            top_friction_impulse_vector_w,
            dim=-1,
        )
    )

    top_normal_norm = (
        torch.linalg.vector_norm(
            top_normal_impulse_vector_w,
            dim=-1,
        )
    )

    # impulse 大于阈值就视为发生 top affordance contact。
    top_contact = (
        top_total_norm
        > cfg.contact_impulse_threshold
    ).to(
        dtype
    )

    # Contact reward：
    #     weighted contact count / 13
    affordance_contact_raw = (
        top_contact
        * contact_weights
    ).sum(
        dim=-1
    ) / hand_contact_normalizer

    # Tangential impulse reward：
    # 每个 hand body 的摩擦 impulse 先按 upper bound 截断，
    # 再乘 contact weight。
    affordance_impulse_raw = (
        torch.minimum(
            top_tangential_norm,
            impulse_upper,
        )
        * contact_weights
    ).sum(
        dim=-1
    )

    # =========================================================================
    # 4. Bottom contact / impulse diagnostics
    # =========================================================================
    # bottom = non-affordance part。
    #
    # 当前迁移版本继续计算 raw diagnostics，
    # 但 TEACHER_REWARD_TERM_NAMES 中没有单独 bottom weighted reward，
    # 因此这里不会改变最终 base_reward。
    bottom_norm = (
        torch.linalg.vector_norm(
            bottom_total_impulse_vector_w,
            dim=-1,
        )
    )

    bottom_contact = (
        bottom_norm
        > cfg.contact_impulse_threshold
    ).to(
        dtype
    )

    bottom_contact_raw = (
        bottom_contact
        * contact_weights
    ).sum(
        dim=-1
    ) / hand_contact_normalizer

    bottom_impulse_raw = (
        torch.minimum(
            bottom_norm,
            impulse_upper,
        )
        * contact_weights
    ).sum(
        dim=-1
    )

    # =========================================================================
    # 5. Support/Table contact
    # =========================================================================
    # env.py 已经把：
    #     Table
    #     Mat
    #     WoodenTable
    # 三个 support filter 的 impulse 向量求和。
    support_norm = (
        torch.linalg.vector_norm(
            support_total_impulse_vector_w,
            dim=-1,
        )
    )

    support_contact = (
        support_norm
        > cfg.contact_impulse_threshold
    ).to(
        dtype
    )

    table_contact_raw = (
        support_contact
        * contact_weights
    ).sum(
        dim=-1
    ) / hand_contact_normalizer

    table_impulse_raw = (
        torch.minimum(
            support_norm,
            impulse_upper,
        )
        * contact_weights
    ).sum(
        dim=-1
    )

    # =========================================================================
    # 6. FR3 arm interaction / collision
    # =========================================================================
    arm_interaction_norm = (
        torch.linalg.vector_norm(
            arm_interaction_total_impulse_w,
            dim=-1,
        )
    )

    arm_interaction_contact = (
        arm_interaction_norm
        > cfg.contact_impulse_threshold
    ).to(
        dtype
    )

    # [B, 6] contact flag -> 每个环境一个标量。
    arm_contact_raw = (
        torch.linalg.vector_norm(
            arm_interaction_contact,
            dim=-1,
        )
    )

    # [B, 6] impulse magnitude -> 每个环境一个标量。
    arm_impulse_raw = (
        torch.linalg.vector_norm(
            arm_interaction_norm,
            dim=-1,
        )
    )

    # 只检查 cfg.reward.arm_collision_body_names 对应 link。
    arm_collision_raw = (
        arm_all_contact.index_select(
            1,
            arm_collision_indices,
        )
        .to(dtype)
        .sum(dim=-1)
    )

    # =========================================================================
    # 7. Push penalty diagnostic
    # =========================================================================
    # 第 0 个 hand body 是 palm/base body。
    # 其余 12 个是 Inspire finger bodies。
    push_raw = torch.clamp(
        (
            top_normal_norm[:, 0]
            - cfg.push_palm_threshold
        ),
        min=0.0,
    )

    push_raw = (
        push_raw
        + torch.clamp(
            (
                top_normal_norm[:, 1:]
                - cfg.push_finger_threshold
            ),
            min=0.0,
        ).sum(
            dim=-1
        )
    )

    push_raw = torch.clamp(
        push_raw,
        max=cfg.push_maximum,
    )

    # 当前 cfg.push_reward_scale = -0.0，
    # 因此这个 raw term 目前主要用于保持原接口与诊断。
    # 不在这里擅自改变其 reward scale。

    # =========================================================================
    # 8. Wrist motion penalties
    # =========================================================================
    wrist_speed = (
        torch.linalg.vector_norm(
            wrist_linear_velocity_w,
            dim=-1,
        )
    )

    wrist_vel_raw = (
        wrist_speed.square()
    )

    # 超过阈值后加强 wrist linear velocity penalty。
    wrist_vel_raw = torch.where(
        (
            wrist_speed
            > cfg.wrist_velocity_threshold
        ),
        (
            wrist_vel_raw
            * cfg.wrist_velocity_multiplier
        ),
        wrist_vel_raw,
    )

    wrist_qvel_raw = (
        wrist_angular_velocity_w
        .square()
        .sum(
            dim=-1
        )
    )

    # =========================================================================
    # 9. Object motion penalties
    # =========================================================================
    obj_vel_raw = (
        object_top_linear_velocity_w
        .square()
        .sum(
            dim=-1
        )
    )

    obj_qvel_raw = (
        object_top_angular_velocity_w
        .square()
        .sum(
            dim=-1
        )
    )

    obj_displacement_raw = (
        torch.linalg.vector_norm(
            (
                object_position_w
                - object_initial_position_w
            ),
            dim=-1,
        )
    )

    # =========================================================================
    # 10. FR3 arm joint velocity penalty
    # =========================================================================
    # 超过 threshold 的关节速度先乘 multiplier，再平方。
    scaled_arm_joint_velocity = (
        torch.where(
            (
                arm_joint_velocity.abs()
                > cfg.arm_joint_velocity_threshold
            ),
            (
                arm_joint_velocity
                * cfg.arm_joint_velocity_multiplier
            ),
            arm_joint_velocity,
        )
    )

    arm_joint_vel_raw = (
        scaled_arm_joint_velocity
        .square()
        .sum(
            dim=-1
        )
    )

    # =========================================================================
    # 11. 乘 reward scale
    # =========================================================================
    weighted_terms = {
        "affordance_reward": (
            affordance_raw
            * cfg.affordance_reward_scale
        ),
        "affordance_contact_reward": (
            affordance_contact_raw
            * cfg.affordance_contact_reward_scale
        ),
        "affordance_impulse_reward": (
            affordance_impulse_raw
            * cfg.affordance_impulse_reward_scale
        ),
        "table_reward": (
            table_raw
            * cfg.table_reward_scale
        ),
        "table_contact_reward": (
            table_contact_raw
            * cfg.table_contact_reward_scale
        ),
        "table_impulse_reward": (
            table_impulse_raw
            * cfg.table_impulse_reward_scale
        ),
        "arm_height_reward": (
            arm_height_raw
            * cfg.arm_height_reward_scale
        ),
        "arm_contact_reward": (
            arm_contact_raw
            * cfg.arm_contact_reward_scale
        ),
        "arm_impulse_reward": (
            arm_impulse_raw
            * cfg.arm_impulse_reward_scale
        ),
        "arm_collision_reward": (
            arm_collision_raw
            * cfg.arm_collision_reward_scale
        ),
        "push_reward": (
            push_raw
            * cfg.push_reward_scale
        ),
        "wrist_vel_reward": (
            wrist_vel_raw
            * cfg.wrist_vel_reward_scale
        ),
        "wrist_qvel_reward": (
            wrist_qvel_raw
            * cfg.wrist_qvel_reward_scale
        ),
        "obj_vel_reward": (
            obj_vel_raw
            * cfg.obj_vel_reward_scale
        ),
        "obj_qvel_reward": (
            obj_qvel_raw
            * cfg.obj_qvel_reward_scale
        ),
        "obj_displacement_reward": (
            obj_displacement_raw
            * cfg.obj_displacement_reward_scale
        ),
        "arm_joint_vel_reward": (
            arm_joint_vel_raw
            * cfg.arm_joint_vel_reward_scale
        ),
    }

    # 所有 reward term [B] -> stack [17, B] -> sum [B]。
    base_reward = torch.stack(
        tuple(
            weighted_terms.values()
        ),
        dim=0,
    ).sum(
        dim=0
    )

    # =========================================================================
    # 12. Lift success diagnostic
    # =========================================================================
    # 成功条件只看 root z 相对 reset 初始值是否抬高超过阈值。
    lift_success = (
        (
            object_position_w[:, 2]
            - object_initial_position_w[:, 2]
        )
        > cfg.lift_success_height
    ).to(
        dtype
    )

    # =========================================================================
    # 13. Raw diagnostic terms
    # =========================================================================
    raw_terms = {
        "affordance_distance_raw": affordance_raw,
        "top_contact_raw": affordance_contact_raw,
        "top_impulse_tangential_raw": affordance_impulse_raw,
        "bottom_contact_raw": bottom_contact_raw,
        "bottom_impulse_raw": bottom_impulse_raw,
        "table_height_raw": table_raw,
        "support_contact_raw": table_contact_raw,
        "support_impulse_raw": table_impulse_raw,
        "arm_height_raw": arm_height_raw,
        "arm_contact_raw": arm_contact_raw,
        "arm_impulse_raw": arm_impulse_raw,
        "arm_collision_raw": arm_collision_raw,
        "push_raw": push_raw,
        "wrist_vel_raw": wrist_vel_raw,
        "wrist_qvel_raw": wrist_qvel_raw,
        "obj_vel_raw": obj_vel_raw,
        "obj_qvel_raw": obj_qvel_raw,
        "obj_displacement_raw": obj_displacement_raw,
        "arm_joint_vel_raw": arm_joint_vel_raw,
        "lift_success": lift_success,
    }

    return (
        base_reward,
        weighted_terms,
        raw_terms,
    )


__all__ = [
    "TEACHER_REWARD_TERM_NAMES",
    "TEACHER_TERMINAL_REWARD",
    "compute_teacher_reward_terms",
]
