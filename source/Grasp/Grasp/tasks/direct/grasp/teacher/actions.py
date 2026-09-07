from __future__ import annotations

import torch

from .math_utils import clamp_tensor

# =============================================================================
# Teacher 动作空间
# =============================================================================
# FR3：7 个主动关节
# Inspire：6 个主动关节
#
# 因此 Teacher policy 的动作维度为：
#     7 + 6 = 13
#
# Inspire 其余被动关节不由策略直接输出，而是在 build_full_joint_target()
# 中根据 mimic 关系由对应主动关节计算得到

TEACHER_ARM_ACTION_DIM = 7
TEACHER_HAND_ACTION_DIM = 6
TEACHER_ACTION_DIM = TEACHER_ARM_ACTION_DIM + TEACHER_HAND_ACTION_DIM

# 计算残差活跃目标
def compute_residual_active_target( 
    actions: torch.Tensor,
    current_active_qpos: torch.Tensor,
    active_lower_limits: torch.Tensor,
    active_upper_limits: torch.Tensor,
    arm_scale: float,
    hand_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    把 Teacher 的 13 维策略输出转换为 13 个主动关节的位置目标。
    控制形式保持 RobustDexGrasp Teacher 的 residual position control：q_target = q_current + action * scale
    其中：前 7 维：FR3，使用 arm_scale；后 6 维：Inspire 主动关节，使用 hand_scale
    """
    active_target = current_active_qpos.clone()
    # FR3 7DOF：q_target_hand = q_current_hand + action_hand * hand_scale
    active_target[:, :TEACHER_ARM_ACTION_DIM].add_(
        actions[:, :TEACHER_ARM_ACTION_DIM],
        alpha=arm_scale,
    )
    # Inspire 6DOF：q_target_hand = q_current_hand + action_hand * hand_scale
    active_target[:, TEACHER_ARM_ACTION_DIM:].add_(
        actions[:, TEACHER_ARM_ACTION_DIM:],
        alpha=hand_scale,
    )
    # 把主动关节目标限制在对应的关节位置上下限内
    active_target = clamp_tensor(
        active_target,
        active_lower_limits,
        active_upper_limits,
    )
    return actions, active_target

# 构建完整关节目标
def build_full_joint_target(
    current_joint_pos: torch.Tensor,
    active_target: torch.Tensor,
    active_joint_ids: torch.Tensor,
    passive_joint_ids: torch.Tensor,
    mimic_parent_joint_ids: torch.Tensor,
    mimic_multipliers: torch.Tensor,
    joint_lower_limits: torch.Tensor,
    joint_upper_limits: torch.Tensor,
) -> torch.Tensor:
    """
    由 13 个主动关节目标构造 FR3 + Inspire 的完整关节目标。
    处理顺序：
        current full qpos
              | 写入 13 个主动关节 target
              v
        active + unchanged passive
              | 根据 Inspire mimic 关系计算被动关节
              v
        complete joint target
              | joint-limit clamp
              v
        发送给 IsaacLab / PhysX 的完整位置目标
    """
    full_target = current_joint_pos.clone() # 从机器人当前完整关节位置开始构造目标
    full_target[:, active_joint_ids] = active_target # 将 policy 控制的 13 个主动关节写入完整 target

    passive_target = (
        full_target[:, mimic_parent_joint_ids]
        * mimic_multipliers
    )
    passive_target = clamp_tensor( # 先限制 Inspire 被动关节自身的 joint limits
        passive_target,
        joint_lower_limits[passive_joint_ids],
        joint_upper_limits[passive_joint_ids],
    )
    full_target[:, passive_joint_ids] = passive_target # 把 mimic 计算结果写回完整关节 target

    return clamp_tensor(
        full_target,
        joint_lower_limits,
        joint_upper_limits,
    )

#示例延迟掩码
def sample_delay_mask(
    num_envs: int,
    probability: float,
    device: torch.device | str,
) -> torch.Tensor:
    """
    在 GPU 上为每个并行环境采样一次动作延迟标志。
    返回：
        shape = [num_envs]
        dtype = bool
    True：
        当前 control step 的第一个 physics substep 使用上一时刻 target。
    False：
        第一个 physics substep 直接使用当前 target。
    """
    return (
        torch.rand((num_envs,),dtype=torch.float32,device=device,).lt_(probability)
    )

#选择子步骤关节目标
def select_substep_joint_target(
    current_joint_target: torch.Tensor,
    previous_joint_target: torch.Tensor,
    delay_mask: torch.Tensor,
    physics_substep: int,
    decimation: int,
) -> torch.Tensor:
    if physics_substep == 0:
        return torch.where(
            delay_mask[:, None],
            previous_joint_target,
            current_joint_target,
        )
    return current_joint_target


__all__ = [
    "TEACHER_ACTION_DIM",
    "TEACHER_ARM_ACTION_DIM",
    "TEACHER_HAND_ACTION_DIM",
    "compute_residual_active_target",
    "build_full_joint_target",
    "sample_delay_mask",
    "select_substep_joint_target",
]
