# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# 优先提供非随机化版本
gym.register(
    id="Grasp-OneStep-Direct-v0",
    entry_point=f"{__name__}.grasp_env_onestep:GraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_env_cfg_onestep:GraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "ppo_onestep_cfg_entry_point": f"{agents.__name__}.ppo_onestep_cfg:PPOOneStepRunnerCfg",
        
    },
)

gym.register(
    id="Grasp-OneStep-Randomized-Direct-v0",
    entry_point=f"{__name__}.grasp_env_onestep:GraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_env_cfg_onestep:GraspEnvRandomizedCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "ppo_onestep_cfg_entry_point": f"{agents.__name__}.ppo_onestep_cfg:PPOOneStepRunnerCfg",
    },
)

# 旧 task=grasp alias
gym.register(
    id="grasp",
    entry_point=f"{__name__}.grasp_env_onestep:GraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_env_cfg_onestep:GraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "ppo_onestep_cfg_entry_point": f"{agents.__name__}.ppo_onestep_cfg:PPOOneStepRunnerCfg",
    },
)


# Teacher PPO task.
gym.register(
    id="Grasp-Teacher-Direct-v0",
    entry_point=f"{__name__}.teacher.env:GraspTeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.teacher.env_cfg:GraspTeacherEnvCfg"
        ),
        "ppo_teacher_cfg_entry_point": (
            f"{agents.__name__}.ppo_teacher_cfg:PPOTeacherRunnerCfg"
        ),
    },
)

# Student DAgger/LSTM task.
gym.register(
    id="Grasp-Student-Direct-v0",
    entry_point=f"{__name__}.grasp_env_student:GraspStudentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_env_cfg_student:GraspStudentEnvCfg",
        "ppo_student_cfg_entry_point": (
            f"{agents.__name__}.ppo_student_cfg:PPOStudentRunnerCfg"
        ),
    },
)



