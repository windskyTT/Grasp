# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

##
# Register Gym environments.
##



# from isaaclab_tasks.utils import import_packages

# import gymnasium as gym

# # The blacklist is used to prevent importing configs from sub-packages
# _BLACKLIST_PKGS = ["utils", ".mdp"]
# # Import all configs in this package
# import_packages(__name__, _BLACKLIST_PKGS)

# gym.register(
#     id="Grasp-OneStep-Direct-v0",
#     entry_point=f"{__name__}.grasp_env_onestep:GraspEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.grasp_env_cfg_onestep:GraspEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
#     },
# )

# gym.register(
#     id="grasp",
#     entry_point=f"{__name__}.grasp_env_onestep:GraspEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.grasp_env_cfg_onestep:GraspEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
#     },
# )

from isaaclab_tasks.utils import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp"]
import_packages(__name__, _BLACKLIST_PKGS)