# RobustDexGrasp Teacher source mapping

This package is independent of `tasks/direct/grasp/teacher/`. The source of
algorithm semantics is `/home/windsky/project/RobustDexGrasp`.

| RobustDexGrasp source | IsaacLab target | Migrated role |
|---|---|---|
| `allegro_teacher/Environment.hpp` | `env.py`, `actions.py`, `observations.py`, `rewards.py`, `contact.py`, `resets.py` | UR5+Allegro state, action, observation, contact, reward, and reset state machine |
| `allegro_teacher/cfgs/cfg_reg.yaml` | `cfgs/cfg_reg.yaml`, `env_cfg.py` | original defaults and IsaacLab scene configuration |
| `allegro_teacher/train.py` | `train_teacher.py` | original object repetitions, reset preparation, PPO rollout and checkpoint loop |
| `allegro_teacher/quantitative_eval.py` | `quantitative_eval.py` | original ShapeNet quantitative evaluation |
| `allegro_teacher/visual_eval.py` | `visual_eval.py` | original training-set visual grasp/lift flow |
| `algo/ppo/module.py` | `ppo/module.py` | source MLP, actor, critic, and diagonal Gaussian |
| `algo/ppo/ppo.py` | `ppo/ppo.py` | source PPO update and learning-rate schedule |
| `algo/ppo/storage.py` | `ppo/storage.py` | source rollout storage, GAE, and minibatches |
| `helper/initial_pose_final.py` | `pregrasp.py` | source sampled wrist frames and projection score |
| `helper/inverseKinematicsUR5.py` | `ik.py` | source UR5 DH analytic IK |
| `helper/rotations.py` | `rotations.py` | source quaternion, axis-angle, and frame transforms |
| `env/RaisimGymVecEnvOther.py::observe_vision_new` | `observations.py` | source 102 + 51 affordance-vector observation construction |
| `hardware/hand/AllegroSim.cpp` | `robot_cfg.py`, `contact.py` | source 17 hand body frames and 13 contact bodies |
| `hardware/arm/UR5Sim.cpp` | `robot_cfg.py`, `contact.py` | source six arm joints and six arm bodies |
| `rsc/new_training_set` | `object_set.py` | source training objects and original extra repetitions |
| `rsc/shapenet-30obj` | `object_set.py` | source evaluation objects |

The original layout is explicit: the raw environment vector is 102 values
(`gc`, target residual, 13 contacts, 13 impulses, 17 hand heights, 6 arm
heights, hand center, wrist Euler delta, wrist Euler), the global vector is 128
values, and `observe_vision_new` appends 17 nearest-point vectors (51 values)
for a 153-value Teacher observation.
