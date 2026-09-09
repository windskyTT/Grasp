# RobustDexGrasp Teacher teacher_test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated UR5+Allegro RobustDexGrasp Teacher environment and original PPO path under `teacher_test/`.

**Architecture:** Translate the original `Environment.hpp` state machine into an IsaacLab `DirectRLEnv`; use IsaacLab articulations, rigid objects, and contact sensors for scene data; keep original geometric reset/IK and PPO semantics in independent modules. Register a new task ID without changing the existing Teacher, Student, or one-step tasks.

**Tech Stack:** IsaacLab 2.3.2, Isaac Sim 5.1.0, Gymnasium, PyTorch, PhysX contact sensors, local RobustDexGrasp source/data.

---

### Task 1: Add isolated source mapping and package skeleton

**Files:**
- Create: `source/Grasp/Grasp/tasks/direct/grasp/teacher_test/__init__.py`
- Create: `source/Grasp/Grasp/tasks/direct/grasp/teacher_test/source_mapping.md`
- Modify: `source/Grasp/Grasp/tasks/direct/grasp/__init__.py`

- [x] Add the package marker and mapping document listing every migrated source dependency: `Environment.hpp`, `cfg_reg.yaml`, `train.py`, both evaluation scripts, PPO modules, `initial_pose_final.py`, `inverseKinematicsUR5.py`, `rotations.py`, `RaisimGymVecEnvOther.py`, and the UR5/Allegro hardware body lists.
- [x] Register `Grasp-RobustDexTeacher-UR5-Allegro-v0` directly to `teacher_test.env:RobustDexTeacherEnv` and its independent configuration/runner entry points.
- [x] Parse the new package and registration file with `ast.parse`.

### Task 2: Translate source configuration, robot/object assets, and geometry helpers

**Files:**
- Create: `teacher_test/cfgs/cfg_reg.yaml`
- Create: `teacher_test/env_cfg.py`
- Create: `teacher_test/robot_cfg.py`
- Create: `teacher_test/object_loader.py`
- Create: `teacher_test/object_set.py`
- Create: `teacher_test/math_utils.py`
- Create: `teacher_test/ik.py`
- Create: `teacher_test/rotations.py`
- Create: `teacher_test/pregrasp.py`
- Create: `teacher_test/resets.py`

- [x] Encode the original configuration values, 0.01-second physics step, 0.2-second control step, 70 grasp steps, 22 actions, 102 base observations, 128 global state, and original reward/PPO defaults.
- [x] Resolve and verify Grasp's UR5+Allegro asset joint/link order against the RobustDexGrasp source asset; use original body lists and 6+16 DOF semantics.
- [x] Build original training/evaluation object lists and preserve source ordering/repetitions without modifying `rsc` data.
- [x] Translate analytic UR5 IK, DH transforms, quaternion/rotation conversions, `sample_rot_mats`, visibility-point sampling, and pregrasp candidate scoring without replacing them with another IK/controller.
- [x] Parse all new modules.

### Task 3: Implement the IsaacLab environment and original observation/reward state machine

**Files:**
- Create: `teacher_test/actions.py`
- Create: `teacher_test/observations.py`
- Create: `teacher_test/rewards.py`
- Create: `teacher_test/contact.py`
- Create: `teacher_test/env.py`

- [x] Implement `_setup_scene`, `_pre_physics_step`, `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones`, and `_reset_idx` with IsaacLab APIs.
- [x] Preserve action residual scaling, random first-substep delay, joint clipping, contact body order, force/impulse thresholds, arm/table/object contact distinctions, and the 17 reward terms.
- [x] Construct the exact 153-dimensional policy observation from the original 102-dimensional state plus 51 affordance vectors; expose the 128-dimensional global state needed by evaluation and collision checks.
- [ ] Keep reset state, object placement, visibility points, analytic IK, pregrasp choice, stable-state/lower-point semantics, and self-collision rejection in the independent package.
- [x] Parse all new modules and inspect the resolved dimensions statically.

### Task 4: Port RobustDexGrasp PPO and training/evaluation entry points

**Files:**
- Create: `teacher_test/ppo/__init__.py`
- Create: `teacher_test/ppo/module.py`
- Create: `teacher_test/ppo/ppo.py`
- Create: `teacher_test/ppo/storage.py`
- Create: `teacher_test/train_teacher.py`
- Create: `teacher_test/quantitative_eval.py`
- Create: `teacher_test/visual_eval.py`

- [x] Port the original Actor, Critic, MLP, diagonal Gaussian distribution, storage, GAE, update order, minibatch order, and checkpoint contents with the original hyperparameters.
- [x] Connect training to the new task and preserve dataset/repetition/evaluation/lift/checkpoint behavior; allow only temporary CLI reductions for runtime verification.
- [x] Connect quantitative and visual evaluation to the new environment and preserve their distinct source workflows.
- [x] Parse all entry points and verify no placeholder implementation remains.

### Task 5: Verify the migration within the attachment's allowed boundary

**Files:**
- Modify only files under `teacher_test/` or the independent registration entry.

- [x] Run `pwd`, static AST parsing, and package/task registration checks.
- [ ] Run one-environment headless IsaacLab startup, reset, dimension, action, reward, and short rollout checks.
- [ ] Run one-iteration original Teacher PPO collection/update and minimal quantitative evaluation; attempt visual evaluation only when display support exists.
- [x] Re-read the source mapping and audit robot, action/observation dimensions, datasets, reset/IK/pregrasp, contacts, rewards, done logic, PPO, training, checkpoint, and both evaluation paths.
- [x] Report exact commands, outputs, modified files, and the remaining runtime limitation without claiming unverified parity.
