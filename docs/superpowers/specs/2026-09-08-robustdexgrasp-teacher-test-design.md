# RobustDexGrasp Teacher to IsaacLab teacher_test Design

## Goal

Create an isolated IsaacLab 2.3.2 Direct RL environment at
`source/Grasp/Grasp/tasks/direct/grasp/teacher_test/` that preserves the local
RobustDexGrasp Teacher behavior while replacing RaiSim APIs with IsaacLab and
PhysX APIs.

## Scope and source authority

The algorithm source is the local checkout of RobustDexGrasp under
`/home/windsky/project/RobustDexGrasp`. The existing Grasp `teacher/` package is
not an algorithm source and is not imported or copied. Existing Grasp package
registration and IsaacLab calling conventions may be followed.

The migration keeps UR5 + Allegro, the original `new_training_set` training
dataset and `shapenet-30obj` evaluation dataset, the original object ordering
and repetitions, the original reset/pregrasp/analytic UR5 IK flow, contact
semantics, reward coefficients, episode timing, and RobustDexGrasp PPO.

## Data flow

The IsaacLab environment owns the UR5+Allegro articulation, one top/bottom
object articulation per environment, and the support table. It exposes the
original 102-dimensional environment observation and 128-dimensional global
state semantics. The Teacher observation builder computes the 17 body-to-top
affordance nearest-point vectors (51 values) in the same frame convention as
`observe_vision_new`, yielding exactly 153 policy inputs. Actions are residual
joint-position targets for all 22 DOFs, scaled by the original 6-arm/16-hand
standard deviations and clipped to joint limits.

The environment performs 20 PhysX substeps per policy action at `dt=0.01`,
retains the one-step random target delay, and reports the original 17 reward
terms with reward floor `-2.0`. Reset uses the original XY sampling,
lowest-point placement, z-yaw rotation, surface visibility points, sampled
wrist frames, analytic UR5 IK, and self-collision rejection semantics.

## Files and isolation

New implementation files remain under `teacher_test/`, including environment,
configuration, object/data helpers, analytic IK/rotation helpers, the copied
source PPO implementation with only tensor/interface adaptation, training, and
both evaluation entry points. `grasp/__init__.py` receives only the independent
Gym registration for `Grasp-RobustDexTeacher-UR5-Allegro-v0`.

No optimization, compatibility matrix, defensive fallback, placeholder reward,
placeholder observation, or unrelated refactor is part of this design.

## Verification boundary

After implementation, run static Python parsing/import checks and the smallest
allowed IsaacLab runtime checks from the attachment: registration, one-env
startup/reset/step, dimension checks, short lifecycle rollout, one-iteration
Teacher PPO update, quantitative evaluation, and visual evaluation when the
display permits. Do not claim behavior or performance beyond the commands that
actually complete.
