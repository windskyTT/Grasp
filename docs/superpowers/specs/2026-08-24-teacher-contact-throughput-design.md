# Teacher Contact Throughput Design

## Goal

Reduce Teacher rollout collection time without changing the 0.01 s physics
step, 20-step control decimation, 70-step rollout, action semantics, reward
coefficients, PPO, Student, or one-step code.

## Root cause

The latest run spends 19.307--21.579 seconds collecting one rollout and only
0.041--0.087 seconds learning. The scene currently has 19 filtered contact
sensors, `lazy_sensor_update=False`, and 20 physics substeps per policy step.
Isaac Lab therefore recomputes every contact sensor at every physics substep.
Each sensor also filters five external bodies plus 22 robot collision bodies
and retrieves both the normal force matrix and friction data.

This does not match the RobustDexGrasp control path. That implementation first
performs all 20 `world_->integrate()` calls and then calls
`updateObservation()` once, where it reads the current contact impulses.

## Approved design

Set `InteractiveSceneCfg.lazy_sensor_update=True` so physics stepping only
marks contact sensors stale. Remove the per-substep contact read from
`_apply_action()` and read each sensor once after all 20 physics substeps, when
the current policy-step observation and reward are built.

Keep only the five filtered contact targets that have current observation or
reward consumers: object top, object bottom, Table, Mat, and WoodenTable. Use
each contact sensor's existing unfiltered `net_forces_w` value for the initial
arm collision rejection and the per-step `arm_all_contact` flag. This preserves
the original any-contact meaning without maintaining 22 robot filters on all
19 sensors.

Remove `hand_self_impulse_w` and `arm_self_impulse_w`. They are not consumed by
observations or rewards; their only remaining use is a redundant finite-value
scan. Keep all normal/friction impulse tensors that feed observations or reward
terms.

## Data flow

For every policy step:

1. Zero the policy-step contact buffers.
2. Apply the selected joint target and run 20 physics substeps without forcing
   a contact sensor refresh.
3. When `_get_dones()` builds the step snapshot, refresh every contact sensor
   once.
4. Convert the last physics substep's normal and friction forces to impulses
   using `physics_dt=0.01`.
5. Split the five filtered targets into top, bottom, and support categories;
   derive arm any-contact from `net_forces_w`.
6. Reuse that snapshot for termination, reward, and observation construction.

## Files and boundaries

- Modify `source/Grasp/Grasp/tasks/direct/grasp/teacher/env_cfg.py`.
- Modify `source/Grasp/Grasp/tasks/direct/grasp/teacher/env.py`.
- Do not modify PPO, training iteration counts, physics/control timing, assets,
  Student, or one-step files.
- Do not run test, training, simulation, or playback. Verification is limited
  to Python AST parsing, targeted source checks, and diff inspection.

## Acceptance criteria

- The scene uses lazy contact-sensor updates.
- Contact filtering contains exactly the five external targets.
- `_apply_action()` performs no contact sensor read.
- Current-step contact data is read once after the decimation loop.
- Initial arm collision rejection and `arm_all_contact` use `net_forces_w`.
- No unused robot-filter or self-impulse state remains.
- The two modified Python files parse successfully.
