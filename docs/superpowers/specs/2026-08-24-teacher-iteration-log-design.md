# Teacher iteration terminal log design

## Goal

Replace the current one-line Teacher update message with a full block printed after every PPO update, following the layout of the supplied Isaac Lab training screenshot. This change affects reporting only; PPO optimization, rollout length, rewards, checkpoints, and evaluation remain unchanged.

## Scope

Modify only:

- `scripts/train_teacher.py`
- `source/Grasp/Grasp/algo/ppo_teacher/ppo.py`

Do not add dependencies or change Student and one-step training.

## Metrics

Each update block reports:

- one-based learning iteration and configured maximum iterations;
- throughput in environment steps per second;
- collection and learning duration;
- mean action noise standard deviation;
- mean value-function loss;
- mean surrogate loss;
- mean policy entropy, labelled `Mean entropy loss` to match the requested layout;
- mean episode reward and mean episode length for the rollout segments completed or closed at the 70-step rollout boundary;
- cumulative PPO timesteps and current learning rate;
- iteration duration, elapsed wall time, and ETA.

## Data flow

`run_rollout()` keeps per-environment reward and length accumulators. A terminated or truncated environment closes a segment; remaining segments close at the rollout boundary. It returns their mean reward and length alongside the existing observation and environment log values.

`PPO._train_step()` accumulates entropy together with value and surrogate losses. `PPO.update()` returns the three means after writing the existing TensorBoard metrics. The training loop requests logging every update so `PPO.tot_timesteps` advances every update instead of every tenth update.

The training loop measures collection and learning with `time.perf_counter()`, derives throughput from `num_envs * grasp_steps`, and calculates ETA from the mean duration of updates completed in the current process. Resume keeps checkpoint timesteps while elapsed time and ETA describe the current process.

## Output

The existing `update=...` line is replaced with an 80-column block headed by `Learning iteration current/total`. Checkpoint and evaluation messages remain separate and continue printing after their corresponding iteration block.

## Error behavior and verification

The existing exploding-gradient error remains authoritative; no metric block is printed from an update that fails. Verification is limited to Python AST parsing and `git diff --check`, because project instructions prohibit running tests, training, or simulation.
