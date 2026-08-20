# Teacher 多步播放与评估运行流程

## 1. 入口与用途

```text
入口脚本：scripts/play_teacher.py
Gym task：Grasp-Teacher-Direct-v0
输入：Grasp 多步 Teacher checkpoint
策略：deterministic/noiseless Teacher actor
抓取：70 steps
抬升：30 steps
输出：success rate、mean lift height、reward term mean
```

Teacher 播放不更新网络，不加载 PPO optimizer，不调用 OneStep reference trajectory。

## 2. checkpoint 要求

本文示例使用：

```text
runs_teacher/teacher_smoke_task14/full_0_r.pt
```

播放至少需要：

```text
actor_architecture_state_dict
actor_distribution_state_dict
obs_spec.teacher_dim = 77
action_dim = 13
```

不能使用：

```text
checkpoints/ckpt/inspire.pt
RobustDexGrasp 原始 full_12500_r.pt
Student full_0_r.pt
```

## 3. Headless smoke

```bash
conda activate grasp
cd /home/windsky/project/Grasp
python scripts/play_teacher.py \
  --task Grasp-Teacher-Direct-v0 \
  --num_envs 1 \
  --episodes 1 \
  --checkpoint runs_teacher/teacher_smoke_task14/full_0_r.pt \
  --seed 1 \
  --lift_delta_z 0.005 \
  --device cuda:0 \
  --headless \
  --disable_fabric
```

预期关键输出：

```text
loaded Teacher playback checkpoint
episode=0 success_rate=... mean_lift_height=...
teacher multi-step evaluation summary
episodes=1
trials=1
success_rate=...
mean_lift_height=...
mean_affordance_reward=...
mean_lift_success=...
mean_teacher_reward=...
```

## 4. GUI 播放

去掉 `--headless`：

```bash
python scripts/play_teacher.py \
  --task Grasp-Teacher-Direct-v0 \
  --num_envs 1 \
  --episodes 10 \
  --checkpoint runs_teacher/teacher_smoke_task14/full_0_r.pt \
  --seed 1 \
  --lift_delta_z 0.005 \
  --device cuda:0 \
  --disable_fabric
```

GUI 中观察：

```text
1. 前 70 steps：Teacher 每步重新读取 77 维 observation 并输出动作。
2. 后 30 steps：保持最后的 Inspire 手部动作。
3. FR3 当前末端 pose 每步增加 Z=0.005 m。
4. 物体抬升超过 0.10 m 判定成功。
```

## 5. 参数含义

| 参数 | 含义 |
|---|---|
| `--checkpoint` | 必填，多步 Teacher checkpoint |
| `--num_envs 1` | 同时评估的环境数 |
| `--episodes 10` | 每个环境重复评估 10 个 episode |
| `--lift_delta_z 0.005` | 每个 lift policy step 的 FR3 末端 Z 增量，单位 m |
| `--seed 1` | reset 与 sampler 种子入口 |
| `--headless` | 有该参数时不显示 GUI |
| `--device cuda:0` | 使用第一张 GPU |

## 6. 播放调用链

```text
AppLauncher
  -> load Teacher env/agent cfg
  -> 把 episode 延长到 70+30+2 steps
  -> gym.make()
  -> assert_multistep_environment()
  -> build actor 77->13
  -> 严格加载 Teacher checkpoint
  -> actor.eval() + torch.inference_mode()
  -> 每个 episode：
       env.reset()
       70 次 actor.noiseless_action() + env.step()
       30 次 build_multistep_lift_action() + env.step()
       计算 lift_height > 0.10
  -> 汇总 success_rate 和 reward mean
```

## 7. reward 输出

当前播放脚本固定统计：

```text
affordance_reward
table_reward
arm_height_reward
arm_collision_reward
lift_success
teacher_reward
```

Task 12 完成后，`METRIC_NAMES` 还应加入：

```text
affordance_contact_reward
affordance_impulse_reward
table_contact_reward
table_impulse_reward
push_reward
```

如果文档中列出这些 contact mean，而脚本没有输出，先完成 Task 12 播放日志接入。

## 8. 如何理解结果

```text
trials = episodes x num_envs
success_rate = 成功抓取并抬升的 trial 数 / trials
mean_lift_height = 所有 trial 最终物体 z 增量平均值
```

`success_rate=0` 不等于脚本加载失败。还要看：

```text
checkpoint 是否正确
动作是否变化
物体是否接触
reward 是否 finite
mean_lift_height 是否接近 0
```

## 9. 常见错误

### `still inherits the forbidden OneStep environment`

当前 Teacher 基类尚未完成多步迁移。不要删除检查；完成独立多步环境后再播放。

### checkpoint 缺 `obs_spec` 或 `action_dim`

传入的不是 Grasp 多步 Teacher checkpoint。

### `action_dim=22` 与 13 不匹配

传入的是 RobustDexGrasp UR5+Allegro checkpoint，不能直接加载。

### `arm_controller` 不是 pose

多步 lift helper 需要当前末端 pose action。修正最终 Teacher 多步配置，不调用 OneStep reference 代替。

## 10. 验收

- [ ] checkpoint 严格加载。
- [ ] actor 使用 noiseless action。
- [ ] 前 70 steps 每步策略推理。
- [ ] 后 30 steps 执行多步 lift。
- [ ] 输出 trials、success_rate、mean_lift_height。
- [ ] 输出 reward term mean。
- [ ] 不调用 OneStep API。
