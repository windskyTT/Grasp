# Student 多步播放与评估运行流程

## 1. 入口与用途

```text
入口脚本：scripts/play_student.py
Gym task：Grasp-Student-Direct-v0
checkpoint：Grasp Student checkpoint
抓取：70 steps
抬升：80 steps
策略：Student actor + LSTM encoder，确定性动作
输出：success rate、mean lift height、reward term mean
```

Student 播放不需要 Teacher checkpoint，但必须加载 Student actor 和 LSTM encoder。

## 2. checkpoint 要求

本文使用：

```text
runs_student/student_smoke_task14/full_0_r.pt
```

至少需要：

```text
actor_architecture_state_dict
actor_distribution_state_dict
prop_latent_encoder_state_dict
obs_spec.student_total_dim = 517
obs_spec.history_dim = 44
obs_spec.history_len = 10
obs_spec.prop_latent_dim = 26
obs_spec.aff_vec_dim = 18
student_actor_obs_dim = 88
action_dim = 13
```

不能使用 Teacher checkpoint；Teacher checkpoint 没有 `prop_latent_encoder_state_dict`。

## 3. Headless smoke

```bash
conda activate grasp
cd /home/windsky/project/Grasp
python scripts/play_student.py \
  --task Grasp-Student-Direct-v0 \
  --num_envs 1 \
  --episodes 1 \
  --checkpoint runs_student/student_smoke_task14/full_0_r.pt \
  --seed 1 \
  --lift_delta_z 0.002 \
  --device cuda:0 \
  --headless \
  --disable_fabric
```

预期：

```text
loaded Student playback checkpoint
episode=0 success_rate=... mean_lift_height=...
student multi-step evaluation summary
episodes=1
trials=1
success_rate=...
mean_lift_height=...
mean_affordance_reward=...
mean_lift_success=...
mean_teacher_reward=...
```

## 4. GUI 播放

```bash
python scripts/play_student.py \
  --task Grasp-Student-Direct-v0 \
  --num_envs 1 \
  --episodes 10 \
  --checkpoint runs_student/student_smoke_task14/full_0_r.pt \
  --seed 1 \
  --lift_delta_z 0.002 \
  --device cuda:0 \
  --disable_fabric
```

## 5. 参数含义

| 参数 | 含义 |
|---|---|
| `--checkpoint` | 必填，Student checkpoint |
| `--num_envs 1` | 并行评估环境数 |
| `--episodes 10` | 每个环境评估 episode 数 |
| `--lift_delta_z 0.002` | 每个 lift step 的末端 Z 增量，单位 m |
| `--seed 1` | reset 种子入口 |
| `--headless` | 加上后不显示 GUI |
| `--device cuda:0` | 使用第一张 GPU |

## 6. Student 每个抓取 step 的数据流

```text
env observation policy [N,517]
        │
        ├── history [N,440] -> reshape [N,10,44]
        │                       -> LSTM latent [N,26]
        ├── latest history frame [N,44]
        └── env.last_affordance_vec [N,18]
                │
                ▼
Student actor observation [N,88]
                │
                ▼
actor.noiseless_action() [N,13]
                │
                ▼
env.step(action)
```

Teacher privileged tail 77 不拼回 Student actor。

## 7. 完整播放调用链

```text
AppLauncher
 -> load Student env/agent cfg
 -> episode 延长到 70+80+2
 -> gym.make()
 -> assert_multistep_environment()
 -> 检查 517/44/10/26/18/88/13
 -> build Student actor 88->13
 -> build LSTM encoder 10x44->26
 -> 严格加载 Student checkpoint
 -> actor.eval() + encoder.eval() + inference_mode()
 -> 每个 episode：
      env.reset()
      70 次 build_student_actor_observation() + noiseless action
      80 次 build_multistep_lift_action()
      统计 lift_height > 0.10
 -> 输出 success_rate/reward mean
```

## 8. Student affordance 的真实含义

当前第一版使用：

```python
env.last_affordance_vec
```

它来自模拟器中的物体点云最近点向量，不是 D435 相机图像预测。播放结果不能描述成已经完成 sim-to-real 视觉 Student。

## 9. reward 输出

Student 环境复用 Teacher reward。当前播放统计 6 个旧项；Task 12 完成后应扩展 contact/impulse/push 项。若 contact mean 未出现，检查播放脚本 `METRIC_NAMES`。

## 10. Teacher play 与 Student play 的区别

| 项目 | Teacher play | Student play |
|---|---|---|
| actor input | 77 | 88 |
| history LSTM | 无 | 10x44->26 |
| checkpoint encoder | 无 | 必须有 |
| grasp steps | 70 | 70 |
| lift steps | 30 | 80 |
| lift delta z 默认 | 0.005 | 0.002 |
| 动作 | 13 | 13 |

不能用 `play_teacher.py` 播放 Student checkpoint，也不能用 `play_student.py` 播放 Teacher checkpoint。

## 11. 常见错误

### 缺 `prop_latent_encoder_state_dict`

传入了 Teacher checkpoint 或不完整 Student checkpoint。

### Student history dimension mismatch

checkpoint 和环境必须同时是 history_dim=44、history_len=10。

### Student Actor observation mismatch

checkpoint 必须是 88 维 Student actor，不能加载 77 维 Teacher actor。

### `still depends on the forbidden OneStep path`

Student/Teacher 环境或配置仍依赖 OneStep。完成独立多步迁移，不删除检查。

### 成功率为 0

依次检查：

```text
Student checkpoint 是否完成训练
LSTM 是否严格加载
last_affordance_vec 是否 finite
13 维 action 是否变化
手是否接触物体
lift 阶段手部动作是否保持
```

## 12. 验收

- [ ] Student actor 与 LSTM 都严格加载。
- [ ] 使用 517 容器但 actor 只消费 88 维。
- [ ] 每个 grasp step 重新运行 LSTM 和 actor。
- [ ] 前 70 steps 不使用 Teacher action。
- [ ] 后 80 steps 保持最后手部动作并抬升 FR3。
- [ ] 输出 trials、success_rate 和 mean_lift_height。
- [ ] 不调用 OneStep reference trajectory。
