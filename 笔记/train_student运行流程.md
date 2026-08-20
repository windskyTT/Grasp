# Student 多步 DAgger/PPO 训练运行流程

## 1. Student 不是从零开始的普通 PPO

```text
入口脚本：scripts/train_student.py
Gym task：Grasp-Student-Direct-v0
Teacher checkpoint：必填
Student checkpoint：可选，仅用于恢复 Student
算法：frozen Teacher expert + DAgger imitation + PPO + LSTM history encoder
机器人：FR3 + Inspire
动作：13 维
输出目录：runs_student
```

Student 首次训练流程：

```text
Teacher actor 77->13 作为冻结 expert
Teacher critic 77->1 初始化 Student critic
Student actor 88->13
LSTM 10x44->26
```

## 2. 输入维度

环境 policy container：

```text
517
├── history 440 = 10 x 44
└── Teacher privileged tail 77
```

Student actor 输入：

```text
latest history frame 44
+ LSTM latent 26
+ Student affordance 18
= 88
```

Student critic 输入：

```text
Teacher privileged observation 77
```

第一版 `student_affordance` 来自模拟器物体点云生成的 `env.last_affordance_vec`，不是相机感知网络输出。

## 3. 运行前提

1. 已完成多步 Teacher 训练并得到 77/13 checkpoint。
2. Teacher/Student 环境和配置均脱离 OneStep。
3. Student history 是 10 帧，每帧 44 维。
4. Student actor input 是 88，不是 77。
5. `student_driven_ratio=1.0`，环境实际执行 Student action。
6. Task 13 checkpoint 严格加载已完成。

## 4. Teacher checkpoint

本文使用：

```text
runs_teacher/teacher_smoke_task14/full_0_r.pt
```

它必须是 Grasp 多步 Teacher checkpoint，不能使用：

```text
checkpoints/ckpt/inspire.pt
RobustDexGrasp 原始 full_12500_r.pt
Student checkpoint
```

## 5. 第一次 Student smoke

```bash
conda activate grasp
cd /home/windsky/project/Grasp
python scripts/train_student.py \
  --task Grasp-Student-Direct-v0 \
  --teacher_checkpoint runs_teacher/teacher_smoke_task14/full_0_r.pt \
  --num_envs 2 \
  --max_iterations 1 \
  --run_name student_smoke_task14 \
  --seed 1 \
  --device cuda:0 \
  --headless \
  --disable_fabric
```

预期日志：

```text
loaded frozen Teacher expert checkpoint
update=0 ppo_ratio=0.000000
prop mse loss=...
action mse loss=...
average reward=...
saved checkpoint: .../runs_student/student_smoke_task14/full_0_r.pt
```

## 6. 参数含义

| 参数 | 含义 |
|---|---|
| `--teacher_checkpoint` | 必填，冻结 Teacher expert 的来源 |
| `--student_checkpoint` | 可选，Student 完整恢复训练 checkpoint |
| `--num_envs 2` | 并行环境数，第一次先用 2 |
| `--max_iterations 1` | 目标总 update 数 |
| `--run_name student_smoke_task14` | 新 Student run 目录 |
| `--seed 1` | 随机种子入口 |
| `--torch_deterministic` | 可选，启用 PyTorch 确定性要求 |
| `--headless` | 不显示 GUI |
| `--device cuda:0` | 使用第一张 GPU |

## 7. Student 训练调用链

```text
AppLauncher
 -> load Student env/agent cfg
 -> gym.make("Grasp-Student-Direct-v0")
 -> 检查 517/77/44/10/26/18/13
 -> build Teacher expert 77->13
 -> build Student actor 88->13
 -> build Student critic 77->1
 -> build LSTM encoder 10x44->26
 -> load/freeze Teacher checkpoint
 -> create Dagger storage/optimizer
 -> 可选加载 Student resume checkpoint
 -> 每个 update：
      根据绝对 update 更新 ppo_ratio curriculum
      env.reset()
      70 次：
        history + LSTM + affordance -> Student action
        Teacher expert -> imitation label
        env.step(Student action)
        storage 保存 88 actor obs / 77 critic obs / 517 total obs
      Dagger.update()
      保存 Student checkpoint
```

## 8. 为什么 `student_driven_ratio=1.0`

当前 storage 保存 Student sampled action 和对应 log probability。若环境执行 Teacher action，却保存 Student log probability，会破坏 PPO on-policy 比率。

因此第一版：

```text
Teacher 提供 imitation label
Student action 实际驱动环境
```

## 9. Student checkpoint 内容

必须包含：

```text
actor_architecture_state_dict
actor_distribution_state_dict
critic_architecture_state_dict
optimizer_state_dict
prop_latent_encoder_state_dict
obs_spec
student_actor_obs_dim
action_dim
update
```

最关键区别：Student checkpoint 必须有：

```text
prop_latent_encoder_state_dict
```

否则 Student 播放时 26 维 latent 没有训练权重。

## 10. 输出目录

```text
runs_student/student_smoke_task14/
├── env.yaml
├── agent.yaml
├── run.json
├── full_0_r.pt
└── Jul12_HH-MM-SS/
    └── events.out.tfevents...
```

## 11. TensorBoard

```bash
conda activate grasp
cd /home/windsky/project/Grasp
tensorboard --logdir runs_student/student_smoke_task14 --port 6007
```

主要曲线：

```text
Loss/prop_mse
Loss/action_mse
Train/ppo_ratio
Train/average_reward
```

## 12. 恢复 Student 训练

Student resume 仍然必须传 Teacher checkpoint：

```bash
python scripts/train_student.py \
  --task Grasp-Student-Direct-v0 \
  --teacher_checkpoint runs_teacher/teacher_smoke_task14/full_0_r.pt \
  --student_checkpoint runs_student/student_smoke_task14/full_0_r.pt \
  --num_envs 2 \
  --max_iterations 2 \
  --run_name student_resume_task14 \
  --seed 1 \
  --device cuda:0 \
  --headless \
  --disable_fabric
```

预期：

```text
loaded frozen Teacher expert checkpoint
loaded Student checkpoint: ... loaded_update=0, next_update=1
update=1 ppo_ratio=0.000500 ...
saved checkpoint: .../runs_student/student_resume_task14/full_1_r.pt
```

恢复顺序是：

```text
先加载 Teacher expert
再用 Student checkpoint 覆盖 Student actor/critic/LSTM/optimizer
```

## 13. 较长训练示例

```bash
python scripts/train_student.py \
  --task Grasp-Student-Direct-v0 \
  --teacher_checkpoint runs_teacher/teacher_multistep_001/full_500_r.pt \
  --num_envs 64 \
  --max_iterations 50001 \
  --run_name student_multistep_001 \
  --seed 1 \
  --device cuda:0 \
  --headless \
  --disable_fabric
```

环境数必须根据 3070Ti 实际显存从 2 逐步增加。

## 14. 常见错误

### `--teacher_checkpoint` 缺失

Student 不是从零普通 PPO，Teacher expert 必填。

### Teacher action dim 是 22

传入了 RobustDexGrasp 原始 checkpoint，不是 Grasp FR3+Inspire Teacher。

### Student actor observation mismatch

正确值是 88。不要把 Teacher 77 维 observation 直接输入 Student actor。

### Student total observation mismatch

检查 10x44+77=517，不能改切片掩盖错误。

### optimizer parameter group mismatch

保存时与恢复时的 `update_mlp` 或网络结构不同。完整 resume 不能跳过 optimizer。

### CUDA out of memory

降低 `--num_envs`，不要删除 LSTM、点云或 contact sensor 后称为同一 Student 算法。

## 15. 验收

- [ ] Teacher expert 严格加载并冻结。
- [ ] Student actor 输入为 88。
- [ ] Student critic 输入为 77。
- [ ] history 为 10x44。
- [ ] LSTM latent 为 26。
- [ ] affordance 为 18。
- [ ] 环境执行 Student action。
- [ ] checkpoint 包含 LSTM state dict。
- [ ] Student checkpoint 可恢复到下一 update。
