# Teacher 多步 PPO 训练运行流程

## 1. 这份文档对应什么

```text
入口脚本：scripts/train_teacher.py
Gym task：Grasp-Teacher-Direct-v0
算法：RobustDexGrasp-style Teacher PPO
环境：IsaacLab DirectRLEnv 多步抓取
机器人：FR3 + Inspire tactile hand
动作：13 维
Teacher observation：77 维
输出目录：runs_teacher
```

Teacher 在每个 policy step 根据当前 77 维 observation 输出一个 13 维动作，连续执行 70 个 grasp steps，再进行一次 PPO update。

本路线不能调用：

```text
PPOOneStep
generate_reaching_plan_idx()
compute_reference_actions()
```

## 2. 运行前提

必须已经完成：

1. `Grasp-Teacher-Direct-v0` 注册。
2. Teacher 环境和配置已脱离 `grasp_env_onestep`。
3. Teacher observation 是 77 维。
4. FR3+Inspire action 是 13 维。
5. episode 长度与 `grasp_steps=70` 相同。
6. Task 12 的 contact sensor/reward 已接入。
7. Task 13 checkpoint 严格加载已完成。

检查 OneStep import：

```bash
cd /home/windsky/project/Grasp
rg -n \
  "from \.grasp_env_onestep import|from \.grasp_env_cfg_onestep import" \
  source/Grasp/Grasp/tasks/direct/grasp/grasp_env_teacher.py \
  source/Grasp/Grasp/tasks/direct/grasp/grasp_env_cfg_teacher.py
```

最终应无实际继承 import。仍有输出时不要开始正式训练。

## 3. 环境与资产

```text
conda 环境：grasp
IsaacLab：2.3.2
Isaac Sim：5.1.0
GPU：RTX 3070Ti
driver：580
项目根目录：/home/windsky/project/Grasp
机器人 USD：assets/robots/inspire_tac/fr3_inspire_tac_L_right_safety_visual_realistic.usd
物体目录：assets/union_ycb_unidex
默认物体列表：assets/union_ycb_unidex/union_ycb_debugset.yaml
点云目录：assets/union_ycb_unidex/pointclouds
```

Teacher 使用完整物体点云的最近点向量作为第一版 affordance 近似。

## 4. 第一次 smoke 命令

```bash
conda activate grasp
cd /home/windsky/project/Grasp
python scripts/train_teacher.py \
  --task Grasp-Teacher-Direct-v0 \
  --num_envs 2 \
  --max_iterations 1 \
  --run_name teacher_smoke_task14 \
  --seed 1 \
  --device cuda:0 \
  --headless \
  --disable_fabric
```

预期关键日志：

```text
update=0 average reward=...
mean std=...
saved checkpoint: .../runs_teacher/teacher_smoke_task14/full_0_r.pt
```

## 5. 参数含义

| 参数 | 含义 |
|---|---|
| `--task Grasp-Teacher-Direct-v0` | 选择 Teacher Gym task |
| `--num_envs 2` | 并行环境数，3070Ti 第一次先用 2 |
| `--max_iterations 1` | 目标 PPO 总 update 数，smoke 只运行 update 0 |
| `--run_name teacher_smoke_task14` | 输出目录名，已存在时会明确报错 |
| `--seed 1` | 环境、PyTorch 和 sampler 使用的种子入口 |
| `--device cuda:0` | 使用第一张 NVIDIA GPU |
| `--headless` | 不打开 Isaac Sim GUI |
| `--disable_fabric` | 禁用 Fabric 路径，和当前多资产验证方式一致 |
| `--torch_deterministic` | 可选，要求 PyTorch 尽量使用确定性算法 |

## 6. 训练调用链

```text
AppLauncher(args_cli)
    -> load_cfg_from_registry(task, "env_cfg_entry_point")
    -> load_cfg_from_registry(task, "ppo_teacher_cfg_entry_point")
    -> gym.make("Grasp-Teacher-Direct-v0")
    -> env.reset()
    -> build Actor 77 -> 13
    -> build Critic 77 -> 1
    -> build PPO rollout storage [num_envs,70]
    -> 每个 update：
         env.reset()
         70 次 actor.sample() -> env.step(action)
         PPO storage.add_transitions()
         PPO.update()
         enforce_minimum_std()
         actor.update()
    -> save full_update_r.pt
```

每个环境 step 都直接执行当前 Teacher action，不生成 OneStep reference trajectory。

## 7. Teacher reward

最终 reward 至少包含：

```text
affordance_reward
affordance_contact_reward
affordance_impulse_reward
table_reward
table_contact_reward
table_impulse_reward
arm_height_reward
arm_collision_reward
push_reward
lift_success
teacher_reward
```

Task 12 中 IsaacLab sensor 提供法向接触力，法向冲量近似为最近 20 个 physics steps 的力积分，不是完整 RaiSim 切向+法向冲量。

## 8. 输出文件

smoke 正常结束后：

```text
runs_teacher/teacher_smoke_task14/
├── env.yaml
├── agent.yaml
├── run.json
├── full_0_r.pt
└── Jul12_HH-MM-SS/
    └── events.out.tfevents...
```

时间子目录由 TensorBoard `SummaryWriter` 自动创建，实际名称以运行时间为准。

Teacher checkpoint 包含：

```text
actor_architecture_state_dict
actor_distribution_state_dict
critic_architecture_state_dict
optimizer_state_dict
obs_spec
action_dim
update
```

## 9. 查看 TensorBoard

另开终端：

```bash
conda activate grasp
cd /home/windsky/project/Grasp
tensorboard --logdir runs_teacher/teacher_smoke_task14 --port 6006
```

浏览器访问：

```text
http://localhost:6006
```

## 10. 恢复 Teacher 训练

先确认 checkpoint：

```text
runs_teacher/teacher_smoke_task14/full_0_r.pt
```

运行：

```bash
python scripts/train_teacher.py \
  --task Grasp-Teacher-Direct-v0 \
  --num_envs 2 \
  --max_iterations 2 \
  --checkpoint runs_teacher/teacher_smoke_task14/full_0_r.pt \
  --run_name teacher_resume_task14 \
  --seed 1 \
  --device cuda:0 \
  --headless \
  --disable_fabric
```

预期：

```text
loaded Teacher checkpoint: ... loaded_update=0, next_update=1
update=1 average reward=...
saved checkpoint: .../runs_teacher/teacher_resume_task14/full_1_r.pt
```

`--max_iterations 2` 是目标总 update 数，不是额外训练 2 次。

## 11. 较长训练示例

先根据 `nvidia-smi` 逐步增加环境数：

```bash
python scripts/train_teacher.py \
  --task Grasp-Teacher-Direct-v0 \
  --num_envs 64 \
  --max_iterations 50001 \
  --run_name teacher_multistep_001 \
  --seed 1 \
  --device cuda:0 \
  --headless \
  --disable_fabric
```

默认 `save_interval=500`，会保存 update 0、500、1000 等 checkpoint，正常结束时还会保存最后一个 update。

程序被强制结束时不保证执行最后保存；不要依赖 Ctrl+C 自动保存，使用周期 checkpoint。

## 12. 常见错误

### 仍依赖 OneStep

先完成独立多步 Teacher 环境和配置，不删除 MRO 检查。

### CUDA out of memory

把 `--num_envs 64` 降为 32、16 或 2。不要关闭 contact reward 后继续假装同一算法。

### run 目录已存在

更换 `--run_name`。脚本使用 `exist_ok=False`，不会覆盖旧实验。

### checkpoint action_dim 不匹配

Grasp Teacher 必须是 13 维。RobustDexGrasp 原始 22 维 checkpoint 和 OneStep `inspire.pt` 不能直接恢复。

### 点云缺失

Teacher 路线要求 `enable_point_cloud=True`，检查物体列表对应的 `.npy` 是否位于 `assets/union_ycb_unidex/pointclouds`。

## 13. Smoke 验收

- [ ] 无 OneStep 继承。
- [ ] observation 为 77 维。
- [ ] action 为 13 维。
- [ ] rollout 为 70 steps。
- [ ] reward 为 finite。
- [ ] contact raw 在真实接触时能非零。
- [ ] 生成 `full_0_r.pt`。
- [ ] checkpoint 可恢复到 update 1。
