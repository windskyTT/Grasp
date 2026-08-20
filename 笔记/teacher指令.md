# Grasp Teacher 教师策略运行指令

下面的指令仿照 `/home/windsky/project/Grasp/笔记/onestep指令.md` 编写，运行目录为 `/home/windsky/project/Grasp`，使用 IsaacLab 外部项目和 `grasp` conda 环境。

## 0. 当前代码状态

截至 2026-08-14，Teacher 链路尚不能按本文命令正常运行。不要把命令出现窗口或创建输出目录当作 Teacher 已复现成功。

当前至少有以下阻塞项：

1. `grasp_env_cfg_teacher.py` 在 `make_teacher_contact_sensor_cfgs()` 定义前调用它，会产生导入级 `NameError`。
2. `TEACHER_CONTACT_FILTER_PRIM_PATHS` 未定义。
3. 环境读取 `self.cfg.contact_sensors`，而配置实际放在 `self.cfg.teacher_reward.contact_sensors`。
4. `GraspTeacherEnv` 和 `GraspTeacherEnvCfg` 仍继承 OneStep。
5. Teacher reset 仍会调用 `generate_reaching_plan_idx()`。
6. `play_teacher.py` 会检查 MRO，因此当前播放命令必然以 `still inherits the forbidden OneStep environment` 退出。
7. resume 路径中 `load_checkpoint()` 引用了作用域内不存在的 `agent_cfg`。

完整核查和修改顺序见同目录：

```text
/home/windsky/project/相关文件/2026.8.14/教师策略修改1.md
```

以下训练、恢复和播放命令是“完成上述修复后”的标准运行方法；当前先保存作后续验收指令。

## 1. 每次运行前进入环境

```bash
cd /home/windsky/project/Grasp
conda activate grasp
```

第一次使用外部项目，或修改了 `source/Grasp` 的安装状态后，执行一次：

```bash
python -m pip install -e source/Grasp
```

本项目是 Isaac Sim 预编译二进制安装方式，代码默认从 Grasp 根目录运行。不要到 `scripts` 子目录中运行。

## 2. 正式运行前的 Teacher 独立性检查

先检查 Teacher 是否仍直接依赖 OneStep：

```bash
rg -n \
  "from \.grasp_env_onestep import|from \.grasp_env_cfg_onestep import" \
  source/Grasp/Grasp/tasks/direct/grasp/grasp_env_teacher.py \
  source/Grasp/Grasp/tasks/direct/grasp/grasp_env_cfg_teacher.py
```

修复完成后的预期结果：无输出。

再检查 Teacher reset 是否仍使用 OneStep reference API：

```bash
rg -n \
  "generate_reaching_plan_idx|compute_reference_actions|tracking_reference" \
  source/Grasp/Grasp/tasks/direct/grasp/grasp_env_teacher.py \
  source/Grasp/Grasp/tasks/direct/grasp/grasp_env_cfg_teacher.py
```

修复完成后的预期结果：Teacher 环境和配置中无调用；注释或明确的禁止检查可以保留。

检查 task 是否注册：

```bash
python scripts/list_envs.py --keyword Teacher
```

预期列表中包含：

```text
Grasp-Teacher-Direct-v0
```

只有以上三项通过后，才执行训练 smoke。

## 3. Teacher 训练 smoke

第一次只使用 2 个环境和 1 次 PPO update：

```bash
python scripts/train_teacher.py \
  --task Grasp-Teacher-Direct-v0 \
  --num_envs 2 \
  --max_iterations 1 \
  --run_name teacher_smoke_001 \
  --seed 1 \
  --device cuda:0 \
  --headless \
  --disable_fabric
```

修复后的预期关键输出：

```text
update=0 average reward=...
mean std=...
saved checkpoint: .../runs_teacher/teacher_smoke_001/full_0_r.pt
```

还必须人工确认日志中没有：

```text
NameError
still inherits the forbidden OneStep environment
generate_reaching_plan_idx
compute_reference_actions
observation contract mismatch
contact sensor history shape mismatch
nan
inf
```

Teacher smoke 的合格标准不是仅生成 checkpoint，还应满足：

- observation 的每个命名字段都为 finite。
- action=0 表示保持当前关节目标附近，而不是把 EEF 发送到世界原点。
- 真实手-物体接触时 contact/impulse term 非零。
- 无接触时 contact/impulse term 为零。
- rollout 为 70 个 policy steps。
- 训练过程不读取 OneStep reference trajectory。

## 4. 较长 Teacher 训练

RTX 3070Ti 先从 16 或 32 个环境开始，根据显存逐步增加。示例：

```bash
python scripts/train_teacher.py \
  --task Grasp-Teacher-Direct-v0 \
  --num_envs 32 \
  --max_iterations 50001 \
  --run_name teacher_multistep_001 \
  --seed 1 \
  --device cuda:0 \
  --headless \
  --disable_fabric
```

当前默认参数对应：

```text
算法：RobustDexGrasp-style PPO
网络：Actor [128,128]，Critic [128,128]
激活：LeakyReLU
grasp_steps：70
learning epochs：4
mini-batches：4
gamma：0.996
lambda：0.95
初始 std：1.0
最小 std：0.2
reward clip：-2.0
checkpoint interval：500 updates
```

注意：这些超参数已经迁入不等于环境语义已经迁入。正式长训练前必须完成 `教师策略修改1.md` 中的 observation、action、reward 和 reset 验收。

## 5. 训练输出

默认输出目录：

```text
runs_teacher/<run_name>/
```

正常应包含：

```text
runs_teacher/teacher_multistep_001/
├── env.yaml
├── agent.yaml
├── run.json
├── full_0_r.pt
├── full_500_r.pt
└── <TensorBoard 时间目录>/
    └── events.out.tfevents...
```

Teacher checkpoint 应包含：

```text
actor_architecture_state_dict
actor_distribution_state_dict
critic_architecture_state_dict
optimizer_state_dict
obs_spec
action_dim
update
```

## 6. 查看 TensorBoard

另开一个终端：

```bash
cd /home/windsky/project/Grasp
conda activate grasp
tensorboard --logdir runs_teacher/teacher_multistep_001 --port 6006
```

浏览器访问：

```text
http://localhost:6006
```

## 7. 恢复 Teacher 训练

先确认 checkpoint，例如：

```text
runs_teacher/teacher_multistep_001/full_500_r.pt
```

完成 `load_checkpoint()` 的 `agent_cfg` 作用域修复后，运行：

```bash
python scripts/train_teacher.py \
  --task Grasp-Teacher-Direct-v0 \
  --num_envs 32 \
  --max_iterations 1001 \
  --checkpoint runs_teacher/teacher_multistep_001/full_500_r.pt \
  --run_name teacher_multistep_resume_001 \
  --seed 1 \
  --device cuda:0 \
  --headless \
  --disable_fabric
```

预期从 update 501 开始，`--max_iterations 1001` 表示目标总 update 数，不是额外再训练 1001 次。

预期日志：

```text
loaded Teacher checkpoint: ..., loaded_update=500, next_update=501
update=501 average reward=...
```

当前未修复版本不要使用 `--checkpoint`，否则会在 `load_checkpoint()` 中因 `agent_cfg` 未定义而失败。

## 8. Teacher Headless 播放与评估

完成独立多步环境修复后，用 smoke checkpoint 测试：

```bash
python scripts/play_teacher.py \
  --task Grasp-Teacher-Direct-v0 \
  --num_envs 1 \
  --episodes 1 \
  --checkpoint runs_teacher/teacher_smoke_001/full_0_r.pt \
  --seed 1 \
  --lift_delta_z 0.005 \
  --device cuda:0 \
  --headless \
  --disable_fabric
```

预期流程：

```text
reset
-> 70 次 deterministic Teacher policy action
-> 保持最后的 Inspire 手指动作
-> 30 次 lift action
-> 统计 object z 增量是否大于 0.10 m
```

预期汇总：

```text
teacher multi-step evaluation summary
episodes=1
trials=1
success_rate=...
mean_lift_height=...
```

完成 reward metric 补齐后，还应输出：

```text
mean_affordance_reward
mean_affordance_contact_reward
mean_affordance_impulse_reward
mean_table_reward
mean_table_contact_reward
mean_table_impulse_reward
mean_arm_height_reward
mean_arm_collision_reward
mean_push_reward
mean_teacher_reward
```

## 9. Teacher GUI 播放

GUI 模式去掉 `--headless`，先只用 1 个环境：

```bash
python scripts/play_teacher.py \
  --task Grasp-Teacher-Direct-v0 \
  --num_envs 1 \
  --episodes 10 \
  --checkpoint runs_teacher/teacher_multistep_001/full_500_r.pt \
  --seed 1 \
  --lift_delta_z 0.005 \
  --device cuda:0 \
  --disable_fabric
```

GUI 中重点检查：

1. reset 后手位于物体 affordance 附近，姿态可达且没有初始穿模。
2. 前 70 steps 每一步都由 Teacher 重新推理，而不是回放 reference trajectory。
3. 动作是平滑的残差控制，没有不断把末端拉向世界原点。
4. 手指接触物体时 contact/impulse 指标发生变化。
5. lift 阶段保持最后的抓握，不重新张开手。
6. 物体抬升超过 `0.10 m` 时判定成功。

## 10. 定量评估尚缺的参数

当前 `play_teacher.py` 还没有以下 RobustDexGrasp quantitative evaluation 能力，因此现在没有等价命令：

- `--multi_object_list`。
- `--object_name` 单物体选择。
- seen/unseen category set 选择。
- `--lift_steps 30/100`。
- 逐物体 attempts、failures、failure rate。
- 失败物体名称列表。

这些参数补齐后，应分别提供：

```text
单物体 GUI visual evaluation
训练集批量 quantitative evaluation
seen set quantitative evaluation
unseen set quantitative evaluation
```

在 CLI 尚未实现前，不要把 `--multi_object_list` 或 `--lift_steps` 手工加到命令中；当前 parser 不支持它们。

## 11. checkpoint 不能混用

不能直接用于 Grasp Teacher 的 checkpoint：

```text
/home/windsky/project/RobustDexGrasp/raisimGymTorch/data_all/teacher/teacher_ckpt/full_12500_r.pt
/home/windsky/project/Grasp/checkpoints/ckpt/inspire.pt
任意 Student checkpoint
```

原因：

- RobustDexGrasp 原 Teacher 是 153 维 observation、22 维 UR5+Allegro action。
- Grasp 要使用 FR3+Inspire，并在最终 observation schema 固定后重新训练 Teacher。
- `inspire.pt` 属于 DemoGrasp OneStep 路线，不是 RobustDexGrasp 多步 Teacher checkpoint。

## 12. 最终运行验收

- [ ] Teacher 配置可导入。
- [ ] `Grasp-Teacher-Direct-v0` 可列出和构建。
- [ ] Teacher 无 OneStep 继承。
- [ ] reset 无 reference trajectory 调用。
- [ ] action 是 FR3+Inspire 多步残差关节控制。
- [ ] observation 字段语义与文档一致且全为 finite。
- [ ] reward 各项与 RobustDexGrasp 对齐或有明确等价替代。
- [ ] 70-step rollout 可完成 PPO update。
- [ ] 生成 `full_0_r.pt`。
- [ ] checkpoint 可从 update 0 恢复到 update 1。
- [ ] Headless play 完成 70-step grasp + lift。
- [ ] GUI play 动作、接触和抬升正常。
- [ ] 定量评估能输出总成功率和逐物体成功/失败统计。

当前代码在上述验收全部通过前，不应开始 50001 updates 的正式长训练。
