# Grasp 运行指令

下面的指令是在 `/home/windsky/project/Grasp` 目录下运行的 IsaacLab 版本指令，对应 DemoGrasp README 里的 `Play the trained RL policies` 和 `Training` 两段。

DemoGrasp 原仓库是 IsaacGym + Hydra 写法，例如 `task=grasp`、`task.env.xxx=...`。当前 Grasp 是 IsaacLab 外部项目，主要改成 argparse 写法，例如 `--task Grasp-OneStep-Direct-v0`、`--arm_controller pose`。

## 0. 每次运行前先进入环境

```bash
cd /home/windsky/project/Grasp
conda activate grasp
```

如果第一次运行时出现 `ModuleNotFoundError: No module named 'Grasp'`，先在 Grasp 根目录安装一次外部项目包：

```bash
python -m pip install -e source/Grasp
```

## 1. Play the trained RL policies 对应指令

DemoGrasp 原始 play 指令使用 `ckpt/inspire.pt`。在当前 Grasp 中，已有 checkpoint 路径是：

```text
checkpoints/ckpt/inspire.pt
```

对应的 Grasp 播放指令：

```bash
python scripts/run_rl_grasp.py \
    --task Grasp-OneStep-Direct-v0 \
    --test \
    --num_envs 175 \
    --arm_controller pose \
    --observation_type "eefpose+objinitpose+objpcl" \
    --multi_object_list "union_ycb_unidex/union_ycb_debugset.yaml" \
    --randomize_tracking_reference \
    --randomize_grasp_pose \
    --tracking_reference_file "tasks/grasp_ref_inspire.pkl" \
    --episode_length 50 \
    --enable_point_cloud \
    --is_vision \
    --checkpoint "checkpoints/ckpt/inspire.pt" \
    --device cuda:0 \
    --headless
```

参数对应关系：

- `task=grasp`：在 Grasp 中可以用旧 alias `grasp`，但更推荐写正式任务名 `Grasp-OneStep-Direct-v0`。
- `train=PPOOneStep`：Grasp 的 `scripts/run_rl_grasp.py` 默认读取 `ppo_onestep_cfg_entry_point`，不用再写 `train=PPOOneStep`。
- `checkpoint='ckpt/inspire.pt'`：Grasp 中改成 `--checkpoint "checkpoints/ckpt/inspire.pt"`。
- `task.env.asset.multiObjectList=...`：Grasp 中改成 `--multi_object_list ...`。
- `task.env.armController=pose`：Grasp 中改成 `--arm_controller pose`。
- `task.env.observationType=...`：Grasp 中改成 `--observation_type ...`。
- `task.env.enablePointCloud=True`：Grasp 中改成 `--enable_point_cloud`。
- `train.params.is_vision=True`：Grasp 中改成 `--is_vision`。
- `headless=True`：Grasp 中改成 `--headless`。

如果要像 DemoGrasp README 里说的那样评估 unseen object categories，把物体列表换成：

```bash
--multi_object_list "union_ycb_unidex/test_set_unseen_cat.yaml"
```

如果想在 GUI 里看策略动作，不要用 175 个环境，先改成少量环境并去掉 `--headless`：

```bash
python scripts/run_rl_grasp.py \
    --task Grasp-OneStep-Direct-v0 \
    --test \
    --num_envs 1 \
    --arm_controller pose \
    --observation_type "eefpose+objinitpose+objpcl" \
    --multi_object_list "union_ycb_unidex/union_ycb_debugset.yaml" \
    --randomize_tracking_reference \
    --randomize_grasp_pose \
    --tracking_reference_file "tasks/grasp_ref_inspire.pkl" \
    --episode_length 50 \
    --enable_point_cloud \
    --is_vision \
    --checkpoint "checkpoints/ckpt/inspire.pt" \
    --device cuda:0 \
    --video \
    --video_length 50 \
    --headless
```

## 2. Training 对应指令

推荐先用当前仓库专门写好的训练入口：

```bash
python -u scripts/train_ppo_onestep.py \
    --task Grasp-OneStep-Direct-v0 \
    --num_envs 1024 \
    --max_iterations 2000 \
    --run_name inspire_debugset \
    --device cuda:0 \
    --headless
```

这个入口在代码里已经对应了 DemoGrasp Inspire hand 的训练设置：

- `armController=pose` 对应 `env_cfg.control.arm_controller = "pose"`。
- `trackingReferenceFile=tasks/grasp_ref_inspire.pkl` 对应默认参考轨迹 `source/Grasp/Grasp/reference/grasp_ref_inspire.pkl`。
- `trackingReferenceLiftTimestep=13` 对应默认 `tracking_reference_lift_timestep = 13`。
- `multiObjectList=union_ycb_unidex/union_ycb_debugset.yaml` 对应默认物体列表。
- `randomizeTrackingReference=True` 对应代码中开启 `randomize_tracking_reference`。
- `randomizeGraspPose=True` 对应代码中开启 `randomize_grasp_pose`。
- `resetDofPosRandomInterval=0.2` 对应配置默认值 `reset_dof_pos_random_interval = 0.2`。
- `observationType=eefpose+objinitpose+objpcl` 对应代码中设置的 observation。
- `episodeLength=40` 对应代码中设置的 episode 长度。
- `enablePointCloud=True` 对应代码中开启点云。
- `train.params.is_vision=True` 对应代码中设置 `agent_cfg.is_vision = True`。
- `enableRobotTableCollision=False` 对应代码中设置 `env_cfg.enable_robot_table_collision = False`。

DemoGrasp README 写的是 `num_envs=7000`。你这台 3070Ti 上建议先用 `1024` 或更小确认能跑，再逐步加大；如果显存足够，再改成更接近原始指令的：

```bash
python -u scripts/train_ppo_onestep.py \
    --task Grasp-OneStep-Direct-v0 \
    --num_envs 7000 \
    --max_iterations 2000 \
    --run_name inspire_debugset_7000env \
    --device cuda:0 \
    --headless
```

训练输出默认保存在：

```text
runs_ppo/<run_name>/
```

其中会保存：

- `env.yaml`：本次环境配置。
- `agent.yaml`：本次 PPOOneStep 配置。
- `config.json`：环境和算法配置汇总。
- `model_<iteration>.pt`：训练 checkpoint。

如果要继续训练某个已有模型，可以加：

```bash
--checkpoint runs_ppo/<run_name>/model_<iteration>.pt
```

## 3. 可视化场景指令

只想打开 Isaac Sim GUI 看 Grasp 场景、机器人、桌子和物体，不训练也不加载策略，用 zero agent：

```bash
python scripts/zero_agent.py \
    --task Grasp-OneStep-Direct-v0 \
    --num_envs 1 \
    --device cuda:0 \
    --disable_fabric
```

注意这里不要加 `--headless`，否则不会弹出 GUI。

如果想看 DemoGrasp 参考轨迹回放，而不是静态 zero action，可以用：

```bash
python scripts/run_rl_grasp.py \
    --task Grasp-OneStep-Direct-v0 \
    --num_envs 1 \
    --debug test_demo_replay \
    --arm_controller pose \
    --tracking_reference_file "tasks/grasp_ref_inspire.pkl" \
    --episode_length 50 \
    --multi_object_list "union_ycb_unidex/union_ycb_debugset.yaml" \
    --device cuda:0
```

这条指令不会训练，也不会加载 `inspire.pt`，只是用 `compute_reference_actions()` 播放参考抓取动作，适合先检查场景和机器人动作是否正常。
