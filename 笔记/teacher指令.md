可以开始 Teacher 训练，现有训练入口、任务注册、PPO 配置、依赖和资产路径都已接通，不需要再添加代码。

推荐使用完整的 88 个环境：

```bash
cd /home/windsky/project/Grasp
conda activate grasp

python scripts/train_teacher.py \
  --task Grasp-Teacher-Direct-v0 \
  --num_envs 88 \
  --device cuda:0 \
  --headless
```

默认配置：

- 50,001 次迭代
- 每次 rollout 为 70 步
- 默认关闭 biased 扰动
- 默认关闭训练中定期评估
- 日志和 checkpoint 位于：

```text
/home/windsky/project/Grasp/runs_teacher/<任务名_时间>/
```

3070Ti 8GB 如果出现 CUDA 显存不足，可以降低为：

```bash
python scripts/train_teacher.py \
  --task Grasp-Teacher-Direct-v0 \
  --num_envs 44 \
  --device cuda:0 \
  --headless
```

启用 biased 扰动时增加：

```text
--biased
```

续训命令：

```bash
python scripts/train_teacher.py \
  --task Grasp-Teacher-Direct-v0 \
  --num_envs 88 \
  --device cuda:0 \
  --headless \
  --checkpoint /home/windsky/project/Grasp/runs_teacher/<运行目录>/full_500_r.pt
```

当前确认结果包括 Isaac Lab `v2.3.2`、Isaac Sim 5.1.0、Grasp editable 安装、训练依赖和训练入口静态检查通过。本轮遵守项目限制，没有实际启动训练；如果启动后产生报错，将内容写入 `错误.md` 后再让我读取处理。