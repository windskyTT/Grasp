# Teacher PPO 全 GPU 数据通路设计

## 目标

在不改变 RobustDexGrasp Teacher 控制语义的前提下，消除训练 rollout 和 PPO 更新中的逐步 CPU/GPU 数据往返，缩短 collection 时间和 ETA。

本次保持 `decimation=20`、`sim.dt=0.01`、70 步 rollout、Teacher 观测/动作维度、奖励、终止条件和 PPO 数学定义不变。速度结果不能直接与 `decimation=2` 的 `Isaac-Reach-Franka-v0` 比较。

## 根因

错误记录显示单轮 collection 为 21.370 秒，learning 仅为 0.106 秒，瓶颈在环境交互和 rollout 数据通路，不在 PPO 反向传播。

当前 Teacher 训练链路继承了 RobustDexGrasp 的 CPU/NumPy PPO 接口：

- 环境生成的 CUDA 观测每步转换为 NumPy。
- Actor 在 CUDA 上计算均值后立即复制到 CPU，并用 NumPy 采样动作。
- 动作每步从 NumPy 复制回 CUDA，再传给环境。
- 奖励、终止状态和截断状态每步复制到 CPU。
- RolloutStorage 以 NumPy 保存轨迹，更新前再整体复制到 CUDA。
- 每个 reward 日志项每步调用 `.item()`，造成多次 CUDA 同步。

## 方案比较

### 方案一：只减少日志同步

把 reward 日志从逐步 `.item()` 改为 GPU 累积。修改最小，但观测、动作和 PPO Storage 仍在 CPU/GPU 之间往返，只能解决部分开销。

### 方案二：Teacher PPO 全 GPU，保留控制语义（采用）

将 Actor 采样、训练接口、RolloutStorage、GAE 和 rollout 统计全部改为 CUDA Tensor，只在每轮输出日志时读取少量标量。该方案直接处理已定位的根因，同时不改变环境物理和策略控制周期。

### 方案三：全 GPU 并把 `decimation` 改为 2

表面 steps/s 会明显提高，但策略控制周期会从 0.2 秒变为 0.02 秒，70 步 episode 的物理时长也随之改变，偏离 RobustDexGrasp Teacher 语义，因此不采用。

## 数据流设计

每轮训练的数据流如下：

1. `env.reset()` 直接返回 CUDA observation Tensor。
2. `PPO.act()` 接收 Tensor；Actor 在 CUDA 上计算动作均值并用 Torch 正态噪声采样，返回 CUDA action Tensor 和 log probability Tensor。
3. action Tensor 直接传给 `env.step()`，不再经过 NumPy。
4. observation、action、reward、done、log probability、mean 和 std 直接写入 CUDA RolloutStorage。
5. Critic value、GAE returns 和 normalized advantages 全部在 CUDA 上计算。
6. PPO mini-batch 从同一 CUDA Storage 读取，更新过程不再重复构造 Tensor。
7. reward 项、episode reward 和 episode length 在 CUDA 上累计；每轮结束后仅将最终打印和 TensorBoard 所需标量传到 CPU。

## 修改边界

### Teacher PPO

- `ppo_teacher/sampler.py`：以 Torch CUDA Tensor 生成正态噪声和 log probability，不再使用 NumPy RNG。
- `ppo_teacher/module.py`：Actor 的 `sample()` 和 `noiseless_action()` 接收并返回 Tensor；分布直接使用设备上的 `std`；删除仅为刷新 CPU `std_np` 服务的 update 路径。
- `ppo_teacher/storage.py`：所有轨迹数组在目标 device 上直接分配为 Tensor；GAE、优势归一化和 mini-batch 索引保持在 GPU。
- `ppo_teacher/ppo.py`：`act()`、`step()` 和 `update()` 统一使用 Tensor，不再调用 `torch.from_numpy()`。

### Teacher 入口

- `scripts/train_teacher.py`：rollout 不再转换 observation、action、reward 和 done；统计在 GPU 上累计；结束时集中取标量。
- `scripts/play_teacher.py` 和 `teacher_multistep_eval.py`：使用 Tensor observation/action，以适配 Actor 的统一 Tensor 接口。

### 不修改

- 不修改 `env_cfg.py` 中的 `decimation=20`、仿真步长和环境数量。
- 不修改 Teacher 奖励、观测、重置、终止和接触语义。
- 不修改 Student、onestep 或其 PPO 实现。
- 不添加兼容层或第二套 NumPy/Tensor API。

## 随机性与 checkpoint

动作分布仍为对角高斯分布，均值、可训练标准差和 log probability 公式不变。采样随机源由 NumPy RNG 改为 Torch CUDA RNG，因此相同 seed 不保证与旧 NumPy 版本产生逐元素相同动作，但分布语义不变。

Actor 和 distribution 的可训练参数名称及形状保持不变，现有 Teacher checkpoint 的 state dict 结构不因本次数据通路修改而变化。

## 错误处理

保留现有 rollout 长度、checkpoint 合同和梯度爆炸检查。不会新增 fallback、自动 CPU 降级或额外兼容分支；设备错误应直接暴露。

## 验证范围

根据项目约束，不运行 test、训练、仿真或播放。修改后仅进行以下静态验证：

- 检查本次 Teacher 训练路径中不存在 `.cpu().numpy()`、`torch.from_numpy()` 或 NumPy rollout buffer。
- 检查 Student 和 onestep 没有被修改。
- 检查 Python 文件可解析。
- 检查 git diff 仅包含本方案所需修改，不覆盖工作区原有改动。

真实性能提升需要用户随后按原训练命令运行，并比较新的 collection 时间与 steps/s；由于 Teacher 使用 20 个物理子步、复杂接触和更多刚体，设计不承诺达到 Reach Demo 的 1926 steps/s。
