from datetime import datetime
import os #前两行 用于生成带有时间戳的日志目录，方便记录不同批次的训练结果
import torch
import torch.nn as nn
import torch.optim as optim #上面三行 深度学习框架 PyTorch 及其神经网络和优化器模块，构成了算法的底层张量计算和梯度下降基础
from torch.utils.tensorboard import SummaryWriter #TensorBoard 的写入接口，用于可视化训练过程中的损失函数、学习率等指标。
from .storage import RolloutStorage #一个自定义的外部类（存储经验回放的缓冲区），用于在环境中收集和管理轨迹（Transitions），如状态、动作、奖励等


class PPO:
    def __init__(self, #此部分定义了 PPO 算法运行所需的所有核心组件和超参数
                 actor, #策略网络（Actor，负责输出动作概率分布）
                 critic, #价值网络（Critic，负责评估当前状态的预期收益）
                 num_envs,
                 num_transitions_per_env, #每次策略更新前，单个环境需要收集的交互步数。
                 num_learning_epochs, #训练轮数。每次智能体与环境交互收集满一个批次（Batch）的数据后，使用这批数据重复更新神经网络的次数
                 num_mini_batches, #微批次数量。在每个训练轮次（Epoch）中，将全量数据划分为若干个子集（Mini-batch），网络每次使用一个微批次计算梯度并更新一次参数
                 clip_param=0.2,  #策略更新截断阈值，通常表示为 \epsilon
                 gamma=0.998, #折扣因子（Discount Factor）
                 lam=0.95, #广义优势估计衰减参数（GAE Lambda），通常表示为 \lambda，取值范围 $[0, 1]$
                 value_loss_coef=0.5, #价值网络损失系数
                 entropy_coef=0.0, #策略熵系数
                 learning_rate=5e-4, #基础学习率，决定了参数更新时沿梯度方向迈出的步长
                 max_grad_norm=0.5, #最大梯度范数，用于梯度裁剪（Gradient Clipping）
                 learning_rate_schedule='adaptive', #从学习率调度策略
                 desired_kl=0.01, #目标 KL 散度，仅在自适应学习率开启时生效
                 use_clipped_value_loss=True, #价值损失截断
                 log_dir='run', #日志输出目录
                 device='cpu', #计算设备
                 shuffle_batch=True): #是否在划分微批次前随机打乱数据

        # PPO components 实例化 RolloutStorage 缓冲区，预先分配内存以存储状态 (obs)、动作 (action) 等数据。
        self.actor = actor
        self.critic = critic
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor.obs_shape, critic.obs_shape, actor.action_shape, device)

        #确定批次数据的采样方式。通常使用打乱（shuffle）以打破样本间的时序相关性，提高训练稳定性
        if shuffle_batch:
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        else:
            self.batch_sampler = self.storage.mini_batch_generator_inorder

        #使用 Adam 优化器，将 Actor 和 Critic 的参数合并到一个优化器中统一更新。设定计算设备（CPU 或 GPU）
        self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=learning_rate)
        self.device = device

        # env parameters 
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # PPO parameters
        self.clip_param = clip_param #PPO 的核心截断参数（通常为 0.2），用于限制新旧策略的差异
        self.num_learning_epochs = num_learning_epochs #每次收集完数据后，对这些数据进行重复学习的轮数
        self.num_mini_batches = num_mini_batches #将一次收集的数据划分为多少个小批量（Mini-batch）进行更新
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef #上面两个 损失函数中价值损失和熵奖励的权重。熵用于鼓励策略探索
        self.gamma = gamma 
        self.lam = lam #上面两个 折扣因子和 广义优势估计 (GAE) 的衰减系数，用于平衡偏差与方差。
        self.max_grad_norm = max_grad_norm #梯度裁剪的阈值，防止梯度爆炸
        self.use_clipped_value_loss = use_clipped_value_loss #是否对价值网络的更新也应用类似 PPO 的截断机制

        # Log 初始化日志系统。flush_secs=10 表示每 10 秒将数据写入磁盘
        self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0

        # ADAM
        self.learning_rate = learning_rate
        self.desired_kl = desired_kl
        self.schedule = learning_rate_schedule

        # temps 初始化临时变量，用于在 act 和 step 方法之间传递数据。初始化梯度爆炸标志位
        self.actions = None
        self.actions_log_prob = None
        self.actor_obs = None

        # tests
        self.is_exploding_gradient = False

    #功能: 根据当前观测状态，让策略网络输出动作
    '''
    torch.no_grad(): 在推理和数据收集阶段不需要计算梯度，此操作可显著节省内存和计算资源。
    网络输出不仅包括动作本身，还包括该动作的对数概率 (actions_log_prob)，这将在计算策略比率时使用
    '''
    def act(self, actor_obs, student_driven_ratio=0):
        self.actor_obs = actor_obs
        with torch.no_grad():
            self.actions, self.actions_log_prob = self.actor.sample(actor_obs)
        return self.actions
    
    #功能: 将环境反馈的一步（Transition）存入缓冲区。
    # 保存的数据包括：观测状态、价值状态、动作、策略分布的均值和标准差、奖励 (rews)、环境结束标志 (dones) 以及动作概率
    def step(self, value_obs, rews, dones):
        self.storage.add_transitions(
            self.actor_obs,
            value_obs,
            self.actions,
            self.actor.action_mean,
            self.actor.distribution.std,
            rews,
            dones,
            self.actions_log_prob,
        )
        
    #功能: 触发模型参数的更新。
    # 首先，使用当前的 Critic 网络预测最后一个状态的价值 (last_values)。这对于计算不完整轨迹的返回值（Bootstrapping）至关重要
    def update(self, actor_obs, value_obs, log_this_iteration, update):
        last_values = self.critic.predict(value_obs)

        # Learning step 调用缓冲区的方法，利用 GAE (Generalized Advantage Estimation) 计算每个状态的优势函数 (Advantage) 和目标回报 (Return)
        self.storage.compute_returns(last_values, self.critic, self.gamma, self.lam)
        mean_value_loss, mean_surrogate_loss, mean_entropy, infos = self._train_step(log_this_iteration) #_train_step: 调用核心训练逻辑，执行梯度下降
        self.storage.clear() #clear: 训练完成后，清空缓冲区，为下一轮数据收集做准备（因为 PPO 是同策略 On-Policy 算法，旧数据不能重复使用）

        #如果训练步骤返回空信息，说明检测到了梯度爆炸，标记状态并提前终止。否则，记录日志
        if infos is None:
            self.is_exploding_gradient = True
            return

        if log_this_iteration:
            self.log({**locals(), **infos, 'it': update})
        return mean_value_loss, mean_surrogate_loss, mean_entropy

    #功能: 将评估损失、替代损失、动作方差和学习率等关键指标写入 TensorBoard
    def log(self, variables):
        self.tot_timesteps += self.num_transitions_per_env * self.num_envs
        mean_std = self.actor.distribution.std.mean()
        self.writer.add_scalar('PPO/value_function', variables['mean_value_loss'], variables['it'])
        self.writer.add_scalar('PPO/surrogate', variables['mean_surrogate_loss'], variables['it'])
        self.writer.add_scalar('PPO/mean_noise_std', mean_std.item(), variables['it'])
        self.writer.add_scalar('PPO/learning_rate', self.learning_rate, variables['it'])
    
    #对收集到的一批数据，进行 num_learning_epochs 次重复迭代。
    # 每次迭代中，将数据划分为多个 Mini-batch 处理
    def _train_step(self, log_this_iteration):
        mean_value_loss = torch.zeros((), device=self.device)
        mean_surrogate_loss = torch.zeros((), device=self.device)
        mean_entropy = torch.zeros((), device=self.device)
        for epoch in range(self.num_learning_epochs):
            for actor_obs_batch, critic_obs_batch, actions_batch, old_sigma_batch, old_mu_batch, current_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
                    in self.batch_sampler(self.num_mini_batches):

                #使用当前最新的网络权重，重新评估之前收集到的状态和动作，得到新的对数概率、策略熵和价值估计。这是与旧策略数据进行对比的基础
                actions_log_prob_batch, entropy_batch = self.actor.evaluate(actor_obs_batch, actions_batch)
                value_batch = self.critic.evaluate(critic_obs_batch)

                # Adjusting the learning rate using KL divergence 自适应学习率 (基于 KL 散度)
                mu_batch = self.actor.action_mean
                sigma_batch = self.actor.distribution.std

                # KL 计算新旧策略（高斯分布）之间的 KL 散度解析解。KL 散度衡量了策略更新前后分布的差异程度。
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.no_grad():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        '''
                        如果策略变化过大（KL 散度超过目标值的两倍），则按比例缩小学习率，以收紧步长。
                        如果策略变化过小（KL 散度小于目标值的一半），则放大学习率，以加速训练。  
                        动态更新 Adam 优化器中的学习率参数。
                        '''
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.2)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.2)

                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                # Surrogate loss 策略替代损失 (Surrogate Loss) - PPO 核心
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                   1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss 计算价值网络的均方误差 (MSE)
                if self.use_clipped_value_loss:
                    value_clipped = current_values_batch + (value_batch - current_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step 总损失 = 策略损失 + 价值损失加权 - 熵奖励加权。由于优化器执行的是梯度下降，所以奖励项（熵）使用减号
                self.optimizer.zero_grad() #清空过往梯度
                loss.backward() #执行反向传播 loss.backward() 计算梯度
                nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.max_grad_norm)
                # clip_grad_norm_: 全局梯度裁剪。将梯度向量的 L2 范数限制在 max_grad_norm 内，这是强化学习中防止训练崩溃的标准操作
                self.optimizer.step() #更新网络参数

                # check exploding gradient 异常检测
                for name, parameters in self.actor.architecture.state_dict().items():
                    if "weight" in name:
                        w = parameters.detach()
                        if torch.isnan(w).any():
                            print("------------------------ find exploding gradient")
                            return None, None, None

                if log_this_iteration:
                    mean_value_loss += value_loss.detach()
                    mean_surrogate_loss += surrogate_loss.detach()
                    mean_entropy += entropy_batch.mean().detach()

        if log_this_iteration:
            num_updates = self.num_learning_epochs * self.num_mini_batches
            mean_value_loss = (mean_value_loss / num_updates).item()
            mean_surrogate_loss = (
                mean_surrogate_loss / num_updates
            ).item()
            mean_entropy = (mean_entropy / num_updates).item()

        return mean_value_loss, mean_surrogate_loss, mean_entropy, locals()

    def check_exploding_gradient(self): #提供一个外部接口，允许训练主循环查询是否发生了梯度爆炸，以便采取重置环境或重新初始化网络的策略
        return self.is_exploding_gradient
