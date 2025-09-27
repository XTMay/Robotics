# 阶段二：深度强化学习与进阶应用

## 第4讲：深度Q网络（DQN）理论

### 4.1 从Q-learning到DQN：函数逼近的必要性

#### 4.1.1 表格式Q-learning的局限性

传统Q-learning使用表格存储Q值：
- **内存问题**：状态空间爆炸（连续状态、高维状态）
- **泛化问题**：无法处理未见过的状态
- **学习效率**：每个状态需要单独学习

**示例**：
- 围棋：约 $10^{170}$ 种状态
- 机器人导航：连续的位置和角度信息
- 图像状态：$256^{84 \times 84 \times 3}$ 种可能

#### 4.1.2 函数逼近的解决方案

使用神经网络 $Q_\theta(s,a)$ 逼近真实Q函数 $Q^*(s,a)$：

**优势**：
- 处理高维/连续状态
- 状态间的泛化能力
- 参数共享，提高学习效率

### 4.2 DQN算法原理

#### 4.2.1 基本DQN结构

```
输入状态 s → 神经网络 → Q(s,a₁), Q(s,a₂), ..., Q(s,aₙ)
```

**网络结构**：
- **输入**：状态表示（如像素值、传感器数据）
- **隐藏层**：全连接层或卷积层
- **输出**：每个动作的Q值

#### 4.2.2 损失函数

DQN的训练目标是最小化贝尔曼误差：

$$L(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a))^2]$$

其中：
- $\theta$：当前网络参数
- $\theta^-$：目标网络参数
- $(s, a, r, s')$：经验样本

### 4.3 关键技术改进

#### 4.3.1 经验回放（Experience Replay）

**问题**：
- 样本之间高度相关
- 数据分布不稳定
- 样本利用率低

**解决方案**：
使用经验池存储历史经验 $(s_t, a_t, r_t, s_{t+1})$

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

**优势**：
- 打破数据相关性
- 重复利用珍贵经验
- 稳定训练过程

#### 4.3.2 目标网络（Target Network）

**问题**：
Q网络既用于选择动作，又用于计算目标值，导致训练不稳定

**解决方案**：
使用两个网络：
- **主网络** $Q_\theta(s,a)$：用于选择动作和更新
- **目标网络** $Q_{\theta^-}(s,a)$：用于计算目标Q值

**更新策略**：
- 主网络：每步更新
- 目标网络：每C步复制主网络参数

```python
# 目标网络更新
if step % target_update_freq == 0:
    target_network.load_state_dict(main_network.state_dict())
```

### 4.4 DQN算法流程

```python
# DQN伪代码
初始化经验回放池 D
初始化主网络 Q(s,a;θ) 和目标网络 Q(s,a;θ⁻)
θ⁻ ← θ

for episode = 1 to M:
    初始化状态 s₁
    for t = 1 to T:
        # ε-贪婪选择动作
        if random() < ε:
            aₜ = random_action()
        else:
            aₜ = argmax_a Q(sₜ, a; θ)

        # 执行动作
        执行 aₜ，观察 rₜ, sₜ₊₁

        # 存储经验
        D.store(sₜ, aₜ, rₜ, sₜ₊₁)

        # 训练网络
        if len(D) > batch_size:
            # 采样批次
            batch = D.sample(batch_size)

            # 计算目标Q值
            for each (s, a, r, s') in batch:
                if s' is terminal:
                    y = r
                else:
                    y = r + γ * max_a' Q(s', a'; θ⁻)

            # 更新主网络
            loss = (y - Q(s, a; θ))²
            θ ← θ - α * ∇loss

        # 更新目标网络
        if t % C == 0:
            θ⁻ ← θ
```

### 4.5 DQN的改进变体

#### 4.5.1 Double DQN (DDQN)

**问题**：DQN会过高估计Q值

**原因**：同一网络既选择动作又评估动作

**解决方案**：
- 主网络选择动作：$a^* = \arg\max_a Q_\theta(s', a)$
- 目标网络评估动作：$Q_{\theta^-}(s', a^*)$

$$y = r + \gamma Q_{\theta^-}(s', \arg\max_a Q_\theta(s', a))$$

#### 4.5.2 优先经验回放（Prioritized Experience Replay）

**思想**：重要的经验应该被更频繁地回放

**优先级**：TD误差的绝对值
$$p_i = |\delta_i| + \epsilon$$

**采样概率**：
$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

#### 4.5.3 Dueling DQN

**网络结构**：将Q网络分解为价值函数和优势函数

$$Q(s,a) = V(s) + A(s,a) - \frac{1}{|A|}\sum_{a'} A(s,a')$$

## 第5讲：神经网络基础回顾

### 5.1 PyTorch基础

#### 5.1.1 张量操作

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 创建张量
x = torch.randn(32, 4)  # batch_size=32, feature_dim=4
y = torch.randn(32, 2)  # batch_size=32, output_dim=2

# 基本操作
print(f"Shape: {x.shape}")
print(f"Device: {x.device}")
print(f"Dtype: {x.dtype}")

# GPU使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
```

#### 5.1.2 网络定义

```python
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建网络
net = DQN(state_size=4, action_size=2)
print(net)

# 参数统计
total_params = sum(p.numel() for p in net.parameters())
print(f"Total parameters: {total_params}")
```

### 5.2 训练循环

```python
def train_step(network, target_network, batch, optimizer, gamma=0.99):
    states, actions, rewards, next_states, dones = batch

    # 转换为张量
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.BoolTensor(dones)

    # 当前Q值
    current_q_values = network(states).gather(1, actions.unsqueeze(1))

    # 目标Q值
    next_q_values = target_network(next_states).max(1)[0].detach()
    target_q_values = rewards + (gamma * next_q_values * ~dones)

    # 计算损失
    loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

## 第6讲：实践练习 - CartPole环境

### 6.1 环境介绍

CartPole-v1是经典的控制任务：
- **目标**：保持杆子平衡尽可能长的时间
- **状态**：[位置, 速度, 角度, 角速度] (4维连续)
- **动作**：[向左推, 向右推] (2个离散动作)
- **奖励**：每个时间步 +1，杆子倒下结束
- **成功标准**：连续100个episode平均奖励 ≥ 195

### 6.2 完整的DQN实现

```python
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import copy

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=1e-3,
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, buffer_size=10000, batch_size=32):

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # 神经网络
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # 经验回放
        self.memory = ReplayBuffer(buffer_size)

        # 训练记录
        self.scores = []
        self.losses = []
        self.epsilons = []

        # 更新目标网络
        self.update_target_network()

    def update_target_network(self):
        """复制主网络参数到目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def choose_action(self, state):
        """ε-贪婪策略选择动作"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        return np.argmax(q_values.cpu().data.numpy())

    def store_experience(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        """从经验回放中学习"""
        if len(self.memory) < self.batch_size:
            return

        # 采样批次数据
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # 当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # 优化网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 记录损失
        self.losses.append(loss.item())

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn(episodes=1000, target_update_freq=10):
    """训练DQN智能体"""
    env = gym.make('CartPole-v1')

    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )

    scores_window = deque(maxlen=100)

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        total_reward = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, info = env.step(action)

            # 修改奖励函数以加速学习
            if done and total_reward < 499:
                reward = -10  # 提前结束的惩罚

            agent.store_experience(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward

            if done or truncated:
                break

        # 更新目标网络
        if episode % target_update_freq == 0:
            agent.update_target_network()

        # 记录分数
        scores_window.append(total_reward)
        agent.scores.append(total_reward)
        agent.epsilons.append(agent.epsilon)

        # 打印进度
        if episode % 100 == 0:
            print(f'Episode {episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {agent.epsilon:.3f}')

        # 检查是否解决了任务
        if np.mean(scores_window) >= 195.0:
            print(f'\\nEnvironment solved in {episode-100+1} episodes!\\tAverage Score: {np.mean(scores_window):.2f}')
            break

    env.close()
    return agent

def test_agent(agent, episodes=10, render=False):
    """测试训练好的智能体"""
    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    test_scores = []
    agent.epsilon = 0  # 关闭探索

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        total_reward = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, info = env.step(action)

            state = next_state
            total_reward += reward

            if done or truncated:
                break

        test_scores.append(total_reward)
        print(f'Test Episode {episode + 1}: Score = {total_reward}')

    env.close()
    return test_scores

def visualize_training(agent):
    """可视化训练过程"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 得分曲线
    ax1.plot(agent.scores)
    ax1.set_title('Training Scores')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')

    # 添加移动平均线
    if len(agent.scores) >= 100:
        moving_avg = []
        for i in range(99, len(agent.scores)):
            moving_avg.append(np.mean(agent.scores[i-99:i+1]))
        ax1.plot(range(99, len(agent.scores)), moving_avg, 'r-', linewidth=2, label='100-episode average')
        ax1.axhline(y=195, color='g', linestyle='--', label='Solved threshold')
        ax1.legend()

    # 2. 损失曲线
    if agent.losses:
        ax2.plot(agent.losses)
        ax2.set_title('Training Loss')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('MSE Loss')
        ax2.set_yscale('log')

    # 3. Epsilon衰减
    ax3.plot(agent.epsilons)
    ax3.set_title('Epsilon Decay')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')

    # 4. Q值分布
    # 可视化不同状态下的Q值
    sample_states = np.random.randn(1000, 4) * [2.4, 2, 0.2, 2]  # 采样一些状态
    sample_states = torch.FloatTensor(sample_states).to(agent.device)

    with torch.no_grad():
        q_values = agent.q_network(sample_states).cpu().numpy()

    ax4.hist(q_values[:, 0], alpha=0.5, label='Action 0 (Left)', bins=50)
    ax4.hist(q_values[:, 1], alpha=0.5, label='Action 1 (Right)', bins=50)
    ax4.set_title('Q-value Distribution')
    ax4.set_xlabel('Q-value')
    ax4.set_ylabel('Frequency')
    ax4.legend()

    plt.tight_layout()
    plt.show()

def analyze_network(agent):
    """分析网络参数和性能"""
    print("=== DQN 网络分析 ===")

    # 网络参数统计
    total_params = sum(p.numel() for p in agent.q_network.parameters())
    trainable_params = sum(p.numel() for p in agent.q_network.parameters() if p.requires_grad)

    print(f"总参数数量: {total_params}")
    print(f"可训练参数: {trainable_params}")

    # 层级信息
    print("\\n网络结构:")
    for name, param in agent.q_network.named_parameters():
        print(f"{name}: {param.shape}")

    # 权重分布
    print("\\n权重统计:")
    for name, param in agent.q_network.named_parameters():
        if 'weight' in name:
            print(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")

# 主训练和测试流程
if __name__ == "__main__":
    print("开始训练 DQN 智能体...")
    agent = train_dqn(episodes=1000)

    print("\\n可视化训练过程...")
    visualize_training(agent)

    print("\\n分析网络...")
    analyze_network(agent)

    print("\\n测试智能体...")
    test_scores = test_agent(agent, episodes=10)
    print(f"平均测试得分: {np.mean(test_scores):.2f}")

    # 保存模型
    torch.save(agent.q_network.state_dict(), 'dqn_cartpole.pth')
    print("模型已保存为 dqn_cartpole.pth")
```

### 6.3 超参数敏感性分析

```python
def hyperparameter_sensitivity():
    """DQN超参数敏感性分析"""

    # 测试参数组合
    learning_rates = [1e-4, 1e-3, 1e-2]
    batch_sizes = [16, 32, 64]
    target_update_freqs = [5, 10, 20]

    results = {}

    for lr in learning_rates:
        for batch_size in batch_sizes:
            for update_freq in target_update_freqs:
                print(f"Testing: lr={lr}, batch_size={batch_size}, update_freq={update_freq}")

                env = gym.make('CartPole-v1')
                agent = DQNAgent(
                    state_size=env.observation_space.shape[0],
                    action_size=env.action_space.n,
                    learning_rate=lr,
                    batch_size=batch_size
                )

                # 快速训练
                scores = []
                for episode in range(200):  # 减少训练时间
                    state = env.reset()
                    if isinstance(state, tuple):
                        state = state[0]

                    total_reward = 0
                    while True:
                        action = agent.choose_action(state)
                        next_state, reward, done, truncated, info = env.step(action)

                        agent.store_experience(state, action, reward, next_state, done)
                        agent.learn()

                        state = next_state
                        total_reward += reward

                        if done or truncated:
                            break

                    if episode % update_freq == 0:
                        agent.update_target_network()

                    scores.append(total_reward)

                # 计算性能指标
                final_performance = np.mean(scores[-50:])  # 最后50个episode的平均
                results[(lr, batch_size, update_freq)] = final_performance
                env.close()

    # 找出最佳参数
    best_params = max(results, key=results.get)
    best_performance = results[best_params]

    print(f"\\n最佳参数组合:")
    print(f"Learning Rate: {best_params[0]}")
    print(f"Batch Size: {best_params[1]}")
    print(f"Target Update Frequency: {best_params[2]}")
    print(f"最佳性能: {best_performance:.2f}")

    return results

# 运行超参数分析
sensitivity_results = hyperparameter_sensitivity()
```

### 6.4 改进实验

#### 6.4.1 Double DQN实现

```python
class DoubleDQNAgent(DQNAgent):
    def learn(self):
        """Double DQN学习方法"""
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Double DQN: 主网络选择动作，目标网络评估
        next_actions = self.q_network(next_states).max(1)[1].detach()
        next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()

        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

### 6.5 练习题

#### 练习1：网络架构实验
尝试不同的网络架构：
- 增加/减少隐藏层数量
- 改变隐藏层大小
- 尝试不同的激活函数

#### 练习2：奖励工程
设计不同的奖励函数：
- 基于杆子角度的连续奖励
- 基于位置的惩罚
- 形状奖励（shaping reward）

#### 练习3：实现Dueling DQN
```python
class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU()
        )

        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )

        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, action_size)
        )

    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        q_values = values + advantages - advantages.mean(dim=1, keepdim=True)
        return q_values
```

### 6.6 课后思考题

1. 为什么DQN需要经验回放和目标网络？
2. Double DQN如何解决过估计问题？
3. 在什么情况下DQN可能失效？
4. 如何将DQN扩展到连续动作空间？

### 6.7 拓展阅读

- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning.
- Van Hasselt, H., et al. (2016). Deep reinforcement learning with double q-learning.
- Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning.