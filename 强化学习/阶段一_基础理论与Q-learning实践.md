# 阶段一：基础理论与Q-learning实践

## 第1讲：强化学习基础概念

### 1.1 什么是强化学习？

强化学习（Reinforcement Learning, RL）是机器学习的一个重要分支，它研究智能体（Agent）如何在与环境的交互中学习最优的行为策略。

**核心思想**：
- 智能体通过试错（Trial and Error）的方式学习
- 从环境的反馈（奖励或惩罚）中改进决策
- 目标是最大化长期累积奖励

### 1.2 与其他机器学习方法的区别

| 特征 | 监督学习 | 无监督学习 | 强化学习 |
|------|----------|------------|----------|
| 数据类型 | 标注数据 | 无标注数据 | 交互数据 |
| 学习方式 | 从样本学习 | 发现模式 | 试错学习 |
| 反馈类型 | 即时准确反馈 | 无反馈 | 延迟奖励信号 |
| 目标 | 预测准确性 | 数据理解 | 长期奖励最大化 |

### 1.3 强化学习的核心要素

#### 1.3.1 智能体（Agent）
- 做出决策的主体
- 机器人控制系统
- 游戏AI等

#### 1.3.2 环境（Environment）
- 智能体所处的外部世界
- 物理世界、游戏场景等
- 会根据智能体的动作发生变化

#### 1.3.3 状态（State）
- 描述环境当前情况的信息
- 例如：机器人的位置、速度、传感器读数
- 记为 $s_t$

#### 1.3.4 动作（Action）
- 智能体可以执行的行为
- 例如：向前移动、转向、停止
- 记为 $a_t$

#### 1.3.5 奖励（Reward）
- 环境对智能体动作的反馈
- 正奖励：好的行为
- 负奖励：不好的行为
- 记为 $r_t$

### 1.4 马尔可夫决策过程（MDP）

强化学习问题通常建模为马尔可夫决策过程：

**定义**：MDP = $(S, A, P, R, \gamma)$

- $S$：状态空间
- $A$：动作空间
- $P$：状态转移概率 $P(s'|s,a)$
- $R$：奖励函数 $R(s,a,s')$
- $\gamma$：折扣因子 $[0,1]$

**马尔可夫性质**：
$$P(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1}|s_t, a_t)$$

未来状态只依赖于当前状态和动作，与历史无关。

### 1.5 策略、价值函数、Q函数

#### 1.5.1 策略（Policy）
策略 $\pi$ 定义了智能体在每个状态下选择动作的规则：

- **确定性策略**：$a = \pi(s)$
- **随机策略**：$a \sim \pi(\cdot|s)$

#### 1.5.2 价值函数（Value Function）
状态价值函数 $V^\pi(s)$：从状态 $s$ 开始，遵循策略 $\pi$ 的期望累积奖励

$$V^\pi(s) = \mathbb{E}_\pi[\sum_{t=0}^{\infty} \gamma^t r_{t+1} | s_0 = s]$$

#### 1.5.3 Q函数（Action-Value Function）
动作价值函数 $Q^\pi(s,a)$：在状态 $s$ 执行动作 $a$，然后遵循策略 $\pi$ 的期望累积奖励

$$Q^\pi(s,a) = \mathbb{E}_\pi[\sum_{t=0}^{\infty} \gamma^t r_{t+1} | s_0 = s, a_0 = a]$$

**关系**：$V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s,a)$

### 1.6 贝尔曼方程

**状态价值函数的贝尔曼方程**：
$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

**Q函数的贝尔曼方程**：
$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]$$

**最优贝尔曼方程**：
$$Q^*(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')]$$

## 第2讲：Q-learning 算法原理

### 2.1 时间差分学习（TD Learning）

时间差分学习是强化学习的核心思想，它结合了：
- **蒙特卡洛方法**：使用实际经验
- **动态规划**：使用现有估计值

**TD误差**：
$$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$

**TD更新规则**：
$$V(s_t) \leftarrow V(s_t) + \alpha \delta_t$$

### 2.2 Q-learning 算法

Q-learning 是一种无模型（model-free）的时间差分学习算法。

**核心思想**：
- 直接学习最优Q函数 $Q^*(s,a)$
- 不需要知道环境模型 $P(s'|s,a)$ 和 $R(s,a,s')$
- 使用 $\epsilon$-贪婪策略平衡探索与利用

**Q-learning 更新规则**：
$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha [r_{t+1} + \gamma \max_{a'} Q(s_{t+1},a') - Q(s_t,a_t)]$$

**算法步骤**：
1. 初始化 Q表
2. 选择动作（$\epsilon$-贪婪）
3. 执行动作，观察奖励和新状态
4. 更新Q值
5. 重复步骤2-4

### 2.3 探索与利用（ε-贪婪策略）

**探索与利用的平衡**：
- **利用（Exploitation）**：选择当前认为最好的动作
- **探索（Exploration）**：尝试新的动作以发现更好的策略

**ε-贪婪策略**：
$$a_t = \begin{cases}
\arg\max_a Q(s_t,a) & \text{概率为 } 1-\epsilon \\
\text{随机动作} & \text{概率为 } \epsilon
\end{cases}$$

**ε衰减策略**：
- 开始时：$\epsilon$ 较大（多探索）
- 训练过程中：$\epsilon$ 逐渐减小（多利用）

```python
epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

### 2.4 超参数的作用

#### 2.4.1 学习率（Learning Rate）α
- **作用**：控制Q值更新的步长
- **范围**：$(0, 1]$
- **选择原则**：
  - 太大：学习不稳定
  - 太小：学习速度慢
  - 常用值：0.1 - 0.5

#### 2.4.2 折扣因子（Discount Factor）γ
- **作用**：控制对未来奖励的重视程度
- **范围**：$[0, 1]$
- **选择原则**：
  - 接近1：重视长期奖励
  - 接近0：重视即时奖励
  - 常用值：0.9 - 0.99

## 第3讲：实践练习 - FrozenLake 环境

### 3.1 环境介绍

FrozenLake 是 OpenAI Gym 中的经典环境：
- **目标**：从起点 S 到达目标 G
- **障碍**：避免掉入洞中 H
- **地面**：安全的冰面 F

```
SFFF
FHFH
FFFH
HFFG
```

**状态空间**：16个位置（4×4网格）
**动作空间**：4个动作（上下左右）
**奖励设置**：
- 到达目标：+1
- 掉入洞中：0
- 其他情况：0

### 3.2 完整的Q-learning实现

```python
import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1,
                 discount_factor=0.95, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # 初始化Q表
        self.q_table = np.zeros((state_size, action_size))

        # 记录训练过程
        self.training_rewards = []
        self.training_epsilons = []

    def choose_action(self, state):
        """ε-贪婪策略选择动作"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)  # 探索
        else:
            return np.argmax(self.q_table[state])  # 利用

    def learn(self, state, action, reward, next_state, done):
        """Q-learning 更新规则"""
        current_q = self.q_table[state, action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])

        # 更新Q值
        self.q_table[state, action] += self.learning_rate * (target_q - current_q)

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_agent(episodes=1000):
    """训练Q-learning智能体"""
    env = gym.make('FrozenLake-v1', is_slippery=False)  # 确定性环境

    agent = QLearningAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )

    scores = []

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # 处理新版本gym的返回格式

        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, info = env.step(action)

            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        scores.append(total_reward)
        agent.training_rewards.append(total_reward)
        agent.training_epsilons.append(agent.epsilon)

        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")

    env.close()
    return agent

def test_agent(agent, episodes=100):
    """测试训练好的智能体"""
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')

    test_scores = []

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        total_reward = 0
        done = False

        while not done:
            action = np.argmax(agent.q_table[state])  # 贪婪策略，不再探索
            next_state, reward, done, truncated, info = env.step(action)

            state = next_state
            total_reward += reward

        test_scores.append(total_reward)

    env.close()
    return test_scores

def visualize_training(agent):
    """可视化训练过程"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 奖励曲线
    ax1.plot(agent.training_rewards)
    ax1.set_title('Training Rewards Over Episodes')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')

    # 移动平均
    window_size = 100
    if len(agent.training_rewards) >= window_size:
        moving_avg = np.convolve(agent.training_rewards,
                               np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(agent.training_rewards)),
                moving_avg, color='red', linewidth=2, label='Moving Average')
    ax1.legend()

    # 2. Epsilon衰减
    ax2.plot(agent.training_epsilons)
    ax2.set_title('Epsilon Decay Over Episodes')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')

    # 3. Q表热力图
    sns.heatmap(agent.q_table, annot=True, fmt='.2f', ax=ax3)
    ax3.set_title('Final Q-Table Heatmap')
    ax3.set_xlabel('Action')
    ax3.set_ylabel('State')

    # 4. 最优策略
    policy = np.argmax(agent.q_table, axis=1)
    policy_grid = policy.reshape(4, 4)
    action_symbols = ['↑', '↓', '←', '→']

    # 创建策略可视化
    policy_display = np.array([[action_symbols[action] for action in row]
                              for row in policy_grid])

    im = ax4.imshow(np.ones((4,4)), cmap='Blues', alpha=0.3)
    ax4.set_title('Learned Policy')

    for i in range(4):
        for j in range(4):
            ax4.text(j, i, policy_display[i,j],
                    ha='center', va='center', fontsize=20)

    ax4.set_xticks(range(4))
    ax4.set_yticks(range(4))
    ax4.set_xlabel('Column')
    ax4.set_ylabel('Row')

    plt.tight_layout()
    plt.show()

def analyze_q_values(agent):
    """分析Q值的收敛性"""
    print("=== Q-Table 分析 ===")
    print(f"Q表维度: {agent.q_table.shape}")
    print(f"最大Q值: {np.max(agent.q_table):.3f}")
    print(f"最小Q值: {np.min(agent.q_table):.3f}")
    print(f"Q值标准差: {np.std(agent.q_table):.3f}")

    print("\n各状态的最优动作:")
    for state in range(agent.state_size):
        best_action = np.argmax(agent.q_table[state])
        best_q_value = agent.q_table[state, best_action]
        action_names = ['Up', 'Down', 'Left', 'Right']
        print(f"State {state}: {action_names[best_action]} (Q={best_q_value:.3f})")

# 主训练和测试流程
if __name__ == "__main__":
    print("开始训练 Q-learning 智能体...")
    agent = train_agent(episodes=1000)

    print("\n训练完成！开始可视化...")
    visualize_training(agent)

    print("\n分析 Q-Table...")
    analyze_q_values(agent)

    print("\n测试智能体性能...")
    test_scores = test_agent(agent, episodes=100)
    success_rate = np.mean(test_scores)
    print(f"测试成功率: {success_rate:.2%}")
```

### 3.3 超参数调优实验

```python
def hyperparameter_experiment():
    """超参数敏感性分析"""
    # 测试不同的学习率
    learning_rates = [0.01, 0.1, 0.3, 0.5, 0.7]
    discount_factors = [0.8, 0.9, 0.95, 0.99]

    results = {}

    for lr in learning_rates:
        for gamma in discount_factors:
            print(f"Testing lr={lr}, gamma={gamma}")

            env = gym.make('FrozenLake-v1', is_slippery=False)
            agent = QLearningAgent(
                state_size=env.observation_space.n,
                action_size=env.action_space.n,
                learning_rate=lr,
                discount_factor=gamma
            )

            # 快速训练
            for episode in range(500):
                state = env.reset()
                if isinstance(state, tuple):
                    state = state[0]

                done = False
                while not done:
                    action = agent.choose_action(state)
                    next_state, reward, done, truncated, info = env.step(action)
                    agent.learn(state, action, reward, next_state, done)
                    state = next_state

                agent.decay_epsilon()

            # 测试性能
            test_rewards = []
            for _ in range(100):
                state = env.reset()
                if isinstance(state, tuple):
                    state = state[0]

                total_reward = 0
                done = False
                while not done:
                    action = np.argmax(agent.q_table[state])
                    next_state, reward, done, truncated, info = env.step(action)
                    state = next_state
                    total_reward += reward
                test_rewards.append(total_reward)

            results[(lr, gamma)] = np.mean(test_rewards)
            env.close()

    # 可视化结果
    lr_values = []
    gamma_values = []
    performance_values = []

    for (lr, gamma), performance in results.items():
        lr_values.append(lr)
        gamma_values.append(gamma)
        performance_values.append(performance)

    # 创建热力图
    result_matrix = np.zeros((len(learning_rates), len(discount_factors)))
    for i, lr in enumerate(learning_rates):
        for j, gamma in enumerate(discount_factors):
            result_matrix[i][j] = results[(lr, gamma)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(result_matrix,
                xticklabels=discount_factors,
                yticklabels=learning_rates,
                annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Hyperparameter Sensitivity Analysis')
    plt.xlabel('Discount Factor (γ)')
    plt.ylabel('Learning Rate (α)')
    plt.show()

    # 找出最佳参数
    best_params = max(results, key=results.get)
    best_performance = results[best_params]
    print(f"最佳参数: lr={best_params[0]}, gamma={best_params[1]}")
    print(f"最佳性能: {best_performance:.3f}")

# 运行超参数实验
hyperparameter_experiment()
```

### 3.4 练习题

#### 练习1：环境变化
修改代码，在有滑动的FrozenLake环境中训练智能体：
```python
env = gym.make('FrozenLake-v1', is_slippery=True)
```
观察并分析性能变化。

#### 练习2：自定义奖励
设计新的奖励函数，例如：
- 每步给予小的负奖励（-0.01）以鼓励快速到达目标
- 距离目标越近奖励越大

#### 练习3：策略可视化
编写函数将学到的策略在4×4网格中可视化显示。

#### 练习4：收敛性分析
分析Q值的收敛过程，绘制不同状态-动作对的Q值变化曲线。

### 3.5 课后思考题

1. 为什么Q-learning能够收敛到最优策略？
2. 在什么情况下Q-learning可能不收敛？
3. 如何处理连续状态空间的问题？
4. ε-贪婪策略的缺点是什么？有什么改进方法？

### 3.6 拓展阅读

- Watkins, C. J. C. H. (1989). Learning from delayed rewards.
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction.
- 在线资源：OpenAI Gym 文档