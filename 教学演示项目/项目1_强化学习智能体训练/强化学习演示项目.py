#!/usr/bin/env python3
"""
项目1：强化学习智能体训练演示
===============================

本项目演示了完整的强化学习训练流程，包括：
1. 环境建模
2. 智能体设计
3. 训练过程
4. 性能评估
5. 结果可视化

作者：机器人课程组
日期：2024年
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict
import random
import json
import os
import time
from datetime import datetime
import matplotlib
import platform

# 设置中文字体 - 多重备用方案
def setup_chinese_fonts():
    """设置中文字体的函数"""
    import matplotlib.font_manager as fm

    # 首先尝试设置系统特定的字体
    if platform.system() == 'Darwin':  # macOS
        font_candidates = ['Arial Unicode MS', 'PingFang SC', 'Helvetica', 'DejaVu Sans']
    elif platform.system() == 'Windows':
        font_candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'DejaVu Sans']
    else:  # Linux
        font_candidates = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'SimHei', 'Liberation Sans']

    # 检查可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 选择第一个可用的字体
    selected_font = 'DejaVu Sans'  # 默认字体
    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            break

    # 设置字体
    plt.rcParams['font.sans-serif'] = [selected_font] + font_candidates
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10

    print(f"字体设置完成，使用字体: {selected_font}")
    return selected_font

# 调用字体设置函数
try:
    setup_chinese_fonts()
except Exception as e:
    print(f"字体设置警告: {e}")
    # 使用基本设置作为备用
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
sns.set_style("whitegrid")
matplotlib.rcParams['figure.figsize'] = [10, 8]
matplotlib.rcParams['figure.dpi'] = 100

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class GridWorldEnvironment:
    """
    网格世界环境

    特点：
    - 可配置大小的网格世界
    - 随机障碍物分布
    - 动态目标位置
    - 丰富的奖励机制
    """

    def __init__(self, size=10, obstacle_prob=0.2, dynamic_goal=False):
        self.size = size
        self.obstacle_prob = obstacle_prob
        self.dynamic_goal = dynamic_goal

        # 动作空间：上下左右停止
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
        self.action_effects = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

        # 统计信息
        self.episode_count = 0
        self.step_count = 0

        self.reset()

    def reset(self):
        """重置环境"""
        # 生成地图
        self.grid = np.random.random((self.size, self.size)) < self.obstacle_prob

        # 确保起点和终点没有障碍物
        self.start_pos = (0, 0)
        self.goal_pos = (self.size-1, self.size-1)
        self.grid[self.start_pos] = False
        self.grid[self.goal_pos] = False

        # 如果是动态目标，随机改变目标位置
        if self.dynamic_goal and self.episode_count > 0 and self.episode_count % 50 == 0:
            while True:
                new_goal = (np.random.randint(self.size//2, self.size),
                           np.random.randint(self.size//2, self.size))
                if not self.grid[new_goal] and new_goal != self.start_pos:
                    self.goal_pos = new_goal
                    break

        self.agent_pos = self.start_pos
        self.episode_count += 1
        self.episode_step = 0
        self.max_steps = self.size * self.size * 2

        return self.get_state()

    def get_state(self):
        """获取当前状态"""
        # 状态包括：智能体位置、目标位置、局部环境信息
        state = []

        # 智能体位置（归一化）
        state.extend([self.agent_pos[0] / self.size, self.agent_pos[1] / self.size])

        # 目标位置（归一化）
        state.extend([self.goal_pos[0] / self.size, self.goal_pos[1] / self.size])

        # 到目标的距离和方向
        dx = self.goal_pos[0] - self.agent_pos[0]
        dy = self.goal_pos[1] - self.agent_pos[1]
        distance = np.sqrt(dx**2 + dy**2) / (self.size * np.sqrt(2))
        direction = np.arctan2(dy, dx) / np.pi
        state.extend([distance, np.sin(direction), np.cos(direction)])

        # 周围环境（3x3网格）
        for i in range(-1, 2):
            for j in range(-1, 2):
                x, y = self.agent_pos[0] + i, self.agent_pos[1] + j
                if 0 <= x < self.size and 0 <= y < self.size:
                    if (x, y) == self.goal_pos:
                        state.append(1.0)  # 目标
                    elif self.grid[x, y]:
                        state.append(-1.0)  # 障碍物
                    else:
                        state.append(0.0)  # 空地
                else:
                    state.append(-1.0)  # 边界

        return np.array(state, dtype=np.float32)

    def step(self, action):
        """执行动作"""
        self.step_count += 1
        self.episode_step += 1

        # 执行动作
        dx, dy = self.action_effects[action]
        new_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

        # 检查边界和障碍物
        if (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size and
            not self.grid[new_pos]):
            self.agent_pos = new_pos
            collision = False
        else:
            collision = True

        # 计算奖励
        reward = self.calculate_reward(collision)

        # 检查终止条件
        done = self.is_done()

        return self.get_state(), reward, done, {'collision': collision}

    def calculate_reward(self, collision):
        """计算奖励"""
        # 到达目标
        if self.agent_pos == self.goal_pos:
            return 100.0

        # 碰撞惩罚
        if collision:
            return -10.0

        # 距离奖励（鼓励接近目标）
        distance = np.sqrt((self.goal_pos[0] - self.agent_pos[0])**2 +
                          (self.goal_pos[1] - self.agent_pos[1])**2)
        distance_reward = -distance * 0.1

        # 时间惩罚
        time_penalty = -0.01

        # 探索奖励（鼓励访问新位置）
        exploration_bonus = 0.01 if not hasattr(self, 'visited') else 0
        if not hasattr(self, 'visited'):
            self.visited = set()
        if self.agent_pos not in self.visited:
            self.visited.add(self.agent_pos)
            exploration_bonus = 0.1

        return distance_reward + time_penalty + exploration_bonus

    def is_done(self):
        """检查是否结束"""
        return (self.agent_pos == self.goal_pos or
                self.episode_step >= self.max_steps)

    def render(self, save_path=None):
        """可视化环境"""
        fig, ax = plt.subplots(figsize=(8, 8))

        # 绘制网格
        display_grid = np.zeros((self.size, self.size))
        display_grid[self.grid] = -1  # 障碍物
        display_grid[self.goal_pos] = 1  # 目标
        display_grid[self.agent_pos] = 0.5  # 智能体

        im = ax.imshow(display_grid, cmap='RdYlGn', vmin=-1, vmax=1)

        # 添加网格线
        ax.set_xticks(np.arange(-0.5, self.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.size, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

        # 标记特殊位置
        ax.text(self.agent_pos[1], self.agent_pos[0], 'A',
                ha='center', va='center', fontsize=16, fontweight='bold', color='blue')
        ax.text(self.goal_pos[1], self.goal_pos[0], 'G',
                ha='center', va='center', fontsize=16, fontweight='bold', color='red')

        ax.set_title(f'Grid World - Episode: {self.episode_count}, Step: {self.episode_step}')
        plt.colorbar(im, ax=ax, label='Cell Type')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

class DQNNetwork(nn.Module):
    """深度Q网络"""

    def __init__(self, state_size, action_size, hidden_sizes=[128, 128]):
        super(DQNNetwork, self).__init__()

        layers = []
        input_size = state_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_size = hidden_size

        layers.append(nn.Linear(input_size, action_size))

        self.network = nn.Sequential(*layers)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """DQN智能体"""

    def __init__(self, state_size, action_size, learning_rate=1e-3,
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # 神经网络
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # 经验回放
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.update_frequency = 4
        self.target_update_frequency = 100
        self.learn_step = 0

        # 训练记录
        self.training_history = defaultdict(list)

    def act(self, state, training=True):
        """选择动作"""
        if training and random.random() <= self.epsilon:
            return random.choice(range(self.action_size))

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        """学习"""
        if len(self.memory) < self.batch_size:
            return 0

        # 采样经验
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # 当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # 目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # 更新目标网络
        self.learn_step += 1
        if self.learn_step % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # 衰减探索率
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        # 记录训练信息
        self.training_history['loss'].append(loss.item())
        self.training_history['epsilon'].append(self.epsilon)
        self.training_history['q_value'].append(current_q_values.mean().item())

        return loss.item()

class TrainingManager:
    """训练管理器"""

    def __init__(self, env, agent, save_dir="results"):
        self.env = env
        self.agent = agent
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 训练记录
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': [],
            'average_q_values': [],
            'exploration_rate': [],
            'training_time': []
        }

    def train(self, num_episodes=1000, save_interval=100, render_interval=200):
        """训练智能体"""
        print(f"开始训练，总共 {num_episodes} 个回合")
        print(f"使用设备: {self.agent.device}")

        start_time = time.time()

        for episode in range(num_episodes):
            episode_start = time.time()
            state = self.env.reset()
            total_reward = 0
            steps = 0

            while True:
                action = self.agent.act(state)
                next_state, reward, done, info = self.env.step(action)

                self.agent.remember(state, action, reward, next_state, done)

                # 学习
                if steps % self.agent.update_frequency == 0:
                    loss = self.agent.learn()

                state = next_state
                total_reward += reward
                steps += 1

                if done:
                    break

            # 记录统计信息
            episode_time = time.time() - episode_start
            self.training_stats['episode_rewards'].append(total_reward)
            self.training_stats['episode_lengths'].append(steps)
            self.training_stats['exploration_rate'].append(self.agent.epsilon)
            self.training_stats['training_time'].append(episode_time)

            # 计算成功率
            if episode >= 99:
                recent_rewards = self.training_stats['episode_rewards'][-100:]
                success_count = sum(1 for r in recent_rewards if r > 50)
                success_rate = success_count / 100
                self.training_stats['success_rate'].append(success_rate)

            # 打印进度
            if episode % 100 == 0:
                avg_reward = np.mean(self.training_stats['episode_rewards'][-100:])
                avg_length = np.mean(self.training_stats['episode_lengths'][-100:])
                success_rate = self.training_stats['success_rate'][-1] if self.training_stats['success_rate'] else 0

                print(f"Episode {episode:4d} | "
                      f"Avg Reward: {avg_reward:6.1f} | "
                      f"Avg Length: {avg_length:5.1f} | "
                      f"Success Rate: {success_rate:5.1%} | "
                      f"Epsilon: {self.agent.epsilon:.3f}")

            # 保存模型
            if episode % save_interval == 0 and episode > 0:
                self.save_model(f"model_episode_{episode}.pth")

            # 渲染
            if episode % render_interval == 0:
                self.env.render(save_path=f"{self.save_dir}/episode_{episode}_visualization.png")

        training_duration = time.time() - start_time
        print(f"\n训练完成！总耗时: {training_duration:.1f}秒")

        # 保存最终模型和统计数据
        self.save_model("final_model.pth")
        self.save_training_stats()

        return self.training_stats

    def save_model(self, filename):
        """保存模型"""
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            'q_network_state_dict': self.agent.q_network.state_dict(),
            'target_network_state_dict': self.agent.target_network.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'epsilon': self.agent.epsilon,
            'training_history': self.agent.training_history
        }, filepath)

    def save_training_stats(self):
        """保存训练统计数据"""
        stats_file = os.path.join(self.save_dir, "training_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(self.training_stats, f, indent=2)

    def evaluate(self, num_episodes=100):
        """评估智能体性能"""
        print(f"\n开始评估，测试 {num_episodes} 个回合...")

        self.agent.epsilon = 0  # 关闭探索

        test_rewards = []
        test_lengths = []
        success_count = 0

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0

            while True:
                action = self.agent.act(state, training=False)
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                steps += 1

                if done:
                    if self.env.agent_pos == self.env.goal_pos:
                        success_count += 1
                    break

            test_rewards.append(total_reward)
            test_lengths.append(steps)

        # 计算评估指标
        avg_reward = np.mean(test_rewards)
        avg_length = np.mean(test_lengths)
        success_rate = success_count / num_episodes

        print(f"评估结果:")
        print(f"  平均奖励: {avg_reward:.2f}")
        print(f"  平均步数: {avg_length:.1f}")
        print(f"  成功率: {success_rate:.1%}")

        return {
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'success_rate': success_rate,
            'rewards': test_rewards,
            'lengths': test_lengths
        }

class ResultsAnalyzer:
    """结果分析器"""

    def __init__(self, training_stats, eval_results, save_dir="results"):
        self.training_stats = training_stats
        self.eval_results = eval_results
        self.save_dir = save_dir

    def generate_report(self):
        """生成完整的分析报告"""
        print("\n生成分析报告...")

        # 创建图表
        self.create_training_plots()
        self.create_evaluation_plots()
        self.create_summary_report()

        print(f"报告已保存到 {self.save_dir}/")

    def create_training_plots(self):
        """创建训练过程图表"""
        # 确保中文字体设置
        try:
            setup_chinese_fonts()
        except:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('强化学习训练过程分析 / RL Training Process Analysis', fontsize=16, fontweight='bold')

        # 奖励曲线
        episodes = range(len(self.training_stats['episode_rewards']))
        axes[0, 0].plot(episodes, self.training_stats['episode_rewards'], alpha=0.3, color='blue')
        if len(episodes) > 100:
            smoothed = np.convolve(self.training_stats['episode_rewards'],
                                 np.ones(100)/100, mode='valid')
            axes[0, 0].plot(range(99, len(episodes)), smoothed, linewidth=2, color='red')
        axes[0, 0].set_title('回合奖励变化 / Episode Rewards')
        axes[0, 0].set_xlabel('回合数 / Episodes')
        axes[0, 0].set_ylabel('总奖励 / Total Reward')
        axes[0, 0].grid(True, alpha=0.3)

        # 回合长度
        axes[0, 1].plot(episodes, self.training_stats['episode_lengths'], alpha=0.3, color='green')
        if len(episodes) > 100:
            smoothed = np.convolve(self.training_stats['episode_lengths'],
                                 np.ones(100)/100, mode='valid')
            axes[0, 1].plot(range(99, len(episodes)), smoothed, linewidth=2, color='darkgreen')
        axes[0, 1].set_title('回合长度变化 / Episode Length')
        axes[0, 1].set_xlabel('回合数 / Episodes')
        axes[0, 1].set_ylabel('步数 / Steps')
        axes[0, 1].grid(True, alpha=0.3)

        # 成功率
        if self.training_stats['success_rate']:
            success_episodes = range(99, len(self.training_stats['episode_rewards']))
            axes[0, 2].plot(success_episodes, self.training_stats['success_rate'],
                           linewidth=2, color='purple')
            axes[0, 2].set_title('成功率变化 / Success Rate')
            axes[0, 2].set_xlabel('回合数 / Episodes')
            axes[0, 2].set_ylabel('成功率 / Success Rate')
            axes[0, 2].set_ylim(0, 1)
            axes[0, 2].grid(True, alpha=0.3)

        # 探索率衰减
        axes[1, 0].plot(episodes, self.training_stats['exploration_rate'],
                       linewidth=2, color='orange')
        axes[1, 0].set_title('探索率衰减 / Exploration Rate')
        axes[1, 0].set_xlabel('回合数 / Episodes')
        axes[1, 0].set_ylabel('ε值 / Epsilon')
        axes[1, 0].grid(True, alpha=0.3)

        # 训练时间分布
        axes[1, 1].hist(self.training_stats['training_time'], bins=30,
                       alpha=0.7, color='cyan', edgecolor='black')
        axes[1, 1].set_title('单回合训练时间分布 / Training Time Distribution')
        axes[1, 1].set_xlabel('时间 (秒) / Time (s)')
        axes[1, 1].set_ylabel('频次 / Frequency')
        axes[1, 1].grid(True, alpha=0.3)

        # 最终性能统计
        final_rewards = self.training_stats['episode_rewards'][-100:]
        performance_data = [
            ('平均奖励', np.mean(final_rewards)),
            ('最大奖励', np.max(final_rewards)),
            ('最小奖励', np.min(final_rewards)),
            ('标准差', np.std(final_rewards))
        ]

        metrics, values = zip(*performance_data)
        bars = axes[1, 2].bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
        axes[1, 2].set_title('最终性能指标 / Final Performance (最后100回合)')
        axes[1, 2].set_ylabel('数值 / Value')

        # 添加数值标签
        for bar, value in zip(bars, values):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{value:.1f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/training_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()

    def create_evaluation_plots(self):
        """创建评估结果图表"""
        # 确保中文字体设置
        try:
            setup_chinese_fonts()
        except:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('智能体性能评估结果 / Agent Performance Evaluation', fontsize=16, fontweight='bold')

        # 测试奖励分布
        axes[0, 0].hist(self.eval_results['rewards'], bins=20, alpha=0.7,
                       color='lightblue', edgecolor='black')
        axes[0, 0].axvline(self.eval_results['avg_reward'], color='red',
                          linestyle='--', linewidth=2, label=f"平均值: {self.eval_results['avg_reward']:.1f}")
        axes[0, 0].set_title('测试奖励分布 / Test Reward Distribution')
        axes[0, 0].set_xlabel('奖励 / Reward')
        axes[0, 0].set_ylabel('频次 / Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 测试步数分布
        axes[0, 1].hist(self.eval_results['lengths'], bins=20, alpha=0.7,
                       color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(self.eval_results['avg_length'], color='red',
                          linestyle='--', linewidth=2, label=f"平均值: {self.eval_results['avg_length']:.1f}")
        axes[0, 1].set_title('测试步数分布')
        axes[0, 1].set_xlabel('步数')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 性能对比
        train_final = np.mean(self.training_stats['episode_rewards'][-100:])
        test_avg = self.eval_results['avg_reward']

        comparison_data = ['训练后期平均', '测试平均']
        comparison_values = [train_final, test_avg]

        bars = axes[1, 0].bar(comparison_data, comparison_values,
                             color=['lightcoral', 'lightblue'])
        axes[1, 0].set_title('训练vs测试性能对比')
        axes[1, 0].set_ylabel('平均奖励')

        for bar, value in zip(bars, comparison_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{value:.1f}', ha='center', va='bottom')

        # 成功率可视化
        success_rate = self.eval_results['success_rate']
        labels = ['成功', '失败']
        sizes = [success_rate, 1 - success_rate]
        colors = ['lightgreen', 'lightcoral']

        axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                      startangle=90)
        axes[1, 1].set_title(f'测试成功率: {success_rate:.1%}')

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/evaluation_results.png", dpi=300, bbox_inches='tight')
        plt.show()

    def create_summary_report(self):
        """创建总结报告"""
        report = f"""
强化学习智能体训练与评估报告
==============================

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. 训练概况
-----------
总训练回合数: {len(self.training_stats['episode_rewards'])}
总训练时间: {sum(self.training_stats['training_time']):.1f} 秒
平均每回合时间: {np.mean(self.training_stats['training_time']):.3f} 秒

2. 学习性能
-----------
最终探索率: {self.training_stats['exploration_rate'][-1]:.4f}
最后100回合平均奖励: {np.mean(self.training_stats['episode_rewards'][-100:]):.2f}
最后100回合平均步数: {np.mean(self.training_stats['episode_lengths'][-100:]):.1f}
最终成功率: {self.training_stats['success_rate'][-1] if self.training_stats['success_rate'] else 0:.1%}

3. 测试结果
-----------
测试回合数: {len(self.eval_results['rewards'])}
平均奖励: {self.eval_results['avg_reward']:.2f}
平均步数: {self.eval_results['avg_length']:.1f}
成功率: {self.eval_results['success_rate']:.1%}

4. 性能分析
-----------
奖励标准差: {np.std(self.eval_results['rewards']):.2f}
步数标准差: {np.std(self.eval_results['lengths']):.2f}
最优奖励: {np.max(self.eval_results['rewards']):.2f}
最少步数: {np.min(self.eval_results['lengths'])}

5. 结论
-------
{'智能体训练成功，测试性能良好。' if self.eval_results['success_rate'] > 0.8 else '智能体训练效果一般，建议调整超参数或增加训练时间。'}
"""

        with open(f"{self.save_dir}/summary_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)

        print(report)

def main():
    """主函数 - 完整的演示流程"""
    print("强化学习智能体训练演示")
    print("=" * 50)

    # 1. 创建环境
    print("1. 创建训练环境...")
    env = GridWorldEnvironment(size=8, obstacle_prob=0.15, dynamic_goal=False)
    state_size = len(env.get_state())
    action_size = len(env.actions)
    print(f"   环境大小: {env.size}x{env.size}")
    print(f"   状态维度: {state_size}")
    print(f"   动作数量: {action_size}")

    # 2. 创建智能体
    print("\n2. 创建DQN智能体...")
    agent = DQNAgent(state_size, action_size,
                    learning_rate=1e-3, gamma=0.99,
                    epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.998)
    print(f"   使用设备: {agent.device}")
    print(f"   网络结构: {state_size} -> 128 -> 128 -> {action_size}")

    # 3. 训练智能体
    print("\n3. 开始训练...")
    trainer = TrainingManager(env, agent, save_dir="results")
    training_stats = trainer.train(num_episodes=2000, save_interval=200, render_interval=400)

    # 4. 评估性能
    print("\n4. 评估智能体性能...")
    eval_results = trainer.evaluate(num_episodes=100)

    # 5. 生成分析报告
    print("\n5. 生成分析报告...")
    analyzer = ResultsAnalyzer(training_stats, eval_results, save_dir="results")
    analyzer.generate_report()

    print("\n演示完成！所有结果已保存到 'results' 目录")

if __name__ == "__main__":
    main()