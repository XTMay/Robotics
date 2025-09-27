#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫地机器人强化学习快速测试版本
=============================

这是扫地机器人项目的快速测试版本，用于验证代码正确性
- 减少训练轮数到50轮
- 简化网络结构
- 更频繁的路径保存
"""

import sys
import os

# 添加主项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from 扫地机器人强化学习演示项目 import *

def quick_test():
    """快速测试函数"""
    print("=" * 50)
    print("扫地机器人强化学习快速测试")
    print("=" * 50)

    # 创建环境（更小的房间）
    print("创建测试环境...")
    env = VacuumCleanerEnvironment(room_size=8, num_obstacles=3,
                                   dirt_density=0.2, max_battery=50)
    state_size = len(env.get_state())
    action_size = len(env.actions)
    print(f"   房间大小: {env.room_size}x{env.room_size}")
    print(f"   状态维度: {state_size}")
    print(f"   动作数量: {action_size}")

    # 创建智能体（简化参数）
    print("创建智能体...")
    agent = VacuumAgent(state_size, action_size,
                       learning_rate=2e-3, gamma=0.95,
                       epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.95)

    # 训练（少量轮次）
    print("开始快速训练...")
    trainer = VacuumTrainingManager(env, agent, save_dir="quick_test_results")
    training_stats = trainer.train(num_episodes=50, save_interval=10, render_interval=10)

    print("训练完成！")
    print(f"最终平均奖励: {np.mean(training_stats['episode_rewards'][-10:]):.1f}")
    print(f"最终覆盖率: {np.mean(training_stats['coverage_rates'][-10:])*100:.1f}%")

    # 简单评估
    print("进行评估...")
    eval_results = trainer.evaluate(num_episodes=5)

    print("快速测试完成！")
    print(f"评估平均奖励: {eval_results['avg_reward']:.1f}")
    print(f"评估覆盖率: {eval_results['avg_coverage_rate']*100:.1f}%")

    print(f"\n检查生成的文件:")
    print(f"- quick_test_results/episode_paths/ - 路径图")

if __name__ == "__main__":
    quick_test()