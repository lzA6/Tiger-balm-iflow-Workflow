#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习智能体系统
Reinforcement Learning Agent System

作者: Quantum AI Team
版本: 5.1.0
日期: 2025-11-12
"""

import numpy as np
import random
import json
import pickle
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from pathlib import Path
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RLState:
    """强化学习状态"""
    agent_id: str
    environment_state: Dict[str, Any]
    internal_state: np.ndarray
    timestamp: float
    episode: int
    step: int

@dataclass
class RLAction:
    """强化学习动作"""
    action_id: str
    action_type: str
    parameters: Dict[str, Any]
    confidence: float
    expected_reward: float

@dataclass
class RLReward:
    """强化学习奖励"""
    immediate_reward: float
    delayed_reward: float
    social_reward: float
    intrinsic_reward: float
    total_reward: float
    reward_components: Dict[str, float]

class QLearningAgent:
    """Q学习智能体"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        初始化Q学习智能体
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            learning_rate: 学习率
            discount_factor: 折扣因子
            epsilon: 探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q表初始化
        self.q_table = defaultdict(lambda: np.zeros(action_dim))
        self.state_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        
        # 性能指标
        self.total_steps = 0
        self.episode_count = 0
        
    def encode_state(self, state: Dict[str, Any]) -> Tuple:
        """编码状态为可哈希的元组"""
        # 简单的状态编码策略
        encoded = []
        for key, value in sorted(state.items()):
            if isinstance(value, (int, float)):
                encoded.append(round(value, 3))
            elif isinstance(value, str):
                encoded.append(hash(value) % 1000)
            elif isinstance(value, (list, tuple)):
                encoded.append(tuple(round(v, 3) if isinstance(v, (int, float)) else v for v in value[:5]))
            else:
                encoded.append(str(value)[:10])
        
        return tuple(encoded)
    
    def select_action(self, state: Dict[str, Any], available_actions: List[int]) -> int:
        """选择动作（ε-贪婪策略）"""
        encoded_state = self.encode_state(state)
        
        # 探索
        if random.random() < self.epsilon:
            action = random.choice(available_actions)
        else:
            # 利用
            q_values = self.q_table[encoded_state]
            # 只考虑可用动作
            masked_q = np.full(self.action_dim, -np.inf)
            for action in available_actions:
                masked_q[action] = q_values[action]
            action = np.argmax(masked_q)
        
        self.total_steps += 1
        
        # 衰减探索率
        if self.total_steps % 100 == 0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return action
    
    def update_q_value(self, 
                      state: Dict[str, Any], 
                      action: int, 
                      reward: float, 
                      next_state: Dict[str, Any], 
                      done: bool):
        """更新Q值"""
        encoded_state = self.encode_state(state)
        encoded_next_state = self.encode_state(next_state)
        
        # Q学习更新规则
        current_q = self.q_table[encoded_state][action]
        
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[encoded_next_state])
        
        # Q(s,a) = Q(s,a) + α[r + γmaxQ(s',a') - Q(s,a)]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        self.q_table[encoded_state][action] = new_q
        
        # 记录历史
        self.state_history.append(encoded_state)
        self.reward_history.append(reward)
    
    def get_action_values(self, state: Dict[str, Any]) -> np.ndarray:
        """获取状态的动作值"""
        encoded_state = self.encode_state(state)
        return self.q_table[encoded_state].copy()
    
    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Q学习模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = defaultdict(lambda: np.zeros(self.action_dim), model_data['q_table'])
            self.epsilon = model_data['epsilon']
            self.total_steps = model_data['total_steps']
            self.episode_count = model_data['episode_count']
            
            logger.info(f"Q学习模型已从 {filepath} 加载")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")

class MultiAgentRLSystem:
    """多智能体强化学习系统"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化多智能体RL系统
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.agents = {}
        self.communication_graph = {}
        self.shared_memory = {}
        self.episode_rewards = defaultdict(list)
        self.global_step = 0
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                import yaml
                return yaml.safe_load(f)
        return {
            'agents': {
                'workflow_optimizer': {
                    'state_dim': 20,
                    'action_dim': 10,
                    'learning_rate': 0.1
                },
                'resource_manager': {
                    'state_dim': 15,
                    'action_dim': 8,
                    'learning_rate': 0.05
                },
                'quality_controller': {
                    'state_dim': 25,
                    'action_dim': 12,
                    'learning_rate': 0.15
                }
            },
            'communication': {
                'enabled': True,
                'message_passing': True,
                'knowledge_sharing': True
            }
        }
    
    def add_agent(self, agent_id: str, state_dim: int, action_dim: int, **kwargs):
        """添加智能体"""
        agent = QLearningAgent(state_dim, action_dim, **kwargs)
        self.agents[agent_id] = agent
        logger.info(f"添加智能体: {agent_id}")
    
    def initialize_agents(self):
        """初始化配置中的智能体"""
        for agent_id, config in self.config.get('agents', {}).items():
            self.add_agent(agent_id, **config)
    
    def coordinate_agents(self, global_state: Dict[str, Any]) -> Dict[str, int]:
        """协调多智能体决策"""
        actions = {}
        
        # 获取各智能体的局部状态
        local_states = self._extract_local_states(global_state)
        
        # 智能体间通信
        if self.config.get('communication', {}).get('enabled', False):
            self._agent_communication(local_states)
        
        # 各智能体决策
        for agent_id, agent in self.agents.items():
            if agent_id in local_states:
                local_state = local_states[agent_id]
                available_actions = self._get_available_actions(agent_id, local_state)
                
                action = agent.select_action(local_state, available_actions)
                actions[agent_id] = action
        
        return actions
    
    def _extract_local_states(self, global_state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """提取各智能体的局部状态"""
        local_states = {}
        
        for agent_id in self.agents.keys():
            # 基于智能体ID提取相关状态信息
            local_state = {
                'global_step': self.global_step,
                'agent_id': agent_id,
                'timestamp': time.time()
            }
            
            # 添加全局状态的相关子集
            if agent_id == 'workflow_optimizer':
                local_state.update({
                    'task_queue_length': global_state.get('task_queue_length', 0),
                    'system_load': global_state.get('system_load', 0.5),
                    'active_workflows': global_state.get('active_workflows', 0)
                })
            elif agent_id == 'resource_manager':
                local_state.update({
                    'cpu_usage': global_state.get('cpu_usage', 0.5),
                    'memory_usage': global_state.get('memory_usage', 0.5),
                    'available_resources': global_state.get('available_resources', 100)
                })
            elif agent_id == 'quality_controller':
                local_state.update({
                    'error_rate': global_state.get('error_rate', 0.0),
                    'test_coverage': global_state.get('test_coverage', 0.8),
                    'code_quality_score': global_state.get('code_quality_score', 0.8)
                })
            
            # 添加共享记忆
            if agent_id in self.shared_memory:
                local_state['shared_memory'] = self.shared_memory[agent_id]
            
            local_states[agent_id] = local_state
        
        return local_states
    
    def _get_available_actions(self, agent_id: str, state: Dict[str, Any]) -> List[int]:
        """获取智能体的可用动作"""
        if agent_id == 'workflow_optimizer':
            return list(range(10))  # 10种工作流优化动作
        elif agent_id == 'resource_manager':
            return list(range(8))   # 8种资源管理动作
        elif agent_id == 'quality_controller':
            return list(range(12))  # 12种质量控制动作
        else:
            return list(range(self.agents[agent_id].action_dim))
    
    def _agent_communication(self, local_states: Dict[str, Dict[str, Any]]):
        """智能体间通信"""
        if not self.config.get('communication', {}).get('message_passing', False):
            return
        
        # 简单的消息传递机制
        messages = {}
        
        for agent_id, state in local_states.items():
            # 生成消息（基于状态）
            message = {
                'sender': agent_id,
                'state_summary': {k: v for k, v in state.items() if isinstance(v, (int, float, str))},
                'timestamp': time.time()
            }
            messages[agent_id] = message
        
        # 分发消息到其他智能体
        for agent_id, agent in self.agents.items():
            if agent_id in self.shared_memory:
                # 更新共享记忆
                self.shared_memory[agent_id].update(messages)
            else:
                self.shared_memory[agent_id] = messages
    
    def update_agents(self, 
                     global_state: Dict[str, Any], 
                     actions: Dict[str, int], 
                     rewards: Dict[str, float], 
                     next_global_state: Dict[str, Any]):
        """更新所有智能体"""
        next_local_states = self._extract_local_states(next_global_state)
        current_local_states = self._extract_local_states(global_state)
        
        for agent_id, agent in self.agents.items():
            if agent_id in actions and agent_id in current_local_states and agent_id in next_local_states:
                # 更新Q值
                agent.update_q_value(
                    current_local_states[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    next_local_states[agent_id],
                    done=False
                )
                
                # 记录奖励
                self.episode_rewards[agent_id].append(rewards[agent_id])
        
        self.global_step += 1
    
    def calculate_rewards(self, 
                         global_state: Dict[str, Any], 
                         actions: Dict[str, int], 
                         next_global_state: Dict[str, Any]) -> Dict[str, RLReward]:
        """计算多智能体奖励"""
        rewards = {}
        
        for agent_id in self.agents.keys():
            if agent_id not in actions:
                continue
            
            # 即时奖励
            immediate_reward = self._calculate_immediate_reward(
                agent_id, global_state, actions[agent_id], next_global_state
            )
            
            # 延迟奖励（基于长期效果）
            delayed_reward = self._calculate_delayed_reward(
                agent_id, global_state, next_global_state
            )
            
            # 社会奖励（基于其他智能体的表现）
            social_reward = self._calculate_social_reward(
                agent_id, global_state, actions, next_global_state
            )
            
            # 内在奖励（探索奖励）
            intrinsic_reward = self._calculate_intrinsic_reward(
                agent_id, actions[agent_id]
            )
            
            # 总奖励
            total_reward = (immediate_reward * 0.4 + 
                          delayed_reward * 0.3 + 
                          social_reward * 0.2 + 
                          intrinsic_reward * 0.1)
            
            rewards[agent_id] = RLReward(
                immediate_reward=immediate_reward,
                delayed_reward=delayed_reward,
                social_reward=social_reward,
                intrinsic_reward=intrinsic_reward,
                total_reward=total_reward,
                reward_components={
                    'immediate': immediate_reward,
                    'delayed': delayed_reward,
                    'social': social_reward,
                    'intrinsic': intrinsic_reward
                }
            )
        
        return rewards
    
    def _calculate_immediate_reward(self, 
                                  agent_id: str, 
                                  state: Dict[str, Any], 
                                  action: int, 
                                  next_state: Dict[str, Any]) -> float:
        """计算即时奖励"""
        if agent_id == 'workflow_optimizer':
            # 基于工作流性能指标
            current_throughput = state.get('throughput', 0)
            next_throughput = next_state.get('throughput', 0)
            return (next_throughput - current_throughput) * 10
        
        elif agent_id == 'resource_manager':
            # 基于资源利用率
            current_efficiency = state.get('resource_efficiency', 0.5)
            next_efficiency = next_state.get('resource_efficiency', 0.5)
            return (next_efficiency - current_efficiency) * 5
        
        elif agent_id == 'quality_controller':
            # 基于质量指标
            current_quality = state.get('code_quality_score', 0.5)
            next_quality = next_state.get('code_quality_score', 0.5)
            return (next_quality - current_quality) * 8
        
        return 0.0
    
    def _calculate_delayed_reward(self, 
                                agent_id: str, 
                                state: Dict[str, Any], 
                                next_state: Dict[str, Any]) -> float:
        """计算延迟奖励"""
        # 基于趋势的奖励
        if len(self.episode_rewards[agent_id]) > 10:
            recent_rewards = self.episode_rewards[agent_id][-10:]
            trend = np.mean(recent_rewards[-5:]) - np.mean(recent_rewards[:5])
            return trend * 2
        
        return 0.0
    
    def _calculate_social_reward(self, 
                               agent_id: str, 
                               state: Dict[str, Any], 
                               actions: Dict[str, int], 
                               next_state: Dict[str, Any]) -> float:
        """计算社会奖励"""
        social_reward = 0.0
        
        # 协作奖励
        for other_agent_id, other_action in actions.items():
            if other_agent_id != agent_id:
                # 简单的协作机制：相似动作获得奖励
                if agent_id in actions and abs(actions[agent_id] - other_action) <= 2:
                    social_reward += 0.1
        
        # 全局性能奖励
        global_improvement = next_state.get('global_performance', 0) - state.get('global_performance', 0)
        social_reward += global_improvement * 0.5
        
        return social_reward
    
    def _calculate_intrinsic_reward(self, agent_id: str, action: int) -> float:
        """计算内在奖励（探索奖励）"""
        agent = self.agents[agent_id]
        
        # 基于动作选择频率的探索奖励
        if len(agent.state_history) > 0:
            recent_states = list(agent.state_history)[-100:]
            action_counts = defaultdict(int)
            
            for state in recent_states:
                q_values = agent.q_table[state]
                chosen_action = np.argmax(q_values)
                action_counts[chosen_action] += 1
            
            # 稀有动作获得更多奖励
            total_actions = sum(action_counts.values())
            if total_actions > 0:
                action_frequency = action_counts.get(action, 0) / total_actions
                intrinsic_reward = 1.0 - action_frequency  # 越稀有奖励越高
                return intrinsic_reward * 0.5
        
        return 0.0
    
    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """获取性能指标"""
        metrics = {}
        
        for agent_id, agent in self.agents.items():
            if agent_id in self.episode_rewards and len(self.episode_rewards[agent_id]) > 0:
                rewards = self.episode_rewards[agent_id]
                
                metrics[agent_id] = {
                    'average_reward': np.mean(rewards),
                    'reward_std': np.std(rewards),
                    'max_reward': np.max(rewards),
                    'min_reward': np.min(rewards),
                    'total_episodes': len(rewards),
                    'exploration_rate': agent.epsilon,
                    'total_steps': agent.total_steps
                }
                
                # 计算学习曲线斜率
                if len(rewards) > 20:
                    recent_avg = np.mean(rewards[-10:])
                    early_avg = np.mean(rewards[:10])
                    learning_slope = (recent_avg - early_avg) / len(rewards)
                    metrics[agent_id]['learning_slope'] = learning_slope
        
        return metrics
    
    def save_all_models(self, directory: str):
        """保存所有智能体模型"""
        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            filepath = directory_path / f"{agent_id}_model.pkl"
            agent.save_model(str(filepath))
        
        # 保存系统配置
        config_path = directory_path / "system_config.json"
        system_data = {
            'config': self.config,
            'global_step': self.global_step,
            'episode_rewards': dict(self.episode_rewards),
            'shared_memory': self.shared_memory
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(system_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"所有模型已保存到: {directory}")
    
    def load_all_models(self, directory: str):
        """加载所有智能体模型"""
        directory_path = Path(directory)
        
        for agent_id, agent in self.agents.items():
            filepath = directory_path / f"{agent_id}_model.pkl"
            if filepath.exists():
                agent.load_model(str(filepath))
        
        # 加载系统配置
        config_path = directory_path / "system_config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                system_data = json.load(f)
            
            self.global_step = system_data.get('global_step', 0)
            self.episode_rewards = defaultdict(list, system_data.get('episode_rewards', {}))
            self.shared_memory = system_data.get('shared_memory', {})
        
        logger.info(f"所有模型已从 {directory} 加载")

# 示例使用
def example_multi_agent_training():
    """示例多智能体训练"""
    # 创建多智能体系统
    system = MultiAgentRLSystem()
    system.initialize_agents()
    
    # 模拟训练环境
    for episode in range(100):
        global_state = {
            'task_queue_length': random.randint(1, 20),
            'system_load': random.random(),
            'active_workflows': random.randint(1, 5),
            'cpu_usage': random.random(),
            'memory_usage': random.random(),
            'available_resources': random.randint(50, 100),
            'error_rate': random.random() * 0.1,
            'test_coverage': random.random(),
            'code_quality_score': random.random(),
            'throughput': random.random() * 100,
            'resource_efficiency': random.random(),
            'global_performance': random.random()
        }
        
        # 智能体决策
        actions = system.coordinate_agents(global_state)
        
        # 模拟环境响应
        next_global_state = {
            'task_queue_length': max(0, global_state['task_queue_length'] - random.randint(0, 3)),
            'system_load': max(0, min(1, global_state['system_load'] + random.uniform(-0.1, 0.1))),
            'active_workflows': max(0, global_state['active_workflows'] + random.randint(-1, 2)),
            'cpu_usage': max(0, min(1, global_state['cpu_usage'] + random.uniform(-0.1, 0.1))),
            'memory_usage': max(0, min(1, global_state['memory_usage'] + random.uniform(-0.1, 0.1))),
            'available_resources': max(0, global_state['available_resources'] + random.randint(-10, 10)),
            'error_rate': max(0, min(1, global_state['error_rate'] + random.uniform(-0.01, 0.01))),
            'test_coverage': max(0, min(1, global_state['test_coverage'] + random.uniform(-0.05, 0.05))),
            'code_quality_score': max(0, min(1, global_state['code_quality_score'] + random.uniform(-0.05, 0.05))),
            'throughput': max(0, min(200, global_state['throughput'] + random.uniform(-5, 10))),
            'resource_efficiency': max(0, min(1, global_state['resource_efficiency'] + random.uniform(-0.05, 0.05))),
            'global_performance': max(0, min(1, global_state['global_performance'] + random.uniform(-0.02, 0.05)))
        }
        
        # 计算奖励
        rewards = system.calculate_rewards(global_state, actions, next_global_state)
        reward_dict = {agent_id: reward.total_reward for agent_id, reward in rewards.items()}
        
        # 更新智能体
        system.update_agents(global_state, actions, reward_dict, next_global_state)
        
        if episode % 10 == 0:
            metrics = system.get_performance_metrics()
            print(f"Episode {episode}: {metrics}")
    
    # 保存模型
    system.save_all_models("rl_models")

if __name__ == "__main__":
    example_multi_agent_training()