#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应量子退火算法实现
Adaptive Quantum Annealing Implementation

作者: Quantum AI Team
版本: 5.1.0
日期: 2025-11-12
"""

import numpy as np
import random
import math
import time
import json
import logging
from typing import List, Dict, Tuple, Any, Callable, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnnealingState:
    """退火状态"""
    energy: float
    configuration: np.ndarray
    temperature: float
    iteration: int
    acceptance_rate: float
    improvement_rate: float
    tunneling_probability: float
    cooling_rate: float

@dataclass
class AdaptiveParameters:
    """自适应参数"""
    initial_temperature: float = 1000.0
    min_temperature: float = 0.001
    cooling_schedule: str = "adaptive"  # adaptive, exponential, linear
    tunneling_strength: float = 0.1
    complexitiy_adaptation: bool = True
    convergence_threshold: float = 1e-6
    max_iterations: int = 10000
    stagnation_threshold: int = 100

class AdaptiveQuantumAnnealing:
    """自适应量子退火算法"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化自适应量子退火
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.params = AdaptiveParameters(**self.config.get('annealing', {}))
        self.history = []
        self.best_state = None
        self.complexity_metrics = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                import yaml
                return yaml.safe_load(f)
        return {}
    
    def optimize(self, 
                 objective_function: Callable[[np.ndarray], float],
                 dimensions: int,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 constraints: Optional[List[Callable]] = None) -> AnnealingState:
        """
        执行自适应量子退火优化
        
        Args:
            objective_function: 目标函数
            dimensions: 问题维度
            bounds: 变量边界 [(min, max), ...]
            constraints: 约束条件列表
            
        Returns:
            最优状态
        """
        logger.info(f"开始自适应量子退火优化 - 维度: {dimensions}")
        
        # 初始化
        current_state = self._initialize_state(objective_function, dimensions, bounds, constraints)
        self.best_state = current_state
        self.history = [asdict(current_state)]
        
        # 计算问题复杂度
        if self.params.complexitiy_adaptation:
            complexity = self._estimate_complexity(objective_function, dimensions)
            self._adapt_parameters_for_complexity(complexity)
        
        # 主退火循环
        for iteration in range(self.params.max_iterations):
            # 生成邻域解
            candidate = self._generate_candidate(current_state, bounds, constraints)
            
            # 计算能量（目标函数值）
            candidate.energy = objective_function(candidate.configuration)
            
            # 自适应调整参数
            self._adaptive_adjustment(current_state, iteration)
            
            # 量子隧穿决策
            if self._quantum_tunneling_decision(current_state, candidate):
                current_state = candidate
                self._update_best_state(current_state)
            
            # 更新状态
            current_state.iteration = iteration
            current_state.temperature = self._calculate_temperature(iteration)
            
            # 记录历史
            self.history.append(asdict(current_state))
            
            # 检查收敛
            if self._check_convergence(current_state):
                logger.info(f"收敛于迭代 {iteration}, 最优能量: {self.best_state.energy}")
                break
            
            # 检查停滞
            if self._check_stagnation(iteration):
                self._escape_stagnation(current_state, bounds, constraints)
        
        logger.info(f"优化完成 - 最优能量: {self.best_state.energy}")
        return self.best_state
    
    def _initialize_state(self, 
                          objective_function: Callable,
                          dimensions: int,
                          bounds: Optional[List[Tuple[float, float]]],
                          constraints: Optional[List]) -> AnnealingState:
        """初始化状态"""
        # 生成初始配置
        if bounds:
            configuration = np.array([
                random.uniform(b[0], b[1]) for b in bounds
            ])
        else:
            configuration = np.random.randn(dimensions)
        
        # 应用约束
        if constraints:
            configuration = self._apply_constraints(configuration, constraints)
        
        # 计算初始能量
        energy = objective_function(configuration)
        
        return AnnealingState(
            energy=energy,
            configuration=configuration,
            temperature=self.params.initial_temperature,
            iteration=0,
            acceptance_rate=1.0,
            improvement_rate=0.0,
            tunneling_probability=self.params.tunneling_strength,
            cooling_rate=0.95
        )
    
    def _estimate_complexity(self, objective_function: Callable, dimensions: int) -> float:
        """估计问题复杂度"""
        # 采样评估复杂度
        samples = min(100, dimensions * 10)
        energies = []
        
        for _ in range(samples):
            sample = np.random.randn(dimensions)
            try:
                energy = objective_function(sample)
                energies.append(energy)
            except:
                continue
        
        if len(energies) < 2:
            return 1.0
        
        # 计算复杂度指标
        energy_variance = np.var(energies)
        energy_range = np.max(energies) - np.min(energies)
        gradient_estimate = self._estimate_gradient_complexity(objective_function, dimensions)
        
        # 综合复杂度评分
        complexity = (energy_variance + energy_range + gradient_estimate) / 3.0
        self.complexity_metrics = {
            'variance': energy_variance,
            'range': energy_range,
            'gradient': gradient_estimate,
            'overall': complexity
        }
        
        logger.info(f"问题复杂度估计: {complexity:.4f}")
        return complexity
    
    def _estimate_gradient_complexity(self, objective_function: Callable, dimensions: int) -> float:
        """估计梯度复杂度"""
        base_point = np.random.randn(dimensions)
        epsilon = 1e-5
        
        gradients = []
        for i in range(min(10, dimensions)):
            point_plus = base_point.copy()
            point_plus[i] += epsilon
            point_minus = base_point.copy()
            point_minus[i] -= epsilon
            
            try:
                grad = (objective_function(point_plus) - objective_function(point_minus)) / (2 * epsilon)
                gradients.append(abs(grad))
            except:
                continue
        
        return np.mean(gradients) if gradients else 1.0
    
    def _adapt_parameters_for_complexity(self, complexity: float):
        """根据复杂度自适应调整参数"""
        if complexity > 2.0:  # 高复杂度
            self.params.tunneling_strength = min(0.3, self.params.tunneling_strength * 1.5)
            self.params.cooling_schedule = "adaptive"
            self.params.convergence_threshold = 1e-4
        elif complexity > 1.0:  # 中等复杂度
            self.params.tunneling_strength = min(0.2, self.params.tunneling_strength * 1.2)
        else:  # 低复杂度
            self.params.tunneling_strength = max(0.05, self.params.tunneling_strength * 0.8)
            self.params.cooling_schedule = "exponential"
        
        logger.info(f"自适应调整参数 - 隧穿强度: {self.params.tunneling_strength:.3f}")
    
    def _generate_candidate(self, 
                           current_state: AnnealingState,
                           bounds: Optional[List[Tuple[float, float]]],
                           constraints: Optional[List]) -> AnnealingState:
        """生成候选解"""
        # 基于当前温度和隧穿概率生成扰动
        perturbation_scale = current_state.temperature * 0.1
        
        # 量子隧穿扰动
        if random.random() < current_state.tunneling_probability:
            # 大幅度隧穿跳跃
            perturbation = np.random.randn(len(current_state.configuration)) * perturbation_scale * 3
        else:
            # 常规局部搜索
            perturbation = np.random.randn(len(current_state.configuration)) * perturbation_scale
        
        new_configuration = current_state.configuration + perturbation
        
        # 应用边界约束
        if bounds:
            for i, (min_val, max_val) in enumerate(bounds):
                new_configuration[i] = np.clip(new_configuration[i], min_val, max_val)
        
        # 应用自定义约束
        if constraints:
            new_configuration = self._apply_constraints(new_configuration, constraints)
        
        return AnnealingState(
            energy=0.0,  # 稍后计算
            configuration=new_configuration,
            temperature=current_state.temperature,
            iteration=current_state.iteration,
            acceptance_rate=current_state.acceptance_rate,
            improvement_rate=current_state.improvement_rate,
            tunneling_probability=current_state.tunneling_probability,
            cooling_rate=current_state.cooling_rate
        )
    
    def _apply_constraints(self, configuration: np.ndarray, constraints: List[Callable]) -> np.ndarray:
        """应用约束条件"""
        for constraint in constraints:
            try:
                configuration = constraint(configuration)
            except:
                continue
        return configuration
    
    def _adaptive_adjustment(self, state: AnnealingState, iteration: int):
        """自适应调整参数"""
        # 基于历史性能调整
        if len(self.history) > 10:
            recent_history = self.history[-10:]
            improvements = [h['energy'] for h in recent_history]
            
            # 计算改进率
            if len(improvements) > 1:
                improvement_trend = improvements[0] - improvements[-1]
                state.improvement_rate = improvement_trend / abs(improvements[0]) if improvements[0] != 0 else 0
            
            # 动态调整隧穿概率
            if state.improvement_rate < 0.01:  # 改进缓慢，增加隧穿
                state.tunneling_probability = min(0.5, state.tunneling_probability * 1.1)
            elif state.improvement_rate > 0.1:  # 改进良好，减少隧穿
                state.tunneling_probability = max(0.01, state.tunneling_probability * 0.9)
        
        # 自适应冷却速率
        if self.params.cooling_schedule == "adaptive":
            # 基于接受率调整冷却速率
            if state.acceptance_rate > 0.8:  # 接受率太高，加快冷却
                state.cooling_rate = min(0.99, state.cooling_rate * 1.05)
            elif state.acceptance_rate < 0.2:  # 接受率太低，减慢冷却
                state.cooling_rate = max(0.85, state.cooling_rate * 0.95)
    
    def _quantum_tunneling_decision(self, current: AnnealingState, candidate: AnnealingState) -> bool:
        """量子隧穿决策"""
        delta_energy = candidate.energy - current.energy
        
        # 计算接受概率
        if delta_energy < 0:
            acceptance_prob = 1.0
        else:
            # 经典Metropolis准则 + 量子隧穿增强
            classical_prob = math.exp(-delta_energy / (current.temperature + 1e-10))
            quantum_enhancement = candidate.tunneling_probability * math.exp(-math.sqrt(delta_energy) / (current.temperature + 1e-10))
            acceptance_prob = classical_prob + quantum_enhancement
        
        # 更新接受率
        if len(self.history) > 0:
            current.acceptance_rate = 0.9 * current.acceptance_rate + 0.1 * acceptance_prob
        
        return random.random() < acceptance_prob
    
    def _calculate_temperature(self, iteration: int) -> float:
        """计算当前温度"""
        if self.params.cooling_schedule == "exponential":
            return self.params.initial_temperature * (0.95 ** iteration)
        elif self.params.cooling_schedule == "linear":
            return max(self.params.min_temperature, 
                      self.params.initial_temperature - iteration * (self.params.initial_temperature - self.params.min_temperature) / self.params.max_iterations)
        else:  # adaptive
            # 基于性能的自适应冷却
            if len(self.history) > 5:
                recent_improvement = self.history[-5]['energy'] - self.history[-1]['energy']
                if recent_improvement > 0:
                    # 有改进，保持较高温度
                    return max(self.params.min_temperature, self.best_state.temperature * 0.98)
                else:
                    # 无改进，快速降温
                    return max(self.params.min_temperature, self.best_state.temperature * 0.9)
            return self.best_state.temperature * 0.95
    
    def _update_best_state(self, state: AnnealingState):
        """更新最佳状态"""
        if self.best_state is None or state.energy < self.best_state.energy:
            self.best_state = AnnealingState(
                energy=state.energy,
                configuration=state.configuration.copy(),
                temperature=state.temperature,
                iteration=state.iteration,
                acceptance_rate=state.acceptance_rate,
                improvement_rate=state.improvement_rate,
                tunneling_probability=state.tunneling_probability,
                cooling_rate=state.cooling_rate
            )
    
    def _check_convergence(self, state: AnnealingState) -> bool:
        """检查收敛条件"""
        if len(self.history) < 10:
            return False
        
        # 能量变化很小
        recent_energies = [h['energy'] for h in self.history[-10:]]
        energy_variance = np.var(recent_energies)
        
        if energy_variance < self.params.convergence_threshold:
            return True
        
        # 温度接近最小值
        if state.temperature < self.params.min_temperature * 2:
            return True
        
        return False
    
    def _check_stagnation(self, iteration: int) -> bool:
        """检查停滞状态"""
        if len(self.history) < self.params.stagnation_threshold:
            return False
        
        # 检查最近是否有显著改进
        recent_energies = [h['energy'] for h in self.history[-self.params.stagnation_threshold:]]
        improvement = recent_energies[0] - recent_energies[-1]
        
        # 相对改进小于阈值
        relative_improvement = improvement / abs(recent_energies[0]) if recent_energies[0] != 0 else 0
        
        return relative_improvement < 0.001
    
    def _escape_stagnation(self, state: AnnealingState, bounds: Optional[List], constraints: Optional[List]):
        """逃离停滞状态"""
        logger.info("检测到停滞，执行逃离策略...")
        
        # 重新加热
        state.temperature = min(state.temperature * 2, self.params.initial_temperature * 0.5)
        
        # 增加隧穿概率
        state.tunneling_probability = min(state.tunneling_probability * 2, 0.5)
        
        # 生成新的随机解
        if bounds:
            state.configuration = np.array([random.uniform(b[0], b[1]) for b in bounds])
        else:
            state.configuration = np.random.randn(len(state.configuration))
        
        if constraints:
            state.configuration = self._apply_constraints(state.configuration, constraints)
    
    def get_optimization_report(self) -> Dict:
        """获取优化报告"""
        if not self.history:
            return {"error": "没有优化历史记录"}
        
        energies = [h['energy'] for h in self.history]
        temperatures = [h['temperature'] for h in self.history]
        
        report = {
            "best_energy": self.best_state.energy if self.best_state else None,
            "best_configuration": self.best_state.configuration.tolist() if self.best_state else None,
            "total_iterations": len(self.history),
            "initial_energy": energies[0],
            "final_energy": energies[-1],
            "energy_improvement": energies[0] - energies[-1],
            "convergence_iteration": self._find_convergence_iteration(),
            "complexity_metrics": self.complexity_metrics,
            "performance_metrics": {
                "energy_variance": np.var(energies),
                "energy_std": np.std(energies),
                "temperature_range": [min(temperatures), max(temperatures)],
                "average_acceptance_rate": np.mean([h['acceptance_rate'] for h in self.history])
            }
        }
        
        return report
    
    def _find_convergence_iteration(self) -> Optional[int]:
        """找到收敛迭代"""
        if len(self.history) < 10:
            return None
        
        for i in range(10, len(self.history)):
            recent_energies = [h['energy'] for h in self.history[i-10:i]]
            if np.var(recent_energies) < self.params.convergence_threshold:
                return i
        
        return None
    
    def save_history(self, filepath: str):
        """保存优化历史"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'parameters': asdict(self.params),
                'history': self.history,
                'best_state': asdict(self.best_state) if self.best_state else None,
                'complexity_metrics': self.complexity_metrics
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"优化历史已保存到: {filepath}")

# 示例使用
def example_optimization():
    """示例优化问题"""
    def rastrigin_function(x):
        """Rastrigin函数 - 多模态优化测试函数"""
        n = len(x)
        return 10 * n + sum([(xi ** 2 - 10 * np.cos(2 * np.pi * xi)) for xi in x])
    
    # 创建优化器
    optimizer = AdaptiveQuantumAnnealing()
    
    # 设置优化参数
    dimensions = 10
    bounds = [(-5.12, 5.12)] * dimensions
    
    # 执行优化
    result = optimizer.optimize(
        objective_function=rastrigin_function,
        dimensions=dimensions,
        bounds=bounds
    )
    
    # 输出结果
    print(f"最优解: {result.configuration}")
    print(f"最优值: {result.energy}")
    
    # 生成报告
    report = optimizer.get_optimization_report()
    print(f"能量改进: {report['energy_improvement']}")
    print(f"收敛迭代: {report['convergence_iteration']}")
    
    # 保存历史
    optimizer.save_history("optimization_history.json")

if __name__ == "__main__":
    example_optimization()