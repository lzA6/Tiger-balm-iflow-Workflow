#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 全能工作流V5 - 量子计算集成模块
Universal Workflow V5 - Quantum Computing Integration Module

集成量子退火、量子优化、量子机器学习等量子算法
提供量子计算能力增强和量子优化解决方案
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import traceback
from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from collections import defaultdict, deque
import math
import random
import hashlib

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class QuantumAlgorithm(Enum):
    """量子算法枚举"""
    QUANTUM_ANNEALING = "quantum_annealing"
    QUANTUM_APPROXIMATION = "quantum_approximation"
    QUANTUM_OPTIMIZATION = "quantum_optimization"
    QUANTUM_MACHINE_LEARNING = "quantum_machine_learning"
    QUANTUM_SEARCH = "quantum_search"
    QUANTUM_SORTING = "quantum_sorting"
    QUANTUM_CLUSTERING = "quantum_clustering"
    QUANTUM_CLASSIFICATION = "quantum_classification"

class QuantumGate(Enum):
    """量子门枚举"""
    HADAMARD = "hadamard"
    PAULI_X = "pauli_x"
    PAULI_Y = "pauli_y"
    PAULI_Z = "pauli_z"
    CNOT = "cnot"
    SWAP = "swap"
    PHASE = "phase"
    ROTATION = "rotation"

class QuantumState(Enum):
    """量子态枚举"""
    ZERO = "|0>"
    ONE = "|1>"
    PLUS = "|+>"
    MINUS = "|->"
    SUPERPOSITION = "superposition"
    ENTANGLED = "entangled"

@dataclass
class Qubit:
    """量子比特"""
    id: int
    state: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], dtype=complex))
    measurement: Optional[int] = None
    entangled_with: set = field(default_factory=set)
    
    def apply_gate(self, gate: np.ndarray):
        """应用量子门"""
        self.state = gate @ self.state
    
    def measure(self) -> int:
        """测量量子比特"""
        probabilities = np.abs(self.state) ** 2
        measurement = np.random.choice([0, 1], p=probabilities)
        self.measurement = measurement
        self.state = np.array([1.0, 0.0]) if measurement == 0 else np.array([0.0, 1.0])
        return measurement

@dataclass
class QuantumCircuit:
    """量子电路"""
    qubits: List[Qubit]
    gates: List[Tuple[str, List[int], np.ndarray]] = field(default_factory=list)
    depth: int = 0
    
    def add_qubit(self, qubit: Qubit):
        """添加量子比特"""
        self.qubits.append(qubit)
    
    def add_gate(self, gate_name: str, qubit_indices: List[int], gate_matrix: np.ndarray):
        """添加量子门"""
        self.gates.append((gate_name, qubit_indices, gate_matrix))
        self.depth += 1
    
    def apply_hadamard(self, qubit_index: int):
        """应用Hadamard门"""
        hadamard_gate = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
        self.add_gate("H", [qubit_index], hadamard_gate)
        self.qubits[qubit_index].apply_gate(hadamard_gate)
    
    def apply_pauli_x(self, qubit_index: int):
        """应用Pauli-X门"""
        pauli_x_gate = np.array([[0, 1], [1, 0]], dtype=complex)
        self.add_gate("X", [qubit_index], pauli_x_gate)
        self.qubits[qubit_index].apply_gate(pauli_x_gate)
    
    def apply_cnot(self, control_index: int, target_index: int):
        """应用CNOT门"""
        cnot_gate = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1],
                              [0, 0, 1, 0]], dtype=complex)
        
        # 创建复合态
        control_state = self.qubits[control_index].state
        target_state = self.qubits[target_index].state
        
        # 计算张量积
        combined_state = np.kron(control_state, target_state)
        
        # 应用CNOT门
        new_combined_state = cnot_gate @ combined_state
        
        # 重构量子比特状态
        new_control_state = np.zeros(2, dtype=complex)
        new_target_state = np.zeros(2, dtype=complex)
        
        for i in range(2):
            for j in range(2):
                idx = i * 2 + j
                new_control_state[i] += new_combined_state[idx]
                new_target_state[j] += new_combined_state[idx]
        
        self.qubits[control_index].state = new_control_state
        self.qubits[target_index].state = new_target_state
        
        # 记录纠缠
        self.qubits[control_index].entangled_with.add(target_index)
        self.qubits[target_index].entangled_with.add(control_index)
        
        self.add_gate("CNOT", [control_index, target_index], cnot_gate)
    
    def measure_all(self) -> List[int]:
        """测量所有量子比特"""
        measurements = []
        for qubit in self.qubits:
            measurement = qubit.measure()
            measurements.append(measurement)
        return measurements
    
    def get_state_vector(self) -> np.ndarray:
        """获取整个电路的状态向量"""
        if not self.qubits:
            return np.array([1.0], dtype=complex)
        
        state_vector = self.qubits[0].state
        for qubit in self.qubits[1:]:
            state_vector = np.kron(state_vector, qubit.state)
        
        return state_vector

class QuantumAnnealingOptimizer:
    """量子退火优化器"""
    
    def __init__(self, num_qubits: int = 10, temperature_schedule: str = "exponential"):
        self.num_qubits = num_qubits
        self.temperature_schedule = temperature_schedule
        self.initial_temperature = 10.0
        self.final_temperature = 0.01
        self.max_iterations = 1000
        
    def optimize(self, objective_function: Callable[[np.ndarray], float],
                  problem_size: int) -> Tuple[np.ndarray, float]:
        """量子退火优化"""
        # 初始化量子态
        circuit = QuantumCircuit([])
        
        # 创建量子比特
        for i in range(self.num_qubits):
            circuit.add_qubit(Qubit(i))
        
        # 初始化叠加态
        for i in range(self.num_qubits):
            circuit.apply_hadamard(i)
        
        best_solution = None
        best_energy = float('inf')
        
        for iteration in range(self.max_iterations):
            # 计算当前温度
            temperature = self._calculate_temperature(iteration)
            
            # 获取当前状态
            state_vector = circuit.get_state_vector()
            
            # 将量子态编码为解
            current_solution = self._decode_quantum_state(state_vector, problem_size)
            current_energy = objective_function(current_solution)
            
            # 更新最佳解
            if current_energy < best_energy:
                best_energy = current_energy
                best_solution = current_solution.copy()
            
            # 量子扰动
            if self._should_accept_perturbation(current_energy, best_energy, temperature):
                self._apply_quantum_perturbation(circuit)
        
        return best_solution, best_energy
    
    def _calculate_temperature(self, iteration: int) -> float:
        """计算当前温度"""
        if self.temperature_schedule == "exponential":
            return self.initial_temperature * (self.final_temperature / self.initial_temperature) ** (iteration / self.max_iterations)
        elif self.temperature_schedule == "linear":
            return self.initial_temperature - (self.initial_temperature - self.final_temperature) * (iteration / self.max_iterations)
        else:
            return self.initial_temperature / (1 + iteration)
    
    def _decode_quantum_state(self, state_vector: np.ndarray, problem_size: int) -> np.ndarray:
        """将量子态解码为解"""
        # 简化版：使用量子态的振幅来生成解
        probabilities = np.abs(state_vector) ** 2
        
        # 归一化
        probabilities = probabilities / np.sum(probabilities)
        
        # 生成解
        solution = np.random.choice(len(probabilities), size=problem_size, p=probabilities)
        
        return solution.astype(float)
    
    def _should_accept_perturbation(self, current_energy: float, best_energy: float, temperature: float) -> bool:
        """判断是否接受扰动"""
        if current_energy < best_energy:
            return True
        
        # Metropolis准则
        energy_diff = current_energy - best_energy
        probability = np.exp(-energy_diff / temperature)
        
        return np.random.random() < probability
    
    def _apply_quantum_perturbation(self, circuit: QuantumCircuit):
        """应用量子扰动"""
        # 随机选择扰动类型
        perturbation_type = np.random.choice(["hadamard", "pauli", "cnot"])
        
        if perturbation_type == "hadamard":
            # 随机Hadamard门
            qubit_index = np.random.randint(0, len(circuit.qubits))
            circuit.apply_hadamard(qubit_index)
        
        elif perturbation_type == "pauli":
            # 随机Pauli门
            qubit_index = np.random.randint(0, len(circuit.qubits))
            circuit.apply_pauli_x(qubit_index)
        
        elif perturbation_type == "cnot" and len(circuit.qubits) >= 2:
            # 随机CNOT门
            control_index = np.random.randint(0, len(circuit.qubits))
            target_index = np.random.choice([i for i in range(len(circuit.qubits)) if i != control_index])
            circuit.apply_cnot(control_index, target_index)

class QuantumOptimizer:
    """量子优化器"""
    
    def __init__(self):
        self.annealer = QuantumAnnealingOptimizer()
        self.optimization_history = []
        
    def optimize_traveling_salesman(self, cities: List[Tuple[float, float]]) -> Tuple[List[int], float]:
        """旅行商问题量子优化"""
        num_cities = len(cities)
        
        def distance_matrix(cities: List[Tuple[float, float]]) -> np.ndarray:
            """计算距离矩阵"""
            n = len(cities)
            matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        matrix[i][j] = np.sqrt((cities[i][0] - cities[j][0])**2 + 
                                               (cities[i][1] - cities[j][1])**2)
            
            return matrix
        
        def objective_function(solution: np.ndarray) -> float:
            """目标函数：计算总距离"""
            total_distance = 0.0
            
            for i in range(len(solution)):
                current_city = int(solution[i])
                next_city = int(solution[(i + 1) % len(solution)])
                total_distance += dist_matrix[current_city][next_city]
            
            return total_distance
        
        # 计算距离矩阵
        dist_matrix = distance_matrix(cities)
        
        # 使用量子退火优化
        best_solution, best_distance = self.annealer.optimize(
            lambda x: objective_function(x),
            num_cities
        )
        
        # 将解转换为城市索引
        tour = [int(city) for city in best_solution]
        
        return tour, best_distance
    
    def optimize_portfolio(self, returns: np.ndarray, risk_tolerance: float = 0.05) -> Tuple[np.ndarray, float]:
        """投资组合量子优化"""
        num_assets = returns.shape[1]
        
        def objective_function(weights: np.ndarray) -> float:
            """目标函数：夏普比率最大化"""
            # 计算组合收益
            portfolio_return = np.mean(returns @ weights)
            
            # 计算组合风险
            portfolio_risk = np.sqrt(weights.T @ np.cov(returns.T) @ weights)
            
            # 夏普比率（考虑风险厌恶）
            sharpe_ratio = portfolio_return / portfolio_risk
            
            # 风险惩罚
            risk_penalty = max(0, portfolio_risk - risk_tolerance) * 10
            
            return -(sharpe_ratio - risk_penalty)
        
        # 使用量子退火优化
        best_weights, best_score = self.annealer.optimize(
            objective_function,
            num_assets
        )
        
        # 归一化权重
        best_weights = np.abs(best_weights)
        best_weights = best_weights / np.sum(best_weights)
        
        # 计算最终分数
        portfolio_return = np.mean(returns @ best_weights)
        portfolio_risk = np.sqrt(best_weights.T @ np.cov(returns.T) @ best_weights)
        final_sharpe = portfolio_return / portfolio_risk
        
        return best_weights, final_sharpe
    
    def optimize_knapsack(self, values: List[float], weights: List[float], 
                         capacity: float) -> Tuple[List[bool], float]:
        """背包问题量子优化"""
        num_items = len(values)
        
        def objective_function(solution: np.ndarray) -> float:
            """目标函数：总价值"""
            total_value = 0.0
            total_weight = 0.0
            
            for i in range(num_items):
                if solution[i] > 0.5:  # 选择物品
                    total_value += values[i]
                    total_weight += weights[i]
            
            # 如果超过容量，施加惩罚
            if total_weight > capacity:
                penalty = (total_weight - capacity) * 100
                return total_value - penalty
            
            return total_value
        
        # 使用量子退火优化
        best_solution, best_value = self.annealer.optimize(
            objective_function,
            num_items
        )
        
        # 转换为选择列表
        selected_items = [int(x > 0.5) for x in best_solution]
        
        # 验证约束
        total_weight = sum(weights[i] for i, selected in enumerate(selected_items) if selected)
        if total_weight > capacity:
            # 修复约束违反
            selected_items = self._fix_knapsack_constraint(selected_items, values, weights, capacity)
            best_value = sum(values[i] for i, selected in enumerate(selected_items) if selected)
        
        return selected_items, best_value
    
    def _fix_knapsack_constraint(self, selected_items: List[bool], values: List[float], 
                               weights: List[float], capacity: float) -> List[bool]:
        """修复背包约束违反"""
        # 计算当前重量
        current_weight = sum(weights[i] for i, selected in enumerate(selected_items) if selected)
        
        # 如果超过容量，移除一些物品
        while current_weight > capacity:
            # 计算价值重量比
            ratios = []
            for i, selected in enumerate(selected_items):
                if selected:
                    ratio = values[i] / weights[i]
                    ratios.append((i, ratio))
            
            # 按比例排序，移除价值重量比最低的物品
            ratios.sort(key=lambda x: x[1])
            
            if ratios:
                item_to_remove = ratios[0][0]
                selected_items[item_to_remove] = False
                current_weight -= weights[item_to_remove]
        
        return selected_items

class QuantumMachineLearning:
    """量子机器学习"""
    
    def __init__(self):
        self.models = {}
        self.training_history = []
        
    def quantum_neural_network(self, input_data: np.ndarray, target_data: np.ndarray,
                               hidden_size: int = 10) -> Dict[str, Any]:
        """量子神经网络"""
        input_size = input_data.shape[1]
        output_size = target_data.shape[1]
        
        # 创建量子电路
        circuit = QuantumCircuit([])
        
        # 输入层量子比特
        for i in range(input_size):
            circuit.add_qubit(Qubit(i))
            circuit.apply_hadamard(i)
        
        # 隐藏层量子比特
        for i in range(hidden_size):
            circuit.add_qubit(Qubit(input_size + i))
            circuit.apply_hadamard(input_size + i)
        
        # 输出层量子比特
        for i in range(output_size):
            circuit.add_qubit(Qubit(input_size + hidden_size + i))
        
        # 创建纠缠
        for i in range(input_size):
            for j in range(hidden_size):
                circuit.apply_cnot(i, input_size + j)
        
        for i in range(hidden_size):
            for j in range(output_size):
                circuit.apply_cnot(input_size + i, input_size + hidden_size + j)
        
        # 训练过程（简化版）
        training_loss = 0.0
        
        for epoch in range(100):
            epoch_loss = 0.0
            
            for sample_idx in range(len(input_data)):
                # 编码输入数据
                input_sample = input_data[sample_idx]
                target_sample = target_data[sample_idx]
                
                # 将输入编码为量子态
                for i, value in enumerate(input_sample):
                    if value > 0.5:
                        circuit.apply_pauli_x(i)
                
                # 测量输出
                measurements = circuit.measure_all()
                
                # 计算损失
                predicted = np.array(measurements[-output_size:]) / len(measurements[-output_size:])
                loss = np.mean((predicted - target_sample) ** 2)
                epoch_loss += loss
            
            training_loss = epoch_loss / len(input_data)
        
        return {
            "circuit": circuit,
            "training_loss": training_loss,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size
        }
    
    def quantum_support_vector_machine(self, data: np.ndarray, labels: np.ndarray,
                                        kernel_type: str = "quantum") -> Dict[str, Any]:
        """量子支持向量机"""
        # 量子核函数
        def quantum_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
            """量子核函数"""
            # 创建量子态
            circuit = QuantumCircuit([])
            
            # 编码数据点
            for i, value in enumerate(x1):
                circuit.add_qubit(Qubit(i))
                if value > 0.5:
                    circuit.apply_hadamard(i)
            
            for i, value in enumerate(x2):
                if i >= len(circuit.qubits):
                    circuit.add_qubit(Qubit(i))
                if value > 0.5:
                    circuit.apply_hadamard(i)
            
            # 创建纠缠
            for i in range(min(len(x1), len(x2))):
                circuit.apply_cnot(i, i + len(x1))
            
            # 测量纠缠度
            state_vector = circuit.get_state_vector()
            entanglement = np.abs(np.sum(state_vector)) / len(state_vector)
            
            return entanglement
        
        # 计算核矩阵
        n_samples = len(data)
        kernel_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i, n_samples):
                kernel_value = quantum_kernel(data[i], data[j])
                kernel_matrix[i, j] = kernel_value
                kernel_matrix[j, i] = kernel_value
        
        # 简化的SVM训练
        alpha = np.ones(n_samples)
        b = 0.0
        
        # 梯度下降训练
        learning_rate = 0.01
        for epoch in range(100):
            for i in range(n_samples):
                decision = np.sum(alpha * labels[i] * kernel_matrix[i]) + b
                if decision < 1:
                    alpha[i] += learning_rate
                    b += learning_rate * labels[i]
        
        return {
            "kernel_matrix": kernel_matrix,
            "alpha": alpha,
            "b": b,
            "support_vectors": np.where(alpha > 0)[0].tolist(),
            "kernel_type": kernel_type
        }
    
    def quantum_clustering(self, data: np.ndarray, n_clusters: int) -> Dict[str, Any]:
        """量子聚类"""
        n_samples = data.shape[0]
        n_features = data.shape[1]
        
        # 创建量子电路
        circuit = QuantumCircuit([])
        
        # 数据编码量子比特
        for i in range(n_features):
            circuit.add_qubit(Qubit(i))
        
        # 聚类中心量子比特
        for i in range(n_clusters):
            circuit.add_qubit(Qubit(n_features + i))
        
        # 编码数据点
        cluster_assignments = []
        
        for sample_idx in range(n_samples):
            sample = data[sample_idx]
            
            # 重置电路
            for qubit in circuit.qubits:
                qubit.state = np.array([1.0, 0.0], dtype=complex)
                qubit.measurement = None
                qubit.entangled_with.clear()
            
            # 编码数据
            for i, value in enumerate(sample):
                if value > 0.5:
                    circuit.apply_pauli_x(i)
            
            # 创建与聚类中心的纠缠
            for i in range(n_features):
                for j in range(n_clusters):
                    circuit.apply_cnot(i, n_features + j)
            
            # 测量聚类中心
            center_measurements = circuit.measure_all()[-n_clusters:]
            
            # 选择最近的聚类中心
            cluster_id = np.argmax(center_measurements)
            cluster_assignments.append(cluster_id)
        
        return {
            "cluster_assignments": cluster_assignments,
            "n_clusters": n_clusters,
            "cluster_centers": self._calculate_cluster_centers(data, cluster_assignments, n_clusters)
        }
    
    def _calculate_cluster_centers(self, data: np.ndarray, assignments: List[int], n_clusters: int) -> List[np.ndarray]:
        """计算聚类中心"""
        cluster_centers = []
        
        for cluster_id in range(n_clusters):
            cluster_data = data[np.array(assignments) == cluster_id]
            
            if len(cluster_data) > 0:
                center = np.mean(cluster_data, axis=0)
            else:
                center = np.zeros(data.shape[1])
            
            cluster_centers.append(center)
        
        return cluster_centers

class QuantumComputingIntegration:
    """量子计算集成"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.optimizer = QuantumOptimizer()
        self.qml = QuantumMachineLearning()
        self.performance_metrics = defaultdict(list)
        
    def solve_optimization_problem(self, problem_type: str, **kwargs) -> Dict[str, Any]:
        """求解优化问题"""
        if problem_type == "traveling_salesman":
            cities = kwargs.get("cities", [])
            if not cities:
                # 生成随机城市
                cities = [(np.random.uniform(-10, 10), np.random.uniform(-10, 10)) for _ in range(10)]
            
            tour, distance = self.optimizer.optimize_traveling_salesman(cities)
            
            return {
                "problem_type": problem_type,
                "solution": tour,
                "objective_value": distance,
                "cities": cities
            }
        
        elif problem_type == "portfolio_optimization":
            returns = kwargs.get("returns")
            risk_tolerance = kwargs.get("risk_tolerance", 0.05)
            
            if returns is None:
                # 生成随机收益数据
                returns = np.random.randn(100, 5) * 0.01 + 0.001
            
            weights, sharpe_ratio = self.optimizer.optimize_portfolio(returns, risk_tolerance)
            
            return {
                "problem_type": problem_type,
                "solution": weights.tolist(),
                "sharpe_ratio": sharpe_ratio,
                "risk_tolerance": risk_tolerance
            }
        
        elif problem_type == "knapsack":
            values = kwargs.get("values", [])
            weights = kwargs.get("weights", [])
            capacity = kwargs.get("capacity", 50)
            
            if not values or not weights:
                # 生成随机数据
                n_items = 20
                values = np.random.uniform(1, 100, n_items).tolist()
                weights = np.random.uniform(1, 20, n_items).tolist()
                capacity = sum(weights) * 0.5
            
            selected_items, total_value = self.optimizer.optimize_knapsack(values, weights, capacity)
            
            return {
                "problem_type": problem_type,
                "solution": selected_items,
                "total_value": total_value,
                "capacity": capacity
            }
        
        else:
            raise ValueError(f"不支持的优化问题类型: {problem_type}")
    
    def train_quantum_model(self, model_type: str, **kwargs) -> Dict[str, Any]:
        """训练量子模型"""
        if model_type == "quantum_neural_network":
            input_data = kwargs.get("input_data")
            target_data = kwargs.get("target_data")
            hidden_size = kwargs.get("hidden_size", 10)
            
            if input_data is None or target_data is None:
                # 生成随机数据
                n_samples = 100
                input_size = 5
                output_size = 2
                
                input_data = np.random.randn(n_samples, input_size)
                target_data = np.random.randint(0, 2, (n_samples, output_size))
            
            model_info = self.qml.quantum_neural_network(input_data, target_data, hidden_size)
            
            return {
                "model_type": model_type,
                "model_info": model_info,
                "training_data_shape": input_data.shape
            }
        
        elif model_type == "quantum_svm":
            data = kwargs.get("data")
            labels = kwargs.get("labels")
            kernel_type = kwargs.get("kernel_type", "quantum")
            
            if data is None or labels is None:
                # 生成随机数据
                n_samples = 100
                n_features = 5
                
                data = np.random.randn(n_samples, n_features)
                labels = np.random.randint(0, 2, n_samples)
            
            model_info = self.qml.quantum_support_vector_machine(data, labels, kernel_type)
            
            return {
                "model_type": model_type,
                "model_info": model_info,
                "data_shape": data.shape
            }
        
        elif model_type == "quantum_clustering":
            data = kwargs.get("data")
            n_clusters = kwargs.get("n_clusters", 3)
            
            if data is None:
                # 生成随机数据
                n_samples = 100
                n_features = 5
                
                data = np.random.randn(n_samples, n_features)
            
            cluster_info = self.qml.quantum_clustering(data, n_clusters)
            
            return {
                "model_type": model_type,
                "cluster_info": cluster_info,
                "data_shape": data.shape
            }
        
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def quantum_fourier_transform(self, data: np.ndarray) -> np.ndarray:
        """量子傅里叶变换（简化版）"""
        # 创建量子电路
        circuit = QuantumCircuit([])
        
        # 数据编码
        n_qubits = int(np.log2(len(data)))
        if 2 ** n_qubits != len(data):
            # 填充到最近的2的幂
            padded_length = 2 ** (n_qubits + 1)
            padded_data = np.zeros(padded_length, dtype=complex)
            padded_data[:len(data)] = data
            data = padded_data
            n_qubits = int(np.log2(len(data)))
        
        for i in range(n_qubits):
            circuit.add_qubit(Qubit(i))
            circuit.apply_hadamard(i)
        
        # 编码数据到量子态
        for i, value in enumerate(data):
            if i >= n_qubits:
                break
            if abs(value) > 0.001:
                amplitude = min(1.0, abs(value))
                phase = np.angle(value)
                
                # 应用相位门
                phase_gate = np.array([[1, 0], [0, np.exp(1j * phase)]], dtype=complex)
                circuit.qubits[i].apply_gate(phase_gate)
                
                # 应用振幅缩放
                circuit.qubits[i].state *= amplitude
        
        # 量子傅里叶变换
        for i in range(n_qubits):
            circuit.apply_hadamard(i)
        
        # 获取变换结果
        state_vector = circuit.get_state_vector()
        
        # 解码结果
        result = np.zeros(len(data), dtype=complex)
        
        for i in range(min(len(data), len(state_vector))):
            result[i] = state_vector[i]
        
        return result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        metrics = {}
        
        for metric_name, values in self.performance_metrics.items():
            if values:
                metrics[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values)
                }
        
        return metrics

# 全局量子计算集成实例
_quantum_integration = None

def get_quantum_integration(config: Dict[str, Any] = None) -> QuantumComputingIntegration:
    """获取量子计算集成实例"""
    global _quantum_integration
    if _quantum_integration is None:
        _quantum_integration = QuantumComputingIntegration(config)
    return _quantum_integration

if __name__ == "__main__":
    # 测试代码
    def test_quantum_integration():
        integration = get_quantum_integration()
        
        # 测试旅行商问题
        cities = [(0, 0), (1, 2), (3, 1), (5, 4), (2, 3), (4, 0), (6, 5), (1, 6), (3, 3), (5, 2)]
        
        tsp_result = integration.solve_optimization_problem("traveling_salesman", cities=cities)
        print("旅行商问题结果:")
        print(f"路径: {tsp_result['solution']}")
        print(f"总距离: {tsp_result['objective_value']:.2f}")
        
        # 测试投资组合优化
        returns = np.random.randn(100, 5) * 0.01 + 0.001
        
        portfolio_result = integration.solve_optimization_problem(
            "portfolio_optimization", 
            returns=returns, 
            risk_tolerance=0.05
        )
        print("\n投资组合优化结果:")
        print(f"权重: {[f'{w:.3f}' for w in portfolio_result['solution']]}")
        print(f"夏普比率: {portfolio_result['sharpe_ratio']:.4f}")
        
        # 测试量子神经网络
        input_data = np.random.randn(50, 3)
        target_data = np.random.randint(0, 2, (50, 2))
        
        nn_result = integration.train_quantum_model(
            "quantum_neural_network",
            input_data=input_data,
            target_data=target_data,
            hidden_size=8
        )
        print("\n量子神经网络结果:")
        print(f"训练损失: {nn_result['model_info']['training_loss']:.4f}")
        print(f"网络结构: {nn_result['model_info']['input_size']}-{nn_result['model_info']['hidden_size']}-{nn_result['model_info']['output_size']}")
        
        # 测试量子聚类
        cluster_data = np.random.randn(50, 4)
        
        cluster_result = integration.train_quantum_model(
            "quantum_clustering",
            data=cluster_data,
            n_clusters=3
        )
        print("\n量子聚类结果:")
        print(f"聚类分配: {cluster_result['cluster_info']['cluster_assignments'][:10]}...")
        print(f"聚类数量: {cluster_result['cluster_info']['n_clusters']}")
        
        # 获取性能指标
        metrics = integration.get_performance_metrics()
        print("\n性能指标:")
        for metric, stats in metrics.items():
            print(f"{metric}: {stats}")
    
    # 运行测试
    test_quantum_integration()
