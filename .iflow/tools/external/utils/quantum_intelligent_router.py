#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量子智能路由系统
Quantum Intelligent Routing System

作者: Quantum AI Team
版本: 5.1.0
日期: 2025-11-12
"""

import numpy as np
import json
import time
import hashlib
import logging
import requests
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path
import threading
import psutil
import subprocess
import re

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelCapability:
    """模型能力描述"""
    name: str
    provider: str
    max_tokens: int
    supports_functions: bool
    supports_vision: bool
    supports_code_execution: bool
    cost_per_1k_tokens: float
    latency_ms: float
    reliability: float
    capabilities: List[str]
    api_endpoint: str
    current_load: float = 0.0
    success_rate: float = 1.0
    last_response_time: float = 0.0

@dataclass
class RoutingRequest:
    """路由请求"""
    task_type: str
    complexity: str
    language: Optional[str]
    context_length: int
    urgency: str
    budget_constraint: Optional[float]
    quality_requirement: str
    special_requirements: List[str]
    user_context: Dict[str, Any]

@dataclass
class RoutingDecision:
    """路由决策"""
    selected_models: List[str]
    collaboration_strategy: str
    expected_performance: float
    estimated_cost: float
    estimated_latency: float
    confidence: float
    reasoning: str

class QuantumRouter:
    """量子智能路由器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化量子路由器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.models = self._initialize_models()
        self.routing_history = deque(maxlen=1000)
        self.performance_cache = {}
        self.quantum_state = np.random.randn(64)  # 量子态向量
        self.entanglement_matrix = np.eye(len(self.models))  # 纠缠矩阵
        
        # 性能监控
        self.request_count = 0
        self.success_count = 0
        self.total_latency = 0.0
        self.total_cost = 0.0
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                import yaml
                return yaml.safe_load(f)
        
        # 默认配置
        return {
            'routing_algorithm': 'quantum_parallel',
            'cache_enabled': True,
            'cache_ttl': 300,
            'load_balancing': 'quantum_weighted',
            'performance_threshold': 0.8,
            'cost_sensitivity': 0.3
        }
    
    def _initialize_models(self) -> Dict[str, ModelCapability]:
        """初始化模型能力"""
        models = {}
        
        # 默认模型配置
        default_models = [
            {
                'name': 'GPT-4-Turbo',
                'provider': 'OpenAI',
                'max_tokens': 128000,
                'supports_functions': True,
                'supports_vision': True,
                'supports_code_execution': True,
                'cost_per_1k_tokens': 0.03,
                'latency_ms': 500,
                'reliability': 0.99,
                'capabilities': ['reasoning', 'coding', 'analysis', 'vision', 'math'],
                'api_endpoint': 'https://api.openai.com/v1/chat/completions'
            },
            {
                'name': 'Claude-3.5-Sonnet',
                'provider': 'Anthropic',
                'max_tokens': 200000,
                'supports_functions': True,
                'supports_vision': True,
                'supports_code_execution': True,
                'cost_per_1k_tokens': 0.015,
                'latency_ms': 600,
                'reliability': 0.98,
                'capabilities': ['reasoning', 'coding', 'analysis', 'vision', 'writing'],
                'api_endpoint': 'https://api.anthropic.com/v1/messages'
            },
            {
                'name': 'DeepSeek-V2.5',
                'provider': 'DeepSeek',
                'max_tokens': 128000,
                'supports_functions': True,
                'supports_vision': False,
                'supports_code_execution': True,
                'cost_per_1k_tokens': 0.002,
                'latency_ms': 700,
                'reliability': 0.95,
                'capabilities': ['reasoning', 'coding', 'analysis', 'math'],
                'api_endpoint': 'https://api.deepseek.com/v1/chat/completions'
            }
        ]
        
        for model_config in default_models:
            model = ModelCapability(**model_config)
            models[model.name] = model
        
        return models
    
    def route_request(self, request: RoutingRequest) -> RoutingDecision:
        """
        智能路由请求
        
        Args:
            request: 路由请求
            
        Returns:
            路由决策
        """
        start_time = time.time()
        self.request_count += 1
        
        logger.info(f"处理路由请求: {request.task_type} - 复杂度: {request.complexity}")
        
        # 检查缓存
        cache_key = self._generate_cache_key(request)
        if self.config.get('cache_enabled', True) and cache_key in self.performance_cache:
            cached_result = self.performance_cache[cache_key]
            if time.time() - cached_result['timestamp'] < self.config.get('cache_ttl', 300):
                logger.info("使用缓存的路由结果")
                return cached_result['decision']
        
        # 量子并行评估
        quantum_evaluations = self._quantum_parallel_evaluation(request)
        
        # 量子纠缠优化
        optimized_combinations = self._quantum_entanglement_optimization(quantum_evaluations, request)
        
        # 选择最优组合
        best_combination = self._select_optimal_combination(optimized_combinations, request)
        
        # 生成路由决策
        decision = RoutingDecision(
            selected_models=best_combination['models'],
            collaboration_strategy=best_combination['strategy'],
            expected_performance=best_combination['performance'],
            estimated_cost=best_combination['cost'],
            estimated_latency=best_combination['latency'],
            confidence=best_combination['confidence'],
            reasoning=best_combination['reasoning']
        )
        
        # 更新量子态
        self._update_quantum_state(request, decision)
        
        # 缓存结果
        if self.config.get('cache_enabled', True):
            self.performance_cache[cache_key] = {
                'decision': decision,
                'timestamp': time.time()
            }
        
        # 记录历史
        self.routing_history.append({
            'request': asdict(request),
            'decision': asdict(decision),
            'timestamp': time.time()
        })
        
        # 更新性能指标
        processing_time = time.time() - start_time
        self.total_latency += processing_time
        
        logger.info(f"路由完成 - 选择模型: {decision.selected_models} - 置信度: {decision.confidence:.3f}")
        
        return decision
    
    def _generate_cache_key(self, request: RoutingRequest) -> str:
        """生成缓存键"""
        key_data = f"{request.task_type}_{request.complexity}_{request.language}_{request.context_length}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _quantum_parallel_evaluation(self, request: RoutingRequest) -> List[Dict]:
        """量子并行评估所有可能的模型组合"""
        evaluations = []
        
        # 生成所有可能的模型组合
        model_names = list(self.models.keys())
        
        # 单模型评估
        for model_name in model_names:
            evaluation = self._evaluate_model_combination([model_name], request)
            evaluations.append(evaluation)
        
        # 双模型协作评估
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                evaluation = self._evaluate_model_combination([model1, model2], request)
                evaluations.append(evaluation)
        
        # 三模型协作评估（高复杂度任务）
        if request.complexity in ['high', 'critical']:
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names[i+1:], i+1):
                    for k, model3 in enumerate(model_names[j+1:], j+1):
                        evaluation = self._evaluate_model_combination([model1, model2, model3], request)
                        evaluations.append(evaluation)
        
        return evaluations
    
    def _evaluate_model_combination(self, model_names: List[str], request: RoutingRequest) -> Dict:
        """评估模型组合"""
        models = [self.models[name] for name in model_names]
        
        # 能力匹配评分
        capability_score = self._calculate_capability_score(models, request)
        
        # 性能评分
        performance_score = self._calculate_performance_score(models, request)
        
        # 成本评分
        cost_score = self._calculate_cost_score(models, request)
        
        # 可用性评分
        availability_score = self._calculate_availability_score(models)
        
        # 协作效应评分
        collaboration_score = self._calculate_collaboration_score(models, request)
        
        # 综合评分（量子加权）
        quantum_weights = self._calculate_quantum_weights(len(model_names))
        total_score = (
            capability_score * quantum_weights[0] +
            performance_score * quantum_weights[1] +
            cost_score * quantum_weights[2] +
            availability_score * quantum_weights[3] +
            collaboration_score * quantum_weights[4]
        )
        
        # 估算成本和延迟
        estimated_cost = sum(model.cost_per_1k_tokens for model in models) * (request.context_length / 1000)
        estimated_latency = max(model.latency_ms for model in models) * (1 + 0.2 * (len(model_names) - 1))
        
        # 确定协作策略
        strategy = self._determine_collaboration_strategy(model_names, request)
        
        return {
            'models': model_names,
            'strategy': strategy,
            'performance': total_score,
            'cost': estimated_cost,
            'latency': estimated_latency,
            'confidence': min(1.0, total_score),
            'capability_score': capability_score,
            'performance_score': performance_score,
            'cost_score': cost_score,
            'availability_score': availability_score,
            'collaboration_score': collaboration_score
        }
    
    def _calculate_capability_score(self, models: List[ModelCapability], request: RoutingRequest) -> float:
        """计算能力匹配评分"""
        required_capabilities = self._get_required_capabilities(request)
        total_score = 0.0
        total_weight = 0.0
        
        for model in models:
            model_score = 0.0
            for capability, weight in required_capabilities.items():
                if capability in model.capabilities:
                    model_score += weight
            
            # 标准化
            max_possible = sum(weight for weight in required_capabilities.values())
            if max_possible > 0:
                model_score = model_score / max_possible
            
            total_score += model_score
        
        return total_score / len(models) if models else 0.0
    
    def _get_required_capabilities(self, request: RoutingRequest) -> Dict[str, float]:
        """获取任务所需能力"""
        capabilities = {}
        
        if request.task_type == 'coding':
            capabilities.update({'coding': 0.4, 'reasoning': 0.3, 'analysis': 0.3})
        elif request.task_type == 'reasoning':
            capabilities.update({'reasoning': 0.5, 'analysis': 0.3, 'math': 0.2})
        elif request.task_type == 'analysis':
            capabilities.update({'analysis': 0.4, 'reasoning': 0.3, 'writing': 0.3})
        elif request.task_type == 'vision':
            capabilities.update({'vision': 0.6, 'reasoning': 0.4})
        elif request.task_type == 'writing':
            capabilities.update({'writing': 0.5, 'reasoning': 0.3, 'analysis': 0.2})
        else:
            capabilities.update({'reasoning': 0.3, 'analysis': 0.3, 'coding': 0.2, 'writing': 0.2})
        
        # 特殊需求
        for requirement in request.special_requirements:
            if requirement in capabilities:
                capabilities[requirement] *= 1.5
            else:
                capabilities[requirement] = 0.5
        
        return capabilities
    
    def _calculate_performance_score(self, models: List[ModelCapability], request: RoutingRequest) -> float:
        """计算性能评分"""
        # 基于可靠性和延迟的综合评分
        reliability_score = np.mean([model.reliability for model in models])
        
        # 延迟评分（越低越好）
        avg_latency = np.mean([model.latency_ms for model in models])
        latency_score = max(0, 1 - avg_latency / 2000)  # 2000ms作为基准
        
        # 成功率评分
        success_rate_score = np.mean([model.success_rate for model in models])
        
        # 复杂度调整
        complexity_multiplier = 1.0
        if request.complexity == 'high':
            complexity_multiplier = 1.2
        elif request.complexity == 'critical':
            complexity_multiplier = 1.5
        
        return (reliability_score * 0.4 + latency_score * 0.3 + success_rate_score * 0.3) * complexity_multiplier
    
    def _calculate_cost_score(self, models: List[ModelCapability], request: RoutingRequest) -> float:
        """计算成本评分（成本越低评分越高）"""
        total_cost = sum(model.cost_per_1k_tokens for model in models)
        
        # 成本评分（反向）
        max_reasonable_cost = 0.1  # 每1K tokens $0.1作为合理上限
        cost_score = max(0, 1 - total_cost / max_reasonable_cost)
        
        # 预算约束调整
        if request.budget_constraint:
            budget_score = max(0, 1 - total_cost / request.budget_constraint)
            cost_score = min(cost_score, budget_score)
        
        return cost_score
    
    def _calculate_availability_score(self, models: List[ModelCapability]) -> float:
        """计算可用性评分"""
        # 基于当前负载和可靠性
        load_scores = [1 - model.current_load for model in models]
        reliability_scores = [model.reliability for model in models]
        
        avg_load_score = np.mean(load_scores)
        avg_reliability = np.mean(reliability_scores)
        
        return (avg_load_score * 0.6 + avg_reliability * 0.4)
    
    def _calculate_collaboration_score(self, models: List[ModelCapability], request: RoutingRequest) -> float:
        """计算协作效应评分"""
        if len(models) <= 1:
            return 1.0
        
        # 模型互补性评分
        complementarity_score = 0.0
        all_capabilities = set()
        
        for model in models:
            model_capabilities = set(model.capabilities)
            new_capabilities = model_capabilities - all_capabilities
            complementarity_score += len(new_capabilities)
            all_capabilities.update(model_capabilities)
        
        complementarity_score = complementarity_score / len(all_capabilities) if all_capabilities else 0.0
        
        # 协作复杂度惩罚
        complexity_penalty = 0.1 * (len(models) - 1)
        
        return max(0, complementarity_score - complexity_penalty)
    
    def _calculate_quantum_weights(self, num_models: int) -> List[float]:
        """计算量子权重"""
        # 基于量子态的动态权重
        base_weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # 能力、性能、成本、可用性、协作
        
        # 量子纠缠调整
        quantum_adjustment = np.tanh(np.mean(self.quantum_state[:5]))
        
        # 模型数量调整
        model_factor = 1.0 / (1.0 + 0.1 * (num_models - 1))
        
        weights = []
        for i, base_weight in enumerate(base_weights):
            quantum_weight = base_weight * (1 + quantum_adjustment * 0.1) * model_factor
            weights.append(quantum_weight)
        
        # 归一化
        total_weight = sum(weights)
        return [w / total_weight for w in weights]
    
    def _determine_collaboration_strategy(self, model_names: List[str], request: RoutingRequest) -> str:
        """确定协作策略"""
        if len(model_names) == 1:
            return "single_model"
        
        if request.complexity == 'critical':
            return "parallel_consensus"
        elif request.complexity == 'high':
            return "sequential_refinement"
        elif request.urgency == 'high':
            return "parallel_voting"
        else:
            return "ensemble_averaging"
    
    def _quantum_entanglement_optimization(self, evaluations: List[Dict], request: RoutingRequest) -> List[Dict]:
        """量子纠缠优化"""
        # 更新纠缠矩阵
        self._update_entanglement_matrix(evaluations)
        
        # 应用纠缠效应
        optimized_evaluations = []
        
        for evaluation in evaluations:
            # 计算纠缠增益
            entanglement_gain = self._calculate_entanglement_gain(evaluation['models'])
            
            # 应用增益
            optimized_evaluation = evaluation.copy()
            optimized_evaluation['performance'] *= (1 + entanglement_gain)
            optimized_evaluation['confidence'] *= (1 + entanglement_gain * 0.5)
            
            optimized_evaluations.append(optimized_evaluation)
        
        return optimized_evaluations
    
    def _update_entanglement_matrix(self, evaluations: List[Dict]):
        """更新纠缠矩阵"""
        model_names = list(self.models.keys())
        n = len(model_names)
        
        # 基于历史性能更新纠缠强度
        for evaluation in evaluations:
            models = evaluation['models']
            performance = evaluation['performance']
            
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    if i != j and model1 in model_names and model2 in model_names:
                        idx1 = model_names.index(model1)
                        idx2 = model_names.index(model2)
                        
                        # 增强纠缠强度
                        self.entanglement_matrix[idx1][idx2] *= (1 + performance * 0.01)
                        self.entanglement_matrix[idx2][idx1] *= (1 + performance * 0.01)
        
        # 归一化纠缠矩阵
        max_entanglement = np.max(self.entanglement_matrix)
        if max_entanglement > 0:
            self.entanglement_matrix /= max_entanglement
    
    def _calculate_entanglement_gain(self, model_names: List[str]) -> float:
        """计算纠缠增益"""
        if len(model_names) <= 1:
            return 0.0
        
        model_indices = []
        for name in model_names:
            if name in self.models:
                model_indices.append(list(self.models.keys()).index(name))
        
        if len(model_indices) < 2:
            return 0.0
        
        # 计算模型间的纠缠强度
        total_entanglement = 0.0
        count = 0
        
        for i in range(len(model_indices)):
            for j in range(i + 1, len(model_indices)):
                idx1, idx2 = model_indices[i], model_indices[j]
                total_entanglement += self.entanglement_matrix[idx1][idx2]
                count += 1
        
        avg_entanglement = total_entanglement / count if count > 0 else 0.0
        
        # 纠缠增益
        return avg_entanglement * 0.2  # 最大20%的性能提升
    
    def _select_optimal_combination(self, evaluations: List[Dict], request: RoutingRequest) -> Dict:
        """选择最优组合"""
        # 过滤满足基本要求的组合
        filtered_evaluations = []
        
        for evaluation in evaluations:
            # 预算约束
            if request.budget_constraint and evaluation['cost'] > request.budget_constraint:
                continue
            
            # 质量要求
            if request.quality_requirement == 'high' and evaluation['confidence'] < 0.8:
                continue
            elif request.quality_requirement == 'critical' and evaluation['confidence'] < 0.9:
                continue
            
            # 延迟要求
            if request.urgency == 'high' and evaluation['latency'] > 1000:
                continue
            elif request.urgency == 'critical' and evaluation['latency'] > 500:
                continue
            
            filtered_evaluations.append(evaluation)
        
        if not filtered_evaluations:
            # 如果没有满足要求的组合，选择最好的可用组合
            filtered_evaluations = evaluations
        
        # 按性能排序
        filtered_evaluations.sort(key=lambda x: x['performance'], reverse=True)
        
        # 生成推理
        best = filtered_evaluations[0]
        best['reasoning'] = self._generate_reasoning(best, request, filtered_evaluations)
        
        return best
    
    def _generate_reasoning(self, best_evaluation: Dict, request: RoutingRequest, all_evaluations: List[Dict]) -> str:
        """生成决策推理"""
        reasoning_parts = []
        
        # 模型选择理由
        reasoning_parts.append(f"选择模型组合: {', '.join(best_evaluation['models'])}")
        
        # 性能优势
        if len(all_evaluations) > 1:
            performance_diff = best_evaluation['performance'] - all_evaluations[1]['performance']
            reasoning_parts.append(f"性能优势: +{performance_diff:.3f}")
        
        # 成本考虑
        if request.budget_constraint:
            reasoning_parts.append(f"预估成本: ${best_evaluation['cost']:.4f} (预算: ${request.budget_constraint})")
        
        # 协作策略
        reasoning_parts.append(f"协作策略: {best_evaluation['strategy']}")
        
        # 置信度
        reasoning_parts.append(f"决策置信度: {best_evaluation['confidence']:.1%}")
        
        return " | ".join(reasoning_parts)
    
    def _update_quantum_state(self, request: RoutingRequest, decision: RoutingDecision):
        """更新量子态"""
        # 基于请求和决策更新量子态
        state_update = np.random.randn(64) * 0.01  # 小幅随机扰动
        
        # 基于性能反馈调整
        performance_feedback = decision.confidence * 0.1
        self.quantum_state = self.quantum_state * (1 - performance_feedback) + state_update
        
        # 归一化
        norm = np.linalg.norm(self.quantum_state)
        if norm > 0:
            self.quantum_state /= norm
    
    def detect_ide_models(self) -> Dict[str, str]:
        """检测用户当前IDE的AI模型"""
        detected_models = {}
        
        # 检测VS Code
        vscode_models = self._detect_vscode_models()
        if vscode_models:
            detected_models['vscode'] = vscode_models
        
        # 检测JetBrains IDEs
        jetbrains_models = self._detect_jetbrains_models()
        if jetbrains_models:
            detected_models['jetbrains'] = jetbrains_models
        
        # 检测其他IDE
        other_ide_models = self._detect_other_ide_models()
        if other_ide_models:
            detected_models.update(other_ide_models)
        
        logger.info(f"检测到IDE模型: {detected_models}")
        return detected_models
    
    def _detect_vscode_models(self) -> Optional[str]:
        """检测VS Code的AI模型"""
        try:
            # 检查VS Code进程
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if proc.info['name'] == 'Code.exe' or 'code' in proc.info.get('name', '').lower():
                    cmdline = proc.info.get('cmdline', [])
                    
                    # 检查GitHub Copilot
                    if any('copilot' in str(cmd).lower() for cmd in cmdline):
                        return 'GitHub Copilot (GPT-4)'
                    
                    # 检查其他VS Code AI扩展
                    if any('ai' in str(cmd).lower() for cmd in cmdline):
                        return 'VS Code AI Extension'
            
            # 检查VS Code设置文件
            vscode_settings = Path.home() / 'AppData' / 'Roaming' / 'Code' / 'User' / 'settings.json'
            if vscode_settings.exists():
                try:
                    with open(vscode_settings, 'r', encoding='utf-8') as f:
                        settings = json.load(f)
                        
                    if 'github.copilot' in settings:
                        return 'GitHub Copilot (GPT-4)'
                        
                except:
                    pass
            
        except Exception as e:
            logger.debug(f"VS Code检测失败: {e}")
        
        return None
    
    def _detect_jetbrains_models(self) -> Optional[str]:
        """检测JetBrains IDE的AI模型"""
        try:
            # 检查JetBrains进程
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                name = proc.info.get('name', '').lower()
                if any(ide in name for ide in ['idea', 'pycharm', 'webstorm', 'phpstorm', 'goland']):
                    cmdline = proc.info.get('cmdline', [])
                    
                    # 检查JetBrains AI
                    if any('ai' in str(cmd).lower() for cmd in cmdline):
                        return 'JetBrains AI Assistant'
            
        except Exception as e:
            logger.debug(f"JetBrains检测失败: {e}")
        
        return None
    
    def _detect_other_ide_models(self) -> Dict[str, str]:
        """检测其他IDE的AI模型"""
        detected = {}
        
        try:
            # 检查常见AI工具进程
            ai_processes = {
                'Cursor': 'Cursor AI',
                'Tabnine': 'Tabnine AI',
                'Codeium': 'Codeium AI',
                'Amazon CodeWhisperer': 'Amazon CodeWhisperer'
            }
            
            for proc in psutil.process_iter(['pid', 'name']):
                name = proc.info.get('name', '')
                for ai_tool, model_name in ai_processes.items():
                    if ai_tool.lower() in name.lower():
                        detected[ai_tool] = model_name
                        break
            
        except Exception as e:
            logger.debug(f"其他IDE检测失败: {e}")
        
        return detected
    
    def update_model_performance(self, model_name: str, success: bool, response_time: float):
        """更新模型性能"""
        if model_name in self.models:
            model = self.models[model_name]
            
            # 更新成功率
            if success:
                self.success_count += 1
                model.success_rate = 0.9 * model.success_rate + 0.1 * 1.0
            else:
                model.success_rate = 0.9 * model.success_rate + 0.1 * 0.0
            
            # 更新响应时间
            model.last_response_time = response_time
            model.latency_ms = 0.8 * model.latency_ms + 0.2 * response_time
            
            # 更新负载
            model.current_load = max(0, model.current_load - 0.1)
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        success_rate = self.success_count / self.request_count if self.request_count > 0 else 0
        avg_latency = self.total_latency / self.request_count if self.request_count > 0 else 0
        
        model_stats = {}
        for name, model in self.models.items():
            model_stats[name] = {
                'success_rate': model.success_rate,
                'avg_latency': model.latency_ms,
                'current_load': model.current_load,
                'reliability': model.reliability
            }
        
        return {
            'total_requests': self.request_count,
            'success_rate': success_rate,
            'average_latency': avg_latency,
            'model_statistics': model_stats,
            'cache_hit_rate': len(self.performance_cache) / max(1, self.request_count),
            'quantum_coherence': np.linalg.norm(self.quantum_state),
            'entanglement_strength': np.mean(self.entanglement_matrix)
        }

# 示例使用
def example_routing():
    """示例路由使用"""
    router = QuantumRouter()
    
    # 检测IDE模型
    ide_models = router.detect_ide_models()
    print(f"检测到的IDE模型: {ide_models}")
    
    # 创建路由请求
    request = RoutingRequest(
        task_type='coding',
        complexity='high',
        language='python',
        context_length=5000,
        urgency='medium',
        budget_constraint=0.1,
        quality_requirement='high',
        special_requirements=['functions'],
        user_context={'project_type': 'web_development'}
    )
    
    # 路由请求
    decision = router.route_request(request)
    
    print(f"路由决策:")
    print(f"  选择模型: {decision.selected_models}")
    print(f"  协作策略: {decision.collaboration_strategy}")
    print(f"  预期性能: {decision.expected_performance:.3f}")
    print(f"  估算成本: ${decision.estimated_cost:.4f}")
    print(f"  估算延迟: {decision.estimated_latency:.0f}ms")
    print(f"  置信度: {decision.confidence:.1%}")
    print(f"  推理: {decision.reasoning}")
    
    # 模拟更新性能
    router.update_model_performance(decision.selected_models[0], True, 550)
    
    # 获取统计信息
    stats = router.get_routing_statistics()
    print(f"\n路由统计:")
    print(f"  总请求数: {stats['total_requests']}")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  平均延迟: {stats['average_latency']:.0f}ms")
    print(f"  缓存命中率: {stats['cache_hit_rate']:.1%}")

if __name__ == "__main__":
    example_routing()
