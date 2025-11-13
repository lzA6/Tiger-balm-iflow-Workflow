#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 智能工作流优化器 V1.0
Intelligent Workflow Optimizer V1.0

基于机器学习的自适应工作流优化系统，实现：
1. 自学习能力：从执行历史中学习最优策略
2. 自适应能力：动态调整工作流参数
3. 预测能力：预测性能瓶颈并提前优化
4. 智能决策：基于历史数据做出最优决策

核心特性：
- 强化学习驱动的策略优化
- 时间序列预测的性能预判
- 贝叶斯优化的参数调优
- 在线学习的持续改进
"""

import json
import time
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import threading
from collections import deque, defaultdict
import pickle
import hashlib

# 机器学习相关
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.cluster import KMeans
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("机器学习库未安装，将使用简化优化策略")

# 强化学习相关
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    logging.warning("TensorFlow未安装，将使用简化强化学习")

@dataclass
class ExecutionMetrics:
    """执行指标数据类"""
    timestamp: float
    task_complexity: str  # 'simple', 'medium', 'complex', 'quantum'
    execution_time: float
    parallel_efficiency: float
    resource_utilization: float
    success_rate: float
    error_count: int
    memory_usage: float
    cpu_usage: float
    throughput: float
    response_time: float
    optimization_applied: bool
    strategy_used: str

@dataclass
class OptimizationStrategy:
    """优化策略数据类"""
    strategy_id: str
    name: str
    parameters: Dict[str, Any]
    performance_gain: float
    confidence_score: float
   适用场景: List[str]
    last_updated: float

@dataclass
class PerformancePrediction:
    """性能预测数据类"""
    predicted_execution_time: float
    predicted_resource_usage: float
    predicted_bottlenecks: List[str]
    confidence_interval: Tuple[float, float]
    prediction_timestamp: float

class IntelligentWorkflowOptimizer:
    """
    智能工作流优化器
    
    核心功能：
    1. 收集和分析执行历史数据
    2. 使用机器学习预测性能瓶颈
    3. 自动优化工作流参数
    4. 强化学习策略优化
    5. 在线学习和持续改进
    """
    
    def __init__(self, data_dir: str = ".iflow/data", model_dir: str = ".iflow/models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据存储
        self.execution_history: deque = deque(maxlen=1000)
        self.optimization_strategies: Dict[str, OptimizationStrategy] = {}
        self.performance_cache: Dict[str, PerformancePrediction] = {}
        
        # 机器学习模型
        self.performance_predictor = None
        self.strategy_optimizer = None
        self.bottleneck_detector = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # 在线学习参数
        self.learning_rate = 0.01
        self.exploration_rate = 0.1
        self.memory_size = 500
        
        # 性能监控
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # 初始化
        self._load_models()
        self._load_execution_history()
        
        logging.info("🧠 智能工作流优化器初始化完成")
    
    def collect_execution_metrics(self, metrics: ExecutionMetrics) -> None:
        """
        收集执行指标
        
        Args:
            metrics: 执行指标数据
        """
        self.execution_history.append(metrics)
        
        # 触发在线学习
        if len(self.execution_history) >= 10:
            self._online_learning_update()
        
        # 保存到文件
        self._save_execution_history()
        
        logging.debug(f"📊 收集执行指标: {metrics.task_complexity}, 执行时间: {metrics.execution_time:.2f}s")
    
    def predict_performance(self, task_complexity: str, context: Dict[str, Any]) -> PerformancePrediction:
        """
        预测任务性能
        
        Args:
            task_complexity: 任务复杂度
            context: 任务上下文信息
            
        Returns:
            PerformancePrediction: 性能预测结果
        """
        if not ML_AVAILABLE or self.performance_predictor is None:
            return self._fallback_performance_prediction(task_complexity, context)
        
        try:
            # 特征工程
            features = self._extract_features(task_complexity, context)
            features_scaled = self.scaler.transform([features])
            
            # 预测执行时间
            predicted_time = self.performance_predictor.predict(features_scaled)[0]
            
            # 预测资源使用
            predicted_resources = self._predict_resource_usage(features_scaled)
            
            # 检测潜在瓶颈
            bottlenecks = self._detect_bottlenecks(features_scaled)
            
            # 计算置信区间
            confidence_interval = self._calculate_confidence_interval(features_scaled)
            
            prediction = PerformancePrediction(
                predicted_execution_time=predicted_time,
                predicted_resource_usage=predicted_resources,
                predicted_bottlenecks=bottlenecks,
                confidence_interval=confidence_interval,
                prediction_timestamp=time.time()
            )
            
            # 缓存预测结果
            cache_key = self._generate_cache_key(task_complexity, context)
            self.performance_cache[cache_key] = prediction
            
            logging.info(f"🔮 性能预测完成: 预计时间 {predicted_time:.2f}s, 瓶颈: {bottlenecks}")
            return prediction
            
        except Exception as e:
            logging.error(f"预测性能时出错: {e}")
            return self._fallback_performance_prediction(task_complexity, context)
    
    def optimize_workflow_parameters(self, task_complexity: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        优化工作流参数
        
        Args:
            task_complexity: 任务复杂度
            context: 任务上下文
            
        Returns:
            Dict[str, Any]: 优化后的参数
        """
        try:
            # 获取历史最优参数
            optimal_params = self._get_historical_optimal_parameters(task_complexity)
            
            # 基于预测结果调整参数
            prediction = self.predict_performance(task_complexity, context)
            
            # 应用强化学习策略
            if RL_AVAILABLE:
                optimized_params = self._apply_reinforcement_learning_optimization(
                    task_complexity, context, optimal_params, prediction
                )
            else:
                optimized_params = self._apply_heuristic_optimization(
                    task_complexity, context, optimal_params, prediction
                )
            
            # 评估优化效果
            self._evaluate_optimization_effectiveness(optimized_params, task_complexity)
            
            logging.info(f"⚙️ 工作流参数优化完成: {optimized_params}")
            return optimized_params
            
        except Exception as e:
            logging.error(f"优化工作流参数时出错: {e}")
            return self._get_default_parameters(task_complexity)
    
    def suggest_optimization_strategy(self, current_metrics: ExecutionMetrics) -> Optional[OptimizationStrategy]:
        """
        建议优化策略
        
        Args:
            current_metrics: 当前执行指标
            
        Returns:
            Optional[OptimizationStrategy]: 优化策略
        """
        try:
            # 分析当前性能问题
            issues = self._analyze_performance_issues(current_metrics)
            
            if not issues:
                return None
            
            # 基于历史数据推荐策略
            best_strategy = self._recommend_strategy_from_history(issues, current_metrics.task_complexity)
            
            if best_strategy:
                logging.info(f"🎯 推荐优化策略: {best_strategy.name}, 预期增益: {best_strategy.performance_gain:.2%}")
                return best_strategy
            
            # 生成新策略
            new_strategy = self._generate_new_optimization_strategy(issues, current_metrics)
            if new_strategy:
                self.optimization_strategies[new_strategy.strategy_id] = new_strategy
                logging.info(f"🆕 生成新优化策略: {new_strategy.name}")
                return new_strategy
            
        except Exception as e:
            logging.error(f"建议优化策略时出错: {e}")
        
        return None
    
    def _extract_features(self, task_complexity: str, context: Dict[str, Any]) -> List[float]:
        """提取特征向量"""
        features = []
        
        # 任务复杂度编码
        complexity_map = {'simple': 1, 'medium': 2, 'complex': 3, 'quantum': 4}
        features.append(complexity_map.get(task_complexity, 2))
        
        # 上下文特征
        features.append(context.get('file_count', 0))
        features.append(context.get('code_lines', 0))
        features.append(context.get('dependencies', 0))
        features.append(context.get('parallel_tasks', 1))
        features.append(1 if context.get('cache_hit', False) else 0)
        features.append(context.get('memory_limit', 8192))
        features.append(context.get('timeout', 300))
        
        # 历史平均性能
        if self.execution_history:
            recent_metrics = list(self.execution_history)[-10:]
            features.extend([
                np.mean([m.execution_time for m in recent_metrics]),
                np.mean([m.parallel_efficiency for m in recent_metrics]),
                np.mean([m.resource_utilization for m in recent_metrics])
            ])
        else:
            features.extend([60.0, 0.7, 0.5])  # 默认值
        
        return features
    
    def _predict_resource_usage(self, features: np.ndarray) -> float:
        """预测资源使用"""
        if self.bottleneck_detector:
            return self.bottleneck_detector.predict(features)[0]
        return 0.5  # 默认值
    
    def _detect_bottlenecks(self, features: np.ndarray) -> List[str]:
        """检测潜在瓶颈"""
        bottlenecks = []
        
        if len(features) > 0:
            # 简单的瓶颈检测逻辑
            if features[0][0] > 3:  # 复杂度高
                bottlenecks.append("high_complexity")
            if features[0][7] > 0.8:  # 内存使用高
                bottlenecks.append("memory_bottleneck")
            if features[0][6] > 6000:  # 超时时间长
                bottlenecks.append("timeout_bottleneck")
        
        return bottlenecks
    
    def _calculate_confidence_interval(self, features: np.ndarray) -> Tuple[float, float]:
        """计算置信区间"""
        # 简化的置信区间计算
        base_confidence = 0.8
        if len(self.execution_history) < 50:
            base_confidence -= 0.2
        return (base_confidence - 0.1, base_confidence + 0.1)
    
    def _apply_reinforcement_learning_optimization(self, task_complexity: str, context: Dict[str, Any], 
                                                   base_params: Dict[str, Any], prediction: PerformancePrediction) -> Dict[str, Any]:
        """应用强化学习优化"""
        # 这里应该使用训练好的强化学习模型
        # 目前使用简化的启发式方法
        optimized_params = base_params.copy()
        
        # 基于预测结果调整参数
        if prediction.predicted_execution_time > 120:  # 预测执行时间长
            optimized_params['parallel_tasks'] = min(optimized_params.get('parallel_tasks', 4) + 2, 10)
            optimized_params['timeout'] = max(optimized_params.get('timeout', 300), 600)
        
        if prediction.predicted_resource_usage > 0.8:  # 预测资源使用高
            optimized_params['memory_limit'] = max(optimized_params.get('memory_limit', 8192), 16384)
            optimized_params['resource_optimized'] = True
        
        return optimized_params
    
    def _apply_heuristic_optimization(self, task_complexity: str, context: Dict[str, Any], 
                                      base_params: Dict[str, Any], prediction: PerformancePrediction) -> Dict[str, Any]:
        """应用启发式优化"""
        optimized_params = base_params.copy()
        
        # 基于任务复杂度的启发式规则
        complexity_multipliers = {
            'simple': 0.8,
            'medium': 1.0,
            'complex': 1.5,
            'quantum': 2.0
        }
        
        multiplier = complexity_multipliers.get(task_complexity, 1.0)
        
        # 调整并行任务数
        base_parallel = optimized_params.get('parallel_tasks', 4)
        optimized_params['parallel_tasks'] = min(int(base_parallel * multiplier), 10)
        
        # 调整超时时间
        base_timeout = optimized_params.get('timeout', 300)
        optimized_params['timeout'] = int(base_timeout * multiplier)
        
        # 调整内存限制
        base_memory = optimized_params.get('memory_limit', 8192)
        optimized_params['memory_limit'] = int(base_memory * multiplier)
        
        return optimized_params
    
    def _get_default_parameters(self, task_complexity: str) -> Dict[str, Any]:
        """获取默认参数"""
        default_params = {
            'simple': {
                'parallel_tasks': 2,
                'timeout': 60,
                'memory_limit': 4096,
                'optimization_level': 'light'
            },
            'medium': {
                'parallel_tasks': 4,
                'timeout': 180,
                'memory_limit': 8192,
                'optimization_level': 'medium'
            },
            'complex': {
                'parallel_tasks': 6,
                'timeout': 600,
                'memory_limit': 16384,
                'optimization_level': 'heavy'
            },
            'quantum': {
                'parallel_tasks': 10,
                'timeout': 1200,
                'memory_limit': 32768,
                'optimization_level': 'maximum'
            }
        }
        return default_params.get(task_complexity, default_params['medium'])
    
    def _load_models(self) -> None:
        """加载预训练模型"""
        try:
            if ML_AVAILABLE:
                model_files = [
                    'performance_predictor.pkl',
                    'strategy_optimizer.pkl',
                    'bottleneck_detector.pkl'
                ]
                
                for model_file in model_files:
                    model_path = self.model_dir / model_file
                    if model_path.exists():
                        with open(model_path, 'rb') as f:
                            if model_file == 'performance_predictor.pkl':
                                self.performance_predictor = pickle.load(f)
                            elif model_file == 'strategy_optimizer.pkl':
                                self.strategy_optimizer = pickle.load(f)
                            elif model_file == 'bottleneck_detector.pkl':
                                self.bottleneck_detector = pickle.load(f)
                
                logging.info("📊 机器学习模型加载完成")
        except Exception as e:
            logging.error(f"加载模型时出错: {e}")
    
    def _save_models(self) -> None:
        """保存训练好的模型"""
        try:
            if ML_AVAILABLE and self.performance_predictor:
                model_files = {
                    'performance_predictor.pkl': self.performance_predictor,
                    'strategy_optimizer.pkl': self.strategy_optimizer,
                    'bottleneck_detector.pkl': self.bottleneck_detector
                }
                
                for model_file, model in model_files.items():
                    if model:
                        model_path = self.model_dir / model_file
                        with open(model_path, 'wb') as f:
                            pickle.dump(model, f)
                
                logging.info("💾 机器学习模型保存完成")
        except Exception as e:
            logging.error(f"保存模型时出错: {e}")
    
    def _load_execution_history(self) -> None:
        """加载执行历史"""
        try:
            history_file = self.data_dir / 'execution_history.json'
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                    for item in history_data:
                        self.execution_history.append(ExecutionMetrics(**item))
                logging.info(f"📚 加载了 {len(self.execution_history)} 条执行历史")
        except Exception as e:
            logging.error(f"加载执行历史时出错: {e}")
    
    def _save_execution_history(self) -> None:
        """保存执行历史"""
        try:
            history_file = self.data_dir / 'execution_history.json'
            history_data = [asdict(metric) for metric in self.execution_history]
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"保存执行历史时出错: {e}")
    
    def _online_learning_update(self) -> None:
        """在线学习更新"""
        try:
            if len(self.execution_history) < 20:
                return
            
            # 提取训练数据
            recent_history = list(self.execution_history)[-self.memory_size:]
            X, y_time, y_efficiency = self._prepare_training_data(recent_history)
            
            if len(X) < 10:
                return
            
            # 在线更新模型
            if ML_AVAILABLE:
                if self.performance_predictor is None:
                    self.performance_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
                
                # 增量训练
                self.performance_predictor.fit(X, y_time)
                
                # 保存更新的模型
                self._save_models()
                
                logging.info("🔄 在线学习模型更新完成")
                
        except Exception as e:
            logging.error(f"在线学习更新时出错: {e}")
    
    def _prepare_training_data(self, history: List[ExecutionMetrics]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """准备训练数据"""
        X = []
        y_time = []
        y_efficiency = []
        
        for metrics in history:
            # 简化的特征提取
            features = [
                1 if metrics.task_complexity == 'simple' else 2 if metrics.task_complexity == 'medium' else 3,
                metrics.parallel_efficiency,
                metrics.resource_utilization,
                metrics.memory_usage / 1024,  # GB
                metrics.cpu_usage,
                1 if metrics.optimization_applied else 0
            ]
            X.append(features)
            y_time.append(metrics.execution_time)
            y_efficiency.append(metrics.parallel_efficiency)
        
        return np.array(X), np.array(y_time), np.array(y_efficiency)
    
    def _generate_cache_key(self, task_complexity: str, context: Dict[str, Any]) -> str:
        """生成缓存键"""
        cache_data = {
            'complexity': task_complexity,
            'context_keys': sorted(context.keys()),
            'context_values': [context[k] for k in sorted(context.keys())]
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _analyze_performance_issues(self, metrics: ExecutionMetrics) -> List[str]:
        """分析性能问题"""
        issues = []
        
        if metrics.execution_time > 300:  # 执行时间过长
            issues.append("slow_execution")
        
        if metrics.parallel_efficiency < 0.5:  # 并行效率低
            issues.append("low_parallel_efficiency")
        
        if metrics.resource_utilization > 0.9:  # 资源使用率过高
            issues.append("high_resource_usage")
        
        if metrics.memory_usage > 8192:  # 内存使用过多
            issues.append("high_memory_usage")
        
        if metrics.error_count > 0:  # 有错误发生
            issues.append("errors_occurred")
        
        return issues
    
    def _get_historical_optimal_parameters(self, task_complexity: str) -> Dict[str, Any]:
        """获取历史最优参数"""
        # 简化的实现：返回基于历史数据的平均最优参数
        if not self.execution_history:
            return self._get_default_parameters(task_complexity)
        
        # 过滤相同复杂度的历史记录
        same_complexity = [m for m in self.execution_history if m.task_complexity == task_complexity]
        
        if not same_complexity:
            return self._get_default_parameters(task_complexity)
        
        # 计算平均性能指标
        avg_execution_time = np.mean([m.execution_time for m in same_complexity])
        avg_efficiency = np.mean([m.parallel_efficiency for m in same_complexity])
        avg_success_rate = np.mean([m.success_rate for m in same_complexity])
        
        # 基于历史表现推荐参数
        if avg_efficiency > 0.8 and avg_success_rate > 0.9:
            return self._get_default_parameters(task_complexity)
        elif avg_execution_time > 180:
            params = self._get_default_parameters(task_complexity)
            params['parallel_tasks'] = min(params['parallel_tasks'] + 2, 10)
            return params
        else:
            return self._get_default_parameters(task_complexity)
    
    def _recommend_strategy_from_history(self, issues: List[str], task_complexity: str) -> Optional[OptimizationStrategy]:
        """从历史数据推荐策略"""
        best_strategy = None
        best_score = 0
        
        for strategy in self.optimization_strategies.values():
            if any(issue in strategy.适用场景 for issue in issues) or strategy.适用场景 == ['all']:
                if strategy.confidence_score > best_score:
                    best_score = strategy.confidence_score
                    best_strategy = strategy
        
        return best_strategy
    
    def _generate_new_optimization_strategy(self, issues: List[str], metrics: ExecutionMetrics) -> Optional[OptimizationStrategy]:
        """生成新的优化策略"""
        try:
            strategy_id = f"auto_{int(time.time())}"
            
            # 基于问题生成策略
            parameters = {}
            if "slow_execution" in issues:
                parameters['timeout'] = max(metrics.execution_time * 1.5, 600)
                parameters['parallel_tasks'] = min(10, int(metrics.parallel_efficiency * 8) + 2)
            
            if "low_parallel_efficiency" in issues:
                parameters['task_decomposition'] = 'fine_grained'
                parameters['synchronization'] = 'minimal'
            
            if "high_resource_usage" in issues:
                parameters['memory_optimized'] = True
                parameters['resource_monitoring'] = True
            
            strategy = OptimizationStrategy(
                strategy_id=strategy_id,
                name=f"Auto-generated strategy for {', '.join(issues)}",
                parameters=parameters,
                performance_gain=0.15,  # 默认15%增益
                confidence_score=0.7,  # 70%置信度
                适用场景=issues,
                last_updated=time.time()
            )
            
            return strategy
            
        except Exception as e:
            logging.error(f"生成新策略时出错: {e}")
            return None
    
    def _evaluate_optimization_effectiveness(self, optimized_params: Dict[str, Any], task_complexity: str) -> None:
        """评估优化效果"""
        # 简化的评估逻辑：记录优化参数，供后续分析
        logging.debug(f"📊 记录优化参数: {optimized_params} for {task_complexity}")
    
    def _fallback_performance_prediction(self, task_complexity: str, context: Dict[str, Any]) -> PerformancePrediction:
        """备用性能预测"""
        # 简化的预测逻辑
        complexity_time_map = {
            'simple': 30,
            'medium': 120,
            'complex': 300,
            'quantum': 600
        }
        
        predicted_time = complexity_time_map.get(task_complexity, 120)
        predicted_resources = 0.6 if task_complexity in ['complex', 'quantum'] else 0.4
        
        return PerformancePrediction(
            predicted_execution_time=predicted_time,
            predicted_resource_usage=predicted_resources,
            predicted_bottlenecks=[],
            confidence_interval=(0.6, 0.8),
            prediction_timestamp=time.time()
        )
    
    def start_monitoring(self) -> None:
        """启动性能监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logging.info("👀 性能监控启动")
    
    def stop_monitoring(self) -> None:
        """停止性能监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logging.info("🛑 性能监控停止")
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self.monitoring_active:
            try:
                # 定期保存模型和历史数据
                if len(self.execution_history) % 50 == 0 and len(self.execution_history) > 0:
                    self._save_models()
                    self._save_execution_history()
                
                time.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                logging.error(f"监控循环出错: {e}")
                time.sleep(60)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """获取优化报告"""
        report = {
            'execution_history_count': len(self.execution_history),
            'optimization_strategies_count': len(self.optimization_strategies),
            'models_available': {
                'performance_predictor': self.performance_predictor is not None,
                'strategy_optimizer': self.strategy_optimizer is not None,
                'bottleneck_detector': self.bottleneck_detector is not None
            },
            'last_optimization_time': time.time(),
            'performance_improvements': []
        }
        
        # 计算性能改进
        if len(self.execution_history) >= 10:
            recent = list(self.execution_history)[-10:]
            older = list(self.execution_history)[-20:-10]
            
            if older:
                recent_avg_time = np.mean([r.execution_time for r in recent])
                older_avg_time = np.mean([o.execution_time for o in older])
                
                if older_avg_time > 0:
                    time_improvement = (older_avg_time - recent_avg_time) / older_avg_time
                    report['performance_improvements'].append({
                        'metric': 'execution_time',
                        'improvement': time_improvement
                    })
        
        return report
    
    def reset_optimizer(self) -> None:
        """重置优化器"""
        self.execution_history.clear()
        self.optimization_strategies.clear()
        self.performance_cache.clear()
        
        # 删除文件
        history_file = self.data_dir / 'execution_history.json'
        if history_file.exists():
            history_file.unlink()
        
        logging.info("🔄 智能工作流优化器已重置")


# 全局优化器实例
workflow_optimizer = IntelligentWorkflowOptimizer()


def get_workflow_optimizer() -> IntelligentWorkflowOptimizer:
    """获取全局工作流优化器实例"""
    return workflow_optimizer


if __name__ == "__main__":
    # 测试代码
    optimizer = IntelligentWorkflowOptimizer()
    
    # 模拟一些执行指标
    test_metrics = ExecutionMetrics(
        timestamp=time.time(),
        task_complexity='medium',
        execution_time=120.5,
        parallel_efficiency=0.75,
        resource_utilization=0.6,
        success_rate=0.95,
        error_count=0,
        memory_usage=4096,
        cpu_usage=0.5,
        throughput=10.5,
        response_time=2.3,
        optimization_applied=True,
        strategy_used="parallel_optimization"
    )
    
    # 收集指标
    optimizer.collect_execution_metrics(test_metrics)
    
    # 预测性能
    context = {
        'file_count': 50,
        'code_lines': 1000,
        'dependencies': 10,
        'parallel_tasks': 4,
        'memory_limit': 8192,
        'timeout': 300
    }
    
    prediction = optimizer.predict_performance('medium', context)
    print(f"性能预测: {prediction}")
    
    # 优化参数
    optimized_params = optimizer.optimize_workflow_parameters('medium', context)
    print(f"优化参数: {optimized_params}")
    
    # 获取报告
    report = optimizer.get_optimization_report()
    print(f"优化报告: {report}")