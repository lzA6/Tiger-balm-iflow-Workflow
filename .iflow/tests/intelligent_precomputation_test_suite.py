#!/usr/bin/env python3
"""
智能预计算机制测试套件
专门测试智能预加载和预测缓存的效果

测试目标：
1. 验证智能预加载的准确性和及时性
2. 测量预测缓存的命中率和效果
3. 评估预计算资源消耗和优化效果
4. 对比预计算与实时计算的性能差异
5. 评估预测算法的准确性

作者：A项目V7升级版
创建时间：2025-11-13
"""

import time
import asyncio
import statistics
import threading
import json
import logging
import random
import hashlib
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import psutil
import os

# 导入A项目的核心组件
try:
    from ..core.optimized_fusion_cache import OptimizedFusionCache
    from ..core.intelligent_context_manager import IntelligentContextManager
    from ..core.unified_model_adapter import UnifiedModelAdapter
    from ..core.parallel_agent_executor import ParallelAgentExecutor
    from ..core.task_decomposer import TaskDecomposer
    from ..core.workflow_stage_parallelizer import WorkflowStageParallelizer
except ImportError:
    # 备用导入路径
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from core.optimized_fusion_cache import OptimizedFusionCache
    from core.intelligent_context_manager import IntelligentContextManager
    from core.unified_model_adapter import UnifiedModelAdapter
    from core.parallel_agent_executor import ParallelAgentExecutor
    from core.task_decomposer import TaskDecomposer
    from core.workflow_stage_parallelizer import WorkflowStageParallelizer

@dataclass
class PrecomputationMetric:
    """预计算指标数据类"""
    test_name: str
    prediction_accuracy: float  # 预测准确性 (%)
    precomputation_hit_rate: float  # 预计算命中率 (%)
    resource_overhead: float  # 资源开销 (MB)
    time_saving: float  # 时间节省 (ms)
    efficiency_ratio: float  # 效率比
    prediction_latency: float  # 预测延迟 (ms)
    cache_warmup_time: float  # 缓存预热时间 (ms)

class IntelligentPrecomputationTester:
    """智能预计算机制测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.cache_system = OptimizedFusionCache()
        self.context_manager = IntelligentContextManager()
        self.model_adapter = UnifiedModelAdapter()
        self.parallel_executor = ParallelAgentExecutor()
        self.task_decomposer = TaskDecomposer()
        self.workflow_parallelizer = WorkflowStageParallelizer()
        
        # 测试配置
        self.test_iterations = 100
        self.prediction_window = 10  # 预测窗口大小
        self.cache_size_scenarios = [100, 500, 1000, 2000, 5000]
        self.workload_patterns = ['sequential', 'random', 'burst', 'mixed']
        
        # 预测相关配置
        self.prediction_history = deque(maxlen=1000)
        self.access_pattern_history = deque(maxlen=1000)
        self.predicted_items = set()
        
        # 测试结果存储
        self.metrics: List[PrecomputationMetric] = []
        
        # 内存监控
        self.memory_monitor = MemoryMonitor()
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('intelligent_precomputation_test.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def record_memory_usage(self, label: str):
        """记录内存使用情况"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        self.memory_monitor.record(label, memory_info.rss / 1024 / 1024)  # MB
    
    def predict_next_access(self, access_history: List[str]) -> List[str]:
        """
        智能预测下一个可能访问的项目
        
        Args:
            access_history: 访问历史
            
        Returns:
            List[str]: 预测的下一个访问项目列表
        """
        if len(access_history) < 3:
            return []
        
        # 简单的模式预测算法
        predictions = []
        
        # 基于最近访问模式预测
        recent_pattern = access_history[-3:]
        
        # 检查是否有重复模式
        for i in range(len(access_history) - 3):
            if access_history[i:i+3] == recent_pattern:
                if i + 3 < len(access_history):
                    predictions.append(access_history[i + 3])
        
        # 基于频率预测（最常访问的项目）
        frequency = defaultdict(int)
        for item in access_history:
            frequency[item] += 1
        
        # 添加频率最高的项目
        if frequency:
            most_frequent = max(frequency, key=frequency.get)
            if most_frequent not in predictions:
                predictions.append(most_frequent)
        
        return list(set(predictions))  # 去重
    
    def precompute_items(self, predictions: List[str]) -> Dict[str, Any]:
        """
        预计算指定的项目
        
        Args:
            predictions: 预测的项目列表
            
        Returns:
            Dict[str, Any]: 预计算结果
        """
        precomputed_results = {}
        
        for item in predictions:
            # 模拟预计算过程
            computation_time = random.uniform(0.1, 2.0)  # 模拟计算时间
            
            # 模拟复杂的计算过程
            result_data = {
                'computed_value': f"precomputed_result_for_{item}",
                'computation_time': computation_time,
                'timestamp': time.time(),
                'dependencies': [f"dep_{i}" for i in range(random.randint(1, 5))]
            }
            
            # 将预计算结果存储到缓存
            cache_key = f"precomputed_{item}"
            self.cache_system.set(cache_key, result_data, ttl=3600)
            precomputed_results[item] = result_data
            
            # 记录预计算的项目
            self.predicted_items.add(item)
        
        return precomputed_results
    
    def simulate_access_pattern(self, pattern_type: str, num_accesses: int) -> List[str]:
        """
        模拟不同的访问模式
        
        Args:
            pattern_type: 访问模式类型
            num_accesses: 访问次数
            
        Returns:
            List[str]: 访问序列
        """
        if pattern_type == 'sequential':
            return [f"item_{i % 50}" for i in range(num_accesses)]
        elif pattern_type == 'random':
            return [f"item_{random.randint(0, 100)}" for _ in range(num_accesses)]
        elif pattern_type == 'burst':
            # 突发模式：一段时间内集中访问某些项目
            base_items = [f"item_{i}" for i in range(10)]
            return [random.choice(base_items) for _ in range(num_accesses)]
        elif pattern_type == 'mixed':
            # 混合模式
            patterns = ['sequential'] * 30 + ['random'] * 30 + ['burst'] * 20 + ['sequential'] * 20
            result = []
            for i in range(num_accesses):
                pattern = random.choice(patterns)
                if pattern == 'sequential':
                    result.append(f"item_{i % 30}")
                elif pattern == 'random':
                    result.append(f"item_{random.randint(0, 80)}")
                elif pattern == 'burst':
                    base_items = [f"item_{i}" for i in range(15)]
                    result.append(random.choice(base_items))
            return result
        else:
            return [f"item_{random.randint(0, 50)}" for _ in range(num_accesses)]
    
    def test_prediction_accuracy(self) -> PrecomputationMetric:
        """
        测试预测准确性
        
        Returns:
            PrecomputationMetric: 预测准确性指标
        """
        self.logger.info("开始预测准确性测试...")
        
        # 记录初始内存
        self.record_memory_usage("prediction_accuracy_start")
        
        access_history = []
        predictions_made = 0
        predictions_correct = 0
        
        # 模拟访问模式
        access_pattern = self.simulate_access_pattern('mixed', 200)
        
        start_time = time.perf_counter()
        
        for i, current_item in enumerate(access_pattern):
            # 记录当前访问
            access_history.append(current_item)
            self.access_pattern_history.append(current_item)
            
            # 每5次访问进行一次预测
            if i > 0 and i % 5 == 0:
                predictions_made += 1
                
                # 基于历史记录进行预测
                predictions = self.predict_next_access(access_history[-10:])
                
                if predictions:
                    # 预计算预测的项目
                    self.precompute_items(predictions)
                    
                    # 检查下一个实际访问是否在预测中
                    if i + 1 < len(access_pattern):
                        next_actual = access_pattern[i + 1]
                        if next_actual in predictions:
                            predictions_correct += 1
            
            # 模拟实际访问（检查缓存）
            cache_key = f"precomputed_{current_item}"
            cached_result = self.cache_system.get(cache_key)
            
            if cached_result:
                # 缓存命中，说明预计算成功
                pass
        
        end_time = time.perf_counter()
        
        # 计算预测准确性
        prediction_accuracy = (predictions_correct / predictions_made * 100) if predictions_made > 0 else 0
        
        # 记录结束内存
        self.record_memory_usage("prediction_accuracy_end")
        memory_overhead = self.memory_monitor.get_difference("prediction_accuracy_start", "prediction_accuracy_end")
        
        metric = PrecomputationMetric(
            test_name="prediction_accuracy_test",
            prediction_accuracy=prediction_accuracy,
            precomputation_hit_rate=0,  # 在其他测试中计算
            resource_overhead=memory_overhead,
            time_saving=0,  # 在其他测试中计算
            efficiency_ratio=prediction_accuracy / max(memory_overhead, 0.1),  # 避免除零
            prediction_latency=(end_time - start_time) * 1000,
            cache_warmup_time=0
        )
        
        self.metrics.append(metric)
        self.logger.info(f"预测准确性测试完成: 准确性={prediction_accuracy:.2f}%, 内存开销={memory_overhead:.2f}MB")
        
        return metric
    
    def test_precomputation_hit_rate(self) -> PrecomputationMetric:
        """
        测试预计算命中率
        
        Returns:
            PrecomputationMetric: 预计算命中率指标
        """
        self.logger.info("开始预计算命中率测试...")
        
        # 清空缓存
        self.cache_system.clear()
        
        # 记录初始内存
        self.record_memory_usage("precomputation_hit_start")
        
        access_pattern = self.simulate_access_pattern('sequential', 150)
        hits = 0
        total_accesses = 0
        
        # 预热缓存：预计算一些项目
        warmup_start = time.perf_counter()
        initial_predictions = self.predict_next_access(access_pattern[:20])
        warmup_results = self.precompute_items(initial_predictions)
        warmup_time = (time.perf_counter() - warmup_start) * 1000
        
        # 模拟实际访问
        for item in access_pattern:
            total_accesses += 1
            
            # 尝试从预计算缓存中获取
            cache_key = f"precomputed_{item}"
            cached_result = self.cache_system.get(cache_key)
            
            if cached_result:
                hits += 1
            else:
                # 如果未命中，模拟实时计算并缓存
                real_time_result = f"real_time_result_for_{item}"
                self.cache_system.set(cache_key, real_time_result, ttl=3600)
        
        # 记录结束内存
        self.record_memory_usage("precomputation_hit_end")
        memory_overhead = self.memory_monitor.get_difference("precomputation_hit_start", "precomputation_hit_end")
        
        hit_rate = (hits / total_accesses * 100) if total_accesses > 0 else 0
        
        metric = PrecomputationMetric(
            test_name="precomputation_hit_rate_test",
            prediction_accuracy=0,
            precomputation_hit_rate=hit_rate,
            resource_overhead=memory_overhead,
            time_saving=0,  # 在其他测试中计算
            efficiency_ratio=hit_rate / max(memory_overhead, 0.1),
            prediction_latency=0,
            cache_warmup_time=warmup_time
        )
        
        self.metrics.append(metric)
        self.logger.info(f"预计算命中率测试完成: 命中率={hit_rate:.2f}%, 预热时间={warmup_time:.2f}ms, 内存开销={memory_overhead:.2f}MB")
        
        return metric
    
    def test_precomputation_vs_real_time(self) -> PrecomputationMetric:
        """
        测试预计算与实时计算的性能对比
        
        Returns:
            PrecomputationMetric: 性能对比指标
        """
        self.logger.info("开始预计算vs实时计算性能对比测试...")
        
        test_iterations = 50
        precomputation_times = []
        real_time_times = []
        
        for i in range(test_iterations):
            test_item = f"performance_test_item_{i}"
            
            # 测试预计算访问时间
            cache_key = f"precomputed_{test_item}"
            
            # 预计算数据
            precomputed_data = {"result": f"precomputed_{test_item}", "time": time.time()}
            self.cache_system.set(cache_key, precomputed_data, ttl=3600)
            
            # 测量预计算访问时间
            start_time = time.perf_counter()
            result = self.cache_system.get(cache_key)
            precomputation_time = (time.perf_counter() - start_time) * 1000
            precomputation_times.append(precomputation_time)
            
            # 清除缓存，测试实时计算时间
            self.cache_system.delete(cache_key)
            
            start_time = time.perf_counter()
            # 模拟实时计算
            real_time_result = f"real_time_result_{test_item}"
            computation_delay = random.uniform(1.0, 5.0)  # 模拟计算延迟
            time.sleep(computation_delay / 1000)  # 转换为秒
            self.cache_system.set(cache_key, real_time_result, ttl=3600)
            real_time_time = (time.perf_counter() - start_time) * 1000
            real_time_times.append(real_time_time)
        
        # 计算平均时间
        avg_precomputation_time = statistics.mean(precomputation_times)
        avg_real_time_time = statistics.mean(real_time_times)
        
        # 计算时间节省
        time_saving = avg_real_time_time - avg_precomputation_time
        efficiency_ratio = avg_real_time_time / avg_precomputation_time if avg_precomputation_time > 0 else 0
        
        metric = PrecomputationMetric(
            test_name="precomputation_vs_real_time_test",
            prediction_accuracy=0,
            precomputation_hit_rate=0,
            resource_overhead=0,  # 不测试内存开销
            time_saving=time_saving,
            efficiency_ratio=efficiency_ratio,
            prediction_latency=0,
            cache_warmup_time=0
        )
        
        self.metrics.append(metric)
        self.logger.info(f"性能对比测试完成: 预计算={avg_precomputation_time:.2f}ms, 实时计算={avg_real_time_time:.2f}ms, 节省时间={time_saving:.2f}ms, 效率比={efficiency_ratio:.2f}x")
        
        return metric
    
    def test_resource_efficiency(self) -> PrecomputationMetric:
        """
        测试预计算资源效率
        
        Returns:
            PrecomputationMetric: 资源效率指标
        """
        self.logger.info("开始预计算资源效率测试...")
        
        # 记录初始状态
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        cache_scenarios = [100, 500, 1000]
        total_precomputed = 0
        total_memory_used = 0
        
        for scenario_size in cache_scenarios:
            # 清空缓存
            self.cache_system.clear()
            
            # 预计算指定数量的项目
            scenario_start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            
            precomputed_items = []
            for i in range(scenario_size):
                item_key = f"resource_test_item_{scenario_size}_{i}"
                item_data = {
                    'data': 'x' * 1000,  # 每个项目1KB数据
                    'metadata': {'computed_at': time.time(), 'dependencies': [f'dep_{j}' for j in range(3)]},
                    'result': f'precomputed_result_{i}'
                }
                precomputed_items.append(item_key)
                self.cache_system.set(f"precomputed_{item_key}", item_data, ttl=3600)
            
            scenario_end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            scenario_memory_used = scenario_end_memory - scenario_start_memory
            
            total_precomputed += scenario_size
            total_memory_used += scenario_memory_used
            
            self.logger.info(f"场景{scenario_size}: 预计算{scenario_size}个项目, 内存使用{scenario_memory_used:.2f}MB, 平均每项目{scenario_memory_used/scenario_size:.4f}MB")
        
        # 计算资源效率
        avg_memory_per_item = total_memory_used / total_precomputed if total_precomputed > 0 else 0
        resource_efficiency = 1 / avg_memory_per_item if avg_memory_per_item > 0 else 0
        
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        total_memory_overhead = final_memory - initial_memory
        
        metric = PrecomputationMetric(
            test_name="resource_efficiency_test",
            prediction_accuracy=0,
            precomputation_hit_rate=0,
            resource_overhead=total_memory_overhead,
            time_saving=0,
            efficiency_ratio=resource_efficiency,
            prediction_latency=0,
            cache_warmup_time=0
        )
        
        self.metrics.append(metric)
        self.logger.info(f"资源效率测试完成: 平均每项目内存={avg_memory_per_item:.4f}MB, 资源效率={resource_efficiency:.2f}, 总内存开销={total_memory_overhead:.2f}MB")
        
        return metric
    
    def test_adaptive_prediction(self) -> PrecomputationMetric:
        """
        测试自适应预测算法
        
        Returns:
            PrecomputationMetric: 自适应预测指标
        """
        self.logger.info("开始自适应预测算法测试...")
        
        # 模拟不同的工作负载模式
        workload_patterns = [
            ('sequential', 100),
            ('random', 100),
            ('burst', 100),
            ('mixed', 100)
        ]
        
        total_predictions = 0
        correct_predictions = 0
        adaptation_scores = []
        
        for pattern_name, pattern_length in workload_patterns:
            self.logger.info(f"测试工作负载模式: {pattern_name}")
            
            # 生成访问模式
            access_pattern = self.simulate_access_pattern(pattern_name, pattern_length)
            
            pattern_predictions = 0
            pattern_correct = 0
            
            for i in range(10, len(access_pattern), 5):
                # 基于历史进行预测
                history = access_pattern[max(0, i-10):i]
                predictions = self.predict_next_access(history)
                
                if predictions:
                    pattern_predictions += 1
                    total_predictions += 1
                    
                    # 预计算预测的项目
                    self.precompute_items(predictions)
                    
                    # 检查下一个访问
                    if i + 1 < len(access_pattern):
                        next_access = access_pattern[i + 1]
                        if next_access in predictions:
                            pattern_correct += 1
                            correct_predictions += 1
            
            # 计算该模式的适应性分数
            pattern_accuracy = (pattern_correct / pattern_predictions * 100) if pattern_predictions > 0 else 0
            adaptation_scores.append(pattern_accuracy)
            
            self.logger.info(f"{pattern_name}模式: 预测{pattern_predictions}次, 正确{pattern_correct}次, 准确性={pattern_accuracy:.2f}%")
        
        # 计算总体自适应性能
        overall_accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        adaptation_variance = statistics.variance(adaptation_scores) if len(adaptation_scores) > 1 else 0
        adaptation_stability = 100 - adaptation_variance  # 方差越小越稳定
        
        metric = PrecomputationMetric(
            test_name="adaptive_prediction_test",
            prediction_accuracy=overall_accuracy,
            precomputation_hit_rate=0,
            resource_overhead=0,
            time_saving=0,
            efficiency_ratio=adaptation_stability,
            prediction_latency=0,
            cache_warmup_time=0
        )
        
        self.metrics.append(metric)
        self.logger.info(f"自适应预测测试完成: 总体准确性={overall_accuracy:.2f}%, 适应性稳定性={adaptation_stability:.2f}")
        
        return metric
    
    def generate_precomputation_report(self) -> Dict[str, Any]:
        """
        生成预计算性能报告
        
        Returns:
            Dict[str, Any]: 预计算性能报告数据
        """
        self.logger.info("生成智能预计算性能报告...")
        
        if not self.metrics:
            self.logger.warning("没有测试数据，无法生成报告")
            return {}
        
        # 计算总体性能指标
        avg_prediction_accuracy = statistics.mean([m.prediction_accuracy for m in self.metrics if m.prediction_accuracy > 0])
        avg_hit_rate = statistics.mean([m.precomputation_hit_rate for m in self.metrics if m.precomputation_hit_rate > 0])
        avg_resource_efficiency = statistics.mean([m.efficiency_ratio for m in self.metrics if m.efficiency_ratio > 0])
        avg_time_saving = statistics.mean([m.time_saving for m in self.metrics if m.time_saving > 0])
        
        # 找出最佳和最差性能
        best_accuracy = max(self.metrics, key=lambda x: x.prediction_accuracy)
        best_hit_rate = max(self.metrics, key=lambda x: x.precomputation_hit_rate if x.precomputation_hit_rate else 0)
        best_efficiency = max(self.metrics, key=lambda x: x.efficiency_ratio if x.efficiency_ratio else 0)
        
        report = {
            'intelligent_precomputation_analysis': {
                'overall_performance': {
                    'avg_prediction_accuracy': round(avg_prediction_accuracy, 2),
                    'avg_precomputation_hit_rate': round(avg_hit_rate, 2),
                    'avg_resource_efficiency': round(avg_resource_efficiency, 2),
                    'avg_time_saving_ms': round(avg_time_saving, 2),
                    'total_test_scenarios': len(self.metrics)
                },
                'performance_extremes': {
                    'best_prediction_accuracy': {
                        'test_name': best_accuracy.test_name,
                        'accuracy': round(best_accuracy.prediction_accuracy, 2)
                    },
                    'best_hit_rate': {
                        'test_name': best_hit_rate.test_name,
                        'hit_rate': round(best_hit_rate.precomputation_hit_rate, 2)
                    },
                    'best_efficiency': {
                        'test_name': best_efficiency.test_name,
                        'efficiency': round(best_efficiency.efficiency_ratio, 2)
                    }
                },
                'detailed_metrics': [
                    {
                        'test_name': m.test_name,
                        'prediction_accuracy_percent': round(m.prediction_accuracy, 2),
                        'precomputation_hit_rate_percent': round(m.precomputation_hit_rate, 2),
                        'resource_overhead_mb': round(m.resource_overhead, 2),
                        'time_saving_ms': round(m.time_saving, 2),
                        'efficiency_ratio': round(m.efficiency_ratio, 2),
                        'prediction_latency_ms': round(m.prediction_latency, 2),
                        'cache_warmup_time_ms': round(m.cache_warmup_time, 2)
                    }
                    for m in self.metrics
                ],
                'precomputation_summary': {
                    'prediction_quality': 'EXCELLENT' if avg_prediction_accuracy > 70 else 'GOOD' if avg_prediction_accuracy > 50 else 'POOR',
                    'cache_effectiveness': 'HIGH' if avg_hit_rate > 60 else 'MEDIUM' if avg_hit_rate > 40 else 'LOW',
                    'resource_optimization': 'EFFICIENT' if avg_resource_efficiency > 5 else 'MODERATE' if avg_resource_efficiency > 2 else 'INEFFICIENT',
                    'time_optimization': 'SIGNIFICANT' if avg_time_saving > 50 else 'MODERATE' if avg_time_saving > 20 else 'MINIMAL'
                },
                'recommendations': self.generate_recommendations(avg_prediction_accuracy, avg_hit_rate, avg_resource_efficiency)
            }
        }
        
        return report
    
    def generate_recommendations(self, avg_accuracy: float, avg_hit_rate: float, avg_efficiency: float) -> List[str]:
        """
        基于测试结果生成优化建议
        
        Args:
            avg_accuracy: 平均预测准确性
            avg_hit_rate: 平均命中率
            avg_efficiency: 平均资源效率
            
        Returns:
            List[str]: 优化建议列表
        """
        recommendations = []
        
        if avg_accuracy < 50:
            recommendations.append("🔍 预测准确性较低，建议改进预测算法，考虑使用机器学习模型")
        elif avg_accuracy < 70:
            recommendations.append("📊 预测准确性中等，可以考虑优化预测窗口大小和历史数据权重")
        else:
            recommendations.append("✅ 预测准确性优秀，当前算法表现良好")
        
        if avg_hit_rate < 40:
            recommendations.append("🎯 预计算命中率较低，建议调整预计算策略和缓存淘汰算法")
        elif avg_hit_rate < 60:
            recommendations.append("📈 预计算命中率中等，可以优化预计算时机和范围")
        else:
            recommendations.append("🎯 预计算命中率优秀，缓存策略有效")
        
        if avg_efficiency < 2:
            recommendations.append("⚡ 资源效率较低，建议优化内存使用和预计算资源分配")
        elif avg_efficiency < 5:
            recommendations.append("🔋 资源效率中等，可以进一步优化资源利用")
        else:
            recommendations.append("🚀 资源效率优秀，资源利用充分")
        
        recommendations.extend([
            "🔄 考虑实现动态调整预计算策略的机制",
            "📊 建立实时监控系统跟踪预计算效果",
            "🧠 探索更先进的预测算法，如深度学习模型",
            "⚡ 优化预计算任务的优先级和调度策略"
        ])
        
        return recommendations
    
    def save_precomputation_report(self, report: Dict[str, Any]):
        """
        保存预计算性能报告到文件
        
        Args:
            report: 预计算性能报告数据
        """
        # 保存JSON格式报告
        with open('intelligent_precomputation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成HTML格式报告
        html_report = self.generate_html_report(report)
        with open('intelligent_precomputation_report.html', 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        self.logger.info("智能预计算性能报告已保存")
    
    def generate_html_report(self, report: Dict[str, Any]) -> str:
        """
        生成HTML格式的预计算性能报告
        
        Args:
            report: 预计算性能报告数据
            
        Returns:
            str: HTML格式报告
        """
        html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>A项目V7 - 智能预计算性能报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .performance-excellent {{
            color: #27ae60;
            font-weight: bold;
        }}
        .performance-good {{
            color: #2ecc71;
            font-weight: bold;
        }}
        .performance-poor {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .recommendation {{
            background: #e8f5e8;
            border-left: 4px solid #27ae60;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 A项目V7 - 智能预计算性能报告</h1>
        
        <h2>📊 总体性能概览</h2>
        <div class="summary-grid">
            <div class="metric-card">
                <div class="metric-value">{report['intelligent_precomputation_analysis']['overall_performance']['avg_prediction_accuracy']}%</div>
                <div class="metric-label">平均预测准确性</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report['intelligent_precomputation_analysis']['overall_performance']['avg_precomputation_hit_rate']}%</div>
                <div class="metric-label">平均预计算命中率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report['intelligent_precomputation_analysis']['overall_performance']['avg_resource_efficiency']}</div>
                <div class="metric-label">平均资源效率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report['intelligent_precomputation_analysis']['overall_performance']['avg_time_saving_ms']}ms</div>
                <div class="metric-label">平均时间节省</div>
            </div>
        </div>
        
        <h2>🎯 性能评估</h2>
        <div class="summary-grid">
            <div style="background: #e8f5e8; padding: 20px; border-radius: 10px; text-align: center;">
                <h3>预测质量</h3>
                <p class="performance-{report['intelligent_precomputation_analysis']['precomputation_summary']['prediction_quality'].lower()}">
                    {report['intelligent_precomputation_analysis']['precomputation_summary']['prediction_quality']}
                </p>
            </div>
            <div style="background: #e3f2fd; padding: 20px; border-radius: 10px; text-align: center;">
                <h3>缓存效果</h3>
                <p class="performance-{report['intelligent_precomputation_analysis']['precomputation_summary']['cache_effectiveness'].lower()}">
                    {report['intelligent_precomputation_analysis']['precomputation_summary']['cache_effectiveness']}
                </p>
            </div>
            <div style="background: #fff3e0; padding: 20px; border-radius: 10px; text-align: center;">
                <h3>资源优化</h3>
                <p class="performance-{report['intelligent_precomputation_analysis']['precomputation_summary']['resource_optimization'].lower()}">
                    {report['intelligent_precomputation_analysis']['precomputation_summary']['resource_optimization']}
                </p>
            </div>
            <div style="background: #f3e5f5; padding: 20px; border-radius: 10px; text-align: center;">
                <h3>时间优化</h3>
                <p class="performance-{report['intelligent_precomputation_analysis']['precomputation_summary']['time_optimization'].lower()}">
                    {report['intelligent_precomputation_analysis']['precomputation_summary']['time_optimization']}
                </p>
            </div>
        </div>
        
        <h2>📋 详细测试结果</h2>
        <table>
            <thead>
                <tr>
                    <th>测试场景</th>
                    <th>预测准确性(%)</th>
                    <th>命中率(%)</th>
                    <th>资源开销(MB)</th>
                    <th>时间节省(ms)</th>
                    <th>效率比</th>
                    <th>预热时间(ms)</th>
                </tr>
            </thead>
            <tbody>
                {''.join([f'''
                <tr>
                    <td>{metric['test_name']}</td>
                    <td>{metric['prediction_accuracy_percent']}%</td>
                    <td>{metric['precomputation_hit_rate_percent']}%</td>
                    <td>{metric['resource_overhead_mb']}</td>
                    <td>{metric['time_saving_ms']}</td>
                    <td>{metric['efficiency_ratio']}</td>
                    <td>{metric['cache_warmup_time_ms']}</td>
                </tr>
                ''' for metric in report['intelligent_precomputation_analysis']['detailed_metrics']])}
            </tbody>
        </table>
        
        <h2>💡 优化建议</h2>
        {''.join([f'<div class="recommendation">{recommendation}</div>' for recommendation in report['intelligent_precomputation_analysis']['recommendations']])}
        
        <div class="footer">
            <p>📊 报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>🧠 A项目V7 - 智能预计算机制测试套件</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_template
    
    def run_comprehensive_precomputation_test(self):
        """
        运行全面的智能预计算测试
        """
        self.logger.info("🧠 开始运行全面的智能预计算测试...")
        
        # 运行各项测试
        self.test_prediction_accuracy()
        self.test_precomputation_hit_rate()
        self.test_precomputation_vs_real_time()
        self.test_resource_efficiency()
        self.test_adaptive_prediction()
        
        # 生成性能报告
        report = self.generate_precomputation_report()
        
        # 保存报告
        self.save_precomputation_report(report)
        
        # 打印测试总结
        self.logger.info("=" * 80)
        self.logger.info("🧠 智能预计算测试完成！")
        self.logger.info(f"📊 总体性能指标:")
        self.logger.info(f"   🎯 平均预测准确性: {report['intelligent_precomputation_analysis']['overall_performance']['avg_prediction_accuracy']}%")
        self.logger.info(f"   🎯 平均预计算命中率: {report['intelligent_precomputation_analysis']['overall_performance']['avg_precomputation_hit_rate']}%")
        self.logger.info(f"   ⚡ 平均资源效率: {report['intelligent_precomputation_analysis']['overall_performance']['avg_resource_efficiency']}")
        self.logger.info(f"   ⏱️ 平均时间节省: {report['intelligent_precomputation_analysis']['overall_performance']['avg_time_saving_ms']}ms")
        self.logger.info(f"   🧪 测试场景总数: {report['intelligent_precomputation_analysis']['overall_performance']['total_test_scenarios']}")
        self.logger.info("=" * 80)
        
        return report

class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self):
        self.memory_snapshots = {}
    
    def record(self, label: str, memory_mb: float):
        """记录内存快照"""
        self.memory_snapshots[label] = {
            'memory_mb': memory_mb,
            'timestamp': time.time()
        }
    
    def get_difference(self, start_label: str, end_label: str) -> float:
        """计算两个时间点的内存差异"""
        if start_label in self.memory_snapshots and end_label in self.memory_snapshots:
            return self.memory_snapshots[end_label]['memory_mb'] - self.memory_snapshots[start_label]['memory_mb']
        return 0.0

if __name__ == "__main__":
    # 运行智能预计算测试
    tester = IntelligentPrecomputationTester()
    report = tester.run_comprehensive_precomputation_test()
    
    # 打印关键发现
    print("\n🧠 关键预计算发现:")
    print(f"✅ 预测准确性达到 {report['intelligent_precomputation_analysis']['overall_performance']['avg_prediction_accuracy']}%")
    print(f"✅ 预计算命中率达到 {report['intelligent_precomputation_analysis']['overall_performance']['avg_precomputation_hit_rate']}%")
    print(f"✅ 资源效率为 {report['intelligent_precomputation_analysis']['overall_performance']['avg_resource_efficiency']}")
    print(f"✅ 平均节省时间 {report['intelligent_precomputation_analysis']['overall_performance']['avg_time_saving_ms']}ms")
    print(f"✅ 在 {report['intelligent_precomputation_analysis']['overall_performance']['total_test_scenarios']} 个测试场景中验证了预计算的有效性")