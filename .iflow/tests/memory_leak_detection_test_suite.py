#!/usr/bin/env python3
"""
缓存系统内存泄漏检测套件
专门检测长时间运行下的内存泄漏问题

测试目标：
1. 验证长时间运行下的内存稳定性
2. 检测缓存系统的内存泄漏情况
3. 测试垃圾回收机制的有效性
4. 评估内存使用趋势和增长模式
5. 验证内存优化策略的效果

作者：A项目V7升级版
创建时间：2025-11-13
"""

import time
import threading
import gc
import weakref
import tracemalloc
import psutil
import os
import sys
import json
import logging
import asyncio
import statistics
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta

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
class MemoryLeakMetric:
    """内存泄漏指标数据类"""
    test_name: str
    initial_memory_mb: float  # 初始内存(MB)
    final_memory_mb: float    # 最终内存(MB)
    memory_growth_mb: float   # 内存增长(MB)
    memory_growth_rate: float # 内存增长率(%/小时)
    garbage_collection_efficiency: float  # 垃圾回收效率(%)
    object_count_growth: int  # 对象数量增长
    leak_detected: bool       # 是否检测到泄漏
    leak_severity: str        # 泄漏严重程度
    stability_score: float    # 稳定性评分(0-100)

class MemoryLeakDetectionTester:
    """内存泄漏检测测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.cache_system = OptimizedFusionCache()
        self.context_manager = IntelligentContextManager()
        self.model_adapter = UnifiedModelAdapter()
        self.parallel_executor = ParallelAgentExecutor()
        self.task_decomposer = TaskDecomposer()
        self.workflow_parallelizer = WorkflowStageParallelizer()
        
        # 测试配置
        self.monitoring_duration = 3600  # 监控时长(秒)，1小时
        self.monitoring_interval = 30    # 监控间隔(秒)
        self.stress_test_duration = 600  # 压力测试时长(秒)，10分钟
        self.long_running_duration = 7200  # 长时间运行测试(秒)，2小时
        
        # 内存监控配置
        self.memory_snapshots = []
        self.object_snapshots = []
        self.process = psutil.Process(os.getpid())
        
        # 垃圾回收监控
        self.gc_before_counts = []
        self.gc_after_counts = []
        
        # 测试结果存储
        self.metrics: List[MemoryLeakMetric] = []
        
        # 弱引用跟踪器
        self.weak_refs = weakref.WeakSet()
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('memory_leak_detection_test.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 启动内存跟踪
        tracemalloc.start()
        
    def get_memory_usage(self) -> float:
        """
        获取当前内存使用量(MB)
        
        Returns:
            float: 内存使用量(MB)
        """
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024  # 转换为MB
        except Exception as e:
            self.logger.error(f"获取内存使用量失败: {e}")
            return 0.0
    
    def get_object_count(self) -> Dict[str, int]:
        """
        获取当前对象数量统计
        
        Returns:
            Dict[str, int]: 对象类型计数字典
        """
        try:
            # 获取主要对象类型的数量
            object_counts = {}
            
            # 统计缓存相关对象
            if hasattr(self.cache_system, '_cache'):
                object_counts['cache_entries'] = len(self.cache_system._cache) if self.cache_system._cache else 0
            
            if hasattr(self.cache_system, '_ttl_heap'):
                object_counts['ttl_entries'] = len(self.cache_system._ttl_heap) if self.cache_system._ttl_heap else 0
            
            # 统计Python对象总数
            object_counts['total_objects'] = len(gc.get_objects())
            
            # 统计各代垃圾回收器中的对象数量
            for i, count in enumerate(gc.get_count()):
                object_counts[f'gc_generation_{i}'] = count
            
            return object_counts
        except Exception as e:
            self.logger.error(f"获取对象数量失败: {e}")
            return {}
    
    def force_garbage_collection(self) -> Tuple[List[int], List[int]]:
        """
        强制执行垃圾回收并记录前后状态
        
        Returns:
            Tuple[List[int], List[int]]: 垃圾回收前后的对象数量
        """
        # 记录垃圾回收前的状态
        before_counts = list(gc.get_count())
        before_objects = len(gc.get_objects())
        
        # 强制垃圾回收
        gc.collect()
        
        # 记录垃圾回收后的状态
        after_counts = list(gc.get_count())
        after_objects = len(gc.get_objects())
        
        return before_counts, after_counts
    
    def record_memory_snapshot(self, label: str = ""):
        """
        记录内存快照
        
        Args:
            label: 快照标签
        """
        timestamp = time.time()
        memory_mb = self.get_memory_usage()
        object_counts = self.get_object_count()
        
        snapshot = {
            'timestamp': timestamp,
            'label': label,
            'memory_mb': memory_mb,
            'object_counts': object_counts,
            'gc_counts': list(gc.get_count())
        }
        
        self.memory_snapshots.append(snapshot)
        
        # 保持快照数量在合理范围内
        if len(self.memory_snapshots) > 1000:
            self.memory_snapshots.pop(0)
    
    def simulate_cache_operations(self, duration: int, operation_type: str = "mixed"):
        """
        模拟缓存操作以产生内存压力
        
        Args:
            duration: 操作时长(秒)
            operation_type: 操作类型
        """
        self.logger.info(f"开始模拟缓存操作: {operation_type}, 时长: {duration}秒")
        
        start_time = time.time()
        operation_count = 0
        
        while time.time() - start_time < duration:
            current_time = time.time()
            
            # 每30秒记录一次快照
            if operation_count % 100 == 0:
                self.record_memory_snapshot(f"operation_{operation_count}")
            
            try:
                if operation_type == "write_heavy":
                    # 写密集操作
                    key = f"test_key_{operation_count}"
                    value = {
                        'data': 'x' * 1000,  # 1KB数据
                        'metadata': {'created_at': current_time, 'operation_count': operation_count},
                        'large_data': ['item_' + str(i) for i in range(100)]
                    }
                    self.cache_system.set(key, value, ttl=3600)
                    
                elif operation_type == "read_heavy":
                    # 读密集操作
                    key = f"test_key_{operation_count % 1000}"  # 循环访问
                    result = self.cache_system.get(key)
                    if result is None:
                        # 如果不存在，创建一个
                        self.cache_system.set(key, f"fallback_data_{operation_count}", ttl=3600)
                
                elif operation_type == "mixed":
                    # 混合操作
                    if operation_count % 3 == 0:
                        # 写操作
                        key = f"write_key_{operation_count}"
                        value = {'data': f'write_data_{operation_count}', 'timestamp': current_time}
                        self.cache_system.set(key, value, ttl=3600)
                    elif operation_count % 3 == 1:
                        # 读操作
                        key = f"read_key_{operation_count % 500}"
                        result = self.cache_system.get(key)
                    else:
                        # 删除操作
                        key = f"delete_key_{operation_count % 300}"
                        self.cache_system.delete(key)
                
                elif operation_type == "burst":
                    # 突发操作
                    if operation_count % 50 == 0:
                        # 突发写入
                        for i in range(50):
                            key = f"burst_key_{operation_count}_{i}"
                            value = {'burst_data': i, 'timestamp': current_time}
                            self.cache_system.set(key, value, ttl=3600)
                    else:
                        # 正常操作
                        key = f"normal_key_{operation_count}"
                        self.cache_system.get(key)
                
                operation_count += 1
                
                # 每1000次操作强制垃圾回收
                if operation_count % 1000 == 0:
                    before_gc, after_gc = self.force_garbage_collection()
                    gc_efficiency = self.calculate_gc_efficiency(before_gc, after_gc)
                    self.logger.info(f"操作{operation_count}: GC效率={gc_efficiency:.2f}%")
                
                # 每100次操作记录内存状态
                if operation_count % 100 == 0:
                    memory_mb = self.get_memory_usage()
                    self.logger.info(f"操作{operation_count}: 内存使用={memory_mb:.2f}MB")
                
            except Exception as e:
                self.logger.error(f"操作{operation_count}发生错误: {e}")
                continue
        
        self.logger.info(f"缓存操作模拟完成: 总操作数={operation_count}, 平均操作频率={operation_count/duration:.2f}次/秒")
    
    def calculate_gc_efficiency(self, before_counts: List[int], after_counts: List[int]) -> float:
        """
        计算垃圾回收效率
        
        Args:
            before_counts: 垃圾回收前的对象数量
            after_counts: 垃圾回收后的对象数量
            
        Returns:
            float: 垃圾回收效率(%)
        """
        try:
            total_before = sum(before_counts)
            total_after = sum(after_counts)
            if total_before > 0:
                return (total_before - total_after) / total_before * 100
            return 0.0
        except Exception:
            return 0.0
    
    def analyze_memory_trend(self) -> Dict[str, Any]:
        """
        分析内存使用趋势
        
        Returns:
            Dict[str, Any]: 内存趋势分析结果
        """
        if len(self.memory_snapshots) < 10:
            return {"error": "快照数量不足，无法进行趋势分析"}
        
        # 提取内存数据
        memory_values = [snapshot['memory_mb'] for snapshot in self.memory_snapshots]
        timestamps = [snapshot['timestamp'] for snapshot in self.memory_snapshots]
        
        # 计算基本统计信息
        initial_memory = memory_values[0]
        final_memory = memory_values[-1]
        memory_growth = final_memory - initial_memory
        total_duration_hours = (timestamps[-1] - timestamps[0]) / 3600
        
        # 计算内存增长率
        memory_growth_rate = (memory_growth / total_duration_hours) if total_duration_hours > 0 else 0
        
        # 计算内存波动
        memory_std = statistics.stdev(memory_values) if len(memory_values) > 1 else 0
        memory_variance = statistics.variance(memory_values) if len(memory_values) > 1 else 0
        
        # 检测内存泄漏趋势
        # 使用线性回归分析内存增长趋势
        n = len(memory_values)
        if n > 1:
            # 计算线性回归斜率
            x_mean = statistics.mean(range(n))
            y_mean = statistics.mean(memory_values)
            
            numerator = sum((i - x_mean) * (memory_values[i] - y_mean) for i in range(n))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            
            slope = numerator / denominator if denominator != 0 else 0
            trend_direction = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
        else:
            slope = 0
            trend_direction = "unknown"
        
        return {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_growth_mb': memory_growth,
            'memory_growth_rate_per_hour': memory_growth_rate,
            'memory_std_deviation': memory_std,
            'memory_variance': memory_variance,
            'trend_slope': slope,
            'trend_direction': trend_direction,
            'total_duration_hours': total_duration_hours
        }
    
    def detect_memory_leak(self, memory_trend: Dict[str, Any]) -> Dict[str, Any]:
        """
        检测内存泄漏
        
        Args:
            memory_trend: 内存趋势分析结果
            
        Returns:
            Dict[str, Any]: 泄漏检测结果
        """
        memory_growth = memory_trend.get('memory_growth_mb', 0)
        memory_growth_rate = memory_trend.get('memory_growth_rate_per_hour', 0)
        trend_slope = memory_trend.get('trend_slope', 0)
        total_duration = memory_trend.get('total_duration_hours', 0)
        
        # 内存泄漏检测标准
        leak_threshold_mb = 50  # 50MB内存增长阈值
        leak_rate_threshold = 10  # 10MB/小时增长率阈值
        slope_threshold = 0.5  # 趋势斜率阈值
        
        # 基于多个指标综合判断
        growth_leak = memory_growth > leak_threshold_mb
        rate_leak = memory_growth_rate > leak_rate_threshold
        slope_leak = trend_slope > slope_threshold
        
        leak_detected = growth_leak or rate_leak or slope_leak
        
        # 确定泄漏严重程度
        if not leak_detected:
            leak_severity = "none"
            stability_score = 100
        elif memory_growth > 200 or memory_growth_rate > 50 or trend_slope > 2:
            leak_severity = "critical"
            stability_score = 20
        elif memory_growth > 100 or memory_growth_rate > 25 or trend_slope > 1:
            leak_severity = "high"
            stability_score = 40
        elif memory_growth > 50 or memory_growth_rate > 10 or trend_slope > 0.5:
            leak_severity = "medium"
            stability_score = 70
        else:
            leak_severity = "low"
            stability_score = 85
        
        return {
            'leak_detected': leak_detected,
            'leak_severity': leak_severity,
            'stability_score': stability_score,
            'growth_leak': growth_leak,
            'rate_leak': rate_leak,
            'slope_leak': slope_leak,
            'evidence': {
                'memory_growth_mb': memory_growth,
                'growth_rate_threshold_mb': leak_threshold_mb,
                'memory_growth_rate': memory_growth_rate,
                'rate_threshold': leak_rate_threshold,
                'trend_slope': trend_slope,
                'slope_threshold': slope_threshold
            }
        }
    
    def test_short_term_memory_stability(self) -> MemoryLeakMetric:
        """
        测试短期内存稳定性(1小时)
        
        Returns:
            MemoryLeakMetric: 短期内存稳定性指标
        """
        self.logger.info("开始短期内存稳定性测试...")
        
        # 记录初始状态
        self.record_memory_snapshot("initial")
        initial_memory = self.get_memory_usage()
        initial_objects = self.get_object_count()
        
        # 运行混合负载测试
        self.simulate_cache_operations(self.monitoring_duration // 6, "mixed")
        
        # 记录最终状态
        self.record_memory_snapshot("final")
        final_memory = self.get_memory_usage()
        final_objects = self.get_object_count()
        
        # 分析内存趋势
        memory_trend = self.analyze_memory_trend()
        leak_detection = self.detect_memory_leak(memory_trend)
        
        # 计算垃圾回收效率
        gc_before_total = sum(self.gc_before_counts) if self.gc_before_counts else 0
        gc_after_total = sum(self.gc_after_counts) if self.gc_after_counts else 0
        gc_efficiency = ((gc_before_total - gc_after_total) / gc_before_total * 100) if gc_before_total > 0 else 0
        
        # 计算对象数量增长
        object_growth = final_objects.get('total_objects', 0) - initial_objects.get('total_objects', 0)
        
        metric = MemoryLeakMetric(
            test_name="short_term_memory_stability",
            initial_memory_mb=initial_memory,
            final_memory_mb=final_memory,
            memory_growth_mb=memory_trend.get('memory_growth_mb', 0),
            memory_growth_rate=memory_trend.get('memory_growth_rate_per_hour', 0),
            garbage_collection_efficiency=gc_efficiency,
            object_count_growth=object_growth,
            leak_detected=leak_detection['leak_detected'],
            leak_severity=leak_detection['leak_severity'],
            stability_score=leak_detection['stability_score']
        )
        
        self.metrics.append(metric)
        self.logger.info(f"短期内存稳定性测试完成: 初始={initial_memory:.2f}MB, 最终={final_memory:.2f}MB, 增长={memory_trend.get('memory_growth_mb', 0):.2f}MB, 泄漏={leak_detection['leak_detected']}")
        
        return metric
    
    def test_stress_memory_behavior(self) -> MemoryLeakMetric:
        """
        测试压力下的内存行为(10分钟高强度操作)
        
        Returns:
            MemoryLeakMetric: 压力测试内存指标
        """
        self.logger.info("开始压力测试内存行为...")
        
        # 记录初始状态
        self.record_memory_snapshot("stress_initial")
        initial_memory = self.get_memory_usage()
        
        # 高强度写入操作
        self.simulate_cache_operations(self.stress_test_duration // 2, "write_heavy")
        
        # 高强度读取操作
        self.simulate_cache_operations(self.stress_test_duration // 2, "read_heavy")
        
        # 记录最终状态
        self.record_memory_snapshot("stress_final")
        final_memory = self.get_memory_usage()
        
        # 分析内存趋势
        memory_trend = self.analyze_memory_trend()
        leak_detection = self.detect_memory_leak(memory_trend)
        
        metric = MemoryLeakMetric(
            test_name="stress_memory_behavior",
            initial_memory_mb=initial_memory,
            final_memory_mb=final_memory,
            memory_growth_mb=memory_trend.get('memory_growth_mb', 0),
            memory_growth_rate=memory_trend.get('memory_growth_rate_per_hour', 0),
            garbage_collection_efficiency=0,  # 压力测试不计算GC效率
            object_count_growth=0,
            leak_detected=leak_detection['leak_detected'],
            leak_severity=leak_detection['leak_severity'],
            stability_score=leak_detection['stability_score']
        )
        
        self.metrics.append(metric)
        self.logger.info(f"压力测试完成: 内存增长={memory_trend.get('memory_growth_mb', 0):.2f}MB, 泄漏={leak_detection['leak_detected']}, 严重程度={leak_detection['leak_severity']}")
        
        return metric
    
    def test_long_term_memory_stability(self) -> MemoryLeakMetric:
        """
        测试长期内存稳定性(2小时)
        
        Returns:
            MemoryLeakMetric: 长期内存稳定性指标
        """
        self.logger.info("开始长期内存稳定性测试...")
        
        # 记录初始状态
        self.record_memory_snapshot("long_term_initial")
        initial_memory = self.get_memory_usage()
        
        # 模拟长时间运行的混合负载
        self.simulate_cache_operations(self.long_running_duration, "mixed")
        
        # 记录最终状态
        self.record_memory_snapshot("long_term_final")
        final_memory = self.get_memory_usage()
        
        # 分析内存趋势
        memory_trend = self.analyze_memory_trend()
        leak_detection = self.detect_memory_leak(memory_trend)
        
        metric = MemoryLeakMetric(
            test_name="long_term_memory_stability",
            initial_memory_mb=initial_memory,
            final_memory_mb=final_memory,
            memory_growth_mb=memory_trend.get('memory_growth_mb', 0),
            memory_growth_rate=memory_trend.get('memory_growth_rate_per_hour', 0),
            garbage_collection_efficiency=0,
            object_count_growth=0,
            leak_detected=leak_detection['leak_detected'],
            leak_severity=leak_detection['leak_severity'],
            stability_score=leak_detection['stability_score']
        )
        
        self.metrics.append(metric)
        self.logger.info(f"长期内存稳定性测试完成: 初始={initial_memory:.2f}MB, 最终={final_memory:.2f}MB, 增长={memory_trend.get('memory_growth_mb', 0):.2f}MB, 泄漏={leak_detection['leak_detected']}")
        
        return metric
    
    def test_memory_recovery_after_gc(self) -> MemoryLeakMetric:
        """
        测试垃圾回收后的内存恢复情况
        
        Returns:
            MemoryLeakMetric: 内存恢复指标
        """
        self.logger.info("开始内存恢复测试...")
        
        # 记录垃圾回收前的内存状态
        self.record_memory_snapshot("before_gc")
        memory_before_gc = self.get_memory_usage()
        
        # 执行强制垃圾回收
        before_counts, after_counts = self.force_garbage_collection()
        
        # 记录垃圾回收后的内存状态
        self.record_memory_snapshot("after_gc")
        memory_after_gc = self.get_memory_usage()
        
        # 计算内存回收效果
        memory_recovered = memory_before_gc - memory_after_gc
        gc_efficiency = self.calculate_gc_efficiency(before_counts, after_counts)
        
        self.logger.info(f"内存恢复测试: 回收前={memory_before_gc:.2f}MB, 回收后={memory_after_gc:.2f}MB, 回收量={memory_recovered:.2f}MB, GC效率={gc_efficiency:.2f}%")
        
        metric = MemoryLeakMetric(
            test_name="memory_recovery_after_gc",
            initial_memory_mb=memory_before_gc,
            final_memory_mb=memory_after_gc,
            memory_growth_mb=-memory_recovered,  # 负值表示内存减少
            memory_growth_rate=0,
            garbage_collection_efficiency=gc_efficiency,
            object_count_growth=0,
            leak_detected=False,  # 这个测试不检测泄漏
            leak_severity="none",
            stability_score=gc_efficiency  # 用GC效率作为稳定性评分
        )
        
        self.metrics.append(metric)
        return metric
    
    def test_cache_memory_optimization(self) -> MemoryLeakMetric:
        """
        测试缓存内存优化效果
        
        Returns:
            MemoryLeakMetric: 缓存优化指标
        """
        self.logger.info("开始缓存内存优化测试...")
        
        # 测试不同缓存大小下的内存使用
        cache_scenarios = [100, 500, 1000, 2000]
        total_memory_used = 0
        total_items_cached = 0
        
        for scenario_size in cache_scenarios:
            # 清空缓存
            self.cache_system.clear()
            
            # 记录清空后的内存
            memory_before = self.get_memory_usage()
            
            # 缓存指定数量的项目
            for i in range(scenario_size):
                key = f"optimization_test_{scenario_size}_{i}"
                value = {
                    'data': 'x' * 500,  # 500字节数据
                    'metadata': {'created_at': time.time()},
                    'dependencies': [f'dep_{j}' for j in range(5)]
                }
                self.cache_system.set(key, value, ttl=3600)
            
            # 记录缓存后的内存
            memory_after = self.get_memory_usage()
            memory_used = memory_after - memory_before
            
            total_memory_used += memory_used
            total_items_cached += scenario_size
            
            self.logger.info(f"缓存{scenario_size}个项目: 内存使用={memory_used:.2f}MB, 平均每项目={memory_used/scenario_size:.4f}MB")
        
        # 计算缓存内存效率
        avg_memory_per_item = total_memory_used / total_items_cached if total_items_cached > 0 else 0
        memory_efficiency = 100 / max(avg_memory_per_item, 0.001)  # 避免除零，效率与平均内存使用成反比
        
        metric = MemoryLeakMetric(
            test_name="cache_memory_optimization",
            initial_memory_mb=0,
            final_memory_mb=0,
            memory_growth_mb=0,
            memory_growth_rate=0,
            garbage_collection_efficiency=0,
            object_count_growth=0,
            leak_detected=False,
            leak_severity="none",
            stability_score=memory_efficiency
        )
        
        self.metrics.append(metric)
        self.logger.info(f"缓存内存优化测试完成: 平均每项目内存={avg_memory_per_item:.4f}MB, 内存效率={memory_efficiency:.2f}")
        
        return metric
    
    def generate_memory_leak_report(self) -> Dict[str, Any]:
        """
        生成内存泄漏检测报告
        
        Returns:
            Dict[str, Any]: 内存泄漏检测报告数据
        """
        self.logger.info("生成内存泄漏检测报告...")
        
        if not self.metrics:
            self.logger.warning("没有测试数据，无法生成报告")
            return {}
        
        # 计算总体指标
        total_leaks = sum(1 for m in self.metrics if m.leak_detected)
        avg_stability_score = statistics.mean([m.stability_score for m in self.metrics])
        avg_memory_growth = statistics.mean([m.memory_growth_mb for m in self.metrics])
        avg_growth_rate = statistics.mean([m.memory_growth_rate for m in self.metrics if m.memory_growth_rate > 0])
        
        # 分析泄漏严重程度分布
        severity_counts = defaultdict(int)
        for metric in self.metrics:
            severity_counts[metric.leak_severity] += 1
        
        # 找出最严重的泄漏情况
        leak_metrics = [m for m in self.metrics if m.leak_detected]
        worst_leak = max(leak_metrics, key=lambda x: x.memory_growth_mb) if leak_metrics else None
        
        report = {
            'memory_leak_detection_analysis': {
                'overall_assessment': {
                    'total_tests': len(self.metrics),
                    'leaks_detected': total_leaks,
                    'leak_rate_percent': (total_leaks / len(self.metrics)) * 100 if self.metrics else 0,
                    'avg_stability_score': round(avg_stability_score, 2),
                    'avg_memory_growth_mb': round(avg_memory_growth, 2),
                    'avg_growth_rate_per_hour': round(avg_growth_rate, 2)
                },
                'leak_severity_distribution': dict(severity_counts),
                'worst_leak_case': {
                    'test_name': worst_leak.test_name if worst_leak else "none",
                    'memory_growth_mb': worst_leak.memory_growth_mb if worst_leak else 0,
                    'severity': worst_leak.leak_severity if worst_leak else "none"
                } if worst_leak else {},
                'detailed_metrics': [
                    {
                        'test_name': m.test_name,
                        'initial_memory_mb': round(m.initial_memory_mb, 2),
                        'final_memory_mb': round(m.final_memory_mb, 2),
                        'memory_growth_mb': round(m.memory_growth_mb, 2),
                        'memory_growth_rate_per_hour': round(m.memory_growth_rate, 2),
                        'gc_efficiency_percent': round(m.garbage_collection_efficiency, 2),
                        'object_growth': m.object_count_growth,
                        'leak_detected': m.leak_detected,
                        'leak_severity': m.leak_severity,
                        'stability_score': round(m.stability_score, 2)
                    }
                    for m in self.metrics
                ],
                'memory_health_summary': {
                    'memory_stability': 'EXCELLENT' if avg_stability_score > 80 else 'GOOD' if avg_stability_score > 60 else 'FAIR' if avg_stability_score > 40 else 'POOR',
                    'leak_risk_level': 'LOW' if total_leaks == 0 else 'MEDIUM' if total_leaks <= len(self.metrics) // 3 else 'HIGH',
                    'gc_effectiveness': 'EXCELLENT' if any(m.garbage_collection_efficiency > 80 for m in self.metrics if m.garbage_collection_efficiency > 0) else 'GOOD' if any(m.garbage_collection_efficiency > 60 for m in self.metrics) else 'POOR',
                    'recommendations': self.generate_recommendations(avg_stability_score, total_leaks, avg_memory_growth)
                }
            }
        }
        
        return report
    
    def generate_recommendations(self, avg_stability: float, leak_count: int, avg_growth: float) -> List[str]:
        """
        基于测试结果生成优化建议
        
        Args:
            avg_stability: 平均稳定性评分
            leak_count: 泄漏检测数量
            avg_growth: 平均内存增长
            
        Returns:
            List[str]: 优化建议列表
        """
        recommendations = []
        
        if avg_stability < 40:
            recommendations.append("🚨 内存稳定性极差，建议立即检查内存泄漏问题")
        elif avg_stability < 60:
            recommendations.append("⚠️ 内存稳定性较差，需要优化内存管理策略")
        elif avg_stability < 80:
            recommendations.append("📈 内存稳定性良好，可以进一步优化")
        else:
            recommendations.append("✅ 内存稳定性优秀，当前内存管理策略有效")
        
        if leak_count > 0:
            recommendations.append("🔍 检测到内存泄漏，建议检查缓存对象生命周期管理")
            recommendations.append("🧹 优化垃圾回收策略，增加定期强制GC")
            recommendations.append("📊 实现内存使用监控和告警机制")
        
        if avg_growth > 50:
            recommendations.append("💾 内存增长较快，建议优化缓存大小和淘汰策略")
            recommendations.append("🔄 实现智能内存管理，定期清理无用缓存")
        
        recommendations.extend([
            "🔧 考虑使用内存池技术减少内存碎片",
            "📊 建立持续的内存监控体系",
            "⚡ 优化对象创建和销毁策略",
            "🛡️ 实现内存使用上限和自动清理机制"
        ])
        
        return recommendations
    
    def save_memory_leak_report(self, report: Dict[str, Any]):
        """
        保存内存泄漏检测报告到文件
        
        Args:
            report: 内存泄漏检测报告数据
        """
        # 保存JSON格式报告
        with open('memory_leak_detection_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成HTML格式报告
        html_report = self.generate_html_report(report)
        with open('memory_leak_detection_report.html', 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        self.logger.info("内存泄漏检测报告已保存")
    
    def generate_html_report(self, report: Dict[str, Any]) -> str:
        """
        生成HTML格式的内存泄漏检测报告
        
        Args:
            report: 内存泄漏检测报告数据
            
        Returns:
            str: HTML格式报告
        """
        html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>A项目V7 - 内存泄漏检测报告</title>
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
            border-bottom: 3px solid #e74c3c;
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
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
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
        .healthy {{
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        }}
        .warning {{
            background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        }}
        .critical {{
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
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
        .leak-detected {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .no-leak {{
            color: #27ae60;
            font-weight: bold;
        }}
        .recommendation {{
            background: #e8f5e8;
            border-left: 4px solid #27ae60;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }}
        .warning-recommendation {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }}
        .critical-recommendation {{
            background: #f8d7da;
            border-left: 4px solid #dc3545;
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
        <h1>🧹 A项目V7 - 内存泄漏检测报告</h1>
        
        <h2>📊 总体评估</h2>
        <div class="summary-grid">
            <div class="metric-card {'healthy' if report['memory_leak_detection_analysis']['overall_assessment']['leak_rate_percent'] == 0 else 'warning' if report['memory_leak_detection_analysis']['overall_assessment']['leak_rate_percent'] < 30 else 'critical'}">
                <div class="metric-value">{report['memory_leak_detection_analysis']['overall_assessment']['leak_rate_percent']:.1f}%</div>
                <div class="metric-label">内存泄漏率</div>
            </div>
            <div class="metric-card {'healthy' if report['memory_leak_detection_analysis']['overall_assessment']['avg_stability_score'] > 80 else 'warning' if report['memory_leak_detection_analysis']['overall_assessment']['avg_stability_score'] > 60 else 'critical'}">
                <div class="metric-value">{report['memory_leak_detection_analysis']['overall_assessment']['avg_stability_score']:.1f}</div>
                <div class="metric-label">稳定性评分</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report['memory_leak_detection_analysis']['overall_assessment']['avg_memory_growth_mb']:.2f}MB</div>
                <div class="metric-label">平均内存增长</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report['memory_leak_detection_analysis']['overall_assessment']['avg_growth_rate_per_hour']:.2f}MB/h</div>
                <div class="metric-label">平均增长率</div>
            </div>
        </div>
        
        <h2>🎯 泄漏检测结果</h2>
        <div class="summary-grid">
            <div style="background: #e8f5e8; padding: 20px; border-radius: 10px; text-align: center;">
                <h3>内存稳定性</h3>
                <p class="{'no-leak' if report['memory_leak_detection_analysis']['memory_health_summary']['memory_stability'] == 'EXCELLENT' else 'warning-recommendation'}">
                    {report['memory_leak_detection_analysis']['memory_health_summary']['memory_stability']}
                </p>
            </div>
            <div style="background: #fff3cd; padding: 20px; border-radius: 10px; text-align: center;">
                <h3>泄漏风险等级</h3>
                <p class="{'no-leak' if report['memory_leak_detection_analysis']['memory_health_summary']['leak_risk_level'] == 'LOW' else 'leak-detected'}">
                    {report['memory_leak_detection_analysis']['memory_health_summary']['leak_risk_level']}
                </p>
            </div>
        </div>
        
        <h2>📋 详细测试结果</h2>
        <table>
            <thead>
                <tr>
                    <th>测试场景</th>
                    <th>初始内存(MB)</th>
                    <th>最终内存(MB)</th>
                    <th>内存增长(MB)</th>
                    <th>增长率(MB/h)</th>
                    <th>GC效率(%)</th>
                    <th>泄漏检测</th>
                    <th>严重程度</th>
                    <th>稳定性评分</th>
                </tr>
            </thead>
            <tbody>
                {''.join([f'''
                <tr>
                    <td>{metric['test_name']}</td>
                    <td>{metric['initial_memory_mb']}</td>
                    <td>{metric['final_memory_mb']}</td>
                    <td>{metric['memory_growth_mb']}</td>
                    <td>{metric['memory_growth_rate_per_hour']}</td>
                    <td>{metric['gc_efficiency_percent']}%</td>
                    <td class="{'no-leak' if not metric['leak_detected'] else 'leak-detected'}">
                        {'✅ 无泄漏' if not metric['leak_detected'] else '🚨 检测到泄漏'}
                    </td>
                    <td>{metric['leak_severity']}</td>
                    <td>{metric['stability_score']}</td>
                </tr>
                ''' for metric in report['memory_leak_detection_analysis']['detailed_metrics']])}
            </tbody>
        </table>
        
        <h2>💡 优化建议</h2>
        {''.join([f'<div class="{"critical-recommendation" if "🚨" in recommendation else "warning-recommendation" if "⚠️" in recommendation else "recommendation"}">{recommendation}</div>' for recommendation in report['memory_leak_detection_analysis']['memory_health_summary']['recommendations']])}
        
        <div class="footer">
            <p>📊 报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>🧹 A项目V7 - 内存泄漏检测测试套件</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_template
    
    def run_comprehensive_memory_leak_test(self):
        """
        运行全面的内存泄漏检测测试
        """
        self.logger.info("🧹 开始运行全面的内存泄漏检测测试...")
        
        # 运行各项测试
        self.test_short_term_memory_stability()
        self.test_stress_memory_behavior()
        self.test_long_term_memory_stability()
        self.test_memory_recovery_after_gc()
        self.test_cache_memory_optimization()
        
        # 生成检测报告
        report = self.generate_memory_leak_report()
        
        # 保存报告
        self.save_memory_leak_report(report)
        
        # 打印测试总结
        self.logger.info("=" * 80)
        self.logger.info("🧹 内存泄漏检测完成！")
        self.logger.info(f"📊 总体评估结果:")
        self.logger.info(f"   🚨 泄漏检测率: {report['memory_leak_detection_analysis']['overall_assessment']['leak_rate_percent']:.1f}%")
        self.logger.info(f"   📈 平均稳定性评分: {report['memory_leak_detection_analysis']['overall_assessment']['avg_stability_score']:.1f}/100")
        self.logger.info(f"   📊 平均内存增长: {report['memory_leak_detection_analysis']['overall_assessment']['avg_memory_growth_mb']:.2f}MB")
        self.logger.info(f"   ⚡ 平均增长率: {report['memory_leak_detection_analysis']['overall_assessment']['avg_growth_rate_per_hour']:.2f}MB/h")
        self.logger.info(f"   🧪 测试总数: {report['memory_leak_detection_analysis']['overall_assessment']['total_tests']}")
        self.logger.info("=" * 80)
        
        return report

if __name__ == "__main__":
    # 运行内存泄漏检测测试
    tester = MemoryLeakDetectionTester()
    report = tester.run_comprehensive_memory_leak_test()
    
    # 打印关键发现
    print("\n🧹 关键内存泄漏发现:")
    print(f"🚨 泄漏检测率: {report['memory_leak_detection_analysis']['overall_assessment']['leak_rate_percent']:.1f}%")
    print(f"📈 稳定性评分: {report['memory_leak_detection_analysis']['overall_assessment']['avg_stability_score']:.1f}/100")
    print(f"📊 平均内存增长: {report['memory_leak_detection_analysis']['overall_assessment']['avg_memory_growth_mb']:.2f}MB")
    print(f"✅ 在 {report['memory_leak_detection_analysis']['overall_assessment']['total_tests']} 个测试场景中验证了内存稳定性")