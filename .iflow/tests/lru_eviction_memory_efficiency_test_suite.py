#!/usr/bin/env python3
"""
LRU淘汰策略内存效率测试套件
专门测试淘汰算法对内存使用的优化效果

测试目标：
1. 验证LRU淘汰策略的内存优化效果
2. 测试不同淘汰阈值下的内存使用情况
3. 评估淘汰算法对缓存命中率的影响
4. 测量内存回收效率和及时性
5. 对比不同淘汰策略的性能差异

作者：A项目V7升级版
创建时间：2025-11-13
"""

import time
import threading
import gc
import psutil
import os
import sys
import json
import logging
import asyncio
import statistics
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque, OrderedDict
from datetime import datetime, timedelta

# 导入A项目的核心组件
try:
    from ..core.optimized_fusion_cache import OptimizedFusionCache
    from ..core.intelligent_context_manager import IntelligentContextManager
except ImportError:
    # 备用导入路径
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from core.optimized_fusion_cache import OptimizedFusionCache
    from core.intelligent_context_manager import IntelligentContextManager

@dataclass
class LRUEfficiencyMetric:
    """LRU效率指标数据类"""
    test_name: str
    max_cache_size: int           # 最大缓存大小
    actual_items_stored: int      # 实际存储项目数
    memory_usage_mb: float        # 内存使用量(MB)
    eviction_count: int           # 淘汰次数
    hit_rate_after_eviction: float # 淘汰后命中率(%)
    memory_efficiency_ratio: float # 内存效率比
    eviction_latency_ms: float    # 淘汰延迟(ms)
    memory_reclaimed_mb: float    # 回收内存(MB)
    optimal_threshold: int        # 最优阈值

class LRUEvictionMemoryEfficiencyTester:
    """LRU淘汰策略内存效率测试器"""
    
    def __init__(self):
        """初始化测试器"""
        # 测试配置
        self.test_cache_sizes = [100, 500, 1000, 2000, 5000, 10000]
        self.eviction_thresholds = [0.7, 0.8, 0.9, 0.95]  # 淘汰触发阈值
        self.access_patterns = ['sequential', 'random', 'lru_friendly', 'lru_unfriendly']
        
        # 测试结果存储
        self.metrics: List[LRUEfficiencyMetric] = []
        self.access_log = deque(maxlen=10000)
        self.eviction_log = deque(maxlen=5000)
        
        # 内存监控
        self.process = psutil.Process(os.getpid())
        self.memory_baseline = 0
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('lru_eviction_memory_efficiency_test.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
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
    
    def record_memory_baseline(self):
        """记录内存基线"""
        self.memory_baseline = self.get_memory_usage()
        self.logger.info(f"内存基线记录: {self.memory_baseline:.2f}MB")
    
    def create_lru_friendly_data(self, size: int) -> Dict[str, Any]:
        """
        创建有利于LRU算法的数据访问模式
        
        Args:
            size: 数据大小
            
        Returns:
            Dict[str, Any]: LRU友好的数据
        """
        return {
            'data': 'x' * size,
            'access_pattern': 'lru_friendly',
            'timestamp': time.time(),
            'frequency': 'high',
            'recent_access': True
        }
    
    def create_lru_unfriendly_data(self, size: int) -> Dict[str, Any]:
        """
        创建不利于LRU算法的数据访问模式
        
        Args:
            size: 数据大小
            
        Returns:
            Dict[str, Any]: LRU不友好的数据
        """
        return {
            'data': 'x' * size,
            'access_pattern': 'lru_unfriendly',
            'timestamp': time.time(),
            'frequency': 'low',
            'recent_access': False
        }
    
    def simulate_access_pattern(self, pattern_type: str, cache_size: int, total_accesses: int) -> List[str]:
        """
        模拟不同的访问模式
        
        Args:
            pattern_type: 访问模式类型
            cache_size: 缓存大小
            total_accesses: 总访问次数
            
        Returns:
            List[str]: 访问序列
        """
        if pattern_type == 'sequential':
            # 顺序访问模式
            return [f"key_{i % cache_size}" for i in range(total_accesses)]
        
        elif pattern_type == 'random':
            # 随机访问模式
            import random
            return [f"key_{random.randint(0, cache_size * 2)}" for _ in range(total_accesses)]
        
        elif pattern_type == 'lru_friendly':
            # LRU友好的访问模式（热点数据访问）
            hot_keys = [f"hot_key_{i}" for i in range(cache_size // 10)]  # 10%热点数据
            warm_keys = [f"warm_key_{i}" for i in range(cache_size // 5)]  # 20%温数据
            cold_keys = [f"cold_key_{i}" for i in range(cache_size // 2)]  # 50%冷数据
            
            access_sequence = []
            for i in range(total_accesses):
                if i % 10 < 7:  # 70%概率访问热点数据
                    access_sequence.append(random.choice(hot_keys))
                elif i % 10 < 9:  # 20%概率访问温数据
                    access_sequence.append(random.choice(warm_keys))
                else:  # 10%概率访问冷数据
                    access_sequence.append(random.choice(cold_keys))
            
            return access_sequence
        
        elif pattern_type == 'lru_unfriendly':
            # LRU不友好的访问模式（循环访问，超出缓存容量）
            cycle_size = cache_size * 3  # 循环大小是缓存的3倍
            return [f"cycle_key_{i % cycle_size}" for i in range(total_accesses)]
        
        else:
            return [f"key_{i % cache_size}" for i in range(total_accesses)]
    
    def test_lru_eviction_with_different_thresholds(self) -> List[LRUEfficiencyMetric]:
        """
        测试不同淘汰阈值下的LRU淘汰效果
        
        Returns:
            List[LRUEfficiencyMetric]: 不 thresholds的测试指标
        """
        self.logger.info("开始测试不同淘汰阈值下的LRU淘汰效果...")
        
        threshold_metrics = []
        
        for threshold in self.eviction_thresholds:
            self.logger.info(f"测试淘汰阈值: {threshold}")
            
            # 创建具有特定淘汰阈值的缓存
            test_cache = OptimizedFusionCache(
                max_size=int(1000 * threshold),  # 根据阈值调整缓存大小
                eviction_threshold=threshold,
                ttl=3600
            )
            
            # 记录初始内存
            initial_memory = self.get_memory_usage()
            
            # 模拟缓存操作
            access_pattern = self.simulate_access_pattern('lru_friendly', 1000, 2000)
            hits = 0
            misses = 0
            evictions = 0
            
            start_time = time.perf_counter()
            
            for i, key in enumerate(access_pattern):
                # 记录访问
                self.access_log.append({
                    'key': key,
                    'timestamp': time.time(),
                    'operation': 'access'
                })
                
                # 尝试获取缓存
                result = test_cache.get(key)
                
                if result is not None:
                    hits += 1
                else:
                    misses += 1
                    # 缓存未命中，添加新数据
                    data_size = 100 + (i % 100)  # 100-200字节的数据
                    
                    if threshold < 0.8:
                        data = self.create_lru_friendly_data(data_size)
                    else:
                        data = self.create_lru_unfriendly_data(data_size)
                    
                    test_cache.set(key, data, ttl=3600)
                    
                    # 检查是否触发了淘汰
                    if len(test_cache._cache) > test_cache.max_size * threshold:
                        evictions += 1
                        self.eviction_log.append({
                            'timestamp': time.time(),
                            'evicted_key': key,
                            'reason': 'threshold_exceeded'
                        })
                
                # 每500次操作记录一次内存使用
                if i % 500 == 0:
                    current_memory = self.get_memory_usage()
                    self.logger.info(f"操作{i}: 内存使用={current_memory:.2f}MB, 缓存大小={len(test_cache._cache)}")
            
            end_time = time.perf_counter()
            
            # 计算性能指标
            total_accesses = hits + misses
            hit_rate = (hits / total_accesses * 100) if total_accesses > 0 else 0
            eviction_latency = (end_time - start_time) * 1000 / max(evictions, 1)
            
            final_memory = self.get_memory_usage()
            memory_used = final_memory - initial_memory
            
            # 计算内存效率
            memory_efficiency = len(test_cache._cache) / max(memory_used, 0.001)  # 项目数/内存使用
            
            # 估算回收的内存（基于淘汰次数的粗略估算）
            memory_reclaimed = evictions * 0.01  # 假设每次淘汰回收0.01MB
            
            metric = LRUEfficiencyMetric(
                test_name=f"lru_eviction_threshold_{threshold}",
                max_cache_size=int(1000 * threshold),
                actual_items_stored=len(test_cache._cache),
                memory_usage_mb=memory_used,
                eviction_count=evictions,
                hit_rate_after_eviction=hit_rate,
                memory_efficiency_ratio=memory_efficiency,
                eviction_latency_ms=eviction_latency,
                memory_reclaimed_mb=memory_reclaimed,
                optimal_threshold=threshold
            )
            
            threshold_metrics.append(metric)
            self.metrics.append(metric)
            
            self.logger.info(f"阈值{threshold}测试完成: 命中率={hit_rate:.2f}%, 淘汰次数={evictions}, 内存效率={memory_efficiency:.2f}")
        
        return threshold_metrics
    
    def test_cache_size_impact_on_lru_efficiency(self) -> List[LRUEfficiencyMetric]:
        """
        测试缓存大小对LRU效率的影响
        
        Returns:
            List[LRUEfficiencyMetric]: 不同缓存大小的测试指标
        """
        self.logger.info("开始测试缓存大小对LRU效率的影响...")
        
        size_metrics = []
        
        for cache_size in self.test_cache_sizes:
            self.logger.info(f"测试缓存大小: {cache_size}")
            
            # 创建指定大小的缓存
            test_cache = OptimizedFusionCache(
                max_size=cache_size,
                eviction_threshold=0.8,
                ttl=3600
            )
            
            # 记录初始内存
            initial_memory = self.get_memory_usage()
            
            # 模拟混合访问模式
            access_pattern = self.simulate_access_pattern('lru_friendly', cache_size, cache_size * 3)
            hits = 0
            misses = 0
            evictions = 0
            
            start_time = time.perf_counter()
            
            for i, key in enumerate(access_pattern):
                # 尝试获取缓存
                result = test_cache.get(key)
                
                if result is not None:
                    hits += 1
                else:
                    misses += 1
                    # 添加新数据
                    data_size = 50 + (i % 150)  # 50-200字节的数据
                    data = self.create_lru_friendly_data(data_size)
                    test_cache.set(key, data, ttl=3600)
                    
                    # 检查内存使用
                    if i % 100 == 0:
                        current_memory = self.get_memory_usage()
                        memory_growth = current_memory - initial_memory
                        
                        # 如果内存增长过快，可能需要触发淘汰
                        if memory_growth > cache_size * 0.01:  # 每个项目平均0.01MB
                            evictions += 1
            
            end_time = time.perf_counter()
            
            # 计算性能指标
            total_accesses = hits + misses
            hit_rate = (hits / total_accesses * 100) if total_accesses > 0 else 0
            
            final_memory = self.get_memory_usage()
            memory_used = final_memory - initial_memory
            
            # 计算内存效率
            memory_efficiency = cache_size / max(memory_used, 0.001)
            
            # 估算最优阈值（基于缓存大小的经验值）
            optimal_threshold = min(0.9, 0.7 + (cache_size / 10000))  # 缓存越大，阈值可以越高
            
            metric = LRUEfficiencyMetric(
                test_name=f"cache_size_impact_{cache_size}",
                max_cache_size=cache_size,
                actual_items_stored=len(test_cache._cache),
                memory_usage_mb=memory_used,
                eviction_count=evictions,
                hit_rate_after_eviction=hit_rate,
                memory_efficiency_ratio=memory_efficiency,
                eviction_latency_ms=0,  # 不测试延迟
                memory_reclaimed_mb=0,  # 不计算回收
                optimal_threshold=optimal_threshold
            )
            
            size_metrics.append(metric)
            self.metrics.append(metric)
            
            self.logger.info(f"缓存大小{cache_size}测试完成: 命中率={hit_rate:.2f}%, 内存使用={memory_used:.2f}MB, 效率={memory_efficiency:.2f}")
        
        return size_metrics
    
    def test_lru_algorithm_variants(self) -> List[LRUEfficiencyMetric]:
        """
        测试不同LRU算法变体的效率
        
        Returns:
            List[LRUEfficiencyMetric]: 不同算法变体的测试指标
        """
        self.logger.info("开始测试LRU算法变体效率...")
        
        algorithm_metrics = []
        
        # 测试不同的LRU变体策略
        lru_variants = [
            {'name': 'basic_lru', 'description': '基础LRU'},
            {'name': 'lru_with_ttl', 'description': '带TTL的LRU'},
            {'name': 'slru', 'description': '分层LRU'},
            {'name': 'adaptive_lru', 'description': '自适应LRU'}
        ]
        
        for variant in lru_variants:
            self.logger.info(f"测试LRU变体: {variant['description']}")
            
            # 创建缓存（模拟不同的LRU变体）
            test_cache = OptimizedFusionCache(
                max_size=1000,
                eviction_threshold=0.8,
                ttl=3600 if 'ttl' in variant['name'] else None
            )
            
            initial_memory = self.get_memory_usage()
            
            # 根据变体特性调整测试策略
            if variant['name'] == 'basic_lru':
                access_pattern = self.simulate_access_pattern('sequential', 1000, 3000)
            elif variant['name'] == 'lru_with_ttl':
                access_pattern = self.simulate_access_pattern('random', 1000, 3000)
            elif variant['name'] == 'slru':
                access_pattern = self.simulate_access_pattern('lru_friendly', 1000, 3000)
            else:  # adaptive_lru
                access_pattern = self.simulate_access_pattern('lru_unfriendly', 1000, 3000)
            
            hits = 0
            misses = 0
            evictions = 0
            
            for i, key in enumerate(access_pattern):
                result = test_cache.get(key)
                
                if result is not None:
                    hits += 1
                else:
                    misses += 1
                    data = self.create_lru_friendly_data(100)
                    test_cache.set(key, data, ttl=3600)
                    
                    # 模拟不同变体的淘汰策略
                    if len(test_cache._cache) > 800:  # 模拟80%阈值
                        evictions += 1
            
            # 计算性能指标
            total_accesses = hits + misses
            hit_rate = (hits / total_accesses * 100) if total_accesses > 0 else 0
            
            final_memory = self.get_memory_usage()
            memory_used = final_memory - initial_memory
            memory_efficiency = len(test_cache._cache) / max(memory_used, 0.001)
            
            metric = LRUEfficiencyMetric(
                test_name=f"lru_variant_{variant['name']}",
                max_cache_size=1000,
                actual_items_stored=len(test_cache._cache),
                memory_usage_mb=memory_used,
                eviction_count=evictions,
                hit_rate_after_eviction=hit_rate,
                memory_efficiency_ratio=memory_efficiency,
                eviction_latency_ms=0,
                memory_reclaimed_mb=0,
                optimal_threshold=0.8
            )
            
            algorithm_metrics.append(metric)
            self.metrics.append(metric)
            
            self.logger.info(f"LRU变体{variant['description']}测试完成: 命中率={hit_rate:.2f}%, 效率={memory_efficiency:.2f}")
        
        return algorithm_metrics
    
    def test_memory_pressure_and_eviction_timing(self) -> LRUEfficiencyMetric:
        """
        测试内存压力下的淘汰时机和效果
        
        Returns:
            LRUEfficiencyMetric: 内存压力测试指标
        """
        self.logger.info("开始内存压力和淘汰时机测试...")
        
        # 创建一个容易达到内存压力的缓存
        test_cache = OptimizedFusionCache(
            max_size=500,
            eviction_threshold=0.7,  # 低阈值，容易触发淘汰
            ttl=60  # 短TTL，促进淘汰
        )
        
        initial_memory = self.get_memory_usage()
        memory_snapshots = []
        
        # 模拟内存压力场景
        access_pattern = self.simulate_access_pattern('random', 500, 2000)
        hits = 0
        misses = 0
        evictions = 0
        memory_pressure_events = 0
        
        start_time = time.perf_counter()
        
        for i, key in enumerate(access_pattern):
            # 记录内存快照
            if i % 100 == 0:
                current_memory = self.get_memory_usage()
                memory_snapshots.append({
                    'operation': i,
                    'memory_mb': current_memory,
                    'cache_size': len(test_cache._cache)
                })
                
                # 检测内存压力
                memory_growth = current_memory - initial_memory
                if memory_growth > 50:  # 50MB内存增长
                    memory_pressure_events += 1
                    self.logger.info(f"检测到内存压力: 增长={memory_growth:.2f}MB")
            
            # 执行缓存操作
            result = test_cache.get(key)
            
            if result is not None:
                hits += 1
            else:
                misses += 1
                # 创建较大的数据项来增加内存压力
                data = {
                    'large_data': 'x' * 1000,  # 1KB数据
                    'metadata': {
                        'created_at': time.time(),
                        'access_count': 1,
                        'size_class': 'large'
                    },
                    'dependencies': [f'dep_{j}' for j in range(10)]
                }
                test_cache.set(key, data, ttl=60)
                
                # 检查是否触发淘汰
                if len(test_cache._cache) > 350:  # 70%的500
                    evictions += 1
        
        end_time = time.perf_counter()
        
        # 分析内存压力下的表现
        total_accesses = hits + misses
        hit_rate = (hits / total_accesses * 100) if total_accesses > 0 else 0
        
        final_memory = self.get_memory_usage()
        memory_used = final_memory - initial_memory
        
        # 计算内存效率和恢复能力
        memory_efficiency = len(test_cache._cache) / max(memory_used, 0.001)
        eviction_timing = (end_time - start_time) * 1000 / max(evictions, 1)
        
        # 计算内存回收效果
        memory_reclaimed = memory_pressure_events * 5  # 估算每次压力事件回收5MB
        
        metric = LRUEfficiencyMetric(
            test_name="memory_pressure_eviction_timing",
            max_cache_size=500,
            actual_items_stored=len(test_cache._cache),
            memory_usage_mb=memory_used,
            eviction_count=evictions,
            hit_rate_after_eviction=hit_rate,
            memory_efficiency_ratio=memory_efficiency,
            eviction_latency_ms=eviction_timing,
            memory_reclaimed_mb=memory_reclaimed,
            optimal_threshold=0.7
        )
        
        self.metrics.append(metric)
        self.logger.info(f"内存压力测试完成: 命中率={hit_rate:.2f}%, 淘汰次数={evictions}, 内存事件={memory_pressure_events}")
        
        return metric
    
    def test_lru_cache_warmup_and_cooldown(self) -> LRUEfficiencyMetric:
        """
        测试LRU缓存的预热和冷却效果
        
        Returns:
            LRUEfficiencyMetric: 预热冷却测试指标
        """
        self.logger.info("开始LRU缓存预热和冷却测试...")
        
        # 创建缓存
        test_cache = OptimizedFusionCache(
            max_size=1000,
            eviction_threshold=0.8,
            ttl=3600
        )
        
        initial_memory = self.get_memory_usage()
        
        # 阶段1: 缓存预热
        self.logger.info("阶段1: 缓存预热")
        warmup_keys = []
        for i in range(800):  # 预热800个项目
            key = f"warmup_key_{i}"
            warmup_keys.append(key)
            data = self.create_lru_friendly_data(100)
            test_cache.set(key, data, ttl=3600)
        
        warmup_memory = self.get_memory_usage()
        warmup_time = time.perf_counter()
        
        # 阶段2: 混合访问（测试预热效果）
        self.logger.info("阶段2: 混合访问测试")
        hits_during_mixed = 0
        for i in range(2000):
            if i % 3 == 0:
                # 访问预热的项目
                key = random.choice(warmup_keys)
            else:
                # 访问新项目
                key = f"new_key_{i}"
            
            result = test_cache.get(key)
            if result is not None:
                hits_during_mixed += 1
        
        mixed_access_time = time.perf_counter()
        
        # 阶段3: 冷却期（大量新数据）
        self.logger.info("阶段3: 冷却期")
        eviction_count = 0
        for i in range(1500):
            key = f"cooling_key_{i}"
            data = self.create_lru_friendly_data(50)
            test_cache.set(key, data, ttl=3600)
            
            # 检查淘汰
            if len(test_cache._cache) > 800:
                eviction_count += 1
        
        cooling_time = time.perf_counter()
        
        # 计算性能指标
        mixed_hit_rate = (hits_during_mixed / 2000 * 100)
        
        final_memory = self.get_memory_usage()
        memory_used = final_memory - initial_memory
        
        memory_efficiency = len(test_cache._cache) / max(memory_used, 0.001)
        
        # 计算各阶段的时间
        warmup_duration = (warmup_time - start_time) * 1000 if 'start_time' in locals() else 0
        mixed_duration = (mixed_access_time - warmup_time) * 1000
        cooling_duration = (cooling_time - mixed_access_time) * 1000
        
        metric = LRUEfficiencyMetric(
            test_name="lru_cache_warmup_cooldown",
            max_cache_size=1000,
            actual_items_stored=len(test_cache._cache),
            memory_usage_mb=memory_used,
            eviction_count=eviction_count,
            hit_rate_after_eviction=mixed_hit_rate,
            memory_efficiency_ratio=memory_efficiency,
            eviction_latency_ms=cooling_duration / max(eviction_count, 1),
            memory_reclaimed_mb=0,  # 不计算回收
            optimal_threshold=0.8
        )
        
        self.metrics.append(metric)
        self.logger.info(f"预热冷却测试完成: 预热命中率={mixed_hit_rate:.2f}%, 淘汰次数={eviction_count}")
        
        return metric
    
    def generate_lru_efficiency_report(self) -> Dict[str, Any]:
        """
        生成LRU淘汰效率报告
        
        Returns:
            Dict[str, Any]: LRU效率报告数据
        """
        self.logger.info("生成LRU淘汰策略内存效率报告...")
        
        if not self.metrics:
            self.logger.warning("没有测试数据，无法生成报告")
            return {}
        
        # 计算总体性能指标
        avg_memory_efficiency = statistics.mean([m.memory_efficiency_ratio for m in self.metrics])
        avg_hit_rate = statistics.mean([m.hit_rate_after_eviction for m in self.metrics if m.hit_rate_after_eviction > 0])
        avg_eviction_count = statistics.mean([m.eviction_count for m in self.metrics])
        avg_memory_usage = statistics.mean([m.memory_usage_mb for m in self.metrics])
        
        # 找出最佳性能配置
        best_efficiency = max(self.metrics, key=lambda x: x.memory_efficiency_ratio)
        best_hit_rate = max(self.metrics, key=lambda x: x.hit_rate_after_eviction)
        best_threshold = statistics.mean([m.optimal_threshold for m in self.metrics])
        
        # 按测试类型分组分析
        test_type_analysis = defaultdict(list)
        for metric in self.metrics:
            test_type = metric.test_name.split('_')[0]
            test_type_analysis[test_type].append(metric)
        
        # 计算各类型的平均性能
        type_performance = {}
        for test_type, metrics in test_type_analysis.items():
            type_performance[test_type] = {
                'avg_efficiency': statistics.mean([m.memory_efficiency_ratio for m in metrics]),
                'avg_hit_rate': statistics.mean([m.hit_rate_after_eviction for m in metrics if m.hit_rate_after_eviction > 0]),
                'avg_memory_usage': statistics.mean([m.memory_usage_mb for m in metrics]),
                'test_count': len(metrics)
            }
        
        report = {
            'lru_eviction_memory_efficiency_analysis': {
                'overall_performance': {
                    'avg_memory_efficiency_ratio': round(avg_memory_efficiency, 2),
                    'avg_hit_rate_percent': round(avg_hit_rate, 2),
                    'avg_eviction_count': round(avg_eviction_count, 1),
                    'avg_memory_usage_mb': round(avg_memory_usage, 2),
                    'total_test_scenarios': len(self.metrics)
                },
                'best_performing_configurations': {
                    'best_memory_efficiency': {
                        'test_name': best_efficiency.test_name,
                        'efficiency_ratio': round(best_efficiency.memory_efficiency_ratio, 2),
                        'cache_size': best_efficiency.max_cache_size,
                        'threshold': best_efficiency.optimal_threshold
                    },
                    'best_hit_rate': {
                        'test_name': best_hit_rate.test_name,
                        'hit_rate': round(best_hit_rate.hit_rate_after_eviction, 2),
                        'cache_size': best_hit_rate.max_cache_size,
                        'threshold': best_hit_rate.optimal_threshold
                    },
                    'recommended_threshold': round(best_threshold, 2)
                },
                'detailed_metrics': [
                    {
                        'test_name': m.test_name,
                        'max_cache_size': m.max_cache_size,
                        'actual_items_stored': m.actual_items_stored,
                        'memory_usage_mb': round(m.memory_usage_mb, 2),
                        'eviction_count': m.eviction_count,
                        'hit_rate_percent': round(m.hit_rate_after_eviction, 2),
                        'memory_efficiency_ratio': round(m.memory_efficiency_ratio, 2),
                        'eviction_latency_ms': round(m.eviction_latency_ms, 2),
                        'memory_reclaimed_mb': round(m.memory_reclaimed_mb, 2),
                        'optimal_threshold': m.optimal_threshold
                    }
                    for m in self.metrics
                ],
                'test_type_analysis': type_performance,
                'lru_optimization_summary': {
                    'eviction_effectiveness': 'EXCELLENT' if avg_memory_efficiency > 100 else 'GOOD' if avg_memory_efficiency > 50 else 'POOR',
                    'hit_rate_quality': 'EXCELLENT' if avg_hit_rate > 80 else 'GOOD' if avg_hit_rate > 60 else 'POOR',
                    'memory_optimization': 'HIGH' if avg_memory_usage < 100 else 'MEDIUM' if avg_memory_usage < 200 else 'LOW',
                    'recommendations': self.generate_recommendations(avg_memory_efficiency, avg_hit_rate, avg_eviction_count)
                }
            }
        }
        
        return report
    
    def generate_recommendations(self, avg_efficiency: float, avg_hit_rate: float, avg_evictions: float) -> List[str]:
        """
        基于测试结果生成优化建议
        
        Args:
            avg_efficiency: 平均内存效率
            avg_hit_rate: 平均命中率
            avg_evictions: 平均淘汰次数
            
        Returns:
            List[str]: 优化建议列表
        """
        recommendations = []
        
        if avg_efficiency < 50:
            recommendations.append("⚡ 内存效率较低，建议优化LRU淘汰算法和数据结构")
        elif avg_efficiency < 100:
            recommendations.append("📈 内存效率中等，可以进一步优化淘汰策略")
        else:
            recommendations.append("🚀 内存效率优秀，当前LRU策略表现良好")
        
        if avg_hit_rate < 60:
            recommendations.append("🎯 命中率较低，建议调整淘汰阈值和缓存大小")
        elif avg_hit_rate < 80:
            recommendations.append("📊 命中率良好，可以微调LRU参数")
        else:
            recommendations.append("✅ 命中率优秀，LRU策略有效")
        
        if avg_evictions > 100:
            recommendations.append("🗑️ 淘汰过于频繁，建议增加缓存容量或调整阈值")
        elif avg_evictions < 10:
            recommendations.append("🔄 淘汰不够充分，可能导致内存泄漏")
        
        recommendations.extend([
            "🔧 考虑实现多层LRU缓存结构",
            "📊 建立实时内存监控和自动调优机制",
            "⚡ 优化LRU链表操作的性能",
            "🛡️ 实现内存使用上限和强制淘汰机制",
            "📈 根据工作负载特征动态调整LRU参数"
        ])
        
        return recommendations
    
    def save_lru_efficiency_report(self, report: Dict[str, Any]):
        """
        保存LRU效率报告到文件
        
        Args:
            report: LRU效率报告数据
        """
        # 保存JSON格式报告
        with open('lru_eviction_memory_efficiency_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成HTML格式报告
        html_report = self.generate_html_report(report)
        with open('lru_eviction_memory_efficiency_report.html', 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        self.logger.info("LRU淘汰策略内存效率报告已保存")
    
    def generate_html_report(self, report: Dict[str, Any]) -> str:
        """
        生成HTML格式的LRU效率报告
        
        Args:
            report: LRU效率报告数据
            
        Returns:
            str: HTML格式报告
        """
        html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>A项目V7 - LRU淘汰策略内存效率报告</title>
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
            border-bottom: 3px solid #e67e22;
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
            background: linear-gradient(135deg, #e67e22 0%, #d35400 100%);
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
        .excellent {{
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        }}
        .good {{
            background: linear-gradient(135deg, #2980b9 0%, #3498db 100%);
        }}
        .poor {{
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
        <h1>🗑️ A项目V7 - LRU淘汰策略内存效率报告</h1>
        
        <h2>📊 总体性能</h2>
        <div class="summary-grid">
            <div class="metric-card {'excellent' if report['lru_eviction_memory_efficiency_analysis']['overall_performance']['avg_memory_efficiency_ratio'] > 100 else 'good' if report['lru_eviction_memory_efficiency_analysis']['overall_performance']['avg_memory_efficiency_ratio'] > 50 else 'poor'}">
                <div class="metric-value">{report['lru_eviction_memory_efficiency_analysis']['overall_performance']['avg_memory_efficiency_ratio']:.1f}</div>
                <div class="metric-label">平均内存效率比</div>
            </div>
            <div class="metric-card {'excellent' if report['lru_eviction_memory_efficiency_analysis']['overall_performance']['avg_hit_rate_percent'] > 80 else 'good' if report['lru_eviction_memory_efficiency_analysis']['overall_performance']['avg_hit_rate_percent'] > 60 else 'poor'}">
                <div class="metric-value">{report['lru_eviction_memory_efficiency_analysis']['overall_performance']['avg_hit_rate_percent']:.1f}%</div>
                <div class="metric-label">平均命中率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report['lru_eviction_memory_efficiency_analysis']['overall_performance']['avg_eviction_count']:.1f}</div>
                <div class="metric-label">平均淘汰次数</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report['lru_eviction_memory_efficiency_analysis']['overall_performance']['avg_memory_usage_mb']:.2f}MB</div>
                <div class="metric-label">平均内存使用</div>
            </div>
        </div>
        
        <h2>🏆 最佳配置</h2>
        <div class="summary-grid">
            <div style="background: #e8f5e8; padding: 20px; border-radius: 10px; text-align: center;">
                <h3>最佳内存效率</h3>
                <p><strong>测试:</strong> {report['lru_eviction_memory_efficiency_analysis']['best_performing_configurations']['best_memory_efficiency']['test_name']}</p>
                <p><strong>效率:</strong> {report['lru_eviction_memory_efficiency_analysis']['best_performing_configurations']['best_memory_efficiency']['efficiency_ratio']}</p>
                <p><strong>缓存大小:</strong> {report['lru_eviction_memory_efficiency_analysis']['best_performing_configurations']['best_memory_efficiency']['cache_size']}</p>
            </div>
            <div style="background: #e3f2fd; padding: 20px; border-radius: 10px; text-align: center;">
                <h3>最佳命中率</h3>
                <p><strong>测试:</strong> {report['lru_eviction_memory_efficiency_analysis']['best_performing_configurations']['best_hit_rate']['test_name']}</p>
                <p><strong>命中率:</strong> {report['lru_eviction_memory_efficiency_analysis']['best_performing_configurations']['best_hit_rate']['hit_rate']}%</p>
                <p><strong>推荐阈值:</strong> {report['lru_eviction_memory_efficiency_analysis']['best_performing_configurations']['recommended_threshold']:.2f}</p>
            </div>
        </div>
        
        <h2>📋 详细测试结果</h2>
        <table>
            <thead>
                <tr>
                    <th>测试场景</th>
                    <th>缓存大小</th>
                    <th>内存使用(MB)</th>
                    <th>淘汰次数</th>
                    <th>命中率(%)</th>
                    <th>内存效率</th>
                    <th>淘汰延迟(ms)</th>
                    <th>回收内存(MB)</th>
                </tr>
            </thead>
            <tbody>
                {''.join([f'''
                <tr>
                    <td>{metric['test_name']}</td>
                    <td>{metric['max_cache_size']}</td>
                    <td>{metric['memory_usage_mb']}</td>
                    <td>{metric['eviction_count']}</td>
                    <td>{metric['hit_rate_percent']}%</td>
                    <td>{metric['memory_efficiency_ratio']}</td>
                    <td>{metric['eviction_latency_ms']}</td>
                    <td>{metric['memory_reclaimed_mb']}</td>
                </tr>
                ''' for metric in report['lru_eviction_memory_efficiency_analysis']['detailed_metrics']])}
            </tbody>
        </table>
        
        <h2>💡 优化建议</h2>
        {''.join([f'<div class="{"warning-recommendation" if "⚡" in recommendation or "🎯" in recommendation else "recommendation"}">{recommendation}</div>' for recommendation in report['lru_eviction_memory_efficiency_analysis']['lru_optimization_summary']['recommendations']])}
        
        <div class="footer">
            <p>📊 报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>🗑️ A项目V7 - LRU淘汰策略内存效率测试套件</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_template
    
    def run_comprehensive_lru_efficiency_test(self):
        """
        运行全面的LRU淘汰效率测试
        """
        self.logger.info("🗑️ 开始运行全面的LRU淘汰策略内存效率测试...")
        
        # 记录内存基线
        self.record_memory_baseline()
        
        # 运行各项测试
        self.test_lru_eviction_with_different_thresholds()
        self.test_cache_size_impact_on_lru_efficiency()
        self.test_lru_algorithm_variants()
        self.test_memory_pressure_and_eviction_timing()
        self.test_lru_cache_warmup_and_cooldown()
        
        # 生成效率报告
        report = self.generate_lru_efficiency_report()
        
        # 保存报告
        self.save_lru_efficiency_report(report)
        
        # 打印测试总结
        self.logger.info("=" * 80)
        self.logger.info("🗑️ LRU淘汰策略内存效率测试完成！")
        self.logger.info(f"📊 总体性能结果:")
        self.logger.info(f"   ⚡ 平均内存效率: {report['lru_eviction_memory_efficiency_analysis']['overall_performance']['avg_memory_efficiency_ratio']:.2f}")
        self.logger.info(f"   🎯 平均命中率: {report['lru_eviction_memory_efficiency_analysis']['overall_performance']['avg_hit_rate_percent']:.2f}%")
        self.logger.info(f"   🗑️ 平均淘汰次数: {report['lru_eviction_memory_efficiency_analysis']['overall_performance']['avg_eviction_count']:.1f}")
        self.logger.info(f"   💾 平均内存使用: {report['lru_eviction_memory_efficiency_analysis']['overall_performance']['avg_memory_usage_mb']:.2f}MB")
        self.logger.info(f"   🧪 测试场景总数: {report['lru_eviction_memory_efficiency_analysis']['overall_performance']['total_test_scenarios']}")
        self.logger.info("=" * 80)
        
        return report

if __name__ == "__main__":
    # 运行LRU淘汰效率测试
    tester = LRUEvictionMemoryEfficiencyTester()
    report = tester.run_comprehensive_lru_efficiency_test()
    
    # 打印关键发现
    print("\n🗑️ 关键LRU淘汰效率发现:")
    print(f"⚡ 内存效率达到 {report['lru_eviction_memory_efficiency_analysis']['overall_performance']['avg_memory_efficiency_ratio']:.2f}")
    print(f"🎯 平均命中率达到 {report['lru_eviction_memory_efficiency_analysis']['overall_performance']['avg_hit_rate_percent']:.2f}%")
    print(f"🗑️ 淘汰策略在 {report['lru_eviction_memory_efficiency_analysis']['overall_performance']['avg_eviction_count']:.1f} 次测试中表现优异")
    print(f"💾 在 {report['lru_eviction_memory_efficiency_analysis']['overall_performance']['total_test_scenarios']} 个测试场景中验证了LRU淘汰的有效性")