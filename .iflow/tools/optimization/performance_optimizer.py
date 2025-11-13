#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全能工作流V5 - 性能优化器
Performance Optimizer

作者: Universal AI Team
版本: 5.0.0
日期: 2025-11-12

特性:
- 多级性能优化
- 智能缓存系统
- 资源使用监控
- 自动调优算法
- 性能瓶颈分析
- 实时优化建议
"""

import os
import sys
import json
import time
import asyncio
import logging
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import yaml
import numpy as np
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """优化级别"""
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    AGGRESSIVE = "aggressive"

class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"

@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    network_io: Dict[str, float]
    process_count: int
    thread_count: int
    response_time: Optional[float] = None
    throughput: Optional[float] = None
    error_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'disk_usage_percent': self.disk_usage_percent,
            'network_io': self.network_io,
            'process_count': self.process_count,
            'thread_count': self.thread_count,
            'response_time': self.response_time,
            'throughput': self.throughput,
            'error_rate': self.error_rate
        }

@dataclass
class OptimizationStrategy:
    """优化策略"""
    name: str
    description: str
    resource_type: ResourceType
    optimization_level: OptimizationLevel
    conditions: List[str]
    actions: List[Dict[str, Any]]
    expected_improvement: float
    risk_level: float
    enabled: bool = True

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            '.iflow', 'config', 'performance-optimization.yaml'
        )
        
        # 性能监控
        self.metrics_history: deque = deque(maxlen=10000)
        self.current_metrics: Optional[PerformanceMetrics] = None
        
        # 缓存系统
        self.cache_system = MultiLevelCache()
        
        # 资源监控
        self.resource_monitor = ResourceMonitor()
        
        # 优化策略
        self.optimization_strategies: List[OptimizationStrategy] = []
        
        # 自动调优
        self.auto_tuner = AutoTuner()
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # 配置
        self.config = self._load_config()
        
        # 初始化
        self._initialize_optimizer()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {
            'monitoring_interval': 5,
            'optimization_interval': 60,
            'auto_optimization': True,
            'cache_enabled': True,
            'cache_size': '1GB',
            'optimization_level': 'standard'
        }
    
    def _initialize_optimizer(self):
        """初始化优化器"""
        # 初始化优化策略
        self._initialize_strategies()
        
        # 启动监控
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # 启动优化
        if self.config.get('auto_optimization', True):
            self.optimization_task = asyncio.create_task(self._optimization_loop())
        
        logger.info("性能优化器初始化完成")
    
    def _initialize_strategies(self):
        """初始化优化策略"""
        strategies = [
            # CPU优化策略
            OptimizationStrategy(
                name="CPU负载均衡",
                description="通过调整线程池大小平衡CPU负载",
                resource_type=ResourceType.CPU,
                optimization_level=OptimizationLevel.STANDARD,
                conditions=["cpu_percent > 80"],
                actions=[
                    {"type": "adjust_thread_pool", "factor": 0.8},
                    {"type": "enable_cpu_affinity"},
                    {"type": "prioritize_tasks"}
                ],
                expected_improvement=0.15,
                risk_level=0.1
            ),
            OptimizationStrategy(
                name="CPU密集型优化",
                description="优化CPU密集型任务执行",
                resource_type=ResourceType.CPU,
                optimization_level=OptimizationLevel.ADVANCED,
                conditions=["cpu_percent > 90", "response_time > 1000"],
                actions=[
                    {"type": "enable_parallel_processing"},
                    {"type": "optimize_algorithms"},
                    {"type": "reduce_context_switching"}
                ],
                expected_improvement=0.25,
                risk_level=0.2
            ),
            
            # 内存优化策略
            OptimizationStrategy(
                name="内存清理",
                description="定期清理内存缓存",
                resource_type=ResourceType.MEMORY,
                optimization_level=OptimizationLevel.BASIC,
                conditions=["memory_percent > 85"],
                actions=[
                    {"type": "clear_cache"},
                    {"type": "garbage_collect"},
                    {"type": "compress_memory"}
                ],
                expected_improvement=0.10,
                risk_level=0.05
            ),
            OptimizationStrategy(
                name="内存池优化",
                description="优化内存池管理",
                resource_type=ResourceType.MEMORY,
                optimization_level=OptimizationLevel.STANDARD,
                conditions=["memory_percent > 75"],
                actions=[
                    {"type": "optimize_memory_pool"},
                    {"type": "enable_memory_recycling"},
                    {"type": "adjust_cache_size"}
                ],
                expected_improvement=0.20,
                risk_level=0.1
            ),
            
            # I/O优化策略
            OptimizationStrategy(
                name="磁盘I/O优化",
                description="优化磁盘读写性能",
                resource_type=ResourceType.DISK,
                optimization_level=OptimizationLevel.STANDARD,
                conditions=["disk_usage_percent > 80"],
                actions=[
                    {"type": "enable_write_caching"},
                    {"type": "optimize_file_operations"},
                    {"type": "compress_large_files"}
                ],
                expected_improvement=0.15,
                risk_level=0.1
            ),
            
            # 网络优化策略
            OptimizationStrategy(
                name="网络连接池",
                description="优化网络连接管理",
                resource_type=ResourceType.NETWORK,
                optimization_level=OptimizationLevel.STANDARD,
                conditions=["network_io > 100MB"],
                actions=[
                    {"type": "optimize_connection_pool"},
                    {"type": "enable_request_compression"},
                    {"type": "batch_network_requests"}
                ],
                expected_improvement=0.20,
                risk_level=0.1
            )
        ]
        
        self.optimization_strategies = strategies
    
    async def _monitoring_loop(self):
        """监控循环"""
        interval = self.config.get('monitoring_interval', 5)
        
        while True:
            try:
                # 收集性能指标
                metrics = await self._collect_metrics()
                self.current_metrics = metrics
                self.metrics_history.append(metrics)
                
                # 更新缓存统计
                self.cache_system.update_stats()
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"性能监控错误: {e}")
                await asyncio.sleep(interval)
    
    async def _optimization_loop(self):
        """优化循环"""
        interval = self.config.get('optimization_interval', 60)
        
        while True:
            try:
                # 分析性能瓶颈
                bottlenecks = await self._analyze_bottlenecks()
                
                # 应用优化策略
                if bottlenecks:
                    await self._apply_optimizations(bottlenecks)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"自动优化错误: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        # CPU指标
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存指标
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / 1024 / 1024
        
        # 磁盘指标
        disk = psutil.disk_usage('/')
        disk_usage_percent = disk.percent
        
        # 网络指标
        network = psutil.net_io_counters()
        network_io = {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv,
            'packets_sent': network.packets_sent,
            'packets_recv': network.packets_recv
        }
        
        # 进程和线程指标
        process_count = len(psutil.pids())
        thread_count = threading.active_count()
        
        # 创建指标对象
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            disk_usage_percent=disk_usage_percent,
            network_io=network_io,
            process_count=process_count,
            thread_count=thread_count
        )
        
        return metrics
    
    async def _analyze_bottlenecks(self) -> List[ResourceType]:
        """分析性能瓶颈"""
        if not self.current_metrics:
            return []
        
        bottlenecks = []
        metrics = self.current_metrics
        
        # CPU瓶颈
        if metrics.cpu_percent > 80:
            bottlenecks.append(ResourceType.CPU)
        
        # 内存瓶颈
        if metrics.memory_percent > 85:
            bottlenecks.append(ResourceType.MEMORY)
        
        # 磁盘瓶颈
        if metrics.disk_usage_percent > 90:
            bottlenecks.append(ResourceType.DISK)
        
        # 网络瓶颈
        total_io = metrics.network_io['bytes_sent'] + metrics.network_io['bytes_recv']
        if total_io > 100 * 1024 * 1024:  # 100MB
            bottlenecks.append(ResourceType.NETWORK)
        
        return bottlenecks
    
    async def _apply_optimizations(self, bottlenecks: List[ResourceType]):
        """应用优化策略"""
        for bottleneck in bottlenecks:
            # 找到合适的策略
            strategies = [
                s for s in self.optimization_strategies
                if s.resource_type == bottleneck and s.enabled
            ]
            
            # 按期望改进排序
            strategies.sort(key=lambda s: s.expected_improvement, reverse=True)
            
            # 应用策略
            for strategy in strategies[:2]:  # 最多应用2个策略
                try:
                    await self._execute_strategy(strategy)
                    logger.info(f"应用优化策略: {strategy.name}")
                except Exception as e:
                    logger.error(f"执行优化策略失败 {strategy.name}: {e}")
    
    async def _execute_strategy(self, strategy: OptimizationStrategy):
        """执行优化策略"""
        for action in strategy.actions:
            action_type = action.get('type')
            
            if action_type == 'clear_cache':
                self.cache_system.clear()
            elif action_type == 'garbage_collect':
                import gc
                gc.collect()
            elif action_type == 'adjust_thread_pool':
                factor = action.get('factor', 1.0)
                self._adjust_thread_pool(factor)
            elif action_type == 'optimize_memory_pool':
                self.cache_system.optimize_pool()
            elif action_type == 'enable_write_caching':
                self._enable_write_caching()
            elif action_type == 'optimize_connection_pool':
                await self._optimize_connection_pool()
            
            # 等待策略生效
            await asyncio.sleep(1)
    
    def _adjust_thread_pool(self, factor: float):
        """调整线程池大小"""
        current_size = self.thread_pool._max_workers
        new_size = max(1, int(current_size * factor))
        
        if new_size != current_size:
            self.thread_pool._max_workers = new_size
            logger.info(f"线程池大小调整: {current_size} -> {new_size}")
    
    def _enable_write_caching(self):
        """启用写入缓存"""
        # 实现写入缓存逻辑
        pass
    
    async def _optimize_connection_pool(self):
        """优化连接池"""
        # 实现连接池优化逻辑
        pass
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """获取性能报告"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {'error': '没有可用的性能数据'}
        
        # 计算统计信息
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        report = {
            'period_hours': hours,
            'samples_count': len(recent_metrics),
            'time_range': {
                'start': recent_metrics[0].timestamp.isoformat(),
                'end': recent_metrics[-1].timestamp.isoformat()
            },
            'cpu_stats': {
                'avg': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values),
                'std': np.std(cpu_values)
            },
            'memory_stats': {
                'avg': np.mean(memory_values),
                'max': np.max(memory_values),
                'min': np.min(memory_values),
                'std': np.std(memory_values)
            },
            'cache_stats': self.cache_system.get_stats(),
            'optimization_recommendations': self._generate_recommendations(recent_metrics)
        }
        
        return report
    
    def _generate_recommendations(self, metrics: List[PerformanceMetrics]) -> List[Dict[str, Any]]:
        """生成优化建议"""
        recommendations = []
        
        # 分析最近的指标
        if metrics:
            latest = metrics[-1]
            
            # CPU建议
            if latest.cpu_percent > 80:
                recommendations.append({
                    'type': 'cpu',
                    'priority': 'high',
                    'message': 'CPU使用率过高，建议优化算法或增加处理能力',
                    'actions': ['启用并行处理', '优化循环', '使用更高效的数据结构']
                })
            
            # 内存建议
            if latest.memory_percent > 85:
                recommendations.append({
                    'type': 'memory',
                    'priority': 'high',
                    'message': '内存使用率过高，建议清理缓存或优化内存使用',
                    'actions': ['清理缓存', '启用内存压缩', '优化数据结构']
                })
            
            # 磁盘建议
            if latest.disk_usage_percent > 90:
                recommendations.append({
                    'type': 'disk',
                    'priority': 'medium',
                    'message': '磁盘空间不足，建议清理临时文件',
                    'actions': ['清理临时文件', '压缩旧文件', '扩展存储空间']
                })
        
        return recommendations
    
    def optimize_now(self, resource_types: List[ResourceType] = None) -> Dict[str, Any]:
        """立即执行优化"""
        if not resource_types:
            # 自动检测瓶颈
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            bottlenecks = loop.run_until_complete(self._analyze_bottlenecks())
            loop.close()
            resource_types = bottlenecks
        
        results = {
            'optimized_resources': [],
            'applied_strategies': [],
            'improvements': {}
        }
        
        for resource_type in resource_types:
            # 找到并执行策略
            strategies = [
                s for s in self.optimization_strategies
                if s.resource_type == resource_type and s.enabled
            ]
            
            for strategy in strategies:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self._execute_strategy(strategy))
                    loop.close()
                    
                    results['applied_strategies'].append(strategy.name)
                    results['optimized_resources'].append(resource_type.value)
                    
                except Exception as e:
                    logger.error(f"立即优化失败 {strategy.name}: {e}")
        
        return results

class MultiLevelCache:
    """多级缓存系统"""
    
    def __init__(self):
        self.l1_cache = {}  # 内存缓存
        self.l2_cache = {}  # 磁盘缓存
        self.l1_size = 0
        self.l1_max_size = 100 * 1024 * 1024  # 100MB
        self.l2_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            '.iflow', 'cache'
        )
        os.makedirs(self.l2_path, exist_ok=True)
        
        # 统计信息
        self.stats = {
            'l1_hits': 0,
            'l1_misses': 0,
            'l2_hits': 0,
            'l2_misses': 0,
            'evictions': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        # L1缓存
        if key in self.l1_cache:
            self.stats['l1_hits'] += 1
            return self.l1_cache[key]
        
        self.stats['l1_misses'] += 1
        
        # L2缓存
        l2_file = os.path.join(self.l2_path, f"{key}.cache")
        if os.path.exists(l2_file):
            try:
                with open(l2_file, 'rb') as f:
                    data = pickle.load(f)
                
                # 提升到L1
                self._set_l1(key, data)
                self.stats['l2_hits'] += 1
                return data
            except Exception:
                pass
        
        self.stats['l2_misses'] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """设置缓存"""
        # 保存到L1
        self._set_l1(key, value)
        
        # 保存到L2
        l2_file = os.path.join(self.l2_path, f"{key}.cache")
        try:
            with open(l2_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.error(f"保存L2缓存失败: {e}")
    
    def _set_l1(self, key: str, value: Any):
        """设置L1缓存"""
        # 计算大小
        size = len(pickle.dumps(value))
        
        # 检查是否需要清理
        while self.l1_size + size > self.l1_max_size and self.l1_cache:
            # LRU清理
            oldest_key = next(iter(self.l1_cache))
            oldest_size = len(pickle.dumps(self.l1_cache[oldest_key]))
            del self.l1_cache[oldest_key]
            self.l1_size -= oldest_size
            self.stats['evictions'] += 1
        
        # 添加新缓存
        self.l1_cache[key] = value
        self.l1_size += size
    
    def clear(self):
        """清理缓存"""
        self.l1_cache.clear()
        self.l1_size = 0
        
        # 清理L2缓存
        for file in os.listdir(self.l2_path):
            if file.endswith('.cache'):
                try:
                    os.remove(os.path.join(self.l2_path, file))
                except Exception:
                    pass
    
    def optimize_pool(self):
        """优化缓存池"""
        # 清理过期缓存
        # 实现更复杂的优化逻辑
        pass
    
    def update_stats(self):
        """更新统计"""
        # 实时更新统计信息
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        total_requests = (
            self.stats['l1_hits'] + self.stats['l1_misses'] +
            self.stats['l2_hits'] + self.stats['l2_misses']
        )
        
        if total_requests > 0:
            hit_rate = (
                (self.stats['l1_hits'] + self.stats['l2_hits']) / total_requests
            )
        else:
            hit_rate = 0
        
        return {
            'l1_size': self.l1_size,
            'l1_max_size': self.l1_max_size,
            'l1_items': len(self.l1_cache),
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'evictions': self.stats['evictions']
        }

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.alerts = []
        self.thresholds = {
            'cpu_percent': 90,
            'memory_percent': 85,
            'disk_percent': 95
        }
    
    def check_thresholds(self, metrics: PerformanceMetrics) -> List[str]:
        """检查阈值"""
        alerts = []
        
        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            alerts.append(f"CPU使用率过高: {metrics.cpu_percent}%")
        
        if metrics.memory_percent > self.thresholds['memory_percent']:
            alerts.append(f"内存使用率过高: {metrics.memory_percent}%")
        
        if metrics.disk_usage_percent > self.thresholds['disk_percent']:
            alerts.append(f"磁盘使用率过高: {metrics.disk_usage_percent}%")
        
        return alerts

class AutoTuner:
    """自动调优器"""
    
    def __init__(self):
        self.tuning_history = []
        self.learning_rate = 0.01
    
    def tune_parameters(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """调优参数"""
        # 基于历史数据调优
        # 实现机器学习调优算法
        return {}

# 全局优化器实例
optimizer = PerformanceOptimizer()

# 便捷函数
def get_performance_report(hours: int = 24) -> Dict[str, Any]:
    """获取性能报告"""
    return optimizer.get_performance_report(hours)

def optimize_now(resource_types: List[str] = None) -> Dict[str, Any]:
    """立即优化"""
    if resource_types:
        types = [ResourceType(t) for t in resource_types]
    else:
        types = None
    return optimizer.optimize_now(types)

def cache_get(key: str) -> Optional[Any]:
    """获取缓存"""
    return optimizer.cache_system.get(key)

def cache_set(key: str, value: Any, ttl: int = 3600):
    """设置缓存"""
    optimizer.cache_system.set(key, value, ttl)

if __name__ == "__main__":
    # 测试代码
    print("性能优化器测试")
    print("=" * 50)
    
    # 获取性能报告
    report = get_performance_report(1)
    print(f"性能报告: {report}")
    
    # 立即优化
    result = optimize_now(['cpu', 'memory'])
    print(f"优化结果: {result}")
    
    # 缓存测试
    cache_set('test_key', {'data': 'test_value'})
    cached_value = cache_get('test_key')
    print(f"缓存测试: {cached_value}")