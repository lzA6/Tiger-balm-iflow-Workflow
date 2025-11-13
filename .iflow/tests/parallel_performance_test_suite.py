#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 并行执行性能测试套件
专门测试多智能体并行处理的加速比和效率，验证V7升级版的性能提升效果。
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import asyncio
import logging
import time
import unittest
import statistics
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import threading
import psutil
import gc
from dataclasses import dataclass, field

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from iflow.core.parallel_agent_executor import ParallelAgentExecutor, AgentRole
from iflow.core.task_decomposer import TaskDecomposer
from iflow.core.workflow_stage_parallelizer import WorkflowStageParallelizer, WorkflowStage, WorkflowStageInfo
from iflow.core.optimized_fusion_cache import OptimizedFusionCache

logger = logging.getLogger(__name__)

@dataclass
class PerformanceTestResult:
    """性能测试结果"""
    test_name: str
    test_type: str
    start_time: float
    end_time: float
    execution_time: float
    success: bool
    error: Optional[str] = None
    
    # 并行性能指标
    serial_baseline: Optional[float] = None
    parallel_time: Optional[float] = None
    speedup_ratio: Optional[float] = None
    efficiency: Optional[float] = None
    throughput: Optional[float] = None
    
    # 资源使用指标
    cpu_usage_avg: Optional[float] = None
    memory_usage_avg: Optional[float] = None
    memory_peak: Optional[float] = None
    resource_utilization: Optional[Dict[str, Any]] = None
    
    # 质量指标
    quality_score: Optional[float] = None
    accuracy: Optional[float] = None
    consistency_score: Optional[float] = None
    
    # 测试配置
    test_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    additional_metrics: Optional[Dict[str, Any]] = field(default_factory=dict)

class SystemResourceMonitor:
    """系统资源监控器"""
    
    def __init__(self):
        self.monitoring = False
        self.resource_data = []
        self.monitor_task = None
        self.process = psutil.Process()
    
    async def _monitor_resources(self):
        """监控系统资源"""
        while self.monitoring:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                # 内存使用
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # 系统整体内存
                system_memory = psutil.virtual_memory().percent
                
                self.resource_data.append({
                    "timestamp": time.time(),
                    "cpu_percent": cpu_percent,
                    "memory_mb": memory_mb,
                    "system_memory_percent": system_memory,
                    "num_threads": threading.active_count()
                })
                
                await asyncio.sleep(0.1)  # 每100ms采样一次
                
            except Exception as e:
                logger.error(f"资源监控错误: {e}")
                break
    
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.resource_data = []
        self.monitor_task = asyncio.create_task(self._monitor_resources())
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                asyncio.run_until_complete(self.monitor_task)
            except:
                pass
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """获取资源使用摘要"""
        if not self.resource_data:
            return {}
        
        cpu_usage = [d["cpu_percent"] for d in self.resource_data]
        memory_usage = [d["memory_mb"] for d in self.resource_data]
        system_memory = [d["system_memory_percent"] for d in self.resource_data]
        
        return {
            "avg_cpu_usage": statistics.mean(cpu_usage),
            "max_cpu_usage": max(cpu_usage),
            "avg_memory_usage_mb": statistics.mean(memory_usage),
            "peak_memory_usage_mb": max(memory_usage),
            "avg_system_memory_percent": statistics.mean(system_memory),
            "max_system_memory_percent": max(system_memory),
            "monitoring_duration": len(self.resource_data) * 0.1,
            "total_samples": len(self.resource_data)
        }

class ParallelPerformanceTestSuite:
    """并行执行性能测试套件"""
    
    def __init__(self):
        self.test_results = []
        self.resource_monitor = SystemResourceMonitor()
        
        # 测试配置
        self.test_config = {
            "max_concurrent_levels": [1, 2, 4, 8, 16, 32],
            "test_iterations": 5,
            "warmup_iterations": 2,
            "timeout_seconds": 300,
            "task_complexity_levels": ["simple", "moderate", "complex", "expert"]
        }
        
        # 性能基准
        self.performance_baseline = {}
        
        logger.info("并行执行性能测试套件初始化完成")
    
    async def run_comprehensive_performance_tests(self) -> Dict[str, Any]:
        """运行全面的性能测试"""
        print("=" * 90)
        print("🚀 A项目并行执行性能测试套件 V7")
        print("=" * 90)
        
        try:
            # 1. 并行加速比测试
            await self._test_parallel_speedup()
            
            # 2. 可扩展性测试
            await self._test_scalability()
            
            # 3. 吞吐量测试
            await self._test_throughput()
            
            # 4. 资源效率测试
            await self._test_resource_efficiency()
            
            # 5. 负载均衡测试
            await self._test_load_balancing()
            
            # 6. 缓存性能影响测试
            await self._test_cache_performance_impact()
            
        except Exception as e:
            logger.error(f"性能测试执行失败: {e}")
            raise
        
        # 生成性能报告
        return self._generate_performance_report()
    
    async def _test_parallel_speedup(self):
        """测试并行加速比"""
        print("\n📊 测试并行加速比...")
        
        for complexity in self.test_config["task_complexity_levels"]:
            print(f"\n   📈 测试复杂度: {complexity}")
            
            for concurrency in self.test_config["max_concurrent_levels"]:
                if concurrency == 1:
                    # 基准测试（串行）
                    baseline_result = await self._run_serial_baseline_test(complexity)
                    self.test_results.append(baseline_result)
                else:
                    # 并行测试
                    parallel_result = await self._run_parallel_test(complexity, concurrency)
                    self.test_results.append(parallel_result)
                    
                    # 计算加速比
                    if baseline_result.execution_time > 0:
                        speedup = baseline_result.execution_time / parallel_result.execution_time
                        efficiency = speedup / concurrency
                        
                        parallel_result.speedup_ratio = speedup
                        parallel_result.efficiency = efficiency
                        parallel_result.serial_baseline = baseline_result.execution_time
                        
                        print(f"      🔧 {concurrency}并发: 加速比={speedup:.2f}x, 效率={efficiency:.2%}")
    
    async def _run_serial_baseline_test(self, complexity: str) -> PerformanceTestResult:
        """运行串行基准测试"""
        test_name = f"串行基准测试-{complexity}"
        result = PerformanceTestResult(
            test_name=test_name,
            test_type="serial_baseline",
            start_time=time.time(),
            test_config={"complexity": complexity, "concurrency": 1}
        )
        
        try:
            # 创建单智能体执行器
            executor = ParallelAgentExecutor(max_concurrent_agents=1, enable_cache=False)
            
            # 创建串行任务
            subtasks = self._generate_test_subtasks(complexity, 5)
            
            # 执行任务
            self.resource_monitor.start_monitoring()
            
            serial_result = await executor.execute_parallel_task(
                task_description=f"串行基准测试-{complexity}",
                expert_assignments={"专家1": AgentRole.SPECIALIST},
                subtasks=[subtasks[0]]  # 只用第一个任务
            )
            
            self.resource_monitor.stop_monitoring()
            
            result.end_time = time.time()
            result.execution_time = result.end_time - result.start_time
            result.success = serial_result.success
            result.resource_utilization = serial_result.resource_usage
            result.resource_utilization.update(self.resource_monitor.get_resource_summary())
            result.quality_score = serial_result.quality_score
            
        except Exception as e:
            result.error = str(e)
        
        return result
    
    async def _run_parallel_test(self, complexity: str, concurrency: int) -> PerformanceTestResult:
        """运行并行测试"""
        test_name = f"并行测试-{complexity}-{concurrency}并发"
        result = PerformanceTestResult(
            test_name=test_name,
            test_type="parallel",
            start_time=time.time(),
            test_config={"complexity": complexity, "concurrency": concurrency}
        )
        
        try:
            # 创建并行执行器
            executor = ParallelAgentExecutor(max_concurrent_agents=concurrency, enable_cache=False)
            
            # 创建并行任务
            subtasks = self._generate_test_subtasks(complexity, concurrency * 2)
            
            # 分配专家
            expert_assignments = {f"专家{i}": AgentRole.SPECIALIST for i in range(concurrency)}
            
            # 转换子任务格式
            executor_subtasks = []
            for i, subtask in enumerate(subtasks[:concurrency * 2]):
                executor_subtasks.append({
                    "description": subtask.subtask_description,
                    "preferred_agent": f"专家{i % concurrency}",
                    "role": "SPECIALIST",
                    "priority": subtask.priority,
                    "dependencies": [],
                    "estimated_duration": subtask.estimated_duration * 0.5  # 减少单个任务时间
                })
            
            # 执行并行任务
            self.resource_monitor.start_monitoring()
            
            parallel_result = await executor.execute_parallel_task(
                task_description=f"并行测试-{complexity}-{concurrency}并发",
                expert_assignments=expert_assignments,
                subtasks=executor_subtasks
            )
            
            self.resource_monitor.stop_monitoring()
            
            result.end_time = time.time()
            result.execution_time = result.end_time - result.start_time
            result.success = parallel_result.success
            result.resource_utilization = parallel_result.resource_usage
            result.resource_utilization.update(self.resource_monitor.get_resource_summary())
            result.quality_score = parallel_result.quality_score
            
        except Exception as e:
            result.error = str(e)
        
        return result
    
    async def _test_scalability(self):
        """测试可扩展性"""
        print("\n📈 测试可扩展性...")
        
        scalability_results = []
        
        for concurrency in self.test_config["max_concurrent_levels"]:
            if concurrency == 1:
                continue  # 跳过串行测试
            
            # 运行可扩展性测试
            test_name = f"可扩展性测试-{concurrency}并发"
            result = PerformanceTestResult(
                test_name=test_name,
                test_type="scalability",
                start_time=time.time(),
                test_config={"concurrency": concurrency}
            )
            
            try:
                # 创建大量任务测试可扩展性
                executor = ParallelAgentExecutor(max_concurrent_agents=concurrency, enable_cache=False)
                
                # 生成大量子任务
                subtasks = self._generate_test_subtasks("complex", concurrency * 5)
                
                # 分配专家
                expert_assignments = {f"专家{i}": AgentRole.SPECIALIST for i in range(min(concurrency, 16))}
                
                # 转换任务格式
                executor_subtasks = []
                for i, subtask in enumerate(subtasks[:concurrency * 3]):
                    executor_subtasks.append({
                        "description": f"可扩展性任务{i}: {subtask.subtask_description}",
                        "preferred_agent": f"专家{i % len(expert_assignments)}",
                        "role": "SPECIALIST",
                        "priority": subtask.priority,
                        "dependencies": [],
                        "estimated_duration": subtask.estimated_duration * 0.3
                    })
                
                # 执行测试
                start_time = time.time()
                scalability_result = await executor.execute_parallel_task(
                    task_description=f"可扩展性测试-{concurrency}并发",
                    expert_assignments=expert_assignments,
                    subtasks=executor_subtasks
                )
                execution_time = time.time() - start_time
                
                # 计算吞吐量
                throughput = len(executor_subtasks) / execution_time if execution_time > 0 else 0
                
                result.end_time = time.time()
                result.execution_time = execution_time
                result.success = scalability_result.success
                result.throughput = throughput
                result.resource_utilization = scalability_result.resource_usage
                result.test_config.update({
                    "tasks_count": len(executor_subtasks),
                    "agents_count": len(expert_assignments)
                })
                
                scalability_results.append(result)
                
                print(f"      📊 {concurrency}并发: 吞吐量={throughput:.2f}任务/秒, 成功率={scalability_result.success}")
                
            except Exception as e:
                result.error = str(e)
                scalability_results.append(result)
        
        self.test_results.extend(scalability_results)
    
    async def _test_throughput(self):
        """测试吞吐量"""
        print("\n🔄 测试吞吐量...")
        
        # 高负载吞吐量测试
        test_name = "高负载吞吐量测试"
        result = PerformanceTestResult(
            test_name=test_name,
            test_type="throughput",
            start_time=time.time(),
            test_config={"load_type": "high", "duration": "60s"}
        )
        
        try:
            # 创建高并发测试
            executor = ParallelAgentExecutor(max_concurrent_agents=32, enable_cache=True)
            
            # 生成大量短任务
            subtasks = self._generate_test_subtasks("simple", 100)
            
            expert_assignments = {f"专家{i}": AgentRole.SPECIALIST for i in range(16)}
            
            executor_subtasks = []
            for i, subtask in enumerate(subtasks[:80]):
                executor_subtasks.append({
                    "description": f"吞吐量任务{i}: {subtask.subtask_description}",
                    "preferred_agent": f"专家{i % 16}",
                    "role": "SPECIALIST",
                    "priority": 1,
                    "dependencies": [],
                    "estimated_duration": 0.1  # 很短的任务
                })
            
            # 执行高吞吐量测试
            start_time = time.time()
            throughput_result = await executor.execute_parallel_task(
                task_description="高负载吞吐量测试",
                expert_assignments=expert_assignments,
                subtasks=executor_subtasks
            )
            execution_time = time.time() - start_time
            
            # 计算吞吐量指标
            total_tasks = len(executor_subtasks)
            completed_tasks = len([r for r in throughput_result.subtask_results.values() 
                                 if r.status.value == "completed"])
            throughput = total_tasks / execution_time if execution_time > 0 else 0
            completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
            
            result.end_time = time.time()
            result.execution_time = execution_time
            result.success = throughput_result.success
            result.throughput = throughput
            result.accuracy = completion_rate
            result.resource_utilization = throughput_result.resource_usage
            
            print(f"      ⚡ 吞吐量: {throughput:.2f}任务/秒, 完成率: {completion_rate:.2%}")
            
        except Exception as e:
            result.error = str(e)
        
        self.test_results.append(result)
    
    async def _test_resource_efficiency(self):
        """测试资源效率"""
        print("\n⚡ 测试资源效率...")
        
        efficiency_tests = []
        
        for concurrency in [4, 8, 16]:
            test_name = f"资源效率测试-{concurrency}并发"
            result = PerformanceTestResult(
                test_name=test_name,
                test_type="resource_efficiency",
                start_time=time.time(),
                test_config={"concurrency": concurrency}
            )
            
            try:
                # 创建资源效率测试
                executor = ParallelAgentExecutor(max_concurrent_agents=concurrency, enable_cache=False)
                
                # 创建中等复杂度任务
                subtasks = self._generate_test_subtasks("moderate", concurrency * 3)
                
                expert_assignments = {f"专家{i}": AgentRole.SPECIALIST for i in range(concurrency)}
                
                executor_subtasks = []
                for i, subtask in enumerate(subtasks[:concurrency * 2]):
                    executor_subtasks.append({
                        "description": f"效率任务{i}: {subtask.subtask_description}",
                        "preferred_agent": f"专家{i % concurrency}",
                        "role": "SPECIALIST",
                        "priority": subtask.priority,
                        "dependencies": [],
                        "estimated_duration": subtask.estimated_duration
                    })
                
                # 监控资源使用
                self.resource_monitor.start_monitoring()
                
                efficiency_result = await executor.execute_parallel_task(
                    task_description=f"资源效率测试-{concurrency}并发",
                    expert_assignments=expert_assignments,
                    subtasks=executor_subtasks
                )
                
                self.resource_monitor.stop_monitoring()
                
                resource_summary = self.resource_monitor.get_resource_summary()
                
                # 计算资源效率
                total_work = len(executor_subtasks)
                total_time = time.time() - result.start_time
                work_per_cpu_second = total_work / (resource_summary.get("avg_cpu_usage", 1) * total_time)
                memory_efficiency = total_work / resource_summary.get("peak_memory_usage_mb", 1)
                
                result.end_time = time.time()
                result.execution_time = total_time
                result.success = efficiency_result.success
                result.resource_utilization = resource_summary
                result.additional_metrics = {
                    "work_per_cpu_second": work_per_cpu_second,
                    "memory_efficiency": memory_efficiency,
                    "cpu_utilization": resource_summary.get("avg_cpu_usage"),
                    "memory_utilization": resource_summary.get("peak_memory_usage_mb")
                }
                
                efficiency_tests.append(result)
                
                print(f"      💡 {concurrency}并发: 工作效率={work_per_cpu_second:.2f}, 内存效率={memory_efficiency:.2f}")
                
            except Exception as e:
                result.error = str(e)
                efficiency_tests.append(result)
        
        self.test_results.extend(efficiency_tests)
    
    async def _test_load_balancing(self):
        """测试负载均衡"""
        print("\n⚖️ 测试负载均衡...")
        
        test_name = "负载均衡测试"
        result = PerformanceTestResult(
            test_name=test_name,
            test_type="load_balancing",
            start_time=time.time(),
            test_config={"agents": 8, "uneven_load": True}
        )
        
        try:
            # 创建负载不均衡的测试场景
            executor = ParallelAgentExecutor(max_concurrent_agents=8, enable_cache=False)
            
            # 创建不均匀的任务负载
            subtasks = []
            
            # 一些重任务
            for i in range(4):
                subtasks.append({
                    "description": f"重任务{i}: 复杂的数据处理",
                    "preferred_agent": f"专家{i}",
                    "role": "SPECIALIST",
                    "priority": 1,
                    "dependencies": [],
                    "estimated_duration": 2.0
                })
            
            # 一些轻任务
            for i in range(4, 12):
                subtasks.append({
                    "description": f"轻任务{i}: 简单的数据处理",
                    "preferred_agent": f"专家{i % 8}",
                    "role": "SPECIALIST",
                    "priority": 2,
                    "dependencies": [],
                    "estimated_duration": 0.2
                })
            
            expert_assignments = {f"专家{i}": AgentRole.SPECIALIST for i in range(8)}
            
            # 执行负载均衡测试
            start_time = time.time()
            balancing_result = await executor.execute_parallel_task(
                task_description="负载均衡测试",
                expert_assignments=expert_assignments,
                subtasks=subtasks
            )
            execution_time = time.time() - start_time
            
            # 分析负载分布
            agent_workload = {}
            for task_id, task_result in balancing_result.subtask_results.items():
                # 这里应该从实际执行结果中提取每个智能体的工作量
                # 简化实现
                agent_name = task_result.assigned_agent if hasattr(task_result, 'assigned_agent') else "unknown"
                if agent_name not in agent_workload:
                    agent_workload[agent_name] = 0
                agent_workload[agent_name] += 1
            
            # 计算负载均衡度
            if agent_workload:
                workload_values = list(agent_workload.values())
                workload_std = statistics.stdev(workload_values) if len(workload_values) > 1 else 0
                workload_mean = statistics.mean(workload_values)
                balance_score = 1 / (1 + workload_std / workload_mean) if workload_mean > 0 else 0
            else:
                balance_score = 0
            
            result.end_time = time.time()
            result.execution_time = execution_time
            result.success = balancing_result.success
            result.consistency_score = balance_score
            result.additional_metrics = {
                "agent_workload_distribution": agent_workload,
                "workload_balance_score": balance_score,
                "avg_execution_time": balancing_result.overall_duration if hasattr(balancing_result, 'overall_duration') else execution_time
            }
            
            print(f"      🎯 负载均衡度: {balance_score:.2f}")
            
        except Exception as e:
            result.error = str(e)
        
        self.test_results.append(result)
    
    async def _test_cache_performance_impact(self):
        """测试缓存对性能的影响"""
        print("\n💾 测试缓存性能影响...")
        
        # 对比有缓存和无缓存的性能
        cache_test_results = []
        
        for cache_enabled in [False, True]:
            test_name = f"缓存性能测试-{'启用' if cache_enabled else '禁用'}"
            result = PerformanceTestResult(
                test_name=test_name,
                test_type="cache_performance",
                start_time=time.time(),
                test_config={"cache_enabled": cache_enabled}
            )
            
            try:
                # 创建测试任务
                executor = ParallelAgentExecutor(max_concurrent_agents=8, enable_cache=cache_enabled)
                
                # 第一轮：冷启动
                subtasks1 = self._generate_test_subtasks("moderate", 10)
                executor_subtasks1 = []
                for i, subtask in enumerate(subtasks1[:8]):
                    executor_subtasks1.append({
                        "description": f"缓存测试任务{i}: {subtask.subtask_description}",
                        "preferred_agent": f"专家{i % 4}",
                        "role": "SPECIALIST",
                        "priority": subtask.priority,
                        "dependencies": [],
                        "estimated_duration": subtask.estimated_duration * 0.5
                    })
                
                start_time = time.time()
                result1 = await executor.execute_parallel_task(
                    task_description=f"缓存测试-第一轮",
                    expert_assignments={f"专家{i}": AgentRole.SPECIALIST for i in range(4)},
                    subtasks=executor_subtasks1
                )
                first_run_time = time.time() - start_time
                
                # 第二轮：可能命中缓存
                subtasks2 = self._generate_test_subtasks("moderate", 10)
                executor_subtasks2 = []
                for i, subtask in enumerate(subtasks2[:8]):
                    executor_subtasks2.append({
                        "description": f"缓存测试任务{i}: {subtask.subtask_description}",
                        "preferred_agent": f"专家{i % 4}",
                        "role": "SPECIALIST",
                        "priority": subtask.priority,
                        "dependencies": [],
                        "estimated_duration": subtask.estimated_duration * 0.5
                    })
                
                start_time = time.time()
                result2 = await executor.execute_parallel_task(
                    task_description=f"缓存测试-第二轮",
                    expert_assignments={f"专家{i}": AgentRole.SPECIALIST for i in range(4)},
                    subtasks=executor_subtasks2
                )
                second_run_time = time.time() - start_time
                
                # 计算缓存效果
                speedup_ratio = first_run_time / second_run_time if second_run_time > 0 else 1.0
                
                result.end_time = time.time()
                result.execution_time = first_run_time + second_run_time
                result.success = result1.success and result2.success
                result.speedup_ratio = speedup_ratio
                result.additional_metrics = {
                    "first_run_time": first_run_time,
                    "second_run_time": second_run_time,
                    "cache_speedup": speedup_ratio,
                    "cache_hit_rate": getattr(executor.cache, 'get_cache_statistics', lambda: {'cache_hit_rate': 0})()['cache_hit_rate'] if cache_enabled and executor.cache else 0
                }
                
                cache_test_results.append(result)
                
                print(f"      📊 缓存{'启用' if cache_enabled else '禁用'}: 加速比={speedup_ratio:.2f}x")
                
            except Exception as e:
                result.error = str(e)
                cache_test_results.append(result)
        
        self.test_results.extend(cache_test_results)
    
    def _generate_test_subtasks(self, complexity: str, count: int) -> List:
        """生成测试子任务"""
        decomposer = TaskDecomposer()
        
        # 创建测试任务描述
        complexity_tasks = {
            "simple": "简单的数据处理任务",
            "moderate": "中等复杂度的算法实现",
            "complex": "复杂的系统架构设计",
            "expert": "专家级别的性能优化任务"
        }
        
        task_description = f"{complexity_tasks.get(complexity, '测试任务')} (包含{count}个子任务)"
        
        # 分解任务
        subtasks = decomposer.decompose_task(
            original_task=task_description,
            domain="性能测试",
            max_subtasks=count
        )
        
        return subtasks
    
    def _generate_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        print("\n" + "=" * 90)
        print("📊 A项目并行执行性能测试报告")
        print("=" * 90)
        
        # 分析测试结果
        parallel_tests = [r for r in self.test_results if r.test_type == "parallel"]
        scalability_tests = [r for r in self.test_results if r.test_type == "scalability"]
        throughput_tests = [r for r in self.test_results if r.test_type == "throughput"]
        efficiency_tests = [r for r in self.test_results if r.test_type == "resource_efficiency"]
        
        # 计算关键指标
        performance_summary = {
            "parallel_performance": self._analyze_parallel_performance(parallel_tests),
            "scalability_analysis": self._analyze_scalability(scalability_tests),
            "throughput_analysis": self._analyze_throughput(throughput_tests),
            "efficiency_analysis": self._analyze_efficiency(efficiency_tests),
            "overall_assessment": self._generate_overall_assessment()
        }
        
        # 打印性能摘要
        print(f"\n📈 性能摘要:")
        print(f"   🚀 最高加速比: {performance_summary['parallel_performance']['max_speedup']:.2f}x")
        print(f"   ⚡ 最高吞吐量: {performance_summary['throughput_analysis']['max_throughput']:.2f}任务/秒")
        print(f"   💡 最佳效率: {performance_summary['efficiency_analysis']['max_efficiency']:.2%}")
        print(f"   📊 平均加速比: {performance_summary['parallel_performance']['avg_speedup']:.2f}x")
        
        # 打印建议
        recommendations = performance_summary['overall_assessment']['recommendations']
        if recommendations:
            print(f"\n💡 优化建议:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "=" * 90)
        print("✅ 并行执行性能测试完成")
        print("=" * 90)
        
        return performance_summary
    
    def _analyze_parallel_performance(self, parallel_tests: List[PerformanceTestResult]) -> Dict[str, Any]:
        """分析并行性能"""
        if not parallel_tests:
            return {"max_speedup": 0, "avg_speedup": 0, "max_efficiency": 0, "avg_efficiency": 0}
        
        speedups = [r.speedup_ratio for r in parallel_tests if r.speedup_ratio]
        efficiencies = [r.efficiency for r in parallel_tests if r.efficiency]
        
        return {
            "max_speedup": max(speedups) if speedups else 0,
            "avg_speedup": statistics.mean(speedups) if speedups else 0,
            "max_efficiency": max(efficiencies) if efficiencies else 0,
            "avg_efficiency": statistics.mean(efficiencies) if efficiencies else 0,
            "test_count": len(parallel_tests)
        }
    
    def _analyze_scalability(self, scalability_tests: List[PerformanceTestResult]) -> Dict[str, Any]:
        """分析可扩展性"""
        if not scalability_tests:
            return {"max_throughput": 0, "scalability_score": 0, "concurrency_levels": []}
        
        throughputs = [(r.test_config.get("concurrency", 0), r.throughput or 0) for r in scalability_tests]
        throughputs.sort()
        
        max_throughput = max([t[1] for t in throughputs]) if throughputs else 0
        
        # 计算可扩展性分数（基于吞吐量增长趋势）
        if len(throughputs) >= 2:
            # 简单的线性趋势分析
            x = [t[0] for t in throughputs]
            y = [t[1] for t in throughputs]
            if len(x) > 1:
                correlation = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
                scalability_score = max(0, correlation)  # 只取正相关
            else:
                scalability_score = 0
        else:
            scalability_score = 0
        
        return {
            "max_throughput": max_throughput,
            "scalability_score": scalability_score,
            "concurrency_levels": throughputs
        }
    
    def _analyze_throughput(self, throughput_tests: List[PerformanceTestResult]) -> Dict[str, Any]:
        """分析吞吐量"""
        if not throughput_tests:
            return {"max_throughput": 0, "avg_throughput": 0, "success_rate": 0}
        
        throughputs = [r.throughput for r in throughput_tests if r.throughput]
        success_rates = [1 if r.success else 0 for r in throughput_tests]
        
        return {
            "max_throughput": max(throughputs) if throughputs else 0,
            "avg_throughput": statistics.mean(throughputs) if throughputs else 0,
            "success_rate": statistics.mean(success_rates) if success_rates else 0
        }
    
    def _analyze_efficiency(self, efficiency_tests: List[PerformanceTestResult]) -> Dict[str, Any]:
        """分析资源效率"""
        if not efficiency_tests:
            return {"max_efficiency": 0, "avg_efficiency": 0, "resource_optimization": {}}
        
        efficiencies = []
        work_per_cpu = []
        memory_efficiency = []
        
        for test in efficiency_tests:
            if test.additional_metrics:
                if "work_per_cpu_second" in test.additional_metrics:
                    work_per_cpu.append(test.additional_metrics["work_per_cpu_second"])
                if "memory_efficiency" in test.additional_metrics:
                    memory_efficiency.append(test.additional_metrics["memory_efficiency"])
        
        avg_work_per_cpu = statistics.mean(work_per_cpu) if work_per_cpu else 0
        avg_memory_efficiency = statistics.mean(memory_efficiency) if memory_efficiency else 0
        
        return {
            "max_efficiency": 0,  # 需要具体计算
            "avg_efficiency": 0,
            "resource_optimization": {
                "avg_work_per_cpu": avg_work_per_cpu,
                "avg_memory_efficiency": avg_memory_efficiency
            }
        }
    
    def _generate_overall_assessment(self) -> Dict[str, Any]:
        """生成总体评估"""
        recommendations = []
        
        # 基于测试结果生成建议
        parallel_perf = self._analyze_parallel_performance([r for r in self.test_results if r.test_type == "parallel"])
        
        if parallel_perf["max_speedup"] < 2.0:
            recommendations.append("并行加速比偏低，建议优化任务分解策略")
        
        if parallel_perf["avg_efficiency"] < 0.5:
            recommendations.append("并行效率较低，建议优化负载均衡算法")
        
        scalability_perf = self._analyze_scalability([r for r in self.test_results if r.test_type == "scalability"])
        if scalability_perf["scalability_score"] < 0.7:
            recommendations.append("可扩展性不足，建议优化资源管理策略")
        
        if not recommendations:
            recommendations.append("性能表现优秀，系统运行良好！")
        
        return {
            "overall_score": min(10.0, max(1.0, parallel_perf["avg_speedup"] * 0.3 + 
                                          scalability_perf["scalability_score"] * 0.3 +
                                          parallel_perf["avg_efficiency"] * 0.4)),
            "recommendations": recommendations
        }

async def main():
    """主测试函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行性能测试
    performance_test_suite = ParallelPerformanceTestSuite()
    report = await performance_test_suite.run_comprehensive_performance_tests()
    
    # 保存性能报告
    report_file = PROJECT_ROOT / "iflow" / "tests" / "reports" / f"parallel_performance_report_{int(time.time())}.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n📄 性能测试报告已保存到: {report_file}")

if __name__ == "__main__":
    asyncio.run(main())