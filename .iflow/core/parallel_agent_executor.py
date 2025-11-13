#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 智能体并行执行引擎 V2
实现多个智能体同时处理任务的不同部分，大幅提升工作流执行效率。
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict, deque
import threading
import concurrent.futures
from contextlib import asynccontextmanager

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentRole(Enum):
    """智能体角色"""
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    REVIEWER = "reviewer"
    INTEGRATOR = "integrator"
    VALIDATOR = "validator"

@dataclass
class SubTask:
    """子任务"""
    task_id: str
    parent_task_id: str
    description: str
    assigned_agent: str
    agent_role: AgentRole
    priority: int
    dependencies: List[str]
    estimated_duration: float
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class ParallelExecutionResult:
    """并行执行结果"""
    task_id: str
    success: bool
    subtask_results: Dict[str, SubTask]
    aggregated_result: Any
    execution_time: float
    quality_score: float
    resource_usage: Dict[str, Any]
    coordination_overhead: float

class AgentResource:
    """智能体资源管理器"""
    
    def __init__(self, max_concurrent_agents: int = 10):
        self.max_concurrent_agents = max_concurrent_agents
        self.active_agents: Dict[str, SubTask] = {}
        self.agent_availability: Dict[str, bool] = {}
        self.agent_load: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
    
    def register_agent(self, agent_id: str, available: bool = True):
        """注册智能体"""
        with self._lock:
            self.agent_availability[agent_id] = available
            if agent_id not in self.agent_load:
                self.agent_load[agent_id] = 0
    
    def is_agent_available(self, agent_id: str) -> bool:
        """检查智能体是否可用"""
        with self._lock:
            return (self.agent_availability.get(agent_id, False) and
                    self.agent_load[agent_id] < 3 and  # 每个智能体最多处理3个任务
                    len(self.active_agents) < self.max_concurrent_agents)
    
    def assign_task(self, agent_id: str, subtask: SubTask):
        """分配任务给智能体"""
        with self._lock:
            self.active_agents[subtask.task_id] = subtask
            self.agent_load[agent_id] += 1
            subtask.status = TaskStatus.RUNNING
            subtask.start_time = time.time()
    
    def complete_task(self, task_id: str, result: Any = None, error: Optional[str] = None):
        """完成任务"""
        with self._lock:
            if task_id in self.active_agents:
                subtask = self.active_agents[task_id]
                subtask.result = result
                subtask.error = error
                subtask.end_time = time.time()
                subtask.status = TaskStatus.FAILED if error else TaskStatus.COMPLETED
                
                # 更新智能体负载
                if subtask.assigned_agent in self.agent_load:
                    self.agent_load[subtask.assigned_agent] -= 1
                
                # 移除活跃任务
                del self.active_agents[task_id]
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用情况"""
        with self._lock:
            return {
                "max_concurrent_agents": self.max_concurrent_agents,
                "active_tasks": len(self.active_agents),
                "available_agents": sum(1 for available in self.agent_availability.values() if available),
                "agent_load": dict(self.agent_load),
                "utilization_rate": len(self.active_agents) / self.max_concurrent_agents
            }

class TaskDependencyResolver:
    """任务依赖解析器"""
    
    def __init__(self):
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)
    
    def add_task(self, task_id: str, dependencies: List[str]):
        """添加任务及其依赖"""
        self.dependency_graph[task_id] = set(dependencies)
        for dep in dependencies:
            self.reverse_dependencies[dep].add(task_id)
    
    def get_ready_tasks(self, completed_tasks: Set[str]) -> Set[str]:
        """获取可以执行的任务"""
        ready_tasks = set()
        for task_id, deps in self.dependency_graph.items():
            if task_id not in completed_tasks and deps.issubset(completed_tasks):
                ready_tasks.add(task_id)
        return ready_tasks
    
    def get_task_level(self, task_id: str) -> int:
        """获取任务层级（用于优先级排序）"""
        if not self.dependency_graph[task_id]:
            return 0
        
        max_level = 0
        for dep in self.dependency_graph[task_id]:
            max_level = max(max_level, self.get_task_level(dep))
        return max_level + 1
    
    def has_cycle(self) -> bool:
        """检查是否存在循环依赖"""
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.reverse_dependencies[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in self.dependency_graph:
            if node not in visited:
                if dfs(node):
                    return True
        return False

class ParallelAgentExecutor:
    """
    智能体并行执行引擎
    """
    
    def __init__(self, max_concurrent_agents: int = 10, enable_cache: bool = True):
        self.executor_id = str(uuid.uuid4())
        self.max_concurrent_agents = max_concurrent_agents
        
        # 核心组件
        self.resource_manager = AgentResource(max_concurrent_agents)
        self.dependency_resolver = TaskDependencyResolver()
        
        # 缓存系统（如果启用）
        self.enable_cache = enable_cache
        self.cache = None
        if enable_cache:
            try:
                from .optimized_fusion_cache import OptimizedFusionCache
                self.cache = OptimizedFusionCache(cache_size=500, ttl_hours=12)
            except ImportError:
                logger.warning("无法导入缓存系统，将禁用缓存功能")
        
        # 执行状态
        self.running_tasks: Dict[str, SubTask] = {}
        self.completed_tasks: Dict[str, SubTask] = {}
        self.failed_tasks: Dict[str, SubTask] = {}
        
        # 统计信息
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "avg_execution_time": 0.0,
            "avg_parallel_efficiency": 0.0,
            "resource_utilization": 0.0
        }
        
        # 锁机制
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        
        logger.info(f"智能体并行执行引擎初始化完成 (ID: {self.executor_id})")
    
    async def execute_parallel_task(self, task_description: str, 
                                  expert_assignments: Dict[str, AgentRole],
                                  subtasks: List[Dict[str, Any]]) -> ParallelExecutionResult:
        """
        执行并行任务
        """
        task_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info(f"开始并行执行任务: {task_id}")
            
            # 1. 注册智能体
            for agent_id in expert_assignments:
                self.resource_manager.register_agent(agent_id, True)
            
            # 2. 创建子任务
            created_subtasks = await self._create_subtasks(task_id, subtasks, expert_assignments)
            
            # 3. 解析依赖关系
            await self._build_dependency_graph(created_subtasks)
            
            # 4. 执行并行任务
            execution_result = await self._execute_subtasks_parallel(created_subtasks)
            
            # 5. 聚合结果
            aggregated_result = await self._aggregate_results(execution_result)
            
            # 6. 计算性能指标
            execution_time = time.time() - start_time
            quality_score = self._calculate_quality_score(execution_result)
            resource_usage = self.resource_manager.get_resource_usage()
            coordination_overhead = self._calculate_coordination_overhead(execution_result)
            
            # 7. 更新统计
            self._update_statistics(True, execution_time, coordination_overhead)
            
            # 8. 缓存结果（如果启用）
            if self.cache:
                await self._cache_execution_result(
                    task_description, expert_assignments, aggregated_result, 
                    quality_score, execution_time
                )
            
            result = ParallelExecutionResult(
                task_id=task_id,
                success=True,
                subtask_results=execution_result,
                aggregated_result=aggregated_result,
                execution_time=execution_time,
                quality_score=quality_score,
                resource_usage=resource_usage,
                coordination_overhead=coordination_overhead
            )
            
            logger.info(f"并行任务执行完成: {task_id} (耗时: {execution_time:.2f}s, 质量: {quality_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"并行任务执行失败: {e}")
            self._update_statistics(False, time.time() - start_time, 0)
            return ParallelExecutionResult(
                task_id=task_id,
                success=False,
                subtask_results={},
                aggregated_result=None,
                execution_time=time.time() - start_time,
                quality_score=0.0,
                resource_usage=self.resource_manager.get_resource_usage(),
                coordination_overhead=0.0
            )
    
    async def _create_subtasks(self, parent_task_id: str, subtasks: List[Dict[str, Any]], 
                             expert_assignments: Dict[str, AgentRole]) -> Dict[str, SubTask]:
        """创建子任务"""
        created_subtasks = {}
        
        for i, subtask_config in enumerate(subtasks):
            subtask_id = f"{parent_task_id}_sub_{i}"
            
            # 分配智能体和角色
            assigned_agent = subtask_config.get("preferred_agent")
            if not assigned_agent or assigned_agent not in expert_assignments:
                # 选择可用的智能体
                available_agents = [agent for agent, role in expert_assignments.items() 
                                  if role == subtask_config.get("role", AgentRole.SPECIALIST)]
                assigned_agent = available_agents[0] if available_agents else f"agent_{i}"
            
            agent_role = expert_assignments.get(assigned_agent, AgentRole.SPECIALIST)
            
            subtask = SubTask(
                task_id=subtask_id,
                parent_task_id=parent_task_id,
                description=subtask_config["description"],
                assigned_agent=assigned_agent,
                agent_role=agent_role,
                priority=subtask_config.get("priority", 5),
                dependencies=subtask_config.get("dependencies", []),
                estimated_duration=subtask_config.get("estimated_duration", 1.0)
            )
            
            created_subtasks[subtask_id] = subtask
        
        return created_subtasks
    
    async def _build_dependency_graph(self, subtasks: Dict[str, SubTask]):
        """构建依赖关系图"""
        for subtask in subtasks.values():
            self.dependency_resolver.add_task(subtask.task_id, subtask.dependencies)
        
        if self.dependency_resolver.has_cycle():
            raise ValueError("检测到循环依赖，无法执行并行任务")
    
    async def _execute_subtasks_parallel(self, subtasks: Dict[str, SubTask]) -> Dict[str, SubTask]:
        """并行执行子任务"""
        completed_tasks = set()
        all_results = {}
        
        # 创建任务执行器
        async def execute_single_task(task_id: str, subtask: SubTask) -> Tuple[str, SubTask]:
            """执行单个子任务"""
            try:
                # 等待依赖完成
                while not set(subtask.dependencies).issubset(completed_tasks):
                    await asyncio.sleep(0.1)
                
                # 检查智能体可用性
                while not self.resource_manager.is_agent_available(subtask.assigned_agent):
                    await asyncio.sleep(0.1)
                
                # 分配任务
                self.resource_manager.assign_task(subtask.assigned_agent, subtask)
                
                # 模拟智能体执行（实际应该调用智能体API）
                await self._simulate_agent_execution(subtask)
                
                # 完成任务
                self.resource_manager.complete_task(task_id, subtask.result, subtask.error)
                completed_tasks.add(task_id)
                
                return task_id, subtask
                
            except Exception as e:
                subtask.error = str(e)
                subtask.status = TaskStatus.FAILED
                self.resource_manager.complete_task(task_id, None, str(e))
                return task_id, subtask
        
        # 并行执行所有可执行的任务
        while len(completed_tasks) < len(subtasks):
            ready_tasks = self.dependency_resolver.get_ready_tasks(completed_tasks)
            executable_tasks = [task_id for task_id in ready_tasks 
                              if self.resource_manager.is_agent_available(subtasks[task_id].assigned_agent)]
            
            if not executable_tasks:
                await asyncio.sleep(0.1)  # 等待资源释放
                continue
            
            # 并行执行可执行的任务
            tasks = [execute_single_task(task_id, subtasks[task_id]) for task_id in executable_tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"子任务执行异常: {result}")
                    continue
                task_id, subtask = result
                all_results[task_id] = subtask
        
        return all_results
    
    async def _simulate_agent_execution(self, subtask: SubTask):
        """模拟智能体执行（实际应该替换为真实的智能体调用）"""
        # 这里应该调用真实的智能体API
        # 目前使用模拟实现
        
        # 模拟执行时间
        await asyncio.sleep(min(subtask.estimated_duration, 2.0))
        
        # 模拟结果生成
        if subtask.agent_role == AgentRole.SPECIALIST:
            subtask.result = f"专家 {subtask.assigned_agent} 完成了任务: {subtask.description}"
        elif subtask.agent_role == AgentRole.REVIEWER:
            subtask.result = f"评审员 {subtask.assigned_agent} 审查了相关部分"
        elif subtask.agent_role == AgentRole.INTEGRATOR:
            subtask.result = f"集成师 {subtask.assigned_agent} 整合了各个部分"
        else:
            subtask.result = f"智能体 {subtask.assigned_agent} 处理了: {subtask.description}"
        
        subtask.status = TaskStatus.COMPLETED
    
    async def _aggregate_results(self, subtask_results: Dict[str, SubTask]) -> Any:
        """聚合子任务结果"""
        successful_results = []
        failed_results = []
        
        for subtask in subtask_results.values():
            if subtask.status == TaskStatus.COMPLETED and subtask.result:
                successful_results.append({
                    "agent": subtask.assigned_agent,
                    "role": subtask.agent_role.value,
                    "result": subtask.result,
                    "quality": 0.9  # 模拟质量评分
                })
            else:
                failed_results.append({
                    "agent": subtask.assigned_agent,
                    "error": subtask.error or "未知错误"
                })
        
        # 构建聚合结果
        aggregated_result = {
            "summary": "并行执行完成",
            "successful_agents": len(successful_results),
            "failed_agents": len(failed_results),
            "individual_results": successful_results,
            "failures": failed_results,
            "execution_summary": {
                "total_subtasks": len(subtask_results),
                "success_rate": len(successful_results) / len(subtask_results) if subtask_results else 0,
                "avg_execution_time": self._calculate_avg_execution_time(subtask_results)
            }
        }
        
        return aggregated_result
    
    def _calculate_quality_score(self, subtask_results: Dict[str, SubTask]) -> float:
        """计算质量评分"""
        completed_tasks = [t for t in subtask_results.values() if t.status == TaskStatus.COMPLETED]
        if not completed_tasks:
            return 0.0
        
        # 基于成功率和执行时间计算质量
        success_rate = len(completed_tasks) / len(subtask_results)
        avg_execution_time = self._calculate_avg_execution_time(subtask_results)
        
        # 质量评分公式
        time_penalty = min(avg_execution_time / 10.0, 0.5)  # 执行时间惩罚
        quality_score = success_rate * (1.0 - time_penalty)
        
        return max(0.0, min(1.0, quality_score))
    
    def _calculate_avg_execution_time(self, subtask_results: Dict[str, SubTask]) -> float:
        """计算平均执行时间"""
        completed_tasks = [t for t in subtask_results.values() 
                         if t.status == TaskStatus.COMPLETED and t.start_time and t.end_time]
        
        if not completed_tasks:
            return 0.0
        
        total_time = sum(t.end_time - t.start_time for t in completed_tasks)
        return total_time / len(completed_tasks)
    
    def _calculate_coordination_overhead(self, subtask_results: Dict[str, SubTask]) -> float:
        """计算协调开销"""
        total_tasks = len(subtask_results)
        completed_tasks = len([t for t in subtask_results.values() if t.status == TaskStatus.COMPLETED])
        
        # 协调开销 = (总时间 - 理想并行时间) / 总时间
        if total_tasks <= 1:
            return 0.0
        
        # 简化的协调开销计算
        coordination_overhead = (total_tasks - completed_tasks) / total_tasks
        return max(0.0, min(1.0, coordination_overhead))
    
    def _update_statistics(self, success: bool, execution_time: float, coordination_overhead: float):
        """更新统计信息"""
        with self._lock:
            self.stats["total_executions"] += 1
            if success:
                self.stats["successful_executions"] += 1
            else:
                self.stats["failed_executions"] += 1
            
            # 更新平均执行时间（指数移动平均）
            alpha = 0.1
            self.stats["avg_execution_time"] = (
                alpha * execution_time + 
                (1 - alpha) * self.stats["avg_execution_time"]
            )
            
            # 更新平均协调开销
            self.stats["avg_parallel_efficiency"] = (
                alpha * (1 - coordination_overhead) +
                (1 - alpha) * self.stats["avg_parallel_efficiency"]
            )
            
            # 更新资源利用率
            resource_usage = self.resource_manager.get_resource_usage()
            self.stats["resource_utilization"] = resource_usage.get("utilization_rate", 0.0)
    
    async def _cache_execution_result(self, task_description: str, expert_assignments: Dict[str, AgentRole],
                                    result: Any, quality_score: float, execution_time: float):
        """缓存执行结果"""
        if not self.cache:
            return
        
        try:
            # 构建缓存上下文
            context = {
                "expert_assignments": {k: v.value for k, v in expert_assignments.items()},
                "task_type": "parallel_execution"
            }
            
            # 存储到缓存
            self.cache.put_cache_result(
                task=task_description,
                context=context,
                selected_experts=list(expert_assignments.keys()),
                fusion_mode="parallel",
                result=result,
                quality_score=quality_score,
                execution_time=execution_time
            )
        except Exception as e:
            logger.warning(f"缓存执行结果失败: {e}")
    
    def get_executor_statistics(self) -> Dict[str, Any]:
        """获取执行器统计信息"""
        with self._lock:
            resource_usage = self.resource_manager.get_resource_usage()
            
            return {
                "executor_id": self.executor_id,
                "statistics": self.stats.copy(),
                "resource_usage": resource_usage,
                "cache_stats": self.cache.get_cache_statistics() if self.cache else None,
                "active_tasks": len(self.running_tasks),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks)
            }
    
    def stop(self):
        """停止执行器"""
        self._stop_event.set()
        logger.info("智能体并行执行引擎已停止")

# --- 使用示例 ---
async def main():
    """示例使用"""
    # 创建并行执行器
    executor = ParallelAgentExecutor(max_concurrent_agents=5, enable_cache=True)
    
    # 定义专家分配
    expert_assignments = {
        "架构师": AgentRole.SPECIALIST,
        "开发专家": AgentRole.SPECIALIST,
        "测试专家": AgentRole.VALIDATOR,
        "集成专家": AgentRole.INTEGRATOR
    }
    
    # 定义子任务
    subtasks = [
        {
            "description": "设计系统架构",
            "preferred_agent": "架构师",
            "role": AgentRole.SPECIALIST,
            "priority": 1,
            "dependencies": [],
            "estimated_duration": 2.0
        },
        {
            "description": "实现核心功能",
            "preferred_agent": "开发专家",
            "role": AgentRole.SPECIALIST,
            "priority": 2,
            "dependencies": ["sub_0"],  # 依赖架构设计
            "estimated_duration": 3.0
        },
        {
            "description": "编写测试用例",
            "preferred_agent": "测试专家",
            "role": AgentRole.VALIDATOR,
            "priority": 3,
            "dependencies": ["sub_1"],  # 依赖核心功能实现
            "estimated_duration": 1.5
        },
        {
            "description": "集成和部署",
            "preferred_agent": "集成专家",
            "role": AgentRole.INTEGRATOR,
            "priority": 4,
            "dependencies": ["sub_1", "sub_2"],  # 依赖功能实现和测试
            "estimated_duration": 2.0
        }
    ]
    
    # 执行并行任务
    result = await executor.execute_parallel_task(
        task_description="开发一个高性能的电商系统",
        expert_assignments=expert_assignments,
        subtasks=subtasks
    )
    
    print(f"执行结果: {result.success}")
    print(f"执行时间: {result.execution_time:.2f}s")
    print(f"质量评分: {result.quality_score:.2f}")
    print(f"资源使用: {result.resource_usage}")
    
    # 获取统计信息
    stats = executor.get_executor_statistics()
    print(f"\n执行器统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    asyncio.run(main())