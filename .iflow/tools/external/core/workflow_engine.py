#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全能工作流V5 - 智能工作流引擎
Intelligent Workflow Engine

作者: Universal AI Team
版本: 5.0.0
日期: 2025-11-12

特性:
- 智能任务路由和调度
- 并行执行和资源优化
- 动态工作流编排
- 实时监控和自适应
- 故障恢复和容错
- 性能分析和优化
"""

import os
import sys
import json
import time
import asyncio
import logging
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
import yaml
import uuid
import networkx as nx
import numpy as np

# 导入内部模块
from agent_registry import registry, AgentMetadata
from model_adapter import adapter, TaskType

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    RETRYING = "retrying"

class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5

class WorkflowStatus(Enum):
    """工作流状态"""
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

@dataclass
class Task:
    """任务定义"""
    id: str
    name: str
    description: str
    task_type: TaskType
    agent_id: Optional[str] = None
    priority: TaskPriority = TaskPriority.NORMAL
    dependencies: List[str] = field(default_factory=list)
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'task_type': self.task_type.value,
            'agent_id': self.agent_id,
            'priority': self.priority.value,
            'dependencies': self.dependencies,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'metadata': self.metadata,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'timeout': self.timeout
        }

@dataclass
class Workflow:
    """工作流定义"""
    id: str
    name: str
    description: str
    tasks: Dict[str, Task]
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'tasks': {tid: task.to_dict() for tid, task in self.tasks.items()},
            'metadata': self.metadata,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }

class WorkflowEngine:
    """工作流引擎"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            '.iflow', 'config', 'workflow-engine-config.yaml'
        )
        
        # 工作流存储
        self.workflows: Dict[str, Workflow] = {}
        self.active_workflows: Dict[str, Workflow] = {}
        
        # 任务队列
        self.task_queue = asyncio.PriorityQueue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # 依赖图
        self.dependency_graphs: Dict[str, nx.DiGraph] = {}
        
        # 资源管理
        self.resource_pool = ResourcePool()
        
        # 执行器
        self.task_executor = TaskExecutor()
        
        # 监控
        self.monitor = WorkflowMonitor()
        
        # 配置
        self.config = self._load_config()
        
        # 初始化
        self._initialize_engine()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {
            'max_concurrent_tasks': 10,
            'task_timeout': 300,
            'retry_delay': 5,
            'auto_retry': True,
            'parallel_execution': True,
            'smart_routing': True
        }
    
    def _initialize_engine(self):
        """初始化引擎"""
        # 创建必要的目录
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # 启动任务调度器
        asyncio.create_task(self._task_scheduler())
        
        # 启动监控
        asyncio.create_task(self._monitoring_loop())
        
        logger.info("工作流引擎初始化完成")
    
    async def create_workflow(self, 
                            name: str,
                            description: str,
                            tasks: List[Dict[str, Any]],
                            metadata: Dict[str, Any] = None) -> str:
        """创建工作流"""
        try:
            workflow_id = str(uuid.uuid4())
            
            # 创建任务
            task_dict = {}
            for task_data in tasks:
                task = Task(
                    id=task_data.get('id', str(uuid.uuid4())),
                    name=task_data['name'],
                    description=task_data.get('description', ''),
                    task_type=TaskType(task_data.get('task_type', 'reasoning')),
                    agent_id=task_data.get('agent_id'),
                    priority=TaskPriority(task_data.get('priority', 2)),
                    dependencies=task_data.get('dependencies', []),
                    input_data=task_data.get('input_data', {}),
                    metadata=task_data.get('metadata', {}),
                    max_retries=task_data.get('max_retries', 3),
                    timeout=task_data.get('timeout')
                )
                task_dict[task.id] = task
            
            # 创建工作流
            workflow = Workflow(
                id=workflow_id,
                name=name,
                description=description,
                tasks=task_dict,
                metadata=metadata or {}
            )
            
            # 验证工作流
            if not self._validate_workflow(workflow):
                raise ValueError("工作流验证失败")
            
            # 构建依赖图
            self._build_dependency_graph(workflow)
            
            # 保存工作流
            self.workflows[workflow_id] = workflow
            
            logger.info(f"工作流创建成功: {workflow_id}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"创建工作流失败: {e}")
            raise
    
    async def execute_workflow(self, workflow_id: str) -> bool:
        """执行工作流"""
        try:
            if workflow_id not in self.workflows:
                logger.error(f"工作流不存在: {workflow_id}")
                return False
            
            workflow = self.workflows[workflow_id]
            
            # 检查状态
            if workflow.status != WorkflowStatus.DRAFT:
                logger.warning(f"工作流状态不正确: {workflow.status}")
                return False
            
            # 更新状态
            workflow.status = WorkflowStatus.ACTIVE
            workflow.started_at = datetime.now()
            self.active_workflows[workflow_id] = workflow
            
            # 初始化任务队列
            await self._initialize_task_queue(workflow)
            
            logger.info(f"工作流开始执行: {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"执行工作流失败: {e}")
            if workflow_id in self.workflows:
                self.workflows[workflow_id].status = WorkflowStatus.FAILED
            return False
    
    async def pause_workflow(self, workflow_id: str) -> bool:
        """暂停工作流"""
        if workflow_id not in self.active_workflows:
            return False
        
        workflow = self.active_workflows[workflow_id]
        workflow.status = WorkflowStatus.PAUSED
        
        # 暂停运行中的任务
        for task_id, task in workflow.tasks.items():
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.PAUSED
                if task_id in self.running_tasks:
                    self.running_tasks[task_id].cancel()
        
        return True
    
    async def resume_workflow(self, workflow_id: str) -> bool:
        """恢复工作流"""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        if workflow.status != WorkflowStatus.PAUSED:
            return False
        
        workflow.status = WorkflowStatus.ACTIVE
        
        # 恢复暂停的任务
        for task_id, task in workflow.tasks.items():
            if task.status == TaskStatus.PAUSED:
                task.status = TaskStatus.PENDING
                await self._enqueue_task(workflow_id, task)
        
        return True
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """取消工作流"""
        if workflow_id not in self.active_workflows:
            return False
        
        workflow = self.active_workflows[workflow_id]
        workflow.status = WorkflowStatus.CANCELLED
        
        # 取消所有任务
        for task_id, task in workflow.tasks.items():
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.PAUSED]:
                task.status = TaskStatus.CANCELLED
                if task_id in self.running_tasks:
                    self.running_tasks[task_id].cancel()
        
        # 移除活跃工作流
        del self.active_workflows[workflow_id]
        
        return True
    
    def _validate_workflow(self, workflow: Workflow) -> bool:
        """验证工作流"""
        # 检查任务
        if not workflow.tasks:
            return False
        
        # 检查依赖
        for task_id, task in workflow.tasks.items():
            for dep_id in task.dependencies:
                if dep_id not in workflow.tasks:
                    logger.error(f"任务 {task_id} 的依赖 {dep_id} 不存在")
                    return False
        
        # 检查循环依赖
        if self._has_circular_dependencies(workflow):
            logger.error("工作流存在循环依赖")
            return False
        
        return True
    
    def _build_dependency_graph(self, workflow: Workflow):
        """构建依赖图"""
        graph = nx.DiGraph()
        
        # 添加节点
        for task_id in workflow.tasks:
            graph.add_node(task_id)
        
        # 添加边
        for task_id, task in workflow.tasks.items():
            for dep_id in task.dependencies:
                graph.add_edge(dep_id, task_id)
        
        self.dependency_graphs[workflow.id] = graph
    
    def _has_circular_dependencies(self, workflow: Workflow) -> bool:
        """检查循环依赖"""
        graph = nx.DiGraph()
        
        for task_id, task in workflow.tasks.items():
            for dep_id in task.dependencies:
                graph.add_edge(task_id, dep_id)
        
        try:
            nx.find_cycle(graph)
            return True
        except nx.NetworkXNoCycle:
            return False
    
    async def _initialize_task_queue(self, workflow: Workflow):
        """初始化任务队列"""
        # 找到没有依赖的任务
        ready_tasks = [
            task for task in workflow.tasks.values()
            if not task.dependencies and task.status == TaskStatus.PENDING
        ]
        
        # 加入队列
        for task in ready_tasks:
            await self._enqueue_task(workflow.id, task)
    
    async def _enqueue_task(self, workflow_id: str, task: Task):
        """将任务加入队列"""
        # 优先级队列使用负数实现高优先级
        priority = -task.priority.value
        await self.task_queue.put((priority, workflow_id, task.id))
    
    async def _task_scheduler(self):
        """任务调度器"""
        while True:
            try:
                # 获取任务
                priority, workflow_id, task_id = await self.task_queue.get()
                
                # 检查工作流状态
                if workflow_id not in self.active_workflows:
                    continue
                
                workflow = self.active_workflows[workflow_id]
                if workflow.status != WorkflowStatus.ACTIVE:
                    continue
                
                task = workflow.tasks[task_id]
                
                # 检查任务状态
                if task.status != TaskStatus.PENDING:
                    continue
                
                # 检查依赖
                if not self._check_dependencies_completed(workflow, task):
                    # 依赖未完成，重新入队
                    await asyncio.sleep(1)
                    await self._enqueue_task(workflow_id, task)
                    continue
                
                # 检查资源
                if not await self.resource_pool.acquire_resources(task):
                    # 资源不足，重新入队
                    await asyncio.sleep(0.5)
                    await self._enqueue_task(workflow_id, task)
                    continue
                
                # 执行任务
                asyncio_task = asyncio.create_task(
                    self._execute_task(workflow_id, task)
                )
                self.running_tasks[task_id] = asyncio_task
                
            except Exception as e:
                logger.error(f"任务调度错误: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task(self, workflow_id: str, task: Task):
        """执行任务"""
        try:
            # 更新任务状态
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            # 选择智能体
            agent_id = await self._select_agent(task)
            if agent_id:
                task.agent_id = agent_id
            
            # 执行任务
            result = await self.task_executor.execute(task)
            
            # 更新任务结果
            task.output_data = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            # 释放资源
            await self.resource_pool.release_resources(task)
            
            # 检查并激活后续任务
            await self._activate_dependent_tasks(workflow_id, task)
            
            # 检查工作流是否完成
            await self._check_workflow_completion(workflow_id)
            
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            await self.resource_pool.release_resources(task)
            
        except Exception as e:
            logger.error(f"任务执行失败 {task.id}: {e}")
            task.error_message = str(e)
            
            # 重试逻辑
            if task.retry_count < task.max_retries and self.config.get('auto_retry', True):
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                await asyncio.sleep(self.config.get('retry_delay', 5))
                await self._enqueue_task(workflow_id, task)
            else:
                task.status = TaskStatus.FAILED
                await self.resource_pool.release_resources(task)
                
                # 工作流失败
                await self._handle_workflow_failure(workflow_id, task)
        
        finally:
            # 移除运行任务
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]
    
    async def _select_agent(self, task: Task) -> Optional[str]:
        """选择智能体"""
        if not self.config.get('smart_routing', True):
            return None
        
        # 获取推荐智能体
        recommendations = registry.recommend_agents(task.description, limit=5)
        
        if recommendations:
            # 选择可用且评分最高的智能体
            for agent_id, score in recommendations:
                agent = registry.get_agent(agent_id)
                if agent and agent.status.value == 'active':
                    return agent_id
        
        return None
    
    def _check_dependencies_completed(self, workflow: Workflow, task: Task) -> bool:
        """检查依赖是否完成"""
        for dep_id in task.dependencies:
            dep_task = workflow.tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
    
    async def _activate_dependent_tasks(self, workflow_id: str, completed_task: Task):
        """激活依赖任务"""
        workflow = self.active_workflows[workflow_id]
        
        # 找到依赖此任务的任务
        for task_id, task in workflow.tasks.items():
            if (completed_task.id in task.dependencies and 
                task.status == TaskStatus.PENDING and
                self._check_dependencies_completed(workflow, task)):
                
                await self._enqueue_task(workflow_id, task)
    
    async def _check_workflow_completion(self, workflow_id: str):
        """检查工作流是否完成"""
        workflow = self.active_workflows[workflow_id]
        
        # 检查所有任务状态
        all_completed = all(
            task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
            for task in workflow.tasks.values()
        )
        
        if all_completed:
            # 检查是否有失败的任务
            has_failed = any(
                task.status == TaskStatus.FAILED
                for task in workflow.tasks.values()
            )
            
            workflow.status = WorkflowStatus.FAILED if has_failed else WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.now()
            
            # 移除活跃工作流
            del self.active_workflows[workflow_id]
            
            logger.info(f"工作流完成: {workflow_id}, 状态: {workflow.status}")
    
    async def _handle_workflow_failure(self, workflow_id: str, failed_task: Task):
        """处理工作流失败"""
        workflow = self.active_workflows[workflow_id]
        
        # 取消所有待执行的任务
        for task in workflow.tasks.values():
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
        
        # 更新工作流状态
        workflow.status = WorkflowStatus.FAILED
        workflow.completed_at = datetime.now()
        
        # 移除活跃工作流
        del self.active_workflows[workflow_id]
        
        logger.error(f"工作流失败: {workflow_id}, 失败任务: {failed_task.id}")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while True:
            try:
                # 更新监控数据
                await self.monitor.update_stats(self)
                
                # 检查超时任务
                await self._check_task_timeouts()
                
                await asyncio.sleep(10)  # 10秒监控一次
                
            except Exception as e:
                logger.error(f"监控错误: {e}")
                await asyncio.sleep(30)
    
    async def _check_task_timeouts(self):
        """检查任务超时"""
        current_time = datetime.now()
        
        for workflow_id, workflow in list(self.active_workflows.items()):
            for task_id, task in workflow.tasks.items():
                if (task.status == TaskStatus.RUNNING and 
                    task.timeout and 
                    task.started_at and
                    (current_time - task.started_at).seconds > task.timeout):
                    
                    logger.warning(f"任务超时: {task_id}")
                    
                    # 取消任务
                    if task_id in self.running_tasks:
                        self.running_tasks[task_id].cancel()
                    
                    task.status = TaskStatus.FAILED
                    task.error_message = "任务超时"
                    
                    await self._handle_workflow_failure(workflow_id, task)
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """获取工作流状态"""
        if workflow_id not in self.workflows:
            return None
        
        workflow = self.workflows[workflow_id]
        
        # 统计任务状态
        task_stats = defaultdict(int)
        for task in workflow.tasks.values():
            task_stats[task.status.value] += 1
        
        return {
            'workflow_id': workflow_id,
            'name': workflow.name,
            'status': workflow.status.value,
            'created_at': workflow.created_at.isoformat(),
            'started_at': workflow.started_at.isoformat() if workflow.started_at else None,
            'completed_at': workflow.completed_at.isoformat() if workflow.completed_at else None,
            'task_stats': dict(task_stats),
            'total_tasks': len(workflow.tasks)
        }
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎统计"""
        return {
            'total_workflows': len(self.workflows),
            'active_workflows': len(self.active_workflows),
            'running_tasks': len(self.running_tasks),
            'queued_tasks': self.task_queue.qsize(),
            'resource_usage': self.resource_pool.get_usage(),
            'monitoring_stats': self.monitor.get_stats()
        }

class ResourcePool:
    """资源池"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.current_usage = 0
        self.lock = asyncio.Lock()
    
    async def acquire_resources(self, task: Task) -> bool:
        """获取资源"""
        async with self.lock:
            if self.current_usage < self.max_concurrent:
                self.current_usage += 1
                return True
            return False
    
    async def release_resources(self, task: Task):
        """释放资源"""
        async with self.lock:
            self.current_usage = max(0, self.current_usage - 1)
    
    def get_usage(self) -> Dict[str, Any]:
        """获取使用情况"""
        return {
            'max_concurrent': self.max_concurrent,
            'current_usage': self.current_usage,
            'available': self.max_concurrent - self.current_usage,
            'utilization': self.current_usage / self.max_concurrent
        }

class TaskExecutor:
    """任务执行器"""
    
    async def execute(self, task: Task) -> Dict[str, Any]:
        """执行任务"""
        try:
            # 准备输入
            input_data = task.input_data.copy()
            input_data['task_description'] = task.description
            
            # 构建消息
            messages = [
                {'role': 'system', 'content': '你是一个专业的任务执行助手。'},
                {'role': 'user', 'content': json.dumps(input_data, ensure_ascii=False)}
            ]
            
            # 调用模型适配器
            response = await adapter.route_request(
                task.task_type,
                messages
            )
            
            # 解析响应
            result = {
                'content': response.get('content', ''),
                'model_used': response.get('model', ''),
                'tokens_used': response.get('usage', {}).get('total_tokens', 0),
                'execution_time': time.time()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"任务执行失败: {e}")
            raise

class WorkflowMonitor:
    """工作流监控器"""
    
    def __init__(self):
        self.stats = {
            'total_workflows': 0,
            'completed_workflows': 0,
            'failed_workflows': 0,
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_execution_time': 0,
            'last_updated': None
        }
        self.execution_times = deque(maxlen=1000)
    
    async def update_stats(self, engine: WorkflowEngine):
        """更新统计"""
        # 计算工作流统计
        total_workflows = len(engine.workflows)
        completed_workflows = sum(
            1 for w in engine.workflows.values()
            if w.status == WorkflowStatus.COMPLETED
        )
        failed_workflows = sum(
            1 for w in engine.workflows.values()
            if w.status == WorkflowStatus.FAILED
        )
        
        # 计算任务统计
        total_tasks = sum(len(w.tasks) for w in engine.workflows.values())
        completed_tasks = sum(
            sum(1 for t in w.tasks.values() if t.status == TaskStatus.COMPLETED)
            for w in engine.workflows.values()
        )
        failed_tasks = sum(
            sum(1 for t in w.tasks.values() if t.status == TaskStatus.FAILED)
            for w in engine.workflows.values()
        )
        
        # 更新统计
        self.stats.update({
            'total_workflows': total_workflows,
            'completed_workflows': completed_workflows,
            'failed_workflows': failed_workflows,
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'last_updated': datetime.now().isoformat()
        })
        
        # 计算平均执行时间
        if self.execution_times:
            self.stats['avg_execution_time'] = statistics.mean(self.execution_times)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return self.stats.copy()

# 全局引擎实例
engine = WorkflowEngine()

# 便捷函数
async def create_workflow(name: str, description: str, tasks: List[Dict]) -> str:
    """创建工作流"""
    return await engine.create_workflow(name, description, tasks)

async def execute_workflow(workflow_id: str) -> bool:
    """执行工作流"""
    return await engine.execute_workflow(workflow_id)

def get_workflow_status(workflow_id: str) -> Optional[Dict[str, Any]]:
    """获取工作流状态"""
    return engine.get_workflow_status(workflow_id)

def get_engine_stats() -> Dict[str, Any]:
    """获取引擎统计"""
    return engine.get_engine_stats()

if __name__ == "__main__":
    # 测试代码
    async def test_engine():
        print("工作流引擎测试")
        print("=" * 50)
        
        # 创建测试工作流
        tasks = [
            {
                'id': 'task1',
                'name': '分析需求',
                'description': '分析项目需求',
                'task_type': 'analysis',
                'priority': 3
            },
            {
                'id': 'task2',
                'name': '设计方案',
                'description': '设计技术方案',
                'task_type': 'reasoning',
                'priority': 2,
                'dependencies': ['task1']
            },
            {
                'id': 'task3',
                'name': '实现代码',
                'description': '编写实现代码',
                'task_type': 'coding',
                'priority': 2,
                'dependencies': ['task2']
            }
        ]
        
        workflow_id = await create_workflow(
            "测试工作流",
            "用于测试的工作流",
            tasks
        )
        
        print(f"创建工作流: {workflow_id}")
        
        # 执行工作流
        success = await execute_workflow(workflow_id)
        print(f"执行结果: {success}")
        
        # 等待执行完成
        await asyncio.sleep(5)
        
        # 获取状态
        status = get_workflow_status(workflow_id)
        print(f"工作流状态: {status}")
        
        # 获取引擎统计
        stats = get_engine_stats()
        print(f"引擎统计: {stats}")
    
    asyncio.run(test_engine())