#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全能工作流V5 - 多智能体协同编排系统
Universal Workflow V5 - Multi-Agent Orchestrator

基于Swarm Intelligence的智能体协同工作流编排
"""

import os
import sys
import json
import yaml
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future
import uuid
import heapq
from collections import defaultdict, deque

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class AgentState(Enum):
    """智能体状态"""
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    FAILED = "failed"
    COMPLETED = "completed"

class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class AgentCapability:
    """智能体能力"""
    name: str
    description: str
    tools: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 1
    specialties: List[str] = field(default_factory=list)

@dataclass
class Agent:
    """智能体"""
    id: str
    name: str
    type: str
    capabilities: List[AgentCapability]
    state: AgentState = AgentState.IDLE
    current_tasks: List[str] = field(default_factory=list)
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Task:
    """任务"""
    id: str
    name: str
    description: str
    required_capabilities: List[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowStep:
    """工作流步骤"""
    id: str
    name: str
    tasks: List[str]
    parallel: bool = False
    condition: Optional[str] = None
    on_success: Optional[str] = None
    on_failure: Optional[str] = None

@dataclass
class Workflow:
    """工作流"""
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    variables: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    current_step: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class SwarmIntelligence:
    """群体智能算法"""
    
    def __init__(self):
        self.pheromone_trails = defaultdict(float)  # 信息素轨迹
        self.agent_performance = defaultdict(float)  # 智能体性能
        self.task_success_rate = defaultdict(float)  # 任务成功率
        
    def update_pheromone(self, task_id: str, agent_id: str, success: bool, execution_time: float):
        """更新信息素"""
        # 成功任务增加信息素
        if success:
            pheromone = 1.0 / (1.0 + execution_time / 60.0)  # 基于执行时间
            self.pheromone_trails[(task_id, agent_id)] += pheromone
        
        # 更新智能体性能
        performance = 1.0 if success else -0.5
        self.agent_performance[agent_id] += performance * 0.1
        
        # 更新任务成功率
        self.task_success_rate[task_id] += (1.0 if success else 0.0) * 0.1
    
    def get_pheromone(self, task_id: str, agent_id: str) -> float:
        """获取信息素浓度"""
        return self.pheromone_trails.get((task_id, agent_id), 0.1)  # 默认值
    
    def select_best_agent(self, task: Task, available_agents: List[Agent]) -> Optional[Agent]:
        """选择最佳智能体"""
        if not available_agents:
            return None
        
        # 计算每个智能体的适应度
        fitness_scores = []
        for agent in available_agents:
            score = 0.0
            
            # 能力匹配度
            capability_match = sum(1 for cap in task.required_capabilities 
                                 if cap in [c.name for c in agent.capabilities])
            score += capability_match * 10.0
            
            # 信息素浓度
            pheromone = self.get_pheromone(task.id, agent.id)
            score += pheromone * 5.0
            
            # 智能体性能
            performance = self.agent_performance.get(agent.id, 0.0)
            score += performance * 3.0
            
            # 任务成功率
            success_rate = self.task_success_rate.get(task.id, 0.5)
            score += success_rate * 2.0
            
            # 负载均衡
            load_penalty = len(agent.current_tasks) * 2.0
            score -= load_penalty
            
            fitness_scores.append((agent, score))
        
        # 选择适应度最高的智能体
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        return fitness_scores[0][0] if fitness_scores else None

class MultiAgentOrchestrator:
    """多智能体编排器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.workflows: Dict[str, Workflow] = {}
        self.task_queue: List[Tuple[int, str]] = []  # (priority, task_id) 优先级队列
        self.swarm_intelligence = SwarmIntelligence()
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=self.config.get("max_workers", 10))
        self.running_tasks: Dict[str, Future] = {}
        
        # 统计信息
        self.statistics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_workflows": 0,
            "completed_workflows": 0,
            "failed_workflows": 0,
            "average_execution_time": 0.0
        }
        
        # 注册内置智能体
        self._register_builtin_agents()
        
        logger.info("多智能体编排器初始化完成")
    
    def _register_builtin_agents(self):
        """注册内置智能体"""
        # 全能工程师
        self.register_agent(Agent(
            id="universal-engineer",
            name="全能工程师",
            type="general",
            capabilities=[
                AgentCapability(
                    name="project-analysis",
                    description="项目分析能力",
                    tools=["analyze_project", "list_directory", "read_file"],
                    max_concurrent_tasks=3
                ),
                AgentCapability(
                    name="code-generation",
                    description="代码生成能力",
                    tools=["write_file", "execute_command"],
                    max_concurrent_tasks=2
                ),
                AgentCapability(
                    name="architecture-design",
                    description="架构设计能力",
                    tools=["analyze_project", "write_file"],
                    max_concurrent_tasks=1
                )
            ]
        ))
        
        # 前端架构师
        self.register_agent(Agent(
            id="frontend-architect",
            name="前端架构师",
            type="frontend",
            capabilities=[
                AgentCapability(
                    name="react-development",
                    description="React开发能力",
                    tools=["write_file", "execute_command"],
                    max_concurrent_tasks=2
                ),
                AgentCapability(
                    name="vue-development",
                    description="Vue开发能力",
                    tools=["write_file", "execute_command"],
                    max_concurrent_tasks=2
                ),
                AgentCapability(
                    name="ui-design",
                    description="UI设计能力",
                    tools=["write_file"],
                    max_concurrent_tasks=1
                )
            ]
        ))
        
        # 后端架构师
        self.register_agent(Agent(
            id="backend-architect",
            name="后端架构师",
            type="backend",
            capabilities=[
                AgentCapability(
                    name="api-design",
                    description="API设计能力",
                    tools=["write_file", "execute_command"],
                    max_concurrent_tasks=2
                ),
                AgentCapability(
                    name="database-design",
                    description="数据库设计能力",
                    tools=["write_file", "execute_command"],
                    max_concurrent_tasks=1
                ),
                AgentCapability(
                    name="microservices",
                    description="微服务架构能力",
                    tools=["write_file", "execute_command"],
                    max_concurrent_tasks=1
                )
            ]
        ))
        
        # AI工程师
        self.register_agent(Agent(
            id="ai-engineer",
            name="AI工程师",
            type="ai",
            capabilities=[
                AgentCapability(
                    name="ml-model-development",
                    description="机器学习模型开发",
                    tools=["write_file", "execute_command"],
                    max_concurrent_tasks=1
                ),
                AgentCapability(
                    name="data-analysis",
                    description="数据分析能力",
                    tools=["read_file", "write_file", "execute_command"],
                    max_concurrent_tasks=2
                ),
                AgentCapability(
                    name="deep-learning",
                    description="深度学习能力",
                    tools=["write_file", "execute_command"],
                    max_concurrent_tasks=1
                )
            ]
        ))
        
        # 测试工程师
        self.register_agent(Agent(
            id="test-engineer",
            name="测试工程师",
            type="testing",
            capabilities=[
                AgentCapability(
                    name="unit-testing",
                    description="单元测试能力",
                    tools=["write_file", "execute_command"],
                    max_concurrent_tasks=3
                ),
                AgentCapability(
                    name="integration-testing",
                    description="集成测试能力",
                    tools=["write_file", "execute_command"],
                    max_concurrent_tasks=2
                ),
                AgentCapability(
                    name="performance-testing",
                    description="性能测试能力",
                    tools=["execute_command"],
                    max_concurrent_tasks=1
                )
            ]
        ))
        
        # DevOps工程师
        self.register_agent(Agent(
            id="devops-engineer",
            name="DevOps工程师",
            type="devops",
            capabilities=[
                AgentCapability(
                    name="ci-cd",
                    description="CI/CD能力",
                    tools=["write_file", "execute_command"],
                    max_concurrent_tasks=2
                ),
                AgentCapability(
                    name="deployment",
                    description="部署能力",
                    tools=["execute_command"],
                    max_concurrent_tasks=1
                ),
                AgentCapability(
                    name="monitoring",
                    description="监控能力",
                    tools=["execute_command"],
                    max_concurrent_tasks=2
                )
            ]
        ))
    
    def register_agent(self, agent: Agent):
        """注册智能体"""
        self.agents[agent.id] = agent
        logger.info(f"注册智能体: {agent.name} ({agent.id})")
    
    def create_task(self, name: str, description: str, required_capabilities: List[str] = None,
                   priority: TaskPriority = TaskPriority.MEDIUM, dependencies: List[str] = None,
                   timeout: int = None, metadata: Dict[str, Any] = None) -> str:
        """创建任务"""
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            name=name,
            description=description,
            required_capabilities=required_capabilities or [],
            priority=priority,
            dependencies=dependencies or [],
            timeout=timeout,
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        
        # 添加到优先级队列
        heapq.heappush(self.task_queue, (-priority.value, task_id))
        
        self.statistics["total_tasks"] += 1
        logger.info(f"创建任务: {name} ({task_id})")
        
        return task_id
    
    def create_workflow(self, name: str, description: str, steps: List[Dict[str, Any]],
                       variables: Dict[str, Any] = None) -> str:
        """创建工作流"""
        workflow_id = str(uuid.uuid4())
        
        workflow_steps = []
        for step_data in steps:
            step = WorkflowStep(
                id=str(uuid.uuid4()),
                name=step_data["name"],
                tasks=step_data["tasks"],
                parallel=step_data.get("parallel", False),
                condition=step_data.get("condition"),
                on_success=step_data.get("on_success"),
                on_failure=step_data.get("on_failure")
            )
            workflow_steps.append(step)
        
        workflow = Workflow(
            id=workflow_id,
            name=name,
            description=description,
            steps=workflow_steps,
            variables=variables or {}
        )
        
        self.workflows[workflow_id] = workflow
        self.statistics["total_workflows"] += 1
        logger.info(f"创建工作流: {name} ({workflow_id})")
        
        return workflow_id
    
    def execute_workflow(self, workflow_id: str) -> bool:
        """执行工作流"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            logger.error(f"工作流不存在: {workflow_id}")
            return False
        
        workflow.status = TaskStatus.RUNNING
        workflow.started_at = datetime.now()
        
        try:
            # 执行工作流步骤
            for i, step in enumerate(workflow.steps):
                workflow.current_step = i
                
                # 检查条件
                if step.condition and not self._evaluate_condition(step.condition, workflow.variables):
                    logger.info(f"步骤条件不满足，跳过: {step.name}")
                    continue
                
                # 执行步骤
                success = self._execute_workflow_step(step, workflow)
                
                if not success:
                    # 处理失败
                    if step.on_failure:
                        self._handle_step_failure(step, workflow)
                    else:
                        workflow.status = TaskStatus.FAILED
                        workflow.completed_at = datetime.now()
                        self.statistics["failed_workflows"] += 1
                        return False
                else:
                    # 处理成功
                    if step.on_success:
                        self._handle_step_success(step, workflow)
            
            # 工作流完成
            workflow.status = TaskStatus.COMPLETED
            workflow.completed_at = datetime.now()
            self.statistics["completed_workflows"] += 1
            logger.info(f"工作流执行完成: {workflow.name}")
            return True
            
        except Exception as e:
            logger.error(f"工作流执行失败: {e}")
            workflow.status = TaskStatus.FAILED
            workflow.completed_at = datetime.now()
            self.statistics["failed_workflows"] += 1
            return False
    
    def _execute_workflow_step(self, step: WorkflowStep, workflow: Workflow) -> bool:
        """执行工作流步骤"""
        logger.info(f"执行步骤: {step.name}")
        
        if step.parallel:
            # 并行执行任务
            futures = []
            for task_id in step.tasks:
                task = self.tasks.get(task_id)
                if task:
                    future = self.executor.submit(self.execute_task, task_id)
                    futures.append(future)
            
            # 等待所有任务完成
            results = []
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"并行任务执行失败: {e}")
                    results.append(False)
            
            # 检查是否所有任务都成功
            return all(results)
        else:
            # 顺序执行任务
            for task_id in step.tasks:
                success = self.execute_task(task_id)
                if not success:
                    return False
            return True
    
    def _evaluate_condition(self, condition: str, variables: Dict[str, Any]) -> bool:
        """评估条件"""
        try:
            # 简单的条件评估（实际应更安全）
            return eval(condition, {"__builtins__": {}}, variables)
        except Exception as e:
            logger.error(f"条件评估失败: {e}")
            return False
    
    def _handle_step_success(self, step: WorkflowStep, workflow: Workflow):
        """处理步骤成功"""
        logger.info(f"步骤成功: {step.name}")
        # 可以在这里添加成功后的处理逻辑
    
    def _handle_step_failure(self, step: WorkflowStep, workflow: Workflow):
        """处理步骤失败"""
        logger.error(f"步骤失败: {step.name}")
        # 可以在这里添加失败后的处理逻辑
    
    def execute_task(self, task_id: str) -> bool:
        """执行任务"""
        task = self.tasks.get(task_id)
        if not task:
            logger.error(f"任务不存在: {task_id}")
            return False
        
        # 检查依赖
        if not self._check_dependencies(task):
            logger.warning(f"任务依赖未满足: {task.name}")
            task.status = TaskStatus.WAITING
            return False
        
        # 选择智能体
        available_agents = self._get_available_agents(task)
        if not available_agents:
            logger.warning(f"没有可用的智能体执行任务: {task.name}")
            task.status = TaskStatus.WAITING
            return False
        
        best_agent = self.swarm_intelligence.select_best_agent(task, available_agents)
        if not best_agent:
            logger.error(f"无法找到合适的智能体: {task.name}")
            return False
        
        # 分配任务
        task.assigned_agent = best_agent.id
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        best_agent.state = AgentState.BUSY
        best_agent.current_tasks.append(task_id)
        
        # 执行任务
        start_time = datetime.now()
        try:
            logger.info(f"智能体 {best_agent.name} 开始执行任务: {task.name}")
            
            # 模拟任务执行（实际应调用智能体的执行方法）
            result = self._simulate_task_execution(task, best_agent)
            
            # 更新任务状态
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            task.progress = 1.0
            
            # 更新智能体状态
            best_agent.state = AgentState.IDLE
            best_agent.current_tasks.remove(task_id)
            best_agent.completed_tasks += 1
            
            # 更新统计信息
            execution_time = (datetime.now() - start_time).total_seconds()
            best_agent.total_execution_time += execution_time
            self.statistics["completed_tasks"] += 1
            
            # 更新群体智能
            self.swarm_intelligence.update_pheromone(task_id, best_agent.id, True, execution_time)
            
            logger.info(f"任务执行完成: {task.name}")
            return True
            
        except Exception as e:
            # 处理任务失败
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            best_agent.state = AgentState.IDLE
            best_agent.current_tasks.remove(task_id)
            best_agent.failed_tasks += 1
            
            self.statistics["failed_tasks"] += 1
            
            # 更新群体智能
            execution_time = (datetime.now() - start_time).total_seconds()
            self.swarm_intelligence.update_pheromone(task_id, best_agent.id, False, execution_time)
            
            logger.error(f"任务执行失败: {task.name} - {e}")
            
            # 重试逻辑
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                heapq.heappush(self.task_queue, (-task.priority.value, task_id))
                logger.info(f"任务重试: {task.name} (第{task.retry_count}次)")
            
            return False
    
    def _check_dependencies(self, task: Task) -> bool:
        """检查任务依赖"""
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
    
    def _get_available_agents(self, task: Task) -> List[Agent]:
        """获取可用的智能体"""
        available = []
        
        for agent in self.agents.values():
            # 检查智能体状态
            if agent.state != AgentState.IDLE:
                continue
            
            # 检查能力匹配
            agent_capabilities = [cap.name for cap in agent.capabilities]
            if not any(cap in agent_capabilities for cap in task.required_capabilities):
                continue
            
            # 检查并发限制
            if len(agent.current_tasks) >= max(cap.max_concurrent_tasks for cap in agent.capabilities):
                continue
            
            available.append(agent)
        
        return available
    
    def _simulate_task_execution(self, task: Task, agent: Agent) -> Any:
        """模拟任务执行（实际应调用真实的执行方法）"""
        import time
        import random
        
        # 模拟执行时间
        execution_time = random.uniform(1, 10)
        time.sleep(execution_time)
        
        # 模拟成功率
        if random.random() > 0.1:  # 90%成功率
            return f"任务 {task.name} 执行成功"
        else:
            raise Exception("模拟任务失败")
    
    def get_task_status(self, task_id: str) -> Optional[Task]:
        """获取任务状态"""
        return self.tasks.get(task_id)
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Workflow]:
        """获取工作流状态"""
        return self.workflows.get(workflow_id)
    
    def get_agent_status(self, agent_id: str) -> Optional[Agent]:
        """获取智能体状态"""
        return self.agents.get(agent_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 计算平均执行时间
        completed_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]
        if completed_tasks:
            avg_time = sum(
                (t.completed_at - t.started_at).total_seconds() 
                for t in completed_tasks 
                if t.started_at and t.completed_at
            ) / len(completed_tasks)
            self.statistics["average_execution_time"] = avg_time
        
        # 添加智能体统计
        agent_stats = {}
        for agent_id, agent in self.agents.items():
            agent_stats[agent_id] = {
                "name": agent.name,
                "state": agent.state.value,
                "current_tasks": len(agent.current_tasks),
                "completed_tasks": agent.completed_tasks,
                "failed_tasks": agent.failed_tasks,
                "total_execution_time": agent.total_execution_time,
                "performance": self.swarm_intelligence.agent_performance.get(agent_id, 0.0)
            }
        
        self.statistics["agents"] = agent_stats
        
        # 添加任务统计
        task_stats = {
            "pending": sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING),
            "running": sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING),
            "waiting": sum(1 for t in self.tasks.values() if t.status == TaskStatus.WAITING),
            "completed": sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
        }
        
        self.statistics["tasks"] = task_stats
        
        return self.statistics
    
    def shutdown(self):
        """关闭编排器"""
        logger.info("正在关闭多智能体编排器...")
        
        # 等待所有运行中的任务完成
        for future in list(self.running_tasks.values()):
            try:
                future.result(timeout=10)
            except Exception as e:
                logger.error(f"等待任务完成时出错: {e}")
        
        # 关闭执行器
        self.executor.shutdown(wait=True)
        
        logger.info("多智能体编排器已关闭")

# 全局编排器实例
_orchestrator = None

def get_orchestrator(config: Dict[str, Any] = None) -> MultiAgentOrchestrator:
    """获取编排器实例"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiAgentOrchestrator(config)
    return _orchestrator

if __name__ == "__main__":
    # 测试代码
    config = {
        "max_workers": 5
    }
    
    orchestrator = get_orchestrator(config)
    
    # 创建测试任务
    task1 = orchestrator.create_task(
        "分析项目结构",
        "分析当前项目的架构和技术栈",
        required_capabilities=["project-analysis"],
        priority=TaskPriority.HIGH
    )
    
    task2 = orchestrator.create_task(
        "生成前端代码",
        "生成React前端代码",
        required_capabilities=["react-development"],
        priority=TaskPriority.MEDIUM,
        dependencies=[task1]
    )
    
    task3 = orchestrator.create_task(
        "生成后端API",
        "生成RESTful API",
        required_capabilities=["api-design"],
        priority=TaskPriority.MEDIUM,
        dependencies=[task1]
    )
    
    # 创建工作流
    workflow_id = orchestrator.create_workflow(
        "全栈开发工作流",
        "从项目分析到前后端开发的完整流程",
        [
            {
                "name": "项目分析",
                "tasks": [task1],
                "parallel": False
            },
            {
                "name": "代码生成",
                "tasks": [task2, task3],
                "parallel": True
            }
        ]
    )
    
    # 执行工作流
    print("开始执行工作流...")
    success = orchestrator.execute_workflow(workflow_id)
    
    if success:
        print("工作流执行成功！")
    else:
        print("工作流执行失败！")
    
    # 显示统计信息
    print("\n统计信息:")
    stats = orchestrator.get_statistics()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    
    # 关闭编排器
    orchestrator.shutdown()
