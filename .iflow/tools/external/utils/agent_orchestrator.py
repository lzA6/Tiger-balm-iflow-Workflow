#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能体编排器 - 多智能体协作和工作流编排系统
Agent Orchestrator - Multi-Agent Collaboration and Workflow Orchestration System

作者: Quantum AI Team
版本: 5.3.0
日期: 2025-11-12
"""

import os
import sys
import json
import asyncio
import logging
import uuid
import time
from typing import Dict, List, Any, Optional, Union, Callable, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """智能体状态"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"

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
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class AgentCapability:
    """智能体能力"""
    agent_id: str
    agent_type: str
    skills: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 3
    expertise_level: float = 0.8
    reliability_score: float = 0.9
    cost_per_hour: float = 0.0
    specializations: List[str] = field(default_factory=list)

@dataclass
class Task:
    """任务定义"""
    task_id: str
    task_type: str
    title: str
    description: str
    priority: TaskPriority
    required_skills: List[str] = field(default_factory=list)
    estimated_duration: int = 300  # 秒
    dependencies: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    requirements: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None

@dataclass
class TaskExecution:
    """任务执行"""
    execution_id: str
    task: Task
    assigned_agent: str
    status: TaskStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Agent:
    """智能体定义"""
    agent_id: str
    agent_type: str
    name: str
    description: str
    capability: AgentCapability
    status: AgentStatus = AgentStatus.IDLE
    current_tasks: List[str] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    total_tasks: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    last_active: Optional[datetime] = None

class AgentOrchestrator:
    """智能体编排器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化编排器"""
        self.config_path = config_path or "agent_orchestrator_config.yaml"
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.executions: Dict[str, TaskExecution] = {}
        self.task_queue: List[Task] = []
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=20)
        
        # 监控
        self.monitoring_active = False
        self.monitor_thread = None
        
        # 统计
        self.statistics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'active_agents': 0,
            'average_completion_time': 0.0,
            'total_cost': 0.0
        }
        
        # 加载配置
        self._load_configuration()
        
        # 加载智能体定义
        self._load_agent_definitions()
        
        # 启动监控
        self._start_monitoring()
        
        logger.info("🚀 智能体编排器初始化完成")
    
    def _load_configuration(self):
        """加载配置"""
        default_config = {
            'orchestrator': {
                'max_concurrent_tasks': 50,
                'task_timeout': 3600,
                'retry_attempts': 3,
                'retry_delay': 5,
                'load_balancing': 'round_robin',
                'auto_scaling': True
            },
            'monitoring': {
                'check_interval': 5,
                'performance_threshold': 0.8,
                'health_check_interval': 30
            }
        }
        
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
                logger.info(f"📄 已加载配置文件: {self.config_path}")
            except Exception as e:
                logger.error(f"❌ 加载配置文件失败: {e}")
        
        self.config = default_config
    
    def _load_agent_definitions(self):
        """加载智能体定义"""
        agents_dir = Path(".iflow/agents")
        if not agents_dir.exists():
            logger.warning(f"⚠️ 智能体目录不存在: {agents_dir}")
            return
        
        for agent_file in agents_dir.glob("**/*.md"):
            try:
                with open(agent_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 解析智能体定义
                agent_data = self._parse_agent_definition(content, agent_file)
                
                if agent_data:
                    agent = Agent(
                        agent_id=agent_data['agent_id'],
                        agent_type=agent_data['agent_type'],
                        name=agent_data['name'],
                        description=agent_data['description'],
                        capability=AgentCapability(
                            agent_id=agent_data['agent_id'],
                            agent_type=agent_data['agent_type'],
                            skills=agent_data.get('skills', []),
                            max_concurrent_tasks=agent_data.get('max_concurrent_tasks', 3),
                            expertise_level=agent_data.get('expertise_level', 0.8),
                            reliability_score=agent_data.get('reliability_score', 0.9),
                            cost_per_hour=agent_data.get('cost_per_hour', 0.0),
                            specializations=agent_data.get('specializations', [])
                        )
                    )
                    
                    self.agents[agent.agent_id] = agent
                    logger.info(f"🤖 加载智能体: {agent.name} (ID: {agent.agent_id})")
                    
            except Exception as e:
                logger.error(f"❌ 加载智能体失败 {agent_file}: {e}")
        
        logger.info(f"📋 已加载 {len(self.agents)} 个智能体")
    
    def _parse_agent_definition(self, content: str, file_path: Path) -> Optional[Dict[str, Any]]:
        """解析智能体定义"""
        lines = content.split('\n')
        
        # 提取基本信息
        agent_data = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                continue
            
            if ':' in line:
                key, value = line.split(':', 1)
                agent_data[key.strip()] = value.strip()
        
        # 从文件名推断agent_id
        agent_data['agent_id'] = file_path.stem
        agent_data['skills'] = self._extract_skills(content)
        agent_data['agent_type'] = agent_data.get('agent_type', 'general')
        agent_data['name'] = agent_data.get('name', agent_data['agent_id'])
        agent_data['description'] = agent_data.get('description', '')
        
        return agent_data if 'agent_id' in agent_data else None
    
    def _extract_skills(self, content: str) -> List[str]:
        """提取技能列表"""
        skills = []
        
        # 查找技能相关关键词
        skill_patterns = [
            r'技能[:：:]\s*([^\n]+)',
            r'专长[:：:]\s*([^\n]+)',
            r'能力[:：:]\s*([^\n]+)',
            r'Skills[:：:]\s*([^\n]+)',
            r'核心能力[:：:]\s*([^\n]+)'
        ]
        
        import re
        for pattern in skill_patterns:
            matches = re.findall(pattern, content)
            skills.extend(matches)
        
        # 清理和去重
        skills = list(set(skill.strip() for skill in skills if skill.strip()))
        
        return skills
    
    def _start_monitoring(self):
        """启动监控"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🔍 智能体编排监控已启动")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 检查智能体健康状态
                self._check_agent_health()
                
                # 更新统计信息
                self._update_statistics()
                
                # 检查超时任务
                self._check_timeouts()
                
                time.sleep(self.config['monitoring']['check_interval'])
                
            except Exception as e:
                logger.error(f"❌ 监控循环错误: {e}")
                time.sleep(10)
    
    def _check_agent_health(self):
        """检查智能体健康状态"""
        for agent_id, agent in self.agents.items():
            # 检查任务执行时间
            if agent.last_active:
                idle_time = (datetime.now() - agent.last_active).total_seconds()
                if idle_time > 300:  # 5分钟无活动
                    agent.status = AgentStatus.IDLE
                    logger.info(f"🔄 智能体 {agent.name} 进入空闲状态")
    
    def _update_statistics(self):
        """更新统计信息"""
        self.statistics['total_tasks'] = len(self.tasks)
        self.statistics['completed_tasks'] = len(self.completed_tasks)
        self.statistics['failed_tasks'] = len(self.failed_tasks)
        self.statistics['active_agents'] = len([
            a for a in self.agents.values() if a.status == AgentStatus.BUSY
        ])
        
        # 计算平均完成时间
        completed_executions = [
            e for e in self.executions.values() 
            if e.status == TaskStatus.COMPLETED and e.end_time and e.start_time
        ]
        
        if completed_executions:
            total_time = sum(
                (e.end_time - e.start_time).total_seconds()
                for e in completed_executions
            )
            self.statistics['average_completion_time'] = total_time / len(completed_executions)
        
        # 计算总成本
        total_cost = 0.0
        for execution in self.executions.values():
            if execution.assigned_agent and execution.assigned_agent in self.agents:
                cost_per_hour = self.agents[execution.assigned_agent].capability.cost_per_hour
                if execution.start_time and execution.end_time:
                    duration_hours = (execution.end_time - execution.start_time).total_seconds() / 3600
                    total_cost += duration_hours * cost_per_hour
        
        self.statistics['total_cost'] = total_cost
    
    def _check_timeouts(self):
        """检查超时任务"""
        timeout_threshold = self.config['orchestrator']['task_timeout']
        
        for execution in self.executions.values():
            if execution.status == TaskStatus.RUNNING:
                if execution.start_time:
                    elapsed = (datetime.now() - execution.start_time).total_seconds()
                    if elapsed > timeout_threshold:
                        logger.warning(f"⚠️ 任务超时: {execution.task.title} (ID: {execution.task.task_id})")
                        execution.status = TaskStatus.FAILED
                        execution.error = "执行超时"
                        execution.end_time = datetime.now()
                        self.failed_tasks.append(execution.task.task_id)
    
    def create_task(self, 
                    task_type: str,
                    title: str,
                    description: str,
                    priority: TaskPriority = TaskPriority.MEDIUM,
                    required_skills: List[str] = None,
                    estimated_duration: int = 300,
                    dependencies: List[str] = None,
                    context: Dict[str, Any] = None,
                    requirements: Dict[str, Any] = None,
                    deadline: datetime = None) -> str:
        """创建任务"""
        task_id = str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            title=title,
            description=description,
            priority=priority,
            required_skills=required_skills or [],
            estimated_duration=estimated_duration,
            dependencies=dependencies or [],
            context=context or {},
            requirements=requirements or {},
            deadline=deadline
        )
        
        self.tasks[task_id] = task
        self.task_queue.append(task)
        
        logger.info(f"📝 创建任务: {title} (ID: {task_id})")
        
        return task_id
    
    async def submit_task(self, task_id: str) -> bool:
        """提交任务执行"""
        if task_id not in self.tasks:
            logger.error(f"❌ 任务不存在: {task_id}")
            return False
        
        task = self.tasks[task_id]
        
        # 检查依赖
        if not self._check_dependencies(task):
            logger.error(f"❌ 任务依赖未满足: {task.title}")
            return False
        
        # 选择合适的智能体
        assigned_agent = await self._select_agent_for_task(task)
        
        if not assigned_agent:
            logger.error(f"❌ 没有合适的智能体处理任务: {task.title}")
            return False
        
        # 创建执行实例
        execution = TaskExecution(
            execution_id=str(uuid.uuid4()),
            task=task,
            assigned_agent=assigned_agent,
            status=TaskStatus.PENDING
        )
        
        self.executions[execution.execution_id] = execution
        
        # 更新智能体状态
        self.agents[assigned_agent].current_tasks.append(task_id)
        self.agents[assigned_agent].status = AgentStatus.BUSY
        self.agents[assigned_agent].last_active = datetime.now()
        
        # 异步执行任务
        asyncio.create_task(self._execute_task(execution))
        
        logger.info(f"🚀 提交任务执行: {task.title} -> {assigned_agent}")
        
        return True
    
    def _check_dependencies(self, task: Task) -> bool:
        """检查任务依赖"""
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                logger.error(f"❌ 依赖任务不存在: {dep_id}")
                return False
            
            dep_task = self.tasks[dep_id]
            if dep_id not in self.completed_tasks:
                logger.error(f"❌ 依赖任务未完成: {dep_task.title}")
                return False
        
        return True
    
    async def _select_agent_for_task(self, task: Task) -> Optional[str]:
        """为任务选择智能体"""
        candidate_agents = []
        
        for agent_id, agent in self.agents.items():
            # 检查智能体是否空闲
            if agent.status != AgentStatus.IDLE:
                continue
            
            # 检查技能匹配度
            skill_match = self._calculate_skill_match(task, agent)
            if skill_match > 0.5:  # 技能匹配度阈值
                candidate_agents.append((agent_id, skill_match))
        
        if not candidate_agents:
            return None
        
        # 选择最匹配的智能体
        best_agent = max(candidate_agents, key=lambda x: x[1])
        
        return best_agent[0]
    
    def _calculate_skill_match(self, task: Task, agent: Agent) -> float:
        """计算技能匹配度"""
        if not task.required_skills:
            return 0.8  # 没有技能要求时给予默认分数
        
        agent_skills = set(agent.capability.skills)
        required_skills = set(task.required_skills)
        
        if not required_skills:
            return 0.8
        
        # 计算匹配度
        match_count = len(required_skills & agent_skills)
        match_score = match_count / len(required_skills)
        
        # 考虑专业程度
        expertise_bonus = agent.capability.expertise_level * 0.2
        
        return min(1.0, match_score + expertise_bonus)
    
    async def _execute_task(self, execution: TaskExecution):
        """执行任务"""
        task = execution.task
        agent = self.agents[execution.assigned_agent]
        
        execution.status = TaskStatus.RUNNING
        execution.start_time = datetime.now()
        
        logger.info(f"🔄 开始执行任务: {task.title} (智能体: {agent.name})")
        
        try:
            # 在线程池中执行任务
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._execute_task_sync,
                execution
            )
            
            execution.result = result
            execution.status = TaskStatus.COMPLETED
            execution.end_time = datetime.now()
            execution.progress = 1.0
            
            # 更新智能体状态
            agent.current_tasks.remove(task.task_id)
            agent.completed_tasks.append(task.task_id)
            agent.total_tasks += 1
            
            # 更新性能指标
            execution_time = (execution.end_time - execution.start_time).total_seconds()
            agent.performance_metrics['avg_execution_time'] = (
                sum(agent.performance_metrics.get('avg_execution_time', []) + [execution_time]) / 
                len(agent.performance_metrics.get('avg_execution_time', []) + [execution_time])
            )
            
            self.completed_tasks.append(task.task_id)
            
            logger.info(f"✅ 任务完成: {task.title} (耗时: {execution_time:.2f}秒)")
            
        except Exception as e:
            execution.status = TaskStatus.FAILED
            execution.error = str(e)
            execution.end_time = datetime.now()
            
            # 更新智能体状态
            agent.current_tasks.remove(task.task_id)
            agent.status = AgentStatus.ERROR
            
            self.failed_tasks.append(task.task_id)
            
            logger.error(f"❌ 任务失败: {task.title} - {e}")
    
    def _execute_task_sync(self, execution: TaskExecution) -> Dict[str, Any]:
        """同步执行任务"""
        task = execution.task
        agent = self.agents[execution.assigned_agent]
        
        # 这里应该调用智能体的具体执行方法
        # 由于智能体是定义在markdown文件中的，我们需要实现一个通用的执行框架
        
        # 模拟任务执行
        logger.info(f"🤖 智能体 {agent.name} 正在执行任务: {task.title}")
        
        # 根据任务类型执行相应逻辑
        if task.task_type == "code_generation":
            return self._execute_code_generation(task, agent)
        elif task.task_type == "code_analysis":
            return self._execute_code_analysis(task, agent)
        elif task.task_type == "documentation":
            return self._execute_documentation(task, agent)
        elif task.task_type == "testing":
            return self._execute_testing(task, agent)
        else:
            return self._execute_general_task(task, agent)
    
    def _execute_code_generation(self, task: Task, agent: Agent) -> Dict[str, Any]:
        """执行代码生成任务"""
        logger.info(f"💻 执行代码生成: {task.title}")
        
        # 模拟代码生成过程
        time.sleep(min(task.estimated_duration / 10, 5))  # 模拟执行时间
        
        return {
            'code': f"# Generated code for {task.title}\nprint('Hello, {task.title}')",
            'language': 'python',
            'lines': 10,
            'quality_score': 0.9
        }
    
    def _execute_code_analysis(self, task: Task, agent: Agent) -> Dict[str, Any]:
        """执行代码分析任务"""
        logger.info(f"🔍 执行代码分析: {task.title}")
        
        # 模拟代码分析过程
        time.sleep(min(task.estimated_duration / 10, 3))
        
        return {
            'analysis_result': f"Analysis of {task.title}",
            'issues_found': 2,
            'suggestions': ["优化算法复杂度", "添加错误处理"],
            'quality_score': 0.85
        }
    
    def _execute_documentation(self, task: Task, agent: Agent) -> Dict[str, Any]:
        """执行文档生成任务"""
        logger.info(f"📚 执行文档生成: {task.title}")
        
        # 模拟文档生成过程
        time.sleep(min(task.estimated_duration / 10, 2))
        
        return {
            'documentation': f"# {task.title}\n\n## 概述\n这是{task.description}",
            'format': 'markdown',
            'sections': ['概述', '使用方法', 'API参考'],
            'word_count': 500
        }
    
    def _execute_testing(self, task: Task, agent: Agent) -> Dict[str, Any]:
        """执行测试任务"""
        logger.info(f("🧪 执行测试任务: {task.title}")
        
        # 模拟测试过程
        time.sleep(min(task.estimated_duration / 10, 4))
        
        return {
            'test_results': f"Test results for {task.title}",
            'tests_run': 10,
            'tests_passed': 9,
            'coverage': 0.9,
            'test_report': 'test_report.html'
        }
    
    def _execute_general_task(self, task: Task, agent: Agent) -> Dict[str, Any]:
        """执行通用任务"""
        logger.info(f"⚙️ 执行通用任务: {task.title}")
        
        # 模拟通用任务执行
        time.sleep(min(task.estimated_duration / 10, 2))
        
        return {
            'result': f"Completed {task.title}",
            'status': 'success',
            'metadata': task.context
        }
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """获取智能体状态"""
        if agent_id not in self.agents:
            return {}
        
        agent = self.agents[agent_id]
        
        return {
            'agent_id': agent.agent_id,
            'name': agent.name,
            'type': agent.agent_type,
            'status': agent.status.value,
            'current_tasks': agent.current_tasks,
            'completed_tasks': agent.completed_tasks,
            'total_tasks': agent.total_tasks,
            'performance_metrics': agent.performance_metrics,
            'last_active': agent.last_active.isoformat() if agent.last_active else None,
            'capability': asdict(agent.capability)
        }
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        if task_id not in self.tasks:
            return {}
        
        task = self.tasks[task_id]
        
        # 查找执行记录
        execution = None
        for exec_id, exec_data in self.executions.items():
            if exec_data.task.task_id == task_id:
                execution = exec_data
                break
        
        return {
            'task_id': task.task_id,
            'title': task.title,
            'type': task.task_type,
            'priority': task.priority.value,
            'status': execution.status.value if execution else 'pending',
            'progress': execution.progress if execution else 0.0,
            'assigned_agent': execution.assigned_agent if execution else None,
            'created_at': task.created_at.isoformat(),
            'estimated_duration': task.estimated_duration,
            'deadline': task.deadline.isoformat() if task.deadline else None,
            'result': execution.result if execution else None,
            'error': execution.error if execution else None
        }
    
    def get_orchestration_stats(self) -> Dict[str, Any]:
        """获取编排统计信息"""
        return {
            'total_agents': len(self.agents),
            'active_agents': len([a for a in self.agents.values() if a.status == AgentStatus.BUSY]),
            'idle_agents': len([a for a in self.agents.values() if a.status == AgentStatus.IDLE]),
            'total_tasks': len(self.tasks),
            'pending_tasks': len(self.task_queue),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'active_executions': len([e for e in self.executions.values() if e.status == TaskStatus.RUNNING]),
            'statistics': self.statistics
        }
    
    def shutdown(self):
        """关闭编排器"""
        logger.info("🛑 正在关闭智能体编排器...")
        
        # 停止监控
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        logger.info("✅ 智能体编排器已关闭")

# 示例使用
async def main():
    """主函数示例"""
    orchestrator = AgentOrchestrator()
    
    # 创建任务
    task_id = orchestrator.create_task(
        task_type="code_generation",
        title="实现快速排序算法",
        description="实现一个高效的快速排序算法",
        priority=TaskPriority.HIGH,
        required_skills=["python", "algorithms"],
        estimated_duration=600
    )
    
    # 提交任务
    success = await orchestrator.submit_task(task_id)
    
    if success:
        # 等待任务完成
        while True:
            status = orchestrator.get_task_status(task_id)
            if status['status'] in ['completed', 'failed']:
                break
            time.sleep(5)
        
        print(f"任务状态: {status}")
        print(f"任务结果: {status.get('result')}")
    
    # 获取统计信息
    stats = orchestrator.get_orchestration_stats()
    print(f"编排统计: {stats}")
    
    orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())