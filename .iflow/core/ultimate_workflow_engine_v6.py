#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌌 终极工作流引擎 V6 (Ultimate Workflow Engine V6)
T-MIA凤凰架构的核心指挥官，集成iflow CLI深度支持和智能工具调用优化

你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import asyncio
import logging
import time
import uuid
import hashlib
import traceback
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
from enum import Enum
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import concurrent.futures
from collections import defaultdict, deque

# 动态添加项目根目录到sys.path
try:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from iflow.core.ultimate_cognitive_core_v6 import UltimateCognitiveCoreV6
    from iflow.adapters.universal_llm_adapter_v14 import UltimateLLMAdapterV14
    from iflow.core.ultimate_arq_engine_v6 import UltimateARQEngineV6
    from iflow.core.ultimate_consciousness_system_v6 import UltimateConsciousnessSystemV6
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.error(f"关键模块导入失败: {e}")
    sys.exit(1)

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('workflow_engine_v6.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 枚举定义 ---
class WorkflowState(Enum):
    """工作流状态"""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    TOOL_CALLING = "tool_calling"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """任务优先级"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ExecutionMode(Enum):
    """执行模式"""
    AUTONOMOUS = "autonomous"
    INTERACTIVE = "interactive"
    BATCH = "batch"
    STREAMING = "streaming"

@dataclass
class TaskContext:
    """任务上下文"""
    task_id: str
    user_input: str
    execution_mode: ExecutionMode
    priority: TaskPriority
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    timeout: Optional[int] = None

@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    task_id: str
    output: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    tool_calls: List[Dict] = field(default_factory=list)
    validation_results: Dict = field(default_factory=dict)
    confidence_score: float = 0.0

class UltimateWorkflowEngineV6:
    """
    终极工作流引擎V6 - T-MIA凤凰架构的核心指挥官
    集成iflow CLI深度支持、智能工具调用优化、多模型智能路由
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.config = config or {}
        self.engine_id = f"UWE-V6-{uuid.uuid4().hex[:8]}"
        
        # 核心组件
        self.model_adapter: Optional[UltimateLLMAdapterV14] = None
        self.cognitive_core: Optional[UltimateCognitiveCoreV6] = None
        self.arq_engine: Optional[UltimateARQEngineV6] = None
        self.consciousness_system: Optional[UltimateConsciousnessSystemV6] = None
        
        # 执行管理
        self.active_workflows: Dict[str, Dict] = {}
        self.task_queue = asyncio.Queue()
        self.execution_contexts: Dict[str, TaskContext] = {}
        self.result_cache: Dict[str, ExecutionResult] = {}
        
        # 性能监控
        self.performance_metrics = {
            'total_executions': 0,
            'success_rate': 0.0,
            'avg_execution_time': 0.0,
            'tool_call_success_rate': 0.0,
            'model_utilization': defaultdict(int)
        }
        
        # 并发控制
        self.max_concurrent_tasks = self.config.get('max_concurrent_tasks', 5)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        
        # 缓存配置
        self.max_cache_size = self.config.get('max_cache_size', 1000)
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1小时
        
        self._initialized = False
        logger.info(f"🚀 终极工作流引擎V6初始化完成 - Engine ID: {self.engine_id}")
    
    async def initialize(self):
        """异步初始化所有核心组件"""
        if self._initialized:
            logger.info("引擎已初始化，跳过...")
            return
        
        with self._lock:
            if self._initialized:
                return
            
            logger.info("🔧 开始初始化T-MIA凤凰架构核心组件...")
            
            try:
                # 1. 初始化模型适配器V14
                start_time = time.time()
                self.model_adapter = UltimateLLMAdapterV14()
                logger.info(f"✅ 模型适配器V14初始化完成 ({time.time() - start_time:.2f}s)")
                
                # 2. 初始化ARQ引擎V6
                start_time = time.time()
                self.arq_engine = UltimateARQEngineV6()
                logger.info(f"✅ ARQ引擎V6初始化完成 ({time.time() - start_time:.2f}s)")
                
                # 3. 初始化意识流系统V6
                start_time = time.time()
                self.consciousness_system = UltimateConsciousnessSystemV6()
                logger.info(f"✅ 意识流系统V6初始化完成 ({time.time() - start_time:.2f}s)")
                
                # 4. 初始化认知核心V6（注入其他组件）
                start_time = time.time()
                self.cognitive_core = UltimateCognitiveCoreV6(
                    model_adapter=self.model_adapter,
                    arq_engine=self.arq_engine,
                    consciousness_system=self.consciousness_system
                )
                logger.info(f"✅ 认知核心V6初始化完成 ({time.time() - start_time:.2f}s)")
                
                # 5. 预热模型和缓存
                await self._preheat_system()
                
                self._initialized = True
                logger.info("🎉 T-MIA凤凰架构初始化完成！")
                
            except Exception as e:
                logger.error(f"❌ 初始化失败: {e}", exc_info=True)
                raise
    
    async def _preheat_system(self):
        """预热系统"""
        logger.info("🔥 开始系统预热...")
        
        # 预热模型适配器
        if self.model_adapter:
            await self.model_adapter.preheat()
        
        # 预热认知核心
        if self.cognitive_core:
            await self.cognitive_core.preheat()
        
        logger.info("✅ 系统预热完成")
    
    async def execute_task(
        self, 
        user_input: str, 
        execution_mode: ExecutionMode = ExecutionMode.AUTONOMOUS,
        priority: TaskPriority = TaskPriority.MEDIUM,
        dependencies: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> ExecutionResult:
        """
        执行单个任务
        
        Args:
            user_input: 用户输入的任务描述
            execution_mode: 执行模式
            priority: 任务优先级
            dependencies: 依赖任务ID列表
            metadata: 任务元数据
        
        Returns:
            ExecutionResult: 执行结果
        """
        if not self._initialized:
            await self.initialize()
        
        task_id = str(uuid.uuid4())
        context = TaskContext(
            task_id=task_id,
            user_input=user_input,
            execution_mode=execution_mode,
            priority=priority,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        
        self.execution_contexts[task_id] = context
        
        logger.info(f"📋 开始执行任务 [{task_id}]: {user_input[:50]}...")
        
        start_time = time.time()
        
        try:
            # 1. 智能任务分析和规划
            planning_result = await self._analyze_and_plan(task_id, user_input, context)
            if not planning_result.success:
                return ExecutionResult(
                    success=False,
                    task_id=task_id,
                    error=planning_result.error,
                    execution_time=time.time() - start_time
                )
            
            # 2. 执行工作流
            execution_result = await self._execute_workflow(task_id, planning_result.output, context)
            
            # 3. 验证和优化结果
            final_result = await self._validate_and_optimize(task_id, execution_result, context)
            
            # 更新性能指标
            self._update_performance_metrics(success=True, execution_time=time.time() - start_time)
            
            logger.info(f"✅ 任务 [{task_id}] 执行完成，耗时: {time.time() - start_time:.2f}s")
            return final_result
            
        except Exception as e:
            error_msg = f"任务执行异常: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # 更新性能指标
            self._update_performance_metrics(success=False, execution_time=time.time() - start_time)
            
            return ExecutionResult(
                success=False,
                task_id=task_id,
                error=error_msg,
                execution_time=time.time() - start_time
            )
    
    async def _analyze_and_plan(self, task_id: str, user_input: str, context: TaskContext) -> ExecutionResult:
        """智能任务分析和规划"""
        logger.debug(f"🔍 任务分析 [{task_id}]")
        
        try:
            # 调用认知核心进行深度分析
            analysis_result = await self.cognitive_core.analyze_task(
                task_description=user_input,
                context=context.metadata,
                execution_mode=context.execution_mode.value
            )
            
            # ARQ引擎验证分析结果
            if self.arq_engine:
                validation_result = await self.arq_engine.validate_task_analysis(
                    task_input=user_input,
                    analysis=analysis_result
                )
                
                if not validation_result.get('valid', True):
                    return ExecutionResult(
                        success=False,
                        task_id=task_id,
                        error=f"任务分析验证失败: {validation_result.get('reason', 'Unknown')}"
                    )
            
            return ExecutionResult(
                success=True,
                task_id=task_id,
                output=analysis_result,
                confidence_score=analysis_result.get('confidence', 0.8)
            )
            
        except Exception as e:
            logger.error(f"任务分析失败 [{task_id}]: {e}")
            return ExecutionResult(
                success=False,
                task_id=task_id,
                error=f"任务分析失败: {str(e)}"
            )
    
    async def _execute_workflow(self, task_id: str, plan: Dict, context: TaskContext) -> ExecutionResult:
        """执行工作流"""
        logger.debug(f"⚙️ 执行工作流 [{task_id}]")
        
        try:
            # 根据执行模式选择策略
            if context.execution_mode == ExecutionMode.STREAMING:
                result = await self._execute_streaming_workflow(task_id, plan, context)
            elif context.execution_mode == ExecutionMode.BATCH:
                result = await self._execute_batch_workflow(task_id, plan, context)
            else:
                result = await self._execute_autonomous_workflow(task_id, plan, context)
            
            return result
            
        except Exception as e:
            logger.error(f"工作流执行失败 [{task_id}]: {e}")
            return ExecutionResult(
                success=False,
                task_id=task_id,
                error=f"工作流执行失败: {str(e)}"
            )
    
    async def _execute_autonomous_workflow(self, task_id: str, plan: Dict, context: TaskContext) -> ExecutionResult:
        """自主执行工作流"""
        logger.info(f"🤖 开始自主执行 [{task_id}]")
        
        try:
            # 调用认知核心执行完整工作流
            workflow_result = await self.cognitive_core.execute_workflow(
                task_plan=plan,
                context=context.metadata,
                tools_enabled=True
            )
            
            return ExecutionResult(
                success=True,
                task_id=task_id,
                output=workflow_result.get('output'),
                tool_calls=workflow_result.get('tool_calls', []),
                confidence_score=workflow_result.get('confidence', 0.8),
                execution_time=workflow_result.get('execution_time', 0.0)
            )
            
        except Exception as e:
            logger.error(f"自主工作流执行失败 [{task_id}]: {e}")
            return ExecutionResult(
                success=False,
                task_id=task_id,
                error=f"自主工作流执行失败: {str(e)}"
            )
    
    async def _execute_streaming_workflow(self, task_id: str, plan: Dict, context: TaskContext) -> ExecutionResult:
        """流式执行工作流"""
        logger.info(f"🌊 开始流式执行 [{task_id}]")
        
        try:
            # 流式执行逻辑
            streaming_result = await self.cognitive_core.execute_streaming_workflow(
                task_plan=plan,
                context=context.metadata
            )
            
            return ExecutionResult(
                success=True,
                task_id=task_id,
                output=streaming_result.get('output'),
                tool_calls=streaming_result.get('tool_calls', []),
                confidence_score=streaming_result.get('confidence', 0.8)
            )
            
        except Exception as e:
            logger.error(f"流式工作流执行失败 [{task_id}]: {e}")
            return ExecutionResult(
                success=False,
                task_id=task_id,
                error=f"流式工作流执行失败: {str(e)}"
            )
    
    async def _execute_batch_workflow(self, task_id: str, plan: Dict, context: TaskContext) -> ExecutionResult:
        """批量执行工作流"""
        logger.info(f"📦 开始批量执行 [{task_id}]")
        
        try:
            # 批量执行逻辑
            batch_result = await self.cognitive_core.execute_batch_workflow(
                task_plan=plan,
                context=context.metadata
            )
            
            return ExecutionResult(
                success=True,
                task_id=task_id,
                output=batch_result.get('output'),
                tool_calls=batch_result.get('tool_calls', []),
                confidence_score=batch_result.get('confidence', 0.8)
            )
            
        except Exception as e:
            logger.error(f"批量工作流执行失败 [{task_id}]: {e}")
            return ExecutionResult(
                success=False,
                task_id=task_id,
                error=f"批量工作流执行失败: {str(e)}"
            )
    
    async def _validate_and_optimize(self, task_id: str, result: ExecutionResult, context: TaskContext) -> ExecutionResult:
        """验证和优化结果"""
        logger.debug(f"✅ 结果验证和优化 [{task_id}]")
        
        try:
            # 1. 基础验证
            validation_results = {}
            
            # 2. 如果有ARQ引擎，进行深度验证
            if self.arq_engine and result.success:
                validation_input = {
                    'task_id': task_id,
                    'result': result.output,
                    'context': context.metadata
                }
                
                arq_validation = await self.arq_engine.validate_execution_result(validation_input)
                validation_results.update(arq_validation)
            
            # 3. 如果验证失败，尝试自动修复
            if not validation_results.get('valid', True):
                logger.warning(f"结果验证失败，尝试自动修复 [{task_id}]")
                repair_result = await self._attempt_automatic_repair(task_id, result, validation_results)
                if repair_result.success:
                    result = repair_result
                    logger.info(f"✅ 自动修复成功 [{task_id}]")
            
            # 4. 缓存结果
            if result.success:
                await self._cache_execution_result(task_id, result)
            
            result.validation_results = validation_results
            return result
            
        except Exception as e:
            logger.error(f"结果验证和优化失败 [{task_id}]: {e}")
            return result
    
    async def _attempt_automatic_repair(self, task_id: str, result: ExecutionResult, validation_results: Dict) -> ExecutionResult:
        """尝试自动修复"""
        logger.info(f"🔧 尝试自动修复 [{task_id}]")
        
        try:
            # 调用认知核心进行修复
            repair_input = {
                'original_result': result,
                'validation_errors': validation_results.get('errors', []),
                'task_context': self.execution_contexts.get(task_id)
            }
            
            repair_result = await self.cognitive_core.attempt_repair(repair_input)
            
            if repair_result.get('success', False):
                logger.info(f"✅ 自动修复成功 [{task_id}]")
                return ExecutionResult(
                    success=True,
                    task_id=task_id,
                    output=repair_result.get('repaired_output'),
                    error=None,
                    confidence_score=repair_result.get('confidence', 0.7)
                )
            else:
                logger.info(f"❌ 自动修复失败 [{task_id}]")
                return result
                
        except Exception as e:
            logger.error(f"自动修复过程异常 [{task_id}]: {e}")
            return result
    
    async def _cache_execution_result(self, task_id: str, result: ExecutionResult):
        """缓存执行结果"""
        try:
            if len(self.result_cache) >= self.max_cache_size:
                # 移除最旧的缓存项
                oldest_key = next(iter(self.result_cache))
                del self.result_cache[oldest_key]
            
            self.result_cache[task_id] = result
            logger.debug(f"📊 结果已缓存 [{task_id}]")
            
        except Exception as e:
            logger.warning(f"缓存结果失败 [{task_id}]: {e}")
    
    def _update_performance_metrics(self, success: bool, execution_time: float):
        """更新性能指标"""
        self.performance_metrics['total_executions'] += 1
        
        if success:
            # 更新成功率
            total = self.performance_metrics['total_executions']
            successful = sum(1 for r in self.result_cache.values() if r.success)
            self.performance_metrics['success_rate'] = successful / total if total > 0 else 0.0
            
            # 更新平均执行时间
            times = [r.execution_time for r in self.result_cache.values() if r.success]
            self.performance_metrics['avg_execution_time'] = sum(times) / len(times) if times else 0.0
        
        logger.debug(f"📈 性能指标更新: 总执行数={self.performance_metrics['total_executions']}, 成功率={self.performance_metrics['success_rate']:.2%}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'engine_id': self.engine_id,
            'initialized': self._initialized,
            'active_workflows': len(self.active_workflows),
            'performance_metrics': dict(self.performance_metrics),
            'cache_size': len(self.result_cache),
            'execution_contexts': len(self.execution_contexts),
            'components': {
                'model_adapter': self.model_adapter is not None,
                'cognitive_core': self.cognitive_core is not None,
                'arq_engine': self.arq_engine is not None,
                'consciousness_system': self.consciousness_system is not None
            }
        }
    
    async def shutdown(self):
        """优雅关闭引擎"""
        logger.info("🛑 开始关闭终极工作流引擎V6...")
        
        try:
            # 关闭认知核心
            if self.cognitive_core:
                await self.cognitive_core.close()
                logger.info("✅ 认知核心已关闭")
            
            # 关闭意识流系统
            if self.consciousness_system:
                self.consciousness_system.close()
                logger.info("✅ 意识流系统已关闭")
            
            # 关闭线程池
            self.executor.shutdown(wait=True)
            logger.info("✅ 线程池已关闭")
            
            logger.info("🎉 终极工作流引擎V6已完全关闭")
            
        except Exception as e:
            logger.error(f"关闭过程中出现错误: {e}", exc_info=True)

# --- 示例使用 ---
async def main():
    """示例使用"""
    print("🚀 启动终极工作流引擎V6演示")
    print("=" * 60)
    
    engine = UltimateWorkflowEngineV6()
    
    # 初始化
    await engine.initialize()
    
    # 执行复杂任务
    task = "分析一个电商平台的性能瓶颈，并提出一套完整的、包含前端、后端和数据库的优化方案。"
    metadata = {
        "platform_tech_stack": ["React", "Node.js", "PostgreSQL"],
        "current_issues": ["页面加载慢", "高并发下API响应延迟高"]
    }
    
    print(f"\n📋 执行任务: {task[:50]}...")
    
    result = await engine.execute_task(
        user_input=task,
        execution_mode=ExecutionMode.AUTONOMOUS,
        priority=TaskPriority.HIGH,
        metadata=metadata
    )
    
    print(f"\n📊 执行结果:")
    print(f"- 成功: {result.success}")
    print(f"- 耗时: {result.execution_time:.2f}秒")
    print(f"- 置信度: {result.confidence_score:.2f}")
    
    if result.success:
        print(f"- 输出长度: {len(str(result.output))} 字符")
        print(f"- 工具调用: {len(result.tool_calls)} 次")
    else:
        print(f"- 错误: {result.error}")
    
    # 获取系统状态
    status = await engine.get_system_status()
    print(f"\n🔧 系统状态:")
    print(f"- 引擎ID: {status['engine_id']}")
    print(f"- 已初始化: {status['initialized']}")
    print(f"- 活跃工作流: {status['active_workflows']}")
    print(f"- 成功率: {status['performance_metrics']['success_rate']:.2%}")
    
    # 关闭引擎
    await engine.shutdown()
    print("\n✅ 演示完成")

if __name__ == "__main__":
    # 确保在Windows上asyncio事件循环正常工作
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断执行")
    except Exception as e:
        logger.error(f"程序执行异常: {e}", exc_info=True)
        sys.exit(1)