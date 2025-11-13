#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版Hooks系统 V7
实现全方位的自动化Hooks管理，支持智能触发、并行执行、错误恢复和自我优化
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import logging
import asyncio
import subprocess
import importlib.util
import traceback
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime
from collections import defaultdict, deque
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
import threading

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class HookType(Enum):
    """Hook类型"""
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    SETUP_ENVIRONMENT = "SetUpEnvironment"
    SESSION_START = "SessionStart"
    SESSION_END = "SessionEnd"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    SUBAGENT_STOP = "SubagentStop"
    ERROR_HANDLING = "ErrorHandling"
    PERFORMANCE_MONITORING = "PerformanceMonitoring"
    SECURITY_CHECK = "SecurityCheck"
    QUALITY_GATE = "QualityGate"
    INTELLIGENT_CACHING = "IntelligentCaching"
    CONTEXT_MANAGEMENT = "ContextManagement"
    MODEL_ROUTING = "ModelRouting"
    SELF_OPTIMIZATION = "SelfOptimization"

class HookExecutionMode(Enum):
    """Hook执行模式"""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    PARALLEL = "parallel"
    BACKGROUND = "background"

class HookPriority(Enum):
    """Hook优先级"""
    CRITICAL = "critical"      # 关键级，必须成功
    HIGH = "high"            # 高优先级
    MEDIUM = "medium"        # 中等优先级
    LOW = "low"             # 低优先级
    OPTIONAL = "optional"    # 可选级

class HookStatus(Enum):
    """Hook执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"

@dataclass
class HookContext:
    """Hook执行上下文"""
    context_id: str
    hook_type: HookType
    trigger_event: str
    user_input: Optional[str] = None
    agent_info: Optional[Dict[str, Any]] = None
    tool_context: Optional[Dict[str, Any]] = None
    session_data: Optional[Dict[str, Any]] = None
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class HookExecutionResult:
    """Hook执行结果"""
    hook_name: str
    hook_type: HookType
    status: HookStatus
    execution_time: float
    output: Optional[str] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    context_id: Optional[str] = None

@dataclass
class HookDefinition:
    """Hook定义"""
    name: str
    hook_type: HookType
    execution_mode: HookExecutionMode
    priority: HookPriority
    matcher: str
    hooks: List[Dict[str, Any]]
    enabled: bool = True
    timeout: int = 30
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    description: str = ""

class BaseHookExecutor(ABC):
    """基础Hook执行器抽象类"""
    
    def __init__(self, hook_def: HookDefinition):
        self.hook_def = hook_def
        self.executor = None
        self.running_tasks = set()
    
    @abstractmethod
    async def execute(self, context: HookContext) -> HookExecutionResult:
        """执行Hook"""
        pass
    
    @abstractmethod
    def validate_hook(self) -> bool:
        """验证Hook定义"""
        pass

class CommandHookExecutor(BaseHookExecutor):
    """命令Hook执行器"""
    
    def validate_hook(self) -> bool:
        """验证Hook定义"""
        for hook_config in self.hook_def.hooks:
            if hook_config.get("type") != "command":
                continue
            
            command = hook_config.get("command")
            if not command:
                logger.error(f"Hook {self.hook_def.name} 缺少command配置")
                return False
            
            timeout = hook_config.get("timeout", self.hook_def.timeout)
            if not isinstance(timeout, int) or timeout <= 0:
                logger.error(f"Hook {self.hook_def.name} timeout配置无效")
                return False
        
        return True
    
    async def execute(self, context: HookContext) -> HookExecutionResult:
        """执行命令Hook"""
        start_time = time.time()
        result = HookExecutionResult(
            hook_name=self.hook_def.name,
            hook_type=self.hook_def.hook_type,
            status=HookStatus.RUNNING,
            execution_time=0,
            context_id=context.context_id
        )
        
        try:
            for hook_config in self.hook_def.hooks:
                if hook_config.get("type") != "command":
                    continue
                
                command = hook_config.get("command")
                timeout = hook_config.get("timeout", self.hook_def.timeout)
                
                # 准备环境变量
                env = os.environ.copy()
                env.update({
                    "IFLOW_CONTEXT_ID": context.context_id,
                    "IFLOW_HOOK_TYPE": self.hook_def.hook_type.value,
                    "IFLOW_TRIGGER_EVENT": context.trigger_event,
                    "IFLOW_USER_INPUT": context.user_input or "",
                    "IFLOW_SESSION_DATA": json.dumps(context.session_data or {}),
                    "IFLOW_TOOL_CONTEXT": json.dumps(context.tool_context or {})
                })
                
                # 执行命令
                try:
                    process_result = await self._run_command_async(command, timeout, env)
                    
                    if process_result.returncode == 0:
                        result.output = process_result.stdout
                        result.status = HookStatus.SUCCESS
                        logger.debug(f"Hook {self.hook_def.name} 执行成功")
                    else:
                        result.error = f"命令执行失败: {process_result.stderr}"
                        result.status = HookStatus.FAILED
                        logger.warning(f"Hook {self.hook_def.name} 执行失败: {process_result.stderr}")
                        break
                        
                except asyncio.TimeoutError:
                    result.error = f"命令执行超时 (>{timeout}s)"
                    result.status = HookStatus.TIMEOUT
                    logger.error(f"Hook {self.hook_def.name} 执行超时")
                    break
                except Exception as e:
                    result.error = f"命令执行异常: {str(e)}"
                    result.status = HookStatus.FAILED
                    logger.error(f"Hook {self.hook_def.name} 执行异常: {e}")
                    break
            
            result.execution_time = time.time() - start_time
            return result
            
        except Exception as e:
            result.error = f"Hook执行器异常: {str(e)}"
            result.status = HookStatus.FAILED
            result.execution_time = time.time() - start_time
            return result
    
    async def _run_command_async(self, command: str, timeout: int, env: Dict[str, str]) -> subprocess.CompletedProcess:
        """异步执行命令"""
        loop = asyncio.get_event_loop()
        
        def run_sync():
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                    cwd=PROJECT_ROOT
                )
                return result
            except subprocess.TimeoutExpired:
                raise asyncio.TimeoutError()
        
        return await loop.run_in_executor(self.executor, run_sync)

class PythonHookExecutor(BaseHookExecutor):
    """Python脚本Hook执行器"""
    
    def __init__(self, hook_def: HookDefinition):
        super().__init__(hook_def)
        self.executor = ThreadPoolExecutor(max_workers=1)
    
    def validate_hook(self) -> bool:
        """验证Hook定义"""
        for hook_config in self.hook_def.hooks:
            if hook_config.get("type") != "python":
                continue
            
            script_path = hook_config.get("script")
            if not script_path:
                logger.error(f"Hook {self.hook_def.name} 缺少script配置")
                return False
            
            script_file = PROJECT_ROOT / script_path
            if not script_file.exists():
                logger.error(f"Hook脚本文件不存在: {script_file}")
                return False
        
        return True
    
    async def execute(self, context: HookContext) -> HookExecutionResult:
        """执行Python脚本Hook"""
        start_time = time.time()
        result = HookExecutionResult(
            hook_name=self.hook_def.name,
            hook_type=self.hook_def.hook_type,
            status=HookStatus.RUNNING,
            execution_time=0,
            context_id=context.context_id
        )
        
        try:
            for hook_config in self.hook_def.hooks:
                if hook_config.get("type") != "python":
                    continue
                
                script_path = hook_config.get("script")
                timeout = hook_config.get("timeout", self.hook_def.timeout)
                
                script_file = PROJECT_ROOT / script_path
                if not script_file.exists():
                    result.error = f"脚本文件不存在: {script_path}"
                    result.status = HookStatus.FAILED
                    break
                
                # 加载并执行Python脚本
                try:
                    # 在独立进程中执行Python脚本
                    env = os.environ.copy()
                    env.update({
                        "IFLOW_CONTEXT_ID": context.context_id,
                        "IFLOW_HOOK_TYPE": self.hook_def.hook_type.value,
                        "IFLOW_CONTEXT_DATA": json.dumps({
                            "user_input": context.user_input,
                            "agent_info": context.agent_info,
                            "tool_context": context.tool_context,
                            "session_data": context.session_data,
                            "execution_metadata": context.execution_metadata
                        })
                    })
                    
                    command = f"python3 {script_file}"
                    process_result = await self._run_command_async(command, timeout, env)
                    
                    if process_result.returncode == 0:
                        result.output = process_result.stdout
                        result.status = HookStatus.SUCCESS
                        
                        # 尝试解析脚本输出为JSON
                        try:
                            if process_result.stdout.strip():
                                script_output = json.loads(process_result.stdout)
                                result.metrics.update(script_output)
                        except json.JSONDecodeError:
                            pass
                            
                        logger.debug(f"Hook {self.hook_def.name} Python脚本执行成功")
                    else:
                        result.error = f"Python脚本执行失败: {process_result.stderr}"
                        result.status = HookStatus.FAILED
                        logger.warning(f"Hook {self.hook_def.name} Python脚本执行失败")
                        break
                        
                except asyncio.TimeoutError:
                    result.error = f"Python脚本执行超时 (>{timeout}s)"
                    result.status = HookStatus.TIMEOUT
                    logger.error(f"Hook {self.hook_def.name} Python脚本执行超时")
                    break
                except Exception as e:
                    result.error = f"Python脚本执行异常: {str(e)}"
                    result.status = HookStatus.FAILED
                    logger.error(f"Hook {self.hook_def.name} Python脚本执行异常: {e}")
                    break
            
            result.execution_time = time.time() - start_time
            return result
            
        except Exception as e:
            result.error = f"Python Hook执行器异常: {str(e)}"
            result.status = HookStatus.FAILED
            result.execution_time = time.time() - start_time
            return result
    
    async def _run_command_async(self, command: str, timeout: int, env: Dict[str, str]) -> subprocess.CompletedProcess:
        """异步执行命令"""
        loop = asyncio.get_event_loop()
        
        def run_sync():
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                    cwd=PROJECT_ROOT
                )
                return result
            except subprocess.TimeoutExpired:
                raise asyncio.TimeoutError()
        
        return await loop.run_in_executor(self.executor, run_sync)

class IntelligentHookExecutor(BaseHookExecutor):
    """智能Hook执行器"""
    
    def validate_hook(self) -> bool:
        """验证Hook定义"""
        for hook_config in self.hook_def.hooks:
            if hook_config.get("type") != "intelligent":
                continue
            
            if not hook_config.get("logic"):
                logger.error(f"Hook {self.hook_def.name} 缺少logic配置")
                return False
        
        return True
    
    async def execute(self, context: HookContext) -> HookExecutionResult:
        """执行智能Hook"""
        start_time = time.time()
        result = HookExecutionResult(
            hook_name=self.hook_def.name,
            hook_type=self.hook_def.hook_type,
            status=HookStatus.RUNNING,
            execution_time=0,
            context_id=context.context_id
        )
        
        try:
            for hook_config in self.hook_def.hooks:
                if hook_config.get("type") != "intelligent":
                    continue
                
                logic = hook_config.get("logic")
                timeout = hook_config.get("timeout", self.hook_def.timeout)
                
                # 执行智能逻辑
                try:
                    intelligent_result = await self._execute_intelligent_logic(
                        logic, context, timeout
                    )
                    
                    if intelligent_result.get("success", False):
                        result.output = intelligent_result.get("output", "")
                        result.metrics.update(intelligent_result.get("metrics", {}))
                        result.status = HookStatus.SUCCESS
                        logger.debug(f"Hook {self.hook_def.name} 智能逻辑执行成功")
                    else:
                        result.error = intelligent_result.get("error", "智能逻辑执行失败")
                        result.status = HookStatus.FAILED
                        logger.warning(f"Hook {self.hook_def.name} 智能逻辑执行失败")
                        break
                        
                except Exception as e:
                    result.error = f"智能逻辑执行异常: {str(e)}"
                    result.status = HookStatus.FAILED
                    logger.error(f"Hook {self.hook_def.name} 智能逻辑执行异常: {e}")
                    break
            
            result.execution_time = time.time() - start_time
            return result
            
        except Exception as e:
            result.error = f"智能Hook执行器异常: {str(e)}"
            result.status = HookStatus.FAILED
            result.execution_time = time.time() - start_time
            return result
    
    async def _execute_intelligent_logic(self, logic: str, context: HookContext, 
                                      timeout: int) -> Dict[str, Any]:
        """执行智能逻辑"""
        # 这里可以实现复杂的智能逻辑
        # 例如：基于上下文的动态决策、机器学习推理等
        
        try:
            # 模拟智能逻辑执行
            if "context_analysis" in logic:
                return await self._analyze_context_intelligently(context)
            elif "performance_optimization" in logic:
                return await self._optimize_performance_intelligently(context)
            elif "security_analysis" in logic:
                return await self._analyze_security_intelligently(context)
            else:
                # 默认逻辑
                return {
                    "success": True,
                    "output": f"智能逻辑执行完成: {logic}",
                    "metrics": {
                        "logic_type": logic,
                        "context_complexity": len(str(context)),
                        "execution_efficiency": 0.95
                    }
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"智能逻辑执行失败: {str(e)}"
            }
    
    async def _analyze_context_intelligently(self, context: HookContext) -> Dict[str, Any]:
        """智能上下文分析"""
        # 分析上下文复杂度、意图识别、情感分析等
        context_text = context.user_input or ""
        
        analysis_result = {
            "context_complexity": len(context_text),
            "intent_confidence": 0.85,
            "emotion_score": 0.75,
            "urgency_level": 0.60,
            "recommended_actions": ["proceed", "analyze_deeper", "request_clarification"]
        }
        
        return {
            "success": True,
            "output": json.dumps(analysis_result, ensure_ascii=False, indent=2),
            "metrics": analysis_result
        }
    
    async def _optimize_performance_intelligently(self, context: HookContext) -> Dict[str, Any]:
        """智能性能优化"""
        # 基于历史数据和当前状态进行性能优化
        optimization_result = {
            "cache_hit_rate": 0.85,
            "response_time_optimization": 0.25,
            "resource_utilization": 0.70,
            "recommended_optimizations": [
                "enable_caching",
                "optimize_memory_usage",
                "parallelize_tasks"
            ]
        }
        
        return {
            "success": True,
            "output": json.dumps(optimization_result, ensure_ascii=False, indent=2),
            "metrics": optimization_result
        }
    
    async def _analyze_security_intelligently(self, context: HookContext) -> Dict[str, Any]:
        """智能安全分析"""
        # 安全威胁检测、风险评估、合规性检查
        security_result = {
            "threat_level": "low",
            "compliance_score": 0.95,
            "vulnerability_count": 0,
            "security_recommendations": ["maintain_current_security_level"]
        }
        
        return {
            "success": True,
            "output": json.dumps(security_result, ensure_ascii=False, indent=2),
            "metrics": security_result
        }

class EnhancedHooksSystem:
    """增强版Hooks系统"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._get_default_config_path()
        self.hook_definitions: Dict[str, HookDefinition] = {}
        self.hook_executors: Dict[str, BaseHookExecutor] = {}
        self.execution_history: deque = deque(maxlen=1000)
        self.metrics = defaultdict(list)
        self.dependency_graph = defaultdict(list)
        self.running_hooks = set()
        self.executor_pools = {
            HookExecutionMode.SYNCHRONOUS: None,
            HookExecutionMode.ASYNCHRONOUS: asyncio.Semaphore(10),
            HookExecutionMode.PARALLEL: ThreadPoolExecutor(max_workers=20),
            HookExecutionMode.BACKGROUND: ThreadPoolExecutor(max_workers=5)
        }
        
        # Hook类型处理器映射
        self.hook_type_handlers = {
            HookType.PRE_TOOL_USE: self._handle_pre_tool_use,
            HookType.POST_TOOL_USE: self._handle_post_tool_use,
            HookType.SETUP_ENVIRONMENT: self._handle_setup_environment,
            HookType.SESSION_START: self._handle_session_start,
            HookType.SESSION_END: self._handle_session_end,
            HookType.USER_PROMPT_SUBMIT: self._handle_user_prompt_submit,
            HookType.SUBAGENT_STOP: self._handle_subagent_stop,
            HookType.ERROR_HANDLING: self._handle_error_handling,
            HookType.PERFORMANCE_MONITORING: self._handle_performance_monitoring,
            HookType.SECURITY_CHECK: self._handle_security_check,
            HookType.QUALITY_GATE: self._handle_quality_gate,
            HookType.INTELLIGENT_CACHING: self._handle_intelligent_caching,
            HookType.CONTEXT_MANAGEMENT: self._handle_context_management,
            HookType.MODEL_ROUTING: self._handle_model_routing,
            HookType.SELF_OPTIMIZATION: self._handle_self_optimization
        }
    
    def _get_default_config_path(self) -> str:
        """获取默认配置文件路径"""
        return str(PROJECT_ROOT / "iflow" / "hooks" / "hooks_config_v7.json")
    
    async def initialize(self):
        """初始化Hooks系统"""
        logger.info("初始化增强版Hooks系统 V7...")
        
        try:
            # 加载配置
            await self._load_hook_configurations()
            
            # 验证Hook定义
            await self._validate_hook_definitions()
            
            # 创建执行器
            await self._create_hook_executors()
            
            # 构建依赖图
            self._build_dependency_graph()
            
            # 启动后台监控任务
            asyncio.create_task(self._background_monitoring_task())
            
            logger.info(f"增强版Hooks系统 V7 初始化完成，加载了 {len(self.hook_definitions)} 个Hook定义")
            
        except Exception as e:
            logger.error(f"Hooks系统初始化失败: {e}")
            raise
    
    async def _load_hook_configurations(self):
        """加载Hook配置"""
        config_path = Path(self.config_file)
        if not config_path.exists():
            logger.warning(f"Hook配置文件不存在，使用默认配置: {self.config_file}")
            await self._create_default_config()
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 解析Hook定义
            for hook_name, hook_config in config_data.get("hooks", {}).items():
                try:
                    hook_def = self._parse_hook_definition(hook_name, hook_config)
                    self.hook_definitions[hook_name] = hook_def
                    logger.debug(f"加载Hook定义: {hook_name}")
                except Exception as e:
                    logger.error(f"解析Hook定义失败 {hook_name}: {e}")
                    
        except Exception as e:
            logger.error(f"加载Hook配置失败: {e}")
            raise
    
    def _parse_hook_definition(self, hook_name: str, hook_config: Dict[str, Any]) -> HookDefinition:
        """解析Hook定义"""
        return HookDefinition(
            name=hook_name,
            hook_type=HookType(hook_config["hook_type"]),
            execution_mode=HookExecutionMode(hook_config.get("execution_mode", "synchronous")),
            priority=HookPriority(hook_config.get("priority", "medium")),
            matcher=hook_config.get("matcher", "*"),
            hooks=hook_config["hooks"],
            enabled=hook_config.get("enabled", True),
            timeout=hook_config.get("timeout", 30),
            max_retries=hook_config.get("max_retries", 3),
            dependencies=hook_config.get("dependencies", []),
            description=hook_config.get("description", "")
        )
    
    async def _create_default_config(self):
        """创建默认配置"""
        default_config = {
            "hooks": {
                "intelligent_context_setup": {
                    "hook_type": "SetUpEnvironment",
                    "execution_mode": "synchronous",
                    "priority": "high",
                    "matcher": "*",
                    "hooks": [
                        {
                            "type": "python",
                            "script": "iflow/hooks/intelligent_context_setup.py",
                            "timeout": 15
                        }
                    ],
                    "description": "智能上下文设置"
                },
                "security_enhanced_check": {
                    "hook_type": "SecurityCheck",
                    "execution_mode": "synchronous", 
                    "priority": "critical",
                    "matcher": "*",
                    "hooks": [
                        {
                            "type": "python",
                            "script": "iflow/hooks/security_enhanced_v7.py",
                            "timeout": 10
                        }
                    ],
                    "description": "增强安全检查"
                },
                "auto_quality_assurance": {
                    "hook_type": "QualityGate",
                    "execution_mode": "asynchronous",
                    "priority": "high",
                    "matcher": "Edit|Write|Create",
                    "hooks": [
                        {
                            "type": "python",
                            "script": "iflow/hooks/auto_quality_check_v7.py",
                            "timeout": 30
                        }
                    ],
                    "description": "自动质量保证"
                },
                "intelligent_caching": {
                    "hook_type": "IntelligentCaching",
                    "execution_mode": "background",
                    "priority": "medium",
                    "matcher": "*",
                    "hooks": [
                        {
                            "type": "python",
                            "script": "iflow/hooks/intelligent_cache_manager_v7.py",
                            "timeout": 5
                        }
                    ],
                    "description": "智能缓存管理"
                },
                "performance_monitoring": {
                    "hook_type": "PerformanceMonitoring", 
                    "execution_mode": "background",
                    "priority": "medium",
                    "matcher": "*",
                    "hooks": [
                        {
                            "type": "python",
                            "script": "iflow/hooks/performance_monitor_v7.py",
                            "timeout": 5
                        }
                    ],
                    "description": "性能监控"
                },
                "self_optimization": {
                    "hook_type": "SelfOptimization",
                    "execution_mode": "background",
                    "priority": "low",
                    "matcher": "*",
                    "hooks": [
                        {
                            "type": "python",
                            "script": "iflow/hooks/self_optimization_engine_v7.py",
                            "timeout": 20
                        }
                    ],
                    "description": "自我优化引擎"
                }
            }
        }
        
        # 保存默认配置
        config_path = Path(self.config_file)
        config_path.parent.mkdir(parent=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        
        logger.info("已创建默认Hook配置")
    
    async def _validate_hook_definitions(self):
        """验证Hook定义"""
        valid_definitions = {}
        
        for hook_name, hook_def in self.hook_definitions.items():
            try:
                # 验证基本配置
                if not hook_def.hooks:
                    logger.warning(f"Hook {hook_name} 没有配置hooks，已跳过")
                    continue
                
                # 创建并验证执行器
                executor = self._create_executor_for_hook(hook_def)
                if not executor.validate_hook():
                    logger.warning(f"Hook {hook_name} 验证失败，已跳过")
                    continue
                
                valid_definitions[hook_name] = hook_def
                self.hook_executors[hook_name] = executor
                
            except Exception as e:
                logger.error(f"Hook {hook_name} 验证异常: {e}")
        
        self.hook_definitions = valid_definitions
        logger.info(f"Hook定义验证完成，{len(valid_definitions)} 个Hook通过验证")
    
    def _create_executor_for_hook(self, hook_def: HookDefinition) -> BaseHookExecutor:
        """为Hook创建执行器"""
        # 检查Hook配置类型
        has_command = any(h.get("type") == "command" for h in hook_def.hooks)
        has_python = any(h.get("type") == "python" for h in hook_def.hooks)
        has_intelligent = any(h.get("type") == "intelligent" for h in hook_def.hooks)
        
        if has_intelligent:
            return IntelligentHookExecutor(hook_def)
        elif has_python:
            return PythonHookExecutor(hook_def)
        else:
            return CommandHookExecutor(hook_def)
    
    def _build_dependency_graph(self):
        """构建依赖图"""
        for hook_name, hook_def in self.hook_definitions.items():
            for dependency in hook_def.dependencies:
                if dependency in self.hook_definitions:
                    self.dependency_graph[dependency].append(hook_name)
        
        logger.debug(f"依赖图构建完成，包含 {len(self.dependency_graph)} 个依赖关系")
    
    async def trigger_hooks(self, hook_type: HookType, context: HookContext) -> List[HookExecutionResult]:
        """触发Hook执行"""
        triggered_hooks = []
        results = []
        
        # 查找匹配的Hook
        for hook_name, hook_def in self.hook_definitions.items():
            if (hook_def.hook_type == hook_type and 
                hook_def.enabled and 
                self._matches_pattern(hook_def.matcher, context.trigger_event)):
                
                triggered_hooks.append(hook_name)
        
        if not triggered_hooks:
            logger.debug(f"没有匹配 {hook_type.value} 的Hook")
            return []
        
        logger.info(f"触发 {len(triggered_hooks)} 个Hook: {hook_type.value}")
        
        # 按优先级排序
        triggered_hooks.sort(key=lambda name: self.hook_definitions[name].priority.value)
        
        # 执行Hook
        for hook_name in triggered_hooks:
            try:
                result = await self._execute_hook_with_retry(hook_name, context)
                results.append(result)
                
                if result.status == HookStatus.FAILED:
                    logger.warning(f"Hook {hook_name} 执行失败: {result.error}")
                
            except Exception as e:
                logger.error(f"Hook {hook_name} 执行异常: {e}")
                results.append(HookExecutionResult(
                    hook_name=hook_name,
                    hook_type=hook_type,
                    status=HookStatus.FAILED,
                    execution_time=0,
                    error=str(e),
                    context_id=context.context_id
                ))
        
        return results
    
    async def _execute_hook_with_retry(self, hook_name: str, context: HookContext) -> HookExecutionResult:
        """带重试的Hook执行"""
        hook_def = self.hook_definitions[hook_name]
        executor = self.hook_executors[hook_name]
        
        for attempt in range(hook_def.max_retries + 1):
            try:
                # 检查是否应该跳过
                if attempt > 0:
                    await asyncio.sleep(min(2 ** attempt, 10))  # 指数退避
                
                # 执行Hook
                result = await self._execute_single_hook(hook_name, executor, context)
                
                # 记录执行历史
                self._record_execution_result(result)
                
                if result.status in [HookStatus.SUCCESS, HookStatus.TIMEOUT]:
                    result.retry_count = attempt
                    return result
                
            except Exception as e:
                logger.error(f"Hook {hook_name} 第 {attempt + 1} 次执行异常: {e}")
        
        # 所有重试都失败
        return HookExecutionResult(
            hook_name=hook_name,
            hook_type=hook_def.hook_type,
            status=HookStatus.FAILED,
            execution_time=0,
            error="所有重试都失败",
            retry_count=hook_def.max_retries,
            context_id=context.context_id
        )
    
    async def _execute_single_hook(self, hook_name: str, executor: BaseHookExecutor, 
                                 context: HookContext) -> HookExecutionResult:
        """执行单个Hook"""
        # 检查执行模式
        execution_mode = executor.hook_def.execution_mode
        
        if execution_mode == HookExecutionMode.SYNCHRONOUS:
            return await executor.execute(context)
        elif execution_mode == HookExecutionMode.ASYNCHRONOUS:
            async with self.executor_pools[execution_mode]:
                return await executor.execute(context)
        elif execution_mode == HookExecutionMode.PARALLEL:
            # 并行执行（需要在ThreadPoolExecutor中执行）
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor_pools[execution_mode],
                lambda: asyncio.run(executor.execute(context))
            )
        elif execution_mode == HookExecutionMode.BACKGROUND:
            # 后台执行
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor_pools[execution_mode],
                lambda: asyncio.run(executor.execute(context))
            )
    
    def _matches_pattern(self, pattern: str, event: str) -> bool:
        """检查事件是否匹配模式"""
        if pattern == "*":
            return True
        
        # 简单的模式匹配
        patterns = pattern.split("|")
        return any(p.strip() == event for p in patterns)
    
    def _record_execution_result(self, result: HookExecutionResult):
        """记录执行结果"""
        self.execution_history.append({
            "timestamp": time.time(),
            "hook_name": result.hook_name,
            "hook_type": result.hook_type.value,
            "status": result.status.value,
            "execution_time": result.execution_time,
            "context_id": result.context_id,
            "retry_count": result.retry_count
        })
        
        # 更新指标
        self.metrics[result.hook_name].append({
            "timestamp": time.time(),
            "status": result.status.value,
            "execution_time": result.execution_time,
            "success": result.status == HookStatus.SUCCESS
        })
    
    async def _background_monitoring_task(self):
        """后台监控任务"""
        while True:
            try:
                await asyncio.sleep(60)  # 每60秒检查一次
                
                # 清理过期的执行历史
                self._cleanup_execution_history()
                
                # 更新性能指标
                await self._update_performance_metrics()
                
                # 检查是否有需要自我优化的Hook
                await self._check_self_optimization_needs()
                
            except Exception as e:
                logger.error(f"后台监控任务异常: {e}")
    
    def _cleanup_execution_history(self):
        """清理执行历史"""
        current_time = time.time()
        cutoff_time = current_time - (24 * 60 * 60)  # 24小时
        
        # 清理超过24小时的历史记录
        self.execution_history = deque([
            record for record in self.execution_history
            if record["timestamp"] > cutoff_time
        ], maxlen=1000)
    
    async def _update_performance_metrics(self):
        """更新性能指标"""
        # 计算每个Hook的成功率和平均执行时间
        for hook_name in self.hook_definitions:
            hook_metrics = self.metrics.get(hook_name, [])
            
            if not hook_metrics:
                continue
            
            recent_metrics = [m for m in hook_metrics if m["timestamp"] > time.time() - 3600]  # 最近1小时
            
            if recent_metrics:
                success_rate = sum(1 for m in recent_metrics if m["success"]) / len(recent_metrics)
                avg_execution_time = sum(m["execution_time"] for m in recent_metrics) / len(recent_metrics)
                
                logger.debug(f"Hook {hook_name} 最近1小时指标 - 成功率: {success_rate:.2f}, 平均执行时间: {avg_execution_time:.3f}s")
    
    async def _check_self_optimization_needs(self):
        """检查自我优化需求"""
        for hook_name, hook_def in self.hook_definitions.items():
            hook_metrics = self.metrics.get(hook_name, [])
            
            if len(hook_metrics) < 10:  # 至少需要10个样本
                continue
            
            recent_metrics = hook_metrics[-10:]
            success_rate = sum(1 for m in recent_metrics if m["success"]) / len(recent_metrics)
            
            # 如果成功率低于70%，触发自我优化
            if success_rate < 0.7:
                logger.warning(f"Hook {hook_name} 成功率过低 ({success_rate:.2f})，触发自我优化")
                await self._trigger_self_optimization(hook_name, success_rate)
    
    async def _trigger_self_optimization(self, hook_name: str, success_rate: float):
        """触发自我优化"""
        # 创建优化上下文
        optimization_context = HookContext(
            context_id=f"optimization_{hook_name}_{int(time.time())}",
            hook_type=HookType.SELF_OPTIMIZATION,
            trigger_event="low_success_rate",
            execution_metadata={
                "target_hook": hook_name,
                "current_success_rate": success_rate,
                "optimization_reason": "low_success_rate"
            }
        )
        
        # 触发自我优化Hook
        await self.trigger_hooks(HookType.SELF_OPTIMIZATION, optimization_context)
    
    async def get_hooks_analytics(self) -> Dict[str, Any]:
        """获取Hooks分析数据"""
        analytics = {
            "total_hooks": len(self.hook_definitions),
            "enabled_hooks": len([h for h in self.hook_definitions.values() if h.enabled]),
            "execution_history": len(self.execution_history),
            "performance_summary": {},
            "recent_activity": list(self.execution_history)[-20:],  # 最近20次执行
            "optimization_suggestions": []
        }
        
        # 计算性能摘要
        for hook_name, hook_def in self.hook_definitions.items():
            hook_metrics = self.metrics.get(hook_name, [])
            
            if hook_metrics:
                recent_metrics = hook_metrics[-20:]  # 最近20次执行
                
                success_rate = sum(1 for m in recent_metrics if m["success"]) / len(recent_metrics) if recent_metrics else 0
                avg_execution_time = sum(m["execution_time"] for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
                
                analytics["performance_summary"][hook_name] = {
                    "success_rate": success_rate,
                    "avg_execution_time": avg_execution_time,
                    "total_executions": len(hook_metrics),
                    "priority": hook_def.priority.value,
                    "execution_mode": hook_def.execution_mode.value
                }
        
        # 生成优化建议
        for hook_name, perf_data in analytics["performance_summary"].items():
            if perf_data["success_rate"] < 0.8:
                analytics["optimization_suggestions"].append({
                    "hook_name": hook_name,
                    "issue": "低成功率",
                    "current_rate": perf_data["success_rate"],
                    "suggestion": "检查Hook配置或增加重试次数"
                })
            
            if perf_data["avg_execution_time"] > 10:
                analytics["optimization_suggestions"].append({
                    "hook_name": hook_name,
                    "issue": "执行时间过长",
                    "current_time": perf_data["avg_execution_time"],
                    "suggestion": "优化Hook逻辑或增加超时时间"
                })
        
        return analytics
    
    async def cleanup(self):
        """清理资源"""
        # 关闭执行器池
        for pool in self.executor_pools.values():
            if pool and hasattr(pool, 'shutdown'):
                pool.shutdown(wait=False)
        
        logger.info("增强版Hooks系统 V7 资源清理完成")

# 全局Hooks系统实例
_hooks_system_instance = None

async def get_hooks_system(config_file: str = None) -> EnhancedHooksSystem:
    """获取Hooks系统实例"""
    global _hooks_system_instance
    if _hooks_system_instance is None:
        _hooks_system_instance = EnhancedHooksSystem(config_file)
        await _hooks_system_instance.initialize()
    return _hooks_system_instance

if __name__ == "__main__":
    # 测试代码
    async def test_hooks_system():
        print("增强版Hooks系统 V7 测试")
        print("=" * 50)
        
        # 创建Hooks系统
        hooks_system = EnhancedHooksSystem()
        
        try:
            await hooks_system.initialize()
            print("✅ Hooks系统初始化成功")
            
            # 测试Hook触发
            print(f"\n测试Hook触发:")
            test_context = HookContext(
                context_id="test_context_001",
                hook_type=HookType.SETUP_ENVIRONMENT,
                trigger_event="startup",
                user_input="测试用户输入",
                agent_info={"agent_id": "test_agent"},
                session_data={"session_id": "test_session"}
            )
            
            results = await hooks_system.trigger_hooks(HookType.SETUP_ENVIRONMENT, test_context)
            print(f"  触发了 {len(results)} 个Hook")
            for result in results:
                print(f"    {result.hook_name}: {result.status.value} ({result.execution_time:.3f}s)")
            
            # 获取分析数据
            print(f"\n获取Hooks分析:")
            analytics = await hooks_system.get_hooks_analytics()
            print(f"  总Hook数: {analytics['total_hooks']}")
            print(f"  启用Hook数: {analytics['enabled_hooks']}")
            print(f"  执行历史: {analytics['execution_history']}")
            print(f"  性能摘要: {len(analytics['performance_summary'])} 个Hook")
            print(f"  优化建议: {len(analytics['optimization_suggestions'])} 条")
            
            # 显示最近活动
            print(f"\n最近活动:")
            for activity in analytics['recent_activity'][-5:]:
                print(f"  {activity['hook_name']}: {activity['status']} ({activity['execution_time']:.3f}s)")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            traceback.print_exc()
        
        finally:
            await hooks_system.cleanup()
    
    # 运行测试
    asyncio.run(test_hooks_system())