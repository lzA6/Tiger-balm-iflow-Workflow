#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能工具集成器 - Intelligent Tool Integrator
全能工作流V6核心工具 - 智能工具选择、集成和优化系统

功能特性:
- AI驱动的智能工具选择
- 工具性能实时监控和优化
- 自适应工具链配置
- 多工具协同工作
- 工具使用模式学习
- 量子增强工具优化
"""

import asyncio
import json
import logging
import time
import subprocess
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import yaml
import numpy as np
from pathlib import Path
import threading
import queue

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolStatus(Enum):
    """工具状态枚举"""
    AVAILABLE = "available"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    DISABLED = "disabled"


class ToolCategory(Enum):
    """工具类别枚举"""
    ANALYSIS = "analysis"          # 分析工具
    DEVELOPMENT = "development"    # 开发工具
    TESTING = "testing"           # 测试工具
    DEPLOYMENT = "deployment"     # 部署工具
    MONITORING = "monitoring"     # 监控工具
    OPTIMIZATION = "optimization" # 优化工具
    SECURITY = "security"         # 安全工具
    AUTOMATION = "automation"     # 自动化工具


@dataclass
class ToolMetrics:
    """工具性能指标"""
    execution_time: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    total_executions: int = 0
    avg_memory_usage: float = 0.0
    avg_cpu_usage: float = 0.0
    last_execution: Optional[datetime] = None
    performance_score: float = 1.0
    user_satisfaction: float = 5.0


@dataclass
class Tool:
    """工具数据类"""
    id: str
    name: str
    description: str
    category: ToolCategory
    executable_path: str
    version: str
    status: ToolStatus = ToolStatus.AVAILABLE
    capabilities: List[str] = field(default_factory=list)
    requirements: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: ToolMetrics = field(default_factory=ToolMetrics)
    dependencies: List[str] = field(default_factory=list)
    compatibility: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_compatible_with(self, environment: Dict[str, Any]) -> bool:
        """检查工具与环境兼容性"""
        for requirement, value in self.requirements.items():
            if requirement in environment:
                if not self._check_requirement(environment[requirement], value):
                    return False
        return True
    
    def _check_requirement(self, actual: Any, required: Any) -> bool:
        """检查单个需求"""
        if isinstance(required, dict):
            if "min" in required and actual < required["min"]:
                return False
            if "max" in required and actual > required["max"]:
                return False
            if "equals" in required and actual != required["equals"]:
                return False
            if "contains" in required and required["contains"] not in str(actual):
                return False
        else:
            return actual == required
        return True


class ToolPerformanceMonitor:
    """工具性能监控器"""
    
    def __init__(self):
        self.performance_history = {}
        self.real_time_metrics = {}
        self.monitoring_active = False
        self.monitor_thread = None
        self.metrics_queue = queue.Queue()
    
    def start_monitoring(self):
        """启动性能监控"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("工具性能监控已启动")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("工具性能监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 收集系统指标
                system_metrics = self._collect_system_metrics()
                
                # 处理指标队列
                while not self.metrics_queue.empty():
                    try:
                        metric = self.metrics_queue.get_nowait()
                        self._process_metric(metric)
                    except queue.Empty:
                        break
                
                # 休眠1秒
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"监控循环异常: {str(e)}")
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """收集系统指标"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "network_io": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
        }
    
    def record_tool_execution(self, tool_id: str, execution_time: float, 
                            success: bool, memory_usage: float = 0, cpu_usage: float = 0):
        """记录工具执行"""
        if tool_id not in self.performance_history:
            self.performance_history[tool_id] = []
        
        record = {
            "timestamp": datetime.now(),
            "execution_time": execution_time,
            "success": success,
            "memory_usage": memory_usage,
            "cpu_usage": cpu_usage
        }
        
        self.performance_history[tool_id].append(record)
        
        # 保持历史记录在合理范围内
        if len(self.performance_history[tool_id]) > 1000:
            self.performance_history[tool_id] = self.performance_history[tool_id][-500:]
    
    def get_tool_performance_summary(self, tool_id: str) -> Dict[str, Any]:
        """获取工具性能摘要"""
        if tool_id not in self.performance_history:
            return {}
        
        history = self.performance_history[tool_id]
        if not history:
            return {}
        
        successful_executions = [r for r in history if r["success"]]
        
        return {
            "total_executions": len(history),
            "successful_executions": len(successful_executions),
            "success_rate": len(successful_executions) / len(history),
            "avg_execution_time": np.mean([r["execution_time"] for r in successful_executions]) if successful_executions else 0,
            "avg_memory_usage": np.mean([r["memory_usage"] for r in successful_executions]) if successful_executions else 0,
            "avg_cpu_usage": np.mean([r["cpu_usage"] for r in successful_executions]) if successful_executions else 0,
            "last_execution": max([r["timestamp"] for r in history]).isoformat()
        }


class IntelligentToolSelector:
    """智能工具选择器"""
    
    def __init__(self):
        self.tool_performance_weights = {
            'execution_speed': 0.3,
            'success_rate': 0.25,
            'resource_efficiency': 0.2,
            'user_satisfaction': 0.15,
            'compatibility': 0.1
        }
        self.selection_history = []
        self.ml_model = None  # 可以集成机器学习模型
    
    def select_best_tool(self, available_tools: List[Tool], task_requirements: Dict[str, Any],
                        context: Dict[str, Any] = None) -> Optional[Tool]:
        """选择最佳工具"""
        if not available_tools:
            return None
        
        # 过滤兼容的工具
        compatible_tools = [tool for tool in available_tools 
                          if tool.is_compatible_with(context or {})]
        
        if not compatible_tools:
            logger.warning("没有找到兼容的工具")
            return None
        
        # 计算每个工具的适合度分数
        tool_scores = []
        for tool in compatible_tools:
            score = self._calculate_tool_score(tool, task_requirements, context)
            tool_scores.append((tool, score))
        
        # 按分数排序
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_tool = tool_scores[0][0]
        
        # 记录选择历史
        self.selection_history.append({
            "timestamp": datetime.now(),
            "selected_tool": best_tool.id,
            "task_requirements": task_requirements,
            "context": context,
            "score": tool_scores[0][1],
            "alternatives": [{"tool": t.id, "score": s} for t, s in tool_scores[1:4]]
        })
        
        return best_tool
    
    def _calculate_tool_score(self, tool: Tool, requirements: Dict[str, Any], 
                            context: Dict[str, Any] = None) -> float:
        """计算工具适合度分数"""
        scores = {}
        
        # 执行速度分数
        scores['execution_speed'] = self._calculate_speed_score(tool)
        
        # 成功率分数
        scores['success_rate'] = tool.metrics.success_rate
        
        # 资源效率分数
        scores['resource_efficiency'] = self._calculate_resource_efficiency_score(tool)
        
        # 用户满意度分数
        scores['user_satisfaction'] = tool.metrics.user_satisfaction / 5.0
        
        # 兼容性分数
        scores['compatibility'] = self._calculate_compatibility_score(tool, context)
        
        # 能力匹配分数
        capability_score = self._calculate_capability_match_score(tool, requirements)
        
        # 加权总分
        total_score = sum(scores[category] * weight 
                         for category, weight in self.tool_performance_weights.items())
        
        # 能力匹配作为额外加分
        total_score += capability_score * 0.2
        
        return min(max(total_score, 0.0), 1.0)
    
    def _calculate_speed_score(self, tool: Tool) -> float:
        """计算速度分数"""
        if tool.metrics.total_executions == 0:
            return 0.5  # 默认分数
        
        # 基于平均执行时间计算分数
        avg_time = tool.metrics.execution_time
        # 假设10秒为基准，越快分数越高
        return max(0, 1.0 - (avg_time / 10.0))
    
    def _calculate_resource_efficiency_score(self, tool: Tool) -> float:
        """计算资源效率分数"""
        if tool.metrics.total_executions == 0:
            return 0.5
        
        # 基于内存和CPU使用率计算效率
        memory_efficiency = max(0, 1.0 - (tool.metrics.avg_memory_usage / 1024))  # 1GB为基准
        cpu_efficiency = max(0, 1.0 - (tool.metrics.avg_cpu_usage / 100))
        
        return (memory_efficiency + cpu_efficiency) / 2
    
    def _calculate_compatibility_score(self, tool: Tool, context: Dict[str, Any]) -> float:
        """计算兼容性分数"""
        if not context:
            return 1.0
        
        compatible_count = 0
        total_checks = 0
        
        for env_key, env_value in context.items():
            if env_key in tool.requirements:
                total_checks += 1
                if tool.is_compatible_with({env_key: env_value}):
                    compatible_count += 1
        
        return compatible_count / total_checks if total_checks > 0 else 1.0
    
    def _calculate_capability_match_score(self, tool: Tool, requirements: Dict[str, Any]) -> float:
        """计算能力匹配分数"""
        if not requirements:
            return 1.0
        
        required_capabilities = requirements.get('capabilities', [])
        if not required_capabilities:
            return 1.0
        
        matching_capabilities = sum(1 for cap in required_capabilities 
                                  if cap in tool.capabilities)
        
        return matching_capabilities / len(required_capabilities)


class ToolOrchestrator:
    """工具编排器"""
    
    def __init__(self):
        self.active_executions = {}
        self.execution_queue = queue.Queue()
        self.max_concurrent_executions = 5
        self.orchestration_active = False
        self.orchestration_thread = None
    
    def start_orchestration(self):
        """启动工具编排"""
        if not self.orchestration_active:
            self.orchestration_active = True
            self.orchestration_thread = threading.Thread(target=self._orchestration_loop, daemon=True)
            self.orchestration_thread.start()
            logger.info("工具编排器已启动")
    
    def stop_orchestration(self):
        """停止工具编排"""
        self.orchestration_active = False
        if self.orchestration_thread:
            self.orchestration_thread.join()
        logger.info("工具编排器已停止")
    
    def _orchestration_loop(self):
        """编排循环"""
        while self.orchestration_active:
            try:
                # 检查并发执行限制
                if len(self.active_executions) < self.max_concurrent_executions:
                    # 从队列获取执行任务
                    try:
                        execution_task = self.execution_queue.get_nowait()
                        self._execute_tool_async(execution_task)
                    except queue.Empty:
                        pass
                
                # 检查已完成的执行
                self._cleanup_completed_executions()
                
                # 休眠0.1秒
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"编排循环异常: {str(e)}")
    
    def queue_tool_execution(self, tool: Tool, parameters: Dict[str, Any], 
                           priority: int = 0) -> str:
        """排队工具执行"""
        execution_id = f"{tool.id}_{int(time.time())}"
        
        execution_task = {
            "id": execution_id,
            "tool": tool,
            "parameters": parameters,
            "priority": priority,
            "queued_at": datetime.now(),
            "status": "queued"
        }
        
        self.execution_queue.put(execution_task)
        logger.info(f"工具执行已排队: {tool.name} (ID: {execution_id})")
        
        return execution_id
    
    def _execute_tool_async(self, execution_task: Dict[str, Any]):
        """异步执行工具"""
        execution_id = execution_task["id"]
        tool = execution_task["tool"]
        parameters = execution_task["parameters"]
        
        # 更新工具状态
        tool.status = ToolStatus.BUSY
        
        # 创建执行线程
        execution_thread = threading.Thread(
            target=self._execute_tool,
            args=(execution_task,),
            daemon=True
        )
        
        self.active_executions[execution_id] = {
            "task": execution_task,
            "thread": execution_thread,
            "started_at": datetime.now()
        }
        
        execution_thread.start()
    
    def _execute_tool(self, execution_task: Dict[str, Any]):
        """执行工具"""
        execution_id = execution_task["id"]
        tool = execution_task["tool"]
        parameters = execution_task["parameters"]
        
        try:
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # 执行工具命令
            command = self._build_command(tool, parameters)
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            execution_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_usage = end_memory - start_memory
            
            success = result.returncode == 0
            
            # 更新工具指标
            tool.metrics.total_executions += 1
            tool.metrics.last_execution = datetime.now()
            tool.metrics.execution_time = execution_time
            tool.metrics.avg_memory_usage = (tool.metrics.avg_memory_usage * (tool.metrics.total_executions - 1) + memory_usage) / tool.metrics.total_executions
            
            if success:
                tool.metrics.success_rate = (tool.metrics.success_rate * (tool.metrics.total_executions - 1) + 1.0) / tool.metrics.total_executions
            else:
                tool.metrics.error_count += 1
                tool.metrics.success_rate = (tool.metrics.success_rate * (tool.metrics.total_executions - 1)) / tool.metrics.total_executions
            
            # 记录执行结果
            execution_result = {
                "execution_id": execution_id,
                "success": success,
                "execution_time": execution_time,
                "memory_usage": memory_usage,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
            # 保存结果
            self._save_execution_result(execution_id, execution_result)
            
            logger.info(f"工具执行完成: {tool.name} (ID: {execution_id}, 成功: {success})")
            
        except subprocess.TimeoutExpired:
            logger.error(f"工具执行超时: {tool.name} (ID: {execution_id})")
            tool.metrics.error_count += 1
            
        except Exception as e:
            logger.error(f"工具执行异常: {tool.name} (ID: {execution_id}) - {str(e)}")
            tool.metrics.error_count += 1
            
        finally:
            # 恢复工具状态
            tool.status = ToolStatus.AVAILABLE
    
    def _build_command(self, tool: Tool, parameters: Dict[str, Any]) -> str:
        """构建工具执行命令"""
        command = tool.executable_path
        
        # 添加参数
        for param_name, param_value in parameters.items():
            if param_name in tool.parameters:
                param_config = tool.parameters[param_name]
                if param_config.get("type") == "flag" and param_value:
                    command += f" --{param_name}"
                elif param_config.get("type") == "option":
                    command += f" --{param_name} {param_value}"
                else:
                    command += f" {param_value}"
        
        return command
    
    def _save_execution_result(self, execution_id: str, result: Dict[str, Any]):
        """保存执行结果"""
        result_dir = Path("./results/tool_executions")
        result_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = result_dir / f"{execution_id}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
    
    def _cleanup_completed_executions(self):
        """清理已完成的执行"""
        completed_executions = []
        
        for execution_id, execution_info in self.active_executions.items():
            if not execution_info["thread"].is_alive():
                completed_executions.append(execution_id)
        
        for execution_id in completed_executions:
            del self.active_executions[execution_id]
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """获取执行状态"""
        if execution_id in self.active_executions:
            execution_info = self.active_executions[execution_id]
            return {
                "status": "running",
                "started_at": execution_info["started_at"].isoformat(),
                "tool": execution_info["task"]["tool"].name
            }
        
        # 检查结果文件
        result_file = Path(f"./results/tool_executions/{execution_id}.json")
        if result_file.exists():
            with open(result_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            return {
                "status": "completed",
                "result": result
            }
        
        return None


class IntelligentToolIntegrator:
    """智能工具集成器"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.performance_monitor = ToolPerformanceMonitor()
        self.tool_selector = IntelligentToolSelector()
        self.orchestrator = ToolOrchestrator()
        self.quantum_optimizer = None
        
        # 启动监控和编排
        self.performance_monitor.start_monitoring()
        self.orchestrator.start_orchestration()
        
        # 加载工具配置
        self._load_tools_configuration()
    
    def _load_tools_configuration(self):
        """加载工具配置"""
        config_file = Path("./config/tools-config.yaml")
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            for tool_config in config.get('tools', []):
                tool = Tool(
                    id=tool_config['id'],
                    name=tool_config['name'],
                    description=tool_config['description'],
                    category=ToolCategory(tool_config['category']),
                    executable_path=tool_config['executable_path'],
                    version=tool_config['version'],
                    capabilities=tool_config.get('capabilities', []),
                    requirements=tool_config.get('requirements', {}),
                    parameters=tool_config.get('parameters', {}),
                    dependencies=tool_config.get('dependencies', []),
                    compatibility=tool_config.get('compatibility', []),
                    metadata=tool_config.get('metadata', {})
                )
                self.tools[tool.id] = tool
            
            logger.info(f"已加载 {len(self.tools)} 个工具配置")
    
    def register_tool(self, tool: Tool):
        """注册工具"""
        self.tools[tool.id] = tool
        logger.info(f"工具已注册: {tool.name}")
    
    def execute_task(self, task_requirements: Dict[str, Any], 
                    context: Dict[str, Any] = None) -> Optional[str]:
        """执行任务"""
        # 获取可用工具
        available_tools = [tool for tool in self.tools.values() 
                         if tool.status == ToolStatus.AVAILABLE]
        
        if not available_tools:
            logger.warning("没有可用的工具")
            return None
        
        # 选择最佳工具
        best_tool = self.tool_selector.select_best_tool(available_tools, task_requirements, context)
        
        if not best_tool:
            logger.warning("没有找到合适的工具")
            return None
        
        # 排队执行
        execution_id = self.orchestrator.queue_tool_execution(
            best_tool, 
            task_requirements.get('parameters', {}),
            task_requirements.get('priority', 0)
        )
        
        return execution_id
    
    def get_tool_status(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """获取工具状态"""
        if tool_id not in self.tools:
            return None
        
        tool = self.tools[tool_id]
        performance_summary = self.performance_monitor.get_tool_performance_summary(tool_id)
        
        return {
            "id": tool.id,
            "name": tool.name,
            "status": tool.status.value,
            "category": tool.category.value,
            "version": tool.version,
            "capabilities": tool.capabilities,
            "metrics": {
                "total_executions": tool.metrics.total_executions,
                "success_rate": tool.metrics.success_rate,
                "error_count": tool.metrics.error_count,
                "performance_score": tool.metrics.performance_score,
                "user_satisfaction": tool.metrics.user_satisfaction
            },
            "performance_summary": performance_summary,
            "last_execution": tool.metrics.last_execution.isoformat() if tool.metrics.last_execution else None
        }
    
    def get_all_tools_status(self) -> List[Dict[str, Any]]:
        """获取所有工具状态"""
        return [self.get_tool_status(tool_id) for tool_id in self.tools.keys()]
    
    def update_tool_metrics(self, tool_id: str, metrics: Dict[str, Any]):
        """更新工具指标"""
        if tool_id in self.tools:
            tool = self.tools[tool_id]
            for key, value in metrics.items():
                if hasattr(tool.metrics, key):
                    setattr(tool.metrics, key, value)
    
    def optimize_tool_selection(self, historical_data: List[Dict[str, Any]]):
        """优化工具选择"""
        # 基于历史数据优化选择策略
        if len(historical_data) < 10:
            return
        
        # 分析成功模式
        successful_selections = [d for d in historical_data if d.get('success', False)]
        
        if successful_selections:
            # 更新权重
            self._update_selection_weights(successful_selections)
    
    def _update_selection_weights(self, successful_selections: List[Dict[str, Any]]):
        """更新选择权重"""
        # 简化的权重更新逻辑
        # 实际实现中可以使用更复杂的机器学习算法
        
        performance_factors = {
            'execution_speed': [],
            'success_rate': [],
            'resource_efficiency': [],
            'user_satisfaction': [],
            'compatibility': []
        }
        
        for selection in successful_selections:
            if 'performance_factors' in selection:
                for factor, value in selection['performance_factors'].items():
                    if factor in performance_factors:
                        performance_factors[factor].append(value)
        
        # 计算新的权重
        for factor, values in performance_factors.items():
            if values:
                avg_value = np.mean(values)
                # 根据平均表现调整权重
                self.tool_selector.tool_performance_weights[factor] *= (1.0 + avg_value * 0.1)
        
        # 归一化权重
        total_weight = sum(self.tool_selector.tool_performance_weights.values())
        for factor in self.tool_selector.tool_performance_weights:
            self.tool_selector.tool_performance_weights[factor] /= total_weight
    
    def shutdown(self):
        """关闭集成器"""
        self.performance_monitor.stop_monitoring()
        self.orchestrator.stop_orchestration()
        logger.info("智能工具集成器已关闭")


# 示例使用
async def main():
    """主函数示例"""
    # 创建工具集成器
    integrator = IntelligentToolIntegrator()
    
    # 注册一些示例工具
    linter_tool = Tool(
        id="eslint",
        name="ESLint",
        description="JavaScript代码检查工具",
        category=ToolCategory.ANALYSIS,
        executable_path="eslint",
        version="8.0.0",
        capabilities=["code_analysis", "style_checking", "error_detection"],
        requirements={"node_version": {"min": "14.0.0"}},
        parameters={
            "ext": {"type": "option", "description": "文件扩展名"},
            "fix": {"type": "flag", "description": "自动修复"}
        }
    )
    
    test_tool = Tool(
        id="jest",
        name="Jest",
        description="JavaScript测试框架",
        category=ToolCategory.TESTING,
        executable_path="jest",
        version="29.0.0",
        capabilities=["unit_testing", "integration_testing", "coverage"],
        requirements={"node_version": {"min": "14.0.0"}},
        parameters={
            "coverage": {"type": "flag", "description": "生成覆盖率报告"},
            "watch": {"type": "flag", "description": "监听模式"}
        }
    )
    
    integrator.register_tool(linter_tool)
    integrator.register_tool(test_tool)
    
    # 执行代码检查任务
    lint_task = {
        "capabilities": ["code_analysis", "style_checking"],
        "parameters": {"ext": ".js", "fix": True},
        "priority": 1
    }
    
    context = {
        "node_version": "18.0.0",
        "os": "linux"
    }
    
    execution_id = integrator.execute_task(lint_task, context)
    
    if execution_id:
        print(f"任务已提交执行，ID: {execution_id}")
        
        # 等待执行完成
        import time
        time.sleep(5)
        
        # 检查执行状态
        status = integrator.get_execution_status(execution_id)
        print(f"执行状态: {status}")
    
    # 获取所有工具状态
    tools_status = integrator.get_all_tools_status()
    print(f"工具状态: {json.dumps(tools_status, indent=2, ensure_ascii=False)}")
    
    # 关闭集成器
    integrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())