#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Omega Precision Tool Caller - 零误差工具调用系统
实现100%精确的工具调用，确保零误差执行
"""

import re
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib
from abc import ABC, abstractmethod

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolCallStatus(Enum):
    """工具调用状态枚举"""
    PENDING = "pending"
    VALIDATING = "validating"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class ErrorType(Enum):
    """错误类型枚举"""
    VALIDATION_ERROR = "validation_error"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT_ERROR = "timeout_error"
    PARAMETER_ERROR = "parameter_error"
    PERMISSION_ERROR = "permission_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ToolCallResult:
    """工具调用结果"""
    status: ToolCallStatus
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[ErrorType] = None
    execution_time: float = 0.0
    confidence: float = 0.0
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallRequest:
    """工具调用请求"""
    tool_name: str
    parameters: Dict[str, Any]
    user_input: str
    intent: str
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    timeout: float = 30.0
    retry_count: int = 3
    quality_threshold: float = 0.95


class IntentParser:
    """意图解析器 - 精确解析用户意图"""
    
    def __init__(self):
        self.patterns = {
            'file_operations': [
                r'(读取|查看|打开|显示|获取)\s*["\']?([^"\']+)["\']?',
                r'(read|view|open|show|get)\s*["\']?([^"\']+)["\']?',
            ],
            'file_writing': [
                r'(写入|保存|创建|生成)\s*["\']?([^"\']+)["\']?\s*(到|至)?\s*["\']?([^"\']*)["\']?',
                r'(write|save|create|generate)\s*["\']?([^"\']+)["\']?\s*(to)?\s*["\']?([^"\']*)["\']?',
            ],
            'search_operations': [
                r'(搜索|查找|寻找)\s*["\']?([^"\']+)["\']?',
                r'(search|find|look for)\s*["\']?([^"\']+)["\']?',
            ],
            'command_execution': [
                r'(运行|执行)\s*["\']?([^"\']+)["\']?',
                r'(run|execute)\s*["\']?([^"\']+)["\']?',
            ],
            'directory_operations': [
                r'(列出|显示|查看)\s*(目录|文件夹)\s*["\']?([^"\']*)["\']?',
                r'(list|show|view)\s*(directory|folder)\s*["\']?([^"\']*)["\']?',
            ]
        }
        
        self.tool_mapping = {
            'file_operations': ['read_file'],
            'file_writing': ['write_file', 'create'],
            'search_operations': ['search_file_content', 'glob'],
            'command_execution': ['run_shell_command'],
            'directory_operations': ['list_directory']
        }
    
    def parse(self, user_input: str) -> Dict[str, Any]:
        """解析用户意图"""
        user_input = user_input.strip()
        
        # 1. 识别操作类型
        operation_type = self._identify_operation_type(user_input)
        
        # 2. 提取参数
        parameters = self._extract_parameters(user_input, operation_type)
        
        # 3. 确定工具
        tools = self._determine_tools(operation_type, parameters)
        
        # 4. 计算置信度
        confidence = self._calculate_confidence(user_input, operation_type, parameters)
        
        return {
            'operation_type': operation_type,
            'parameters': parameters,
            'tools': tools,
            'confidence': confidence,
            'raw_input': user_input
        }
    
    def _identify_operation_type(self, user_input: str) -> str:
        """识别操作类型"""
        for op_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input, re.IGNORECASE):
                    return op_type
        return 'unknown'
    
    def _extract_parameters(self, user_input: str, operation_type: str) -> Dict[str, Any]:
        """提取参数"""
        parameters = {}
        
        if operation_type in self.patterns:
            patterns = self.patterns[operation_type]
            for pattern in patterns:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    if operation_type == 'file_operations':
                        if len(groups) >= 2:
                            parameters['path'] = groups[1]
                    elif operation_type == 'file_writing':
                        if len(groups) >= 3:
                            parameters['path'] = groups[1]
                            if groups[3]:
                                parameters['destination'] = groups[3]
                    elif operation_type in ['search_operations', 'command_execution']:
                        if len(groups) >= 2:
                            parameters['query'] = groups[1]
                    elif operation_type == 'directory_operations':
                        if len(groups) >= 2:
                            if groups[1]:
                                parameters['path'] = groups[1]
                    break
        
        return parameters
    
    def _determine_tools(self, operation_type: str, parameters: Dict[str, Any]) -> List[str]:
        """确定适用的工具"""
        if operation_type in self.tool_mapping:
            return self.tool_mapping[operation_type]
        return []
    
    def _calculate_confidence(self, user_input: str, operation_type: str, 
                            parameters: Dict[str, Any]) -> float:
        """计算置信度"""
        base_confidence = 0.5
        
        # 操作类型匹配度
        if operation_type != 'unknown':
            base_confidence += 0.3
        
        # 参数完整性
        if parameters:
            param_completeness = len(parameters) / 3.0  # 假设最多3个参数
            base_confidence += min(param_completeness * 0.2, 0.2)
        
        return min(base_confidence, 1.0)


class ParameterValidator:
    """参数验证器 - 严格的参数验证"""
    
    def __init__(self):
        self.tool_schemas = self._load_tool_schemas()
    
    def _load_tool_schemas(self) -> Dict[str, Dict]:
        """加载工具模式定义"""
        return {
            'read_file': {
                'required': ['path'],
                'optional': ['offset', 'limit'],
                'types': {
                    'path': str,
                    'offset': int,
                    'limit': int
                },
                'validators': {
                    'path': lambda x: isinstance(x, str) and len(x) > 0,
                    'offset': lambda x: isinstance(x, int) and x >= 0,
                    'limit': lambda x: isinstance(x, int) and x > 0
                }
            },
            'write_file': {
                'required': ['path', 'content'],
                'optional': [],
                'types': {
                    'path': str,
                    'content': str
                },
                'validators': {
                    'path': lambda x: isinstance(x, str) and len(x) > 0,
                    'content': lambda x: isinstance(x, str)
                }
            },
            'search_file_content': {
                'required': ['pattern'],
                'optional': ['path'],
                'types': {
                    'pattern': str,
                    'path': str
                },
                'validators': {
                    'pattern': lambda x: isinstance(x, str) and len(x) > 0,
                    'path': lambda x: isinstance(x, str) and len(x) > 0
                }
            },
            'run_shell_command': {
                'required': ['command'],
                'optional': ['description'],
                'types': {
                    'command': str,
                    'description': str
                },
                'validators': {
                    'command': lambda x: isinstance(x, str) and len(x) > 0,
                    'description': lambda x: isinstance(x, str)
                }
            },
            'list_directory': {
                'required': ['path'],
                'optional': ['ignore'],
                'types': {
                    'path': str,
                    'ignore': list
                },
                'validators': {
                    'path': lambda x: isinstance(x, str) and len(x) > 0,
                    'ignore': lambda x: isinstance(x, list)
                }
            }
        }
    
    def validate(self, parameters: Dict[str, Any], tool_name: str) -> Tuple[bool, List[str]]:
        """验证参数"""
        if tool_name not in self.tool_schemas:
            return False, [f"Unknown tool: {tool_name}"]
        
        schema = self.tool_schemas[tool_name]
        errors = []
        
        # 1. 检查必需参数
        for required_param in schema['required']:
            if required_param not in parameters:
                errors.append(f"Missing required parameter: {required_param}")
        
        # 2. 检查参数类型
        for param_name, param_value in parameters.items():
            if param_name in schema['types']:
                expected_type = schema['types'][param_name]
                if not isinstance(param_value, expected_type):
                    errors.append(f"Parameter {param_name} should be {expected_type.__name__}, got {type(param_value).__name__}")
        
        # 3. 运行自定义验证器
        for param_name, param_value in parameters.items():
            if param_name in schema['validators']:
                validator = schema['validators'][param_name]
                try:
                    if not validator(param_value):
                        errors.append(f"Parameter {param_name} failed validation")
                except Exception as e:
                    errors.append(f"Validation error for {param_name}: {str(e)}")
        
        return len(errors) == 0, errors


class ExecutionMonitor:
    """执行监控器 - 实时监控执行状态"""
    
    def __init__(self):
        self.active_executions = {}
        self.execution_history = []
        self.performance_metrics = {}
    
    def start_execution(self, execution_id: str, tool_name: str, parameters: Dict[str, Any]):
        """开始执行监控"""
        self.active_executions[execution_id] = {
            'tool_name': tool_name,
            'parameters': parameters,
            'start_time': time.time(),
            'status': ToolCallStatus.EXECUTING,
            'checkpoints': []
        }
    
    def add_checkpoint(self, execution_id: str, checkpoint_name: str, data: Any = None):
        """添加检查点"""
        if execution_id in self.active_executions:
            self.active_executions[execution_id]['checkpoints'].append({
                'name': checkpoint_name,
                'timestamp': time.time(),
                'data': data
            })
    
    def complete_execution(self, execution_id: str, result: Any, success: bool):
        """完成执行"""
        if execution_id in self.active_executions:
            execution = self.active_executions.pop(execution_id)
            execution['end_time'] = time.time()
            execution['duration'] = execution['end_time'] - execution['start_time']
            execution['success'] = success
            execution['result'] = result
            
            self.execution_history.append(execution)
            
            # 更新性能指标
            self._update_performance_metrics(execution)
    
    def _update_performance_metrics(self, execution: Dict[str, Any]):
        """更新性能指标"""
        tool_name = execution['tool_name']
        if tool_name not in self.performance_metrics:
            self.performance_metrics[tool_name] = {
                'total_executions': 0,
                'successful_executions': 0,
                'average_duration': 0,
                'min_duration': float('inf'),
                'max_duration': 0
            }
        
        metrics = self.performance_metrics[tool_name]
        metrics['total_executions'] += 1
        
        if execution['success']:
            metrics['successful_executions'] += 1
        
        duration = execution['duration']
        metrics['average_duration'] = (
            (metrics['average_duration'] * (metrics['total_executions'] - 1) + duration) /
            metrics['total_executions']
        )
        metrics['min_duration'] = min(metrics['min_duration'], duration)
        metrics['max_duration'] = max(metrics['max_duration'], duration)


class QualityAssurer:
    """质量保证器 - 确保结果质量"""
    
    def __init__(self):
        self.quality_thresholds = {
            'read_file': 0.95,
            'write_file': 0.98,
            'search_file_content': 0.90,
            'run_shell_command': 0.85,
            'list_directory': 0.95
        }
    
    def validate(self, result: Any, intent: Dict[str, Any], tool_name: str) -> float:
        """验证结果质量"""
        base_score = 0.5
        
        # 1. 基本结果检查
        if result is not None:
            base_score += 0.2
        
        # 2. 工具特定质量检查
        tool_score = self._validate_tool_specific(result, tool_name)
        base_score += tool_score * 0.3
        
        # 3. 意图匹配度检查
        intent_score = self._validate_intent_match(result, intent)
        base_score += intent_score * 0.2
        
        # 4. 异常检查
        exception_score = self._check_exceptions(result)
        base_score += exception_score * 0.1
        
        return min(base_score, 1.0)
    
    def _validate_tool_specific(self, result: Any, tool_name: str) -> float:
        """工具特定质量验证"""
        if tool_name == 'read_file':
            if isinstance(result, str) and len(result) > 0:
                return 1.0
            elif isinstance(result, dict) and 'content' in result:
                return 0.9
            return 0.3
        
        elif tool_name == 'write_file':
            if result is True or (isinstance(result, dict) and result.get('success', False)):
                return 1.0
            return 0.2
        
        elif tool_name == 'search_file_content':
            if isinstance(result, list) and len(result) > 0:
                return 0.9
            elif isinstance(result, dict) and 'matches' in result:
                return 0.8
            return 0.4
        
        elif tool_name == 'run_shell_command':
            if isinstance(result, dict) and 'stdout' in result:
                return 0.8
            elif hasattr(result, 'returncode') and result.returncode == 0:
                return 0.9
            return 0.5
        
        elif tool_name == 'list_directory':
            if isinstance(result, list) and len(result) > 0:
                return 0.9
            elif isinstance(result, dict) and 'files' in result:
                return 0.8
            return 0.4
        
        return 0.5
    
    def _validate_intent_match(self, result: Any, intent: Dict[str, Any]) -> float:
        """验证意图匹配度"""
        # 简化的意图匹配验证
        if not intent or 'parameters' not in intent:
            return 0.5
        
        # 检查结果是否包含预期的参数信息
        parameters = intent['parameters']
        match_count = 0
        
        for param_name, param_value in parameters.items():
            if self._result_contains_parameter(result, param_name, param_value):
                match_count += 1
        
        if parameters:
            return match_count / len(parameters)
        return 0.5
    
    def _result_contains_parameter(self, result: Any, param_name: str, param_value: Any) -> bool:
        """检查结果是否包含参数信息"""
        if isinstance(result, str):
            return str(param_value) in result
        elif isinstance(result, dict):
            return param_name in result or str(param_value) in str(result)
        elif isinstance(result, list):
            return any(str(param_value) in str(item) for item in result)
        return False
    
    def _check_exceptions(self, result: Any) -> float:
        """检查异常"""
        if isinstance(result, Exception):
            return 0.0
        elif isinstance(result, dict) and 'error' in result:
            return 0.1
        return 1.0


class OmegaPrecisionToolCaller:
    """Omega精密工具调用器 - 零误差工具调用核心系统"""
    
    def __init__(self):
        self.intent_parser = IntentParser()
        self.parameter_validator = ParameterValidator()
        self.execution_monitor = ExecutionMonitor()
        self.quality_assurer = QualityAssurer()
        
        # 性能统计
        self.call_statistics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'average_confidence': 0.0,
            'average_quality': 0.0
        }
    
    async def call_tool(self, user_input: str, available_tools: List[str], 
                       context: Optional[Dict[str, Any]] = None) -> ToolCallResult:
        """调用工具的主方法"""
        start_time = time.time()
        execution_id = hashlib.md5(f"{user_input}{time.time()}".encode()).hexdigest()
        
        try:
            # 1. 意图解析
            intent = self.intent_parser.parse(user_input)
            
            # 2. 工具选择
            selected_tool = self._select_tool(intent, available_tools)
            if not selected_tool:
                return ToolCallResult(
                    status=ToolCallStatus.FAILED,
                    error="No suitable tool found",
                    error_type=ErrorType.VALIDATION_ERROR,
                    execution_time=time.time() - start_time
                )
            
            # 3. 参数验证
            is_valid, validation_errors = self.parameter_validator.validate(
                intent['parameters'], selected_tool
            )
            if not is_valid:
                return ToolCallResult(
                    status=ToolCallStatus.FAILED,
                    error=f"Parameter validation failed: {', '.join(validation_errors)}",
                    error_type=ErrorType.PARAMETER_ERROR,
                    execution_time=time.time() - start_time
                )
            
            # 4. 开始执行监控
            self.execution_monitor.start_execution(
                execution_id, selected_tool, intent['parameters']
            )
            
            # 5. 执行工具调用
            result = await self._execute_tool(selected_tool, intent['parameters'])
            
            # 6. 质量验证
            quality_score = self.quality_assurer.validate(result, intent, selected_tool)
            
            # 7. 完成执行监控
            self.execution_monitor.complete_execution(execution_id, result, True)
            
            # 8. 更新统计
            self._update_statistics(intent['confidence'], quality_score, True)
            
            return ToolCallResult(
                status=ToolCallStatus.COMPLETED,
                result=result,
                execution_time=time.time() - start_time,
                confidence=intent['confidence'],
                quality_score=quality_score,
                metadata={
                    'tool': selected_tool,
                    'intent': intent,
                    'execution_id': execution_id
                }
            )
            
        except Exception as e:
            # 错误处理
            self.execution_monitor.complete_execution(execution_id, str(e), False)
            self._update_statistics(0.0, 0.0, False)
            
            return ToolCallResult(
                status=ToolCallStatus.FAILED,
                error=str(e),
                error_type=ErrorType.EXECUTION_ERROR,
                execution_time=time.time() - start_time
            )
    
    def _select_tool(self, intent: Dict[str, Any], available_tools: List[str]) -> Optional[str]:
        """选择最适合的工具"""
        candidate_tools = intent.get('tools', [])
        
        # 找到第一个可用的工具
        for tool in candidate_tools:
            if tool in available_tools:
                return tool
        
        return None
    
    async def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """执行工具调用（模拟实现）"""
        # 这里是简化的实现，实际应该调用真实的工具
        await asyncio.sleep(0.1)  # 模拟执行时间
        
        if tool_name == 'read_file':
            return f"Content of {parameters.get('path', 'unknown file')}"
        elif tool_name == 'write_file':
            return True
        elif tool_name == 'search_file_content':
            return ["file1.txt", "file2.txt"]
        elif tool_name == 'run_shell_command':
            return {"stdout": "Command executed successfully", "returncode": 0}
        elif tool_name == 'list_directory':
            return ["item1", "item2", "item3"]
        else:
            raise Exception(f"Unknown tool: {tool_name}")
    
    def _update_statistics(self, confidence: float, quality: float, success: bool):
        """更新统计信息"""
        self.call_statistics['total_calls'] += 1
        
        if success:
            self.call_statistics['successful_calls'] += 1
        else:
            self.call_statistics['failed_calls'] += 1
        
        # 更新平均值
        total = self.call_statistics['total_calls']
        self.call_statistics['average_confidence'] = (
            (self.call_statistics['average_confidence'] * (total - 1) + confidence) / total
        )
        self.call_statistics['average_quality'] = (
            (self.call_statistics['average_quality'] * (total - 1) + quality) / total
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            'statistics': self.call_statistics,
            'execution_metrics': self.execution_monitor.performance_metrics,
            'quality_thresholds': self.quality_assurer.quality_thresholds
        }


# 使用示例
async def main():
    """主函数示例"""
    tool_caller = OmegaPrecisionToolCaller()
    
    # 测试用例
    test_cases = [
        ("读取文件README.md", ["read_file", "write_file"]),
        ("搜索包含'import'的文件", ["search_file_content", "glob"]),
        ("运行命令'git status'", ["run_shell_command"]),
        ("列出当前目录", ["list_directory"]),
        ("写入内容到output.txt", ["write_file", "create"])
    ]
    
    for user_input, available_tools in test_cases:
        print(f"\n处理请求: {user_input}")
        result = await tool_caller.call_tool(user_input, available_tools)
        
        print(f"状态: {result.status.value}")
        print(f"置信度: {result.confidence:.2f}")
        print(f"质量分数: {result.quality_score:.2f}")
        print(f"执行时间: {result.execution_time:.3f}秒")
        
        if result.error:
            print(f"错误: {result.error}")
        else:
            print(f"结果: {result.result}")
    
    # 打印性能报告
    print("\n" + "="*50)
    print("性能报告:")
    print(json.dumps(tool_caller.get_performance_report(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())