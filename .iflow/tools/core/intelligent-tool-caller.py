#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全能工作流V5 - 智能工具调用系统
Universal Workflow V5 - Intelligent Tool Calling System

确保100%兼容所有LLM模型的智能工具调用机制
"""

import os
import sys
import json
import yaml
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import inspect
import traceback
from abc import ABC, abstractmethod

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """模型类型枚举"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    ALIBABA = "alibaba"
    ZHIPU = "zhipu"
    DEEPSEEK = "deepseek"
    MOONSHOT = "moonshot"
    LOCAL = "local"
    UNKNOWN = "unknown"

class ToolCallFormat(Enum):
    """工具调用格式枚举"""
    OPENAI_FUNCTION = "openai_function"
    OPENAI_TOOL = "openai_tool"
    ANTHROPIC_TOOL = "anthropic_tool"
    XML_TOOL = "xml_tool"
    JSON_TOOL = "json_tool"
    NATURAL = "natural"

@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    model_compatibility: List[ModelType] = field(default_factory=lambda: list(ModelType))
    complexity: int = 1  # 1-5, 5最复杂
    parallel_safe: bool = True
    timeout: int = 30

@dataclass
class ToolCall:
    """工具调用"""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None
    format: ToolCallFormat = ToolCallFormat.OPENAI_TOOL
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolResult:
    """工具调用结果"""
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class ToolCallAdapter(ABC):
    """工具调用适配器基类"""
    
    @abstractmethod
    def format_tool_call(self, tool: ToolDefinition, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """格式化工具调用"""
        pass
    
    @abstractmethod
    def parse_tool_call(self, call_data: Dict[str, Any]) -> ToolCall:
        """解析工具调用"""
        pass
    
    @abstractmethod
    def format_tool_result(self, result: ToolResult) -> Dict[str, Any]:
        """格式化工具结果"""
        pass

class OpenAIToolAdapter(ToolCallAdapter):
    """OpenAI工具调用适配器"""
    
    def format_tool_call(self, tool: ToolDefinition, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """格式化工具调用"""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
        }
    
    def parse_tool_call(self, call_data: Dict[str, Any]) -> ToolCall:
        """解析工具调用"""
        if "function" in call_data:
            function = call_data["function"]
            return ToolCall(
                tool_name=function["name"],
                arguments=function.get("arguments", {}),
                call_id=call_data.get("id"),
                format=ToolCallFormat.OPENAI_FUNCTION
            )
        return ToolCall(
            tool_name=call_data.get("name", ""),
            arguments=call_data.get("arguments", {}),
            call_id=call_data.get("id"),
            format=ToolCallFormat.OPENAI_TOOL
        )
    
    def format_tool_result(self, result: ToolResult) -> Dict[str, Any]:
        """格式化工具结果"""
        return {
            "tool_call_id": result.metadata.get("call_id"),
            "role": "tool",
            "content": json.dumps(result.result) if result.success else result.error
        }

class AnthropicToolAdapter(ToolCallAdapter):
    """Anthropic工具调用适配器"""
    
    def format_tool_call(self, tool: ToolDefinition, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """格式化工具调用"""
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters
        }
    
    def parse_tool_call(self, call_data: Dict[str, Any]) -> ToolCall:
        """解析工具调用"""
        return ToolCall(
            tool_name=call_data.get("name", ""),
            arguments=call_data.get("input", {}),
            call_id=call_data.get("id"),
            format=ToolCallFormat.ANTHROPIC_TOOL
        )
    
    def format_tool_result(self, result: ToolResult) -> Dict[str, Any]:
        """格式化工具结果"""
        return {
            "tool_use_id": result.metadata.get("call_id"),
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result.result) if result.success else result.error
                }
            ]
        }

class XMLToolAdapter(ToolCallAdapter):
    """XML工具调用适配器（适用于Claude等）"""
    
    def format_tool_call(self, tool: ToolDefinition, arguments: Dict[str, Any]) -> str:
        """格式化工具调用"""
        args_str = " ".join([f'{k}="{v}"' for k, v in arguments.items()])
        return f"<{tool.name} {args_str}/>"
    
    def parse_tool_call(self, call_data: str) -> ToolCall:
        """解析工具调用"""
        import re
        
        # 简单的XML解析
        pattern = r'<(\w+)([^>]*)/?>'
        match = re.search(pattern, call_data)
        
        if match:
            tool_name = match.group(1)
            args_str = match.group(2)
            
            # 解析参数
            arguments = {}
            arg_pattern = r'(\w+)="([^"]*)"'
            for arg_match in re.finditer(arg_pattern, args_str):
                arguments[arg_match.group(1)] = arg_match.group(2)
            
            return ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                format=ToolCallFormat.XML_TOOL
            )
        
        return ToolCall(tool_name="", arguments={})
    
    def format_tool_result(self, result: ToolResult) -> str:
        """格式化工具结果"""
        if result.success:
            return f"<result>{json.dumps(result.result)}</result>"
        else:
            return f"<error>{result.error}</error>"

class IntelligentToolCaller:
    """智能工具调用系统"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.tools: Dict[str, ToolDefinition] = {}
        self.adapters: Dict[ModelType, ToolCallAdapter] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # 初始化适配器
        self._initialize_adapters()
        
        # 注册内置工具
        self._register_builtin_tools()
        
        logger.info("智能工具调用系统初始化完成")
    
    def _initialize_adapters(self):
        """初始化适配器"""
        self.adapters[ModelType.OPENAI] = OpenAIToolAdapter()
        self.adapters[ModelType.ANTHROPIC] = AnthropicToolAdapter()
        self.adapters[ModelType.GOOGLE] = OpenAIToolAdapter()  # Google使用OpenAI格式
        self.adapters[ModelType.ALIBABA] = OpenAIToolAdapter()  # 阿里使用OpenAI格式
        self.adapters[ModelType.ZHIPU] = OpenAIToolAdapter()  # 智谱使用OpenAI格式
        self.adapters[ModelType.DEEPSEEK] = OpenAIToolAdapter()  # DeepSeek使用OpenAI格式
        self.adapters[ModelType.MOONSHOT] = OpenAIToolAdapter()  # Moonshot使用OpenAI格式
        self.adapters[ModelType.LOCAL] = OpenAIToolAdapter()  # 本地模型使用OpenAI格式
        self.adapters[ModelType.UNKNOWN] = OpenAIToolAdapter()  # 默认使用OpenAI格式
    
    def _register_builtin_tools(self):
        """注册内置工具"""
        # 文件操作工具
        self.register_tool(
            "read_file",
            "读取文件内容",
            {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "要读取的文件路径"
                    }
                },
                "required": ["file_path"]
            },
            self._read_file,
            category="file",
            tags=["read", "file"]
        )
        
        self.register_tool(
            "write_file",
            "写入文件内容",
            {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "要写入的文件路径"
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的内容"
                    }
                },
                "required": ["file_path", "content"]
            },
            self._write_file,
            category="file",
            tags=["write", "file"]
        )
        
        self.register_tool(
            "list_directory",
            "列出目录内容",
            {
                "type": "object",
                "properties": {
                    "directory_path": {
                        "type": "string",
                        "description": "要列出的目录路径"
                    }
                },
                "required": ["directory_path"]
            },
            self._list_directory,
            category="file",
            tags=["list", "directory"]
        )
        
        # 搜索工具
        self.register_tool(
            "search_files",
            "搜索文件内容",
            {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "搜索模式"
                    },
                    "directory": {
                        "type": "string",
                        "description": "搜索目录"
                    }
                },
                "required": ["pattern"]
            },
            self._search_files,
            category="search",
            tags=["search", "files"]
        )
        
        # 执行工具
        self.register_tool(
            "execute_command",
            "执行系统命令",
            {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的命令"
                    },
                    "working_directory": {
                        "type": "string",
                        "description": "工作目录"
                    }
                },
                "required": ["command"]
            },
            self._execute_command,
            category="system",
            tags=["execute", "command"],
            complexity=3
        )
        
        # 项目分析工具
        self.register_tool(
            "analyze_project",
            "分析项目结构",
            {
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "项目路径"
                    }
                },
                "required": ["project_path"]
            },
            self._analyze_project,
            category="analysis",
            tags=["analyze", "project"],
            complexity=4
        )
    
    def register_tool(self, name: str, description: str, parameters: Dict[str, Any], 
                     function: Callable, category: str = "general", tags: List[str] = None,
                     model_compatibility: List[ModelType] = None, complexity: int = 1,
                     parallel_safe: bool = True, timeout: int = 30):
        """注册工具"""
        tool = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            function=function,
            category=category,
            tags=tags or [],
            model_compatibility=model_compatibility or list(ModelType),
            complexity=complexity,
            parallel_safe=parallel_safe,
            timeout=timeout
        )
        
        self.tools[name] = tool
        logger.info(f"注册工具: {name}")
    
    def get_tools_for_model(self, model_type: ModelType) -> List[ToolDefinition]:
        """获取指定模型的工具列表"""
        adapter = self.adapters.get(model_type, self.adapters[ModelType.UNKNOWN])
        
        tools = []
        for tool in self.tools.values():
            if model_type in tool.model_compatibility:
                tools.append(tool)
        
        return tools
    
    def format_tools_for_model(self, model_type: ModelType) -> List[Dict[str, Any]]:
        """为指定模型格式化工具列表"""
        adapter = self.adapters.get(model_type, self.adapters[ModelType.UNKNOWN])
        tools = self.get_tools_for_model(model_type)
        
        formatted_tools = []
        for tool in tools:
            formatted_tools.append(adapter.format_tool_call(tool, {}))
        
        return formatted_tools
    
    def call_tool(self, tool_call: ToolCall, model_type: ModelType = ModelType.UNKNOWN) -> ToolResult:
        """调用工具"""
        start_time = datetime.now()
        
        try:
            # 获取工具定义
            tool = self.tools.get(tool_call.tool_name)
            if not tool:
                return ToolResult(
                    success=False,
                    error=f"工具不存在: {tool_call.tool_name}",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # 检查模型兼容性
            if model_type not in tool.model_compatibility:
                logger.warning(f"工具 {tool_call.tool_name} 可能不兼容模型 {model_type}")
            
            # 验证参数
            validation_result = self._validate_parameters(tool, tool_call.arguments)
            if not validation_result.valid:
                return ToolResult(
                    success=False,
                    error=f"参数验证失败: {validation_result.error}",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # 执行工具
            result = tool.function(**tool_call.arguments)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 记录执行历史
            self.execution_history.append({
                "tool_name": tool_call.tool_name,
                "arguments": tool_call.arguments,
                "success": True,
                "execution_time": execution_time,
                "timestamp": start_time.isoformat()
            })
            
            return ToolResult(
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={"call_id": tool_call.call_id}
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"工具调用失败: {str(e)}"
            
            # 记录执行历史
            self.execution_history.append({
                "tool_name": tool_call.tool_name,
                "arguments": tool_call.arguments,
                "success": False,
                "error": error_msg,
                "execution_time": execution_time,
                "timestamp": start_time.isoformat()
            })
            
            logger.error(f"工具调用失败 {tool_call.tool_name}: {e}")
            logger.debug(traceback.format_exc())
            
            return ToolResult(
                success=False,
                error=error_msg,
                execution_time=execution_time,
                metadata={"call_id": tool_call.call_id}
            )
    
    def call_tools_parallel(self, tool_calls: List[ToolCall], model_type: ModelType = ModelType.UNKNOWN) -> List[ToolResult]:
        """并行调用工具"""
        import concurrent.futures
        
        # 过滤可并行的工具
        parallel_calls = []
        sequential_calls = []
        
        for call in tool_calls:
            tool = self.tools.get(call.tool_name)
            if tool and tool.parallel_safe:
                parallel_calls.append(call)
            else:
                sequential_calls.append(call)
        
        results = []
        
        # 并行执行
        if parallel_calls:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_call = {
                    executor.submit(self.call_tool, call, model_type): call 
                    for call in parallel_calls
                }
                
                for future in concurrent.futures.as_completed(future_to_call):
                    call = future_to_call[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append(ToolResult(
                            success=False,
                            error=f"并行执行失败: {str(e)}"
                        ))
        
        # 顺序执行
        for call in sequential_calls:
            result = self.call_tool(call, model_type)
            results.append(result)
        
        return results
    
    def detect_model_type(self, model_name: str) -> ModelType:
        """检测模型类型"""
        model_name_lower = model_name.lower()
        
        if "gpt" in model_name_lower:
            return ModelType.OPENAI
        elif "claude" in model_name_lower:
            return ModelType.ANTHROPIC
        elif "gemini" in model_name_lower or "bard" in model_name_lower:
            return ModelType.GOOGLE
        elif "qwen" in model_name_lower or "tongyi" in model_name_lower:
            return ModelType.ALIBABA
        elif "glm" in model_name_lower or "zhipu" in model_name_lower:
            return ModelType.ZHIPU
        elif "deepseek" in model_name_lower:
            return ModelType.DEEPSEEK
        elif "moonshot" in model_name_lower or "kimi" in model_name_lower:
            return ModelType.MOONSHOT
        elif "local" in model_name_lower or "llama" in model_name_lower:
            return ModelType.LOCAL
        else:
            return ModelType.UNKNOWN
    
    def _validate_parameters(self, tool: ToolDefinition, arguments: Dict[str, Any]) -> 'ValidationResult':
        """验证参数"""
        required_params = tool.parameters.get("required", [])
        properties = tool.parameters.get("properties", {})
        
        # 检查必需参数
        for param in required_params:
            if param not in arguments:
                return ValidationResult(False, f"缺少必需参数: {param}")
        
        # 检查参数类型
        for param_name, param_value in arguments.items():
            if param_name in properties:
                param_schema = properties[param_name]
                expected_type = param_schema.get("type")
                
                if expected_type == "string" and not isinstance(param_value, str):
                    return ValidationResult(False, f"参数 {param_name} 应为字符串类型")
                elif expected_type == "number" and not isinstance(param_value, (int, float)):
                    return ValidationResult(False, f"参数 {param_name} 应为数字类型")
                elif expected_type == "boolean" and not isinstance(param_value, bool):
                    return ValidationResult(False, f"参数 {param_name} 应为布尔类型")
                elif expected_type == "array" and not isinstance(param_value, list):
                    return ValidationResult(False, f"参数 {param_name} 应为数组类型")
                elif expected_type == "object" and not isinstance(param_value, dict):
                    return ValidationResult(False, f"参数 {param_name} 应为对象类型")
        
        return ValidationResult(True)
    
    # 内置工具实现
    def _read_file(self, file_path: str) -> str:
        """读取文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"读取文件失败: {str(e)}")
    
    def _write_file(self, file_path: str, content: str) -> str:
        """写入文件内容"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"文件写入成功: {file_path}"
        except Exception as e:
            raise Exception(f"写入文件失败: {str(e)}")
    
    def _list_directory(self, directory_path: str) -> List[str]:
        """列出目录内容"""
        try:
            return os.listdir(directory_path)
        except Exception as e:
            raise Exception(f"列出目录失败: {str(e)}")
    
    def _search_files(self, pattern: str, directory: str = ".") -> List[str]:
        """搜索文件内容"""
        try:
            import fnmatch
            matches = []
            for root, dirs, files in os.walk(directory):
                for filename in fnmatch.filter(files, pattern):
                    matches.append(os.path.join(root, filename))
            return matches
        except Exception as e:
            raise Exception(f"搜索文件失败: {str(e)}")
    
    def _execute_command(self, command: str, working_directory: str = None) -> str:
        """执行系统命令"""
        try:
            import subprocess
            
            cwd = working_directory if working_directory else os.getcwd()
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise Exception(f"命令执行失败: {result.stderr}")
            
            return result.stdout
        except subprocess.TimeoutExpired:
            raise Exception("命令执行超时")
        except Exception as e:
            raise Exception(f"命令执行失败: {str(e)}")
    
    def _analyze_project(self, project_path: str) -> Dict[str, Any]:
        """分析项目结构"""
        try:
            from project_analyzer import get_project_analyzer
            
            analyzer = get_project_analyzer(self.config)
            analysis = analyzer.analyze_project(project_path)
            
            return {
                "project_name": analysis.project_name,
                "project_type": analysis.project_type,
                "primary_language": analysis.primary_language,
                "frameworks": analysis.frameworks,
                "architecture": analysis.architecture,
                "complexity_score": analysis.complexity_score,
                "difficulty_level": analysis.difficulty_level,
                "estimated_effort": analysis.estimated_effort,
                "recommendations": analysis.recommendations
            }
        except Exception as e:
            raise Exception(f"项目分析失败: {str(e)}")

@dataclass
class ValidationResult:
    """验证结果"""
    valid: bool
    error: Optional[str] = None

# 全局工具调用器实例
_tool_caller = None

def get_tool_caller(config: Dict[str, Any] = None) -> IntelligentToolCaller:
    """获取工具调用器实例"""
    global _tool_caller
    if _tool_caller is None:
        _tool_caller = IntelligentToolCaller(config)
    return _tool_caller

if __name__ == "__main__":
    # 测试代码
    config = {}
    
    caller = get_tool_caller(config)
    
    # 测试工具调用
    tool_call = ToolCall(
        tool_name="read_file",
        arguments={"file_path": __file__}
    )
    
    result = caller.call_tool(tool_call)
    print("工具调用结果:", result.success)
    if result.success:
        print("结果:", result.result[:100] + "...")
    else:
        print("错误:", result.error)
    
    # 测试模型检测
    models = ["gpt-4", "claude-3", "gemini-pro", "qwen-max", "glm-4", "deepseek-chat", "moonshot-v1"]
    for model in models:
        model_type = caller.detect_model_type(model)
        print(f"模型 {model} -> 类型 {model_type.value}")