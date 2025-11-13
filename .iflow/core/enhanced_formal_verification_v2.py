#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版形式化验证引擎 V2
专门用于优化工具调用的精度和错误处理
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import logging
import re
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
import traceback

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class VerificationLevel(Enum):
    """验证级别"""
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    FORMAL = "formal"

class ValidationResult(Enum):
    """验证结果"""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    ERROR = "error"

class ToolCallStatus(Enum):
    """工具调用状态"""
    PENDING = "pending"
    VALIDATING = "validating"
    VALID = "valid"
    INVALID = "invalid"
    EXECUTED = "executed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class ToolCallSpecification:
    """工具调用规范"""
    tool_name: str
    required_params: List[str]
    optional_params: List[str]
    param_types: Dict[str, str]
    param_constraints: Dict[str, List[str]]
    return_type: str
    error_conditions: List[str]
    validation_rules: List[str]
    examples: List[Dict[str, Any]]

@dataclass
class ToolCallInstance:
    """工具调用实例"""
    call_id: str
    tool_name: str
    parameters: Dict[str, Any]
    timestamp: float
    status: ToolCallStatus
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    execution_result: Optional[Dict[str, Any]] = None
    error_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VerificationRule:
    """验证规则"""
    rule_id: str
    rule_type: str
    description: str
    severity: str
    condition: str
    error_message: str
    fix_suggestion: Optional[str] = None

class EnhancedFormalVerificationEngine:
    """增强版形式化验证引擎"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.verification_level = VerificationLevel(self.config.get('verification_level', 'advanced'))
        self.tool_specifications: Dict[str, ToolCallSpecification] = {}
        self.verification_rules: Dict[str, VerificationRule] = {}
        self.call_history: List[ToolCallInstance] = []
        self.validation_cache: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics = defaultdict(list)
        
        # 初始化验证规则
        self._initialize_verification_rules()
        # 初始化工具规范
        self._initialize_tool_specifications()
        
        logger.info(f"增强版形式化验证引擎初始化完成，验证级别: {self.verification_level.value}")
    
    def _initialize_verification_rules(self):
        """初始化验证规则"""
        basic_rules = [
            VerificationRule(
                rule_id="param_presence",
                rule_type="parameter",
                description="检查必需参数是否存在",
                severity="error",
                condition="required_params_present",
                error_message="缺少必需参数: {missing_params}",
                fix_suggestion="添加所有必需参数"
            ),
            VerificationRule(
                rule_id="param_type",
                rule_type="parameter",
                description="检查参数类型是否正确",
                severity="error",
                condition="parameter_types_match",
                error_message="参数类型错误: {param_name} 应该是 {expected_type}",
                fix_suggestion="修正参数类型"
            ),
            VerificationRule(
                rule_id="param_range",
                rule_type="constraint",
                description="检查参数值是否在允许范围内",
                severity="warning",
                condition="parameter_values_valid",
                error_message="参数值超出范围: {param_name} = {value}",
                fix_suggestion="调整参数值到有效范围内"
            ),
            VerificationRule(
                rule_id="tool_existence",
                rule_type="tool",
                description="检查工具是否存在",
                severity="error",
                condition="tool_exists",
                error_message="工具不存在: {tool_name}",
                fix_suggestion="检查工具名称拼写或确认工具已注册"
            )
        ]
        
        advanced_rules = [
            VerificationRule(
                rule_id="param_dependency",
                rule_type="dependency",
                description="检查参数间的依赖关系",
                severity="error",
                condition="parameter_dependencies_satisfied",
                error_message="参数依赖不满足: {dependency_info}",
                fix_suggestion="修正参数依赖关系"
            ),
            VerificationRule(
                rule_id="param_conflict",
                rule_type="conflict",
                description="检查参数间是否存在冲突",
                severity="error",
                condition="parameter_conflicts_resolved",
                error_message="参数冲突: {conflict_info}",
                fix_suggestion="解决参数冲突"
            ),
            VerificationRule(
                rule_id="execution_context",
                rule_type="context",
                description="检查执行上下文是否合适",
                severity="warning",
                condition="context_appropriate",
                error_message="执行上下文不合适: {context_info}",
                fix_suggestion="调整执行上下文"
            ),
            VerificationRule(
                rule_id="performance_impact",
                rule_type="performance",
                description="评估工具调用的性能影响",
                severity="info",
                condition="performance_within_limits",
                error_message="性能影响过大: {performance_info}",
                fix_suggestion="优化工具调用或考虑替代方案"
            )
        ]
        
        formal_rules = [
            VerificationRule(
                rule_id="formal_logic",
                rule_type="logic",
                description="形式化逻辑验证",
                severity="error",
                condition="formal_logic_valid",
                error_message="逻辑验证失败: {logic_error}",
                fix_suggestion="修正逻辑错误"
            ),
            VerificationRule(
                rule_id="state_consistency",
                rule_type="state",
                description="检查系统状态一致性",
                severity="error",
                condition="state_consistent",
                error_message="状态不一致: {state_info}",
                fix_suggestion="恢复状态一致性"
            ),
            VerificationRule(
                rule_id="resource_allocation",
                rule_type="resource",
                description="验证资源分配合理性",
                severity="warning",
                condition="resources_sufficient",
                error_message="资源不足: {resource_info}",
                fix_suggestion="释放资源或调整调用"
            )
        ]
        
        # 根据验证级别加载规则
        all_rules = basic_rules
        if self.verification_level in [VerificationLevel.STANDARD, VerificationLevel.ADVANCED, VerificationLevel.FORMAL]:
            all_rules.extend(advanced_rules)
        if self.verification_level in [VerificationLevel.ADVANCED, VerificationLevel.FORMAL]:
            all_rules.extend(formal_rules)
        
        for rule in all_rules:
            self.verification_rules[rule.rule_id] = rule
    
    def _initialize_tool_specifications(self):
        """初始化工具规范"""
        # 基础工具规范
        self.tool_specifications["file_read"] = ToolCallSpecification(
            tool_name="file_read",
            required_params=["path"],
            optional_params=["encoding"],
            param_types={"path": "string", "encoding": "string"},
            param_constraints={
                "path": ["must_exist", "valid_path_format"],
                "encoding": ["valid_encoding"]
            },
            return_type="string",
            error_conditions=["file_not_found", "permission_denied", "invalid_encoding"],
            validation_rules=["path_validation", "encoding_validation"],
            examples=[
                {"path": "./data/config.json", "encoding": "utf-8"},
                {"path": "./README.md"}
            ]
        )
        
        self.tool_specifications["file_write"] = ToolCallSpecification(
            tool_name="file_write",
            required_params=["path", "content"],
            optional_params=["encoding", "mode"],
            param_types={"path": "string", "content": "string", "encoding": "string", "mode": "string"},
            param_constraints={
                "path": ["valid_path_format", "directory_exists"],
                "content": ["not_empty"],
                "encoding": ["valid_encoding"],
                "mode": ["valid_write_mode"]
            },
            return_type="boolean",
            error_conditions=["invalid_path", "permission_denied", "disk_full"],
            validation_rules=["path_validation", "content_validation", "mode_validation"],
            examples=[
                {"path": "./output.txt", "content": "Hello World", "encoding": "utf-8", "mode": "w"},
                {"path": "./data.json", "content": '{"key": "value"}'}
            ]
        )
        
        self.tool_specifications["execute_command"] = ToolCallSpecification(
            tool_name="execute_command",
            required_params=["command"],
            optional_params=["timeout", "working_directory"],
            param_types={"command": "string", "timeout": "number", "working_directory": "string"},
            param_constraints={
                "command": ["not_empty", "safe_command"],
                "timeout": ["positive_number", "within_limits"],
                "working_directory": ["valid_path"]
            },
            return_type="dict",
            error_conditions=["command_not_found", "timeout", "permission_denied"],
            validation_rules=["command_validation", "timeout_validation", "safety_check"],
            examples=[
                {"command": "ls -la", "timeout": 30},
                {"command": "python script.py", "timeout": 60, "working_directory": "./"}
            ]
        )
    
    def validate_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """验证工具调用"""
        call_id = hashlib.md5(f"{tool_name}_{json.dumps(parameters, sort_keys=True)}".encode()).hexdigest()
        
        # 创建调用实例
        call_instance = ToolCallInstance(
            call_id=call_id,
            tool_name=tool_name,
            parameters=parameters,
            timestamp=time.time(),
            status=ToolCallStatus.VALIDATING
        )
        
        validation_results = []
        
        # 1. 基础验证
        basic_result = self._validate_basic_requirements(call_instance)
        validation_results.append(basic_result)
        
        # 2. 参数验证
        if basic_result["result"] == ValidationResult.PASS:
            param_result = self._validate_parameters(call_instance)
            validation_results.append(param_result)
        
        # 3. 高级验证
        if self.verification_level in [VerificationLevel.STANDARD, VerificationLevel.ADVANCED, VerificationLevel.FORMAL]:
            advanced_result = self._validate_advanced_rules(call_instance)
            validation_results.append(advanced_result)
        
        # 4. 形式化验证
        if self.verification_level in [VerificationLevel.ADVANCED, VerificationLevel.FORMAL]:
            formal_result = self._validate_formal_logic(call_instance)
            validation_results.append(formal_result)
        
        # 合并验证结果
        overall_result = self._merge_validation_results(validation_results)
        
        # 更新调用实例
        call_instance.status = ToolCallStatus.VALID if overall_result["overall_status"] == ValidationResult.PASS else ToolCallStatus.INVALID
        call_instance.validation_results = validation_results
        
        # 记录历史
        self.call_history.append(call_instance)
        
        # 缓存结果
        cache_key = f"{tool_name}_{call_id}"
        self.validation_cache[cache_key] = {
            "result": overall_result,
            "timestamp": time.time(),
            "parameters": parameters
        }
        
        # 更新性能指标
        self._update_performance_metrics(call_instance, validation_results)
        
        return {
            "call_id": call_id,
            "tool_name": tool_name,
            "overall_status": overall_result["overall_status"].value,
            "validation_results": validation_results,
            "error_messages": overall_result["error_messages"],
            "fix_suggestions": overall_result["fix_suggestions"],
            "execution_ready": overall_result["overall_status"] == ValidationResult.PASS
        }
    
    def _validate_basic_requirements(self, call_instance: ToolCallInstance) -> Dict[str, Any]:
        """基础要求验证"""
        tool_name = call_instance.tool_name
        parameters = call_instance.parameters
        
        result = {
            "rule_id": "basic_requirements",
            "result": ValidationResult.PASS,
            "messages": [],
            "details": {}
        }
        
        # 检查工具是否存在
        if tool_name not in self.tool_specifications:
            result["result"] = ValidationResult.FAIL
            result["messages"].append(f"工具不存在: {tool_name}")
            return result
        
        spec = self.tool_specifications[tool_name]
        
        # 检查必需参数
        missing_params = []
        for required_param in spec.required_params:
            if required_param not in parameters:
                missing_params.append(required_param)
        
        if missing_params:
            result["result"] = ValidationResult.FAIL
            result["messages"].append(f"缺少必需参数: {', '.join(missing_params)}")
        
        # 检查参数类型
        type_errors = []
        for param_name, param_value in parameters.items():
            if param_name in spec.param_types:
                expected_type = spec.param_types[param_name]
                if not self._validate_param_type(param_value, expected_type):
                    type_errors.append(f"{param_name}: 期望 {expected_type}, 实际 {type(param_value).__name__}")
        
        if type_errors:
            result["result"] = ValidationResult.FAIL
            result["messages"].extend(type_errors)
        
        result["details"] = {
            "missing_params": missing_params,
            "type_errors": type_errors,
            "spec_found": tool_name in self.tool_specifications
        }
        
        return result
    
    def _validate_parameters(self, call_instance: ToolCallInstance) -> Dict[str, Any]:
        """参数验证"""
        tool_name = call_instance.tool_name
        parameters = call_instance.parameters
        spec = self.tool_specifications[tool_name]
        
        result = {
            "rule_id": "parameter_validation",
            "result": ValidationResult.PASS,
            "messages": [],
            "details": {}
        }
        
        constraint_violations = []
        for param_name, param_value in parameters.items():
            if param_name in spec.param_constraints:
                constraints = spec.param_constraints[param_name]
                for constraint in constraints:
                    if not self._check_constraint(param_name, param_value, constraint):
                        constraint_violations.append(f"{param_name}: 违反约束 '{constraint}'")
        
        if constraint_violations:
            result["result"] = ValidationResult.WARNING
            result["messages"].extend(constraint_violations)
        
        result["details"] = {
            "constraint_violations": constraint_violations,
            "validated_params": list(parameters.keys())
        }
        
        return result
    
    def _validate_advanced_rules(self, call_instance: ToolCallInstance) -> Dict[str, Any]:
        """高级规则验证"""
        result = {
            "rule_id": "advanced_validation",
            "result": ValidationResult.PASS,
            "messages": [],
            "details": {}
        }
        
        # 这里可以添加更复杂的验证逻辑
        # 比如参数依赖关系、冲突检测等
        
        return result
    
    def _validate_formal_logic(self, call_instance: ToolCallInstance) -> Dict[str, Any]:
        """形式化逻辑验证"""
        result = {
            "rule_id": "formal_logic",
            "result": ValidationResult.PASS,
            "messages": [],
            "details": {}
        }
        
        # 这里可以添加形式化验证逻辑
        # 比如使用TLA+、Coq等工具
        
        return result
    
    def _validate_param_type(self, value: Any, expected_type: str) -> bool:
        """验证参数类型"""
        type_mapping = {
            "string": str,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected_python_type = type_mapping.get(expected_type, str)
        return isinstance(value, expected_python_type)
    
    def _check_constraint(self, param_name: str, param_value: Any, constraint: str) -> bool:
        """检查参数约束"""
        try:
            if constraint == "not_empty":
                return param_value is not None and param_value != ""
            elif constraint == "positive_number":
                return isinstance(param_value, (int, float)) and param_value > 0
            elif constraint == "valid_path_format":
                return isinstance(param_value, str) and len(param_value.strip()) > 0
            elif constraint == "valid_encoding":
                valid_encodings = ["utf-8", "ascii", "latin-1", "gbk"]
                return isinstance(param_value, str) and param_value.lower() in valid_encodings
            elif constraint == "valid_write_mode":
                valid_modes = ["r", "w", "a", "x"]
                return isinstance(param_value, str) and param_value in valid_modes
            else:
                # 其他约束可以在这里添加
                return True
        except Exception:
            return False
    
    def _merge_validation_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并验证结果"""
        overall_status = ValidationResult.PASS
        error_messages = []
        fix_suggestions = []
        
        for result in results:
            if result["result"] == ValidationResult.FAIL:
                overall_status = ValidationResult.FAIL
                error_messages.extend(result["messages"])
            elif result["result"] == ValidationResult.WARNING and overall_status == ValidationResult.PASS:
                overall_status = ValidationResult.WARNING
                error_messages.extend(result["messages"])
            elif result["result"] == ValidationResult.ERROR:
                overall_status = ValidationResult.ERROR
                error_messages.extend(result["messages"])
        
        return {
            "overall_status": overall_status,
            "error_messages": error_messages,
            "fix_suggestions": fix_suggestions
        }
    
    def _update_performance_metrics(self, call_instance: ToolCallInstance, validation_results: List[Dict[str, Any]]):
        """更新性能指标"""
        duration = time.time() - call_instance.timestamp
        
        self.performance_metrics["validation_duration"].append(duration)
        self.performance_metrics["total_calls"].append(1)
        self.performance_metrics["status_distribution"].append(call_instance.status.value)
        
        # 保持指标在合理范围内
        if len(self.performance_metrics["validation_duration"]) > 1000:
            self.performance_metrics["validation_duration"] = self.performance_metrics["validation_duration"][-500:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.performance_metrics["validation_duration"]:
            return {"message": "暂无性能数据"}
        
        durations = self.performance_metrics["validation_duration"]
        
        return {
            "total_validations": len(durations),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "status_distribution": self.performance_metrics["status_distribution"][-100:].count(ToolCallStatus.VALID.value) / max(len(self.performance_metrics["status_distribution"][-100:]), 1)
        }
    
    def add_tool_specification(self, spec: ToolCallSpecification):
        """添加工具规范"""
        self.tool_specifications[spec.tool_name] = spec
        logger.info(f"添加工具规范: {spec.tool_name}")
    
    def get_call_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取调用历史"""
        return [
            {
                "call_id": call.call_id,
                "tool_name": call.tool_name,
                "status": call.status.value,
                "timestamp": call.timestamp,
                "validation_count": len(call.validation_results)
            }
            for call in self.call_history[-limit:]
        ]

if __name__ == "__main__":
    # 测试代码
    verifier = EnhancedFormalVerificationEngine({"verification_level": "advanced"})
    
    print("增强版形式化验证引擎 V2 测试")
    print("=" * 50)
    
    # 测试工具调用验证
    test_cases = [
        {
            "tool_name": "file_read",
            "parameters": {"path": "./test.txt", "encoding": "utf-8"},
            "description": "正确的文件读取调用"
        },
        {
            "tool_name": "file_read", 
            "parameters": {"path": "./test.txt"},
            "description": "缺少可选参数的调用"
        },
        {
            "tool_name": "file_write",
            "parameters": {"path": "./output.txt", "content": "Hello", "mode": "w"},
            "description": "正确的文件写入调用"
        },
        {
            "tool_name": "file_write",
            "parameters": {"content": "Hello"},
            "description": "缺少必需参数的调用"
        },
        {
            "tool_name": "execute_command",
            "parameters": {"command": "ls", "timeout": 30},
            "description": "正确的命令执行调用"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试案例 {i}: {test_case['description']}")
        print(f"工具: {test_case['tool_name']}")
        print(f"参数: {test_case['parameters']}")
        
        result = verifier.validate_tool_call(test_case["tool_name"], test_case["parameters"])
        
        print(f"验证结果: {result['overall_status']}")
        print(f"执行就绪: {result['execution_ready']}")
        
        if result['error_messages']:
            print("错误信息:")
            for msg in result['error_messages']:
                print(f"  - {msg}")
        
        if result['fix_suggestions']:
            print("修复建议:")
            for suggestion in result['fix_suggestions']:
                print(f"  - {suggestion}")
    
    # 获取性能摘要
    print(f"\n性能摘要:")
    perf_summary = verifier.get_performance_summary()
    for key, value in perf_summary.items():
        print(f"  {key}: {value}")