#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版提示词工程和响应格式化系统 V2
专门用于优化提示词工程和响应格式化，提升LLM交互质量
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
import hashlib
import jinja2

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class PromptType(Enum):
    """提示词类型"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ARQ_QUERY = "arq_query"
    CONSCIOUSNESS_STREAM = "consciousness_stream"
    MULTI_MODEL_ROUTING = "multi_model_routing"

class ResponseFormat(Enum):
    """响应格式"""
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    MARKDOWN = "markdown"
    PLAIN_TEXT = "plain_text"
    STRUCTURED = "structured"
    TOOL_CALL_FORMAT = "tool_call_format"

class PromptQuality(Enum):
    """提示词质量"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    POOR = "poor"
    UNUSABLE = "unusable"

@dataclass
class PromptTemplate:
    """提示词模板"""
    template_id: str
    template_name: str
    template_type: PromptType
    content: str
    placeholders: List[str]
    examples: List[Dict[str, Any]]
    constraints: List[str]
    quality_score: float
    usage_count: int = 0
    success_rate: float = 0.0
    version: str = "1.0"

@dataclass
class PromptContext:
    """提示词上下文"""
    context_id: str
    user_input: str
    system_state: Dict[str, Any]
    agent_info: Dict[str, Any]
    tool_context: Dict[str, Any]
    consciousness_context: Dict[str, Any]
    arq_context: Dict[str, Any]
    timestamp: float

@dataclass
class FormattedResponse:
    """格式化响应"""
    response_id: str
    original_response: str
    formatted_content: Dict[str, Any]
    format_type: ResponseFormat
    quality_score: float
    validation_results: List[Dict[str, Any]]
    processing_time: float
    model_info: Dict[str, Any]

class EnhancedPromptEngineeringSystem:
    """增强版提示词工程系统"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 模板管理
        self.prompt_templates: Dict[str, PromptTemplate] = {}
        self.template_categories = defaultdict(list)
        
        # 上下文管理
        self.context_history: List[PromptContext] = []
        self.context_cache: Dict[str, PromptContext] = {}
        
        # 响应格式化
        self.response_formatters = {}
        self.format_validation_rules = {}
        
        # 质量评估
        self.quality_metrics = defaultdict(list)
        self.feedback_loop = []
        
        # 初始化系统
        self._initialize_prompt_templates()
        self._initialize_response_formatters()
        self._initialize_format_validation_rules()
        
        logger.info("增强版提示词工程系统 V2 初始化完成")
    
    def _initialize_prompt_templates(self):
        """初始化提示词模板"""
        # ARQ查询模板
        self.prompt_templates["arq_query_template"] = PromptTemplate(
            template_id="arq_query_v2",
            template_name="ARQ专注推理查询",
            template_type=PromptType.ARQ_QUERY,
            content="""
**角色：终极ARQ推理引擎 V2**
**任务：** 对当前任务进行深度合规性与结构化推理分析
**推理模式：** {{ reasoning_mode }}
**核心规则：**
{{ core_rules }}

**上下文信息：**
{{ consciousness_context }}

**当前任务：**
{{ current_task }}

**指令：**
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard, think intensely）。

严格按照以下JSON Schema进行推理和输出：
**JSON_SCHEMA：**
{{ json_schema }}

**特别注意：**
1. 必须检查所有相关规则的合规性
2. 推理过程必须结构化、逻辑清晰
3. 输出必须是有效的JSON格式
4. 确保推理步骤的完整性和准确性
""",
            placeholders=[
                "reasoning_mode", "core_rules", "consciousness_context",
                "current_task", "json_schema"
            ],
            examples=[
                {
                    "reasoning_mode": "structured",
                    "core_rules": "1. 所有代码必须通过安全审计\n2. 性能优化必须达到+50%以上",
                    "current_task": "设计一个安全的用户认证系统"
                }
            ],
            constraints=[
                "必须包含JSON Schema定义",
                "必须引用核心规则",
                "必须包含推理模式说明"
            ],
            quality_score=0.95
        )
        
        # 系统提示词模板
        self.prompt_templates["system_prompt_template"] = PromptTemplate(
            template_id="system_prompt_v2",
            template_name="智能系统提示词",
            template_type=PromptType.SYSTEM,
            content="""
你是一个终极智能代理系统，具备以下核心能力：

**核心使命：**
构建具备递归自我改进能力、计算验证驱动、合规性强制执行的超凡元智能代理系统。

**架构原则：**
1. 内核化：所有核心功能必须封装为可插拔、可独立优化的专业内核
2. 计算验证：所有关键决策和算法必须通过形式化验证和基准测试
3. 合规强制：采用ARQ机制，确保所有行为严格遵循预设的指导原则
4. 递归学习：系统必须能够学习如何更好地学习，实现指数级智能放大

**当前上下文：**
{{ agent_context }}

**任务要求：**
{{ task_requirements }}

**输出格式：**
{{ output_format }}

**特别强调：**
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
""",
            placeholders=[
                "agent_context", "task_requirements", "output_format"
            ],
            examples=[
                {
                    "agent_context": "你正在处理一个复杂的系统架构设计任务",
                    "task_requirements": "需要考虑性能、安全性、可扩展性等多个维度",
                    "output_format": "JSON格式，包含详细的分析和建议"
                }
            ],
            constraints=[
                "必须引用架构原则",
                "必须包含核心使命",
                "必须使用强调的思考要求"
            ],
            quality_score=0.92
        )
        
        # 工具调用模板
        self.prompt_templates["tool_call_template"] = PromptTemplate(
            template_id="tool_call_v2",
            template_name="精确工具调用",
            template_type=PromptType.TOOL_CALL,
            content="""
**工具调用请求**
工具名称：{{ tool_name }}
调用原因：{{ call_reason }}
参数说明：
{{ parameter_description }}

**参数详情：**
```json
{{ parameters }}
```

**上下文信息：**
{{ tool_context }}

**预期结果：**
{{ expected_result }}

**注意事项：**
- 确保参数格式正确
- 验证参数值在有效范围内
- 考虑工具调用的副作用
""",
            placeholders=[
                "tool_name", "call_reason", "parameter_description",
                "parameters", "tool_context", "expected_result"
            ],
            examples=[
                {
                    "tool_name": "file_read",
                    "call_reason": "需要读取配置文件进行分析",
                    "parameter_description": "指定要读取的文件路径和编码格式",
                    "parameters": '{"path": "./config.json", "encoding": "utf-8"}',
                    "expected_result": "返回文件的完整内容"
                }
            ],
            constraints=[
                "参数必须是有效的JSON格式",
                "必须说明调用原因",
                "必须包含上下文信息"
            ],
            quality_score=0.90
        )
        
        # 多模型路由模板
        self.prompt_templates["multi_model_routing_template"] = PromptTemplate(
            template_id="multi_model_routing_v2",
            template_name="智能模型路由",
            template_type=PromptType.MULTI_MODEL_ROUTING,
            content="""
**模型路由决策**
任务类型：{{ task_type }}
复杂度评估：{{ complexity_level }}
专业领域：{{ domain_area }}
预期输出：{{ expected_output }}

**可用模型：**
{{ available_models }}

**推荐模型：** {{ recommended_model }}
**推荐理由：**
{{ recommendation_reason }}

**路由策略：**
{{ routing_strategy }}

**备选方案：**
{{ fallback_models }}

**质量保证：**
- 模型能力匹配度：{{ capability_match }}%
- 成本效益评估：{{ cost_effectiveness }}
- 响应时间预估：{{ response_time }}秒
""",
            placeholders=[
                "task_type", "complexity_level", "domain_area", "expected_output",
                "available_models", "recommended_model", "recommendation_reason",
                "routing_strategy", "fallback_models", "capability_match",
                "cost_effectiveness", "response_time"
            ],
            examples=[
                {
                    "task_type": "代码生成",
                    "complexity_level": "high",
                    "domain_area": "Python后端开发",
                    "recommended_model": "GPT-4",
                    "recommendation_reason": "在代码生成和复杂逻辑处理方面表现最佳"
                }
            ],
            constraints=[
                "必须包含能力匹配度评估",
                "必须提供备选方案",
                "必须考虑成本效益"
            ],
            quality_score=0.88
        )
        
        # 按类型组织模板
        for template_id, template in self.prompt_templates.items():
            self.template_categories[template.template_type.value].append(template_id)
    
    def _initialize_response_formatters(self):
        """初始化响应格式化器"""
        # JSON格式化器
        self.response_formatters["json_formatter"] = {
            "format_type": ResponseFormat.JSON,
            "format_function": self._format_as_json,
            "description": "将响应格式化为结构化的JSON"
        }
        
        # XML格式化器
        self.response_formatters["xml_formatter"] = {
            "format_type": ResponseFormat.XML,
            "format_function": self._format_as_xml,
            "description": "将响应格式化为结构化的XML"
        }
        
        # 工具调用格式化器
        self.response_formatters["tool_call_formatter"] = {
            "format_type": ResponseFormat.TOOL_CALL_FORMAT,
            "format_function": self._format_as_tool_call,
            "description": "将响应格式化为工具调用格式"
        }
        
        # 结构化格式化器
        self.response_formatters["structured_formatter"] = {
            "format_type": ResponseFormat.STRUCTURED,
            "format_function": self._format_as_structured,
            "description": "将响应格式化为结构化格式"
        }
    
    def _initialize_format_validation_rules(self):
        """初始化格式验证规则"""
        self.format_validation_rules = {
            "json_validation": [
                "valid_json_syntax",
                "required_fields_present",
                "field_types_correct",
                "nested_structure_valid"
            ],
            "xml_validation": [
                "valid_xml_syntax",
                "proper_tag_closing",
                "namespace_consistency",
                "schema_compliance"
            ],
            "tool_call_validation": [
                "tool_name_valid",
                "parameters_format_correct",
                "parameter_types_match",
                "execution_context_present"
            ]
        }
    
    def generate_prompt(self, template_type: PromptType, context: PromptContext, 
                       custom_parameters: Dict[str, Any] = None) -> str:
        """生成提示词"""
        # 查找匹配的模板
        template_candidates = self.template_categories.get(template_type.value, [])
        
        if not template_candidates:
            logger.warning(f"未找到类型为 {template_type.value} 的模板")
            return self._generate_fallback_prompt(template_type, context, custom_parameters)
        
        # 选择最佳模板（基于使用次数和成功率）
        best_template = self._select_best_template(template_candidates)
        
        # 准备模板参数
        template_params = self._prepare_template_parameters(template_type, context, custom_parameters)
        
        try:
            # 使用Jinja2模板引擎进行渲染
            environment = jinja2.Environment()
            template = environment.from_string(best_template.content)
            generated_prompt = template.render(**template_params)
            
            # 更新模板使用统计
            best_template.usage_count += 1
            
            logger.debug(f"成功生成提示词，使用模板: {best_template.template_name}")
            return generated_prompt
            
        except Exception as e:
            logger.error(f"提示词生成失败: {e}")
            return self._generate_fallback_prompt(template_type, context, custom_parameters)
    
    def _select_best_template(self, candidates: List[str]) -> PromptTemplate:
        """选择最佳模板"""
        best_template = None
        best_score = -1
        
        for template_id in candidates:
            template = self.prompt_templates[template_id]
            # 综合评分：质量分数 * 成功率 * 使用次数权重
            usage_weight = min(template.usage_count / 100 + 1, 2)  # 使用次数权重，上限2
            score = template.quality_score * template.success_rate * usage_weight
            
            if score > best_score:
                best_score = score
                best_template = template
        
        return best_template or list(self.prompt_templates.values())[0]  # 返回第一个作为备选
    
    def _prepare_template_parameters(self, template_type: PromptType, 
                                   context: PromptContext, 
                                   custom_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """准备模板参数"""
        base_params = {
            "user_input": context.user_input,
            "agent_context": json.dumps(context.agent_info, ensure_ascii=False, indent=2),
            "task_requirements": self._extract_task_requirements(context.user_input),
            "output_format": self._determine_output_format(template_type),
        }
        
        # 根据模板类型添加特定参数
        if template_type == PromptType.ARQ_QUERY:
            base_params.update({
                "reasoning_mode": self._determine_reasoning_mode(context),
                "core_rules": self._format_core_rules(),
                "consciousness_context": self._format_consciousness_context(context),
                "current_task": context.user_input,
                "json_schema": self._generate_arq_json_schema()
            })
        
        elif template_type == PromptType.TOOL_CALL:
            base_params.update({
                "tool_name": context.tool_context.get("tool_name", "unknown"),
                "call_reason": context.tool_context.get("reason", "未指定"),
                "parameter_description": self._describe_tool_parameters(context.tool_context),
                "parameters": json.dumps(context.tool_context.get("parameters", {}), ensure_ascii=False, indent=2),
                "tool_context": json.dumps(context.tool_context, ensure_ascii=False, indent=2),
                "expected_result": context.tool_context.get("expected_result", "未指定")
            })
        
        elif template_type == PromptType.MULTI_MODEL_ROUTING:
            base_params.update({
                "task_type": self._classify_task_type(context.user_input),
                "complexity_level": self._assess_complexity(context),
                "domain_area": self._identify_domain_area(context.user_input),
                "expected_output": self._predict_output_format(context),
                "available_models": self._list_available_models(),
                "recommended_model": self._recommend_model(context),
                "recommendation_reason": self._generate_recommendation_reason(context),
                "routing_strategy": self._define_routing_strategy(context)
            })
        
        # 合并自定义参数
        if custom_parameters:
            base_params.update(custom_parameters)
        
        return base_params
    
    def _extract_task_requirements(self, user_input: str) -> str:
        """提取任务要求"""
        # 简化的任务要求提取
        requirements = []
        
        if any(keyword in user_input.lower() for keyword in ["设计", "创建", "构建"]):
            requirements.append("需要提供详细的设计方案")
        if any(keyword in user_input.lower() for keyword in ["分析", "评估", "检查"]):
            requirements.append("需要进行深入的分析和评估")
        if any(keyword in user_input.lower() for keyword in ["优化", "改进", "提升"]):
            requirements.append("需要提供优化建议和改进方案")
        if any(keyword in user_input.lower() for keyword in ["代码", "编程", "开发"]):
            requirements.append("需要提供高质量的代码实现")
        
        return "；".join(requirements) if requirements else "根据具体内容提供相应的解决方案"
    
    def _determine_output_format(self, template_type: PromptType) -> str:
        """确定输出格式"""
        format_mapping = {
            PromptType.ARQ_QUERY: "结构化JSON格式，包含推理步骤和合规性检查",
            PromptType.SYSTEM: "清晰的系统说明和指导原则",
            PromptType.TOOL_CALL: "标准化的工具调用格式",
            PromptType.MULTI_MODEL_ROUTING: "包含评估指标的路由决策格式"
        }
        return format_mapping.get(template_type, "标准文本格式")
    
    def _determine_reasoning_mode(self, context: PromptContext) -> str:
        """确定推理模式"""
        # 基于任务类型和上下文确定推理模式
        user_input = context.user_input.lower()
        
        if any(keyword in user_input for keyword in ["分析", "评估", "诊断"]):
            return "analytical"
        elif any(keyword in user_input for keyword in ["设计", "创建", "构建"]):
            return "creative"
        elif any(keyword in user_input for keyword in ["解决", "修复", "处理"]):
            return "problem_solving"
        elif any(keyword in user_input for keyword in ["决定", "选择", "推荐"]):
            return "decision_making"
        else:
            return "structured"
    
    def _format_core_rules(self) -> str:
        """格式化核心规则"""
        rules = [
            "1. 所有代码必须通过安全审计",
            "2. 性能优化必须达到+50%以上",
            "3. 绝不创建冗余文件",
            "4. 确保工具调用精度达到100%",
            "5. 维护系统架构的一致性"
        ]
        return "\n".join(rules)
    
    def _format_consciousness_context(self, context: PromptContext) -> str:
        """格式化意识流上下文"""
        consciousness_data = context.consciousness_context
        if not consciousness_data:
            return "暂无相关上下文信息"
        
        formatted_context = []
        for key, value in consciousness_data.items():
            formatted_context.append(f"{key}: {value}")
        
        return "\n".join(formatted_context)
    
    def _generate_arq_json_schema(self) -> str:
        """生成ARQ JSON Schema"""
        schema = {
            "type": "object",
            "properties": {
                "rule_check": {
                    "type": "string",
                    "description": "当前任务是否违反任何核心规则？(是/否/不适用)"
                },
                "activated_rules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "如果适用，列出激活的规则编号"
                },
                "context_analysis": {
                    "type": "string",
                    "description": "从上下文中提取的关键信息和历史经验"
                },
                "reasoning_steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "string"},
                            "rationale": {"type": "string"},
                            "confidence": {"type": "number"}
                        }
                    },
                    "description": "结构化的推理步骤"
                },
                "tool_required": {
                    "type": "boolean",
                    "description": "下一步是否需要调用工具？"
                },
                "next_action_plan": {
                    "type": "string",
                    "description": "基于规则和推理的下一步行动计划"
                }
            },
            "required": ["rule_check", "next_action_plan"]
        }
        return json.dumps(schema, ensure_ascii=False, indent=2)
    
    def format_response(self, raw_response: str, target_format: ResponseFormat, 
                       model_info: Dict[str, Any] = None) -> FormattedResponse:
        """格式化响应"""
        start_time = time.time()
        
        # 选择格式化器
        formatter_key = f"{target_format.value}_formatter"
        if formatter_key not in self.response_formatters:
            logger.warning(f"未找到格式化器: {formatter_key}")
            target_format = ResponseFormat.PLAIN_TEXT
            formatter_key = "plain_text_formatter"
        
        formatter = self.response_formatters[formatter_key]
        
        try:
            # 执行格式化
            formatted_content = formatter["format_function"](raw_response)
            
            # 验证格式化结果
            validation_results = self._validate_formatted_response(formatted_content, target_format)
            
            # 计算质量分数
            quality_score = self._calculate_format_quality(formatted_content, validation_results)
            
            processing_time = time.time() - start_time
            
            return FormattedResponse(
                response_id=str(uuid.uuid4()),
                original_response=raw_response,
                formatted_content=formatted_content,
                format_type=target_format,
                quality_score=quality_score,
                validation_results=validation_results,
                processing_time=processing_time,
                model_info=model_info or {}
            )
            
        except Exception as e:
            logger.error(f"响应格式化失败: {e}")
            return self._create_error_response(raw_response, str(e), model_info)
    
    def _format_as_json(self, raw_response: str) -> Dict[str, Any]:
        """格式化为JSON"""
        try:
            # 尝试直接解析JSON
            if raw_response.strip().startswith('{') and raw_response.strip().endswith('}'):
                return json.loads(raw_response)
            
            # 提取JSON内容（如果包含在文本中）
            json_match = re.search(r'```json\s*(.*?)\s*```', raw_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # 尝试解析整个响应
            return json.loads(raw_response)
            
        except json.JSONDecodeError:
            # 如果解析失败，尝试结构化提取
            return self._extract_structured_data_from_text(raw_response)
    
    def _format_as_xml(self, raw_response: str) -> str:
        """格式化为XML"""
        # 简化的XML格式化
        structured_data = self._extract_structured_data_from_text(raw_response)
        return self._dict_to_xml(structured_data)
    
    def _format_as_tool_call(self, raw_response: str) -> Dict[str, Any]:
        """格式化为工具调用"""
        # 提取工具调用信息
        tool_call_data = {
            "tool_name": "",
            "parameters": {},
            "call_reason": "",
            "confidence": 0.0
        }
        
        # 简化的工具调用格式提取
        if "tool_name" in raw_response:
            # 尝试从响应中提取工具调用信息
            pass
        
        return tool_call_data
    
    def _format_as_structured(self, raw_response: str) -> Dict[str, Any]:
        """格式化为结构化格式"""
        return self._extract_structured_data_from_text(raw_response)
    
    def _extract_structured_data_from_text(self, text: str) -> Dict[str, Any]:
        """从文本中提取结构化数据"""
        data = {}
        
        # 提取标题
        title_match = re.search(r'^(#.*?)\n', text, re.MULTILINE)
        if title_match:
            data["title"] = title_match.group(1).strip('# ').strip()
        
        # 提取列表项
        list_items = re.findall(r'^[-*]\s+(.*?)$', text, re.MULTILINE)
        if list_items:
            data["list_items"] = list_items
        
        # 提取代码块
        code_blocks = re.findall(r'```(?:\w+)?\s*(.*?)\s*```', text, re.DOTALL)
        if code_blocks:
            data["code_blocks"] = code_blocks
        
        # 提取关键信息
        key_info = {}
        for line in text.split('\n'):
            if ':' in line and not line.startswith('#'):
                key, value = line.split(':', 1)
                key_info[key.strip()] = value.strip()
        
        if key_info:
            data["key_info"] = key_info
        
        return data
    
    def _dict_to_xml(self, data: Dict[str, Any], root_name: str = "response") -> str:
        """将字典转换为XML"""
        xml_parts = [f'<{root_name}>']
        
        def _dict_to_xml_recursive(d, indent=2):
            parts = []
            for key, value in d.items():
                indent_str = ' ' * indent
                if isinstance(value, dict):
                    parts.append(f'{indent_str}<{key}>')
                    parts.append(_dict_to_xml_recursive(value, indent + 2))
                    parts.append(f'{indent_str}</{key}>')
                elif isinstance(value, list):
                    for item in value:
                        parts.append(f'{indent_str}<{key}_item>')
                        if isinstance(item, dict):
                            parts.append(_dict_to_xml_recursive(item, indent + 4))
                        else:
                            parts.append(f'{' ' * (indent + 4)}{item}')
                        parts.append(f'{indent_str}</{key}_item>')
                else:
                    parts.append(f'{indent_str}<{key}>{value}</{key}>')
            return '\n'.join(parts)
        
        xml_parts.append(_dict_to_xml_recursive(data))
        xml_parts.append(f'</{root_name}>')
        
        return '\n'.join(xml_parts)
    
    def _validate_formatted_response(self, formatted_content: Any, 
                                   target_format: ResponseFormat) -> List[Dict[str, Any]]:
        """验证格式化响应"""
        validation_rules = self.format_validation_rules.get(f"{target_format.value}_validation", [])
        
        results = []
        for rule in validation_rules:
            result = {
                "rule": rule,
                "passed": False,
                "details": ""
            }
            
            try:
                if rule == "valid_json_syntax" and target_format == ResponseFormat.JSON:
                    result["passed"] = self._validate_json_syntax(formatted_content)
                    result["details"] = "JSON语法验证"
                elif rule == "required_fields_present":
                    result["passed"] = self._validate_required_fields(formatted_content)
                    result["details"] = "必需字段验证"
                elif rule == "tool_name_valid":
                    result["passed"] = self._validate_tool_name(formatted_content)
                    result["details"] = "工具名称验证"
                # 添加更多验证规则...
                
            except Exception as e:
                result["details"] = f"验证错误: {e}"
            
            results.append(result)
        
        return results
    
    def _validate_json_syntax(self, content: Any) -> bool:
        """验证JSON语法"""
        try:
            if isinstance(content, dict):
                json.dumps(content)
                return True
            return False
        except (TypeError, ValueError):
            return False
    
    def _validate_required_fields(self, content: Any) -> bool:
        """验证必需字段"""
        if not isinstance(content, dict):
            return False
        
        # 根据格式类型检查必需字段
        required_fields = {
            ResponseFormat.JSON: ["rule_check", "next_action_plan"],
            ResponseFormat.TOOL_CALL_FORMAT: ["tool_name", "parameters"]
        }
        
        # 这里可以根据具体需求实现字段验证
        return True
    
    def _validate_tool_name(self, content: Any) -> bool:
        """验证工具名称"""
        if not isinstance(content, dict):
            return False
        
        tool_name = content.get("tool_name", "")
        # 简化的工具名称验证
        return isinstance(tool_name, str) and len(tool_name) > 0
    
    def _calculate_format_quality(self, formatted_content: Any, 
                                validation_results: List[Dict[str, Any]]) -> float:
        """计算格式质量分数"""
        if not validation_results:
            return 0.0
        
        passed_count = sum(1 for result in validation_results if result["passed"])
        total_count = len(validation_results)
        
        base_quality = passed_count / total_count
        
        # 根据内容复杂度调整质量分数
        if isinstance(formatted_content, dict):
            content_quality = min(len(formatted_content) / 10, 1.0)  # 内容丰富度
            return (base_quality + content_quality) / 2
        
        return base_quality
    
    def _create_error_response(self, raw_response: str, error_message: str, 
                             model_info: Dict[str, Any]) -> FormattedResponse:
        """创建错误响应"""
        return FormattedResponse(
            response_id=str(uuid.uuid4()),
            original_response=raw_response,
            formatted_content={
                "error": True,
                "error_message": error_message,
                "original_response": raw_response
            },
            format_type=ResponseFormat.PLAIN_TEXT,
            quality_score=0.0,
            validation_results=[{"rule": "error_handling", "passed": False, "details": error_message}],
            processing_time=0.0,
            model_info=model_info or {}
        )
    
    def _generate_fallback_prompt(self, template_type: PromptType, 
                                context: PromptContext, 
                                custom_parameters: Dict[str, Any]) -> str:
        """生成备用提示词"""
        fallback_prompts = {
            PromptType.SYSTEM: "你是一个智能助手，请根据用户需求提供帮助。",
            PromptType.USER: context.user_input,
            PromptType.TOOL_CALL: f"调用工具: {custom_parameters.get('tool_name', 'unknown')}",
            PromptType.ARQ_QUERY: f"分析任务: {context.user_input}",
        }
        return fallback_prompts.get(template_type, "请提供具体的指导。")
    
    def get_prompt_quality_report(self) -> Dict[str, Any]:
        """获取提示词质量报告"""
        total_templates = len(self.prompt_templates)
        total_usage = sum(template.usage_count for template in self.prompt_templates.values())
        
        avg_quality = sum(template.quality_score for template in self.prompt_templates.values()) / total_templates
        avg_success_rate = sum(template.success_rate for template in self.prompt_templates.values()) / total_templates
        
        return {
            "total_templates": total_templates,
            "total_usage": total_usage,
            "average_quality": avg_quality,
            "average_success_rate": avg_success_rate,
            "template_distribution": {k: len(v) for k, v in self.template_categories.items()},
            "most_used_templates": self._get_most_used_templates(),
            "quality_distribution": self._get_quality_distribution()
        }
    
    def _get_most_used_templates(self) -> List[Dict[str, Any]]:
        """获取使用最频繁的模板"""
        sorted_templates = sorted(
            self.prompt_templates.values(),
            key=lambda x: x.usage_count,
            reverse=True
        )
        return [
            {
                "template_name": template.template_name,
                "usage_count": template.usage_count,
                "success_rate": template.success_rate
            }
            for template in sorted_templates[:5]
        ]
    
    def _get_quality_distribution(self) -> Dict[str, int]:
        """获取质量分布"""
        distribution = {"excellent": 0, "good": 0, "adequate": 0, "poor": 0, "unusable": 0}
        
        for template in self.prompt_templates.values():
            if template.quality_score >= 0.9:
                distribution["excellent"] += 1
            elif template.quality_score >= 0.8:
                distribution["good"] += 1
            elif template.quality_score >= 0.7:
                distribution["adequate"] += 1
            elif template.quality_score >= 0.5:
                distribution["poor"] += 1
            else:
                distribution["unusable"] += 1
        
        return distribution

# 全局提示词工程系统实例
_prompt_engineering_instance = None

def get_prompt_engineering_system(config: Dict[str, Any] = None) -> EnhancedPromptEngineeringSystem:
    """获取提示词工程系统实例"""
    global _prompt_engineering_instance
    if _prompt_engineering_instance is None:
        _prompt_engineering_instance = EnhancedPromptEngineeringSystem(config)
    return _prompt_engineering_instance

if __name__ == "__main__":
    # 测试代码
    print("增强版提示词工程系统 V2 测试")
    print("=" * 50)
    
    # 创建系统实例
    prompt_system = get_prompt_engineering_system()
    
    # 创建测试上下文
    test_context = PromptContext(
        context_id="test_context_001",
        user_input="请设计一个高可用的微服务架构系统",
        system_state={"mode": "design", "complexity": "high"},
        agent_info={"agent_id": "architect_agent", "capabilities": ["design", "analysis"]},
        tool_context={"tool_name": "architectural_design", "parameters": {}},
        consciousness_context={"previous_decisions": [], "current_focus": "architecture"},
        arq_context={"compliance_level": "strict", "reasoning_mode": "structured"},
        timestamp=time.time()
    )
    
    # 测试提示词生成
    print("测试提示词生成:")
    for template_type in [PromptType.ARQ_QUERY, PromptType.SYSTEM, PromptType.TOOL_CALL]:
        generated_prompt = prompt_system.generate_prompt(template_type, test_context)
        print(f"\n{template_type.value} 模板:")
        print("-" * 30)
        print(generated_prompt[:200] + "..." if len(generated_prompt) > 200 else generated_prompt)
    
    # 测试响应格式化
    print(f"\n测试响应格式化:")
    test_response = '{"rule_check": "否", "next_action_plan": "开始设计微服务架构", "confidence": 0.95}'
    
    for format_type in [ResponseFormat.JSON, ResponseFormat.STRUCTURED]:
        formatted = prompt_system.format_response(test_response, format_type)
        print(f"\n{format_type.value} 格式化结果:")
        print(f"  质量分数: {formatted.quality_score:.2f}")
        print(f"  处理时间: {formatted.processing_time:.3f}s")
        print(f"  验证结果: {len(formatted.validation_results)} 个规则")
    
    # 获取质量报告
    print(f"\n提示词质量报告:")
    quality_report = prompt_system.get_prompt_quality_report()
    for key, value in quality_report.items():
        print(f"  {key}: {value}")