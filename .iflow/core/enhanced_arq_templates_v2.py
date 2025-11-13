#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版ARQ推理模板系统 V2
基于B项目ARQ V2.0的优秀实现，进一步优化推理模板和形式化验证
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class ReasoningMode(Enum):
    """推理模式增强版"""
    STRUCTURED = "structured"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    CRITICAL = "critical"
    QUANTUM = "quantum"
    SYSTEMIC = "systemic"
    STRATEGIC = "strategic"
    TACTICAL = "tactical"

class ProblemType(Enum):
    """问题类型"""
    DESIGN = "design"
    ANALYTICAL = "analytical"
    PROBLEM_SOLVING = "problem_solving"
    DECISION_MAKING = "decision_making"
    EVALUATION = "evaluation"
    OPTIMIZATION = "optimization"
    INNOVATION = "innovation"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    CODE_REVIEW = "code_review"

@dataclass
class ReasoningStep:
    """推理步骤增强版"""
    step_id: str
    step_type: str
    content: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReasoningTemplate:
    """推理模板"""
    template_id: str
    problem_type: ProblemType
    reasoning_mode: ReasoningMode
    steps: List[Dict[str, Any]]
    prompts: Dict[str, str]
    validation_rules: List[str]
    complexity_level: int
    estimated_time: int  # 预估时间（分钟）
    success_criteria: List[str]
    failure_patterns: List[str]

class EnhancedReasoningTemplates:
    """增强版推理模板系统"""
    
    def __init__(self):
        self.templates: Dict[str, ReasoningTemplate] = {}
        self.template_categories = {}
        self._load_enhanced_templates()
    
    def _load_enhanced_templates(self):
        """加载增强版推理模板"""
        # 结构化分析模板
        self.templates["structured_analytical"] = ReasoningTemplate(
            template_id="structured_analytical_v2",
            problem_type=ProblemType.ANALYTICAL,
            reasoning_mode=ReasoningMode.STRUCTURED,
            steps=[
                "problem_definition",
                "information_gathering", 
                "data_analysis",
                "hypothesis_formation",
                "evidence_evaluation",
                "logical_deduction",
                "conclusion_validation",
                "recommendation_generation"
            ],
            prompts={
                "problem_definition": "明确定义问题的核心目标、约束条件和成功标准。识别关键利益相关者和预期结果。",
                "information_gathering": "系统性地收集所有相关信息、数据和上下文。验证信息来源的可靠性和时效性。",
                "data_analysis": "深入分析收集的数据，识别模式、趋势和异常。使用适当的分析工具和方法。",
                "hypothesis_formation": "基于数据分析形成初步假设。确保假设是可验证的、具体的和相关的。",
                "evidence_evaluation": "系统性地评估证据对假设的支持程度。识别证据的质量、相关性和局限性。",
                "logical_deduction": "进行严格的逻辑推理，从假设和证据中得出结论。确保推理过程的严密性。",
                "conclusion_validation": "验证结论的合理性和可靠性。检查是否存在逻辑漏洞或未考虑的因素。",
                "recommendation_generation": "基于验证的结论生成具体的行动建议。确保建议是可行的、可操作的。"
            },
            validation_rules=[
                "检查问题定义是否清晰明确",
                "验证信息收集的全面性",
                "确认数据分析方法的适当性",
                "评估假设的合理性",
                "验证证据的充分性",
                "检查逻辑推理的严密性",
                "确认结论的有效性",
                "评估建议的可行性"
            ],
            complexity_level=8,
            estimated_time=15,
            success_criteria=[
                "问题定义清晰且可衡量",
                "信息收集全面且可靠", 
                "分析方法科学且适当",
                "假设合理且可验证",
                "证据充分且相关",
                "推理逻辑严密",
                "结论有效且可靠",
                "建议具体且可行"
            ],
            failure_patterns=[
                "问题定义模糊或过于宽泛",
                "信息收集不完整或有偏见",
                "分析方法不当或有误",
                "假设不合理或无法验证",
                "证据不足或不相关",
                "逻辑推理有漏洞",
                "结论无效或不可靠",
                "建议空泛或不可行"
            ]
        )
        
        # 创新设计模板
        self.templates["creative_design"] = ReasoningTemplate(
            template_id="creative_design_v2",
            problem_type=ProblemType.DESIGN,
            reasoning_mode=ReasoningMode.CREATIVE,
            steps=[
                "requirement_analysis",
                "constraint_identification",
                "ideation_brainstorm",
                "concept_development",
                "feasibility_assessment",
                "prototype_design",
                "iteration_refinement",
                "final_validation"
            ],
            prompts={
                "requirement_analysis": "深入分析用户需求、技术约束和业务目标。识别显性和隐性需求。",
                "constraint_identification": "明确技术、时间、资源和法规约束。区分硬约束和软约束。",
                "ideation_brainstorm": "进行发散性思维，产生大量创新想法。鼓励突破性思考。",
                "concept_development": "将创意想法发展为具体概念和方案。确保概念的完整性和一致性。",
                "feasibility_assessment": "评估技术可行性和资源需求。进行风险评估和成本效益分析。",
                "prototype_design": "设计原型和验证方案。创建可测试的模型或演示。",
                "iteration_refinement": "基于反馈迭代改进方案。持续优化设计和功能。",
                "final_validation": "验证最终方案是否满足所有需求和约束。"
            },
            validation_rules=[
                "检查需求分析的完整性",
                "验证约束识别的准确性",
                "评估创意的原创性",
                "确认概念的可行性",
                "验证原型的有效性",
                "检查迭代的改进效果",
                "确认最终方案的完整性"
            ],
            complexity_level=9,
            estimated_time=20,
            success_criteria=[
                "需求分析全面且准确",
                "约束识别完整且合理",
                "创意想法丰富且新颖",
                "概念设计完整且一致",
                "可行性评估准确",
                "原型设计有效",
                "迭代改进明显",
                "最终方案完善"
            ],
            failure_patterns=[
                "需求分析不完整",
                "约束识别有遗漏",
                "创意想法贫乏",
                "概念设计不完整",
                "可行性评估错误",
                "原型设计无效",
                "迭代改进不明显",
                "最终方案不完善"
            ]
        )
        
        # 批判性思维模板
        self.templates["critical_evaluation"] = ReasoningTemplate(
            template_id="critical_evaluation_v2",
            problem_type=ProblemType.EVALUATION,
            reasoning_mode=ReasoningMode.CRITICAL,
            steps=[
                "claim_identification",
                "source_credibility",
                "evidence_examination",
                "bias_detection",
                "logical_analysis",
                "alternative_consideration",
                "error_identification",
                "judgment_formation"
            ],
            prompts={
                "claim_identification": "识别需要评估的核心主张、论点或结论。明确评估的对象和范围。",
                "source_credibility": "评估信息来源的可信度、权威性和偏见。检查作者资质和动机。",
                "evidence_examination": "仔细检查支持证据的质量、数量和相关性。识别证据的局限性和漏洞。",
                "bias_detection": "检测潜在的偏见、假设和预设立场。包括认知偏见和文化偏见。",
                "logical_analysis": "分析论证的逻辑结构。检查推理的有效性、一致性和完整性。",
                "alternative_consideration": "考虑替代观点、解释和可能性。评估其他合理的解释。",
                "error_identification": "识别论证中的逻辑错误、事实错误和推理谬误。",
                "judgment_formation": "基于全面分析形成判断。明确结论的确定性和局限性。"
            },
            validation_rules=[
                "检查主张识别的准确性",
                "验证来源可信度评估",
                "确认证据检查的全面性",
                "评估偏见检测的深度",
                "验证逻辑分析的严密性",
                "检查替代考虑的充分性",
                "确认错误识别的准确性",
                "评估判断形成的合理性"
            ],
            complexity_level=10,
            estimated_time=18,
            success_criteria=[
                "主张识别准确",
                "来源可信度评估合理",
                "证据检查全面",
                "偏见检测深入",
                "逻辑分析严密",
                "替代考虑充分",
                "错误识别准确",
                "判断形成合理"
            ],
            failure_patterns=[
                "主张识别错误",
                "来源可信度评估不当",
                "证据检查不全面",
                "偏见检测不深入",
                "逻辑分析有漏洞",
                "替代考虑不充分",
                "错误识别不准确",
                "判断形成不合理"
            ]
        )
        
        # 系统架构模板
        self.templates["systemic_architecture"] = ReasoningTemplate(
            template_id="systemic_architecture_v2",
            problem_type=ProblemType.ARCHITECTURE,
            reasoning_mode=ReasoningMode.SYSTEMIC,
            steps=[
                "system_boundary",
                "component_analysis",
                "interaction_mapping",
                "dependency_analysis",
                "interface_design",
                "integration_strategy",
                "scalability_planning",
                "maintenance_design"
            ],
            prompts={
                "system_boundary": "明确定义系统的边界、范围和上下文。识别内部和外部元素。",
                "component_analysis": "分析系统的关键组件、模块和子系统。理解其功能和职责。",
                "interaction_mapping": "绘制组件间的交互关系和数据流。识别同步和异步交互。",
                "dependency_analysis": "分析组件间的依赖关系。识别强依赖和弱依赖，循环依赖。",
                "interface_design": "设计组件间的接口和协议。确保接口的一致性和可扩展性。",
                "integration_strategy": "制定系统集成策略。考虑渐进式集成和回滚方案。",
                "scalability_planning": "规划系统的可扩展性。考虑水平扩展和垂直扩展策略。",
                "maintenance_design": "设计系统的可维护性。考虑监控、日志、更新和故障处理。"
            },
            validation_rules=[
                "检查系统边界定义的清晰性",
                "验证组件分析的完整性",
                "确认交互映射的准确性",
                "评估依赖分析的深度",
                "验证接口设计的一致性",
                "检查集成策略的可行性",
                "确认可扩展性规划的合理性",
                "评估可维护性设计的完善性"
            ],
            complexity_level=10,
            estimated_time=25,
            success_criteria=[
                "系统边界清晰",
                "组件分析完整",
                "交互映射准确",
                "依赖分析深入",
                "接口设计一致",
                "集成策略可行",
                "可扩展性规划合理",
                "可维护性设计完善"
            ],
            failure_patterns=[
                "系统边界模糊",
                "组件分析不完整",
                "交互映射不准确",
                "依赖分析不深入",
                "接口设计不一致",
                "集成策略不可行",
                "可扩展性规划不合理",
                "可维护性设计不完善"
            ]
        )
        
        # 优化模板
        self.templates["optimization_strategy"] = ReasoningTemplate(
            template_id="optimization_strategy_v2",
            problem_type=ProblemType.OPTIMIZATION,
            reasoning_mode=ReasoningMode.STRATEGIC,
            steps=[
                "performance_baseline",
                "bottleneck_identification",
                "optimization_target",
                "strategy_development",
                "implementation_plan",
                "measurement_framework",
                "iteration_cycle",
                "result_validation"
            ],
            prompts={
                "performance_baseline": "建立性能基线，明确当前状态和关键指标。收集基准数据。",
                "bottleneck_identification": "识别性能瓶颈和约束点。使用分析工具和方法定位问题。",
                "optimization_target": "设定明确的优化目标和期望改进程度。确保目标是可衡量的。",
                "strategy_development": "开发优化策略和方法。考虑短期和长期解决方案。",
                "implementation_plan": "制定详细的实施计划。包括时间表、资源和风险评估。",
                "measurement_framework": "建立测量框架和指标体系。确保能够准确评估优化效果。",
                "iteration_cycle": "建立迭代优化循环。包括测试、测量、调整和验证。",
                "result_validation": "验证优化结果的有效性和可持续性。确认达到预期目标。"
            },
            validation_rules=[
                "检查性能基线的准确性",
                "验证瓶颈识别的正确性",
                "确认优化目标的合理性",
                "评估策略的可行性",
                "验证实施计划的完整性",
                "检查测量框架的有效性",
                "确认迭代循环的合理性",
                "评估结果验证的严格性"
            ],
            complexity_level=9,
            estimated_time=22,
            success_criteria=[
                "性能基线准确",
                "瓶颈识别正确",
                "优化目标合理",
                "策略可行",
                "计划完整",
                "框架有效",
                "循环合理",
                "结果可信"
            ],
            failure_patterns=[
                "性能基线不准确",
                "瓶颈识别错误",
                "优化目标不合理",
                "策略不可行",
                "计划不完整",
                "框架无效",
                "循环不合理",
                "结果不可信"
            ]
        )
        
        # 按类别组织模板
        self.template_categories = {
            "分析类": ["structured_analytical"],
            "设计类": ["creative_design"],
            "评估类": ["critical_evaluation"],
            "架构类": ["systemic_architecture"],
            "优化类": ["optimization_strategy"]
        }
    
    def get_template(self, reasoning_mode: ReasoningMode, problem_type: ProblemType) -> Optional[ReasoningTemplate]:
        """获取推理模板"""
        template_key = f"{reasoning_mode.value}_{problem_type.value}"
        
        # 精确匹配
        if template_key in self.templates:
            return self.templates[template_key]
        
        # 模糊匹配 - 基于问题类型
        for key, template in self.templates.items():
            if template.problem_type == problem_type:
                return template
        
        # 模糊匹配 - 基于推理模式
        for key, template in self.templates.items():
            if template.reasoning_mode == reasoning_mode:
                return template
        
        # 返回默认模板
        return self.templates.get("structured_analytical")
    
    def get_template_by_id(self, template_id: str) -> Optional[ReasoningTemplate]:
        """根据ID获取模板"""
        for template in self.templates.values():
            if template.template_id == template_id:
                return template
        return None
    
    def list_templates(self) -> Dict[str, List[str]]:
        """列出所有模板"""
        return {
            "total_count": len(self.templates),
            "by_category": self.template_categories,
            "all_templates": list(self.templates.keys())
        }
    
    def validate_template_completeness(self, template: ReasoningTemplate) -> Dict[str, Any]:
        """验证模板完整性"""
        validation_result = {
            "template_id": template.template_id,
            "is_complete": True,
            "issues": [],
            "score": 0.0
        }
        
        # 检查必需字段
        required_fields = ["steps", "prompts", "validation_rules", "success_criteria"]
        for field in required_fields:
            if not getattr(template, field):
                validation_result["is_complete"] = False
                validation_result["issues"].append(f"缺少必需字段: {field}")
        
        # 检查步骤和提示的一致性
        if len(template.steps) != len(template.prompts):
            validation_result["is_complete"] = False
            validation_result["issues"].append("步骤数量与提示数量不匹配")
        
        # 检查步骤完整性
        for step in template.steps:
            if step not in template.prompts:
                validation_result["is_complete"] = False
                validation_result["issues"].append(f"步骤 '{step}' 缺少对应的提示")
        
        # 计算完整性分数
        total_checks = 5
        passed_checks = total_checks - len(validation_result["issues"])
        validation_result["score"] = passed_checks / total_checks
        
        return validation_result
    
    def analyze_problem_type(self, problem_description: str) -> Dict[str, Any]:
        """分析问题类型"""
        problem_lower = problem_description.lower()
        
        # 关键词匹配
        type_mapping = {
            ProblemType.DESIGN: ["设计", "创建", "构建", "开发", "架构"],
            ProblemType.ANALYTICAL: ["分析", "评估", "审查", "检查", "诊断"],
            ProblemType.PROBLEM_SOLVING: ["解决", "修复", "处理", "应对", "排除"],
            ProblemType.DECISION_MAKING: ["决定", "选择", "推荐", "建议", "判断"],
            ProblemType.EVALUATION: ["评价", "评审", "验证", "测试", "确认"],
            ProblemType.OPTIMIZATION: ["优化", "改进", "提升", "增强", "效率"],
            ProblemType.INNOVATION: ["创新", "突破", "发明", "创造", "革新"],
            ProblemType.DEBUGGING: ["调试", "排错", "故障", "错误", "异常"],
            ProblemType.ARCHITECTURE: ["架构", "结构", "系统", "设计", "框架"],
            ProblemType.CODE_REVIEW: ["代码审查", "代码评审", "代码检查", "代码分析"]
        }
        
        scores = {}
        for problem_type, keywords in type_mapping.items():
            score = sum(1 for keyword in keywords if keyword in problem_lower)
            if score > 0:
                scores[problem_type] = score
        
        # 选择得分最高的类型
        if scores:
            best_match = max(scores, key=scores.get)
            confidence = scores[best_match] / max(len(problem_lower.split()), 1)
        else:
            best_match = ProblemType.ANALYTICAL  # 默认类型
            confidence = 0.3
        
        return {
            "detected_type": best_match.value,
            "confidence": min(confidence, 1.0),
            "scores": {k.value: v for k, v in scores.items()},
            "description": problem_description[:100] + "..." if len(problem_description) > 100 else problem_description
        }

# 全局模板实例
_enhanced_templates_instance = None

def get_enhanced_templates() -> EnhancedReasoningTemplates:
    """获取增强版推理模板实例"""
    global _enhanced_templates_instance
    if _enhanced_templates_instance is None:
        _enhanced_templates_instance = EnhancedReasoningTemplates()
    return _enhanced_templates_instance

if __name__ == "__main__":
    # 测试代码
    templates = get_enhanced_templates()
    
    print("增强版ARQ推理模板系统 V2")
    print("=" * 50)
    
    # 列出所有模板
    template_list = templates.list_templates()
    print(f"模板总数: {template_list['total_count']}")
    print("按类别分组:")
    for category, template_names in template_list['by_category'].items():
        print(f"  {category}: {', '.join(template_names)}")
    
    # 测试问题类型分析
    test_problems = [
        "如何设计一个高可用的微服务架构系统？",
        "分析当前系统的性能瓶颈并提供优化建议",
        "评估这个算法的时间复杂度和空间复杂度",
        "解决数据库查询性能慢的问题",
        "选择最适合的技术栈来开发Web应用"
    ]
    
    print("\n问题类型分析测试:")
    for problem in test_problems:
        analysis = templates.analyze_problem_type(problem)
        print(f"问题: {problem}")
        print(f"  类型: {analysis['detected_type']}, 置信度: {analysis['confidence']:.2f}")
    
    # 测试模板获取
    print("\n模板获取测试:")
    template = templates.get_template(ReasoningMode.STRUCTURED, ProblemType.ANALYTICAL)
    if template:
        print(f"获取到模板: {template.template_id}")
        print(f"步骤数量: {len(template.steps)}")
        print(f"复杂度: {template.complexity_level}/10")
        print(f"预估时间: {template.estimated_time}分钟")