#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 智能任务分解器 V2
将复杂的大任务智能拆分成可并行执行的小任务，最大化并行执行效率。
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import logging
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
import asyncio

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """任务类型"""
    ANALYSIS = "analysis"          # 分析类任务
    DESIGN = "design"              # 设计类任务
    IMPLEMENTATION = "implementation"  # 实现类任务
    TESTING = "testing"            # 测试类任务
    DEPLOYMENT = "deployment"      # 部署类任务
    INTEGRATION = "integration"    # 集成类任务
    OPTIMIZATION = "optimization"  # 优化类任务
    DOCUMENTATION = "documentation" # 文档类任务

class DependencyType(Enum):
    """依赖类型"""
    SEQUENTIAL = "sequential"      # 顺序依赖：必须等待前序任务完成
    DATA = "data"                  # 数据依赖：需要前序任务的数据输出
    RESOURCE = "resource"          # 资源依赖：需要前序任务释放资源
    COORDINATION = "coordination"  # 协调依赖：需要前序任务的协调结果

@dataclass
class DecomposedTask:
    """分解后的任务"""
    task_id: str
    original_task: str
    subtask_description: str
    task_type: TaskType
    priority: int
    estimated_duration: float
    estimated_complexity: float
    required_experts: List[str]
    dependencies: List[Tuple[str, DependencyType]]
    parallelizable: bool
    resource_requirements: Dict[str, Any]
    quality_criteria: List[str]
    output_format: str
    
    # 执行状态
    status: str = "pending"
    assigned_agent: Optional[str] = None
    actual_duration: Optional[float] = None
    quality_score: Optional[float] = None

class TaskComplexityAnalyzer:
    """任务复杂度分析器"""
    
    def __init__(self):
        # 复杂度关键词映射
        self.complexity_keywords = {
            "simple": {
                "keywords": ["简单", "基础", "快速", "直接", "基本"],
                "base_complexity": 1.0
            },
            "moderate": {
                "keywords": ["分析", "设计", "实现", "配置", "开发"],
                "base_complexity": 2.5
            },
            "complex": {
                "keywords": ["架构", "系统", "集成", "优化", "重构"],
                "base_complexity": 5.0
            },
            "expert": {
                "keywords": ["高级", "深度", "复杂", "专家", "专业"],
                "base_complexity": 7.5
            },
            "master": {
                "keywords": ["大师", "精通", "全面", "综合", "战略"],
                "base_complexity": 10.0
            }
        }
        
        # 领域复杂度调整因子
        self.domain_factors = {
            "人工智能": 1.8,
            "区块链": 1.7,
            "量子计算": 2.0,
            "网络安全": 1.6,
            "大数据": 1.5,
            "云计算": 1.4,
            "移动开发": 1.2,
            "前端开发": 1.1,
            "后端开发": 1.3,
            "数据库": 1.3,
            "DevOps": 1.5
        }
    
    def analyze_complexity(self, task: str, domain: Optional[str] = None) -> float:
        """分析任务复杂度"""
        task_lower = task.lower()
        
        # 基础复杂度
        base_complexity = 1.0
        
        for level, config in self.complexity_keywords.items():
            for keyword in config["keywords"]:
                if keyword in task_lower:
                    base_complexity = max(base_complexity, config["base_complexity"])
                    break
        
        # 领域调整
        domain_factor = 1.0
        if domain and domain in self.domain_factors:
            domain_factor = self.domain_factors[domain]
        
        # 任务长度调整
        length_factor = min(1.0 + len(task) / 500, 2.0)
        
        # 关键词数量调整
        keyword_count = sum(1 for config in self.complexity_keywords.values() 
                           for keyword in config["keywords"] if keyword in task_lower)
        keyword_factor = 1.0 + keyword_count * 0.1
        
        final_complexity = base_complexity * domain_factor * length_factor * keyword_factor
        
        logger.debug(f"任务复杂度分析: {task[:50]}... -> {final_complexity:.2f}")
        return min(final_complexity, 10.0)  # 最大复杂度为10

class DependencyAnalyzer:
    """依赖关系分析器"""
    
    def __init__(self):
        # 依赖关系关键词
        self.dependency_keywords = {
            DependencyType.SEQUENTIAL: [
                "然后", "接着", "之后", "接下来", "随后", "before", "after", "then", "next"
            ],
            DependencyType.DATA: [
                "基于", "根据", "使用", "依赖", "需要", "require", "based on", "using", "depending on"
            ],
            DependencyType.RESOURCE: [
                "释放", "占用", "资源", "环境", "setup", "teardown", "resource", "environment"
            ],
            DependencyType.COORDINATION: [
                "协调", "沟通", "讨论", "review", "coordinate", "communicate", "discuss"
            ]
        }
    
    def analyze_dependencies(self, task: str, context_tasks: List[DecomposedTask]) -> List[Tuple[str, DependencyType]]:
        """分析任务依赖关系"""
        dependencies = []
        task_lower = task.lower()
        
        # 分析上下文任务的依赖
        for context_task in context_tasks:
            # 基于关键词分析依赖类型
            for dep_type, keywords in self.dependency_keywords.items():
                for keyword in keywords:
                    if keyword in task_lower:
                        dependencies.append((context_task.task_id, dep_type))
                        break
        
        # 分析任务描述中的隐含依赖
        if any(word in task_lower for word in ["首先", "第一步", "initial"]):
            # 这是初始任务，可能被其他任务依赖
            pass
        
        if any(word in task_lower for word in ["最后", "最终", "final", "complete"]):
            # 这是最终任务，可能依赖其他所有任务
            for context_task in context_tasks:
                if (context_task.task_id, DependencyType.SEQUENTIAL) not in dependencies:
                    dependencies.append((context_task.task_id, DependencyType.SEQUENTIAL))
        
        return dependencies

class TaskDecomposer:
    """
    智能任务分解器
    """
    
    def __init__(self):
        self.complexity_analyzer = TaskComplexityAnalyzer()
        self.dependency_analyzer = DependencyAnalyzer()
        
        # 任务分解规则
        self.decomposition_rules = self._load_decomposition_rules()
        
        # 专家需求映射
        self.expert_requirements = self._load_expert_requirements()
    
    def _load_decomposition_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载任务分解规则"""
        return {
            "软件开发": [
                {
                    "pattern": r"(设计|架构|architecture)",
                    "subtasks": [
                        {"description": "需求分析和系统架构设计", "type": TaskType.DESIGN, "priority": 1},
                        {"description": "技术选型和框架确定", "type": TaskType.DESIGN, "priority": 2}
                    ],
                    "parallelizable": False
                },
                {
                    "pattern": r"(开发|实现|implement)",
                    "subtasks": [
                        {"description": "核心模块开发", "type": TaskType.IMPLEMENTATION, "priority": 3},
                        {"description": "辅助功能开发", "type": TaskType.IMPLEMENTATION, "priority": 4},
                        {"description": "接口开发和集成", "type": TaskType.INTEGRATION, "priority": 5}
                    ],
                    "parallelizable": True
                },
                {
                    "pattern": r"(测试|test)",
                    "subtasks": [
                        {"description": "单元测试编写", "type": TaskType.TESTING, "priority": 6},
                        {"description": "集成测试执行", "type": TaskType.TESTING, "priority": 7},
                        {"description": "性能测试验证", "type": TaskType.TESTING, "priority": 8}
                    ],
                    "parallelizable": True
                }
            ],
            "系统优化": [
                {
                    "pattern": r"(优化|optimize|performance)",
                    "subtasks": [
                        {"description": "性能瓶颈分析", "type": TaskType.ANALYSIS, "priority": 1},
                        {"description": "优化方案设计", "type": TaskType.DESIGN, "priority": 2},
                        {"description": "优化实施", "type": TaskType.IMPLEMENTATION, "priority": 3},
                        {"description": "效果验证", "type": TaskType.TESTING, "priority": 4}
                    ],
                    "parallelizable": False
                }
            ]
        }
    
    def _load_expert_requirements(self) -> Dict[TaskType, List[str]]:
        """加载专家需求映射"""
        return {
            TaskType.ANALYSIS: ["分析师", "架构师"],
            TaskType.DESIGN: ["架构师", "设计师", "技术专家"],
            TaskType.IMPLEMENTATION: ["开发专家", "工程师"],
            TaskType.TESTING: ["测试专家", "质量工程师"],
            TaskType.DEPLOYMENT: ["DevOps专家", "运维工程师"],
            TaskType.INTEGRATION: ["集成专家", "系统工程师"],
            TaskType.OPTIMIZATION: ["性能专家", "优化工程师"],
            TaskType.DOCUMENTATION: ["文档专家", "技术作家"]
        }
    
    def decompose_task(self, original_task: str, domain: Optional[str] = None,
                      max_subtasks: int = 10) -> List[DecomposedTask]:
        """
        分解任务
        """
        start_time = time.time()
        
        # 1. 分析任务复杂度
        complexity = self.complexity_analyzer.analyze_complexity(original_task, domain)
        
        # 2. 识别任务类型
        task_types = self._identify_task_types(original_task)
        
        # 3. 生成子任务
        subtasks = self._generate_subtasks(original_task, task_types, complexity, domain)
        
        # 4. 分析依赖关系
        self._analyze_task_dependencies(subtasks)
        
        # 5. 优化并行性
        self._optimize_parallelization(subtasks)
        
        # 6. 分配资源需求
        self._assign_resource_requirements(subtasks)
        
        # 7. 设置质量标准
        self._set_quality_criteria(subtasks)
        
        # 记录分解时间
        decomposition_time = time.time() - start_time
        logger.info(f"任务分解完成: {len(subtasks)} 个子任务 (耗时: {decomposition_time:.3f}s)")
        
        return subtasks[:max_subtasks]  # 限制最大子任务数量
    
    def _identify_task_types(self, task: str) -> List[TaskType]:
        """识别任务类型"""
        task_lower = task.lower()
        identified_types = []
        
        type_keywords = {
            TaskType.ANALYSIS: ["分析", "研究", "调研", "评估", "analyze", "research", "evaluate"],
            TaskType.DESIGN: ["设计", "架构", "规划", "design", "architecture", "plan"],
            TaskType.IMPLEMENTATION: ["实现", "开发", "编码", "implement", "develop", "code"],
            TaskType.TESTING: ["测试", "验证", "检验", "test", "verify", "validate"],
            TaskType.DEPLOYMENT: ["部署", "发布", "上线", "deploy", "release", "launch"],
            TaskType.INTEGRATION: ["集成", "整合", "连接", "integrate", "merge", "connect"],
            TaskType.OPTIMIZATION: ["优化", "改进", "提升", "optimize", "improve", "enhance"],
            TaskType.DOCUMENTATION: ["文档", "说明", "手册", "document", "manual", "guide"]
        }
        
        for task_type, keywords in type_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                identified_types.append(task_type)
        
        # 如果没有识别到类型，默认为实现类型
        if not identified_types:
            identified_types = [TaskType.IMPLEMENTATION]
        
        return identified_types
    
    def _generate_subtasks(self, original_task: str, task_types: List[TaskType], 
                          complexity: float, domain: Optional[str]) -> List[DecomposedTask]:
        """生成子任务"""
        subtasks = []
        
        # 基于任务类型生成默认子任务
        for i, task_type in enumerate(task_types):
            base_subtasks = self._get_base_subtasks_for_type(task_type, original_task)
            subtasks.extend(base_subtasks)
        
        # 如果复杂度很高，进一步细分
        if complexity > 7.0:
            subtasks = self._subdivide_complex_tasks(subtasks)
        
        # 为每个子任务分配ID和属性
        for i, subtask in enumerate(subtasks):
            subtask.task_id = f"task_{i:03d}"
            subtask.original_task = original_task
            subtask.estimated_complexity = complexity / len(subtasks)
            subtask.required_experts = self.expert_requirements.get(subtask.task_type, ["通用专家"])
            
            # 估算持续时间
            base_duration = self._estimate_task_duration(subtask.task_type, subtask.estimated_complexity)
            subtask.estimated_duration = base_duration
            
            # 设置输出格式
            subtask.output_format = self._get_output_format(subtask.task_type)
        
        return subtasks
    
    def _get_base_subtasks_for_type(self, task_type: TaskType, original_task: str) -> List[DecomposedTask]:
        """获取任务类型的基础子任务"""
        base_subtasks = []
        
        if task_type == TaskType.ANALYSIS:
            base_subtasks = [
                DecomposedTask(
                    task_id="",  # 稍后设置
                    original_task=original_task,
                    subtask_description="收集和分析需求",
                    task_type=TaskType.ANALYSIS,
                    priority=1,
                    estimated_duration=0.0,  # 稍后设置
                    estimated_complexity=0.0,  # 稍后设置
                    required_experts=[],
                    dependencies=[],
                    parallelizable=False,
                    resource_requirements={"time": "30分钟", "tools": ["需求分析工具"]},
                    quality_criteria=[],
                    output_format=""
                ),
                DecomposedTask(
                    task_id="",  # 稍后设置
                    original_task=original_task,
                    subtask_description="技术可行性分析",
                    task_type=TaskType.ANALYSIS,
                    priority=2,
                    estimated_duration=0.0,  # 稍后设置
                    estimated_complexity=0.0,  # 稍后设置
                    required_experts=[],
                    dependencies=[],
                    parallelizable=True,
                    resource_requirements={"time": "45分钟", "tools": ["技术评估工具"]},
                    quality_criteria=[],
                    output_format=""
                )
            ]
        
        elif task_type == TaskType.DESIGN:
            base_subtasks = [
                DecomposedTask(
                    task_id="",  # 稍后设置
                    original_task=original_task,
                    subtask_description="系统架构设计",
                    task_type=TaskType.DESIGN,
                    priority=3,
                    estimated_duration=0.0,  # 稍后设置
                    estimated_complexity=0.0,  # 稍后设置
                    required_experts=[],
                    dependencies=[],
                    parallelizable=False,
                    resource_requirements={"time": "2小时", "tools": ["设计工具", "架构图工具"]},
                    quality_criteria=[],
                    output_format=""
                ),
                DecomposedTask(
                    task_id="",  # 稍后设置
                    original_task=original_task,
                    subtask_description="详细设计文档",
                    task_type=TaskType.DESIGN,
                    priority=4,
                    estimated_duration=0.0,  # 稍后设置
                    estimated_complexity=0.0,  # 稍后设置
                    required_experts=[],
                    dependencies=[],
                    parallelizable=True,
                    resource_requirements={"time": "1.5小时", "tools": ["文档工具"]},
                    quality_criteria=[],
                    output_format=""
                )
            ]
        
        elif task_type == TaskType.IMPLEMENTATION:
            base_subtasks = [
                DecomposedTask(
                    task_id="",  # 稍后设置
                    original_task=original_task,
                    subtask_description="核心功能实现",
                    task_type=TaskType.IMPLEMENTATION,
                    priority=5,
                    estimated_duration=0.0,  # 稍后设置
                    estimated_complexity=0.0,  # 稍后设置
                    required_experts=[],
                    dependencies=[],
                    parallelizable=False,
                    resource_requirements={"time": "4小时", "tools": ["开发环境", "代码编辑器"]},
                    quality_criteria=[],
                    output_format=""
                ),
                DecomposedTask(
                    task_id="",  # 稍后设置
                    original_task=original_task,
                    subtask_description="辅助功能实现",
                    task_type=TaskType.IMPLEMENTATION,
                    priority=6,
                    estimated_duration=0.0,  # 稍后设置
                    estimated_complexity=0.0,  # 稍后设置
                    required_experts=[],
                    dependencies=[],
                    parallelizable=True,
                    resource_requirements={"time": "2小时", "tools": ["开发环境"]},
                    quality_criteria=[],
                    output_format=""
                )
            ]
        
        # 添加其他任务类型的默认子任务...
        
        return base_subtasks
    
    def _subdivide_complex_tasks(self, subtasks: List[DecomposedTask]) -> List[DecomposedTask]:
        """细分复杂任务"""
        refined_subtasks = []
        
        for subtask in subtasks:
            if subtask.estimated_complexity > 3.0:
                # 将复杂任务进一步细分
                refinement_factor = int(subtask.estimated_complexity / 2.0)
                
                for i in range(refinement_factor):
                    refined_subtask = DecomposedTask(
                        task_id="",  # 稍后设置
                        original_task=subtask.original_task,
                        subtask_description=f"{subtask.subtask_description} - 第{i+1}部分",
                        task_type=subtask.task_type,
                        priority=subtask.priority + i,
                        estimated_duration=subtask.estimated_duration / refinement_factor,
                        estimated_complexity=subtask.estimated_complexity / refinement_factor,
                        required_experts=subtask.required_experts,
                        dependencies=subtask.dependencies.copy(),
                        parallelizable=subtask.parallelizable,
                        resource_requirements=subtask.resource_requirements.copy(),
                        quality_criteria=subtask.quality_criteria.copy(),
                        output_format=subtask.output_format
                    )
                    refined_subtasks.append(refined_subtask)
            else:
                refined_subtasks.append(subtask)
        
        return refined_subtasks
    
    def _analyze_task_dependencies(self, subtasks: List[DecomposedTask]):
        """分析任务依赖关系"""
        for i, subtask in enumerate(subtasks):
            # 基于优先级设置依赖
            dependencies = []
            for j in range(i):
                if subtasks[j].priority < subtask.priority:
                    dependencies.append((subtasks[j].task_id, DependencyType.SEQUENTIAL))
            
            # 特殊依赖规则
            if subtask.task_type == TaskType.IMPLEMENTATION:
                # 实现任务通常依赖设计任务
                for dep_task in subtasks[:i]:
                    if dep_task.task_type in [TaskType.ANALYSIS, TaskType.DESIGN]:
                        dependencies.append((dep_task.task_id, DependencyType.DATA))
            
            elif subtask.task_type == TaskType.TESTING:
                # 测试任务依赖实现任务
                for dep_task in subtasks[:i]:
                    if dep_task.task_type in [TaskType.IMPLEMENTATION, TaskType.INTEGRATION]:
                        dependencies.append((dep_task.task_id, DependencyType.DATA))
            
            subtask.dependencies = dependencies
    
    def _optimize_parallelization(self, subtasks: List[DecomposedTask]):
        """优化并行性"""
        # 识别可以并行执行的任务
        for subtask in subtasks:
            # 如果没有强依赖，标记为可并行
            if not any(dep[1] == DependencyType.SEQUENTIAL for dep in subtask.dependencies):
                subtask.parallelizable = True
            
            # 同类型任务通常可以并行
            if subtask.task_type in [TaskType.IMPLEMENTATION, TaskType.TESTING, TaskType.OPTIMIZATION]:
                subtask.parallelizable = True
    
    def _assign_resource_requirements(self, subtasks: List[DecomposedTask]):
        """分配资源需求"""
        for subtask in subtasks:
            # 基于任务类型分配资源
            if subtask.task_type == TaskType.IMPLEMENTATION:
                subtask.resource_requirements.update({
                    "cpu": "medium",
                    "memory": "medium",
                    "storage": "low"
                })
            elif subtask.task_type == TaskType.ANALYSIS:
                subtask.resource_requirements.update({
                    "cpu": "low",
                    "memory": "medium",
                    "storage": "medium"
                })
            elif subtask.task_type == TaskType.DESIGN:
                subtask.resource_requirements.update({
                    "cpu": "low",
                    "memory": "low",
                    "storage": "medium"
                })
    
    def _set_quality_criteria(self, subtasks: List[DecomposedTask]):
        """设置质量标准"""
        quality_standards = {
            TaskType.ANALYSIS: ["完整性", "准确性", "可行性"],
            TaskType.DESIGN: ["合理性", "可扩展性", "一致性"],
            TaskType.IMPLEMENTATION: ["功能性", "性能", "可维护性"],
            TaskType.TESTING: ["覆盖率", "准确性", "可靠性"],
            TaskType.DEPLOYMENT: ["稳定性", "安全性", "可用性"],
            TaskType.INTEGRATION: ["兼容性", "数据一致性", "接口稳定性"],
            TaskType.OPTIMIZATION: ["性能提升", "资源利用率", "响应时间"],
            TaskType.DOCUMENTATION: ["清晰性", "完整性", "准确性"]
        }
        
        for subtask in subtasks:
            subtask.quality_criteria = quality_standards.get(subtask.task_type, ["质量标准待定"])
    
    def _estimate_task_duration(self, task_type: TaskType, complexity: float) -> float:
        """估算任务持续时间（小时）"""
        base_durations = {
            TaskType.ANALYSIS: 1.0,
            TaskType.DESIGN: 2.0,
            TaskType.IMPLEMENTATION: 4.0,
            TaskType.TESTING: 2.0,
            TaskType.DEPLOYMENT: 1.0,
            TaskType.INTEGRATION: 3.0,
            TaskType.OPTIMIZATION: 2.5,
            TaskType.DOCUMENTATION: 1.5
        }
        
        base_duration = base_durations.get(task_type, 2.0)
        complexity_factor = 1.0 + (complexity - 1.0) / 5.0
        
        return base_duration * complexity_factor
    
    def _get_output_format(self, task_type: TaskType) -> str:
        """获取输出格式"""
        formats = {
            TaskType.ANALYSIS: "分析报告 (Markdown)",
            TaskType.DESIGN: "设计文档 (Markdown + 架构图)",
            TaskType.IMPLEMENTATION: "源代码 (Python/JavaScript等)",
            TaskType.TESTING: "测试报告 + 测试用例",
            TaskType.DEPLOYMENT: "部署脚本 + 配置文件",
            TaskType.INTEGRATION: "集成方案 + 接口文档",
            TaskType.OPTIMIZATION: "优化报告 + 性能数据",
            TaskType.DOCUMENTATION: "用户手册 + API文档"
        }
        
        return formats.get(task_type, "待定格式")

# --- 使用示例 ---
def main():
    """示例使用"""
    # 创建任务分解器
    decomposer = TaskDecomposer()
    
    # 示例任务
    complex_task = """
    开发一个高性能的电商系统，需要包含用户管理、商品管理、订单处理、支付集成、
    库存管理、推荐系统等功能。系统需要支持高并发访问，具备良好的可扩展性和
    安全性。要求提供完整的前端界面、后端API、数据库设计和部署方案。
    """
    
    # 分解任务
    subtasks = decomposer.decompose_task(
        original_task=complex_task,
        domain="电商系统开发",
        max_subtasks=15
    )
    
    print(f"任务分解结果: 共 {len(subtasks)} 个子任务")
    print("=" * 80)
    
    for i, subtask in enumerate(subtasks, 1):
        print(f"{i:2d}. [{subtask.task_type.value.upper()}] {subtask.subtask_description}")
        print(f"    优先级: {subtask.priority}, 复杂度: {subtask.estimated_complexity:.1f}")
        print(f"    预估时间: {subtask.estimated_duration:.1f}小时")
        print(f"    所需专家: {', '.join(subtask.required_experts)}")
        print(f"    可并行: {'是' if subtask.parallelizable else '否'}")
        if subtask.dependencies:
            deps = [f"{dep[0]}({dep[1].value})" for dep in subtask.dependencies]
            print(f"    依赖: {', '.join(deps)}")
        print()
    
    # 统计信息
    parallelizable_count = sum(1 for t in subtasks if t.parallelizable)
    total_duration = sum(t.estimated_duration for t in subtasks)
    sequential_duration = sum(t.estimated_duration for t in subtasks if not t.parallelizable)
    
    print("=" * 80)
    print("分解统计:")
    print(f"可并行任务: {parallelizable_count}/{len(subtasks)} ({parallelizable_count/len(subtasks)*100:.1f}%)")
    print(f"串行总时间: {sequential_duration:.1f}小时")
    print(f"并行总时间: {total_duration:.1f}小时")
    print(f"并行加速比: {total_duration/sequential_duration:.2f}x")

if __name__ == "__main__":
    main()