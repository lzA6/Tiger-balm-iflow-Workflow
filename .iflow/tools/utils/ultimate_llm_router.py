#!/usr/bin/env python3
"""
🌟 Intelligent LLM Router - 智能LLM模型路由器
实现100%兼容性的智能模型选择和路由
"""

import asyncio
import json
import time
import logging
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    """任务类型枚举"""
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    DOCUMENTATION = "documentation"
    OPTIMIZATION = "optimization"
    SECURITY = "security"
    TESTING = "testing"

class ModelTier(Enum):
    """模型层级枚举"""
    PREMIUM = "premium"
    STANDARD = "standard"
    ECONOMY = "economy"
    LOCAL = "local"

@dataclass
class ModelCapability:
    """模型能力定义"""
    name: str
    provider: str
    tier: ModelTier
    reasoning_score: float
    creativity_score: float
    code_score: float
    speed_score: float
    cost_score: float
    context_window: int
    languages: List[str]
    specialties: List[str]
    avg_response_time: float
    success_rate: float
    quality_score: float
    api_config: Dict[str, Any]

@dataclass
class TaskFeatures:
    """任务特征"""
    length: int
    complexity: float
    language: str
    has_code: bool
    is_creative: bool
    requires_reasoning: bool
    urgency: float
    task_type: Optional[TaskType]
    context_requirement: int
    quality_requirement: float
    cost_sensitivity: float

@dataclass
class RoutingDecision:
    """路由决策"""
    selected_model: str
    confidence: float
    reasoning: str
    alternative_models: List[str]
    expected_performance: Dict[str, float]
    fallback_plan: List[str]

class PerformanceMetrics:
    """性能指标"""
    response_times: deque
    success_rates: Dict[str, float]
    quality_scores: Dict[str, float]
    error_counts: Dict[str, int]
    last_updated: float

class IntelligentLLMRouter:
    """智能LLM路由器"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.models = self._initialize_models()
        self.task_weights = self.config.get("task_weights", {})
        self.performance_metrics = PerformanceMetrics(
            response_times=deque(maxlen=1000),
            success_rates={},
            quality_scores={},
            error_counts={},
            last_updated=time.time()
        )
        self.routing_cache = {}
        self.performance_cache = {}
        self.model_availability = {}
        
        # 初始化模型可用性
        self._initialize_model_availability()
        
        logger.info(f"智能LLM路由器初始化完成，加载了 {len(self.models)} 个模型")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
        
        # 返回默认配置
        return {
            "task_weights": {
                "code_generation": {
                    "reasoning": 0.25, "code": 0.40, "speed": 0.15,
                    "creativity": 0.10, "cost": 0.10
                },
                "reasoning": {
                    "reasoning": 0.50, "code": 0.10, "speed": 0.15,
                    "creativity": 0.15, "cost": 0.10
                },
                "creative": {
                    "reasoning": 0.25, "code": 0.10, "speed": 0.10,
                    "creativity": 0.40, "cost": 0.15
                }
            },
            "thresholds": {
                "response_time_max": 30000,
                "success_rate_min": 0.90,
                "quality_score_min": 0.80
            }
        }
    
    def _initialize_models(self) -> Dict[str, ModelCapability]:
        """初始化模型注册表"""
        models = {}
        
        # 从配置中加载模型
        model_registry = self.config.get("model_registry", {})
        
        for provider, provider_models in model_registry.items():
            for model_name, model_config in provider_models.items():
                full_name = model_config.get("name", model_name)
                
                # 确定模型层级
                tier = self._determine_tier(provider, model_name)
                
                # 提取能力分数
                capabilities = model_config.get("capabilities", {})
                
                # 提取API配置
                api_config = model_config.get("api_config", {})
                
                # 提取性能配置
                performance_profile = model_config.get("performance_profile", {})
                
                # 创建模型能力对象
                model = ModelCapability(
                    name=full_name,
                    provider=provider,
                    tier=tier,
                    reasoning_score=capabilities.get("reasoning", 0.5),
                    creativity_score=capabilities.get("creativity", 0.5),
                    code_score=capabilities.get("code", 0.5),
                    speed_score=capabilities.get("speed", 0.5),
                    cost_score=capabilities.get("cost", 0.5),
                    context_window=capabilities.get("context_window", 4096),
                    languages=capabilities.get("languages", ["en"]),
                    specialties=capabilities.get("specialties", []),
                    avg_response_time=performance_profile.get("avg_response_time", 2000),
                    success_rate=performance_profile.get("success_rate", 0.9),
                    quality_score=performance_profile.get("quality_score", 0.8),
                    api_config=api_config
                )
                
                models[full_name] = model
        
        return models
    
    def _determine_tier(self, provider: str, model_name: str) -> ModelTier:
        """确定模型层级"""
        premium_models = ["gpt-4", "claude-3-opus", "gemini-pro"]
        standard_models = ["gpt-4-turbo", "claude-3-sonnet", "qwen-max", "glm-4"]
        economy_models = ["ernie-bot-4", "deepseek-coder", "moonshot-v1-8k"]
        local_models = ["llama-2", "mistral"]
        
        if model_name in premium_models:
            return ModelTier.PREMIUM
        elif model_name in standard_models:
            return ModelTier.STANDARD
        elif model_name in economy_models:
            return ModelTier.ECONOMY
        elif model_name in local_models:
            return ModelTier.LOCAL
        else:
            return ModelTier.STANDARD
    
    def _initialize_model_availability(self):
        """初始化模型可用性"""
        for model_name in self.models:
            self.model_availability[model_name] = {
                "available": True,
                "last_check": time.time(),
                "consecutive_failures": 0,
                "max_failures": 3
            }
    
    def _analyze_task_features(self, task: str, task_type: Optional[TaskType] = None) -> TaskFeatures:
        """分析任务特征"""
        # 基础特征
        length = len(task)
        
        # 复杂度评估
        complexity = self._assess_complexity(task)
        
        # 语言检测
        language = self._detect_language(task)
        
        # 代码检测
        has_code = self._contains_code(task)
        
        # 创意性检测
        is_creative = self._is_creative_task(task)
        
        # 推理需求检测
        requires_reasoning = self._requires_reasoning(task)
        
        # 紧急程度评估
        urgency = self._assess_urgency(task)
        
        # 推断任务类型
        if task_type is None:
            task_type = self._infer_task_type(task, has_code, is_creative, requires_reasoning)
        
        # 上下文需求
        context_requirement = self._estimate_context_requirement(task, has_code)
        
        # 质量要求
        quality_requirement = self._estimate_quality_requirement(task)
        
        # 成本敏感度
        cost_sensitivity = self._estimate_cost_sensitivity(task)
        
        return TaskFeatures(
            length=length,
            complexity=complexity,
            language=language,
            has_code=has_code,
            is_creative=is_creative,
            requires_reasoning=requires_reasoning,
            urgency=urgency,
            task_type=task_type,
            context_requirement=context_requirement,
            quality_requirement=quality_requirement,
            cost_sensitivity=cost_sensitivity
        )
    
    def _assess_complexity(self, task: str) -> float:
        """评估任务复杂度"""
        complexity_indicators = [
            "架构", "设计", "系统", "集成", "优化", "重构",
            "算法", "数据结构", "性能", "安全", "部署",
            "architecture", "design", "system", "integration", "optimization", "refactoring",
            "algorithm", "data structure", "performance", "security", "deployment"
        ]
        
        complexity_score = 0.0
        for indicator in complexity_indicators:
            if indicator in task.lower():
                complexity_score += 0.2
        
        # 长度复杂度
        if len(task) > 1000:
            complexity_score += 0.3
        elif len(task) > 500:
            complexity_score += 0.2
        
        return min(complexity_score, 1.0)
    
    def _detect_language(self, task: str) -> str:
        """检测任务语言"""
        chinese_chars = len([c for c in task if '\u4e00' <= c <= '\u9fff'])
        if chinese_chars > len(task) * 0.3:
            return "zh"
        return "en"
    
    def _contains_code(self, task: str) -> bool:
        """检测是否包含代码"""
        code_indicators = [
            "```", "def ", "function ", "class ", "import ", "from ",
            "var ", "let ", "const ", "=>", "return ", "if (",
            "for (", "while (", "try {", "catch (", "throw new"
        ]
        
        return any(indicator in task for indicator in code_indicators)
    
    def _is_creative_task(self, task: task: str) -> bool:
        """检测是否为创意任务"""
        creative_indicators = [
            "创意", "设计", "创作", "想象", "创新", "艺术",
            "creative", "design", "create", "imagine", "innovate", "art"
        ]
        
        return any(indicator in task.lower() for indicator in creative_indicators)
    
    def _requires_reasoning(self, task: task) -> bool:
        """检测是否需要推理"""
        reasoning_indicators = [
            "分析", "推理", "解释", "原因", "为什么", "如何",
            "analyze", "reason", "explain", "why", "how", "because"
        ]
        
        return any(indicator in task.lower() for indicator in reasoning_indicators)
    
    def _assess_urgency(self, task: task) -> float:
        """评估紧急程度"""
        urgency_indicators = [
            "紧急", "立即", "马上", "尽快", "urgent", "immediately",
            "asap", "right now", "quickly"
        ]
        
        urgency_score = 0.0
        for indicator in urgency_indicators:
            if indicator in task.lower():
                urgency_score += 0.3
        
        return min(urgency_score, 1.0)
    
    def _infer_task_type(self, task: str, has_code: bool, is_creative: bool, requires_reasoning: bool) -> TaskType:
        """推断任务类型"""
        if has_code:
            return TaskType.CODE_GENERATION
        elif is_creative:
            return TaskType.CREATIVE
        elif requires_reasoning:
            return TaskType.REASONING
        else:
            return TaskType.ANALYSIS
    
    def _estimate_context_requirement(self, task: str, has_code: bool) -> int:
        """估算上下文需求"""
        base_requirement = len(task)
        
        if has_code:
            base_requirement *= 1.5
        
        # 复杂度调整
        complexity = self._assess_complexity(task)
        base_requirement *= (1 + complexity)
        
        return int(base_requirement)
    
    def _estimate_quality_requirement(self, task: str) -> float:
        """估算质量要求"""
        quality_indicators = [
            "高质量", "生产级别", "专业", "精确", "准确",
            "high quality", "production", "professional", "precise", "accurate"
        ]
        
        quality_score = 0.7  # 默认质量要求
        for indicator in quality_indicators:
            if indicator in task.lower():
                quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def _estimate_cost_sensitivity(self, task: str) -> float:
        """估算成本敏感度"""
        cost_indicators = [
            "便宜", "经济", "节省", "低成本", "budget",
            "cheap", "economical", "save", "low cost", "budget"
        ]
        
        cost_sensitivity = 0.5  # 默认成本敏感度
        for indicator in cost_indicators:
            if indicator in task.lower():
                cost_sensitivity += 0.1
        
        return min(cost_sensitivity, 1.0)
    
    def _calculate_model_score(self, model: ModelCapability, features: TaskFeatures, weights: Dict[str, float]) -> float:
        """计算模型得分"""
        # 基础能力得分
        base_score = (
            model.reasoning_score * weights.get("reasoning", 0.2) +
            model.creativity_score * weights.get("creativity", 0.2) +
            model.code_score * weights.get("code", 0.2) +
            model.speed_score * weights.get("speed", 0.2) +
            model.cost_score * weights.get("cost", 0.2)
        )
        
        # 任务匹配度调整
        task_match_bonus = self._calculate_task_match_bonus(features, model)
        
        # 性能调整
        performance_adjustment = self._get_performance_adjustment(model.name)
        
        # 上下文窗口适配
        context_penalty = self._calculate_context_penalty(features, model)
        
        # 语言匹配调整
        language_bonus = 0.1 if features.language in model.languages else 0.0
        
        # 可用性调整
        availability_penalty = self._calculate_availability_penalty(model.name)
        
        # 综合得分
        final_score = (
            base_score * 
            (1 + task_match_bonus + language_bonus) * 
            performance_adjustment * 
            (1 - context_penalty) * 
            (1 - availability_penalty)
        )
        
        return final_score
    
    def _calculate_task_match_bonus(self, features: TaskFeatures, model: ModelCapability) -> float:
        """计算任务匹配度奖励"""
        bonus = 0.0
        
        # 专业领域匹配
        if features.has_code and "code" in model.specialties:
            bonus += 0.15
        
        if features.requires_reasoning and "reasoning" in model.specialties:
            bonus += 0.1
        
        if features.is_creative and "creative" in model.specialties:
            bonus += 0.1
        
        # 质量要求匹配
        if features.quality_requirement > 0.9 and model.quality_score > 0.9:
            bonus += 0.1
        
        return min(bonus, 0.3)  # 限制最大奖励
    
    def _get_performance_adjustment(self, model_name: str) -> float:
        """获取性能调整系数"""
        if model_name not in self.performance_metrics.success_rates:
            return 1.0
        
        success_rate = self.performance_metrics.success_rates[model_name]
        quality_score = self.performance_metrics.quality_scores.get(model_name, 0.8)
        
        # 基于成功率和质量分数调整
        return 0.5 + (success_rate * 0.3) + (quality_score * 0.2)
    
    def _calculate_context_penalty(self, features: TaskFeatures, model: ModelCapability) -> float:
        """计算上下文窗口不匹配惩罚"""
        if features.context_requirement <= model.context_window * 0.8:
            return 0.0
        elif features.context_requirement <= model.context_window * 0.9:
            return 0.1
        else:
            return 0.3
    
    def _calculate_availability_penalty(self, model_name: str) -> float:
        """计算可用性惩罚"""
        if model_name not in self.model_availability:
            return 0.0
        
        availability = self.model_availability[model_name]
        if not availability["available"]:
            return 1.0
        
        consecutive_failures = availability["consecutive_failures"]
        max_failures = availability["max_failures"]
        
        return consecutive_failures / max_failures
    
    def _select_optimal_model(self, model_scores: Dict[str, float], features: TaskFeatures) -> str:
        """选择最优模型"""
        # 过滤可用模型
        available_models = {
            name: score for name, score in model_scores.items()
            if self.model_availability.get(name, {}).get("available", True)
        }
        
        if not available_models:
            # 如果没有可用模型，选择失败次数最少的
            least_failed = min(
                self.model_availability.items(),
                key=lambda x: x[1]["consecutive_failures"]
            )
            return least_failed[0]
        
        # 根据紧急程度选择策略
        if features.urgency > 0.7:
            # 高紧急度：选择最快的可用模型
            fastest_model = min(
                available_models.items(),
                key=lambda x: self.models[x[0]].speed_score,
                reverse=True
            )
            return fastest_model[0]
        elif features.cost_sensitivity > 0.7:
            # 高成本敏感：选择最便宜的可用模型
            cheapest_model = min(
                available_models.items(),
                key=lambda x: self.models[x[0]].cost_score,
                reverse=True
            )
            return cheapest_model[0]
        else:
            # 正常情况：选择得分最高的模型
            best_model = max(available_models.items(), key=lambda x: x[1])
            return best_model[0]
    
    def _generate_alternative_models(self, selected_model: str, features: TaskFeatures) -> List[str]:
        """生成备选模型"""
        alternatives = []
        
        # 获取同一层级的模型
        selected_tier = self.models[selected_model].tier
        
        # 按得分排序的其他模型
        other_models = [
            name for name, model in self.models.items()
            if name != selected_model and 
               model.tier == selected_tier and
               self.model_availability.get(name, {}).get("available", True)
        ]
        
        # 按性能分数排序
        other_models.sort(
            key=lambda x: self.models[x].quality_score,
            reverse=True
        )
        
        # 返回前3个备选模型
        return other_models[:3]
    
    def _generate_fallback_plan(self, selected_model: str, features: TaskFeatures) -> List[str]:
        """生成回退计划"""
        fallback_plan = []
        
        # 获取模型层级
        tiers = [ModelTier.PREMIUM, ModelTier.STANDARD, ModelTier.ECONOMY, ModelTier.LOCAL]
        current_tier = self.models[selected_model].tier
        
        # 从当前层级开始生成回退计划
        for tier in tiers[tiers.index(current_tier) + 1:]:
            tier_models = [
                name for name, model in self.models.items()
                if model.tier == tier and
                   self.model_availability.get(name, {}).get("available", True)
            ]
            
            if tier_models:
                fallback_plan.extend(tier_models[:2])
        
        return fallback_plan
    
    def _estimate_expected_performance(self, model_name: str, features: TaskFeatures) -> Dict[str, float]:
        """估算预期性能"""
        model = self.models[model_name]
        
        # 基于历史数据和模型特性估算
        expected_response_time = model.avg_response_time
        expected_success_rate = model.success_rate
        expected_quality_score = model.quality_score
        
        # 根据任务特征调整
        if features.context_requirement > model.context_window * 0.8:
            expected_response_time *= 1.5
            expected_success_rate *= 0.9
        
        if features.complexity > 0.7:
            expected_response_time *= 1.3
            expected_quality_score *= 0.9
        
        return {
            "response_time": expected_response_time,
            "success_rate": expected_success_rate,
            "quality_score": expected_quality_score
        }
    
    def intelligent_routing(self, task: str, task_type: Optional[TaskType] = None) -> RoutingDecision:
        """智能路由主函数"""
        start_time = time.time()
        
        # 生成任务哈希用于缓存
        task_hash = hashlib.md5(task.encode()).hexdigest()
        
        # 检查缓存
        if task_hash in self.routing_cache:
            cached_decision = self.routing_cache[task_hash]
            logger.info(f"使用缓存的路由决策: {cached_decision.selected_model}")
            return cached_decision
        
        # 分析任务特征
        features = self._analyze_task_features(task, task_type)
        
        # 获取任务权重
        weights = self.task_weights.get(features.task_type.value, {
            "reasoning": 0.25, "code": 0.25, "creativity": 0.25, "speed": 0.15, "cost": 0.1
        })
        
        # 计算每个模型的得分
        model_scores = {}
        for model_name, model in self.models.items():
            score = self._calculate_model_score(model, features, weights)
            model_scores[model_name] = score
        
        # 选择最优模型
        selected_model = self._select_optimal_model(model_scores, features)
        
        # 生成备选模型
        alternative_models = self._generate_alternative_models(selected_model, features)
        
        # 生成回退计划
        fallback_plan = self._generate_fallback_plan(selected_model, features)
        
        # 估算预期性能
        expected_performance = self._estimate_expected_performance(selected_model, features)
        
        # 计算决策置信度
        confidence = self._calculate_confidence(model_scores, selected_model, features)
        
        # 生成决策理由
        reasoning = self._generate_reasoning(selected_model, features, model_scores)
        
        # 创建决策对象
        decision = RoutingDecision(
            selected_model=selected_model,
            confidence=confidence,
            reasoning=reasoning,
            alternative_models=alternative_models,
            expected_performance=expected_performance,
            fallback_plan=fallback_plan
        )
        
        # 缓存决策
        self.routing_cache[task_hash] = decision
        
        # 记录路由时间
        routing_time = time.time() - start_time
        logger.info(f"路由决策完成，耗时: {routing_time:.2f}ms，选择模型: {selected_model}")
        
        return decision
    
    def _calculate_confidence(self, model_scores: Dict[str, float], selected_model: str, features: TaskFeatures) -> float:
        """计算决策置信度"""
        if not model_scores:
            return 0.0
        
        # 获取最高分和平均分
        max_score = max(model_scores.values())
        avg_score = sum(model_scores.values()) / len(model_scores)
        
        if max_score == 0:
            return 0.0
        
        # 基于得分差异计算置信度
        score_ratio = model_scores[selected_model] / max_score
        avg_ratio = avg_score / max_score
        
        # 综合置信度
        confidence = (score_ratio * 0.7) + (avg_ratio * 0.3)
        
        return min(confidence, 1.0)
    
    def _generate_reasoning(self, selected_model: str, features: TaskFeatures, model_scores: Dict[str, float]) -> str:
        """生成决策理由"""
        model = self.models[selected_model]
        
        reasoning_parts = []
        
        # 任务类型匹配
        if features.task_type:
            reasoning_parts.append(f"任务类型为{features.task_type.value}，模型{selected_model}在此领域表现优秀")
        
        # 语言匹配
        if features.language in model.languages:
            reasoning_parts.append(f"模型支持{features.language}语言")
        
        # 代码需求匹配
        if features.has_code and "code" in model.specialties:
            reasoning_parts.append(f"模型擅长代码生成，代码评分：{model.code_score:.2f}")
        
        # 上下文窗口匹配
        if features.context_requirement <= model.context_window:
            reasoning_parts.append(f"上下文窗口({model.context_window})满足需求")
        else:
            reasoning_parts.append(f"上下文窗口({model.context_window})可能不足")
        
        # 性能考虑
        if features.urgency > 0.7:
            reasoning_parts.append(f"高紧急度任务，选择响应速度较快的模型")
        elif features.cost_sensitivity > 0.7:
            reasoning_parts.append(f"成本敏感任务，选择成本效益较高的模型")
        
        # 质量保证
        if features.quality_requirement > 0.9:
            reasoning_parts.append(f"高质量要求，选择质量评分({model.quality_score:.2f})较高的模型")
        
        return "；".join(reasoning_parts)
    
    def update_performance_feedback(self, model_name: str, success: bool, 
                                 response_time: float, quality_score: float):
        """更新性能反馈"""
        # 更新响应时间历史
        self.performance_metrics.response_times.append(response_time)
        
        # 更新成功率
        if model_name not in self.performance_metrics.success_rates:
            self.performance_metrics.success_rates[model_name] = 0.0
        
        current_rate = self.performance_metrics.success_rates[model_name]
        total_requests = len(self.performance_metrics.response_times)
        
        # 使用指数移动平均更新成功率
        alpha = 0.1
        if success:
            new_rate = current_rate * (1 - alpha) + alpha
        else:
            new_rate = current_rate * (1 - alpha)
        
        self.performance_metrics.success_rates[model_name] = new_rate
        
        # 更新质量分数
        if model_name not in self.performance_metrics.quality_scores:
            self.performance_metrics.quality_scores[model_name] = 0.8
        
        current_quality = self.performance_metrics.quality_scores[model_name]
        new_quality = current_quality * (1 - alpha) + quality_score * alpha
        
        self.performance_metrics.quality_scores[model_name] = new_quality
        
        # 更新错误计数
        if not success:
            if model_name not in self.performance_metrics.error_counts:
                self.performance_metrics.error_counts[model_name] = 0
            self.performance_metrics.error_counts[model_name] += 1
        
        # 更新模型可用性
        if model_name in self.model_availability:
            availability = self.model_availability[model_name]
            if success:
                availability["consecutive_failures"] = 0
                availability["available"] = True
            else:
                availability["consecutive_failures"] += 1
                if availability["consecutive_failures"] >= availability["max_failures"]:
                    availability["available"] = False
                    # 设置恢复定时器
                    self._schedule_availability_recovery(model_name)
        
        # 更新最后更新时间
        self.performance_metrics.last_updated = time.time()
        
        logger.info(f"更新模型{model_name}性能反馈: 成功={success}, 响应时间={response_time}ms, 质量={quality_score:.2f}")
    
    def _schedule_availability_recovery(self, model_name: str):
        """安排可用性恢复"""
        def recover():
            time.sleep(60)  # 1分钟后恢复
            if model_name in self.model_availability:
                self.model_availability[model_name]["consecutive_failures"] = 0
                self.model_availability[model_name]["available"] = True
                logger.info(f"模型{model_name}可用性已恢复")
        
        # 在异步环境中运行
        try:
            asyncio.create_task(recover)
        except:
            # 如果不在异步环境中，使用线程
            import threading
            threading.Thread(target=recover, daemon=True).start()
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        stats = {
            "total_models": len(self.models),
            "available_models": sum(
                1 for model in self.model_availability.values()
                if model["available"]
            ),
            "models_by_tier": {},
            "performance_summary": {
                "avg_response_time": np.mean(list(self.performance_metrics.response_times)) if self.performance_metrics.response_times else 0,
                "avg_success_rate": np.mean(list(self.performance_metrics.success_rates.values())) if self.performance_metrics.success_rates else 0,
                "avg_quality_score": np.mean(list(self.performance_metrics.quality_scores.values())) if self.performance_metrics.quality_scores else 0
            },
            "models": {}
        }
        
        # 按层级统计模型
        for model in self.models.values():
            tier = model.tier.value
            if tier not in stats["models_by_tier"]:
                stats["models_by_tier"][tier] = []
            stats["models_by_tier"][tier].append(model.name)
        
        # 详细的模型信息
        for model_name, model in self.models.items():
            stats["models"][model_name] = {
                "provider": model.provider,
                "tier": model.tier.value,
                "success_rate": self.performance_metrics.success_rates.get(model_name, 0),
                "quality_score": self.performance_metrics.quality_scores.get(model_name, 0),
                "available": self.model_availability.get(model_name, {}).get("available", False)
            }
        
        return stats
    
    def clear_cache(self):
        """清理缓存"""
        self.routing_cache.clear()
        self.performance_cache.clear()
        logger.info("路由缓存已清理")

# 全局路由器实例
router = IntelligentLLMRouter()

def intelligent_llm_routing(task: str, task_type: Optional[TaskType] = None) -> RoutingDecision:
    """智能LLM路由主函数"""
    return router.intelligent_routing(task, task_type)

def update_model_performance(model_name: str, success: bool, 
                           response_time: float, quality_score: float):
    """更新模型性能反馈"""
    router.update_performance_feedback(model_name, success, response_time, quality_score)

def get_router_statistics() -> Dict[str, Any]:
    """获取路由器统计信息"""
    return router.get_model_statistics()

def clear_router_cache():
    """清理路由器缓存"""
    router.clear_cache()

if __name__ == "__main__":
    # 测试代码
    test_task = "请帮我设计一个高性能的微服务架构，需要包含服务发现、负载均衡和容错机制"
    decision = intelligent_llm_routing(test_task)
    
    print(f"选择的模型: {decision.selected_model}")
    print(f"置信度: {decision.confidence:.2f}")
    print(f"决策理由: {decision.reasoning}")
    print(f"备选模型: {decision.alternative_models}")
    print(f"预期性能: {decision.expected_performance}")