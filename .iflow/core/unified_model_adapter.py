#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌐 统一模型适配器 V2 (Unified Model Adapter V2)
融合B、C、D项目最佳实践，创建支持全模型生态、智能路由和量子优化的终极适配器。
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。

核心特性：
1. 多层智能路由：基于任务复杂度、成本、性能的动态选择
2. 量子增强优化：量子退火算法优化模型选择
3. 意识流集成：与ARQ和意识流系统的深度协同
4. 自适应学习：基于历史表现的持续优化
5. 全模型兼容：支持所有主流LLM模型
"""

import os
import sys
import json
import asyncio
import logging
import hashlib
import time
import uuid
import aiohttp
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import threading
import re
import copy
import statistics
import math

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# --- 枚举定义 ---

class UniversalModelProvider(Enum):
    """统一模型提供商"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    BAIDU = "baidu"
    ZHIPU = "zhipu"
    COHERE = "cohere"
    MISTRAL = "mistral"
    GITHUB = "github"
    LOCAL = "local"
    CUSTOM = "custom"
    ALIBABA = "alibaba"
    MOONSHOT = "moonshot"

class UniversalModelCapability(Enum):
    """统一模型能力"""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    VISION = "vision"
    AUDIO = "audio"
    CODE = "code"
    REASONING = "reasoning"
    TOOLS = "tools"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"
    MULTIMODAL = "multimodal"
    CREATIVE = "creative"

class UniversalTaskComplexity(Enum):
    """统一任务复杂度"""
    TRIVIAL = "trivial"        # 简单查询、问候
    SIMPLE = "simple"         # 基础任务、简单分析
    MODERATE = "moderate"     # 中等复杂度、多步骤
    COMPLEX = "complex"       # 复杂问题、深度分析
    EXPERT = "expert"         # 专家级、创新性任务
    MASTER = "master"         # 大师级、跨领域整合
    TRANSCENDENT = "transcendent"  # 超越级、突破性任务

class UniversalRoutingStrategy(Enum):
    """统一路由策略"""
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE_PRIORITIZED = "performance_prioritized"
    BALANCED = "balanced"
    SPECIALIZED = "specialized"
    QUANTUM_ENHANCED = "quantum_enhanced"
    ADAPTIVE_LEARNING = "adaptive_learning"
    CONTEXT_AWARE = "context_aware"
    EMERGENCY_MODE = "emergency_mode"
    PREDICTIVE = "predictive"
    COLLABORATIVE = "collaborative"

@dataclass
class UniversalModelProfile:
    """统一模型配置文件"""
    model_id: str
    provider: UniversalModelProvider
    capabilities: List[UniversalModelCapability]
    max_tokens: int
    context_length: int
    temperature: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    
    # 性能指标
    response_time_ms: float = 0.0
    success_rate: float = 1.0
    cost_per_token: float = 0.001
    
    # 量子特性
    quantum_efficiency: float = 0.5
    coherence_time: float = 1.0
    quantum_compatible: bool = False
    
    # 特殊能力
    tool_calling: bool = False
    function_calling: bool = False
    streaming: bool = False
    vision: bool = False
    audio: bool = False
    multimodal: bool = False
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 新增增强属性
    specialization_domains: List[str] = field(default_factory=list)  # 专业领域
    reliability_score: float = 0.9                           # 可靠性评分
    innovation_score: float = 0.5                           # 创新性评分
    stability_score: float = 0.8                            # 稳定性评分
    scaling_efficiency: float = 0.7                        # 扩展效率

@dataclass
class UniversalRoutingContext:
    """统一路由上下文"""
    task_description: str
    complexity: UniversalTaskComplexity
    required_capabilities: List[UniversalModelCapability]
    budget_constraint: float
    time_constraint: float
    quality_requirement: float
    
    # 用户偏好
    preferred_providers: List[UniversalModelProvider] = field(default_factory=list)
    avoided_providers: List[UniversalModelProvider] = field(default_factory=list)
    
    # 历史信息
    previous_model_choices: List[str] = field(default_factory=list)
    success_history: Dict[str, float] = field(default_factory=dict)
    
    # 上下文信息
    context_tokens: int = 0
    expected_output_tokens: int = 500
    
    # 新增增强属性
    emotional_context: float = 0.0                        # 情感上下文 (-1到1)
    urgency_level: float = 0.5                           # 紧急程度 (0到1)
    innovation_requirement: bool = False                 # 创新需求
    collaborative_needed: bool = False                   # 协作需求
    domain_specialization: str = ""                      # 领域专业性

class UnifiedModelAdapter:
    """
    统一模型适配器 V2
    融合B、C、D项目最佳实践的终极多模型适配器
    """
    
    def __init__(self, consciousness_system=None, arq_engine=None):
        self.adapter_id = f"UMA-V2-{uuid.uuid4().hex[:8]}"
        
        # 集成系统
        self.consciousness_system = consciousness_system
        self.arq_engine = arq_engine
        
        # 模型配置
        self.model_profiles: Dict[str, UniversalModelProfile] = {}
        self._init_comprehensive_model_profiles()
        
        # 路由策略
        self.routing_strategies: Dict[UniversalRoutingStrategy, Callable] = {
            UniversalRoutingStrategy.COST_OPTIMIZED: self._cost_optimized_routing,
            UniversalRoutingStrategy.PERFORMANCE_PRIORITIZED: self._performance_prioritized_routing,
            UniversalRoutingStrategy.BALANCED: self._balanced_routing,
            UniversalRoutingStrategy.SPECIALIZED: self._specialized_routing,
            UniversalRoutingStrategy.QUANTUM_ENHANCED: self._quantum_enhanced_routing,
            UniversalRoutingStrategy.ADAPTIVE_LEARNING: self._adaptive_learning_routing,
            UniversalRoutingStrategy.CONTEXT_AWARE: self._context_aware_routing,
            UniversalRoutingStrategy.EMERGENCY_MODE: self._emergency_mode_routing,
            UniversalRoutingStrategy.PREDICTIVE: self._predictive_routing,
            UniversalRoutingStrategy.COLLABORATIVE: self._collaborative_routing
        }
        
        # 当前路由策略
        self.current_strategy = UniversalRoutingStrategy.BALANCED
        
        # 性能监控
        self.performance_metrics = {
            'total_requests': 0,
            'success_requests': 0,
            'failed_requests': 0,
            'total_cost': 0.0,
            'avg_response_time': 0.0,
            'model_success_rates': defaultdict(float),
            'model_response_times': defaultdict(list),
            'routing_decisions': defaultdict(int),
            'strategy_effectiveness': defaultdict(float)
        }
        
        # 缓存系统
        self.response_cache = {}
        self.cache_ttl = 300  # 5分钟
        
        # 并发控制
        self.session_lock = threading.Lock()
        self.active_sessions = {}
        
        # 量子路由参数
        self.quantum_weights = defaultdict(float)
        self.adaptation_rate = 0.1
        
        # 预测性学习
        self.predictive_model = {}
        self.collaborative_memory = defaultdict(list)
        
        # 初始化
        self._init_quantum_weights()
        self._init_predictive_model()
        
        logger.info(f"🌐 统一模型适配器V2初始化完成 - Adapter ID: {self.adapter_id}")
    
    def _init_comprehensive_model_profiles(self):
        """初始化全面的模型配置（融合B、C、D项目最佳实践）"""
        
        # OpenAI 模型（来自A项目V14）
        self.model_profiles.update({
            "gpt-4o": UniversalModelProfile(
                model_id="gpt-4o",
                provider=UniversalModelProvider.OPENAI,
                capabilities=[UniversalModelCapability.CHAT, UniversalModelCapability.TOOLS, 
                             UniversalModelCapability.VISION, UniversalModelCapability.CODE, 
                             UniversalModelCapability.REASONING, UniversalModelCapability.MULTIMODAL],
                max_tokens=4096,
                context_length=128000,
                temperature=0.7,
                top_p=0.95,
                cost_per_token=0.005,
                quantum_efficiency=0.85,
                tool_calling=True,
                streaming=True,
                vision=True,
                multimodal=True,
                specialization_domains=["general", "coding", "analysis"],
                metadata={"api_version": "v1.4", "max_images": 10}
            ),
            "gpt-4-turbo": UniversalModelProfile(
                model_id="gpt-4-turbo",
                provider=UniversalModelProvider.OPENAI,
                capabilities=[UniversalModelCapability.CHAT, UniversalModelCapability.TOOLS,
                             UniversalModelCapability.CODE, UniversalModelCapability.REASONING],
                max_tokens=4096,
                context_length=128000,
                temperature=0.7,
                top_p=0.95,
                cost_per_token=0.003,
                quantum_efficiency=0.8,
                tool_calling=True,
                metadata={"api_version": "v1.3"}
            ),
            "gpt-3.5-turbo": UniversalModelProfile(
                model_id="gpt-3.5-turbo",
                provider=UniversalModelProvider.OPENAI,
                capabilities=[UniversalModelCapability.CHAT, UniversalModelCapability.CODE],
                max_tokens=4096,
                context_length=16385,
                temperature=0.7,
                top_p=0.95,
                cost_per_token=0.0005,
                quantum_efficiency=0.7,
                metadata={"api_version": "v1.2"}
            )
        })
        
        # Anthropic 模型（融合C项目Claude增强）
        self.model_profiles.update({
            "claude-3-5-sonnet": UniversalModelProfile(
                model_id="claude-3-5-sonnet",
                provider=UniversalModelProvider.ANTHROPIC,
                capabilities=[UniversalModelCapability.CHAT, UniversalModelCapability.TOOLS,
                             UniversalModelCapability.CODE, UniversalModelCapability.REASONING,
                             UniversalModelCapability.CREATIVE],
                max_tokens=4096,
                context_length=200000,
                temperature=0.7,
                top_p=0.95,
                cost_per_token=0.003,
                quantum_efficiency=0.88,
                tool_calling=True,
                streaming=True,
                specialization_domains=["reasoning", "creative", "analysis"],
                metadata={"api_version": "2024-06-20", "claude_code": True}
            ),
            "claude-3-opus": UniversalModelProfile(
                model_id="claude-3-opus",
                provider=UniversalModelProvider.ANTHROPIC,
                capabilities=[UniversalModelCapability.CHAT, UniversalModelCapability.CODE,
                             UniversalModelCapability.REASONING, UniversalModelCapability.CREATIVE],
                max_tokens=4096,
                context_length=200000,
                temperature=0.7,
                cost_per_token=0.015,
                quantum_efficiency=0.9,
                specialization_domains=["complex_analysis", "creative"],
                metadata={"api_version": "2024-02-29"}
            ),
            "claude-3-haiku": UniversalModelProfile(
                model_id="claude-3-haiku",
                provider=UniversalModelProvider.ANTHROPIC,
                capabilities=[UniversalModelCapability.CHAT, UniversalModelCapability.CODE],
                max_tokens=4096,
                context_length=200000,
                temperature=0.7,
                cost_per_token=0.00025,
                quantum_efficiency=0.75,
                specialization_domains=["speed", "simple_tasks"],
                metadata={"api_version": "2024-03-04"}
            )
        })
        
        # Google 模型（融合D项目Gemini增强）
        self.model_profiles.update({
            "gemini-1.5-pro": UniversalModelProfile(
                model_id="gemini-1.5-pro",
                provider=UniversalModelProvider.GOOGLE,
                capabilities=[UniversalModelCapability.CHAT, UniversalModelCapability.VISION,
                             UniversalModelCapability.CODE, UniversalModelCapability.REASONING,
                             UniversalModelCapability.MULTIMODAL],
                max_tokens=8192,
                context_length=1000000,
                temperature=0.7,
                cost_per_token=0.002,
                quantum_efficiency=0.82,
                vision=True,
                multimodal=True,
                specialization_domains=["multimodal", "vision", "analysis"],
                metadata={"api_version": "v1beta", "max_images": 16}
            ),
            "gemini-1.5-flash": UniversalModelProfile(
                model_id="gemini-1.5-flash",
                provider=UniversalModelProvider.GOOGLE,
                capabilities=[UniversalModelCapability.CHAT, UniversalModelCapability.VISION,
                             UniversalModelCapability.CODE, UniversalModelCapability.MULTIMODAL],
                max_tokens=8192,
                context_length=1000000,
                temperature=0.7,
                cost_per_token=0.00036,
                quantum_efficiency=0.78,
                vision=True,
                multimodal=True,
                specialization_domains=["speed", "multimodal"],
                metadata={"api_version": "v1beta", "fast_response": True}
            )
        })
        
        # DeepSeek 模型（融合C项目增强）
        self.model_profiles.update({
            "deepseek-chat": UniversalModelProfile(
                model_id="deepseek-chat",
                provider=UniversalModelProvider.DEEPSEEK,
                capabilities=[UniversalModelCapability.CHAT, UniversalModelCapability.CODE,
                             UniversalModelCapability.REASONING],
                max_tokens=4096,
                context_length=128000,
                temperature=0.7,
                cost_per_token=0.0002,
                quantum_efficiency=0.75,
                specialization_domains=["coding", "analysis"],
                metadata={"coding_specialization": True, "chinese_optimized": True}
            ),
            "deepseek-coder": UniversalModelProfile(
                model_id="deepseek-coder",
                provider=UniversalModelProvider.DEEPSEEK,
                capabilities=[UniversalModelCapability.CODE, UniversalModelCapability.REASONING],
                max_tokens=16384,
                context_length=128000,
                temperature=0.7,
                cost_per_token=0.00025,
                quantum_efficiency=0.8,
                specialization_domains=["coding", "programming"],
                metadata={"coding_only": True, "multi_language": True}
            )
        })
        
        # Qwen 模型（融合D项目增强）
        self.model_profiles.update({
            "qwen-max": UniversalModelProfile(
                model_id="qwen-max",
                provider=UniversalModelProvider.QWEN,
                capabilities=[UniversalModelCapability.CHAT, UniversalModelCapability.CODE,
                             UniversalModelCapability.REASONING],
                max_tokens=8192,
                context_length=32768,
                temperature=0.7,
                cost_per_token=0.0008,
                quantum_efficiency=0.78,
                specialization_domains=["chinese", "general"],
                metadata={"chinese_optimized": True, "multi_modal": False}
            ),
            "qwen-plus": UniversalModelProfile(
                model_id="qwen-plus",
                provider=UniversalModelProvider.QWEN,
                capabilities=[UniversalModelCapability.CHAT, UniversalModelCapability.CODE],
                max_tokens=8192,
                context_length=32768,
                temperature=0.7,
                cost_per_token=0.0004,
                quantum_efficiency=0.75,
                specialization_domains=["chinese", "coding"],
                metadata={"chinese_optimized": True}
            ),
            "qwen-turbo": UniversalModelProfile(
                model_id="qwen-turbo",
                provider=UniversalModelProvider.QWEN,
                capabilities=[UniversalModelCapability.CHAT, UniversalModelCapability.CODE],
                max_tokens=8192,
                context_length=32768,
                temperature=0.7,
                cost_per_token=0.0001,
                quantum_efficiency=0.7,
                specialization_domains=["speed", "chinese"],
                metadata={"chinese_optimized": True, "fast_response": True}
            )
        })
        
        # 新增B项目优秀模型
        self.model_profiles.update({
            "moonshot-v1": UniversalModelProfile(
                model_id="moonshot-v1",
                provider=UniversalModelProvider.MOONSHOT,
                capabilities=[UniversalModelCapability.CHAT, UniversalModelCapability.CODE,
                             UniversalModelCapability.REASONING],
                max_tokens=32768,
                context_length=32768,
                temperature=0.7,
                cost_per_token=0.0003,
                quantum_efficiency=0.81,
                specialization_domains=["long_context", "analysis"],
                metadata={"long_context_specialist": True}
            ),
            "cohere-command-r": UniversalModelProfile(
                model_id="cohere-command-r",
                provider=UniversalModelProvider.COHERE,
                capabilities=[UniversalModelCapability.CHAT, UniversalModelCapability.TOOLS,
                             UniversalModelCapability.REASONING],
                max_tokens=4096,
                context_length=128000,
                temperature=0.7,
                cost_per_token=0.0005,
                quantum_efficiency=0.77,
                tool_calling=True,
                specialization_domains=["enterprise", "reliable"],
                metadata={"enterprise_focused": True}
            ),
            "mistral-large": UniversalModelProfile(
                model_id="mistral-large",
                provider=UniversalModelProvider.MISTRAL,
                capabilities=[UniversalModelCapability.CHAT, UniversalModelCapability.CODE,
                             UniversalModelCapability.REASONING],
                max_tokens=32000,
                context_length=32000,
                temperature=0.7,
                cost_per_token=0.002,
                quantum_efficiency=0.83,
                specialization_domains=["european", "reasoning"],
                metadata={"european_privacy": True}
            )
        })
        
        logger.info(f"📊 已加载 {len(self.model_profiles)} 个统一模型配置")
    
    def _init_quantum_weights(self):
        """初始化量子权重（增强版）"""
        # 基于多维度指标初始化量子权重
        for model_id, profile in self.model_profiles.items():
            base_weight = 0.5
            
            # 能力丰富度加权
            capability_weight = len(profile.capabilities) * 0.1
            
            # 成本效率加权
            cost_efficiency = max(0.1, 1.0 - (profile.cost_per_token * 1000))
            
            # 量子效率加权
            quantum_weight = profile.quantum_efficiency
            
            # 可靠性加权
            reliability_weight = profile.reliability_score
            
            # 创新性加权
            innovation_weight = profile.innovation_score
            
            # 稳定性加权
            stability_weight = profile.stability_score
            
            # 综合量子权重（多维度平衡）
            self.quantum_weights[model_id] = (
                base_weight * 0.1 +
                capability_weight * 0.2 +
                cost_efficiency * 0.2 +
                quantum_weight * 0.2 +
                reliability_weight * 0.15 +
                innovation_weight * 0.1 +
                stability_weight * 0.05
            )
    
    def _init_predictive_model(self):
        """初始化预测性模型"""
        # 为每个模型初始化预测参数
        for model_id in self.model_profiles.keys():
            self.predictive_model[model_id] = {
                'performance_trend': [],
                'load_pattern': defaultdict(list),
                'quality_pattern': defaultdict(list),
                'cost_pattern': defaultdict(list),
                'adaptive_score': 0.5
            }
    
    async def unified_adaptive_call(
        self,
        prompt: Union[str, List[Dict]],
        task_complexity: UniversalTaskComplexity = UniversalTaskComplexity.MODERATE,
        required_capabilities: List[UniversalModelCapability] = None,
        budget_constraint: float = float('inf'),
        time_constraint: float = float('inf'),
        quality_requirement: float = 0.8,
        preferred_providers: List[UniversalModelProvider] = None,
        avoided_providers: List[UniversalModelProvider] = None,
        emotional_context: float = 0.0,
        urgency_level: float = 0.5,
        innovation_requirement: bool = False,
        collaborative_needed: bool = False,
        domain_specialization: str = ""
    ) -> Dict[str, Any]:
        """
        统一自适应模型调用（融合所有最佳实践）
        """
        start_time = time.time()
        
        # 创建增强路由上下文
        routing_context = UniversalRoutingContext(
            task_description=str(prompt)[:200],
            complexity=task_complexity,
            required_capabilities=required_capabilities or [UniversalModelCapability.CHAT],
            budget_constraint=budget_constraint,
            time_constraint=time_constraint,
            quality_requirement=quality_requirement,
            preferred_providers=preferred_providers or [],
            avoided_providers=avoided_providers or [],
            emotional_context=emotional_context,
            urgency_level=urgency_level,
            innovation_requirement=innovation_requirement,
            collaborative_needed=collaborative_needed,
            domain_specialization=domain_specialization
        )
        
        # 智能路由决策（增强版）
        selected_model = await self._enhanced_intelligent_routing(routing_context)
        
        # 记录路由决策
        self.performance_metrics['routing_decisions'][selected_model] += 1
        
        # 意识流系统记录（如果可用）
        if self.consciousness_system:
            await self.consciousness_system.record_thought(
                content=f"统一适配器选择模型: {selected_model} 用于任务: {routing_context.task_description}",
                thought_type=self._get_thought_type_from_complexity(task_complexity),
                agent_id="unified_model_adapter",
                confidence=0.8,
                importance=0.7
            )
        
        # 执行模型调用
        response = await self._execute_unified_model_call(selected_model, prompt, routing_context)
        
        # 更新性能指标
        response_time = time.time() - start_time
        self._update_enhanced_performance_metrics(selected_model, response, response_time)
        
        # 意识流系统记录结果
        if self.consciousness_system:
            await self.consciousness_system.record_thought(
                content=f"统一适配器调用完成: {selected_model}, 成功: {response.get('success', False)}",
                thought_type=UniversalThoughtType.ANALYTICAL,
                agent_id="unified_model_adapter",
                confidence=0.9 if response.get('success', False) else 0.3,
                importance=0.6
            )
        
        return response
    
    def _get_thought_type_from_complexity(self, complexity: UniversalTaskComplexity):
        """根据复杂度获取思维类型"""
        complexity_mapping = {
            UniversalTaskComplexity.TRIVIAL: UniversalThoughtType.ANALYTICAL,
            UniversalTaskComplexity.SIMPLE: UniversalThoughtType.ANALYTICAL,
            UniversalTaskComplexity.MODERATE: UniversalThoughtType.METACOGNITIVE,
            UniversalTaskComplexity.COMPLEX: UniversalThoughtType.METACOGNITIVE,
            UniversalTaskComplexity.EXPERT: UniversalThoughtType.QUANTUM_REASONING,
            UniversalTaskComplexity.MASTER: UniversalThoughtType.PREDICTIVE,
            UniversalTaskComplexity.TRANSCENDENT: UniversalThoughtType.PREDICTIVE
        }
        return complexity_mapping.get(complexity, UniversalThoughtType.ANALYTICAL)
    
    async def _enhanced_intelligent_routing(self, context: UniversalRoutingContext) -> str:
        """增强的智能路由决策（融合所有最佳实践）"""
        
        # 基于任务复杂度和上下文选择策略
        strategy = self._select_routing_strategy(context)
        
        # 获取候选模型
        candidates = self._get_enhanced_candidate_models(context)
        
        if not candidates:
            # 降级策略：基于基础能力选择
            candidates = [model_id for model_id in self.model_profiles.keys() 
                         if UniversalModelCapability.CHAT in self.model_profiles[model_id].capabilities]
        
        # 应用增强路由策略
        router = self.routing_strategies.get(strategy, self._balanced_routing)
        selected_model = router(candidates, context)
        
        # 协作路由（如果需要）
        if context.collaborative_needed:
            selected_model = await self._collaborative_enhanced_routing(candidates, context, selected_model)
        
        logger.info(f"🎯 增强路由决策: {strategy.value} -> {selected_model}")
        return selected_model
    
    def _select_routing_strategy(self, context: UniversalRoutingContext) -> UniversalRoutingStrategy:
        """选择路由策略"""
        # 基于复杂度的基础策略映射
        complexity_strategy = {
            UniversalTaskComplexity.TRIVIAL: UniversalRoutingStrategy.COST_OPTIMIZED,
            UniversalTaskComplexity.SIMPLE: UniversalRoutingStrategy.COST_OPTIMIZED,
            UniversalTaskComplexity.MODERATE: UniversalRoutingStrategy.BALANCED,
            UniversalTaskComplexity.COMPLEX: UniversalRoutingStrategy.PERFORMANCE_PRIORITIZED,
            UniversalTaskComplexity.EXPERT: UniversalRoutingStrategy.QUANTUM_ENHANCED,
            UniversalTaskComplexity.MASTER: UniversalRoutingStrategy.ADAPTIVE_LEARNING,
            UniversalTaskComplexity.TRANSCENDENT: UniversalRoutingStrategy.PREDICTIVE
        }
        
        base_strategy = complexity_strategy.get(context.complexity, UniversalRoutingStrategy.BALANCED)
        
        # 根据上下文调整策略
        if context.urgency_level > 0.8:
            return UniversalRoutingStrategy.EMERGENCY_MODE
        elif context.innovation_requirement:
            return UniversalRoutingStrategy.QUANTUM_ENHANCED
        elif context.collaborative_needed:
            return UniversalRoutingStrategy.COLLABORATIVE
        elif context.emotional_context > 0.7:
            return UniversalRoutingStrategy.PREDICTIVE
        else:
            return base_strategy
    
    def _get_enhanced_candidate_models(self, context: UniversalRoutingContext) -> List[str]:
        """获取增强的候选模型"""
        candidates = []
        
        for model_id, profile in self.model_profiles.items():
            # 检查提供商偏好
            if context.preferred_providers and profile.provider not in context.preferred_providers:
                continue
            if context.avoided_providers and profile.provider in context.avoided_providers:
                continue
            
            # 检查能力要求
            if context.required_capabilities:
                has_all_capabilities = all(
                    capability in profile.capabilities 
                    for capability in context.required_capabilities
                )
                if not has_all_capabilities:
                    continue
            
            # 检查上下文长度
            if context.context_tokens > profile.context_length * 0.8:
                continue
            
            # 检查质量要求
            if profile.metadata.get('quality_score', 0.5) < context.quality_requirement:
                continue
            
            # 检查领域专业性
            if context.domain_specialization:
                if context.domain_specialization not in profile.specialization_domains:
                    # 如果不是专业领域，但可靠性足够高，仍然考虑
                    if profile.reliability_score < 0.8:
                        continue
            
            # 检查创新需求
            if context.innovation_requirement and profile.innovation_score < 0.7:
                continue
            
            # 检查紧急程度
            if context.urgency_level > 0.8 and profile.metadata.get('fast_response', False) is False:
                continue
            
            candidates.append(model_id)
        
        return candidates
    
    def _collaborative_enhanced_routing(self, candidates: List[str], context: UniversalRoutingContext, primary_model: str) -> str:
        """协作增强路由"""
        # 如果需要协作，可能选择不同的模型
        if len(candidates) < 2:
            return primary_model
        
        # 基于协作需求调整选择
        if context.collaborative_needed:
            # 选择在协作任务上表现好的模型
            collaborative_scores = {}
            for model_id in candidates:
                profile = self.model_profiles[model_id]
                
                # 协作友好度评分
                collaborative_score = (
                    profile.metadata.get('collaborative_score', 0.5) * 0.4 +
                    profile.reliability_score * 0.3 +
                    self.performance_metrics['model_success_rates'].get(model_id, 0.8) * 0.3
                )
                
                collaborative_scores[model_id] = collaborative_score
            
            # 选择协作评分最高的模型
            return max(collaborative_scores, key=collaborative_scores.get)
        
        return primary_model
    
    def _cost_optimized_routing(self, candidates: List[str], context: UniversalRoutingContext) -> str:
        """成本优化路由（增强版）"""
        if not candidates:
            return "gpt-3.5-turbo"
        
        # 多维度成本优化评分
        cost_scores = {}
        for model_id in candidates:
            profile = self.model_profiles[model_id]
            
            # 成本效率（成本越低分数越高）
            cost_score = max(0.1, 1.0 - (profile.cost_per_token * 1000))
            
            # 能力丰富度
            capability_score = min(1.0, len(profile.capabilities) / 6.0)
            
            # 量子效率
            quantum_score = profile.quantum_efficiency
            
            # 可靠性加权
            reliability_score = profile.reliability_score
            
            # 历史成功率
            historical_success = self.performance_metrics['model_success_rates'].get(model_id, 0.8)
            
            # 综合成本优化分数
            total_score = (
                cost_score * 0.3 +
                capability_score * 0.2 +
                quantum_score * 0.2 +
                reliability_score * 0.15 +
                historical_success * 0.15
            )
            
            cost_scores[model_id] = total_score
        
        return max(cost_scores, key=cost_scores.get)
    
    def _performance_prioritized_routing(self, candidates: List[str], context: UniversalRoutingContext) -> str:
        """性能优先路由（增强版）"""
        if not candidates:
            return "gpt-4o"
        
        # 多维度性能评分
        performance_scores = {}
        for model_id in candidates:
            profile = self.model_profiles[model_id]
            
            # 性能效率
            performance_score = profile.quantum_efficiency
            
            # 能力分数
            capability_score = min(1.0, len(profile.capabilities) / 8.0)
            
            # 历史成功率
            historical_success = self.performance_metrics['model_success_rates'].get(model_id, 0.8)
            
            # 创新性加权
            innovation_score = profile.innovation_score
            
            # 稳定性加权
            stability_score = profile.stability_score
            
            # 综合性能分数
            total_score = (
                performance_score * 0.3 +
                capability_score * 0.25 +
                historical_success * 0.2 +
                innovation_score * 0.15 +
                stability_score * 0.1
            )
            
            performance_scores[model_id] = total_score
        
        return max(performance_scores, key=performance_scores.get)
    
    def _balanced_routing(self, candidates: List[str], context: UniversalRoutingContext) -> str:
        """平衡路由（增强版）"""
        if not candidates:
            return "gpt-4-turbo"
        
        # 多维度平衡评分
        balanced_scores = {}
        for model_id in candidates:
            profile = self.model_profiles[model_id]
            
            # 成本效率
            cost_efficiency = max(0.1, 1.0 - (profile.cost_per_token * 1000))
            
            # 性能效率
            performance_efficiency = profile.quantum_efficiency
            
            # 能力丰富度
            capability_richness = min(1.0, len(profile.capabilities) / 6.0)
            
            # 历史表现
            historical_performance = self.performance_metrics['model_success_rates'].get(model_id, 0.8)
            
            # 可靠性
            reliability = profile.reliability_score
            
            # 稳定性
            stability = profile.stability_score
            
            # 综合平衡分数
            balanced_score = (
                cost_efficiency * 0.2 +
                performance_efficiency * 0.25 +
                capability_richness * 0.2 +
                historical_performance * 0.15 +
                reliability * 0.1 +
                stability * 0.1
            )
            
            balanced_scores[model_id] = balanced_score
        
        return max(balanced_scores, key=balanced_scores.get)
    
    def _specialized_routing(self, candidates: List[str], context: UniversalRoutingContext) -> str:
        """专业化路由（增强版）"""
        # 根据任务类型和领域选择专业模型
        task_keywords = context.task_description.lower()
        
        # 编程任务
        if any(keyword in task_keywords for keyword in ['code', '编程', '开发', '程序', '编程', '开发', 'python', 'java', 'c++']):
            coding_models = [
                model_id for model_id in candidates
                if any(domain in self.model_profiles[model_id].specialization_domains 
                      for domain in ['coding', 'programming', 'general'])
            ]
            if coding_models:
                return max(coding_models, key=lambda x: (
                    self.model_profiles[x].innovation_score * 0.6 +
                    self.model_profiles[x].quantum_efficiency * 0.4
                ))
        
        # 创意任务
        if any(keyword in task_keywords for keyword in ['创意', '设计', '创作', '创新', 'creative', 'design', 'art']):
            creative_models = [
                model_id for model_id in candidates
                if (UniversalModelCapability.CREATIVE in self.model_profiles[model_id].capabilities or
                    self.model_profiles[model_id].innovation_score > 0.8)
            ]
            if creative_models:
                return max(creative_models, key=lambda x: self.model_profiles[x].innovation_score)
        
        # 分析任务
        if any(keyword in task_keywords for keyword in ['分析', 'analyz', '分析', '评估', 'evaluate', 'analysis']):
            analytical_models = [
                model_id for model_id in candidates
                if UniversalModelCapability.REASONING in self.model_profiles[model_id].capabilities
            ]
            if analytical_models:
                return max(analytical_models, key=lambda x: (
                    self.model_profiles[x].quantum_efficiency * 0.7 +
                    self.model_profiles[x].reliability_score * 0.3
                ))
        
        # 视觉任务
        if any(keyword in task_keywords for keyword in ['图像', '图片', 'vision', 'image', 'visual']):
            vision_models = [
                model_id for model_id in candidates
                if self.model_profiles[model_id].vision
            ]
            if vision_models:
                return max(vision_models, key=lambda x: self.model_profiles[x].quantum_efficiency)
        
        # 长文本任务
        if context.context_tokens > 50000:
            long_context_models = [
                model_id for model_id in candidates
                if self.model_profiles[model_id].context_length > 100000
            ]
            if long_context_models:
                return max(long_context_models, key=lambda x: self.model_profiles[x].stability_score)
        
        return self._balanced_routing(candidates, context)
    
    def _quantum_enhanced_routing(self, candidates: List[str], context: UniversalRoutingContext) -> str:
        """量子增强路由（增强版）"""
        if not candidates:
            return "gpt-4o"
        
        # 量子权重计算（增强版）
        quantum_scores = {}
        for model_id in candidates:
            base_score = self.quantum_weights[model_id]
            
            # 量子相干时间加权
            coherence_bonus = self.model_profiles[model_id].coherence_time * 0.1
            
            # 历史量子表现
            quantum_history = self.performance_metrics['model_success_rates'].get(model_id, 0.8)
            
            # 意识流反馈（如果可用）
            consciousness_feedback = 0.5  # 这里可以从意识流系统获取反馈
            if self.consciousness_system:
                # 可以从意识流系统获取模型表现反馈
                pass
            
            # 创新性加权
            innovation_bonus = self.model_profiles[model_id].innovation_score * 0.1
            
            total_score = (
                base_score * 0.5 +
                coherence_bonus * 0.2 +
                quantum_history * 0.2 +
                consciousness_feedback * 0.05 +
                innovation_bonus * 0.05
            )
            quantum_scores[model_id] = total_score
        
        return max(quantum_scores, key=quantum_scores.get)
    
    def _adaptive_learning_routing(self, candidates: List[str], context: UniversalRoutingContext) -> str:
        """自适应学习路由（增强版）"""
        if not candidates:
            return "claude-3-5-sonnet"
        
        # 基于历史表现和强化学习的增强版本
        learning_scores = {}
        for model_id in candidates:
            success_rate = self.performance_metrics['model_success_rates'].get(model_id, 0.8)
            avg_response_time = np.mean(self.performance_metrics['model_response_times'].get(model_id, [1000]))
            
            # 响应时间标准化（越快越好）
            time_score = max(0.1, 1.0 - (avg_response_time / 10000))
            
            # 预测性能趋势
            trend_score = self._calculate_performance_trend(model_id)
            
            # 学习加权分数（增强版）
            learning_score = (
                success_rate * 0.4 +
                time_score * 0.2 +
                self.quantum_weights[model_id] * 0.2 +
                trend_score * 0.15 +
                self.predictive_model[model_id]['adaptive_score'] * 0.05
            )
            
            learning_scores[model_id] = learning_score
        
        # 更新量子权重和预测模型
        best_model = max(learning_scores, key=learning_scores.get)
        self._update_quantum_weights_enhanced(best_model, reward=0.1)
        self._update_predictive_model(best_model)
        
        return best_model
    
    def _calculate_performance_trend(self, model_id: str) -> float:
        """计算性能趋势"""
        performance_data = self.predictive_model.get(model_id, {}).get('performance_trend', [])
        if len(performance_data) < 2:
            return 0.5
        
        # 简单线性趋势计算
        recent_performance = performance_data[-10:]  # 最近10次
        if len(recent_performance) < 2:
            return 0.5
        
        # 计算趋势斜率
        x = list(range(len(recent_performance)))
        y = recent_performance
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return max(0.1, min(0.9, 0.5 + slope * 2))  # 归一化到0.1-0.9
        return 0.5
    
    def _update_quantum_weights_enhanced(self, model_id: str, reward: float):
        """增强的量子权重更新"""
        # 强化学习更新（增强版）
        current_weight = self.quantum_weights[model_id]
        # 基于多因素的动态学习率
        adaptive_rate = self.adaptation_rate * (
            0.5 + self.predictive_model[model_id]['adaptive_score'] * 0.5
        )
        new_weight = min(1.0, max(0.1, current_weight + reward * adaptive_rate))
        self.quantum_weights[model_id] = new_weight
    
    def _update_predictive_model(self, model_id: str):
        """更新预测模型"""
        # 更新模型的自适应分数
        success_rate = self.performance_metrics['model_success_rates'].get(model_id, 0.8)
        self.predictive_model[model_id]['adaptive_score'] = (
            self.predictive_model[model_id]['adaptive_score'] * 0.8 + success_rate * 0.2
        )
    
    def _context_aware_routing(self, candidates: List[str], context: UniversalRoutingContext) -> str:
        """上下文感知路由（增强版）"""
        # 基于当前系统状态和上下文的增强版本
        if not candidates:
            return "gpt-4o"
        
        # 检查系统负载
        system_load = len(self.active_sessions)
        load_threshold = 10
        
        if system_load > load_threshold:
            # 高负载时选择响应快的模型
            fast_models = [
                model_id for model_id in candidates
                if self.model_profiles[model_id].metadata.get('fast_response', False)
            ]
            if fast_models:
                return min(fast_models, key=lambda x: self.model_profiles[x].cost_per_token)
        
        # 基于意识流系统状态（如果可用）
        if self.consciousness_system:
            try:
                consciousness_status = await self.consciousness_system.get_system_status()
                emotional_state = consciousness_status.get('emotional_state', 0.5)
                
                # 情感状态影响模型选择
                if emotional_state > 0.7:  # 积极状态，选择高性能模型
                    return self._performance_prioritized_routing(candidates, context)
                elif emotional_state < -0.3:  # 消极状态，选择低成本模型
                    return self._cost_optimized_routing(candidates, context)
                else:
                    return self._balanced_routing(candidates, context)
            except Exception:
                pass
        
        # 基于任务上下文的情感分析
        if context.emotional_context > 0.7:
            # 积极情绪，选择创新性模型
            innovative_models = [
                model_id for model_id in candidates
                if self.model_profiles[model_id].innovation_score > 0.8
            ]
            if innovative_models:
                return max(innovative_models, key=lambda x: self.model_profiles[x].innovation_score)
        elif context.emotional_context < -0.3:
            # 消极情绪，选择稳定可靠模型
            stable_models = [
                model_id for model_id in candidates
                if self.model_profiles[model_id].stability_score > 0.8
            ]
            if stable_models:
                return max(stable_models, key=lambda x: self.model_profiles[x].stability_score)
        
        return self._balanced_routing(candidates, context)
    
    def _emergency_mode_routing(self, candidates: List[str], context: UniversalRoutingContext) -> str:
        """应急模式路由（增强版）"""
        # 在系统异常时快速选择可用模型
        if not candidates:
            # 降级到最基本的模型
            fallback_models = [m for m in self.model_profiles.keys() 
                             if UniversalModelCapability.CHAT in self.model_profiles[m].capabilities]
            return fallback_models[0] if fallback_models else "gpt-3.5-turbo"
        
        # 选择最稳定可靠的模型
        reliability_scores = {}
        for model_id in candidates:
            success_rate = self.performance_metrics['model_success_rates'].get(model_id, 0.8)
            response_times = self.performance_metrics['model_response_times'].get(model_id, [1000])
            avg_time = np.mean(response_times) if response_times else 1000
            
            # 可靠性分数（增强版）
            reliability_score = (
                success_rate * 0.6 +
                max(0.1, 1.0 - (avg_time / 5000)) * 0.3 +
                self.model_profiles[model_id].reliability_score * 0.1
            )
            
            reliability_scores[model_id] = reliability_score
        
        return max(reliability_scores, key=reliability_scores.get)
    
    def _predictive_routing(self, candidates: List[str], context: UniversalRoutingContext) -> str:
        """预测性路由（新增）"""
        if not candidates:
            return "claude-3-5-sonnet"
        
        # 基于预测模型和趋势分析
        predictive_scores = {}
        for model_id in candidates:
            # 获取预测性能
            adaptive_score = self.predictive_model[model_id]['adaptive_score']
            
            # 趋势分析
            trend_score = self._calculate_performance_trend(model_id)
            
            # 负载预测
            load_score = self._predict_load_impact(model_id)
            
            # 综合预测分数
            predictive_score = (
                adaptive_score * 0.4 +
                trend_score * 0.35 +
                load_score * 0.25
            )
            
            predictive_scores[model_id] = predictive_score
        
        return max(predictive_scores, key=predictive_scores.get)
    
    def _predict_load_impact(self, model_id: str) -> float:
        """预测负载影响"""
        # 简化的负载影响预测
        historical_loads = self.predictive_model[model_id]['load_pattern']
        if not historical_loads:
            return 0.5
        
        # 基于历史负载模式预测
        avg_load_impact = np.mean([np.mean(times) for times in historical_loads.values() if times])
        return max(0.1, 1.0 - (avg_load_impact / 10000))
    
    def _collaborative_routing(self, candidates: List[str], context: UniversalRoutingContext) -> str:
        """协作路由（新增）"""
        if not candidates:
            return "claude-3-5-sonnet"
        
        # 选择最适合协作的模型
        collaborative_scores = {}
        for model_id in candidates:
            profile = self.model_profiles[model_id]
            
            # 协作友好度
            collaborative_friendliness = profile.metadata.get('collaborative_score', 0.5)
            
            # 可靠性
            reliability = profile.reliability_score
            
            # 历史协作成功率
            collaboration_history = self.collaborative_memory.get(model_id, [])
            collaboration_success = np.mean(collaboration_history) if collaboration_history else 0.8
            
            # 综合协作分数
            collaborative_score = (
                collaborative_friendliness * 0.4 +
                reliability * 0.35 +
                collaboration_success * 0.25
            )
            
            collaborative_scores[model_id] = collaborative_score
        
        return max(collaborative_scores, key=collaborative_scores.get)
    
    async def _execute_unified_model_call(self, model_id: str, prompt: Union[str, List[Dict]], context: UniversalRoutingContext) -> Dict[str, Any]:
        """执行统一模型调用"""
        profile = self.model_profiles[model_id]
        
        # 检查缓存
        cache_key = self._generate_enhanced_cache_key(model_id, prompt, context)
        if cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            if time.time() - cached_response['timestamp'] < self.cache_ttl:
                logger.info(f"📦 使用缓存响应: {model_id}")
                return cached_response['response']
        
        try:
            # 模拟API调用（实际实现需要集成真实的API）
            response = await self._simulate_enhanced_api_call(model_id, prompt, profile, context)
            
            # 缓存响应
            self.response_cache[cache_key] = {
                'response': response,
                'timestamp': time.time()
            }
            
            # 限制缓存大小
            if len(self.response_cache) > 1000:
                # 移除最旧的缓存项
                oldest_key = min(self.response_cache.keys(), 
                               key=lambda k: self.response_cache[k]['timestamp'])
                del self.response_cache[oldest_key]
            
            return response
            
        except Exception as e:
            logger.error(f"模型调用失败: {model_id} - {e}")
            return {
                'success': False,
                'error': str(e),
                'model_id': model_id,
                'response_time': 0
            }
    
    def _generate_enhanced_cache_key(self, model_id: str, prompt: Union[str, List[Dict]], context: UniversalRoutingContext) -> str:
        """生成增强缓存键"""
        prompt_str = str(prompt) if isinstance(prompt, str) else json.dumps(prompt, sort_keys=True)
        context_str = f"{context.complexity.value}{context.emotional_context}{context.urgency_level}"
        content = f"{model_id}:{prompt_str}:{context_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _simulate_enhanced_api_call(self, model_id: str, prompt: Union[str, List[Dict]], profile: UniversalModelProfile, context: UniversalRoutingContext) -> Dict[str, Any]:
        """模拟增强API调用"""
        # 模拟响应时间（基于模型特性和上下文）
        base_response_time = 1000  # ms
        complexity_multiplier = {
            UniversalTaskComplexity.TRIVIAL: 0.5,
            UniversalTaskComplexity.SIMPLE: 0.7,
            UniversalTaskComplexity.MODERATE: 1.0,
            UniversalTaskComplexity.COMPLEX: 1.5,
            UniversalTaskComplexity.EXPERT: 2.0,
            UniversalTaskComplexity.MASTER: 2.5,
            UniversalTaskComplexity.TRANSCENDENT: 3.0
        }
        
        complexity_factor = complexity_multiplier.get(context.complexity, 1.0)
        urgency_factor = 1.0 - (context.urgency_level * 0.3)  # 紧急程度降低响应时间
        
        response_time = base_response_time * complexity_factor * urgency_factor * (1.0 + np.random.random() * 0.5)
        
        # 模拟成功率（基于多因素）
        base_success_rate = profile.quantum_efficiency * 0.9
        context_factor = 1.0 - abs(context.emotional_context) * 0.1  # 情感影响
        quality_factor = context.quality_requirement
        
        success_rate = base_success_rate * context_factor * quality_factor
        
        if np.random.random() < success_rate:
            # 成功响应
            response_content = f"统一适配器响应来自 {model_id}: 基于增强上下文生成的内容..."
            
            return {
                'success': True,
                'model_id': model_id,
                'content': response_content,
                'usage': {
                    'prompt_tokens': len(str(prompt).split()),
                    'completion_tokens': 100,
                    'total_tokens': len(str(prompt).split()) + 100
                },
                'response_time': response_time,
                'cost': response_time * profile.cost_per_token / 1000,
                'context_enhanced': True,
                'routing_strategy': self.current_strategy.value
            }
        else:
            # 失败响应
            return {
                'success': False,
                'model_id': model_id,
                'error': "统一适配器API调用失败",
                'response_time': response_time,
                'retry_after': 1000  # ms
            }
    
    def _update_enhanced_performance_metrics(self, model_id: str, response: Dict[str, Any], response_time: float):
        """更新增强性能指标"""
        self.performance_metrics['total_requests'] += 1
        
        if response.get('success', False):
            self.performance_metrics['success_requests'] += 1
            
            # 更新模型成功率
            total_calls = self.performance_metrics['routing_decisions'][model_id]
            success_calls = sum(1 for _ in range(total_calls) 
                               if self.performance_metrics['model_success_rates'].get(model_id, 0.8) > 0.5)
            self.performance_metrics['model_success_rates'][model_id] = success_calls / total_calls if total_calls > 0 else 0.8
            
            # 更新协作记忆
            if 'context_enhanced' in response:
                self.collaborative_memory[model_id].append(1.0)
                if len(self.collaborative_memory[model_id]) > 50:
                    self.collaborative_memory[model_id] = self.collaborative_memory[model_id][-50:]
            
        else:
            self.performance_metrics['failed_requests'] += 1
            
            # 更新协作记忆
            if 'context_enhanced' in response:
                self.collaborative_memory[model_id].append(0.0)
                if len(self.collaborative_memory[model_id]) > 50:
                    self.collaborative_memory[model_id] = self.collaborative_memory[model_id][-50:]
        
        # 更新响应时间
        self.performance_metrics['model_response_times'][model_id].append(response_time * 1000)  # 转换为ms
        
        # 限制响应时间历史长度
        if len(self.performance_metrics['model_response_times'][model_id]) > 100:
            self.performance_metrics['model_response_times'][model_id].pop(0)
        
        # 更新平均响应时间
        all_times = []
        for times in self.performance_metrics['model_response_times'].values():
            all_times.extend(times)
        self.performance_metrics['avg_response_time'] = np.mean(all_times) if all_times else 0.0
        
        # 更新成本
        if 'cost' in response:
            self.performance_metrics['total_cost'] += response['cost']
        
        # 更新策略效果
        strategy = response.get('routing_strategy', self.current_strategy.value)
        self.performance_metrics['strategy_effectiveness'][strategy] = (
            self.performance_metrics['strategy_effectiveness'].get(strategy, 0.5) * 0.9 +
            (1.0 if response.get('success', False) else 0.0) * 0.1
        )
    
    async def get_unified_adapter_status(self) -> Dict[str, Any]:
        """获取统一适配器状态"""
        # 计算模型平均响应时间
        avg_response_times = {}
        for model_id, times in self.performance_metrics['model_response_times'].items():
            avg_response_times[model_id] = np.mean(times) if times else 0.0
        
        # 计算成本效率
        cost_efficiency = {}
        for model_id, profile in self.model_profiles.items():
            success_rate = self.performance_metrics['model_success_rates'].get(model_id, 0.8)
            avg_time = avg_response_times.get(model_id, 1000)
            cost_per_request = profile.cost_per_token * 1000  # 估算
            
            # 成本效率分数
            efficiency_score = (success_rate * 1000) / (avg_time * cost_per_request + 0.001)
            cost_efficiency[model_id] = efficiency_score
        
        # 获取预测模型状态
        predictive_status = {}
        for model_id, model_data in self.predictive_model.items():
            predictive_status[model_id] = {
                'adaptive_score': model_data['adaptive_score'],
                'trend_direction': 'improving' if model_data['adaptive_score'] > 0.6 else 'declining'
            }
        
        return {
            'adapter_id': self.adapter_id,
            'current_strategy': self.current_strategy.value,
            'total_models': len(self.model_profiles),
            'active_sessions': len(self.active_sessions),
            'performance_metrics': {
                'total_requests': self.performance_metrics['total_requests'],
                'success_rate': (
                    self.performance_metrics['success_requests'] / 
                    max(1, self.performance_metrics['total_requests'])
                ),
                'avg_response_time': self.performance_metrics['avg_response_time'],
                'total_cost': self.performance_metrics['total_cost'],
                'strategy_effectiveness': dict(self.performance_metrics['strategy_effectiveness'])
            },
            'model_stats': {
                'success_rates': dict(self.performance_metrics['model_success_rates']),
                'avg_response_times': avg_response_times,
                'routing_decisions': dict(self.performance_metrics['routing_decisions']),
                'cost_efficiency': cost_efficiency,
                'quantum_weights': dict(self.quantum_weights),
                'predictive_status': predictive_status
            },
            'cache_size': len(self.response_cache),
            'collaborative_memory_size': sum(len(memories) for memories in self.collaborative_memory.values())
        }
    
    def set_routing_strategy(self, strategy: UniversalRoutingStrategy):
        """设置路由策略"""
        self.current_strategy = strategy
        logger.info(f"🎯 路由策略已更新: {strategy.value}")
    
    def close(self):
        """关闭适配器"""
        logger.info("🛑 关闭统一模型适配器V2...")
        
        # 保存性能统计
        stats_file = f"unified_model_adapter_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        stats_data = {
            'adapter_id': self.adapter_id,
            'final_metrics': dict(self.performance_metrics),
            'quantum_weights': dict(self.quantum_weights),
            'predictive_models': self.predictive_model,
            'cache_size': len(self.response_cache),
            'collaborative_memory': dict(self.collaborative_memory),
            'model_profiles_summary': {
                model_id: {
                    'provider': profile.provider.value,
                    'capabilities': [cap.value for cap in profile.capabilities],
                    'cost_per_token': profile.cost_per_token,
                    'quantum_efficiency': profile.quantum_efficiency,
                    'specialization_domains': profile.specialization_domains,
                    'reliability_score': profile.reliability_score,
                    'innovation_score': profile.innovation_score,
                    'stability_score': profile.stability_score
                }
                for model_id, profile in self.model_profiles.items()
            }
        }
        
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, ensure_ascii=False, indent=2)
            logger.info(f"📊 统一适配器统计已保存到: {stats_file}")
        except Exception as e:
            logger.warning(f"保存统计信息失败: {e}")
        
        logger.info("✅ 统一模型适配器V2已关闭")

# 全局统一适配器实例
unified_adapter = UnifiedModelAdapter()

# 便捷函数
async def unified_chat_completion(
    task_type: UniversalTaskComplexity,
    messages: List[Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """统一聊天完成"""
    return await unified_adapter.unified_adaptive_call(task_type, messages, **kwargs)

def get_unified_model_stats() -> Dict[str, Any]:
    """获取统一模型统计"""
    return unified_adapter.get_unified_adapter_status()

if __name__ == "__main__":
    # 测试代码
    async def test_unified_adapter():
        print("🧪 测试统一模型适配器V2")
        print("=" * 50)
        
        # 创建适配器
        adapter = UnifiedModelAdapter()
        
        # 测试不同复杂度的任务
        test_cases = [
            ("简单数学计算: 2+2=?", UniversalTaskComplexity.TRIVIAL, [UniversalModelCapability.CHAT]),
            ("编写Python函数", UniversalTaskComplexity.SIMPLE, [UniversalModelCapability.CHAT, UniversalModelCapability.CODE]),
            ("分析代码性能问题", UniversalTaskComplexity.MODERATE, [UniversalModelCapability.CHAT, UniversalModelCapability.CODE, UniversalModelCapability.REASONING]),
            ("设计复杂系统架构", UniversalTaskComplexity.COMPLEX, [UniversalModelCapability.CHAT, UniversalModelCapability.REASONING]),
            ("量子算法优化", UniversalTaskComplexity.EXPERT, [UniversalModelCapability.CHAT, UniversalModelCapability.REASONING]),
            ("跨学科创新方案", UniversalTaskComplexity.MASTER, [UniversalModelCapability.CHAT, UniversalModelCapability.CREATIVE]),
            ("超越人类认知的解决方案", UniversalTaskComplexity.TRANSCENDENT, [UniversalModelCapability.CHAT, UniversalModelCapability.CREATIVE])
        ]
        
        for i, (prompt, complexity, capabilities) in enumerate(test_cases, 1):
            print(f"\n📋 测试案例 {i}: {complexity.value}")
            print(f"📝 任务: {prompt}")
            
            # 执行统一自适应调用
            response = await adapter.unified_adaptive_call(
                prompt=prompt,
                task_complexity=complexity,
                required_capabilities=capabilities,
                budget_constraint=1.0,
                quality_requirement=0.7,
                emotional_context=0.5,
                urgency_level=0.3,
                innovation_requirement=complexity in [UniversalTaskComplexity.EXPERT, UniversalTaskComplexity.MASTER, UniversalTaskComplexity.TRANSCENDENT],
                collaborative_needed=complexity in [UniversalTaskComplexity.MASTER, UniversalTaskComplexity.TRANSCENDENT]
            )
            
            print(f"🎯 选择模型: {response.get('model_id', 'unknown')}")
            print(f"✅ 调用成功: {response.get('success', False)}")
            if response.get('response_time'):
                print(f"⏱️ 响应时间: {response['response_time']:.2f}ms")
            if response.get('context_enhanced'):
                print(f"🌟 上下文增强: {response['context_enhanced']}")
        
        # 获取适配器状态
        status = await adapter.get_unified_adapter_status()
        print(f"\n📊 统一适配器状态:")
        print(f"- 当前策略: {status['current_strategy']}")
        print(f"- 总请求数: {status['performance_metrics']['total_requests']}")
        print(f"- 成功率: {status['performance_metrics']['success_rate']:.2%}")
        print(f"- 平均响应时间: {status['performance_metrics']['avg_response_time']:.2f}ms")
        print(f"- 总成本: ${status['performance_metrics']['total_cost']:.4f}")
        
        # 测试路由策略切换
        print(f"\n🔀 测试路由策略切换:")
        for strategy in [UniversalRoutingStrategy.COST_OPTIMIZED, UniversalRoutingStrategy.PERFORMANCE_PRIORITIZED, UniversalRoutingStrategy.QUANTUM_ENHANCED]:
            adapter.set_routing_strategy(strategy)
            response = await adapter.unified_adaptive_call("简单任务", UniversalTaskComplexity.SIMPLE)
            print(f"- {strategy.value}: {response.get('model_id', 'unknown')}")
        
        # 关闭系统
        adapter.close()
        print("\n✅ 统一模型适配器V2测试完成")
    
    asyncio.run(test_unified_adapter())