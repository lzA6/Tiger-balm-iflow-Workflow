#!/usr/bin/env python3
"""
人因工程与信任设计引擎 V1
为A项目iflow工作流系统提供全领域用户体验和信任保障

核心理念：以用户为中心，构建可信、可用、包容的AI工作流系统

六大核心维度：
1. 认知负荷与无障碍设计
2. 信任与透明度仪表板  
3. 行为经济学干预
4. 多模态交互体验
5. 情感计算与共情设计
6. 社会责任与数字公平

你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）
"""

import json
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import threading
import statistics
from collections import defaultdict, deque

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserProfileType(Enum):
    """用户类型枚举"""
    NOVICE = "novice"           # 新手用户
    INTERMEDIATE = "intermediate"  # 中级用户
    EXPERT = "expert"          # 专家用户
    ELDERLY = "elderly"        # 老年用户
    DISABLED = "disabled"      # 残障用户
    EXECUTIVE = "executive"    # 管理层用户

class CognitiveLoadLevel(Enum):
    """认知负荷等级"""
    LOW = "low"           # 低负荷
    MODERATE = "moderate"  # 中等负荷
    HIGH = "high"         # 高负荷
    OVERWHELMED = "overwhelmed"  # 超负荷

class TrustLevel(Enum):
    """信任等级"""
    NONE = "none"         # 无信任
    LOW = "low"          # 低信任
    MEDIUM = "medium"    # 中等信任
    HIGH = "high"        # 高信任
    COMPLETE = "complete"  # 完全信任

class AccessibilityLevel(Enum):
    """无障碍等级"""
    NONE = "none"        # 无障碍支持
    BASIC = "basic"      # 基础支持
    ADVANCED = "advanced"  # 高级支持
    UNIVERSAL = "universal"  # 通用支持

@dataclass
class UserContext:
    """用户上下文信息"""
    user_id: str
    profile_type: UserProfileType
    cognitive_load: CognitiveLoadLevel
    trust_level: TrustLevel
    accessibility_needs: List[str]
    preferences: Dict[str, Any]
    session_history: List[Dict[str, Any]] = field(default_factory=list)
    emotional_state: str = "neutral"
    task_complexity: str = "medium"
    time_pressure: str = "low"
    cultural_background: str = "default"

@dataclass
class InteractionMetrics:
    """交互指标"""
    timestamp: datetime
    user_id: str
    interaction_type: str
    duration: float  # 交互持续时间
    success_rate: float  # 成功率
    error_count: int
    assistance_requests: int
    satisfaction_score: float
    cognitive_effort: float
    trust_change: float

@dataclass
class AccessibilityFeature:
    """无障碍功能"""
    feature_id: str
    feature_name: str
    description: str
    target_disability: List[str]
    implementation_level: AccessibilityLevel
    effectiveness_score: float
    user_satisfaction: float

class CognitiveLoadAnalyzer:
    """认知负荷分析器"""
    
    def __init__(self):
        """初始化认知负荷分析器"""
        # 认知负荷影响因素权重
        self.load_factors = {
            "interface_complexity": 0.25,
            "information_density": 0.20,
            "task_complexity": 0.20,
            "time_pressure": 0.15,
            "error_frequency": 0.10,
            "learning_curve": 0.10
        }
        
        # 认知负荷计算阈值
        self.load_thresholds = {
            "low": 0.3,
            "moderate": 0.6,
            "high": 0.8,
            "overwhelmed": 1.0
        }
    
    def analyze_cognitive_load(self, user_context: UserContext, 
                              interaction_data: Dict[str, Any]) -> CognitiveLoadLevel:
        """
        分析用户认知负荷
        
        Args:
            user_context: 用户上下文
            interaction_data: 交互数据
            
        Returns:
            认知负荷等级
        """
        try:
            # 计算各因素得分
            scores = {}
            
            # 界面复杂度
            interface_complexity = self._calculate_interface_complexity(
                interaction_data.get("interface_elements", 0),
                interaction_data.get("navigation_depth", 0)
            )
            scores["interface_complexity"] = interface_complexity
            
            # 信息密度
            info_density = self._calculate_information_density(
                interaction_data.get("information_chunks", 0),
                interaction_data.get("screen_clutter", 0)
            )
            scores["information_density"] = info_density
            
            # 任务复杂度
            task_complexity = self._calculate_task_complexity(
                user_context.task_complexity,
                interaction_data.get("decision_points", 0)
            )
            scores["task_complexity"] = task_complexity
            
            # 时间压力
            time_pressure = self._calculate_time_pressure(
                user_context.time_pressure,
                interaction_data.get("response_time", 0)
            )
            scores["time_pressure"] = time_pressure
            
            # 错误频率
            error_frequency = self._calculate_error_frequency(
                interaction_data.get("error_count", 0),
                interaction_data.get("total_interactions", 1)
            )
            scores["error_frequency"] = error_frequency
            
            # 学习曲线
            learning_curve = self._calculate_learning_curve(
                user_context.profile_type,
                interaction_data.get("session_count", 0)
            )
            scores["learning_curve"] = learning_curve
            
            # 加权计算总负荷
            total_load = sum(
                scores[factor] * weight 
                for factor, weight in self.load_factors.items()
            )
            
            # 确定负荷等级
            if total_load <= self.load_thresholds["low"]:
                load_level = CognitiveLoadLevel.LOW
            elif total_load <= self.load_thresholds["moderate"]:
                load_level = CognitiveLoadLevel.MODERATE
            elif total_load <= self.load_thresholds["high"]:
                load_level = CognitiveLoadLevel.HIGH
            else:
                load_level = CognitiveLoadLevel.OVERWHELMED
            
            return load_level
            
        except Exception as e:
            logger.error(f"认知负荷分析失败: {e}")
            return CognitiveLoadLevel.MODERATE
    
    def _calculate_interface_complexity(self, elements_count: int, navigation_depth: int) -> float:
        """计算界面复杂度"""
        # 简化计算：基于元素数量和导航深度
        complexity_score = min((elements_count * 0.1 + navigation_depth * 0.2) / 10.0, 1.0)
        return complexity_score
    
    def _calculate_information_density(self, chunks: int, clutter: int) -> float:
        """计算信息密度"""
        # 基于认知心理学的7±2法则
        optimal_chunks = 7
        density_score = max(0, min(1, (chunks - optimal_chunks) / 10.0))
        clutter_penalty = min(clutter * 0.1, 0.5)
        return density_score + clutter_penalty
    
    def _calculate_task_complexity(self, task_complexity: str, decision_points: int) -> float:
        """计算任务复杂度"""
        complexity_mapping = {"low": 0.2, "medium": 0.5, "high": 0.8}
        base_score = complexity_mapping.get(task_complexity, 0.5)
        decision_penalty = min(decision_points * 0.1, 0.3)
        return base_score + decision_penalty
    
    def _calculate_time_pressure(self, time_pressure: str, response_time: float) -> float:
        """计算时间压力"""
        pressure_mapping = {"low": 0.1, "medium": 0.3, "high": 0.6}
        base_score = pressure_mapping.get(time_pressure, 0.3)
        
        # 响应时间过长增加压力
        if response_time > 3.0:  # 3秒阈值
            time_penalty = min((response_time - 3.0) * 0.1, 0.4)
            return base_score + time_penalty
        return base_score
    
    def _calculate_error_frequency(self, error_count: int, total_interactions: int) -> float:
        """计算错误频率"""
        if total_interactions == 0:
            return 0.0
        error_rate = error_count / total_interactions
        return min(error_rate * 2.0, 1.0)  # 放大错误影响
    
    def _calculate_learning_curve(self, profile_type: UserProfileType, session_count: int) -> float:
        """计算学习曲线影响"""
        # 新手用户前期学习曲线较陡
        if profile_type == UserProfileType.NOVICE and session_count < 5:
            return 0.7
        elif profile_type == UserProfileType.ELDERLY:
            return 0.4
        elif profile_type == UserProfileType.DISABLED:
            return 0.3
        else:
            return 0.1

class TrustBuilder:
    """信任构建器"""
    
    def __init__(self):
        """初始化信任构建器"""
        # 信任构建因素权重
        self.trust_factors = {
            "transparency": 0.25,      # 透明度
            "competence": 0.25,        # 能力
            "reliability": 0.20,       # 可靠性
            "security": 0.15,          # 安全性
            "empathy": 0.10,           # 共情
            "fairness": 0.05           # 公平性
        }
        
        # 信任等级阈值
        self.trust_thresholds = {
            "none": 0.2,
            "low": 0.4,
            "medium": 0.6,
            "high": 0.8,
            "complete": 1.0
        }
    
    def build_trust(self, user_context: UserContext, system_actions: List[Dict[str, Any]]) -> TrustLevel:
        """
        构建用户信任
        
        Args:
            user_context: 用户上下文
            system_actions: 系统行为记录
            
        Returns:
            信任等级
        """
        try:
            trust_scores = {}
            
            # 透明度评分
            transparency_score = self._assess_transparency(system_actions)
            trust_scores["transparency"] = transparency_score
            
            # 能力评分
            competence_score = self._assess_competence(system_actions)
            trust_scores["competence"] = competence_score
            
            # 可靠性评分
            reliability_score = self._assess_reliability(system_actions)
            trust_scores["reliability"] = reliability_score
            
            # 安全性评分
            security_score = self._assess_security(system_actions)
            trust_scores["security"] = security_score
            
            # 共情评分
            empathy_score = self._assess_empathy(user_context, system_actions)
            trust_scores["empathy"] = empathy_score
            
            # 公平性评分
            fairness_score = self._assess_fairness(system_actions)
            trust_scores["fairness"] = fairness_score
            
            # 加权计算总信任度
            total_trust = sum(
                trust_scores[factor] * weight 
                for factor, weight in self.trust_factors.items()
            )
            
            # 确定信任等级
            if total_trust <= self.trust_thresholds["none"]:
                trust_level = TrustLevel.NONE
            elif total_trust <= self.trust_thresholds["low"]:
                trust_level = TrustLevel.LOW
            elif total_trust <= self.trust_thresholds["medium"]:
                trust_level = TrustLevel.MEDIUM
            elif total_trust <= self.trust_thresholds["high"]:
                trust_level = TrustLevel.HIGH
            else:
                trust_level = TrustLevel.COMPLETE
            
            return trust_level
            
        except Exception as e:
            logger.error(f"信任构建失败: {e}")
            return TrustLevel.LOW
    
    def _assess_transparency(self, system_actions: List[Dict[str, Any]]) -> float:
        """评估透明度"""
        if not system_actions:
            return 0.1
        
        # 检查是否有解释性行为
        explanation_count = sum(1 for action in system_actions 
                              if action.get("type") == "explanation")
        transparency_ratio = explanation_count / len(system_actions)
        
        # 检查决策过程的可见性
        process_visible = any(action.get("shows_process", False) 
                            for action in system_actions)
        
        return min(transparency_ratio + (0.3 if process_visible else 0.0), 1.0)
    
    def _assess_competence(self, system_actions: List[Dict[str, Any]]) -> float:
        """评估能力"""
        if not system_actions:
            return 0.1
        
        # 成功率
        success_actions = [a for a in system_actions if a.get("success", False)]
        success_rate = len(success_actions) / len(system_actions)
        
        # 问题解决能力
        problem_solved = sum(1 for a in system_actions 
                           if a.get("problem_solved", False))
        problem_solving_rate = problem_solved / max(len(system_actions), 1)
        
        return (success_rate * 0.7 + problem_solving_rate * 0.3)
    
    def _assess_reliability(self, system_actions: List[Dict[str, Any]]) -> float:
        """评估可靠性"""
        if not system_actions:
            return 0.1
        
        # 一致性检查
        consistent_actions = sum(1 for a in system_actions 
                                if a.get("consistent", True))
        consistency_rate = consistent_actions / len(system_actions)
        
        # 可用性检查
        available_actions = sum(1 for a in system_actions 
                               if a.get("available", True))
        availability_rate = available_actions / len(system_actions)
        
        return (consistency_rate * 0.6 + availability_rate * 0.4)
    
    def _assess_security(self, system_actions: List[Dict[str, Any]]) -> float:
        """评估安全性"""
        security_behaviors = [
            "data_encrypted",
            "authentication_required", 
            "access_logged",
            "privacy_protected"
        ]
        
        security_score = 0.0
        for behavior in security_behaviors:
            behavior_count = sum(1 for a in system_actions 
                                if a.get(behavior, False))
            security_score += behavior_count / len(system_actions)
        
        return min(security_score / len(security_behaviors), 1.0)
    
    def _assess_empathy(self, user_context: UserContext, system_actions: List[Dict[str, Any]]) -> float:
        """评估共情能力"""
        empathy_behaviors = [
            "acknowledges_emotion",
            "adapts_to_user_state",
            "provides_encouragement",
            "shows_understanding"
        ]
        
        empathy_score = 0.0
        for behavior in empathy_behaviors:
            behavior_count = sum(1 for a in system_actions 
                                if a.get(behavior, False))
            empathy_score += behavior_count / len(system_actions)
        
        # 用户类型调整
        if user_context.profile_type in [UserProfileType.ELDERLY, UserProfileType.DISABLED]:
            empathy_weight = 0.4
        else:
            empathy_weight = 0.1
        
        return min((empathy_score + empathy_weight) / 2.0, 1.0)
    
    def _assess_fairness(self, system_actions: List[Dict[str, Any]]) -> float:
        """评估公平性"""
        fairness_indicators = [
            "no_bias_detected",
            "equal_treatment",
            "accessible_to_all",
            "transparent_criteria"
        ]
        
        fairness_score = 0.0
        for indicator in fairness_indicators:
            indicator_count = sum(1 for a in system_actions 
                                 if a.get(indicator, False))
            fairness_score += indicator_count / len(system_actions)
        
        return min(fairness_score / len(fairness_indicators), 1.0)

class AccessibilityManager:
    """无障碍管理器"""
    
    def __init__(self):
        """初始化无障碍管理器"""
        self.supported_features = self._initialize_accessibility_features()
        self.user_adaptations = {}  # user_id -> adaptation_settings
        
        # 无障碍需求映射
        self.disability_mappings = {
            "visual": ["screen_reader", "high_contrast", "text_to_speech"],
            "hearing": ["subtitles", "visual_alerts", "sign_language"],
            "motor": ["voice_control", "keyboard_navigation", "switch_access"],
            "cognitive": ["simplified_interface", "step_by_step", "error_prevention"],
            "elderly": ["large_text", "simplified_navigation", "voice_assistance"]
        }
    
    def _initialize_accessibility_features(self) -> Dict[str, AccessibilityFeature]:
        """初始化无障碍功能"""
        return {
            "screen_reader": AccessibilityFeature(
                feature_id="screen_reader",
                feature_name="屏幕阅读器支持",
                description="为视障用户提供完整的屏幕阅读器支持",
                target_disability=["visual"],
                implementation_level=AccessibilityLevel.UNIVERSAL,
                effectiveness_score=0.95,
                user_satisfaction=0.92
            ),
            "high_contrast": AccessibilityFeature(
                feature_id="high_contrast",
                feature_name="高对比度模式",
                description="提供高对比度色彩方案，改善视觉可读性",
                target_disability=["visual"],
                implementation_level=AccessibilityLevel.ADVANCED,
                effectiveness_score=0.88,
                user_satisfaction=0.85
            ),
            "text_to_speech": AccessibilityFeature(
                feature_id="text_to_speech",
                feature_name="文本转语音",
                description="将界面文本转换为语音输出",
                target_disability=["visual", "learning"],
                implementation_level=AccessibilityLevel.ADVANCED,
                effectiveness_score=0.90,
                user_satisfaction=0.88
            ),
            "voice_control": AccessibilityFeature(
                feature_id="voice_control",
                feature_name="语音控制",
                description="支持语音命令操作界面",
                target_disability=["motor", "elderly"],
                implementation_level=AccessibilityLevel.ADVANCED,
                effectiveness_score=0.85,
                user_satisfaction=0.82
            ),
            "keyboard_navigation": AccessibilityFeature(
                feature_id="keyboard_navigation",
                feature_name="键盘导航",
                description="完整的键盘操作支持，无需鼠标",
                target_disability=["motor"],
                implementation_level=AccessibilityLevel.UNIVERSAL,
                effectiveness_score=0.93,
                user_satisfaction=0.90
            ),
            "simplified_interface": AccessibilityFeature(
                feature_id="simplified_interface",
                feature_name="简化界面",
                description="减少界面复杂度，突出核心功能",
                target_disability=["cognitive", "elderly", "novice"],
                implementation_level=AccessibilityLevel.ADVANCED,
                effectiveness_score=0.87,
                user_satisfaction=0.84
            ),
            "step_by_step": AccessibilityFeature(
                feature_id="step_by_step",
                feature_name="分步指导",
                description="提供详细的分步操作指导",
                target_disability=["cognitive", "novice"],
                implementation_level=AccessibilityLevel.ADVANCED,
                effectiveness_score=0.91,
                user_satisfaction=0.89
            ),
            "error_prevention": AccessibilityFeature(
                feature_id="error_prevention",
                feature_name="错误预防",
                description="主动预防用户错误，提供确认机制",
                target_disability=["cognitive", "elderly"],
                implementation_level=AccessibilityLevel.ADVANCED,
                effectiveness_score=0.89,
                user_satisfaction=0.86
            )
        }
    
    def assess_user_accessibility_needs(self, user_context: UserContext) -> Dict[str, Any]:
        """
        评估用户无障碍需求
        
        Args:
            user_context: 用户上下文
            
        Returns:
            无障碍需求评估结果
        """
        try:
            user_needs = []
            recommended_features = []
            
            # 基于用户类型推断需求
            if user_context.profile_type == UserProfileType.ELDERLY:
                user_needs.extend(["age_related_vision", "motor_coordination", "cognitive_processing"])
                recommended_features.extend(["large_text", "simplified_interface", "voice_control"])
            
            elif user_context.profile_type == UserProfileType.DISABLED:
                # 需要更详细的残疾类型信息
                user_needs.extend(user_context.accessibility_needs)
                for need in user_context.accessibility_needs:
                    if need in self.disability_mappings:
                        recommended_features.extend(self.disability_mappings[need])
            
            # 基于认知负荷调整
            if user_context.cognitive_load in [CognitiveLoadLevel.HIGH, CognitiveLoadLevel.OVERWHELMED]:
                recommended_features.extend(["simplified_interface", "step_by_step", "error_prevention"])
            
            # 去重并排序
            recommended_features = list(set(recommended_features))
            
            # 计算无障碍等级
            if len(recommended_features) >= 6:
                accessibility_level = AccessibilityLevel.UNIVERSAL
            elif len(recommended_features) >= 4:
                accessibility_level = AccessibilityLevel.ADVANCED
            elif len(recommended_features) >= 2:
                accessibility_level = AccessibilityLevel.BASIC
            else:
                accessibility_level = AccessibilityLevel.NONE
            
            assessment = {
                "user_id": user_context.user_id,
                "identified_needs": user_needs,
                "recommended_features": recommended_features,
                "accessibility_level": accessibility_level.value,
                "implementation_priority": self._calculate_priority(user_context, recommended_features)
            }
            
            return assessment
            
        except Exception as e:
            logger.error(f"无障碍需求评估失败: {e}")
            return {"error": str(e)}
    
    def _calculate_priority(self, user_context: UserContext, features: List[str]) -> str:
        """计算实施优先级"""
        # 基于用户类型和当前状态计算优先级
        base_priority = 1.0
        
        if user_context.profile_type == UserProfileType.ELDERLY:
            base_priority += 0.3
        elif user_context.profile_type == UserProfileType.DISABLED:
            base_priority += 0.4
        
        if user_context.cognitive_load == CognitiveLoadLevel.OVERWHELMED:
            base_priority += 0.3
        elif user_context.cognitive_load == CognitiveLoadLevel.HIGH:
            base_priority += 0.2
        
        if len(features) >= 5:
            base_priority += 0.2
        
        if base_priority >= 1.8:
            return "high"
        elif base_priority >= 1.3:
            return "medium"
        else:
            return "low"

class BehaviorEconomicEngine:
    """行为经济学引擎"""
    
    def __init__(self):
        """初始化行为经济学引擎"""
        # 行为经济学原理映射
        self.behavioral_principles = {
            "loss_aversion": {
                "description": "损失厌恶 - 人们对损失的敏感度高于收益",
                "implementation": self._apply_loss_aversion,
                "weight": 0.25
            },
            "social_proof": {
                "description": "社会证明 - 人们倾向于跟随他人的行为",
                "implementation": self._apply_social_proof,
                "weight": 0.20
            },
            "anchoring": {
                "description": "锚定效应 - 初始信息对决策的强烈影响",
                "implementation": self._apply_anchoring,
                "weight": 0.15
            },
            "scarcity": {
                "description": "稀缺性 - 稀缺的物品更有价值",
                "implementation": self._apply_scarcity,
                "weight": 0.15
            },
            "commitment_consistency": {
                "description": "承诺一致性 - 人们希望保持行为一致",
                "implementation": self._apply_commitment_consistency,
                "weight": 0.15
            },
            "reciprocity": {
                "description": "互惠原则 - 人们倾向于回报他人的善意",
                "implementation": self._apply_reciprocity,
                "weight": 0.10
            }
        }
    
    def apply_behavioral_interventions(self, user_context: UserContext, 
                                     intervention_target: str) -> List[Dict[str, Any]]:
        """
        应用行为经济学干预
        
        Args:
            user_context: 用户上下文
            intervention_target: 干预目标
            
        Returns:
            行为干预策略列表
        """
        try:
            interventions = []
            
            # 根据用户类型和目标选择合适的原理
            applicable_principles = self._select_applicable_principles(
                user_context, intervention_target
            )
            
            for principle_name in applicable_principles:
                principle = self.behavioral_principles[principle_name]
                
                # 生成具体干预策略
                intervention = principle["implementation"](
                    user_context, intervention_target
                )
                
                if intervention:
                    interventions.append({
                        "principle": principle_name,
                        "description": principle["description"],
                        "intervention": intervention,
                        "expected_impact": principle["weight"],
                        "implementation_complexity": self._assess_complexity(intervention)
                    })
            
            # 按预期影响排序
            interventions.sort(key=lambda x: x["expected_impact"], reverse=True)
            
            return interventions
            
        except Exception as e:
            logger.error(f"行为经济学干预应用失败: {e}")
            return []
    
    def _select_applicable_principles(self, user_context: UserContext, 
                                    target: str) -> List[str]:
        """选择适用的行为经济学原理"""
        applicable = []
        
        # 基于用户类型选择
        if user_context.profile_type in [UserProfileType.NOVICE, UserProfileType.ELDERLY]:
            applicable.extend(["social_proof", "commitment_consistency", "reciprocity"])
        
        # 基于任务目标选择
        if "engagement" in target.lower():
            applicable.extend(["loss_aversion", "scarcity", "social_proof"])
        elif "completion" in target.lower():
            applicable.extend(["commitment_consistency", "anchoring"])
        elif "adoption" in target.lower():
            applicable.extend(["reciprocity", "social_proof"])
        
        # 基于认知负荷选择
        if user_context.cognitive_load in [CognitiveLoadLevel.HIGH, CognitiveLoadLevel.OVERWHELMED]:
            applicable.extend(["anchoring", "commitment_consistency"])
        
        # 去重并保持权重顺序
        principle_order = list(self.behavioral_principles.keys())
        return sorted(list(set(applicable)), key=lambda x: principle_order.index(x))
    
    def _apply_loss_aversion(self, user_context: UserContext, target: str) -> Dict[str, Any]:
        """应用损失厌恶原理"""
        return {
            "message": f"已有{user_context.preferences.get('progress', 75)}%的任务完成，放弃将失去所有进度",
            "framing": "强调不行动的损失而非行动的收益",
            "timing": "在用户犹豫时立即显示",
            "personalization": f"基于用户的{user_context.emotional_state}情绪状态调整语气"
        }
    
    def _apply_social_proof(self, user_context: UserContext, target: str) -> Dict[str, Any]:
        """应用社会证明原理"""
        return {
            "message": "95%的类似用户都选择了这个选项，他们取得了显著的进步",
            "evidence": "显示同行用户的成功案例和统计数据",
            "credibility": "提供真实的用户评价和成果展示",
            "relevance": f"匹配{user_context.profile_type.value}用户的典型行为模式"
        }
    
    def _apply_anchoring(self, user_context: UserContext, target: str) -> Dict[str, Any]:
        """应用锚定效应原理"""
        return {
            "initial_anchor": "设置一个合理的高标准起点",
            "adjustment_path": "提供渐进式的调整选项",
            "reference_points": f"基于{user_context.profile_type.value}用户的平均表现",
            "commitment_trap": "让用户逐步增加投入，避免大幅调整"
        }
    
    def _apply_scarcity(self, user_context: UserContext, target: str) -> Dict[str, Any]:
        """应用稀缺性原理"""
        return {
            "limited_time": "强调时间限制和机会窗口",
            "exclusive_access": "突出独特价值和稀有性",
            "urgency_cues": "使用倒计时和进度指示器",
            "fomo_trigger": f"针对{user_context.emotional_state}状态的FOMO策略"
        }
    
    def _apply_commitment_consistency(self, user_context: UserContext, target: str) -> Dict[str, Any]:
        """应用承诺一致性原理"""
        return {
            "public_commitment": "让用户公开承诺目标",
            "progress_tracking": "详细记录和展示进展",
            "consistency_reminders": "定期提醒用户的一致性",
            "identity_alignment": f"将行为与{user_context.profile_type.value}身份认同绑定"
        }
    
    def _apply_reciprocity(self, user_context: UserContext, target: str) -> Dict[str, Any]:
        """应用互惠原则"""
        return {
            "initial_gift": "提供有价值的信息或工具",
            "personalized_help": f"基于{user_context.preferences}的定制化支持",
            "expectation_setting": "暗示未来回报的可能性",
            "relationship_building": "建立长期的互惠关系"
        }
    
    def _assess_complexity(self, intervention: Dict[str, Any]) -> str:
        """评估实施复杂度"""
        complexity_factors = len(intervention.keys())
        if complexity_factors <= 2:
            return "low"
        elif complexity_factors <= 4:
            return "medium"
        else:
            return "high"

class HumanFactorsTrustEngine:
    """人因工程与信任设计引擎主类"""
    
    def __init__(self):
        """初始化人因工程与信任设计引擎"""
        self.cognitive_analyzer = CognitiveLoadAnalyzer()
        self.trust_builder = TrustBuilder()
        self.accessibility_manager = AccessibilityManager()
        self.behavior_engine = BehaviorEconomicEngine()
        
        # 用户数据库
        self.user_profiles = {}  # user_id -> UserContext
        self.interaction_history = defaultdict(list)  # user_id -> interactions
        
        # 配置参数
        self.adaptation_thresholds = {
            "cognitive_load_trigger": CognitiveLoadLevel.HIGH,
            "trust_decrease_threshold": 0.2,
            "accessibility_trigger": ["elderly", "disabled"]
        }
        
        logger.info("人因工程与信任设计引擎初始化完成")
    
    def analyze_user_context(self, user_id: str, context_data: Dict[str, Any]) -> UserContext:
        """
        分析用户上下文
        
        Args:
            user_id: 用户ID
            context_data: 上下文数据
            
        Returns:
            用户上下文对象
        """
        try:
            # 用户类型推断
            profile_type = self._infer_user_profile(context_data)
            
            # 认知负荷分析
            cognitive_load = self.cognitive_analyzer.analyze_cognitive_load(
                UserContext(
                    user_id=user_id,
                    profile_type=profile_type,
                    cognitive_load=CognitiveLoadLevel.MODERATE,
                    trust_level=TrustLevel.MEDIUM,
                    accessibility_needs=context_data.get("accessibility_needs", []),
                    preferences=context_data.get("preferences", {}),
                    emotional_state=context_data.get("emotional_state", "neutral"),
                    task_complexity=context_data.get("task_complexity", "medium"),
                    time_pressure=context_data.get("time_pressure", "low"),
                    cultural_background=context_data.get("cultural_background", "default")
                ),
                context_data.get("interaction_data", {})
            )
            
            # 信任等级评估
            trust_level = self.trust_builder.build_trust(
                UserContext(
                    user_id=user_id,
                    profile_type=profile_type,
                    cognitive_load=cognitive_load,
                    trust_level=TrustLevel.MEDIUM,
                    accessibility_needs=context_data.get("accessibility_needs", []),
                    preferences=context_data.get("preferences", {}),
                    emotional_state=context_data.get("emotional_state", "neutral"),
                    task_complexity=context_data.get("task_complexity", "medium"),
                    time_pressure=context_data.get("time_pressure", "low"),
                    cultural_background=context_data.get("cultural_background", "default")
                ),
                context_data.get("system_actions", [])
            )
            
            # 构建用户上下文
            user_context = UserContext(
                user_id=user_id,
                profile_type=profile_type,
                cognitive_load=cognitive_load,
                trust_level=trust_level,
                accessibility_needs=context_data.get("accessibility_needs", []),
                preferences=context_data.get("preferences", {}),
                session_history=context_data.get("session_history", []),
                emotional_state=context_data.get("emotional_state", "neutral"),
                task_complexity=context_data.get("task_complexity", "medium"),
                time_pressure=context_data.get("time_pressure", "low"),
                cultural_background=context_data.get("cultural_background", "default")
            )
            
            # 保存到数据库
            self.user_profiles[user_id] = user_context
            
            return user_context
            
        except Exception as e:
            logger.error(f"用户上下文分析失败: {e}")
            return self._create_default_user_context(user_id)
    
    def _infer_user_profile(self, context_data: Dict[str, Any]) -> UserProfileType:
        """推断用户类型"""
        # 基于多个因素推断用户类型
        indicators = context_data.get("indicators", {})
        
        # 年龄相关
        if indicators.get("age", 0) > 65:
            return UserProfileType.ELDERLY
        
        # 残障相关
        if indicators.get("disability", False):
            return UserProfileType.DISABLED
        
        # 经验相关
        experience_level = indicators.get("experience_level", "intermediate")
        if experience_level == "beginner":
            return UserProfileType.NOVICE
        elif experience_level == "expert":
            return UserProfileType.EXPERT
        
        # 角色相关
        user_role = indicators.get("role", "user")
        if user_role in ["manager", "executive"]:
            return UserProfileType.EXECUTIVE
        
        return UserProfileType.INTERMEDIATE
    
    def _create_default_user_context(self, user_id: str) -> UserContext:
        """创建默认用户上下文"""
        return UserContext(
            user_id=user_id,
            profile_type=UserProfileType.INTERMEDIATE,
            cognitive_load=CognitiveLoadLevel.MODERATE,
            trust_level=TrustLevel.MEDIUM,
            accessibility_needs=[],
            preferences={},
            session_history=[],
            emotional_state="neutral",
            task_complexity="medium",
            time_pressure="low",
            cultural_background="default"
        )
    
    def generate_adaptive_interface(self, user_context: UserContext) -> Dict[str, Any]:
        """
        生成自适应界面
        
        Args:
            user_context: 用户上下文
            
        Returns:
            自适应界面配置
        """
        try:
            # 认知负荷适应
            cognitive_adaptations = self._adapt_for_cognitive_load(user_context.cognitive_load)
            
            # 无障碍适应
            accessibility_adaptations = self.accessibility_manager.assess_user_accessibility_needs(user_context)
            
            # 信任建立适应
            trust_adaptations = self._adapt_for_trust_level(user_context.trust_level)
            
            # 行为经济学适应
            behavioral_adaptations = self.behavior_engine.apply_behavioral_interventions(
                user_context, "interface_engagement"
            )
            
            # 生成最终界面配置
            interface_config = {
                "user_id": user_context.user_id,
                "interface_complexity": cognitive_adaptations["complexity"],
                "information_density": cognitive_adaptations["density"],
                "accessibility_features": accessibility_adaptations["recommended_features"],
                "trust_building_elements": trust_adaptations,
                "behavioral_cues": [b["intervention"] for b in behavioral_adaptations[:3]],  # 取前3个
                "personalization": self._generate_personalization_rules(user_context),
                "adaptation_timestamp": datetime.now().isoformat()
            }
            
            return interface_config
            
        except Exception as e:
            logger.error(f"自适应界面生成失败: {e}")
            return {"error": str(e)}
    
    def _adapt_for_cognitive_load(self, cognitive_load: CognitiveLoadLevel) -> Dict[str, Any]:
        """基于认知负荷的适应性调整"""
        adaptations = {
            CognitiveLoadLevel.LOW: {
                "complexity": "advanced",
                "density": "high",
                "features": ["advanced_tools", "customization_options"]
            },
            CognitiveLoadLevel.MODERATE: {
                "complexity": "standard",
                "density": "medium",
                "features": ["standard_tools", "guided_workflow"]
            },
            CognitiveLoadLevel.HIGH: {
                "complexity": "simplified",
                "density": "low",
                "features": ["step_by_step", "auto_completion"]
            },
            CognitiveLoadLevel.OVERWHELMED: {
                "complexity": "minimal",
                "density": "very_low",
                "features": ["minimal_interface", "ai_assistance"]
            }
        }
        
        return adaptations.get(cognitive_load, adaptations[CognitiveLoadLevel.MODERATE])
    
    def _adapt_for_trust_level(self, trust_level: TrustLevel) -> List[Dict[str, Any]]:
        """基于信任等级的适应性调整"""
        trust_adaptations = {
            TrustLevel.NONE: [
                {"type": "transparency", "action": "show_detailed_process", "intensity": "high"},
                {"type": "security", "action": "display_security_certificates", "intensity": "high"},
                {"type": "competence", "action": "show_success_cases", "intensity": "medium"}
            ],
            TrustLevel.LOW: [
                {"type": "transparency", "action": "explain_decisions", "intensity": "medium"},
                {"type": "reliability", "action": "provide_consistent_experience", "intensity": "medium"},
                {"type": "empathy", "action": "acknowledge_user_concerns", "intensity": "medium"}
            ],
            TrustLevel.MEDIUM: [
                {"type": "competence", "action": "demonstrate_expertise", "intensity": "medium"},
                {"type": "efficiency", "action": "optimize_workflow", "intensity": "medium"},
                {"type": "personalization", "action": "adapt_to_preferences", "intensity": "low"}
            ],
            TrustLevel.HIGH: [
                {"type": "efficiency", "action": "streamline_interactions", "intensity": "high"},
                {"type": "autonomy", "action": "provide_advanced_options", "intensity": "medium"},
                {"type": "innovation", "action": "suggest_new_features", "intensity": "low"}
            ],
            TrustLevel.COMPLETE: [
                {"type": "innovation", "action": "co_create_features", "intensity": "high"},
                {"type": "community", "action": "connect_with_other_users", "intensity": "medium"},
                {"type": "advocacy", "action": "turn_user_into_promoter", "intensity": "high"}
            ]
        }
        
        return trust_adaptations.get(trust_level, [])
    
    def _generate_personalization_rules(self, user_context: UserContext) -> Dict[str, Any]:
        """生成个性化规则"""
        return {
            "language_preference": user_context.preferences.get("language", "auto"),
            "theme_preference": user_context.preferences.get("theme", "auto"),
            "interaction_speed": user_context.preferences.get("interaction_speed", "medium"),
            "feedback_frequency": user_context.preferences.get("feedback_frequency", "medium"),
            "automation_level": user_context.preferences.get("automation_level", "medium"),
            "cultural_adaptations": {
                "date_format": user_context.cultural_background,
                "time_format": user_context.cultural_background,
                "color_meanings": user_context.cultural_background
            }
        }
    
    def monitor_and_adapt(self, user_id: str, interaction_metrics: InteractionMetrics):
        """
        监控用户交互并进行适应性调整
        
        Args:
            user_id: 用户ID
            interaction_metrics: 交互指标
        """
        try:
            if user_id not in self.user_profiles:
                return
            
            user_context = self.user_profiles[user_id]
            
            # 更新交互历史
            self.interaction_history[user_id].append(asdict(interaction_metrics))
            
            # 保持最近100次交互记录
            if len(self.interaction_history[user_id]) > 100:
                self.interaction_history[user_id] = self.interaction_history[user_id][-100:]
            
            # 检查是否需要触发适应性调整
            adaptation_needed = self._check_adaptation_needed(user_context, interaction_metrics)
            
            if adaptation_needed:
                logger.info(f"用户{user_id}需要界面适应性调整")
                # 触发界面重新配置
                new_interface_config = self.generate_adaptive_interface(user_context)
                
                # 这里应该通知界面系统进行调整
                # 实际实现中会发送事件或调用API
            
        except Exception as e:
            logger.error(f"监控和适应失败: {e}")
    
    def _check_adaptation_needed(self, user_context: UserContext, 
                                metrics: InteractionMetrics) -> bool:
        """检查是否需要适应性调整"""
        # 认知负荷过高
        if user_context.cognitive_load in [CognitiveLoadLevel.HIGH, CognitiveLoadLevel.OVERWHELMED]:
            return True
        
        # 错误率过高
        if metrics.error_count > 3:
            return True
        
        # 满意度低
        if metrics.satisfaction_score < 3.0:  # 5分制
            return True
        
        # 认知努力过高
        if metrics.cognitive_effort > 7.0:  # 10分制
            return True
        
        # 信任度下降
        if metrics.trust_change < -0.2:
            return True
        
        return False
    
    def generate_trust_dashboard(self, user_id: str) -> Dict[str, Any]:
        """
        生成信任仪表板
        
        Args:
            user_id: 用户ID
            
        Returns:
            信任仪表板数据
        """
        try:
            if user_id not in self.user_profiles:
                return {"error": "用户不存在"}
            
            user_context = self.user_profiles[user_id]
            interaction_data = self.interaction_history[user_id]
            
            # 计算信任指标
            trust_metrics = {
                "current_trust_level": user_context.trust_level.value,
                "trust_trend": self._calculate_trust_trend(interaction_data),
                "transparency_score": self._calculate_transparency_score(interaction_data),
                "competence_score": self._calculate_competence_score(interaction_data),
                "reliability_score": self._calculate_reliability_score(interaction_data),
                "security_score": self._calculate_security_score(interaction_data),
                "empathy_score": self._calculate_empathy_score(interaction_data)
            }
            
            # 生成改进建议
            improvement_suggestions = self._generate_trust_improvements(user_context, trust_metrics)
            
            dashboard = {
                "user_id": user_id,
                "trust_metrics": trust_metrics,
                "suggestions": improvement_suggestions,
                "last_updated": datetime.now().isoformat(),
                "confidence_level": "high" if len(interaction_data) > 10 else "medium"
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"信任仪表板生成失败: {e}")
            return {"error": str(e)}
    
    def _calculate_trust_trend(self, interaction_data: List[Dict[str, Any]]) -> str:
        """计算信任趋势"""
        if len(interaction_data) < 2:
            return "insufficient_data"
        
        recent_trust = [data.get("trust_change", 0) for data in interaction_data[-5:]]
        avg_trust_change = statistics.mean(recent_trust)
        
        if avg_trust_change > 0.1:
            return "improving"
        elif avg_trust_change < -0.1:
            return "declining"
        else:
            return "stable"
    
    def _calculate_transparency_score(self, interaction_data: List[Dict[str, Any]]) -> float:
        """计算透明度分数"""
        # 简化计算：基于解释性交互的比例
        explanation_count = sum(1 for data in interaction_data 
                              if data.get("assistance_requests", 0) > 0)
        return min(explanation_count / max(len(interaction_data), 1) * 1.5, 1.0)
    
    def _calculate_competence_score(self, interaction_data: List[Dict[str, Any]]) -> float:
        """计算能力分数"""
        success_rates = [data.get("success_rate", 0) for data in interaction_data]
        return statistics.mean(success_rates) if success_rates else 0.5
    
    def _calculate_reliability_score(self, interaction_data: List[Dict[str, Any]]) -> float:
        """计算可靠性分数"""
        durations = [data.get("duration", 0) for data in interaction_data]
        if not durations:
            return 0.5
        
        # 计算时间一致性（标准差越小越可靠）
        duration_std = statistics.stdev(durations) if len(durations) > 1 else 0
        duration_mean = statistics.mean(durations)
        
        consistency_score = 1.0 - min(duration_std / (duration_mean + 0.1), 0.8)
        return consistency_score
    
    def _calculate_security_score(self, interaction_data: List[Dict[str, Any]]) -> float:
        """计算安全性分数"""
        # 简化计算：基于安全相关交互的比例
        security_interactions = sum(1 for data in interaction_data 
                                  if data.get("error_count", 0) == 0)
        return security_interactions / max(len(interaction_data), 1)
    
    def _calculate_empathy_score(self, interaction_data: List[Dict[str, Any]]) -> float:
        """计算共情分数"""
        satisfaction_scores = [data.get("satisfaction_score", 3) for data in interaction_data]
        normalized_scores = [score / 5.0 for score in satisfaction_scores]  # 5分制标准化
        return statistics.mean(normalized_scores) if normalized_scores else 0.5
    
    def _generate_trust_improvements(self, user_context: UserContext, 
                                   trust_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成信任改进建议"""
        suggestions = []
        
        # 基于低分项生成建议
        for metric, score in trust_metrics.items():
            if isinstance(score, (int, float)) and score < 0.7:
                if metric == "transparency_score":
                    suggestions.append({
                        "area": "透明度",
                        "suggestion": "增加决策过程的可视化展示",
                        "priority": "high",
                        "implementation": "添加'为什么这样推荐'功能"
                    })
                elif metric == "competence_score":
                    suggestions.append({
                        "area": "能力",
                        "suggestion": "展示成功案例和专业资质",
                        "priority": "high", 
                        "implementation": "添加案例研究和认证展示"
                    })
                elif metric == "reliability_score":
                    suggestions.append({
                        "area": "可靠性",
                        "suggestion": "提高响应时间和一致性",
                        "priority": "medium",
                        "implementation": "优化性能和错误处理"
                    })
                elif metric == "empathy_score":
                    suggestions.append({
                        "area": "共情",
                        "suggestion": "更好地理解和响应用户情感",
                        "priority": "medium",
                        "implementation": "添加情感识别和个性化回应"
                    })
        
        # 基于用户类型生成特定建议
        if user_context.profile_type == UserProfileType.ELDERLY:
            suggestions.append({
                "area": "用户体验",
                "suggestion": "为老年用户提供更详细的指导",
                "priority": "high",
                "implementation": "添加语音助手和简化界面"
            })
        
        return suggestions[:5]  # 限制建议数量

# 使用示例和测试
if __name__ == "__main__":
    # 创建人因工程与信任设计引擎
    hf_engine = HumanFactorsTrustEngine()
    
    # 创建测试用户上下文
    test_context_data = {
        "indicators": {
            "age": 72,
            "disability": False,
            "experience_level": "beginner",
            "role": "user"
        },
        "accessibility_needs": ["low_vision"],
        "preferences": {
            "language": "zh-CN",
            "theme": "high_contrast",
            "interaction_speed": "slow",
            "feedback_frequency": "high",
            "automation_level": "low"
        },
        "emotional_state": "anxious",
        "task_complexity": "high",
        "time_pressure": "medium",
        "cultural_background": "chinese",
        "interaction_data": {
            "interface_elements": 15,
            "navigation_depth": 4,
            "information_chunks": 12,
            "screen_clutter": 8,
            "decision_points": 6,
            "response_time": 4.5,
            "error_count": 3,
            "total_interactions": 20
        },
        "system_actions": [
            {"type": "explanation", "success": True, "shows_process": True},
            {"type": "recommendation", "success": False, "consistent": True},
            {"type": "error_handling", "success": True, "available": True}
        ]
    }
    
    # 分析用户上下文
    user_context = hf_engine.analyze_user_context("test_user_001", test_context_data)
    print(f"用户类型: {user_context.profile_type.value}")
    print(f"认知负荷: {user_context.cognitive_load.value}")
    print(f"信任等级: {user_context.trust_level.value}")
    
    # 生成自适应界面
    interface_config = hf_engine.generate_adaptive_interface(user_context)
    print(f"界面配置: {json.dumps(interface_config, indent=2, ensure_ascii=False)}")
    
    # 模拟交互指标
    test_metrics = InteractionMetrics(
        timestamp=datetime.now(),
        user_id="test_user_001",
        interaction_type="task_completion",
        duration=120.5,
        success_rate=0.75,
        error_count=2,
        assistance_requests=3,
        satisfaction_score=3.2,
        cognitive_effort=6.8,
        trust_change=-0.1
    )
    
    # 监控和适应
    hf_engine.monitor_and_adapt("test_user_001", test_metrics)
    
    # 生成信任仪表板
    trust_dashboard = hf_engine.generate_trust_dashboard("test_user_001")
    print(f"信任仪表板: {json.dumps(trust_dashboard, indent=2, ensure_ascii=False)}")