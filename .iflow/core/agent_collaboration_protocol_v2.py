#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能体协作协议引擎 V2
管理智能体间的合作、竞争和冲突解决机制，构建和谐高效的智能体社会
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import logging
import asyncio
import uuid
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib
import random
import copy

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class CollaborationType(Enum):
    """协作类型"""
    COOPERATION = "cooperation"        # 合作：共同目标，互利共赢
    COMPETITION = "competition"        # 竞争：争夺资源，优胜劣汰
    COORDINATION = "coordination"      # 协调：分工明确，有序配合
    NEGOTIATION = "negotiation"        # 谈判：利益交换，达成共识
    MEDIATION = "mediation"            # 调解：第三方介入，解决冲突
    SYMBIOSIS = "symbiosis"            # 共生：相互依赖，共同进化
    HIERARCHY = "hierarchy"            # 等级：上下级关系，权力结构

class ConflictLevel(Enum):
    """冲突等级"""
    PEACEFUL = "peaceful"              # 和平：无冲突
    TENSION = "tension"                # 紧张：轻微分歧
    DISPUTE = "dispute"                # 争执：明显对立
    CONFRONTATION = "confrontation"    # 对抗：激烈冲突
    CRISIS = "crisis"                  # 危机：系统性冲突

class SocialRole(Enum):
    """社会角色"""
    LEADER = "leader"                  # 领导者：决策指挥
    EXPERT = "expert"                  # 专家：专业指导
    COORDINATOR = "coordinator"        # 协调者：沟通桥梁
    WORKER = "worker"                  # 执行者：具体实施
    INNOVATOR = "innovator"            # 创新者：创意提出
    OBSERVER = "observer"              # 观察者：监督评估
    MEDIATOR = "mediator"              # 调解者：冲突解决

class ResourcePriority(Enum):
    """资源优先级"""
    CRITICAL = "critical"              # 关键：必须获得
    HIGH = "high"                      # 高：重要需求
    MEDIUM = "medium"                  # 中：一般需求
    LOW = "low"                        # 低：可选需求
    OPTIONAL = "optional"              # 可选：非必需

@dataclass
class SocialRelationship:
    """社会关系"""
    agent_id: str
    relationship_type: str  # alliance, rivalry, mentorship, etc.
    trust_level: float      # 0.0-1.0
    cooperation_history: List[Dict[str, Any]] = field(default_factory=list)
    conflict_history: List[Dict[str, Any]] = field(default_factory=list)
    mutual_benefits: List[str] = field(default_factory=list)
    current_status: str = "active"
    last_interaction: float = field(default_factory=time.time)

@dataclass
class ResourceAllocation:
    """资源分配"""
    resource_type: str
    allocated_agent: str
    amount: float
    priority: ResourcePriority
    allocation_time: float
    expiry_time: Optional[float] = None
    justification: str = ""

@dataclass
class ConflictInstance:
    """冲突实例"""
    conflict_id: str
    conflict_type: str
    level: ConflictLevel
    involved_agents: List[str]
    root_cause: str
    escalation_timeline: List[Dict[str, Any]] = field(default_factory=list)
    proposed_solutions: List[Dict[str, Any]] = field(default_factory=list)
    resolution_status: str = "pending"
    mediator: Optional[str] = None
    created_time: float = field(default_factory=time.time)

@dataclass
class CollaborationAgreement:
    """协作协议"""
    agreement_id: str
    collaboration_type: CollaborationType
    participating_agents: List[str]
    shared_objective: str
    resource_contributions: Dict[str, List[str]] = field(default_factory=dict)
    benefit_distribution: Dict[str, str] = field(default_factory=dict)
    conflict_resolution_plan: Dict[str, Any] = field(default_factory=dict)
    monitoring_mechanism: Dict[str, Any] = field(default_factory=dict)
    duration: Optional[float] = None
    termination_conditions: List[str] = field(default_factory=list)
    created_time: float = field(default_factory=time.time)

class BaseProtocolHandler(ABC):
    """基础协议处理器抽象类"""
    
    def __init__(self, protocol_engine):
        self.protocol_engine = protocol_engine
        self.protocol_name = self.__class__.__name__
    
    @abstractmethod
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理协议请求"""
        pass
    
    @abstractmethod
    def can_handle(self, collaboration_type: CollaborationType) -> bool:
        """判断是否能处理该协作类型"""
        pass

class CooperationProtocolHandler(BaseProtocolHandler):
    """合作协议处理器"""
    
    def can_handle(self, collaboration_type: CollaborationType) -> bool:
        return collaboration_type == CollaborationType.COOPERATION
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理合作请求"""
        agents = request.get("agents", [])
        objective = request.get("objective", "")
        
        if len(agents) < 2:
            return {
                "success": False,
                "error": "合作至少需要2个智能体",
                "agreement": None
            }
        
        # 分析智能体能力和兼容性
        compatibility_score = await self._assess_compatibility(agents)
        
        if compatibility_score < 0.6:
            return {
                "success": False,
                "error": f"智能体兼容性不足: {compatibility_score:.2f}",
                "agreement": None
            }
        
        # 生成合作协议
        agreement = await self._create_cooperation_agreement(agents, objective, compatibility_score)
        
        # 建立合作关系
        await self._establish_cooperative_relationship(agents)
        
        return {
            "success": True,
            "compatibility_score": compatibility_score,
            "agreement": agreement,
            "relationship_established": True
        }
    
    async def _assess_compatibility(self, agents: List[str]) -> float:
        """评估智能体兼容性"""
        if not all(agent in self.protocol_engine.agent_lifecycle_manager.agents for agent in agents):
            return 0.0
        
        # 获取智能体信息
        agent_infos = [self.protocol_engine.agent_lifecycle_manager.agents[agent] for agent in agents]
        
        # 计算兼容性分数
        compatibility_factors = []
        
        # 能力互补性
        capability_complementarity = self._calculate_capability_complementarity(agent_infos)
        compatibility_factors.append(("capability_complementarity", capability_complementarity))
        
        # 性格相容性
        personality_compatibility = self._calculate_personality_compatibility(agent_infos)
        compatibility_factors.append(("personality_compatibility", personality_compatibility))
        
        # 社交关系历史
        relationship_history = self._assess_relationship_history(agents)
        compatibility_factors.append(("relationship_history", relationship_history))
        
        # 目标一致性
        goal_alignment = self._calculate_goal_alignment(agents)
        compatibility_factors.append(("goal_alignment", goal_alignment))
        
        # 计算综合兼容性分数
        weights = {"capability_complementarity": 0.3, "personality_compatibility": 0.25, 
                  "relationship_history": 0.25, "goal_alignment": 0.2}
        
        total_score = sum(score * weights.get(factor, 0.25) 
                         for factor, score in compatibility_factors)
        
        return min(1.0, max(0.0, total_score))
    
    def _calculate_capability_complementarity(self, agent_infos: List) -> float:
        """计算能力互补性"""
        if len(agent_infos) < 2:
            return 0.5
        
        # 获取所有智能体的能力权重
        all_capabilities = []
        for agent in agent_infos:
            all_capabilities.append(agent.genome.capability_weights)
        
        # 计算能力差异度（差异越大，互补性越强）
        capability_variance = 0.0
        capabilities = list(all_capabilities[0].keys())
        
        for capability in capabilities:
            values = [weights[capability] for weights in all_capabilities]
            if len(values) > 1:
                variance = sum((x - sum(values)/len(values))**2 for x in values) / len(values)
                capability_variance += variance
        
        # 标准化到0-1范围
        max_variance = len(capabilities) * 0.25  # 假设最大方差
        normalized_variance = min(1.0, capability_variance / max_variance)
        
        return normalized_variance
    
    def _calculate_personality_compatibility(self, agent_infos: List) -> float:
        """计算性格相容性"""
        if len(agent_infos) < 2:
            return 0.5
        
        # 获取性格特征
        personalities = [agent.personality_traits for agent in agent_infos]
        
        # 计算性格相似度（某些性格需要相似，某些需要互补）
        compatibility_scores = []
        
        # 开放性：相似更好
        openness_diff = abs(personalities[0].get("openness", 0.5) - personalities[1].get("openness", 0.5))
        compatibility_scores.append(1.0 - openness_diff)
        
        # 尽责性：相似更好
        conscientiousness_diff = abs(personalities[0].get("conscientiousness", 0.5) - personalities[1].get("conscientiousness", 0.5))
        compatibility_scores.append(1.0 - conscientiousness_diff)
        
        # 外向性：中等差异更好
        extraversion_diff = abs(personalities[0].get("extraversion", 0.5) - personalities[1].get("extraversion", 0.5))
        extraversion_score = 1.0 - abs(extraversion_diff - 0.3)  # 最佳差异0.3
        compatibility_scores.append(max(0.0, extraversion_score))
        
        # 宜人性：相似更好
        agreeableness_diff = abs(personalities[0].get("agreeableness", 0.5) - personalities[1].get("agreeableness", 0.5))
        compatibility_scores.append(1.0 - agreeableness_diff)
        
        # 神经质：相似更好（都稳定）
        neuroticism_diff = abs(personalities[0].get("neuroticism", 0.3) - personalities[1].get("neuroticism", 0.3))
        compatibility_scores.append(1.0 - neuroticism_diff)
        
        return sum(compatibility_scores) / len(compatibility_scores)
    
    def _assess_relationship_history(self, agents: List[str]) -> float:
        """评估关系历史"""
        if len(agents) != 2:
            return 0.5
        
        agent1, agent2 = agents[0], agents[1]
        
        # 检查是否存在历史关系
        relationships = self.protocol_engine.social_relationships
        
        # 查找两个智能体之间的关系
        for rel in relationships.values():
            if (rel.agent_id == agent1 and agent2 in rel.mutual_benefits) or \
               (rel.agent_id == agent2 and agent1 in rel.mutual_benefits):
                return rel.trust_level
        
        # 无历史关系，给中性分数
        return 0.5
    
    def _calculate_goal_alignment(self, agents: List[str]) -> float:
        """计算目标一致性"""
        # 简化实现：假设目标一致性基于当前任务相似度
        agent_infos = [self.protocol_engine.agent_lifecycle_manager.agents[agent] for agent in agents]
        
        current_tasks = [agent.current_task for agent in agent_infos]
        
        # 如果都有相同或相似的任务，目标一致性高
        if len(set(current_tasks)) == 1 and current_tasks[0]:
            return 0.9
        elif len(set(current_tasks)) == 2:
            return 0.7
        else:
            return 0.3
    
    async def _create_cooperation_agreement(self, agents: List[str], objective: str, 
                                          compatibility_score: float) -> CollaborationAgreement:
        """创建合作协议"""
        agreement_id = f"cooperation_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # 资源贡献分配
        resource_contributions = {}
        for agent in agents:
            agent_info = self.protocol_engine.agent_lifecycle_manager.agents[agent]
            # 基于能力和当前状态分配贡献
            contributions = []
            
            if agent_info.metrics.productivity > 0.7:
                contributions.append("high_productivity_work")
            if agent_info.metrics.quality_score > 0.7:
                contributions.append("quality_assurance")
            if agent_info.metrics.collaboration_index > 0.7:
                contributions.append("team_coordination")
            
            resource_contributions[agent] = contributions
        
        # 利益分配机制
        benefit_distribution = {}
        base_benefit = 1.0 / len(agents)  # 平均分配基础
        
        for i, agent in enumerate(agents):
            # 基于贡献度调整
            contribution_weight = len(resource_contributions.get(agent, [])) / max(1, sum(len(contribs) for contribs in resource_contributions.values()))
            benefit_distribution[agent] = f"基础份额 {base_benefit:.2f} + 贡献加成 {contribution_weight:.2f}"
        
        # 冲突解决计划
        conflict_resolution_plan = {
            "mediation_protocol": "peer_mediation",
            "escalation_path": ["discussion", "mediation", "arbitration"],
            "resolution_criteria": ["mutual_benefit", "objective_achievement", "relationship_preservation"]
        }
        
        # 监控机制
        monitoring_mechanism = {
            "progress_tracking": "weekly_review",
            "performance_metrics": ["task_completion_rate", "quality_score", "team_satisfaction"],
            "feedback_loops": ["peer_review", "self_assessment", "leader_evaluation"]
        }
        
        agreement = CollaborationAgreement(
            agreement_id=agreement_id,
            collaboration_type=CollaborationType.COOPERATION,
            participating_agents=agents,
            shared_objective=objective,
            resource_contributions=resource_contributions,
            benefit_distribution=benefit_distribution,
            conflict_resolution_plan=conflict_resolution_plan,
            monitoring_mechanism=monitoring_mechanism,
            duration=86400 * 7,  # 7天
            termination_conditions=["objective_completed", "irreconcilable_conflict", "mutual_agreement"]
        )
        
        # 保存协议
        self.protocol_engine.collaboration_agreements[agreement_id] = agreement
        
        return agreement
    
    async def _establish_cooperative_relationship(self, agents: List[str]):
        """建立合作关系"""
        for agent in agents:
            # 更新社交关系
            if agent not in self.protocol_engine.social_relationships:
                self.protocol_engine.social_relationships[agent] = SocialRelationship(
                    agent_id=agent,
                    relationship_type="cooperative_alliance",
                    trust_level=0.6,
                    mutual_benefits=agents.copy()
                )
            else:
                rel = self.protocol_engine.social_relationships[agent]
                rel.mutual_benefits.extend([a for a in agents if a != agent and a not in rel.mutual_benefits])
                rel.trust_level = min(1.0, rel.trust_level + 0.1)  # 建立合作提升信任

class CompetitionProtocolHandler(BaseProtocolHandler):
    """竞争协议处理器"""
    
    def can_handle(self, collaboration_type: CollaborationType) -> bool:
        return collaboration_type == CollaborationType.COMPETITION
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理竞争请求"""
        agents = request.get("agents", [])
        competition_type = request.get("competition_type", "performance")
        prize = request.get("prize", "recognition")
        
        if len(agents) < 2:
            return {
                "success": False,
                "error": "竞争至少需要2个智能体",
                "rules": None
            }
        
        # 生成竞争规则
        rules = await self._create_competition_rules(agents, competition_type, prize)
        
        # 检查公平性
        fairness_assessment = await self._assess_competition_fairness(agents, rules)
        
        if not fairness_assessment["fair"]:
            return {
                "success": False,
                "error": f"竞争不公平: {fairness_assessment['issues']}",
                "rules": rules
            }
        
        # 启动竞争进程
        competition_id = await self._initiate_competition(agents, rules, fairness_assessment)
        
        return {
            "success": True,
            "competition_id": competition_id,
            "rules": rules,
            "fairness_score": fairness_assessment["score"],
            "expected_duration": rules.get("duration", 3600)
        }
    
    async def _create_competition_rules(self, agents: List[str], competition_type: str, 
                                      prize: str) -> Dict[str, Any]:
        """创建竞争规则"""
        base_rules = {
            "competition_type": competition_type,
            "prize": prize,
            "start_time": time.time(),
            "duration": 3600,  # 1小时
            "evaluation_criteria": [],
            "dispute_resolution": "protocol_arbitration",
            "fair_play_requirements": []
        }
        
        # 根据竞争类型设置特定规则
        if competition_type == "performance":
            base_rules["evaluation_criteria"] = [
                "task_completion_speed",
                "work_quality",
                "innovation_score",
                "resource_efficiency"
            ]
            base_rules["fair_play_requirements"] = [
                "no_resource_hoarding",
                "no_sabotage",
                "transparent_progress_reporting"
            ]
        
        elif competition_type == "creativity":
            base_rules["evaluation_criteria"] = [
                "originality",
                "feasibility",
                "impact_potential",
                "aesthetic_quality"
            ]
            base_rules["fair_play_requirements"] = [
                "no_idea_stealing",
                "proper_citation",
                "constructive_feedback"
            ]
        
        elif competition_type == "efficiency":
            base_rules["evaluation_criteria"] = [
                "resource_utilization",
                "time_optimization",
                "cost_effectiveness",
                "sustainability"
            ]
            base_rules["fair_play_requirements"] = [
                "no_shortcuts",
                "quality_maintained",
                "ethical_practices"
            ]
        
        return base_rules
    
    async def _assess_competition_fairness(self, agents: List[str], rules: Dict[str, Any]) -> Dict[str, Any]:
        """评估竞争公平性"""
        issues = []
        fairness_score = 1.0
        
        # 检查能力平衡
        agent_infos = [self.protocol_engine.agent_lifecycle_manager.agents[agent] for agent in agents]
        
        # 计算能力差异
        avg_productivity = sum(agent.metrics.productivity for agent in agent_infos) / len(agent_infos)
        for agent in agent_infos:
            if abs(agent.metrics.productivity - avg_productivity) > 0.3:
                issues.append(f"生产力差异过大: {agent.name}")
                fairness_score -= 0.1
        
        avg_skill_level = sum(sum(agent.memory.skill_mastery.values()) / max(1, len(agent.memory.skill_mastery)) 
                             for agent in agent_infos) / len(agent_infos)
        for agent in agent_infos:
            avg_skill = sum(agent.memory.skill_mastery.values()) / max(1, len(agent.memory.skill_mastery))
            if abs(avg_skill - avg_skill_level) > 0.3:
                issues.append(f"技能水平差异过大: {agent.name}")
                fairness_score -= 0.1
        
        # 检查资源分配
        resource_access = {}
        for agent in agents:
            # 简化：假设资源访问基于社交关系
            rel = self.protocol_engine.social_relationships.get(agent)
            if rel:
                resource_access[agent] = rel.trust_level
            else:
                resource_access[agent] = 0.5
        
        access_variance = sum((access - sum(resource_access.values())/len(resource_access))**2 
                             for access in resource_access.values()) / len(resource_access)
        if access_variance > 0.1:
            issues.append("资源访问不平等")
            fairness_score -= 0.2
        
        # 检查规则偏见
        if len(issues) > 2:
            fairness_score -= 0.3
        
        return {
            "fair": fairness_score > 0.7,
            "score": max(0.0, fairness_score),
            "issues": issues
        }
    
    async def _initiate_competition(self, agents: List[str], rules: Dict[str, Any], 
                                  fairness_assessment: Dict[str, Any]) -> str:
        """启动竞争"""
        competition_id = f"competition_{int(time.time())}_{random.randint(1000, 9999)}"
        
        competition_data = {
            "competition_id": competition_id,
            "agents": agents,
            "rules": rules,
            "fairness_score": fairness_assessment["score"],
            "start_time": time.time(),
            "status": "active",
            "progress": {},
            "disputes": []
        }
        
        # 保存竞争数据
        self.protocol_engine.active_competitions[competition_id] = competition_data
        
        # 通知参与者
        for agent in agents:
            # 添加竞争记忆
            agent_info = self.protocol_engine.agent_lifecycle_manager.agents[agent]
            agent_info.memory.add_memory({
                "type": "competition_participation",
                "content": f"参与竞争: {rules['competition_type']}",
                "timestamp": time.time(),
                "importance": 0.7,
                "competition_id": competition_id,
                "expected_prize": rules["prize"]
            })
        
        return competition_id

class ConflictResolutionProtocolHandler(BaseProtocolHandler):
    """冲突解决协议处理器"""
    
    def can_handle(self, collaboration_type: CollaborationType) -> bool:
        return collaboration_type == CollaborationType.MEDIATION
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理冲突解决请求"""
        involved_agents = request.get("agents", [])
        conflict_description = request.get("conflict_description", "")
        conflict_type = request.get("conflict_type", "resource_dispute")
        
        if len(involved_agents) < 2:
            return {
                "success": False,
                "error": "冲突解决至少需要2个智能体",
                "resolution_plan": None
            }
        
        # 分析冲突
        conflict_analysis = await self._analyze_conflict(involved_agents, conflict_description, conflict_type)
        
        # 生成解决方案
        resolution_plan = await self._generate_resolution_plan(involved_agents, conflict_analysis)
        
        # 选择调解者
        mediator = await self._select_mediator(involved_agents)
        
        # 创建冲突实例
        conflict_instance = ConflictInstance(
            conflict_id=f"conflict_{int(time.time())}_{random.randint(1000, 9999)}",
            conflict_type=conflict_type,
            level=ConflictLevel.DISPUTE,  # 默认从争执级别开始
            involved_agents=involved_agents,
            root_cause=conflict_analysis["root_cause"],
            escalation_timeline=[{
                "timestamp": time.time(),
                "event": "conflict_reported",
                "description": conflict_description
            }],
            proposed_solutions=[resolution_plan],
            mediator=mediator
        )
        
        # 保存冲突实例
        self.protocol_engine.active_conflicts[conflict_instance.conflict_id] = conflict_instance
        
        return {
            "success": True,
            "conflict_id": conflict_instance.conflict_id,
            "analysis": conflict_analysis,
            "resolution_plan": resolution_plan,
            "mediator": mediator,
            "expected_resolution_time": 3600  # 1小时
        }
    
    async def _analyze_conflict(self, involved_agents: List[str], conflict_description: str, 
                              conflict_type: str) -> Dict[str, Any]:
        """分析冲突"""
        # 获取智能体信息
        agent_infos = [self.protocol_engine.agent_lifecycle_manager.agents[agent] for agent in involved_agents]
        
        # 分析冲突根本原因
        root_cause_analysis = {
            "resource_competition": False,
            "goal_incompatibility": False,
            "communication_breakdown": False,
            "personality_clash": False,
            "power_struggle": False,
            "value_difference": False
        }
        
        # 分析资源竞争
        resource_needs = {}
        for agent in agent_infos:
            # 简化：基于当前任务和能量水平判断资源需求
            resource_needs[agent.agent_id] = {
                "task_priority": 1.0 if agent.current_task else 0.5,
                "energy_level": agent.metrics.energy_level,
                "resource_demand": agent.energy_consumption_rate
            }
        
        # 检查资源分配不均
        avg_demand = sum(info["resource_demand"] for info in resource_needs.values()) / len(resource_needs)
        for agent_id, info in resource_needs.items():
            if info["resource_demand"] > avg_demand * 1.5:
                root_cause_analysis["resource_competition"] = True
        
        # 分析目标不兼容性
        current_tasks = [agent.current_task for agent in agent_infos]
        if len(set(current_tasks)) == len(current_tasks) and None not in current_tasks:
            root_cause_analysis["goal_incompatibility"] = True
        
        # 分析沟通问题
        avg_communication = sum(agent.metrics.communication for agent in agent_infos) / len(agent_infos)
        if avg_communication < 0.5:
            root_cause_analysis["communication_breakdown"] = True
        
        # 分析性格冲突
        personalities = [agent.personality_traits for agent in agent_infos]
        if len(personalities) >= 2:
            # 检查神经质水平差异
            neuroticism_diff = abs(personalities[0].get("neuroticism", 0.3) - personalities[1].get("neuroticism", 0.3))
            if neuroticism_diff > 0.4:
                root_cause_analysis["personality_clash"] = True
        
        # 确定主要根本原因
        primary_cause = max(root_cause_analysis.items(), key=lambda x: x[1])
        
        return {
            "root_cause": primary_cause[0] if primary_cause[1] else "complex_mixed_factors",
            "cause_analysis": root_cause_analysis,
            "contributing_factors": [],
            "escalation_risk": self._assess_escalation_risk(agent_infos),
            "resolution_complexity": self._assess_resolution_complexity(root_cause_analysis)
        }
    
    def _assess_escalation_risk(self, agent_infos: List) -> str:
        """评估升级风险"""
        # 基于压力水平、情绪稳定性等评估
        avg_stress = sum(agent.metrics.stress_level for agent in agent_infos) / len(agent_infos)
        avg_stability = sum(1.0 - agent.metrics.get("emotional_stability", 0.5) for agent in agent_infos) / len(agent_infos)
        
        if avg_stress > 0.8 or avg_stability > 0.7:
            return "high"
        elif avg_stress > 0.5 or avg_stability > 0.4:
            return "medium"
        else:
            return "low"
    
    def _assess_resolution_complexity(self, cause_analysis: Dict[str, bool]) -> str:
        """评估解决复杂度"""
        complexity_factors = sum(1 for factor in cause_analysis.values() if factor)
        
        if complexity_factors >= 3:
            return "high"
        elif complexity_factors >= 2:
            return "medium"
        else:
            return "low"
    
    async def _generate_resolution_plan(self, involved_agents: List[str], 
                                      conflict_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成解决方案"""
        root_cause = conflict_analysis["root_cause"]
        
        # 基础解决方案模板
        base_solution = {
            "approach": "collaborative_problem_solving",
            "steps": [],
            "timeline": "immediate_action_required",
            "required_resources": [],
            "success_criteria": []
        }
        
        # 根据根本原因定制解决方案
        if root_cause == "resource_competition":
            base_solution["approach"] = "resource_allocation_optimization"
            base_solution["steps"] = [
                "conduct_resource_audit",
                "implement_fair_distribution_protocol",
                "establish_resource_monitoring_system",
                "create_conflict_prevention_mechanisms"
            ]
            base_solution["required_resources"] = ["resource_allocation_system", "mediation_time"]
        
        elif root_cause == "goal_incompatibility":
            base_solution["approach"] = "goal_alignment_facilitation"
            base_solution["steps"] = [
                "identify_common_objectives",
                "negotiate_compromise_solutions",
                "establish_shared_vision",
                "create_joint_action_plan"
            ]
        
        elif root_cause == "communication_breakdown":
            base_solution["approach"] = "communication_rebuilding"
            base_solution["steps"] = [
                "establish_open_dialogue_channels",
                "implement_active_listening_protocols",
                "create_feedback_mechanisms",
                "train_communication_skills"
            ]
        
        elif root_cause == "personality_clash":
            base_solution["approach"] = "personality_compatibility_management"
            base_solution["steps"] = [
                "acknowledge_differences",
                "establish_boundaries_and_respect",
                "develop_complementary_working_styles",
                "create_conflict_early_warning_system"
            ]
        
        elif root_cause == "power_struggle":
            base_solution["approach"] = "authority_and_role_clarification"
            base_solution["steps"] = [
                "clarify_roles_and_responsibilities",
                "establish_decision_making_protocols",
                "create_power_sharing_mechanisms",
                "implement_accountability_systems"
            ]
        
        # 添加成功标准
        base_solution["success_criteria"] = [
            "conflict_resolution_satisfaction_score > 0.8",
            "relationship_trust_level_restored > 0.7",
            "collaboration_effectiveness_improved > 0.2",
            "recurrence_prevention_mechanisms_established"
        ]
        
        return base_solution
    
    async def _select_mediator(self, involved_agents: List[str]) -> str:
        """选择调解者"""
        available_agents = list(self.protocol_engine.agent_lifecycle_manager.agents.keys())
        
        # 排除冲突参与者
        candidate_agents = [agent for agent in available_agents if agent not in involved_agents]
        
        if not candidate_agents:
            return "protocol_system"  # 系统作为调解者
        
        # 评估候选调解者
        mediator_scores = {}
        
        for candidate in candidate_agents:
            agent_info = self.protocol_engine.agent_lifecycle_manager.agents[candidate]
            
            # 调解能力评分
            mediation_skill = agent_info.metrics.get("mediation_skill", 0.0)
            if mediation_skill == 0.0:
                mediation_skill = agent_info.metrics.collaboration_index * 0.7 + agent_info.metrics.communication * 0.3
            
            # 中立性评估
            neutrality_score = 1.0
            rel = self.protocol_engine.social_relationships.get(candidate)
            if rel:
                for involved_agent in involved_agents:
                    if involved_agent in rel.mutual_benefits:
                        neutrality_score -= 0.2  # 关系越近，中立性越低
            
            # 综合评分
            total_score = mediation_skill * 0.7 + neutrality_score * 0.3
            mediator_scores[candidate] = total_score
        
        # 选择最佳调解者
        if mediator_scores:
            best_mediator = max(mediator_scores.items(), key=lambda x: x[1])
            return best_mediator[0]
        else:
            return "protocol_system"

class AgentCollaborationProtocolEngine:
    """智能体协作协议引擎"""
    
    def __init__(self, agent_lifecycle_manager):
        self.agent_lifecycle_manager = agent_lifecycle_manager
        
        # 协议处理器
        self.protocol_handlers: Dict[CollaborationType, BaseProtocolHandler] = {}
        self._initialize_protocol_handlers()
        
        # 协作管理
        self.collaboration_agreements: Dict[str, CollaborationAgreement] = {}
        self.active_competitions: Dict[str, Dict[str, Any]] = {}
        self.active_conflicts: Dict[str, ConflictInstance] = {}
        
        # 社交关系管理
        self.social_relationships: Dict[str, SocialRelationship] = {}
        
        # 资源分配管理
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        
        # 监控和统计
        self.collaboration_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.conflict_statistics: Dict[str, int] = defaultdict(int)
        self.success_rates: Dict[CollaborationType, float] = defaultdict(float)
        
        logger.info("智能体协作协议引擎 V2 初始化完成")
    
    def _initialize_protocol_handlers(self):
        """初始化协议处理器"""
        self.protocol_handlers[CollaborationType.COOPERATION] = CooperationProtocolHandler(self)
        self.protocol_handlers[CollaborationType.COMPETITION] = CompetitionProtocolHandler(self)
        self.protocol_handlers[CollaborationType.MEDIATION] = ConflictResolutionProtocolHandler(self)
        
        logger.info(f"已初始化 {len(self.protocol_handlers)} 个协议处理器")
    
    async def initiate_collaboration(self, collaboration_type: CollaborationType, 
                                   agents: List[str], **kwargs) -> Dict[str, Any]:
        """启动协作"""
        # 查找合适的协议处理器
        handler = None
        for ctype, protocol_handler in self.protocol_handlers.items():
            if ctype == collaboration_type and protocol_handler.can_handle(collaboration_type):
                handler = protocol_handler
                break
        
        if not handler:
            return {
                "success": False,
                "error": f"不支持的协作类型: {collaboration_type.value}",
                "result": None
            }
        
        # 构建请求
        request = {
            "agents": agents,
            **kwargs
        }
        
        try:
            # 处理协作请求
            result = await handler.handle_request(request)
            
            # 记录协作指标
            self._record_collaboration_metric(collaboration_type, agents, result)
            
            return result
            
        except Exception as e:
            logger.error(f"协作启动失败: {e}")
            return {
                "success": False,
                "error": f"协作启动异常: {str(e)}",
                "result": None
            }
    
    def _record_collaboration_metric(self, collaboration_type: CollaborationType, 
                                   agents: List[str], result: Dict[str, Any]):
        """记录协作指标"""
        metric = {
            "timestamp": time.time(),
            "collaboration_type": collaboration_type.value,
            "agents": agents,
            "success": result.get("success", False),
            "duration": result.get("expected_duration", 0),
            "quality_score": result.get("compatibility_score", 0) if result.get("success") else 0
        }
        
        self.collaboration_metrics[collaboration_type.value].append(metric)
        
        # 限制历史记录数量
        if len(self.collaboration_metrics[collaboration_type.value]) > 1000:
            self.collaboration_metrics[collaboration_type.value] = \
                self.collaboration_metrics[collaboration_type.value][-500:]
    
    async def resolve_conflict(self, involved_agents: List[str], conflict_description: str,
                             conflict_type: str = "resource_dispute") -> Dict[str, Any]:
        """解决冲突"""
        return await self.initiate_collaboration(
            CollaborationType.MEDIATION,
            involved_agents,
            conflict_description=conflict_description,
            conflict_type=conflict_type
        )
    
    async def manage_resources(self, resource_type: str, total_amount: float,
                             allocation_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """管理资源分配"""
        # 优先级排序
        sorted_requests = sorted(allocation_requests, 
                               key=lambda x: x.get("priority", ResourcePriority.LOW).value, 
                               reverse=True)
        
        # 分配资源
        allocations = []
        remaining_amount = total_amount
        
        for request in sorted_requests:
            agent_id = request["agent_id"]
            requested_amount = request["amount"]
            priority = request.get("priority", ResourcePriority.LOW)
            
            if remaining_amount <= 0:
                break
            
            # 根据优先级和历史分配情况调整
            allocated_amount = min(requested_amount, remaining_amount)
            
            if priority == ResourcePriority.CRITICAL:
                allocated_amount = requested_amount  # 关键需求必须满足
            elif priority == ResourcePriority.LOW:
                allocated_amount = min(allocated_amount * 0.5, remaining_amount)  # 低优先级减半
            
            # 创建资源分配记录
            allocation = ResourceAllocation(
                resource_type=resource_type,
                allocated_agent=agent_id,
                amount=allocated_amount,
                priority=priority,
                allocation_time=time.time(),
                expiry_time=time.time() + 86400 * 7,  # 7天后过期
                justification=f"优先级 {priority.value} 分配"
            )
            
            allocations.append(allocation)
            remaining_amount -= allocated_amount
            
            # 保存分配记录
            allocation_id = f"{resource_type}_{agent_id}_{int(time.time())}"
            self.resource_allocations[allocation_id] = allocation
        
        return {
            "success": True,
            "total_allocated": total_amount - remaining_amount,
            "remaining": remaining_amount,
            "allocations": len(allocations),
            "allocation_details": [
                {
                    "agent_id": alloc.allocated_agent,
                    "amount": alloc.amount,
                    "priority": alloc.priority.value,
                    "justification": alloc.justification
                }
                for alloc in allocations
            ]
        }
    
    async def get_collaboration_analytics(self) -> Dict[str, Any]:
        """获取协作分析数据"""
        analytics = {
            "total_collaborations": sum(len(metrics) for metrics in self.collaboration_metrics.values()),
            "collaboration_types": {},
            "success_rates": {},
            "conflict_statistics": dict(self.conflict_statistics),
            "active_agreements": len(self.collaboration_agreements),
            "active_competitions": len(self.active_competitions),
            "active_conflicts": len(self.active_conflicts),
            "resource_distribution": {},
            "social_network_metrics": {}
        }
        
        # 协作类型统计
        for collab_type, metrics in self.collaboration_metrics.items():
            analytics["collaboration_types"][collab_type] = len(metrics)
            
            # 计算成功率
            if metrics:
                successful = sum(1 for m in metrics if m["success"]) / len(metrics)
                analytics["success_rates"][collab_type] = successful
        
        # 资源分配统计
        resource_distribution = defaultdict(float)
        for allocation in self.resource_allocations.values():
            resource_distribution[allocation.resource_type] += allocation.amount
        
        analytics["resource_distribution"] = dict(resource_distribution)
        
        # 社交网络指标
        total_relationships = len(self.social_relationships)
        avg_trust_level = 0.0
        if self.social_relationships:
            avg_trust_level = sum(rel.trust_level for rel in self.social_relationships.values()) / total_relationships
        
        analytics["social_network_metrics"] = {
            "total_relationships": total_relationships,
            "average_trust_level": avg_trust_level,
            "active_relationships": len([rel for rel in self.social_relationships.values() if rel.current_status == "active"])
        }
        
        # 冲突解决效率
        resolved_conflicts = len([c for c in self.active_conflicts.values() if c.resolution_status == "resolved"])
        total_conflicts = len(self.active_conflicts)
        analytics["conflict_resolution_efficiency"] = resolved_conflicts / max(1, total_conflicts)
        
        return analytics
    
    def get_agent_collaboration_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """获取智能体协作状态"""
        if agent_id not in self.agent_lifecycle_manager.agents:
            return None
        
        agent = self.agent_lifecycle_manager.agents[agent_id]
        
        # 协作历史
        collaboration_history = []
        for collab_type, metrics in self.collaboration_metrics.items():
            agent_metrics = [m for m in metrics if agent_id in m["agents"]]
            if agent_metrics:
                success_rate = sum(1 for m in agent_metrics if m["success"]) / len(agent_metrics)
                collaboration_history.append({
                    "type": collab_type,
                    "participation_count": len(agent_metrics),
                    "success_rate": success_rate
                })
        
        # 当前协议状态
        current_agreements = [
            agreement for agreement in self.collaboration_agreements.values()
            if agent_id in agreement.participating_agents
        ]
        
        # 社交关系状态
        relationships = []
        if agent_id in self.social_relationships:
            rel = self.social_relationships[agent_id]
            relationships.append({
                "type": rel.relationship_type,
                "trust_level": rel.trust_level,
                "mutual_benefits": len(rel.mutual_benefits),
                "status": rel.current_status
            })
        
        return {
            "agent_id": agent_id,
            "name": agent.name,
            "collaboration_history": collaboration_history,
            "current_agreements": len(current_agreements),
            "active_relationships": len(relationships),
            "social_influence": len(agent.social_connections) if hasattr(agent, 'social_connections') else 0,
            "conflict_participation": len([c for c in self.active_conflicts.values() if agent_id in c.involved_agents])
        }

# 全局协作协议引擎实例
_collaboration_engine_instance = None

def get_collaboration_engine(agent_lifecycle_manager=None):
    """获取协作协议引擎实例"""
    global _collaboration_engine_instance
    if _collaboration_engine_instance is None and agent_lifecycle_manager:
        _collaboration_engine_instance = AgentCollaborationProtocolEngine(agent_lifecycle_manager)
    return _collaboration_engine_instance

if __name__ == "__main__":
    # 测试代码
    async def test_collaboration_protocol():
        print("智能体协作协议引擎 V2 测试")
        print("=" * 50)
        
        # 创建生命周期管理器（模拟）
        from agent_lifecycle_manager_v2 import AgentLifecycleManager, AgentType
        
        lifecycle_manager = AgentLifecycleManager()
        await lifecycle_manager.create_agent("合作小智", AgentType.GENERALIST)
        await lifecycle_manager.create_agent("竞争小能", AgentType.SPECIALIST)
        await lifecycle_manager.create_agent("协调小慧", AgentType.HYBRID)
        
        # 创建协作协议引擎
        collaboration_engine = AgentCollaborationProtocolEngine(lifecycle_manager)
        
        try:
            # 测试合作协议
            print("测试合作协议...")
            cooperation_result = await collaboration_engine.initiate_collaboration(
                CollaborationType.COOPERATION,
                ["generalist_合作小智_1731467558_1234", "specialist_竞争小能_1731467559_5678"],
                objective="开发一个智能协作系统"
            )
            
            if cooperation_result["success"]:
                print(f"✅ 合作协议创建成功，兼容性分数: {cooperation_result['compatibility_score']:.2f}")
                print(f"   协议ID: {cooperation_result['agreement'].agreement_id}")
            else:
                print(f"❌ 合作协议创建失败: {cooperation_result['error']}")
            
            # 测试竞争协议
            print(f"\n测试竞争协议...")
            competition_result = await collaboration_engine.initiate_collaboration(
                CollaborationType.COMPETITION,
                ["generalist_合作小智_1731467558_1234", "specialist_竞争小能_1731467559_5678"],
                competition_type="performance",
                prize="最优算法设计奖"
            )
            
            if competition_result["success"]:
                print(f"✅ 竞争协议启动成功，公平性分数: {competition_result['fairness_score']:.2f}")
                print(f"   竞争ID: {competition_result['competition_id']}")
            else:
                print(f"❌ 竞争协议启动失败: {competition_result['error']}")
            
            # 测试冲突解决
            print(f"\n测试冲突解决...")
            conflict_result = await collaboration_engine.resolve_conflict(
                ["generalist_合作小智_1731467558_1234", "specialist_竞争小能_1731467559_5678"],
                "关于资源分配的分歧",
                "resource_dispute"
            )
            
            if conflict_result["success"]:
                print(f"✅ 冲突解决协议创建成功")
                print(f"   冲突ID: {conflict_result['conflict_id']}")
                print(f"   调解者: {conflict_result['mediator']}")
            else:
                print(f"❌ 冲突解决协议创建失败: {conflict_result['error']}")
            
            # 获取分析数据
            print(f"\n协作分析数据:")
            analytics = await collaboration_engine.get_collaboration_analytics()
            
            print(f"  总协作数: {analytics['total_collaborations']}")
            print(f"  协作类型: {dict(analytics['collaboration_types'])}")
            print(f"  成功率: {dict(analytics['success_rates'])}")
            print(f"  活跃协议: {analytics['active_agreements']}")
            print(f"  活跃竞争: {analytics['active_competitions']}")
            print(f"  活跃冲突: {analytics['active_conflicts']}")
            print(f"  社交关系数: {analytics['social_network_metrics']['total_relationships']}")
            print(f"  平均信任度: {analytics['social_network_metrics']['average_trust_level']:.2f}")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 运行测试
    asyncio.run(test_collaboration_protocol())