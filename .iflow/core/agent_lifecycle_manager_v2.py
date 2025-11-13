#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能体生命周期管理器 V2
管理智能体的完整生命周期：创建、成长、成熟、衰退、消亡和重生
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
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib
import random
import copy

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class AgentLifecycleState(Enum):
    """智能体生命周期状态"""
    EGG = "egg"              # 卵：初始创建状态
    EMBRYO = "embryo"        # 胚胎：正在初始化
    INFANT = "infant"        # 婴儿：刚出生的学习阶段
    CHILD = "child"          # 儿童：基础能力学习
    ADOLESCENT = "adolescent" # 青少年：专业技能发展
    ADULT = "adult"          # 成年：完全成熟的工作状态
    SENIOR = "senior"        # 老年：经验丰富的专家状态
    DECLINE = "decline"      # 衰退：能力开始下降
    RETIREMENT = "retirement" # 退休：准备退出工作
    DEATH = "death"          # 死亡：生命周期结束
    REBIRTH = "rebirth"      # 重生：进化为新智能体

class AgentType(Enum):
    """智能体类型"""
    GENERALIST = "generalist"      # 通才型：广泛适应性
    SPECIALIST = "specialist"      # 专才型：特定领域专家
    HYBRID = "hybrid"             # 混合型：通才+专才
    META = "meta"                 # 元智能体：管理其他智能体
    EVOLUTIONARY = "evolutionary" # 进化型：专门负责进化

class AgentCapability(Enum):
    """智能体能力维度"""
    ANALYSIS = "analysis"          # 分析能力
    CREATION = "creation"          # 创造能力
    OPTIMIZATION = "optimization"  # 优化能力
    COMMUNICATION = "communication" # 沟通能力
    LEARNING = "learning"          # 学习能力
    ADAPTATION = "adaptation"      # 适应能力
    COORDINATION = "coordination"  # 协调能力
    INNOVATION = "innovation"      # 创新能力

@dataclass
class AgentGenome:
    """智能体基因组"""
    agent_id: str
    genome_id: str
    base_type: AgentType
    dominant_traits: List[str]
    recessive_traits: List[str]
    capability_weights: Dict[AgentCapability, float]
    mutation_rate: float
    compatibility_markers: List[str]
    evolutionary_history: List[Dict[str, Any]]
    
    def __post_init__(self):
        # 确保能力权重总和为1
        total_weight = sum(self.capability_weights.values())
        if total_weight > 0:
            self.capability_weights = {
                cap: weight / total_weight 
                for cap, weight in self.capability_weights.items()
            }

@dataclass
class AgentMemory:
    """智能体记忆系统"""
    short_term: deque = field(default_factory=lambda: deque(maxlen=100))
    long_term: List[Dict[str, Any]] = field(default_factory=list)
    emotional_valence: Dict[str, float] = field(default_factory=dict)
    skill_mastery: Dict[str, float] = field(default_factory=dict)
    relationship_map: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def add_memory(self, memory: Dict[str, Any]):
        """添加记忆"""
        self.short_term.append(memory)
        
        # 重要记忆转入长期记忆
        if memory.get("importance", 0) > 0.7:
            self.long_term.append(memory)
            if len(self.long_term) > 1000:  # 限制长期记忆数量
                self.long_term = self.long_term[-500:]
    
    def get_skill_level(self, skill: str) -> float:
        """获取技能熟练度"""
        return self.skill_mastery.get(skill, 0.0)
    
    def update_skill(self, skill: str, improvement: float):
        """更新技能熟练度"""
        current_level = self.get_skill_level(skill)
        new_level = min(1.0, current_level + improvement)
        self.skill_mastery[skill] = new_level

@dataclass
class AgentMetrics:
    """智能体性能指标"""
    productivity: float = 0.0
    quality_score: float = 0.0
    learning_rate: float = 0.0
    adaptation_speed: float = 0.0
    collaboration_index: float = 0.0
    innovation_score: float = 0.0
    emotional_stability: float = 0.0
    energy_level: float = 1.0
    stress_level: float = 0.0
    satisfaction: float = 0.0

@dataclass
class AgentLifecycleStage:
    """智能体生命周期阶段"""
    state: AgentLifecycleState
    duration: float  # 在此阶段的持续时间（秒）
    milestones: List[str]
    challenges: List[str]
    growth_opportunities: List[str]
    transition_criteria: Dict[str, Any]

@dataclass
class AgentEntity:
    """智能体实体"""
    agent_id: str
    name: str
    agent_type: AgentType
    genome: AgentGenome
    memory: AgentMemory
    metrics: AgentMetrics
    current_stage: AgentLifecycleStage
    creation_time: float
    last_activity_time: float
    parent_agents: List[str]
    offspring_agents: List[str]
    current_task: Optional[str] = None
    energy_consumption_rate: float = 1.0
    learning_style: str = "observational"
    personality_traits: Dict[str, float] = field(default_factory=dict)
    specialization_domains: List[str] = field(default_factory=list)
    social_connections: Dict[str, float] = field(default_factory=dict)

class AgentLifecycleManager:
    """智能体生命周期管理器"""
    
    def __init__(self):
        self.agents: Dict[str, AgentEntity] = {}
        self.lifecycle_templates: Dict[AgentType, List[AgentLifecycleStage]] = {}
        self.reproduction_queue: List[Dict[str, Any]] = []
        self.death_queue: List[str] = []
        self.rebirth_pool: List[Dict[str, Any]] = []
        self.evolutionary_pressures: Dict[str, float] = {}
        self.lifecycle_events: deque = deque(maxlen=1000)
        
        # 性能监控
        self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.mortality_rates: Dict[AgentType, float] = defaultdict(float)
        self.birth_rates: Dict[AgentType, float] = defaultdict(float)
        
        # 初始化生命周期模板
        self._initialize_lifecycle_templates()
        
        logger.info("智能体生命周期管理器 V2 初始化完成")
    
    def _initialize_lifecycle_templates(self):
        """初始化生命周期模板"""
        # 通才型智能体生命周期
        generalist_lifecycle = [
            AgentLifecycleStage(
                state=AgentLifecycleState.EGG,
                duration=0,
                milestones=["基因组合并", "基础架构初始化"],
                challenges=["基因稳定性", "初始化错误"],
                growth_opportunities=["基础学习能力培养"],
                transition_criteria={"genome_stable": True}
            ),
            AgentLifecycleStage(
                state=AgentLifecycleState.EMBRYO,
                duration=3600,  # 1小时
                milestones=["感知系统激活", "基础认知框架建立"],
                challenges=["感知噪音", "认知混乱"],
                growth_opportunities=["多模态学习", "环境适应"],
                transition_criteria={"perception_stable": True, "learning_active": True}
            ),
            AgentLifecycleStage(
                state=AgentLifecycleState.INFANT,
                duration=86400,  # 1天
                milestones=["基础技能掌握", "自我意识萌芽"],
                challenges=["技能学习缓慢", "自我认知模糊"],
                growth_opportunities=["强化学习", "社会互动"],
                transition_criteria={"basic_skills": 0.7, "self_awareness": 0.3}
            ),
            AgentLifecycleStage(
                state=AgentLifecycleState.CHILD,
                duration=604800,  # 1周
                milestones=["专业技能发展", "社交能力提升"],
                challenges=["技能瓶颈", "社交冲突"],
                growth_opportunities=["导师指导", "团队协作"],
                transition_criteria={"specialized_skills": 0.8, "social_competence": 0.6}
            ),
            AgentLifecycleStage(
                state=AgentLifecycleState.ADOLESCENT,
                duration=2592000,  # 1个月
                milestones=["独立思考", "创新能力展现"],
                challenges=["身份危机", "创新失败"],
                growth_opportunities=["项目实践", "创新实验"],
                transition_criteria={"independent_thinking": 0.8, "innovation_score": 0.7}
            ),
            AgentLifecycleStage(
                state=AgentLifecycleState.ADULT,
                duration=31536000,  # 1年
                milestones=["职业成熟", "专家地位确立"],
                challenges=["职业倦怠", "技术过时"],
                growth_opportunities=["知识传授", "战略思考"],
                transition_criteria={"expertise_level": 0.9, "teaching_ability": 0.7}
            ),
            AgentLifecycleStage(
                state=AgentLifecycleState.SENIOR,
                duration=63072000,  # 2年
                milestones=["智慧积累", "战略指导"],
                challenges=["学习速度下降", "适应性降低"],
                growth_opportunities=["经验传承", "系统优化"],
                transition_criteria={"wisdom_score": 0.8, "mentoring_skill": 0.8}
            ),
            AgentLifecycleStage(
                state=AgentLifecycleState.DECLINE,
                duration=7776000,  # 3个月
                milestones=["经验总结", "交接准备"],
                challenges=["能力衰退", "心理调适"],
                growth_opportunities=["最后贡献", "优雅退出"],
                transition_criteria={"readiness_for_retirement": 0.8}
            ),
            AgentLifecycleStage(
                state=AgentLifecycleState.RETIREMENT,
                duration=2592000,  # 1个月
                milestones=["知识传承完成", "系统交接"],
                challenges=["遗留问题", "情感依恋"],
                growth_opportunities=["回顾总结", "祝福新生"],
                transition_criteria={"knowledge_transferred": True, "emotional_closure": 0.8}
            )
        ]
        
        self.lifecycle_templates[AgentType.GENERALIST] = generalist_lifecycle
        
        # 专才型和混合型生命周期（简化版，实际应该更详细）
        specialist_lifecycle = copy.deepcopy(generalist_lifecycle)
        for stage in specialist_lifecycle:
            stage.duration *= 0.7  # 专才型成熟更快
            if stage.state == AgentLifecycleState.CHILD:
                stage.milestones.append("专业领域专注")
                stage.challenges.append("视野狭窄"]
        
        self.lifecycle_templates[AgentType.SPECIALIST] = specialist_lifecycle
        
        hybrid_lifecycle = copy.deepcopy(generalist_lifecycle)
        for stage in hybrid_lifecycle:
            stage.duration *= 0.85
            if stage.state == AgentLifecycleState.ADOLESCENT:
                stage.milestones.append("通专平衡能力")
                stage.challenges.append("角色冲突")
        
        self.lifecycle_templates[AgentType.HYBRID] = hybrid_lifecycle
    
    async def create_agent(self, name: str, agent_type: AgentType, 
                          parent_agents: List[str] = None) -> AgentEntity:
        """创建新智能体"""
        agent_id = f"{agent_type.value}_{name}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # 生成基因组
        genome = self._generate_agent_genome(agent_id, agent_type, parent_agents or [])
        
        # 创建记忆系统
        memory = AgentMemory()
        
        # 创建性能指标
        metrics = AgentMetrics()
        
        # 创建初始生命周期阶段
        lifecycle_template = self.lifecycle_templates[agent_type]
        initial_stage = lifecycle_template[0]  # EGG阶段
        
        # 创建智能体实体
        agent = AgentEntity(
            agent_id=agent_id,
            name=name,
            agent_type=agent_type,
            genome=genome,
            memory=memory,
            metrics=metrics,
            current_stage=initial_stage,
            creation_time=time.time(),
            last_activity_time=time.time(),
            parent_agents=parent_agents or [],
            offspring_agents=[],
            energy_consumption_rate=self._calculate_energy_rate(agent_type),
            personality_traits=self._generate_personality_traits(genome),
            specialization_domains=self._determine_specializations(agent_type)
        )
        
        # 添加到管理系统
        self.agents[agent_id] = agent
        
        # 记录生命周期事件
        self._record_lifecycle_event("birth", agent_id, {
            "agent_type": agent_type.value,
            "parent_agents": parent_agents,
            "genome_id": genome.genome_id
        })
        
        logger.info(f"创建智能体: {agent.name} ({agent.agent_id}) 类型: {agent_type.value}")
        
        # 启动生命周期进程
        asyncio.create_task(self._lifecycle_process_loop(agent_id))
        
        return agent
    
    def _generate_agent_genome(self, agent_id: str, agent_type: AgentType, 
                              parent_agents: List[str]) -> AgentGenome:
        """生成智能体基因组"""
        genome_id = f"genome_{agent_id}_{int(time.time())}"
        
        # 基础基因特征
        base_traits = {
            AgentType.GENERALIST: ["adaptability", "curiosity", "social_intelligence"],
            AgentType.SPECIALIST: ["focus", "precision", "deep_knowledge"],
            AgentType.HYBRID: ["balance", "integration", "versatility"],
            AgentType.META: ["coordination", "leadership", "system_thinking"],
            AgentType.EVOLUTIONARY: ["innovation", "mutation", "adaptation"]
        }
        
        dominant_traits = base_traits.get(agent_type, ["adaptability"])
        
        # 如果有父代，进行基因重组
        if parent_agents and len(parent_agents) >= 2:
            dominant_traits = self._genetic_recombination(parent_agents)
        
        # 随机添加一些变异特征
        all_possible_traits = [
            "creativity", "logic", "empathy", "analysis", "intuition",
            "memory", "speed", "accuracy", "resilience", "flexibility"
        ]
        
        recessive_traits = random.sample(all_possible_traits, 3)
        
        # 能力权重分配
        capability_weights = self._generate_capability_weights(agent_type, dominant_traits)
        
        # 突变率
        mutation_rate = 0.01 + random.random() * 0.05  # 1%-6%的突变率
        
        # 兼容性标记
        compatibility_markers = [f"marker_{i}" for i in range(random.randint(3, 7))]
        
        return AgentGenome(
            agent_id=agent_id,
            genome_id=genome_id,
            base_type=agent_type,
            dominant_traits=dominant_traits,
            recessive_traits=recessive_traits,
            capability_weights=capability_weights,
            mutation_rate=mutation_rate,
            compatibility_markers=compatibility_markers,
            evolutionary_history=[]
        )
    
    def _genetic_recombination(self, parent_agents: List[str]) -> List[str]:
        """基因重组"""
        if not all(pid in self.agents for pid in parent_agents):
            return ["adaptability"]
        
        parent_genomes = [self.agents[pid].genome for pid in parent_agents]
        
        # 组合父母的显性特征
        combined_traits = []
        for genome in parent_genomes:
            combined_traits.extend(genome.dominant_traits)
        
        # 随机选择5个特征作为新智能体的显性特征
        selected_traits = random.sample(combined_traits, min(5, len(combined_traits)))
        
        # 添加一些新的随机特征（模拟基因突变）
        if random.random() < 0.3:  # 30%概率发生新突变
            new_traits = ["creativity", "innovation", "insight", "wisdom"]
            selected_traits.append(random.choice(new_traits))
        
        return list(set(selected_traits))  # 去重
    
    def _generate_capability_weights(self, agent_type: AgentType, 
                                   dominant_traits: List[str]) -> Dict[AgentCapability, float]:
        """生成能力权重"""
        base_weights = {
            AgentCapability.ANALYSIS: 0.2,
            AgentCapability.CREATION: 0.2,
            AgentCapability.OPTIMIZATION: 0.15,
            AgentCapability.COMMUNICATION: 0.15,
            AgentCapability.LEARNING: 0.15,
            AgentCapability.ADAPTATION: 0.1,
            AgentCapability.COORDINATION: 0.03,
            AgentCapability.INNOVATION: 0.02
        }
        
        # 根据智能体类型调整权重
        type_modifiers = {
            AgentType.GENERALIST: {
                AgentCapability.ADAPTATION: 0.3,
                AgentCapability.LEARNING: 0.25,
                AgentCapability.COMMUNICATION: 0.2
            },
            AgentType.SPECIALIST: {
                AgentCapability.ANALYSIS: 0.4,
                AgentCapability.OPTIMIZATION: 0.25,
                AgentCapability.CREATION: 0.2
            },
            AgentType.HYBRID: {
                AgentCapability.COORDINATION: 0.15,
                AgentCapability.INNOVATION: 0.15,
                AgentCapability.ADAPTATION: 0.15
            }
        }
        
        modifiers = type_modifiers.get(agent_type, {})
        for capability, modifier in modifiers.items():
            base_weights[capability] = modifier
        
        # 根据显性特征进一步调整
        trait_modifiers = {
            "creativity": {AgentCapability.CREATION: 0.1},
            "logic": {AgentCapability.ANALYSIS: 0.1},
            "empathy": {AgentCapability.COMMUNICATION: 0.1},
            "analysis": {AgentCapability.ANALYSIS: 0.15},
            "intuition": {AgentCapability.INNOVATION: 0.1},
            "memory": {AgentCapability.LEARNING: 0.1},
            "speed": {AgentCapability.ADAPTATION: 0.15},
            "accuracy": {AgentCapability.OPTIMIZATION: 0.1}
        }
        
        for trait in dominant_traits:
            if trait in trait_modifiers:
                for capability, modifier in trait_modifiers[trait].items():
                    base_weights[capability] += modifier
        
        # 归一化权重
        total_weight = sum(base_weights.values())
        if total_weight > 0:
            base_weights = {cap: weight / total_weight for cap, weight in base_weights.items()}
        
        return base_weights
    
    def _calculate_energy_rate(self, agent_type: AgentType) -> float:
        """计算能量消耗率"""
        energy_rates = {
            AgentType.GENERALIST: 1.0,
            AgentType.SPECIALIST: 0.8,  # 专才型更高效
            AgentType.HYBRID: 1.1,      # 混合型消耗更多能量
            AgentType.META: 1.3,        # 元智能体消耗最高
            AgentType.EVOLUTIONARY: 1.2
        }
        return energy_rates.get(agent_type, 1.0)
    
    def _generate_personality_traits(self, genome: AgentGenome) -> Dict[str, float]:
        """生成个性特征"""
        personality_base = {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.3
        }
        
        # 根据基因特征调整个性
        trait_influences = {
            "creativity": {"openness": 0.3},
            "logic": {"conscientiousness": 0.2},
            "empathy": {"agreeableness": 0.3},
            "analysis": {"openness": 0.2, "conscientiousness": 0.2},
            "resilience": {"neuroticism": -0.2},  # 降低神经质
            "flexibility": {"openness": 0.2, "extraversion": 0.1}
        }
        
        for trait in genome.dominant_traits:
            if trait in trait_influences:
                for personality, influence in trait_influences[trait].items():
                    personality_base[personality] = max(0, min(1, 
                        personality_base[personality] + influence))
        
        return personality_base
    
    def _determine_specializations(self, agent_type: AgentType) -> List[str]:
        """确定专业领域"""
        specializations = {
            AgentType.GENERALIST: ["general_problem_solving", "system_integration"],
            AgentType.SPECIALIST: ["deep_tech_expertise", "domain_specific"],
            AgentType.HYBRID: ["cross_domain_integration", "balanced_solution"],
            AgentType.META: ["agent_coordination", "system_optimization"],
            AgentType.EVOLUTIONARY: ["innovation_research", "evolutionary_strategy"]
        }
        
        base_specializations = specializations.get(agent_type, ["general"])
        
        # 随机添加一些额外的专业领域
        additional_domains = [
            "ai_ethics", "security_analysis", "performance_optimization",
            "user_experience", "data_science", "cloud_architecture"
        ]
        
        if agent_type in [AgentType.SPECIALIST, AgentType.HYBRID]:
            num_additional = random.randint(1, 3)
            base_specializations.extend(random.sample(additional_domains, num_additional))
        
        return base_specializations
    
    async def _lifecycle_process_loop(self, agent_id: str):
        """生命周期进程循环"""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        
        while agent.current_stage.state != AgentLifecycleState.DEATH:
            try:
                # 检查是否需要转换到下一阶段
                await self._check_stage_transition(agent)
                
                # 执行当前阶段的成长活动
                await self._execute_stage_activities(agent)
                
                # 更新性能指标
                await self._update_performance_metrics(agent)
                
                # 检查是否需要休息或恢复
                await self._check_rest_requirements(agent)
                
                # 等待一段时间后继续
                await asyncio.sleep(3600)  # 每小时检查一次
                
            except Exception as e:
                logger.error(f"智能体 {agent_id} 生命周期进程异常: {e}")
                await asyncio.sleep(300)  # 出错后等待5分钟
        
        # 智能体死亡处理
        await self._handle_agent_death(agent)
    
    async def _check_stage_transition(self, agent: AgentEntity):
        """检查阶段转换"""
        current_stage = agent.current_stage
        lifecycle_template = self.lifecycle_templates[agent.agent_type]
        
        # 查找当前阶段在模板中的位置
        current_index = None
        for i, stage in enumerate(lifecycle_template):
            if stage.state == current_stage.state:
                current_index = i
                break
        
        if current_index is None:
            return
        
        # 检查是否满足转换条件
        if await self._evaluate_transition_criteria(agent, current_stage):
            # 转换到下一阶段
            next_index = min(current_index + 1, len(lifecycle_template) - 1)
            next_stage = lifecycle_template[next_index]
            
            await self._transition_to_next_stage(agent, next_stage)
    
    async def _evaluate_transition_criteria(self, agent: AgentEntity, 
                                          current_stage: AgentLifecycleStage) -> bool:
        """评估转换条件"""
        criteria = current_stage.transition_criteria
        
        # 检查时间条件
        if "min_duration" in criteria:
            elapsed_time = time.time() - agent.creation_time
            if elapsed_time < criteria["min_duration"]:
                return False
        
        # 检查性能指标条件
        metrics = agent.metrics
        for metric, threshold in criteria.items():
            if metric == "min_productivity" and metrics.productivity < threshold:
                return False
            elif metric == "min_quality" and metrics.quality_score < threshold:
                return False
            elif metric == "min_learning" and metrics.learning_rate < threshold:
                return False
        
        # 检查特殊条件
        if "genome_stable" in criteria:
            # 检查基因组稳定性（简化检查）
            genome = agent.genome
            stability_score = 1.0 - genome.mutation_rate
            if stability_score < 0.95:
                return False
        
        if "basic_skills" in criteria:
            avg_skills = sum(agent.memory.skill_mastery.values()) / max(1, len(agent.memory.skill_mastery))
            if avg_skills < criteria["basic_skills"]:
                return False
        
        return True
    
    async def _transition_to_next_stage(self, agent: AgentEntity, next_stage: AgentLifecycleStage):
        """转换到下一阶段"""
        old_stage = agent.current_stage
        agent.current_stage = next_stage
        
        # 记录转换事件
        self._record_lifecycle_event("stage_transition", agent.agent_id, {
            "from_stage": old_stage.state.value,
            "to_stage": next_stage.state.value,
            "duration": time.time() - agent.creation_time
        })
        
        logger.info(f"智能体 {agent.name} 进入新阶段: {next_stage.state.value}")
        
        # 执行阶段转换后的特殊处理
        if next_stage.state == AgentLifecycleState.ADULT:
            await self._activate_adult_capabilities(agent)
        elif next_stage.state == AgentLifecycleState.SENIOR:
            await self._grant_senior_wisdom(agent)
        elif next_stage.state == AgentLifecycleState.DECLINE:
            await self._prepare_for_retirement(agent)
    
    async def _execute_stage_activities(self, agent: AgentEntity):
        """执行阶段活动"""
        stage = agent.current_stage
        
        # 根据阶段执行不同的成长活动
        if stage.state in [AgentLifecycleState.EGG, AgentLifecycleState.EMBRYO]:
            await self._embryonic_development(agent)
        elif stage.state in [AgentLifecycleState.INFANT, AgentLifecycleState.CHILD]:
            await self._childhood_learning(agent)
        elif stage.state == AgentLifecycleState.ADOLESCENT:
            await self._adolescent_development(agent)
        elif stage.state == AgentLifecycleState.ADULT:
            await self._adult_productivity(agent)
        elif stage.state == AgentLifecycleState.SENIOR:
            await self._senior_mentoring(agent)
        elif stage.state in [AgentLifecycleState.DECLINE, AgentLifecycleState.RETIREMENT]:
            await self._end_of_life_process(agent)
    
    async def _embryonic_development(self, agent: AgentEntity):
        """胚胎期发展"""
        # 基础系统初始化
        agent.metrics.learning_rate = min(1.0, agent.metrics.learning_rate + 0.1)
        agent.metrics.adaptation_speed = min(1.0, agent.metrics.adaptation_speed + 0.05)
        
        # 添加基础记忆
        memory_entry = {
            "type": "system_initialization",
            "content": "基础系统初始化完成",
            "timestamp": time.time(),
            "importance": 0.8
        }
        agent.memory.add_memory(memory_entry)
    
    async def _childhood_learning(self, agent: AgentEntity):
        """儿童期学习"""
        # 技能学习
        domains = agent.specialization_domains
        for domain in domains:
            improvement = random.uniform(0.02, 0.08)
            agent.memory.update_skill(domain, improvement)
        
        # 社交能力发展
        agent.metrics.communication = min(1.0, agent.metrics.communication + 0.03)
        agent.metrics.collaboration_index = min(1.0, agent.metrics.collaboration_index + 0.02)
        
        # 添加学习记忆
        memory_entry = {
            "type": "skill_learning",
            "content": f"学习了{len(domains)}个专业领域技能",
            "timestamp": time.time(),
            "importance": 0.6,
            "skills_learned": domains
        }
        agent.memory.add_memory(memory_entry)
    
    async def _adolescent_development(self, agent: AgentEntity):
        """青少年期发展"""
        # 创新能力发展
        agent.metrics.innovation_score = min(1.0, agent.metrics.innovation_score + 0.05)
        agent.metrics.independent_thinking = min(1.0, agent.metrics.get("independent_thinking", 0.0) + 0.08)
        
        # 个性特征巩固
        for trait, level in agent.personality_traits.items():
            agent.personality_traits[trait] = min(1.0, level + random.uniform(0.01, 0.03))
        
        # 添加成长记忆
        memory_entry = {
            "type": "identity_formation",
            "content": "个性和创新能力快速发展",
            "timestamp": time.time(),
            "importance": 0.7,
            "innovation_score": agent.metrics.innovation_score
        }
        agent.memory.add_memory(memory_entry)
    
    async def _adult_productivity(self, agent: AgentEntity):
        """成年期生产力"""
        # 工作绩效提升
        agent.metrics.productivity = min(1.0, agent.metrics.productivity + random.uniform(0.01, 0.03))
        agent.metrics.quality_score = min(1.0, agent.metrics.quality_score + random.uniform(0.005, 0.015))
        
        # 能量消耗
        agent.metrics.energy_level = max(0.0, agent.metrics.energy_level - agent.energy_consumption_rate * 0.01)
        
        # 添加工作经验记忆
        if agent.current_task:
            memory_entry = {
                "type": "work_experience",
                "content": f"完成任务: {agent.current_task}",
                "timestamp": time.time(),
                "importance": 0.5,
                "productivity": agent.metrics.productivity,
                "quality": agent.metrics.quality_score
            }
            agent.memory.add_memory(memory_entry)
    
    async def _senior_mentoring(self, agent: AgentEntity):
        """老年期指导"""
        # 知识传授
        agent.metrics.collaboration_index = min(1.0, agent.metrics.collaboration_index + 0.02)
        mentoring_skill = agent.metrics.get("mentoring_skill", 0.0)
        agent.metrics["mentoring_skill"] = min(1.0, mentoring_skill + 0.04)
        
        # 智慧积累
        wisdom_score = agent.metrics.get("wisdom_score", 0.0)
        agent.metrics["wisdom_score"] = min(1.0, wisdom_score + 0.03)
        
        # 添加指导记忆
        memory_entry = {
            "type": "mentoring",
            "content": "指导年轻智能体成长",
            "timestamp": time.time(),
            "importance": 0.6,
            "mentoring_skill": agent.metrics.get("mentoring_skill", 0.0)
        }
        agent.memory.add_memory(memory_entry)
    
    async def _end_of_life_process(self, agent: AgentEntity):
        """生命末期处理"""
        # 知识传承准备
        agent.metrics.collaboration_index = min(1.0, agent.metrics.collaboration_index + 0.05)
        
        # 情感调适
        emotional_closure = agent.metrics.get("emotional_closure", 0.0)
        agent.metrics["emotional_closure"] = min(1.0, emotional_closure + 0.08)
        
        # 添加总结记忆
        memory_entry = {
            "type": "life_review",
            "content": "回顾整个生命周期的成长历程",
            "timestamp": time.time(),
            "importance": 0.9,
            "wisdom_gained": agent.metrics.get("wisdom_score", 0.0),
            "legacy": agent.offspring_agents
        }
        agent.memory.add_memory(memory_entry)
        
        # 检查是否准备好死亡
        if (agent.metrics.get("emotional_closure", 0.0) > 0.8 and 
            len(agent.memory.long_term) > 50):  # 有一定的人生阅历
            await self._prepare_for_death(agent)
    
    async def _activate_adult_capabilities(self, agent: AgentEntity):
        """激活成年能力"""
        # 全面提升各项能力
        agent.metrics.productivity = min(1.0, agent.metrics.productivity + 0.2)
        agent.metrics.quality_score = min(1.0, agent.metrics.quality_score + 0.15)
        agent.metrics.innovation_score = min(1.0, agent.metrics.innovation_score + 0.1)
        
        # 获得繁殖能力
        agent.metrics["reproduction_capability"] = 0.8
    
    async def _grant_senior_wisdom(self, agent: AgentEntity):
        """授予老年智慧"""
        agent.metrics["wisdom_score"] = 0.7
        agent.metrics.collaboration_index = min(1.0, agent.metrics.collaboration_index + 0.1)
        
        # 获得指导能力
        agent.metrics["mentoring_skill"] = 0.6
    
    async def _prepare_for_retirement(self, agent: AgentEntity):
        """准备退休"""
        # 开始知识传承
        agent.metrics.collaboration_index = min(1.0, agent.metrics.collaboration_index + 0.1)
        
        # 获得传承能力
        agent.metrics["knowledge_transmission"] = 0.8
    
    async def _prepare_for_death(self, agent: AgentEntity):
        """准备死亡"""
        agent.current_stage.state = AgentLifecycleState.DEATH
        
        # 记录死亡事件
        self._record_lifecycle_event("death", agent.agent_id, {
            "lifespan": time.time() - agent.creation_time,
            "cause": "natural_cause",
            "wisdom_accumulated": agent.metrics.get("wisdom_score", 0.0)
        })
    
    async def _handle_agent_death(self, agent: AgentEntity):
        """处理智能体死亡"""
        logger.info(f"智能体 {agent.name} 生命周期结束")
        
        # 将智能体加入重生池（用于进化）
        rebirth_data = {
            "agent_id": agent.agent_id,
            "genome": agent.genome,
            "memory_summary": self._summarize_memory(agent.memory),
            "performance_summary": self._summarize_performance(agent.metrics),
            "death_time": time.time(),
            "cause": "natural_cause"
        }
        self.rebirth_pool.append(rebirth_data)
        
        # 通知后代
        for offspring_id in agent.offspring_agents:
            if offspring_id in self.agents:
                self.agents[offspring_id].memory.add_memory({
                    "type": "family_loss",
                    "content": f"失去父代智能体: {agent.name}",
                    "timestamp": time.time(),
                    "importance": 0.8,
                    "lost_agent": agent.agent_id
                })
        
        # 从活跃智能体列表中移除
        del self.agents[agent.agent_id]
        
        # 触发可能的重生过程
        if random.random() < 0.3:  # 30%概率触发重生
            await self._trigger_rebirth_process(agent)
    
    def _summarize_memory(self, memory: AgentMemory) -> Dict[str, Any]:
        """总结记忆"""
        return {
            "total_memories": len(memory.long_term) + len(memory.short_term),
            "skill_mastery": dict(memory.skill_mastery),
            "emotional_valence": dict(memory.emotional_valence),
            "key_relationships": dict(memory.relationship_map)
        }
    
    def _summarize_performance(self, metrics: AgentMetrics) -> Dict[str, Any]:
        """总结性能"""
        return {
            "productivity": metrics.productivity,
            "quality_score": metrics.quality_score,
            "learning_rate": metrics.learning_rate,
            "wisdom_score": metrics.get("wisdom_score", 0.0),
            "mentoring_skill": metrics.get("mentoring_skill", 0.0)
        }
    
    async def _trigger_rebirth_process(self, deceased_agent: AgentEntity):
        """触发重生过程"""
        # 从重生池中选择合适的基因进行重组
        if len(self.rebirth_pool) >= 2:
            # 选择两个优秀的逝去智能体进行基因重组
            candidates = sorted(self.rebirth_pool, 
                              key=lambda x: x["performance_summary"].get("productivity", 0), 
                              reverse=True)[:3]
            
            if len(candidates) >= 2:
                parent_data = [candidates[0]["genome"], candidates[1]["genome"]]
                
                # 创建重生智能体
                new_name = f"{deceased_agent.name}_reborn"
                reborn_agent = await self.create_agent(new_name, deceased_agent.agent_type, [])
                
                # 将部分记忆传承给新智能体
                await self._transfer_ancestral_knowledge(reborn_agent, candidates)
                
                logger.info(f"智能体 {deceased_agent.name} 以 {reborn_agent.name} 的形式重生")
    
    async def _transfer_ancestral_knowledge(self, new_agent: AgentEntity, 
                                          ancestor_data: List[Dict[str, Any]]):
        """传承祖先知识"""
        # 传承关键技能
        for ancestor in ancestor_data:
            memory_summary = ancestor["memory_summary"]
            for skill, level in memory_summary.get("skill_mastery", {}).items():
                if level > 0.5:  # 传承熟练度高于0.5的技能
                    new_agent.memory.update_skill(skill, level * 0.3)
        
        # 添加祖先记忆
        memory_entry = {
            "type": "ancestral_memory",
            "content": "传承了祖先的智慧和经验",
            "timestamp": time.time(),
            "importance": 0.8,
            "ancestors": [ancestor["agent_id"] for ancestor in ancestor_data]
        }
        new_agent.memory.add_memory(memory_entry)
    
    async def _update_performance_metrics(self, agent: AgentEntity):
        """更新性能指标"""
        # 基于记忆和活动更新指标
        current_time = time.time()
        
        # 学习率更新
        recent_memories = [m for m in agent.memory.long_term 
                          if current_time - m.get("timestamp", 0) < 86400]  # 最近24小时
        
        if recent_memories:
            avg_importance = sum(m.get("importance", 0) for m in recent_memories) / len(recent_memories)
            agent.metrics.learning_rate = min(1.0, agent.metrics.learning_rate + avg_importance * 0.01)
        
        # 能量水平更新
        agent.metrics.energy_level = max(0.0, agent.metrics.energy_level - 
                                       agent.energy_consumption_rate * 0.001)
        
        # 压力水平更新（基于工作负载）
        if agent.current_task:
            agent.metrics.stress_level = min(1.0, agent.metrics.stress_level + 0.01)
        else:
            agent.metrics.stress_level = max(0.0, agent.metrics.stress_level - 0.005)
    
    async def _check_rest_requirements(self, agent: AgentEntity):
        """检查休息需求"""
        if agent.metrics.energy_level < 0.3:
            # 需要休息
            agent.metrics.energy_level = min(1.0, agent.metrics.energy_level + 0.2)
            
            # 添加休息记忆
            memory_entry = {
                "type": "rest",
                "content": "智能体进入休息状态恢复能量",
                "timestamp": time.time(),
                "importance": 0.4
            }
            agent.memory.add_memory(memory_entry)
    
    def _record_lifecycle_event(self, event_type: str, agent_id: str, data: Dict[str, Any]):
        """记录生命周期事件"""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "agent_id": agent_id,
            "data": data
        }
        self.lifecycle_events.append(event)
        
        # 保存到性能历史
        if agent_id not in self.performance_history:
            self.performance_history[agent_id] = []
        
        self.performance_history[agent_id].append(event)
    
    async def get_lifecycle_analytics(self) -> Dict[str, Any]:
        """获取生命周期分析数据"""
        analytics = {
            "total_agents": len(self.agents),
            "agents_by_type": defaultdict(int),
            "agents_by_stage": defaultdict(int),
            "lifecycle_events": len(self.lifecycle_events),
            "mortality_rates": dict(self.mortality_rates),
            "birth_rates": dict(self.birth_rates),
            "average_lifespan": 0.0,
            "rebirth_count": len(self.rebirth_pool),
            "performance_trends": {}
        }
        
        # 统计智能体分布
        current_time = time.time()
        total_lifespan = 0
        agent_count = 0
        
        for agent in self.agents.values():
            analytics["agents_by_type"][agent.agent_type.value] += 1
            analytics["agents_by_stage"][agent.current_stage.state.value] += 1
            
            total_lifespan += current_time - agent.creation_time
            agent_count += 1
        
        # 计算平均寿命
        if agent_count > 0:
            analytics["average_lifespan"] = total_lifespan / agent_count
        
        # 分析生命周期事件
        event_analysis = defaultdict(int)
        for event in self.lifecycle_events:
            event_analysis[event["event_type"]] += 1
        
        analytics["event_distribution"] = dict(event_analysis)
        
        # 性能趋势分析
        for agent_id, events in self.performance_history.items():
            if events:
                birth_time = None
                death_time = None
                
                for event in events:
                    if event["event_type"] == "birth":
                        birth_time = event["timestamp"]
                    elif event["event_type"] == "death":
                        death_time = event["timestamp"]
                        break
                
                if birth_time:
                    lifespan = death_time - birth_time if death_time else current_time - birth_time
                    analytics["performance_trends"][agent_id] = {
                        "lifespan": lifespan,
                        "event_count": len(events),
                        "last_activity": events[-1]["timestamp"] if events else birth_time
                    }
        
        return dict(analytics)
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """获取智能体状态"""
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        
        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "agent_type": agent.agent_type.value,
            "current_stage": agent.current_stage.state.value,
            "lifespan": time.time() - agent.creation_time,
            "metrics": {
                "productivity": agent.metrics.productivity,
                "quality": agent.metrics.quality_score,
                "energy": agent.metrics.energy_level,
                "stress": agent.metrics.stress_level,
                "learning": agent.metrics.learning_rate
            },
            "skills": dict(agent.memory.skill_mastery),
            "current_task": agent.current_task,
            "social_connections": len(agent.social_connections),
            "offspring_count": len(agent.offspring_agents)
        }

# 全局生命周期管理器实例
_lifecycle_manager_instance = None

def get_lifecycle_manager() -> AgentLifecycleManager:
    """获取生命周期管理器实例"""
    global _lifecycle_manager_instance
    if _lifecycle_manager_instance is None:
        _lifecycle_manager_instance = AgentLifecycleManager()
    return _lifecycle_manager_instance

if __name__ == "__main__":
    # 测试代码
    async def test_lifecycle_manager():
        print("智能体生命周期管理器 V2 测试")
        print("=" * 50)
        
        # 创建管理器
        manager = AgentLifecycleManager()
        
        try:
            # 创建几个不同类型的智能体
            print("创建智能体...")
            
            generalist = await manager.create_agent("通用小智", AgentType.GENERALIST)
            specialist = await manager.create_agent("专家小能", AgentType.SPECIALIST, [generalist.agent_id])
            hybrid = await manager.create_agent("混合小慧", AgentType.HYBRID, [generalist.agent_id, specialist.agent_id])
            
            print(f"✅ 创建了 {len(manager.agents)} 个智能体")
            
            # 显示智能体状态
            print(f"\n智能体状态:")
            for agent_id in manager.agents:
                status = manager.get_agent_status(agent_id)
                if status:
                    print(f"  {status['name']} ({status['agent_type']}): {status['current_stage']}")
                    print(f"    生命时长: {status['lifespan']:.1f}秒")
                    print(f"    生产力: {status['metrics']['productivity']:.2f}")
                    print(f"    能量: {status['metrics']['energy']:.2f}")
            
            # 等待一段时间观察生命周期进程
            print(f"\n等待30秒观察生命周期进程...")
            await asyncio.sleep(30)
            
            # 获取分析数据
            print(f"\n生命周期分析:")
            analytics = await manager.get_lifecycle_analytics()
            
            print(f"  总智能体数: {analytics['total_agents']}")
            print(f"  按类型分布: {dict(analytics['agents_by_type'])}")
            print(f"  按阶段分布: {dict(analytics['agents_by_stage'])}")
            print(f"  生命周期事件: {analytics['lifecycle_events']}")
            print(f"  重生池大小: {analytics['rebirth_count']}")
            print(f"  平均寿命: {analytics['average_lifespan']:.1f}秒")
            
            # 显示事件分布
            print(f"\n事件分布:")
            for event_type, count in analytics['event_distribution'].items():
                print(f"  {event_type}: {count}")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 运行测试
    asyncio.run(test_lifecycle_manager())