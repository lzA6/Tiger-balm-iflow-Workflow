#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
进化算法引擎 V2
实现智能体的自然选择、适者生存和物种进化
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import logging
import asyncio
import uuid
import time
import math
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib
import copy
from functools import partial

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class EvolutionaryEra(Enum):
    """进化时代"""
    PREBIRTH = "prebirth"              # 前生命：基因池准备
    BIRTH = "birth"                    # 生命诞生：新智能体创建
    ADAPTATION = "adaptation"          # 适应期：环境适应和学习
    COMPETITION = "competition"        # 竞争期：资源竞争和选择
    REPRODUCTION = "reproduction"      # 繁殖期：基因重组和繁衍
    SELECTION = "selection"            # 选择期：优胜劣汰
    EXTINCTION = "extinction"          # 灭绝期：物种更新
    RENAISSANCE = "renaissance"        # 文艺复兴：创新爆发

class SelectionPressure(Enum):
    """选择压力"""
    ENVIRONMENTAL = "environmental"    # 环境压力
    COMPETITIVE = "competitive"        # 竞争压力
    RESOURCE = "resource"              # 资源压力
    SOCIAL = "social"                  # 社会压力
    INNOVATIVE = "innovative"          # 创新压力

class GeneticMutationType(Enum):
    """基因突变类型"""
    POINT_MUTATION = "point_mutation"  # 点突变：单个基因位点变化
    INSERTION = "insertion"            # 插入突变：基因序列插入
    DELETION = "deletion"              # 缺失突变：基因序列删除
    DUPLICATION = "duplication"        # 重复突变：基因序列复制
    INVERSION = "inversion"            # 倒位突变：基因序列反转
    TRANSLOCATION = "translocation"    # 易位突变：基因序列移动

class EvolutionaryFitness(Enum):
    """进化适应度"""
    SURVIVAL = "survival"              # 生存能力
    REPRODUCTION = "reproduction"      # 繁殖能力
    ADAPTATION = "adaptation"          # 适应能力
    INNOVATION = "innovation"          # 创新能力
    COLLABORATION = "collaboration"    # 协作能力
    EFFICIENCY = "efficiency"          # 效率能力

@dataclass
class EvolutionaryGene:
    """进化基因"""
    gene_id: str
    gene_name: str
    gene_type: str
    value: Union[float, int, str, bool]
    mutable: bool
    epigenetic_markers: Dict[str, float] = field(default_factory=dict)
    interaction_partners: List[str] = field(default_factory=list)
    mutation_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class EvolutionaryChromosome:
    """进化染色体"""
    chromosome_id: str
    genes: Dict[str, EvolutionaryGene]
    length: int
    recombination_rate: float
    mutation_rate: float
    epigenetic_state: Dict[str, float] = field(default_factory=dict)

@dataclass
class EvolutionaryIndividual:
    """进化个体"""
    individual_id: str
    genome: Dict[str, EvolutionaryChromosome]
    phenotype: Dict[str, Any]
    fitness_scores: Dict[EvolutionaryFitness, float]
    evolutionary_age: int
    generation: int
    parent_ids: List[str]
    offspring_ids: List[str]
    mutation_count: int = 0
    epigenetic_modifications: int = 0
    evolutionary_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class EvolutionaryPopulation:
    """进化种群"""
    population_id: str
    individuals: Dict[str, EvolutionaryIndividual]
    population_size: int
    genetic_diversity: float
    average_fitness: float
    evolutionary_pressure: Dict[SelectionPressure, float]
    carrying_capacity: int
    migration_rate: float
    speciation_threshold: float
    population_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class EvolutionaryEvent:
    """进化事件"""
    event_id: str
    event_type: str
    timestamp: float
    affected_population: str
    magnitude: float
    evolutionary_impact: Dict[str, Any]
    trigger_conditions: List[str]
    expected_outcomes: List[str]
    actual_outcomes: List[str] = field(default_factory=list)

class BaseEvolutionaryOperator(ABC):
    """基础进化算子抽象类"""
    
    def __init__(self, engine):
        self.engine = engine
        self.operator_name = self.__class__.__name__
    
    @abstractmethod
    async def apply(self, population: EvolutionaryPopulation, **kwargs) -> EvolutionaryPopulation:
        """应用进化算子"""
        pass
    
    @abstractmethod
    def can_apply(self, population: EvolutionaryPopulation, **kwargs) -> bool:
        """判断是否可以应用"""
        pass

class NaturalSelectionOperator(BaseEvolutionaryOperator):
    """自然选择算子"""
    
    def can_apply(self, population: EvolutionaryPopulation, **kwargs) -> bool:
        return len(population.individuals) > 10  # 种群足够大才进行选择
    
    async def apply(self, population: EvolutionaryPopulation, 
                   selection_pressure: SelectionPressure = SelectionPressure.COMPETITIVE,
                   selection_intensity: float = 0.3) -> EvolutionaryPopulation:
        """应用自然选择"""
        logger.info(f"应用自然选择算子，压力类型: {selection_pressure.value}")
        
        # 计算适应度阈值
        fitness_threshold = self._calculate_fitness_threshold(population, selection_intensity)
        
        # 评估每个个体的生存能力
        survival_decisions = await self._evaluate_survival(population, fitness_threshold, selection_pressure)
        
        # 移除不适应的个体
        survivors = {}
        deceased = []
        
        for individual_id, decision in survival_decisions.items():
            if decision["survive"]:
                survivors[individual_id] = population.individuals[individual_id]
            else:
                deceased.append({
                    "individual_id": individual_id,
                    "cause": decision["cause"],
                    "fitness_score": decision["fitness_score"]
                })
        
        # 更新种群
        new_population = copy.deepcopy(population)
        new_population.individuals = survivors
        
        # 记录进化事件
        selection_event = EvolutionaryEvent(
            event_id=f"selection_{int(time.time())}_{random.randint(1000, 9999)}",
            event_type="natural_selection",
            timestamp=time.time(),
            affected_population=population.population_id,
            magnitude=selection_intensity,
            evolutionary_impact={
                "original_size": len(population.individuals),
                "survivors": len(survivors),
                "mortality_rate": len(deceased) / len(population.individuals),
                "selection_pressure": selection_pressure.value,
                "fitness_threshold": fitness_threshold
            },
            trigger_conditions=[f"selection_pressure_{selection_pressure.value}"],
            expected_outcomes=["population_optimization", "fitness_improvement"],
            actual_outcomes=deceased
        )
        
        new_population.population_history.append(selection_event)
        
        # 更新种群统计
        new_population.population_size = len(survivors)
        new_population.average_fitness = self._calculate_average_fitness(survivors)
        new_population.genetic_diversity = self._calculate_genetic_diversity(survivors)
        
        logger.info(f"自然选择完成：{len(population.individuals)} -> {len(survivors)} 个个体")
        
        return new_population
    
    def _calculate_fitness_threshold(self, population: EvolutionaryPopulation, 
                                   selection_intensity: float) -> float:
        """计算适应度阈值"""
        fitness_scores = []
        for individual in population.individuals.values():
            avg_fitness = sum(individual.fitness_scores.values()) / len(individual.fitness_scores)
            fitness_scores.append(avg_fitness)
        
        fitness_scores.sort()
        
        # 基于选择强度确定阈值位置
        threshold_index = int(len(fitness_scores) * (1.0 - selection_intensity))
        threshold_index = max(0, min(threshold_index, len(fitness_scores) - 1))
        
        return fitness_scores[threshold_index]
    
    async def _evaluate_survival(self, population: EvolutionaryPopulation, 
                              fitness_threshold: float, 
                              selection_pressure: SelectionPressure) -> Dict[str, Dict[str, Any]]:
        """评估生存能力"""
        survival_decisions = {}
        
        for individual_id, individual in population.individuals.items():
            # 基础适应度评估
            base_fitness = sum(individual.fitness_scores.values()) / len(individual.fitness_scores)
            
            # 压力特异性评估
            pressure_modifier = self._calculate_pressure_modifier(individual, selection_pressure)
            
            # 年龄和经验修正
            age_modifier = self._calculate_age_modifier(individual)
            
            # 综合生存评分
            survival_score = base_fitness * pressure_modifier * age_modifier
            
            # 生存决策
            survive = survival_score >= fitness_threshold
            cause = self._determine_selection_cause(survival_score, fitness_threshold, selection_pressure)
            
            survival_decisions[individual_id] = {
                "survive": survive,
                "cause": cause,
                "fitness_score": survival_score,
                "threshold": fitness_threshold
            }
        
        return survival_decisions
    
    def _calculate_pressure_modifier(self, individual: EvolutionaryIndividual, 
                                   selection_pressure: SelectionPressure) -> float:
        """计算压力修正因子"""
        if selection_pressure == SelectionPressure.ENVIRONMENTAL:
            # 环境适应性
            return individual.fitness_scores.get(EvolutionaryFitness.ADAPTATION, 0.5)
        
        elif selection_pressure == SelectionPressure.COMPETITIVE:
            # 竞争能力
            return individual.fitness_scores.get(EvolutionaryFitness.SURVIVAL, 0.5)
        
        elif selection_pressure == SelectionPressure.RESOURCE:
            # 资源利用效率
            return individual.fitness_scores.get(EvolutionaryFitness.EFFICIENCY, 0.5)
        
        elif selection_pressure == SelectionPressure.SOCIAL:
            # 社会协作能力
            return individual.fitness_scores.get(EvolutionaryFitness.COLLABORATION, 0.5)
        
        elif selection_pressure == SelectionPressure.INNOVATIVE:
            # 创新能力
            return individual.fitness_scores.get(EvolutionaryFitness.INNOVATION, 0.5)
        
        return 1.0
    
    def _calculate_age_modifier(self, individual: EvolutionaryIndividual) -> float:
        """计算年龄修正因子"""
        # 年轻个体（1-3代）：成长潜力加成
        if individual.generation <= 3:
            return 1.0 + (0.1 * (3 - individual.generation))
        
        # 成熟个体（4-10代）：经验加成
        elif individual.generation <= 10:
            return 1.0
        
        # 老年个体（11代以上）：衰老惩罚
        else:
            age_penalty = 0.05 * (individual.generation - 10)
            return max(0.3, 1.0 - age_penalty)  # 最低保留30%生存率
    
    def _determine_selection_cause(self, survival_score: float, threshold: float,
                                 selection_pressure: SelectionPressure) -> str:
        """确定选择原因"""
        if survival_score < threshold * 0.5:
            return f"严重不适应{selection_pressure.value}"
        elif survival_score < threshold:
            return f"部分不适应{selection_pressure.value}"
        else:
            return "适应环境"

class GeneticRecombinationOperator(BaseEvolutionaryOperator):
    """基因重组算子"""
    
    def can_apply(self, population: EvolutionaryPopulation, **kwargs) -> bool:
        return len(population.individuals) >= 4  # 至少需要4个个体进行重组
    
    async def apply(self, population: EvolutionaryPopulation,
                   recombination_rate: float = 0.7,
                   crossover_probability: float = 0.8) -> EvolutionaryPopulation:
        """应用基因重组"""
        logger.info(f"应用基因重组算子，重组率: {recombination_rate}")
        
        # 选择繁殖个体
        breeding_pairs = await self._select_breeding_pairs(population, recombination_rate)
        
        # 执行基因重组
        new_offspring = []
        for pair in breeding_pairs:
            offspring = await self._perform_genetic_recombination(pair[0], pair[1], crossover_probability)
            new_offspring.append(offspring)
        
        # 将后代加入种群
        new_population = copy.deepcopy(population)
        for offspring in new_offspring:
            new_population.individuals[offspring.individual_id] = offspring
        
        # 记录进化事件
        recombination_event = EvolutionaryEvent(
            event_id=f"recombination_{int(time.time())}_{random.randint(1000, 9999)}",
            event_type="genetic_recombination",
            timestamp=time.time(),
            affected_population=population.population_id,
            magnitude=recombination_rate,
            evolutionary_impact={
                "breeding_pairs": len(breeding_pairs),
                "new_offspring": len(new_offspring),
                "genetic_diversity_increase": len(new_offspring) / len(population.individuals)
            },
            trigger_conditions=["reproduction_season"],
            expected_outcomes=["genetic_diversity_enhancement", "adaptive_trait_inheritance"],
            actual_outcomes=[{
                "offspring_id": offspring.individual_id,
                "parents": offspring.parent_ids
            } for offspring in new_offspring]
        )
        
        new_population.population_history.append(recombination_event)
        
        # 更新种群统计
        new_population.population_size = len(new_population.individuals)
        new_population.genetic_diversity = self._calculate_genetic_diversity(new_population.individuals)
        
        logger.info(f"基因重组完成：产生 {len(new_offspring)} 个新个体")
        
        return new_population
    
    async def _select_breeding_pairs(self, population: EvolutionaryPopulation, 
                                   recombination_rate: float) -> List[Tuple[str, str]]:
        """选择繁殖配对"""
        individuals = list(population.individuals.values())
        
        # 基于适应度选择繁殖个体
        fitness_scores = []
        for individual in individuals:
            avg_fitness = sum(individual.fitness_scores.values()) / len(individual.fitness_scores)
            fitness_scores.append((individual.individual_id, avg_fitness))
        
        # 按适应度排序
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 选择前N%的个体作为繁殖候选
        breeding_candidates = fitness_scores[:max(2, int(len(fitness_scores) * recombination_rate))]
        
        # 配对（避免近亲繁殖）
        breeding_pairs = []
        used_individuals = set()
        
        for i in range(0, len(breeding_candidates) - 1, 2):
            if i + 1 < len(breeding_candidates):
                parent1_id = breeding_candidates[i][0]
                parent2_id = breeding_candidates[i + 1][0]
                
                if parent1_id not in used_individuals and parent2_id not in used_individuals:
                    breeding_pairs.append((parent1_id, parent2_id))
                    used_individuals.update([parent1_id, parent2_id])
        
        return breeding_pairs
    
    async def _perform_genetic_recombination(self, parent1_id: str, parent2_id: str,
                                           crossover_probability: float) -> EvolutionaryIndividual:
        """执行基因重组"""
        parent1 = self.engine.evolutionary_populations["default"].individuals[parent1_id]
        parent2 = self.engine.evolutionary_populations["default"].individuals[parent2_id]
        
        # 生成后代ID
        offspring_id = f"offspring_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # 重组基因组
        offspring_genome = await self._recombine_genomes(parent1.genome, parent2.genome, crossover_probability)
        
        # 生成表型
        offspring_phenotype = await self._generate_phenotype(offspring_genome)
        
        # 计算适应度
        offspring_fitness = await self._calculate_fitness(offspring_phenotype)
        
        # 创建后代个体
        offspring = EvolutionaryIndividual(
            individual_id=offspring_id,
            genome=offspring_genome,
            phenotype=offspring_phenotype,
            fitness_scores=offspring_fitness,
            evolutionary_age=0,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1_id, parent2_id],
            offspring_ids=[]
        )
        
        # 更新父代的后代列表
        parent1.offspring_ids.append(offspring_id)
        parent2.offspring_ids.append(offspring_id)
        
        return offspring
    
    async def _recombine_genomes(self, genome1: Dict[str, EvolutionaryChromosome], 
                               genome2: Dict[str, EvolutionaryChromosome],
                               crossover_probability: float) -> Dict[str, EvolutionaryChromosome]:
        """重组基因组"""
        offspring_genome = {}
        
        for chromosome_id in genome1.keys():
            if chromosome_id not in genome2:
                continue
            
            parent1_chromosome = genome1[chromosome_id]
            parent2_chromosome = genome2[chromosome_id]
            
            # 执行交叉重组
            if random.random() < crossover_probability:
                crossover_point = random.randint(1, parent1_chromosome.length - 1)
                offspring_chromosome = await self._perform_crossover(
                    parent1_chromosome, parent2_chromosome, crossover_point
                )
            else:
                # 随机选择一个亲本的完整染色体
                chosen_parent = random.choice([parent1_chromosome, parent2_chromosome])
                offspring_chromosome = copy.deepcopy(chosen_parent)
            
            # 应用突变
            if random.random() < offspring_chromosome.mutation_rate:
                await self._apply_mutations(offspring_chromosome)
            
            offspring_genome[chromosome_id] = offspring_chromosome
        
        return offspring_genome
    
    async def _perform_crossover(self, chromosome1: EvolutionaryChromosome,
                               chromosome2: EvolutionaryChromosome,
                               crossover_point: int) -> EvolutionaryChromosome:
        """执行交叉重组"""
        offspring_chromosome = copy.deepcopy(chromosome1)
        
        # 交换染色体片段
        for i, (gene_id, gene) in enumerate(chromosome2.genes.items()):
            if i >= crossover_point:
                offspring_chromosome.genes[gene_id] = copy.deepcopy(gene)
        
        return offspring_chromosome
    
    async def _apply_mutations(self, chromosome: EvolutionaryChromosome):
        """应用基因突变"""
        mutation_type = random.choice(list(GeneticMutationType))
        
        if mutation_type == GeneticMutationType.POINT_MUTATION:
            await self._apply_point_mutation(chromosome)
        elif mutation_type == GeneticMutationType.INSERTION:
            await self._apply_insertion_mutation(chromosome)
        elif mutation_type == GeneticMutationType.DELETION:
            await self._apply_deletion_mutation(chromosome)
        elif mutation_type == GeneticMutationType.DUPLICATION:
            await self._apply_duplication_mutation(chromosome)
        elif mutation_type == GeneticMutationType.INVERSION:
            await self._apply_inversion_mutation(chromosome)
        elif mutation_type == GeneticMutationType.TRANSLOCATION:
            await self._apply_translocation_mutation(chromosome)
    
    async def _apply_point_mutation(self, chromosome: EvolutionaryChromosome):
        """应用点突变"""
        if not chromosome.genes:
            return
        
        target_gene_id = random.choice(list(chromosome.genes.keys()))
        target_gene = chromosome.genes[target_gene_id]
        
        if target_gene.mutable:
            # 根据基因类型应用不同突变
            if isinstance(target_gene.value, (int, float)):
                # 数值突变
                mutation_strength = random.uniform(-0.1, 0.1)
                if isinstance(target_gene.value, int):
                    target_gene.value = int(target_gene.value * (1 + mutation_strength))
                else:
                    target_gene.value = target_gene.value * (1 + mutation_strength)
                    target_gene.value = max(0.0, min(1.0, target_gene.value))  # 限制在0-1范围
            
            elif isinstance(target_gene.value, str):
                # 字符串突变
                if target_gene.value:
                    mutation_index = random.randint(0, len(target_gene.value) - 1)
                    new_char = random.choice("abcdefghijklmnopqrstuvwxyz0123456789")
                    target_gene.value = (target_gene.value[:mutation_index] + 
                                       new_char + target_gene.value[mutation_index + 1:])
            
            # 记录突变历史
            target_gene.mutation_history.append({
                "mutation_type": "point_mutation",
                "timestamp": time.time(),
                "mutation_strength": mutation_strength if 'mutation_strength' in locals() else 0.1
            })
    
    async def _generate_phenotype(self, genome: Dict[str, EvolutionaryChromosome]) -> Dict[str, Any]:
        """生成表型"""
        phenotype = {
            "physical_traits": {},
            "behavioral_traits": {},
            "cognitive_traits": {},
            "adaptive_traits": {}
        }
        
        # 从基因组提取表型特征
        for chromosome_id, chromosome in genome.items():
            for gene_id, gene in chromosome.genes.items():
                trait_category = gene.gene_type.split("_")[0]  # 如 "physical", "behavioral" 等
                
                if trait_category in phenotype:
                    phenotype[trait_category][gene.gene_name] = gene.value
        
        return phenotype
    
    async def _calculate_fitness(self, phenotype: Dict[str, Any]) -> Dict[EvolutionaryFitness, float]:
        """计算适应度"""
        fitness_scores = {}
        
        # 生存适应度
        survival_factors = []
        if "physical" in phenotype:
            survival_factors.append(sum(phenotype["physical"].values()) / max(1, len(phenotype["physical"])))
        if "adaptive" in phenotype:
            survival_factors.append(sum(phenotype["adaptive"].values()) / max(1, len(phenotype["adaptive"])))
        
        fitness_scores[EvolutionaryFitness.SURVIVAL] = sum(survival_factors) / max(1, len(survival_factors)) if survival_factors else 0.5
        
        # 繁殖适应度
        reproduction_factors = []
        if "behavioral" in phenotype:
            reproduction_factors.append(sum(phenotype["behavioral"].values()) / max(1, len(phenotype["behavioral"])))
        
        fitness_scores[EvolutionaryFitness.REPRODUCTION] = sum(reproduction_factors) / max(1, len(reproduction_factors)) if reproduction_factors else 0.5
        
        # 其他适应度指标
        fitness_scores[EvolutionaryFitness.ADAPTATION] = random.uniform(0.3, 0.9)
        fitness_scores[EvolutionaryFitness.INNOVATION] = random.uniform(0.2, 0.8)
        fitness_scores[EvolutionaryFitness.COLLABORATION] = random.uniform(0.4, 0.9)
        fitness_scores[EvolutionaryFitness.EFFICIENCY] = random.uniform(0.3, 0.8)
        
        return fitness_scores

class GeneticDriftOperator(BaseEvolutionaryOperator):
    """遗传漂变算子"""
    
    def can_apply(self, population: EvolutionaryPopulation, **kwargs) -> bool:
        return population.population_size < 50  # 小种群容易发生遗传漂变
    
    async def apply(self, population: EvolutionaryPopulation,
                   drift_intensity: float = 0.1) -> EvolutionaryPopulation:
        """应用遗传漂变"""
        logger.info(f"应用遗传漂变算子，漂变强度: {drift_intensity}")
        
        new_population = copy.deepcopy(population)
        
        # 随机改变基因频率
        for individual in new_population.individuals.values():
            await self._apply_genetic_drift(individual, drift_intensity)
        
        # 记录进化事件
        drift_event = EvolutionaryEvent(
            event_id=f"drift_{int(time.time())}_{random.randint(1000, 9999)}",
            event_type="genetic_drift",
            timestamp=time.time(),
            affected_population=population.population_id,
            magnitude=drift_intensity,
            evolutionary_impact={
                "population_size": population.population_size,
                "drift_effect": "random_genetic_changes",
                "diversity_change": drift_intensity * 0.5
            },
            trigger_conditions=["small_population_size"],
            expected_outcomes=["genetic_diversity_change"],
            actual_outcomes=["random_allele_frequency_changes"]
        )
        
        new_population.population_history.append(drift_event)
        
        return new_population
    
    async def _apply_genetic_drift(self, individual: EvolutionaryIndividual, drift_intensity: float):
        """应用遗传漂变"""
        for chromosome in individual.genome.values():
            for gene in chromosome.genes.values():
                if gene.mutable and random.random() < drift_intensity:
                    # 随机改变基因值
                    if isinstance(gene.value, (int, float)):
                        change = random.uniform(-drift_intensity, drift_intensity)
                        if isinstance(gene.value, int):
                            gene.value += int(change * 10)
                        else:
                            gene.value = max(0.0, min(1.0, gene.value + change))

class EvolutionaryAlgorithmEngine:
    """进化算法引擎"""
    
    def __init__(self):
        # 进化算子
        self.evolutionary_operators: Dict[str, BaseEvolutionaryOperator] = {}
        self._initialize_evolutionary_operators()
        
        # 进化种群
        self.evolutionary_populations: Dict[str, EvolutionaryPopulation] = {}
        
        # 进化时代管理
        self.current_era: EvolutionaryEra = EvolutionaryEra.PREBIRTH
        self.evolutionary_clock: int = 0
        self.generation_count: int = 0
        
        # 进化参数
        self.evolutionary_parameters = {
            "selection_pressure": 0.3,
            "recombination_rate": 0.7,
            "mutation_rate": 0.01,
            "carrying_capacity": 100,
            "speciation_threshold": 0.8,
            "environment_change_rate": 0.1
        }
        
        # 进化监控
        self.evolutionary_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.evolutionary_events: List[EvolutionaryEvent] = []
        
        # 环境模拟
        self.environmental_factors: Dict[str, float] = {}
        self.ecological_niches: Dict[str, Dict[str, Any]] = {}
        
        logger.info("进化算法引擎 V2 初始化完成")
    
    def _initialize_evolutionary_operators(self):
        """初始化进化算子"""
        # 注意：这里需要先创建引擎实例，然后在各个算子中引用
        # 由于循环依赖问题，我们在算子中通过参数传递引擎引用
        
        logger.info(f"已初始化进化算子（将在运行时动态绑定）")
    
    async def initialize_population(self, population_id: str, initial_size: int = 50,
                                  genome_complexity: int = 10) -> EvolutionaryPopulation:
        """初始化进化种群"""
        logger.info(f"初始化种群: {population_id}, 初始大小: {initial_size}")
        
        # 创建初始基因池
        initial_genome = await self._create_initial_genome(genome_complexity)
        
        # 生成初始个体
        individuals = {}
        for i in range(initial_size):
            individual = await self._create_initial_individual(f"{population_id}_individual_{i}", initial_genome)
            individuals[individual.individual_id] = individual
        
        # 计算初始种群统计
        genetic_diversity = self._calculate_genetic_diversity(individuals)
        average_fitness = self._calculate_average_fitness(individuals)
        
        # 创建种群
        population = EvolutionaryPopulation(
            population_id=population_id,
            individuals=individuals,
            population_size=initial_size,
            genetic_diversity=genetic_diversity,
            average_fitness=average_fitness,
            evolutionary_pressure={
                SelectionPressure.ENVIRONMENTAL: 0.2,
                SelectionPressure.COMPETITIVE: 0.3,
                SelectionPressure.RESOURCE: 0.2,
                SelectionPressure.SOCIAL: 0.1,
                SelectionPressure.INNOVATIVE: 0.2
            },
            carrying_capacity=self.evolutionary_parameters["carrying_capacity"],
            migration_rate=0.01,
            speciation_threshold=self.evolutionary_parameters["speciation_threshold"]
        )
        
        self.evolutionary_populations[population_id] = population
        
        logger.info(f"种群初始化完成: {population_id}, 遗传多样性: {genetic_diversity:.2f}, 平均适应度: {average_fitness:.2f}")
        
        return population
    
    async def _create_initial_genome(self, complexity: int) -> Dict[str, EvolutionaryChromosome]:
        """创建初始基因组"""
        genome = {}
        
        # 创建多个染色体
        for chromo_id in range(5):  # 5条染色体
            chromosome_id = f"chromosome_{chromo_id}"
            
            # 创建基因
            genes = {}
            for gene_id in range(complexity):
                gene = EvolutionaryGene(
                    gene_id=f"{chromosome_id}_gene_{gene_id}",
                    gene_name=f"trait_{gene_id}",
                    gene_type=random.choice(["physical_strength", "behavioral_intelligence", "cognitive_memory", "adaptive_resilience"]),
                    value=random.uniform(0.0, 1.0),
                    mutable=random.random() < 0.8,  # 80%的基因可突变
                    epigenetic_markers={"methylation": random.uniform(0.0, 1.0)},
                    interaction_partners=[]
                )
                genes[gene.gene_id] = gene
            
            # 创建染色体
            chromosome = EvolutionaryChromosome(
                chromosome_id=chromosome_id,
                genes=genes,
                length=complexity,
                recombination_rate=random.uniform(0.1, 0.5),
                mutation_rate=random.uniform(0.001, 0.01)
            )
            
            genome[chromosome_id] = chromosome
        
        return genome
    
    async def _create_initial_individual(self, individual_id: str, 
                                       genome: Dict[str, EvolutionaryChromosome]) -> EvolutionaryIndividual:
        """创建初始个体"""
        # 生成表型
        phenotype = await self._generate_phenotype_from_genome(genome)
        
        # 计算初始适应度
        fitness_scores = await self._calculate_individual_fitness(phenotype)
        
        individual = EvolutionaryIndividual(
            individual_id=individual_id,
            genome=genome,
            phenotype=phenotype,
            fitness_scores=fitness_scores,
            evolutionary_age=0,
            generation=1,
            parent_ids=[],
            offspring_ids=[]
        )
        
        return individual
    
    async def _generate_phenotype_from_genome(self, genome: Dict[str, EvolutionaryChromosome]) -> Dict[str, Any]:
        """从基因组生成表型"""
        phenotype = {
            "physical": {},
            "behavioral": {},
            "cognitive": {},
            "adaptive": {}
        }
        
        for chromosome in genome.values():
            for gene in chromosome.genes.values():
                trait_category = gene.gene_type.split("_")[0]
                if trait_category in ["physical", "behavioral", "cognitive", "adaptive"]:
                    phenotype[trait_category][gene.gene_name] = gene.value
        
        return phenotype
    
    async def _calculate_individual_fitness(self, phenotype: Dict[str, Any]) -> Dict[EvolutionaryFitness, float]:
        """计算个体适应度"""
        fitness_scores = {}
        
        # 基于表型特征计算适应度
        for fitness_type in EvolutionaryFitness:
            base_score = random.uniform(0.3, 0.8)
            
            # 根据表型调整适应度
            if fitness_type == EvolutionaryFitness.SURVIVAL:
                if "physical" in phenotype:
                    physical_score = sum(phenotype["physical"].values()) / max(1, len(phenotype["physical"]))
                    base_score = (base_score + physical_score) / 2
            
            elif fitness_type == EvolutionaryFitness.ADAPTATION:
                if "adaptive" in phenotype:
                    adaptive_score = sum(phenotype["adaptive"].values()) / max(1, len(phenotype["adaptive"]))
                    base_score = (base_score + adaptive_score) / 2
            
            fitness_scores[fitness_type] = base_score
        
        return fitness_scores
    
    async def run_evolutionary_cycle(self, population_id: str = "default") -> Dict[str, Any]:
        """运行进化循环"""
        if population_id not in self.evolutionary_populations:
            logger.error(f"种群不存在: {population_id}")
            return {"success": False, "error": "Population not found"}
        
        population = self.evolutionary_populations[population_id]
        
        logger.info(f"开始进化循环: {population_id}, 当前时代: {self.current_era.value}")
        
        cycle_results = {
            "era": self.current_era.value,
            "generation": self.generation_count,
            "population_size": len(population.individuals),
            "average_fitness": population.average_fitness,
            "genetic_diversity": population.genetic_diversity,
            "applied_operators": [],
            "new_events": []
        }
        
        try:
            # 根据当前时代应用不同的进化算子
            if self.current_era == EvolutionaryEra.BIRTH:
                # 生命诞生期
                await self._apply_birth_phase(population)
                cycle_results["applied_operators"].append("birth_phase")
            
            elif self.current_era == EvolutionaryEra.ADAPTATION:
                # 适应期
                await self._apply_adaptation_phase(population)
                cycle_results["applied_operators"].append("adaptation_phase")
            
            elif self.current_era == EvolutionaryEra.COMPETITION:
                # 竞争期
                await self._apply_competition_phase(population)
                cycle_results["applied_operators"].append("competition_phase")
            
            elif self.current_era == EvolutionaryEra.REPRODUCTION:
                # 繁殖期
                await self._apply_reproduction_phase(population)
                cycle_results["applied_operators"].append("reproduction_phase")
            
            elif self.current_era == EvolutionaryEra.SELECTION:
                # 选择期
                await self._apply_selection_phase(population)
                cycle_results["applied_operators"].append("selection_phase")
            
            elif self.current_era == EvolutionaryEra.EXTINCTION:
                # 灭绝期
                await self._apply_extinction_phase(population)
                cycle_results["applied_operators"].append("extinction_phase")
            
            # 更新进化时代
            await self._update_evolutionary_era()
            
            # 更新种群统计
            population.average_fitness = self._calculate_average_fitness(population.individuals)
            population.genetic_diversity = self._calculate_genetic_diversity(population.individuals)
            population.population_size = len(population.individuals)
            
            # 记录进化指标
            self._record_evolutionary_metric(population_id, cycle_results)
            
            logger.info(f"进化循环完成: {len(cycle_results['applied_operators'])} 个算子已应用")
            
            return {"success": True, "results": cycle_results}
            
        except Exception as e:
            logger.error(f"进化循环执行失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def _update_evolutionary_era(self):
        """更新进化时代"""
        era_transitions = {
            EvolutionaryEra.PREBIRTH: EvolutionaryEra.BIRTH,
            EvolutionaryEra.BIRTH: EvolutionaryEra.ADAPTATION,
            EvolutionaryEra.ADAPTATION: EvolutionaryEra.COMPETITION,
            EvolutionaryEra.COMPETITION: EvolutionaryEra.REPRODUCTION,
            EvolutionaryEra.REPRODUCTION: EvolutionaryEra.SELECTION,
            EvolutionaryEra.SELECTION: EvolutionaryEra.EXTINCTION,
            EvolutionaryEra.EXTINCTION: EvolutionaryEra.RENAISSANCE,
            EvolutionaryEra.RENAISSANCE: EvolutionaryEra.ADAPTATION  # 循环进化
        }
        
        current_era = self.current_era
        next_era = era_transitions.get(current_era, EvolutionaryEra.BIRTH)
        
        # 基于种群状态调整时代转换
        if hasattr(self, 'evolutionary_populations') and self.evolutionary_populations:
            population = list(self.evolutionary_populations.values())[0]
            
            # 如果种群灭绝，强制进入重生期
            if population.population_size == 0:
                next_era = EvolutionaryEra.RENAISSANCE
        
        self.current_era = next_era
        self.evolutionary_clock += 1
        
        if next_era == EvolutionaryEra.BIRTH:
            self.generation_count += 1
        
        logger.info(f"进化时代转换: {current_era.value} -> {next_era.value}")
    
    def _calculate_average_fitness(self, individuals: Dict[str, EvolutionaryIndividual]) -> float:
        """计算平均适应度"""
        if not individuals:
            return 0.0
        
        total_fitness = 0.0
        count = 0
        
        for individual in individuals.values():
            individual_fitness = sum(individual.fitness_scores.values()) / len(individual.fitness_scores)
            total_fitness += individual_fitness
            count += 1
        
        return total_fitness / count if count > 0 else 0.0
    
    def _calculate_genetic_diversity(self, individuals: Dict[str, EvolutionaryIndividual]) -> float:
        """计算遗传多样性"""
        if not individuals:
            return 0.0
        
        # 简化计算：基于基因值的方差
        all_gene_values = []
        for individual in individuals.values():
            for chromosome in individual.genome.values():
                for gene in chromosome.genes.values():
                    if isinstance(gene.value, (int, float)):
                        all_gene_values.append(gene.value)
        
        if not all_gene_values:
            return 0.0
        
        # 计算方差作为多样性指标
        mean_value = sum(all_gene_values) / len(all_gene_values)
        variance = sum((x - mean_value) ** 2 for x in all_gene_values) / len(all_gene_values)
        
        # 标准化到0-1范围
        max_variance = 0.25  # 假设最大方差
        normalized_diversity = min(1.0, variance / max_variance)
        
        return normalized_diversity
    
    def _record_evolutionary_metric(self, population_id: str, metric_data: Dict[str, Any]):
        """记录进化指标"""
        metric = {
            "timestamp": time.time(),
            "population_id": population_id,
            "evolutionary_clock": self.evolutionary_clock,
            "generation": self.generation_count,
            "era": self.current_era.value,
            **metric_data
        }
        
        self.evolutionary_metrics[population_id].append(metric)
        
        # 限制历史记录数量
        if len(self.evolutionary_metrics[population_id]) > 1000:
            self.evolutionary_metrics[population_id] = self.evolutionary_metrics[population_id][-500:]
    
    async def get_evolutionary_analytics(self, population_id: str = "default") -> Dict[str, Any]:
        """获取进化分析数据"""
        if population_id not in self.evolutionary_populations:
            return {"error": "Population not found"}
        
        population = self.evolutionary_populations[population_id]
        
        analytics = {
            "population_id": population_id,
            "current_era": self.current_era.value,
            "evolutionary_clock": self.evolutionary_clock,
            "generation": self.generation_count,
            "population_size": population.population_size,
            "average_fitness": population.average_fitness,
            "genetic_diversity": population.genetic_diversity,
            "carrying_capacity": population.carrying_capacity,
            "extinction_risk": self._calculate_extinction_risk(population),
            "evolutionary_trends": {},
            "recent_events": [],
            "fitness_distribution": {},
            "genetic_health": {}
        }
        
        # 进化趋势分析
        if population_id in self.evolutionary_metrics:
            metrics = self.evolutionary_metrics[population_id]
            
            if len(metrics) >= 10:
                # 计算适应度趋势
                recent_fitness = [m["average_fitness"] for m in metrics[-10:]]
                fitness_trend = (recent_fitness[-1] - recent_fitness[0]) / max(recent_fitness[0], 0.1)
                analytics["evolutionary_trends"]["fitness_trend"] = fitness_trend
                
                # 计算种群大小趋势
                recent_sizes = [m["population_size"] for m in metrics[-10:]]
                size_trend = (recent_sizes[-1] - recent_sizes[0]) / max(recent_sizes[0], 1)
                analytics["evolutionary_trends"]["population_trend"] = size_trend
        
        # 最近事件
        recent_events = [event for event in self.evolutionary_events 
                        if event.affected_population == population_id][-10:]
        analytics["recent_events"] = [
            {
                "event_type": event.event_type,
                "timestamp": event.timestamp,
                "magnitude": event.magnitude,
                "impact": event.evolutionary_impact
            }
            for event in recent_events
        ]
        
        # 适应度分布
        fitness_scores = []
        for individual in population.individuals.values():
            avg_fitness = sum(individual.fitness_scores.values()) / len(individual.fitness_scores)
            fitness_scores.append(avg_fitness)
        
        if fitness_scores:
            analytics["fitness_distribution"] = {
                "min": min(fitness_scores),
                "max": max(fitness_scores),
                "mean": sum(fitness_scores) / len(fitness_scores),
                "std": (sum((x - sum(fitness_scores)/len(fitness_scores))**2 for x in fitness_scores) / len(fitness_scores))**0.5
            }
        
        # 遗传健康度
        analytics["genetic_health"] = {
            "diversity_score": population.genetic_diversity,
            "inbreeding_depression": self._calculate_inbreeding_depression(population),
            "mutation_load": self._calculate_mutation_load(population)
        }
        
        return analytics
    
    def _calculate_extinction_risk(self, population: EvolutionaryPopulation) -> float:
        """计算灭绝风险"""
        risk_factors = []
        
        # 种群大小风险
        size_risk = 1.0 - (population.population_size / population.carrying_capacity)
        risk_factors.append(max(0.0, size_risk))
        
        # 遗传多样性风险
        diversity_risk = 1.0 - population.genetic_diversity
        risk_factors.append(diversity_risk)
        
        # 适应度风险
        fitness_risk = 1.0 - population.average_fitness
        risk_factors.append(fitness_risk)
        
        # 综合灭绝风险
        extinction_risk = sum(risk_factors) / len(risk_factors)
        
        return min(1.0, extinction_risk)
    
    def _calculate_inbreeding_depression(self, population: EvolutionaryPopulation) -> float:
        """计算近交衰退"""
        # 简化计算：基于遗传多样性和种群大小
        if population.population_size < 10:
            return 0.8  # 小种群近交衰退严重
        elif population.genetic_diversity < 0.3:
            return 0.6  # 低多样性导致近交衰退
        else:
            return 0.1  # 健康种群
    
    def _calculate_mutation_load(self, population: EvolutionaryPopulation) -> float:
        """计算突变负荷"""
        total_mutations = 0
        total_genes = 0
        
        for individual in population.individuals.values():
            for chromosome in individual.genome.values():
                for gene in chromosome.genes.values():
                    total_genes += 1
                    if gene.mutation_history:
                        total_mutations += len(gene.mutation_history)
        
        if total_genes == 0:
            return 0.0
        
        mutation_load = total_mutations / total_genes
        
        # 标准化到0-1范围
        return min(1.0, mutation_load / 10)  # 假设平均每个基因10次突变为上限

# 全局进化算法引擎实例
_evolutionary_engine_instance = None

def get_evolutionary_engine() -> EvolutionaryAlgorithmEngine:
    """获取进化算法引擎实例"""
    global _evolutionary_engine_instance
    if _evolutionary_engine_instance is None:
        _evolutionary_engine_instance = EvolutionaryAlgorithmEngine()
    return _evolutionary_engine_instance

if __name__ == "__main__":
    # 测试代码
    async def test_evolutionary_engine():
        print("进化算法引擎 V2 测试")
        print("=" * 50)
        
        # 创建进化算法引擎
        engine = EvolutionaryAlgorithmEngine()
        
        try:
            # 初始化种群
            print("初始化进化种群...")
            population = await engine.initialize_population("test_population", initial_size=20)
            
            print(f"✅ 种群初始化成功")
            print(f"   种群ID: {population.population_id}")
            print(f"   初始大小: {population.population_size}")
            print(f"   遗传多样性: {population.genetic_diversity:.2f}")
            print(f"   平均适应度: {population.average_fitness:.2f}")
            
            # 运行多个进化循环
            print(f"\n运行进化循环...")
            for cycle in range(5):
                print(f"  进化循环 {cycle + 1}:")
                
                result = await engine.run_evolutionary_cycle("test_population")
                
                if result["success"]:
                    cycle_data = result["results"]
                    print(f"    时代: {cycle_data['era']}")
                    print(f"    种群大小: {cycle_data['population_size']}")
                    print(f"    平均适应度: {cycle_data['average_fitness']:.2f}")
                    print(f"    遗传多样性: {cycle_data['genetic_diversity']:.2f}")
                    print(f"    应用算子: {len(cycle_data['applied_operators'])} 个")
                else:
                    print(f"    ❌ 进化循环失败: {result['error']}")
                
                await asyncio.sleep(1)  # 等待1秒模拟进化时间
            
            # 获取进化分析
            print(f"\n进化分析:")
            analytics = await engine.get_evolutionary_analytics("test_population")
            
            print(f"  当前时代: {analytics['current_era']}")
            print(f"  进化时钟: {analytics['evolutionary_clock']}")
            print(f"  代数: {analytics['generation']}")
            print(f"  种群大小: {analytics['population_size']}")
            print(f"  平均适应度: {analytics['average_fitness']:.2f}")
            print(f"  遗传多样性: {analytics['genetic_diversity']:.2f}")
            print(f"  灭绝风险: {analytics['extinction_risk']:.2f}")
            
            # 显示适应度分布
            if "fitness_distribution" in analytics:
                fitness_dist = analytics["fitness_distribution"]
                print(f"  适应度分布 - 最小: {fitness_dist['min']:.2f}, 最大: {fitness_dist['max']:.2f}, 平均: {fitness_dist['mean']:.2f}")
            
            # 显示遗传健康度
            if "genetic_health" in analytics:
                health = analytics["genetic_health"]
                print(f"  遗传健康度 - 多样性: {health['diversity_score']:.2f}, 近交衰退: {health['inbreeding_depression']:.2f}")
            
            # 显示最近事件
            print(f"\n最近进化事件:")
            for event in analytics["recent_events"][-3:]:
                print(f"  {event['event_type']}: {event['magnitude']:.2f} (时间: {event['timestamp']:.0f})")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 运行测试
    asyncio.run(test_evolutionary_engine())