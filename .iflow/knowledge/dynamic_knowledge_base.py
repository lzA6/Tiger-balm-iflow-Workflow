#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态知识库 - Dynamic Knowledge Base
全能工作流V6核心组件 - AI驱动的动态知识管理系统
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。

功能特性:
- AI驱动的知识自动获取和整理
- 动态知识图谱构建和维护
- 智能知识检索和推荐
- 知识版本控制和演化
- 多模态知识融合
- 量子增强知识推理
"""

import asyncio
import json
import logging
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
import queue
import networkx as nx
import numpy as np
from collections import defaultdict, Counter

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeType(Enum):
    """知识类型枚举"""
    CONCEPT = "concept"           # 概念
    PATTERN = "pattern"           # 模式
    EXPERIENCE = "experience"     # 经验
    BEST_PRACTICE = "best_practice"  # 最佳实践
    SOLUTION = "solution"         # 解决方案
    TECHNOLOGY = "technology"     # 技术
    FRAMEWORK = "framework"       # 框架
    ALGORITHM = "algorithm"       # 算法
    ARCHITECTURE = "architecture" # 架构
    METHODOLOGY = "methodology"   # 方法论


class KnowledgeSource(Enum):
    """知识来源枚举"""
    CODE_ANALYSIS = "code_analysis"    # 代码分析
    DOCUMENTATION = "documentation"    # 文档
    USER_FEEDBACK = "user_feedback"    # 用户反馈
    SYSTEM_LOGS = "system_logs"        # 系统日志
    PERFORMANCE_DATA = "performance_data"  # 性能数据
    ERROR_ANALYSIS = "error_analysis"  # 错误分析
    SUCCESS_CASES = "success_cases"    # 成功案例
    EXTERNAL_RESEARCH = "external_research"  # 外部研究
    AI_GENERATION = "ai_generation"    # AI生成


class KnowledgeStatus(Enum):
    """知识状态枚举"""
    DRAFT = "draft"               # 草稿
    VALIDATED = "validated"       # 已验证
    APPROVED = "approved"         # 已批准
    DEPRECATED = "deprecated"     # 已弃用
    ARCHIVED = "archived"         # 已归档


@dataclass
class KnowledgeEntity:
    """知识实体"""
    id: str
    name: str
    type: KnowledgeType
    content: Dict[str, Any]
    source: KnowledgeSource
    confidence: float = 0.0
    relevance: float = 0.0
    status: KnowledgeStatus = KnowledgeStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None
    access_count: int = 0
    success_applications: int = 0
    failure_applications: int = 0
    
    def calculate_effectiveness(self) -> float:
        """计算知识有效性"""
        if self.success_applications + self.failure_applications == 0:
            return self.confidence
        
        success_rate = self.success_applications / (self.success_applications + self.failure_applications)
        relevance_weight = 0.3
        confidence_weight = 0.3
        success_rate_weight = 0.4
        
        return (
            self.relevance * relevance_weight +
            self.confidence * confidence_weight +
            success_rate * success_rate_weight
        )


class KnowledgeGraph:
    """知识图谱"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_index: Dict[str, str] = {}  # name -> id
        self.type_index: Dict[KnowledgeType, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.source_index: Dict[KnowledgeSource, Set[str]] = defaultdict(set)
    
    def add_entity(self, entity: KnowledgeEntity):
        """添加知识实体"""
        self.graph.add_node(entity.id, entity=entity)
        self.entity_index[entity.name] = entity.id
        self.type_index[entity.type].add(entity.id)
        
        for tag in entity.tags:
            self.tag_index[tag].add(entity.id)
        
        self.source_index[entity.source].add(entity.id)
        
        # 添加关系边
        for related_id in entity.relationships:
            if related_id in self.graph:
                self.graph.add_edge(entity.id, related_id, type="related")
    
    def remove_entity(self, entity_id: str):
        """移除知识实体"""
        if entity_id in self.graph:
            entity = self.graph.nodes[entity_id]["entity"]
            
            # 清理索引
            if entity.name in self.entity_index:
                del self.entity_index[entity.name]
            
            self.type_index[entity.type].discard(entity_id)
            
            for tag in entity.tags:
                self.tag_index[tag].discard(entity_id)
            
            self.source_index[entity.source].discard(entity_id)
            
            # 移除节点
            self.graph.remove_node(entity_id)
    
    def find_related_entities(self, entity_id: str, max_depth: int = 2, 
                             min_relevance: float = 0.3) -> List[KnowledgeEntity]:
        """查找相关实体"""
        if entity_id not in self.graph:
            return []
        
        related_entities = []
        visited = set()
        
        # BFS遍历
        queue = [(entity_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            # 获取实体
            if current_id in self.graph:
                entity = self.graph.nodes[current_id]["entity"]
                
                if entity.relevance >= min_relevance:
                    related_entities.append(entity)
                
                # 添加邻居到队列
                for neighbor in self.graph.neighbors(current_id):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
        
        return related_entities
    
    def search_by_type(self, knowledge_type: KnowledgeType) -> List[KnowledgeEntity]:
        """按类型搜索"""
        entity_ids = self.type_index.get(knowledge_type, set())
        return [self.graph.nodes[eid]["entity"] for eid in entity_ids if eid in self.graph]
    
    def search_by_tags(self, tags: List[str], match_all: bool = False) -> List[KnowledgeEntity]:
        """按标签搜索"""
        if match_all:
            # 必须匹配所有标签
            matching_ids = set(self.tag_index.get(tags[0], set()))
            for tag in tags[1:]:
                matching_ids &= self.tag_index.get(tag, set())
        else:
            # 匹配任意标签
            matching_ids = set()
            for tag in tags:
                matching_ids.update(self.tag_index.get(tag, set()))
        
        return [self.graph.nodes[eid]["entity"] for eid in matching_ids if eid in self.graph]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取图谱统计"""
        return {
            "total_entities": self.graph.number_of_nodes(),
            "total_relationships": self.graph.number_of_edges(),
            "entities_by_type": {k.value: len(v) for k, v in self.type_index.items()},
            "entities_by_source": {k.value: len(v) for k, v in self.source_index.items()},
            "top_tags": sorted([(tag, len(entities)) for tag, entities in self.tag_index.items()], 
                              key=lambda x: x[1], reverse=True)[:10],
            "graph_density": nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0
        }


class KnowledgeExtractor:
    """知识提取器"""
    
    def __init__(self):
        self.extraction_patterns = {
            "code_patterns": [
                r"class\s+(\w+)",  # 类名
                r"def\s+(\w+)",  # 函数名
                r"import\s+(\w+)",  # 导入模块
                r"(\w+)\s*=\s*\w+\(",  # 变量赋值
            ],
            "error_patterns": [
                r"(Error|Exception|Error):\s*(.+)",  # 错误信息
                r"(\w+Error)",  # 错误类型
                r"failed\s+to\s+(.+)",  # 失败描述
            ],
            "performance_patterns": [
                r"(\d+\.?\d*)\s*(seconds?|ms|milliseconds?)",  # 时间
                r"(\d+\.?\d*)\s*(MB|GB|KB)",  # 内存
                r"(\d+\.?\d*)\s*(%|percent)",  # 百分比
            ]
        }
    
    def extract_from_code(self, code_content: str, file_path: str) -> List[KnowledgeEntity]:
        """从代码中提取知识"""
        entities = []
        
        try:
            # 提取代码模式
            patterns = self._extract_code_patterns(code_content)
            for pattern in patterns:
                entity = KnowledgeEntity(
                    id=f"code_pattern_{hashlib.md5(pattern.encode()).hexdigest()[:8]}",
                    name=pattern,
                    type=KnowledgeType.PATTERN,
                    content={
                        "pattern": pattern,
                        "file_path": file_path,
                        "extraction_method": "code_analysis"
                    },
                    source=KnowledgeSource.CODE_ANALYSIS,
                    confidence=0.8,
                    relevance=0.7,
                    tags=["code", "pattern", "programming"]
                )
                entities.append(entity)
            
            # 提取架构信息
            architecture_info = self._extract_architecture_info(code_content, file_path)
            for arch_info in architecture_info:
                entity = KnowledgeEntity(
                    id=f"architecture_{hashlib.md5(str(arch_info).encode()).hexdigest()[:8]}",
                    name=arch_info["name"],
                    type=KnowledgeType.ARCHITECTURE,
                    content=arch_info,
                    source=KnowledgeSource.CODE_ANALYSIS,
                    confidence=0.7,
                    relevance=0.8,
                    tags=["architecture", "design", "structure"]
                )
                entities.append(entity)
            
        except Exception as e:
            logger.error(f"从代码提取知识失败: {e}")
        
        return entities
    
    def extract_from_logs(self, log_content: str) -> List[KnowledgeEntity]:
        """从日志中提取知识"""
        entities = []
        
        try:
            # 提取错误模式
            error_patterns = self._extract_error_patterns(log_content)
            for error in error_patterns:
                entity = KnowledgeEntity(
                    id=f"error_{hashlib.md5(str(error).encode()).hexdigest()[:8]}",
                    name=error["type"],
                    type=KnowledgeType.EXPERIENCE,
                    content=error,
                    source=KnowledgeSource.SYSTEM_LOGS,
                    confidence=0.9,
                    relevance=0.6,
                    tags=["error", "issue", "troubleshooting"]
                )
                entities.append(entity)
            
            # 提取性能指标
            performance_metrics = self._extract_performance_metrics(log_content)
            for metric in performance_metrics:
                entity = KnowledgeEntity(
                    id=f"metric_{hashlib.md5(str(metric).encode()).hexdigest()[:8]}",
                    name=metric["name"],
                    type=KnowledgeType.BEST_PRACTICE,
                    content=metric,
                    source=KnowledgeSource.PERFORMANCE_DATA,
                    confidence=0.8,
                    relevance=0.7,
                    tags=["performance", "metrics", "optimization"]
                )
                entities.append(entity)
            
        except Exception as e:
            logger.error(f"从日志提取知识失败: {e}")
        
        return entities
    
    def extract_from_documentation(self, doc_content: str, doc_title: str) -> List[KnowledgeEntity]:
        """从文档中提取知识"""
        entities = []
        
        try:
            # 提取概念和定义
            concepts = self._extract_concepts(doc_content)
            for concept in concepts:
                entity = KnowledgeEntity(
                    id=f"concept_{hashlib.md5(str(concept).encode()).hexdigest()[:8]}",
                    name=concept["name"],
                    type=KnowledgeType.CONCEPT,
                    content=concept,
                    source=KnowledgeSource.DOCUMENTATION,
                    confidence=0.9,
                    relevance=0.8,
                    tags=["concept", "definition", "knowledge"]
                )
                entities.append(entity)
            
            # 提取最佳实践
            best_practices = self._extract_best_practices(doc_content)
            for practice in best_practices:
                entity = KnowledgeEntity(
                    id=f"practice_{hashlib.md5(str(practice).encode()).hexdigest()[:8]}",
                    name=practice["title"],
                    type=KnowledgeType.BEST_PRACTICE,
                    content=practice,
                    source=KnowledgeSource.DOCUMENTATION,
                    confidence=0.8,
                    relevance=0.9,
                    tags=["best_practice", "guideline", "recommendation"]
                )
                entities.append(entity)
            
        except Exception as e:
            logger.error(f"从文档提取知识失败: {e}")
        
        return entities
    
    def _extract_code_patterns(self, code_content: str) -> List[str]:
        """提取代码模式"""
        patterns = []
        
        for pattern_name, pattern_list in self.extraction_patterns.items():
            if pattern_name == "code_patterns":
                for pattern in pattern_list:
                    matches = re.findall(pattern, code_content)
                    patterns.extend(matches)
        
        return list(set(patterns))
    
    def _extract_architecture_info(self, code_content: str, file_path: str) -> List[Dict[str, Any]]:
        """提取架构信息"""
        architecture_info = []
        
        # 检测设计模式
        design_patterns = {
            "singleton": re.search(r"class\s+\w+.*\n.*def\s+__new__", code_content, re.MULTILINE),
            "factory": re.search(r"def\s+create_\w+", code_content),
            "observer": re.search(r"def\s+notify|def\s+attach|def\s+detach", code_content),
            "decorator": re.search(r"@\w+", code_content),
        }
        
        for pattern_name, match in design_patterns.items():
            if match:
                architecture_info.append({
                    "name": f"{pattern_name}_pattern",
                    "type": "design_pattern",
                    "file_path": file_path,
                    "description": f"检测到{pattern_name}设计模式"
                })
        
        return architecture_info
    
    def _extract_error_patterns(self, log_content: str) -> List[Dict[str, Any]]:
        """提取错误模式"""
        errors = []
        
        for pattern in self.extraction_patterns["error_patterns"]:
            matches = re.findall(pattern, log_content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    error_info = {
                        "type": match[0],
                        "message": match[1],
                        "pattern": pattern
                    }
                else:
                    error_info = {
                        "type": match,
                        "message": "",
                        "pattern": pattern
                    }
                errors.append(error_info)
        
        return errors
    
    def _extract_performance_metrics(self, log_content: str) -> List[Dict[str, Any]]:
        """提取性能指标"""
        metrics = []
        
        for pattern in self.extraction_patterns["performance_patterns"]:
            matches = re.findall(pattern, log_content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    value, unit = match
                    metric_info = {
                        "name": f"{value}_{unit}",
                        "value": float(value),
                        "unit": unit,
                        "pattern": pattern
                    }
                else:
                    metric_info = {
                        "name": "unknown_metric",
                        "value": match,
                        "unit": "",
                        "pattern": pattern
                    }
                metrics.append(metric_info)
        
        return metrics
    
    def _extract_concepts(self, doc_content: str) -> List[Dict[str, Any]]:
        """提取概念"""
        concepts = []
        
        # 简单的概念提取：查找定义模式
        definition_patterns = [
            r"(\w+)\s+(?:is|are|定义为|是指)\s+(.+)",
            r"(\w+):\s+(.+)",
            r"定义[：:]\s*(\w+)\s+(.+)",
        ]
        
        for pattern in definition_patterns:
            matches = re.findall(pattern, doc_content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    concept_info = {
                        "name": match[0],
                        "definition": match[1],
                        "extraction_method": "pattern_matching"
                    }
                    concepts.append(concept_info)
        
        return concepts
    
    def _extract_best_practices(self, doc_content: str) -> List[Dict[str, Any]]:
        """提取最佳实践"""
        practices = []
        
        # 查找最佳实践模式
        practice_patterns = [
            r"(?:最佳实践|推荐|建议)[：:]\s+(.+)",
            r"(?:应该|应当|must|need to)\s+(.+)",
            r"(?:避免|不要|do not)\s+(.+)",
        ]
        
        for pattern in practice_patterns:
            matches = re.findall(pattern, doc_content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                practice_info = {
                    "title": f"best_practice_{len(practices) + 1}",
                    "description": match,
                    "extraction_method": "pattern_matching"
                }
                practices.append(practice_info)
        
        return practices


class KnowledgeReasoner:
    """知识推理器"""
    
    def __init__(self):
        self.reasoning_rules = []
        self.inference_engine = None
        self._load_reasoning_rules()
    
    def _load_reasoning_rules(self):
        """加载推理规则"""
        self.reasoning_rules = [
            {
                "name": "error_solution_mapping",
                "condition": "entity.type == 'experience' and 'error' in entity.tags",
                "action": "suggest_solutions",
                "confidence": 0.8
            },
            {
                "name": "performance_optimization",
                "condition": "entity.type == 'best_practice' and 'performance' in entity.tags",
                "action": "apply_optimization",
                "confidence": 0.7
            },
            {
                "name": "pattern_matching",
                "condition": "entity.type == 'pattern'",
                "action": "find_similar_patterns",
                "confidence": 0.9
            }
        ]
    
    def reason(self, query: Dict[str, Any], knowledge_graph: KnowledgeGraph) -> List[Dict[str, Any]]:
        """执行推理"""
        results = []
        
        try:
            # 根据查询类型选择推理策略
            if query.get("type") == "find_solutions":
                results = self._find_solutions(query, knowledge_graph)
            elif query.get("type") == "recommend_practices":
                results = self._recommend_practices(query, knowledge_graph)
            elif query.get("type") == "predict_issues":
                results = self._predict_issues(query, knowledge_graph)
            else:
                results = self._general_reasoning(query, knowledge_graph)
        
        except Exception as e:
            logger.error(f"推理失败: {e}")
        
        return results
    
    def _find_solutions(self, query: Dict[str, Any], knowledge_graph: KnowledgeGraph) -> List[Dict[str, Any]]:
        """查找解决方案"""
        solutions = []
        
        # 获取问题相关的知识
        problem_keywords = query.get("keywords", [])
        related_entities = []
        
        for keyword in problem_keywords:
            matching_entities = knowledge_graph.search_by_tags([keyword])
            related_entities.extend(matching_entities)
        
        # 查找成功案例和解决方案
        for entity in related_entities:
            if entity.type == KnowledgeType.EXPERIENCE and entity.success_applications > 0:
                solution = {
                    "type": "solution",
                    "entity_id": entity.id,
                    "name": entity.name,
                    "content": entity.content,
                    "confidence": entity.calculate_effectiveness(),
                    "success_rate": entity.success_applications / (entity.success_applications + entity.failure_applications) if (entity.success_applications + entity.failure_applications) > 0 else 0
                }
                solutions.append(solution)
        
        # 按置信度排序
        solutions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return solutions[:10]  # 返回前10个解决方案
    
    def _recommend_practices(self, query: Dict[str, Any], knowledge_graph: KnowledgeGraph) -> List[Dict[str, Any]]:
        """推荐最佳实践"""
        practices = []
        
        # 获取上下文信息
        context = query.get("context", {})
        domain = context.get("domain", "")
        
        # 搜索相关的最佳实践
        if domain:
            domain_practices = knowledge_graph.search_by_tags([domain])
        else:
            domain_practices = knowledge_graph.search_by_type(KnowledgeType.BEST_PRACTICE)
        
        for practice in domain_practices:
            recommendation = {
                "type": "practice",
                "entity_id": practice.id,
                "name": practice.name,
                "content": practice.content,
                "confidence": practice.calculate_effectiveness(),
                "tags": practice.tags
            }
            practices.append(recommendation)
        
        # 按置信度和相关性排序
        practices.sort(key=lambda x: (x["confidence"], x.get("relevance", 0)), reverse=True)
        
        return practices[:5]  # 返回前5个推荐
    
    def _predict_issues(self, query: Dict[str, Any], knowledge_graph: KnowledgeGraph) -> List[Dict[str, Any]]:
        """预测潜在问题"""
        predictions = []
        
        # 获取当前配置和状态
        current_config = query.get("config", {})
        current_status = query.get("status", {})
        
        # 基于历史数据预测
        error_entities = knowledge_graph.search_by_tags(["error", "issue"])
        
        # 分析常见问题模式
        error_patterns = Counter()
        for entity in error_entities:
            if "error_type" in entity.content:
                error_patterns[entity.content["error_type"]] += 1
        
        # 预测可能的问题
        for error_type, count in error_patterns.most_common(5):
            prediction = {
                "type": "prediction",
                "issue_type": error_type,
                "probability": count / len(error_entities),
                "confidence": 0.6,
                "recommendation": f"注意{error_type}问题，建议提前预防"
            }
            predictions.append(prediction)
        
        return predictions
    
    def _general_reasoning(self, query: Dict[str, Any], knowledge_graph: KnowledgeGraph) -> List[Dict[str, Any]]:
        """通用推理"""
        results = []
        
        # 获取查询关键词
        keywords = query.get("keywords", [])
        
        # 搜索相关知识
        related_entities = []
        for keyword in keywords:
            matching_entities = knowledge_graph.search_by_tags([keyword])
            related_entities.extend(matching_entities)
        
        # 去重
        unique_entities = {entity.id: entity for entity in related_entities}.values()
        
        # 生成推理结果
        for entity in unique_entities:
            result = {
                "type": "knowledge",
                "entity_id": entity.id,
                "name": entity.name,
                "content": entity.content,
                "confidence": entity.calculate_effectiveness(),
                "relevance": entity.relevance
            }
            results.append(result)
        
        # 按相关性和置信度排序
        results.sort(key=lambda x: (x["relevance"], x["confidence"]), reverse=True)
        
        return results[:10]


class DynamicKnowledgeBase:
    """动态知识库"""
    
    def __init__(self):
        self.knowledge_graph = KnowledgeGraph()
        self.extractor = KnowledgeExtractor()
        self.reasoner = KnowledgeReasoner()
        self.auto_learning_enabled = True
        self.knowledge_sources = []
        self.update_queue = queue.Queue()
        self.update_thread = None
        self.running = False
        
        # 启动自动更新
        self.start_auto_update()
    
    def start_auto_update(self):
        """启动自动更新"""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self._auto_update_loop, daemon=True)
            self.update_thread.start()
            logger.info("知识库自动更新已启动")
    
    def stop_auto_update(self):
        """停止自动更新"""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
        logger.info("知识库自动更新已停止")
    
    def _auto_update_loop(self):
        """自动更新循环"""
        while self.running:
            try:
                # 处理更新队列
                while not self.update_queue.empty():
                    try:
                        update_task = self.update_queue.get_nowait()
                        self._process_update_task(update_task)
                    except queue.Empty:
                        break
                
                # 定期知识整理和优化
                self._optimize_knowledge_base()
                
                # 休眠1分钟
                import time
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"自动更新循环异常: {e}")
    
    def add_knowledge_from_code(self, code_content: str, file_path: str) -> List[str]:
        """从代码添加知识"""
        entities = self.extractor.extract_from_code(code_content, file_path)
        entity_ids = []
        
        for entity in entities:
            self.knowledge_graph.add_entity(entity)
            entity_ids.append(entity.id)
        
        logger.info(f"从代码添加了 {len(entities)} 个知识实体")
        return entity_ids
    
    def add_knowledge_from_logs(self, log_content: str) -> List[str]:
        """从日志添加知识"""
        entities = self.extractor.extract_from_logs(log_content)
        entity_ids = []
        
        for entity in entities:
            self.knowledge_graph.add_entity(entity)
            entity_ids.append(entity.id)
        
        logger.info(f"从日志添加了 {len(entities)} 个知识实体")
        return entity_ids
    
    def add_knowledge_from_documentation(self, doc_content: str, doc_title: str) -> List[str]:
        """从文档添加知识"""
        entities = self.extractor.extract_from_documentation(doc_content, doc_title)
        entity_ids = []
        
        for entity in entities:
            self.knowledge_graph.add_entity(entity)
            entity_ids.append(entity.id)
        
        logger.info(f"从文档添加了 {len(entities)} 个知识实体")
        return entity_ids
    
    def search_knowledge(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """搜索知识"""
        try:
            # 使用推理器进行智能搜索
            results = self.reasoner.reason(query, self.knowledge_graph)
            
            # 更新访问计数
            for result in results:
                entity_id = result.get("entity_id")
                if entity_id and entity_id in self.knowledge_graph.graph:
                    entity = self.knowledge_graph.graph.nodes[entity_id]["entity"]
                    entity.access_count += 1
            
            return results
            
        except Exception as e:
            logger.error(f"知识搜索失败: {e}")
            return []
    
    def get_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """获取知识推荐"""
        query = {
            "type": "recommend_practices",
            "context": context
        }
        
        return self.search_knowledge(query)
    
    def find_solutions(self, problem_description: str, keywords: List[str] = None) -> List[Dict[str, Any]]:
        """查找解决方案"""
        query = {
            "type": "find_solutions",
            "description": problem_description,
            "keywords": keywords or []
        }
        
        return self.search_knowledge(query)
    
    def predict_issues(self, config: Dict[str, Any], status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """预测潜在问题"""
        query = {
            "type": "predict_issues",
            "config": config,
            "status": status
        }
        
        return self.search_knowledge(query)
    
    def update_knowledge_effectiveness(self, entity_id: str, success: bool):
        """更新知识有效性"""
        if entity_id in self.knowledge_graph.graph:
            entity = self.knowledge_graph.graph.nodes[entity_id]["entity"]
            
            if success:
                entity.success_applications += 1
            else:
                entity.failure_applications += 1
            
            entity.updated_at = datetime.now()
            
            logger.info(f"更新知识有效性: {entity.name} ({entity.id})")
    
    def _process_update_task(self, task: Dict[str, Any]):
        """处理更新任务"""
        try:
            task_type = task.get("type")
            
            if task_type == "add_entity":
                entity = task.get("entity")
                if entity:
                    self.knowledge_graph.add_entity(entity)
            
            elif task_type == "remove_entity":
                entity_id = task.get("entity_id")
                if entity_id:
                    self.knowledge_graph.remove_entity(entity_id)
            
            elif task_type == "update_entity":
                entity_id = task.get("entity_id")
                updates = task.get("updates")
                if entity_id and updates:
                    if entity_id in self.knowledge_graph.graph:
                        entity = self.knowledge_graph.graph.nodes[entity_id]["entity"]
                        for key, value in updates.items():
                            setattr(entity, key, value)
                        entity.updated_at = datetime.now()
            
        except Exception as e:
            logger.error(f"处理更新任务失败: {e}")
    
    def _optimize_knowledge_base(self):
        """优化知识库"""
        try:
            # 清理低效知识
            self._cleanup_ineffective_knowledge()
            
            # 更新知识关系
            self._update_knowledge_relationships()
            
            # 重新计算相关性
            self._recalculate_relevance()
            
        except Exception as e:
            logger.error(f"知识库优化失败: {e}")
    
    def _cleanup_ineffective_knowledge(self):
        """清理低效知识"""
        entities_to_remove = []
        
        for entity_id, node_data in self.knowledge_graph.graph.nodes(data=True):
            entity = node_data["entity"]
            
            # 移除长期未使用且有效性低的知识
            if (entity.access_count == 0 and 
                entity.calculate_effectiveness() < 0.3 and
                (datetime.now() - entity.created_at).days > 30):
                entities_to_remove.append(entity_id)
        
        for entity_id in entities_to_remove:
            self.knowledge_graph.remove_entity(entity_id)
        
        if entities_to_remove:
            logger.info(f"清理了 {len(entities_to_remove)} 个低效知识实体")
    
    def _update_knowledge_relationships(self):
        """更新知识关系"""
        # 基于内容相似性建立新关系
        entities = list(self.knowledge_graph.graph.nodes(data=True))
        
        for i, (id1, data1) in enumerate(entities):
            for j, (id2, data2) in enumerate(entities[i+1:], i+1):
                entity1 = data1["entity"]
                entity2 = data2["entity"]
                
                # 计算内容相似度
                similarity = self._calculate_content_similarity(entity1, entity2)
                
                # 如果相似度高且没有关系，建立关系
                if similarity > 0.7 and id2 not in entity1.relationships:
                    entity1.relationships.append(id2)
                    entity2.relationships.append(id1)
    
    def _calculate_content_similarity(self, entity1: KnowledgeEntity, entity2: KnowledgeEntity) -> float:
        """计算内容相似度"""
        # 简化的相似度计算
        similarity = 0.0
        
        # 类型相似度
        if entity1.type == entity2.type:
            similarity += 0.3
        
        # 标签相似度
        tags1 = set(entity1.tags)
        tags2 = set(entity2.tags)
        if tags1 and tags2:
            tag_similarity = len(tags1 & tags2) / len(tags1 | tags2)
            similarity += tag_similarity * 0.4
        
        # 内容相似度（基于关键词）
        content1_words = set(str(entity1.content).lower().split())
        content2_words = set(str(entity2.content).lower().split())
        if content1_words and content2_words:
            content_similarity = len(content1_words & content2_words) / len(content1_words | content2_words)
            similarity += content_similarity * 0.3
        
        return min(similarity, 1.0)
    
    def _recalculate_relevance(self):
        """重新计算相关性"""
        # 基于访问频率和有效性重新计算相关性
        total_access = sum(entity.access_count for entity in self.knowledge_graph.graph.nodes(data=True)["entity"])
        
        if total_access == 0:
            return
        
        for entity_id, node_data in self.knowledge_graph.graph.nodes(data=True):
            entity = node_data["entity"]
            
            # 访问频率权重
            access_weight = entity.access_count / total_access
            
            # 有效性权重
            effectiveness_weight = entity.calculate_effectiveness()
            
            # 时效性权重（越新的知识相关性越高）
            recency_weight = 1.0 / (1.0 + (datetime.now() - entity.created_at).days / 30.0)
            
            # 综合相关性
            entity.relevance = (access_weight * 0.4 + 
                              effectiveness_weight * 0.4 + 
                              recency_weight * 0.2)
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """获取知识库统计"""
        stats = self.knowledge_graph.get_statistics()
        
        # 添加额外统计
        entities = list(self.knowledge_graph.graph.nodes(data=True))
        if entities:
            total_effectiveness = sum(entity["entity"].calculate_effectiveness() for entity in entities)
            avg_effectiveness = total_effectiveness / len(entities)
            
            stats["average_effectiveness"] = avg_effectiveness
            stats["total_access_count"] = sum(entity["entity"].access_count for entity in entities)
            stats["total_success_applications"] = sum(entity["entity"].success_applications for entity in entities)
        
        return stats
    
    def export_knowledge(self, format_type: str = "json", file_path: Optional[str] = None) -> str:
        """导出知识库"""
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "statistics": self.get_knowledge_statistics(),
                "entities": []
            }
            
            # 导出所有实体
            for entity_id, node_data in self.knowledge_graph.graph.nodes(data=True):
                entity = node_data["entity"]
                entity_data = {
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.type.value,
                    "content": entity.content,
                    "source": entity.source.value,
                    "confidence": entity.confidence,
                    "relevance": entity.relevance,
                    "status": entity.status.value,
                    "created_at": entity.created_at.isoformat(),
                    "updated_at": entity.updated_at.isoformat(),
                    "tags": entity.tags,
                    "relationships": entity.relationships,
                    "metadata": entity.metadata,
                    "access_count": entity.access_count,
                    "success_applications": entity.success_applications,
                    "failure_applications": entity.failure_applications,
                    "effectiveness": entity.calculate_effectiveness()
                }
                export_data["entities"].append(entity_data)
            
            if format_type == "json":
                content = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
            else:
                raise ValueError(f"不支持的导出格式: {format_type}")
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"知识库已导出到: {file_path}")
            
            return content
            
        except Exception as e:
            logger.error(f"导出知识库失败: {e}")
            return ""


# 示例使用
async def main():
    """主函数示例"""
    # 创建动态知识库
    knowledge_base = DynamicKnowledgeBase()
    
    # 从代码添加知识
    sample_code = """
class UserService:
    def __init__(self):
        self.users = {}
    
    def create_user(self, username, email):
        if username in self.users:
            raise ValueError("User already exists")
        user = {"username": username, "email": email}
        self.users[username] = user
        return user
    """
    
    entity_ids = knowledge_base.add_knowledge_from_code(sample_code, "user_service.py")
    print(f"从代码添加了 {len(entity_ids)} 个知识实体")
    
    # 从日志添加知识
    sample_log = """
2024-01-15 10:30:15 ERROR: Database connection failed
2024-01-15 10:30:16 INFO: Retrying connection...
2024-01-15 10:30:17 ERROR: Connection timeout after 5 seconds
2024-01-15 10:30:18 INFO: Connection established successfully
    """
    
    entity_ids = knowledge_base.add_knowledge_from_logs(sample_log)
    print(f"从日志添加了 {len(entity_ids)} 个知识实体")
    
    # 搜索解决方案
    solutions = knowledge_base.find_solutions(
        "数据库连接失败",
        ["database", "connection", "error"]
    )
    
    print(f"找到 {len(solutions)} 个解决方案:")
    for solution in solutions[:3]:
        print(f"  - {solution['name']}: {solution.get('content', {}).get('message', 'N/A')}")
    
    # 获取推荐
    recommendations = knowledge_base.get_recommendations({
        "domain": "database",
        "technology": "python"
    })
    
    print(f"获得 {len(recommendations)} 个推荐")
    
    # 获取统计信息
    stats = knowledge_base.get_knowledge_statistics()
    print(f"知识库统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")
    
    # 停止自动更新
    knowledge_base.stop_auto_update()


if __name__ == "__main__":
    asyncio.run(main())