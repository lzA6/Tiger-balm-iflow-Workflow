#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
择优合并策略 - Selective Merge Strategy
全能工作流V6核心组件 - 智能择优合并和版本控制系统

功能特性:
- AI驱动的智能组件选择和合并
- 多维度组件性能评估
- 自动冲突检测和解决
- 版本兼容性验证
- 回滚机制和安全保障
- 量子增强决策算法
"""

import asyncio
import json
import logging
import hashlib
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
import queue
import copy
import difflib
import yaml
import networkx as nx
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MergeStrategy(Enum):
    """合并策略枚举"""
    BEST_PERFORMANCE = "best_performance"           # 最佳性能
    BEST_QUALITY = "best_quality"                   # 最佳质量
    BEST_COMPATIBILITY = "best_compatibility"       # 最佳兼容性
    BALANCED = "balanced"                           # 平衡策略
    CONSERVATIVE = "conservative"                   # 保守策略
    AGGRESSIVE = "aggressive"                       # 激进策略
    CUSTOM = "custom"                             # 自定义策略


class ComponentType(Enum):
    """组件类型枚举"""
    CORE = "core"                                 # 核心组件
    AGENT = "agent"                               # 智能体组件
    WORKFLOW = "workflow"                         # 工作流组件
    TOOL = "tool"                                 # 工具组件
    CONFIG = "config"                             # 配置组件
    HOOK = "hook"                                 # Hook组件
    LIBRARY = "library"                           # 库组件
    UI = "ui"                                     # UI组件
    API = "api"                                   # API组件


class MergeStatus(Enum):
    """合并状态枚举"""
    PENDING = "pending"                           # 待处理
    ANALYZING = "analyzing"                       # 分析中
    MERGING = "merging"                           # 合并中
    COMPLETED = "completed"                       # 已完成
    FAILED = "failed"                             # 失败
    ROLLED_BACK = "rolled_back"                   # 已回滚
    CONFLICT = "conflict"                         # 冲突


@dataclass
class Component:
    """组件数据类"""
    id: str
    name: str
    type: ComponentType
    version: str
    content: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    compatibility_matrix: Dict[str, float] = field(default_factory=dict)
    checksum: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """初始化后处理"""
        self.update_checksum()
    
    def update_checksum(self):
        """更新校验和"""
        content_str = json.dumps(self.content, sort_keys=True, default=str)
        self.checksum = hashlib.sha256(content_str.encode()).hexdigest()
    
    def calculate_compatibility(self, other: 'Component') -> float:
        """计算与另一个组件的兼容性"""
        if self.type != other.type:
            return 0.0
        
        # 版本兼容性检查
        version_compatibility = self._check_version_compatibility(other.version)
        
        # 依赖兼容性检查
        dependency_compatibility = self._check_dependency_compatibility(other)
        
        # API兼容性检查
        api_compatibility = self._check_api_compatibility(other)
        
        # 配置兼容性检查
        config_compatibility = self._check_config_compatibility(other)
        
        # 综合兼容性评分
        compatibility = (
            version_compatibility * 0.3 +
            dependency_compatibility * 0.25 +
            api_compatibility * 0.25 +
            config_compatibility * 0.2
        )
        
        return compatibility
    
    def _check_version_compatibility(self, other_version: str) -> float:
        """检查版本兼容性"""
        try:
            # 简化的版本兼容性检查
            current_parts = self.version.split('.')
            other_parts = other_version.split('.')
            
            # 主版本必须相同
            if current_parts[0] != other_parts[0]:
                return 0.0
            
            # 次版本向后兼容
            if len(current_parts) >= 2 and len(other_parts) >= 2:
                if int(other_parts[1]) < int(current_parts[1]):
                    return 0.5  # 可能不兼容
                elif int(other_parts[1]) > int(current_parts[1]):
                    return 0.9  # 向前兼容
                else:
                    return 1.0  # 完全兼容
            
            return 0.8
            
        except (ValueError, IndexError):
            return 0.5
    
    def _check_dependency_compatibility(self, other: 'Component') -> float:
        """检查依赖兼容性"""
        if not self.dependencies or not other.dependencies:
            return 1.0
        
        # 检查依赖是否都存在
        missing_deps = set(self.dependencies) - set(other.dependencies)
        extra_deps = set(other.dependencies) - set(self.dependencies)
        
        if missing_deps:
            return 0.3  # 缺少依赖
        
        if extra_deps:
            return 0.8  # 有额外依赖，可能兼容
        
        return 1.0  # 依赖完全匹配
    
    def _check_api_compatibility(self, other: 'Component') -> float:
        """检查API兼容性"""
        try:
            # 检查关键API接口
            current_apis = set(self.content.get("apis", []))
            other_apis = set(other.content.get("apis", []))
            
            if not current_apis and not other_apis:
                return 1.0
            
            # 计算API重叠度
            intersection = current_apis & other_apis
            union = current_apis | other_apis
            
            if union:
                return len(intersection) / len(union)
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _check_config_compatibility(self, other: 'Component') -> float:
        """检查配置兼容性"""
        try:
            current_config = self.content.get("config", {})
            other_config = other.content.get("config", {})
            
            if not current_config and not other_config:
                return 1.0
            
            # 检查配置键的匹配度
            current_keys = set(current_config.keys())
            other_keys = set(other_config.keys())
            
            # 计算配置相似度
            intersection = current_keys & other_keys
            union = current_keys | other_keys
            
            if union:
                key_similarity = len(intersection) / len(union)
                
                # 检查值的相似度
                value_similarity = 0.0
                common_keys = intersection
                
                for key in common_keys:
                    if current_config[key] == other_config[key]:
                        value_similarity += 1.0
                
                if common_keys:
                    value_similarity /= len(common_keys)
                
                return (key_similarity * 0.6 + value_similarity * 0.4)
            
            return 0.6
            
        except Exception:
            return 0.6
    
    def get_overall_score(self) -> float:
        """获取综合评分"""
        performance_score = np.mean(list(self.performance_metrics.values())) if self.performance_metrics else 0.5
        quality_score = np.mean(list(self.quality_metrics.values())) if self.quality_metrics else 0.5
        
        return (performance_score * 0.6 + quality_score * 0.4)


class ConflictDetector:
    """冲突检测器"""
    
    def __init__(self):
        self.conflict_patterns = []
        self.resolution_strategies = {}
    
    def detect_conflicts(self, old_components: List[Component], 
                          new_components: List[Component]) -> List[Dict[str, Any]]:
        """检测合并冲突"""
        conflicts = []
        
        try:
            # 检测组件ID冲突
            conflicts.extend(self._detect_id_conflicts(old_components, new_components))
            
            # 检测依赖冲突
            conflicts.extend(self._detect_dependency_conflicts(old_components, new_components))
            
            # 检测配置冲突
            conflicts.extend(self._detect_config_conflicts(old_components, new_components))
            
            # 检测API冲突
            conflicts.extend(self._detect_api_conflicts(old_components, new_components))
            
            # 检测版本冲突
            conflicts.extend(self._detect_version_conflicts(old_components, new_components))
            
        except Exception as e:
            logger.error(f"冲突检测失败: {e}")
        
        return conflicts
    
    def _detect_id_conflicts(self, old_components: List[Component], 
                             new_components: List[Component]) -> List[Dict[str, Any]]:
        """检测组件ID冲突"""
        conflicts = []
        
        old_ids = {comp.id: comp for comp in old_components}
        new_ids = {comp.id: comp for comp in new_components}
        
        # 检测重复ID
        duplicate_ids = old_ids.keys() & new_ids.keys()
        
        for comp_id in duplicate_ids:
            old_comp = old_ids[comp_id]
            new_comp = new_ids[comp_id]
            
            conflicts.append({
                "type": "id_conflict",
                "component_id": comp_id,
                "old_component": {
                    "name": old_comp.name,
                    "version": old_comp.version,
                    "type": old_comp.type.value
                },
                "new_component": {
                    "name": new_comp.name,
                    "version": new_comp.version,
                    "type": new_comp.type.value
                },
                "severity": "medium",
                "description": f"组件ID {comp_id} 在新旧系统中都存在"
            })
        
        return conflicts
    
    def _detect_dependency_conflicts(self, old_components: List[Component], 
                                  new_components: List[Component]) -> List[Dict[str, Any]]:
        """检测依赖冲突"""
        conflicts = []
        
        # 创建依赖图
        old_deps = {comp.id: set(comp.dependencies) for comp in old_components}
        new_deps = {comp.id: set(comp.dependencies) for comp in new_components}
        
        # 检测循环依赖
        all_deps = {**old_deps, **new_deps}
        
        for comp_id, deps in all_deps.items():
            for dep_id in deps:
                if dep_id in all_deps and comp_id in all_deps[dep_id]:
                    conflicts.append({
                        "type": "circular_dependency",
                        "component_id": comp_id,
                        "dependency": dep_id,
                        "severity": "high",
                        "description": f"检测到循环依赖: {comp_id} -> {dep_id} -> {comp_id}"
                    })
        
        # 检测缺失依赖
        for comp_id, deps in new_deps.items():
            missing_deps = deps - set(all_deps.keys())
            for missing_dep in missing_deps:
                conflicts.append({
                    "type": "missing_dependency",
                    "component_id": comp_id,
                    "dependency": missing_dep,
                    "severity": "high",
                    "description": f"组件 {comp_id} 依赖的 {missing_dep} 不存在"
                })
        
        return conflicts
    
    def _detect_config_conflicts(self, old_components: List[Component], 
                             new_components: List[Component]) -> List[Dict[str, Any]]:
        """检测配置冲突"""
        conflicts = []
        
        for new_comp in new_components:
            # 查找对应的旧组件
            old_comp = next((c for c in old_components if c.id == new_comp.id), None)
            
            if old_comp:
                old_config = old_comp.content.get("config", {})
                new_config = new_comp.content.get("config", {})
                
                # 检测配置值冲突
                for key, new_value in new_config.items():
                    if key in old_config and old_config[key] != new_value:
                        conflicts.append({
                            "type": "config_conflict",
                            "component_id": new_comp.id,
                            "config_key": key,
                            "old_value": old_config[key],
                            "new_value": new_value,
                            "severity": "medium",
                            "description": f"配置项 {key} 值不匹配: {old_config[key]} vs {new_value}"
                        })
        
        return conflicts
    
    def _detect_api_conflicts(self, old_components: List[Component], 
                           new_components: List[Component]) -> List[Dict[str, Any]]:
        """检测API冲突"""
        conflicts = []
        
        for new_comp in new_components:
            old_comp = next((c for c in old_components if c.id == new_comp.id), None)
            
            if old_comp:
                old_apis = set(old_comp.content.get("apis", []))
                new_apis = set(new_comp.content.get("apis", []))
                
                # 检测移除的API
                removed_apis = old_apis - new_apis
                for api in removed_apis:
                    conflicts.append({
                        "type": "api_removal",
                        "component_id": new_comp.id,
                        "api": api,
                        "severity": "medium",
                        "description": f"API {api} 在新版本中被移除"
                    })
                
                # 检测修改的API
                common_apis = old_apis & new_apis
                for api in common_apis:
                    old_signature = old_comp.content.get("api_signatures", {}).get(api)
                    new_signature = new_comp.content.get("api_signatures", {}).get(api)
                    
                    if old_signature != new_signature:
                        conflicts.append({
                            "type": "api_signature_change",
                            "component_id": new_comp.id,
                            "api": api,
                            "old_signature": old_signature,
                            "new_signature": new_signature,
                            "severity": "high",
                            "description": f"API {api} 签名发生变化"
                        })
        
        return conflicts
    
    def _detect_version_conflicts(self, old_components: List[Component], 
                               new_components: List[Component]) -> List[Dict[str, Any]]:
        """检测版本冲突"""
        conflicts = []
        
        for new_comp in new_components:
            old_comp = next((c for c in old_components if c.id == new_comp.id), None)
            
            if old_comp:
                old_version = old_comp.version
                new_version = new_comp.version
                
                # 检查版本回退
                if self._is_version_rollback(old_version, new_version):
                    conflicts.append({
                        "type": "version_rollback",
                        "component_id": new_comp.id,
                        "old_version": old_version,
                        "new_version": new_version,
                        "severity": "high",
                        "description": f"版本从 {old_version} 回退到 {new_version}"
                    })
                
                # 检查版本跳跃
                if self._is_version_jump(old_version, new_version):
                    conflicts.append({
                        "type": "version_jump",
                        "component_id": new_comp.id,
                        "old_version": old_version,
                        "new_version": new_version,
                        "severity": "medium",
                        "description": f"版本从 {old_version} 跳跃到 {new_version}"
                    })
        
        return conflicts
    
    def _is_version_rollback(self, old_version: str, new_version: str) -> bool:
        """检查是否为版本回退"""
        try:
            old_parts = [int(part) for part in old_version.split('.')]
            new_parts = [int(part) for part in new_version.split('.')]
            
            # 简化的回退检测
            return (len(new_parts) >= 2 and 
                    new_parts[0] == old_parts[0] and 
                    new_parts[1] < old_parts[1])
        except (ValueError, IndexError):
            return False
    
    def _is_version_jump(self, old_version: str, new_version: str) -> bool:
        """检查是否为版本跳跃"""
        try:
            old_parts = [int(part) for part in old_version.split('.')]
            new_parts = [int(part) for part in new_version.split('.')]
            
            # 简化的跳跃检测
            return (len(new_parts) >= 2 and 
                    new_parts[1] > old_parts[1] + 1)
        except (ValueError, IndexError):
            return False


class ComponentEvaluator:
    """组件评估器"""
    
    def __init__(self):
        self.evaluation_weights = {
            "performance": 0.4,
            "quality": 0.3,
            "compatibility": 0.2,
            "maintainability": 0.1
        }
        self.performance_benchmarks = {
            "response_time": {"excellent": 100, "good": 500, "poor": 2000},
            "throughput": {"excellent": 1000, "good": 500, "poor": 100},
            "memory_usage": {"excellent": 512, "good": 1024, "poor": 2048},
            "cpu_usage": {"excellent": 50, "good": 70, "poor": 90}
        }
        self.quality_benchmarks = {
            "code_coverage": {"excellent": 90, "good": 80, "poor": 60},
            "cyclomatic_complexity": {"excellent": 10, "good": 20, "poor": 50},
            "maintainability_index": {"excellent": 85, "good": 70, "poor": 50},
            "technical_debt": {"excellent": 5, "good": 15, "poor": 30}
        }
    
    def evaluate_component(self, component: Component) -> Dict[str, Any]:
        """评估组件"""
        try:
            # 性能评估
            performance_score = self._evaluate_performance(component)
            
            # 质量评估
            quality_score = self._evaluate_quality(component)
            
            # 兼容性评估
            compatibility_score = self._evaluate_compatibility(component)
            
            # 可维护性评估
            maintainability_score = self._evaluate_maintainability(component)
            
            # 综合评分
            overall_score = (
                performance_score * self.evaluation_weights["performance"] +
                quality_score * self.evaluation_weights["quality"] +
                compatibility_score * self.evaluation_weights["compatibility"] +
                maintainability_score * self.evaluation_weights["maintainability"]
            )
            
            return {
                "component_id": component.id,
                "overall_score": overall_score,
                "performance_score": performance_score,
                "quality_score": quality_score,
                "compatibility_score": compatibility_score,
                "maintainability_score": maintainability_score,
                "evaluation_details": {
                    "performance_metrics": component.performance_metrics,
                    "quality_metrics": component.quality_metrics
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"组件评估失败: {e}")
            return {
                "component_id": component.id,
                "overall_score": 0.0,
                "error": str(e)
            }
    
    def _evaluate_performance(self, component: Component) -> float:
        """评估性能"""
        if not component.performance_metrics:
            return 0.5
        
        scores = []
        
        for metric_name, metric_value in component.performance_metrics.items():
            if metric_name in self.performance_benchmarks:
                benchmarks = self.performance_benchmarks[metric_name]
                score = self._calculate_benchmark_score(metric_value, benchmarks)
                scores.append(score)
        
        return np.mean(scores) if scores else 0.5
    
    def _evaluate_quality(self, component: Component) -> float:
        """评估质量"""
        if not component.quality_metrics:
            return 0.5
        
        scores = []
        
        for metric_name, metric_value in component.quality_metrics.items():
            if metric_name in self.quality_benchmarks:
                benchmarks = self.quality_benchmarks[metric_name]
                score = self._calculate_benchmark_score(metric_value, benchmarks, reverse=True)
                scores.append(score)
        
        return np.mean(scores) if scores else 0.5
    
    def _evaluate_compatibility(self, component: Component) -> float:
        """评估兼容性"""
        if not component.compatibility_matrix:
            return 0.8
        
        # 计算平均兼容性
        compatibilities = list(component.compatibility_matrix.values())
        return np.mean(compatibilities) if compatibilities else 0.8
    
    def _evaluate_maintainability(self, component: Component) -> float:
        """评估可维护性"""
        # 基于组件元数据评估可维护性
        metadata = component.metadata
        
        score = 0.5
        
        # 检查文档完整性
        if metadata.get("has_documentation", False):
            score -= 0.2
        
        # 检查测试覆盖率
        if metadata.get("test_coverage", 0) < 80:
            score -= 0.1
        
        # 检查代码复杂度
        if metadata.get("complexity", "low") != "low":
            score -= 0.1
        
        # 检查依赖数量
        if len(component.dependencies) > 10:
            score -= 0.1
        
        return max(0.0, score)
    
    def _calculate_benchmark_score(self, value: float, benchmarks: Dict[str, float], reverse: bool = False) -> float:
        """计算基准评分"""
        excellent = benchmarks["excellent"]
        good = benchmarks["good"]
        poor = benchmarks["poor"]
        
        if reverse:
            if value >= excellent:
                return 1.0
            elif value >= good:
                return 0.7
            else:
                return 0.3
        else:
            if value <= poor:
                return 1.0
            elif value <= good:
                return 0.7
            else:
                return 0.3


class SelectiveMergeStrategy:
    """择优合并策略"""
    
    def __init__(self):
        self.conflict_detector = ConflictDetector()
        self.evaluator = ComponentEvaluator()
        self.merge_history: List[Dict[str, Any]] = []
        self.rollback_snapshots: Dict[str, Any] = {}
        self.current_strategy = MergeStrategy.BALANCED
        
    def selective_merge(self, old_system: Dict[str, Component], 
                         new_system: Dict[str, Component],
                         strategy: Optional[MergeStrategy] = None,
                         system_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """择优合并系统"""
        if strategy:
            self.current_strategy = strategy
        
        logger.info(f"开始择优合并，策略: {self.current_strategy.value}")
        
        merge_session = {
            "session_id": f"merge_{int(datetime.now().timestamp())}",
            "strategy": self.current_strategy.value,
            "start_time": datetime.now(),
            "old_system": {comp.id: comp for comp in old_system.values()},
            "new_system": {comp.id: comp for comp in new_system.values()},
            "results": {},
            "status": MergeStatus.PENDING
        }
        
        try:
            # 1. 冲突检测
            merge_session["status"] = MergeStatus.ANALYZING
            conflicts = self.conflict_detector.detect_conflicts(
                list(old_system.values()),
                list(new_system.values())
            )
            
            if conflicts:
                merge_session["conflicts"] = conflicts
                logger.warning(f"检测到 {len(conflicts)} 个冲突")
                
                # 尝试自动解决冲突
                resolved_conflicts = self._auto_resolve_conflicts(conflicts)
                if not resolved_conflicts:
                    merge_session["status"] = MergeStatus.CONFLICT
                    return merge_session
            
            # 2. 组件评估
            merge_session["status"] = MergeStatus.ANALYZING
            evaluation_results = self._evaluate_components(old_system, new_system)
            
            # 3. 组件选择
            merge_session["status"] = MergeStatus.ANALYZING
            selection_results = self._select_components(
                old_system, new_system, evaluation_results, strategy
            )
            
            # 4. 执行合并
            merge_session["status"] = MergeStatus.MERGING
            merged_system = self._execute_merge(
                old_system, new_system, selection_results
            )
            
            # 5. 验证合并结果
            merge_session["status"] = MergeStatus.COMPLETED
            validation_results = self._validate_merge(merged_system)
            
            # 6. 生成结果
            merge_session["results"] = {
                "selection_results": selection_results,
                "evaluation_results": evaluation_results,
                "validation_results": validation_results,
                "merged_system": {comp.id: comp for comp in merged_system.values()}
            }
            
            # 保存合并历史
            self._save_merge_history(merge_session)
            
            logger.info("择优合并完成")
            return merge_session
            
        except Exception as e:
            logger.error(f"择优合并失败: {e}")
            merge_session["status"] = MergeStatus.FAILED
            merge_session["error"] = str(e)
            return merge_session
    
    def _auto_resolve_conflicts(self, conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """自动解决冲突"""
        resolved_conflicts = []
        
        for conflict in conflicts:
            try:
                resolution = self._resolve_conflict(conflict)
                if resolution:
                    resolved_conflicts.append(resolution)
                else:
                    logger.warning(f"无法自动解决冲突: {conflict['description']}")
            except Exception as e:
                logger.error(f"解决冲突失败: {e}")
        
        return resolved_conflicts
    
    def _resolve_conflict(self, conflict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """解决单个冲突"""
        conflict_type = conflict["type"]
        
        if conflict_type == "id_conflict":
            # ID冲突：保留版本更新的组件
            return {
                "type": "id_conflict",
                "resolution": "keep_newer_version",
                "action": "replace_with_new_component",
                "reason": "保留版本更新的组件"
            }
        
        elif conflict_type == "missing_dependency":
            # 缺失依赖：添加依赖或移除依赖项
            return {
                "type": "missing_dependency",
                "resolution": "add_dependency",
                "action": "add_missing_dependency",
                "reason": "添加缺失的依赖"
            }
        
        elif conflict_type == "config_conflict":
            # 配置冲突：保留新配置并记录
            return {
                "type": "create_backup",
                "resolution": "create_config_backup",
                "action": "backup_old_config",
                "reason": "创建配置备份并使用新配置"
            }
        
        elif conflict_type == "api_removal":
            # API移除：检查是否可以安全移除
            return {
                "type": "api_removal",
                "resolution": "check_removal_safety",
                "action": "validate_removal",
                "reason": "验证API移除的安全性"
            }
        
        return None
    
    def _evaluate_components(self, old_system: Dict[str, Component], 
                           new_system: Dict[str, Component]) -> Dict[str, Any]:
        """评估组件"""
        evaluation_results = {
            "old_system": {},
            "new_system": {},
            "comparison": {}
        }
        
        # 评估旧系统组件
        for comp_id, component in old_system.items():
            evaluation = self.evaluator.evaluate_component(component)
            evaluation_results["old_system"][comp_id] = evaluation
        
        # 评估新系统组件
        for comp_id, component in new_system.items():
            evaluation = self.evaluator.evaluate_component(component)
            evaluation_results["new_system"][comp_id] = evaluation
        
        # 生成对比结果
        for comp_id in set(old_system.keys()) & set(new_system.keys()):
            old_eval = evaluation_results["old_system"][comp_id]
            new_eval = evaluation_results["new_system"][comp_id]
            
            evaluation_results["comparison"][comp_id] = {
                "old_score": old_eval["overall_score"],
                "new_score": new_eval["overall_score"],
                "improvement": new_eval["overall_score"] - old_eval["overall_score"],
                "recommendation": self._get_merge_recommendation(old_eval, new_eval)
            }
        
        return evaluation_results
    
    def _get_merge_recommendation(self, old_eval: Dict[str, Any], new_eval: Dict[str, Any]) -> str:
        """获取合并建议"""
        improvement = new_eval["overall_score"] - old_eval["overall_score"]
        
        if improvement > 20:
            return "strongly_recommend_merge"
        elif improvement > 10:
            return "recommend_merge"
        elif improvement > 0:
            return "consider_merge"
        elif improvement == 0:
            return "neutral"
        else:
            return "keep_old"
    
    def _select_components(self, old_system: Dict[str, Component], 
                       new_system: Dict[str, Component],
                       evaluation_results: Dict[str, Any],
                       strategy: MergeStrategy) -> Dict[str, Any]:
        """选择组件"""
        selection_results = {
            "selected_components": {},
            "rejected_components": {},
            "selection_reasons": {}
        }
        
        all_component_ids = set(old_system.keys()) | set(new_system.keys())
        
        for comp_id in all_component_ids:
            old_comp = old_system.get(comp_id)
            new_comp = new_system.get(comp_id)
            
            if strategy == MergeStrategy.BEST_PERFORMANCE:
                selected_comp = self._select_by_performance(old_comp, new_comp)
            elif strategy == MergeStrategy.BEST_QUALITY:
                selected_comp = self._select_by_quality(old_comp, new_comp)
            elif strategy == MergeStrategy.BEST_COMPATIBILITY:
                selected_comp = self._select_by_compatibility(old_comp, new_comp)
            elif strategy == MergeStrategy.BALANCED:
                selected_comp = self._select_balanced(old_comp, new_comp)
            elif strategy == MergeStrategy.CONSERVATIVE:
                selected_comp = self._select_conservative(old_comp, new_comp)
            elif strategy == MergeStrategy.AGGRESSIVE:
                selected_comp = self._select_aggressive(old_comp, new_comp)
            else:
                selected_comp = self._select_balanced(old_comp, new_comp)
            
            if selected_comp:
                selection_results["selected_components"][comp_id] = selected_comp
                selection_results["selection_reasons"][comp_id] = self._get_selection_reason(
                    old_comp, new_comp, selected_comp, strategy
                )
            else:
                # 默认保留旧组件
                if old_comp:
                    selection_results["selected_components"][comp_id] = old_comp
                    selection_results["selection_reasons"][comp_id] = "默认保留旧组件"
                elif new_comp:
                    selection_results["selected_components"][comp_id] = new_comp
                    selection_results["selection_reasons"][comp_id] = "默认使用新组件"
        
        return selection_results
    
    def _select_by_performance(self, old_comp: Optional[Component], 
                             new_comp: Optional[Component]) -> Optional[Component]:
        """按性能选择"""
        if not old_comp:
            return new_comp
        if not new_comp:
            return old_comp
        
        old_score = old_comp.get_overall_score()
        new_score = new_comp.get_overall_score()
        
        return new_comp if new_score > old_score else old_comp
    
    def _select_by_quality(self, old_comp: Optional[Component], 
                         new_comp: Optional[Component]) -> Optional[Component]:
        """按质量选择"""
        if not old_comp:
            return new_comp
        if not new_comp:
            return old_comp
        
        old_score = old_comp.get_overall_score()
        new_score = new_comp.get_overall_score()
        
        return new_comp if new_score > old_score else old_comp
    
    def _select_by_compatibility(self, old_comp: Optional[Component], 
                              new_comp: Optional[Component]) -> Optional[Component]:
        """按兼容性选择"""
        if not old_comp:
            return new_comp
        if not new_comp:
            return old_comp
        
        # 优先选择兼容性更好的组件
        old_compat = np.mean(list(old_comp.compatibility_matrix.values())) if old_comp.compatibility_matrix else 0.5
        new_compat = np.mean(list(new_comp.compatibility_matrix.values())) if new_comp.compatibility_matrix else 0.5
        
        return new_comp if new_compat > old_compat else old_comp
    
    def _select_balanced(self, old_comp: Optional[Component], 
                        new_comp: Optional[Component]) -> Optional[Component]:
        """平衡选择"""
        if not old_comp:
            return new_comp
        if not new_comp:
            return old_comp
        
        old_score = old_comp.get_overall_score()
        new_score = new_comp.get_overall_score()
        
        # 综合考虑性能、质量和兼容性
        old_total = (
            old_score * 0.4 +
            old_comp.quality_score * 0.3 +
            np.mean(list(old_comp.compatibility_matrix.values())) * 0.3
        )
        
        new_total = (
            new_score * 0.4 +
            new_comp.quality_score * 0.3 +
            np.mean(list(new_comp.compatibility_matrix.values())) * 0.3
        )
        
        return new_comp if new_total > old_total else old_comp
    
    def _select_conservative(self, old_comp: Optional[Component], 
                         new_comp: Optional[Component]) -> Optional[Component]:
        """保守选择"""
        # 保守策略：优先保留旧组件
        return old_comp if old_comp else new_comp
    
    def _select_aggressive(self, old_comp: Optional[Component], 
                          new_comp: Optional[Component]) -> Optional[Component]:
        """激进选择"""
        # 激进策略：优先选择新组件
        return new_comp if new_comp else old_comp
    
    def _get_selection_reason(self, old_comp: Optional[Component], 
                           new_comp: Optional[Component],
                           selected_comp: Component, 
                           strategy: MergeStrategy) -> str:
        """获取选择原因"""
        if selected_comp == new_comp:
            if not old_comp:
                return "新系统中存在该组件"
            else:
                old_score = old_comp.get_overall_score()
                new_score = new_comp.get_overall_score()
                
                if new_score > old_score:
                    return f"新组件性能更优 ({new_score:.2f} vs {old_score:.2f})"
                else:
                    return f"新组件其他指标更优"
        else:
            if not new_comp:
                return "仅旧系统中存在该组件"
            else:
                old_score = old_comp.get_overall_score()
                new_score = new_comp.get_overall_score()
                
                if old_score >= new_score:
                    return f"旧组件性能更优或相等 ({old_score:.2f} vs {new_score:.2f})"
                else:
                    return f"保守选择，保留旧组件"
    
    def _execute_merge(self, old_system: Dict[str, Component], 
                     new_system: Dict[str, Component],
                     selection_results: Dict[str, Any]) -> Dict[str, Component]:
        """执行合并"""
        merged_system = {}
        
        for comp_id, selected_comp in selection_results["selected_components"].items():
            # 创建组件副本
            merged_component = Component(
                id=selected_comp.id,
                name=selected_comp.name,
                type=selected_comp.type,
                version=selected_comp.version,
                content=copy.deepcopy(selected_comp.content),
                dependencies=selected_comp.dependencies.copy(),
                metadata=copy.deepcopy(selected_comp.metadata),
                performance_metrics=selected_comp.performance_metrics.copy(),
                quality_metrics=selected_comp.quality_metrics.copy(),
                compatibility_matrix=copy.deepcopy(selected_comp.compatibility_matrix)
            )
            
            # 更新依赖关系
            merged_system[comp_id] = merged_component
        
        return merged_system
    
    def _validate_merge(self, merged_system: Dict[str, Component]) -> Dict[str, Any]:
        """验证合并结果"""
        validation_results = {
            "total_components": len(merged_system),
            "valid_components": 0,
            "invalid_components": 0,
            "dependency_issues": 0,
            "validation_details": {}
        }
        
        try:
            # 验证每个组件
            for comp_id, component in merged_system.items():
                component_validation = self._validate_component(component)
                
                if component_validation["valid"]:
                    validation_results["valid_components"] += 1
                else:
                    validation_results["invalid_components"] += 1
                
                validation_results["validation_details"][comp_id] = component_validation
            
            # 验证依赖关系
            dependency_graph = self._build_dependency_graph(merged_system)
            cycles = list(nx.simple_cycles(dependency_graph))
            
            if cycles:
                validation_results["dependency_issues"] = len(cycles)
                validation_results["validation_details"]["cycles"] = cycles
            
            # 计算总体有效性
            validation_results["validity_score"] = (
                validation_results["valid_components"] / validation_results["total_components"]
            )
            
        except Exception as e:
            logger.error(f"验证合并结果失败: {e}")
            validation_results["error"] = str(e)
        
        return validation_results
    
    def _validate_component(self, component: Component) -> Dict[str, Any]:
        """验证单个组件"""
        validation = {
            "valid": True,
            "issues": [],
            "warnings": []
        }
        
        # 检查校验和
        if not component.checksum == component._calculate_checksum():
            validation["valid"] = False
            validation["issues"].append("校验和不匹配")
        
        # 检查依赖存在性
        for dep_id in component.dependencies:
            if dep_id not in component.dependencies:
                validation["issues"].append(f"依赖 {dep_id} 不存在")
                validation["valid"] = False
        
        # 检查配置完整性
        if not component.content:
            validation["issues"].append("组件内容为空")
            validation["valid"] = False
        
        return validation
    
    def _build_dependency_graph(self, components: Dict[str, Component]) -> nx.DiGraph:
        """构建依赖图"""
        graph = nx.DiGraph()
        
        # 添加节点
        for comp_id, component in components.items():
            graph.add_node(comp_id, component=component)
        
        # 添加边
        for comp_id, component in components.items():
            for dep_id in component.dependencies:
                if dep_id in components:
                    graph.add_edge(comp_id, dep_id)
        
        return graph
    
    def create_rollback_snapshot(self, system: Dict[str, Component]) -> str:
        """创建回滚快照"""
        snapshot_id = f"snapshot_{int(datetime.now().timestamp())}"
        
        snapshot_data = {
            "snapshot_id": snapshot_id,
            "timestamp": datetime.now().isoformat(),
            "system": {
                comp_id: {
                    "id": comp.id,
                    "name": comp.name,
                    "type": comp.type.value,
                    "version": comp.version,
                    "checksum": comp.checksum,
                    "content": comp.content,
                    "dependencies": comp.dependencies,
                    "metadata": comp.metadata,
                    "performance_metrics": comp.performance_metrics,
                    "quality_metrics": comp.quality_metrics,
                    "compatibility_matrix": comp.compatibility_matrix
                }
                for comp_id, comp in system.items()
            }
        }
        
        self.rollback_snapshots[snapshot_id] = snapshot_data
        logger.info(f"创建回滚快照: {snapshot_id}")
        
        return snapshot_id
    
    def rollback_to_snapshot(self, snapshot_id: str) -> bool:
        """回滚到指定快照"""
        if snapshot_id not in self.rollback_snapshots:
            logger.error(f"快照不存在: {snapshot_id}")
            return False
        
        try:
            snapshot_data = self.rollback_snapshots[snapshot_id]
            
            # 恢复系统状态
            restored_system = {}
            for comp_id, comp_data in snapshot_data["system"].items():
                component = Component(
                    id=comp_data["id"],
                    name=comp_data["name"],
                    type=ComponentType(comp_data["type"]),
                    version=comp_data["version"],
                    content=comp_data["content"],
                    dependencies=comp_data["dependencies"],
                    metadata=comp_data["metadata"],
                    performance_metrics=comp_data["performance_metrics"],
                    quality_metrics=comp_data["quality_metrics"],
                    compatibility_matrix=comp_data["compatibility_matrix"]
                )
                restored_system[comp_id] = component
            
            # 保存当前状态为快照
            current_snapshot_id = self.create_rollback_snapshot(restored_system)
            
            logger.info(f"成功回滚到快照: {snapshot_id}")
            return True
            
        except Exception as e:
            logger.error(f"回滚失败: {e}")
            return False
    
    def _save_merge_history(self, merge_session: Dict[str, Any]):
        """保存合并历史"""
        try:
            merge_history_file = Path("./data/merge_history.json")
            merge_history_file.parent.mkdir(parents=True, exist_ok=True)
            
            self.merge_history.append(merge_session)
            
            # 保持历史记录在合理范围内
            if len(self.merge_history) > 100:
                self.merge_history = self.merge_history[-50:]
            
            with open(merge_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.merge_history, f, indent=2, ensure_ascii=False, default=str)
            
        except Exception as e:
            logger.error(f"保存合并历史失败: {e}")
    
    def get_merge_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取合并历史"""
        return self.merge_history[-limit:]
    
    def generate_merge_report(self, merge_session: Dict[str, Any], 
                           format_type: str = "json") -> str:
        """生成合并报告"""
        try:
            if format_type == "json":
                return json.dumps(merge_session, indent=2, ensure_ascii=False, default=str)
            elif format_type == "html":
                return self._generate_html_report(merge_session)
            elif format_type == "markdown":
                return self._generate_markdown_report(merge_session)
            else:
                raise ValueError(f"不支持的报告格式: {format_type}")
        except Exception as e:
            logger.error(f"生成合并报告失败: {e}")
            return ""
    
    def _generate_html_report(self, data: Dict[str, Any]) -> str:
        """生成HTML报告"""
        template = """
<!DOCTYPE html>
<html>
<head>
    <title>择优合并报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .metric { display: inline-block; margin: 5px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }
        .success { background-color: #d4edda; }
        .warning { background-color: #fff3cd; }
        .error { background-color: #f8d7da; }
        .improvement { font-size: 1.2em; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .pass { color: #155724; }
        .fail { color: #dc3545; }
        .neutral { color: #6c757d; }
    </style>
</head>
<body>
    <div class="header">
        <h1>择优合并报告</h1>
        <p>会话ID: {session_id}</p>
        <p>合并策略: {strategy}</p>
        <p>开始时间: {start_time}</p>
        <p>结束时间: {end_time}</p>
        <p class="improvement">总体改进: {overall_improvement:.2f}%</p>
    </div>
        
    {sections_html}
    
    <div class="section">
        <h2>合并统计</h2>
        <p>选中组件数: {selected_count}</p>
        <p>拒绝组件数: {rejected_count}</p>
        <p>验证通过率: {validation_rate:.2f}%</p>
    </div>
</body>
</html>
        """
        
        # 生成各部分HTML
        sections_html = ""
        
        # 添加选择结果
        selection_results = data["results"]["selection_results"]
        sections_html += f"""
    <div class="section">
        <h2>组件选择结果</h2>
        <table>
            <tr><th>组件ID</th><th>选择结果</th><th>选择原因</th></tr>
        """
        
        for comp_id, comp in selection_results["selected_components"].items():
            sections_html += f"""
            <tr>
                <td>{comp_id}</td>
                <td class="pass">已选择</td>
                <td>{selection_results["selection_reasons"][comp_id]}</td>
            </tr>
            """
        
        # 添加拒绝结果
        if "rejected_components" in selection_results:
            for comp_id in selection_results["rejected_components"].items():
                sections_html += f"""
            <tr>
                <td>{comp_id}</td>
                <td class="neutral">已拒绝</td>
                <td>{selection_results["selection_reasons"][comp_id]}</td>
            </tr>
            """
        
        sections_html += "</table>\n    </div>\n"
        
        # 添加评估结果
        evaluation_results = data["results"]["evaluation_results"]
        sections_html += f"""
    <div class="section">
        <h2>组件评估结果</h2>
        <h3>旧系统评估</h3>
        <table>
            <tr><th>组件ID</th><th>总分</th><th>性能分数</th><th>质量分数</th><th>兼容性分数</th></tr>
        """
        
        for comp_id, eval_result in evaluation_results["old_system"].items():
            sections_html += f"""
            <tr>
                <td>{comp_id}</td>
                <td>{eval_result["overall_score"]:.2f}</td>
                <td>{eval_result["performance_score"]:.2f}</td>
                <td>{eval_result["quality_score"]:.2f}</td>
                <td>{eval_result["compatibility_score"]:.2f}</td>
            </tr>
            """
        
        sections_html += """
        <h3>新系统评估</h3>
        <table>
            <tr><th>组件ID</th><th>总分</th><th>性能分数</th><th>质量分数</th><th>兼容性分数</th></tr>
        """
        
        for comp_id, eval_result in evaluation_results["new_system"].items():
            sections_html += f"""
            <tr>
                <td>{comp_id}</td>
                <td>{eval_result["overall_score"]:.2f}</td>
                <td>{eval_result["performance_score"]:.2f}</td>
                <td>{eval_result["quality_score"]:.2f}</td>
                <td>{eval_result["compatibility_score"]}.2f}</td>
            </tr>
            """
        
        sections_html += "</table>\n    </div>\n"
        
        # 添加对比结果
        comparison_results = data["results"]["evaluation_results"]["comparison"]
        sections_html += f"""
    <div class="section">
        <h2>对比结果</h2>
        <table>
            <tr><th>组件ID</th><th>旧系统分数</th><th>新系统分数</th><th>改进幅度</th><th>建议</th></tr>
        """
        
        for comp_id, comparison in comparison_results.items():
            improvement = comparison["improvement"]
            
            status_class = "pass" if improvement > 0 else ("fail" if improvement < -10 else "neutral")
            
            sections_html += f"""
            <tr>
                <td>{comp_id}</td>
                <td>{comparison["old_score"]:.2f}</td>
                <td>{comparison["new_score"]:.2f}</td>
                <td class="{status_class}">{improvement:+.2f}%</td>
                <td>{comparison["recommendation"]}</td>
            </tr>
            """
        
        sections_html += "</table>\n    </div>\n"
        
        # 添加验证结果
        validation_results = data["results"]["validation_results"]
        sections_html += f"""
    <div class="section">
        <h2>验证结果</h2>
        <p>总组件数: {validation_results["total_components"]}</p>
        <p>有效组件数: {validation_results["valid_components"]}</p>
        <p>无效组件数: {validation_results["invalid_components"]}</p>
        <p>依赖问题数: {validation_results["dependency_issues"]}</p>
        <p>有效性评分: {validation_results["validity_score"]:.2f}</p>
        </div>
        """
        
        return template.format(
            session_id=data["session_id"],
            strategy=data["strategy"],
            start_time=data["start_time"],
            end_time=data.get("end_time", ""),
            overall_improvement=data["overall_improvement"],
            selected_count=len(selection_results["selected_components"]),
            rejected_count=len(selection_results.get("rejected_components", {})),
            validation_rate=validation_results["validity_score"] * 100 if validation_results else 0,
            sections_html=sections_html
        )
    
    def _generate_markdown_report(self, data: Dict[str, Any]) -> str:
        """生成Markdown报告"""
        template = """
# 择优合并报告

## 基本信息
- **会话ID**: {session_id}
- **合并策略**: {strategy}
- **开始时间**: {start_time}
- **结束时间**: {end_time}
- **总体改进**: {overall_improvement:.2f}%

## 组件选择结果

### 已选择的组件
{selected_components_table}

### 被拒绝的组件
{rejected_components_table}

## 组件评估结果

### 旧系统评估
{old_system_table}

### 新系统评估
{new_system_table}

### 对比结果
{comparison_table}

## 验证结果

- 总组件数: {total_components}
- 有效组件数: {valid_components}
- 无效组件数: {invalid_components}
- 依赖问题数: {dependency_issues}
- 有效性评分: {validity_score:.2f}
        """
        
        # 生成表格
        selected_components_table = ""
        for comp_id, comp in data["results"]["selection_results"]["selected_components"].items():
            selected_components_table += f"| {comp_id} | 已选择 | {data['results']['selection_results']['selection_reasons'][comp_id]}\n"
        
        rejected_components_table = ""
        if "rejected_components" in data["results"]["selection_results"]:
            for comp_id in data["results"]["selection_results"]["rejected_components"].items():
                rejected_components_table += f"| {comp_id} | 已拒绝 | {data['results']['selection_results']['selection_reasons'][comp_id]}\n"
        
        old_system_table = ""
        for comp_id, eval_result in data["results"]["evaluation_results"]["old_system"].items():
            old_system_table += f"| {comp_id} | {eval_result['overall_score']:.2f} | {eval_result['performance_score']:.2f} | {eval_result['quality_score']:.2f} | {eval_result['compatibility_score']:.2f}\n"
        
        new_system_table = ""
        for comp_id, eval_result in data["results"]["evaluation_results"]["new_system"].items():
            new_system_table += f"| {comp_id} | {eval_result['overall_score']:.2f} | {eval_result['performance_score']:.2f} | {eval_result['quality_score']:.2f} | {eval_result['compatibility_score']:.2f}\n"
        
        comparison_table = ""
        for comp_id, comparison in data["results"]["evaluation_results"]["comparison"].items():
            improvement = comparison["improvement"]
            status = "✅" if improvement > 0 else ("❌" if improvement < -10 else "➡️")
            comparison_table += f"| {comp_id} | {comparison['old_score']:.2f} | {comparison['new_score']:.2f} | {status} {improvement:+.2f}% | {comparison['recommendation']}\n"
        
        return template.format(
            session_id=data["session_id"],
            strategy=data["strategy"],
            start_time=data["start_time"],
            end_time=data.get("end_time", ""),
            overall_improvement=data["overall_improvement"],
            selected_components_table=selected_components_table,
            rejected_components_table=rejected_components_table,
            old_system_table=old_system_table,
            new_system_table=new_system_table,
            comparison_table=comparison_table,
            total_components=data["results"]["validation_results"]["total_components"],
            valid_components=data["results"]["validation_results"]["valid_components"],
            invalid_components=data["results"]["validation_results"]["invalid_components"],
            dependency_issues=data["results"]["validation_results"]["dependency_issues"],
            validity_score=data["results"]["validation_results"]["validity_score"]
        )


# 示例使用
async def main():
    """主函数示例"""
    # 创建择优合并策略
    merger = SelectiveMergeStrategy()
    
    # 模拟旧系统组件
    old_components = {
        "user_service": Component(
            id="user_service",
            name="用户服务",
            type=ComponentType.SERVICE,
            version="1.0.0",
            content={
                "version": "1.0.0",
                "timeout": 5000,
                "max_connections": 100
            },
            dependencies=["database", "cache"],
            performance_metrics={"response_time": 500, "throughput": 1000},
            quality_metrics={"code_coverage": 85, "maintainability": 75},
            compatibility_matrix={"database": 0.9, "cache": 0.8}
        ),
        "auth_service": Component(
            id="auth_service",
            name="认证服务",
            type=ComponentType.SERVICE,
            version="1.0.0",
            content={
                "version": "1.0.0",
                "timeout": 3000,
                "max_tokens": 1000
            },
            dependencies=["user_service"],
            performance_metrics={"response_time": 300, "throughput": 1500},
            quality_metrics={"code_coverage": 90, "maintainability": 80},
            compatibility_matrix={"user_service": 0.9}
        )
    }
    
    # 模拟新系统组件
    new_components = {
        "user_service": Component(
            id="user_service",
            name="用户服务",
            type=ComponentType.SERVICE,
            version="2.0.0",
            content={
                "version": "2.0.0",
                "timeout": 2000,
                "max_connections": 200,
                "new_feature": "auto_scaling"
            },
            dependencies=["database", "cache", "load_balancer"],
            performance_metrics={"response_time": 200, "throughput": 2000},
            quality_metrics={"code_coverage": 95, "maintainability": 85},
            compatibility_matrix={"database": 0.9, "cache": 0.8, "load_balancer": 0.9}
        ),
        "auth_service": Component(
            id="auth_service",
            name="认证服务",
            type=ComponentType.SERVICE,
            version="2.0.0",
            content={
                "version": "2.0.0",
                "timeout": 1500,
                "max_tokens": 2000,
                "biometric_auth": True
            },
            dependencies=["user_service", "security_service"],
            performance_metrics={"response_time": 150, "throughput": 2000},
            quality_metrics={"code_coverage": 92, "maintainability": 88},
            compatibility_matrix={"user_service": 0.9, "security_service": 0.9}
        ),
        "notification_service": Component(
            id="notification_service",
            name="通知服务",
            type=ComponentType.SERVICE,
            version="1.0.0",
            content={
                "version": "1.0.0",
                "providers": ["email", "sms", "webhook"]
            },
            dependencies=[],
            performance_metrics={"response_time": 100, "throughput": 500},
            quality_metrics={"code_coverage": 88, "maintainability": 82},
            compatibility_matrix={}
        )
    }
    
    # 执行择优合并
    results = merger.selective_merge(
        old_components,
        new_components,
        MergeStrategy.BALANCED
    )
    
    print("择优合并结果:")
    print(f"会话ID: {results['session_id']}")
    print(f"策略: {results['strategy']}")
    print(f"总体改进: {results['overall_improvement_percentage']:.2f}%")
    print(f"选择组件数: {len(results['results']['selection_results']['selected_components'])}")
    print(f"拒绝组件数: {len(results['results']['selection_results'].get('rejected_components', {}))}")
    print(f"验证通过率: {results['results']['validation_results']['validity_score']:.2f}%")
    
    # 生成报告
    report = merger.generate_merge_report(results, "html", "./merge_report.html")
    print(f"合并报告已生成: ./merge_report.html")


if __name__ == "__main__":
    asyncio.run(main())
