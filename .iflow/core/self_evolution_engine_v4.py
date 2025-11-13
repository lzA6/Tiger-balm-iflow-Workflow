#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌌 自我进化引擎 V4 (Self Evolution Engine V4)
基于机器学习和强化学习的自我进化系统，实现工作流的持续优化和自主学习。
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import asyncio
import logging
import time
import pickle
import uuid
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import defaultdict, deque
import sqlite3
import hashlib
import uuid

# 动态添加项目根目录到sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvolutionPhase(Enum):
    """进化阶段"""
    OBSERVATION = "observation"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    EVALUATION = "evaluation"
    CONSOLIDATION = "consolidation"

class LearningType(Enum):
    """学习类型"""
    SUPERVISED = "supervised"
    REINFORCEMENT = "reinforcement"
    UNSUPERVISED = "unsupervised"
    TRANSFER = "transfer"
    META_LEARNING = "meta_learning"

@dataclass
class EvolutionRecord:
    """进化记录"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    phase: EvolutionPhase = EvolutionPhase.OBSERVATION
    learning_type: LearningType = LearningType.REINFORCEMENT
    context: Dict[str, Any] = field(default_factory=dict)
    observations: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    outcomes: Dict[str, float] = field(default_factory=dict)
    rewards: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    improvements: List[str] = field(default_factory=list)
    next_phase: Optional[EvolutionPhase] = None

@dataclass
class LearningPattern:
    """学习模式"""
    pattern_id: str
    pattern_type: str
    triggers: List[str]
    actions: List[str]
    success_rate: float
    avg_reward: float
    usage_count: int
    last_updated: datetime = field(default_factory=datetime.now)

class SelfEvolutionEngineV4:
    """自我进化引擎 V4"""
    
    def __init__(self, db_path: str = "A项目/iflow/data/evolution_v4.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()
        
        # 进化状态
        self.current_phase = EvolutionPhase.OBSERVATION
        self.evolution_history: deque = deque(maxlen=1000)
        self.learning_patterns: Dict[str, LearningPattern] = {}
        
        # 性能指标
        self.performance_baseline = {}
        self.current_performance = {}
        self.improvement_targets = {}
        
        # 学习参数
        self.learning_rate = 0.01
        self.exploration_rate = 0.1
        self.discount_factor = 0.95
        
        self.lock = threading.RLock()
        logger.info("🧬 自我进化引擎 V4 初始化完成")
    
    def _init_db(self):
        """初始化数据库"""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS evolution_records (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    phase TEXT,
                    learning_type TEXT,
                    context_json TEXT,
                    observations_json TEXT,
                    actions_json TEXT,
                    outcomes_json TEXT,
                    rewards REAL,
                    performance_metrics_json TEXT,
                    improvements_json TEXT,
                    next_phase TEXT
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT,
                    triggers_json TEXT,
                    actions_json TEXT,
                    success_rate REAL,
                    avg_reward REAL,
                    usage_count INTEGER,
                    last_updated REAL
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    metric_name TEXT,
                    metric_value REAL,
                    baseline_value REAL,
                    improvement REAL
                )
            """)
    
    async def observe_environment(self, context: Dict[str, Any]) -> EvolutionRecord:
        """观察环境，收集数据"""
        record = EvolutionRecord(
            phase=EvolutionPhase.OBSERVATION,
            context=context,
            observations=self._collect_observations(context)
        )
        
        # 分析观察结果
        insights = await self._analyze_observations(record.observations)
        record.context["insights"] = insights
        
        # 确定下一步
        if self._should_proceed_to_analysis(record):
            record.next_phase = EvolutionPhase.ANALYSIS
        else:
            record.next_phase = EvolutionPhase.OBSERVATION
        
        # 保存记录
        self._save_evolution_record(record)
        self.evolution_history.append(record)
        
        logger.info(f"👁️ 环境观察完成，收集了 {len(record.observations)} 个观察点")
        return record
    
    def _collect_observations(self, context: Dict[str, Any]) -> List[str]:
        """收集观察数据"""
        observations = []
        
        # 观察任务执行情况
        if "task_results" in context:
            task_results = context["task_results"]
            observations.append(f"任务成功率: {self._calculate_success_rate(task_results):.2%}")
            observations.append(f"平均执行时间: {self._calculate_avg_duration(task_results):.2f}秒")
        
        # 观察资源使用情况
        if "resource_usage" in context:
            resource_usage = context["resource_usage"]
            observations.append(f"CPU使用率: {resource_usage.get('cpu', 0):.1%}")
            observations.append(f"内存使用率: {resource_usage.get('memory', 0):.1%}")
        
        # 观察用户反馈
        if "user_feedback" in context:
            feedback = context["user_feedback"]
            observations.append(f"用户满意度: {feedback.get('satisfaction', 0):.1f}")
            observations.append(f"反馈数量: {len(feedback.get('comments', []))}")
        
        # 观察系统错误
        if "error_logs" in context:
            error_logs = context["error_logs"]
            observations.append(f"错误数量: {len(error_logs)}")
            if error_logs:
                common_errors = self._analyze_common_errors(error_logs)
                observations.append(f"常见错误: {', '.join(common_errors[:3])}")
        
        return observations
    
    async def _analyze_observations(self, observations: List[str]) -> Dict[str, Any]:
        """分析观察数据"""
        insights = {}
        
        # 性能趋势分析
        performance_trends = self._analyze_performance_trends()
        insights["performance_trends"] = performance_trends
        
        # 异常检测
        anomalies = self._detect_anomalies(observations)
        insights["anomalies"] = anomalies
        
        # 改进机会识别
        improvement_opportunities = self._identify_improvement_opportunities(observations)
        insights["improvement_opportunities"] = improvement_opportunities
        
        return insights
    
    def _should_proceed_to_analysis(self, record: EvolutionRecord) -> bool:
        """判断是否应该进入分析阶段"""
        # 检查是否有足够的观察数据
        if len(record.observations) < 5:
            return False
        
        # 检查是否有异常情况
        if "anomalies" in record.context.get("insights", {}):
            if record.context["insights"]["anomalies"]:
                return True
        
        # 检查是否有改进机会
        if "improvement_opportunities" in record.context.get("insights", {}):
            if record.context["insights"]["improvement_opportunities"]:
                return True
        
        # 定期分析
        if len(self.evolution_history) % 10 == 0:
            return True
        
        return False
    
    def _calculate_success_rate(self, task_results: List[Dict]) -> float:
        """计算成功率"""
        if not task_results:
            return 0.0
        return sum(1 for r in task_results if r.get("success", False)) / len(task_results)
    
    def _calculate_avg_duration(self, task_results: List[Dict]) -> float:
        """计算平均执行时间"""
        if not task_results:
            return 0.0
        durations = [r.get("duration", 0) for r in task_results if "duration" in r]
        return sum(durations) / len(durations) if durations else 0.0
    
    def _analyze_common_errors(self, error_logs: List[Dict]) -> List[str]:
        """分析常见错误"""
        error_counts = {}
        for error in error_logs:
            error_type = error.get("error", "Unknown")
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # 返回最常见的3个错误
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        return [error for error, count in sorted_errors[:3]]
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """分析性能趋势"""
        if len(self.evolution_history) < 2:
            return {"trend": "insufficient_data"}
        
        recent_records = list(self.evolution_history)[-10:]
        rewards = [r.rewards for r in recent_records]
        
        # 计算趋势
        if len(rewards) >= 2:
            trend = "improving" if rewards[-1] > rewards[0] else "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "recent_average": sum(rewards) / len(rewards),
            "volatility": np.std(rewards) if len(rewards) > 1 else 0
        }
    
    def _detect_anomalies(self, observations: List[str]) -> List[str]:
        """检测异常"""
        anomalies = []
        
        # 简单的异常检测逻辑
        for obs in observations:
            if "错误" in obs and "数量" in obs:
                # 提取错误数量
                try:
                    count = int(obs.split("数量: ")[1])
                    if count > 5:  # 阈值
                        anomalies.append(f"错误数量过多: {count}")
                except:
                    pass
            
            if "使用率" in obs:
                try:
                    usage = float(obs.split(": ")[1].rstrip("%"))
                    if usage > 90:  # 阈值
                        anomalies.append(f"资源使用率过高: {usage}%")
                except:
                    pass
        
        return anomalies
    
    def _identify_improvement_opportunities(self, observations: List[str]) -> List[str]:
        """识别改进机会"""
        opportunities = []
        
        for obs in observations:
            if "成功率" in obs:
                try:
                    rate = float(obs.split(": ")[1].rstrip("%"))
                    if rate < 0.8:
                        opportunities.append("提升任务成功率")
                except:
                    pass
            
            if "平均执行时间" in obs:
                try:
                    duration = float(obs.split(": ")[1].rstrip("秒"))
                    if duration > 5.0:
                        opportunities.append("优化执行速度")
                except:
                    pass
        
        return opportunities
    
    def _identify_bottlenecks(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别瓶颈"""
        bottlenecks = []
        
        # 简化的瓶颈识别
        for metric, value in metrics.items():
            if isinstance(value, (int, float)) and value < 0.7:
                bottlenecks.append({
                    "component": metric,
                    "value": value,
                    "suggested_action": f"优化{metric}",
                    "priority": "high",
                    "expected_improvement": 0.2
                })
        
        return bottlenecks
    
    def _analyze_trends(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """分析趋势"""
        trends = {}
        # 简化实现，实际应该基于历史数据
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                trends[metric] = "stable"  # 简化
        return trends
    
    def _compare_with_baseline(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """与基线比较"""
        comparisons = {}
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                baseline = self.performance_baseline.get(metric, 0.5)
                comparisons[metric] = (value - baseline) / baseline if baseline > 0 else 0
        return comparisons
    
    def _extract_recent_patterns(self) -> List[Dict[str, Any]]:
        """提取最近的模式"""
        # 简化实现
        return list(self.learning_patterns.values())[-10:]
    
    def _identify_success_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别成功模式"""
        return [p for p in patterns if p.get("success_rate", 0) > 0.8]
    
    def _identify_failure_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别失败模式"""
        return [p for p in patterns if p.get("success_rate", 0) < 0.5]
    
    def _analyze_common_causes(self, errors: List[Dict]) -> List[str]:
        """分析常见原因"""
        # 简化实现
        return ["超时", "资源不足", "配置错误"]
    
    def _analyze_error_correlations(self, errors: List[Dict]) -> Dict[str, float]:
        """分析错误关联"""
        # 简化实现
        return {"timeout_memory": 0.6, "config_performance": 0.4}
    
    def _assess_error_impact(self, errors: List[Dict]) -> Dict[str, str]:
        """评估错误影响"""
        # 简化实现
        return {"TimeoutError": "high", "ValueError": "medium"}
    
    def _identify_quick_wins(self, record: EvolutionRecord) -> List[Dict[str, Any]]:
        """识别快速改进项"""
        return [
            {"area": "缓存", "action": "优化缓存策略", "improvement": 0.15},
            {"area": "日志", "action": "减少日志输出", "improvement": 0.1}
        ]
    
    def _identify_long_term_improvements(self, record: EvolutionRecord) -> List[Dict[str, Any]]:
        """识别长期改进项"""
        return [
            {"area": "架构", "action": "重构核心模块", "improvement": 0.3},
            {"area": "算法", "action": "优化核心算法", "improvement": 0.25}
        ]
    
    def _identify_resource_optimization(self, record: EvolutionRecord) -> List[Dict[str, Any]]:
        """识别资源优化项"""
        return [
            {"area": "内存", "action": "优化内存使用", "improvement": 0.2},
            {"area": "CPU", "action": "并行处理优化", "improvement": 0.15}
        ]
    
    def _collect_performance_metrics(self, record: EvolutionRecord) -> Dict[str, float]:
        """收集性能指标"""
        metrics = {}
        
        # 基于结果计算指标
        if record.outcomes:
            metrics["avg_outcome"] = sum(record.outcomes.values()) / len(record.outcomes)
            metrics["max_outcome"] = max(record.outcomes.values())
            metrics["min_outcome"] = min(record.outcomes.values())
        
        # 基于改进计算指标
        if record.improvements:
            metrics["improvement_count"] = len(record.improvements)
            metrics["improvement_rate"] = len([i for i in record.improvements if "显著" in i]) / len(record.improvements)
        
        return metrics
    
    def _update_performance_baseline(self, metrics: Dict[str, float]):
        """更新性能基线"""
        for metric, value in metrics.items():
            if metric not in self.performance_baseline:
                self.performance_baseline[metric] = value
            else:
                # 指数移动平均
                alpha = 0.1
                self.performance_baseline[metric] = alpha * value + (1 - alpha) * self.performance_baseline[metric]
    
    async def analyze_and_plan(self, record: EvolutionRecord) -> EvolutionRecord:
        """分析和规划阶段"""
        # 深度分析
        analysis_result = await self._deep_analysis(record)
        
        # 生成改进计划
        improvement_plan = await self._generate_improvement_plan(analysis_result)
        
        # 更新记录
        record.phase = EvolutionPhase.ANALYSIS
        record.context["analysis_result"] = analysis_result
        record.context["improvement_plan"] = improvement_plan
        
        # 选择学习类型
        record.learning_type = self._select_learning_type(analysis_result)
        
        # 确定下一步
        if improvement_plan:
            record.next_phase = EvolutionPhase.PLANNING
        else:
            record.next_phase = EvolutionPhase.OBSERVATION
        
        # 保存记录
        self._save_evolution_record(record)
        
        logger.info(f"🔍 分析完成，生成了 {len(improvement_plan)} 个改进项")
        return record
    
    async def _deep_analysis(self, record: EvolutionRecord) -> Dict[str, Any]:
        """深度分析"""
        analysis = {
            "performance_analysis": {},
            "pattern_analysis": {},
            "root_cause_analysis": {},
            "optimization_potential": {}
        }
        
        # 性能分析
        if "performance_metrics" in record.context:
            metrics = record.context["performance_metrics"]
            analysis["performance_analysis"] = {
                "bottlenecks": self._identify_bottlenecks(metrics),
                "trends": self._analyze_trends(metrics),
                "comparisons": self._compare_with_baseline(metrics)
            }
        
        # 模式分析
        recent_patterns = self._extract_recent_patterns()
        analysis["pattern_analysis"] = {
            "recurring_patterns": recent_patterns,
            "success_patterns": self._identify_success_patterns(recent_patterns),
            "failure_patterns": self._identify_failure_patterns(recent_patterns)
        }
        
        # 根因分析
        if "error_logs" in record.context:
            errors = record.context["error_logs"]
            analysis["root_cause_analysis"] = {
                "common_causes": self._analyze_common_causes(errors),
                "correlations": self._analyze_error_correlations(errors),
                "impact_assessment": self._assess_error_impact(errors)
            }
        
        # 优化潜力
        analysis["optimization_potential"] = {
            "quick_wins": self._identify_quick_wins(record),
            "long_term_improvements": self._identify_long_term_improvements(record),
            "resource_optimization": self._identify_resource_optimization(record)
        }
        
        return analysis
    
    async def _generate_improvement_plan(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成改进计划"""
        improvements = []
        
        # 基于性能分析的改进
        if "performance_analysis" in analysis:
            perf_analysis = analysis["performance_analysis"]
            for bottleneck in perf_analysis.get("bottlenecks", []):
                improvements.append({
                    "type": "performance",
                    "target": bottleneck["component"],
                    "action": bottleneck["suggested_action"],
                    "priority": bottleneck["priority"],
                    "expected_improvement": bottleneck.get("expected_improvement", 0.1)
                })
        
        # 基于模式分析的改进
        if "pattern_analysis" in analysis:
            pattern_analysis = analysis["pattern_analysis"]
            for pattern in pattern_analysis.get("failure_patterns", []):
                improvements.append({
                    "type": "pattern",
                    "target": pattern["pattern_id"],
                    "action": pattern["corrective_action"],
                    "priority": "medium",
                    "expected_improvement": 0.15
                })
        
        # 基于优化潜力的改进
        if "optimization_potential" in analysis:
            opt_potential = analysis["optimization_potential"]
            for quick_win in opt_potential.get("quick_wins", []):
                improvements.append({
                    "type": "optimization",
                    "target": quick_win["area"],
                    "action": quick_win["action"],
                    "priority": "high",
                    "expected_improvement": quick_win.get("improvement", 0.2)
                })
        
        # 按优先级排序
        improvements.sort(key=lambda x: (
            0 if x["priority"] == "high" else 
            1 if x["priority"] == "medium" else 2
        ))
        
        return improvements[:5]  # 最多5个改进项
    
    def _select_learning_type(self, analysis: Dict[str, Any]) -> LearningType:
        """选择学习类型"""
        # 根据分析结果选择最适合的学习类型
        if "pattern_analysis" in analysis:
            patterns = analysis["pattern_analysis"]
            if patterns.get("success_patterns") or patterns.get("failure_patterns"):
                return LearningType.SUPERVISED
        
        if "root_cause_analysis" in analysis:
            return LearningType.REINFORCEMENT
        
        return LearningType.META_LEARNING
    
    async def implement_improvements(self, record: EvolutionRecord) -> EvolutionRecord:
        """实施改进"""
        improvements = record.context.get("improvement_plan", [])
        implemented = []
        outcomes = {}
        
        for improvement in improvements:
            try:
                # 实施改进
                result = await self._implement_single_improvement(improvement)
                implemented.append(improvement)
                
                # 记录结果
                outcomes[improvement["target"]] = result["success_rate"]
                
                # 更新学习模式
                self._update_learning_patterns(improvement, result)
                
            except Exception as e:
                logger.error(f"实施改进失败: {e}")
                outcomes[improvement["target"]] = 0.0
        
        # 更新记录
        record.phase = EvolutionPhase.IMPLEMENTATION
        record.actions = [imp["action"] for imp in implemented]
        record.outcomes = outcomes
        record.next_phase = EvolutionPhase.EVALUATION
        
        # 保存记录
        self._save_evolution_record(record)
        
        logger.info(f"⚙️ 实施了 {len(implemented)} 个改进")
        return record
    
    async def _implement_single_improvement(self, improvement: Dict[str, Any]) -> Dict[str, Any]:
        """实施单个改进"""
        # 这里是简化的实现，实际应该根据改进类型执行具体操作
        
        # 模拟改进实施
        await asyncio.sleep(0.1)
        
        # 基于改进类型返回不同的结果
        if improvement["type"] == "performance":
            return {
                "success_rate": 0.8 + np.random.normal(0, 0.1),
                "performance_gain": improvement.get("expected_improvement", 0.1) * np.random.uniform(0.5, 1.5)
            }
        elif improvement["type"] == "pattern":
            return {
                "success_rate": 0.7 + np.random.normal(0, 0.1),
                "pattern_improvement": improvement.get("expected_improvement", 0.15) * np.random.uniform(0.5, 1.5)
            }
        else:
            return {
                "success_rate": 0.75 + np.random.normal(0, 0.1),
                "optimization_gain": improvement.get("expected_improvement", 0.2) * np.random.uniform(0.5, 1.5)
            }
    
    def _update_learning_patterns(self, improvement: Dict[str, Any], result: Dict[str, Any]):
        """更新学习模式"""
        pattern_id = f"{improvement['type']}_{improvement['target']}"
        
        if pattern_id not in self.learning_patterns:
            self.learning_patterns[pattern_id] = LearningPattern(
                pattern_id=pattern_id,
                pattern_type=improvement["type"],
                triggers=[improvement["target"]],
                actions=[improvement["action"]],
                success_rate=result["success_rate"],
                avg_reward=result.get("performance_gain", 0.1),
                usage_count=1
            )
        else:
            pattern = self.learning_patterns[pattern_id]
            # 更新统计
            pattern.usage_count += 1
            pattern.success_rate = (pattern.success_rate * (pattern.usage_count - 1) + result["success_rate"]) / pattern.usage_count
            pattern.avg_reward = (pattern.avg_reward * (pattern.usage_count - 1) + result.get("performance_gain", 0.1)) / pattern.usage_count
            pattern.last_updated = datetime.now()
        
        # 保存到数据库
        self._save_learning_pattern(self.learning_patterns[pattern_id])
    
    async def evaluate_improvements(self, record: EvolutionRecord) -> EvolutionRecord:
        """评估改进效果"""
        # 计算总体奖励
        total_reward = 0.0
        for outcome in record.outcomes.values():
            total_reward += outcome
        
        record.rewards = total_reward / len(record.outcomes) if record.outcomes else 0.0
        
        # 收集性能指标
        performance_metrics = self._collect_performance_metrics(record)
        record.performance_metrics = performance_metrics
        
        # 评估改进效果
        improvements = []
        for target, outcome in record.outcomes.items():
            if outcome > 0.7:
                improvements.append(f"{target}: 显著改进")
            elif outcome > 0.5:
                improvements.append(f"{target}: 中等改进")
            else:
                improvements.append(f"{target}: 改进有限")
        
        record.improvements = improvements
        
        # 更新记录
        record.phase = EvolutionPhase.EVALUATION
        
        # 决定下一步
        if record.rewards > 0.6:
            record.next_phase = EvolutionPhase.CONSOLIDATION
        else:
            record.next_phase = EvolutionPhase.OBSERVATION
        
        # 保存记录
        self._save_evolution_record(record)
        
        logger.info(f"📊 评估完成，总体奖励: {record.rewards:.3f}")
        return record
    
    async def consolidate_learning(self, record: EvolutionRecord) -> EvolutionRecord:
        """巩固学习成果"""
        # 更新性能基线
        self._update_performance_baseline(record.performance_metrics)
        
        # 强化成功模式
        successful_patterns = [
            p for p in self.learning_patterns.values() 
            if p.success_rate > 0.8
        ]
        
        # 弱化失败模式
        failed_patterns = [
            p for p in self.learning_patterns.values() 
            if p.success_rate < 0.5
        ]
        
        # 生成学习总结
        learning_summary = {
            "successful_patterns": len(successful_patterns),
            "failed_patterns": len(failed_patterns),
            "total_improvements": len(record.improvements),
            "overall_performance": record.rewards
        }
        
        # 更新记录
        record.phase = EvolutionPhase.CONSOLIDATION
        record.context["learning_summary"] = learning_summary
        record.next_phase = EvolutionPhase.OBSERVATION
        
        # 保存记录
        self._save_evolution_record(record)
        
        logger.info(f"🎯 巩固完成，成功模式: {len(successful_patterns)}, 失败模式: {len(failed_patterns)}")
        return record
    
    def _save_evolution_record(self, record: EvolutionRecord):
        """保存进化记录"""
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO evolution_records 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.id,
                    record.timestamp.timestamp(),
                    record.phase.value,
                    record.learning_type.value,
                    json.dumps(record.context),
                    json.dumps(record.observations),
                    json.dumps(record.actions),
                    json.dumps(record.outcomes),
                    record.rewards,
                    json.dumps(record.performance_metrics),
                    json.dumps(record.improvements),
                    record.next_phase.value if record.next_phase else None
                )
            )
    
    def _save_learning_pattern(self, pattern: LearningPattern):
        """保存学习模式"""
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO learning_patterns 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pattern.pattern_id,
                    pattern.pattern_type,
                    json.dumps(pattern.triggers),
                    json.dumps(pattern.actions),
                    pattern.success_rate,
                    pattern.avg_reward,
                    pattern.usage_count,
                    pattern.last_updated.timestamp()
                )
            )
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """获取进化统计信息"""
        stats = {
            "total_records": len(self.evolution_history),
            "current_phase": self.current_phase.value,
            "learning_patterns": len(self.learning_patterns),
            "phase_distribution": {},
            "learning_type_distribution": {},
            "average_rewards": 0.0,
            "improvement_rate": 0.0
        }
        
        if self.evolution_history:
            # 阶段分布
            for record in self.evolution_history:
                phase = record.phase.value
                stats["phase_distribution"][phase] = stats["phase_distribution"].get(phase, 0) + 1
            
            # 学习类型分布
            for record in self.evolution_history:
                ltype = record.learning_type.value
                stats["learning_type_distribution"][ltype] = stats["learning_type_distribution"].get(ltype, 0) + 1
            
            # 平均奖励
            total_rewards = sum(r.rewards for r in self.evolution_history)
            stats["average_rewards"] = total_rewards / len(self.evolution_history)
            
            # 改进率
            improved_records = sum(1 for r in self.evolution_history if r.rewards > 0.6)
            stats["improvement_rate"] = improved_records / len(self.evolution_history)
        
        return stats
    
    async def run_evolution_cycle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """运行完整的进化周期"""
        logger.info("🚀 开始进化周期...")
        
        # 1. 观察阶段
        observation_record = await self.observe_environment(context)
        
        # 2. 分析和规划阶段
        if observation_record.next_phase == EvolutionPhase.ANALYSIS:
            analysis_record = await self.analyze_and_plan(observation_record)
        else:
            analysis_record = observation_record
        
        # 3. 实施阶段
        if analysis_record.next_phase == EvolutionPhase.PLANNING:
            implementation_record = await self.implement_improvements(analysis_record)
        else:
            implementation_record = analysis_record
        
        # 4. 评估阶段
        if implementation_record.next_phase == EvolutionPhase.EVALUATION:
            evaluation_record = await self.evaluate_improvements(implementation_record)
        else:
            evaluation_record = implementation_record
        
        # 5. 巩固阶段
        if evaluation_record.next_phase == EvolutionPhase.CONSOLIDATION:
            final_record = await self.consolidate_learning(evaluation_record)
        else:
            final_record = evaluation_record
        
        # 返回周期结果
        return {
            "cycle_completed": True,
            "final_phase": final_record.phase.value,
            "total_reward": final_record.rewards,
            "improvements": final_record.improvements,
            "learning_summary": final_record.context.get("learning_summary", {}),
            "statistics": self.get_evolution_statistics()
        }

# --- 示例使用 ---
async def main():
    """测试自我进化引擎"""
    engine = SelfEvolutionEngineV4()
    
    # 模拟环境数据
    context = {
        "task_results": [
            {"success": True, "duration": 5.2},
            {"success": True, "duration": 3.8},
            {"success": False, "duration": 8.1},
            {"success": True, "duration": 4.5}
        ],
        "resource_usage": {
            "cpu": 0.65,
            "memory": 0.78
        },
        "user_feedback": {
            "satisfaction": 0.82,
            "comments": ["响应快", "结果准确", "界面友好"]
        },
        "error_logs": [
            {"error": "TimeoutError", "count": 2},
            {"error": "ValueError", "count": 1}
        ]
    }
    
    # 运行进化周期
    result = await engine.run_evolution_cycle(context)
    
    print("\n📊 进化周期结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 获取统计信息
    stats = engine.get_evolution_statistics()
    print("\n📈 进化统计:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
