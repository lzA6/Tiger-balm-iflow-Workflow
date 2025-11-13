#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 优化的智能体融合缓存系统 V2
高效缓存和预计算智能体融合结果，大幅提升工作流执行效率。
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import asyncio
import logging
import time
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import numpy as np
from functools import lru_cache

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

@dataclass
class FusionCacheEntry:
    """融合缓存条目"""
    task_hash: str
    task_description: str
    selected_experts: List[str]
    fusion_mode: str
    result: Any
    quality_score: float
    execution_time: float
    timestamp: float
    hit_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    context_similarity: float = 0.0  # 上下文相似度

@dataclass
class PrecomputedPattern:
    """预计算模式"""
    pattern_hash: str
    task_keywords: Set[str]
    common_experts: List[str]
    optimal_fusion_mode: str
    success_rate: float
    avg_quality_score: float
    last_updated: float

class OptimizedFusionCache:
    """
    优化的智能体融合缓存系统
    """
    
    def __init__(self, cache_size: int = 1000, ttl_hours: int = 24):
        self.cache_size = cache_size
        self.ttl_hours = ttl_hours
        
        # 主缓存字典
        self.cache: Dict[str, FusionCacheEntry] = {}
        
        # 预计算模式库
        self.patterns: Dict[str, PrecomputedPattern] = {}
        
        # 访问频率统计
        self.access_frequency: Dict[str, int] = defaultdict(int)
        
        # LRU队列
        self.lru_queue: deque = deque()
        
        # 缓存统计
        self.stats = {
            "hits": 0,
            "misses": 0,
            "precomputed_hits": 0,
            "total_requests": 0,
            "avg_response_time": 0.0,
            "cache_efficiency": 0.0
        }
        
        # 锁机制
        self._lock = threading.RLock()
        
        # 加载持久化缓存
        self._load_persisted_cache()
        
        logger.info("优化的智能体融合缓存系统初始化完成")
    
    def _generate_task_hash(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """生成任务哈希"""
        context_str = json.dumps(context or {}, sort_keys=True)
        combined = f"{task}:{context_str}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def _calculate_similarity(self, task1: str, task2: str) -> float:
        """计算任务相似度"""
        # 简化的相似度计算
        words1 = set(task1.lower().split())
        words2 = set(task2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def get_cached_result(self, task: str, context: Optional[Dict[str, Any]] = None) -> Optional[FusionCacheEntry]:
        """获取缓存结果"""
        with self._lock:
            task_hash = self._generate_task_hash(task, context)
            self.stats["total_requests"] += 1
            
            if task_hash in self.cache:
                entry = self.cache[task_hash]
                
                # 检查TTL
                if time.time() - entry.timestamp > self.ttl_hours * 3600:
                    self._remove_entry(task_hash)
                    self.stats["misses"] += 1
                    return None
                
                # 更新访问统计
                entry.hit_count += 1
                entry.last_accessed = time.time()
                self._update_lru(task_hash)
                self.stats["hits"] += 1
                
                logger.info(f"缓存命中: {task[:50]}...")
                return entry
            
            self.stats["misses"] += 1
            return None
    
    def _update_lru(self, task_hash: str):
        """更新LRU队列"""
        if task_hash in self.lru_queue:
            self.lru_queue.remove(task_hash)
        self.lru_queue.append(task_hash)
    
    def _remove_entry(self, task_hash: str):
        """移除缓存条目"""
        if task_hash in self.cache:
            del self.cache[task_hash]
        if task_hash in self.access_frequency:
            del self.access_frequency[task_hash]
    
    def _evict_lru_entries(self):
        """LRU淘汰机制"""
        while len(self.cache) >= self.cache_size and self.lru_queue:
            lru_hash = self.lru_queue.popleft()
            if lru_hash in self.cache:
                del self.cache[lru_hash]
    
    def put_cache_result(self, task: str, context: Optional[Dict[str, Any]], 
                        selected_experts: List[str], fusion_mode: str,
                        result: Any, quality_score: float, execution_time: float):
        """存储缓存结果"""
        with self._lock:
            task_hash = self._generate_task_hash(task, context)
            
            # 创建缓存条目
            entry = FusionCacheEntry(
                task_hash=task_hash,
                task_description=task,
                selected_experts=selected_experts,
                fusion_mode=fusion_mode,
                result=result,
                quality_score=quality_score,
                execution_time=execution_time,
                timestamp=time.time()
            )
            
            # 淘汰旧条目
            if len(self.cache) >= self.cache_size:
                self._evict_lru_entries()
            
            # 存储新条目
            self.cache[task_hash] = entry
            self._update_lru(task_hash)
            
            # 更新访问频率
            self.access_frequency[task_hash] += 1
            
            logger.info(f"缓存存储: {task[:50]}... (质量: {quality_score:.2f})")
    
    def find_similar_tasks(self, task: str, threshold: float = 0.7) -> List[FusionCacheEntry]:
        """查找相似任务"""
        with self._lock:
            similar_tasks = []
            
            for entry in self.cache.values():
                similarity = self._calculate_similarity(task, entry.task_description)
                if similarity >= threshold:
                    entry.context_similarity = similarity
                    similar_tasks.append(entry)
            
            # 按相似度排序
            similar_tasks.sort(key=lambda x: x.context_similarity, reverse=True)
            return similar_tasks[:10]  # 返回前10个最相似的
    
    def get_precomputed_pattern(self, task_keywords: Set[str]) -> Optional[PrecomputedPattern]:
        """获取预计算模式"""
        with self._lock:
            pattern_hash = hashlib.md5(str(sorted(task_keywords)).encode()).hexdigest()
            
            if pattern_hash in self.patterns:
                pattern = self.patterns[pattern_hash]
                
                # 检查是否需要更新
                if time.time() - pattern.last_updated > 3600:  # 1小时更新一次
                    return None
                
                self.stats["precomputed_hits"] += 1
                logger.info(f"预计算模式命中: {task_keywords}")
                return pattern
            
            return None
    
    def update_precomputed_pattern(self, task_keywords: Set[str], 
                                 common_experts: List[str], 
                                 fusion_mode: str,
                                 success_rate: float,
                                 quality_score: float):
        """更新预计算模式"""
        with self._lock:
            pattern_hash = hashlib.md5(str(sorted(task_keywords)).encode()).hexdigest()
            
            if pattern_hash in self.patterns:
                pattern = self.patterns[pattern_hash]
                # 指数移动平均更新
                alpha = 0.1
                pattern.success_rate = alpha * success_rate + (1 - alpha) * pattern.success_rate
                pattern.avg_quality_score = alpha * quality_score + (1 - alpha) * pattern.avg_quality_score
                pattern.last_updated = time.time()
            else:
                pattern = PrecomputedPattern(
                    pattern_hash=pattern_hash,
                    task_keywords=task_keywords,
                    common_experts=common_experts,
                    optimal_fusion_mode=fusion_mode,
                    success_rate=success_rate,
                    avg_quality_score=quality_score,
                    last_updated=time.time()
                )
                self.patterns[pattern_hash] = pattern
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total_requests = self.stats["total_requests"]
            if total_requests == 0:
                cache_hit_rate = 0.0
            else:
                cache_hit_rate = self.stats["hits"] / total_requests
            
            # 计算平均响应时间
            avg_response_time = self.stats["avg_response_time"]
            
            # 缓存效率
            cache_efficiency = (self.stats["hits"] + self.stats["precomputed_hits"]) / max(total_requests, 1)
            
            return {
                "cache_hit_rate": cache_hit_rate,
                "precomputed_hit_rate": self.stats["precomputed_hits"] / max(total_requests, 1),
                "total_cache_entries": len(self.cache),
                "total_patterns": len(self.patterns),
                "total_requests": total_requests,
                "avg_response_time": avg_response_time,
                "cache_efficiency": cache_efficiency,
                "memory_usage_mb": self._estimate_memory_usage()
            }
    
    def _estimate_memory_usage(self) -> float:
        """估算内存使用量"""
        try:
            # 简单估算
            cache_size = len(pickle.dumps(self.cache, protocol=pickle.HIGHEST_PROTOCOL))
            pattern_size = len(pickle.dumps(self.patterns, protocol=pickle.HIGHEST_PROTOCOL))
            total_bytes = cache_size + pattern_size
            return total_bytes / (1024 * 1024)  # 转换为MB
        except:
            return 0.0
    
    def cleanup_expired_entries(self):
        """清理过期条目"""
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for task_hash, entry in self.cache.items():
                if current_time - entry.timestamp > self.ttl_hours * 3600:
                    expired_keys.append(task_hash)
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                logger.info(f"清理了 {len(expired_keys)} 个过期缓存条目")
    
    def _load_persisted_cache(self):
        """加载持久化缓存"""
        cache_file = PROJECT_ROOT / ".iflow" / "cache" / "fusion_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cache = data.get('cache', {})
                    self.patterns = data.get('patterns', {})
                    self.access_frequency = defaultdict(int, data.get('access_frequency', {}))
                logger.info("持久化缓存加载成功")
            except Exception as e:
                logger.error(f"加载持久化缓存失败: {e}")
    
    def persist_cache(self):
        """持久化缓存"""
        cache_dir = PROJECT_ROOT / ".iflow" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = cache_dir / "fusion_cache.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'cache': self.cache,
                    'patterns': self.patterns,
                    'access_frequency': dict(self.access_frequency)
                }, f)
            logger.info("缓存持久化成功")
        except Exception as e:
            logger.error(f"持久化缓存失败: {e}")
    
    def clear_cache(self):
        """清空缓存"""
        with self._lock:
            self.cache.clear()
            self.patterns.clear()
            self.access_frequency.clear()
            self.lru_queue.clear()
            logger.info("缓存已清空")
    
    async def background_maintenance(self):
        """后台维护任务"""
        while True:
            try:
                # 清理过期条目
                self.cleanup_expired_entries()
                
                # 持久化缓存
                self.persist_cache()
                
                # 更新统计
                stats = self.get_cache_statistics()
                logger.debug(f"缓存统计: {stats}")
                
                # 每小时执行一次
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"后台维护任务出错: {e}")
                await asyncio.sleep(600)  # 出错后等待10分钟

class IntelligentFusionOptimizer:
    """
    智能融合优化器
    """
    
    def __init__(self, cache: OptimizedFusionCache):
        self.cache = cache
        self.prediction_model = {}
        self.optimization_history = deque(maxlen=1000)
    
    def predict_optimal_experts(self, task: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """预测最优专家组合"""
        # 从相似任务中学习
        similar_tasks = self.cache.find_similar_tasks(task, threshold=0.6)
        
        if not similar_tasks:
            return []
        
        # 统计专家使用频率
        expert_frequency = defaultdict(int)
        for task_entry in similar_tasks:
            for expert in task_entry.selected_experts:
                expert_frequency[expert] += 1
        
        # 按频率排序
        sorted_experts = sorted(expert_frequency.items(), key=lambda x: x[1], reverse=True)
        return [expert for expert, freq in sorted_experts[:5]]
    
    def predict_optimal_fusion_mode(self, task_complexity: str, expert_count: int) -> str:
        """预测最优融合模式"""
        # 基于历史数据的简单预测
        mode_scores = {
            "sequential": 0.8,
            "parallel": 0.9,
            "collaborative": 0.85,
            "hierarchical": 0.95,
            "adaptive": 1.0
        }
        
        # 根据复杂度和专家数量调整
        if task_complexity in ["simple", "moderate"]:
            mode_scores["sequential"] += 0.1
        elif task_complexity in ["complex", "expert"]:
            mode_scores["hierarchical"] += 0.1
            mode_scores["adaptive"] += 0.1
        
        if expert_count <= 2:
            mode_scores["collaborative"] += 0.1
        elif expert_count > 5:
            mode_scores["parallel"] += 0.1
        
        # 返回得分最高的模式
        return max(mode_scores.items(), key=lambda x: x[1])[0]
    
    def optimize_fusion_parameters(self, task: str, base_experts: List[str], 
                                 base_mode: str) -> Dict[str, Any]:
        """优化融合参数"""
        # 预测优化的专家组合
        predicted_experts = self.predict_optimal_experts(task)
        
        # 如果预测的专家组合更好，使用预测结果
        if len(predicted_experts) > len(base_experts) * 0.5:
            optimized_experts = list(set(base_experts + predicted_experts))
        else:
            optimized_experts = base_experts
        
        # 预测优化的融合模式
        task_complexity = self._infer_complexity(task)
        optimized_mode = self.predict_optimal_fusion_mode(task_complexity, len(optimized_experts))
        
        return {
            "optimized_experts": optimized_experts,
            "optimized_mode": optimized_mode,
            "confidence": 0.8 if predicted_experts else 0.6,
            "optimization_reason": "基于历史相似任务的优化建议" if predicted_experts else "使用默认优化策略"
        }
    
    def _infer_complexity(self, task: str) -> str:
        """推断任务复杂度"""
        task_lower = task.lower()
        
        if any(keyword in task_lower for keyword in ["简单", "基础", "快速"]):
            return "simple"
        elif any(keyword in task_lower for keyword in ["分析", "设计", "实现"]):
            return "moderate"
        elif any(keyword in task_lower for keyword in ["架构", "系统", "集成"]):
            return "complex"
        elif any(keyword in task_lower for keyword in ["高级", "深度", "专家"]):
            return "expert"
        else:
            return "moderate"

# --- 使用示例 ---
async def main():
    """示例使用"""
    # 创建缓存系统
    cache = OptimizedFusionCache(cache_size=500, ttl_hours=12)
    
    # 创建优化器
    optimizer = IntelligentFusionOptimizer(cache)
    
    # 模拟缓存使用
    task = "设计一个高性能的电商系统架构"
    result = cache.get_cached_result(task)
    
    if not result:
        # 模拟计算结果
        cache.put_cache_result(
            task=task,
            context={"domain": "电商", "scale": "大型"},
            selected_experts=["架构师", "性能专家", "安全专家"],
            fusion_mode="hierarchical",
            result="架构设计方案...",
            quality_score=0.95,
            execution_time=2.5
        )
    
    # 获取统计信息
    stats = cache.get_cache_statistics()
    print(f"缓存统计: {json.dumps(stats, indent=2)}")
    
    # 预测优化
    optimization = optimizer.optimize_fusion_parameters(
        task, ["架构师"], "sequential"
    )
    print(f"优化建议: {optimization}")

if __name__ == "__main__":
    asyncio.run(main())