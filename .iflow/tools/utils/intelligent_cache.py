#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 智能缓存系统 vΩ - Intelligent Cache System
Intelligent Cache System vΩ - 基于机器学习的智能缓存系统

实现预测性缓存、分层缓存、智能失效等高级缓存功能，
大幅提升模型适配器的响应速度和性能。
"""

import asyncio
import json
import time
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict, defaultdict
import numpy as np
from abc import ABC, abstractmethod

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """缓存层级枚举"""
    L1_HOT = "l1_hot"      # 热点数据
    L2_WARM = "l2_warm"    # 常用数据
    L3_COLD = "l3_cold"    # 冷数据

class CacheStrategy(Enum):
    """缓存策略枚举"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    PREDICTIVE = "predictive"

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[float] = None
    size: int = 0
    level: CacheLevel = CacheLevel.L3_COLD
    prediction_score: float = 0.0
    dependencies: Set[str] = field(default_factory=set)

@dataclass
class CacheStats:
    """缓存统计"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    hit_rate: float = 0.0
    avg_access_time: float = 0.0
    memory_usage: int = 0

class CacheEvictionPolicy(ABC):
    """缓存淘汰策略抽象基类"""
    
    @abstractmethod
    async def select_victim(self, cache: 'IntelligentCache') -> Optional[str]:
        """选择要淘汰的缓存项"""
        pass

class LRUEvictionPolicy(CacheEvictionPolicy):
    """LRU淘汰策略"""
    
    async def select_victim(self, cache: 'IntelligentCache') -> Optional[str]:
        """选择最近最少使用的项"""
        oldest_access = None
        victim_key = None
        
        for key, entry in cache.cache_data.items():
            if oldest_access is None or entry.last_accessed < oldest_access:
                oldest_access = entry.last_accessed
                victim_key = key
        
        return victim_key

class PredictiveEvictionPolicy(CacheEvictionPolicy):
    """预测性淘汰策略"""
    
    async def select_victim(self, cache: 'IntelligentCache') -> Optional[str]:
        """基于预测选择淘汰项"""
        min_prediction = float('inf')
        victim_key = None
        
        for key, entry in cache.cache_data.items():
            # 综合考虑预测分数和访问时间
            time_factor = (datetime.now() - entry.last_accessed).total_seconds() / 3600  # 小时
            combined_score = entry.prediction_score - time_factor * 0.1
            
            if combined_score < min_prediction:
                min_prediction = combined_score
                victim_key = key
        
        return victim_key

class UsagePatternAnalyzer:
    """使用模式分析器"""
    
    def __init__(self):
        self.access_patterns = defaultdict(list)
        self.pattern_weights = defaultdict(float)
        self.last_analysis = datetime.now()
    
    def record_access(self, key: str, context: Dict[str, Any] = None):
        """记录访问模式"""
        timestamp = datetime.now()
        self.access_patterns[key].append({
            "timestamp": timestamp,
            "context": context or {}
        })
        
        # 保持最近100次访问记录
        if len(self.access_patterns[key]) > 100:
            self.access_patterns[key] = self.access_patterns[key][-100:]
    
    def predict_next_access(self, key: str) -> float:
        """预测下次访问概率"""
        if key not in self.access_patterns or len(self.access_patterns[key]) < 2:
            return 0.1  # 默认低概率
        
        pattern = self.access_patterns[key]
        
        # 计算访问频率
        recent_accesses = [p for p in pattern if 
                          (datetime.now() - p["timestamp"]).total_seconds() < 3600]
        
        if not recent_accesses:
            return 0.05
        
        # 基于最近访问频率预测
        frequency = len(recent_accesses) / 3600  # 每秒访问次数
        
        # 计算访问间隔规律性
        if len(recent_accesses) > 1:
            intervals = []
            for i in range(1, len(recent_accesses)):
                interval = (recent_accesses[i]["timestamp"] - recent_accesses[i-1]["timestamp"]).total_seconds()
                intervals.append(interval)
            
            # 间隔越规律，预测分数越高
            if intervals:
                interval_std = np.std(intervals)
                interval_mean = np.mean(intervals)
                regularity = max(0, 1 - interval_std / interval_mean) if interval_mean > 0 else 0
            else:
                regularity = 0
        else:
            regularity = 0
        
        # 综合预测分数
        prediction_score = min(1.0, frequency * 100 + regularity * 0.5)
        
        return prediction_score
    
    def analyze_patterns(self):
        """分析访问模式"""
        for key in self.access_patterns:
            prediction_score = self.predict_next_access(key)
            self.pattern_weights[key] = prediction_score
        
        self.last_analysis = datetime.now()

class IntelligentCache:
    """智能缓存系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_data: Dict[str, CacheEntry] = {}
        self.level_limits = {
            CacheLevel.L1_HOT: config.get("tiered_cache", {}).get("l1_size", 100),
            CacheLevel.L2_WARM: config.get("tiered_cache", {}).get("l2_size", 500),
            CacheLevel.L3_COLD: config.get("tiered_cache", {}).get("l3_size", 1000)
        }
        self.total_limit = sum(self.level_limits.values())
        self.current_size = 0
        self.stats = CacheStats()
        self.pattern_analyzer = UsagePatternAnalyzer()
        self.eviction_policy = self._create_eviction_policy()
        self.cleanup_task = None
        
    def _create_eviction_policy(self) -> CacheEvictionPolicy:
        """创建淘汰策略"""
        strategy = self.config.get("strategy", "predictive")
        
        if strategy == CacheStrategy.PREDICTIVE.value:
            return PredictiveEvictionPolicy()
        elif strategy == CacheStrategy.LRU.value:
            return LRUEvictionPolicy()
        else:
            return LRUEvictionPolicy()
    
    async def initialize(self):
        """初始化缓存系统"""
        # 启动后台清理任务
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Intelligent cache initialized")
    
    async def get(self, key: str, context: Dict[str, Any] = None) -> Optional[Any]:
        """获取缓存值"""
        start_time = time.time()
        
        # 记录访问模式
        self.pattern_analyzer.record_access(key, context)
        
        if key not in self.cache_data:
            self.stats.misses += 1
            self.stats.total_requests += 1
            self._update_hit_rate()
            return None
        
        entry = self.cache_data[key]
        
        # 检查TTL
        if entry.ttl and (datetime.now() - entry.created_at).total_seconds() > entry.ttl:
            await self._remove_entry(key)
            self.stats.misses += 1
            self.stats.total_requests += 1
            self._update_hit_rate()
            return None
        
        # 更新访问信息
        entry.last_accessed = datetime.now()
        entry.access_count += 1
        
        # 更新缓存层级
        await self._promote_entry(key)
        
        # 更新统计
        self.stats.hits += 1
        self.stats.total_requests += 1
        self._update_hit_rate()
        
        # 更新平均访问时间
        access_time = time.time() - start_time
        self.stats.avg_access_time = (
            (self.stats.avg_access_time * (self.stats.total_requests - 1) + access_time) /
            self.stats.total_requests
        )
        
        logger.debug(f"Cache hit for key: {key}")
        return entry.value
    
    async def put(self, key: str, value: Any, ttl: Optional[float] = None, 
                  context: Dict[str, Any] = None) -> bool:
        """存储缓存值"""
        # 计算值的大小（简化计算）
        value_size = len(str(value).encode('utf-8'))
        
        # 检查是否需要淘汰
        await self._ensure_capacity(value_size)
        
        # 创建缓存条目
        now = datetime.now()
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=now,
            last_accessed=now,
            ttl=ttl,
            size=value_size,
            prediction_score=self.pattern_analyzer.predict_next_access(key)
        )
        
        # 如果键已存在，更新大小
        if key in self.cache_data:
            old_entry = self.cache_data[key]
            self.current_size -= old_entry.size
        
        # 存储条目
        self.cache_data[key] = entry
        self.current_size += value_size
        
        # 设置初始层级
        await self._promote_entry(key)
        
        # 记录访问模式
        self.pattern_analyzer.record_access(key, context)
        
        logger.debug(f"Cache stored for key: {key}")
        return True
    
    async def remove(self, key: str) -> bool:
        """删除缓存项"""
        if key not in self.cache_data:
            return False
        
        await self._remove_entry(key)
        logger.debug(f"Cache removed for key: {key}")
        return True
    
    async def _remove_entry(self, key: str):
        """删除缓存条目"""
        if key in self.cache_data:
            entry = self.cache_data[key]
            del self.cache_data[key]
            self.current_size -= entry.size
    
    async def _promote_entry(self, key: str):
        """提升缓存条目层级"""
        if key not in self.cache_data:
            return
        
        entry = self.cache_data[key]
        
        # 基于访问频率和预测分数决定层级
        access_frequency = entry.access_count / max(1, (datetime.now() - entry.created_at).total_seconds() / 3600)
        
        if access_frequency > 10 or entry.prediction_score > 0.8:
            new_level = CacheLevel.L1_HOT
        elif access_frequency > 1 or entry.prediction_score > 0.5:
            new_level = CacheLevel.L2_WARM
        else:
            new_level = CacheLevel.L3_COLD
        
        # 如果层级提升，检查容量限制
        if new_level.value < entry.level.value:
            await self._ensure_level_capacity(new_level)
        
        entry.level = new_level
    
    async def _ensure_capacity(self, new_entry_size: int):
        """确保有足够容量"""
        while self.current_size + new_entry_size > self.total_limit:
            victim_key = await self.eviction_policy.select_victim(self)
            if victim_key:
                await self._remove_entry(victim_key)
                self.stats.evictions += 1
            else:
                break
    
    async def _ensure_level_capacity(self, level: CacheLevel):
        """确保层级容量限制"""
        level_count = sum(1 for entry in self.cache_data.values() if entry.level == level)
        level_limit = self.level_limits[level]
        
        while level_count >= level_limit:
            # 找到该层级中预测分数最低的项
            candidates = [(key, entry) for key, entry in self.cache_data.items() 
                         if entry.level == level]
            
            if not candidates:
                break
            
            # 按预测分数排序
            candidates.sort(key=lambda x: x[1].prediction_score)
            
            # 降级或淘汰最低分数的项
            victim_key, victim_entry = candidates[0]
            
            if level == CacheLevel.L1_HOT:
                victim_entry.level = CacheLevel.L2_WARM
            elif level == CacheLevel.L2_WARM:
                victim_entry.level = CacheLevel.L3_COLD
            else:
                await self._remove_entry(victim_key)
                self.stats.evictions += 1
                break
            
            level_count -= 1
    
    async def _cleanup_loop(self):
        """后台清理循环"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟清理一次
                await self._cleanup_expired()
                await self._analyze_patterns()
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _cleanup_expired(self):
        """清理过期条目"""
        now = datetime.now()
        expired_keys = []
        
        for key, entry in self.cache_data.items():
            if entry.ttl and (now - entry.created_at).total_seconds() > entry.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            await self._remove_entry(key)
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
    
    async def _analyze_patterns(self):
        """分析访问模式"""
        self.pattern_analyzer.analyze_patterns()
        
        # 更新所有条目的预测分数
        for key, entry in self.cache_data.items():
            entry.prediction_score = self.pattern_analyzer.predict_next_access(key)
    
    def _update_hit_rate(self):
        """更新命中率"""
        if self.stats.total_requests > 0:
            self.stats.hit_rate = self.stats.hits / self.stats.total_requests
    
    async def prefetch(self, keys: List[str], fetch_func: callable):
        """预取缓存"""
        for key in keys:
            if key not in self.cache_data:
                try:
                    # 异步获取数据
                    value = await fetch_func(key)
                    if value is not None:
                        await self.put(key, value)
                        logger.debug(f"Prefetched key: {key}")
                except Exception as e:
                    logger.warning(f"Prefetch failed for key {key}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        self.stats.memory_usage = self.current_size
        
        # 按层级统计
        level_stats = defaultdict(int)
        for entry in self.cache_data.values():
            level_stats[entry.level.value] += 1
        
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "evictions": self.stats.evictions,
            "total_requests": self.stats.total_requests,
            "hit_rate": self.stats.hit_rate,
            "avg_access_time": self.stats.avg_access_time,
            "memory_usage": self.stats.memory_usage,
            "total_entries": len(self.cache_data),
            "level_distribution": dict(level_stats)
        }
    
    async def clear(self):
        """清空缓存"""
        self.cache_data.clear()
        self.current_size = 0
        self.stats = CacheStats()
        logger.info("Cache cleared")
    
    async def destroy(self):
        """销毁缓存系统"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        await self.clear()
        logger.info("Intelligent cache destroyed")

class PredictiveCacheManager:
    """预测性缓存管理器"""
    
    def __init__(self, cache: IntelligentCache):
        self.cache = cache
        self.prediction_model = None
        self.feature_history = []
        
    async def initialize(self):
        """初始化预测模型"""
        # 这里可以加载机器学习模型
        # 简化实现，使用基于规则的预测
        logger.info("Predictive cache manager initialized")
    
    async def predict_and_prefetch(self, current_key: str, related_keys: List[str], 
                                  fetch_func: callable):
        """预测并预取相关键"""
        # 分析当前访问模式
        current_pattern = self.cache.pattern_analyzer.access_patterns.get(current_key, [])
        
        if len(current_pattern) < 3:
            return  # 历史数据不足
        
        # 预测接下来可能访问的键
        predictions = []
        for related_key in related_keys:
            if related_key not in self.cache.cache_data:
                # 计算预测分数
                score = await self._calculate_prediction_score(current_key, related_key)
                predictions.append((related_key, score))
        
        # 按预测分数排序
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # 预取高分项
        prefetch_threshold = 0.3
        for key, score in predictions:
            if score > prefetch_threshold:
                await self.cache.prefetch([key], fetch_func)
                logger.debug(f"Prefetched predicted key: {key} (score: {score:.3f})")
    
    async def _calculate_prediction_score(self, current_key: str, predicted_key: str) -> float:
        """计算预测分数"""
        # 简化的预测逻辑
        current_pattern = self.cache.pattern_analyzer.access_patterns.get(current_key, [])
        predicted_pattern = self.cache.pattern_analyzer.access_patterns.get(predicted_key, [])
        
        # 时间相关性
        if current_pattern and predicted_pattern:
            current_times = [p["timestamp"] for p in current_pattern[-10:]]
            predicted_times = [p["timestamp"] for p in predicted_pattern[-10:]]
            
            # 计算时间相关性
            correlation = self._calculate_temporal_correlation(current_times, predicted_times)
        else:
            correlation = 0.0
        
        # 键的相似性
        key_similarity = self._calculate_key_similarity(current_key, predicted_key)
        
        # 综合预测分数
        prediction_score = 0.6 * correlation + 0.4 * key_similarity
        
        return prediction_score
    
    def _calculate_temporal_correlation(self, times1: List[datetime], times2: List[datetime]) -> float:
        """计算时间相关性"""
        if not times1 or not times2:
            return 0.0
        
        # 简化的相关性计算
        # 检查两个时间序列是否有相似的模式
        recent1 = times1[-5:] if len(times1) >= 5 else times1
        recent2 = times2[-5:] if len(times2) >= 5 else times2
        
        if len(recent1) != len(recent2):
            return 0.0
        
        # 计算时间间隔的相似度
        intervals1 = [(recent1[i] - recent1[i-1]).total_seconds() for i in range(1, len(recent1))]
        intervals2 = [(recent2[i] - recent2[i-1]).total_seconds() for i in range(1, len(recent2))]
        
        if not intervals1 or not intervals2:
            return 0.0
        
        # 简单的相关性计算
        correlation = 1.0 - min(1.0, np.std(intervals1) / (np.mean(intervals1) + 0.001))
        
        return correlation
    
    def _calculate_key_similarity(self, key1: str, key2: str) -> float:
        """计算键的相似性"""
        # 简单的字符串相似性
        common_chars = set(key1.lower()) & set(key2.lower())
        total_chars = set(key1.lower()) | set(key2.lower())
        
        if not total_chars:
            return 0.0
        
        similarity = len(common_chars) / len(total_chars)
        return similarity

# 全局缓存实例
_intelligent_cache = None
_predictive_manager = None

async def get_intelligent_cache(config: Dict[str, Any]) -> IntelligentCache:
    """获取智能缓存实例"""
    global _intelligent_cache
    if _intelligent_cache is None:
        _intelligent_cache = IntelligentCache(config)
        await _intelligent_cache.initialize()
    return _intelligent_cache

async def get_predictive_manager(cache: IntelligentCache) -> PredictiveCacheManager:
    """获取预测性缓存管理器"""
    global _predictive_manager
    if _predictive_manager is None:
        _predictive_manager = PredictiveCacheManager(cache)
        await _predictive_manager.initialize()
    return _predictive_manager

if __name__ == "__main__":
    async def test_intelligent_cache():
        """测试智能缓存"""
        config = {
            "strategy": "predictive",
            "tiered_cache": {
                "l1_size": 10,
                "l2_size": 50,
                "l3_size": 100
            }
        }
        
        cache = await get_intelligent_cache(config)
        
        # 测试基本操作
        await cache.put("key1", "value1")
        value = await cache.get("key1")
        print(f"Got value: {value}")
        
        # 测试统计
        stats = cache.get_stats()
        print(f"Cache stats: {stats}")
        
        # 清理
        await cache.destroy()
    
    asyncio.run(test_intelligent_cache())