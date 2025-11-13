#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 智能上下文感知缓存系统 V5 (Intelligent Context-Aware Cache V5)
基于AI的智能缓存系统，能够理解上下文语义、预测访问模式、自动优化缓存策略。
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
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, OrderedDict
import threading
import sqlite3
import uuid
import warnings
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """缓存策略"""
    LRU = "lru"
    LFU = "lfu"
    SEMANTIC = "semantic"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"

class ContextType(Enum):
    """上下文类型"""
    TASK = "task"
    CONVERSATION = "conversation"
    CODE = "code"
    DOCUMENT = "document"
    QUERY = "query"
    RESULT = "result"

@dataclass
class CacheEntry:
    """缓存条目"""
    id: str
    key: str
    value: Any
    context_type: ContextType
    context_vector: Optional[np.ndarray]
    access_count: int = 0
    access_frequency: float = 0.0
    last_access: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    size_bytes: int = 0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    semantic_neighbors: Set[str] = field(default_factory=set)
    access_pattern: List[datetime] = field(default_factory=list)

@dataclass
class AccessPattern:
    """访问模式"""
    entry_id: str
    timestamps: List[datetime]
    patterns: Dict[str, Any] = field(default_factory=dict)
    predicted_next_access: Optional[datetime] = None
    prediction_confidence: float = 0.0

class IntelligentContextCacheV5:
    """
    智能上下文感知缓存系统 V5
    """
    
    def __init__(self, 
                 max_size: int = 10000,
                 max_memory_mb: int = 1024,
                 db_path: str = "A项目/iflow/data/context_cache_v5.db"):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()
        
        # 缓存存储
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_memory_usage = 0
        
        # 语义模型
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.vector_index = faiss.IndexFlatL2(self.embedding_dim)
        self.id_to_vector: Dict[str, np.ndarray] = {}
        
        # 访问模式分析
        self.access_patterns: Dict[str, AccessPattern] = {}
        self.pattern_analyzer = None
        
        # 缓存策略
        self.current_strategy = CacheStrategy.ADAPTIVE
        self.strategy_performance = defaultdict(float)
        
        # 聚类模型（用于语义分组）
        self.cluster_model = KMeans(n_clusters=50, random_state=42)
        self.cluster_labels = {}
        
        # 预测模型
        self.prediction_model = None
        self._initialize_prediction_model()
        
        # 线程安全
        self.lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "semantic_hits": 0,
            "predictive_hits": 0,
            "total_requests": 0
        }
        
        # 后台任务
        self.background_tasks = []
        self.running = True
        
        # 启动后台任务
        self._start_background_tasks()
        
        logger.info("智能上下文感知缓存系统V5初始化完成")
    
    def _init_db(self):
        """初始化数据库"""
        with self.conn:
            # 缓存条目表
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id TEXT PRIMARY KEY,
                    key TEXT,
                    value_blob BLOB,
                    context_type TEXT,
                    context_vector BLOB,
                    access_count INTEGER DEFAULT 0,
                    access_frequency REAL DEFAULT 0.0,
                    last_access REAL,
                    created_at REAL,
                    expires_at REAL,
                    size_bytes INTEGER,
                    tags TEXT,
                    metadata TEXT,
                    semantic_neighbors TEXT
                )
            """)
            
            # 访问模式表
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS access_patterns (
                    entry_id TEXT PRIMARY KEY,
                    timestamps TEXT,
                    patterns TEXT,
                    predicted_next_access REAL,
                    prediction_confidence REAL
                )
            """)
            
            # 策略性能表
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    strategy TEXT PRIMARY KEY,
                    hit_rate REAL,
                    memory_efficiency REAL,
                    access_latency REAL,
                    updated_at REAL
                )
            """)
    
    def _initialize_prediction_model(self):
        """初始化预测模型"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            self.prediction_model = RandomForestRegressor(n_estimators=100, random_state=42)
            logger.info("预测模型初始化成功")
        except ImportError:
            logger.warning("sklearn未安装，使用简单预测模型")
            self.prediction_model = None
    
    def _start_background_tasks(self):
        """启动后台任务"""
        # 定期清理过期条目
        task1 = asyncio.create_task(self._periodic_cleanup())
        self.background_tasks.append(task1)
        
        # 更新访问模式分析
        task2 = asyncio.create_task(self._update_access_patterns())
        self.background_tasks.append(task2)
        
        # 优化缓存策略
        task3 = asyncio.create_task(self._optimize_cache_strategy())
        self.background_tasks.append(task3)
        
        # 重建向量索引
        task4 = asyncio.create_task(self._rebuild_vector_index())
        self.background_tasks.append(task4)
    
    async def get(self, key: str, context: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        获取缓存值，支持语义相似性搜索
        """
        with self.lock:
            self.stats["total_requests"] += 1
            
            # 1. 精确匹配
            if key in self.cache:
                entry = self.cache[key]
                
                # 检查是否过期
                if entry.expires_at and datetime.now() > entry.expires_at:
                    await self._remove_entry(key)
                    self.stats["misses"] += 1
                    return None
                
                # 更新访问信息
                self._update_access_info(entry)
                self.stats["hits"] += 1
                return entry.value
            
            # 2. 语义搜索
            if context:
                semantic_result = await self._semantic_search(key, context)
                if semantic_result:
                    self.stats["semantic_hits"] += 1
                    return semantic_result
            
            # 3. 预测性缓存
            predictive_result = await self._predictive_search(key, context)
            if predictive_result:
                self.stats["predictive_hits"] += 1
                return predictive_result
            
            self.stats["misses"] += 1
            return None
    
    async def put(self, 
                  key: str, 
                  value: Any, 
                  context_type: ContextType = ContextType.TASK,
                  ttl: Optional[int] = None,
                  tags: Optional[Set[str]] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        存储缓存值
        """
        with self.lock:
            # 计算值的大小
            value_bytes = pickle.dumps(value)
            size_bytes = len(value_bytes)
            
            # 检查内存限制
            if size_bytes > self.max_memory_bytes:
                logger.warning(f"缓存项太大: {size_bytes} bytes > {self.max_memory_bytes} bytes")
                return ""
            
            # 如果需要，清理空间
            await self._ensure_space(size_bytes)
            
            # 生成上下文向量
            context_vector = None
            if isinstance(key, str):
                try:
                    context_vector = self.embedding_model.encode([key])[0]
                except Exception as e:
                    logger.warning(f"生成上下文向量失败: {e}")
            
            # 创建缓存条目
            entry = CacheEntry(
                id=str(uuid.uuid4()),
                key=key,
                value=value,
                context_type=context_type,
                context_vector=context_vector,
                size_bytes=size_bytes,
                tags=tags or set(),
                metadata=metadata or {},
                expires_at=datetime.now() + timedelta(seconds=ttl) if ttl else None
            )
            
            # 存储到缓存
            self.cache[key] = entry
            self.current_memory_usage += size_bytes
            
            # 更新向量索引
            if context_vector is not None:
                vector_id = len(self.id_to_vector)
                self.vector_index.add(context_vector.reshape(1, -1))
                self.id_to_vector[vector_id] = entry.id
            
            # 查找语义邻居
            if context_vector is not None:
                await self._find_semantic_neighbors(entry)
            
            # 存储到数据库
            self._store_entry_in_db(entry)
            
            # 初始化访问模式
            self.access_patterns[entry.id] = AccessPattern(
                entry_id=entry.id,
                timestamps=[datetime.now()]
            )
            
            return entry.id
    
    async def _semantic_search(self, key: str, context: Dict[str, Any]) -> Optional[Any]:
        """语义搜索"""
        try:
            # 生成查询向量
            query_vector = self.embedding_model.encode([key])[0]
            
            # 搜索相似的向量
            k = min(5, len(self.id_to_vector))
            if k == 0:
                return None
            
            distances, indices = self.vector_index.search(query_vector.reshape(1, -1), k)
            
            # 检查相似度阈值
            similarity_threshold = 0.8
            for i, idx in enumerate(indices[0]):
                if idx in self.id_to_vector:
                    entry_id = self.id_to_vector[idx]
                    
                    # 找到对应的缓存条目
                    for entry in self.cache.values():
                        if entry.id == entry_id:
                            # 计算余弦相似度
                            if entry.context_vector is not None:
                                similarity = cosine_similarity(
                                    query_vector.reshape(1, -1),
                                    entry.context_vector.reshape(1, -1)
                                )[0][0]
                                
                                if similarity > similarity_threshold:
                                    # 更新访问信息
                                    self._update_access_info(entry)
                                    return entry.value
            
            return None
            
        except Exception as e:
            logger.error(f"语义搜索失败: {e}")
            return None
    
    async def _predictive_search(self, key: str, context: Optional[Dict[str, Any]]) -> Optional[Any]:
        """预测性搜索"""
        try:
            # 基于上下文和访问模式预测可能的缓存项
            if not context:
                return None
            
            # 简单的基于规则的预测
            predicted_keys = []
            
            # 如果是代码相关的查询
            if "code" in context.get("type", "").lower():
                # 查找相关的代码缓存
                for entry_key, entry in self.cache.items():
                    if entry.context_type == ContextType.CODE:
                        predicted_keys.append(entry_key)
            
            # 如果是对话上下文
            elif "conversation" in context.get("type", "").lower():
                # 查找最近的对话缓存
                recent_entries = sorted(
                    [e for e in self.cache.values() if e.context_type == ContextType.CONVERSATION],
                    key=lambda x: x.last_access,
                    reverse=True
                )[:3]
                predicted_keys.extend([e.key for e in recent_entries])
            
            # 返回第一个预测项
            if predicted_keys:
                predicted_key = predicted_keys[0]
                if predicted_key in self.cache:
                    entry = self.cache[predicted_key]
                    self._update_access_info(entry)
                    return entry.value
            
            return None
            
        except Exception as e:
            logger.error(f"预测性搜索失败: {e}")
            return None
    
    def _update_access_info(self, entry: CacheEntry):
        """更新访问信息"""
        now = datetime.now()
        entry.access_count += 1
        entry.last_access = now
        entry.access_pattern.append(now)
        
        # 计算访问频率
        time_diff = (now - entry.created_at).total_seconds()
        if time_diff > 0:
            entry.access_frequency = entry.access_count / time_diff
        
        # 更新访问模式
        if entry.id in self.access_patterns:
            self.access_patterns[entry.id].timestamps.append(now)
            
            # 限制时间戳数量
            if len(self.access_patterns[entry.id].timestamps) > 100:
                self.access_patterns[entry.id].timestamps = \
                    self.access_patterns[entry.id].timestamps[-50:]
        
        # 移动到LRU末尾
        self.cache.move_to_end(entry.key)
    
    async def _find_semantic_neighbors(self, entry: CacheEntry):
        """查找语义邻居"""
        if entry.context_vector is None:
            return
        
        try:
            # 搜索最相似的条目
            k = min(10, len(self.id_to_vector))
            if k == 0:
                return
            
            distances, indices = self.vector_index.search(
                entry.context_vector.reshape(1, -1), 
                k
            )
            
            # 添加语义邻居
            threshold = 0.7  # 相似度阈值
            for i, idx in enumerate(indices[0]):
                if idx in self.id_to_vector:
                    neighbor_id = self.id_to_vector[idx]
                    if neighbor_id != entry.id:
                        # 计算相似度
                        for e in self.cache.values():
                            if e.id == neighbor_id and e.context_vector is not None:
                                similarity = cosine_similarity(
                                    entry.context_vector.reshape(1, -1),
                                    e.context_vector.reshape(1, -1)
                                )[0][0]
                                
                                if similarity > threshold:
                                    entry.semantic_neighbors.add(neighbor_id)
                                    e.semantic_neighbors.add(entry.id)
            
        except Exception as e:
            logger.error(f"查找语义邻居失败: {e}")
    
    async def _ensure_space(self, required_bytes: int):
        """确保有足够空间"""
        # 清理过期条目
        await self._cleanup_expired()
        
        # 如果仍然需要空间，使用当前策略清理
        while (self.current_memory_usage + required_bytes > self.max_memory_bytes or
               len(self.cache) >= self.max_size):
            
            if self.current_strategy == CacheStrategy.LRU:
                await self._evict_lru()
            elif self.current_strategy == CacheStrategy.LFU:
                await self._evict_lfu()
            elif self.current_strategy == CacheStrategy.SEMANTIC:
                await self._evict_semantic()
            elif self.current_strategy == CacheStrategy.PREDICTIVE:
                await self._evict_predictive()
            else:  # ADAPTIVE
                await self._evict_adaptive()
    
    async def _evict_lru(self):
        """LRU淘汰"""
        if self.cache:
            key, entry = self.cache.popitem(last=False)
            await self._remove_entry(key)
            self.stats["evictions"] += 1
    
    async def _evict_lfu(self):
        """LFU淘汰"""
        if not self.cache:
            return
        
        # 找到访问频率最低的条目
        min_frequency = float('inf')
        lfu_key = None
        
        for key, entry in self.cache.items():
            if entry.access_frequency < min_frequency:
                min_frequency = entry.access_frequency
                lfu_key = key
        
        if lfu_key:
            await self._remove_entry(lfu_key)
            self.stats["evictions"] += 1
    
    async def _evict_semantic(self):
        """语义淘汰（保留语义多样性）"""
        if not self.cache:
            return
        
        # 找到语义密度最高的区域
        cluster_density = defaultdict(int)
        
        for entry in self.cache.values():
            if entry.context_vector is not None:
                # 简单的聚类
                cluster_id = hash(entry.context_vector.tobytes()) % 50
                cluster_density[cluster_id] += 1
        
        # 找到最密集的簇
        if cluster_density:
            densest_cluster = max(cluster_density.items(), key=lambda x: x[1])[0]
            
            # 从最密集的簇中淘汰一个
            for key, entry in self.cache.items():
                if entry.context_vector is not None:
                    cluster_id = hash(entry.context_vector.tobytes()) % 50
                    if cluster_id == densest_cluster:
                        await self._remove_entry(key)
                        self.stats["evictions"] += 1
                        break
    
    async def _evict_predictive(self):
        """预测性淘汰（基于预测的下次访问时间）"""
        if not self.cache:
            return
        
        # 找到最不可能被再次访问的条目
        min_priority = float('inf')
        evict_key = None
        
        for key, entry in self.cache.items():
            # 计算优先级分数
            score = 0
            
            # 访问频率权重
            score += entry.access_frequency * 0.3
            
            # 最近访问时间权重
            time_since_last = (datetime.now() - entry.last_access).total_seconds()
            score -= time_since_last * 0.001
            
            # 语义连接权重
            score += len(entry.semantic_neighbors) * 0.1
            
            # 大小惩罚
            score -= entry.size_bytes * 0.000001
            
            if score < min_priority:
                min_priority = score
                evict_key = key
        
        if evict_key:
            await self._remove_entry(evict_key)
            self.stats["evictions"] += 1
    
    async def _evict_adaptive(self):
        """自适应淘汰（结合多种策略）"""
        # 根据当前性能选择最佳策略
        if self.strategy_performance[CacheStrategy.LRU.value] > 0.8:
            await self._evict_lru()
        elif self.strategy_performance[CacheStrategy.LFU.value] > 0.8:
            await self._evict_lfu()
        elif self.strategy_performance[CacheStrategy.SEMANTIC.value] > 0.8:
            await self._evict_semantic()
        else:
            # 默认使用LRU
            await self._evict_lru()
    
    async def _remove_entry(self, key: str):
        """移除缓存条目"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_memory_usage -= entry.size_bytes
            
            # 从访问模式中移除
            if entry.id in self.access_patterns:
                del self.access_patterns[entry.id]
            
            # 从向量索引中移除（FAISS不支持删除，这里简化处理）
            # 实际应用中可能需要使用支持删除的向量数据库
    
    async def _cleanup_expired(self):
        """清理过期条目"""
        now = datetime.now()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if entry.expires_at and now > entry.expires_at:
                expired_keys.append(key)
        
        for key in expired_keys:
            await self._remove_entry(key)
    
    def _store_entry_in_db(self, entry: CacheEntry):
        """存储条目到数据库"""
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO cache_entries 
                (id, key, value_blob, context_type, context_vector, 
                 access_count, access_frequency, last_access, created_at, 
                 expires_at, size_bytes, tags, metadata, semantic_neighbors)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.key,
                    pickle.dumps(entry.value),
                    entry.context_type.value,
                    pickle.dumps(entry.context_vector) if entry.context_vector is not None else None,
                    entry.access_count,
                    entry.access_frequency,
                    entry.last_access.timestamp(),
                    entry.created_at.timestamp(),
                    entry.expires_at.timestamp() if entry.expires_at else None,
                    entry.size_bytes,
                    json.dumps(list(entry.tags)),
                    json.dumps(entry.metadata),
                    json.dumps(list(entry.semantic_neighbors))
                )
            )
    
    async def _periodic_cleanup(self):
        """定期清理"""
        while self.running:
            try:
                await asyncio.sleep(300)  # 5分钟
                await self._cleanup_expired()
                
                # 更新策略性能
                await self._update_strategy_performance()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"定期清理失败: {e}")
    
    async def _update_access_patterns(self):
        """更新访问模式"""
        while self.running:
            try:
                await asyncio.sleep(600)  # 10分钟
                
                for entry_id, pattern in self.access_patterns.items():
                    if len(pattern.timestamps) > 5:
                        # 分析访问模式
                        timestamps = pattern.timestamps
                        intervals = [
                            (timestamps[i+1] - timestamps[i]).total_seconds()
                            for i in range(len(timestamps)-1)
                        ]
                        
                        if intervals:
                            pattern.patterns = {
                                "avg_interval": np.mean(intervals),
                                "std_interval": np.std(intervals),
                                "min_interval": np.min(intervals),
                                "max_interval": np.max(intervals),
                                "access_count": len(timestamps),
                                "regularity": 1.0 / (np.std(intervals) + 1)
                            }
                            
                            # 预测下次访问时间
                            if self.prediction_model and len(intervals) > 10:
                                try:
                                    # 准备特征
                                    features = np.array([
                                        len(intervals),
                                        np.mean(intervals),
                                        np.std(intervals),
                                        np.min(intervals),
                                        np.max(intervals)
                                    ]).reshape(1, -1)
                                    
                                    # 预测（简化实现）
                                    predicted_interval = np.mean(intervals[-5:])
                                    pattern.predicted_next_access = timestamps[-1] + timedelta(seconds=predicted_interval)
                                    pattern.prediction_confidence = 0.7
                                except Exception as e:
                                    logger.warning(f"预测访问模式失败: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"更新访问模式失败: {e}")
    
    async def _optimize_cache_strategy(self):
        """优化缓存策略"""
        while self.running:
            try:
                await asyncio.sleep(1800)  # 30分钟
                
                # 评估各策略性能
                strategies = [CacheStrategy.LRU, CacheStrategy.LFU, 
                            CacheStrategy.SEMANTIC, CacheStrategy.PREDICTIVE]
                
                best_strategy = self.current_strategy
                best_performance = self.strategy_performance.get(self.current_strategy.value, 0.5)
                
                for strategy in strategies:
                    performance = self.strategy_performance.get(strategy.value, 0.5)
                    if performance > best_performance:
                        best_strategy = strategy
                        best_performance = performance
                
                if best_strategy != self.current_strategy:
                    logger.info(f"切换缓存策略: {self.current_strategy.value} -> {best_strategy.value}")
                    self.current_strategy = best_strategy
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"优化缓存策略失败: {e}")
    
    async def _rebuild_vector_index(self):
        """重建向量索引"""
        while self.running:
            try:
                await asyncio.sleep(3600)  # 1小时
                
                # 如果向量索引过大，重建
                if len(self.id_to_vector) > 10000:
                    logger.info("重建向量索引...")
                    
                    # 收集所有向量
                    vectors = []
                    id_mapping = {}
                    
                    for entry in self.cache.values():
                        if entry.context_vector is not None:
                            vectors.append(entry.context_vector)
                            id_mapping[len(vectors)-1] = entry.id
                    
                    # 重建索引
                    self.vector_index = faiss.IndexFlatL2(self.embedding_dim)
                    if vectors:
                        vectors_array = np.array(vectors)
                        self.vector_index.add(vectors_array)
                        self.id_to_vector = id_mapping
                    
                    logger.info(f"向量索引重建完成，包含{len(vectors)}个向量")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"重建向量索引失败: {e}")
    
    async def _update_strategy_performance(self):
        """更新策略性能"""
        total_requests = self.stats["total_requests"]
        if total_requests == 0:
            return
        
        hit_rate = self.stats["hits"] / total_requests
        
        # 更新当前策略性能
        self.strategy_performance[self.current_strategy.value] = hit_rate
        
        # 存储到数据库
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO strategy_performance 
                (strategy, hit_rate, memory_efficiency, access_latency, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    self.current_strategy.value,
                    hit_rate,
                    self.current_memory_usage / self.max_memory_bytes,
                    0.001,  # 模拟延迟
                    datetime.now().timestamp()
                )
            )
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self.stats["total_requests"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        # 按类型统计
        type_stats = defaultdict(int)
        for entry in self.cache.values():
            type_stats[entry.context_type.value] += 1
        
        # 访问模式统计
        pattern_stats = {
            "total_patterns": len(self.access_patterns),
            "regular_patterns": sum(1 for p in self.access_patterns.values() 
                                 if p.patterns.get("regularity", 0) > 0.5),
            "predicted_accesses": sum(1 for p in self.access_patterns.values() 
                                   if p.predicted_next_access is not None)
        }
        
        return {
            "total_entries": len(self.cache),
            "memory_usage": {
                "used": self.current_memory_usage,
                "max": self.max_memory_bytes,
                "percentage": self.current_memory_usage / self.max_memory_bytes
            },
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "semantic_hits": self.stats["semantic_hits"],
            "predictive_hits": self.stats["predictive_hits"],
            "evictions": self.stats["evictions"],
            "current_strategy": self.current_strategy.value,
            "type_distribution": dict(type_stats),
            "access_patterns": pattern_stats,
            "strategy_performance": dict(self.strategy_performance)
        }
    
    async def clear(self, pattern: Optional[str] = None):
        """清理缓存"""
        with self.lock:
            if pattern:
                # 按模式清理
                keys_to_remove = []
                for key in self.cache:
                    if pattern in key:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    await self._remove_entry(key)
                
                logger.info(f"清理了{len(keys_to_remove)}个匹配'{pattern}'的缓存条目")
            else:
                # 清理所有
                self.cache.clear()
                self.current_memory_usage = 0
                self.access_patterns.clear()
                
                # 重建向量索引
                self.vector_index = faiss.IndexFlatL2(self.embedding_dim)
                self.id_to_vector.clear()
                
                logger.info("缓存已完全清理")
    
    async def preload_cache(self, data: List[Tuple[str, Any, ContextType]], 
                           tags: Optional[Set[str]] = None):
        """预加载缓存"""
        logger.info(f"预加载{len(data)}个缓存条目...")
        
        for key, value, context_type in data:
            await self.put(
                key=key,
                value=value,
                context_type=context_type,
                tags=tags
            )
        
        logger.info("预加载完成")
    
    def close(self):
        """关闭缓存系统"""
        self.running = False
        
        # 取消后台任务
        for task in self.background_tasks:
            task.cancel()
        
        # 关闭数据库连接
        self.conn.close()
        
        logger.info("智能上下文感知缓存系统V5已关闭")