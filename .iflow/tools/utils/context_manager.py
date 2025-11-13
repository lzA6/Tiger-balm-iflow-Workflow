#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能上下文管理和压缩系统
Intelligent Context Management and Compression System

作者: Quantum AI Team
版本: 5.2.0
日期: 2025-11-12
"""

import os
import re
import json
import time
import pickle
import hashlib
import zlib
import lzma
import gzip
import bz2
import asyncio
import logging
import threading
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
from collections import defaultdict, deque, OrderedDict
import numpy as np
from datetime import datetime, timedelta
import sys

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompressionAlgorithm(Enum):
    """压缩算法"""
    NONE = "none"
    ZLIB = "zlib"
    LZMA = "lzma"
    GZIP = "gzip"
    BZ2 = "bz2"
    ADAPTIVE = "adaptive"

class ContextPriority(Enum):
    """上下文优先级"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    TEMPORARY = "temporary"

@dataclass
class ContextChunk:
    """上下文块"""
    chunk_id: str
    content: str
    size: int
    compressed_size: int
    compression_algorithm: CompressionAlgorithm
    priority: ContextPriority
    tags: Set[str]
    access_count: int
    last_accessed: float
    created_at: float
    expires_at: Optional[float]
    dependencies: Set[str]
    metadata: Dict[str, Any]

@dataclass
class ContextIndex:
    """上下文索引"""
    term: str
    chunk_ids: List[str]
    frequency: int
    last_accessed: float
    relevance_score: float

@dataclass
class ContextMetrics:
    """上下文指标"""
    total_chunks: int
    total_size: int
    compressed_size: int
    compression_ratio: float
    access_frequency: float
    hit_rate: float
    memory_usage: float
    cache_efficiency: float

class ContextCompressor:
    """上下文压缩器"""
    
    def __init__(self):
        """初始化压缩器"""
        self.compression_methods = {
            CompressionAlgorithm.ZLIB: self._compress_zlib,
            CompressionAlgorithm.LZMA: self._compress_lzma,
            CompressionAlgorithm.GZIP: self._compress_gzip,
            CompressionAlgorithm.BZ2: self._compress_bz2
        }
        
        self.decompression_methods = {
            CompressionAlgorithm.ZLIB: self._decompress_zlib,
            CompressionAlgorithm.LZMA: self._decompress_lzma,
            CompressionAlgorithm.GZIP: self._decompress_gzip,
            CompressionAlgorithm.BZ2: self._decompress_bz2
        }
        
        self.algorithm_performance = defaultdict(list)
    
    def compress(self, content: str, algorithm: CompressionAlgorithm) -> Tuple[bytes, CompressionAlgorithm]:
        """压缩内容"""
        if algorithm == CompressionAlgorithm.NONE:
            return content.encode('utf-8'), algorithm
        
        if algorithm == CompressionAlgorithm.ADAPTIVE:
            # 自适应选择最佳压缩算法
            algorithm = self._select_best_algorithm(content)
        
        if algorithm in self.compression_methods:
            start_time = time.time()
            compressed = self.compression_methods[algorithm](content)
            compression_time = time.time() - start_time
            
            # 记录性能
            self.algorithm_performance[algorithm].append({
                'size': len(content),
                'compressed_size': len(compressed),
                'time': compression_time,
                'ratio': len(compressed) / len(content)
            })
            
            return compressed, algorithm
        
        # 默认使用zlib
        return self._compress_zlib(content), CompressionAlgorithm.ZLIB
    
    def decompress(self, compressed: bytes, algorithm: CompressionAlgorithm) -> str:
        """解压缩内容"""
        if algorithm == CompressionAlgorithm.NONE:
            return compressed.decode('utf-8')
        
        if algorithm in self.decompression_methods:
            return self.decompression_methods[algorithm](compressed)
        
        raise ValueError(f"不支持的压缩算法: {algorithm}")
    
    def _compress_zlib(self, content: str) -> bytes:
        """ZLIB压缩"""
        return zlib.compress(content.encode('utf-8'))
    
    def _decompress_zlib(self, compressed: bytes) -> str:
        """ZLIB解压缩"""
        return zlib.decompress(compressed).decode('utf-8')
    
    def _compress_lzma(self, content: str) -> bytes:
        """LZMA压缩"""
        return lzma.compress(content.encode('utf-8'))
    
    def _decompress_lzma(self, compressed: bytes) -> str:
        """LZMA解压缩"""
        return lzma.decompress(compressed).decode('utf-8')
    
    def _compress_gzip(self, content: str) -> bytes:
        """GZIP压缩"""
        return gzip.compress(content.encode('utf-8'))
    
    def _decompress_gzip(self, compressed: bytes) -> str:
        """GZIP解压缩"""
        return gzip.decompress(compressed).decode('utf-8')
    
    def _compress_bz2(self, content: str) -> bytes:
        """BZ2压缩"""
        return bz2.compress(content.encode('utf-8'))
    
    def _decompress_bz2(self, compressed: bytes) -> str:
        """BZ2解压缩"""
        return bz2.decompress(compressed).decode('utf-8')
    
    def _select_best_algorithm(self, content: str) -> CompressionAlgorithm:
        """选择最佳压缩算法"""
        algorithms = [
            CompressionAlgorithm.ZLIB,
            CompressionAlgorithm.LZMA,
            CompressionAlgorithm.GZIP,
            CompressionAlgorithm.BZ2
        ]
        
        best_algorithm = CompressionAlgorithm.ZLIB
        best_ratio = 1.0
        
        for algorithm in algorithms:
            try:
                compressed = self.compression_methods[algorithm](content)
                ratio = len(compressed) / len(content)
                
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_algorithm = algorithm
                    
            except Exception:
                continue
        
        return best_algorithm
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = {}
        
        for algorithm, records in self.algorithm_performance.items():
            if records:
                avg_ratio = sum(r['ratio'] for r in records) / len(records)
                avg_time = sum(r['time'] for r in records) / len(records)
                
                stats[algorithm.value] = {
                    'uses': len(records),
                    'avg_compression_ratio': avg_ratio,
                    'avg_compression_time': avg_time,
                    'total_size_saved': sum(r['size'] - r['compressed_size'] for r in records)
                }
        
        return stats

class ContextCache:
    """上下文缓存"""
    
    def __init__(self, max_size: int = 1000, max_memory: int = 100 * 1024 * 1024):
        """初始化缓存"""
        self.max_size = max_size
        self.max_memory = max_memory
        
        # 使用有序字典实现LRU
        self.cache = OrderedDict()
        self.current_memory = 0
        self.access_order = deque(maxlen=max_size)
        self.hit_count = 0
        self.miss_count = 0
        
        # 锁统计
        self.lock = threading.RLock()
    
    def get(self, chunk_id: str) -> Optional[ContextChunk]:
        """获取上下文块"""
        with self.lock:
            if chunk_id in self.cache:
                chunk = self.cache[chunk_id]
                chunk.access_count += 1
                chunk.last_accessed = time.time()
                
                # 移动到末尾（LRU）
                self.cache.move_to_end(chunk_id)
                self.hit_count += 1
                
                return chunk
            else:
                self.miss_count += 1
                return None
    
    def put(self, chunk: ContextChunk):
        """存储上下文块"""
        with self.lock:
            # 检查内存限制
            if chunk_id not in self.cache:
                self.current_memory += chunk.compressed_size
                
                # 如果超出内存限制，移除最旧的块
                while (self.current_memory > self.max_memory or 
                       len(self.cache) >= self.max_size):
                    oldest_id = next(iter(self.cache))
                    oldest_chunk = self.cache[oldest_id]
                    self.current_memory -= oldest_chunk.compressed_size
                    del self.cache[oldest_id]
                    self.access_order.popleft()
            
            # 更新或添加
            self.cache[chunk_id] = chunk
            chunk.access_count = 1
            chunk.last_accessed = time.time()
            
            # 移动到末尾（LRU）
            if chunk_id in self.cache:
                self.cache.move_to_end(chunk_id)
            else:
                self.access_order.append(chunk_id)
    
    def remove(self, chunk_id: str) -> Optional[ContextChunk]:
        """移除上下文块"""
        with self.lock:
            if chunk_id in self.cache:
                chunk = self.cache.pop(chunk_id)
                self.current_memory -= chunk.compressed_size
                
                # 从访问顺序中移除
                if chunk_id in self.access_order:
                    self.access_order.remove(chunk_id)
                
                return chunk
            return None
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.current_memory = 0
            self.access_order.clear()
            self.hit_count = 0
            self.miss_count = 0
    
    def get_hit_rate(self) -> float:
        """获取命中率"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def get_memory_usage(self) -> int:
        """获取内存使用量"""
        return self.current_memory
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'memory_usage': self.current_memory,
            'max_memory': self.max_memory,
            'hit_rate': self.get_hit_rate(),
            'hit_count': self.hit_count,
            'miss_count': self.miss_count
        }

class ContextIndexer:
    """上下文索引器"""
    
    def __init__(self):
        """初始化索引器"""
        self.index = {}
        self.term_frequency = defaultdict(int)
        self.term_last_accessed = {}
        self.relevance_scores = {}
    
    def index_chunk(self, chunk: ContextChunk):
        """索引上下文块"""
        # 提取关键词
        terms = self._extract_terms(chunk.content)
        
        # 更新索引
        for term in terms:
            if term not in self.index:
                self.index[term] = []
            
            if chunk.chunk_id not in self.index[term]:
                self.index[term].append(chunk.chunk_id)
            
            # 更新频率
            self.term_frequency[term] += 1
            self.term_last_accessed[term] = time.time()
        
        # 计算相关性分数
        self._calculate_relevance_scores()
    
    def remove_chunk(self, chunk_id: str):
        """移除上下文块索引"""
        for term, chunk_ids in self.index.items():
            if chunk_id in chunk_ids:
                chunk_ids.remove(chunk_id)
                if not chunk_ids:
                    del self.index[term]
                    del self.term_frequency[term]
                    del self.term_last_accessed[term]
                    del self.relevance_scores[term]
    
    def _extract_terms(self, content: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取
        # 可以使用更复杂的NLP技术
        
        # 移除标点符号并分词
        words = re.findall(r'\b\w+\b', content.lower())
        
        # 过滤停用词
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'must', 'shall', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'them', 'their', 'what', 'which', 'who', 'when', 'where',
            'why', 'how', 'all', 'any', 'both', 'each', 'every', 'other', 'another', 'same'
        }
        
        # 过滤短词和停用词
        terms = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # 去重
        return list(set(terms))
    
    def _calculate_relevance_scores(self):
        """计算相关性分数"""
        current_time = time.time()
        max_frequency = max(self.term_frequency.values()) if self.term_frequency else 1
        
        for term in self.term_frequency:
            # 基于频率和时间衰减计算分数
            frequency_score = self.term_frequency[term] / max_frequency
            time_decay = np.exp(-(current_time - self.term_last_accessed[term]) / (24 * 60 * 60))  # 24小时衰减
            
            self.relevance_scores[term] = frequency_score * time_decay
    
    def search(self, query: str, limit: int = 10) -> List[str]:
        """搜索相关上下文块"""
        query_terms = self._extract_terms(query)
        
        # 计算查询词的相关性分数
        term_scores = {}
        for term in query_terms:
            term_scores[term] = self.relevance_scores.get(term, 0.0)
        
        # 搜索包含查询词的上下文块
        chunk_scores = defaultdict(float)
        
        for term, score in term_scores.items():
            if term in self.index:
                for chunk_id in self.index[term]:
                    chunk_scores[chunk_id] += score
        
        # 排序并返回最相关的块ID
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [chunk_id for chunk_id, _ in sorted_chunks[:limit]]
    
    def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计"""
        return {
            'total_terms': len(self.index),
            'total_frequency': sum(self.term_frequency.values()),
            'indexed_chunks': sum(len(chunk_ids) for chunk_ids in self.index.values()),
            'avg_relevance_score': np.mean(list(self.relevance_scores.values())) if self.relevance_scores else 0
        }

class ContextManager:
    """上下文管理器"""
    
    def __init__(self, 
                 max_chunks: int = 1000,
                 max_memory: int = 100 * 1024 * 1024,
                 default_ttl: int = 3600):  # 1小时
                 ):
        """初始化上下文管理器"""
        self.max_chunks = max_chunks
        self.max_memory = max_memory
        self.default_ttl = default_ttl
        
        # 核心组件
        self.compressor = ContextCompressor()
        self.cache = ContextCache(max_chunks, max_memory)
        self.indexer = ContextIndexer()
        
        # 上下文块存储
        self.chunks = {}
        
        # 配置
        self.config = {
            'auto_cleanup': True,
            'cleanup_interval': 300,  # 5分钟
            'compression_threshold': 1024,  # 1KB
            'priority_decay_rate': 0.1,
            'relevance_threshold': 0.3
        }
        
        # 统计信息
        self.metrics = ContextMetrics(
            total_chunks=0,
            total_size=0,
            compressed_size=0,
            compression_ratio=0.0,
            access_frequency=0.0,
            hit_rate=0.0,
            memory_usage=0.0,
            cache_efficiency=0.0
        )
        
        # 启动清理任务
        self.cleanup_task = None
        self.is_running = False
        
        # 锁
        self.lock = threading.RLock()
    
    def start(self):
        """启动上下文管理器"""
        if self.is_running:
            logger.warning("⚠️ 上下文管理器已在运行")
            return
        
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("🚀 上下文管理器已启动")
    
    async def stop(self):
        """停止上下文管理器"""
        self.is_running = False
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 上下文管理器已停止")
    
    async def _cleanup_loop(self):
        """清理循环"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config['cleanup_interval'])
                
                if self.config['auto_cleanup']:
                    await self._cleanup_expired_chunks()
                    await self._cleanup_low_priority_chunks()
                    await self._update_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ 清理循环错误: {e}")
                await asyncio.sleep(60)
    
    def add_context(self, 
                    content: str,
                    priority: ContextPriority = ContextPriority.MEDIUM,
                    tags: Optional[List[str]] = None,
                    ttl: Optional[int] = None,
                    dependencies: Optional[List[str]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """添加上下文"""
        with self.lock:
            chunk_id = f"chunk_{int(time.time())}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
            
            # 检查内容大小
            content_size = len(content.encode('utf-8'))
            
            # 压缩内容
            if content_size > self.config['compression_threshold']:
                compressed_content, algorithm = self.compressor.compress(content, CompressionAlgorithm.ADAPTIVE)
            else:
                compressed_content = content.encode('utf-8')
                algorithm = CompressionAlgorithm.NONE
            
            # 创建上下文块
            expires_at = time.time() + (ttl or self.default_ttl)
            
            chunk = ContextChunk(
                chunk_id=chunk_id,
                content=content,
                size=content_size,
                compressed_size=len(compressed_content),
                compression_algorithm=algorithm,
                priority=priority,
                tags=set(tags or []),
                access_count=0,
                last_accessed=time.time(),
                created_at=time.time(),
                expires_at=expires_at,
                dependencies=set(dependencies or []),
                metadata=metadata or {}
            )
            
            # 存储块
            self.chunks[chunk_id] = chunk
            self.cache.put(chunk)
            
            # 索引块
            self.indexer.index_chunk(chunk)
            
            logger.debug(f"📝 添加上下文块: {chunk_id} (大小: {content_size} 字节)")
            
            return chunk_id
    
    def get_context(self, chunk_id: str) -> Optional[str]:
        """获取上下文"""
        with self.lock:
            chunk = self.cache.get(chunk_id)
            
            if chunk:
                # 检查是否过期
                if chunk.expires_at and time.time() > chunk.expires_at:
                    self.remove_context(chunk_id)
                    return None
                
                return chunk.content
            
            return None
    
    def remove_context(self, chunk_id: str) -> bool:
        """移除上下文"""
        with self.lock:
            # 从缓存移除
            chunk = self.cache.remove(chunk_id)
            
            # 从存储移除
            if chunk_id in self.chunks:
                del self.chunks[chunk_id]
            
            # 从索引移除
            self.indexer.remove_chunk(chunk_id)
            
            logger.debug(f"🗑️ 移除上下文块: {chunk_id}")
            
            return chunk is not None
    
    def search_context(self, 
                    query: str, 
                    limit: int = 10,
                    min_relevance: float = 0.3) -> List[str]:
        """搜索上下文"""
        with self.lock:
            # 使用索引器搜索
            candidate_ids = self.indexer.search(query, limit * 2)  # 获取更多候选
            
            # 过滤相关性分数
            relevant_ids = []
            for chunk_id in candidate_ids:
                chunk = self.cache.get(chunk_id)
                if chunk:
                    # 检查是否过期
                    if chunk.expires_at and time.time() > chunk.expires_at:
                        continue
                    
                    # 检查相关性
                    terms = self.indexer._extract_terms(query)
                    relevance = sum(self.indexer.relevance_scores.get(term, 0) for term in terms if term in self.indexer.index)
                    
                    if relevance >= min_relevance:
                        relevant_ids.append(chunk_id)
            
            # 按相关性排序
            relevant_ids.sort(key=lambda x: self.indexer.relevance_scores.get(x.split('_')[0], 0), reverse=True)
            
            return relevant_ids[:limit]
    
    def get_related_contexts(self, chunk_id: str, limit: int = 5) -> List[str]:
        """获取相关上下文"""
        with self.lock:
            chunk = self.cache.get(chunk_id)
            
            if not chunk:
                return []
            
            # 基于依赖关系查找相关块
            related_ids = []
            
            for dep_id in chunk.dependencies:
                if dep_id in self.chunks:
                    related_ids.append(dep_id)
            
            # 基于共同标签查找相关块
            for other_id, other_chunk in self.chunks.items():
                if (other_id != chunk_id and 
                    chunk.tags & other_chunk.tags and
                    other_id not in related_ids):
                    related_ids.append(other_id)
            
            # 按优先级和最近访问时间排序
            related_ids.sort(key=lambda x: (
                self.chunks[x].priority.value * 0.5 + 
                (time.time() - self.chunks[x].last_accessed) * 0.5
            ), reverse=True)
            
            return related_ids[:limit]
    
    def update_context_priority(self, chunk_id: str, priority: ContextPriority):
        """更新上下文优先级"""
        with self.lock:
            chunk = self.cache.get(chunk_id)
            
            if chunk:
                old_priority = chunk.priority
                chunk.priority = priority
                chunk.last_accessed = time.time()
                
                # 调整相关块的优先级
                for related_id in chunk.dependencies:
                    related_chunk = self.cache.get(related_id)
                    if related_chunk:
                        # 轻微调整优先级
                        priority_diff = (priority.value - old_priority.value) * self.config['priority_decay_rate']
                        new_priority_value = related_chunk.priority.value + priority_diff
                        related_chunk.priority = ContextPriority(
                            min(max(0.0, max(1.0, new_priority_value))
                        )
                
                logger.debug(f"🔄 更新上下文优先级: {chunk_id} -> {priority.value}")
    
    def get_context_metrics(self) -> ContextMetrics:
        """获取上下文指标"""
        with self.lock:
            # 更新指标
            self._update_metrics()
            
            return self.metrics
    
    def _update_metrics(self):
        """更新指标"""
        # 计算总大小
        total_size = sum(chunk.size for chunk in self.chunks.values())
        compressed_size = sum(chunk.compressed_size for chunk in self.chunks.values())
        
        # 计算压缩比
        compression_ratio = compressed_size / total_size if total_size > 0 else 1.0
        
        # 计算访问频率
        total_accesses = sum(chunk.access_count for chunk in self.chunks.values())
        time_window = 3600  # 1小时窗口
        recent_accesses = sum(
            chunk.access_count for chunk in self.chunks.values()
            if time.time() - chunk.last_accessed <= time_window
        )
        access_frequency = recent_accesses / max(1, len(self.chunks))
        
        # 获取缓存统计
        cache_info = self.cache.get_cache_info()
        
        # 计算缓存效率
        memory_usage = cache_info['memory_usage']
        max_memory = cache_info['max_memory']
        cache_efficiency = 1.0 - (memory_usage / max_memory) if max_memory > 0 else 1.0
        
        self.metrics = ContextMetrics(
            total_chunks=len(self.chunks),
            total_size=total_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            access_frequency=access_frequency,
            hit_rate=cache_info['hit_rate'],
            memory_usage=memory_usage,
            cache_efficiency=cache_efficiency
        )
    
    async def _cleanup_expired_chunks(self):
        """清理过期上下文"""
        current_time = time.time()
        expired_ids = []
        
        for chunk_id, chunk in self.chunks.items():
            if chunk.expires_at and current_time > chunk.expires_at:
                expired_ids.append(chunk_id)
        
        for chunk_id in expired_ids:
            self.remove_context(chunk_id)
        
        if expired_ids:
            logger.info(f"🗑️ 清理了 {len(expired_ids)} 个过期上下文块")
    
    async def _cleanup_low_priority_chunks(self):
        """清理低优先级上下文"""
        with self.lock:
            # 按优先级和访问时间排序
            sorted_chunks = sorted(
                self.chunks.items(),
                key=lambda x: (
                    x[1].priority.value * 0.3 + 
                    (current_time - x[1].last_accessed) * 0.7
                )
            )
            
            # 计算需要移除的数量
            excess_count = len(self.chunks) - self.max_chunks
            
            if excess_count > 0:
                # 移除最低优先级的块
                for i in range(excess_count):
                    if i < len(sorted_chunks):
                        chunk_id = sorted_chunks[i][0]
                        self.remove_context(chunk_id)
                        
                        logger.debug(f"🗑️ 清理低优先级上下文块: {chunk_id}")
                
                logger.info(f"🗑️ 清理了 {excess_count} 个低优先级上下文块")
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """获取压缩统计"""
        return self.compressor.get_performance_stats()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self.cache.get_cache_info()
    
    def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计"""
        return self.indexer.get_index_stats()
    
    def get_full_stats(self) -> Dict[str, Any]:
        """获取完整统计"""
        return {
            'chunks': {
                'total': len(self.chunks),
                'by_priority': {
                    priority.value: len([c for c in self.chunks.values() if c.priority == priority])
                    for priority in ContextPriority
                },
                'by_type': {
                    pattern_type: len([c for c in self.chunks.values() if pattern_type in c.tags])
                    for pattern_type in set(tag for c in self.chunks.values() for tag in c.tags)
                }
            },
            'compression': self.get_compression_stats(),
            'cache': self.get_cache_stats(),
            'index': self.get_index_stats(),
            'metrics': asdict(self.get_context_metrics())
        }

# 全局上下文管理器实例
context_manager = ContextManager()

# 便捷函数
def add_context(content: str, 
                priority: ContextPriority = ContextPriority.MEDIUM,
                tags: Optional[List[str]] = None,
                ttl: Optional[int] = None) -> str:
    """便捷的上下文添加函数"""
    return context_manager.add_context(
        content=content,
        priority=priority,
        tags=tags,
        ttl=ttl
    )

def get_context(chunk_id: str) -> Optional[str]:
    """便捷的上下文获取函数"""
    return context_manager.get_context(chunk_id)

def search_context(query: str, limit: int = 10) -> List[str]:
    """便捷的上下文搜索函数"""
    return context_manager.search_context(query, limit)

def get_related_contexts(chunk_id: str, limit: int = 5) -> List[str]:
    """便捷的相关上下文获取函数"""
    return context_manager.get_related_contexts(chunk_id, limit)

# 示例使用
async def example_usage():
    """示例使用"""
    print("🧠 智能上下文管理和压缩系统示例")
    
    # 启动上下文管理器
    print("\n1. 启动上下文管理器:")
    context_manager.start()
    
    # 添加上下文
    print("\n2. 添加上下文:")
    
    # 高优先级上下文
    chunk_id1 = add_context(
        content="这是一个重要的系统配置文件，包含数据库连接信息和API密钥。",
        priority=ContextPriority.CRITICAL,
        tags=['config', 'database', 'security'],
        ttl=7200  # 2小时
    )
    print(f"  添加关键上下文: {chunk_id1}")
    
    # 中等优先级上下文
    chunk_id2 = add_context(
        content="这是一个用户手册文档，详细说明了系统的使用方法和最佳实践。",
        priority=ContextPriority.MEDIUM,
        tags=['documentation', 'user_guide'],
        ttl=3600  # 1小时
    )
    print(f"  添加文档上下文: {chunk_id2}")
    
    # 低优先级上下文
    chunk_id3 = add_context(
        content="这是一个临时的调试信息，将在系统重启后清理。",
        priority=ContextPriority.TEMPORARY,
        tags=['debug', 'temporary'],
        ttl=300  # 5分钟
    )
    print(f"  添加临时上下文: {chunk_id3}")
    
    # 大内容（会被压缩）
    large_content = "这是一个很大的文本内容，包含大量的代码示例和详细说明。" * 1000
    chunk_id4 = add_context(
        content=large_content,
        priority=ContextPriority.LOW,
        tags=['large', 'examples'],
        ttl=1800  # 30分钟
    )
    print(f"添加大内容上下文: {chunk_id4} (会被压缩)")
    
    # 搜索上下文
    print("\n3. 搜索上下文:")
    search_results = search_context("系统 配置", limit=3)
    for i, result_id in enumerate(search_results, 1):
        content = get_context(result_id)
        if content:
            print(f"  {i}. {result_id}: {content[:50]}...")
    
    # 获取相关上下文
    print("\n4. 获取相关上下文:")
    related_ids = get_related_contexts(chunk_id1, limit=3)
    for i, related_id in enumerate(related_ids, 1):
        content = get_context(related_id)
        if content:
            print(f"  {i}. {related_id}: {content[:50]}...")
    
    # 获取统计信息
    print("\n5. 统计信息:")
    stats = context_manager.get_full_stats()
    
    print(f"  总上下文块数: {stats['chunks']['total']}")
    print(f"  总大小: {stats['metrics']['total_size']} 字节")
    print(f"  压缩后大小: {stats['metrics']['compressed_size']} 字节")
    print(f"  压缩比: {stats['metrics']['compression_ratio']:.2f}")
    print(f"  缓存命中率: {stats['cache']['hit_rate']:.2f}")
    print(f"  内存使用: {stats['cache']['memory_usage'] / (1024*1024):.2f} MB")
    print(f"  缓存效率: {stats['metrics']['cache_efficiency']:.2f}")
    
    # 压缩统计
    print("\n6. 压缩算法统计:")
    comp_stats = context_manager.get_compression_stats()
    for algorithm, stats in comp_stats.items():
        print(f"  {algorithm}:")
        print(f"    使用次数: {stats['uses']}")
        print(f"    平均压缩比: {stats['avg_compression_ratio']:.2f}")
        print(f"    平均压缩时间: {stats['avg_compression_time']:.4f}s")
    
    # 等待一段时间让系统运行
    print("\n7. 等待系统运行...")
    await asyncio.sleep(2)
    
    # 再次获取统计信息
    print("\n8. 更新后的统计信息:")
    updated_stats = context_manager.get_full_stats()
    print(f"  缓存命中率: {updated_stats['cache']['hit_rate']:.2f}")
    
    # 停止上下文管理器
    print("\n9. 停止上下文管理器:")
    await context_manager.stop()
    
    print("\n✅ 智能上下文管理和压缩系统示例完成")

if __name__ == "__main__":
    asyncio.run(example_usage())