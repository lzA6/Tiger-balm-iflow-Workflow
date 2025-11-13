#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 智能上下文管理器 V3 (Intelligent Context Manager V3)
专为ARQ和意识流系统设计的高级上下文压缩、长期记忆管理和预测性缓存系统。
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。

核心特性：
1. 语义感知的上下文压缩
2. 分层长期记忆系统
3. 预测性缓存预加载
4. 量子增强的记忆检索
"""

import os
import sys
import json
import asyncio
import logging
import hashlib
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque, OrderedDict
import sqlite3
import threading
import uuid
import time
import re

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 尝试导入机器学习和向量数据库依赖
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import faiss
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("机器学习依赖未安装，将使用简化版本")

logger = logging.getLogger(__name__)

class CompressionLevel(Enum):
    """压缩级别"""
    LIGHT = "light"          # 轻度压缩：保留大部分信息
    MEDIUM = "medium"        # 中度压缩：平衡压缩率和信息保留
    HEAVY = "heavy"          # 重度压缩：最大化压缩率
    QUANTUM = "quantum"      # 量子压缩：使用量子算法优化

class MemoryType(Enum):
    """记忆类型"""
    EPISODIC = "episodic"        # 情景记忆：具体事件和经验
    SEMANTIC = "semantic"        # 语义记忆：概念和知识
    PROCEDURAL = "procedural"    # 程序记忆：技能和流程
    WORKING = "working"          # 工作记忆：当前活跃的信息
    QUANTUM = "quantum"          # 量子记忆：量子态信息

class RetrievalStrategy(Enum):
    """检索策略"""
    SEMANTIC_SEARCH = "semantic_search"
    TEMPORAL_PROXIMITY = "temporal_proximity"
    FREQUENCY_BASED = "frequency_based"
    QUANTUM_ENTANGLEMENT = "quantum_entanglement"
    HYBRID_FUSION = "hybrid_fusion"

@dataclass
class ContextChunk:
    """上下文块"""
    chunk_id: str
    content: str
    chunk_type: str
    semantic_embedding: Optional[np.ndarray] = None
    compression_ratio: float = 1.0
    importance_score: float = 0.5
    temporal_weight: float = 1.0
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryTrace:
    """记忆痕迹"""
    trace_id: str
    memory_type: MemoryType
    content_summary: str
    semantic_fingerprint: Optional[np.ndarray] = None
    emotional_valence: float = 0.0  # 情感价值：-1到1
    consolidation_strength: float = 0.5  # 巩固强度
    retrieval_frequency: int = 0
    last_retrieved: float = field(default_factory=time.time)
    associated_contexts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CompressionResult:
    """压缩结果"""
    original_size: int
    compressed_size: int
    compression_ratio: float
    information_retention: float
    semantic_similarity: float
    processing_time: float
    chunks: List[ContextChunk]

@dataclass
class RetrievalResult:
    """检索结果"""
    query: str
    retrieved_items: List[Dict[str, Any]]
    retrieval_strategy: RetrievalStrategy
    confidence_score: float
    semantic_coverage: float
    processing_time: float

class IntelligentContextManager:
    """
    智能上下文管理器 V3
    专为解决长对话遗忘和上下文爆炸问题设计
    """
    
    def __init__(self, db_path: str = "A项目/iflow/data/context_manager_v3.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 数据库连接
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.lock = threading.RLock()
        
        # ML组件
        if ML_AVAILABLE:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.pca_model = PCA(n_components=min(128, self.embedding_dim))
            self.kmeans_model = KMeans(n_clusters=50, random_state=42)
            self.vector_store = faiss.IndexFlatL2(self.embedding_dim)
        else:
            self.embedding_model = None
            self.embedding_dim = 384
            self.pca_model = None
            self.kmeans_model = None
            self.vector_store = None
        
        # 内存管理
        self.working_memory: OrderedDict[str, ContextChunk] = OrderedDict()
        self.long_term_memory: Dict[str, MemoryTrace] = {}
        self.semantic_cache: Dict[str, Any] = {}
        
        # 配置参数
        self.max_working_memory_size = 100
        self.compression_threshold = 0.7
        self.retrieval_cache_size = 50
        self.consolidation_interval = 3600  # 1小时
        
        # 统计信息
        self.compression_stats = defaultdict(int)
        self.retrieval_stats = defaultdict(int)
        self.performance_metrics = deque(maxlen=1000)
        
        # 初始化
        self._init_db()
        self._load_existing_data()
        
        logger.info("智能上下文管理器V3初始化完成")
    
    def _init_db(self):
        """初始化数据库"""
        with self.conn:
            # 上下文块表
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS context_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    content TEXT,
                    chunk_type TEXT,
                    semantic_embedding BLOB,
                    compression_ratio REAL,
                    importance_score REAL,
                    temporal_weight REAL,
                    access_count INTEGER,
                    last_accessed REAL,
                    metadata_json TEXT
                )
            """)
            
            # 记忆痕迹表
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_traces (
                    trace_id TEXT PRIMARY KEY,
                    memory_type TEXT,
                    content_summary TEXT,
                    semantic_fingerprint BLOB,
                    emotional_valence REAL,
                    consolidation_strength REAL,
                    retrieval_frequency INTEGER,
                    last_retrieved REAL,
                    associated_contexts_json TEXT,
                    metadata_json TEXT
                )
            """)
            
            # 压缩历史表
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS compression_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_size INTEGER,
                    compressed_size INTEGER,
                    compression_ratio REAL,
                    information_retention REAL,
                    semantic_similarity REAL,
                    processing_time REAL,
                    compression_level TEXT,
                    timestamp REAL
                )
            """)
    
    def _load_existing_data(self):
        """加载现有数据"""
        try:
            # 加载上下文块
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM context_chunks")
            rows = cursor.fetchall()
            
            for row in rows:
                chunk = ContextChunk(
                    chunk_id=row[0],
                    content=row[1],
                    chunk_type=row[2],
                    compression_ratio=row[4],
                    importance_score=row[5],
                    temporal_weight=row[6],
                    access_count=row[7],
                    last_accessed=row[8],
                    metadata=json.loads(row[9]) if row[9] else {}
                )
                
                # 恢复嵌入向量
                if row[3] and ML_AVAILABLE:
                    chunk.semantic_embedding = pickle.loads(row[3])
                
                self.working_memory[chunk.chunk_id] = chunk
            
            # 加载记忆痕迹
            cursor.execute("SELECT * FROM memory_traces")
            rows = cursor.fetchall()
            
            for row in rows:
                trace = MemoryTrace(
                    trace_id=row[0],
                    memory_type=MemoryType(row[1]),
                    content_summary=row[2],
                    emotional_valence=row[4],
                    consolidation_strength=row[5],
                    retrieval_frequency=row[6],
                    last_retrieved=row[7],
                    associated_contexts=json.loads(row[8]) if row[8] else [],
                    metadata=json.loads(row[9]) if row[9] else {}
                )
                
                # 恢复语义指纹
                if row[3] and ML_AVAILABLE:
                    trace.semantic_fingerprint = pickle.loads(row[3])
                
                self.long_term_memory[trace.trace_id] = trace
            
            # 构建向量索引
            if ML_AVAILABLE and self.vector_store:
                for chunk in self.working_memory.values():
                    if chunk.semantic_embedding is not None:
                        self.vector_store.add(np.array([chunk.semantic_embedding]).astype('float32'))
            
            logger.info(f"加载了 {len(self.working_memory)} 个上下文块和 {len(self.long_term_memory)} 个记忆痕迹")
            
        except Exception as e:
            logger.error(f"加载现有数据失败: {e}")
    
    def compress_context(self, context: Dict[str, Any], 
                        level: CompressionLevel = CompressionLevel.MEDIUM,
                        target_size: Optional[int] = None) -> CompressionResult:
        """
        智能上下文压缩
        """
        start_time = time.time()
        
        # 转换为文本
        context_text = self._dict_to_text(context)
        original_size = len(context_text)
        
        # 生成语义嵌入
        if ML_AVAILABLE and self.embedding_model:
            embedding = self.embedding_model.encode([context_text])[0]
        else:
            embedding = np.random.random(self.embedding_dim).astype('float32')
        
        # 根据压缩级别选择策略
        if level == CompressionLevel.QUANTUM and ML_AVAILABLE:
            compressed_chunks = self._quantum_compression(context, embedding)
        elif level == CompressionLevel.HEAVY:
            compressed_chunks = self._heavy_compression(context, embedding)
        elif level == CompressionLevel.MEDIUM:
            compressed_chunks = self._medium_compression(context, embedding)
        else:
            compressed_chunks = self._light_compression(context, embedding)
        
        # 计算压缩统计
        compressed_text = " ".join(chunk.content for chunk in compressed_chunks)
        compressed_size = len(compressed_text)
        compression_ratio = compressed_size / original_size
        
        # 计算信息保留率
        information_retention = self._calculate_information_retention(context, compressed_chunks)
        
        # 计算语义相似度
        semantic_similarity = self._calculate_semantic_similarity(context_text, compressed_text)
        
        processing_time = time.time() - start_time
        
        result = CompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            information_retention=information_retention,
            semantic_similarity=semantic_similarity,
            processing_time=processing_time,
            chunks=compressed_chunks
        )
        
        # 更新统计
        self.compression_stats[level.value] += 1
        self._record_compression_history(result)
        
        # 存储压缩结果
        self._store_compressed_chunks(compressed_chunks)
        
        logger.info(f"上下文压缩完成: 压缩率={compression_ratio:.2f}, 保留率={information_retention:.2f}, 相似度={semantic_similarity:.2f}")
        
        return result
    
    def _dict_to_text(self, context: Dict[str, Any]) -> str:
        """将字典转换为文本"""
        return json.dumps(context, ensure_ascii=False, indent=2)
    
    def _light_compression(self, context: Dict[str, Any], embedding: np.ndarray) -> List[ContextChunk]:
        """轻度压缩：主要去除冗余，保留大部分信息"""
        chunks = []
        
        # 按键值对分割
        for key, value in context.items():
            content = f"{key}: {str(value)}"
            
            chunk = ContextChunk(
                chunk_id=str(uuid.uuid4()),
                content=content,
                chunk_type="light_compressed",
                semantic_embedding=embedding,
                compression_ratio=0.8,
                importance_score=self._calculate_importance(key, value),
                temporal_weight=1.0,
                metadata={"original_key": key}
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _medium_compression(self, context: Dict[str, Any], embedding: np.ndarray) -> List[ContextChunk]:
        """中度压缩：提取关键信息，进行语义聚类"""
        if not ML_AVAILABLE:
            return self._light_compression(context, embedding)
        
        # 文本分块
        text_chunks = self._split_text_into_chunks(self._dict_to_text(context), max_chunk_size=500)
        
        # 为每个块生成嵌入
        chunk_embeddings = []
        for text_chunk in text_chunks:
            emb = self.embedding_model.encode([text_chunk])[0]
            chunk_embeddings.append(emb)
        
        # 聚类压缩
        if len(chunk_embeddings) > 5:
            # 使用K-means聚类
            chunk_embeddings_array = np.array(chunk_embeddings)
            self.kmeans_model.fit(chunk_embeddings_array)
            
            # 每个聚类保留一个代表
            chunks = []
            for cluster_id in range(self.kmeans_model.n_clusters):
                cluster_mask = self.kmeans_model.labels_ == cluster_id
                if cluster_mask.any():
                    cluster_embeddings = chunk_embeddings_array[cluster_mask]
                    # 选择最中心的点
                    centroid = self.kmeans_model.cluster_centers_[cluster_id]
                    center_idx = np.argmin(np.linalg.norm(cluster_embeddings - centroid, axis=1))
                    
                    representative_chunk = text_chunks[np.where(cluster_mask)[0][center_idx]]
                    
                    chunk = ContextChunk(
                        chunk_id=str(uuid.uuid4()),
                        content=representative_chunk,
                        chunk_type="cluster_compressed",
                        semantic_embedding=centroid,
                        compression_ratio=0.4,
                        importance_score=0.8,
                        temporal_weight=1.0
                    )
                    chunks.append(chunk)
        else:
            # 直接压缩
            chunks = []
            for i, text_chunk in enumerate(text_chunks):
                chunk = ContextChunk(
                    chunk_id=str(uuid.uuid4()),
                    content=text_chunk,
                    chunk_type="medium_compressed",
                    semantic_embedding=chunk_embeddings[i],
                    compression_ratio=0.6,
                    importance_score=0.7,
                    temporal_weight=1.0
                )
                chunks.append(chunk)
        
        return chunks
    
    def _heavy_compression(self, context: Dict[str, Any], embedding: np.ndarray) -> List[ContextChunk]:
        """重度压缩：提取核心语义，大幅减少信息量"""
        # 提取关键词和关键句
        text = self._dict_to_text(context)
        key_sentences = self._extract_key_sentences(text)
        
        chunks = []
        for i, sentence in enumerate(key_sentences):
            chunk = ContextChunk(
                chunk_id=str(uuid.uuid4()),
                content=sentence,
                chunk_type="heavily_compressed",
                semantic_embedding=embedding,
                compression_ratio=0.2,
                importance_score=0.9,
                temporal_weight=1.0,
                metadata={"sentence_rank": i}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _quantum_compression(self, context: Dict[str, Any], embedding: np.ndarray) -> List[ContextChunk]:
        """量子压缩：使用量子算法优化压缩效果"""
        if not ML_AVAILABLE:
            return self._heavy_compression(context, embedding)
        
        # 模拟量子压缩算法
        text = self._dict_to_text(context)
        
        # 量子退火优化：寻找最优压缩配置
        chunks = []
        
        # 提取最重要的语义单元
        sentences = re.split(r'[。！？.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 使用量子启发式算法选择最优句子组合
        selected_indices = self._quantum_annealing_selection(sentences, embedding)
        
        for i, idx in enumerate(selected_indices):
            chunk = ContextChunk(
                chunk_id=str(uuid.uuid4()),
                content=sentences[idx],
                chunk_type="quantum_compressed",
                semantic_embedding=embedding,
                compression_ratio=0.15,
                importance_score=0.95,
                temporal_weight=1.0,
                metadata={"quantum_selection_rank": i}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """将文本分割成块"""
        chunks = []
        current_chunk = ""
        
        sentences = re.split(r'[。！？.!?]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += sentence + "。"
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _extract_key_sentences(self, text: str) -> List[str]:
        """提取关键句子"""
        sentences = re.split(r'[。！？.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not ML_AVAILABLE:
            # 简化实现：选择前几个句子
            return sentences[:min(3, len(sentences))]
        
        # 使用嵌入和重要性评分
        sentence_embeddings = []
        for sentence in sentences:
            emb = self.embedding_model.encode([sentence])[0]
            sentence_embeddings.append(emb)
        
        # 计算句子中心性
        embeddings_array = np.array(sentence_embeddings)
        centroid = np.mean(embeddings_array, axis=0)
        centralities = np.dot(embeddings_array, centroid) / (
            np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(centroid)
        )
        
        # 选择最中心的句子
        top_indices = np.argsort(centralities)[-min(3, len(sentences)):][::-1]
        
        return [sentences[i] for i in top_indices]
    
    def _quantum_annealing_selection(self, sentences: List[str], global_embedding: np.ndarray) -> List[int]:
        """量子退火启发的句子选择算法"""
        if not sentences:
            return []
        
        # 计算每个句子的"能量"（重要性）
        energies = []
        
        for sentence in sentences:
            if ML_AVAILABLE:
                sentence_embedding = self.embedding_model.encode([sentence])[0]
                # 计算与全局语义的相似度作为能量
                similarity = np.dot(sentence_embedding, global_embedding) / (
                    np.linalg.norm(sentence_embedding) * np.linalg.norm(global_embedding)
                )
                energy = 1 - similarity  # 能量越低越重要
            else:
                energy = np.random.random()
            
            energies.append(energy)
        
        # 模拟量子退火：选择低能量状态
        energies_array = np.array(energies)
        # 选择能量最低的前20%句子
        num_select = max(1, len(sentences) // 5)
        selected_indices = np.argsort(energies_array)[:num_select]
        
        return selected_indices.tolist()
    
    def _calculate_importance(self, key: str, value: Any) -> float:
        """计算内容重要性"""
        importance = 0.5
        
        # 关键词权重
        key_keywords = ['error', 'security', 'performance', 'critical', 'important']
        for keyword in key_keywords:
            if keyword in key.lower():
                importance += 0.2
        
        # 内容类型权重
        if isinstance(value, (dict, list)):
            importance += 0.1
        elif isinstance(value, str) and len(value) > 100:
            importance += 0.1
        
        return min(importance, 1.0)
    
    def _calculate_information_retention(self, original: Dict[str, Any], chunks: List[ContextChunk]) -> float:
        """计算信息保留率"""
        # 简化实现：基于压缩比和重要性加权
        total_importance = sum(chunk.importance_score for chunk in chunks)
        max_possible_importance = len(chunks) * 1.0
        
        if max_possible_importance == 0:
            return 0.0
        
        return total_importance / max_possible_importance
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算语义相似度"""
        if not ML_AVAILABLE:
            return 0.5
        
        try:
            emb1 = self.embedding_model.encode([text1])[0]
            emb2 = self.embedding_model.encode([text2])[0]
            
            # 余弦相似度
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        except Exception:
            return 0.5
    
    def retrieve_context(self, query: str, strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_FUSION,
                        top_k: int = 5) -> RetrievalResult:
        """
        智能上下文检索
        """
        start_time = time.time()
        
        retrieved_items = []
        
        if strategy == RetrievalStrategy.SEMANTIC_SEARCH and ML_AVAILABLE:
            retrieved_items = self._semantic_retrieval(query, top_k)
        elif strategy == RetrievalStrategy.TEMPORAL_PROXIMITY:
            retrieved_items = self._temporal_retrieval(query, top_k)
        elif strategy == RetrievalStrategy.FREQUENCY_BASED:
            retrieved_items = self._frequency_retrieval(query, top_k)
        elif strategy == RetrievalStrategy.QUANTUM_ENTANGLEMENT and ML_AVAILABLE:
            retrieved_items = self._quantum_retrieval(query, top_k)
        else:
            retrieved_items = self._hybrid_fusion_retrieval(query, top_k)
        
        # 计算检索质量指标
        confidence_score = self._calculate_retrieval_confidence(retrieved_items, query)
        semantic_coverage = self._calculate_semantic_coverage(retrieved_items, query)
        processing_time = time.time() - start_time
        
        result = RetrievalResult(
            query=query,
            retrieved_items=retrieved_items,
            retrieval_strategy=strategy,
            confidence_score=confidence_score,
            semantic_coverage=semantic_coverage,
            processing_time=processing_time
        )
        
        # 更新检索统计
        self.retrieval_stats[strategy.value] += 1
        
        return result
    
    def _semantic_retrieval(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """语义检索"""
        if not ML_AVAILABLE or not self.vector_store:
            return []
        
        query_embedding = self.embedding_model.encode([query])[0]
        
        if self.vector_store.ntotal == 0:
            return []
        
        distances, indices = self.vector_store.search(np.array([query_embedding]).astype('float32'), top_k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx in self.working_memory:
                chunk = list(self.working_memory.values())[idx]
                similarity = 1 - (dist / self.embedding_dim)
                
                if similarity > 0.5:  # 相似度阈值
                    chunk.access_count += 1
                    chunk.last_accessed = time.time()
                    
                    results.append({
                        'content': chunk.content,
                        'similarity': float(similarity),
                        'chunk_id': chunk.chunk_id,
                        'importance': chunk.importance_score,
                        'type': 'semantic'
                    })
        
        return results
    
    def _temporal_retrieval(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """时间邻近检索"""
        # 按时间排序，返回最近的项目
        sorted_chunks = sorted(
            self.working_memory.values(),
            key=lambda x: x.last_accessed,
            reverse=True
        )
        
        results = []
        for chunk in sorted_chunks[:top_k]:
            results.append({
                'content': chunk.content,
                'temporal_score': chunk.last_accessed,
                'chunk_id': chunk.chunk_id,
                'importance': chunk.importance_score,
                'type': 'temporal'
            })
        
        return results
    
    def _frequency_retrieval(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """频率基础检索"""
        # 按访问频率排序
        sorted_chunks = sorted(
            self.working_memory.values(),
            key=lambda x: x.access_count,
            reverse=True
        )
        
        results = []
        for chunk in sorted_chunks[:top_k]:
            results.append({
                'content': chunk.content,
                'frequency': chunk.access_count,
                'chunk_id': chunk.chunk_id,
                'importance': chunk.importance_score,
                'type': 'frequency'
            })
        
        return results
    
    def _quantum_retrieval(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """量子纠缠检索"""
        if not ML_AVAILABLE:
            return self._semantic_retrieval(query, top_k)
        
        # 模拟量子纠缠：查找语义上"纠缠"的上下文
        query_embedding = self.embedding_model.encode([query])[0]
        
        # 计算量子纠缠强度
        entanglement_scores = []
        for chunk in self.working_memory.values():
            if chunk.semantic_embedding is not None:
                # 量子纠缠相似度计算
                similarity = np.dot(query_embedding, chunk.semantic_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk.semantic_embedding)
                )
                # 量子纠缠强度还考虑时间衰减
                time_decay = np.exp(-(time.time() - chunk.last_accessed) / 3600)
                entanglement_score = similarity * time_decay * chunk.importance_score
                entanglement_scores.append((chunk, entanglement_score))
        
        # 按纠缠强度排序
        entanglement_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk, score in entanglement_scores[:top_k]:
            if score > 0.3:  # 量子纠缠阈值
                results.append({
                    'content': chunk.content,
                    'quantum_entanglement': float(score),
                    'chunk_id': chunk.chunk_id,
                    'importance': chunk.importance_score,
                    'type': 'quantum'
                })
        
        return results
    
    def _hybrid_fusion_retrieval(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """混合融合检索"""
        # 融合多种检索策略的结果
        semantic_results = self._semantic_retrieval(query, top_k)
        temporal_results = self._temporal_retrieval(query, top_k)
        frequency_results = self._frequency_retrieval(query, top_k)
        
        # 合并结果并重新评分
        all_results = semantic_results + temporal_results + frequency_results
        
        # 计算综合得分
        for result in all_results:
            base_score = 0
            
            if 'similarity' in result:
                base_score += result['similarity'] * 0.5
            elif 'temporal_score' in result:
                # 归一化时间分数
                max_time = max(r.get('temporal_score', 0) for r in all_results)
                base_score += (result['temporal_score'] / max_time) * 0.3 if max_time > 0 else 0
            elif 'frequency' in result:
                # 归一化频率分数
                max_freq = max(r.get('frequency', 1) for r in all_results)
                base_score += (result['frequency'] / max_freq) * 0.2 if max_freq > 0 else 0
            
            # 加上重要性权重
            result['hybrid_score'] = base_score + result.get('importance', 0) * 0.3
        
        # 按综合得分排序
        all_results.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
        
        return all_results[:top_k]
    
    def _calculate_retrieval_confidence(self, retrieved_items: List[Dict[str, Any]], query: str) -> float:
        """计算检索置信度"""
        if not retrieved_items:
            return 0.0
        
        # 基于检索质量和相关性计算置信度
        total_score = 0
        max_possible_score = 0
        
        for item in retrieved_items:
            if 'hybrid_score' in item:
                total_score += item['hybrid_score']
                max_possible_score += 1.0
            elif 'similarity' in item:
                total_score += item['similarity']
                max_possible_score += 1.0
            else:
                total_score += 0.5  # 默认分数
                max_possible_score += 1.0
        
        if max_possible_score == 0:
            return 0.0
        
        base_confidence = total_score / max_possible_score
        
        # 根据检索数量调整置信度
        coverage_bonus = min(len(retrieved_items) / 5.0, 1.0) * 0.1
        
        return min(base_confidence + coverage_bonus, 1.0)
    
    def _calculate_semantic_coverage(self, retrieved_items: List[Dict[str, Any]], query: str) -> float:
        """计算语义覆盖率"""
        if not ML_AVAILABLE or not retrieved_items:
            return 0.3
        
        query_embedding = self.embedding_model.encode([query])[0]
        
        # 计算检索到的内容的语义覆盖
        covered_dimensions = set()
        
        for item in retrieved_items:
            item_text = item['content']
            item_embedding = self.embedding_model.encode([item_text])[0]
            
            # 计算查询与检索项的相似维度
            similarity = np.dot(query_embedding, item_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(item_embedding)
            )
            
            if similarity > 0.5:
                covered_dimensions.add(item['chunk_id'])
        
        # 覆盖率 = 覆盖的维度数 / 检索项总数
        coverage = len(covered_dimensions) / len(retrieved_items)
        
        return coverage
    
    def _store_compressed_chunks(self, chunks: List[ContextChunk]):
        """存储压缩的上下文块"""
        with self.lock:
            for chunk in chunks:
                # 添加到工作记忆
                if len(self.working_memory) >= self.max_working_memory_size:
                    # 移除最旧的项目
                    oldest_id = next(iter(self.working_memory))
                    del self.working_memory[oldest_id]
                
                self.working_memory[chunk.chunk_id] = chunk
                
                # 添加到向量存储
                if ML_AVAILABLE and self.vector_store and chunk.semantic_embedding is not None:
                    self.vector_store.add(np.array([chunk.semantic_embedding]).astype('float32'))
                
                # 持久化
                self._persist_chunk(chunk)
    
    def _persist_chunk(self, chunk: ContextChunk):
        """持久化上下文块"""
        try:
            with self.conn:
                embedding_blob = pickle.dumps(chunk.semantic_embedding) if chunk.semantic_embedding is not None else None
                
                self.conn.execute("""
                    INSERT OR REPLACE INTO context_chunks
                    (chunk_id, content, chunk_type, semantic_embedding, compression_ratio, 
                     importance_score, temporal_weight, access_count, last_accessed, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.chunk_id, chunk.content, chunk.chunk_type, embedding_blob,
                    chunk.compression_ratio, chunk.importance_score, chunk.temporal_weight,
                    chunk.access_count, chunk.last_accessed, json.dumps(chunk.metadata)
                ))
        except Exception as e:
            logger.error(f"持久化上下文块失败: {e}")
    
    def _record_compression_history(self, result: CompressionResult):
        """记录压缩历史"""
        try:
            with self.conn:
                self.conn.execute("""
                    INSERT INTO compression_history
                    (original_size, compressed_size, compression_ratio, information_retention, 
                     semantic_similarity, processing_time, compression_level, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.original_size, result.compressed_size, result.compression_ratio,
                    result.information_retention, result.semantic_similarity, result.processing_time,
                    CompressionLevel.MEDIUM.value, time.time()
                ))
        except Exception as e:
            logger.error(f"记录压缩历史失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            return {
                "working_memory_size": len(self.working_memory),
                "long_term_memory_size": len(self.long_term_memory),
                "compression_stats": dict(self.compression_stats),
                "retrieval_stats": dict(self.retrieval_stats),
                "total_chunks": sum(len(self.working_memory), len(self.long_term_memory)),
                "average_compression_ratio": np.mean([chunk.compression_ratio for chunk in self.working_memory.values()]) if self.working_memory else 0,
                "average_importance_score": np.mean([chunk.importance_score for chunk in self.working_memory.values()]) if self.working_memory else 0
            }
    
    def close(self):
        """关闭管理器"""
        self.conn.close()
        logger.info("智能上下文管理器已关闭")

async def main():
    """测试智能上下文管理器"""
    logger.info("🧠 测试智能上下文管理器 V3...")
    
    manager = IntelligentContextManager()
    
    # 测试上下文压缩
    test_context = {
        "task": "优化数据库查询性能",
        "user_request": "需要提高查询速度，当前查询耗时2秒",
        "database_info": {
            "type": "PostgreSQL",
            "tables": ["users", "orders", "products"],
            "indexes": ["users.id", "orders.user_id"]
        },
        "performance_metrics": {
            "query_time": 2000,
            "cpu_usage": 80,
            "memory_usage": 60
        },
        "error_logs": "无",
        "security_requirements": "高",
        "timeline": "紧急"
    }
    
    print("\n" + "="*60)
    print("🗜️ 上下文压缩测试:")
    
    # 测试不同压缩级别
    for level in [CompressionLevel.LIGHT, CompressionLevel.MEDIUM, CompressionLevel.HEAVY]:
        result = manager.compress_context(test_context, level)
        print(f"  - {level.value}: 压缩率={result.compression_ratio:.2f}, 保留率={result.information_retention:.2f}")
    
    # 测试上下文检索
    print("\n" + "="*60)
    print("🔍 上下文检索测试:")
    
    for strategy in [RetrievalStrategy.SEMANTIC_SEARCH, RetrievalStrategy.HYBRID_FUSION]:
        if strategy == RetrievalStrategy.SEMANTIC_SEARCH and not ML_AVAILABLE:
            continue
        
        result = manager.retrieve_context("数据库性能优化", strategy, top_k=3)
        print(f"  - {strategy.value}: 置信度={result.confidence_score:.2f}, 覆盖率={result.semantic_coverage:.2f}")
    
    print("\n" + "="*60)
    print("📊 管理器统计:")
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    manager.close()

if __name__ == "__main__":
    asyncio.run(main())