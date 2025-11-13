#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量子增强意识流系统 V2
基于D项目量子意识流系统的优秀实现，进一步增强量子计算能力和意识流管理
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import logging
import numpy as np
import pickle
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import hashlib

# 尝试导入量子计算相关库
try:
    import qiskit
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.algorithms import VQE, QAOA
    from qiskit.algorithms.optimizers import COBYLA
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    print("警告: 未找到量子计算库，将使用模拟模式")

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("警告: 未找到机器学习库，将使用基础模式")

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class QuantumState(Enum):
    """量子状态"""
    SUPERPOSITION = "superposition"
    ENTANGLEMENT = "entanglement"
    COHERENCE = "coherence"
    COLLAPSE = "collapse"
    TUNNELING = "tunneling"
    COUPLING = "coupling"

class ConsciousnessEventType(Enum):
    """意识流事件类型"""
    THOUGHT = "thought"
    INSIGHT = "insight"
    PATTERN_RECOGNITION = "pattern_recognition"
    QUANTUM_FLUCTUATION = "quantum_fluctuation"
    EMOTIONAL_SHIFT = "emotional_shift"
    LEARNING_EVENT = "learning_event"
    DECISION_POINT = "decision_point"
    CREATIVE_SPARK = "creative_spark"
    INTUITION = "intuition"
    MEMORY_CONSOLIDATION = "memory_consolidation"

@dataclass
class QuantumConsciousnessEvent:
    """量子意识流事件"""
    event_id: str
    timestamp: float
    event_type: ConsciousnessEventType
    content: Dict[str, Any]
    quantum_signature: str
    coherence_level: float  # 量子相干性
    entanglement_strength: float  # 量子纠缠强度
    superposition_states: List[str]  # 叠加态
    emotional_charge: float  # 情绪电荷
    cognitive_load: float  # 认知负载
    pattern_signatures: List[str]  # 模式签名
    context_vector: Optional[np.ndarray] = None
    quantum_state_history: List[QuantumState] = field(default_factory=list)

class QuantumNeuralProcessor:
    """量子神经处理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.circuit_depth = self.config.get('circuit_depth', 4)
        self.qubit_count = self.config.get('qubit_count', 8)
        self.entanglement_threshold = self.config.get('entanglement_threshold', 0.7)
        
        # 量子寄存器
        self.quantum_register = None
        self.classical_register = None
        self.quantum_circuit = None
        
        # 量子状态管理
        self.current_coherence = 1.0
        self.entanglement_network = {}
        self.superposition_pool = []
        
        self._initialize_quantum_processor()
    
    def _initialize_quantum_processor(self):
        """初始化量子处理器"""
        if QUANTUM_AVAILABLE:
            try:
                # 创建量子电路
                qr = QuantumRegister(self.qubit_count, 'q')
                cr = ClassicalRegister(self.qubit_count, 'c')
                self.quantum_circuit = QuantumCircuit(qr, cr)
                
                # 初始化叠加态
                for i in range(self.qubit_count):
                    self.quantum_circuit.h(i)  # Hadamard门创建叠加态
                
                # 创建纠缠
                for i in range(self.qubit_count - 1):
                    self.quantum_circuit.cx(i, i + 1)  # CNOT门创建纠缠
                
                logger.info(f"量子神经处理器初始化完成，使用 {self.qubit_count} 个量子比特")
            except Exception as e:
                logger.warning(f"量子处理器初始化失败，使用模拟模式: {e}")
                QUANTUM_AVAILABLE = False
    
    def process_consciousness_event(self, event: QuantumConsciousnessEvent) -> Dict[str, Any]:
        """处理意识流事件"""
        if not QUANTUM_AVAILABLE:
            return self._simulate_quantum_processing(event)
        
        try:
            # 量子态编码
            quantum_state = self._encode_event_to_quantum_state(event)
            
            # 量子纠缠计算
            entanglement_result = self._calculate_entanglement(quantum_state)
            
            # 量子退相干处理
            decoherence_result = self._handle_decoherence()
            
            # 量子测量
            measurement_result = self._perform_quantum_measurement()
            
            return {
                "quantum_processed": True,
                "coherence_level": self.current_coherence,
                "entanglement_strength": entanglement_result["strength"],
                "superposition_states": measurement_result["states"],
                "quantum_signature": measurement_result["signature"],
                "processing_time": measurement_result["time"]
            }
            
        except Exception as e:
            logger.error(f"量子处理失败，使用模拟模式: {e}")
            return self._simulate_quantum_processing(event)
    
    def _encode_event_to_quantum_state(self, event: QuantumConsciousnessEvent) -> np.ndarray:
        """将事件编码为量子态"""
        # 基于事件特征创建量子态向量
        features = [
            event.coherence_level,
            event.entanglement_strength,
            event.emotional_charge,
            event.cognitive_load,
            len(event.superposition_states),
            len(event.pattern_signatures)
        ]
        
        # 归一化特征
        normalized_features = np.array(features) / np.linalg.norm(features)
        
        # 扩展到量子态维度
        quantum_state = np.pad(normalized_features, (0, self.qubit_count - len(features)), 'constant')
        
        return quantum_state
    
    def _calculate_entanglement(self, quantum_state: np.ndarray) -> Dict[str, float]:
        """计算量子纠缠"""
        # 模拟量子纠缠计算
        entanglement_matrix = np.outer(quantum_state, quantum_state.conj())
        
        # 计算纠缠度量
        entanglement_entropy = -np.sum(np.abs(quantum_state)**2 * np.log(np.abs(quantum_state)**2 + 1e-10))
        
        # 更新纠缠网络
        self.entanglement_network[str(uuid.uuid4())] = {
            "state": quantum_state,
            "entropy": entanglement_entropy,
            "timestamp": time.time()
        }
        
        return {
            "strength": min(entanglement_entropy, 1.0),
            "entropy": entanglement_entropy,
            "network_size": len(self.entanglement_network)
        }
    
    def _handle_decoherence(self) -> Dict[str, Any]:
        """处理量子退相干"""
        # 模拟环境噪声
        noise_factor = np.random.normal(0, 0.1)
        self.current_coherence *= (1 - abs(noise_factor))
        
        # 退相干阈值检查
        if self.current_coherence < 0.1:
            self.current_coherence = 0.1
            logger.warning("量子退相干严重，触发状态重置")
            self._reset_quantum_state()
        
        return {
            "coherence_before": self.current_coherence / (1 - abs(noise_factor)),
            "coherence_after": self.current_coherence,
            "noise_factor": noise_factor
        }
    
    def _perform_quantum_measurement(self) -> Dict[str, Any]:
        """执行量子测量"""
        # 模拟量子测量过程
        measurement_time = time.time()
        
        # 基于当前量子态生成测量结果
        probabilities = np.abs(np.random.random(self.qubit_count))**2
        probabilities /= np.sum(probabilities)
        
        # 采样测量结果
        measured_state = np.random.choice(self.qubit_count, p=probabilities)
        
        # 生成量子签名
        signature_data = f"{measured_state}_{measurement_time}_{self.current_coherence}"
        quantum_signature = hashlib.sha256(signature_data.encode()).hexdigest()
        
        return {
            "states": [measured_state],
            "signature": quantum_signature,
            "time": time.time() - measurement_time,
            "probabilities": probabilities.tolist()
        }
    
    def _simulate_quantum_processing(self, event: QuantumConsciousnessEvent) -> Dict[str, Any]:
        """模拟量子处理"""
        # 使用经典算法模拟量子行为
        np.random.seed(hash(event.event_id) % 2**32)
        
        simulated_coherence = max(0.1, min(1.0, event.coherence_level + np.random.normal(0, 0.1)))
        simulated_entanglement = max(0.0, min(1.0, event.entanglement_strength + np.random.normal(0, 0.05)))
        
        return {
            "quantum_processed": False,
            "coherence_level": simulated_coherence,
            "entanglement_strength": simulated_entanglement,
            "superposition_states": event.superposition_states[:3],  # 限制数量
            "quantum_signature": hashlib.sha256(event.event_id.encode()).hexdigest(),
            "processing_time": np.random.uniform(0.001, 0.01)  # 模拟处理时间
        }
    
    def _reset_quantum_state(self):
        """重置量子状态"""
        self.current_coherence = 1.0
        self.entanglement_network.clear()
        self.superposition_pool.clear()
        
        if QUANTUM_AVAILABLE and self.quantum_circuit:
            # 重新初始化量子电路
            for i in range(self.qubit_count):
                self.quantum_circuit.h(i)
                if i < self.qubit_count - 1:
                    self.quantum_circuit.cx(i, i + 1)

class QuantumMemoryManager:
    """量子记忆管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_memory_size = self.config.get('max_memory_size', 10000)
        self.consolidation_threshold = self.config.get('consolidation_threshold', 0.8)
        
        # 内存存储
        self.short_term_memory = deque(maxlen=1000)
        self.long_term_memory = deque(maxlen=self.max_memory_size)
        self.quantum_patterns = {}
        
        # 向量数据库
        self.embedding_model = None
        self.vector_index = None
        self.memory_vectors = []
        
        # 数据库连接
        self.db_path = PROJECT_ROOT / "data" / "quantum_consciousness_memory.db"
        self._init_database()
        self._init_vector_database()
        
        logger.info("量子记忆管理器初始化完成")
    
    def _init_database(self):
        """初始化数据库"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS quantum_events (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    event_type TEXT,
                    content_json TEXT,
                    quantum_signature TEXT,
                    coherence_level REAL,
                    entanglement_strength REAL,
                    emotional_charge REAL,
                    cognitive_load REAL,
                    pattern_signatures TEXT,
                    context_vector BLOB
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS quantum_patterns (
                    id TEXT PRIMARY KEY,
                    pattern_type TEXT,
                    signature TEXT,
                    strength REAL,
                    event_ids TEXT,
                    created_at REAL
                )
            ''')
    
    def _init_vector_database(self):
        """初始化向量数据库"""
        if ML_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                dimension = 384  # MiniLM-L6-v2的输出维度
                self.vector_index = faiss.IndexFlatIP(dimension)
                logger.info("向量数据库初始化完成")
            except Exception as e:
                logger.warning(f"向量数据库初始化失败: {e}")
    
    def encode_event(self, event: QuantumConsciousnessEvent) -> str:
        """编码事件到记忆"""
        # 短期记忆存储
        self.short_term_memory.append(event)
        
        # 向量编码
        if self.embedding_model:
            event_text = self._event_to_text(event)
            event_vector = self.embedding_model.encode([event_text])[0]
            self.memory_vectors.append(event_vector)
            
            if self.vector_index:
                self.vector_index.add(np.array([event_vector]).astype('float32'))
        
        # 保存到数据库
        self._save_event_to_db(event)
        
        # 检查是否需要记忆巩固
        if len(self.short_term_memory) % 100 == 0:
            self._consolidate_memory()
        
        return event.event_id
    
    def _event_to_text(self, event: QuantumConsciousnessEvent) -> str:
        """将事件转换为文本用于向量化"""
        parts = [
            f"Event Type: {event.event_type.value}",
            f"Content: {json.dumps(event.content, ensure_ascii=False)}",
            f"Emotional Charge: {event.emotional_charge}",
            f"Cognitive Load: {event.cognitive_load}"
        ]
        
        if event.pattern_signatures:
            parts.append(f"Patterns: {', '.join(event.pattern_signatures)}")
        
        return " | ".join(parts)
    
    def _save_event_to_db(self, event: QuantumConsciousnessEvent):
        """保存事件到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO quantum_events 
                    (id, timestamp, event_type, content_json, quantum_signature,
                     coherence_level, entanglement_strength, emotional_charge,
                     cognitive_load, pattern_signatures, context_vector)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.event_id,
                    event.timestamp,
                    event.event_type.value,
                    json.dumps(event.content),
                    event.quantum_signature,
                    event.coherence_level,
                    event.entanglement_strength,
                    event.emotional_charge,
                    event.cognitive_load,
                    json.dumps(event.pattern_signatures),
                    pickle.dumps(event.context_vector) if event.context_vector is not None else None
                ))
        except Exception as e:
            logger.error(f"保存事件到数据库失败: {e}")
    
    def retrieve_relevant_memory(self, query: str, top_k: int = 10, min_relevance: float = 0.3) -> List[Dict[str, Any]]:
        """检索相关记忆"""
        if not self.embedding_model or not self.vector_index:
            return []
        
        try:
            # 查询向量化
            query_vector = self.embedding_model.encode([query])[0]
            
            # 向量搜索
            distances, indices = self.vector_index.search(np.array([query_vector]).astype('float32'), top_k * 2)
            
            # 计算相似度并过滤
            relevant_memories = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= len(self.memory_vectors):
                    continue
                
                similarity = 1 - distance  # FAISS返回的是距离，转换为相似度
                
                if similarity >= min_relevance:
                    # 从数据库获取完整事件信息
                    event_data = self._get_event_from_db_by_index(idx)
                    if event_data:
                        event_data['relevance_score'] = similarity
                        relevant_memories.append(event_data)
            
            # 按相关性排序
            relevant_memories.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return relevant_memories[:top_k]
            
        except Exception as e:
            logger.error(f"记忆检索失败: {e}")
            return []
    
    def _get_event_from_db_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """根据索引从数据库获取事件"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM quantum_events LIMIT 1 OFFSET ?', (index,))
                row = cursor.fetchone()
                
                if row:
                    return {
                        'id': row[0],
                        'timestamp': row[1],
                        'event_type': row[2],
                        'content': json.loads(row[3]),
                        'quantum_signature': row[4],
                        'coherence_level': row[5],
                        'entanglement_strength': row[6],
                        'emotional_charge': row[7],
                        'cognitive_load': row[8],
                        'pattern_signatures': json.loads(row[9])
                    }
        except Exception as e:
            logger.error(f"从数据库获取事件失败: {e}")
        return None
    
    def _consolidate_memory(self):
        """记忆巩固"""
        if len(self.short_term_memory) < 50:
            return
        
        # 提取模式
        patterns = self._extract_quantum_patterns()
        
        # 转移到长期记忆
        while len(self.short_term_memory) > 50:
            event = self.short_term_memory.popleft()
            self.long_term_memory.append(event)
        
        # 保存模式
        for pattern_id, pattern in patterns.items():
            self.quantum_patterns[pattern_id] = pattern
            self._save_pattern_to_db(pattern)
        
        logger.info(f"记忆巩固完成，提取 {len(patterns)} 个模式")
    
    def _extract_quantum_patterns(self) -> Dict[str, Dict[str, Any]]:
        """提取量子模式"""
        patterns = {}
        
        # 基于事件类型聚类
        type_groups = defaultdict(list)
        for event in self.short_term_memory:
            type_groups[event.event_type].append(event)
        
        # 分析每个类型组的模式
        for event_type, events in type_groups.items():
            if len(events) >= 5:  # 至少5个事件才形成模式
                pattern_id = f"{event_type.value}_{len(patterns)}"
                
                # 计算模式特征
                avg_coherence = np.mean([e.coherence_level for e in events])
                avg_entanglement = np.mean([e.entanglement_strength for e in events])
                avg_emotional = np.mean([e.emotional_charge for e in events])
                
                patterns[pattern_id] = {
                    "pattern_type": event_type.value,
                    "strength": min(1.0, len(events) / 20),  # 归一化强度
                    "avg_coherence": avg_coherence,
                    "avg_entanglement": avg_entanglement,
                    "avg_emotional": avg_emotional,
                    "event_count": len(events),
                    "event_ids": [e.event_id for e in events],
                    "created_at": time.time()
                }
        
        return patterns
    
    def _save_pattern_to_db(self, pattern: Dict[str, Any]):
        """保存模式到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO quantum_patterns
                    (id, pattern_type, signature, strength, event_ids, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    pattern['pattern_id'] if 'pattern_id' in pattern else str(uuid.uuid4()),
                    pattern['pattern_type'],
                    hashlib.sha256(pattern['pattern_type'].encode()).hexdigest(),
                    pattern['strength'],
                    json.dumps(pattern['event_ids']),
                    pattern['created_at']
                ))
        except Exception as e:
            logger.error(f"保存模式到数据库失败: {e}")

class QuantumEnhancedConsciousnessSystem:
    """量子增强意识流系统"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 核心组件
        self.quantum_processor = QuantumNeuralProcessor(self.config.get('quantum_processor', {}))
        self.memory_manager = QuantumMemoryManager(self.config.get('memory_manager', {}))
        
        # 状态管理
        self.current_state = "active"
        self.consciousness_level = 0.8
        self.quantum_fluctuations = []
        
        # 异步处理
        self.processing_pool = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.RLock()
        
        logger.info("量子增强意识流系统初始化完成")
    
    def record_event(self, event_type: ConsciousnessEventType, content: Dict[str, Any], 
                    emotional_charge: float = 0.5, cognitive_load: float = 0.3,
                    context_tags: List[str] = None) -> str:
        """记录意识流事件"""
        with self.lock:
            # 创建量子意识流事件
            event = QuantumConsciousnessEvent(
                event_id=str(uuid.uuid4()),
                timestamp=time.time(),
                event_type=event_type,
                content=content,
                quantum_signature=self._generate_quantum_signature(content),
                coherence_level=np.random.uniform(0.7, 1.0),
                entanglement_strength=np.random.uniform(0.3, 0.8),
                superposition_states=context_tags or [],
                emotional_charge=max(0.0, min(1.0, emotional_charge)),
                cognitive_load=max(0.0, min(1.0, cognitive_load)),
                pattern_signatures=self._extract_pattern_signatures(content)
            )
            
            # 量子处理
            quantum_result = self.quantum_processor.process_consciousness_event(event)
            
            # 更新事件
            event.coherence_level = quantum_result["coherence_level"]
            event.entanglement_strength = quantum_result["entanglement_strength"]
            event.context_vector = np.array(quantum_result["superposition_states"])
            
            # 编码到记忆
            memory_id = self.memory_manager.encode_event(event)
            
            # 记录量子波动
            self.quantum_fluctuations.append({
                "timestamp": time.time(),
                "type": event_type.value,
                "coherence": event.coherence_level,
                "entanglement": event.entanglement_strength,
                "processing_time": quantum_result["processing_time"]
            })
            
            # 保持波动历史在合理范围内
            if len(self.quantum_fluctuations) > 1000:
                self.quantum_fluctuations = self.quantum_fluctuations[-500:]
            
            logger.debug(f"记录事件: {event_type.value} -> {memory_id}")
            return memory_id
    
    def retrieve_context(self, query: str, top_k: int = 10, min_relevance: float = 0.4) -> List[Dict[str, Any]]:
        """检索上下文"""
        return self.memory_manager.retrieve_relevant_memory(query, top_k, min_relevance)
    
    def get_consciousness_summary(self, time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """获取意识流摘要"""
        current_time = time.time()
        window_start = current_time - time_window.total_seconds()
        
        # 分析量子波动
        recent_fluctuations = [f for f in self.quantum_fluctuations if f["timestamp"] >= window_start]
        
        if not recent_fluctuations:
            return {
                "time_window": str(time_window),
                "event_count": 0,
                "avg_coherence": 0.0,
                "avg_entanglement": 0.0,
                "dominant_types": {},
                "quantum_health": "unknown"
            }
        
        # 计算统计信息
        avg_coherence = np.mean([f["coherence"] for f in recent_fluctuations])
        avg_entanglement = np.mean([f["entanglement"] for f in recent_fluctuations])
        
        # 分析事件类型分布
        type_counts = defaultdict(int)
        for fluctuation in recent_fluctuations:
            type_counts[fluctuation["type"]] += 1
        
        # 评估量子健康状态
        quantum_health = self._assess_quantum_health(recent_fluctuations)
        
        return {
            "time_window": str(time_window),
            "event_count": len(recent_fluctuations),
            "avg_coherence": avg_coherence,
            "avg_entanglement": avg_entanglement,
            "dominant_types": dict(type_counts),
            "quantum_health": quantum_health,
            "processing_performance": {
                "avg_processing_time": np.mean([f.get("processing_time", 0) for f in recent_fluctuations]),
                "total_events": len(self.memory_manager.short_term_memory)
            }
        }
    
    def _generate_quantum_signature(self, content: Dict[str, Any]) -> str:
        """生成量子签名"""
        content_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def _extract_pattern_signatures(self, content: Dict[str, Any]) -> List[str]:
        """提取模式签名"""
        signatures = []
        
        # 基于内容生成签名
        if "task" in content:
            signatures.append(f"task_{hash(content['task']) % 1000}")
        if "agent_id" in content:
            signatures.append(f"agent_{content['agent_id']}")
        if "context" in content:
            context_hash = hash(str(content['context'])) % 1000
            signatures.append(f"context_{context_hash}")
        
        return signatures
    
    def _assess_quantum_health(self, fluctuations: List[Dict[str, Any]]) -> str:
        """评估量子健康状态"""
        if not fluctuations:
            return "unstable"
        
        coherence_values = [f["coherence"] for f in fluctuations]
        entanglement_values = [f["entanglement"] for f in fluctuations]
        
        avg_coherence = np.mean(coherence_values)
        avg_entanglement = np.mean(entanglement_values)
        
        # 健康评估
        if avg_coherence > 0.8 and avg_entanglement > 0.6:
            return "excellent"
        elif avg_coherence > 0.6 and avg_entanglement > 0.4:
            return "good"
        elif avg_coherence > 0.4 and avg_entanglement > 0.2:
            return "fair"
        else:
            return "poor"

# 全局量子意识流系统实例
_quantum_consciousness_instance = None

def get_quantum_consciousness_system(config: Dict[str, Any] = None) -> QuantumEnhancedConsciousnessSystem:
    """获取量子增强意识流系统实例"""
    global _quantum_consciousness_instance
    if _quantum_consciousness_instance is None:
        _quantum_consciousness_instance = QuantumEnhancedConsciousnessSystem(config)
    return _quantum_consciousness_instance

if __name__ == "__main__":
    # 测试代码
    print("量子增强意识流系统 V2 测试")
    print("=" * 50)
    
    # 创建系统实例
    config = {
        "quantum_processor": {"qubit_count": 8, "circuit_depth": 4},
        "memory_manager": {"max_memory_size": 5000}
    }
    
    consciousness_system = get_quantum_consciousness_system(config)
    
    # 测试事件记录
    test_events = [
        {
            "type": ConsciousnessEventType.THOUGHT,
            "content": {"task": "分析系统架构", "complexity": "high"},
            "emotional_charge": 0.8,
            "cognitive_load": 0.9
        },
        {
            "type": ConsciousnessEventType.INSIGHT,
            "content": {"insight": "发现性能瓶颈", "solution": "缓存优化"},
            "emotional_charge": 0.9,
            "cognitive_load": 0.6
        },
        {
            "type": ConsciousnessEventType.DECISION_POINT,
            "content": {"decision": "选择技术栈", "options": ["React", "Vue", "Angular"]},
            "emotional_charge": 0.7,
            "cognitive_load": 0.8
        }
    ]
    
    print("记录测试事件:")
    for i, event_data in enumerate(test_events, 1):
        event_id = consciousness_system.record_event(
            event_type=event_data["type"],
            content=event_data["content"],
            emotional_charge=event_data["emotional_charge"],
            cognitive_load=event_data["cognitive_load"]
        )
        print(f"  事件 {i}: {event_data['type'].value} -> ID: {event_id}")
    
    # 测试上下文检索
    print(f"\n测试上下文检索:")
    query = "系统架构分析"
    relevant_memories = consciousness_system.retrieve_context(query, top_k=5)
    print(f"  检索到 {len(relevant_memories)} 个相关记忆")
    
    # 测试意识流摘要
    print(f"\n意识流摘要:")
    summary = consciousness_system.get_consciousness_summary()
    for key, value in summary.items():
        if key != "dominant_types":
            print(f"  {key}: {value}")
    
    print(f"\n主导事件类型: {summary.get('dominant_types', {})}")