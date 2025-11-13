#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能体记忆、学习和通信系统
Agent Memory, Learning and Communication System

作者: Quantum AI Team
版本: 5.1.0
日期: 2025-11-12
"""

import numpy as np
import json
import pickle
import time
import logging
import hashlib
from typing import Dict, List, Tuple, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from pathlib import Path
import threading
import sqlite3
import uuid

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryItem:
    """记忆项"""
    id: str
    agent_id: str
    content: Any
    memory_type: str  # episodic, semantic, procedural, working
    importance: float  # 0.0-1.0
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    tags: Set[str] = field(default_factory=set)
    connections: Set[str] = field(default_factory=set)  # 连接的记忆ID
    embedding: Optional[np.ndarray] = None

@dataclass
class LearningExperience:
    """学习经验"""
    id: str
    agent_id: str
    situation: Dict[str, Any]
    action: Any
    outcome: Any
    reward: float
    lesson: str
    timestamp: float
    generalization_score: float = 0.0
    applicability_context: Set[str] = field(default_factory=set)

@dataclass
class CommunicationMessage:
    """通信消息"""
    id: str
    sender_id: str
    receiver_id: str
    message_type: str  # request, response, broadcast, alert
    content: Any
    priority: int  # 1-10
    timestamp: float
    ttl: float = 3600  # 生存时间(秒)
    requires_ack: bool = False
    ack_received: bool = False

class AgentMemory:
    """智能体记忆系统"""
    
    def __init__(self, agent_id: str, storage_path: Optional[str] = None):
        """
        初始化记忆系统
        
        Args:
            agent_id: 智能体ID
            storage_path: 存储路径
        """
        self.agent_id = agent_id
        self.storage_path = storage_path or f"memory_{agent_id}.db"
        
        # 记忆存储
        self.episodic_memory = {}  # 情节记忆
        self.semantic_memory = {}   # 语义记忆
        self.procedural_memory = {} # 程序性记忆
        self.working_memory = deque(maxlen=7)  # 工作记忆(7±2规则)
        
        # 学习经验
        self.learning_experiences = {}
        
        # 记忆索引
        self.tag_index = defaultdict(set)
        self.connection_index = defaultdict(set)
        self.temporal_index = defaultdict(list)
        
        # 记忆参数
        self.consolidation_threshold = 0.7
        self.forgetting_rate = 0.01
        self.consolidation_interval = 100  # 记忆巩固间隔
        
        # 初始化数据库
        self._init_database()
        
        # 加载现有记忆
        self._load_memories()
    
    def _init_database(self):
        """初始化数据库"""
        self.conn = sqlite3.connect(self.storage_path)
        self.cursor = self.conn.cursor()
        
        # 创建表
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                agent_id TEXT,
                memory_type TEXT,
                content TEXT,
                importance REAL,
                timestamp REAL,
                access_count INTEGER,
                last_accessed REAL,
                tags TEXT,
                connections TEXT,
                embedding BLOB
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_experiences (
                id TEXT PRIMARY KEY,
                agent_id TEXT,
                situation TEXT,
                action TEXT,
                outcome TEXT,
                reward REAL,
                lesson TEXT,
                timestamp REAL,
                generalization_score REAL,
                applicability_context TEXT
            )
        ''')
        
        self.conn.commit()
    
    def store_memory(self, 
                     content: Any, 
                     memory_type: str, 
                     importance: float = 0.5,
                     tags: Optional[List[str]] = None) -> str:
        """
        存储记忆
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            importance: 重要性
            tags: 标签
            
        Returns:
            记忆ID
        """
        memory_id = str(uuid.uuid4())
        
        # 创建记忆项
        memory_item = MemoryItem(
            id=memory_id,
            agent_id=self.agent_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            timestamp=time.time(),
            tags=set(tags or [])
        )
        
        # 生成嵌入向量
        memory_item.embedding = self._generate_embedding(content)
        
        # 存储到相应的记忆系统
        if memory_type == 'episodic':
            self.episodic_memory[memory_id] = memory_item
        elif memory_type == 'semantic':
            self.semantic_memory[memory_id] = memory_item
        elif memory_type == 'procedural':
            self.procedural_memory[memory_id] = memory_item
        elif memory_type == 'working':
            self.working_memory.append(memory_item)
        
        # 更新索引
        self._update_indices(memory_item)
        
        # 存储到数据库
        self._store_memory_to_db(memory_item)
        
        # 检查是否需要巩固
        if len(self.episodic_memory) % self.consolidation_interval == 0:
            self._consolidate_memories()
        
        logger.debug(f"存储记忆: {memory_id} - 类型: {memory_type}")
        return memory_id
    
    def retrieve_memory(self, 
                       query: Any, 
                       memory_type: Optional[str] = None,
                       max_results: int = 5) -> List[MemoryItem]:
        """
        检索记忆
        
        Args:
            query: 查询内容
            memory_type: 记忆类型
            max_results: 最大结果数
            
        Returns:
            相关记忆列表
        """
        # 生成查询嵌入
        query_embedding = self._generate_embedding(query)
        
        # 搜索相关记忆
        candidates = []
        
        # 根据类型搜索
        search_spaces = []
        if memory_type:
            if memory_type == 'episodic':
                search_spaces.append(self.episodic_memory)
            elif memory_type == 'semantic':
                search_spaces.append(self.semantic_memory)
            elif memory_type == 'procedural':
                search_spaces.append(self.procedural_memory)
            elif memory_type == 'working':
                search_spaces.append(list(self.working_memory))
        else:
            search_spaces = [self.episodic_memory, self.semantic_memory, self.procedural_memory]
        
        for memory_space in search_spaces:
            for memory_item in memory_space.values():
                if isinstance(memory_space, dict):
                    candidates.append(memory_item)
                else:  # working_memory is a deque
                    candidates.append(memory_item)
        
        # 计算相似度
        scored_memories = []
        for memory in candidates:
            if memory.embedding is not None:
                similarity = self._calculate_similarity(query_embedding, memory.embedding)
                
                # 考虑重要性和访问频率
                access_factor = memory.access_count * 0.1
                importance_factor = memory.importance * 0.3
                recency_factor = self._calculate_recency_factor(memory.timestamp)
                
                total_score = similarity * 0.6 + importance_factor + access_factor + recency_factor
                scored_memories.append((total_score, memory))
        
        # 排序并返回top-k
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        # 更新访问统计
        result_memories = []
        for score, memory in scored_memories[:max_results]:
            memory.access_count += 1
            memory.last_accessed = time.time()
            result_memories.append(memory)
        
        return result_memories
    
    def store_learning_experience(self, 
                                 situation: Dict[str, Any],
                                 action: Any,
                                 outcome: Any,
                                 reward: float,
                                 lesson: str) -> str:
        """
        存储学习经验
        
        Args:
            situation: 情境
            action: 动作
            outcome: 结果
            reward: 奖励
            lesson: 学到的经验
            
        Returns:
            经验ID
        """
        experience_id = str(uuid.uuid4())
        
        experience = LearningExperience(
            id=experience_id,
            agent_id=self.agent_id,
            situation=situation,
            action=action,
            outcome=outcome,
            reward=reward,
            lesson=lesson,
            timestamp=time.time()
        )
        
        # 计算泛化分数
        experience.generalization_score = self._calculate_generalization_score(experience)
        
        # 提取适用性上下文
        experience.applicability_context = self._extract_applicability_context(situation)
        
        # 存储经验
        self.learning_experiences[experience_id] = experience
        
        # 存储到数据库
        self._store_experience_to_db(experience)
        
        # 如果是重要的经验，转换为语义记忆
        if reward > 0.7 or abs(reward) < 0.1:
            self._convert_experience_to_memory(experience)
        
        logger.debug(f"存储学习经验: {experience_id} - 奖励: {reward}")
        return experience_id
    
    def get_relevant_experiences(self, situation: Dict[str, Any], max_results: int = 3) -> List[LearningExperience]:
        """
        获取相关经验
        
        Args:
            situation: 当前情境
            max_results: 最大结果数
            
        Returns:
            相关经验列表
        """
        # 提取情境特征
        situation_context = self._extract_applicability_context(situation)
        
        # 计算相关性分数
        scored_experiences = []
        for experience in self.learning_experiences.values():
            relevance = self._calculate_situation_relevance(situation_context, experience.applicability_context)
            reward_factor = experience.reward * 0.3
            recency_factor = self._calculate_recency_factor(experience.timestamp)
            
            total_score = relevance * 0.5 + reward_factor + recency_factor
            scored_experiences.append((total_score, experience))
        
        # 排序并返回
        scored_experiences.sort(key=lambda x: x[0], reverse=True)
        return [exp for score, exp in scored_experiences[:max_results]]
    
    def _generate_embedding(self, content: Any) -> np.ndarray:
        """生成内容嵌入向量"""
        # 简化的嵌入生成（实际应用中可以使用更复杂的模型）
        content_str = str(content)
        
        # 基于内容哈希生成伪嵌入
        hash_object = hashlib.md5(content_str.encode())
        hex_dig = hash_object.hexdigest()
        
        # 转换为数值向量
        embedding = np.array([int(hex_dig[i:i+2], 16) / 255.0 for i in range(0, min(32, len(hex_dig)), 2)])
        
        # 填充到固定长度
        if len(embedding) < 16:
            embedding = np.pad(embedding, (0, 16 - len(embedding)))
        
        return embedding
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算嵌入相似度"""
        # 确保向量长度一致
        min_len = min(len(embedding1), len(embedding2))
        if min_len == 0:
            return 0.0
        
        # 余弦相似度
        emb1 = embedding1[:min_len]
        emb2 = embedding2[:min_len]
        
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_recency_factor(self, timestamp: float) -> float:
        """计算时间因子"""
        age = time.time() - timestamp
        # 越新的记忆时间因子越高
        return np.exp(-age / (30 * 24 * 3600))  # 30天衰减
    
    def _update_indices(self, memory_item: MemoryItem):
        """更新索引"""
        # 标签索引
        for tag in memory_item.tags:
            self.tag_index[tag].add(memory_item.id)
        
        # 时间索引
        date_key = time.strftime("%Y-%m-%d", time.localtime(memory_item.timestamp))
        self.temporal_index[date_key].append(memory_item.id)
    
    def _store_memory_to_db(self, memory_item: MemoryItem):
        """存储记忆到数据库"""
        self.cursor.execute('''
            INSERT OR REPLACE INTO memories 
            (id, agent_id, memory_type, content, importance, timestamp, 
             access_count, last_accessed, tags, connections, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory_item.id,
            memory_item.agent_id,
            memory_item.memory_type,
            json.dumps(memory_item.content),
            memory_item.importance,
            memory_item.timestamp,
            memory_item.access_count,
            memory_item.last_accessed,
            json.dumps(list(memory_item.tags)),
            json.dumps(list(memory_item.connections)),
            pickle.dumps(memory_item.embedding) if memory_item.embedding is not None else None
        ))
        self.conn.commit()
    
    def _store_experience_to_db(self, experience: LearningExperience):
        """存储经验到数据库"""
        self.cursor.execute('''
            INSERT OR REPLACE INTO learning_experiences 
            (id, agent_id, situation, action, outcome, reward, lesson, 
             timestamp, generalization_score, applicability_context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            experience.id,
            experience.agent_id,
            json.dumps(experience.situation),
            json.dumps(experience.action),
            json.dumps(experience.outcome),
            experience.reward,
            experience.lesson,
            experience.timestamp,
            experience.generalization_score,
            json.dumps(list(experience.applicability_context))
        ))
        self.conn.commit()
    
    def _load_memories(self):
        """加载现有记忆"""
        # 加载记忆
        self.cursor.execute('SELECT * FROM memories WHERE agent_id = ?', (self.agent_id,))
        rows = self.cursor.fetchall()
        
        for row in rows:
            memory_item = MemoryItem(
                id=row[0],
                agent_id=row[1],
                memory_type=row[2],
                content=json.loads(row[3]),
                importance=row[4],
                timestamp=row[5],
                access_count=row[6],
                last_accessed=row[7],
                tags=set(json.loads(row[8])),
                connections=set(json.loads(row[9])),
                embedding=pickle.loads(row[10]) if row[10] else None
            )
            
            # 存储到相应的记忆系统
            if memory_item.memory_type == 'episodic':
                self.episodic_memory[memory_item.id] = memory_item
            elif memory_item.memory_type == 'semantic':
                self.semantic_memory[memory_item.id] = memory_item
            elif memory_item.memory_type == 'procedural':
                self.procedural_memory[memory_item.id] = memory_item
            
            # 更新索引
            self._update_indices(memory_item)
        
        # 加载学习经验
        self.cursor.execute('SELECT * FROM learning_experiences WHERE agent_id = ?', (self.agent_id,))
        rows = self.cursor.fetchall()
        
        for row in rows:
            experience = LearningExperience(
                id=row[0],
                agent_id=row[1],
                situation=json.loads(row[2]),
                action=json.loads(row[3]),
                outcome=json.loads(row[4]),
                reward=row[5],
                lesson=row[6],
                timestamp=row[7],
                generalization_score=row[8],
                applicability_context=set(json.loads(row[9]))
            )
            self.learning_experiences[experience.id] = experience
        
        logger.info(f"加载了 {len(self.episodic_memory)} 个情节记忆, {len(self.semantic_memory)} 个语义记忆, {len(self.procedural_memory)} 个程序性记忆, {len(self.learning_experiences)} 个学习经验")
    
    def _consolidate_memories(self):
        """巩固记忆"""
        # 选择重要的情节记忆进行巩固
        important_episodic = [
            mem for mem in self.episodic_memory.values()
            if mem.importance > self.consolidation_threshold
        ]
        
        for episodic_memory in important_episodic:
            # 转换为语义记忆
            semantic_content = {
                'original_episode': episodic_memory.content,
                'abstract_pattern': self._extract_pattern(episodic_memory.content),
                'generalized_lesson': episodic_memory.content.get('lesson', ''),
                'consolidation_timestamp': time.time()
            }
            
            # 创建语义记忆
            semantic_memory_id = self.store_memory(
                content=semantic_content,
                memory_type='semantic',
                importance=episodic_memory.importance * 0.8,
                tags=['consolidated', 'episodic_derived'] + list(episodic_memory.tags)
            )
            
            # 建立连接
            episodic_memory.connections.add(semantic_memory_id)
            
            logger.debug(f"巩固记忆: {episodic_memory.id} -> {semantic_memory_id}")
    
    def _extract_pattern(self, content: Any) -> str:
        """提取模式"""
        # 简化的模式提取
        if isinstance(content, dict):
            keys = list(content.keys())
            return f"pattern_{'_'.join(sorted(keys)[:3])}"
        else:
            return f"pattern_{type(content).__name__}"
    
    def _calculate_generalization_score(self, experience: LearningExperience) -> float:
        """计算泛化分数"""
        # 基于情境复杂度和奖励的泛化分数
        situation_complexity = len(str(experience.situation)) / 1000.0  # 简化
        reward_magnitude = abs(experience.reward)
        
        # 复杂情境和中等奖励的泛化分数更高
        if reward_magnitude > 0.1 and reward_magnitude < 0.9:
            return min(1.0, situation_complexity * 0.5 + 0.5)
        else:
            return min(1.0, situation_complexity * 0.3)
    
    def _extract_applicability_context(self, situation: Dict[str, Any]) -> Set[str]:
        """提取适用性上下文"""
        context = set()
        
        # 基于情境类型
        if 'task_type' in situation:
            context.add(f"task_{situation['task_type']}")
        
        if 'complexity' in situation:
            context.add(f"complexity_{situation['complexity']}")
        
        if 'domain' in situation:
            context.add(f"domain_{situation['domain']}")
        
        # 基于键值
        for key, value in situation.items():
            if isinstance(value, str):
                context.add(f"{key}_{value}")
            elif isinstance(value, (int, float)):
                if value > 0.5:
                    context.add(f"{key}_high")
                else:
                    context.add(f"{key}_low")
        
        return context
    
    def _calculate_situation_relevance(self, current_context: Set[str], experience_context: Set[str]) -> float:
        """计算情境相关性"""
        if not current_context or not experience_context:
            return 0.0
        
        # Jaccard相似度
        intersection = len(current_context.intersection(experience_context))
        union = len(current_context.union(experience_context))
        
        return intersection / union if union > 0 else 0.0
    
    def _convert_experience_to_memory(self, experience: LearningExperience):
        """将经验转换为记忆"""
        memory_content = {
            'experience_id': experience.id,
            'lesson': experience.lesson,
            'situation_pattern': self._extract_pattern(experience.situation),
            'action_outcome': {
                'action': experience.action,
                'outcome': experience.outcome,
                'reward': experience.reward
            },
            'generalization_score': experience.generalization_score
        }
        
        self.store_memory(
            content=memory_content,
            memory_type='semantic',
            importance=min(1.0, abs(experience.reward)),
            tags=['learning_experience', 'lesson'] + list(experience.applicability_context)
        )
    
    def forget_memories(self):
        """遗忘记忆"""
        current_time = time.time()
        
        # 检查工作记忆
        self.working_memory = deque(
            [mem for mem in self.working_memory 
             if current_time - mem.timestamp < 3600],  # 1小时
            maxlen=7
        )
        
        # 检查长期记忆
        for memory_dict in [self.episodic_memory, self.semantic_memory, self.procedural_memory]:
            to_remove = []
            for memory_id, memory in memory_dict.items():
                # 遗忘概率基于重要性和时间
                age = current_time - memory.timestamp
                forgetting_probability = self.forgetting_rate * age / (24 * 3600) * (1 - memory.importance)
                
                if np.random.random() < forgetting_probability:
                    to_remove.append(memory_id)
            
            for memory_id in to_remove:
                del memory_dict[memory_id]
                # 从数据库删除
                self.cursor.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
            
            if to_remove:
                logger.debug(f"遗忘了 {len(to_remove)} 个记忆")
        
        self.conn.commit()
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取记忆统计"""
        total_memories = len(self.episodic_memory) + len(self.semantic_memory) + len(self.procedural_memory)
        
        # 计算平均重要性
        all_importances = []
        for memory_dict in [self.episodic_memory, self.semantic_memory, self.procedural_memory]:
            all_importances.extend([mem.importance for mem in memory_dict.values()])
        
        avg_importance = np.mean(all_importances) if all_importances else 0.0
        
        # 标签统计
        tag_counts = {}
        for tag, memory_ids in self.tag_index.items():
            tag_counts[tag] = len(memory_ids)
        
        return {
            'total_memories': total_memories,
            'episodic_count': len(self.episodic_memory),
            'semantic_count': len(self.semantic_memory),
            'procedural_count': len(self.procedural_memory),
            'working_memory_count': len(self.working_memory),
            'learning_experiences_count': len(self.learning_experiences),
            'average_importance': avg_importance,
            'tag_counts': dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        }

class AgentCommunicationSystem:
    """智能体通信系统"""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        初始化通信系统
        
        Args:
            storage_path: 存储路径
        """
        self.storage_path = storage_path or "communication.db"
        self.message_queue = deque()
        self.subscribers = defaultdict(set)
        self.message_history = {}
        
        # 初始化数据库
        self._init_database()
        
        # 通信线程
        self.communication_thread = threading.Thread(target=self._communication_loop, daemon=True)
        self.communication_thread.start()
    
    def _init_database(self):
        """初始化数据库"""
        self.conn = sqlite3.connect(self.storage_path)
        self.cursor = self.conn.cursor()
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                sender_id TEXT,
                receiver_id TEXT,
                message_type TEXT,
                content TEXT,
                priority INTEGER,
                timestamp REAL,
                ttl REAL,
                requires_ack BOOLEAN,
                ack_received BOOLEAN
            )
        ''')
        
        self.conn.commit()
    
    def send_message(self, 
                     sender_id: str,
                     receiver_id: str,
                     message_type: str,
                     content: Any,
                     priority: int = 5,
                     ttl: float = 3600,
                     requires_ack: bool = False) -> str:
        """
        发送消息
        
        Args:
            sender_id: 发送者ID
            receiver_id: 接收者ID
            message_type: 消息类型
            content: 消息内容
            priority: 优先级
            ttl: 生存时间
            requires_ack: 是否需要确认
            
        Returns:
            消息ID
        """
        message_id = str(uuid.uuid4())
        
        message = CommunicationMessage(
            id=message_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            priority=priority,
            timestamp=time.time(),
            ttl=ttl,
            requires_ack=requires_ack
        )
        
        # 存储消息
        self.message_history[message_id] = message
        self._store_message_to_db(message)
        
        # 添加到队列
        self.message_queue.append(message)
        
        logger.debug(f"发送消息: {message_id} 从 {sender_id} 到 {receiver_id}")
        return message_id
    
    def subscribe(self, agent_id: str, message_types: List[str]):
        """
        订阅消息
        
        Args:
            agent_id: 智能体ID
            message_types: 消息类型列表
        """
        for message_type in message_types:
            self.subscribers[message_type].add(agent_id)
        
        logger.info(f"智能体 {agent_id} 订阅了消息类型: {message_types}")
    
    def get_messages(self, agent_id: str, message_type: Optional[str] = None) -> List[CommunicationMessage]:
        """
        获取消息
        
        Args:
            agent_id: 智能体ID
            message_type: 消息类型
            
        Returns:
            消息列表
        """
        messages = []
        
        for message in self.message_history.values():
            if message.receiver_id == agent_id:
                if message_type is None or message.message_type == message_type:
                    messages.append(message)
        
        # 按优先级和时间排序
        messages.sort(key=lambda x: (-x.priority, x.timestamp))
        
        return messages
    
    def _communication_loop(self):
        """通信循环"""
        while True:
            try:
                # 处理消息队列
                if self.message_queue:
                    message = self.message_queue.popleft()
                    self._process_message(message)
                
                # 清理过期消息
                self._cleanup_expired_messages()
                
                time.sleep(0.1)  # 100ms间隔
                
            except Exception as e:
                logger.error(f"通信循环错误: {e}")
                time.sleep(1)
    
    def _process_message(self, message: CommunicationMessage):
        """处理消息"""
        # 检查订阅
        if message.message_type in self.subscribers:
            for subscriber_id in self.subscribers[message.message_type]:
                if subscriber_id == message.receiver_id or message.receiver_id == "broadcast":
                    # 消息已送达
                    logger.debug(f"消息 {message.id} 已送达给 {subscriber_id}")
                    
                    # 如果需要确认，发送确认消息
                    if message.requires_ack and not message.ack_received:
                        ack_message = CommunicationMessage(
                            id=str(uuid.uuid4()),
                            sender_id=message.receiver_id,
                            receiver_id=message.sender_id,
                            message_type="ack",
                            content={"original_message_id": message.id},
                            priority=1,
                            timestamp=time.time()
                        )
                        self.message_queue.append(ack_message)
    
    def _cleanup_expired_messages(self):
        """清理过期消息"""
        current_time = time.time()
        expired_messages = []
        
        for message_id, message in self.message_history.items():
            if current_time - message.timestamp > message.ttl:
                expired_messages.append(message_id)
        
        for message_id in expired_messages:
            del self.message_history[message_id]
            self.cursor.execute('DELETE FROM messages WHERE id = ?', (message_id,))
        
        if expired_messages:
            self.conn.commit()
            logger.debug(f"清理了 {len(expired_messages)} 个过期消息")
    
    def _store_message_to_db(self, message: CommunicationMessage):
        """存储消息到数据库"""
        self.cursor.execute('''
            INSERT OR REPLACE INTO messages 
            (id, sender_id, receiver_id, message_type, content, priority, 
             timestamp, ttl, requires_ack, ack_received)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            message.id,
            message.sender_id,
            message.receiver_id,
            message.message_type,
            json.dumps(message.content),
            message.priority,
            message.timestamp,
            message.ttl,
            message.requires_ack,
            message.ack_received
        ))
        self.conn.commit()

# 示例使用
def example_memory_communication():
    """示例记忆和通信系统使用"""
    # 创建记忆系统
    memory_system = AgentMemory("agent_001")
    
    # 存储不同类型的记忆
    episodic_id = memory_system.store_memory(
        content={"task": "code_review", "file": "main.py", "issues": ["naming", "structure"]},
        memory_type="episodic",
        importance=0.8,
        tags=["coding", "review"]
    )
    
    semantic_id = memory_system.store_memory(
        content={"concept": "clean_code", "principles": ["readability", "maintainability"]},
        memory_type="semantic",
        importance=0.9,
        tags=["coding", "best_practices"]
    )
    
    # 存储学习经验
    experience_id = memory_system.store_learning_experience(
        situation={"task": "debugging", "language": "python", "complexity": "medium"},
        action="used_print_statements",
        outcome="found_bug_quickly",
        reward=0.7,
        lesson="print_statements_are_useful_for_quick_debugging"
    )
    
    # 检索相关记忆
    query = {"task": "debugging", "language": "python"}
    relevant_memories = memory_system.retrieve_memory(query)
    print(f"找到 {len(relevant_memories)} 个相关记忆")
    
    # 获取相关经验
    relevant_experiences = memory_system.get_relevant_experiences(query)
    print(f"找到 {len(relevant_experiences)} 个相关经验")
    
    # 创建通信系统
    comm_system = AgentCommunicationSystem()
    
    # 订阅消息
    comm_system.subscribe("agent_001", ["request", "response"])
    comm_system.subscribe("agent_002", ["request", "response"])
    
    # 发送消息
    message_id = comm_system.send_message(
        sender_id="agent_001",
        receiver_id="agent_002",
        message_type="request",
        content={"action": "help_with_debugging", "file": "buggy_code.py"},
        priority=7,
        requires_ack=True
    )
    
    # 获取消息
    time.sleep(0.2)  # 等待消息处理
    messages = comm_system.get_messages("agent_002")
    print(f"agent_002 收到 {len(messages)} 个消息")
    
    # 获取统计信息
    memory_stats = memory_system.get_memory_statistics()
    print(f"记忆统计: {memory_stats}")

if __name__ == "__main__":
    example_memory_communication()