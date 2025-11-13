#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全能工作流V5 - 智能上下文压缩和注入系统
Universal Workflow V5 - Intelligent Context Compression and Injection System

基于向量嵌入和语义分析的智能上下文管理
"""

import os
import sys
import json
import yaml
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import re
import math
from collections import defaultdict

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class ContextType(Enum):
    """上下文类型"""
    PROJECT = "project"
    WORKFLOW = "workflow"
    AGENT = "agent"
    KNOWLEDGE = "knowledge"
    MEMORY = "memory"
    TOOL = "tool"
    CONFIG = "config"
    HISTORY = "history"

class CompressionLevel(Enum):
    """压缩级别"""
    NONE = 0
    LIGHT = 1
    MEDIUM = 2
    HEAVY = 3
    EXTREME = 4

@dataclass
class ContextChunk:
    """上下文块"""
    id: str
    content: str
    type: ContextType
    source: str
    importance: float = 0.5
    tokens: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

@dataclass
class ContextCompression:
    """上下文压缩结果"""
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    chunks: List[ContextChunk]
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class ContextEmbedder:
    """上下文嵌入器（简化版）"""
    
    def __init__(self):
        self.cache = {}
    
    def embed(self, text: str) -> List[float]:
        """生成文本嵌入"""
        # 简化的嵌入实现（实际应使用真实的嵌入模型）
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        # 生成伪嵌入向量（基于词频和字符特征）
        words = re.findall(r'\w+', text.lower())
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        
        # 创建固定长度的向量（128维）
        vector = []
        for i in range(128):
            # 基于词频和位置生成向量值
            value = 0.0
            for j, (word, freq) in enumerate(word_freq.items()):
                if (i + j) % 128 == 0:
                    value += freq * math.sin(i * 0.1)
            vector.append(value / (len(words) + 1))
        
        self.cache[text_hash] = vector
        return vector
    
    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算向量相似度"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class ContextCompressor:
    """上下文压缩器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.embedder = ContextEmbedder()
        self.chunks: Dict[str, ContextChunk] = {}
        self.compression_history: List[ContextCompression] = []
        
        # 配置参数
        self.max_tokens = self.config.get("max_tokens", 8000)
        self.min_tokens = self.config.get("min_tokens", 1000)
        self.importance_threshold = self.config.get("importance_threshold", 0.3)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.8)
        
        logger.info("上下文压缩器初始化完成")
    
    def add_context(self, content: str, context_type: ContextType, source: str, 
                   importance: float = 0.5, tags: List[str] = None, 
                   metadata: Dict[str, Any] = None) -> str:
        """添加上下文"""
        chunk_id = hashlib.md5(f"{content}{source}{datetime.now()}".encode()).hexdigest()
        
        # 估算token数量（简化计算）
        tokens = len(content.split()) * 1.3  # 假设平均1.3个字符/token
        
        # 生成嵌入
        embedding = self.embedder.embed(content)
        
        chunk = ContextChunk(
            id=chunk_id,
            content=content,
            type=context_type,
            source=source,
            importance=importance,
            tokens=int(tokens),
            tags=tags or [],
            metadata=metadata or {},
            embedding=embedding
        )
        
        self.chunks[chunk_id] = chunk
        logger.debug(f"添加上下文块: {chunk_id} ({context_type.value})")
        
        return chunk_id
    
    def remove_context(self, chunk_id: str) -> bool:
        """移除上下文"""
        if chunk_id in self.chunks:
            del self.chunks[chunk_id]
            logger.debug(f"移除上下文块: {chunk_id}")
            return True
        return False
    
    def get_context(self, chunk_id: str) -> Optional[ContextChunk]:
        """获取上下文"""
        chunk = self.chunks.get(chunk_id)
        if chunk:
            chunk.last_accessed = datetime.now()
            chunk.access_count += 1
        return chunk
    
    def compress_context(self, query: str = "", max_tokens: int = None, 
                        level: CompressionLevel = CompressionLevel.MEDIUM) -> ContextCompression:
        """压缩上下文"""
        max_tokens = max_tokens or self.max_tokens
        
        # 获取所有上下文块
        all_chunks = list(self.chunks.values())
        
        if not all_chunks:
            return ContextCompression(
                original_tokens=0,
                compressed_tokens=0,
                compression_ratio=1.0,
                chunks=[],
                summary="无上下文"
            )
        
        # 计算原始token数
        original_tokens = sum(chunk.tokens for chunk in all_chunks)
        
        # 根据压缩级别选择策略
        if level == CompressionLevel.NONE:
            selected_chunks = all_chunks
        elif level == CompressionLevel.LIGHT:
            selected_chunks = self._light_compression(all_chunks, query)
        elif level == CompressionLevel.MEDIUM:
            selected_chunks = self._medium_compression(all_chunks, query)
        elif level == CompressionLevel.HEAVY:
            selected_chunks = self._heavy_compression(all_chunks, query)
        else:  # EXTREME
            selected_chunks = self._extreme_compression(all_chunks, query)
        
        # 确保不超过最大token限制
        selected_chunks = self._enforce_token_limit(selected_chunks, max_tokens)
        
        # 计算压缩后token数
        compressed_tokens = sum(chunk.tokens for chunk in selected_chunks)
        
        # 生成摘要
        summary = self._generate_summary(selected_chunks, query)
        
        # 创建压缩结果
        compression = ContextCompression(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            chunks=selected_chunks,
            summary=summary,
            metadata={
                "level": level.value,
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # 记录压缩历史
        self.compression_history.append(compression)
        
        logger.info(f"上下文压缩完成: {original_tokens} -> {compressed_tokens} tokens (压缩比: {compression.compression_ratio:.2f})")
        
        return compression
    
    def _light_compression(self, chunks: List[ContextChunk], query: str) -> List[ContextChunk]:
        """轻度压缩"""
        # 过滤低重要性的块
        filtered = [chunk for chunk in chunks if chunk.importance >= self.importance_threshold]
        
        # 如果有查询，按相关性排序
        if query:
            query_embedding = self.embedder.embed(query)
            filtered.sort(key=lambda x: self.embedder.similarity(query_embedding, x.embedding), reverse=True)
        
        return filtered
    
    def _medium_compression(self, chunks: List[ContextChunk], query: str) -> List[ContextChunk]:
        """中度压缩"""
        # 去重相似内容
        deduplicated = self._deduplicate_chunks(chunks)
        
        # 按重要性排序
        deduplicated.sort(key=lambda x: x.importance, reverse=True)
        
        # 如果有查询，调整排序
        if query:
            query_embedding = self.embedder.embed(query)
            for chunk in deduplicated:
                similarity = self.embedder.similarity(query_embedding, chunk.embedding)
                chunk.metadata["query_similarity"] = similarity
            
            # 综合重要性和相似性排序
            deduplicated.sort(key=lambda x: x.importance * 0.7 + x.metadata.get("query_similarity", 0) * 0.3, reverse=True)
        
        return deduplicated
    
    def _heavy_compression(self, chunks: List[ContextChunk], query: str) -> List[ContextChunk]:
        """重度压缩"""
        # 先去重
        deduplicated = self._deduplicate_chunks(chunks)
        
        # 按类型分组，保留每个类型最重要的内容
        type_groups = defaultdict(list)
        for chunk in deduplicated:
            type_groups[chunk.type].append(chunk)
        
        selected = []
        for context_type, type_chunks in type_groups.items():
            # 按重要性排序，保留前50%
            type_chunks.sort(key=lambda x: x.importance, reverse=True)
            keep_count = max(1, len(type_chunks) // 2)
            selected.extend(type_chunks[:keep_count])
        
        # 如果有查询，进一步筛选
        if query:
            query_embedding = self.embedder.embed(query)
            selected_with_similarity = []
            for chunk in selected:
                similarity = self.embedder.similarity(query_embedding, chunk.embedding)
                chunk.metadata["query_similarity"] = similarity
                selected_with_similarity.append((chunk, similarity))
            
            # 保留相似性最高的内容
            selected_with_similarity.sort(key=lambda x: x[1], reverse=True)
            selected = [chunk for chunk, _ in selected_with_similarity[:len(selected) // 2]]
        
        return selected
    
    def _extreme_compression(self, chunks: List[ContextChunk], query: str) -> List[ContextChunk]:
        """极度压缩"""
        # 去重
        deduplicated = self._deduplicate_chunks(chunks)
        
        # 只保留最重要的内容
        deduplicated.sort(key=lambda x: x.importance, reverse=True)
        
        # 保留前20%或至少1个
        keep_count = max(1, len(deduplicated) // 5)
        selected = deduplicated[:keep_count]
        
        # 如果有查询，只保留最相关的
        if query and len(selected) > 1:
            query_embedding = self.embedder.embed(query)
            selected_with_similarity = []
            for chunk in selected:
                similarity = self.embedder.similarity(query_embedding, chunk.embedding)
                selected_with_similarity.append((chunk, similarity))
            
            # 只保留最相关的1个
            selected_with_similarity.sort(key=lambda x: x[1], reverse=True)
            selected = [selected_with_similarity[0][0]]
        
        return selected
    
    def _deduplicate_chunks(self, chunks: List[ContextChunk]) -> List[ContextChunk]:
        """去重相似内容"""
        if not chunks:
            return []
        
        deduplicated = []
        processed = set()
        
        for chunk in chunks:
            is_duplicate = False
            
            for existing_id in processed:
                existing_chunk = self.chunks[existing_id]
                
                # 检查内容相似性
                if chunk.embedding and existing_chunk.embedding:
                    similarity = self.embedder.similarity(chunk.embedding, existing_chunk.embedding)
                    if similarity > self.similarity_threshold:
                        # 保留更重要的
                        if chunk.importance > existing_chunk.importance:
                            deduplicated = [c for c in deduplicated if c.id != existing_id]
                            deduplicated.append(chunk)
                            processed.remove(existing_id)
                            processed.add(chunk.id)
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                deduplicated.append(chunk)
                processed.add(chunk.id)
        
        return deduplicated
    
    def _enforce_token_limit(self, chunks: List[ContextChunk], max_tokens: int) -> List[ContextChunk]:
        """强制执行token限制"""
        if not chunks:
            return []
        
        selected = []
        current_tokens = 0
        
        for chunk in chunks:
            if current_tokens + chunk.tokens <= max_tokens:
                selected.append(chunk)
                current_tokens += chunk.tokens
            else:
                # 尝试截断最后一个块
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > self.min_tokens:
                    # 截断内容
                    truncated_content = self._truncate_content(chunk.content, remaining_tokens)
                    truncated_chunk = ContextChunk(
                        id=chunk.id + "_truncated",
                        content=truncated_content,
                        type=chunk.type,
                        source=chunk.source,
                        importance=chunk.importance,
                        tokens=remaining_tokens,
                        tags=chunk.tags,
                        metadata={**chunk.metadata, "truncated": True}
                    )
                    selected.append(truncated_chunk)
                break
        
        return selected
    
    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """截断内容"""
        # 简单的截断策略（实际应更智能）
        words = content.split()
        max_words = int(max_tokens / 1.3)  # 假设平均1.3个字符/token
        
        if len(words) <= max_words:
            return content
        
        # 截断并添加省略号
        truncated = " ".join(words[:max_words])
        return truncated + "...\n[内容已截断]"
    
    def _generate_summary(self, chunks: List[ContextChunk], query: str) -> str:
        """生成摘要"""
        if not chunks:
            return "无上下文内容"
        
        # 统计信息
        total_chunks = len(chunks)
        total_tokens = sum(chunk.tokens for chunk in chunks)
        
        # 类型分布
        type_counts = defaultdict(int)
        for chunk in chunks:
            type_counts[chunk.type.value] += 1
        
        # 来源分布
        sources = list(set(chunk.source for chunk in chunks))
        
        # 构建摘要
        summary_parts = [
            f"上下文摘要: 共{total_chunks}个块, {total_tokens}个tokens",
            f"类型分布: {', '.join([f'{k}({v})' for k, v in type_counts.items()])}",
            f"来源: {', '.join(sources[:5])}" + ("..." if len(sources) > 5 else "")
        ]
        
        if query:
            summary_parts.append(f"查询: {query}")
        
        # 添加关键内容预览
        if chunks:
            most_important = max(chunks, key=lambda x: x.importance)
            preview = most_important.content[:100] + "..." if len(most_important.content) > 100 else most_important.content
            summary_parts.append(f"最重要内容预览: {preview}")
        
        return "\n".join(summary_parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.chunks:
            return {"total_chunks": 0, "total_tokens": 0}
        
        total_chunks = len(self.chunks)
        total_tokens = sum(chunk.tokens for chunk in self.chunks.values())
        
        # 类型分布
        type_counts = defaultdict(int)
        for chunk in self.chunks.values():
            type_counts[chunk.type.value] += 1
        
        # 重要性分布
        importance_ranges = {"low": 0, "medium": 0, "high": 0}
        for chunk in self.chunks.values():
            if chunk.importance < 0.3:
                importance_ranges["low"] += 1
            elif chunk.importance < 0.7:
                importance_ranges["medium"] += 1
            else:
                importance_ranges["high"] += 1
        
        # 访问统计
        accessed_chunks = sum(1 for chunk in self.chunks.values() if chunk.access_count > 0)
        
        return {
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "type_distribution": dict(type_counts),
            "importance_distribution": importance_ranges,
            "accessed_chunks": accessed_chunks,
            "compression_history_count": len(self.compression_history)
        }

class ContextInjector:
    """上下文注入器"""
    
    def __init__(self, compressor: ContextCompressor):
        self.compressor = compressor
        self.injection_history: List[Dict[str, Any]] = []
        
        # 注入模板
        self.templates = {
            "project": "## 项目信息\n{content}",
            "workflow": "## 工作流信息\n{content}",
            "agent": "## 智能体信息\n{content}",
            "knowledge": "## 知识库\n{content}",
            "memory": "## 记忆信息\n{content}",
            "tool": "## 工具信息\n{content}",
            "config": "## 配置信息\n{content}",
            "history": "## 历史记录\n{content}"
        }
    
    def inject_context(self, prompt: str, context_types: List[ContextType] = None,
                      max_tokens: int = 2000, level: CompressionLevel = CompressionLevel.MEDIUM) -> str:
        """注入上下文到提示"""
        # 获取指定类型的上下文
        if context_types:
            relevant_chunks = []
            for chunk in self.compressor.chunks.values():
                if chunk.type in context_types:
                    relevant_chunks.append(chunk)
        else:
            relevant_chunks = list(self.compressor.chunks.values())
        
        # 临时添加所有相关块到压缩器
        temp_compressor = ContextCompressor()
        for chunk in relevant_chunks:
            temp_compressor.chunks[chunk.id] = chunk
        
        # 压缩上下文
        compression = temp_compressor.compress_context(
            query=prompt,
            max_tokens=max_tokens,
            level=level
        )
        
        # 构建上下文字符串
        context_parts = []
        
        # 按类型组织内容
        type_groups = defaultdict(list)
        for chunk in compression.chunks:
            type_groups[chunk.type].append(chunk)
        
        # 按重要性排序类型
        type_priority = {
            ContextType.PROJECT: 1,
            ContextType.WORKFLOW: 2,
            ContextType.AGENT: 3,
            ContextType.KNOWLEDGE: 4,
            ContextType.CONFIG: 5,
            ContextType.TOOL: 6,
            ContextType.MEMORY: 7,
            ContextType.HISTORY: 8
        }
        
        sorted_types = sorted(type_groups.keys(), key=lambda x: type_priority.get(x, 9))
        
        for context_type in sorted_types:
            chunks = type_groups[context_type]
            if chunks:
                template = self.templates.get(context_type.value, "## {type}\n{content}")
                
                # 合并同类型内容
                type_content = []
                for chunk in chunks:
                    type_content.append(f"### {chunk.source}\n{chunk.content}")
                
                context_parts.append(template.format(
                    type=context_type.value,
                    content="\n\n".join(type_content)
                ))
        
        # 构建最终提示
        if context_parts:
            context_str = "\n\n".join(context_parts)
            injected_prompt = f"{context_str}\n\n## 用户请求\n{prompt}"
        else:
            injected_prompt = prompt
        
        # 记录注入历史
        self.injection_history.append({
            "timestamp": datetime.now().isoformat(),
            "prompt_length": len(prompt),
            "injected_length": len(injected_prompt),
            "context_types": [t.value for t in (context_types or [])],
            "compression_ratio": compression.compression_ratio,
            "chunks_used": len(compression.chunks)
        })
        
        return injected_prompt
    
    def get_injection_statistics(self) -> Dict[str, Any]:
        """获取注入统计信息"""
        if not self.injection_history:
            return {"total_injections": 0}
        
        total_injections = len(self.injection_history)
        avg_compression_ratio = sum(h["compression_ratio"] for h in self.injection_history) / total_injections
        avg_chunks_used = sum(h["chunks_used"] for h in self.injection_history) / total_injections
        
        return {
            "total_injections": total_injections,
            "average_compression_ratio": avg_compression_ratio,
            "average_chunks_used": avg_chunks_used,
            "last_injection": self.injection_history[-1]["timestamp"]
        }

# 全局实例
_compressor = None
_injector = None

def get_context_compressor(config: Dict[str, Any] = None) -> ContextCompressor:
    """获取上下文压缩器实例"""
    global _compressor
    if _compressor is None:
        _compressor = ContextCompressor(config)
    return _compressor

def get_context_injector(config: Dict[str, Any] = None) -> ContextInjector:
    """获取上下文注入器实例"""
    global _injector
    if _injector is None:
        compressor = get_context_compressor(config)
        _injector = ContextInjector(compressor)
    return _injector

if __name__ == "__main__":
    # 测试代码
    config = {
        "max_tokens": 4000,
        "importance_threshold": 0.3,
        "similarity_threshold": 0.8
    }
    
    compressor = get_context_compressor(config)
    injector = get_context_injector(config)
    
    # 添加测试上下文
    compressor.add_context(
        "这是一个测试项目，使用Python开发。",
        ContextType.PROJECT,
        "test_project",
        importance=0.8
    )
    
    compressor.add_context(
        "项目包含前端React和后端Flask。",
        ContextType.PROJECT,
        "tech_stack",
        importance=0.7
    )
    
    compressor.add_context(
        "用户需要实现用户认证功能。",
        ContextType.WORKFLOW,
        "user_request",
        importance=0.9
    )
    
    # 测试压缩
    compression = compressor.compress_context(
        query="用户认证",
        level=CompressionLevel.MEDIUM
    )
    
    print("压缩结果:")
    print(f"原始tokens: {compression.original_tokens}")
    print(f"压缩tokens: {compression.compressed_tokens}")
    print(f"压缩比: {compression.compression_ratio:.2f}")
    print(f"摘要: {compression.summary}")
    
    # 测试注入
    prompt = "请帮我实现用户认证功能"
    injected_prompt = injector.inject_context(
        prompt,
        context_types=[ContextType.PROJECT, ContextType.WORKFLOW],
        max_tokens=1000
    )
    
    print("\n注入后的提示:")
    print(injected_prompt)
    
    # 统计信息
    print("\n压缩器统计:")
    print(json.dumps(compressor.get_statistics(), ensure_ascii=False, indent=2))
    
    print("\n注入器统计:")
    print(json.dumps(injector.get_injection_statistics(), ensure_ascii=False, indent=2))
