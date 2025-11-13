#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全能工作流V5 - 智能体注册和发现系统
Agent Registry and Discovery System

作者: Universal AI Team
版本: 5.0.0
日期: 2025-11-12

特性:
- 智能体自动发现和注册
- 动态加载和热更新
- 智能匹配和推荐
- 能力评估和评分
- 版本控制和依赖管理
- 分布式智能体网络
"""

import os
import sys
import json
import time
import asyncio
import logging
import importlib
import inspect
import hashlib
from typing import Dict, List, Any, Optional, Union, Type, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import yaml
import pickle

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    """智能体类型"""
    CORE = "core"                    # 核心智能体
    SPECIALIZED = "specialized"      # 专业智能体
    CUSTOM = "custom"                # 自定义智能体
    EXTERNAL = "external"            # 外部智能体
    QUANTUM = "quantum"              # 量子智能体
    HYBRID = "hybrid"                # 混合智能体

class AgentStatus(Enum):
    """智能体状态"""
    INACTIVE = "inactive"            # 未激活
    ACTIVE = "active"                # 激活中
    BUSY = "busy"                    # 忙碌中
    ERROR = "error"                  # 错误状态
    UPDATING = "updating"            # 更新中
    MAINTENANCE = "maintenance"      # 维护中

class CapabilityLevel(Enum):
    """能力等级"""
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    MASTER = 5
    GRANDMASTER = 6

@dataclass
class AgentCapability:
    """智能体能力定义"""
    name: str
    description: str
    level: CapabilityLevel
    experience_years: float
    success_rate: float
    confidence: float
    tools: List[str]
    certifications: List[str]
    specializations: List[str]
    performance_metrics: Dict[str, float]
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'level': self.level.value,
            'experience_years': self.experience_years,
            'success_rate': self.success_rate,
            'confidence': self.confidence,
            'tools': self.tools,
            'certifications': self.certifications,
            'specializations': self.specializations,
            'performance_metrics': self.performance_metrics,
            'last_updated': self.last_updated.isoformat()
        }

@dataclass
class AgentMetadata:
    """智能体元数据"""
    id: str
    name: str
    version: str
    description: str
    author: str
    created_at: datetime
    updated_at: datetime
    agent_type: AgentType
    status: AgentStatus
    capabilities: List[AgentCapability]
    dependencies: List[str]
    tags: List[str]
    file_path: str
    class_name: str
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'agent_type': self.agent_type.value,
            'status': self.status.value,
            'capabilities': [cap.to_dict() for cap in self.capabilities],
            'dependencies': self.dependencies,
            'tags': self.tags,
            'file_path': self.file_path,
            'class_name': self.class_name,
            'config': self.config
        }

class AgentRegistry:
    """智能体注册表"""
    
    def __init__(self, registry_path: str = None):
        self.registry_path = registry_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            '.iflow', 'agents'
        )
        self.agents: Dict[str, AgentMetadata] = {}
        self.agent_instances: Dict[str, Any] = {}
        self.capability_index: Dict[str, List[str]] = defaultdict(list)
        self.type_index: Dict[AgentType, List[str]] = defaultdict(list)
        self.tag_index: Dict[str, List[str]] = defaultdict(list)
        self.dependency_graph: Dict[str, List[str]] = defaultdict(list)
        self.reverse_dependency_graph: Dict[str, List[str]] = defaultdict(list)
        
        # 性能监控
        self.performance_history: Dict[str, List[Dict]] = defaultdict(list)
        self.usage_stats: Dict[str, Dict] = defaultdict(dict)
        
        # 配置
        self.config = self._load_config()
        
        # 初始化
        self._initialize_registry()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        config_path = os.path.join(self.registry_path, 'config', 'registry-config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {
            'auto_discovery': True,
            'hot_reload': True,
            'performance_monitoring': True,
            'max_agents': 1000,
            'cache_timeout': 3600
        }
    
    def _initialize_registry(self):
        """初始化注册表"""
        # 创建必要的目录
        os.makedirs(self.registry_path, exist_ok=True)
        os.makedirs(os.path.join(self.registry_path, 'core'), exist_ok=True)
        os.makedirs(os.path.join(self.registry_path, 'specialized'), exist_ok=True)
        os.makedirs(os.path.join(self.registry_path, 'config'), exist_ok=True)
        
        # 加载已注册的智能体
        if self.config.get('auto_discovery', True):
            self.discover_agents()
        
        # 加载缓存
        self._load_cache()
        
        logger.info(f"智能体注册表初始化完成，已注册 {len(self.agents)} 个智能体")
    
    def discover_agents(self):
        """发现并注册智能体"""
        logger.info("开始发现智能体...")
        
        # 遍历所有智能体目录
        for root, dirs, files in os.walk(self.registry_path):
            # 跳过配置目录
            if 'config' in root:
                continue
                
            for file in files:
                if file.endswith('.md') and not file.startswith('.'):
                    agent_path = os.path.join(root, file)
                    self._register_agent_from_file(agent_path)
        
        # 构建索引
        self._build_indexes()
        
        logger.info(f"智能体发现完成，共发现 {len(self.agents)} 个智能体")
    
    def _register_agent_from_file(self, file_path: str):
        """从文件注册智能体"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析智能体信息
            agent_info = self._parse_agent_file(content, file_path)
            if agent_info:
                agent_id = agent_info['id']
                
                # 创建智能体元数据
                metadata = AgentMetadata(
                    id=agent_id,
                    name=agent_info['name'],
                    version=agent_info.get('version', '1.0.0'),
                    description=agent_info.get('description', ''),
                    author=agent_info.get('author', 'Unknown'),
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    agent_type=AgentType(agent_info.get('type', 'specialized')),
                    status=AgentStatus.INACTIVE,
                    capabilities=self._parse_capabilities(agent_info),
                    dependencies=agent_info.get('dependencies', []),
                    tags=agent_info.get('tags', []),
                    file_path=file_path,
                    class_name=agent_info.get('class_name', ''),
                    config=agent_info.get('config', {})
                )
                
                # 注册智能体
                self.agents[agent_id] = metadata
                
                logger.debug(f"注册智能体: {agent_id}")
                
        except Exception as e:
            logger.error(f"注册智能体失败 {file_path}: {e}")
    
    def _parse_agent_file(self, content: str, file_path: str) -> Optional[Dict[str, Any]]:
        """解析智能体文件"""
        # 简化的解析逻辑
        lines = content.split('\n')
        info = {}
        
        # 提取基本信息
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                info['name'] = line[2:].strip()
            elif line.startswith('- **角色**:'):
                info['description'] = line.split(':')[-1].strip()
            elif line.startswith('- **能力级别**:'):
                info['level'] = line.split(':')[-1].strip()
            elif line.startswith('- **专业领域**:'):
                info['domain'] = line.split(':')[-1].strip()
        
        # 生成ID
        if 'name' in info:
            info['id'] = self._generate_agent_id(info['name'], file_path)
            
            # 确定类型
            if 'core' in file_path.lower():
                info['type'] = 'core'
            elif 'specialized' in file_path.lower():
                info['type'] = 'specialized'
            else:
                info['type'] = 'custom'
            
            return info
        
        return None
    
    def _parse_capabilities(self, agent_info: Dict[str, Any]) -> List[AgentCapability]:
        """解析智能体能力"""
        capabilities = []
        
        # 基于级别创建能力
        level_str = agent_info.get('level', '中级')
        level_map = {
            '初级': CapabilityLevel.BEGINNER,
            '中级': CapabilityLevel.INTERMEDIATE,
            '高级': CapabilityLevel.ADVANCED,
            '专家': CapabilityLevel.EXPERT,
            '大师': CapabilityLevel.MASTER,
            '宗师': CapabilityLevel.GRANDMASTER
        }
        
        level = level_map.get(level_str, CapabilityLevel.INTERMEDIATE)
        
        # 创建主要能力
        main_capability = AgentCapability(
            name=agent_info.get('domain', '通用'),
            description=agent_info.get('description', ''),
            level=level,
            experience_years=float(level.value * 2),
            success_rate=0.85 + (level.value * 0.02),
            confidence=0.8 + (level.value * 0.03),
            tools=[],
            certifications=[],
            specializations=[],
            performance_metrics={},
            last_updated=datetime.now()
        )
        
        capabilities.append(main_capability)
        return capabilities
    
    def _generate_agent_id(self, name: str, file_path: str) -> str:
        """生成智能体ID"""
        # 基于名称和路径生成唯一ID
        content = f"{name}_{file_path}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _build_indexes(self):
        """构建索引"""
        # 清空索引
        self.capability_index.clear()
        self.type_index.clear()
        self.tag_index.clear()
        self.dependency_graph.clear()
        self.reverse_dependency_graph.clear()
        
        # 构建新索引
        for agent_id, metadata in self.agents.items():
            # 类型索引
            self.type_index[metadata.agent_type].append(agent_id)
            
            # 标签索引
            for tag in metadata.tags:
                self.tag_index[tag].append(agent_id)
            
            # 能力索引
            for capability in metadata.capabilities:
                self.capability_index[capability.name].append(agent_id)
            
            # 依赖图
            for dep in metadata.dependencies:
                self.dependency_graph[agent_id].append(dep)
                self.reverse_dependency_graph[dep].append(agent_id)
    
    def register_agent(self, metadata: AgentMetadata) -> bool:
        """注册智能体"""
        try:
            # 验证智能体
            if not self._validate_agent(metadata):
                return False
            
            # 检查依赖
            if not self._check_dependencies(metadata):
                logger.warning(f"智能体 {metadata.id} 依赖不满足")
                return False
            
            # 注册智能体
            self.agents[metadata.id] = metadata
            
            # 更新索引
            self._update_indexes(metadata)
            
            # 保存缓存
            self._save_cache()
            
            logger.info(f"智能体注册成功: {metadata.id}")
            return True
            
        except Exception as e:
            logger.error(f"注册智能体失败: {e}")
            return False
    
    def unregister_agent(self, agent_id: str) -> bool:
        """注销智能体"""
        try:
            if agent_id not in self.agents:
                return False
            
            # 检查依赖
            dependents = self.reverse_dependency_graph.get(agent_id, [])
            if dependents:
                logger.warning(f"智能体 {agent_id} 被其他智能体依赖: {dependents}")
                return False
            
            # 停用智能体实例
            if agent_id in self.agent_instances:
                self.deactivate_agent(agent_id)
            
            # 移除注册
            metadata = self.agents.pop(agent_id)
            
            # 更新索引
            self._remove_from_indexes(metadata)
            
            # 保存缓存
            self._save_cache()
            
            logger.info(f"智能体注销成功: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"注销智能体失败: {e}")
            return False
    
    def activate_agent(self, agent_id: str, config: Dict[str, Any] = None) -> bool:
        """激活智能体"""
        try:
            if agent_id not in self.agents:
                logger.error(f"智能体不存在: {agent_id}")
                return False
            
            metadata = self.agents[agent_id]
            
            # 检查状态
            if metadata.status == AgentStatus.ACTIVE:
                logger.warning(f"智能体已激活: {agent_id}")
                return True
            
            # 加载智能体类
            agent_class = self._load_agent_class(metadata)
            if not agent_class:
                logger.error(f"无法加载智能体类: {agent_id}")
                return False
            
            # 创建实例
            instance = agent_class(config or metadata.config)
            
            # 注册实例
            self.agent_instances[agent_id] = instance
            
            # 更新状态
            metadata.status = AgentStatus.ACTIVE
            metadata.updated_at = datetime.now()
            
            logger.info(f"智能体激活成功: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"激活智能体失败: {e}")
            if agent_id in self.agents:
                self.agents[agent_id].status = AgentStatus.ERROR
            return False
    
    def deactivate_agent(self, agent_id: str) -> bool:
        """停用智能体"""
        try:
            if agent_id not in self.agents:
                return False
            
            # 移除实例
            if agent_id in self.agent_instances:
                instance = self.agent_instances.pop(agent_id)
                # 清理资源
                if hasattr(instance, 'cleanup'):
                    instance.cleanup()
            
            # 更新状态
            self.agents[agent_id].status = AgentStatus.INACTIVE
            self.agents[agent_id].updated_at = datetime.now()
            
            logger.info(f"智能体停用成功: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"停用智能体失败: {e}")
            return False
    
    def get_agent(self, agent_id: str) -> Optional[AgentMetadata]:
        """获取智能体元数据"""
        return self.agents.get(agent_id)
    
    def get_agent_instance(self, agent_id: str) -> Optional[Any]:
        """获取智能体实例"""
        return self.agent_instances.get(agent_id)
    
    def find_agents(self, 
                   capability: Optional[str] = None,
                   agent_type: Optional[AgentType] = None,
                   tags: Optional[List[str]] = None,
                   status: Optional[AgentStatus] = None,
                   limit: int = 10) -> List[AgentMetadata]:
        """查找智能体"""
        candidates = list(self.agents.values())
        
        # 能力过滤
        if capability:
            capability_agents = self.capability_index.get(capability, [])
            candidates = [a for a in candidates if a.id in capability_agents]
        
        # 类型过滤
        if agent_type:
            type_agents = self.type_index.get(agent_type, [])
            candidates = [a for a in candidates if a.id in type_agents]
        
        # 标签过滤
        if tags:
            tag_agents = set()
            for tag in tags:
                tag_agents.update(self.tag_index.get(tag, []))
            candidates = [a for a in candidates if a.id in tag_agents]
        
        # 状态过滤
        if status:
            candidates = [a for a in candidates if a.status == status]
        
        # 限制数量
        return candidates[:limit]
    
    def recommend_agents(self, task_description: str, 
                        limit: int = 5) -> List[tuple[str, float]]:
        """推荐智能体"""
        # 提取关键词
        keywords = self._extract_keywords(task_description)
        
        # 计算匹配分数
        scores = []
        for agent_id, metadata in self.agents.items():
            score = self._calculate_match_score(metadata, keywords)
            if score > 0:
                scores.append((agent_id, score))
        
        # 排序并返回
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简化的关键词提取
        import re
        words = re.findall(r'\w+', text.lower())
        # 过滤停用词
        stop_words = {'的', '是', '在', '和', '与', '或', '但', '如果', '那么', 'the', 'is', 'at', 'which', 'on'}
        keywords = [w for w in words if len(w) > 2 and w not in stop_words]
        return keywords
    
    def _calculate_match_score(self, metadata: AgentMetadata, keywords: List[str]) -> float:
        """计算匹配分数"""
        score = 0.0
        
        # 名称匹配
        name_words = metadata.name.lower().split()
        name_score = sum(1 for kw in keywords if kw in name_words)
        score += name_score * 0.3
        
        # 描述匹配
        desc_words = metadata.description.lower().split()
        desc_score = sum(1 for kw in keywords if kw in desc_words)
        score += desc_score * 0.2
        
        # 能力匹配
        cap_score = 0
        for capability in metadata.capabilities:
            cap_words = capability.name.lower().split() + capability.description.lower().split()
            cap_score += sum(1 for kw in keywords if kw in cap_words)
        score += cap_score * 0.3
        
        # 标签匹配
        tag_score = sum(1 for kw in keywords if any(kw in tag.lower() for tag in metadata.tags))
        score += tag_score * 0.2
        
        # 归一化
        max_score = len(keywords) * 1.0
        return min(score / max_score, 1.0) if max_score > 0 else 0.0
    
    def _validate_agent(self, metadata: AgentMetadata) -> bool:
        """验证智能体"""
        # 基本验证
        if not metadata.id or not metadata.name:
            return False
        
        # 检查文件是否存在
        if not os.path.exists(metadata.file_path):
            return False
        
        return True
    
    def _check_dependencies(self, metadata: AgentMetadata) -> bool:
        """检查依赖"""
        for dep in metadata.dependencies:
            if dep not in self.agents:
                return False
        return True
    
    def _load_agent_class(self, metadata: AgentMetadata) -> Optional[Type]:
        """加载智能体类"""
        try:
            # 简化实现 - 实际应该从Python文件加载类
            # 这里返回一个通用的智能体类
            class GenericAgent:
                def __init__(self, config):
                    self.config = config
                
                def execute(self, task):
                    return f"执行任务: {task}"
                
                def cleanup(self):
                    pass
            
            return GenericAgent
            
        except Exception as e:
            logger.error(f"加载智能体类失败: {e}")
            return None
    
    def _update_indexes(self, metadata: AgentMetadata):
        """更新索引"""
        # 类型索引
        self.type_index[metadata.agent_type].append(metadata.id)
        
        # 标签索引
        for tag in metadata.tags:
            self.tag_index[tag].append(metadata.id)
        
        # 能力索引
        for capability in metadata.capabilities:
            self.capability_index[capability.name].append(metadata.id)
        
        # 依赖图
        for dep in metadata.dependencies:
            self.dependency_graph[metadata.id].append(dep)
            self.reverse_dependency_graph[dep].append(metadata.id)
    
    def _remove_from_indexes(self, metadata: AgentMetadata):
        """从索引中移除"""
        # 类型索引
        if metadata.id in self.type_index[metadata.agent_type]:
            self.type_index[metadata.agent_type].remove(metadata.id)
        
        # 标签索引
        for tag in metadata.tags:
            if metadata.id in self.tag_index[tag]:
                self.tag_index[tag].remove(metadata.id)
        
        # 能力索引
        for capability in metadata.capabilities:
            if metadata.id in self.capability_index[capability.name]:
                self.capability_index[capability.name].remove(metadata.id)
        
        # 依赖图
        if metadata.id in self.dependency_graph:
            del self.dependency_graph[metadata.id]
        
        for dep in metadata.dependencies:
            if metadata.id in self.reverse_dependency_graph.get(dep, []):
                self.reverse_dependency_graph[dep].remove(metadata.id)
    
    def _load_cache(self):
        """加载缓存"""
        cache_path = os.path.join(self.registry_path, 'config', 'registry_cache.pkl')
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.agents = cache_data.get('agents', {})
                    self.performance_history = cache_data.get('performance_history', defaultdict(list))
                    self.usage_stats = cache_data.get('usage_stats', defaultdict(dict))
            except Exception as e:
                logger.error(f"加载缓存失败: {e}")
    
    def _save_cache(self):
        """保存缓存"""
        cache_path = os.path.join(self.registry_path, 'config', 'registry_cache.pkl')
        try:
            cache_data = {
                'agents': self.agents,
                'performance_history': self.performance_history,
                'usage_stats': self.usage_stats
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """获取注册表统计信息"""
        stats = {
            'total_agents': len(self.agents),
            'active_agents': sum(1 for a in self.agents.values() if a.status == AgentStatus.ACTIVE),
            'agent_types': {t.value: len(agents) for t, agents in self.type_index.items()},
            'total_capabilities': len(self.capability_index),
            'dependency_graph_size': len(self.dependency_graph),
            'last_updated': datetime.now().isoformat()
        }
        return stats
    
    def export_registry(self, file_path: str):
        """导出注册表"""
        export_data = {
            'metadata': {
                'exported_at': datetime.now().isoformat(),
                'version': '5.0.0',
                'total_agents': len(self.agents)
            },
            'agents': {agent_id: metadata.to_dict() for agent_id, metadata in self.agents.items()},
            'indexes': {
                'capability_index': dict(self.capability_index),
                'type_index': {t.value: agents for t, agents in self.type_index.items()},
                'tag_index': dict(self.tag_index)
            }
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"注册表导出成功: {file_path}")

# 全局注册表实例
registry = AgentRegistry()

# 便捷函数
def get_agent(agent_id: str) -> Optional[AgentMetadata]:
    """获取智能体"""
    return registry.get_agent(agent_id)

def find_agents(**kwargs) -> List[AgentMetadata]:
    """查找智能体"""
    return registry.find_agents(**kwargs)

def recommend_agents(task: str, limit: int = 5) -> List[tuple[str, float]]:
    """推荐智能体"""
    return registry.recommend_agents(task, limit)

def activate_agent(agent_id: str, config: Dict[str, Any] = None) -> bool:
    """激活智能体"""
    return registry.activate_agent(agent_id, config)

def deactivate_agent(agent_id: str) -> bool:
    """停用智能体"""
    return registry.deactivate_agent(agent_id)

if __name__ == "__main__":
    # 测试代码
    print("智能体注册表系统")
    print("=" * 50)
    
    # 显示统计信息
    stats = registry.get_registry_stats()
    print(f"总智能体数: {stats['total_agents']}")
    print(f"激活智能体数: {stats['active_agents']}")
    print(f"智能体类型分布: {stats['agent_types']}")
    
    # 推荐示例
    print("\n推荐智能体示例:")
    recommendations = recommend_agents("Python开发", limit=3)
    for agent_id, score in recommendations:
        agent = registry.get_agent(agent_id)
        print(f"- {agent.name} (分数: {score:.2f})")