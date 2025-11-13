#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强上下文理解模块 - 深度语义分析和意图识别
Enhanced Context Understanding - Deep Semantic Analysis and Intent Recognition

作者: Quantum AI Team
版本: 5.2.0
日期: 2025-11-12
"""

import re
import json
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import numpy as np
from collections import defaultdict, Counter

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentType(Enum):
    """意图类型枚举"""
    CREATE = "create"          # 创建/生成
    ANALYZE = "analyze"        # 分析/检查
    OPTIMIZE = "optimize"      # 优化/改进
    DEBUG = "debug"            # 调试/修复
    LEARN = "learn"            # 学习/理解
    DEPLOY = "deploy"          # 部署/发布
    REFACTOR = "refactor"      # 重构/整理
    TEST = "test"              # 测试/验证
    DOCUMENT = "document"      # 文档/说明
    RESEARCH = "research"      # 研究/探索
    DESIGN = "design"          # 设计/规划
    INTEGRATE = "integrate"    # 集成/连接
    MONITOR = "monitor"        # 监控/观察

class ComplexityLevel(Enum):
    """复杂度级别"""
    TRIVIAL = "trivial"        # 微不足道
    SIMPLE = "simple"          # 简单
    MODERATE = "moderate"      # 中等
    COMPLEX = "complex"        # 复杂
    CRITICAL = "critical"      # 关键/复杂

class UrgencyLevel(Enum):
    """紧急程度级别"""
    LOW = "low"                # 低
    NORMAL = "normal"          # 正常
    HIGH = "high"              # 高
    CRITICAL = "critical"      # 紧急

@dataclass
class SemanticFeature:
    """语义特征"""
    keywords: List[str]
    entities: List[Dict[str, str]]
    concepts: List[str]
    relationships: List[Dict[str, Any]]
    sentiment: float  # -1 to 1
    formality: float  # 0 to 1
    specificity: float  # 0 to 1

@dataclass
class ContextualFeature:
    """上下文特征"""
    domain: str
    language: Optional[str]
    framework: Optional[str]
    environment: Optional[str]
    scale: str  # small, medium, large, enterprise
    stakeholders: List[str]

@dataclass
class TemporalFeature:
    """时间特征"""
    timeframe: str  # immediate, short, medium, long
    dependencies: List[str]
    sequence: List[str]
    constraints: List[str]

@dataclass
class EnhancedContext:
    """增强的上下文理解结果"""
    primary_intent: IntentType
    secondary_intents: List[IntentType]
    complexity: ComplexityLevel
    urgency: UrgencyLevel
    semantic_features: SemanticFeature
    contextual_features: ContextualFeature
    temporal_features: TemporalFeature
    confidence: float
    ambiguity_score: float
    suggested_actions: List[str]
    risk_factors: List[str]
    success_criteria: List[str]

class EnhancedContextUnderstanding:
    """增强上下文理解系统"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化增强上下文理解系统"""
        self.config_path = config_path or "context_understanding_config.json"
        self.context_history = []
        self.learning_enabled = True
        
        # 加载配置
        self._load_configuration()
        
        # 初始化知识库
        self._initialize_knowledge_base()
        
        logger.info("🧠 增强上下文理解系统初始化完成")
    
    def _load_configuration(self):
        """加载配置"""
        self.intent_patterns = self._load_intent_patterns()
        self.domain_keywords = self._load_domain_keywords()
        self.complexity_indicators = self._load_complexity_indicators()
        self.urgency_indicators = self._load_urgency_indicators()
    
    def _load_intent_patterns(self) -> Dict[IntentType, List[str]]:
        """加载意图模式"""
        return {
            IntentType.CREATE: [
                "创建", "生成", "实现", "开发", "编写", "构建", "制作", "设计", "建立",
                "新增", "添加", "产生", "产出", "创造", "打造", "编写", "编码"
            ],
            IntentType.ANALYZE: [
                "分析", "评估", "检查", "审查", "诊断", "研究", "调查", "探索",
                "查看", "理解", "解释", "说明", "解析", "审视", "考察"
            ],
            IntentType.OPTIMIZE: [
                "优化", "改进", "提升", "加速", "增强", "精简", "简化", "完善",
                "调优", "改善", "升级", "强化", "提高", "提升", "优化"
            ],
            IntentType.DEBUG: [
                "调试", "排错", "修复", "解决", "处理", "修正", "纠正", "消除",
                "排查", "定位", "找错", "除错", "纠错", "修复", "解决"
            ],
            IntentType.LEARN: [
                "学习", "了解", "掌握", "熟悉", "研究", "探索", "发现", "认识",
                "理解", "明白", "弄懂", "掌握", "学会", "体会", "领悟"
            ],
            IntentType.DEPLOY: [
                "部署", "发布", "上线", "运行", "启动", "执行", "实施", "落地",
                "投产", "发布", "部署", "运行", "启动", "实施"
            ],
            IntentType.REFACTOR: [
                "重构", "整理", "优化代码", "改进结构", "简化", "规范", "标准化",
                "重构", "重写", "重组", "调整", "优化", "改进"
            ],
            IntentType.TEST: [
                "测试", "验证", "检查", "确认", "验证", "检验", "测试", "试运行",
                "验证", "检查", "测试", "确认", "检验"
            ],
            IntentType.DOCUMENT: [
                "文档", "说明", "记录", "描述", "解释", "注释", "编写文档", "记录",
                "说明", "描述", "解释", "文档化", "记录"
            ],
            IntentType.RESEARCH: [
                "研究", "调研", "探索", "查找", "搜索", "调查", "分析", "考察",
                "研究", "调研", "探索", "查找", "搜索"
            ],
            IntentType.DESIGN: [
                "设计", "规划", "架构", "方案", "策略", "计划", "安排", "布局",
                "设计", "规划", "架构", "制定", "安排"
            ],
            IntentType.INTEGRATE: [
                "集成", "整合", "连接", "合并", "融合", "结合", "对接", "连通",
                "集成", "整合", "连接", "合并", "融合"
            ],
            IntentType.MONITOR: [
                "监控", "观察", "监视", "跟踪", "检测", "关注", "留意", "查看",
                "监控", "观察", "监视", "跟踪", "检测"
            ]
        }
    
    def _load_domain_keywords(self) -> Dict[str, List[str]]:
        """加载领域关键词"""
        return {
            "web": [
                "网站", "网页", "前端", "后端", "浏览器", "服务器", "API", "HTTP",
                "HTML", "CSS", "JavaScript", "React", "Vue", "Angular", "Node.js"
            ],
            "mobile": [
                "移动", "手机", "APP", "应用", "iOS", "Android", "React Native",
                "Flutter", "移动端", "触摸", "响应式", "移动应用"
            ],
            "ai": [
                "AI", "人工智能", "机器学习", "深度学习", "神经网络", "模型",
                "训练", "预测", "分类", "回归", "聚类", "强化学习"
            ],
            "data": [
                "数据", "数据库", "大数据", "数据科学", "分析", "处理", "存储",
                "SQL", "NoSQL", "数据仓库", "ETL", "数据管道"
            ],
            "system": [
                "系统", "架构", "分布式", "微服务", "云原生", "容器", "Kubernetes",
                "Docker", "DevOps", "运维", "基础设施", "平台"
            ],
            "security": [
                "安全", "加密", "认证", "授权", "防火墙", "漏洞", "攻击", "防护",
                "密码学", "网络安全", "信息安全", "数据保护"
            ],
            "performance": [
                "性能", "优化", "速度", "延迟", "吞吐量", "并发", "缓存", "负载",
                "扩展性", "可伸缩性", "性能调优", "瓶颈"
            ],
            "testing": [
                "测试", "单元测试", "集成测试", "端到端测试", "自动化测试",
                "质量保证", "测试驱动", "持续集成", "测试覆盖率"
            ]
        }
    
    def _load_complexity_indicators(self) -> Dict[ComplexityLevel, List[str]]:
        """加载复杂度指标"""
        return {
            ComplexityLevel.TRIVIAL: [
                "简单", "容易", "基础", "基本", "入门", "示例", "演示", "练习",
                "快速", "马上", "立即", "简单", "容易"
            ],
            ComplexityLevel.SIMPLE: [
                "标准", "常规", "普通", "一般", "日常", "常见", "标准", "典型",
                "中等", "一般", "常规", "标准"
            ],
            ComplexityLevel.MODERATE: [
                "中等", "适中", "合理", "适当", "需要", "应该", "考虑", "规划",
                "中等", "适中", "需要", "考虑"
            ],
            ComplexityLevel.COMPLEX: [
                "复杂", "困难", "挑战", "高级", "深度", "详细", "全面", "综合",
                "复杂", "困难", "挑战", "高级", "深度"
            ],
            ComplexityLevel.CRITICAL: [
                "关键", "重要", "核心", "紧急", "严重", "重大", "关键路径", "核心",
                "关键", "重要", "核心", "紧急", "严重"
            ]
        }
    
    def _load_urgency_indicators(self) -> Dict[UrgencyLevel, List[str]]:
        """加载紧急度指标"""
        return {
            UrgencyLevel.LOW: [
                "可以", "建议", "可选", "稍后", "有空", "不急", "空闲", "方便时",
                "可以", "建议", "可选", "稍后"
            ],
            UrgencyLevel.NORMAL: [
                "需要", "应该", "要", "计划", "安排", "准备", "考虑", "处理",
                "需要", "应该", "要", "计划"
            ],
            UrgencyLevel.HIGH: [
                "尽快", "优先", "重要", "紧急", "急需", "立即", "马上", "赶快",
                "尽快", "优先", "重要", "紧急"
            ],
            UrgencyLevel.CRITICAL: [
                "紧急", "立即", "马上", "严重", "关键", "重要", "急需", "刻不容缓",
                "紧急", "立即", "马上", "严重", "关键"
            ]
        }
    
    def _initialize_knowledge_base(self):
        """初始化知识库"""
        self.entity_patterns = {
            "technology": [
                r"(Python|Java|JavaScript|TypeScript|Go|Rust|C\+\+|C#|PHP|Ruby|Swift|Kotlin)",
                r"(React|Vue|Angular|Node\.js|Django|Flask|Spring|Express\.js|Laravel)",
                r"(MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch|Cassandra|Oracle)",
                r"(Docker|Kubernetes|Jenkins|Git|AWS|Azure|GCP|Terraform)"
            ],
            "file": [
                r"([a-zA-Z0-9_\-]+\.(py|js|ts|java|cpp|c|h|css|html|json|yaml|yml|md|sql))",
                r"([a-zA-Z0-9_\-\/]+\/[a-zA-Z0-9_\-]+\.(py|js|ts|java|cpp|c|h|css|html|json|yaml|yml|md|sql))"
            ],
            "metric": [
                r"(\d+(?:\.\d+)?)\s*(?:%|ms|s|MB|GB|TB|KB|bytes|requests/sec|QPS|TPS)",
                r"(性能|速度|延迟|吞吐量|内存|CPU|磁盘|网络)\s*[:：]\s*(\d+(?:\.\d+)?)"
            ],
            "version": [
                r"v?(\d+(?:\.\d+)*(?:\.[a-zA-Z0-9]+)?)",
                r"(version|ver)\s*[:：]\s*(\d+(?:\.\d+)*)"
            ]
        }
        
        self.concept_patterns = {
            "architecture": [
                "架构", "设计", "模式", "结构", "框架", "体系", "组件", "模块"
            ],
            "performance": [
                "性能", "速度", "效率", "优化", "延迟", "吞吐量", "并发", "扩展"
            ],
            "security": [
                "安全", "加密", "认证", "授权", "防护", "漏洞", "风险", "威胁"
            ],
            "quality": [
                "质量", "测试", "验证", "检查", "标准", "规范", "最佳实践", "可靠"
            ]
        }
    
    def understand_context(self, text: str, conversation_history: List[str] = None) -> EnhancedContext:
        """深度理解上下文"""
        logger.info(f"🧠 深度理解上下文: {text[:50]}...")
        
        start_time = time.time()
        
        # 1. 语义特征提取
        semantic_features = self._extract_semantic_features(text)
        
        # 2. 上下文特征提取
        contextual_features = self._extract_contextual_features(text, semantic_features)
        
        # 3. 时间特征提取
        temporal_features = self._extract_temporal_features(text, conversation_history)
        
        # 4. 意图识别
        primary_intent, secondary_intents = self._identify_intents(text, semantic_features)
        
        # 5. 复杂度评估
        complexity = self._assess_complexity(text, semantic_features, contextual_features)
        
        # 6. 紧急度评估
        urgency = self._assess_urgency(text, semantic_features, temporal_features)
        
        # 7. 置信度计算
        confidence = self._calculate_confidence(
            semantic_features, contextual_features, temporal_features,
            primary_intent, complexity, urgency
        )
        
        # 8. 歧义度评估
        ambiguity_score = self._assess_ambiguity(text, semantic_features, secondary_intents)
        
        # 9. 建议行动生成
        suggested_actions = self._generate_suggested_actions(
            primary_intent, complexity, urgency, contextual_features
        )
        
        # 10. 风险因素识别
        risk_factors = self._identify_risk_factors(text, complexity, contextual_features)
        
        # 11. 成功标准定义
        success_criteria = self._define_success_criteria(primary_intent, complexity, contextual_features)
        
        # 构建增强上下文
        enhanced_context = EnhancedContext(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            complexity=complexity,
            urgency=urgency,
            semantic_features=semantic_features,
            contextual_features=contextual_features,
            temporal_features=temporal_features,
            confidence=confidence,
            ambiguity_score=ambiguity_score,
            suggested_actions=suggested_actions,
            risk_factors=risk_factors,
            success_criteria=success_criteria
        )
        
        # 存储历史
        self.context_history.append({
            'timestamp': time.time(),
            'text': text,
            'context': enhanced_context,
            'processing_time': time.time() - start_time
        })
        
        logger.info(f"✅ 上下文理解完成: {primary_intent.value} - 置信度: {confidence:.2f}")
        return enhanced_context
    
    def _extract_semantic_features(self, text: str) -> SemanticFeature:
        """提取语义特征"""
        # 关键词提取
        keywords = self._extract_keywords(text)
        
        # 实体识别
        entities = self._extract_entities(text)
        
        # 概念识别
        concepts = self._extract_concepts(text)
        
        # 关系识别
        relationships = self._extract_relationships(text)
        
        # 情感分析
        sentiment = self._analyze_sentiment(text)
        
        # 正式度分析
        formality = self._analyze_formality(text)
        
        # 具体性分析
        specificity = self._analyze_specificity(text)
        
        return SemanticFeature(
            keywords=keywords,
            entities=entities,
            concepts=concepts,
            relationships=relationships,
            sentiment=sentiment,
            formality=formality,
            specificity=specificity
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 技术关键词
        tech_keywords = set()
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    tech_keywords.add(keyword)
        
        # 动作词
        action_words = set()
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    action_words.add(pattern)
        
        # 数量和度量词
        quantity_words = re.findall(r'\d+(?:\.\d+)?(?:%|ms|s|MB|GB|TB|KB|bytes|requests/sec|QPS|TPS)', text)
        
        # 合并并去重
        all_keywords = list(tech_keywords | action_words | set(quantity_words))
        
        return sorted(all_keywords, key=len, reverse=True)[:20]  # 返回前20个最重要的关键词
    
    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """提取实体"""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group(1) if match.groups() else match.group(0)
                    entities.append({
                        'type': entity_type,
                        'text': entity_text,
                        'position': match.span()
                    })
        
        return entities
    
    def _extract_concepts(self, text: str) -> List[str]:
        """提取概念"""
        concepts = []
        text_lower = text.lower()
        
        for concept_type, concept_words in self.concept_patterns.items():
            for word in concept_words:
                if word in text_lower:
                    concepts.append(concept_type)
                    break  # 每个概念类型只添加一次
        
        return list(set(concepts))
    
    def _extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """提取关系"""
        relationships = []
        
        # 因果关系
        causal_patterns = [
            r'因为(.+?)，所以(.+?)',
            r'由于(.+?)，(.+?)',
            r'(.+?)导致(.+?)',
            r'(.+?)引起(.+?)'
        ]
        
        for pattern in causal_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                relationships.append({
                    'type': 'causal',
                    'source': match.group(1).strip(),
                    'target': match.group(2).strip() if len(match.groups()) > 1 else None
                })
        
        # 条件关系
        conditional_patterns = [
            r'如果(.+?)，(.+?)',
            r'当(.+?)时，(.+?)',
            r'(.+?)的话，(.+?)'
        ]
        
        for pattern in conditional_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                relationships.append({
                    'type': 'conditional',
                    'condition': match.group(1).strip(),
                    'consequence': match.group(2).strip() if len(match.groups()) > 1 else None
                })
        
        return relationships
    
    def _analyze_sentiment(self, text: str) -> float:
        """分析情感"""
        positive_words = ['好', '棒', '优秀', '完美', '成功', '高效', '快速', '稳定', '满意']
        negative_words = ['错', '坏', '失败', '慢', '问题', '错误', '困难', '复杂', '紧急']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.0  # 中性
        
        return (positive_count - negative_count) / total_sentiment_words
    
    def _analyze_formality(self, text: str) -> float:
        """分析正式度"""
        formal_indicators = ['请', '您', '贵', "非常", "十分", "特别", "感谢", "麻烦", "不好意思"]
        informal_indicators = ['哈', '嘿', '嗯', '哦', '啊', '吧', '嘛', '啦', '呢']
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in text)
        informal_count = sum(1 for indicator in informal_indicators if indicator in text)
        
        total_indicators = formal_count + informal_count
        if total_indicators == 0:
            return 0.5  # 中等正式度
        
        return formal_count / total_indicators
    
    def _analyze_specificity(self, text: str) -> float:
        """分析具体性"""
        # 具体性指标：数字、专有名词、技术术语、文件路径等
        specificity_indicators = [
            r'\d+(?:\.\d+)?',  # 数字
            r'[A-Z][a-zA-Z]+',  # 专有名词
            r'[a-zA-Z0-9_\-]+\.[a-zA-Z]+',  # 文件扩展名
            r'[a-zA-Z0-9_\-\/]+\.[a-zA-Z0-9_\-\/]+',  # 文件路径
            r'https?://[^\s]+',  # URL
        ]
        
        specificity_score = 0
        text_length = len(text)
        
        for pattern in specificity_indicators:
            matches = re.findall(pattern, text)
            specificity_score += len(matches) * 0.1
        
        # 标准化到0-1范围
        return min(1.0, specificity_score / (text_length / 10))
    
    def _extract_contextual_features(self, text: str, semantic_features: SemanticFeature) -> ContextualFeature:
        """提取上下文特征"""
        # 领域识别
        domain = self._identify_domain(text, semantic_features)
        
        # 语言识别
        language = self._identify_language(text, semantic_features)
        
        # 框架识别
        framework = self._identify_framework(text, semantic_features)
        
        # 环境识别
        environment = self._identify_environment(text, semantic_features)
        
        # 规模识别
        scale = self._identify_scale(text, semantic_features)
        
        # 利益相关者识别
        stakeholders = self._identify_stakeholders(text, semantic_features)
        
        return ContextualFeature(
            domain=domain,
            language=language,
            framework=framework,
            environment=environment,
            scale=scale,
            stakeholders=stakeholders
        )
    
    def _identify_domain(self, text: str, semantic_features: SemanticFeature) -> str:
        """识别领域"""
        domain_scores = {}
        text_lower = text.lower()
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score
        
        # 考虑实体中的技术栈
        tech_entities = [e for e in semantic_features.entities if e['type'] == 'technology']
        for entity in tech_entities:
            entity_text = entity['text'].lower()
            for domain, keywords in self.domain_keywords.items():
                if any(keyword in entity_text for keyword in keywords):
                    domain_scores[domain] = domain_scores.get(domain, 0) + 2
        
        if not domain_scores or max(domain_scores.values()) == 0:
            return 'general'
        
        return max(domain_scores, key=domain_scores.get)
    
    def _identify_language(self, text: str, semantic_features: SemanticFeature) -> Optional[str]:
        """识别编程语言"""
        language_patterns = {
            'python': ['python', 'py', '.py', 'django', 'flask', 'pandas', 'numpy'],
            'javascript': ['javascript', 'js', '.js', 'node', 'nodejs', 'npm', 'react', 'vue', 'angular'],
            'typescript': ['typescript', 'ts', '.ts'],
            'java': ['java', '.java', 'spring', 'maven', 'gradle'],
            'go': ['go', '.go', 'golang'],
            'rust': ['rust', '.rs', 'cargo'],
            'cpp': ['cpp', 'c++', '.cpp', 'gcc', 'clang'],
            'c': ['c', '.c'],
            'html': ['html', '.html', 'css', 'web'],
            'sql': ['sql', 'database', 'mysql', 'postgresql', 'oracle'],
            'bash': ['bash', 'shell', 'sh', 'linux', 'unix']
        }
        
        text_lower = text.lower()
        language_scores = {}
        
        for language, patterns in language_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            language_scores[language] = score
        
        if not language_scores or max(language_scores.values()) == 0:
            return None
        
        return max(language_scores, key=language_scores.get)
    
    def _identify_framework(self, text: str, semantic_features: SemanticFeature) -> Optional[str]:
        """识别框架"""
        framework_patterns = {
            'react': ['react', 'jsx', 'tsx', 'hooks', 'component'],
            'vue': ['vue', 'vuex', 'vue-router'],
            'angular': ['angular', 'typescript', 'rxjs'],
            'django': ['django', 'python', 'mvc'],
            'flask': ['flask', 'python', 'blueprint'],
            'spring': ['spring', 'java', 'boot', 'mvc'],
            'express': ['express', 'node', 'middleware'],
            'laravel': ['laravel', 'php', 'mvc', 'eloquent'],
            'tensorflow': ['tensorflow', 'tf', 'neural', 'ml'],
            'pytorch': ['pytorch', 'torch', 'neural', 'ml']
        }
        
        text_lower = text.lower()
        framework_scores = {}
        
        for framework, patterns in framework_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            framework_scores[framework] = score
        
        if not framework_scores or max(framework_scores.values()) == 0:
            return None
        
        return max(framework_scores, key=framework_scores.get)
    
    def _identify_environment(self, text: str, semantic_features: SemanticFeature) -> Optional[str]:
        """识别环境"""
        environment_patterns = {
            'development': ['开发', 'dev', '本地', '测试', 'debug', '调试'],
            'staging': ['预发布', 'staging', 'uat', '验收', '测试环境'],
            'production': ['生产', 'prod', '线上', '正式', '发布'],
            'cloud': ['云', 'cloud', 'aws', 'azure', 'gcp', '阿里云', '腾讯云'],
            'container': ['容器', 'docker', 'kubernetes', 'k8s', 'pod'],
            'mobile': ['移动', '手机', 'app', 'ios', 'android']
        }
        
        text_lower = text.lower()
        environment_scores = {}
        
        for environment, patterns in environment_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            environment_scores[environment] = score
        
        if not environment_scores or max(environment_scores.values()) == 0:
            return None
        
        return max(environment_scores, key=environment_scores.get)
    
    def _identify_scale(self, text: str, semantic_features: SemanticFeature) -> str:
        """识别规模"""
        scale_indicators = {
            'small': ['小', '个人', '简单', '基础', '原型', 'demo', '示例', '练习'],
            'medium': ['中等', '团队', '标准', '常规', '企业', '商业'],
            'large': ['大', '大规模', '企业级', '复杂', '系统', '平台'],
            'enterprise': ['企业', '商业', '生产', '关键', '重要', '核心', '大型']
        }
        
        text_lower = text.lower()
        scale_scores = {}
        
        for scale, indicators in scale_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            scale_scores[scale] = score
        
        if not scale_scores or max(scale_scores.values()) == 0:
            return 'medium'
        
        return max(scale_scores, key=scale_scores.get)
    
    def _identify_stakeholders(self, text: str, semantic_features: SemanticFeature) -> List[str]:
        """识别利益相关者"""
        stakeholder_patterns = {
            'user': ['用户', '客户', '消费者', '访客', '使用者'],
            'developer': ['开发者', '程序员', '工程师', '开发团队'],
            'manager': ['经理', '主管', '领导', '管理层', '决策者'],
            'admin': ['管理员', '运维', '系统管理员', 'IT'],
            'business': ['业务', '产品', '市场', '销售', '运营']
        }
        
        text_lower = text.lower()
        stakeholders = []
        
        for stakeholder, patterns in stakeholder_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                stakeholders.append(stakeholder)
        
        return stakeholders
    
    def _extract_temporal_features(self, text: str, conversation_history: List[str] = None) -> TemporalFeature:
        """提取时间特征"""
        # 时间框架识别
        timeframe = self._identify_timeframe(text)
        
        # 依赖关系识别
        dependencies = self._identify_dependencies(text)
        
        # 序列关系识别
        sequence = self._identify_sequence(text)
        
        # 约束条件识别
        constraints = self._identify_constraints(text)
        
        return TemporalFeature(
            timeframe=timeframe,
            dependencies=dependencies,
            sequence=sequence,
            constraints=constraints
        )
    
    def _identify_timeframe(self, text: str) -> str:
        """识别时间框架"""
        timeframe_patterns = {
            'immediate': ['立即', '马上', '现在', '当前', '立刻', '即刻'],
            'short': ['很快', '短期内', '近期', '几天内', '本周内', '下周'],
            'medium': ['中期', '一个月', '几周', '季度内', '几个月'],
            'long': ['长期', '半年', '一年', '未来', '规划', '路线图']
        }
        
        text_lower = text.lower()
        timeframe_scores = {}
        
        for timeframe, patterns in timeframe_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            timeframe_scores[timeframe] = score
        
        if not timeframe_scores or max(timeframe_scores.values()) == 0:
            return 'medium'
        
        return max(timeframe_scores, key=timeframe_scores.get)
    
    def _identify_dependencies(self, text: str) -> List[str]:
        """识别依赖关系"""
        dependency_patterns = [
            r'需要(.+?)',
            r'依赖(.+?)',
            r'基于(.+?)',
            r'使用(.+?)',
            r'调用(.+?)',
            r'引用(.+?)',
            r'导入(.+?)'
        ]
        
        dependencies = []
        text_lower = text.lower()
        
        for pattern in dependency_patterns:
            matches = re.findall(pattern, text)
            dependencies.extend(matches)
        
        return list(set(dependencies))[:5]  # 返回前5个最重要的依赖
    
    def _identify_sequence(self, text: str) -> List[str]:
        """识别序列关系"""
        sequence_patterns = [
            r'首先(.+?)，?然后(.+?)',
            r'第一步(.+?)，?第二步(.+?)',
            r'先(.+?)，?再(.+?)',
            r'开始(.+?)，?接着(.+?)'
        ]
        
        sequence = []
        
        for pattern in sequence_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    sequence.extend(match)
                else:
                    sequence.append(match)
        
        return list(set(sequence))[:5]  # 返回前5个最重要的序列步骤
    
    def _identify_constraints(self, text: str) -> List[str]:
        """识别约束条件"""
        constraint_patterns = [
            r'限制(.+?)',
            r'约束(.+?)',
            r'要求(.+?)',
            r'必须(.+?)',
            r'不能(.+?)',
            r'禁止(.+?)',
            r'只允许(.+?)'
        ]
        
        constraints = []
        
        for pattern in constraint_patterns:
            matches = re.findall(pattern, text)
            constraints.extend(matches)
        
        return list(set(constraints))[:5]  # 返回前5个最重要的约束
    
    def _identify_intents(self, text: str, semantic_features: SemanticFeature) -> Tuple[IntentType, List[IntentType]]:
        """识别意图"""
        intent_scores = {}
        text_lower = text.lower()
        
        # 基于关键词匹配
        for intent_type, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            intent_scores[intent_type] = score
        
        # 基于语义特征增强
        if semantic_features.sentiment > 0.3:  # 积极情感
            intent_scores[IntentType.CREATE] = intent_scores.get(IntentType.CREATE, 0) + 1
            intent_scores[IntentType.OPTIMIZE] = intent_scores.get(IntentType.OPTIMIZE, 0) + 1
        
        if semantic_features.sentiment < -0.3:  # 消极情感
            intent_scores[IntentType.DEBUG] = intent_scores.get(IntentType.DEBUG, 0) + 2
            intent_scores[IntentType.ANALYZE] = intent_scores.get(IntentType.ANALYZE, 0) + 1
        
        # 基于概念增强
        if 'architecture' in semantic_features.concepts:
            intent_scores[IntentType.DESIGN] = intent_scores.get(IntentType.DESIGN, 0) + 2
        
        if 'performance' in semantic_features.concepts:
            intent_scores[IntentType.OPTIMIZE] = intent_scores.get(IntentType.OPTIMIZE, 0) + 2
        
        if 'security' in semantic_features.concepts:
            intent_scores[IntentType.ANALYZE] = intent_scores.get(IntentType.ANALYZE, 0) + 1
            intent_scores[IntentType.TEST] = intent_scores.get(IntentType.TEST, 0) + 1
        
        if 'quality' in semantic_features.concepts:
            intent_scores[IntentType.TEST] = intent_scores.get(IntentType.TEST, 0) + 2
            intent_scores[IntentType.DOCUMENT] = intent_scores.get(IntentType.DOCUMENT, 0) + 1
        
        if not intent_scores or max(intent_scores.values()) == 0:
            return IntentType.ANALYZE, []
        
        # 排序获取主要意图和次要意图
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        primary_intent = sorted_intents[0][0]
        secondary_intents = [intent for intent, score in sorted_intents[1:4] if score > 0]
        
        return primary_intent, secondary_intents
    
    def _assess_complexity(self, text: str, semantic_features: SemanticFeature, contextual_features: ContextualFeature) -> ComplexityLevel:
        """评估复杂度"""
        complexity_scores = {}
        text_lower = text.lower()
        
        # 基于关键词
        for complexity, indicators in self.complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            complexity_scores[complexity] = score
        
        # 基于语义特征
        if semantic_features.specificity > 0.7:
            complexity_scores[ComplexityLevel.COMPLEX] = complexity_scores.get(ComplexityLevel.COMPLEX, 0) + 1
        
        if len(semantic_features.relationships) > 3:
            complexity_scores[ComplexityLevel.COMPLEX] = complexity_scores.get(ComplexityLevel.COMPLEX, 0) + 1
        
        if len(semantic_features.entities) > 5:
            complexity_scores[ComplexityLevel.MODERATE] = complexity_scores.get(ComplexityLevel.MODERATE, 0) + 1
        
        # 基于上下文特征
        if contextual_features.scale == 'enterprise':
            complexity_scores[ComplexityLevel.CRITICAL] = complexity_scores.get(ComplexityLevel.CRITICAL, 0) + 2
        elif contextual_features.scale == 'large':
            complexity_scores[ComplexityLevel.COMPLEX] = complexity_scores.get(ComplexityLevel.COMPLEX, 0) + 1
        
        if contextual_features.environment == 'production':
            complexity_scores[ComplexityLevel.CRITICAL] = complexity_scores.get(ComplexityLevel.CRITICAL, 0) + 1
        
        if len(contextual_features.stakeholders) > 2:
            complexity_scores[ComplexityLevel.MODERATE] = complexity_scores.get(ComplexityLevel.MODERATE, 0) + 1
        
        if not complexity_scores or max(complexity_scores.values()) == 0:
            return ComplexityLevel.SIMPLE
        
        return max(complexity_scores, key=complexity_scores.get)
    
    def _assess_urgency(self, text: str, semantic_features: SemanticFeature, temporal_features: TemporalFeature) -> UrgencyLevel:
        """评估紧急度"""
        urgency_scores = {}
        text_lower = text.lower()
        
        # 基于关键词
        for urgency, indicators in self.urgency_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            urgency_scores[urgency] = score
        
        # 基于情感特征
        if semantic_features.sentiment < -0.5:  # 强烈负面情感
            urgency_scores[UrgencyLevel.HIGH] = urgency_scores.get(UrgencyLevel.HIGH, 0) + 2
        
        # 基于时间特征
        if temporal_features.timeframe == 'immediate':
            urgency_scores[UrgencyLevel.CRITICAL] = urgency_scores.get(UrgencyLevel.CRITICAL, 0) + 3
        elif temporal_features.timeframe == 'short':
            urgency_scores[UrgencyLevel.HIGH] = urgency_scores.get(UrgencyLevel.HIGH, 0) + 2
        
        if len(temporal_features.constraints) > 2:
            urgency_scores[UrgencyLevel.HIGH] = urgency_scores.get(UrgencyLevel.HIGH, 0) + 1
        
        if not urgency_scores or max(urgency_scores.values()) == 0:
            return UrgencyLevel.NORMAL
        
        return max(urgency_scores, key=urgency_scores.get)
    
    def _calculate_confidence(self, semantic_features: SemanticFeature, contextual_features: ContextualFeature,
                              temporal_features: TemporalFeature, primary_intent: IntentType,
                              complexity: ComplexityLevel, urgency: UrgencyLevel) -> float:
        """计算置信度"""
        confidence_factors = []
        
        # 语义特征置信度
        semantic_confidence = min(1.0, (len(semantic_features.keywords) + 
                                     len(semantic_features.entities) + 
                                     len(semantic_features.concepts)) / 10.0)
        confidence_factors.append(semantic_confidence)
        
        # 上下文特征置信度
        contextual_confidence = 0.5  # 基础分数
        if contextual_features.domain != 'general':
            contextual_confidence += 0.2
        if contextual_features.language:
            contextual_confidence += 0.1
        if contextual_features.framework:
            contextual_confidence += 0.1
        if contextual_features.environment:
            contextual_confidence += 0.1
        confidence_factors.append(min(1.0, contextual_confidence))
        
        # 时间特征置信度
        temporal_confidence = min(1.0, (len(temporal_features.dependencies) + 
                                     len(temporal_features.sequence) + 
                                     len(temporal_features.constraints)) / 5.0)
        confidence_factors.append(temporal_confidence)
        
        # 意图明确度
        intent_confidence = 0.7  # 基础分数
        if primary_intent not in [IntentType.ANALYZE]:  # 非默认意图
            intent_confidence += 0.2
        confidence_factors.append(intent_confidence)
        
        # 复杂度和紧急度一致性
        consistency_confidence = 0.8
        if complexity == ComplexityLevel.CRITICAL and urgency == UrgencyLevel.CRITICAL:
            consistency_confidence = 1.0
        elif complexity == ComplexityLevel.TRIVIAL and urgency == UrgencyLevel.LOW:
            consistency_confidence = 1.0
        confidence_factors.append(consistency_confidence)
        
        return np.mean(confidence_factors)
    
    def _assess_ambiguity(self, text: str, semantic_features: SemanticFeature, secondary_intents: List[IntentType]) -> float:
        """评估歧义度"""
        ambiguity_factors = []
        
        # 次要意图数量
        intent_ambiguity = min(1.0, len(secondary_intents) / 3.0)
        ambiguity_factors.append(intent_ambiguity)
        
        # 关键词模糊度
        keyword_ambiguity = 0.0
        if len(semantic_features.keywords) < 3:
            keyword_ambiguity = 0.8
        elif len(semantic_features.keywords) < 6:
            keyword_ambiguity = 0.4
        ambiguity_factors.append(keyword_ambiguity)
        
        # 实体识别模糊度
        entity_ambiguity = 0.0
        if len(semantic_features.entities) == 0:
            entity_ambiguity = 0.6
        elif len(semantic_features.entities) < 3:
            entity_ambiguity = 0.3
        ambiguity_factors.append(entity_ambiguity)
        
        # 文本长度模糊度
        text_length = len(text)
        if text_length < 20:
            length_ambiguity = 0.8
        elif text_length < 50:
            length_ambiguity = 0.4
        else:
            length_ambiguity = 0.1
        ambiguity_factors.append(length_ambiguity)
        
        return np.mean(ambiguity_factors)
    
    def _generate_suggested_actions(self, primary_intent: IntentType, complexity: ComplexityLevel,
                                  urgency: UrgencyLevel, contextual_features: ContextualFeature) -> List[str]:
        """生成建议行动"""
        actions = []
        
        # 基于意图的行动
        intent_actions = {
            IntentType.CREATE: [
                "制定详细的实现计划",
                "准备必要的开发环境",
                "设计系统架构",
                "编写核心功能代码"
            ],
            IntentType.ANALYZE: [
                "收集相关数据和文档",
                "执行深度分析",
                "生成分析报告",
                "提供改进建议"
            ],
            IntentType.OPTIMIZE: [
                "识别性能瓶颈",
                "制定优化策略",
                "实施优化方案",
                "验证优化效果"
            ],
            IntentType.DEBUG: [
                "重现问题现象",
                "分析错误日志",
                "定位根本原因",
                "实施修复方案"
            ],
            IntentType.LEARN: [
                "收集学习资料",
                "制定学习计划",
                "实践应用所学",
                "总结学习成果"
            ],
            IntentType.DEPLOY: [
                "准备部署环境",
                "配置部署参数",
                "执行部署流程",
                "验证部署结果"
            ],
            IntentType.REFACTOR: [
                "分析现有代码结构",
                "制定重构计划",
                "逐步重构代码",
                "测试重构结果"
            ],
            IntentType.TEST: [
                "设计测试用例",
                "编写测试代码",
                "执行测试验证",
                "生成测试报告"
            ],
            IntentType.DOCUMENT: [
                "整理文档结构",
                "编写技术文档",
                "审查文档质量",
                "发布文档"
            ],
            IntentType.RESEARCH: [
                "确定研究方向",
                "收集相关资料",
                "分析研究成果",
                "总结研究结论"
            ],
            IntentType.DESIGN: [
                "分析需求约束",
                "设计系统方案",
                "评估设计方案",
                "输出设计文档"
            ],
            IntentType.INTEGRATE: [
                "分析集成需求",
                "设计集成方案",
                "实施集成工作",
                "测试集成效果"
            ],
            IntentType.MONITOR: [
                "配置监控系统",
                "设置监控指标",
                "监控运行状态",
                "分析监控数据"
            ]
        }
        
        actions.extend(intent_actions.get(primary_intent, []))
        
        # 基于复杂度的行动调整
        if complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.CRITICAL]:
            actions.insert(0, "进行详细的需求分析")
            actions.insert(1, "制定项目实施计划")
            actions.append("进行风险评估和管理")
        
        # 基于紧急度的行动调整
        if urgency == UrgencyLevel.CRITICAL:
            actions.insert(0, "立即采取应急措施")
            actions.insert(1, "通知相关利益相关者")
        elif urgency == UrgencyLevel.HIGH:
            actions.insert(0, "优先处理关键任务")
        
        # 基于上下文的行动调整
        if contextual_features.environment == 'production':
            actions.append("确保生产环境稳定性")
            actions.append("准备回滚方案")
        
        if contextual_features.scale == 'enterprise':
            actions.append("考虑企业级安全和合规要求")
            actions.append("制定详细的沟通计划")
        
        return actions[:8]  # 返回前8个最重要的行动
    
    def _identify_risk_factors(self, text: str, complexity: ComplexityLevel, contextual_features: ContextualFeature) -> List[str]:
        """识别风险因素"""
        risks = []
        
        # 基于复杂度的风险
        if complexity == ComplexityLevel.CRITICAL:
            risks.extend([
                "技术复杂度高，可能影响项目进度",
                "需要更多资源和时间投入",
                "存在较高的技术风险"
            ])
        elif complexity == ComplexityLevel.COMPLEX:
            risks.extend([
                "需要仔细规划和管理",
                "可能遇到技术挑战",
                "需要团队协作配合"
            ])
        
        # 基于上下文的风险
        if contextual_features.environment == 'production':
            risks.extend([
                "生产环境变更风险",
                "可能影响现有系统稳定性",
                "需要充分的测试验证"
            ])
        
        if contextual_features.scale == 'enterprise':
            risks.extend([
                "企业级部署复杂度高",
                "需要考虑安全和合规要求",
                "利益相关者众多，沟通成本高"
            ])
        
        # 基于文本内容的风险
        risk_indicators = ['风险', '问题', '错误', '失败', '困难', '挑战', '复杂', '紧急']
        text_lower = text.lower()
        
        for indicator in risk_indicators:
            if indicator in text_lower:
                risks.append(f"文本中提到了{indicator}，需要特别关注")
        
        return list(set(risks))[:5]  # 返回前5个最重要的风险
    
    def _define_success_criteria(self, primary_intent: IntentType, complexity: ComplexityLevel,
                                contextual_features: ContextualFeature) -> List[str]:
        """定义成功标准"""
        criteria = []
        
        # 基于意图的成功标准
        intent_criteria = {
            IntentType.CREATE: [
                "功能实现完整且符合需求",
                "代码质量达到标准",
                "通过相关测试验证",
                "文档完善清晰"
            ],
            IntentType.ANALYZE: [
                "分析结果准确可靠",
                "发现关键问题和机会",
                "提供可行的改进建议",
                "报告内容详实有用"
            ],
            IntentType.OPTIMIZE: [
                "性能指标显著改善",
                "系统稳定性不受影响",
                "优化效果可量化验证",
                "资源使用更加高效"
            ],
            IntentType.DEBUG: [
                "问题得到根本解决",
                "修复方案稳定可靠",
                "问题不再重现",
                "预防措施到位"
            ],
            IntentType.LEARN: [
                "掌握了目标知识和技能",
                "能够独立应用所学",
                "学习成果可验证",
                "建立了持续学习机制"
            ],
            IntentType.DEPLOY: [
                "部署过程顺利完成",
                "系统运行正常稳定",
                "性能指标达到预期",
                "监控告警配置完善"
            ],
            IntentType.REFACTOR: [
                "代码结构更加清晰",
                "可维护性显著提升",
                "功能行为保持一致",
                "测试覆盖率不降低"
            ],
            IntentType.TEST: [
                "测试覆盖率达到要求",
                "发现并修复了关键问题",
                "测试结果可重现",
                "测试报告详实准确"
            ],
            IntentType.DOCUMENT: [
                "文档内容准确完整",
                "结构清晰易于理解",
                "示例和说明充分",
                "文档格式规范统一"
            ],
            IntentType.RESEARCH: [
                "研究目标明确达成",
                "数据收集充分可靠",
                "分析结论有理有据",
                "研究成果具有实用价值"
            ],
            IntentType.DESIGN: [
                "设计方案满足所有需求",
                "技术选型合理可行",
                "架构设计可扩展",
                "设计文档完整规范"
            ],
            IntentType.INTEGRATE: [
                "集成功能正常工作",
                "数据传输准确可靠",
                "系统稳定性不受影响",
                "集成方案可维护"
            ],
            IntentType.MONITOR: [
                "监控系统正常运行",
                "关键指标有效监控",
                "告警机制及时准确",
                "监控数据可用于决策"
            ]
        }
        
        criteria.extend(intent_criteria.get(primary_intent, []))
        
        # 基于复杂度的标准调整
        if complexity == ComplexityLevel.CRITICAL:
            criteria.append("项目按时按质量交付")
            criteria.append("风险得到有效控制")
            criteria.append("利益相关者满意度达标")
        
        # 基于上下文的标准调整
        if contextual_features.environment == 'production':
            criteria.append("生产环境零事故")
            criteria.append("用户体验不受影响")
        
        if contextual_features.scale == 'enterprise':
            criteria.append("符合企业级标准")
            criteria.append("通过安全和合规审查")
        
        return criteria[:6]  # 返回前6个最重要的标准
    
    def get_understanding_statistics(self) -> Dict[str, Any]:
        """获取理解统计信息"""
        if not self.context_history:
            return {
                'total_contexts': 0,
                'average_confidence': 0.0,
                'most_common_intent': None,
                'most_common_complexity': None,
                'most_common_urgency': None,
                'average_processing_time': 0.0
            }
        
        total_contexts = len(self.context_history)
        
        # 计算平均置信度
        confidences = [ctx['context'].confidence for ctx in self.context_history]
        avg_confidence = np.mean(confidences)
        
        # 最常见的意图
        intents = [ctx['context'].primary_intent.value for ctx in self.context_history]
        intent_counter = Counter(intents)
        most_common_intent = intent_counter.most_common(1)[0][0] if intent_counter else None
        
        # 最常见的复杂度
        complexities = [ctx['context'].complexity.value for ctx in self.context_history]
        complexity_counter = Counter(complexities)
        most_common_complexity = complexity_counter.most_common(1)[0][0] if complexity_counter else None
        
        # 最常见的紧急度
        urgencies = [ctx['context'].urgency.value for ctx in self.context_history]
        urgency_counter = Counter(urgencies)
        most_common_urgency = urgency_counter.most_common(1)[0][0] if urgency_counter else None
        
        # 平均处理时间
        processing_times = [ctx['processing_time'] for ctx in self.context_history]
        avg_processing_time = np.mean(processing_times)
        
        return {
            'total_contexts': total_contexts,
            'average_confidence': avg_confidence,
            'most_common_intent': most_common_intent,
            'most_common_complexity': most_common_complexity,
            'most_common_urgency': most_common_urgency,
            'average_processing_time': avg_processing_time,
            'intent_distribution': dict(intent_counter),
            'complexity_distribution': dict(complexity_counter),
            'urgency_distribution': dict(urgency_counter)
        }
    
    def learn_from_feedback(self, context: EnhancedContext, feedback: Dict[str, Any]):
        """从反馈中学习"""
        if not self.learning_enabled:
            return
        
        # 记录反馈数据
        feedback_data = {
            'timestamp': time.time(),
            'context_id': id(context),
            'predicted_intent': context.primary_intent.value,
            'predicted_complexity': context.complexity.value,
            'predicted_urgency': context.urgency.value,
            'predicted_confidence': context.confidence,
            'actual_intent': feedback.get('actual_intent'),
            'actual_complexity': feedback.get('actual_complexity'),
            'actual_urgency': feedback.get('actual_urgency'),
            'satisfaction_score': feedback.get('satisfaction_score', 0.5),
            'corrections': feedback.get('corrections', [])
        }
        
        # 更新模式权重（简化版学习机制）
        if feedback.get('actual_intent') and feedback['actual_intent'] != context.primary_intent.value:
            # 这里可以实现更复杂的学习算法
            logger.info(f"🧠 学习反馈: 预测意图 {context.primary_intent.value} -> 实际意图 {feedback['actual_intent']}")
        
        if feedback.get('satisfaction_score', 0.5) < 0.3:
            logger.warning(f"🧠 低满意度反馈: {feedback.get('corrections', '无具体反馈')}")
        
        logger.debug(f"🧠 反馈学习完成: 满意度 {feedback.get('satisfaction_score', 0.5):.2f}")

# 示例使用
def example_enhanced_context_usage():
    """示例增强上下文理解使用"""
    understanding = EnhancedContextUnderstanding()
    
    # 测试用例
    test_cases = [
        "我需要优化Python机器学习模型的训练性能，特别是图像分类部分",
        "帮我在生产环境部署React应用，需要确保高可用性和安全性",
        "分析这个复杂的微服务架构问题，找出性能瓶颈并提供解决方案",
        "创建一个用户认证系统，包括注册、登录、密码重置功能",
        "调试这个内存泄漏问题，系统在高并发情况下会出现崩溃"
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n🧠 测试用例 {i+1}: {test_case}")
        
        context = understanding.understand_context(test_case)
        
        print(f"🎯 主要意图: {context.primary_intent.value}")
        print(f"🔧 次要意图: {[intent.value for intent in context.secondary_intents]}")
        print(f"📊 复杂度: {context.complexity.value}")
        print(f"⚡ 紧急度: {context.urgency.value}")
        print(f"📈 置信度: {context.confidence:.2f}")
        print(f"❓ 歧义度: {context.ambiguity_score:.2f}")
        print(f"🌐 领域: {context.contextual_features.domain}")
        print(f"💻 语言: {context.contextual_features.language or '未指定'}")
        print(f"🏗️  框架: {context.contextual_features.framework or '未指定'}")
        print(f"📏 规模: {context.contextual_features.scale}")
        print(f"🤝 利益相关者: {context.contextual_features.stakeholders}")
        print(f"⏰ 时间框架: {context.temporal_features.timeframe}")
        
        print(f"💡 建议行动:")
        for j, action in enumerate(context.suggested_actions[:3], 1):
            print(f"  {j}. {action}")
        
        print(f"⚠️  风险因素:")
        for j, risk in enumerate(context.risk_factors[:2], 1):
            print(f"  {j}. {risk}")
        
        print(f"✅ 成功标准:")
        for j, criterion in enumerate(context.success_criteria[:2], 1):
            print(f"  {j}. {criterion}")
        
        print("-" * 60)
    
    # 显示统计信息
    stats = understanding.get_understanding_statistics()
    print(f"\n📊 理解统计:")
    print(f"  总处理上下文: {stats['total_contexts']}")
    print(f"  平均置信度: {stats['average_confidence']:.2f}")
    print(f"  最常见意图: {stats['most_common_intent']}")
    print(f"  最常见复杂度: {stats['most_common_complexity']}")
    print(f"  最常见紧急度: {stats['most_common_urgency']}")
    print(f"  平均处理时间: {stats['average_processing_time']:.3f}秒")

if __name__ == "__main__":
    example_enhanced_context_usage()