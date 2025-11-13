#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能MCP管理器 - 自动识别和调用MCP工具
Intelligent MCP Manager - Automatic Recognition and Invocation of MCP Tools

作者: Quantum AI Team
版本: 5.2.0
日期: 2025-11-12
"""

import re
import json
import time
import logging
import asyncio
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import threading
from collections import defaultdict, deque

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    """任务类型枚举"""
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    DEPLOYMENT = "deployment"
    RESEARCH = "research"
    DESIGN = "design"
    VALIDATION = "validation"

@dataclass
class MCPTool:
    """MCP工具定义"""
    name: str
    description: str
    capabilities: List[str]
    task_types: List[TaskType]
    priority: int  # 1-10
    command_pattern: str
    example_usage: str
    success_indicators: List[str]
    failure_indicators: List[str]
    auto_call_threshold: float  # 0.0-1.0
    performance_score: float = 0.0

@dataclass
class ContextAnalysis:
    """上下文分析结果"""
    task_type: TaskType
    complexity: str  # simple, medium, complex, critical
    urgency: str  # low, normal, high, critical
    domain: str  # web, mobile, ai, data, system
    language: Optional[str]
    tools_needed: List[str]
    confidence: float
    keywords: List[str]
    user_intent: str

class IntelligentMCPManager:
    """智能MCP管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化智能MCP管理器"""
        self.config_path = config_path or "mcp_tools_config.json"
        self.available_tools = {}
        self.tool_performance = defaultdict(list)
        self.context_history = deque(maxlen=100)
        self.auto_call_history = defaultdict(list)
        self.learning_enabled = True
        
        # 加载工具配置
        self._load_tools_configuration()
        
        # 初始化默认工具
        self._initialize_default_tools()
        
        # 性能监控线程
        self.performance_monitor_thread = threading.Thread(target=self._performance_monitor, daemon=True)
        self.performance_monitor_thread.start()
        
        logger.info("🤖 智能MCP管理器初始化完成")
    
    def _load_tools_configuration(self):
        """加载工具配置"""
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                for tool_config in config.get('tools', []):
                    tool = MCPTool(
                        name=tool_config['name'],
                        description=tool_config['description'],
                        capabilities=tool_config['capabilities'],
                        task_types=[TaskType(t) for t in tool_config['task_types']],
                        priority=tool_config['priority'],
                        command_pattern=tool_config.get('command_pattern', ''),
                        example_usage=tool_config.get('example_usage', ''),
                        success_indicators=tool_config.get('success_indicators', []),
                        failure_indicators=tool_config.get('failure_indicators', []),
                        auto_call_threshold=tool_config.get('auto_call_threshold', 0.7),
                        performance_score=tool_config.get('performance_score', 0.5)
                    )
                    self.available_tools[tool.name] = tool
                    
                logger.info(f"📄 加载了 {len(self.available_tools)} 个工具配置")
                    
            except Exception as e:
                logger.error(f"❌ 加载工具配置失败: {e}")
    
    def _initialize_default_tools(self):
        """初始化默认工具"""
        default_tools = [
            {
                "name": "adaptive_quantum_annealing",
                "description": "自适应量子退火优化算法",
                "capabilities": ["optimization", "quantum_computing", "parameter_tuning"],
                "task_types": ["optimization", "analysis"],
                "priority": 9,
                "command_pattern": "python tools/adaptive_quantum_annealing.py",
                "example_usage": "优化项目性能或复杂问题",
                "success_indicators": ["收敛", "优化完成", "找到最优解"],
                "failure_indicators": ["错误", "不收敛", "超时"],
                "auto_call_threshold": 0.8,
                "performance_score": 0.9
            },
            {
                "name": "reinforcement_learning_agent",
                "description": "强化学习智能体决策系统",
                "capabilities": ["machine_learning", "decision_making", "agent_coordination"],
                "task_types": ["analysis", "optimization", "coordination"],
                "priority": 8,
                "command_pattern": "python tools/reinforcement_learning_agent.py",
                "example_usage": "多智能体协作决策或策略优化",
                "success_indicators": ["学习完成", "策略改进", "决策优化"],
                "failure_indicators": ["学习失败", "收敛慢", "策略错误"],
                "auto_call_threshold": 0.7,
                "performance_score": 0.8
            },
            {
                "name": "quantum_intelligent_router",
                "description": "量子智能路由系统",
                "capabilities": ["routing", "model_selection", "load_balancing", "ide_detection"],
                "task_types": ["analysis", "coordination"],
                "priority": 9,
                "command_pattern": "python tools/quantum_intelligent_router.py",
                "example_usage": "智能路由任务到最优AI模型",
                "success_indicators": ["路由完成", "模型选择", "负载均衡"],
                "failure_indicators": ["路由失败", "模型不可用", "负载过高"],
                "auto_call_threshold": 0.9,
                "performance_score": 0.9
            },
            {
                "name": "agent_memory_system",
                "description": "智能体记忆和学习系统",
                "capabilities": ["memory", "learning", "knowledge_management", "experience_tracking"],
                "task_types": ["analysis", "documentation", "learning"],
                "priority": 7,
                "command_pattern": "python tools/agent_memory_system.py",
                "example_usage": "存储和检索开发经验和知识",
                "success_indicators": ["记忆存储", "知识检索", "经验学习"],
                "failure_indicators": ["存储失败", "检索错误", "学习问题"],
                "auto_call_threshold": 0.6,
                "performance_score": 0.7
            }
        ]
        
        for tool_config in default_tools:
            if tool_config['name'] not in self.available_tools:
                tool = MCPTool(
                    name=tool_config['name'],
                    description=tool_config['description'],
                    capabilities=tool_config['capabilities'],
                    task_types=[TaskType(t) for t in tool_config['task_types']],
                    priority=tool_config['priority'],
                    command_pattern=tool_config['command_pattern'],
                    example_usage=tool_config['example_usage'],
                    success_indicators=tool_config['success_indicators'],
                    failure_indicators=tool_config['failure_indicators'],
                    auto_call_threshold=tool_config['auto_call_threshold'],
                    performance_score=tool_config['performance_score']
                )
                self.available_tools[tool.name] = tool
        
        logger.info(f"🔧 初始化了 {len(self.available_tools)} 个默认工具")
    
    def analyze_context(self, user_input: str, conversation_history: List[str] = None) -> ContextAnalysis:
        """分析用户上下文和意图"""
        logger.info(f"🔍 分析上下文: {user_input[:50]}...")
        
        # 关键词提取
        keywords = self._extract_keywords(user_input)
        
        # 任务类型识别
        task_type = self._identify_task_type(user_input, keywords)
        
        # 复杂度评估
        complexity = self._assess_complexity(user_input, keywords)
        
        # 紧急程度评估
        urgency = self._assess_urgency(user_input, keywords)
        
        # 领域识别
        domain = self._identify_domain(user_input, keywords)
        
        # 语言识别
        language = self._identify_language(user_input, keywords)
        
        # 工具需求分析
        tools_needed = self._analyze_tool_needs(task_type, complexity, domain)
        
        # 置信度计算
        confidence = self._calculate_confidence(keywords, task_type, complexity)
        
        # 用户意图提取
        user_intent = self._extract_user_intent(user_input, keywords)
        
        analysis = ContextAnalysis(
            task_type=task_type,
            complexity=complexity,
            urgency=urgency,
            domain=domain,
            language=language,
            tools_needed=tools_needed,
            confidence=confidence,
            keywords=keywords,
            user_intent=user_intent
        )
        
        # 存储分析历史
        self.context_history.append(analysis)
        
        logger.info(f"📊 上下文分析完成: {task_type.value} - 置信度: {confidence:.2f}")
        return analysis
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 技术关键词
        tech_keywords = [
            '优化', '算法', '机器学习', '深度学习', '神经网络', '量子计算',
            '路由', '调度', '负载均衡', '缓存', '性能', '效率',
            '记忆', '学习', '经验', '知识', '推理', '决策',
            '代码', '编程', '开发', '测试', '调试', '重构', '部署',
            '分析', '设计', '架构', '系统', '项目', '工作流',
            'python', 'java', 'javascript', 'typescript', 'react', 'vue', 'angular',
            '数据库', 'api', '前端', '后端', '全栈', '微服务'
        ]
        
        # 情感关键词
        sentiment_keywords = [
            '紧急', '重要', '关键', '优先', '快速', '立即', '马上',
            '复杂', '困难', '挑战', '问题', '错误', '故障', '异常',
            '简单', '容易', '基础', '基本', '入门', '示例', '演示'
        ]
        
        # 领域关键词
        domain_keywords = [
            'web', '网站', '前端', '后端', '移动', '桌面', '游戏',
            'ai', '人工智能', '数据科学', '大数据', '云计算',
            '区块链', '安全', '网络', '系统', '运维', 'DevOps'
        ]
        
        found_keywords = []
        text_lower = text.lower()
        
        for keyword_list in [tech_keywords, sentiment_keywords, domain_keywords]:
            for keyword in keyword_list:
                if keyword in text_lower:
                    found_keywords.append(keyword)
        
        return list(set(found_keywords))
    
    def _identify_task_type(self, text: str, keywords: List[str]) -> TaskType:
        """识别任务类型"""
        task_patterns = {
            TaskType.CODE_GENERATION: ['代码', '编程', '实现', '开发', '编写', '生成'],
            TaskType.ANALYSIS: ['分析', '评估', '检查', '审查', '诊断'],
            TaskType.OPTIMIZATION: ['优化', '改进', '提升', '加速', '调优'],
            TaskType.DOCUMENTATION: ['文档', '说明', '指南', '教程', '手册'],
            TaskType.TESTING: ['测试', '验证', '检查', '质量保证'],
            TaskType.DEBUGGING: ['调试', '排错', '故障', '问题', '错误'],
            TaskType.REFACTORING: ['重构', '改进', '整理', '优化代码'],
            TaskType.DEPLOYMENT: ['部署', '发布', '上线', '运维'],
            TaskType.RESEARCH: ['研究', '调研', '探索', '查找'],
            TaskType.DESIGN: ['设计', '架构', '规划', '方案']
        }
        
        text_lower = text.lower()
        
        for task_type, patterns in task_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return task_type
        
        return TaskType.ANALYSIS  # 默认分析类型
    
    def _assess_complexity(self, text: str, keywords: List[str]) -> str:
        """评估复杂度"""
        complexity_indicators = {
            'simple': ['简单', '容易', '基础', '基本', '示例', '演示'],
            'medium': ['中等', '一般', '常规', '标准'],
            'complex': ['复杂', '困难', '挑战', '高级', '深度'],
            'critical': ['关键', '重要', '紧急', '严重', '核心']
        }
        
        text_lower = text.lower()
        
        for level, indicators in complexity_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    return level
        
        # 基于长度和关键词数量评估
        if len(text) > 500 or len(keywords) > 10:
            return 'complex'
        elif len(text) > 200 or len(keywords) > 5:
            return 'medium'
        else:
            return 'simple'
    
    def _assess_urgency(self, text: str, keywords: List[str]) -> str:
        """评估紧急程度"""
        urgency_indicators = {
            'critical': ['紧急', '立即', '马上', '严重', '关键', '重要'],
            'high': ['高', '优先', '尽快', '需要', '必须'],
            'normal': ['正常', '标准', '常规', '一般'],
            'low': ['低', '可以', '建议', '可选', '稍后']
        }
        
        text_lower = text.lower()
        
        for level, indicators in urgency_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    return level
        
        return 'normal'
    
    def _identify_domain(self, text: str, keywords: List[str]) -> str:
        """识别领域"""
        domain_patterns = {
            'web': ['web', '网站', '前端', '后端', '网页', '浏览器'],
            'mobile': ['移动', '手机', 'app', '应用', 'ios', 'android'],
            'ai': ['ai', '人工智能', '机器学习', '深度学习', '神经网络'],
            'data': ['数据', '数据库', '大数据', '分析', '处理'],
            'system': ['系统', '架构', '设计', '基础设施', '运维']
        }
        
        text_lower = text.lower()
        
        for domain, patterns in domain_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return domain
        
        return 'general'
    
    def _identify_language(self, text: str, keywords: List[str]) -> Optional[str]:
        """识别编程语言"""
        language_patterns = {
            'python': ['python', 'py', '.py'],
            'javascript': ['javascript', 'js', '.js', 'node', 'nodejs'],
            'typescript': ['typescript', 'ts', '.ts'],
            'java': ['java', '.java'],
            'go': ['go', '.go'],
            'rust': ['rust', '.rs'],
            'cpp': ['cpp', 'c++', '.cpp'],
            'c': ['c', '.c'],
            'html': ['html', '.html', 'web'],
            'css': ['css', '.css', '样式'],
            'sql': ['sql', 'database', '数据库']
        }
        
        text_lower = text.lower()
        
        for language, patterns in language_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return language
        
        return None
    
    def _analyze_tool_needs(self, task_type: TaskType, complexity: str, domain: str) -> List[str]:
        """分析工具需求"""
        tool_needs = []
        
        # 基于任务类型的工具需求
        task_tool_mapping = {
            TaskType.OPTIMIZATION: ['adaptive_quantum_annealing', 'reinforcement_learning_agent'],
            TaskType.CODE_GENERATION: ['quantum_intelligent_router', 'agent_memory_system'],
            TaskType.ANALYSIS: ['quantum_intelligent_router', 'agent_memory_system'],
            TaskType.DOCUMENTATION: ['agent_memory_system'],
            TaskType.TESTING: ['agent_memory_system'],
            TaskType.DEBUGGING: ['agent_memory_system', 'quantum_intelligent_router'],
            TaskType.REFACTORING: ['reinforcement_learning_agent', 'agent_memory_system'],
            TaskType.DEPLOYMENT: ['quantum_intelligent_router']
        }
        
        # 基于复杂度的工具选择
        if complexity in ['complex', 'critical']:
            # 高复杂度需要多个工具协作
            base_tools = task_tool_mapping.get(task_type, [])
            if base_tools:
                tool_needs.extend(base_tools)
            
            # 添加通用工具
            tool_needs.extend(['agent_memory_system', 'quantum_intelligent_router'])
        else:
            # 简单任务使用单个工具
            tool_needs = task_tool_mapping.get(task_type, ['quantum_intelligent_router'])
        
        # 基于领域的工具调整
        if domain == 'ai':
            if 'reinforcement_learning_agent' not in tool_needs:
                tool_needs.append('reinforcement_learning_agent')
        elif domain == 'system':
            if 'adaptive_quantum_annealing' not in tool_needs:
                tool_needs.append('adaptive_quantum_annealing')
        
        return list(set(tool_needs))
    
    def _calculate_confidence(self, keywords: List[str], task_type: TaskType, complexity: str) -> float:
        """计算置信度"""
        base_confidence = 0.5
        
        # 基于关键词匹配度
        keyword_score = min(1.0, len(keywords) / 10.0)
        
        # 基于任务类型明确度
        type_confidence = 0.8 if task_type != TaskType.ANALYSIS else 0.6
        
        # 基于复杂度匹配
        complexity_scores = {'simple': 0.9, 'medium': 0.7, 'complex': 0.5, 'critical': 0.3}
        complexity_score = complexity_scores.get(complexity, 0.5)
        
        confidence = (base_confidence + keyword_score + type_confidence + complexity_score) / 4.0
        return min(1.0, confidence)
    
    def _extract_user_intent(self, text: str, keywords: List[str]) -> str:
        """提取用户意图"""
        # 简化的意图提取
        if any(word in text.lower() for word in ['优化', '改进', '提升']):
            return "优化系统性能"
        elif any(word in text.lower() for word in ['分析', '检查', '评估']):
            return "分析现状问题"
        elif any(word in text.lower() for word in ['学习', '经验', '教训']):
            return "从经验中学习"
        elif any(word in text.lower() for word in ['实现', '开发', '创建']):
            return "开发新功能"
        elif any(word in text.lower() for word in ['修复', '解决', '处理']):
            return "解决问题"
        else:
            return "一般咨询"
    
    def should_auto_call_tool(self, tool_name: str, context: ContextAnalysis) -> bool:
        """判断是否应该自动调用工具"""
        if tool_name not in self.available_tools:
            return False
        
        tool = self.available_tools[tool_name]
        
        # 基于阈值判断
        if context.confidence < tool.auto_call_threshold:
            return False
        
        # 基于紧急程度判断
        if context.urgency in ['critical', 'high'] and tool.priority >= 7:
            return True
        
        # 基于任务类型匹配
        if context.task_type in tool.task_types and tool.priority >= 8:
            return True
        
        # 基于性能分数判断
        if tool.performance_score >= 0.8:
            return True
        
        return False
    
    def select_optimal_tools(self, context: ContextAnalysis) -> List[MCPTool]:
        """选择最优工具组合"""
        candidate_tools = []
        
        for tool_name in context.tools_needed:
            if tool_name in self.available_tools:
                tool = self.available_tools[tool_name]
                
                # 计算工具匹配分数
                match_score = 0.0
                
                # 任务类型匹配
                if context.task_type in tool.task_types:
                    match_score += 0.4
                
                # 优先级权重
                match_score += tool.priority / 10.0 * 0.3
                
                # 性能权重
                match_score += tool.performance_score * 0.2
                
                # 复杂度适配
                complexity_scores = {'simple': 0.3, 'medium': 0.2, 'complex': 0.1, 'critical': 0.05}
                match_score += complexity_scores.get(context.complexity, 0.2)
                
                candidate_tools.append((match_score, tool))
        
        # 排序并返回
        candidate_tools.sort(key=lambda x: x[0], reverse=True)
        return [tool for score, tool in candidate_tools]
    
    async def auto_call_tools(self, context: ContextAnalysis) -> Dict[str, Any]:
        """自动调用工具"""
        logger.info("🤖 自动调用MCP工具...")
        
        results = {}
        
        # 选择最优工具
        optimal_tools = self.select_optimal_tools(context)
        
        for tool in optimal_tools:
            if self.should_auto_call_tool(tool.name, context):
                try:
                    logger.info(f"🔧 自动调用工具: {tool.name}")
                    
                    # 执行工具调用
                    result = await self._execute_tool(tool, context)
                    results[tool.name] = result
                    
                    # 更新性能历史
                    self.tool_performance[tool.name].append({
                        'timestamp': time.time(),
                        'success': result.get('success', False),
                        'execution_time': result.get('execution_time', 0),
                        'context': context.user_intent
                    })
                    
                    # 学习和优化
                    if self.learning_enabled:
                        self._update_tool_performance(tool.name, result)
                    
                except Exception as e:
                    logger.error(f"❌ 工具调用失败 {tool.name}: {e}")
                    results[tool.name] = {
                        'success': False,
                        'error': str(e),
                        'execution_time': 0
                    }
        
        return results
    
    async def _execute_tool(self, tool: MCPTool, context: ContextAnalysis) -> Dict[str, Any]:
        """执行工具调用"""
        start_time = time.time()
        
        try:
            # 构建命令
            command = tool.command_pattern
            
            # 添加上下文参数
            if context.domain:
                command += f" --domain {context.domain}"
            if context.language:
                command += f" --language {context.language}"
            if context.complexity:
                command += f" --complexity {context.complexity}"
            
            # 执行命令
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            stdout, stderr = await process.communicate()
            
            execution_time = time.time() - start_time
            
            # 分析执行结果
            success = self._analyze_execution_result(
                stdout, stderr, tool.success_indicators, tool.failure_indicators
            )
            
            return {
                'success': success,
                'stdout': stdout,
                'stderr': stderr,
                'execution_time': execution_time,
                'command': command
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _analyze_execution_result(self, stdout: str, stderr: str, 
                                 success_indicators: List[str], 
                                 failure_indicators: List[str]) -> bool:
        """分析执行结果"""
        stdout_lower = stdout.lower() if stdout else ""
        stderr_lower = stderr.lower() if stderr else ""
        
        # 检查成功指标
        for indicator in success_indicators:
            if indicator.lower() in stdout_lower:
                return True
        
        # 检查失败指标
        for indicator in failure_indicators:
            if indicator.lower() in stderr_lower or indicator.lower() in stdout_lower:
                return False
        
        # 默认认为成功（如果没有明确的失败指标）
        return True
    
    def _update_tool_performance(self, tool_name: str, result: Dict[str, Any]):
        """更新工具性能分数"""
        if tool_name not in self.available_tools:
            return
        
        tool = self.available_tools[tool_name]
        
        # 基于执行结果更新性能分数
        current_score = tool.performance_score
        success = result.get('success', False)
        execution_time = result.get('execution_time', 0)
        
        if success:
            # 成功执行提升性能分数
            improvement = min(0.1, 1.0 - current_score)
            tool.performance_score = min(1.0, current_score + improvement)
        else:
            # 失败执行降低性能分数
            degradation = min(0.2, current_score)
            tool.performance_score = max(0.1, current_score - degradation)
        
        # 记录自动调用历史
        self.auto_call_history[tool_name].append({
            'timestamp': time.time(),
            'success': success,
            'execution_time': execution_time
        })
        
        # 限制历史记录数量
        if len(self.auto_call_history[tool_name]) > 100:
            self.auto_call_history[tool_name].pop(0)
        
        logger.debug(f"📊 更新工具性能: {tool_name} - 新分数: {tool.performance_score:.2f}")
    
    def _performance_monitor(self):
        """性能监控线程"""
        while True:
            try:
                time.sleep(30)  # 30秒间隔
                
                # 分析工具性能趋势
                for tool_name, history in self.tool_performance.items():
                    if len(history) >= 5:
                        recent_performance = history[-5:]
                        success_rate = sum(1 for record in recent_performance if record['success']) / len(recent_performance)
                        avg_time = sum(record['execution_time'] for record in recent_performance) / len(recent_performance)
                        
                        # 更新工具性能分数
                        if tool_name in self.available_tools:
                            tool = self.available_tools[tool_name]
                            if success_rate > 0.8 and avg_time < 10:
                                tool.performance_score = min(1.0, tool.performance_score + 0.05)
                            elif success_rate < 0.5 or avg_time > 30:
                                tool.performance_score = max(0.1, tool.performance_score - 0.1)
                        
                        logger.debug(f"📈 性能监控: {tool_name} - 成功率: {success_rate:.2f}, 平均时间: {avg_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ 性能监控错误: {e}")
    
    def get_tool_recommendations(self, context: ContextAnalysis) -> List[MCPTool]:
        """获取工具推荐"""
        recommendations = []
        
        # 基于上下文推荐工具
        all_tools = list(self.available_tools.values())
        
        for tool in all_tools:
            # 计算推荐分数
            recommendation_score = 0.0
            
            # 任务类型匹配度
            if context.task_type in tool.task_types:
                recommendation_score += 0.4
            
            # 领域匹配度
            domain_tools = {
                'ai': ['reinforcement_learning_agent', 'adaptive_quantum_annealing'],
                'system': ['adaptive_quantum_annealing', 'quantum_intelligent_router'],
                'web': ['quantum_intelligent_router', 'agent_memory_system'],
                'data': ['agent_memory_system', 'quantum_intelligent_router']
            }
            
            if context.domain in domain_tools:
                for recommended_tool in domain_tools[context.domain]:
                    if recommended_tool in self.available_tools:
                        tool = self.available_tools[recommended_tool]
                        recommendation_score += 0.3
            
            # 性能分数权重
            recommendation_score += tool.performance_score * 0.3
            
            # 基于复杂度的工具适配
            if context.complexity == 'simple':
                simple_tools = ['quantum_intelligent_router', 'agent_memory_system']
                if tool.name in simple_tools:
                    recommendation_score += 0.2
            elif context.complexity in ['complex', 'critical']:
                complex_tools = ['adaptive_quantum_annealing', 'reinforcement_learning_agent']
                if tool.name in complex_tools:
                    recommendation_score += 0.2
            
            recommendations.append((recommendation_score, tool))
        
        # 排序并返回
        recommendations.sort(key=lambda x: x[0], reverse=True)
        return [tool for score, tool in recommendations]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_tools = len(self.available_tools)
        auto_call_count = sum(len(history) for history in self.auto_call_history.values())
        avg_performance = sum(tool.performance_score for tool in self.available_tools.values()) / len(self.available_tools) if self.available_tools else 0
        
        return {
            'total_tools': total_tools,
            'auto_call_count': auto_call_count,
            'average_performance': avg_performance,
            'context_analysis_count': len(self.context_history),
            'performance_monitoring_active': self.performance_monitor_thread.is_alive(),
            'learning_enabled': self.learning_enabled,
            'tool_details': {
                tool.name: {
                    'name': tool.name,
                    'performance_score': tool.performance_score,
                    'auto_call_count': len(self.auto_call_history.get(tool.name, [])),
                    'success_rate': self._calculate_success_rate(tool.name)
                } for tool in self.available_tools.values()
            }
        }
    
    def _calculate_success_rate(self, tool_name: str) -> float:
        """计算工具成功率"""
        history = self.auto_call_history.get(tool_name, [])
        if not history:
            return 0.0
        
        successful_calls = sum(1 for record in history if record['success'])
        return successful_calls / len(history)
    
    def save_configuration(self, filepath: str = None):
        """保存配置"""
        if filepath is None:
            filepath = self.config_path
        
        config = {
            'tools': [asdict(tool) for tool in self.available_tools.values()],
            'learning_enabled': self.learning_enabled,
            'auto_call_history': dict(self.auto_call_history),
            'performance_history': dict(self.tool_performance)
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 配置已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ 保存配置失败: {e}")
    
    def load_configuration(self, filepath: str = None):
        """加载配置"""
        if filepath is None:
            filepath = self.config_path
        
        if not Path(filepath).exists():
            logger.warning(f"⚠️ 配置文件不存在: {filepath}")
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 重新加载工具配置
            self.available_tools.clear()
            for tool_config in config.get('tools', []):
                tool = MCPTool(
                    name=tool_config['name'],
                    description=tool_config['description'],
                    capabilities=tool_config['capabilities'],
                    task_types=[TaskType(t) for t in tool_config['task_types']],
                    priority=tool_config['priority'],
                    command_pattern=tool_config.get('command_pattern', ''),
                    example_usage=tool_config.get('example_usage', ''),
                    success_indicators=tool_config.get('success_indicators', []),
                    failure_indicators=tool_config.get('failure_indicators', []),
                    auto_call_threshold=tool_config.get('auto_call_threshold', 0.7),
                    performance_score=tool_config.get('performance_score', 0.5)
                )
                self.available_tools[tool.name] = tool
            
            self.learning_enabled = config.get('learning_enabled', True)
            self.auto_call_history = defaultdict(list, config.get('auto_call_history', {}))
            self.tool_performance = defaultdict(list, config.get('performance_history', {}))
            
            logger.info(f"📂 配置已从 {filepath} 加载")
            
        except Exception as e:
            logger.error(f"❌ 加载配置失败: {e}")

# 示例使用
async def example_intelligent_mcp_usage():
    """示例智能MCP使用"""
    manager = IntelligentMCPManager()
    
    # 模拟用户输入
    user_inputs = [
        "我需要优化这个Python项目的性能，特别是算法部分",
        "帮我分析这个复杂的系统架构问题",
        "创建一个用户认证系统的代码示例",
        "调试这个内存泄漏问题",
        "从之前的错误中学习经验"
    ]
    
    for user_input in user_inputs:
        print(f"\n📝 用户输入: {user_input}")
        
        # 分析上下文
        context = manager.analyze_context(user_input)
        
        print(f"🔍 分析结果: {context.task_type.value} - 复杂度: {context.complexity} - 置信度: {context.confidence:.2f}")
        print(f"🎯 用户意图: {context.user_intent}")
        print(f"🔧 推荐工具: {[tool.name for tool in manager.select_optimal_tools(context)]}")
        
        # 自动调用工具（如果满足条件）
        if context.confidence > 0.7:
            results = await manager.auto_call_tools(context)
            print(f"🤖 自动调用结果: {list(results.keys())}")
            
            for tool_name, result in results.items():
                if result['success']:
                    print(f"✅ {tool_name}: 执行成功 ({result['execution_time']:.2f}s)")
                else:
                    print(f"❌ {tool_name}: 执行失败 - {result.get('error', '未知错误')}")
        
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(example_intelligent_mcp_usage())
