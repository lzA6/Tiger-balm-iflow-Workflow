#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Omega Project Difficulty Analyzer - 自动项目难度识别和自适应调整系统
基于AI分析项目复杂度，自动调整执行策略和资源配置
"""

import os
import re
import json
import ast
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path
import hashlib

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DifficultyLevel(Enum):
    """难度级别枚举"""
    TRIVIAL = 1      # 微不足道 (< 30分钟)
    EASY = 2         # 简单 (30分钟 - 2小时)
    MEDIUM = 3       # 中等 (2小时 - 1天)
    HARD = 4         # 困难 (1天 - 3天)
    EXPERT = 5       # 专家级 (3天 - 1周)
    LEGENDARY = 6    # 传奇级 (> 1周)


class ProjectType(Enum):
    """项目类型枚举"""
    WEB_APPLICATION = "web_application"
    MOBILE_APPLICATION = "mobile_application"
    DESKTOP_APPLICATION = "desktop_application"
    API_SERVICE = "api_service"
    MACHINE_LEARNING = "machine_learning"
    DATA_ANALYSIS = "data_analysis"
    GAME_DEVELOPMENT = "game_development"
    SYSTEM_TOOL = "system_tool"
    LIBRARY_FRAMEWORK = "library_framework"
    BLOCKCHAIN = "blockchain"
    IOT_EMBEDDED = "iot_embedded"
    QUANTUM_COMPUTING = "quantum_computing"
    UNKNOWN = "unknown"


@dataclass
class ComplexityMetrics:
    """复杂度指标"""
    code_complexity: float = 0.0
    dependency_complexity: float = 0.0
    architectural_complexity: float = 0.0
    domain_complexity: float = 0.0
    scale_complexity: float = 0.0
    innovation_complexity: float = 0.0
    integration_complexity: float = 0.0
    overall_complexity: float = 0.0


@dataclass
class ProjectAnalysis:
    """项目分析结果"""
    project_path: str
    project_type: ProjectType
    difficulty_level: DifficultyLevel
    estimated_time: float  # 小时
    required_skills: List[str]
    complexity_metrics: ComplexityMetrics
    risk_factors: List[str]
    recommendations: List[str]
    resource_requirements: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodeAnalyzer:
    """代码分析器"""
    
    def __init__(self):
        self.language_patterns = {
            'python': r'\.py$',
            'javascript': r'\.(js|jsx|ts|tsx)$',
            'java': r'\.java$',
            'cpp': r'\.(cpp|cc|cxx|h|hpp)$',
            'csharp': r'\.cs$',
            'go': r'\.go$',
            'rust': r'\.rs$',
            'php': r'\.php$',
            'ruby': r'\.rb$',
            'swift': r'\.swift$',
            'kotlin': r'\.(kt|kts)$'
        }
        
        self.framework_patterns = {
            'react': r'react|React',
            'vue': r'vue|Vue',
            'angular': r'angular|Angular',
            'django': r'django|Django',
            'flask': r'flask|Flask',
            'fastapi': r'fastapi|FastAPI',
            'spring': r'spring|Spring',
            'express': r'express|Express',
            'tensorflow': r'tensorflow|TensorFlow',
            'pytorch': r'pytorch|PyTorch',
            'unity': r'unity|Unity',
            'ethereum': r'ethereum|Ethereum'
        }
    
    def analyze_codebase(self, project_path: str) -> Dict[str, Any]:
        """分析代码库"""
        analysis = {
            'languages': {},
            'frameworks': set(),
            'file_count': 0,
            'total_lines': 0,
            'max_file_size': 0,
            'average_file_size': 0,
            'complexity_scores': [],
            'dependencies': set(),
            'architecture_patterns': set()
        }
        
        file_sizes = []
        
        # 遍历项目文件
        for root, dirs, files in os.walk(project_path):
            # 跳过常见的忽略目录
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'target', 'build']]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                try:
                    # 获取文件信息
                    file_size = os.path.getsize(file_path)
                    file_sizes.append(file_size)
                    analysis['file_count'] += 1
                    analysis['max_file_size'] = max(analysis['max_file_size'], file_size)
                    
                    # 分析语言
                    language = self._detect_language(file)
                    if language:
                        if language not in analysis['languages']:
                            analysis['languages'][language] = {'files': 0, 'lines': 0}
                        analysis['languages'][language]['files'] += 1
                    
                    # 分析源代码文件
                    if self._is_source_file(file):
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                            # 统计行数
                            lines = len(content.splitlines())
                            analysis['total_lines'] += lines
                            if language:
                                analysis['languages'][language]['lines'] += lines
                            
                            # 分析复杂度
                            complexity = self._calculate_file_complexity(content, language)
                            if complexity > 0:
                                analysis['complexity_scores'].append(complexity)
                            
                            # 检测框架
                            frameworks = self._detect_frameworks(content)
                            analysis['frameworks'].update(frameworks)
                            
                            # 检测依赖
                            dependencies = self._detect_dependencies(content, language)
                            analysis['dependencies'].update(dependencies)
                            
                            # 检测架构模式
                            patterns = self._detect_architecture_patterns(content, file_path)
                            analysis['architecture_patterns'].update(patterns)
                
                except Exception as e:
                    logger.warning(f"Error analyzing file {file_path}: {str(e)}")
        
        # 计算平均文件大小
        if file_sizes:
            analysis['average_file_size'] = sum(file_sizes) / len(file_sizes)
        
        # 转换set为list以便JSON序列化
        analysis['frameworks'] = list(analysis['frameworks'])
        analysis['dependencies'] = list(analysis['dependencies'])
        analysis['architecture_patterns'] = list(analysis['architecture_patterns'])
        
        return analysis
    
    def _detect_language(self, filename: str) -> Optional[str]:
        """检测文件语言"""
        for language, pattern in self.language_patterns.items():
            if re.search(pattern, filename):
                return language
        return None
    
    def _is_source_file(self, filename: str) -> bool:
        """判断是否为源代码文件"""
        source_extensions = ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.cc', '.cxx', 
                           '.h', '.hpp', '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.kts']
        return any(filename.endswith(ext) for ext in source_extensions)
    
    def _calculate_file_complexity(self, content: str, language: str) -> float:
        """计算文件复杂度"""
        complexity = 0.0
        
        try:
            # 基础复杂度指标
            lines = len(content.splitlines())
            complexity += min(lines / 100.0, 5.0)  # 行数复杂度，最高5分
            
            # 语言特定复杂度分析
            if language == 'python':
                complexity += self._python_complexity(content)
            elif language in ['javascript', 'typescript']:
                complexity += self._javascript_complexity(content)
            elif language == 'java':
                complexity += self._java_complexity(content)
            elif language in ['cpp', 'cc', 'cxx']:
                complexity += self._cpp_complexity(content)
            
            # 通用复杂度指标
            complexity += self._generic_complexity(content)
            
        except Exception as e:
            logger.warning(f"Error calculating complexity: {str(e)}")
            complexity = 1.0  # 默认复杂度
        
        return complexity
    
    def _python_complexity(self, content: str) -> float:
        """Python代码复杂度分析"""
        complexity = 0.0
        
        try:
            tree = ast.parse(content)
            
            # 计算嵌套深度
            max_depth = self._calculate_ast_depth(tree)
            complexity += min(max_depth / 3.0, 3.0)
            
            # 统计复杂结构
            class_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            function_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            
            complexity += min(class_count / 5.0, 2.0)  # 类数量
            complexity += min(function_count / 10.0, 2.0)  # 函数数量
            
            # 检测高级特性
            advanced_features = ['async', 'await', 'yield', 'decorator', 'metaclass']
            for feature in advanced_features:
                if feature in content:
                    complexity += 0.5
        
        except:
            # 如果AST解析失败，使用简单指标
            complexity += content.count('class ') * 0.5
            complexity += content.count('def ') * 0.3
            complexity += content.count('import ') * 0.2
        
        return complexity
    
    def _javascript_complexity(self, content: str) -> float:
        """JavaScript代码复杂度分析"""
        complexity = 0.0
        
        # 统计函数和类
        complexity += min(content.count('function ') / 5.0, 2.0)
        complexity += min(content.count('class ') / 3.0, 2.0)
        complexity += min(content.count('=>') / 10.0, 1.0)  # 箭头函数
        
        # 统计复杂结构
        complexity += min(content.count('Promise') / 3.0, 1.0)
        complexity += min(content.count('async') / 5.0, 1.0)
        complexity += min(content.count('await') / 5.0, 1.0)
        
        # 框架特定复杂度
        if 'react' in content.lower():
            complexity += min(content.count('useState') / 3.0, 1.0)
            complexity += min(content.count('useEffect') / 3.0, 1.0)
        
        return complexity
    
    def _java_complexity(self, content: str) -> float:
        """Java代码复杂度分析"""
        complexity = 0.0
        
        # 统计类和方法
        complexity += min(content.count('class ') / 3.0, 2.0)
        complexity += min(content.count('public ') / 5.0, 2.0)
        complexity += min(content.count('private ') / 5.0, 1.0)
        
        # 统计复杂结构
        complexity += min(content.count('interface ') / 2.0, 1.0)
        complexity += min(content.count('extends ') / 3.0, 1.0)
        complexity += min(content.count('implements ') / 3.0, 1.0)
        
        # 注解和泛型
        complexity += min(content.count('@') / 10.0, 1.0)
        complexity += min(content.count('<') * content.count('>') / 20.0, 1.0)
        
        return complexity
    
    def _cpp_complexity(self, content: str) -> float:
        """C++代码复杂度分析"""
        complexity = 0.0
        
        # 统计类和模板
        complexity += min(content.count('class ') / 3.0, 2.0)
        complexity += min(content.count('template') / 2.0, 2.0)
        
        # 统计指针和内存操作
        complexity += min(content.count('*') / 10.0, 1.0)
        complexity += min(content.count('&') / 10.0, 1.0)
        complexity += min(content.count('new ') / 5.0, 1.0)
        complexity += min(content.count('delete ') / 5.0, 1.0)
        
        # 统计复杂特性
        complexity += min(content.count('virtual') / 3.0, 1.0)
        complexity += min(content.count('override') / 3.0, 1.0)
        complexity += min(content.count('constexpr') / 3.0, 1.0)
        
        return complexity
    
    def _generic_complexity(self, content: str) -> float:
        """通用复杂度分析"""
        complexity = 0.0
        
        # 控制流复杂度
        complexity += min(content.count('if ') / 10.0, 2.0)
        complexity += min(content.count('for ') / 5.0, 1.5)
        complexity += min(content.count('while ') / 5.0, 1.5)
        complexity += min(content.count('switch ') / 3.0, 1.0)
        
        # 异常处理
        complexity += min(content.count('try') / 3.0, 1.0)
        complexity += min(content.count('catch') / 3.0, 1.0)
        complexity += min(content.count('finally') / 3.0, 0.5)
        
        # 并发和线程
        concurrent_keywords = ['thread', 'mutex', 'lock', 'async', 'await', 'goroutine', 'channel']
        for keyword in concurrent_keywords:
            complexity += min(content.count(keyword) / 3.0, 0.5)
        
        return complexity
    
    def _calculate_ast_depth(self, node, depth=0) -> int:
        """计算AST深度"""
        if not hasattr(node, 'body'):
            return depth
        
        max_child_depth = depth
        for child in ast.iter_child_nodes(node):
            child_depth = self._calculate_ast_depth(child, depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def _detect_frameworks(self, content: str) -> List[str]:
        """检测使用的框架"""
        detected = []
        
        for framework, pattern in self.framework_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                detected.append(framework)
        
        return detected
    
    def _detect_dependencies(self, content: str, language: str) -> List[str]:
        """检测依赖项"""
        dependencies = []
        
        if language == 'python':
            # Python import语句
            imports = re.findall(r'import\s+(\w+)', content)
            imports += re.findall(r'from\s+(\w+)', content)
            dependencies.extend(imports)
        
        elif language in ['javascript', 'typescript']:
            # JavaScript/TypeScript import/require
            imports = re.findall(r'import.*from\s+[\'"]([^\'"]+)[\'"]', content)
            imports += re.findall(r'require\([\'"]([^\'"]+)[\'"]\)', content)
            dependencies.extend(imports)
        
        elif language == 'java':
            # Java import语句
            imports = re.findall(r'import\s+([\w\.]+);', content)
            dependencies.extend(imports)
        
        return dependencies
    
    def _detect_architecture_patterns(self, content: str, file_path: str) -> List[str]:
        """检测架构模式"""
        patterns = []
        
        # MVC模式检测
        if any(keyword in file_path.lower() for keyword in ['model', 'view', 'controller']):
            patterns.append('MVC')
        
        # 微服务模式检测
        microservice_keywords = ['service', 'api', 'gateway', 'discovery']
        if any(keyword in content.lower() for keyword in microservice_keywords):
            patterns.append('Microservices')
        
        # 事件驱动模式检测
        event_keywords = ['event', 'publish', 'subscribe', 'emit', 'listener']
        if any(keyword in content.lower() for keyword in event_keywords):
            patterns.append('Event-Driven')
        
        # 仓储模式检测
        if 'repository' in content.lower():
            patterns.append('Repository')
        
        # 工厂模式检测
        if 'factory' in content.lower():
            patterns.append('Factory')
        
        return patterns


class DifficultyCalculator:
    """难度计算器"""
    
    def __init__(self):
        self.difficulty_thresholds = {
            DifficultyLevel.TRIVIAL: (0, 2.0),
            DifficultyLevel.EASY: (2.0, 4.0),
            DifficultyLevel.MEDIUM: (4.0, 7.0),
            DifficultyLevel.HARD: (7.0, 10.0),
            DifficultyLevel.EXPERT: (10.0, 15.0),
            DifficultyLevel.LEGENDARY: (15.0, float('inf'))
        }
        
        self.time_multipliers = {
            DifficultyLevel.TRIVIAL: 1.0,
            DifficultyLevel.EASY: 2.0,
            DifficultyLevel.MEDIUM: 4.0,
            DifficultyLevel.HARD: 8.0,
            DifficultyLevel.EXPERT: 16.0,
            DifficultyLevel.LEGENDARY: 32.0
        }
    
    def calculate_difficulty(self, code_analysis: Dict[str, Any]) -> Tuple[DifficultyLevel, float, ComplexityMetrics]:
        """计算项目难度"""
        metrics = self._calculate_complexity_metrics(code_analysis)
        overall_score = metrics.overall_complexity
        
        # 确定难度级别
        difficulty = DifficultyLevel.MEDIUM  # 默认中等
        for level, (min_score, max_score) in self.difficulty_thresholds.items():
            if min_score <= overall_score < max_score:
                difficulty = level
                break
        
        # 估算时间
        estimated_time = self._estimate_time(difficulty, metrics)
        
        return difficulty, estimated_time, metrics
    
    def _calculate_complexity_metrics(self, code_analysis: Dict[str, Any]) -> ComplexityMetrics:
        """计算复杂度指标"""
        metrics = ComplexityMetrics()
        
        # 1. 代码复杂度
        if code_analysis.get('complexity_scores'):
            avg_complexity = np.mean(code_analysis['complexity_scores'])
            max_complexity = max(code_analysis['complexity_scores'])
            metrics.code_complexity = (avg_complexity + max_complexity) / 2.0
        
        # 2. 依赖复杂度
        dependency_count = len(code_analysis.get('dependencies', []))
        metrics.dependency_complexity = min(dependency_count / 10.0, 5.0)
        
        # 3. 架构复杂度
        pattern_count = len(code_analysis.get('architecture_patterns', []))
        framework_count = len(code_analysis.get('frameworks', []))
        metrics.architectural_complexity = min((pattern_count + framework_count) / 3.0, 5.0)
        
        # 4. 领域复杂度
        language_count = len(code_analysis.get('languages', {}))
        metrics.domain_complexity = min(language_count * 0.5 + framework_count * 0.3, 5.0)
        
        # 5. 规模复杂度
        file_count = code_analysis.get('file_count', 0)
        total_lines = code_analysis.get('total_lines', 0)
        metrics.scale_complexity = min(
            (file_count / 100.0) + (total_lines / 10000.0),
            5.0
        )
        
        # 6. 创新复杂度
        advanced_frameworks = ['tensorflow', 'pytorch', 'quantum', 'blockchain', 'unity']
        innovation_score = sum(0.5 for fw in code_analysis.get('frameworks', []) 
                             if any(adv in fw.lower() for adv in advanced_frameworks))
        metrics.innovation_complexity = min(innovation_score, 5.0)
        
        # 7. 集成复杂度
        integration_score = metrics.dependency_complexity * 0.3 + \
                           metrics.architectural_complexity * 0.4 + \
                           metrics.domain_complexity * 0.3
        metrics.integration_complexity = min(integration_score, 5.0)
        
        # 8. 总体复杂度
        weights = {
            'code_complexity': 0.25,
            'dependency_complexity': 0.15,
            'architectural_complexity': 0.20,
            'domain_complexity': 0.15,
            'scale_complexity': 0.10,
            'innovation_complexity': 0.10,
            'integration_complexity': 0.05
        }
        
        metrics.overall_complexity = sum(
            getattr(metrics, metric) * weight
            for metric, weight in weights.items()
        )
        
        return metrics
    
    def _estimate_time(self, difficulty: DifficultyLevel, metrics: ComplexityMetrics) -> float:
        """估算项目时间（小时）"""
        base_time = {
            DifficultyLevel.TRIVIAL: 0.5,   # 30分钟
            DifficultyLevel.EASY: 4.0,      # 4小时
            DifficultyLevel.MEDIUM: 12.0,   # 12小时 (1.5天)
            DifficultyLevel.HARD: 40.0,     # 40小时 (5天)
            DifficultyLevel.EXPERT: 120.0,  # 120小时 (15天)
            DifficultyLevel.LEGENDARY: 240.0 # 240小时 (30天)
        }
        
        estimated_time = base_time.get(difficulty, 40.0)
        
        # 基于复杂度指标调整时间
        complexity_multiplier = 1.0 + (metrics.overall_complexity - 5.0) * 0.1
        estimated_time *= max(complexity_multiplier, 0.5)
        
        return estimated_time


class AdaptiveStrategy:
    """自适应策略"""
    
    def __init__(self):
        self.strategies = {
            DifficultyLevel.TRIVIAL: self._trivial_strategy,
            DifficultyLevel.EASY: self._easy_strategy,
            DifficultyLevel.MEDIUM: self._medium_strategy,
            DifficultyLevel.HARD: self._hard_strategy,
            DifficultyLevel.EXPERT: self._expert_strategy,
            DifficultyLevel.LEGENDARY: self._legendary_strategy
        }
    
    def generate_strategy(self, analysis: ProjectAnalysis) -> Dict[str, Any]:
        """生成自适应策略"""
        strategy_func = self.strategies.get(analysis.difficulty_level, self._medium_strategy)
        base_strategy = strategy_func(analysis)
        
        # 基于项目类型调整策略
        type_adjustments = self._get_type_adjustments(analysis.project_type)
        
        # 合并策略
        for key, value in type_adjustments.items():
            if key in base_strategy:
                if isinstance(base_strategy[key], list) and isinstance(value, list):
                    base_strategy[key].extend(value)
                elif isinstance(base_strategy[key], dict) and isinstance(value, dict):
                    base_strategy[key].update(value)
                else:
                    base_strategy[key] = value
            else:
                base_strategy[key] = value
        
        return base_strategy
    
    def _trivial_strategy(self, analysis: ProjectAnalysis) -> Dict[str, Any]:
        """微不足道项目策略"""
        return {
            'approach': 'direct_implementation',
            'team_size': 1,
            'timeline_hours': analysis.estimated_time,
            'quality_gates': ['basic_testing'],
            'tools': ['basic_ide', 'linter'],
            'review_process': 'self_review',
            'deployment': 'direct_deployment',
            'monitoring': 'basic_logging',
            'risk_mitigation': []
        }
    
    def _easy_strategy(self, analysis: ProjectAnalysis) -> Dict[str, Any]:
        """简单项目策略"""
        return {
            'approach': 'iterative_development',
            'team_size': 1,
            'timeline_hours': analysis.estimated_time,
            'quality_gates': ['unit_testing', 'code_review'],
            'tools': ['ide', 'linter', 'unit_test_framework'],
            'review_process': 'peer_review',
            'deployment': 'automated_deployment',
            'monitoring': 'basic_monitoring',
            'risk_mitigation': ['backup_strategy']
        }
    
    def _medium_strategy(self, analysis: ProjectAnalysis) -> Dict[str, Any]:
        """中等项目策略"""
        return {
            'approach': 'agile_development',
            'team_size': 2,
            'timeline_hours': analysis.estimated_time,
            'quality_gates': ['unit_testing', 'integration_testing', 'code_review', 'documentation'],
            'tools': ['advanced_ide', 'linter', 'test_framework', 'ci_cd', 'documentation_tool'],
            'review_process': 'formal_review',
            'deployment': 'staged_deployment',
            'monitoring': 'comprehensive_monitoring',
            'risk_mitigation': ['backup_strategy', 'rollback_plan', 'testing_strategy']
        }
    
    def _hard_strategy(self, analysis: ProjectAnalysis) -> Dict[str, Any]:
        """困难项目策略"""
        return {
            'approach': 'structured_development',
            'team_size': 3,
            'timeline_hours': analysis.estimated_time,
            'quality_gates': ['comprehensive_testing', 'formal_review', 'architecture_review', 'security_audit'],
            'tools': ['professional_ide', 'advanced_testing', 'ci_cd', 'monitoring', 'security_tools'],
            'review_process': 'multiple_reviews',
            'deployment': 'blue_green_deployment',
            'monitoring': 'advanced_monitoring',
            'risk_mitigation': ['comprehensive_backup', 'disaster_recovery', 'security_hardening', 'performance_testing']
        }
    
    def _expert_strategy(self, analysis: ProjectAnalysis) -> Dict[str, Any]:
        """专家级项目策略"""
        return {
            'approach': 'enterprise_development',
            'team_size': 5,
            'timeline_hours': analysis.estimated_time,
            'quality_gates': ['full_testing_suite', 'expert_review', 'architecture_review', 'security_audit', 'performance_testing'],
            'tools': ['enterprise_tools', 'advanced_ci_cd', 'comprehensive_monitoring', 'security_suite', 'performance_tools'],
            'review_process': 'expert_review_process',
            'deployment': 'canary_deployment',
            'monitoring': 'enterprise_monitoring',
            'risk_mitigation': ['enterprise_backup', 'disaster_recovery', 'security_compliance', 'performance_optimization']
        }
    
    def _legendary_strategy(self, analysis: ProjectAnalysis) -> Dict[str, Any]:
        """传奇级项目策略"""
        return {
            'approach': 'strategic_development',
            'team_size': 8,
            'timeline_hours': analysis.estimated_time,
            'quality_gates': ['enterprise_testing', 'multiple_expert_reviews', 'formal_verification', 'security_certification'],
            'tools': ['cutting_edge_tools', 'enterprise_ci_cd', 'ai_assisted_development', 'advanced_monitoring'],
            'review_process': 'formal_verification_process',
            'deployment': 'gradual_rollout',
            'monitoring': 'ai_monitoring',
            'risk_mitigation': ['enterprise_disaster_recovery', 'formal_verification', 'security_certification', 'continuous_optimization']
        }
    
    def _get_type_adjustments(self, project_type: ProjectType) -> Dict[str, Any]:
        """获取项目类型特定调整"""
        adjustments = {
            ProjectType.WEB_APPLICATION: {
                'tools': ['frontend_framework', 'backend_framework', 'database_tools'],
                'quality_gates': ['browser_compatibility', 'responsive_testing'],
                'deployment': ['cdn_deployment', 'ssl_configuration']
            },
            ProjectType.MOBILE_APPLICATION: {
                'tools': ['mobile_ide', 'device_testing', 'app_store_tools'],
                'quality_gates': ['device_compatibility', 'performance_testing'],
                'deployment': ['app_store_deployment']
            },
            ProjectType.MACHINE_LEARNING: {
                'tools': ['ml_frameworks', 'data_processing_tools', 'model_validation'],
                'quality_gates': ['model_validation', 'data_quality_check'],
                'team_size_adjustment': 1  # 需要数据科学家
            },
            ProjectType.BLOCKCHAIN: {
                'tools': ['blockchain_framework', 'smart_contract_tools', 'security_audit'],
                'quality_gates': ['smart_contract_audit', 'security_testing'],
                'risk_mitigation': ['security_hardening', 'audit_trail']
            },
            ProjectType.QUANTUM_COMPUTING: {
                'tools': ['quantum_simulator', 'quantum_framework', 'specialized_hardware'],
                'quality_gates': ['quantum_validation', 'algorithm_verification'],
                'team_size_adjustment': 2,  # 需要量子专家
                'risk_mitigation': ['quantum_error_correction', 'validation_testing']
            }
        }
        
        return adjustments.get(project_type, {})


class OmegaProjectDifficultyAnalyzer:
    """Omega项目难度分析器 - 主系统"""
    
    def __init__(self):
        self.code_analyzer = CodeAnalyzer()
        self.difficulty_calculator = DifficultyCalculator()
        self.adaptive_strategy = AdaptiveStrategy()
        
        # 分析历史
        self.analysis_history = []
        self.performance_metrics = {
            'total_analyses': 0,
            'accuracy_score': 0.0,
            'average_analysis_time': 0.0
        }
    
    async def analyze_project(self, project_path: str) -> ProjectAnalysis:
        """分析项目难度"""
        start_time = time.time()
        
        logger.info(f"开始分析项目: {project_path}")
        
        # 1. 检测项目类型
        project_type = self._detect_project_type(project_path)
        
        # 2. 分析代码库
        code_analysis = self.code_analyzer.analyze_codebase(project_path)
        
        # 3. 计算难度
        difficulty, estimated_time, complexity_metrics = self.difficulty_calculator.calculate_difficulty(code_analysis)
        
        # 4. 识别所需技能
        required_skills = self._identify_required_skills(code_analysis, project_type)
        
        # 5. 识别风险因素
        risk_factors = self._identify_risk_factors(code_analysis, complexity_metrics)
        
        # 6. 生成建议
        recommendations = self._generate_recommendations(difficulty, complexity_metrics, risk_factors)
        
        # 7. 计算资源需求
        resource_requirements = self._calculate_resource_requirements(difficulty, complexity_metrics)
        
        # 8. 生成自适应策略
        analysis = ProjectAnalysis(
            project_path=project_path,
            project_type=project_type,
            difficulty_level=difficulty,
            estimated_time=estimated_time,
            required_skills=required_skills,
            complexity_metrics=complexity_metrics,
            risk_factors=risk_factors,
            recommendations=recommendations,
            resource_requirements=resource_requirements
        )
        
        # 9. 生成自适应策略
        analysis.metadata['adaptive_strategy'] = self.adaptive_strategy.generate_strategy(analysis)
        
        # 10. 更新性能指标
        analysis_time = time.time() - start_time
        self._update_metrics(analysis_time)
        
        # 11. 保存分析历史
        self.analysis_history.append(analysis)
        
        logger.info(f"项目分析完成: 难度级别={difficulty.name}, 预估时间={estimated_time:.1f}小时")
        
        return analysis
    
    def _detect_project_type(self, project_path: str) -> ProjectType:
        """检测项目类型"""
        # 检查配置文件
        config_files = {
            'package.json': ProjectType.WEB_APPLICATION,
            'pom.xml': ProjectType.WEB_APPLICATION,
            'build.gradle': ProjectType.WEB_APPLICATION,
            'requirements.txt': ProjectType.WEB_APPLICATION,
            'pubspec.yaml': ProjectType.MOBILE_APPLICATION,
            'Cargo.toml': ProjectType.DESKTOP_APPLICATION,
            'setup.py': ProjectType.LIBRARY_FRAMEWORK,
            'truffle-config.js': ProjectType.BLOCKCHAIN,
            'hardhat.config.js': ProjectType.BLOCKCHAIN
        }
        
        for root, dirs, files in os.walk(project_path):
            for file in files:
                if file in config_files:
                    return config_files[file]
        
        # 检查文件内容
        for root, dirs, files in os.walk(project_path):
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.rs')):
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read().lower()
                            
                            if any(keyword in content for keyword in ['tensorflow', 'pytorch', 'sklearn', 'machine learning']):
                                return ProjectType.MACHINE_LEARNING
                            elif any(keyword in content for keyword in ['unity', 'monobehaviour', 'gameobject']):
                                return ProjectType.GAME_DEVELOPMENT
                            elif any(keyword in content for keyword in ['ethereum', 'solidity', 'web3']):
                                return ProjectType.BLOCKCHAIN
                            elif any(keyword in content for keyword in ['qiskit', 'cirq', 'quantum']):
                                return ProjectType.QUANTUM_COMPUTING
                    except:
                        continue
        
        return ProjectType.UNKNOWN
    
    def _identify_required_skills(self, code_analysis: Dict[str, Any], project_type: ProjectType) -> List[str]:
        """识别所需技能"""
        skills = []
        
        # 基于编程语言
        for language in code_analysis.get('languages', {}):
            skills.append(f"{language}_programming")
        
        # 基于框架
        for framework in code_analysis.get('frameworks', []):
            skills.append(f"{framework}_framework")
        
        # 基于项目类型
        type_skills = {
            ProjectType.WEB_APPLICATION: ['frontend_development', 'backend_development', 'web_security'],
            ProjectType.MOBILE_APPLICATION: ['mobile_development', 'ui_ux_design', 'mobile_testing'],
            ProjectType.MACHINE_LEARNING: ['data_science', 'ml_engineering', 'statistics'],
            ProjectType.BLOCKCHAIN: ['blockchain_development', 'cryptography', 'smart_contracts'],
            ProjectType.QUANTUM_COMPUTING: ['quantum_physics', 'quantum_algorithms', 'linear_algebra']
        }
        
        skills.extend(type_skills.get(project_type, []))
        
        # 去重
        return list(set(skills))
    
    def _identify_risk_factors(self, code_analysis: Dict[str, Any], metrics: ComplexityMetrics) -> List[str]:
        """识别风险因素"""
        risks = []
        
        # 复杂度风险
        if metrics.overall_complexity > 10:
            risks.append("高复杂度项目，需要仔细规划")
        
        # 规模风险
        if code_analysis.get('file_count', 0) > 1000:
            risks.append("大型项目，需要良好的架构设计")
        
        # 依赖风险
        if len(code_analysis.get('dependencies', [])) > 50:
            risks.append("依赖项过多，存在依赖冲突风险")
        
        # 技术风险
        advanced_frameworks = ['tensorflow', 'pytorch', 'quantum', 'blockchain']
        if any(fw in code_analysis.get('frameworks', []) for fw in advanced_frameworks):
            risks.append("使用先进技术，需要专业知识")
        
        # 维护性风险
        if metrics.code_complexity > 5:
            risks.append("代码复杂度高，维护成本可能较高")
        
        return risks
    
    def _generate_recommendations(self, difficulty: DifficultyLevel, metrics: ComplexityMetrics, risks: List[str]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于难度的建议
        if difficulty.value >= DifficultyLevel.HARD.value:
            recommendations.append("建议分阶段实施，降低项目风险")
            recommendations.append("增加代码审查和测试覆盖率")
        
        # 基于复杂度的建议
        if metrics.architectural_complexity > 4:
            recommendations.append("重视架构设计，考虑使用设计模式")
        
        if metrics.dependency_complexity > 4:
            recommendations.append("管理依赖项，考虑依赖注入")
        
        # 基于风险的建议
        if "高复杂度项目" in risks:
            recommendations.append("考虑重构简化复杂模块")
        
        if "依赖项过多" in risks:
            recommendations.append("评估和清理不必要的依赖")
        
        # 通用建议
        recommendations.extend([
            "建立完善的文档体系",
            "实施持续集成和持续部署",
            "定期进行代码重构"
        ])
        
        return recommendations
    
    def _calculate_resource_requirements(self, difficulty: DifficultyLevel, metrics: ComplexityMetrics) -> Dict[str, Any]:
        """计算资源需求"""
        base_requirements = {
            DifficultyLevel.TRIVIAL: {'developers': 1, 'hours_per_day': 4},
            DifficultyLevel.EASY: {'developers': 1, 'hours_per_day': 6},
            DifficultyLevel.MEDIUM: {'developers': 2, 'hours_per_day': 8},
            DifficultyLevel.HARD: {'developers': 3, 'hours_per_day': 8},
            DifficultyLevel.EXPERT: {'developers': 5, 'hours_per_day': 8},
            DifficultyLevel.LEGENDARY: {'developers': 8, 'hours_per_day': 8}
        }
        
        requirements = base_requirements.get(difficulty, base_requirements[DifficultyLevel.MEDIUM]).copy()
        
        # 基于复杂度调整
        complexity_multiplier = 1.0 + (metrics.overall_complexity - 5.0) * 0.1
        requirements['developers'] = max(1, int(requirements['developers'] * complexity_multiplier))
        
        # 添加其他资源
        requirements.update({
            'testing_hours_ratio': 0.2,  # 测试时间占比
            'review_hours_ratio': 0.15,   # 代码审查时间占比
            'documentation_hours_ratio': 0.1,  # 文档时间占比
            'contingency_ratio': 0.2      # 应急时间占比
        })
        
        return requirements
    
    def _update_metrics(self, analysis_time: float):
        """更新性能指标"""
        self.performance_metrics['total_analyses'] += 1
        
        # 更新平均分析时间
        total = self.performance_metrics['total_analyses']
        current_avg = self.performance_metrics['average_analysis_time']
        self.performance_metrics['average_analysis_time'] = (
            (current_avg * (total - 1) + analysis_time) / total
        )
    
    def get_analysis_report(self, project_path: str) -> Dict[str, Any]:
        """获取分析报告"""
        # 查找历史分析
        analysis = None
        for hist in self.analysis_history:
            if hist.project_path == project_path:
                analysis = hist
                break
        
        if not analysis:
            return {'error': 'Project not analyzed yet'}
        
        return {
            'project_info': {
                'path': analysis.project_path,
                'type': analysis.project_type.value,
                'difficulty': analysis.difficulty_level.name,
                'estimated_time_hours': analysis.estimated_time
            },
            'complexity_metrics': {
                'code_complexity': analysis.complexity_metrics.code_complexity,
                'dependency_complexity': analysis.complexity_metrics.dependency_complexity,
                'architectural_complexity': analysis.complexity_metrics.architectural_complexity,
                'domain_complexity': analysis.complexity_metrics.domain_complexity,
                'scale_complexity': analysis.complexity_metrics.scale_complexity,
                'innovation_complexity': analysis.complexity_metrics.innovation_complexity,
                'integration_complexity': analysis.complexity_metrics.integration_complexity,
                'overall_complexity': analysis.complexity_metrics.overall_complexity
            },
            'required_skills': analysis.required_skills,
            'risk_factors': analysis.risk_factors,
            'recommendations': analysis.recommendations,
            'resource_requirements': analysis.resource_requirements,
            'adaptive_strategy': analysis.metadata.get('adaptive_strategy', {}),
            'performance_metrics': self.performance_metrics
        }


# 使用示例
async def main():
    """主函数示例"""
    analyzer = OmegaProjectDifficultyAnalyzer()
    
    # 分析当前项目
    project_path = "."
    print(f"分析项目: {project_path}")
    
    analysis = await analyzer.analyze_project(project_path)
    
    print(f"\n项目分析结果:")
    print(f"项目类型: {analysis.project_type.value}")
    print(f"难度级别: {analysis.difficulty_level.name}")
    print(f"预估时间: {analysis.estimated_time:.1f} 小时")
    print(f"所需技能: {', '.join(analysis.required_skills)}")
    print(f"风险因素: {', '.join(analysis.risk_factors)}")
    print(f"建议: {', '.join(analysis.recommendations[:3])}...")
    
    # 获取完整报告
    report = analyzer.get_analysis_report(project_path)
    
    print("\n" + "="*50)
    print("详细分析报告:")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
