#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能项目分析和架构设计系统
Intelligent Project Analysis and Architecture Design System

作者: Quantum AI Team
版本: 5.2.0
日期: 2025-11-12
"""

import os
import re
import ast
import json
import time
import asyncio
import logging
import subprocess
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
from collections import defaultdict, Counter
import hashlib
import mimetypes
import sys

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProjectType(Enum):
    """项目类型"""
    WEB_APP = "web_application"
    MOBILE_APP = "mobile_application"
    DESKTOP_APP = "desktop_application"
    API_SERVICE = "api_service"
    MICROSERVICE = "microservice"
    LIBRARY = "library"
    CLI_TOOL = "cli_tool"
    DATA_PIPELINE = "data_pipeline"
    MACHINE_LEARNING = "machine_learning"
    GAME = "game"
    EMBEDDED = "embedded"
    UNKNOWN = "unknown"

class ArchitecturePattern(Enum):
    """架构模式"""
    MONOLITH = "monolith"
    MICROSERVICES = "microservices"
    EVENT_DRIVEN = "event_driven"
    CQRS = "cqrs"
    HEXAGONAL = "hexagonal"
    LAYERED = "layered"
    MVC = "mvc"
    MVP = "mvp"
    MVVM = "mvvm"
    CLEAN_ARCHITECTURE = "clean_architecture"
    SERVERLESS = "serverless"
    UNKNOWN = "unknown"

class TechnologyStack(Enum):
    """技术栈"""
    PYTHON_DJANGO = "python_django"
    PYTHON_FASTAPI = "python_fastapi"
    PYTHON_FLASK = "python_flask"
    NODE_EXPRESS = "node_express"
    REACT = "react"
    VUE = "vue"
    ANGULAR = "angular"
    JAVA_SPRING = "java_spring"
    DOTNET_CORE = "dotnet_core"
    GO = "go"
    RUST = "rust"
    UNKNOWN = "unknown"

@dataclass
class ProjectMetrics:
    """项目指标"""
    total_files: int
    total_lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int
    file_types: Dict[str, int]
    dependencies: Dict[str, List[str]]
    complexity_score: float
    maintainability_index: float
    test_coverage: float

@dataclass
class ArchitectureComponent:
    """架构组件"""
    component_id: str
    name: str
    type: str  # service, controller, model, etc.
    file_path: str
    dependencies: List[str]
    complexity: float
    responsibilities: List[str]
    interfaces: List[str]

@dataclass
class ArchitectureAnalysis:
    """架构分析结果"""
    project_type: ProjectType
    architecture_pattern: ArchitecturePattern
    technology_stack: TechnologyStack
    components: List[ArchitectureComponent]
    metrics: ProjectMetrics
    recommendations: List[str]
    issues: List[str]
    strengths: List[str]

class ProjectAnalyzer:
    """项目分析器"""
    
    def __init__(self):
        """初始化项目分析器"""
        self.file_analyzers = {
            '.py': self._analyze_python_file,
            '.js': self._analyze_javascript_file,
            '.ts': self._analyze_typescript_file,
            '.java': self._analyze_java_file,
            '.cs': self._analyze_csharp_file,
            '.go': self._analyze_go_file,
            '.rs': self._analyze_rust_file,
            '.cpp': self._analyze_cpp_file,
            '.c': self._analyze_cpp_file,
            '.h': self._analyze_cpp_file,
            '.hpp': self._analyze_cpp_file,
            '.json': self._analyze_json_file,
            '.yaml': self._analyze_yaml_file,
            '.yml': self._analyze_yaml_file,
            '.md': self._analyze_markdown_file,
            '.txt': self._analyze_text_file
        }
        
        self.project_indicators = {
            ProjectType.WEB_APP: [
                'package.json', 'requirements.txt', 'pipfile', 'poetry.lock',
                'index.html', 'app.js', 'main.js', 'server.js', 'app.py',
                'templates', 'static', 'public', 'src', 'components'
            ],
            ProjectType.MOBILE_APP: [
                'android', 'ios', 'mobile', 'react-native', 'flutter',
                'cordova', 'ionic', 'xamarin', 'native', 'app.json'
            ],
            ProjectType.DESKTOP_APP: [
                'electron', 'qt', 'gtk', 'wxwidgets', 'winforms', 'wpf',
                'desktop', 'gui', 'ui', 'mainwindow', 'application'
            ],
            ProjectType.API_SERVICE: [
                'api', 'rest', 'graphql', 'grpc', 'endpoint', 'service',
                'controller', 'handler', 'route', 'middleware'
            ],
            ProjectType.MICROSERVICE: [
                'microservice', 'service', 'docker', 'kubernetes', 'k8s',
                'consul', 'eureka', 'zuul', 'gateway', 'discovery'
            ],
            ProjectType.LIBRARY: [
                'lib', 'library', 'package', 'module', 'setup.py',
                'pom.xml', 'build.gradle', 'cargo.toml', 'go.mod'
            ],
            ProjectType.CLI_TOOL: [
                'cli', 'command', 'terminal', 'console', 'argparse',
                'click', 'commander', 'yargs', 'main.py', 'index.js'
            ],
            ProjectType.DATA_PIPELINE: [
                'pipeline', 'etl', 'spark', 'hadoop', 'airflow',
                'kafka', 'data', 'analytics', 'batch', 'stream'
            ],
            ProjectType.MACHINE_LEARNING: [
                'ml', 'machine_learning', 'ai', 'model', 'training',
                'tensorflow', 'pytorch', 'scikit', 'jupyter', 'notebook'
            ],
            ProjectType.GAME: [
                'game', 'unity', 'unreal', 'godot', 'pygame',
                'sprite', 'engine', 'physics', 'rendering', 'player'
            ]
        }
        
        self.technology_indicators = {
            TechnologyStack.PYTHON_DJANGO: [
                'django', 'wsgi.py', 'settings.py', 'urls.py', 'views.py',
                'models.py', 'forms.py', 'admin.py', 'manage.py'
            ],
            TechnologyStack.PYTHON_FASTAPI: [
                'fastapi', 'pydantic', 'uvicorn', 'main.py', 'api',
                'endpoint', 'async', 'dependency injection'
            ],
            TechnologyStack.PYTHON_FLASK: [
                'flask', 'app.py', 'route', 'template', 'jinja2',
                'werkzeug', 'request', 'response'
            ],
            TechnologyStack.NODE_EXPRESS: [
                'express', 'node.js', 'npm', 'package.json', 'app.js',
                'server.js', 'middleware', 'router', 'req', 'res'
            ],
            TechnologyStack.REACT: [
                'react', 'jsx', 'component', 'hooks', 'state',
                'props', 'render', 'useEffect', 'useState'
            ],
            TechnologyStack.VUE: [
                'vue', 'vue.js', 'component', 'template', 'script',
                'data', 'methods', 'computed', 'watch'
            ],
            TechnologyStack.ANGULAR: [
                'angular', 'typescript', 'component', 'service',
                'module', 'directive', 'pipe', 'injectable'
            ],
            TechnologyStack.JAVA_SPRING: [
                'spring', 'springboot', '@controller', '@service',
                '@repository', '@entity', 'application.properties',
                'pom.xml', 'maven'
            ],
            TechnologyStack.DOTNET_CORE: [
                '.net', 'csharp', 'asp.net', 'controller', 'model',
                'view', 'startup.cs', 'program.cs', 'project.json'
            ],
            TechnologyStack.GO: [
                'go', 'golang', 'package main', 'func main',
                'gorilla', 'gin', 'echo', 'handler', 'middleware'
            ],
            TechnologyStack.RUST: [
                'rust', 'cargo.toml', 'fn main', 'impl', 'struct',
                'trait', 'mod', 'use', 'std', 'tokio'
            ]
        }
    
    async def analyze_project(self, project_path: str) -> ArchitectureAnalysis:
        """分析项目"""
        logger.info(f"🔍 开始分析项目: {project_path}")
        
        project_path = Path(project_path)
        if not project_path.exists():
            raise ValueError(f"项目路径不存在: {project_path}")
        
        # 扫描项目文件
        files = await self._scan_project_files(project_path)
        
        # 分析项目指标
        metrics = await self._analyze_project_metrics(files, project_path)
        
        # 检测项目类型
        project_type = self._detect_project_type(files, project_path)
        
        # 检测技术栈
        technology_stack = self._detect_technology_stack(files, project_path)
        
        # 分析架构组件
        components = await self._analyze_architecture_components(files, project_path, technology_stack)
        
        # 检测架构模式
        architecture_pattern = self._detect_architecture_pattern(components, project_type, technology_stack)
        
        # 生成建议和问题
        recommendations, issues, strengths = await self._generate_recommendations(
            project_type, architecture_pattern, technology_stack, components, metrics
        )
        
        analysis = ArchitectureAnalysis(
            project_type=project_type,
            architecture_pattern=architecture_pattern,
            technology_stack=technology_stack,
            components=components,
            metrics=metrics,
            recommendations=recommendations,
            issues=issues,
            strengths=strengths
        )
        
        logger.info(f"✅ 项目分析完成: {project_type.value} - {technology_stack.value}")
        return analysis
    
    async def _scan_project_files(self, project_path: Path) -> List[Path]:
        """扫描项目文件"""
        files = []
        
        # 排除的目录
        exclude_dirs = {
            '.git', '.svn', '__pycache__', 'node_modules', '.vscode',
            '.idea', 'build', 'dist', 'target', 'bin', 'obj', 'out',
            '.pytest_cache', '.coverage', 'htmlcov', '.tox', 'venv', 'env'
        }
        
        # 排除的文件扩展名
        exclude_extensions = {
            '.pyc', '.pyo', '.pyd', '.dll', '.exe', '.so', '.dylib',
            '.log', '.tmp', '.cache', '.bak', '.swp', '.swo'
        }
        
        for file_path in project_path.rglob('*'):
            if file_path.is_file():
                # 检查是否在排除目录中
                if any(exclude_dir in file_path.parts for exclude_dir in exclude_dirs):
                    continue
                
                # 检查文件扩展名
                if file_path.suffix.lower() in exclude_extensions:
                    continue
                
                files.append(file_path)
        
        logger.info(f"📁 扫描到 {len(files)} 个文件")
        return files
    
    async def _analyze_project_metrics(self, files: List[Path], project_path: Path) -> ProjectMetrics:
        """分析项目指标"""
        total_files = len(files)
        total_lines = 0
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        file_types = Counter()
        dependencies = defaultdict(list)
        
        # 分析每个文件
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                lines = content.split('\n')
                file_total_lines = len(lines)
                file_code_lines = 0
                file_comment_lines = 0
                file_blank_lines = 0
                
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        file_blank_lines += 1
                    elif stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*'):
                        file_comment_lines += 1
                    else:
                        file_code_lines += 1
                
                total_lines += file_total_lines
                code_lines += file_code_lines
                comment_lines += file_comment_lines
                blank_lines += file_blank_lines
                
                # 统计文件类型
                file_types[file_path.suffix.lower()] += 1
                
                # 提取依赖
                file_deps = self._extract_dependencies(content, file_path.suffix.lower())
                for dep in file_deps:
                    dependencies[file_path.suffix.lower()].append(dep)
                
            except Exception as e:
                logger.debug(f"分析文件失败 {file_path}: {e}")
        
        # 计算复杂度和可维护性指数
        complexity_score = self._calculate_complexity_score(code_lines, comment_lines, total_files)
        maintainability_index = self._calculate_maintainability_index(code_lines, comment_lines, complexity_score)
        
        # 估算测试覆盖率
        test_coverage = self._estimate_test_coverage(files, project_path)
        
        return ProjectMetrics(
            total_files=total_files,
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            file_types=dict(file_types),
            dependencies=dict(dependencies),
            complexity_score=complexity_score,
            maintainability_index=maintainability_index,
            test_coverage=test_coverage
        )
    
    def _extract_dependencies(self, content: str, file_extension: str) -> List[str]:
        """提取依赖"""
        dependencies = []
        
        if file_extension == '.py':
            # Python imports
            import_patterns = [
                r'^import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)',
                r'^from\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import',
                r'^from\s+\.([a-zA-Z_][a-zA-Z0-9_]*)\s+import'
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                dependencies.extend(matches)
        
        elif file_extension in ['.js', '.ts']:
            # JavaScript/TypeScript imports
            import_patterns = [
                r'^import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',
                r'^const\s+.*=\s+require\([\'"]([^\'"]+)[\'"]',
                r'^import\s+[\'"]([^\'"]+)[\'"]'
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                dependencies.extend(matches)
        
        elif file_extension == '.java':
            # Java imports
            import_pattern = r'^import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)'
            matches = re.findall(import_pattern, content, re.MULTILINE)
            dependencies.extend(matches)
        
        elif file_extension == '.cs':
            # C# using statements
            using_pattern = r'^using\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)'
            matches = re.findall(using_pattern, content, re.MULTILINE)
            dependencies.extend(matches)
        
        elif file_extension == '.go':
            # Go imports
            import_pattern = r'^import\s+[\'"]([^\'"]+)[\'"]'
            matches = re.findall(import_pattern, content, re.MULTILINE)
            dependencies.extend(matches)
        
        elif file_extension == '.rs':
            # Rust imports
            import_patterns = [
                r'^use\s+([a-zA-Z_][a-zA-Z0-9_]*(?::[a-zA-Z_][a-zA-Z0-9_]*)*)',
                r'^extern\s+crate\s+([a-zA-Z_][a-zA-Z0-9_]*)'
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                dependencies.extend(matches)
        
        return list(set(dependencies))  # 去重
    
    def _calculate_complexity_score(self, code_lines: int, comment_lines: int, file_count: int) -> float:
        """计算复杂度分数"""
        if code_lines == 0:
            return 0.0
        
        # 基础复杂度基于代码行数
        base_complexity = min(1.0, code_lines / 10000.0)
        
        # 注释比例影响
        comment_ratio = comment_lines / (code_lines + comment_lines) if (code_lines + comment_lines) > 0 else 0
        comment_factor = 1.0 - comment_ratio  # 注释越多，复杂度越低
        
        # 文件数量影响
        file_factor = min(1.0, file_count / 100.0)
        
        complexity_score = (base_complexity * 0.5 + comment_factor * 0.3 + file_factor * 0.2)
        
        return min(1.0, complexity_score)
    
    def _calculate_maintainability_index(self, code_lines: int, comment_lines: int, complexity_score: float) -> float:
        """计算可维护性指数"""
        if code_lines == 0:
            return 100.0
        
        # 简化的可维护性指数计算
        comment_ratio = comment_lines / (code_lines + comment_lines) if (code_lines + comment_lines) > 0 else 0
        
        # 基础分数
        base_score = 100.0
        
        # 复杂度影响
        complexity_penalty = complexity_score * 30.0
        
        # 注释奖励
        comment_bonus = comment_ratio * 20.0
        
        maintainability_index = base_score - complexity_penalty + comment_bonus
        
        return max(0.0, min(100.0, maintainability_index))
    
    def _estimate_test_coverage(self, files: List[Path], project_path: Path) -> float:
        """估算测试覆盖率"""
        test_files = 0
        source_files = 0
        
        for file_path in files:
            file_name = file_path.name.lower()
            file_dir = file_path.parent.name.lower()
            
            # 检查是否为测试文件
            is_test_file = (
                'test' in file_name or
                'spec' in file_name or
                file_name.startswith('test_') or
                file_name.endswith('_test') or
                file_name.endswith('_spec') or
                'tests' in file_dir or
                'test' in file_dir
            )
            
            if is_test_file:
                test_files += 1
            elif file_path.suffix in ['.py', '.js', '.ts', '.java', '.cs', '.go', '.rs']:
                source_files += 1
        
        if source_files == 0:
            return 0.0
        
        coverage = (test_files / (source_files + test_files)) * 100
        return min(100.0, coverage)
    
    def _detect_project_type(self, files: List[Path], project_path: Path) -> ProjectType:
        """检测项目类型"""
        type_scores = {}
        
        # 检查每个项目类型的指示器
        for project_type, indicators in self.project_indicators.items():
            score = 0
            
            # 检查文件名
            for file_path in files:
                file_name = file_path.name.lower()
                file_path_str = str(file_path).lower()
                
                for indicator in indicators:
                    if indicator in file_name or indicator in file_path_str:
                        score += 1
            
            # 检查目录名
            for dir_path in project_path.rglob('*'):
                if dir_path.is_dir():
                    dir_name = dir_path.name.lower()
                    for indicator in indicators:
                        if indicator in dir_name:
                            score += 2  # 目录名权重更高
            
            type_scores[project_type] = score
        
        # 返回得分最高的项目类型
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 0:
                return best_type
        
        return ProjectType.UNKNOWN
    
    def _detect_technology_stack(self, files: List[Path], project_path: Path) -> TechnologyStack:
        """检测技术栈"""
        stack_scores = {}
        
        # 检查每个技术栈的指示器
        for tech_stack, indicators in self.technology_indicators.items():
            score = 0
            
            # 检查文件内容
            for file_path in files[:20]:  # 限制检查的文件数量
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().lower()
                    
                    for indicator in indicators:
                        score += content.count(indicator)
                        
                except Exception:
                    continue
            
            # 检查文件名
            for file_path in files:
                file_name = file_path.name.lower()
                file_path_str = str(file_path).lower()
                
                for indicator in indicators:
                    if indicator in file_name or indicator in file_path_str:
                        score += 5  # 文件名权重更高
            
            stack_scores[tech_stack] = score
        
        # 返回得分最高的技术栈
        if stack_scores:
            best_stack = max(stack_scores, key=stack_scores.get)
            if stack_scores[best_stack] > 0:
                return best_stack
        
        return TechnologyStack.UNKNOWN
    
    async def _analyze_architecture_components(self, files: List[Path], project_path: Path, technology_stack: TechnologyStack) -> List[ArchitectureComponent]:
        """分析架构组件"""
        components = []
        
        # 根据技术栈选择合适的分析器
        if technology_stack in [TechnologyStack.PYTHON_DJANGO, TechnologyStack.PYTHON_FASTAPI, TechnologyStack.PYTHON_FLASK]:
            components.extend(await self._analyze_python_components(files))
        elif technology_stack == TechnologyStack.NODE_EXPRESS:
            components.extend(await self._analyze_nodejs_components(files))
        elif technology_stack == TechnologyStack.JAVA_SPRING:
            components.extend(await self._analyze_java_components(files))
        elif technology_stack == TechnologyStack.DOTNET_CORE:
            components.extend(await self._analyze_csharp_components(files))
        elif technology_stack == TechnologyStack.GO:
            components.extend(await self._analyze_go_components(files))
        elif technology_stack == TechnologyStack.RUST:
            components.extend(await self._analyze_rust_components(files))
        else:
            # 通用分析
            components.extend(await self._analyze_generic_components(files))
        
        return components
    
    async def _analyze_python_components(self, files: List[Path]) -> List[ArchitectureComponent]:
        """分析Python组件"""
        components = []
        
        for file_path in files:
            if file_path.suffix != '.py':
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 解析AST
                try:
                    tree = ast.parse(content)
                except SyntaxError:
                    continue
                
                # 分析类和函数
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        component = ArchitectureComponent(
                            component_id=f"class_{node.name}_{file_path.name}",
                            name=node.name,
                            type="class",
                            file_path=str(file_path),
                            dependencies=self._extract_python_dependencies(node),
                            complexity=self._calculate_python_complexity(node),
                            responsibilities=self._extract_python_responsibilities(node),
                            interfaces=self._extract_python_interfaces(node)
                        )
                        components.append(component)
                    
                    elif isinstance(node, ast.FunctionDef):
                        # 检查是否为视图函数、控制器等
                        func_type = self._classify_python_function(node, content)
                        
                        component = ArchitectureComponent(
                            component_id=f"function_{node.name}_{file_path.name}",
                            name=node.name,
                            type=func_type,
                            file_path=str(file_path),
                            dependencies=self._extract_python_dependencies(node),
                            complexity=self._calculate_python_complexity(node),
                            responsibilities=[f"实现{node.name}功能"],
                            interfaces=[]
                        )
                        components.append(component)
                
            except Exception as e:
                logger.debug(f"分析Python文件失败 {file_path}: {e}")
        
        return components
    
    def _extract_python_dependencies(self, node) -> List[str]:
        """提取Python依赖"""
        dependencies = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    dependencies.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    dependencies.append(child.func.attr)
        
        return list(set(dependencies))
    
    def _calculate_python_complexity(self, node) -> float:
        """计算Python复杂度"""
        complexity = 1.0
        
        # 计算圈复杂度
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return min(10.0, complexity)
    
    def _extract_python_responsibilities(self, node) -> List[str]:
        """提取Python职责"""
        responsibilities = []
        
        # 从文档字符串提取职责
        if (hasattr(node, 'body') and node.body and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Str)):
            docstring = node.body[0].value.s
            responsibilities.append(docstring)
        
        # 从方法名推断职责
        if hasattr(node, 'name'):
            name = node.name.lower()
            if 'get' in name:
                responsibilities.append("数据获取")
            elif 'set' in name:
                responsibilities.append("数据设置")
            elif 'process' in name:
                responsibilities.append("数据处理")
            elif 'validate' in name:
                responsibilities.append("数据验证")
        
        return responsibilities
    
    def _extract_python_interfaces(self, node) -> List[str]:
        """提取Python接口"""
        interfaces = []
        
        # 从基类提取接口
        if hasattr(node, 'bases'):
            for base in node.bases:
                if isinstance(base, ast.Name):
                    interfaces.append(base.id)
        
        return interfaces
    
    def _classify_python_function(self, node, content: str) -> str:
        """分类Python函数"""
        name = node.name.lower()
        
        # Django视图函数
        if 'request' in content or 'HttpResponse' in content:
            return "view"
        
        # API函数
        if 'api' in name or 'endpoint' in name:
            return "api"
        
        # 控制器函数
        if 'controller' in name or 'handler' in name:
            return "controller"
        
        # 服务函数
        if 'service' in name or 'business' in name:
            return "service"
        
        # 模型函数
        if 'model' in name or 'entity' in name:
            return "model"
        
        # 工具函数
        if 'util' in name or 'helper' in name:
            return "utility"
        
        return "function"
    
    async def _analyze_nodejs_components(self, files: List[Path]) -> List[ArchitectureComponent]:
        """分析Node.js组件"""
        components = []
        
        for file_path in files:
            if file_path.suffix not in ['.js', '.ts']:
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 使用正则表达式分析
                # 函数定义
                func_pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:function|\([^)]*\)\s*=>))'
                func_matches = re.findall(func_pattern, content)
                
                for match in func_matches:
                    func_name = match[0] or match[1]
                    
                    component = ArchitectureComponent(
                        component_id=f"function_{func_name}_{file_path.name}",
                        name=func_name,
                        type=self._classify_js_function(func_name, content),
                        file_path=str(file_path),
                        dependencies=self._extract_js_dependencies(content),
                        complexity=self._calculate_js_complexity(content),
                        responsibilities=[f"实现{func_name}功能"],
                        interfaces=[]
                    )
                    components.append(component)
                
                # 类定义（TypeScript）
                if file_path.suffix == '.ts':
                    class_pattern = r'class\s+(\w+)'
                    class_matches = re.findall(class_pattern, content)
                    
                    for class_name in class_matches:
                        component = ArchitectureComponent(
                            component_id=f"class_{class_name}_{file_path.name}",
                            name=class_name,
                            type="class",
                            file_path=str(file_path),
                            dependencies=self._extract_js_dependencies(content),
                            complexity=self._calculate_js_complexity(content),
                            responsibilities=[f"{class_name}类"],
                            interfaces=[]
                        )
                        components.append(component)
                
            except Exception as e:
                logger.debug(f"分析Node.js文件失败 {file_path}: {e}")
        
        return components
    
    def _extract_js_dependencies(self, content: str) -> List[str]:
        """提取JavaScript依赖"""
        dependencies = []
        
        # 函数调用
        call_pattern = r'(\w+)\s*\('
        matches = re.findall(call_pattern, content)
        dependencies.extend(matches)
        
        return list(set(dependencies))
    
    def _calculate_js_complexity(self, content: str) -> float:
        """计算JavaScript复杂度"""
        complexity = 1.0
        
        # 计算圈复杂度
        complexity_keywords = ['if', 'else if', 'for', 'while', 'switch', 'case', 'catch']
        for keyword in complexity_keywords:
            complexity += content.count(keyword)
        
        return min(10.0, complexity)
    
    def _classify_js_function(self, name: str, content: str) -> str:
        """分类JavaScript函数"""
        name_lower = name.lower()
        
        if 'controller' in name_lower or 'handler' in name_lower:
            return "controller"
        elif 'service' in name_lower or 'business' in name_lower:
            return "service"
        elif 'model' in name_lower or 'entity' in name_lower:
            return "model"
        elif 'util' in name_lower or 'helper' in name_lower:
            return "utility"
        elif 'middleware' in name_lower:
            return "middleware"
        elif 'route' in name_lower:
            return "route"
        else:
            return "function"
    
    async def _analyze_java_components(self, files: List[Path]) -> List[ArchitectureComponent]:
        """分析Java组件"""
        components = []
        
        for file_path in files:
            if file_path.suffix != '.java':
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 类定义
                class_pattern = r'(?:public\s+|private\s+|protected\s+)?(?:abstract\s+|final\s+)?class\s+(\w+)'
                class_matches = re.findall(class_pattern, content)
                
                for class_name in class_matches:
                    component = ArchitectureComponent(
                        component_id=f"class_{class_name}_{file_path.name}",
                        name=class_name,
                        type=self._classify_java_class(class_name, content),
                        file_path=str(file_path),
                        dependencies=self._extract_java_dependencies(content),
                        complexity=self._calculate_java_complexity(content),
                        responsibilities=[f"{class_name}类"],
                        interfaces=self._extract_java_interfaces(content)
                    )
                    components.append(component)
                
                # 接口定义
                interface_pattern = r'(?:public\s+|private\s+|protected\s+)?interface\s+(\w+)'
                interface_matches = re.findall(interface_pattern, content)
                
                for interface_name in interface_matches:
                    component = ArchitectureComponent(
                        component_id=f"interface_{interface_name}_{file_path.name}",
                        name=interface_name,
                        type="interface",
                        file_path=str(file_path),
                        dependencies=self._extract_java_dependencies(content),
                        complexity=self._calculate_java_complexity(content),
                        responsibilities=[f"{interface_name}接口"],
                        interfaces=[]
                    )
                    components.append(component)
                
            except Exception as e:
                logger.debug(f"分析Java文件失败 {file_path}: {e}")
        
        return components
    
    def _extract_java_dependencies(self, content: str) -> List[str]:
        """提取Java依赖"""
        dependencies = []
        
        # 方法调用
        method_pattern = r'(\w+)\s*\('
        matches = re.findall(method_pattern, content)
        dependencies.extend(matches)
        
        return list(set(dependencies))
    
    def _calculate_java_complexity(self, content: str) -> float:
        """计算Java复杂度"""
        complexity = 1.0
        
        # 计算圈复杂度
        complexity_keywords = ['if', 'else if', 'for', 'while', 'switch', 'case', 'catch']
        for keyword in complexity_keywords:
            complexity += content.count(keyword)
        
        return min(10.0, complexity)
    
    def _extract_java_interfaces(self, content: str) -> List[str]:
        """提取Java接口"""
        interfaces = []
        
        # 实现的接口
        implements_pattern = r'implements\s+([^\{]+)'
        match = re.search(implements_pattern, content)
        if match:
            interfaces.extend([i.strip() for i in match.group(1).split(',')])
        
        return interfaces
    
    def _classify_java_class(self, name: str, content: str) -> str:
        """分类Java类"""
        name_lower = name.lower()
        
        if 'controller' in name_lower or '@Controller' in content:
            return "controller"
        elif 'service' in name_lower or '@Service' in content:
            return "service"
        elif 'repository' in name_lower or '@Repository' in content:
            return "repository"
        elif 'entity' in name_lower or '@Entity' in content:
            return "entity"
        elif 'model' in name_lower:
            return "model"
        elif 'config' in name_lower or '@Configuration' in content:
            return "configuration"
        else:
            return "class"
    
    async def _analyze_csharp_components(self, files: List[Path]) -> List[ArchitectureComponent]:
        """分析C#组件"""
        # 类似Java的分析逻辑
        return await self._analyze_java_components(files)
    
    async def _analyze_go_components(self, files: List[Path]) -> List[ArchitectureComponent]:
        """分析Go组件"""
        components = []
        
        for file_path in files:
            if file_path.suffix != '.go':
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 结构体定义
                struct_pattern = r'type\s+(\w+)\s+struct'
                struct_matches = re.findall(struct_pattern, content)
                
                for struct_name in struct_matches:
                    component = ArchitectureComponent(
                        component_id=f"struct_{struct_name}_{file_path.name}",
                        name=struct_name,
                        type="struct",
                        file_path=str(file_path),
                        dependencies=self._extract_go_dependencies(content),
                        complexity=self._calculate_go_complexity(content),
                        responsibilities=[f"{struct_name}结构体"],
                        interfaces=[]
                    )
                    components.append(component)
                
                # 接口定义
                interface_pattern = r'type\s+(\w+)\s+interface'
                interface_matches = re.findall(interface_pattern, content)
                
                for interface_name in interface_matches:
                    component = ArchitectureComponent(
                        component_id=f"interface_{interface_name}_{file_path.name}",
                        name=interface_name,
                        type="interface",
                        file_path=str(file_path),
                        dependencies=self._extract_go_dependencies(content),
                        complexity=self._calculate_go_complexity(content),
                        responsibilities=[f"{interface_name}接口"],
                        interfaces=[]
                    )
                    components.append(component)
                
                # 函数定义
                func_pattern = r'func\s+(?:\([^)]*\)\s*)?(\w+)'
                func_matches = re.findall(func_pattern, content)
                
                for func_name in func_matches:
                    component = ArchitectureComponent(
                        component_id=f"function_{func_name}_{file_path.name}",
                        name=func_name,
                        type=self._classify_go_function(func_name, content),
                        file_path=str(file_path),
                        dependencies=self._extract_go_dependencies(content),
                        complexity=self._calculate_go_complexity(content),
                        responsibilities=[f"实现{func_name}功能"],
                        interfaces=[]
                    )
                    components.append(component)
                
            except Exception as e:
                logger.debug(f"分析Go文件失败 {file_path}: {e}")
        
        return components
    
    def _extract_go_dependencies(self, content: str) -> List[str]:
        """提取Go依赖"""
        dependencies = []
        
        # 函数调用
        call_pattern = r'(\w+)\s*\('
        matches = re.findall(call_pattern, content)
        dependencies.extend(matches)
        
        return list(set(dependencies))
    
    def _calculate_go_complexity(self, content: str) -> float:
        """计算Go复杂度"""
        complexity = 1.0
        
        # 计算圈复杂度
        complexity_keywords = ['if', 'else if', 'for', 'switch', 'case', 'select']
        for keyword in complexity_keywords:
            complexity += content.count(keyword)
        
        return min(10.0, complexity)
    
    def _classify_go_function(self, name: str, content: str) -> str:
        """分类Go函数"""
        name_lower = name.lower()
        
        if 'handler' in name_lower:
            return "handler"
        elif 'service' in name_lower:
            return "service"
        elif 'process' in name_lower:
            return "processor"
        elif 'validate' in name_lower:
            return "validator"
        else:
            return "function"
    
    async def _analyze_rust_components(self, files: List[Path]) -> List[ArchitectureComponent]:
        """分析Rust组件"""
        components = []
        
        for file_path in files:
            if file_path.suffix != '.rs':
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 结构体定义
                struct_pattern = r'struct\s+(\w+)'
                struct_matches = re.findall(struct_pattern, content)
                
                for struct_name in struct_matches:
                    component = ArchitectureComponent(
                        component_id=f"struct_{struct_name}_{file_path.name}",
                        name=struct_name,
                        type="struct",
                        file_path=str(file_path),
                        dependencies=self._extract_rust_dependencies(content),
                        complexity=self._calculate_rust_complexity(content),
                        responsibilities=[f"{struct_name}结构体"],
                        interfaces=self._extract_rust_traits(content)
                    )
                    components.append(component)
                
                # 特征定义
                trait_pattern = r'trait\s+(\w+)'
                trait_matches = re.findall(trait_pattern, content)
                
                for trait_name in trait_matches:
                    component = ArchitectureComponent(
                        component_id=f"trait_{trait_name}_{file_path.name}",
                        name=trait_name,
                        type="trait",
                        file_path=str(file_path),
                        dependencies=self._extract_rust_dependencies(content),
                        complexity=self._calculate_rust_complexity(content),
                        responsibilities=[f"{trait_name}特征"],
                        interfaces=[]
                    )
                    components.append(component)
                
                # 函数定义
                func_pattern = r'fn\s+(\w+)'
                func_matches = re.findall(func_pattern, content)
                
                for func_name in func_matches:
                    component = ArchitectureComponent(
                        component_id=f"function_{func_name}_{file_path.name}",
                        name=func_name,
                        type=self._classify_rust_function(func_name, content),
                        file_path=str(file_path),
                        dependencies=self._extract_rust_dependencies(content),
                        complexity=self._calculate_rust_complexity(content),
                        responsibilities=[f"实现{func_name}功能"],
                        interfaces=[]
                    )
                    components.append(component)
                
            except Exception as e:
                logger.debug(f"分析Rust文件失败 {file_path}: {e}")
        
        return components
    
    def _extract_rust_dependencies(self, content: str) -> List[str]:
        """提取Rust依赖"""
        dependencies = []
        
        # 函数调用
        call_pattern = r'(\w+)::'
        matches = re.findall(call_pattern, content)
        dependencies.extend(matches)
        
        return list(set(dependencies))
    
    def _calculate_rust_complexity(self, content: str) -> float:
        """计算Rust复杂度"""
        complexity = 1.0
        
        # 计算圈复杂度
        complexity_keywords = ['if', 'else if', 'for', 'while', 'match', 'loop']
        for keyword in complexity_keywords:
            complexity += content.count(keyword)
        
        return min(10.0, complexity)
    
    def _extract_rust_traits(self, content: str) -> List[str]:
        """提取Rust特征"""
        traits = []
        
        # 实现的特征
        impl_pattern = r'impl\s+(\w+)\s+for\s+\w+'
        matches = re.findall(impl_pattern, content)
        traits.extend(matches)
        
        return traits
    
    def _classify_rust_function(self, name: str, content: str) -> str:
        """分类Rust函数"""
        name_lower = name.lower()
        
        if 'main' in name_lower:
            return "main"
        elif 'new' in name_lower:
            return "constructor"
        elif 'process' in name_lower:
            return "processor"
        elif 'validate' in name_lower:
            return "validator"
        else:
            return "function"
    
    async def _analyze_generic_components(self, files: List[Path]) -> List[ArchitectureComponent]:
        """通用组件分析"""
        components = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # 简单的组件分析
                component = ArchitectureComponent(
                    component_id=f"file_{file_path.name}",
                    name=file_path.stem,
                    type="file",
                    file_path=str(file_path),
                    dependencies=[],
                    complexity=1.0,
                    responsibilities=[f"{file_path.name}文件"],
                    interfaces=[]
                )
                components.append(component)
                
            except Exception as e:
                logger.debug(f"分析文件失败 {file_path}: {e}")
        
        return components
    
    def _detect_architecture_pattern(self, components: List[ArchitectureComponent], project_type: ProjectType, technology_stack: TechnologyStack) -> ArchitecturePattern:
        """检测架构模式"""
        pattern_scores = {}
        
        # 单体架构
        monolith_indicators = [
            len(components) < 20,
            not any(c.type == "microservice" for c in components),
            project_type in [ProjectType.WEB_APP, ProjectType.DESKTOP_APP]
        ]
        pattern_scores[ArchitecturePattern.MONOLITH] = sum(monolith_indicators)
        
        # 微服务架构
        microservice_indicators = [
            len(components) >= 20,
            any("service" in c.name.lower() or "microservice" in c.name.lower() for c in components),
            any("docker" in c.file_path.lower() for c in components),
            project_type in [ProjectType.API_SERVICE, ProjectType.MICROSERVICE]
        ]
        pattern_scores[ArchitecturePattern.MICROSERVICES] = sum(microservice_indicators)
        
        # 事件驱动架构
        event_driven_indicators = [
            any("event" in c.name.lower() or "message" in c.name.lower() for c in components),
            any("queue" in c.file_path.lower() or "kafka" in c.file_path.lower() for c in components),
            any("publisher" in c.name.lower() or "subscriber" in c.name.lower() for c in components)
        ]
        pattern_scores[ArchitecturePattern.EVENT_DRIVEN] = sum(event_driven_indicators)
        
        # 分层架构
        layered_indicators = [
            any("controller" in c.type for c in components),
            any("service" in c.type for c in components),
            any("repository" in c.type for c in components),
            any("model" in c.type for c in components)
        ]
        pattern_scores[ArchitecturePattern.LAYERED] = sum(layered_indicators)
        
        # MVC架构
        mvc_indicators = [
            any("controller" in c.type for c in components),
            any("model" in c.type for c in components),
            any("view" in c.type or "template" in c.type for c in components),
            technology_stack in [TechnologyStack.PYTHON_DJANGO, TechnologyStack.NODE_EXPRESS]
        ]
        pattern_scores[ArchitecturePattern.MVC] = sum(mvc_indicators)
        
        # 返回得分最高的架构模式
        if pattern_scores:
            best_pattern = max(pattern_scores, key=pattern_scores.get)
            if pattern_scores[best_pattern] > 0:
                return best_pattern
        
        return ArchitecturePattern.UNKNOWN
    
    async def _generate_recommendations(self, project_type: ProjectType, architecture_pattern: ArchitecturePattern, 
                                       technology_stack: TechnologyStack, components: List[ArchitectureComponent], 
                                       metrics: ProjectMetrics) -> Tuple[List[str], List[str], List[str]]:
        """生成建议、问题和优势"""
        recommendations = []
        issues = []
        strengths = []
        
        # 基于指标的建议
        if metrics.complexity_score > 0.7:
            recommendations.append("考虑重构高复杂度的代码，提高可维护性")
            issues.append(f"代码复杂度过高 ({metrics.complexity_score:.2f})")
        
        if metrics.maintainability_index < 60:
            recommendations.append("增加代码注释和文档，提高可维护性")
            issues.append(f"可维护性指数较低 ({metrics.maintainability_index:.2f})")
        
        if metrics.test_coverage < 30:
            recommendations.append("增加单元测试和集成测试，提高测试覆盖率")
            issues.append(f"测试覆盖率过低 ({metrics.test_coverage:.1f}%)")
        
        # 基于架构的建议
        if architecture_pattern == ArchitecturePattern.UNKNOWN:
            recommendations.append("明确架构模式，提高代码组织性")
            issues.append("架构模式不明确")
        
        # 基于组件的建议
        if len(components) > 50:
            recommendations.append("考虑拆分大型项目，采用微服务架构")
        
        high_complexity_components = [c for c in components if c.complexity > 7]
        if high_complexity_components:
            recommendations.append(f"重构 {len(high_complexity_components)} 个高复杂度组件")
        
        # 优势
        if metrics.maintainability_index > 80:
            strengths.append("代码可维护性良好")
        
        if metrics.test_coverage > 70:
            strengths.append("测试覆盖率较高")
        
        if architecture_pattern != ArchitecturePattern.UNKNOWN:
            strengths.append(f"采用清晰的 {architecture_pattern.value} 架构模式")
        
        if technology_stack != TechnologyStack.UNKNOWN:
            strengths.append(f"使用成熟的 {technology_stack.value} 技术栈")
        
        return recommendations, issues, strengths
    
    def _analyze_python_file(self, file_path: Path) -> Dict[str, Any]:
        """分析Python文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            analysis = {
                'classes': [],
                'functions': [],
                'imports': [],
                'complexity': 0
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    analysis['classes'].append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    analysis['functions'].append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        analysis['imports'].append(f"{module}.{alias.name}")
            
            return analysis
            
        except Exception as e:
            logger.debug(f"分析Python文件失败 {file_path}: {e}")
            return {}
    
    def _analyze_javascript_file(self, file_path: Path) -> Dict[str, Any]:
        """分析JavaScript文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = {
                'functions': [],
                'classes': [],
                'imports': [],
                'complexity': 0
            }
            
            # 函数定义
            func_pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:function|\([^)]*\)\s*=>))'
            func_matches = re.findall(func_pattern, content)
            for match in func_matches:
                func_name = match[0] or match[1]
                analysis['functions'].append(func_name)
            
            # 类定义（ES6+）
            class_pattern = r'class\s+(\w+)'
            class_matches = re.findall(class_pattern, content)
            analysis['classes'].extend(class_matches)
            
            # 导入
            import_pattern = r'(?:import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]|const\s+.*=\s+require\([\'"]([^\'"]+)[\'"])'
            import_matches = re.findall(import_pattern, content)
            for match in import_matches:
                module = match[0] or match[1]
                analysis['imports'].append(module)
            
            return analysis
            
        except Exception as e:
            logger.debug(f"分析JavaScript文件失败 {file_path}: {e}")
            return {}
    
    def _analyze_typescript_file(self, file_path: Path) -> Dict[str, Any]:
        """分析TypeScript文件"""
        # 类似JavaScript的分析，加上TypeScript特有的分析
        return self._analyze_javascript_file(file_path)
    
    def _analyze_java_file(self, file_path: Path) -> Dict[str, Any]:
        """分析Java文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = {
                'classes': [],
                'interfaces': [],
                'methods': [],
                'imports': [],
                'complexity': 0
            }
            
            # 类定义
            class_pattern = r'(?:public\s+|private\s+|protected\s+)?(?:abstract\s+|final\s+)?class\s+(\w+)'
            class_matches = re.findall(class_pattern, content)
            analysis['classes'].extend(class_matches)
            
            # 接口定义
            interface_pattern = r'(?:public\s+|private\s+|protected\s+)?interface\s+(\w+)'
            interface_matches = re.findall(interface_pattern, content)
            analysis['interfaces'].extend(interface_matches)
            
            # 方法定义
            method_pattern = r'(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:final\s+)?(?:abstract\s+)?(?:\w+\s+)?(\w+)\s*\([^)]*\)'
            method_matches = re.findall(method_pattern, content)
            analysis['methods'].extend(method_matches)
            
            # 导入
            import_pattern = r'import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)'
            import_matches = re.findall(import_pattern, content)
            analysis['imports'].extend(import_matches)
            
            return analysis
            
        except Exception as e:
            logger.debug(f"分析Java文件失败 {file_path}: {e}")
            return {}
    
    def _analyze_csharp_file(self, file_path: Path) -> Dict[str, Any]:
        """分析C#文件"""
        # 类似Java的分析
        return self._analyze_java_file(file_path)
    
    def _analyze_go_file(self, file_path: Path) -> Dict[str, Any]:
        """分析Go文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = {
                'structs': [],
                'interfaces': [],
                'functions': [],
                'imports': [],
                'complexity': 0
            }
            
            # 结构体定义
            struct_pattern = r'type\s+(\w+)\s+struct'
            struct_matches = re.findall(struct_pattern, content)
            analysis['structs'].extend(struct_matches)
            
            # 接口定义
            interface_pattern = r'type\s+(\w+)\s+interface'
            interface_matches = re.findall(interface_pattern, content)
            analysis['interfaces'].extend(interface_matches)
            
            # 函数定义
            func_pattern = r'func\s+(?:\([^)]*\)\s*)?(\w+)'
            func_matches = re.findall(func_pattern, content)
            analysis['functions'].extend(func_matches)
            
            # 导入
            import_pattern = r'import\s+[\'"]([^\'"]+)[\'"]'
            import_matches = re.findall(import_pattern, content)
            analysis['imports'].extend(import_matches)
            
            return analysis
            
        except Exception as e:
            logger.debug(f"分析Go文件失败 {file_path}: {e}")
            return {}
    
    def _analyze_rust_file(self, file_path: Path) -> Dict[str, Any]:
        """分析Rust文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = {
                'structs': [],
                'traits': [],
                'functions': [],
                'imports': [],
                'complexity': 0
            }
            
            # 结构体定义
            struct_pattern = r'struct\s+(\w+)'
            struct_matches = re.findall(struct_pattern, content)
            analysis['structs'].extend(struct_matches)
            
            # 特征定义
            trait_pattern = r'trait\s+(\w+)'
            trait_matches = re.findall(trait_pattern, content)
            analysis['traits'].extend(trait_matches)
            
            # 函数定义
            func_pattern = r'fn\s+(\w+)'
            func_matches = re.findall(func_pattern, content)
            analysis['functions'].extend(func_matches)
            
            # 导入
            import_pattern = r'use\s+([a-zA-Z_][a-zA-Z0-9_]*(?::[a-zA-Z_][a-zA-Z0-9_]*)*)'
            import_matches = re.findall(import_pattern, content)
            analysis['imports'].extend(import_matches)
            
            return analysis
            
        except Exception as e:
            logger.debug(f"分析Rust文件失败 {file_path}: {e}")
            return {}
    
    def _analyze_cpp_file(self, file_path: Path) -> Dict[str, Any]:
        """分析C++文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = {
                'classes': [],
                'functions': [],
                'includes': [],
                'complexity': 0
            }
            
            # 类定义
            class_pattern = r'class\s+(\w+)'
            class_matches = re.findall(class_pattern, content)
            analysis['classes'].extend(class_matches)
            
            # 函数定义
            func_pattern = r'(?:\w+\s+)?(\w+)\s*\([^)]*\)\s*(?:const\s*)?{'
            func_matches = re.findall(func_pattern, content)
            analysis['functions'].extend(func_matches)
            
            # 包含文件
            include_pattern = r'#include\s*[<"]([^>"]+)[>"]'
            include_matches = re.findall(include_pattern, content)
            analysis['includes'].extend(include_matches)
            
            return analysis
            
        except Exception as e:
            logger.debug(f"分析C++文件失败 {file_path}: {e}")
            return {}
    
    def _analyze_json_file(self, file_path: Path) -> Dict[str, Any]:
        """分析JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            data = json.loads(content)
            
            analysis = {
                'type': 'json',
                'keys': list(data.keys()) if isinstance(data, dict) else [],
                'size': len(content),
                'structure': type(data).__name__
            }
            
            return analysis
            
        except Exception as e:
            logger.debug(f"分析JSON文件失败 {file_path}: {e}")
            return {}
    
    def _analyze_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """分析YAML文件"""
        try:
            import yaml
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            data = yaml.safe_load(content)
            
            analysis = {
                'type': 'yaml',
                'keys': list(data.keys()) if isinstance(data, dict) else [],
                'size': len(content),
                'structure': type(data).__name__
            }
            
            return analysis
            
        except Exception as e:
            logger.debug(f"分析YAML文件失败 {file_path}: {e}")
            return {}
    
    def _analyze_markdown_file(self, file_path: Path) -> Dict[str, Any]:
        """分析Markdown文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            analysis = {
                'type': 'markdown',
                'headings': [],
                'links': [],
                'size': len(content),
                'lines': len(lines)
            }
            
            # 提取标题
            for line in lines:
                if line.startswith('#'):
                    level = len(line) - len(line.lstrip('#'))
                    title = line.lstrip('# ').strip()
                    analysis['headings'].append({'level': level, 'title': title})
            
            # 提取链接
            link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
            link_matches = re.findall(link_pattern, content)
            analysis['links'].extend(link_matches)
            
            return analysis
            
        except Exception as e:
            logger.debug(f"分析Markdown文件失败 {file_path}: {e}")
            return {}
    
    def _analyze_text_file(self, file_path: Path) -> Dict[str, Any]:
        """分析文本文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            analysis = {
                'type': 'text',
                'size': len(content),
                'lines': len(lines),
                'words': len(content.split())
            }
            
            return analysis
            
        except Exception as e:
            logger.debug(f"分析文本文件失败 {file_path}: {e}")
            return {}

# 全局项目分析器实例
project_analyzer = ProjectAnalyzer()

# 便捷函数
async def analyze_project(project_path: str) -> ArchitectureAnalysis:
    """便捷的项目分析函数"""
    return await project_analyzer.analyze_project(project_path)

# 示例使用
async def example_usage():
    """示例使用"""
    print("🔍 项目分析器示例")
    
    # 分析当前项目
    current_path = Path.cwd()
    print(f"\n1. 分析当前项目: {current_path}")
    
    try:
        analysis = await analyze_project(str(current_path))
        
        print(f"项目类型: {analysis.project_type.value}")
        print(f"架构模式: {analysis.architecture_pattern.value}")
        print(f"技术栈: {analysis.technology_stack.value}")
        print(f"组件数量: {len(analysis.components)}")
        
        print(f"\n项目指标:")
        print(f"  总文件数: {analysis.metrics.total_files}")
        print(f"  总代码行数: {analysis.metrics.code_lines}")
        print(f"  复杂度分数: {analysis.metrics.complexity_score:.2f}")
        print(f"  可维护性指数: {analysis.metrics.maintainability_index:.2f}")
        print(f"  测试覆盖率: {analysis.metrics.test_coverage:.1f}%")
        
        print(f"\n建议:")
        for rec in analysis.recommendations[:5]:
            print(f"  • {rec}")
        
        print(f"\n问题:")
        for issue in analysis.issues[:5]:
            print(f"  • {issue}")
        
        print(f"\n优势:")
        for strength in analysis.strengths[:5]:
            print(f"  • {strength}")
        
    except Exception as e:
        print(f"❌ 项目分析失败: {e}")
    
    print("\n✅ 项目分析器示例完成")

if __name__ == "__main__":
    asyncio.run(example_usage())