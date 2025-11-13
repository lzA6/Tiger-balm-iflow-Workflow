#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全能工作流V5 - 项目分析器
Universal Workflow V5 - Project Analyzer

自动识别项目架构、技术栈、复杂度和难度
"""

import os
import sys
import json
import yaml
import ast
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import fnmatch
import re

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

@dataclass
class ProjectMetrics:
    """项目指标"""
    total_files: int = 0
    total_lines: int = 0
    code_files: int = 0
    test_files: int = 0
    doc_files: int = 0
    config_files: int = 0
    asset_files: int = 0
    
    # 语言分布
    language_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # 框架检测
    frameworks: List[str] = field(default_factory=list)
    
    # 依赖统计
    dependencies: List[str] = field(default_factory=list)
    
    # 复杂度指标
    cyclomatic_complexity: float = 0.0
    code_duplication: float = 0.0
    test_coverage: float = 0.0
    
    # 架构指标
    architecture_score: float = 0.0
    modularity_score: float = 0.0
    coupling_score: float = 0.0

@dataclass
class ArchitecturePattern:
    """架构模式"""
    name: str
    pattern_type: str  # monolithic, microservices, layered, modular
    indicators: List[str]
    confidence: float = 0.0
    files_involved: List[str] = field(default_factory=list)

@dataclass
class ProjectAnalysis:
    """项目分析结果"""
    project_path: str
    project_name: str
    project_type: str
    primary_language: str
    frameworks: List[str]
    architecture: str
    complexity_score: float
    difficulty_level: str  # easy, medium, hard, expert
    estimated_effort: str  # hours, days, weeks, months
    recommendations: List[str]
    metrics: ProjectMetrics
    patterns: List[ArchitecturePattern] = field(default_factory=list)
    analysis_time: datetime = field(default_factory=datetime.now)

class ProjectAnalyzer:
    """项目分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.patterns = self._load_architecture_patterns()
        
        # 语言检测规则
        self.language_rules = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.jsx': 'React',
            '.tsx': 'React',
            '.java': 'Java',
            '.kt': 'Kotlin',
            '.swift': 'Swift',
            '.go': 'Go',
            '.rs': 'Rust',
            '.cpp': 'C++',
            '.c': 'C',
            '.cs': 'C#',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.scala': 'Scala',
            '.r': 'R',
            '.m': 'MATLAB',
            '.sh': 'Shell',
            '.sql': 'SQL',
            '.html': 'HTML',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.sass': 'SASS',
            '.less': 'LESS'
        }
        
        # 框架检测规则
        self.framework_rules = {
            # JavaScript/TypeScript
            'package.json': self._detect_js_framework,
            'yarn.lock': 'Yarn',
            'package-lock.json': 'npm',
            'tsconfig.json': 'TypeScript',
            'webpack.config.js': 'Webpack',
            'vite.config.js': 'Vite',
            'rollup.config.js': 'Rollup',
            '.nuxt': 'Nuxt.js',
            'next.config.js': 'Next.js',
            'gatsby.config.js': 'Gatsby',
            'angular.json': 'Angular',
            'vue.config.js': 'Vue',
            'svelte.config.js': 'Svelte',
            
            # Python
            'requirements.txt': 'Python',
            'setup.py': 'Python',
            'pyproject.toml': 'Python',
            'Pipfile': 'Python',
            'poetry.lock': 'Poetry',
            'environment.yml': 'Ansible',
            'Dockerfile': 'Docker',
            'docker-compose.yml': 'Docker Compose',
            'manage.py': 'Django',
            'app.py': 'Flask',
            'main.py': 'Generic Python',
            
            # Java
            'pom.xml': 'Maven',
            'build.gradle': 'Gradle',
            'settings.gradle': 'Gradle',
            'application.properties': 'Spring Boot',
            'AndroidManifest.xml': 'Android',
            
            # Go
            'go.mod': 'Go',
            'go.sum': 'Go',
            'main.go': 'Go',
            
            # Rust
            'Cargo.toml': 'Rust',
            'src/main.rs': 'Rust',
            
            # C#
            '.csproj': 'C#',
            'Program.cs': 'C#',
            
            # Ruby
            'Gemfile': 'Ruby',
            'Rakefile': 'Ruby',
            
            # PHP
            'composer.json': 'PHP',
            'index.php': 'PHP',
            
            # Swift
            'Package.swift': 'Swift',
            'Podfile': 'Swift',
            
            # Kotlin
            'build.gradle.kts': 'Kotlin',
            'MainActivity.kt': 'Kotlin'
        }
        
        logger.info("项目分析器初始化完成")
    
    def _load_architecture_patterns(self) -> List[ArchitecturePattern]:
        """加载架构模式"""
        patterns = [
            ArchitecturePattern(
                name="Monolithic",
                pattern_type="monolithic",
                indicators=[
                    "单一部署单元",
                    "紧耦合组件",
                    "单体数据库",
                    "共享代码库"
                ],
                confidence=0.0
            ),
            ArchitecturePattern(
                name="Microservices",
                pattern_type="microservices",
                indicators=[
                    "服务拆分",
                    "API网关",
                    "服务发现",
                    "独立数据库",
                    "容器化部署"
                ],
                confidence=0.0
            ),
            ArchitecturePattern(
                name="Layered",
                pattern_type="layered",
                indicators=[
                    "表现层",
                    "业务层",
                    "数据访问层",
                    "清晰的分层结构"
                ],
                confidence=0.0
            ),
            ArchitecturePattern(
                name="Modular",
                pattern_type="modular",
                indicators=[
                    "模块化设计",
                    "依赖注入",
                    "接口隔离",
                    "插件架构"
                ],
                confidence=0.0
            ),
            ArchitecturePattern(
                name="Event-Driven",
                pattern_type="event_driven",
                indicators=[
                    "事件总线",
                    "消息队列",
                    "发布订阅",
                    "异步处理"
                ],
                confidence=0.0
            ),
            ArchitecturePattern(
                name="Serverless",
                pattern_type="serverless",
                indicators=[
                    "函数即服务",
                    "无服务器管理",
                    "按需计费",
                    "自动扩缩"
                ],
                confidence=0.0
            )
        ]
        
        return patterns
    
    def analyze_project(self, project_path: str) -> ProjectAnalysis:
        """分析项目"""
        logger.info(f"开始分析项目: {project_path}")
        
        project_path = Path(project_path).resolve()
        project_name = project_path.name
        
        # 基础分析
        metrics = self._analyze_metrics(project_path)
        
        # 框架检测
        frameworks = self._detect_frameworks(project_path)
        
        # 架构分析
        patterns = self._analyze_architecture(project_path)
        
        # 计算复杂度和难度
        complexity_score = self._calculate_complexity(metrics, patterns)
        difficulty_level = self._assess_difficulty(complexity_score)
        estimated_effort = self._estimate_effort(metrics, complexity_score)
        
        # 生成建议
        recommendations = self._generate_recommendations(metrics, frameworks, patterns)
        
        # 确定主要语言
        primary_language = self._determine_primary_language(metrics)
        
        # 确定项目类型
        project_type = self._determine_project_type(frameworks, metrics)
        
        # 确定架构类型
        architecture = self._determine_architecture(patterns)
        
        analysis = ProjectAnalysis(
            project_path=str(project_path),
            project_name=project_name,
            project_type=project_type,
            primary_language=primary_language,
            frameworks=frameworks,
            architecture=architecture,
            complexity_score=complexity_score,
            difficulty_level=difficulty_level,
            estimated_effort=estimated_effort,
            recommendations=recommendations,
            metrics=metrics,
            patterns=patterns
        )
        
        logger.info(f"项目分析完成: {project_name}")
        return analysis
    
    def _analyze_metrics(self, project_path: Path) -> ProjectMetrics:
        """分析项目指标"""
        metrics = ProjectMetrics()
        
        # 遍历所有文件
        for root, dirs, files in os.walk(project_path):
            for file in files:
                file_path = Path(root) / file
                try:
                    # 获取文件扩展名
                    ext = file_path.suffix.lower()
                    
                    # 更新统计
                    metrics.total_files += 1
                    
                    # 计算行数
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            metrics.total_lines += len(f.readlines())
                    except:
                        pass
                    
                    # 分类文件
                    if ext in ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.kt', '.swift', '.go', '.rs', '.cpp', '.c', '.cs', '.php', '.rb', '.scala', '.r', '.m', '.sh', '.sql']:
                        metrics.code_files += 1
                        metrics.language_distribution[self.language_rules.get(ext, 'Other')] += 1
                    elif ext in ['.html', '.css', '.scss', '.sass', '.less']:
                        metrics.code_files += 1
                        metrics.language_distribution[self.language_rules.get(ext, 'Other')] += 1
                    elif ext in ['.md', '.rst', '.txt']:
                        metrics.doc_files += 1
                    elif ext in ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg']:
                        metrics.config_files += 1
                    elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.woff', '.woff2']:
                        metrics.asset_files += 1
                    elif ext in ['.test', '.spec']:
                        metrics.test_files += 1
                        metrics.language_distribution[self.language_rules.get(ext, 'Other')] += 1
                        
                except Exception as e:
                    logger.warning(f"处理文件失败 {file_path}: {e}")
        
        return metrics
    
    def _detect_frameworks(self, project_path: Path) -> List[str]:
        """检测项目框架"""
        frameworks = []
        
        # 检查配置文件
        for file_path in project_path.rglob("*"):
            if file_path.is_file():
                file_name = file_path.name
                
                # 检查框架规则
                if file_name in self.framework_rules:
                    detect_func = self.framework_rules[file_name]
                    detected_frameworks = detect_func(project_path)
                    if detected_frameworks:
                        frameworks.extend(detected_frameworks)
        
        return list(set(frameworks))
    
    def _detect_js_framework(self, project_path: Path) -> List[str]:
        """检测JavaScript框架"""
        frameworks = []
        
        package_json = project_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json, 'r', encoding='utf-8') as f:
                    package_data = json.load(f)
                    
                    dependencies = {
                        **package_data.get("dependencies", {}),
                        **package_data.get("devDependencies", {})
                    }
                    
                    # 检测框架
                    if "react" in dependencies:
                        frameworks.append("React")
                    if "vue" in dependencies:
                        frameworks.append("Vue.js")
                    if "angular" in dependencies:
                        frameworks.append("Angular")
                    if "express" in dependencies:
                        frameworks.append("Express.js")
                    if "next" in dependencies:
                        frameworks.append("Next.js")
                    if "nuxt" in dependencies:
                        frameworks.append("Nuxt.js")
                    if "gatsby" in dependencies:
                        frameworks.append("Gatsby")
                    if "svelte" in dependencies:
                        frameworks.append("Svelte")
                    if "sapper" in dependencies:
                        frameworks.append("Sapper")
                    
            except Exception as e:
                logger.warning(f"解析package.json失败: {e}")
        
        return frameworks
    
    def _detect_python_framework(self, project_path: Path) -> List[str]:
        """检测Python框架"""
        frameworks = []
        
        # 检查requirements.txt
        requirements = project_path / "requirements.txt"
        if requirements.exists():
            try:
                with open(requirements, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # 检测框架
                    if "django" in content:
                        frameworks.append("Django")
                    if "flask" in content:
                        frameworks.append("Flask")
                    if "fastapi" in content:
                        frameworks.append("FastAPI")
                    if "starlette" in content:
                        frameworks.append("Starlette")
                    if "sqlalchemy" in content:
                        frameworks.append("SQLAlchemy")
                    if "pandas" in content:
                        frameworks.append("Pandas")
                    if "numpy" in content:
                        frameworks.append("NumPy")
                    if "scikit-learn" in content:
                        frameworks.append("Scikit-Learn")
                    if "tensorflow" in content:
                        frameworks.append("TensorFlow")
                    if "pytorch" in content:
                        frameworks.append("PyTorch")
                    
            except Exception as e:
                logger.warning(f"解析requirements.txt失败: {e}")
        
        # 检查特定文件
        if (project_path / "manage.py").exists():
            frameworks.append("Django")
        if (project_path / "app.py").exists():
            frameworks.append("Flask")
        if (project_path / "main.py").exists():
            frameworks.append("Generic Python")
        
        return frameworks
    
    def _analyze_architecture(self, project_path: Path) -> List[ArchitecturePattern]:
        """分析项目架构"""
        patterns = []
        
        # 复制模式模板
        pattern_templates = {p.name: p for p in self.patterns}
        
        # 检测指标
        for pattern_name, pattern in pattern_templates.items():
            confidence = 0.0
            matched_indicators = 0
            
            for indicator in pattern.indicators:
                # 在文件名中搜索
                if any(indicator.lower() in f.lower() for f in self._get_all_files(project_path)):
                    matched_indicators += 1
            
            # 计算置信度
            if pattern.indicators:
                confidence = matched_indicators / len(pattern.indicators)
            
            # 更新模式
            pattern.confidence = confidence
            patterns.append(pattern)
        
        return patterns
    
    def _get_all_files(self, project_path: Path) -> List[str]:
        """获取所有文件路径"""
        files = []
        for root, dirs, filenames in os.walk(project_path):
            for filename in filenames:
                files.append(str(Path(root) / filename))
        return files
    
    def _calculate_complexity(self, metrics: ProjectMetrics, patterns: List[ArchitecturePattern]) -> float:
        """计算复杂度分数"""
        # 基础复杂度
        base_complexity = 0.0
        
        # 文件数量影响
        if metrics.total_files > 1000:
            base_complexity += 0.3
        elif metrics.total_files > 100:
            base_complexity += 0.2
        elif metrics.total_files > 10:
            base_complexity += 0.1
        
        # 代码行数影响
        if metrics.total_lines > 100000:
            base_complexity += 0.3
        elif metrics.total_lines > 10000:
            base_complexity += 0.2
        elif metrics.total_lines > 1000:
            base_complexity += 0.1
        
        # 语言多样性影响
        language_diversity = len([v for v in metrics.language_distribution.values() if v > 0])
        if language_diversity > 5:
            base_complexity += 0.2
        elif language_diversity > 3:
            base_complexity += 0.1
        
        # 架构复杂度
        architecture_complexity = 0.0
        for pattern in patterns:
            if pattern.pattern_type == "microservices":
                architecture_complexity += 0.4
            elif pattern.pattern_type == "event_driven":
                architecture_complexity += 0.3
            elif pattern.pattern_type == "layered":
                architecture_complexity += 0.2
            elif pattern.pattern_type == "modular":
                architecture_complexity += 0.2
        
        # 测试覆盖率影响（反向）
        if metrics.test_files > 0:
            test_ratio = metrics.test_files / metrics.code_files if metrics.code_files > 0 else 0
            if test_ratio < 0.1:
                base_complexity += 0.3
            elif test_ratio < 0.3:
                base_complexity += 0.2
            elif test_ratio < 0.5:
                base_complexity += 0.1
        
        # 框架数量影响
        if len(metrics.frameworks) > 5:
            base_complexity += 0.2
        elif len(metrics.frameworks) > 2:
            base_complexity += 0.1
        
        return min(base_complexity + architecture_complexity, 1.0)
    
    def _assess_difficulty(self, complexity_score: float) -> str:
        """评估难度等级"""
        if complexity_score < 0.2:
            return "easy"
        elif complexity_score < 0.4:
            return "medium"
        elif complexity_score < 0.7:
            return "hard"
        else:
            return "expert"
    
    def _estimate_effort(self, metrics: ProjectMetrics, complexity_score: float) -> str:
        """估算工作量"""
        # 基于代码行数和复杂度估算
        base_effort = metrics.total_lines / 50  # 假设每行代码2分钟
        
        # 复杂度调整
        if complexity_score > 0.7:
            base_effort *= 3
        elif complexity_score > 0.4:
            base_effort *= 2
        
        # 框架调整
        framework_multiplier = 1.0
        if len(metrics.frameworks) > 3:
            framework_multiplier = 1.5
        elif len(metrics.frameworks) > 1:
            framework_multiplier = 1.2
        
        adjusted_effort = base_effort * framework_multiplier
        
        # 转换为时间单位
        if adjusted_effort < 8:
            return f"{int(adjusted_effort)} hours"
        elif adjusted_effort < 40:
            return f"{int(adjusted_effort / 8)} days"
        elif adjusted_effort < 160:
            return f"{int(adjusted_effort / 40)} weeks"
        else:
            return f"{int(adjusted_effort / 160)} months"
    
    def _generate_recommendations(self, metrics: ProjectMetrics, frameworks: List[str], patterns: List[ArchitecturePattern]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 测试覆盖率建议
        if metrics.test_files == 0:
            recommendations.append("建议添加单元测试以提高代码质量")
        else:
            test_ratio = metrics.test_files / metrics.code_files if metrics.code_files > 0 else 0
            if test_ratio < 0.3:
                recommendations.append(f"测试覆盖率较低({test_ratio:.1%})，建议增加测试用例")
        
        # 文档建议
        if metrics.doc_files == 0:
            recommendations.append("建议添加项目文档（README、API文档等）")
        
        # 代码组织建议
        if metrics.total_files > 1000 and not self._has_good_structure(metrics):
            recommendations.append("建议优化项目结构，考虑模块化组织")
        
        # 架构建议
        if not patterns:
            recommendations.append("建议采用清晰的架构模式（如分层架构或微服务）")
        else:
            best_pattern = max(patterns, key=lambda p: p.confidence)
            if best_pattern.confidence < 0.5:
                recommendations.append(f"架构模式不够清晰，建议强化{best_pattern.name}模式")
        
        # 框架建议
        if not frameworks:
            recommendations.append("建议使用成熟的框架来提高开发效率")
        
        # 依赖管理建议
        if metrics.config_files == 0:
            recommendations.append("建议添加依赖管理文件（如package.json、requirements.txt等）")
        
        return recommendations
    
    def _has_good_structure(self, metrics: ProjectMetrics) -> bool:
        """检查是否有良好的结构"""
        # 简单的启发式规则
        if metrics.code_files > 0:
            doc_ratio = metrics.doc_files / metrics.code_files
            config_ratio = metrics.config_files / metrics.code_files
            test_ratio = metrics.test_files / metrics.code_files
            
            # 良好的结构应该有适当的文档和配置
            return doc_ratio > 0.05 and config_ratio > 0.1 and test_ratio > 0.1
        
        return False
    
    def _determine_primary_language(self, metrics: ProjectMetrics) -> str:
        """确定主要编程语言"""
        if not metrics.language_distribution:
            return "Unknown"
        
        # 运行数量最多的语言
        primary_language = max(metrics.language_distribution.items(), key=lambda x: x[1])[0]
        return primary_language
    
    def _determine_project_type(self, frameworks: List[str], metrics: ProjectMetrics) -> str:
        """确定项目类型"""
        if not frameworks:
            return "Unknown"
        
        # 基于框架确定项目类型
        if "React" in frameworks or "Vue" in frameworks or "Angular" in frameworks:
            return "Frontend Application"
        elif "Django" in frameworks or "Flask" in frameworks or "FastAPI" in frameworks:
            return "Backend API"
        elif "Next.js" in frameworks or "Nuxt.js" in frameworks or "Gatsby" in frameworks:
            return "Full-0stack Application"
        elif "Docker" in frameworks or "Kubernetes" in frameworks:
            return "Containerized Application"
        elif "TensorFlow" in frameworks or "PyTorch" in frameworks or "Scikit-Learn" in frameworks:
            return "Machine Learning Project"
        elif "Android" in frameworks or "iOS" in frameworks or "React Native" in frameworks:
            return "Mobile Application"
        else:
            return "Generic"
    
    def _determine_architecture(self, patterns: List[ArchitecturePattern]) -> str:
        """确定架构类型"""
        if not patterns:
            return "Unknown"
        
        # 返回置信度最高的模式
        best_pattern = max(patterns, key=lambda p: p.confidence)
        return best_pattern.name

# 全局分析器实例
_project_analyzer = None

def get_project_analyzer(config: Dict[str, Any] = None) -> ProjectAnalyzer:
    """获取项目分析器实例"""
    global _project_analyzer
    if _project_analyzer is None:
        _project_analyzer = ProjectAnalyzer(config or {})
    return _project_analyzer

if __name__ == "__main__":
    # 测试代码
    config = {}
    
    analyzer = get_project_analyzer(config)
    
    # 分析当前项目
    analysis = analyzer.analyze_project(str(PROJECT_ROOT))
    
    # 输出分析结果
    print(json.dumps({
        "project_name": analysis.project_name,
        "project_type": analysis.project_type,
        "primary_language": analysis.primary_language,
        "frameworks": analysis.frameworks,
        "architecture": analysis.architecture,
        "complexity_score": analysis.complexity_score,
        "difficulty_level": analysis.difficulty_level,
        "estimated_effort": analysis.estimated_effort,
        "recommendations": analysis.recommendations
    }, ensure_ascii=False, indent=2))