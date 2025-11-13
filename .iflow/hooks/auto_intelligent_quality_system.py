#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ 自动智能质量系统 V4 (Auto Intelligent Quality System V4)
全自动审查、测试、优化、修复的一体化质量保障系统。
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import asyncio
import logging
import subprocess
import time
import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
import queue

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class QualityIssueType(Enum):
    """质量问题类型"""
    SYNTAX_ERROR = "syntax_error"
    STYLE_VIOLATION = "style_violation"
    SECURITY_VULNERABILITY = "security_vulnerability"
    PERFORMANCE_ISSUE = "performance_issue"
    LOGIC_ERROR = "logic_error"
    DOCUMENTATION_MISSING = "documentation_missing"
    TEST_COVERAGE_LOW = "test_coverage_low"
    DEPENDENCY_ISSUE = "dependency_issue"
    CODE_COMPLEXITY = "code_complexity"
    BEST_PRACTICE_VIOLATION = "best_practice_violation"

class Severity(Enum):
    """严重程度"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class QualityIssue:
    """质量问题"""
    id: str
    type: QualityIssueType
    severity: Severity
    file_path: str
    line_number: Optional[int]
    description: str
    suggestion: str
    auto_fixable: bool = False
    fixed: bool = False
    fix_applied: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityReport:
    """质量报告"""
    timestamp: datetime
    total_issues: int
    issues_by_type: Dict[str, int]
    issues_by_severity: Dict[str, int]
    issues: List[QualityIssue]
    auto_fixes_applied: int
    metrics: Dict[str, float]
    recommendations: List[str]

class AutoIntelligentQualitySystemV4:
    """
    自动智能质量系统 V4
    """
    
    def __init__(self, project_root: str = "A项目/iflow"):
        self.project_root = Path(project_root)
        self.issues_queue = queue.Queue()
        self.running = False
        self.worker_thread = None
        
        # 质量检查器
        self.checkers = {
            "syntax": self._check_syntax,
            "style": self._check_style,
            "security": self._check_security,
            "performance": self._check_performance,
            "documentation": self._check_documentation,
            "test_coverage": self._check_test_coverage,
            "dependencies": self._check_dependencies,
            "complexity": self._check_complexity,
            "best_practices": self._check_best_practices
        }
        
        # 自动修复器
        self.fixers = {
            "syntax": self._fix_syntax,
            "style": self._fix_style,
            "documentation": self._fix_documentation,
            "simple_security": self._fix_simple_security,
            "performance": self._fix_performance
        }
        
        # 统计信息
        self.stats = {
            "total_checks": 0,
            "issues_found": 0,
            "auto_fixes_applied": 0,
            "critical_issues_resolved": 0
        }
        
        logger.info("自动智能质量系统V4初始化完成")

    def start_monitoring(self):
        """开始监控"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("质量监控已启动")

    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("质量监控已停止")

    def _worker_loop(self):
        """工作循环"""
        while self.running:
            try:
                # 获取待检查文件
                files_to_check = self._get_files_to_check()
                
                for file_path in files_to_check:
                    if not self.running:
                        break
                    
                    # 执行质量检查
                    issues = self._check_file_quality(file_path)
                    
                    # 自动修复
                    auto_fixed = self._auto_fix_issues(issues)
                    
                    # 更新统计
                    self.stats["total_checks"] += 1
                    self.stats["issues_found"] += len(issues)
                    self.stats["auto_fixes_applied"] += auto_fixed
                    
                    # 休息一下避免过度占用资源
                    time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"质量检查循环错误: {e}")
                time.sleep(1)

    def _get_files_to_check(self) -> List[Path]:
        """获取需要检查的文件"""
        files_to_check = []
        
        # 检查Python文件
        for py_file in self.project_root.rglob("*.py"):
            if not self._should_ignore_file(py_file):
                files_to_check.append(py_file)
        
        # 检查配置文件
        for config_file in self.project_root.rglob("*.yaml"):
            if not self._should_ignore_file(config_file):
                files_to_check.append(config_file)
        
        for config_file in self.project_root.rglob("*.json"):
            if not self._should_ignore_file(config_file):
                files_to_check.append(config_file)
        
        # 检查文档文件
        for doc_file in self.project_root.rglob("*.md"):
            if not self._should_ignore_file(doc_file):
                files_to_check.append(doc_file)
        
        return files_to_check

    def _should_ignore_file(self, file_path: Path) -> bool:
        """判断是否应该忽略文件"""
        ignore_patterns = [
            "__pycache__",
            ".git",
            "node_modules",
            ".pytest_cache",
            ".coverage",
            "build",
            "dist",
            ".venv",
            "venv",
            "env"
        ]
        
        for pattern in ignore_patterns:
            if pattern in str(file_path):
                return True
        
        return False

    async def check_file(self, file_path: str) -> QualityReport:
        """检查单个文件"""
        file_path = Path(file_path)
        issues = self._check_file_quality(file_path)
        
        # 生成报告
        report = QualityReport(
            timestamp=datetime.now(),
            total_issues=len(issues),
            issues_by_type={},
            issues_by_severity={},
            issues=issues,
            auto_fixes_applied=0,
            metrics=self._calculate_metrics(file_path, issues),
            recommendations=self._generate_recommendations(issues)
        )
        
        # 统计问题
        for issue in issues:
            report.issues_by_type[issue.type.value] = report.issues_by_type.get(issue.type.value, 0) + 1
            report.issues_by_severity[issue.severity.value] = report.issues_by_severity.get(issue.severity.value, 0) + 1
        
        return report

    def _check_file_quality(self, file_path: Path) -> List[QualityIssue]:
        """检查文件质量"""
        issues = []
        
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 执行各种检查
            for checker_name, checker_func in self.checkers.items():
                try:
                    checker_issues = checker_func(file_path, content)
                    issues.extend(checker_issues)
                except Exception as e:
                    logger.warning(f"检查器 {checker_name} 失败: {e}")
        
        except Exception as e:
            issues.append(QualityIssue(
                id=f"read_error_{int(time.time())}",
                type=QualityIssueType.SYNTAX_ERROR,
                severity=Severity.HIGH,
                file_path=str(file_path),
                line_number=None,
                description=f"无法读取文件: {str(e)}",
                suggestion="检查文件权限和编码",
                auto_fixable=False
            ))
        
        return issues

    def _check_syntax(self, file_path: Path, content: str) -> List[QualityIssue]:
        """检查语法错误"""
        issues = []
        
        if file_path.suffix == '.py':
            try:
                # 使用ast解析检查语法
                ast.parse(content)
            except SyntaxError as e:
                issues.append(QualityIssue(
                    id=f"syntax_{int(time.time())}",
                    type=QualityIssueType.SYNTAX_ERROR,
                    severity=Severity.CRITICAL,
                    file_path=str(file_path),
                    line_number=e.lineno,
                    description=f"语法错误: {e.msg}",
                    suggestion="修复语法错误",
                    auto_fixable=False
                ))
        
        return issues

    def _check_style(self, file_path: Path, content: str) -> List[QualityIssue]:
        """检查代码风格"""
        issues = []
        
        if file_path.suffix == '.py':
            lines = content.split('\n')
            
            # 检查行长度
            for i, line in enumerate(lines, 1):
                if len(line) > 120:
                    issues.append(QualityIssue(
                        id=f"line_length_{int(time.time())}_{i}",
                        type=QualityIssueType.STYLE_VIOLATION,
                        severity=Severity.MEDIUM,
                        file_path=str(file_path),
                        line_number=i,
                        description=f"行过长 ({len(line)} 字符)",
                        suggestion="将长行拆分为多行",
                        auto_fixable=True
                    ))
            
            # 检查尾随空格
            for i, line in enumerate(lines, 1):
                if line.endswith(' '):
                    issues.append(QualityIssue(
                        id=f"trailing_space_{int(time.time())}_{i}",
                        type=QualityIssueType.STYLE_VIOLATION,
                        severity=Severity.LOW,
                        file_path=str(file_path),
                        line_number=i,
                        description="行尾有多余空格",
                        suggestion="移除行尾空格",
                        auto_fixable=True
                    ))
        
        return issues

    def _check_security(self, file_path: Path, content: str) -> List[QualityIssue]:
        """检查安全问题"""
        issues = []
        
        # 检查硬编码密钥
        sensitive_patterns = [
            r'(password|passwd|pwd|secret|token|key)\s*=\s*["\'][^"\']*["\']',
            r'(api_key|apikey)\s*=\s*["\'][^"\']*["\']',
            r'(private_key|privatekey)\s*=\s*["\'][^"\']*["\']'
        ]
        
        for pattern in sensitive_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                issues.append(QualityIssue(
                    id=f"security_{int(time.time())}_{line_num}",
                    type=QualityIssueType.SECURITY_VULNERABILITY,
                    severity=Severity.CRITICAL,
                    file_path=str(file_path),
                    line_number=line_num,
                    description="发现硬编码敏感信息",
                    suggestion="使用环境变量或配置文件",
                    auto_fixable=False
                ))
        
        return issues

    def _check_performance(self, file_path: Path, content: str) -> List[QualityIssue]:
        """检查性能问题"""
        issues = []
        
        if file_path.suffix == '.py':
            # 检查循环中的重复计算
            lines = content.split('\n')
            in_loop = False
            loop_vars = set()
            
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                
                # 检测循环开始
                if stripped.startswith(('for ', 'while ')):
                    in_loop = True
                    # 提取循环变量
                    match = re.search(r'(for|while)\s+(\w+)', stripped)
                    if match:
                        loop_vars.add(match.group(2))
                
                elif stripped in ('break', 'continue', 'pass'):
                    in_loop = False
                    loop_vars.clear()
                
                # 检查循环内的重复函数调用
                elif in_loop:
                    for var in loop_vars:
                        pattern = rf'{var}\.\w*\('
                        matches = re.findall(pattern, line)
                        if len(matches) > 3:
                            issues.append(QualityIssue(
                                id=f"performance_{int(time.time())}_{i}",
                                type=QualityIssueType.PERFORMANCE_ISSUE,
                                severity=Severity.MEDIUM,
                                file_path=str(file_path),
                                line_number=i,
                                description=f"循环中重复调用 {var} 方法",
                                suggestion="将结果缓存到循环外",
                                auto_fixable=False
                            ))
        
        return issues

    def _check_documentation(self, file_path: Path, content: str) -> List[QualityIssue]:
        """检查文档"""
        issues = []
        
        # 检查模块文档字符串
        if file_path.suffix == '.py':
            if not content.startswith('"""') and not content.startswith("'''"):
                issues.append(QualityIssue(
                    id=f"docstring_{int(time.time())}",
                    type=QualityIssueType.DOCUMENTATION_MISSING,
                    severity=Severity.MEDIUM,
                    file_path=str(file_path),
                    line_number=1,
                    description="缺少模块文档字符串",
                    suggestion="添加模块文档字符串",
                    auto_fixable=True
                ))
        
        return issues

    def _check_test_coverage(self, file_path: Path, content: str) -> List[QualityIssue]:
        """检查测试覆盖率"""
        issues = []
        
        # 简化的测试覆盖率检查
        if file_path.name.startswith('test_') or 'tests' in str(file_path):
            return issues
        
        # 检查是否有对应的测试文件
        test_file = file_path.parent / f"test_{file_path.name}"
        if not test_file.exists():
            issues.append(QualityIssue(
                id=f"test_coverage_{int(time.time())}",
                type=QualityIssueType.TEST_COVERAGE_LOW,
                severity=Severity.MEDIUM,
                file_path=str(file_path),
                line_number=None,
                description="缺少对应的测试文件",
                suggestion=f"创建测试文件 {test_file.name}",
                auto_fixable=False
            ))
        
        return issues

    def _check_dependencies(self, file_path: Path, content: str) -> List[QualityIssue]:
        """检查依赖问题"""
        issues = []
        
        # 检查过时的依赖
        outdated_packages = [
            'urllib3',
            'requests==1.x',
            'numpy==1.x',
            'pandas==1.x'
        ]
        
        for package in outdated_packages:
            if package in content:
                issues.append(QualityIssue(
                    id=f"dependency_{int(time.time())}",
                    type=QualityIssueType.DEPENDENCY_ISSUE,
                    severity=Severity.MEDIUM,
                    file_path=str(file_path),
                    line_number=None,
                    description=f"使用过时的依赖: {package}",
                    suggestion="更新到最新版本",
                    auto_fixable=False
                ))
        
        return issues

    def _check_complexity(self, file_path: Path, content: str) -> List[QualityIssue]:
        """检查代码复杂度"""
        issues = []
        
        if file_path.suffix == '.py':
            try:
                tree = ast.parse(content)
                
                # 检查函数复杂度
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        complexity = self._calculate_complexity(node)
                        if complexity > 10:
                            issues.append(QualityIssue(
                                id=f"complexity_{int(time.time())}",
                                type=QualityIssueType.CODE_COMPLEXITY,
                                severity=Severity.HIGH,
                                file_path=str(file_path),
                                line_number=node.lineno,
                                description=f"函数 {node.name} 复杂度过高 ({complexity})",
                                suggestion="重构函数，降低复杂度",
                                auto_fixable=False
                            ))
            except Exception as e:
                logger.warning(f"复杂度检查失败: {e}")
        
        return issues

    def _check_best_practices(self, file_path: Path, content: str) -> List[QualityIssue]:
        """检查最佳实践"""
        issues = []
        
        if file_path.suffix == '.py':
            lines = content.split('\n')
            
            # 检查裸露的except
            for i, line in enumerate(lines, 1):
                if 'except:' in line and line.strip() == 'except:':
                    issues.append(QualityIssue(
                        id=f"best_practice_{int(time.time())}_{i}",
                        type=QualityIssueType.BEST_PRACTICE_VIOLATION,
                        severity=Severity.MEDIUM,
                        file_path=str(file_path),
                        line_number=i,
                        description="使用裸露的except语句",
                        suggestion="指定具体的异常类型",
                        auto_fixable=False
                    ))
        
        return issues

    def _calculate_complexity(self, node: ast.AST) -> int:
        """计算圈复杂度"""
        complexity = 1  # 基础复杂度
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity

    async def _auto_fix_issues(self, issues: List[QualityIssue]) -> int:
        """自动修复问题"""
        fixed_count = 0
        
        # 按文件分组
        issues_by_file = {}
        for issue in issues:
            if issue.auto_fixable and not issue.fixed:
                if issue.file_path not in issues_by_file:
                    issues_by_file[issue.file_path] = []
                issues_by_file[issue.file_path].append(issue)
        
        # 逐个文件修复
        for file_path, file_issues in issues_by_file.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 应用修复
                for issue in file_issues:
                    fixer = self.fixers.get(issue.type.value)
                    if fixer:
                        try:
                            content = fixer(file_path, content, issue)
                            issue.fixed = True
                            issue.fix_applied = "自动修复成功"
                            fixed_count += 1
                        except Exception as e:
                            logger.warning(f"修复失败 {issue.id}: {e}")
                
                # 写回文件
                if any(issue.fixed for issue in file_issues):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
            except Exception as e:
                logger.error(f"修复文件失败 {file_path}: {e}")
        
        return fixed_count

    def _fix_syntax(self, file_path: Path, content: str, issue: QualityIssue) -> str:
        """修复语法错误（占位符实现）"""
        # 语法错误通常需要手动修复
        return content

    def _fix_style(self, file_path: Path, content: str, issue: QualityIssue) -> str:
        """修复风格问题"""
        lines = content.split('\n')
        
        if issue.line_number and issue.line_number <= len(lines):
            line_idx = issue.line_number - 1
            line = lines[line_idx]
            
            if "行尾有多余空格" in issue.description:
                lines[line_idx] = line.rstrip()
            elif "行过长" in issue.description:
                # 简单的行拆分
                words = line.split()
                new_lines = []
                current_line = ""
                for word in words:
                    if len(current_line + word) <= 100:
                        current_line += " " + word if current_line else word
                    else:
                        new_lines.append(current_line)
                        current_line = "    " + word
                
                if current_line:
                    new_lines.append(current_line)
                
                # 替换原行
                lines[line_idx:line_idx+1] = new_lines
        
        return '\n'.join(lines)

    def _fix_documentation(self, file_path: Path, content: str, issue: QualityIssue) -> str:
        """修复文档问题"""
        if "缺少模块文档字符串" in issue.description:
            docstring = f'"""\n{file_path.stem} 模块\n\n自动生成的模块文档\n"""\n\n'
            return docstring + content
        
        return content

    def _fix_simple_security(self, file_path: Path, content: str, issue: QualityIssue) -> str:
        """修复简单安全问题（占位符实现）"""
        # 安全问题需要手动修复
        return content

    def _fix_performance(self, file_path: Path, content: str, issue: QualityIssue) -> str:
        """修复性能问题（占位符实现）"""
        # 性能问题需要手动修复
        return content

    def _calculate_metrics(self, file_path: Path, issues: List[QualityIssue]) -> Dict[str, float]:
        """计算质量指标"""
        total_issues = len(issues)
        
        metrics = {
            "quality_score": max(0, 100 - total_issues * 5),  # 简化的质量评分
            "critical_issues": sum(1 for i in issues if i.severity == Severity.CRITICAL),
            "high_issues": sum(1 for i in issues if i.severity == Severity.HIGH),
            "medium_issues": sum(1 for i in issues if i.severity == Severity.MEDIUM),
            "low_issues": sum(1 for i in issues if i.severity == Severity.LOW),
            "auto_fixable_ratio": sum(1 for i in issues if i.auto_fixable) / total_issues if total_issues > 0 else 0
        }
        
        return metrics

    def _generate_recommendations(self, issues: List[QualityIssue]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 统计问题类型
        type_counts = {}
        for issue in issues:
            type_counts[issue.type.value] = type_counts.get(issue.type.value, 0) + 1
        
        # 生成建议
        if type_counts.get("syntax_error", 0) > 0:
            recommendations.append("优先修复语法错误，这些错误会导致代码无法运行")
        
        if type_counts.get("security_vulnerability", 0) > 0:
            recommendations.append("立即处理安全问题，避免潜在的安全风险")
        
        if type_counts.get("performance_issue", 0) > 5:
            recommendations.append("优化性能问题，提升代码执行效率")
        
        if type_counts.get("documentation_missing", 0) > 3:
            recommendations.append("完善文档，提高代码可维护性")
        
        return recommendations

    async def generate_quality_report(self) -> Dict[str, Any]:
        """生成质量报告"""
        all_issues = []
        
        # 收集所有问题
        files_to_check = self._get_files_to_check()
        for file_path in files_to_check:
            file_issues = self._check_file_quality(file_path)
            all_issues.extend(file_issues.issues)
        
        # 生成报告
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_files_checked": len(files_to_check),
            "total_issues": len(all_issues),
            "issues_by_type": {},
            "issues_by_severity": {},
            "auto_fixes_applied": self.stats["auto_fixes_applied"],
            "critical_issues_resolved": self.stats["critical_issues_resolved"],
            "overall_score": 0,
            "recommendations": []
        }
        
        # 统计问题
        for issue in all_issues:
            report["issues_by_type"][issue.type.value] = report["issues_by_type"].get(issue.type.value, 0) + 1
            report["issues_by_severity"][issue.severity.value] = report["issues_by_severity"].get(issue.severity.value, 0) + 1
        
        # 计算总体评分
        if report["total_issues"] == 0:
            report["overall_score"] = 100
        else:
            critical_weight = report["issues_by_severity"].get("critical", 0) * 10
            high_weight = report["issues_by_severity"].get("high", 0) * 5
            medium_weight = report["issues_by_severity"].get("medium", 0) * 2
            low_weight = report["issues_by_severity"].get("low", 0) * 1
            
            max_penalty = report["total_files_checked"] * 10
            penalty = critical_weight + high_weight + medium_weight + low_weight
            report["overall_score"] = max(0, 100 - (penalty / max_penalty * 100))
        
        # 生成建议
        report["recommendations"] = self._generate_recommendations(all_issues)
        
        return report

    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "running": self.running,
            "stats": self.stats,
            "checkers_available": list(self.checkers.keys()),
            "fixers_available": list(self.fixers.keys())
        }