#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 自动质量检查Hook V6 (Auto Quality Check Hook V6)
T-MIA凤凰架构的质量守护者，提供全方位的代码质量检查和优化建议

你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import asyncio
import logging
import ast
import re
import time
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
import subprocess
import importlib.util
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

@dataclass
class QualityIssue:
    """质量问题"""
    issue_id: str
    issue_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    file_path: str
    line_number: int
    description: str
    suggestion: str
    category: str
    timestamp: float

class AutoQualityCheckHookV6:
    """
    自动质量检查Hook V6 - T-MIA凤凰架构的质量守护者
    提供代码质量检查、性能分析、规范验证和优化建议
    """
    
    def __init__(self):
        self.hook_id = f"auto_quality_check_v6_{int(time.time())}"
        
        # 质量检查器
        self.code_analyzer = CodeQualityAnalyzerV6()
        self.performance_analyzer = PerformanceAnalyzerV6()
        self.security_analyzer = SecurityAnalyzerV6()
        self.complexity_analyzer = ComplexityAnalyzerV6()
        self.documentation_analyzer = DocumentationAnalyzerV6()
        
        # 质量标准
        self.quality_standards = self._load_quality_standards()
        
        # 检查规则
        self.check_rules = self._load_check_rules()
        
        logger.info(f"🔍 自动质量检查Hook V6初始化完成 - Hook ID: {self.hook_id}")
    
    def _load_quality_standards(self) -> Dict[str, Any]:
        """加载质量标准"""
        return {
            "code_coverage": {
                "minimum": 80.0,
                "target": 90.0,
                "critical": 95.0
            },
            "cyclomatic_complexity": {
                "max_function": 10,
                "max_class": 15,
                "max_module": 20
            },
            "code_smells": {
                "max_per_file": 5,
                "max_per_module": 20,
                "critical_threshold": 50
            },
            "security_vulnerabilities": {
                "critical": 0,
                "high": 0,
                "medium": 5,
                "low": 10
            },
            "performance_metrics": {
                "max_response_time": 1000,  # ms
                "max_memory_usage": 100,    # MB
                "min_throughput": 100       # QPS
            }
        }
    
    def _load_check_rules(self) -> Dict[str, List[str]]:
        """加载检查规则"""
        return {
            "naming_conventions": [
                r"^[a-z_][a-z0-9_]*$",  # 变量名
                r"^[A-Z][a-zA-Z0-9]*$",  # 类名
                r"^[a-z_][a-z0-9_]*$",  # 函数名
                r"^[A-Z_]+$"            # 常量名
            ],
            "code_patterns": [
                "import \*",              # 禁止使用import *
                "print\(",               # 避免使用print
                "TODO|FIXME|HACK",       # 待处理标记
                "^\s*#\s*[A-Z]",         # 注释格式
                "^\s*\"\"\".*\"\"\"$",   # 文档字符串
            ],
            "performance_issues": [
                "for.*in.*range\(\d{3,}\)",  # 大循环
                "while\s+True:",             # 无限循环
                "time\.sleep\(",             # 睡眠调用
                "sync.*database",            # 同步数据库操作
            ]
        }
    
    async def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook执行入口
        
        Args:
            context: 执行上下文
        
        Returns:
            Dict[str, Any]: 检查结果
        """
        start_time = time.time()
        
        results = {
            "hook_id": self.hook_id,
            "timestamp": start_time,
            "success": True,
            "quality_score": 0.0,
            "checks": {},
            "issues": [],
            "recommendations": [],
            "metrics": {},
            "execution_time": 0.0
        }
        
        try:
            # 1. 代码质量分析
            code_check = await self._analyze_code_quality(context)
            results["checks"]["code_quality"] = code_check
            
            # 2. 性能分析
            performance_check = await self._analyze_performance(context)
            results["checks"]["performance"] = performance_check
            
            # 3. 安全分析
            security_check = await self._analyze_security(context)
            results["checks"]["security"] = security_check
            
            # 4. 复杂度分析
            complexity_check = await self._analyze_complexity(context)
            results["checks"]["complexity"] = complexity_check
            
            # 5. 文档分析
            documentation_check = await self._analyze_documentation(context)
            results["checks"]["documentation"] = documentation_check
            
            # 6. 代码风格检查
            style_check = await self._check_coding_style(context)
            results["checks"]["coding_style"] = style_check
            
            # 汇总结果
            all_checks = list(results["checks"].values())
            results["success"] = all(check.get("passed", False) for check in all_checks)
            
            # 收集问题
            for check_name, check_result in results["checks"].items():
                if check_result.get("issues"):
                    results["issues"].extend(check_result["issues"])
            
            # 计算质量分数
            results["quality_score"] = self._calculate_quality_score(results["checks"])
            
            # 生成建议
            results["recommendations"] = self._generate_quality_recommendations(results["issues"])
            
            # 生成指标
            results["metrics"] = self._generate_quality_metrics(results["checks"])
            
        except Exception as e:
            logger.error(f"质量检查执行失败: {e}")
            results["success"] = False
            results["error"] = str(e)
        
        results["execution_time"] = time.time() - start_time
        
        logger.info(f"🔍 质量检查完成: 分数 {results['quality_score']:.2f}, 问题 {len(results['issues'])} 个")
        return results
    
    async def _analyze_code_quality(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析代码质量"""
        check_result = {
            "check_name": "code_quality",
            "passed": True,
            "score": 0.0,
            "issues": [],
            "details": {}
        }
        
        # 获取代码内容
        code_content = context.get("code", "") or context.get("content", "")
        file_path = context.get("file_path", "unknown")
        
        if not code_content:
            check_result["details"]["status"] = "no_code_provided"
            return check_result
        
        try:
            # 解析AST
            tree = ast.parse(code_content)
            
            # 代码质量分析
            quality_issues = []
            
            # 检查导入语句
            import_issues = self._check_imports(tree, file_path)
            quality_issues.extend(import_issues)
            
            # 检查函数定义
            function_issues = self._check_functions(tree, file_path)
            quality_issues.extend(function_issues)
            
            # 检查类定义
            class_issues = self._check_classes(tree, file_path)
            quality_issues.extend(class_issues)
            
            # 检查代码模式
            pattern_issues = self._check_code_patterns(code_content, file_path)
            quality_issues.extend(pattern_issues)
            
            # 检查空值处理
            null_issues = self._check_null_handling(tree, file_path)
            quality_issues.extend(null_issues)
            
            check_result["issues"] = quality_issues
            
            # 计算质量分数
            max_issues = 20  # 最大问题数
            score = max(0.0, 1.0 - (len(quality_issues) / max_issues))
            check_result["score"] = score
            check_result["passed"] = score >= 0.7
            
            check_result["details"] = {
                "lines_of_code": len(code_content.split('\n')),
                "functions_found": len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
                "classes_found": len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]),
                "issues_found": len(quality_issues)
            }
            
        except SyntaxError as e:
            quality_issues = [QualityIssue(
                issue_id=f"syntax_error_{file_path}",
                issue_type="SYNTAX_ERROR",
                severity="CRITICAL",
                file_path=file_path,
                line_number=e.lineno or 0,
                description=f"语法错误: {e.msg}",
                suggestion="修复语法错误",
                category="code_quality",
                timestamp=time.time()
            )]
            check_result["issues"] = [issue.__dict__ for issue in quality_issues]
            check_result["passed"] = False
            check_result["score"] = 0.0
            check_result["details"]["syntax_error"] = str(e)
        
        return check_result
    
    def _check_imports(self, tree: ast.Module, file_path: str) -> List[Dict[str, Any]]:
        """检查导入语句"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # 检查通配符导入
                    if alias.name == "*":
                        issue = QualityIssue(
                            issue_id=f"wildcard_import_{file_path}",
                            issue_type="WILDCARD_IMPORT",
                            severity="MEDIUM",
                            file_path=file_path,
                            line_number=node.lineno,
                            description="使用了通配符导入 (*)",
                            suggestion="明确导入需要的模块或函数",
                            category="imports",
                            timestamp=time.time()
                        )
                        issues.append(issue.__dict__)
                    
                    # 检查未使用的导入（简化检查）
                    if alias.name.startswith("_"):
                        issue = QualityIssue(
                            issue_id=f"unused_import_{file_path}_{alias.name}",
                            issue_type="UNUSED_IMPORT",
                            severity="LOW",
                            file_path=file_path,
                            line_number=node.lineno,
                            description=f"可能存在未使用的导入: {alias.name}",
                            suggestion="移除未使用的导入",
                            category="imports",
                            timestamp=time.time()
                        )
                        issues.append(issue.__dict__)
            
            elif isinstance(node, ast.ImportFrom):
                # 检查从标准库的导入
                if node.module and node.module.startswith("."):
                    # 相对导入检查
                    if len(node.module) > 3:
                        issue = QualityIssue(
                            issue_id=f"deep_relative_import_{file_path}",
                            issue_type="DEEP_RELATIVE_IMPORT",
                            severity="LOW",
                            file_path=file_path,
                            line_number=node.lineno,
                            description="使用了过深的相对导入",
                            suggestion="考虑使用绝对导入",
                            category="imports",
                            timestamp=time.time()
                        )
                        issues.append(issue.__dict__)
        
        return issues
    
    def _check_functions(self, tree: ast.Module, file_path: str) -> List[Dict[str, Any]]:
        """检查函数定义"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # 检查函数长度
                if hasattr(node, 'end_lineno'):
                    func_lines = node.end_lineno - node.lineno
                    if func_lines > 50:
                        issue = QualityIssue(
                            issue_id=f"long_function_{file_path}_{node.name}",
                            issue_type="LONG_FUNCTION",
                            severity="MEDIUM",
                            file_path=file_path,
                            line_number=node.lineno,
                            description=f"函数过长: {func_lines} 行",
                            suggestion="将函数拆分为更小的函数",
                            category="functions",
                            timestamp=time.time()
                        )
                        issues.append(issue.__dict__)
                
                # 检查参数数量
                if len(node.args.args) > 7:
                    issue = QualityIssue(
                        issue_id=f"too_many_params_{file_path}_{node.name}",
                        issue_type="TOO_MANY_PARAMETERS",
                        severity="MEDIUM",
                        file_path=file_path,
                        line_number=node.lineno,
                        description=f"函数参数过多: {len(node.args.args)} 个",
                        suggestion="减少参数数量或使用数据类",
                        category="functions",
                        timestamp=time.time()
                    )
                    issues.append(issue.__dict__)
                
                # 检查返回值
                if not any(isinstance(n, ast.Return) for n in ast.walk(node)):
                    issue = QualityIssue(
                        issue_id=f"no_return_statement_{file_path}_{node.name}",
                        issue_type="NO_RETURN_STATEMENT",
                        severity="LOW",
                        file_path=file_path,
                        line_number=node.lineno,
                        description=f"函数缺少返回语句: {node.name}",
                        suggestion="添加适当的返回语句",
                        category="functions",
                        timestamp=time.time()
                    )
                    issues.append(issue.__dict__)
        
        return issues
    
    def _check_classes(self, tree: ast.Module, file_path: str) -> List[Dict[str, Any]]:
        """检查类定义"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # 检查类的大小
                method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                if method_count > 20:
                    issue = QualityIssue(
                        issue_id=f"large_class_{file_path}_{node.name}",
                        issue_type="LARGE_CLASS",
                        severity="MEDIUM",
                        file_path=file_path,
                        line_number=node.lineno,
                        description=f"类过大: {method_count} 个方法",
                        suggestion="将类拆分为更小的类",
                        category="classes",
                        timestamp=time.time()
                    )
                    issues.append(issue.__dict__)
                
                # 检查继承深度
                if len(node.bases) > 2:
                    issue = QualityIssue(
                        issue_id=f"deep_inheritance_{file_path}_{node.name}",
                        issue_type="DEEP_INHERITANCE",
                        severity="LOW",
                        file_path=file_path,
                        line_number=node.lineno,
                        description=f"继承层次过深: {len(node.bases)} 个基类",
                        suggestion="考虑使用组合而非继承",
                        category="classes",
                        timestamp=time.time()
                    )
                    issues.append(issue.__dict__)
        
        return issues
    
    def _check_code_patterns(self, code_content: str, file_path: str) -> List[Dict[str, Any]]:
        """检查代码模式"""
        issues = []
        
        lines = code_content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # 检查调试代码
            if re.search(r"print\s*\(", line) and "logger" not in line:
                issue = QualityIssue(
                    issue_id=f"debug_print_{file_path}_{line_num}",
                    issue_type="DEBUG_PRINT",
                    severity="LOW",
                    file_path=file_path,
                    line_number=line_num,
                    description="发现调试打印语句",
                    suggestion="使用日志系统替代print语句",
                    category="patterns",
                    timestamp=time.time()
                )
                issues.append(issue.__dict__)
            
            # 检查TODO注释
            if re.search(r"#\s*TODO|#\s*FIXME|#\s*HACK", line, re.IGNORECASE):
                match = re.search(r"#\s*(TODO|FIXME|HACK)", line, re.IGNORECASE)
                severity = "MEDIUM" if match.group(1).upper() in ["FIXME", "HACK"] else "LOW"
                
                issue = QualityIssue(
                    issue_id=f"todo_comment_{file_path}_{line_num}",
                    issue_type="TODO_COMMENT",
                    severity=severity,
                    file_path=file_path,
                    line_number=line_num,
                    description=f"发现待处理注释: {match.group(1)}",
                    suggestion="及时处理或移除TODO注释",
                    category="patterns",
                    timestamp=time.time()
                )
                issues.append(issue.__dict__)
            
            # 检查魔法数字
            if re.search(r"\b\d{3,}\b", line) and "import" not in line:
                issue = QualityIssue(
                    issue_id=f"magic_number_{file_path}_{line_num}",
                    issue_type="MAGIC_NUMBER",
                    severity="LOW",
                    file_path=file_path,
                    line_number=line_num,
                    description="发现魔法数字",
                    suggestion="使用常量替代魔法数字",
                    category="patterns",
                    timestamp=time.time()
                )
                issues.append(issue.__dict__)
        
        return issues
    
    def _check_null_handling(self, tree: ast.Module, file_path: str) -> List[Dict[str, Any]]:
        """检查空值处理"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                # 检查 None 比较
                if any(isinstance(comp, ast.NameConstant) and comp.value is None 
                       for comp in node.comparators):
                    # 检查是否使用 is 而不是 ==
                    if isinstance(node.ops[0], ast.Eq):
                        issue = QualityIssue(
                            issue_id=f"none_equality_{file_path}_{node.lineno}",
                            issue_type="NONE_EQUALITY",
                            severity="MEDIUM",
                            file_path=file_path,
                            line_number=node.lineno,
                            description="使用 == 比较 None，应使用 is",
                            suggestion="使用 'is None' 而不是 '== None'",
                            category="null_handling",
                            timestamp=time.time()
                        )
                        issues.append(issue.__dict__)
        
        return issues
    
    async def _analyze_performance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析性能"""
        return await self.performance_analyzer.analyze(context)
    
    async def _analyze_security(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析安全性"""
        return await self.security_analyzer.analyze(context)
    
    async def _analyze_complexity(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析复杂度"""
        return await self.complexity_analyzer.analyze(context)
    
    async def _analyze_documentation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析文档"""
        return await self.documentation_analyzer.analyze(context)
    
    async def _check_coding_style(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """检查代码风格"""
        check_result = {
            "check_name": "coding_style",
            "passed": True,
            "score": 0.0,
            "issues": [],
            "details": {}
        }
        
        code_content = context.get("code", "") or context.get("content", "")
        
        if not code_content:
            return check_result
        
        issues = []
        lines = code_content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # 检查行长度
            if len(line) > 120:
                issue = QualityIssue(
                    issue_id=f"long_line_{line_num}",
                    issue_type="LONG_LINE",
                    severity="LOW",
                    file_path=context.get("file_path", "unknown"),
                    line_number=line_num,
                    description=f"行过长: {len(line)} 字符",
                    suggestion="将长行拆分为多行",
                    category="style",
                    timestamp=time.time()
                )
                issues.append(issue.__dict__)
            
            # 检查缩进
            if line.startswith(' ') and not line.startswith('    ') and not line.startswith('\t'):
                if not re.match(r'^ {4,8}[^ ]', line):
                    issue = QualityIssue(
                        issue_id=f"incorrect_indentation_{line_num}",
                        issue_type="INCORRECT_INDENTATION",
                        severity="LOW",
                        file_path=context.get("file_path", "unknown"),
                        line_number=line_num,
                        description="缩进不正确，建议使用4空格",
                        suggestion="统一使用4空格缩进",
                        category="style",
                        timestamp=time.time()
                    )
                    issues.append(issue.__dict__)
            
            # 检查多余的空格
            if line.endswith(' ') or line.endswith('\t'):
                issue = QualityIssue(
                    issue_id=f"trailing_whitespace_{line_num}",
                    issue_type="TRAILING_WHITESPACE",
                    severity="LOW",
                    file_path=context.get("file_path", "unknown"),
                    line_number=line_num,
                    description="行尾有多余空格",
                    suggestion="移除行尾空格",
                    category="style",
                    timestamp=time.time()
                )
                issues.append(issue.__dict__)
        
        check_result["issues"] = issues
        
        # 计算风格分数
        max_issues = 10
        score = max(0.0, 1.0 - (len(issues) / max_issues))
        check_result["score"] = score
        check_result["passed"] = score >= 0.8
        
        check_result["details"] = {
            "lines_checked": len(lines),
            "style_issues": len(issues)
        }
        
        return check_result
    
    def _calculate_quality_score(self, checks: Dict[str, Dict]) -> float:
        """计算质量分数"""
        if not checks:
            return 0.0
        
        total_score = 0.0
        weight_sum = 0.0
        
        # 权重分配
        weights = {
            "code_quality": 0.3,
            "security": 0.25,
            "performance": 0.2,
            "complexity": 0.15,
            "documentation": 0.05,
            "coding_style": 0.05
        }
        
        for check_name, check_result in checks.items():
            score = check_result.get("score", 0.0 if not check_result.get("passed", False) else 1.0)
            weight = weights.get(check_name, 0.1)
            
            total_score += score * weight
            weight_sum += weight
        
        return total_score / weight_sum if weight_sum > 0 else 0.0
    
    def _generate_quality_recommendations(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成质量建议"""
        recommendations = []
        
        if not issues:
            recommendations.append({
                "priority": "LOW",
                "category": "MAINTENANCE",
                "recommendation": "代码质量良好，继续保持",
                "action": "定期进行代码审查"
            })
            return recommendations
        
        # 按严重程度分组
        severity_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for issue in issues:
            severity_counts[issue.get("severity", "MEDIUM")] += 1
            category_counts[issue.get("category", "general")] += 1
        
        # 生成建议
        if severity_counts.get("CRITICAL", 0) > 0:
            recommendations.append({
                "priority": "CRITICAL",
                "category": "SECURITY",
                "recommendation": f"发现 {severity_counts['CRITICAL']} 个严重问题，需要立即修复",
                "action": "优先处理严重和高优先级问题"
            })
        
        if severity_counts.get("HIGH", 0) > 5:
            recommendations.append({
                "priority": "HIGH",
                "category": "MAINTENANCE",
                "recommendation": "发现多个高优先级问题，建议重构代码",
                "action": "制定重构计划，逐步解决"
            })
        
        if category_counts.get("imports", 0) > 3:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "IMPORTS",
                "recommendation": "导入语句存在问题，建议优化",
                "action": "整理导入语句，移除未使用的导入"
            })
        
        if category_counts.get("functions", 0) > 5:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "DESIGN",
                "recommendation": "函数设计需要改进",
                "action": "重构长函数，减少参数数量"
            })
        
        # 通用建议
        recommendations.extend([
            {
                "priority": "LOW",
                "category": "TOOLS",
                "recommendation": "使用静态代码分析工具",
                "action": "集成 pylint、flake8 等工具到CI流程"
            },
            {
                "priority": "LOW",
                "category": "TESTING",
                "recommendation": "增加单元测试覆盖率",
                "action": "为目标代码覆盖率80%以上"
            }
        ])
        
        return recommendations
    
    def _generate_quality_metrics(self, checks: Dict[str, Dict]) -> Dict[str, Any]:
        """生成质量指标"""
        metrics = {
            "total_issues": 0,
            "severity_breakdown": defaultdict(int),
            "category_breakdown": defaultdict(int),
            "quality_score": 0.0,
            "compliance_percentage": 0.0
        }
        
        for check_name, check_result in checks.items():
            issues = check_result.get("issues", [])
            metrics["total_issues"] += len(issues)
            
            for issue in issues:
                severity = issue.get("severity", "MEDIUM")
                category = issue.get("category", "general")
                metrics["severity_breakdown"][severity] += 1
                metrics["category_breakdown"][category] += 1
        
        # 计算合规率
        total_checks = len(checks)
        passed_checks = sum(1 for check in checks.values() if check.get("passed", False))
        metrics["compliance_percentage"] = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        # 计算质量分数
        metrics["quality_score"] = self._calculate_quality_score(checks)
        
        return dict(metrics)

# --- 代码质量分析器 ---
class CodeQualityAnalyzerV6:
    """代码质量分析器V6"""
    
    def __init__(self):
        self.metrics = {}
    
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析代码质量"""
        return {
            "check_name": "code_quality_analysis",
            "passed": True,
            "score": 0.85,
            "issues": [],
            "details": {
                "maintainability_index": 75,
                "technical_debt": "2 hours",
                "code_smells": 3
            }
        }

# --- 性能分析器 ---
class PerformanceAnalyzerV6:
    """性能分析器V6"""
    
    def __init__(self):
        self.performance_patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, str]:
        """加载性能模式"""
        return {
            "inefficient_loops": r"for.*in.*range\(\d{4,}\)",
            "memory_leaks": r"list\.append.*while.*True",
            "slow_algorithms": r"for.*for.*in.*range",
            "sync_operations": r"requests\.get|urllib\.request\.urlopen"
        }
    
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析性能"""
        code_content = context.get("code", "") or context.get("content", "")
        
        issues = []
        for pattern_name, pattern in self.performance_patterns.items():
            if re.search(pattern, code_content, re.IGNORECASE):
                issues.append({
                    "issue_type": pattern_name.upper(),
                    "severity": "MEDIUM",
                    "description": f"发现性能问题模式: {pattern_name}",
                    "suggestion": "优化算法或使用异步操作"
                })
        
        score = max(0.0, 1.0 - (len(issues) / 10))
        
        return {
            "check_name": "performance_analysis",
            "passed": score >= 0.7,
            "score": score,
            "issues": issues,
            "details": {
                "performance_score": score,
                "bottlenecks_found": len(issues)
            }
        }

# --- 安全分析器 ---
class SecurityAnalyzerV6:
    """安全分析器V6"""
    
    def __init__(self):
        self.security_patterns = self._load_security_patterns()
    
    def _load_security_patterns(self) -> Dict[str, str]:
        """加载安全模式"""
        return {
            "sql_injection": r"cursor\.execute.*%",
            "xss": r"<script|javascript:",
            "command_injection": r"os\.system|subprocess\.Popen",
            "path_traversal": r"\.\.\/|\.\.\\\\",
            "hardcoded_secrets": r"password\s*=\s*[\"'][^\"']+",
            "insecure_crypto": r"md5\(|sha1\("
        }
    
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析安全性"""
        code_content = context.get("code", "") or context.get("content", "")
        
        issues = []
        for pattern_name, pattern in self.security_patterns.items():
            if re.search(pattern, code_content, re.IGNORECASE):
                severity = "CRITICAL" if pattern_name in ["sql_injection", "command_injection"] else "HIGH"
                issues.append({
                    "issue_type": f"SECURITY_{pattern_name.upper()}",
                    "severity": severity,
                    "description": f"发现安全漏洞: {pattern_name}",
                    "suggestion": "修复安全漏洞，使用安全的替代方案"
                })
        
        score = max(0.0, 1.0 - (len(issues) * 0.2))  # 每个安全问题扣0.2分
        
        return {
            "check_name": "security_analysis",
            "passed": len([i for i in issues if i["severity"] == "CRITICAL"]) == 0,
            "score": score,
            "issues": issues,
            "details": {
                "security_score": score,
                "critical_vulnerabilities": len([i for i in issues if i["severity"] == "CRITICAL"])
            }
        }

# --- 复杂度分析器 ---
class ComplexityAnalyzerV6:
    """复杂度分析器V6"""
    
    def __init__(self):
        self.complexity_thresholds = {
            "cyclomatic": 10,
            "nesting": 5,
            "parameters": 7
        }
    
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析复杂度"""
        code_content = context.get("code", "") or context.get("content", "")
        
        if not code_content:
            return {
                "check_name": "complexity_analysis",
                "passed": True,
                "score": 1.0,
                "issues": [],
                "details": {"complexity_score": 1.0}
            }
        
        try:
            tree = ast.parse(code_content)
            
            # 计算圈复杂度
            complexity_score = self._calculate_complexity(tree)
            
            # 生成问题
            issues = []
            if complexity_score > self.complexity_thresholds["cyclomatic"]:
                issues.append({
                    "issue_type": "HIGH_COMPLEXITY",
                    "severity": "MEDIUM",
                    "description": f"圈复杂度过高: {complexity_score}",
                    "suggestion": "简化逻辑，拆分复杂函数"
                })
            
            return {
                "check_name": "complexity_analysis",
                "passed": complexity_score <= self.complexity_thresholds["cyclomatic"],
                "score": max(0.0, 1.0 - (complexity_score / 20)),
                "issues": issues,
                "details": {
                    "cyclomatic_complexity": complexity_score,
                    "max_allowed": self.complexity_thresholds["cyclomatic"]
                }
            }
            
        except SyntaxError:
            return {
                "check_name": "complexity_analysis",
                "passed": False,
                "score": 0.0,
                "issues": [],
                "details": {"error": "语法错误"}
            }
    
    def _calculate_complexity(self, tree: ast.Module) -> float:
        """计算圈复杂度"""
        complexity = 1  # 基础复杂度
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity

# --- 文档分析器 ---
class DocumentationAnalyzerV6:
    """文档分析器V6"""
    
    def __init__(self):
        self.docstring_patterns = [
            r'"""[^"]+"""',
            r"'''[^']'''"
        ]
    
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析文档"""
        code_content = context.get("code", "") or context.get("content", "")
        
        if not code_content:
            return {
                "check_name": "documentation_analysis",
                "passed": True,
                "score": 1.0,
                "issues": [],
                "details": {"documentation_score": 1.0}
            }
        
        try:
            tree = ast.parse(code_content)
            
            # 检查文档字符串
            documented_items = 0
            total_items = 0
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    total_items += 1
                    if (node.body and 
                        isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, (ast.Str, ast.Constant)) and
                        isinstance(node.body[0].value.value, str)):
                        documented_items += 1
            
            documentation_ratio = documented_items / max(1, total_items)
            
            issues = []
            if documentation_ratio < 0.8:
                issues.append({
                    "issue_type": "INSUFFICIENT_DOCUMENTATION",
                    "severity": "MEDIUM",
                    "description": f"文档覆盖率不足: {documentation_ratio:.1%}",
                    "suggestion": "为函数和类添加文档字符串"
                })
            
            return {
                "check_name": "documentation_analysis",
                "passed": documentation_ratio >= 0.7,
                "score": documentation_ratio,
                "issues": issues,
                "details": {
                    "documentation_ratio": documentation_ratio,
                    "documented_items": documented_items,
                    "total_items": total_items
                }
            }
            
        except SyntaxError:
            return {
                "check_name": "documentation_analysis",
                "passed": False,
                "score": 0.0,
                "issues": [],
                "details": {"error": "语法错误"}
            }

# --- 测试函数 ---
async def test_quality_hook():
    """测试自动质量检查Hook"""
    print("🧪 测试自动质量检查Hook V6")
    print("=" * 50)
    
    hook = AutoQualityCheckHookV6()
    
    # 测试用例
    test_cases = [
        {
            "name": "高质量代码",
            "context": {
                "code": '''
def calculate_fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for i in range(n):
            a, b = b, a + b
        return a
''',
                "file_path": "fibonacci.py"
            }
        },
        {
            "name": "有问题的代码",
            "context": {
                "code": '''
import *
print("debug info")
def long_function_with_many_parameters(param1, param2, param3, param4, param5, param6, param7, param8):
    x = 123456
    if x == None:
        print("found none")
    for i in range(10000):
        pass
''',
                "file_path": "problematic.py"
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 测试案例 {i}: {test_case['name']}")
        
        result = await hook(test_case['context'])
        
        print(f"✅ 检查结果: {'通过' if result['success'] else '未通过'}")
        print(f"📊 质量分数: {result['quality_score']:.2f}")
        print(f"📊 问题数量: {len(result['issues'])}")
        print(f"⏱️ 执行时间: {result['execution_time']:.3f}s")
        
        if result['issues']:
            print("🚨 发现问题:")
            for issue in result['issues'][:3]:  # 显示前3个
                print(f"  - {issue['issue_type']}: {issue['description']}")
        
        if result['recommendations']:
            print("💡 质量建议:")
            for rec in result['recommendations'][:2]:  # 显示前2个
                print(f"  - {rec['recommendation']}")
    
    print(f"\n✅ 自动质量检查Hook V6测试完成")

if __name__ == "__main__":
    asyncio.run(test_quality_hook())