#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌌 自动测试与修复系统 V5 (Auto Test & Heal System V5)
智能化的自动测试、bug检测和自动修复系统，实现零人工值守的全自动质量保障。
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import asyncio
import logging
import traceback
import time
import subprocess
import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import threading
import sqlite3
import importlib.util
import inspect

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class IssueSeverity(Enum):
    """问题严重程度"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class IssueType(Enum):
    """问题类型"""
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    RUNTIME_ERROR = "runtime_error"
    LOGIC_ERROR = "logic_error"
    PERFORMANCE = "performance"
    SECURITY = "security"
    COMPATIBILITY = "compatibility"
    CONFIGURATION = "configuration"
    TEST_FAILURE = "test_failure"
    DEPRECATION = "deprecation"

@dataclass
class Issue:
    """问题记录"""
    id: str
    type: IssueType
    severity: IssueSeverity
    file_path: str
    line_number: Optional[int]
    description: str
    stack_trace: str
    context: Dict[str, Any] = field(default_factory=dict)
    detected_at: float = field(default_factory=time.time)
    fixed: bool = False
    fix_attempts: int = 0
    resolution: Optional[str] = None
    auto_fixable: bool = True
    confidence: float = 0.8

@dataclass
class FixStrategy:
    """修复策略"""
    issue_type: IssueType
    pattern: str
    replacement: str
    confidence: float
    description: str
    examples: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)

class AutoTestHealSystemV5:
    """
    自动测试与修复系统 V5
    """
    
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self.db_path = self.project_root / "data" / "test_heal_v5.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()
        
        # 测试和修复配置
        self.config = {
            "auto_fix": True,
            "test_on_import": True,
            "continuous_monitoring": True,
            "max_fix_attempts": 3,
            "confidence_threshold": 0.7
        }
        
        # 问题检测器
        self.detectors = self._initialize_detectors()
        
        # 修复策略库
        self.fix_strategies = self._load_fix_strategies()
        
        # 监控状态
        self.monitoring = False
        self.monitor_thread = None
        self.last_scan_time = 0
        
        # 统计信息
        self.stats = {
            "total_issues": 0,
            "fixed_issues": 0,
            "auto_fixed_issues": 0,
            "manual_intervention_required": 0,
            "last_scan": None
        }
        
        logger.info("自动测试与修复系统V5初始化完成")
    
    def _init_db(self):
        """初始化数据库"""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS issues (
                    id TEXT PRIMARY KEY,
                    type TEXT,
                    severity TEXT,
                    file_path TEXT,
                    line_number INTEGER,
                    description TEXT,
                    stack_trace TEXT,
                    context TEXT,
                    detected_at REAL,
                    fixed BOOLEAN DEFAULT FALSE,
                    fix_attempts INTEGER DEFAULT 0,
                    resolution TEXT,
                    auto_fixable BOOLEAN DEFAULT TRUE,
                    confidence REAL DEFAULT 0.8
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS fix_history (
                    id TEXT PRIMARY KEY,
                    issue_id TEXT,
                    strategy_id TEXT,
                    applied_at REAL,
                    success BOOLEAN,
                    result TEXT,
                    confidence REAL
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS test_results (
                    id TEXT PRIMARY KEY,
                    test_name TEXT,
                    status TEXT,
                    executed_at REAL,
                    duration REAL,
                    result TEXT,
                    metrics TEXT
                )
            """)
    
    def _initialize_detectors(self) -> Dict[str, Any]:
        """初始化检测器"""
        detectors = {}
        
        # 语法检测器
        detectors["syntax"] = SyntaxDetector()
        
        # 导入检测器
        detectors["import"] = ImportDetector()
        
        # 运行时错误检测器
        detectors["runtime"] = RuntimeErrorDetector()
        
        # 性能检测器
        detectors["performance"] = PerformanceDetector()
        
        # 安全检测器
        detectors["security"] = SecurityDetector()
        
        # 兼容性检测器
        detectors["compatibility"] = CompatibilityDetector()
        
        # 测试失败检测器
        detectors["test"] = TestFailureDetector()
        
        return detectors
    
    def _load_fix_strategies(self) -> Dict[IssueType, List[FixStrategy]]:
        """加载修复策略"""
        strategies = {
            IssueType.SYNTAX_ERROR: [
                FixStrategy(
                    issue_type=IssueType.SYNTAX_ERROR,
                    pattern=r"async def\s+(\w+)\s*\([^)]*\s*:",
                    replacement="async def \\1(\\2):",
                    confidence=0.9,
                    description="修复async函数定义语法",
                    examples=["async def my_func():", "async def process(data):"]
                ),
                FixStrategy(
                    issue_type=IssueType.SYNTAX_ERROR,
                    pattern=r"def\s+(\w+)\s*\([^)]*\s*:",
                    replacement="def \\1(\\2):",
                    confidence=0.9,
                    description="修复函数定义语法",
                    examples=["def my_func():", "def process(data):"]
                ),
                FixStrategy(
                    issue_type=IssueType.SYNTAX_ERROR,
                    pattern=r"return\s+([^;]+);",
                    replacement="return \\1;",
                    confidence=0.8,
                    description="修复return语句缺少冒号",
                    examples=["return x", "return result"]
                )
            ],
            IssueType.IMPORT_ERROR: [
                FixStrategy(
                    issue_type=IssueType.IMPORT_ERROR,
                    pattern=r"from\s+(\S+)\s+import\s+(\S+)",
                    replacement="from \\1 import \\2",
                    confidence=0.9,
                    description="修复import语句格式",
                    examples=["from os import path", "from sys import argv"]
                ),
                FixStrategy(
                    issue_type=IssueType.IMPORT_ERROR,
                    pattern=r"import\s+(\S+)",
                    replacement="import \\1",
                    confidence=0.8,
                    description="修复单行import语句",
                    examples=["import os", "import sys"]
                )
            ],
            IssueType.RUNTIME_ERROR: [
                FixStrategy(
                    issue_type=IssueType.RUNTIME_ERROR,
                    pattern=r"NameError:\s+'([^']+)'\s+is\s+not\s+defined",
                    replacement="# 检查变量定义\n# \\1 可能需要在此处定义",
                    confidence=0.7,
                    description="修复NameError",
                    examples=["NameError: 'var' is not defined"]
                ),
                FixStrategy(
                    issue_type=IssueType.RUNTIME_ERROR,
                    pattern=r"AttributeError:\s+'([^']+)'\s+object\s+has\s+no\s+attribute\s+'([^']+)'",
                    replacement="# 检查对象属性\n# 确保 \\1 对象有 \\2 属性",
                    confidence=0.7,
                    description="修复AttributeError",
                    examples=["AttributeError: 'obj' has no attribute 'attr'"]
                )
            ],
            IssueType.PERFORMANCE: [
                FixStrategy(
                    issue_type=IssueType.PERFORMANCE,
                    pattern=r"for\s+(\w+)\s+in\s+(.+):\s*#.*",
                    replacement="for \\1 in \\2:\n    # 优化循环性能\n    pass",
                    confidence=0.6,
                    description="优化循环性能",
                    examples=["for item in data: # 慢循环"]
                ),
                FixStrategy(
                    issue_type=IssueType.PERFORMANCE,
                    pattern=r"list\((.*?))\s*\*\s+",
                    replacement="list((\\1))",
                    confidence=0.8,
                    description="优化列表生成",
                    examples=["list((x for x in items))"]
                )
            ],
            IssueType.SECURITY: [
                FixStrategy(
                    issue_type=IssueType.SECURITY,
                    pattern=r"eval\s*\(",
                    replacement="# 避免使用eval\n# 使用更安全的方法",
                    confidence=1.0,
                    description="移除eval使用",
                    examples=["eval("]
                ),
                FixStrategy(
                    issue_type=IssueType.SECURITY,
                    pattern=r"exec\s*\(",
                    replacement="# 避免使用exec\n# 使用更安全的方法",
                    confidence=1.0,
                    description="移除exec使用",
                    examples=["exec("]
                )
            ]
        }
        
        return strategies
    
    async def scan_project(self, path: str = None) -> Dict[str, Any]:
        """扫描项目检测问题"""
        scan_path = Path(path) if path else self.project_root
        
        logger.info(f"开始扫描项目: {scan_path}")
        
        issues = []
        
        # 扫行各种检测器
        for detector_name, detector in self.detectors.items():
            try:
                detector_issues = await detector.detect(scan_path)
                issues.extend(detector_issues)
                logger.info(f"{detector_name}检测器发现 {len(detector_issues)}个问题")
            except Exception as e:
                logger.error(f"{detector_name}检测器运行失败: {e}")
        
        # 存储问题到数据库
        await self._store_issues(issues)
        
        # 更新统计
        self.stats["total_issues"] = len(issues)
        self.stats["last_scan"] = datetime.now()
        self.last_scan_time = time.time()
        
        # 分析问题分布
        issue_summary = self._analyze_issues(issues)
        
        result = {
            "scan_path": str(scan_path),
            "total_issues": len(issues),
            "issues_by_severity": issue_summary["by_severity"],
            "issues_by_type": issue_summary["by_type"],
            "auto_fixable": len([i for i in issues if i.auto_fixable]),
            "requires_manual": len([i for i in issues if not i.auto_fixable]),
            "scan_time": time.time() - (self.last_scan_time - len(issues) * 0.1)
        }
        
        logger.info(f"扫描完成，发现{len(issues)}个问题")
        
        return result
    
    async def auto_fix_issues(self, issue_ids: List[str] = None) -> Dict[str, Any]:
        """自动修复问题"""
        if issue_ids:
            # 修复指定问题
            issues_to_fix = await self._get_issues_by_ids(issue_ids)
        else:
            # 修复所有可自动修复的问题
            issues_to_fix = await self._get_auto_fixable_issues()
        
        if not issues_to_fix:
            return {"message": "没有需要修复的问题"}
        
        logger.info(f"开始自动修复{len(issues_to_fix)}个问题")
        
        fix_results = []
        
        for issue in issues_to_fix:
            try:
                fix_result = await self._fix_issue(issue)
                fix_results.append(fix_result)
                
                if fix_result["success"]:
                    self.stats["auto_fixed_issues"] += 1
                    logger.info(f"成功修复问题: {issue.description[:50]}...")
                else:
                    logger.warning(f"修复失败: {issue.description[:50]}...")
                    
            except Exception as e:
                logger.error(f"修复问题时出错: {e}")
                fix_results.append(f"修复失败: {str(e)}")
        
        return {
            "total_attempted": len(issues_to_fix),
            "successful_fixes": len([r for r in fix_results if "成功" in r]),
            "results": fix_results
        }
    
    async def _fix_issue(self, issue: Issue) -> Dict[str, Any]:
        """修复单个问题"""
        strategies = self.fix_strategies.get(issue.type, [])
        
        for strategy in strategies:
            if strategy.confidence < self.config["confidence_threshold"]:
                continue
            
            try:
                # 读取文件内容
                with open(issue.file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 应用修复策略
                new_content, matches = re.subn(
                    strategy.pattern,
                    strategy.replacement,
                    content,
                    count=1
                )
                
                if matches > 0:
                    # 写回文件
                    with open(issue.file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    
                    # 更新问题状态
                    await self._mark_issue_fixed(
                        issue.id,
                        f"应用策略: {strategy.description}",
                        strategy.confidence
                    )
                    
                    return {
                        "success": True,
                        "issue_id": issue.id,
                        "strategy": strategy.description,
                        "matches": matches,
                        "confidence": strategy.confidence
                    }
                    
            except Exception as e:
                logger.error(f"应用修复策略失败: {e}")
                continue
        
        return {
            "success": False,
            "issue_id": issue.id,
            "error": "没有找到合适的修复策略"
        }
    
    async def run_continuous_monitoring(self):
        """运行持续监控"""
        self.monitoring = True
        
        while self.monitoring:
            try:
                # 扫行扫描
                scan_result = await self.scan_project()
                
                # 自动修复问题
                if scan_result["auto_fixable"] > 0:
                    fix_result = await self.auto_fix_issues()
                    logger.info(f"自动修复完成: {fix_result['successful_fixes']}/{fix_result['total_attempted']}")
                
                # 等待一段时间
                await asyncio.sleep(60)  # 每分钟扫描一次
                
            except Exception as e:
                logger.error(f"监控过程中出错: {e}")
                await asyncio.sleep(60)
    
    def start_monitoring(self):
        """启动监控"""
        if not self.monitor_thread or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(
                target=self.run_continuous_monitoring,
                daemon=True
            )
            self.monitor_thread.start()
            logger.info("持续监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            self.monitor_thread = None
            logger.info("持续监控已停止")
    
    async def run_tests(self, test_paths: List[str] = None) -> Dict[str, Any]:
        """运行测试"""
        if not test_paths:
            # 查找测试文件
            test_paths = self._find_test_files()
        
        if not test_paths:
            return {"message": "未找到测试文件"}
        
        logger.info(f"运行{len(test_paths)}个测试文件")
        
        test_results = []
        
        for test_path in test_paths:
            try:
                result = await self._run_single_test(test_path)
                test_results.append(result)
            except Exception as e:
                test_results.append({
                    "test_path": test_path,
                    "status": "error",
                    "error": str(e)
                })
        
        # 存储测试结果
        await self._store_test_results(test_results)
        
        # 分析测试结果
        summary = self._analyze_test_results(test_results)
        
        return {
            "total_tests": len(test_paths),
            "passed": summary["passed"],
            "failed": summary["failed"],
            "coverage": summary["coverage"],
            "results": test_results
        }
    
    async def _run_single_test(self, test_path: str) -> Dict[str, Any]:
        """运行单个测试"""
        start_time = time.time()
        
        try:
            # 动态导入测试模块
            spec = importlib.util.spec_from_file_location(test_path)
            module = importlib.util.module_from_spec(spec)
            
            # 查找测试函数
            test_functions = [
                name for name, obj in inspect.getmembers(module)
                if name.startswith('test_') and inspect.isfunction(obj)
            ]
            
            if not test_functions:
                return {
                    "test_path": test_path,
                    "status": "no_tests",
                    "message": "未找到测试函数"
                }
            
            # 运行所有测试函数
            test_results = []
            for test_func in test_functions:
                try:
                    # 创建测试实例
                    if test_func.__code__.co_argcount == 0:
                        # 无参数测试
                        result = test_func()
                    else:
                        # 有参数测试
                        result = test_func(None)  # 简化处理
                    
                    test_results.append({
                        "function": test_func.__name__,
                        "status": "passed",
                        "result": str(result)
                    })
                    
                except Exception as e:
                    test_results.append({
                        "function": test_func.__name__,
                        "status": "failed",
                        "error": str(e)
                    })
            
            duration = time.time() - start_time
            
            return {
                "test_path": test_path,
                "status": "completed",
                "duration": duration,
                "results": test_results
            }
            
        except Exception as e:
            return {
                "test_path": test_path,
                "status": "error",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    def _find_test_files(self) -> List[str]:
        """查找测试文件"""
        test_patterns = [
            "**/test_*.py",
            "**/tests/**/*.py",
            "**/*_test.py",
            "**/test_*.py"
        ]
        
        test_files = []
        for pattern in test_patterns:
            test_files.extend(self.project_root.glob(pattern))
        
        return [str(f) for f in test_files if f.is_file()]
    
    async def _store_issues(self, issues: List[Issue]):
        """存储问题到数据库"""
        with self.conn:
            for issue in issues:
                self.conn.execute(
                    """
                    INSERT INTO issues 
                    (id, type, severity, file_path, line_number, description, stack_trace, context, detected_at, fixed, fix_attempts, resolution, auto_fixable, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        issue.id,
                        issue.type.value,
                        issue.severity.value,
                        issue.file_path,
                        issue.line_number,
                        issue.description,
                        issue.stack_trace,
                        json.dumps(issue.context),
                        issue.detected_at,
                        issue.fixed,
                        issue.fix_attempts,
                        issue.resolution,
                        issue.auto_fixable,
                        issue.confidence
                    )
                )
    
    async def _get_issues_by_ids(self, issue_ids: List[str]) -> List[Issue]:
        """根据ID获取问题"""
        if not issue_ids:
            return []
        
        placeholders = ",".join(["?" for _ in issue_ids])
        
        with self.conn:
            cursor = self.conn.execute(
                f"SELECT * FROM issues WHERE id IN ({placeholders})",
                issue_ids
            )
            
            issues = []
            for row in cursor:
                issues.append(Issue(
                    id=row[0],
                    type=IssueType(row[1]),
                    severity=IssueSeverity(row[2]),
                    file_path=row[3],
                    line_number=row[4],
                    description=row[5],
                    stack_trace=row[6],
                    context=json.loads(row[7]),
                    detected_at=row[8],
                    fixed=bool(row[9]),
                    fix_attempts=row[10],
                    resolution=row[11],
                    auto_fixable=bool(row[12]),
                    confidence=row[13]
                ))
        
        return issues
    
    async def _get_auto_fixable_issues(self) -> List[Issue]:
        """获取可自动修复的问题"""
        with self.conn:
            cursor = self.conn.execute(
                "SELECT * FROM issues WHERE auto_fixable = 1 AND fixed = 0"
            )
            
            issues = []
            for row in cursor:
                issues.append(Issue(
                    id=row[0],
                    type=IssueType(row[1]),
                    severity=IssueSeverity(row[2]),
                    file_path=row[3],
                    line_number=row[4],
                    description=row[5],
                    stack_trace=row[6],
                    context=json.loads(row[7]),
                    detected_at=row[8],
                    fixed=bool(row[9]),
                    fix_attempts=row[10],
                    resolution=row[11],
                    auto_fixable=bool(row[12]),
                    confidence=row[13]
                ))
        
        return issues
    
    async def _mark_issue_fixed(self, issue_id: str, resolution: str, confidence: float):
        """标记问题已修复"""
        with self.conn:
            self.conn.execute(
                "UPDATE issues SET fixed = 1, resolution = ?, fix_attempts = fix_attempts + 1 WHERE id = ?",
                (resolution, issue_id)
            )
    
    async def _store_test_results(self, results: List[Dict[str, Any]]):
        """存储测试结果"""
        with self.conn:
            for result in results:
                test_id = str(uuid.uuid4())
                
                self.conn.execute(
                    """
                    INSERT INTO test_results 
                    (id, test_name, status, executed_at, duration, result, metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        test_id,
                        result.get("test_path", ""),
                        result.get("status", "unknown"),
                        result.get("executed_at", time.time()),
                        result.get("duration", 0),
                        str(result.get("result", "")),
                        json.dumps({
                            "results": result.get("results", [])
                        })
                    )
                )
    
    def _analyze_issues(self, issues: List[Issue]) -> Dict[str, Any]:
        """分析问题分布"""
        by_severity = defaultdict(int)
        by_type = defaultdict(int)
        
        for issue in issues:
            by_severity[issue.severity.value] += 1
            by_type[issue.type.value] += 1
        
        return {
            "by_severity": dict(by_severity),
            "by_type": dict(by_type)
        }
    
    def _analyze_test_results(self, results: List[Dict[str, any]]) -> Dict[str, Any]:
        """分析测试结果"""
        passed = len([r for r in results if r.get("status") == "passed"])
        failed = len([r for r in results if r.get("status") == "failed"])
        
        # 计算覆盖率（简化版）
        total_functions = sum(len(r.get("results", [])) for r in results if r.get("results"))
        passed_functions = sum(
            len([item for item in r.get("results", []) if item.get("status") == "passed"])
            for r in results if r.get("results")
        )
        
        coverage = (passed_functions / total_functions * 100) if total_functions > 0 else 0
        
        return {
            "passed": passed,
            "failed": failed,
            "coverage": coverage
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        with self.conn:
            cursor = self.conn.execute(
                """
                SELECT COUNT(*) as total_issues,
                       COUNT(CASE WHEN fixed = 1) as fixed_issues,
                       COUNT(CASE WHEN auto_fixable = 1) as auto_fixable_issues,
                       MAX(detected_at) as last_issue_time
                FROM issues
                """
            )
            
            row = cursor.fetchone()
            
            if row:
                total_issues, fixed_issues, auto_fixable_issues, last_issue_time = row
                
                health_score = 0.0
                if total_issues > 0:
                    health_score = (fixed_issues / total_issues) * 100
                
                return {
                    "health_score": health_score,
                    "total_issues": total_issues,
                    "fixed_issues": fixed_issues,
                    "auto_fixable_issues": auto_fixable_issues,
                    "last_issue_time": datetime.fromtimestamp(last_issue_time) if last_issue_time else None,
                    "monitoring": self.monitoring,
                    "stats": self.stats
                }
        
        return {
            "health_score": 100.0,
            "total_issues": 0,
            "fixed_issues": 0,
            "auto_fixable_issues": 0,
            "last_issue_time": None,
            "monitoring": self.monitoring,
            "stats": self.stats
        }

# 检测器基类
class BaseDetector:
    """检测器基类"""
    
    def __init__(self):
        self.name = self.__class__.__name__
    
    async def detect(self, path: Path) -> List[Issue]:
        """检测问题"""
        raise NotImplementedError

class SyntaxDetector(BaseDetector):
    """语法检测器"""
    
    def __init__(self):
        super().__init__()
        self.name = "syntax_detector"
    
    async def detect(self, path: Path) -> List[Issue]:
        """检测语法错误"""
        issues = []
        
        for py_file in path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 尝试解析AST
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    # 提取错误信息
                    error_msg = str(e)
                    line_num = self._extract_line_number(error_msg)
                    
                    issue = Issue(
                        id=str(uuid.uuid4()),
                        type=IssueType.SYNTAX_ERROR,
                        severity=IssueSeverity.HIGH,
                        file_path=str(py_file),
                        line_number=line_num,
                        description=f"语法错误: {error_msg}",
                        stack_trace=traceback.format_exc(),
                        context={
                            "file_type": "python",
                            "file_size": len(content)
                        },
                        auto_fixable=True,
                        confidence=0.9
                    )
                    issues.append(issue)
                    
            except Exception as e:
                logger.error(f"读取文件{py_file}时出错: {e}")
        
        return issues
    
    def _extract_line_number(self, error_msg: str) -> Optional[int]:
        """从错误信息中提取行号"""
        match = re.search(r"line (\d+)", error_msg)
        return int(match.group(1)) if match else None

class ImportDetector(BaseDetector):
    """导入检测器"""
    
    def __init__(self):
        super().__init__()
        self.name = "import_detector"
    
    async def detect(self, path: Path) -> List[Issue]:
        """检测导入错误"""
        issues = []
        
        for py_file in path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查导入语句
                import_lines = [
                    line.strip() for line in content.split('\n') if line.strip().startswith('import ') or line.strip().startswith('from ')
                ]
                
                for line in import_lines:
                    try:
                        # 验证导入语法
                        compile(line, '<string>', 'exec')
                    except SyntaxError as e:
                        line_num = content.find(line) // len(content.split('\n')) + 1
                        
                        issue = Issue(
                            id=str(uuid.uuid4()),
                            type=IssueType.IMPORT_ERROR,
                            severity=IssueSeverity.HIGH,
                            file_path=str(py_file),
                            line_number=line_num,
                            description=f"导入错误: {str(e)}",
                            stack_trace=traceback.format_exc(),
                            context={"import_line": line},
                            auto_fixable=True,
                            confidence=0.8
                        )
                        issues.append(issue)
                        
            except Exception as e:
                logger.error(f"读取文件{py_file}时出错: {e}")
        
        return issues

class RuntimeErrorDetector(BaseDetector):
    """运行时错误检测器"""
    
    def __init__(self):
        super().__init__()
        self.name = "runtime_detector"
    
    async def detect(self, path: Path) -> List[Issue]:
        """检测运行时错误"""
        issues = []
        
        # 检查常见的运行时错误模式
        error_patterns = [
            (r"NameError:\s+'([^']+)'\s+is\s+not\s+defined", IssueType.RUNTIME_ERROR, IssueSeverity.HIGH),
            (r"AttributeError:\s+'([^']+)'\s+object\s+has\s+no\s+attribute\s+'([^']+)'", IssueType.RUNTIME_ERROR, IssueSeverity.HIGH),
            (r"TypeError:\s+'([^']+)'\s+object\s+is\s+not\s+callable", IssueType.RUNTIME_ERROR, IssueSeverity.HIGH),
            (r"ValueError:\s+", IssueType.RUNTIME_ERROR, IssueSeverity.MEDIUM),
            (r"KeyError:\s+", IssueType.RUNTIME_ERROR, IssueSeverity.HIGH),
            (r"IndexError:\s+", IssueType.RUPTIME_ERROR, IssueSeverity.MEDIUM)
        ]
        
        for py_file in path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern, issue_type, severity in error_patterns:
                    if re.search(pattern, content):
                        matches = re.findall(pattern, content)
                        for match in matches:
                            line_num = content.find(match) // len(content.split('\n')) + 1
                            
                            issue = Issue(
                                id=str(uuid.uuid4()),
                                type=issue_type,
                                severity=severity,
                                file_path=str(py_file),
                                line_number=line_num,
                                description=f"运行时错误: {match}",
                                stack_trace="",
                                context={"error_pattern": pattern},
                                auto_fixable=False,
                                confidence=0.7
                            )
                            issues.append(issue)
                            
            except Exception as e:
                logger.error(f"读取文件{py_file}时出错: {e}")
        
        return issues

class PerformanceDetector(BaseDetector):
    """性能检测器"""
    
    def __init__(self):
        super().__init__()
        self.name = "performance_detector"
    
    async def detect(self, path: Path) -> List[Issue]:
        """检测性能问题"""
        issues = []
        
        performance_patterns = [
            (r"for\s+\w+\s+in\s+(.+):\s*#.*", IssueType.PERFORMANCE, IssueType.MEDIUM),
            (r"list\((.*?))\s*\*\s+", IssueType.PERFORMANCE, IssueType.MEDIUM),
            (r"\.join\(.*?\)\s*\*\s+", IssueType.PERFORMANCE, IssueType.MEDIUM),
            (r"while\s+True:\s*#.*", IssueType.PERFORMANCE, IssueType.MEDIUM),
            (r"range\(.*?\)\s*\*\s+", IssueType.PERFORMANCE, IssueType.MEDIUM)
        ]
        
        for py_file in path.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                for pattern, issue_type, severity in performance_patterns:
                    if re.search(pattern, content):
                        matches = re.findall(pattern, content)
                        for match in matches:
                            line_num = content.find(match) // len(content.split('\n')) + 1
                            
                            issue = Issue(
                                id=str(uuid.uuid4()),
                                type=issue_type,
                                severity=severity,
                                file_path=str(py_file),
                                line_number=line_num,
                                description=f"性能问题: {match}",
                                stack_trace="",
                                context={"performance_pattern": pattern},
                                auto_fixable=True,
                                confidence=0.6
                            )
                            issues.append(issue)
                            
            except Exception as e:
                logger.error(f"读取文件{py_file}时出错: {e}")
        
        return issues

class SecurityDetector(BaseDetector):
    """安全检测器"""
    
    def __init__(self):
        super().__init__()
        self.name = "security_detector"
    
    async def detect(self, path: Path) -> List[Issue]:
        """检测安全问题"""
        issues = []
        
        security_patterns = [
            (r"eval\s*\(", IssueType.SECURITY, IssueType.CRITICAL),
            (r"exec\s*\(", IssueType.SECURITY, IssueType.CRITICAL),
            (r"subprocess\.\w+\(", IssueType.SECURITY, IssueType.HIGH),
            (r"os\.system\(", IssueType.SECURITY, IssueType.HIGH),
            (r"open\(", IssueType.SECURITY, IssueType.MEDIUM),
            (r"file\(", IssueType.SECURITY, IssueType.MEDIUM)
        ]
        
        for py_file in path.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                for pattern, issue_type, severity in security_patterns:
                    if re.search(pattern, content):
                        matches = re.findall(pattern, content)
                        for match in matches:
                            line_num = content.find(match) // len(content.split('\n')) + 1
                            
                            issue = Issue(
                                id=str(uuid.uuid4()),
                                type=issue_type,
                                severity=severity,
                                file_path=str(py_file),
                                line_number=line_num,
                                description=f"安全问题: 使用了不安全的{pattern[:-2]}",
                                stack_trace="",
                                context={"security_pattern": pattern},
                                auto_fixable=pattern in ["eval", "exec"],
                                confidence=0.95
                            )
                            issues.append(issue)
                            
            except Exception as e:
                logger.error(f"读取文件{py_file}时出错: {e}")
        
        return issues

class CompatibilityDetector(BaseDetector):
    """兼容性检测器"""
    
    def __init__(self):
        super().__init__()
        self.name = "compatibility_detector"
    
    async def detect(self, path: Path) -> List[Issue]:
        """检测兼容性问题"""
        issues = []
        
        # 检查Python版本兼容性问题
        py_files = list(path.rglob("*.py"))
        
        if py_files:
            # 检查Python版本特性使用情况
            for py_file in py_files[:10]:  # 只检查前10个文件以节省时间
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 检查Python 3.8+特性
                    if "match case" in content and "python_version" not in content:
                        issue = Issue(
                            id=str(uuid.uuid4()),
                            type=IssueType.COMPATIBILITY,
                            severity=IssueSeverity.MEDIUM,
                            file_path=str(py_file),
                            description="使用了Python 3.10+的match case语法",
                            stack_trace="",
                            context={"feature": "match case"},
                            auto_fixable=False,
                            confidence=0.8
                        )
                        issues.append(issue)
                        
                except Exception as e:
                    logger.error(f"读取文件{py_file}时出错: {e}")
        
        return issues

class TestFailureDetector(BaseDetector):
    """测试失败检测器"""
    
    def __init__(self):
        super().__init__()
        self.name = "test_failure_detector"
    
    async def detect(self, path: Path) -> List[Issue]:
        """检测测试失败"""
        issues = []
        
        # 检查测试文件中的失败模式
        test_patterns = [
            (r"assert\s+", IssueType.TEST_FAILURE, IssueType.HIGH),
            (r"self\.(assert|fail)", IssueType.TEST_FAILURE, IssueType.HIGH),
            (r"raise\s+AssertionError", IssueType.TEST_FAILURE, IssueType.HIGH),
            (r"unittest\.case\(", IssueType.TEST_FAILURE, IssueType.HIGH)
        ]
        
        for py_file in path.rglob("*test*.py"):
            try:
                with open(py_file, "r", encoding='utf-8') as f:
                    content = f.read()
                    
                    for pattern, issue_type, severity in test_patterns:
                        if re.search(pattern, content):
                            matches = re.findall(pattern, content)
                            for match in matches:
                                line_num = content.find(match) // len(content.split('\n')) + 1
                                
                                issue = Issue(
                                    id=str(uuid.uuid4()),
                                    type=issue_type,
                                    severity=severity,
                                    file_path=str(py_file),
                                    line_number=line_num,
                                    description=f"测试失败: {match}",
                                    stack_trace="",
                                    context={"test_pattern": pattern},
                                    auto_fixable=False
                                )
                            issues.append(issue)
                            
            except Exception as e:
                logger.error(f"读取文件{py_file}时出错: {e}")
        
        return issues

# 示例使用
async def main():
    """示例使用"""
    # 初始化系统
    test_heal_system = AutoTestHealSystemV5()
    
    # 扫行扫描
    scan_result = await test_heal_system.scan_project()
    print(f"扫描结果: {scan_result}")
    
    # 自动修复
    fix_result = await test_heal_system.auto_fix_issues()
    print(f"修复结果: {fix_result}")
    
    # 运行测试
    test_result = await test_heal_system.run_tests()
    print(f"测试结果: {test_result}")
    
    # 获取健康状态
    health = test_heal_system.get_system_health()
    print(f"系统健康: {health}")
    
    # 启动持续监控
    test_heal_system.start_monitoring()
    print("持续监控已启动（按Ctrl+C停止）")
    
    try:
        while test_heal_system.monitoring:
            await asyncio.sleep(10)
    except KeyboardInterrupt:
        print("\n停止监控")
        test_heal_system.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())
