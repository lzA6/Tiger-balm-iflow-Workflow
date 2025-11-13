#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全能工作流V5 - 自动测试和验证系统
Automated Testing and Validation System

作者: Universal AI Team
版本: 5.0.0
日期: 2025-11-12

特性:
- 全自动测试执行
- 多维度验证
- 智能测试生成
- 性能基准测试
- 回归测试
- 测试报告生成
"""

import os
import sys
import json
import time
import asyncio
import logging
import unittest
import pytest
import subprocess
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
import yaml
import coverage
import bandit
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestType(Enum):
    """测试类型"""
    UNIT = "unit"
    INTEGRATION = "integration"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    SECURITY = "security"
    REGRESSION = "regression"
    ACCEPTANCE = "acceptance"

class TestStatus(Enum):
    """测试状态"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

class SeverityLevel(Enum):
    """严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class TestCase:
    """测试用例"""
    id: str
    name: str
    description: str
    test_type: TestType
    module_path: str
    test_function: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_result: Any = None
    timeout: int = 60
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'test_type': self.test_type.value,
            'module_path': self.module_path,
            'test_function': self.test_function,
            'parameters': self.parameters,
            'expected_result': self.expected_result,
            'timeout': self.timeout,
            'tags': self.tags,
            'dependencies': self.dependencies
        }

@dataclass
class TestResult:
    """测试结果"""
    test_id: str
    status: TestStatus
    execution_time: float
    start_time: datetime
    end_time: datetime
    output: str
    error_message: Optional[str] = None
    assertions: int = 0
    passed_assertions: int = 0
    coverage: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'test_id': self.test_id,
            'status': self.status.value,
            'execution_time': self.execution_time,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'output': self.output,
            'error_message': self.error_message,
            'assertions': self.assertions,
            'passed_assertions': self.passed_assertions,
            'coverage': self.coverage,
            'performance_metrics': self.performance_metrics
        }

@dataclass
class TestSuite:
    """测试套件"""
    id: str
    name: str
    description: str
    test_cases: List[TestCase]
    setup_actions: List[str] = field(default_factory=list)
    teardown_actions: List[str] = field(default_factory=list)
    parallel_execution: bool = True
    max_workers: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'test_cases': [tc.to_dict() for tc in self.test_cases],
            'setup_actions': self.setup_actions,
            'teardown_actions': self.teardown_actions,
            'parallel_execution': self.parallel_execution,
            'max_workers': self.max_workers
        }

class AutoTestSystem:
    """自动测试系统"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            '.iflow', 'config', 'quality-assurance.yaml'
        )
        
        # 测试存储
        self.test_suites: Dict[str, TestSuite] = {}
        self.test_results: Dict[str, List[TestResult]] = defaultdict(list)
        
        # 测试执行器
        self.test_executor = TestExecutor()
        
        # 测试生成器
        self.test_generator = TestGenerator()
        
        # 覆盖率工具
        self.coverage_tool = CoverageTool()
        
        # 性能测试器
        self.performance_tester = PerformanceTester()
        
        # 安全测试器
        self.security_tester = SecurityTester()
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        
        # 配置
        self.config = self._load_config()
        
        # 初始化
        self._initialize_system()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {
            'auto_generate_tests': True,
            'run_coverage': True,
            'run_performance_tests': True,
            'run_security_tests': True,
            'test_timeout': 300,
            'parallel_execution': True,
            'max_workers': 4
        }
    
    def _initialize_system(self):
        """初始化系统"""
        # 创建测试目录
        test_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tests'
        )
        os.makedirs(test_dir, exist_ok=True)
        
        # 发现现有测试
        self._discover_existing_tests()
        
        # 自动生成测试
        if self.config.get('auto_generate_tests', True):
            self._auto_generate_tests()
        
        logger.info(f"自动测试系统初始化完成，发现 {len(self.test_suites)} 个测试套件")
    
    def _discover_existing_tests(self):
        """发现现有测试"""
        test_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tests'
        )
        
        if not os.path.exists(test_dir):
            return
        
        # 遍历测试文件
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.startswith('test_') and file.endswith('.py'):
                    self._parse_test_file(os.path.join(root, file))
    
    def _parse_test_file(self, file_path: str):
        """解析测试文件"""
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取测试函数
            import ast
            tree = ast.parse(content)
            
            test_cases = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    test_case = TestCase(
                        id=f"{os.path.basename(file_path)}_{node.name}",
                        name=node.name,
                        description=f"测试函数: {node.name}",
                        test_type=TestType.UNIT,
                        module_path=file_path,
                        test_function=node.name
                    )
                    test_cases.append(test_case)
            
            if test_cases:
                suite = TestSuite(
                    id=os.path.basename(file_path),
                    name=f"测试套件: {os.path.basename(file_path)}",
                    description=f"从 {file_path} 自动生成的测试套件",
                    test_cases=test_cases
                )
                self.test_suites[suite.id] = suite
                
        except Exception as e:
            logger.error(f"解析测试文件失败 {file_path}: {e}")
    
    def _auto_generate_tests(self):
        """自动生成测试"""
        # 扫描源代码
        source_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tools'
        )
        
        if not os.path.exists(source_dir):
            return
        
        # 为每个模块生成测试
        for file in os.listdir(source_dir):
            if file.endswith('.py') and not file.startswith('__'):
                module_path = os.path.join(source_dir, file)
                generated_tests = self.test_generator.generate_for_module(module_path)
                
                if generated_tests:
                    suite_id = f"generated_{file[:-3]}"
                    suite = TestSuite(
                        id=suite_id,
                        name=f"自动生成测试: {file}",
                        description=f"为 {file} 自动生成的测试",
                        test_cases=generated_tests
                    )
                    self.test_suites[suite_id] = suite
    
    async def run_test_suite(self, suite_id: str) -> Dict[str, Any]:
        """运行测试套件"""
        if suite_id not in self.test_suites:
            return {'error': f'测试套件不存在: {suite_id}'}
        
        suite = self.test_suites[suite_id]
        
        # 执行setup
        await self._execute_setup_actions(suite)
        
        try:
            # 运行测试
            if suite.parallel_execution:
                results = await self._run_tests_parallel(suite)
            else:
                results = await self._run_tests_sequential(suite)
            
            # 保存结果
            self.test_results[suite_id] = results
            
            # 执行teardown
            await self._execute_teardown_actions(suite)
            
            # 生成报告
            report = self._generate_test_report(suite_id, results)
            
            return report
            
        except Exception as e:
            logger.error(f"运行测试套件失败 {suite_id}: {e}")
            return {'error': str(e)}
    
    async def _execute_setup_actions(self, suite: TestSuite):
        """执行setup操作"""
        for action in suite.setup_actions:
            try:
                await self._execute_action(action)
            except Exception as e:
                logger.error(f"Setup操作失败 {action}: {e}")
    
    async def _execute_teardown_actions(self, suite: TestSuite):
        """执行teardown操作"""
        for action in suite.teardown_actions:
            try:
                await self._execute_action(action)
            except Exception as e:
                logger.error(f"Teardown操作失败 {action}: {e}")
    
    async def _execute_action(self, action: str):
        """执行操作"""
        # 实现具体操作逻辑
        pass
    
    async def _run_tests_parallel(self, suite: TestSuite) -> List[TestResult]:
        """并行运行测试"""
        results = []
        
        # 创建任务
        tasks = []
        for test_case in suite.test_cases:
            task = asyncio.create_task(
                self._run_single_test(test_case)
            )
            tasks.append(task)
        
        # 等待完成
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        for task_result in completed_tasks:
            if isinstance(task_result, Exception):
                logger.error(f"测试执行异常: {task_result}")
            else:
                results.append(task_result)
        
        return results
    
    async def _run_tests_sequential(self, suite: TestSuite) -> List[TestResult]:
        """顺序运行测试"""
        results = []
        
        for test_case in suite.test_cases:
            result = await self._run_single_test(test_case)
            results.append(result)
            
            # 检查是否应该继续
            if result.status == TestStatus.ERROR and test_case.test_type == TestType.SYSTEM:
                logger.warning(f"系统测试失败，停止执行: {test_case.id}")
                break
        
        return results
    
    async def _run_single_test(self, test_case: TestCase) -> TestResult:
        """运行单个测试"""
        start_time = datetime.now()
        
        try:
            # 执行测试
            result = await self.test_executor.execute(test_case)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # 创建测试结果
            test_result = TestResult(
                test_id=test_case.id,
                status=result.get('status', TestStatus.PASSED),
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                output=result.get('output', ''),
                error_message=result.get('error'),
                assertions=result.get('assertions', 0),
                passed_assertions=result.get('passed_assertions', 0)
            )
            
            # 收集覆盖率
            if self.config.get('run_coverage', True):
                coverage_data = self.coverage_tool.get_coverage(test_case.module_path)
                test_result.coverage = coverage_data
            
            # 收集性能指标
            if test_case.test_type == TestType.PERFORMANCE:
                perf_metrics = self.performance_tester.get_metrics(test_case.id)
                test_result.performance_metrics = perf_metrics
            
            return test_result
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return TestResult(
                test_id=test_case.id,
                status=TestStatus.ERROR,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                output='',
                error_message=str(e)
            )
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        all_results = {}
        total_summary = {
            'total_suites': len(self.test_suites),
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': 0,
            'total_time': 0
        }
        
        for suite_id in self.test_suites:
            suite_result = await self.run_test_suite(suite_id)
            all_results[suite_id] = suite_result
            
            # 更新汇总
            if 'summary' in suite_result:
                summary = suite_result['summary']
                total_summary['total_tests'] += summary.get('total_tests', 0)
                total_summary['passed'] += summary.get('passed', 0)
                total_summary['failed'] += summary.get('failed', 0)
                total_summary['skipped'] += summary.get('skipped', 0)
                total_summary['errors'] += summary.get('errors', 0)
                total_summary['total_time'] += summary.get('total_time', 0)
        
        # 计算通过率
        if total_summary['total_tests'] > 0:
            total_summary['pass_rate'] = total_summary['passed'] / total_summary['total_tests']
        else:
            total_summary['pass_rate'] = 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': total_summary,
            'suite_results': all_results
        }
    
    def _generate_test_report(self, suite_id: str, results: List[TestResult]) -> Dict[str, Any]:
        """生成测试报告"""
        # 统计信息
        total_tests = len(results)
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)
        errors = sum(1 for r in results if r.status == TestStatus.ERROR)
        
        total_time = sum(r.execution_time for r in results)
        
        # 覆盖率汇总
        coverage_summary = {}
        if results and results[0].coverage:
            coverage_keys = results[0].coverage.keys()
            for key in coverage_keys:
                values = [r.coverage.get(key, 0) for r in results if key in r.coverage]
                if values:
                    coverage_summary[key] = sum(values) / len(values)
        
        # 性能指标汇总
        performance_summary = {}
        perf_results = [r for r in results if r.performance_metrics]
        if perf_results:
            for key in perf_results[0].performance_metrics.keys():
                values = [r.performance_metrics.get(key, 0) for r in perf_results if key in r.performance_metrics]
                if values:
                    performance_summary[key] = {
                        'avg': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values)
                    }
        
        report = {
            'suite_id': suite_id,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed': passed,
                'failed': failed,
                'skipped': skipped,
                'errors': errors,
                'pass_rate': passed / total_tests if total_tests > 0 else 0,
                'total_time': total_time
            },
            'coverage': coverage_summary,
            'performance': performance_summary,
            'test_results': [r.to_dict() for r in results]
        }
        
        # 保存报告
        self._save_test_report(suite_id, report)
        
        return report
    
    def _save_test_report(self, suite_id: str, report: Dict[str, Any]):
        """保存测试报告"""
        reports_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'test_reports'
        )
        os.makedirs(reports_dir, exist_ok=True)
        
        report_file = os.path.join(
            reports_dir,
            f"{suite_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    
    def get_test_history(self, suite_id: str = None, days: int = 7) -> Dict[str, Any]:
        """获取测试历史"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # 读取历史报告
        reports_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'test_reports'
        )
        
        history = []
        if os.path.exists(reports_dir):
            for file in os.listdir(reports_dir):
                if file.endswith('.json'):
                    file_path = os.path.join(reports_dir, file)
                    
                    # 检查日期
                    file_date = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_date > cutoff_date:
                        
                        # 检查套件ID
                        if suite_id is None or suite_id in file:
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    report = json.load(f)
                                    history.append(report)
                            except Exception as e:
                                logger.error(f"读取测试报告失败 {file}: {e}")
        
        # 按时间排序
        history.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return {
            'period_days': days,
            'suite_id': suite_id,
            'reports': history,
            'total_reports': len(history)
        }

class TestExecutor:
    """测试执行器"""
    
    async def execute(self, test_case: TestCase) -> Dict[str, Any]:
        """执行测试用例"""
        try:
            # 根据测试类型选择执行方式
            if test_case.test_type == TestType.UNIT:
                return await self._execute_unit_test(test_case)
            elif test_case.test_type == TestType.INTEGRATION:
                return await self._execute_integration_test(test_case)
            elif test_case.test_type == TestType.PERFORMANCE:
                return await self._execute_performance_test(test_case)
            elif test_case.test_type == TestType.SECURITY:
                return await self._execute_security_test(test_case)
            else:
                return await self._execute_generic_test(test_case)
                
        except Exception as e:
            return {
                'status': TestStatus.ERROR,
                'error': str(e)
            }
    
    async def _execute_unit_test(self, test_case: TestCase) -> Dict[str, Any]:
        """执行单元测试"""
        # 使用pytest执行
        cmd = [
            'python', '-m', 'pytest',
            test_case.module_path,
            '-k', test_case.test_function,
            '-v'
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        output = stdout.decode() + stderr.decode()
        
        return {
            'status': TestStatus.PASSED if process.returncode == 0 else TestStatus.FAILED,
            'output': output,
            'assertions': output.count('assert'),
            'passed_assertions': output.count('PASSED')
        }
    
    async def _execute_integration_test(self, test_case: TestCase) -> Dict[str, Any]:
        """执行集成测试"""
        # 实现集成测试逻辑
        return await self._execute_generic_test(test_case)
    
    async def _execute_performance_test(self, test_case: TestCase) -> Dict[str, Any]:
        """执行性能测试"""
        # 实现性能测试逻辑
        return await self._execute_generic_test(test_case)
    
    async def _execute_security_test(self, test_case: TestCase) -> Dict[str, Any]:
        """执行安全测试"""
        # 使用bandit进行安全扫描
        cmd = ['bandit', '-r', os.path.dirname(test_case.module_path)]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        output = stdout.decode() + stderr.decode()
        
        return {
            'status': TestStatus.PASSED if process.returncode == 0 else TestStatus.FAILED,
            'output': output
        }
    
    async def _execute_generic_test(self, test_case: TestCase) -> Dict[str, Any]:
        """执行通用测试"""
        # 动态导入并执行测试函数
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(
            "test_module",
            test_case.module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 获取测试函数
        test_func = getattr(module, test_case.test_function)
        
        # 执行测试
        if asyncio.iscoroutinefunction(test_func):
            result = await test_func(**test_case.parameters)
        else:
            result = test_func(**test_case.parameters)
        
        return {
            'status': TestStatus.PASSED if result else TestStatus.FAILED,
            'output': str(result)
        }

class TestGenerator:
    """测试生成器"""
    
    def generate_for_module(self, module_path: str) -> List[TestCase]:
        """为模块生成测试"""
        test_cases = []
        
        try:
            # 读取模块内容
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析AST
            import ast
            tree = ast.parse(content)
            
            # 为每个函数生成测试
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    test_case = TestCase(
                        id=f"test_{node.name}",
                        name=f"test_{node.name}",
                        description=f"自动生成的测试: {node.name}",
                        test_type=TestType.UNIT,
                        module_path=module_path,
                        test_function=node.name
                    )
                    test_cases.append(test_case)
                    
        except Exception as e:
            logger.error(f"生成测试失败 {module_path}: {e}")
        
        return test_cases

class CoverageTool:
    """覆盖率工具"""
    
    def __init__(self):
        self.cov = coverage.Coverage()
    
    def start(self):
        """开始收集覆盖率"""
        self.cov.start()
    
    def stop(self):
        """停止收集覆盖率"""
        self.cov.stop()
    
    def get_coverage(self, module_path: str) -> Dict[str, float]:
        """获取覆盖率数据"""
        try:
            # 分析覆盖率
            self.cov.save()
            
            # 获取覆盖率报告
            report = coverage.CoverageData()
            
            return {
                'line_coverage': 0.0,  # 简化实现
                'branch_coverage': 0.0,
                'function_coverage': 0.0
            }
            
        except Exception:
            return {}

class PerformanceTester:
    """性能测试器"""
    
    def __init__(self):
        self.metrics = {}
    
    def get_metrics(self, test_id: str) -> Dict[str, float]:
        """获取性能指标"""
        return self.metrics.get(test_id, {
            'response_time': 0.0,
            'throughput': 0.0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0
        })

class SecurityTester:
    """安全测试器"""
    
    def __init__(self):
        self.vulnerabilities = []
    
    def scan(self, module_path: str) -> List[Dict[str, Any]]:
        """扫描安全漏洞"""
        # 使用bandit扫描
        try:
            result = subprocess.run(
                ['bandit', '-r', module_path, '-f', 'json'],
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                return data.get('results', [])
                
        except Exception:
            pass
        
        return []

# 全局测试系统实例
test_system = AutoTestSystem()

# 便捷函数
async def run_all_tests() -> Dict[str, Any]:
    """运行所有测试"""
    return await test_system.run_all_tests()

async def run_test_suite(suite_id: str) -> Dict[str, Any]:
    """运行测试套件"""
    return await test_system.run_test_suite(suite_id)

def get_test_history(suite_id: str = None, days: int = 7) -> Dict[str, Any]:
    """获取测试历史"""
    return test_system.get_test_history(suite_id, days)

if __name__ == "__main__":
    # 测试代码
    async def test_system():
        print("自动测试系统测试")
        print("=" * 50)
        
        # 运行所有测试
        results = await run_all_tests()
        
        print(f"测试结果:")
        print(f"总测试套件: {results['summary']['total_suites']}")
        print(f"总测试数: {results['summary']['total_tests']}")
        print(f"通过: {results['summary']['passed']}")
        print(f"失败: {results['summary']['failed']}")
        print(f"通过率: {results['summary']['pass_rate']:.2%}")
        
        # 获取测试历史
        history = get_test_history(days=1)
        print(f"\n历史报告数: {history['total_reports']}")
    
    asyncio.run(test_system())
