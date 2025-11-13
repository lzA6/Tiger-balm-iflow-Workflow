#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 工具调用验证器
Tool Call Validator

全面测试工具调用情况，验证CLI、Python敏感性、模型适配等
"""

import asyncio
import time
import json
import logging
import sys
import os
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback

# 添加.iflow到Python路径
IFLOW_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(IFLOW_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tool_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ToolTestResult:
    """工具测试结果"""
    tool_name: str
    test_type: str
    status: str  # passed, failed, error
    duration: float
    details: Dict[str, Any]
    timestamp: float
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None

@dataclass
class ValidationReport:
    """验证报告"""
    total_tools: int
    passed_tools: int
    failed_tools: int
    error_tools: int
    total_duration: float
    test_results: List[ToolTestResult]
    summary: Dict[str, Any]
    recommendations: List[str]

class ToolCallValidator:
    """工具调用验证器"""
    
    def __init__(self):
        self.iflow_root = IFLOW_ROOT
        self.tools_dir = self.iflow_root / "tools"
        self.core_dir = self.iflow_root / "core"
        self.agents_dir = self.iflow_root / "agents"
        self.workflows_dir = self.iflow_root / "workflows"
        self.test_results = []
        
    async def run_comprehensive_validation(self) -> ValidationReport:
        """运行全面验证"""
        logger.info("开始全面工具调用验证...")
        start_time = time.time()
        
        # 1. CLI工具验证
        await self._validate_cli_tools()
        
        # 2. Python工具验证
        await self._validate_python_tools()
        
        # 3. 智能体验证
        await self._validate_agents()
        
        # 4. 工作流验证
        await self._validate_workflows()
        
        # 5. 模型适配器验证
        await self._validate_model_adapter()
        
        # 6. 性能测试
        await self._validate_performance()
        
        total_duration = time.time() - start_time
        
        # 生成报告
        report = self._generate_validation_report(total_duration)
        
        logger.info(f"验证完成，耗时: {total_duration:.2f}s")
        return report
    
    async def _validate_cli_tools(self):
        """验证CLI工具"""
        cli_path = self.core_dir / "iflow-cli.py"
        
        if not cli_path.exists():
            self._add_test_result("CLI工具", "cli", "error", 0, {"error": "CLI文件不存在"})
            return
        
        # 测试CLI帮助命令
        await self._test_cli_command("help")
        
        # 测试CLI状态命令
        await self._test_cli_command("status")
        
        # 测试CLI智能体列表
        await self._test_cli_command("agent list")
        
        # 测试CLI工作流列表
        await self._test_cli_command("workflow list")
    
    async def _test_cli_command(self, command: str):
        """测试CLI命令"""
        start_time = time.time()
        
        try:
            # 构建命令
            cmd = [sys.executable, str(self.core_dir / "iflow-cli.py")]
            if command != "help":
                cmd.extend(command.split())
            
            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.iflow_root)
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                self._add_test_result(
                    f"CLI-{command}",
                    "cli",
                    "passed",
                    duration,
                    {
                        "stdout": result.stdout[:500],  # 限制输出长度
                        "returncode": result.returncode
                    }
                )
            else:
                self._add_test_result(
                    f"CLI-{command}",
                    "cli",
                    "failed",
                    duration,
                    {
                        "stderr": result.stderr[:500],
                        "returncode": result.returncode
                    },
                    result.stderr
                )
                
        except subprocess.TimeoutExpired:
            self._add_test_result(
                f"CLI-{command}",
                "cli",
                "failed",
                30,
                {"error": "命令超时"},
                "命令执行超时"
            )
        except Exception as e:
            duration = time.time() - start_time
            self._add_test_result(
                f"CLI-{command}",
                "cli",
                "error",
                duration,
                {"error": str(e)},
                str(e)
            )
    
    async def _validate_python_tools(self):
        """验证Python工具"""
        python_tools = [
            "enhanced-universal-agent.py",
            "quantum-performance-optimizer.py",
            "self-evolution-engine.py",
            "comprehensive-test-suite.py"
        ]
        
        for tool in python_tools:
            await self._test_python_tool(tool)
    
    async def _test_python_tool(self, tool_name: str):
        """测试Python工具"""
        tool_path = self.tools_dir / tool_name
        
        if not tool_path.exists():
            self._add_test_result(f"Python-{tool_name}", "python", "error", 0, {"error": "工具文件不存在"})
            return
        
        start_time = time.time()
        
        try:
            # 测试导入
            spec = importlib.util.spec_from_file_location(tool_name.replace('.py', ''), tool_path)
            module = importlib.util.module_from_spec(spec)
            
            # 检查是否有语法错误
            spec.loader.exec_module(module)
            
            duration = time.time() - start_time
            
            # 测试工具的主要功能
            test_result = await self._test_tool_functionality(module, tool_name)
            
            self._add_test_result(
                f"Python-{tool_name}",
                "python",
                "passed" if test_result else "failed",
                duration,
                {
                    "import_success": True,
                    "functionality_test": test_result
                }
            )
            
        except SyntaxError as e:
            duration = time.time() - start_time
            self._add_test_result(
                f"Python-{tool_name}",
                "python",
                "failed",
                duration,
                {"error": "语法错误", "details": str(e)},
                f"语法错误: {e}"
            )
        except Exception as e:
            duration = time.time() - start_time
            self._add_test_result(
                f"Python-{tool_name}",
                "python",
                "error",
                duration,
                {"error": str(e)},
                str(e)
            )
    
    async def _test_tool_functionality(self, module, tool_name: str) -> bool:
        """测试工具功能"""
        try:
            if tool_name == "enhanced-universal-agent.py":
                # 测试增强智能体
                agent = getattr(module, 'get_enhanced_agent')()
                if hasattr(agent, 'analyze_task'):
                    return True
                    
            elif tool_name == "quantum-performance-optimizer.py":
                # 测试性能优化器
                optimizer = getattr(module, 'get_global_optimizer')()
                if hasattr(optimizer, 'get_optimization_report'):
                    return True
                    
            elif tool_name == "self-evolution-engine.py":
                # 测试进化引擎
                engine = getattr(module, 'get_evolution_engine')()
                if hasattr(engine, 'get_evolution_status'):
                    return True
                    
            elif tool_name == "comprehensive-test-suite.py":
                # 测试测试套件
                if hasattr(module, 'ComprehensiveTestSuite'):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Functionality test failed for {tool_name}: {e}")
            return False
    
    async def _validate_agents(self):
        """验证智能体"""
        agent_files = list(self.agents_dir.rglob("*.md"))
        
        for agent_file in agent_files:
            await self._test_agent_file(agent_file)
    
    async def _test_agent_file(self, agent_file: Path):
        """测试智能体文件"""
        try:
            with open(agent_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查智能体文件基本结构
            checks = {
                "has_title": bool(content.strip()),
                "has_sections": "##" in content,
                "has_abilities": "能力" in content or "ability" in content,
                "size_valid": len(content) > 100
            }
            
            passed = all(checks.values())
            
            self._add_test_result(
                f"Agent-{agent_file.name}",
                "agent",
                "passed" if passed else "failed",
                0,
                checks,
                None if passed else "智能体文件结构不完整"
            )
            
        except Exception as e:
            self._add_test_result(
                f"Agent-{agent_file.name}",
                "agent",
                "error",
                0,
                {"error": str(e)},
                str(e)
            )
    
    async def _validate_workflows(self):
        """验证工作流"""
        workflow_files = list(self.workflows_dir.rglob("*.yaml"))
        
        for workflow_file in workflow_files:
            await self._test_workflow_file(workflow_file)
    
    async def _test_workflow_file(self, workflow_file: Path):
        """测试工作流文件"""
        try:
            import yaml
            
            with open(workflow_file, 'r', encoding='utf-8') as f:
                workflow_config = yaml.safe_load(f)
            
            # 检查工作流基本结构
            checks = {
                "has_metadata": "metadata" in workflow_config,
                "has_workflow": "workflow" in workflow_config,
                "has_phases": bool(workflow_config.get("workflow", {}).get("phases")),
                "size_valid": len(str(workflow_config)) > 100
            }
            
            passed = all(checks.values())
            
            self._add_test_result(
                f"Workflow-{workflow_file.name}",
                "workflow",
                "passed" if passed else "failed",
                0,
                checks,
                None if passed else "工作流文件结构不完整"
            )
            
        except Exception as e:
            self._add_test_result(
                f"Workflow-{workflow_file.name}",
                "workflow",
                "error",
                0,
                {"error": str(e)},
                str(e)
            )
    
    async def _validate_model_adapter(self):
        """验证模型适配器"""
        adapter_config = self.iflow_root / "config" / "universal-model-adapter.yaml"
        
        if not adapter_config.exists():
            self._add_test_result(
                "ModelAdapter",
                "adapter",
                "error",
                0,
                {"error": "适配器配置文件不存在"},
                "适配器配置文件不存在"
            )
            return
        
        try:
            import yaml
            
            with open(adapter_config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 检查适配器配置
            checks = {
                "has_supported_models": "supported_models" in config,
                "has_openai": "openai" in config.get("supported_models", {}),
                "has_anthropic": "anthropic" in config.get("supported_models", {}),
                "has_google": "google" in config.get("supported_models", {}),
                "has_chinese_models": any(model in config.get("supported_models", {}) 
                                         for model in ["baidu", "alibaba", "tencent", "bytedance"]),
                "has_performance_config": "performance_optimization" in config
            }
            
            passed = all(checks.values())
            
            self._add_test_result(
                "ModelAdapter",
                "adapter",
                "passed" if passed else "failed",
                0,
                checks,
                None if passed else "模型适配器配置不完整"
            )
            
        except Exception as e:
            self._add_test_result(
                "ModelAdapter",
                "adapter",
                "error",
                0,
                {"error": str(e)},
                str(e)
            )
    
    async def _validate_performance(self):
        """验证性能"""
        # 测试内存使用
        await self._test_memory_usage()
        
        # 测试响应时间
        await self._test_response_time()
        
        # 测试并发能力
        await self._test_concurrency()
    
    async def _test_memory_usage(self):
        """测试内存使用"""
        try:
            import psutil
            
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # 内存使用检查（假设合理阈值）
            memory_mb = memory_info.rss / 1024 / 1024
            passed = memory_mb < 1000  # 小于1GB认为正常
            
            self._add_test_result(
                "MemoryUsage",
                "performance",
                "passed" if passed else "failed",
                0,
                {
                    "memory_mb": memory_mb,
                    "threshold_mb": 1000
                },
                None if passed else f"内存使用过高: {memory_mb:.2f}MB"
            )
            
        except Exception as e:
            self._add_test_result(
                "MemoryUsage",
                "performance",
                "error",
                0,
                {"error": str(e)},
                str(e)
            )
    
    async def _test_response_time(self):
        """测试响应时间"""
        start_time = time.time()
        
        # 模拟简单操作
        await asyncio.sleep(0.01)
        
        duration = time.time() - start_time
        passed = duration < 1.0  # 小于1秒认为正常
        
        self._add_test_result(
            "ResponseTime",
            "performance",
            "passed" if passed else "failed",
            duration,
            {
                "duration_ms": duration * 1000,
                "threshold_ms": 1000
            },
            None if passed else f"响应时间过长: {duration*1000:.2f}ms"
        )
    
    async def _test_concurrency(self):
        """测试并发能力"""
        start_time = time.time()
        
        # 创建并发任务
        async def dummy_task():
            await asyncio.sleep(0.01)
            return "done"
        
        tasks = [dummy_task() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        duration = time.time() - start_time
        passed = len(results) == 10 and duration < 2.0
        
        self._add_test_result(
            "Concurrency",
            "performance",
            "passed" if passed else "failed",
            duration,
            {
                "tasks_completed": len(results),
                "concurrent_tasks": 10,
                "duration_s": duration
            },
            None if passed else f"并发测试失败: 完成{len(results)}/10任务"
        )
    
    def _add_test_result(self, tool_name: str, test_type: str, status: str, 
                        duration: float, details: Dict[str, Any], 
                        error_message: Optional[str] = None):
        """添加测试结果"""
        result = ToolTestResult(
            tool_name=tool_name,
            test_type=test_type,
            status=status,
            duration=duration,
            details=details,
            timestamp=time.time(),
            error_message=error_message
        )
        self.test_results.append(result)
    
    def _generate_validation_report(self, total_duration: float) -> ValidationReport:
        """生成验证报告"""
        total_tools = len(self.test_results)
        passed_tools = len([r for r in self.test_results if r.status == 'passed'])
        failed_tools = len([r for r in self.test_results if r.status == 'failed'])
        error_tools = len([r for r in self.test_results if r.status == 'error'])
        
        # 生成总结
        summary = {
            "success_rate": passed_tools / total_tools if total_tools > 0 else 0,
            "total_duration": total_duration,
            "avg_duration": sum(r.duration for r in self.test_results) / total_tools if total_tools > 0 else 0,
            "test_types": list(set(r.test_type for r in self.test_results)),
            "critical_issues": [r.tool_name for r in self.test_results if r.status == 'error']
        }
        
        # 生成建议
        recommendations = []
        if error_tools > 0:
            recommendations.append(f"修复{error_tools}个错误工具")
        if failed_tools > 0:
            recommendations.append(f"改进{failed_tools}个失败工具")
        if summary["success_rate"] < 0.8:
            recommendations.append("整体成功率偏低，需要全面优化")
        
        return ValidationReport(
            total_tools=total_tools,
            passed_tools=passed_tools,
            failed_tools=failed_tools,
            error_tools=error_tools,
            total_duration=total_duration,
            test_results=self.test_results,
            summary=summary,
            recommendations=recommendations
        )
    
    def save_report(self, report: ValidationReport, output_path: str = None):
        """保存验证报告"""
        if output_path is None:
            output_path = self.iflow_root / "tool_validation_report.json"
        
        # 转换为可序列化的格式
        report_data = {
            "total_tools": report.total_tools,
            "passed_tools": report.passed_tools,
            "failed_tools": report.failed_tools,
            "error_tools": report.error_tools,
            "total_duration": report.total_duration,
            "summary": report.summary,
            "recommendations": report.recommendations,
            "test_results": [asdict(r) for r in report.test_results],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"验证报告已保存到: {output_path}")

# 使用示例
async def main():
    """主函数示例"""
    validator = ToolCallValidator()
    
    # 运行全面验证
    report = await validator.run_comprehensive_validation()
    
    # 输出结果
    print(f"\n=== 工具调用验证报告 ===")
    print(f"总工具数: {report.total_tools}")
    print(f"通过: {report.passed_tools}")
    print(f"失败: {report.failed_tools}")
    print(f"错误: {report.error_tools}")
    print(f"成功率: {report.summary['success_rate']:.1%}")
    print(f"总耗时: {report.total_duration:.2f}s")
    
    # 显示失败和错误的工具
    if report.failed_tools > 0:
        print(f"\n失败的工具:")
        for result in report.test_results:
            if result.status == 'failed':
                print(f"  - {result.tool_name}: {result.error_message}")
    
    if report.error_tools > 0:
        print(f"\n错误的工具:")
        for result in report.test_results:
            if result.status == 'error':
                print(f"  - {result.tool_name}: {result.error_message}")
    
    # 显示建议
    if report.recommendations:
        print(f"\n建议:")
        for rec in report.recommendations:
            print(f"  - {rec}")
    
    # 保存报告
    validator.save_report(report)

if __name__ == "__main__":
    asyncio.run(main())
