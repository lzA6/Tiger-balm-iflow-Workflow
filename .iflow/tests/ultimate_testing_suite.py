#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 终极测试套件 V3 - T-MIA 架构版
Ultimate Testing Suite V3 - T-MIA Architecture Edition

一个能够真实调用T-MIA终极工作流引擎、执行端到端复杂任务、
并进行科学的、多维度量化评估的自动化测试与对比框架。

你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import asyncio
import logging
import time
import statistics
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import psutil

# 动态添加项目根目录到sys.path
try:
    project_root = Path(__file__).resolve().parent.parent.parent
    if project_root.name != 'A项目':
         project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from iflow.core.ultimate_workflow_engine import TMIAUltimateWorkflowEngine
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.error(f"关键模块导入失败: {e}。请确保脚本在正确的项目结构下运行。")
    sys.exit(1)

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- 枚举与数据类 (借鉴自 C项目 V10) ---

class TestCategory(Enum):
    FUNCTIONALITY = "functionality"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    INTELLIGENCE = "intelligence" # 新增：智能程度

@dataclass
class TestScenario:
    name: str
    description: str
    task_description: str
    input_data: Dict[str, Any]
    test_function: Callable
    category: TestCategory
    expected_keywords: List[str] = field(default_factory=list)
    complexity: int = 5

@dataclass
class TestResult:
    scenario_name: str
    system_name: str
    success: bool
    execution_time: float
    metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)

# --- 终极测试套件 V3 ---

class UltimateTestingSuite:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.scenarios: List[TestScenario] = []
        self.systems: Dict[str, TMIAUltimateWorkflowEngine] = {}
        self.output_dir = Path(self.config.get('output_dir', 'A项目/iflow/tests/reports'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_weights = {
            'quality': 0.3, 'efficiency': 0.3, 'intelligence': 0.4
        }
        self._initialize_scenarios()

    async def initialize_systems(self):
        """初始化所有待测试的系统实例"""
        logger.info("初始化 T-MIA 终极工作流引擎用于测试...")
        
        # 在此可以初始化不同配置的引擎实例以进行对比
        # 例如: old_engine_config, new_engine_config
        engine_v3 = TMIAUltimateWorkflowEngine()
        await engine_v3.initialize()
        
        self.systems['TMIA_Engine_V3'] = engine_v3
        logger.info("已添加测试系统: TMIA_Engine_V3")

    def _initialize_scenarios(self):
        """初始化内置的基准测试场景"""
        self.scenarios = [
            TestScenario(
                name="复杂任务-代码与架构",
                description="测试引擎处理一个包含代码生成、分析和架构设计的复杂任务的能力。",
                task_description="我需要为一个新的社交媒体功能（“动态Gifs”）设计并实现后端API。请使用Python FastAPI，功能包括上传GIF，通过标签搜索GIF，并记录观看次数。数据库请使用PostgreSQL。请确保代码是生产级别的，包含错误处理、日志记录和单元测试。",
                input_data={
                    "tech_stack": ["Python", "FastAPI", "PostgreSQL"],
                    "feature": "Dynamic Gifs"
                },
                test_function=self.run_end_to_end_task,
                category=TestCategory.FUNCTIONALITY,
                expected_keywords=["FastAPI", "PostgreSQL", "def upload_gif", "def search_gifs", "CREATE TABLE gifs"],
                complexity=8
            ),
            TestScenario(
                name="性能分析与优化",
                description="测试引擎分析性能问题并提出优化方案的能力。",
                task_description="分析一个电商平台的性能瓶颈，并提出一套完整的、包含前端、后端和数据库的优化方案。",
                input_data={
                    "platform_tech_stack": ["React", "Node.js", "PostgreSQL"],
                    "current_issues": ["页面加载慢", "高并发下API响应延迟高"]
                },
                test_function=self.run_end_to_end_task,
                category=TestCategory.PERFORMANCE,
                expected_keywords=["缓存", "CDN", "数据库索引", "代码分割", "懒加载"],
                complexity=9
            ),
        ]

    async def run_comparison(self, system_names: List[str]) -> Dict[str, Any]:
        """运行指定系统间的对比测试"""
        systems_to_test = {name: self.systems[name] for name in system_names if name in self.systems}
        if not systems_to_test:
            raise ValueError("没有已注册的系统可供测试。")

        logger.info(f"开始对比测试: {', '.join(systems_to_test.keys())}")
        all_results: Dict[str, TestResult] = {}
        for scenario in self.scenarios:
            for system_name, system_instance in systems_to_test.items():
                logger.info(f"执行场景 '{scenario.name}' 于系统 '{system_name}'")
                result = await self._run_single_test(scenario, system_name, system_instance)
                all_results[f"{scenario.name}_{system_name}"] = result
        
        report = self._generate_report(list(systems_to_test.keys()), all_results)
        self._save_report(report)
        return report

    async def _run_single_test(self, scenario: TestScenario, system_name: str, system_instance: TMIAUltimateWorkflowEngine) -> TestResult:
        """运行单个端到端测试并评估结果"""
        process = psutil.Process(os.getpid())
        
        # 测试前收集系统状态
        cpu_before = process.cpu_percent(interval=None)
        mem_before = process.memory_info().rss
        start_time = time.time()
        
        try:
            # 执行测试函数
            output = await asyncio.wait_for(
                scenario.test_function(system_instance, scenario), 
                timeout=300
            )
            
            # 测试后收集系统状态
            execution_time = time.time() - start_time
            cpu_after = process.cpu_percent(interval=None)
            mem_after = process.memory_info().rss
            
            # 评估结果
            quality = self._evaluate_quality(output, scenario.expected_keywords)
            intelligence = self._evaluate_intelligence(output)
            efficiency = self._evaluate_efficiency(execution_time, cpu_after - cpu_before, mem_after - mem_before, scenario.complexity)

            overall_score = (quality * self.evaluation_weights['quality'] +
                             efficiency * self.evaluation_weights['efficiency'] +
                             intelligence * self.evaluation_weights['intelligence'])
            
            return TestResult(
                scenario_name=scenario.name, system_name=system_name, success=True,
                execution_time=execution_time,
                metrics={
                    "overall_score": overall_score, "quality": quality, "efficiency": efficiency, "intelligence": intelligence,
                    "cpu_usage": cpu_after - cpu_before, "memory_usage_mb": (mem_after - mem_before) / (1024*1024)
                },
                artifacts={'output': output}
            )
        except Exception as e:
            logger.error(f"测试 '{scenario.name}' 在 '{system_name}' 上失败: {e}", exc_info=True)
            return TestResult(
                scenario_name=scenario.name, system_name=system_name, success=False,
                execution_time=time.time() - start_time, error_message=traceback.format_exc()
            )

    async def run_end_to_end_task(self, engine: TMIAUltimateWorkflowEngine, scenario: TestScenario) -> Dict[str, Any]:
        """一个通用的端到端任务执行函数"""
        result = await engine.execute_workflow(scenario.task_description, scenario.input_data)
        if result['status'] == 'FAILED':
            raise Exception(f"工作流执行失败: {result.get('error', '未知错误')}")
        return result['result']

    # --- 评估方法 (更科学) ---
    def _evaluate_quality(self, output: Dict[str, Any], expected_keywords: List[str]) -> float:
        """评估结果的质量和相关性"""
        output_text = json.dumps(output)
        if not output_text or not expected_keywords: return 0.0
        matches = sum(1 for keyword in expected_keywords if keyword.lower() in output_text.lower())
        return matches / len(expected_keywords)

    def _evaluate_intelligence(self, output: Dict[str, Any]) -> float:
        """评估结果的智能程度 (例如，计划的深度)"""
        reasoning = output.get('reasoning', {})
        if not reasoning: return 0.1
        
        decomposition_depth = len(reasoning.get('problem_decomposition', []))
        rules_activated = len(reasoning.get('activated_rules', []))
        
        score = 0.0
        score += min(decomposition_depth / 5.0, 1.0) * 0.6 # 最多5层分解
        score += min(rules_activated / 3.0, 1.0) * 0.4   # 最多3个规则
        return score

    def _evaluate_efficiency(self, exec_time: float, cpu: float, mem_mb: float, complexity: int) -> float:
        """评估执行效率 (时间、CPU、内存)"""
        # 目标：复杂任务（10）应该在60秒内完成
        time_score = max(0.0, 1.0 - (exec_time / (complexity * 6)))
        
        # 资源分数：CPU和内存使用越低越好
        cpu_score = max(0.0, 1.0 - (cpu / 100.0))
        mem_score = max(0.0, 1.0 - (mem_mb / 512.0)) # 假设512MB是资源占用的一个阈值
        
        return time_score * 0.5 + cpu_score * 0.25 + mem_score * 0.25

    # --- 报告生成与保存 ---
    def _generate_report(self, system_names: List[str], results: Dict[str, TestResult]) -> Dict:
        # ... (与V8版本类似，但数据结构更新)
        # 此处简化
        final_scores = defaultdict(list)
        for result in results.values():
            if result.success:
                final_scores[result.system_name].append(result.metrics['overall_score'])
        
        avg_scores = {name: statistics.mean(scores) if scores else 0 for name, scores in final_scores.items()}
        winner = max(avg_scores, key=avg_scores.get) if avg_scores else "N/A"
        
        return {
            "test_date": datetime.now().isoformat(),
            "winner": winner,
            "average_scores": avg_scores,
            "detailed_results": {k: asdict(v) for k, v in results.items()}
        }

    def _save_report(self, report: Dict):
        report_path = self.output_dir / f"ultimate_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False, default=str)
        logger.info(f"终极对比测试报告已保存至: {report_path}")


# --- 主执行函数 ---
async def main():
    logger.info("--- 启动终极测试套件V3 ---")
    suite = UltimateTestingSuite()
    
    try:
        await suite.initialize_systems()
        report = await suite.run_comparison(list(suite.systems.keys()))

        print("\n--- 终极测试报告摘要 ---")
        print(f"测试日期: {report['test_date']}")
        print(f"🏆 最终获胜者: {report['winner']}")
        print("\n📊 系统平均总分:")
        for system, score in report['average_scores'].items():
            print(f"  - {system}: {score:.3f}")
    except Exception as e:
        logger.error(f"测试框架执行失败: {e}", exc_info=True)
    finally:
        logger.info("--- 关闭测试套件 ---")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())