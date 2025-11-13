#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌌 终极性能基准测试系统 (Ultimate Performance Benchmark System)
融合了V10的先进架构与V4的深度分析能力，为iflow提供全面、精准、可进化的自动化性能评估。
"""

import os
import sys
import json
import asyncio
import logging
import time
import threading
import statistics
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import psutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from abc import ABC, abstractmethod

# ==============================================================================
# 核心配置与日志
# ==============================================================================

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 数据类定义 (源自 V10 的清晰结构)
# ==============================================================================

class BenchmarkType(Enum):
    """基准测试类型"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    CONCURRENCY = "concurrency"
    STRESS = "stress"
    ENDURANCE = "endurance"

@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    name: str
    benchmark_type: BenchmarkType
    duration: timedelta
    target_function: Callable
    warmup_time: timedelta = field(default_factory=lambda: timedelta(seconds=10))
    concurrent_users: int = 1
    parameters: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Dict[str, float] = field(default_factory=dict)

@dataclass
class BenchmarkResult:
    """单个基准测试的结果"""
    config_name: str
    benchmark_type: BenchmarkType
    start_time: datetime
    end_time: datetime
    duration: float
    metrics: Dict[str, Any]
    samples: List[Dict[str, float]] = field(default_factory=list)
    success: bool = False
    error_message: Optional[str] = None
    analysis: Dict[str, Any] = field(default_factory=dict) # 新增：用于存储深度分析结果

# ==============================================================================
# 基准测试基类 (源自 V10 的优雅设计)
# ==============================================================================

class BaseBenchmark(ABC):
    """基准测试抽象基类"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.samples = []
        self.is_running = False
        
    @abstractmethod
    async def setup(self):
        """设置测试环境"""
        pass
    
    @abstractmethod
    async def execute(self) -> Dict[str, Any]:
        """执行核心测试逻辑"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """清理测试环境"""
        pass
    
    @abstractmethod
    def calculate_metrics(self) -> Dict[str, Any]:
        """根据样本计算最终指标"""
        pass
    
    async def run(self) -> BenchmarkResult:
        """运行完整的基准测试流程"""
        logger.info(f"▶️ 开始基准测试: {self.config.name} ({self.config.benchmark_type.value})")
        
        result = BenchmarkResult(
            config_name=self.config.name,
            benchmark_type=self.config.benchmark_type,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration=0.0,
            metrics={}
        )
        
        try:
            await self.setup()
            
            if self.config.warmup_time.total_seconds() > 0:
                logger.info(f"🔥 预热 {self.config.warmup_time.total_seconds()} 秒...")
                await asyncio.sleep(self.config.warmup_time.total_seconds())
            
            logger.info(f"🚀 执行测试，持续 {self.config.duration.total_seconds()} 秒...")
            start_time = time.time()
            
            self.is_running = True
            execution_summary = await self.execute()
            
            end_time = time.time()
            self.is_running = False
            
            result.end_time = datetime.now()
            result.duration = end_time - start_time
            
            logger.info("📈 计算指标...")
            calculated_metrics = self.calculate_metrics()
            
            result.metrics.update(execution_summary)
            result.metrics.update(calculated_metrics)
            result.samples = self.samples.copy()
            result.success = True
            
        except Exception as e:
            logger.error(f"❌ 基准测试失败: {self.config.name}, 错误: {e}", exc_info=True)
            result.end_time = datetime.now()
            result.error_message = str(e)
        finally:
            await self.cleanup()

        logger.info(f"⏹️ 基准测试完成: {self.config.name}")
        return result

# ==============================================================================
# 具体基准测试实现 (源自 V10)
# ==============================================================================

class ResponseTimeBenchmark(BaseBenchmark):
    """响应时间基准测试"""
    
    async def setup(self):
        logger.info(f"  设置响应时间测试环境 for {self.config.name}")

    async def execute(self) -> Dict[str, Any]:
        if not self.config.target_function:
            raise ValueError("目标函数未设置")
        
        end_time = time.time() + self.config.duration.total_seconds()
        
        while time.time() < end_time and self.is_running:
            start_sample_time = time.time()
            success = True
            error = None
            try:
                if asyncio.iscoroutinefunction(self.config.target_function):
                    await self.config.target_function(**self.config.parameters)
                else:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, lambda: self.config.target_function(**self.config.parameters))
            except Exception as e:
                success = False
                error = str(e)
            
            response_time = time.time() - start_sample_time
            self.samples.append({
                'timestamp': start_sample_time,
                'response_time': response_time,
                'success': 1 if success else 0,
            })
            if error:
                self.samples[-1]['error'] = error

        return {'total_requests': len(self.samples)}

    async def cleanup(self):
        logger.info(f"  清理响应时间测试环境 for {self.config.name}")

    def calculate_metrics(self) -> Dict[str, Any]:
        if not self.samples: return {}
        
        successful_samples = [s for s in self.samples if s['success'] == 1]
        if not successful_samples: return {'error_rate': 1.0}
        
        response_times = [s['response_time'] for s in successful_samples]
        
        return {
            'avg_response_time': statistics.mean(response_times),
            'p95_response_time': np.percentile(response_times, 95),
            'p99_response_time': np.percentile(response_times, 99),
            'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0,
            'error_rate': 1.0 - len(successful_samples) / len(self.samples)
        }

class ThroughputBenchmark(BaseBenchmark):
    """吞吐量基准测试"""

    async def setup(self):
        logger.info(f"  设置吞吐量测试环境 for {self.config.name}")

    async def execute(self) -> Dict[str, Any]:
        if not self.config.target_function:
            raise ValueError("目标函数未设置")

        async def worker():
            while self.is_running:
                start_sample_time = time.time()
                success = True
                error = None
                try:
                    if asyncio.iscoroutinefunction(self.config.target_function):
                        await self.config.target_function(**self.config.parameters)
                    else:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, lambda: self.config.target_function(**self.config.parameters))
                except Exception as e:
                    success = False
                    error = str(e)
                
                response_time = time.time() - start_sample_time
                self.samples.append({
                    'timestamp': start_sample_time,
                    'response_time': response_time,
                    'success': 1 if success else 0,
                })
                if error:
                    self.samples[-1]['error'] = error

        tasks = [asyncio.create_task(worker()) for _ in range(self.config.concurrent_users)]
        await asyncio.sleep(self.config.duration.total_seconds())
        self.is_running = False
        await asyncio.gather(*tasks, return_exceptions=True)

        return {'total_requests': len(self.samples)}

    async def cleanup(self):
        logger.info(f"  清理吞吐量测试环境 for {self.config.name}")

    def calculate_metrics(self) -> Dict[str, Any]:
        if not self.samples: return {}
        
        successful_samples = [s for s in self.samples if s['success'] == 1]
        if not successful_samples: return {'throughput_rps': 0, 'error_rate': 1.0}

        total_duration = self.config.duration.total_seconds()
        throughput_rps = len(successful_samples) / total_duration

        return {
            'throughput_rps': throughput_rps,
            'total_successful_requests': len(successful_samples),
            'error_rate': 1.0 - len(successful_samples) / len(self.samples)
        }

# ==============================================================================
# 深度分析引擎 (源自 V4 的智能核心)
# ==============================================================================

class AdvancedPatternAnalyzer:
    """高级模式分析器 (融合V4的量子分析思想)"""

    def analyze(self, result: BenchmarkResult) -> Dict[str, Any]:
        """对单个测试结果进行深度分析"""
        logger.info(f"🔬 对 {result.config_name} 进行深度分析...")
        
        analysis = {
            'patterns': self._identify_patterns(result.samples),
            'anomalies': self._detect_anomalies(result.samples),
            'optimization_opportunities': self._identify_optimization_opportunities(result.metrics),
            'quantum_insights': self._generate_quantum_insights(result.metrics)
        }
        return analysis

    def _identify_patterns(self, samples: List[Dict[str, float]]) -> List[str]:
        """识别性能模式"""
        if len(samples) < 20: return []
        
        patterns = []
        response_times = [s['response_time'] for s in samples if s['success']]
        
        # 趋势分析
        try:
            coeffs = np.polyfit(range(len(response_times)), response_times, 1)
            slope = coeffs[0]
            if abs(slope) > (statistics.mean(response_times) * 0.01): # 超过均值1%的斜率
                direction = "恶化" if slope > 0 else "改善"
                patterns.append(f"响应时间存在明显的线性{direction}趋势。")
        except Exception:
            pass # 无法计算趋势

        return patterns

    def _detect_anomalies(self, samples: List[Dict[str, float]]) -> List[str]:
        """使用统计方法检测异常点"""
        if len(samples) < 10: return []

        anomalies = []
        response_times = [s['response_time'] for s in samples if s['success']]
        mean = statistics.mean(response_times)
        stdev = statistics.stdev(response_times) if len(response_times) > 1 else 0

        if stdev == 0: return []

        # 3-sigma 规则
        upper_bound = mean + 3 * stdev
        for i, sample in enumerate(samples):
            if sample.get('success') and sample['response_time'] > upper_bound:
                anomalies.append(f"在样本 {i} 处检测到高延迟异常: {sample['response_time']:.3f}s (远超均值 {mean:.3f}s)")
        
        return anomalies

    def _identify_optimization_opportunities(self, metrics: Dict[str, Any]) -> List[str]:
        """识别优化机会"""
        opportunities = []
        if metrics.get('p99_response_time', 0) > 1.5:
            opportunities.append("P99 延迟过高，表明存在长尾请求，建议排查慢查询或GC暂停。")
        if metrics.get('error_rate', 0) > 0.05:
            opportunities.append("错误率高于5%，建议检查系统稳定性和异常处理逻辑。")
        if metrics.get('throughput_rps', float('inf')) < 50:
            opportunities.append("吞吐量较低，建议检查是否存在I/O瓶颈或锁竞争。")
        return opportunities

    def _generate_quantum_insights(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """生成模拟的量子洞察 (概念继承)"""
        potential = 0
        if metrics.get('avg_response_time', 0) > 0.8: potential += 0.3
        if metrics.get('throughput_rps', float('inf')) < 100: potential += 0.4
        
        insights = {'quantum_optimization_potential': min(potential, 1.0)}
        if potential > 0.5:
            insights['recommended_algorithms'] = ['Quantum Annealing for optimization', 'Grover\'s Algorithm for search']
            insights['estimated_speedup'] = f"{1.5 + potential:.1f}x"
        
        return insights

# ==============================================================================
# 报告生成器 (融合 V4 的可视化能力)
# ==============================================================================

class ReportGenerator:
    """生成包含摘要、图表和建议的综合报告"""

    def __init__(self, output_dir: str = "benchmark_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate(self, results: List[BenchmarkResult], session_id: str) -> str:
        """生成主报告"""
        report_path = self.output_dir / f"report_{session_id}"
        report_path.mkdir(exist_ok=True)
        
        logger.info(f"📄 生成报告到: {report_path}")

        # 生成JSON
        json_report = self._generate_json(results)
        with open(report_path / "report.json", 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, default=str)

        # 生成Markdown
        md_content = self._generate_markdown(json_report, session_id)
        with open(report_path / "report.md", 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        # 生成图表
        self._generate_visualizations(results, report_path)

        return str(report_path)

    def _generate_json(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """将结果编译为字典格式"""
        detailed_results = []
        for r in results:
            res_dict = asdict(r)
            # 移除大的样本数据以保持JSON报告简洁
            res_dict.pop('samples', None)
            detailed_results.append(res_dict)

        return {
            'summary': self._create_summary(results),
            'detailed_results': detailed_results
        }

    def _create_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """创建摘要信息"""
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.success)
        
        return {
            'total_benchmarks': total_tests,
            'successful_benchmarks': successful_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
            'total_duration_seconds': sum(r.duration for r in results)
        }

    def _generate_markdown(self, report_data: Dict[str, Any], session_id: str) -> str:
        """生成Markdown格式的报告"""
        md = f"# 性能基准测试报告 (Session: {session_id})\n\n"
        
        # 摘要
        summary = report_data['summary']
        md += "## 📊 测试摘要\n\n"
        md += f"- **总测试数**: {summary['total_benchmarks']}\n"
        md += f"- **成功率**: {summary['success_rate']:.2%}\n"
        md += f"- **总耗时**: {summary['total_duration_seconds']:.2f} 秒\n\n"
        md += "![测试成功率](success_rate.png)\n\n"

        # 详细结果
        md += "## 🔬 详细结果\n\n"
        for result in report_data['detailed_results']:
            status = "✅" if result['success'] else "❌"
            md += f"### {status} {result['config_name']} ({result['benchmark_type']})\n\n"
            md += "| 指标 | 数值 |\n|---|---|\n"
            for key, value in result['metrics'].items():
                if isinstance(value, float):
                    md += f"| {key} | {value:.3f} |\n"
                else:
                    md += f"| {key} | {value} |\n"
            md += "\n"
            
            # 深度分析
            analysis = result.get('analysis', {})
            if analysis.get('patterns'):
                md += "**识别的模式:**\n"
                for p in analysis['patterns']: md += f"- {p}\n"
            if analysis.get('anomalies'):
                md += "**检测到的异常:**\n"
                for a in analysis['anomalies']: md += f"- {a}\n"
            if analysis.get('optimization_opportunities'):
                md += "**优化机会:**\n"
                for o in analysis['optimization_opportunities']: md += f"- {o}\n"
            if analysis.get('quantum_insights', {}).get('recommended_algorithms'):
                md += "**量子洞察:**\n"
                md += f"- 优化潜力: {analysis['quantum_insights']['quantum_optimization_potential']:.1%}\n"
                md += f"- 推荐算法: {', '.join(analysis['quantum_insights']['recommended_algorithms'])}\n"
            md += "\n"

        return md

    def _generate_visualizations(self, results: List[BenchmarkResult], report_path: Path):
        """生成图表 (继承自V4)"""
        try:
            plt.style.use('seaborn-v0_8-darkgrid')

            # 1. 成功率饼图
            summary = self._create_summary(results)
            fig, ax = plt.subplots()
            ax.pie([summary['successful_benchmarks'], summary['total_benchmarks'] - summary['successful_benchmarks']],
                   labels=['成功', '失败'], autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
            ax.set_title('基准测试成功率')
            plt.savefig(report_path / "success_rate.png")
            plt.close(fig)

            # 2. 各测试响应时间对比
            fig, ax = plt.subplots(figsize=(10, 6))
            names = [r.config_name for r in results if r.success and 'avg_response_time' in r.metrics]
            times = [r.metrics['avg_response_time'] for r in results if r.success and 'avg_response_time' in r.metrics]
            if names:
                ax.barh(names, times, color='skyblue')
                ax.set_xlabel('平均响应时间 (秒)')
                ax.set_title('各测试平均响应时间')
                plt.tight_layout()
                plt.savefig(report_path / "response_times.png")
            plt.close(fig)

            logger.info("🎨 可视化图表已生成。")
        except Exception as e:
            logger.warning(f"⚠️ 生成可视化图表失败: {e}")

# ==============================================================================
# 主控制系统
# ==============================================================================

class UltimatePerformanceBenchmark:
    """终极性能基准测试主系统"""
    
    def __init__(self, report_dir: str = "benchmark_reports"):
        self.configs: List[BenchmarkConfig] = []
        self.report_generator = ReportGenerator(output_dir=report_dir)
        self.analyzer = AdvancedPatternAnalyzer()

    def register(self, config: BenchmarkConfig):
        """注册一个基准测试配置"""
        self.configs.append(config)
        logger.info(f"✅ 已注册基准测试: {config.name}")

    async def run_all(self) -> str:
        """运行所有已注册的基准测试并生成报告"""
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info(f"🏁 开始执行所有基准测试 (Session: {session_id})")
        
        all_results = []
        for config in self.configs:
            benchmark_class = None
            if config.benchmark_type == BenchmarkType.RESPONSE_TIME:
                benchmark_class = ResponseTimeBenchmark
            elif config.benchmark_type == BenchmarkType.THROUGHPUT:
                benchmark_class = ThroughputBenchmark
            # ...可以扩展其他类型
            
            if benchmark_class:
                benchmark_instance = benchmark_class(config)
                result = await benchmark_instance.run()
                
                # 进行深度分析
                if result.success:
                    result.analysis = self.analyzer.analyze(result)

                all_results.append(result)
            else:
                logger.warning(f"未找到基准测试类型 {config.benchmark_type} 的实现。")

        report_path = self.report_generator.generate(all_results, session_id)
        logger.info(f"🎉 所有基准测试完成。报告位于: {report_path}")
        return report_path

# ==============================================================================
# 示例用法
# ==============================================================================

async def example_task_fast():
    """模拟一个快速任务"""
    await asyncio.sleep(np.random.uniform(0.05, 0.1))

async def example_task_slow():
    """模拟一个较慢且可能失败的任务"""
    await asyncio.sleep(np.random.uniform(0.2, 0.5))
    if np.random.rand() < 0.1:
        raise ConnectionError("模拟网络连接失败")

async def main():
    """主执行函数"""
    logger.info("🚀 初始化终极性能基准测试系统...")
    
    benchmark_system = UltimatePerformanceBenchmark()

    # 注册测试
    benchmark_system.register(BenchmarkConfig(
        name="API_Fast_Response_Time",
        benchmark_type=BenchmarkType.RESPONSE_TIME,
        duration=timedelta(seconds=15),
        target_function=example_task_fast
    ))
    
    benchmark_system.register(BenchmarkConfig(
        name="API_Slow_Throughput",
        benchmark_type=BenchmarkType.THROUGHPUT,
        duration=timedelta(seconds=20),
        concurrent_users=10,
        target_function=example_task_slow
    ))

    # 运行所有测试
    report_directory = await benchmark_system.run_all()
    
    print("\n" + "="*60)
    print(f"✅ 测试完成！请查看报告目录: {report_directory}")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())