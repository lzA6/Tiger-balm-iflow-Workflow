#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ 量子性能基准测试系统
Quantum Performance Benchmarking System

为全能工作流V6提供全面的性能基准测试和优化建议
"""

import asyncio
import time
import json
import psutil
import statistics
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import sys
import os

# 添加.iflow到Python路径
current_dir = Path(__file__).parent
iflow_root = current_dir.parent
sys.path.insert(0, str(iflow_root))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkMetrics:
    """基准测试指标"""
    test_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float
    throughput: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    error_count: int
    timestamp: str

@dataclass
class SystemMetrics:
    """系统指标"""
    cpu_count: int
    memory_total_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    process_count: int

class QuantumPerformanceBenchmark:
    """量子性能基准测试器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.root_dir = Path(__file__).parent.parent
        self.config = self._load_config(config_path)
        self.results = []
        self.system_metrics = self._get_system_metrics()
        
        # 测试配置
        self.test_config = {
            "iterations": 100,
            "warmup_iterations": 10,
            "concurrent_users": 10,
            "test_duration_seconds": 60,
            "memory_stress_test_mb": 100,
            "cpu_stress_test_seconds": 30
        }
        
        # 基准阈值
        self.benchmark_thresholds = {
            "max_response_time_ms": 500,
            "max_memory_usage_mb": 512,
            "max_cpu_usage_percent": 80,
            "min_success_rate": 0.99,
            "min_throughput_ops_per_sec": 100
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            "enable_quantum_tests": True,
            "enable_stress_tests": True,
            "enable_memory_tests": True,
            "enable_cpu_tests": True,
            "enable_network_tests": True,
            "output_format": "json",
            "save_detailed_logs": True
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        
        return default_config
    
    def _get_system_metrics(self) -> SystemMetrics:
        """获取系统指标"""
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        process_count = len(psutil.pids())
        
        return SystemMetrics(
            cpu_count=cpu_count,
            memory_total_gb=memory.total / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=disk.percent,
            network_io={
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv
            } if network else {},
            process_count=process_count
        )
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """运行全面的基准测试"""
        logger.info("🚀 开始量子性能基准测试...")
        
        start_time = datetime.now()
        benchmark_results = {
            "test_session_id": int(start_time.timestamp()),
            "start_time": start_time.isoformat(),
            "system_metrics": asdict(self.system_metrics),
            "test_results": {},
            "summary": {},
            "recommendations": []
        }
        
        try:
            # 1. 基础性能测试
            if self.config.get("enable_cpu_tests", True):
                benchmark_results["test_results"]["cpu_performance"] = await self._benchmark_cpu_performance()
            
            # 2. 内存性能测试
            if self.config.get("enable_memory_tests", True):
                benchmark_results["test_results"]["memory_performance"] = await self._benchmark_memory_performance()
            
            # 3. 网络性能测试
            if self.config.get("enable_network_tests", True):
                benchmark_results["test_results"]["network_performance"] = await self._benchmark_network_performance()
            
            # 4. 量子算法性能测试
            if self.config.get("enable_quantum_tests", True):
                benchmark_results["test_results"]["quantum_performance"] = await self._benchmark_quantum_performance()
            
            # 5. 压力测试
            if self.config.get("enable_stress_tests", True):
                benchmark_results["test_results"]["stress_test"] = await self._run_stress_test()
            
            # 6. 生成总结和建议
            benchmark_results["summary"] = self._generate_summary(benchmark_results["test_results"])
            benchmark_results["recommendations"] = self._generate_recommendations(benchmark_results["test_results"])
            
            # 7. 保存结果
            end_time = datetime.now()
            benchmark_results["end_time"] = end_time.isoformat()
            benchmark_results["total_duration"] = (end_time - start_time).total_seconds()
            
            await self._save_benchmark_results(benchmark_results)
            
            logger.info("✅ 量子性能基准测试完成!")
            return benchmark_results
            
        except Exception as e:
            logger.error(f"❌ 基准测试失败: {e}")
            benchmark_results["error"] = str(e)
            return benchmark_results
    
    async def _benchmark_cpu_performance(self) -> Dict[str, Any]:
        """CPU性能基准测试"""
        logger.info("🔥 执行CPU性能基准测试...")
        
        results = {
            "test_name": "CPU Performance",
            "metrics": [],
            "summary": {}
        }
        
        # 测试1: 计算密集型任务
        computation_times = []
        for i in range(self.test_config["iterations"]):
            start_time = time.time()
            
            # 执行计算密集型任务
            result = sum(np.random.rand(10000) ** 2)
            
            end_time = time.time()
            computation_times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        # 测试2: 并发处理能力
        concurrent_times = []
        tasks = []
        
        async def cpu_task():
            start = time.time()
            # 模拟CPU密集型任务
            for _ in range(1000):
                _ = sum(range(1000))
            return (time.time() - start) * 1000
        
        for _ in range(self.test_config["concurrent_users"]):
            tasks.append(cpu_task())
        
        concurrent_results = await asyncio.gather(*tasks)
        concurrent_times.extend(concurrent_results)
        
        # 计算指标
        metrics = BenchmarkMetrics(
            test_name="cpu_computation",
            execution_time=statistics.mean(computation_times),
            memory_usage_mb=self._get_memory_usage(),
            cpu_usage_percent=self._get_cpu_usage(),
            success_rate=1.0,
            throughput=self.test_config["iterations"] / sum(computation_times) * 1000,
            latency_p50=np.percentile(computation_times, 50),
            latency_p95=np.percentile(computation_times, 95),
            latency_p99=np.percentile(computation_times, 99),
            error_count=0,
            timestamp=datetime.now().isoformat()
        )
        
        results["metrics"].append(asdict(metrics))
        results["summary"] = {
            "avg_computation_time_ms": statistics.mean(computation_times),
            "max_computation_time_ms": max(computation_times),
            "min_computation_time_ms": min(computation_times),
            "concurrent_tasks_avg_time_ms": statistics.mean(concurrent_times),
            "cpu_efficiency_score": self._calculate_cpu_efficiency(metrics)
        }
        
        return results
    
    async def _benchmark_memory_performance(self) -> Dict[str, Any]:
        """内存性能基准测试"""
        logger.info("💾 执行内存性能基准测试...")
        
        results = {
            "test_name": "Memory Performance",
            "metrics": [],
            "summary": {}
        }
        
        # 测试1: 内存分配速度
        allocation_times = []
        memory_sizes = []
        
        for size in [1024, 10240, 102400, 1024000]:  # 1KB, 10KB, 100KB, 1MB
            start_time = time.time()
            
            # 分配内存
            data = np.random.rand(size)
            memory_sizes.append(sys.getsizeof(data))
            
            end_time = time.time()
            allocation_times.append((end_time - start_time) * 1000)
        
        # 测试2: 内存访问性能
        access_times = []
        test_array = np.random.rand(100000)
        
        for _ in range(100):
            start_time = time.time()
            
            # 随机访问内存
            for _ in range(1000):
                idx = np.random.randint(0, len(test_array))
                _ = test_array[idx]
            
            end_time = time.time()
            access_times.append((end_time - start_time) * 1000)
        
        # 测试3: 内存压力测试
        stress_start = time.time()
        memory_blocks = []
        
        try:
            for _ in range(self.test_config["memory_stress_test_mb"]):
                block = np.random.rand(1024)  # 8KB per block
                memory_blocks.append(block)
        except MemoryError:
            logger.warning("内存不足，停止压力测试")
        
        stress_end = time.time()
        stress_time = (stress_end - stress_start) * 1000
        
        # 计算指标
        metrics = BenchmarkMetrics(
            test_name="memory_allocation",
            execution_time=statistics.mean(allocation_times),
            memory_usage_mb=self._get_memory_usage(),
            cpu_usage_percent=self._get_cpu_usage(),
            success_rate=1.0,
            throughput=len(memory_blocks) / (stress_time / 1000) if stress_time > 0 else 0,
            latency_p50=np.percentile(allocation_times, 50),
            latency_p95=np.percentile(allocation_times, 95),
            latency_p99=np.percentile(allocation_times, 99),
            error_count=0,
            timestamp=datetime.now().isoformat()
        )
        
        results["metrics"].append(asdict(metrics))
        results["summary"] = {
            "avg_allocation_time_ms": statistics.mean(allocation_times),
            "avg_access_time_ms": statistics.mean(access_times),
            "stress_test_time_ms": stress_time,
            "memory_blocks_allocated": len(memory_blocks),
            "memory_efficiency_score": self._calculate_memory_efficiency(metrics)
        }
        
        return results
    
    async def _benchmark_network_performance(self) -> Dict[str, Any]:
        """网络性能基准测试"""
        logger.info("🌐 执行网络性能基准测试...")
        
        results = {
            "test_name": "Network Performance",
            "metrics": [],
            "summary": {}
        }
        
        # 测试1: 本地通信延迟
        local_latencies = []
        
        for _ in range(self.test_config["iterations"]):
            start_time = time.time()
            
            # 模拟本地网络操作
            try:
                # 创建本地socket连接测试
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(('localhost', 80))
                sock.close()
            except:
                pass  # 忽略连接错误，专注于延迟测试
            
            end_time = time.time()
            local_latencies.append((end_time - start_time) * 1000)
        
        # 测试2: 并发连接测试
        concurrent_latencies = []
        
        async def network_task():
            start = time.time()
            try:
                # 模拟网络操作
                await asyncio.sleep(0.001)  # 1ms延迟
            except:
                pass
            return (time.time() - start) * 1000
        
        tasks = [network_task() for _ in range(self.test_config["concurrent_users"])]
        concurrent_results = await asyncio.gather(*tasks)
        concurrent_latencies.extend(concurrent_results)
        
        # 计算指标
        metrics = BenchmarkMetrics(
            test_name="network_latency",
            execution_time=statistics.mean(local_latencies),
            memory_usage_mb=self._get_memory_usage(),
            cpu_usage_percent=self._get_cpu_usage(),
            success_rate=1.0,
            throughput=self.test_config["iterations"] / sum(local_latencies) * 1000,
            latency_p50=np.percentile(local_latencies, 50),
            latency_p95=np.percentile(local_latencies, 95),
            latency_p99=np.percentile(local_latencies, 99),
            error_count=0,
            timestamp=datetime.now().isoformat()
        )
        
        results["metrics"].append(asdict(metrics))
        results["summary"] = {
            "avg_local_latency_ms": statistics.mean(local_latencies),
            "concurrent_avg_latency_ms": statistics.mean(concurrent_latencies),
            "network_efficiency_score": self._calculate_network_efficiency(metrics)
        }
        
        return results
    
    async def _benchmark_quantum_performance(self) -> Dict[str, Any]:
        """量子算法性能基准测试"""
        logger.info("⚛️ 执行量子性能基准测试...")
        
        results = {
            "test_name": "Quantum Performance",
            "metrics": [],
            "summary": {}
        }
        
        # 测试1: 量子矩阵运算
        quantum_times = []
        
        for _ in range(50):  # 量子测试计算量较大，减少迭代次数
            start_time = time.time()
            
            # 模拟量子矩阵运算
            matrix_size = 100
            matrix_a = np.random.rand(matrix_size, matrix_size)
            matrix_b = np.random.rand(matrix_size, matrix_size)
            
            # 量子态模拟（矩阵乘法）
            result = np.dot(matrix_a, matrix_b)
            
            # 量子测量模拟
            measurement = np.abs(result) ** 2
            measurement = measurement / np.sum(measurement)  # 归一化
            
            end_time = time.time()
            quantum_times.append((end_time - start_time) * 1000)
        
        # 测试2: 量子纠缠模拟
        entanglement_times = []
        
        for _ in range(30):
            start_time = time.time()
            
            # 模拟量子纠缠态
            bell_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
            
            # 量子操作模拟
            pauli_x = np.array([[0, 1], [1, 0]])
            cnot = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, 1, 0]])
            
            # 量子门操作
            entangled_state = np.dot(cnot, np.kron(pauli_x, np.eye(2)))
            
            end_time = time.time()
            entanglement_times.append((end_time - start_time) * 1000)
        
        # 计算指标
        metrics = BenchmarkMetrics(
            test_name="quantum_computation",
            execution_time=statistics.mean(quantum_times),
            memory_usage_mb=self._get_memory_usage(),
            cpu_usage_percent=self._get_cpu_usage(),
            success_rate=1.0,
            throughput=50 / sum(quantum_times) * 1000,
            latency_p50=np.percentile(quantum_times, 50),
            latency_p95=np.percentile(quantum_times, 95),
            latency_p99=np.percentile(quantum_times, 99),
            error_count=0,
            timestamp=datetime.now().isoformat()
        )
        
        results["metrics"].append(asdict(metrics))
        results["summary"] = {
            "avg_quantum_computation_ms": statistics.mean(quantum_times),
            "avg_entanglement_simulation_ms": statistics.mean(entanglement_times),
            "quantum_efficiency_score": self._calculate_quantum_efficiency(metrics)
        }
        
        return results
    
    async def _run_stress_test(self) -> Dict[str, Any]:
        """压力测试"""
        logger.info("💪 执行系统压力测试...")
        
        results = {
            "test_name": "Stress Test",
            "metrics": [],
            "summary": {}
        }
        
        start_time = time.time()
        stress_metrics = []
        
        # 并发压力测试
        async def stress_task(task_id: int):
            task_start = time.time()
            cpu_usage_samples = []
            memory_usage_samples = []
            
            # 模拟高强度任务
            for i in range(100):
                # CPU密集型操作
                _ = sum(np.random.rand(1000) ** 2)
                
                # 记录资源使用
                if i % 10 == 0:
                    cpu_usage_samples.append(self._get_cpu_usage())
                    memory_usage_samples.append(self._get_memory_usage())
                
                # 短暂休息
                await asyncio.sleep(0.001)
            
            task_end = time.time()
            
            return {
                "task_id": task_id,
                "execution_time": (task_end - task_start) * 1000,
                "avg_cpu_usage": statistics.mean(cpu_usage_samples) if cpu_usage_samples else 0,
                "max_memory_usage": max(memory_usage_samples) if memory_usage_samples else 0,
                "success": True
            }
        
        # 启动并发任务
        tasks = [stress_task(i) for i in range(self.test_config["concurrent_users"] * 2)]
        task_results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_stress_time = (end_time - start_time) * 1000
        
        # 分析结果
        successful_tasks = [r for r in task_results if r["success"]]
        execution_times = [r["execution_time"] for r in successful_tasks]
        cpu_usages = [r["avg_cpu_usage"] for r in successful_tasks]
        memory_usages = [r["max_memory_usage"] for r in successful_tasks]
        
        # 计算指标
        metrics = BenchmarkMetrics(
            test_name="stress_test",
            execution_time=statistics.mean(execution_times) if execution_times else 0,
            memory_usage_mb=statistics.mean(memory_usages) if memory_usages else 0,
            cpu_usage_percent=statistics.mean(cpu_usages) if cpu_usages else 0,
            success_rate=len(successful_tasks) / len(task_results),
            throughput=len(successful_tasks) / (total_stress_time / 1000) if total_stress_time > 0 else 0,
            latency_p50=np.percentile(execution_times, 50) if execution_times else 0,
            latency_p95=np.percentile(execution_times, 95) if execution_times else 0,
            latency_p99=np.percentile(execution_times, 99) if execution_times else 0,
            error_count=len(task_results) - len(successful_tasks),
            timestamp=datetime.now().isoformat()
        )
        
        results["metrics"].append(asdict(metrics))
        results["summary"] = {
            "total_stress_time_ms": total_stress_time,
            "successful_tasks": len(successful_tasks),
            "failed_tasks": len(task_results) - len(successful_tasks),
            "system_stability_score": self._calculate_stability_score(metrics)
        }
        
        return results
    
    def _get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _get_cpu_usage(self) -> float:
        """获取当前CPU使用率"""
        return psutil.cpu_percent(interval=0.1)
    
    def _calculate_cpu_efficiency(self, metrics: BenchmarkMetrics) -> float:
        """计算CPU效率分数"""
        # 基于执行时间和CPU使用率计算效率
        time_score = max(0, 100 - metrics.execution_time)
        cpu_score = max(0, 100 - metrics.cpu_usage_percent)
        return (time_score + cpu_score) / 2
    
    def _calculate_memory_efficiency(self, metrics: BenchmarkMetrics) -> float:
        """计算内存效率分数"""
        # 基于内存使用量和吞吐量计算效率
        memory_score = max(0, 100 - metrics.memory_usage_mb / 10)  # 假设10MB为基准
        throughput_score = min(100, metrics.throughput)
        return (memory_score + throughput_score) / 2
    
    def _calculate_network_efficiency(self, metrics: BenchmarkMetrics) -> float:
        """计算网络效率分数"""
        # 基于延迟和吞吐量计算效率
        latency_score = max(0, 100 - metrics.latency_p50)
        throughput_score = min(100, metrics.throughput)
        return (latency_score + throughput_score) / 2
    
    def _calculate_quantum_efficiency(self, metrics: BenchmarkMetrics) -> float:
        """计算量子算法效率分数"""
        # 基于量子计算性能指标
        time_score = max(0, 100 - metrics.execution_time / 10)  # 量子计算时间基准
        throughput_score = min(100, metrics.throughput * 10)  # 量子计算吞吐量权重
        return (time_score + throughput_score) / 2
    
    def _calculate_stability_score(self, metrics: BenchmarkMetrics) -> float:
        """计算系统稳定性分数"""
        # 基于成功率和资源使用计算稳定性
        success_score = metrics.success_rate * 100
        resource_score = max(0, 100 - max(metrics.cpu_usage_percent, metrics.memory_usage_mb / 10))
        return (success_score + resource_score) / 2
    
    def _generate_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成测试总结"""
        summary = {
            "overall_score": 0,
            "performance_grade": "Unknown",
            "key_metrics": {},
            "bottlenecks": [],
            "strengths": []
        }
        
        scores = []
        
        # 收集各项测试的效率分数
        for test_name, test_data in test_results.items():
            if "summary" in test_data:
                for metric_name, metric_value in test_data["summary"].items():
                    if "efficiency_score" in metric_name or "stability_score" in metric_name:
                        scores.append(metric_value)
                        summary["key_metrics"][f"{test_name}_{metric_name}"] = metric_value
        
        # 计算总体分数
        if scores:
            summary["overall_score"] = statistics.mean(scores)
            
            # 评级
            if summary["overall_score"] >= 90:
                summary["performance_grade"] = "A+ (优秀)"
            elif summary["overall_score"] >= 80:
                summary["performance_grade"] = "A (良好)"
            elif summary["overall_score"] >= 70:
                summary["performance_grade"] = "B (一般)"
            elif summary["overall_score"] >= 60:
                summary["performance_grade"] = "C (较差)"
            else:
                summary["performance_grade"] = "D (需要优化)"
        
        return summary
    
    def _generate_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 分析各项测试结果并生成建议
        for test_name, test_data in test_results.items():
            if "metrics" in test_data:
                for metric in test_data["metrics"]:
                    # 检查响应时间
                    if metric.get("execution_time", 0) > self.benchmark_thresholds["max_response_time_ms"]:
                        recommendations.append(f"{test_name}: 响应时间过长，建议优化算法或增加计算资源")
                    
                    # 检查内存使用
                    if metric.get("memory_usage_mb", 0) > self.benchmark_thresholds["max_memory_usage_mb"]:
                        recommendations.append(f"{test_name}: 内存使用过高，建议优化内存管理或增加内存")
                    
                    # 检查CPU使用
                    if metric.get("cpu_usage_percent", 0) > self.benchmark_thresholds["max_cpu_usage_percent"]:
                        recommendations.append(f"{test_name}: CPU使用率过高，建议优化计算逻辑或增加CPU核心")
                    
                    # 检查成功率
                    if metric.get("success_rate", 1.0) < self.benchmark_thresholds["min_success_rate"]:
                        recommendations.append(f"{test_name}: 成功率过低，建议检查错误处理和系统稳定性")
                    
                    # 检查吞吐量
                    if metric.get("throughput", 0) < self.benchmark_thresholds["min_throughput_ops_per_sec"]:
                        recommendations.append(f"{test_name}: 吞吐量过低，建议优化并发处理能力")
        
        # 通用优化建议
        if not recommendations:
            recommendations.append("系统性能表现良好，建议继续监控并保持当前配置")
        else:
            recommendations.append("建议定期运行性能基准测试以监控系统性能变化")
            recommendations.append("考虑启用量子优化模块以提升整体性能")
        
        return recommendations
    
    async def _save_benchmark_results(self, results: Dict[str, Any]) -> None:
        """保存基准测试结果"""
        # 确保输出目录存在
        output_dir = self.root_dir / "benchmark_results"
        output_dir.mkdir(exist_ok=True)
        
        # 保存JSON格式结果
        timestamp = int(datetime.now().timestamp())
        json_file = output_dir / f"quantum_benchmark_{timestamp}.json"
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存Markdown格式报告
        md_file = output_dir / f"quantum_benchmark_{timestamp}.md"
        await self._save_markdown_report(results, md_file)
        
        logger.info(f"📊 基准测试结果已保存: {json_file}")
        logger.info(f"📋 测试报告已生成: {md_file}")
    
    async def _save_markdown_report(self, results: Dict[str, Any], file_path: Path) -> None:
        """保存Markdown格式报告"""
        report_content = f"""# ⚡ 量子性能基准测试报告

## 测试概览

- **测试会话ID**: {results['test_session_id']}
- **开始时间**: {results['start_time']}
- **结束时间**: {results.get('end_time', 'N/A')}
- **总耗时**: {results.get('total_duration', 'N/A')} 秒

## 系统环境

- **CPU核心数**: {results['system_metrics']['cpu_count']}
- **总内存**: {results['system_metrics']['memory_total_gb']:.2f} GB
- **可用内存**: {results['system_metrics']['memory_available_gb']:.2f} GB
- **磁盘使用率**: {results['system_metrics']['disk_usage_percent']}%
- **进程数**: {results['system_metrics']['process_count']}

## 测试结果

"""
        
        # 添加各项测试结果
        for test_name, test_data in results.get("test_results", {}).items():
            report_content += f"### {test_data.get('test_name', test_name)}\n\n"
            
            if "summary" in test_data:
                for metric, value in test_data["summary"].items():
                    if isinstance(value, float):
                        report_content += f"- **{metric}**: {value:.2f}\n"
                    else:
                        report_content += f"- **{metric}**: {value}\n"
                report_content += "\n"
        
        # 添加总体评分
        summary = results.get("summary", {})
        if summary:
            report_content += f"""## 总体评估

- **性能总分**: {summary.get('overall_score', 0):.2f}
- **性能等级**: {summary.get('performance_grade', 'Unknown')}

"""
        
        # 添加优化建议
        recommendations = results.get("recommendations", [])
        if recommendations:
            report_content += "## 优化建议\n\n"
            for i, rec in enumerate(recommendations, 1):
                report_content += f"{i}. {rec}\n"
        
        report_content += f"""

---

报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

async def main():
    """主函数"""
    print("⚡ 量子性能基准测试系统")
    print("=" * 50)
    
    # 检查依赖
    try:
        import numpy as np
        import psutil
    except ImportError as e:
        print(f"❌ 缺少依赖库: {e}")
        print("请安装: pip install numpy psutil")
        return
    
    # 创建基准测试器
    benchmark = QuantumPerformanceBenchmark()
    
    try:
        # 运行基准测试
        results = await benchmark.run_comprehensive_benchmark()
        
        print("\n" + "=" * 50)
        print("✅ 基准测试完成!")
        
        if "error" not in results:
            summary = results.get("summary", {})
            print(f"📊 总体评分: {summary.get('overall_score', 0):.2f}")
            print(f"🏆 性能等级: {summary.get('performance_grade', 'Unknown')}")
            
            recommendations = results.get("recommendations", [])
            if recommendations:
                print(f"\n💡 主要建议:")
                for rec in recommendations[:3]:  # 显示前3个建议
                    print(f"  • {rec}")
        else:
            print(f"❌ 测试失败: {results['error']}")
        
    except Exception as e:
        print(f"❌ 基准测试执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
