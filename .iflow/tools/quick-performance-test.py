#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ 快速性能测试系统
Quick Performance Test System

为全能工作流V6提供快速性能验证
"""

import asyncio
import time
import json
import psutil
import statistics
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickPerformanceTest:
    """快速性能测试器"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent
        self.results = {}
        
    async def run_quick_test(self) -> Dict[str, Any]:
        """运行快速性能测试"""
        logger.info("🚀 开始快速性能测试...")
        
        start_time = datetime.now()
        
        try:
            # 基础系统指标
            system_info = self._get_system_info()
            
            # CPU性能测试
            cpu_result = await self._test_cpu_performance()
            
            # 内存性能测试
            memory_result = await self._test_memory_performance()
            
            # 文件I/O测试
            io_result = await self._test_io_performance()
            
            # 并发测试
            concurrent_result = await self._test_concurrent_performance()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # 汇总结果
            results = {
                "test_session_id": int(start_time.timestamp()),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "system_info": system_info,
                "test_results": {
                    "cpu": cpu_result,
                    "memory": memory_result,
                    "io": io_result,
                    "concurrent": concurrent_result
                },
                "overall_score": self._calculate_overall_score(cpu_result, memory_result, io_result, concurrent_result),
                "performance_grade": self._get_performance_grade(self._calculate_overall_score(cpu_result, memory_result, io_result, concurrent_result)),
                "recommendations": self._generate_recommendations(cpu_result, memory_result, io_result, concurrent_result)
            }
            
            # 保存结果
            await self._save_results(results)
            
            logger.info("✅ 快速性能测试完成!")
            return results
            
        except Exception as e:
            logger.error(f"❌ 测试失败: {e}")
            return {"error": str(e)}
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_count": cpu_count,
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "memory_percent": memory.percent,
            "disk_total_gb": disk.total / (1024**3),
            "disk_free_gb": disk.free / (1024**3),
            "disk_percent": disk.percent
        }
    
    async def _test_cpu_performance(self) -> Dict[str, Any]:
        """测试CPU性能"""
        logger.info("🔥 测试CPU性能...")
        
        # 简单计算测试
        start_time = time.time()
        
        # 执行计算密集型任务
        total = 0
        for i in range(100000):
            total += i * i
        
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # 毫秒
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        return {
            "computation_time_ms": execution_time,
            "cpu_usage_percent": cpu_percent,
            "operations_per_second": 100000 / (execution_time / 1000) if execution_time > 0 else 0,
            "score": max(0, 100 - execution_time / 10)  # 简单评分
        }
    
    async def _test_memory_performance(self) -> Dict[str, Any]:
        """测试内存性能"""
        logger.info("💾 测试内存性能...")
        
        # 内存分配测试
        start_time = time.time()
        
        # 分配内存
        data = []
        for i in range(1000):
            data.append([0] * 100)  # 分配10000个整数
        
        end_time = time.time()
        allocation_time = (end_time - start_time) * 1000
        
        # 内存使用
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # 内存访问测试
        start_time = time.time()
        for row in data:
            _ = sum(row)
        end_time = time.time()
        access_time = (end_time - start_time) * 1000
        
        return {
            "allocation_time_ms": allocation_time,
            "access_time_ms": access_time,
            "memory_usage_mb": memory_mb,
            "elements_allocated": len(data) * 100,
            "score": max(0, 100 - memory_mb / 10)  # 简单评分
        }
    
    async def _test_io_performance(self) -> Dict[str, Any]:
        """测试文件I/O性能"""
        logger.info("💾 测试文件I/O性能...")
        
        # 创建测试文件
        test_file = self.root_dir / "temp_test_file.txt"
        test_data = "测试数据 " * 10000  # 约80KB数据
        
        # 写入测试
        start_time = time.time()
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_data)
        end_time = time.time()
        write_time = (end_time - start_time) * 1000
        
        # 读取测试
        start_time = time.time()
        with open(test_file, 'r', encoding='utf-8') as f:
            _ = f.read()
        end_time = time.time()
        read_time = (end_time - start_time) * 1000
        
        # 清理测试文件
        try:
            test_file.unlink()
        except:
            pass
        
        return {
            "write_time_ms": write_time,
            "read_time_ms": read_time,
            "data_size_bytes": len(test_data.encode('utf-8')),
            "write_throughput_mb_per_sec": (len(test_data.encode('utf-8')) / 1024 / 1024) / (write_time / 1000) if write_time > 0 else 0,
            "read_throughput_mb_per_sec": (len(test_data.encode('utf-8')) / 1024 / 1024) / (read_time / 1000) if read_time > 0 else 0,
            "score": max(0, 100 - (write_time + read_time) / 10)  # 简单评分
        }
    
    async def _test_concurrent_performance(self) -> Dict[str, Any]:
        """测试并发性能"""
        logger.info("⚡ 测试并发性能...")
        
        async def simple_task(task_id: int):
            """简单任务"""
            start = time.time()
            # 模拟一些工作
            total = 0
            for i in range(1000):
                total += i
            await asyncio.sleep(0.001)  # 1ms模拟I/O
            return {
                "task_id": task_id,
                "execution_time_ms": (time.time() - start) * 1000,
                "result": total
            }
        
        # 并发执行任务
        start_time = time.time()
        tasks = [simple_task(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = (end_time - start_time) * 1000
        execution_times = [r["execution_time_ms"] for r in results]
        
        return {
            "concurrent_tasks": len(tasks),
            "total_time_ms": total_time,
            "avg_task_time_ms": statistics.mean(execution_times),
            "max_task_time_ms": max(execution_times),
            "min_task_time_ms": min(execution_times),
            "throughput_tasks_per_sec": len(tasks) / (total_time / 1000) if total_time > 0 else 0,
            "score": max(0, 100 - total_time / 10)  # 简单评分
        }
    
    def _calculate_overall_score(self, cpu_result: Dict, memory_result: Dict, 
                                io_result: Dict, concurrent_result: Dict) -> float:
        """计算总体评分"""
        scores = [
            cpu_result.get("score", 0),
            memory_result.get("score", 0),
            io_result.get("score", 0),
            concurrent_result.get("score", 0)
        ]
        return statistics.mean(scores) if scores else 0
    
    def _get_performance_grade(self, overall_score: float) -> str:
        """获取性能等级"""
        if overall_score >= 90:
            return "A+ (优秀)"
        elif overall_score >= 80:
            return "A (良好)"
        elif overall_score >= 70:
            return "B (一般)"
        elif overall_score >= 60:
            return "C (较差)"
        else:
            return "D (需要优化)"
    
    def _generate_recommendations(self, cpu_result: Dict, memory_result: Dict, 
                                 io_result: Dict, concurrent_result: Dict) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # CPU建议
        if cpu_result.get("computation_time_ms", 0) > 100:
            recommendations.append("CPU计算时间较长，建议优化算法或增加CPU资源")
        
        # 内存建议
        if memory_result.get("memory_usage_mb", 0) > 100:
            recommendations.append("内存使用较高，建议优化内存管理")
        
        # I/O建议
        if io_result.get("write_time_ms", 0) > 50 or io_result.get("read_time_ms", 0) > 50:
            recommendations.append("文件I/O性能较慢，建议使用更快的存储设备")
        
        # 并发建议
        if concurrent_result.get("throughput_tasks_per_sec", 0) < 100:
            recommendations.append("并发处理能力有待提升，建议优化异步处理逻辑")
        
        if not recommendations:
            recommendations.append("系统性能表现良好，继续保持当前配置")
        
        return recommendations
    
    async def _save_results(self, results: Dict[str, Any]) -> None:
        """保存测试结果"""
        # 确保输出目录存在
        output_dir = self.root_dir / "benchmark_results"
        output_dir.mkdir(exist_ok=True)
        
        # 保存JSON格式结果
        timestamp = int(datetime.now().timestamp())
        json_file = output_dir / f"quick_performance_{timestamp}.json"
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 测试结果已保存: {json_file}")
        
        # 保存到results属性供后续使用
        self.results = results.get("test_results", {})

async def main():
    """主函数"""
    print("⚡ 快速性能测试系统")
    print("=" * 50)
    
    # 创建测试器
    test = QuickPerformanceTest()
    
    try:
        # 运行测试
        results = await test.run_quick_test()
        
        print("\n" + "=" * 50)
        print("✅ 快速性能测试完成!")
        
        if "error" not in results:
            print(f"📊 总体评分: {results.get('overall_score', 0):.2f}")
            print(f"🏆 性能等级: {results.get('performance_grade', 'Unknown')}")
            
            # 显示各项测试结果
            test_results = results.get("test_results", {})
            for test_name, test_data in test_results.items():
                print(f"\n🔍 {test_name.upper()}测试:")
                for key, value in test_data.items():
                    if key != "score":
                        if isinstance(value, float):
                            print(f"  {key}: {value:.2f}")
                        else:
                            print(f"  {key}: {value}")
            
            # 显示建议
            recommendations = results.get("recommendations", [])
            if recommendations:
                print(f"\n💡 优化建议:")
                for rec in recommendations:
                    print(f"  • {rec}")
        else:
            print(f"❌ 测试失败: {results['error']}")
        
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())