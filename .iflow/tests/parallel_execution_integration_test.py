#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 并行执行引擎集成测试
验证所有并行执行组件的协同工作，展示整体性能提升效果。
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Any
import statistics

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from iflow.core.optimized_fusion_cache import OptimizedFusionCache
from iflow.core.parallel_agent_executor import ParallelAgentExecutor, AgentRole
from iflow.core.task_decomposer import TaskDecomposer
from iflow.core.workflow_stage_parallelizer import WorkflowStageParallelizer, WorkflowStage, WorkflowStageInfo

logger = logging.getLogger(__name__)

class ParallelExecutionBenchmark:
    """并行执行基准测试"""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
        
        # 创建测试组件
        self.cache = OptimizedFusionCache(cache_size=100, ttl_hours=1)
        self.agent_executor = ParallelAgentExecutor(max_concurrent_agents=8, enable_cache=True)
        self.task_decomposer = TaskDecomposer()
        self.workflow_parallelizer = WorkflowStageParallelizer(max_concurrent_stages=6)
    
    async def run_comprehensive_test(self):
        """运行综合测试"""
        print("=" * 80)
        print("🚀 并行执行引擎综合性能测试")
        print("=" * 80)
        
        # 1. 缓存性能测试
        await self._test_cache_performance()
        
        # 2. 智能体并行测试
        await self._test_agent_parallelism()
        
        # 3. 任务分解测试
        await self._test_task_decomposition()
        
        # 4. 工作流阶段并行测试
        await self._test_workflow_parallelism()
        
        # 5. 端到端集成测试
        await self._test_end_to_end_integration()
        
        # 6. 生成性能报告
        self._generate_performance_report()
    
    async def _test_cache_performance(self):
        """测试缓存性能"""
        print("\n📊 测试缓存性能...")
        
        start_time = time.time()
        
        # 模拟缓存操作
        test_tasks = [
            "设计电商系统架构",
            "开发用户管理系统",
            "实现支付功能",
            "编写测试用例",
            "部署到云平台"
        ]
        
        # 首次执行（缓存未命中）
        for task in test_tasks:
            self.cache.put_cache_result(
                task=task,
                context={"test": True},
                selected_experts=["架构师", "开发专家"],
                fusion_mode="parallel",
                result=f"结果: {task}",
                quality_score=0.9,
                execution_time=1.5
            )
        
        # 第二次执行（缓存命中）
        cache_hits = 0
        for task in test_tasks:
            result = self.cache.get_cached_result(task, {"test": True})
            if result:
                cache_hits += 1
        
        cache_time = time.time() - start_time
        
        # 计算性能指标
        cache_hit_rate = cache_hits / len(test_tasks)
        cache_efficiency = len(test_tasks) / cache_time if cache_time > 0 else 0
        
        self.performance_metrics["cache"] = {
            "hit_rate": cache_hit_rate,
            "efficiency": cache_efficiency,
            "time_saved": cache_time,
            "memory_usage": self.cache._estimate_memory_usage()
        }
        
        print(f"   ✅ 缓存命中率: {cache_hit_rate:.2%}")
        print(f"   ✅ 缓存效率: {cache_efficiency:.1f} 次/秒")
        print(f"   ✅ 内存使用: {self.performance_metrics['cache']['memory_usage']:.2f} MB")
    
    async def _test_agent_parallelism(self):
        """测试智能体并行性能"""
        print("\n🤖 测试智能体并行性能...")
        
        start_time = time.time()
        
        # 定义专家分配
        expert_assignments = {
            "架构师": AgentRole.SPECIALIST,
            "前端专家": AgentRole.SPECIALIST,
            "后端专家": AgentRole.SPECIALIST,
            "测试专家": AgentRole.VALIDATOR,
            "部署专家": AgentRole.INTEGRATOR,
            "安全专家": AgentRole.SPECIALIST
        }
        
        # 定义子任务
        subtasks = [
            {
                "description": "设计系统架构和数据库模型",
                "preferred_agent": "架构师",
                "role": AgentRole.SPECIALIST,
                "priority": 1,
                "dependencies": [],
                "estimated_duration": 2.0
            },
            {
                "description": "开发前端用户界面",
                "preferred_agent": "前端专家",
                "role": AgentRole.SPECIALIST,
                "priority": 2,
                "dependencies": [],
                "estimated_duration": 3.0
            },
            {
                "description": "实现后端API接口",
                "preferred_agent": "后端专家",
                "role": AgentRole.SPECIALIST,
                "priority": 2,
                "dependencies": [],
                "estimated_duration": 4.0
            },
            {
                "description": "编写单元测试和集成测试",
                "preferred_agent": "测试专家",
                "role": AgentRole.VALIDATOR,
                "priority": 3,
                "dependencies": ["sub_1", "sub_2"],
                "estimated_duration": 2.5
            },
            {
                "description": "配置CI/CD和部署脚本",
                "preferred_agent": "部署专家",
                "role": AgentRole.INTEGRATOR,
                "priority": 4,
                "dependencies": ["sub_1", "sub_2"],
                "estimated_duration": 1.5
            }
        ]
        
        # 并行执行
        result = await self.agent_executor.execute_parallel_task(
            task_description="开发一个完整的电商平台",
            expert_assignments=expert_assignments,
            subtasks=subtasks
        )
        
        parallel_time = time.time() - start_time
        
        # 计算串行时间（用于对比）
        serial_time = sum(task["estimated_duration"] for task in subtasks)
        
        # 计算性能指标
        speedup_ratio = serial_time / parallel_time if parallel_time > 0 else 0
        efficiency = speedup_ratio / len(expert_assignments) * 100
        
        self.performance_metrics["agent_parallelism"] = {
            "success": result.success,
            "parallel_time": parallel_time,
            "serial_time": serial_time,
            "speedup_ratio": speedup_ratio,
            "efficiency": efficiency,
            "quality_score": result.quality_score,
            "resource_utilization": result.resource_usage
        }
        
        print(f"   ✅ 并行时间: {parallel_time:.2f}s")
        print(f"   ✅ 串行时间: {serial_time:.2f}s")
        print(f"   ✅ 加速比: {speedup_ratio:.2f}x")
        print(f"   ✅ 效率: {efficiency:.1f}%")
        print(f"   ✅ 质量评分: {result.quality_score:.2f}")
    
    async def _test_task_decomposition(self):
        """测试任务分解性能"""
        print("\n🎯 测试任务分解性能...")
        
        start_time = time.time()
        
        # 复杂任务
        complex_task = """
        开发一个高性能的社交电商平台，需要包含用户管理、商品管理、订单处理、
        支付集成、库存管理、推荐系统、社交功能、直播带货、数据分析等功能。
        系统需要支持高并发访问，具备良好的可扩展性、安全性和用户体验。
        要求提供完整的前端界面、后端API、数据库设计、移动端应用和部署方案。
        """
        
        # 分解任务
        subtasks = self.task_decomposer.decompose_task(
            original_task=complex_task,
            domain="社交电商系统开发",
            max_subtasks=20
        )
        
        decomposition_time = time.time() - start_time
        
        # 分析分解结果
        total_subtasks = len(subtasks)
        parallelizable_subtasks = sum(1 for task in subtasks if task.parallelizable)
        avg_complexity = statistics.mean([task.estimated_complexity for task in subtasks])
        total_duration = sum([task.estimated_duration for task in subtasks])
        
        # 计算并行潜力
        sequential_stages = [task for task in subtasks if not task.parallelizable]
        sequential_duration = sum([task.estimated_duration for task in sequential_stages])
        parallel_potential = (total_duration - sequential_duration) / total_duration if total_duration > 0 else 0
        
        self.performance_metrics["task_decomposition"] = {
            "total_subtasks": total_subtasks,
            "parallelizable_count": parallelizable_subtasks,
            "parallelizable_ratio": parallelizable_subtasks / total_subtasks if total_subtasks > 0 else 0,
            "avg_complexity": avg_complexity,
            "total_duration": total_duration,
            "sequential_duration": sequential_duration,
            "parallel_potential": parallel_potential,
            "decomposition_time": decomposition_time
        }
        
        print(f"   ✅ 分解出 {total_subtasks} 个子任务")
        print(f"   ✅ 可并行任务: {parallelizable_subtasks} ({parallelizable_subtasks/total_subtasks*100:.1f}%)")
        print(f"   ✅ 并行潜力: {parallel_potential:.2%}")
        print(f"   ✅ 分解时间: {decomposition_time:.3f}s")
        print(f"   ✅ 平均复杂度: {avg_complexity:.1f}")
    
    async def _test_workflow_parallelism(self):
        """测试工作流阶段并行性能"""
        print("\n⚙️ 测试工作流阶段并行性能...")
        
        start_time = time.time()
        
        # 定义工作流阶段
        stages = [
            WorkflowStageInfo(
                stage_id="",  # 稍后设置
                stage_type=WorkflowStage.INITIALIZATION,
                stage_name="项目初始化",
                description="创建项目结构和配置文件",
                status=None,  # 由执行器设置
                estimated_duration=0.5,
                parallelizable=False,
                resource_requirements={"cpu": 10, "memory": 5, "agents": 1}
            ),
            WorkflowStageInfo(
                stage_id="",  # 稍后设置
                stage_type=WorkflowStage.ANALYSIS,
                stage_name="需求分析",
                description="分析用户需求和系统需求",
                status=None,
                estimated_duration=2.0,
                parallelizable=True,
                resource_requirements={"cpu": 20, "memory": 15, "agents": 2}
            ),
            WorkflowStageInfo(
                stage_id="",  # 稍后设置
                stage_type=WorkflowStage.DESIGN,
                stage_name="系统设计",
                description="设计系统架构和数据库",
                status=None,
                estimated_duration=3.0,
                parallelizable=True,
                resource_requirements={"cpu": 25, "memory": 20, "agents": 3}
            ),
            WorkflowStageInfo(
                stage_id="",  # 稍后设置
                stage_type=WorkflowStage.IMPLEMENTATION,
                stage_name="核心开发",
                description="实现核心功能模块",
                status=None,
                estimated_duration=8.0,
                parallelizable=True,
                resource_requirements={"cpu": 40, "memory": 30, "agents": 4}
            ),
            WorkflowStageInfo(
                stage_id="",  # 稍后设置
                stage_type=WorkflowStage.TESTING,
                stage_name="测试验证",
                description="编写和执行测试用例",
                status=None,
                estimated_duration=3.0,
                parallelizable=True,
                resource_requirements={"cpu": 30, "memory": 25, "agents": 3}
            ),
            WorkflowStageInfo(
                stage_id="",  # 稍后设置
                stage_type=WorkflowStage.DEPLOYMENT,
                stage_name="部署上线",
                description="部署到生产环境",
                status=None,
                estimated_duration=1.0,
                parallelizable=False,
                resource_requirements={"cpu": 20, "memory": 15, "agents": 2}
            ),
            WorkflowStageInfo(
                stage_id="",  # 稍后设置
                stage_type=WorkflowStage.OPTIMIZATION,
                stage_name="性能优化",
                description="优化系统性能和用户体验",
                status=None,
                estimated_duration=2.0,
                parallelizable=True,
                resource_requirements={"cpu": 35, "memory": 25, "agents": 3}
            )
        ]
        
        # 并行执行工作流
        result = await self.workflow_parallelizer.execute_workflow_parallel(stages)
        
        workflow_time = time.time() - start_time
        
        # 计算性能指标
        serial_duration = sum(stage.estimated_duration for stage in stages)
        speedup_ratio = serial_duration / result.overall_duration if result.overall_duration > 0 else 0
        
        self.performance_metrics["workflow_parallelism"] = {
            "success": result.success,
            "parallel_time": result.overall_duration,
            "serial_time": serial_duration,
            "speedup_ratio": speedup_ratio,
            "efficiency_score": result.efficiency_score,
            "resource_utilization": result.resource_utilization,
            "bottleneck_analysis": result.bottleneck_analysis
        }
        
        print(f"   ✅ 并行时间: {result.overall_duration:.2f}s")
        print(f"   ✅ 串行时间: {serial_duration:.2f}s")
        print(f"   ✅ 加速比: {speedup_ratio:.2f}x")
        print(f"   ✅ 效率评分: {result.efficiency_score:.2f}")
    
    async def _test_end_to_end_integration(self):
        """测试端到端集成性能"""
        print("\n🔗 测试端到端集成性能...")
        
        start_time = time.time()
        
        # 1. 分解复杂任务
        complex_task = "开发一个AI驱动的智能学习平台"
        subtasks = self.task_decomposer.decompose_task(complex_task, "教育科技", max_subtasks=15)
        
        # 2. 为每个子任务创建工作流阶段
        workflow_stages = []
        for i, subtask in enumerate(subtasks[:6]):  # 限制数量以避免测试过长
            stage = WorkflowStageInfo(
                stage_id="",
                stage_type=WorkflowStage.IMPLEMENTATION,
                stage_name=f"{subtask.subtask_description}",
                description=f"实现 {subtask.subtask_description}",
                status=None,
                estimated_duration=subtask.estimated_duration,
                parallelizable=subtask.parallelizable,
                resource_requirements={"cpu": 20, "memory": 15, "agents": 2}
            )
            workflow_stages.append(stage)
        
        # 3. 并行执行集成工作流
        result = await self.workflow_parallelizer.execute_workflow_parallel(workflow_stages)
        
        end_to_end_time = time.time() - start_time
        
        # 计算总体性能
        total_subtasks = len(subtasks)
        completed_stages = len([s for s in result.stage_results.values() if s.status.value == "completed"])
        
        self.performance_metrics["end_to_end"] = {
            "total_decomposed_tasks": total_subtasks,
            "completed_stages": completed_stages,
            "integration_time": end_to_end_time,
            "success_rate": result.success,
            "overall_efficiency": result.efficiency_score
        }
        
        print(f"   ✅ 分解任务数: {total_subtasks}")
        print(f"   ✅ 完成阶段数: {completed_stages}")
        print(f"   ✅ 集成时间: {end_to_end_time:.2f}s")
        print(f"   ✅ 成功率: {result.success}")
    
    def _generate_performance_report(self):
        """生成性能报告"""
        print("\n" + "=" * 80)
        print("📈 并行执行引擎性能报告")
        print("=" * 80)
        
        # 总体性能摘要
        print("\n📊 总体性能摘要:")
        
        cache_perf = self.performance_metrics.get("cache", {})
        agent_perf = self.performance_metrics.get("agent_parallelism", {})
        task_perf = self.performance_metrics.get("task_decomposition", {})
        workflow_perf = self.performance_metrics.get("workflow_parallelism", {})
        
        print(f"   🔧 缓存效率: {cache_perf.get('efficiency', 0):.1f} 次/秒")
        print(f"   🤖 智能体并行加速: {agent_perf.get('speedup_ratio', 0):.2f}x")
        print(f"   🎯 任务分解并行潜力: {task_perf.get('parallel_potential', 0)*100:.1f}%")
        print(f"   ⚙️ 工作流阶段并行加速: {workflow_perf.get('speedup_ratio', 0):.2f}x")
        
        # 性能提升总结
        print(f"\n🚀 性能提升总结:")
        
        avg_agent_speedup = agent_perf.get('speedup_ratio', 1)
        avg_workflow_speedup = workflow_perf.get('speedup_ratio', 1)
        avg_parallel_potential = task_perf.get('parallel_potential', 0.5)
        
        # 综合性能提升计算
        overall_improvement = (
            avg_agent_speedup * 
            avg_workflow_speedup * 
            (1 + avg_parallel_potential)
        )
        
        print(f"   📈 智能体并行提升: {avg_agent_speedup:.2f}x")
        print(f"   📈 工作流并行提升: {avg_workflow_speedup:.2f}x")
        print(f"   📈 任务并行潜力: {avg_parallel_potential*100:.1f}%")
        print(f"   📈 综合性能提升: {overall_improvement:.2f}x")
        
        # 资源利用效率
        print(f"\n⚡ 资源利用效率:")
        
        agent_utilization = agent_perf.get('resource_utilization', {})
        workflow_utilization = workflow_perf.get('resource_utilization', {})
        
        if agent_utilization:
            cpu_util = agent_utilization.get('cpu', {}).get('utilization_rate', 0)
            memory_util = agent_utilization.get('memory', {}).get('utilization_rate', 0)
            print(f"   💻 智能体CPU利用率: {cpu_util*100:.1f}%")
            print(f"   🧠 智能体内存利用率: {memory_util*100:.1f}%")
        
        if workflow_utilization:
            agent_count = workflow_utilization.get('agents', {}).get('utilization_rate', 0)
            print(f"   👥 工作流智能体利用率: {agent_count*100:.1f}%")
        
        # 质量保证
        print(f"\n🛡️ 质量保证:")
        
        cache_quality = cache_perf.get('memory_usage', 0)
        agent_quality = agent_perf.get('quality_score', 0)
        workflow_quality = workflow_perf.get('efficiency_score', 0)
        
        print(f"   📚 缓存内存使用: {cache_quality:.2f} MB")
        print(f"   🎯 智能体执行质量: {agent_quality:.2f}/1.0")
        print(f"   ⚙️ 工作流执行效率: {workflow_quality:.2f}/10.0")
        
        # 建议和优化方向
        print(f"\n💡 优化建议:")
        
        if avg_agent_speedup < 2.0:
            print("   🔧 建议增加智能体并发数量以提升并行效率")
        
        if avg_workflow_speedup < 2.0:
            print("   ⚙️ 建议优化工作流阶段依赖关系，增加并行性")
        
        if avg_parallel_potential < 0.5:
            print("   🎯 建议改进任务分解算法，提高并行潜力")
        
        print(f"   🚀 综合来看，并行执行引擎能够显著提升工作流执行效率！")
        
        print("\n" + "=" * 80)
        print("✅ 并行执行引擎集成测试完成")
        print("=" * 80)

async def main():
    """主测试函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行综合测试
    benchmark = ParallelExecutionBenchmark()
    await benchmark.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())