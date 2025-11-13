#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 综合评估系统 V5 (Comprehensive Evaluation System V5)
对A项目进行全面的功能测试和性能评估。
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import asyncio
import logging
import time
import psutil
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import unittest
from dataclasses import dataclass, field

# 动态添加项目根目录到sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """测试结果"""
    name: str
    status: str  # passed, failed, error
    duration: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PerformanceMetrics:
    """性能指标"""
    cpu_usage: float
    memory_usage: float
    response_time: float
    throughput: float
    error_rate: float
    timestamp: datetime = field(default_factory=datetime.now)

class ComprehensiveEvaluationV5:
    """
    综合评估系统 V5
    """
    
    def __init__(self):
        self.test_results: List[TestResult] = []
        self.performance_metrics: List[PerformanceMetrics] = []
        self.start_time = datetime.now()
        
        # 测试组件
        self.components = {
            "model_adapter": None,
            "consciousness_system": None,
            "arq_engine": None,
            "fusion_agent": None,
            "workflow_engine": None,
            "test_heal_system": None,
            "maintenance_system": None,
            "context_cache": None,
            "hook_integration": None,
            "evolution_engine": None
        }
        
        # 性能基准
        self.benchmarks = {
            "response_time": 1.0,  # 秒
            "memory_usage": 512,   # MB
            "cpu_usage": 80,       # 百分比
            "success_rate": 0.95   # 百分比
        }
        
        logger.info("综合评估系统V5初始化完成")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("🚀 开始运行全面测试和性能评估...")
        
        # 1. 导入测试
        await self._test_imports()
        
        # 2. 初始化组件测试
        await self._test_initialization()
        
        # 3. 功能测试
        await self._test_functionality()
        
        # 4. 性能测试
        await self._test_performance()
        
        # 5. 集成测试
        await self._test_integration()
        
        # 6. 稳定性测试
        await self._test_stability()
        
        # 7. 生成评估报告
        report = await self._generate_evaluation_report()
        
        return report
    
    async def _test_imports(self):
        """测试模块导入"""
        logger.info("📦 测试模块导入...")
        
        imports_to_test = [
            ("universal_llm_adapter_v13", "iflow.adapters"),
            ("ultimate_consciousness_system_v5", "iflow.core"),
            ("ultimate_arq_engine_v5", "iflow.core"),
            ("ultimate_fusion_agent_v5", "iflow.agents"),
            ("ultimate_workflow_engine_v4", "iflow.core"),
            ("auto_test_heal_system_v5", "iflow.core"),
            ("predictive_maintenance_system_v5", "iflow.core"),
            ("intelligent_context_cache_v5", "iflow.core"),
            ("comprehensive_hook_manager_v4", "iflow.hooks"),
            ("self_evolution_engine_v4", "iflow.core")
        ]
        
        for module_name, package in imports_to_test:
            start_time = time.time()
            try:
                module = __import__(f"{package}.{module_name}", fromlist=[module_name])
                duration = time.time() - start_time
                
                self.test_results.append(TestResult(
                    name=f"导入_{module_name}",
                    status="passed",
                    duration=duration,
                    message=f"成功导入{module_name}",
                    details={"module": module_name, "package": package}
                ))
                
                logger.info(f"✅ {module_name} 导入成功")
                
            except Exception as e:
                duration = time.time() - start_time
                
                self.test_results.append(TestResult(
                    name=f"导入_{module_name}",
                    status="error",
                    duration=duration,
                    message=f"导入失败: {str(e)}",
                    details={"error": traceback.format_exc()}
                ))
                
                logger.error(f"❌ {module_name} 导入失败: {e}")
    
    async def _test_initialization(self):
        """测试组件初始化"""
        logger.info("🔧 测试组件初始化...")
        
        # 测试工作流引擎初始化
        await self._test_workflow_engine_initialization()
        
        # 测试各组件初始化
        if self.components["workflow_engine"]:
            await self._test_component_initialization()
    
    async def _test_workflow_engine_initialization(self):
        """测试工作流引擎初始化"""
        start_time = time.time()
        
        try:
            from iflow.core.ultimate_workflow_engine_v4 import UltimateWorkflowEngineV4
            
            # 创建引擎实例
            engine = UltimateWorkflowEngineV4()
            
            # 初始化
            await engine.initialize()
            
            duration = time.time() - start_time
            
            self.test_results.append(TestResult(
                name="工作流引擎初始化",
                status="passed",
                duration=duration,
                message="工作流引擎初始化成功",
                details={"initialized": engine._initialized}
            ))
            
            # 保存引擎引用
            self.components["workflow_engine"] = engine
            
            # 获取其他组件引用
            self.components["model_adapter"] = engine.model_adapter
            self.components["consciousness_system"] = engine.consciousness_system
            self.components["arq_engine"] = engine.arq_engine
            self.components["fusion_agent"] = engine.fusion_agent
            self.components["test_heal_system"] = getattr(engine, 'test_heal_system', None)
            self.components["maintenance_system"] = getattr(engine, 'maintenance_system', None)
            self.components["context_cache"] = getattr(engine, 'context_cache', None)
            self.components["hook_integration"] = engine.hook_integration
            self.components["evolution_engine"] = engine.evolution_engine
            
            logger.info("✅ 工作流引擎初始化成功")
            
        except Exception as e:
            duration = time.time() - start_time
            
            self.test_results.append(TestResult(
                name="工作流引擎初始化",
                status="error",
                duration=duration,
                message=f"初始化失败: {str(e)}",
                details={"error": traceback.format_exc()}
            ))
            
            logger.error(f"❌ 工作流引擎初始化失败: {e}")
    
    async def _test_component_initialization(self):
        """测试各组件初始化状态"""
        components_to_test = [
            ("模型适配器", "model_adapter"),
            ("意识系统", "consciousness_system"),
            ("ARQ引擎", "arq_engine"),
            ("融合智能体", "fusion_agent"),
            ("自动测试修复系统", "test_heal_system"),
            ("预测性维护系统", "maintenance_system"),
            ("智能缓存系统", "context_cache"),
            ("Hook集成系统", "hook_integration"),
            ("进化引擎", "evolution_engine")
        ]
        
        for name, key in components_to_test:
            start_time = time.time()
            
            if self.components.get(key):
                duration = time.time() - start_time
                
                self.test_results.append(TestResult(
                    name=f"{name}状态",
                    status="passed",
                    duration=duration,
                    message=f"{name}已初始化",
                    details={"component": key}
                ))
                
                logger.info(f"✅ {name}已初始化")
            else:
                duration = time.time() - start_time
                
                self.test_results.append(TestResult(
                    name=f"{name}状态",
                    status="failed",
                    duration=duration,
                    message=f"{name}未初始化",
                    details={"component": key}
                ))
                
                logger.warning(f"⚠️ {name}未初始化")
    
    async def _test_functionality(self):
        """测试功能"""
        logger.info("⚙️ 测试功能...")
        
        # 测试任务执行
        await self._test_task_execution()
        
        # 测试意识系统
        await self._test_consciousness_system()
        
        # 测试ARQ引擎
        await self._test_arq_engine()
        
        # 测试融合智能体
        await self._test_fusion_agent()
        
        # 测试缓存系统
        await self._test_context_cache()
        
        # 测试维护系统
        await self._test_maintenance_system()
    
    async def _test_task_execution(self):
        """测试任务执行"""
        if not self.components["workflow_engine"]:
            return
        
        start_time = time.time()
        
        try:
            # 执行简单任务
            result = await self.components["workflow_engine"].execute_task(
                "分析1+1等于多少",
                priority="medium"
            )
            
            duration = time.time() - start_time
            
            if result.get("success"):
                self.test_results.append(TestResult(
                    name="任务执行",
                    status="passed",
                    duration=duration,
                    message="任务执行成功",
                    details={"task": "分析1+1等于多少", "result": result}
                ))
                
                logger.info("✅ 任务执行成功")
            else:
                self.test_results.append(TestResult(
                    name="任务执行",
                    status="failed",
                    duration=duration,
                    message=f"任务执行失败: {result.get('error')}",
                    details={"result": result}
                ))
                
                logger.warning(f"⚠️ 任务执行失败: {result.get('error')}")
                
        except Exception as e:
            duration = time.time() - start_time
            
            self.test_results.append(TestResult(
                name="任务执行",
                status="error",
                duration=duration,
                message=f"任务执行异常: {str(e)}",
                details={"error": traceback.format_exc()}
            ))
            
            logger.error(f"❌ 任务执行异常: {e}")
    
    async def _test_consciousness_system(self):
        """测试意识系统"""
        if not self.components["consciousness_system"]:
            return
        
        start_time = time.time()
        
        try:
            from iflow.core.ultimate_consciousness_system_v4 import ThoughtType
            
            # 记录测试思想
            thought = await self.components["consciousness_system"].record_thought(
                content="这是一个测试思想",
                thought_type=ThoughtType.ANALYTICAL,
                confidence=0.9,
                importance=0.8
            )
            
            # 获取意识上下文
            context = await self.components["consciousness_system"].get_consciousness_context()
            
            duration = time.time() - start_time
            
            if thought and context:
                self.test_results.append(TestResult(
                    name="意识系统功能",
                    status="passed",
                    duration=duration,
                    message="意识系统功能正常",
                    details={"thought_id": thought.id, "context_keys": list(context.keys())}
                ))
                
                logger.info("✅ 意识系统功能正常")
            else:
                self.test_results.append(TestResult(
                    name="意识系统功能",
                    status="failed",
                    duration=duration,
                    message="意识系统功能异常",
                    details={"thought": thought, "context": context}
                ))
                
                logger.warning("⚠️ 意识系统功能异常")
                
        except Exception as e:
            duration = time.time() - start_time
            
            self.test_results.append(TestResult(
                name="意识系统功能",
                status="error",
                duration=duration,
                message=f"意识系统测试异常: {str(e)}",
                details={"error": traceback.format_exc()}
            ))
            
            logger.error(f"❌ 意识系统测试异常: {e}")
    
    async def _test_arq_engine(self):
        """测试ARQ引擎"""
        if not self.components["arq_engine"] or not self.components["model_adapter"]:
            return
        
        start_time = time.time()
        
        try:
            # 处理简单推理
            result = await self.components["arq_engine"].process_reasoning(
                task="测试ARQ推理",
                context=[{"type": "test"}],
                llm_adapter=self.components["model_adapter"]
            )
            
            duration = time.time() - start_time
            
            if result.get("success"):
                self.test_results.append(TestResult(
                    name="ARQ引擎功能",
                    status="passed",
                    duration=duration,
                    message="ARQ引擎功能正常",
                    details={"result": result}
                ))
                
                logger.info("✅ ARQ引擎功能正常")
            else:
                self.test_results.append(TestResult(
                    name="ARQ引擎功能",
                    status="failed",
                    duration=duration,
                    message=f"ARQ引擎功能异常: {result.get('error')}",
                    details={"result": result}
                ))
                
                logger.warning(f"⚠️ ARQ引擎功能异常: {result.get('error')}")
                
        except Exception as e:
            duration = time.time() - start_time
            
            self.test_results.append(TestResult(
                name="ARQ引擎功能",
                status="error",
                duration=duration,
                message=f"ARQ引擎测试异常: {str(e)}",
                details={"error": traceback.format_exc()}
            ))
            
            logger.error(f"❌ ARQ引擎测试异常: {e}")
    
    async def _test_fusion_agent(self):
        """测试融合智能体"""
        if not self.components["fusion_agent"]:
            return
        
        start_time = time.time()
        
        try:
            # 执行分析任务
            analysis = await self.components["fusion_agent"].analyze_task("测试任务分析")
            
            duration = time.time() - start_time
            
            if analysis:
                self.test_results.append(TestResult(
                    name="融合智能体功能",
                    status="passed",
                    duration=duration,
                    message="融合智能体功能正常",
                    details={"analysis": analysis}
                ))
                
                logger.info("✅ 融合智能体功能正常")
            else:
                self.test_results.append(TestResult(
                    name="融合智能体功能",
                    status="failed",
                    duration=duration,
                    message="融合智能体功能异常",
                    details={"analysis": analysis}
                ))
                
                logger.warning("⚠️ 融合智能体功能异常")
                
        except Exception as e:
            duration = time.time() - start_time
            
            self.test_results.append(TestResult(
                name="融合智能体功能",
                status="error",
                duration=duration,
                message=f"融合智能体测试异常: {str(e)}",
                details={"error": traceback.format_exc()}
            ))
            
            logger.error(f"❌ 融合智能体测试异常: {e}")
    
    async def _test_context_cache(self):
        """测试缓存系统"""
        if not self.components["context_cache"]:
            return
        
        start_time = time.time()
        
        try:
            from iflow.core.intelligent_context_cache_v5 import ContextType
            
            # 存储测试数据
            cache_id = await self.components["context_cache"].put(
                key="test_key",
                value="test_value",
                context_type=ContextType.TASK
            )
            
            # 获取数据
            cached_value = await self.components["context_cache"].get("test_key")
            
            # 获取缓存统计
            stats = await self.components["context_cache"].get_cache_stats()
            
            duration = time.time() - start_time
            
            if cache_id and cached_value == "test_value":
                self.test_results.append(TestResult(
                    name="缓存系统功能",
                    status="passed",
                    duration=duration,
                    message="缓存系统功能正常",
                    details={"cache_id": cache_id, "stats": stats}
                ))
                
                logger.info("✅ 缓存系统功能正常")
            else:
                self.test_results.append(TestResult(
                    name="缓存系统功能",
                    status="failed",
                    duration=duration,
                    message="缓存系统功能异常",
                    details={"cache_id": cache_id, "cached_value": cached_value}
                ))
                
                logger.warning("⚠️ 缓存系统功能异常")
                
        except Exception as e:
            duration = time.time() - start_time
            
            self.test_results.append(TestResult(
                name="缓存系统功能",
                status="error",
                duration=duration,
                message=f"缓存系统测试异常: {str(e)}",
                details={"error": traceback.format_exc()}
            ))
            
            logger.error(f"❌ 缓存系统测试异常: {e}")
    
    async def _test_maintenance_system(self):
        """测试维护系统"""
        if not self.components["maintenance_system"]:
            return
        
        start_time = time.time()
        
        try:
            # 获取系统健康状态
            health = await self.components["maintenance_system"].get_system_health()
            
            duration = time.time() - start_time
            
            if health:
                self.test_results.append(TestResult(
                    name="维护系统功能",
                    status="passed",
                    duration=duration,
                    message="维护系统功能正常",
                    details={"health": health}
                ))
                
                logger.info("✅ 维护系统功能正常")
            else:
                self.test_results.append(TestResult(
                    name="维护系统功能",
                    status="failed",
                    duration=duration,
                    message="维护系统功能异常",
                    details={"health": health}
                ))
                
                logger.warning("⚠️ 维护系统功能异常")
                
        except Exception as e:
            duration = time.time() - start_time
            
            self.test_results.append(TestResult(
                name="维护系统功能",
                status="error",
                duration=duration,
                message=f"维护系统测试异常: {str(e)}",
                details={"error": traceback.format_exc()}
            ))
            
            logger.error(f"❌ 维护系统测试异常: {e}")
    
    async def _test_performance(self):
        """测试性能"""
        logger.info("📊 测试性能...")
        
        # 收集系统资源使用情况
        await self._collect_performance_metrics()
        
        # 测试响应时间
        await self._test_response_time()
        
        # 测试吞吐量
        await self._test_throughput()
        
        # 测试内存使用
        await self._test_memory_usage()
    
    async def _collect_performance_metrics(self):
        """收集性能指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            
            # 记录性能指标
            self.performance_metrics.append(PerformanceMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory_mb,
                response_time=0.0,  # 将在其他测试中填充
                throughput=0.0,    # 将在其他测试中填充
                error_rate=0.0     # 将在其他测试中填充
            ))
            
            logger.info(f"📈 性能指标 - CPU: {cpu_percent:.1f}%, 内存: {memory_mb:.1f}MB")
            
        except Exception as e:
            logger.error(f"收集性能指标失败: {e}")
    
    async def _test_response_time(self):
        """测试响应时间"""
        if not self.components["workflow_engine"]:
            return
        
        response_times = []
        test_tasks = [
            "计算2+2",
            "分析天气",
            "推荐书籍",
            "解释概念",
            "生成代码"
        ]
        
        for task in test_tasks:
            start_time = time.time()
            
            try:
                await self.components["workflow_engine"].execute_task(task)
                response_time = time.time() - start_time
                response_times.append(response_time)
                
            except Exception as e:
                logger.warning(f"任务'{task}'执行失败: {e}")
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            
            # 更新最新的性能指标
            if self.performance_metrics:
                self.performance_metrics[-1].response_time = avg_response_time
            
            # 记录测试结果
            status = "passed" if avg_response_time < self.benchmarks["response_time"] else "failed"
            
            self.test_results.append(TestResult(
                name="响应时间",
                status=status,
                duration=avg_response_time,
                message=f"平均响应时间: {avg_response_time:.3f}秒",
                details={
                    "avg_response_time": avg_response_time,
                    "benchmark": self.benchmarks["response_time"],
                    "response_times": response_times
                }
            ))
            
            logger.info(f"⏱️ 平均响应时间: {avg_response_time:.3f}秒")
    
    async def _test_throughput(self):
        """测试吞吐量"""
        if not self.components["workflow_engine"]:
            return
        
        start_time = time.time()
        task_count = 10
        completed_tasks = 0
        
        # 并发执行任务
        tasks = []
        for i in range(task_count):
            task = self.components["workflow_engine"].execute_task(f"测试任务{i}")
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计成功任务
        for result in results:
            if isinstance(result, dict) and result.get("success"):
                completed_tasks += 1
        
        total_time = time.time() - start_time
        throughput = completed_tasks / total_time if total_time > 0 else 0
        
        # 更新最新的性能指标
        if self.performance_metrics:
            self.performance_metrics[-1].throughput = throughput
        
        # 记录测试结果
        self.test_results.append(TestResult(
            name="吞吐量",
            status="passed",
            duration=total_time,
            message=f"吞吐量: {throughput:.2f}任务/秒",
            details={
                "throughput": throughput,
                "completed_tasks": completed_tasks,
                "total_tasks": task_count,
                "total_time": total_time
            }
        ))
        
        logger.info(f"🚀 吞吐量: {throughput:.2f}任务/秒")
    
    async def _test_memory_usage(self):
        """测试内存使用"""
        if not self.performance_metrics:
            return
        
        memory_usage = self.performance_metrics[-1].memory_usage
        benchmark = self.benchmarks["memory_usage"]
        
        status = "passed" if memory_usage < benchmark else "failed"
        
        self.test_results.append(TestResult(
            name="内存使用",
            status=status,
            duration=0.0,
            message=f"内存使用: {memory_usage:.1f}MB",
            details={
                "memory_usage": memory_usage,
                "benchmark": benchmark
            }
        ))
        
        logger.info(f"💾 内存使用: {memory_usage:.1f}MB")
    
    async def _test_integration(self):
        """测试集成"""
        logger.info("🔗 测试集成...")
        
        # 测试组件间协作
        await self._test_component_collaboration()
        
        # 测试工作流执行
        await self._test_workflow_execution()
    
    async def _test_component_collaboration(self):
        """测试组件协作"""
        if not all([self.components["workflow_engine"], 
                   self.components["consciousness_system"],
                   self.components["fusion_agent"]]):
            return
        
        start_time = time.time()
        
        try:
            # 执行一个需要多个组件协作的任务
            result = await self.components["workflow_engine"].execute_task(
                "写一个Python函数计算斐波那契数列",
                priority="high"
            )
            
            duration = time.time() - start_time
            
            if result.get("success"):
                self.test_results.append(TestResult(
                    name="组件协作",
                    status="passed",
                    duration=duration,
                    message="组件协作正常",
                    details={"result": result}
                ))
                
                logger.info("✅ 组件协作正常")
            else:
                self.test_results.append(TestResult(
                    name="组件协作",
                    status="failed",
                    duration=duration,
                    message=f"组件协作失败: {result.get('error')}",
                    details={"result": result}
                ))
                
                logger.warning(f"⚠️ 组件协作失败: {result.get('error')}")
                
        except Exception as e:
            duration = time.time() - start_time
            
            self.test_results.append(TestResult(
                name="组件协作",
                status="error",
                duration=duration,
                message=f"组件协作异常: {str(e)}",
                details={"error": traceback.format_exc()}
            ))
            
            logger.error(f"❌ 组件协作异常: {e}")
    
    async def _test_workflow_execution(self):
        """测试工作流执行"""
        if not self.components["workflow_engine"]:
            return
        
        start_time = time.time()
        
        try:
            # 定义复杂工作流
            workflow = {
                "name": "测试工作流",
                "steps": [
                    {"description": "分析问题", "critical": False},
                    {"description": "设计方案", "critical": False},
                    {"description": "实现代码", "critical": True},
                    {"description": "测试验证", "critical": False}
                ],
                "context": {"type": "test"}
            }
            
            # 执行工作流
            result = await self.components["workflow_engine"].execute_complex_workflow(workflow)
            
            duration = time.time() - start_time
            
            if result.get("success"):
                self.test_results.append(TestResult(
                    name="工作流执行",
                    status="passed",
                    duration=duration,
                    message="工作流执行成功",
                    details={"result": result}
                ))
                
                logger.info("✅ 工作流执行成功")
            else:
                self.test_results.append(TestResult(
                    name="工作流执行",
                    status="failed",
                    duration=duration,
                    message=f"工作流执行失败: {result.get('error')}",
                    details={"result": result}
                ))
                
                logger.warning(f"⚠️ 工作流执行失败: {result.get('error')}")
                
        except Exception as e:
            duration = time.time() - start_time
            
            self.test_results.append(TestResult(
                name="工作流执行",
                status="error",
                duration=duration,
                message=f"工作流执行异常: {str(e)}",
                details={"error": traceback.format_exc()}
            ))
            
            logger.error(f"❌ 工作流执行异常: {e}")
    
    async def _test_stability(self):
        """测试稳定性"""
        logger.info("🛡️ 测试稳定性...")
        
        # 压力测试
        await self._test_stress()
        
 # 错误恢复测试
        await self._test_error_recovery()
    
    async def _test_stress(self):
        """压力测试"""
        if not self.components["workflow_engine"]:
            return
        
        start_time = time.time()
        
        try:
            # 执行大量任务
            task_count = 50
            success_count = 0
            
            for i in range(task_count):
                try:
                    result = await self.components["workflow_engine"].execute_task(
                        f"压力测试任务{i}",
                        priority="low"
                    )
                    
                    if result.get("success"):
                        success_count += 1
                        
                except Exception as e:
                    logger.warning(f"压力测试任务{i}失败: {e}")
            
            duration = time.time() - start_time
            success_rate = success_count / task_count
            
            # 更新最新的性能指标
            if self.performance_metrics:
                self.performance_metrics[-1].error_rate = 1 - success_rate
            
            status = "passed" if success_rate > 0.8 else "failed"
            
            self.test_results.append(TestResult(
                name="压力测试",
                status=status,
                duration=duration,
                message=f"成功率: {success_rate:.2%}",
                details={
                    "task_count": task_count,
                    "success_count": success_count,
                    "success_rate": success_rate
                }
            ))
            
            logger.info(f"💪 压力测试完成，成功率: {success_rate:.2%}")
            
        except Exception as e:
            duration = time.time() - start_time
            
            self.test_results.append(TestResult(
                name="压力测试",
                status="error",
                duration=duration,
                message=f"压力测试异常: {str(e)}",
                details={"error": traceback.format_exc()}
            ))
            
            logger.error(f"❌ 压力测试异常: {e}")
    
    async def _test_error_recovery(self):
        """错误恢复测试"""
        if not self.components["workflow_engine"]:
            return
        
        start_time = time.time()
        
        try:
            # 故意执行一个可能失败的任务
            result = await self.components["workflow_engine"].execute_task(
                "这是一个故意设计的可能导致错误的测试任务，包含无效输入和特殊字符：@#$%^&*()",
                priority="low"
            )
            
            duration = time.time() - start_time
            
            # 检查系统是否仍然响应
            recovery_result = await self.components["workflow_engine"].execute_task(
                "系统恢复测试",
                priority="low"
            )
            
            if recovery_result.get("success"):
                self.test_results.append(TestResult(
                    name="错误恢复",
                    status="passed",
                    duration=duration,
                    message="错误恢复成功",
                    details={
                        "original_result": result,
                        "recovery_result": recovery_result
                    }
                ))
                
                logger.info("✅ 错误恢复成功")
            else:
                self.test_results.append(TestResult(
                    name="错误恢复",
                    status="failed",
                    duration=duration,
                    message="错误恢复失败",
                    details={
                        "original_result": result,
                        "recovery_result": recovery_result
                    }
                ))
                
                logger.warning("⚠️ 错误恢复失败")
                
        except Exception as e:
            duration = time.time() - start_time
            
            self.test_results.append(TestResult(
                name="错误恢复",
                status="error",
                duration=duration,
                message=f"错误恢复测试异常: {str(e)}",
                details={"error": traceback.format_exc()}
            ))
            
            logger.error(f"❌ 错误恢复测试异常: {e}")
    
    async def _generate_evaluation_report(self) -> Dict[str, Any]:
        """生成评估报告"""
        logger.info("📝 生成评估报告...")
        
        # 统计测试结果
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.status == "passed")
        failed_tests = sum(1 for r in self.test_results if r.status == "failed")
        error_tests = sum(1 for r in self.test_results if r.status == "error")
        
        # 计算成功率
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        # 性能分析
        performance_summary = {}
        if self.performance_metrics:
            latest_metrics = self.performance_metrics[-1]
            performance_summary = {
                "cpu_usage": latest_metrics.cpu_usage,
                "memory_usage": latest_metrics.memory_usage,
                "response_time": latest_metrics.response_time,
                "throughput": latest_metrics.throughput,
                "error_rate": latest_metrics.error_rate
            }
        
        # 生成报告
        report = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "evaluation_duration": (datetime.now() - self.start_time).total_seconds(),
            "test_summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "success_rate": success_rate
            },
            "performance_summary": performance_summary,
            "benchmarks": self.benchmarks,
            "test_results": [
                {
                    "name": r.name,
                    "status": r.status,
                    "duration": r.duration,
                    "message": r.message,
                    "timestamp": r.timestamp.isoformat()
                } for r in self.test_results
            ],
            "component_status": {
                name: (component is not None).__str__()
                for name, component in self.components.items()
            },
            "recommendations": self._generate_recommendations()
        }
        
        # 保存报告
        report_path = project_root / "A项目" / "iflow" / "reports" / "comprehensive_evaluation_v5.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 评估报告已保存到: {report_path}")
        
        # 显示摘要
        logger.info("\n" + "="*50)
        logger.info("📊 评估结果摘要:")
        logger.info(f"总测试数: {total_tests}")
        logger.info(f"通过测试: {passed_tests}")
        logger.info(f"失败测试: {failed_tests}")
        logger.info(f"错误测试: {error_tests}")
        logger.info(f"成功率: {success_rate:.2%}")
        
        if performance_summary:
            logger.info(f"CPU使用率: {performance_summary['cpu_usage']:.1f}%")
            logger.info(f"内存使用: {performance_summary['memory_usage']:.1f}MB")
            logger.info(f"响应时间: {performance_summary['response_time']:.3f}秒")
            logger.info(f"吞吐量: {performance_summary['throughput']:.2f}任务/秒")
            logger.info(f"错误率: {performance_summary['error_rate']:.2%}")
        
        logger.info("="*50)
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于测试结果的建议
        failed_tests = [r for r in self.test_results if r.status == "failed"]
        error_tests = [r for r in self.test_results if r.status == "error"]
        
        if failed_tests:
            recommendations.append(f"修复{len(failed_tests)}个失败的测试")
        
        if error_tests:
            recommendations.append(f"解决{len(error_tests)}个错误测试")
        
        # 基于性能的建议
        if self.performance_metrics:
            latest_metrics = self.performance_metrics[-1]
            
            if latest_metrics.response_time > self.benchmarks["response_time"]:
                recommendations.append("优化响应时间，当前超过基准值")
            
            if latest_metrics.memory_usage > self.benchmarks["memory_usage"]:
                recommendations.append("优化内存使用，当前超过基准值")
            
            if latest_metrics.cpu_usage > self.benchmarks["cpu_usage"]:
                recommendations.append("优化CPU使用率，当前超过基准值")
            
            if latest_metrics.error_rate > (1 - self.benchmarks["success_rate"]):
                recommendations.append("降低错误率，提高系统稳定性")
        
        # 基于组件状态的建议
        missing_components = [
            name for name, component in self.components.items() 
            if component is None
        ]
        
        if missing_components:
            recommendations.append(f"初始化缺失的组件: {', '.join(missing_components)}")
        
        # 通用建议
        if not recommendations:
            recommendations.append("系统运行良好，继续保持")
        
        return recommendations

async def main():
    """主函数"""
    evaluator = ComprehensiveEvaluationV5()
    report = await evaluator.run_all_tests()
    
    # 返回退出码
    success_rate = report["test_summary"]["success_rate"]
    exit_code = 0 if success_rate > 0.8 else 1
    
    return exit_code

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)