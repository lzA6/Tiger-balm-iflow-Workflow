#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 工作流阶段并行执行器 V2
实现工作流不同阶段的并行执行，最大化整体执行效率。
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict, deque
import threading
from contextlib import asynccontextmanager

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class WorkflowStage(Enum):
    """工作流阶段"""
    INITIALIZATION = "initialization"      # 初始化阶段
    ANALYSIS = "analysis"                  # 分析阶段
    DESIGN = "design"                      # 设计阶段
    IMPLEMENTATION = "implementation"      # 实现阶段
    TESTING = "testing"                    # 测试阶段
    DEPLOYMENT = "deployment"              # 部署阶段
    OPTIMIZATION = "optimization"          # 优化阶段
    MONITORING = "monitoring"              # 监控阶段

class StageStatus(Enum):
    """阶段状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"

@dataclass
class WorkflowStageInfo:
    """工作流阶段信息"""
    stage_id: str
    stage_type: WorkflowStage
    stage_name: str
    description: str
    status: StageStatus
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    progress: float = 0.0  # 0.0-1.0
    result: Optional[Any] = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    parallelizable: bool = True
    priority: int = 5
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: float = 1.0

@dataclass
class ParallelWorkflowResult:
    """并行工作流执行结果"""
    workflow_id: str
    success: bool
    stage_results: Dict[str, WorkflowStageInfo]
    overall_duration: float
    efficiency_score: float
    resource_utilization: Dict[str, Any]
    bottleneck_analysis: Dict[str, Any]
    quality_metrics: Dict[str, Any]

class StageDependencyManager:
    """阶段依赖管理器"""
    
    def __init__(self):
        self.stage_graph: Dict[str, WorkflowStageInfo] = {}
        self.dependency_matrix: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.execution_order: List[str] = []
    
    def add_stage(self, stage: WorkflowStageInfo):
        """添加阶段"""
        self.stage_graph[stage.stage_id] = stage
        
        # 构建依赖关系
        for dep_id in stage.dependencies:
            self.dependency_matrix[stage.stage_id].add(dep_id)
            self.reverse_dependencies[dep_id].add(stage.stage_id)
    
    def calculate_execution_order(self) -> List[str]:
        """计算执行顺序（拓扑排序）"""
        # 使用Kahn算法进行拓扑排序
        in_degree = defaultdict(int)
        all_stages = set(self.stage_graph.keys())
        
        # 计算入度
        for stage_id in all_stages:
            in_degree[stage_id] = len(self.dependency_matrix[stage_id])
        
        # 找到所有入度为0的节点
        queue = deque([stage_id for stage_id in all_stages if in_degree[stage_id] == 0])
        result = []
        
        while queue:
            stage_id = queue.popleft()
            result.append(stage_id)
            
            # 更新依赖该节点的所有节点的入度
            for dependent_id in self.reverse_dependencies[stage_id]:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)
        
        if len(result) != len(all_stages):
            raise ValueError("检测到循环依赖，无法确定执行顺序")
        
        self.execution_order = result
        return result
    
    def get_ready_stages(self, completed_stages: Set[str]) -> Set[str]:
        """获取可以执行的阶段"""
        ready_stages = set()
        
        for stage_id, stage in self.stage_graph.items():
            if stage_id not in completed_stages and stage.status == StageStatus.PENDING:
                # 检查所有依赖是否完成
                dependencies_met = all(dep_id in completed_stages 
                                     for dep_id in stage.dependencies)
                if dependencies_met:
                    ready_stages.add(stage_id)
        
        return ready_stages
    
    def get_parallelizable_stages(self, ready_stages: Set[str]) -> Set[str]:
        """获取可以并行执行的阶段"""
        parallelizable_stages = set()
        
        for stage_id in ready_stages:
            stage = self.stage_graph[stage_id]
            if stage.parallelizable:
                # 检查是否有资源冲突
                if not self._has_resource_conflicts(stage_id, parallelizable_stages):
                    parallelizable_stages.add(stage_id)
        
        return parallelizable_stages
    
    def _has_resource_conflicts(self, stage_id: str, existing_stages: Set[str]) -> bool:
        """检查是否存在资源冲突"""
        new_stage = self.stage_graph[stage_id]
        new_resources = new_stage.resource_requirements
        
        for existing_id in existing_stages:
            existing_stage = self.stage_graph[existing_id]
            existing_resources = existing_stage.resource_requirements
            
            # 简单的资源冲突检测
            # 实际应该更复杂的资源管理
            if (new_resources.get("exclusive", False) or 
                existing_resources.get("exclusive", False)):
                return True
        
        return False

class WorkflowResourceAllocator:
    """工作流资源分配器"""
    
    def __init__(self, total_resources: Dict[str, Any]):
        self.total_resources = total_resources
        self.allocated_resources: Dict[str, Dict[str, Any]] = {}
        self.available_resources = total_resources.copy()
        self._lock = threading.RLock()
    
    def allocate_resources(self, stage_id: str, required_resources: Dict[str, Any]) -> bool:
        """分配资源"""
        with self._lock:
            # 检查资源是否足够
            if self._check_resource_availability(required_resources):
                # 分配资源
                self.allocated_resources[stage_id] = required_resources.copy()
                
                # 更新可用资源
                for resource_type, amount in required_resources.items():
                    if resource_type in self.available_resources:
                        self.available_resources[resource_type] -= amount
                
                logger.info(f"为阶段 {stage_id} 分配资源: {required_resources}")
                return True
            else:
                logger.warning(f"阶段 {stage_id} 资源不足: {required_resources}")
                return False
    
    def release_resources(self, stage_id: str):
        """释放资源"""
        with self._lock:
            if stage_id in self.allocated_resources:
                released_resources = self.allocated_resources[stage_id]
                
                # 恢复可用资源
                for resource_type, amount in released_resources.items():
                    if resource_type in self.available_resources:
                        self.available_resources[resource_type] += amount
                
                # 移除分配记录
                del self.allocated_resources[stage_id]
                
                logger.info(f"阶段 {stage_id} 完成，释放资源: {released_resources}")
    
    def _check_resource_availability(self, required_resources: Dict[str, Any]) -> bool:
        """检查资源可用性"""
        for resource_type, required_amount in required_resources.items():
            available_amount = self.available_resources.get(resource_type, 0)
            if available_amount < required_amount:
                return False
        return True
    
    def get_resource_utilization(self) -> Dict[str, Any]:
        """获取资源使用情况"""
        with self._lock:
            utilization = {}
            for resource_type, total_amount in self.total_resources.items():
                used_amount = total_amount - self.available_resources.get(resource_type, total_amount)
                utilization[resource_type] = {
                    "total": total_amount,
                    "used": used_amount,
                    "available": self.available_resources.get(resource_type, 0),
                    "utilization_rate": used_amount / total_amount if total_amount > 0 else 0
                }
            return utilization

class WorkflowStageParallelizer:
    """
    工作流阶段并行执行器
    """
    
    def __init__(self, max_concurrent_stages: int = 5):
        self.parallelizer_id = str(uuid.uuid4())
        self.max_concurrent_stages = max_concurrent_stages
        
        # 核心组件
        self.dependency_manager = StageDependencyManager()
        self.resource_allocator = WorkflowResourceAllocator({
            "cpu": 100,      # CPU使用百分比
            "memory": 100,   # 内存使用百分比
            "io": 100,       # IO使用百分比
            "network": 100,  # 网络使用百分比
            "agents": 10     # 并发智能体数量
        })
        
        # 执行状态
        self.active_stages: Dict[str, WorkflowStageInfo] = {}
        self.completed_stages: Dict[str, WorkflowStageInfo] = {}
        self.failed_stages: Dict[str, WorkflowStageInfo] = {}
        
        # 统计信息
        self.execution_stats = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "avg_execution_time": 0.0,
            "avg_efficiency_score": 0.0,
            "max_parallel_stages": 0,
            "resource_conflicts": 0
        }
        
        # 执行控制
        self._stop_event = threading.Event()
        self._execution_lock = threading.RLock()
        
        logger.info(f"工作流阶段并行执行器初始化完成 (ID: {self.parallelizer_id})")
    
    async def execute_workflow_parallel(self, stages: List[WorkflowStageInfo]) -> ParallelWorkflowResult:
        """
        并行执行工作流阶段
        """
        workflow_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info(f"开始并行执行工作流: {workflow_id}")
            
            # 1. 初始化阶段
            await self._initialize_stages(workflow_id, stages)
            
            # 2. 构建依赖关系
            await self._build_dependency_graph()
            
            # 3. 计算执行顺序
            execution_order = self.dependency_manager.calculate_execution_order()
            logger.debug(f"执行顺序: {execution_order}")
            
            # 4. 并行执行阶段
            stage_results = await self._execute_stages_parallel()
            
            # 5. 计算性能指标
            overall_duration = time.time() - start_time
            efficiency_score = self._calculate_efficiency_score(stage_results, overall_duration)
            resource_utilization = self.resource_allocator.get_resource_utilization()
            bottleneck_analysis = self._analyze_bottlenecks(stage_results)
            quality_metrics = self._calculate_quality_metrics(stage_results)
            
            # 6. 更新统计
            success = all(stage.status == StageStatus.COMPLETED for stage in stage_results.values())
            self._update_execution_stats(success, overall_duration, efficiency_score)
            
            result = ParallelWorkflowResult(
                workflow_id=workflow_id,
                success=success,
                stage_results=stage_results,
                overall_duration=overall_duration,
                efficiency_score=efficiency_score,
                resource_utilization=resource_utilization,
                bottleneck_analysis=bottleneck_analysis,
                quality_metrics=quality_metrics
            )
            
            logger.info(f"工作流并行执行完成: {workflow_id} (耗时: {overall_duration:.2f}s, 效率: {efficiency_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"工作流并行执行失败: {e}")
            return ParallelWorkflowResult(
                workflow_id=workflow_id,
                success=False,
                stage_results={},
                overall_duration=time.time() - start_time,
                efficiency_score=0.0,
                resource_utilization=self.resource_allocator.get_resource_utilization(),
                bottleneck_analysis={},
                quality_metrics={}
            )
    
    async def _initialize_stages(self, workflow_id: str, stages: List[WorkflowStageInfo]):
        """初始化阶段"""
        for i, stage in enumerate(stages):
            stage.stage_id = f"{workflow_id}_stage_{i:02d}_{stage.stage_type.value}"
            stage.status = StageStatus.PENDING
            
            # 设置默认资源需求
            if not stage.resource_requirements:
                stage.resource_requirements = self._get_default_resource_requirements(stage.stage_type)
            
            self.dependency_manager.add_stage(stage)
    
    def _get_default_resource_requirements(self, stage_type: WorkflowStage) -> Dict[str, Any]:
        """获取默认资源需求"""
        requirements = {
            WorkflowStage.INITIALIZATION: {"cpu": 10, "memory": 5, "agents": 1},
            WorkflowStage.ANALYSIS: {"cpu": 20, "memory": 15, "agents": 2},
            WorkflowStage.DESIGN: {"cpu": 25, "memory": 20, "agents": 3},
            WorkflowStage.IMPLEMENTATION: {"cpu": 40, "memory": 30, "agents": 4},
            WorkflowStage.TESTING: {"cpu": 30, "memory": 25, "agents": 3},
            WorkflowStage.DEPLOYMENT: {"cpu": 20, "memory": 15, "agents": 2},
            WorkflowStage.OPTIMIZATION: {"cpu": 35, "memory": 25, "agents": 3},
            WorkflowStage.MONITORING: {"cpu": 15, "memory": 10, "agents": 1}
        }
        return requirements.get(stage_type, {"cpu": 10, "memory": 5, "agents": 1})
    
    async def _build_dependency_graph(self):
        """构建依赖关系图"""
        # 自动添加一些默认依赖关系
        stage_order = [
            WorkflowStage.INITIALIZATION,
            WorkflowStage.ANALYSIS,
            WorkflowStage.DESIGN,
            WorkflowStage.IMPLEMENTATION,
            WorkflowStage.TESTING,
            WorkflowStage.DEPLOYMENT,
            WorkflowStage.OPTIMIZATION,
            WorkflowStage.MONITORING
        ]
        
        stage_type_to_info = {stage.stage_type: stage for stage in self.dependency_manager.stage_graph.values()}
        
        # 添加顺序依赖
        for i in range(1, len(stage_order)):
            current_stage_type = stage_order[i]
            previous_stage_type = stage_order[i-1]
            
            if current_stage_type in stage_type_to_info and previous_stage_type in stage_type_to_info:
                current_stage = stage_type_to_info[current_stage_type]
                previous_stage = stage_type_to_info[previous_stage_type]
                
                if previous_stage.stage_id not in current_stage.dependencies:
                    current_stage.dependencies.append(previous_stage.stage_id)
    
    async def _execute_stages_parallel(self) -> Dict[str, WorkflowStageInfo]:
        """并行执行阶段"""
        completed_stages = set()
        all_results = {}
        
        # 创建阶段执行器
        async def execute_single_stage(stage_id: str) -> Tuple[str, WorkflowStageInfo]:
            """执行单个阶段"""
            try:
                stage = self.dependency_manager.stage_graph[stage_id]
                
                # 等待依赖完成
                while not set(stage.dependencies).issubset(completed_stages):
                    await asyncio.sleep(0.1)
                
                # 分配资源
                while not self.resource_allocator.allocate_resources(stage_id, stage.resource_requirements):
                    await asyncio.sleep(0.1)  # 等待资源释放
                
                try:
                    # 执行阶段
                    await self._execute_stage(stage)
                    
                    # 标记完成
                    completed_stages.add(stage_id)
                    return stage_id, stage
                    
                finally:
                    # 释放资源
                    self.resource_allocator.release_resources(stage_id)
                
            except Exception as e:
                logger.error(f"阶段执行异常: {stage_id} - {e}")
                stage.status = StageStatus.FAILED
                stage.error = str(e)
                return stage_id, stage
        
        # 并行执行所有阶段
        while len(completed_stages) < len(self.dependency_manager.stage_graph):
            ready_stages = self.dependency_manager.get_ready_stages(completed_stages)
            parallelizable_stages = self.dependency_manager.get_parallelizable_stages(ready_stages)
            
            if not parallelizable_stages:
                await asyncio.sleep(0.1)  # 等待资源释放
                continue
            
            # 并行执行可执行的阶段
            tasks = [execute_single_stage(stage_id) for stage_id in parallelizable_stages]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"阶段执行异常: {result}")
                    continue
                stage_id, stage = result
                all_results[stage_id] = stage
        
        return all_results
    
    async def _execute_stage(self, stage: WorkflowStageInfo):
        """执行阶段（模拟）"""
        stage.status = StageStatus.RUNNING
        stage.start_time = time.time()
        
        logger.info(f"开始执行阶段: {stage.stage_name}")
        
        # 模拟阶段执行
        await self._simulate_stage_execution(stage)
        
        # 更新阶段状态
        stage.end_time = time.time()
        stage.duration = stage.end_time - stage.start_time
        stage.status = StageStatus.COMPLETED
        stage.progress = 1.0
        stage.result = f"阶段 {stage.stage_name} 执行完成"
        
        logger.info(f"阶段完成: {stage.stage_name} (耗时: {stage.duration:.2f}s)")
    
    async def _simulate_stage_execution(self, stage: WorkflowStageInfo):
        """模拟阶段执行"""
        # 根据阶段类型和复杂度模拟执行时间
        base_duration = stage.estimated_duration
        complexity_factor = 1.0
        
        # 模拟进度更新
        for i in range(10):
            await asyncio.sleep(base_duration * 0.1)
            stage.progress = (i + 1) / 10.0
            
            # 模拟一些阶段特定的处理
            if stage.stage_type == WorkflowStage.IMPLEMENTATION:
                # 实现阶段可能更复杂
                complexity_factor = 1.5
            elif stage.stage_type == WorkflowStage.TESTING:
                # 测试阶段需要更多时间
                complexity_factor = 1.2
    
    def _calculate_efficiency_score(self, stage_results: Dict[str, WorkflowStageInfo], 
                                  overall_duration: float) -> float:
        """计算效率评分"""
        completed_stages = [s for s in stage_results.values() if s.status == StageStatus.COMPLETED]
        
        if not completed_stages:
            return 0.0
        
        # 计算理想串行时间
        serial_time = sum(s.estimated_duration for s in completed_stages)
        
        # 计算并行效率
        efficiency = serial_time / overall_duration if overall_duration > 0 else 0.0
        
        # 考虑成功率
        success_rate = len(completed_stages) / len(stage_results)
        
        # 综合效率评分
        efficiency_score = efficiency * success_rate
        
        return min(max(efficiency_score, 0.0), 10.0)  # 限制在0-10之间
    
    def _analyze_bottlenecks(self, stage_results: Dict[str, WorkflowStageInfo]) -> Dict[str, Any]:
        """分析瓶颈"""
        durations = [(stage.stage_name, stage.duration or 0) for stage in stage_results.values()]
        durations.sort(key=lambda x: x[1], reverse=True)
        
        # 计算平均执行时间
        avg_duration = sum(d[1] for d in durations) / len(durations) if durations else 0
        
        # 识别瓶颈阶段
        bottlenecks = [d for d in durations if d[1] > avg_duration * 1.5]
        
        return {
            "slowest_stage": durations[0] if durations else None,
            "avg_stage_duration": avg_duration,
            "bottleneck_stages": bottlenecks,
            "max_parallel_efficiency": len([s for s in stage_results.values() if s.parallelizable])
        }
    
    def _calculate_quality_metrics(self, stage_results: Dict[str, WorkflowStageInfo]) -> Dict[str, Any]:
        """计算质量指标"""
        total_stages = len(stage_results)
        completed_stages = len([s for s in stage_results.values() if s.status == StageStatus.COMPLETED])
        failed_stages = len([s for s in stage_results.values() if s.status == StageStatus.FAILED])
        
        return {
            "completion_rate": completed_stages / total_stages if total_stages > 0 else 0,
            "failure_rate": failed_stages / total_stages if total_stages > 0 else 0,
            "avg_stage_quality": 0.85,  # 模拟值
            "resource_optimization_score": 0.9,  # 模拟值
            "parallel_execution_score": 0.92  # 模拟值
        }
    
    def _update_execution_stats(self, success: bool, execution_time: float, efficiency_score: float):
        """更新执行统计"""
        with self._execution_lock:
            self.execution_stats["total_workflows"] += 1
            if success:
                self.execution_stats["successful_workflows"] += 1
            else:
                self.execution_stats["failed_workflows"] += 1
            
            # 更新平均执行时间
            alpha = 0.1
            self.execution_stats["avg_execution_time"] = (
                alpha * execution_time + 
                (1 - alpha) * self.execution_stats["avg_execution_time"]
            )
            
            # 更新平均效率评分
            self.execution_stats["avg_efficiency_score"] = (
                alpha * efficiency_score +
                (1 - alpha) * self.execution_stats["avg_efficiency_score"]
            )
    
    def get_parallelizer_statistics(self) -> Dict[str, Any]:
        """获取并行执行器统计信息"""
        with self._execution_lock:
            resource_utilization = self.resource_allocator.get_resource_utilization()
            
            return {
                "parallelizer_id": self.parallelizer_id,
                "execution_stats": self.execution_stats.copy(),
                "resource_utilization": resource_utilization,
                "active_stages": len(self.active_stages),
                "completed_stages": len(self.completed_stages),
                "failed_stages": len(self.failed_stages),
                "max_concurrent_stages": self.max_concurrent_stages
            }
    
    def stop(self):
        """停止并行执行器"""
        self._stop_event.set()
        logger.info("工作流阶段并行执行器已停止")

# --- 使用示例 ---
async def main():
    """示例使用"""
    # 创建并行执行器
    parallelizer = WorkflowStageParallelizer(max_concurrent_stages=4)
    
    # 定义工作流阶段
    stages = [
        WorkflowStageInfo(
            stage_id="",  # 稍后设置
            stage_type=WorkflowStage.INITIALIZATION,
            stage_name="系统初始化",
            description="初始化开发环境和配置",
            status=StageStatus.PENDING,
            estimated_duration=0.5,
            parallelizable=False
        ),
        WorkflowStageInfo(
            stage_id="",  # 稍后设置
            stage_type=WorkflowStage.ANALYSIS,
            stage_name="需求分析",
            description="分析用户需求和系统需求",
            status=StageStatus.PENDING,
            estimated_duration=2.0,
            parallelizable=True
        ),
        WorkflowStageInfo(
            stage_id="",  # 稍后设置
            stage_type=WorkflowStage.DESIGN,
            stage_name="系统设计",
            description="设计系统架构和数据库",
            status=StageStatus.PENDING,
            estimated_duration=3.0,
            parallelizable=True
        ),
        WorkflowStageInfo(
            stage_id="",  # 稍后设置
            stage_type=WorkflowStage.IMPLEMENTATION,
            stage_name="核心开发",
            description="实现核心功能模块",
            status=StageStatus.PENDING,
            estimated_duration=8.0,
            parallelizable=True
        ),
        WorkflowStageInfo(
            stage_id="",  # 稍后设置
            stage_type=WorkflowStage.TESTING,
            stage_name="测试验证",
            description="编写和执行测试用例",
            status=StageStatus.PENDING,
            estimated_duration=3.0,
            parallelizable=True
        ),
        WorkflowStageInfo(
            stage_id="",  # 稍后设置
            stage_type=WorkflowStage.DEPLOYMENT,
            stage_name="部署上线",
            description="部署到生产环境",
            status=StageStatus.PENDING,
            estimated_duration=1.0,
            parallelizable=False
        )
    ]
    
    # 执行并行工作流
    result = await parallelizer.execute_workflow_parallel(stages)
    
    print(f"工作流执行结果: {result.success}")
    print(f"总体执行时间: {result.overall_duration:.2f}s")
    print(f"效率评分: {result.efficiency_score:.2f}")
    print(f"资源使用: {result.resource_utilization}")
    print(f"瓶颈分析: {result.bottleneck_analysis}")
    
    # 获取统计信息
    stats = parallelizer.get_parallelizer_statistics()
    print(f"\n并行执行器统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    asyncio.run(main())