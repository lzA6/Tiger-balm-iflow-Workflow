#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 状态转换引擎 vΩ - State Transition Engine
State Transition Engine vΩ - 智能工作流状态转换管理系统

实现复杂的状态转换逻辑、条件验证、优先级管理和错误恢复，
确保工作流状态的平滑转换和系统的稳定性。
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from collections import defaultdict, deque
import threading
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StateStatus(Enum):
    """状态状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    COMPLETED = "completed"
    PENDING = "pending"

class TransitionType(Enum):
    """转换类型枚举"""
    NORMAL = "normal"
    ERROR_RECOVERY = "error_recovery"
    ROLLBACK = "rollback"
    OPTIMIZATION = "optimization"
    EVOLUTION = "evolution"

@dataclass
class State:
    """工作流状态"""
    id: str
    name: str
    description: str
    status: StateStatus = StateStatus.INACTIVE
    entry_conditions: Set[str] = field(default_factory=set)
    exit_conditions: Set[str] = field(default_factory=set)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Transition:
    """状态转换"""
    id: str
    from_state: str
    to_state: str
    conditions: List[str] = field(default_factory=list)
    priority: int = 1
    transition_type: TransitionType = TransitionType.NORMAL
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TransitionEvent:
    """转换事件"""
    id: str
    transition_id: str
    timestamp: datetime
    success: bool
    duration: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class StateTransitionEngine:
    """状态转换引擎"""
    
    def __init__(self):
        self.states: Dict[str, State] = {}
        self.transitions: Dict[str, Transition] = {}
        self.state_graph = nx.DiGraph()
        self.current_state: Optional[str] = None
        self.transition_history: deque = deque(maxlen=1000)
        self.condition_evaluators: Dict[str, Callable] = {}
        self.action_executors: Dict[str, Callable] = {}
        self.state_listeners: Dict[str, List[Callable]] = defaultdict(list)
        self.transition_listeners: List[Callable] = []
        self.error_handlers: List[Callable] = []
        self.is_running = False
        self.transition_lock = threading.Lock()
        
    def register_state(self, state: State) -> bool:
        """注册状态"""
        if state.id in self.states:
            logger.warning(f"State {state.id} already exists")
            return False
        
        self.states[state.id] = state
        self.state_graph.add_node(state.id, state=state)
        logger.info(f"State {state.id} registered")
        return True
    
    def register_transition(self, transition: Transition) -> bool:
        """注册转换"""
        if transition.id in self.transitions:
            logger.warning(f"Transition {transition.id} already exists")
            return False
        
        if transition.from_state not in self.states or transition.to_state not in self.states:
            logger.error(f"Invalid transition: {transition.from_state} -> {transition.to_state}")
            return False
        
        self.transitions[transition.id] = transition
        self.state_graph.add_edge(transition.from_state, transition.to_state, transition=transition)
        logger.info(f"Transition {transition.id} registered: {transition.from_state} -> {transition.to_state}")
        return True
    
    def register_condition_evaluator(self, condition: str, evaluator: Callable):
        """注册条件评估器"""
        self.condition_evaluators[condition] = evaluator
        logger.debug(f"Condition evaluator registered: {condition}")
    
    def register_action_executor(self, action: str, executor: Callable):
        """注册动作执行器"""
        self.action_executors[action] = executor
        logger.debug(f"Action executor registered: {action}")
    
    def add_state_listener(self, state_id: str, listener: Callable):
        """添加状态监听器"""
        self.state_listeners[state_id].append(listener)
    
    def add_transition_listener(self, listener: Callable):
        """添加转换监听器"""
        self.transition_listeners.append(listener)
    
    def add_error_handler(self, handler: Callable):
        """添加错误处理器"""
        self.error_handlers.append(handler)
    
    async def start(self, initial_state: str) -> bool:
        """启动状态机"""
        if initial_state not in self.states:
            logger.error(f"Initial state {initial_state} not found")
            return False
        
        self.current_state = initial_state
        self.is_running = True
        
        # 进入初始状态
        success = await self._enter_state(initial_state)
        if success:
            logger.info(f"State machine started in state: {initial_state}")
            return True
        else:
            logger.error(f"Failed to enter initial state: {initial_state}")
            return False
    
    async def _enter_state(self, state_id: str) -> bool:
        """进入状态"""
        if state_id not in self.states:
            logger.error(f"State {state_id} not found")
            return False
        
        state = self.states[state_id]
        
        # 检查进入条件
        if not await self._evaluate_conditions(state.entry_conditions):
            logger.error(f"Entry conditions not met for state: {state_id}")
            return False
        
        # 更新状态
        state.status = StateStatus.ACTIVE
        state.started_at = datetime.now()
        state.retry_count = 0
        
        # 通知监听器
        await self._notify_state_listeners(state_id, "enter")
        
        # 执行状态动作
        success = await self._execute_state_actions(state)
        
        if success:
            logger.info(f"Successfully entered state: {state_id}")
        else:
            state.status = StateStatus.ERROR
            logger.error(f"Failed to execute actions for state: {state_id}")
        
        return success
    
    async def _execute_state_actions(self, state: State) -> bool:
        """执行状态动作"""
        for action in state.actions:
            action_name = action.get("name")
            if not action_name:
                continue
            
            if action_name not in self.action_executors:
                logger.warning(f"No executor found for action: {action_name}")
                continue
            
            try:
                executor = self.action_executors[action_name]
                await executor(action)
                logger.debug(f"Action executed successfully: {action_name}")
            except Exception as e:
                logger.error(f"Action execution failed: {action_name}, error: {e}")
                
                # 调用错误处理器
                await self._handle_error(f"action_execution_failed", {
                    "action": action_name,
                    "state": state.id,
                    "error": str(e)
                })
                
                return False
        
        return True
    
    async def evaluate_transitions(self) -> Optional[str]:
        """评估状态转换"""
        if not self.current_state or self.current_state not in self.states:
            return None
        
        current_state = self.states[self.current_state]
        
        # 获取所有可能的转换
        possible_transitions = []
        for transition_id, transition in self.transitions.items():
            if transition.from_state == self.current_state and transition.enabled:
                priority = transition.priority
                conditions_met = await self._evaluate_conditions(transition.conditions)
                
                if conditions_met:
                    possible_transitions.append((priority, transition_id, transition))
        
        if not possible_transitions:
            return None
        
        # 按优先级排序，选择最高优先级的转换
        possible_transitions.sort(key=lambda x: x[0], reverse=True)
        _, best_transition_id, best_transition = possible_transitions[0]
        
        return best_transition_id
    
    async def transition_to(self, transition_id: str) -> bool:
        """执行状态转换"""
        if transition_id not in self.transitions:
            logger.error(f"Transition {transition_id} not found")
            return False
        
        transition = self.transitions[transition_id]
        
        if transition.from_state != self.current_state:
            logger.error(f"Transition {transition_id} not applicable to current state")
            return False
        
        # 检查转换条件
        if not await self._evaluate_conditions(transition.conditions):
            logger.error(f"Transition conditions not met: {transition_id}")
            return False
        
        start_time = time.time()
        
        try:
            # 退出当前状态
            await self._exit_state(self.current_state)
            
            # 进入新状态
            success = await self._enter_state(transition.to_state)
            
            if success:
                # 更新当前状态
                old_state = self.current_state
                self.current_state = transition.to_state
                
                # 记录转换事件
                duration = time.time() - start_time
                event = TransitionEvent(
                    id=f"event_{datetime.now().timestamp()}",
                    transition_id=transition_id,
                    timestamp=datetime.now(),
                    success=True,
                    duration=duration
                )
                self.transition_history.append(event)
                
                # 通知监听器
                await self._notify_transition_listeners(old_state, transition.to_state, transition)
                
                logger.info(f"Transition completed: {old_state} -> {transition.to_state}")
                return True
            else:
                # 转换失败，尝试回滚
                await self._handle_transition_failure(transition_id, "state_entry_failed")
                return False
                
        except Exception as e:
            # 转换异常，尝试回滚
            await self._handle_transition_failure(transition_id, str(e))
            return False
    
    async def _exit_state(self, state_id: str):
        """退出状态"""
        if state_id not in self.states:
            return
        
        state = self.states[state_id]
        
        # 检查退出条件
        if not await self._evaluate_conditions(state.exit_conditions):
            logger.warning(f"Exit conditions not met for state: {state_id}")
        
        # 更新状态
        state.status = StateStatus.COMPLETED
        state.completed_at = datetime.now()
        
        # 通知监听器
        await self._notify_state_listeners(state_id, "exit")
        
        logger.debug(f"Exited state: {state_id}")
    
    async def _evaluate_conditions(self, conditions: List[str]) -> bool:
        """评估条件列表"""
        for condition in conditions:
            if condition not in self.condition_evaluators:
                logger.warning(f"No evaluator found for condition: {condition}")
                continue
            
            try:
                evaluator = self.condition_evaluators[condition]
                result = await evaluator()
                
                if not result:
                    logger.debug(f"Condition not met: {condition}")
                    return False
                    
            except Exception as e:
                logger.error(f"Condition evaluation failed: {condition}, error: {e}")
                return False
        
        return True
    
    async def _notify_state_listeners(self, state_id: str, event: str):
        """通知状态监听器"""
        for listener in self.state_listeners[state_id]:
            try:
                await listener(state_id, event)
            except Exception as e:
                logger.error(f"State listener error: {e}")
    
    async def _notify_transition_listeners(self, from_state: str, to_state: str, transition: Transition):
        """通知转换监听器"""
        for listener in self.transition_listeners:
            try:
                await listener(from_state, to_state, transition)
            except Exception as e:
                logger.error(f"Transition listener error: {e}")
    
    async def _handle_error(self, error_type: str, context: Dict[str, Any]):
        """处理错误"""
        for handler in self.error_handlers:
            try:
                await handler(error_type, context)
            except Exception as e:
                logger.error(f"Error handler failed: {e}")
    
    async def _handle_transition_failure(self, transition_id: str, error_message: str):
        """处理转换失败"""
        # 记录失败事件
        event = TransitionEvent(
            id=f"event_{datetime.now().timestamp()}",
            transition_id=transition_id,
            timestamp=datetime.now(),
            success=False,
            duration=0,
            error_message=error_message
        )
        self.transition_history.append(event)
        
        # 调用错误处理器
        await self._handle_error("transition_failure", {
            "transition_id": transition_id,
            "error_message": error_message,
            "current_state": self.current_state
        })
        
        # 尝试错误恢复转换
        await self._attempt_error_recovery()
    
    async def _attempt_error_recovery(self):
        """尝试错误恢复"""
        # 查找错误恢复转换
        for transition_id, transition in self.transitions.items():
            if (transition.from_state == self.current_state and 
                transition.transition_type == TransitionType.ERROR_RECOVERY and
                transition.enabled):
                
                if await self._evaluate_conditions(transition.conditions):
                    logger.info(f"Attempting error recovery transition: {transition_id}")
                    await self.transition_to(transition_id)
                    return
        
        logger.warning("No error recovery transition found")
    
    async def run_state_machine(self):
        """运行状态机主循环"""
        while self.is_running:
            try:
                # 评估转换
                transition_id = await self.evaluate_transitions()
                
                if transition_id:
                    # 执行转换
                    await self.transition_to(transition_id)
                
                # 等待一段时间再检查
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"State machine loop error: {e}")
                await self._handle_error("state_machine_error", {"error": str(e)})
                await asyncio.sleep(1)
    
    def get_current_state(self) -> Optional[State]:
        """获取当前状态"""
        if self.current_state:
            return self.states.get(self.current_state)
        return None
    
    def get_state_graph(self) -> Dict[str, Any]:
        """获取状态图信息"""
        return {
            "nodes": list(self.state_graph.nodes()),
            "edges": list(self.state_graph.edges()),
            "current_state": self.current_state,
            "node_count": self.state_graph.number_of_nodes(),
            "edge_count": self.state_graph.number_of_edges()
        }
    
    def get_transition_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取转换历史"""
        history = list(self.transition_history)
        history.reverse()
        return [
            {
                "id": event.id,
                "transition_id": event.transition_id,
                "timestamp": event.timestamp.isoformat(),
                "success": event.success,
                "duration": event.duration,
                "error_message": event.error_message
            }
            for event in history[:limit]
        ]
    
    def get_state_metrics(self) -> Dict[str, Any]:
        """获取状态指标"""
        state_stats = defaultdict(int)
        transition_stats = defaultdict(int)
        
        for state in self.states.values():
            state_stats[state.status.value] += 1
        
        for transition in self.transitions.values():
            transition_stats[transition.transition_type.value] += 1
        
        return {
            "total_states": len(self.states),
            "total_transitions": len(self.transitions),
            "state_distribution": dict(state_stats),
            "transition_distribution": dict(transition_stats),
            "current_state": self.current_state,
            "is_running": self.is_running
        }
    
    async def stop(self):
        """停止状态机"""
        self.is_running = False
        logger.info("State transition engine stopped")

# 默认条件评估器
async def default_condition_evaluator(condition: str) -> bool:
    """默认条件评估器"""
    # 这里可以实现默认的条件评估逻辑
    # 简化实现，总是返回True
    return True

# 默认动作执行器
async def default_action_executor(action: Dict[str, Any]) -> bool:
    """默认动作执行器"""
    # 这里可以实现默认的动作执行逻辑
    # 简化实现，总是返回True
    return True

# 全局状态转换引擎实例
_state_engine = None

async def get_state_engine() -> StateTransitionEngine:
    """获取状态转换引擎实例"""
    global _state_engine
    if _state_engine is None:
        _state_engine = StateTransitionEngine()
        
        # 注册默认评估器和执行器
        _state_engine.register_condition_evaluator("default", default_condition_evaluator)
        _state_engine.register_action_executor("default", default_action_executor)
        
    return _state_engine

if __name__ == "__main__":
    async def test_state_engine():
        """测试状态转换引擎"""
        engine = await get_state_engine()
        
        # 注册测试状态
        initial_state = State(
            id="initial",
            name="Initial State",
            description="Initial state for testing"
        )
        
        processing_state = State(
            id="processing",
            name="Processing State",
            description="Processing state for testing"
        )
        
        completed_state = State(
            id="completed",
            name="Completed State",
            description="Completed state for testing"
        )
        
        engine.register_state(initial_state)
        engine.register_state(processing_state)
        engine.register_state(completed_state)
        
        # 注册测试转换
        transition1 = Transition(
            id="init_to_process",
            from_state="initial",
            to_state="processing",
            conditions=["default"],
            priority=1
        )
        
        transition2 = Transition(
            id="process_to_complete",
            from_state="processing",
            to_state="completed",
            conditions=["default"],
            priority=1
        )
        
        engine.register_transition(transition1)
        engine.register_transition(transition2)
        
        # 启动状态机
        await engine.start("initial")
        
        # 运行几轮转换
        for _ in range(5):
            transition_id = await engine.evaluate_transitions()
            if transition_id:
                await engine.transition_to(transition_id)
            await asyncio.sleep(0.5)
        
        # 获取指标
        metrics = engine.get_state_metrics()
        print(f"State metrics: {metrics}")
        
        # 停止状态机
        await engine.stop()
    
    asyncio.run(test_state_engine())