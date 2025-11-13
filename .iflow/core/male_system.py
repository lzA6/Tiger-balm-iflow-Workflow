import time
from typing import Dict, List, Any, Callable

class RecursiveMetaLearningEngine:
    """
    递归元学习引擎 (Recursive Meta-Learning Engine) - v2.0
    实现系统级的自我诊断、自我修复、自我优化和递归学习。
    融合了更强大的性能监控和进化策略。
    你一定要超级思考、极限思考、深度思考、全力思考、超强思考、认真仔细思考（ultrathink、think really super hard、think intensely）。
    """

    def __init__(self):
        self.performance_history: List[Dict[str, Any]] = []
        self.evolution_strategies: Dict[str, Any] = {"current_strategy": "balanced"}
        self.performance_monitor = PerformanceMonitor()
        self.evolution_interval_seconds = 3600 # 每小时进化一次
        self.last_evolution_time = time.time()

    def learn_from_execution(self, execution_log: Dict[str, Any]):
        """从单次执行日志中学习，并记录性能。"""
        self.performance_history.append(execution_log)
        self.performance_monitor.record(execution_log)
        
        if time.time() - self.last_evolution_time > self.evolution_interval_seconds:
            self._analyze_and_evolve()
            self.last_evolution_time = time.time()

    def _analyze_and_evolve(self):
        """分析历史性能并进化策略。"""
        if len(self.performance_history) < 20: # 需要更多样本
            return

        metrics = self.performance_monitor.get_metrics()
        print(f"MALE: 开始进化分析，当前性能指标: {metrics}")

        # 进化决策
        if metrics.get("error_rate", 0) > 0.15:
            self.evolution_strategies["current_strategy"] = "stability_focused"
            print("MALE: 错误率偏高，切换到【稳定优先】策略。")
        elif metrics.get("avg_latency_ms", 0) > 500:
            self.evolution_strategies["current_strategy"] = "speed_focused"
            print("MALE: 平均延迟较高，切换到【速度优先】策略。")
        elif metrics.get("success_rate", 1) > 0.98:
            self.evolution_strategies["current_strategy"] = "exploration_focused"
            print("MALE: 系统表现卓越，切换到【探索优先】策略，尝试新工具和方法。")
        else:
            self.evolution_strategies["current_strategy"] = "balanced"
            print("MALE: 系统表现均衡，维持【平衡】策略。")
        
        # 清理旧历史记录
        self.performance_history = self.performance_history[-200:]
        print("MALE: 进化分析完成。")

class PerformanceMonitor:
    """性能监控器"""
    def __init__(self, retention_period_seconds: int = 86400):
        self.records: List[Dict[str, Any]] = []
        self.retention_period = retention_period_seconds

    def record(self, log: Dict[str, Any]):
        log['timestamp'] = time.time()
        self.records.append(log)
        self._cleanup()

    def get_metrics(self) -> Dict[str, float]:
        if not self.records:
            return {}
        
        total_tasks = len(self.records)
        successful_tasks = sum(1 for r in self.records if r.get("status") == "success")
        error_tasks = total_tasks - successful_tasks
        
        latencies = [r.get('latency_ms', 0) for r in self.records if 'latency_ms' in r]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        return {
            "total_tasks": float(total_tasks),
            "success_rate": float(successful_tasks / total_tasks) if total_tasks > 0 else 1.0,
            "error_rate": float(error_tasks / total_tasks) if total_tasks > 0 else 0.0,
            "avg_latency_ms": float(avg_latency),
        }

    def _cleanup(self):
        """清理过期记录"""
        current_time = time.time()
        self.records = [r for r in self.records if current_time - r['timestamp'] < self.retention_period]