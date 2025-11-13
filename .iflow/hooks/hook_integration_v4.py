#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌌 Hook集成系统 V4 (Hook Integration System V4)
将Hook系统深度集成到工作流引擎中，实现全自动的质量保障和代码审查。
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import threading
import watchdog.observers
import watchdog.events

# 动态添加项目根目录到sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from iflow.hooks.comprehensive_hook_manager_v4 import ComprehensiveHookManagerV4, HookType

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class IntegrationConfig:
    """集成配置"""
    monitor_file_changes: bool = True
    auto_trigger_hooks: bool = True
    monitor_directories: List[str] = None
    exclude_patterns: List[str] = None
    debounce_interval: float = 1.0  # 防抖间隔（秒）
    
    def __post_init__(self):
        if self.monitor_directories is None:
            self.monitor_directories = ["A项目/iflow"]
        if self.exclude_patterns is None:
            self.exclude_patterns = [
                "__pycache__",
                ".git",
                "node_modules",
                ".pytest_cache",
                "*.pyc"
            ]

class FileChangeHandler(watchdog.events.FileSystemEventHandler):
    """文件变更处理器"""
    
    def __init__(self, hook_manager: ComprehensiveHookManagerV4, config: IntegrationConfig):
        self.hook_manager = hook_manager
        self.config = config
        self.last_trigger = {}
        self.lock = threading.Lock()
        
    def on_modified(self, event):
        """文件修改事件"""
        if event.is_directory:
            return
            
        self._handle_file_change(event.src_path, "modified")
    
    def on_created(self, event):
        """文件创建事件"""
        if event.is_directory:
            return
            
        self._handle_file_change(event.src_path, "created")
    
    def _handle_file_change(self, file_path: str, change_type: str):
        """处理文件变更"""
        # 检查排除模式
        for pattern in self.config.exclude_patterns:
            if pattern in file_path:
                return
        
        # 防抖处理
        now = time.time()
        with self.lock:
            if file_path in self.last_trigger:
                if now - self.last_trigger[file_path] < self.config.debounce_interval:
                    return
            self.last_trigger[file_path] = now
        
        # 触发Hook
        logger.info(f"📝 检测到文件变更: {file_path} ({change_type})")
        
        # 异步触发Hook
        asyncio.create_task(
            self.hook_manager.trigger_code_change_hooks(file_path, change_type)
        )

class HookIntegrationSystemV4:
    """Hook集成系统 V4"""
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()
        self.hook_manager = ComprehensiveHookManagerV4()
        self.file_observer = None
        self.is_monitoring = False
        self.lock = threading.RLock()
        
        logger.info("🚀 Hook集成系统 V4 初始化中...")
    
    async def initialize(self):
        """初始化集成系统"""
        with self.lock:
            if self.is_monitoring:
                return
            
            # 初始化Hook管理器
            await self.hook_manager.initialize()
            
            # 设置文件监控
            if self.config.monitor_file_changes:
                await self._setup_file_monitoring()
            
            self.is_monitoring = True
            logger.info("✅ Hook集成系统 V4 初始化完成")
    
    async def _setup_file_monitoring(self):
        """设置文件监控"""
        self.file_observer = watchdog.observers.Observer()
        
        for directory in self.config.monitor_directories:
            dir_path = Path(directory)
            if dir_path.exists():
                event_handler = FileChangeHandler(self.hook_manager, self.config)
                self.file_observer.schedule(
                    event_handler,
                    str(dir_path),
                    recursive=True
                )
                logger.info(f"📁 监控目录: {dir_path}")
        
        self.file_observer.start()
        logger.info("👀 文件监控已启动")
    
    async def shutdown(self):
        """关闭集成系统"""
        with self.lock:
            if not self.is_monitoring:
                return
            
            if self.file_observer:
                self.file_observer.stop()
                self.file_observer.join()
                self.file_observer = None
            
            self.is_monitoring = False
            logger.info("🔌 Hook集成系统已关闭")
    
    async def trigger_manual_hooks(self, hook_type: HookType, context: Dict[str, Any] = None):
        """手动触发Hooks"""
        logger.info(f"🔧 手动触发 {hook_type.value} Hooks")
        results = await self.hook_manager.execute_hooks_by_type(hook_type, context)
        
        # 记录结果
        success_count = sum(1 for r in results if r.success)
        logger.info(f"✅ 执行完成: {success_count}/{len(results)} 成功")
        
        return results
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """获取集成状态"""
        hook_stats = self.hook_manager.get_hook_statistics()
        
        return {
            "is_monitoring": self.is_monitoring,
            "monitoring_directories": self.config.monitor_directories,
            "file_observer_active": self.file_observer.is_alive() if self.file_observer else False,
            "hook_statistics": hook_stats,
            "recent_executions": self.hook_manager.get_recent_executions(10)
        }
    
    async def run_health_check(self) -> Dict[str, Any]:
        """运行健康检查"""
        health_status = {
            "overall_health": "healthy",
            "checks": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # 检查Hook管理器状态
        hook_stats = self.hook_manager.get_hook_statistics()
        if hook_stats["total_hooks"] == 0:
            health_status["checks"]["hooks"] = {"status": "warning", "message": "没有注册的Hooks"}
            health_status["overall_health"] = "warning"
        else:
            success_rate = hook_stats["success_rate"]
            if success_rate < 0.8:
                health_status["checks"]["hooks"] = {
                    "status": "error", 
                    "message": f"Hook成功率过低: {success_rate:.2%}"
                }
                health_status["overall_health"] = "error"
            else:
                health_status["checks"]["hooks"] = {
                    "status": "healthy",
                    "message": f"Hook运行正常，成功率: {success_rate:.2%}"
                }
        
        # 检查文件监控状态
        if self.config.monitor_file_changes:
            if self.file_observer and self.file_observer.is_alive():
                health_status["checks"]["file_monitoring"] = {
                    "status": "healthy",
                    "message": "文件监控正常运行"
                }
            else:
                health_status["checks"]["file_monitoring"] = {
                    "status": "error",
                    "message": "文件监控未运行"
                }
                health_status["overall_health"] = "error"
        
        # 检查磁盘空间
        try:
            import shutil
            total, used, free = shutil.disk_usage(project_root)
            free_percent = free / total
            if free_percent < 0.1:
                health_status["checks"]["disk_space"] = {
                    "status": "error",
                    "message": f"磁盘空间不足: {free_percent:.2%}"
                }
                health_status["overall_health"] = "error"
            else:
                health_status["checks"]["disk_space"] = {
                    "status": "healthy",
                    "message": f"磁盘空间充足: {free_percent:.2%}"
                }
        except Exception as e:
            health_status["checks"]["disk_space"] = {
                "status": "warning",
                "message": f"无法检查磁盘空间: {str(e)}"
            }
        
        return health_status
    
    async def generate_integration_report(self) -> Dict[str, Any]:
        """生成集成报告"""
        # 获取Hook统计
        hook_stats = self.hook_manager.get_hook_statistics()
        
        # 获取最近执行记录
        recent_executions = self.hook_manager.get_recent_executions(100)
        
        # 分析执行趋势
        execution_trend = {}
        if recent_executions:
            # 按小时统计
            hourly_counts = {}
            for execution in recent_executions:
                hour = datetime.fromisoformat(execution["timestamp"]).hour
                hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
            
            execution_trend["hourly_distribution"] = hourly_counts
            
            # 计算平均执行时间
            avg_duration = sum(e["duration"] for e in recent_executions) / len(recent_executions)
            execution_trend["average_duration"] = avg_duration
            
            # 计算成功率
            success_count = sum(1 for e in recent_executions if e["success"])
            execution_trend["recent_success_rate"] = success_count / len(recent_executions)
        
        # 生成建议
        recommendations = []
        
        if hook_stats["success_rate"] < 0.9:
            recommendations.append("Hook成功率较低，建议检查失败的Hook并修复问题")
        
        if hook_stats["average_duration"] > 10:
            recommendations.append("Hook平均执行时间较长，建议优化性能或启用并行执行")
        
        if execution_trend.get("recent_success_rate", 1.0) < 0.8:
            recommendations.append("最近的Hook执行成功率下降，建议查看最近的错误日志")
        
        return {
            "report_timestamp": datetime.now().isoformat(),
            "hook_statistics": hook_stats,
            "execution_trend": execution_trend,
            "integration_status": await self.get_integration_status(),
            "health_check": await self.run_health_check(),
            "recommendations": recommendations
        }

# --- 示例使用 ---
async def main():
    """测试Hook集成系统"""
    integration = HookIntegrationSystemV4()
    await integration.initialize()
    
    try:
        # 显示集成状态
        print("\n📊 集成状态:")
        status = await integration.get_integration_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))
        
        # 运行健康检查
        print("\n🏥 健康检查:")
        health = await integration.run_health_check()
        print(json.dumps(health, indent=2, ensure_ascii=False))
        
        # 手动触发Hook
        print("\n🔧 手动触发预提交Hook:")
        results = await integration.trigger_manual_hooks(
            HookType.PRE_COMMIT,
            {"file_path": "A项目/iflow/core/ultimate_workflow_engine_v4.py"}
        )
        for result in results:
            print(f"  - {result.hook_id}: {result.message}")
        
        # 生成报告
        print("\n📋 生成集成报告...")
        report = await integration.generate_integration_report()
        
        # 保存报告
        report_path = project_root / "A项目" / "iflow" / "reports" / "hook_integration_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"📄 报告已保存到: {report_path}")
        
        # 等待一段时间观察文件监控
        print("\n⏳ 监控文件变更中... (按Ctrl+C停止)")
        while True:
            await asyncio.sleep(10)
            
    except KeyboardInterrupt:
        print("\n👋 停止监控...")
    finally:
        await integration.shutdown()

if __name__ == "__main__":
    asyncio.run(main())