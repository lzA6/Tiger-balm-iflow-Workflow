#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
错误处理和用户体验系统
Error Handling and User Experience System

作者: Quantum AI Team
版本: 5.2.0
日期: 2025-11-12
"""

import asyncio
import time
import traceback
import logging
import json
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import sys
from functools import wraps
import threading
from collections import defaultdict, deque

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """错误类别"""
    SYSTEM = "system"
    NETWORK = "network"
    FILE_IO = "file_io"
    VALIDATION = "validation"
    PERMISSION = "permission"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    CONFIGURATION = "configuration"
    USER_INPUT = "user_input"
    UNKNOWN = "unknown"

@dataclass
class ErrorInfo:
    """错误信息"""
    error_id: str
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    details: Optional[str]
    stack_trace: Optional[str]
    context: Dict[str, Any]
    suggestions: List[str]
    recovery_actions: List[str]
    user_friendly_message: str

@dataclass
class UserFeedback:
    """用户反馈"""
    feedback_id: str
    timestamp: float
    error_id: str
    rating: int  # 1-5
    comment: Optional[str]
    helpful: bool
    resolved: bool

class UserExperienceManager:
    """用户体验管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化用户体验管理器"""
        self.config_path = config_path or "ux_config.json"
        self.error_history = deque(maxlen=1000)
        self.feedback_history = deque(maxlen=500)
        self.error_patterns = defaultdict(list)
        self.suggestions_cache = {}
        self.recovery_strategies = {}
        
        # 配置
        self.config = {
            "enable_friendly_messages": True,
            "enable_suggestions": True,
            "enable_recovery_actions": True,
            "max_error_display_length": 200,
            "auto_error_analysis": True,
            "collect_feedback": True
        }
        
        # 错误消息模板
        self.error_templates = {
            ErrorCategory.SYSTEM: [
                "系统遇到了一些问题，请稍后再试。",
                "系统正在处理您的请求，请耐心等待。",
                "系统资源可能不足，建议关闭其他应用程序。"
            ],
            ErrorCategory.NETWORK: [
                "网络连接似乎有问题，请检查您的网络设置。",
                "无法连接到服务器，请稍后再试。",
                "网络响应较慢，请耐心等待或检查网络连接。"
            ],
            ErrorCategory.FILE_IO: [
                "文件操作失败，请检查文件权限和路径。",
                "无法读取或写入文件，请确保文件未被其他程序占用。",
                "磁盘空间可能不足，请清理磁盘后重试。"
            ],
            ErrorCategory.VALIDATION: [
                "输入的数据格式不正确，请检查后重试。",
                "参数验证失败，请确保所有必填项都已填写。",
                "数据格式错误，请按照要求提供正确的格式。"
            ],
            ErrorCategory.PERMISSION: [
                "您没有执行此操作的权限，请联系管理员。",
                "访问被拒绝，请检查您的权限设置。",
                "操作需要更高的权限级别，请以管理员身份运行。"
            ],
            ErrorCategory.RESOURCE: [
                "系统资源不足，请稍后再试。",
                "内存使用过高，建议关闭其他应用程序。",
                "CPU使用率过高，请等待系统负载降低。"
            ],
            ErrorCategory.TIMEOUT: [
                "操作超时，请稍后再试。",
                "服务器响应时间过长，请检查网络连接。",
                "处理时间过长，建议简化您的请求。"
            ],
            ErrorCategory.CONFIGURATION: [
                "配置文件有错误，请检查配置设置。",
                "配置项缺失或无效，请更新配置文件。",
                "配置版本不兼容，请更新到最新版本。"
            ],
            ErrorCategory.USER_INPUT: [
                "输入的内容不正确，请重新输入。",
                "命令格式错误，请参考帮助文档。",
                "参数值无效，请使用有效的参数值。"
            ],
            ErrorCategory.UNKNOWN: [
                "发生了未知错误，请联系技术支持。",
                "系统遇到了意外问题，请稍后再试。",
                "操作无法完成，请检查输入后重试。"
            ]
        }
        
        # 恢复策略
        self.recovery_strategies = {
            ErrorCategory.SYSTEM: [
                "重启相关服务",
                "检查系统日志",
                "清理系统缓存",
                "重置系统配置"
            ],
            ErrorCategory.NETWORK: [
                "检查网络连接",
                "刷新DNS缓存",
                "重置网络适配器",
                "更换网络环境"
            ],
            ErrorCategory.FILE_IO: [
                "检查文件权限",
                "验证文件路径",
                "关闭占用程序",
                "检查磁盘空间"
            ],
            ErrorCategory.VALIDATION: [
                "检查输入格式",
                "参考示例格式",
                "验证必填项",
                "使用默认值"
            ],
            ErrorCategory.PERMISSION: [
                "以管理员身份运行",
                "修改文件权限",
                "联系系统管理员",
                "使用sudo命令"
            ],
            ErrorCategory.RESOURCE: [
                "关闭其他程序",
                "增加虚拟内存",
                "升级硬件配置",
                "优化系统设置"
            ],
            ErrorCategory.TIMEOUT: [
                "增加超时时间",
                "简化请求内容",
                "分批处理数据",
                "重试操作"
            ],
            ErrorCategory.CONFIGURATION: [
                "检查配置文件",
                "重置为默认配置",
                "更新配置版本",
                "验证配置语法"
            ],
            ErrorCategory.USER_INPUT: [
                "查看帮助文档",
                "检查命令语法",
                "使用示例格式",
                "验证参数值"
            ],
            ErrorCategory.UNKNOWN: [
                "查看详细日志",
                "联系技术支持",
                "重启应用程序",
                "报告错误信息"
            ]
        }
        
        # 加载配置
        self._load_configuration()
        
        # 初始化错误分析
        self._initialize_error_analysis()
        
        logger.info("🎨 用户体验管理器初始化完成")
    
    def _load_configuration(self):
        """加载配置"""
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
                logger.info("📄 UX配置加载完成")
            except Exception as e:
                logger.error(f"❌ 加载UX配置失败: {e}")
    
    def _initialize_error_analysis(self):
        """初始化错误分析"""
        # 预定义错误模式和建议
        error_patterns = {
            "permission_denied": {
                "category": ErrorCategory.PERMISSION,
                "suggestions": [
                    "以管理员身份运行程序",
                    "检查文件/目录权限",
                    "使用sudo命令提升权限"
                ],
                "recovery_actions": [
                    "chmod +x 文件名",
                    "sudo chown 用户:组 文件名",
                    "以管理员身份运行"
                ]
            },
            "file_not_found": {
                "category": ErrorCategory.FILE_IO,
                "suggestions": [
                    "检查文件路径是否正确",
                    "确认文件是否存在",
                    "使用绝对路径"
                ],
                "recovery_actions": [
                    "ls -la 检查文件",
                    "pwd 确认当前目录",
                    "find / -name 文件名"
                ]
            },
            "connection_refused": {
                "category": ErrorCategory.NETWORK,
                "suggestions": [
                    "检查服务是否运行",
                    "验证端口是否正确",
                    "检查防火墙设置"
                ],
                "recovery_actions": [
                    "systemctl status 服务名",
                    "netstat -tlnp | grep 端口",
                    "telnet 主机 端口"
                ]
            },
            "timeout": {
                "category": ErrorCategory.TIMEOUT,
                "suggestions": [
                    "增加超时时间",
                    "检查网络连接",
                    "简化请求内容"
                ],
                "recovery_actions": [
                    "ping 目标主机",
                    "traceroute 目标主机",
                    "调整超时参数"
                ]
            },
            "memory_error": {
                "category": ErrorCategory.RESOURCE,
                "suggestions": [
                    "关闭其他应用程序",
                    "增加虚拟内存",
                    "优化代码内存使用"
                ],
                "recovery_actions": [
                    "free -h 检查内存",
                    "top 查看进程",
                    "kill -9 进程ID"
                ]
            }
        }
        
        self.error_patterns.update(error_patterns)
        logger.info(f"🔧 初始化了 {len(error_patterns)} 个错误模式")
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """处理错误并生成用户友好的错误信息"""
        error_id = self._generate_error_id()
        timestamp = time.time()
        
        # 分析错误
        error_info = self._analyze_error(error, error_id, timestamp, context or {})
        
        # 存储错误历史
        self.error_history.append(error_info)
        
        # 更新错误模式
        self._update_error_patterns(error_info)
        
        # 记录日志
        self._log_error(error_info)
        
        return error_info
    
    def _generate_error_id(self) -> str:
        """生成错误ID"""
        return f"ERR_{int(time.time() * 1000)}_{hash(str(time.time())) % 10000:04d}"
    
    def _analyze_error(self, error: Exception, error_id: str, timestamp: float, context: Dict[str, Any]) -> ErrorInfo:
        """分析错误"""
        # 获取错误信息
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # 确定错误类别和严重程度
        category, severity = self._classify_error(error, error_message, stack_trace)
        
        # 生成用户友好消息
        user_friendly_message = self._generate_user_friendly_message(category, error_message)
        
        # 获取建议和恢复操作
        suggestions = self._get_suggestions(category, error_message, stack_trace)
        recovery_actions = self._get_recovery_actions(category, error_message, stack_trace)
        
        return ErrorInfo(
            error_id=error_id,
            timestamp=timestamp,
            severity=severity,
            category=category,
            message=error_message,
            details=f"{error_type}: {error_message}",
            stack_trace=stack_trace,
            context=context,
            suggestions=suggestions,
            recovery_actions=recovery_actions,
            user_friendly_message=user_friendly_message
        )
    
    def _classify_error(self, error: Exception, error_message: str, stack_trace: str) -> tuple[ErrorCategory, ErrorSeverity]:
        """分类错误"""
        error_message_lower = error_message.lower()
        stack_trace_lower = stack_trace.lower()
        
        # 错误关键词映射
        category_keywords = {
            ErrorCategory.PERMISSION: ["permission denied", "access denied", "unauthorized", "forbidden"],
            ErrorCategory.FILE_IO: ["file not found", "no such file", "permission denied", "io error", "disk full"],
            ErrorCategory.NETWORK: ["connection refused", "timeout", "network unreachable", "dns error"],
            ErrorCategory.VALIDATION: ["invalid", "validation", "required", "missing", "format"],
            ErrorCategory.RESOURCE: ["memory", "cpu", "resource", "limit exceeded"],
            ErrorCategory.TIMEOUT: ["timeout", "timed out"],
            ErrorCategory.CONFIGURATION: ["configuration", "config", "setting", "option"],
            ErrorCategory.SYSTEM: ["system", "oserror", "runtime", "internal"]
        }
        
        # 确定类别
        category = ErrorCategory.UNKNOWN
        for cat, keywords in category_keywords.items():
            if any(keyword in error_message_lower or keyword in stack_trace_lower for keyword in keywords):
                category = cat
                break
        
        # 确定严重程度
        severity = ErrorSeverity.MEDIUM
        if any(keyword in error_message_lower for keyword in ["critical", "fatal", "exception"]):
            severity = ErrorSeverity.CRITICAL
        elif any(keyword in error_message_lower for keyword in ["error", "failed", "unable"]):
            severity = ErrorSeverity.HIGH
        elif any(keyword in error_message_lower for keyword in ["warning", "deprecated"]):
            severity = ErrorSeverity.LOW
        
        return category, severity
    
    def _generate_user_friendly_message(self, category: ErrorCategory, error_message: str) -> str:
        """生成用户友好消息"""
        if not self.config["enable_friendly_messages"]:
            return error_message
        
        templates = self.error_templates.get(category, self.error_templates[ErrorCategory.UNKNOWN])
        
        # 简单的错误消息选择逻辑
        import random
        return random.choice(templates)
    
    def _get_suggestions(self, category: ErrorCategory, error_message: str, stack_trace: str) -> List[str]:
        """获取建议"""
        if not self.config["enable_suggestions"]:
            return []
        
        # 从错误模式中获取建议
        for pattern, info in self.error_patterns.items():
            if pattern in error_message.lower():
                return info["suggestions"]
        
        # 从类别获取默认建议
        return self.recovery_strategies.get(category, [])[:2]  # 只返回前2个作为建议
    
    def _get_recovery_actions(self, category: ErrorCategory, error_message: str, stack_trace: str) -> List[str]:
        """获取恢复操作"""
        if not self.config["enable_recovery_actions"]:
            return []
        
        # 从错误模式中获取恢复操作
        for pattern, info in self.error_patterns.items():
            if pattern in error_message.lower():
                return info["recovery_actions"]
        
        # 从类别获取默认恢复操作
        return self.recovery_strategies.get(category, [])
    
    def _update_error_patterns(self, error_info: ErrorInfo):
        """更新错误模式"""
        pattern_key = f"{error_info.category.value}_{error_info.severity.value}"
        self.error_patterns[pattern_key].append({
            "timestamp": error_info.timestamp,
            "message": error_info.message,
            "context": error_info.context
        })
        
        # 限制历史记录数量
        if len(self.error_patterns[pattern_key]) > 100:
            self.error_patterns[pattern_key].pop(0)
    
    def _log_error(self, error_info: ErrorInfo):
        """记录错误日志"""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error_info.severity, logging.ERROR)
        
        logger.log(log_level, f"错误 [{error_info.error_id}]: {error_info.user_friendly_message}")
        
        if error_info.details:
            logger.debug(f"详细信息: {error_info.details}")
        
        if error_info.context:
            logger.debug(f"上下文: {error_info.context}")
    
    def format_error_for_user(self, error_info: ErrorInfo, include_suggestions: bool = True) -> str:
        """为用户格式化错误信息"""
        lines = []
        
        # 主要错误消息
        lines.append(f"❌ {error_info.user_friendly_message}")
        
        # 错误ID（用于支持）
        lines.append(f"错误ID: {error_info.error_id}")
        
        # 建议和恢复操作
        if include_suggestions:
            if error_info.suggestions:
                lines.append("\n💡 建议:")
                for suggestion in error_info.suggestions[:3]:  # 最多显示3个建议
                    lines.append(f"  • {suggestion}")
            
            if error_info.recovery_actions:
                lines.append("\n🔧 可尝试的解决方法:")
                for action in error_info.recovery_actions[:3]:  # 最多显示3个操作
                    lines.append(f"  • {action}")
        
        # 联系支持信息
        if error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            lines.append("\n📞 如果问题持续存在，请联系技术支持并提供错误ID。")
        
        return "\n".join(lines)
    
    def collect_feedback(self, error_id: str, rating: int, comment: Optional[str] = None, helpful: bool = False, resolved: bool = False) -> str:
        """收集用户反馈"""
        feedback_id = f"FB_{int(time.time() * 1000)}_{hash(str(time.time())) % 10000:04d}"
        
        feedback = UserFeedback(
            feedback_id=feedback_id,
            timestamp=time.time(),
            error_id=error_id,
            rating=rating,
            comment=comment,
            helpful=helpful,
            resolved=resolved
        )
        
        self.feedback_history.append(feedback)
        
        # 分析反馈
        self._analyze_feedback(feedback)
        
        logger.info(f"📝 收到用户反馈: {feedback_id} - 评分: {rating}")
        
        return feedback_id
    
    def _analyze_feedback(self, feedback: UserFeedback):
        """分析用户反馈"""
        # 查找对应的错误信息
        error_info = None
        for error in self.error_history:
            if error.error_id == feedback.error_id:
                error_info = error
                break
        
        if not error_info:
            return
        
        # 根据反馈调整错误处理策略
        if feedback.rating <= 2:  # 低评分
            logger.warning(f"⚠️ 用户对错误处理不满意: {feedback.error_id}")
            if feedback.comment:
                logger.warning(f"用户评论: {feedback.comment}")
        
        elif feedback.rating >= 4:  # 高评分
            logger.info(f"✅ 用户对错误处理满意: {feedback.error_id}")
        
        # 更新错误处理策略
        self._update_error_handling_strategy(error_info, feedback)
    
    def _update_error_handling_strategy(self, error_info: ErrorInfo, feedback: UserFeedback):
        """更新错误处理策略"""
        # 这里可以实现更复杂的策略更新逻辑
        # 例如：基于用户反馈调整建议内容、优先级等
        pass
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        if not self.error_history:
            return {"error": "没有错误历史记录"}
        
        # 统计各类错误
        category_stats = defaultdict(int)
        severity_stats = defaultdict(int)
        
        for error in self.error_history:
            category_stats[error.category.value] += 1
            severity_stats[error.severity.value] += 1
        
        # 统计反馈
        feedback_stats = {
            "total_feedback": len(self.feedback_history),
            "average_rating": 0,
            "helpful_count": 0,
            "resolved_count": 0
        }
        
        if self.feedback_history:
            ratings = [f.rating for f in self.feedback_history if f.rating]
            helpful_count = sum(1 for f in self.feedback_history if f.helpful)
            resolved_count = sum(1 for f in self.feedback_history if f.resolved)
            
            feedback_stats["average_rating"] = sum(ratings) / len(ratings) if ratings else 0
            feedback_stats["helpful_count"] = helpful_count
            feedback_stats["resolved_count"] = resolved_count
        
        return {
            "total_errors": len(self.error_history),
            "category_distribution": dict(category_stats),
            "severity_distribution": dict(severity_stats),
            "feedback_statistics": feedback_stats,
            "most_common_errors": self._get_most_common_errors()
        }
    
    def _get_most_common_errors(self, limit: int = 5) -> List[Dict[str, Any]]:
        """获取最常见的错误"""
        error_counts = defaultdict(int)
        
        for error in self.error_history:
            error_key = f"{error.category.value}: {error.message[:50]}"
            error_counts[error_key] += 1
        
        # 排序并返回最常见的错误
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                "error_pattern": pattern,
                "count": count,
                "percentage": (count / len(self.error_history)) * 100
            }
            for pattern, count in sorted_errors[:limit]
        ]
    
    def save_configuration(self, filepath: str = None):
        """保存配置"""
        if filepath is None:
            filepath = self.config_path
        
        config = {
            "config": self.config,
            "error_templates": {k.value: v for k, v in self.error_templates.items()},
            "recovery_strategies": {k.value: v for k, v in self.recovery_strategies.items()},
            "statistics": self.get_error_statistics()
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 UX配置已保存到: {filepath}")
        except Exception as e:
            logger.error(f"❌ 保存UX配置失败: {e}")

# 全局用户体验管理器实例
ux_manager = UserExperienceManager()

def user_friendly_error_handler(func):
    """用户友好的错误处理装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # 处理错误
            error_info = ux_manager.handle_error(e, {"function": func.__name__, "args": args, "kwargs": kwargs})
            
            # 格式化并显示错误
            formatted_error = ux_manager.format_error_for_user(error_info)
            print(formatted_error)
            
            # 收集用户反馈（非交互式环境跳过）
            if hasattr(sys, 'ps1') or hasattr(sys, 'ps2'):  # 检查是否在交互式环境
                try:
                    rating = int(input("\n请为错误处理评分 (1-5): "))
                    helpful = input("这个错误信息有帮助吗？ (y/n): ").lower() == 'y'
                    comment = input("有什么建议吗？ (可选): ")
                    
                    ux_manager.collect_feedback(
                        error_info.error_id,
                        rating,
                        comment if comment else None,
                        helpful,
                        resolved=False
                    )
                except (ValueError, KeyboardInterrupt):
                    pass
            
            return None
    return wrapper

def async_user_friendly_error_handler(func):
    """异步用户友好的错误处理装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # 处理错误
            error_info = ux_manager.handle_error(e, {"function": func.__name__, "args": args, "kwargs": kwargs})
            
            # 格式化并显示错误
            formatted_error = ux_manager.format_error_for_user(error_info)
            print(formatted_error)
            
            return None
    return wrapper

# 示例使用
@user_friendly_error_handler
def example_function_with_error():
    """示例函数 - 可能产生错误"""
    # 模拟文件不存在的错误
    with open("nonexistent_file.txt", "r") as f:
        content = f.read()
    return content

@async_user_friendly_error_handler
async def example_async_function_with_error():
    """示例异步函数 - 可能产生网络错误"""
    # 模拟网络超时错误
    await asyncio.sleep(5)  # 模拟长时间操作
    raise TimeoutError("网络请求超时")

async def example_ux_usage():
    """示例用户体验使用"""
    print("🎨 用户体验管理器示例")
    
    # 测试同步错误处理
    print("\n1. 测试同步错误处理:")
    result = example_function_with_error()
    
    # 测试异步错误处理
    print("\n2. 测试异步错误处理:")
    await example_async_function_with_error()
    
    # 获取错误统计
    print("\n3. 错误统计信息:")
    stats = ux_manager.get_error_statistics()
    print(f"  总错误数: {stats['total_errors']}")
    print(f"  类别分布: {stats['category_distribution']}")
    print(f"  严重程度分布: {stats['severity_distribution']}")
    
    if stats['feedback_statistics']['total_feedback'] > 0:
        print(f"  平均评分: {stats['feedback_statistics']['average_rating']:.2f}")
        print(f"  有帮助反馈: {stats['feedback_statistics']['helpful_count']}")
    
    # 显示常见错误
    print("\n4. 最常见错误:")
    for error in stats['most_common_errors'][:3]:
        print(f"  {error['error_pattern']}: {error['count']}次 ({error['percentage']:.1f}%)")
    
    # 保存配置
    ux_manager.save_configuration()
    
    print("\n✅ 用户体验管理器示例完成")

if __name__ == "__main__":
    asyncio.run(example_ux_usage())
