# 📋 模式化设计：任务管理模式

## 📖 模式概述

任务管理模式是一种层次化任务组织方式，通过持久化记忆处理复杂的多步骤操作。该模式专注于任务分解、进度跟踪和协作管理。

## 🎯 激活触发器

### 触发条件
- **复杂操作**: 涉及>3个步骤需要协调的操作
- **多文件范围**: 跨>2个目录或>3个文件的范围
- **复杂依赖**: 需要分阶段处理的复杂依赖关系
- **手动标志**: `--task-manage`, `--delegate`
- **质量改进**: 需要优化、细化、增强的请求

## 🏗️ 任务层次结构

### 📊 层次架构
```
📋 计划 (Plan) → write_memory("plan", goal_statement)
→ 🎯 阶段 (Phase) → write_memory("phase_X", milestone)
  → 📦 任务 (Task) → write_memory("task_X.Y", deliverable)
    → ✓ 待办 (Todo) → TodoWrite + write_memory("todo_X.Y.Z", status)
```

### 🎯 层次定义
- **计划层**: 整体目标和愿景
- **阶段层**: 重大里程碑和检查点
- **任务层**: 具体可交付成果
- **待办层**: 原子级行动项

## 💾 记忆操作

### 🚀 会话开始
```python
def session_start_operations():
    """会话开始时的记忆操作"""
    # 1. 显示现有任务状态
    memories = list_memories()
    
    # 2. 读取当前计划
    current_plan = read_memory("current_plan")
    
    # 3. 理解当前状态
    think_about_collected_information()
    
    return {
        "existing_memories": memories,
        "current_plan": current_plan,
        "readiness": assess_readiness()
    }
```

### 🔄 执行期间
```python
def during_execution_operations():
    """执行期间的记忆操作"""
    # 1. 写入任务完成状态
    write_memory("task_2.1", "completed: auth middleware")
    
    # 2. 验证任务遵循性
    think_about_task_adherence()
    
    # 3. 并行更新TodoWrite状态
    update_todowrite_status()
    
    # 4. 定期检查点记录
    if time_since_last_checkpoint() > 1800:  # 30分钟
        write_memory("checkpoint", current_state())
```

### 🏁 会话结束
```python
def session_end_operations():
    """会话结束时的记忆操作"""
    # 1. 评估完成状态
    completion_status = think_about_whether_you_are_done()
    
    # 2. 写入会话摘要
    write_memory("session_summary", outcomes)
    
    # 3. 清理已完成的临时项目
    cleanup_completed_memories()
    
    return completion_status
```

## 🔄 执行模式

### 📋 执行流程
1. **加载**: list_memories() → read_memory() → 恢复状态
2. **计划**: 创建层次结构 → 为每个层级write_memory()
3. **跟踪**: TodoWrite + 并行内存更新
4. **执行**: 随着任务完成更新内存
5. **检查点**: 定期write_memory()保存状态
6. **完成**: 最终内存更新和结果记录

### 🎯 状态管理
```python
class TaskStateManager:
    def __init__(self):
        self.memory_interface = MemoryInterface()
        self.progress_tracker = ProgressTracker()
        
    def create_task_hierarchy(self, goal, complexity):
        """创建任务层次结构"""
        hierarchy = {
            "plan": {
                "goal": goal,
                "complexity": complexity,
                "timestamp": time.time()
            },
            "phases": self.calculate_phases(goal, complexity),
            "tasks": {},
            "todos": {}
        }
        
        # 写入计划到内存
        self.memory_interface.write_memory("plan", hierarchy["plan"])
        return hierarchy
    
    def track_progress(self, task_id, status, metrics=None):
        """跟踪任务进度"""
        progress_data = {
            "task_id": task_id,
            "status": status,
            "timestamp": time.time(),
            "metrics": metrics or {},
            "completion_percentage": self.calculate_completion(task_id, status)
        }
        
        # 更新内存中的进度
        self.memory_interface.write_memory(f"progress_{task_id}", progress_data)
        self.progress_tracker.update_progress(task_id, progress_data)
        
        return progress_data
```

## 🛠️ 工具选择

### 📊 工具映射表
| 任务类型 | 主要工具 | 内存键 |
|---------|---------|--------|
| 分析 | Sequential MCP | "analysis_results" |
| 实现 | MultiEdit/Morphllm | "code_changes" |
| UI组件 | Magic MCP | "ui_components" |
| 测试 | Playwright MCP | "test_results" |
| 文档 | Context7 MCP | "doc_patterns" |

### 🔧 工具集成
```python
class ToolSelector:
    def __init__(self):
        self.tool_mappings = {
            "analysis": {
                "primary": "SequentialMCP",
                "memory_key": "analysis_results",
                "fallback": "Context7MCP"
            },
            "implementation": {
                "primary": "MultiEdit",
                "memory_key": "code_changes",
                "fallback": "Morphllm"
            },
            "testing": {
                "primary": "PlaywrightMCP",
                "memory_key": "test_results",
                "fallback": "SequentialMCP"
            }
        }
    
    def select_tool(self, task_type, context):
        """根据任务类型选择工具"""
        if task_type not in self.tool_mappings:
            return self.fallback_tool_selection(context)
        
        tool_config = self.tool_mappings[task_type]
        
        # 根据上下文和可用性选择具体工具
        selected_tool = self.evaluate_tool_availability(tool_config)
        
        return {
            "tool": selected_tool,
            "memory_key": tool_config["memory_key"],
            "parameters": self.generate_tool_parameters(task_type, context)
        }
```

## 📝 内存架构

### 🏗️ 内存模式
```python
MEMORY_SCHEMA = {
    "plan_[timestamp]": "整体目标声明",
    "phase_[1-5]": "主要里程碑描述",
    "task_[phase].[number]": "特定交付成果状态",
    "todo_[task].[number]": "原子行动完成状态",
    "checkpoint_[timestamp]": "当前状态快照",
    "blockers": "需要关注的活跃障碍",
    "decisions": "已做出的关键架构/设计选择",
    "progress_[task_id]": "任务进度详细信息",
    "metrics_[task_id]": "任务相关指标",
    "dependencies_[task_id]": "任务依赖关系"
}
```

### 💾 内存管理
```python
class MemoryManager:
    def __init__(self):
        self.memory_backend = MemoryBackend()
        self.schema_validator = SchemaValidator()
        
    def write_structured_memory(self, key, data, schema=None):
        """写入结构化内存"""
        if schema and not self.schema_validator.validate(data, schema):
            raise ValueError("数据不符合预期架构")
        
        # 添加元数据
        enriched_data = {
            "data": data,
            "metadata": {
                "timestamp": time.time(),
                "version": self.get_current_version(),
                "schema": schema,
                "source": "TaskManagementMode"
            }
        }
        
        return self.memory_backend.write(key, enriched_data)
    
    def read_task_hierarchy(self):
        """读取任务层次结构"""
        hierarchy = {}
        
        # 读取计划
        plan = self.memory_backend.read("plan")
        if plan:
            hierarchy["plan"] = plan
            
            # 读取阶段
            phases = []
            for i in range(1, 6):  # 最多5个阶段
                phase = self.memory_backend.read(f"phase_{i}")
                if phase:
                    phases.append(phase)
            hierarchy["phases"] = phases
            
            # 读取任务和待办
            hierarchy["tasks"] = self.read_all_tasks()
            hierarchy["todos"] = self.read_all_todos()
        
        return hierarchy
```

## 🎨 使用示例

### 📁 会话1：开始认证任务
```python
# 会话开始
memories = list_memories()  # → 空
write_memory("plan_auth", "实现JWT认证系统")
write_memory("phase_1", "分析 - 安全需求审查")
write_memory("task_1.1", "pending: 审查现有认证模式")

# 创建TodoWrite任务
todos = [
    "审查用户认证流程",
    "分析安全漏洞风险", 
    "研究最佳实践",
    "设计认证架构",
    "制定实施计划"
]
TodoWrite.create_todos(todos)

# 执行任务1.1
execute_task_1_1()
write_memory("task_1.1", "completed: 发现3种模式")
```

### 🔄 会话2：中断后恢复
```python
# 恢复会话
memories = list_memories()  # → 显示 plan_auth, phase_1, task_1.1
plan = read_memory("plan_auth")  # → "实现JWT认证系统"
context = think_about_collected_information()
adherence = think_about_task_adherence()

# 继续执行
write_memory("phase_2", "实施 - 中间件和端点")
write_memory("task_2.1", "pending: 实现认证中间件")
write_memory("task_2.2", "pending: 创建认证端点")

# 更新进度
update_progress("task_2.1", "in_progress")
```

### ✅ 会话3：完成检查
```python
# 完成检查
completion_check = think_about_whether_you_are_done()

if not completion_check["all_tasks_complete"]:
    # 完成剩余测试任务
    complete_remaining_tests()
    
# 最终记录
write_memory("outcome_auth", "成功实现，测试覆盖率95%")
write_memory("session_summary", "认证系统完成并通过验证")

# 清理临时状态
cleanup_temporary_states()
```

## 📊 进度跟踪

### 🎯 跟踪指标
```python
class ProgressMetrics:
    def __init__(self):
        self.completion_rate = 0.0
        self.velocity = 0.0
        self.blocker_count = 0
        self.quality_score = 0.0
        
    def calculate_completion_rate(self, hierarchy):
        """计算完成率"""
        total_tasks = self.count_total_tasks(hierarchy)
        completed_tasks = self.count_completed_tasks(hierarchy)
        
        return (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
    
    def calculate_velocity(self, time_period):
        """计算执行速度"""
        tasks_completed = self.get_tasks_completed_in_period(time_period)
        return tasks_completed / time_period
    
    def assess_blockers(self):
        """评估障碍"""
        blockers = read_memory("blockers")
        severity_scores = [self.assess_blocker_severity(blocker) for blocker in blockers]
        
        return {
            "count": len(blockers),
            "average_severity": sum(severity_scores) / len(severity_scores) if severity_scores else 0,
            "estimated_delay": self.calculate_delay_impact(blockers)
        }
```

### 📈 可视化报告
```python
class ProgressVisualizer:
    def __init__(self):
        self.chart_generator = ChartGenerator()
        
    def generate_progress_report(self, hierarchy):
        """生成进度报告"""
        completion_rate = self.calculate_completion_rate(hierarchy)
        velocity = self.calculate_velocity(24)  # 24小时
        blockers = self.assess_blockers()
        
        report = {
            "completion_chart": self.chart_generator.create_completion_chart(hierarchy),
            "velocity_trend": self.chart_generator.create_velocity_trend(),
            "blocker_analysis": blockers,
            "risk_assessment": self.assess_project_risks(hierarchy),
            "recommendations": self.generate_recommendations(hierarchy)
        }
        
        return report
```

## 🔧 最佳实践

### 📋 任务设计
- **SMART原则**: 确保任务具体、可衡量、可实现、相关、有时限
- **适当分解**: 将大任务分解为可管理的子任务
- **依赖管理**: 清晰识别和管理任务依赖关系
- **优先级排序**: 根据重要性和紧急性排序任务

### 💾 记忆管理
- **定期备份**: 定期备份重要记忆数据
- **版本控制**: 为重要决策和状态变更创建版本
- **清理策略**: 建立内存清理和优化策略
- **一致性维护**: 确保内存数据的一致性和准确性

### 🔄 协作管理
- **透明沟通**: 保持任务状态和进度的透明度
- **定期同步**: 定期同步团队成员的任务状态
- **冲突解决**: 建立有效的冲突解决机制
- **知识共享**: 促进团队成员间的知识共享

## 🎯 高级功能

### 🤖 自动化集成
```python
class AutomatedTaskManager:
    def __init__(self):
        self.ai_planner = AIPlanner()
        self.automated_executor = AutomatedExecutor()
        
    def auto_plan_tasks(self, project_requirements):
        """自动生成任务计划"""
        # AI分析项目需求
        analysis = self.ai_planner.analyze_requirements(project_requirements)
        
        # 自动生成任务分解
        task_breakdown = self.ai_planner.generate_task_breakdown(analysis)
        
        # 优化任务顺序和依赖
        optimized_plan = self.ai_planner.optimize_task_sequence(task_breakdown)
        
        return optimized_plan
    
    def auto_track_progress(self):
        """自动跟踪进度"""
        # 监控任务执行状态
        status_updates = self.automated_executor.get_status_updates()
        
        # 自动更新内存
        for update in status_updates:
            self.memory_manager.write_memory(update.key, update.data)
        
        # 生成进度报告
        return self.generate_automated_progress_report()
```

### 🧠 智能优化
```python
class IntelligentOptimizer:
    def __init__(self):
        self.ml_optimizer = MachineLearningOptimizer()
        self.pattern_analyzer = PatternAnalyzer()
        
    def optimize_task_allocation(self, team_capabilities, task_requirements):
        """优化任务分配"""
        # 分析团队能力
        capabilities = self.analyze_team_capabilities(team_capabilities)
        
        # 匹配任务需求
        optimal_allocation = self.ml_optimizer.optimize_allocation(
            capabilities, task_requirements
        )
        
        return optimal_allocation
    
    def predict_completion_time(self, current_progress, historical_data):
        """预测完成时间"""
        # 分析历史模式
        patterns = self.pattern_analyzer.extract_patterns(historical_data)
        
        # 预测未来趋势
        prediction = self.ml_optimizer.predict_completion_time(
            current_progress, patterns
        )
        
        return prediction
```

## 📚 相关资源

### 📖 学习资料
- 《敏捷项目管理》- Jim Highsmith
- 《Scrum指南》- Ken Schwaber, Jeff Sutherland
- 《项目管理知识体系指南》- PMI
- 《OKR工作法》- John Doerr

### 🛠️ 工具推荐
- **项目管理**: Jira, Asana, Trello
- **任务跟踪**: Todoist, Microsoft To Do
- **协作工具**: Slack, Microsoft Teams
- **文档管理**: Notion, Confluence

### 🔗 方法论
- **敏捷开发**: SCRUM, Kanban, XP
- **项目管理**: PRINCE2, PMBOK
- **任务管理**: GTD, Eisenhower Matrix
- **团队协作**: DevOps, Lean

---

*本文档最后更新时间: 2025年11月13日*
*版本: V6.0*
*状态: 已完成*