import shutil
from pathlib import Path
import json
import datetime
from typing import Dict, List, Any

# 定义工具的理想分类
TOOL_CLASSIFICATION = {
    # Analysis
    "project-analyzer.py": "analysis",
    "tool-call-validator.py": "analysis",
    "quantum-analyzer.py": "analysis",
    # Core
    "agent_orchestrator.py": "core",
    "multi-agent-orchestrator.py": "core",
    "enhanced-universal-agent.py": "core",
    "state_transition_engine.py": "core",
    "intelligent-tool-caller.py": "core",
    "intelligent-tool-integrator.py": "core",
    "omega-precision-tool-caller.py": "core",
    # Optimization
    "performance_optimizer.py": "optimization",
    "quantum-performance-optimizer.py": "optimization",
    "quantum_optimizer.py": "optimization",
    "adaptive_quantum_annealing.py": "optimization",
    # Quantum
    "quantum-integration-engine.py": "quantum",
    "enterprise-quantum-deployment.py": "quantum",
    "quantum-cloud-platform.py": "quantum",
    "quantum-code-generator.py": "quantum",
    "quantum-computing-integration.py": "quantum",
    "quantum-graphical-editor.py": "quantum",
    # Security
    "security_monitor.py": "security",
    "quantum-security-scanner.py": "security",
    # Testing
    "test-runner.py": "testing",
    "auto_test_system.py": "testing",
    "quick-performance-test.py": "testing",
    "simple_performance_test.py": "testing",
    "quantum-performance-benchmark.py": "testing",
    "ultimate_performance_benchmark.py": "testing",
    # Other Core/Utils
    "context-compressor.py": "utils",
    "intelligent_cache.py": "utils",
    "context_manager.py": "utils",
    "ultimate_llm_router.py": "utils",
    "code_generator.py": "utils",
    "error_handler.py": "utils",
    "init_workflow.py": "utils",
    "agent_memory_system.py": "utils",
    "enhanced_context_understanding.py": "utils",
    # Self-Evolution
    "ultimate_self_evolution_engine.py": "evolution",
    "reinforcement_learning_agent.py": "evolution",
    # Project Analysis
    "omega-project-difficulty-analyzer.py": "analysis"
}

# 定义需要被整合或删除的冗余工具
REDUNDANT_TOOLS = {
    "quick-performance-test.py": "ultimate_performance_benchmark.py",
    "simple_performance_test.py": "ultimate_performance_benchmark.py",
    "quantum-performance-benchmark.py": "ultimate_performance_benchmark.py",
    "agent_orchestrator.py": "multi-agent-orchestrator.py"
}


def optimize_tools_structure(base_path: Path) -> None:
    """
    智能重构 a-project/iflow/tools 目录。
    1. 创建标准化的子目录。
    2. 将现有工具移动到正确的分类目录中。
    3. 识别并报告冗余工具。
    你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
    """
    tools_path = base_path / "iflow" / "tools"
    if not tools_path.is_dir():
        print(f"错误：找不到 'tools' 目录：{tools_path}")
        return

    print("开始重构工具目录结构...")
    report: Dict[str, List[Any]] = {
        "moved_files": [],
        "redundant_files_to_delete": [],
        "unclassified_files": [],
        "errors": []
    }

    # 1. 确保所有目标目录都存在
    target_dirs = set(TOOL_CLASSIFICATION.values())
    for dir_name in target_dirs:
        (tools_path / dir_name).mkdir(exist_ok=True)
    
    # 2. 遍历所有文件和目录进行分类和移动
    all_files = [p for p in tools_path.rglob('*.py') if p.is_file()]

    for source_path in all_files:
        file_name = source_path.name
        
        # 跳过此脚本本身
        if file_name == "optimize_tools.py":
            continue

        # 检查是否为冗余文件
        if file_name in REDUNDANT_TOOLS:
            report["redundant_files_to_delete"].append({
                "file": str(source_path.relative_to(base_path)),
                "reason": f"冗余，功能可被 '{REDUNDANT_TOOLS[file_name]}' 替代。"
            })
            continue

        # 检查是否在分类中
        if file_name in TOOL_CLASSIFICATION:
            target_dir_name = TOOL_CLASSIFICATION[file_name]
            target_dir_path = tools_path / target_dir_name
            destination_path = target_dir_path / file_name

            if source_path.parent != target_dir_path:
                 try:
                    print(f"移动 '{source_path.relative_to(base_path)}' 到 '{destination_path.relative_to(base_path)}'")
                    shutil.move(str(source_path), str(destination_path))
                    report["moved_files"].append({
                        "from": str(source_path.relative_to(base_path)),
                        "to": str(destination_path.relative_to(base_path))
                    })
                 except Exception as e:
                    error_msg = f"移动文件 {source_path} 失败: {e}"
                    print(error_msg)
                    report["errors"].append(error_msg)
        elif source_path.parent == tools_path: # 只报告顶层的未分类文件
            report["unclassified_files"].append(str(source_path.relative_to(base_path)))

    # 3. 生成报告
    report_path = base_path / "iflow" / "reports" / f"tools_optimization_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n--- 工具优化报告 ---")
    print(f"报告已生成: {report_path}")
    print(f"移动文件数: {len(report['moved_files'])}")
    print(f"建议删除的冗余文件数: {len(report['redundant_files_to_delete'])}")
    print(f"未分类文件数: {len(report['unclassified_files'])}")
    print(f"错误数: {len(report['errors'])}")
    print("\n建议删除的冗余文件:")
    for item in report["redundant_files_to_delete"]:
        print(f"- {item['file']} ({item['reason']})")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    optimize_tools_structure(project_root)