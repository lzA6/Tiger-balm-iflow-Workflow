import os
import shutil
from pathlib import Path

# 定义智能体分类规则
AGENT_CLASSIFICATION = {
    # Meta System
    "系统进化师": "meta_system",
    "root-cause-analyst": "meta_system",
    # Strategic Command
    "技术愿景师": "strategic_command",
    "requirements-analyst": "strategic_command",
    "system-architect": "strategic_command",
    # Universal Execution (remains as is)
    "全能工程师": "universal_execution",
    "backend-architect": "universal_execution",
    "devops-architect": "universal_execution",
    "frontend-architect": "universal_execution",
    "performance-engineer": "universal_execution",
    "python-expert": "universal_execution",
    "quality-engineer": "universal_execution",
    "refactoring-expert": "universal_execution",
    "security-engineer": "universal_execution",
    # Knowledge Wisdom
    "知识图谱师": "knowledge_wisdom",
    "learning-guide": "knowledge_wisdom",
    "socratic-mentor": "knowledge_wisdom",
    "technical-writer": "knowledge_wisdom",
    # Innovation Discovery
    "创新发现师": "innovation_discovery",
}

def refactor_agent_structure(base_path: Path):
    """
    重构智能体目录结构，将所有智能体按分类移动到指定的分层目录中。
    你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
    """
    agents_path = base_path / "iflow" / "agents"
    if not agents_path.is_dir():
        print(f"错误：找不到 'agents' 目录：{agents_path}")
        return

    print("开始重构智能体目录结构...")

    # 1. 确保所有目标目录都存在
    target_dirs = set(AGENT_CLASSIFICATION.values())
    for dir_name in target_dirs:
        (agents_path / dir_name).mkdir(exist_ok=True)

    # 2. 遍历所有子目录，查找 .md 文件并移动
    for root, _, files in os.walk(agents_path):
        current_dir = Path(root)
        # 避免在目标目录内再次移动
        if current_dir.name in target_dirs and current_dir != agents_path:
             # 如果我们已经在一个分类目录里，就跳过，防止重复移动
             if any(parent.name in target_dirs for parent in current_dir.parents):
                 continue

        for file in files:
            if file.endswith(".md"):
                agent_name = Path(file).stem
                if agent_name in AGENT_CLASSIFICATION:
                    target_dir_name = AGENT_CLASSIFICATION[agent_name]
                    target_dir_path = agents_path / target_dir_name
                    
                    source_path = current_dir / file
                    destination_path = target_dir_path / file
                    
                    if source_path != destination_path:
                        try:
                            print(f"移动 '{source_path.relative_to(base_path)}' 到 '{destination_path.relative_to(base_path)}'")
                            shutil.move(str(source_path), str(destination_path))
                        except Exception as e:
                            print(f"移动文件 {source_path} 失败: {e}")

    print("智能体目录结构重构完成。")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    refactor_agent_structure(project_root)