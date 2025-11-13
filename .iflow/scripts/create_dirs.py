import os

dirs_to_create = [
    "A项目/iflow/agents/universal_execution",
    "A项目/iflow/agents/strategic_command",
    "A项目/iflow/agents/knowledge_wisdom",
    "A项目/iflow/agents/meta_system",
]

for d in dirs_to_create:
    os.makedirs(d, exist_ok=True)