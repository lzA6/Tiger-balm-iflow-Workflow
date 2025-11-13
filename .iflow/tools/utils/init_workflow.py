#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全能工作流V5初始化脚本
OmniWorkflow V5 Initialization Script

作者: Quantum AI Team
版本: 5.0.0
日期: 2025-11-12
"""

import os
import sys
import json
import yaml
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WorkflowInitializer:
    """工作流初始化器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.iflow_dir = self.project_root / '.iflow'
        self.config_dir = self.iflow_dir / 'config'
        self.workflows_dir = self.iflow_dir / 'workflows'
        self.tools_dir = self.project_root / 'tools'
        
        # 必需的目录结构
        self.required_dirs = [
            self.iflow_dir,
            self.config_dir,
            self.workflows_dir,
            self.tools_dir,
            self.project_root / 'docs',
            self.project_root / 'scripts',
            self.project_root / 'tests',
            self.project_root / 'examples'
        ]
        
        # 必需的配置文件
        self.required_configs = [
            'quantum-config.yaml',
            'model-adapter.yaml',
            'quality-gate.yaml'
        ]
        
        # 必需的工作流文件
        self.required_workflows = [
            'quantum-fullstack-development.yaml',
            'quantum-ai-project.yaml',
            'quantum-mobile-application.yaml'
        ]
    
    def initialize(self) -> bool:
        """初始化工作流"""
        try:
            logger.info("开始初始化全能工作流V5...")
            
            # 检查Python版本
            self._check_python_version()
            
            # 创建目录结构
            self._create_directory_structure()
            
            # 验证配置文件
            self._validate_configurations()
            
            # 初始化量子配置
            self._initialize_quantum_config()
            
            # 验证模型适配器
            self._validate_model_adapter()
            
            # 创建示例项目
            self._create_example_project()
            
            # 运行健康检查
            self._run_health_check()
            
            logger.info("✅ 全能工作流V5初始化成功！")
            return True
            
        except Exception as e:
            logger.error(f"❌ 初始化失败: {e}")
            return False
    
    def _check_python_version(self):
        """检查Python版本"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 9):
            raise RuntimeError(f"需要Python 3.9+，当前版本: {version.major}.{version.minor}")
        logger.info(f"✅ Python版本检查通过: {version.major}.{version.minor}.{version.micro}")
    
    def _create_directory_structure(self):
        """创建目录结构"""
        logger.info("创建目录结构...")
        
        for directory in self.required_dirs:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"  📁 {directory}")
        
        # 创建.gitignore
        gitignore_path = self.project_root / '.gitignore'
        if not gitignore_path.exists():
            gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Quantum
quantum_cache/
quantum_states/

# Temporary
tmp/
temp/
"""
            gitignore_path.write_text(gitignore_content.strip())
            logger.info(f"  📄 .gitignore")
    
    def _validate_configurations(self):
        """验证配置文件"""
        logger.info("验证配置文件...")
        
        for config_file in self.required_configs:
            config_path = self.config_dir / config_file
            if not config_path.exists():
                raise FileNotFoundError(f"缺少配置文件: {config_file}")
            
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                        yaml.safe_load(f)
                    else:
                        json.load(f)
                logger.info(f"  ✅ {config_file}")
            except Exception as e:
                raise ValueError(f"配置文件格式错误 {config_file}: {e}")
    
    def _initialize_quantum_config(self):
        """初始化量子配置"""
        logger.info("初始化量子配置...")
        
        quantum_config_path = self.config_dir / 'quantum-config.yaml'
        
        with open(quantum_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 验证量子配置
        required_sections = ['project', 'quantum', 'models', 'agents']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"量子配置缺少必需部分: {section}")
        
        # 设置默认值
        if 'quantum' not in config or not config['quantum'].get('enabled'):
            config['quantum'] = {
                'enabled': True,
                'qubits': 32,
                'algorithm': 'quantum-annealing',
                'optimization_level': 'maximum'
            }
        
        # 保存更新后的配置
        with open(quantum_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info("  ✅ 量子配置初始化完成")
    
    def _validate_model_adapter(self):
        """验证模型适配器"""
        logger.info("验证模型适配器...")
        
        model_adapter_path = self.config_dir / 'model-adapter.yaml'
        
        with open(model_adapter_path, 'r', encoding='utf-8') as f:
            adapter_config = yaml.safe_load(f)
        
        # 验证提供商配置
        if 'providers' not in adapter_config:
            raise ValueError("模型适配器缺少providers配置")
        
        # 验证路由配置
        if 'routing' not in adapter_config:
            raise ValueError("模型适配器缺少routing配置")
        
        logger.info("  ✅ 模型适配器验证通过")
    
    def _create_example_project(self):
        """创建示例项目"""
        logger.info("创建示例项目...")
        
        example_dir = self.project_root / 'examples' / 'quantum-hello-world'
        example_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建示例配置
        example_config = {
            "project": {
                "name": "quantum-hello-world",
                "version": "1.0.0",
                "type": "quantum-demo",
                "description": "量子工作流示例项目"
            },
            "workflow": "quantum-fullstack-development",
            "quantum": {
                "enabled": True,
                "qubits": 8,
                "optimization_level": "medium"
            }
        }
        
        config_path = example_dir / 'project-config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(example_config, f, default_flow_style=False, allow_unicode=True)
        
        # 创建示例Python文件
        example_py = example_dir / 'main.py'
        example_py_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量子Hello World示例
Quantum Hello World Example
"""

import sys
import os

# 添加工作流路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from omniworkflow import QuantumWorkflow

def main():
    """主函数"""
    print("🚀 启动量子Hello World示例...")
    
    # 创建量子工作流
    workflow = QuantumWorkflow()
    
    # 加载配置
    workflow.load_config('project-config.yaml')
    
    # 执行量子问候
    result = workflow.execute_quantum_greeting()
    
    print(f"✨ 量子问候结果: {result}")
    print("🎉 示例执行完成！")

if __name__ == "__main__":
    main()
'''
        
        example_py.write_text(example_py_content)
        
        # 创建README
        readme_path = example_dir / 'README.md'
        readme_content = '''# 量子Hello World示例

这是一个展示全能工作流V5基本功能的示例项目。

## 运行示例

```bash
cd examples/quantum-hello-world
python main.py
```

## 预期输出

```
🚀 启动量子Hello World示例...
✨ 量子问候结果: Hello Quantum World!
🎉 示例执行完成！
```
'''
        
        readme_path.write_text(readme_content)
        
        logger.info("  ✅ 示例项目创建完成")
    
    def _run_health_check(self):
        """运行健康检查"""
        logger.info("运行健康检查...")
        
        health_check_script = self.tools_dir / 'health_check.py'
        
        if health_check_script.exists():
            import subprocess
            try:
                result = subprocess.run(
                    [sys.executable, str(health_check_script)],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root
                )
                
                if result.returncode == 0:
                    logger.info("  ✅ 健康检查通过")
                    logger.info(f"  📊 检查结果:\n{result.stdout}")
                else:
                    logger.warning(f"  ⚠️ 健康检查警告:\n{result.stderr}")
            except Exception as e:
                logger.warning(f"  ⚠️ 无法运行健康检查: {e}")
        else:
            logger.warning("  ⚠️ 健康检查脚本不存在")

class ProjectCreator:
    """项目创建器"""
    
    def __init__(self):
        self.templates_dir = Path(__file__).parent.parent / '.iflow' / 'templates'
    
    def create_project(self, project_name: str, project_type: str = "quantum-fullstack") -> bool:
        """创建新项目"""
        try:
            logger.info(f"创建新项目: {project_name} (类型: {project_type})")
            
            # 创建项目目录
            project_dir = Path.cwd() / project_name
            project_dir.mkdir(exist_ok=True)
            
            # 创建项目结构
            self._create_project_structure(project_dir, project_type)
            
            # 生成项目配置
            self._generate_project_config(project_dir, project_name, project_type)
            
            # 创建初始文件
            self._create_initial_files(project_dir, project_type)
            
            logger.info(f"✅ 项目 {project_name} 创建成功！")
            logger.info(f"📁 项目位置: {project_dir}")
            logger.info(f"🚀 开始开发: cd {project_name} && python main.py")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 项目创建失败: {e}")
            return False
    
    def _create_project_structure(self, project_dir: Path, project_type: str):
        """创建项目结构"""
        dirs = ['src', 'tests', 'docs', 'config', 'scripts']
        
        if project_type in ["quantum-fullstack", "quantum-web"]:
            dirs.extend(['frontend', 'backend', 'api'])
        
        for dir_name in dirs:
            (project_dir / dir_name).mkdir(exist_ok=True)
    
    def _generate_project_config(self, project_dir: Path, project_name: str, project_type: str):
        """生成项目配置"""
        config = {
            "project": {
                "name": project_name,
                "version": "1.0.0",
                "type": project_type,
                "description": f"{project_name} - {project_type}项目"
            },
            "workflow": self._get_workflow_template(project_type),
            "quantum": {
                "enabled": True,
                "optimization_level": "medium"
            }
        }
        
        config_path = project_dir / 'project-config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    def _get_workflow_template(self, project_type: str) -> str:
        """获取工作流模板"""
        workflow_map = {
            "quantum-fullstack": "quantum-fullstack-development",
            "quantum-ai": "quantum-ai-project",
            "quantum-mobile": "quantum-mobile-application",
            "quantum-web": "quantum-fullstack-development"
        }
        return workflow_map.get(project_type, "quantum-fullstack-development")
    
    def _create_initial_files(self, project_dir: Path, project_type: str):
        """创建初始文件"""
        # 创建主文件
        main_file = project_dir / 'main.py'
        main_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
{project_dir.name} 主文件
Main file for {project_dir.name}
"""

import sys
import os
from pathlib import Path

# 添加工作流路径
workflow_path = Path(__file__).parent.parent / '.iflow'
sys.path.insert(0, str(workflow_path))

def main():
    """主函数"""
    print("🚀 启动 {project_dir.name}...")
    
    # TODO: 在这里添加您的代码
    
    print("✅ 执行完成！")

if __name__ == "__main__":
    main()
'''
        main_file.write_text(main_content)
        
        # 创建README
        readme_file = project_dir / 'README.md'
        readme_content = f'''# {project_dir.name}

{project_dir.name} - 基于{project_type}的项目

## 快速开始

```bash
python main.py
```

## 项目结构

```
{project_dir.name}/
├── src/              # 源代码
├── tests/             # 测试文件
├── docs/              # 文档
├── config/            # 配置文件
├── scripts/           # 脚本
├── main.py            # 主文件
├── project-config.yaml # 项目配置
└── README.md          # 项目说明
```

## 开发指南

1. 修改 `src/` 目录下的源代码
2. 在 `tests/` 目录添加测试
3. 在 `docs/` 目录添加文档
4. 使用 `python main.py` 运行项目

## 更多信息

- [全能工作流文档](../docs/README.md)
- [API文档](../docs/api.md)
- [示例项目](../examples/)
'''
        readme_file.write_text(readme_content)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="全能工作流V5初始化工具")
    parser.add_argument('--init', action='store_true', help='初始化工作流')
    parser.add_argument('--create', metavar='PROJECT_NAME', help='创建新项目')
    parser.add_argument('--type', choices=['quantum-fullstack', 'quantum-ai', 'quantum-mobile', 'quantum-web'], 
                       default='quantum-fullstack', help='项目类型')
    parser.add_argument('--health', action='store_true', help='运行健康检查')
    
    args = parser.parse_args()
    
    if args.init:
        initializer = WorkflowInitializer()
        success = initializer.initialize()
        sys.exit(0 if success else 1)
    
    elif args.create:
        creator = ProjectCreator()
        success = creator.create_project(args.create, args.type)
        sys.exit(0 if success else 1)
    
    elif args.health:
        # 运行健康检查
        from tools.health_check import HealthChecker
        checker = HealthChecker()
        health = checker.check()
        print(f"健康检查结果: {'✅ 健康' if health else '❌ 有问题'}")
        sys.exit(0 if health else 1)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()