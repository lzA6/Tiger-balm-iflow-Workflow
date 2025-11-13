#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iFlow 工具初始化脚本
确保所有工具依赖已安装，环境配置正确
"""

import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """检查必要的依赖"""
    required_packages = [
        "yaml",
        "asyncio",
        "concurrent.futures",
        "pathlib",
        "logging",
        "json",
        "datetime"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖已满足")
    return True

def setup_environment():
    """设置环境变量"""
    tools_dir = Path(__file__).parent
    iflow_root = tools_dir.parent
    
    # 添加到Python路径
    if str(iflow_root) not in sys.path:
        sys.path.insert(0, str(iflow_root))
    
    # 设置环境变量
    os.environ["IFLOW_ROOT"] = str(iflow_root)
    os.environ["IFLOW_TOOLS"] = str(tools_dir)
    
    print(f"✅ 环境设置完成")
    print(f"   IFLOW_ROOT: {iflow_root}")
    print(f"   IFLOW_TOOLS: {tools_dir}")

def main():
    """主函数"""
    print("🚀 初始化iFlow工具环境...")
    
    if not check_dependencies():
        sys.exit(1)
    
    setup_environment()
    
    print("\n🎯 工具环境初始化完成！")
    print("\n可用工具:")
    tools_dir = Path(__file__).parent
    for category_dir in tools_dir.iterdir():
        if category_dir.is_dir() and category_dir.name != "__pycache__":
            tools = list(category_dir.glob("*.py"))
            if tools:
                print(f"\n{category_dir.name}/")
                for tool in tools:
                    print(f"  - {tool.stem}")

if __name__ == "__main__":
    main()
