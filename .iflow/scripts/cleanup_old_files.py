#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理旧版本文件和冗余代码
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import shutil
from pathlib import Path
import json
from datetime import datetime

def cleanup_old_reports():
    """清理旧的报告文件"""
    reports_dir = Path(__file__).parent.parent / "reports"
    if reports_dir.exists():
        for file in reports_dir.glob("*.json"):
            try:
                file.unlink()
                print(f"已删除: {file}")
            except Exception as e:
                print(f"删除失败 {file}: {e}")

def cleanup_test_coverage():
    """清理旧的测试覆盖率文件"""
    coverage_dir = Path(__file__).parent.parent / "tests" / "coverage"
    if coverage_dir.exists():
        try:
            shutil.rmtree(coverage_dir)
            print(f"已删除覆盖率目录: {coverage_dir}")
        except Exception as e:
            print(f"删除失败 {coverage_dir}: {e}")

def cleanup_temp_files():
    """清理临时文件"""
    temp_dir = Path(__file__).parent.parent / "temp"
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
            print(f"已删除临时目录: {temp_dir}")
        except Exception as e:
            print(f"删除失败 {temp_dir}: {e}")
    
    # 重新创建空的temp目录
    temp_dir.mkdir(exist_ok=True)

def consolidate_duplicate_tools():
    """整合重复的工具文件"""
    tools_dir = Path(__file__).parent.parent / "tools"
    
    # 检查并标记重复的工具
    tool_files = {}
    for file in tools_dir.rglob("*.py"):
        if file.name in tool_files:
            print(f"发现重复文件: {file} 和 {tool_files[file.name]}")
        else:
            tool_files[file.name] = file

def main():
    """主函数"""
    print("开始清理旧版本文件...")
    
    # 备份当前状态
    backup_dir = Path(__file__).parent.parent / "backups" / f"cleanup_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # 清理各类文件
    cleanup_old_reports()
    cleanup_test_coverage()
    cleanup_temp_files()
    consolidate_duplicate_tools()
    
    print("清理完成！")

if __name__ == "__main__":
    main()