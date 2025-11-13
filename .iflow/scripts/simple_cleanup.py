#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版清理脚本
只清理明显的冗余文件，保留所有现有功能
"""

import os
import sys
import json
import shutil
import time
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent

def main():
    print("开始保守清理A项目...")
    print("=" * 50)
    
    # 查找重复文件
    duplicates_to_clean = [
        ("README文件", [".iflow/README.md", ".iflow/README_V4.md"]),
        ("变更日志", [".iflow/CHANGELOG.md", ".iflow/CHANGELOG_V4.md"]),
        ("Hook管理器", [
            ".iflow/hooks/comprehensive_hook_manager.py",
            ".iflow/hooks/comprehensive_hook_manager_v4.py",
            ".iflow/hooks/comprehensive-hook-manager.py", 
            ".iflow/hooks/comprehensive-hook-manager-v4.py"
        ]),
        ("质量检查", [
            ".iflow/hooks/auto_quality_check.py",
            ".iflow/hooks/auto_quality_check_v6.py"
        ]),
    ]
    
    cleanup_results = {
        "kept_files": [],
        "removed_files": [],
        "errors": []
    }
    
    # 清理重复文件
    for category, file_list in duplicates_to_clean:
        existing_files = []
        for file_path_str in file_list:
            file_path = PROJECT_ROOT / file_path_str
            if file_path.exists():
                existing_files.append((file_path_str, file_path))
        
        if len(existing_files) <= 1:
            continue
            
        print(f"\n处理 {category}:")
        
        # 保留最新修改的文件
        latest_file = max(existing_files, key=lambda x: x[1].stat().st_mtime)
        files_to_remove = [f for f in existing_files if f != latest_file]
        
        # 保留文件
        cleanup_results["kept_files"].append({
            "file": latest_file[0],
            "reason": "保留最新版本",
            "category": category
        })
        print(f"  保留: {latest_file[0]}")
        
        # 删除其他文件（备份）
        for remove_file in files_to_remove:
            try:
                backup_path = str(remove_file[1]) + ".backup"
                shutil.move(str(remove_file[1]), backup_path)
                cleanup_results["removed_files"].append({
                    "file": remove_file[0],
                    "backup": backup_path,
                    "category": category
                })
                print(f"  备份并删除: {remove_file[0]} -> {backup_path}")
            except Exception as e:
                error_msg = f"删除失败: {remove_file[0]} - {e}"
                cleanup_results["errors"].append(error_msg)
                print(f"  错误: {error_msg}")
    
    # 清理临时文件
    print(f"\n清理临时文件...")
    temp_files = ["temp_delete.py"]
    
    for temp_file in temp_files:
        file_path = PROJECT_ROOT / temp_file
        if file_path.exists():
            try:
                backup_path = str(file_path) + ".backup"
                shutil.move(str(file_path), backup_path)
                cleanup_results["removed_files"].append({
                    "file": temp_file,
                    "backup": backup_path
                })
                print(f"  清理临时文件: {temp_file}")
            except Exception as e:
                error_msg = f"清理临时文件失败: {temp_file} - {e}"
                cleanup_results["errors"].append(error_msg)
                print(f"  错误: {error_msg}")
    
    # 生成报告
    print("\n" + "=" * 50)
    print("清理总结:")
    
    total_removed = len(cleanup_results["removed_files"])
    total_kept = len(cleanup_results["kept_files"])
    
    print(f"总计删除文件: {total_removed} 个")
    print(f"总计保留文件: {total_kept} 个")
    
    if cleanup_results["errors"]:
        print(f"清理错误: {len(cleanup_results['errors'])} 个")
    
    # 保存报告
    report_path = PROJECT_ROOT / ".iflow" / "simple_cleanup_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(cleanup_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n清理报告已保存: {report_path}")
    print("\n清理完成！所有删除的文件都已备份。")
    print("建议检查备份文件确认无误后，再手动删除.backup文件。")

if __name__ == "__main__":
    main()