#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
激进清理计划 - 渐进式清理重复文件和旧版本
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import shutil
from pathlib import Path
import json

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def get_duplicate_files_analysis():
    """分析重复文件"""
    duplicates = {
        "hook_managers": [
            "iflow/hooks/comprehensive_hook_manager.py",
            "iflow/hooks/comprehensive_hook_manager_v4.py", 
            "iflow/hooks/comprehensive-hook-manager.py",
            "iflow/hooks/comprehensive-hook-manager-v4.py"
        ],
        "auto_quality_checks": [
            "iflow/hooks/auto_quality_check.py",
            "iflow/hooks/auto_quality_check_v6.py"
        ],
        "cleanup_scripts": [
            "iflow/scripts/cleanup_v4.py",
            "iflow/scripts/conservative_cleanup_v2.py",
            "iflow/scripts/simple_cleanup.py",
            "iflow/scripts/intelligent_cleanup.py",
            "iflow/scripts/fixed_cleanup_script.py",
            "iflow/scripts/smart_cleanup_manager.py",
            "iflow/scripts/refactor_agents.py",
            "iflow/scripts/cleanup_old_files.py"
        ],
        "test_versions": [
            "iflow/tests/simple_test_v4.py"
        ],
        "ultimate_versions": {
            "consciousness_system": [
                "iflow/core/ultimate_consciousness_system_v4.py",
                "iflow/core/ultimate_consciousness_system_v5.py", 
                "iflow/core/ultimate_consciousness_system_v6.py"
            ],
            "workflow_engine": [
                "iflow/core/ultimate_workflow_engine_v4.py",
                "iflow/core/ultimate_workflow_engine_v6.py"
            ],
            "arq_engine": [
                "iflow/core/ultimate_arq_engine_v4.py",
                "iflow/core/ultimate_arq_engine_v5.py",
                "iflow/core/ultimate_arq_engine_v6.py"
            ],
            "llm_adapter": [
                "iflow/adapters/universal_llm_adapter_v11.py",
                "iflow/adapters/universal_llm_adapter_v12.py", 
                "iflow/adapters/universal_llm_adapter_v13.py",
                "iflow/adapters/ultimate_llm_adapter_v14.py"
            ]
        }
    }
    return duplicates

def analyze_file_versions(duplicates):
    """分析文件版本，确定保留哪个"""
    retention_plan = {
        "keep": [],
        "delete": [],
        "analyze": []  # 需要进一步分析的
    }
    
    # Hook管理器 - 保留最新版本
    hook_files = duplicates["hook_managers"]
    hook_files.sort()
    retention_plan["keep"].append(hook_files[-1])  # 保留最后一个（最新）
    retention_plan["delete"].extend(hook_files[:-1])
    
    # 质量检查 - 保留最新版本
    quality_files = duplicates["auto_quality_checks"]
    quality_files.sort()
    retention_plan["keep"].append(quality_files[-1])
    retention_plan["delete"].extend(quality_files[:-1])
    
    # 清理脚本 - 只保留一个最完整的
    cleanup_files = duplicates["cleanup_scripts"]
    # 保留 aggressive_cleanup_plan.py 和 intelligent_cleanup.py
    retention_plan["keep"].extend([
        "iflow/scripts/intelligent_cleanup.py"
    ])
    retention_plan["delete"].extend([f for f in cleanup_files if f != "iflow/scripts/intelligent_cleanup.py"])
    
    # 测试文件 - 删除旧版本
    retention_plan["delete"].extend(duplicates["test_versions"])
    
    # 核心组件版本分析
    for component, files in duplicates["ultimate_versions"].items():
        files.sort()
        # 保留最高版本号
        latest = files[-1]
        retention_plan["keep"].append(latest)
        retention_plan["delete"].extend(files[:-1])
        
        print(f"组件 {component}: 保留 {latest}")
    
    return retention_plan

def execute_cleanup(retention_plan):
    """执行清理"""
    deleted_files = []
    errors = []
    
    print("🗑️ 开始执行清理计划...")
    print(f"将删除 {len(retention_plan['delete'])} 个文件")
    print(f"将保留 {len(retention_plan['keep'])} 个文件")
    
    # 显示将要删除的文件
    print("\n📋 将删除的文件:")
    for file_path in retention_plan["delete"]:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            print(f"  - {file_path}")
        else:
            print(f"  - {file_path} (不存在)")
    
    # 显示将要保留的文件
    print("\n✅ 将保留的文件:")
    for file_path in retention_plan["keep"]:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            print(f"  + {file_path}")
        else:
            print(f"  + {file_path} (不存在)")
            retention_plan["analyze"].append(file_path)
    
    # 确认删除
    confirm = input("\n⚠️ 确认执行清理？(输入 'yes' 确认): ")
    if confirm.lower() != 'yes':
        print("❌ 清理已取消")
        return False
    
    # 执行删除
    for file_path in retention_plan["delete"]:
        full_path = PROJECT_ROOT / file_path
        try:
            if full_path.exists():
                # 备份到回收站目录
                trash_dir = PROJECT_ROOT / ".trash"
                trash_dir.mkdir(exist_ok=True)
                
                backup_path = trash_dir / f"{file_path.replace('/', '_').replace('\\', '_')}"
                shutil.move(str(full_path), str(backup_path))
                deleted_files.append(file_path)
                print(f"🗑️ 已移动: {file_path} -> .trash/{backup_path.name}")
        except Exception as e:
            error_msg = f"删除 {file_path} 失败: {e}"
            print(f"❌ {error_msg}")
            errors.append(error_msg)
    
    # 生成清理报告
    generate_cleanup_report(deleted_files, errors, retention_plan)
    
    return len(errors) == 0

def generate_cleanup_report(deleted_files, errors, retention_plan):
    """生成清理报告"""
    report = {
        "cleanup_date": "2025-11-13",
        "strategy": "渐进式激进清理",
        "deleted_files": deleted_files,
        "kept_files": retention_plan["keep"],
        "errors": errors,
        "summary": {
            "total_deleted": len(deleted_files),
            "total_kept": len(retention_plan["keep"]),
            "errors_count": len(errors)
        }
    }
    
    # 保存报告
    report_file = PROJECT_ROOT / "清理报告_20251113.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 清理报告已保存到: {report_file}")

def main():
    """主函数"""
    print("🧹 开始A项目渐进式清理")
    print("=" * 50)
    
    # 分析重复文件
    duplicates = get_duplicate_files_analysis()
    print(f"🔍 发现 {len(duplicates)} 类重复文件")
    
    # 分析版本并制定保留计划
    retention_plan = analyze_file_versions(duplicates)
    
    # 执行清理
    success = execute_cleanup(retention_plan)
    
    if success:
        print("\n✅ 清理完成！项目结构已优化")
    else:
        print("\n⚠️ 清理完成但有错误，请查看报告")
    
    return success

if __name__ == "__main__":
    main()