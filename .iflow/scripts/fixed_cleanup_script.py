#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复后的保守清理脚本
只清理明显的冗余文件，保留所有现有功能
"""

import os
import sys
import json
import shutil
import time
from pathlib import Path
from typing import List, Dict, Set, Optional

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class ConservativeCleanup:
    """保守清理器"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or PROJECT_ROOT
        self.cleanup_log = []
        self.protected_files = set()
        self.setup_protected_files()
    
    def setup_protected_files(self):
        """设置受保护的文件（不清理）"""
        # 受保护的核心文件
        protected_list = [
            # 核心引擎文件
            "ultimate_arq_engine.py",
            "ultimate_consciousness_system.py",
            "male_system.py",
            "dkcm_system.py",
            "rpfv_system.py",
            
            # 关键工具文件
            "intelligent_tool_caller.py",
            "multi_agent_orchestrator.py",
            
            # 重要配置文件
            "settings.json",
            "principles.md",
            "rules.md",
        ]
        self.protected_files = set(protected_list)
    
    def find_duplicate_files(self) -> Dict[str, List[str]]:
        """查找重复文件"""
        duplicates = {}
        
        # 查找版本号重复的文件
        version_patterns = [
            ("README", ["README.md", "README_V4.md"]),
            ("CHANGELOG", ["CHANGELOG.md", "CHANGELOG_V4.md"]),
            ("Hook管理器", ["comprehensive_hook_manager.py", "comprehensive_hook_manager_v4.py", 
                           "comprehensive-hook-manager.py", "comprehensive-hook-manager-v4.py"]),
            ("质量检查", ["auto_quality_check.py", "auto_quality_check_v6.py"]),
        ]
        
        for category, files in version_patterns:
            existing_files = []
            for file_pattern in files:
                file_path = self.project_root / ".iflow" / file_pattern
                if file_path.exists():
                    existing_files.append(str(file_path))
            
            if len(existing_files) > 1:
                duplicates[category] = existing_files
        
        return duplicates
    
    def analyze_file_importance(self, file_path: Path) -> Dict[str, any]:
        """分析文件重要性"""
        if not file_path.exists():
            return {"status": "missing"}
        
        # 检查是否是受保护文件
        if file_path.name in self.protected_files:
            return {"status": "protected", "reason": "核心文件"}
        
        # 检查文件大小和修改时间
        stat = file_path.stat()
        return {
            "status": "analyzable",
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "is_recent": (stat.st_mtime > (time.time() - 86400 * 30))  # 30天内
        }
    
    def should_keep_file(self, file_path: Path, duplicates: List[str]) -> bool:
        """判断是否保留文件"""
        analysis = self.analyze_file_importance(file_path)
        
        if analysis["status"] == "protected":
            return True
        
        if analysis["status"] == "missing":
            return False
        
        # 如果是最新的文件，保留
        if analysis.get("is_recent", False):
            return True
        
        # 如果文件较大（可能包含更多内容），保留
        if analysis.get("size", 0) > 10000:  # 10KB
            return True
        
        # 如果是唯一存在的文件，保留
        if len(duplicates) == 1:
            return True
        
        return False
    
    def cleanup_duplicates(self) -> Dict[str, any]:
        """清理重复文件"""
        results = {
            "kept_files": [],
            "removed_files": [],
            "errors": [],
            "summary": {}
        }
        
        duplicates = self.find_duplicate_files()
        
        for category, file_list in duplicates.items():
            if len(file_list) <= 1:
                continue
            
            kept = None
            removed = []
            
            for file_path_str in file_list:
                file_path = Path(file_path_str)
                
                if self.should_keep_file(file_path, file_list):
                    if not kept:
                        kept = file_path_str
                        results["kept_files"].append({
                            "file": file_path_str,
                            "reason": "保留最新/最重要的版本",
                            "category": category
                        })
                        print(f"✅ 保留: {file_path_str} (类别: {category})")
                    else:
                        # 已经有一个保留的文件，这个标记为删除
                        removed.append(file_path_str)
                else:
                    removed.append(file_path_str)
            
            # 备份并删除冗余文件
            for remove_path in removed:
                try:
                    backup_path = remove_path + ".backup"
                    shutil.move(remove_path, backup_path)
                    results["removed_files"].append({
                        "file": remove_path,
                        "backup": backup_path,
                        "category": category
                    })
                    print(f"🗑️  备份并删除: {remove_path} -> {backup_path}")
                except Exception as e:
                    error_msg = f"删除失败: {remove_path} - {e}"
                    results["errors"].append(error_msg)
                    print(f"❌ {error_msg}")
            
            results["summary"][category] = {
                "original_count": len(file_list),
                "kept_count": 1 if kept else 0,
                "removed_count": len(removed)
            }
        
        return results
    
    def cleanup_temp_files(self) -> Dict[str, any]:
        """清理临时文件"""
        temp_patterns = [
            "*.tmp",
            "*.temp",
            "*.log.bak",
            "temp_delete.py",
            "*.pyc",
            "__pycache__"
        ]
        
        results = {
            "removed_files": [],
            "errors": []
        }
        
        for pattern in temp_patterns:
            for file_path in self.project_root.glob(f"**/{pattern}"):
                if file_path.is_file():
                    try:
                        backup_path = str(file_path) + ".backup"
                        shutil.move(str(file_path), backup_path)
                        results["removed_files"].append({
                            "file": str(file_path),
                            "backup": backup_path
                        })
                        print(f"🗑️  清理临时文件: {file_path}")
                    except Exception as e:
                        error_msg = f"清理临时文件失败: {file_path} - {e}"
                        results["errors"].append(error_msg)
                        print(f"❌ {error_msg}")
        
        return results
    
    def generate_cleanup_report(self) -> Dict[str, any]:
        """生成清理报告"""
        report = {
            "timestamp": time.time(),
            "project_root": str(self.project_root),
            "duplicate_cleanup": self.cleanup_duplicates(),
            "temp_cleanup": self.cleanup_temp_files(),
            "recommendations": []
        }
        
        # 添加建议
        report["recommendations"] = [
            "✅ 已完成保守清理，只删除了明显的重复文件",
            "📦 所有删除的文件都已备份，扩展名为.backup",
            "📁 建议后续可以手动检查备份文件，确认无误后再删除",
            "📝 建议更新版本控制系统，提交清理后的结果"
        ]
        
        return report
    
    def save_cleanup_report(self, report: Dict[str, any]):
        """保存清理报告"""
        report_path = self.project_root / ".iflow" / "conservative_cleanup_report.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 清理报告已保存: {report_path}")

def main():
    """主函数"""
    print("🚀 开始保守清理A项目...")
    print("=" * 50)
    
    cleanup = ConservativeCleanup()
    report = cleanup.generate_cleanup_report()
    cleanup.save_cleanup_report(report)
    
    print("\n" + "=" * 50)
    print("📊 清理总结:")
    
    # 统计结果
    total_removed = 0
    total_kept = 0
    
    for category, summary in report["duplicate_cleanup"]["summary"].items():
        print(f"  {category}: 原{summary['original_count']}个 -> 保留{summary['kept_count']}个 -> 删除{summary['removed_count']}个")
        total_removed += summary["removed_count"]
        total_kept += summary["kept_count"]
    
    print(f"\n🗑️  总计删除文件: {total_removed} 个")
    print(f"✅ 总计保留文件: {total_kept} 个")
    
    if report["temp_cleanup"]["removed_files"]:
        print(f"🧹 清理临时文件: {len(report['temp_cleanup']['removed_files'])} 个")
    
    if report["duplicate_cleanup"]["errors"]:
        print(f"⚠️  清理错误: {len(report['duplicate_cleanup']['errors'])} 个")
    
    print("\n💡 清理完成！所有删除的文件都已备份。")
    print("   建议检查备份文件确认无误后，再手动删除.backup文件。")

if __name__ == "__main__":
    main()