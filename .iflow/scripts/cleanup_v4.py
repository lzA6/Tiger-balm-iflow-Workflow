#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V4版本清理脚本
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import shutil
from pathlib import Path

def cleanup_old_versions():
    """清理旧版本文件"""
    base_path = Path(r"C:\Users\Administrator.DESKTOP-EGNE9ND\Desktop\测试工作流\A项目\iflow")
    
    # 需要删除的旧版本文件列表
    old_files = [
        "core/ultimate_consciousness_system.py",  # 有v4版本
        "core/ultimate_workflow_engine.py",       # 有v4版本
        "adapters/universal_llm_adapter.py",      # 有v12版本
        "hooks/comprehensive-hook-manager.py",     # 有v4版本
    ]
    
    # 需要删除的__pycache__目录
    pycache_dirs = []
    for root, dirs, files in os.walk(base_path):
        if "__pycache__" in dirs:
            pycache_dirs.append(Path(root) / "__pycache__")
    
    print("🗑️  开始清理旧版本文件...")
    
    # 删除旧文件
    for file_path in old_files:
        full_path = base_path / file_path
        if full_path.exists():
            try:
                # 先备份到backup目录
                backup_dir = base_path / "backup" / "old_versions"
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_path = backup_dir / full_path.name
                shutil.copy2(full_path, backup_path)
                
                # 删除原文件
                full_path.unlink()
                print(f"✅ 已删除: {file_path} (备份到 {backup_path})")
            except Exception as e:
                print(f"❌ 删除失败: {file_path} - {e}")
        else:
            print(f"⚠️  文件不存在: {file_path}")
    
    # 删除__pycache__目录
    for pycache_path in pycache_dirs:
        try:
            shutil.rmtree(pycache_path)
            print(f"✅ 已删除缓存目录: {pycache_path}")
        except Exception as e:
            print(f"❌ 删除缓存失败: {pycache_path} - {e}")
    
    print("\n🧹 清理完成!")

if __name__ == "__main__":
    cleanup_old_versions()