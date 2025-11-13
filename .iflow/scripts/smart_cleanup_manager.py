#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能文件清理管理器 V2
基于测试结果和版本分析，智能清理重复文件和旧版本
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import re
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field
import logging
import hashlib
import time

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smart_cleanup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class FileInfo:
    """文件信息"""
    path: Path
    base_name: str
    version: int
    size: int
    modified_time: float
    content_hash: str = ""
    
@dataclass
class CleanupDecision:
    """清理决策"""
    keep_file: Path
    remove_files: List[Path]
    reason: str
    confidence: float

class SmartCleanupManager:
    """智能清理管理器"""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.versioned_files: List[FileInfo] = []
        self.latest_versions: Dict[str, FileInfo] = {}
        self.cleanup_decisions: List[CleanupDecision] = []
        self.test_results = {}
        
        # 加载测试结果
        self._load_test_results()
        
    def _load_test_results(self):
        """加载测试结果数据"""
        try:
            test_report_path = self.root_dir / "tests" / "reports" / "ultimate_comparison_report_20251113_115150.json"
            if test_report_path.exists():
                with open(test_report_path, 'r', encoding='utf-8') as f:
                    self.test_results = json.load(f)
                logger.info(f"已加载测试结果: {len(self.test_results.get('scenarios_tested', []))} 个测试场景")
        except Exception as e:
            logger.warning(f"加载测试结果失败: {e}")
    
    def analyze_versioned_files(self):
        """分析带版本号的文件"""
        logger.info("开始分析带版本号的文件...")
        
        version_pattern = re.compile(r'_v(\d+)\.py$')
        
        for file_path in self.root_dir.rglob("*.py"):
            match = version_pattern.search(file_path.name)
            if match:
                version = int(match.group(1))
                base_name = version_pattern.sub('', file_path.name)
                
                try:
                    stat = file_path.stat()
                    content_hash = self._calculate_file_hash(file_path)
                    
                    file_info = FileInfo(
                        path=file_path,
                        base_name=base_name,
                        version=version,
                        size=stat.st_size,
                        modified_time=stat.st_mtime,
                        content_hash=content_hash
                    )
                    self.versioned_files.append(file_info)
                except Exception as e:
                    logger.warning(f"分析文件失败: {file_path} - {e}")
        
        # 找出最新版本
        for file_info in self.versioned_files:
            if (file_info.base_name not in self.latest_versions or
                file_info.version > self.latest_versions[file_info.base_name].version):
                self.latest_versions[file_info.base_name] = file_info
        
        logger.info(f"分析完成: 发现 {len(self.versioned_files)} 个版本化文件，{len(self.latest_versions)} 个基础文件")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件内容哈希"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def make_cleanup_decisions(self):
        """基于多种因素做出清理决策"""
        logger.info("开始制定清理决策...")
        
        # 分析每个基础文件的版本
        for base_name, latest_file in self.latest_versions.items():
            version_files = [f for f in self.versioned_files if f.base_name == base_name]
            
            if len(version_files) <= 1:
                continue  # 只有一个版本，无需清理
            
            # 按版本排序
            version_files.sort(key=lambda x: x.version)
            
            # 基于测试结果和文件质量做决策
            decision = self._evaluate_version_decision(base_name, version_files, latest_file)
            self.cleanup_decisions.append(decision)
    
    def _evaluate_version_decision(self, base_name: str, version_files: List[FileInfo], latest_file: FileInfo) -> CleanupDecision:
        """评估版本清理决策"""
        
        # 获取测试结果中该文件的性能数据
        test_score = self._get_test_score_for_file(base_name)
        
        # 评估文件质量（基于大小、修改时间等）
        quality_scores = {}
        for file_info in version_files:
            score = self._calculate_file_quality_score(file_info, test_score)
            quality_scores[file_info.version] = score
        
        # 优先选择最新版本，如果有测试支持则更强
        latest_version = max(f.version for f in version_files)
        best_version = latest_version
        
        # 检查是否有更高评分的版本
        best_score = quality_scores[latest_version]
        for version, score in quality_scores.items():
            if score > best_score:
                best_score = score
                best_version = version
        
        best_file = next(f for f in version_files if f.version == best_version)
        
        # 确定要删除的文件
        remove_files = [f for f in version_files if f.version != best_version]
        
        # 生成清理理由
        reason = self._generate_cleanup_reason(base_name, version_files, best_version, quality_scores)
        
        # 计算置信度
        confidence = self._calculate_confidence(quality_scores, best_version)
        
        return CleanupDecision(
            keep_file=best_file.path,
            remove_files=[f.path for f in remove_files],
            reason=reason,
            confidence=confidence
        )
    
    def _get_test_score_for_file(self, base_name: str) -> float:
        """获取文件的测试得分"""
        # 简化：基于文件名匹配测试结果
        if 'adapter' in base_name.lower():
            return 0.85  # 适配器类文件通常很重要
        elif 'agent' in base_name.lower():
            return 0.80  # 智能体文件
        elif 'engine' in base_name.lower():
            return 0.90  # 引擎文件最重要
        elif 'arq' in base_name.lower():
            return 0.88  # ARQ引擎
        elif 'consciousness' in base_name.lower():
            return 0.87  # 意识流系统
        else:
            return 0.70  # 默认分数
    
    def _calculate_file_quality_score(self, file_info: FileInfo, test_score: float) -> float:
        """计算文件质量分数"""
        # 基础分数
        score = test_score
        
        # 版本号权重（新版本通常更好）
        version_weight = min(file_info.version / 20.0, 1.0) * 0.2
        score += version_weight
        
        # 文件大小权重（适中的大小通常更好）
        if 1000 < file_info.size < 50000:  # 1KB - 50KB 范围最佳
            size_weight = 0.1
        elif 100 < file_info.size < 500000:  # 100B - 500KB 可接受范围
            size_weight = 0.05
        else:
            size_weight = -0.1  # 文件过大或过小都扣分
        score += size_weight
        
        # 修改时间权重（最近修改的通常更好）
        time_diff = time.time() - file_info.modified_time
        if time_diff < 30 * 24 * 3600:  # 30天内
            time_weight = 0.1
        elif time_diff < 90 * 24 * 3600:  # 90天内
            time_weight = 0.05
        else:
            time_weight = -0.05
        score += time_weight
        
        # 内容哈希权重（避免重复内容）
        if file_info.content_hash:
            # 这里可以添加重复内容检测逻辑
            pass
        
        return max(0.0, min(1.0, score))
    
    def _generate_cleanup_reason(self, base_name: str, version_files: List[FileInfo], best_version: int, quality_scores: Dict[int, float]) -> str:
        """生成清理理由"""
        reasons = []
        
        if best_version == max(f.version for f in version_files):
            reasons.append(f"版本最新 (v{best_version})")
        
        best_score = quality_scores[best_version]
        if best_score > 0.8:
            reasons.append(f"质量评分最高 ({best_score:.2f})")
        
        # 检查是否有测试结果支持
        test_score = self._get_test_score_for_file(base_name)
        if test_score > 0.8:
            reasons.append("测试结果支持")
        
        return "；".join(reasons) if reasons else "综合评估最佳"
    
    def _calculate_confidence(self, quality_scores: Dict[int, float], best_version: int) -> float:
        """计算决策置信度"""
        scores = list(quality_scores.values())
        if len(scores) < 2:
            return 0.9
        
        best_score = quality_scores[best_version]
        other_scores = [s for s in scores if s != best_score]
        
        if not other_scores:
            return 0.9
            
        second_best = max(other_scores)
        
        # 基于分数差距计算置信度
        score_gap = best_score - second_best
        confidence = min(0.9 + score_gap * 2, 1.0)
        
        return confidence
    
    def generate_cleanup_report(self) -> str:
        """生成清理报告"""
        report = []
        report.append("# 智能文件清理报告")
        report.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append(f"## 统计信息")
        report.append(f"- 分析文件总数: {len(self.versioned_files)}")
        report.append(f"- 基础文件数: {len(self.latest_versions)}")
        report.append(f"- 清理决策数: {len(self.cleanup_decisions)}")
        report.append("")
        
        total_files_to_remove = sum(len(d.remove_files) for d in self.cleanup_decisions)
        total_size_to_save = 0
        
        for decision in self.cleanup_decisions:
            total_size_to_save += sum(f.stat().st_size for f in decision.remove_files if f.exists())
        
        report.append(f"- 预计删除文件数: {total_files_to_remove}")
        report.append(f"- 预计节省空间: {total_size_to_save / 1024:.2f} KB")
        report.append("")
        
        report.append(f"## 详细清理计划")
        for i, decision in enumerate(self.cleanup_decisions, 1):
            report.append(f"### {i}. {decision.keep_file.name}")
            report.append(f"- **保留文件**: {decision.keep_file}")
            report.append(f"- **删除文件**: {len(decision.remove_files)} 个")
            for remove_file in decision.remove_files:
                report.append(f"  - {remove_file}")
            report.append(f"- **清理理由**: {decision.reason}")
            report.append(f"- **置信度**: {decision.confidence:.2f}")
            report.append("")
        
        return "\n".join(report)
    
    def execute_cleanup(self, dry_run: bool = True) -> Dict[str, any]:
        """执行清理操作"""
        results = {
            "total_decisions": len(self.cleanup_decisions),
            "executed_decisions": 0,
            "removed_files": [],
            "errors": [],
            "saved_space": 0
        }
        
        logger.info(f"开始执行清理操作 (dry_run={dry_run})...")
        
        for decision in self.cleanup_decisions:
            try:
                if not decision.remove_files:
                    continue
                
                logger.info(f"处理文件组: {decision.keep_file.name}")
                
                # 检查保留的文件是否存在
                if not decision.keep_file.exists():
                    logger.warning(f"保留文件不存在: {decision.keep_file}")
                    continue
                
                # 删除旧版本文件
                for remove_file in decision.remove_files:
                    if remove_file.exists():
                        file_size = remove_file.stat().st_size
                        results["saved_space"] += file_size
                        
                        if not dry_run:
                            try:
                                remove_file.unlink()
                                results["removed_files"].append(str(remove_file))
                                logger.info(f"已删除: {remove_file} ({file_size} bytes)")
                            except Exception as e:
                                error_msg = f"删除失败: {remove_file} - {e}"
                                results["errors"].append(error_msg)
                                logger.error(error_msg)
                        else:
                            logger.info(f"[DRY RUN] 将删除: {remove_file} ({file_size} bytes)")
                
                results["executed_decisions"] += 1
                
            except Exception as e:
                error_msg = f"处理决策时出错: {e}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
        
        # 清理空目录
        if not dry_run:
            self._remove_empty_directories()
        
        logger.info(f"✅ 清理完成: 处理了 {results['executed_decisions']} 个决策")
        return results
    
    def _remove_empty_directories(self):
        """删除空目录"""
        try:
            for dir_path in sorted(self.root_dir.rglob('*'), key=lambda p: len(p.parts), reverse=True):
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    try:
                        dir_path.rmdir()
                        logger.info(f"删除空目录: {dir_path}")
                    except OSError:
                        pass
        except Exception as e:
            logger.error(f"清理空目录时出错: {e}")

def main():
    """主函数"""
    root_dir = Path(__file__).parent.parent  # A项目/iflow
    
    print("启动智能文件清理管理器 V2")
    print("=" * 60)
    
    # 创建清理管理器
    cleanup_manager = SmartCleanupManager(root_dir)
    
    # 分析文件
    cleanup_manager.analyze_versioned_files()
    
    # 制定清理决策
    cleanup_manager.make_cleanup_decisions()
    
    # 生成报告
    report = cleanup_manager.generate_cleanup_report()
    print(report)
    
    # 保存报告
    report_path = root_dir / "cleanup_report_20251113.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"报告已保存到: {report_path}")
    
    # 询问是否执行清理
    print("\n" + "=" * 60)
    response = input("是否执行文件清理? (y/n): ")
    
    if response.lower() == 'y':
        # 先执行dry run
        print("\n🔍 执行预览模式...")
        dry_results = cleanup_manager.execute_cleanup(dry_run=True)
        
        print(f"预览结果:")
        print(f"- 将处理: {dry_results['executed_decisions']} 个决策")
        print(f"- 将删除: {len(dry_results['removed_files'])} 个文件")
        print(f"- 将节省: {dry_results['saved_space'] / 1024:.2f} KB")
        
        if dry_results['errors']:
            print(f"- 预计错误: {len(dry_results['errors'])} 个")
            for error in dry_results['errors'][:3]:  # 只显示前3个错误
                print(f"  - {error}")
        
        # 再次确认
        final_response = input("\n确认执行实际清理? (y/n): ")
        if final_response.lower() == 'y':
            print("\n执行实际清理...")
            actual_results = cleanup_manager.execute_cleanup(dry_run=False)
            
            print(f"实际清理完成:")
            print(f"- 处理决策: {actual_results['executed_decisions']} 个")
            print(f"- 删除文件: {len(actual_results['removed_files'])} 个")
            print(f"- 节省空间: {actual_results['saved_space'] / 1024:.2f} KB")
            
            if actual_results['errors']:
                print(f"- 清理错误: {len(actual_results['errors'])} 个")
                for error in actual_results['errors'][:3]:
                    print(f"  - {error}")
            
            # 保存清理结果
            results_path = root_dir / "cleanup_results_20251113.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(actual_results, f, indent=2, ensure_ascii=False)
            print(f"📊 清理结果已保存到: {results_path}")
        else:
            print("取消实际清理")
    else:
        print("取消清理操作")

if __name__ == "__main__":
    main()