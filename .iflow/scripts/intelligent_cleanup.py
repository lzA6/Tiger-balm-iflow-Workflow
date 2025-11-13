import os
import json
import re
from collections import defaultdict
from pathlib import Path
import datetime
from typing import Dict, List, Any, DefaultDict, Tuple

class IntelligentCleanup:
    """
    一个智能清理工具，用于分析项目结构，识别冗余、过时或可合并的文件，
    并生成一份详细的清理建议报告。
    你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
    """

    def __init__(self, project_root: str):
        self.project_root: Path = Path(project_root)
        self.analysis_results: Dict[str, Any] = {
            "redundant_files": [],
            "versioned_files": defaultdict(list),
            "potential_merges": [],
            "empty_dirs": [],
            "analysis_summary": ""
        }

    def analyze(self) -> None:
        """
        执行全面的项目分析。
        """
        print("Starting comprehensive analysis of the project structure...")
        for root, dirs, files in os.walk(self.project_root):
            # 检查空目录
            if not dirs and not files:
                self.analysis_results["empty_dirs"].append(str(Path(root)))
                continue

            for file in files:
                file_path = Path(root) / file
                self._analyze_file(file_path)

        self._process_versioned_files()
        self._generate_summary()
        print("Analysis complete.")

    def _analyze_file(self, file_path: Path) -> None:
        """
        分析单个文件，检查其是否冗余或为特定版本。
        """
        # 匹配版本号模式, e.g., my-file-v2.py, my-file-v10.py
        version_match = re.match(r"(.+?)(_v|-v)(\d+)(\..+)", file_path.name)
        if version_match:
            base_name = version_match.group(1) + version_match.group(4)
            version = int(version_match.group(3))
            self.analysis_results["versioned_files"][str(file_path.parent / base_name)].append((version, str(file_path)))
            return

        # 匹配时间戳模式, e.g., report_20251113_113759.json
        timestamp_match = re.match(r"(.+?_)([\d_]+)(\.json|\.log|\.md|\.html)", file_path.name)
        if timestamp_match and "ultimate" not in file_path.name: # 排除关键的对比报告
             self.analysis_results["redundant_files"].append({
                "file": str(file_path),
                "reason": "Timestamped artifact file, likely a log or report."
            })
             return

        # 匹配备份文件
        if file_path.name.endswith(('.bak', '_backup.json', '.old')):
            self.analysis_results["redundant_files"].append({
                "file": str(file_path),
                "reason": "Backup file, likely obsolete."
            })

    def _process_versioned_files(self) -> None:
        """
        处理收集到的版本化文件，找出过时的版本。
        """
        for base_path, versions in self.analysis_results["versioned_files"].items():
            if len(versions) > 1:
                versions.sort(key=lambda x: x[0], reverse=True)
                latest_version = versions[0]
                
                for version, file_path in versions[1:]:
                    self.analysis_results["redundant_files"].append({
                        "file": file_path,
                        "reason": f"Outdated version. The latest is version {latest_version[0]} located at {latest_version[1]}."
                    })

    def _generate_summary(self) -> None:
        """
        生成分析总结。
        """
        num_redundant = len(self.analysis_results["redundant_files"])
        num_versioned_groups = len(self.analysis_results["versioned_files"])
        num_empty_dirs = len(self.analysis_results["empty_dirs"])

        summary = (
            f"Intelligent Cleanup Analysis Summary:\n"
            f"=====================================\n"
            f"Project Root: {self.project_root}\n"
            f"Timestamp: {datetime.datetime.now().isoformat()}\n\n"
            f"Found {num_redundant} redundant or outdated files recommended for deletion.\n"
            f"Found {num_versioned_groups} groups of versioned files, keeping only the latest.\n"
            f"Found {num_empty_dirs} empty directories that could be removed.\n\n"
            f"This analysis is the first step towards creating a clean, efficient, and maintainable project structure. "
            f"By removing clutter, we pave the way for seamless and intelligent upgrades."
        )
        self.analysis_results["analysis_summary"] = summary

    def get_report(self) -> str:
        """
        返回JSON格式的完整分析报告。
        """
        # 清理用于处理的数据
        del self.analysis_results["versioned_files"]
        return json.dumps(self.analysis_results, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    project_to_analyze = Path(__file__).parent.parent.parent
    cleanup_analyzer = IntelligentCleanup(project_to_analyze)
    cleanup_analyzer.analyze()
    report = cleanup_analyzer.get_report()
    
    report_path = project_to_analyze / "iflow" / "reports" / f"cleanup_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f"\nCleanup analysis report generated at: {report_path}")
    print("\n--- Cleanup Report Summary ---")
    parsed_report = json.loads(report)
    print(parsed_report["analysis_summary"])
    print("\n--- Files to Delete ---")
    for item in parsed_report["redundant_files"]:
        print(f"- {item['file']} (Reason: {item['reason']})")
    print("\n--- Empty Dirs to Remove ---")
    for item in parsed_report["empty_dirs"]:
        print(f"- {item}")
