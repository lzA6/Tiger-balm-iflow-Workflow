#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能版本分析器 - 分析各版本功能，择优保留并合并
你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import hashlib

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class SmartVersionAnalyzer:
    """智能版本分析器"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.analysis_results = {}
        self.merge_recommendations = {}
        
    def scan_version_groups(self) -> Dict[str, List[str]]:
        """扫描版本组"""
        version_patterns = [
            r'(.+?)(_v|-v)(\d+)\.py$',
            r'(.+?)(v\d+)\.py$',
            r'(.+?)(_v|-v)(\d+)\.md$'
        ]
        
        version_groups = {}
        
        # 递归扫描所有Python和Markdown文件
        for file_path in self.project_root.rglob('*.py'):
            relative_path = file_path.relative_to(self.project_root)
            for pattern in version_patterns:
                match = re.match(pattern, file_path.name)
                if match:
                    base_name = match.group(1)
                    version_key = f"{relative_path.parent}/{base_name}"
                    if version_key not in version_groups:
                        version_groups[version_key] = []
                    version_groups[version_key].append(str(relative_path))
                    break
        
        # 扫描Markdown文件
        for file_path in self.project_root.rglob('*.md'):
            relative_path = file_path.relative_to(self.project_root)
            for pattern in version_patterns:
                match = re.match(pattern, file_path.name)
                if match:
                    base_name = match.group(1)
                    version_key = f"{relative_path.parent}/{base_name}"
                    if version_key not in version_groups:
                        version_groups[version_key] = []
                    version_groups[version_key].append(str(relative_path))
                    break
        
        return version_groups
    
    def analyze_file_content(self, file_path: str) -> Dict[str, Any]:
        """分析文件内容"""
        full_path = self.project_root / file_path
        if not full_path.exists():
            return {"error": "文件不存在"}
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return {"error": f"读取文件失败: {e}"}
        
        # 计算文件特征
        features = {
            "file_size": len(content),
            "line_count": len(content.splitlines()),
            "function_count": len(re.findall(r'^\s*def\s+\w+', content, re.MULTILINE)),
            "class_count": len(re.findall(r'^\s*class\s+\w+', content, re.MULTILINE)),
            "import_count": len(re.findall(r'^\s*import\s+|^from\s+.*import', content, re.MULTILINE)),
            "comment_count": len(re.findall(r'#.*$|""".*?"""|\'\'\'.*?\'\'\'', content, re.DOTALL)),
            "version_info": self.extract_version_info(content),
            "last_modified": full_path.stat().st_mtime,
            "content_hash": hashlib.md5(content.encode()).hexdigest(),
            "complexity_score": self.calculate_complexity(content),
            "feature_keywords": self.extract_feature_keywords(content)
        }
        
        return features
    
    def extract_version_info(self, content: str) -> Dict[str, str]:
        """提取版本信息"""
        version_patterns = [
            r'version\s*[=:]\s*["\']?(\d+\.\d+(?:\.\d+)?)["\']?',
            r'__version__\s*=\s*["\']?(\d+\.\d+(?:\.\d+)?)["\']?',
            r'#\s*版本\s*:?(\d+\.\d+(?:\.\d+)?)',
            r'#\s*v?ersion\s*:?(\d+)',
        ]
        
        version_info = {}
        for pattern in version_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                version_info[pattern] = matches[0]
        
        return version_info
    
    def calculate_complexity(self, content: str) -> float:
        """计算代码复杂度"""
        # 简单的复杂度计算
        complexity_factors = {
            "if_count": len(re.findall(r'\bif\b', content)),
            "for_count": len(re.findall(r'\bfor\b', content)),
            "while_count": len(re.findall(r'\bwhile\b', content)),
            "try_count": len(re.findall(r'\btry\b', content)),
            "except_count": len(re.findall(r'\bexcept\b', content)),
            "function_def_count": len(re.findall(r'\bdef\s+\w+', content)),
            "class_def_count": len(re.findall(r'\bclass\s+\w+', content)),
        }
        
        # 加权计算复杂度
        weights = {
            "if_count": 1,
            "for_count": 2,
            "while_count": 2,
            "try_count": 1,
            "except_count": 2,
            "function_def_count": 0.5,
            "class_def_count": 1,
        }
        
        total_complexity = sum(
            complexity_factors.get(key, 0) * weight
            for key, weight in weights.items()
        )
        
        # 归一化到0-10范围
        normalized_complexity = min(total_complexity / 10, 10.0)
        return round(normalized_complexity, 2)
    
    def extract_feature_keywords(self, content: str) -> List[str]:
        """提取功能关键词"""
        keywords = [
            "ARQ", "consciousness", "workflow", "adapter", "agent", 
            "quantum", "optimization", "security", "performance",
            "cache", "hook", "test", "analysis", "evolution"
        ]
        
        found_keywords = []
        content_lower = content.lower()
        for keyword in keywords:
            if keyword.lower() in content_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def compare_versions(self, file_group: List[str]) -> Dict[str, Any]:
        """比较版本"""
        analysis = {}
        for file_path in file_group:
            analysis[file_path] = self.analyze_file_content(file_path)
        
        # 找出最佳版本
        best_file = None
        best_score = 0
        
        for file_path, features in analysis.items():
            if "error" in features:
                continue
                
            # 计算综合评分
            score = (
                features["line_count"] * 0.1 +  # 代码量
                features["function_count"] * 2 +  # 函数数量
                features["class_count"] * 3 +  # 类数量
                features["complexity_score"] * 5 +  # 复杂度权重
                len(features["feature_keywords"]) * 2 +  # 功能关键词
                features["import_count"] * 0.5  # 导入数量
            )
            
            # 版本号权重
            version_match = re.search(r'v(\d+)', file_path)
            if version_match:
                score += int(version_match.group(1)) * 2
            
            if score > best_score:
                best_score = score
                best_file = file_path
        
        return {
            "analysis": analysis,
            "best_file": best_file,
            "best_score": best_score,
            "recommendation": self.generate_recommendation(file_group, analysis, best_file)
        }
    
    def generate_recommendation(self, file_group: List[str], analysis: Dict, best_file: str) -> Dict[str, Any]:
        """生成推荐"""
        if not best_file:
            return {"action": "manual_review", "reason": "无法确定最佳版本"}
        
        recommendations = []
        for file_path in file_group:
            if file_path == best_file:
                recommendations.append({
                    "file": file_path,
                    "action": "keep",
                    "reason": "综合评分最高"
                })
            else:
                # 检查是否有独特功能
                unique_features = self.find_unique_features(file_path, best_file, analysis)
                if unique_features:
                    recommendations.append({
                        "file": file_path,
                        "action": "merge",
                        "reason": f"包含独特功能: {', '.join(unique_features)}"
                    })
                else:
                    recommendations.append({
                        "file": file_path,
                        "action": "delete",
                        "reason": "功能已被最佳版本包含"
                    })
        
        return {
            "recommendations": recommendations,
            "merge_suggestions": self.generate_merge_suggestions(file_group, analysis, best_file)
        }
    
    def find_unique_features(self, file1: str, file2: str, analysis: Dict) -> List[str]:
        """查找独特功能"""
        if file1 not in analysis or file2 not in analysis:
            return []
        
        features1 = analysis[file1].get("feature_keywords", [])
        features2 = analysis[file2].get("feature_keywords", [])
        
        # 找出file1有但file2没有的功能
        unique_features = [f for f in features1 if f not in features2]
        return unique_features
    
    def generate_merge_suggestions(self, file_group: List[str], analysis: Dict, best_file: str) -> List[Dict[str, Any]]:
        """生成合并建议"""
        merge_suggestions = []
        
        for file_path in file_group:
            if file_path == best_file or file_path not in analysis:
                continue
            
            features = analysis[file_path]
            if "error" in features:
                continue
            
            # 简单的内容比较
            try:
                with open(self.project_root / file_path, 'r', encoding='utf-8') as f:
                    content1 = f.read()
                with open(self.project_root / best_file, 'r', encoding='utf-8') as f:
                    content2 = f.read()
                
                # 找出差异
                diff_analysis = self.analyze_content_differences(content1, content2)
                if diff_analysis["has_unique_content"]:
                    merge_suggestions.append({
                        "source_file": file_path,
                        "target_file": best_file,
                        "differences": diff_analysis["differences"],
                        "merge_strategy": "manual_review"
                    })
            except Exception as e:
                merge_suggestions.append({
                    "source_file": file_path,
                    "error": str(e)
                })
        
        return merge_suggestions
    
    def analyze_content_differences(self, content1: str, content2: str) -> Dict[str, Any]:
        """分析内容差异"""
        lines1 = set(content1.splitlines())
        lines2 = set(content2.splitlines())
        
        unique_to_1 = lines1 - lines2
        unique_to_2 = lines2 - lines1
        
        has_unique = len(unique_to_1) > 0 or len(unique_to_2) > 0
        
        return {
            "has_unique_content": has_unique,
            "differences": {
                "unique_to_source": list(unique_to_1),
                "unique_to_target": list(unique_to_2),
                "unique_count_source": len(unique_to_1),
                "unique_count_target": len(unique_to_2)
            }
        }
    
    def run_analysis(self) -> Dict[str, Any]:
        """运行完整分析"""
        print("[ANALYZER] 开始智能版本分析...")
        
        # 扫描版本组
        version_groups = self.scan_version_groups()
        print(f"[INFO] 发现 {len(version_groups)} 个版本组")
        
        analysis_results = {}
        for base_name, file_list in version_groups.items():
            print(f"\n[GROUP] 分析版本组: {base_name}")
            print(f"   文件列表: {', '.join(file_list)}")
            
            comparison = self.compare_versions(file_list)
            analysis_results[base_name] = comparison
            
            # 输出推荐
            recommendation = comparison.get("recommendation", {})
            if "recommendations" in recommendation:
                for rec in recommendation["recommendations"]:
                    action_symbol = "[KEEP]" if rec["action"] == "keep" else "[MERGE]" if rec["action"] == "merge" else "[DELETE]"
                    print(f"   {action_symbol} {rec['file']} -> {rec['action']}: {rec['reason']}")
        
        # 生成总结报告
        summary = self.generate_summary(analysis_results)
        
        return {
            "version_groups": version_groups,
            "analysis_results": analysis_results,
            "summary": summary
        }
    
    def generate_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成总结"""
        total_files = 0
        keep_count = 0
        merge_count = 0
        delete_count = 0
        
        for base_name, result in analysis_results.items():
            recommendation = result.get("recommendation", {})
            if "recommendations" in recommendation:
                for rec in recommendation["recommendations"]:
                    total_files += 1
                    if rec["action"] == "keep":
                        keep_count += 1
                    elif rec["action"] == "merge":
                        merge_count += 1
                    elif rec["action"] == "delete":
                        delete_count += 1
        
        return {
            "total_version_groups": len(analysis_results),
            "total_files_analyzed": total_files,
            "files_to_keep": keep_count,
            "files_to_merge": merge_count,
            "files_to_delete": delete_count,
            "estimated_reduction": f"{delete_count}/{total_files} ({delete_count/total_files*100:.1f}%)"
        }

def main():
    """主函数"""
    analyzer = SmartVersionAnalyzer()
    results = analyzer.run_analysis()
    
    # 保存结果
    output_file = PROJECT_ROOT / "智能版本分析报告_20251113.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 输出总结
    summary = results["summary"]
    print(f"\n[SUMMARY] 分析总结:")
    print(f"   版本组数量: {summary['total_version_groups']}")
    print(f"   分析文件数: {summary['total_files_analyzed']}")
    print(f"   保留文件数: {summary['files_to_keep']}")
    print(f"   合并文件数: {summary['files_to_merge']}")
    print(f"   删除文件数: {summary['files_to_delete']}")
    print(f"   预估减少: {summary['estimated_reduction']}")
    
    print(f"\n[REPORT] 详细报告已保存到: {output_file}")

if __name__ == "__main__":
    main()