#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动代码生成和补全系统
Automatic Code Generation and Completion System

作者: Quantum AI Team
版本: 5.2.0
日期: 2025-11-12
"""

import os
import re
import ast
import json
import time
import hashlib
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
from collections import defaultdict, Counter
import difflib
import subprocess
import sys

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeLanguage(Enum):
    """编程语言枚举"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    JSON = "json"
    YAML = "yaml"
    MARKDOWN = "markdown"

class CodeComplexity(Enum):
    """代码复杂度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class CodePattern:
    """代码模式"""
    pattern_id: str
    language: CodeLanguage
    pattern_type: str  # function, class, module, etc.
    template: str
    description: str
    variables: List[str]
    complexity: CodeComplexity
    tags: List[str]
    usage_count: int = 0
    success_rate: float = 1.0

@dataclass
class CodeSuggestion:
    """代码建议"""
    suggestion_id: str
    language: CodeLanguage
    context: str
    suggested_code: str
    confidence: float
    explanation: str
    pattern_id: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class GenerationRequest:
    """代码生成请求"""
    request_id: str
    language: CodeLanguage
    description: str
    context: Optional[str]
    requirements: List[str]
    constraints: List[str]
    complexity: CodeComplexity
    style_preferences: Dict[str, Any]

class CodePatternDatabase:
    """代码模式数据库"""
    
    def __init__(self, db_path: Optional[str] = None):
        """初始化模式数据库"""
        self.db_path = db_path or "code_patterns.json"
        self.patterns = {}
        self.pattern_index = defaultdict(list)
        self.load_patterns()
        
        # 初始化内置模式
        self._initialize_builtin_patterns()
    
    def load_patterns(self):
        """加载模式"""
        if Path(self.db_path).exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for pattern_data in data.get('patterns', []):
                        pattern = CodePattern(**pattern_data)
                        self.patterns[pattern.pattern_id] = pattern
                        self._index_pattern(pattern)
                logger.info(f"📚 加载了 {len(self.patterns)} 个代码模式")
            except Exception as e:
                logger.error(f"❌ 加载代码模式失败: {e}")
    
    def save_patterns(self):
        """保存模式"""
        data = {
            'patterns': [asdict(pattern) for pattern in self.patterns.values()],
            'last_updated': time.time()
        }
        
        try:
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 保存了 {len(self.patterns)} 个代码模式")
        except Exception as e:
            logger.error(f"❌ 保存代码模式失败: {e}")
    
    def _index_pattern(self, pattern: CodePattern):
        """索引模式"""
        # 按语言索引
        self.pattern_index['language'][pattern.language.value].append(pattern.pattern_id)
        
        # 按类型索引
        self.pattern_index['type'][pattern.pattern_type].append(pattern.pattern_id)
        
        # 按复杂度索引
        self.pattern_index['complexity'][pattern.complexity.value].append(pattern.pattern_id)
        
        # 按标签索引
        for tag in pattern.tags:
            self.pattern_index['tag'][tag].append(pattern.pattern_id)
    
    def _initialize_builtin_patterns(self):
        """初始化内置模式"""
        builtin_patterns = [
            # Python 函数模式
            CodePattern(
                pattern_id="python_function_basic",
                language=CodeLanguage.PYTHON,
                pattern_type="function",
                template="""def {function_name}({parameters}):
    """
    {description}
    
    Args:
        {args_doc}
    
    Returns:
        {return_doc}
    """
    {body}""",
                description="基础Python函数模板",
                variables=["function_name", "parameters", "description", "args_doc", "return_doc", "body"],
                complexity=CodeComplexity.LOW,
                tags=["function", "basic", "documentation"]
            ),
            
            # Python 类模式
            CodePattern(
                pattern_id="python_class_basic",
                language=CodeLanguage.PYTHON,
                pattern_type="class",
                template="""class {class_name}:
    """
    {description}
    """
    
    def __init__(self{init_params}):
        {init_body}
    
    def {method_name}(self{method_params}):
        """
        {method_description}
        """
        {method_body}""",
                description="基础Python类模板",
                variables=["class_name", "description", "init_params", "init_body", "method_name", "method_params", "method_description", "method_body"],
                complexity=CodeComplexity.MEDIUM,
                tags=["class", "basic", "oop"]
            ),
            
            # JavaScript 函数模式
            CodePattern(
                pattern_id="javascript_function_arrow",
                language=CodeLanguage.JAVASCRIPT,
                pattern_type="function",
                template="""const {function_name} = ({parameters}) => {{
    {description}
    
    {body}
}};""",
                description="JavaScript箭头函数模板",
                variables=["function_name", "parameters", "description", "body"],
                complexity=CodeComplexity.LOW,
                tags=["function", "arrow", "es6"]
            ),
            
            # React 组件模式
            CodePattern(
                pattern_id="react_component_functional",
                language=CodeLanguage.JAVASCRIPT,
                pattern_type="component",
                template="""import React from 'react';

{imports}

const {component_name} = ({props}) => {{
    {description}
    
    return (
        <div className="{class_name}">
            {jsx_content}
        </div>
    );
}};

export default {component_name};""",
                description="React函数组件模板",
                variables=["imports", "component_name", "props", "description", "class_name", "jsx_content"],
                complexity=CodeComplexity.MEDIUM,
                tags=["react", "component", "functional", "jsx"]
            ),
            
            # Go 函数模式
            CodePattern(
                pattern_id="go_function_basic",
                language=CodeLanguage.GO,
                pattern_type="function",
                template="""// {description}
func {function_name}({parameters}) {return_type} {{
    {body}
}}""",
                description="Go函数模板",
                variables=["description", "function_name", "parameters", "return_type", "body"],
                complexity=CodeComplexity.LOW,
                tags=["function", "go", "basic"]
            ),
            
            # Rust 函数模式
            CodePattern(
                pattern_id="rust_function_basic",
                language=CodeLanguage.RUST,
                pattern_type="function",
                template="""/// {description}
fn {function_name}({parameters}) -> {return_type} {{
    {body}
}}""",
                description="Rust函数模板",
                variables=["description", "function_name", "parameters", "return_type", "body"],
                complexity=CodeComplexity.LOW,
                tags=["function", "rust", "basic"]
            ),
            
            # SQL 查询模式
            CodePattern(
                pattern_id="sql_select_basic",
                language=CodeLanguage.SQL,
                pattern_type="query",
                template="""SELECT {columns}
FROM {table}
{where_clause}
{group_by}
{order_by}
{limit_clause};""",
                description="基础SQL查询模板",
                variables=["columns", "table", "where_clause", "group_by", "order_by", "limit_clause"],
                complexity=CodeComplexity.LOW,
                tags=["sql", "select", "query"]
            ),
            
            # HTML 模板模式
            CodePattern(
                pattern_id="html_template_basic",
                language=CodeLanguage.HTML,
                pattern_type="template",
                template="""<!DOCTYPE html>
<html lang="{lang}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {styles}
</head>
<body>
    {content}
    {scripts}
</body>
</html>""",
                description="基础HTML模板",
                variables=["lang", "title", "styles", "content", "scripts"],
                complexity=CodeComplexity.LOW,
                tags=["html", "template", "basic"]
            )
        ]
        
        for pattern in builtin_patterns:
            if pattern.pattern_id not in self.patterns:
                self.patterns[pattern.pattern_id] = pattern
                self._index_pattern(pattern)
        
        logger.info(f"🔧 初始化了 {len(builtin_patterns)} 个内置代码模式")
    
    def search_patterns(self, 
                       language: CodeLanguage,
                       pattern_type: Optional[str] = None,
                       complexity: Optional[CodeComplexity] = None,
                       tags: Optional[List[str]] = None) -> List[CodePattern]:
        """搜索模式"""
        candidates = set(self.pattern_index['language'].get(language.value, []))
        
        if pattern_type:
            type_candidates = set(self.pattern_index['type'].get(pattern_type, []))
            candidates &= type_candidates
        
        if complexity:
            complexity_candidates = set(self.pattern_index['complexity'].get(complexity.value, []))
            candidates &= complexity_candidates
        
        if tags:
            tag_candidates = set()
            for tag in tags:
                tag_candidates.update(self.pattern_index['tag'].get(tag, []))
            candidates &= tag_candidates
        
        return [self.patterns[pattern_id] for pattern_id in candidates]
    
    def get_pattern(self, pattern_id: str) -> Optional[CodePattern]:
        """获取模式"""
        return self.patterns.get(pattern_id)
    
    def add_pattern(self, pattern: CodePattern):
        """添加模式"""
        self.patterns[pattern.pattern_id] = pattern
        self._index_pattern(pattern)
        logger.info(f"➕ 添加代码模式: {pattern.pattern_id}")
    
    def update_pattern_usage(self, pattern_id: str, success: bool):
        """更新模式使用统计"""
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            pattern.usage_count += 1
            # 更新成功率
            pattern.success_rate = (pattern.success_rate * (pattern.usage_count - 1) + (1 if success else 0)) / pattern.usage_count

class CodeAnalyzer:
    """代码分析器"""
    
    def __init__(self):
        """初始化代码分析器"""
        self.language_detectors = {
            CodeLanguage.PYTHON: self._detect_python,
            CodeLanguage.JAVASCRIPT: self._detect_javascript,
            CodeLanguage.TYPESCRIPT: self._detect_typescript,
            CodeLanguage.JAVA: self._detect_java,
            CodeLanguage.CSHARP: self._detect_csharp,
            CodeLanguage.GO: self._detect_go,
            CodeLanguage.RUST: self._detect_rust,
            CodeLanguage.CPP: self._detect_cpp,
            CodeLanguage.HTML: self._detect_html,
            CodeLanguage.CSS: self._detect_css,
            CodeLanguage.SQL: self._detect_sql,
            CodeLanguage.JSON: self._detect_json,
            CodeLanguage.YAML: self._detect_yaml
        }
    
    def detect_language(self, code: str, file_extension: Optional[str] = None) -> CodeLanguage:
        """检测编程语言"""
        # 首先根据文件扩展名判断
        if file_extension:
            extension_map = {
                '.py': CodeLanguage.PYTHON,
                '.js': CodeLanguage.JAVASCRIPT,
                '.ts': CodeLanguage.TYPESCRIPT,
                '.jsx': CodeLanguage.JAVASCRIPT,
                '.tsx': CodeLanguage.TYPESCRIPT,
                '.java': CodeLanguage.JAVA,
                '.cs': CodeLanguage.CSHARP,
                '.go': CodeLanguage.GO,
                '.rs': CodeLanguage.RUST,
                '.cpp': CodeLanguage.CPP,
                '.cc': CodeLanguage.CPP,
                '.cxx': CodeLanguage.CPP,
                '.c': CodeLanguage.CPP,
                '.h': CodeLanguage.CPP,
                '.hpp': CodeLanguage.CPP,
                '.html': CodeLanguage.HTML,
                '.htm': CodeLanguage.HTML,
                '.css': CodeLanguage.CSS,
                '.sql': CodeLanguage.SQL,
                '.json': CodeLanguage.JSON,
                '.yaml': CodeLanguage.YAML,
                '.yml': CodeLanguage.YAML,
                '.md': CodeLanguage.MARKDOWN
            }
            
            if file_extension.lower() in extension_map:
                return extension_map[file_extension.lower()]
        
        # 根据内容检测
        language_scores = {}
        for language, detector in self.language_detectors.items():
            score = detector(code)
            language_scores[language] = score
        
        # 返回得分最高的语言
        if language_scores:
            return max(language_scores, key=language_scores.get)
        
        return CodeLanguage.PYTHON  # 默认返回Python
    
    def _detect_python(self, code: str) -> float:
        """检测Python代码"""
        indicators = [
            'def ', 'class ', 'import ', 'from ', 'if __name__',
            'self.', 'elif ', 'try:', 'except:', 'finally:',
            'with open(', 'print(', 'len(', 'range('
        ]
        score = sum(1 for indicator in indicators if indicator in code)
        return score / len(indicators)
    
    def _detect_javascript(self, code: str) -> float:
        """检测JavaScript代码"""
        indicators = [
            'function ', 'const ', 'let ', 'var ', '=>', '===', '!==',
            'console.log', 'document.', 'window.', 'Array.',
            'Object.', 'Promise', 'async ', 'await '
        ]
        score = sum(1 for indicator in indicators if indicator in code)
        return score / len(indicators)
    
    def _detect_typescript(self, code: str) -> float:
        """检测TypeScript代码"""
        js_score = self._detect_javascript(code)
        ts_indicators = [
            ': string', ': number', ': boolean', ': void', ': any',
            'interface ', 'type ', 'enum ', 'as ', '|', '&',
            'public ', 'private ', 'protected ', 'readonly '
        ]
        ts_score = sum(1 for indicator in ts_indicators if indicator in code)
        return js_score + (ts_score / len(ts_indicators))
    
    def _detect_java(self, code: str) -> float:
        """检测Java代码"""
        indicators = [
            'public class', 'private ', 'public ', 'protected ',
            'static void main', 'System.out', 'import java.',
            'extends ', 'implements ', '@Override', 'ArrayList',
            'HashMap', 'String[] args'
        ]
        score = sum(1 for indicator in indicators if indicator in code)
        return score / len(indicators)
    
    def _detect_csharp(self, code: str) -> float:
        """检测C#代码"""
        indicators = [
            'using System', 'namespace ', 'public class',
            'private ', 'public ', 'protected ', 'static ',
            'Console.WriteLine', 'List<', 'Dictionary<', 'string.',
            'int ', 'bool ', 'var '
        ]
        score = sum(1 for indicator in indicators if indicator in code)
        return score / len(indicators)
    
    def _detect_go(self, code: str) -> float:
        """检测Go代码"""
        indicators = [
            'package main', 'func main()', 'import (', 'fmt.',
            'func ', 'var ', 'const ', 'type ', 'struct ',
            'interface ', 'go func(', 'chan ', 'select {'
        ]
        score = sum(1 for indicator in indicators if indicator in code)
        return score / len(indicators)
    
    def _detect_rust(self, code: str) -> float:
        """检测Rust代码"""
        indicators = [
            'fn main()', 'use std::', 'fn ', 'let mut', 'impl ',
            'struct ', 'enum ', 'match ', 'Option<', 'Result<',
            'vec!', 'String::', '&str', 'println!'
        ]
        score = sum(1 for indicator in indicators if indicator in code)
        return score / len(indicators)
    
    def _detect_cpp(self, code: str) -> float:
        """检测C++代码"""
        indicators = [
            '#include', 'std::', 'int main', 'using namespace',
            'class ', 'public:', 'private:', 'protected:',
            'cout <<', 'cin >>', 'vector<', 'std::string'
        ]
        score = sum(1 for indicator in indicators if indicator in code)
        return score / len(indicators)
    
    def _detect_html(self, code: str) -> float:
        """检测HTML代码"""
        indicators = [
            '<!DOCTYPE html>', '<html>', '<head>', '<body>',
            '<div', '<span', '<p>', '<a href=', '<img src=',
            '<script', '<style', '<link', '<meta', '<title'
        ]
        score = sum(1 for indicator in indicators if indicator in code)
        return score / len(indicators)
    
    def _detect_css(self, code: str) -> float:
        """检测CSS代码"""
        indicators = [
            '{', '}', 'margin:', 'padding:', 'color:', 'background:',
            'font-size:', 'display:', 'position:', 'width:', 'height:',
            'border:', '#', 'px', 'em', 'rem'
        ]
        score = sum(1 for indicator in indicators if indicator in code)
        return score / len(indicators)
    
    def _detect_sql(self, code: str) -> float:
        """检测SQL代码"""
        indicators = [
            'SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE',
            'JOIN', 'INNER JOIN', 'LEFT JOIN', 'GROUP BY', 'ORDER BY',
            'CREATE TABLE', 'ALTER TABLE', 'DROP TABLE'
        ]
        score = sum(1 for indicator in indicators if indicator.upper() in code.upper())
        return score / len(indicators)
    
    def _detect_json(self, code: str) -> float:
        """检测JSON代码"""
        try:
            json.loads(code.strip())
            return 1.0
        except:
            # 检查JSON特征
            indicators = ['{', '}', '"', ':', ',', '[', ']']
            score = sum(1 for indicator in indicators if indicator in code)
            return score / len(indicators)
    
    def _detect_yaml(self, code: str) -> float:
        """检测YAML代码"""
        indicators = ['key:', '- item', '  ', '\n', ': ', '|', '>']
        score = sum(1 for indicator in indicators if indicator in code)
        return score / len(indicators)
    
    def analyze_complexity(self, code: str, language: CodeLanguage) -> CodeComplexity:
        """分析代码复杂度"""
        # 基础指标
        lines = len(code.split('\n'))
        chars = len(code)
        
        # 语言特定的复杂度指标
        if language == CodeLanguage.PYTHON:
            return self._analyze_python_complexity(code)
        elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
            return self._analyze_js_complexity(code)
        elif language == CodeLanguage.JAVA:
            return self._analyze_java_complexity(code)
        elif language == CodeLanguage.CSHARP:
            return self._analyze_csharp_complexity(code)
        elif language == CodeLanguage.GO:
            return self._analyze_go_complexity(code)
        elif language == CodeLanguage.RUST:
            return self._analyze_rust_complexity(code)
        elif language == CodeLanguage.CPP:
            return self._analyze_cpp_complexity(code)
        else:
            # 通用复杂度分析
            if lines < 10:
                return CodeComplexity.LOW
            elif lines < 30:
                return CodeComplexity.MEDIUM
            elif lines < 100:
                return CodeComplexity.HIGH
            else:
                return CodeComplexity.VERY_HIGH
    
    def _analyze_python_complexity(self, code: str) -> CodeComplexity:
        """分析Python代码复杂度"""
        lines = code.split('\n')
        
        # 计算圈复杂度
        complexity_keywords = ['if', 'elif', 'for', 'while', 'except', 'with']
        complexity_score = sum(1 for line in lines if any(keyword in line for keyword in complexity_keywords))
        
        # 计算嵌套深度
        max_indent = 0
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent // 4)  # 假设4个空格为一个缩进级别
        
        total_score = complexity_score + max_indent
        
        if total_score < 5:
            return CodeComplexity.LOW
        elif total_score < 15:
            return CodeComplexity.MEDIUM
        elif total_score < 30:
            return CodeComplexity.HIGH
        else:
            return CodeComplexity.VERY_HIGH
    
    def _analyze_js_complexity(self, code: str) -> CodeComplexity:
        """分析JavaScript代码复杂度"""
        lines = code.split('\n')
        
        # 计算圈复杂度
        complexity_keywords = ['if', 'else if', 'for', 'while', 'catch', 'switch', 'case']
        complexity_score = sum(1 for line in lines if any(keyword in line for keyword in complexity_keywords))
        
        # 计算函数嵌套
        function_count = sum(1 for line in lines if 'function' in line or '=>' in line)
        
        total_score = complexity_score + function_count
        
        if total_score < 5:
            return CodeComplexity.LOW
        elif total_score < 15:
            return CodeComplexity.MEDIUM
        elif total_score < 30:
            return CodeComplexity.HIGH
        else:
            return CodeComplexity.VERY_HIGH
    
    def _analyze_java_complexity(self, code: str) -> CodeComplexity:
        """分析Java代码复杂度"""
        lines = code.split('\n')
        
        # 计算圈复杂度
        complexity_keywords = ['if', 'else if', 'for', 'while', 'catch', 'switch', 'case']
        complexity_score = sum(1 for line in lines if any(keyword in line for keyword in complexity_keywords))
        
        # 计算类和方法数量
        class_count = sum(1 for line in lines if 'class ' in line)
        method_count = sum(1 for line in lines if 'public ' in line and '(' in line)
        
        total_score = complexity_score + class_count + method_count
        
        if total_score < 8:
            return CodeComplexity.LOW
        elif total_score < 20:
            return CodeComplexity.MEDIUM
        elif total_score < 40:
            return CodeComplexity.HIGH
        else:
            return CodeComplexity.VERY_HIGH
    
    def _analyze_csharp_complexity(self, code: str) -> CodeComplexity:
        """分析C#代码复杂度"""
        return self._analyze_java_complexity(code)  # 类似的复杂度分析
    
    def _analyze_go_complexity(self, code: str) -> CodeComplexity:
        """分析Go代码复杂度"""
        lines = code.split('\n')
        
        # 计算圈复杂度
        complexity_keywords = ['if', 'else if', 'for', 'switch', 'case', 'select']
        complexity_score = sum(1 for line in lines if any(keyword in line for keyword in complexity_keywords))
        
        # 计算goroutine和channel使用
        go_features = sum(1 for line in lines if 'go ' in line or 'chan ' in line or '<-' in line)
        
        total_score = complexity_score + go_features
        
        if total_score < 5:
            return CodeComplexity.LOW
        elif total_score < 15:
            return CodeComplexity.MEDIUM
        elif total_score < 30:
            return CodeComplexity.HIGH
        else:
            return CodeComplexity.VERY_HIGH
    
    def _analyze_rust_complexity(self, code: str) -> CodeComplexity:
        """分析Rust代码复杂度"""
        lines = code.split('\n')
        
        # 计算圈复杂度
        complexity_keywords = ['if', 'else if', 'for', 'while', 'match', 'loop']
        complexity_score = sum(1 for line in lines if any(keyword in line for keyword in complexity_keywords))
        
        # 计算所有权和借用复杂度
        rust_features = sum(1 for line in lines if '&' in line or 'mut ' in line or 'move ' in line)
        
        total_score = complexity_score + rust_features
        
        if total_score < 5:
            return CodeComplexity.LOW
        elif total_score < 15:
            return CodeComplexity.MEDIUM
        elif total_score < 30:
            return CodeComplexity.HIGH
        else:
            return CodeComplexity.VERY_HIGH
    
    def _analyze_cpp_complexity(self, code: str) -> CodeComplexity:
        """分析C++代码复杂度"""
        lines = code.split('\n')
        
        # 计算圈复杂度
        complexity_keywords = ['if', 'else if', 'for', 'while', 'switch', 'case', 'catch']
        complexity_score = sum(1 for line in lines if any(keyword in line for keyword in complexity_keywords))
        
        # 计算模板和指针复杂度
        cpp_features = sum(1 for line in lines if 'template' in line or '*' in line or '&' in line)
        
        total_score = complexity_score + cpp_features
        
        if total_score < 5:
            return CodeComplexity.LOW
        elif total_score < 15:
            return CodeComplexity.MEDIUM
        elif total_score < 30:
            return CodeComplexity.HIGH
        else:
            return CodeComplexity.VERY_HIGH
    
    def extract_context(self, code: str, cursor_position: int) -> str:
        """提取代码上下文"""
        lines = code.split('\n')
        
        # 找到光标所在行
        current_line = 0
        char_count = 0
        for i, line in enumerate(lines):
            if char_count + len(line) + 1 > cursor_position:
                current_line = i
                break
            char_count += len(line) + 1
        
        # 提取上下文（前后各几行）
        context_lines = 5
        start_line = max(0, current_line - context_lines)
        end_line = min(len(lines), current_line + context_lines + 1)
        
        context = '\n'.join(lines[start_line:end_line])
        return context

class CodeGenerator:
    """代码生成器"""
    
    def __init__(self, pattern_database: Optional[CodePatternDatabase] = None):
        """初始化代码生成器"""
        self.pattern_db = pattern_database or CodePatternDatabase()
        self.analyzer = CodeAnalyzer()
        self.generation_cache = {}
        self.performance_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'cache_hits': 0,
            'average_generation_time': 0
        }
    
    def generate_code(self, request: GenerationRequest) -> CodeSuggestion:
        """生成代码"""
        start_time = time.time()
        
        # 检查缓存
        cache_key = self._generate_cache_key(request)
        if cache_key in self.generation_cache:
            self.performance_stats['cache_hits'] += 1
            cached_result = self.generation_cache[cache_key]
            logger.info(f"🎯 缓存命中: {request.request_id}")
            return cached_result
        
        try:
            # 搜索合适的模式
            patterns = self._search_patterns(request)
            
            if not patterns:
                # 如果没有找到模式，使用通用生成
                suggestion = self._generate_generic_code(request)
            else:
                # 使用最佳模式生成代码
                best_pattern = self._select_best_pattern(patterns, request)
                suggestion = self._generate_from_pattern(best_pattern, request)
            
            # 更新性能统计
            self._update_performance_stats(start_time, True)
            
            # 缓存结果
            self.generation_cache[cache_key] = suggestion
            
            # 更新模式使用统计
            if suggestion.pattern_id:
                self.pattern_db.update_pattern_usage(suggestion.pattern_id, True)
            
            logger.info(f"✅ 代码生成成功: {request.request_id}")
            return suggestion
            
        except Exception as e:
            logger.error(f"❌ 代码生成失败: {e}")
            self._update_performance_stats(start_time, False)
            
            # 返回错误建议
            return CodeSuggestion(
                suggestion_id=f"error_{request.request_id}",
                language=request.language,
                context=request.context or "",
                suggested_code=f"// 代码生成失败: {str(e)}",
                confidence=0.0,
                explanation=f"生成代码时出错: {str(e)}",
                pattern_id=None,
                metadata={"error": str(e)}
            )
    
    def complete_code(self, 
                      code: str, 
                      cursor_position: int,
                      language: Optional[CodeLanguage] = None) -> List[CodeSuggestion]:
        """代码补全"""
        # 检测语言
        if not language:
            language = self.analyzer.detect_language(code)
        
        # 提取上下文
        context = self.analyzer.extract_context(code, cursor_position)
        
        # 分析上下文，生成补全建议
        suggestions = []
        
        # 基于模式的补全
        patterns = self.pattern_db.search_patterns(language)
        for pattern in patterns[:5]:  # 限制建议数量
            suggestion = self._complete_from_pattern(pattern, context, cursor_position)
            if suggestion:
                suggestions.append(suggestion)
        
        # 基于语法的补全
        syntax_suggestions = self._complete_from_syntax(context, language, cursor_position)
        suggestions.extend(syntax_suggestions)
        
        # 排序并返回最佳建议
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        
        return suggestions[:10]  # 返回前10个建议
    
    def _generate_cache_key(self, request: GenerationRequest) -> str:
        """生成缓存键"""
        content = f"{request.language.value}_{request.description}_{request.complexity.value}"
        content += "_".join(request.requirements) + "_".join(request.constraints)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _search_patterns(self, request: GenerationRequest) -> List[CodePattern]:
        """搜索合适的模式"""
        patterns = self.pattern_db.search_patterns(
            language=request.language,
            complexity=request.complexity
        )
        
        # 根据需求过滤
        if request.requirements:
            filtered_patterns = []
            for pattern in patterns:
                if any(req in pattern.description.lower() or 
                      req in " ".join(pattern.tags).lower() 
                      for req in [r.lower() for r in request.requirements]):
                    filtered_patterns.append(pattern)
            patterns = filtered_patterns
        
        # 按成功率和使用次数排序
        patterns.sort(key=lambda p: (p.success_rate, p.usage_count), reverse=True)
        
        return patterns
    
    def _select_best_pattern(self, patterns: List[CodePattern], request: GenerationRequest) -> CodePattern:
        """选择最佳模式"""
        if not patterns:
            return None
        
        # 简单选择策略：选择成功率和使用次数最高的模式
        return patterns[0]
    
    def _generate_from_pattern(self, pattern: CodePattern, request: GenerationRequest) -> CodeSuggestion:
        """从模式生成代码"""
        try:
            # 准备变量替换
            variables = self._extract_variables_from_request(request, pattern)
            
            # 替换模板变量
            generated_code = pattern.template
            for var, value in variables.items():
                generated_code = generated_code.replace(f"{{{var}}}", value)
            
            # 计算置信度
            confidence = self._calculate_confidence(pattern, request)
            
            return CodeSuggestion(
                suggestion_id=f"pattern_{pattern.pattern_id}_{request.request_id}",
                language=request.language,
                context=request.context or "",
                suggested_code=generated_code,
                confidence=confidence,
                explanation=f"基于模式 '{pattern.description}' 生成",
                pattern_id=pattern.pattern_id,
                metadata={
                    "pattern_type": pattern.pattern_type,
                    "complexity": pattern.complexity.value,
                    "tags": pattern.tags
                }
            )
            
        except Exception as e:
            logger.error(f"❌ 从模式生成代码失败: {e}")
            raise
    
    def _extract_variables_from_request(self, request: GenerationRequest, pattern: CodePattern) -> Dict[str, str]:
        """从请求中提取变量"""
        variables = {}
        
        # 基础变量提取
        if "function_name" in pattern.variables:
            function_name = self._extract_function_name(request.description)
            variables["function_name"] = function_name
        
        if "class_name" in pattern.variables:
            class_name = self._extract_class_name(request.description)
            variables["class_name"] = class_name
        
        if "description" in pattern.variables:
            variables["description"] = request.description
        
        if "body" in pattern.variables:
            body = self._generate_function_body(request.description, request.requirements)
            variables["body"] = body
        
        # 基于语言特定的变量
        if request.language == CodeLanguage.PYTHON:
            variables.update(self._extract_python_variables(request, pattern))
        elif request.language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
            variables.update(self._extract_js_variables(request, pattern))
        elif request.language == CodeLanguage.JAVA:
            variables.update(self._extract_java_variables(request, pattern))
        elif request.language == CodeLanguage.GO:
            variables.update(self._extract_go_variables(request, pattern))
        elif request.language == CodeLanguage.RUST:
            variables.update(self._extract_rust_variables(request, pattern))
        
        return variables
    
    def _extract_function_name(self, description: str) -> str:
        """提取函数名"""
        # 简单的函数名提取逻辑
        words = description.lower().split()
        
        # 寻找动词
        verbs = ['create', 'get', 'set', 'update', 'delete', 'calculate', 'process', 'handle', 'validate', 'convert']
        for word in words:
            if word in verbs:
                # 找到下一个名词作为函数名
                verb_index = words.index(word)
                if verb_index + 1 < len(words):
                    return f"{word}_{words[verb_index + 1]}"
        
        # 默认函数名
        return "process_data"
    
    def _extract_class_name(self, description: str) -> str:
        """提取类名"""
        words = description.lower().split()
        
        # 寻找名词
        nouns = ['user', 'product', 'order', 'service', 'manager', 'handler', 'controller', 'model', 'entity']
        for word in words:
            if word in nouns:
                return word.title()
        
        # 默认类名
        return "DataProcessor"
    
    def _generate_function_body(self, description: str, requirements: List[str]) -> str:
        """生成函数体"""
        body_lines = []
        
        # 基于需求生成代码
        for req in requirements:
            if "validate" in req.lower():
                body_lines.append("    # 验证输入")
                body_lines.append("    if not input_data:")
                body_lines.append("        raise ValueError('输入数据不能为空')")
            elif "process" in req.lower():
                body_lines.append("    # 处理数据")
                body_lines.append("    processed_data = process_data(input_data)")
            elif "return" in req.lower():
                body_lines.append("    # 返回结果")
                body_lines.append("    return processed_data")
        
        if not body_lines:
            body_lines.append("    # TODO: 实现功能")
            body_lines.append("    pass")
        
        return "\n".join(body_lines)
    
    def _extract_python_variables(self, request: GenerationRequest, pattern: CodePattern) -> Dict[str, str]:
        """提取Python特定变量"""
        variables = {}
        
        if "parameters" in pattern.variables:
            variables["parameters"] = "self, data"
        
        if "args_doc" in pattern.variables:
            variables["args_doc"] = "data: 输入数据"
        
        if "return_doc" in pattern.variables:
            variables["return_doc"] = "处理后的数据"
        
        if "init_params" in pattern.variables:
            variables["init_params"] = ""
        
        if "init_body" in pattern.variables:
            variables["init_body"] = "        self.data = None"
        
        if "method_name" in pattern.variables:
            variables["method_name"] = "process"
        
        if "method_params" in pattern.variables:
            variables["method_params"] = ""
        
        if "method_description" in pattern.variables:
            variables["method_description"] = "处理数据的方法"
        
        if "method_body" in pattern.variables:
            variables["method_body"] = "        return self.data"
        
        return variables
    
    def _extract_js_variables(self, request: GenerationRequest, pattern: CodePattern) -> Dict[str, str]:
        """提取JavaScript特定变量"""
        variables = {}
        
        if "parameters" in pattern.variables:
            variables["parameters"] = "data"
        
        if "body" in pattern.variables:
            variables["body"] = "    // 处理数据\n    return processedData;"
        
        if "imports" in pattern.variables:
            variables["imports"] = "import React from 'react';"
        
        if "component_name" in pattern.variables:
            variables["component_name"] = "MyComponent"
        
        if "props" in pattern.variables:
            variables["props"] = ""
        
        if "class_name" in pattern.variables:
            variables["class_name"] = "my-component"
        
        if "jsx_content" in pattern.variables:
            variables["jsx_content"] = "        <div>Hello, World!</div>"
        
        return variables
    
    def _extract_java_variables(self, request: GenerationRequest, pattern: CodePattern) -> Dict[str, str]:
        """提取Java特定变量"""
        variables = {}
        
        if "parameters" in pattern.variables:
            variables["parameters"] = "String data"
        
        if "return_type" in pattern.variables:
            variables["return_type"] = "String"
        
        if "body" in pattern.variables:
            variables["body"] = "        // TODO: 实现功能\n        return null;"
        
        return variables
    
    def _extract_go_variables(self, request: GenerationRequest, pattern: CodePattern) -> Dict[str, str]:
        """提取Go特定变量"""
        variables = {}
        
        if "parameters" in pattern.variables:
            variables["parameters"] = "data string"
        
        if "return_type" in pattern.variables:
            variables["return_type"] = "string"
        
        if "body" in pattern.variables:
            variables["body"] = "    // TODO: 实现功能\n    return \"\""
        
        return variables
    
    def _extract_rust_variables(self, request: GenerationRequest, pattern: CodePattern) -> Dict[str, str]:
        """提取Rust特定变量"""
        variables = {}
        
        if "parameters" in pattern.variables:
            variables["parameters"] = "data: &str"
        
        if "return_type" in pattern.variables:
            variables["return_type"] = "String"
        
        if "body" in pattern.variables:
            variables["body"] = "    // TODO: 实现功能\n    String::new()"
        
        return variables
    
    def _calculate_confidence(self, pattern: CodePattern, request: GenerationRequest) -> float:
        """计算置信度"""
        confidence = 0.5  # 基础置信度
        
        # 基于成功率调整
        confidence += pattern.success_rate * 0.3
        
        # 基于使用次数调整
        if pattern.usage_count > 10:
            confidence += 0.1
        
        # 基于复杂度匹配调整
        if pattern.complexity == request.complexity:
            confidence += 0.1
        
        # 基于标签匹配调整
        if request.requirements:
            matching_tags = sum(1 for req in request.requirements 
                              if req.lower() in " ".join(pattern.tags).lower())
            if matching_tags > 0:
                confidence += min(0.2, matching_tags * 0.05)
        
        return min(1.0, confidence)
    
    def _generate_generic_code(self, request: GenerationRequest) -> CodeSuggestion:
        """生成通用代码"""
        # 简单的通用代码生成逻辑
        language = request.language
        
        if language == CodeLanguage.PYTHON:
            code = f"# {request.description}\ndef process_data():\n    # TODO: 实现\n    pass"
        elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
            code = f"// {request.description}\nfunction processData() {{\n    // TODO: 实现\n}}"
        elif language == CodeLanguage.JAVA:
            code = f"// {request.description}\npublic class Processor {{\n    public void process() {{\n        // TODO: 实现\n    }}\n}}"
        elif language == CodeLanguage.GO:
            code = f"// {request.description}\npackage main\n\nfunc main() {{\n    // TODO: 实现\n}}"
        elif language == CodeLanguage.RUST:
            code = f"// {request.description}\nfn main() {{\n    // TODO: 实现\n}}"
        else:
            code = f"// {request.description}\n// TODO: 实现"
        
        return CodeSuggestion(
            suggestion_id=f"generic_{request.request_id}",
            language=request.language,
            context=request.context or "",
            suggested_code=code,
            confidence=0.3,
            explanation="通用代码生成，建议根据具体需求进行调整",
            pattern_id=None,
            metadata={"generation_type": "generic"}
        )
    
    def _complete_from_pattern(self, pattern: CodePattern, context: str, cursor_position: int) -> Optional[CodeSuggestion]:
        """从模式进行代码补全"""
        try:
            # 分析上下文，提取部分匹配的模板
            context_lines = context.split('\n')
            current_line = ""
            
            # 找到光标所在行
            char_count = 0
            for line in context_lines:
                if char_count + len(line) + 1 > cursor_position:
                    current_line = line
                    break
                char_count += len(line) + 1
            
            # 检查是否有部分匹配的模式
            if any(keyword in current_line for keyword in ['def', 'function', 'class', 'interface']):
                # 生成补全建议
                variables = self._extract_variables_from_context(context, pattern)
                
                # 替换模板变量
                completion = pattern.template
                for var, value in variables.items():
                    completion = completion.replace(f"{{{var}}}", value)
                
                # 只返回补全部分
                completion = self._extract_completion_part(completion, current_line)
                
                return CodeSuggestion(
                    suggestion_id=f"complete_{pattern.pattern_id}",
                    language=pattern.language,
                    context=context,
                    suggested_code=completion,
                    confidence=0.7,
                    explanation=f"基于模式 '{pattern.description}' 的补全",
                    pattern_id=pattern.pattern_id,
                    metadata={"completion_type": "pattern"}
                )
            
        except Exception as e:
            logger.debug(f"模式补全失败: {e}")
        
        return None
    
    def _complete_from_syntax(self, context: str, language: CodeLanguage, cursor_position: int) -> List[CodeSuggestion]:
        """基于语法的代码补全"""
        suggestions = []
        
        try:
            # 基于语法的简单补全
            context_lines = context.split('\n')
            current_line = ""
            
            # 找到光标所在行
            char_count = 0
            for line in context_lines:
                if char_count + len(line) + 1 > cursor_position:
                    current_line = line
                    break
                char_count += len(line) + 1
            
            # 语言特定的语法补全
            if language == CodeLanguage.PYTHON:
                suggestions.extend(self._python_syntax_completion(current_line))
            elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
                suggestions.extend(self._js_syntax_completion(current_line))
            elif language == CodeLanguage.JAVA:
                suggestions.extend(self._java_syntax_completion(current_line))
            elif language == CodeLanguage.GO:
                suggestions.extend(self._go_syntax_completion(current_line))
            elif language == CodeLanguage.RUST:
                suggestions.extend(self._rust_syntax_completion(current_line))
            
        except Exception as e:
            logger.debug(f"语法补全失败: {e}")
        
        return suggestions
    
    def _python_syntax_completion(self, line: str) -> List[CodeSuggestion]:
        """Python语法补全"""
        suggestions = []
        
        if line.strip().startswith('def '):
            suggestions.append(CodeSuggestion(
                suggestion_id="python_func_def",
                language=CodeLanguage.PYTHON,
                context=line,
                suggested_code="():\n    \"\"\"函数描述\"\"\"\n    pass",
                confidence=0.8,
                explanation="函数定义补全",
                pattern_id=None,
                metadata={"completion_type": "syntax"}
            ))
        
        elif line.strip().startswith('class '):
            suggestions.append(CodeSuggestion(
                suggestion_id="python_class_def",
                language=CodeLanguage.PYTHON,
                context=line,
                suggested_code=":\n    def __init__(self):\n        pass",
                confidence=0.8,
                explanation="类定义补全",
                pattern_id=None,
                metadata={"completion_type": "syntax"}
            ))
        
        elif 'import ' in line:
            suggestions.append(CodeSuggestion(
                suggestion_id="python_import",
                language=CodeLanguage.PYTHON,
                context=line,
                suggested_code="\nfrom ",
                confidence=0.7,
                explanation="import语句补全",
                pattern_id=None,
                metadata={"completion_type": "syntax"}
            ))
        
        return suggestions
    
    def _js_syntax_completion(self, line: str) -> List[CodeSuggestion]:
        """JavaScript语法补全"""
        suggestions = []
        
        if line.strip().startswith('function '):
            suggestions.append(CodeSuggestion(
                suggestion_id="js_func_def",
                language=CodeLanguage.JAVASCRIPT,
                context=line,
                suggested_code="() {\n    // TODO: 实现\n}",
                confidence=0.8,
                explanation="函数定义补全",
                pattern_id=None,
                metadata={"completion_type": "syntax"}
            ))
        
        elif line.strip().startswith('const '):
            suggestions.append(CodeSuggestion(
                suggestion_id="js_const_def",
                language=CodeLanguage.JAVASCRIPT,
                context=line,
                suggested_code=" = ",
                confidence=0.7,
                explanation="常量定义补全",
                pattern_id=None,
                metadata={"completion_type": "syntax"}
            ))
        
        elif 'import ' in line:
            suggestions.append(CodeSuggestion(
                suggestion_id="js_import",
                language=CodeLanguage.JAVASCRIPT,
                context=line,
                suggested_code=" from ",
                confidence=0.7,
                explanation="import语句补全",
                pattern_id=None,
                metadata={"completion_type": "syntax"}
            ))
        
        return suggestions
    
    def _java_syntax_completion(self, line: str) -> List[CodeSuggestion]:
        """Java语法补全"""
        suggestions = []
        
        if line.strip().startswith('public class '):
            suggestions.append(CodeSuggestion(
                suggestion_id="java_class_def",
                language=CodeLanguage.JAVA,
                context=line,
                suggested_code=" {\n    public static void main(String[] args) {\n        // TODO: 实现\n    }\n}",
                confidence=0.8,
                explanation="类定义补全",
                pattern_id=None,
                metadata={"completion_type": "syntax"}
            ))
        
        elif line.strip().startswith('public void '):
            suggestions.append(CodeSuggestion(
                suggestion_id="java_method_def",
                language=CodeLanguage.JAVA,
                context=line,
                suggested_code="() {\n    // TODO: 实现\n}",
                confidence=0.8,
                explanation="方法定义补全",
                pattern_id=None,
                metadata={"completion_type": "syntax"}
            ))
        
        return suggestions
    
    def _go_syntax_completion(self, line: str) -> List[CodeSuggestion]:
        """Go语法补全"""
        suggestions = []
        
        if line.strip().startswith('func '):
            suggestions.append(CodeSuggestion(
                suggestion_id="go_func_def",
                language=CodeLanguage.GO,
                context=line,
                suggested_code="() {\n    // TODO: 实现\n}",
                confidence=0.8,
                explanation="函数定义补全",
                pattern_id=None,
                metadata={"completion_type": "syntax"}
            ))
        
        elif 'import ' in line:
            suggestions.append(CodeSuggestion(
                suggestion_id="go_import",
                language=CodeLanguage.GO,
                context=line,
                suggested_code=" \"\"",
                confidence=0.7,
                explanation="import语句补全",
                pattern_id=None,
                metadata={"completion_type": "syntax"}
            ))
        
        return suggestions
    
    def _rust_syntax_completion(self, line: str) -> List[CodeSuggestion]:
        """Rust语法补全"""
        suggestions = []
        
        if line.strip().startswith('fn '):
            suggestions.append(CodeSuggestion(
                suggestion_id="rust_func_def",
                language=CodeLanguage.RUST,
                context=line,
                suggested_code="() {\n    // TODO: 实现\n}",
                confidence=0.8,
                explanation="函数定义补全",
                pattern_id=None,
                metadata={"completion_type": "syntax"}
            ))
        
        elif line.strip().startswith('struct '):
            suggestions.append(CodeSuggestion(
                suggestion_id="rust_struct_def",
                language=CodeLanguage.RUST,
                context=line,
                suggested_code=" {\n    // TODO: 定义字段\n}",
                confidence=0.8,
                explanation="结构体定义补全",
                pattern_id=None,
                metadata={"completion_type": "syntax"}
            ))
        
        return suggestions
    
    def _extract_variables_from_context(self, context: str, pattern: CodePattern) -> Dict[str, str]:
        """从上下文提取变量"""
        variables = {}
        
        # 简单的上下文变量提取
        lines = context.split('\n')
        
        for line in lines:
            if 'def ' in line and 'function_name' in pattern.variables:
                func_name = line.split('def ')[1].split('(')[0].strip()
                variables['function_name'] = func_name
            elif 'class ' in line and 'class_name' in pattern.variables:
                class_name = line.split('class ')[1].split(':')[0].split('(')[0].strip()
                variables['class_name'] = class_name
        
        return variables
    
    def _extract_completion_part(self, template: str, current_line: str) -> str:
        """提取补全部分"""
        template_lines = template.split('\n')
        current_line_stripped = current_line.strip()
        
        # 找到模板中与当前行最匹配的部分
        for i, template_line in enumerate(template_lines):
            if current_line_stripped and template_line.strip().startswith(current_line_stripped[:10]):
                # 返回从匹配点开始的剩余部分
                remaining_lines = template_lines[i:]
                completion = '\n'.join(remaining_lines)
                
                # 移除已经输入的部分
                if completion.startswith(current_line):
                    completion = completion[len(current_line):]
                
                return completion
        
        # 如果没有匹配，返回整个模板
        return template
    
    def _update_performance_stats(self, start_time: float, success: bool):
        """更新性能统计"""
        self.performance_stats['total_generations'] += 1
        
        if success:
            self.performance_stats['successful_generations'] += 1
        
        generation_time = time.time() - start_time
        total_time = self.performance_stats['average_generation_time'] * (self.performance_stats['total_generations'] - 1)
        self.performance_stats['average_generation_time'] = (total_time + generation_time) / self.performance_stats['total_generations']
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.performance_stats.copy()
    
    def save_patterns(self):
        """保存模式数据库"""
        self.pattern_db.save_patterns()

# 全局代码生成器实例
code_generator = CodeGenerator()

# 便捷函数
def generate_code(language: CodeLanguage, 
                 description: str,
                 requirements: Optional[List[str]] = None,
                 complexity: CodeComplexity = CodeComplexity.MEDIUM,
                 context: Optional[str] = None) -> CodeSuggestion:
    """便捷的代码生成函数"""
    request = GenerationRequest(
        request_id=f"manual_{int(time.time())}",
        language=language,
        description=description,
        context=context,
        requirements=requirements or [],
        constraints=[],
        complexity=complexity,
        style_preferences={}
    )
    
    return code_generator.generate_code(request)

def complete_code(code: str, 
                  cursor_position: int,
                  language: Optional[CodeLanguage] = None) -> List[CodeSuggestion]:
    """便捷的代码补全函数"""
    return code_generator.complete_code(code, cursor_position, language)

# 示例使用
async def example_usage():
    """示例使用"""
    print("🔧 代码生成器示例")
    
    # 生成Python函数
    print("\n1. 生成Python函数:")
    suggestion = generate_code(
        language=CodeLanguage.PYTHON,
        description="创建一个处理用户数据的函数",
        requirements=["validate", "process", "return"],
        complexity=CodeComplexity.MEDIUM
    )
    print(f"建议ID: {suggestion.suggestion_id}")
    print(f"置信度: {suggestion.confidence:.2f}")
    print(f"生成的代码:\n{suggestion.suggested_code}")
    
    # 代码补全
    print("\n2. 代码补全:")
    code = "def process_user_data"
    cursor_position = len(code)
    completions = complete_code(code, cursor_position)
    
    for i, completion in enumerate(completions[:3], 1):
        print(f"补全建议 {i}:")
        print(f"  置信度: {completion.confidence:.2f}")
        print(f"  代码: {completion.suggested_code}")
        print(f"  说明: {completion.explanation}")
    
    # 显示性能统计
    print("\n3. 性能统计:")
    stats = code_generator.get_performance_stats()
    print(f"  总生成次数: {stats['total_generations']}")
    print(f"  成功生成次数: {stats['successful_generations']}")
    print(f"  缓存命中次数: {stats['cache_hits']}")
    print(f"  平均生成时间: {stats['average_generation_time']:.3f}秒")
    
    # 保存模式
    code_generator.save_patterns()
    
    print("\n✅ 代码生成器示例完成")

if __name__ == "__main__":
    asyncio.run(example_usage())