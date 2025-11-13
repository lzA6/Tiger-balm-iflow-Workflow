#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 增强版全能智能体
Enhanced Universal Agent

整合V9.1版本的所有优秀功能，优化Python敏感性和CLI调用机制
"""

import os
import sys
import json
import time
import logging
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import re
import subprocess
import importlib.util
from pathlib import Path

# 添加.iflow到Python路径，解决Python敏感性问题
IFLOW_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(IFLOW_ROOT))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpertiseDomain(Enum):
    """专家领域枚举（增强版）"""
    # 编程语言
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"
    GOLANG = "golang"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    
    # 框架和技术
    REACT = "react"
    VUE = "vue"
    ANGULAR = "angular"
    NODEJS = "nodejs"
    DJANGO = "django"
    FLASK = "flask"
    FASTAPI = "fastapi"
    SPRING = "spring"
    EXPRESS = "express"
    
    # 专业领域
    AI_ML = "ai_ml"
    DATA_SCIENCE = "data_science"
    BLOCKCHAIN = "blockchain"
    QUANTUM = "quantum"
    SECURITY = "security"
    DEVOPS = "devops"
    MOBILE = "mobile"
    WEB3 = "web3"
    IOT = "iot"
    
    # 架构和设计
    ARCHITECTURE = "architecture"
    MICROSERVICES = "microservices"
    CLOUD_NATIVE = "cloud_native"
    SYSTEM_DESIGN = "system_design"

@dataclass
class ExpertCapability:
    """专家能力"""
    domain: ExpertiseDomain
    proficiency: float  # 0-1
    experience_years: int
    recent_projects: List[str]
    tools: List[str]
    certifications: List[str]

class CLIManager:
    """CLI管理器 - 解决Python敏感性问题"""
    
    def __init__(self):
        self.python_executable = sys.executable
        self.env_vars = os.environ.copy()
        self.setup_python_environment()
    
    def setup_python_environment(self):
        """设置Python环境，解决敏感性问题"""
        # 确保Python路径正确
        python_paths = [
            str(IFLOW_ROOT),
            str(IFLOW_ROOT / "tools"),
            str(IFLOW_ROOT / "core"),
            str(IFLOW_ROOT / "agents")
        ]
        
        for path in python_paths:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        # 设置环境变量
        self.env_vars['PYTHONPATH'] = os.pathsep.join(python_paths)
        self.env_vars['PYTHONIOENCODING'] = 'utf-8'
        self.env_vars['PYTHONUNBUFFERED'] = '1'
    
    async def run_python_script(self, script_path: str, args: List[str] = None) -> Dict[str, Any]:
        """运行Python脚本，解决敏感性"""
        try:
            cmd = [self.python_executable, script_path]
            if args:
                cmd.extend(args)
            
            # 使用subprocess运行，避免直接导入的敏感性问题
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.env_vars,
                cwd=str(IFLOW_ROOT)
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                'success': process.returncode == 0,
                'stdout': stdout.decode('utf-8'),
                'stderr': stderr.decode('utf-8'),
                'returncode': process.returncode
            }
            
        except Exception as e:
            logger.error(f"Failed to run Python script {script_path}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def import_module_safely(self, module_name: str, module_path: str = None):
        """安全导入模块"""
        try:
            if module_path:
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            else:
                return importlib.import_module(module_name)
        except Exception as e:
            logger.error(f"Failed to import module {module_name}: {e}")
            return None

class ModelAdapter:
    """模型适配器 - 增强版"""
    
    def __init__(self):
        self.supported_models = {
            # OpenAI系列
            'openai': ['gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-3.5-turbo'],
            # Anthropic系列
            'anthropic': ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku'],
            # Google系列
            'google': ['gemini-1.5-pro', 'gemini-1.5-flash'],
            # 国产模型
            'baidu': ['ernie-4.0', 'ernie-3.5'],
            'alibaba': ['qwen-turbo', 'qwen-plus', 'qwen-max'],
            'tencent': ['hunyuan-pro', 'hunyuan-standard'],
            'bytedance': ['doubao-pro', 'doubao-lite'],
            # 本地模型
            'local': ['llama-3', 'mixtral', 'qwen-72b']
        }
        
        self.model_capabilities = {}
        self.load_model_capabilities()
    
    def load_model_capabilities(self):
        """加载模型能力"""
        for provider, models in self.supported_models.items():
            for model in models:
                self.model_capabilities[model] = {
                    'provider': provider,
                    'max_tokens': self._get_max_tokens(model),
                    'supports_functions': self._supports_functions(model),
                    'supports_vision': self._supports_vision(model),
                    'cost_per_token': self._get_cost(model)
                }
    
    def _get_max_tokens(self, model: str) -> int:
        """获取模型最大token数"""
        if 'gpt-4' in model:
            return 128000
        elif 'claude-3' in model:
            return 200000
        elif 'gemini' in model:
            return 2097152
        elif 'ernie' in model or 'qwen' in model:
            return 128000
        else:
            return 8192
    
    def _supports_functions(self, model: str) -> bool:
        """是否支持函数调用"""
        return 'gpt' in model or 'claude' in model or 'ernie' in model
    
    def _supports_vision(self, model: str) -> bool:
        """是否支持视觉"""
        return 'vision' in model or 'gemini' in model or 'gpt-4o' in model
    
    def _get_cost(self, model: str) -> float:
        """获取模型成本（每1K token）"""
        if 'gpt-4' in model:
            return 0.03
        elif 'claude-3' in model:
            return 0.015
        elif 'gemini' in model:
            return 0.001
        else:
            return 0.001
    
    def select_optimal_model(self, task: str, requirements: Dict[str, Any]) -> str:
        """选择最优模型"""
        # 分析任务需求
        needs_functions = requirements.get('functions', False)
        needs_vision = requirements.get('vision', False)
        max_context = requirements.get('max_context', 4096)
        budget_constraint = requirements.get('budget', float('inf'))
        
        # 筛选符合条件的模型
        candidates = []
        for model, capabilities in self.model_capabilities.items():
            if needs_functions and not capabilities['supports_functions']:
                continue
            if needs_vision and not capabilities['supports_vision']:
                continue
            if capabilities['max_tokens'] < max_context:
                continue
            if capabilities['cost_per_token'] > budget_constraint:
                continue
            candidates.append(model)
        
        # 根据任务类型选择最优模型
        if 'coding' in task.lower() or 'programming' in task.lower():
            # 编程任务优先选择代码能力强的模型
            preferred = ['gpt-4', 'claude-3-sonnet', 'qwen-max']
        elif 'reasoning' in task.lower():
            # 推理任务优先选择推理能力强的模型
            preferred = ['gpt-4', 'claude-3-opus', 'gemini-1.5-pro']
        else:
            # 通用任务选择性价比高的模型
            preferred = ['gpt-3.5-turbo', 'claude-3-haiku', 'qwen-turbo']
        
        # 返回第一个可用的首选模型
        for model in preferred:
            if model in candidates:
                return model
        
        # 如果没有首选模型，返回第一个候选
        return candidates[0] if candidates else 'gpt-3.5-turbo'

class EnhancedUniversalAgent:
    """增强版全能智能体"""
    
    def __init__(self):
        self.cli_manager = CLIManager()
        self.model_adapter = ModelAdapter()
        self.expertise_domains = self._initialize_expertise()
        self.current_domain = None
        self.performance_metrics = {
            'tasks_completed': 0,
            'success_rate': 0.0,
            'avg_response_time': 0.0,
            'user_satisfaction': 0.0
        }
        
    def _initialize_expertise(self) -> Dict[ExpertiseDomain, ExpertCapability]:
        """初始化专家能力"""
        expertise = {}
        
        # 编程语言专家
        for domain in [ExpertiseDomain.PYTHON, ExpertiseDomain.JAVASCRIPT, ExpertiseDomain.JAVA]:
            expertise[domain] = ExpertCapability(
                domain=domain,
                proficiency=0.95,
                experience_years=10,
                recent_projects=[f"{domain.value}_project_1", f"{domain.value}_project_2"],
                tools=[f"{domain.value}_ide", f"{domain.value}_linter"],
                certifications=[f"{domain.value.upper()}_Expert"]
            )
        
        # 框架专家
        for domain in [ExpertiseDomain.REACT, ExpertiseDomain.DJANGO, ExpertiseDomain.FLASK]:
            expertise[domain] = ExpertCapability(
                domain=domain,
                proficiency=0.90,
                experience_years=8,
                recent_projects=[f"{domain.value}_app", f"{domain.value}_api"],
                tools=[f"{domain.value}_cli", f"{domain.value}_devtools"],
                certifications=[f"{domain.value.upper()}_Professional"]
            )
        
        # 专业领域专家
        for domain in [ExpertiseDomain.AI_ML, ExpertiseDomain.DATA_SCIENCE, ExpertiseDomain.SECURITY]:
            expertise[domain] = ExpertCapability(
                domain=domain,
                proficiency=0.92,
                experience_years=12,
                recent_projects=[f"{domain.value}_system", f"{domain.value}_platform"],
                tools=[f"{domain.value}_tools", f"{domain.value}_framework"],
                certifications=[f"{domain.value.upper()}_Master"]
            )
        
        return expertise
    
    async def analyze_task(self, task_description: str) -> Dict[str, Any]:
        """分析任务并选择最优专家领域"""
        # 任务特征提取
        task_features = self._extract_task_features(task_description)
        
        # 领域匹配
        domain_scores = {}
        for domain, capability in self.expertise_domains.items():
            score = self._calculate_domain_match(task_features, domain, capability)
            domain_scores[domain] = score
        
        # 选择最佳领域
        best_domain = max(domain_scores, key=domain_scores.get)
        self.current_domain = best_domain
        
        # 选择最优模型
        model_requirements = {
            'functions': True,
            'vision': 'image' in task_description.lower(),
            'max_context': 128000,
            'budget': 0.1
        }
        optimal_model = self.model_adapter.select_optimal_model(task_description, model_requirements)
        
        return {
            'selected_domain': best_domain,
            'domain_scores': domain_scores,
            'optimal_model': optimal_model,
            'task_features': task_features,
            'confidence': domain_scores[best_domain]
        }
    
    def _extract_task_features(self, task_description: str) -> Dict[str, Any]:
        """提取任务特征"""
        features = {
            'keywords': [],
            'complexity': 'medium',
            'type': 'general',
            'requirements': []
        }
        
        # 关键词提取
        keywords = re.findall(r'\b\w+\b', task_description.lower())
        features['keywords'] = list(set(keywords))
        
        # 复杂度判断
        if any(word in task_description.lower() for word in ['complex', 'difficult', 'advanced']):
            features['complexity'] = 'high'
        elif any(word in task_description.lower() for word in ['simple', 'basic', 'easy']):
            features['complexity'] = 'low'
        
        # 任务类型判断
        if any(word in task_description.lower() for word in ['code', 'program', 'develop', 'implement']):
            features['type'] = 'coding'
        elif any(word in task_description.lower() for word in ['analyze', 'research', 'investigate']):
            features['type'] = 'analysis'
        elif any(word in task_description.lower() for word in ['design', 'architecture', 'plan']):
            features['type'] = 'design'
        
        return features
    
    def _calculate_domain_match(self, task_features: Dict[str, Any], domain: ExpertiseDomain, capability: ExpertCapability) -> float:
        """计算领域匹配分数"""
        score = 0.0
        
        # 基础能力分数
        score += capability.proficiency * 0.4
        
        # 关键词匹配
        domain_keywords = {
            ExpertiseDomain.PYTHON: ['python', 'django', 'flask', 'fastapi'],
            ExpertiseDomain.JAVASCRIPT: ['javascript', 'react', 'vue', 'nodejs'],
            ExpertiseDomain.REACT: ['react', 'jsx', 'component', 'frontend'],
            ExpertiseDomain.AI_ML: ['ai', 'ml', 'machine learning', 'neural', 'model'],
            ExpertiseDomain.DATA_SCIENCE: ['data', 'analysis', 'pandas', 'numpy'],
            ExpertiseDomain.SECURITY: ['security', 'authentication', 'encryption', 'vulnerability']
        }
        
        if domain in domain_keywords:
            matching_keywords = sum(1 for kw in domain_keywords[domain] if kw in task_features['keywords'])
            score += (matching_keywords / len(domain_keywords[domain])) * 0.3
        
        # 任务类型匹配
        task_type_match = {
            'coding': [ExpertiseDomain.PYTHON, ExpertiseDomain.JAVASCRIPT, ExpertiseDomain.JAVA],
            'analysis': [ExpertiseDomain.DATA_SCIENCE, ExpertiseDomain.AI_ML],
            'design': [ExpertiseDomain.ARCHITECTURE, ExpertiseDomain.SYSTEM_DESIGN]
        }
        
        if task_features['type'] in task_type_match:
            if domain in task_type_match[task_features['type']]:
                score += 0.3
        
        return min(score, 1.0)
    
    async def execute_task(self, task_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行任务"""
        start_time = time.time()
        
        try:
            # 分析任务
            analysis = await self.analyze_task(task_description)
            
            # 准备执行环境
            execution_context = {
                'task': task_description,
                'analysis': analysis,
                'context': context or {},
                'domain': self.current_domain,
                'model': analysis['optimal_model']
            }
            
            # 执行任务（这里应该调用实际的AI模型）
            result = await self._execute_with_model(execution_context)
            
            # 更新性能指标
            execution_time = time.time() - start_time
            self._update_performance_metrics(True, execution_time)
            
            return {
                'success': True,
                'result': result,
                'analysis': analysis,
                'execution_time': execution_time,
                'domain': self.current_domain.value,
                'model_used': analysis['optimal_model']
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_performance_metrics(False, execution_time)
            
            logger.error(f"Task execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'domain': self.current_domain.value if self.current_domain else 'unknown'
            }
    
    async def _execute_with_model(self, context: Dict[str, Any]) -> str:
        """使用模型执行任务"""
        # 这里应该调用实际的AI模型API
        # 模拟执行
        await asyncio.sleep(0.1)
        
        domain = context['domain']
        task = context['task']
        
        # 根据领域生成专业的响应
        if domain == ExpertiseDomain.PYTHON:
            return f"Python专家解决方案：针对任务'{task}'，我建议使用最新的Python 3.12特性..."
        elif domain == ExpertiseDomain.REACT:
            return f"React专家解决方案：针对任务'{task}'，我建议使用React 18和Hooks..."
        elif domain == ExpertiseDomain.AI_ML:
            return f"AI/ML专家解决方案：针对任务'{task}'，我建议使用PyTorch 2.0..."
        else:
            return f"全能专家解决方案：针对任务'{task}'，我将综合多个领域的知识..."
    
    def _update_performance_metrics(self, success: bool, execution_time: float):
        """更新性能指标"""
        self.performance_metrics['tasks_completed'] += 1
        
        if success:
            # 更新成功率
            current_success = self.performance_metrics['success_rate'] * (self.performance_metrics['tasks_completed'] - 1)
            new_success = current_success + 1
            self.performance_metrics['success_rate'] = new_success / self.performance_metrics['tasks_completed']
        
        # 更新平均响应时间
        current_avg = self.performance_metrics['avg_response_time'] * (self.performance_metrics['tasks_completed'] - 1)
        new_avg = current_avg + execution_time
        self.performance_metrics['avg_response_time'] = new_avg / self.performance_metrics['tasks_completed']
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            'metrics': self.performance_metrics,
            'expertise_domains': {domain.value: asdict(capability) for domain, capability in self.expertise_domains.items()},
            'current_domain': self.current_domain.value if self.current_domain else None,
            'supported_models': self.model_adapter.supported_models,
            'timestamp': datetime.now().isoformat()
        }

# 全局实例
_enhanced_agent = None

def get_enhanced_agent() -> EnhancedUniversalAgent:
    """获取增强版全能智能体实例"""
    global _enhanced_agent
    if _enhanced_agent is None:
        _enhanced_agent = EnhancedUniversalAgent()
    return _enhanced_agent

# 使用示例
async def main():
    """主函数示例"""
    agent = get_enhanced_agent()
    
    # 测试任务
    tasks = [
        "创建一个Python Flask API",
        "设计一个React组件库",
        "实现一个机器学习模型",
        "分析数据集并生成报告"
    ]
    
    for task in tasks:
        print(f"\n执行任务: {task}")
        result = await agent.execute_task(task)
        print(f"结果: {result['success']}")
        print(f"领域: {result['domain']}")
        print(f"模型: {result['model_used']}")
        print(f"执行时间: {result['execution_time']:.2f}s")
    
    # 性能报告
    report = agent.get_performance_report()
    print(f"\n性能报告:")
    print(f"任务完成数: {report['metrics']['tasks_completed']}")
    print(f"成功率: {report['metrics']['success_rate']:.2%}")
    print(f"平均响应时间: {report['metrics']['avg_response_time']:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
