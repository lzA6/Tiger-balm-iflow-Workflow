#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全能工作流V5 - 多模型兼容适配器
Multi-Model Compatibility Adapter

作者: Universal AI Team
版本: 5.0.0
日期: 2025-11-12

特性:
- 支持所有主流LLM模型
- 智能路由和负载均衡
- 协议自动适配
- 性能监控和优化
- 故障转移和容错
- 成本优化
"""

import os
import sys
import json
import time
import asyncio
import logging
import aiohttp
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import yaml
import statistics

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTier(Enum):
    """模型等级"""
    TIER1 = "tier1"      # 顶级模型
    TIER2 = "tier2"      # 高级模型
    TIER3 = "tier3"      # 中级模型
    LOCAL = "local"      # 本地模型

class TaskType(Enum):
    """任务类型"""
    REASONING = "reasoning"
    CODING = "coding"
    ANALYSIS = "analysis"
    VISION = "vision"
    WRITING = "writing"
    CHINESE = "chinese"
    MULTIMODAL = "multimodal"

class ProviderType(Enum):
    """提供商类型"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    ALIBABA = "alibaba"
    ZHIPU = "zhipu"
    DEEPSEEK = "deepseek"
    MOONSHOT = "moonshot"
    LOCAL = "local"

@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    provider: ProviderType
    tier: ModelTier
    api_endpoint: str
    max_tokens: int
    supports_functions: bool
    supports_vision: bool
    supports_code_execution: bool
    cost_per_1k_tokens: float
    latency_ms: float
    reliability: float
    capabilities: List[str]
    api_key: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.9
    timeout: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'provider': self.provider.value,
            'tier': self.tier.value,
            'api_endpoint': self.api_endpoint,
            'max_tokens': self.max_tokens,
            'supports_functions': self.supports_functions,
            'supports_vision': self.supports_vision,
            'supports_code_execution': self.supports_code_execution,
            'cost_per_1k_tokens': self.cost_per_1k_tokens,
            'latency_ms': self.latency_ms,
            'reliability': self.reliability,
            'capabilities': self.capabilities,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'timeout': self.timeout
        }

@dataclass
class RequestMetrics:
    """请求指标"""
    model_name: str
    task_type: TaskType
    request_time: datetime
    response_time: float
    tokens_used: int
    cost: float
    success: bool
    error_message: Optional[str] = None

class ModelAdapter:
    """模型适配器"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            '.iflow', 'config', 'model-adapter.yaml'
        )
        self.models: Dict[str, ModelConfig] = {}
        self.task_type_mapping: Dict[TaskType, Dict[str, Any]] = {}
        self.routing_algorithm: Dict[str, Any] = {}
        self.protocol_adapters: Dict[str, Any] = {}
        self.fault_tolerance: Dict[str, Any] = {}
        
        # 性能监控
        self.request_history: deque = deque(maxlen=10000)
        self.model_stats: Dict[str, Dict] = defaultdict(dict)
        self.health_status: Dict[str, bool] = {}
        
        # 负载均衡
        self.load_balancer = LoadBalancer()
        
        # 缓存
        self.response_cache = ResponseCache()
        
        # 初始化
        self._initialize_adapter()
    
    def _initialize_adapter(self):
        """初始化适配器"""
        # 加载配置
        self._load_config()
        
        # 初始化模型
        self._initialize_models()
        
        # 启动健康检查
        asyncio.create_task(self._health_check_loop())
        
        logger.info(f"模型适配器初始化完成，加载了 {len(self.models)} 个模型")
    
    def _load_config(self):
        """加载配置"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
                # 提取配置
                self.task_type_mapping = config.get('task_type_mapping', {})
                self.routing_algorithm = config.get('routing_algorithm', {})
                self.protocol_adapters = config.get('protocol_adapters', {})
                self.fault_tolerance = config.get('fault_tolerance', {})
    
    def _initialize_models(self):
        """初始化模型"""
        # 从配置文件加载模型
        model_categories = self._load_model_categories()
        
        for tier_name, models in model_categories.items():
            tier = ModelTier(tier_name)
            for model_data in models:
                model_config = ModelConfig(
                    name=model_data['name'],
                    provider=ProviderType(model_data['provider']),
                    tier=tier,
                    api_endpoint=model_data['api_endpoint'],
                    max_tokens=model_data['max_tokens'],
                    supports_functions=model_data['supports_functions'],
                    supports_vision=model_data['supports_vision'],
                    supports_code_execution=model_data['supports_code_execution'],
                    cost_per_1k_tokens=model_data['cost_per_1k_tokens'],
                    latency_ms=model_data['latency_ms'],
                    reliability=model_data['reliability'],
                    capabilities=model_data['capabilities'],
                    api_key=os.getenv(f"{model_data['provider'].upper()}_API_KEY"),
                    temperature=0.7,
                    top_p=0.9,
                    timeout=60
                )
                
                self.models[model_config.name] = model_config
                self.health_status[model_config.name] = True
    
    def _load_model_categories(self) -> Dict[str, List[Dict]]:
        """加载模型分类"""
        # 默认模型配置（实际应该从配置文件加载）
        return {
            'tier1': [
                {
                    'name': 'GPT-4-Turbo',
                    'provider': 'openai',
                    'api_endpoint': 'https://api.openai.com/v1/chat/completions',
                    'max_tokens': 128000,
                    'supports_functions': True,
                    'supports_vision': True,
                    'supports_code_execution': True,
                    'cost_per_1k_tokens': 0.03,
                    'latency_ms': 500,
                    'reliability': 0.99,
                    'capabilities': ['reasoning', 'coding', 'analysis', 'vision', 'math']
                },
                {
                    'name': 'Claude-3.5-Sonnet',
                    'provider': 'anthropic',
                    'api_endpoint': 'https://api.anthropic.com/v1/messages',
                    'max_tokens': 200000,
                    'supports_functions': True,
                    'supports_vision': True,
                    'supports_code_execution': True,
                    'cost_per_1k_tokens': 0.015,
                    'latency_ms': 600,
                    'reliability': 0.98,
                    'capabilities': ['reasoning', 'coding', 'analysis', 'vision', 'writing']
                }
            ],
            'tier2': [
                {
                    'name': 'GPT-4',
                    'provider': 'openai',
                    'api_endpoint': 'https://api.openai.com/v1/chat/completions',
                    'max_tokens': 8192,
                    'supports_functions': True,
                    'supports_vision': False,
                    'supports_code_execution': True,
                    'cost_per_1k_tokens': 0.06,
                    'latency_ms': 800,
                    'reliability': 0.98,
                    'capabilities': ['reasoning', 'coding', 'analysis', 'math']
                },
                {
                    'name': 'DeepSeek-V2.5',
                    'provider': 'deepseek',
                    'api_endpoint': 'https://api.deepseek.com/v1/chat/completions',
                    'max_tokens': 128000,
                    'supports_functions': True,
                    'supports_vision': False,
                    'supports_code_execution': True,
                    'cost_per_1k_tokens': 0.002,
                    'latency_ms': 700,
                    'reliability': 0.95,
                    'capabilities': ['reasoning', 'coding', 'analysis', 'math']
                }
            ],
            'tier3': [
                {
                    'name': 'Qwen-2.5-72B',
                    'provider': 'alibaba',
                    'api_endpoint': 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation',
                    'max_tokens': 32768,
                    'supports_functions': True,
                    'supports_vision': False,
                    'supports_code_execution': True,
                    'cost_per_1k_tokens': 0.001,
                    'latency_ms': 900,
                    'reliability': 0.94,
                    'capabilities': ['reasoning', 'coding', 'analysis', 'chinese']
                }
            ]
        }
    
    async def route_request(self, 
                          task_type: TaskType,
                          messages: List[Dict[str, Any]],
                          functions: Optional[List[Dict]] = None,
                          model_preference: Optional[str] = None,
                          budget_limit: Optional[float] = None) -> Dict[str, Any]:
        """路由请求到最优模型"""
        try:
            # 选择最优模型
            selected_model = await self._select_optimal_model(
                task_type, model_preference, budget_limit
            )
            
            if not selected_model:
                return {'error': '没有可用的模型'}
            
            # 适配请求格式
            adapted_request = await self._adapt_request(
                selected_model, messages, functions
            )
            
            # 发送请求
            start_time = time.time()
            response = await self._send_request(selected_model, adapted_request)
            response_time = time.time() - start_time
            
            # 记录指标
            self._record_metrics(selected_model, task_type, response_time, response)
            
            # 适配响应格式
            adapted_response = await self._adapt_response(selected_model, response)
            
            return adapted_response
            
        except Exception as e:
            logger.error(f"请求路由失败: {e}")
            # 尝试故障转移
            return await self._handle_failover(task_type, messages, functions)
    
    async def _select_optimal_model(self,
                                  task_type: TaskType,
                                  preference: Optional[str],
                                  budget_limit: Optional[float]) -> Optional[ModelConfig]:
        """选择最优模型"""
        # 获取候选模型
        candidates = self._get_candidate_models(task_type)
        
        if not candidates:
            return None
        
        # 如果有偏好且可用
        if preference and preference in self.models and self.health_status.get(preference, False):
            return self.models[preference]
        
        # 计算权重分数
        scored_models = []
        for model_name in candidates:
            model = self.models[model_name]
            if not self.health_status.get(model_name, False):
                continue
            
            # 计算综合分数
            score = self._calculate_model_score(model, task_type, budget_limit)
            scored_models.append((model, score))
        
        # 按分数排序
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        return scored_models[0][0] if scored_models else None
    
    def _get_candidate_models(self, task_type: TaskType) -> List[str]:
        """获取候选模型"""
        mapping = self.task_type_mapping.get(task_type.value, {})
        preferred = mapping.get('preferred_models', [])
        fallback = mapping.get('fallback_models', [])
        
        # 合并并去重
        candidates = list(dict.fromkeys(preferred + fallback))
        
        # 过滤可用模型
        return [name for name in candidates if name in self.models]
    
    def _calculate_model_score(self,
                             model: ModelConfig,
                             task_type: TaskType,
                             budget_limit: Optional[float]) -> float:
        """计算模型分数"""
        # 获取权重配置
        weights = self.routing_algorithm.get('weight_calculation', {
            'capability_match': 0.4,
            'performance_score': 0.3,
            'cost_efficiency': 0.2,
            'availability': 0.1
        })
        
        score = 0.0
        
        # 能力匹配分数
        capability_score = 1.0 if task_type.value in model.capabilities else 0.5
        score += capability_score * weights['capability_match']
        
        # 性能分数
        performance_score = self._get_performance_score(model.name)
        score += performance_score * weights['performance_score']
        
        # 成本效率分数
        if budget_limit:
            cost_score = max(0, 1.0 - (model.cost_per_1k_tokens / budget_limit))
        else:
            cost_score = 1.0 - min(model.cost_per_1k_tokens / 0.1, 1.0)
        score += cost_score * weights['cost_efficiency']
        
        # 可用性分数
        availability_score = 1.0 if self.health_status.get(model.name, False) else 0.0
        score += availability_score * weights['availability']
        
        return score
    
    def _get_performance_score(self, model_name: str) -> float:
        """获取性能分数"""
        stats = self.model_stats.get(model_name, {})
        
        # 响应时间分数
        avg_response_time = stats.get('avg_response_time', 1000)
        time_score = max(0, 1.0 - (avg_response_time / 5000))  # 5秒为0分
        
        # 可靠性分数
        success_rate = stats.get('success_rate', 0.95)
        
        # 综合性能分数
        return (time_score + success_rate) / 2
    
    async def _adapt_request(self,
                           model: ModelConfig,
                           messages: List[Dict[str, Any]],
                           functions: Optional[List[Dict]]) -> Dict[str, Any]:
        """适配请求格式"""
        adapter = self.protocol_adapters.get(model.provider.value, {})
        
        # 根据提供商适配请求格式
        if model.provider == ProviderType.OPENAI:
            return self._adapt_openai_request(model, messages, functions)
        elif model.provider == ProviderType.ANTHROPIC:
            return self._adapt_anthropic_request(model, messages, functions)
        elif model.provider == ProviderType.GOOGLE:
            return self._adapt_google_request(model, messages, functions)
        else:
            # 默认OpenAI格式
            return self._adapt_openai_request(model, messages, functions)
    
    def _adapt_openai_request(self,
                            model: ModelConfig,
                            messages: List[Dict[str, Any]],
                            functions: Optional[List[Dict]]) -> Dict[str, Any]:
        """适配OpenAI请求格式"""
        request = {
            'model': model.name,
            'messages': messages,
            'temperature': model.temperature,
            'max_tokens': model.max_tokens,
            'top_p': model.top_p
        }
        
        if functions and model.supports_functions:
            request['functions'] = functions
            request['function_call'] = 'auto'
        
        return request
    
    def _adapt_anthropic_request(self,
                               model: ModelConfig,
                               messages: List[Dict[str, Any]],
                               functions: Optional[List[Dict]]) -> Dict[str, Any]:
        """适配Anthropic请求格式"""
        # 转换消息格式
        anthropic_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                # Anthropic使用system参数
                continue
            anthropic_messages.append({
                'role': msg['role'],
                'content': msg['content']
            })
        
        request = {
            'model': model.name,
            'messages': anthropic_messages,
            'max_tokens': model.max_tokens,
            'temperature': model.temperature
        }
        
        # 添加system消息
        system_messages = [msg['content'] for msg in messages if msg['role'] == 'system']
        if system_messages:
            request['system'] = '\n'.join(system_messages)
        
        # 添加工具
        if functions and model.supports_functions:
            request['tools'] = functions
        
        return request
    
    def _adapt_google_request(self,
                            model: ModelConfig,
                            messages: List[Dict[str, Any]],
                            functions: Optional[List[Dict]]) -> Dict[str, Any]:
        """适配Google请求格式"""
        # 转换消息格式
        contents = []
        for msg in messages:
            if msg['role'] == 'system':
                # Google使用system instruction
                continue
            contents.append({
                'role': 'user' if msg['role'] == 'user' else 'model',
                'parts': [{'text': msg['content']}]
            })
        
        request = {
            'model': f"models/{model.name}",
            'contents': contents,
            'generationConfig': {
                'temperature': model.temperature,
                'maxOutputTokens': model.max_tokens,
                'topP': model.top_p
            }
        }
        
        # 添加系统指令
        system_messages = [msg['content'] for msg in messages if msg['role'] == 'system']
        if system_messages:
            request['systemInstruction'] = {
                'parts': [{'text': '\n'.join(system_messages)}]
            }
        
        # 添加工具
        if functions and model.supports_functions:
            request['tools'] = functions
        
        return request
    
    async def _send_request(self, model: ModelConfig, request: Dict[str, Any]) -> Dict[str, Any]:
        """发送请求"""
        headers = {
            'Content-Type': 'application/json'
        }
        
        # 添加认证
        if model.api_key:
            if model.provider == ProviderType.OPENAI:
                headers['Authorization'] = f'Bearer {model.api_key}'
            elif model.provider == ProviderType.ANTHROPIC:
                headers['x-api-key'] = model.api_key
                headers['anthropic-version'] = '2023-06-01'
            elif model.provider == ProviderType.GOOGLE:
                headers['x-goog-api-key'] = model.api_key
        
        # 发送请求
        timeout = aiohttp.ClientTimeout(total=model.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(model.api_endpoint, json=request, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    error_text = await resp.text()
                    raise Exception(f"API请求失败: {resp.status} - {error_text}")
    
    async def _adapt_response(self, model: ModelConfig, response: Dict[str, Any]) -> Dict[str, Any]:
        """适配响应格式"""
        # 统一响应格式
        adapted = {
            'model': model.name,
            'provider': model.provider.value,
            'content': '',
            'function_call': None,
            'usage': {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            },
            'finish_reason': '',
            'raw_response': response
        }
        
        # 根据提供商解析响应
        if model.provider == ProviderType.OPENAI:
            adapted.update(self._parse_openai_response(response))
        elif model.provider == ProviderType.ANTHROPIC:
            adapted.update(self._parse_anthropic_response(response))
        elif model.provider == ProviderType.GOOGLE:
            adapted.update(self._parse_google_response(response))
        
        return adapted
    
    def _parse_openai_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """解析OpenAI响应"""
        result = {}
        
        if 'choices' in response and response['choices']:
            choice = response['choices'][0]
            result['content'] = choice['message'].get('content', '')
            result['function_call'] = choice['message'].get('function_call')
            result['finish_reason'] = choice.get('finish_reason', '')
        
        if 'usage' in response:
            result['usage'] = response['usage']
        
        return result
    
    def _parse_anthropic_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """解析Anthropic响应"""
        result = {}
        
        if 'content' in response and response['content']:
            content = response['content'][0]
            result['content'] = content.get('text', '')
            if content.get('type') == 'tool_use':
                result['function_call'] = {
                    'name': content.get('name'),
                    'arguments': json.dumps(content.get('input', {}))
                }
        
        if 'usage' in response:
            result['usage'] = {
                'prompt_tokens': response['usage'].get('input_tokens', 0),
                'completion_tokens': response['usage'].get('output_tokens', 0),
                'total_tokens': response['usage'].get('input_tokens', 0) + response['usage'].get('output_tokens', 0)
            }
        
        result['finish_reason'] = response.get('stop_reason', '')
        
        return result
    
    def _parse_google_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """解析Google响应"""
        result = {}
        
        if 'candidates' in response and response['candidates']:
            candidate = response['candidates'][0]
            if 'content' in candidate and candidate['content'].get('parts'):
                result['content'] = candidate['content']['parts'][0].get('text', '')
            result['finish_reason'] = candidate.get('finishReason', '')
        
        if 'usageMetadata' in response:
            metadata = response['usageMetadata']
            result['usage'] = {
                'prompt_tokens': metadata.get('promptTokenCount', 0),
                'completion_tokens': metadata.get('candidatesTokenCount', 0),
                'total_tokens': metadata.get('totalTokenCount', 0)
            }
        
        return result
    
    def _record_metrics(self,
                       model: ModelConfig,
                       task_type: TaskType,
                       response_time: float,
                       response: Dict[str, Any]):
        """记录指标"""
        # 创建指标记录
        metrics = RequestMetrics(
            model_name=model.name,
            task_type=task_type,
            request_time=datetime.now(),
            response_time=response_time * 1000,  # 转换为毫秒
            tokens_used=response.get('usage', {}).get('total_tokens', 0),
            cost=self._calculate_cost(model, response),
            success='error' not in response
        )
        
        # 添加到历史
        self.request_history.append(metrics)
        
        # 更新模型统计
        self._update_model_stats(model.name, metrics)
    
    def _calculate_cost(self, model: ModelConfig, response: Dict[str, Any]) -> float:
        """计算成本"""
        total_tokens = response.get('usage', {}).get('total_tokens', 0)
        return (total_tokens / 1000) * model.cost_per_1k_tokens
    
    def _update_model_stats(self, model_name: str, metrics: RequestMetrics):
        """更新模型统计"""
        stats = self.model_stats[model_name]
        
        # 更新响应时间
        response_times = stats.get('response_times', [])
        response_times.append(metrics.response_time)
        stats['response_times'] = response_times[-100:]  # 保留最近100次
        stats['avg_response_time'] = statistics.mean(response_times)
        
        # 更新成功率
        success_count = stats.get('success_count', 0) + (1 if metrics.success else 0)
        total_count = stats.get('total_count', 0) + 1
        stats['success_count'] = success_count
        stats['total_count'] = total_count
        stats['success_rate'] = success_count / total_count
        
        # 更新成本
        total_cost = stats.get('total_cost', 0) + metrics.cost
        stats['total_cost'] = total_cost
        stats['avg_cost_per_request'] = total_cost / total_count
    
    async def _handle_failover(self,
                             task_type: TaskType,
                             messages: List[Dict[str, Any]],
                             functions: Optional[List[Dict]]) -> Dict[str, Any]:
        """处理故障转移"""
        # 获取备用模型
        candidates = self._get_candidate_models(task_type)
        
        for model_name in candidates:
            if self.health_status.get(model_name, False):
                try:
                    model = self.models[model_name]
                    adapted_request = await self._adapt_request(model, messages, functions)
                    response = await self._send_request(model, adapted_request)
                    return await self._adapt_response(model, response)
                except Exception as e:
                    logger.warning(f"备用模型 {model_name} 也失败: {e}")
                    continue
        
        return {'error': '所有模型都不可用'}
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(30)  # 30秒检查一次
            except Exception as e:
                logger.error(f"健康检查失败: {e}")
                await asyncio.sleep(60)
    
    async def _perform_health_checks(self):
        """执行健康检查"""
        for model_name, model in self.models.items():
            try:
                # 发送简单的健康检查请求
                test_request = {
                    'model': model.name,
                    'messages': [{'role': 'user', 'content': 'Hi'}],
                    'max_tokens': 10
                }
                
                start_time = time.time()
                response = await self._send_request(model, test_request)
                response_time = time.time() - start_time
                
                # 更新健康状态
                self.health_status[model_name] = response_time < 5.0  # 5秒内响应视为健康
                
            except Exception as e:
                logger.warning(f"模型 {model_name} 健康检查失败: {e}")
                self.health_status[model_name] = False
    
    def get_model_stats(self) -> Dict[str, Any]:
        """获取模型统计"""
        return {
            'total_models': len(self.models),
            'healthy_models': sum(1 for h in self.health_status.values() if h),
            'model_details': {
                name: {
                    'config': model.to_dict(),
                    'stats': self.model_stats.get(name, {}),
                    'healthy': self.health_status.get(name, False)
                }
                for name, model in self.models.items()
            }
        }
    
    def get_usage_report(self, days: int = 7) -> Dict[str, Any]:
        """获取使用报告"""
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_requests = [
            r for r in self.request_history 
            if r.request_time > cutoff_time
        ]
        
        report = {
            'period_days': days,
            'total_requests': len(recent_requests),
            'successful_requests': sum(1 for r in recent_requests if r.success),
            'total_cost': sum(r.cost for r in recent_requests),
            'total_tokens': sum(r.tokens_used for r in recent_requests),
            'avg_response_time': statistics.mean([r.response_time for r in recent_requests]) if recent_requests else 0,
            'model_usage': defaultdict(int),
            'task_type_usage': defaultdict(int)
        }
        
        # 统计模型使用情况
        for req in recent_requests:
            report['model_usage'][req.model_name] += 1
            report['task_type_usage'][req.task_type.value] += 1
        
        return report

class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self):
        self.round_robin_index = 0
        self.weighted_round_robin_weights = {}
    
    def select_model(self, models: List[str], strategy: str = 'round_robin') -> str:
        """选择模型"""
        if not models:
            return None
        
        if strategy == 'round_robin':
            return self._round_robin(models)
        elif strategy == 'weighted_round_robin':
            return self._weighted_round_robin(models)
        elif strategy == 'least_connections':
            return self._least_connections(models)
        else:
            return models[0]
    
    def _round_robin(self, models: List[str]) -> str:
        """轮询选择"""
        model = models[self.round_robin_index % len(models)]
        self.round_robin_index += 1
        return model
    
    def _weighted_round_robin(self, models: List[str]) -> str:
        """加权轮询选择"""
        # 简化实现
        return self._round_robin(models)
    
    def _least_connections(self, models: List[str]) -> str:
        """最少连接选择"""
        # 简化实现
        return models[0]

class ResponseCache:
    """响应缓存"""
    
    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        self.ttl = ttl
        self.max_size = max_size
        self.cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """获取缓存"""
        if key in self.cache:
            response, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                return response
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, response: Dict[str, Any]):
        """设置缓存"""
        if len(self.cache) >= self.max_size:
            # 删除最旧的缓存
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (response, datetime.now())
    
    def _generate_key(self, messages: List[Dict], model: str) -> str:
        """生成缓存键"""
        content = json.dumps({'messages': messages, 'model': model}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

# 全局适配器实例
adapter = ModelAdapter()

# 便捷函数
async def chat_completion(task_type: TaskType,
                         messages: List[Dict[str, Any]],
                         **kwargs) -> Dict[str, Any]:
    """聊天完成"""
    return await adapter.route_request(task_type, messages, **kwargs)

def get_model_stats() -> Dict[str, Any]:
    """获取模型统计"""
    return adapter.get_model_stats()

def get_usage_report(days: int = 7) -> Dict[str, Any]:
    """获取使用报告"""
    return adapter.get_usage_report(days)

if __name__ == "__main__":
    # 测试代码
    async def test_adapter():
        print("模型适配器测试")
        print("=" * 50)
        
        # 获取模型统计
        stats = get_model_stats()
        print(f"总模型数: {stats['total_models']}")
        print(f"健康模型数: {stats['healthy_models']}")
        
        # 测试请求
        messages = [
            {'role': 'user', 'content': '你好，请介绍一下Python'}
        ]
        
        response = await chat_completion(
            TaskType.REASONING,
            messages
        )
        
        print(f"\n测试响应: {response.get('content', '')[:100]}...")
    
    asyncio.run(test_adapter())