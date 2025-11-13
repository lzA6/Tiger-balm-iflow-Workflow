#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 测试运行器
Test Runner

简化的测试运行器，用于验证系统功能
"""

import sys
import os
import time
import json
from pathlib import Path

# 添加.iflow到Python路径
IFLOW_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(IFLOW_ROOT))

def test_basic_functionality():
    """测试基本功能"""
    results = {}
    
    # 测试1: 检查目录结构
    print("测试1: 检查目录结构...")
    required_dirs = ['agents', 'tools', 'workflows', 'config', 'core']
    existing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = IFLOW_ROOT / dir_name
        if dir_path.exists():
            existing_dirs.append(dir_name)
    
    results['directory_structure'] = {
        'required': required_dirs,
        'existing': existing_dirs,
        'passed': len(existing_dirs) == len(required_dirs)
    }
    
    # 测试2: 检查核心文件
    print("测试2: 检查核心文件...")
    required_files = [
        'core/iflow-cli.py',
        'tools/enhanced-universal-agent.py',
        'tools/quantum-performance-optimizer.py',
        'tools/self-evolution-engine.py',
        'config/universal-model-adapter.yaml'
    ]
    existing_files = []
    
    for file_path in required_files:
        full_path = IFLOW_ROOT / file_path
        if full_path.exists():
            existing_files.append(file_path)
    
    results['core_files'] = {
        'required': required_files,
        'existing': existing_files,
        'passed': len(existing_files) == len(required_files)
    }
    
    # 测试3: 检查Python导入
    print("测试3: 检查Python导入...")
    import_tests = []
    
    try:
        # 测试导入增强智能体
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "enhanced_universal_agent",
            IFLOW_ROOT / "tools" / "enhanced-universal-agent.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import_tests.append(('enhanced_universal_agent', True, None))
    except Exception as e:
        import_tests.append(('enhanced_universal_agent', False, str(e)))
    
    results['python_imports'] = {
        'tests': import_tests,
        'passed': all(test[1] for test in import_tests)
    }
    
    # 测试4: 检查配置文件
    print("测试4: 检查配置文件...")
    config_tests = []
    
    try:
        import yaml
        with open(IFLOW_ROOT / "config" / "universal-model-adapter.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查关键配置项
        has_supported_models = 'supported_models' in config
        has_openai = 'openai' in config.get('supported_models', {})
        has_chinese_models = any(model in config.get('supported_models', {}) 
                               for model in ['baidu', 'alibaba', 'tencent', 'bytedance'])
        
        config_tests.append(('universal-model-adapter', has_supported_models and has_openai and has_chinese_models, None))
    except Exception as e:
        config_tests.append(('universal-model-adapter', False, str(e)))
    
    results['config_files'] = {
        'tests': config_tests,
        'passed': all(test[1] for test in config_tests)
    }
    
    return results

def test_cli_functionality():
    """测试CLI功能"""
    print("测试5: 检查CLI功能...")
    
    try:
        # 测试CLI模块导入
        spec = importlib.util.spec_from_file_location(
            "iflow_cli",
            IFLOW_ROOT / "core" / "iflow-cli.py"
        )
        cli_module = importlib.util.module_from_spec(spec)
        
        # 检查CLI类是否存在
        has_cli_class = hasattr(cli_module, 'iFlowCLI')
        
        return {
            'cli_import': True,
            'cli_class_exists': has_cli_class,
            'passed': has_cli_class
        }
    except Exception as e:
        return {
            'cli_import': False,
            'cli_class_exists': False,
            'passed': False,
            'error': str(e)
        }

def main():
    """主函数"""
    print("🧪 开始系统测试...")
    start_time = time.time()
    
    # 运行测试
    basic_results = test_basic_functionality()
    cli_results = test_cli_functionality()
    
    # 汇总结果
    all_tests = list(basic_results.values()) + [cli_results]
    passed_tests = sum(1 for test in all_tests if test.get('passed', False))
    total_tests = len(all_tests)
    
    duration = time.time() - start_time
    
    # 输出结果
    print(f"\n=== 测试结果 ===")
    print(f"总测试数: {total_tests}")
    print(f"通过: {passed_tests}")
    print(f"失败: {total_tests - passed_tests}")
    print(f"成功率: {passed_tests/total_tests:.1%}")
    print(f"耗时: {duration:.2f}s")
    
    # 详细结果
    print(f"\n=== 详细结果 ===")
    for test_name, result in basic_results.items():
        status = "✅ 通过" if result['passed'] else "❌ 失败"
        print(f"{test_name}: {status}")
        if not result['passed']:
            print(f"  详情: {result}")
    
    status = "✅ 通过" if cli_results['passed'] else "❌ 失败"
    print(f"CLI功能: {status}")
    if not cli_results['passed']:
        print(f"  详情: {cli_results}")
    
    # 保存测试报告
    report = {
        'timestamp': time.time(),
        'duration': duration,
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': passed_tests/total_tests,
        'basic_results': basic_results,
        'cli_results': cli_results
    }
    
    report_path = IFLOW_ROOT / "test_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n测试报告已保存到: {report_path}")
    
    # 返回成功状态
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)