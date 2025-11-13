#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌌 简化测试 V4 (Simple Test V4)
简化的测试脚本，验证核心功能。
"""

import os
import sys
import json
import asyncio
import logging
import time
from pathlib import Path
from datetime import datetime

# 动态添加项目根目录到sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_basic_imports():
    """测试基础导入"""
    logger.info("🔍 测试基础导入...")
    
    try:
        from iflow.adapters.universal_llm_adapter import UltimateLLMAdapter
        logger.info("✅ UltimateLLMAdapter 导入成功")
    except Exception as e:
        logger.error(f"❌ UltimateLLMAdapter 导入失败: {e}")
        return False
    
    try:
        from iflow.core.ultimate_arq_engine import UltimateAREngine
        logger.info("✅ UltimateAREngine 导入成功")
    except Exception as e:
        logger.error(f"❌ UltimateAREngine 导入失败: {e}")
        return False
    
    try:
        from iflow.agents.ultimate_fusion_agent_v4 import UltimateFusionAgentV4
        logger.info("✅ UltimateFusionAgentV4 导入成功")
    except Exception as e:
        logger.error(f"❌ UltimateFusionAgentV4 导入失败: {e}")
        return False
    
    try:
        from iflow.hooks.auto_intelligent_quality_system import AutoIntelligentQualitySystemV4
        logger.info("✅ AutoIntelligentQualitySystemV4 导入成功")
    except Exception as e:
        logger.error(f"❌ AutoIntelligentQualitySystemV4 导入失败: {e}")
        return False
    
    try:
        from iflow.hooks.comprehensive_hook_manager_v4 import ComprehensiveHookManagerV4
        logger.info("✅ ComprehensiveHookManagerV4 导入成功")
    except Exception as e:
        logger.error(f"❌ ComprehensiveHookManagerV4 导入失败: {e}")
        return False
    
    try:
        from iflow.core.self_evolution_engine_v4 import SelfEvolutionEngineV4
        logger.info("✅ SelfEvolutionEngineV4 导入成功")
    except Exception as e:
        logger.error(f"❌ SelfEvolutionEngineV4 导入失败: {e}")
        return False
    
    return True

async def test_adapter_initialization():
    """测试适配器初始化"""
    logger.info("🔧 测试适配器初始化...")
    
    try:
        from iflow.adapters.universal_llm_adapter import UltimateLLMAdapter
        adapter = UltimateLLMAdapter()
        logger.info("✅ UltimateLLMAdapter 初始化成功")
        
        # 测试模型配置
        models = adapter.get_available_models()
        logger.info(f"📋 可用模型数量: {len(models)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ 适配器初始化失败: {e}")
        return False

async def test_quality_system():
    """测试质量系统"""
    logger.info("🔍 测试质量系统...")
    
    try:
        from iflow.hooks.auto_intelligent_quality_system import AutoIntelligentQualitySystemV4
        quality_system = AutoIntelligentQualitySystemV4()
        await quality_system.initialize()
        logger.info("✅ 质量系统初始化成功")
        
        # 测试文件检查
        test_file = "C:/Users/Administrator.DESKTOP-EGNE9ND/Desktop/测试工作流/A项目/iflow/tests/simple_test_v4.py"
        if Path(test_file).exists():
            report = await quality_system.check_file(test_file)
            logger.info(f"📊 文件检查完成，问题数: {report.total_issues}")
        
        return True
    except Exception as e:
        logger.error(f"❌ 质量系统测试失败: {e}")
        return False

async def test_hook_manager():
    """测试Hook管理器"""
    logger.info("🔗 测试Hook管理器...")
    
    try:
        from iflow.hooks.comprehensive_hook_manager_v4 import ComprehensiveHookManagerV4
        hook_manager = ComprehensiveHookManagerV4()
        await hook_manager.initialize()
        logger.info("✅ Hook管理器初始化成功")
        
        # 获取统计信息
        stats = hook_manager.get_hook_statistics()
        logger.info(f"📈 Hook统计: {stats['total_hooks']} 个Hook")
        
        return True
    except Exception as e:
        logger.error(f"❌ Hook管理器测试失败: {e}")
        return False

async def test_evolution_engine():
    """测试进化引擎"""
    logger.info("🧬 测试进化引擎...")
    
    try:
        from iflow.core.self_evolution_engine_v4 import SelfEvolutionEngineV4
        evolution_engine = SelfEvolutionEngineV4()
        logger.info("✅ 进化引擎初始化成功")
        
        # 获取统计信息
        stats = evolution_engine.get_evolution_statistics()
        logger.info(f"📊 进化统计: {stats['total_records']} 条记录")
        
        return True
    except Exception as e:
        logger.error(f"❌ 进化引擎测试失败: {e}")
        return False

async def generate_test_report(results):
    """生成测试报告"""
    report = {
        "test_timestamp": datetime.now().isoformat(),
        "test_results": results,
        "summary": {
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r["success"]),
            "failed_tests": sum(1 for r in results if not r["success"])
        }
    }
    
    # 保存报告
    report_path = project_root / "A项目" / "iflow" / "reports" / "simple_test_report_v4.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 测试报告已保存到: {report_path}")
    return report

async def main():
    """主测试函数"""
    logger.info("🚀 开始运行简化测试套件 V4...")
    
    results = []
    
    # 运行各项测试
    tests = [
        ("基础导入测试", test_basic_imports),
        ("适配器初始化测试", test_adapter_initialization),
        ("质量系统测试", test_quality_system),
        ("Hook管理器测试", test_hook_manager),
        ("进化引擎测试", test_evolution_engine)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"运行 {test_name}...")
        success = await test_func()
        results.append({
            "name": test_name,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
    
    # 生成报告
    report = await generate_test_report(results)
    
    # 显示结果
    logger.info("\n" + "="*50)
    logger.info("📊 测试结果摘要:")
    summary = report["summary"]
    logger.info(f"总测试数: {summary['total_tests']}")
    logger.info(f"通过测试: {summary['passed_tests']}")
    logger.info(f"失败测试: {summary['failed_tests']}")
    logger.info(f"成功率: {summary['passed_tests']/summary['total_tests']:.2%}")
    
    logger.info("\n✅ 简化测试套件运行完成")

if __name__ == "__main__":
    asyncio.run(main())