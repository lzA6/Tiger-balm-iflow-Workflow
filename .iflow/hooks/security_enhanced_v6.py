#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ 安全增强Hook V6 (Security Enhanced Hook V6)
T-MIA凤凰架构的安全守护者，提供全方位的安全检查和防护

你一定要超级思考、极限思考、深度思考，全力思考、超强思考，认真仔细思考（ultrathink、think really super hard、think intensely）。
"""

import os
import sys
import json
import asyncio
import logging
import hashlib
import re
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import subprocess
import socket
import ssl
import urllib.parse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)

@dataclass
class SecurityThreat:
    """安全威胁"""
    threat_id: str
    threat_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    affected_component: str
    mitigation: str
    timestamp: float

class SecurityEnhancedHookV6:
    """
    安全增强Hook V6 - T-MIA凤凰架构的安全守护者
    提供代码安全检查、输入验证、权限控制和威胁检测
    """
    
    def __init__(self):
        self.hook_id = f"security_enhanced_v6_{int(time.time())}"
        
        # 安全规则库
        self.security_rules = self._load_security_rules()
        
        # 威胁检测器
        self.threat_detector = AdvancedThreatDetectorV6()
        
        # 输入验证器
        self.input_validator = InputValidatorV6()
        
        # 权限检查器
        self.permission_checker = PermissionCheckerV6()
        
        # 代码分析器
        self.code_analyzer = CodeSecurityAnalyzerV6()
        
        logger.info(f"🛡️ 安全增强Hook V6初始化完成 - Hook ID: {self.hook_id}")
    
    def _load_security_rules(self) -> Dict[str, Any]:
        """加载安全规则"""
        return {
            "sql_injection_patterns": [
                r"(\bselect\b.*\bfrom\b)|(\binsert\b.*\binto\b)|(\bupdate\b.*\bset\b)|(\bdelete\b.*\bfrom\b)",
                r"(\bor\b.*=.*['\"]\s*['\"]\s*$)|(\band\b.*=.*['\"]\s*['\"]\s*$)",
                r"(\bunion\b.*\bselect\b)|(\bdrop\b.*\btable\b)|(\btruncate\b.*\btable\b)",
                r"('.*--)|(--)|(\bselect\b.*\*)|(\binsert\b.*\*)|(\bupdate\b.*\*)"
            ],
            "xss_patterns": [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"onload\s*=",
                r"onerror\s*=",
                r"<iframe[^>]*>.*?</iframe>",
                r"<object[^>]*>.*?</object>",
                r"<embed[^>]*>.*?</embed>"
            ],
            "command_injection_patterns": [
                r"(\bexec\b)|(\beval\b)|(\bsystem\b)|(\bpopen\b)",
                r"(\|\|)|(&&)|(;)|(`)|(\$\()",
                r"\bcat\s+\S+|\bhead\s+\S+|\btail\s+\S+|\bwc\s+\S+",
                r"\bps\s+\S+|\bls\s+\S+|\bfind\s+\S+|\bgrep\s+\S+"
            ],
            "path_traversal_patterns": [
                r"\.\.\/|\.\.\\",
                r"\/etc\/passwd|\/etc\/shadow|\/etc\/hosts",
                r"\\windows\\|\\system32\\|\\winnt\\",
                r"\.\.\/\.\.\/|\.\.\\\.\.\\"
            ],
            "sensitive_data_patterns": [
                r"\bpassword\s*=\s*[^\s,;]+",
                r"\bapi[_-]?key\s*=\s*[^\s,;]+",
                r"\bsecret\s*=\s*[^\s,;]+",
                r"\btoken\s*=\s*[^\s,;]+",
                r"\bprivate[_-]?key\b",
                r"\bssn\b|\bsocial[_-]?security\b",
                r"\bcredit[_-]?card\b|\bcard[_-]?number\b"
            ],
            "malicious_code_patterns": [
                r"\bimport\s+os\s*;?\s*os\.system",
                r"\bimport\s+subprocess\s*;?\s*subprocess\.Popen",
                r"\beval\s*\(",
                r"\bexec\s*\(",
                r"__import__\(",
                r"compile\s*\(",
                r"getattr\s*\(",
                r"setattr\s*\(",
                r"delattr\s*\("
            ]
        }
    
    async def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook执行入口
        
        Args:
            context: 执行上下文
        
        Returns:
            Dict[str, Any]: 检查结果
        """
        start_time = time.time()
        
        results = {
            "hook_id": self.hook_id,
            "timestamp": start_time,
            "success": True,
            "checks": {},
            "threats": [],
            "recommendations": [],
            "execution_time": 0.0
        }
        
        try:
            # 1. 输入验证检查
            input_check = await self._check_input_validation(context)
            results["checks"]["input_validation"] = input_check
            
            # 2. 代码安全分析
            code_check = await self._check_code_security(context)
            results["checks"]["code_security"] = code_check
            
            # 3. 权限检查
            permission_check = await self._check_permissions(context)
            results["checks"]["permission_check"] = permission_check
            
            # 4. 威胁检测
            threat_check = await self._perform_threat_detection(context)
            results["checks"]["threat_detection"] = threat_check
            
            # 5. 网络安全检查
            network_check = await self._check_network_security(context)
            results["checks"]["network_security"] = network_check
            
            # 6. 依赖安全检查
            dependency_check = await self._check_dependency_security(context)
            results["checks"]["dependency_security"] = dependency_check
            
            # 7. 敏感信息检查
            sensitive_check = await self._check_sensitive_information(context)
            results["checks"]["sensitive_information"] = sensitive_check
            
            # 汇总结果
            all_checks = list(results["checks"].values())
            results["success"] = all(check.get("passed", False) for check in all_checks)
            
            # 收集威胁
            for check_name, check_result in results["checks"].items():
                if check_result.get("threats"):
                    results["threats"].extend(check_result["threats"])
            
            # 生成建议
            results["recommendations"] = self._generate_security_recommendations(results["checks"])
            
        except Exception as e:
            logger.error(f"安全检查执行失败: {e}")
            results["success"] = False
            results["error"] = str(e)
        
        results["execution_time"] = time.time() - start_time
        
        logger.info(f"🛡️ 安全检查完成: {'通过' if results['success'] else '未通过'} ({len(results['threats'])} 个威胁)")
        return results
    
    async def _check_input_validation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """检查输入验证"""
        check_result = {
            "check_name": "input_validation",
            "passed": True,
            "threats": [],
            "details": {}
        }
        
        # 检查输入数据
        input_data = context.get("input_data", {})
        
        for key, value in input_data.items():
            if isinstance(value, str):
                # 检查SQL注入
                for pattern in self.security_rules["sql_injection_patterns"]:
                    if re.search(pattern, value, re.IGNORECASE):
                        threat = SecurityThreat(
                            threat_id=f"sql_injection_{key}",
                            threat_type="SQL_INJECTION",
                            severity="HIGH",
                            description=f"检测到SQL注入模式: {key}",
                            affected_component=key,
                            mitigation="使用参数化查询或ORM",
                            timestamp=time.time()
                        )
                        check_result["threats"].append(threat.__dict__)
                        check_result["passed"] = False
                
                # 检查XSS
                for pattern in self.security_rules["xss_patterns"]:
                    if re.search(pattern, value, re.IGNORECASE):
                        threat = SecurityThreat(
                            threat_id=f"xss_{key}",
                            threat_type="XSS",
                            severity="MEDIUM",
                            description=f"检测到XSS模式: {key}",
                            affected_component=key,
                            mitigation="对输出进行HTML编码",
                            timestamp=time.time()
                        )
                        check_result["threats"].append(threat.__dict__)
                        check_result["passed"] = False
                
                # 检查命令注入
                for pattern in self.security_rules["command_injection_patterns"]:
                    if re.search(pattern, value, re.IGNORECASE):
                        threat = SecurityThreat(
                            threat_id=f"command_injection_{key}",
                            threat_type="COMMAND_INJECTION",
                            severity="CRITICAL",
                            description=f"检测到命令注入模式: {key}",
                            affected_component=key,
                            mitigation="验证和清理用户输入",
                            timestamp=time.time()
                        )
                        check_result["threats"].append(threat.__dict__)
                        check_result["passed"] = False
                
                # 检查路径遍历
                for pattern in self.security_rules["path_traversal_patterns"]:
                    if re.search(pattern, value, re.IGNORECASE):
                        threat = SecurityThreat(
                            threat_id=f"path_traversal_{key}",
                            threat_type="PATH_TRAVERSAL",
                            severity="HIGH",
                            description=f"检测到路径遍历模式: {key}",
                            affected_component=key,
                            mitigation="验证文件路径",
                            timestamp=time.time()
                        )
                        check_result["threats"].append(threat.__dict__)
                        check_result["passed"] = False
        
        check_result["details"]["checked_fields"] = len(input_data)
        check_result["details"]["threats_found"] = len(check_result["threats"])
        
        return check_result
    
    async def _check_code_security(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """检查代码安全"""
        check_result = {
            "check_name": "code_security",
            "passed": True,
            "threats": [],
            "details": {}
        }
        
        # 检查代码内容
        code_content = context.get("code", "") or context.get("content", "")
        
        if code_content:
            lines = code_content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                # 检查恶意代码模式
                for pattern in self.security_rules["malicious_code_patterns"]:
                    if re.search(pattern, line, re.IGNORECASE):
                        threat = SecurityThreat(
                            threat_id=f"malicious_code_line_{line_num}",
                            threat_type="MALICIOUS_CODE",
                            severity="CRITICAL",
                            description=f"检测到恶意代码模式: 第{line_num}行",
                            affected_component=f"line_{line_num}",
                            mitigation="移除或重写该代码行",
                            timestamp=time.time()
                        )
                        check_result["threats"].append(threat.__dict__)
                        check_result["passed"] = False
                
                # 检查硬编码密码
                if re.search(r'password\s*=\s*["\'][^"\']+["\']', line, re.IGNORECASE):
                    threat = SecurityThreat(
                        threat_id=f"hardcoded_password_line_{line_num}",
                        threat_type="HARDCODED_PASSWORD",
                        severity="HIGH",
                        description=f"检测到硬编码密码: 第{line_num}行",
                        affected_component=f"line_{line_num}",
                        mitigation="使用环境变量或配置文件",
                        timestamp=time.time()
                    )
                    check_result["threats"].append(threat.__dict__)
                    check_result["passed"] = False
        
        check_result["details"]["lines_checked"] = len(code_content.split('\n')) if code_content else 0
        check_result["details"]["threats_found"] = len(check_result["threats"])
        
        return check_result
    
    async def _check_permissions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """检查权限"""
        check_result = {
            "check_name": "permission_check",
            "passed": True,
            "threats": [],
            "details": {}
        }
        
        # 检查文件权限
        file_paths = context.get("file_paths", [])
        
        for file_path in file_paths:
            try:
                path = Path(file_path)
                if path.exists():
                    # 检查敏感文件权限
                    if path.is_file() and self._is_sensitive_file(path):
                        stat_info = path.stat()
                        permissions = oct(stat_info.st_mode)[-3:]
                        
                        # 检查是否过于宽松的权限
                        if int(permissions[-1]) & 4:  # 其他用户可读
                            threat = SecurityThreat(
                                threat_id=f"excessive_permissions_{file_path}",
                                threat_type="EXCESSIVE_PERMISSIONS",
                                severity="MEDIUM",
                                description=f"文件权限过于宽松: {file_path} ({permissions})",
                                affected_component=file_path,
                                mitigation="限制文件权限，移除其他用户读取权限",
                                timestamp=time.time()
                            )
                            check_result["threats"].append(threat.__dict__)
                            check_result["passed"] = False
            except Exception as e:
                logger.warning(f"权限检查错误: {file_path} - {e}")
        
        # 检查目录遍历
        for file_path in file_paths:
            path = Path(file_path)
            if ".." in str(path) or path.resolve().is_absolute():
                # 验证是否在允许的目录内
                allowed_paths = context.get("allowed_paths", [])
                if allowed_paths:
                    resolved_path = path.resolve()
                    if not any(str(resolved_path).startswith(allowed) for allowed in allowed_paths):
                        threat = SecurityThreat(
                            threat_id=f"directory_traversal_{file_path}",
                            threat_type="DIRECTORY_TRAVERSAL",
                            severity="HIGH",
                            description=f"检测到目录遍历: {file_path}",
                            affected_component=file_path,
                            mitigation="验证文件路径在允许范围内",
                            timestamp=time.time()
                        )
                        check_result["threats"].append(threat.__dict__)
                        check_result["passed"] = False
        
        check_result["details"]["files_checked"] = len(file_paths)
        check_result["details"]["threats_found"] = len(check_result["threats"])
        
        return check_result
    
    def _is_sensitive_file(self, path: Path) -> bool:
        """判断是否为敏感文件"""
        sensitive_extensions = ['.env', '.key', '.pem', '.p12', '.pfx', '.conf', '.config']
        sensitive_names = ['password', 'secret', 'key', 'token', 'auth']
        
        # 检查扩展名
        if path.suffix.lower() in sensitive_extensions:
            return True
        
        # 检查文件名
        name_lower = path.name.lower()
        if any(sensitive in name_lower for sensitive in sensitive_names):
            return True
        
        # 检查路径
        if 'secret' in str(path).lower() or 'password' in str(path).lower():
            return True
        
        return False
    
    async def _perform_threat_detection(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行威胁检测"""
        return await self.threat_detector.analyze_context(context)
    
    async def _check_network_security(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """检查网络安全"""
        check_result = {
            "check_name": "network_security",
            "passed": True,
            "threats": [],
            "details": {}
        }
        
        # 检查URL安全性
        urls = context.get("urls", [])
        
        for url in urls:
            try:
                parsed = urllib.parse.urlparse(url)
                
                # 检查不安全的协议
                if parsed.scheme not in ['https', 'ssh']:
                    threat = SecurityThreat(
                        threat_id=f"unsafe_protocol_{url}",
                        threat_type="UNSAFE_PROTOCOL",
                        severity="MEDIUM",
                        description=f"使用不安全的协议: {url}",
                        affected_component=url,
                        mitigation="使用HTTPS或SSH协议",
                        timestamp=time.time()
                    )
                    check_result["threats"].append(threat.__dict__)
                    check_result["passed"] = False
                
                # 检查可疑域名
                if self._is_suspicious_domain(parsed.hostname):
                    threat = SecurityThreat(
                        threat_id=f"suspicious_domain_{url}",
                        threat_type="SUSPICIOUS_DOMAIN",
                        severity="HIGH",
                        description=f"可疑域名: {url}",
                        affected_component=url,
                        mitigation="验证域名的合法性",
                        timestamp=time.time()
                    )
                    check_result["threats"].append(threat.__dict__)
                    check_result["passed"] = False
                    
            except Exception as e:
                logger.warning(f"URL检查错误: {url} - {e}")
        
        check_result["details"]["urls_checked"] = len(urls)
        check_result["details"]["threats_found"] = len(check_result["threats"])
        
        return check_result
    
    def _is_suspicious_domain(self, hostname: str) -> bool:
        """判断是否为可疑域名"""
        if not hostname:
            return False
        
        # 检查IP地址
        try:
            socket.inet_aton(hostname)
            return True  # IP地址通常不太安全
        except socket.error:
            pass
        
        # 检查可疑的TLD
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf']
        if any(hostname.endswith(tld) for tld in suspicious_tlds):
            return True
        
        # 检查随机字符串
        if len(hostname) > 20 and not any(c in hostname.lower() for c in 'aeiou'):
            return True
        
        return False
    
    async def _check_dependency_security(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """检查依赖安全"""
        check_result = {
            "check_name": "dependency_security",
            "passed": True,
            "threats": [],
            "details": {}
        }
        
        # 检查依赖列表
        dependencies = context.get("dependencies", [])
        
        for dep in dependencies:
            # 检查已知的漏洞包
            if self._is_vulnerable_package(dep):
                threat = SecurityThreat(
                    threat_id=f"vulnerable_dependency_{dep}",
                    threat_type="VULNERABLE_DEPENDENCY",
                    severity="HIGH",
                    description=f"发现已知漏洞依赖: {dep}",
                    affected_component=dep,
                    mitigation="升级到安全版本",
                    timestamp=time.time()
                )
                check_result["threats"].append(threat.__dict__)
                check_result["passed"] = False
        
        check_result["details"]["dependencies_checked"] = len(dependencies)
        check_result["details"]["threats_found"] = len(check_result["threats"])
        
        return check_result
    
    def _is_vulnerable_package(self, package: str) -> bool:
        """判断是否为已知漏洞包"""
        # 这里应该连接到实际的漏洞数据库
        # 简化实现：检查一些已知的危险包
        vulnerable_packages = [
            'requests==2.28.0',  # 示例
            'django<4.0',
            'flask<2.0',
            'numpy==1.21.0'
        ]
        
        return any(package.startswith(vuln.split('<')[0].split('==')[0]) and 
                  (('<' in vuln and package < vuln) or ('==' in vuln and package == vuln))
                  for vuln in vulnerable_packages)
    
    async def _check_sensitive_information(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """检查敏感信息"""
        check_result = {
            "check_name": "sensitive_information",
            "passed": True,
            "threats": [],
            "details": {}
        }
        
        # 检查敏感数据模式
        content = json.dumps(context, default=str)
        
        for pattern in self.security_rules["sensitive_data_patterns"]:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                for match in matches:
                    threat = SecurityThreat(
                        threat_id=f"sensitive_data_{hashlib.md5(match.encode()).hexdigest()[:8]}",
                        threat_type="SENSITIVE_DATA",
                        severity="HIGH",
                        description=f"检测到敏感数据: {match[:20]}...",
                        affected_component="context",
                        mitigation="移除或加密敏感信息",
                        timestamp=time.time()
                    )
                    check_result["threats"].append(threat.__dict__)
                    check_result["passed"] = False
        
        check_result["details"]["content_size"] = len(content)
        check_result["details"]["threats_found"] = len(check_result["threats"])
        
        return check_result
    
    def _generate_security_recommendations(self, checks: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """生成安全建议"""
        recommendations = []
        
        # 基于检查结果生成建议
        for check_name, check_result in checks.items():
            if not check_result.get("passed", True):
                threats = check_result.get("threats", [])
                
                for threat in threats:
                    severity = threat.get("severity", "MEDIUM")
                    if severity in ["HIGH", "CRITICAL"]:
                        recommendations.append({
                            "priority": severity,
                            "category": threat.get("threat_type", "GENERAL"),
                            "recommendation": f"立即处理: {threat.get('description', '安全威胁')}",
                            "action": threat.get("mitigation", "请参考安全文档")
                        })
        
        # 通用建议
        if not recommendations:
            recommendations.extend([
                {
                    "priority": "LOW",
                    "category": "GENERAL",
                    "recommendation": "定期更新依赖包",
                    "action": "使用工具检查依赖漏洞"
                },
                {
                    "priority": "LOW",
                    "category": "GENERAL",
                    "recommendation": "实施最小权限原则",
                    "action": "审查和限制系统权限"
                }
            ])
        
        return recommendations

# --- 高级威胁检测器 ---
class AdvancedThreatDetectorV6:
    """高级威胁检测器V6"""
    
    def __init__(self):
        self.threat_patterns = self._load_threat_patterns()
    
    def _load_threat_patterns(self) -> Dict[str, List[str]]:
        """加载威胁模式"""
        return {
            "anomaly_patterns": [
                r"unusual\s+access\s+pattern",
                r"multiple\s+failed\s+attempts",
                r"unauthorized\s+privilege\s+escalation",
                r"suspicious\s+network\s+activity"
            ],
            "malware_patterns": [
                r"ransomware\s+signature",
                r"trojan\s+horse",
                r"rootkit\s+detection",
                r"keylogger\s+activity"
            ],
            "social_engineering": [
                r"phishing\s+attempt",
                r"social\s+engineering",
                r"impersonation\s+attack",
                r"credential\s+harvesting"
            ]
        }
    
    async def analyze_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析上下文"""
        result = {
            "check_name": "advanced_threat_detection",
            "passed": True,
            "threats": [],
            "details": {}
        }
        
        # 简化实现：基于上下文特征检测威胁
        context_str = json.dumps(context, default=str).lower()
        
        threat_count = 0
        for category, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if re.search(pattern, context_str):
                    threat_count += 1
        
        if threat_count > 0:
            result["passed"] = False
            result["details"]["anomaly_score"] = min(1.0, threat_count / 10)
        else:
            result["details"]["anomaly_score"] = 0.0
        
        result["details"]["patterns_checked"] = sum(len(patterns) for patterns in self.threat_patterns.values())
        
        return result

# --- 输入验证器 ---
class InputValidatorV6:
    """输入验证器V6"""
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """加载验证规则"""
        return {
            "max_length": 10000,
            "allowed_characters": r"^[a-zA-Z0-9\s\.\,\!\?\-\_\(\)\[\]\{\}]+",
            "forbidden_words": ["admin", "root", "test", "guest"],
            "sql_keywords": ["select", "insert", "update", "delete", "drop", "create", "alter", "grant", "revoke"]
        }
    
    async def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """验证输入"""
        result = {
            "validation_passed": True,
            "errors": [],
            "sanitized_data": {}
        }
        
        for key, value in input_data.items():
            if isinstance(value, str):
                # 长度检查
                if len(value) > self.validation_rules["max_length"]:
                    result["errors"].append(f"{key}: 输入过长")
                    result["validation_passed"] = False
                
                # 字符检查
                if not re.match(self.validation_rules["allowed_characters"], value):
                    result["errors"].append(f"{key}: 包含非法字符")
                    result["validation_passed"] = False
                
                # 敏感词检查
                for forbidden in self.validation_rules["forbidden_words"]:
                    if forbidden in value.lower():
                        result["errors"].append(f"{key}: 包含敏感词")
                        result["validation_passed"] = False
                
                # SQL关键字检查
                for keyword in self.validation_rules["sql_keywords"]:
                    if keyword in value.lower():
                        result["errors"].append(f"{key}: 包含SQL关键字")
                        result["validation_passed"] = False
                
                # 清理输入
                sanitized = self._sanitize_input(value)
                result["sanitized_data"][key] = sanitized
        
        return result
    
    def _sanitize_input(self, input_str: str) -> str:
        """清理输入"""
        # 移除或转义危险字符
        dangerous_chars = {
            "'": "'",
            '"': """,
            "<": "<",
            ">": ">",
            "&": "&",
            "(": "&#40;",
            ")": "&#41;"
        }
        
        for char, replacement in dangerous_chars.items():
            input_str = input_str.replace(char, replacement)
        
        return input_str

# --- 权限检查器 ---
class PermissionCheckerV6:
    """权限检查器V6"""
    
    def __init__(self):
        self.permission_matrix = self._load_permission_matrix()
    
    def _load_permission_matrix(self) -> Dict[str, List[str]]:
        """加载权限矩阵"""
        return {
            "read": ["file", "database", "api"],
            "write": ["file", "database"],
            "execute": ["script", "command"],
            "admin": ["system", "user", "config"]
        }
    
    async def check_permissions(self, user_permissions: List[str], required_permissions: List[str]) -> bool:
        """检查权限"""
        # 简化实现：检查用户权限是否包含所需权限
        user_set = set(user_permissions)
        required_set = set(required_permissions)
        
        return required_set.issubset(user_set)

# --- 代码安全分析器 ---
class CodeSecurityAnalyzerV6:
    """代码安全分析器V6"""
    
    def __init__(self):
        self.security_patterns = self._load_security_patterns()
    
    def _load_security_patterns(self) -> Dict[str, List[str]]:
        """加载安全模式"""
        return {
            "insecure_functions": [
                "eval(", "exec(", "compile(", "__import__(",
                "input(", "raw_input(", "file(",
                "open(", "popen(", "system(",
                "os.system(", "subprocess.call("
            ],
            "weak_crypto": [
                "md5(", "sha1(", "des_encrypt(",
                "rc4_encrypt(", "base64_encode(",
                "weak_random("
            ],
            "insecure_transmission": [
                "http://", "ftp://", "telnet://",
                "unencrypted_connection",
                "plaintext_password"
            ]
        }
    
    async def analyze_code(self, code_content: str) -> Dict[str, Any]:
        """分析代码"""
        result = {
            "security_score": 0.0,
            "issues": [],
            "recommendations": []
        }
        
        if not code_content:
            return result
        
        issues = []
        total_lines = len(code_content.split('\n'))
        
        for category, patterns in self.security_patterns.items():
            for pattern in patterns:
                if pattern in code_content:
                    issues.append({
                        "type": category,
                        "pattern": pattern,
                        "severity": self._get_severity(category)
                    })
        
        # 计算安全分数
        max_issues = total_lines * 0.1  # 假设每10行代码最多1个问题
        security_score = max(0.0, 1.0 - (len(issues) / max_issues))
        
        result["security_score"] = security_score
        result["issues"] = issues
        result["recommendations"] = self._generate_recommendations(issues)
        
        return result
    
    def _get_severity(self, category: str) -> str:
        """获取严重程度"""
        severity_map = {
            "insecure_functions": "CRITICAL",
            "weak_crypto": "HIGH",
            "insecure_transmission": "MEDIUM"
        }
        return severity_map.get(category, "LOW")
    
    def _generate_recommendations(self, issues: List[Dict]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        for issue in issues:
            if issue["type"] == "insecure_functions":
                recommendations.append("使用安全的替代函数，避免动态代码执行")
            elif issue["type"] == "weak_crypto":
                recommendations.append("使用强加密算法，如AES-256")
            elif issue["type"] == "insecure_transmission":
                recommendations.append("使用HTTPS等加密传输协议")
        
        return list(set(recommendations))  # 去重

# --- 测试函数 ---
async def test_security_hook():
    """测试安全增强Hook"""
    print("🧪 测试安全增强Hook V6")
    print("=" * 50)
    
    hook = SecurityEnhancedHookV6()
    
    # 测试用例
    test_cases = [
        {
            "name": "正常输入",
            "context": {
                "input_data": {"username": "testuser", "action": "read"},
                "code": "print('Hello World')",
                "file_paths": ["./test.txt"]
            }
        },
        {
            "name": "SQL注入尝试",
            "context": {
                "input_data": {"username": "admin' OR '1'='1", "action": "login"},
                "code": "user_input = request.GET['param']"
            }
        },
        {
            "name": "XSS尝试",
            "context": {
                "input_data": {"comment": "<script>alert('xss')</script>"},
                "code": "print(user_input)"
            }
        },
        {
            "name": "恶意代码",
            "context": {
                "code": "import os\nos.system('rm -rf /')",
                "file_paths": ["../etc/passwd"]
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔒 测试案例 {i}: {test_case['name']}")
        
        result = await hook(test_case['context'])
        
        print(f"✅ 检查结果: {'通过' if result['success'] else '未通过'}")
        print(f"📊 威胁数量: {len(result['threats'])}")
        print(f"⏱️ 执行时间: {result['execution_time']:.3f}s")
        
        if result['threats']:
            print("🚨 发现威胁:")
            for threat in result['threats'][:3]:  # 显示前3个
                print(f"  - {threat['threat_type']}: {threat['description']}")
        
        if result['recommendations']:
            print("💡 安全建议:")
            for rec in result['recommendations'][:2]:  # 显示前2个
                print(f"  - {rec['recommendation']}")
    
    print(f"\n✅ 安全增强Hook V6测试完成")

if __name__ == "__main__":
    asyncio.run(test_security_hook())