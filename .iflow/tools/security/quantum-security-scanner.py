#!/usr/bin/env python3
"""
量子安全扫描器
基于量子算法的后量子密码学安全扫描工具
"""

import ast
import re
import json
import hashlib
import os
import sys
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import subprocess
import tempfile

class SecurityLevel(Enum):
    """安全级别枚举"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class QuantumSecurityStandard(Enum):
    """量子安全标准枚举"""
    NIST_PQC = "nist_pqc"
    ISO_27001 = "iso_27001"
    OWASP_TOP10 = "owasp_top10"
    CUSTOM = "custom"

@dataclass
class SecurityVulnerability:
    """安全漏洞数据类"""
    id: str
    title: str
    description: str
    severity: SecurityLevel
    cwe_id: Optional[str]
    file_path: str
    line_number: int
    quantum_risk_score: float
    post_quantum_impact: str
    recommendation: str
    references: List[str]

class QuantumSecurityScanner:
    """量子安全扫描器"""
    
    def __init__(self):
        self.vulnerabilities = []
        self.quantum_algorithms = {}
        self.post_quantum_crypto = {
            'CRYSTALS-Kyber': 'key_encapsulation',
            'CRYSTALS-Dilithium': 'digital_signature',
            'FALCON': 'digital_signature',
            'NTRU': 'encryption',
            'SPHINCS+': 'digital_signature'
        }
        
    def scan_file(self, file_path: str) -> Dict[str, Any]:
        """扫描单个文件的安全漏洞"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            scan_result = {
                'file_path': file_path,
                'scan_timestamp': self._get_timestamp(),
                'vulnerabilities': self._scan_vulnerabilities(content, tree, file_path),
                'quantum_security_assessment': self._quantum_security_assessment(content),
                'post_quantum_readiness': self._assess_post_quantum_readiness(content),
                'security_recommendations': self._generate_security_recommendations(),
                'compliance_status': self._check_compliance_standards(content)
            }
            
            return scan_result
            
        except Exception as e:
            return {
                'file_path': file_path,
                'error': str(e),
                'scan_timestamp': self._get_timestamp(),
                'vulnerabilities': [],
                'quantum_security_assessment': {},
                'post_quantum_readiness': {},
                'security_recommendations': [],
                'compliance_status': {}
            }
    
    def _scan_vulnerabilities(self, content: str, tree: ast.AST, file_path: str) -> List[SecurityVulnerability]:
        """扫描安全漏洞"""
        vulnerabilities = []
        
        # SQL注入漏洞
        vulnerabilities.extend(self._scan_sql_injection(content, file_path))
        
        # 硬编码密码
        vulnerabilities.extend(self._scan_hardcoded_credentials(content, file_path))
        
        # XSS漏洞
        vulnerabilities.extend(self._scan_xss_vulnerabilities(content, file_path))
        
        # 路径遍历
        vulnerabilities.extend(self._scan_path_traversal(content, file_path))
        
        # 不安全的反序列化
        vulnerabilities.extend(self._scan_unsafe_deserialization(content, file_path))
        
        # 弱加密算法
        vulnerabilities.extend(self._scan_weak_cryptography(content, file_path))
        
        # 不安全的随机数生成
        vulnerabilities.extend(self._scan_weak_randomness(content, file_path))
        
        # 命令注入
        vulnerabilities.extend(self._scan_command_injection(content, file_path))
        
        # 不安全的文件操作
        vulnerabilities.extend(self._scan_unsafe_file_operations(content, file_path))
        
        # 量子安全相关漏洞
        vulnerabilities.extend(self._scan_quantum_security_issues(content, file_path))
        
        return vulnerabilities
    
    def _scan_sql_injection(self, content: str, file_path: str) -> List[SecurityVulnerability]:
        """扫描SQL注入漏洞"""
        vulnerabilities = []
        
        # SQL注入模式
        sql_patterns = [
            r'(execute|exec|query).*\+.*\w+',
            r'(execute|exec|query).*%.*\w+',
            r'f".*{.*}.*\b(SELECT|INSERT|UPDATE|DELETE)\b',
            r'SELECT.*FROM.*\+.*\w+',
            r'cursor\.execute.*\+',
            r'cursor\.execute.*%.*\w+'
        ]
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern in sql_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerability = SecurityVulnerability(
                        id=f"SQL_INJECTION_{line_num}",
                        title="SQL注入漏洞",
                        description="检测到潜在的SQL注入漏洞，可能允许攻击者执行恶意SQL命令",
                        severity=SecurityLevel.CRITICAL,
                        cwe_id="CWE-89",
                        file_path=file_path,
                        line_number=line_num,
                        quantum_risk_score=0.95,
                        post_quantum_impact="量子计算可能加速SQL注入攻击的自动化",
                        recommendation="使用参数化查询或ORM框架，避免字符串拼接SQL语句",
                        references=[
                            "https://owasp.org/www-community/attacks/SQL_Injection",
                            "https://cwe.mitre.org/data/definitions/89.html"
                        ]
                    )
                    vulnerabilities.append(vulnerability)
                    
        return vulnerabilities
    
    def _scan_hardcoded_credentials(self, content: str, file_path: str) -> List[SecurityVulnerability]:
        """扫描硬编码凭据"""
        vulnerabilities = []
        
        # 硬编码凭据模式
        credential_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'passwd\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'key\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'private_key\s*=\s*["\'][^"\']+["\']'
        ]
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern in credential_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerability = SecurityVulnerability(
                        id=f"HARDCODED_CREDENTIALS_{line_num}",
                        title="硬编码凭据",
                        description="检测到硬编码的密码、密钥或其他敏感信息",
                        severity=SecurityLevel.HIGH,
                        cwe_id="CWE-798",
                        file_path=file_path,
                        line_number=line_num,
                        quantum_risk_score=0.90,
                        post_quantum_impact="量子计算机可能破解传统加密，暴露的凭据风险更高",
                        recommendation="使用环境变量、密钥管理系统或量子密钥分发(QKD)",
                        references=[
                            "https://owasp.org/www-project-top-ten/2017/A2_2017-Broken_Authentication",
                            "https://cwe.mitre.org/data/definitions/798.html"
                        ]
                    )
                    vulnerabilities.append(vulnerability)
                    
        return vulnerabilities
    
    def _scan_xss_vulnerabilities(self, content: str, file_path: str) -> List[SecurityVulnerability]:
        """扫描XSS漏洞"""
        vulnerabilities = []
        
        # XSS模式
        xss_patterns = [
            r'innerHTML\s*=',
            r'outerHTML\s*=',
            r'document\.write\s*\(',
            r'eval\s*\(',
            r'function\s*\(\s*\)\s*{.*return.*\+.*}',
            r'<script[^>]*>.*</script>'
        ]
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern in xss_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerability = SecurityVulnerability(
                        id=f"XSS_VULNERABILITY_{line_num}",
                        title="跨站脚本攻击(XSS)漏洞",
                        description="检测到潜在的XSS漏洞，可能允许攻击者注入恶意脚本",
                        severity=SecurityLevel.HIGH,
                        cwe_id="CWE-79",
                        file_path=file_path,
                        line_number=line_num,
                        quantum_risk_score=0.85,
                        post_quantum_impact="量子计算可能加速XSS载荷的生成和优化",
                        recommendation="使用安全的DOM操作方法，输入验证和输出编码",
                        references=[
                            "https://owasp.org/www-project-top-ten/2017/A7_2017-Cross-Site_Scripting_(XSS)",
                            "https://cwe.mitre.org/data/definitions/79.html"
                        ]
                    )
                    vulnerabilities.append(vulnerability)
                    
        return vulnerabilities
    
    def _scan_path_traversal(self, content: str, file_path: str) -> List[SecurityVulnerability]:
        """扫描路径遍历漏洞"""
        vulnerabilities = []
        
        # 路径遍历模式
        path_traversal_patterns = [
            r'\.\./|\.\.\\',
            r'open\s*\([^)]*\.\.',
            r'read\s*\([^)]*\.\.',
            r'file\s*\([^)]*\.\.',
            r'os\.path\.join.*\.\.',
            r'pathlib\.Path.*\.\.'
        ]
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern in path_traversal_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerability = SecurityVulnerability(
                        id=f"PATH_TRAVERSAL_{line_num}",
                        title="路径遍历漏洞",
                        description="检测到潜在的路径遍历漏洞，可能允许访问系统敏感文件",
                        severity=SecurityLevel.HIGH,
                        cwe_id="CWE-22",
                        file_path=file_path,
                        line_number=line_num,
                        quantum_risk_score=0.80,
                        post_quantum_impact="量子算法可能加速路径遍历攻击的自动化",
                        recommendation="验证和规范化文件路径，使用白名单机制",
                        references=[
                            "https://owasp.org/www-project-top-ten/2017/A5_2017-Broken_Access_Control",
                            "https://cwe.mitre.org/data/definitions/22.html"
                        ]
                    )
                    vulnerabilities.append(vulnerability)
                    
        return vulnerabilities
    
    def _scan_unsafe_deserialization(self, content: str, file_path: str) -> List[SecurityVulnerability]:
        """扫描不安全的反序列化"""
        vulnerabilities = []
        
        # 不安全反序列化模式
        unsafe_deserialization_patterns = [
            r'pickle\.loads?\s*\(',
            r'cPickle\.loads?\s*\(',
            r'marshal\.loads?\s*\(',
            r'yaml\.loads?\s*\(',
            r'json\.loads?\s*\(',
            r'eval\s*\(',
            r'exec\s*\('
        ]
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern in unsafe_deserialization_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerability = SecurityVulnerability(
                        id=f"UNSAFE_DESERIALIZATION_{line_num}",
                        title="不安全的反序列化",
                        description="检测到不安全的反序列化操作，可能导致远程代码执行",
                        severity=SecurityLevel.HIGH,
                        cwe_id="CWE-502",
                        file_path=file_path,
                        line_number=line_num,
                        quantum_risk_score=0.88,
                        post_quantum_impact="量子计算可能加速反序列化漏洞的利用",
                        recommendation="使用安全的序列化格式，验证输入数据",
                        references=[
                            "https://owasp.org/www-project-top-ten/2017/A8_2017-Insecure_Deserialization",
                            "https://cwe.mitre.org/data/definitions/502.html"
                        ]
                    )
                    vulnerabilities.append(vulnerability)
                    
        return vulnerabilities
    
    def _scan_weak_cryptography(self, content: str, file_path: str) -> List[SecurityVulnerability]:
        """扫描弱加密算法"""
        vulnerabilities = []
        
        # 弱加密算法模式
        weak_crypto_patterns = {
            r'md5': 'MD5哈希算法已被破解',
            r'sha1': 'SHA1哈希算法已被破解',
            r'des': 'DES加密算法密钥长度不足',
            r'rc4': 'RC4流密码算法存在漏洞',
            r'blowfish': 'Blowfish加密算法已过时'
        }
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern, description in weak_crypto_patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerability = SecurityVulnerability(
                        id=f"WEAK_CRYPTOGRAPHY_{line_num}",
                        title="弱加密算法",
                        description=f"检测到弱加密算法: {description}",
                        severity=SecurityLevel.MEDIUM,
                        cwe_id="CWE-327",
                        file_path=file_path,
                        line_number=line_num,
                        quantum_risk_score=0.95,
                        post_quantum_impact="量子计算机可以轻易破解这些弱加密算法",
                        recommendation="使用强加密算法，如AES-256、SHA-256或后量子密码学算法",
                        references=[
                            "https://owasp.org/www-project-top-ten/2017/A3_2017-Sensitive_Data_Exposure",
                            "https://cwe.mitre.org/data/definitions/327.html"
                        ]
                    )
                    vulnerabilities.append(vulnerability)
                    
        return vulnerabilities
    
    def _scan_weak_randomness(self, content: str, file_path: str) -> List[SecurityVulnerability]:
        """扫描弱随机数生成"""
        vulnerabilities = []
        
        # 弱随机数生成模式
        weak_random_patterns = [
            r'random\.random\s*\(',
            r'math\.random\s*\(',
            r'random\.randint\s*\(',
            r'time\.time\s*\(',
            r'os\.urandom\s*\('
        ]
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern in weak_random_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerability = SecurityVulnerability(
                        id=f"WEAK_RANDOMNESS_{line_num}",
                        title="弱随机数生成",
                        description="检测到弱随机数生成器，可能被预测",
                        severity=SecurityLevel.MEDIUM,
                        cwe_id="CWE-338",
                        file_path=file_path,
                        line_number=line_num,
                        quantum_risk_score=0.70,
                        post_quantum_impact="量子随机数生成器提供真正的不可预测性",
                        recommendation="使用密码学安全的随机数生成器，如secrets模块或量子随机数生成器",
                        references=[
                            "https://owasp.org/www-project-cheatsheets/cheatsheets/Cryptographic_Storage_Cheat_Sheet.html",
                            "https://cwe.mitre.org/data/definitions/338.html"
                        ]
                    )
                    vulnerabilities.append(vulnerability)
                    
        return vulnerabilities
    
    def _scan_command_injection(self, content: str, file_path: str) -> List[SecurityVulnerability]:
        """扫描命令注入"""
        vulnerabilities = []
        
        # 命令注入模式
        command_injection_patterns = [
            r'system\s*\(',
            r'subprocess\.call\s*\(',
            r'subprocess\.run\s*\(',
            r'os\.system\s*\(',
            r'os\.popen\s*\(',
            r'eval\s*\(',
            r'exec\s*\('
        ]
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern in command_injection_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerability = SecurityVulnerability(
                        id=f"COMMAND_INJECTION_{line_num}",
                        title="命令注入",
                        description="检测到潜在的命令注入漏洞，可能允许执行任意系统命令",
                        severity=SecurityLevel.CRITICAL,
                        cwe_id="CWE-78",
                        file_path=file_path,
                        line_number=line_num,
                        quantum_risk_score=0.92,
                        post_quantum_impact="量子计算可能加速命令注入攻击的自动化",
                        recommendation="使用安全的API，避免直接执行用户输入",
                        references=[
                            "https://owasp.org/www-project-top-ten/2017/A1_2017-Injection",
                            "https://cwe.mitre.org/data/definitions/78.html"
                        ]
                    )
                    vulnerabilities.append(vulnerability)
                    
        return vulnerabilities
    
    def _scan_unsafe_file_operations(self, content: str, file_path: str) -> List[SecurityVulnerability]:
        """扫描不安全的文件操作"""
        vulnerabilities = []
        
        # 不安全文件操作模式
        unsafe_file_patterns = [
            r'open\s*\([^)]*\.\.',
            r'open\s*\([^)]*user',
            r'open\s*\([^)]*input',
            r'file\s*\([^)]*\.\.',
            r'with\s+open\s*\([^)]*\.\.'
        ]
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern in unsafe_file_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerability = SecurityVulnerability(
                        id=f"UNSAFE_FILE_OPERATION_{line_num}",
                        title="不安全的文件操作",
                        description="检测到不安全的文件操作，可能导致文件系统攻击",
                        severity=SecurityLevel.MEDIUM,
                        cwe_id="CWE-22",
                        file_path=file_path,
                        line_number=line_num,
                        quantum_risk_score=0.75,
                        post_quantum_impact="量子算法可能加速文件系统攻击的自动化",
                        recommendation="验证文件路径，使用安全的文件操作API",
                        references=[
                            "https://owasp.org/www-project-top-ten/2017/A5_2017-Broken_Access_Control",
                            "https://cwe.mitre.org/data/definitions/22.html"
                        ]
                    )
                    vulnerabilities.append(vulnerability)
                    
        return vulnerabilities
    
    def _scan_quantum_security_issues(self, content: str, file_path: str) -> List[SecurityVulnerability]:
        """扫描量子安全相关漏洞"""
        vulnerabilities = []
        
        # 量子安全模式
        quantum_security_patterns = [
            (r'quantum.*key.*hardcode', "量子密钥硬编码"),
            (r'quantum.*algorithm.*insecure', "不安全的量子算法实现"),
            (r'quantum.*measurement.*untrusted', "不可信的量子测量"),
            (r'quantum.*entanglement.*exposed', "量子纠缠暴露"),
            (r'quantum.*superposition.*collapse', "量子叠加态非预期坍塌")
        ]
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern, description in quantum_security_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerability = SecurityVulnerability(
                        id=f"QUANTUM_SECURITY_{line_num}",
                        title=f"量子安全问题: {description}",
                        description=f"检测到量子安全问题: {description}",
                        severity=SecurityLevel.HIGH,
                        cwe_id="CWE-200",
                        file_path=file_path,
                        line_number=line_num,
                        quantum_risk_score=0.95,
                        post_quantum_impact="直接影响量子系统的安全性",
                        recommendation="遵循量子计算安全最佳实践，使用量子安全协议",
                        references=[
                            "https://nist.gov/quantum",
                            "https://arxiv.org/abs/quant-ph"
                        ]
                    )
                    vulnerabilities.append(vulnerability)
                    
        return vulnerabilities
    
    def _quantum_security_assessment(self, content: str) -> Dict[str, Any]:
        """量子安全评估"""
        return {
            'quantum_readiness_score': self._calculate_quantum_readiness(content),
            'post_quantum_migration_complexity': self._estimate_migration_complexity(content),
            'quantum_algorithm_usage': self._analyze_quantum_algorithm_usage(content),
            'quantum_key_management': self._assess_quantum_key_management(content),
            'quantum_resistance_level': self._evaluate_quantum_resistance(content)
        }
    
    def _assess_post_quantum_readiness(self, content: str) -> Dict[str, Any]:
        """评估后量子密码学准备度"""
        readiness_score = 0.0
        recommendations = []
        
        # 检查是否使用了后量子密码学算法
        for algo in self.post_quantum_crypto:
            if algo.lower() in content.lower():
                readiness_score += 0.2
                recommendations.append(f"已使用后量子算法: {algo}")
        
        # 检查密钥长度
        if re.search(r'key.*size.*\b(256|512|1024)\b', content, re.IGNORECASE):
            readiness_score += 0.1
            
        # 检查加密库
        crypto_libraries = ['cryptography', 'pycryptodome', 'qiskit', 'cirq']
        for lib in crypto_libraries:
            if lib in content.lower():
                readiness_score += 0.05
                
        return {
            'readiness_score': min(1.0, readiness_score),
            'recommendations': recommendations,
            'migration_priority': 'high' if readiness_score < 0.5 else 'medium'
        }
    
    def _generate_security_recommendations(self) -> List[str]:
        """生成安全建议"""
        recommendations = [
            "实施零信任安全架构",
            "采用后量子密码学算法",
            "使用量子密钥分发(QKD)保护敏感通信",
            "实施量子安全的多因素认证",
            "定期进行量子安全风险评估",
            "建立量子安全事件响应计划",
            "使用量子随机数生成器增强安全性",
            "实施量子安全的区块链技术"
        ]
        
        return recommendations
    
    def _check_compliance_standards(self, content: str) -> Dict[str, Any]:
        """检查合规标准"""
        compliance_status = {}
        
        # NIST后量子密码学标准
        compliance_status['nist_pqc'] = self._check_nist_pqc_compliance(content)
        
        # ISO 27001标准
        compliance_status['iso_27001'] = self._check_iso_27001_compliance(content)
        
        # OWASP Top 10
        compliance_status['owasp_top10'] = self._check_owasp_compliance(content)
        
        return compliance_status
    
    # 辅助方法
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _calculate_quantum_readiness(self, content: str) -> float:
        """计算量子准备度分数"""
        score = 0.0
        
        # 检查量子相关关键词
        quantum_keywords = ['quantum', 'qubit', 'entanglement', 'superposition', 'measurement']
        for keyword in quantum_keywords:
            if keyword in content.lower():
                score += 0.1
                
        # 检查量子库
        quantum_libs = ['qiskit', 'cirq', 'pennylane', 'braket']
        for lib in quantum_libs:
            if lib in content.lower():
                score += 0.2
                
        return min(1.0, score)
    
    def _estimate_migration_complexity(self, content: str) -> str:
        """估算迁移复杂度"""
        complexity_indicators = 0
        
        # 检查加密相关代码
        if re.search(r'(encrypt|decrypt|hash|cipher)', content, re.IGNORECASE):
            complexity_indicators += 1
            
        # 检查密钥管理
        if re.search(r'(key|secret|password|token)', content, re.IGNORECASE):
            complexity_indicators += 1
            
        # 检查协议实现
        if re.search(r'(ssl|tls|https|ssh)', content, re.IGNORECASE):
            complexity_indicators += 1
            
        if complexity_indicators >= 3:
            return "high"
        elif complexity_indicators >= 2:
            return "medium"
        else:
            return "low"
    
    def _analyze_quantum_algorithm_usage(self, content: str) -> Dict[str, Any]:
        """分析量子算法使用情况"""
        quantum_algorithms = {
            'grover': False,
            'shor': False,
            'quantum_fourier': False,
            'variational': False,
            'quantum_annealing': False
        }
        
        for algo in quantum_algorithms:
            if algo.lower() in content.lower():
                quantum_algorithms[algo] = True
                
        return {
            'algorithms_used': [k for k, v in quantum_algorithms.items() if v],
            'count': sum(quantum_algorithms.values())
        }
    
    def _assess_quantum_key_management(self, content: str) -> Dict[str, Any]:
        """评估量子密钥管理"""
        return {
            'qkd_implemented': 'quantum key distribution' in content.lower(),
            'key_rotation': 'key rotation' in content.lower(),
            'key_escrow': 'key escrow' in content.lower(),
            'forward_secrecy': 'forward secrecy' in content.lower()
        }
    
    def _evaluate_quantum_resistance(self, content: str) -> str:
        """评估量子抗性"""
        if 'post-quantum' in content.lower() or 'post_quantum' in content.lower():
            return "high"
        elif 'quantum' in content.lower():
            return "medium"
        else:
            return "low"
    
    def _check_nist_pqc_compliance(self, content: str) -> Dict[str, Any]:
        """检查NIST后量子密码学合规性"""
        compliance = {
            'compliant': False,
            'algorithms': [],
            'score': 0.0
        }
        
        for algo in self.post_quantum_crypto:
            if algo.lower() in content.lower():
                compliance['algorithms'].append(algo)
                compliance['score'] += 0.2
                
        compliance['compliant'] = compliance['score'] >= 0.6
        
        return compliance
    
    def _check_iso_27001_compliance(self, content: str) -> Dict[str, Any]:
        """检查ISO 27001合规性"""
        return {
            'compliant': 'information security' in content.lower(),
            'score': 0.5 if 'information security' in content.lower() else 0.0
        }
    
    def _check_owasp_compliance(self, content: str) -> Dict[str, Any]:
        """检查OWASP Top 10合规性"""
        owasp_controls = [
            'injection',
            'broken_authentication',
            'sensitive_data',
            'xml_external_entities',
            'broken_access',
            'security_misconfiguration',
            'xss',
            'insecure_deserialization',
            'vulnerable_components',
            'logging_monitoring'
        ]
        
        implemented_controls = []
        for control in owasp_controls:
            if control.replace('_', ' ') in content.lower():
                implemented_controls.append(control)
                
        return {
            'compliant': len(implemented_controls) >= 7,
            'implemented_controls': implemented_controls,
            'score': len(implemented_controls) / len(owasp_controls)
        }

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("Usage: python quantum-security-scanner.py <file_path>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
        
    scanner = QuantumSecurityScanner()
    result = scanner.scan_file(file_path)
    
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()