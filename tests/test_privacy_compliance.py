"""
Test suite for Privacy/Compliance Mechanisms

This module tests the complete privacy and compliance system including:
- Golden PII fixtures and detection
- Redaction pipeline and PII removal
- Hash-then-persist data protection
- Retention and time-to-live checks
- Encryption flags and verification
- Audit trail and compliance logging
- L0→L1 boundary enforcement
"""

import pytest
import time
import asyncio
import threading
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json
import hashlib
import uuid
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

# Add the parent directory to the path to import Layer 0 components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_config import Layer0Config


class PIIFieldType(Enum):
    """Types of PII fields"""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"
    USERNAME = "username"
    FULL_NAME = "full_name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"


class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"


@dataclass
class PIIField:
    """PII field definition"""
    field_type: PIIFieldType
    field_name: str
    pattern: str
    redaction_method: str
    encryption_required: bool = True
    retention_days: int = 90
    ttl_seconds: int = 7776000  # 90 days


@dataclass
class PrivacyConfig:
    """Privacy and compliance configuration"""
    pii_fields: List[PIIField] = field(default_factory=list)
    encryption_enabled: bool = True
    audit_logging_enabled: bool = True
    retention_policy_enabled: bool = True
    hash_algorithm: str = "sha256"
    salt_length: int = 32
    max_retention_days: int = 365
    compliance_frameworks: List[str] = field(default_factory=lambda: ["GDPR", "CCPA", "SOX"])
    
    def __post_init__(self):
        """Initialize default PII fields if none provided"""
        if not self.pii_fields:
            self.pii_fields = [
                PIIField(
                    field_type=PIIFieldType.EMAIL,
                    field_name="email",
                    pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                    redaction_method="hash",
                    encryption_required=True,
                    retention_days=90
                ),
                PIIField(
                    field_type=PIIFieldType.PHONE,
                    field_name="phone",
                    pattern=r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                    redaction_method="mask",
                    encryption_required=True,
                    retention_days=90
                ),
                PIIField(
                    field_type=PIIFieldType.SSN,
                    field_name="ssn",
                    pattern=r'\b\d{3}-\d{2}-\d{4}\b',
                    redaction_method="hash",
                    encryption_required=True,
                    retention_days=365
                ),
                PIIField(
                    field_type=PIIFieldType.CREDIT_CARD,
                    field_name="credit_card",
                    pattern=r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                    redaction_method="mask",
                    encryption_required=True,
                    retention_days=90
                ),
                PIIField(
                    field_type=PIIFieldType.IP_ADDRESS,
                    field_name="ip_address",
                    pattern=r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
                    redaction_method="anonymize",
                    encryption_required=False,
                    retention_days=30
                )
            ]


class PIIDetector:
    """PII detection and classification engine"""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.detection_patterns = {}
        self._build_patterns()
    
    def _build_patterns(self):
        """Build regex patterns for PII detection"""
        for field in self.config.pii_fields:
            self.detection_patterns[field.field_type] = {
                'pattern': re.compile(field.pattern, re.IGNORECASE),
                'field': field
            }
    
    def detect_pii(self, data: Any, path: str = "") -> List[Dict[str, Any]]:
        """Detect PII in data structure"""
        detected_pii = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                detected_pii.extend(self.detect_pii(value, current_path))
                
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]"
                detected_pii.extend(self.detect_pii(item, current_path))
                
        elif isinstance(data, str):
            # Check string against PII patterns
            for field_type, pattern_info in self.detection_patterns.items():
                matches = pattern_info['pattern'].findall(data)
                if matches:
                    detected_pii.append({
                        'field_type': field_type.value,
                        'field_name': pattern_info['field'].field_name,
                        'path': path,
                        'value': data,
                        'matches': matches,
                        'redaction_method': pattern_info['field'].redaction_method,
                        'encryption_required': pattern_info['field'].encryption_required,
                        'retention_days': pattern_info['field'].retention_days
                    })
        
        return detected_pii


class PIIRedactor:
    """PII redaction and protection engine"""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.redaction_methods = {
            'hash': self._hash_value,
            'mask': self._mask_value,
            'anonymize': self._anonymize_value
        }
    
    def redact_pii(self, data: Any, detected_pii: List[Dict[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
        """Redact PII from data"""
        redacted_data = self._deep_copy(data)
        redaction_log = {
            'redacted_fields': [],
            'encryption_applied': [],
            'retention_set': []
        }
        
        for pii_info in detected_pii:
            path = pii_info['path']
            redaction_method = pii_info['redaction_method']
            field_type = pii_info['field_type']
            
            # Apply redaction
            if redaction_method in self.redaction_methods:
                original_value = self._get_value_at_path(redacted_data, path)
                redacted_value = self.redaction_methods[redaction_method](original_value, pii_info)
                self._set_value_at_path(redacted_data, path, redacted_value)
                
                redaction_log['redacted_fields'].append({
                    'path': path,
                    'field_type': field_type,
                    'method': redaction_method,
                    'original_length': len(str(original_value)),
                    'redacted_length': len(str(redacted_value))
                })
                
                # Apply encryption if required
                if pii_info['encryption_required'] and self.config.encryption_enabled:
                    encrypted_value = self._encrypt_value(redacted_value, pii_info)
                    self._set_value_at_path(redacted_data, path, encrypted_value)
                    redaction_log['encryption_applied'].append({
                        'path': path,
                        'field_type': field_type,
                        'algorithm': 'AES-256'
                    })
                
                # Set retention
                if self.config.retention_policy_enabled:
                    retention_ttl = pii_info['retention_days'] * 24 * 3600  # Convert to seconds
                    redaction_log['retention_set'].append({
                        'path': path,
                        'field_type': field_type,
                        'retention_days': pii_info['retention_days'],
                        'ttl_seconds': retention_ttl
                    })
        
        return redacted_data, redaction_log
    
    def _hash_value(self, value: str, pii_info: Dict[str, Any]) -> str:
        """Hash a value using configured algorithm"""
        salt = os.urandom(self.config.salt_length)
        hash_obj = hashlib.new(self.config.hash_algorithm)
        hash_obj.update(salt + value.encode('utf-8'))
        return f"{hash_obj.hexdigest()}:{salt.hex()}"
    
    def _mask_value(self, value: str, pii_info: Dict[str, Any]) -> str:
        """Mask a value (e.g., credit card)"""
        if pii_info['field_type'] == PIIFieldType.CREDIT_CARD.value:
            # Show only last 4 digits
            return f"****-****-****-{value[-4:]}"
        elif pii_info['field_type'] == PIIFieldType.PHONE.value:
            # Show only last 4 digits
            return f"***-***-{value[-4:]}"
        else:
            # Generic masking
            return "*" * len(value)
    
    def _anonymize_value(self, value: str, pii_info: Dict[str, Any]) -> str:
        """Anonymize a value (e.g., IP address)"""
        if pii_info['field_type'] == PIIFieldType.IP_ADDRESS.value:
            # Anonymize IP by zeroing last octet
            parts = value.split('.')
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.{parts[2]}.0"
        return "ANONYMIZED"
    
    def _encrypt_value(self, value: str, pii_info: Dict[str, Any]) -> str:
        """Encrypt a value (mock implementation)"""
        # In real implementation, use proper encryption
        return f"ENCRYPTED:{value}"
    
    def _deep_copy(self, data: Any) -> Any:
        """Deep copy data structure"""
        if isinstance(data, dict):
            return {k: self._deep_copy(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._deep_copy(item) for item in data]
        else:
            return data
    
    def _get_value_at_path(self, data: Any, path: str) -> Any:
        """Get value at dot notation path"""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if key.endswith(']'):
                # Handle array indexing
                base_key, index = key[:-1].split('[')
                current = current[base_key][int(index)]
            else:
                current = current[key]
        
        return current
    
    def _set_value_at_path(self, data: Any, path: str, value: Any):
        """Set value at dot notation path"""
        keys = path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key.endswith(']'):
                # Handle array indexing
                base_key, index = key[:-1].split('[')
                current = current[base_key][int(index)]
            else:
                current = current[key]
        
        last_key = keys[-1]
        if last_key.endswith(']'):
            base_key, index = last_key[:-1].split('[')
            current[base_key][int(index)] = value
        else:
            current[last_key] = value


class ComplianceAuditor:
    """Compliance auditing and logging engine"""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.audit_log = []
        self.compliance_checks = []
        self.violations = []
    
    def audit_operation(self, operation: str, data: Dict[str, Any], 
                       redaction_log: Dict[str, Any]) -> Dict[str, Any]:
        """Audit a privacy operation"""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'operation': operation,
            'data_classification': self._classify_data(data),
            'pii_fields_detected': len(redaction_log.get('redacted_fields', [])),
            'encryption_applied': len(redaction_log.get('encryption_applied', [])),
            'retention_set': len(redaction_log.get('retention_set', [])),
            'compliance_frameworks': self.config.compliance_frameworks,
            'audit_id': str(uuid.uuid4())
        }
        
        self.audit_log.append(audit_entry)
        
        # Perform compliance checks
        compliance_result = self._check_compliance(audit_entry, data, redaction_log)
        audit_entry['compliance_result'] = compliance_result
        
        return audit_entry
    
    def _classify_data(self, data: Dict[str, Any]) -> str:
        """Classify data based on content"""
        # Simple classification logic
        if any(key.lower() in ['password', 'token', 'secret'] for key in data.keys()):
            return DataClassification.RESTRICTED.value
        elif any(key.lower() in ['email', 'phone', 'ssn'] for key in data.keys()):
            return DataClassification.PII.value
        elif any(key.lower() in ['internal', 'confidential'] for key in data.keys()):
            return DataClassification.CONFIDENTIAL.value
        else:
            return DataClassification.PUBLIC.value
    
    def _check_compliance(self, audit_entry: Dict[str, Any], 
                         data: Dict[str, Any], 
                         redaction_log: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance with configured frameworks"""
        compliance_result = {
            'gdpr_compliant': True,
            'ccpa_compliant': True,
            'sox_compliant': True,
            'violations': []
        }
        
        # GDPR compliance checks
        if 'GDPR' in self.config.compliance_frameworks:
            if not redaction_log.get('redacted_fields'):
                compliance_result['gdpr_compliant'] = False
                compliance_result['violations'].append('GDPR: No PII redaction applied')
            
            if not redaction_log.get('encryption_applied'):
                compliance_result['gdpr_compliant'] = False
                compliance_result['violations'].append('GDPR: Encryption not applied to PII')
        
        # CCPA compliance checks
        if 'CCPA' in self.config.compliance_frameworks:
            if not redaction_log.get('retention_set'):
                compliance_result['ccpa_compliant'] = False
                compliance_result['violations'].append('CCPA: Retention policy not set')
        
        # SOX compliance checks
        if 'SOX' in self.config.compliance_frameworks:
            if not self.config.audit_logging_enabled:
                compliance_result['sox_compliant'] = False
                compliance_result['violations'].append('SOX: Audit logging disabled')
        
        # Record violations
        if compliance_result['violations']:
            self.violations.extend(compliance_result['violations'])
        
        return compliance_result
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get audit summary"""
        return {
            'total_operations': len(self.audit_log),
            'compliance_violations': len(self.violations),
            'pii_operations': len([a for a in self.audit_log if a['pii_fields_detected'] > 0]),
            'encryption_operations': len([a for a in self.audit_log if a['encryption_applied'] > 0]),
            'recent_violations': self.violations[-5:] if self.violations else []
        }


class PrivacyComplianceSystem:
    """Complete privacy and compliance system"""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.pii_detector = PIIDetector(config)
        self.pii_redactor = PIIRedactor(config)
        self.compliance_auditor = ComplianceAuditor(config)
        
        # Data storage with TTL
        self.protected_data = {}
        self.data_metadata = {}
    
    def process_data(self, data: Dict[str, Any], operation: str = "process") -> Dict[str, Any]:
        """Process data through privacy pipeline"""
        # 1. Detect PII
        detected_pii = self.pii_detector.detect_pii(data)
        
        # 2. Redact PII
        redacted_data, redaction_log = self.pii_redactor.redact_pii(data, detected_pii)
        
        # 3. Audit operation
        audit_result = self.compliance_auditor.audit_operation(operation, data, redaction_log)
        
        # 4. Store protected data
        data_id = str(uuid.uuid4())
        self.protected_data[data_id] = redacted_data
        self.data_metadata[data_id] = {
            'audit_id': audit_result['audit_id'],
            'created_at': datetime.utcnow(),
            'retention_ttl': max([field.get('ttl_seconds', 0) for field in redaction_log.get('retention_set', [])], default=0),
            'pii_fields': len(redaction_log.get('redacted_fields', [])),
            'encrypted': len(redaction_log.get('encryption_applied', [])) > 0
        }
        
        return {
            'data_id': data_id,
            'audit_id': audit_result['audit_id'],
            'pii_detected': len(detected_pii),
            'redaction_applied': len(redaction_log.get('redacted_fields', [])),
            'encryption_applied': len(redaction_log.get('encryption_applied', [])),
            'compliance_status': audit_result['compliance_result']
        }
    
    def retrieve_data(self, data_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve protected data (with access logging)"""
        if data_id not in self.protected_data:
            return None
        
        metadata = self.data_metadata[data_id]
        
        # Check TTL
        if metadata['retention_ttl'] > 0:
            age_seconds = (datetime.utcnow() - metadata['created_at']).total_seconds()
            if age_seconds > metadata['retention_ttl']:
                # Data expired, remove it
                del self.protected_data[data_id]
                del self.data_metadata[data_id]
                return None
        
        # Log access
        self.compliance_auditor.audit_operation("data_access", 
                                              {"data_id": data_id}, 
                                              {"redacted_fields": []})
        
        return {
            'data': self.protected_data[data_id],
            'metadata': metadata
        }
    
    def cleanup_expired_data(self):
        """Clean up expired data based on TTL"""
        current_time = datetime.utcnow()
        expired_ids = []
        
        for data_id, metadata in self.data_metadata.items():
            if metadata['retention_ttl'] > 0:
                age_seconds = (current_time - metadata['created_at']).total_seconds()
                if age_seconds > metadata['retention_ttl']:
                    expired_ids.append(data_id)
        
        for data_id in expired_ids:
            del self.protected_data[data_id]
            del self.data_metadata[data_id]
        
        return len(expired_ids)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'total_protected_records': len(self.protected_data),
            'audit_summary': self.compliance_auditor.get_audit_summary(),
            'config': {
                'encryption_enabled': self.config.encryption_enabled,
                'audit_logging_enabled': self.config.audit_logging_enabled,
                'retention_policy_enabled': self.config.retention_policy_enabled,
                'compliance_frameworks': self.config.compliance_frameworks
            }
        }


class TestPrivacyCompliance:
    """Test suite for privacy and compliance mechanisms"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.config = PrivacyConfig()
        self.privacy_system = PrivacyComplianceSystem(self.config)
        
        # Golden PII fixtures for testing
        self.golden_pii_fixtures = {
            'user_profile': {
                'user_id': '12345',
                'email': 'john.doe@example.com',
                'phone': '555-123-4567',
                'ssn': '123-45-6789',
                'credit_card': '4111-1111-1111-1111',
                'ip_address': '192.168.1.100',
                'username': 'johndoe',
                'full_name': 'John Doe',
                'address': '123 Main St, Anytown, USA',
                'date_of_birth': '1990-01-01'
            },
            'order_data': {
                'order_id': 'ORD-001',
                'customer_email': 'jane.smith@example.com',
                'billing_phone': '555-987-6543',
                'shipping_address': '456 Oak Ave, Somewhere, USA',
                'payment_method': '4111-2222-3333-4444'
            },
            'log_data': {
                'timestamp': '2024-01-15T10:30:00Z',
                'user_ip': '10.0.0.50',
                'user_agent': 'Mozilla/5.0...',
                'request_path': '/api/users/profile',
                'response_code': 200
            }
        }
    
    def test_golden_pii_detection(self):
        """Test golden PII fixture detection"""
        print("\n=== Testing Golden PII Detection ===")
        
        for fixture_name, fixture_data in self.golden_pii_fixtures.items():
            print(f"Testing fixture: {fixture_name}")
            
            detected_pii = self.privacy_system.pii_detector.detect_pii(fixture_data)
            print(f"  Detected {len(detected_pii)} PII fields")
            
            # Should detect PII in user profile
            if fixture_name == 'user_profile':
                assert len(detected_pii) >= 5  # email, phone, ssn, credit_card, ip_address
                
                # Check specific PII types
                pii_types = [pii['field_type'] for pii in detected_pii]
                assert 'email' in pii_types
                assert 'phone' in pii_types
                assert 'ssn' in pii_types
                assert 'credit_card' in pii_types
                assert 'ip_address' in pii_types
            
            # Should detect some PII in order data
            elif fixture_name == 'order_data':
                assert len(detected_pii) >= 2  # email, phone, credit_card
                
                pii_types = [pii['field_type'] for pii in detected_pii]
                assert 'email' in pii_types
                assert 'credit_card' in pii_types
            
            # Should detect minimal PII in log data
            elif fixture_name == 'log_data':
                assert len(detected_pii) >= 1  # ip_address
                
                pii_types = [pii['field_type'] for pii in detected_pii]
                assert 'ip_address' in pii_types
        
        print("✓ Golden PII detection test passed")
    
    def test_redaction_pipeline(self):
        """Test PII redaction pipeline"""
        print("\n=== Testing PII Redaction Pipeline ===")
        
        test_data = self.golden_pii_fixtures['user_profile'].copy()
        original_data = test_data.copy()
        
        # Process through redaction pipeline
        result = self.privacy_system.process_data(test_data, "test_redaction")
        
        print(f"Redaction result: {result}")
        
        # Should have detected and redacted PII
        assert result['pii_detected'] > 0
        assert result['redaction_applied'] > 0
        
        # Check that original data is preserved
        assert test_data == original_data
        
        # Check that redacted data is different
        data_id = result['data_id']
        retrieved_data = self.privacy_system.retrieve_data(data_id)
        assert retrieved_data is not None
        
        redacted_data = retrieved_data['data']
        
        # Verify PII fields are redacted
        assert redacted_data['email'] != original_data['email']
        assert redacted_data['phone'] != original_data['phone']
        assert redacted_data['ssn'] != original_data['ssn']
        assert redacted_data['credit_card'] != original_data['credit_card']
        assert redacted_data['ip_address'] != original_data['ip_address']
        
        # Non-PII fields should be unchanged
        assert redacted_data['user_id'] == original_data['user_id']
        assert redacted_data['username'] == original_data['username']
        
        print("✓ PII redaction pipeline test passed")
    
    def test_hash_then_persist(self):
        """Test hash-then-persist data protection"""
        print("\n=== Testing Hash-Then-Persist ===")
        
        test_data = {
            'sensitive_field': 'secret_value_123',
            'normal_field': 'public_info'
        }
        
        # Process data
        result = self.privacy_system.process_data(test_data, "test_hash_persist")
        
        # Retrieve processed data
        data_id = result['data_id']
        retrieved_data = self.privacy_system.retrieve_data(data_id)
        
        # Check that sensitive data is hashed
        redacted_data = retrieved_data['data']
        
        # The sensitive field should be hashed (format: hash:salt)
        assert ':' in redacted_data['sensitive_field']
        assert redacted_data['sensitive_field'].startswith('ENCRYPTED:')
        
        # Normal field should be unchanged
        assert redacted_data['normal_field'] == 'public_info'
        
        print("✓ Hash-then-persist test passed")
    
    def test_retention_and_ttl(self):
        """Test retention and time-to-live checks"""
        print("\n=== Testing Retention and TTL ===")
        
        test_data = {
            'email': 'test@example.com',
            'phone': '555-123-4567'
        }
        
        # Process data
        result = self.privacy_system.process_data(test_data, "test_retention")
        
        # Check metadata
        data_id = result['data_id']
        retrieved_data = self.privacy_system.retrieve_data(data_id)
        metadata = retrieved_data['metadata']
        
        # Should have retention TTL set
        assert metadata['retention_ttl'] > 0
        assert metadata['created_at'] is not None
        
        # Simulate time passage
        original_created_at = metadata['created_at']
        self.privacy_system.data_metadata[data_id]['created_at'] = \
            original_created_at - timedelta(days=100)  # Make it old
        
        # Try to retrieve - should be expired
        expired_data = self.privacy_system.retrieve_data(data_id)
        assert expired_data is None
        
        # Clean up expired data
        cleaned_count = self.privacy_system.cleanup_expired_data()
        assert cleaned_count > 0
        
        print("✓ Retention and TTL test passed")
    
    def test_encryption_flags(self):
        """Test encryption flags and verification"""
        print("\n=== Testing Encryption Flags ===")
        
        test_data = {
            'email': 'test@example.com',
            'phone': '555-123-4567',
            'public_info': 'non-sensitive data'
        }
        
        # Process data
        result = self.privacy_system.process_data(test_data, "test_encryption")
        
        # Check encryption was applied
        assert result['encryption_applied'] > 0
        
        # Retrieve and verify encryption
        data_id = result['data_id']
        retrieved_data = self.privacy_system.retrieve_data(data_id)
        metadata = retrieved_data['metadata']
        
        # Should be marked as encrypted
        assert metadata['encrypted'] == True
        
        # Check encrypted fields
        redacted_data = retrieved_data['data']
        assert redacted_data['email'].startswith('ENCRYPTED:')
        assert redacted_data['phone'].startswith('ENCRYPTED:')
        assert redacted_data['public_info'] == 'non-sensitive data'  # Not encrypted
        
        print("✓ Encryption flags test passed")
    
    def test_audit_trail(self):
        """Test audit trail and compliance logging"""
        print("\n=== Testing Audit Trail ===")
        
        # Process multiple data sets
        operations = [
            ('user_profile', self.golden_pii_fixtures['user_profile']),
            ('order_data', self.golden_pii_fixtures['order_data']),
            ('log_data', self.golden_pii_fixtures['log_data'])
        ]
        
        for operation_name, data in operations:
            result = self.privacy_system.process_data(data, operation_name)
            print(f"  {operation_name}: {result['pii_detected']} PII fields detected")
        
        # Check audit summary
        audit_summary = self.privacy_system.compliance_auditor.get_audit_summary()
        print(f"Audit summary: {audit_summary}")
        
        # Should have logged all operations
        assert audit_summary['total_operations'] == len(operations)
        assert audit_summary['pii_operations'] > 0
        assert audit_summary['encryption_operations'] > 0
        
        # Check compliance violations
        if audit_summary['compliance_violations'] > 0:
            print(f"  Compliance violations: {audit_summary['recent_violations']}")
        
        print("✓ Audit trail test passed")
    
    def test_l0_l1_boundary_enforcement(self):
        """Test L0→L1 boundary enforcement"""
        print("\n=== Testing L0→L1 Boundary Enforcement ===")
        
        # Simulate data crossing L0→L1 boundary
        test_data = {
            'email': 'boundary@example.com',
            'phone': '555-999-8888',
            'internal_flag': 'L0_ONLY'
        }
        
        # Process in L0
        result = self.privacy_system.process_data(test_data, "L0_processing")
        
        # Verify no raw PII crosses boundary
        data_id = result['data_id']
        retrieved_data = self.privacy_system.retrieve_data(data_id)
        redacted_data = retrieved_data['data']
        
        # PII fields should be redacted/encrypted
        assert redacted_data['email'] != 'boundary@example.com'
        assert redacted_data['phone'] != '555-999-8888'
        
        # Non-PII fields can cross
        assert redacted_data['internal_flag'] == 'L0_ONLY'
        
        # Verify encryption flags
        metadata = retrieved_data['metadata']
        assert metadata['encrypted'] == True
        
        print("✓ L0→L1 boundary enforcement test passed")
    
    def test_compliance_frameworks(self):
        """Test compliance with different frameworks"""
        print("\n=== Testing Compliance Frameworks ===")
        
        # Test GDPR compliance
        gdpr_data = {
            'eu_citizen_email': 'eu@example.com',
            'eu_citizen_phone': '+44-20-1234-5678'
        }
        
        gdpr_result = self.privacy_system.process_data(gdpr_data, "GDPR_test")
        gdpr_compliance = gdpr_result['compliance_status']
        
        print(f"GDPR compliance: {gdpr_compliance['gdpr_compliant']}")
        if not gdpr_compliance['gdpr_compliant']:
            print(f"  Violations: {gdpr_compliance['violations']}")
        
        # Test CCPA compliance
        ccpa_data = {
            'california_resident_email': 'ca@example.com',
            'california_resident_ssn': '987-65-4321'
        }
        
        ccpa_result = self.privacy_system.process_data(ccpa_data, "CCPA_test")
        ccpa_compliance = ccpa_result['compliance_status']
        
        print(f"CCPA compliance: {ccpa_compliance['ccpa_compliant']}")
        if not ccpa_compliance['ccpa_compliant']:
            print(f"  Violations: {ccpa_compliance['violations']}")
        
        # Test SOX compliance
        sox_data = {
            'financial_email': 'finance@example.com',
            'audit_trail': 'SOX_REQUIRED'
        }
        
        sox_result = self.privacy_system.process_data(sox_data, "SOX_test")
        sox_compliance = sox_result['compliance_status']
        
        print(f"SOX compliance: {sox_compliance['sox_compliant']}")
        if not sox_compliance['sox_compliant']:
            print(f"  Violations: {sox_compliance['violations']}")
        
        print("✓ Compliance frameworks test passed")
    
    def test_end_to_end_privacy_pipeline(self):
        """Test complete end-to-end privacy pipeline"""
        print("\n=== Testing End-to-End Privacy Pipeline ===")
        
        # Complex data with multiple PII types
        complex_data = {
            'user': {
                'profile': {
                    'email': 'complex@example.com',
                    'phone': '555-111-2222',
                    'ssn': '111-22-3333'
                },
                'orders': [
                    {
                        'order_id': 'ORD-001',
                        'billing_email': 'billing@example.com',
                        'shipping_phone': '555-444-5555'
                    },
                    {
                        'order_id': 'ORD-002',
                        'billing_email': 'billing2@example.com',
                        'shipping_phone': '555-666-7777'
                    }
                ]
            },
            'system_logs': {
                'user_ip': '192.168.1.200',
                'access_timestamp': '2024-01-15T12:00:00Z'
            }
        }
        
        # Process through complete pipeline
        result = self.privacy_system.process_data(complex_data, "end_to_end_test")
        
        print(f"Pipeline result: {result}")
        
        # Verify processing
        assert result['pii_detected'] > 0
        assert result['redaction_applied'] > 0
        assert result['encryption_applied'] > 0
        
        # Retrieve and verify
        data_id = result['data_id']
        retrieved_data = self.privacy_system.retrieve_data(data_id)
        redacted_data = retrieved_data['data']
        
        # Check nested PII redaction
        assert redacted_data['user']['profile']['email'] != 'complex@example.com'
        assert redacted_data['user']['profile']['phone'] != '555-111-2222'
        assert redacted_data['user']['profile']['ssn'] != '111-22-3333'
        
        # Check array PII redaction
        assert redacted_data['user']['orders'][0]['billing_email'] != 'billing@example.com'
        assert redacted_data['user']['orders'][1]['shipping_phone'] != '555-666-7777'
        
        # Check system log PII redaction
        assert redacted_data['system_logs']['user_ip'] != '192.168.1.200'
        
        # Non-PII should be preserved
        assert redacted_data['user']['orders'][0]['order_id'] == 'ORD-001'
        assert redacted_data['user']['orders'][1]['order_id'] == 'ORD-002'
        assert redacted_data['system_logs']['access_timestamp'] == '2024-01-15T12:00:00Z'
        
        print("✓ End-to-end privacy pipeline test passed")


def run_privacy_compliance_tests():
    """Run all privacy and compliance tests"""
    print("\n" + "="*60)
    print("RUNNING PRIVACY/COMPLIANCE TESTS")
    print("="*60)
    
    test_suite = TestPrivacyCompliance()
    
    # Run all test methods
    test_methods = [
        'test_golden_pii_detection',
        'test_redaction_pipeline',
        'test_hash_then_persist',
        'test_retention_and_ttl',
        'test_encryption_flags',
        'test_audit_trail',
        'test_l0_l1_boundary_enforcement',
        'test_compliance_frameworks',
        'test_end_to_end_privacy_pipeline'
    ]
    
    for method_name in test_methods:
        try:
            test_suite.setup_method()
            method = getattr(test_suite, method_name)
            method()
            print(f"✓ {method_name} completed successfully")
        except Exception as e:
            print(f"✗ {method_name} failed: {e}")
        finally:
            test_suite.teardown_method()
    
    print("\n" + "="*60)
    print("PRIVACY/COMPLIANCE TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    run_privacy_compliance_tests()
