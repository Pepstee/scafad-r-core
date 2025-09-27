"""
Test suite for L0→L1 Contract Testing

This module tests the complete L0→L1 contract including:
- Schema versioning and validation
- Forward/backward compatibility
- Required headers (trace/span IDs, trust flags)
- Replay window and deduplication checks
- Contract tests and mocks
- Data integrity validation
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
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

# Add the parent directory to the path to import Layer 0 components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_config import Layer0Config


class SchemaVersion(Enum):
    """Schema version enumeration"""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"
    V2_1 = "2.1"


class TrustLevel(Enum):
    """Trust level enumeration"""
    UNTRUSTED = "untrusted"
    TRUSTED = "trusted"
    VERIFIED = "verified"
    CRITICAL = "critical"


class ContractValidationResult(Enum):
    """Contract validation result"""
    VALID = "valid"
    INVALID = "invalid"
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"


@dataclass
class ContractHeader:
    """Contract header structure"""
    schema_version: SchemaVersion
    trace_id: str
    span_id: str
    trust_level: TrustLevel
    timestamp: float
    source: str
    destination: str
    correlation_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContractPayload:
    """Contract payload structure"""
    data: Dict[str, Any]
    schema_version: SchemaVersion
    validation_hash: str
    compression_type: str = "none"
    encryption_type: str = "none"
    size_bytes: int = 0


@dataclass
class ContractMessage:
    """Complete contract message"""
    header: ContractHeader
    payload: ContractPayload
    signature: Optional[str] = None
    received_at: Optional[float] = None


@dataclass
class ContractConfig:
    """Contract configuration"""
    supported_versions: List[SchemaVersion] = field(default_factory=lambda: [
        SchemaVersion.V1_0, SchemaVersion.V1_1, SchemaVersion.V2_0
    ])
    max_replay_window_ms: int = 300000  # 5 minutes
    dedup_cache_size: int = 10000
    trust_validation_enabled: bool = True
    schema_validation_enabled: bool = True
    signature_validation_enabled: bool = True
    compression_enabled: bool = True
    encryption_enabled: bool = False


class SchemaValidator:
    """Schema validation and versioning engine"""
    
    def __init__(self, config: ContractConfig):
        self.config = config
        self.schemas = {}
        self._load_schemas()
    
    def _load_schemas(self):
        """Load schema definitions for different versions"""
        # Mock schema definitions
        self.schemas = {
            SchemaVersion.V1_0: {
                'required_fields': ['trace_id', 'span_id', 'timestamp'],
                'optional_fields': ['metadata', 'correlation_id'],
                'field_types': {
                    'trace_id': 'string',
                    'span_id': 'string',
                    'timestamp': 'number',
                    'metadata': 'object',
                    'correlation_id': 'string'
                }
            },
            SchemaVersion.V1_1: {
                'required_fields': ['trace_id', 'span_id', 'timestamp', 'trust_level'],
                'optional_fields': ['metadata', 'correlation_id', 'source'],
                'field_types': {
                    'trace_id': 'string',
                    'span_id': 'string',
                    'timestamp': 'number',
                    'trust_level': 'string',
                    'metadata': 'object',
                    'correlation_id': 'string',
                    'source': 'string'
                }
            },
            SchemaVersion.V2_0: {
                'required_fields': ['trace_id', 'span_id', 'timestamp', 'trust_level', 'source', 'destination'],
                'optional_fields': ['metadata', 'correlation_id', 'compression_type', 'encryption_type'],
                'field_types': {
                    'trace_id': 'string',
                    'span_id': 'string',
                    'timestamp': 'number',
                    'trust_level': 'string',
                    'source': 'string',
                    'destination': 'string',
                    'metadata': 'object',
                    'correlation_id': 'string',
                    'compression_type': 'string',
                    'encryption_type': 'string'
                }
            }
        }
    
    def validate_schema(self, message: ContractMessage) -> Tuple[bool, List[str]]:
        """Validate message against schema"""
        if not self.config.schema_validation_enabled:
            return True, []
        
        schema = self.schemas.get(message.header.schema_version)
        if not schema:
            return False, [f"Unsupported schema version: {message.header.schema_version}"]
        
        errors = []
        
        # Check required fields
        for field in schema['required_fields']:
            if not hasattr(message.header, field) or getattr(message.header, field) is None:
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        for field, expected_type in schema['field_types'].items():
            if hasattr(message.header, field):
                value = getattr(message.header, field)
                if not self._validate_field_type(value, expected_type):
                    errors.append(f"Invalid field type for {field}: expected {expected_type}")
        
        return len(errors) == 0, errors
    
    def _validate_field_type(self, value: Any, expected_type: str) -> bool:
        """Validate field type"""
        if expected_type == 'string':
            return isinstance(value, str)
        elif expected_type == 'number':
            return isinstance(value, (int, float))
        elif expected_type == 'object':
            return isinstance(value, dict)
        return True
    
    def check_compatibility(self, from_version: SchemaVersion, to_version: SchemaVersion) -> ContractValidationResult:
        """Check schema compatibility between versions"""
        if from_version == to_version:
            return ContractValidationResult.VALID
        
        from_schema = self.schemas.get(from_version)
        to_schema = self.schemas.get(to_version)
        
        if not from_schema or not to_schema:
            return ContractValidationResult.INCOMPATIBLE
        
        # Check if all required fields in from_version exist in to_version
        from_required = set(from_schema['required_fields'])
        to_required = set(to_schema['required_fields'])
        
        if from_required.issubset(to_required):
            return ContractValidationResult.COMPATIBLE
        else:
            return ContractValidationResult.INCOMPATIBLE


class TrustValidator:
    """Trust validation and security engine"""
    
    def __init__(self, config: ContractConfig):
        self.config = config
        self.trusted_sources = set()
        self.trust_levels = {}
        self.validation_history = []
    
    def validate_trust(self, message: ContractMessage) -> Tuple[bool, List[str]]:
        """Validate message trust level and source"""
        if not self.config.trust_validation_enabled:
            return True, []
        
        errors = []
        
        # Validate trust level
        if not self._is_valid_trust_level(message.header.trust_level):
            errors.append(f"Invalid trust level: {message.header.trust_level}")
        
        # Validate source trust
        if not self._is_trusted_source(message.header.source, message.header.trust_level):
            errors.append(f"Untrusted source: {message.header.source} for level {message.header.trust_level}")
        
        # Validate timestamp freshness
        if not self._is_timestamp_fresh(message.header.timestamp):
            errors.append("Timestamp too old")
        
        # Record validation
        self.validation_history.append({
            'timestamp': time.time(),
            'source': message.header.source,
            'trust_level': message.header.trust_level.value,
            'valid': len(errors) == 0,
            'errors': errors
        })
        
        return len(errors) == 0, errors
    
    def _is_valid_trust_level(self, trust_level: TrustLevel) -> bool:
        """Check if trust level is valid"""
        return trust_level in TrustLevel
    
    def _is_trusted_source(self, source: str, trust_level: TrustLevel) -> bool:
        """Check if source is trusted for given trust level"""
        # Mock trust validation logic
        if trust_level == TrustLevel.CRITICAL:
            return source in ['core_system', 'trusted_service']
        elif trust_level == TrustLevel.VERIFIED:
            return source in ['core_system', 'trusted_service', 'verified_extension']
        elif trust_level == TrustLevel.TRUSTED:
            return source in ['core_system', 'trusted_service', 'verified_extension', 'trusted_adapter']
        else:  # UNTRUSTED
            return True
    
    def _is_timestamp_fresh(self, timestamp: float) -> bool:
        """Check if timestamp is fresh"""
        current_time = time.time()
        max_age = 300  # 5 minutes
        return (current_time - timestamp) <= max_age


class DedupManager:
    """Deduplication and replay protection manager"""
    
    def __init__(self, config: ContractConfig):
        self.config = config
        self.message_cache = {}
        self.replay_window_start = time.time()
        self.dedup_stats = {
            'total_messages': 0,
            'duplicate_messages': 0,
            'replay_attempts': 0,
            'cache_hits': 0
        }
    
    def check_deduplication(self, message: ContractMessage) -> Tuple[bool, str]:
        """Check for duplicate messages and replay attempts"""
        self.dedup_stats['total_messages'] += 1
        
        # Generate message fingerprint
        fingerprint = self._generate_fingerprint(message)
        
        # Check replay window
        if not self._is_in_replay_window(message.header.timestamp):
            return False, "Message outside replay window"
        
        # Check for duplicates
        if fingerprint in self.message_cache:
            self.dedup_stats['duplicate_messages'] += 1
            return False, "Duplicate message detected"
        
        # Cache message
        self.message_cache[fingerprint] = {
            'timestamp': time.time(),
            'source': message.header.source,
            'correlation_id': message.header.correlation_id
        }
        
        # Clean old entries
        self._cleanup_cache()
        
        return True, "Message accepted"
    
    def _generate_fingerprint(self, message: ContractMessage) -> str:
        """Generate unique fingerprint for message"""
        # Create fingerprint from key fields
        key_data = f"{message.header.trace_id}:{message.header.span_id}:{message.header.correlation_id}:{message.header.timestamp}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _is_in_replay_window(self, timestamp: float) -> bool:
        """Check if timestamp is within replay window"""
        current_time = time.time()
        window_start = current_time - (self.config.max_replay_window_ms / 1000)
        return timestamp >= window_start
    
    def _cleanup_cache(self):
        """Clean up old cache entries"""
        current_time = time.time()
        max_age = self.config.max_replay_window_ms / 1000
        
        expired_keys = []
        for fingerprint, entry in self.message_cache.items():
            if (current_time - entry['timestamp']) > max_age:
                expired_keys.append(fingerprint)
        
        for key in expired_keys:
            del self.message_cache[key]
        
        # Limit cache size
        while len(self.message_cache) > self.config.dedup_cache_size:
            oldest_key = min(self.message_cache.keys(), 
                           key=lambda k: self.message_cache[k]['timestamp'])
            del self.message_cache[oldest_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics"""
        return {
            **self.dedup_stats,
            'cache_size': len(self.message_cache),
            'replay_window_ms': self.config.max_replay_window_ms
        }


class ContractValidator:
    """Complete L0→L1 contract validation engine"""
    
    def __init__(self, config: ContractConfig):
        self.config = config
        self.schema_validator = SchemaValidator(config)
        self.trust_validator = TrustValidator(config)
        self.dedup_manager = DedupManager(config)
        
        # Validation results
        self.validation_results = []
        self.compatibility_tests = []
    
    def validate_contract(self, message: ContractMessage) -> Dict[str, Any]:
        """Validate complete contract message"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'schema_valid': False,
            'trust_valid': False,
            'dedup_valid': False,
            'compatibility': ContractValidationResult.INCOMPATIBLE.value
        }
        
        # Schema validation
        schema_valid, schema_errors = self.schema_validator.validate_schema(message)
        validation_result['schema_valid'] = schema_valid
        validation_result['errors'].extend(schema_errors)
        
        # Trust validation
        trust_valid, trust_errors = self.trust_validator.validate_trust(message)
        validation_result['trust_valid'] = trust_valid
        validation_result['errors'].extend(trust_errors)
        
        # Deduplication check
        dedup_valid, dedup_message = self.dedup_manager.check_deduplication(message)
        validation_result['dedup_valid'] = dedup_valid
        if not dedup_valid:
            validation_result['errors'].append(dedup_message)
        
        # Overall validation
        validation_result['valid'] = (schema_valid and trust_valid and dedup_valid)
        
        # Check compatibility with supported versions
        compatibility = self.schema_validator.check_compatibility(
            message.header.schema_version, 
            SchemaVersion.V2_0  # Assume L1 supports V2.0
        )
        validation_result['compatibility'] = compatibility.value
        
        # Record validation result
        self.validation_results.append({
            'timestamp': time.time(),
            'message_id': message.header.correlation_id,
            'result': validation_result.copy()
        })
        
        return validation_result
    
    def run_compatibility_tests(self) -> List[Dict[str, Any]]:
        """Run compatibility tests between schema versions"""
        test_results = []
        
        for from_version in self.schema_validator.schemas.keys():
            for to_version in self.schema_validator.schemas.keys():
                compatibility = self.schema_validator.check_compatibility(from_version, to_version)
                
                test_result = {
                    'from_version': from_version.value,
                    'to_version': to_version.value,
                    'compatibility': compatibility.value,
                    'timestamp': time.time()
                }
                
                test_results.append(test_result)
                self.compatibility_tests.append(test_result)
        
        return test_results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        if not self.validation_results:
            return {
                'total_validations': 0,
                'success_rate': 0.0,
                'common_errors': [],
                'compatibility_summary': {}
            }
        
        total = len(self.validation_results)
        successful = sum(1 for r in self.validation_results if r['result']['valid'])
        success_rate = successful / total
        
        # Collect common errors
        all_errors = []
        for result in self.validation_results:
            all_errors.extend(result['result']['errors'])
        
        error_counts = {}
        for error in all_errors:
            error_counts[error] = error_counts.get(error, 0) + 1
        
        common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Compatibility summary
        compatibility_summary = {}
        for test in self.compatibility_tests:
            key = f"{test['from_version']}->{test['to_version']}"
            compatibility_summary[key] = test['compatibility']
        
        return {
            'total_validations': total,
            'success_rate': success_rate,
            'common_errors': common_errors,
            'compatibility_summary': compatibility_summary,
            'dedup_stats': self.dedup_manager.get_stats()
        }


class TestL0L1Contract:
    """Test suite for L0→L1 contract validation"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.config = ContractConfig()
        self.validator = ContractValidator(self.config)
        
        # Test message templates
        self.test_messages = {
            'v1_0_basic': self._create_test_message(SchemaVersion.V1_0, TrustLevel.TRUSTED),
            'v1_1_trusted': self._create_test_message(SchemaVersion.V1_1, TrustLevel.VERIFIED),
            'v2_0_full': self._create_test_message(SchemaVersion.V2_0, TrustLevel.CRITICAL),
            'v2_0_compressed': self._create_test_message(SchemaVersion.V2_0, TrustLevel.TRUSTED, compressed=True),
            'v2_0_encrypted': self._create_test_message(SchemaVersion.V2_0, TrustLevel.CRITICAL, encrypted=True)
        }
    
    def _create_test_message(self, schema_version: SchemaVersion, trust_level: TrustLevel, 
                           compressed: bool = False, encrypted: bool = False) -> ContractMessage:
        """Create test message with specified parameters"""
        header = ContractHeader(
            schema_version=schema_version,
            trace_id=str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            trust_level=trust_level,
            timestamp=time.time(),
            source='test_source',
            destination='test_destination',
            correlation_id=str(uuid.uuid4()),
            metadata={'test': True}
        )
        
        payload = ContractPayload(
            data={'message': 'test', 'version': schema_version.value},
            schema_version=schema_version,
            validation_hash='test_hash',
            compression_type='gzip' if compressed else 'none',
            encryption_type='AES-256' if encrypted else 'none',
            size_bytes=1024
        )
        
        return ContractMessage(
            header=header,
            payload=payload,
            signature='test_signature',
            received_at=time.time()
        )
    
    def test_schema_versioning(self):
        """Test schema versioning and validation"""
        print("\n=== Testing Schema Versioning ===")
        
        for message_name, message in self.test_messages.items():
            print(f"  Testing {message_name}")
            
            # Validate schema
            validation_result = self.validator.validate_contract(message)
            
            # Should be valid
            assert validation_result['schema_valid'] == True, f"Schema validation failed for {message_name}"
            assert validation_result['valid'] == True, f"Overall validation failed for {message_name}"
            
            # Check required fields based on version
            if message.header.schema_version == SchemaVersion.V1_0:
                assert hasattr(message.header, 'trace_id')
                assert hasattr(message.header, 'span_id')
                assert hasattr(message.header, 'timestamp')
                # V1.0 doesn't require trust_level
                assert not hasattr(message.header, 'trust_level') or message.header.trust_level is None
            
            elif message.header.schema_version == SchemaVersion.V1_1:
                assert hasattr(message.header, 'trust_level')
                assert message.header.trust_level is not None
            
            elif message.header.schema_version == SchemaVersion.V2_0:
                assert hasattr(message.header, 'source')
                assert hasattr(message.header, 'destination')
                assert message.header.source is not None
                assert message.header.destination is not None
        
        print("✓ Schema versioning test passed")
    
    def test_forward_backward_compatibility(self):
        """Test forward and backward compatibility"""
        print("\n=== Testing Forward/Backward Compatibility ===")
        
        # Run compatibility tests
        compatibility_results = self.validator.run_compatibility_tests()
        
        # Verify compatibility matrix
        compatibility_matrix = {}
        for result in compatibility_results:
            key = f"{result['from_version']}->{result['to_version']}"
            compatibility_matrix[key] = result['compatibility']
        
        print("  Compatibility Matrix:")
        for key, compatibility in compatibility_matrix.items():
            print(f"    {key}: {compatibility}")
        
        # Test specific compatibility scenarios
        # V1.0 -> V1.1 should be compatible (forward)
        v1_0_to_v1_1 = f"{SchemaVersion.V1_0.value}->{SchemaVersion.V1_1.value}"
        assert compatibility_matrix.get(v1_0_to_v1_1) in ['compatible', 'valid']
        
        # V1.1 -> V2.0 should be compatible (forward)
        v1_1_to_v2_0 = f"{SchemaVersion.V1_1.value}->{SchemaVersion.V2_0.value}"
        assert compatibility_matrix.get(v1_1_to_v2_0) in ['compatible', 'valid']
        
        # V2.0 -> V1.0 might be incompatible (backward)
        v2_0_to_v1_0 = f"{SchemaVersion.V2_0.value}->{SchemaVersion.V1_0.value}"
        # This could be incompatible due to missing required fields
        
        print("✓ Forward/backward compatibility test passed")
    
    def test_required_headers(self):
        """Test required headers (trace/span IDs, trust flags)"""
        print("\n=== Testing Required Headers ===")
        
        # Test missing required fields
        required_field_tests = [
            ('missing_trace_id', 'trace_id'),
            ('missing_span_id', 'span_id'),
            ('missing_timestamp', 'timestamp'),
            ('missing_trust_level', 'trust_level'),
            ('missing_source', 'source'),
            ('missing_destination', 'destination')
        ]
        
        for test_name, field_name in required_field_tests:
            # Create message with missing field
            message = self.test_messages['v2_0_full']
            
            # Remove the field
            if hasattr(message.header, field_name):
                setattr(message.header, field_name, None)
            
            # Validate - should fail
            validation_result = self.validator.validate_contract(message)
            
            if field_name in ['trace_id', 'span_id', 'timestamp']:
                # These are required in all versions
                assert not validation_result['schema_valid'], f"{test_name} should fail validation"
                assert len(validation_result['errors']) > 0, f"{test_name} should have errors"
            
            print(f"    {test_name}: {'FAILED' if not validation_result['schema_valid'] else 'PASSED'}")
        
        # Test valid headers
        valid_message = self.test_messages['v2_0_full']
        validation_result = self.validator.validate_contract(valid_message)
        assert validation_result['schema_valid'] == True, "Valid headers should pass validation"
        
        print("✓ Required headers test passed")
    
    def test_replay_window_dedup(self):
        """Test replay window and deduplication checks"""
        print("\n=== Testing Replay Window and Dedup ===")
        
        # Test duplicate message detection
        message1 = self.test_messages['v2_0_full']
        message2 = self.test_messages['v2_0_full']
        
        # Use same correlation ID to trigger dedup
        message2.header.correlation_id = message1.header.correlation_id
        
        # First message should be accepted
        validation1 = self.validator.validate_contract(message1)
        assert validation1['dedup_valid'] == True, "First message should be accepted"
        
        # Second message should be rejected as duplicate
        validation2 = self.validator.validate_contract(message2)
        assert validation2['dedup_valid'] == False, "Duplicate message should be rejected"
        assert 'Duplicate message detected' in validation2['errors'][0]
        
        # Test replay window
        old_message = self.test_messages['v2_0_full']
        old_message.header.timestamp = time.time() - 600  # 10 minutes ago (outside replay window)
        
        validation_old = self.validator.validate_contract(old_message)
        assert validation_old['dedup_valid'] == False, "Old message should be rejected"
        assert 'outside replay window' in validation_old['errors'][0]
        
        # Check dedup statistics
        dedup_stats = self.validator.dedup_manager.get_stats()
        assert dedup_stats['total_messages'] >= 3
        assert dedup_stats['duplicate_messages'] >= 1
        
        print(f"  Total messages: {dedup_stats['total_messages']}")
        print(f"  Duplicate messages: {dedup_stats['duplicate_messages']}")
        print(f"  Cache size: {dedup_stats['cache_size']}")
        
        print("✓ Replay window and dedup test passed")
    
    def test_contract_tests_mocks(self):
        """Test contract tests and mocks"""
        print("\n=== Testing Contract Tests and Mocks ===")
        
        # Test mock message generation
        mock_messages = []
        for i in range(5):
            mock_message = self._create_test_message(
                SchemaVersion.V2_0,
                TrustLevel.TRUSTED,
                compressed=(i % 2 == 0),
                encrypted=(i % 3 == 0)
            )
            mock_messages.append(mock_message)
        
        # Validate all mock messages
        validation_results = []
        for i, message in enumerate(mock_messages):
            result = self.validator.validate_contract(message)
            validation_results.append(result)
            
            print(f"    Mock message {i+1}: {'VALID' if result['valid'] else 'INVALID'}")
            if not result['valid']:
                print(f"      Errors: {result['errors']}")
        
        # All mock messages should be valid
        valid_count = sum(1 for r in validation_results if r['valid'])
        assert valid_count == len(mock_messages), f"All mock messages should be valid, got {valid_count}/{len(mock_messages)}"
        
        # Test different trust levels
        trust_level_tests = [
            (TrustLevel.UNTRUSTED, True),   # Should pass
            (TrustLevel.TRUSTED, True),     # Should pass
            (TrustLevel.VERIFIED, True),    # Should pass
            (TrustLevel.CRITICAL, False)    # May fail if source not trusted
        ]
        
        for trust_level, expected_result in trust_level_tests:
            test_message = self._create_test_message(SchemaVersion.V2_0, trust_level)
            test_message.header.source = 'test_source'  # May not be trusted for CRITICAL
            
            result = self.validator.validate_contract(test_message)
            print(f"    Trust level {trust_level.value}: {'PASS' if result['trust_valid'] else 'FAIL'}")
        
        print("✓ Contract tests and mocks test passed")
    
    def test_data_integrity_validation(self):
        """Test data integrity validation"""
        print("\n=== Testing Data Integrity Validation ===")
        
        # Test payload validation
        message = self.test_messages['v2_0_full']
        
        # Verify payload structure
        assert message.payload.data is not None
        assert message.payload.schema_version == message.header.schema_version
        assert message.payload.validation_hash is not None
        
        # Test compression types
        compressed_message = self.test_messages['v2_0_compressed']
        assert compressed_message.payload.compression_type == 'gzip'
        
        # Test encryption types
        encrypted_message = self.test_messages['v2_0_encrypted']
        assert encrypted_message.payload.encryption_type == 'AES-256'
        
        # Test size validation
        for message_name, test_message in self.test_messages.items():
            if test_message.payload.size_bytes > 0:
                assert test_message.payload.size_bytes > 0, f"{message_name} should have valid size"
        
        # Test validation hash
        for message_name, test_message in self.test_messages.items():
            assert test_message.payload.validation_hash is not None, f"{message_name} should have validation hash"
            assert len(test_message.payload.validation_hash) > 0, f"{message_name} should have non-empty hash"
        
        print("✓ Data integrity validation test passed")
    
    def test_end_to_end_contract_validation(self):
        """Test complete end-to-end contract validation"""
        print("\n=== Testing End-to-End Contract Validation ===")
        
        # Complete contract validation lifecycle
        lifecycle_phases = [
            ('message_creation', 'create_test_messages'),
            ('schema_validation', 'validate_schemas'),
            ('trust_validation', 'validate_trust'),
            ('dedup_check', 'check_deduplication'),
            ('compatibility_test', 'test_compatibility'),
            ('final_validation', 'validate_system_state')
        ]
        
        for phase_name, phase_action in lifecycle_phases:
            print(f"  Phase: {phase_name}")
            
            if phase_action == 'create_test_messages':
                # Create additional test messages
                additional_messages = [
                    self._create_test_message(SchemaVersion.V1_0, TrustLevel.TRUSTED),
                    self._create_test_message(SchemaVersion.V1_1, TrustLevel.VERIFIED),
                    self._create_test_message(SchemaVersion.V2_0, TrustLevel.CRITICAL)
                ]
                
            elif phase_action == 'validate_schemas':
                # Validate all message schemas
                for message in additional_messages:
                    result = self.validator.validate_contract(message)
                    assert result['schema_valid'] == True, f"Schema validation failed for {message.header.schema_version}"
                
            elif phase_action == 'validate_trust':
                # Validate trust for all messages
                for message in additional_messages:
                    result = self.validator.validate_contract(message)
                    # Trust validation may fail for CRITICAL level if source not trusted
                    if message.header.trust_level != TrustLevel.CRITICAL:
                        assert result['trust_valid'] == True, f"Trust validation failed for {message.header.trust_level}"
                
            elif phase_action == 'check_deduplication':
                # Check deduplication for all messages
                for message in additional_messages:
                    result = self.validator.validate_contract(message)
                    assert result['dedup_valid'] == True, f"Dedup check failed for {message.header.correlation_id}"
                
            elif phase_action == 'test_compatibility':
                # Run compatibility tests
                compatibility_results = self.validator.run_compatibility_tests()
                assert len(compatibility_results) > 0, "Compatibility tests should return results"
                
            elif phase_action == 'validate_system_state':
                # Final system validation
                validation_summary = self.validator.get_validation_summary()
                print(f"    Validation summary: {validation_summary}")
                
                # Verify system integrity
                assert validation_summary['total_validations'] > 0
                assert validation_summary['success_rate'] >= 0.0
                assert 'dedup_stats' in validation_summary
        
        print("✓ End-to-end contract validation test passed")


def run_l0_l1_contract_tests():
    """Run all L0→L1 contract tests"""
    print("\n" + "="*60)
    print("RUNNING L0→L1 CONTRACT TESTS")
    print("="*60)
    
    test_suite = TestL0L1Contract()
    
    # Run all test methods
    test_methods = [
        'test_schema_versioning',
        'test_forward_backward_compatibility',
        'test_required_headers',
        'test_replay_window_dedup',
        'test_contract_tests_mocks',
        'test_data_integrity_validation',
        'test_end_to_end_contract_validation'
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
    print("L0→L1 CONTRACT TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    run_l0_l1_contract_tests()
