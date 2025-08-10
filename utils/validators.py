"""SCAFAD Input Validation and Security
====================================

Comprehensive input validation, sanitization, and security checks for SCAFAD Layer 0.
Provides protection against common attack vectors and ensures data integrity.
"""

from __future__ import annotations
import re
import json
import base64
import ipaddress
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from urllib.parse import urlparse
from dataclasses import dataclass
from enum import Enum

# Security patterns and constants
SUSPICIOUS_PATTERNS = [
    r'<script[^>]*>',  # Script injection
    r'javascript:',     # JavaScript protocol
    r'vbscript:',      # VBScript protocol
    r'data:text/html', # Data URI with HTML
    r'eval\s*\(',      # Eval calls
    r'exec\s*\(',      # Exec calls
    r'import\s+os',    # OS imports
    r'__import__',     # Dynamic imports
    r'subprocess',     # Subprocess calls
    r'shell=True',     # Shell execution
]

DANGEROUS_EXTENSIONS = {
    '.exe', '.bat', '.cmd', '.com', '.scr', '.pif', '.vbs', '.js',
    '.jar', '.app', '.deb', '.rpm', '.dmg', '.pkg', '.msi'
}

MAX_STRING_LENGTH = 10000
MAX_DICT_KEYS = 1000
MAX_LIST_LENGTH = 1000
MAX_NESTING_DEPTH = 10


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    field_path: Optional[str] = None
    sanitized_value: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'severity': self.severity.value,
            'message': self.message,
            'field_path': self.field_path,
            'has_sanitized_value': self.sanitized_value is not None
        }


class InputValidator:
    """Comprehensive input validation and sanitization"""
    
    def __init__(self, 
                 max_string_length: int = MAX_STRING_LENGTH,
                 max_dict_keys: int = MAX_DICT_KEYS,
                 max_list_length: int = MAX_LIST_LENGTH,
                 max_nesting_depth: int = MAX_NESTING_DEPTH):
        self.max_string_length = max_string_length
        self.max_dict_keys = max_dict_keys
        self.max_list_length = max_list_length
        self.max_nesting_depth = max_nesting_depth
        
        # Compile regex patterns for performance
        self._suspicious_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in SUSPICIOUS_PATTERNS]
    
    def validate_and_sanitize(self, data: Any, field_path: str = "root") -> Tuple[Any, List[ValidationResult]]:
        """Validate and sanitize input data recursively"""
        issues = []
        sanitized = self._validate_recursive(data, field_path, 0, issues)
        return sanitized, issues
    
    def _validate_recursive(self, data: Any, field_path: str, depth: int, issues: List[ValidationResult]) -> Any:
        """Recursive validation and sanitization"""
        # Check nesting depth
        if depth > self.max_nesting_depth:
            issues.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Maximum nesting depth ({self.max_nesting_depth}) exceeded",
                field_path=field_path
            ))
            return None
        
        # Handle different data types
        if isinstance(data, dict):
            return self._validate_dict(data, field_path, depth, issues)
        elif isinstance(data, list):
            return self._validate_list(data, field_path, depth, issues)
        elif isinstance(data, str):
            return self._validate_string(data, field_path, issues)
        elif isinstance(data, (int, float)):
            return self._validate_number(data, field_path, issues)
        elif isinstance(data, bool):
            return data  # Booleans are safe
        elif data is None:
            return None  # None is safe
        else:
            # Unknown type - convert to string and validate
            str_data = str(data)
            issues.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message=f"Unknown type {type(data).__name__} converted to string",
                field_path=field_path
            ))
            return self._validate_string(str_data, field_path, issues)
    
    def _validate_dict(self, data: Dict[str, Any], field_path: str, depth: int, issues: List[ValidationResult]) -> Dict[str, Any]:
        """Validate dictionary"""
        if len(data) > self.max_dict_keys:
            issues.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Dictionary has too many keys ({len(data)} > {self.max_dict_keys})",
                field_path=field_path
            ))
            # Truncate to maximum allowed keys
            data = dict(list(data.items())[:self.max_dict_keys])
        
        sanitized = {}
        for key, value in data.items():
            # Validate key
            sanitized_key = self._validate_string(str(key), f"{field_path}.{key}", issues)
            if sanitized_key is None:
                continue
                
            # Validate value
            sanitized_value = self._validate_recursive(value, f"{field_path}.{key}", depth + 1, issues)
            sanitized[sanitized_key] = sanitized_value
        
        return sanitized
    
    def _validate_list(self, data: List[Any], field_path: str, depth: int, issues: List[ValidationResult]) -> List[Any]:
        """Validate list"""
        if len(data) > self.max_list_length:
            issues.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"List has too many items ({len(data)} > {self.max_list_length})",
                field_path=field_path
            ))
            # Truncate to maximum allowed length
            data = data[:self.max_list_length]
        
        sanitized = []
        for i, item in enumerate(data):
            sanitized_item = self._validate_recursive(item, f"{field_path}[{i}]", depth + 1, issues)
            sanitized.append(sanitized_item)
        
        return sanitized
    
    def _validate_string(self, data: str, field_path: str, issues: List[ValidationResult]) -> Optional[str]:
        """Validate and sanitize string"""
        if not isinstance(data, str):
            data = str(data)
        
        # Check length
        if len(data) > self.max_string_length:
            issues.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"String truncated from {len(data)} to {self.max_string_length} characters",
                field_path=field_path
            ))
            data = data[:self.max_string_length]
        
        # Check for suspicious patterns
        for pattern in self._suspicious_patterns:
            if pattern.search(data):
                issues.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Suspicious pattern detected: {pattern.pattern}",
                    field_path=field_path
                ))
                # Remove or sanitize the suspicious content
                data = pattern.sub('[REMOVED]', data)
        
        # Remove control characters except newlines and tabs
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', data)
        
        if sanitized != data:
            issues.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message="Control characters removed",
                field_path=field_path
            ))
        
        return sanitized
    
    def _validate_number(self, data: Union[int, float], field_path: str, issues: List[ValidationResult]) -> Union[int, float]:
        """Validate numeric data"""
        # Check for special float values
        if isinstance(data, float):
            if not (float('-inf') < data < float('inf')):
                issues.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid float value: {data}",
                    field_path=field_path
                ))
                return 0.0
            
            import math
            if math.isnan(data):
                issues.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="NaN value detected",
                    field_path=field_path
                ))
                return 0.0
        
        # Check reasonable bounds
        if isinstance(data, int) and abs(data) > 2**53:  # JavaScript safe integer limit
            issues.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Large integer value: {data}",
                field_path=field_path
            ))
        
        return data


def validate_aws_lambda_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize AWS Lambda event"""
    if not isinstance(event, dict):
        raise ValueError("Event must be a dictionary")
    
    validator = InputValidator()
    sanitized_event, issues = validator.validate_and_sanitize(event, "event")
    
    # Check for critical issues
    critical_issues = [issue for issue in issues if issue.severity == ValidationSeverity.CRITICAL]
    if critical_issues:
        raise SecurityError(f"Critical security issues detected: {[issue.message for issue in critical_issues]}")
    
    # Ensure required Lambda fields are present
    if 'Records' in event and not isinstance(event.get('Records'), list):
        raise ValueError("Records field must be a list if present")
    
    return sanitized_event


def sanitize_user_input(input_data: Any, strict_mode: bool = False) -> Tuple[Any, List[ValidationResult]]:
    """Sanitize user input data with comprehensive validation"""
    validator = InputValidator()
    
    if strict_mode:
        # In strict mode, use more restrictive limits
        validator.max_string_length = 1000
        validator.max_dict_keys = 100
        validator.max_list_length = 100
        validator.max_nesting_depth = 5
    
    return validator.validate_and_sanitize(input_data)


def check_payload_safety(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Comprehensive payload safety checks"""
    issues = []
    
    if not isinstance(payload, dict):
        return False, ["Payload must be a dictionary"]
    
    # Size checks
    try:
        payload_str = json.dumps(payload)
        if len(payload_str) > 1024 * 1024:  # 1MB limit
            issues.append("Payload size exceeds 1MB limit")
    except Exception as e:
        issues.append(f"Payload serialization failed: {e}")
    
    # Check for suspicious keys
    suspicious_keys = ['__proto__', 'constructor', 'prototype', 'eval', 'exec']
    for key in payload:
        if str(key).lower() in suspicious_keys:
            issues.append(f"Suspicious key detected: {key}")
    
    # Check for file paths or URLs that might be dangerous
    def check_value_safety(value: Any, path: str = "") -> None:
        if isinstance(value, str):
            # Check for file extensions
            for ext in DANGEROUS_EXTENSIONS:
                if ext in value.lower():
                    issues.append(f"Potentially dangerous file extension in {path}: {ext}")
            
            # Check for URL schemes
            try:
                parsed = urlparse(value)
                if parsed.scheme in ['file', 'ftp', 'sftp', 'smb']:
                    issues.append(f"Potentially dangerous URL scheme in {path}: {parsed.scheme}")
            except Exception:
                pass
                
        elif isinstance(value, dict):
            for k, v in value.items():
                check_value_safety(v, f"{path}.{k}" if path else str(k))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                check_value_safety(item, f"{path}[{i}]" if path else f"[{i}]")
    
    check_value_safety(payload)
    
    return len(issues) == 0, issues


def validate_ip_address(ip_str: str) -> Tuple[bool, Optional[str]]:
    """Validate IP address and return type (IPv4/IPv6)"""
    try:
        ip = ipaddress.ip_address(ip_str)
        if ip.is_private:
            return True, f"IPv{ip.version} (private)"
        elif ip.is_loopback:
            return True, f"IPv{ip.version} (loopback)"
        elif ip.is_multicast:
            return True, f"IPv{ip.version} (multicast)"
        else:
            return True, f"IPv{ip.version} (public)"
    except ValueError:
        return False, None


def validate_email(email: str) -> bool:
    """Basic email validation"""
    if not isinstance(email, str) or len(email) > 254:
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_url(url: str, allowed_schemes: Optional[Set[str]] = None) -> Tuple[bool, Optional[str]]:
    """Validate URL with optional scheme restrictions"""
    if not isinstance(url, str) or len(url) > 2048:
        return False, "Invalid URL format or too long"
    
    try:
        parsed = urlparse(url)
        
        if not parsed.scheme:
            return False, "Missing URL scheme"
        
        if not parsed.netloc:
            return False, "Missing network location"
        
        allowed_schemes = allowed_schemes or {'http', 'https'}
        if parsed.scheme.lower() not in allowed_schemes:
            return False, f"Disallowed scheme: {parsed.scheme}"
        
        return True, None
    except Exception as e:
        return False, f"URL parsing error: {e}"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    if not isinstance(filename, str):
        filename = str(filename)
    
    # Remove path separators and dangerous characters
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Ensure not empty
    if not filename:
        filename = 'unnamed_file'
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        max_name_length = 250 - len(ext)
        filename = name[:max_name_length] + ('.' + ext if ext else '')
    
    return filename


def validate_base64(data: str) -> Tuple[bool, Optional[bytes]]:
    """Validate and decode base64 data"""
    try:
        # Add padding if needed
        missing_padding = len(data) % 4
        if missing_padding:
            data += '=' * (4 - missing_padding)
        
        decoded = base64.b64decode(data, validate=True)
        return True, decoded
    except Exception:
        return False, None


class SecurityError(Exception):
    """Exception raised for security-related validation failures"""
    pass


class ValidationError(Exception):
    """Exception raised for general validation failures"""
    def __init__(self, message: str, issues: List[ValidationResult]):
        super().__init__(message)
        self.issues = issues


# Predefined validators for common use cases
class CommonValidators:
    """Collection of commonly used validators"""
    
    @staticmethod
    def validate_scafad_telemetry(telemetry_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate SCAFAD telemetry record structure"""
        issues = []
        
        required_fields = ['event_id', 'timestamp', 'function_id']
        for field in required_fields:
            if field not in telemetry_data:
                issues.append(f"Missing required field: {field}")
        
        # Validate timestamp
        if 'timestamp' in telemetry_data:
            try:
                ts = float(telemetry_data['timestamp'])
                if ts <= 0 or ts > 9999999999:  # Reasonable timestamp bounds
                    issues.append("Timestamp out of reasonable range")
            except (ValueError, TypeError):
                issues.append("Invalid timestamp format")
        
        # Validate event_id format
        if 'event_id' in telemetry_data:
            event_id = telemetry_data['event_id']
            if not isinstance(event_id, str) or len(event_id) > 100:
                issues.append("Invalid event_id format")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_anomaly_score(score: Any) -> Tuple[bool, Optional[float]]:
        """Validate anomaly score is within valid range"""
        try:
            score_float = float(score)
            if 0.0 <= score_float <= 1.0:
                return True, score_float
            else:
                return False, None
        except (ValueError, TypeError):
            return False, None


# Module exports
__all__ = [
    'ValidationSeverity', 'ValidationResult', 'InputValidator', 'SecurityError', 'ValidationError',
    'validate_aws_lambda_event', 'sanitize_user_input', 'check_payload_safety',
    'validate_ip_address', 'validate_email', 'validate_url', 'sanitize_filename', 'validate_base64',
    'CommonValidators'
]
