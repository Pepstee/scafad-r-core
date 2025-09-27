"""SCAFAD Utility Helpers
====================

Core utility functions for SCAFAD Layer 0 operations.
Provides safe data handling, hashing, formatting, and string utilities.
"""

from __future__ import annotations
import json
import hashlib
import time
import re
import os
import base64
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import asdict, is_dataclass
from enum import Enum

# Type aliases for clarity
JSONSerializable = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]


# ==============================================================================
# JSON and Serialization Utilities
# ==============================================================================

def safe_json_parse(json_string: str, default: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Safely parse JSON string with fallback default."""
    if not isinstance(json_string, str):
        return default
    try:
        result = json.loads(json_string)
        return result if isinstance(result, dict) else default
    except (json.JSONDecodeError, ValueError, TypeError):
        return default


def safe_json_dumps(obj: Any, indent: Optional[int] = None, sort_keys: bool = True) -> str:
    """Safely serialize object to JSON with enhanced support for custom types."""
    try:
        return json.dumps(obj, indent=indent, sort_keys=sort_keys, default=_json_serializer, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        # Fallback to string representation
        return json.dumps({"error": "serialization_failed", "repr": str(obj), "type": str(type(obj))})


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for complex types."""
    if is_dataclass(obj):
        return asdict(obj)
    elif isinstance(obj, Enum):
        return obj.value
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    elif isinstance(obj, bytes):
        return base64.b64encode(obj).decode('ascii')
    else:
        return str(obj)


# ==============================================================================
# Hashing and Cryptographic Utilities
# ==============================================================================

def calculate_hash(data: Any, algorithm: str = "sha256", normalize: bool = True) -> str:
    """Calculate hash of data with optional normalization."""
    try:
        hasher = hashlib.new(algorithm)
        
        if normalize:
            # Normalize data for consistent hashing
            if isinstance(data, dict):
                serialized = safe_json_dumps(data, sort_keys=True)
            elif isinstance(data, (list, tuple)):
                serialized = safe_json_dumps(sorted(data) if all(isinstance(x, str) for x in data) else data)
            else:
                serialized = safe_json_dumps(data)
        else:
            serialized = str(data)
            
        hasher.update(serialized.encode('utf-8'))
        return hasher.hexdigest()
    except Exception as e:
        # Fallback hash
        return hashlib.sha256(f"error:{str(e)}:{repr(data)}".encode('utf-8')).hexdigest()


def calculate_payload_hash(payload: Dict[str, Any], exclude_keys: Optional[List[str]] = None) -> str:
    """Calculate hash of payload excluding specified keys."""
    exclude_keys = exclude_keys or ['timestamp', 'correlation_id', 'request_id']
    
    filtered_payload = {
        k: v for k, v in payload.items() 
        if k not in exclude_keys
    }
    
    return calculate_hash(filtered_payload)


def constant_time_compare(a: str, b: str) -> bool:
    """Constant-time string comparison to prevent timing attacks."""
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    if len(a) != len(b):
        return False
        
    result = 0
    for x, y in zip(a.encode('utf-8'), b.encode('utf-8')):
        result |= x ^ y
    return result == 0


# ==============================================================================
# Time and Timestamp Utilities
# ==============================================================================

def format_timestamp(ts: float, format_type: str = "iso") -> str:
    """Format timestamp in various formats."""
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        
        if format_type == "iso":
            return dt.isoformat()
        elif format_type == "human":
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        elif format_type == "compact":
            return dt.strftime("%Y%m%d_%H%M%S")
        elif format_type == "milliseconds":
            return f"{int(ts * 1000)}"
        else:
            return f"{ts:.6f}"
    except (ValueError, OSError, OverflowError):
        return f"{ts:.6f}"


def current_timestamp() -> float:
    """Get current timestamp as float."""
    return time.time()


def timestamp_to_datetime(ts: float) -> datetime:
    """Convert timestamp to datetime object."""
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def calculate_duration(start_ts: float, end_ts: Optional[float] = None) -> float:
    """Calculate duration between timestamps."""
    end_ts = end_ts or current_timestamp()
    return max(0.0, end_ts - start_ts)


# ==============================================================================
# String and Text Processing
# ==============================================================================

def sanitize_string(value: str, max_length: int = 1000, allow_patterns: Optional[List[str]] = None) -> str:
    """Sanitize string for safe logging and processing."""
    if not isinstance(value, str):
        value = str(value)
        
    # Truncate if too long
    if len(value) > max_length:
        value = value[:max_length] + "..."
    
    # Remove control characters but preserve newlines and tabs
    value = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', value)
    
    # Apply allow patterns if specified
    if allow_patterns:
        for pattern in allow_patterns:
            if not re.match(pattern, value):
                return "<sanitized>"
    
    return value


def extract_correlation_id(data: Dict[str, Any], fallback_keys: Optional[List[str]] = None) -> Optional[str]:
    """Extract correlation ID from data using multiple possible keys."""
    keys_to_check = ['correlation_id', 'correlationId', 'request_id', 'requestId', 'trace_id', 'traceId']
    
    if fallback_keys:
        keys_to_check.extend(fallback_keys)
    
    for key in keys_to_check:
        if key in data and data[key]:
            return sanitize_string(str(data[key]), max_length=100)
    
    return None


def generate_unique_id(prefix: str = "scafad", include_timestamp: bool = True) -> str:
    """Generate a unique ID with optional prefix and timestamp."""
    if include_timestamp:
        ts = format_timestamp(current_timestamp(), "compact")
        random_part = os.urandom(4).hex()
        return f"{prefix}_{ts}_{random_part}"
    else:
        random_part = os.urandom(8).hex()
        return f"{prefix}_{random_part}"


# ==============================================================================
# Data Structure Utilities
# ==============================================================================

def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(data: Dict[str, Any], separator: str = ".", prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dictionary structure."""
    result = {}
    
    for key, value in data.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            result.update(flatten_dict(value, separator, new_key))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    result.update(flatten_dict(item, separator, f"{new_key}[{i}]"))
                else:
                    result[f"{new_key}[{i}]"] = item
        else:
            result[new_key] = value
    
    return result


def safe_get_nested(data: Dict[str, Any], keys: Union[str, List[str]], default: Any = None) -> Any:
    """Safely get nested value from dictionary."""
    if isinstance(keys, str):
        keys = keys.split('.')
    
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def filter_dict_keys(data: Dict[str, Any], allowed_keys: List[str], include_nested: bool = False) -> Dict[str, Any]:
    """Filter dictionary to only include allowed keys."""
    result = {}
    
    for key in allowed_keys:
        if key in data:
            value = data[key]
            if include_nested and isinstance(value, dict):
                # For nested dicts, recursively filter if allowed keys contains patterns
                nested_allowed = [k.split('.', 1)[1] for k in allowed_keys if '.' in k and k.startswith(f"{key}.")]
                if nested_allowed:
                    result[key] = filter_dict_keys(value, nested_allowed, include_nested)
                else:
                    result[key] = value
            else:
                result[key] = value
    
    return result


# ==============================================================================
# Error and Exception Utilities
# ==============================================================================

def safe_execute(func, *args, default_result=None, log_errors: bool = False, **kwargs) -> Tuple[Any, Optional[Exception]]:
    """Execute function safely, returning result and any exception."""
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        if log_errors:
            import logging
            logging.getLogger(__name__).warning(f"Safe execution failed: {e}")
        return default_result, e


def format_exception(exception: Exception, include_traceback: bool = False) -> Dict[str, Any]:
    """Format exception for logging or serialization."""
    result = {
        'type': type(exception).__name__,
        'message': str(exception),
        'timestamp': current_timestamp()
    }
    
    if include_traceback:
        import traceback
        result['traceback'] = traceback.format_exc()
    
    return result


# ==============================================================================
# Validation and Type Checking
# ==============================================================================

def is_valid_json(data: str) -> bool:
    """Check if string is valid JSON."""
    try:
        json.loads(data)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def validate_required_keys(data: Dict[str, Any], required_keys: List[str]) -> Tuple[bool, List[str]]:
    """Validate that all required keys are present in dictionary."""
    missing_keys = [key for key in required_keys if key not in data or data[key] is None]
    return len(missing_keys) == 0, missing_keys


def normalize_anomaly_type(anomaly_type: Any) -> str:
    """Normalize anomaly type to standard string representation."""
    if hasattr(anomaly_type, 'value'):  # Enum
        return str(anomaly_type.value).upper()
    return str(anomaly_type).upper().replace(' ', '_')


# ==============================================================================
# Module exports
# ==============================================================================

__all__ = [
    # JSON utilities
    'safe_json_parse', 'safe_json_dumps',
    # Hashing
    'calculate_hash', 'calculate_payload_hash', 'constant_time_compare',
    # Time utilities
    'format_timestamp', 'current_timestamp', 'timestamp_to_datetime', 'calculate_duration',
    # String utilities
    'sanitize_string', 'extract_correlation_id', 'generate_unique_id',
    # Data structure utilities
    'deep_merge_dicts', 'flatten_dict', 'safe_get_nested', 'filter_dict_keys',
    # Error utilities
    'safe_execute', 'format_exception',
    # Validation utilities
    'is_valid_json', 'validate_required_keys', 'normalize_anomaly_type'
]
