# utils/helpers.py
from __future__ import annotations
import json
import hashlib
from typing import Any, Dict, Optional

def safe_json_parse(json_string: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(json_string)
    except Exception:
        return None

def calculate_hash(data: Any, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    h.update(repr(data).encode("utf-8"))
    return h.hexdigest()

def format_timestamp(ts: float, format_type: str = "iso") -> str:
    # Placeholder formatting; integrate datetime if desired
    return f"{ts:.6f}"
