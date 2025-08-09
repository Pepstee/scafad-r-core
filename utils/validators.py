# utils/validators.py
from __future__ import annotations
from typing import Any, Dict

def validate_aws_lambda_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal event validator; expand with strict schema checks.
    """
    if not isinstance(event, dict):
        raise ValueError("Event must be a dict")
    return event

def sanitize_user_input(input_data: Any) -> Any:
    """
    Strip dangerous fields / enforce basic types.
    """
    return input_data  # TODO: implement

def check_payload_safety(payload: Dict[str, Any]) -> bool:
    """
    Basic security checks to prevent obvious injection vectors.
    """
    return True  # TODO: implement
