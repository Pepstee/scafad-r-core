"""
layer1 — SCAFAD Behavioural Intake Zone
========================================

Validates, sanitises, applies privacy controls, and hashes incoming
telemetry. Produces a Layer1ProcessedRecord for downstream layers.

Public API
----------
  Layer1CanonicalPipeline   — top-level L1 orchestrator
  Layer1ProcessedRecord     — output dataclass
  InputValidationGateway    — WP-3.1
  PreservationAssessment    — WP-3.2
  PrivacyComplianceFilter   — WP-3.3
  SanitisationProcessor     — WP-3.4
  DeferredHashingManager    — WP-3.5
"""

from .pipeline import Layer1CanonicalPipeline, Layer1ProcessedRecord
from .validation import InputValidationGateway
from .preservation import PreservationAssessment, CRITICAL_FIELDS
from .privacy import PrivacyComplianceFilter
from .sanitisation import SanitisationProcessor
from .hashing import DeferredHashingManager

__all__ = [
    "Layer1CanonicalPipeline",
    "Layer1ProcessedRecord",
    "InputValidationGateway",
    "PreservationAssessment",
    "CRITICAL_FIELDS",
    "PrivacyComplianceFilter",
    "SanitisationProcessor",
    "DeferredHashingManager",
]
