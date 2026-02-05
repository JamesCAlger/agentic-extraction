"""
Validation module for measuring extraction accuracy against ground truth.

This module provides:
- Schema definitions matching the validation dataset structure
- Validation harness for comparing extractions to ground truth
- Metrics computation for accuracy reporting
"""

from .schemas import (
    ConfidenceLevel,
    ExtractedValue,
    ShareClassExtraction,
    AllocationTarget,
    FundExtraction,
)
from .harness import (
    ValidationResult,
    GroundTruth,
    load_ground_truth,
    validate_extraction,
    validate_fund,
    create_empty_extraction,
    create_mock_extraction,
)
from .metrics import (
    ValidationMetrics,
    compute_metrics,
    print_metrics_report,
)

__all__ = [
    # Schemas
    "ConfidenceLevel",
    "ExtractedValue",
    "ShareClassExtraction",
    "AllocationTarget",
    "FundExtraction",
    # Harness
    "ValidationResult",
    "GroundTruth",
    "load_ground_truth",
    "validate_extraction",
    "validate_fund",
    "create_empty_extraction",
    "create_mock_extraction",
    # Metrics
    "ValidationMetrics",
    "compute_metrics",
    "print_metrics_report",
]
