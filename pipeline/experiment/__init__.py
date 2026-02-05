"""
Extraction Experimentation Framework.

This module provides systematic A/B testing of extraction configurations:
- Config loading with base + override merging
- Experiment runner with full tracing
- Ground truth evaluation (precision/recall/F1)
- Comparison reports between experiments

Usage:
    python -m pipeline.experiment run --config configs/experiments/per_section.yaml
    python -m pipeline.experiment evaluate --run exp_20260109_baseline
    python -m pipeline.experiment compare --baseline exp_1 --variant exp_2
"""

from .config import ExperimentConfig, load_config, merge_configs
from .runner import ExperimentRunner, ExperimentRun
from .evaluator import (
    GroundTruth,
    load_ground_truth,
    evaluate_extraction,
    EvaluationResult,
)

__all__ = [
    "ExperimentConfig",
    "load_config",
    "merge_configs",
    "ExperimentRunner",
    "ExperimentRun",
    "GroundTruth",
    "load_ground_truth",
    "evaluate_extraction",
    "EvaluationResult",
]
