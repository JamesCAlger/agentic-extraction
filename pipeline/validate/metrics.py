"""
Metrics computation for extraction validation.

This module provides:
- Accuracy metrics (overall, by field, by category, by confidence)
- Error analysis
- Reporting utilities
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .harness import ValidationResult, ErrorType
from .schemas import ConfidenceLevel


@dataclass
class CategoryMetrics:
    """Metrics for a category of fields."""
    category: str
    total: int = 0
    correct: int = 0
    wrong_value: int = 0
    missed: int = 0
    hallucinated: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return 1.0 - self.accuracy


@dataclass
class ConfidenceMetrics:
    """Metrics broken down by confidence level."""
    confidence: ConfidenceLevel
    total: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


@dataclass
class FieldMetrics:
    """Metrics for a single field across funds."""
    field_name: str
    category: str
    total: int = 0
    correct: int = 0
    error_type_counts: dict[str, int] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


@dataclass
class ValidationMetrics:
    """Complete validation metrics."""
    # Overall
    total_fields: int = 0
    correct_fields: int = 0

    # By error type
    wrong_value_count: int = 0
    missed_count: int = 0
    hallucinated_count: int = 0

    # By category
    category_metrics: dict[str, CategoryMetrics] = field(default_factory=dict)

    # By confidence
    confidence_metrics: dict[str, ConfidenceMetrics] = field(default_factory=dict)

    # By field (for identifying worst performers)
    field_metrics: dict[str, FieldMetrics] = field(default_factory=dict)

    # Detailed results
    all_results: list[ValidationResult] = field(default_factory=list)
    error_results: list[ValidationResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Overall accuracy."""
        return self.correct_fields / self.total_fields if self.total_fields > 0 else 0.0

    @property
    def error_count(self) -> int:
        """Total number of errors."""
        return self.wrong_value_count + self.missed_count + self.hallucinated_count

    def get_worst_fields(self, n: int = 10) -> list[tuple[str, float]]:
        """Get the n worst performing fields by accuracy."""
        field_accuracies = [
            (name, metrics.accuracy)
            for name, metrics in self.field_metrics.items()
            if metrics.total > 0
        ]
        field_accuracies.sort(key=lambda x: x[1])
        return field_accuracies[:n]

    def get_best_fields(self, n: int = 10) -> list[tuple[str, float]]:
        """Get the n best performing fields by accuracy."""
        field_accuracies = [
            (name, metrics.accuracy)
            for name, metrics in self.field_metrics.items()
            if metrics.total > 0
        ]
        field_accuracies.sort(key=lambda x: x[1], reverse=True)
        return field_accuracies[:n]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "overall": {
                "total_fields": self.total_fields,
                "correct_fields": self.correct_fields,
                "accuracy": self.accuracy,
            },
            "errors_by_type": {
                "wrong_value": self.wrong_value_count,
                "missed": self.missed_count,
                "hallucinated": self.hallucinated_count,
            },
            "by_category": {
                name: {
                    "total": m.total,
                    "correct": m.correct,
                    "accuracy": m.accuracy,
                }
                for name, m in self.category_metrics.items()
            },
            "by_confidence": {
                name: {
                    "total": m.total,
                    "correct": m.correct,
                    "accuracy": m.accuracy,
                }
                for name, m in self.confidence_metrics.items()
            },
            "worst_fields": self.get_worst_fields(10),
        }


def compute_metrics(results: list[ValidationResult]) -> ValidationMetrics:
    """
    Compute comprehensive metrics from validation results.

    Args:
        results: List of ValidationResult from validate_fund()

    Returns:
        ValidationMetrics with all computed metrics
    """
    metrics = ValidationMetrics()
    metrics.all_results = results

    # Initialize confidence metrics
    for conf in ConfidenceLevel:
        metrics.confidence_metrics[conf.value] = ConfidenceMetrics(confidence=conf)

    for result in results:
        metrics.total_fields += 1

        # Overall accuracy
        if result.correct:
            metrics.correct_fields += 1
        else:
            metrics.error_results.append(result)

        # By error type
        if result.error_type == ErrorType.WRONG_VALUE:
            metrics.wrong_value_count += 1
        elif result.error_type == ErrorType.MISSED:
            metrics.missed_count += 1
        elif result.error_type == ErrorType.HALLUCINATED:
            metrics.hallucinated_count += 1

        # By category
        category = result.category or "uncategorized"
        if category not in metrics.category_metrics:
            metrics.category_metrics[category] = CategoryMetrics(category=category)

        cat_metrics = metrics.category_metrics[category]
        cat_metrics.total += 1
        if result.correct:
            cat_metrics.correct += 1
        elif result.error_type == ErrorType.WRONG_VALUE:
            cat_metrics.wrong_value += 1
        elif result.error_type == ErrorType.MISSED:
            cat_metrics.missed += 1
        elif result.error_type == ErrorType.HALLUCINATED:
            cat_metrics.hallucinated += 1

        # By confidence
        conf_key = result.confidence.value
        if conf_key in metrics.confidence_metrics:
            conf_metrics = metrics.confidence_metrics[conf_key]
            conf_metrics.total += 1
            if result.correct:
                conf_metrics.correct += 1

        # By field
        field_name = result.field_name
        if field_name not in metrics.field_metrics:
            metrics.field_metrics[field_name] = FieldMetrics(
                field_name=field_name,
                category=category,
            )

        field_metrics = metrics.field_metrics[field_name]
        field_metrics.total += 1
        if result.correct:
            field_metrics.correct += 1

        error_type_key = result.error_type.value
        if error_type_key not in field_metrics.error_type_counts:
            field_metrics.error_type_counts[error_type_key] = 0
        field_metrics.error_type_counts[error_type_key] += 1

    return metrics


def compute_aggregate_metrics(
    all_results: list[list[ValidationResult]]
) -> ValidationMetrics:
    """
    Compute aggregate metrics across multiple funds.

    Args:
        all_results: List of results from multiple funds

    Returns:
        Aggregated ValidationMetrics
    """
    # Flatten all results
    flattened = []
    for fund_results in all_results:
        flattened.extend(fund_results)

    return compute_metrics(flattened)


def print_metrics_report(
    metrics: ValidationMetrics,
    title: str = "EXTRACTION VALIDATION REPORT",
    show_all_errors: bool = False,
) -> str:
    """
    Generate a formatted metrics report.

    Args:
        metrics: ValidationMetrics to report
        title: Report title
        show_all_errors: Whether to show all error details

    Returns:
        Formatted report string
    """
    lines = []
    sep = "=" * 60

    lines.append(sep)
    lines.append(f" {title}")
    lines.append(sep)
    lines.append("")

    # Overall accuracy
    lines.append("OVERALL ACCURACY")
    lines.append("-" * 40)
    lines.append(f"  Total fields:   {metrics.total_fields}")
    lines.append(f"  Correct:        {metrics.correct_fields}")
    lines.append(f"  Accuracy:       {metrics.accuracy:.1%}")
    lines.append("")

    # Errors by type
    lines.append("ERRORS BY TYPE")
    lines.append("-" * 40)
    lines.append(f"  Wrong value:    {metrics.wrong_value_count}")
    lines.append(f"  Missed:         {metrics.missed_count}")
    lines.append(f"  Hallucinated:   {metrics.hallucinated_count}")
    lines.append("")

    # By confidence
    lines.append("ACCURACY BY CONFIDENCE LEVEL")
    lines.append("-" * 40)
    for conf_name, conf_metrics in sorted(metrics.confidence_metrics.items()):
        if conf_metrics.total > 0:
            lines.append(
                f"  {conf_name:15} {conf_metrics.correct:3}/{conf_metrics.total:3} "
                f"({conf_metrics.accuracy:.1%})"
            )
    lines.append("")

    # By category
    lines.append("ACCURACY BY CATEGORY")
    lines.append("-" * 40)
    for cat_name, cat_metrics in sorted(
        metrics.category_metrics.items(),
        key=lambda x: x[1].accuracy
    ):
        if cat_metrics.total > 0:
            lines.append(
                f"  {cat_name:30} {cat_metrics.correct:3}/{cat_metrics.total:3} "
                f"({cat_metrics.accuracy:.1%})"
            )
    lines.append("")

    # Worst performing fields
    worst_fields = metrics.get_worst_fields(10)
    if worst_fields:
        lines.append("WORST PERFORMING FIELDS")
        lines.append("-" * 40)
        for field_name, accuracy in worst_fields:
            field_m = metrics.field_metrics[field_name]
            lines.append(
                f"  {field_name:40} {field_m.correct}/{field_m.total} ({accuracy:.1%})"
            )
        lines.append("")

    # Error details
    if show_all_errors and metrics.error_results:
        lines.append("ERROR DETAILS")
        lines.append("-" * 40)
        for result in metrics.error_results[:20]:  # Limit to first 20
            lines.append(f"  Field: {result.field_name}")
            lines.append(f"    Expected:  {result.expected}")
            lines.append(f"    Extracted: {result.extracted}")
            lines.append(f"    Error:     {result.error_type.value}")
            lines.append("")

        if len(metrics.error_results) > 20:
            lines.append(f"  ... and {len(metrics.error_results) - 20} more errors")
            lines.append("")

    lines.append(sep)

    return "\n".join(lines)


def export_errors_to_csv(
    metrics: ValidationMetrics,
    output_path: str,
) -> None:
    """
    Export error details to CSV for analysis.

    Args:
        metrics: ValidationMetrics with error results
        output_path: Path to output CSV file
    """
    import csv

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'field_name', 'category', 'expected', 'extracted',
            'error_type', 'confidence', 'citation'
        ])
        writer.writeheader()

        for result in metrics.error_results:
            writer.writerow(result.to_dict())


def compare_metrics(
    baseline: ValidationMetrics,
    current: ValidationMetrics,
) -> dict:
    """
    Compare two sets of metrics to measure improvement.

    Args:
        baseline: Previous metrics
        current: Current metrics

    Returns:
        Dictionary with comparison results
    """
    return {
        "accuracy_delta": current.accuracy - baseline.accuracy,
        "accuracy_pct_change": (
            (current.accuracy - baseline.accuracy) / baseline.accuracy * 100
            if baseline.accuracy > 0 else 0
        ),
        "errors_reduced": baseline.error_count - current.error_count,
        "baseline_accuracy": baseline.accuracy,
        "current_accuracy": current.accuracy,
        "category_deltas": {
            cat: current.category_metrics.get(cat, CategoryMetrics(cat)).accuracy
                 - baseline.category_metrics.get(cat, CategoryMetrics(cat)).accuracy
            for cat in set(baseline.category_metrics.keys()) | set(current.category_metrics.keys())
        }
    }
