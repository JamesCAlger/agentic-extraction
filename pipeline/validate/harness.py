"""
Validation harness for comparing extractions against ground truth.

This module provides:
- Loading ground truth from Excel validation dataset
- Comparing extractions to expected values
- Categorizing errors by type
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from .schemas import (
    ConfidenceLevel,
    ExtractedValue,
    ShareClassExtraction,
    AllocationTarget,
    FundExtraction,
)


class ErrorType(str, Enum):
    """Type of extraction error."""
    CORRECT = "correct"           # Extraction matches expected
    WRONG_VALUE = "wrong_value"   # Extracted different value than expected
    MISSED = "missed"             # Expected value exists but extraction returned None
    HALLUCINATED = "hallucinated" # Extraction returned value when expected is None
    TYPE_MISMATCH = "type_mismatch"  # Value type doesn't match


@dataclass
class ValidationResult:
    """Result of validating a single field extraction."""
    field_name: str
    category: str
    expected: Any
    extracted: Any
    correct: bool
    error_type: ErrorType
    confidence: ConfidenceLevel
    citation: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for reporting."""
        return {
            "field_name": self.field_name,
            "category": self.category,
            "expected": self.expected,
            "extracted": self.extracted,
            "correct": self.correct,
            "error_type": self.error_type.value,
            "confidence": self.confidence.value,
            "citation": self.citation,
            "notes": self.notes,
        }


@dataclass
class GroundTruth:
    """Ground truth data loaded from validation dataset."""
    fund_name: str
    cik: str
    datapoints: dict[str, Any] = field(default_factory=dict)
    categories: dict[str, str] = field(default_factory=dict)  # field -> category
    notes: dict[str, str] = field(default_factory=dict)  # field -> notes


# ─── Field Name Mapping ───
# Maps validation set column names to FundExtraction field names
FIELD_NAME_MAP = {
    # Fund data
    "fund name": "fund_name",
    "fund manager": "fund_manager",
    "sponsor": "sponsor",
    "fund cik": "fund_cik",
    "fund currency": "fund_currency",
    "fund type": "fund_type",
    "fiscal year end": "fiscal_year_end",
    "number of share types": "number_of_share_types",
    "fund target investment type": "fund_target_investment_type",
    "fund target asset class": "fund_target_asset_class",
    "fund target asset classes, as defined by gp": "fund_target_asset_classes_gp",
    "fund target geography": "fund_target_geography",
    "target commitments invested": "target_commitments_invested",
    "target commitments invested incl subscription lines": "target_commitments_invested_incl_sub",
    "maximum fund-level leverage %": "max_fund_leverage_pct",
    "maximum fund-level leverage basis": "max_fund_leverage_basis",

    # Fees and expenses
    "management fee %": "management_fee_pct",
    "management fee basis": "management_fee_basis",
    "estimated subscription interest payments": "estimated_subscription_interest",
    "estimated acquired fund fees and expenses": "estimated_affe",
    "estimated other expenses": "estimated_other_expenses",

    # Repurchase and distributions
    "repurchase %": "repurchase_pct",
    "maximum repurchase %": "max_repurchase_pct",
    "maximum repurchase basis": "repurchase_basis",
    "repurchase frequency": "repurchase_frequency",
    "lock-up period (years)": "lockup_period_years",
    "lock-up maximum early repurchase deduction": "lockup_early_repurchase_deduction",
    "lock-up early repurchase deduction": "lockup_early_repurchase_deduction",
    "mandatory repurchase minimum": "mandatory_repurchase_minimum",
    "target distribution frequency": "target_distribution_frequency",
    "default distribution policy": "default_distribution_policy",

    # Share class fields
    "share name": "share_name",
    "offering price": "offering_price",
    "fee at commitment (sales load) %": "sales_load_percent",
    "distribution / service fee %": "distribution_fee_percent",
    "minimum initial investment": "minimum_initial",
    "minimum follow-on investment": "minimum_followon",
    "repurchase threshold": "repurchase_threshold",

    # Allocation targets
    "allocation target minimum": "target_min",
    "allocation target maximum": "target_max",
    "allocation target - single point": "target_single",

    # Concentration limits
    "maximum allocation to single asset": "max_single_asset",
    "maximum allocation to single fund": "max_single_fund",
    "maximum allocation to single manager": "max_single_manager",
    "maximum allocation to single country": "max_single_country",
    "maximum allocation to single sector": "max_single_sector",
    "derivatives": "derivatives_allowed",
    "expected derivative use": "expected_derivative_use",
}


def normalize_field_name(name: str) -> str:
    """Normalize a field name for lookup."""
    if pd.isna(name):
        return ""
    normalized = str(name).lower().strip()
    return FIELD_NAME_MAP.get(normalized, normalized.replace(" ", "_").replace("-", "_"))


def normalize_value(value: Any) -> Any:
    """Normalize a value for comparison."""
    if pd.isna(value):
        return None
    if isinstance(value, str):
        v_lower = value.lower().strip()
        if v_lower in ('none', 'n/a', 'not applicable', '', 'nan'):
            return None
        # Try to parse as number
        try:
            if '.' in value:
                return float(value.replace(',', ''))
            return int(value.replace(',', ''))
        except ValueError:
            return value.strip()
    return value


def load_ground_truth(
    excel_path: Union[str, Path],
    sheet_name: str
) -> GroundTruth:
    """
    Load ground truth from validation Excel file.

    Args:
        excel_path: Path to datapoints to extract.xlsx
        sheet_name: Sheet name ('stepstone' or 'blackstone')

    Returns:
        GroundTruth with all datapoints
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Determine value column name (varies by sheet)
    value_col = None
    for col in df.columns:
        if col.lower() not in ('data category', 'datapoint', 'notes'):
            value_col = col
            break

    if value_col is None:
        raise ValueError(f"Could not find value column in sheet {sheet_name}")

    ground_truth = GroundTruth(
        fund_name=sheet_name.title(),
        cik="",
    )

    current_share_class = None
    current_allocation_category = None
    current_allocation_type = None  # 'investment_type', 'asset_class', 'geography'

    for _, row in df.iterrows():
        category = str(row.get('data category', '')).strip() if pd.notna(row.get('data category')) else ''
        datapoint = str(row.get('datapoint', '')).strip() if pd.notna(row.get('datapoint')) else ''
        value = row.get(value_col)
        notes = row.get('notes') if pd.notna(row.get('notes')) else None

        if not datapoint:
            continue

        # Normalize the field name
        field_name = normalize_field_name(datapoint)
        normalized_value = normalize_value(value)

        # Track category for each field
        ground_truth.categories[field_name] = category

        # Track notes
        if notes:
            ground_truth.notes[field_name] = str(notes)

        # Handle share class sections
        if category.lower().startswith('share type'):
            # Extract share class number
            match = re.search(r'share type (\d+)', category.lower())
            if match:
                class_num = int(match.group(1))
                class_key = f"share_class_{class_num}"

                if field_name == "share_name":
                    current_share_class = class_key

                if current_share_class:
                    full_key = f"{current_share_class}.{field_name}"
                    ground_truth.datapoints[full_key] = normalized_value

        # Handle allocation targets
        elif category.lower() in ('investment type', 'asset class', 'geography'):
            # The first row with just a name (no allocation target) is the category name
            if datapoint.lower() not in ('allocation target minimum', 'allocation target maximum', 'allocation target - single point'):
                current_allocation_category = datapoint.lower().replace(' ', '_')
                current_allocation_type = category.lower().replace(' ', '_')
            else:
                if current_allocation_category and current_allocation_type:
                    full_key = f"{current_allocation_type}.{current_allocation_category}.{field_name}"
                    ground_truth.datapoints[full_key] = normalized_value

        # Handle concentration limits
        elif category.lower() == 'concentration limits':
            ground_truth.datapoints[field_name] = normalized_value

        # Handle regular fields
        else:
            ground_truth.datapoints[field_name] = normalized_value

        # Extract CIK
        if field_name == "fund_cik" and normalized_value:
            ground_truth.cik = str(normalized_value)

    return ground_truth


def validate_extraction(
    extracted: ExtractedValue,
    expected: Any,
    field_name: str,
    category: str = "",
    tolerance: float = 0.001,
) -> ValidationResult:
    """
    Validate a single extracted value against expected.

    Args:
        extracted: The extracted value with metadata
        expected: The expected ground truth value
        field_name: Name of the field
        category: Category of the field
        tolerance: Tolerance for numeric comparison

    Returns:
        ValidationResult with comparison details
    """
    expected_normalized = normalize_value(expected)
    extracted_value = extracted.value

    # Determine correctness and error type
    if expected_normalized is None and extracted_value is None:
        # Both None - correct (true negative)
        correct = True
        error_type = ErrorType.CORRECT
    elif expected_normalized is None and extracted_value is not None:
        # Expected None but got value - hallucination
        correct = False
        error_type = ErrorType.HALLUCINATED
    elif expected_normalized is not None and extracted_value is None:
        # Expected value but got None - missed
        correct = False
        error_type = ErrorType.MISSED
    else:
        # Both have values - compare
        if extracted.matches(expected_normalized, tolerance):
            correct = True
            error_type = ErrorType.CORRECT
        else:
            correct = False
            error_type = ErrorType.WRONG_VALUE

    return ValidationResult(
        field_name=field_name,
        category=category,
        expected=expected_normalized,
        extracted=extracted_value,
        correct=correct,
        error_type=error_type,
        confidence=extracted.confidence,
        citation=extracted.citation,
    )


def validate_fund(
    extraction: FundExtraction,
    ground_truth: GroundTruth,
) -> list[ValidationResult]:
    """
    Validate a complete fund extraction against ground truth.

    Args:
        extraction: The FundExtraction to validate
        ground_truth: Ground truth loaded from validation set

    Returns:
        List of ValidationResult for each field
    """
    results = []

    # Validate top-level fields
    top_level_fields = [
        "fund_name", "fund_manager", "sponsor", "fund_cik", "fund_currency",
        "fund_type", "fiscal_year_end", "number_of_share_types",
        "fund_target_investment_type", "fund_target_asset_class",
        "fund_target_asset_classes_gp", "fund_target_geography",
        "target_commitments_invested", "target_commitments_invested_incl_sub",
        "max_fund_leverage_pct", "max_fund_leverage_basis",
        "management_fee_pct", "management_fee_basis",
        "estimated_subscription_interest", "estimated_affe", "estimated_other_expenses",
        "repurchase_pct", "max_repurchase_pct", "repurchase_basis",
        "repurchase_frequency", "lockup_period_years", "lockup_early_repurchase_deduction",
        "mandatory_repurchase_minimum", "target_distribution_frequency",
        "default_distribution_policy",
        "max_single_asset", "max_single_fund", "max_single_manager",
        "max_single_country", "max_single_sector",
        "derivatives_allowed", "expected_derivative_use",
    ]

    for field_name in top_level_fields:
        if field_name in ground_truth.datapoints:
            extracted = getattr(extraction, field_name, ExtractedValue())
            expected = ground_truth.datapoints[field_name]
            category = ground_truth.categories.get(field_name, "")

            result = validate_extraction(extracted, expected, field_name, category)
            results.append(result)

    # Validate share classes
    for key, expected in ground_truth.datapoints.items():
        if key.startswith("share_class_"):
            parts = key.split(".")
            if len(parts) == 2:
                class_key, field_name = parts
                class_num = int(class_key.replace("share_class_", "")) - 1

                if class_num < len(extraction.share_classes):
                    share_class = extraction.share_classes[class_num]
                    extracted = getattr(share_class, field_name, ExtractedValue())
                else:
                    extracted = ExtractedValue()

                category = f"Share Type {class_num + 1}"
                result = validate_extraction(extracted, expected, key, category)
                results.append(result)

    # Validate allocation targets
    for key, expected in ground_truth.datapoints.items():
        for alloc_type in ['investment_type', 'asset_class', 'geography']:
            if key.startswith(f"{alloc_type}."):
                parts = key.split(".")
                if len(parts) == 3:
                    _, category_name, target_field = parts

                    # Find matching allocation target
                    targets_list = getattr(extraction, f"{alloc_type}_targets", [])
                    extracted = ExtractedValue()

                    for target in targets_list:
                        if target.category_name.lower().replace(' ', '_') == category_name:
                            extracted = getattr(target, target_field, ExtractedValue())
                            break

                    category = alloc_type.replace('_', ' ').title()
                    result = validate_extraction(extracted, expected, key, category)
                    results.append(result)

    return results


def create_empty_extraction() -> FundExtraction:
    """Create an empty FundExtraction for testing."""
    return FundExtraction()


def create_mock_extraction(ground_truth: GroundTruth, accuracy: float = 1.0) -> FundExtraction:
    """
    Create a mock extraction from ground truth for testing.

    Args:
        ground_truth: Ground truth to base extraction on
        accuracy: Fraction of fields to extract correctly (0.0 to 1.0)

    Returns:
        FundExtraction with values from ground truth
    """
    import random

    extraction = FundExtraction()

    # Fill in top-level fields
    for field_name, expected_value in ground_truth.datapoints.items():
        if "." not in field_name:  # Top-level field
            if hasattr(extraction, field_name):
                if random.random() < accuracy:
                    ev = ExtractedValue(
                        value=expected_value,
                        confidence=ConfidenceLevel.EXPLICIT,
                        citation=f"Mock citation for {field_name}",
                    )
                else:
                    ev = ExtractedValue()
                setattr(extraction, field_name, ev)

    # Fill in share classes
    share_class_data: dict[int, dict] = {}
    for key, value in ground_truth.datapoints.items():
        if key.startswith("share_class_"):
            parts = key.split(".")
            if len(parts) == 2:
                class_num = int(parts[0].replace("share_class_", ""))
                field_name = parts[1]

                if class_num not in share_class_data:
                    share_class_data[class_num] = {}

                if random.random() < accuracy:
                    share_class_data[class_num][field_name] = ExtractedValue(
                        value=value,
                        confidence=ConfidenceLevel.EXPLICIT,
                    )
                else:
                    share_class_data[class_num][field_name] = ExtractedValue()

    # Create ShareClassExtraction objects
    for class_num in sorted(share_class_data.keys()):
        sc = ShareClassExtraction(**share_class_data[class_num])
        extraction.share_classes.append(sc)

    # Fill in allocation targets
    alloc_data: dict[str, dict[str, dict]] = {
        "investment_type": {},
        "asset_class": {},
        "geography": {},
    }

    for key, value in ground_truth.datapoints.items():
        for alloc_type in alloc_data.keys():
            if key.startswith(f"{alloc_type}."):
                parts = key.split(".")
                if len(parts) == 3:
                    category_name = parts[1]
                    target_field = parts[2]

                    if category_name not in alloc_data[alloc_type]:
                        alloc_data[alloc_type][category_name] = {}

                    if random.random() < accuracy:
                        alloc_data[alloc_type][category_name][target_field] = ExtractedValue(
                            value=value,
                            confidence=ConfidenceLevel.EXPLICIT,
                        )
                    else:
                        alloc_data[alloc_type][category_name][target_field] = ExtractedValue()

    # Create AllocationTarget objects
    for alloc_type, categories in alloc_data.items():
        targets_list = getattr(extraction, f"{alloc_type}_targets")
        for category_name, fields in categories.items():
            target = AllocationTarget(
                category_name=category_name,
                target_min=fields.get("target_min", ExtractedValue()),
                target_max=fields.get("target_max", ExtractedValue()),
                target_single=fields.get("target_single", ExtractedValue()),
            )
            targets_list.append(target)

    return extraction
