"""
Pydantic schemas for validation against ground truth.

These schemas match the structure of the validation dataset (datapoints to extract.xlsx)
and provide a standardized format for extraction results with provenance tracking.
"""

from decimal import Decimal
from enum import Enum
from typing import Optional, Union, Any
from pydantic import BaseModel, Field, field_validator


class ConfidenceLevel(str, Enum):
    """Confidence in the extracted value."""
    EXPLICIT = "explicit"      # Value directly stated in text with clear citation
    INFERRED = "inferred"      # Value derived from context or calculation
    NOT_FOUND = "not_found"    # Searched but not found in document


class ExtractedValue(BaseModel):
    """
    A single extracted value with provenance tracking.

    This wrapper provides:
    - The extracted value (any type)
    - Confidence level
    - Citation from source text
    - Section where found
    """
    value: Optional[Union[str, float, int, bool, Decimal]] = None
    confidence: ConfidenceLevel = ConfidenceLevel.NOT_FOUND
    citation: Optional[str] = Field(
        None,
        description="Exact quote from source text supporting this value"
    )
    section: Optional[str] = Field(
        None,
        description="Section title where value was found"
    )

    @field_validator('value', mode='before')
    @classmethod
    def normalize_value(cls, v: Any) -> Optional[Union[str, float, int, bool, Decimal]]:
        """Normalize values for consistent comparison."""
        if v is None:
            return None
        if isinstance(v, str):
            v_lower = v.lower().strip()
            # Handle "none" as null
            if v_lower in ('none', 'n/a', 'not applicable', ''):
                return None
            # Try to convert numeric strings
            try:
                if '.' in v or '%' in v:
                    return float(v.replace('%', '').replace(',', '').strip())
                return int(v.replace(',', '').strip())
            except ValueError:
                return v.strip()
        return v

    def is_found(self) -> bool:
        """Check if a value was found."""
        return self.confidence != ConfidenceLevel.NOT_FOUND and self.value is not None

    def matches(self, expected: Any, tolerance: float = 0.001) -> bool:
        """
        Check if extracted value matches expected value.

        Args:
            expected: The ground truth value
            tolerance: Tolerance for numeric comparison

        Returns:
            True if values match within tolerance
        """
        # Handle None cases
        if self.value is None and expected is None:
            return True
        if self.value is None or expected is None:
            return False

        # Normalize expected value
        if isinstance(expected, str):
            expected_lower = expected.lower().strip()
            if expected_lower in ('none', 'n/a', 'not applicable', ''):
                return self.value is None

        # Numeric comparison with tolerance
        try:
            extracted_num = float(self.value) if self.value is not None else None
            expected_num = float(expected)
            if extracted_num is not None:
                return abs(extracted_num - expected_num) <= tolerance
        except (ValueError, TypeError):
            pass

        # String comparison (case-insensitive)
        if isinstance(self.value, str) and isinstance(expected, str):
            return self.value.lower().strip() == expected.lower().strip()

        # Direct comparison
        return self.value == expected


class ShareClassExtraction(BaseModel):
    """
    Extraction for a single share class.

    Matches validation set structure:
    - share_name
    - offering_price
    - sales_load_percent (fee at commitment)
    - distribution_fee_percent
    - minimum_initial
    - minimum_followon
    - repurchase_threshold
    """
    share_name: ExtractedValue = Field(default_factory=ExtractedValue)
    offering_price: ExtractedValue = Field(default_factory=ExtractedValue)
    sales_load_percent: ExtractedValue = Field(default_factory=ExtractedValue)
    distribution_fee_percent: ExtractedValue = Field(default_factory=ExtractedValue)
    minimum_initial: ExtractedValue = Field(default_factory=ExtractedValue)
    minimum_followon: ExtractedValue = Field(default_factory=ExtractedValue)
    repurchase_threshold: ExtractedValue = Field(default_factory=ExtractedValue)


class AllocationTarget(BaseModel):
    """
    Allocation target for a category (investment type, asset class, or geography).

    Matches validation set structure:
    - category_name: e.g., "secondary funds", "private equity", "north america"
    - target_min: Minimum allocation percentage
    - target_max: Maximum allocation percentage
    - target_single: Single-point target (if not a range)
    """
    category_name: str
    target_min: ExtractedValue = Field(default_factory=ExtractedValue)
    target_max: ExtractedValue = Field(default_factory=ExtractedValue)
    target_single: ExtractedValue = Field(default_factory=ExtractedValue)


class FundExtraction(BaseModel):
    """
    Complete fund extraction matching the validation dataset structure.

    Categories from validation set:
    1. Fund data (metadata, targets, leverage)
    2. Fees and expenses
    3. Repurchase and distributions
    4. Share classes (variable count: 3-4)
    5. Investment type allocation targets
    6. Asset class allocation targets
    7. Geography allocation targets
    8. Concentration limits
    """

    # ─── Fund Metadata ───
    fund_name: ExtractedValue = Field(default_factory=ExtractedValue)
    fund_manager: ExtractedValue = Field(default_factory=ExtractedValue)
    sponsor: ExtractedValue = Field(default_factory=ExtractedValue)
    fund_cik: ExtractedValue = Field(default_factory=ExtractedValue)
    fund_currency: ExtractedValue = Field(default_factory=ExtractedValue)
    fund_type: ExtractedValue = Field(default_factory=ExtractedValue)
    fiscal_year_end: ExtractedValue = Field(default_factory=ExtractedValue)
    number_of_share_types: ExtractedValue = Field(default_factory=ExtractedValue)

    # ─── Fund Targets ───
    fund_target_investment_type: ExtractedValue = Field(default_factory=ExtractedValue)
    fund_target_asset_class: ExtractedValue = Field(default_factory=ExtractedValue)
    fund_target_asset_classes_gp: ExtractedValue = Field(default_factory=ExtractedValue)
    fund_target_geography: ExtractedValue = Field(default_factory=ExtractedValue)
    target_commitments_invested: ExtractedValue = Field(default_factory=ExtractedValue)
    target_commitments_invested_incl_sub: ExtractedValue = Field(default_factory=ExtractedValue)

    # ─── Leverage ───
    max_fund_leverage_pct: ExtractedValue = Field(default_factory=ExtractedValue)
    max_fund_leverage_basis: ExtractedValue = Field(default_factory=ExtractedValue)

    # ─── Fees and Expenses ───
    management_fee_pct: ExtractedValue = Field(default_factory=ExtractedValue)
    management_fee_basis: ExtractedValue = Field(default_factory=ExtractedValue)
    estimated_subscription_interest: ExtractedValue = Field(default_factory=ExtractedValue)
    estimated_affe: ExtractedValue = Field(default_factory=ExtractedValue)
    estimated_other_expenses: ExtractedValue = Field(default_factory=ExtractedValue)

    # ─── Repurchase and Distributions ───
    repurchase_pct: ExtractedValue = Field(default_factory=ExtractedValue)
    max_repurchase_pct: ExtractedValue = Field(default_factory=ExtractedValue)
    repurchase_basis: ExtractedValue = Field(default_factory=ExtractedValue)
    repurchase_frequency: ExtractedValue = Field(default_factory=ExtractedValue)
    lockup_period_years: ExtractedValue = Field(default_factory=ExtractedValue)
    lockup_early_repurchase_deduction: ExtractedValue = Field(default_factory=ExtractedValue)
    mandatory_repurchase_minimum: ExtractedValue = Field(default_factory=ExtractedValue)
    target_distribution_frequency: ExtractedValue = Field(default_factory=ExtractedValue)
    default_distribution_policy: ExtractedValue = Field(default_factory=ExtractedValue)

    # ─── Share Classes (variable count) ───
    share_classes: list[ShareClassExtraction] = Field(default_factory=list)

    # ─── Allocation Targets ───
    investment_type_targets: list[AllocationTarget] = Field(default_factory=list)
    asset_class_targets: list[AllocationTarget] = Field(default_factory=list)
    geography_targets: list[AllocationTarget] = Field(default_factory=list)

    # ─── Concentration Limits ───
    max_single_asset: ExtractedValue = Field(default_factory=ExtractedValue)
    max_single_fund: ExtractedValue = Field(default_factory=ExtractedValue)
    max_single_manager: ExtractedValue = Field(default_factory=ExtractedValue)
    max_single_country: ExtractedValue = Field(default_factory=ExtractedValue)
    max_single_sector: ExtractedValue = Field(default_factory=ExtractedValue)
    derivatives_allowed: ExtractedValue = Field(default_factory=ExtractedValue)
    expected_derivative_use: ExtractedValue = Field(default_factory=ExtractedValue)

    def get_field(self, field_path: str) -> Optional[ExtractedValue]:
        """
        Get an extracted value by field path.

        Args:
            field_path: Dot-separated path, e.g., "management_fee_pct" or "share_classes.0.minimum_initial"

        Returns:
            ExtractedValue or None if not found
        """
        parts = field_path.split('.')
        obj: Any = self

        for part in parts:
            if obj is None:
                return None

            # Handle list index
            if part.isdigit():
                idx = int(part)
                if isinstance(obj, list) and idx < len(obj):
                    obj = obj[idx]
                else:
                    return None
            # Handle attribute access
            elif hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None

        if isinstance(obj, ExtractedValue):
            return obj
        return None

    def count_fields(self) -> dict[str, int]:
        """Count fields by status."""
        counts = {
            "total": 0,
            "found": 0,
            "explicit": 0,
            "inferred": 0,
            "not_found": 0,
        }

        def count_extracted_value(ev: ExtractedValue):
            counts["total"] += 1
            if ev.is_found():
                counts["found"] += 1
            if ev.confidence == ConfidenceLevel.EXPLICIT:
                counts["explicit"] += 1
            elif ev.confidence == ConfidenceLevel.INFERRED:
                counts["inferred"] += 1
            else:
                counts["not_found"] += 1

        # Count top-level ExtractedValue fields
        for field_name, field_value in self:
            if isinstance(field_value, ExtractedValue):
                count_extracted_value(field_value)
            elif isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, ShareClassExtraction):
                        for _, ev in item:
                            if isinstance(ev, ExtractedValue):
                                count_extracted_value(ev)
                    elif isinstance(item, AllocationTarget):
                        count_extracted_value(item.target_min)
                        count_extracted_value(item.target_max)
                        count_extracted_value(item.target_single)

        return counts
