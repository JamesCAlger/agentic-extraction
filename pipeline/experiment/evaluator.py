"""
Evaluation engine for comparing extractions against ground truth.

Provides:
- JSON ground truth loading
- Field-level comparison with type coercion
- Precision/recall/F1 calculation
- Per-fund and aggregate metrics
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Ground Truth Data Structures
# =============================================================================


@dataclass
class FieldGroundTruth:
    """Ground truth for a single field."""

    field_path: str
    expected: Any
    source_section: Optional[str] = None
    source_quote: Optional[str] = None
    nullable: bool = False
    tolerance: Optional[float] = None
    match_mode: str = "exact"  # exact | subset | superset
    notes: Optional[str] = None


@dataclass
class GroundTruth:
    """Complete ground truth for a fund."""

    fund_name: str
    cik: str
    filing_path: str
    filing_date: Optional[str] = None
    form_type: Optional[str] = None
    created: Optional[str] = None
    verified_by: Optional[str] = None
    notes: Optional[str] = None
    fields: dict[str, FieldGroundTruth] = field(default_factory=dict)


def load_ground_truth(path: Union[str, Path]) -> GroundTruth:
    """
    Load ground truth from JSON file.

    Args:
        path: Path to ground truth JSON file

    Returns:
        GroundTruth object with all expected values
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Parse fields
    fields = {}
    for field_path, field_data in data.get("fields", {}).items():
        if isinstance(field_data, dict):
            fields[field_path] = FieldGroundTruth(
                field_path=field_path,
                expected=field_data.get("expected"),
                source_section=field_data.get("source_section"),
                source_quote=field_data.get("source_quote"),
                nullable=field_data.get("nullable", False),
                tolerance=field_data.get("tolerance"),
                match_mode=field_data.get("match_mode", "exact"),
                notes=field_data.get("notes"),
            )
        else:
            # Simple value format (just the expected value)
            fields[field_path] = FieldGroundTruth(
                field_path=field_path,
                expected=field_data,
            )

    return GroundTruth(
        fund_name=data.get("fund_name", ""),
        cik=data.get("cik", ""),
        filing_path=data.get("filing_path", ""),
        filing_date=data.get("filing_date"),
        form_type=data.get("form_type"),
        created=data.get("created"),
        verified_by=data.get("verified_by"),
        notes=data.get("notes"),
        fields=fields,
    )


def load_all_ground_truth(ground_truth_dir: Union[str, Path]) -> dict[str, GroundTruth]:
    """
    Load all ground truth files from a directory.

    Args:
        ground_truth_dir: Directory containing JSON ground truth files

    Returns:
        Dict mapping fund_name -> GroundTruth
    """
    ground_truth_dir = Path(ground_truth_dir)
    result = {}

    for gt_file in ground_truth_dir.glob("*.json"):
        try:
            gt = load_ground_truth(gt_file)
            result[gt.fund_name] = gt
            logger.info(f"Loaded ground truth for {gt.fund_name} ({len(gt.fields)} fields)")
        except Exception as e:
            logger.warning(f"Error loading {gt_file}: {e}")

    return result


# =============================================================================
# Value Comparison
# =============================================================================

# Word numbers to numeric values
WORD_TO_NUMBER = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1000000, "billion": 1000000000,
    # Common variations
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "half": 0.5, "quarter": 0.25,
    # Years as words
    "a year": 1, "one year": 1, "two years": 2, "three years": 3,
}

# Currency and formatting patterns to strip
CURRENCY_PATTERNS = [
    r"^\$\s*",           # Leading dollar sign
    r"^€\s*",            # Leading euro sign
    r"^£\s*",            # Leading pound sign
    r"^USD\s*",          # USD prefix
    r"^US\$\s*",         # US$ prefix
    r"\s*dollars?$",     # "dollars" suffix
    r"\s*USD$",          # USD suffix
]


def parse_word_number(text: str) -> Optional[float]:
    """
    Parse word numbers to numeric values.

    Handles:
    - Simple: "one" -> 1
    - Compound: "twenty-five" -> 25 (not fully implemented, but handles common cases)
    - With multipliers: "one million" -> 1000000

    Returns None if not a recognized word number.
    """
    text = text.lower().strip()

    # Direct lookup
    if text in WORD_TO_NUMBER:
        return float(WORD_TO_NUMBER[text])

    # Handle compound numbers like "twenty five" or "twenty-five"
    parts = re.split(r'[\s-]+', text)
    if len(parts) == 2:
        first, second = parts
        if first in WORD_TO_NUMBER and second in WORD_TO_NUMBER:
            first_val = WORD_TO_NUMBER[first]
            second_val = WORD_TO_NUMBER[second]
            # Handle tens + ones: "twenty five" = 25
            if first_val >= 20 and first_val < 100 and second_val < 10:
                return float(first_val + second_val)
            # Handle multipliers: "five hundred" = 500, "two million" = 2000000
            if second_val in (100, 1000, 1000000, 1000000000):
                return float(first_val * second_val)

    # Handle three-part numbers: "one hundred twenty"
    if len(parts) == 3:
        if parts[0] in WORD_TO_NUMBER and parts[1] == "hundred" and parts[2] in WORD_TO_NUMBER:
            return float(WORD_TO_NUMBER[parts[0]] * 100 + WORD_TO_NUMBER[parts[2]])

    return None


def normalize_numeric_string(value: str) -> Optional[Union[int, float]]:
    """
    Normalize a string that may contain a numeric value.

    Handles:
    - Currency: "$2,500" -> 2500
    - Commas: "1,000,000" -> 1000000
    - Percentages: "2.5%" -> 2.5
    - Word numbers: "one" -> 1
    - Mixed: "$1.5 million" -> 1500000
    - Ranges: Takes first number from "5-25%" -> 5

    Returns None if not parseable as numeric.
    """
    if not value:
        return None

    original = value
    value = value.strip()

    # Check for word numbers first (before stripping characters)
    word_num = parse_word_number(value)
    if word_num is not None:
        return word_num

    # Remove currency symbols and prefixes
    for pattern in CURRENCY_PATTERNS:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE)

    # Handle multiplier suffixes: "1.5 million", "2M", "3K"
    multiplier = 1
    multiplier_patterns = [
        (r"\s*(million|mm|m)\s*$", 1000000),
        (r"\s*(billion|b)\s*$", 1000000000),
        (r"\s*(thousand|k)\s*$", 1000),
    ]
    for pattern, mult in multiplier_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            value = re.sub(pattern, "", value, re.IGNORECASE)
            multiplier = mult
            break

    # Remove commas, percentage signs, and extra whitespace
    value = value.replace(",", "").replace("%", "").strip()

    # Handle ranges like "5-25" or "5 to 25" - take first number
    range_match = re.match(r"^([\d.]+)\s*[-–—]\s*[\d.]+", value)
    if range_match:
        value = range_match.group(1)

    # Handle "X to Y" format
    to_match = re.match(r"^([\d.]+)\s+to\s+[\d.]+", value, re.IGNORECASE)
    if to_match:
        value = to_match.group(1)

    # Try to parse as number
    try:
        if "." in value:
            return float(value) * multiplier
        else:
            result = int(value) * multiplier
            # Return as int if no decimal, float otherwise
            return result if multiplier == 1 else float(result)
    except ValueError:
        pass

    # Try extracting just the first number found
    num_match = re.search(r"([\d,]+\.?\d*)", original)
    if num_match:
        try:
            extracted = num_match.group(1).replace(",", "")
            if "." in extracted:
                return float(extracted) * multiplier
            return int(extracted) * multiplier
        except ValueError:
            pass

    return None


# Synonym mappings - map equivalent terms to canonical form
# Comprehensive mapping aligned with validation_rules.py
VALUE_SYNONYMS = {
    # ==========================================================================
    # DISTRIBUTION POLICY SYNONYMS
    # ==========================================================================
    "reinvested": "drip",
    "drip": "drip",
    "dividend reinvestment": "drip",
    "dividend reinvestment plan": "drip",
    "reinvestment": "drip",
    "automatically reinvested": "drip",
    "reinvest": "drip",
    "cash": "cash",
    "cash distribution": "cash",
    "paid in cash": "cash",

    # ==========================================================================
    # REPURCHASE BASIS SYNONYMS
    # ==========================================================================
    # Outstanding shares variants -> "shares"
    "nav": "nav",
    "net_assets": "nav",
    "net assets": "nav",
    "net asset value": "nav",
    "of nav": "nav",
    "number_of_shares": "shares",
    "shares": "shares",
    "outstanding shares": "shares",
    "shares outstanding": "shares",
    "total shares": "shares",
    "of shares": "shares",
    # Total assets variants
    "total assets": "total_assets",
    "gross assets": "total_assets",

    # ==========================================================================
    # LEVERAGE BASIS SYNONYMS
    # ==========================================================================
    # Asset coverage variants (300% coverage = 33.33% debt-to-assets)
    "asset_coverage": "asset_coverage",
    "asset coverage": "asset_coverage",
    "asset coverage ratio": "asset_coverage",
    "300%": "asset_coverage",
    "300% coverage": "asset_coverage",
    "300 percent": "asset_coverage",
    "1940 act": "asset_coverage",
    "investment company act": "asset_coverage",
    "section 18": "asset_coverage",
    # Total assets variants
    "total_assets": "total_assets",
    # Note: "total assets" already mapped above
    "gross assets": "total_assets",
    "total fund assets": "total_assets",
    # Net assets variants
    # Note: "net assets" already mapped to "nav" for repurchase_basis
    # For leverage_basis, we need context - handled in normalize_value
    # Debt-to-equity variants
    "debt_to_equity": "debt_to_equity",
    "debt to equity": "debt_to_equity",
    "debt-to-equity": "debt_to_equity",
    "debt/equity": "debt_to_equity",
    "leverage ratio": "debt_to_equity",

    # ==========================================================================
    # FREQUENCY SYNONYMS (for distribution, repurchase, hurdle rate)
    # ==========================================================================
    # Monthly variants
    "monthly": "monthly",
    "each month": "monthly",
    "per month": "monthly",
    "every month": "monthly",
    "paid monthly": "monthly",
    # Quarterly variants
    "quarterly": "quarterly",
    "each quarter": "quarterly",
    "per quarter": "quarterly",
    "every quarter": "quarterly",
    "paid quarterly": "quarterly",
    "four times per year": "quarterly",
    "quarterly basis": "quarterly",
    "calendar quarter": "quarterly",
    # Annual variants
    "annually": "annually",
    "annual": "annually",
    "yearly": "annually",
    "once per year": "annually",
    "paid annually": "annually",
    "per annum": "annually",
    "per year": "annually",
    "annualized": "annually",
    # Daily
    "daily": "daily",
    "each day": "daily",
    "per day": "daily",
    # Semi-annual
    "semi-annually": "semi_annually",
    "semi-annual": "semi_annually",
    "twice per year": "semi_annually",
    "two times per year": "semi_annually",

    # ==========================================================================
    # FEE BASIS SYNONYMS
    # ==========================================================================
    "net_profits": "net_profits",
    "net profits": "net_profits",
    "total_return": "total_return",
    "total return": "total_return",
    "net_investment_income": "net_investment_income",
    "net investment income": "net_investment_income",
    "realized_gains": "realized_gains",
    "realized gains": "realized_gains",
    "nav_appreciation": "nav_appreciation",
}

# Field name aliases - maps ground truth field names to extraction schema field names
# Used when comparing nested objects like share_classes
FIELD_NAME_ALIASES = {
    # Ground truth uses distribution_fee_pct, schema uses distribution_servicing_fee_pct
    "distribution_fee_pct": "distribution_servicing_fee_pct",
    "distribution_servicing_fee_pct": "distribution_fee_pct",
}

# Field path mappings - maps ground truth field paths to extraction field paths
# Used when extraction stores normalized values in different fields
FIELD_PATH_MAPPINGS = {
    # Ground truth expects quarterly rate in hurdle_rate_as_stated
    # But extraction may find annualized rate and store quarterly in hurdle_rate_quarterly
    "incentive_fee.hurdle_rate_as_stated": "incentive_fee.hurdle_rate_quarterly",
}


def normalize_class_name(name: str) -> str:
    """
    Normalize a share class name for comparison.

    Handles variations like:
    - "Class U Shares" -> "class u"
    - "Class U" -> "class u"
    - "Class I Advisory" -> "class i advisory"
    """
    if not name:
        return ""
    # Lowercase and strip
    normalized = name.lower().strip()
    # Remove "shares" suffix (with optional leading space)
    if normalized.endswith(" shares"):
        normalized = normalized[:-7].strip()
    elif normalized.endswith("shares"):
        normalized = normalized[:-6].strip()
    return normalized


def normalize_value(value: Any) -> Any:
    """
    Normalize a value for comparison.

    Handles:
    - None/null variations
    - String/number coercion (including currency like "$2,500" -> 2500)
    - Word numbers ("one" -> 1)
    - Percentage normalization ("2.5%" -> 2.5)
    - Synonym mapping (e.g., "reinvested" -> "drip")
    """
    if value is None:
        return None

    if isinstance(value, str):
        v_lower = value.lower().strip()

        # Handle null-like strings
        if v_lower in ("none", "n/a", "not applicable", "", "null", "nan"):
            return None

        # Check for synonyms first
        if v_lower in VALUE_SYNONYMS:
            return VALUE_SYNONYMS[v_lower]

        # Try robust numeric conversion (handles currency, word numbers, etc.)
        numeric = normalize_numeric_string(value)
        if numeric is not None:
            return numeric

        # Return lowercase for consistent string comparison
        return value.strip().lower()

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value

    return value


def values_match(
    extracted: Any,
    expected: Any,
    tolerance: Optional[float] = None,
    match_mode: str = "exact",
) -> bool:
    """
    Compare extracted value against expected.

    Args:
        extracted: Value from extraction
        expected: Ground truth value
        tolerance: Numeric tolerance (absolute difference)
        match_mode: How to compare lists (exact, subset, superset)

    Returns:
        True if values match
    """
    # Normalize both values
    extracted_norm = normalize_value(extracted)
    expected_norm = normalize_value(expected)

    # Both None = match
    if extracted_norm is None and expected_norm is None:
        return True

    # SEMANTIC SIMPLIFICATION: For boolean-like fields, null ≈ false
    # This handles cases like:
    # - expected=null (N/A), extracted="false" → match (both mean "no")
    # - expected=false, extracted=null → match (both mean "no")
    if extracted_norm is None and expected_norm is False:
        return True
    if expected_norm is None and isinstance(extracted_norm, str) and extracted_norm.lower() == "false":
        return True
    if expected_norm is None and extracted_norm is False:
        return True

    # One None = no match (for non-boolean cases)
    if extracted_norm is None or expected_norm is None:
        return False

    # Boolean comparison - MUST CHECK BEFORE NUMERIC!
    # (In Python, bool is a subclass of int, so isinstance(False, int) is True)
    if isinstance(expected_norm, bool):
        if isinstance(extracted_norm, bool):
            return extracted_norm == expected_norm
        # Handle string "true"/"false"
        if isinstance(extracted_norm, str):
            return extracted_norm.lower() == str(expected_norm).lower()
        # SEMANTIC SIMPLIFICATION: For boolean fields, treat null as false
        # Rationale: The distinction between "doesn't exist" and "not applicable"
        # rarely matters for downstream use. Users want "yes/no" not "N/A".
        if expected_norm is False and extracted_norm is None:
            return True
        return False

    # Numeric comparison with tolerance
    if isinstance(expected_norm, (int, float)):
        try:
            extracted_num = float(extracted_norm)
            expected_num = float(expected_norm)
            tol = tolerance if tolerance is not None else 0.01
            return abs(extracted_num - expected_num) <= tol
        except (ValueError, TypeError):
            return False

    # String comparison (case-insensitive)
    if isinstance(expected_norm, str) and isinstance(extracted_norm, str):
        return extracted_norm.lower() == expected_norm.lower()

    # List comparison
    if isinstance(expected_norm, list) and isinstance(extracted_norm, list):
        return compare_lists(extracted_norm, expected_norm, match_mode)

    # Direct comparison
    return extracted_norm == expected_norm


def compare_lists(
    extracted: list,
    expected: list,
    match_mode: str = "exact",
) -> bool:
    """
    Compare two lists based on match mode.

    Args:
        extracted: Extracted list
        expected: Expected list
        match_mode: exact | subset | superset

    Returns:
        True if lists match according to mode
    """
    if match_mode == "exact":
        # Order-independent exact match
        if len(extracted) != len(expected):
            return False
        # For dicts, compare by key fields
        if extracted and isinstance(extracted[0], dict):
            return compare_dict_lists(extracted, expected)
        return set(map(str, extracted)) == set(map(str, expected))

    elif match_mode == "subset":
        # All extracted must be in expected (no false positives)
        for item in extracted:
            if not any(item_matches(item, exp) for exp in expected):
                return False
        return True

    elif match_mode == "superset":
        # All expected must be in extracted (no false negatives)
        for exp in expected:
            if not any(item_matches(item, exp) for item in extracted):
                return False
        return True

    return False


def get_field_value(item: dict, field_name: str) -> Any:
    """
    Get a field value from a dict, checking aliases if needed.

    Args:
        item: Dictionary to get value from
        field_name: Field name to look for

    Returns:
        Field value or None if not found
    """
    # Direct match
    if field_name in item:
        return item[field_name]

    # Check aliases
    alias = FIELD_NAME_ALIASES.get(field_name)
    if alias and alias in item:
        return item[alias]

    return None


def compare_dict_lists(extracted: list[dict], expected: list[dict]) -> bool:
    """
    Compare lists of dicts (e.g., share classes).

    Matches by key field (class_name) and compares other fields.
    Handles field name aliases, type coercion, and class name normalization.
    """
    if len(extracted) != len(expected):
        return False

    # Match by class_name or first string field (normalized)
    for exp_item in expected:
        key = exp_item.get("class_name") or list(exp_item.values())[0]
        key_normalized = normalize_class_name(str(key)) if key else ""
        matching = [
            e for e in extracted
            if normalize_class_name(str(e.get("class_name") or list(e.values())[0])) == key_normalized
        ]
        if not matching:
            return False

        # Compare fields
        ext_item = matching[0]
        for field_name, exp_value in exp_item.items():
            # Skip class_name as it's already matched
            if field_name == "class_name":
                continue

            # Get extracted value (with alias support)
            ext_value = get_field_value(ext_item, field_name)

            # If expected value exists, extracted must match
            if exp_value is not None:
                if ext_value is None:
                    return False
                if not values_match(ext_value, exp_value):
                    return False

    return True


def item_matches(extracted: Any, expected: Any) -> bool:
    """Check if an extracted item matches an expected item."""
    if isinstance(extracted, dict) and isinstance(expected, dict):
        # Match by class_name field (with normalization for "Shares" suffix)
        if "class_name" in expected:
            ext_name = extracted.get("class_name")
            exp_name = expected.get("class_name")
            if ext_name and exp_name:
                return normalize_class_name(str(ext_name)) == normalize_class_name(str(exp_name))
        # Fallback to first key
        key = list(expected.keys())[0]
        if key in expected and key in extracted:
            return normalize_value(extracted[key]) == normalize_value(expected[key])
    return normalize_value(extracted) == normalize_value(expected)


# =============================================================================
# Evaluation Results
# =============================================================================


@dataclass
class FieldEvaluationResult:
    """Result of evaluating a single field."""

    field_path: str
    expected: Any
    extracted: Any
    is_correct: bool
    error_type: str  # correct | wrong_value | missed | hallucinated
    ground_truth: Optional[FieldGroundTruth] = None


@dataclass
class FundEvaluationResult:
    """Result of evaluating a single fund."""

    fund_name: str
    field_results: dict[str, FieldEvaluationResult] = field(default_factory=dict)

    @property
    def total_fields(self) -> int:
        return len(self.field_results)

    @property
    def correct_fields(self) -> int:
        return sum(1 for r in self.field_results.values() if r.is_correct)

    @property
    def precision(self) -> float:
        """Correct / Extracted (how many extractions were right)."""
        extracted = sum(
            1 for r in self.field_results.values()
            if r.extracted is not None
        )
        if extracted == 0:
            return 0.0
        correct = sum(
            1 for r in self.field_results.values()
            if r.is_correct and r.extracted is not None
        )
        return correct / extracted

    @property
    def recall(self) -> float:
        """Correct / Expected (how many expected values were found)."""
        expected_non_null = sum(
            1 for r in self.field_results.values()
            if r.expected is not None
        )
        if expected_non_null == 0:
            return 1.0  # Nothing expected, recall is perfect
        correct = sum(
            1 for r in self.field_results.values()
            if r.is_correct and r.expected is not None
        )
        return correct / expected_non_null

    @property
    def f1_score(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def accuracy(self) -> float:
        """Simple accuracy (correct / total)."""
        if self.total_fields == 0:
            return 0.0
        return self.correct_fields / self.total_fields

    def error_breakdown(self) -> dict[str, int]:
        """Count of each error type."""
        breakdown = {"correct": 0, "wrong_value": 0, "missed": 0, "hallucinated": 0}
        for r in self.field_results.values():
            breakdown[r.error_type] = breakdown.get(r.error_type, 0) + 1
        return breakdown


@dataclass
class EvaluationResult:
    """Complete evaluation result across all funds."""

    run_id: str
    ground_truth_version: str
    fund_results: dict[str, FundEvaluationResult] = field(default_factory=dict)

    @property
    def overall_precision(self) -> float:
        """Aggregate precision across all funds."""
        total_extracted = 0
        total_correct = 0
        for fund in self.fund_results.values():
            for r in fund.field_results.values():
                if r.extracted is not None:
                    total_extracted += 1
                    if r.is_correct:
                        total_correct += 1
        return total_correct / total_extracted if total_extracted > 0 else 0.0

    @property
    def overall_recall(self) -> float:
        """Aggregate recall across all funds."""
        total_expected = 0
        total_correct = 0
        for fund in self.fund_results.values():
            for r in fund.field_results.values():
                if r.expected is not None:
                    total_expected += 1
                    if r.is_correct:
                        total_correct += 1
        return total_correct / total_expected if total_expected > 0 else 1.0

    @property
    def overall_f1(self) -> float:
        """Aggregate F1 score."""
        p, r = self.overall_precision, self.overall_recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def overall_accuracy(self) -> float:
        """Aggregate accuracy."""
        total = sum(f.total_fields for f in self.fund_results.values())
        correct = sum(f.correct_fields for f in self.fund_results.values())
        return correct / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "run_id": self.run_id,
            "ground_truth_version": self.ground_truth_version,
            "overall": {
                "precision": self.overall_precision,
                "recall": self.overall_recall,
                "f1_score": self.overall_f1,
                "accuracy": self.overall_accuracy,
            },
            "per_fund": {
                name: {
                    "precision": fund.precision,
                    "recall": fund.recall,
                    "f1_score": fund.f1_score,
                    "accuracy": fund.accuracy,
                    "total_fields": fund.total_fields,
                    "correct_fields": fund.correct_fields,
                    "error_breakdown": fund.error_breakdown(),
                }
                for name, fund in self.fund_results.items()
            },
            "per_field": self._per_field_accuracy(),
        }

    def _per_field_accuracy(self) -> dict[str, dict]:
        """Calculate accuracy for each field across funds."""
        field_stats: dict[str, dict] = {}

        for fund_name, fund in self.fund_results.items():
            for field_path, result in fund.field_results.items():
                if field_path not in field_stats:
                    field_stats[field_path] = {
                        "total": 0,
                        "correct": 0,
                        "funds_correct": [],
                        "funds_incorrect": [],
                    }

                field_stats[field_path]["total"] += 1
                if result.is_correct:
                    field_stats[field_path]["correct"] += 1
                    field_stats[field_path]["funds_correct"].append(fund_name)
                else:
                    field_stats[field_path]["funds_incorrect"].append(fund_name)

        # Calculate accuracy for each field
        for field_path, stats in field_stats.items():
            stats["accuracy"] = (
                stats["correct"] / stats["total"]
                if stats["total"] > 0
                else 0.0
            )

        return field_stats


# =============================================================================
# Evaluation Functions
# =============================================================================


def get_nested_value(data: dict, path: str) -> Any:
    """
    Get a nested value from a dict using dot notation.

    Handles:
    - Simple paths: "fund_type"
    - Nested paths: "incentive_fee.hurdle_rate_pct"
    - Array access: "share_classes.share_classes[0].class_name"
    """
    if not data:
        return None

    parts = path.split(".")
    current = data

    for part in parts:
        if current is None:
            return None

        # Handle array index notation: field[0]
        match = re.match(r"(\w+)\[(\d+)\]", part)
        if match:
            field_name, index = match.groups()
            if isinstance(current, dict):
                current = current.get(field_name)
            if isinstance(current, list) and int(index) < len(current):
                current = current[int(index)]
            else:
                return None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None

    return current


def evaluate_field(
    extraction: dict,
    gt_field: FieldGroundTruth,
) -> FieldEvaluationResult:
    """
    Evaluate a single field extraction against ground truth.

    Args:
        extraction: Full extraction result dict
        gt_field: Ground truth for this field

    Returns:
        FieldEvaluationResult with comparison details
    """
    # Check for field path mapping (e.g., hurdle_rate_as_stated -> hurdle_rate_quarterly)
    extraction_path = gt_field.field_path
    if gt_field.field_path in FIELD_PATH_MAPPINGS:
        mapped_path = FIELD_PATH_MAPPINGS[gt_field.field_path]
        mapped_value = get_nested_value(extraction, mapped_path)
        # Use mapped path if it has a value, otherwise fall back to original
        if mapped_value is not None:
            extraction_path = mapped_path

    extracted = get_nested_value(extraction, extraction_path)
    expected = gt_field.expected

    # Determine if correct
    is_correct = values_match(
        extracted,
        expected,
        tolerance=gt_field.tolerance,
        match_mode=gt_field.match_mode,
    )

    # Determine error type
    if is_correct:
        error_type = "correct"
    elif expected is None and extracted is not None:
        error_type = "hallucinated"
    elif expected is not None and extracted is None:
        error_type = "missed"
    else:
        error_type = "wrong_value"

    return FieldEvaluationResult(
        field_path=gt_field.field_path,
        expected=expected,
        extracted=extracted,
        is_correct=is_correct,
        error_type=error_type,
        ground_truth=gt_field,
    )


def expand_list_field_to_subfields(
    field_path: str,
    gt_field: FieldGroundTruth,
) -> list[FieldGroundTruth]:
    """
    Expand a list-of-dicts field into individual sub-fields.

    E.g., share_classes with expected=[{class_name: "S", min: 2500}, ...]
    becomes:
      - share_classes.S.class_name
      - share_classes.S.minimum_initial_investment
      - etc.

    This enables per-sub-field evaluation instead of all-or-nothing.
    """
    expected = gt_field.expected

    # Only expand if it's a list of dicts
    if not isinstance(expected, list):
        return [gt_field]
    if not expected or not isinstance(expected[0], dict):
        return [gt_field]

    sub_fields = []
    for item in expected:
        # Use class_name as key if available, otherwise use index
        key = item.get("class_name", item.get("name"))
        if not key:
            # Can't expand without a key - fall back to original
            return [gt_field]

        # Create a sub-field for each attribute in this item
        for attr_name, attr_value in item.items():
            sub_path = f"{field_path}.{key}.{attr_name}"
            sub_fields.append(FieldGroundTruth(
                field_path=sub_path,
                expected=attr_value,
                source_section=gt_field.source_section,
                source_quote=gt_field.source_quote,
                nullable=gt_field.nullable,
                tolerance=gt_field.tolerance,
                match_mode="exact",  # Individual fields use exact match
                notes=gt_field.notes,
            ))

    return sub_fields


def evaluate_list_subfield(
    extraction: dict,
    gt_field: FieldGroundTruth,
) -> FieldEvaluationResult:
    """
    Evaluate a sub-field within a list-of-dicts structure.

    E.g., for share_classes.share_classes.Class S.minimum_initial_investment:
    - base_field = "share_classes.share_classes" (path to the list)
    - item_key = "Class S" (class_name value to match)
    - attr_name = "minimum_initial_investment" (attribute to extract)
    """
    parts = gt_field.field_path.split(".")
    if len(parts) < 3:
        # Not a valid sub-field path
        return FieldEvaluationResult(
            field_path=gt_field.field_path,
            expected=gt_field.expected,
            extracted=None,
            is_correct=False,
            error_type="missed",
            ground_truth=gt_field,
        )

    # Last part is the attribute name
    attr_name = parts[-1]    # e.g., "minimum_initial_investment"
    # Second to last is the item key (class name)
    item_key = parts[-2]     # e.g., "Class S"
    # Everything before is the base field path
    base_field = ".".join(parts[:-2])  # e.g., "share_classes.share_classes"

    # Get the list from extraction
    extracted_list = get_nested_value(extraction, base_field)
    if not isinstance(extracted_list, list):
        return FieldEvaluationResult(
            field_path=gt_field.field_path,
            expected=gt_field.expected,
            extracted=None,
            is_correct=False,
            error_type="missed",
            ground_truth=gt_field,
        )

    # Find matching item by class_name (normalized to handle "Class U" vs "Class U Shares")
    extracted_value = None
    item_key_normalized = normalize_class_name(item_key)
    for item in extracted_list:
        if isinstance(item, dict):
            item_name = item.get("class_name") or item.get("name")
            if normalize_class_name(item_name) == item_key_normalized:
                # Found matching item - get the attribute value (with alias support)
                extracted_value = get_field_value(item, attr_name)
                break

    expected = gt_field.expected

    # Determine if correct
    # Special handling for class_name field - use class name normalization
    if attr_name == "class_name" and isinstance(extracted_value, str) and isinstance(expected, str):
        is_correct = normalize_class_name(extracted_value) == normalize_class_name(expected)
    else:
        is_correct = values_match(
            extracted_value,
            expected,
            tolerance=gt_field.tolerance,
            match_mode="exact",
        )

    # Determine error type
    if is_correct:
        error_type = "correct"
    elif expected is None and extracted_value is not None:
        error_type = "hallucinated"
    elif expected is not None and extracted_value is None:
        error_type = "missed"
    else:
        error_type = "wrong_value"

    return FieldEvaluationResult(
        field_path=gt_field.field_path,
        expected=expected,
        extracted=extracted_value,
        is_correct=is_correct,
        error_type=error_type,
        ground_truth=gt_field,
    )


def evaluate_extraction(
    extraction: dict,
    ground_truth: GroundTruth,
) -> FundEvaluationResult:
    """
    Evaluate a fund extraction against ground truth.

    Args:
        extraction: Extraction result dict
        ground_truth: Ground truth for this fund

    Returns:
        FundEvaluationResult with all field comparisons
    """
    result = FundEvaluationResult(fund_name=ground_truth.fund_name)

    for field_path, gt_field in ground_truth.fields.items():
        # Check if this is a list-of-dicts field that should be expanded
        expected = gt_field.expected
        if (isinstance(expected, list) and expected and
            isinstance(expected[0], dict) and
            gt_field.match_mode == "exact"):
            # Expand into sub-fields for granular evaluation
            sub_fields = expand_list_field_to_subfields(field_path, gt_field)
            for sub_field in sub_fields:
                if "." in sub_field.field_path and sub_field.field_path.count(".") >= 2:
                    # This is an expanded sub-field
                    field_result = evaluate_list_subfield(extraction, sub_field)
                else:
                    # Fallback - couldn't expand
                    field_result = evaluate_field(extraction, sub_field)
                result.field_results[sub_field.field_path] = field_result
        else:
            field_result = evaluate_field(extraction, gt_field)
            result.field_results[field_path] = field_result

    return result


def evaluate_run(
    run,  # ExperimentRun - imported dynamically to avoid circular import
    ground_truth_dir: Union[str, Path],
) -> EvaluationResult:
    """
    Evaluate a complete experiment run against ground truth.

    Args:
        run: ExperimentRun to evaluate
        ground_truth_dir: Directory containing ground truth JSON files

    Returns:
        EvaluationResult with all metrics
    """
    # Load all ground truth
    all_gt = load_all_ground_truth(ground_truth_dir)

    result = EvaluationResult(
        run_id=run.run_id,
        ground_truth_version=str(ground_truth_dir),
    )

    for fund_name, fund_result in run.fund_results.items():
        # Find matching ground truth
        gt = all_gt.get(fund_name)
        if not gt:
            logger.warning(f"No ground truth found for {fund_name}")
            continue

        # Evaluate this fund
        fund_eval = evaluate_extraction(
            fund_result.extraction,
            gt,
        )
        result.fund_results[fund_name] = fund_eval

        logger.info(
            f"{fund_name}: {fund_eval.correct_fields}/{fund_eval.total_fields} "
            f"({fund_eval.accuracy:.1%} accuracy)"
        )

    return result


# =============================================================================
# Reporting
# =============================================================================


def print_evaluation_report(
    evaluation: EvaluationResult,
    title: str = "EXTRACTION EVALUATION REPORT",
    show_field_details: bool = False,
) -> str:
    """
    Generate a formatted evaluation report.

    Args:
        evaluation: EvaluationResult to report on
        title: Report title
        show_field_details: Whether to show per-field breakdown

    Returns:
        Formatted report string
    """
    lines = []
    sep = "=" * 60

    lines.append(sep)
    lines.append(f" {title}")
    lines.append(sep)
    lines.append(f"Run ID: {evaluation.run_id}")
    lines.append(f"Ground Truth: {evaluation.ground_truth_version}")
    lines.append("")

    # Overall metrics
    lines.append("OVERALL METRICS")
    lines.append("-" * 40)
    lines.append(f"  Precision:  {evaluation.overall_precision:.1%}")
    lines.append(f"  Recall:     {evaluation.overall_recall:.1%}")
    lines.append(f"  F1 Score:   {evaluation.overall_f1:.1%}")
    lines.append(f"  Accuracy:   {evaluation.overall_accuracy:.1%}")
    lines.append("")

    # Per-fund breakdown
    lines.append("PER-FUND BREAKDOWN")
    lines.append("-" * 40)
    for fund_name, fund in evaluation.fund_results.items():
        lines.append(f"  {fund_name[:35]:<35}")
        lines.append(
            f"    Correct: {fund.correct_fields}/{fund.total_fields} "
            f"({fund.accuracy:.1%})"
        )
        errors = fund.error_breakdown()
        lines.append(
            f"    Errors: {errors['wrong_value']} wrong, "
            f"{errors['missed']} missed, {errors['hallucinated']} hallucinated"
        )
    lines.append("")

    # Per-field accuracy (if requested)
    if show_field_details:
        lines.append("PER-FIELD ACCURACY")
        lines.append("-" * 40)
        field_stats = evaluation._per_field_accuracy()

        # Sort by accuracy (worst first)
        sorted_fields = sorted(
            field_stats.items(),
            key=lambda x: x[1]["accuracy"],
        )

        for field_path, stats in sorted_fields:
            acc = stats["accuracy"]
            lines.append(
                f"  {field_path:<45} {stats['correct']}/{stats['total']} ({acc:.0%})"
            )
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)
