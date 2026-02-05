"""
Comprehensive tests for the validation harness (Phase 1).

Tests cover:
1. ExtractedValue schema and value matching
2. FundExtraction schema structure
3. Ground truth loading from Excel
4. Validation comparison logic
5. Metrics computation
6. Full end-to-end validation workflow
"""

import pytest
from pathlib import Path
from decimal import Decimal

# Import validation module components
from pipeline.validate.schemas import (
    ConfidenceLevel,
    ExtractedValue,
    ShareClassExtraction,
    AllocationTarget,
    FundExtraction,
)
from pipeline.validate.harness import (
    ErrorType,
    ValidationResult,
    GroundTruth,
    load_ground_truth,
    validate_extraction,
    validate_fund,
    create_empty_extraction,
    create_mock_extraction,
    normalize_value,
    normalize_field_name,
)
from pipeline.validate.metrics import (
    ValidationMetrics,
    compute_metrics,
    print_metrics_report,
    compare_metrics,
)


# ─── Test Data ───

EXCEL_PATH = Path(__file__).parent.parent / "datapoints to extract.xlsx"


class TestExtractedValue:
    """Tests for ExtractedValue schema."""

    def test_creation_with_defaults(self):
        """Test default ExtractedValue creation."""
        ev = ExtractedValue()
        assert ev.value is None
        assert ev.confidence == ConfidenceLevel.NOT_FOUND
        assert ev.citation is None
        assert ev.section is None

    def test_creation_with_values(self):
        """Test ExtractedValue with explicit values."""
        ev = ExtractedValue(
            value=0.014,
            confidence=ConfidenceLevel.EXPLICIT,
            citation="management fee of 1.4%",
            section="FEES AND EXPENSES",
        )
        assert ev.value == 0.014
        assert ev.confidence == ConfidenceLevel.EXPLICIT
        assert ev.citation == "management fee of 1.4%"

    def test_is_found(self):
        """Test is_found method."""
        ev_found = ExtractedValue(value=100, confidence=ConfidenceLevel.EXPLICIT)
        ev_not_found = ExtractedValue()
        ev_none_value = ExtractedValue(value=None, confidence=ConfidenceLevel.EXPLICIT)

        assert ev_found.is_found() is True
        assert ev_not_found.is_found() is False
        assert ev_none_value.is_found() is False

    def test_normalize_none_strings(self):
        """Test that 'none' strings are normalized to None."""
        ev1 = ExtractedValue(value="none")
        ev2 = ExtractedValue(value="N/A")
        ev3 = ExtractedValue(value="not applicable")

        assert ev1.value is None
        assert ev2.value is None
        assert ev3.value is None

    def test_normalize_numeric_strings(self):
        """Test that numeric strings are converted."""
        ev_int = ExtractedValue(value="5000")
        ev_float = ExtractedValue(value="1.4")
        ev_pct = ExtractedValue(value="25%")

        assert ev_int.value == 5000
        assert ev_float.value == 1.4
        assert ev_pct.value == 25.0

    def test_matches_exact(self):
        """Test exact value matching."""
        ev = ExtractedValue(value=0.014, confidence=ConfidenceLevel.EXPLICIT)

        assert ev.matches(0.014) is True
        assert ev.matches(0.02, tolerance=0.001) is False  # Outside tolerance
        assert ev.matches("0.014") is True

    def test_matches_with_tolerance(self):
        """Test numeric matching with tolerance."""
        ev = ExtractedValue(value=0.014, confidence=ConfidenceLevel.EXPLICIT)

        assert ev.matches(0.0141, tolerance=0.001) is True
        assert ev.matches(0.02, tolerance=0.001) is False

    def test_matches_strings(self):
        """Test string matching (case-insensitive)."""
        ev = ExtractedValue(value="quarterly", confidence=ConfidenceLevel.EXPLICIT)

        assert ev.matches("quarterly") is True
        assert ev.matches("QUARTERLY") is True
        assert ev.matches("Quarterly") is True
        assert ev.matches("monthly") is False

    def test_matches_none(self):
        """Test None matching."""
        ev_none = ExtractedValue(value=None, confidence=ConfidenceLevel.NOT_FOUND)
        ev_value = ExtractedValue(value=100, confidence=ConfidenceLevel.EXPLICIT)

        assert ev_none.matches(None) is True
        assert ev_none.matches(100) is False
        assert ev_value.matches(None) is False


class TestShareClassExtraction:
    """Tests for ShareClassExtraction schema."""

    def test_creation_with_defaults(self):
        """Test default ShareClassExtraction."""
        sc = ShareClassExtraction()
        assert sc.share_name.value is None
        assert sc.minimum_initial.value is None

    def test_creation_with_values(self):
        """Test ShareClassExtraction with values."""
        sc = ShareClassExtraction(
            share_name=ExtractedValue(value="Class I", confidence=ConfidenceLevel.EXPLICIT),
            minimum_initial=ExtractedValue(value=1000000, confidence=ConfidenceLevel.EXPLICIT),
            sales_load_percent=ExtractedValue(value=0, confidence=ConfidenceLevel.EXPLICIT),
        )
        assert sc.share_name.value == "Class I"
        assert sc.minimum_initial.value == 1000000
        assert sc.sales_load_percent.value == 0


class TestAllocationTarget:
    """Tests for AllocationTarget schema."""

    def test_creation(self):
        """Test AllocationTarget creation."""
        at = AllocationTarget(
            category_name="private equity",
            target_min=ExtractedValue(value=0.6, confidence=ConfidenceLevel.EXPLICIT),
            target_max=ExtractedValue(value=0.8, confidence=ConfidenceLevel.EXPLICIT),
        )
        assert at.category_name == "private equity"
        assert at.target_min.value == 0.6
        assert at.target_max.value == 0.8


class TestFundExtraction:
    """Tests for FundExtraction schema."""

    def test_creation_with_defaults(self):
        """Test default FundExtraction."""
        fe = FundExtraction()
        assert fe.fund_name.value is None
        assert fe.share_classes == []
        assert fe.investment_type_targets == []

    def test_get_field_simple(self):
        """Test get_field for simple fields."""
        fe = FundExtraction(
            fund_name=ExtractedValue(value="Test Fund", confidence=ConfidenceLevel.EXPLICIT),
        )
        result = fe.get_field("fund_name")
        assert result is not None
        assert result.value == "Test Fund"

    def test_get_field_nested(self):
        """Test get_field for nested fields."""
        sc = ShareClassExtraction(
            share_name=ExtractedValue(value="Class I", confidence=ConfidenceLevel.EXPLICIT),
        )
        fe = FundExtraction(share_classes=[sc])

        result = fe.get_field("share_classes.0.share_name")
        assert result is not None
        assert result.value == "Class I"

    def test_count_fields(self):
        """Test count_fields method."""
        fe = FundExtraction(
            fund_name=ExtractedValue(value="Test", confidence=ConfidenceLevel.EXPLICIT),
            fund_cik=ExtractedValue(value="0001234567", confidence=ConfidenceLevel.INFERRED),
            fund_type=ExtractedValue(),  # NOT_FOUND
        )
        counts = fe.count_fields()

        assert counts["explicit"] >= 1
        assert counts["inferred"] >= 1
        assert counts["not_found"] >= 1


class TestNormalization:
    """Tests for value normalization functions."""

    def test_normalize_value_none(self):
        """Test normalizing None values."""
        assert normalize_value(None) is None
        assert normalize_value("none") is None
        assert normalize_value("N/A") is None
        assert normalize_value("") is None

    def test_normalize_value_numeric(self):
        """Test normalizing numeric values."""
        assert normalize_value(100) == 100
        assert normalize_value("100") == 100
        assert normalize_value("1.5") == 1.5
        assert normalize_value("1,000") == 1000

    def test_normalize_value_string(self):
        """Test normalizing string values."""
        assert normalize_value("quarterly") == "quarterly"
        assert normalize_value("  quarterly  ") == "quarterly"

    def test_normalize_field_name(self):
        """Test field name normalization."""
        assert normalize_field_name("fund name") == "fund_name"
        assert normalize_field_name("management fee %") == "management_fee_pct"
        assert normalize_field_name("lock-up period (years)") == "lockup_period_years"


class TestValidateExtraction:
    """Tests for validate_extraction function."""

    def test_correct_match(self):
        """Test correct extraction matching."""
        ev = ExtractedValue(value=0.014, confidence=ConfidenceLevel.EXPLICIT)
        result = validate_extraction(ev, 0.014, "management_fee_pct")

        assert result.correct is True
        assert result.error_type == ErrorType.CORRECT

    def test_wrong_value(self):
        """Test wrong value detection."""
        ev = ExtractedValue(value=0.02, confidence=ConfidenceLevel.EXPLICIT)
        result = validate_extraction(ev, 0.014, "management_fee_pct")

        assert result.correct is False
        assert result.error_type == ErrorType.WRONG_VALUE

    def test_missed(self):
        """Test missed value detection."""
        ev = ExtractedValue()  # NOT_FOUND
        result = validate_extraction(ev, 0.014, "management_fee_pct")

        assert result.correct is False
        assert result.error_type == ErrorType.MISSED

    def test_hallucinated(self):
        """Test hallucination detection."""
        ev = ExtractedValue(value=0.014, confidence=ConfidenceLevel.EXPLICIT)
        result = validate_extraction(ev, None, "management_fee_pct")

        assert result.correct is False
        assert result.error_type == ErrorType.HALLUCINATED

    def test_true_negative(self):
        """Test true negative (both None)."""
        ev = ExtractedValue()  # NOT_FOUND
        result = validate_extraction(ev, None, "management_fee_pct")

        assert result.correct is True
        assert result.error_type == ErrorType.CORRECT


@pytest.mark.skipif(not EXCEL_PATH.exists(), reason="Validation Excel file not found")
class TestLoadGroundTruth:
    """Tests for loading ground truth from Excel."""

    def test_load_stepstone(self):
        """Test loading StepStone ground truth."""
        gt = load_ground_truth(EXCEL_PATH, "stepstone")

        assert gt.fund_name == "Stepstone"
        assert gt.cik == "1789470"
        assert len(gt.datapoints) > 0

        # Check some expected fields
        assert "fund_name" in gt.datapoints
        assert "management_fee_pct" in gt.datapoints

    def test_load_blackstone(self):
        """Test loading Blackstone ground truth."""
        gt = load_ground_truth(EXCEL_PATH, "blackstone")

        assert gt.fund_name == "Blackstone"
        assert gt.cik == "2032432"
        assert len(gt.datapoints) > 0

    def test_stepstone_values(self):
        """Test specific StepStone values."""
        gt = load_ground_truth(EXCEL_PATH, "stepstone")

        # Check management fee (1.4%)
        assert gt.datapoints.get("management_fee_pct") == 0.014

        # Check repurchase frequency
        assert gt.datapoints.get("repurchase_frequency") == "quarterly"

        # Check number of share types
        assert gt.datapoints.get("number_of_share_types") == 3

    def test_share_class_loading(self):
        """Test share class data is loaded correctly."""
        gt = load_ground_truth(EXCEL_PATH, "stepstone")

        # Check share class 1 (Class S)
        assert "share_class_1.share_name" in gt.datapoints
        assert gt.datapoints.get("share_class_1.share_name") == "Class S"

        # Check share class 3 (Class I)
        assert "share_class_3.share_name" in gt.datapoints
        assert gt.datapoints.get("share_class_3.minimum_initial") == 1000000

    def test_allocation_target_loading(self):
        """Test allocation target data is loaded correctly."""
        gt = load_ground_truth(EXCEL_PATH, "stepstone")

        # Check investment type allocations
        assert "investment_type.secondary_funds.target_min" in gt.datapoints
        assert gt.datapoints.get("investment_type.secondary_funds.target_min") == 0.4


class TestValidateFund:
    """Tests for full fund validation."""

    @pytest.mark.skipif(not EXCEL_PATH.exists(), reason="Validation Excel file not found")
    def test_validate_empty_extraction(self):
        """Test validating empty extraction against ground truth."""
        gt = load_ground_truth(EXCEL_PATH, "stepstone")
        extraction = create_empty_extraction()

        results = validate_fund(extraction, gt)

        assert len(results) > 0
        # All should be either MISSED (if expected exists) or CORRECT (if expected is None)
        for r in results:
            if r.expected is not None:
                assert r.error_type == ErrorType.MISSED

    @pytest.mark.skipif(not EXCEL_PATH.exists(), reason="Validation Excel file not found")
    def test_validate_mock_perfect(self):
        """Test validating perfect mock extraction."""
        gt = load_ground_truth(EXCEL_PATH, "stepstone")
        extraction = create_mock_extraction(gt, accuracy=1.0)

        results = validate_fund(extraction, gt)

        # Count correct results
        correct_count = sum(1 for r in results if r.correct)
        total_count = len(results)

        # Should be very high accuracy (some fields may not be filled)
        assert correct_count / total_count > 0.8

    @pytest.mark.skipif(not EXCEL_PATH.exists(), reason="Validation Excel file not found")
    def test_validate_mock_partial(self):
        """Test validating partial mock extraction."""
        gt = load_ground_truth(EXCEL_PATH, "stepstone")
        extraction = create_mock_extraction(gt, accuracy=0.5)

        results = validate_fund(extraction, gt)

        # Should have mix of correct and errors
        correct_count = sum(1 for r in results if r.correct)
        total_count = len(results)

        # Accuracy should be roughly 50%
        assert 0.3 < correct_count / total_count < 0.7


class TestComputeMetrics:
    """Tests for metrics computation."""

    def test_compute_all_correct(self):
        """Test metrics for all correct results."""
        results = [
            ValidationResult(
                field_name="field1", category="cat1",
                expected=100, extracted=100,
                correct=True, error_type=ErrorType.CORRECT,
                confidence=ConfidenceLevel.EXPLICIT,
            ),
            ValidationResult(
                field_name="field2", category="cat1",
                expected="test", extracted="test",
                correct=True, error_type=ErrorType.CORRECT,
                confidence=ConfidenceLevel.EXPLICIT,
            ),
        ]
        metrics = compute_metrics(results)

        assert metrics.total_fields == 2
        assert metrics.correct_fields == 2
        assert metrics.accuracy == 1.0
        assert metrics.error_count == 0

    def test_compute_mixed_results(self):
        """Test metrics for mixed results."""
        results = [
            ValidationResult(
                field_name="correct_field", category="fees",
                expected=100, extracted=100,
                correct=True, error_type=ErrorType.CORRECT,
                confidence=ConfidenceLevel.EXPLICIT,
            ),
            ValidationResult(
                field_name="wrong_field", category="fees",
                expected=100, extracted=200,
                correct=False, error_type=ErrorType.WRONG_VALUE,
                confidence=ConfidenceLevel.EXPLICIT,
            ),
            ValidationResult(
                field_name="missed_field", category="repurchase",
                expected=100, extracted=None,
                correct=False, error_type=ErrorType.MISSED,
                confidence=ConfidenceLevel.NOT_FOUND,
            ),
        ]
        metrics = compute_metrics(results)

        assert metrics.total_fields == 3
        assert metrics.correct_fields == 1
        assert metrics.wrong_value_count == 1
        assert metrics.missed_count == 1
        assert metrics.hallucinated_count == 0
        assert abs(metrics.accuracy - 0.333) < 0.01

    def test_metrics_by_category(self):
        """Test metrics breakdown by category."""
        results = [
            ValidationResult(
                field_name="fee1", category="fees",
                expected=100, extracted=100,
                correct=True, error_type=ErrorType.CORRECT,
                confidence=ConfidenceLevel.EXPLICIT,
            ),
            ValidationResult(
                field_name="fee2", category="fees",
                expected=200, extracted=200,
                correct=True, error_type=ErrorType.CORRECT,
                confidence=ConfidenceLevel.EXPLICIT,
            ),
            ValidationResult(
                field_name="repurchase1", category="repurchase",
                expected=100, extracted=50,
                correct=False, error_type=ErrorType.WRONG_VALUE,
                confidence=ConfidenceLevel.EXPLICIT,
            ),
        ]
        metrics = compute_metrics(results)

        assert "fees" in metrics.category_metrics
        assert metrics.category_metrics["fees"].accuracy == 1.0
        assert metrics.category_metrics["repurchase"].accuracy == 0.0

    def test_metrics_by_confidence(self):
        """Test metrics breakdown by confidence level."""
        results = [
            ValidationResult(
                field_name="explicit_correct", category="test",
                expected=100, extracted=100,
                correct=True, error_type=ErrorType.CORRECT,
                confidence=ConfidenceLevel.EXPLICIT,
            ),
            ValidationResult(
                field_name="inferred_wrong", category="test",
                expected=100, extracted=50,
                correct=False, error_type=ErrorType.WRONG_VALUE,
                confidence=ConfidenceLevel.INFERRED,
            ),
        ]
        metrics = compute_metrics(results)

        assert metrics.confidence_metrics["explicit"].accuracy == 1.0
        assert metrics.confidence_metrics["inferred"].accuracy == 0.0

    def test_worst_fields(self):
        """Test identification of worst performing fields."""
        results = [
            ValidationResult(
                field_name="good_field", category="test",
                expected=100, extracted=100,
                correct=True, error_type=ErrorType.CORRECT,
                confidence=ConfidenceLevel.EXPLICIT,
            ),
            ValidationResult(
                field_name="bad_field", category="test",
                expected=100, extracted=50,
                correct=False, error_type=ErrorType.WRONG_VALUE,
                confidence=ConfidenceLevel.EXPLICIT,
            ),
        ]
        metrics = compute_metrics(results)

        worst = metrics.get_worst_fields(5)
        assert len(worst) == 2
        assert worst[0][0] == "bad_field"
        assert worst[0][1] == 0.0


class TestPrintMetricsReport:
    """Tests for report generation."""

    def test_print_report(self):
        """Test report generation."""
        results = [
            ValidationResult(
                field_name="test_field", category="test",
                expected=100, extracted=100,
                correct=True, error_type=ErrorType.CORRECT,
                confidence=ConfidenceLevel.EXPLICIT,
            ),
        ]
        metrics = compute_metrics(results)
        report = print_metrics_report(metrics)

        assert "EXTRACTION VALIDATION REPORT" in report
        assert "OVERALL ACCURACY" in report
        assert "100.0%" in report


class TestCompareMetrics:
    """Tests for metrics comparison."""

    def test_compare_improvement(self):
        """Test comparing improved metrics."""
        baseline_results = [
            ValidationResult("f1", "c1", 100, 50, False, ErrorType.WRONG_VALUE, ConfidenceLevel.EXPLICIT),
            ValidationResult("f2", "c1", 100, None, False, ErrorType.MISSED, ConfidenceLevel.NOT_FOUND),
        ]
        current_results = [
            ValidationResult("f1", "c1", 100, 100, True, ErrorType.CORRECT, ConfidenceLevel.EXPLICIT),
            ValidationResult("f2", "c1", 100, 100, True, ErrorType.CORRECT, ConfidenceLevel.EXPLICIT),
        ]

        baseline = compute_metrics(baseline_results)
        current = compute_metrics(current_results)

        comparison = compare_metrics(baseline, current)

        assert comparison["accuracy_delta"] == 1.0  # 0% -> 100%
        assert comparison["errors_reduced"] == 2


@pytest.mark.skipif(not EXCEL_PATH.exists(), reason="Validation Excel file not found")
class TestEndToEnd:
    """End-to-end validation tests."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        # 1. Load ground truth
        gt_stepstone = load_ground_truth(EXCEL_PATH, "stepstone")
        gt_blackstone = load_ground_truth(EXCEL_PATH, "blackstone")

        assert len(gt_stepstone.datapoints) > 50
        assert len(gt_blackstone.datapoints) > 50

        # 2. Create mock extractions
        extraction_stepstone = create_mock_extraction(gt_stepstone, accuracy=0.8)
        extraction_blackstone = create_mock_extraction(gt_blackstone, accuracy=0.8)

        # 3. Validate
        results_stepstone = validate_fund(extraction_stepstone, gt_stepstone)
        results_blackstone = validate_fund(extraction_blackstone, gt_blackstone)

        # 4. Compute metrics
        metrics_stepstone = compute_metrics(results_stepstone)
        metrics_blackstone = compute_metrics(results_blackstone)

        # 5. Print reports
        report_stepstone = print_metrics_report(metrics_stepstone, "StepStone Validation")
        report_blackstone = print_metrics_report(metrics_blackstone, "Blackstone Validation")

        assert "StepStone Validation" in report_stepstone
        assert "Blackstone Validation" in report_blackstone

        # 6. Verify reasonable accuracy
        assert metrics_stepstone.accuracy > 0.6
        assert metrics_blackstone.accuracy > 0.6

    def test_datapoint_counts(self):
        """Verify we have expected number of datapoints."""
        gt_stepstone = load_ground_truth(EXCEL_PATH, "stepstone")
        gt_blackstone = load_ground_truth(EXCEL_PATH, "blackstone")

        # StepStone has ~93 rows, Blackstone has ~88 rows
        # After normalization, we should have significant datapoints
        total_datapoints = len(gt_stepstone.datapoints) + len(gt_blackstone.datapoints)

        print(f"\nTotal datapoints: {total_datapoints}")
        print(f"  StepStone: {len(gt_stepstone.datapoints)}")
        print(f"  Blackstone: {len(gt_blackstone.datapoints)}")

        # Should have at least 100 datapoints total
        assert total_datapoints >= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
