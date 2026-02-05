"""
Grounding validation for LLM extractions.

Verifies that extracted values actually appear in the source text,
helping catch hallucinations and omissions.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class GroundingResult:
    """Result of grounding validation for a single field."""
    field_name: str
    is_grounded: bool
    grounding_score: float  # 0.0 to 1.0
    issues: list[str] = field(default_factory=list)
    verified_values: list[str] = field(default_factory=list)
    unverified_values: list[str] = field(default_factory=list)


@dataclass
class GroundingReport:
    """Complete grounding report for an extraction."""
    total_fields: int
    grounded_fields: int
    grounding_rate: float
    field_results: dict[str, GroundingResult] = field(default_factory=dict)
    overall_issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_fields": self.total_fields,
            "grounded_fields": self.grounded_fields,
            "grounding_rate": self.grounding_rate,
            "field_results": {
                k: {
                    "is_grounded": v.is_grounded,
                    "grounding_score": v.grounding_score,
                    "issues": v.issues,
                    "verified_values": v.verified_values,
                    "unverified_values": v.unverified_values,
                }
                for k, v in self.field_results.items()
            },
            "overall_issues": self.overall_issues,
        }


class GroundingValidator:
    """
    Validates that LLM extractions are grounded in source text.

    Checks:
    1. Numeric values appear in the source
    2. Evidence quotes are verbatim (fuzzy match)
    3. Key identifiers (class names, etc.) appear in source
    """

    # Numeric patterns to search for
    NUMERIC_PATTERNS = [
        r"{value}%",              # 25%
        r"{value} percent",        # 25 percent
        r"{value}",                # Just the number
    ]

    def __init__(self, fuzzy_threshold: float = 0.8):
        """
        Initialize validator.

        Args:
            fuzzy_threshold: Minimum similarity for quote matching (0-1)
        """
        self.fuzzy_threshold = fuzzy_threshold

    def validate_extraction(
        self,
        extraction: dict,
        source_text: str,
    ) -> GroundingReport:
        """
        Validate all fields in an extraction against source text.

        Args:
            extraction: The extraction result dict
            source_text: The original document text

        Returns:
            GroundingReport with validation results
        """
        source_lower = source_text.lower()
        field_results = {}

        # Validate each extracted field
        for field_name, field_value in extraction.items():
            if field_value is None:
                continue
            if field_name in ["filing_id", "cik", "fund_name", "xbrl_fees",
                             "chunks_processed", "extraction_errors"]:
                continue

            result = self._validate_field(field_name, field_value, source_text, source_lower)
            if result:
                field_results[field_name] = result

        # Calculate overall grounding
        if field_results:
            grounded = sum(1 for r in field_results.values() if r.is_grounded)
            total = len(field_results)
            rate = grounded / total if total > 0 else 0
        else:
            grounded = 0
            total = 0
            rate = 0

        return GroundingReport(
            total_fields=total,
            grounded_fields=grounded,
            grounding_rate=rate,
            field_results=field_results,
        )

    def _validate_field(
        self,
        field_name: str,
        field_value: Any,
        source_text: str,
        source_lower: str,
    ) -> Optional[GroundingResult]:
        """Validate a single field's extraction."""

        if not isinstance(field_value, dict):
            return None

        issues = []
        verified = []
        unverified = []

        # Check evidence quote if present
        citation = field_value.get("citation", {})
        if isinstance(citation, dict):
            evidence = citation.get("evidence_quote", "")
            if evidence:
                if self._find_quote_in_text(evidence, source_lower):
                    verified.append(f"quote: {evidence[:50]}...")
                else:
                    # Try partial match
                    if self._find_partial_quote(evidence, source_lower):
                        verified.append(f"partial_quote: {evidence[:50]}...")
                    else:
                        issues.append(f"Evidence quote not found in source")
                        unverified.append(f"quote: {evidence[:50]}...")

        # Check numeric values based on field type
        if field_name == "repurchase_terms":
            self._validate_repurchase(field_value, source_lower, verified, unverified, issues)
        elif field_name == "share_classes":
            self._validate_share_classes(field_value, source_lower, verified, unverified, issues)
        elif field_name == "allocation_targets":
            self._validate_allocations(field_value, source_lower, verified, unverified, issues)
        elif field_name == "concentration_limits":
            self._validate_concentration(field_value, source_lower, verified, unverified, issues)
        elif field_name == "incentive_fee":
            self._validate_incentive_fee(field_value, source_lower, verified, unverified, issues)
        elif field_name == "expense_cap":
            self._validate_expense_cap(field_value, source_lower, verified, unverified, issues)

        # Calculate grounding score
        total_checks = len(verified) + len(unverified)
        if total_checks > 0:
            score = len(verified) / total_checks
        else:
            score = 0.5  # No specific values to check

        is_grounded = score >= 0.5 and len(issues) == 0

        return GroundingResult(
            field_name=field_name,
            is_grounded=is_grounded,
            grounding_score=score,
            issues=issues,
            verified_values=verified,
            unverified_values=unverified,
        )

    def _validate_repurchase(self, value: dict, source: str, verified: list, unverified: list, issues: list):
        """Validate repurchase terms extraction."""
        # Check percentage values
        for pct_field in ["repurchase_percentage_min", "repurchase_percentage_max",
                          "repurchase_percentage_typical", "early_repurchase_fee"]:
            pct = value.get(pct_field)
            if pct is not None:
                if self._find_numeric_in_text(pct, source):
                    verified.append(f"{pct_field}: {pct}%")
                else:
                    unverified.append(f"{pct_field}: {pct}%")

        # Check notice period
        notice = value.get("notice_period_days")
        if notice is not None:
            if self._find_numeric_in_text(notice, source) or f"{notice} day" in source or f"{notice} calendar" in source:
                verified.append(f"notice_period: {notice} days")
            else:
                unverified.append(f"notice_period: {notice} days")

        # Check fund structure keywords
        structure = value.get("fund_structure", "")
        if structure == "interval_fund":
            if "interval" in source:
                verified.append("fund_structure: interval")
            else:
                issues.append("Extracted 'interval_fund' but 'interval' not found in source")
                unverified.append("fund_structure: interval")
        elif structure == "tender_offer_fund":
            if "tender" in source:
                verified.append("fund_structure: tender_offer")
            else:
                unverified.append("fund_structure: tender_offer")

    def _validate_share_classes(self, value: dict, source: str, verified: list, unverified: list, issues: list):
        """Validate share classes extraction."""
        classes = value.get("share_classes", [])
        for sc in classes:
            class_name = sc.get("class_name", "")
            if class_name:
                # Check if class name appears in source
                if class_name.lower() in source:
                    verified.append(f"class_name: {class_name}")
                else:
                    unverified.append(f"class_name: {class_name}")

            # Check fees
            for fee_field in ["sales_load_pct", "distribution_servicing_fee_pct"]:
                fee = sc.get(fee_field)
                if fee is not None and fee != 0:
                    if self._find_numeric_in_text(fee, source):
                        verified.append(f"{class_name}.{fee_field}: {fee}%")
                    else:
                        unverified.append(f"{class_name}.{fee_field}: {fee}%")

            # Check minimum investment
            min_inv = sc.get("minimum_initial_investment")
            if min_inv is not None:
                # Format min_inv for display (handle both numeric and string values)
                try:
                    min_inv_display = f"${int(min_inv):,}" if isinstance(min_inv, (int, float)) else f"${min_inv}"
                except (ValueError, TypeError):
                    min_inv_display = f"${min_inv}"

                # Format variations: $1,000,000 or 1000000 or 1,000,000
                if self._find_numeric_in_text(min_inv, source, is_currency=True):
                    verified.append(f"{class_name}.minimum: {min_inv_display}")
                else:
                    unverified.append(f"{class_name}.minimum: {min_inv_display}")

    def _validate_allocations(self, value: dict, source: str, verified: list, unverified: list, issues: list):
        """Validate allocation targets extraction."""
        allocations = value.get("allocations", [])
        for alloc in allocations:
            asset_class = alloc.get("asset_class", "")
            if asset_class:
                # Check if asset class appears in source
                if asset_class.lower() in source or self._fuzzy_match_term(asset_class, source):
                    verified.append(f"asset_class: {asset_class}")
                else:
                    unverified.append(f"asset_class: {asset_class}")

            # Check percentage values
            for pct_field in ["target_percentage", "range_min", "range_max"]:
                pct = alloc.get(pct_field)
                if pct is not None:
                    if self._find_numeric_in_text(pct, source):
                        verified.append(f"{asset_class}.{pct_field}: {pct}%")
                    else:
                        unverified.append(f"{asset_class}.{pct_field}: {pct}%")

    def _validate_concentration(self, value: dict, source: str, verified: list, unverified: list, issues: list):
        """Validate concentration limits extraction."""
        limits = value.get("limits", [])
        for limit in limits:
            pct = limit.get("limit_percentage")
            if pct is not None:
                if self._find_numeric_in_text(pct, source):
                    verified.append(f"limit: {pct}%")
                else:
                    unverified.append(f"limit: {pct}%")

        # Check for "no limit" language if has_concentration_limits is False
        if value.get("has_concentration_limits") is False:
            no_limit_phrases = ["no limit", "no restriction", "without limit", "unlimited"]
            if any(phrase in source for phrase in no_limit_phrases):
                verified.append("no_limit_language: found")
            else:
                issues.append("Extracted 'no limits' but no such language found in source")

    def _validate_incentive_fee(self, value: dict, source: str, verified: list, unverified: list, issues: list):
        """Validate incentive fee extraction."""
        if value.get("has_incentive_fee") is False:
            # Check for negative language
            if "does not" in source and "incentive" in source:
                verified.append("no_incentive_fee: confirmed")
            elif "no performance" in source or "no incentive" in source:
                verified.append("no_incentive_fee: confirmed")
        else:
            fee_rate = value.get("incentive_fee_rate")
            if fee_rate is not None:
                if self._find_numeric_in_text(fee_rate, source):
                    verified.append(f"incentive_fee_rate: {fee_rate}%")
                else:
                    unverified.append(f"incentive_fee_rate: {fee_rate}%")

            hurdle = value.get("hurdle_rate")
            if hurdle is not None:
                if self._find_numeric_in_text(hurdle, source):
                    verified.append(f"hurdle_rate: {hurdle}%")
                else:
                    unverified.append(f"hurdle_rate: {hurdle}%")

    def _validate_expense_cap(self, value: dict, source: str, verified: list, unverified: list, issues: list):
        """Validate expense cap extraction."""
        cap_rate = value.get("cap_rate")
        if cap_rate is not None:
            if self._find_numeric_in_text(cap_rate, source):
                verified.append(f"cap_rate: {cap_rate}%")
            else:
                unverified.append(f"cap_rate: {cap_rate}%")

        # Check for waiver language
        if value.get("has_fee_waiver"):
            if "waive" in source or "waiver" in source:
                verified.append("fee_waiver_language: found")
            else:
                unverified.append("fee_waiver_language: not found")

    def _find_numeric_in_text(self, value: Any, text: str, is_currency: bool = False) -> bool:
        """Check if a numeric value appears in the text."""
        if value is None:
            return False

        try:
            num = float(value)
        except (ValueError, TypeError):
            return False

        # Try different formats
        formats = [
            f"{num}%",
            f"{num:.0f}%",
            f"{num:.1f}%",
            f"{num:.2f}%",
            f"{num} percent",
            f"{int(num)}",
        ]

        if is_currency:
            # Add currency formats
            formats.extend([
                f"${num:,.0f}",
                f"${num:,}",
                f"{num:,.0f}",
                f"{num:,}",
            ])

        for fmt in formats:
            if fmt.lower() in text:
                return True

        return False

    def _find_quote_in_text(self, quote: str, text: str, min_length: int = 20) -> bool:
        """Check if a quote appears in the text (case-insensitive)."""
        if len(quote) < min_length:
            return quote.lower() in text

        # Normalize whitespace
        quote_norm = " ".join(quote.lower().split())
        text_norm = " ".join(text.split())

        return quote_norm in text_norm

    def _find_partial_quote(self, quote: str, text: str, min_words: int = 5) -> bool:
        """Check if significant portions of a quote appear in text."""
        words = quote.lower().split()
        if len(words) < min_words:
            return quote.lower() in text

        # Check for consecutive word sequences
        for i in range(len(words) - min_words + 1):
            phrase = " ".join(words[i:i + min_words])
            if phrase in text:
                return True

        return False

    def _fuzzy_match_term(self, term: str, text: str) -> bool:
        """Check if a term appears in text with some flexibility."""
        term_lower = term.lower()

        # Try exact match
        if term_lower in text:
            return True

        # Try individual significant words
        words = [w for w in term_lower.split() if len(w) > 3]
        if len(words) >= 2:
            matches = sum(1 for w in words if w in text)
            return matches >= len(words) * 0.5

        return False


def validate_extraction(extraction: dict, source_text: str) -> GroundingReport:
    """
    Convenience function to validate an extraction.

    Args:
        extraction: The extraction result dict
        source_text: The original document text

    Returns:
        GroundingReport with validation results
    """
    validator = GroundingValidator()
    return validator.validate_extraction(extraction, source_text)


def correct_ungrounded_share_class_minimums(
    extraction: dict,
    source_text: str,
) -> tuple[dict, list[str]]:
    """
    Correct share class minimum investments that are not grounded in source text.

    This is a CRITICAL anti-hallucination fix. LLMs often hallucinate typical minimum
    investment values (e.g., $2,500, $1,000,000) from training data even when the
    document doesn't contain these values.

    Args:
        extraction: The extraction result dict (will be modified in place)
        source_text: The original document text

    Returns:
        Tuple of (modified extraction, list of corrections made)
    """
    corrections = []
    validator = GroundingValidator()
    source_lower = source_text.lower()

    share_classes_data = extraction.get("share_classes", {})
    if not share_classes_data:
        return extraction, corrections

    classes = share_classes_data.get("share_classes", [])
    if not classes:
        return extraction, corrections

    for sc in classes:
        class_name = sc.get("class_name", "unknown")

        # Check minimum_initial_investment
        min_initial = sc.get("minimum_initial_investment")
        if min_initial is not None:
            if not validator._find_numeric_in_text(min_initial, source_lower, is_currency=True):
                corrections.append(
                    f"{class_name}.minimum_initial_investment: {min_initial} -> null "
                    f"(value not found in source text - likely hallucinated)"
                )
                sc["minimum_initial_investment"] = None
                logger.warning(
                    f"[Grounding] Corrected hallucinated minimum_initial_investment "
                    f"for {class_name}: {min_initial} -> null"
                )

        # Check minimum_additional_investment
        min_additional = sc.get("minimum_additional_investment")
        if min_additional is not None:
            if not validator._find_numeric_in_text(min_additional, source_lower, is_currency=True):
                corrections.append(
                    f"{class_name}.minimum_additional_investment: {min_additional} -> null "
                    f"(value not found in source text - likely hallucinated)"
                )
                sc["minimum_additional_investment"] = None
                logger.warning(
                    f"[Grounding] Corrected hallucinated minimum_additional_investment "
                    f"for {class_name}: {min_additional} -> null"
                )

    if corrections:
        logger.info(f"[Grounding] Corrected {len(corrections)} hallucinated share class minimums")

    return extraction, corrections
