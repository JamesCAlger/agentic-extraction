"""
Post-extraction validation rules for extraction quality improvement.

This module applies deterministic rules to clean up extraction results,
fixing common errors caused by:
1. Schema mismatch (e.g., direct credit fund vs fund-of-funds)
2. Cross-field inconsistencies (e.g., hurdle rate without incentive fee)
3. Hallucinated values that don't match fund type
4. Field disambiguation errors (e.g., distribution_frequency vs repurchase_frequency)
5. Semantic synonyms (e.g., "asset coverage" = "asset_coverage")
"""

import logging
import re
from typing import Any, Optional
from dataclasses import dataclass, field

from .fund_classifier import (
    FundStrategy,
    get_null_fields_for_strategy,
    get_nullable_fields_for_strategy,
)

logger = logging.getLogger(__name__)


# =============================================================================
# FIELD DISAMBIGUATION SYNONYMS
# =============================================================================
# Maps various phrasings to canonical values for consistent extraction/evaluation

# Leverage basis synonyms - maps document language to schema enum values
LEVERAGE_BASIS_SYNONYMS = {
    # Asset coverage variants (300% coverage = 33.33% debt-to-assets)
    "asset coverage": "asset_coverage",
    "asset coverage ratio": "asset_coverage",
    "300%": "asset_coverage",
    "300% coverage": "asset_coverage",
    "300 percent": "asset_coverage",
    "1940 act": "asset_coverage",
    "investment company act": "asset_coverage",
    "section 18": "asset_coverage",
    # Total assets variants
    "total assets": "total_assets",
    "gross assets": "total_assets",
    "total fund assets": "total_assets",
    # Net assets variants
    "net assets": "net_assets",
    "net asset value": "net_assets",
    "nav": "net_assets",
    # Debt-to-equity variants
    "debt to equity": "debt_to_equity",
    "debt-to-equity": "debt_to_equity",
    "debt/equity": "debt_to_equity",
    "leverage ratio": "debt_to_equity",
}

# Repurchase basis synonyms - what the repurchase % is calculated against
REPURCHASE_BASIS_SYNONYMS = {
    # Outstanding shares variants
    "outstanding shares": "shares",
    "number of shares": "shares",
    "shares outstanding": "shares",
    "total shares": "shares",
    "of shares": "shares",
    # NAV / net assets variants
    "net assets": "nav",
    "net asset value": "nav",
    "of nav": "nav",
    # Gross assets
    "total assets": "total_assets",
    "gross assets": "total_assets",
}

# Distribution frequency synonyms - how often distributions are paid
DISTRIBUTION_FREQUENCY_SYNONYMS = {
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
    # Annual variants
    "annually": "annually",
    "annual": "annually",
    "yearly": "annually",
    "once per year": "annually",
    "paid annually": "annually",
    # Daily (for accrual purposes)
    "daily": "daily",
    "each day": "daily",
    "per day": "daily",
    # Semi-annual
    "semi-annually": "semi_annually",
    "semi-annual": "semi_annually",
    "twice per year": "semi_annually",
    "two times per year": "semi_annually",
}

# Repurchase frequency synonyms
REPURCHASE_FREQUENCY_SYNONYMS = {
    # Quarterly variants
    "quarterly": "quarterly",
    "each quarter": "quarterly",
    "per quarter": "quarterly",
    "every quarter": "quarterly",
    "four times per year": "quarterly",
    "quarterly basis": "quarterly",
    # Monthly variants
    "monthly": "monthly",
    "each month": "monthly",
    # Annual variants
    "annually": "annually",
    "annual": "annually",
    "yearly": "annually",
    # Semi-annual
    "semi-annually": "semi_annually",
    "semi-annual": "semi_annually",
}

# Hurdle rate frequency synonyms
HURDLE_RATE_FREQUENCY_SYNONYMS = {
    # Quarterly variants (stated rate is per quarter)
    "quarterly": "quarterly",
    "per quarter": "quarterly",
    "each quarter": "quarterly",
    "calendar quarter": "quarterly",
    "quarterly basis": "quarterly",
    # Annual variants (stated rate is per year)
    "annually": "annual",
    "annual": "annual",
    "per annum": "annual",
    "per year": "annual",
    "annualized": "annual",
    "yearly": "annual",
}

# Distribution policy synonyms
DISTRIBUTION_POLICY_SYNONYMS = {
    # DRIP / reinvested
    "reinvested": "drip",
    "reinvestment": "drip",
    "drip": "drip",
    "dividend reinvestment": "drip",
    "dividend reinvestment plan": "drip",
    "automatically reinvested": "drip",
    "reinvest": "drip",
    # Cash
    "cash": "cash",
    "cash distribution": "cash",
    "paid in cash": "cash",
}


def normalize_leverage_basis(value: Optional[str]) -> Optional[str]:
    """
    Normalize leverage_basis value to canonical form.

    Args:
        value: Raw leverage_basis value from extraction

    Returns:
        Canonical leverage_basis value or original if not recognized
    """
    if not value:
        return value

    # Normalize underscores to spaces and lowercase
    value_lower = value.lower().strip().replace("_", " ")

    # Direct match
    if value_lower in LEVERAGE_BASIS_SYNONYMS:
        return LEVERAGE_BASIS_SYNONYMS[value_lower]

    # Partial match - check if any synonym is contained in value
    for synonym, canonical in LEVERAGE_BASIS_SYNONYMS.items():
        if synonym in value_lower:
            return canonical

    return value


def normalize_repurchase_basis(value: Optional[str]) -> Optional[str]:
    """
    Normalize repurchase_basis value to canonical form.

    Args:
        value: Raw repurchase_basis value from extraction

    Returns:
        Canonical repurchase_basis value or original if not recognized
    """
    if not value:
        return value

    # Normalize underscores to spaces and lowercase
    value_lower = value.lower().strip().replace("_", " ")

    # Direct match
    if value_lower in REPURCHASE_BASIS_SYNONYMS:
        return REPURCHASE_BASIS_SYNONYMS[value_lower]

    # Partial match
    for synonym, canonical in REPURCHASE_BASIS_SYNONYMS.items():
        if synonym in value_lower:
            return canonical

    return value


def normalize_frequency(value: Optional[str], field_type: str = "distribution") -> Optional[str]:
    """
    Normalize frequency values (distribution, repurchase, hurdle rate).

    Args:
        value: Raw frequency value from extraction
        field_type: Type of frequency ("distribution", "repurchase", "hurdle")

    Returns:
        Canonical frequency value or original if not recognized
    """
    if not value:
        return value

    # Normalize underscores to spaces and lowercase
    value_lower = value.lower().strip().replace("_", " ")

    synonyms = {
        "distribution": DISTRIBUTION_FREQUENCY_SYNONYMS,
        "repurchase": REPURCHASE_FREQUENCY_SYNONYMS,
        "hurdle": HURDLE_RATE_FREQUENCY_SYNONYMS,
    }.get(field_type, DISTRIBUTION_FREQUENCY_SYNONYMS)

    # Direct match
    if value_lower in synonyms:
        return synonyms[value_lower]

    # Partial match
    for synonym, canonical in synonyms.items():
        if synonym in value_lower:
            return canonical

    return value


def normalize_distribution_policy(value: Optional[str]) -> Optional[str]:
    """
    Normalize distribution_policy value (cash vs drip).

    Args:
        value: Raw distribution_policy value from extraction

    Returns:
        Canonical distribution_policy value ("cash" or "drip")
    """
    if not value:
        return value

    # Normalize underscores to spaces and lowercase
    value_lower = value.lower().strip().replace("_", " ")

    if value_lower in DISTRIBUTION_POLICY_SYNONYMS:
        return DISTRIBUTION_POLICY_SYNONYMS[value_lower]

    for synonym, canonical in DISTRIBUTION_POLICY_SYNONYMS.items():
        if synonym in value_lower:
            return canonical

    return value


@dataclass
class ValidationCorrection:
    """Record of a correction made by validation rules."""
    field_path: str
    original_value: Any
    corrected_value: Any
    rule_name: str
    reason: str


@dataclass
class ValidationReport:
    """Report of all validation corrections applied."""
    fund_name: str
    fund_strategy: Optional[str]
    corrections: list[ValidationCorrection] = field(default_factory=list)

    @property
    def correction_count(self) -> int:
        return len(self.corrections)

    def to_dict(self) -> dict:
        return {
            "fund_name": self.fund_name,
            "fund_strategy": self.fund_strategy,
            "correction_count": self.correction_count,
            "corrections": [
                {
                    "field_path": c.field_path,
                    "original_value": c.original_value,
                    "corrected_value": c.corrected_value,
                    "rule_name": c.rule_name,
                    "reason": c.reason,
                }
                for c in self.corrections
            ],
        }


# =============================================================================
# FUND STRATEGY DETECTION
# =============================================================================

def detect_fund_strategy(extraction: dict) -> str:
    """
    Detect fund strategy from extraction results.

    Returns one of:
    - "fund_of_funds": Invests in other PE/VC funds (StepStone, Hamilton Lane, Carlyle)
    - "direct_credit": Makes direct loans to companies (Blue Owl)
    - "direct_equity": Makes direct equity investments
    - "hybrid": Mix of direct and fund investments (Blackstone)
    - "unknown": Cannot determine
    """
    # Check allocation targets for fund-of-funds indicators
    allocation = extraction.get("allocation_targets", {})
    if allocation:
        allocations = allocation.get("allocations", [])
        for alloc in allocations:
            asset_class = (alloc.get("asset_class") or "").lower()
            # Fund-of-funds indicators
            if any(term in asset_class for term in [
                "primary fund", "secondary fund", "co-investment",
                "fund investment", "portfolio fund", "underlying fund"
            ]):
                return "fund_of_funds"

        # Check for secondary_funds fields (fund-of-funds specific)
        if allocation.get("secondary_funds_min_pct") is not None:
            return "fund_of_funds"
        if allocation.get("secondary_funds_max_pct") is not None:
            return "fund_of_funds"

    # Check incentive fee for underlying fund range (fund-of-funds indicator)
    incentive_fee = extraction.get("incentive_fee", {})
    if incentive_fee:
        if incentive_fee.get("underlying_fund_incentive_range") is not None:
            return "fund_of_funds"

    # Check fund name for strategy hints
    fund_name = (extraction.get("fund_name") or "").lower()

    # Direct credit indicators
    if any(term in fund_name for term in [
        "credit", "lending", "loan", "debt", "income",
    ]):
        # But exclude "private credit fund-of-funds"
        if "fund" in fund_name and any(term in fund_name for term in ["multi", "diversified", "portfolio"]):
            return "hybrid"
        return "direct_credit"

    # Fund-of-funds name indicators
    if any(term in fund_name for term in [
        "private markets", "secondaries", "secondary", "multi-asset",
        "private assets", "diversified"
    ]):
        return "fund_of_funds"

    return "unknown"


# =============================================================================
# VALIDATION RULES
# =============================================================================

def apply_direct_credit_rules(extraction: dict, report: ValidationReport) -> dict:
    """
    Apply validation rules for direct credit funds.

    Direct credit funds (like Blue Owl) should NOT have:
    - allocation_targets.* (they don't invest in funds)
    - concentration_limits.max_single_fund_pct (no underlying funds)
    """
    # Fields that should be null for direct credit funds
    fof_fields = [
        ("allocation_targets", "secondary_funds_min_pct"),
        ("allocation_targets", "secondary_funds_max_pct"),
        ("allocation_targets", "direct_investments_min_pct"),
        ("allocation_targets", "direct_investments_max_pct"),
        ("concentration_limits", "max_single_fund_pct"),
        ("incentive_fee", "underlying_fund_incentive_range"),
    ]

    for parent_field, child_field in fof_fields:
        parent = extraction.get(parent_field, {})
        if parent and parent.get(child_field) is not None:
            original = parent[child_field]
            parent[child_field] = None
            report.corrections.append(ValidationCorrection(
                field_path=f"{parent_field}.{child_field}",
                original_value=original,
                corrected_value=None,
                rule_name="direct_credit_nullify_fof_fields",
                reason="Direct credit funds don't invest in underlying funds",
            ))
            logger.info(
                f"[Validation] Nullified {parent_field}.{child_field} for direct credit fund: "
                f"{original} -> null"
            )

    return extraction


def apply_no_incentive_fee_rules(extraction: dict, report: ValidationReport) -> dict:
    """
    Apply validation rules when has_incentive_fee is False.

    If no incentive fee exists, related fields should be null.
    """
    incentive_fee = extraction.get("incentive_fee", {})
    if not incentive_fee:
        return extraction

    # Only apply if has_incentive_fee is explicitly False
    if incentive_fee.get("has_incentive_fee") is not False:
        return extraction

    # Fields that should be null when no incentive fee
    dependent_fields = [
        "incentive_fee_pct",
        "hurdle_rate_pct",
        "hurdle_rate_as_stated",
        "hurdle_rate_frequency",
        "high_water_mark",
        "has_catch_up",
        "catch_up_rate_pct",
        "catch_up_ceiling_pct",
        "fee_basis",
        "crystallization_frequency",
    ]

    for field_name in dependent_fields:
        if incentive_fee.get(field_name) is not None:
            original = incentive_fee[field_name]
            incentive_fee[field_name] = None
            report.corrections.append(ValidationCorrection(
                field_path=f"incentive_fee.{field_name}",
                original_value=original,
                corrected_value=None,
                rule_name="no_incentive_fee_nullify_dependent",
                reason="No incentive fee exists (has_incentive_fee=False)",
            ))
            logger.info(
                f"[Validation] Nullified incentive_fee.{field_name}: "
                f"{original} -> null (no incentive fee)"
            )

    return extraction


def apply_no_hurdle_rate_rules(extraction: dict, report: ValidationReport) -> dict:
    """
    Apply validation rules when hurdle_rate_pct is null.

    If no hurdle rate, related hurdle fields should be null.
    """
    incentive_fee = extraction.get("incentive_fee", {})
    if not incentive_fee:
        return extraction

    # Only apply if hurdle_rate_pct is null
    if incentive_fee.get("hurdle_rate_pct") is not None:
        return extraction

    # Fields that depend on hurdle rate
    hurdle_dependent_fields = [
        "hurdle_rate_as_stated",
        "hurdle_rate_frequency",
    ]

    for field_name in hurdle_dependent_fields:
        if incentive_fee.get(field_name) is not None:
            original = incentive_fee[field_name]
            incentive_fee[field_name] = None
            report.corrections.append(ValidationCorrection(
                field_path=f"incentive_fee.{field_name}",
                original_value=original,
                corrected_value=None,
                rule_name="no_hurdle_rate_nullify_dependent",
                reason="No hurdle rate exists (hurdle_rate_pct=null)",
            ))
            logger.info(
                f"[Validation] Nullified incentive_fee.{field_name}: "
                f"{original} -> null (no hurdle rate)"
            )

    return extraction


def apply_no_catch_up_rules(extraction: dict, report: ValidationReport) -> dict:
    """
    Apply validation rules when has_catch_up is False.

    If no catch-up provision, related fields should be null.
    """
    incentive_fee = extraction.get("incentive_fee", {})
    if not incentive_fee:
        return extraction

    # Only apply if has_catch_up is explicitly False or null
    has_catch_up = incentive_fee.get("has_catch_up")
    if has_catch_up is True:
        return extraction

    # Fields that depend on catch-up
    catch_up_dependent_fields = [
        "catch_up_rate_pct",
        "catch_up_ceiling_pct",
    ]

    for field_name in catch_up_dependent_fields:
        if incentive_fee.get(field_name) is not None:
            original = incentive_fee[field_name]
            incentive_fee[field_name] = None
            report.corrections.append(ValidationCorrection(
                field_path=f"incentive_fee.{field_name}",
                original_value=original,
                corrected_value=None,
                rule_name="no_catch_up_nullify_dependent",
                reason="No catch-up provision exists",
            ))
            logger.info(
                f"[Validation] Nullified incentive_fee.{field_name}: "
                f"{original} -> null (no catch-up)"
            )

    return extraction


def apply_repurchase_consistency_rules(extraction: dict, report: ValidationReport) -> dict:
    """
    Apply validation rules for repurchase terms consistency.

    Rules:
    - If repurchase_amount_pct is set, repurchase_percentage_min/max should be null
    - If repurchase_percentage_min/max is set, repurchase_amount_pct should be null
    """
    repurchase = extraction.get("repurchase_terms", {})
    if not repurchase:
        return extraction

    amount_pct = repurchase.get("repurchase_amount_pct")
    pct_min = repurchase.get("repurchase_percentage_min")
    pct_max = repurchase.get("repurchase_percentage_max")

    # If both formats are populated, prefer min/max if they differ, else prefer amount_pct
    if amount_pct is not None and (pct_min is not None or pct_max is not None):
        # If min != max, prefer min/max format (more informative)
        if pct_min != pct_max and pct_min is not None and pct_max is not None:
            original = amount_pct
            repurchase["repurchase_amount_pct"] = None
            report.corrections.append(ValidationCorrection(
                field_path="repurchase_terms.repurchase_amount_pct",
                original_value=original,
                corrected_value=None,
                rule_name="repurchase_format_consistency",
                reason="Prefer min/max format when range is specified",
            ))
        else:
            # Prefer amount_pct when min == max
            if pct_min is not None:
                repurchase["repurchase_percentage_min"] = None
            if pct_max is not None:
                repurchase["repurchase_percentage_max"] = None

    return extraction


def apply_leverage_basis_consistency(extraction: dict, report: ValidationReport) -> dict:
    """
    Apply validation rules for leverage basis consistency.

    If max_leverage_pct is set but leverage_basis is null, infer the basis.
    """
    leverage = extraction.get("leverage_limits", {})
    if not leverage:
        return extraction

    max_pct = leverage.get("max_leverage_pct")
    basis = leverage.get("leverage_basis")

    # If max_pct is around 33.33 and basis is null, likely asset_coverage
    if max_pct is not None and basis is None:
        try:
            max_val = float(max_pct) if isinstance(max_pct, str) else max_pct
            if 32 <= max_val <= 34:
                leverage["leverage_basis"] = "asset_coverage"
                report.corrections.append(ValidationCorrection(
                    field_path="leverage_limits.leverage_basis",
                    original_value=None,
                    corrected_value="asset_coverage",
                    rule_name="leverage_basis_inference",
                    reason="max_leverage_pct ~33% implies standard 1940 Act 300% coverage",
                ))
        except (ValueError, TypeError):
            pass

    return extraction


# =============================================================================
# PHASE 4: FIELD DISAMBIGUATION RULES
# =============================================================================
# These rules fix common LLM mistakes where similar-sounding fields get confused.
# Based on error analysis showing 40-60% accuracy on these fields.

def apply_distribution_frequency_disambiguation(
    extraction: dict,
    report: ValidationReport,
) -> dict:
    """
    Fix distribution_frequency confusion with repurchase_frequency.

    Common LLM mistake: Extracting "quarterly" for distribution_frequency when
    the document says "quarterly repurchase offers" (which is repurchase, not distribution).

    Rules:
    1. If distribution_frequency == repurchase_frequency == "quarterly", likely confused
    2. Check evidence text for "repurchase" indicators
    3. If fund_type is interval_fund and distribution not explicitly stated, likely monthly
    """
    distribution = extraction.get("distribution_terms", {})
    repurchase = extraction.get("repurchase_terms", {})

    if not distribution:
        return extraction

    dist_freq = distribution.get("distribution_frequency")
    repurchase_freq = repurchase.get("repurchase_frequency") if repurchase else None

    # Rule 1: If both are quarterly and we have evidence, check for confusion
    if dist_freq == "quarterly" and repurchase_freq == "quarterly":
        # Check evidence for repurchase indicators
        evidence = distribution.get("evidence", "").lower()

        repurchase_indicators = [
            "repurchase offer",
            "tender offer",
            "repurchase of shares",
            "buy back",
            "redemption offer",
        ]

        # Check if evidence mentions repurchase but NOT distribution/dividend
        has_repurchase_indicator = any(ind in evidence for ind in repurchase_indicators)
        distribution_indicators = [
            "distribution", "dividend", "income payment", "pays monthly",
            "pays quarterly", "declares dividend", "distribution frequency"
        ]
        has_distribution_indicator = any(ind in evidence for ind in distribution_indicators)

        if has_repurchase_indicator and not has_distribution_indicator:
            original = dist_freq
            distribution["distribution_frequency"] = None
            report.corrections.append(ValidationCorrection(
                field_path="distribution_terms.distribution_frequency",
                original_value=original,
                corrected_value=None,
                rule_name="distribution_repurchase_confusion",
                reason="Evidence mentions 'repurchase' but not 'distribution' - likely confused with repurchase_frequency",
            ))
            logger.info(
                f"[Validation] Nullified distribution_frequency: '{original}' -> null "
                "(confused with repurchase_frequency)"
            )

    # Rule 2: Interval funds typically have monthly distributions, not quarterly
    # If distribution_frequency is quarterly but fund is interval_fund, flag for review
    fund_type = extraction.get("fund_type")
    if dist_freq == "quarterly" and fund_type == "interval_fund":
        evidence = distribution.get("evidence", "").lower()
        # Only flag if evidence doesn't explicitly mention quarterly distributions
        if "quarterly distribution" not in evidence and "distributions quarterly" not in evidence:
            # Don't auto-correct, but log a warning
            logger.warning(
                f"[Validation] distribution_frequency='quarterly' for interval_fund - "
                "verify this is correct (most interval funds pay monthly)"
            )

    return extraction


def apply_leverage_basis_disambiguation(
    extraction: dict,
    report: ValidationReport,
) -> dict:
    """
    Fix leverage_basis interpretation errors.

    Common LLM mistakes:
    1. Setting leverage_basis="total_assets" when document mentions "asset coverage"
    2. Not recognizing 300% coverage ratio = 33.33% leverage
    3. Returning leverage_pct with % symbol as string

    Rules:
    1. If max_leverage_pct ~33% and basis is "total_assets", should be "asset_coverage"
    2. Clean up max_leverage_pct if it contains % symbol
    3. If evidence mentions "300%" or "asset coverage", basis should be "asset_coverage"
    """
    leverage = extraction.get("leverage_limits", {})
    if not leverage:
        return extraction

    max_pct = leverage.get("max_leverage_pct")
    basis = leverage.get("leverage_basis")
    evidence = leverage.get("evidence", "").lower()

    # Rule 1: Clean up max_leverage_pct if it's a string with %
    if isinstance(max_pct, str):
        # Remove % and other formatting
        cleaned = re.sub(r'[%\s]', '', max_pct)
        try:
            max_val = float(cleaned)
            if max_val != float(max_pct.replace('%', '').strip()):
                leverage["max_leverage_pct"] = str(max_val)
                report.corrections.append(ValidationCorrection(
                    field_path="leverage_limits.max_leverage_pct",
                    original_value=max_pct,
                    corrected_value=str(max_val),
                    rule_name="leverage_pct_format_cleanup",
                    reason="Removed % symbol from leverage percentage",
                ))
        except (ValueError, TypeError):
            pass

    # Rule 2: If max_leverage_pct ~33% but basis is "total_assets", check evidence
    if max_pct is not None and basis == "total_assets":
        try:
            max_val = float(str(max_pct).replace('%', '').strip())
            if 32 <= max_val <= 34:
                # Check if evidence mentions asset coverage
                asset_coverage_indicators = [
                    "asset coverage", "300%", "300 percent",
                    "1940 act", "investment company act",
                ]
                if any(ind in evidence for ind in asset_coverage_indicators):
                    original = basis
                    leverage["leverage_basis"] = "asset_coverage"
                    report.corrections.append(ValidationCorrection(
                        field_path="leverage_limits.leverage_basis",
                        original_value=original,
                        corrected_value="asset_coverage",
                        rule_name="leverage_basis_asset_coverage_correction",
                        reason="max_leverage_pct ~33% with asset coverage in evidence indicates asset_coverage basis",
                    ))
                    logger.info(
                        f"[Validation] Corrected leverage_basis: '{original}' -> 'asset_coverage' "
                        "(evidence mentions asset coverage)"
                    )
        except (ValueError, TypeError):
            pass

    # Rule 3: If evidence clearly mentions "300% asset coverage" but basis is wrong
    if "300%" in evidence and "asset coverage" in evidence:
        if basis not in ["asset_coverage", "asset_coverage_ratio"]:
            original = basis
            leverage["leverage_basis"] = "asset_coverage"
            if original is not None:
                report.corrections.append(ValidationCorrection(
                    field_path="leverage_limits.leverage_basis",
                    original_value=original,
                    corrected_value="asset_coverage",
                    rule_name="leverage_basis_evidence_override",
                    reason="Evidence explicitly mentions '300% asset coverage'",
                ))
                logger.info(
                    f"[Validation] Corrected leverage_basis: '{original}' -> 'asset_coverage' "
                    "(evidence explicitly mentions 300% asset coverage)"
                )

    return extraction


def apply_hurdle_rate_frequency_disambiguation(
    extraction: dict,
    report: ValidationReport,
) -> dict:
    """
    Fix hurdle_rate_frequency interpretation errors.

    Common LLM mistakes:
    1. Setting hurdle_rate_frequency="annual" when rate is stated quarterly (1.5% per quarter)
    2. Setting hurdle_rate_frequency="quarterly" when rate is annualized (6% annual)

    Rules:
    1. If hurdle_rate_as_stated is small (< 3%) and frequency is "annual", likely quarterly
    2. If hurdle_rate_pct is ~4x hurdle_rate_as_stated, frequency should be "quarterly"
    3. Check evidence for explicit frequency indicators
    """
    incentive_fee = extraction.get("incentive_fee", {})
    if not incentive_fee:
        return extraction

    hurdle_pct = incentive_fee.get("hurdle_rate_pct")
    hurdle_stated = incentive_fee.get("hurdle_rate_as_stated")
    frequency = incentive_fee.get("hurdle_rate_frequency")
    evidence = incentive_fee.get("evidence", "").lower()

    if frequency is None or hurdle_pct is None:
        return extraction

    # Rule 1: Check if evidence explicitly mentions quarterly/annual
    quarterly_indicators = [
        "per quarter", "quarterly hurdle", "each quarter",
        "calendar quarter", "quarterly basis", "1.5% quarterly",
        "1.25% quarterly", "quarterly rate"
    ]
    annual_indicators = [
        "per annum", "annual hurdle", "annualized", "per year",
        "annual rate", "yearly"
    ]

    evidence_suggests_quarterly = any(ind in evidence for ind in quarterly_indicators)
    evidence_suggests_annual = any(ind in evidence for ind in annual_indicators)

    # Rule 2: If hurdle_rate_as_stated is small (< 3%) and frequency is "annual", likely quarterly
    if hurdle_stated is not None and frequency == "annual":
        try:
            stated_val = float(str(hurdle_stated).replace('%', '').strip())
            if stated_val < 3 and evidence_suggests_quarterly:
                original = frequency
                incentive_fee["hurdle_rate_frequency"] = "quarterly"
                report.corrections.append(ValidationCorrection(
                    field_path="incentive_fee.hurdle_rate_frequency",
                    original_value=original,
                    corrected_value="quarterly",
                    rule_name="hurdle_frequency_quarterly_correction",
                    reason=f"hurdle_rate_as_stated={stated_val}% is too small for annual; evidence mentions quarterly",
                ))
                logger.info(
                    f"[Validation] Corrected hurdle_rate_frequency: '{original}' -> 'quarterly' "
                    f"(stated rate {stated_val}% suggests quarterly)"
                )
        except (ValueError, TypeError):
            pass

    # Rule 3: If hurdle_rate_pct ~4x hurdle_rate_as_stated, frequency should be quarterly
    if hurdle_stated is not None and hurdle_pct is not None:
        try:
            stated_val = float(str(hurdle_stated).replace('%', '').strip())
            pct_val = float(str(hurdle_pct).replace('%', '').strip())

            # Check if pct is approximately 4x stated (quarterly -> annual)
            if stated_val > 0 and 3.5 <= (pct_val / stated_val) <= 4.5:
                if frequency != "quarterly":
                    original = frequency
                    incentive_fee["hurdle_rate_frequency"] = "quarterly"
                    report.corrections.append(ValidationCorrection(
                        field_path="incentive_fee.hurdle_rate_frequency",
                        original_value=original,
                        corrected_value="quarterly",
                        rule_name="hurdle_frequency_ratio_correction",
                        reason=f"hurdle_rate_pct ({pct_val}) is ~4x hurdle_rate_as_stated ({stated_val}), indicating quarterly measurement",
                    ))
                    logger.info(
                        f"[Validation] Corrected hurdle_rate_frequency: '{original}' -> 'quarterly' "
                        f"(pct {pct_val} is ~4x stated {stated_val})"
                    )
        except (ValueError, TypeError, ZeroDivisionError):
            pass

    # Rule 4: If hurdle_rate_as_stated == hurdle_rate_pct AND crystallization is quarterly,
    # the extractor likely failed to decompose the annualized rate.
    # Evidence pattern: "X% annualized hurdle rate" paid "quarterly in arrears"
    # means the stated quarterly rate is X/4, and frequency is quarterly.
    crystallization = incentive_fee.get("crystallization_frequency")
    if (
        hurdle_stated is not None
        and hurdle_pct is not None
        and frequency == "annual"
        and crystallization == "quarterly"
    ):
        try:
            stated_val = float(str(hurdle_stated).replace('%', '').strip())
            pct_val = float(str(hurdle_pct).replace('%', '').strip())

            # Both values are the same = extractor didn't decompose
            if stated_val == pct_val and pct_val > 0:
                # Check evidence for annualized + quarterly pattern
                # Evidence may be in top-level 'evidence' or per-field '_evidence'
                evidence_texts = [evidence]
                per_field_ev = incentive_fee.get("_evidence", {})
                for ev_field in ["hurdle_rate_frequency", "hurdle_rate_pct", "crystallization_frequency"]:
                    ev_text = per_field_ev.get(ev_field, "")
                    if ev_text:
                        evidence_texts.append(ev_text.lower())
                combined_evidence = " ".join(evidence_texts)

                annualized_indicators = [
                    "annualized", "per annum", "annual rate",
                    "annualized hurdle", "annualized preferred",
                ]
                quarterly_payment_indicators = [
                    "quarterly in arrears", "paid quarterly",
                    "each calendar quarter", "per quarter",
                    "quarterly basis",
                ]

                has_annualized = any(ind in combined_evidence for ind in annualized_indicators)
                has_quarterly_payment = any(ind in combined_evidence for ind in quarterly_payment_indicators)

                if has_annualized and has_quarterly_payment:
                    # Fix frequency: quarterly measurement, not annual
                    original_freq = frequency
                    incentive_fee["hurdle_rate_frequency"] = "quarterly"
                    report.corrections.append(ValidationCorrection(
                        field_path="incentive_fee.hurdle_rate_frequency",
                        original_value=original_freq,
                        corrected_value="quarterly",
                        rule_name="hurdle_frequency_annualized_quarterly_correction",
                        reason=(
                            f"hurdle_rate_as_stated ({stated_val}) == hurdle_rate_pct ({pct_val}) "
                            f"with crystallization_frequency='quarterly' and evidence mentions "
                            f"'annualized' + 'quarterly' payment — frequency is quarterly"
                        ),
                    ))
                    logger.info(
                        f"[Validation] Corrected hurdle_rate_frequency: '{original_freq}' -> 'quarterly' "
                        f"(annualized rate {pct_val}% paid quarterly)"
                    )

                    # Fix as_stated: derive quarterly rate from annualized
                    quarterly_rate = round(pct_val / 4, 4)
                    # Clean up trailing zeros: 1.2500 -> 1.25
                    quarterly_str = f"{quarterly_rate:g}"
                    original_stated = hurdle_stated
                    incentive_fee["hurdle_rate_as_stated"] = quarterly_str
                    report.corrections.append(ValidationCorrection(
                        field_path="incentive_fee.hurdle_rate_as_stated",
                        original_value=original_stated,
                        corrected_value=quarterly_str,
                        rule_name="hurdle_as_stated_quarterly_derivation",
                        reason=(
                            f"Derived quarterly stated rate: {pct_val}% annualized / 4 = {quarterly_str}% per quarter"
                        ),
                    ))
                    logger.info(
                        f"[Validation] Corrected hurdle_rate_as_stated: '{original_stated}' -> '{quarterly_str}' "
                        f"({pct_val}% / 4 quarters)"
                    )
        except (ValueError, TypeError, ZeroDivisionError):
            pass

    return extraction


def apply_catch_up_rate_disambiguation(
    extraction: dict,
    report: ValidationReport,
) -> dict:
    """
    Fix catch_up_rate_pct interpretation errors.

    Common issue: catch_up_rate_pct should be 100 for "full catch-up" or "100% catch-up",
    but LLM sometimes returns the incentive fee rate instead.

    Rules:
    1. If has_catch_up=True and evidence mentions "full catch-up", rate should be 100
    2. If catch_up_rate_pct equals incentive_fee_pct, likely wrong (should be 100 for full catch-up)
    """
    incentive_fee = extraction.get("incentive_fee", {})
    if not incentive_fee:
        return extraction

    has_catch_up = incentive_fee.get("has_catch_up")
    catch_up_rate = incentive_fee.get("catch_up_rate_pct")
    incentive_rate = incentive_fee.get("incentive_fee_pct")
    evidence = incentive_fee.get("evidence", "").lower()

    if not has_catch_up:
        return extraction

    # Rule 1: Check for "full catch-up" or "100% catch-up" in evidence
    full_catchup_indicators = [
        "full catch-up", "100% catch-up", "full catch up",
        "100 percent catch-up", "complete catch-up"
    ]

    if any(ind in evidence for ind in full_catchup_indicators):
        if catch_up_rate is not None:
            try:
                rate_val = float(str(catch_up_rate).replace('%', '').strip())
                if rate_val != 100:
                    original = catch_up_rate
                    incentive_fee["catch_up_rate_pct"] = "100"
                    report.corrections.append(ValidationCorrection(
                        field_path="incentive_fee.catch_up_rate_pct",
                        original_value=original,
                        corrected_value="100",
                        rule_name="catch_up_full_rate_correction",
                        reason="Evidence mentions 'full catch-up', rate should be 100%",
                    ))
                    logger.info(
                        f"[Validation] Corrected catch_up_rate_pct: '{original}' -> '100' "
                        "(evidence mentions full catch-up)"
                    )
            except (ValueError, TypeError):
                pass
        elif catch_up_rate is None:
            # If has_catch_up but no rate, and evidence says "full", set to 100
            incentive_fee["catch_up_rate_pct"] = "100"
            report.corrections.append(ValidationCorrection(
                field_path="incentive_fee.catch_up_rate_pct",
                original_value=None,
                corrected_value="100",
                rule_name="catch_up_full_rate_inference",
                reason="has_catch_up=True and evidence mentions 'full catch-up'",
            ))

    # Rule 2: If catch_up_rate equals incentive_fee_pct, likely confused
    if catch_up_rate is not None and incentive_rate is not None:
        try:
            catch_val = float(str(catch_up_rate).replace('%', '').strip())
            inc_val = float(str(incentive_rate).replace('%', '').strip())

            # Common mistake: returning 10% (incentive rate) instead of 100% (catch-up rate)
            if catch_val == inc_val and catch_val < 50:
                # Check if evidence suggests full catch-up
                if any(ind in evidence for ind in ["catch-up", "catch up"]):
                    original = catch_up_rate
                    incentive_fee["catch_up_rate_pct"] = "100"
                    report.corrections.append(ValidationCorrection(
                        field_path="incentive_fee.catch_up_rate_pct",
                        original_value=original,
                        corrected_value="100",
                        rule_name="catch_up_rate_confusion_correction",
                        reason=f"catch_up_rate_pct ({catch_val}) equals incentive_fee_pct - likely should be 100% for standard catch-up",
                    ))
                    logger.info(
                        f"[Validation] Corrected catch_up_rate_pct: '{original}' -> '100' "
                        f"(was same as incentive_fee_pct {inc_val}%)"
                    )
        except (ValueError, TypeError):
            pass

    return extraction


def apply_field_normalization(
    extraction: dict,
    report: ValidationReport,
) -> dict:
    """
    Apply field value normalization using synonym mappings.

    This ensures extracted values use canonical forms for consistent evaluation.
    For example: "asset coverage" -> "asset_coverage", "reinvested" -> "drip"

    Args:
        extraction: The extraction result dict
        report: Validation report to record corrections

    Returns:
        Extraction dict with normalized field values
    """
    # Normalize leverage_limits.leverage_basis
    leverage = extraction.get("leverage_limits", {})
    if leverage and leverage.get("leverage_basis"):
        original = leverage["leverage_basis"]
        normalized = normalize_leverage_basis(original)
        if normalized != original:
            leverage["leverage_basis"] = normalized
            report.corrections.append(ValidationCorrection(
                field_path="leverage_limits.leverage_basis",
                original_value=original,
                corrected_value=normalized,
                rule_name="leverage_basis_normalization",
                reason=f"Normalized '{original}' to canonical form '{normalized}'",
            ))
            logger.debug(f"[Normalization] leverage_basis: '{original}' -> '{normalized}'")

    # Normalize repurchase_terms.repurchase_basis
    repurchase = extraction.get("repurchase_terms", {})
    if repurchase and repurchase.get("repurchase_basis"):
        original = repurchase["repurchase_basis"]
        normalized = normalize_repurchase_basis(original)
        if normalized != original:
            repurchase["repurchase_basis"] = normalized
            report.corrections.append(ValidationCorrection(
                field_path="repurchase_terms.repurchase_basis",
                original_value=original,
                corrected_value=normalized,
                rule_name="repurchase_basis_normalization",
                reason=f"Normalized '{original}' to canonical form '{normalized}'",
            ))
            logger.debug(f"[Normalization] repurchase_basis: '{original}' -> '{normalized}'")

    # Normalize repurchase_terms.repurchase_frequency
    if repurchase and repurchase.get("repurchase_frequency"):
        original = repurchase["repurchase_frequency"]
        normalized = normalize_frequency(original, "repurchase")
        if normalized != original:
            repurchase["repurchase_frequency"] = normalized
            report.corrections.append(ValidationCorrection(
                field_path="repurchase_terms.repurchase_frequency",
                original_value=original,
                corrected_value=normalized,
                rule_name="repurchase_frequency_normalization",
                reason=f"Normalized '{original}' to canonical form '{normalized}'",
            ))
            logger.debug(f"[Normalization] repurchase_frequency: '{original}' -> '{normalized}'")

    # Normalize distribution_terms.distribution_frequency
    distribution = extraction.get("distribution_terms", {})
    if distribution and distribution.get("distribution_frequency"):
        original = distribution["distribution_frequency"]
        normalized = normalize_frequency(original, "distribution")
        if normalized != original:
            distribution["distribution_frequency"] = normalized
            report.corrections.append(ValidationCorrection(
                field_path="distribution_terms.distribution_frequency",
                original_value=original,
                corrected_value=normalized,
                rule_name="distribution_frequency_normalization",
                reason=f"Normalized '{original}' to canonical form '{normalized}'",
            ))
            logger.debug(f"[Normalization] distribution_frequency: '{original}' -> '{normalized}'")

    # Normalize distribution_terms.distribution_policy (default_policy)
    if distribution and distribution.get("default_policy"):
        original = distribution["default_policy"]
        normalized = normalize_distribution_policy(original)
        if normalized != original:
            distribution["default_policy"] = normalized
            report.corrections.append(ValidationCorrection(
                field_path="distribution_terms.default_policy",
                original_value=original,
                corrected_value=normalized,
                rule_name="distribution_policy_normalization",
                reason=f"Normalized '{original}' to canonical form '{normalized}'",
            ))
            logger.debug(f"[Normalization] default_policy: '{original}' -> '{normalized}'")

    # Normalize incentive_fee.hurdle_rate_frequency
    incentive_fee = extraction.get("incentive_fee", {})
    if incentive_fee and incentive_fee.get("hurdle_rate_frequency"):
        original = incentive_fee["hurdle_rate_frequency"]
        normalized = normalize_frequency(original, "hurdle")
        if normalized != original:
            incentive_fee["hurdle_rate_frequency"] = normalized
            report.corrections.append(ValidationCorrection(
                field_path="incentive_fee.hurdle_rate_frequency",
                original_value=original,
                corrected_value=normalized,
                rule_name="hurdle_rate_frequency_normalization",
                reason=f"Normalized '{original}' to canonical form '{normalized}'",
            ))
            logger.debug(f"[Normalization] hurdle_rate_frequency: '{original}' -> '{normalized}'")

    return extraction


def apply_nullable_field_handling(
    extraction: dict,
    report: ValidationReport,
    fund_strategy: str,
) -> dict:
    """
    Apply nullable field handling based on fund strategy.

    This function:
    1. Nullifies fields that should NOT exist for this fund type
    2. Logs fields that are legitimately nullable (may not be disclosed)

    Args:
        extraction: The extraction result dict
        report: Validation report to record corrections
        fund_strategy: The detected fund strategy

    Returns:
        Extraction dict with strategy-inappropriate fields nullified
    """
    try:
        strategy = FundStrategy(fund_strategy)
    except ValueError:
        # Unknown strategy, don't nullify anything
        return extraction

    # Get fields that should be null for this strategy
    null_fields = get_null_fields_for_strategy(strategy)

    def set_nested_null(data: dict, field_path: str) -> Optional[Any]:
        """Set a nested field to null and return the original value."""
        parts = field_path.split(".")
        current = data

        # Navigate to parent
        for part in parts[:-1]:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None  # Path doesn't exist

        # Set the final field to null
        final_key = parts[-1]
        if isinstance(current, dict) and final_key in current:
            original = current[final_key]
            if original is not None:
                current[final_key] = None
                return original
        return None

    # Nullify strategy-inappropriate fields
    for field_path in null_fields:
        original = set_nested_null(extraction, field_path)
        if original is not None:
            report.corrections.append(ValidationCorrection(
                field_path=field_path,
                original_value=original,
                corrected_value=None,
                rule_name="nullable_field_by_strategy",
                reason=f"Field '{field_path}' not applicable for {strategy.value} fund",
            ))
            logger.debug(
                f"[Nullable] Nullified {field_path} for {strategy.value} fund "
                f"(was: {original})"
            )

    return extraction


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

def apply_validation_rules(
    extraction: dict,
    fund_strategy: Optional[str] = None,
) -> tuple[dict, ValidationReport]:
    """
    Apply all validation rules to an extraction result.

    Args:
        extraction: The extraction result dict
        fund_strategy: Optional override for fund strategy detection

    Returns:
        Tuple of (corrected extraction, validation report)
    """
    fund_name = extraction.get("fund_name", "unknown")

    # Detect fund strategy if not provided
    if fund_strategy is None:
        fund_strategy = detect_fund_strategy(extraction)

    report = ValidationReport(
        fund_name=fund_name,
        fund_strategy=fund_strategy,
    )

    logger.info(f"[Validation] Applying rules to {fund_name} (strategy: {fund_strategy})")

    # Apply nullable field handling based on fund strategy
    # This nullifies fields that shouldn't exist for this fund type
    extraction = apply_nullable_field_handling(extraction, report, fund_strategy)

    # Apply fund-strategy-specific rules (legacy - being replaced by nullable field handling)
    if fund_strategy == "direct_credit":
        extraction = apply_direct_credit_rules(extraction, report)

    # Apply cross-field consistency rules
    extraction = apply_no_incentive_fee_rules(extraction, report)
    extraction = apply_no_hurdle_rate_rules(extraction, report)
    extraction = apply_no_catch_up_rules(extraction, report)
    extraction = apply_repurchase_consistency_rules(extraction, report)
    extraction = apply_leverage_basis_consistency(extraction, report)

    # Phase 4: Field disambiguation rules
    extraction = apply_distribution_frequency_disambiguation(extraction, report)
    extraction = apply_leverage_basis_disambiguation(extraction, report)
    extraction = apply_hurdle_rate_frequency_disambiguation(extraction, report)
    extraction = apply_catch_up_rate_disambiguation(extraction, report)

    # Phase 5: Field value normalization (synonym mapping)
    # This ensures extracted values use canonical forms for consistent evaluation
    extraction = apply_field_normalization(extraction, report)

    if report.correction_count > 0:
        logger.info(f"[Validation] Applied {report.correction_count} corrections to {fund_name}")
    else:
        logger.debug(f"[Validation] No corrections needed for {fund_name}")

    return extraction, report
