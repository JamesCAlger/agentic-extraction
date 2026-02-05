"""
Ensemble Extraction with T4 Escalation.

This module implements the "agreement + escalation" strategy:
1. Run T3 k=10 extraction
2. Run Reranker 200->10 extraction
3. Compare results per field:
   - If both agree on NON-NULL value: Accept with high confidence
   - If disagreement OR both null: Escalate to T4 agentic extraction

This balances accuracy (T4) with cost efficiency (accept when methods agree).

=============================================================================
T4 INTEGRATION NOTES FOR FUTURE CLAUDE AGENTS
=============================================================================

T4 changes in tier4_agentic.py automatically flow through to this module.

HOW IT WORKS:
- This module imports Tier4Agent and FIELD_SPECS from tier4_agentic.py
- When T4 runs, it uses the current implementation of Tier4Agent
- FieldSpec definitions in FIELD_SPECS control extraction hints and behavior

WHAT TO UPDATE IF T4 CHANGES:

1. If you ADD new fields to T4:
   - Add field path mapping to FIELD_PATH_TO_T4_SPEC dict below
   - Add field path to COMPARISON_FIELDS list if it should be compared
   - The field will automatically be escalated when T3/Reranker disagree

2. If you MODIFY T4 FieldSpec (in tier4_agentic.py):
   - Changes flow through automatically
   - No changes needed here

3. If you CHANGE T4 Tier4Agent API:
   - Update the _run_t4_escalation method below
   - Update how we call agent.extract() and handle Tier4ExtractionResult

4. If you CHANGE T4 configuration options:
   - Update EnsembleT4Config dataclass below
   - Update config mapping in runner.py _extract_with_ensemble_t4 method

EXPERIMENT TRACKING:
- Experiment results are in data/experiments/exp_*_ensemble_t4_*/
- Trace includes field_decisions showing T3/Reranker/T4 values per field
- Evaluation compares final values against ground truth
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..parse.models import ChunkedDocument
from .scoped_agentic import RerankerConfig
from .hybrid_retriever import HybridConfig
from .embedding_retriever import EmbeddingConfig
from .share_class_extractor import (
    TwoPassShareClassExtractor,
    TwoPassConfig,
    convert_two_pass_to_dict,
    discover_share_classes,
    DiscoveryResult,
)
from .grounding_strategies import (
    GroundingStrategyConfig,
    BaseGroundingStrategy,
    GroundingVerification,
    create_grounding_strategy,
    format_claim,
)

logger = logging.getLogger(__name__)


# =============================================================================
# VALUE COMPARISON UTILITIES
# =============================================================================


def normalize_value(v: Any) -> Any:
    """
    Normalize a value for comparison.

    Handles: percentage symbols, decimal formatting, type differences.
    """
    if v is None:
        return None

    # Convert to string and clean
    s = str(v).strip().lower()

    # Remove percentage symbol
    s = s.rstrip('%')

    # Try to parse as number for numeric comparison
    try:
        num = float(s)
        return round(num, 4)
    except (TypeError, ValueError):
        pass

    return s


def values_equal(v1: Any, v2: Any) -> bool:
    """
    Check if two values are equal with format normalization.
    """
    if v1 == v2:
        return True
    if v1 is None or v2 is None:
        return False

    n1 = normalize_value(v1)
    n2 = normalize_value(v2)

    if n1 == n2:
        return True

    # Numeric tolerance comparison
    if isinstance(n1, (int, float)) and isinstance(n2, (int, float)):
        return abs(float(n1) - float(n2)) < 0.001

    return False


def both_null(v1: Any, v2: Any) -> bool:
    """Check if both values are null/None."""
    return v1 is None and v2 is None


# =============================================================================
# CONFIDENCE CALCULATION UTILITIES
# =============================================================================


def calculate_field_confidence(
    extraction_result: dict,
    field_path: str,
    require_grounding: bool = True,
    grounding_strategy: Optional[BaseGroundingStrategy] = None,
    source_chunks: Optional[list[str]] = None,
) -> float:
    """
    Calculate confidence score for a field extraction.

    CRITICAL: If require_grounding=True and the field is not grounded,
    returns 0.0 regardless of other signals. Ungrounded = no confidence.

    Args:
        extraction_result: Full extraction result dict with grounding info
        field_path: Dot-notation path to field (e.g., "incentive_fee.has_incentive_fee")
        require_grounding: If True, ungrounded fields get 0.0 confidence
        grounding_strategy: Optional configurable grounding strategy for verification
        source_chunks: Optional source chunks for grounding verification

    Returns:
        Confidence score between 0.0 and 1.0
    """
    # Get field value and evidence
    field_value = get_nested_value(extraction_result, field_path)

    # Get evidence from parent section
    parts = field_path.split(".")
    evidence = None
    if len(parts) >= 2:
        evidence = get_nested_value(extraction_result, parts[0] + ".evidence")

    # If we have a grounding strategy, use it for verification
    if grounding_strategy is not None and field_value is not None:
        # Format claim and verify
        claim = format_claim(field_path.split(".")[-1], field_value)
        evidence_text = str(evidence) if evidence else ""

        verification = grounding_strategy.verify(
            claim=claim,
            evidence=evidence_text,
            source_chunks=source_chunks,
            field_name=field_path,
            value=field_value,
        )

        if require_grounding and not verification.is_grounded:
            return 0.0

        # Return grounding confidence
        return verification.confidence

    # Fallback to existing grounding info if no strategy provided
    grounding = extraction_result.get("grounding", {})
    field_results = grounding.get("field_results", {})

    # Try to find grounding for this field (may be stored with different key format)
    grounding_result = None
    for key in [field_path, field_path.split(".")[-1], field_path.replace(".", "_")]:
        if key in field_results:
            grounding_result = field_results[key]
            break

    # CRITICAL: No grounding = zero confidence
    if require_grounding:
        if grounding_result is None:
            # No grounding info available - assume not grounded
            return 0.0
        if not grounding_result.get("is_grounded", False):
            # Explicitly not grounded
            return 0.0

    # Calculate composite confidence from available signals
    confidence_components = []

    # 1. Grounding score (if available)
    if grounding_result:
        grounding_score = grounding_result.get("grounding_score", 0.0)
        confidence_components.append(("grounding", grounding_score, 0.5))  # 50% weight

    # 2. LLM confidence (if stored in extraction)
    # Check for confidence in the field value itself
    if isinstance(field_value, dict) and "confidence" in field_value:
        llm_conf = field_value.get("confidence", 0.5)
        confidence_components.append(("llm", llm_conf, 0.3))  # 30% weight

    # 3. Evidence presence bonus
    if evidence and len(str(evidence)) > 50:
        confidence_components.append(("evidence", 0.8, 0.2))  # 20% weight, evidence = 0.8
    elif evidence:
        confidence_components.append(("evidence", 0.5, 0.2))
    else:
        confidence_components.append(("evidence", 0.2, 0.2))

    # Calculate weighted average
    if not confidence_components:
        return 0.5  # Default neutral confidence if no signals

    total_weight = sum(w for _, _, w in confidence_components)
    weighted_sum = sum(score * weight for _, score, weight in confidence_components)

    return weighted_sum / total_weight if total_weight > 0 else 0.5


def get_field_grounding_status(extraction_result: dict, field_path: str) -> tuple[bool, float]:
    """
    Get grounding status for a field.

    Returns:
        Tuple of (is_grounded, grounding_score)
    """
    grounding = extraction_result.get("grounding", {})
    field_results = grounding.get("field_results", {})

    # Try different key formats
    for key in [field_path, field_path.split(".")[-1], field_path.replace(".", "_")]:
        if key in field_results:
            result = field_results[key]
            return result.get("is_grounded", False), result.get("grounding_score", 0.0)

    # No grounding info found
    return False, 0.0


def get_nested_value(obj: dict, path: str) -> Any:
    """
    Get a nested value from a dict using dot notation.

    E.g., get_nested_value(d, "incentive_fee.has_incentive_fee")
    """
    parts = path.split(".")
    current = obj
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None

    # Handle {value: X} format
    if isinstance(current, dict) and "value" in current:
        return current["value"]
    return current


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class AdversarialValidationConfig:
    """
    Configuration for LLM-as-a-judge validation in ensemble flow.

    Validation runs on ACCEPTED fields (when T3/Reranker agree) BEFORE T4.
    If validation fails, the field is escalated to T4 for re-extraction.

    This catches false confidence cases where both methods agree on wrong value.
    """
    enabled: bool = True  # Enable LLM-as-a-judge validation on accepted fields
    lightweight: bool = False  # False = full adversarial (LLM-as-a-judge), True = simple check
    model: str = "claude-sonnet-4-20250514"  # Model for LLM-as-a-judge validation
    lightweight_model: str = "gpt-4o-mini"  # Model for lightweight validation (if lightweight=True)
    validate_booleans: bool = True  # Always validate boolean fields (prone to hallucination)
    validate_all_accepted: bool = True  # NEW: Validate ALL accepted fields (not just booleans)
    validate_all: bool = True  # Validate all field types (deprecated, use validate_all_accepted)
    require_exact_quote: bool = True  # Require exact quote from evidence (more strict)
    escalate_on_rejection: bool = True  # Escalate to T4 if validation fails


@dataclass
class EnsembleT4Config:
    """Configuration for ensemble extraction with T4 escalation."""

    # T3 configuration
    t3_top_k_sections: int = 10
    t3_max_chunks_per_section: int = 10

    # Reranker configuration
    reranker_first_pass_n: int = 200
    reranker_top_k: int = 10
    reranker_model: str = "rerank-v3.5"
    reranker_score_threshold: float = 0.2

    # T4 configuration
    t4_model: str = "gpt-4o"
    t4_max_iterations: int = 12
    t4_timeout_seconds: int = 180
    t4_confidence_threshold: float = 0.8

    # LLM configuration
    extraction_model: str = "gpt-4o-mini"
    extraction_provider: str = "openai"
    delay_between_calls: float = 1.0
    requests_per_minute: int = 40

    # Escalation behavior
    run_t4_escalation: bool = True  # If False, don't run T4 even when fields are escalated
    escalate_on_disagreement: bool = True
    escalate_on_both_null: bool = True  # Scenario 2 = True, Scenario 1 = False

    # Hybrid retrieval (replaces keyword-only T3 with keyword+dense+RRF)
    use_hybrid_retrieval: bool = False
    hybrid_embedding_provider: str = "openai"
    hybrid_embedding_model: str = "text-embedding-3-small"
    hybrid_rrf_k: int = 60  # RRF constant
    hybrid_keyword_top_k: int = 20  # Chunks from keyword retrieval
    hybrid_dense_top_k: int = 20  # Chunks from dense retrieval
    hybrid_final_top_k: int = 10  # Final chunks after RRF fusion

    # Multi-query expansion (replaces static keywords with expanded queries + RRF)
    use_multi_query: bool = False
    multi_query_expansion_method: str = "programmatic"  # "programmatic", "llm", or "hybrid"
    multi_query_retrieval_strategy: str = "keyword"  # "keyword", "dense", or "hybrid"
    multi_query_rrf_k: int = 60
    multi_query_per_query_top_k: int = 15
    multi_query_final_top_k: int = 10
    multi_query_embedding_provider: str = "openai"  # For dense/hybrid retrieval
    multi_query_embedding_model: str = "text-embedding-3-small"
    multi_query_holistic_extraction: bool = False  # Extract all fields in a group together

    # Adversarial validation (when T3 and Reranker agree)
    adversarial_validation: AdversarialValidationConfig = field(default_factory=AdversarialValidationConfig)

    # Ensemble methods to run (e.g., ["multi_query", "reranker"] or ["multi_query"] only)
    methods: list[str] = field(default_factory=lambda: ["multi_query", "reranker"])

    # Hybrid routing: when methods disagree, use field-type-based preference
    # instead of escalating to T4. Based on empirical analysis of which method
    # works better for each field type.
    use_hybrid_routing: bool = False

    # Confidence-based routing: when methods disagree, use confidence scores
    # to pick the more reliable extraction. CRITICAL: ungrounded = 0 confidence.
    use_confidence_routing: bool = False
    confidence_min_gap: float = 0.25  # Min confidence gap to prefer one method
    confidence_low_threshold: float = 0.3  # Below this, both are uncertain → escalate
    confidence_high_threshold: float = 0.7  # Both above this but disagree → escalate
    confidence_require_grounding: bool = True  # If True, ungrounded = 0 confidence

    # Grounding strategy configuration for confidence routing
    # When set, uses configurable grounding (NLI, LLM-judge, hybrid) instead of exact match
    grounding_strategy: Optional[str] = None  # None = use existing exact match, or "nli", "llm_judge", "hybrid"
    grounding_nli_model: str = "cross-encoder/nli-deberta-v3-base"
    grounding_nli_entailment_threshold: float = 0.7
    grounding_nli_contradiction_threshold: float = 0.7
    grounding_llm_judge_model: str = "gpt-4o-mini"
    grounding_llm_judge_provider: str = "openai"
    grounding_hybrid_nli_high_threshold: float = 0.85
    grounding_hybrid_nli_low_threshold: float = 0.4
    grounding_hybrid_use_llm_for_ambiguous: bool = True

    # Two-pass share class extraction (DEPRECATED - use discovery_first instead)
    # DISABLED: Caused 10.6% regression in exp_20260122_175330. Issues:
    # - sales_load_pct and distribution_fee_pct not extracted (no T4 spec)
    # - Share class name normalization (Class I Shares vs Class I)
    # - Blackstone Class I Advisory fields missed
    share_class_two_pass_enabled: bool = False
    share_class_two_pass_discovery_model: Optional[str] = None  # Uses extraction_model if None
    share_class_two_pass_discovery_max_chunks: int = 20
    share_class_two_pass_per_class_max_chunks: int = 15

    # Discovery-first share class extraction (RECOMMENDED)
    # Runs discovery BEFORE T1 to identify which classes exist, then passes
    # discovered class names to T1/T3/Reranker prompts for targeted extraction.
    # Benefits:
    # - Finds rare classes (Class R, Class U) reliably
    # - Reduces hallucination (can't invent classes not in discovered list)
    # - Improves retrieval (class-specific keyword queries)
    # - All fields still extracted by T1 (no field loss like two-pass)
    share_class_discovery_first_enabled: bool = True
    share_class_discovery_model: Optional[str] = None  # Uses extraction_model if None
    share_class_discovery_max_chunks: int = 25


@dataclass
class FieldResult:
    """Result for a single field extraction."""
    field_name: str
    t3_value: Any = None
    reranker_value: Any = None
    final_value: Any = None

    # Decision tracking
    values_agree: bool = False
    both_null: bool = False
    escalated_to_t4: bool = False
    escalation_reason: str = ""  # "none", "disagreement", "both_null", "llm_judge_rejected"

    # Source tracking
    source: str = ""  # "t3", "reranker", "t4", "agreement"
    t4_success: bool = False
    t4_confidence: float = 0.0
    t4_evidence: str = ""

    # Grounding info (if available)
    t3_grounded: Optional[bool] = None
    reranker_grounded: Optional[bool] = None

    # Confidence scores (for confidence-based routing)
    t3_confidence: float = 0.0
    reranker_confidence: float = 0.0

    # LLM-as-a-judge validation (runs on accepted fields before T4)
    adversarial_validated: Optional[bool] = None  # None=not run, True=passed, False=rejected
    adversarial_problems: list[str] = field(default_factory=list)
    adversarial_confidence: float = 0.0


@dataclass
class EnsembleExtractionResult:
    """Complete result from ensemble extraction."""
    fund_name: str
    extraction: dict[str, Any]
    field_results: dict[str, FieldResult] = field(default_factory=dict)

    # Statistics
    total_fields: int = 0
    accepted_count: int = 0
    escalated_count: int = 0
    t4_success_count: int = 0

    # Timing
    t3_duration_seconds: float = 0.0
    reranker_duration_seconds: float = 0.0
    t4_duration_seconds: float = 0.0
    total_duration_seconds: float = 0.0

    # Cost tracking (estimates)
    t3_cost_estimate: float = 0.0
    reranker_cost_estimate: float = 0.0
    t4_cost_estimate: float = 0.0


@dataclass
class EnsembleTrace:
    """Trace for observability."""
    extraction_mode: str = "ensemble_t4"
    config: dict = field(default_factory=dict)

    # Per-field details
    field_decisions: dict[str, dict] = field(default_factory=dict)

    # Aggregate stats
    agreement_rate: float = 0.0
    escalation_rate: float = 0.0
    t4_success_rate: float = 0.0

    # Timing breakdown
    discovery_duration_seconds: float = 0.0
    t3_duration_seconds: float = 0.0
    reranker_duration_seconds: float = 0.0
    t4_duration_seconds: float = 0.0

    # Discovery-first share class info
    discovered_share_classes: list[str] = field(default_factory=list)
    discovery_chunks_used: int = 0
    discovery_reasoning: Optional[str] = None


# =============================================================================
# FIELD DEFINITIONS FOR COMPARISON
# =============================================================================

# Fields to compare between T3 and Reranker
# These map to ground truth field paths
# ALL fields should be included to ensure T4 escalation when needed
COMPARISON_FIELDS = [
    # Fund type
    "fund_type",

    # Incentive fee fields (ALL)
    "incentive_fee.has_incentive_fee",
    "incentive_fee.incentive_fee_pct",
    "incentive_fee.hurdle_rate_pct",
    "incentive_fee.hurdle_rate_as_stated",
    "incentive_fee.hurdle_rate_frequency",
    "incentive_fee.high_water_mark",
    "incentive_fee.has_catch_up",
    "incentive_fee.catch_up_rate_pct",
    "incentive_fee.catch_up_ceiling_pct",
    "incentive_fee.fee_basis",
    "incentive_fee.crystallization_frequency",
    "incentive_fee.underlying_fund_incentive_range",

    # Expense cap fields
    "expense_cap.has_expense_cap",
    "expense_cap.expense_cap_pct",

    # Repurchase terms (ALL)
    "repurchase_terms.repurchase_frequency",
    "repurchase_terms.repurchase_amount_pct",
    "repurchase_terms.repurchase_basis",
    "repurchase_terms.repurchase_percentage_min",
    "repurchase_terms.repurchase_percentage_max",
    "repurchase_terms.lock_up_period_years",
    "repurchase_terms.early_repurchase_fee_pct",

    # Leverage limits (ALL) - use leverage_limits prefix to match extraction output
    "leverage_limits.uses_leverage",
    "leverage_limits.max_leverage_pct",
    "leverage_limits.leverage_basis",

    # Distribution terms (ALL)
    "distribution_terms.distribution_frequency",
    "distribution_terms.default_distribution_policy",

    # Allocation targets (ALL)
    "allocation_targets.secondary_funds_min_pct",
    "allocation_targets.secondary_funds_max_pct",
    "allocation_targets.direct_investments_min_pct",
    "allocation_targets.direct_investments_max_pct",
    "allocation_targets.secondary_investments_min_pct",

    # Concentration limits (ALL)
    "concentration_limits.max_single_asset_pct",
    "concentration_limits.max_single_fund_pct",
    "concentration_limits.max_single_sector_pct",
]

# Share class fields are handled separately due to nested structure
# These are the per-class fields to compare
SHARE_CLASS_FIELDS = [
    "minimum_initial_investment",
    "minimum_additional_investment",
    "sales_load_pct",
    "distribution_fee_pct",
    "management_fee_pct",
    "affe_pct",
    "interest_expense_pct",
    "other_expenses_pct",
    "total_expense_ratio_pct",
    "net_expense_ratio_pct",
    "fee_waiver_pct",
    "incentive_fee_xbrl_pct",
]

# Share class fee fields that are typically the same across all classes.
# Once extracted for one class, propagate to other classes instead of re-extracting.
# Fields like total_expense_ratio_pct and net_expense_ratio_pct are excluded because
# they differ by class (distribution fees vary per class).
UNIVERSAL_FEE_FIELDS = {
    "management_fee_pct",
    "affe_pct",
    "interest_expense_pct",
    "other_expenses_pct",
    "incentive_fee_xbrl_pct",
}

# Boolean fields that should be adversarially validated when T3/Reranker agree
# These are prone to hallucination due to weak programmatic grounding
BOOLEAN_FIELDS = [
    "incentive_fee.has_incentive_fee",
    "incentive_fee.high_water_mark",
    "incentive_fee.has_catch_up",
    "expense_cap.has_expense_cap",
    "leverage_limits.uses_leverage",
]

# When a parent boolean is false and both methods agree, skip child fields
# (they must be null — no need to escalate to T4)
BOOLEAN_GATED_FIELDS = {
    "incentive_fee.has_incentive_fee": [
        "incentive_fee.incentive_fee_pct",
        "incentive_fee.hurdle_rate_pct",
        "incentive_fee.hurdle_rate_as_stated",
        "incentive_fee.hurdle_rate_frequency",
        "incentive_fee.high_water_mark",
        "incentive_fee.has_catch_up",
        "incentive_fee.catch_up_rate_pct",
        "incentive_fee.catch_up_ceiling_pct",
        "incentive_fee.fee_basis",
        "incentive_fee.crystallization_frequency",
        "incentive_fee.underlying_fund_incentive_range",
    ],
    "expense_cap.has_expense_cap": [
        "expense_cap.expense_cap_pct",
    ],
    "leverage_limits.uses_leverage": [
        "leverage_limits.max_leverage_pct",
        "leverage_limits.leverage_basis",
    ],
    "incentive_fee.has_catch_up": [
        "incentive_fee.catch_up_rate_pct",
        "incentive_fee.catch_up_ceiling_pct",
    ],
}

# Mapping from field path prefixes to section title keywords for chunk retrieval
# Used to get fuller evidence context for LLM-as-a-judge validation
FIELD_TO_SECTION_KEYWORDS = {
    "incentive_fee": [
        "fee", "incentive", "performance", "adviser", "advisory", "management",
        "expense", "compensation", "carried interest", "hurdle", "catch-up",
    ],
    "expense_cap": [
        "fee", "expense", "limitation", "waiver", "reimbursement", "cap",
        "adviser", "advisory", "management",
    ],
    "repurchase_terms": [
        "repurchase", "tender", "offer", "redemption", "interval", "liquidity",
        "repurchases of shares", "periodic repurchase",
    ],
    "leverage_limits": [
        "leverage", "borrowing", "debt", "credit facility", "1940 act",
        "asset coverage",
    ],
    "distribution_terms": [
        "distribution", "dividend", "reinvestment", "drip", "income",
    ],
    "share_classes": [
        "share", "class", "plan of distribution", "sales load", "fee",
        "minimum", "investment", "purchase",
    ],
    "allocation_targets": [
        "investment", "allocation", "strategy", "portfolio", "objective",
    ],
    "concentration_limits": [
        "concentration", "diversification", "limit", "restriction",
    ],
}

# =============================================================================
# HYBRID ROUTING: Field-Type Based Method Preference
# =============================================================================
#
# Based on empirical analysis comparing MQ Holistic vs Baseline (T3+Reranker):
# - MQ Holistic excels at: boolean/categorical, distribution fees, cross-reference
# - Baseline excels at: precise numeric extraction, basis/type fields
#
# When methods disagree, prefer the method that historically works better for
# that field type, rather than escalating to T4.

# Fields where MQ Holistic (multi-query) is preferred
# These benefit from broader context and cross-document understanding
MQ_HOLISTIC_PREFERRED_FIELDS = [
    # Boolean fields - require understanding full document context
    "incentive_fee.has_incentive_fee",
    "incentive_fee.high_water_mark",
    "incentive_fee.has_catch_up",
    "expense_cap.has_expense_cap",
    "leverage_limits.uses_leverage",

    # Distribution/fee percentages - scattered across document
    "share_classes.distribution_fee_pct",  # Share class distribution fees

    # Frequency/policy fields - may require cross-reference
    "distribution_terms.distribution_frequency",
    "repurchase_terms.lock_up_period_years",

    # Fields requiring document-wide understanding
    "incentive_fee.underlying_fund_incentive_range",
]

# Fields where Baseline (Reranker) is preferred
# These have explicit single-location values that precise retrieval finds better
BASELINE_PREFERRED_FIELDS = [
    # Hurdle rate fields - explicit numeric values
    "incentive_fee.hurdle_rate_as_stated",
    "incentive_fee.hurdle_rate_frequency",

    # Leverage fields - explicit in prospectus
    "leverage_limits.leverage_basis",
    "leverage_limits.max_leverage_pct",

    # Basis/type fields - explicit categorical
    "repurchase_terms.repurchase_basis",
]

# Map comparison field paths to T4 field specs
# Note: Some T4 specs may not exist yet - those fields will skip T4 and use fallback
FIELD_PATH_TO_T4_SPEC = {
    # Fund type
    "fund_type": "fund_type",

    # Incentive fee
    "incentive_fee.has_incentive_fee": "has_incentive_fee",
    "incentive_fee.incentive_fee_pct": "incentive_fee_pct",
    "incentive_fee.hurdle_rate_pct": "hurdle_rate_pct",
    "incentive_fee.hurdle_rate_as_stated": "hurdle_rate_as_stated",
    "incentive_fee.hurdle_rate_frequency": "hurdle_rate_frequency",
    "incentive_fee.high_water_mark": "high_water_mark",
    "incentive_fee.has_catch_up": "has_catch_up",
    "incentive_fee.catch_up_rate_pct": "catch_up_rate_pct",
    "incentive_fee.catch_up_ceiling_pct": "catch_up_ceiling_pct",
    "incentive_fee.fee_basis": "fee_basis",
    "incentive_fee.crystallization_frequency": "crystallization_frequency",
    "incentive_fee.underlying_fund_incentive_range": "underlying_fund_incentive_range",

    # Expense cap
    "expense_cap.has_expense_cap": "has_expense_cap",
    "expense_cap.expense_cap_pct": "expense_cap_pct",

    # Repurchase terms
    "repurchase_terms.repurchase_frequency": "repurchase_frequency",
    "repurchase_terms.repurchase_amount_pct": "repurchase_amount_pct",
    "repurchase_terms.repurchase_basis": "repurchase_basis",
    "repurchase_terms.repurchase_percentage_min": "repurchase_percentage_min",
    "repurchase_terms.repurchase_percentage_max": "repurchase_percentage_max",
    "repurchase_terms.lock_up_period_years": "lock_up_period_years",
    "repurchase_terms.early_repurchase_fee_pct": "early_repurchase_fee_pct",

    # Leverage limits
    "leverage_limits.uses_leverage": "uses_leverage",
    "leverage_limits.max_leverage_pct": "max_leverage_pct",
    "leverage_limits.leverage_basis": "leverage_basis",

    # Distribution terms
    "distribution_terms.distribution_frequency": "distribution_frequency",
    "distribution_terms.default_distribution_policy": "default_distribution_policy",

    # Allocation targets
    "allocation_targets.secondary_funds_min_pct": "secondary_funds_min_pct",
    "allocation_targets.secondary_funds_max_pct": "secondary_funds_max_pct",
    "allocation_targets.direct_investments_min_pct": "direct_investments_min_pct",
    "allocation_targets.direct_investments_max_pct": "direct_investments_max_pct",
    "allocation_targets.secondary_investments_min_pct": "secondary_investments_min_pct",

    # Concentration limits
    "concentration_limits.max_single_asset_pct": "max_single_asset_pct",
    "concentration_limits.max_single_fund_pct": "max_single_fund_pct",
    "concentration_limits.max_single_sector_pct": "max_single_sector_pct",

    # Share class fields (for per-class escalation)
    "minimum_initial_investment": "minimum_investment",
    "minimum_additional_investment": "minimum_additional_investment",
    "management_fee_pct": "management_fee_pct",
    "affe_pct": "affe_pct",
    "interest_expense_pct": "interest_expense_pct",
    "other_expenses_pct": "other_expenses_pct",
    "total_expense_ratio_pct": "total_expense_ratio_pct",
    "net_expense_ratio_pct": "net_expense_ratio_pct",
    "fee_waiver_pct": "fee_waiver_pct",
    "incentive_fee_xbrl_pct": "incentive_fee_xbrl_pct",
}


# =============================================================================
# XBRL FEE MERGE HELPER
# =============================================================================

# Mapping from iXBRL parser field names to ShareClassDetails field names
_XBRL_TO_SCHEMA_FEE_MAP = {
    "management_fee_pct": "management_fee_pct",
    "affe_pct": "affe_pct",
    "interest_expense_pct": "interest_expense_pct",
    "other_expenses_pct": "other_expenses_pct",
    "total_expense_ratio_pct": "total_expense_ratio_pct",
    "net_expense_ratio_pct": "net_expense_ratio_pct",
    "fee_waiver_pct": "fee_waiver_pct",
    "incentive_fee_pct": "incentive_fee_xbrl_pct",  # rename to avoid conflict with fund-level field
    "sales_load_pct": "sales_load_pct",
    "distribution_servicing_fee_pct": "distribution_servicing_fee_pct",
}


def _merge_xbrl_fees_into_share_classes(result: dict, xbrl_values: dict) -> None:
    """Merge XBRL T0 fee values into share class dicts, filling None fields only.

    xbrl_values has structure from ixbrl_parser:
    {
        "share_class_fee_table": {
            "by_class": {"Class A": {"management_fee_pct": {"value": 1.62}, ...}},
            "fund_level": {"management_fee_pct": {"value": 1.25}, ...}
        },
        "numeric_fields": {"management_fee_pct": {"Class A": {"value": 1.62}}},
        ...
    }
    """
    share_classes_data = result.get("share_classes", {})
    if isinstance(share_classes_data, dict):
        sc_list = share_classes_data.get("share_classes", [])
    else:
        return

    if not sc_list or not xbrl_values:
        return

    # Use share_class_fee_table if available (preferred), else fall back to numeric_fields
    fee_table = xbrl_values.get("share_class_fee_table", {})
    by_class = fee_table.get("by_class", {})
    fund_level = fee_table.get("fund_level", {})

    # If no fee table, try building from numeric_fields
    if not by_class and not fund_level:
        numeric_fields = xbrl_values.get("numeric_fields", {})
        for field_name, field_data in numeric_fields.items():
            if not isinstance(field_data, dict):
                continue
            # Check if it's per-class (has class name keys) or fund-level (has "value" key)
            if "value" in field_data:
                fund_level[field_name] = field_data
            else:
                for cls_name, cls_data in field_data.items():
                    if cls_name not in by_class:
                        by_class[cls_name] = {}
                    by_class[cls_name][field_name] = cls_data

    merged_count = 0
    for sc in sc_list:
        if not isinstance(sc, dict):
            continue
        class_name = sc.get("class_name", "")
        if not class_name:
            continue

        # Find XBRL fees for this class (exact then case-insensitive)
        xbrl_class_fees = by_class.get(class_name)
        if not xbrl_class_fees:
            for xbrl_name, xbrl_fees in by_class.items():
                if xbrl_name.lower() == class_name.lower():
                    xbrl_class_fees = xbrl_fees
                    break
        if not xbrl_class_fees:
            xbrl_class_fees = {}

        # Fill None fields with XBRL values (class-level first, then fund-level)
        for xbrl_field, schema_field in _XBRL_TO_SCHEMA_FEE_MAP.items():
            if sc.get(schema_field) is not None:
                continue
            # Try class-level
            fee_entry = xbrl_class_fees.get(xbrl_field) or fund_level.get(xbrl_field)
            if fee_entry and isinstance(fee_entry, dict) and "value" in fee_entry:
                sc[schema_field] = fee_entry["value"]
                merged_count += 1
                logger.info(f"[XBRL Merge] {class_name}.{schema_field} = {fee_entry['value']} (from T0)")

    if merged_count > 0:
        logger.info(f"[XBRL Merge] Merged {merged_count} fee fields from XBRL into share classes")


# =============================================================================
# ENSEMBLE EXTRACTOR
# =============================================================================


class EnsembleT4Extractor:
    """
    Ensemble extractor with T4 escalation.

    Runs T3 and Reranker extractions, compares results, and escalates
    disagreements and both-null cases to T4 agentic extraction.
    """

    def __init__(
        self,
        config: EnsembleT4Config,
        api_key: Optional[str] = None,
    ):
        """
        Initialize ensemble extractor.

        Args:
            config: Ensemble configuration
            api_key: API key for LLM calls (uses env var if None)
        """
        self.config = config
        self.api_key = api_key
        self._t4_agent = None
        self._adversarial_validator = None
        self._hybrid_extractor = None
        self._multi_query_extractor = None
        self._grounding_strategy = None
        self._two_pass_extractor = None

        # Initialize grounding strategy if configured
        if config.grounding_strategy:
            grounding_config = GroundingStrategyConfig(
                strategy=config.grounding_strategy,
                nli_model=config.grounding_nli_model,
                nli_entailment_threshold=config.grounding_nli_entailment_threshold,
                nli_contradiction_threshold=config.grounding_nli_contradiction_threshold,
                llm_judge_model=config.grounding_llm_judge_model,
                llm_judge_provider=config.grounding_llm_judge_provider,
                hybrid_nli_high_threshold=config.grounding_hybrid_nli_high_threshold,
                hybrid_nli_low_threshold=config.grounding_hybrid_nli_low_threshold,
                hybrid_use_llm_for_ambiguous=config.grounding_hybrid_use_llm_for_ambiguous,
            )
            self._grounding_strategy = create_grounding_strategy(grounding_config, api_key)
            logger.info(f"[Ensemble] Initialized grounding strategy: {config.grounding_strategy}")

    @property
    def multi_query_extractor(self):
        """Lazy-load multi-query extractor when multi-query is enabled."""
        if self._multi_query_extractor is None and self.config.use_multi_query:
            from .multi_query_extractor import MultiQueryExtractor, MultiQueryConfig
            from .llm_provider import RateLimitConfig

            mq_config = MultiQueryConfig(
                retrieval_strategy=self.config.multi_query_retrieval_strategy,
                expansion_method=self.config.multi_query_expansion_method,
                rrf_k=self.config.multi_query_rrf_k,
                per_query_top_k=self.config.multi_query_per_query_top_k,
                final_top_k=self.config.multi_query_final_top_k,
                embedding_provider=self.config.multi_query_embedding_provider,
                embedding_model=self.config.multi_query_embedding_model,
            )

            rate_limit = RateLimitConfig(
                delay_between_calls=self.config.delay_between_calls,
                requests_per_minute=self.config.requests_per_minute,
            )

            self._multi_query_extractor = MultiQueryExtractor(
                model=self.config.extraction_model,
                multi_query_config=mq_config,
                rate_limit=rate_limit,
                holistic_extraction=self.config.multi_query_holistic_extraction,
            )
            logger.info(
                f"[Ensemble] Initialized multi-query extractor: "
                f"expansion={self.config.multi_query_expansion_method}, "
                f"retrieval={self.config.multi_query_retrieval_strategy}, "
                f"holistic={self.config.multi_query_holistic_extraction}"
            )
        return self._multi_query_extractor

    @property
    def hybrid_extractor(self):
        """Lazy-load hybrid extractor when hybrid retrieval is enabled."""
        if self._hybrid_extractor is None and self.config.use_hybrid_retrieval:
            from .hybrid_extractor import HybridExtractor
            from .llm_provider import RateLimitConfig

            embedding_config = EmbeddingConfig(
                provider=self.config.hybrid_embedding_provider,
                model=self.config.hybrid_embedding_model,
                top_k=self.config.hybrid_dense_top_k,
                cache_embeddings=True,
            )

            hybrid_config = HybridConfig(
                rrf_k=self.config.hybrid_rrf_k,
                keyword_top_k=self.config.hybrid_keyword_top_k,
                dense_top_k=self.config.hybrid_dense_top_k,
                final_top_k=self.config.hybrid_final_top_k,
            )

            rate_limit = RateLimitConfig(
                delay_between_calls=self.config.delay_between_calls,
                requests_per_minute=self.config.requests_per_minute,
            )

            self._hybrid_extractor = HybridExtractor(
                model=self.config.extraction_model,
                embedding_config=embedding_config,
                hybrid_config=hybrid_config,
                rate_limit=rate_limit,
            )
            logger.info(
                f"[Ensemble] Initialized hybrid extractor with "
                f"{self.config.hybrid_embedding_provider}/{self.config.hybrid_embedding_model}"
            )
        return self._hybrid_extractor

    @property
    def adversarial_validator(self):
        """Lazy-load LLM-as-a-judge validator (full adversarial by default, or lightweight)."""
        if self._adversarial_validator is None:
            adv_config = self.config.adversarial_validation
            if adv_config and adv_config.enabled:
                if adv_config.lightweight:
                    # Use lightweight validator (faster, cheaper, less strict)
                    from .adversarial_validator import LightweightValidator
                    self._adversarial_validator = LightweightValidator(
                        model=adv_config.lightweight_model,
                    )
                    logger.info(
                        f"[Ensemble] Using lightweight validator with {adv_config.lightweight_model}"
                    )
                else:
                    # Use full LLM-as-a-judge validator (two-phase: quote + adversarial critique)
                    from .adversarial_validator import AdversarialValidator
                    self._adversarial_validator = AdversarialValidator(
                        model=adv_config.model,
                        require_exact_quote=adv_config.require_exact_quote,
                    )
                    logger.info(
                        f"[Ensemble] Using LLM-as-a-judge validator with {adv_config.model} "
                        f"(require_exact_quote={adv_config.require_exact_quote})"
                    )
        return self._adversarial_validator

    @property
    def two_pass_extractor(self):
        """Lazy-load two-pass share class extractor when enabled."""
        if self._two_pass_extractor is None and self.config.share_class_two_pass_enabled:
            two_pass_config = TwoPassConfig(
                extraction_model=self.config.extraction_model,
                discovery_model=self.config.share_class_two_pass_discovery_model,
                delay_between_calls=self.config.delay_between_calls,
                requests_per_minute=self.config.requests_per_minute,
                discovery_max_chunks=self.config.share_class_two_pass_discovery_max_chunks,
                per_class_max_chunks=self.config.share_class_two_pass_per_class_max_chunks,
            )
            self._two_pass_extractor = TwoPassShareClassExtractor(
                config=two_pass_config,
                api_key=self.api_key,
            )
            logger.info(
                f"[Ensemble] Initialized two-pass share class extractor: "
                f"discovery_chunks={two_pass_config.discovery_max_chunks}, "
                f"per_class_chunks={two_pass_config.per_class_max_chunks}"
            )
        return self._two_pass_extractor

    def extract(
        self,
        chunked_doc: ChunkedDocument,
        xbrl_values: dict,
        fund_name: str,
        html_content: str,
    ) -> tuple[EnsembleExtractionResult, EnsembleTrace]:
        """
        Run ensemble extraction on a document.

        Args:
            chunked_doc: Chunked document
            xbrl_values: Pre-extracted XBRL values
            fund_name: Fund name
            html_content: Raw HTML content

        Returns:
            Tuple of (extraction_result, trace)
        """
        start_time = time.time()

        logger.info(f"[Ensemble] Starting extraction for {fund_name}")

        # Step 0: Run share class discovery (if enabled)
        # This identifies which share classes exist BEFORE extraction
        discovered_share_classes: Optional[list[str]] = None
        discovery_result: Optional[DiscoveryResult] = None

        if self.config.share_class_discovery_first_enabled:
            logger.info("[Ensemble] Running share class discovery (Step 0)...")
            discovery_start = time.time()

            discovery_model = (
                self.config.share_class_discovery_model
                or self.config.extraction_model
            )
            discovery_result = discover_share_classes(
                chunked_doc=chunked_doc,
                model=discovery_model,
                max_chunks=self.config.share_class_discovery_max_chunks,
                delay_between_calls=self.config.delay_between_calls,
                requests_per_minute=self.config.requests_per_minute,
            )

            discovery_duration = time.time() - discovery_start
            discovered_share_classes = discovery_result.share_class_names

            if discovered_share_classes:
                logger.info(
                    f"[Ensemble] Discovery found {len(discovered_share_classes)} classes "
                    f"in {discovery_duration:.1f}s: {discovered_share_classes}"
                )
            else:
                logger.warning(
                    f"[Ensemble] Discovery found no share classes in {discovery_duration:.1f}s"
                )

        # Step 1: Run first extraction method (Multi-Query, Hybrid, or T3)
        # Pass discovered_share_classes for targeted share class extraction
        if self.config.use_multi_query:
            # Use multi-query expansion with RRF fusion
            logger.info(
                f"[Ensemble] Running Multi-Query extraction "
                f"(expansion={self.config.multi_query_expansion_method}, "
                f"retrieval={self.config.multi_query_retrieval_strategy})..."
            )
            t3_start = time.time()
            t3_result, t3_trace = self._run_multi_query_extraction(
                chunked_doc, xbrl_values, fund_name, html_content,
                discovered_share_classes=discovered_share_classes,
            )
            t3_duration = time.time() - t3_start
            logger.info(f"[Ensemble] Multi-Query completed in {t3_duration:.1f}s")
        elif self.config.use_hybrid_retrieval:
            # Use hybrid retrieval (keyword + dense + RRF fusion)
            logger.info("[Ensemble] Running Hybrid (keyword+dense+RRF) extraction...")
            t3_start = time.time()
            t3_result, t3_trace = self._run_hybrid_extraction(
                chunked_doc, xbrl_values, fund_name, html_content,
                discovered_share_classes=discovered_share_classes,
            )
            t3_duration = time.time() - t3_start
            logger.info(f"[Ensemble] Hybrid completed in {t3_duration:.1f}s")
        else:
            # Use keyword-only T3 extraction
            logger.info("[Ensemble] Running T3 k=10 extraction...")
            t3_start = time.time()
            t3_result, t3_trace = self._run_t3_extraction(
                chunked_doc, xbrl_values, fund_name, html_content,
                discovered_share_classes=discovered_share_classes,
            )
            t3_duration = time.time() - t3_start
            logger.info(f"[Ensemble] T3 completed in {t3_duration:.1f}s")

        # Step 2: Run Reranker extraction (if configured)
        run_reranker = "reranker" in self.config.methods
        rr_result = {}
        rr_trace = None
        rr_duration = 0.0

        if run_reranker:
            logger.info("[Ensemble] Running Reranker 200->10 extraction...")
            rr_start = time.time()
            rr_result, rr_trace = self._run_reranker_extraction(
                chunked_doc, xbrl_values, fund_name, html_content,
                discovered_share_classes=discovered_share_classes,
            )
            rr_duration = time.time() - rr_start
            logger.info(f"[Ensemble] Reranker completed in {rr_duration:.1f}s")
        else:
            logger.info("[Ensemble] Skipping Reranker (single-method mode)")

        # Step 3: Compare results and decide escalations
        if run_reranker:
            field_results, fields_to_escalate = self._compare_and_decide(
                t3_result, rr_result, chunked_doc=chunked_doc
            )
        else:
            # Single-method mode: use multi-query results directly
            field_results = self._convert_to_field_results(t3_result)
            fields_to_escalate = []

        logger.info(
            f"[Ensemble] Comparison: {len(field_results) - len(fields_to_escalate)} accepted, "
            f"{len(fields_to_escalate)} to escalate"
        )

        # Step 4: Run T4 on escalated fields (if enabled)
        t4_duration = 0.0
        t4_success_count = 0
        if fields_to_escalate and self.config.run_t4_escalation:
            logger.info(f"[Ensemble] Running T4 on {len(fields_to_escalate)} fields...")
            t4_start = time.time()
            t4_success_count = self._run_t4_escalation(
                chunked_doc, field_results, fields_to_escalate
            )
            t4_duration = time.time() - t4_start
            logger.info(f"[Ensemble] T4 completed in {t4_duration:.1f}s, {t4_success_count} successful")
        elif fields_to_escalate and not self.config.run_t4_escalation:
            logger.info(f"[Ensemble] Skipping T4 escalation (disabled by config), {len(fields_to_escalate)} fields would have been escalated")

        # Step 4.5: Run two-pass share class extraction (if enabled)
        two_pass_result = None
        if self.config.share_class_two_pass_enabled:
            logger.info("[Ensemble] Running two-pass share class extraction...")
            two_pass_start = time.time()
            two_pass_result = self._run_two_pass_share_class_extraction(chunked_doc)
            two_pass_duration = time.time() - two_pass_start
            logger.info(
                f"[Ensemble] Two-pass completed in {two_pass_duration:.1f}s, "
                f"{len(two_pass_result.share_classes) if two_pass_result else 0} classes extracted"
            )

        # Step 5: Build final extraction result
        final_extraction = self._build_final_extraction(
            t3_result, rr_result, field_results, xbrl_values, fund_name,
            two_pass_result=two_pass_result,
            discovered_share_classes=discovered_share_classes,
        )

        # Step 5b: Apply Phase 4 validation rules (disambiguation, normalization)
        from .validation_rules import apply_validation_rules
        final_extraction["fund_name"] = fund_name  # Needed by validation rules for strategy detection
        final_extraction, validation_report = apply_validation_rules(final_extraction)
        if validation_report.correction_count > 0:
            final_extraction["validation_corrections"] = validation_report.to_dict()
            logger.info(
                f"[Ensemble] Validation: {validation_report.correction_count} corrections applied "
                f"(strategy: {validation_report.fund_strategy})"
            )

        total_duration = time.time() - start_time

        # Calculate discovery duration if discovery was run
        discovery_duration = (
            discovery_result.duration_ms / 1000.0
            if discovery_result else 0.0
        )

        # Build result
        result = EnsembleExtractionResult(
            fund_name=fund_name,
            extraction=final_extraction,
            field_results=field_results,
            total_fields=len(field_results),
            accepted_count=len(field_results) - len(fields_to_escalate),
            escalated_count=len(fields_to_escalate),
            t4_success_count=t4_success_count,
            t3_duration_seconds=t3_duration,
            reranker_duration_seconds=rr_duration,
            t4_duration_seconds=t4_duration,
            total_duration_seconds=total_duration,
        )

        # Build trace with discovery info
        trace = self._build_trace(
            result, field_results,
            discovered_share_classes=discovered_share_classes,
            discovery_result=discovery_result,
            discovery_duration=discovery_duration,
        )

        # Count LLM-as-a-judge validation results for logging
        validated_fields = sum(1 for fr in field_results.values() if fr.adversarial_validated is not None)
        rejected_by_judge = sum(1 for fr in field_results.values() if fr.escalation_reason == "llm_judge_rejected")

        logger.info(
            f"[Ensemble] Complete: {result.accepted_count} accepted, "
            f"{result.escalated_count} escalated ({rejected_by_judge} by LLM-as-a-judge), "
            f"{result.t4_success_count} T4 successes, "
            f"{validated_fields} fields validated, "
            f"total {total_duration:.1f}s"
        )

        return result, trace

    def _run_t3_extraction(
        self,
        chunked_doc: ChunkedDocument,
        xbrl_values: dict,
        fund_name: str,
        html_content: str,
        discovered_share_classes: Optional[list[str]] = None,
    ) -> tuple[dict, Optional[dict]]:
        """Run standard T3 extraction with k=10."""
        from .extractor import DocumentExtractor

        # Log if discovery provided classes (will be used for validation)
        if discovered_share_classes:
            logger.info(
                f"[T3] Extraction informed by discovery: {discovered_share_classes}"
            )

        extractor = DocumentExtractor(
            api_key=self.api_key,
            model=self.config.extraction_model,
            provider=self.config.extraction_provider,
            use_examples=True,
            enable_grounding=True,
            enable_observability=True,
            tier0_enabled=True,
            tier1_enabled=False,
            tier2_enabled=False,
            tier3_enabled=True,
            tier3_only=True,
            tier3_top_k_sections=self.config.t3_top_k_sections,
            tier3_max_chunks_per_section=self.config.t3_max_chunks_per_section,
            delay_between_calls=self.config.delay_between_calls,
            requests_per_minute=self.config.requests_per_minute,
        )

        result = extractor.extract(
            chunked_doc=chunked_doc,
            xbrl_values=xbrl_values,
            fund_name=fund_name,
            html_content=html_content,
        )

        trace = None
        if extractor.current_trace:
            trace = extractor.current_trace.to_dict()

        return result, trace

    def _run_reranker_extraction(
        self,
        chunked_doc: ChunkedDocument,
        xbrl_values: dict,
        fund_name: str,
        html_content: str,
        discovered_share_classes: Optional[list[str]] = None,
    ) -> tuple[dict, Optional[dict]]:
        """Run T3 extraction with Cohere reranker."""
        from .extractor import DocumentExtractor

        # Log if discovery provided classes (will be used for validation)
        if discovered_share_classes:
            logger.info(
                f"[Reranker] Extraction informed by discovery: {discovered_share_classes}"
            )

        reranker_config = RerankerConfig(
            enabled=True,
            model=self.config.reranker_model,
            first_pass_n=self.config.reranker_first_pass_n,
            top_k=self.config.reranker_top_k,
            score_threshold=self.config.reranker_score_threshold,
        )

        extractor = DocumentExtractor(
            api_key=self.api_key,
            model=self.config.extraction_model,
            provider=self.config.extraction_provider,
            use_examples=True,
            enable_grounding=True,
            enable_observability=True,
            tier0_enabled=True,
            tier1_enabled=False,
            tier2_enabled=False,
            tier3_enabled=True,
            tier3_only=True,
            tier3_top_k_sections=self.config.t3_top_k_sections,
            tier3_max_chunks_per_section=self.config.t3_max_chunks_per_section,
            reranker_config=reranker_config,
            delay_between_calls=self.config.delay_between_calls,
            requests_per_minute=self.config.requests_per_minute,
        )

        result = extractor.extract(
            chunked_doc=chunked_doc,
            xbrl_values=xbrl_values,
            fund_name=fund_name,
            html_content=html_content,
        )

        trace = None
        if extractor.current_trace:
            trace = extractor.current_trace.to_dict()

        return result, trace

    def _run_hybrid_extraction(
        self,
        chunked_doc: ChunkedDocument,
        xbrl_values: dict,
        fund_name: str,
        html_content: str,
        discovered_share_classes: Optional[list[str]] = None,
    ) -> tuple[dict, Optional[dict]]:
        """
        Run hybrid retrieval (keyword + dense + RRF fusion) extraction.

        Uses HybridExtractor which combines keyword and embedding-based retrieval
        with Reciprocal Rank Fusion to capture chunks found by either method.

        This replaces T3 keyword-only extraction when use_hybrid_retrieval is enabled.
        """
        # Log if discovery provided classes (will be used for validation)
        if discovered_share_classes:
            logger.info(
                f"[Hybrid] Extraction informed by discovery: {discovered_share_classes}"
            )

        if not self.hybrid_extractor:
            logger.error("[Ensemble] Hybrid extractor not initialized")
            return {}, None

        # Run hybrid extraction
        hybrid_result = self.hybrid_extractor.extract(
            chunked_doc=chunked_doc,
            fund_name=fund_name,
        )

        # Convert HybridExtractionResult to the format expected by comparison
        # The hybrid extractor returns a result with .extraction dict
        result = hybrid_result.extraction

        # Add xbrl_values (tier 0) to result
        if xbrl_values:
            result["xbrl_fees"] = xbrl_values
            _merge_xbrl_fees_into_share_classes(result, xbrl_values)

        # Build trace dict with hybrid-specific info
        trace = {
            "extraction_mode": "hybrid",
            "retrieval_stats": hybrid_result.retrieval_stats,
            "total_time_s": hybrid_result.total_time_s,
            "field_traces": {
                field: {
                    "chunks_retrieved": t.chunks_retrieved,
                    "retrieval_sources": t.retrieval_sources,
                    "extraction_time_ms": t.extraction_time_ms,
                }
                for field, t in hybrid_result.traces.items()
            },
        }

        return result, trace

    def _run_multi_query_extraction(
        self,
        chunked_doc: ChunkedDocument,
        xbrl_values: dict,
        fund_name: str,
        html_content: str,
        discovered_share_classes: Optional[list[str]] = None,
    ) -> tuple[dict, Optional[dict]]:
        """
        Run multi-query expansion extraction.

        Uses MultiQueryExtractor which generates multiple query variations
        per field and fuses results with RRF.

        This replaces T3 keyword-only extraction when use_multi_query is enabled.
        """
        # Log if discovery provided classes (will be used for validation)
        if discovered_share_classes:
            logger.info(
                f"[MultiQuery] Extraction informed by discovery: {discovered_share_classes}"
            )

        if not self.multi_query_extractor:
            logger.error("[Ensemble] Multi-query extractor not initialized")
            return {}, None

        # Run multi-query extraction
        mq_result = self.multi_query_extractor.extract(
            chunked_doc=chunked_doc,
            fund_name=fund_name,
            cik="",  # Will be filled by caller
            filing_id="",
        )

        # Convert result to the format expected by comparison
        result = mq_result.extraction

        # Add xbrl_values (tier 0) to result
        if xbrl_values:
            result["xbrl_fees"] = xbrl_values
            _merge_xbrl_fees_into_share_classes(result, xbrl_values)

        # Build trace dict with multi-query-specific info
        trace = {
            "extraction_mode": "multi_query",
            "expansion_method": self.config.multi_query_expansion_method,
            "retrieval_strategy": self.config.multi_query_retrieval_strategy,
            "retrieval_stats": mq_result.retrieval_stats,
            "expansion_stats": mq_result.expansion_stats,
            "total_time_s": mq_result.total_time_s,
            "field_traces": {
                field: {
                    "num_queries": t.num_queries,
                    "expansion_source": t.expansion_source,
                    "chunks_retrieved": t.chunks_retrieved,
                    "queries_contributed": t.queries_contributed,
                    "extraction_time_ms": t.extraction_time_ms,
                }
                for field, t in mq_result.traces.items()
            },
        }

        return result, trace

    def _run_two_pass_share_class_extraction(
        self,
        chunked_doc: ChunkedDocument,
    ):
        """
        Run two-pass share class extraction.

        This method uses the TwoPassShareClassExtractor to:
        1. Discover all share class names in the document
        2. Extract fields for each discovered class with targeted retrieval

        Returns:
            TwoPassExtractionResult with discovered and extracted share classes
        """
        if not self.two_pass_extractor:
            logger.error("[Ensemble] Two-pass extractor not initialized")
            return None

        try:
            result = self.two_pass_extractor.extract(chunked_doc)
            return result
        except Exception as e:
            logger.error(f"[Ensemble] Two-pass extraction failed: {e}")
            return None

    def _compare_and_decide(
        self,
        t3_result: dict,
        rr_result: dict,
        chunked_doc: Optional[ChunkedDocument] = None,
    ) -> tuple[dict[str, FieldResult], list[str]]:
        """
        Compare T3 and Reranker results, decide which fields to escalate.

        Includes adversarial validation for boolean fields when T3/Reranker agree,
        to catch false confidence cases where both methods agree on wrong value.

        Args:
            t3_result: T3 extraction result dict
            rr_result: Reranker extraction result dict
            chunked_doc: Optional chunked document for retrieving fuller evidence context

        Returns:
            Tuple of (field_results dict, list of field paths to escalate)
        """
        field_results = {}
        fields_to_escalate = []
        skip_fields: set[str] = set()  # Fields to skip due to parent boolean = false

        # Check if adversarial validation is enabled
        adv_config = self.config.adversarial_validation
        run_adversarial = (
            adv_config
            and adv_config.enabled
            and self.adversarial_validator
        )

        # Compare standard fields
        for field_path in COMPARISON_FIELDS:
            # Skip fields gated by a parent boolean that both methods agreed is false
            if field_path in skip_fields:
                logger.info(f"[Ensemble] Skipping '{field_path}' — parent boolean is false")
                field_results[field_path] = FieldResult(
                    field_name=field_path,
                    t3_value=None,
                    reranker_value=None,
                    final_value=None,
                    values_agree=True,
                    both_null=True,
                    escalated_to_t4=False,
                    escalation_reason="none",
                    source="parent_false",
                )
                continue
            t3_value = get_nested_value(t3_result, field_path)
            rr_value = get_nested_value(rr_result, field_path)

            field_result, should_escalate = self._decide_field(
                field_path, t3_value, rr_value, t3_result, rr_result
            )

            # LLM-as-a-judge validation for accepted (agreed) fields
            # This runs BEFORE T4, escalating fields that fail validation
            should_validate = (
                run_adversarial
                and not should_escalate
                and field_result.values_agree
                and not field_result.both_null
                and (
                    adv_config.validate_all_accepted  # Validate ALL accepted fields
                    or adv_config.validate_all  # Deprecated: same as validate_all_accepted
                    or (adv_config.validate_booleans and field_path in BOOLEAN_FIELDS)
                )
            )

            if should_validate:
                # Get evidence from extraction results, with fuller context from chunks
                evidence = self._get_field_evidence(
                    t3_result, rr_result, field_path, chunked_doc=chunked_doc
                )

                if evidence:
                    # Determine expected type for validation
                    expected_type = "boolean" if field_path in BOOLEAN_FIELDS else "text"
                    if "pct" in field_path or "rate" in field_path:
                        expected_type = "percentage"
                    elif "investment" in field_path or "amount" in field_path:
                        expected_type = "currency"

                    logger.info(
                        f"[Ensemble] Running LLM-as-a-judge validation for '{field_path}' "
                        f"(value={field_result.final_value}, type={expected_type})"
                    )
                    adv_result = self.adversarial_validator.validate(
                        field_name=field_path.split(".")[-1],  # Use short field name
                        value=field_result.final_value,
                        evidence=evidence,
                        expected_type=expected_type,
                    )

                    field_result.adversarial_validated = adv_result.is_valid
                    field_result.adversarial_problems = adv_result.problems
                    field_result.adversarial_confidence = adv_result.confidence

                    if not adv_result.is_valid and adv_config.escalate_on_rejection:
                        # Escalate to T4 - LLM-as-a-judge rejected the agreed value
                        logger.warning(
                            f"[Ensemble] LLM-as-a-judge REJECTED '{field_path}' → escalating to T4: "
                            f"{adv_result.problems}"
                        )
                        should_escalate = True
                        field_result.escalated_to_t4 = True
                        field_result.escalation_reason = "llm_judge_rejected"
                        field_result.final_value = None
                        field_result.source = ""
                    else:
                        logger.info(
                            f"[Ensemble] LLM-as-a-judge PASSED '{field_path}' "
                            f"(confidence={adv_result.confidence:.2f})"
                        )

            # If this is a boolean parent that both methods agreed is false,
            # mark all child fields for skipping
            if (
                field_path in BOOLEAN_GATED_FIELDS
                and field_result.values_agree
                and not field_result.escalated_to_t4
                and field_result.final_value is False
            ):
                children = BOOLEAN_GATED_FIELDS[field_path]
                skip_fields.update(children)
                logger.info(
                    f"[Ensemble] Parent '{field_path}' is false (agreed) — "
                    f"skipping {len(children)} child fields"
                )

            field_results[field_path] = field_result
            if should_escalate:
                fields_to_escalate.append(field_path)

        # Compare share_classes fields (nested structure)
        share_class_results, share_class_escalations = self._compare_share_classes(
            t3_result, rr_result, chunked_doc=chunked_doc
        )
        field_results.update(share_class_results)
        fields_to_escalate.extend(share_class_escalations)

        return field_results, fields_to_escalate

    def _get_field_evidence(
        self,
        t3_result: dict,
        rr_result: dict,
        field_path: str,
        chunked_doc: Optional[ChunkedDocument] = None,
        min_evidence_chars: int = 500,
        max_evidence_chars: int = 4000,
    ) -> Optional[str]:
        """
        Get evidence text for a field from T3 or Reranker results.

        If the stored evidence is too short (< min_evidence_chars), retrieves
        fuller context from the chunked document to provide the LLM-as-a-judge
        validator with enough context to make accurate decisions.

        Also includes T3's reasoning (key_findings, extraction_rationale) to help
        the validator understand WHY the value was extracted.

        Args:
            t3_result: T3 extraction result dict
            rr_result: Reranker extraction result dict
            field_path: Dot-notation field path (e.g., "incentive_fee.has_incentive_fee")
            chunked_doc: Optional chunked document for retrieving fuller context
            min_evidence_chars: Minimum evidence length before retrieving more context
            max_evidence_chars: Maximum evidence length to return

        Returns:
            Evidence text with reasoning context, or None if not found
        """
        evidence = None
        reasoning_context = ""
        extraction_quote = ""

        # Try to get the extraction's own evidence quote (citation.evidence_quote)
        # This is the specific text the extraction LLM cited when producing the value
        parts = field_path.split(".")
        if len(parts) >= 2:
            citation_path = parts[0] + ".citation.evidence_quote"
            quote = get_nested_value(t3_result, citation_path)
            if not quote:
                quote = get_nested_value(rr_result, citation_path)
            if quote and isinstance(quote, str) and len(quote.strip()) > 5:
                extraction_quote = f"EXTRACTION'S CITED EVIDENCE:\n\"{quote.strip()}\"\n\n"

        # FIRST: Try per-field evidence stored by multi-query extractor
        # This is the exact chunk text the extraction LLM saw for this field - most relevant
        if len(parts) >= 2:
            mq_evidence_path = parts[0] + "._evidence." + parts[1]
            mq_evidence = get_nested_value(t3_result, mq_evidence_path)
            if not mq_evidence:
                mq_evidence = get_nested_value(rr_result, mq_evidence_path)
            if mq_evidence and isinstance(mq_evidence, str) and len(mq_evidence) > 20:
                evidence = mq_evidence
                logger.debug(
                    f"[Ensemble] Using multi-query extraction evidence for '{field_path}' "
                    f"({len(evidence)} chars)"
                )

        # Try to get evidence from the parent section (e.g., incentive_fee.evidence)
        if not evidence and len(parts) >= 2:
            section_path = parts[0] + ".evidence"
            evidence = get_nested_value(t3_result, section_path)
            if not evidence:
                evidence = get_nested_value(rr_result, section_path)

            # Get T3's reasoning to help validator understand the extraction
            reasoning_path = parts[0] + ".reasoning"
            reasoning = get_nested_value(t3_result, reasoning_path)
            if reasoning and isinstance(reasoning, dict):
                reasoning_parts = []

                # Add key findings
                key_findings = reasoning.get("key_findings", [])
                if key_findings:
                    findings_str = "; ".join(key_findings[:5])  # Limit to 5
                    reasoning_parts.append(f"Key findings: {findings_str}")

                # Add extraction rationale
                rationale = reasoning.get("extraction_rationale")
                if rationale:
                    reasoning_parts.append(f"Rationale: {rationale}")

                # Add reasoning steps
                steps = reasoning.get("reasoning_steps", [])
                if steps and isinstance(steps, list):
                    for step in steps[:3]:  # Limit to 3 steps
                        if isinstance(step, dict):
                            obs = step.get("observation", "")
                            concl = step.get("conclusion", "")
                            if obs and concl:
                                reasoning_parts.append(f"- Found: {obs} → {concl}")

                if reasoning_parts:
                    reasoning_context = "T3 EXTRACTION REASONING:\n" + "\n".join(reasoning_parts) + "\n\n"

        # Try section_text if available
        if not evidence:
            section_path = parts[0] + ".section_text" if parts else None
            if section_path:
                evidence = get_nested_value(t3_result, section_path)
                if not evidence:
                    evidence = get_nested_value(rr_result, section_path)

        # If evidence is too short, retrieve fuller context from chunks
        if evidence and len(evidence) < min_evidence_chars and chunked_doc:
            fuller_evidence = self._get_chunks_for_field(
                chunked_doc, field_path, max_chars=max_evidence_chars
            )
            if fuller_evidence and len(fuller_evidence) > len(evidence):
                logger.debug(
                    f"[Ensemble] Expanded evidence for '{field_path}': "
                    f"{len(evidence)} -> {len(fuller_evidence)} chars"
                )
                evidence = fuller_evidence

        # If no evidence found at all, try to get from chunks
        if not evidence and chunked_doc:
            evidence = self._get_chunks_for_field(
                chunked_doc, field_path, max_chars=max_evidence_chars
            )

        # Prepend extraction quote and reasoning context to evidence
        if evidence:
            prefix = extraction_quote + reasoning_context
            if prefix:
                evidence = prefix + "SOURCE EVIDENCE:\n" + evidence

        return evidence

    def _get_chunks_for_field(
        self,
        chunked_doc: ChunkedDocument,
        field_path: str,
        max_chars: int = 4000,
        max_chunks: int = 10,
    ) -> Optional[str]:
        """
        Retrieve relevant chunks from the chunked document for a field.

        Uses section title keyword matching to find sections likely to contain
        evidence for the field, then concatenates their chunks.

        Args:
            chunked_doc: The chunked document
            field_path: Dot-notation field path (e.g., "incentive_fee.has_incentive_fee")
            max_chars: Maximum total characters to return
            max_chunks: Maximum number of chunks to include

        Returns:
            Concatenated chunk text, or None if no relevant chunks found
        """
        # Get the field prefix (e.g., "incentive_fee" from "incentive_fee.has_incentive_fee")
        parts = field_path.split(".")
        field_prefix = parts[0] if parts else field_path

        # Get section keywords for this field
        section_keywords = FIELD_TO_SECTION_KEYWORDS.get(field_prefix, [])
        if not section_keywords:
            # Fallback: use the field prefix as a keyword
            section_keywords = [field_prefix.replace("_", " ")]

        # Find relevant sections
        relevant_chunks = []
        for section in chunked_doc.chunked_sections:
            section_title_lower = section.section_title.lower()

            # Check if any keyword matches the section title
            matches = any(kw.lower() in section_title_lower for kw in section_keywords)
            if matches:
                for chunk in section.chunks:
                    relevant_chunks.append({
                        "text": chunk.content,
                        "section": section.section_title,
                        "score": sum(1 for kw in section_keywords if kw.lower() in section_title_lower),
                    })

        if not relevant_chunks:
            return None

        # Sort by score (more keyword matches = higher priority)
        relevant_chunks.sort(key=lambda x: x["score"], reverse=True)

        # Concatenate chunks up to max_chars
        result_parts = []
        total_chars = 0
        chunks_used = 0

        for chunk_info in relevant_chunks[:max_chunks]:
            chunk_text = chunk_info["text"]
            if total_chars + len(chunk_text) > max_chars:
                # Include partial chunk if needed
                remaining = max_chars - total_chars
                if remaining > 200:  # Only include if meaningful amount
                    result_parts.append(chunk_text[:remaining] + "...")
                break

            result_parts.append(chunk_text)
            total_chars += len(chunk_text)
            chunks_used += 1

        if not result_parts:
            return None

        # Join with section separators for clarity
        result = "\n\n---\n\n".join(result_parts)
        logger.debug(
            f"[Ensemble] Retrieved {chunks_used} chunks ({total_chars} chars) for '{field_path}'"
        )

        return result

    def _decide_field(
        self,
        field_path: str,
        t3_value: Any,
        rr_value: Any,
        t3_result: Optional[dict] = None,
        rr_result: Optional[dict] = None,
    ) -> tuple[FieldResult, bool]:
        """
        Decide whether to accept or escalate a single field.

        Routing priority:
        1. Agreement on non-null → accept
        2. Both null → escalate (if configured)
        3. Confidence-based routing (if enabled) → use more confident method
        4. Hybrid routing (if enabled) → use field-type preference
        5. One null, one value → trust the finder (no escalation)
        6. Escalate on value-vs-value disagreement (if configured)
        7. Default: pick non-null value

        CRITICAL for confidence routing: ungrounded = 0 confidence.

        Returns:
            Tuple of (FieldResult, should_escalate)
        """
        agree = values_equal(t3_value, rr_value)
        are_both_null = both_null(t3_value, rr_value)

        # Calculate confidence scores if confidence routing is enabled
        t3_conf = 0.0
        rr_conf = 0.0
        if self.config.use_confidence_routing and t3_result and rr_result:
            t3_conf = calculate_field_confidence(
                t3_result, field_path,
                require_grounding=self.config.confidence_require_grounding,
                grounding_strategy=self._grounding_strategy,
            )
            rr_conf = calculate_field_confidence(
                rr_result, field_path,
                require_grounding=self.config.confidence_require_grounding,
                grounding_strategy=self._grounding_strategy,
            )

        # Decision logic
        if agree and not are_both_null:
            # Accept - both agree on non-null value
            escalation_reason = "none"
            escalated = False
            final_value = t3_value
            source = "agreement"
        elif are_both_null and self.config.escalate_on_both_null:
            # Escalate - both null, need T4 to find value
            escalation_reason = "both_null"
            escalated = True
            final_value = None
            source = ""
        elif not agree and self.config.use_confidence_routing:
            # CONFIDENCE-BASED ROUTING: Use confidence scores to pick winner
            final_value, source, escalated, escalation_reason = self._apply_confidence_routing(
                field_path, t3_value, rr_value, t3_conf, rr_conf
            )
        elif not agree and self.config.use_hybrid_routing:
            # HYBRID ROUTING: Use field-type-based preference instead of escalating
            final_value, source, escalated, escalation_reason = self._apply_hybrid_routing(
                field_path, t3_value, rr_value
            )
        elif not agree and self.config.escalate_on_disagreement:
            # Escalate - true value-vs-value disagreement, need T4 to resolve
            escalation_reason = "disagreement"
            escalated = True
            final_value = None
            source = ""
        else:
            # Default accept (when escalation disabled)
            escalation_reason = "none"
            escalated = False
            final_value = t3_value if t3_value is not None else rr_value
            source = "t3" if t3_value is not None else "reranker"

        result = FieldResult(
            field_name=field_path,
            t3_value=t3_value,
            reranker_value=rr_value,
            final_value=final_value,
            values_agree=agree,
            both_null=are_both_null,
            escalated_to_t4=escalated,
            escalation_reason=escalation_reason,
            source=source,
            t3_confidence=t3_conf,
            reranker_confidence=rr_conf,
        )

        return result, escalated

    def _apply_confidence_routing(
        self,
        field_path: str,
        t3_value: Any,
        rr_value: Any,
        t3_conf: float,
        rr_conf: float,
    ) -> tuple[Any, str, bool, str]:
        """
        Apply confidence-based routing to resolve disagreement.

        Decision logic:
        1. If confidence gap > min_gap → use higher confidence value
        2. If both below low_threshold → escalate (both uncertain)
        3. If both above high_threshold but disagree → escalate (high-stakes disagreement)
        4. Otherwise → use higher confidence value

        CRITICAL: Ungrounded fields have 0 confidence, so they will lose to grounded fields.

        Returns:
            Tuple of (final_value, source, should_escalate, escalation_reason)
        """
        gap = abs(t3_conf - rr_conf)
        max_conf = max(t3_conf, rr_conf)

        logger.debug(
            f"[Confidence] {field_path}: MQ={t3_conf:.2f}, Reranker={rr_conf:.2f}, gap={gap:.2f}"
        )

        # Case 1: Clear winner (significant confidence gap)
        if gap >= self.config.confidence_min_gap:
            if t3_conf > rr_conf:
                if t3_value is not None:
                    logger.info(
                        f"[Confidence] {field_path}: Using MQ (conf={t3_conf:.2f} vs {rr_conf:.2f})"
                    )
                    return t3_value, "mq_confident", False, "none"
                else:
                    # MQ more confident but returned null - use reranker if available
                    if rr_value is not None:
                        return rr_value, "reranker_fallback", False, "none"
                    return None, "", True, "both_null"
            else:
                if rr_value is not None:
                    logger.info(
                        f"[Confidence] {field_path}: Using Reranker (conf={rr_conf:.2f} vs {t3_conf:.2f})"
                    )
                    return rr_value, "reranker_confident", False, "none"
                else:
                    # Reranker more confident but returned null - use MQ if available
                    if t3_value is not None:
                        return t3_value, "mq_fallback", False, "none"
                    return None, "", True, "both_null"

        # Case 2: Both low confidence → escalate
        if max_conf < self.config.confidence_low_threshold:
            logger.info(
                f"[Confidence] {field_path}: Both low confidence ({t3_conf:.2f}, {rr_conf:.2f}) → escalate"
            )
            return None, "", True, "both_low_confidence"

        # Case 3: Both high confidence but disagree → escalate (investigate)
        if (t3_conf >= self.config.confidence_high_threshold and
            rr_conf >= self.config.confidence_high_threshold):
            logger.info(
                f"[Confidence] {field_path}: Both confident ({t3_conf:.2f}, {rr_conf:.2f}) but disagree → escalate"
            )
            return None, "", True, "confident_disagreement"

        # Case 4: No clear signal - use higher confidence (small gap)
        if t3_conf >= rr_conf:
            if t3_value is not None:
                logger.debug(f"[Confidence] {field_path}: Slight MQ preference ({t3_conf:.2f})")
                return t3_value, "mq_slight", False, "none"
            elif rr_value is not None:
                return rr_value, "reranker_fallback", False, "none"
        else:
            if rr_value is not None:
                logger.debug(f"[Confidence] {field_path}: Slight Reranker preference ({rr_conf:.2f})")
                return rr_value, "reranker_slight", False, "none"
            elif t3_value is not None:
                return t3_value, "mq_fallback", False, "none"

        # Both null
        return None, "", True, "both_null"

    def _apply_hybrid_routing(
        self,
        field_path: str,
        t3_value: Any,
        rr_value: Any,
    ) -> tuple[Any, str, bool, str]:
        """
        Apply hybrid routing to resolve disagreement based on field type.

        Returns:
            Tuple of (final_value, source, should_escalate, escalation_reason)
        """
        # Check if field matches MQ Holistic preference patterns
        is_mq_preferred = self._is_mq_preferred_field(field_path)
        is_baseline_preferred = self._is_baseline_preferred_field(field_path)

        if is_mq_preferred:
            # Prefer T3 (multi-query) value for this field type
            if t3_value is not None:
                logger.debug(f"[Hybrid] Using MQ value for {field_path}: {t3_value}")
                return t3_value, "mq_preferred", False, "none"
            elif rr_value is not None:
                # MQ returned null, fall back to reranker
                logger.debug(f"[Hybrid] MQ null, using reranker for {field_path}: {rr_value}")
                return rr_value, "reranker_fallback", False, "none"
            else:
                # Both null - escalate
                return None, "", True, "both_null"

        elif is_baseline_preferred:
            # Prefer Reranker value for this field type
            if rr_value is not None:
                logger.debug(f"[Hybrid] Using reranker value for {field_path}: {rr_value}")
                return rr_value, "reranker_preferred", False, "none"
            elif t3_value is not None:
                # Reranker returned null, fall back to MQ
                logger.debug(f"[Hybrid] Reranker null, using MQ for {field_path}: {t3_value}")
                return t3_value, "mq_fallback", False, "none"
            else:
                # Both null - escalate
                return None, "", True, "both_null"

        else:
            # No preference - use default logic (prefer non-null, escalate if needed)
            if self.config.escalate_on_disagreement:
                return None, "", True, "disagreement"
            else:
                # Pick MQ by default if available
                if t3_value is not None:
                    return t3_value, "t3", False, "none"
                elif rr_value is not None:
                    return rr_value, "reranker", False, "none"
                else:
                    return None, "", False, "none"

    def _is_mq_preferred_field(self, field_path: str) -> bool:
        """Check if field should prefer MQ Holistic extraction."""
        # Direct match
        if field_path in MQ_HOLISTIC_PREFERRED_FIELDS:
            return True

        # Check share class distribution_fee_pct pattern
        if "distribution_fee_pct" in field_path:
            return True

        return False

    def _is_baseline_preferred_field(self, field_path: str) -> bool:
        """Check if field should prefer Baseline (Reranker) extraction."""
        return field_path in BASELINE_PREFERRED_FIELDS

    def _compare_share_classes(
        self,
        t3_result: dict,
        rr_result: dict,
        chunked_doc: Optional[ChunkedDocument] = None,
    ) -> tuple[dict[str, FieldResult], list[str]]:
        """
        Compare share_classes fields between T3 and Reranker.

        Share classes are nested: share_classes.share_classes[i].field_name
        We compare per-class and per-field.

        Also runs LLM-as-a-judge validation on accepted share class fields.

        Args:
            t3_result: T3 extraction result dict
            rr_result: Reranker extraction result dict
            chunked_doc: Optional chunked document for retrieving fuller evidence context

        Returns:
            Tuple of (field_results dict, list of field paths to escalate)
        """
        field_results = {}
        fields_to_escalate = []

        # Check if adversarial validation is enabled
        adv_config = self.config.adversarial_validation
        run_adversarial = (
            adv_config
            and adv_config.enabled
            and self.adversarial_validator
        )

        # Get share_classes from both results
        t3_classes = get_nested_value(t3_result, "share_classes.share_classes") or []
        rr_classes = get_nested_value(rr_result, "share_classes.share_classes") or []

        # Build class name to data mapping
        t3_by_name = {c.get("class_name"): c for c in t3_classes if isinstance(c, dict)}
        rr_by_name = {c.get("class_name"): c for c in rr_classes if isinstance(c, dict)}

        # Get all class names from both
        all_class_names = set(t3_by_name.keys()) | set(rr_by_name.keys())

        for class_name in all_class_names:
            if not class_name:
                continue

            t3_class = t3_by_name.get(class_name, {})
            rr_class = rr_by_name.get(class_name, {})

            # Compare each share class field
            for sc_field in SHARE_CLASS_FIELDS:
                field_path = f"share_classes.{class_name}.{sc_field}"
                t3_value = t3_class.get(sc_field)
                rr_value = rr_class.get(sc_field)

                field_result, should_escalate = self._decide_field(
                    field_path, t3_value, rr_value, t3_result, rr_result
                )

                # LLM-as-a-judge validation for accepted share class fields
                should_validate = (
                    run_adversarial
                    and not should_escalate
                    and field_result.values_agree
                    and not field_result.both_null
                    and (adv_config.validate_all_accepted or adv_config.validate_all)
                )

                if should_validate:
                    # Get evidence from share_classes section, with fuller context from chunks
                    evidence = self._get_field_evidence(
                        t3_result, rr_result, "share_classes", chunked_doc=chunked_doc
                    )

                    if evidence:
                        # Determine expected type
                        if "pct" in sc_field:
                            expected_type = "percentage"
                        elif "investment" in sc_field:
                            expected_type = "currency"
                        else:
                            expected_type = "text"

                        logger.info(
                            f"[Ensemble] Running LLM-as-a-judge validation for '{field_path}' "
                            f"(value={field_result.final_value}, type={expected_type})"
                        )
                        adv_result = self.adversarial_validator.validate(
                            field_name=f"{class_name} {sc_field}",
                            value=field_result.final_value,
                            evidence=evidence,
                            expected_type=expected_type,
                        )

                        field_result.adversarial_validated = adv_result.is_valid
                        field_result.adversarial_problems = adv_result.problems
                        field_result.adversarial_confidence = adv_result.confidence

                        if not adv_result.is_valid and adv_config.escalate_on_rejection:
                            logger.warning(
                                f"[Ensemble] LLM-as-a-judge REJECTED '{field_path}' → escalating to T4: "
                                f"{adv_result.problems}"
                            )
                            should_escalate = True
                            field_result.escalated_to_t4 = True
                            field_result.escalation_reason = "llm_judge_rejected"
                            field_result.final_value = None
                            field_result.source = ""
                        else:
                            logger.info(
                                f"[Ensemble] LLM-as-a-judge PASSED '{field_path}' "
                                f"(confidence={adv_result.confidence:.2f})"
                            )

                field_results[field_path] = field_result
                if should_escalate:
                    fields_to_escalate.append(field_path)

        return field_results, fields_to_escalate

    def _run_t4_escalation(
        self,
        chunked_doc: ChunkedDocument,
        field_results: dict[str, FieldResult],
        fields_to_escalate: list[str],
    ) -> int:
        """
        Run T4 extraction on escalated fields.

        Returns:
            Number of successfully extracted fields
        """
        from .tier4_agentic import (
            Tier4Agent,
            Tier4Config as AgentTier4Config,
            AgentModel,
            FIELD_SPECS,
            FieldSpec,
            PriorTierResult,
        )

        # Map model string to enum
        model_mapping = {
            "gpt-4o": AgentModel.GPT_4O,
            "gpt-4o-mini": AgentModel.GPT_4O_MINI,
            "claude-sonnet-4-20250514": AgentModel.CLAUDE_SONNET,
            "claude-opus-4-20250514": AgentModel.CLAUDE_OPUS,
            "claude-haiku-4-5-20251001": AgentModel.CLAUDE_HAIKU,
        }
        model = model_mapping.get(self.config.t4_model, AgentModel.GPT_4O)

        # Create T4 config
        agent_config = AgentTier4Config(
            model=model,
            max_iterations=self.config.t4_max_iterations,
            timeout_seconds=self.config.t4_timeout_seconds,
            confidence_threshold=self.config.t4_confidence_threshold,
        )

        # Create agent (reuse for all fields)
        agent = Tier4Agent(config=agent_config, document=chunked_doc)

        success_count = 0

        # Track universal fee field values extracted from the first class.
        # key: field_name (e.g. "management_fee_pct"), value: (final_value, confidence, evidence)
        universal_fee_cache: dict[str, tuple] = {}
        propagated_count = 0

        for field_path in fields_to_escalate:
            # Handle share_classes fields specially
            # Field path format: share_classes.{class_name}.{field_name}
            if field_path.startswith("share_classes."):
                parts = field_path.split(".")
                if len(parts) >= 3:
                    class_name = parts[1]
                    sc_field_name = parts[2]

                    # Check if this universal fee field was already extracted for another class
                    if sc_field_name in UNIVERSAL_FEE_FIELDS and sc_field_name in universal_fee_cache:
                        cached_value, cached_conf, cached_evidence = universal_fee_cache[sc_field_name]
                        field_result = field_results[field_path]
                        if cached_value is not None:
                            field_result.final_value = cached_value
                            field_result.source = "t4_propagated"
                            field_result.t4_success = True
                            field_result.t4_confidence = cached_conf
                            field_result.t4_evidence = cached_evidence
                            success_count += 1
                            propagated_count += 1
                        else:
                            # First class extraction returned null — propagate null
                            field_result.final_value = None
                            field_result.source = "t4_propagated"
                            field_result.t4_success = False
                        logger.info(
                            f"[Ensemble] Propagated universal fee: {field_path} = {cached_value} "
                            f"(from prior class)"
                        )
                        continue

                    t4_field_name = FIELD_PATH_TO_T4_SPEC.get(sc_field_name)

                    if not t4_field_name or t4_field_name not in FIELD_SPECS:
                        logger.warning(f"[Ensemble] No T4 spec for share class field {sc_field_name}, skipping")
                        self._apply_fallback(field_results[field_path])
                        continue

                    # Get base spec and customize for this class
                    base_spec = FIELD_SPECS[t4_field_name]
                    field_spec = FieldSpec(
                        name=f"{base_spec.name}_{class_name}",
                        description=f"{base_spec.description} for {class_name}",
                        expected_type=base_spec.expected_type,
                        examples=base_spec.examples.copy() if base_spec.examples else [],
                        aliases=base_spec.aliases + [class_name.lower()] if base_spec.aliases else [class_name.lower()],
                        extraction_hints=[f"Find the {base_spec.name} specifically for {class_name} shares."] + (base_spec.extraction_hints or []),
                    )
                else:
                    logger.warning(f"[Ensemble] Invalid share_classes field path: {field_path}")
                    continue
            else:
                # Standard field path lookup
                t4_field_name = FIELD_PATH_TO_T4_SPEC.get(field_path)
                if not t4_field_name or t4_field_name not in FIELD_SPECS:
                    logger.warning(f"[Ensemble] No T4 spec for {field_path}, using fallback")
                    self._apply_fallback(field_results[field_path])
                    continue

                field_spec = FIELD_SPECS[t4_field_name]

            field_result = field_results[field_path]

            # Create prior tier summary with T3 evidence
            prior_results = [
                PriorTierResult(
                    tier=3,
                    status="not_found" if field_result.both_null else "disagreement",
                    extraction_result=field_result.t3_value,
                    failure_reason=field_result.escalation_reason,
                ),
            ]

            logger.info(f"[Ensemble] T4 extracting: {field_path}")

            try:
                # Run T4 extraction
                t4_result = agent.extract(field_spec=field_spec, prior_results=prior_results)

                if t4_result.success and t4_result.value is not None:
                    field_result.final_value = t4_result.value
                    field_result.source = "t4"
                    field_result.t4_success = True
                    field_result.t4_confidence = t4_result.confidence
                    field_result.t4_evidence = t4_result.evidence[:200] if t4_result.evidence else ""
                    success_count += 1
                    logger.info(f"[Ensemble] T4 found: {field_path} = {t4_result.value}")

                    # Cache universal fee field for propagation to other classes
                    if field_path.startswith("share_classes.") and sc_field_name in UNIVERSAL_FEE_FIELDS:
                        universal_fee_cache[sc_field_name] = (
                            t4_result.value,
                            t4_result.confidence,
                            t4_result.evidence[:200] if t4_result.evidence else "",
                        )
                else:
                    # T4 failed, fall back to T3 or Reranker value
                    self._apply_fallback(field_result)
                    logger.info(f"[Ensemble] T4 not found: {field_path}")

                    # Cache null result too so we don't re-extract on other classes
                    if field_path.startswith("share_classes.") and sc_field_name in UNIVERSAL_FEE_FIELDS:
                        universal_fee_cache[sc_field_name] = (None, None, "")

            except Exception as e:
                logger.error(f"[Ensemble] T4 error for {field_path}: {e}")
                self._apply_fallback(field_result)

        if propagated_count > 0:
            logger.info(
                f"[Ensemble] Universal fee propagation: {propagated_count} fields "
                f"skipped T4 via cross-class propagation"
            )

        return success_count

    def _apply_fallback(self, field_result: FieldResult):
        """Apply T3/Reranker fallback when T4 fails or is unavailable.

        Prefers Reranker over T3 because:
        - Reranker uses semantic search (Cohere rerank) for better context ranking
        - T3 uses keyword-based scoring which can miss semantic matches
        - When T4 escalation fails, Reranker likely found more relevant evidence
        """
        # Prefer Reranker - it uses semantic search with better context ranking
        if field_result.reranker_value is not None:
            field_result.final_value = field_result.reranker_value
            field_result.source = "reranker_fallback"
        elif field_result.t3_value is not None:
            field_result.final_value = field_result.t3_value
            field_result.source = "t3_fallback"
        else:
            field_result.final_value = None
            field_result.source = "not_found"

    def _build_final_extraction(
        self,
        t3_result: dict,
        rr_result: dict,
        field_results: dict[str, FieldResult],
        xbrl_values: dict,
        fund_name: str,
        two_pass_result=None,
        discovered_share_classes: Optional[list[str]] = None,
    ) -> dict:
        """Build the final extraction dict with resolved values."""
        import copy
        # Deep copy to avoid modifying original
        final = copy.deepcopy(t3_result)

        # Override with resolved field values
        for field_path, field_result in field_results.items():
            if field_result.final_value is not None:
                # Handle share_classes fields specially
                if field_path.startswith("share_classes.") and "." in field_path[14:]:
                    self._set_share_class_value(final, field_path, field_result.final_value)
                else:
                    self._set_nested_value(final, field_path, field_result.final_value)

        # Override share_classes with two-pass result if available
        if two_pass_result and two_pass_result.share_classes:
            two_pass_dict = convert_two_pass_to_dict(two_pass_result)
            final["share_classes"] = two_pass_dict
            logger.info(
                f"[Ensemble] Using two-pass share classes: "
                f"{[sc.class_name for sc in two_pass_result.share_classes]}"
            )

        # Validate share classes against discovery (if discovery was run)
        if discovered_share_classes and "share_classes" in final:
            final = self._validate_share_classes_against_discovery(
                final, discovered_share_classes
            )

        # Add ensemble metadata
        final["ensemble_metadata"] = {
            "extraction_mode": "ensemble_t4",
            "accepted_fields": [
                fp for fp, fr in field_results.items() if not fr.escalated_to_t4
            ],
            "escalated_fields": [
                fp for fp, fr in field_results.items() if fr.escalated_to_t4
            ],
            "t4_successes": [
                fp for fp, fr in field_results.items() if fr.t4_success
            ],
            "two_pass_share_classes_enabled": two_pass_result is not None,
            "two_pass_classes_discovered": (
                [sc.class_name for sc in two_pass_result.share_classes]
                if two_pass_result else []
            ),
            "discovery_first_enabled": discovered_share_classes is not None,
            "discovered_share_classes": discovered_share_classes or [],
        }

        return final

    def _validate_share_classes_against_discovery(
        self,
        extraction: dict,
        discovered_classes: list[str],
    ) -> dict:
        """
        Validate extracted share classes against discovery results.

        - Removes share classes not found in discovery (hallucinations)
        - Logs warnings for any filtered classes
        - Ensures all discovered classes have entries (adds empty if missing)

        Args:
            extraction: The extraction dict with share_classes
            discovered_classes: List of class names from discovery

        Returns:
            Updated extraction dict with validated share classes
        """
        share_classes_data = extraction.get("share_classes", {})
        extracted_classes = share_classes_data.get("share_classes", [])

        if not isinstance(extracted_classes, list):
            return extraction

        # Normalize discovered class names for comparison
        discovered_normalized = {c.lower().strip() for c in discovered_classes}

        # Filter extracted classes to only those in discovered list
        validated_classes = []
        filtered_out = []

        for sc in extracted_classes:
            class_name = sc.get("class_name", "")
            class_name_normalized = class_name.lower().strip()

            # Check if this class was discovered
            if class_name_normalized in discovered_normalized:
                validated_classes.append(sc)
            else:
                # Check for partial matches (e.g., "Class I" vs "Class I Shares")
                matched = False
                for discovered in discovered_classes:
                    if (class_name_normalized in discovered.lower() or
                        discovered.lower() in class_name_normalized):
                        validated_classes.append(sc)
                        matched = True
                        break

                if not matched:
                    filtered_out.append(class_name)

        # Log filtered classes
        if filtered_out:
            logger.warning(
                f"[Ensemble] Filtered {len(filtered_out)} share classes not in discovery: "
                f"{filtered_out}"
            )

        # Check for discovered classes missing from extraction
        extracted_names = {sc.get("class_name", "").lower().strip() for sc in validated_classes}
        for discovered in discovered_classes:
            discovered_norm = discovered.lower().strip()
            # Check if this discovered class is in extraction (with some flexibility)
            found = any(
                discovered_norm in name or name in discovered_norm
                for name in extracted_names
            )
            if not found:
                logger.warning(
                    f"[Ensemble] Discovered class '{discovered}' not found in extraction"
                )

        # Update extraction with validated classes
        extraction["share_classes"]["share_classes"] = validated_classes
        extraction["share_classes"]["_discovery_validation"] = {
            "discovered": discovered_classes,
            "extracted": [sc.get("class_name") for sc in extracted_classes],
            "validated": [sc.get("class_name") for sc in validated_classes],
            "filtered_out": filtered_out,
        }

        return extraction

    def _set_share_class_value(self, obj: dict, path: str, value: Any):
        """
        Set a share_classes field value.

        Path format: share_classes.{class_name}.{field_name}
        Target structure: obj["share_classes"]["share_classes"][i][field_name]
        where i is the index of the class with matching class_name.
        """
        parts = path.split(".")
        if len(parts) < 3:
            return

        class_name = parts[1]
        field_name = parts[2]

        # Navigate to share_classes.share_classes array
        if "share_classes" not in obj:
            obj["share_classes"] = {"share_classes": []}
        if "share_classes" not in obj["share_classes"]:
            obj["share_classes"]["share_classes"] = []

        classes = obj["share_classes"]["share_classes"]

        # Find the class by name or create it
        target_class = None
        for cls in classes:
            if isinstance(cls, dict) and cls.get("class_name") == class_name:
                target_class = cls
                break

        if target_class is None:
            # Create new class entry
            target_class = {"class_name": class_name}
            classes.append(target_class)

        # Set the field value
        target_class[field_name] = value

    def _set_nested_value(self, obj: dict, path: str, value: Any):
        """Set a nested value in a dict using dot notation."""
        parts = path.split(".")
        current = obj
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        final_key = parts[-1]
        if isinstance(current.get(final_key), dict) and "value" in current[final_key]:
            current[final_key]["value"] = value
        else:
            current[final_key] = value

    def _build_trace(
        self,
        result: EnsembleExtractionResult,
        field_results: dict[str, FieldResult],
        discovered_share_classes: Optional[list[str]] = None,
        discovery_result: Optional[DiscoveryResult] = None,
        discovery_duration: float = 0.0,
    ) -> EnsembleTrace:
        """Build observability trace with LLM-as-a-judge validation statistics."""
        adv_config = self.config.adversarial_validation

        trace = EnsembleTrace(
            extraction_mode="ensemble_t4",
            config={
                "t3_top_k": self.config.t3_top_k_sections,
                "reranker_first_pass_n": self.config.reranker_first_pass_n,
                "reranker_top_k": self.config.reranker_top_k,
                "t4_model": self.config.t4_model,
                "t4_max_iterations": self.config.t4_max_iterations,
                "escalate_on_disagreement": self.config.escalate_on_disagreement,
                "escalate_on_both_null": self.config.escalate_on_both_null,
                "share_class_discovery_first_enabled": self.config.share_class_discovery_first_enabled,
                # LLM-as-a-judge validation config
                "llm_judge_enabled": adv_config.enabled if adv_config else False,
                "llm_judge_validate_all": adv_config.validate_all_accepted if adv_config else False,
                "llm_judge_model": adv_config.model if adv_config and not adv_config.lightweight else (adv_config.lightweight_model if adv_config else None),
                "llm_judge_lightweight": adv_config.lightweight if adv_config else True,
            },
            discovery_duration_seconds=discovery_duration,
            t3_duration_seconds=result.t3_duration_seconds,
            reranker_duration_seconds=result.reranker_duration_seconds,
            t4_duration_seconds=result.t4_duration_seconds,
            discovered_share_classes=discovered_share_classes or [],
            discovery_chunks_used=discovery_result.chunks_used if discovery_result else 0,
            discovery_reasoning=discovery_result.reasoning if discovery_result else None,
        )

        # Calculate rates
        if result.total_fields > 0:
            trace.agreement_rate = result.accepted_count / result.total_fields
            trace.escalation_rate = result.escalated_count / result.total_fields
            if result.escalated_count > 0:
                trace.t4_success_rate = result.t4_success_count / result.escalated_count

        # Count LLM-as-a-judge validation results
        validated_count = 0
        validation_passed_count = 0
        validation_rejected_count = 0

        # Field-level decisions
        for field_path, fr in field_results.items():
            trace.field_decisions[field_path] = {
                "t3_value": fr.t3_value,
                "reranker_value": fr.reranker_value,
                "final_value": fr.final_value,
                "values_agree": fr.values_agree,
                "both_null": fr.both_null,
                "escalated": fr.escalated_to_t4,
                "escalation_reason": fr.escalation_reason,
                "source": fr.source,
                "t4_success": fr.t4_success,
                "t4_confidence": fr.t4_confidence,
                # LLM-as-a-judge validation info
                "llm_judge_validated": fr.adversarial_validated,
                "llm_judge_confidence": fr.adversarial_confidence,
                "llm_judge_problems": fr.adversarial_problems,
            }

            # Track validation statistics
            if fr.adversarial_validated is not None:
                validated_count += 1
                if fr.adversarial_validated:
                    validation_passed_count += 1
                else:
                    validation_rejected_count += 1

        # Add validation statistics to config
        trace.config["llm_judge_validated_count"] = validated_count
        trace.config["llm_judge_passed_count"] = validation_passed_count
        trace.config["llm_judge_rejected_count"] = validation_rejected_count
        if validated_count > 0:
            trace.config["llm_judge_pass_rate"] = validation_passed_count / validated_count

        return trace

    def _convert_to_field_results(
        self,
        extraction_result: dict,
    ) -> dict[str, FieldResult]:
        """
        Convert raw extraction dict to FieldResult format for single-method mode.

        In single-method mode (e.g., multi-query only), we don't have a comparison
        between T3 and Reranker. This method creates FieldResult objects directly
        from the extraction results.

        Args:
            extraction_result: Raw extraction dict from multi-query or T3

        Returns:
            Dict mapping field paths to FieldResult objects
        """
        field_results = {}

        # Process all non-share-class fields
        for section_name, section_data in extraction_result.items():
            if section_name in ("xbrl_fees", "fund_type", "fund_type_flags"):
                continue

            if section_name == "share_classes":
                # Handle share classes separately
                share_classes_list = section_data.get("share_classes", [])
                if isinstance(share_classes_list, list):
                    for sc in share_classes_list:
                        if not isinstance(sc, dict):
                            continue
                        class_name = sc.get("class_name", "")
                        if not class_name:
                            continue
                        for field_name, value in sc.items():
                            if field_name != "class_name":
                                field_path = f"share_classes.share_classes.{class_name}.{field_name}"
                                field_results[field_path] = FieldResult(
                                    field_name=field_path,
                                    t3_value=value,
                                    reranker_value=None,
                                    final_value=value,
                                    values_agree=True,
                                    both_null=value is None,
                                    escalated_to_t4=False,
                                    escalation_reason="none",
                                    source="multi_query",
                                )
            elif isinstance(section_data, dict):
                # Handle nested section (e.g., incentive_fee.has_incentive_fee)
                for field_name, value in section_data.items():
                    if field_name == "evidence":
                        continue
                    field_path = f"{section_name}.{field_name}"
                    field_results[field_path] = FieldResult(
                        field_name=field_path,
                        t3_value=value,
                        reranker_value=None,
                        final_value=value,
                        values_agree=True,
                        both_null=value is None,
                        escalated_to_t4=False,
                        escalation_reason="none",
                        source="multi_query",
                    )
            else:
                # Simple field value
                field_results[section_name] = FieldResult(
                    field_name=section_name,
                    t3_value=section_data,
                    reranker_value=None,
                    final_value=section_data,
                    values_agree=True,
                    both_null=section_data is None,
                    escalated_to_t4=False,
                    escalation_reason="none",
                    source="multi_query",
                )

        return field_results


def convert_ensemble_to_extraction_format(
    result: EnsembleExtractionResult,
) -> dict:
    """Convert ensemble result to standard extraction format."""
    return result.extraction
