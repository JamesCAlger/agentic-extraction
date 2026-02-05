"""
Fund Type Classification Module.

Classifies funds by their investment strategy AND structure to enable schema routing
and prevent hallucinations for fields that don't apply to certain fund types.

Fund Strategies:
- fund_of_funds: Invests in other PE/VC funds (StepStone, Hamilton Lane, Carlyle)
- direct_credit: Makes direct loans to companies (Blue Owl)
- direct_equity: Makes direct equity investments
- hybrid: Mix of direct and fund investments (Blackstone)

Fund Structures:
- interval_fund: Registered under 1940 Act; mandatory quarterly repurchases (5-25% of NAV)
- tender_offer_fund: Registered; discretionary repurchases at board's option
- unknown: Unable to determine structure
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .llm_provider import (
    create_raw_client,
    call_llm_json,
    RateLimitConfig,
    detect_provider,
)

logger = logging.getLogger(__name__)


class FundStrategy(str, Enum):
    """Fund investment strategy categories."""
    FUND_OF_FUNDS = "fund_of_funds"
    DIRECT_CREDIT = "direct_credit"
    DIRECT_EQUITY = "direct_equity"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class FundStructure(str, Enum):
    """Fund legal/structural categories (affects repurchase behavior)."""
    INTERVAL_FUND = "interval_fund"
    TENDER_OFFER_FUND = "tender_offer_fund"
    UNKNOWN = "unknown"


@dataclass
class FundClassification:
    """Result of fund classification."""
    strategy: FundStrategy
    confidence: float  # 0.0 to 1.0
    evidence: str
    reasoning: str
    classification_method: str  # "heuristic" or "llm"
    # NEW: Fund structure classification
    structure: FundStructure = FundStructure.UNKNOWN
    structure_confidence: float = 0.0
    structure_evidence: str = ""

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "reasoning": self.reasoning,
            "classification_method": self.classification_method,
            "structure": self.structure.value,
            "structure_confidence": self.structure_confidence,
            "structure_evidence": self.structure_evidence,
        }

    @property
    def is_fund_of_funds(self) -> bool:
        """Returns True if fund invests in other funds (FOF, hybrid with FOF)."""
        return self.strategy in (FundStrategy.FUND_OF_FUNDS, FundStrategy.HYBRID)

    @property
    def is_direct_investment(self) -> bool:
        """Returns True if fund makes direct investments (credit, equity, hybrid)."""
        return self.strategy in (
            FundStrategy.DIRECT_CREDIT,
            FundStrategy.DIRECT_EQUITY,
            FundStrategy.HYBRID,
        )

    @property
    def is_interval_fund(self) -> bool:
        """Returns True if fund is an interval fund with mandatory repurchases."""
        return self.structure == FundStructure.INTERVAL_FUND

    @property
    def is_tender_offer_fund(self) -> bool:
        """Returns True if fund is a tender offer fund with discretionary repurchases."""
        return self.structure == FundStructure.TENDER_OFFER_FUND


# =============================================================================
# HEURISTIC CLASSIFICATION
# =============================================================================

# Keywords indicating fund-of-funds strategy
FOF_KEYWORDS = [
    # Primary indicators (high confidence)
    "fund-of-funds", "fund of funds",
    "invests in other private", "invests in private equity funds",
    "invests in underlying funds", "portfolio funds",
    "underlying private funds", "underlying managers",
    "secondary investments", "secondaries fund",
    "secondary private equity", "secondary market",
    "co-investments alongside", "co-invest alongside",
    "commitments to other funds",
    # Secondary indicators (medium confidence)
    "private markets fund", "private assets fund",
    "diversified private equity", "multi-manager",
    "allocations to", "target allocation",
]

# Keywords indicating direct credit strategy
DIRECT_CREDIT_KEYWORDS = [
    # Primary indicators (high confidence)
    "direct lending", "directly originates loans",
    "senior secured loans", "first lien loans",
    "private credit", "corporate lending",
    "middle market lending", "middle-market loans",
    "originate loans", "loan origination",
    "direct loan", "directly lend",
    # Secondary indicators (medium confidence)
    "credit fund", "income fund",
    "floating rate loans", "leveraged loans",
    "debt investments", "debt securities",
]

# Keywords indicating direct equity strategy
DIRECT_EQUITY_KEYWORDS = [
    # Primary indicators (high confidence)
    "direct equity investments", "direct investments in companies",
    "co-invest directly", "invests directly in",
    "operating companies", "portfolio companies",
    # Secondary indicators (medium confidence)
    "buyout fund", "growth equity",
]

# Keywords indicating hybrid strategy
HYBRID_KEYWORDS = [
    # Primary indicators
    "combination of direct", "mix of direct and fund",
    "both direct investments and", "alongside fund investments",
    "opportunistic allocation",
    # Blackstone-specific patterns
    "multi-asset", "multi asset", "credit and income",
]

# =============================================================================
# FUND STRUCTURE KEYWORDS (Interval vs Tender Offer)
# =============================================================================

# Keywords indicating interval fund structure
INTERVAL_FUND_KEYWORDS = [
    # Primary indicators (high confidence)
    "interval fund",
    "Rule 23c-3",
    "periodic repurchase offers",
    "mandatory repurchase",
    "repurchase offers will be made quarterly",
    "quarterly repurchase offer",
    "between 5% and 25% of",
    "at least 5% but no more than 25%",
    # Secondary indicators
    "repurchase offer amount",
    "repurchase pricing date",
    "repurchase request deadline",
    "interval fund procedures",
]

# Keywords indicating tender offer fund structure
TENDER_OFFER_FUND_KEYWORDS = [
    # Primary indicators (high confidence)
    "tender offer fund",
    "discretionary tender offer",
    "at the discretion of the Board",
    "Board may determine",
    "may elect to offer",
    "no assurance that any tender offer",
    # Secondary indicators
    "discretionary repurchase",
    "Board discretion",
    "may conduct tender offers",
    "not obligated to conduct",
]


def classify_structure_by_heuristics(text: str, fund_name: str = "") -> tuple[FundStructure, float, str]:
    """
    Classify fund structure (interval vs tender offer) using keyword heuristics.

    Args:
        text: Document text (cover section, summary, or full text)
        fund_name: Fund name for additional context

    Returns:
        Tuple of (structure, confidence, evidence_string)
    """
    text_lower = text.lower()
    name_lower = fund_name.lower()

    interval_evidence = []
    tender_evidence = []

    # Check interval fund keywords
    for keyword in INTERVAL_FUND_KEYWORDS:
        if keyword.lower() in text_lower or keyword.lower() in name_lower:
            interval_evidence.append(keyword)

    # Check tender offer fund keywords
    for keyword in TENDER_OFFER_FUND_KEYWORDS:
        if keyword.lower() in text_lower or keyword.lower() in name_lower:
            tender_evidence.append(keyword)

    interval_score = len(interval_evidence)
    tender_score = len(tender_evidence)

    if interval_score == 0 and tender_score == 0:
        return FundStructure.UNKNOWN, 0.0, ""

    if interval_score > tender_score:
        confidence = min(0.9, 0.4 + (interval_score * 0.15))
        return FundStructure.INTERVAL_FUND, confidence, ", ".join(interval_evidence[:3])
    elif tender_score > interval_score:
        confidence = min(0.9, 0.4 + (tender_score * 0.15))
        return FundStructure.TENDER_OFFER_FUND, confidence, ", ".join(tender_evidence[:3])
    else:
        # Tie: check for explicit "interval fund" phrase
        if "interval fund" in text_lower:
            return FundStructure.INTERVAL_FUND, 0.7, "interval fund"
        elif "tender offer" in text_lower:
            return FundStructure.TENDER_OFFER_FUND, 0.7, "tender offer"
        return FundStructure.UNKNOWN, 0.3, ""


def classify_by_heuristics(text: str, fund_name: str = "") -> FundClassification:
    """
    Classify fund strategy using keyword heuristics.

    Args:
        text: Document text (cover section, summary, or full text)
        fund_name: Fund name for additional context

    Returns:
        FundClassification with heuristic-based result
    """
    text_lower = text.lower()
    name_lower = fund_name.lower()

    # Track evidence for each strategy
    fof_evidence = []
    credit_evidence = []
    equity_evidence = []
    hybrid_evidence = []

    # Check fund-of-funds keywords
    for keyword in FOF_KEYWORDS:
        if keyword in text_lower or keyword in name_lower:
            fof_evidence.append(keyword)

    # Check direct credit keywords
    for keyword in DIRECT_CREDIT_KEYWORDS:
        if keyword in text_lower or keyword in name_lower:
            credit_evidence.append(keyword)

    # Check direct equity keywords
    for keyword in DIRECT_EQUITY_KEYWORDS:
        if keyword in text_lower or keyword in name_lower:
            equity_evidence.append(keyword)

    # Check hybrid keywords
    for keyword in HYBRID_KEYWORDS:
        if keyword in text_lower or keyword in name_lower:
            hybrid_evidence.append(keyword)

    # Determine strategy based on evidence
    scores = {
        FundStrategy.FUND_OF_FUNDS: len(fof_evidence),
        FundStrategy.DIRECT_CREDIT: len(credit_evidence),
        FundStrategy.DIRECT_EQUITY: len(equity_evidence),
        FundStrategy.HYBRID: len(hybrid_evidence),
    }

    max_score = max(scores.values())
    if max_score == 0:
        return FundClassification(
            strategy=FundStrategy.UNKNOWN,
            confidence=0.0,
            evidence="No strategy keywords found",
            reasoning="Unable to classify fund strategy from available text",
            classification_method="heuristic",
        )

    # Get top strategy
    top_strategy = max(scores, key=scores.get)

    # Check for hybrid pattern (multiple high scores)
    high_scores = sum(1 for s in scores.values() if s >= max_score * 0.5)
    if high_scores > 1 and max_score > 1:
        top_strategy = FundStrategy.HYBRID

    # Build evidence string
    evidence_map = {
        FundStrategy.FUND_OF_FUNDS: fof_evidence,
        FundStrategy.DIRECT_CREDIT: credit_evidence,
        FundStrategy.DIRECT_EQUITY: equity_evidence,
        FundStrategy.HYBRID: hybrid_evidence,
    }
    evidence = evidence_map.get(top_strategy, [])[:5]  # Top 5 matches

    # Calculate confidence based on evidence strength
    confidence = min(0.9, 0.3 + (max_score * 0.15))

    # Also classify fund structure (interval vs tender offer)
    structure, structure_confidence, structure_evidence = classify_structure_by_heuristics(
        text, fund_name
    )

    return FundClassification(
        strategy=top_strategy,
        confidence=confidence,
        evidence=", ".join(evidence),
        reasoning=f"Found {max_score} keywords matching {top_strategy.value} strategy",
        classification_method="heuristic",
        structure=structure,
        structure_confidence=structure_confidence,
        structure_evidence=structure_evidence,
    )


# =============================================================================
# LLM-BASED CLASSIFICATION
# =============================================================================

CLASSIFICATION_PROMPT = """You are a financial analyst classifying investment fund strategies.

Analyze the following fund information and classify it into ONE of these categories:

1. **fund_of_funds**: Invests primarily in other private equity/venture capital funds
   - Key indicators: "invests in underlying funds", "portfolio funds", "secondaries", "co-investments alongside GP sponsors"
   - Does NOT directly lend to or invest in operating companies

2. **direct_credit**: Makes direct loans to companies
   - Key indicators: "direct lending", "originates loans", "senior secured loans", "private credit"
   - Lends directly to borrowers rather than investing in funds

3. **direct_equity**: Makes direct equity investments in companies
   - Key indicators: "direct equity investments", "portfolio companies", "buyout", "growth equity"
   - Buys ownership stakes in companies rather than investing in funds

4. **hybrid**: Combination of multiple strategies
   - Key indicators: "multi-asset", "opportunistic", "both direct investments and fund investments"
   - Mixes direct investments with fund investments

FUND NAME: {fund_name}

DOCUMENT TEXT:
{document_text}

Respond with JSON in this exact format:
{{
  "strategy": "<one of: fund_of_funds, direct_credit, direct_equity, hybrid>",
  "confidence": <0.0 to 1.0>,
  "evidence": "<exact quote from the text supporting your classification>",
  "reasoning": "<brief explanation of your classification>"
}}
"""


async def classify_by_llm(
    document_text: str,
    fund_name: str,
    model: str = "gpt-4o-mini",
    rate_limit_config: Optional[RateLimitConfig] = None,
) -> FundClassification:
    """
    Classify fund strategy using LLM.

    Args:
        document_text: Document text (cover section, summary, or first few chunks)
        fund_name: Fund name for context
        model: LLM model to use (default: gpt-4o-mini for cost efficiency)
        rate_limit_config: Optional rate limiting configuration

    Returns:
        FundClassification with LLM-based result
    """
    # Truncate text to reasonable size for classification
    max_chars = 8000
    if len(document_text) > max_chars:
        document_text = document_text[:max_chars] + "..."

    prompt = CLASSIFICATION_PROMPT.format(
        fund_name=fund_name,
        document_text=document_text,
    )

    try:
        # Get provider and create client
        provider = detect_provider(model)
        client = create_raw_client(provider, rate_limit_config)

        # Call LLM for classification
        result = call_llm_json(
            client=client,
            model=model,
            prompt=prompt,
            system_prompt="You are a financial analyst. Respond only with valid JSON.",
        )

        if result:
            strategy_str = result.get("strategy", "unknown").lower()
            try:
                strategy = FundStrategy(strategy_str)
            except ValueError:
                strategy = FundStrategy.UNKNOWN

            return FundClassification(
                strategy=strategy,
                confidence=float(result.get("confidence", 0.7)),
                evidence=str(result.get("evidence", ""))[:500],
                reasoning=str(result.get("reasoning", "")),
                classification_method="llm",
            )

    except Exception as e:
        logger.warning(f"LLM classification failed: {e}")

    # Fall back to heuristics on LLM failure
    return classify_by_heuristics(document_text, fund_name)


# =============================================================================
# MAIN CLASSIFICATION FUNCTION
# =============================================================================

def classify_fund(
    document_text: str,
    fund_name: str,
    use_llm: bool = False,
    model: str = "gpt-4o-mini",
) -> FundClassification:
    """
    Classify fund investment strategy.

    Args:
        document_text: Document text for classification
        fund_name: Fund name for context
        use_llm: Whether to use LLM for classification (default: False for speed)
        model: LLM model to use if use_llm=True

    Returns:
        FundClassification with strategy, confidence, and evidence
    """
    # Start with heuristic classification (fast, no API cost)
    heuristic_result = classify_by_heuristics(document_text, fund_name)

    # If heuristic has high confidence, use it
    if heuristic_result.confidence >= 0.7:
        logger.info(
            f"Fund '{fund_name}' classified as {heuristic_result.strategy.value} "
            f"(confidence: {heuristic_result.confidence:.2f}, method: heuristic)"
        )
        return heuristic_result

    # If use_llm is enabled and heuristic confidence is low, try LLM
    if use_llm:
        # Note: This is synchronous wrapper; in async context, use classify_by_llm directly
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        llm_result = loop.run_until_complete(
            classify_by_llm(document_text, fund_name, model)
        )

        if llm_result.confidence > heuristic_result.confidence:
            logger.info(
                f"Fund '{fund_name}' classified as {llm_result.strategy.value} "
                f"(confidence: {llm_result.confidence:.2f}, method: llm)"
            )
            return llm_result

    logger.info(
        f"Fund '{fund_name}' classified as {heuristic_result.strategy.value} "
        f"(confidence: {heuristic_result.confidence:.2f}, method: heuristic)"
    )
    return heuristic_result


# =============================================================================
# SCHEMA ROUTING BASED ON FUND TYPE
# =============================================================================

# Fields that should be null for direct credit funds (not fund-of-funds)
# Direct credit funds lend directly to companies - no underlying fund structure
DIRECT_CREDIT_NULL_FIELDS = [
    # Fund-of-funds specific allocation fields
    "allocation_targets.secondary_funds_min_pct",
    "allocation_targets.secondary_funds_max_pct",
    "allocation_targets.direct_investments_min_pct",
    "allocation_targets.direct_investments_max_pct",
    "allocation_targets.secondary_investments_min_pct",
    "allocation_targets.secondary_investments_max_pct",
    # Fund-of-funds concentration limits
    "concentration_limits.max_single_fund_pct",
    # Underlying fund fee disclosure (irrelevant for direct lending)
    "incentive_fee.underlying_fund_incentive_range",
]

# Fields that should be null for fund-of-funds (not direct credit)
FUND_OF_FUNDS_NULL_FIELDS = [
    # Fund-of-funds typically don't have these at the fund level
    # (they exist at underlying fund level, not at FOF level)
    # FOFs have fund-level incentive fee on the FOF itself, but may also
    # disclose underlying_fund_incentive_range which is NOT null
]

# Fields that should be null for direct equity funds
DIRECT_EQUITY_NULL_FIELDS = DIRECT_CREDIT_NULL_FIELDS.copy()  # Same as direct credit

# Fields that should be treated as nullable (may not exist in document)
# These fields are legitimately not disclosed by some funds
NULLABLE_BY_FUND_TYPE = {
    FundStrategy.DIRECT_CREDIT: [
        # Credit funds may not have formal allocation targets
        "allocation_targets.allocations",
        # May not have concentration limits beyond regulatory
        "concentration_limits.max_single_sector_pct",
    ],
    FundStrategy.FUND_OF_FUNDS: [
        # FOFs may not disclose direct lending metrics
        "leverage_limits.max_leverage_pct",  # Some FOFs don't use leverage
    ],
    FundStrategy.HYBRID: [
        # Hybrid funds may not have clear allocation targets
        "allocation_targets.allocations",
    ],
}

# Fields that MUST exist for each fund type (validation check)
REQUIRED_FIELDS_BY_TYPE = {
    FundStrategy.DIRECT_CREDIT: [
        "incentive_fee.has_incentive_fee",
        "repurchase_terms.repurchase_frequency",
        "distribution_terms.distribution_frequency",
    ],
    FundStrategy.FUND_OF_FUNDS: [
        "incentive_fee.has_incentive_fee",
        "repurchase_terms.repurchase_frequency",
        # FOFs should have allocation targets
        "allocation_targets.allocations",
    ],
    FundStrategy.HYBRID: [
        "incentive_fee.has_incentive_fee",
        "repurchase_terms.repurchase_frequency",
    ],
}


def get_null_fields_for_strategy(strategy: FundStrategy) -> list[str]:
    """
    Get list of fields that should be null for a given fund strategy.

    Args:
        strategy: The fund's investment strategy

    Returns:
        List of field paths that should be null for this strategy
    """
    if strategy == FundStrategy.DIRECT_CREDIT:
        return DIRECT_CREDIT_NULL_FIELDS
    elif strategy == FundStrategy.DIRECT_EQUITY:
        return DIRECT_EQUITY_NULL_FIELDS
    elif strategy == FundStrategy.FUND_OF_FUNDS:
        return FUND_OF_FUNDS_NULL_FIELDS
    else:
        # For hybrid or unknown, don't null any fields
        return []


def get_nullable_fields_for_strategy(strategy: FundStrategy) -> list[str]:
    """
    Get list of fields that are legitimately nullable for a fund strategy.

    These fields may not be disclosed and should not be penalized as "missed".

    Args:
        strategy: The fund's investment strategy

    Returns:
        List of field paths that are nullable for this strategy
    """
    return NULLABLE_BY_FUND_TYPE.get(strategy, [])


def get_required_fields_for_strategy(strategy: FundStrategy) -> list[str]:
    """
    Get list of fields that should exist for a fund strategy.

    Used for validation - if these are missing, it's a retrieval failure.

    Args:
        strategy: The fund's investment strategy

    Returns:
        List of field paths that should exist for this strategy
    """
    return REQUIRED_FIELDS_BY_TYPE.get(strategy, [])
