"""
Per-Datapoint Tier3 Extraction

This module implements granular per-datapoint extraction where each data point
gets its own keyword configuration and focused LLM call.

Benefits over field-level extraction:
1. More specific keywords for each value
2. Simpler LLM prompts (one question per call)
3. Better negative keyword filtering per datapoint
4. Easier to debug which datapoints fail
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Any
import time

from .llm_provider import (
    create_raw_client,
    call_llm_json,
    RateLimitConfig,
    detect_provider,
    resolve_model_name,
)
from .scoped_agentic import (
    score_section_for_field,
    select_top_sections,
    find_relevant_chunks,
    ScoredSection,
)
from ..parse.models import ChunkedDocument, ChunkedSection, Chunk, SectionType

logger = logging.getLogger(__name__)


# =============================================================================
# PER-DATAPOINT KEYWORD CONFIGURATION
# =============================================================================

DATAPOINT_KEYWORDS = {
    # ===========================================================================
    # INCENTIVE FEE DATAPOINTS
    # ===========================================================================
    "incentive_fee.has_incentive_fee": {
        "high_value": [
            "incentive fee",
            "performance fee",
            "performance-based fee",
            "incentive allocation",
            "carried interest",
            "performance allocation",
        ],
        "medium_value": [
            "adviser compensation",
            "advisor compensation",
            "performance-based",
        ],
        "low_value": [],
        "negative_high": [
            "no incentive fee",
            "does not charge an incentive fee",
            "underlying fund fees",
            "AFFE",
            "acquired fund fees",
        ],
        "negative_medium": [
            "underlying funds",
            "Investment Funds charge",
            "Investment Funds typically",
        ],
    },

    "incentive_fee.incentive_fee_pct": {
        "high_value": [
            "10% of",
            "12.5% of",
            "15% of",
            "20% of",
            "10 percent",
            "incentive fee equal to",
            "performance fee of",
            "incentive fee of",
        ],
        "medium_value": [
            "% of net profits",
            "% of the excess",
            "percent of",
            "incentive fee",
        ],
        "low_value": [],
        "negative_high": [
            "underlying funds typically charge 10% to 20%",
            "Investment Funds typically",
            "AFFE",
        ],
        "negative_medium": [
            "expense ratio",
            "management fee",
        ],
    },

    "incentive_fee.hurdle_rate_pct": {
        "high_value": [
            "hurdle rate",
            "hurdle rate of",
            "5% annualized",
            "6% annualized",
            "8% annualized",
            "5.0% annualized",
            "1.25% quarterly",
            "1.5% quarterly",
            "2% quarterly",
            "preferred return",
            "annualized hurdle rate",
        ],
        "medium_value": [
            "annualized hurdle",
            "before incentive",
            "exceeds",
            "threshold",
            "benchmark rate",
            "hurdle",
        ],
        "low_value": [
            "SOFR",
            "treasury",
            "T-bill",
        ],
        "negative_high": [
            "underlying funds",
            "Investment Funds may",
        ],
        "negative_medium": [],
    },

    "incentive_fee.hurdle_rate_as_stated": {
        "high_value": [
            "quarterly rate of",
            "1.25%",
            "1.5%",
            "2% per quarter",
            "SOFR plus",
            "T-bill plus",
            "quarterly hurdle rate",
        ],
        "medium_value": [
            "quarterly hurdle",
            "per quarter",
        ],
        "low_value": [],
        "negative_high": [],
        "negative_medium": [],
    },

    "incentive_fee.hurdle_rate_frequency": {
        "high_value": [
            "quarterly hurdle",
            "annual hurdle",
            "annualized rate",
            "per quarter",
            "quarterly rate",
        ],
        "medium_value": [
            "quarterly",
            "annual",
            "annualized",
        ],
        "low_value": [],
        "negative_high": [],
        "negative_medium": [],
    },

    "incentive_fee.high_water_mark": {
        "high_value": [
            "high water mark",
            "high-water mark",
            "highwater mark",
            "loss recovery account",
            "loss carryforward",
            "deficit recovery",
            "cumulative loss recovery",
            "recover prior losses",
            "losses must be recovered",
            "loss recovery",
        ],
        "medium_value": [
            "prior period losses",
            "deficit carryforward",
            "crystallization",
            "prevent double charging",
        ],
        "low_value": [],
        "negative_high": [
            "no high water mark",
            "without a high water mark",
        ],
        "negative_medium": [],
    },

    "incentive_fee.has_catch_up": {
        "high_value": [
            "catch-up",
            "catch up",
            "catchup",
            "full catch-up",
            "100% catch-up",
            "catch-up provision",
            "with a catch-up",
        ],
        "medium_value": [
            "until the adviser receives",
            "until the advisor receives",
        ],
        "low_value": [],
        "negative_high": [
            "no catch-up",
            "without catch-up",
        ],
        "negative_medium": [],
    },

    "incentive_fee.catch_up_rate_pct": {
        "high_value": [
            "100% catch-up",
            "catch-up of 100%",
            "full catch-up",
            "catch-up rate",
        ],
        "medium_value": [
            "catch-up",
            "catchup",
        ],
        "low_value": [],
        "negative_high": [],
        "negative_medium": [],
    },

    "incentive_fee.fee_basis": {
        "high_value": [
            "net investment income",
            "pre-incentive fee net investment income",
            "net profits",
            "net appreciation",
            "capital gains",
            "realized and unrealized",
            "net profits of the fund",
        ],
        "medium_value": [
            "income",
            "profits",
            "appreciation",
        ],
        "low_value": [],
        "negative_high": [],
        "negative_medium": [],
    },

    "incentive_fee.crystallization_frequency": {
        "high_value": [
            "quarterly in arrears",
            "calculated quarterly",
            "annually in arrears",
            "crystallize quarterly",
            "crystallize annually",
            "payable quarterly",
        ],
        "medium_value": [
            "each quarter",
            "quarterly basis",
            "annual basis",
        ],
        "low_value": [],
        "negative_high": [],
        "negative_medium": [],
    },

    # ===========================================================================
    # EXPENSE CAP DATAPOINTS
    # ===========================================================================
    "expense_cap.has_expense_cap": {
        "high_value": [
            "expense limitation",
            "expense cap",
            "fee waiver",
            "expense reimbursement",
            "contractual cap",
            "voluntary cap",
            "expense limitation agreement",
            "agreed to waive",
        ],
        "medium_value": [
            "waive",
            "reimburse",
            "limit expenses",
        ],
        "low_value": [],
        "negative_high": [
            "no expense limitation",
            "no expense cap",
        ],
        "negative_medium": [
            "total annual expenses",
        ],
    },

    "expense_cap.expense_cap_pct": {
        "high_value": [
            "capped at",
            "not to exceed",
            "limited to",
            "1.50%",
            "1.75%",
            "2.00%",
            "2.25%",
            "expense limitation of",
            "cap of",
        ],
        "medium_value": [
            "% of net assets",
            "% of average",
            "percentage of",
        ],
        "low_value": [],
        "negative_high": [
            "total expense ratio was",
        ],
        "negative_medium": [],
    },

    # ===========================================================================
    # REPURCHASE TERMS DATAPOINTS
    # ===========================================================================
    "repurchase_terms.repurchase_frequency": {
        "high_value": [
            "quarterly repurchase",
            "quarterly tender offer",
            "quarterly repurchase offer",
            "quarterly basis",
            "each calendar quarter",
            "four times per year",
            "repurchase offers quarterly",
        ],
        "medium_value": [
            "repurchase offer",
            "tender offer",
            "quarterly",
            "semi-annual",
            "annual",
        ],
        "low_value": [],
        "negative_high": [],
        "negative_medium": [
            "may elect not to",
            "no obligation",
        ],
    },

    "repurchase_terms.repurchase_amount_pct": {
        "high_value": [
            "5% of outstanding",
            "5% to 25%",
            "at least 5%",
            "up to 25%",
            "between 5% and 25%",
            "repurchase 5%",
            "tender 5%",
            "offer to repurchase",
        ],
        "medium_value": [
            "% of shares",
            "% of net assets",
            "percentage of",
        ],
        "low_value": [],
        "negative_high": [],
        "negative_medium": [],
    },

    "repurchase_terms.repurchase_basis": {
        "high_value": [
            "of outstanding shares",
            "number of shares outstanding",
            "of net assets",
            "percentage of shares",
            "percentage of NAV",
            "of the Fund's outstanding",
        ],
        "medium_value": [
            "shares outstanding",
            "net assets",
            "NAV",
        ],
        "low_value": [],
        "negative_high": [],
        "negative_medium": [],
    },

    "repurchase_terms.lock_up_period_years": {
        "high_value": [
            "lock-up period",
            "one year lock",
            "12 month lock",
            "one-year holding",
            "holding period of one year",
            "must hold for",
            "shares held less than one year",
            "within one year",
        ],
        "medium_value": [
            "lock-up",
            "lock up",
            "lockup",
            "holding period",
        ],
        "low_value": [],
        "negative_high": [
            "no lock-up",
            "no holding period",
        ],
        "negative_medium": [],
    },

    "repurchase_terms.early_repurchase_fee_pct": {
        "high_value": [
            "early repurchase fee of 2%",
            "early withdrawal charge of 2%",
            "2% fee for shares held less than",
            "2.00% early repurchase",
            "early redemption fee",
            "2% early repurchase fee",
            "early repurchase fee equal to 2%",
        ],
        "medium_value": [
            "early repurchase fee",
            "early withdrawal fee",
            "early redemption",
        ],
        "low_value": [],
        "negative_high": [
            "no early repurchase fee",
            "no early withdrawal charge",
        ],
        "negative_medium": [],
    },

    # ===========================================================================
    # LEVERAGE DATAPOINTS
    # ===========================================================================
    "leverage_limits.uses_leverage": {
        "high_value": [
            "may borrow",
            "may use leverage",
            "credit facility",
            "borrowing",
            "line of credit",
            "will borrow",
            "intends to use leverage",
        ],
        "medium_value": [
            "leverage",
            "debt",
        ],
        "low_value": [],
        "negative_high": [
            "will not borrow",
            "does not use leverage",
            "will not use leverage",
        ],
        "negative_medium": [],
    },

    "leverage_limits.max_leverage_pct": {
        "high_value": [
            "borrow up to 33",
            "borrow up to 50",
            "33-1/3%",
            "33 1/3%",
            "leverage will not exceed",
            "not exceed 50%",
            "not exceed 33%",
            "asset coverage of 300%",
            "borrowings not to exceed",
        ],
        "medium_value": [
            "% of total assets",
            "% of net assets",
            "up to",
        ],
        "low_value": [],
        "negative_high": [],
        "negative_medium": [
            "leverage risk",
            "risks of leverage",
        ],
    },

    "leverage_limits.leverage_basis": {
        "high_value": [
            "of total assets",
            "of managed assets",
            "of net assets",
            "debt-to-equity",
            "asset coverage ratio",
            "percentage of total assets",
        ],
        "medium_value": [
            "total assets",
            "managed assets",
            "net assets",
        ],
        "low_value": [],
        "negative_high": [],
        "negative_medium": [],
    },

    # ===========================================================================
    # DISTRIBUTION DATAPOINTS
    # ===========================================================================
    "distribution_terms.distribution_frequency": {
        "high_value": [
            "distributions will be paid quarterly",
            "quarterly distributions",
            "monthly distributions",
            "distribute quarterly",
            "distribute monthly",
            "pay dividends quarterly",
            "intends to pay distributions quarterly",
        ],
        "medium_value": [
            "distribution",
            "dividend",
            "quarterly",
            "monthly",
        ],
        "low_value": [],
        "negative_high": [],
        "negative_medium": [
            "tax treatment of distributions",
        ],
    },

    "distribution_terms.default_distribution_policy": {
        "high_value": [
            "automatically reinvested",
            "reinvested unless",
            "reinvested in additional shares",
            "DRIP",
            "dividend reinvestment plan",
            "paid in cash",
            "cash distributions",
            "distributions will be reinvested",
        ],
        "medium_value": [
            "reinvested",
            "reinvestment",
            "cash",
        ],
        "low_value": [],
        "negative_high": [],
        "negative_medium": [],
    },

    # ===========================================================================
    # SHARE CLASSES DATAPOINTS
    # ===========================================================================
    "share_classes.share_classes": {
        "high_value": [
            "Class S shares",
            "Class D shares",
            "Class I shares",
            "Class A shares",
            "Institutional shares",
            "minimum initial investment",
            "minimum purchase",
            "sales load",
            "distribution fee",
        ],
        "medium_value": [
            "share class",
            "Class S",
            "Class D",
            "Class I",
            "minimum",
        ],
        "low_value": [],
        "negative_high": [
            "hypothetical investment",
            "expense example",
        ],
        "negative_medium": [
            "risk factors",
        ],
    },

    # ===========================================================================
    # ALLOCATION TARGETS DATAPOINTS
    # ===========================================================================
    "allocation_targets.secondary_funds_min_pct": {
        "high_value": [
            "secondary investments",
            "secondary funds",
            "secondaries allocation",
            "invest in secondaries",
            "secondary transactions",
            "secondary market",
        ],
        "medium_value": [
            "secondaries",
            "secondary",
        ],
        "low_value": [],
        "negative_high": [],
        "negative_medium": [],
    },

    "allocation_targets.secondary_funds_max_pct": {
        "high_value": [
            "secondary investments",
            "secondary funds",
            "up to",
            "maximum",
        ],
        "medium_value": [
            "secondaries",
            "secondary",
        ],
        "low_value": [],
        "negative_high": [],
        "negative_medium": [],
    },

    "allocation_targets.direct_investments_min_pct": {
        "high_value": [
            "direct investments",
            "direct co-investments",
            "co-investment",
            "directly in portfolio companies",
            "direct equity",
        ],
        "medium_value": [
            "direct",
            "co-invest",
        ],
        "low_value": [],
        "negative_high": [],
        "negative_medium": [],
    },

    "allocation_targets.direct_investments_max_pct": {
        "high_value": [
            "direct investments",
            "co-investments",
            "up to",
            "maximum",
        ],
        "medium_value": [
            "direct",
            "co-invest",
        ],
        "low_value": [],
        "negative_high": [],
        "negative_medium": [],
    },

    "allocation_targets.secondary_investments_min_pct": {
        "high_value": [
            "secondary investments",
            "secondary transactions",
            "secondaries",
        ],
        "medium_value": [
            "secondary",
        ],
        "low_value": [],
        "negative_high": [],
        "negative_medium": [],
    },

    # ===========================================================================
    # CONCENTRATION LIMITS DATAPOINTS
    # ===========================================================================
    "concentration_limits.max_single_asset_pct": {
        "high_value": [
            "single issuer",
            "single investment",
            "no more than 25%",
            "no more than 15%",
            "will not invest more than",
            "any one issuer",
        ],
        "medium_value": [
            "concentration",
            "single",
            "issuer",
        ],
        "low_value": [],
        "negative_high": [],
        "negative_medium": [
            "concentration risk",
        ],
    },

    "concentration_limits.max_single_fund_pct": {
        "high_value": [
            "single fund",
            "any one fund",
            "underlying fund",
            "Investment Fund",
        ],
        "medium_value": [
            "fund",
            "investment",
        ],
        "low_value": [],
        "negative_high": [],
        "negative_medium": [],
    },

    "concentration_limits.max_single_sector_pct": {
        "high_value": [
            "single industry",
            "any one industry",
            "single sector",
            "industry concentration",
        ],
        "medium_value": [
            "industry",
            "sector",
        ],
        "low_value": [],
        "negative_high": [],
        "negative_medium": [],
    },
}

# Penalty values for negative keywords
NEGATIVE_PENALTY_HIGH = 4
NEGATIVE_PENALTY_MEDIUM = 2


# =============================================================================
# DATAPOINT SCORING
# =============================================================================

@dataclass
class ScoredChunk:
    """A chunk with its relevance score for a specific datapoint."""
    chunk: Chunk
    score: int
    matching_keywords: list[str]


def score_chunk_for_datapoint(
    chunk: Chunk,
    datapoint_name: str,
) -> ScoredChunk:
    """
    Score a chunk's relevance to a specific datapoint.

    Args:
        chunk: The chunk to score
        datapoint_name: Full datapoint path like "incentive_fee.hurdle_rate_pct"

    Returns:
        ScoredChunk with score and matching keywords
    """
    keywords_config = DATAPOINT_KEYWORDS.get(datapoint_name, {})
    if not keywords_config:
        # Fall back to field-level keywords
        field_name = datapoint_name.split(".")[0]
        keywords_config = DATAPOINT_KEYWORDS.get(f"{field_name}.{field_name}", {})

    content_lower = chunk.content.lower()
    score = 0
    matching_keywords = []

    # Score positive keywords
    for kw in keywords_config.get("high_value", []):
        if kw.lower() in content_lower:
            score += 3
            matching_keywords.append(f"{kw} (+3)")

    for kw in keywords_config.get("medium_value", []):
        if kw.lower() in content_lower:
            score += 2
            matching_keywords.append(f"{kw} (+2)")

    for kw in keywords_config.get("low_value", []):
        if kw.lower() in content_lower:
            score += 1
            matching_keywords.append(f"{kw} (+1)")

    # Apply negative keyword penalties
    for kw in keywords_config.get("negative_high", []):
        if kw.lower() in content_lower:
            score -= NEGATIVE_PENALTY_HIGH
            matching_keywords.append(f"NEG:{kw} (-{NEGATIVE_PENALTY_HIGH})")

    for kw in keywords_config.get("negative_medium", []):
        if kw.lower() in content_lower:
            score -= NEGATIVE_PENALTY_MEDIUM
            matching_keywords.append(f"NEG:{kw} (-{NEGATIVE_PENALTY_MEDIUM})")

    return ScoredChunk(
        chunk=chunk,
        score=score,
        matching_keywords=matching_keywords,
    )


def get_top_chunks_for_datapoint(
    chunked_doc: ChunkedDocument,
    datapoint_name: str,
    top_k: int = 10,
) -> list[ScoredChunk]:
    """
    Get top K chunks most relevant to a specific datapoint.

    Args:
        chunked_doc: The chunked document
        datapoint_name: Full datapoint path like "incentive_fee.hurdle_rate_pct"
        top_k: Number of top chunks to return

    Returns:
        List of top K scored chunks, sorted by score descending
    """
    all_chunks = []
    for section in chunked_doc.chunked_sections:
        for chunk in section.chunks:
            scored = score_chunk_for_datapoint(chunk, datapoint_name)
            if scored.score > 0:
                all_chunks.append(scored)

    # Sort by score descending
    all_chunks.sort(key=lambda x: x.score, reverse=True)

    return all_chunks[:top_k]


# =============================================================================
# DATAPOINT EXTRACTION PROMPTS
# =============================================================================

DATAPOINT_PROMPTS = {
    # Incentive Fee
    "incentive_fee.has_incentive_fee": """Does this fund charge an incentive fee or performance fee at the FUND level (not underlying fund fees)?

Look for:
- "incentive fee", "performance fee", "performance-based fee"
- "incentive allocation", "carried interest"

IMPORTANT: Ignore references to underlying fund fees or AFFE.

Return JSON:
{"has_incentive_fee": true/false, "evidence": "<quote>"}""",

    "incentive_fee.incentive_fee_pct": """What is the incentive fee percentage charged by THIS fund?

Look for patterns like:
- "incentive fee equal to 10%"
- "performance fee of 12.5%"
- "10% of net profits"

IMPORTANT: This is the FUND-LEVEL fee, not underlying fund fees.

Return JSON:
{"incentive_fee_pct": "<number as string or null>", "evidence": "<quote>"}""",

    "incentive_fee.hurdle_rate_pct": """What is the hurdle rate for the incentive fee, expressed as an ANNUALIZED percentage?

Look for:
- "5% annualized hurdle rate"
- "hurdle rate of 8%"
- "1.25% quarterly" (= 5% annualized)
- "preferred return of 6%"

If stated quarterly, multiply by 4 to get annualized rate.

Return JSON:
{"hurdle_rate_pct": "<annualized number as string or null>", "evidence": "<quote>"}""",

    "incentive_fee.hurdle_rate_as_stated": """What is the PERIODIC hurdle rate BEFORE annualization?

IMPORTANT: Extract ONLY the number of the quarterly/monthly rate, NOT the annualized rate.

Examples:
- "1.25% quarterly" -> "1.25"
- "1.5% per quarter" -> "1.5"
- "2% monthly" -> "2"

If only an annualized rate is given (e.g., "5% annualized"), return null.

Return JSON:
{"hurdle_rate_as_stated": "<just the number like '1.25' or null>", "evidence": "<quote>"}""",

    "incentive_fee.hurdle_rate_frequency": """Is the hurdle rate stated on a quarterly or annual basis?

Look for:
- "quarterly hurdle" → "quarterly"
- "annualized hurdle" → "annual"
- "1.25% per quarter" → "quarterly"

Return JSON:
{"hurdle_rate_frequency": "quarterly"/"annual"/null, "evidence": "<quote>"}""",

    "incentive_fee.high_water_mark": """Does the fund use a high water mark or loss recovery mechanism?

Look for:
- "high water mark", "high-water mark"
- "loss recovery account", "loss carryforward"
- "deficit recovery", "prior losses must be recovered"

Return JSON:
{"high_water_mark": true/false/null, "evidence": "<quote>"}""",

    "incentive_fee.has_catch_up": """Does the incentive fee have a catch-up provision?

Look for:
- "catch-up", "full catch-up", "100% catch-up"
- "catch-up provision"

Return JSON:
{"has_catch_up": true/false/null, "evidence": "<quote>"}""",

    "incentive_fee.fee_basis": """What is the basis for calculating the incentive fee?

Options:
- "net_investment_income" (income-based)
- "net_profits" (total return including gains)
- "nav_appreciation" (NAV growth only)

Return JSON:
{"fee_basis": "<basis or null>", "evidence": "<quote>"}""",

    "incentive_fee.crystallization_frequency": """How often is the incentive fee crystallized/paid?

Look for:
- "quarterly in arrears" → "quarterly"
- "calculated and paid quarterly" → "quarterly"
- "annually" → "annual"

Return JSON:
{"crystallization_frequency": "quarterly"/"annual"/null, "evidence": "<quote>"}""",

    # Expense Cap
    "expense_cap.has_expense_cap": """Does the fund have an expense limitation or fee waiver agreement?

Look for:
- "expense limitation agreement"
- "fee waiver", "expense cap"
- "agreed to waive", "reimburse expenses"

Return JSON:
{"has_expense_cap": true/false, "evidence": "<quote>"}""",

    "expense_cap.expense_cap_pct": """What is the expense cap percentage?

Look for:
- "capped at 1.75%"
- "expense limitation of 2.00%"
- "limited to 1.50% of net assets"

Return JSON:
{"expense_cap_pct": "<number as string or null>", "evidence": "<quote>"}""",

    # Repurchase Terms
    "repurchase_terms.repurchase_frequency": """How often does the fund conduct repurchase offers?

Look for:
- "quarterly repurchase offers" → "quarterly"
- "semi-annual" → "semi-annual"
- "annual" → "annual"

Return JSON:
{"repurchase_frequency": "quarterly"/"semi-annual"/"annual"/null, "evidence": "<quote>"}""",

    "repurchase_terms.repurchase_amount_pct": """What percentage of shares does the fund offer to repurchase?

Look for:
- "5% to 25%"
- "at least 5%"
- "offer to repurchase 5%"

Return the MINIMUM percentage if a range is given.

Return JSON:
{"repurchase_amount_pct": "<number as string or null>", "evidence": "<quote>"}""",

    "repurchase_terms.repurchase_basis": """Is the repurchase offer based on outstanding shares or net assets?

Look for:
- "of outstanding shares" → "outstanding_shares"
- "of net assets" → "net_assets"
- "of NAV" → "nav"

Return JSON:
{"repurchase_basis": "outstanding_shares"/"net_assets"/"nav"/null, "evidence": "<quote>"}""",

    "repurchase_terms.lock_up_period_years": """What is the lock-up period in years?

Look for:
- "one year lock-up" → 1
- "12 months" → 1
- "no lock-up" → 0

Return JSON:
{"lock_up_period_years": <number or null>, "evidence": "<quote>"}""",

    "repurchase_terms.early_repurchase_fee_pct": """What is the early repurchase or early withdrawal fee percentage?

Look for:
- "2% early repurchase fee"
- "early withdrawal charge of 2%"
- "no early repurchase fee" → 0

Return JSON:
{"early_repurchase_fee_pct": "<number as string or null>", "evidence": "<quote>"}""",

    # Leverage
    "leverage_limits.uses_leverage": """Does the fund use or intend to use leverage/borrowing?

Look for:
- "may borrow", "will borrow"
- "credit facility", "line of credit"
- "intends to use leverage"

Return JSON:
{"uses_leverage": true/false, "evidence": "<quote>"}""",

    "leverage_limits.max_leverage_pct": """What is the maximum leverage as a percentage?

Look for:
- "borrow up to 33-1/3%" → 33.33
- "not exceed 50%" → 50
- "asset coverage of 300%" → 33.33 (means can borrow 1/3)

Return JSON:
{"max_leverage_pct": "<number as string or null>", "evidence": "<quote>"}""",

    "leverage_limits.leverage_basis": """What is the leverage calculated as a percentage of?

Look for:
- "of total assets" → "total_assets"
- "of net assets" → "net_assets"
- "of managed assets" → "managed_assets"

Return JSON:
{"leverage_basis": "total_assets"/"net_assets"/"managed_assets"/null, "evidence": "<quote>"}""",

    # Distribution Terms
    "distribution_terms.distribution_frequency": """How often does the fund pay distributions?

Look for:
- "quarterly distributions" → "quarterly"
- "monthly distributions" → "monthly"
- "annual distributions" → "annual"

Return JSON:
{"distribution_frequency": "monthly"/"quarterly"/"annual"/null, "evidence": "<quote>"}""",

    "distribution_terms.default_distribution_policy": """What is the default distribution policy (cash or reinvested)?

Look for:
- "automatically reinvested" → "reinvested"
- "DRIP", "dividend reinvestment plan" → "reinvested"
- "paid in cash" → "cash"

Return JSON:
{"default_distribution_policy": "cash"/"reinvested"/null, "evidence": "<quote>"}""",

    # Share Classes
    "share_classes.share_classes": """List all share classes with their minimum initial investments.

Extract for each class:
- Class name (S, D, I, etc.)
- Minimum initial investment amount
- Distribution/servicing fee if shown

Return JSON:
{
    "share_classes": [
        {"class_name": "Class S", "minimum_initial_investment": 2500, "distribution_fee_pct": "0.85"},
        {"class_name": "Class I", "minimum_initial_investment": 1000000, "distribution_fee_pct": null}
    ],
    "evidence": "<quote>"
}""",

    # Allocation Targets
    "allocation_targets.secondary_funds_min_pct": """What is the minimum allocation to secondary investments?

Return JSON:
{"secondary_funds_min_pct": "<number or null>", "evidence": "<quote>"}""",

    "allocation_targets.secondary_funds_max_pct": """What is the maximum allocation to secondary investments?

Return JSON:
{"secondary_funds_max_pct": "<number or null>", "evidence": "<quote>"}""",

    "allocation_targets.direct_investments_min_pct": """What is the minimum allocation to direct/co-investments?

Return JSON:
{"direct_investments_min_pct": "<number or null>", "evidence": "<quote>"}""",

    "allocation_targets.direct_investments_max_pct": """What is the maximum allocation to direct/co-investments?

Return JSON:
{"direct_investments_max_pct": "<number or null>", "evidence": "<quote>"}""",

    # Concentration Limits
    "concentration_limits.max_single_asset_pct": """What is the maximum investment in a single issuer/asset?

Return JSON:
{"max_single_asset_pct": "<number or null>", "evidence": "<quote>"}""",

    "concentration_limits.max_single_fund_pct": """What is the maximum investment in a single underlying fund?

Return JSON:
{"max_single_fund_pct": "<number or null>", "evidence": "<quote>"}""",

    "concentration_limits.max_single_sector_pct": """What is the maximum investment in a single industry/sector?

Return JSON:
{"max_single_sector_pct": "<number or null>", "evidence": "<quote>"}""",
}


# =============================================================================
# PER-DATAPOINT EXTRACTION
# =============================================================================

@dataclass
class DatapointExtractionResult:
    """Result from extracting a single datapoint."""
    datapoint_name: str
    value: Any
    evidence: Optional[str]
    chunks_searched: int
    top_chunk_score: int
    extraction_time_ms: int


@dataclass
class PerDatapointExtractionTrace:
    """Full trace of per-datapoint extraction."""
    fund_name: str
    total_datapoints: int
    successful_extractions: int
    total_chunks_searched: int
    total_extraction_time_ms: int
    datapoint_results: dict[str, DatapointExtractionResult]


def extract_single_datapoint(
    chunked_doc: ChunkedDocument,
    datapoint_name: str,
    client,
    model: str,
    provider: str,
    rate_limit: Optional[RateLimitConfig] = None,
    top_k_chunks: int = 10,
) -> DatapointExtractionResult:
    """
    Extract a single datapoint from the document.

    Args:
        chunked_doc: The chunked document
        datapoint_name: Full path like "incentive_fee.hurdle_rate_pct"
        client: LLM client
        model: Model name
        provider: Provider name
        rate_limit: Rate limiting config
        top_k_chunks: Number of top chunks to use

    Returns:
        DatapointExtractionResult with extracted value
    """
    start_time = time.time()

    # Get top chunks for this datapoint
    top_chunks = get_top_chunks_for_datapoint(chunked_doc, datapoint_name, top_k_chunks)

    if not top_chunks:
        return DatapointExtractionResult(
            datapoint_name=datapoint_name,
            value=None,
            evidence=None,
            chunks_searched=0,
            top_chunk_score=0,
            extraction_time_ms=int((time.time() - start_time) * 1000),
        )

    # Combine chunk content
    combined_text = "\n\n".join([
        f"[Section: {sc.chunk.section_title}]\n{sc.chunk.content}"
        for sc in top_chunks
    ])

    # Limit size
    if len(combined_text) > 12000:
        combined_text = combined_text[:12000]

    # Get prompt for this datapoint
    prompt = DATAPOINT_PROMPTS.get(datapoint_name)
    if not prompt:
        # Generate generic prompt
        field_name = datapoint_name.split(".")[-1]
        prompt = f"""Extract the {field_name.replace('_', ' ')} from the text.

Return JSON:
{{"{field_name}": <value or null>, "evidence": "<quote>"}}"""

    full_prompt = f"{prompt}\n\nTEXT:\n{combined_text}"

    messages = [
        {"role": "system", "content": "You are a precise financial document analyst. Extract exact values from SEC filings. Return valid JSON only."},
        {"role": "user", "content": full_prompt},
    ]

    try:
        result = call_llm_json(
            client=client,
            provider=provider,
            model=model,
            messages=messages,
            rate_limit=rate_limit,
        )

        # Extract the value and evidence
        if isinstance(result, dict):
            # Get the primary value (first non-evidence key)
            value = None
            evidence = result.get("evidence")
            for key, val in result.items():
                if key != "evidence":
                    value = val
                    break

            return DatapointExtractionResult(
                datapoint_name=datapoint_name,
                value=value,
                evidence=evidence,
                chunks_searched=len(top_chunks),
                top_chunk_score=top_chunks[0].score if top_chunks else 0,
                extraction_time_ms=int((time.time() - start_time) * 1000),
            )

    except Exception as e:
        logger.error(f"Failed to extract {datapoint_name}: {e}")

    return DatapointExtractionResult(
        datapoint_name=datapoint_name,
        value=None,
        evidence=None,
        chunks_searched=len(top_chunks),
        top_chunk_score=top_chunks[0].score if top_chunks else 0,
        extraction_time_ms=int((time.time() - start_time) * 1000),
    )


class PerDatapointExtractor:
    """
    Extracts data points one at a time with datapoint-specific keywords.
    """

    # Define which datapoints to extract (matching ground truth fields)
    DATAPOINTS_TO_EXTRACT = [
        # Incentive Fee
        "incentive_fee.has_incentive_fee",
        "incentive_fee.incentive_fee_pct",
        "incentive_fee.hurdle_rate_pct",
        "incentive_fee.hurdle_rate_as_stated",
        "incentive_fee.hurdle_rate_frequency",
        "incentive_fee.high_water_mark",
        "incentive_fee.has_catch_up",
        "incentive_fee.fee_basis",
        "incentive_fee.crystallization_frequency",
        # Expense Cap
        "expense_cap.has_expense_cap",
        "expense_cap.expense_cap_pct",
        # Repurchase Terms
        "repurchase_terms.repurchase_frequency",
        "repurchase_terms.repurchase_amount_pct",
        "repurchase_terms.repurchase_basis",
        "repurchase_terms.lock_up_period_years",
        "repurchase_terms.early_repurchase_fee_pct",
        # Leverage
        "leverage_limits.uses_leverage",
        "leverage_limits.max_leverage_pct",
        "leverage_limits.leverage_basis",
        # Distribution
        "distribution_terms.distribution_frequency",
        "distribution_terms.default_distribution_policy",
        # Share Classes
        "share_classes.share_classes",
        # Allocation Targets (for funds that have them)
        "allocation_targets.secondary_funds_min_pct",
        "allocation_targets.secondary_funds_max_pct",
        "allocation_targets.direct_investments_min_pct",
        "allocation_targets.direct_investments_max_pct",
        # Concentration Limits
        "concentration_limits.max_single_asset_pct",
        "concentration_limits.max_single_fund_pct",
        "concentration_limits.max_single_sector_pct",
    ]

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        delay_between_calls: float = 0.5,
        requests_per_minute: int = 60,
        top_k_chunks: int = 10,
    ):
        self.model = resolve_model_name(model)
        self.provider = provider or detect_provider(model).value
        self.api_key = api_key
        self.top_k_chunks = top_k_chunks

        self.rate_limit = RateLimitConfig(
            delay_between_calls=delay_between_calls,
            requests_per_minute=requests_per_minute,
        )

        self.client = create_raw_client(
            provider=self.provider,
            model=self.model,
            api_key=api_key,
            rate_limit=self.rate_limit,
        )

    def extract_all(
        self,
        chunked_doc: ChunkedDocument,
        fund_name: str,
    ) -> tuple[dict, PerDatapointExtractionTrace]:
        """
        Extract all datapoints from a document.

        Args:
            chunked_doc: The chunked document
            fund_name: Name of the fund

        Returns:
            Tuple of (extraction_result dict, trace)
        """
        logger.info(f"[Per-Datapoint] Extracting {len(self.DATAPOINTS_TO_EXTRACT)} datapoints for {fund_name}")

        start_time = time.time()
        datapoint_results = {}
        extraction_result = {}
        successful = 0
        total_chunks = 0

        for datapoint_name in self.DATAPOINTS_TO_EXTRACT:
            logger.info(f"  Extracting: {datapoint_name}")

            result = extract_single_datapoint(
                chunked_doc=chunked_doc,
                datapoint_name=datapoint_name,
                client=self.client,
                model=self.model,
                provider=self.provider,
                rate_limit=self.rate_limit,
                top_k_chunks=self.top_k_chunks,
            )

            datapoint_results[datapoint_name] = result
            total_chunks += result.chunks_searched

            if result.value is not None:
                successful += 1
                # Add to extraction result dict
                self._add_to_result(extraction_result, datapoint_name, result.value)

            logger.info(f"    → {result.value} (score: {result.top_chunk_score}, {result.extraction_time_ms}ms)")

        trace = PerDatapointExtractionTrace(
            fund_name=fund_name,
            total_datapoints=len(self.DATAPOINTS_TO_EXTRACT),
            successful_extractions=successful,
            total_chunks_searched=total_chunks,
            total_extraction_time_ms=int((time.time() - start_time) * 1000),
            datapoint_results=datapoint_results,
        )

        logger.info(f"[Per-Datapoint] Complete: {successful}/{len(self.DATAPOINTS_TO_EXTRACT)} extracted in {trace.total_extraction_time_ms}ms")

        return extraction_result, trace

    def _add_to_result(self, result: dict, datapoint_name: str, value: Any):
        """Add an extracted value to the result dict, creating nested structure as needed."""
        parts = datapoint_name.split(".")

        if len(parts) == 2:
            field_name, subfield = parts

            if field_name not in result:
                result[field_name] = {}

            result[field_name][subfield] = value
        else:
            # Simple field
            result[datapoint_name] = value
