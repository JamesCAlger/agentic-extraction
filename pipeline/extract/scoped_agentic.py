"""
Tier 3: Keyword-Guided Scoped Agentic Search with Optional Reranking

This module implements scoped agentic extraction for fields that fail
batch extraction due to large document size or scattered information.

The approach:
1. Score all sections by field-specific keywords (first-pass retrieval)
2. Optionally rerank chunks using Cohere Rerank API (semantic relevance)
3. Select top K candidate chunks based on reranker scores
4. Run focused LLM extraction on reranked chunks
5. Aggregate and return best result

The reranker step significantly improves chunk selection by using
semantic understanding rather than just keyword matching.
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from decimal import Decimal

from .llm_provider import (
    create_raw_client,
    call_llm_json,
    RateLimitConfig,
    detect_provider,
    resolve_model_name,
    LLMProvider,
)

from ..parse.models import ChunkedDocument, ChunkedSection, Chunk, SectionType

logger = logging.getLogger(__name__)


# =============================================================================
# RERANKER CONFIGURATION
# =============================================================================

@dataclass
class RerankerConfig:
    """Configuration for Cohere Reranker."""
    enabled: bool = False
    model: str = "rerank-v3.5"  # Cohere rerank model
    first_pass_n: int = 50  # Chunks to send to reranker from keyword scoring
    top_k: int = 15  # Chunks to return from reranker
    score_threshold: float = 0.3  # Minimum reranker score to include
    api_key: Optional[str] = None  # If None, uses COHERE_API_KEY env var


@dataclass
class RerankedChunk:
    """A chunk with its reranker relevance score."""
    chunk: Chunk
    relevance_score: float
    original_index: int


# =============================================================================
# DOCVQA-STYLE RERANKER QUERIES
# =============================================================================
# These natural language questions are used by the reranker to find relevant chunks.
# Format: field_name -> query string

# =============================================================================
# ADAPTIVE RETRIEVAL SETTINGS
# =============================================================================
# Document size thresholds and corresponding settings

def get_adaptive_settings(num_chunks: int) -> dict:
    """
    Get retrieval settings based on document size.

    Large documents (like StepStone with 2800+ chunks) need more aggressive
    retrieval to find scattered information.

    Args:
        num_chunks: Total number of chunks in the document

    Returns:
        Dict with top_k, max_chunks_per_section, first_pass_n settings
    """
    if num_chunks > 2000:
        # Very large documents (StepStone: 2800+ chunks)
        return {
            "top_k_sections": 20,  # Up from default 5-10
            "max_chunks_per_section": 15,  # Up from default 10
            "first_pass_n": 400,  # Up from default 50-200
            "reranker_top_k": 25,  # More chunks to reranker
        }
    elif num_chunks > 1000:
        # Large documents
        return {
            "top_k_sections": 15,
            "max_chunks_per_section": 12,
            "first_pass_n": 300,
            "reranker_top_k": 20,
        }
    elif num_chunks > 500:
        # Medium documents
        return {
            "top_k_sections": 10,
            "max_chunks_per_section": 10,
            "first_pass_n": 200,
            "reranker_top_k": 15,
        }
    else:
        # Small documents (default)
        return {
            "top_k_sections": 5,
            "max_chunks_per_section": 10,
            "first_pass_n": 50,
            "reranker_top_k": 10,
        }


RERANKER_QUERIES = {
    "minimum_investment": (
        "What is the minimum initial investment amount required to purchase shares "
        "in each share class (Class S, Class D, Class I)?"
    ),
    "minimum_additional_investment": (
        "What is the minimum subsequent or additional investment amount required "
        "for each share class after the initial purchase?"
    ),
    "share_classes": (
        "What share classes does the fund offer, and what are the fees, minimums, "
        "and distribution fees for each class (Class S, Class D, Class I)?"
    ),
    "incentive_fee": (
        "What is the incentive fee or performance fee percentage charged by this fund, "
        "including the hurdle rate, high water mark policy, and catch-up provision?"
    ),
    "expense_cap": (
        "What is the expense limitation or fee waiver agreement, including the "
        "cap percentage and when it expires?"
    ),
    "repurchase_terms": (
        "How often does the fund conduct repurchase offers, what percentage of shares "
        "is offered for repurchase, and are there any lock-up periods or early repurchase fees?"
    ),
    "repurchase_basis": (
        "Is the repurchase offer calculated as a percentage of outstanding shares, "
        "net assets, or NAV?"
    ),
    "lock_up": (
        "What is the lock-up period for shares and what early repurchase fee or "
        "early withdrawal charge applies to shares redeemed before one year?"
    ),
    "leverage_limits": (
        "What is the fund's maximum borrowing or leverage limit as a percentage of assets, "
        "and does the fund have a credit facility?"
    ),
    "distribution_terms": (
        "How often does the fund pay distributions (monthly, quarterly, annually) "
        "and what is the default distribution policy (cash or reinvested/DRIP)?"
    ),
    "allocation_targets": (
        "What percentage of the fund is allocated to private equity, private credit, "
        "real estate, infrastructure, or other asset classes?"
    ),
    "concentration_limits": (
        "What is the maximum percentage the fund can invest in a single issuer, "
        "industry, or underlying fund?"
    ),
    "hurdle_rate": (
        "What is the hurdle rate or preferred return that must be achieved before "
        "incentive fees are charged?"
    ),
    "incentive_fee_high_water_mark": (
        "Does the fund use a high water mark, loss recovery account, or deficit carryforward "
        "mechanism to prevent double-charging incentive fees on recovered losses?"
    ),
    "fund_metadata": (
        "What is the fund name, investment manager or adviser, fiscal year end, "
        "and whether it is an interval fund or tender offer fund?"
    ),
    # =========================================================================
    # FIELD-SPECIFIC SEMANTIC QUERIES FOR PROBLEMATIC FIELDS
    # =========================================================================
    # These additional queries target specific sub-fields that have low accuracy
    # (e.g., StepStone's allocation_targets and concentration_limits)
    "secondary_funds_min_pct": (
        "What percentage range does the fund allocate to secondary investments? "
        "What is the minimum allocation to secondary funds or secondary market investments?"
    ),
    "secondary_funds_max_pct": (
        "What is the maximum percentage the fund can allocate to secondary investments "
        "or secondary market opportunities?"
    ),
    "direct_investments_min_pct": (
        "What is the minimum percentage the fund allocates to direct investments, "
        "co-investments, or direct equity/credit positions?"
    ),
    "direct_investments_max_pct": (
        "What is the maximum percentage the fund can invest directly in companies "
        "rather than through underlying funds?"
    ),
    "max_single_asset_pct": (
        "What is the maximum percentage the fund can invest in any single issuer, "
        "company, or direct investment?"
    ),
    "max_single_fund_pct": (
        "What is the maximum percentage the fund can commit to any single underlying "
        "fund, portfolio fund, or investment vehicle?"
    ),
    "max_single_sector_pct": (
        "What is the maximum percentage the fund can allocate to any single industry "
        "sector, geographic region, or strategy?"
    ),
    "underlying_fund_incentive_range": (
        "What incentive fees or carried interest do the underlying funds or portfolio "
        "funds typically charge? What is the range of incentive fees charged by "
        "the Investment Funds in which the fund invests?"
    ),
}


class CohereReranker:
    """
    Cohere Reranker for semantic chunk relevance scoring.

    Uses Cohere's rerank API to score chunks by semantic relevance
    to a natural language query, improving on keyword-based retrieval.
    """

    def __init__(self, config: RerankerConfig):
        """
        Initialize Cohere Reranker.

        Args:
            config: RerankerConfig with API key and parameters
        """
        self.config = config
        self.api_key = config.api_key or os.getenv("COHERE_API_KEY")
        self._client = None
        self._last_call_time = 0.0
        # Production tier: 10,000 calls/min, use 100ms delay
        self._min_delay = 0.1  # 100ms between calls for production tier
        self._call_count = 0

        if not self.api_key and config.enabled:
            logger.warning("Cohere API key not found. Set COHERE_API_KEY env var or pass api_key.")

    @property
    def client(self):
        """Lazy-load Cohere client."""
        if self._client is None:
            try:
                import cohere
                self._client = cohere.ClientV2(api_key=self.api_key)
            except ImportError:
                logger.error("cohere package not installed. Run: pip install cohere")
                raise
        return self._client

    def rerank_chunks(
        self,
        chunks: list[Chunk],
        field_name: str,
        custom_query: Optional[str] = None,
    ) -> list[RerankedChunk]:
        """
        Rerank chunks by semantic relevance to field.

        Args:
            chunks: List of chunks to rerank (already filtered by keywords)
            field_name: Field being extracted (for query selection)
            custom_query: Optional custom query (overrides default)

        Returns:
            List of RerankedChunk sorted by relevance score (highest first)
        """
        if not self.config.enabled or not self.api_key:
            logger.debug("Reranker disabled, returning chunks as-is")
            return [
                RerankedChunk(chunk=c, relevance_score=1.0, original_index=i)
                for i, c in enumerate(chunks)
            ]

        if not chunks:
            return []

        # Get query for this field
        query = custom_query or RERANKER_QUERIES.get(
            field_name,
            f"What is the {field_name.replace('_', ' ')} for this fund?"
        )

        # Limit to first_pass_n chunks
        candidates = chunks[:self.config.first_pass_n]

        # Prepare documents for reranking
        documents = [chunk.content for chunk in candidates]

        # Rate limiting with exponential backoff for trial keys
        elapsed = time.time() - self._last_call_time
        if elapsed < self._min_delay:
            time.sleep(self._min_delay - elapsed)

        self._call_count += 1
        max_retries = 3
        retry_delay = 6.0  # Start with 6 seconds for trial key (10/min limit)
        response = None

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"    [Reranker] Call #{self._call_count}: Reranking {len(documents)} chunks "
                    f"for '{field_name}'"
                )

                response = self.client.rerank(
                    model=self.config.model,
                    query=query,
                    documents=documents,
                    top_n=self.config.top_k,
                )

                self._last_call_time = time.time()
                break  # Success, exit retry loop

            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"    [Reranker] Rate limited, waiting {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # Final attempt failed or non-rate-limit error
                    logger.error(f"    [Reranker] Error after retries: {e}")
                    return [
                        RerankedChunk(chunk=c, relevance_score=1.0, original_index=i)
                        for i, c in enumerate(candidates[:self.config.top_k])
                    ]

        if response is None:
            # All retries exhausted
            logger.error("    [Reranker] All retries exhausted")
            return [
                RerankedChunk(chunk=c, relevance_score=1.0, original_index=i)
                for i, c in enumerate(candidates[:self.config.top_k])
            ]

        try:
            # Build reranked results with threshold filtering
            results = []
            for r in response.results:
                if r.relevance_score >= self.config.score_threshold:
                    results.append(RerankedChunk(
                        chunk=candidates[r.index],
                        relevance_score=r.relevance_score,
                        original_index=r.index,
                    ))

            logger.info(
                f"    [Reranker] Returned {len(results)} chunks "
                f"(threshold: {self.config.score_threshold})"
            )

            # Log top 3 scores for debugging
            for i, r in enumerate(results[:3]):
                logger.debug(
                    f"      #{i+1}: score={r.relevance_score:.3f}, "
                    f"section={r.chunk.section_title[:40]}"
                )

            return results

        except Exception as e:
            logger.error(f"    [Reranker] Error: {e}")
            # Fall back to returning original chunks without reranking
            return [
                RerankedChunk(chunk=c, relevance_score=1.0, original_index=i)
                for i, c in enumerate(candidates[:self.config.top_k])
            ]

    def get_stats(self) -> dict:
        """Return reranker statistics for experiment tracking."""
        return {
            "enabled": self.config.enabled,
            "model": self.config.model,
            "first_pass_n": self.config.first_pass_n,
            "top_k": self.config.top_k,
            "score_threshold": self.config.score_threshold,
            "total_calls": self._call_count,
        }


# =============================================================================
# KEYWORD CONFIGURATION
# =============================================================================

# =============================================================================
# NEGATIVE KEYWORDS - Patterns that indicate definitional/non-data sections
# =============================================================================
# These patterns suggest a section is explaining concepts rather than stating
# actual fund-specific values. They should be penalized in scoring.

# Global negative keywords that apply to ALL fields
# Based on analysis of actual SEC filings and extraction errors
GLOBAL_NEGATIVE_KEYWORDS = {
    # Definitional language - explains what a term means, not actual values
    "definitional": {
        "high_penalty": [  # -4 points - strong indicator of definitions
            "is defined as",
            "shall mean",
            "refers to",
            "is a common term for",
            "is commonly defined",
            "for purposes of this",
            "as used herein",
            "the term",
        ],
        "medium_penalty": [  # -2 points - moderate indicator
            "means",
            "are typically",
            "is typically",
            "are generally",
            "in general",
            "commonly refers",
        ],
    },
    # Glossary and definitions sections
    "glossary": {
        "high_penalty": [
            "glossary of terms",
            "definitions section",
            "defined terms",
            "certain defined terms",
            "key definitions",
        ],
        "medium_penalty": [
            "glossary",
            "definitions",
        ],
    },
    # Risk explanations - describe what a risk IS, not actual fund terms
    "risk_explanation": {
        "high_penalty": [
            "is the risk that",
            "risk is the possibility",
            "this risk arises when",
            "this risk occurs when",
            "risks associated with",
        ],
        "medium_penalty": [
            "may be subject to various risks",
            "risks may include",
            "potential risks include",
        ],
    },
    # Hypothetical/example language - not actual fund values
    "hypothetical": {
        "high_penalty": [
            "hypothetical example",
            "for illustration purposes",
            "hypothetical investment",
            "hypothetical $1,000",
        ],
        "medium_penalty": [
            "for example",
            "such as",
            "example:",
            "illustrative",
        ],
    },
    # Generic market commentary - not fund-specific
    "market_commentary": {
        "medium_penalty": [
            "in the market",
            "market conditions",
            "economic conditions",
            "industry trends",
        ],
    },
}

# Field-specific negative keywords
# These help distinguish fund-level terms from underlying fund terms
FIELD_NEGATIVE_KEYWORDS = {
    # For incentive_fee: distinguish fund-level vs underlying fund fees
    "incentive_fee": {
        "high_penalty": [  # -4 points
            "underlying fund fees",
            "underlying funds charge",
            "underlying funds typically charge",
            "Investment Funds charge",
            "Investment Funds typically charge",
            "Investment Funds may charge",
            "private funds typically charge",
            "investment vehicles charge",
            "AFFE",  # Acquired Fund Fees and Expenses = underlying, not fund-level
            "acquired fund fees",
        ],
        "medium_penalty": [  # -2 points
            "underlying fund",
            "underlying funds",
            "Investment Funds may",
            "private funds may",
            "generally charge",  # Often refers to industry norms, not this fund
        ],
    },
    # For expense_cap: distinguish actual caps from total expenses
    "expense_cap": {
        "high_penalty": [
            "total annual expenses",  # Actual expense ratio, not a cap
            "total expenses were",
            "total operating expenses",  # Historical, not a limit
        ],
        "medium_penalty": [
            "total expenses",
            "expense ratio",  # May be actual ratio, not cap
        ],
    },
    # For leverage_limits: filter out general leverage risk discussions
    "leverage_limits": {
        "high_penalty": [
            "leverage risk is the risk",
            "risks of leverage",
            "leveraging risk",
        ],
        "medium_penalty": [
            "leverage may magnify",
            "leverage can increase",
            "leverage risk",
        ],
    },
    # For minimum_investment: filter out example calculations and intermediary-delegated minimums
    "minimum_investment": {
        "high_penalty": [
            "hypothetical investment",
            "example investment",
            # Intermediary-delegated minimums - not fund-level
            "determined by financial intermediaries",
            "set by selling agents",
            "policies of your financial intermediary",
            "no minimum investment at the fund level",
            "broker-dealer may impose",
            "your financial advisor may require",
        ],
        "medium_penalty": [
            "example:",
            "projected dollar amount",  # Expense example tables
            "financial intermediary",
            "selling agent",
            "broker-dealer",
        ],
    },
    # For share_classes: filter out risk discussions and intermediary-delegated minimums
    "share_classes": {
        "high_penalty": [
            # Intermediary-delegated minimums - not fund-level
            "determined by financial intermediaries",
            "set by selling agents",
            "policies of your financial intermediary",
            "no minimum investment at the fund level",
        ],
        "medium_penalty": [
            "risks of investing",
            "risk factors",
            "principal risks",
            "financial intermediary may impose",
        ],
    },
    # For allocation_targets: filter out risk/market discussions
    "allocation_targets": {
        "medium_penalty": [
            "sector risk",
            "concentration risk",
            "market conditions may",
        ],
    },
    # For distribution_terms: filter out tax discussions AND repurchase/incentive fee patterns
    "distribution_terms": {
        "high_penalty": [
            # These are NOT distribution frequency - common confusion
            "quarterly repurchase",
            "repurchase offer",
            "tender offer",
            "incentive fee",
            "performance fee",
            "quarterly in arrears",  # This is incentive fee crystallization, not distribution
        ],
        "medium_penalty": [
            "tax consequences",
            "tax treatment",
            "taxable income",
            "for tax purposes",
            "repurchase",
            "tender",
            "incentive",
        ],
    },
    # For repurchase_terms: filter out risk warnings
    "repurchase_terms": {
        "medium_penalty": [
            "liquidity risk",
            "may not be able to repurchase",
            "no guarantee",
        ],
    },
    # For concentration_limits: filter out risk explanations
    "concentration_limits": {
        "medium_penalty": [
            "concentration risk",
            "concentrated positions may",
        ],
    },
}

# Penalty values for negative keyword tiers
NEGATIVE_PENALTY_HIGH = 4
NEGATIVE_PENALTY_MEDIUM = 2


# Keywords for scoring sections - field-specific
# Includes synonyms for semantic equivalents to catch variant terminology
FIELD_KEYWORDS = {
    "minimum_investment": {
        "high_value": [  # 3 points each
            "minimum initial investment",
            "minimum investment",
            "$2,500",
            "$2500",
            "$1,000,000",
            "$1000000",
            "minimum subsequent",
        ],
        "medium_value": [  # 2 points each
            "minimum",
            "initial investment",
            "class s",
            "class d",
            "class i",
            "purchase shares",
        ],
        "low_value": [  # 1 point each
            "investment",
            "shares",
            "purchase",
            "offering",
        ],
    },
    "share_classes": {
        "high_value": [
            "class s shares",
            "class d shares",
            "class i shares",
            "class i advisory",
            "minimum initial investment",
            "minimum subsequent investment",
            "minimum additional investment",
            "minimum additional purchase",
            "distribution fee",
            "distribution and/or service fee",
            "shareholder servicing fee",
            "12b-1 fee",
            "sales load",
            "sales charge",
            "upfront placement fee",
            "brokerage commission",
            "no sales load",
            "no distribution fee",
            # Specific dollar amounts
            "minimum investment of $2,500",
            "minimum investment of $500",
            "subsequent investment of $500",
            "additional investment of $500",
        ],
        "medium_value": [
            "class s",
            "class d",
            "class i",
            "share class",
            "minimum",
            "$2,500",
            "$2500",
            "$500",
            "$1,000,000",
            "$1000000",
            "0.85%",
            "0.75%",
            "0.25%",
            "3.5%",
            "1.5%",
            "selling agents",
            "financial intermediaries",
            "not subject to",
            "no upfront",
            "waived",
        ],
        "low_value": [
            "shares",
            "fee",
            "investment",
            "purchase",
        ],
    },
    # Lock-up and early redemption keywords (with synonyms)
    "lock_up": {
        "high_value": [
            "early repurchase fee",
            "early withdrawal charge",
            "lock-up period",
            "early redemption fee",
            "2% fee",
            "within one year",
            "within 1 year",
            "shares held less than",
            # Synonyms
            "holding period requirement",
            "redemption holdback",
            "short-term trading fee",
        ],
        "medium_value": [
            "lock-up",
            "lock up",
            "lockup",
            "early repurchase",
            "early withdrawal",
            "early redemption",
            "12 months",
            "one year",
            "1 year",
            # Synonyms
            "holding period",
            "soft lock",
            "redemption penalty",
        ],
        "low_value": [
            "repurchase",
            "withdrawal",
            "redemption",
            "fee",
        ],
    },
    "leverage_limits": {
        "high_value": [
            "may borrow up to",
            "leverage will not exceed",
            "debt-to-equity ratio",
            "33-1/3%",
            "50% of total",
            "asset coverage ratio",
            "credit facility",
            # 1940 Act asset coverage patterns (CRITICAL for interval funds)
            "asset coverage (as defined in the 1940 Act)",
            "asset coverage of 300%",
            "asset coverage of at least 300%",
            "300% asset coverage",
            "150% asset coverage",
            "1940 Act limit",
            "comply with the 300%",
            # Blue Owl specific section names
            "Borrowings",
            "borrowings",
        ],
        "medium_value": [
            "leverage",
            "borrowing",
            "debt",
            "total assets",
            "net assets",
            "300%",
            "150%",
            "asset coverage",
            "as defined in the 1940 Act",
        ],
        "low_value": [
            "borrow",
            "loan",
            "credit",
        ],
    },
    # Distribution terms keywords
    "distribution_terms": {
        "high_value": [
            "distribution policy",
            "dividend policy",
            "distributions will be paid",
            "dividend reinvestment plan",
            "DRIP",
            "distributions will be reinvested",
            # Monthly distribution patterns (important for credit funds like Blue Owl)
            "declare daily distributions",
            "pay monthly",
            "paid monthly",
            "monthly distributions",
            "intends to declare",
            "distribute them to shareholders monthly",
            "distributions declared daily",
            # Blue Owl specific section names
            "Annual Distribution Requirement",
            "annual distribution requirement",
            "Distribution Plan",
            "distribution plan",
        ],
        "medium_value": [
            "distribution",
            "dividend",
            "quarterly distribution",
            "monthly distribution",
            "annual distribution",
            "reinvested",
            "cash distribution",
            "daily",
            "monthly",
            "income distributions",
            # NOTE: "Distribution and Servicing Fee" is about fees, NOT distribution frequency
            # So we don't add high keywords for "servicing fee"
        ],
        "low_value": [
            "income",
            "capital gains",
            "payout",
        ],
    },
    # Minimum additional/subsequent investment keywords (with synonyms)
    "minimum_additional_investment": {
        "high_value": [
            "minimum subsequent investment",
            "minimum additional investment",
            "additional purchases",
            "subsequent purchases",
            "minimum for additional",
            # Synonyms
            "follow-on investment",
            "incremental contribution",
            "additional subscription",
        ],
        "medium_value": [
            "subsequent investment",
            "additional investment",
            "$500",
            "$100",
            "minimum",
            "follow-on",
            # Synonyms
            "incremental",
            "subsequent purchase",
        ],
        "low_value": [
            "purchase",
            "investment",
            "additional",
        ],
    },
    # Repurchase basis keywords
    "repurchase_basis": {
        "high_value": [
            "of outstanding shares",
            "of net assets",
            "of NAV",
            "number of shares",
            "percentage of shares",
        ],
        "medium_value": [
            "repurchase offer",
            "tender offer",
            "quarterly repurchase",
            "5% of",
            "25% of",
        ],
        "low_value": [
            "repurchase",
            "shares",
            "NAV",
        ],
    },
    # =============================================================================
    # SEMANTIC-RISK FIELDS (high synonym variance)
    # =============================================================================
    # High water mark / loss recovery mechanism
    "incentive_fee_high_water_mark": {
        "high_value": [
            # Standard terms
            "high water mark",
            "high-water mark",
            "highwater mark",
            # Semantic equivalents
            "loss recovery account",
            "loss carryforward",
            "deficit recovery",
            "deficit carryforward",
            "cumulative loss recovery",
            "prior period losses",
            "recover prior losses",
            "recoup losses",
            # Mechanism descriptions
            "crystallization",
            "prevent double charging",
            "losses must be recovered",
        ],
        "medium_value": [
            "incentive fee",
            "performance fee",
            "incentive allocation",
            "performance allocation",
            "carried interest",
            "promote",
            "loss recovery",
            "deficit",
            "cumulative loss",
            "prior losses",
        ],
        "low_value": [
            "incentive",
            "performance",
            "hurdle",
            "benchmark",
        ],
    },
    # Expense cap / fee waiver
    "expense_cap": {
        "high_value": [
            # Standard terms
            "expense cap",
            "expense limitation",
            "expense limit",
            # Semantic equivalents
            "fee waiver",
            "fee waiver agreement",
            "expense reimbursement",
            "contractual cap",
            "voluntary cap",
            "operating expense limit",
            "total expense cap",
            # Mechanism descriptions
            "waive fees",
            "reimburse expenses",
            "limit total expenses",
        ],
        "medium_value": [
            "expense limitation agreement",
            "waiver",
            "reimbursement",
            "capped at",
            "limited to",
            "not exceed",
            "contractual arrangement",
        ],
        "low_value": [
            "expenses",
            "waive",
            "cap",
            "limit",
        ],
    },
    # Repurchase terms (interval vs tender)
    "repurchase_terms": {
        "high_value": [
            "repurchase offer",
            "tender offer",
            "interval fund",
            "quarterly repurchase",
            "repurchase at NAV",
            "5% to 25%",
            "at least 5%",
            "up to 25%",
            # Interval fund specific patterns
            "repurchase of shares",
            "repurchases of shares",
            "Rule 23c-3",
            "periodic repurchase offers",
            "quarterly repurchase offers",
            "between 5% and 25%",
            "minimum amount of 5%",
            "repurchase requests",
            "repurchase offer amount",
            "repurchase pricing date",
            # Blue Owl specific section names (HIGH PRIORITY)
            "Repurchase Request Deadline",
            "repurchase request deadline",
            "Repurchase Pricing Date",
            "repurchase pricing date",
            "Early Repurchase Fee",
            "early repurchase fee",
            # Direct credit fund patterns (Blue Owl)
            "discretion of the Board",
            "board may determine",
            "outstanding Shares",
        ],
        "medium_value": [
            "repurchase",
            "tender",
            "redemption",
            "liquidity",
            "quarterly",
            "semi-annual",
            "annual",
            "5%",
            "25%",
            "outstanding shares",
            "net asset value",
            "pricing date",
        ],
        "low_value": [
            "shares",
            "NAV",
            "offer",
            "board",
        ],
    },
    # Incentive fee structure (enhanced from N-2 document analysis)
    "incentive_fee": {
        "high_value": [
            # Standard terms
            "incentive fee",
            "performance fee",
            "incentive allocation",
            "performance allocation",
            "carried interest",
            "promote",
            # Specific percentages from N-2s
            "10% of profits",
            "10% of the excess",
            "12.5% of",
            "15% of profits",
            "20% of profits",
            # Blackstone-specific
            "Pre-Incentive Fee Net Investment Income",
            "quarterly in arrears",
            # Hamilton Lane specific
            "net profits of the Fund",
            # Catch-up terms
            "full catch-up",
            "catch-up provision",
            "with a catch-up",
        ],
        "medium_value": [
            "incentive",
            "performance-based",
            "profit share",
            "profit allocation",
            "above hurdle",
            "exceeds benchmark",
            "outperformance",
            "annualized hurdle",
            "5% annualized",
            "8% annualized",
            "subject to a",
            "hurdle rate",
        ],
        "low_value": [
            "performance",
            "profits",
            "hurdle",
            "adviser",
            "advisor",
        ],
    },
    # Hurdle rate
    "hurdle_rate": {
        "high_value": [
            "hurdle rate",
            "preferred return",
            "benchmark rate",
            "threshold return",
            "minimum return",
            "8% hurdle",
            "6% hurdle",
            "5% hurdle",
            "5% annualized hurdle",
            "1.25% quarterly",
            # Quarterly hurdle patterns (Blue Owl style)
            "1.50% per quarter",
            "1.5% per quarter",
            "per calendar quarter",
            "quarterly hurdle",
            "computed quarterly",
            "measured quarterly",
            "6.00% annualized",
        ],
        "medium_value": [
            "hurdle",
            "preferred",
            "threshold",
            "before incentive",
            "must exceed",
            "SOFR",
            "treasury",
            "T-bill",
            "annualized",
            "quarterly",
            "per quarter",
        ],
        "low_value": [
            "return",
            "rate",
            "benchmark",
        ],
    },
    # =============================================================================
    # NEW FIELDS FOR TIER 3 COMPLETE COVERAGE
    # =============================================================================
    # Allocation targets / investment strategy
    # ENHANCED: More fund-of-funds specific keywords for large docs (StepStone)
    "allocation_targets": {
        "high_value": [
            # Standard allocation terms
            "target allocation",
            "asset allocation",
            "allocation target",
            "investment allocation",
            "portfolio allocation",
            "% of assets",
            "% of the portfolio",
            "% of net assets",
            "private equity",
            "private credit",
            "real estate",
            "infrastructure",
            # Fund-of-funds specific ranges (common in large docs)
            "40% to 70%",  # Secondary funds range
            "20% to 50%",  # Direct investments range
            "0% to 60%",  # Common allocation range
            "0% to 40%",
            "10% to 30%",
            # Fund-of-funds investment type descriptions
            "investment types",
            "secondary transactions",
            "secondary market",
            "primary fund investments",
            "at least 80%",  # Common allocation threshold
            # NEW: Fund-of-funds specific sections
            "Investment Policies",
            "Investment Program",
            "Types of Investments",
            "Principal Investment Strategies",
            "Investment Strategies and Policies",
            # NEW: Secondary investment terms
            "secondary fund interests",
            "secondary market transactions",
            "secondaries transactions",
            "secondary purchases",
            "secondary LP interests",
            # NEW: Direct/co-investment terms
            "direct co-investments",
            "co-investment opportunities",
            "alongside GPs",
            "alongside general partners",
        ],
        "medium_value": [
            "allocation",
            "diversified",
            "investment strategy",
            "investment objective",
            "primary investments",
            "secondary investments",
            "co-investments",
            "fund-of-funds",
            "direct investments",
            # Additional for large docs
            "Investment Funds",
            "portfolio companies",
            "primary funds",
            "secondaries",
            # NEW: Target ranges and percentages
            "normally invest",
            "will invest",
            "intends to invest",
            "expect to allocate",
            "typically allocate",
            "target range",
            "allocation range",
            "guideline range",
        ],
        "low_value": [
            "invest",
            "portfolio",
            "assets",
            "strategy",
        ],
    },
    # Concentration limits / investment restrictions
    # ENHANCED: More fund-of-funds specific keywords for large docs (StepStone)
    "concentration_limits": {
        "high_value": [
            # Standard concentration terms
            "concentration limit",
            "investment restriction",
            "fundamental policy",
            "non-fundamental policy",
            "may not invest more than",
            "will not invest more than",
            "no more than 25%",
            "no more than 15%",
            "no more than 20%",
            "no more than 10%",
            "single issuer",
            "single industry",
            # Fund-of-funds specific
            "25% of its total assets",
            "25% of the Fund",
            "single Investment Fund",
            "single portfolio fund",
            "any single",
            # NEW: Investment policies section headers
            "Investment Restrictions",
            "Investment Limitations",
            "Operating Policies",
            "Fundamental Policies",
            "Non-Fundamental Policies",
            # NEW: Fund-of-funds concentration terms
            "maximum commitment",
            "maximum allocation",
            "commitment limit",
            "single manager",
            "single GP",
            "single strategy",
            "sector concentration",
            "geographic concentration",
            # NEW: Specific percentage patterns
            "will not exceed",
            "shall not exceed",
            "not to exceed",
            "maximum of 25%",
            "maximum of 20%",
            "maximum of 15%",
        ],
        "medium_value": [
            "concentration",
            "restriction",
            "limitation",
            "diversification",
            "% of total assets",
            "% of net assets",
            "fundamental",
            "non-fundamental",
            # Additional for fund-of-funds
            "Investment Fund",
            "underlying fund",
            "portfolio fund",
            # NEW: Additional diversification terms
            "diversified fund",
            "non-diversified",
            "prudent diversification",
            "investment limits",
            "position limits",
        ],
        "low_value": [
            "limit",
            "policy",
            "invest",
            "assets",
        ],
    },
    # Fund metadata
    "fund_metadata": {
        "high_value": [
            "fund name",
            "investment manager",
            "investment adviser",
            "investment advisor",
            "fiscal year end",
            "interval fund",
            "tender offer fund",
            "closed-end fund",
            "registered under",
        ],
        "medium_value": [
            "adviser",
            "advisor",
            "manager",
            "sponsor",
            "fiscal year",
            "december 31",
            "march 31",
            "june 30",
            "1940 act",
        ],
        "low_value": [
            "fund",
            "management",
            "registered",
        ],
    },
}


# =============================================================================
# XBRL-FIRST SEARCH CONFIGURATION
# =============================================================================

# XBRL sections have authoritative boundaries but often don't contain the data
# we need (e.g., repurchase terms are in HTML sections, not XBRL tables).
# Experiment 2026-01-11: XBRL_SECTION_BONUS = 15 gave 40.5% accuracy
#                        XBRL_SECTION_BONUS = 0  gave 57.9% accuracy (+17.4%)
# Keep at 0 to let keyword relevance drive section selection.
XBRL_SECTION_BONUS = 0

# Penalty for oversized sections (likely broken or too noisy)
# Sections over this threshold get a score penalty
MAX_REASONABLE_SECTION_CHARS = 50000
OVERSIZED_SECTION_PENALTY = 10


@dataclass
class ScoredSection:
    """A section with its keyword relevance score."""
    section: ChunkedSection
    score: int
    matching_keywords: list[str]
    is_xbrl: bool = False


@dataclass
class ScopedExtractionResult:
    """Result from scoped agentic extraction."""
    field_name: str
    value: Optional[dict]
    source_section: str
    confidence: str
    chunks_searched: int
    sections_searched: int
    # Reranker metrics (populated when reranker is enabled)
    reranker_enabled: bool = False
    reranker_chunks_input: int = 0  # Chunks sent to reranker
    reranker_chunks_output: int = 0  # Chunks returned after threshold
    reranker_top_scores: list = field(default_factory=list)  # Top 3 relevance scores


# =============================================================================
# KEYWORD SCORING
# =============================================================================

def score_section_for_field(
    section: ChunkedSection,
    field_name: str,
) -> ScoredSection:
    """
    Score a section's relevance to a field based on keyword matches.

    Implements XBRL-first search: XBRL sections get a significant bonus
    because they have authoritative boundaries and structured content.

    Also applies NEGATIVE keyword penalties for sections containing:
    - Definitional language ("is defined as", "refers to")
    - Glossary/definition sections
    - Risk explanations ("is the risk that")
    - Hypothetical examples
    - Underlying fund references (for fund-level fields)

    Args:
        section: The chunked section to score
        field_name: The field we're trying to extract

    Returns:
        ScoredSection with score and matching keywords
    """
    # Extract root field name for keyword lookup (e.g., "allocation_targets.secondary_funds_min_pct" -> "allocation_targets")
    root_field = field_name.split('.')[0] if '.' in field_name else field_name
    keywords_config = FIELD_KEYWORDS.get(root_field, FIELD_KEYWORDS.get("share_classes"))

    # Combine all chunk content for scoring
    full_text = " ".join(chunk.content for chunk in section.chunks).lower()

    score = 0
    matching_keywords = []
    is_xbrl = section.section_type == SectionType.XBRL_TEXT_BLOCK

    # XBRL-FIRST: Give XBRL sections a significant bonus
    # XBRL sections have authoritative boundaries and are more reliable
    if is_xbrl:
        score += XBRL_SECTION_BONUS
        matching_keywords.append(f"xbrl_section (+{XBRL_SECTION_BONUS})")

    # Score high-value keywords (3 points)
    for kw in keywords_config.get("high_value", []):
        if kw.lower() in full_text:
            score += 3
            matching_keywords.append(f"{kw} (+3)")

    # Score medium-value keywords (2 points)
    for kw in keywords_config.get("medium_value", []):
        if kw.lower() in full_text:
            score += 2
            matching_keywords.append(f"{kw} (+2)")

    # Score low-value keywords (1 point)
    for kw in keywords_config.get("low_value", []):
        if kw.lower() in full_text:
            score += 1
            matching_keywords.append(f"{kw} (+1)")

    # =========================================================================
    # NEGATIVE KEYWORD PENALTIES
    # Penalize sections with definitional/non-data language
    # =========================================================================

    # Apply GLOBAL negative keywords (apply to all fields)
    for category, penalties in GLOBAL_NEGATIVE_KEYWORDS.items():
        # High penalty keywords (-4 points)
        for kw in penalties.get("high_penalty", []):
            if kw.lower() in full_text:
                score -= NEGATIVE_PENALTY_HIGH
                matching_keywords.append(f"NEG:{kw} (-{NEGATIVE_PENALTY_HIGH})")

        # Medium penalty keywords (-2 points)
        for kw in penalties.get("medium_penalty", []):
            if kw.lower() in full_text:
                score -= NEGATIVE_PENALTY_MEDIUM
                matching_keywords.append(f"NEG:{kw} (-{NEGATIVE_PENALTY_MEDIUM})")

    # Apply FIELD-SPECIFIC negative keywords
    field_negatives = FIELD_NEGATIVE_KEYWORDS.get(field_name, {})
    if field_negatives:
        # High penalty keywords (-4 points)
        for kw in field_negatives.get("high_penalty", []):
            if kw.lower() in full_text:
                score -= NEGATIVE_PENALTY_HIGH
                matching_keywords.append(f"NEG:{kw} (-{NEGATIVE_PENALTY_HIGH})")

        # Medium penalty keywords (-2 points)
        for kw in field_negatives.get("medium_penalty", []):
            if kw.lower() in full_text:
                score -= NEGATIVE_PENALTY_MEDIUM
                matching_keywords.append(f"NEG:{kw} (-{NEGATIVE_PENALTY_MEDIUM})")

    # Bonus: if section title contains relevant terms
    title_lower = section.section_title.lower()
    if "distribution" in title_lower or "share" in title_lower:
        score += 5
        matching_keywords.append("title_bonus (+5)")

    # TITLE-BASED NEGATIVE PENALTY
    # Penalize sections with definitional/risk titles
    negative_title_patterns = [
        "risk",
        "glossary",
        "definitions",
        "tax",
        "erisa",
        "legal",
    ]
    for pattern in negative_title_patterns:
        if pattern in title_lower:
            score -= NEGATIVE_PENALTY_MEDIUM
            matching_keywords.append(f"NEG_TITLE:{pattern} (-{NEGATIVE_PENALTY_MEDIUM})")
            break  # Only apply once

    # Penalty for oversized sections (likely noisy or broken)
    section_chars = sum(chunk.char_count for chunk in section.chunks)
    if section_chars > MAX_REASONABLE_SECTION_CHARS:
        score -= OVERSIZED_SECTION_PENALTY
        matching_keywords.append(f"oversized_penalty (-{OVERSIZED_SECTION_PENALTY})")

    return ScoredSection(
        section=section,
        score=score,
        matching_keywords=matching_keywords,
        is_xbrl=is_xbrl,
    )


def select_top_sections(
    chunked_doc: ChunkedDocument,
    field_name: str,
    top_k: int = 5,
) -> list[ScoredSection]:
    """
    Select top K sections most likely to contain the target field.

    Implements XBRL-first search: XBRL sections are prioritized due to
    their authoritative boundaries. If XBRL sections are insufficient,
    HTML sections are used as fallback.

    Args:
        chunked_doc: The chunked document
        field_name: Field we're trying to extract
        top_k: Number of top sections to return

    Returns:
        List of top K scored sections, sorted by score descending
    """
    scored_sections = []

    for section in chunked_doc.chunked_sections:
        scored = score_section_for_field(section, field_name)
        if scored.score > 0:  # Only include sections with some relevance
            scored_sections.append(scored)

    # Sort by score descending
    scored_sections.sort(key=lambda x: x.score, reverse=True)

    selected = scored_sections[:top_k]

    # Log XBRL vs HTML selection for debugging
    if selected:
        xbrl_count = sum(1 for s in selected if s.is_xbrl)
        html_count = len(selected) - xbrl_count
        logger.debug(
            f"select_top_sections({field_name}): {xbrl_count} XBRL, {html_count} HTML sections selected"
        )
        for i, s in enumerate(selected[:3]):  # Log top 3
            section_type = "XBRL" if s.is_xbrl else "HTML"
            logger.debug(
                f"  #{i+1}: [{section_type}] {s.section.section_title[:40]} (score={s.score})"
            )

    return selected


# =============================================================================
# SCOPED EXTRACTION
# =============================================================================

def find_relevant_chunks(
    section: ChunkedSection,
    field_name: str,
    max_chunks: int = 10,
    min_chunks_fallback: int = 3,
) -> list[Chunk]:
    """
    Find the most relevant chunks within a section for the target field.

    Uses keyword matching to find chunks most likely to contain the value.
    If fewer than min_chunks_fallback have keyword matches, includes top chunks
    by position to ensure the reranker/LLM has content to work with.
    """
    # Extract root field name for keyword lookup (e.g., "allocation_targets.secondary_funds_min_pct" -> "allocation_targets")
    root_field = field_name.split('.')[0] if '.' in field_name else field_name
    keywords_config = FIELD_KEYWORDS.get(root_field, FIELD_KEYWORDS.get("share_classes"))
    all_keywords = (
        keywords_config.get("high_value", []) +
        keywords_config.get("medium_value", [])
    )

    # Score each chunk - include ALL chunks, not just those with matches
    chunk_scores = []
    for chunk in section.chunks:
        content_lower = chunk.content.lower()
        score = sum(1 for kw in all_keywords if kw.lower() in content_lower)
        chunk_scores.append((chunk, score))

    # Sort by score (highest first), then by position (earlier first for ties)
    chunk_scores.sort(key=lambda x: (-x[1], section.chunks.index(x[0])))

    # Get chunks with keyword matches
    matched_chunks = [(c, s) for c, s in chunk_scores if s > 0]

    # If we have enough keyword matches, use them
    if len(matched_chunks) >= min_chunks_fallback:
        return [chunk for chunk, _ in matched_chunks[:max_chunks]]

    # Otherwise, include top chunks even without keyword matches
    # This ensures the reranker/LLM has content to evaluate
    logger.debug(
        f"    [find_relevant_chunks] Only {len(matched_chunks)} keyword matches for {field_name}, "
        f"including {min_chunks_fallback} chunks as fallback"
    )
    return [chunk for chunk, _ in chunk_scores[:max(max_chunks, min_chunks_fallback)]]


def extract_minimum_investment_from_chunks(
    chunks: list[Chunk],
    client,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    rate_limit: Optional[RateLimitConfig] = None,
) -> Optional[dict]:
    """
    Extract minimum investment values from focused chunks.

    Returns dict with class_name -> minimum_investment mapping.
    """
    if not chunks:
        return None

    # Combine chunk content
    combined_text = "\n\n".join([
        f"[From: {chunk.section_title}]\n{chunk.content}"
        for chunk in chunks
    ])

    # Limit size
    if len(combined_text) > 15000:
        combined_text = combined_text[:15000]

    prompt = """Extract the MINIMUM INITIAL INVESTMENT amounts for each share class.

Look for patterns like:
- "minimum initial investment for Class S is $X"
- "minimum investment of $X for Class I"
- "$2,500" or "$1,000,000" mentioned near class names

Return a JSON object with this structure:
{
    "class_s_minimum": <number or null>,
    "class_d_minimum": <number or null>,
    "class_i_minimum": <number or null>,
    "class_i_advisory_minimum": <number or null>,
    "evidence": "<quote from text showing the values>"
}

TEXT:
""" + combined_text

    messages = [
        {"role": "system", "content": "You are a precise financial document analyst. Extract exact values from the text."},
        {"role": "user", "content": prompt},
    ]

    try:
        result = call_llm_json(
            client=client,
            provider=provider,
            model=model,
            messages=messages,
            rate_limit=rate_limit,
        )
        return result

    except Exception as e:
        logger.error(f"Scoped extraction failed: {e}")
        return None


def extract_lock_up_from_chunks(
    chunks: list[Chunk],
    client,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    rate_limit: Optional[RateLimitConfig] = None,
) -> Optional[dict]:
    """
    Extract lock-up period and early redemption fee from focused chunks.

    Returns dict with lock-up details.
    """
    if not chunks:
        return None

    # Combine chunk content
    combined_text = "\n\n".join([
        f"[From: {chunk.section_title}]\n{chunk.content}"
        for chunk in chunks
    ])

    # Limit size
    if len(combined_text) > 15000:
        combined_text = combined_text[:15000]

    prompt = """Extract LOCK-UP PERIOD and EARLY REDEMPTION FEE information.

Look for patterns like:
- "lock-up period of 12 months" or "lock-up period of one year"
- "early repurchase fee of 2%"
- "early withdrawal charge of 2%"
- "2% fee for shares held less than one year"
- "shares must be held for at least one year"

Return a JSON object with this structure:
{
    "lock_up_period_years": <number or null>,  // e.g., 1 for "one year" or "12 months"
    "early_repurchase_fee_pct": <number or null>,  // e.g., 2 for "2%"
    "early_repurchase_fee_period": "<string or null>",  // e.g., "within 1 year of purchase"
    "evidence": "<quote from text showing the values>"
}

TEXT:
""" + combined_text

    messages = [
        {"role": "system", "content": "You are a precise financial document analyst. Extract exact values from the text."},
        {"role": "user", "content": prompt},
    ]

    try:
        result = call_llm_json(
            client=client,
            provider=provider,
            model=model,
            messages=messages,
            rate_limit=rate_limit,
        )
        return result

    except Exception as e:
        logger.error(f"Lock-up extraction failed: {e}")
        return None


def extract_leverage_from_chunks(
    chunks: list[Chunk],
    client,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    rate_limit: Optional[RateLimitConfig] = None,
) -> Optional[dict]:
    """
    Extract leverage limits from focused chunks.

    Returns dict with leverage details.
    """
    if not chunks:
        return None

    # Combine chunk content
    combined_text = "\n\n".join([
        f"[From: {chunk.section_title}]\n{chunk.content}"
        for chunk in chunks
    ])

    # Limit size
    if len(combined_text) > 15000:
        combined_text = combined_text[:15000]

    prompt = """Extract LEVERAGE LIMITS and BORROWING information.

Look for patterns like:
- "may borrow up to 33-1/3% of total assets"
- "leverage will not exceed 50%"
- "debt-to-equity ratio of 0.5:1" (means 50%)
- "maintain an asset coverage ratio of at least 300%" (means can borrow 33%)
- "credit facility of $X million"

Note on interpretation:
- "33-1/3% of total assets" → max_leverage_pct = 33
- "debt-to-equity ratio of 0.5:1" → max_leverage_pct = 50
- "asset coverage ratio of 300%" → max_leverage_pct = 33

Return a JSON object with this structure:
{
    "uses_leverage": <true/false>,
    "max_leverage_pct": <number or null>,  // as percentage (e.g., 33, 50)
    "leverage_basis": "<string or null>",  // "total_assets", "net_assets", "debt_to_equity"
    "credit_facility_size": <number in millions or null>,
    "evidence": "<quote from text showing the values>"
}

TEXT:
""" + combined_text

    messages = [
        {"role": "system", "content": "You are a precise financial document analyst. Extract exact values from the text."},
        {"role": "user", "content": prompt},
    ]

    try:
        result = call_llm_json(
            client=client,
            provider=provider,
            model=model,
            messages=messages,
            rate_limit=rate_limit,
        )
        return result

    except Exception as e:
        logger.error(f"Leverage extraction failed: {e}")
        return None


def extract_distribution_terms_from_chunks(
    chunks: list[Chunk],
    client,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    rate_limit: Optional[RateLimitConfig] = None,
) -> Optional[dict]:
    """
    Extract distribution/dividend policy from focused chunks.

    Returns dict with distribution details.
    """
    if not chunks:
        return None

    # Combine chunk content
    combined_text = "\n\n".join([
        f"[From: {chunk.section_title}]\n{chunk.content}"
        for chunk in chunks
    ])

    # Limit size
    if len(combined_text) > 15000:
        combined_text = combined_text[:15000]

    prompt = """Extract DISTRIBUTION/DIVIDEND POLICY information.

Look for patterns like:
- "distributions will be paid quarterly/monthly/annually"
- "unless shareholders elect otherwise, distributions will be reinvested"
- "dividend reinvestment plan (DRIP)"
- "distributions are reinvested by default"
- "target annual distribution of X%"

Return a JSON object with this structure:
{
    "distribution_frequency": "<string or null>",  // "monthly", "quarterly", "annual"
    "default_distribution_policy": "<string or null>",  // "cash", "reinvested", "DRIP"
    "target_distribution_rate": <number or null>,  // as percentage if stated
    "evidence": "<quote from text showing the values>"
}

IMPORTANT: Return null (not strings like "not_found") for any field not found.

TEXT:
""" + combined_text

    messages = [
        {"role": "system", "content": "You are a precise financial document analyst. Extract exact values from the text."},
        {"role": "user", "content": prompt},
    ]

    try:
        result = call_llm_json(
            client=client,
            provider=provider,
            model=model,
            messages=messages,
            rate_limit=rate_limit,
        )
        return result

    except Exception as e:
        logger.error(f"Distribution terms extraction failed: {e}")
        return None


def extract_minimum_additional_from_chunks(
    chunks: list[Chunk],
    client,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    rate_limit: Optional[RateLimitConfig] = None,
) -> Optional[dict]:
    """
    Extract minimum additional/subsequent investment amounts from focused chunks.

    Returns dict with class_name -> minimum_additional_investment mapping.
    """
    if not chunks:
        return None

    # Combine chunk content
    combined_text = "\n\n".join([
        f"[From: {chunk.section_title}]\n{chunk.content}"
        for chunk in chunks
    ])

    # Limit size
    if len(combined_text) > 15000:
        combined_text = combined_text[:15000]

    prompt = """Extract the MINIMUM ADDITIONAL/SUBSEQUENT INVESTMENT amounts for each share class.

CRITICAL: Look for BOTH universal statements AND class-specific statements:

1. UNIVERSAL statements apply to ALL share classes:
   - "minimum subsequent investment in our Common Shares is $X" → applies to ALL classes
   - "minimum subsequent investment of $X per transaction" → applies to ALL classes
   - "subsequent purchases of $X" (without class name) → applies to ALL classes

2. CLASS-SPECIFIC statements override universal:
   - "Class I minimum subsequent investment is $X" → only for Class I
   - "$X for Class S additional purchases" → only for Class S

If you find a universal statement like "minimum subsequent investment is $500",
apply that $500 to ALL four share classes unless a class-specific override exists.

Look for patterns like:
- "minimum subsequent investment in our Common Shares is $X"
- "minimum subsequent investment of $X per transaction"
- "minimum additional investment of $X"
- "additional purchases of at least $X"
- "$500 minimum subsequent purchase"

Return a JSON object with this structure:
{
    "class_s_minimum_additional": <number or null>,
    "class_d_minimum_additional": <number or null>,
    "class_i_minimum_additional": <number or null>,
    "class_i_advisory_minimum_additional": <number or null>,
    "universal_minimum_additional": <number or null if a universal value was found>,
    "evidence": "<quote from text showing the values>"
}

IMPORTANT:
- Return null (not strings) for any field not found
- If universal_minimum_additional is found, fill in ALL class fields with that value

TEXT:
""" + combined_text

    messages = [
        {"role": "system", "content": "You are a precise financial document analyst. Extract exact values from the text."},
        {"role": "user", "content": prompt},
    ]

    try:
        result = call_llm_json(
            client=client,
            provider=provider,
            model=model,
            messages=messages,
            rate_limit=rate_limit,
        )
        return result

    except Exception as e:
        logger.error(f"Minimum additional extraction failed: {e}")
        return None


def extract_repurchase_basis_from_chunks(
    chunks: list[Chunk],
    client,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    rate_limit: Optional[RateLimitConfig] = None,
) -> Optional[dict]:
    """
    Extract repurchase basis (shares vs NAV) from focused chunks.

    Returns dict with repurchase basis details.
    """
    if not chunks:
        return None

    # Combine chunk content
    combined_text = "\n\n".join([
        f"[From: {chunk.section_title}]\n{chunk.content}"
        for chunk in chunks
    ])

    # Limit size
    if len(combined_text) > 15000:
        combined_text = combined_text[:15000]

    prompt = """Extract the REPURCHASE BASIS - whether repurchases are based on number of shares or net assets/NAV.

Look for patterns like:
- "repurchase 5% of outstanding shares" → basis is "number_of_shares"
- "repurchase 5% of net assets" → basis is "net_assets"
- "repurchase at NAV" → basis is "NAV"
- "percentage of the Fund's shares" → basis is "number_of_shares"

Return a JSON object with this structure:
{
    "repurchase_basis": "<string or null>",  // "number_of_shares", "net_assets", "NAV"
    "evidence": "<quote from text showing the basis>"
}

TEXT:
""" + combined_text

    messages = [
        {"role": "system", "content": "You are a precise financial document analyst. Extract exact values from the text."},
        {"role": "user", "content": prompt},
    ]

    try:
        result = call_llm_json(
            client=client,
            provider=provider,
            model=model,
            messages=messages,
            rate_limit=rate_limit,
        )
        return result

    except Exception as e:
        logger.error(f"Repurchase basis extraction failed: {e}")
        return None


def extract_incentive_fee_from_chunks(
    chunks: list[Chunk],
    client,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    rate_limit: Optional[RateLimitConfig] = None,
) -> Optional[dict]:
    """
    Extract incentive fee structure from focused chunks.
    """
    if not chunks:
        return None

    combined_text = "\n\n".join([
        f"[From: {chunk.section_title}]\n{chunk.content}"
        for chunk in chunks
    ])

    if len(combined_text) > 20000:
        combined_text = combined_text[:20000]

    prompt = """Extract INCENTIVE FEE / PERFORMANCE FEE information.

Look for:
- Incentive fee percentage (e.g., "10%", "12.5%", "20%")
- Hurdle rate / preferred return (e.g., "5% annualized", "1.25% quarterly", "8%")
- Catch-up provisions
- High water mark provisions
- Fee basis (net profits, net investment income, NAV appreciation)
- Crystallization frequency (quarterly, annual)

Return JSON:
{
    "has_incentive_fee": <true/false>,
    "incentive_fee_pct": <number as string or null>,
    "hurdle_rate_as_stated": <the hurdle rate NUMBER as stated in the document, e.g. "5" for "5% annualized" or "1.25" for "1.25% quarterly">,
    "hurdle_rate_frequency": <"quarterly"/"annual" - the frequency of the stated rate>,
    "high_water_mark": <true/false/null>,
    "has_catch_up": <true/false/null>,
    "fee_basis": <"net_investment_income"/"net_profits"/"nav_appreciation" or null>,
    "crystallization_frequency": <"quarterly"/"annual" or null>,
    "evidence": "<quote showing fee structure>"
}

EXAMPLES:
- "5% annualized hurdle": hurdle_rate_as_stated="5", hurdle_rate_frequency="annual"
- "1.25% quarterly hurdle": hurdle_rate_as_stated="1.25", hurdle_rate_frequency="quarterly"
- "subject to a hurdle rate of 1.5% per quarter (6% annualized)": hurdle_rate_as_stated="1.5", hurdle_rate_frequency="quarterly"

TEXT:
""" + combined_text

    messages = [
        {"role": "system", "content": "You are a precise financial document analyst. Extract exact fee structures from the text. Pay attention to both fund-level fees and underlying fund fees."},
        {"role": "user", "content": prompt},
    ]

    try:
        result = call_llm_json(
            client=client,
            provider=provider,
            model=model,
            messages=messages,
            rate_limit=rate_limit,
        )
        return result
    except Exception as e:
        logger.error(f"Incentive fee extraction failed: {e}")
        return None


def extract_expense_cap_from_chunks(
    chunks: list[Chunk],
    client,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    rate_limit: Optional[RateLimitConfig] = None,
) -> Optional[dict]:
    """
    Extract expense cap / fee waiver information from focused chunks.
    """
    if not chunks:
        return None

    combined_text = "\n\n".join([
        f"[From: {chunk.section_title}]\n{chunk.content}"
        for chunk in chunks
    ])

    if len(combined_text) > 15000:
        combined_text = combined_text[:15000]

    prompt = """Extract EXPENSE CAP / FEE WAIVER information.

Look for:
- Expense limitation/cap percentage
- Fee waiver agreements
- Expense reimbursement arrangements
- Cap expiration date

Return JSON:
{
    "has_expense_cap": <true/false>,
    "expense_cap_pct": <number as string or null>,
    "cap_expiration": <date string or null>,
    "cap_type": <"contractual"/"voluntary" or null>,
    "evidence": "<quote showing expense cap>"
}

TEXT:
""" + combined_text

    messages = [
        {"role": "system", "content": "You are a precise financial document analyst. Extract exact values from the text."},
        {"role": "user", "content": prompt},
    ]

    try:
        result = call_llm_json(
            client=client,
            provider=provider,
            model=model,
            messages=messages,
            rate_limit=rate_limit,
        )
        return result
    except Exception as e:
        logger.error(f"Expense cap extraction failed: {e}")
        return None


def extract_repurchase_terms_from_chunks(
    chunks: list[Chunk],
    client,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    rate_limit: Optional[RateLimitConfig] = None,
) -> Optional[dict]:
    """
    Extract repurchase/tender offer terms from focused chunks.
    """
    if not chunks:
        return None

    combined_text = "\n\n".join([
        f"[From: {chunk.section_title}]\n{chunk.content}"
        for chunk in chunks
    ])

    if len(combined_text) > 15000:
        combined_text = combined_text[:15000]

    prompt = """Extract REPURCHASE / TENDER OFFER terms.

Look for:
- Repurchase frequency (quarterly, semi-annual, annual)
- Repurchase amount percentage (e.g., "5% to 25%", "at least 5%")
- Whether basis is shares outstanding or NAV
- Lock-up period
- Early repurchase fee

Return JSON:
{
    "repurchase_frequency": <"quarterly"/"semi-annual"/"annual" or null>,
    "repurchase_amount_pct": <minimum percentage as string or null>,
    "repurchase_max_pct": <maximum percentage as string or null>,
    "repurchase_basis": <"number_of_shares"/"nav" or null>,
    "lock_up_period_years": <number or null>,
    "early_repurchase_fee_pct": <number as string or null>,
    "evidence": "<quote showing repurchase terms>"
}

TEXT:
""" + combined_text

    messages = [
        {"role": "system", "content": "You are a precise financial document analyst. Extract exact values from the text."},
        {"role": "user", "content": prompt},
    ]

    try:
        result = call_llm_json(
            client=client,
            provider=provider,
            model=model,
            messages=messages,
            rate_limit=rate_limit,
        )
        return result
    except Exception as e:
        logger.error(f"Repurchase terms extraction failed: {e}")
        return None


def normalize_percentage_value(value) -> Optional[str]:
    """
    Normalize a percentage value by stripping % sign.

    Post-processing to ensure consistent format for evaluation.
    Examples:
      - "0.75%" -> "0.75"
      - "3.5" -> "3.5"
      - 0.75 -> "0.75"
      - null/None -> None
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        # Strip % sign and whitespace
        cleaned = value.strip().rstrip('%').strip()
        if cleaned == "" or cleaned.lower() in ("null", "none", "n/a"):
            return None
        return cleaned
    return str(value)


def post_process_share_classes(result: Optional[dict]) -> Optional[dict]:
    """
    Post-process share_classes extraction result to normalize formats.

    - Strips % from percentage values
    - Ensures consistent data types
    """
    if not result:
        return result

    share_classes = result.get("share_classes", [])
    if not isinstance(share_classes, list):
        return result

    for sc in share_classes:
        if not isinstance(sc, dict):
            continue
        # Normalize percentage fields
        for pct_field in ["sales_load_pct", "distribution_fee_pct", "distribution_servicing_fee_pct"]:
            if pct_field in sc:
                sc[pct_field] = normalize_percentage_value(sc[pct_field])

    return result


def extract_share_classes_from_chunks(
    chunks: list[Chunk],
    client,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    rate_limit: Optional[RateLimitConfig] = None,
) -> Optional[dict]:
    """
    Extract share class information from focused chunks.
    """
    if not chunks:
        return None

    combined_text = "\n\n".join([
        f"[From: {chunk.section_title}]\n{chunk.content}"
        for chunk in chunks
    ])

    if len(combined_text) > 20000:
        combined_text = combined_text[:20000]

    prompt = """Extract SHARE CLASS information.

Look for EACH share class (Class S, Class D, Class I, Class I Advisory, etc.) and extract ALL of:
1. Minimum initial investment (dollar amount, e.g., $2,500 or $1,000,000)
2. Minimum additional/subsequent investment (dollar amount, often $500)
3. Sales load / front-end load / placement fee cap (percentage)
4. Distribution/servicing fee (12b-1 fee, percentage)

CRITICAL - WHERE TO FIND EACH VALUE:
- Minimum initial investment: Look for "minimum initial investment of $X" or "minimum investment for Class X is $X"
  - Class S/D often have lower minimums like $2,500
  - Class I often has higher minimums like $1,000,000
- Minimum additional investment: Look for "minimum subsequent investment" or "minimum additional purchase" - often $500
- Sales load: Look for "upfront placement fee", "brokerage commission", or percentage caps for intermediaries
- Distribution fee: Look for "distribution fee", "servicing fee", "12b-1 fee", or "distribution and/or service fee"

CRITICAL - NULL vs 0 for FEES:
- Return "0" (string zero) if the fee is explicitly NOT charged:
  - "Class I is not subject to any distribution fee" -> distribution_fee_pct = "0"
  - "No sales load" -> sales_load_pct = "0"
  - "Class I shares —" (dash means zero) -> distribution_fee_pct = "0"
- Return null ONLY if the fee is never mentioned for that class

CRITICAL - Sales load interpretation:
- If no direct sales load but intermediaries may charge up to a maximum, use that maximum
- Example: "intermediaries may charge up to 3.5%" -> sales_load_pct = "3.5"

Return JSON (percentages as strings WITHOUT % sign):
{
    "share_classes": [
        {
            "class_name": "<name>",
            "minimum_initial_investment": <number or null>,
            "minimum_additional_investment": <number or null>,
            "sales_load_pct": "<number as string or null>",
            "distribution_fee_pct": "<number as string or null>"
        }
    ],
    "evidence": "<quote showing share class details>"
}

TEXT:
""" + combined_text

    messages = [
        {"role": "system", "content": "You are a precise financial document analyst. Extract exact values for each share class. Return percentages as plain numbers without the % symbol."},
        {"role": "user", "content": prompt},
    ]

    try:
        result = call_llm_json(
            client=client,
            provider=provider,
            model=model,
            messages=messages,
            rate_limit=rate_limit,
        )
        # Post-process to normalize percentage format
        return post_process_share_classes(result)
    except Exception as e:
        logger.error(f"Share classes extraction failed: {e}")
        return None


def extract_allocation_targets_from_chunks(
    chunks: list[Chunk],
    client,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    rate_limit: Optional[RateLimitConfig] = None,
) -> Optional[dict]:
    """
    Extract allocation targets / investment strategy from focused chunks.
    """
    if not chunks:
        return None

    combined_text = "\n\n".join([
        f"[From: {chunk.section_title}]\n{chunk.content}"
        for chunk in chunks
    ])

    if len(combined_text) > 15000:
        combined_text = combined_text[:15000]

    prompt = """Extract ALLOCATION TARGETS / INVESTMENT STRATEGY.

Look for:
- Target allocations by asset class (private equity, private credit, real estate, etc.)
- Primary vs secondary investments
- Co-investments allocation
- Geographic allocation targets

Return JSON:
{
    "allocations": [
        {
            "asset_class": "<name>",
            "target_pct": <number or null>,
            "range_min_pct": <number or null>,
            "range_max_pct": <number or null>
        }
    ],
    "strategy_description": "<brief description>",
    "evidence": "<quote showing allocation targets>"
}

TEXT:
""" + combined_text

    messages = [
        {"role": "system", "content": "You are a precise financial document analyst. Extract allocation targets and ranges."},
        {"role": "user", "content": prompt},
    ]

    try:
        result = call_llm_json(
            client=client,
            provider=provider,
            model=model,
            messages=messages,
            rate_limit=rate_limit,
        )
        return result
    except Exception as e:
        logger.error(f"Allocation targets extraction failed: {e}")
        return None


def extract_concentration_limits_from_chunks(
    chunks: list[Chunk],
    client,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    rate_limit: Optional[RateLimitConfig] = None,
) -> Optional[dict]:
    """
    Extract concentration limits / investment restrictions from focused chunks.
    """
    if not chunks:
        return None

    combined_text = "\n\n".join([
        f"[From: {chunk.section_title}]\n{chunk.content}"
        for chunk in chunks
    ])

    if len(combined_text) > 15000:
        combined_text = combined_text[:15000]

    prompt = """Extract CONCENTRATION LIMITS / INVESTMENT RESTRICTIONS.

Look for:
- Maximum investment in single issuer
- Maximum investment in single industry
- Fundamental vs non-fundamental policies
- Diversification requirements

Return JSON:
{
    "limits": [
        {
            "limit_type": "<description>",
            "max_pct": <number or null>,
            "is_fundamental": <true/false/null>
        }
    ],
    "evidence": "<quote showing concentration limits>"
}

TEXT:
""" + combined_text

    messages = [
        {"role": "system", "content": "You are a precise financial document analyst. Extract concentration limits and restrictions."},
        {"role": "user", "content": prompt},
    ]

    try:
        result = call_llm_json(
            client=client,
            provider=provider,
            model=model,
            messages=messages,
            rate_limit=rate_limit,
        )
        return result
    except Exception as e:
        logger.error(f"Concentration limits extraction failed: {e}")
        return None


# =============================================================================
# MAIN SCOPED AGENTIC FUNCTION
# =============================================================================

def scoped_agentic_extract(
    chunked_doc: ChunkedDocument,
    field_name: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    provider: Optional[str] = None,
    top_k: int = 5,
    max_chunks_per_section: int = 10,
    delay_between_calls: float = 0.0,
    requests_per_minute: Optional[int] = None,
    reranker_config: Optional[RerankerConfig] = None,
) -> ScopedExtractionResult:
    """
    Tier 3: Keyword-guided scoped agentic extraction with optional reranking.

    1. Score all sections by field-specific keywords (first-pass retrieval)
    2. Collect relevant chunks from top sections
    3. Optionally rerank chunks using Cohere Rerank API
    4. Run focused LLM extraction on top chunks
    5. Return best result

    Args:
        chunked_doc: The chunked document
        field_name: Field to extract (e.g., "minimum_investment")
        api_key: API key (provider-specific, or use env vars)
        model: Model to use (e.g., "gpt-4o-mini", "claude-sonnet", "gemini-flash")
        provider: Provider name ("openai", "anthropic", "google"). Auto-detected if None.
        top_k: Number of top sections to search (default: 5)
        max_chunks_per_section: Max chunks to extract from each section
        delay_between_calls: Seconds to wait between API calls (rate limiting)
        requests_per_minute: Max requests per minute (None = no limit)
        reranker_config: Optional RerankerConfig for Cohere reranking

    Returns:
        ScopedExtractionResult with extracted value, metadata, and reranker stats
    """
    logger.info(f"  [Tier 3] Scoped agentic search for: {field_name}")

    # Resolve model and provider
    resolved_model = resolve_model_name(model)
    resolved_provider = provider or detect_provider(model).value

    # Initialize reranker tracking
    reranker_enabled = reranker_config and reranker_config.enabled
    reranker_chunks_input = 0
    reranker_chunks_output = 0
    reranker_top_scores = []

    # Stage A: Score and select top sections using keywords
    top_sections = select_top_sections(chunked_doc, field_name, top_k)

    if not top_sections:
        logger.warning(f"    No relevant sections found for {field_name}")
        return ScopedExtractionResult(
            field_name=field_name,
            value=None,
            source_section="none",
            confidence="not_found",
            chunks_searched=0,
            sections_searched=0,
            reranker_enabled=reranker_enabled,
        )

    logger.info(f"    Top {len(top_sections)} sections by keyword score:")
    for i, scored in enumerate(top_sections):
        logger.info(f"      {i+1}. {scored.section.section_title[:50]} (score: {scored.score})")

    # Stage B: Collect relevant chunks from top sections using keywords
    all_relevant_chunks = []
    for scored in top_sections:
        relevant_chunks = find_relevant_chunks(
            scored.section,
            field_name,
            max_chunks_per_section,
        )
        all_relevant_chunks.extend(relevant_chunks)

    logger.info(f"    Found {len(all_relevant_chunks)} relevant chunks across {len(top_sections)} sections")

    # Stage B2: Optional reranking step
    chunks_for_extraction = all_relevant_chunks
    if reranker_config and reranker_config.enabled:
        reranker = CohereReranker(reranker_config)
        reranked = reranker.rerank_chunks(all_relevant_chunks, field_name)

        # Track reranker metrics
        reranker_chunks_input = min(len(all_relevant_chunks), reranker_config.first_pass_n)
        reranker_chunks_output = len(reranked)
        reranker_top_scores = [r.relevance_score for r in reranked[:3]]

        # Use reranked chunks for extraction
        chunks_for_extraction = [r.chunk for r in reranked]
        logger.info(
            f"    [Reranker] Using {len(chunks_for_extraction)} reranked chunks "
            f"(from {reranker_chunks_input} candidates)"
        )

    # Rate limiting config
    rate_limit = None
    if delay_between_calls > 0 or requests_per_minute:
        rate_limit = RateLimitConfig(
            delay_between_calls=delay_between_calls,
            requests_per_minute=requests_per_minute,
        )

    # Stage C: Run focused extraction with provider abstraction
    client = create_raw_client(
        provider=resolved_provider,
        model=resolved_model,
        api_key=api_key,
        rate_limit=rate_limit,
    )

    # Stage C: Run focused extraction with provider abstraction
    if field_name == "minimum_investment":
        result = extract_minimum_investment_from_chunks(
            chunks_for_extraction,
            client,
            resolved_model,
            resolved_provider,
            rate_limit,
        )
    elif field_name == "lock_up":
        result = extract_lock_up_from_chunks(
            chunks_for_extraction,
            client,
            resolved_model,
            resolved_provider,
            rate_limit,
        )
    elif field_name == "leverage_limits":
        result = extract_leverage_from_chunks(
            chunks_for_extraction,
            client,
            resolved_model,
            resolved_provider,
            rate_limit,
        )
    elif field_name == "distribution_terms":
        result = extract_distribution_terms_from_chunks(
            chunks_for_extraction,
            client,
            resolved_model,
            resolved_provider,
            rate_limit,
        )
    elif field_name == "minimum_additional_investment":
        result = extract_minimum_additional_from_chunks(
            chunks_for_extraction,
            client,
            resolved_model,
            resolved_provider,
            rate_limit,
        )
    elif field_name == "repurchase_basis":
        result = extract_repurchase_basis_from_chunks(
            chunks_for_extraction,
            client,
            resolved_model,
            resolved_provider,
            rate_limit,
        )
    elif field_name == "incentive_fee":
        result = extract_incentive_fee_from_chunks(
            chunks_for_extraction,
            client,
            resolved_model,
            resolved_provider,
            rate_limit,
        )
    elif field_name == "expense_cap":
        result = extract_expense_cap_from_chunks(
            chunks_for_extraction,
            client,
            resolved_model,
            resolved_provider,
            rate_limit,
        )
    elif field_name == "repurchase_terms":
        result = extract_repurchase_terms_from_chunks(
            chunks_for_extraction,
            client,
            resolved_model,
            resolved_provider,
            rate_limit,
        )
    elif field_name == "share_classes":
        result = extract_share_classes_from_chunks(
            chunks_for_extraction,
            client,
            resolved_model,
            resolved_provider,
            rate_limit,
        )
    elif field_name == "allocation_targets":
        result = extract_allocation_targets_from_chunks(
            chunks_for_extraction,
            client,
            resolved_model,
            resolved_provider,
            rate_limit,
        )
    elif field_name == "concentration_limits":
        result = extract_concentration_limits_from_chunks(
            chunks_for_extraction,
            client,
            resolved_model,
            resolved_provider,
            rate_limit,
        )
    else:
        # Generic extraction for other fields - default to minimum_investment
        result = extract_minimum_investment_from_chunks(
            chunks_for_extraction,
            client,
            resolved_model,
            resolved_provider,
            rate_limit,
        )

    source_section = top_sections[0].section.section_title if top_sections else "unknown"

    return ScopedExtractionResult(
        field_name=field_name,
        value=result,
        source_section=source_section,
        confidence="explicit" if result else "not_found",
        chunks_searched=len(chunks_for_extraction),
        sections_searched=len(top_sections),
        reranker_enabled=reranker_enabled,
        reranker_chunks_input=reranker_chunks_input,
        reranker_chunks_output=reranker_chunks_output,
        reranker_top_scores=reranker_top_scores,
    )


def apply_scoped_results_to_share_classes(
    extraction_result: dict,
    scoped_result: ScopedExtractionResult,
) -> dict:
    """
    Apply scoped agentic minimum investment results to share_classes extraction.

    Updates the extraction_result dict in place with values from scoped search.
    """
    if not scoped_result.value:
        return extraction_result

    scoped_values = scoped_result.value

    # Get or create share_classes structure
    share_classes = extraction_result.get("share_classes", {})
    if not isinstance(share_classes, dict):
        return extraction_result

    classes_list = share_classes.get("share_classes", [])
    if not classes_list:
        return extraction_result

    # Map scoped results to share classes
    class_mapping = {
        "class_s": "Class S",
        "class_d": "Class D",
        "class_i": "Class I",
        "class_i_advisory": "Class I Advisory",
    }

    for key, class_name in class_mapping.items():
        min_key = f"{key}_minimum"
        if min_key in scoped_values and scoped_values[min_key] is not None:
            # Find matching class in list
            for cls in classes_list:
                if cls.get("class_name", "").lower() == class_name.lower():
                    if cls.get("minimum_initial_investment") is None:
                        cls["minimum_initial_investment"] = scoped_values[min_key]
                        logger.info(f"    [Tier 3] Set {class_name} minimum_initial_investment = {scoped_values[min_key]}")

    return extraction_result


def apply_scoped_lock_up_to_repurchase(
    extraction_result: dict,
    scoped_result: ScopedExtractionResult,
) -> dict:
    """
    Apply scoped agentic lock-up results to repurchase_terms extraction.

    Updates the extraction_result dict in place with lock-up values from scoped search.
    """
    if not scoped_result.value:
        return extraction_result

    scoped_values = scoped_result.value

    # Get or create repurchase_terms structure
    repurchase = extraction_result.get("repurchase_terms", {})
    if not isinstance(repurchase, dict):
        return extraction_result

    # Apply lock-up values if not already present
    if scoped_values.get("lock_up_period_years") is not None:
        if repurchase.get("lock_up_period_years") is None:
            repurchase["lock_up_period_years"] = scoped_values["lock_up_period_years"]
            logger.info(f"    [Tier 3] Set lock_up_period_years = {scoped_values['lock_up_period_years']}")

    if scoped_values.get("early_repurchase_fee_pct") is not None:
        if repurchase.get("early_repurchase_fee_pct") is None:
            repurchase["early_repurchase_fee_pct"] = scoped_values["early_repurchase_fee_pct"]
            logger.info(f"    [Tier 3] Set early_repurchase_fee_pct = {scoped_values['early_repurchase_fee_pct']}")

    if scoped_values.get("early_repurchase_fee_period") is not None:
        if repurchase.get("early_repurchase_fee_period") is None:
            repurchase["early_repurchase_fee_period"] = scoped_values["early_repurchase_fee_period"]
            logger.info(f"    [Tier 3] Set early_repurchase_fee_period = {scoped_values['early_repurchase_fee_period']}")

    extraction_result["repurchase_terms"] = repurchase
    return extraction_result


def apply_scoped_leverage_to_result(
    extraction_result: dict,
    scoped_result: ScopedExtractionResult,
) -> dict:
    """
    Apply scoped agentic leverage results to leverage_limits extraction.

    Updates the extraction_result dict in place with leverage values from scoped search.
    """
    if not scoped_result.value:
        return extraction_result

    scoped_values = scoped_result.value

    # Get or create leverage_limits structure
    leverage = extraction_result.get("leverage_limits", {})
    if not isinstance(leverage, dict):
        leverage = {}

    # Apply leverage values if not already present
    if scoped_values.get("uses_leverage") is not None:
        leverage["uses_leverage"] = scoped_values["uses_leverage"]

    if scoped_values.get("max_leverage_pct") is not None:
        if leverage.get("max_leverage_pct") is None:
            leverage["max_leverage_pct"] = scoped_values["max_leverage_pct"]
            logger.info(f"    [Tier 3] Set max_leverage_pct = {scoped_values['max_leverage_pct']}")

    if scoped_values.get("leverage_basis") is not None:
        if leverage.get("leverage_basis") is None:
            leverage["leverage_basis"] = scoped_values["leverage_basis"]
            logger.info(f"    [Tier 3] Set leverage_basis = {scoped_values['leverage_basis']}")

    if scoped_values.get("credit_facility_size") is not None:
        if leverage.get("credit_facility_size") is None:
            leverage["credit_facility_size"] = scoped_values["credit_facility_size"]
            logger.info(f"    [Tier 3] Set credit_facility_size = {scoped_values['credit_facility_size']}")

    extraction_result["leverage_limits"] = leverage
    return extraction_result


def apply_scoped_distribution_to_result(
    extraction_result: dict,
    scoped_result: ScopedExtractionResult,
) -> dict:
    """
    Apply scoped agentic distribution results to distribution_terms extraction.

    Updates the extraction_result dict in place with distribution values from scoped search.
    """
    if not scoped_result.value:
        return extraction_result

    scoped_values = scoped_result.value

    # Get or create distribution_terms structure
    dist = extraction_result.get("distribution_terms", {})
    if not isinstance(dist, dict):
        dist = {}

    # Apply distribution values if not already present
    if scoped_values.get("distribution_frequency") is not None:
        if dist.get("distribution_frequency") is None:
            dist["distribution_frequency"] = scoped_values["distribution_frequency"]
            logger.info(f"    [Tier 3] Set distribution_frequency = {scoped_values['distribution_frequency']}")

    if scoped_values.get("default_distribution_policy") is not None:
        if dist.get("default_distribution_policy") is None:
            dist["default_distribution_policy"] = scoped_values["default_distribution_policy"]
            logger.info(f"    [Tier 3] Set default_distribution_policy = {scoped_values['default_distribution_policy']}")

    if scoped_values.get("target_distribution_rate") is not None:
        if dist.get("target_distribution_rate") is None:
            dist["target_distribution_rate"] = scoped_values["target_distribution_rate"]
            logger.info(f"    [Tier 3] Set target_distribution_rate = {scoped_values['target_distribution_rate']}")

    extraction_result["distribution_terms"] = dist
    return extraction_result


def apply_scoped_minimum_additional_to_share_classes(
    extraction_result: dict,
    scoped_result: ScopedExtractionResult,
) -> dict:
    """
    Apply scoped agentic minimum additional investment results to share_classes.

    Updates the extraction_result dict in place with values from scoped search.
    Handles universal values that apply to all share classes.
    """
    if not scoped_result.value:
        return extraction_result

    scoped_values = scoped_result.value

    # Get share_classes structure
    share_classes = extraction_result.get("share_classes", {})
    if not isinstance(share_classes, dict):
        return extraction_result

    classes_list = share_classes.get("share_classes", [])
    if not classes_list:
        return extraction_result

    # Check for universal value that applies to all classes
    universal_value = scoped_values.get("universal_minimum_additional")

    # Map scoped results to share classes
    class_mapping = {
        "class_s": "Class S",
        "class_d": "Class D",
        "class_i": "Class I",
        "class_i_advisory": "Class I Advisory",
    }

    for key, class_name in class_mapping.items():
        min_key = f"{key}_minimum_additional"
        # Use class-specific value if available, otherwise use universal value
        value = scoped_values.get(min_key)
        if value is None and universal_value is not None:
            value = universal_value

        if value is not None:
            # Find matching class in list
            for cls in classes_list:
                cls_name = cls.get("class_name", "")
                # Match case-insensitive and handle variations like "Class I Advisory"
                if cls_name.lower() == class_name.lower() or \
                   (key == "class_i_advisory" and "advisory" in cls_name.lower()):
                    if cls.get("minimum_additional_investment") is None:
                        cls["minimum_additional_investment"] = value
                        logger.info(f"    [Tier 3] Set {cls_name} minimum_additional_investment = {value}")

    return extraction_result


def apply_scoped_repurchase_basis_to_result(
    extraction_result: dict,
    scoped_result: ScopedExtractionResult,
) -> dict:
    """
    Apply scoped agentic repurchase basis results to repurchase_terms.

    Updates the extraction_result dict in place with repurchase_basis from scoped search.
    """
    if not scoped_result.value:
        return extraction_result

    scoped_values = scoped_result.value

    # Get or create repurchase_terms structure
    repurchase = extraction_result.get("repurchase_terms", {})
    if not isinstance(repurchase, dict):
        return extraction_result

    # Apply repurchase_basis if not already present
    if scoped_values.get("repurchase_basis") is not None:
        if repurchase.get("repurchase_basis") is None:
            repurchase["repurchase_basis"] = scoped_values["repurchase_basis"]
            logger.info(f"    [Tier 3] Set repurchase_basis = {scoped_values['repurchase_basis']}")

    extraction_result["repurchase_terms"] = repurchase
    return extraction_result


# =============================================================================
# HURDLE RATE NORMALIZATION
# =============================================================================

def normalize_hurdle_rate(rate_as_stated: float, frequency: str) -> dict:
    """
    Normalize hurdle rate to both quarterly and annual forms.

    Args:
        rate_as_stated: The rate number as extracted from the document
        frequency: Either "quarterly" or "annual"

    Returns:
        dict with hurdle_rate_pct (annual), hurdle_rate_quarterly,
        hurdle_rate_as_stated, hurdle_rate_frequency
    """
    if rate_as_stated is None:
        return {
            "hurdle_rate_pct": None,
            "hurdle_rate_quarterly": None,
            "hurdle_rate_as_stated": None,
            "hurdle_rate_frequency": frequency,
        }

    try:
        rate = float(rate_as_stated)
    except (ValueError, TypeError):
        return {
            "hurdle_rate_pct": None,
            "hurdle_rate_quarterly": None,
            "hurdle_rate_as_stated": rate_as_stated,
            "hurdle_rate_frequency": frequency,
        }

    if frequency == "annual":
        # Rate is annual, calculate quarterly
        quarterly = round(rate / 4, 4)
        annual = rate
    elif frequency == "quarterly":
        # Rate is quarterly, calculate annual
        quarterly = rate
        annual = round(rate * 4, 4)
    else:
        # Unknown frequency, assume annual
        quarterly = round(rate / 4, 4) if rate else None
        annual = rate

    return {
        "hurdle_rate_pct": str(annual) if annual is not None else None,
        "hurdle_rate_quarterly": str(quarterly) if quarterly is not None else None,
        "hurdle_rate_as_stated": str(rate_as_stated),
        "hurdle_rate_frequency": frequency,
    }


# =============================================================================
# TIER 3-ONLY MODE APPLY FUNCTIONS
# =============================================================================

def apply_scoped_incentive_fee_to_result(
    extraction_result: dict,
    scoped_result: ScopedExtractionResult,
) -> dict:
    """
    Apply scoped agentic incentive_fee results to extraction.
    For Tier 3-only mode, this directly sets the incentive_fee field.

    Includes post-processing to normalize hurdle rates:
    - If annual rate extracted, calculates quarterly equivalent
    - If quarterly rate extracted, calculates annual equivalent
    """
    if not scoped_result.value:
        return extraction_result

    scoped_values = scoped_result.value

    # Get or create incentive_fee structure
    incentive_fee = extraction_result.get("incentive_fee", {}) or {}

    # Apply non-hurdle-rate values from scoped extraction
    for key in ["has_incentive_fee", "incentive_fee_pct", "high_water_mark",
                "has_catch_up", "fee_basis", "crystallization_frequency", "evidence"]:
        if scoped_values.get(key) is not None:
            incentive_fee[key] = scoped_values[key]
            logger.info(f"    [Tier 3] Set incentive_fee.{key} = {scoped_values[key]}")

    # Post-process hurdle rate: normalize to both quarterly and annual
    rate_as_stated = scoped_values.get("hurdle_rate_as_stated")
    frequency = scoped_values.get("hurdle_rate_frequency")

    if rate_as_stated is not None or frequency is not None:
        normalized = normalize_hurdle_rate(rate_as_stated, frequency)

        # Apply normalized values
        incentive_fee["hurdle_rate_pct"] = normalized["hurdle_rate_pct"]
        incentive_fee["hurdle_rate_quarterly"] = normalized["hurdle_rate_quarterly"]
        incentive_fee["hurdle_rate_as_stated"] = normalized["hurdle_rate_as_stated"]
        incentive_fee["hurdle_rate_frequency"] = normalized["hurdle_rate_frequency"]

        logger.info(f"    [Tier 3] Normalized hurdle rate: as_stated={rate_as_stated}, "
                    f"freq={frequency} -> annual={normalized['hurdle_rate_pct']}, "
                    f"quarterly={normalized['hurdle_rate_quarterly']}")

    extraction_result["incentive_fee"] = incentive_fee
    return extraction_result


def apply_scoped_expense_cap_to_result(
    extraction_result: dict,
    scoped_result: ScopedExtractionResult,
) -> dict:
    """
    Apply scoped agentic expense_cap results to extraction.
    For Tier 3-only mode, this directly sets the expense_cap field.
    """
    if not scoped_result.value:
        return extraction_result

    scoped_values = scoped_result.value

    # Get or create expense_cap structure
    expense_cap = extraction_result.get("expense_cap", {}) or {}

    # Apply all values from scoped extraction
    for key in ["has_expense_cap", "expense_cap_pct", "expense_cap_expires",
                "waiver_recapture_period_years", "evidence"]:
        if scoped_values.get(key) is not None:
            expense_cap[key] = scoped_values[key]
            logger.info(f"    [Tier 3] Set expense_cap.{key} = {scoped_values[key]}")

    extraction_result["expense_cap"] = expense_cap
    return extraction_result


def apply_scoped_repurchase_terms_to_result(
    extraction_result: dict,
    scoped_result: ScopedExtractionResult,
) -> dict:
    """
    Apply scoped agentic repurchase_terms results to extraction.
    For Tier 3-only mode, this directly sets the full repurchase_terms field.
    """
    if not scoped_result.value:
        return extraction_result

    scoped_values = scoped_result.value

    # Get or create repurchase_terms structure
    repurchase_terms = extraction_result.get("repurchase_terms", {}) or {}

    # Apply all values from scoped extraction
    for key in ["repurchase_frequency", "repurchase_amount_pct", "repurchase_basis",
                "lock_up_period_years", "early_repurchase_fee_pct",
                "early_repurchase_fee_period", "evidence"]:
        if scoped_values.get(key) is not None:
            repurchase_terms[key] = scoped_values[key]
            logger.info(f"    [Tier 3] Set repurchase_terms.{key} = {scoped_values[key]}")

    extraction_result["repurchase_terms"] = repurchase_terms
    return extraction_result


def apply_scoped_share_classes_to_result(
    extraction_result: dict,
    scoped_result: ScopedExtractionResult,
) -> dict:
    """
    Apply scoped agentic share_classes results to extraction.
    For Tier 3-only mode, this directly sets the full share_classes field.
    """
    if not scoped_result.value:
        return extraction_result

    scoped_values = scoped_result.value

    # For share_classes, the structure should include a list of classes
    if isinstance(scoped_values, dict):
        extraction_result["share_classes"] = scoped_values
        class_list = scoped_values.get("share_classes", [])
        logger.info(f"    [Tier 3] Set share_classes with {len(class_list)} classes")

    return extraction_result


def apply_scoped_allocation_targets_to_result(
    extraction_result: dict,
    scoped_result: ScopedExtractionResult,
) -> dict:
    """
    Apply scoped agentic allocation_targets results to extraction.
    For Tier 3-only mode, this directly sets the allocation_targets field.
    """
    if not scoped_result.value:
        return extraction_result

    scoped_values = scoped_result.value

    # Get or create allocation_targets structure
    allocation = extraction_result.get("allocation_targets", {}) or {}

    # Apply all values from scoped extraction
    for key in ["private_equity_pct", "private_credit_pct", "real_estate_pct",
                "infrastructure_pct", "other_pct", "primary_pct", "secondary_pct",
                "co_investment_pct", "evidence"]:
        if scoped_values.get(key) is not None:
            allocation[key] = scoped_values[key]
            logger.info(f"    [Tier 3] Set allocation_targets.{key} = {scoped_values[key]}")

    extraction_result["allocation_targets"] = allocation
    return extraction_result


def apply_scoped_concentration_limits_to_result(
    extraction_result: dict,
    scoped_result: ScopedExtractionResult,
) -> dict:
    """
    Apply scoped agentic concentration_limits results to extraction.
    For Tier 3-only mode, this directly sets the concentration_limits field.
    """
    if not scoped_result.value:
        return extraction_result

    scoped_values = scoped_result.value

    # Get or create concentration_limits structure
    concentration = extraction_result.get("concentration_limits", {}) or {}

    # Apply all values from scoped extraction
    for key in ["single_issuer_limit_pct", "single_industry_limit_pct",
                "is_diversified", "evidence"]:
        if scoped_values.get(key) is not None:
            concentration[key] = scoped_values[key]
            logger.info(f"    [Tier 3] Set concentration_limits.{key} = {scoped_values[key]}")

    extraction_result["concentration_limits"] = concentration
    return extraction_result
