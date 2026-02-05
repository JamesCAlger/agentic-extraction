"""
Multi-Query Expansion for SEC Document Retrieval.

This module generates multiple query variations for each extraction field,
improving recall by capturing terminology variations that static keywords miss.

Architecture:
1. Pre-generate expansions at startup (or load from cache)
2. For each field, generate 5-8 query variations:
   - Keyword synonyms (domain-specific)
   - Natural language questions
   - Numeric/percentage patterns
   - Section title patterns
3. Cache expansions persistently (deterministic per field)

Production considerations:
- Persistent caching to avoid repeated LLM calls
- Fallback to static keywords if expansion fails
- Domain-specific prompt tuned for SEC documents
- Batch generation to minimize API calls
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class QueryExpansionConfig:
    """Configuration for multi-query expansion."""

    # Expansion method: "llm", "programmatic", or "hybrid"
    # - llm: Generate expansions using LLM (best quality, higher cost)
    # - programmatic: Use rule-based expansion from static mappings (fast, free)
    # - hybrid: Combine programmatic base with LLM enhancement
    expansion_method: str = "llm"

    # LLM settings (only used if expansion_method includes LLM)
    model: str = "gpt-4o-mini"
    temperature: float = 0.3  # Low temperature for consistent expansions

    # Expansion settings
    num_keyword_synonyms: int = 5
    num_questions: int = 3
    num_patterns: int = 3

    # Cache settings
    cache_dir: str = "data/cache/query_expansions"
    cache_enabled: bool = True

    # Fallback behavior
    fallback_to_static: bool = True

    # Rate limiting
    delay_between_calls: float = 0.1


@dataclass
class ExpandedQueries:
    """Container for expanded queries for a single field."""

    field_name: str

    # Different query types
    keyword_synonyms: list[str] = field(default_factory=list)
    natural_language_questions: list[str] = field(default_factory=list)
    pattern_variations: list[str] = field(default_factory=list)
    section_titles: list[str] = field(default_factory=list)

    # Metadata
    source: str = "llm"  # "llm", "cache", "static_fallback"
    generation_model: Optional[str] = None

    def all_queries(self) -> list[str]:
        """Return all queries as a flat list, deduplicated."""
        all_q = (
            self.keyword_synonyms +
            self.natural_language_questions +
            self.pattern_variations +
            self.section_titles
        )
        # Deduplicate while preserving order
        seen = set()
        result = []
        for q in all_q:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                result.append(q)
        return result

    def to_dict(self) -> dict:
        """Serialize for caching."""
        return {
            "field_name": self.field_name,
            "keyword_synonyms": self.keyword_synonyms,
            "natural_language_questions": self.natural_language_questions,
            "pattern_variations": self.pattern_variations,
            "section_titles": self.section_titles,
            "source": self.source,
            "generation_model": self.generation_model,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExpandedQueries":
        """Deserialize from cache."""
        return cls(
            field_name=data["field_name"],
            keyword_synonyms=data.get("keyword_synonyms", []),
            natural_language_questions=data.get("natural_language_questions", []),
            pattern_variations=data.get("pattern_variations", []),
            section_titles=data.get("section_titles", []),
            source=data.get("source", "cache"),
            generation_model=data.get("generation_model"),
        )


# =============================================================================
# FIELD DEFINITIONS (What we're extracting)
# =============================================================================

# Field descriptions for the LLM to understand context
FIELD_DESCRIPTIONS = {
    "minimum_investment": {
        "description": "The minimum initial investment amount required to purchase shares in each share class",
        "examples": ["$2,500 minimum", "$1,000,000 for Class I", "minimum initial investment of $500"],
        "context": "Often found in 'Plan of Distribution', 'Purchase of Shares', or share class tables",
    },
    "minimum_additional_investment": {
        "description": "The minimum subsequent/additional investment amount after the initial purchase",
        "examples": ["$500 minimum additional", "subsequent investment of $100", "additional purchases"],
        "context": "Usually near minimum initial investment information",
    },
    "share_classes": {
        "description": "Fund share classes with their fees, minimums, and distribution fees",
        "examples": ["Class S, Class D, Class I", "0.85% distribution fee", "12b-1 fee"],
        "context": "Found in 'Share Classes', 'Fees and Expenses', or prospectus summary",
    },
    "incentive_fee": {
        "description": "Performance fee or carried interest charged by the fund adviser",
        "examples": ["10% of profits above hurdle", "20% carried interest", "performance allocation"],
        "context": "Found in 'Advisory Fee', 'Incentive Fee', 'Compensation' sections",
    },
    "hurdle_rate": {
        "description": "The minimum return threshold before incentive fees apply",
        "examples": ["5% annualized hurdle", "8% preferred return", "1.5% quarterly hurdle"],
        "context": "Usually described alongside incentive fee terms",
    },
    "incentive_fee_high_water_mark": {
        "description": "Mechanism to prevent double-charging incentive fees on recovered losses",
        "examples": ["high water mark", "loss recovery account", "deficit carryforward"],
        "context": "Part of incentive fee calculation methodology",
    },
    "expense_cap": {
        "description": "Fee waiver or expense limitation agreement capping total expenses",
        "examples": ["expense cap of 1.25%", "fee waiver agreement", "expense reimbursement"],
        "context": "Found in 'Fee Waiver', 'Expense Limitation', or fee tables",
    },
    "repurchase_terms": {
        "description": "How the fund offers liquidity - repurchase frequency, percentage, and basis",
        "examples": ["quarterly repurchase offers", "5% to 25% of shares", "tender offer"],
        "context": "Found in 'Repurchase of Shares', 'Liquidity', 'Tender Offers' sections",
    },
    "lock_up": {
        "description": "Lock-up period and early repurchase/redemption fees",
        "examples": ["2% early repurchase fee", "one year lock-up", "shares held less than 12 months"],
        "context": "Near repurchase terms, often in 'Early Repurchase Fee' section",
    },
    "leverage_limits": {
        "description": "Maximum borrowing or leverage the fund may use",
        "examples": ["300% asset coverage", "may borrow up to 33-1/3%", "credit facility"],
        "context": "Found in 'Borrowings', 'Leverage', 'Investment Restrictions'",
    },
    "distribution_terms": {
        "description": "How often distributions are paid and default reinvestment policy",
        "examples": ["monthly distributions", "quarterly dividends", "DRIP"],
        "context": "Found in 'Distribution Policy', 'Dividends', 'DRIP' sections",
    },
    "allocation_targets": {
        "description": "Target asset allocation percentages across strategies",
        "examples": ["60% private equity", "30% private credit", "target allocation"],
        "context": "Found in 'Investment Strategy', 'Asset Allocation', 'Portfolio Composition'",
    },
    "concentration_limits": {
        "description": "Maximum percentage in single issuer, industry, or fund",
        "examples": ["no more than 25% in single issuer", "diversification policy"],
        "context": "Found in 'Investment Restrictions', 'Concentration', 'Fundamental Policies'",
    },
}


# =============================================================================
# PROGRAMMATIC EXPANSION RULES
# =============================================================================
# These are comprehensive, manually-curated expansion rules for each field.
# Based on analysis of actual SEC N-2 and N-CSR documents.

PROGRAMMATIC_EXPANSIONS = {
    "minimum_investment": {
        "keyword_synonyms": [
            "minimum initial investment",
            "minimum investment",
            "minimum purchase",
            "initial subscription",
            "minimum subscription",
            "minimum amount to invest",
            "investment minimum",
        ],
        "natural_language_questions": [
            "What is the minimum initial investment amount?",
            "What is the minimum purchase amount for each share class?",
            "How much do I need to invest initially?",
        ],
        "pattern_variations": [
            "$2,500", "$2500", "$5,000", "$5000",
            "$1,000,000", "$1000000", "$1,000", "$1000",
            "$500,000", "$500000", "$25,000", "$25000",
            "minimum of $", "at least $",
        ],
        "section_titles": [
            "Minimum Investment",
            "Purchase of Shares",
            "Plan of Distribution",
            "How to Purchase Shares",
            "Investment Minimums",
        ],
    },
    "minimum_additional_investment": {
        "keyword_synonyms": [
            "minimum subsequent investment",
            "minimum additional investment",
            "minimum additional purchase",
            "subsequent purchase",
            "additional subscription",
            "follow-on investment",
            "incremental investment",
        ],
        "natural_language_questions": [
            "What is the minimum subsequent investment amount?",
            "What is the minimum for additional purchases?",
            "What is the follow-on investment minimum?",
        ],
        "pattern_variations": [
            "$500", "$100", "$1,000",
            "subsequent investment of",
            "additional purchase of",
            "minimum for subsequent",
        ],
        "section_titles": [
            "Minimum Investment",
            "Purchase of Shares",
            "Subsequent Purchases",
        ],
    },
    "share_classes": {
        "keyword_synonyms": [
            "share class",
            "class of shares",
            "distribution fee",
            "shareholder servicing fee",
            "12b-1 fee",
            "sales load",
            "sales charge",
            "placement fee",
            "brokerage commission",
        ],
        "natural_language_questions": [
            "What share classes does the fund offer?",
            "What are the fees for each share class?",
            "What is the distribution fee for each class?",
        ],
        "pattern_variations": [
            "Class S", "Class D", "Class I", "Class A",
            "0.85%", "0.75%", "0.25%", "1.0%", "3.5%",
            "no sales load", "no distribution fee",
        ],
        "section_titles": [
            "Share Classes",
            "Classes of Shares",
            "Fees and Expenses",
            "Distribution and Servicing Fees",
            "Sales Charges",
        ],
    },
    "incentive_fee": {
        "keyword_synonyms": [
            "incentive fee",
            "performance fee",
            "performance allocation",
            "incentive allocation",
            "carried interest",
            "promote",
            "performance-based fee",
            "profit share",
            "profit participation",
        ],
        "natural_language_questions": [
            "What is the incentive fee or performance fee?",
            "What percentage of profits goes to the adviser?",
            "What is the carried interest or promote?",
        ],
        "pattern_variations": [
            "10% of", "12.5% of", "15% of", "20% of",
            "10% of profits", "20% carried interest",
            "with a full catch-up", "100% catch-up",
            "subject to a", "above the hurdle",
        ],
        "section_titles": [
            "Incentive Fee",
            "Performance Fee",
            "Advisory Fee",
            "Compensation of Adviser",
            "Management Fee",
        ],
    },
    "hurdle_rate": {
        "keyword_synonyms": [
            "hurdle rate",
            "preferred return",
            "benchmark rate",
            "threshold return",
            "hurdle",
            "minimum return threshold",
            "return hurdle",
        ],
        "natural_language_questions": [
            "What is the hurdle rate before incentive fees?",
            "What is the preferred return or threshold?",
            "What return must be achieved before performance fees?",
        ],
        "pattern_variations": [
            "5% annualized", "5.0% annualized",
            "6% annualized", "8% annualized",
            "1.25% quarterly", "1.5% per quarter",
            "SOFR plus", "Treasury plus",
        ],
        "section_titles": [
            "Incentive Fee",
            "Hurdle Rate",
            "Preferred Return",
        ],
    },
    "incentive_fee_high_water_mark": {
        "keyword_synonyms": [
            "high water mark",
            "high-water mark",
            "highwater mark",
            "loss recovery account",
            "loss carryforward",
            "deficit recovery",
            "deficit carryforward",
            "cumulative loss",
            "loss recovery mechanism",
        ],
        "natural_language_questions": [
            "Does the fund use a high water mark?",
            "Is there a loss recovery mechanism for incentive fees?",
            "How are prior losses treated in fee calculation?",
        ],
        "pattern_variations": [
            "recover prior losses",
            "losses must be recovered",
            "crystallization",
            "prevent double charging",
        ],
        "section_titles": [
            "Incentive Fee",
            "High Water Mark",
            "Loss Recovery",
        ],
    },
    "expense_cap": {
        "keyword_synonyms": [
            "expense cap",
            "expense limitation",
            "fee waiver",
            "expense reimbursement",
            "contractual cap",
            "operating expense limit",
            "expense limitation agreement",
            "fee waiver agreement",
        ],
        "natural_language_questions": [
            "What is the expense cap or limitation?",
            "Is there a fee waiver agreement?",
            "What is the maximum expense ratio?",
        ],
        "pattern_variations": [
            "capped at", "limited to",
            "waive fees", "reimburse expenses",
            "shall not exceed", "cap of",
            "1.25%", "1.5%", "2.0%",
        ],
        "section_titles": [
            "Expense Limitation",
            "Fee Waiver",
            "Expense Cap",
            "Fees and Expenses",
        ],
    },
    "repurchase_terms": {
        "keyword_synonyms": [
            "repurchase offer",
            "tender offer",
            "share repurchase",
            "redemption offer",
            "quarterly repurchase",
            "interval fund",
            "periodic repurchase",
            "liquidity offer",
        ],
        "natural_language_questions": [
            "How often does the fund offer repurchases?",
            "What percentage of shares can be repurchased?",
            "What are the repurchase or tender offer terms?",
        ],
        "pattern_variations": [
            "5% to 25%", "5% of outstanding",
            "at least 5%", "up to 25%",
            "quarterly", "semi-annual", "annual",
            "Rule 23c-3", "at NAV",
        ],
        "section_titles": [
            "Repurchase of Shares",
            "Repurchases of Shares",
            "Tender Offers",
            "Liquidity",
            "Repurchase Offers",
        ],
    },
    "lock_up": {
        "keyword_synonyms": [
            "lock-up period",
            "lock up period",
            "lockup",
            "early repurchase fee",
            "early withdrawal charge",
            "early redemption fee",
            "holding period",
            "short-term trading fee",
        ],
        "natural_language_questions": [
            "Is there a lock-up period?",
            "What is the early repurchase fee?",
            "Are there fees for early redemption?",
        ],
        "pattern_variations": [
            "2% fee", "2.0% fee",
            "within one year", "within 1 year",
            "less than 12 months", "less than one year",
            "held for less than",
        ],
        "section_titles": [
            "Early Repurchase Fee",
            "Lock-Up",
            "Repurchase of Shares",
            "Early Withdrawal",
        ],
    },
    "leverage_limits": {
        "keyword_synonyms": [
            "leverage limit",
            "borrowing limit",
            "maximum leverage",
            "debt limit",
            "asset coverage",
            "credit facility",
            "borrowings",
        ],
        "natural_language_questions": [
            "What is the fund's maximum leverage?",
            "How much can the fund borrow?",
            "What is the asset coverage requirement?",
        ],
        "pattern_variations": [
            "33-1/3%", "33 1/3%",
            "300% asset coverage", "150% asset coverage",
            "may borrow up to", "leverage will not exceed",
            "1940 Act", "as defined in the 1940 Act",
        ],
        "section_titles": [
            "Borrowings",
            "Leverage",
            "Investment Restrictions",
            "Fundamental Policies",
        ],
    },
    "distribution_terms": {
        "keyword_synonyms": [
            "distribution policy",
            "dividend policy",
            "distribution frequency",
            "dividend reinvestment",
            "DRIP",
            "income distribution",
            "capital gains distribution",
        ],
        "natural_language_questions": [
            "How often are distributions paid?",
            "What is the distribution policy?",
            "Is there automatic dividend reinvestment?",
        ],
        "pattern_variations": [
            "monthly", "quarterly", "annually",
            "paid monthly", "declared daily",
            "reinvested automatically",
        ],
        "section_titles": [
            "Distribution Policy",
            "Distributions",
            "Dividends",
            "Dividend Reinvestment Plan",
        ],
    },
    "allocation_targets": {
        "keyword_synonyms": [
            "target allocation",
            "asset allocation",
            "investment allocation",
            "portfolio allocation",
            "strategic allocation",
            "allocation target",
        ],
        "natural_language_questions": [
            "What are the target allocations by asset class?",
            "What percentage goes to private equity vs credit?",
            "How is the portfolio allocated?",
        ],
        "pattern_variations": [
            "private equity", "private credit",
            "real estate", "infrastructure",
            "% of assets", "% of the portfolio",
            "primary investments", "secondary investments",
            "secondaries", "co-investments",
        ],
        "section_titles": [
            "Investment Strategy",
            "Asset Allocation",
            "Portfolio Composition",
            "Investment Objective",
        ],
    },
    "concentration_limits": {
        "keyword_synonyms": [
            "concentration limit",
            "investment restriction",
            "diversification requirement",
            "fundamental policy",
            "non-fundamental policy",
            "single issuer limit",
        ],
        "natural_language_questions": [
            "What are the concentration limits?",
            "What is the maximum in a single issuer?",
            "What are the investment restrictions?",
        ],
        "pattern_variations": [
            "no more than 25%", "no more than 15%",
            "may not invest more than",
            "will not invest more than",
            "single issuer", "single industry",
        ],
        "section_titles": [
            "Investment Restrictions",
            "Concentration",
            "Diversification",
            "Fundamental Policies",
        ],
    },
}


# =============================================================================
# STATIC FALLBACK QUERIES (Used if LLM expansion fails)
# =============================================================================

STATIC_FALLBACK_QUERIES = {
    "minimum_investment": [
        "minimum initial investment",
        "minimum investment amount",
        "What is the minimum investment required?",
        "$2,500 minimum",
        "$1,000,000 Class I",
    ],
    "minimum_additional_investment": [
        "minimum subsequent investment",
        "minimum additional investment",
        "additional purchase minimum",
        "follow-on investment",
    ],
    "share_classes": [
        "share class fees and expenses",
        "Class S Class D Class I",
        "distribution fee 12b-1",
        "sales load upfront fee",
    ],
    "incentive_fee": [
        "incentive fee percentage",
        "performance fee carried interest",
        "10% of profits above hurdle",
        "20% carried interest promote",
        "performance allocation adviser compensation",
    ],
    "hurdle_rate": [
        "hurdle rate preferred return",
        "5% annualized hurdle",
        "benchmark threshold rate",
        "1.5% quarterly hurdle",
    ],
    "incentive_fee_high_water_mark": [
        "high water mark",
        "loss recovery account",
        "deficit carryforward",
        "cumulative loss recovery",
    ],
    "expense_cap": [
        "expense cap limitation",
        "fee waiver agreement",
        "expense reimbursement",
        "operating expense limit",
    ],
    "repurchase_terms": [
        "quarterly repurchase offers",
        "tender offer frequency",
        "5% to 25% of shares",
        "repurchase at NAV",
        "Rule 23c-3 interval fund",
    ],
    "lock_up": [
        "early repurchase fee",
        "lock-up period",
        "2% early redemption",
        "shares held less than one year",
    ],
    "leverage_limits": [
        "maximum borrowing leverage",
        "300% asset coverage 1940 Act",
        "credit facility borrowings",
        "33-1/3% of total assets",
    ],
    "distribution_terms": [
        "distribution policy frequency",
        "monthly quarterly dividends",
        "dividend reinvestment DRIP",
        "distributions paid monthly",
    ],
    "allocation_targets": [
        "target asset allocation",
        "private equity private credit percentage",
        "investment strategy allocation",
        "portfolio composition targets",
    ],
    "concentration_limits": [
        "concentration limit single issuer",
        "diversification policy",
        "no more than 25%",
        "investment restrictions fundamental",
    ],
}


# =============================================================================
# LLM-BASED QUERY EXPANSION
# =============================================================================

EXPANSION_SYSTEM_PROMPT = """You are an expert in SEC filings for private equity funds, specifically N-2 registration statements and N-CSR shareholder reports.

Your task is to generate search query variations to find specific information in these documents. The queries will be used for both keyword matching and semantic search.

Document characteristics:
- Legal/regulatory language with specific terminology
- Fee structures described in multiple ways (e.g., "incentive fee", "performance allocation", "carried interest")
- Percentages often written differently ($2,500 vs $2500, 5% vs 5.0% vs five percent)
- Section titles vary by fund (e.g., "Fees and Expenses" vs "Fund Expenses" vs "Cost of Investing")

Generate diverse queries that capture:
1. Exact terminology variations (synonyms used in SEC documents)
2. Natural language questions an analyst would ask
3. Specific numeric patterns likely to appear
4. Common section titles where this info appears"""


def _build_expansion_prompt(field_name: str, field_info: dict, config: QueryExpansionConfig) -> str:
    """Build the user prompt for query expansion."""
    return f"""Generate search queries for extracting "{field_name}" from SEC fund documents.

Field description: {field_info['description']}

Example values found in documents:
{chr(10).join(f"- {ex}" for ex in field_info['examples'])}

Context: {field_info['context']}

Generate exactly:
- {config.num_keyword_synonyms} keyword synonym phrases (2-5 words each, domain-specific terms)
- {config.num_questions} natural language questions (what an analyst would ask)
- {config.num_patterns} specific patterns with numbers/percentages likely to appear

Output as JSON:
{{
    "keyword_synonyms": ["phrase 1", "phrase 2", ...],
    "natural_language_questions": ["question 1?", "question 2?", ...],
    "pattern_variations": ["10% of profits", "$2,500 minimum", ...],
    "section_titles": ["Fees and Expenses", "Advisory Fee", ...]
}}

Be specific to SEC fund documents. Include variations you've seen in actual N-2 and N-CSR filings."""


class QueryExpander:
    """
    Multi-query expansion for SEC document retrieval.

    Generates query variations using an LLM and caches them persistently.
    Falls back to static queries if LLM expansion fails.
    """

    def __init__(self, config: Optional[QueryExpansionConfig] = None):
        """
        Initialize query expander.

        Args:
            config: Configuration for expansion behavior
        """
        self.config = config or QueryExpansionConfig()
        self._client = None
        self._cache: dict[str, ExpandedQueries] = {}
        self._cache_loaded = False

        # Ensure cache directory exists
        if self.config.cache_enabled:
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def _get_cache_path(self) -> Path:
        """Get path to cache file."""
        # Include config hash in filename for cache invalidation
        # Different cache files for different expansion methods
        config_str = f"{self.config.expansion_method}_{self.config.model}_{self.config.num_keyword_synonyms}_{self.config.num_questions}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return Path(self.config.cache_dir) / f"expansions_{self.config.expansion_method}_{config_hash}.json"

    def _load_cache(self) -> None:
        """Load cached expansions from disk."""
        if self._cache_loaded:
            return

        cache_path = self._get_cache_path()
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for field_name, expansion_data in data.items():
                    self._cache[field_name] = ExpandedQueries.from_dict(expansion_data)

                logger.info(f"Loaded {len(self._cache)} cached query expansions from {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        self._cache_loaded = True

    def _save_cache(self) -> None:
        """Save expansions to disk cache."""
        if not self.config.cache_enabled:
            return

        cache_path = self._get_cache_path()
        try:
            data = {
                field_name: expansion.to_dict()
                for field_name, expansion in self._cache.items()
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(self._cache)} query expansions to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _generate_expansion_llm(self, field_name: str) -> Optional[ExpandedQueries]:
        """Generate query expansions using LLM."""
        field_info = FIELD_DESCRIPTIONS.get(field_name)
        if not field_info:
            logger.warning(f"No field description for '{field_name}', using generic expansion")
            field_info = {
                "description": f"The {field_name.replace('_', ' ')} for this fund",
                "examples": [field_name.replace("_", " ")],
                "context": "Various sections of the fund document",
            }

        prompt = _build_expansion_prompt(field_name, field_info, self.config)

        try:
            import time
            time.sleep(self.config.delay_between_calls)

            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": EXPANSION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)

            expansion = ExpandedQueries(
                field_name=field_name,
                keyword_synonyms=result.get("keyword_synonyms", []),
                natural_language_questions=result.get("natural_language_questions", []),
                pattern_variations=result.get("pattern_variations", []),
                section_titles=result.get("section_titles", []),
                source="llm",
                generation_model=self.config.model,
            )

            logger.info(
                f"Generated {len(expansion.all_queries())} query variations for '{field_name}'"
            )
            return expansion

        except Exception as e:
            logger.error(f"LLM expansion failed for '{field_name}': {e}")
            return None

    def _generate_expansion_programmatic(self, field_name: str) -> ExpandedQueries:
        """
        Generate query expansions using programmatic rules.

        This is fast, free, and deterministic. Uses manually-curated expansion
        rules from PROGRAMMATIC_EXPANSIONS dictionary.

        Args:
            field_name: The extraction field

        Returns:
            ExpandedQueries with programmatic expansions
        """
        # Handle granular field paths (e.g., "incentive_fee.has_incentive_fee" -> "incentive_fee")
        lookup_key = field_name
        if "." in field_name:
            lookup_key = field_name.split(".")[0]

        rules = PROGRAMMATIC_EXPANSIONS.get(lookup_key, {})

        if not rules:
            logger.warning(f"No programmatic rules for '{field_name}' (lookup: {lookup_key}), using minimal expansion")
            return ExpandedQueries(
                field_name=field_name,
                keyword_synonyms=[field_name.replace("_", " ").replace(".", " ")],
                natural_language_questions=[f"What is the {field_name.replace('_', ' ').replace('.', ' ')}?"],
                pattern_variations=[],
                section_titles=[],
                source="programmatic",
            )

        expansion = ExpandedQueries(
            field_name=field_name,
            keyword_synonyms=rules.get("keyword_synonyms", []),
            natural_language_questions=rules.get("natural_language_questions", []),
            pattern_variations=rules.get("pattern_variations", []),
            section_titles=rules.get("section_titles", []),
            source="programmatic",
        )

        logger.info(
            f"Generated {len(expansion.all_queries())} programmatic query variations for '{field_name}'"
        )
        return expansion

    def _get_static_fallback(self, field_name: str) -> ExpandedQueries:
        """Get static fallback queries for a field."""
        queries = STATIC_FALLBACK_QUERIES.get(field_name, [])

        if not queries:
            # Generate minimal fallback from field name
            queries = [
                field_name.replace("_", " "),
                f"What is the {field_name.replace('_', ' ')}?",
            ]

        return ExpandedQueries(
            field_name=field_name,
            keyword_synonyms=queries,
            natural_language_questions=[],
            pattern_variations=[],
            section_titles=[],
            source="static_fallback",
        )

    def get_expanded_queries(self, field_name: str) -> ExpandedQueries:
        """
        Get expanded queries for a field.

        Uses the configured expansion_method:
        - "llm": Generate via LLM (best quality, costs money)
        - "programmatic": Use rule-based expansion (fast, free)
        - "hybrid": Start with programmatic, enhance with LLM

        Checks cache first for LLM/hybrid methods.

        Args:
            field_name: The extraction field (e.g., "incentive_fee")

        Returns:
            ExpandedQueries with all query variations
        """
        method = self.config.expansion_method

        # Programmatic doesn't need caching (it's deterministic and fast)
        if method == "programmatic":
            return self._generate_expansion_programmatic(field_name)

        # Load cache for LLM/hybrid methods
        self._load_cache()

        # Check cache
        if field_name in self._cache:
            logger.debug(f"Using cached expansion for '{field_name}'")
            return self._cache[field_name]

        # Generate based on method
        expansion = None

        if method == "llm":
            expansion = self._generate_expansion_llm(field_name)
        elif method == "hybrid":
            # Start with programmatic base
            prog_expansion = self._generate_expansion_programmatic(field_name)

            # Try to enhance with LLM
            llm_expansion = self._generate_expansion_llm(field_name)

            if llm_expansion:
                # Merge: programmatic base + LLM additions
                expansion = ExpandedQueries(
                    field_name=field_name,
                    keyword_synonyms=list(set(
                        prog_expansion.keyword_synonyms + llm_expansion.keyword_synonyms
                    )),
                    natural_language_questions=list(set(
                        prog_expansion.natural_language_questions + llm_expansion.natural_language_questions
                    )),
                    pattern_variations=list(set(
                        prog_expansion.pattern_variations + llm_expansion.pattern_variations
                    )),
                    section_titles=list(set(
                        prog_expansion.section_titles + llm_expansion.section_titles
                    )),
                    source="hybrid",
                    generation_model=self.config.model,
                )
                logger.info(
                    f"Merged {len(expansion.all_queries())} hybrid query variations for '{field_name}'"
                )
            else:
                # Fall back to programmatic only
                expansion = prog_expansion
                expansion.source = "hybrid_fallback"

        # Fall back to static if everything failed
        if expansion is None and self.config.fallback_to_static:
            logger.info(f"Using static fallback for '{field_name}'")
            expansion = self._get_static_fallback(field_name)

        if expansion and method != "programmatic":
            # Cache the result
            self._cache[field_name] = expansion
            self._save_cache()

        return expansion or self._get_static_fallback(field_name)

    def pregenerate_all(self, field_names: Optional[list[str]] = None) -> dict[str, ExpandedQueries]:
        """
        Pre-generate expansions for all fields.

        Call this at startup to amortize LLM cost.

        Args:
            field_names: Fields to expand (default: all defined fields)

        Returns:
            Dict mapping field names to their expansions
        """
        if field_names is None:
            field_names = list(FIELD_DESCRIPTIONS.keys())

        logger.info(f"Pre-generating query expansions for {len(field_names)} fields...")

        results = {}
        for field_name in field_names:
            results[field_name] = self.get_expanded_queries(field_name)

        logger.info(f"Completed expansion generation for {len(results)} fields")
        return results

    def get_stats(self) -> dict:
        """Return statistics about cached expansions."""
        self._load_cache()

        sources = {"llm": 0, "cache": 0, "static_fallback": 0, "programmatic": 0, "hybrid": 0, "hybrid_fallback": 0}
        total_queries = 0

        for expansion in self._cache.values():
            sources[expansion.source] = sources.get(expansion.source, 0) + 1
            total_queries += len(expansion.all_queries())

        return {
            "expansion_method": self.config.expansion_method,
            "cached_fields": len(self._cache),
            "total_queries": total_queries,
            "avg_queries_per_field": total_queries / max(len(self._cache), 1),
            "sources": sources,
            "cache_path": str(self._get_cache_path()),
            "config": {
                "expansion_method": self.config.expansion_method,
                "model": self.config.model,
                "num_synonyms": self.config.num_keyword_synonyms,
                "num_questions": self.config.num_questions,
                "num_patterns": self.config.num_patterns,
            },
        }


# =============================================================================
# MULTI-QUERY RETRIEVAL WITH RRF
# =============================================================================

@dataclass
class MultiQueryResult:
    """Result from multi-query retrieval with RRF fusion."""

    chunk: "Chunk"  # Forward reference
    rrf_score: float
    query_ranks: dict[str, int]  # query -> rank in that query's results
    num_queries_found: int

    @property
    def found_by_queries(self) -> list[str]:
        """Return list of queries that found this chunk."""
        return list(self.query_ranks.keys())


def multi_query_rrf_fusion(
    query_results: dict[str, list[tuple["Chunk", float]]],  # query -> [(chunk, score), ...]
    rrf_k: int = 60,
    final_top_k: int = 10,
) -> list[MultiQueryResult]:
    """
    Fuse results from multiple queries using Reciprocal Rank Fusion.

    RRF formula: score(chunk) = sum(1 / (k + rank_in_query_i))

    Chunks found by multiple queries get higher scores.

    Args:
        query_results: Dict mapping query string to list of (chunk, score) tuples
        rrf_k: RRF constant (typically 60)
        final_top_k: Number of results to return

    Returns:
        List of MultiQueryResult sorted by RRF score
    """
    from ..parse.models import Chunk

    # Build chunk ID -> (chunk, query_ranks) map
    chunk_data: dict[str, tuple[Chunk, dict[str, int]]] = {}

    def get_chunk_id(chunk: Chunk) -> str:
        """Generate unique chunk ID."""
        content_hash = hash(chunk.content[:200]) if chunk.content else 0
        return f"{chunk.section_title}:{chunk.chunk_index}:{content_hash}"

    penalty_rank = 1000

    for query, results in query_results.items():
        for rank, (chunk, score) in enumerate(results, start=1):
            chunk_id = get_chunk_id(chunk)

            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = (chunk, {})

            chunk_data[chunk_id][1][query] = rank

    # Calculate RRF scores
    fused_results = []

    for chunk_id, (chunk, query_ranks) in chunk_data.items():
        rrf_score = sum(
            1.0 / (rrf_k + rank)
            for rank in query_ranks.values()
        )

        fused_results.append(MultiQueryResult(
            chunk=chunk,
            rrf_score=rrf_score,
            query_ranks=query_ranks,
            num_queries_found=len(query_ranks),
        ))

    # Sort by RRF score (highest first)
    fused_results.sort(key=lambda x: x.rrf_score, reverse=True)

    return fused_results[:final_top_k]


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_query_expander(
    expansion_method: str = "programmatic",
    model: str = "gpt-4o-mini",
    cache_enabled: bool = True,
    cache_dir: str = "data/cache/query_expansions",
    **kwargs,
) -> QueryExpander:
    """
    Factory function to create a query expander.

    Args:
        expansion_method: "programmatic", "llm", or "hybrid"
        model: LLM model for generating expansions (only used if method includes LLM)
        cache_enabled: Whether to cache expansions
        cache_dir: Directory for cache files
        **kwargs: Additional config options

    Returns:
        Configured QueryExpander

    Examples:
        # Fast, free programmatic expansion
        expander = create_query_expander(expansion_method="programmatic")

        # LLM-based expansion (costs money, best quality)
        expander = create_query_expander(expansion_method="llm", model="gpt-4o-mini")

        # Hybrid: programmatic base + LLM enhancement
        expander = create_query_expander(expansion_method="hybrid")
    """
    config = QueryExpansionConfig(
        expansion_method=expansion_method,
        model=model,
        cache_enabled=cache_enabled,
        cache_dir=cache_dir,
        **kwargs,
    )
    return QueryExpander(config)
