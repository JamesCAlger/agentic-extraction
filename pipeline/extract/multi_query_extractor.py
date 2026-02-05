"""
Multi-Query Extraction for SEC Documents.

Uses multi-query expansion and RRF fusion for retrieval, then extracts
using Tier3-style prompts. This is the integration layer for experiments.

Enables A/B testing of:
- Expansion methods: programmatic vs LLM vs hybrid
- Retrieval strategies: keyword vs dense vs hybrid
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Any

from .llm_provider import (
    create_instructor_client,
    RateLimitConfig,
    detect_provider,
    resolve_model_name,
)
from .multi_query_retriever import (
    MultiQueryRetriever,
    MultiQueryConfig,
    MultiQueryRetrievedChunk,
    create_multi_query_retriever,
)
from .per_datapoint_tier3_style import (
    TIER3_STYLE_PROMPTS,
)
from .prompts import SYSTEM_PROMPT
from .scoped_agentic import (
    apply_scoped_incentive_fee_to_result,
    extract_incentive_fee_from_chunks,
    extract_expense_cap_from_chunks,
    extract_repurchase_terms_from_chunks,
    extract_leverage_from_chunks,
    extract_distribution_terms_from_chunks,
    extract_allocation_targets_from_chunks,
    extract_concentration_limits_from_chunks,
    extract_share_classes_from_chunks,
)
from ..parse.models import ChunkedDocument, Chunk

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class MultiQueryExtractionTrace:
    """Trace information for a single field extraction."""

    field_name: str
    num_queries: int
    expansion_source: str  # "programmatic", "llm", "hybrid", etc.
    retrieval_strategy: str  # "keyword", "dense", "hybrid"
    chunks_retrieved: int
    queries_contributed: int  # How many queries found at least one final chunk
    top_chunk_info: list[dict]  # [{rrf_score, queries_found_by, ...}, ...]
    extraction_result: Any
    extraction_time_ms: float
    error: Optional[str] = None


@dataclass
class MultiQueryExtractionResult:
    """Complete extraction result using multi-query retrieval."""

    fund_name: str
    cik: str
    filing_id: str
    extraction: dict
    traces: dict[str, MultiQueryExtractionTrace] = field(default_factory=dict)
    total_time_s: float = 0.0
    retrieval_stats: dict = field(default_factory=dict)
    expansion_stats: dict = field(default_factory=dict)


# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================


def extract_field_from_multi_query_chunks(
    field_name: str,
    chunks: list[MultiQueryRetrievedChunk],
    client,
    model: str,
    provider: str,
    rate_limit: Optional[RateLimitConfig] = None,
    raw_client=None,
) -> tuple[Any, Optional[str], str]:
    """
    Extract a field value from multi-query retrieved chunks.

    Uses the same prompts and schemas as Tier3-style extraction.

    Args:
        field_name: Field to extract
        chunks: Multi-query retrieved chunks
        client: Instructor client
        model: Model name
        provider: LLM provider name
        rate_limit: Rate limiting config
        raw_client: Raw LLM client for generic JSON extraction

    Returns:
        Tuple of (extracted_value, error_message, combined_chunk_text)
    """
    if not chunks:
        return None, "No chunks retrieved", ""

    # Combine chunk content
    combined_text = "\n\n---\n\n".join([
        f"[Section: {chunk.chunk.section_title}]\n{chunk.chunk.content}"
        for chunk in chunks
    ])

    # Limit size
    if len(combined_text) > 12000:
        combined_text = combined_text[:12000]

    # Get config for this field (contains schema and prompt)
    config = TIER3_STYLE_PROMPTS.get(field_name)
    if not config:
        logger.warning(f"No Tier3-style config for {field_name}, using generic prompt")
        prompt = f"""Extract the {field_name.replace('_', ' ')} from the following text.

Return a JSON object with the extracted values. If a value cannot be found, use null.

TEXT:
{combined_text}"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\nYou must respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ]
        try:
            from .llm_provider import call_llm_json
            # Use raw_client for generic JSON extraction (not instructor client)
            extraction_client = raw_client if raw_client else client
            result = call_llm_json(
                client=extraction_client,
                provider=provider,
                model=model,
                messages=messages,
                rate_limit=rate_limit,
            )
            return result, None, combined_text
        except Exception as e:
            logger.error(f"Extraction failed for {field_name}: {e}")
            return None, str(e), combined_text

    # Use schema and prompt from config
    schema = config["schema"]
    prompt_template = config["prompt"]

    # Format prompt with combined text
    user_prompt = prompt_template.format(text=combined_text)

    try:
        create_kwargs = {
            "response_model": schema,
            "max_retries": 2,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }

        if provider != "google":
            create_kwargs["model"] = model

        if provider == "anthropic":
            create_kwargs["max_tokens"] = 4096

        result = client.chat.completions.create(**create_kwargs)
        return result.model_dump(), None, combined_text
    except Exception as e:
        logger.error(f"Extraction failed for {field_name}: {e}")
        return None, str(e), combined_text


# =============================================================================
# MAIN EXTRACTOR CLASS
# =============================================================================


# Fields to extract with multi-query retrieval (granular paths matching TIER3_STYLE_PROMPTS)
MULTI_QUERY_FIELDS = [
    # Incentive fee fields
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
    # Repurchase terms fields
    "repurchase_terms.repurchase_frequency",
    "repurchase_terms.repurchase_amount_pct",
    "repurchase_terms.repurchase_basis",
    "repurchase_terms.repurchase_percentage_min",
    "repurchase_terms.repurchase_percentage_max",
    "repurchase_terms.lock_up_period_years",
    "repurchase_terms.early_repurchase_fee_pct",
    # Leverage limits fields
    "leverage_limits.uses_leverage",
    "leverage_limits.max_leverage_pct",
    "leverage_limits.leverage_basis",
    # Distribution terms fields
    "distribution_terms.distribution_frequency",
    "distribution_terms.default_distribution_policy",
    # Allocation targets fields
    "allocation_targets.secondary_funds_min_pct",
    "allocation_targets.secondary_funds_max_pct",
    "allocation_targets.direct_investments_min_pct",
    "allocation_targets.direct_investments_max_pct",
    "allocation_targets.secondary_investments_min_pct",
    # Concentration limits fields
    "concentration_limits.max_single_asset_pct",
    "concentration_limits.max_single_fund_pct",
    "concentration_limits.max_single_sector_pct",
    # Share classes (single complex field)
    "share_classes.share_classes",
]


# =============================================================================
# HOLISTIC FIELD GROUPS
# =============================================================================

# Field groups for holistic extraction - extracts all related fields together
# Each group maps to a holistic extraction function from scoped_agentic.py
HOLISTIC_FIELD_GROUPS = {
    "incentive_fee": {
        "query_hint": "incentive_fee",  # Used for multi-query retrieval
        "extractor": "extract_incentive_fee_from_chunks",
        "fields": [
            "has_incentive_fee",
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
            "underlying_fund_incentive_range",
        ],
    },
    "expense_cap": {
        "query_hint": "expense_cap",
        "extractor": "extract_expense_cap_from_chunks",
        "fields": ["has_expense_cap", "expense_cap_pct"],
    },
    "repurchase_terms": {
        "query_hint": "repurchase_terms",
        "extractor": "extract_repurchase_terms_from_chunks",
        "fields": [
            "repurchase_frequency",
            "repurchase_amount_pct",
            "repurchase_basis",
            "repurchase_percentage_min",
            "repurchase_percentage_max",
            "lock_up_period_years",
            "early_repurchase_fee_pct",
        ],
    },
    "leverage_limits": {
        "query_hint": "leverage_limits",
        "extractor": "extract_leverage_from_chunks",
        "fields": ["uses_leverage", "max_leverage_pct", "leverage_basis"],
    },
    "distribution_terms": {
        "query_hint": "distribution_terms",
        "extractor": "extract_distribution_terms_from_chunks",
        "fields": ["distribution_frequency", "default_distribution_policy"],
    },
    "allocation_targets": {
        "query_hint": "allocation_targets",
        "extractor": "extract_allocation_targets_from_chunks",
        "fields": [
            "secondary_funds_min_pct",
            "secondary_funds_max_pct",
            "direct_investments_min_pct",
            "direct_investments_max_pct",
            "secondary_investments_min_pct",
        ],
    },
    "concentration_limits": {
        "query_hint": "concentration_limits",
        "extractor": "extract_concentration_limits_from_chunks",
        "fields": ["max_single_asset_pct", "max_single_fund_pct", "max_single_sector_pct"],
    },
    "share_classes": {
        "query_hint": "share_classes",
        "extractor": "extract_share_classes_from_chunks",
        "fields": ["share_classes"],  # Complex nested structure
    },
}

# Map extractor names to functions
HOLISTIC_EXTRACTORS = {
    "extract_incentive_fee_from_chunks": extract_incentive_fee_from_chunks,
    "extract_expense_cap_from_chunks": extract_expense_cap_from_chunks,
    "extract_repurchase_terms_from_chunks": extract_repurchase_terms_from_chunks,
    "extract_leverage_from_chunks": extract_leverage_from_chunks,
    "extract_distribution_terms_from_chunks": extract_distribution_terms_from_chunks,
    "extract_allocation_targets_from_chunks": extract_allocation_targets_from_chunks,
    "extract_concentration_limits_from_chunks": extract_concentration_limits_from_chunks,
    "extract_share_classes_from_chunks": extract_share_classes_from_chunks,
}


class MultiQueryExtractor:
    """
    Extractor using multi-query expansion and retrieval.

    Generates multiple query variations, retrieves chunks for each,
    fuses with RRF, then extracts using Tier3-style prompts.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        multi_query_config: Optional[MultiQueryConfig] = None,
        rate_limit: Optional[RateLimitConfig] = None,
        holistic_extraction: bool = False,
    ):
        """
        Initialize the multi-query extractor.

        Args:
            model: LLM model for extraction
            multi_query_config: Config for multi-query retrieval
            rate_limit: Rate limiting configuration
            holistic_extraction: If True, extracts all fields in a group together
                                 instead of per-field extraction
        """
        self.model = model
        self.rate_limit = rate_limit
        self.holistic_extraction = holistic_extraction

        # Create multi-query retriever
        self.multi_query_config = multi_query_config or MultiQueryConfig()
        self.retriever = MultiQueryRetriever(config=self.multi_query_config)

        # Create instructor client for extraction (for schema-based extraction)
        self.provider = detect_provider(model)
        resolved_model = resolve_model_name(model)
        self.client = create_instructor_client(self.provider, rate_limit=rate_limit)
        self._resolved_model = resolved_model

        # Also create raw client for generic JSON extraction
        from .llm_provider import create_raw_client
        self._raw_client = create_raw_client(self.provider)

        logger.info(
            f"MultiQueryExtractor initialized: "
            f"retrieval={self.multi_query_config.retrieval_strategy}, "
            f"expansion={self.multi_query_config.expansion_method}, "
            f"model={model}, "
            f"holistic={holistic_extraction}"
        )

    def extract(
        self,
        chunked_doc: ChunkedDocument,
        fund_name: str,
        cik: str,
        filing_id: str,
        fields: Optional[list[str]] = None,
    ) -> MultiQueryExtractionResult:
        """
        Extract data from a document using multi-query retrieval.

        Args:
            chunked_doc: The parsed and chunked document
            fund_name: Name of the fund
            cik: CIK number
            filing_id: Filing identifier
            fields: Specific fields to extract (default: MULTI_QUERY_FIELDS)

        Returns:
            MultiQueryExtractionResult with extractions and traces
        """
        # Dispatch to holistic extraction if enabled
        if self.holistic_extraction:
            return self._extract_holistic(
                chunked_doc=chunked_doc,
                fund_name=fund_name,
                cik=cik,
                filing_id=filing_id,
            )

        start_time = time.time()

        # Index document for retrieval
        self.retriever.index_document(chunked_doc)

        # Fields to extract
        fields_to_extract = fields or MULTI_QUERY_FIELDS

        # Results
        extraction = {}
        traces = {}

        for field_name in fields_to_extract:
            field_start = time.time()

            try:
                # Retrieve chunks
                chunks = self.retriever.retrieve(field_name)

                # Build trace info
                retrieval_stats = self.retriever._retrieval_stats

                # Extract value
                result, error, chunk_text = extract_field_from_multi_query_chunks(
                    field_name=field_name,
                    chunks=chunks,
                    client=self.client,
                    model=self._resolved_model,
                    provider=self.provider,
                    rate_limit=self.rate_limit,
                    raw_client=self._raw_client,
                )

                # Process result
                if result and not error:
                    # Store in nested structure for granular field paths
                    # e.g., "incentive_fee.has_incentive_fee" -> extraction["incentive_fee"]["has_incentive_fee"]
                    if "." in field_name:
                        parent, child = field_name.split(".", 1)
                        if parent not in extraction:
                            extraction[parent] = {}
                        # Extract the actual value from the result dict
                        if isinstance(result, dict):
                            # The schema returns the value in a field matching the child name
                            extraction[parent][child] = result.get(child, result)
                        else:
                            extraction[parent][child] = result

                        # Store the chunk text the LLM used for this field
                        # so the validator can see the same evidence
                        if chunk_text:
                            if "_evidence" not in extraction[parent]:
                                extraction[parent]["_evidence"] = {}
                            extraction[parent]["_evidence"][child] = chunk_text
                    else:
                        extraction[field_name] = result

                # Build trace
                traces[field_name] = MultiQueryExtractionTrace(
                    field_name=field_name,
                    num_queries=retrieval_stats.get("num_queries", 0),
                    expansion_source=retrieval_stats.get("expansion_source", "unknown"),
                    retrieval_strategy=retrieval_stats.get("retrieval_strategy", "unknown"),
                    chunks_retrieved=len(chunks),
                    queries_contributed=retrieval_stats.get("queries_contributed", 0),
                    top_chunk_info=[
                        {
                            "rrf_score": c.rrf_score,
                            "queries_found_by": len(c.queries_found_by),
                            "section": c.chunk.section_title[:50] if c.chunk.section_title else "",
                        }
                        for c in chunks[:5]
                    ],
                    extraction_result=result,
                    extraction_time_ms=(time.time() - field_start) * 1000,
                    error=error,
                )

                if error:
                    logger.warning(f"Extraction error for {field_name}: {error}")

            except Exception as e:
                logger.error(f"Failed to extract {field_name}: {e}")
                traces[field_name] = MultiQueryExtractionTrace(
                    field_name=field_name,
                    num_queries=0,
                    expansion_source="error",
                    retrieval_strategy=self.multi_query_config.retrieval_strategy,
                    chunks_retrieved=0,
                    queries_contributed=0,
                    top_chunk_info=[],
                    extraction_result=None,
                    extraction_time_ms=(time.time() - field_start) * 1000,
                    error=str(e),
                )

        total_time = time.time() - start_time

        return MultiQueryExtractionResult(
            fund_name=fund_name,
            cik=cik,
            filing_id=filing_id,
            extraction=extraction,
            traces=traces,
            total_time_s=total_time,
            retrieval_stats=self.retriever.get_stats(),
            expansion_stats=self.retriever.query_expander.get_stats(),
        )

    def _extract_holistic(
        self,
        chunked_doc: ChunkedDocument,
        fund_name: str,
        cik: str,
        filing_id: str,
    ) -> MultiQueryExtractionResult:
        """
        Extract data using holistic mode - retrieves chunks for each field GROUP
        and extracts all fields in the group together.

        This approach preserves context between related fields (e.g., all incentive
        fee fields are extracted together from the same chunks).
        """
        start_time = time.time()

        # Index document for retrieval
        self.retriever.index_document(chunked_doc)

        # Results
        extraction = {}
        traces = {}

        for group_name, group_config in HOLISTIC_FIELD_GROUPS.items():
            group_start = time.time()

            try:
                # Use the group name as query hint for retrieval
                query_hint = group_config["query_hint"]
                chunks = self.retriever.retrieve(query_hint)

                # Build trace info
                retrieval_stats = self.retriever._retrieval_stats

                # Convert MultiQueryRetrievedChunk to Chunk objects for holistic extractors
                raw_chunks = [c.chunk for c in chunks]

                # Get the holistic extraction function
                extractor_name = group_config["extractor"]
                extractor_fn = HOLISTIC_EXTRACTORS.get(extractor_name)

                if not extractor_fn:
                    logger.error(f"No extractor found for {extractor_name}")
                    continue

                # Call the holistic extraction function
                result = extractor_fn(
                    chunks=raw_chunks,
                    client=self._raw_client,
                    model=self._resolved_model,
                    provider=self.provider,
                    rate_limit=self.rate_limit,
                )

                # Store results in extraction dict
                if result:
                    extraction[group_name] = result

                # Build trace for the group
                traces[group_name] = MultiQueryExtractionTrace(
                    field_name=group_name,
                    num_queries=retrieval_stats.get("num_queries", 0),
                    expansion_source=retrieval_stats.get("expansion_source", "unknown"),
                    retrieval_strategy=retrieval_stats.get("retrieval_strategy", "unknown"),
                    chunks_retrieved=len(chunks),
                    queries_contributed=retrieval_stats.get("queries_contributed", 0),
                    top_chunk_info=[
                        {
                            "rrf_score": c.rrf_score,
                            "queries_found_by": len(c.queries_found_by),
                            "section": c.chunk.section_title[:50] if c.chunk.section_title else "",
                        }
                        for c in chunks[:5]
                    ],
                    extraction_result=result,
                    extraction_time_ms=(time.time() - group_start) * 1000,
                    error=None,
                )

                logger.info(
                    f"[Holistic] Extracted {group_name}: "
                    f"{len(chunks)} chunks, "
                    f"{len(result) if result else 0} fields"
                )

            except Exception as e:
                logger.error(f"Failed to extract group {group_name}: {e}")
                traces[group_name] = MultiQueryExtractionTrace(
                    field_name=group_name,
                    num_queries=0,
                    expansion_source="error",
                    retrieval_strategy=self.multi_query_config.retrieval_strategy,
                    chunks_retrieved=0,
                    queries_contributed=0,
                    top_chunk_info=[],
                    extraction_result=None,
                    extraction_time_ms=(time.time() - group_start) * 1000,
                    error=str(e),
                )

        total_time = time.time() - start_time

        return MultiQueryExtractionResult(
            fund_name=fund_name,
            cik=cik,
            filing_id=filing_id,
            extraction=extraction,
            traces=traces,
            total_time_s=total_time,
            retrieval_stats=self.retriever.get_stats(),
            expansion_stats=self.retriever.query_expander.get_stats(),
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_multi_query_extractor(
    model: str = "gpt-4o-mini",
    retrieval_strategy: str = "keyword",
    expansion_method: str = "programmatic",
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small",
    rrf_k: int = 60,
    per_query_top_k: int = 15,
    final_top_k: int = 10,
    rate_limit: Optional[RateLimitConfig] = None,
    holistic_extraction: bool = False,
) -> MultiQueryExtractor:
    """
    Factory function to create a multi-query extractor.

    Args:
        model: LLM model for extraction
        retrieval_strategy: "keyword", "dense", or "hybrid"
        expansion_method: "programmatic", "llm", or "hybrid"
        embedding_provider: For dense/hybrid retrieval
        embedding_model: Embedding model name
        rrf_k: RRF constant
        per_query_top_k: Chunks per query before fusion
        final_top_k: Final chunks after fusion
        rate_limit: Rate limiting config
        holistic_extraction: Extract all fields in a group together

    Returns:
        Configured MultiQueryExtractor
    """
    config = MultiQueryConfig(
        retrieval_strategy=retrieval_strategy,
        expansion_method=expansion_method,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        rrf_k=rrf_k,
        per_query_top_k=per_query_top_k,
        final_top_k=final_top_k,
    )

    return MultiQueryExtractor(
        model=model,
        multi_query_config=config,
        rate_limit=rate_limit,
        holistic_extraction=holistic_extraction,
    )
