"""
Hybrid Retrieval Extractor.

Uses hybrid retrieval (keyword + dense with RRF fusion) to find relevant
chunks, then extracts data using the same prompts as Tier3-style extraction.

This enables A/B testing of retrieval strategies while keeping
extraction logic constant.
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
from .hybrid_retriever import (
    HybridRetriever,
    HybridConfig,
    HybridRetrievedChunk,
    create_hybrid_retriever,
)
from .embedding_retriever import EmbeddingConfig
from .per_datapoint_tier3_style import (
    TIER3_STYLE_PROMPTS,
    Tier3StyleDatapointResult,
)
from .dense_retrieval_extractor import DENSE_RETRIEVAL_QUERIES
from .prompts import SYSTEM_PROMPT
from .scoped_agentic import (
    normalize_hurdle_rate,
    apply_scoped_incentive_fee_to_result,
)
from ..parse.models import ChunkedDocument

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class HybridExtractionTrace:
    """Trace information for a single field extraction."""

    field_name: str
    query: str
    chunks_retrieved: int
    retrieval_sources: dict[str, int]  # {"both": N, "keyword_only": N, "dense_only": N}
    top_chunk_scores: list[dict]  # [{rrf_score, keyword_rank, dense_rank}, ...]
    extraction_result: Any
    extraction_time_ms: float
    error: Optional[str] = None


@dataclass
class HybridExtractionResult:
    """Complete extraction result using hybrid retrieval."""

    fund_name: str
    cik: str
    filing_id: str
    extraction: dict
    traces: dict[str, HybridExtractionTrace] = field(default_factory=dict)
    total_time_s: float = 0.0
    retrieval_stats: dict = field(default_factory=dict)


# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================


def extract_field_from_hybrid_chunks(
    field_name: str,
    chunks: list[HybridRetrievedChunk],
    client,
    model: str,
    provider: str,
    rate_limit: Optional[RateLimitConfig] = None,
) -> tuple[Any, Optional[str]]:
    """
    Extract a field value from hybrid-retrieved chunks.

    Uses the same prompts and schemas as Tier3-style extraction.

    Args:
        field_name: Field to extract
        chunks: Hybrid retrieved chunks
        client: Instructor client
        model: Model name
        provider: LLM provider name
        rate_limit: Rate limiting config

    Returns:
        Tuple of (extracted_value, error_message)
    """
    if not chunks:
        return None, "No chunks retrieved"

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
        # Fall back to generic extraction without schema
        logger.warning(f"No Tier3-style config for {field_name}, using generic prompt")
        prompt = f"Extract the {field_name.replace('_', ' ')} from the following text.\n\nTEXT:\n{combined_text}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            from .llm_provider import call_llm_json
            result = call_llm_json(
                client=client,
                provider=provider,
                model=model,
                messages=messages,
                rate_limit=rate_limit,
            )
            return result, None
        except Exception as e:
            logger.error(f"Extraction failed for {field_name}: {e}")
            return None, str(e)

    # Use schema and prompt from config
    schema = config["schema"]
    prompt_template = config["prompt"]

    # Format prompt with combined text
    user_prompt = prompt_template.format(text=combined_text)

    try:
        # Use instructor for structured extraction
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

        # Convert to dict for consistent handling
        return result.model_dump(), None
    except Exception as e:
        logger.error(f"Extraction failed for {field_name}: {e}")
        return None, str(e)


# =============================================================================
# MAIN EXTRACTOR CLASS
# =============================================================================


class HybridExtractor:
    """
    Extractor using hybrid (keyword + dense) retrieval.

    Combines keyword and embedding-based retrieval using RRF fusion,
    then extracts using Tier3-style prompts.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        embedding_config: Optional[EmbeddingConfig] = None,
        hybrid_config: Optional[HybridConfig] = None,
        rate_limit: Optional[RateLimitConfig] = None,
    ):
        """
        Initialize the hybrid extractor.

        Args:
            model: LLM model for extraction
            embedding_config: Config for embedding retriever
            hybrid_config: Config for hybrid retrieval
            rate_limit: Rate limiting config
        """
        self.model = model
        self.rate_limit = rate_limit

        # Create hybrid retriever
        self.retriever = create_hybrid_retriever(
            embedding_config=embedding_config,
            hybrid_config=hybrid_config,
        )

        # Create LLM client
        self.provider = detect_provider(model)
        self.client = create_instructor_client(provider=self.provider)

        # Fields to extract - comprehensive list from all ground truth files
        self.fields_to_extract = [
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
        ]

    def extract(
        self,
        chunked_doc: ChunkedDocument,
        fund_name: Optional[str] = None,
    ) -> HybridExtractionResult:
        """
        Extract all fields from a document using hybrid retrieval.

        Args:
            chunked_doc: The parsed document
            fund_name: Optional fund name (if not in chunked_doc)

        Returns:
            HybridExtractionResult with all extractions and traces
        """
        start_time = time.time()

        # Get fund name from parameter (Pydantic models don't support getattr with default)
        if not fund_name:
            fund_name = 'Unknown Fund'

        # Index document
        self.retriever.index_document(chunked_doc)

        # Initialize result
        result = HybridExtractionResult(
            fund_name=fund_name,
            cik=chunked_doc.cik,
            filing_id=chunked_doc.filing_id,
            extraction={
                "filing_id": chunked_doc.filing_id,
                "cik": chunked_doc.cik,
                "fund_name": fund_name,
            },
        )

        # Group fields by top-level category
        field_groups = {}
        for field in self.fields_to_extract:
            category = field.split(".")[0]
            if category not in field_groups:
                field_groups[category] = []
            field_groups[category].append(field)

        # Extract each category
        for category, fields in field_groups.items():
            logger.info(f"Extracting {category} ({len(fields)} fields)")
            category_result = {}

            for field in fields:
                field_start = time.time()
                field_key = field.split(".")[-1]

                # Get query for dense retrieval
                query = DENSE_RETRIEVAL_QUERIES.get(
                    field,
                    f"What is the {field.replace('_', ' ')} for this fund?"
                )

                # Retrieve chunks using hybrid method
                chunks = self.retriever.retrieve(
                    field_name=category,
                    query=query,
                    top_k=10,
                )

                # Extract from chunks
                extracted, error = extract_field_from_hybrid_chunks(
                    field_name=field,
                    chunks=chunks,
                    client=self.client,
                    model=self.model,
                    provider=self.provider,
                    rate_limit=self.rate_limit,
                )

                # Store result
                if extracted is not None:
                    # Handle different response formats
                    if isinstance(extracted, dict):
                        if field_key in extracted:
                            category_result[field_key] = extracted[field_key]
                        elif "value" in extracted:
                            category_result[field_key] = extracted["value"]
                        else:
                            category_result[field_key] = extracted
                    else:
                        category_result[field_key] = extracted

                # Build trace
                retrieval_sources = {"both": 0, "keyword_only": 0, "dense_only": 0}
                for chunk in chunks:
                    retrieval_sources[chunk.retrieval_source] += 1

                trace = HybridExtractionTrace(
                    field_name=field,
                    query=query,
                    chunks_retrieved=len(chunks),
                    retrieval_sources=retrieval_sources,
                    top_chunk_scores=[
                        {
                            "rrf_score": c.rrf_score,
                            "keyword_rank": c.keyword_rank,
                            "dense_rank": c.dense_rank,
                        }
                        for c in chunks[:5]
                    ],
                    extraction_result=extracted,
                    extraction_time_ms=(time.time() - field_start) * 1000,
                    error=error,
                )
                result.traces[field] = trace

            # Store category result
            result.extraction[category] = category_result

        # Post-process incentive fee (normalize hurdle rate)
        if "incentive_fee" in result.extraction:
            incentive_fee = result.extraction["incentive_fee"]
            rate_as_stated = incentive_fee.get("hurdle_rate_as_stated")
            frequency = incentive_fee.get("hurdle_rate_frequency")

            if rate_as_stated is not None:
                normalized = normalize_hurdle_rate(rate_as_stated, frequency)
                incentive_fee["hurdle_rate_pct"] = normalized["hurdle_rate_pct"]
                incentive_fee["hurdle_rate_quarterly"] = normalized["hurdle_rate_quarterly"]
                incentive_fee["hurdle_rate_as_stated"] = normalized["hurdle_rate_as_stated"]
                incentive_fee["hurdle_rate_frequency"] = normalized["hurdle_rate_frequency"]

                logger.info(
                    f"[Hybrid] Normalized hurdle rate: as_stated={rate_as_stated}, "
                    f"freq={frequency} -> annual={normalized['hurdle_rate_pct']}, "
                    f"quarterly={normalized['hurdle_rate_quarterly']}"
                )

        # Finalize
        result.total_time_s = time.time() - start_time
        result.retrieval_stats = self.retriever.get_stats()

        logger.info(
            f"Hybrid extraction complete for {fund_name} "
            f"in {result.total_time_s:.1f}s"
        )

        return result


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_hybrid_extractor(
    model: str = "gpt-4o-mini",
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small",
    top_k: int = 10,
    rrf_k: int = 60,
    **kwargs,
) -> HybridExtractor:
    """
    Create a hybrid extractor with common defaults.

    Args:
        model: LLM model for extraction
        embedding_provider: Provider for embeddings
        embedding_model: Model for embeddings
        top_k: Number of chunks to retrieve
        rrf_k: RRF constant
        **kwargs: Additional arguments

    Returns:
        Configured HybridExtractor
    """
    embedding_config = EmbeddingConfig(
        provider=embedding_provider,
        model=embedding_model,
        top_k=top_k * 2,  # Retrieve more for fusion
        cache_embeddings=True,
    )

    hybrid_config = HybridConfig(
        rrf_k=rrf_k,
        keyword_top_k=top_k * 2,
        dense_top_k=top_k * 2,
        final_top_k=top_k,
    )

    return HybridExtractor(
        model=model,
        embedding_config=embedding_config,
        hybrid_config=hybrid_config,
        **kwargs,
    )
