"""
Dense Retrieval Extractor using Embedding-based Retrieval.

This module replaces keyword-based chunk retrieval with embedding-based
semantic similarity retrieval. Uses the same extraction prompts and schemas
as Tier3-style extraction, but retrieves chunks by semantic similarity
instead of keyword scoring.

Comparison with keyword-based Tier3:
- Tier3 (keyword): Score chunks by field-specific keyword matches
- Dense Retrieval: Score chunks by embedding similarity to field query

This allows A/B testing of retrieval methods while keeping extraction constant.
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
from .embedding_retriever import (
    EmbeddingRetriever,
    EmbeddingConfig,
    RetrievedChunk,
)
from .per_datapoint_tier3_style import (
    TIER3_STYLE_PROMPTS,
    Tier3StyleDatapointResult,
    Tier3StyleExtractionTrace,
)
from .prompts import SYSTEM_PROMPT
from ..parse.models import ChunkedDocument

logger = logging.getLogger(__name__)


# =============================================================================
# SEMANTIC QUERIES FOR DENSE RETRIEVAL
# =============================================================================
# These queries are used to retrieve relevant chunks via embedding similarity.
# They're designed to be semantically rich descriptions of what we're looking for.

DENSE_RETRIEVAL_QUERIES = {
    "incentive_fee.has_incentive_fee": (
        "Does this fund charge an incentive fee or performance fee? "
        "Information about incentive allocation, performance-based fees, or carried interest."
    ),
    "incentive_fee.incentive_fee_pct": (
        "What is the incentive fee percentage rate? "
        "The performance fee charged as a percentage of net profits or gains."
    ),
    "incentive_fee.hurdle_rate_pct": (
        "What is the hurdle rate or preferred return for the incentive fee? "
        "The minimum return threshold before incentive fees are charged, "
        "often stated as a quarterly or annual percentage."
    ),
    "incentive_fee.hurdle_rate_as_stated": (
        "What is the periodic hurdle rate before annualization? "
        "Look for quarterly or monthly rates like '1.25% per quarter', '1.5% quarterly'. "
        "Need the raw periodic number, not the annualized equivalent."
    ),
    "incentive_fee.hurdle_rate_frequency": (
        "Is the hurdle rate measured quarterly, annually, or at some other frequency? "
        "The time period over which the hurdle rate is calculated."
    ),
    "incentive_fee.high_water_mark": (
        "Does the fund use a high water mark or loss recovery mechanism? "
        "Information about high water mark, loss carryforward, deficit recovery, "
        "or provisions ensuring losses are recovered before incentive fees are paid."
    ),
    "incentive_fee.has_catch_up": (
        "Does the incentive fee have a catch-up provision? "
        "Whether the adviser receives a full catch-up after the hurdle is exceeded."
    ),
    "incentive_fee.fee_basis": (
        "What is the basis for calculating the incentive fee? "
        "Whether the fee is based on net investment income, net profits, "
        "capital gains, or NAV appreciation."
    ),
    "incentive_fee.crystallization_frequency": (
        "How often does the incentive fee crystallize? "
        "The frequency at which incentive fees are calculated and paid, "
        "such as quarterly or annually."
    ),
    "expense_cap.has_expense_cap": (
        "Does the fund have an expense limitation or fee waiver agreement? "
        "Information about expense caps, fee waivers, or reimbursement agreements."
    ),
    "expense_cap.expense_cap_pct": (
        "What is the expense cap percentage? "
        "The maximum total annual operating expenses as a percentage of net assets."
    ),
    "repurchase_terms.repurchase_frequency": (
        "How often does the fund conduct repurchase offers or tender offers? "
        "The frequency of share repurchases, such as quarterly for interval funds."
    ),
    "repurchase_terms.repurchase_amount_pct": (
        "What percentage of shares is offered for repurchase? "
        "The minimum or target repurchase offer amount as a percentage, "
        "typically 5% to 25% for interval funds."
    ),
    "repurchase_terms.repurchase_basis": (
        "What is the basis for calculating the repurchase offer? "
        "Whether the repurchase percentage is based on outstanding shares, "
        "net assets, or NAV."
    ),
    "repurchase_terms.lock_up_period_years": (
        "What is the lock-up period before shares can be repurchased? "
        "The minimum holding period before investors can participate in repurchases."
    ),
    "repurchase_terms.early_repurchase_fee_pct": (
        "What is the early repurchase fee or early withdrawal charge? "
        "The fee charged for shares redeemed before a certain holding period."
    ),
    "leverage_limits.uses_leverage": (
        "Does the fund use leverage or borrowing? "
        "Information about the fund's use of credit facilities, borrowing, "
        "or leverage for investment purposes."
    ),
    "leverage_limits.max_leverage_pct": (
        "What is the maximum leverage or borrowing limit? "
        "The fund's maximum borrowing as a percentage of assets, "
        "or asset coverage ratio requirements."
    ),
    "leverage_limits.leverage_basis": (
        "What is the basis for measuring leverage limits? "
        "Whether leverage is measured against total assets, net assets, "
        "or managed assets."
    ),
    "distribution_terms.distribution_frequency": (
        "How often does the fund pay dividends or distributions? "
        "The frequency of income distributions like monthly, quarterly, or annually."
    ),
    "distribution_terms.default_distribution_policy": (
        "What is the default distribution policy for dividends? "
        "Whether distributions are automatically reinvested or paid in cash, "
        "and information about DRIP programs."
    ),
    "share_classes.share_classes": (
        "What share classes does the fund offer and what are the minimums? "
        "Information about Class S, Class I, Class D shares, "
        "minimum initial investments, and distribution servicing fees."
    ),
}


# =============================================================================
# DENSE RETRIEVAL EXTRACTION
# =============================================================================

@dataclass
class DenseRetrievalResult:
    """Result from dense retrieval extraction."""
    datapoint_name: str
    value: Any
    evidence: Optional[str]
    confidence: Optional[str]
    chunks_retrieved: int
    top_similarity_score: float
    retrieval_time_ms: int
    extraction_time_ms: int


@dataclass
class DenseRetrievalTrace:
    """Full trace of dense retrieval extraction."""
    fund_name: str
    embedding_provider: str
    embedding_model: str
    total_datapoints: int
    successful_extractions: int
    total_chunks_retrieved: int
    total_time_ms: int
    datapoint_results: dict[str, DenseRetrievalResult]


def extract_with_dense_retrieval(
    chunked_doc: ChunkedDocument,
    datapoint_name: str,
    retriever: EmbeddingRetriever,
    client,
    model: str,
    provider: str,
    top_k: int = 10,
    max_retries: int = 2,
) -> DenseRetrievalResult:
    """
    Extract a single datapoint using embedding-based retrieval.

    Args:
        chunked_doc: Document to extract from
        datapoint_name: Name of datapoint (e.g., "incentive_fee.has_incentive_fee")
        retriever: Pre-initialized EmbeddingRetriever with indexed chunks
        client: Instructor-wrapped LLM client
        model: LLM model name
        provider: LLM provider
        top_k: Number of chunks to retrieve
        max_retries: Max LLM retries

    Returns:
        DenseRetrievalResult with extracted value and metadata
    """
    start_time = time.time()

    # Get config for this datapoint
    config = TIER3_STYLE_PROMPTS.get(datapoint_name)
    if not config:
        logger.warning(f"No extraction config for {datapoint_name}")
        return DenseRetrievalResult(
            datapoint_name=datapoint_name,
            value=None,
            evidence=None,
            confidence=None,
            chunks_retrieved=0,
            top_similarity_score=0.0,
            retrieval_time_ms=0,
            extraction_time_ms=int((time.time() - start_time) * 1000),
        )

    schema = config["schema"]
    prompt_template = config["prompt"]

    # Get query for this datapoint
    query = DENSE_RETRIEVAL_QUERIES.get(
        datapoint_name,
        f"What is the {datapoint_name.replace('_', ' ').replace('.', ' ')} for this fund?"
    )

    # Retrieve chunks by embedding similarity
    retrieval_start = time.time()
    retrieved = retriever.retrieve(query, top_k=top_k, field_name=datapoint_name)
    retrieval_time_ms = int((time.time() - retrieval_start) * 1000)

    if not retrieved:
        return DenseRetrievalResult(
            datapoint_name=datapoint_name,
            value=None,
            evidence=None,
            confidence="not_found",
            chunks_retrieved=0,
            top_similarity_score=0.0,
            retrieval_time_ms=retrieval_time_ms,
            extraction_time_ms=int((time.time() - start_time) * 1000),
        )

    # Combine chunk content
    combined_parts = []
    for r in retrieved:
        combined_parts.append(f"[Section: {r.chunk.section_title}]\n{r.chunk.content}")
    combined_text = "\n\n---\n\n".join(combined_parts)

    # Limit size
    if len(combined_text) > 12000:
        combined_text = combined_text[:12000]

    # Build prompt
    user_prompt = prompt_template.format(text=combined_text)

    try:
        # Use instructor for structured extraction
        create_kwargs = {
            "response_model": schema,
            "max_retries": max_retries,
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

        # Extract the primary value from the schema
        value = None
        field_name = datapoint_name.split(".")[-1]

        if hasattr(result, field_name):
            value = getattr(result, field_name)
        else:
            # Try to find the value field
            for attr in dir(result):
                if not attr.startswith("_") and attr not in ["evidence_quote", "confidence", "model_fields", "model_config"]:
                    val = getattr(result, attr, None)
                    if val is not None and not callable(val):
                        value = val
                        break

        return DenseRetrievalResult(
            datapoint_name=datapoint_name,
            value=value,
            evidence=getattr(result, "evidence_quote", None),
            confidence=getattr(result, "confidence", None),
            chunks_retrieved=len(retrieved),
            top_similarity_score=retrieved[0].similarity_score if retrieved else 0.0,
            retrieval_time_ms=retrieval_time_ms,
            extraction_time_ms=int((time.time() - start_time) * 1000),
        )

    except Exception as e:
        logger.error(f"Failed to extract {datapoint_name}: {e}")
        return DenseRetrievalResult(
            datapoint_name=datapoint_name,
            value=None,
            evidence=None,
            confidence="not_found",
            chunks_retrieved=len(retrieved),
            top_similarity_score=retrieved[0].similarity_score if retrieved else 0.0,
            retrieval_time_ms=retrieval_time_ms,
            extraction_time_ms=int((time.time() - start_time) * 1000),
        )


class DenseRetrievalExtractor:
    """
    Per-datapoint extraction using embedding-based dense retrieval.

    Replaces keyword scoring with semantic similarity for chunk retrieval,
    while keeping the same extraction prompts and schemas.
    """

    DATAPOINTS_TO_EXTRACT = [
        "incentive_fee.has_incentive_fee",
        "incentive_fee.incentive_fee_pct",
        "incentive_fee.hurdle_rate_pct",
        "incentive_fee.hurdle_rate_as_stated",
        "incentive_fee.hurdle_rate_frequency",
        "incentive_fee.high_water_mark",
        "incentive_fee.has_catch_up",
        "incentive_fee.fee_basis",
        "incentive_fee.crystallization_frequency",
        "expense_cap.has_expense_cap",
        "expense_cap.expense_cap_pct",
        "repurchase_terms.repurchase_frequency",
        "repurchase_terms.repurchase_amount_pct",
        "repurchase_terms.repurchase_basis",
        "repurchase_terms.lock_up_period_years",
        "repurchase_terms.early_repurchase_fee_pct",
        "leverage_limits.uses_leverage",
        "leverage_limits.max_leverage_pct",
        "leverage_limits.leverage_basis",
        "distribution_terms.distribution_frequency",
        "distribution_terms.default_distribution_policy",
        "share_classes.share_classes",
    ]

    def __init__(
        self,
        # LLM settings
        model: str = "gpt-4o-mini",
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        delay_between_calls: float = 0.5,
        requests_per_minute: int = 60,
        max_retries: int = 2,
        # Embedding settings
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        embedding_api_key: Optional[str] = None,
        prepend_context: bool = False,
        cache_embeddings: bool = True,
        # Retrieval settings
        top_k_chunks: int = 10,
    ):
        # LLM setup
        self.model = resolve_model_name(model)
        self.provider = provider or detect_provider(model).value
        self.api_key = api_key
        self.max_retries = max_retries
        self.top_k_chunks = top_k_chunks

        self.rate_limit = RateLimitConfig(
            delay_between_calls=delay_between_calls,
            requests_per_minute=requests_per_minute,
        )

        self.client = create_instructor_client(
            provider=self.provider,
            model=self.model,
            api_key=api_key,
            rate_limit=self.rate_limit,
        )

        # Embedding setup
        self.embedding_config = EmbeddingConfig(
            provider=embedding_provider,
            model=embedding_model,
            api_key=embedding_api_key,
            top_k=top_k_chunks,
            prepend_context=prepend_context,
            cache_embeddings=cache_embeddings,
        )
        self._retriever: Optional[EmbeddingRetriever] = None

    def _get_retriever(self, chunked_doc: ChunkedDocument) -> EmbeddingRetriever:
        """Get or create embedding retriever for document."""
        if self._retriever is None or self._retriever._indexed_doc_id != chunked_doc.filing_id:
            self._retriever = EmbeddingRetriever(self.embedding_config)
            logger.info(f"[Dense Retrieval] Indexing {chunked_doc.total_chunks} chunks...")
            self._retriever.index_chunks(chunked_doc)
        return self._retriever

    def extract_all(
        self,
        chunked_doc: ChunkedDocument,
        fund_name: str,
    ) -> tuple[dict, DenseRetrievalTrace]:
        """
        Extract all datapoints using dense retrieval.

        Args:
            chunked_doc: Document to extract from
            fund_name: Name of fund for logging

        Returns:
            Tuple of (extraction_result_dict, trace)
        """
        logger.info(
            f"[Dense Retrieval] Extracting {len(self.DATAPOINTS_TO_EXTRACT)} datapoints "
            f"for {fund_name} using {self.embedding_config.provider}/{self.embedding_config.model}"
        )

        start_time = time.time()

        # Get retriever (will index if needed)
        retriever = self._get_retriever(chunked_doc)

        datapoint_results = {}
        extraction_result = {}
        successful = 0
        total_chunks = 0

        for datapoint_name in self.DATAPOINTS_TO_EXTRACT:
            # Skip if no config
            if datapoint_name not in TIER3_STYLE_PROMPTS:
                logger.warning(f"  Skipping {datapoint_name} (no extraction config)")
                continue

            logger.info(f"  Extracting: {datapoint_name}")

            result = extract_with_dense_retrieval(
                chunked_doc=chunked_doc,
                datapoint_name=datapoint_name,
                retriever=retriever,
                client=self.client,
                model=self.model,
                provider=self.provider,
                top_k=self.top_k_chunks,
                max_retries=self.max_retries,
            )

            datapoint_results[datapoint_name] = result
            total_chunks += result.chunks_retrieved

            if result.value is not None:
                successful += 1
                self._add_to_result(extraction_result, datapoint_name, result.value)

            logger.info(
                f"    -> {result.value} (sim: {result.top_similarity_score:.3f}, "
                f"conf: {result.confidence}, {result.extraction_time_ms}ms)"
            )

        trace = DenseRetrievalTrace(
            fund_name=fund_name,
            embedding_provider=self.embedding_config.provider,
            embedding_model=self.embedding_config.model,
            total_datapoints=len(self.DATAPOINTS_TO_EXTRACT),
            successful_extractions=successful,
            total_chunks_retrieved=total_chunks,
            total_time_ms=int((time.time() - start_time) * 1000),
            datapoint_results=datapoint_results,
        )

        logger.info(
            f"[Dense Retrieval] Complete: {successful}/{len(self.DATAPOINTS_TO_EXTRACT)} "
            f"in {trace.total_time_ms}ms"
        )

        return extraction_result, trace

    def _add_to_result(self, result: dict, datapoint_name: str, value: Any):
        """Add extracted value to result dict."""
        parts = datapoint_name.split(".")
        if len(parts) == 2:
            field_name, subfield = parts
            if field_name not in result:
                result[field_name] = {}
            result[field_name][subfield] = value
        else:
            result[datapoint_name] = value


def convert_dense_retrieval_to_extraction_format(
    results: dict[str, DenseRetrievalResult],
) -> dict:
    """
    Convert dense retrieval results to standard extraction format.

    Args:
        results: Dict mapping datapoint names to DenseRetrievalResult

    Returns:
        Dict in standard extraction format for evaluator
    """
    extraction = {}

    for datapoint_name, result in results.items():
        if result.value is not None:
            parts = datapoint_name.split(".")
            if len(parts) == 2:
                field_name, subfield = parts
                if field_name not in extraction:
                    extraction[field_name] = {}
                extraction[field_name][subfield] = result.value
            else:
                extraction[datapoint_name] = result.value

    return extraction
