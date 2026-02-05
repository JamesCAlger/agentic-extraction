"""
Hybrid Boolean-DocVQA + Tier3-Numeric Extraction

This module implements a hybrid approach that uses:
- DocVQA-style prompts for BOOLEAN fields (proven better at yes/no questions)
- Tier3-style prompts for NUMERIC/CATEGORICAL fields (proven better for complex extraction)
- Tier3's smart keyword-based chunk selection for BOTH (best chunk retrieval)

Based on empirical findings:
- DocVQA: 100% on has_incentive_fee, 100% on has_expense_cap
- Tier3: 100% on hurdle_rate_pct, 100% on expense_cap_pct, 80% on lock_up_period_years
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Any
import time

import instructor
from pydantic import BaseModel, Field
from decimal import Decimal

from .llm_provider import (
    create_instructor_client,
    create_raw_client,
    call_llm_json,
    RateLimitConfig,
    detect_provider,
    resolve_model_name,
)
from .per_datapoint_extractor import (
    DATAPOINT_KEYWORDS,
    DATAPOINT_PROMPTS,
    get_top_chunks_for_datapoint,
    ScoredChunk,
)
from .per_datapoint_tier3_style import (
    TIER3_STYLE_PROMPTS,
    DatapointExtraction,
)
from .prompts import SYSTEM_PROMPT
from ..parse.models import ChunkedDocument

logger = logging.getLogger(__name__)


# =============================================================================
# FIELD CLASSIFICATION
# =============================================================================

# Boolean fields - use DocVQA-style (simple yes/no questions work better)
BOOLEAN_FIELDS = {
    "incentive_fee.has_incentive_fee",
    "incentive_fee.high_water_mark",
    "incentive_fee.has_catch_up",
    "expense_cap.has_expense_cap",
    "leverage_limits.uses_leverage",
}

# Numeric fields - use Tier3-style (need context and structured extraction)
NUMERIC_FIELDS = {
    "incentive_fee.incentive_fee_pct",
    "incentive_fee.hurdle_rate_pct",
    "incentive_fee.catch_up_rate_pct",
    "expense_cap.expense_cap_pct",
    "repurchase_terms.repurchase_amount_pct",
    "repurchase_terms.lock_up_period_years",
    "repurchase_terms.early_repurchase_fee_pct",
    "leverage_limits.max_leverage_pct",
    "allocation_targets.secondary_funds_min_pct",
    "allocation_targets.secondary_funds_max_pct",
    "allocation_targets.direct_investments_min_pct",
    "allocation_targets.direct_investments_max_pct",
    "allocation_targets.secondary_investments_min_pct",
    "concentration_limits.max_single_asset_pct",
    "concentration_limits.max_single_fund_pct",
    "concentration_limits.max_single_sector_pct",
}

# Categorical/String fields - use Tier3-style (benefit from structured output)
CATEGORICAL_FIELDS = {
    "incentive_fee.hurdle_rate_as_stated",
    "incentive_fee.hurdle_rate_frequency",
    "incentive_fee.fee_basis",
    "incentive_fee.crystallization_frequency",
    "repurchase_terms.repurchase_frequency",
    "repurchase_terms.repurchase_basis",
    "leverage_limits.leverage_basis",
    "distribution_terms.distribution_frequency",
    "distribution_terms.default_distribution_policy",
}

# Complex fields - use Tier3-style
COMPLEX_FIELDS = {
    "share_classes.share_classes",
}


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class HybridDatapointResult:
    """Result from hybrid extraction."""
    datapoint_name: str
    value: Any
    evidence: Optional[str]
    confidence: Optional[str]
    extraction_style: str  # "docvqa" or "tier3"
    chunks_searched: int
    top_chunk_score: int
    extraction_time_ms: int


@dataclass
class HybridExtractionTrace:
    """Full trace of hybrid extraction."""
    fund_name: str
    total_datapoints: int
    successful_extractions: int
    docvqa_extractions: int
    tier3_extractions: int
    total_chunks_searched: int
    total_extraction_time_ms: int
    datapoint_results: dict[str, HybridDatapointResult]


# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================

def extract_boolean_docvqa(
    chunked_doc: ChunkedDocument,
    datapoint_name: str,
    client,
    model: str,
    provider: str,
    rate_limit: RateLimitConfig,
    top_k_chunks: int = 10,
) -> HybridDatapointResult:
    """
    Extract a boolean field using DocVQA-style prompt.
    Simple yes/no questions work better for boolean fields.
    """
    start_time = time.time()

    # Get top chunks using Tier3's keyword scoring
    top_chunks = get_top_chunks_for_datapoint(chunked_doc, datapoint_name, top_k_chunks)

    if not top_chunks:
        return HybridDatapointResult(
            datapoint_name=datapoint_name,
            value=None,
            evidence=None,
            confidence="not_found",
            extraction_style="docvqa",
            chunks_searched=0,
            top_chunk_score=0,
            extraction_time_ms=int((time.time() - start_time) * 1000),
        )

    # Combine chunk content
    combined_text = "\n\n".join([
        f"[Section: {sc.chunk.section_title}]\n{sc.chunk.content}"
        for sc in top_chunks
    ])

    if len(combined_text) > 12000:
        combined_text = combined_text[:12000]

    # Get DocVQA prompt
    prompt = DATAPOINT_PROMPTS.get(datapoint_name)
    if not prompt:
        field_name = datapoint_name.split(".")[-1]
        prompt = f"""Does this fund have {field_name.replace('_', ' ')}?

Return JSON:
{{"{field_name}": true/false, "evidence": "<quote>"}}"""

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

        if isinstance(result, dict):
            value = None
            evidence = result.get("evidence")
            for key, val in result.items():
                if key != "evidence":
                    value = val
                    break

            return HybridDatapointResult(
                datapoint_name=datapoint_name,
                value=value,
                evidence=evidence,
                confidence="explicit" if value is not None else "not_found",
                extraction_style="docvqa",
                chunks_searched=len(top_chunks),
                top_chunk_score=top_chunks[0].score if top_chunks else 0,
                extraction_time_ms=int((time.time() - start_time) * 1000),
            )

    except Exception as e:
        logger.error(f"DocVQA extraction failed for {datapoint_name}: {e}")

    return HybridDatapointResult(
        datapoint_name=datapoint_name,
        value=None,
        evidence=None,
        confidence="not_found",
        extraction_style="docvqa",
        chunks_searched=len(top_chunks),
        top_chunk_score=top_chunks[0].score if top_chunks else 0,
        extraction_time_ms=int((time.time() - start_time) * 1000),
    )


def extract_with_tier3_style(
    chunked_doc: ChunkedDocument,
    datapoint_name: str,
    client,
    model: str,
    provider: str,
    max_retries: int = 2,
    top_k_chunks: int = 10,
) -> HybridDatapointResult:
    """
    Extract a numeric/categorical field using Tier3-style prompt.
    Structured extraction with Pydantic schemas works better for complex fields.
    """
    start_time = time.time()

    # Get config for this datapoint
    config = TIER3_STYLE_PROMPTS.get(datapoint_name)
    if not config:
        logger.warning(f"No Tier3-style config for {datapoint_name}")
        return HybridDatapointResult(
            datapoint_name=datapoint_name,
            value=None,
            evidence=None,
            confidence=None,
            extraction_style="tier3",
            chunks_searched=0,
            top_chunk_score=0,
            extraction_time_ms=int((time.time() - start_time) * 1000),
        )

    schema = config["schema"]
    prompt_template = config["prompt"]

    # Get top chunks using Tier3's keyword scoring
    top_chunks = get_top_chunks_for_datapoint(chunked_doc, datapoint_name, top_k_chunks)

    if not top_chunks:
        return HybridDatapointResult(
            datapoint_name=datapoint_name,
            value=None,
            evidence=None,
            confidence="not_found",
            extraction_style="tier3",
            chunks_searched=0,
            top_chunk_score=0,
            extraction_time_ms=int((time.time() - start_time) * 1000),
        )

    # Combine chunk content
    combined_parts = []
    for sc in top_chunks:
        combined_parts.append(f"[Section: {sc.chunk.section_title}]\n{sc.chunk.content}")
    combined_text = "\n\n---\n\n".join(combined_parts)

    if len(combined_text) > 12000:
        combined_text = combined_text[:12000]

    user_prompt = prompt_template.format(text=combined_text)

    try:
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
            for attr in dir(result):
                if not attr.startswith("_") and attr not in ["evidence_quote", "confidence", "model_fields", "model_config"]:
                    val = getattr(result, attr, None)
                    if val is not None and not callable(val):
                        value = val
                        break

        return HybridDatapointResult(
            datapoint_name=datapoint_name,
            value=value,
            evidence=getattr(result, "evidence_quote", None),
            confidence=getattr(result, "confidence", None),
            extraction_style="tier3",
            chunks_searched=len(top_chunks),
            top_chunk_score=top_chunks[0].score if top_chunks else 0,
            extraction_time_ms=int((time.time() - start_time) * 1000),
        )

    except Exception as e:
        logger.error(f"Tier3-style extraction failed for {datapoint_name}: {e}")
        return HybridDatapointResult(
            datapoint_name=datapoint_name,
            value=None,
            evidence=None,
            confidence="not_found",
            extraction_style="tier3",
            chunks_searched=len(top_chunks),
            top_chunk_score=top_chunks[0].score if top_chunks else 0,
            extraction_time_ms=int((time.time() - start_time) * 1000),
        )


# =============================================================================
# HYBRID EXTRACTOR CLASS
# =============================================================================

class HybridBooleanTier3Extractor:
    """
    Hybrid extractor that uses:
    - DocVQA-style prompts for boolean fields
    - Tier3-style prompts for numeric/categorical fields
    - Tier3's keyword-based chunk selection for both
    """

    DATAPOINTS_TO_EXTRACT = [
        # Boolean fields (DocVQA)
        "incentive_fee.has_incentive_fee",
        "incentive_fee.high_water_mark",
        "incentive_fee.has_catch_up",
        "expense_cap.has_expense_cap",
        "leverage_limits.uses_leverage",
        # Numeric fields (Tier3)
        "incentive_fee.incentive_fee_pct",
        "incentive_fee.hurdle_rate_pct",
        "expense_cap.expense_cap_pct",
        "repurchase_terms.repurchase_amount_pct",
        "repurchase_terms.lock_up_period_years",
        "repurchase_terms.early_repurchase_fee_pct",
        "leverage_limits.max_leverage_pct",
        # Categorical fields (Tier3)
        "incentive_fee.hurdle_rate_as_stated",
        "incentive_fee.hurdle_rate_frequency",
        "incentive_fee.fee_basis",
        "incentive_fee.crystallization_frequency",
        "repurchase_terms.repurchase_frequency",
        "repurchase_terms.repurchase_basis",
        "leverage_limits.leverage_basis",
        "distribution_terms.distribution_frequency",
        "distribution_terms.default_distribution_policy",
        # Complex fields (Tier3)
        "share_classes.share_classes",
    ]

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        delay_between_calls: float = 0.5,
        requests_per_minute: int = 60,
        top_k_chunks: int = 10,
        max_retries: int = 2,
    ):
        self.model = resolve_model_name(model)
        self.provider = provider or detect_provider(model).value
        self.api_key = api_key
        self.top_k_chunks = top_k_chunks
        self.max_retries = max_retries

        self.rate_limit = RateLimitConfig(
            delay_between_calls=delay_between_calls,
            requests_per_minute=requests_per_minute,
        )

        # Create both client types
        # Raw client for DocVQA (simple JSON output)
        self.raw_client = create_raw_client(
            provider=self.provider,
            model=self.model,
            api_key=api_key,
            rate_limit=self.rate_limit,
        )

        # Instructor client for Tier3 (structured Pydantic output)
        self.instructor_client = create_instructor_client(
            provider=self.provider,
            model=self.model,
            api_key=api_key,
            rate_limit=self.rate_limit,
        )

    def _get_extraction_style(self, datapoint_name: str) -> str:
        """Determine which extraction style to use for a datapoint."""
        if datapoint_name in BOOLEAN_FIELDS:
            return "docvqa"
        return "tier3"

    def extract_all(
        self,
        chunked_doc: ChunkedDocument,
        fund_name: str,
    ) -> tuple[dict, HybridExtractionTrace]:
        """Extract all datapoints using hybrid approach."""
        logger.info(f"[Hybrid Boolean+Tier3] Extracting {len(self.DATAPOINTS_TO_EXTRACT)} datapoints for {fund_name}")

        start_time = time.time()
        datapoint_results = {}
        extraction_result = {}
        successful = 0
        docvqa_count = 0
        tier3_count = 0
        total_chunks = 0

        for datapoint_name in self.DATAPOINTS_TO_EXTRACT:
            extraction_style = self._get_extraction_style(datapoint_name)
            logger.info(f"  Extracting: {datapoint_name} [{extraction_style}]")

            if extraction_style == "docvqa":
                result = extract_boolean_docvqa(
                    chunked_doc=chunked_doc,
                    datapoint_name=datapoint_name,
                    client=self.raw_client,
                    model=self.model,
                    provider=self.provider,
                    rate_limit=self.rate_limit,
                    top_k_chunks=self.top_k_chunks,
                )
                docvqa_count += 1
            else:
                result = extract_with_tier3_style(
                    chunked_doc=chunked_doc,
                    datapoint_name=datapoint_name,
                    client=self.instructor_client,
                    model=self.model,
                    provider=self.provider,
                    max_retries=self.max_retries,
                    top_k_chunks=self.top_k_chunks,
                )
                tier3_count += 1

            datapoint_results[datapoint_name] = result
            total_chunks += result.chunks_searched

            if result.value is not None:
                successful += 1
                self._add_to_result(extraction_result, datapoint_name, result.value)

            logger.info(f"    -> {result.value} (score: {result.top_chunk_score}, {result.extraction_time_ms}ms)")

        trace = HybridExtractionTrace(
            fund_name=fund_name,
            total_datapoints=len(self.DATAPOINTS_TO_EXTRACT),
            successful_extractions=successful,
            docvqa_extractions=docvqa_count,
            tier3_extractions=tier3_count,
            total_chunks_searched=total_chunks,
            total_extraction_time_ms=int((time.time() - start_time) * 1000),
            datapoint_results=datapoint_results,
        )

        logger.info(f"[Hybrid] Complete: {successful}/{len(self.DATAPOINTS_TO_EXTRACT)} "
                   f"(DocVQA: {docvqa_count}, Tier3: {tier3_count}) in {trace.total_extraction_time_ms}ms")

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
