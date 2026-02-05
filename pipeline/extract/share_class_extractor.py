"""
Two-Pass Share Class Extraction.

This module implements a two-pass extraction strategy for share classes:
1. Pass 1 (Discovery): Identify ALL share class names in the document
2. Pass 2 (Per-Class): Extract fields for each discovered class with targeted retrieval

Benefits over single-pass extraction:
- Better recall for rare share classes (Class R, Class U, etc.)
- Class-specific retrieval finds data more reliably
- Reduces hallucination by scoping extraction to one class at a time
- No example bias - extracts classes actually present in document

Usage:
    extractor = TwoPassShareClassExtractor(config=TwoPassConfig())
    result = extractor.extract(chunked_doc)
    # result.share_classes contains list of ShareClassDetails
"""

import logging
import re
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Any

from .llm_provider import (
    create_instructor_client,
    create_raw_client,
    detect_provider,
    resolve_model_name,
    call_llm_json,
    RateLimitConfig,
)
from .prompts import (
    SYSTEM_PROMPT,
    SHARE_CLASS_DISCOVERY_PROMPT,
    SHARE_CLASS_FIELDS_PROMPT,
)
from .schemas import (
    ShareClassDiscovery,
    ShareClassFieldsExtraction,
    ShareClassDetails,
    ShareClassesExtraction,
    ConfidenceLevel,
)
from ..parse.models import ChunkedDocument, Chunk

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class TwoPassConfig:
    """Configuration for two-pass share class extraction."""

    # Model settings
    extraction_model: str = "gpt-4o-mini"
    discovery_model: Optional[str] = None  # Uses extraction_model if None

    # Rate limiting
    delay_between_calls: float = 0.5
    requests_per_minute: int = 40

    # Discovery settings
    discovery_max_chunks: int = 20  # Chunks to use for discovery
    discovery_sections: list[str] = field(default_factory=lambda: [
        "plan_of_distribution",
        "fee_table",
        "fees_and_expenses",
        "purchase_of_shares",
        "summary_of_fund_expenses",
        "shareholder_information",
    ])

    # Per-class extraction settings
    per_class_max_chunks: int = 15  # Chunks per class extraction
    fallback_to_full_doc: bool = True  # If class-specific retrieval empty


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class ShareClassDiscoveryTrace:
    """Trace for discovery phase."""
    chunks_used: int = 0
    classes_found: list[str] = field(default_factory=list)
    confidence: Optional[str] = None
    reasoning: Optional[str] = None
    duration_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class PerClassExtractionTrace:
    """Trace for per-class extraction."""
    class_name: str = ""
    chunks_retrieved: int = 0
    retrieval_keywords: list[str] = field(default_factory=list)
    extraction_result: Optional[dict] = None
    duration_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class TwoPassExtractionResult:
    """Result from two-pass extraction."""
    share_classes: list[ShareClassDetails] = field(default_factory=list)
    discovery_trace: Optional[ShareClassDiscoveryTrace] = None
    extraction_traces: list[PerClassExtractionTrace] = field(default_factory=list)
    total_duration_ms: float = 0.0


# =============================================================================
# KEYWORD SCORING FOR SHARE CLASS RETRIEVAL
# =============================================================================


def get_share_class_discovery_keywords() -> list[str]:
    """Keywords for discovering share class names."""
    return [
        "class shares",
        "class i",
        "class s",
        "class d",
        "class t",
        "class u",
        "class r",
        "class a",
        "class c",
        "share class",
        "institutional class",
        "advisor class",
        "plan of distribution",
        "fee table",
        "minimum investment",
        "shares outstanding",
    ]


def get_class_specific_keywords(class_name: str) -> list[str]:
    """
    Generate retrieval keywords for a specific share class.

    Args:
        class_name: Normalized class name (e.g., "Class I")

    Returns:
        List of keywords to search for
    """
    # Base keywords with the class name
    base_keywords = [
        class_name.lower(),
        f"{class_name.lower()} shares",
        f"{class_name.lower()} minimum",
        f"{class_name.lower()} investment",
    ]

    # Extract the letter/designation (e.g., "I" from "Class I")
    parts = class_name.split()
    if len(parts) >= 2:
        designation = parts[-1]  # "I", "S", "D", etc.
        base_keywords.extend([
            f"{designation.lower()} shares",
            f"class {designation.lower()}",
        ])

    # Add general share class keywords
    base_keywords.extend([
        "minimum initial investment",
        "minimum additional investment",
        "sales load",
        "placement fee",
        "distribution fee",
        "servicing fee",
        "12b-1",
    ])

    return base_keywords


def score_chunk_for_class(chunk: Chunk, class_name: str) -> int:
    """
    Score a chunk for relevance to a specific share class.

    Args:
        chunk: The chunk to score
        class_name: The class to check for (e.g., "Class I")

    Returns:
        Integer score (higher = more relevant)
    """
    content_lower = chunk.content.lower()
    section_lower = (chunk.section_title or "").lower()
    score = 0

    # Check for class name presence (high value)
    class_lower = class_name.lower()
    if class_lower in content_lower:
        score += 10

    # Check for class designation alone (medium value)
    parts = class_name.split()
    if len(parts) >= 2:
        designation = parts[-1].lower()
        # Match patterns like "I Shares", "Class I"
        if re.search(rf'\b{designation}\s+shares?\b', content_lower):
            score += 5
        if re.search(rf'\bclass\s+{designation}\b', content_lower):
            score += 5

    # Section relevance
    relevant_sections = [
        "plan of distribution",
        "fee table",
        "purchase",
        "minimum investment",
        "shareholder",
    ]
    for section in relevant_sections:
        if section in section_lower:
            score += 3
            break

    # Keywords in content
    keywords = [
        "minimum initial investment",
        "minimum additional",
        "sales load",
        "placement fee",
        "distribution fee",
        "servicing fee",
    ]
    for kw in keywords:
        if kw in content_lower:
            score += 2

    return score


def get_all_chunks(chunked_doc: ChunkedDocument) -> list[Chunk]:
    """Extract all chunks from a ChunkedDocument."""
    all_chunks = []
    for section in chunked_doc.chunked_sections:
        if section.chunks:
            all_chunks.extend(section.chunks)
    return all_chunks


def retrieve_chunks_for_discovery(
    chunked_doc: ChunkedDocument,
    config: TwoPassConfig,
) -> list[Chunk]:
    """
    Retrieve chunks likely to contain share class names.

    Args:
        chunked_doc: The document
        config: Extraction config

    Returns:
        List of chunks for discovery
    """
    all_chunks = get_all_chunks(chunked_doc)
    keywords = get_share_class_discovery_keywords()

    # Score all chunks
    scored_chunks = []
    for chunk in all_chunks:
        content_lower = chunk.content.lower()
        section_lower = (chunk.section_title or "").lower()

        score = 0
        for kw in keywords:
            if kw in content_lower:
                score += 3
            if kw in section_lower:
                score += 5

        # Boost sections that typically contain share class info
        for section_name in config.discovery_sections:
            if section_name.replace("_", " ") in section_lower:
                score += 10
                break

        if score > 0:
            scored_chunks.append((chunk, score))

    # Sort by score and take top N
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    result = [chunk for chunk, _ in scored_chunks[:config.discovery_max_chunks]]

    logger.debug(
        f"[Discovery] Retrieved {len(result)} chunks from "
        f"{len(scored_chunks)} scored chunks"
    )

    return result


def retrieve_chunks_for_class(
    chunked_doc: ChunkedDocument,
    class_name: str,
    config: TwoPassConfig,
) -> list[Chunk]:
    """
    Retrieve chunks mentioning a specific share class.

    Args:
        chunked_doc: The document
        class_name: The class to search for
        config: Extraction config

    Returns:
        List of chunks mentioning this class
    """
    all_chunks = get_all_chunks(chunked_doc)

    # Score chunks for this specific class
    scored_chunks = []
    for chunk in all_chunks:
        score = score_chunk_for_class(chunk, class_name)
        if score > 0:
            scored_chunks.append((chunk, score))

    # Sort by score
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # Take top N
    result = [chunk for chunk, _ in scored_chunks[:config.per_class_max_chunks]]

    # Fallback to full doc sections if no class-specific chunks found
    if not result and config.fallback_to_full_doc:
        logger.warning(
            f"[PerClass] No chunks found for {class_name}, "
            "falling back to discovery chunks"
        )
        result = retrieve_chunks_for_discovery(chunked_doc, config)

    logger.debug(
        f"[PerClass] Retrieved {len(result)} chunks for {class_name}"
    )

    return result


# =============================================================================
# TWO-PASS EXTRACTOR
# =============================================================================


class TwoPassShareClassExtractor:
    """
    Two-pass share class extraction for improved accuracy.

    Pass 1: Discover which share classes exist in the document
    Pass 2: Extract fields for each discovered class with targeted retrieval
    """

    def __init__(
        self,
        config: Optional[TwoPassConfig] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the two-pass extractor.

        Args:
            config: Extraction configuration
            api_key: API key for LLM calls (uses env var if None)
        """
        self.config = config or TwoPassConfig()
        self.api_key = api_key

        # Create LLM clients
        self.provider = detect_provider(self.config.extraction_model)
        self.extraction_model = resolve_model_name(self.config.extraction_model)

        discovery_model = self.config.discovery_model or self.config.extraction_model
        self.discovery_provider = detect_provider(discovery_model)
        self.discovery_model = resolve_model_name(discovery_model)

        # Rate limiting
        self.rate_limit = RateLimitConfig(
            delay_between_calls=self.config.delay_between_calls,
            requests_per_minute=self.config.requests_per_minute,
        )

        # Create clients (lazy initialization)
        self._instructor_client = None
        self._raw_client = None

    @property
    def instructor_client(self):
        """Lazy-load instructor client."""
        if self._instructor_client is None:
            self._instructor_client = create_instructor_client(
                self.provider,
                rate_limit=self.rate_limit,
            )
        return self._instructor_client

    @property
    def raw_client(self):
        """Lazy-load raw client."""
        if self._raw_client is None:
            self._raw_client = create_raw_client(self.provider)
        return self._raw_client

    def extract(
        self,
        chunked_doc: ChunkedDocument,
    ) -> TwoPassExtractionResult:
        """
        Run two-pass extraction on a document.

        Args:
            chunked_doc: The chunked document

        Returns:
            TwoPassExtractionResult with discovered and extracted share classes
        """
        start_time = time.time()

        logger.info("[TwoPass] Starting share class extraction")

        # Pass 1: Discover share classes
        discovery_trace, discovered_classes = self._discover_share_classes(chunked_doc)

        if not discovered_classes:
            logger.warning("[TwoPass] No share classes discovered")
            return TwoPassExtractionResult(
                share_classes=[],
                discovery_trace=discovery_trace,
                extraction_traces=[],
                total_duration_ms=(time.time() - start_time) * 1000,
            )

        logger.info(f"[TwoPass] Discovered {len(discovered_classes)} classes: {discovered_classes}")

        # Pass 2: Extract fields for each class
        share_classes = []
        extraction_traces = []

        for class_name in discovered_classes:
            trace, share_class = self._extract_class_fields(chunked_doc, class_name)
            extraction_traces.append(trace)

            if share_class:
                share_classes.append(share_class)
                logger.info(f"[TwoPass] Extracted {class_name}: {share_class.minimum_initial_investment}")
            else:
                logger.warning(f"[TwoPass] Failed to extract fields for {class_name}")

        total_duration = (time.time() - start_time) * 1000

        logger.info(
            f"[TwoPass] Complete: {len(share_classes)}/{len(discovered_classes)} "
            f"classes extracted in {total_duration:.0f}ms"
        )

        return TwoPassExtractionResult(
            share_classes=share_classes,
            discovery_trace=discovery_trace,
            extraction_traces=extraction_traces,
            total_duration_ms=total_duration,
        )

    def _discover_share_classes(
        self,
        chunked_doc: ChunkedDocument,
    ) -> tuple[ShareClassDiscoveryTrace, list[str]]:
        """
        Pass 1: Discover which share classes exist in the document.

        Args:
            chunked_doc: The document

        Returns:
            Tuple of (trace, list of class names)
        """
        start_time = time.time()
        trace = ShareClassDiscoveryTrace()

        try:
            # Retrieve relevant chunks
            chunks = retrieve_chunks_for_discovery(chunked_doc, self.config)
            trace.chunks_used = len(chunks)

            if not chunks:
                logger.warning("[Discovery] No chunks found for discovery")
                trace.error = "No chunks found"
                return trace, []

            # Combine chunk content
            combined_text = "\n\n---\n\n".join([
                f"[Section: {chunk.section_title or 'Unknown'}]\n{chunk.content}"
                for chunk in chunks
            ])

            # Limit size
            if len(combined_text) > 15000:
                combined_text = combined_text[:15000]

            # Build prompt
            prompt = SHARE_CLASS_DISCOVERY_PROMPT + f"\n\nTEXT:\n{combined_text}"

            # Call LLM with schema
            create_kwargs = {
                "response_model": ShareClassDiscovery,
                "max_retries": 2,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            }

            if self.discovery_provider != "google":
                create_kwargs["model"] = self.discovery_model

            if self.discovery_provider == "anthropic":
                create_kwargs["max_tokens"] = 2048

            result = self.instructor_client.chat.completions.create(**create_kwargs)

            # Extract class names
            discovered = result.share_class_names or []
            trace.classes_found = discovered
            trace.reasoning = result.reasoning
            trace.confidence = result.confidence.value if result.confidence else None

            # Normalize class names
            normalized = self._normalize_class_names(discovered)

            trace.duration_ms = (time.time() - start_time) * 1000

            return trace, normalized

        except Exception as e:
            logger.error(f"[Discovery] Failed: {e}")
            trace.error = str(e)
            trace.duration_ms = (time.time() - start_time) * 1000
            return trace, []

    def _extract_class_fields(
        self,
        chunked_doc: ChunkedDocument,
        class_name: str,
    ) -> tuple[PerClassExtractionTrace, Optional[ShareClassDetails]]:
        """
        Pass 2: Extract fields for a specific share class.

        Args:
            chunked_doc: The document
            class_name: The class to extract

        Returns:
            Tuple of (trace, ShareClassDetails or None)
        """
        start_time = time.time()
        trace = PerClassExtractionTrace(class_name=class_name)

        try:
            # Retrieve chunks for this class
            chunks = retrieve_chunks_for_class(chunked_doc, class_name, self.config)
            trace.chunks_retrieved = len(chunks)
            trace.retrieval_keywords = get_class_specific_keywords(class_name)[:5]

            if not chunks:
                logger.warning(f"[PerClass] No chunks for {class_name}")
                trace.error = "No chunks found"
                return trace, None

            # Combine chunk content
            combined_text = "\n\n---\n\n".join([
                f"[Section: {chunk.section_title or 'Unknown'}]\n{chunk.content}"
                for chunk in chunks
            ])

            # Limit size
            if len(combined_text) > 15000:
                combined_text = combined_text[:15000]

            # Build prompt with class name substitution
            # Note: SHARE_CLASS_FIELDS_PROMPT uses {class_name} for substitution
            # and {{text}} for the text placeholder
            prompt_template = SHARE_CLASS_FIELDS_PROMPT.format(class_name=class_name)
            prompt = prompt_template.format(text=combined_text)

            # Call LLM with schema
            create_kwargs = {
                "response_model": ShareClassFieldsExtraction,
                "max_retries": 2,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            }

            if self.provider != "google":
                create_kwargs["model"] = self.extraction_model

            if self.provider == "anthropic":
                create_kwargs["max_tokens"] = 2048

            result = self.instructor_client.chat.completions.create(**create_kwargs)

            trace.extraction_result = result.model_dump()
            trace.duration_ms = (time.time() - start_time) * 1000

            # Convert to ShareClassDetails
            share_class = ShareClassDetails(
                class_name=class_name,
                minimum_initial_investment=result.minimum_initial_investment,
                minimum_additional_investment=result.minimum_additional_investment,
                minimum_balance_for_repurchase=result.minimum_balance_for_repurchase,
                investor_eligibility=result.investor_eligibility,
                distribution_channel=result.distribution_channel,
                sales_load_pct=result.sales_load_pct,
                distribution_servicing_fee_pct=result.distribution_servicing_fee_pct,
                offering_price_basis=result.offering_price_basis,
            )

            return trace, share_class

        except Exception as e:
            logger.error(f"[PerClass] Failed for {class_name}: {e}")
            trace.error = str(e)
            trace.duration_ms = (time.time() - start_time) * 1000
            return trace, None

    def _normalize_class_names(self, class_names: list[str]) -> list[str]:
        """
        Normalize share class names to consistent format.

        Examples:
            "Class I Shares" -> "Class I"
            "I Shares" -> "Class I"
            "Institutional Class" -> "Institutional Class" (keep as-is)
        """
        normalized = []
        seen = set()

        for name in class_names:
            # Remove trailing "Shares"
            clean = re.sub(r'\s+shares?\s*$', '', name.strip(), flags=re.IGNORECASE)

            # Convert "X Shares" pattern to "Class X"
            match = re.match(r'^([A-Z])\s*$', clean, re.IGNORECASE)
            if match:
                clean = f"Class {match.group(1).upper()}"

            # Standardize "Class X" format
            match = re.match(r'^class\s+([a-z0-9]+(?:\s+[a-z]+)?)$', clean, re.IGNORECASE)
            if match:
                designation = match.group(1)
                # Capitalize each word
                designation = ' '.join(word.capitalize() for word in designation.split())
                clean = f"Class {designation}"

            # De-duplicate
            clean_lower = clean.lower()
            if clean_lower not in seen:
                seen.add(clean_lower)
                normalized.append(clean)

        return normalized


# =============================================================================
# CONVERSION TO STANDARD FORMAT
# =============================================================================


def convert_two_pass_to_share_classes_extraction(
    result: TwoPassExtractionResult,
) -> ShareClassesExtraction:
    """
    Convert TwoPassExtractionResult to standard ShareClassesExtraction format.

    Args:
        result: Two-pass extraction result

    Returns:
        ShareClassesExtraction in standard format
    """
    return ShareClassesExtraction(
        share_classes=result.share_classes,
        confidence=ConfidenceLevel.EXPLICIT if result.share_classes else ConfidenceLevel.NOT_FOUND,
    )


def convert_two_pass_to_dict(result: TwoPassExtractionResult) -> dict:
    """
    Convert TwoPassExtractionResult to dict format matching existing extraction output.

    Args:
        result: Two-pass extraction result

    Returns:
        Dict with share_classes key
    """
    return {
        "share_classes": [
            {
                "class_name": sc.class_name,
                "minimum_initial_investment": float(sc.minimum_initial_investment) if sc.minimum_initial_investment else None,
                "minimum_additional_investment": float(sc.minimum_additional_investment) if sc.minimum_additional_investment else None,
                "minimum_balance_for_repurchase": float(sc.minimum_balance_for_repurchase) if sc.minimum_balance_for_repurchase else None,
                "investor_eligibility": sc.investor_eligibility,
                "distribution_channel": sc.distribution_channel,
                "sales_load_pct": float(sc.sales_load_pct) if sc.sales_load_pct else None,
                "distribution_servicing_fee_pct": float(sc.distribution_servicing_fee_pct) if sc.distribution_servicing_fee_pct else None,
                "offering_price_basis": sc.offering_price_basis,
            }
            for sc in result.share_classes
        ]
    }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_two_pass_extractor(
    model: str = "gpt-4o-mini",
    discovery_model: Optional[str] = None,
    delay_between_calls: float = 0.5,
    requests_per_minute: int = 40,
    discovery_max_chunks: int = 20,
    per_class_max_chunks: int = 15,
) -> TwoPassShareClassExtractor:
    """
    Factory function to create a two-pass share class extractor.

    Args:
        model: LLM model for extraction
        discovery_model: Model for discovery (uses extraction model if None)
        delay_between_calls: Rate limiting delay
        requests_per_minute: Rate limit
        discovery_max_chunks: Chunks for discovery phase
        per_class_max_chunks: Chunks per class extraction

    Returns:
        Configured TwoPassShareClassExtractor
    """
    config = TwoPassConfig(
        extraction_model=model,
        discovery_model=discovery_model,
        delay_between_calls=delay_between_calls,
        requests_per_minute=requests_per_minute,
        discovery_max_chunks=discovery_max_chunks,
        per_class_max_chunks=per_class_max_chunks,
    )

    return TwoPassShareClassExtractor(config=config)


# =============================================================================
# DISCOVERY-ONLY FUNCTION
# =============================================================================


@dataclass
class DiscoveryResult:
    """Result from share class discovery (discovery-only mode)."""
    share_class_names: list[str]
    chunks_used: int
    reasoning: Optional[str] = None
    confidence: Optional[str] = None
    duration_ms: float = 0.0
    error: Optional[str] = None


def discover_share_classes(
    chunked_doc: ChunkedDocument,
    model: str = "gpt-4o-mini",
    max_chunks: int = 20,
    delay_between_calls: float = 0.5,
    requests_per_minute: int = 40,
) -> DiscoveryResult:
    """
    Discover share classes in a document (discovery-only, no field extraction).

    This is designed to run BEFORE T1 extraction to inform the pipeline
    which share classes exist in the document.

    Args:
        chunked_doc: The chunked document
        model: LLM model to use
        max_chunks: Maximum chunks to use for discovery
        delay_between_calls: Rate limiting delay
        requests_per_minute: Rate limit

    Returns:
        DiscoveryResult with list of discovered share class names

    Usage:
        # Run discovery first
        discovery = discover_share_classes(chunked_doc)

        # Pass to extraction pipeline
        extraction_context["discovered_share_classes"] = discovery.share_class_names
        # e.g., ["Class I", "Class S", "Class D", "Class R"]
    """
    config = TwoPassConfig(
        extraction_model=model,
        discovery_model=model,
        delay_between_calls=delay_between_calls,
        requests_per_minute=requests_per_minute,
        discovery_max_chunks=max_chunks,
    )

    extractor = TwoPassShareClassExtractor(config=config)

    # Run only the discovery phase
    trace, discovered_classes = extractor._discover_share_classes(chunked_doc)

    return DiscoveryResult(
        share_class_names=discovered_classes,
        chunks_used=trace.chunks_used,
        reasoning=trace.reasoning,
        confidence=trace.confidence,
        duration_ms=trace.duration_ms,
        error=trace.error,
    )
