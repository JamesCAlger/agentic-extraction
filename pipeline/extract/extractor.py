"""
LLM-based field extractor using instructor for structured output.

This module handles:
1. Selecting appropriate chunks for each field
2. Calling LLM with structured output schemas
3. Aggregating results from multiple chunks
4. Handling conflicts and confidence scoring
5. Full observability via decision tracing
"""

import logging
from typing import Optional, Type, TypeVar
from pydantic import BaseModel
import instructor

from .llm_provider import (
    create_instructor_client,
    RateLimitConfig,
    detect_provider,
    resolve_model_name,
)

from .schemas import (
    ConfidenceLevel,
    # NEW v2.0 schemas
    FundMetadataExtraction,
    LeverageLimitsExtraction,
    DistributionTermsExtraction,
    # Original schemas
    IncentiveFeeExtraction,
    ExpenseCapExtraction,
    RepurchaseTermsExtraction,
    AllocationTargetsExtraction,
    ConcentrationLimitsExtraction,
    ShareClassesExtraction,
    DocumentExtractionResult,
)
from .prompts import (
    SYSTEM_PROMPT,
    INCENTIVE_FEE_PROMPT,
    EXPENSE_CAP_PROMPT,
    REPURCHASE_TERMS_PROMPT,
    ALLOCATION_TARGETS_PROMPT,
    CONCENTRATION_LIMITS_PROMPT,
    SHARE_CLASSES_PROMPT,
    get_prompt_for_field,
    get_prompt_with_examples,
)
from ..parse.models import ChunkedDocument, ChunkedSection, Chunk
from .section_finder import extract_missing_sections
from .grounding import GroundingValidator, GroundingReport
from .validation_rules import apply_validation_rules
from .fund_classifier import classify_fund, FundStrategy
from .scoped_agentic import (
    scoped_agentic_extract,
    apply_scoped_results_to_share_classes,
    apply_scoped_lock_up_to_repurchase,
    apply_scoped_leverage_to_result,
    apply_scoped_distribution_to_result,
    apply_scoped_minimum_additional_to_share_classes,
    apply_scoped_repurchase_basis_to_result,
    # Tier 3-only mode apply functions
    apply_scoped_incentive_fee_to_result,
    apply_scoped_expense_cap_to_result,
    apply_scoped_repurchase_terms_to_result,
    apply_scoped_share_classes_to_result,
    apply_scoped_allocation_targets_to_result,
    apply_scoped_concentration_limits_to_result,
    # Reranker support
    RerankerConfig,
    # Adaptive retrieval for large documents
    get_adaptive_settings,
)
from .observability import (
    ExtractionTrace,
    ObservableExtraction,
    LayerDecision,
    DecisionType,
    MatchType,
    log_tier0_decision,
    log_tier1_decision,
    log_tier2_decision,
    log_tier3_decision,
    log_grounding_decision,
    log_flag_decision,
)


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class FieldExtractor:
    """
    Extracts structured fields from document chunks using LLM.

    Uses instructor library for reliable structured output.
    """

    # Map field names to their Pydantic schemas
    FIELD_SCHEMAS: dict[str, Type[BaseModel]] = {
        # NEW v2.0 schemas
        "fund_metadata": FundMetadataExtraction,
        "leverage_limits": LeverageLimitsExtraction,
        "distribution_terms": DistributionTermsExtraction,
        # Original schemas
        "incentive_fee": IncentiveFeeExtraction,
        "expense_cap": ExpenseCapExtraction,
        "repurchase_terms": RepurchaseTermsExtraction,
        "allocation_targets": AllocationTargetsExtraction,
        "concentration_limits": ConcentrationLimitsExtraction,
        "share_classes": ShareClassesExtraction,
    }

    # Map target fields (from segmenter) to extraction field names
    TARGET_FIELD_MAP = {
        "incentive_fee": "incentive_fee",
        "performance_fee": "incentive_fee",
        "management_fee": "incentive_fee",  # May contain incentive info
        "expense_cap": "expense_cap",
        "repurchase_frequency": "repurchase_terms",
        "repurchase_pct_nav": "repurchase_terms",
        "allocation_targets": "allocation_targets",
        "concentration_limits": "concentration_limits",
        "investment_restrictions": "concentration_limits",
        "share_classes": "share_classes",
        "minimum_investment": "share_classes",
        # NEW v2.0 target fields
        "fund_metadata": "fund_metadata",
        "leverage_limits": "leverage_limits",
        "leverage": "leverage_limits",
        "borrowing": "leverage_limits",
        "distribution_terms": "distribution_terms",
        "distributions": "distribution_terms",
        "dividend_policy": "distribution_terms",
    }

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        max_retries: int = 2,
        use_examples: bool = True,
        max_examples: int = 3,
        per_section_extraction: bool = False,
        delay_between_calls: float = 0.0,
        requests_per_minute: Optional[int] = None,
    ):
        """
        Initialize extractor.

        Args:
            model: Model to use (e.g., "gpt-4o-mini", "claude-sonnet", "gemini-flash")
            api_key: API key (or set OPENAI_API_KEY/ANTHROPIC_API_KEY/GOOGLE_API_KEY env var)
            provider: Provider name ("openai", "anthropic", "google"). Auto-detected if None.
            max_retries: Number of retries for failed extractions
            use_examples: Whether to include few-shot examples in prompts
            max_examples: Maximum number of examples per field
            per_section_extraction: If True, extract each section separately for better accuracy
            delay_between_calls: Seconds to wait between API calls (rate limiting)
            requests_per_minute: Max requests per minute (None = no limit)
        """
        self.model = resolve_model_name(model)
        self.provider = provider or detect_provider(model).value
        self.max_retries = max_retries
        self.use_examples = use_examples
        self.max_examples = max_examples
        self.per_section_extraction = per_section_extraction

        # Rate limiting config
        rate_limit = None
        if delay_between_calls > 0 or requests_per_minute:
            rate_limit = RateLimitConfig(
                delay_between_calls=delay_between_calls,
                requests_per_minute=requests_per_minute,
            )

        # Initialize instructor-wrapped client (supports OpenAI, Anthropic, Google)
        self.client = create_instructor_client(
            provider=self.provider,
            model=self.model,
            api_key=api_key,
            rate_limit=rate_limit,
        )

    def extract_field(
        self,
        field_name: str,
        chunks: list[Chunk],
        schema: Type[T],
        full_document_text: Optional[str] = None,
    ) -> Optional[T]:
        """
        Extract a single field from one or more chunks.

        Args:
            field_name: Name of the field to extract
            chunks: List of relevant chunks
            schema: Pydantic schema for the extraction
            full_document_text: Full document text for dynamic example selection

        Returns:
            Extracted value or None if not found
        """
        if not chunks:
            return None

        # Get prompt - with or without examples based on config
        if self.use_examples:
            prompt_template = get_prompt_with_examples(
                field_name,
                max_examples=self.max_examples,
                include_notes=True,
                document_text=full_document_text,  # Pass for dynamic selection
            )
        else:
            prompt_template = get_prompt_for_field(field_name)

        if not prompt_template:
            logger.warning(f"No prompt template for field: {field_name}")
            return None

        # Combine chunk content
        combined_text = self._combine_chunks(chunks)

        # Build prompt
        user_prompt = prompt_template.format(text=combined_text)

        try:
            # Build create kwargs - Gemini doesn't accept model per-call (set at client creation)
            create_kwargs = {
                "response_model": schema,
                "max_retries": self.max_retries,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            }
            # Only pass model for non-Gemini providers
            if self.provider != "google":
                create_kwargs["model"] = self.model

            # Anthropic requires max_tokens
            if self.provider == "anthropic":
                create_kwargs["max_tokens"] = 4096

            result = self.client.chat.completions.create(**create_kwargs)
            return result
        except Exception as e:
            logger.error(f"Extraction failed for {field_name}: {e}")
            return None

    def _combine_chunks(self, chunks: list[Chunk], max_tokens: int = 8000) -> str:
        """Combine multiple chunks into a single text block."""
        combined = []
        total_tokens = 0

        for chunk in chunks:
            if total_tokens + chunk.token_count > max_tokens:
                break
            combined.append(f"[Section: {chunk.section_title}]\n{chunk.content}")
            total_tokens += chunk.token_count

        return "\n\n---\n\n".join(combined)

    def extract_from_document(
        self,
        chunked_doc: ChunkedDocument,
        fund_name: str,
        full_document_text: Optional[str] = None,
    ) -> DocumentExtractionResult:
        """
        Extract all fields from a chunked document.

        Args:
            chunked_doc: Document with chunked sections
            fund_name: Name of the fund
            full_document_text: Optional full document text for dynamic example selection

        Returns:
            Complete extraction result
        """
        logger.info(f"Extracting from {chunked_doc.filing_id}")

        # Group chunks by target field
        field_chunks = self._group_chunks_by_field(chunked_doc)

        # Build full document text from chunks if not provided (for dynamic example selection)
        if full_document_text is None and self.use_examples:
            all_text_parts = []
            for section in chunked_doc.chunked_sections:
                for chunk in section.chunks:
                    all_text_parts.append(chunk.content)
            full_document_text = "\n".join(all_text_parts)[:50000]  # Limit for efficiency

        # Extract each field
        extractions = {}
        errors = []
        chunks_processed = 0

        for field_name, schema in self.FIELD_SCHEMAS.items():
            chunks = field_chunks.get(field_name, [])
            if not chunks:
                logger.debug(f"No chunks for field: {field_name}")
                continue

            logger.info(f"  Extracting {field_name} from {len(chunks)} chunks")
            chunks_processed += len(chunks)

            try:
                result = self.extract_field(
                    field_name, chunks, schema,
                    full_document_text=full_document_text
                )
                if result and result.confidence != ConfidenceLevel.NOT_FOUND:
                    extractions[field_name] = result
            except Exception as e:
                errors.append(f"{field_name}: {str(e)}")
                logger.error(f"  Error extracting {field_name}: {e}")

        return DocumentExtractionResult(
            filing_id=chunked_doc.filing_id,
            cik=chunked_doc.cik,
            fund_name=fund_name,
            # NEW v2.0 fields
            fund_metadata=extractions.get("fund_metadata"),
            leverage_limits=extractions.get("leverage_limits"),
            distribution_terms=extractions.get("distribution_terms"),
            # Original fields
            incentive_fee=extractions.get("incentive_fee"),
            expense_cap=extractions.get("expense_cap"),
            repurchase_terms=extractions.get("repurchase_terms"),
            allocation_targets=extractions.get("allocation_targets"),
            concentration_limits=extractions.get("concentration_limits"),
            share_classes=extractions.get("share_classes"),
            chunks_processed=chunks_processed,
            extraction_errors=errors,
        )

    def _group_chunks_by_field(
        self, chunked_doc: ChunkedDocument
    ) -> dict[str, list[Chunk]]:
        """Group chunks by the extraction field they support."""
        field_chunks: dict[str, list[Chunk]] = {}

        for section in chunked_doc.chunked_sections:
            # Map target fields to extraction fields
            extraction_fields = set()
            for target_field in section.target_fields:
                mapped = self.TARGET_FIELD_MAP.get(target_field)
                if mapped:
                    extraction_fields.add(mapped)

            # Add section chunks to relevant fields
            for field_name in extraction_fields:
                if field_name not in field_chunks:
                    field_chunks[field_name] = []
                field_chunks[field_name].extend(section.chunks)

        return field_chunks

    def _group_chunks_by_field_and_section(
        self, chunked_doc: ChunkedDocument
    ) -> dict[str, list[tuple[str, list[Chunk]]]]:
        """
        Group chunks by field AND section for per-section extraction.

        Returns:
            Dict mapping field_name -> [(section_id, section_chunks), ...]
        """
        field_sections: dict[str, list[tuple[str, list[Chunk]]]] = {}

        for section in chunked_doc.chunked_sections:
            # Map target fields to extraction fields
            extraction_fields = set()
            for target_field in section.target_fields:
                mapped = self.TARGET_FIELD_MAP.get(target_field)
                if mapped:
                    extraction_fields.add(mapped)

            # Add this section's chunks to each relevant field
            for field_name in extraction_fields:
                if field_name not in field_sections:
                    field_sections[field_name] = []
                # Store section_id with its chunks
                field_sections[field_name].append((section.section_id, list(section.chunks)))

        return field_sections

    def _merge_extractions(
        self,
        extractions: list[T],
        field_name: str,
    ) -> Optional[T]:
        """
        Merge multiple extractions from different sections.

        Strategy:
        - Keep extraction with highest confidence
        - For share_classes, merge lists from multiple sections
        """
        if not extractions:
            return None

        if len(extractions) == 1:
            return extractions[0]

        # Sort by confidence (EXPLICIT > INFERRED > NOT_FOUND)
        confidence_order = {
            ConfidenceLevel.EXPLICIT: 3,
            ConfidenceLevel.INFERRED: 2,
            ConfidenceLevel.NOT_FOUND: 1,
        }

        # Filter out NOT_FOUND extractions
        valid_extractions = [e for e in extractions if e.confidence != ConfidenceLevel.NOT_FOUND]
        if not valid_extractions:
            return None

        # For share_classes, merge the lists
        if field_name == "share_classes":
            return self._merge_share_classes(valid_extractions)

        # For other fields, pick highest confidence
        sorted_extractions = sorted(
            valid_extractions,
            key=lambda x: confidence_order.get(x.confidence, 0),
            reverse=True
        )
        return sorted_extractions[0]

    def _merge_share_classes(self, extractions: list) -> Optional:
        """Merge share_classes from multiple sections by class name."""
        if not extractions:
            return None

        # Collect all share classes by class_name
        merged_classes: dict = {}
        best_confidence = ConfidenceLevel.NOT_FOUND
        best_evidence = None

        for extraction in extractions:
            if extraction.confidence.value > best_confidence.value:
                best_confidence = extraction.confidence
                best_evidence = extraction.evidence_quote if hasattr(extraction, 'evidence_quote') else None

            for sc in extraction.share_classes or []:
                class_name = sc.class_name
                if class_name not in merged_classes:
                    merged_classes[class_name] = sc
                else:
                    # Merge: prefer non-None values for existing class
                    existing = merged_classes[class_name]
                    if sc.minimum_initial_investment is not None and existing.minimum_initial_investment is None:
                        existing.minimum_initial_investment = sc.minimum_initial_investment
                    if sc.minimum_additional_investment is not None and existing.minimum_additional_investment is None:
                        existing.minimum_additional_investment = sc.minimum_additional_investment
                    if hasattr(sc, 'minimum_balance_for_repurchase'):
                        if sc.minimum_balance_for_repurchase is not None and existing.minimum_balance_for_repurchase is None:
                            existing.minimum_balance_for_repurchase = sc.minimum_balance_for_repurchase
                    if hasattr(sc, 'investor_eligibility'):
                        if sc.investor_eligibility is not None and existing.investor_eligibility is None:
                            existing.investor_eligibility = sc.investor_eligibility

        # Create merged result
        return ShareClassesExtraction(
            share_classes=list(merged_classes.values()),
            confidence=best_confidence,
            evidence_quote=best_evidence,
        )

    def extract_from_document_per_section(
        self,
        chunked_doc: ChunkedDocument,
        fund_name: str,
        full_document_text: Optional[str] = None,
        processed_sections: Optional[set[str]] = None,
    ) -> tuple[DocumentExtractionResult, dict[str, list[str]]]:
        """
        Extract all fields from a chunked document using per-section approach.

        Each section is processed individually, then results are merged.
        This improves accuracy by giving LLM focused context.

        Args:
            chunked_doc: Document with chunked sections
            fund_name: Name of the fund
            full_document_text: Optional full document text for dynamic example selection
            processed_sections: Set of section_ids to skip (already processed by earlier tiers)

        Returns:
            Tuple of (extraction_result, sections_processed_per_field)
        """
        logger.info(f"Extracting from {chunked_doc.filing_id} (per-section mode)")

        # Group chunks by field AND section
        field_sections = self._group_chunks_by_field_and_section(chunked_doc)
        processed_sections = processed_sections or set()

        # Build full document text from chunks if not provided
        if full_document_text is None and self.use_examples:
            all_text_parts = []
            for section in chunked_doc.chunked_sections:
                for chunk in section.chunks:
                    all_text_parts.append(chunk.content)
            full_document_text = "\n".join(all_text_parts)[:50000]

        # Extract each field
        extractions = {}
        errors = []
        chunks_processed = 0
        sections_processed_per_field: dict[str, list[str]] = {}
        llm_calls = 0

        for field_name, schema in self.FIELD_SCHEMAS.items():
            sections_for_field = field_sections.get(field_name, [])
            if not sections_for_field:
                logger.debug(f"No sections for field: {field_name}")
                continue

            # Filter out already-processed sections
            active_sections = [
                (sid, chunks) for sid, chunks in sections_for_field
                if sid not in processed_sections
            ]

            if not active_sections:
                logger.debug(f"All sections for {field_name} already processed, skipping")
                continue

            logger.info(f"  Extracting {field_name} from {len(active_sections)} sections")
            sections_processed_per_field[field_name] = []
            section_extractions = []

            # Extract from each section individually
            for section_id, section_chunks in active_sections:
                if not section_chunks:
                    continue

                chunks_processed += len(section_chunks)
                llm_calls += 1
                sections_processed_per_field[field_name].append(section_id)

                try:
                    result = self.extract_field(
                        field_name, section_chunks, schema,
                        full_document_text=full_document_text
                    )
                    if result and result.confidence != ConfidenceLevel.NOT_FOUND:
                        section_extractions.append(result)
                except Exception as e:
                    errors.append(f"{field_name}[{section_id}]: {str(e)}")
                    logger.error(f"  Error extracting {field_name} from {section_id}: {e}")

            # Merge extractions from all sections for this field
            if section_extractions:
                merged = self._merge_extractions(section_extractions, field_name)
                if merged:
                    extractions[field_name] = merged

        logger.info(f"  Per-section extraction: {llm_calls} LLM calls for {len(extractions)} fields")

        result = DocumentExtractionResult(
            filing_id=chunked_doc.filing_id,
            cik=chunked_doc.cik,
            fund_name=fund_name,
            # NEW v2.0 fields
            fund_metadata=extractions.get("fund_metadata"),
            leverage_limits=extractions.get("leverage_limits"),
            distribution_terms=extractions.get("distribution_terms"),
            # Original fields
            incentive_fee=extractions.get("incentive_fee"),
            expense_cap=extractions.get("expense_cap"),
            repurchase_terms=extractions.get("repurchase_terms"),
            allocation_targets=extractions.get("allocation_targets"),
            concentration_limits=extractions.get("concentration_limits"),
            share_classes=extractions.get("share_classes"),
            chunks_processed=chunks_processed,
            extraction_errors=errors,
        )

        return result, sections_processed_per_field


class DocumentExtractor:
    """
    High-level document extraction orchestrator.

    Combines XBRL parsing results with LLM extraction.
    Provides full observability via ExtractionTrace.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        use_examples: bool = True,
        max_examples: int = 3,
        enable_grounding: bool = True,
        enable_observability: bool = True,
        decisions_output_dir: Optional[str] = None,
        per_section_extraction: bool = False,
        # Tier configuration
        tier0_enabled: bool = True,
        tier1_enabled: bool = True,
        tier2_enabled: bool = True,
        tier3_enabled: bool = True,
        tier3_only: bool = False,  # NEW: Tier 3 only mode (skip Tier 1 and 2)
        tier3_top_k_sections: int = 5,
        tier3_max_chunks_per_section: int = 10,
        # Reranker configuration for Tier 3
        reranker_config: Optional[RerankerConfig] = None,
        # Rate limiting
        delay_between_calls: float = 0.0,
        requests_per_minute: Optional[int] = None,
    ):
        """
        Initialize document extractor.

        Args:
            model: Model to use (e.g., "gpt-4o-mini", "claude-sonnet", "gemini-flash")
            api_key: API key (provider-specific, or use env vars)
            provider: Provider name ("openai", "anthropic", "google"). Auto-detected if None.
            use_examples: Whether to use few-shot examples
            max_examples: Maximum examples per field
            enable_grounding: Whether to validate grounding
            enable_observability: Whether to track decision traces
            decisions_output_dir: Directory to save decision traces (optional)
            per_section_extraction: If True, extract each section separately for better accuracy
            tier0_enabled: Enable Tier 0 (XBRL parsing)
            tier1_enabled: Enable Tier 1 (section mapping + LLM)
            tier2_enabled: Enable Tier 2 (regex fallback)
            tier3_enabled: Enable Tier 3 (scoped agentic search)
            tier3_only: If True, skip Tier 1+2 and run ALL fields through Tier 3
            tier3_top_k_sections: Number of top sections for Tier 3 search
            tier3_max_chunks_per_section: Max chunks per section for Tier 3
            reranker_config: Optional RerankerConfig for Cohere reranking in Tier 3
            delay_between_calls: Seconds to wait between API calls (rate limiting)
            requests_per_minute: Max requests per minute (None = no limit)
        """
        self.field_extractor = FieldExtractor(
            model=model,
            api_key=api_key,
            provider=provider,
            use_examples=use_examples,
            max_examples=max_examples,
            per_section_extraction=per_section_extraction,
            delay_between_calls=delay_between_calls,
            requests_per_minute=requests_per_minute,
        )
        self.model = resolve_model_name(model)
        self.provider = provider or detect_provider(model).value
        self.api_key = api_key
        self.delay_between_calls = delay_between_calls
        self.requests_per_minute = requests_per_minute
        self.use_examples = use_examples
        self.enable_grounding = enable_grounding
        self.enable_observability = enable_observability
        self.decisions_output_dir = decisions_output_dir
        self.per_section_extraction = per_section_extraction
        self.grounding_validator = GroundingValidator() if enable_grounding else None

        # Tier configuration
        self.tier0_enabled = tier0_enabled
        self.tier1_enabled = tier1_enabled
        self.tier2_enabled = tier2_enabled
        self.tier3_enabled = tier3_enabled
        self.tier3_only = tier3_only
        self.tier3_top_k_sections = tier3_top_k_sections
        self.tier3_max_chunks_per_section = tier3_max_chunks_per_section
        self.reranker_config = reranker_config

        # Current extraction trace (populated during extract())
        self.current_trace: Optional[ExtractionTrace] = None

        # Section tracking for per-section mode
        self.sections_processed: dict[str, list[str]] = {}

    def extract(
        self,
        chunked_doc: ChunkedDocument,
        xbrl_values: dict,
        fund_name: str,
        html_content: Optional[str] = None,
    ) -> dict:
        """
        Extract all data from a document.

        Combines:
        1. XBRL numeric values (already extracted)
        2. LLM extraction of narrative fields
        3. Fallback search for missing sections
        4. Full observability via decision tracing

        Args:
            chunked_doc: Document with chunked sections
            xbrl_values: Pre-extracted XBRL values
            fund_name: Name of the fund
            html_content: Optional raw HTML for fallback extraction

        Returns:
            Combined extraction result as dictionary
        """
        # Initialize extraction trace for observability
        if self.enable_observability:
            self.current_trace = ExtractionTrace(
                filing_id=chunked_doc.filing_id,
                fund_name=fund_name,
            )

        # Reset section tracking for this extraction
        self.sections_processed = {}

        # Classify fund strategy early for schema routing
        # Use early document text (first ~8000 chars) for classification
        classification_text = ""
        if html_content:
            classification_text = html_content[:8000]
        elif hasattr(chunked_doc, 'chunked_sections') and chunked_doc.chunked_sections:
            # Use first few sections for classification
            for section in chunked_doc.chunked_sections[:3]:
                for chunk in section.chunks[:5]:
                    classification_text += chunk.text + "\n"

        self.fund_classification = classify_fund(classification_text, fund_name)
        logger.info(
            f"  Fund classified as {self.fund_classification.strategy.value} "
            f"(confidence: {self.fund_classification.confidence:.2f})"
        )

        # Calculate document size for adaptive retrieval settings
        # ChunkedDocument has total_chunks field, but fall back to counting if not available
        if hasattr(chunked_doc, 'total_chunks') and chunked_doc.total_chunks:
            total_chunks = chunked_doc.total_chunks
        elif hasattr(chunked_doc, 'chunked_sections'):
            total_chunks = sum(len(section.chunks) for section in chunked_doc.chunked_sections)
        else:
            total_chunks = 0
        adaptive_settings = get_adaptive_settings(total_chunks)

        # Update Tier 3 settings based on document size (overrides __init__ defaults)
        if total_chunks > 500:  # Only for larger documents
            self.tier3_top_k_sections = adaptive_settings["top_k_sections"]
            self.tier3_max_chunks_per_section = adaptive_settings["max_chunks_per_section"]
            if self.reranker_config:
                self.reranker_config.first_pass_n = adaptive_settings["first_pass_n"]
                self.reranker_config.top_k = adaptive_settings["reranker_top_k"]
            logger.info(
                f"  Adaptive retrieval: {total_chunks} chunks -> "
                f"top_k={self.tier3_top_k_sections}, max_chunks={self.tier3_max_chunks_per_section}"
            )

        # Tier 3-only mode: Skip Tier 1 (LLM extraction from mapped sections)
        if self.tier3_only:
            logger.info("  [Tier 3 Only Mode] Skipping Tier 1 section-mapped extraction")
            llm_result = None
        elif self.per_section_extraction:
            # Per-section mode: extract each section individually for better accuracy
            llm_result, self.sections_processed = self.field_extractor.extract_from_document_per_section(
                chunked_doc, fund_name,
                full_document_text=html_content,
            )
        else:
            # Combined mode: extract all sections together (original behavior)
            llm_result = self.field_extractor.extract_from_document(
                chunked_doc, fund_name,
                full_document_text=html_content,
            )

        # Log Tier 1 extractions to trace
        if self.enable_observability and llm_result:
            self._log_tier1_extractions(llm_result)

        # Build initial result
        result = {
            "filing_id": chunked_doc.filing_id,
            "cik": chunked_doc.cik,
            "fund_name": fund_name,

            # Fund type classification (deterministic from N-2 checkboxes)
            "fund_type": xbrl_values.get("fund_type", "other"),
            "fund_type_flags": xbrl_values.get("fund_type_flags", {}),

            # XBRL numeric fields (already structured by share class)
            "xbrl_fees": xbrl_values.get("numeric_fields", {}),

            # NEW v2.0 LLM extractions (None when tier3_only)
            "fund_metadata": (
                llm_result.fund_metadata.model_dump()
                if llm_result and llm_result.fund_metadata else None
            ),
            "leverage_limits": (
                llm_result.leverage_limits.model_dump()
                if llm_result and llm_result.leverage_limits else None
            ),
            "distribution_terms": (
                llm_result.distribution_terms.model_dump()
                if llm_result and llm_result.distribution_terms else None
            ),

            # Original LLM extractions (None when tier3_only)
            "incentive_fee": (
                llm_result.incentive_fee.model_dump()
                if llm_result and llm_result.incentive_fee else None
            ),
            "expense_cap": (
                llm_result.expense_cap.model_dump()
                if llm_result and llm_result.expense_cap else None
            ),
            "repurchase_terms": (
                llm_result.repurchase_terms.model_dump()
                if llm_result and llm_result.repurchase_terms else None
            ),
            "allocation_targets": (
                llm_result.allocation_targets.model_dump()
                if llm_result and llm_result.allocation_targets else None
            ),
            "concentration_limits": (
                llm_result.concentration_limits.model_dump()
                if llm_result and llm_result.concentration_limits else None
            ),
            "share_classes": (
                llm_result.share_classes.model_dump()
                if llm_result and llm_result.share_classes else None
            ),

            # Metadata
            "chunks_processed": llm_result.chunks_processed if llm_result else 0,
            "extraction_errors": list(llm_result.extraction_errors) if llm_result else [],
        }

        # Fallback: search for missing sections in raw HTML using tiered search
        # Skip Tier 2 when in Tier 3-only mode
        if html_content and not self.tier3_only:
            missing_fields = []
            # Check all fields that can be recovered via fallback
            fallback_capable_fields = [
                "repurchase_terms",
                "share_classes",
                "concentration_limits",
                "expense_cap",
                # NEW v2.0 fallback-capable fields
                "leverage_limits",
                "distribution_terms",
            ]
            for field in fallback_capable_fields:
                if not result.get(field):
                    missing_fields.append(field)

            if missing_fields:
                logger.info(f"  Tiered fallback search for: {missing_fields}")
                found_sections = extract_missing_sections(
                    html_content, missing_fields, use_tiered=True
                )

                for field_name, section_text in found_sections.items():
                    if not section_text:
                        # Log that Tier 2 didn't find this field
                        if self.enable_observability:
                            self._log_tier2_extraction(field_name, found=False)
                        continue

                    logger.info(f"    Found {field_name} ({len(section_text)} chars)")
                    schema = self.field_extractor.FIELD_SCHEMAS.get(field_name)
                    if schema:
                        # Create a fake chunk for extraction
                        from ..parse.models import Chunk
                        fake_chunk = Chunk(
                            chunk_id=f"fallback_{field_name}",
                            section_id="fallback",
                            chunk_index=0,
                            content=section_text[:12000],  # Limit size
                            char_start=0,
                            char_end=len(section_text),
                            global_char_start=0,
                            global_char_end=len(section_text),
                            char_count=len(section_text),
                            token_count=len(section_text) // 4,
                            section_title=f"Fallback: {field_name}",
                            content_hash="fallback",
                        )
                        extraction = self.field_extractor.extract_field(
                            field_name, [fake_chunk], schema,
                            full_document_text=html_content,
                        )
                        if extraction and extraction.confidence != ConfidenceLevel.NOT_FOUND:
                            result[field_name] = extraction.model_dump()
                            result["chunks_processed"] += 1
                            # Log successful Tier 2 extraction
                            if self.enable_observability:
                                self._log_tier2_extraction(
                                    field_name,
                                    found=True,
                                    value=extraction.model_dump(),
                                    chars=len(section_text),
                                )

        # Tier 3: Scoped agentic search for missing minimum_investment values
        if self.tier3_enabled and result.get("share_classes"):
            share_classes_list = result["share_classes"].get("share_classes", [])
            has_missing_minimums = any(
                sc.get("minimum_initial_investment") is None
                for sc in share_classes_list
            )

            if has_missing_minimums:
                logger.info("  [Tier 3] Running scoped agentic search for minimum_investment")
                scoped_result = scoped_agentic_extract(
                    chunked_doc=chunked_doc,
                    field_name="minimum_investment",
                    api_key=self.api_key,
                    model=self.model,
                    provider=self.provider,
                    reranker_config=self.reranker_config,
                    top_k=self.tier3_top_k_sections,
                    max_chunks_per_section=self.tier3_max_chunks_per_section,
                    delay_between_calls=self.delay_between_calls,
                    requests_per_minute=self.requests_per_minute,
                )

                if scoped_result.value:
                    result = apply_scoped_results_to_share_classes(result, scoped_result)
                    logger.info(f"    [Tier 3] Applied minimum_investment from scoped search")
                    if self.enable_observability:
                        self._log_tier3_extraction(
                            "minimum_investment",
                            found=True,
                            value=scoped_result.value,
                            sections_searched=scoped_result.sections_searched,
                            top_section=scoped_result.source_section or "",
                            keyword_score=0,  # Not tracked in current implementation
                            chunks_processed=scoped_result.chunks_searched,
                        )
                else:
                    logger.info(f"    [Tier 3] No minimum_investment found in scoped search")
                    if self.enable_observability:
                        self._log_tier3_extraction(
                            "minimum_investment",
                            found=False,
                            sections_searched=scoped_result.sections_searched,
                            chunks_processed=scoped_result.chunks_searched,
                        )

        # Tier 3: Scoped agentic search for missing lock-up and early redemption
        if self.tier3_enabled and result.get("repurchase_terms"):
            repurchase = result["repurchase_terms"]
            has_missing_lockup = (
                repurchase.get("lock_up_period_years") is None and
                repurchase.get("early_repurchase_fee_pct") is None
            )

            if has_missing_lockup:
                logger.info("  [Tier 3] Running scoped agentic search for lock_up")
                scoped_result = scoped_agentic_extract(
                    chunked_doc=chunked_doc,
                    field_name="lock_up",
                    api_key=self.api_key,
                    model=self.model,
                    provider=self.provider,
                    reranker_config=self.reranker_config,
                    top_k=self.tier3_top_k_sections,
                    max_chunks_per_section=self.tier3_max_chunks_per_section,
                    delay_between_calls=self.delay_between_calls,
                    requests_per_minute=self.requests_per_minute,
                )

                if scoped_result.value:
                    result = apply_scoped_lock_up_to_repurchase(result, scoped_result)
                    logger.info(f"    [Tier 3] Applied lock_up from scoped search")
                    if self.enable_observability:
                        self._log_tier3_extraction(
                            "lock_up",
                            found=True,
                            value=scoped_result.value,
                            sections_searched=scoped_result.sections_searched,
                            top_section=scoped_result.source_section or "",
                            keyword_score=0,  # Not tracked in current implementation
                            chunks_processed=scoped_result.chunks_searched,
                        )
                else:
                    logger.info(f"    [Tier 3] No lock_up found in scoped search")
                    if self.enable_observability:
                        self._log_tier3_extraction(
                            "lock_up",
                            found=False,
                            sections_searched=scoped_result.sections_searched,
                            chunks_processed=scoped_result.chunks_searched,
                        )

        # Tier 3: Scoped agentic search for missing leverage limits
        if self.tier3_enabled and not result.get("leverage_limits"):
            logger.info("  [Tier 3] Running scoped agentic search for leverage_limits")
            scoped_result = scoped_agentic_extract(
                chunked_doc=chunked_doc,
                field_name="leverage_limits",
                api_key=self.api_key,
                model=self.model,
                provider=self.provider,
                reranker_config=self.reranker_config,
                top_k=5,
                max_chunks_per_section=10,
                delay_between_calls=self.delay_between_calls,
                requests_per_minute=self.requests_per_minute,
            )

            if scoped_result.value:
                result = apply_scoped_leverage_to_result(result, scoped_result)
                logger.info(f"    [Tier 3] Applied leverage_limits from scoped search")
                if self.enable_observability:
                    self._log_tier3_extraction(
                        "leverage_limits",
                        found=True,
                        value=scoped_result.value,
                        sections_searched=scoped_result.sections_searched,
                        top_section=scoped_result.source_section or "",
                        keyword_score=0,  # Not tracked in current implementation
                        chunks_processed=scoped_result.chunks_searched,
                    )
            else:
                logger.info(f"    [Tier 3] No leverage_limits found in scoped search")
                if self.enable_observability:
                    self._log_tier3_extraction(
                        "leverage_limits",
                        found=False,
                        sections_searched=scoped_result.sections_searched,
                        chunks_processed=scoped_result.chunks_searched,
                    )

        # Tier 3: Scoped agentic search for missing distribution_terms
        if self.tier3_enabled and not result.get("distribution_terms"):
            logger.info("  [Tier 3] Running scoped agentic search for distribution_terms")
            scoped_result = scoped_agentic_extract(
                chunked_doc=chunked_doc,
                field_name="distribution_terms",
                api_key=self.api_key,
                model=self.model,
                provider=self.provider,
                reranker_config=self.reranker_config,
                top_k=5,
                max_chunks_per_section=10,
                delay_between_calls=self.delay_between_calls,
                requests_per_minute=self.requests_per_minute,
            )

            if scoped_result.value:
                result = apply_scoped_distribution_to_result(result, scoped_result)
                logger.info(f"    [Tier 3] Applied distribution_terms from scoped search")
                if self.enable_observability:
                    self._log_tier3_extraction(
                        "distribution_terms",
                        found=True,
                        value=scoped_result.value,
                        sections_searched=scoped_result.sections_searched,
                        top_section=scoped_result.source_section or "",
                        keyword_score=0,  # Not tracked in current implementation
                        chunks_processed=scoped_result.chunks_searched,
                    )
            else:
                logger.info(f"    [Tier 3] No distribution_terms found in scoped search")
                if self.enable_observability:
                    self._log_tier3_extraction(
                        "distribution_terms",
                        found=False,
                        sections_searched=scoped_result.sections_searched,
                        chunks_processed=scoped_result.chunks_searched,
                    )

        # Tier 3: Scoped agentic search for missing minimum_additional_investment
        if self.tier3_enabled and result.get("share_classes"):
            share_classes_list = result["share_classes"].get("share_classes", [])
            has_missing_additional = any(
                sc.get("minimum_additional_investment") is None
                for sc in share_classes_list
            )

            if has_missing_additional:
                logger.info("  [Tier 3] Running scoped agentic search for minimum_additional_investment")
                scoped_result = scoped_agentic_extract(
                    chunked_doc=chunked_doc,
                    field_name="minimum_additional_investment",
                    api_key=self.api_key,
                    model=self.model,
                    provider=self.provider,
                    reranker_config=self.reranker_config,
                    top_k=self.tier3_top_k_sections,
                    max_chunks_per_section=self.tier3_max_chunks_per_section,
                    delay_between_calls=self.delay_between_calls,
                    requests_per_minute=self.requests_per_minute,
                )

                if scoped_result.value:
                    result = apply_scoped_minimum_additional_to_share_classes(result, scoped_result)
                    logger.info(f"    [Tier 3] Applied minimum_additional_investment from scoped search")
                    if self.enable_observability:
                        self._log_tier3_extraction(
                            "minimum_additional_investment",
                            found=True,
                            value=scoped_result.value,
                            sections_searched=scoped_result.sections_searched,
                            top_section=scoped_result.source_section or "",
                            keyword_score=0,  # Not tracked in current implementation
                            chunks_processed=scoped_result.chunks_searched,
                        )
                else:
                    logger.info(f"    [Tier 3] No minimum_additional_investment found in scoped search")
                    if self.enable_observability:
                        self._log_tier3_extraction(
                            "minimum_additional_investment",
                            found=False,
                            sections_searched=scoped_result.sections_searched,
                            chunks_processed=scoped_result.chunks_searched,
                        )

        # Tier 3: Scoped agentic search for missing repurchase_basis
        if self.tier3_enabled and result.get("repurchase_terms"):
            repurchase = result["repurchase_terms"]
            if repurchase.get("repurchase_basis") is None:
                logger.info("  [Tier 3] Running scoped agentic search for repurchase_basis")
                scoped_result = scoped_agentic_extract(
                    chunked_doc=chunked_doc,
                    field_name="repurchase_basis",
                    api_key=self.api_key,
                    model=self.model,
                    provider=self.provider,
                    reranker_config=self.reranker_config,
                    top_k=self.tier3_top_k_sections,
                    max_chunks_per_section=self.tier3_max_chunks_per_section,
                    delay_between_calls=self.delay_between_calls,
                    requests_per_minute=self.requests_per_minute,
                )

                if scoped_result.value:
                    result = apply_scoped_repurchase_basis_to_result(result, scoped_result)
                    logger.info(f"    [Tier 3] Applied repurchase_basis from scoped search")
                    if self.enable_observability:
                        self._log_tier3_extraction(
                            "repurchase_basis",
                            found=True,
                            value=scoped_result.value,
                            sections_searched=scoped_result.sections_searched,
                            top_section=scoped_result.source_section or "",
                            keyword_score=0,  # Not tracked in current implementation
                            chunks_processed=scoped_result.chunks_searched,
                        )
                else:
                    logger.info(f"    [Tier 3] No repurchase_basis found in scoped search")
                    if self.enable_observability:
                        self._log_tier3_extraction(
                            "repurchase_basis",
                            found=False,
                            sections_searched=scoped_result.sections_searched,
                            chunks_processed=scoped_result.chunks_searched,
                        )

        # =================================================================
        # TIER 3-ONLY MODE: Run ALL fields through Tier 3 extraction
        # =================================================================
        if self.tier3_only:
            logger.info("  [Tier 3 Only Mode] Running comprehensive Tier 3 extraction for ALL fields")

            # List of all fields to extract via Tier 3
            tier3_fields = [
                ("incentive_fee", apply_scoped_incentive_fee_to_result),
                ("expense_cap", apply_scoped_expense_cap_to_result),
                ("repurchase_terms", apply_scoped_repurchase_terms_to_result),
                ("share_classes", apply_scoped_share_classes_to_result),
                ("leverage_limits", apply_scoped_leverage_to_result),
                ("distribution_terms", apply_scoped_distribution_to_result),
                ("allocation_targets", apply_scoped_allocation_targets_to_result),
                ("concentration_limits", apply_scoped_concentration_limits_to_result),
            ]

            for field_name, apply_fn in tier3_fields:
                logger.info(f"  [Tier 3 Only] Extracting {field_name}")
                scoped_result = scoped_agentic_extract(
                    chunked_doc=chunked_doc,
                    field_name=field_name,
                    api_key=self.api_key,
                    model=self.model,
                    provider=self.provider,
                    top_k=self.tier3_top_k_sections,
                    max_chunks_per_section=self.tier3_max_chunks_per_section,
                    delay_between_calls=self.delay_between_calls,
                    requests_per_minute=self.requests_per_minute,
                    reranker_config=self.reranker_config,
                )

                if scoped_result.value:
                    result = apply_fn(result, scoped_result)
                    logger.info(f"    [Tier 3 Only] Applied {field_name} from scoped search")
                    if self.enable_observability:
                        self._log_tier3_extraction(
                            field_name,
                            found=True,
                            value=scoped_result.value,
                            sections_searched=scoped_result.sections_searched,
                            top_section=scoped_result.source_section or "",
                            keyword_score=0,
                            chunks_processed=scoped_result.chunks_searched,
                        )
                else:
                    logger.info(f"    [Tier 3 Only] No {field_name} found in scoped search")
                    if self.enable_observability:
                        self._log_tier3_extraction(
                            field_name,
                            found=False,
                            sections_searched=scoped_result.sections_searched,
                            chunks_processed=scoped_result.chunks_searched,
                        )

        # Run grounding validation if enabled
        if self.enable_grounding and html_content:
            grounding_report = self.grounding_validator.validate_extraction(
                result, html_content
            )
            result["grounding"] = grounding_report.to_dict()

            # Log grounding summary
            logger.info(
                f"  Grounding: {grounding_report.grounded_fields}/{grounding_report.total_fields} "
                f"fields grounded ({grounding_report.grounding_rate:.0%})"
            )

            # Log grounding decisions to trace
            if self.enable_observability:
                self._log_grounding_decisions(grounding_report)

            # Flag low-grounding fields for review
            for field_name, field_result in grounding_report.field_results.items():
                if not field_result.is_grounded:
                    logger.warning(
                        f"    Low grounding for {field_name}: {field_result.issues}"
                    )

            # CRITICAL: Correct hallucinated share class minimum investments
            # LLMs often hallucinate values like $2,500, $500, $1,000,000 from training data
            from .grounding import correct_ungrounded_share_class_minimums
            result, corrections = correct_ungrounded_share_class_minimums(result, html_content)
            if corrections:
                result["share_class_minimum_corrections"] = corrections

        # Apply post-extraction validation rules
        # This catches schema mismatches and cross-field inconsistencies
        # Use the classified fund strategy if available
        fund_strategy = None
        if hasattr(self, 'fund_classification') and self.fund_classification:
            fund_strategy = self.fund_classification.strategy.value
            result["fund_classification"] = self.fund_classification.to_dict()

        result, validation_report = apply_validation_rules(result, fund_strategy)
        if validation_report.correction_count > 0:
            result["validation_corrections"] = validation_report.to_dict()
            logger.info(
                f"  Validation: {validation_report.correction_count} corrections applied "
                f"(strategy: {validation_report.fund_strategy})"
            )

        # Finalize and save trace
        if self.enable_observability and self.current_trace:
            self.current_trace.mark_complete()

            # Log summary
            summary = self.current_trace.summarize()
            logger.info(
                f"  Observability: {summary['total_fields']} fields traced, "
                f"{summary['flagged_for_review']} flagged ({summary['flagged_pct']})"
            )

            # Save trace if output directory specified
            if self.decisions_output_dir:
                self.current_trace.save(self.decisions_output_dir)

            # Attach trace summary to result
            result["extraction_trace"] = {
                "summary": summary,
                "trace_available": True,
            }

        return result

    # =========================================================================
    # Observability Helper Methods
    # =========================================================================

    def _log_tier1_extractions(self, llm_result: DocumentExtractionResult):
        """Log Tier 1 (section mapping + LLM) extractions to trace."""
        if not self.current_trace:
            return

        # Map schema attributes to field names
        fields_to_check = [
            ("fund_metadata", llm_result.fund_metadata),
            ("leverage_limits", llm_result.leverage_limits),
            ("distribution_terms", llm_result.distribution_terms),
            ("incentive_fee", llm_result.incentive_fee),
            ("expense_cap", llm_result.expense_cap),
            ("repurchase_terms", llm_result.repurchase_terms),
            ("allocation_targets", llm_result.allocation_targets),
            ("concentration_limits", llm_result.concentration_limits),
            ("share_classes", llm_result.share_classes),
        ]

        for field_name, extraction in fields_to_check:
            obs = ObservableExtraction(field_name=field_name)

            if extraction is not None:
                # Get evidence quote if available
                evidence_quote = ""
                if hasattr(extraction, 'citation') and extraction.citation:
                    evidence_quote = getattr(extraction.citation, 'evidence_quote', '')

                # Get confidence
                confidence = None
                if hasattr(extraction, 'confidence'):
                    conf = extraction.confidence
                    if conf == ConfidenceLevel.EXPLICIT:
                        confidence = 1.0
                    elif conf == ConfidenceLevel.INFERRED:
                        confidence = 0.7
                    else:
                        confidence = 0.3

                obs = log_tier1_decision(
                    obs=obs,
                    found=True,
                    value=extraction.model_dump() if hasattr(extraction, 'model_dump') else extraction,
                    section_title="Mapped section",
                    confidence=confidence,
                    evidence_quote=evidence_quote,
                )
            else:
                obs = log_tier1_decision(
                    obs=obs,
                    found=False,
                    section_title="No mapped section found",
                )

            self.current_trace.add_extraction(obs)

    def _log_tier2_extraction(self, field_name: str, found: bool, value: dict = None, chars: int = 0):
        """Log a Tier 2 (regex fallback) extraction to trace."""
        if not self.current_trace:
            return

        obs = self.current_trace.get_extraction(field_name)
        if not obs:
            obs = ObservableExtraction(field_name=field_name)
            self.current_trace.add_extraction(obs)

        obs = log_tier2_decision(
            obs=obs,
            found=found,
            value=value,
            chars_extracted=chars,
            confidence=0.8 if found else None,
        )

    def _log_tier3_extraction(
        self,
        field_name: str,
        found: bool,
        value: dict = None,
        sections_searched: int = 0,
        top_section: str = "",
        keyword_score: int = 0,
        chunks_processed: int = 0,
    ):
        """Log a Tier 3 (scoped agentic) extraction to trace."""
        if not self.current_trace:
            return

        obs = self.current_trace.get_extraction(field_name)
        if not obs:
            obs = ObservableExtraction(field_name=field_name)
            self.current_trace.add_extraction(obs)

        obs = log_tier3_decision(
            obs=obs,
            found=found,
            value=value,
            sections_searched=sections_searched,
            top_section=top_section,
            keyword_score=keyword_score,
            chunks_processed=chunks_processed,
            confidence=0.75 if found else None,
        )

    def _log_grounding_decisions(self, grounding_report: GroundingReport):
        """Log grounding validation decisions to trace."""
        if not self.current_trace:
            return

        for field_name, field_result in grounding_report.field_results.items():
            obs = self.current_trace.get_extraction(field_name)
            if not obs:
                # Field wasn't tracked before, create it
                obs = ObservableExtraction(field_name=field_name)
                self.current_trace.add_extraction(obs)

            # Determine match type from grounding result
            if field_result.is_grounded:
                if field_result.grounding_score >= 0.9:
                    match_type = MatchType.EXACT
                elif field_result.grounding_score >= 0.7:
                    match_type = MatchType.FUZZY
                else:
                    match_type = MatchType.SEMANTIC
            else:
                match_type = MatchType.NONE

            # Get matched text from verified values
            matched_text = ""
            if field_result.verified_values:
                matched_text = field_result.verified_values[0]

            obs = log_grounding_decision(
                obs=obs,
                is_grounded=field_result.is_grounded,
                match_type=match_type,
                match_score=field_result.grounding_score,
                matched_text=matched_text,
                issues=field_result.issues,
            )

            # Flag ungrounded fields for review
            if not field_result.is_grounded:
                obs = log_flag_decision(
                    obs=obs,
                    reason=f"Ungrounded value: {', '.join(field_result.issues)}",
                    details={"unverified_values": field_result.unverified_values},
                )

    def get_trace(self) -> Optional[ExtractionTrace]:
        """Get the current extraction trace."""
        return self.current_trace

    def explain_field(self, field_name: str) -> str:
        """Get human-readable explanation for a specific field."""
        if not self.current_trace:
            return "Observability not enabled or no extraction performed"

        extraction = self.current_trace.get_extraction(field_name)
        if extraction:
            return extraction.explain_decision()
        return f"Field '{field_name}' not found in trace"


def extract_document(
    chunked_doc: ChunkedDocument,
    xbrl_values: dict,
    fund_name: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    html_content: Optional[str] = None,
    use_examples: bool = True,
    max_examples: int = 3,
    enable_grounding: bool = True,
    enable_observability: bool = True,
    decisions_output_dir: Optional[str] = None,
    per_section_extraction: bool = False,
    delay_between_calls: float = 0.0,
    requests_per_minute: Optional[int] = None,
) -> dict:
    """
    Convenience function to extract all data from a document.

    Args:
        chunked_doc: Document with chunked sections
        xbrl_values: Pre-extracted XBRL values
        fund_name: Name of the fund
        model: Model to use (e.g., "gpt-4o-mini", "claude-sonnet", "gemini-flash")
        api_key: API key (provider-specific)
        provider: Provider name ("openai", "anthropic", "google"). Auto-detected if None.
        html_content: Optional raw HTML for fallback section search
        use_examples: Whether to include few-shot examples in prompts
        max_examples: Maximum number of examples per field
        enable_grounding: Whether to run grounding validation
        enable_observability: Whether to track decision traces
        decisions_output_dir: Directory to save decision traces
        per_section_extraction: If True, extract each section separately for better accuracy
        delay_between_calls: Seconds to wait between API calls (rate limiting)
        requests_per_minute: Max requests per minute (None = no limit)

    Returns:
        Combined extraction result
    """
    extractor = DocumentExtractor(
        model=model,
        api_key=api_key,
        provider=provider,
        use_examples=use_examples,
        max_examples=max_examples,
        enable_grounding=enable_grounding,
        enable_observability=enable_observability,
        decisions_output_dir=decisions_output_dir,
        per_section_extraction=per_section_extraction,
        delay_between_calls=delay_between_calls,
        requests_per_minute=requests_per_minute,
    )
    return extractor.extract(chunked_doc, xbrl_values, fund_name, html_content)
