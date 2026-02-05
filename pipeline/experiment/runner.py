"""
Experiment runner for extraction pipeline.

Runs extraction with a given configuration and captures full traces
for comparison against ground truth.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .config import ExperimentConfig, load_config, save_config

logger = logging.getLogger(__name__)


@dataclass
class FundExtractionResult:
    """Result of extracting a single fund."""

    fund_name: str
    cik: str
    filing_path: str
    extraction: dict[str, Any]
    trace: Optional[dict] = None
    stats: dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class ExperimentRun:
    """Complete experiment run with all fund extractions."""

    run_id: str
    config: ExperimentConfig
    config_hash: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    fund_results: dict[str, FundExtractionResult] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "config": self.config.model_dump(),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "fund_results": {
                name: {
                    "fund_name": result.fund_name,
                    "cik": result.cik,
                    "filing_path": result.filing_path,
                    "extraction": result.extraction,
                    "trace": result.trace,
                    "stats": result.stats,
                    "error": result.error,
                    "duration_seconds": result.duration_seconds,
                }
                for name, result in self.fund_results.items()
            },
            "errors": self.errors,
            "summary": self.summarize(),
        }

    def summarize(self) -> dict:
        """Generate summary statistics."""
        total_funds = len(self.fund_results)
        successful_funds = sum(1 for r in self.fund_results.values() if r.error is None)
        total_duration = sum(r.duration_seconds for r in self.fund_results.values())

        return {
            "total_funds": total_funds,
            "successful_funds": successful_funds,
            "failed_funds": total_funds - successful_funds,
            "total_duration_seconds": total_duration,
            "errors": self.errors,
        }


class ExperimentRunner:
    """
    Runs extraction experiments with configurable parameters.

    Usage:
        runner = ExperimentRunner(config)
        run = runner.run_experiment(fund_filings)
        run.save("data/experiments/exp_20260109_baseline")
    """

    # Default fund-to-filing mapping (can be overridden)
    DEFAULT_FUND_FILINGS = {
        "Blackstone Private Multi-Asset Credit and Income Fund": {
            "path": "data/raw/0002032432/2025-03-05_0001193125-25-047335",
            "cik": "0002032432",
        },
        "StepStone Private Markets": {
            "path": "data/raw/0001789470/2025-08-19_0001193125-25-183488",
            "cik": "0001789470",
        },
        "Hamilton Lane Private Assets Fund": {
            "path": "data/raw/0001803491/2025-07-29_0001213900-25-068734",
            "cik": "0001803491",
        },
        "Blue Owl Alternative Credit Fund": {
            "path": "data/raw/0002059436/2025-08-07_0001628280-25-039019",
            "cik": "0002059436",
        },
        "Carlyle AlpInvest Private Markets Secondaries Fund": {
            "path": "data/raw/0002058263/2025-09-17_0001398344-25-018198",
            "cik": "0002058263",
        },
    }

    def __init__(
        self,
        config: ExperimentConfig,
        api_key: Optional[str] = None,
        output_dir: str = "data/experiments",
    ):
        """
        Initialize experiment runner.

        Args:
            config: ExperimentConfig with all parameters
            api_key: API key (if None, uses env var based on provider)
            output_dir: Base directory for experiment outputs
        """
        self.config = config
        # Get API key based on provider (let underlying code use env vars if not provided)
        provider = config.extraction.provider or "openai"
        if api_key:
            self.api_key = api_key
        elif provider == "anthropic":
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        elif provider == "google":
            self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
        self.output_dir = Path(output_dir)

        if not self.api_key:
            env_var = {"anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY"}.get(provider, "OPENAI_API_KEY")
            raise ValueError(f"API key required for {provider} (set {env_var} or pass api_key)")

    def _generate_run_id(self, name: Optional[str] = None) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = name or self.config.experiment.name or "baseline"
        # Clean name for filesystem
        exp_name = exp_name.replace(" ", "_").replace("/", "_")[:30]
        return f"exp_{timestamp}_{exp_name}"

    def _extract_fund(
        self,
        fund_name: str,
        filing_path: str,
        cik: str,
    ) -> FundExtractionResult:
        """
        Run extraction on a single fund.

        Args:
            fund_name: Name of the fund
            filing_path: Path to raw filing directory
            cik: Fund CIK

        Returns:
            FundExtractionResult with extraction data and trace
        """
        from time import time

        # Import here to avoid circular imports
        from ..parse.processor import process_filing
        from ..parse.chunking_strategies import (
            get_chunking_strategy,
            ChunkingStrategyType,
            ChunkingConfig as StrategyChunkingConfig,
        )
        from ..extract.extractor import DocumentExtractor

        start_time = time()
        raw_dir = Path(filing_path)

        try:
            # Create chunking strategy from config
            chunking_config = StrategyChunkingConfig(
                max_tokens=self.config.chunking.max_tokens,
                overlap_tokens=self.config.chunking.overlap_tokens,
            )

            # Get the strategy type from config
            strategy_type = ChunkingStrategyType(self.config.chunking.strategy)
            logger.info(f"Using chunking strategy: {strategy_type.value}")

            # Create LLM client for hierarchical/RAPTOR strategies if summaries are enabled
            llm_client = None
            if (
                strategy_type in [ChunkingStrategyType.HIERARCHICAL, ChunkingStrategyType.RAPTOR]
                and self.config.chunking.generate_summaries
            ):
                import openai
                llm_client = openai.OpenAI()
                logger.info(f"Using LLM for summaries: {self.config.chunking.summary_model}")

            # Create the strategy
            chunking_strategy = get_chunking_strategy(
                strategy_type=strategy_type,
                config=chunking_config,
                llm_client=llm_client,
                embedding_model=None,  # RAPTOR embedding model would need sentence-transformers
                min_paragraph_tokens=self.config.chunking.min_paragraph_tokens,
            )

            # Process document with config chunk size and strategy
            logger.info(f"Processing {fund_name} from {raw_dir}")
            doc_map, chunked_doc, xbrl_values = process_filing(
                raw_dir,
                max_chunk_tokens=self.config.extraction.max_chunk_tokens,
                overlap_tokens=self.config.extraction.overlap_tokens,
                chunking_strategy=chunking_strategy,
            )

            # Read HTML for fallback extraction
            html_path = raw_dir / "primary.html"
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Check extraction mode
            if self.config.ensemble.enabled and (
                self.config.ensemble.escalate_to_t4 or self.config.ensemble.use_multi_query
            ):
                # Ensemble mode - multi-query and/or T3 + Reranker with T4 escalation
                result, trace = self._extract_with_ensemble_t4(
                    fund_name, cik, html_content, xbrl_values, chunked_doc
                )
            elif self.config.tiers.tier4.only:
                # Tier 4 only mode - unconstrained agentic extraction
                result, trace = self._extract_with_tier4(
                    fund_name, cik, html_content, xbrl_values, chunked_doc
                )
            elif self.config.tiers.hybrid_retrieval:
                # Hybrid retrieval mode (keyword + dense with RRF fusion)
                result, trace = self._extract_with_hybrid_retrieval(
                    fund_name, cik, html_content, xbrl_values, chunked_doc
                )
            elif self.config.tiers.dense_retrieval:
                # Dense retrieval mode (embedding-based)
                result, trace = self._extract_with_dense_retrieval(
                    fund_name, cik, html_content, xbrl_values, chunked_doc
                )
            elif self.config.tiers.hybrid_boolean_tier3:
                # Hybrid: DocVQA for booleans, Tier3 for numerics
                result, trace = self._extract_with_hybrid_boolean_tier3(
                    fund_name, cik, html_content, xbrl_values, chunked_doc
                )
            elif self.config.tiers.per_datapoint_tier3_style:
                # Per-datapoint with Tier3-style prompts (isolates prompt effect)
                result, trace = self._extract_with_per_datapoint_tier3_style(
                    fund_name, cik, html_content, xbrl_values, chunked_doc
                )
            elif self.config.tiers.per_datapoint_tier3:
                # Per-datapoint extraction with granular keywords (DocVQA-style)
                result, trace = self._extract_with_per_datapoint_tier3(
                    fund_name, cik, html_content, xbrl_values, chunked_doc
                )
            elif self.config.tiers.hybrid_docvqa_tier3:
                # Hybrid: DocVQA existence + Tier3 details
                result, trace = self._extract_with_hybrid_docvqa_tier3(
                    fund_name, cik, html_content, xbrl_values, chunked_doc
                )
            elif self.config.tiers.tier3_sequential_docvqa:
                # Tier3 chunk selection + Sequential DocVQA
                result, trace = self._extract_with_tier3_sequential_docvqa(
                    fund_name, cik, html_content, xbrl_values, chunked_doc
                )
            elif self.config.tiers.sequential_docvqa:
                # Sequential conditional DocVQA (raw HTML)
                result, trace = self._extract_with_sequential_docvqa(
                    fund_name, cik, html_content, xbrl_values, chunked_doc
                )
            elif self.config.tiers.tier3_docvqa:
                # Tier 3 retrieval + DocVQA questions
                result, trace = self._extract_with_tier3_docvqa(
                    fund_name, cik, html_content, xbrl_values, chunked_doc
                )
            elif self.config.tiers.docvqa_only:
                # DocVQA-only mode
                result, trace = self._extract_with_docvqa(
                    fund_name, cik, html_content, xbrl_values, chunked_doc
                )
            else:
                # Standard tiered extraction
                result, trace = self._extract_with_tiers(
                    fund_name, html_content, xbrl_values, chunked_doc
                )

            duration = time() - start_time

            return FundExtractionResult(
                fund_name=fund_name,
                cik=cik,
                filing_path=filing_path,
                extraction=result,
                trace=trace,
                stats={
                    "sections": len(chunked_doc.chunked_sections),
                    "chunks": chunked_doc.total_chunks,
                    "tokens": chunked_doc.total_tokens,
                },
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"Error extracting {fund_name}: {e}")
            import traceback
            traceback.print_exc()

            return FundExtractionResult(
                fund_name=fund_name,
                cik=cik,
                filing_path=filing_path,
                extraction={},
                error=str(e),
                duration_seconds=time() - start_time,
            )

    def _extract_with_tiers(
        self,
        fund_name: str,
        html_content: str,
        xbrl_values: dict,
        chunked_doc,
    ) -> tuple[dict, Optional[dict]]:
        """Standard tiered extraction (T0-T3)."""
        from ..extract.extractor import DocumentExtractor
        from ..extract.scoped_agentic import RerankerConfig

        # Build reranker config from experiment config
        reranker_cfg = self.config.tiers.reranker
        reranker_config = RerankerConfig(
            enabled=reranker_cfg.enabled,
            model=reranker_cfg.model,
            first_pass_n=reranker_cfg.first_pass_n,
            top_k=reranker_cfg.top_k,
            score_threshold=reranker_cfg.score_threshold,
        ) if reranker_cfg.enabled else None

        extractor = DocumentExtractor(
            api_key=self.api_key,
            model=self.config.extraction.model,
            provider=self.config.extraction.provider,
            use_examples=self.config.extraction.use_examples,
            max_examples=self.config.extraction.max_examples,
            enable_grounding=self.config.grounding.enabled,
            enable_observability=self.config.observability.enabled,
            per_section_extraction=(
                self.config.extraction.extraction_mode == "per_section"
            ),
            # Tier configuration
            tier0_enabled=self.config.tiers.tier0_enabled,
            tier1_enabled=self.config.tiers.tier1_enabled,
            tier2_enabled=self.config.tiers.tier2_enabled,
            tier3_enabled=self.config.tiers.tier3_enabled,
            tier3_only=self.config.tiers.tier3_only,
            tier3_top_k_sections=self.config.tiers.tier3_top_k_sections,
            tier3_max_chunks_per_section=self.config.tiers.tier3_max_chunks_per_section,
            # Reranker configuration
            reranker_config=reranker_config,
            # Rate limiting
            delay_between_calls=self.config.extraction.delay_between_calls,
            requests_per_minute=self.config.extraction.requests_per_minute,
        )

        logger.info(f"Running tiered extraction for {fund_name}")
        result = extractor.extract(
            chunked_doc=chunked_doc,
            xbrl_values=xbrl_values,
            fund_name=fund_name,
            html_content=html_content,
        )

        # Get trace if observability enabled
        trace = None
        if self.config.observability.enabled and extractor.current_trace:
            trace = extractor.current_trace.to_dict()

        return result, trace

    def _extract_with_docvqa(
        self,
        fund_name: str,
        cik: str,
        html_content: str,
        xbrl_values: dict,
        chunked_doc,
    ) -> tuple[dict, Optional[dict]]:
        """DocVQA question-based extraction."""
        from ..extract.docvqa import DocVQAExtractor, convert_docvqa_to_extraction_format
        from ..parse.ixbrl_parser import IXBRLParser

        logger.info(f"Running DocVQA extraction for {fund_name}")

        extractor = DocVQAExtractor(
            api_key=self.api_key,
            model=self.config.extraction.model,
            provider=self.config.extraction.provider or "openai",
            delay_between_calls=self.config.extraction.delay_between_calls,
        )

        # Extract all fields using DocVQA
        docvqa_results = extractor.extract_all_fields(html_content)

        # Convert to standard extraction format
        extraction = convert_docvqa_to_extraction_format(docvqa_results)

        # Add fund metadata (from XBRL - Tier 0)
        filing_id = f"{cik}_{chunked_doc.chunked_sections[0].section_id.split('_')[0] if chunked_doc.chunked_sections else 'unknown'}"

        # Determine fund type from XBRL flags
        parser = IXBRLParser()
        fund_type_enum, fund_type_flags = parser.extract_fund_type(html_content)
        fund_type = fund_type_enum.value if hasattr(fund_type_enum, 'value') else str(fund_type_enum)

        result = {
            "filing_id": filing_id,
            "cik": cik,
            "fund_name": fund_name,
            "fund_type": fund_type,
            "fund_type_flags": fund_type_flags,
            "xbrl_fees": xbrl_values.get("fees", {}),  # Always include Tier 0 XBRL
            **extraction,  # DocVQA extractions
        }

        # Build trace for observability
        trace = {
            "extraction_mode": "docvqa_only",
            "fields_extracted": list(docvqa_results.keys()),
            "field_details": {
                name: {
                    "questions_asked": len(r.questions_asked),
                    "grounded_answers": sum(1 for qa in r.questions_asked if qa.grounded),
                    "confidence": r.confidence,
                    "synthesized_value": r.synthesized_value,
                }
                for name, r in docvqa_results.items()
            }
        }

        return result, trace

    def _extract_with_tier3_docvqa(
        self,
        fund_name: str,
        cik: str,
        html_content: str,
        xbrl_values: dict,
        chunked_doc,
    ) -> tuple[dict, Optional[dict]]:
        """Tier 3 retrieval + DocVQA question-based extraction."""
        from ..extract.tier3_docvqa import Tier3DocVQAExtractor, convert_tier3_docvqa_to_extraction_format
        from ..parse.ixbrl_parser import IXBRLParser

        logger.info(f"Running Tier3+DocVQA extraction for {fund_name}")

        extractor = Tier3DocVQAExtractor(
            api_key=self.api_key,
            model=self.config.extraction.model,
            provider=self.config.extraction.provider or "openai",
            top_k_sections=self.config.tiers.tier3_top_k_sections,
            max_chunks_per_section=self.config.tiers.tier3_max_chunks_per_section,
            delay_between_calls=self.config.extraction.delay_between_calls,
        )

        # Extract all fields using Tier3+DocVQA
        tier3_docvqa_results = extractor.extract_all_fields(chunked_doc)

        # Convert to standard extraction format
        extraction = convert_tier3_docvqa_to_extraction_format(tier3_docvqa_results)

        # Add fund metadata (from XBRL - Tier 0)
        filing_id = f"{cik}_{chunked_doc.chunked_sections[0].section_id.split('_')[0] if chunked_doc.chunked_sections else 'unknown'}"

        # Determine fund type from XBRL flags
        parser = IXBRLParser()
        fund_type_enum, fund_type_flags = parser.extract_fund_type(html_content)
        fund_type = fund_type_enum.value if hasattr(fund_type_enum, 'value') else str(fund_type_enum)

        result = {
            "filing_id": filing_id,
            "cik": cik,
            "fund_name": fund_name,
            "fund_type": fund_type,
            "fund_type_flags": fund_type_flags,
            "xbrl_fees": xbrl_values.get("fees", {}),  # Always include Tier 0 XBRL
            **extraction,  # Tier3+DocVQA extractions
        }

        # Build trace for observability
        trace = {
            "extraction_mode": "tier3_docvqa",
            "fields_extracted": list(tier3_docvqa_results.keys()),
            "field_details": {
                name: {
                    "sections_searched": r.sections_searched,
                    "chunks_retrieved": r.chunks_retrieved,
                    "questions_asked": r.questions_asked,
                    "grounded_answers": r.grounded_answers,
                    "confidence": r.confidence,
                    "extracted_value": r.extracted_value,
                }
                for name, r in tier3_docvqa_results.items()
            }
        }

        return result, trace

    def _extract_with_sequential_docvqa(
        self,
        fund_name: str,
        cik: str,
        html_content: str,
        xbrl_values: dict,
        chunked_doc,
    ) -> tuple[dict, Optional[dict]]:
        """Sequential conditional DocVQA extraction."""
        from ..extract.sequential_docvqa import (
            SequentialDocVQAExtractor,
            convert_sequential_docvqa_to_extraction_format
        )
        from ..parse.ixbrl_parser import IXBRLParser

        logger.info(f"Running Sequential DocVQA extraction for {fund_name}")

        extractor = SequentialDocVQAExtractor(
            api_key=self.api_key,
            model=self.config.extraction.model,
            provider=self.config.extraction.provider or "openai",
            delay_between_calls=self.config.extraction.delay_between_calls,
        )

        # Extract all fields using Sequential DocVQA
        seq_docvqa_results = extractor.extract_all_fields(html_content)

        # Convert to standard extraction format
        extraction = convert_sequential_docvqa_to_extraction_format(seq_docvqa_results)

        # Add fund metadata (from XBRL - Tier 0)
        filing_id = f"{cik}_{chunked_doc.chunked_sections[0].section_id.split('_')[0] if chunked_doc.chunked_sections else 'unknown'}"

        # Determine fund type from XBRL flags
        parser = IXBRLParser()
        fund_type_enum, fund_type_flags = parser.extract_fund_type(html_content)
        fund_type = fund_type_enum.value if hasattr(fund_type_enum, 'value') else str(fund_type_enum)

        result = {
            "filing_id": filing_id,
            "cik": cik,
            "fund_name": fund_name,
            "fund_type": fund_type,
            "fund_type_flags": fund_type_flags,
            "xbrl_fees": xbrl_values.get("fees", {}),  # Always include Tier 0 XBRL
            **extraction,  # Sequential DocVQA extractions
        }

        # Build trace for observability
        trace = {
            "extraction_mode": "sequential_docvqa",
            "fields_extracted": list(seq_docvqa_results.keys()),
            "field_details": {
                name: {
                    "existence_check": {
                        "exists": r.existence_check.exists,
                        "evidence": r.existence_check.evidence,
                        "confidence": r.existence_check.confidence,
                    },
                    "questions_asked": r.questions_asked,
                    "grounded_answers": r.grounded_answers,
                    "confidence": r.confidence,
                    "grounded": r.grounded,
                    "synthesized_value": r.synthesized_value,
                }
                for name, r in seq_docvqa_results.items()
            }
        }

        return result, trace

    def _extract_with_tier3_sequential_docvqa(
        self,
        fund_name: str,
        cik: str,
        html_content: str,
        xbrl_values: dict,
        chunked_doc,
    ) -> tuple[dict, Optional[dict]]:
        """Tier3 chunk selection + Sequential DocVQA extraction."""
        from ..extract.sequential_docvqa import (
            Tier3SequentialDocVQAExtractor,
            convert_tier3_sequential_to_extraction_format
        )
        from ..parse.ixbrl_parser import IXBRLParser

        logger.info(f"Running Tier3+Sequential DocVQA extraction for {fund_name}")

        extractor = Tier3SequentialDocVQAExtractor(
            api_key=self.api_key,
            model=self.config.extraction.model,
            provider=self.config.extraction.provider or "openai",
            top_k_sections=self.config.tiers.tier3_top_k_sections,
            max_chunks_per_section=self.config.tiers.tier3_max_chunks_per_section,
            delay_between_calls=self.config.extraction.delay_between_calls,
        )

        # Extract all fields using Tier3+Sequential DocVQA
        results = extractor.extract_all_fields(chunked_doc)

        # Convert to standard extraction format
        extraction = convert_tier3_sequential_to_extraction_format(results)

        # Add fund metadata (from XBRL - Tier 0)
        filing_id = f"{cik}_{chunked_doc.chunked_sections[0].section_id.split('_')[0] if chunked_doc.chunked_sections else 'unknown'}"

        # Determine fund type from XBRL flags
        parser = IXBRLParser()
        fund_type_enum, fund_type_flags = parser.extract_fund_type(html_content)
        fund_type = fund_type_enum.value if hasattr(fund_type_enum, 'value') else str(fund_type_enum)

        result = {
            "filing_id": filing_id,
            "cik": cik,
            "fund_name": fund_name,
            "fund_type": fund_type,
            "fund_type_flags": fund_type_flags,
            "xbrl_fees": xbrl_values.get("fees", {}),
            **extraction,
        }

        # Build trace for observability
        trace = {
            "extraction_mode": "tier3_sequential_docvqa",
            "fields_extracted": list(results.keys()),
            "field_details": {
                name: {
                    "sections_searched": r.sections_searched,
                    "chunks_retrieved": r.chunks_retrieved,
                    "existence_check": {
                        "exists": r.existence_check.exists,
                        "evidence": r.existence_check.evidence,
                        "confidence": r.existence_check.confidence,
                    },
                    "questions_asked": r.questions_asked,
                    "grounded_answers": r.grounded_answers,
                    "confidence": r.confidence,
                    "grounded": r.grounded,
                    "synthesized_value": r.synthesized_value,
                }
                for name, r in results.items()
            }
        }

        return result, trace

    def _extract_with_hybrid_docvqa_tier3(
        self,
        fund_name: str,
        cik: str,
        html_content: str,
        xbrl_values: dict,
        chunked_doc,
    ) -> tuple[dict, Optional[dict]]:
        """Hybrid extraction: DocVQA existence + Tier3 details."""
        from ..extract.hybrid_docvqa_tier3 import (
            HybridDocVQATier3Extractor,
            convert_hybrid_to_extraction_format
        )
        from ..parse.ixbrl_parser import IXBRLParser

        logger.info(f"Running Hybrid DocVQA+Tier3 extraction for {fund_name}")

        extractor = HybridDocVQATier3Extractor(
            api_key=self.api_key,
            model=self.config.extraction.model,
            provider=self.config.extraction.provider or "openai",
            top_k_sections=self.config.tiers.tier3_top_k_sections,
            max_chunks_per_section=self.config.tiers.tier3_max_chunks_per_section,
            delay_between_calls=self.config.extraction.delay_between_calls,
        )

        # Extract all fields using hybrid approach
        results = extractor.extract_all_fields(chunked_doc)

        # Convert to standard extraction format
        extraction = convert_hybrid_to_extraction_format(results)

        # Add fund metadata (from XBRL - Tier 0)
        filing_id = f"{cik}_{chunked_doc.chunked_sections[0].section_id.split('_')[0] if chunked_doc.chunked_sections else 'unknown'}"

        # Determine fund type from XBRL flags
        parser = IXBRLParser()
        fund_type_enum, fund_type_flags = parser.extract_fund_type(html_content)
        fund_type = fund_type_enum.value if hasattr(fund_type_enum, 'value') else str(fund_type_enum)

        result = {
            "filing_id": filing_id,
            "cik": cik,
            "fund_name": fund_name,
            "fund_type": fund_type,
            "fund_type_flags": fund_type_flags,
            "xbrl_fees": xbrl_values.get("fees", {}),
            **extraction,
        }

        # Build trace for observability
        total_llm_calls = sum(r.llm_calls for r in results.values())
        trace = {
            "extraction_mode": "hybrid_docvqa_tier3",
            "total_llm_calls": total_llm_calls,
            "fields_extracted": list(results.keys()),
            "field_details": {
                name: {
                    "sections_searched": r.sections_searched,
                    "chunks_retrieved": r.chunks_retrieved,
                    "existence_result": r.existence_result,
                    "existence_evidence": r.existence_evidence,
                    "extraction_method": r.extraction_method,
                    "llm_calls": r.llm_calls,
                    "extraction_result": r.extraction_result,
                }
                for name, r in results.items()
            }
        }

        return result, trace

    def _extract_with_per_datapoint_tier3(
        self,
        fund_name: str,
        cik: str,
        html_content: str,
        xbrl_values: dict,
        chunked_doc,
    ) -> tuple[dict, Optional[dict]]:
        """Per-datapoint extraction with granular keywords."""
        from ..extract.per_datapoint_extractor import PerDatapointExtractor
        from ..parse.ixbrl_parser import IXBRLParser

        logger.info(f"Running Per-Datapoint Tier3 extraction for {fund_name}")

        extractor = PerDatapointExtractor(
            model=self.config.extraction.model,
            provider=self.config.extraction.provider or "openai",
            api_key=self.api_key,
            delay_between_calls=self.config.extraction.delay_between_calls,
            requests_per_minute=self.config.extraction.requests_per_minute or 60,
            top_k_chunks=10,
        )

        # Extract all datapoints
        extraction, trace_obj = extractor.extract_all(chunked_doc, fund_name)

        # Add fund metadata (from XBRL - Tier 0)
        filing_id = f"{cik}_{chunked_doc.chunked_sections[0].section_id.split('_')[0] if chunked_doc.chunked_sections else 'unknown'}"

        # Determine fund type from XBRL flags
        parser = IXBRLParser()
        fund_type_enum, fund_type_flags = parser.extract_fund_type(html_content)
        fund_type = fund_type_enum.value if hasattr(fund_type_enum, 'value') else str(fund_type_enum)

        result = {
            "filing_id": filing_id,
            "cik": cik,
            "fund_name": fund_name,
            "fund_type": fund_type,
            "fund_type_flags": fund_type_flags,
            "xbrl_fees": xbrl_values.get("fees", {}),
            **extraction,
        }

        # Build trace for observability
        trace = {
            "extraction_mode": "per_datapoint_tier3",
            "total_datapoints": trace_obj.total_datapoints,
            "successful_extractions": trace_obj.successful_extractions,
            "total_chunks_searched": trace_obj.total_chunks_searched,
            "total_extraction_time_ms": trace_obj.total_extraction_time_ms,
            "datapoint_details": {
                name: {
                    "value": r.value,
                    "evidence": r.evidence,
                    "chunks_searched": r.chunks_searched,
                    "top_chunk_score": r.top_chunk_score,
                    "extraction_time_ms": r.extraction_time_ms,
                }
                for name, r in trace_obj.datapoint_results.items()
            }
        }

        return result, trace

    def _extract_with_per_datapoint_tier3_style(
        self,
        fund_name: str,
        cik: str,
        html_content: str,
        xbrl_values: dict,
        chunked_doc,
    ) -> tuple[dict, Optional[dict]]:
        """Per-datapoint extraction with Tier3-style prompts (isolates prompt effect)."""
        from ..extract.per_datapoint_tier3_style import PerDatapointTier3StyleExtractor
        from ..parse.ixbrl_parser import IXBRLParser

        logger.info(f"Running Per-Datapoint Tier3-Style extraction for {fund_name}")

        extractor = PerDatapointTier3StyleExtractor(
            model=self.config.extraction.model,
            provider=self.config.extraction.provider or "openai",
            api_key=self.api_key,
            delay_between_calls=self.config.extraction.delay_between_calls,
            requests_per_minute=self.config.extraction.requests_per_minute or 60,
            top_k_chunks=10,
            max_retries=self.config.extraction.max_retries,
        )

        # Extract all datapoints
        extraction, trace_obj = extractor.extract_all(chunked_doc, fund_name)

        # Add fund metadata (from XBRL - Tier 0)
        filing_id = f"{cik}_{chunked_doc.chunked_sections[0].section_id.split('_')[0] if chunked_doc.chunked_sections else 'unknown'}"

        # Determine fund type from XBRL flags
        parser = IXBRLParser()
        fund_type_enum, fund_type_flags = parser.extract_fund_type(html_content)
        fund_type = fund_type_enum.value if hasattr(fund_type_enum, 'value') else str(fund_type_enum)

        result = {
            "filing_id": filing_id,
            "cik": cik,
            "fund_name": fund_name,
            "fund_type": fund_type,
            "fund_type_flags": fund_type_flags,
            "xbrl_fees": xbrl_values.get("fees", {}),
            **extraction,
        }

        # Build trace for observability
        trace = {
            "extraction_mode": "per_datapoint_tier3_style",
            "total_datapoints": trace_obj.total_datapoints,
            "successful_extractions": trace_obj.successful_extractions,
            "total_chunks_searched": trace_obj.total_chunks_searched,
            "total_extraction_time_ms": trace_obj.total_extraction_time_ms,
            "datapoint_details": {
                name: {
                    "value": r.value,
                    "evidence": r.evidence,
                    "confidence": r.confidence,
                    "chunks_searched": r.chunks_searched,
                    "top_chunk_score": r.top_chunk_score,
                    "extraction_time_ms": r.extraction_time_ms,
                }
                for name, r in trace_obj.datapoint_results.items()
            }
        }

        return result, trace

    def _extract_with_dense_retrieval(
        self,
        fund_name: str,
        cik: str,
        html_content: str,
        xbrl_values: dict,
        chunked_doc,
    ) -> tuple[dict, Optional[dict]]:
        """Dense retrieval extraction using embeddings."""
        from ..extract.dense_retrieval_extractor import DenseRetrievalExtractor
        from ..parse.ixbrl_parser import IXBRLParser

        logger.info(f"Running Dense Retrieval extraction for {fund_name}")

        # Get embedding config
        emb_config = self.config.tiers.embedding_retrieval

        extractor = DenseRetrievalExtractor(
            model=self.config.extraction.model,
            provider=self.config.extraction.provider or "openai",
            api_key=self.api_key,
            delay_between_calls=self.config.extraction.delay_between_calls,
            requests_per_minute=self.config.extraction.requests_per_minute or 60,
            max_retries=self.config.extraction.max_retries,
            # Embedding settings from config
            embedding_provider=emb_config.provider,
            embedding_model=emb_config.model,
            prepend_context=emb_config.prepend_context,
            cache_embeddings=emb_config.cache_embeddings,
            top_k_chunks=emb_config.top_k,
        )

        # Extract all datapoints using dense retrieval
        extraction, trace_obj = extractor.extract_all(chunked_doc, fund_name)

        # Add fund metadata (from XBRL - Tier 0)
        filing_id = f"{cik}_{chunked_doc.chunked_sections[0].section_id.split('_')[0] if chunked_doc.chunked_sections else 'unknown'}"

        # Determine fund type from XBRL flags
        parser = IXBRLParser()
        fund_type_enum, fund_type_flags = parser.extract_fund_type(html_content)
        fund_type = fund_type_enum.value if hasattr(fund_type_enum, 'value') else str(fund_type_enum)

        result = {
            "filing_id": filing_id,
            "cik": cik,
            "fund_name": fund_name,
            "fund_type": fund_type,
            "fund_type_flags": fund_type_flags,
            "xbrl_fees": xbrl_values.get("fees", {}),
            **extraction,
        }

        # Build trace for observability
        trace = {
            "extraction_mode": "dense_retrieval",
            "embedding_provider": trace_obj.embedding_provider,
            "embedding_model": trace_obj.embedding_model,
            "total_datapoints": trace_obj.total_datapoints,
            "successful_extractions": trace_obj.successful_extractions,
            "total_chunks_retrieved": trace_obj.total_chunks_retrieved,
            "total_time_ms": trace_obj.total_time_ms,
            "datapoint_details": {
                name: {
                    "value": r.value,
                    "evidence": r.evidence,
                    "confidence": r.confidence,
                    "top_similarity_score": r.top_similarity_score,
                    "chunks_retrieved": r.chunks_retrieved,
                    "retrieval_time_ms": r.retrieval_time_ms,
                    "extraction_time_ms": r.extraction_time_ms,
                }
                for name, r in trace_obj.datapoint_results.items()
            }
        }

        return result, trace

    def _extract_with_hybrid_retrieval(
        self,
        fund_name: str,
        cik: str,
        html_content: str,
        xbrl_values: dict,
        chunked_doc,
    ) -> tuple[dict, Optional[dict]]:
        """Hybrid retrieval: union of keyword + dense with RRF fusion."""
        from ..extract.hybrid_extractor import create_hybrid_extractor
        from ..extract.embedding_retriever import EmbeddingConfig
        from ..extract.hybrid_retriever import HybridConfig
        from ..parse.ixbrl_parser import IXBRLParser

        logger.info(f"Running Hybrid Retrieval extraction for {fund_name}")

        # Get embedding config from experiment config
        emb_config = self.config.tiers.embedding_retrieval

        # Create embedding config
        embedding_config = EmbeddingConfig(
            provider=emb_config.provider,
            model=emb_config.model,
            top_k=emb_config.top_k * 2,  # Retrieve more for fusion
            prepend_context=emb_config.prepend_context,
            cache_embeddings=emb_config.cache_embeddings,
        )

        # Create hybrid config
        hybrid_config = HybridConfig(
            rrf_k=60,
            keyword_top_k=emb_config.top_k * 2,
            dense_top_k=emb_config.top_k * 2,
            final_top_k=emb_config.top_k,
        )

        extractor = create_hybrid_extractor(
            model=self.config.extraction.model,
            embedding_provider=emb_config.provider,
            embedding_model=emb_config.model,
            top_k=emb_config.top_k,
        )

        # Extract all fields
        extraction_result = extractor.extract(chunked_doc, fund_name=fund_name)

        # Add fund metadata (from XBRL - Tier 0)
        parser = IXBRLParser()
        fund_type_enum, fund_type_flags = parser.extract_fund_type(html_content)
        fund_type = fund_type_enum.value if hasattr(fund_type_enum, 'value') else str(fund_type_enum)

        result = extraction_result.extraction
        result["filing_id"] = chunked_doc.filing_id
        result["cik"] = cik
        result["fund_name"] = fund_name
        result["fund_type"] = fund_type
        result["fund_type_flags"] = fund_type_flags
        result["xbrl_fees"] = xbrl_values

        # Build trace
        trace = {
            "extraction_mode": "hybrid_retrieval",
            "embedding_provider": emb_config.provider,
            "embedding_model": emb_config.model,
            "total_time_s": extraction_result.total_time_s,
            "retrieval_stats": extraction_result.retrieval_stats,
            "field_traces": {
                field: {
                    "query": t.query,
                    "chunks_retrieved": t.chunks_retrieved,
                    "retrieval_sources": t.retrieval_sources,
                    "top_chunk_scores": t.top_chunk_scores,
                    "extraction_time_ms": t.extraction_time_ms,
                    "error": t.error,
                }
                for field, t in extraction_result.traces.items()
            }
        }

        return result, trace

    def _extract_with_hybrid_boolean_tier3(
        self,
        fund_name: str,
        cik: str,
        html_content: str,
        xbrl_values: dict,
        chunked_doc,
    ) -> tuple[dict, Optional[dict]]:
        """Hybrid extraction: DocVQA for booleans, Tier3 for numerics."""
        from ..extract.hybrid_boolean_tier3 import HybridBooleanTier3Extractor
        from ..parse.ixbrl_parser import IXBRLParser

        logger.info(f"Running Hybrid Boolean+Tier3 extraction for {fund_name}")

        extractor = HybridBooleanTier3Extractor(
            model=self.config.extraction.model,
            provider=self.config.extraction.provider or "openai",
            api_key=self.api_key,
            delay_between_calls=self.config.extraction.delay_between_calls,
            requests_per_minute=self.config.extraction.requests_per_minute or 60,
            top_k_chunks=10,
            max_retries=self.config.extraction.max_retries,
        )

        # Extract all datapoints using hybrid approach
        extraction, trace_obj = extractor.extract_all(chunked_doc, fund_name)

        # Add fund metadata (from XBRL - Tier 0)
        filing_id = f"{cik}_{chunked_doc.chunked_sections[0].section_id.split('_')[0] if chunked_doc.chunked_sections else 'unknown'}"

        # Determine fund type from XBRL flags
        parser = IXBRLParser()
        fund_type_enum, fund_type_flags = parser.extract_fund_type(html_content)
        fund_type = fund_type_enum.value if hasattr(fund_type_enum, 'value') else str(fund_type_enum)

        result = {
            "filing_id": filing_id,
            "cik": cik,
            "fund_name": fund_name,
            "fund_type": fund_type,
            "fund_type_flags": fund_type_flags,
            "xbrl_fees": xbrl_values.get("fees", {}),
            **extraction,
        }

        # Build trace for observability
        trace = {
            "extraction_mode": "hybrid_boolean_tier3",
            "total_datapoints": trace_obj.total_datapoints,
            "successful_extractions": trace_obj.successful_extractions,
            "docvqa_extractions": trace_obj.docvqa_extractions,
            "tier3_extractions": trace_obj.tier3_extractions,
            "total_chunks_searched": trace_obj.total_chunks_searched,
            "total_extraction_time_ms": trace_obj.total_extraction_time_ms,
            "datapoint_details": {
                name: {
                    "value": r.value,
                    "evidence": r.evidence,
                    "confidence": r.confidence,
                    "extraction_style": r.extraction_style,
                    "chunks_searched": r.chunks_searched,
                    "top_chunk_score": r.top_chunk_score,
                    "extraction_time_ms": r.extraction_time_ms,
                }
                for name, r in trace_obj.datapoint_results.items()
            }
        }

        return result, trace

    def _extract_with_ensemble_t4(
        self,
        fund_name: str,
        cik: str,
        html_content: str,
        xbrl_values: dict,
        chunked_doc,
    ) -> tuple[dict, Optional[dict]]:
        """
        Ensemble extraction with T4 escalation.

        Runs T3 k=10 and Reranker 200->10, compares results, and escalates
        disagreements and both-null cases to T4 agentic extraction.

        This mode provides a balance between accuracy (T4) and cost (accept when agree).

        NOTE: T4 changes in tier4_agentic.py automatically flow through to this
        method since EnsembleT4Extractor imports and uses Tier4Agent directly.

        Args:
            fund_name: Name of the fund
            cik: CIK identifier
            html_content: Raw HTML content
            xbrl_values: Pre-extracted XBRL values
            chunked_doc: Chunked document

        Returns:
            Tuple of (extraction_result, trace_dict)
        """
        from ..extract.ensemble_t4_extractor import (
            EnsembleT4Extractor,
            EnsembleT4Config,
            AdversarialValidationConfig,
        )
        from ..parse.ixbrl_parser import IXBRLParser

        logger.info(f"Running Ensemble T4 extraction for {fund_name}")

        # Build config from experiment settings
        ensemble_cfg = self.config.ensemble
        extraction_cfg = self.config.extraction
        reranker_cfg = self.config.tiers.reranker

        config = EnsembleT4Config(
            # T3 settings
            t3_top_k_sections=ensemble_cfg.t3_top_k_sections,
            t3_max_chunks_per_section=self.config.tiers.tier3_max_chunks_per_section,
            # Reranker settings
            reranker_first_pass_n=ensemble_cfg.reranker_first_pass_n,
            reranker_top_k=ensemble_cfg.reranker_top_k,
            reranker_model=reranker_cfg.model if reranker_cfg.enabled else "rerank-v3.5",
            reranker_score_threshold=reranker_cfg.score_threshold,
            # T4 settings
            t4_model=ensemble_cfg.t4_model,
            t4_max_iterations=ensemble_cfg.t4_max_iterations,
            t4_timeout_seconds=ensemble_cfg.t4_timeout_seconds,
            # LLM settings
            extraction_model=extraction_cfg.model,
            extraction_provider=extraction_cfg.provider or "openai",
            delay_between_calls=extraction_cfg.delay_between_calls,
            requests_per_minute=extraction_cfg.requests_per_minute or 40,
            # Escalation behavior
            run_t4_escalation=ensemble_cfg.escalate_to_t4,  # Whether to actually run T4 on escalated fields
            escalate_on_disagreement=ensemble_cfg.escalate_on_disagreement,
            escalate_on_both_null=ensemble_cfg.escalate_on_both_null,
            # Hybrid retrieval settings
            use_hybrid_retrieval=ensemble_cfg.use_hybrid_retrieval,
            hybrid_embedding_provider=ensemble_cfg.hybrid_embedding_provider,
            hybrid_embedding_model=ensemble_cfg.hybrid_embedding_model,
            hybrid_rrf_k=ensemble_cfg.hybrid_rrf_k,
            hybrid_keyword_top_k=ensemble_cfg.hybrid_keyword_top_k,
            hybrid_dense_top_k=ensemble_cfg.hybrid_dense_top_k,
            hybrid_final_top_k=ensemble_cfg.hybrid_final_top_k,
            # Multi-query expansion settings
            use_multi_query=ensemble_cfg.use_multi_query,
            multi_query_expansion_method=ensemble_cfg.multi_query_expansion_method,
            multi_query_retrieval_strategy=ensemble_cfg.multi_query_retrieval_strategy,
            multi_query_rrf_k=ensemble_cfg.multi_query_rrf_k,
            multi_query_per_query_top_k=ensemble_cfg.multi_query_per_query_top_k,
            multi_query_final_top_k=ensemble_cfg.multi_query_final_top_k,
            multi_query_embedding_provider=ensemble_cfg.multi_query_embedding_provider,
            multi_query_embedding_model=ensemble_cfg.multi_query_embedding_model,
            multi_query_holistic_extraction=ensemble_cfg.multi_query_holistic_extraction,
            # Ensemble method selection
            methods=ensemble_cfg.methods,
            # Hybrid routing: prefer specific method per field type
            use_hybrid_routing=ensemble_cfg.use_hybrid_routing,
            # Confidence-based routing: use confidence scores to pick best method
            use_confidence_routing=ensemble_cfg.use_confidence_routing,
            confidence_min_gap=ensemble_cfg.confidence_min_gap,
            confidence_low_threshold=ensemble_cfg.confidence_low_threshold,
            confidence_high_threshold=ensemble_cfg.confidence_high_threshold,
            confidence_require_grounding=ensemble_cfg.confidence_require_grounding,
            # Grounding strategy settings for confidence routing
            grounding_strategy=ensemble_cfg.grounding_strategy,
            grounding_nli_model=ensemble_cfg.grounding_nli_model,
            grounding_nli_entailment_threshold=ensemble_cfg.grounding_nli_entailment_threshold,
            grounding_nli_contradiction_threshold=ensemble_cfg.grounding_nli_contradiction_threshold,
            grounding_llm_judge_model=ensemble_cfg.grounding_llm_judge_model,
            grounding_llm_judge_provider=ensemble_cfg.grounding_llm_judge_provider,
            grounding_hybrid_nli_high_threshold=ensemble_cfg.grounding_hybrid_nli_high_threshold,
            grounding_hybrid_nli_low_threshold=ensemble_cfg.grounding_hybrid_nli_low_threshold,
            grounding_hybrid_use_llm_for_ambiguous=ensemble_cfg.grounding_hybrid_use_llm_for_ambiguous,
            # Two-pass share class extraction settings (DEPRECATED)
            share_class_two_pass_enabled=ensemble_cfg.share_class_two_pass.enabled,
            share_class_two_pass_discovery_model=ensemble_cfg.share_class_two_pass.discovery_model,
            share_class_two_pass_discovery_max_chunks=ensemble_cfg.share_class_two_pass.discovery_max_chunks,
            share_class_two_pass_per_class_max_chunks=ensemble_cfg.share_class_two_pass.per_class_max_chunks,
            # Discovery-first share class extraction settings (RECOMMENDED)
            share_class_discovery_first_enabled=ensemble_cfg.share_class_discovery_first.enabled,
            share_class_discovery_model=ensemble_cfg.share_class_discovery_first.model,
            share_class_discovery_max_chunks=ensemble_cfg.share_class_discovery_first.max_chunks,
            # Adversarial validation settings
            adversarial_validation=AdversarialValidationConfig(
                enabled=ensemble_cfg.adversarial_validation_enabled,
                lightweight=ensemble_cfg.adversarial_validation_lightweight,
                validate_booleans=ensemble_cfg.adversarial_validation_validate_booleans,
                validate_all=ensemble_cfg.adversarial_validation_validate_all,
                escalate_on_rejection=ensemble_cfg.adversarial_validation_escalate_on_rejection,
            ),
        )

        extractor = EnsembleT4Extractor(config=config, api_key=self.api_key)

        # Run ensemble extraction
        ensemble_result, ensemble_trace = extractor.extract(
            chunked_doc=chunked_doc,
            xbrl_values=xbrl_values,
            fund_name=fund_name,
            html_content=html_content,
        )

        # Build result dict
        result = ensemble_result.extraction

        # Add fund metadata
        filing_id = f"{cik}_{chunked_doc.chunked_sections[0].section_id.split('_')[0] if chunked_doc.chunked_sections else 'unknown'}"
        parser = IXBRLParser()
        fund_type_enum, fund_type_flags = parser.extract_fund_type(html_content)
        fund_type = fund_type_enum.value if hasattr(fund_type_enum, 'value') else str(fund_type_enum)

        result["filing_id"] = filing_id
        result["cik"] = cik
        result["fund_name"] = fund_name
        result["fund_type"] = fund_type
        result["fund_type_flags"] = fund_type_flags
        result["xbrl_fees"] = xbrl_values.get("fees", {})

        # Build trace for observability
        trace = {
            "extraction_mode": "ensemble_t4",
            "config": ensemble_trace.config,
            "statistics": {
                "total_fields": ensemble_result.total_fields,
                "accepted_count": ensemble_result.accepted_count,
                "escalated_count": ensemble_result.escalated_count,
                "t4_success_count": ensemble_result.t4_success_count,
                "agreement_rate": ensemble_trace.agreement_rate,
                "escalation_rate": ensemble_trace.escalation_rate,
                "t4_success_rate": ensemble_trace.t4_success_rate,
            },
            "timing": {
                "t3_duration_seconds": ensemble_result.t3_duration_seconds,
                "reranker_duration_seconds": ensemble_result.reranker_duration_seconds,
                "t4_duration_seconds": ensemble_result.t4_duration_seconds,
                "total_duration_seconds": ensemble_result.total_duration_seconds,
            },
            "field_decisions": ensemble_trace.field_decisions,
        }

        return result, trace

    def _extract_with_tier4(
        self,
        fund_name: str,
        cik: str,
        html_content: str,
        xbrl_values: dict,
        chunked_doc,
        fields_to_extract: Optional[list[str]] = None,
    ) -> tuple[dict, Optional[dict]]:
        """
        Tier 4 unconstrained agentic extraction.

        Can be used standalone or as fallback for fields that failed in Tiers 0-3.

        Args:
            fund_name: Name of the fund
            cik: CIK identifier
            html_content: Raw HTML content
            xbrl_values: Pre-extracted XBRL values
            chunked_doc: Chunked document
            fields_to_extract: Specific fields to extract (None = all configured)

        Returns:
            Tuple of (extraction_result, trace_dict)
        """
        from ..extract.tier4_agentic import (
            Tier4Agent,
            Tier4Config as AgentTier4Config,
            AgentModel,
            FieldSpec,
            PriorTierResult,
            Tier4ExtractionResult,
            FIELD_SPECS,
        )
        from ..parse.ixbrl_parser import IXBRLParser
        from ..parse.models import ChunkedDocument

        logger.info(f"Running Tier 4 agentic extraction for {fund_name}")

        # Get Tier 4 config
        tier4_cfg = self.config.tiers.tier4

        # Map model string to enum
        model_mapping = {
            "gpt-4o": AgentModel.GPT_4O,
            "gpt-4o-mini": AgentModel.GPT_4O_MINI,
            "claude-sonnet-4-20250514": AgentModel.CLAUDE_SONNET,
            "claude-opus-4-20250514": AgentModel.CLAUDE_OPUS,
        }
        model = model_mapping.get(tier4_cfg.model, AgentModel.GPT_4O)

        # Create agent config
        agent_config = AgentTier4Config(
            model=model,
            max_iterations=tier4_cfg.max_iterations,
            timeout_seconds=tier4_cfg.timeout_seconds,
            confidence_threshold=tier4_cfg.confidence_threshold,
        )

        # Create agent
        agent = Tier4Agent(config=agent_config, document=chunked_doc)

        # Determine which fields to extract
        target_fields = fields_to_extract or tier4_cfg.fields_to_extract or list(FIELD_SPECS.keys())

        # Track results
        tier4_results: dict[str, Tier4ExtractionResult] = {}
        extractions = {}

        for field_name in target_fields:
            if field_name not in FIELD_SPECS:
                logger.warning(f"[Tier4] No FieldSpec for '{field_name}', skipping")
                continue

            field_spec = FIELD_SPECS[field_name]

            # Create prior tier summary (empty since we're running standalone)
            prior_results = [
                PriorTierResult(tier=0, status="not_found", failure_reason="Tier 4 standalone mode"),
                PriorTierResult(tier=1, status="not_found", failure_reason="Tier 4 standalone mode"),
                PriorTierResult(tier=2, status="not_found", failure_reason="Tier 4 standalone mode"),
                PriorTierResult(tier=3, status="not_found", failure_reason="Tier 4 standalone mode"),
            ]

            # Run extraction
            result = agent.extract(field_spec=field_spec, prior_results=prior_results)
            tier4_results[field_name] = result

            # Add to extractions if successful
            if result.success and result.value is not None:
                extractions[field_name] = {
                    "value": result.value,
                    "confidence": result.confidence,
                    "evidence": result.evidence,
                    "source_section": result.source_section,
                    "source_chunk_id": result.source_chunk_id,
                }

        # Build extraction result
        filing_id = f"{cik}_{chunked_doc.chunked_sections[0].section_id.split('_')[0] if chunked_doc.chunked_sections else 'unknown'}"

        # Determine fund type from XBRL
        parser = IXBRLParser()
        fund_type_enum, fund_type_flags = parser.extract_fund_type(html_content)
        fund_type = fund_type_enum.value if hasattr(fund_type_enum, 'value') else str(fund_type_enum)

        result = {
            "filing_id": filing_id,
            "cik": cik,
            "fund_name": fund_name,
            "fund_type": fund_type,
            "fund_type_flags": fund_type_flags,
            "xbrl_fees": xbrl_values.get("fees", {}),
            "tier4_extractions": extractions,
        }

        # Build comprehensive trace for observability
        trace = {
            "extraction_mode": "tier4_agentic",
            "model": tier4_cfg.model,
            "max_iterations": tier4_cfg.max_iterations,
            "fields_attempted": list(target_fields),
            "fields_extracted": [f for f, r in tier4_results.items() if r.success],
            "fields_failed": [f for f, r in tier4_results.items() if not r.success],
            "total_duration_seconds": sum(r.duration_seconds for r in tier4_results.values()),
            "total_searches": sum(r.total_searches for r in tier4_results.values()),
            "total_chunks_examined": sum(r.chunks_examined for r in tier4_results.values()),
            "field_details": {
                name: {
                    "success": r.success,
                    "value": r.value,
                    "confidence": r.confidence,
                    "evidence": r.evidence[:200] if r.evidence else None,
                    "source_section": r.source_section,
                    "source_chunk_id": r.source_chunk_id,
                    "stop_reason": r.stop_reason.value,
                    "iterations_used": r.iterations_used,
                    "total_searches": r.total_searches,
                    "chunks_examined": r.chunks_examined,
                    "duration_seconds": r.duration_seconds,
                    "reasoning_trace": r.reasoning_trace[-5:],  # Last 5 reasoning steps
                    "steps": [
                        {
                            "iteration": s.iteration,
                            "thought": s.thought[:150] if s.thought else None,
                            "action": s.action.value,
                            "observation": s.observation[:200] if s.observation else None,
                        }
                        for s in r.steps
                    ],
                }
                for name, r in tier4_results.items()
            },
        }

        return result, trace

    def run_experiment(
        self,
        fund_filings: Optional[dict[str, dict]] = None,
        funds: Optional[list[str]] = None,
        name: Optional[str] = None,
    ) -> ExperimentRun:
        """
        Run extraction experiment on specified funds.

        Args:
            fund_filings: Dict mapping fund_name -> {"path": str, "cik": str}
                         If None, uses DEFAULT_FUND_FILINGS
            funds: Optional list of fund names to filter to (from config or explicit)
            name: Optional experiment name for run ID

        Returns:
            ExperimentRun with all results
        """
        # Use default filings if not provided
        if fund_filings is None:
            fund_filings = self.DEFAULT_FUND_FILINGS.copy()

        # Filter to specific funds if requested
        target_funds = funds or self.config.validation.funds
        if target_funds:
            fund_filings = {
                name: info
                for name, info in fund_filings.items()
                if name in target_funds
            }

        # Generate run ID
        run_id = self._generate_run_id(name)
        logger.info(f"Starting experiment run: {run_id}")
        logger.info(f"Config hash: {self.config.config_hash()}")
        logger.info(f"Funds to process: {list(fund_filings.keys())}")

        # Create run object
        run = ExperimentRun(
            run_id=run_id,
            config=self.config,
            config_hash=self.config.config_hash(),
            started_at=datetime.now(),
        )

        # Extract each fund
        for fund_name, fund_info in fund_filings.items():
            filing_path = fund_info.get("path", "")
            cik = fund_info.get("cik", "")

            # Check path exists
            if not Path(filing_path).exists():
                error_msg = f"Filing path not found: {filing_path}"
                logger.warning(error_msg)
                run.errors.append(f"{fund_name}: {error_msg}")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {fund_name}")
            logger.info(f"{'='*60}")

            result = self._extract_fund(fund_name, filing_path, cik)
            run.fund_results[fund_name] = result

            if result.error:
                run.errors.append(f"{fund_name}: {result.error}")

        run.completed_at = datetime.now()

        # Log summary
        summary = run.summarize()
        logger.info(f"\n{'='*60}")
        logger.info(f"EXPERIMENT COMPLETE: {run_id}")
        logger.info(f"{'='*60}")
        logger.info(f"Funds: {summary['successful_funds']}/{summary['total_funds']} successful")
        logger.info(f"Duration: {summary['total_duration_seconds']:.1f}s")

        return run

    def save_run(self, run: ExperimentRun) -> Path:
        """
        Save experiment run to disk.

        Creates directory structure:
            data/experiments/{run_id}/
                config.yaml
                results.json
                {fund_name}_trace.json (for each fund)

        Args:
            run: ExperimentRun to save

        Returns:
            Path to experiment directory
        """
        exp_dir = self.output_dir / run.run_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        save_config(run.config, exp_dir / "config.yaml")

        # Save full results
        results_path = exp_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(run.to_dict(), f, indent=2, default=str)

        # Save individual fund traces
        for fund_name, result in run.fund_results.items():
            if result.trace:
                # Clean fund name for filename
                safe_name = fund_name.replace(" ", "_").replace("/", "_")[:50]
                trace_path = exp_dir / f"{safe_name}_trace.json"
                with open(trace_path, "w", encoding="utf-8") as f:
                    json.dump(result.trace, f, indent=2, default=str)

        logger.info(f"Saved experiment to: {exp_dir}")
        return exp_dir


def load_run(run_path: str | Path) -> ExperimentRun:
    """
    Load a saved experiment run.

    Args:
        run_path: Path to experiment directory or results.json

    Returns:
        ExperimentRun loaded from disk
    """
    run_path = Path(run_path)

    # Handle both directory and file paths
    if run_path.is_dir():
        results_path = run_path / "results.json"
    else:
        results_path = run_path

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct ExperimentRun
    config = ExperimentConfig.model_validate(data["config"])

    run = ExperimentRun(
        run_id=data["run_id"],
        config=config,
        config_hash=data["config_hash"],
        started_at=datetime.fromisoformat(data["started_at"]),
        completed_at=(
            datetime.fromisoformat(data["completed_at"])
            if data["completed_at"]
            else None
        ),
        errors=data.get("errors", []),
    )

    # Reconstruct fund results
    for fund_name, result_data in data.get("fund_results", {}).items():
        run.fund_results[fund_name] = FundExtractionResult(
            fund_name=result_data["fund_name"],
            cik=result_data.get("cik", ""),
            filing_path=result_data.get("filing_path", ""),
            extraction=result_data.get("extraction", {}),
            trace=result_data.get("trace"),
            stats=result_data.get("stats", {}),
            error=result_data.get("error"),
            duration_seconds=result_data.get("duration_seconds", 0.0),
        )

    return run


def list_runs(output_dir: str = "data/experiments") -> list[dict]:
    """
    List all experiment runs in output directory.

    Returns:
        List of run summaries with run_id, timestamp, config_hash
    """
    output_dir = Path(output_dir)
    runs = []

    if not output_dir.exists():
        return runs

    for exp_dir in sorted(output_dir.iterdir(), reverse=True):
        if not exp_dir.is_dir():
            continue

        results_path = exp_dir / "results.json"
        if not results_path.exists():
            continue

        try:
            with open(results_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            runs.append({
                "run_id": data.get("run_id", exp_dir.name),
                "config_hash": data.get("config_hash", ""),
                "started_at": data.get("started_at", ""),
                "experiment_name": data.get("config", {}).get("experiment", {}).get("name", ""),
                "total_funds": data.get("summary", {}).get("total_funds", 0),
                "successful_funds": data.get("summary", {}).get("successful_funds", 0),
                "path": str(exp_dir),
            })
        except Exception as e:
            logger.warning(f"Error reading {results_path}: {e}")

    return runs
