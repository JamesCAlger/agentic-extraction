"""
Configuration management for extraction experiments.

Supports:
- Loading base config from YAML
- Merging experiment overrides
- Config validation with Pydantic
- Config hashing for reproducibility
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Config Models
# =============================================================================


class ExtractionConfig(BaseModel):
    """LLM extraction settings."""

    model: str = "gpt-4o-mini"
    provider: Optional[str] = None  # openai | anthropic | google (auto-detected from model if None)
    max_chunk_tokens: int = 2000
    overlap_tokens: int = 200
    max_retries: int = 2
    extraction_mode: str = "combined"  # combined | per_section
    use_examples: bool = True
    max_examples: int = 3

    # Rate limiting to avoid TPM/RPM caps
    delay_between_calls: float = 0.0  # Seconds to wait between API calls
    requests_per_minute: Optional[int] = None  # Max requests per minute (None = no limit)


class ChunkingStrategyConfig(BaseModel):
    """Configuration for chunking strategy experiments."""

    strategy: str = "standard"  # standard | paragraph_atomic | hierarchical | raptor
    max_tokens: int = 500  # Maximum tokens per chunk
    overlap_tokens: int = 200  # Overlap between chunks
    min_paragraph_tokens: int = 30  # Minimum tokens for paragraph-atomic (merge smaller)
    generate_summaries: bool = True  # Generate LLM summaries for hierarchical/RAPTOR
    summary_model: str = "gpt-4o-mini"  # Model for summary generation
    raptor_cluster_size: int = 5  # Target cluster size for RAPTOR
    raptor_max_levels: int = 3  # Maximum levels in RAPTOR tree


class RerankerConfig(BaseModel):
    """Cohere Reranker configuration for Tier 3."""

    enabled: bool = False  # Enable Cohere reranking
    model: str = "rerank-v3.5"  # Cohere rerank model
    first_pass_n: int = 50  # Chunks to send to reranker from keyword scoring
    top_k: int = 15  # Chunks to return from reranker
    score_threshold: float = 0.3  # Minimum reranker score to include


class EmbeddingRetrievalConfig(BaseModel):
    """Configuration for embedding-based dense retrieval."""

    enabled: bool = False  # Enable embedding-based retrieval
    provider: str = "openai"  # openai | voyage | local
    model: str = "text-embedding-3-small"  # Model name
    dimensions: Optional[int] = None  # Embedding dimensions (for models that support it)
    top_k: int = 10  # Number of chunks to retrieve
    prepend_context: bool = False  # Add section title to chunk before embedding
    cache_embeddings: bool = True  # Cache embeddings to disk


class AdversarialValidationConfig(BaseModel):
    """Validation configuration for extractions (adversarial or lightweight)."""

    enabled: bool = True  # Enable LLM validation
    lightweight: bool = True  # Use lightweight validation (faster, cheaper, less strict)
    model: str = "claude-sonnet-4-20250514"  # Model for adversarial validation
    lightweight_model: str = "gpt-4o-mini"  # Model for lightweight validation
    validate_booleans: bool = True  # Validate boolean field extractions
    validate_all: bool = False  # Validate ALL field types (not just booleans)
    require_exact_quote: bool = True  # Require exact supporting quote (adversarial only)
    max_retries: int = 2  # Retries if validation fails


class Tier4Config(BaseModel):
    """Tier 4 unconstrained agentic extraction configuration."""

    enabled: bool = False  # Enable Tier 4 agentic fallback
    only: bool = False  # Skip all other tiers, run ONLY Tier 4
    model: str = "gpt-4o"  # Recommend GPT-4o or Claude Sonnet for reasoning
    max_iterations: int = 8  # Max agent loop iterations
    timeout_seconds: int = 120  # Per-field timeout
    confidence_threshold: float = 0.8  # Min confidence to accept extraction
    fields_to_extract: list[str] = Field(default_factory=list)  # Empty = all defined fields
    adversarial_validation: AdversarialValidationConfig = Field(default_factory=AdversarialValidationConfig)


class TierConfig(BaseModel):
    """Tier enable/disable and parameters."""

    tier0_enabled: bool = True
    tier1_enabled: bool = True
    tier2_enabled: bool = True
    tier2_max_chars: int = 12000
    tier3_enabled: bool = True
    tier3_only: bool = False  # Skip Tier 1+2, run ALL fields through Tier 3
    tier3_top_k_sections: int = 5
    tier3_max_chunks_per_section: int = 10
    tier3_keyword_threshold: int = 0

    # Tier 4 unconstrained agentic extraction
    tier4: Tier4Config = Field(default_factory=Tier4Config)

    # Cohere Reranker for Tier 3
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)

    # Embedding-based retrieval (alternative to keyword scoring)
    embedding_retrieval: EmbeddingRetrievalConfig = Field(default_factory=EmbeddingRetrievalConfig)

    # DocVQA extraction mode
    docvqa_enabled: bool = False  # Use DocVQA question-based extraction
    docvqa_only: bool = False  # Skip all tiers, use ONLY DocVQA for extraction

    # Tier 3 + DocVQA hybrid mode
    tier3_docvqa: bool = False  # Use Tier 3 retrieval + DocVQA questions

    # Sequential Conditional DocVQA mode
    sequential_docvqa: bool = False  # Use sequential existence->detail questioning

    # Tier3 + Sequential DocVQA hybrid mode
    tier3_sequential_docvqa: bool = False  # Tier3 chunk selection + sequential questioning

    # Hybrid DocVQA+Tier3 mode (DocVQA for existence, Tier3 for details)
    hybrid_docvqa_tier3: bool = False  # Best of both approaches

    # Per-datapoint Tier3 extraction (granular keywords per datapoint)
    per_datapoint_tier3: bool = False  # Extract each datapoint separately with specific keywords

    # Per-datapoint with Tier3-style prompts (isolates prompt style effect)
    per_datapoint_tier3_style: bool = False  # Same as per_datapoint but uses extraction-style prompts

    # Hybrid Boolean+Tier3 (DocVQA for booleans, Tier3 for numerics)
    hybrid_boolean_tier3: bool = False  # Best prompt style for each field type

    # Dense retrieval mode (uses embeddings instead of keywords for retrieval)
    dense_retrieval: bool = False  # Use embedding-based retrieval instead of keywords

    # Hybrid retrieval mode (union of keyword + dense with RRF fusion)
    hybrid_retrieval: bool = False  # Combine keyword and dense retrieval for better recall


class GroundingConfig(BaseModel):
    """Grounding validation settings."""

    enabled: bool = True
    strict_mode: bool = False
    min_score: float = 0.0

    # Grounding strategy selection
    # Options: exact_match (current), nli, llm_judge, semantic_similarity, hybrid
    strategy: str = "exact_match"

    # NLI settings (for nli and hybrid strategies)
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    nli_entailment_threshold: float = 0.7
    nli_contradiction_threshold: float = 0.7

    # LLM Judge settings (for llm_judge and hybrid strategies)
    llm_judge_model: str = "gpt-4o-mini"
    llm_judge_provider: str = "openai"

    # Semantic similarity settings
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.75

    # Hybrid settings (NLI + LLM fallback)
    hybrid_nli_high_threshold: float = 0.85  # Above this, accept without LLM
    hybrid_nli_low_threshold: float = 0.4   # Below this, reject without LLM
    hybrid_use_llm_for_ambiguous: bool = True


class ConfidenceScoreConfig(BaseModel):
    """Confidence scoring configuration."""

    enabled: bool = False  # Enable confidence scoring
    require_grounding: bool = True  # CRITICAL: Zero confidence if not grounded

    # Thresholds for confidence levels
    high_threshold: float = 0.7
    low_threshold: float = 0.4

    # Component weights (should sum to 1.0)
    weight_llm_confidence: float = 0.25
    weight_grounding: float = 0.35
    weight_evidence: float = 0.20
    weight_retrieval: float = 0.20


class ShareClassTwoPassConfig(BaseModel):
    """Configuration for two-pass share class extraction (DEPRECATED)."""

    enabled: bool = False  # DISABLED - use discovery_first instead
    discovery_model: Optional[str] = None  # Model for discovery (uses extraction model if None)
    extraction_model: str = "gpt-4o-mini"  # Model for per-class extraction
    discovery_max_chunks: int = 20  # Chunks to use for discovery phase
    per_class_max_chunks: int = 15  # Chunks per class extraction
    fallback_to_full_doc: bool = True  # Fall back to full doc if no class-specific chunks


class ShareClassDiscoveryFirstConfig(BaseModel):
    """Configuration for discovery-first share class extraction (RECOMMENDED).

    Discovery-first runs share class discovery BEFORE T1 extraction, then passes
    discovered class names to the extraction pipeline for targeted extraction.

    Benefits over two-pass:
    - Finds rare classes (Class R, Class U) reliably
    - All fields still extracted by T1 (no field loss)
    - Post-extraction validation removes hallucinated classes
    - Faster than two-pass (1 discovery call vs N per-class calls)
    """

    enabled: bool = True  # Enabled by default
    model: Optional[str] = None  # Model for discovery (uses extraction model if None)
    max_chunks: int = 25  # Chunks to use for discovery


class EnsembleConfig(BaseModel):
    """Ensemble selection configuration for combining multiple extraction methods."""

    enabled: bool = False  # Enable ensemble mode
    methods: list[str] = Field(default_factory=lambda: ["t3_k10", "reranker"])

    # Selection parameters
    min_confidence_gap: float = 0.15  # Min gap to prefer one method
    agreement_boost: float = 0.10  # Confidence boost when methods agree

    # Fallback behavior
    default_method: str = "t3_k10"  # Method to use when gap is too small

    # Method-specific configs (optional overrides)
    t3_top_k_sections: int = 10
    reranker_first_pass_n: int = 200
    reranker_top_k: int = 10

    # T4 escalation settings (for ensemble_t4 mode)
    escalate_to_t4: bool = False  # Enable T4 escalation for disagreements/nulls
    escalate_on_disagreement: bool = True  # Escalate when T3 and Reranker disagree
    escalate_on_both_null: bool = True  # Escalate when both return null (Scenario 2)
    t4_model: str = "gpt-4o"  # Model to use for T4 extraction
    t4_max_iterations: int = 12  # Max T4 agent iterations
    t4_timeout_seconds: int = 180  # T4 timeout per field

    # Hybrid retrieval settings (replaces keyword-only T3 with keyword + dense + RRF)
    use_hybrid_retrieval: bool = False  # Use hybrid instead of keyword-only T3
    hybrid_embedding_provider: str = "openai"  # openai, voyage, local
    hybrid_embedding_model: str = "text-embedding-3-small"  # or voyage-finance-2
    hybrid_rrf_k: int = 60  # RRF constant
    hybrid_keyword_top_k: int = 20  # Chunks from keyword retrieval
    hybrid_dense_top_k: int = 20  # Chunks from dense retrieval
    hybrid_final_top_k: int = 10  # Final chunks after RRF fusion

    # Multi-query expansion settings (replaces static keywords with expanded queries + RRF)
    use_multi_query: bool = False  # Use multi-query expansion instead of static keywords
    multi_query_expansion_method: str = "programmatic"  # programmatic, llm, or hybrid
    multi_query_retrieval_strategy: str = "keyword"  # keyword, dense, or hybrid
    multi_query_rrf_k: int = 60  # RRF constant for multi-query fusion
    multi_query_per_query_top_k: int = 15  # Chunks per query before fusion
    multi_query_final_top_k: int = 10  # Final chunks after RRF fusion
    multi_query_embedding_provider: str = "openai"  # For dense/hybrid retrieval
    multi_query_embedding_model: str = "text-embedding-3-small"  # Embedding model
    multi_query_holistic_extraction: bool = False  # Extract all fields in a group together

    # Adversarial validation settings (validates extractions when methods agree)
    adversarial_validation_enabled: bool = True  # Enable LLM validation on agreement
    adversarial_validation_lightweight: bool = True  # Use lightweight validation
    adversarial_validation_validate_booleans: bool = True  # Validate boolean fields
    adversarial_validation_validate_all: bool = False  # Validate all field types
    adversarial_validation_escalate_on_rejection: bool = True  # Escalate to T4 when validation fails
    adversarial_validation_model: str = "claude-sonnet-4-20250514"  # Model for adversarial validation

    # Hybrid routing: when methods disagree, use field-type-based preference
    # instead of escalating to T4. Uses empirically-determined best method per field.
    use_hybrid_routing: bool = False

    # Confidence-based routing: when methods disagree, use confidence scores
    # to pick the more reliable extraction. CRITICAL: ungrounded = 0 confidence.
    use_confidence_routing: bool = False
    confidence_min_gap: float = 0.25  # Min confidence gap to prefer one method
    confidence_low_threshold: float = 0.3  # Below this, both uncertain → escalate
    confidence_high_threshold: float = 0.7  # Both above this but disagree → escalate
    confidence_require_grounding: bool = True  # CRITICAL: ungrounded = 0 confidence

    # Grounding strategy for confidence routing
    # Options: None (use existing exact match), nli, llm_judge, hybrid, semantic_similarity
    grounding_strategy: Optional[str] = None
    grounding_nli_model: str = "cross-encoder/nli-deberta-v3-base"
    grounding_nli_entailment_threshold: float = 0.7
    grounding_nli_contradiction_threshold: float = 0.7
    grounding_llm_judge_model: str = "gpt-4o-mini"
    grounding_llm_judge_provider: str = "openai"
    grounding_hybrid_nli_high_threshold: float = 0.85  # Above this, accept without LLM
    grounding_hybrid_nli_low_threshold: float = 0.4   # Below this, reject without LLM
    grounding_hybrid_use_llm_for_ambiguous: bool = True

    # Two-pass share class extraction (DEPRECATED - use discovery_first instead)
    share_class_two_pass: ShareClassTwoPassConfig = Field(default_factory=ShareClassTwoPassConfig)

    # Discovery-first share class extraction (RECOMMENDED)
    share_class_discovery_first: ShareClassDiscoveryFirstConfig = Field(
        default_factory=ShareClassDiscoveryFirstConfig
    )


class ObservabilityConfig(BaseModel):
    """Tracing and observability settings."""

    enabled: bool = True
    trace_level: str = "full"  # full | summary | minimal
    save_llm_responses: bool = True
    output_dir: Optional[str] = None


class ValidationConfig(BaseModel):
    """Validation fund set configuration."""

    funds: list[str] = Field(default_factory=list)
    ground_truth_dir: str = "configs/ground_truth"


class CostConfig(BaseModel):
    """API cost tracking settings."""

    input_token_cost: float = 0.00015  # per 1K tokens
    output_token_cost: float = 0.0006  # per 1K tokens


class ExperimentMetadata(BaseModel):
    """Experiment metadata (from override configs)."""

    name: Optional[str] = None
    description: Optional[str] = None
    hypothesis: Optional[str] = None


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""

    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    chunking: ChunkingStrategyConfig = Field(default_factory=ChunkingStrategyConfig)
    tiers: TierConfig = Field(default_factory=TierConfig)
    grounding: GroundingConfig = Field(default_factory=GroundingConfig)
    confidence: ConfidenceScoreConfig = Field(default_factory=ConfidenceScoreConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    experiment: ExperimentMetadata = Field(default_factory=ExperimentMetadata)
    field_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def config_hash(self) -> str:
        """
        Generate hash of config for reproducibility tracking.

        Returns:
            SHA256 hash of serialized config (first 12 chars)
        """
        # Serialize to JSON with sorted keys for determinism
        config_json = self.model_dump_json(exclude={"experiment"})
        return hashlib.sha256(config_json.encode()).hexdigest()[:12]

    def get_field_config(self, field_name: str) -> dict[str, Any]:
        """
        Get config for a specific field (with overrides applied).

        Args:
            field_name: Name of the field

        Returns:
            Dict with field-specific settings merged with defaults
        """
        base = {
            "model": self.extraction.model,
            "max_chunk_tokens": self.extraction.max_chunk_tokens,
            "extraction_mode": self.extraction.extraction_mode,
            "use_examples": self.extraction.use_examples,
            "max_examples": self.extraction.max_examples,
        }

        # Apply field-specific overrides
        if field_name in self.field_overrides:
            base.update(self.field_overrides[field_name])

        return base


# =============================================================================
# Config Loading Functions
# =============================================================================


def load_yaml(path: Union[str, Path]) -> dict[str, Any]:
    """Load YAML file and return as dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries.

    Override values take precedence. Nested dicts are merged recursively.

    Args:
        base: Base dictionary
        override: Override dictionary

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def merge_configs(base_dict: dict, override_dict: dict) -> dict:
    """
    Merge base config with experiment override.

    Args:
        base_dict: Base configuration dictionary
        override_dict: Override dictionary (from experiment config)

    Returns:
        Merged configuration dictionary
    """
    return deep_merge(base_dict, override_dict)


def load_config(
    config_path: Union[str, Path],
    base_path: Optional[Union[str, Path]] = None,
) -> ExperimentConfig:
    """
    Load experiment configuration from YAML.

    If config_path is an experiment override, it will be merged with base config.
    If base_path is not provided, looks for configs/base.yaml in project root.

    Args:
        config_path: Path to config file (base or override)
        base_path: Optional explicit path to base config

    Returns:
        ExperimentConfig with all settings resolved
    """
    config_path = Path(config_path)
    config_dict = load_yaml(config_path)

    # Check if this is an override config (has 'experiment' key)
    is_override = "experiment" in config_dict

    if is_override:
        # Load base config and merge
        if base_path is None:
            # Try to find base.yaml relative to config
            base_path = config_path.parent.parent / "base.yaml"
            if not base_path.exists():
                # Try project root
                base_path = Path("configs/base.yaml")

        if base_path.exists():
            base_dict = load_yaml(base_path)
            config_dict = merge_configs(base_dict, config_dict)
            logger.info(f"Merged config from {config_path} with base {base_path}")
        else:
            logger.warning(f"Base config not found at {base_path}, using override only")

    # Parse into Pydantic model
    config = ExperimentConfig.model_validate(config_dict)

    logger.info(f"Loaded config: {config.experiment.name or 'base'} (hash: {config.config_hash()})")

    return config


def save_config(config: ExperimentConfig, output_path: Union[str, Path]) -> Path:
    """
    Save resolved config to YAML file.

    Args:
        config: ExperimentConfig to save
        output_path: Path to save to

    Returns:
        Path where config was saved
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = config.model_dump()

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved config to {output_path}")
    return output_path


# =============================================================================
# Config Validation
# =============================================================================


def validate_config(config: ExperimentConfig) -> list[str]:
    """
    Validate config and return list of warnings/issues.

    Args:
        config: ExperimentConfig to validate

    Returns:
        List of warning messages (empty if all good)
    """
    warnings = []

    # Check for valid extraction mode
    valid_modes = ["combined", "per_section"]
    if config.extraction.extraction_mode not in valid_modes:
        warnings.append(
            f"Invalid extraction_mode: {config.extraction.extraction_mode}. "
            f"Valid options: {valid_modes}"
        )

    # Check for valid trace level
    valid_trace_levels = ["full", "summary", "minimal"]
    if config.observability.trace_level not in valid_trace_levels:
        warnings.append(
            f"Invalid trace_level: {config.observability.trace_level}. "
            f"Valid options: {valid_trace_levels}"
        )

    # Check validation funds exist
    if not config.validation.funds:
        warnings.append("No validation funds configured")

    # Check ground truth directory exists
    gt_dir = Path(config.validation.ground_truth_dir)
    if not gt_dir.exists():
        warnings.append(f"Ground truth directory not found: {gt_dir}")

    # Check for suspicious parameter values
    if config.extraction.max_chunk_tokens < 500:
        warnings.append(
            f"max_chunk_tokens={config.extraction.max_chunk_tokens} is very low, "
            "may cause extraction failures"
        )

    if config.tiers.tier3_top_k_sections > 20:
        warnings.append(
            f"tier3_top_k_sections={config.tiers.tier3_top_k_sections} is high, "
            "may increase costs significantly"
        )

    return warnings
