"""
LLM extraction package for SEC filings.

This package provides structured extraction of narrative fields
from document sections using OpenAI models with instructor.
"""

from .schemas import (
    ConfidenceLevel,
    Citation,
    IncentiveFeeExtraction,
    ExpenseCapExtraction,
    RepurchaseTermsExtraction,
    AllocationTargetsExtraction,
    ConcentrationLimitsExtraction,
    ShareClassesExtraction,
    ShareClassDetails,
    AllocationTarget,
    ConcentrationLimit,
    DocumentExtractionResult,
)

from .prompts import (
    SYSTEM_PROMPT,
    get_prompt_for_field,
    get_prompt_with_examples,
    SECTION_PROMPT_MAP,
)

from .examples import (
    FieldCategory,
    ExtractionExample,
    ExampleLibrary,
    get_examples_for_field,
    format_examples_for_prompt,
    add_example,
    save_examples_to_yaml,
    reload_examples,
    get_example_counts,
    create_example_from_extraction,
)

from .extractor import (
    FieldExtractor,
    DocumentExtractor,
    extract_document,
)

from .observability import (
    DecisionType,
    MatchType,
    LayerDecision,
    ObservableExtraction,
    ExtractionTrace,
    explain_extraction,
    dump_decisions,
    summarize_extraction,
    print_flagged_fields,
    load_trace,
)

from .tier4_agentic import (
    Tier4Config,
    Tier4Agent,
    Tier4ExtractionResult,
    FieldSpec,
    PriorTierResult,
    AgentMemory,
    AgentTools,
    AgentAction,
    AgentStep,
    ConversationTurn,
    SearchAttempt,
    SearchResultStatus,
    FIELD_SPECS,
    create_prior_tier_summary,
)

__all__ = [
    # Schemas
    "ConfidenceLevel",
    "Citation",
    "IncentiveFeeExtraction",
    "ExpenseCapExtraction",
    "RepurchaseTermsExtraction",
    "AllocationTargetsExtraction",
    "ConcentrationLimitsExtraction",
    "ShareClassesExtraction",
    "ShareClassDetails",
    "AllocationTarget",
    "ConcentrationLimit",
    "DocumentExtractionResult",
    # Prompts
    "SYSTEM_PROMPT",
    "get_prompt_for_field",
    "get_prompt_with_examples",
    "SECTION_PROMPT_MAP",
    # Examples
    "FieldCategory",
    "ExtractionExample",
    "ExampleLibrary",
    "get_examples_for_field",
    "format_examples_for_prompt",
    "add_example",
    "save_examples_to_yaml",
    "reload_examples",
    "get_example_counts",
    "create_example_from_extraction",
    # Extractors
    "FieldExtractor",
    "DocumentExtractor",
    "extract_document",
    # Observability
    "DecisionType",
    "MatchType",
    "LayerDecision",
    "ObservableExtraction",
    "ExtractionTrace",
    "explain_extraction",
    "dump_decisions",
    "summarize_extraction",
    "print_flagged_fields",
    "load_trace",
    # Tier 4 Agentic
    "Tier4Config",
    "Tier4Agent",
    "Tier4ExtractionResult",
    "FieldSpec",
    "PriorTierResult",
    "AgentMemory",
    "AgentTools",
    "AgentAction",
    "AgentStep",
    "ConversationTurn",
    "SearchAttempt",
    "SearchResultStatus",
    "FIELD_SPECS",
    "create_prior_tier_summary",
]
