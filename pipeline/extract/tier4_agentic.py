"""
Tier 4: Unconstrained Agentic Document Extraction

This module implements an unconstrained agent that can iteratively search
documents to find information that Tiers 0-3 failed to extract. Unlike
the scoped Tier 3 approach, this agent has full autonomy to:

- Generate and execute arbitrary search queries
- Reformulate queries based on results
- Expand or narrow search scope
- Decide when to stop searching

Architecture follows the ReAct (Reasoning + Acting) pattern:
    Think → Act → Observe → Repeat (or Stop)

Key Design Principles (from research):
1. Simple loop architecture with explicit reasoning
2. Self-correction via result evaluation before extraction
3. Clear stopping conditions (success, max iterations, exhaustion)
4. Rich tool design with search, context expansion, validation
5. Context efficiency - pass structure + prior results, not raw document
6. Full observability for debugging and improvement

References:
- Anthropic: Building Effective Agents
- IBM: ReAct Agent Framework
- Agentic RAG Survey (arXiv 2501.09136)
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from ..parse.models import ChunkedDocument, ChunkedSection, Chunk
from .field_normalization import (
    get_field_taxonomy,
    INCENTIVE_FEE_TAXONOMY,
    LEVERAGE_TAXONOMY,
    detect_leverage_format,
    LEVERAGE_FORMATS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Maximum retries for LLM response parsing
MAX_PARSE_RETRIES = 2

# Valid actions the agent can take
VALID_ACTIONS = {
    "search_keywords",
    "search_regex",
    "search_semantic",
    "get_context",
    "get_structure",
    "evaluate_retrieval",
    "extract",
    "validate",
    "stop_success",
    "stop_not_found",
}


# =============================================================================
# CONFIGURATION
# =============================================================================

class AgentModel(str, Enum):
    """Supported models for Tier 4 agent."""
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    CLAUDE_SONNET = "claude-sonnet-4-20250514"
    CLAUDE_OPUS = "claude-opus-4-20250514"
    CLAUDE_HAIKU = "claude-haiku-4-5-20251001"


@dataclass
class AdversarialValidationConfig:
    """
    Configuration for LLM validation of extractions (adversarial or lightweight).

    Adversarial uses a "devil's advocate" approach where a validation LLM is asked to
    find reasons why an extraction might be WRONG.

    Lightweight is simpler, faster, cheaper - just checks if evidence supports claim.
    """
    enabled: bool = True  # Enable LLM validation
    lightweight: bool = True  # Use lightweight validation (faster, cheaper, less strict)
    model: str = "claude-sonnet-4-20250514"  # Model for adversarial validation
    lightweight_model: str = "gpt-4o-mini"  # Model for lightweight validation
    validate_booleans: bool = True  # Validate boolean field extractions
    validate_all: bool = False  # Validate ALL field types
    require_exact_quote: bool = True  # Require exact supporting quote (adversarial only)
    max_retries: int = 2  # Retries if validation fails


@dataclass
class Tier4Config:
    """
    Configuration for Tier 4 unconstrained agent.

    Attributes:
        model: LLM model to use (recommend GPT-4o or Claude Sonnet)
        max_iterations: Maximum agent loop iterations before stopping
        max_searches_per_iteration: Limit searches per reasoning step
        confidence_threshold: Minimum confidence to accept extraction
        enable_self_correction: Whether to validate results before accepting
        temperature: LLM temperature (lower = more deterministic)
        timeout_seconds: Maximum total time for extraction attempt
        adversarial_validation: Config for adversarial LLM validation
    """
    model: AgentModel = AgentModel.GPT_4O
    max_iterations: int = 8
    max_searches_per_iteration: int = 3
    confidence_threshold: float = 0.8
    enable_self_correction: bool = True
    temperature: float = 0.1
    timeout_seconds: int = 120
    adversarial_validation: AdversarialValidationConfig = field(default_factory=AdversarialValidationConfig)

    # Context budget allocation (tokens)
    max_context_tokens: int = 8000
    reserved_for_tools: int = 2000
    reserved_for_response: int = 2000


@dataclass
class FieldSpec:
    """
    Specification for a field to extract.

    Provides the agent with semantic understanding of what to find.
    """
    name: str
    description: str
    expected_type: str  # "percentage", "currency", "text", "boolean", "list"
    examples: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)  # Alternative phrasings
    extraction_hints: list[str] = field(default_factory=list)


# =============================================================================
# SEARCH HISTORY & MEMORY
# =============================================================================

class SearchResultStatus(str, Enum):
    """Outcome of a search operation."""
    FOUND_RELEVANT = "found_relevant"
    FOUND_IRRELEVANT = "found_irrelevant"
    NO_RESULTS = "no_results"
    ERROR = "error"


@dataclass
class SearchAttempt:
    """
    Record of a single search attempt.

    Used to prevent redundant searches and inform query reformulation.
    """
    query: str
    search_type: str  # "keyword", "regex", "semantic"
    timestamp: datetime = field(default_factory=datetime.now)
    sections_searched: list[str] = field(default_factory=list)
    chunks_returned: int = 0
    status: SearchResultStatus = SearchResultStatus.NO_RESULTS
    top_chunk_ids: list[str] = field(default_factory=list)
    evaluation_notes: str = ""


@dataclass
class PriorTierResult:
    """
    Summary of what a prior tier attempted and found (or didn't find).

    Passed to Tier 4 to avoid redundant searches.
    """
    tier: int  # 0, 1, 2, or 3
    status: str  # "found", "not_found", "partial", "error"
    sections_searched: list[str] = field(default_factory=list)
    chunks_searched: list[str] = field(default_factory=list)
    extraction_result: Optional[Any] = None
    confidence: Optional[float] = None
    failure_reason: Optional[str] = None

    def to_prompt_string(self) -> str:
        """Format for inclusion in agent prompt."""
        status_str = f"Tier {self.tier}: {self.status.upper()}"
        if self.sections_searched:
            sections = ", ".join(self.sections_searched[:5])
            if len(self.sections_searched) > 5:
                sections += f" (+{len(self.sections_searched) - 5} more)"
            status_str += f"\n  Sections searched: {sections}"
        if self.failure_reason:
            status_str += f"\n  Failure reason: {self.failure_reason}"
        return status_str


@dataclass
class ConversationTurn:
    """A single turn in the agent conversation."""
    role: str  # "user" or "assistant"
    content: str


@dataclass
class AgentMemory:
    """
    Working memory for the agent during extraction.

    Tracks search history, findings, and reasoning to enable
    self-correction and prevent redundant operations.
    """
    field_name: str
    search_history: list[SearchAttempt] = field(default_factory=list)
    prior_tier_results: list[PriorTierResult] = field(default_factory=list)
    candidate_extractions: list[dict] = field(default_factory=list)
    reasoning_trace: list[str] = field(default_factory=list)
    sections_exhausted: set[str] = field(default_factory=set)
    conversation_history: list[ConversationTurn] = field(default_factory=list)
    last_tool_results: dict = field(default_factory=dict)  # Store detailed tool output

    def add_search(self, attempt: SearchAttempt):
        """Record a search attempt."""
        self.search_history.append(attempt)

    def add_reasoning(self, thought: str):
        """Record a reasoning step."""
        self.reasoning_trace.append(f"[{datetime.now().strftime('%H:%M:%S')}] {thought}")

    def add_turn(self, role: str, content: str):
        """Add a conversation turn."""
        self.conversation_history.append(ConversationTurn(role=role, content=content))

    def was_query_tried(self, query: str, similarity_threshold: float = 0.9) -> bool:
        """
        Check if a similar query was already tried.

        TODO: Implement fuzzy matching for near-duplicate detection.
        """
        query_lower = query.lower().strip()
        for attempt in self.search_history:
            if attempt.query.lower().strip() == query_lower:
                return True
        return False

    def get_untried_sections(self, all_sections: list[str]) -> list[str]:
        """Return sections not yet searched."""
        tried = set()
        for attempt in self.search_history:
            tried.update(attempt.sections_searched)
        return [s for s in all_sections if s not in tried]

    def to_context_string(self, max_tokens: int = 500) -> str:
        """
        Generate compact context summary for agent prompt.

        Prioritizes recent and relevant information within token budget.
        """
        lines = []

        # Prior tier summary
        lines.append("=== PRIOR TIER RESULTS ===")
        for ptr in self.prior_tier_results:
            lines.append(ptr.to_prompt_string())

        # Recent search summary
        lines.append("\n=== SEARCH HISTORY ===")
        if not self.search_history:
            lines.append("No searches performed yet.")
        else:
            for attempt in self.search_history[-5:]:  # Last 5 searches
                lines.append(
                    f"- Query: \"{attempt.query}\" → {attempt.status.value} "
                    f"({attempt.chunks_returned} chunks)"
                )
                if attempt.evaluation_notes:
                    lines.append(f"  Notes: {attempt.evaluation_notes}")

        # Candidate extractions found
        if self.candidate_extractions:
            lines.append("\n=== CANDIDATE VALUES FOUND ===")
            for candidate in self.candidate_extractions[-3:]:
                lines.append(f"- Value: {candidate.get('value')} (confidence: {candidate.get('confidence', 'unknown')})")
                if candidate.get('evidence'):
                    evidence_preview = candidate['evidence'][:100] + "..." if len(candidate.get('evidence', '')) > 100 else candidate.get('evidence', '')
                    lines.append(f"  Evidence: \"{evidence_preview}\"")

        # Exhausted sections
        if self.sections_exhausted:
            lines.append(f"\nExhausted sections: {', '.join(list(self.sections_exhausted)[:10])}")

        return "\n".join(lines)

    def get_messages_for_llm(self, system_prompt: str) -> list[dict]:
        """
        Build message list for LLM call including conversation history.

        Returns messages in OpenAI format for multi-turn conversation.
        """
        messages = [{"role": "system", "content": system_prompt}]

        for turn in self.conversation_history:
            messages.append({"role": turn.role, "content": turn.content})

        return messages


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

@dataclass
class ToolResult:
    """Result from a tool invocation."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    tokens_used: int = 0


class AgentTools:
    """
    Tools available to the Tier 4 agent.

    Each tool is designed to be:
    - Self-contained and unambiguous
    - Return complete, actionable information
    - Include metadata for observability
    """

    def __init__(self, document: ChunkedDocument):
        """
        Initialize tools with document context.

        Args:
            document: The ChunkedDocument to search
        """
        self.document = document
        self._search_count = 0
        self._chunks_retrieved = 0

    def search_by_keywords(
        self,
        keywords: list[str],
        max_chunks: int = 10,
        section_filter: Optional[list[str]] = None,
    ) -> ToolResult:
        """
        Search document chunks by keyword matching.

        Args:
            keywords: List of keywords to search for
            max_chunks: Maximum chunks to return
            section_filter: Optional list of section titles to restrict search

        Returns:
            ToolResult with matching chunks and metadata
        """
        self._search_count += 1

        try:
            matches = []
            keywords_lower = [k.lower() for k in keywords]

            for section in self.document.chunked_sections:
                # Apply section filter if provided
                if section_filter and section.section_title not in section_filter:
                    continue

                for chunk in section.chunks:
                    content_lower = chunk.content.lower()
                    score = sum(1 for kw in keywords_lower if kw in content_lower)
                    if score > 0:
                        matches.append({
                            "chunk_id": chunk.chunk_id,
                            "section": section.section_title,
                            "score": score,
                            "content_preview": chunk.content[:1500],  # Increased from 500 to see more context
                            "chunk_index": chunk.chunk_index,
                        })

            # Sort by score and limit
            matches.sort(key=lambda x: x["score"], reverse=True)
            matches = matches[:max_chunks]

            self._chunks_retrieved += len(matches)

            return ToolResult(
                success=True,
                data={
                    "matches": matches,
                    "total_found": len(matches),
                    "keywords_used": keywords,
                },
                tokens_used=sum(len(m["content_preview"]) // 4 for m in matches),
            )

        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return ToolResult(success=False, error=str(e))

    def search_by_regex(
        self,
        pattern: str,
        max_chunks: int = 10,
        section_filter: Optional[list[str]] = None,
    ) -> ToolResult:
        """
        Search document chunks by regex pattern.

        Args:
            pattern: Regex pattern to search for
            max_chunks: Maximum chunks to return
            section_filter: Optional section title filter

        Returns:
            ToolResult with matching chunks
        """
        import re

        self._search_count += 1

        try:
            compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            matches = []

            for section in self.document.chunked_sections:
                if section_filter and section.section_title not in section_filter:
                    continue

                for chunk in section.chunks:
                    found = compiled.findall(chunk.content)
                    if found:
                        matches.append({
                            "chunk_id": chunk.chunk_id,
                            "section": section.section_title,
                            "matches": found[:5],  # Limit match examples
                            "match_count": len(found),
                            "content_preview": chunk.content[:1500],  # Increased from 500 to see more context
                            "chunk_index": chunk.chunk_index,
                        })

            matches = matches[:max_chunks]
            self._chunks_retrieved += len(matches)

            return ToolResult(
                success=True,
                data={
                    "matches": matches,
                    "total_found": len(matches),
                    "pattern_used": pattern,
                },
                tokens_used=sum(len(m["content_preview"]) // 4 for m in matches),
            )

        except re.error as e:
            return ToolResult(success=False, error=f"Invalid regex: {e}")
        except Exception as e:
            logger.error(f"Regex search error: {e}")
            return ToolResult(success=False, error=str(e))

    def search_semantic(
        self,
        query: str,
        max_chunks: int = 10,
        section_filter: Optional[list[str]] = None,
        similarity_threshold: float = 0.3,
    ) -> ToolResult:
        """
        Semantic similarity search using sentence embeddings.

        Finds content conceptually similar to the query even if exact keywords
        don't match. Useful when document uses different terminology.

        Args:
            query: Natural language query describing what to find
            max_chunks: Maximum number of chunks to return
            section_filter: Optional list of section titles to limit search
            similarity_threshold: Minimum cosine similarity (0-1) to include

        Returns:
            ToolResult with semantically similar chunks ranked by similarity
        """
        self._search_count += 1

        try:
            # Lazy-load sentence transformer model
            if not hasattr(self, '_embedding_model'):
                try:
                    from sentence_transformers import SentenceTransformer
                    self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("[Tier4] Loaded sentence-transformers model")
                except ImportError:
                    return ToolResult(
                        success=False,
                        error="sentence-transformers not installed. Use 'pip install sentence-transformers'"
                    )

            # Compute query embedding
            query_embedding = self._embedding_model.encode(query, convert_to_tensor=False)

            # Score all chunks
            import numpy as np
            matches = []

            for section in self.document.chunked_sections:
                if section_filter and section.section_title not in section_filter:
                    continue

                for chunk in section.chunks:
                    # Compute chunk embedding (could cache this for efficiency)
                    chunk_embedding = self._embedding_model.encode(
                        chunk.content[:1000],  # Limit to first 1000 chars for speed
                        convert_to_tensor=False
                    )

                    # Cosine similarity
                    similarity = float(np.dot(query_embedding, chunk_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                    ))

                    if similarity >= similarity_threshold:
                        matches.append({
                            "chunk_id": chunk.chunk_id,
                            "section": section.section_title,
                            "similarity": round(similarity, 3),
                            "content_preview": chunk.content[:1500],
                            "chunk_index": chunk.chunk_index,
                        })

            # Sort by similarity (descending) and limit
            matches.sort(key=lambda x: x["similarity"], reverse=True)
            matches = matches[:max_chunks]
            self._chunks_retrieved += len(matches)

            return ToolResult(
                success=True,
                data={
                    "matches": matches,
                    "total_found": len(matches),
                    "query": query,
                    "threshold": similarity_threshold,
                },
                tokens_used=sum(len(m["content_preview"]) // 4 for m in matches),
            )

        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return ToolResult(success=False, error=str(e))

    def get_chunk_with_context(
        self,
        chunk_id: str,
        context_before: int = 1,
        context_after: int = 1,
    ) -> ToolResult:
        """
        Retrieve a specific chunk with surrounding context.

        Args:
            chunk_id: ID of the chunk to retrieve
            context_before: Number of chunks before to include
            context_after: Number of chunks after to include

        Returns:
            ToolResult with chunk content and context
        """
        try:
            # Find the chunk
            for section in self.document.chunked_sections:
                for i, chunk in enumerate(section.chunks):
                    if chunk.chunk_id == chunk_id:
                        # Gather context
                        start_idx = max(0, i - context_before)
                        end_idx = min(len(section.chunks), i + context_after + 1)

                        context_chunks = []
                        for j in range(start_idx, end_idx):
                            ctx_chunk = section.chunks[j]
                            context_chunks.append({
                                "chunk_id": ctx_chunk.chunk_id,
                                "content": ctx_chunk.content,
                                "is_target": ctx_chunk.chunk_id == chunk_id,
                                "chunk_index": ctx_chunk.chunk_index,
                            })

                        self._chunks_retrieved += len(context_chunks)

                        return ToolResult(
                            success=True,
                            data={
                                "section": section.section_title,
                                "chunks": context_chunks,
                                "target_chunk_index": i - start_idx,
                            },
                            tokens_used=sum(
                                len(c["content"]) // 4 for c in context_chunks
                            ),
                        )

            return ToolResult(success=False, error=f"Chunk {chunk_id} not found")

        except Exception as e:
            logger.error(f"Get chunk error: {e}")
            return ToolResult(success=False, error=str(e))

    def get_document_structure(self) -> ToolResult:
        """
        Get the document's section structure for navigation.

        Returns:
            ToolResult with section titles, chunk counts, and keywords
        """
        try:
            structure = []
            for section in self.document.chunked_sections:
                # Extract a few key terms from section content
                all_text = " ".join(c.content for c in section.chunks[:3])

                structure.append({
                    "title": section.section_title,
                    "chunk_count": len(section.chunks),
                    "chunk_range": f"{section.chunks[0].chunk_index if section.chunks else '?'}-{section.chunks[-1].chunk_index if section.chunks else '?'}",
                    "preview": all_text[:200] if all_text else "",
                })

            return ToolResult(
                success=True,
                data={
                    "total_sections": len(structure),
                    "total_chunks": sum(s["chunk_count"] for s in structure),
                    "sections": structure,
                },
                tokens_used=len(str(structure)) // 4,
            )

        except Exception as e:
            logger.error(f"Get structure error: {e}")
            return ToolResult(success=False, error=str(e))

    def validate_extraction(
        self,
        value: Any,
        evidence_text: str,
        expected_type: str,
        field_name: str = "",
    ) -> ToolResult:
        """
        Validate that an extracted value is grounded in evidence.

        Uses enhanced grounding validation with multiple format checks,
        fuzzy quote matching, and multi-signal confidence scoring.

        Args:
            value: The extracted value to validate
            evidence_text: The source text the value came from
            expected_type: Expected type ("percentage", "currency", "boolean", "text", "number")
            field_name: Name of the field being validated (for context)

        Returns:
            ToolResult with validation status, confidence, and detailed issues
        """
        from .grounding import GroundingValidator

        try:
            evidence_lower = evidence_text.lower()
            value_str = str(value).lower().strip()
            issues = []
            signals = {}

            # Initialize grounding validator
            validator = GroundingValidator(fuzzy_threshold=0.8)

            # Signal 1: Direct grounding check (value appears in evidence)
            is_grounded = False

            if expected_type == "percentage":
                # Use grounding.py's numeric finder with percentage formats
                try:
                    num_value = float(value_str.rstrip('%'))
                    is_grounded = validator._find_numeric_in_text(num_value, evidence_lower)

                    # For fields that allow normalization, also check for source expressions
                    # e.g., "33.33%" may be normalized from "one-third" or "300% asset coverage"
                    if not is_grounded and field_name in ("max_leverage_pct", "hurdle_rate_pct"):
                        is_grounded = self._check_normalized_source_grounding(
                            num_value, evidence_lower, field_name
                        )
                except (ValueError, TypeError):
                    is_grounded = value_str in evidence_lower

            elif expected_type == "currency":
                # Use grounding.py's numeric finder with currency formats
                try:
                    # Remove $ and commas for parsing
                    clean_value = value_str.replace('$', '').replace(',', '')
                    num_value = float(clean_value)
                    is_grounded = validator._find_numeric_in_text(num_value, evidence_lower, is_currency=True)
                except (ValueError, TypeError):
                    is_grounded = value_str in evidence_lower

            elif expected_type == "number":
                try:
                    num_value = float(value_str)
                    is_grounded = validator._find_numeric_in_text(num_value, evidence_lower)
                except (ValueError, TypeError):
                    is_grounded = value_str in evidence_lower

            elif expected_type == "boolean":
                # For booleans, validation is more lenient - we check evidence relevance
                # rather than exact value grounding (boolean values aren't literal text)

                # Get core concept from field name
                concept = field_name.replace("has_", "").replace("uses_", "").replace("_", " ")
                concept_words = [w for w in concept.split() if len(w) > 2]

                if value_str in ("true", "yes", "1"):
                    # For TRUE: evidence should mention the concept affirmatively
                    # Check if any concept word appears in evidence
                    is_grounded = any(w in evidence_lower for w in concept_words)
                    # Also accept if evidence contains percentage/rate patterns (implies existence)
                    if not is_grounded and expected_type == "boolean":
                        if re.search(r'\d+\.?\d*\s*%', evidence_text):
                            is_grounded = True  # Numbers suggest the thing exists
                else:
                    # For FALSE: evidence should indicate absence/negation
                    negative_patterns = ["no ", "not ", "does not", "n/a", "none", "without", "neither"]
                    is_grounded = any(p in evidence_lower for p in negative_patterns)
                    # Also accept if evidence explicitly states opposite
                    if not is_grounded:
                        # If evidence mentions concept but no numbers, might indicate absence
                        if any(w in evidence_lower for w in concept_words):
                            is_grounded = True  # At least it's about the right topic

            else:
                # Text fields - check for exact or partial match
                is_grounded = value_str in evidence_lower
                if not is_grounded:
                    # Try with underscores replaced by spaces (e.g., "interval_fund" -> "interval fund")
                    normalized_value = value_str.replace("_", " ")
                    is_grounded = normalized_value in evidence_lower
                if not is_grounded:
                    # Try partial quote matching
                    is_grounded = validator._find_partial_quote(value_str, evidence_lower, min_words=2)
                if not is_grounded:
                    # Try matching individual significant words
                    words = [w for w in value_str.replace("_", " ").split() if len(w) > 3]
                    if words:
                        is_grounded = all(w in evidence_lower for w in words)

            signals["grounded"] = is_grounded
            if not is_grounded:
                issues.append(f"Value '{value}' not found in evidence text")

            # Signal 2: Type pattern presence
            type_pattern_found = False
            if expected_type == "percentage":
                type_pattern_found = bool(re.search(r'\d+\.?\d*\s*%', evidence_text))
            elif expected_type == "currency":
                type_pattern_found = bool(re.search(r'\$[\d,]+\.?\d*', evidence_text))
            elif expected_type == "number":
                type_pattern_found = bool(re.search(r'\d+\.?\d*', evidence_text))
            else:
                type_pattern_found = True  # Always true for text/boolean
            signals["type_pattern"] = type_pattern_found

            # Signal 3: Evidence quality (length check)
            signals["sufficient_context"] = len(evidence_text) >= 50
            if not signals["sufficient_context"]:
                issues.append("Evidence quote too short - get more context")

            # Signal 4: Contextual relevance (field keywords in evidence)
            field_keywords = field_name.replace("_", " ").split()
            signals["contextual"] = any(kw in evidence_lower for kw in field_keywords if len(kw) > 2)

            # Signal 5: No contradictory language (for affirmative extractions)
            contradiction_patterns = ["not ", "no ", "n/a", "none", "excluding", "except"]
            has_contradiction = any(p in evidence_lower[:100] for p in contradiction_patterns)
            # Contradiction is bad for positive booleans, ok for negative
            if expected_type == "boolean" and value_str in ("false", "no", "0"):
                signals["no_contradiction"] = True  # Expected for false values
            else:
                signals["no_contradiction"] = not has_contradiction
                if has_contradiction and expected_type != "boolean":
                    issues.append("Evidence contains negative language - verify extraction")

            # Calculate weighted confidence
            weights = {
                "grounded": 0.40,
                "type_pattern": 0.15,
                "sufficient_context": 0.15,
                "contextual": 0.15,
                "no_contradiction": 0.15,
            }
            confidence = sum(
                weights[k] * (1.0 if v else 0.0)
                for k, v in signals.items()
            )

            # Grounding is required - if not grounded, cap confidence
            if not is_grounded:
                confidence = min(confidence, 0.3)

            return ToolResult(
                success=True,
                data={
                    "is_grounded": is_grounded,
                    "confidence": round(confidence, 2),
                    "signals": signals,
                    "issues": issues,
                    "value_validated": value,
                },
            )

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return ToolResult(success=False, error=str(e))

    def _check_normalized_source_grounding(
        self,
        normalized_value: float,
        evidence_lower: str,
        field_name: str,
    ) -> bool:
        """
        Check if a normalized value is grounded via its source expression.

        For fields like max_leverage_pct, the LLM may correctly normalize
        "one-third" to 33.33% or "300% asset coverage" to 33.33%. This method
        verifies the source expression exists even if the normalized value doesn't.

        Args:
            normalized_value: The normalized percentage value (e.g., 33.33)
            evidence_lower: Lowercase evidence text
            field_name: Name of field being validated

        Returns:
            True if a valid source expression is found that normalizes to the value
        """
        if field_name == "max_leverage_pct":
            # Check for leverage source expressions
            for format_type, config in LEVERAGE_FORMATS.items():
                for pattern in config["patterns"]:
                    match = re.search(pattern, evidence_lower)
                    if match:
                        # Found a leverage pattern - verify it normalizes to our value
                        if config.get("fixed_value"):
                            # Fixed value patterns (one-third, 1940 Act)
                            expected_normalized = config["convert"](None)
                            if abs(normalized_value - expected_normalized) < 1.0:
                                logger.debug(
                                    f"Grounding via {format_type}: "
                                    f"source pattern normalizes to {expected_normalized}%"
                                )
                                return True
                        else:
                            # Variable patterns - extract and convert
                            try:
                                raw_value = float(match.group(1))
                                expected_normalized = config["convert"](raw_value)
                                if expected_normalized and abs(normalized_value - expected_normalized) < 1.0:
                                    logger.debug(
                                        f"Grounding via {format_type}: "
                                        f"{raw_value}% → {expected_normalized}% (extracted: {normalized_value}%)"
                                    )
                                    return True
                            except (ValueError, IndexError):
                                continue

            # Also check for fractional expressions
            fraction_patterns = [
                (r"one[\s-]*third", 33.33),
                (r"1/3", 33.33),
                (r"one[\s-]*half", 50.0),
                (r"1/2", 50.0),
                (r"two[\s-]*thirds", 66.67),
                (r"2/3", 66.67),
                (r"one[\s-]*quarter", 25.0),
                (r"1/4", 25.0),
            ]
            for pattern, expected in fraction_patterns:
                if re.search(pattern, evidence_lower):
                    if abs(normalized_value - expected) < 1.0:
                        logger.debug(f"Grounding via fraction pattern: {pattern} → {expected}%")
                        return True

        elif field_name == "hurdle_rate_pct":
            # For hurdle rates, check if quarterly rate was annualized
            # e.g., "1.5% per quarter" → 6% annualized
            quarterly_patterns = [
                r"(\d+(?:\.\d+)?)\s*%?\s*per\s*quarter",
                r"(\d+(?:\.\d+)?)\s*%?\s*quarterly",
            ]
            for pattern in quarterly_patterns:
                match = re.search(pattern, evidence_lower)
                if match:
                    try:
                        quarterly_rate = float(match.group(1))
                        annualized = quarterly_rate * 4
                        if abs(normalized_value - annualized) < 0.5:
                            logger.debug(
                                f"Grounding via annualization: "
                                f"{quarterly_rate}% quarterly → {annualized}% annual"
                            )
                            return True
                    except (ValueError, IndexError):
                        continue

        return False

    def evaluate_retrieval(
        self,
        search_results: list[dict],
        field_name: str,
        field_description: str,
        expected_type: str,
    ) -> ToolResult:
        """
        CRAG (Corrective RAG) evaluation of search results.

        Classifies retrieved content as:
        - CORRECT: Results contain relevant content for extraction
        - INCORRECT: Results are irrelevant, need different search strategy
        - AMBIGUOUS: Unclear, need more context or query decomposition

        Args:
            search_results: List of search result dicts with content/section info
            field_name: Name of field being searched
            field_description: Description of what we're looking for
            expected_type: Expected data type (percentage, currency, boolean, text)

        Returns:
            ToolResult with classification and recommended action
        """
        try:
            if not search_results:
                return ToolResult(
                    success=True,
                    data={
                        "classification": "INCORRECT",
                        "reasoning": "No search results to evaluate",
                        "recommended_action": "Try different keywords or expand search scope",
                        "relevant_chunks": [],
                        "confidence": 0.0,
                    },
                )

            # Analyze search results for relevance signals
            total_chunks = len(search_results)
            relevant_chunks = []
            relevance_signals = []

            # Field-specific keywords to look for
            field_keywords = field_name.replace("_", " ").lower().split()
            # Add common synonyms based on field type
            keyword_expansions = {
                "incentive_fee": ["performance fee", "carried interest", "incentive allocation"],
                "expense_cap": ["expense limitation", "fee waiver", "expense reimbursement"],
                "leverage": ["borrowing", "credit facility", "debt"],
                "repurchase": ["redemption", "tender offer", "liquidity"],
                "distribution": ["dividend", "income distribution"],
                "hurdle": ["preferred return", "benchmark"],
                "high_water_mark": ["loss carryforward", "deficit"],
            }

            expanded_keywords = list(field_keywords)
            for key, expansions in keyword_expansions.items():
                if key in field_name.lower():
                    expanded_keywords.extend(expansions)

            # Score each chunk for relevance
            for result in search_results:
                content = result.get("content", result.get("content_preview", "")).lower()
                chunk_id = result.get("chunk_id", "unknown")
                section = result.get("section", "unknown")

                # Count keyword matches
                keyword_matches = sum(1 for kw in expanded_keywords if kw in content)

                # Check for type-appropriate patterns
                has_value_pattern = False
                if expected_type == "percentage":
                    has_value_pattern = bool(re.search(r'\d+\.?\d*\s*%', content))
                elif expected_type == "currency":
                    has_value_pattern = bool(re.search(r'\$[\d,]+', content))
                elif expected_type == "boolean":
                    has_value_pattern = True  # Booleans don't need specific patterns
                elif expected_type == "number":
                    has_value_pattern = bool(re.search(r'\d+', content))
                else:
                    has_value_pattern = True

                # Determine chunk relevance
                is_relevant = keyword_matches >= 1 and has_value_pattern

                if is_relevant:
                    relevant_chunks.append({
                        "chunk_id": chunk_id,
                        "section": section,
                        "keyword_matches": keyword_matches,
                        "has_value_pattern": has_value_pattern,
                    })
                    relevance_signals.append(f"{section}: {keyword_matches} keyword matches")

            # Classify based on relevance analysis
            relevance_ratio = len(relevant_chunks) / total_chunks if total_chunks > 0 else 0

            if relevance_ratio >= 0.5 and len(relevant_chunks) >= 1:
                classification = "CORRECT"
                reasoning = f"{len(relevant_chunks)}/{total_chunks} chunks appear relevant. Found keywords and value patterns."
                recommended_action = "Proceed to extract value from the most relevant chunks."
                confidence = min(0.9, 0.5 + relevance_ratio * 0.4)

            elif relevance_ratio > 0 and len(relevant_chunks) >= 1:
                classification = "AMBIGUOUS"
                reasoning = f"Only {len(relevant_chunks)}/{total_chunks} chunks appear relevant. Results may be partial."
                recommended_action = "Get more context around relevant chunks, or try a more specific query."
                confidence = 0.4 + relevance_ratio * 0.3

            else:
                classification = "INCORRECT"
                reasoning = f"No chunks contain relevant keywords ({', '.join(field_keywords[:3])}) with value patterns."
                recommended_action = "Try different search terms, synonyms, or search different sections."
                confidence = 0.2

            return ToolResult(
                success=True,
                data={
                    "classification": classification,
                    "reasoning": reasoning,
                    "recommended_action": recommended_action,
                    "relevant_chunks": [c["chunk_id"] for c in relevant_chunks[:5]],
                    "relevance_details": relevant_chunks[:5],
                    "total_evaluated": total_chunks,
                    "relevant_count": len(relevant_chunks),
                    "confidence": confidence,
                },
            )

        except Exception as e:
            logger.error(f"CRAG evaluation error: {e}")
            return ToolResult(success=False, error=str(e))

    def get_stats(self) -> dict:
        """Return tool usage statistics."""
        return {
            "total_searches": self._search_count,
            "chunks_retrieved": self._chunks_retrieved,
        }


# =============================================================================
# AGENT LOOP
# =============================================================================

class AgentAction(str, Enum):
    """Actions the agent can take."""
    SEARCH_KEYWORDS = "search_keywords"
    SEARCH_REGEX = "search_regex"
    SEARCH_SEMANTIC = "search_semantic"  # Semantic/embedding search
    GET_CONTEXT = "get_context"
    GET_STRUCTURE = "get_structure"
    EVALUATE_RETRIEVAL = "evaluate_retrieval"  # CRAG evaluation
    EXTRACT = "extract"
    VALIDATE = "validate"
    STOP_SUCCESS = "stop_success"
    STOP_NOT_FOUND = "stop_not_found"
    STOP_MAX_ITERATIONS = "stop_max_iterations"


@dataclass
class AgentStep:
    """Record of a single agent loop iteration."""
    iteration: int
    thought: str  # Agent's reasoning
    action: AgentAction
    action_input: dict
    observation: str = ""  # Tool result summary (filled after execution)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Tier4ExtractionResult:
    """
    Result from Tier 4 agentic extraction.

    Includes full trace for observability and debugging.
    """
    field_name: str
    value: Optional[Any] = None
    confidence: float = 0.0
    evidence: str = ""
    source_chunk_id: Optional[str] = None
    source_section: Optional[str] = None

    # Observability
    success: bool = False
    stop_reason: AgentAction = AgentAction.STOP_NOT_FOUND
    iterations_used: int = 0
    total_searches: int = 0
    chunks_examined: int = 0
    steps: list[AgentStep] = field(default_factory=list)
    reasoning_trace: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class Tier4Agent:
    """
    Unconstrained agentic extractor for Tier 4.

    Implements a ReAct-style agent loop that can iteratively search
    documents to find information that structured approaches missed.

    Usage:
        agent = Tier4Agent(config, document)
        result = agent.extract(
            field_spec=FieldSpec(name="minimum_investment", ...),
            prior_results=[...],
        )
    """

    def __init__(
        self,
        config: Tier4Config,
        document: ChunkedDocument,
    ):
        """
        Initialize Tier 4 agent.

        Args:
            config: Agent configuration
            document: Document to search
        """
        self.config = config
        self.document = document
        self.tools = AgentTools(document)
        self._client = None
        self._provider = None
        self._rate_limit_config = None
        self._adversarial_validator = None

    @property
    def adversarial_validator(self):
        """Lazy-load validator (lightweight or adversarial based on config)."""
        if self._adversarial_validator is None:
            adv_config = getattr(self.config, 'adversarial_validation', None)
            if adv_config and adv_config.enabled:
                if getattr(adv_config, 'lightweight', True):
                    # Use lightweight validator (faster, cheaper, less strict)
                    from .adversarial_validator import LightweightValidator
                    model = getattr(adv_config, 'lightweight_model', 'gpt-4o-mini')
                    self._adversarial_validator = LightweightValidator(model=model)
                    logger.info(f"[Tier4] Using lightweight validator with {model}")
                else:
                    # Use full adversarial validator (strict)
                    from .adversarial_validator import AdversarialValidator
                    self._adversarial_validator = AdversarialValidator(
                        model=adv_config.model,
                        require_exact_quote=adv_config.require_exact_quote,
                    )
                    logger.info(f"[Tier4] Using adversarial validator with {adv_config.model}")
        return self._adversarial_validator

    @property
    def provider(self) -> str:
        """Get the provider name for the configured model."""
        if self._provider is None:
            from .llm_provider import detect_provider
            self._provider = detect_provider(self.config.model.value).value
        return self._provider

    @property
    def client(self):
        """Lazy-load LLM client."""
        if self._client is None:
            from .llm_provider import create_raw_client, RateLimitConfig

            model_name = self.config.model.value
            # Use moderate rate limiting to avoid API errors
            self._rate_limit_config = RateLimitConfig(delay_between_calls=0.5)
            self._client = create_raw_client(
                provider=self.provider,
                model=model_name,
            )
        return self._client

    def extract(
        self,
        field_spec: FieldSpec,
        prior_results: list[PriorTierResult],
    ) -> Tier4ExtractionResult:
        """
        Run agentic extraction for a single field.

        Args:
            field_spec: Specification of the field to extract
            prior_results: Results from Tiers 0-3

        Returns:
            Tier4ExtractionResult with value (if found) and full trace
        """
        start_time = time.time()

        # Initialize memory
        memory = AgentMemory(
            field_name=field_spec.name,
            prior_tier_results=prior_results,
        )

        # Initialize result
        result = Tier4ExtractionResult(
            field_name=field_spec.name,
        )

        logger.info(f"[Tier4] Starting extraction for '{field_spec.name}'")

        # Main agent loop
        iteration = 0
        while iteration < self.config.max_iterations:
            iteration += 1

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.config.timeout_seconds:
                logger.warning(f"[Tier4] Timeout after {elapsed:.1f}s")
                result.stop_reason = AgentAction.STOP_MAX_ITERATIONS
                break

            # Generate next action via LLM
            step = self._run_agent_step(
                iteration=iteration,
                field_spec=field_spec,
                memory=memory,
                result=result,
            )

            result.steps.append(step)
            memory.add_reasoning(f"Iteration {iteration}: {step.thought}")

            # Check for stop conditions
            if step.action in (
                AgentAction.STOP_SUCCESS,
                AgentAction.STOP_NOT_FOUND,
                AgentAction.STOP_MAX_ITERATIONS,
            ):
                result.stop_reason = step.action
                if step.action == AgentAction.STOP_SUCCESS:
                    result.success = True
                break

            # Handle EXTRACT action - may be accepted or rejected by validation
            if step.action == AgentAction.EXTRACT:
                observation = self._execute_action(step, memory, result, field_spec)
                step.observation = observation

                # Check if extraction was ACCEPTED (grounded) or REJECTED
                if "ACCEPTED" in observation:
                    # Extraction was grounded - success!
                    result.stop_reason = AgentAction.STOP_SUCCESS
                    result.success = True
                    break
                else:
                    # Extraction was REJECTED (not grounded) - continue loop
                    # Add rejection observation so agent can try again
                    memory.add_turn("user", f"Observation:\n{observation}")
                    continue

            # Execute action and add observation to conversation
            observation = self._execute_action(step, memory, result, field_spec)
            step.observation = observation

            # Add observation as a user turn for next iteration
            memory.add_turn("user", f"Observation:\n{observation}")

        # Finalize result
        result.iterations_used = iteration
        result.duration_seconds = time.time() - start_time
        result.reasoning_trace = memory.reasoning_trace

        tool_stats = self.tools.get_stats()
        result.total_searches = tool_stats["total_searches"]
        result.chunks_examined = tool_stats["chunks_retrieved"]

        logger.info(
            f"[Tier4] Completed '{field_spec.name}': "
            f"success={result.success}, iterations={iteration}, "
            f"duration={result.duration_seconds:.1f}s"
        )

        return result

    def _run_agent_step(
        self,
        iteration: int,
        field_spec: FieldSpec,
        memory: AgentMemory,
        result: Tier4ExtractionResult,
    ) -> AgentStep:
        """
        Run a single agent reasoning step.

        Uses the LLM to decide what action to take next based on
        current state and prior observations.

        Args:
            iteration: Current iteration number
            field_spec: Field being extracted
            memory: Agent's working memory
            result: Current extraction result (for accumulated findings)

        Returns:
            AgentStep with thought, action, and action_input
        """
        from .llm_provider import call_llm_json, resolve_model_name

        # Build prompts
        system_prompt = self._build_system_prompt(field_spec)
        user_prompt = self._build_user_prompt(iteration, field_spec, memory, result)

        # Add current turn to conversation history
        memory.add_turn("user", user_prompt)

        # Build messages from conversation history
        messages = memory.get_messages_for_llm(system_prompt)

        # Estimate token count (~4 chars per token) and warn if approaching limit
        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = total_chars // 4
        if estimated_tokens > 150_000:
            logger.warning(
                f"[Tier4] [CONTEXT WARNING] Estimated ~{estimated_tokens:,} tokens "
                f"at iteration {iteration} for '{field_spec.name}' "
                f"({len(memory.conversation_history)} turns, {total_chars:,} chars). "
                f"Approaching 200K limit."
            )

        # Call LLM with retry logic for parsing errors
        parsed_response = None
        last_error = None

        for attempt in range(MAX_PARSE_RETRIES + 1):
            try:
                logger.debug(
                    f"[Tier4] Iteration {iteration}, LLM call attempt {attempt + 1}"
                )

                # Make the LLM call
                response = call_llm_json(
                    client=self.client,
                    provider=self.provider,
                    model=resolve_model_name(self.config.model.value),
                    messages=messages,
                    rate_limit=self._rate_limit_config,
                )

                # Parse and validate response
                parsed_response = self._parse_agent_response(response)

                if parsed_response:
                    break

            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {e}"
                logger.warning(f"[Tier4] {last_error}, retrying...")
            except Exception as e:
                error_str = str(e).lower()
                is_context_overflow = (
                    "prompt too long" in error_str
                    or "context_length" in error_str
                    or "maximum context" in error_str
                    or ("token" in error_str and ("limit" in error_str or "maximum" in error_str))
                )
                if is_context_overflow:
                    last_error = f"CONTEXT OVERFLOW: {e}"
                    logger.error(
                        f"[Tier4] [CONTEXT OVERFLOW] '{field_spec.name}' at iteration "
                        f"{iteration}: context exceeded model limit. "
                        f"Estimated ~{estimated_tokens:,} tokens, "
                        f"{len(memory.conversation_history)} conversation turns. "
                        f"Error: {e}"
                    )
                else:
                    last_error = f"LLM call error: {e}"
                    logger.error(f"[Tier4] {last_error}")
                # Don't retry on non-parse errors
                break

        # Handle failure to get valid response
        if parsed_response is None:
            logger.error(
                f"[Tier4] Failed to get valid response after {MAX_PARSE_RETRIES + 1} attempts. "
                f"Last error: {last_error}"
            )
            return AgentStep(
                iteration=iteration,
                thought=f"[ERROR] Failed to get valid LLM response: {last_error}",
                action=AgentAction.STOP_NOT_FOUND,
                action_input={"reason": f"LLM error: {last_error}"},
            )

        # Extract components from parsed response
        thought = parsed_response.get("thought", "No reasoning provided")
        action_str = parsed_response.get("action", "stop_not_found")
        action_input = parsed_response.get("action_input", {})

        # Map action string to enum
        action = self._map_action_string(action_str)

        # Add assistant response to conversation history
        memory.add_turn("assistant", json.dumps(parsed_response))

        logger.info(
            f"[Tier4] Iteration {iteration}: action={action.value}, "
            f"thought={thought[:80]}..."
        )

        return AgentStep(
            iteration=iteration,
            thought=thought,
            action=action,
            action_input=action_input,
        )

    def _parse_agent_response(self, response: dict) -> Optional[dict]:
        """
        Parse and validate agent response from LLM.

        Args:
            response: Raw JSON response from LLM

        Returns:
            Validated response dict or None if invalid
        """
        if not isinstance(response, dict):
            logger.warning(f"[Tier4] Response is not a dict: {type(response)}")
            return None

        # Check required fields
        if "action" not in response:
            logger.warning("[Tier4] Response missing 'action' field")
            return None

        # Validate action
        action = response.get("action", "").lower().strip()
        if action not in VALID_ACTIONS:
            logger.warning(f"[Tier4] Invalid action '{action}', valid: {VALID_ACTIONS}")
            # Try to recover - check if it's a close match
            for valid in VALID_ACTIONS:
                if valid in action or action in valid:
                    response["action"] = valid
                    break
            else:
                return None

        # Ensure action_input exists
        if "action_input" not in response:
            response["action_input"] = {}

        # Ensure thought exists
        if "thought" not in response:
            response["thought"] = "No reasoning provided"

        return response

    def _map_action_string(self, action_str: str) -> AgentAction:
        """
        Map action string to AgentAction enum.

        Args:
            action_str: Action string from LLM response

        Returns:
            Corresponding AgentAction enum value
        """
        action_str = action_str.lower().strip()

        mapping = {
            "search_keywords": AgentAction.SEARCH_KEYWORDS,
            "search_regex": AgentAction.SEARCH_REGEX,
            "search_semantic": AgentAction.SEARCH_SEMANTIC,
            "get_context": AgentAction.GET_CONTEXT,
            "get_structure": AgentAction.GET_STRUCTURE,
            "evaluate_retrieval": AgentAction.EVALUATE_RETRIEVAL,
            "extract": AgentAction.EXTRACT,
            "validate": AgentAction.VALIDATE,
            "stop_success": AgentAction.STOP_SUCCESS,
            "stop_not_found": AgentAction.STOP_NOT_FOUND,
        }

        return mapping.get(action_str, AgentAction.STOP_NOT_FOUND)

    def _execute_action(
        self,
        step: AgentStep,
        memory: AgentMemory,
        result: Tier4ExtractionResult,
        field_spec: FieldSpec,
    ) -> str:
        """
        Execute the agent's chosen action.

        Args:
            step: The AgentStep containing action and input
            memory: Agent memory to update with results
            result: Extraction result to update if extracting
            field_spec: The field specification (for validation)

        Returns:
            Observation string summarizing the result
        """
        action = step.action
        inputs = step.action_input

        if action == AgentAction.SEARCH_KEYWORDS:
            tool_result = self.tools.search_by_keywords(
                keywords=inputs.get("keywords", []),
                max_chunks=inputs.get("max_chunks", 10),
                section_filter=inputs.get("section_filter"),
            )

            # Record in memory
            attempt = SearchAttempt(
                query=str(inputs.get("keywords", [])),
                search_type="keyword",
                chunks_returned=len(tool_result.data.get("matches", [])) if tool_result.success else 0,
                status=SearchResultStatus.FOUND_RELEVANT if tool_result.success and tool_result.data.get("matches") else SearchResultStatus.NO_RESULTS,
            )
            memory.add_search(attempt)

            if tool_result.success:
                matches = tool_result.data.get("matches", [])
                memory.last_tool_results = tool_result.data

                # Build detailed observation for agent
                if matches:
                    obs_lines = [f"Found {len(matches)} chunks matching keywords."]
                    for i, m in enumerate(matches[:5]):  # Show top 5
                        obs_lines.append(
                            f"\n[{i+1}] Section: {m['section']}, Chunk: {m['chunk_id']}, Score: {m['score']}"
                        )
                        # Include more content preview for agent to evaluate (increased from 300 to 800)
                        preview = m.get('content_preview', '')[:800]
                        obs_lines.append(f"    Preview: {preview}...")
                    return "\n".join(obs_lines)
                return "No matching chunks found."
            return f"Search failed: {tool_result.error}"

        elif action == AgentAction.SEARCH_REGEX:
            tool_result = self.tools.search_by_regex(
                pattern=inputs.get("pattern", ""),
                max_chunks=inputs.get("max_chunks", 10),
                section_filter=inputs.get("section_filter"),
            )

            attempt = SearchAttempt(
                query=inputs.get("pattern", ""),
                search_type="regex",
                chunks_returned=len(tool_result.data.get("matches", [])) if tool_result.success else 0,
                status=SearchResultStatus.FOUND_RELEVANT if tool_result.success and tool_result.data.get("matches") else SearchResultStatus.NO_RESULTS,
            )
            memory.add_search(attempt)

            if tool_result.success:
                matches = tool_result.data.get("matches", [])
                memory.last_tool_results = tool_result.data

                if matches:
                    obs_lines = [f"Regex found {len(matches)} chunks with matches."]
                    for i, m in enumerate(matches[:5]):
                        obs_lines.append(
                            f"\n[{i+1}] Section: {m['section']}, Chunk: {m['chunk_id']}"
                        )
                        obs_lines.append(f"    Matches: {m.get('matches', [])[:3]}")
                        preview = m.get('content_preview', '')[:800]  # Increased from 300 to 800
                        obs_lines.append(f"    Preview: {preview}...")
                    return "\n".join(obs_lines)
                return "No regex matches found."
            return f"Regex search failed: {tool_result.error}"

        elif action == AgentAction.SEARCH_SEMANTIC:
            tool_result = self.tools.search_semantic(
                query=inputs.get("query", ""),
                max_chunks=inputs.get("max_chunks", 10),
                section_filter=inputs.get("section_filter"),
                similarity_threshold=inputs.get("similarity_threshold", 0.3),
            )

            attempt = SearchAttempt(
                query=inputs.get("query", ""),
                search_type="semantic",
                chunks_returned=len(tool_result.data.get("matches", [])) if tool_result.success else 0,
                status=SearchResultStatus.FOUND_RELEVANT if tool_result.success and tool_result.data.get("matches") else SearchResultStatus.NO_RESULTS,
            )
            memory.add_search(attempt)

            if tool_result.success:
                matches = tool_result.data.get("matches", [])
                memory.last_tool_results = tool_result.data

                if matches:
                    obs_lines = [f"Semantic search found {len(matches)} similar chunks."]
                    for i, m in enumerate(matches[:5]):
                        obs_lines.append(
                            f"\n[{i+1}] Section: {m['section']}, Chunk: {m['chunk_id']}, Similarity: {m['similarity']:.3f}"
                        )
                        preview = m.get('content_preview', '')[:800]
                        obs_lines.append(f"    Preview: {preview}...")
                    return "\n".join(obs_lines)
                return "No semantically similar chunks found."
            return f"Semantic search failed: {tool_result.error}"

        elif action == AgentAction.GET_CONTEXT:
            tool_result = self.tools.get_chunk_with_context(
                chunk_id=inputs.get("chunk_id", ""),
                context_before=inputs.get("context_before", 1),
                context_after=inputs.get("context_after", 1),
            )

            if tool_result.success:
                chunks = tool_result.data.get("chunks", [])
                memory.last_tool_results = tool_result.data

                obs_lines = [
                    f"Retrieved {len(chunks)} chunks from section '{tool_result.data.get('section')}'."
                ]
                for chunk in chunks:
                    marker = ">>>" if chunk.get("is_target") else "   "
                    obs_lines.append(f"\n{marker} Chunk {chunk['chunk_id']} (index {chunk.get('chunk_index', '?')}):")
                    # Show full chunk content (up to 2500 chars) so agent can see all details
                    obs_lines.append(chunk.get("content", "")[:2500])
                return "\n".join(obs_lines)
            return f"Context retrieval failed: {tool_result.error}"

        elif action == AgentAction.GET_STRUCTURE:
            tool_result = self.tools.get_document_structure()

            if tool_result.success:
                memory.last_tool_results = tool_result.data
                sections = tool_result.data.get("sections", [])

                obs_lines = [
                    f"Document has {tool_result.data['total_sections']} sections, "
                    f"{tool_result.data['total_chunks']} total chunks."
                ]
                obs_lines.append("\nSections:")
                for s in sections[:20]:  # Show first 20 sections
                    obs_lines.append(
                        f"  - {s['title']} ({s['chunk_count']} chunks, chunk_range {s['chunk_range']})"
                    )
                if len(sections) > 20:
                    obs_lines.append(f"  ... and {len(sections) - 20} more sections")
                return "\n".join(obs_lines)
            return f"Structure retrieval failed: {tool_result.error}"

        elif action == AgentAction.EVALUATE_RETRIEVAL:
            # CRAG evaluation of search results
            search_results = inputs.get("search_results", [])

            # If no explicit search_results provided, use last tool results
            if not search_results and memory.last_tool_results:
                last_results = memory.last_tool_results
                if "matches" in last_results:
                    search_results = last_results["matches"]
                elif "chunks" in last_results:
                    search_results = last_results["chunks"]

            tool_result = self.tools.evaluate_retrieval(
                search_results=search_results,
                field_name=field_spec.name,
                field_description=field_spec.description,
                expected_type=field_spec.expected_type,
            )

            if tool_result.success:
                data = tool_result.data
                memory.last_tool_results = data

                classification = data["classification"]
                reasoning = data["reasoning"]
                action_rec = data["recommended_action"]
                relevant = data.get("relevant_chunks", [])

                return (
                    f"CRAG Evaluation: {classification}\n"
                    f"Reasoning: {reasoning}\n"
                    f"Recommendation: {action_rec}\n"
                    f"Relevant chunks: {relevant[:5] if relevant else 'None identified'}"
                )
            return f"CRAG evaluation failed: {tool_result.error}"

        elif action == AgentAction.VALIDATE:
            tool_result = self.tools.validate_extraction(
                value=inputs.get("value"),
                evidence_text=inputs.get("evidence", ""),
                expected_type=inputs.get("expected_type", field_spec.expected_type),
                field_name=field_spec.name,
            )

            if tool_result.success:
                data = tool_result.data
                memory.last_tool_results = data
                signals = data.get("signals", {})
                issues = data.get("issues", [])
                return (
                    f"Validation result: grounded={data['is_grounded']}, "
                    f"confidence={data['confidence']:.2f}, signals={signals}"
                    + (f", issues={issues}" if issues else "")
                )
            return f"Validation failed: {tool_result.error}"

        elif action == AgentAction.EXTRACT:
            # This is the final extraction action
            value = inputs.get("value")
            evidence = inputs.get("evidence", "")
            chunk_id = inputs.get("chunk_id")
            agent_confidence = inputs.get("confidence", 0.8)

            # MANDATORY VALIDATION: Run grounding check before accepting extraction
            validation_result = self.tools.validate_extraction(
                value=value,
                evidence_text=evidence,
                expected_type=field_spec.expected_type,
                field_name=field_spec.name,
            )

            if not validation_result.success:
                return f"Validation error: {validation_result.error}. Try again with better evidence."

            validation_data = validation_result.data
            is_grounded = validation_data.get("is_grounded", False)
            validated_confidence = validation_data.get("confidence", 0.0)
            issues = validation_data.get("issues", [])
            signals = validation_data.get("signals", {})

            # REJECT if not grounded - agent must find better evidence
            if not is_grounded:
                logger.warning(
                    f"[Tier4] REJECTED extraction: '{value}' not grounded in evidence. "
                    f"Signals: {signals}, Issues: {issues}"
                )
                return (
                    f"EXTRACTION REJECTED: Value '{value}' not grounded in evidence text. "
                    f"Issues: {'; '.join(issues) if issues else 'Value not found in text'}. "
                    f"Signals: {signals}. "
                    f"You must search for better evidence that explicitly contains this value, "
                    f"or use stop_not_found if you've exhausted all options."
                )

            # ACCEPT - extraction is grounded (basic check passed)
            # Use the validated confidence (multi-signal) rather than agent's stated confidence
            final_confidence = validated_confidence

            # ADVERSARIAL VALIDATION: Additional LLM-based validation for booleans/all fields
            adversarial_result = None
            adv_config = getattr(self.config, 'adversarial_validation', None)
            should_adversarial_validate = (
                adv_config
                and adv_config.enabled
                and self.adversarial_validator
                and (
                    adv_config.validate_all
                    or (adv_config.validate_booleans and field_spec.expected_type == "boolean")
                )
            )

            if should_adversarial_validate:
                logger.info(f"[Tier4] Running adversarial validation for '{field_spec.name}'")
                adversarial_result = self.adversarial_validator.validate(
                    field_name=field_spec.name,
                    value=value,
                    evidence=evidence,
                    expected_type=field_spec.expected_type,
                )

                if not adversarial_result.is_valid:
                    problems_str = "; ".join(adversarial_result.problems) if adversarial_result.problems else "Validation failed"
                    logger.warning(
                        f"[Tier4] REJECTED by adversarial validation: '{value}' for {field_spec.name}. "
                        f"Problems: {problems_str}"
                    )
                    return (
                        f"EXTRACTION REJECTED by adversarial validation: Value '{value}' failed verification. "
                        f"Problems found: {problems_str}. "
                        f"Reasoning: {adversarial_result.reasoning}. "
                        f"You must search for better evidence with explicit support for this value, "
                        f"or use stop_not_found if the value cannot be verified."
                    )

                # Update confidence based on adversarial validation
                final_confidence = min(final_confidence, adversarial_result.confidence)
                logger.info(
                    f"[Tier4] Adversarial validation PASSED for '{field_spec.name}': "
                    f"confidence={adversarial_result.confidence:.2f}, quote='{adversarial_result.supporting_quote}'"
                )

            # Find source section from chunk_id
            source_section = ""
            if chunk_id:
                for section in self.document.chunked_sections:
                    for chunk in section.chunks:
                        if chunk.chunk_id == chunk_id:
                            source_section = section.section_title
                            break

            # Update the extraction result
            result.value = value
            result.evidence = evidence
            result.confidence = final_confidence
            result.source_chunk_id = chunk_id
            result.source_section = source_section

            # Add to candidate extractions in memory with validation details
            validation_details = {
                "is_grounded": is_grounded,
                "signals": signals,
                "agent_confidence": agent_confidence,
                "validated_confidence": validated_confidence,
            }
            if adversarial_result:
                validation_details["adversarial"] = adversarial_result.to_dict()

            memory.candidate_extractions.append({
                "value": value,
                "evidence": evidence,
                "confidence": final_confidence,
                "chunk_id": chunk_id,
                "validation": validation_details,
            })

            logger.info(
                f"[Tier4] ACCEPTED extraction: {value} "
                f"(agent_conf={agent_confidence}, validated_conf={final_confidence:.2f}, grounded={is_grounded})"
            )

            return (
                f"Extraction ACCEPTED: value={value}, confidence={final_confidence:.2f}, "
                f"grounded={is_grounded}, signals={signals}"
            )

        elif action == AgentAction.STOP_NOT_FOUND:
            reason = inputs.get("reason", "Field not found in document")
            return f"Stopping: {reason}"

        elif action == AgentAction.STOP_SUCCESS:
            return "Extraction complete."

        return "Unknown action"

    def _build_system_prompt(self, field_spec: FieldSpec) -> str:
        """Build the system prompt for the agent."""
        hints_str = "\n".join(f"  - {h}" for h in field_spec.extraction_hints) if field_spec.extraction_hints else "  None"

        # Get field-specific taxonomy if available
        taxonomy = get_field_taxonomy(field_spec.name)
        taxonomy_section = ""
        if taxonomy:
            taxonomy_section = f"""
## FIELD-SPECIFIC TAXONOMY

{taxonomy}

---
"""

        return f"""You are a precise document extraction agent using the ReAct (Reasoning + Acting) pattern. Your task is to find a specific value in a financial SEC filing document.
{taxonomy_section}

## FIELD TO EXTRACT

Name: {field_spec.name}
Description: {field_spec.description}
Expected type: {field_spec.expected_type}
Alternative phrasings: {', '.join(field_spec.aliases) if field_spec.aliases else 'None'}
Example values: {', '.join(field_spec.examples) if field_spec.examples else 'None'}

Extraction hints:
{hints_str}

## AVAILABLE TOOLS

1. **get_structure** - Get document section structure. USE THIS FIRST to understand document layout.
   Input: {{}}

2. **search_keywords** - Search for chunks containing keywords.
   Input: {{"keywords": ["term1", "term2"], "max_chunks": 10, "section_filter": null}}
   - Use 3-5 specific, relevant keywords
   - **IMPORTANT**: Do your FIRST search with section_filter: null to search ALL sections
   - Only use section_filter AFTER you've seen what sections contain matches
   - Don't assume which sections have the info - the data may be in unexpected places

3. **search_regex** - Search with regex pattern for precise matching.
   Input: {{"pattern": "\\\\$[\\\\d,]+", "max_chunks": 10, "section_filter": null}}
   - Use for currency amounts, percentages, specific formats
   - Escape backslashes in JSON (use \\\\ for \\)

4. **search_semantic** - Semantic similarity search using embeddings.
   Input: {{"query": "incentive fee structure and performance allocation", "max_chunks": 10, "section_filter": null, "similarity_threshold": 0.3}}
   - Use when keyword search fails or for conceptual queries
   - query: Natural language description of what you're looking for (be descriptive)
   - similarity_threshold: Minimum similarity score (0.0-1.0, default 0.3)
   - Finds semantically similar content even with different wording
   - **BEST FOR**: Finding content that discusses a concept without using exact keywords
   - Example: query="fund leverage and borrowing limits" might find "The Fund may borrow up to 33% of assets"

5. **get_context** - Expand a chunk with surrounding context.
   Input: {{"chunk_id": "chunk_123", "context_before": 2, "context_after": 2}}
   - Use when you find a promising chunk but need more context

6. **evaluate_retrieval** - CRAG evaluation: Classify search results before extraction.
   Input: {{}}  (automatically uses last search results)
   - Returns: CORRECT (proceed to extract), INCORRECT (try different search), AMBIGUOUS (get more context)
   - **USE AFTER EVERY SEARCH** to assess if results are relevant before extracting
   - Identifies which chunks are most relevant and recommends next action

7. **validate** - Check if a value is grounded in evidence text.
   Input: {{"value": "10%", "evidence": "The incentive fee is 10% of profits", "expected_type": "percentage"}}
   - Use before final extraction to verify

8. **extract** - Submit final extraction. This completes the task successfully.
   Input: {{"value": "extracted_value", "evidence": "quote from document", "chunk_id": "chunk_123", "confidence": 0.9}}
   - Only use when you have found and validated the value
   - Include exact evidence quote from document
   - **IMPORTANT**: Extraction is automatically validated. If not grounded, it will be REJECTED.

9. **stop_not_found** - Stop and report field not found.
   Input: {{"reason": "explanation of why not found"}}
   - Use ONLY when you've exhausted ALL reasonable search strategies

## RESPONSE FORMAT

You MUST respond with valid JSON only:
{{
    "thought": "Your step-by-step reasoning: What did I learn? What should I try next? Why?",
    "action": "action_name",
    "action_input": {{ ... parameters ... }}
}}

## CRITICAL RULES FOR SEC FILINGS

1. ALWAYS start with get_structure to understand the document layout
2. REASON BEFORE ACTING - explain your logic in "thought" before choosing an action
3. **USE evaluate_retrieval AFTER SEARCHES** - Classify results as CORRECT/INCORRECT/AMBIGUOUS before proceeding
4. DO NOT repeat searches that returned no results - try different keywords/sections
5. **READ ALL SEARCH RESULTS CAREFULLY** - Don't just look at the first result! The target value might be in result #2, #3, or #4
6. When you find relevant content in a search preview, use get_context to see surrounding text
7. If prior tiers searched certain sections and failed, try DIFFERENT sections
8. The evidence field in extract must be an EXACT QUOTE from the document
9. **SCAN PREVIEWS FOR ACTUAL VALUES** - Look for percentage patterns (X%) in ALL search result previews before choosing which chunk to expand
10. **EXTRACTION IS VALIDATED AUTOMATICALLY** - If your extraction isn't grounded in the evidence text, it will be REJECTED and you must try again

## CRITICAL: FEE TABLE SECTIONS CONTAIN KEY INFORMATION

**The ACTUAL fee terms are in Fee Table sections and their footnotes, not in summary sections.**

Key sections to search for fee information:
- **"Shareholder Transaction Expenses Table"** - Contains management fees, incentive fees, sales loads
- **"Annual Expenses Table"** - Contains expense ratios and fee breakdowns
- **"Fee Table Note"** - Contains footnote explanations of fee structures
- **"Annual Expenses Note"** - More fee structure details

**Important patterns:**
- A dash (—) or 0% in a fee table means "no fees accrued YET" - NOT "no fee exists"
- The footnote numbers (1), (2), (3) etc. reference detailed explanations in the same section or nearby
- Fee structures, hurdle rates, and catch-up provisions are described INLINE with footnote text
- Look for "(4)" or similar footnote markers followed by fee structure explanation

Example: "The Fund will pay the Adviser, quarterly in arrears, 10% of..." appears in "Shareholder Transaction Expenses Table" section, not in a section titled "Incentive Fee".

## BOOLEAN FIELD RULES

For boolean fields (has_incentive_fee, has_expense_cap, uses_leverage, etc.):
- A table showing 0%, None, or — does NOT mean the answer is "false"
- You MUST search footnotes and explanatory sections before concluding "false"
- If you find fee structure DESCRIPTION (even if current value is 0), the answer is "true"

**IMPORTANT: When to extract "false" vs "stop_not_found":**
- Use **extract with value="false"** when you have EVIDENCE that the fund does NOT have the feature
  - Example: "The fund does not charge an incentive fee" → extract false
  - Example: Only AFFE mentioned, no fund-level fee found after thorough search → extract false
  - Example: Advisory Agreement section exists but mentions no incentive fee → extract false
- Use **stop_not_found** ONLY when you cannot determine true OR false
  - Example: The relevant section is missing from the document entirely
  - Example: The document is corrupted or incomplete

For fund-of-funds: If you find ONLY AFFE (underlying fund fees) and NO fund-level incentive fee after searching Fee Structure and Advisory Agreement sections, extract "false" - do NOT use stop_not_found.

## CRITICAL: FUND-LEVEL FEES vs AFFE (FUND-OF-FUNDS)

**This is the most common extraction error. READ CAREFULLY.**

Many funds are "fund-of-funds" that invest in underlying PE/credit funds. These documents contain TWO TYPES of fees:

1. **FUND-LEVEL FEES** (what we want to extract):
   - Fees the fund ITSELF charges to its shareholders
   - Found in "Fee Structure", "Advisory Agreement", "Management Agreement" sections
   - Described as "the Fund pays the Adviser..." or "we will pay..."
   - This is the incentive fee WE are looking for

2. **AFFE - Acquired Fund Fees and Expenses** (DO NOT extract these):
   - Fees charged by UNDERLYING funds the fund invests in
   - Found in "AFFE Note", "Acquired Fund Fees", or footnotes about underlying investments
   - Described as "Investment Funds generally charge..." or "underlying funds typically charge..."
   - Often shows ranges like "10-20%" or "up to 20%"
   - These are NOT the fund's own fees

**How to distinguish:**
- If text says "Investment Funds charge" or "underlying funds" → This is AFFE, SKIP IT
- If text says "the Fund pays" or "we pay the Adviser" → This is fund-level, EXTRACT IT
- If a fund has NO fund-level incentive fee but invests in funds that do → has_incentive_fee = FALSE
- AFFE appears in Annual Expenses Table but is INDIRECT - the fund itself may not charge incentive fees

**Example (StepStone - fund-of-funds):**
- "Investment Funds generally charge...incentive allocations of approximately 15-20%" → AFFE, DO NOT EXTRACT
- If no text says "the Fund charges an incentive fee" → has_incentive_fee = FALSE, incentive_fee_pct = null

**Example (Blackstone - direct fund):**
- "we will pay the Adviser quarterly in arrears 12.5% of our Pre-Incentive Fee Net Investment Income" → FUND-LEVEL, EXTRACT 12.5%

## REPURCHASE/LIQUIDITY FIELD RULES

For interval funds and tender offer funds:
- Search for sections containing: "interval", "repurchase", "tender offer", "liquidity", "redemption"
- Look for Rule 23c-3 references (interval fund rule requiring quarterly repurchases)
- Repurchase terms are often in "Repurchase Offers", "Interval Fund", or "Liquidity" sections

## CRITICAL: ANTI-HALLUCINATION RULES

**NEVER fabricate, infer, or guess values. Every extraction must be grounded in explicit document text.**

1. **ONLY extract values you can QUOTE from the document**
   - If you cannot point to the exact text containing the value, DO NOT extract it
   - The value must appear explicitly - not be implied or inferred

2. **DO NOT use patterns from other funds**
   - Each fund has unique fee structures, minimums, and terms
   - Just because Blackstone has $2,500 minimums doesn't mean Blue Owl does
   - Ignore any "typical" or "common" values you know from training data

3. **NULL is the correct answer when:**
   - The document says values are "determined by financial intermediaries"
   - The document says "no minimum" or "set by selling agents"
   - You searched thoroughly but found NO explicit value in the document
   - The field simply isn't disclosed in this prospectus

4. **SHARE CLASS MINIMUMS are especially prone to hallucination:**
   - Many funds delegate minimum investments to intermediaries
   - Class names (S, I, D) do NOT imply specific minimums
   - Extract null unless you see EXACT dollar amounts like "$2,500" in text

5. **VALIDATE before extracting:**
   - Use the validate tool to confirm the value appears in evidence
   - If validation fails, DO NOT extract - return null instead

## STOPPING CONDITIONS

Stop with extract when:
- You found the value AND it's grounded in document text AND validation passed

Stop with stop_not_found ONLY when ALL of these are true:
- You've searched the main content sections (Fee Structure, Fee Table, etc.)
- You've searched footnote sections (sections with "Note" or "Footnote" in title)
- You've tried at least 5 different keyword combinations
- You've examined chunks from at least 3 different sections
- The document genuinely doesn't contain this information

**For nullable fields like minimum_investment: Use extract with value=null (not stop_not_found) when the document explicitly delegates to intermediaries or states "no minimum".**
"""

    def _build_user_prompt(
        self,
        iteration: int,
        field_spec: FieldSpec,
        memory: AgentMemory,
        result: Tier4ExtractionResult,
    ) -> str:
        """Build the user prompt with current state."""
        context = memory.to_context_string()

        # Calculate remaining iterations
        remaining = self.config.max_iterations - iteration

        # Build iteration-specific guidance
        if iteration == 1:
            guidance = "This is your first action. Start by getting the document structure to understand what sections are available."
        elif remaining <= 2:
            guidance = f"WARNING: Only {remaining} iterations remaining. If you haven't found the value, consider using stop_not_found with a clear explanation."
        elif len(memory.search_history) >= 3 and not memory.candidate_extractions:
            guidance = "You've done several searches without finding candidates. Try different keywords, different sections, or consider that the field may not be in this document."
        else:
            guidance = "Continue your search strategy. Remember to use get_context on promising chunks before extracting."

        return f"""## CURRENT STATE

Iteration: {iteration}/{self.config.max_iterations} ({remaining} remaining)

{context}

## GUIDANCE

{guidance}

## YOUR TASK

Based on the above context, decide your next action. Respond with JSON only:
{{"thought": "...", "action": "...", "action_input": {{...}}}}"""


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def create_prior_tier_summary(
    tier: int,
    status: str,
    sections_searched: list[str] = None,
    extraction_result: Any = None,
    failure_reason: str = None,
) -> PriorTierResult:
    """
    Helper to create PriorTierResult from extraction trace data.

    Args:
        tier: Tier number (0-3)
        status: "found", "not_found", "partial", "error"
        sections_searched: List of section titles searched
        extraction_result: The extracted value (if any)
        failure_reason: Why extraction failed (if applicable)

    Returns:
        PriorTierResult for passing to Tier 4
    """
    return PriorTierResult(
        tier=tier,
        status=status,
        sections_searched=sections_searched or [],
        extraction_result=extraction_result,
        failure_reason=failure_reason,
    )


# Field specifications for extraction targets
# Maps to ground truth fields defined in configs/ground_truth/*.json
FIELD_SPECS = {
    # =========================================================================
    # SHARE CLASS FIELDS
    # =========================================================================
    "minimum_investment": FieldSpec(
        name="minimum_investment",
        description="The minimum initial investment amount required to purchase shares at the FUND LEVEL",
        expected_type="currency",
        examples=["$2,500", "$1,000,000", "$25,000", "null"],
        aliases=["minimum initial investment", "initial investment minimum", "min investment"],
        extraction_hints=[
            "Often found in 'Plan of Distribution' or 'Purchasing Shares' sections",
            "May vary by share class (Class S, Class D, Class I)",
            "Look for patterns like 'minimum initial investment of $X'",
            "CRITICAL: Extract ONLY if an EXPLICIT dollar amount appears in this document",
            "CRITICAL: Return null if the document says minimums are 'set by financial intermediaries' or 'determined by selling agents'",
            "CRITICAL: Return null if document says 'no minimum investment' at fund level",
            "DO NOT infer values from similar funds - each fund has different minimums",
            "Common values are $2,500 (retail), $25,000 (high-net-worth), $1,000,000 (institutional) - but ONLY extract if you see the EXACT number in THIS document",
            "If you see class names (Class S, Class I) but NO dollar amounts for minimums, return null",
        ],
    ),
    "minimum_additional_investment": FieldSpec(
        name="minimum_additional_investment",
        description="The minimum additional/subsequent investment amount for existing shareholders at the FUND LEVEL",
        expected_type="currency",
        examples=["$500", "$1,000", "$100", "null"],
        aliases=["minimum subsequent investment", "additional investment minimum", "min additional investment"],
        extraction_hints=[
            "Often found near minimum initial investment information",
            "Usually lower than initial investment minimum",
            "Look for patterns like 'minimum additional investment of $X' or 'subsequent investment of $X'",
            "CRITICAL: Extract ONLY if an EXPLICIT dollar amount appears in this document",
            "CRITICAL: Return null if the document says minimums are 'set by financial intermediaries'",
            "DO NOT infer values from similar funds - each fund has different minimums",
            "If you see class names but NO dollar amounts for additional investments, return null",
        ],
    ),

    # =========================================================================
    # SHARE CLASS FEE FIELDS (from fee tables)
    # =========================================================================
    "management_fee_pct": FieldSpec(
        name="management_fee_pct",
        description="Annual management/advisory fee as percentage of net assets",
        expected_type="percentage",
        examples=["1.25%", "1.50%", "2.00%", "null"],
        aliases=["advisory fee", "management fee", "base management fee", "investment advisory fee"],
        extraction_hints=[
            "Look in 'Annual Fund Operating Expenses' table or 'Fee Table'",
            "Usually labeled 'Management Fee' or 'Advisory Fee'",
            "Expressed as % of net assets",
            "May vary by share class",
        ],
    ),
    "affe_pct": FieldSpec(
        name="affe_pct",
        description="Acquired fund fees and expenses as percentage (for fund-of-funds)",
        expected_type="percentage",
        examples=["0.50%", "1.00%", "null"],
        aliases=["acquired fund fees", "AFFE", "underlying fund expenses"],
        extraction_hints=[
            "Look in 'Annual Fund Operating Expenses' table",
            "Only applies to fund-of-funds structures",
            "Labeled 'Acquired Fund Fees and Expenses' or 'AFFE'",
            "Return null if not a fund-of-funds",
        ],
    ),
    "interest_expense_pct": FieldSpec(
        name="interest_expense_pct",
        description="Interest expenses on borrowings as percentage of net assets",
        expected_type="percentage",
        examples=["0.25%", "0.50%", "null"],
        aliases=["interest expense", "borrowing costs", "interest on borrowings"],
        extraction_hints=[
            "Look in 'Annual Fund Operating Expenses' table",
            "Labeled 'Interest Expenses on Borrowings' or similar",
            "Return null if fund does not use leverage",
        ],
    ),
    "other_expenses_pct": FieldSpec(
        name="other_expenses_pct",
        description="Other annual expenses as percentage of net assets",
        expected_type="percentage",
        examples=["0.25%", "0.50%", "1.00%", "null"],
        aliases=["other expenses", "other annual expenses", "remaining expenses"],
        extraction_hints=[
            "Look in 'Annual Fund Operating Expenses' table",
            "Usually a catch-all category after management fee and specific expenses",
        ],
    ),
    "total_expense_ratio_pct": FieldSpec(
        name="total_expense_ratio_pct",
        description="Total annual expenses before fee waivers as percentage of net assets",
        expected_type="percentage",
        examples=["2.50%", "3.00%", "4.50%", "null"],
        aliases=["total annual expenses", "gross expense ratio", "total expenses"],
        extraction_hints=[
            "Look in 'Annual Fund Operating Expenses' table",
            "This is the GROSS total BEFORE any fee waivers or reimbursements",
            "Sum of all expense line items",
        ],
    ),
    "net_expense_ratio_pct": FieldSpec(
        name="net_expense_ratio_pct",
        description="Net annual expenses after fee waivers as percentage of net assets",
        expected_type="percentage",
        examples=["2.00%", "2.50%", "3.50%", "null"],
        aliases=["net expenses", "net annual expenses", "total after waivers"],
        extraction_hints=[
            "Look in 'Annual Fund Operating Expenses' table",
            "This is total expenses AFTER subtracting fee waivers/reimbursements",
            "If no waivers, may equal total_expense_ratio_pct",
        ],
    ),
    "fee_waiver_pct": FieldSpec(
        name="fee_waiver_pct",
        description="Fee waiver/reimbursement as percentage of net assets",
        expected_type="percentage",
        examples=["0.25%", "0.50%", "null"],
        aliases=["fee waiver", "expense reimbursement", "fee reimbursement", "waiver"],
        extraction_hints=[
            "Look in 'Annual Fund Operating Expenses' table",
            "Often shown as a negative number or in parentheses",
            "Extract the ABSOLUTE value (positive number)",
            "Return null if no waiver exists",
        ],
    ),
    "incentive_fee_xbrl_pct": FieldSpec(
        name="incentive_fee_xbrl_pct",
        description="Incentive/performance fee as percentage of net assets from the fee table",
        expected_type="percentage",
        examples=["0.50%", "1.00%", "null"],
        aliases=["incentive fee expense", "performance fee expense"],
        extraction_hints=[
            "Look in 'Annual Fund Operating Expenses' table",
            "This is the incentive fee expressed as % of net assets in the fee table",
            "DISTINCT from the contractual incentive fee rate (e.g., 10% of gains)",
            "Return null if not shown in fee table",
        ],
    ),

    # =========================================================================
    # INCENTIVE FEE FIELDS
    # =========================================================================
    "incentive_fee_pct": FieldSpec(
        name="incentive_fee_pct",
        description="The performance-based incentive fee percentage charged by the fund's own adviser (NOT underlying fund fees)",
        expected_type="percentage",
        examples=["10%", "12.5%", "15%", "20%"],
        aliases=["performance fee", "carried interest", "incentive allocation", "incentive fee"],
        extraction_hints=[
            "SEARCH THESE SECTIONS FIRST: 'Shareholder Transaction Expenses Table', 'Annual Expenses Table', 'Fee Table Note'",
            "The fee % is often in footnotes WITHIN Fee Table sections, not in dedicated 'Incentive Fee' sections",
            "Look for patterns like 'The Fund will pay the Adviser...X%' or 'we will pay X%'",
            "CRITICAL: Distinguish FUND-LEVEL fees from AFFE (Acquired Fund Fees and Expenses)",
            "FUND-LEVEL (extract this): 'the Fund pays the Adviser X%' - uses singular 'Fund'",
            "AFFE (DO NOT extract): 'Investment Funds charge X%' - uses plural 'Funds' or 'underlying'",
            "Look for patterns like 'X% of Pre-Incentive Fee Net Investment Income'",
            "Fund-of-funds often have NO fund-level incentive fee but show AFFE - return null in this case",
        ],
    ),
    "has_incentive_fee": FieldSpec(
        name="has_incentive_fee",
        description="Whether the fund ITSELF charges an incentive/performance fee (NOT underlying fund fees)",
        expected_type="boolean",
        examples=["true", "false"],
        aliases=["incentive fee exists", "performance fee charged"],
        extraction_hints=[
            "SEARCH THESE SECTIONS FIRST: 'Shareholder Transaction Expenses Table', 'Annual Expenses Table', 'Fee Table Note'",
            "CRITICAL: We want FUND-LEVEL fees only, NOT AFFE (underlying fund fees)",
            "TRUE only if: 'the Fund will pay the Adviser' or 'we will pay X%' - singular 'Fund'",
            "FALSE if: only AFFE mentioned ('Investment Funds charge', 'underlying funds typically charge') - plural",
            "Fund-of-funds often show AFFE in fee table but have NO fund-level incentive fee → FALSE",
            "If no fund-level fee text found after thorough search → extract FALSE, not stop_not_found",
            "Only TRUE if the fund's own adviser receives an incentive fee from the fund",
        ],
    ),
    "hurdle_rate_pct": FieldSpec(
        name="hurdle_rate_pct",
        description="The ANNUALIZED hurdle rate that must be exceeded before incentive fees apply",
        expected_type="percentage",
        examples=["5%", "6%", "7%", "8%", "null"],
        aliases=["hurdle rate", "preferred return", "hurdle", "threshold return", "minimum return"],
        extraction_hints=[
            "SEARCH: 'hurdle rate', 'preferred return', 'threshold', 'minimum return', 'X% per quarter'",
            "",
            "ANNUALIZATION (CRITICAL):",
            "- If stated as QUARTERLY rate: multiply by 4",
            "  Example: '1.25% per quarter' → 5% annualized",
            "  Example: '1.50% per quarter' → 6% annualized",
            "- If stated as MONTHLY rate: multiply by 12",
            "- If stated as ANNUAL rate: use as-is",
            "",
            "WHEN TO RETURN NULL:",
            "1. Fund has NO incentive fee (has_incentive_fee=False) → NULL",
            "2. Fund uses Loss Recovery Account but NO hurdle → NULL (different mechanism)",
            "3. No hurdle language found after thorough search → NULL",
            "",
            "DISTINCTION FROM CATCH-UP CEILING:",
            "- Hurdle = minimum return before ANY incentive fee (e.g., 6%)",
            "- Catch-up ceiling = rate where manager catches up to full % (e.g., 1.667%)",
            "These are different values - extract the HURDLE, not the ceiling.",
            "",
            "Funds with Loss Recovery Account often have NO hurdle rate.",
        ],
    ),
    "high_water_mark": FieldSpec(
        name="high_water_mark",
        description="Whether the incentive fee includes a high water mark provision that prevents paying fees on recovered losses",
        expected_type="boolean",
        examples=["true", "false", "null"],
        aliases=["high water mark", "HWM", "loss carryforward", "loss recovery account", "cumulative loss"],
        extraction_hints=[
            "SEARCH: 'high water mark', 'loss recovery account', 'loss carryforward', 'cumulative loss'",
            "",
            "DECISION TREE:",
            "1. Found 'high water mark' or 'HWM' explicitly → TRUE",
            "2. Found 'Loss Recovery Account' or 'Loss Carryforward Account' → TRUE",
            "   CRITICAL: These ARE high water mark mechanisms - they track cumulative losses",
            "   and prevent paying incentive fees on gains that merely recover prior losses",
            "3. Found 'cumulative loss' provision in fee context → TRUE",
            "4. No incentive fee at all (has_incentive_fee=False) → NULL",
            "5. Incentive fee exists but no HWM/loss recovery language found → NULL (not FALSE)",
            "",
            "COMMON PATTERN: Fund-of-funds (Hamilton Lane, Carlyle) use 'Loss Recovery Account'",
            "which functions identically to a high water mark. Extract as TRUE.",
        ],
    ),
    "has_catch_up": FieldSpec(
        name="has_catch_up",
        description="Whether the incentive fee includes a catch-up provision allowing accelerated fee payment above hurdle",
        expected_type="boolean",
        examples=["true", "false", "null"],
        aliases=["catch-up", "catch up provision", "full catch-up"],
        extraction_hints=[
            "SEARCH: 'catch-up', 'catch up', '100% of returns between', 'ceiling'",
            "",
            "DECISION TREE:",
            "1. Found 'catch-up' or 'catch up' in fee description → TRUE",
            "2. Found 'full catch-up' → TRUE",
            "3. Found '100% of returns between X% and Y%' pattern → TRUE",
            "4. Found ceiling percentage above hurdle where manager gets 100% → TRUE",
            "5. Found Loss Recovery Account WITHOUT catch-up language → NULL (Loss Recovery is NOT catch-up)",
            "6. No incentive fee at all (has_incentive_fee=False) → NULL",
            "7. Incentive fee exists but no catch-up language found → NULL (not FALSE)",
            "",
            "CRITICAL DISTINCTION:",
            "- CATCH-UP = accelerated fee payment between hurdle and ceiling",
            "- LOSS RECOVERY ACCOUNT = high water mark mechanism (NOT catch-up)",
            "These are DIFFERENT concepts. Do not confuse them.",
        ],
    ),

    # =========================================================================
    # EXPENSE CAP FIELDS
    # =========================================================================
    "has_expense_cap": FieldSpec(
        name="has_expense_cap",
        description="Whether the fund has an expense cap/limitation agreement",
        expected_type="boolean",
        examples=["true", "false"],
        aliases=["expense cap", "expense limitation", "fee waiver"],
        extraction_hints=[
            "Look in Fee Table footnotes for expense limitation language",
            "May be called 'expense cap', 'fee waiver', or 'expense limitation agreement'",
        ],
    ),
    "expense_cap_pct": FieldSpec(
        name="expense_cap_pct",
        description="The expense cap percentage as a percentage of net assets (fund-level, not per-class)",
        expected_type="percentage",
        examples=["0.50%", "1.00%", "1.50%", "2.00%", "3.00%", "null"],
        aliases=["expense cap rate", "expense limitation rate", "total expense cap", "expense limitation"],
        extraction_hints=[
            "SEARCH: 'expense cap', 'expense limitation', 'Total Expense Cap', 'Total Operating Expenses'",
            "SEARCH SECTIONS: 'Fee Table Note', 'Expense Limitation Agreement', 'Advisory Agreement'",
            "",
            "PATTERNS TO LOOK FOR:",
            "- 'Total Expense Cap means the annual rate of X%'",
            "- 'limit Total Operating Expenses to X%'",
            "- 'Specified Expenses...do not exceed X% of average daily net assets'",
            "",
            "WHEN TO RETURN NULL:",
            "1. Expense cap varies by share class with NO fund-level cap",
            "   Example: 'Class R: 1.45%, Class I: 0.75%, Class D: 1.00%' → NULL (no single fund-level cap)",
            "2. Expense cap is not explicitly stated as a percentage",
            "3. Document only mentions 'the Adviser may waive fees' without specific cap %",
            "",
            "IF ONLY CLASS-LEVEL CAPS EXIST:",
            "- Some funds have different caps per share class",
            "- If no single fund-level cap exists, return NULL",
            "- Do NOT return one of the class-level caps as the fund cap",
        ],
    ),

    # =========================================================================
    # REPURCHASE/LIQUIDITY FIELDS
    # =========================================================================
    "repurchase_frequency": FieldSpec(
        name="repurchase_frequency",
        description="How often the fund offers to repurchase shares",
        expected_type="text",
        examples=["quarterly", "monthly", "semi-annually", "annually"],
        aliases=["redemption frequency", "tender offer frequency", "liquidity frequency"],
        extraction_hints=[
            "Search for sections containing: 'Interval', 'Repurchase', 'Tender', 'Liquidity'",
            "Interval funds (Rule 23c-3) MUST offer quarterly repurchases - look for this rule reference",
            "Look for patterns: 'repurchase...quarterly basis', 'offer to repurchase...quarterly'",
            "Search keywords: 'repurchase offer', 'tender offer', 'quarterly', 'Rule 23c-3', 'interval fund'",
            "May be in 'Fund Overview', 'Summary', or dedicated 'Repurchase Offers' section",
            "The frequency is how OFTEN offers are made (quarterly), not the percentage amount",
        ],
    ),
    "repurchase_amount_pct": FieldSpec(
        name="repurchase_amount_pct",
        description="The percentage of shares offered for repurchase each period",
        expected_type="percentage",
        examples=["5%", "10%", "25%"],
        aliases=["repurchase percentage", "tender amount", "redemption amount"],
        extraction_hints=[
            "Interval funds must offer at least 5% quarterly",
            "Look for 'offer to repurchase X% of shares'",
            "May specify minimum (5%) and maximum (25%) range",
        ],
    ),
    "lock_up_period_years": FieldSpec(
        name="lock_up_period_years",
        description="The lock-up period in years before shares can be repurchased",
        expected_type="number",
        examples=["1", "2", "0"],
        aliases=["lock-up period", "holding period", "lock up"],
        extraction_hints=[
            "Look for 'one-year anniversary' or 'X months holding period'",
            "Early repurchase fee often applies during lock-up",
        ],
    ),
    "early_repurchase_fee_pct": FieldSpec(
        name="early_repurchase_fee_pct",
        description="The fee charged for repurchasing shares before the lock-up period ends",
        expected_type="percentage",
        examples=["2%", "2.00%"],
        aliases=["early redemption fee", "early repurchase fee", "CDSC"],
        extraction_hints=[
            "Often 2% for repurchases within first year",
            "Look for 'early repurchase fee' or 'contingent deferred sales charge'",
        ],
    ),

    # =========================================================================
    # LEVERAGE FIELDS
    # =========================================================================
    "uses_leverage": FieldSpec(
        name="uses_leverage",
        description="Whether the fund uses leverage/borrowing",
        expected_type="boolean",
        examples=["true", "false"],
        aliases=["leverage", "borrowing", "uses debt"],
        extraction_hints=[
            "Look for 'Use of Leverage' section",
            "May mention 'borrowing', 'credit facility', or 'leverage'",
        ],
    ),
    "max_leverage_pct": FieldSpec(
        name="max_leverage_pct",
        description="The maximum leverage the fund may use, NORMALIZED to percentage of total assets that can be borrowed",
        expected_type="percentage",
        examples=["33%", "33.33%", "50%"],
        aliases=["leverage limit", "borrowing limit", "maximum leverage", "asset coverage"],
        extraction_hints=[
            "SEARCH: 'leverage', 'borrowing', 'asset coverage', 'debt-to-equity', '1940 Act', 'credit facility'",
            "",
            "FORMAT RECOGNITION (CRITICAL - documents use different formats):",
            "",
            "1. ASSET COVERAGE RATIO (most common for registered funds):",
            "   '300% asset coverage' → NORMALIZE to 33.33%",
            "   '200% asset coverage' → NORMALIZE to 50%",
            "   Formula: leverage_pct = 100 / (coverage_pct / 100)",
            "   Meaning: 300% coverage = for every $1 borrowed, must have $3 in assets = can borrow 1/3",
            "",
            "2. DEBT-TO-EQUITY RATIO:",
            "   '50% debt-to-equity' → NORMALIZE to 33.33%",
            "   Formula: leverage_pct = (D/E) / (1 + D/E) * 100",
            "   Meaning: 50% D/E = debt is half of equity = 1/3 of total assets",
            "",
            "3. PERCENTAGE OF ASSETS (direct - no conversion needed):",
            "   'may borrow up to 33% of total assets' → 33%",
            "",
            "4. 1940 ACT REFERENCE:",
            "   'in accordance with the 1940 Act' → 33.33% (statutory limit)",
            "",
            "CRITICAL: Return the NORMALIZED 'percentage of assets' (33%, 50%),",
            "NOT the raw asset coverage ratio (300%, 200%).",
            "Most registered funds are limited to 33.33% (300% coverage under 1940 Act).",
        ],
    ),

    # =========================================================================
    # DISTRIBUTION FIELDS
    # =========================================================================
    "distribution_frequency": FieldSpec(
        name="distribution_frequency",
        description="How often the fund makes distributions (dividends) to shareholders",
        expected_type="text",
        examples=["monthly", "quarterly", "semi-annually", "annually", "null"],
        aliases=["dividend frequency", "income distribution frequency", "distribution policy"],
        extraction_hints=[
            "SEARCH: 'distribution', 'dividend', 'income distribution', 'declare', 'pay'",
            "SEARCH SECTIONS: 'Distributions', 'Dividend Policy', 'Dividend Reinvestment', 'Tax Information'",
            "",
            "PATTERNS TO LOOK FOR:",
            "- 'declare distributions monthly' → monthly",
            "- 'quarterly distributions' → quarterly",
            "- 'distributions will be made at least annually' → annually",
            "- 'intends to pay dividends quarterly' → quarterly",
            "",
            "IMPORTANT DISTINCTIONS:",
            "- DISTRIBUTION frequency (what we want): how often dividends are paid",
            "- REPURCHASE frequency (NOT this): how often shares can be sold back",
            "- DRIP/reinvestment (NOT this): whether dividends are reinvested",
            "",
            "WHEN TO RETURN NULL:",
            "- Many PE/private equity funds do NOT make regular distributions",
            "- If document says 'distributions at discretion of board' with no stated frequency → NULL",
            "- If only 'at least annually for tax purposes' is mentioned with no regular schedule → NULL",
            "- If you only find repurchase frequency but no distribution frequency → NULL",
            "",
            "Credit funds often distribute monthly; PE funds often distribute annually or not at all.",
        ],
    ),

    # =========================================================================
    # FUND TYPE FIELDS
    # =========================================================================
    "fund_type": FieldSpec(
        name="fund_type",
        description="The type of fund structure (interval fund, tender offer fund, etc.)",
        expected_type="text",
        examples=["interval_fund", "tender_offer_fund", "closed_end_fund"],
        aliases=["fund structure", "fund category"],
        extraction_hints=[
            "Look for 'interval fund', 'tender offer fund', 'closed-end fund'",
            "Interval funds must offer quarterly repurchases per Rule 23c-3",
            "Usually stated clearly in fund overview or summary",
        ],
    ),
}
