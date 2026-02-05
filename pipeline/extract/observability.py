"""
Observability infrastructure for LLM extraction pipeline.

Provides full decision tracing for debugging, auditing, and continuous improvement.
Every extraction can be explained via explain_decision().

Usage:
    obs = ObservableExtraction(field_name="management_fee_pct")
    obs.add_decision(LayerDecision(
        layer_name="tier0_xbrl",
        field_name="management_fee_pct",
        decision=DecisionType.EXTRACT,
        input_value=None,
        output_value=1.25,
        confidence_out=1.0,
        evidence="Parsed from XBRL tag: cef:ManagementFeeOverAssets",
    ))
    print(obs.explain_decision())
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DecisionType(str, Enum):
    """Type of decision made by an architecture layer."""
    EXTRACT = "extract"     # Value extracted from source
    PASS = "pass"           # Value accepted, forwarded unchanged
    REJECT = "reject"       # Value rejected (hallucination, ungrounded)
    MODIFY = "modify"       # Value changed (e.g., confidence downgraded)
    FLAG = "flag"           # Value flagged for human review
    SKIP = "skip"           # Layer skipped (not applicable)
    RETRY = "retry"         # Triggered retry with different strategy
    FALLBACK = "fallback"   # Fell back to alternative extraction method


class MatchType(str, Enum):
    """Type of grounding match found."""
    EXACT = "exact"         # Exact string match
    FUZZY = "fuzzy"         # Fuzzy/partial match
    SEMANTIC = "semantic"   # Semantic/concept match
    NONE = "none"           # No match found


@dataclass
class LayerDecision:
    """
    Single decision from one architecture layer.

    Every layer in the extraction pipeline logs a LayerDecision
    to create a full audit trail.
    """
    layer_name: str              # "tier0_xbrl", "tier1_section", "grounding", etc.
    field_name: str              # "management_fee_pct", "repurchase_frequency"
    decision: DecisionType
    input_value: Any = None      # What came in (None if first extraction)
    output_value: Any = None     # What goes out
    confidence_in: Optional[float] = None
    confidence_out: Optional[float] = None
    evidence: str = ""           # Why this decision was made
    source_location: Optional[str] = None  # Chunk ID, section title, line number
    metadata: dict = field(default_factory=dict)  # Layer-specific data
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "layer_name": self.layer_name,
            "field_name": self.field_name,
            "decision": self.decision.value,
            "input_value": _serialize_value(self.input_value),
            "output_value": _serialize_value(self.output_value),
            "confidence_in": self.confidence_in,
            "confidence_out": self.confidence_out,
            "evidence": self.evidence,
            "source_location": self.source_location,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ObservableExtraction:
    """
    Complete extraction result with full decision trace.

    Captures the journey of a single field through all extraction layers,
    enabling full auditability and debugging.
    """
    field_name: str
    final_value: Any = None
    final_confidence: str = "not_found"  # "explicit", "inferred", "not_found"
    decisions: list[LayerDecision] = field(default_factory=list)
    flagged_for_review: bool = False
    flag_reason: Optional[str] = None
    extraction_source: Optional[str] = None  # Which layer produced the value

    def add_decision(self, decision: LayerDecision):
        """
        Append a decision to the trace and update final state.

        Args:
            decision: The LayerDecision to add
        """
        self.decisions.append(decision)

        # Update final value based on decision type
        if decision.decision in (DecisionType.EXTRACT, DecisionType.MODIFY, DecisionType.PASS):
            if decision.output_value is not None:
                self.final_value = decision.output_value
                self.extraction_source = decision.layer_name

        # Update confidence
        if decision.confidence_out is not None:
            if isinstance(decision.confidence_out, float):
                if decision.confidence_out >= 0.9:
                    self.final_confidence = "explicit"
                elif decision.confidence_out >= 0.6:
                    self.final_confidence = "inferred"
                else:
                    self.final_confidence = "not_found"
            else:
                self.final_confidence = str(decision.confidence_out)

        # Check for flagging
        if decision.decision == DecisionType.FLAG:
            self.flagged_for_review = True
            if decision.evidence:
                self.flag_reason = decision.evidence

    def explain_decision(self) -> str:
        """
        Generate human-readable explanation of extraction path.

        Returns:
            Multi-line string explaining how the value was determined
        """
        lines = [f"{'=' * 50}"]
        lines.append(f"Field: {self.field_name}")
        lines.append(f"{'=' * 50}")
        lines.append(f"Final value: {self.final_value}")
        lines.append(f"Confidence: {self.final_confidence}")
        lines.append(f"Source layer: {self.extraction_source or 'N/A'}")
        lines.append(f"Flagged for review: {self.flagged_for_review}")
        if self.flag_reason:
            lines.append(f"Flag reason: {self.flag_reason}")
        lines.append("")
        lines.append("Decision trace:")
        lines.append("-" * 50)

        for i, d in enumerate(self.decisions, 1):
            lines.append(f"  [{i}] {d.layer_name} -> {d.decision.value}")
            lines.append(f"      Input:  {_format_value(d.input_value)}")
            lines.append(f"      Output: {_format_value(d.output_value)}")
            if d.confidence_in is not None or d.confidence_out is not None:
                conf_str = f"      Confidence: {d.confidence_in} -> {d.confidence_out}"
                lines.append(conf_str)
            if d.evidence:
                # Wrap long evidence text
                evidence_lines = _wrap_text(d.evidence, 60)
                lines.append(f"      Why: {evidence_lines[0]}")
                for el in evidence_lines[1:]:
                    lines.append(f"           {el}")
            if d.source_location:
                lines.append(f"      Location: {d.source_location}")
            if d.metadata:
                for key, val in d.metadata.items():
                    lines.append(f"      {key}: {val}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Export for JSON serialization."""
        return {
            "field_name": self.field_name,
            "final_value": _serialize_value(self.final_value),
            "final_confidence": self.final_confidence,
            "flagged_for_review": self.flagged_for_review,
            "flag_reason": self.flag_reason,
            "extraction_source": self.extraction_source,
            "decision_count": len(self.decisions),
            "decisions": [d.to_dict() for d in self.decisions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ObservableExtraction":
        """Create from dictionary (for loading saved decisions)."""
        obs = cls(
            field_name=data["field_name"],
            final_value=data.get("final_value"),
            final_confidence=data.get("final_confidence", "not_found"),
            flagged_for_review=data.get("flagged_for_review", False),
            flag_reason=data.get("flag_reason"),
            extraction_source=data.get("extraction_source"),
        )
        # Note: decisions are not fully reconstructed (would need DecisionType parsing)
        return obs


@dataclass
class ExtractionTrace:
    """
    Complete extraction trace for a document.

    Contains ObservableExtraction for every field extracted.
    """
    filing_id: str
    fund_name: str
    extractions: dict[str, ObservableExtraction] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def add_extraction(self, extraction: ObservableExtraction):
        """Add an extraction to the trace."""
        self.extractions[extraction.field_name] = extraction

    def get_extraction(self, field_name: str) -> Optional[ObservableExtraction]:
        """Get extraction for a specific field."""
        return self.extractions.get(field_name)

    def mark_complete(self):
        """Mark extraction as complete."""
        self.completed_at = datetime.now()

    def summarize(self) -> dict:
        """
        Generate high-level summary for logging.

        Returns:
            Dictionary with summary statistics
        """
        total = len(self.extractions)
        flagged = sum(1 for e in self.extractions.values() if e.flagged_for_review)

        by_confidence = {"explicit": 0, "inferred": 0, "not_found": 0}
        by_source_layer = {}

        for obs in self.extractions.values():
            # Count by confidence
            conf = obs.final_confidence
            if conf in by_confidence:
                by_confidence[conf] += 1

            # Count by source layer
            if obs.extraction_source:
                layer = obs.extraction_source
                by_source_layer[layer] = by_source_layer.get(layer, 0) + 1

        return {
            "filing_id": self.filing_id,
            "fund_name": self.fund_name,
            "total_fields": total,
            "flagged_for_review": flagged,
            "flagged_pct": f"{flagged/total*100:.1f}%" if total > 0 else "0%",
            "by_confidence": by_confidence,
            "by_source_layer": by_source_layer,
            "duration_seconds": (
                (self.completed_at - self.started_at).total_seconds()
                if self.completed_at else None
            ),
        }

    def to_dict(self) -> dict:
        """Export full trace to dictionary."""
        return {
            "filing_id": self.filing_id,
            "fund_name": self.fund_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "summary": self.summarize(),
            "extractions": {
                name: obs.to_dict()
                for name, obs in self.extractions.items()
            },
        }

    def save(self, output_dir: str | Path):
        """
        Save trace to JSON file.

        Args:
            output_dir: Directory to save to (creates {filing_id}_decisions.json)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{self.filing_id}_decisions.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        logger.info(f"Saved decision trace to {output_path}")
        return output_path


# =============================================================================
# Utility Functions
# =============================================================================

def _serialize_value(value: Any) -> Any:
    """Serialize a value for JSON output."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    # For Decimal, datetime, etc.
    return str(value)


def _format_value(value: Any, max_length: int = 80) -> str:
    """Format a value for display."""
    if value is None:
        return "None"
    s = str(value)
    if len(s) > max_length:
        return s[:max_length - 3] + "..."
    return s


def _wrap_text(text: str, width: int) -> list[str]:
    """Wrap text to specified width."""
    if len(text) <= width:
        return [text]

    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(" ".join(current_line))

    return lines


# =============================================================================
# Helper Functions for Logging Decisions
# =============================================================================

def log_tier0_decision(
    obs: ObservableExtraction,
    found: bool,
    value: Any = None,
    tag_name: str = "",
    context_ref: str = "",
    share_class: str = None,
) -> ObservableExtraction:
    """
    Log a Tier 0 (XBRL) decision.

    Args:
        obs: The ObservableExtraction to update
        found: Whether a value was found
        value: The extracted value
        tag_name: XBRL tag name
        context_ref: XBRL contextRef attribute
        share_class: Parsed share class (if any)

    Returns:
        Updated ObservableExtraction
    """
    if found:
        obs.add_decision(LayerDecision(
            layer_name="tier0_xbrl",
            field_name=obs.field_name,
            decision=DecisionType.EXTRACT,
            input_value=None,
            output_value=value,
            confidence_out=1.0,
            evidence=f"Parsed from XBRL tag: {tag_name}",
            source_location=f"contextRef: {context_ref}" if context_ref else None,
            metadata={
                "tag_name": tag_name,
                "share_class": share_class,
            } if share_class else {"tag_name": tag_name},
        ))
    else:
        obs.add_decision(LayerDecision(
            layer_name="tier0_xbrl",
            field_name=obs.field_name,
            decision=DecisionType.SKIP,
            input_value=None,
            output_value=None,
            evidence="No XBRL tag found for this field",
        ))

    return obs


def log_tier1_decision(
    obs: ObservableExtraction,
    found: bool,
    value: Any = None,
    section_title: str = "",
    chunk_id: str = "",
    confidence: float = None,
    evidence_quote: str = "",
) -> ObservableExtraction:
    """
    Log a Tier 1 (section mapping + LLM) decision.

    Args:
        obs: The ObservableExtraction to update
        found: Whether a value was found
        value: The extracted value
        section_title: Title of the section searched
        chunk_id: ID of the chunk processed
        confidence: Extraction confidence (0-1)
        evidence_quote: Supporting quote from text
    """
    if found:
        obs.add_decision(LayerDecision(
            layer_name="tier1_section",
            field_name=obs.field_name,
            decision=DecisionType.EXTRACT,
            input_value=obs.final_value,
            output_value=value,
            confidence_out=confidence,
            evidence=f"Extracted from section: {section_title}" + (
                f"\nQuote: \"{evidence_quote[:100]}...\"" if evidence_quote else ""
            ),
            source_location=f"chunk: {chunk_id}" if chunk_id else f"section: {section_title}",
            metadata={"section_title": section_title},
        ))
    else:
        obs.add_decision(LayerDecision(
            layer_name="tier1_section",
            field_name=obs.field_name,
            decision=DecisionType.SKIP,
            input_value=obs.final_value,
            output_value=None,
            evidence=f"Field not found in section: {section_title}",
            source_location=f"section: {section_title}",
        ))

    return obs


def log_tier2_decision(
    obs: ObservableExtraction,
    found: bool,
    value: Any = None,
    pattern_used: str = "",
    chars_extracted: int = 0,
    confidence: float = None,
) -> ObservableExtraction:
    """Log a Tier 2 (regex fallback) decision."""
    if found:
        obs.add_decision(LayerDecision(
            layer_name="tier2_regex",
            field_name=obs.field_name,
            decision=DecisionType.EXTRACT,
            input_value=obs.final_value,
            output_value=value,
            confidence_out=confidence,
            evidence=f"Found via regex pattern fallback",
            metadata={
                "pattern": pattern_used,
                "chars_extracted": chars_extracted,
            },
        ))
    else:
        obs.add_decision(LayerDecision(
            layer_name="tier2_regex",
            field_name=obs.field_name,
            decision=DecisionType.SKIP,
            input_value=obs.final_value,
            output_value=None,
            evidence="No matching sections found via regex patterns",
        ))

    return obs


def log_tier3_decision(
    obs: ObservableExtraction,
    found: bool,
    value: Any = None,
    sections_searched: int = 0,
    top_section: str = "",
    keyword_score: int = 0,
    chunks_processed: int = 0,
    confidence: float = None,
) -> ObservableExtraction:
    """Log a Tier 3 (scoped agentic) decision."""
    if found:
        obs.add_decision(LayerDecision(
            layer_name="tier3_agentic",
            field_name=obs.field_name,
            decision=DecisionType.EXTRACT,
            input_value=obs.final_value,
            output_value=value,
            confidence_out=confidence,
            evidence=f"Found via scoped agentic search in '{top_section}'",
            source_location=f"section: {top_section}",
            metadata={
                "sections_searched": sections_searched,
                "keyword_score": keyword_score,
                "chunks_processed": chunks_processed,
            },
        ))
    else:
        obs.add_decision(LayerDecision(
            layer_name="tier3_agentic",
            field_name=obs.field_name,
            decision=DecisionType.SKIP,
            input_value=obs.final_value,
            output_value=None,
            evidence=f"Searched {sections_searched} sections, no value found",
            metadata={
                "sections_searched": sections_searched,
                "chunks_processed": chunks_processed,
            },
        ))

    return obs


def log_grounding_decision(
    obs: ObservableExtraction,
    is_grounded: bool,
    match_type: MatchType = MatchType.NONE,
    match_score: float = 0.0,
    matched_text: str = "",
    issues: list[str] = None,
) -> ObservableExtraction:
    """
    Log a grounding validation decision.

    Args:
        obs: The ObservableExtraction to update
        is_grounded: Whether the value is grounded in source
        match_type: Type of match (exact, fuzzy, semantic)
        match_score: Match score (0-1)
        matched_text: The text that matched
        issues: List of grounding issues found
    """
    issues = issues or []

    if is_grounded:
        obs.add_decision(LayerDecision(
            layer_name="grounding",
            field_name=obs.field_name,
            decision=DecisionType.PASS,
            input_value=obs.final_value,
            output_value=obs.final_value,
            confidence_in=obs.decisions[-1].confidence_out if obs.decisions else None,
            confidence_out=obs.decisions[-1].confidence_out if obs.decisions else None,
            evidence=f"{match_type.value} match found in source",
            metadata={
                "match_type": match_type.value,
                "match_score": match_score,
                "matched_text": matched_text[:100] if matched_text else "",
            },
        ))
    else:
        # Downgrade confidence for ungrounded values
        new_confidence = 0.5 if obs.decisions else 0.3
        obs.add_decision(LayerDecision(
            layer_name="grounding",
            field_name=obs.field_name,
            decision=DecisionType.MODIFY,
            input_value=obs.final_value,
            output_value=obs.final_value,
            confidence_out=new_confidence,
            evidence=f"Value not grounded in source: {', '.join(issues)}",
            metadata={
                "match_type": match_type.value,
                "match_score": match_score,
                "issues": issues,
            },
        ))
        obs.final_confidence = "inferred"

    return obs


def log_flag_decision(
    obs: ObservableExtraction,
    reason: str,
    details: dict = None,
) -> ObservableExtraction:
    """
    Log a decision to flag for human review.

    Args:
        obs: The ObservableExtraction to update
        reason: Why the field is being flagged
        details: Additional details about the flagging
    """
    obs.add_decision(LayerDecision(
        layer_name="confidence_cal",
        field_name=obs.field_name,
        decision=DecisionType.FLAG,
        input_value=obs.final_value,
        output_value=obs.final_value,
        evidence=reason,
        metadata=details or {},
    ))
    obs.flagged_for_review = True
    obs.flag_reason = reason

    return obs


# =============================================================================
# Debugging Utilities
# =============================================================================

def explain_extraction(trace: ExtractionTrace, field_name: str) -> str:
    """
    Get human-readable explanation for a specific field.

    Args:
        trace: The ExtractionTrace containing all extractions
        field_name: Name of the field to explain

    Returns:
        Human-readable explanation string
    """
    extraction = trace.get_extraction(field_name)
    if extraction:
        return extraction.explain_decision()
    return f"Field '{field_name}' not found in extraction trace"


def dump_decisions(trace: ExtractionTrace, output_path: str | Path):
    """
    Export all decisions to JSON file for analysis.

    Args:
        trace: The ExtractionTrace to export
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trace.to_dict(), f, indent=2, default=str)

    logger.info(f"Dumped decisions to {output_path}")


def summarize_extraction(trace: ExtractionTrace) -> dict:
    """
    Generate high-level summary for logging.

    Args:
        trace: The ExtractionTrace to summarize

    Returns:
        Dictionary with summary statistics
    """
    return trace.summarize()


def print_flagged_fields(trace: ExtractionTrace):
    """Print all fields flagged for review."""
    flagged = [
        (name, obs)
        for name, obs in trace.extractions.items()
        if obs.flagged_for_review
    ]

    if not flagged:
        print("No fields flagged for review.")
        return

    print(f"\n{'=' * 60}")
    print(f"FLAGGED FOR REVIEW: {len(flagged)} fields")
    print(f"{'=' * 60}\n")

    for name, obs in flagged:
        print(f"Field: {name}")
        print(f"  Value: {obs.final_value}")
        print(f"  Reason: {obs.flag_reason}")
        print()


def load_trace(filepath: str | Path) -> dict:
    """
    Load a saved decision trace from JSON.

    Args:
        filepath: Path to the JSON file

    Returns:
        Dictionary containing the trace data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
