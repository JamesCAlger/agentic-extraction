"""
Confidence scoring for extraction results.

Computes a 0.0-1.0 confidence score based on multiple signals:
- LLM self-reported confidence
- Grounding validation (value found in source text)
- Evidence quality (quote length, value in quote)
- Retrieval quality (reranker scores, chunk coverage)

This module is designed to be toggleable via config for easy rollback.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List
from enum import Enum

logger = logging.getLogger(__name__)


class ConfidenceLevel(str, Enum):
    """Confidence levels from LLM extraction."""
    EXPLICIT = "explicit"
    INFERRED = "inferred"
    NOT_FOUND = "not_found"


@dataclass
class ConfidenceWeights:
    """
    Configurable weights for confidence score components.

    All weights should sum to 1.0 for normalized scoring.
    Can be adjusted per field type or globally.
    """
    llm_confidence: float = 0.25
    grounding: float = 0.35
    evidence_quality: float = 0.20
    retrieval_quality: float = 0.20

    def __post_init__(self):
        total = self.llm_confidence + self.grounding + self.evidence_quality + self.retrieval_quality
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Confidence weights sum to {total}, normalizing to 1.0")
            self.llm_confidence /= total
            self.grounding /= total
            self.evidence_quality /= total
            self.retrieval_quality /= total


@dataclass
class ConfidenceConfig:
    """Configuration for confidence scoring."""
    enabled: bool = False

    # CRITICAL: If True, confidence is set to 0.0 when grounding fails
    # This is the safest mode - unverified values get zero confidence
    require_grounding: bool = True

    # Default weights
    default_weights: ConfidenceWeights = field(default_factory=ConfidenceWeights)

    # Field-type specific weight overrides (pattern -> weights)
    field_type_weights: Dict[str, ConfidenceWeights] = field(default_factory=dict)

    # Thresholds
    high_confidence_threshold: float = 0.7
    low_confidence_threshold: float = 0.4

    # Evidence quality settings
    min_quote_length_short: int = 20
    min_quote_length_medium: int = 50
    min_quote_length_long: int = 100

    # Reranker score thresholds
    reranker_high_score: float = 0.8
    reranker_medium_score: float = 0.5
    reranker_low_score: float = 0.3


@dataclass
class ConfidenceResult:
    """Result of confidence scoring for a single extraction."""
    field_name: str
    value: Any
    confidence_score: float  # 0.0 to 1.0
    confidence_level: str  # "high", "medium", "low"

    # Component scores (for debugging/analysis)
    llm_score: float = 0.0
    grounding_score: float = 0.0
    evidence_score: float = 0.0
    retrieval_score: float = 0.0

    # Metadata
    signals_used: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "field_name": self.field_name,
            "value": self.value,
            "confidence_score": round(self.confidence_score, 3),
            "confidence_level": self.confidence_level,
            "components": {
                "llm": round(self.llm_score, 3),
                "grounding": round(self.grounding_score, 3),
                "evidence": round(self.evidence_score, 3),
                "retrieval": round(self.retrieval_score, 3),
            },
            "signals_used": self.signals_used,
        }


class ConfidenceScorer:
    """
    Computes confidence scores for extraction results.

    Usage:
        scorer = ConfidenceScorer(config)
        result = scorer.score_extraction(
            field_name="incentive_fee.hurdle_rate_pct",
            value=5.0,
            llm_confidence="explicit",
            evidence_quote="5% annual hurdle rate",
            is_grounded=True,
            grounding_score=1.0,
            reranker_scores=[0.89, 0.72, 0.65],
            chunks_searched=10,
        )
        print(f"Confidence: {result.confidence_score:.2f} ({result.confidence_level})")
    """

    def __init__(self, config: Optional[ConfidenceConfig] = None):
        self.config = config or ConfidenceConfig()

        # Default field type patterns
        self._init_default_field_weights()

    def _init_default_field_weights(self):
        """Initialize default field-type specific weights."""
        if not self.config.field_type_weights:
            self.config.field_type_weights = {
                # Boolean fields - LLM confidence matters more
                "has_*": ConfidenceWeights(
                    llm_confidence=0.35,
                    grounding=0.25,
                    evidence_quality=0.20,
                    retrieval_quality=0.20,
                ),
                # Percentage fields - Grounding matters more
                "*_pct": ConfidenceWeights(
                    llm_confidence=0.20,
                    grounding=0.45,
                    evidence_quality=0.20,
                    retrieval_quality=0.15,
                ),
                # Share class fields - Evidence quality matters more
                "share_classes.*": ConfidenceWeights(
                    llm_confidence=0.20,
                    grounding=0.30,
                    evidence_quality=0.30,
                    retrieval_quality=0.20,
                ),
            }

    def _get_weights_for_field(self, field_name: str) -> ConfidenceWeights:
        """Get the appropriate weights for a field based on its name/type."""
        # Check field-type patterns
        for pattern, weights in self.config.field_type_weights.items():
            if self._match_pattern(field_name, pattern):
                return weights

        return self.config.default_weights

    def _match_pattern(self, field_name: str, pattern: str) -> bool:
        """Simple pattern matching for field names."""
        if pattern.startswith("*") and pattern.endswith("*"):
            return pattern[1:-1] in field_name
        elif pattern.startswith("*"):
            return field_name.endswith(pattern[1:])
        elif pattern.endswith("*"):
            return field_name.startswith(pattern[:-1])
        else:
            return field_name == pattern

    def score_extraction(
        self,
        field_name: str,
        value: Any,
        llm_confidence: Optional[str] = None,
        evidence_quote: Optional[str] = None,
        is_grounded: Optional[bool] = None,
        grounding_score: Optional[float] = None,
        reranker_scores: Optional[List[float]] = None,
        chunks_searched: int = 0,
        sections_searched: int = 0,
    ) -> ConfidenceResult:
        """
        Compute confidence score for an extraction result.

        Args:
            field_name: Name of the field being extracted
            value: The extracted value
            llm_confidence: LLM's self-reported confidence ("explicit", "inferred", "not_found")
            evidence_quote: Verbatim quote supporting the extraction
            is_grounded: Whether value was found in source text
            grounding_score: Grounding validation score (0.0-1.0)
            reranker_scores: Top reranker relevance scores
            chunks_searched: Number of chunks sent to LLM
            sections_searched: Number of sections considered

        Returns:
            ConfidenceResult with overall score and component breakdown
        """
        if not self.config.enabled:
            # Return neutral confidence when disabled
            return ConfidenceResult(
                field_name=field_name,
                value=value,
                confidence_score=0.5,
                confidence_level="medium",
                signals_used=["disabled"],
            )

        # CRITICAL: Strict grounding mode - zero confidence if not grounded
        if self.config.require_grounding:
            if is_grounded is False:
                return ConfidenceResult(
                    field_name=field_name,
                    value=value,
                    confidence_score=0.0,
                    confidence_level="low",
                    llm_score=0.0,
                    grounding_score=0.0,
                    evidence_score=0.0,
                    retrieval_score=0.0,
                    signals_used=["GROUNDING_FAILED:confidence_zeroed"],
                )
            elif is_grounded is None:
                # Grounding not performed - penalize but don't zero
                # This allows graceful degradation when grounding is unavailable
                return ConfidenceResult(
                    field_name=field_name,
                    value=value,
                    confidence_score=0.25,
                    confidence_level="low",
                    llm_score=0.0,
                    grounding_score=0.0,
                    evidence_score=0.0,
                    retrieval_score=0.0,
                    signals_used=["GROUNDING_MISSING:confidence_penalized"],
                )

        weights = self._get_weights_for_field(field_name)
        signals_used = []

        # 1. LLM Confidence Score
        llm_score = self._score_llm_confidence(llm_confidence)
        if llm_confidence:
            signals_used.append(f"llm:{llm_confidence}")

        # 2. Grounding Score
        grounding_component = self._score_grounding(is_grounded, grounding_score)
        if is_grounded is not None:
            signals_used.append(f"grounded:{is_grounded}")

        # 3. Evidence Quality Score
        evidence_score = self._score_evidence(evidence_quote, value)
        if evidence_quote:
            signals_used.append(f"quote_len:{len(evidence_quote)}")

        # 4. Retrieval Quality Score
        retrieval_score = self._score_retrieval(reranker_scores, chunks_searched)
        if reranker_scores:
            signals_used.append(f"reranker_top:{reranker_scores[0]:.2f}")
        else:
            signals_used.append(f"chunks:{chunks_searched}")

        # Weighted combination
        total_score = (
            llm_score * weights.llm_confidence +
            grounding_component * weights.grounding +
            evidence_score * weights.evidence_quality +
            retrieval_score * weights.retrieval_quality
        )

        # Determine confidence level
        if total_score >= self.config.high_confidence_threshold:
            level = "high"
        elif total_score >= self.config.low_confidence_threshold:
            level = "medium"
        else:
            level = "low"

        return ConfidenceResult(
            field_name=field_name,
            value=value,
            confidence_score=total_score,
            confidence_level=level,
            llm_score=llm_score,
            grounding_score=grounding_component,
            evidence_score=evidence_score,
            retrieval_score=retrieval_score,
            signals_used=signals_used,
        )

    def _score_llm_confidence(self, confidence: Optional[str]) -> float:
        """Score based on LLM's self-reported confidence."""
        if confidence is None:
            return 0.5  # Neutral when not provided

        confidence_lower = confidence.lower()
        if confidence_lower == "explicit":
            return 1.0
        elif confidence_lower == "inferred":
            return 0.4
        elif confidence_lower == "not_found":
            return 0.0
        else:
            return 0.5  # Unknown confidence level

    def _score_grounding(
        self,
        is_grounded: Optional[bool],
        grounding_score: Optional[float],
    ) -> float:
        """Score based on grounding validation."""
        if is_grounded is None and grounding_score is None:
            return 0.5  # Neutral when not validated

        score = 0.0

        # Binary grounding check (major component)
        if is_grounded is True:
            score += 0.7
        elif is_grounded is False:
            score += 0.0
        else:
            score += 0.35  # Unknown

        # Grounding score (minor component, for partial matches)
        if grounding_score is not None:
            score += grounding_score * 0.3
        else:
            score += 0.15  # Neutral

        return min(score, 1.0)

    def _score_evidence(
        self,
        evidence_quote: Optional[str],
        value: Any,
    ) -> float:
        """Score based on evidence quality."""
        if not evidence_quote:
            return 0.2  # Low score when no evidence provided

        score = 0.0
        quote_len = len(evidence_quote)

        # Quote length scoring
        if quote_len >= self.config.min_quote_length_long:
            score += 0.6
        elif quote_len >= self.config.min_quote_length_medium:
            score += 0.4
        elif quote_len >= self.config.min_quote_length_short:
            score += 0.2

        # Value appears in quote (strong signal)
        if value is not None:
            value_str = str(value)
            if value_str in evidence_quote:
                score += 0.4
            elif value_str.lower() in evidence_quote.lower():
                score += 0.3

        return min(score, 1.0)

    def _score_retrieval(
        self,
        reranker_scores: Optional[List[float]],
        chunks_searched: int,
    ) -> float:
        """Score based on retrieval quality."""
        # Reranker scores available
        if reranker_scores and len(reranker_scores) > 0:
            top_score = reranker_scores[0]
            if top_score >= self.config.reranker_high_score:
                return 1.0
            elif top_score >= self.config.reranker_medium_score:
                return 0.7
            elif top_score >= self.config.reranker_low_score:
                return 0.4
            else:
                return 0.2

        # Fallback: chunk coverage as proxy
        if chunks_searched >= 10:
            return 0.7
        elif chunks_searched >= 5:
            return 0.5
        elif chunks_searched >= 2:
            return 0.3
        else:
            return 0.1


@dataclass
class EnsembleResult:
    """Result of ensemble selection between two extraction methods."""
    selected_value: Any
    selected_method: str  # "t3_k10", "reranker", "agreement", "review_needed"
    confidence_score: float
    confidence_level: str

    # Individual results
    t3_result: Optional[ConfidenceResult] = None
    reranker_result: Optional[ConfidenceResult] = None

    # Decision metadata
    decision_reason: str = ""
    confidence_gap: float = 0.0

    def to_dict(self) -> dict:
        return {
            "selected_value": self.selected_value,
            "selected_method": self.selected_method,
            "confidence_score": round(self.confidence_score, 3),
            "confidence_level": self.confidence_level,
            "decision_reason": self.decision_reason,
            "confidence_gap": round(self.confidence_gap, 3),
            "t3_confidence": self.t3_result.confidence_score if self.t3_result else None,
            "reranker_confidence": self.reranker_result.confidence_score if self.reranker_result else None,
        }


class FieldTypeRouter:
    """
    Routes fields to the best extraction method based on field type patterns.

    Based on empirical analysis:
    - T3 k=10 is better for: boolean fields, categorical fields, policy descriptions
    - Reranker is better for: share class details, numeric values, investment amounts
    """

    # Field patterns where T3 k=10 performs better
    T3_PREFERRED_PATTERNS = [
        "has_*",                    # Boolean fields
        "*_policy",                 # Policy fields
        "*_basis",                  # Categorical basis fields
        "fund_type",                # Fund type
        "default_*",                # Default settings
    ]

    # Field patterns where Reranker performs better
    RERANKER_PREFERRED_PATTERNS = [
        "share_classes.*minimum*",  # Share class investment minimums
        "*expense_cap_pct",         # Expense percentages
        "distribution_frequency",   # Distribution frequency
        "leverage_basis",           # Leverage basis (specific case)
    ]

    def __init__(self, default_method: str = "t3_k10"):
        """
        Initialize router.

        Args:
            default_method: Method to use when no pattern matches
        """
        self.default_method = default_method

    def route(self, field_name: str) -> str:
        """
        Determine which method to use for a given field.

        Args:
            field_name: Name of the field (e.g., "incentive_fee.has_incentive_fee")

        Returns:
            Method name: "t3_k10" or "reranker"
        """
        # Check T3-preferred patterns
        for pattern in self.T3_PREFERRED_PATTERNS:
            if self._match_pattern(field_name, pattern):
                return "t3_k10"

        # Check Reranker-preferred patterns
        for pattern in self.RERANKER_PREFERRED_PATTERNS:
            if self._match_pattern(field_name, pattern):
                return "reranker"

        return self.default_method

    def _match_pattern(self, field_name: str, pattern: str) -> bool:
        """Simple pattern matching for field names."""
        field_lower = field_name.lower()
        pattern_lower = pattern.lower()

        if pattern_lower.startswith("*") and pattern_lower.endswith("*"):
            return pattern_lower[1:-1] in field_lower
        elif pattern_lower.startswith("*"):
            return field_lower.endswith(pattern_lower[1:])
        elif pattern_lower.endswith("*"):
            return field_lower.startswith(pattern_lower[:-1])
        else:
            return pattern_lower in field_lower


@dataclass
class RoutingResult:
    """Result of field-type routing selection."""
    field_name: str
    selected_value: Any
    selected_method: str
    routing_reason: str

    def to_dict(self) -> dict:
        return {
            "field_name": self.field_name,
            "selected_value": self.selected_value,
            "selected_method": self.selected_method,
            "routing_reason": self.routing_reason,
        }


class EnsembleSelector:
    """
    Selects the best extraction result from multiple methods using confidence scores.

    Strategies:
    - Agreement: If both methods agree, use that value with boosted confidence
    - Confidence-based: Pick the method with higher confidence score
    - Threshold-based: Only pick if confidence gap exceeds threshold
    """

    def __init__(
        self,
        scorer: ConfidenceScorer,
        min_confidence_gap: float = 0.15,
        agreement_boost: float = 0.1,
    ):
        """
        Initialize ensemble selector.

        Args:
            scorer: ConfidenceScorer instance
            min_confidence_gap: Minimum gap needed to prefer one method over another
            agreement_boost: Confidence boost when both methods agree
        """
        self.scorer = scorer
        self.min_confidence_gap = min_confidence_gap
        self.agreement_boost = agreement_boost

    def select(
        self,
        field_name: str,
        t3_result: ConfidenceResult,
        reranker_result: ConfidenceResult,
    ) -> EnsembleResult:
        """
        Select the best result from T3 and Reranker extractions.

        Args:
            field_name: Name of the field
            t3_result: Confidence result from T3 k=10 extraction
            reranker_result: Confidence result from Reranker extraction

        Returns:
            EnsembleResult with selected value and decision metadata
        """
        t3_value = t3_result.value
        rr_value = reranker_result.value
        t3_conf = t3_result.confidence_score
        rr_conf = reranker_result.confidence_score

        # Case 1: Both null
        if t3_value is None and rr_value is None:
            return EnsembleResult(
                selected_value=None,
                selected_method="agreement",
                confidence_score=0.0,
                confidence_level="low",
                t3_result=t3_result,
                reranker_result=reranker_result,
                decision_reason="Both methods returned null",
            )

        # Case 2: One null, one has value
        if t3_value is None:
            return EnsembleResult(
                selected_value=rr_value,
                selected_method="reranker",
                confidence_score=rr_conf,
                confidence_level=reranker_result.confidence_level,
                t3_result=t3_result,
                reranker_result=reranker_result,
                decision_reason="T3 returned null, using reranker",
            )

        if rr_value is None:
            return EnsembleResult(
                selected_value=t3_value,
                selected_method="t3_k10",
                confidence_score=t3_conf,
                confidence_level=t3_result.confidence_level,
                t3_result=t3_result,
                reranker_result=reranker_result,
                decision_reason="Reranker returned null, using T3",
            )

        # Case 3: Both agree
        if self._values_equal(t3_value, rr_value):
            boosted_conf = min(max(t3_conf, rr_conf) + self.agreement_boost, 1.0)
            level = "high" if boosted_conf >= 0.7 else "medium" if boosted_conf >= 0.4 else "low"
            return EnsembleResult(
                selected_value=t3_value,
                selected_method="agreement",
                confidence_score=boosted_conf,
                confidence_level=level,
                t3_result=t3_result,
                reranker_result=reranker_result,
                decision_reason=f"Both methods agree (boosted +{self.agreement_boost})",
            )

        # Case 4: Disagreement - use confidence gap
        confidence_gap = abs(t3_conf - rr_conf)

        if confidence_gap >= self.min_confidence_gap:
            if t3_conf > rr_conf:
                return EnsembleResult(
                    selected_value=t3_value,
                    selected_method="t3_k10",
                    confidence_score=t3_conf,
                    confidence_level=t3_result.confidence_level,
                    t3_result=t3_result,
                    reranker_result=reranker_result,
                    decision_reason=f"T3 higher confidence (gap={confidence_gap:.2f})",
                    confidence_gap=confidence_gap,
                )
            else:
                return EnsembleResult(
                    selected_value=rr_value,
                    selected_method="reranker",
                    confidence_score=rr_conf,
                    confidence_level=reranker_result.confidence_level,
                    t3_result=t3_result,
                    reranker_result=reranker_result,
                    decision_reason=f"Reranker higher confidence (gap={confidence_gap:.2f})",
                    confidence_gap=confidence_gap,
                )

        # Case 5: Too close to call - flag for review or use default
        # Default to T3 as it has fewer hallucinations historically
        return EnsembleResult(
            selected_value=t3_value,
            selected_method="t3_k10",
            confidence_score=t3_conf,
            confidence_level=t3_result.confidence_level,
            t3_result=t3_result,
            reranker_result=reranker_result,
            decision_reason=f"Disagreement but gap too small ({confidence_gap:.2f}), defaulting to T3",
            confidence_gap=confidence_gap,
        )

    def _values_equal(self, v1: Any, v2: Any) -> bool:
        """Check if two values are equal, handling type differences."""
        if v1 == v2:
            return True

        # Handle numeric comparisons with tolerance
        try:
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                return abs(float(v1) - float(v2)) < 0.001
        except (TypeError, ValueError):
            pass

        # String comparison (case-insensitive)
        if isinstance(v1, str) and isinstance(v2, str):
            return v1.lower().strip() == v2.lower().strip()

        return False
