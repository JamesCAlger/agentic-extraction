"""
Grounding Strategies for Extraction Validation.

This module provides multiple approaches to verify that extracted values
are grounded in source text, helping catch hallucinations.

Strategies:
- ExactMatch: Current approach - literal string matching (fast, high false-negative rate)
- NLI: Natural Language Inference model (fast, better semantic understanding)
- LLMJudge: LLM-as-a-judge (slow, best accuracy, handles inference/calculations)
- SemanticSimilarity: Embedding-based similarity (fast, moderate accuracy)
- Hybrid: NLI + LLM fallback (balanced cost/accuracy)

Usage:
    config = GroundingStrategyConfig(strategy="hybrid", nli_threshold=0.7)
    strategy = create_grounding_strategy(config)
    result = strategy.verify(claim="minimum investment is $25,000", evidence="...", source_chunks=[...])
"""

import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class GroundingStrategy(str, Enum):
    """Available grounding strategies."""
    EXACT_MATCH = "exact_match"
    NLI = "nli"
    LLM_JUDGE = "llm_judge"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    HYBRID = "hybrid"


@dataclass
class GroundingStrategyConfig:
    """Configuration for grounding strategies."""

    # Strategy selection
    strategy: str = "exact_match"  # exact_match, nli, llm_judge, semantic_similarity, hybrid

    # NLI settings
    nli_model: str = "cross-encoder/nli-deberta-v3-base"  # or "roberta-large-mnli"
    nli_entailment_threshold: float = 0.7  # Min score to consider "entailed"
    nli_contradiction_threshold: float = 0.7  # Min score to consider "contradicted"

    # LLM Judge settings
    llm_judge_model: str = "gpt-4o-mini"
    llm_judge_provider: str = "openai"
    llm_judge_temperature: float = 0.0

    # Semantic Similarity settings
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.75

    # Hybrid settings (NLI + LLM fallback)
    hybrid_nli_high_threshold: float = 0.85  # Above this, accept without LLM
    hybrid_nli_low_threshold: float = 0.4   # Below this, reject without LLM
    hybrid_use_llm_for_ambiguous: bool = True

    # General settings
    max_evidence_length: int = 2000  # Truncate evidence for efficiency
    cache_results: bool = True


@dataclass
class GroundingVerification:
    """Result of grounding verification for a single claim."""

    is_grounded: bool
    confidence: float  # 0.0 to 1.0 (soft score)
    strategy_used: str

    # Detailed signals
    entailment_score: Optional[float] = None  # From NLI
    contradiction_score: Optional[float] = None  # From NLI
    similarity_score: Optional[float] = None  # From embedding similarity
    llm_judgment: Optional[str] = None  # "SUPPORTED", "NOT_SUPPORTED", "PARTIAL"
    llm_reasoning: Optional[str] = None

    # Issues and explanations
    issues: list[str] = field(default_factory=list)
    explanation: str = ""


class BaseGroundingStrategy(ABC):
    """Abstract base class for grounding strategies."""

    def __init__(self, config: GroundingStrategyConfig):
        self.config = config

    @abstractmethod
    def verify(
        self,
        claim: str,
        evidence: str,
        source_chunks: Optional[list[str]] = None,
        field_name: Optional[str] = None,
        value: Any = None,
    ) -> GroundingVerification:
        """
        Verify if a claim is grounded in the evidence.

        Args:
            claim: The claim to verify (e.g., "minimum investment is $25,000")
            evidence: The primary evidence text
            source_chunks: Additional source chunks for context
            field_name: Name of the field being verified
            value: The extracted value

        Returns:
            GroundingVerification with grounding status and confidence
        """
        pass

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to max length."""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."


class ExactMatchStrategy(BaseGroundingStrategy):
    """
    Exact match grounding strategy (current approach).

    Checks if values literally appear in the source text.
    Fast but has high false-negative rate for format differences.
    """

    def verify(
        self,
        claim: str,
        evidence: str,
        source_chunks: Optional[list[str]] = None,
        field_name: Optional[str] = None,
        value: Any = None,
    ) -> GroundingVerification:
        """Check for exact/near-exact matches in source."""

        # Combine evidence and chunks
        full_source = evidence
        if source_chunks:
            full_source = evidence + " " + " ".join(source_chunks)
        full_source_lower = full_source.lower()

        issues = []
        found = False

        # Check if value appears in text
        if value is not None:
            value_str = str(value).lower()

            # Try different formats
            formats_to_try = [
                value_str,
                value_str.replace(",", ""),
                value_str.replace("$", ""),
                value_str.rstrip("%"),
            ]

            # For numbers, try with/without formatting
            try:
                num = float(str(value).replace(",", "").replace("$", "").replace("%", ""))
                formats_to_try.extend([
                    f"{num}",
                    f"{num:.0f}",
                    f"{num:.1f}",
                    f"{num:.2f}",
                    f"{num:,.0f}",
                    f"${num:,.0f}",
                    f"{num}%",
                ])
            except ValueError:
                pass

            for fmt in formats_to_try:
                if fmt.lower() in full_source_lower:
                    found = True
                    break

        # Check if claim keywords appear
        if not found:
            # Check for key phrases from claim
            claim_words = [w for w in claim.lower().split() if len(w) > 3]
            matches = sum(1 for w in claim_words if w in full_source_lower)
            if matches >= len(claim_words) * 0.5:
                found = True

        if not found:
            issues.append(f"Value '{value}' not found in source text")

        confidence = 1.0 if found else 0.0

        return GroundingVerification(
            is_grounded=found,
            confidence=confidence,
            strategy_used="exact_match",
            issues=issues,
            explanation="Exact string matching" if found else "No exact match found",
        )


class NLIStrategy(BaseGroundingStrategy):
    """
    NLI-based grounding strategy.

    Uses a Natural Language Inference model to check if evidence entails the claim.
    Faster and cheaper than LLM, better semantic understanding than exact match.
    """

    def __init__(self, config: GroundingStrategyConfig):
        super().__init__(config)
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load NLI model."""
        if self._model is None:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import torch

                logger.info(f"[NLI] Loading model: {self.config.nli_model}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.config.nli_model)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.config.nli_model)
                self._model.eval()

                # Move to GPU if available
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
                    logger.info("[NLI] Using GPU")

            except ImportError:
                logger.error("[NLI] transformers library not installed. Install with: pip install transformers torch")
                raise

    def verify(
        self,
        claim: str,
        evidence: str,
        source_chunks: Optional[list[str]] = None,
        field_name: Optional[str] = None,
        value: Any = None,
    ) -> GroundingVerification:
        """Check if evidence entails the claim using NLI."""

        self._load_model()

        import torch

        # Truncate evidence
        evidence = self._truncate_text(evidence, self.config.max_evidence_length)

        # Format for NLI: premise (evidence) -> hypothesis (claim)
        inputs = self._tokenizer(
            evidence,
            claim,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Move to GPU if model is on GPU
        if next(self._model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        # Most NLI models: [contradiction, neutral, entailment] or [entailment, neutral, contradiction]
        # DeBERTa NLI: [contradiction, neutral, entailment]
        if "deberta" in self.config.nli_model.lower():
            contradiction_score = probs[0].item()
            neutral_score = probs[1].item()
            entailment_score = probs[2].item()
        else:
            # RoBERTa MNLI: [contradiction, neutral, entailment]
            contradiction_score = probs[0].item()
            neutral_score = probs[1].item()
            entailment_score = probs[2].item()

        # Determine grounding
        is_grounded = (
            entailment_score >= self.config.nli_entailment_threshold and
            contradiction_score < self.config.nli_contradiction_threshold
        )

        # Calculate confidence (favor entailment, penalize contradiction)
        confidence = entailment_score * (1 - contradiction_score)

        issues = []
        if contradiction_score >= self.config.nli_contradiction_threshold:
            issues.append(f"NLI detected contradiction (score: {contradiction_score:.2f})")
        if entailment_score < self.config.nli_entailment_threshold:
            issues.append(f"NLI entailment score below threshold ({entailment_score:.2f} < {self.config.nli_entailment_threshold})")

        explanation = (
            f"NLI scores - entailment: {entailment_score:.2f}, "
            f"neutral: {neutral_score:.2f}, contradiction: {contradiction_score:.2f}"
        )

        return GroundingVerification(
            is_grounded=is_grounded,
            confidence=confidence,
            strategy_used="nli",
            entailment_score=entailment_score,
            contradiction_score=contradiction_score,
            issues=issues,
            explanation=explanation,
        )


class LLMJudgeStrategy(BaseGroundingStrategy):
    """
    LLM-as-a-judge grounding strategy.

    Uses an LLM to reason about whether evidence supports the claim.
    Most accurate, handles inference/calculations, but slowest and most expensive.
    """

    def __init__(self, config: GroundingStrategyConfig, api_key: Optional[str] = None):
        super().__init__(config)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None

    def _get_client(self):
        """Lazy load LLM client."""
        if self._client is None:
            if self.config.llm_judge_provider == "openai":
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            elif self.config.llm_judge_provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"))
            else:
                raise ValueError(f"Unknown provider: {self.config.llm_judge_provider}")
        return self._client

    def verify(
        self,
        claim: str,
        evidence: str,
        source_chunks: Optional[list[str]] = None,
        field_name: Optional[str] = None,
        value: Any = None,
    ) -> GroundingVerification:
        """Use LLM to judge if evidence supports the claim."""

        client = self._get_client()

        # Build context
        context = evidence
        if source_chunks:
            additional = "\n\n".join(source_chunks[:3])  # Limit chunks
            context = f"{evidence}\n\nAdditional context:\n{additional}"

        context = self._truncate_text(context, self.config.max_evidence_length)

        prompt = f"""You are a precise fact-checker for financial document extraction.

Source Text:
\"\"\"
{context}
\"\"\"

Claim to verify: {claim}
{f"Field: {field_name}" if field_name else ""}
{f"Extracted value: {value}" if value is not None else ""}

Determine if this claim is SUPPORTED by the source text.

Consider:
1. The claim may use different formatting (e.g., "$25,000" vs "twenty-five thousand dollars")
2. The claim may be a reasonable inference from stated facts
3. The claim may involve simple calculations from stated values
4. For boolean claims (e.g., "has X"), look for evidence that X exists or is mentioned

Answer with one of:
- SUPPORTED: The claim is directly stated or clearly inferable from the source
- NOT_SUPPORTED: The claim contradicts the source or has no basis in it
- PARTIAL: The claim is partially supported but some aspects are uncertain

Format your response as:
JUDGMENT: [SUPPORTED/NOT_SUPPORTED/PARTIAL]
CONFIDENCE: [0.0-1.0]
REASONING: [1-2 sentences explaining your judgment]"""

        try:
            if self.config.llm_judge_provider == "openai":
                response = client.chat.completions.create(
                    model=self.config.llm_judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.llm_judge_temperature,
                    max_tokens=200,
                )
                result_text = response.choices[0].message.content
            else:
                response = client.messages.create(
                    model=self.config.llm_judge_model,
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}],
                )
                result_text = response.content[0].text

            # Parse response
            judgment, confidence, reasoning = self._parse_llm_response(result_text)

            is_grounded = judgment == "SUPPORTED"

            issues = []
            if judgment == "NOT_SUPPORTED":
                issues.append(f"LLM judge: claim not supported - {reasoning}")
            elif judgment == "PARTIAL":
                issues.append(f"LLM judge: partial support - {reasoning}")

            return GroundingVerification(
                is_grounded=is_grounded,
                confidence=confidence,
                strategy_used="llm_judge",
                llm_judgment=judgment,
                llm_reasoning=reasoning,
                issues=issues,
                explanation=reasoning,
            )

        except Exception as e:
            logger.error(f"[LLM Judge] Error: {e}")
            return GroundingVerification(
                is_grounded=False,
                confidence=0.0,
                strategy_used="llm_judge",
                issues=[f"LLM judge error: {str(e)}"],
                explanation="Error during LLM verification",
            )

    def _parse_llm_response(self, response: str) -> tuple[str, float, str]:
        """Parse LLM judge response."""
        judgment = "NOT_SUPPORTED"
        confidence = 0.5
        reasoning = ""

        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("JUDGMENT:"):
                j = line.replace("JUDGMENT:", "").strip().upper()
                if "SUPPORTED" in j and "NOT" not in j:
                    judgment = "SUPPORTED"
                elif "PARTIAL" in j:
                    judgment = "PARTIAL"
                else:
                    judgment = "NOT_SUPPORTED"
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    pass
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        return judgment, confidence, reasoning


class SemanticSimilarityStrategy(BaseGroundingStrategy):
    """
    Semantic similarity-based grounding strategy.

    Uses embeddings to find the most similar text span and threshold.
    Fast, but less nuanced than NLI or LLM approaches.
    """

    def __init__(self, config: GroundingStrategyConfig):
        super().__init__(config)
        self._model = None

    def _load_model(self):
        """Lazy load embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"[Semantic] Loading model: {self.config.embedding_model}")
                self._model = SentenceTransformer(self.config.embedding_model)
            except ImportError:
                logger.error("[Semantic] sentence-transformers not installed. Install with: pip install sentence-transformers")
                raise

    def verify(
        self,
        claim: str,
        evidence: str,
        source_chunks: Optional[list[str]] = None,
        field_name: Optional[str] = None,
        value: Any = None,
    ) -> GroundingVerification:
        """Check semantic similarity between claim and evidence."""

        self._load_model()

        # Get all source text
        texts = [evidence]
        if source_chunks:
            texts.extend(source_chunks)

        # Split into sentences for finer-grained matching
        sentences = []
        for text in texts:
            # Simple sentence splitting
            for sent in re.split(r'[.!?]+', text):
                sent = sent.strip()
                if len(sent) > 20:
                    sentences.append(sent)

        if not sentences:
            return GroundingVerification(
                is_grounded=False,
                confidence=0.0,
                strategy_used="semantic_similarity",
                issues=["No valid sentences in source"],
                explanation="Could not extract sentences from source",
            )

        # Compute embeddings
        claim_embedding = self._model.encode([claim])[0]
        sentence_embeddings = self._model.encode(sentences)

        # Find most similar sentence
        from numpy import dot
        from numpy.linalg import norm

        similarities = []
        for sent_emb in sentence_embeddings:
            sim = dot(claim_embedding, sent_emb) / (norm(claim_embedding) * norm(sent_emb))
            similarities.append(sim)

        max_similarity = max(similarities)
        best_idx = similarities.index(max_similarity)
        best_sentence = sentences[best_idx]

        is_grounded = max_similarity >= self.config.similarity_threshold

        issues = []
        if not is_grounded:
            issues.append(
                f"Max similarity {max_similarity:.2f} below threshold {self.config.similarity_threshold}"
            )

        return GroundingVerification(
            is_grounded=is_grounded,
            confidence=max_similarity,
            strategy_used="semantic_similarity",
            similarity_score=max_similarity,
            issues=issues,
            explanation=f"Most similar sentence (score={max_similarity:.2f}): '{best_sentence[:100]}...'",
        )


class HybridStrategy(BaseGroundingStrategy):
    """
    Hybrid grounding strategy: NLI + LLM fallback.

    Fast path: Use NLI for clear entailment/contradiction
    Slow path: Use LLM for ambiguous cases (neutral zone)

    This provides the best balance of accuracy and cost.
    """

    def __init__(self, config: GroundingStrategyConfig, api_key: Optional[str] = None):
        super().__init__(config)
        self._nli = NLIStrategy(config)
        self._llm = LLMJudgeStrategy(config, api_key)

    def verify(
        self,
        claim: str,
        evidence: str,
        source_chunks: Optional[list[str]] = None,
        field_name: Optional[str] = None,
        value: Any = None,
    ) -> GroundingVerification:
        """
        Hybrid verification: NLI first, LLM for ambiguous cases.

        Decision logic:
        1. If NLI entailment > high_threshold: GROUNDED (skip LLM)
        2. If NLI contradiction > threshold: NOT GROUNDED (skip LLM)
        3. Otherwise (neutral zone): Use LLM judge
        """

        # Step 1: NLI check
        nli_result = self._nli.verify(claim, evidence, source_chunks, field_name, value)

        # High confidence entailment - accept without LLM
        if (nli_result.entailment_score is not None and
            nli_result.entailment_score >= self.config.hybrid_nli_high_threshold):
            logger.debug(
                f"[Hybrid] NLI high confidence entailment ({nli_result.entailment_score:.2f}), skipping LLM"
            )
            nli_result.strategy_used = "hybrid_nli_fast"
            return nli_result

        # High confidence contradiction - reject without LLM
        if (nli_result.contradiction_score is not None and
            nli_result.contradiction_score >= self.config.nli_contradiction_threshold):
            logger.debug(
                f"[Hybrid] NLI detected contradiction ({nli_result.contradiction_score:.2f}), skipping LLM"
            )
            nli_result.strategy_used = "hybrid_nli_reject"
            nli_result.is_grounded = False
            nli_result.confidence = 0.1
            return nli_result

        # Low NLI score and not using LLM fallback
        if not self.config.hybrid_use_llm_for_ambiguous:
            nli_result.strategy_used = "hybrid_nli_only"
            return nli_result

        # Step 2: Ambiguous - use LLM judge
        logger.debug(
            f"[Hybrid] NLI ambiguous (ent={nli_result.entailment_score:.2f}, "
            f"contra={nli_result.contradiction_score:.2f}), using LLM"
        )

        llm_result = self._llm.verify(claim, evidence, source_chunks, field_name, value)

        # Combine results
        llm_result.strategy_used = "hybrid_llm_fallback"
        llm_result.entailment_score = nli_result.entailment_score
        llm_result.contradiction_score = nli_result.contradiction_score

        return llm_result


# =============================================================================
# Factory Function
# =============================================================================


def create_grounding_strategy(
    config: GroundingStrategyConfig,
    api_key: Optional[str] = None,
) -> BaseGroundingStrategy:
    """
    Create a grounding strategy based on configuration.

    Args:
        config: Strategy configuration
        api_key: API key for LLM-based strategies

    Returns:
        Configured grounding strategy
    """
    strategy_map = {
        "exact_match": lambda: ExactMatchStrategy(config),
        "nli": lambda: NLIStrategy(config),
        "llm_judge": lambda: LLMJudgeStrategy(config, api_key),
        "semantic_similarity": lambda: SemanticSimilarityStrategy(config),
        "hybrid": lambda: HybridStrategy(config, api_key),
    }

    if config.strategy not in strategy_map:
        raise ValueError(
            f"Unknown grounding strategy: {config.strategy}. "
            f"Available: {list(strategy_map.keys())}"
        )

    logger.info(f"[Grounding] Using strategy: {config.strategy}")
    return strategy_map[config.strategy]()


# =============================================================================
# High-Level Verification Function
# =============================================================================


def verify_extraction_grounding(
    field_name: str,
    value: Any,
    evidence: str,
    source_chunks: Optional[list[str]] = None,
    strategy: Optional[BaseGroundingStrategy] = None,
    config: Optional[GroundingStrategyConfig] = None,
) -> GroundingVerification:
    """
    Verify that an extracted value is grounded in source text.

    Args:
        field_name: Name of the extracted field
        value: The extracted value
        evidence: Primary evidence text
        source_chunks: Additional source chunks
        strategy: Pre-configured grounding strategy (takes precedence)
        config: Configuration if strategy not provided

    Returns:
        GroundingVerification result
    """
    # Create default strategy if not provided
    if strategy is None:
        config = config or GroundingStrategyConfig()
        strategy = create_grounding_strategy(config)

    # Format claim from field name and value
    claim = format_claim(field_name, value)

    return strategy.verify(
        claim=claim,
        evidence=evidence,
        source_chunks=source_chunks,
        field_name=field_name,
        value=value,
    )


def format_claim(field_name: str, value: Any) -> str:
    """
    Format a field name and value into a natural language claim.

    Examples:
        - ("minimum_investment", 25000) -> "the minimum investment is $25,000"
        - ("has_incentive_fee", True) -> "the fund has an incentive fee"
    """
    # Handle boolean values specially
    if isinstance(value, bool):
        field_readable = field_name.replace("_", " ")
        if field_name.startswith("has_"):
            thing = field_name[4:].replace("_", " ")
            return f"the fund has {thing}" if value else f"the fund does not have {thing}"
        return f"{field_readable} is {str(value).lower()}"

    # Handle numeric values
    field_readable = field_name.replace("_", " ")

    if value is None:
        return f"the {field_readable} is not specified"

    # Format currency values
    if "investment" in field_name.lower() or "minimum" in field_name.lower():
        try:
            num = float(str(value).replace(",", "").replace("$", ""))
            return f"the {field_readable} is ${num:,.0f}"
        except ValueError:
            pass

    # Format percentage values
    if "_pct" in field_name or "rate" in field_name or "percentage" in field_name:
        return f"the {field_readable} is {value}%"

    return f"the {field_readable} is {value}"
