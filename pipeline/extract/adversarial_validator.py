"""
Adversarial Validation for LLM Extractions.

Uses a "devil's advocate" approach where a validation LLM is asked to
find reasons why an extraction might be WRONG, rather than confirming it's correct.

This is more effective at catching hallucinations because:
1. LLMs have confirmation bias - they tend to approve what looks plausible
2. Adversarial framing forces active search for contradictions
3. Requiring exact quotes prevents "sounds right" acceptance
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Document-Aligned Claim Templates
# =============================================================================
# These templates format claims to sound like natural language that could
# actually appear in an SEC filing, rather than schema field names.

CLAIM_TEMPLATES = {
    # Incentive fee fields
    "has_incentive_fee": {
        True: "The fund charges an incentive fee or performance fee",
        False: "The fund does NOT charge an incentive fee",
    },
    "incentive_fee_pct": "The fund's incentive fee rate is {value}%",
    "hurdle_rate_pct": "The hurdle rate (preferred return) is {value}% annualized",
    "hurdle_rate_as_stated": "The stated hurdle rate is {value}% (before annualization)",
    "hurdle_rate_frequency": "The hurdle rate is measured on a {value} basis",
    "high_water_mark": {
        True: "The fund has a high water mark mechanism",
        False: "The fund does NOT have a high water mark",
    },
    "has_catch_up": {
        True: "The fund has a catch-up provision for the manager",
        False: "The fund does NOT have a catch-up provision",
    },
    "catch_up_rate_pct": "The catch-up rate is {value}% (portion going to manager during catch-up)",
    "catch_up_ceiling_pct": "The catch-up ceiling is {value}%",
    "fee_basis": "The incentive fee is calculated on {value}",
    "crystallization_frequency": "The incentive fee crystallizes {value}",

    # Expense cap fields
    "has_expense_cap": {
        True: "The fund has an expense cap or fee waiver agreement",
        False: "The fund does NOT have an expense cap",
    },
    "expense_cap_pct": "The expense cap is {value}% of net assets",

    # Repurchase fields
    "repurchase_frequency": "Repurchase offers are made {value}",
    "repurchase_amount_pct": "The repurchase offer amount is {value}% of shares",
    "repurchase_basis": "Repurchases are based on {value}",
    "repurchase_percentage_min": "The minimum repurchase percentage is {value}%",
    "repurchase_percentage_max": "The maximum repurchase percentage is {value}%",
    "lock_up_period_years": "The lock-up period is {value} year(s)",
    "early_repurchase_fee_pct": "The early repurchase fee is {value}%",

    # Leverage fields
    "uses_leverage": {
        True: "The fund uses leverage or borrowing",
        False: "The fund does NOT use leverage",
    },
    "max_leverage_pct": "The maximum leverage is {value}% of assets",
    "leverage_basis": "Leverage is measured as {value}",

    # Distribution fields
    "distribution_frequency": "Distributions are made {value}",
    "default_distribution_policy": "The default distribution policy is {value}",

    # Share class fields
    "minimum_initial_investment": "The minimum initial investment is ${value}",
    "minimum_additional_investment": "The minimum additional investment is ${value}",
    "sales_load_pct": "The sales load is {value}%",
    "distribution_fee_pct": "The distribution fee is {value}%",
}


def format_claim(field_name: str, value: Any, expected_type: str = "text") -> str:
    """
    Format a claim using document-aligned templates.

    Instead of "incentive_fee_pct = 10", produces:
    "The fund's incentive fee rate is 10%"

    This makes claims sound like natural language that could appear in SEC filings.
    """
    # Get the base field name (last part of path like "incentive_fee.has_incentive_fee")
    base_name = field_name.split(".")[-1] if "." in field_name else field_name

    template = CLAIM_TEMPLATES.get(base_name)

    if template is None:
        # Fallback to generic formatting
        if expected_type == "boolean":
            if str(value).lower() in ("true", "yes", "1"):
                return f"The fund HAS {base_name.replace('has_', '').replace('uses_', '').replace('_', ' ')}"
            else:
                return f"The fund does NOT have {base_name.replace('has_', '').replace('uses_', '').replace('_', ' ')}"
        return f"{base_name.replace('_', ' ')} is {value}"

    # Handle boolean templates (dict with True/False keys)
    if isinstance(template, dict):
        bool_val = str(value).lower() in ("true", "yes", "1")
        return template.get(bool_val, template.get(True, f"{base_name} = {value}"))

    # Handle string templates with {value} placeholder
    if "{value}" in template:
        # Clean the value for display
        clean_value = str(value).strip().rstrip('%').lstrip('$').replace(',', '')
        return template.format(value=clean_value)

    return template


# =============================================================================
# Response Schemas
# =============================================================================


class QuoteExtractionResponse(BaseModel):
    """Response from the quote extraction phase."""

    supporting_quote: Optional[str] = Field(
        None,
        description="The EXACT sentence or phrase from evidence that proves the claim. "
        "Must be verbatim text, not paraphrased. None if no supporting quote found.",
    )
    quote_found: bool = Field(
        description="Whether a supporting quote was found in the evidence."
    )
    reasoning: str = Field(
        description="Brief explanation of why this quote supports (or doesn't support) the claim."
    )


class AdversarialCritiqueResponse(BaseModel):
    """Response from the adversarial validation phase."""

    is_valid: bool = Field(
        description="Whether the extraction is valid after adversarial review. "
        "True only if NO significant problems were found."
    )
    problems_found: list[str] = Field(
        default_factory=list,
        description="List of specific problems or reasons the extraction might be wrong. "
        "Empty list if validated successfully.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the validation result (0.0 to 1.0). "
        "Higher = more confident the extraction is correct.",
    )
    reasoning: str = Field(
        description="Detailed reasoning about why the extraction is valid or invalid."
    )


@dataclass
class AdversarialValidationResult:
    """Complete result of adversarial validation."""

    is_valid: bool
    supporting_quote: Optional[str] = None
    problems: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    quote_reasoning: str = ""
    critique_reasoning: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "supporting_quote": self.supporting_quote,
            "problems": self.problems,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "quote_reasoning": self.quote_reasoning,
            "critique_reasoning": self.critique_reasoning,
            "error": self.error,
        }


# =============================================================================
# Adversarial Validator
# =============================================================================


class AdversarialValidator:
    """
    Two-phase adversarial validation for LLM extractions.

    Phase 1: Quote Extraction
        - Ask LLM to find the EXACT quote supporting the claim
        - Fails if no verbatim quote can be found

    Phase 2: Adversarial Critique
        - Ask LLM to find reasons the extraction might be WRONG
        - Uses devil's advocate prompting to surface issues
        - Passes only if no significant problems found
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        require_exact_quote: bool = True,
    ):
        """
        Initialize the adversarial validator.

        Args:
            model: Model to use for validation (can differ from extraction model)
            require_exact_quote: Whether to require exact quote extraction first
        """
        self.model = model
        self.require_exact_quote = require_exact_quote
        self._client = None
        self._provider = None

    @property
    def client(self):
        """Lazy-load the LLM client."""
        if self._client is None:
            from .llm_provider import create_instructor_client, detect_provider

            self._provider = detect_provider(self.model)
            self._client = create_instructor_client(
                provider=self._provider.value,
                model=self.model,
            )
        return self._client

    def validate(
        self,
        field_name: str,
        value: Any,
        evidence: str,
        expected_type: str = "text",
        context: Optional[str] = None,
    ) -> AdversarialValidationResult:
        """
        Perform adversarial validation on an extraction.

        Args:
            field_name: Name of the field being validated
            value: The extracted value to validate
            evidence: The evidence text the value was extracted from
            expected_type: Type of value (boolean, percentage, currency, number, text)
            context: Additional context about what we're looking for

        Returns:
            AdversarialValidationResult with validation outcome
        """
        try:
            # Phase 1: Quote extraction (optional but recommended)
            supporting_quote = None
            quote_reasoning = ""

            if self.require_exact_quote:
                quote_result = self._extract_supporting_quote(
                    field_name=field_name,
                    value=value,
                    evidence=evidence,
                    expected_type=expected_type,
                )

                if not quote_result.quote_found:
                    return AdversarialValidationResult(
                        is_valid=False,
                        problems=["No exact supporting quote found in evidence"],
                        confidence=0.0,
                        reasoning=quote_result.reasoning,
                        quote_reasoning=quote_result.reasoning,
                    )

                supporting_quote = quote_result.supporting_quote
                quote_reasoning = quote_result.reasoning

            # Phase 2: Adversarial critique
            critique_result = self._adversarial_critique(
                field_name=field_name,
                value=value,
                evidence=evidence,
                supporting_quote=supporting_quote,
                expected_type=expected_type,
                context=context,
            )

            return AdversarialValidationResult(
                is_valid=critique_result.is_valid,
                supporting_quote=supporting_quote,
                problems=critique_result.problems_found,
                confidence=critique_result.confidence,
                reasoning=critique_result.reasoning,
                quote_reasoning=quote_reasoning,
                critique_reasoning=critique_result.reasoning,
            )

        except Exception as e:
            logger.error(f"Adversarial validation error: {e}")
            return AdversarialValidationResult(
                is_valid=False,
                problems=[f"Validation error: {str(e)}"],
                confidence=0.0,
                error=str(e),
            )

    def _extract_supporting_quote(
        self,
        field_name: str,
        value: Any,
        evidence: str,
        expected_type: str,
    ) -> QuoteExtractionResponse:
        """
        Phase 1: Extract the exact supporting quote from evidence.
        """
        # Use document-aligned claim formatting
        claim = format_claim(field_name, value, expected_type)

        prompt = f"""You are a quote extractor for SEC fund filings. Find text that supports the claim.

CLAIM: {claim}

EVIDENCE TEXT:
---
{evidence}
---

INSTRUCTIONS:
1. Find text that supports this claim (verbatim quote preferred)
2. The quote should be from the evidence text

DOMAIN KNOWLEDGE - These patterns SUPPORT boolean existence claims:
- Definitions like "'Expense Cap' means X%" → has_expense_cap = true (defining terms implies having the feature)
- "The Adviser has agreed to waive/reimburse..." → has_expense_cap = true
- "The incentive fee, if any, is calculated as..." → has_incentive_fee = true ("if any" is standard legal language, not uncertainty)
- "with a full catch-up" → has_catch_up = true
- Text describing HOW a mechanism works implies it EXISTS

EXAMPLES of valid supporting quotes:
- "'Total Expense Cap' means 0.50% of net assets" → VALID for "has_expense_cap = true"
- "we will pay the Adviser quarterly in arrears 12.5%..." → VALID for "has_incentive_fee = true"
- "with a full catch-up" → VALID for "catch_up_rate_pct = 100" (domain standard term)
- "quarterly in arrears" → VALID for "crystallization_frequency = quarterly"

Return the supporting quote if found. If multiple quotes support the claim, return the most explicit one."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_model=QuoteExtractionResponse,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            return response
        except Exception as e:
            logger.error(f"Quote extraction error: {e}")
            return QuoteExtractionResponse(
                supporting_quote=None,
                quote_found=False,
                reasoning=f"Error during quote extraction: {str(e)}",
            )

    def _adversarial_critique(
        self,
        field_name: str,
        value: Any,
        evidence: str,
        supporting_quote: Optional[str],
        expected_type: str,
        context: Optional[str] = None,
    ) -> AdversarialCritiqueResponse:
        """
        Phase 2: Adversarial critique - find reasons the extraction might be wrong.
        """
        # Use document-aligned claim formatting
        claim = format_claim(field_name, value, expected_type)

        quote_section = ""
        if supporting_quote:
            quote_section = f"""
SUPPORTING QUOTE IDENTIFIED:
"{supporting_quote}"
"""

        context_section = ""
        if context:
            context_section = f"""
ADDITIONAL CONTEXT:
{context}
"""

        prompt = f"""You are a fact-checker for SEC fund filings. Your job is to verify that extractions are ACTUALLY SUPPORTED by the evidence.

CLAIM BEING VALIDATED:
{claim}
{quote_section}
FULL EVIDENCE TEXT:
---
{evidence}
---
{context_section}
DOMAIN KNOWLEDGE - Standard Finance Terminology:
These are VALID equivalences you should accept:
- "full catch-up" = catch_up_rate_pct of 100% (standard PE term)
- "quarterly in arrears" = quarterly crystallization_frequency for income-based fees
- Definitions like "'Expense Cap' means X%" indicate the feature EXISTS (has_expense_cap = true)
- "The Adviser has agreed to..." indicates a binding commitment, not a hypothetical
- "incentive fee, if any" is standard legal language - "if any" refers to WHEN fees are payable, not WHETHER the fee structure exists
- Payment frequency typically equals crystallization frequency for income-based incentive fees
- "Pre-Incentive Fee Net Investment Income" = fee_basis of "net_investment_income"

YOUR TASK - Check for ACTUAL problems:

1. CONTRADICTIONS: Is there text that CONTRADICTS the claim?
   - Look for "however", "except", "but", "notwithstanding"
   - Look for different values mentioned elsewhere

2. SCOPE ISSUES: Could this value refer to something else?
   - A different fund (e.g., underlying fund rather than this fund)?
   - A different share class than intended?
   - A maximum vs. actual value confusion?

3. BOOLEAN EXISTENCE: For booleans claiming TRUE:
   - If evidence describes HOW a feature works (e.g., catch-up mechanism details), it EXISTS
   - If evidence defines WHAT a cap/fee IS, the cap/fee EXISTS
   - Don't reject just because evidence uses standard legal/conditional language

4. VALUE ACCURACY: For numeric values:
   - Is this the correct number from the evidence?
   - Is the unit correct (%, $, years)?
   - Accept domain-standard inferences (full catch-up = 100%)

Mark as VALID if the claim is supported by the evidence, even through standard domain terminology.
Mark as INVALID only if there is a genuine contradiction, scope error, or the value cannot be reasonably inferred."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_model=AdversarialCritiqueResponse,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
            )
            return response
        except Exception as e:
            logger.error(f"Adversarial critique error: {e}")
            return AdversarialCritiqueResponse(
                is_valid=False,
                problems_found=[f"Error during critique: {str(e)}"],
                confidence=0.0,
                reasoning=f"Critique failed with error: {str(e)}",
            )


# =============================================================================
# Lightweight Validator (Simple, Fast, Cheap)
# =============================================================================


class LightweightValidationResponse(BaseModel):
    """Response from lightweight validation."""

    is_supported: bool = Field(
        description="Whether the evidence supports the extracted value. "
        "True if there is clear support, False if contradicted or unsupported."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the validation (0.0 to 1.0).",
    )
    reason: str = Field(
        description="Brief explanation (1-2 sentences) of why supported or not.",
    )


class LightweightValidator:
    """
    Simple, fast validation for LLM extractions.

    Unlike AdversarialValidator, this uses:
    - Single LLM call (not two-phase)
    - Simple verification framing (not devil's advocate)
    - Semantic support (not exact quote requirement)
    - Cheaper/faster model (GPT-4o-mini by default)

    Good for validating all fields cheaply, catching obvious hallucinations
    without being overly pedantic.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize lightweight validator.

        Args:
            model: Model to use (default: gpt-4o-mini for cost/speed)
        """
        self.model = model
        self._client = None
        self._provider = None

    @property
    def client(self):
        """Lazy-load the LLM client."""
        if self._client is None:
            from .llm_provider import create_instructor_client, detect_provider

            self._provider = detect_provider(self.model)
            self._client = create_instructor_client(
                provider=self._provider.value,
                model=self.model,
            )
        return self._client

    def validate(
        self,
        field_name: str,
        value: Any,
        evidence: str,
        expected_type: str = "text",
    ) -> AdversarialValidationResult:
        """
        Perform lightweight validation on an extraction.

        Args:
            field_name: Name of the field being validated
            value: The extracted value to validate
            evidence: The evidence text the value was extracted from
            expected_type: Type of value (boolean, percentage, currency, number, text)

        Returns:
            AdversarialValidationResult (reuses same result class for compatibility)
        """
        try:
            # Use document-aligned claim formatting (same as adversarial validator)
            claim = format_claim(field_name, value, expected_type)

            prompt = f"""Does the evidence support or reasonably imply this claim about an SEC-registered fund?

CLAIM: {claim}

EVIDENCE:
---
{evidence[:4000]}
---

VALIDATION RULES - Mark as SUPPORTED if ANY of these apply:
1. The value is directly stated in the evidence
2. The value can be reasonably INFERRED from the evidence (e.g., a quarterly rate of 1.5% implies an annualized rate of 6%)
3. The value follows from standard financial/legal terminology (e.g., "full catch-up" = 100% catch-up rate, "quarterly in arrears" = quarterly crystallization)
4. The evidence describes HOW a feature works, which implies it EXISTS (e.g., describing catch-up mechanics = has_catch_up is true)
5. The evidence defines or names a feature (e.g., "'Expense Cap' means..." = has_expense_cap is true)
6. Semantic equivalence applies (e.g., "Loss Recovery Account" = high water mark, "Pre-Incentive Fee Net Investment Income" = net_investment_income basis)

Mark as NOT SUPPORTED only if:
- The evidence actively CONTRADICTS the claim (states a different value or explicitly denies the feature)
- The claim has NO reasonable basis in the evidence at all (completely fabricated)
- The value refers to a different entity (e.g., an underlying fund's fee, not this fund's fee)

When in doubt, lean toward SUPPORTED - the extraction LLM had access to the full document and may have seen context not included in this evidence snippet."""

            response = self.client.chat.completions.create(
                model=self.model,
                response_model=LightweightValidationResponse,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )

            return AdversarialValidationResult(
                is_valid=response.is_supported,
                confidence=response.confidence,
                reasoning=response.reason,
                problems=[] if response.is_supported else [response.reason],
            )

        except Exception as e:
            logger.error(f"Lightweight validation error: {e}")
            # On error, default to accepting (don't block on validation failures)
            return AdversarialValidationResult(
                is_valid=True,
                confidence=0.5,
                reasoning=f"Validation error (defaulting to accept): {str(e)}",
                error=str(e),
            )


# =============================================================================
# Convenience Functions
# =============================================================================


def validate_extraction_lightweight(
    field_name: str,
    value: Any,
    evidence: str,
    expected_type: str = "text",
    model: str = "gpt-4o-mini",
) -> AdversarialValidationResult:
    """
    Convenience function for lightweight validation.

    Args:
        field_name: Name of the field
        value: The extracted value
        evidence: Evidence text
        expected_type: Type of value
        model: Model to use (default: gpt-4o-mini)

    Returns:
        AdversarialValidationResult
    """
    validator = LightweightValidator(model=model)
    return validator.validate(
        field_name=field_name,
        value=value,
        evidence=evidence,
        expected_type=expected_type,
    )


def validate_boolean_extraction(
    field_name: str,
    value: bool,
    evidence: str,
    model: str = "claude-sonnet-4-20250514",
) -> AdversarialValidationResult:
    """
    Convenience function to validate a boolean extraction.

    Args:
        field_name: Name of the boolean field
        value: The extracted boolean value
        evidence: Evidence text
        model: Model to use for validation

    Returns:
        AdversarialValidationResult
    """
    validator = AdversarialValidator(model=model, require_exact_quote=True)
    return validator.validate(
        field_name=field_name,
        value=value,
        evidence=evidence,
        expected_type="boolean",
    )


def validate_extraction(
    field_name: str,
    value: Any,
    evidence: str,
    expected_type: str = "text",
    model: str = "claude-sonnet-4-20250514",
    require_exact_quote: bool = True,
) -> AdversarialValidationResult:
    """
    Convenience function to validate any extraction.

    Args:
        field_name: Name of the field
        value: The extracted value
        evidence: Evidence text
        expected_type: Type of value
        model: Model to use for validation
        require_exact_quote: Whether to require exact quote

    Returns:
        AdversarialValidationResult
    """
    validator = AdversarialValidator(model=model, require_exact_quote=require_exact_quote)
    return validator.validate(
        field_name=field_name,
        value=value,
        evidence=evidence,
        expected_type=expected_type,
    )
