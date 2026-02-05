"""
DocVQA (Document Visual Question Answering) extraction approach.

Instead of "extract field X", asks conceptual questions like:
- "Does this fund charge performance fees at the fund level?"
- "Is there a contractual limit on expenses?"

This catches semantic equivalents that direct extraction misses:
- "Loss Recovery Account" = high water mark
- "Total Annual Expenses 4.08%" ≠ expense cap
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Type

from openai import OpenAI
from anthropic import Anthropic
import instructor
from pydantic import BaseModel

# Import extraction schemas for structured validation
from .schemas import (
    IncentiveFeeExtraction,
    ExpenseCapExtraction,
    RepurchaseTermsExtraction,
    LeverageLimitsExtraction,
    DistributionTermsExtraction,
    ShareClassesExtraction,
    AllocationTargetsExtraction,
    ConcentrationLimitsExtraction,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMA MAPPING FOR INSTRUCTOR VALIDATION
# =============================================================================

# Map field names to their Pydantic schema classes
FIELD_SCHEMA_MAP: dict[str, Type[BaseModel]] = {
    "incentive_fee": IncentiveFeeExtraction,
    "expense_cap": ExpenseCapExtraction,
    "repurchase_terms": RepurchaseTermsExtraction,
    "leverage_limits": LeverageLimitsExtraction,
    "distribution_terms": DistributionTermsExtraction,
    "share_classes": ShareClassesExtraction,
    "allocation_targets": AllocationTargetsExtraction,
    "concentration_limits": ConcentrationLimitsExtraction,
}


@dataclass
class QuestionAnswer:
    """Single Q&A result."""
    question: str
    answer: str
    grounded: bool = False
    evidence: Optional[str] = None


@dataclass
class DocVQAResult:
    """Result of DocVQA extraction for a field."""
    field_name: str
    questions_asked: list[QuestionAnswer]
    synthesized_value: Any
    confidence: str  # "explicit", "inferred", "not_found"
    reasoning: str
    grounded: bool = False


# Conceptual questions for each field
# These ask about CONCEPTS, not field names
FIELD_QUESTIONS = {
    "incentive_fee": {
        "questions": [
            "Does this fund charge a performance-based fee or incentive fee at the FUND level (not underlying funds)?",
            "If yes, what percentage is the incentive/performance fee?",
            "Is there a hurdle rate that must be exceeded before incentive fees are charged?",
            "Is there any mechanism to prevent charging fees on recovered losses (high water mark, loss recovery account, deficit carryforward)?",
            "How often are incentive fees calculated and paid (crystallization frequency)?",
        ],
        "synthesis_prompt": """Based on the Q&A above, extract the fund-level incentive fee structure.

IMPORTANT: Distinguish between:
- FUND-LEVEL incentive fees (charged by this fund to investors)
- UNDERLYING FUND fees (charged by funds this fund invests in - these are AFFE, not incentive_fee)

If the document only mentions underlying fund fees (e.g., "Investment Funds charge 15-20%"),
then has_incentive_fee should be FALSE for this fund.

Return:
- has_incentive_fee: true only if THIS FUND charges incentive fees
- incentive_fee_pct: the percentage (null if no fund-level fee)
- hurdle_rate_pct: hurdle rate if mentioned
- high_water_mark: true if there's any loss recovery mechanism
- fee_basis: "net_profits", "capital_gains", or "total_return"
- crystallization_frequency: "quarterly", "annual", etc.
"""
    },

    "expense_cap": {
        "questions": [
            "Is there a contractual LIMIT or CAP on the fund's total expenses?",
            "If yes, what percentage is the expense cap/limit?",
            "Is this cap voluntary or contractual?",
            "When does the expense cap expire (if applicable)?",
        ],
        "synthesis_prompt": """Based on the Q&A above, extract the expense cap information.

IMPORTANT: Distinguish between:
- EXPENSE CAP: A contractual limit that caps expenses (e.g., "expenses capped at 2.00%")
- TOTAL EXPENSES: The actual expense ratio (e.g., "Total Annual Expenses 4.08%") - this is NOT a cap

If the document shows total expenses but no cap/limit, has_expense_cap should be FALSE.

Return:
- has_expense_cap: true only if there's a contractual limit
- expense_cap_pct: the cap percentage (null if no cap)
- cap_type: "contractual" or "voluntary"
- cap_expiration: expiration date if mentioned
"""
    },

    "repurchase_terms": {
        "questions": [
            "How often does the fund offer to repurchase shares (quarterly, semi-annually, annually)?",
            "What percentage of shares does the fund offer to repurchase?",
            "Is the repurchase based on NAV, net assets, or number of shares?",
            "Is there an early repurchase fee or penalty for redeeming within a certain period?",
            "If yes, what is the early repurchase fee percentage and time period?",
            "Is there a minimum repurchase amount?",
        ],
        "synthesis_prompt": """Based on the Q&A above, extract repurchase terms.

Return:
- repurchase_frequency: "quarterly", "semi-annual", "annual", or "monthly"
- repurchase_amount_pct: percentage offered (e.g., "5" for 5%)
- repurchase_basis: "nav", "net_assets", or "number_of_shares"
- early_repurchase_fee_pct: fee percentage if applicable
- early_repurchase_fee_period: period (e.g., "within 1 year")
- minimum_repurchase_amount: dollar amount if specified
"""
    },

    "leverage_limits": {
        "questions": [
            "Does this fund use leverage (borrowing)?",
            "What is the maximum leverage the fund can use?",
            "Is leverage expressed as a percentage of assets, asset coverage ratio, or debt-to-equity?",
            "What is the size of any credit facility?",
        ],
        "synthesis_prompt": """Based on the Q&A above, extract leverage limits.

Return:
- uses_leverage: true/false
- max_leverage_pct: maximum leverage percentage
- leverage_basis: "total_assets", "net_assets", "asset_coverage", or "debt_to_equity"
- credit_facility_size: dollar amount if mentioned
"""
    },

    "distribution_terms": {
        "questions": [
            "How often does the fund pay distributions (monthly, quarterly, annually)?",
            "What is the default distribution policy - cash or reinvested (DRIP)?",
            "Can shareholders choose between cash and reinvestment?",
        ],
        "synthesis_prompt": """Based on the Q&A above, extract distribution terms.

Return:
- distribution_frequency: "monthly", "quarterly", "annual", or null
- default_distribution_policy: "cash", "DRIP", or "shareholder_choice"
"""
    },

    "share_classes": {
        "questions": [
            "What share classes does this fund offer (e.g., Class S, Class I, Class D)?",
            "For each share class, what is the minimum initial investment?",
            "For each share class, what is the minimum additional/subsequent investment?",
            "For each share class, what is the sales load or front-end fee?",
            "For each share class, what is the distribution/servicing fee?",
        ],
        "synthesis_prompt": """Based on the Q&A above, extract share class information.

Return a list of share classes, each with:
- class_name: e.g., "Class S", "Class I"
- minimum_initial_investment: dollar amount
- minimum_additional_investment: dollar amount (null if not stated)
- sales_load_pct: percentage (0 if none)
- distribution_servicing_fee_pct: percentage (0 if none)
"""
    },

    "allocation_targets": {
        "questions": [
            "What percentage of assets does the fund target for private equity/private assets?",
            "What percentage range is allocated to secondary investments?",
            "What percentage range is allocated to direct/co-investments?",
            "What is the fund's investment approach - fund-of-funds, direct, or hybrid?",
        ],
        "synthesis_prompt": """Based on the Q&A above, extract allocation targets.

Return:
- private_assets_target_pct: target percentage for private assets
- secondary_funds_min_pct / secondary_funds_max_pct: range for secondaries
- direct_investments_min_pct / direct_investments_max_pct: range for directs
- approach: "fund_of_funds", "direct", or "hybrid"
"""
    },

    "concentration_limits": {
        "questions": [
            "What is the maximum percentage the fund can invest in a single issuer/company?",
            "What is the maximum percentage the fund can invest in a single underlying fund?",
            "What is the maximum percentage the fund can invest in a single sector/industry?",
            "What is the maximum percentage the fund can invest in a single asset?",
        ],
        "synthesis_prompt": """Based on the Q&A above, extract concentration limits.

IMPORTANT: Only extract explicitly stated limits. If not mentioned, return null.

Return:
- max_single_issuer_pct: percentage limit
- max_single_fund_pct: percentage limit
- max_single_sector_pct: percentage limit
- max_single_asset_pct: percentage limit
"""
    },
}


class DocVQAExtractor:
    """
    Extracts fields by asking conceptual questions.

    Process:
    1. For each field, ask 4-5 conceptual questions
    2. Ground-check each answer against source text
    3. Synthesize grounded answers into structured extraction
    4. Verify final extraction is grounded
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        delay_between_calls: float = 1.0,
        use_instructor: bool = True,
    ):
        self.model = model
        self.provider = provider
        self.delay = delay_between_calls
        self.api_key = api_key
        self.use_instructor = use_instructor

        # Create raw clients for Q&A (not structured extraction)
        if provider == "anthropic":
            self.client = Anthropic(api_key=api_key)
            # Instructor-wrapped client for synthesis with schema validation
            self.instructor_client = instructor.from_anthropic(
                Anthropic(api_key=api_key)
            ) if use_instructor else None
        else:
            self.client = OpenAI(api_key=api_key)
            # Instructor-wrapped client for synthesis with schema validation
            self.instructor_client = instructor.from_openai(
                OpenAI(api_key=api_key)
            ) if use_instructor else None

    def _ask_question(self, question: str, context: str) -> QuestionAnswer:
        """Ask a single question about the document."""
        prompt = f"""Answer this question based ONLY on the provided text.
If the answer is not in the text, say "Not stated in the document."
Quote the relevant text that supports your answer.

TEXT:
{context[:15000]}

QUESTION: {question}

Answer format:
ANSWER: [your answer]
EVIDENCE: [exact quote from text, or "None" if not found]"""

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = response.content[0].text
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Answer questions based only on the provided text. Always quote evidence."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                )
                response_text = response.choices[0].message.content

            # Parse response
            answer = ""
            evidence = None

            if "ANSWER:" in response_text:
                parts = response_text.split("ANSWER:", 1)[1]
                if "EVIDENCE:" in parts:
                    answer_part, evidence_part = parts.split("EVIDENCE:", 1)
                    answer = answer_part.strip()
                    evidence = evidence_part.strip()
                    if evidence.lower() in ["none", "not found", "n/a"]:
                        evidence = None
                else:
                    answer = parts.strip()
            else:
                answer = response_text.strip()

            # Check if grounded
            grounded = evidence is not None and "not stated" not in answer.lower()

            time.sleep(self.delay)

            return QuestionAnswer(
                question=question,
                answer=answer,
                grounded=grounded,
                evidence=evidence
            )

        except Exception as e:
            logger.error(f"Error asking question: {e}")
            return QuestionAnswer(
                question=question,
                answer=f"Error: {e}",
                grounded=False
            )

    def _synthesize_answers(
        self,
        field_name: str,
        qa_results: list[QuestionAnswer],
        context: str
    ) -> dict:
        """
        Synthesize Q&A results into structured extraction.

        Uses instructor with Pydantic schema validation when available,
        with fallback to raw JSON parsing for fields without schemas.
        """
        field_config = FIELD_QUESTIONS.get(field_name, {})
        synthesis_prompt = field_config.get("synthesis_prompt", "Extract the relevant information.")

        # Format Q&A for synthesis
        qa_text = "\n\n".join([
            f"Q: {qa.question}\nA: {qa.answer}\nEvidence: {qa.evidence or 'None'}\nGrounded: {qa.grounded}"
            for qa in qa_results
        ])

        base_prompt = f"""Based on these Q&A results about a fund document:

{qa_text}

{synthesis_prompt}

IMPORTANT:
- Use null for values not found or not applicable
- Only use information from the Q&A results above - do not infer beyond what was stated
- For percentages, use numeric values without % suffix (e.g., 3.5 not "3.5%")
- For dollar amounts, use numeric values without $ or commas (e.g., 5000 not "$5,000")
"""

        # Try instructor-based structured extraction first
        schema_class = FIELD_SCHEMA_MAP.get(field_name)

        if self.use_instructor and self.instructor_client and schema_class:
            try:
                return self._synthesize_with_instructor(
                    field_name, schema_class, base_prompt
                )
            except Exception as e:
                logger.warning(
                    f"Instructor synthesis failed for {field_name}, falling back to raw JSON: {e}"
                )

        # Fallback to raw JSON parsing
        return self._synthesize_raw_json(base_prompt)

    def _synthesize_with_instructor(
        self,
        field_name: str,
        schema_class: Type[BaseModel],
        prompt: str,
    ) -> dict:
        """
        Synthesize using instructor for schema-validated extraction.

        Benefits:
        - Automatic retry on validation failure
        - Type coercion (strings to numbers, etc.)
        - Consistent field names matching ground truth
        """
        logger.debug(f"  [DocVQA] Using instructor synthesis with {schema_class.__name__}")

        if self.provider == "anthropic":
            result = self.instructor_client.messages.create(
                model=self.model,
                max_tokens=2000,
                response_model=schema_class,
                max_retries=2,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            result = self.instructor_client.chat.completions.create(
                model=self.model,
                response_model=schema_class,
                max_retries=2,
                messages=[
                    {
                        "role": "system",
                        "content": "Synthesize Q&A results into the requested structured format."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
            )

        # Convert Pydantic model to dict, excluding None values and internal fields
        result_dict = result.model_dump(exclude_none=False)

        # Remove chain-of-thought fields that aren't needed in final output
        result_dict.pop("reasoning", None)
        result_dict.pop("confidence", None)
        result_dict.pop("citation", None)

        logger.debug(f"  [DocVQA] Instructor synthesis successful: {list(result_dict.keys())}")
        return result_dict

    def _synthesize_raw_json(self, prompt: str) -> dict:
        """
        Fallback synthesis using raw JSON response.

        Used when:
        - No schema defined for field
        - Instructor fails
        - use_instructor=False
        """
        import json

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt + "\n\nReturn as JSON."}],
                )
                response_text = response.content[0].text
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Synthesize Q&A results into structured JSON."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                response_text = response.choices[0].message.content

            # Handle markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            return json.loads(response_text)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse synthesis response as JSON: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error in raw JSON synthesis: {e}")
            return {}

    def extract_field(
        self,
        field_name: str,
        context: str,
    ) -> DocVQAResult:
        """
        Extract a single field using DocVQA approach.

        Args:
            field_name: Field to extract (e.g., "incentive_fee")
            context: Document text to search

        Returns:
            DocVQAResult with extracted value and reasoning
        """
        field_config = FIELD_QUESTIONS.get(field_name)

        if not field_config:
            logger.warning(f"No DocVQA questions defined for field: {field_name}")
            return DocVQAResult(
                field_name=field_name,
                questions_asked=[],
                synthesized_value=None,
                confidence="not_found",
                reasoning="No DocVQA questions defined for this field"
            )

        questions = field_config["questions"]

        # Ask each question
        logger.info(f"  [DocVQA] Asking {len(questions)} questions for {field_name}")
        qa_results = []
        for q in questions:
            qa = self._ask_question(q, context)
            qa_results.append(qa)
            logger.debug(f"    Q: {q[:50]}... -> Grounded: {qa.grounded}")

        # Count grounded answers
        grounded_count = sum(1 for qa in qa_results if qa.grounded)
        logger.info(f"  [DocVQA] {grounded_count}/{len(qa_results)} answers grounded")

        # Synthesize answers
        synthesized = self._synthesize_answers(field_name, qa_results, context)

        # Determine confidence
        if grounded_count == 0:
            confidence = "not_found"
        elif grounded_count < len(qa_results) / 2:
            confidence = "inferred"
        else:
            confidence = "explicit"

        # Build reasoning
        reasoning_parts = []
        for qa in qa_results:
            status = "grounded" if qa.grounded else "ungrounded"
            reasoning_parts.append(f"- {qa.question[:60]}... [{status}]")

        return DocVQAResult(
            field_name=field_name,
            questions_asked=qa_results,
            synthesized_value=synthesized,
            confidence=confidence,
            reasoning="\n".join(reasoning_parts),
            grounded=grounded_count > 0
        )

    def extract_all_fields(
        self,
        context: str,
        fields: Optional[list[str]] = None
    ) -> dict[str, DocVQAResult]:
        """
        Extract all fields using DocVQA approach.

        Args:
            context: Document text
            fields: List of fields to extract (defaults to all)

        Returns:
            Dict mapping field_name -> DocVQAResult
        """
        if fields is None:
            fields = list(FIELD_QUESTIONS.keys())

        results = {}
        for field_name in fields:
            logger.info(f"[DocVQA] Extracting {field_name}")
            results[field_name] = self.extract_field(field_name, context)

        return results


def convert_docvqa_to_extraction_format(docvqa_results: dict[str, DocVQAResult]) -> dict:
    """
    Convert DocVQA results to the standard extraction format.

    This allows comparison with T1-T3 and T3-only extractions.
    """
    extraction = {}

    for field_name, result in docvqa_results.items():
        if result.synthesized_value:
            extraction[field_name] = result.synthesized_value
        else:
            extraction[field_name] = None

    return extraction
