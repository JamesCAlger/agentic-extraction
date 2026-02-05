"""
Tier 3 + DocVQA Hybrid Extraction.

Combines:
1. Tier 3 keyword-based section/chunk retrieval (finds relevant context)
2. DocVQA conceptual questions (asks the right questions on that context)

This addresses both problems:
- DocVQA alone: Context window too small, misses relevant sections
- Tier 3 alone: Direct extraction prompts miss semantic distinctions
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Type

from openai import OpenAI
from anthropic import Anthropic
import instructor
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# Reuse keyword configs from scoped_agentic
from .scoped_agentic import FIELD_KEYWORDS, score_section_for_field

# Import extraction schemas for instructor validation
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


# DocVQA questions with synthesis prompts
DOCVQA_FIELD_CONFIG = {
    "incentive_fee": {
        "questions": [
            "Does THIS FUND (not underlying funds it invests in) charge a performance-based fee or incentive fee to its shareholders?",
            "If this fund charges an incentive fee, what percentage is it?",
            "Is there a hurdle rate that must be exceeded before this fund charges incentive fees?",
            "Does this fund have a high water mark, loss recovery account, or deficit carryforward mechanism?",
        ],
        "synthesis_prompt": """Based on the Q&A above, extract the FUND-LEVEL incentive fee structure.

CRITICAL DISTINCTION:
- FUND-LEVEL incentive fees: Fees THIS FUND charges to its investors (what we want)
- UNDERLYING FUND fees: Fees charged by funds this fund invests in (this is AFFE, NOT incentive_fee)

If the document only mentions underlying/investment fund fees (e.g., "Investment Funds charge 15-20%"),
then has_incentive_fee should be FALSE because THIS FUND doesn't charge an incentive fee.

Return JSON:
{
    "has_incentive_fee": true/false (only true if THIS FUND charges incentive fees),
    "incentive_fee_pct": number or null,
    "hurdle_rate_pct": number or null,
    "high_water_mark": true/false,
    "fee_basis": "net_profits" | "capital_gains" | "total_return" | null,
    "evidence": "exact quote supporting the answer"
}"""
    },

    "expense_cap": {
        "questions": [
            "Is there a contractual LIMIT or CAP on this fund's total expenses (not just actual expenses)?",
            "If there is an expense cap, what percentage is it?",
            "Is this expense cap voluntary or contractual?",
            "When does the expense cap expire?",
        ],
        "synthesis_prompt": """Based on the Q&A above, extract expense cap information.

CRITICAL DISTINCTION:
- EXPENSE CAP: A contractual LIMIT that caps expenses (e.g., "Adviser has agreed to cap expenses at 2.00%")
- ACTUAL EXPENSES: The expense ratio being charged (e.g., "Total Annual Expenses: 4.08%") - NOT a cap

If the document shows actual expenses but NO contractual cap/limit, has_expense_cap should be FALSE.

Return JSON:
{
    "has_expense_cap": true/false,
    "expense_cap_pct": number or null,
    "cap_type": "contractual" | "voluntary" | null,
    "cap_expiration": "date string" or null,
    "evidence": "exact quote supporting the answer"
}"""
    },

    "repurchase_terms": {
        "questions": [
            "How often does this fund offer to repurchase shares from investors (quarterly, semi-annually, annually)?",
            "What percentage of outstanding shares does the fund offer to repurchase?",
            "Is there an early repurchase/redemption fee? If so, what percentage and for what time period?",
            "What is the minimum amount for repurchase requests?",
        ],
        "synthesis_prompt": """Based on the Q&A above, extract repurchase terms.

Return JSON:
{
    "repurchase_frequency": "quarterly" | "semi-annual" | "annual" | "monthly" | null,
    "repurchase_amount_pct": number (e.g., 5 for 5%) or null,
    "repurchase_basis": "nav" | "net_assets" | "number_of_shares" | null,
    "early_repurchase_fee_pct": number or null,
    "early_repurchase_fee_period": "string" or null,
    "minimum_repurchase_amount": number or null,
    "evidence": "exact quote supporting the answer"
}"""
    },

    "share_classes": {
        "questions": [
            "What share classes does this fund offer (e.g., Class S, Class I, Class D)?",
            "For each share class, what is the minimum initial investment amount?",
            "For each share class, what is the sales load or front-end fee percentage?",
            "For each share class, what is the distribution/servicing fee (12b-1 fee) percentage?",
        ],
        "synthesis_prompt": """Based on the Q&A above, extract share class information.

Return JSON with a list of share classes:
{
    "share_classes": [
        {
            "class_name": "Class S",
            "minimum_initial_investment": number or null,
            "minimum_additional_investment": number or null,
            "sales_load_pct": number (0 if none),
            "distribution_servicing_fee_pct": number (0 if none)
        }
    ]
}"""
    },

    "leverage_limits": {
        "questions": [
            "Does this fund use leverage (borrowing)?",
            "What is the maximum leverage this fund can use?",
            "How is leverage expressed - as percentage of assets, asset coverage ratio, or debt-to-equity?",
        ],
        "synthesis_prompt": """Based on the Q&A above, extract leverage information.

Return JSON:
{
    "uses_leverage": true/false,
    "max_leverage_pct": number or null,
    "leverage_basis": "total_assets" | "net_assets" | "asset_coverage" | null,
    "evidence": "exact quote supporting the answer"
}"""
    },

    "distribution_terms": {
        "questions": [
            "How often does this fund pay distributions to shareholders?",
            "Are distributions paid in cash or automatically reinvested by default?",
        ],
        "synthesis_prompt": """Based on the Q&A above, extract distribution terms.

Return JSON:
{
    "distribution_frequency": "monthly" | "quarterly" | "annual" | null,
    "default_distribution_policy": "cash" | "DRIP" | "shareholder_choice" | null,
    "evidence": "exact quote supporting the answer"
}"""
    },

    "allocation_targets": {
        "questions": [
            "What percentage of assets does this fund target for private equity/private assets?",
            "What is the target allocation range for secondary investments?",
            "What is the target allocation range for direct/co-investments?",
        ],
        "synthesis_prompt": """Based on the Q&A above, extract allocation targets.

Return JSON:
{
    "private_assets_target_pct": number or null,
    "secondary_funds_min_pct": number or null,
    "secondary_funds_max_pct": number or null,
    "direct_investments_min_pct": number or null,
    "direct_investments_max_pct": number or null,
    "approach": "fund_of_funds" | "direct" | "hybrid" | null
}"""
    },

    "concentration_limits": {
        "questions": [
            "What is the maximum percentage this fund can invest in a single issuer?",
            "What is the maximum percentage this fund can invest in a single underlying fund?",
            "What is the maximum percentage this fund can invest in a single sector?",
        ],
        "synthesis_prompt": """Based on the Q&A above, extract concentration limits.
Only include explicitly stated limits.

Return JSON:
{
    "max_single_issuer_pct": number or null,
    "max_single_fund_pct": number or null,
    "max_single_sector_pct": number or null,
    "max_single_asset_pct": number or null
}"""
    },
}


@dataclass
class Tier3DocVQAResult:
    """Result from Tier 3 + DocVQA hybrid extraction."""
    field_name: str
    sections_searched: int
    chunks_retrieved: int
    questions_asked: int
    grounded_answers: int
    extracted_value: Any
    evidence: Optional[str] = None
    confidence: str = "not_found"


class Tier3DocVQAExtractor:
    """
    Hybrid extractor combining Tier 3 retrieval with DocVQA questions.

    Process:
    1. Score all sections by field-specific keywords (Tier 3)
    2. Get top K sections and their chunks
    3. Ask DocVQA conceptual questions on those chunks
    4. Synthesize answers into structured extraction
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        top_k_sections: int = 5,
        max_chunks_per_section: int = 10,
        delay_between_calls: float = 1.0,
        use_instructor: bool = True,
    ):
        self.model = model
        self.provider = provider
        self.top_k = top_k_sections
        self.max_chunks = max_chunks_per_section
        self.delay = delay_between_calls
        self.use_instructor = use_instructor

        # Create raw clients for Q&A
        if provider == "anthropic":
            self.client = Anthropic(api_key=api_key)
            # Instructor-wrapped client for schema-validated synthesis
            self.instructor_client = instructor.from_anthropic(
                Anthropic(api_key=api_key)
            ) if use_instructor else None
        else:
            self.client = OpenAI(api_key=api_key)
            # Instructor-wrapped client for schema-validated synthesis
            self.instructor_client = instructor.from_openai(
                OpenAI(api_key=api_key)
            ) if use_instructor else None

    def _get_relevant_chunks(
        self,
        field_name: str,
        chunked_doc,
    ) -> tuple[list[str], int]:
        """
        Use Tier 3 keyword scoring to find relevant chunks.

        Returns:
            Tuple of (list of chunk texts, number of sections searched)
        """
        if field_name not in FIELD_KEYWORDS:
            logger.warning(f"No keywords defined for field: {field_name}")
            return [], 0

        # Score all sections
        section_scores = []
        for section in chunked_doc.chunked_sections:
            # score_section_for_field expects the full section object
            scored = score_section_for_field(section, field_name)
            if scored.score > 0:
                section_scores.append((section, scored.score))

        # Sort by score descending
        section_scores.sort(key=lambda x: x[1], reverse=True)

        # Get top K sections
        top_sections = section_scores[:self.top_k]

        if not top_sections:
            logger.info(f"    No sections found for {field_name}")
            return [], 0

        logger.info(f"    Top {len(top_sections)} sections by keyword score:")
        for section, score in top_sections[:3]:
            title = section.section_title[:50] if section.section_title else "Untitled"
            logger.info(f"      - {title}... (score: {score})")

        # Collect chunks from top sections
        chunks = []
        for section, score in top_sections:
            section_chunks = section.chunks[:self.max_chunks]
            for chunk in section_chunks:
                chunks.append(chunk.content)

        logger.info(f"    Retrieved {len(chunks)} chunks from {len(top_sections)} sections")
        return chunks, len(top_sections)

    def _ask_question(self, question: str, context: str) -> tuple[str, str, bool]:
        """
        Ask a single question on the given context.

        Returns:
            Tuple of (answer, evidence, is_grounded)
        """
        prompt = f"""Answer this question based ONLY on the provided text.
If the answer is not in the text, say "Not stated in the document."
Quote the relevant text that supports your answer.

TEXT:
{context}

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
                    if evidence.lower() in ["none", "not found", "n/a", '"none"']:
                        evidence = None
                else:
                    answer = parts.strip()
            else:
                answer = response_text.strip()

            is_grounded = evidence is not None and "not stated" not in answer.lower()

            time.sleep(self.delay)
            return answer, evidence or "", is_grounded

        except Exception as e:
            logger.error(f"Error asking question: {e}")
            return f"Error: {e}", "", False

    def _synthesize_answers(
        self,
        field_name: str,
        qa_results: list[tuple[str, str, str, bool]],  # question, answer, evidence, grounded
    ) -> dict:
        """
        Synthesize Q&A results into structured extraction.

        Uses instructor with Pydantic schema validation when available,
        with fallback to raw JSON parsing for fields without schemas.
        """
        config = DOCVQA_FIELD_CONFIG.get(field_name, {})
        synthesis_prompt = config.get("synthesis_prompt", "Extract the relevant information as JSON.")

        # Format Q&A for synthesis
        qa_text = "\n\n".join([
            f"Q: {q}\nA: {a}\nEvidence: {e if e else 'None'}\nGrounded: {g}"
            for q, a, e, g in qa_results
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
                return self._synthesize_with_instructor(field_name, schema_class, base_prompt)
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
        logger.debug(f"  [Tier3+DocVQA] Using instructor synthesis with {schema_class.__name__}")

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

        # Convert Pydantic model to dict
        result_dict = result.model_dump(exclude_none=False)

        # Remove chain-of-thought fields that aren't needed in final output
        result_dict.pop("reasoning", None)
        result_dict.pop("confidence", None)
        result_dict.pop("citation", None)

        logger.debug(f"  [Tier3+DocVQA] Instructor synthesis successful: {list(result_dict.keys())}")
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
                        {"role": "system", "content": "Synthesize Q&A results into structured JSON."},
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
        chunked_doc,
    ) -> Tier3DocVQAResult:
        """
        Extract a single field using Tier 3 retrieval + DocVQA questions.
        """
        config = DOCVQA_FIELD_CONFIG.get(field_name)
        if not config:
            logger.warning(f"No DocVQA config for field: {field_name}")
            return Tier3DocVQAResult(
                field_name=field_name,
                sections_searched=0,
                chunks_retrieved=0,
                questions_asked=0,
                grounded_answers=0,
                extracted_value=None,
                confidence="not_found"
            )

        # Step 1: Get relevant chunks via Tier 3 keyword search
        logger.info(f"  [Tier3+DocVQA] Retrieving chunks for {field_name}")
        chunks, sections_searched = self._get_relevant_chunks(field_name, chunked_doc)

        if not chunks:
            return Tier3DocVQAResult(
                field_name=field_name,
                sections_searched=sections_searched,
                chunks_retrieved=0,
                questions_asked=0,
                grounded_answers=0,
                extracted_value=None,
                confidence="not_found"
            )

        # Combine chunks into context (limit to ~30K chars)
        context = "\n\n---\n\n".join(chunks)[:30000]

        # Step 2: Ask DocVQA questions
        questions = config["questions"]
        logger.info(f"  [Tier3+DocVQA] Asking {len(questions)} questions on {len(chunks)} chunks")

        qa_results = []
        grounded_count = 0
        for question in questions:
            answer, evidence, is_grounded = self._ask_question(question, context)
            qa_results.append((question, answer, evidence, is_grounded))
            if is_grounded:
                grounded_count += 1

        logger.info(f"  [Tier3+DocVQA] {grounded_count}/{len(questions)} answers grounded")

        # Step 3: Synthesize answers
        extracted = self._synthesize_answers(field_name, qa_results)

        # Get evidence from synthesis if available
        evidence = extracted.pop("evidence", None) if isinstance(extracted, dict) else None

        # Determine confidence
        if grounded_count == 0:
            confidence = "not_found"
        elif grounded_count < len(questions) / 2:
            confidence = "inferred"
        else:
            confidence = "explicit"

        return Tier3DocVQAResult(
            field_name=field_name,
            sections_searched=sections_searched,
            chunks_retrieved=len(chunks),
            questions_asked=len(questions),
            grounded_answers=grounded_count,
            extracted_value=extracted,
            evidence=evidence,
            confidence=confidence
        )

    def extract_all_fields(
        self,
        chunked_doc,
        fields: Optional[list[str]] = None,
    ) -> dict[str, Tier3DocVQAResult]:
        """
        Extract all fields using Tier 3 + DocVQA hybrid approach.
        """
        if fields is None:
            fields = list(DOCVQA_FIELD_CONFIG.keys())

        results = {}
        for field_name in fields:
            logger.info(f"[Tier3+DocVQA] Extracting {field_name}")
            results[field_name] = self.extract_field(field_name, chunked_doc)

        return results


def convert_tier3_docvqa_to_extraction_format(results: dict[str, Tier3DocVQAResult]) -> dict:
    """Convert Tier3+DocVQA results to standard extraction format."""
    extraction = {}
    for field_name, result in results.items():
        if result.extracted_value:
            extraction[field_name] = result.extracted_value
        else:
            extraction[field_name] = None
    return extraction
