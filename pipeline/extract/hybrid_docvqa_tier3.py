"""
Hybrid DocVQA + Tier3 Extraction.

Combines the best of both approaches:
1. Tier3 keyword scoring for chunk selection (always)
2. DocVQA existence questions (better at yes/no detection)
3. Tier3 direct extraction for details (better at numeric/structured extraction)

Flow:
  Tier3 chunks → DocVQA "Does X exist?" → If YES → Tier3 "Extract X details"
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Type

from openai import OpenAI
from anthropic import Anthropic
import instructor
from pydantic import BaseModel

from .scoped_agentic import score_section_for_field
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


# Schema mapping for Tier3 direct extraction
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


# DocVQA existence questions (simple yes/no)
EXISTENCE_QUESTIONS = {
    "incentive_fee": """Does THIS FUND (not underlying funds it invests in) charge a performance-based fee or incentive fee to its shareholders?

IMPORTANT: Only consider FUND-LEVEL fees charged by THIS fund.
Ignore fees charged by underlying/investment funds (those are AFFE, not incentive fees).

Respond ONLY with:
EXISTS: YES or NO
EVIDENCE: [brief quote if YES, or "Not found" if NO]""",

    "expense_cap": """Does this fund have a contractual EXPENSE CAP or LIMIT on total expenses?

IMPORTANT: Distinguish between:
- EXPENSE CAP: A contractual limit (e.g., "Adviser agrees to cap expenses at 2.00%")
- ACTUAL EXPENSES: The expense ratio being charged (e.g., "Total Expenses: 4.08%") - NOT a cap

Respond ONLY with:
EXISTS: YES or NO
EVIDENCE: [brief quote if YES, or "Not found" if NO]""",

    "repurchase_terms": """Does this fund offer periodic share repurchases or redemptions?

Look for repurchase offers, tender offers, or redemption terms.

Respond ONLY with:
EXISTS: YES or NO
EVIDENCE: [brief quote if YES, or "Not found" if NO]""",

    "leverage_limits": """Does this fund use leverage (borrowing)?

Look for credit facilities, borrowing limits, or leverage policies.

Respond ONLY with:
EXISTS: YES or NO
EVIDENCE: [brief quote if YES, or "Not found" if NO]""",

    "distribution_terms": """Does this fund pay distributions or dividends to shareholders?

Respond ONLY with:
EXISTS: YES or NO
EVIDENCE: [brief quote if YES, or "Not found" if NO]""",

    "share_classes": """Does this fund offer multiple share classes (e.g., Class A, Class I, Class S)?

Respond ONLY with:
EXISTS: YES or NO
EVIDENCE: [list class names if YES, or "Not found" if NO]""",

    "allocation_targets": """Does this fund have specific allocation targets or ranges for investment types?

Look for target percentages for secondaries, co-investments, primaries, etc.

Respond ONLY with:
EXISTS: YES or NO
EVIDENCE: [brief quote if YES, or "Not found" if NO]""",

    "concentration_limits": """Does this fund have concentration limits or diversification requirements?

Look for maximum percentages for single issuer, sector, fund, etc.

Respond ONLY with:
EXISTS: YES or NO
EVIDENCE: [brief quote if YES, or "Not found" if NO]""",
}


# Tier3 direct extraction prompts (for when existence=YES)
TIER3_EXTRACTION_PROMPTS = {
    "incentive_fee": """Extract the FUND-LEVEL incentive fee structure from this text.

IMPORTANT: Only extract fees THIS FUND charges to its investors.
Do NOT include fees charged by underlying funds (that's AFFE).

=== SEMANTIC TAXONOMY ===

HIGH WATER MARK (set high_water_mark = true for ALL of these):
- Explicit "high water mark" or "HWM"
- "Loss Recovery Account" - this IS a high water mark mechanism
- "Loss Carryforward Account" - same as Loss Recovery Account
- Any mechanism preventing fees on recovered losses

CRITICAL: "Loss Recovery Account" = high_water_mark = true
Example: "10% of net profits OVER the Loss Recovery Account balance"
→ high_water_mark = true

CATCH-UP (set has_catch_up = true):
- Explicit "catch-up" or "full catch-up" language
- "100% of returns between X% and Y%"

DISTINCTION: Loss Recovery Account is NOT catch-up. They are different mechanisms.

HURDLE RATE: If stated as quarterly (e.g., "1.5% per quarter"), multiply by 4 for annualized.

Extract as JSON:
{
    "has_incentive_fee": true,
    "incentive_fee_pct": <number or null>,
    "hurdle_rate_pct": <ANNUALIZED number or null>,
    "high_water_mark": <true if HWM/Loss Recovery exists, false otherwise, null if no fee>,
    "has_catch_up": <true/false or null>,
    "fee_basis": <"net_profits" | "net_investment_income" | "total_return" or null>,
    "crystallization_frequency": <"quarterly" | "annual" | "monthly" or null>
}""",

    "expense_cap": """Extract expense cap information from this text.

Extract as JSON:
{
    "has_expense_cap": true,
    "expense_cap_pct": <number or null>,
    "cap_type": <"contractual" | "voluntary" or null>,
    "cap_expiration": <date string or null>
}""",

    "repurchase_terms": """Extract repurchase/redemption terms from this text.

Extract as JSON:
{
    "repurchase_frequency": <"quarterly" | "semi_annual" | "annual" | "monthly" | "discretionary" or null>,
    "repurchase_amount_pct": <number or null>,
    "repurchase_basis": <"nav" | "net_assets" | "outstanding_shares" or null>,
    "early_repurchase_fee_pct": <number or null>,
    "lock_up_period_years": <number or null>
}""",

    "leverage_limits": """Extract leverage/borrowing information from this text.

=== LEVERAGE FORMAT NORMALIZATION ===

CRITICAL: Return max_leverage_pct as the NORMALIZED percentage of assets that can be borrowed.

CONVERSION RULES:
1. "300% asset coverage" → max_leverage_pct = 33.33 (NOT 300)
   Formula: 100 / (coverage/100) = leverage%
2. "50% debt-to-equity" → max_leverage_pct = 33.33
   Formula: D/E / (1 + D/E) * 100
3. "33% of assets" → max_leverage_pct = 33 (no conversion)
4. "1940 Act limits" → max_leverage_pct = 33.33

Common conversions:
- 300% coverage = 33.33%
- 200% coverage = 50%
- 50% D/E = 33.33%

Extract as JSON:
{
    "uses_leverage": true,
    "max_leverage_pct": <NORMALIZED number (33, not 300) or null>,
    "leverage_basis": <"asset_coverage" | "total_assets" | "net_assets" | "debt_to_equity" or null>,
    "credit_facility_size": <number in dollars or null>
}""",

    "distribution_terms": """Extract distribution/dividend terms from this text.

CRITICAL DISTINCTION:
- DISTRIBUTION FREQUENCY = How often DIVIDENDS are paid (what we want)
- REPURCHASE FREQUENCY = How often shares can be redeemed (NOT this field)

Look for: "pay distributions monthly", "declare dividends quarterly"
IGNORE: "quarterly repurchase offers" (that's repurchase, not distribution)

If only repurchase frequency is mentioned and NO distribution frequency, return null.

Extract as JSON:
{
    "distribution_frequency": <"monthly" | "quarterly" | "annual" or null>,
    "default_distribution_policy": <"cash" | "DRIP" | "reinvested" or null>
}""",

    "share_classes": """Extract share class information from this text.

For EACH share class, extract:
{
    "share_classes": [
        {
            "class_name": <string>,
            "minimum_initial_investment": <number or null>,
            "minimum_additional_investment": <number or null>,
            "sales_load_pct": <number, 0 if none>,
            "distribution_servicing_fee_pct": <number, 0 if none>
        }
    ]
}""",

    "allocation_targets": """Extract allocation targets from this text.

Extract as JSON:
{
    "has_allocation_targets": true,
    "allocations": [
        {
            "allocation_type": <"secondary" | "co_investment" | "direct" | "primary">,
            "min_pct": <number or null>,
            "max_pct": <number or null>
        }
    ]
}""",

    "concentration_limits": """Extract concentration limits from this text.

Extract as JSON:
{
    "has_concentration_limits": true,
    "limits": [
        {
            "limit_type": <"single_issuer" | "single_fund" | "single_sector" | "single_asset">,
            "max_pct": <number>
        }
    ]
}""",
}


@dataclass
class HybridExtractionResult:
    """Result from hybrid extraction."""
    field_name: str
    sections_searched: int
    chunks_retrieved: int
    existence_result: bool  # From DocVQA
    existence_evidence: Optional[str]
    extraction_result: Optional[dict]  # From Tier3
    extraction_method: str  # "docvqa_existence" or "tier3_details"
    llm_calls: int


class HybridDocVQATier3Extractor:
    """
    Hybrid extractor combining DocVQA existence checks with Tier3 detail extraction.

    Flow:
    1. Tier3 keyword scoring → select relevant chunks
    2. DocVQA existence question → "Does X exist?"
    3. If YES → Tier3 direct extraction → structured JSON
    4. If NO → return nulls (skip detail extraction)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        top_k_sections: int = 5,
        max_chunks_per_section: int = 10,
        delay_between_calls: float = 1.0,
    ):
        self.model = model
        self.provider = provider
        self.top_k_sections = top_k_sections
        self.max_chunks_per_section = max_chunks_per_section
        self.delay = delay_between_calls
        self.api_key = api_key

        # Create clients
        if provider == "anthropic":
            self.client = Anthropic(api_key=api_key)
            self.instructor_client = instructor.from_anthropic(
                Anthropic(api_key=api_key)
            )
        else:
            self.client = OpenAI(api_key=api_key)
            self.instructor_client = instructor.from_openai(
                OpenAI(api_key=api_key)
            )

    def _get_relevant_chunks(
        self,
        field_name: str,
        chunked_doc,
    ) -> tuple[str, int, int]:
        """Use Tier3 keyword scoring to find relevant chunks."""
        section_scores = []
        for section in chunked_doc.chunked_sections:
            scored = score_section_for_field(section, field_name)
            if scored.score > 0:
                section_scores.append((scored.score, section))

        section_scores.sort(key=lambda x: x[0], reverse=True)
        top_sections = section_scores[:self.top_k_sections]

        if top_sections:
            logger.info(f"    Top sections: {[s.section_title[:30] for _, s in top_sections[:3]]}")

        chunks = []
        for score, section in top_sections:
            section_chunks = section.chunks[:self.max_chunks_per_section]
            chunks.extend(section_chunks)

        combined_text = "\n\n---\n\n".join(
            chunk.content for chunk in chunks
        )

        return combined_text, len(top_sections), len(chunks)

    def _ask_existence(self, field_name: str, context: str) -> tuple[bool, Optional[str]]:
        """DocVQA existence check - returns (exists, evidence)."""
        question = EXISTENCE_QUESTIONS.get(field_name)
        if not question:
            return False, None

        prompt = f"""Based ONLY on this text, answer the question.

TEXT:
{context[:25000]}

QUESTION:
{question}"""

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = response.content[0].text
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Answer concisely based only on the provided text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=500,
                )
                answer = response.choices[0].message.content

            time.sleep(self.delay)

            # Parse response
            exists = "EXISTS: YES" in answer.upper() or "EXISTS:YES" in answer.upper()

            evidence = None
            if "EVIDENCE:" in answer:
                evidence = answer.split("EVIDENCE:", 1)[1].strip().split("\n")[0]

            return exists, evidence

        except Exception as e:
            logger.error(f"Existence check failed: {e}")
            return False, None

    def _extract_details(
        self,
        field_name: str,
        context: str,
    ) -> Optional[dict]:
        """Tier3 direct extraction with instructor schema validation."""
        schema_class = FIELD_SCHEMA_MAP.get(field_name)
        extraction_prompt = TIER3_EXTRACTION_PROMPTS.get(field_name)

        if not schema_class or not extraction_prompt:
            return None

        prompt = f"""Extract structured data from this text.

TEXT:
{context[:25000]}

INSTRUCTIONS:
{extraction_prompt}

Return ONLY valid JSON matching the schema. Use null for missing values."""

        try:
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
                        {"role": "system", "content": "Extract structured data. Return valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                )

            time.sleep(self.delay)

            # Convert to dict, remove chain-of-thought fields
            result_dict = result.model_dump(exclude_none=False)
            result_dict.pop("reasoning", None)
            result_dict.pop("confidence", None)
            result_dict.pop("citation", None)

            return result_dict

        except Exception as e:
            logger.error(f"Detail extraction failed for {field_name}: {e}")
            return None

    def _get_null_result(self, field_name: str) -> dict:
        """Return appropriate null structure for a field."""
        null_results = {
            "incentive_fee": {
                "has_incentive_fee": False,
                "incentive_fee_pct": None,
                "hurdle_rate_pct": None,
                "high_water_mark": None,
                "fee_basis": None,
                "crystallization_frequency": None,
            },
            "expense_cap": {
                "has_expense_cap": False,
                "expense_cap_pct": None,
                "cap_type": None,
                "cap_expiration": None,
            },
            "repurchase_terms": {
                "repurchase_frequency": None,
                "repurchase_amount_pct": None,
                "repurchase_basis": None,
                "early_repurchase_fee_pct": None,
                "lock_up_period_years": None,
            },
            "leverage_limits": {
                "uses_leverage": False,
                "max_leverage_pct": None,
                "leverage_basis": None,
                "credit_facility_size": None,
            },
            "distribution_terms": {
                "distribution_frequency": None,
                "default_distribution_policy": None,
            },
            "share_classes": {
                "share_classes": [],
            },
            "allocation_targets": {
                "has_allocation_targets": False,
                "allocations": [],
            },
            "concentration_limits": {
                "has_concentration_limits": False,
                "limits": [],
            },
        }
        return null_results.get(field_name, {})

    def extract_field(
        self,
        field_name: str,
        chunked_doc,
    ) -> HybridExtractionResult:
        """
        Extract a field using hybrid approach.

        1. Tier3 chunk selection
        2. DocVQA existence check
        3. If exists → Tier3 detail extraction
        """
        logger.info(f"  [Hybrid] Extracting {field_name}")

        # Step 1: Get relevant chunks (Tier3)
        context, sections_searched, chunks_retrieved = self._get_relevant_chunks(
            field_name, chunked_doc
        )
        logger.info(f"    Retrieved {chunks_retrieved} chunks from {sections_searched} sections")

        if not context:
            return HybridExtractionResult(
                field_name=field_name,
                sections_searched=0,
                chunks_retrieved=0,
                existence_result=False,
                existence_evidence=None,
                extraction_result=self._get_null_result(field_name),
                extraction_method="no_context",
                llm_calls=0,
            )

        # Step 2: DocVQA existence check
        exists, evidence = self._ask_existence(field_name, context)
        logger.info(f"    Existence: {exists} (evidence: {evidence[:50] if evidence else 'None'}...)")

        llm_calls = 1

        # Step 3: Tier3 detail extraction (only if exists)
        if exists:
            extraction = self._extract_details(field_name, context)
            llm_calls += 1
            method = "tier3_details"

            if extraction is None:
                extraction = self._get_null_result(field_name)
        else:
            extraction = self._get_null_result(field_name)
            method = "docvqa_existence_only"

        logger.info(f"    Method: {method}, LLM calls: {llm_calls}")

        return HybridExtractionResult(
            field_name=field_name,
            sections_searched=sections_searched,
            chunks_retrieved=chunks_retrieved,
            existence_result=exists,
            existence_evidence=evidence,
            extraction_result=extraction,
            extraction_method=method,
            llm_calls=llm_calls,
        )

    def extract_all_fields(
        self,
        chunked_doc,
        fields: Optional[list[str]] = None,
    ) -> dict[str, HybridExtractionResult]:
        """Extract all fields using hybrid approach."""
        if fields is None:
            fields = list(EXISTENCE_QUESTIONS.keys())

        results = {}
        for field_name in fields:
            logger.info(f"[Hybrid] Processing {field_name}")
            results[field_name] = self.extract_field(field_name, chunked_doc)

        return results


def convert_hybrid_to_extraction_format(
    results: dict[str, HybridExtractionResult]
) -> dict:
    """Convert hybrid results to standard extraction format."""
    extraction = {}
    for field_name, result in results.items():
        if result.extraction_result:
            extraction[field_name] = result.extraction_result
        else:
            extraction[field_name] = None
    return extraction
