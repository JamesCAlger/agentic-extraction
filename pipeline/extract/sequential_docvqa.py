"""
Sequential Conditional DocVQA extraction approach.

Instead of asking all questions at once, this approach:
1. First asks existence questions ("Does this fund have X?")
2. Only asks detail questions if existence = true
3. Uses evidence-first format for better grounding

This reduces hallucinations by:
- Not asking for details when the feature doesn't exist
- Forcing evidence citation before extraction
- Breaking complex extractions into atomic steps
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


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class ExistenceCheckResult:
    """Result of checking if a feature exists."""
    field_name: str
    exists: Optional[bool]  # True/False/None (unknown)
    evidence: Optional[str]
    confidence: str  # "explicit", "inferred", "not_found"


@dataclass
class DetailExtractionResult:
    """Result of extracting details for a field."""
    field_name: str
    sub_field: str
    value: Any
    evidence: Optional[str]
    grounded: bool


@dataclass
class SequentialDocVQAResult:
    """Complete result for a field using sequential extraction."""
    field_name: str
    existence_check: ExistenceCheckResult
    detail_results: list[DetailExtractionResult]
    synthesized_value: Any
    confidence: str  # "explicit", "inferred", "not_found"
    grounded: bool
    questions_asked: int
    grounded_answers: int


# =============================================================================
# FIELD CONFIGURATIONS
# =============================================================================

# Each field has:
# - existence_question: The first question to ask
# - existence_indicator: Field name that indicates existence (for schema)
# - detail_questions: Questions to ask only if existence = True
# - detail_mapping: How detail answers map to schema fields

FIELD_CONFIGS = {
    "incentive_fee": {
        "existence_question": """Look for performance-based fees or incentive fees in this document.

IMPORTANT: Only consider FUND-LEVEL fees (fees charged by THIS fund to its shareholders).
Ignore fees charged by underlying funds (those are AFFE, not incentive fees).

Does this fund charge a performance-based fee or incentive fee at the fund level?

Respond in this format:
EXISTS: [YES/NO/UNKNOWN]
EVIDENCE: [Quote the exact text that supports your answer, or "None found"]""",
        "existence_indicator": "has_incentive_fee",
        "detail_questions": [
            {
                "sub_field": "incentive_fee_pct",
                "question": """What is the incentive/performance fee percentage for this fund?
Look for phrases like "incentive fee of X%" or "performance fee equal to X%".

Respond in this format:
VALUE: [The percentage as a number, e.g., 10 or 15, or NULL if not found]
EVIDENCE: [Quote the exact text]"""
            },
            {
                "sub_field": "hurdle_rate_pct",
                "question": """Is there a hurdle rate that must be exceeded before incentive fees are charged?
Look for "hurdle rate", "preferred return", or "benchmark rate".

Respond in this format:
VALUE: [The percentage as a number, or NULL if not found]
EVIDENCE: [Quote the exact text]"""
            },
            {
                "sub_field": "high_water_mark",
                "question": """Is there a high water mark or loss recovery mechanism?
Look for "high water mark", "loss recovery account", "deficit carryforward".

Respond in this format:
VALUE: [TRUE/FALSE/NULL]
EVIDENCE: [Quote the exact text]"""
            },
            {
                "sub_field": "crystallization_frequency",
                "question": """How often are incentive fees calculated/crystallized?
Look for "annually", "quarterly", or other frequency terms near incentive fee language.

Respond in this format:
VALUE: [quarterly/annual/monthly/other/NULL]
EVIDENCE: [Quote the exact text]"""
            },
        ],
    },

    "expense_cap": {
        "existence_question": """Look for expense caps, expense limits, or fee waivers in this document.

IMPORTANT: Distinguish between:
- EXPENSE CAP: A contractual limit (e.g., "expenses shall not exceed 2.00%")
- TOTAL EXPENSES: The actual expense ratio shown (e.g., "Total Annual Expenses 4.08%") - NOT a cap

Does this fund have a contractual expense cap or limit?

Respond in this format:
EXISTS: [YES/NO/UNKNOWN]
EVIDENCE: [Quote the exact text that supports your answer, or "None found"]""",
        "existence_indicator": "has_expense_cap",
        "detail_questions": [
            {
                "sub_field": "expense_cap_pct",
                "question": """What is the expense cap percentage?
Look for "capped at X%", "limit of X%", "shall not exceed X%".

Respond in this format:
VALUE: [The percentage as a number, or NULL if not found]
EVIDENCE: [Quote the exact text]"""
            },
            {
                "sub_field": "cap_type",
                "question": """Is this expense cap contractual or voluntary?
Look for "contractual", "voluntary", "undertaking", "agreement".

Respond in this format:
VALUE: [contractual/voluntary/NULL]
EVIDENCE: [Quote the exact text]"""
            },
            {
                "sub_field": "cap_expiration",
                "question": """When does the expense cap expire?
Look for expiration dates or terms like "until [date]", "through [date]".

Respond in this format:
VALUE: [The expiration date or period, or NULL if not found]
EVIDENCE: [Quote the exact text]"""
            },
        ],
    },

    "repurchase_terms": {
        "existence_question": """Look for share repurchase or redemption information in this document.

For interval funds and tender offer funds, look for:
- Repurchase offers
- Redemption terms
- Periodic liquidity

Does this document describe share repurchase/redemption terms?

Respond in this format:
EXISTS: [YES/NO/UNKNOWN]
EVIDENCE: [Quote the exact text that supports your answer, or "None found"]""",
        "existence_indicator": "has_repurchase_terms",
        "detail_questions": [
            {
                "sub_field": "repurchase_frequency",
                "question": """How often does the fund offer to repurchase shares?
Look for "quarterly", "semi-annually", "annually", or "monthly".

Respond in this format:
VALUE: [quarterly/semi_annual/annual/monthly/discretionary/NULL]
EVIDENCE: [Quote the exact text]"""
            },
            {
                "sub_field": "repurchase_amount_pct",
                "question": """What percentage of shares is offered for repurchase?
Look for "5%", "up to 25%", or similar percentages.

Respond in this format:
VALUE: [The percentage as a number, or NULL if not found]
EVIDENCE: [Quote the exact text]"""
            },
            {
                "sub_field": "early_repurchase_fee_pct",
                "question": """Is there an early repurchase fee or penalty?
Look for "early repurchase fee", "redemption fee", "CDSC".

Respond in this format:
VALUE: [The percentage as a number, or NULL if none]
EVIDENCE: [Quote the exact text]"""
            },
        ],
    },

    "leverage_limits": {
        "existence_question": """Look for leverage, borrowing, or credit facility information in this document.

Does this fund use leverage (borrowing)?

Respond in this format:
EXISTS: [YES/NO/UNKNOWN]
EVIDENCE: [Quote the exact text that supports your answer, or "None found"]""",
        "existence_indicator": "uses_leverage",
        "detail_questions": [
            {
                "sub_field": "max_leverage_pct",
                "question": """What is the maximum leverage the fund can use?
Look for percentages like "up to 33%", "no more than 30%", or asset coverage ratios.

Respond in this format:
VALUE: [The percentage as a number, or NULL if not found]
EVIDENCE: [Quote the exact text]"""
            },
            {
                "sub_field": "leverage_basis",
                "question": """How is leverage measured/expressed?
Look for "total assets", "net assets", "asset coverage", "debt-to-equity".

Respond in this format:
VALUE: [total_assets/net_assets/asset_coverage/debt_to_equity/NULL]
EVIDENCE: [Quote the exact text]"""
            },
            {
                "sub_field": "credit_facility_size",
                "question": """What is the size of the fund's credit facility?
Look for dollar amounts like "$500 million" or "$1 billion".

Respond in this format:
VALUE: [The amount as a number without $ or commas, or NULL]
EVIDENCE: [Quote the exact text]"""
            },
        ],
    },

    "distribution_terms": {
        "existence_question": """Look for distribution, dividend, or income payment information in this document.

Does this document describe distribution/dividend terms?

Respond in this format:
EXISTS: [YES/NO/UNKNOWN]
EVIDENCE: [Quote the exact text that supports your answer, or "None found"]""",
        "existence_indicator": "has_distributions",
        "detail_questions": [
            {
                "sub_field": "distribution_frequency",
                "question": """How often does the fund pay distributions?
Look for "monthly", "quarterly", "annually".

Respond in this format:
VALUE: [monthly/quarterly/annual/NULL]
EVIDENCE: [Quote the exact text]"""
            },
            {
                "sub_field": "default_distribution_policy",
                "question": """What is the default distribution policy - cash or reinvested?
Look for "DRIP", "reinvestment", "cash", "dividend reinvestment plan".

Respond in this format:
VALUE: [cash/DRIP/shareholder_choice/NULL]
EVIDENCE: [Quote the exact text]"""
            },
        ],
    },

    "share_classes": {
        "existence_question": """Look for share class information in this document.

Does this document describe different share classes (e.g., Class A, Class I, Class S)?

Respond in this format:
EXISTS: [YES/NO/UNKNOWN]
EVIDENCE: [Quote the share class names if found, or "None found"]""",
        "existence_indicator": "has_share_classes",
        # Share classes require special handling - multiple instances
        "detail_questions": [
            {
                "sub_field": "class_details",
                "question": """For EACH share class mentioned, extract:
- Class name
- Minimum initial investment
- Minimum additional investment
- Sales load percentage (0 if none)
- Distribution/servicing fee percentage (0 if none)

Respond in this format for EACH class (separate with ---):
CLASS_NAME: [e.g., Class I]
MIN_INITIAL: [dollar amount as number, or NULL]
MIN_ADDITIONAL: [dollar amount as number, or NULL]
SALES_LOAD: [percentage as number, or 0]
DISTRIBUTION_FEE: [percentage as number, or 0]
EVIDENCE: [Quote the exact text]
---"""
            },
        ],
    },

    "allocation_targets": {
        "existence_question": """Look for investment allocation targets or strategy breakdown in this document.

Does this document describe target allocations (e.g., % to secondaries, % to co-investments)?

Respond in this format:
EXISTS: [YES/NO/UNKNOWN]
EVIDENCE: [Quote the exact text that supports your answer, or "None found"]""",
        "existence_indicator": "has_allocation_targets",
        "detail_questions": [
            {
                "sub_field": "allocations",
                "question": """Extract the allocation targets mentioned.
Look for percentage ranges for:
- Secondaries
- Co-investments
- Direct investments
- Primary fund investments

Respond in this format for EACH allocation type found:
TYPE: [secondary/co_investment/direct/primary]
MIN_PCT: [minimum percentage as number]
MAX_PCT: [maximum percentage as number]
EVIDENCE: [Quote the exact text]
---"""
            },
        ],
    },

    "concentration_limits": {
        "existence_question": """Look for concentration limits or diversification requirements in this document.

Does this document describe limits on concentration (max % in single issuer, sector, etc.)?

Respond in this format:
EXISTS: [YES/NO/UNKNOWN]
EVIDENCE: [Quote the exact text that supports your answer, or "None found"]""",
        "existence_indicator": "has_concentration_limits",
        "detail_questions": [
            {
                "sub_field": "limits",
                "question": """Extract the concentration limits mentioned.
Look for maximum percentages for:
- Single issuer
- Single fund
- Single sector/industry
- Single asset

Respond in this format for EACH limit found:
LIMIT_TYPE: [single_issuer/single_fund/single_sector/single_asset]
MAX_PCT: [maximum percentage as number]
EVIDENCE: [Quote the exact text]
---"""
            },
        ],
    },
}


class SequentialDocVQAExtractor:
    """
    Sequential Conditional DocVQA extractor.

    Process:
    1. Ask existence question for each field
    2. If exists=True, ask detail questions
    3. Synthesize answers into structured extraction
    4. Validate with instructor schemas
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

        # Create clients
        if provider == "anthropic":
            self.client = Anthropic(api_key=api_key)
            self.instructor_client = instructor.from_anthropic(
                Anthropic(api_key=api_key)
            ) if use_instructor else None
        else:
            self.client = OpenAI(api_key=api_key)
            self.instructor_client = instructor.from_openai(
                OpenAI(api_key=api_key)
            ) if use_instructor else None

    def _ask_llm(self, prompt: str, context: str) -> str:
        """Ask a single question to the LLM."""
        full_prompt = f"""Based ONLY on the text below, answer the question.
If the information is not in the text, say so clearly.

TEXT:
{context[:15000]}

QUESTION:
{prompt}"""

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": full_prompt}],
                )
                return response.content[0].text
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Answer based only on the provided text. Quote evidence exactly."},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0,
                )
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"ERROR: {e}"
        finally:
            time.sleep(self.delay)

    def _check_existence(
        self,
        field_name: str,
        context: str
    ) -> ExistenceCheckResult:
        """
        Check if a feature exists in the document.

        Returns ExistenceCheckResult with exists=True/False/None
        """
        field_config = FIELD_CONFIGS.get(field_name)
        if not field_config:
            return ExistenceCheckResult(
                field_name=field_name,
                exists=None,
                evidence=None,
                confidence="not_found"
            )

        question = field_config["existence_question"]
        response = self._ask_llm(question, context)

        # Parse response
        exists = None
        evidence = None
        confidence = "not_found"

        response_upper = response.upper()
        if "EXISTS: YES" in response_upper or "EXISTS:YES" in response_upper:
            exists = True
            confidence = "explicit"
        elif "EXISTS: NO" in response_upper or "EXISTS:NO" in response_upper:
            exists = False
            confidence = "explicit"
        elif "EXISTS: UNKNOWN" in response_upper:
            exists = None
            confidence = "inferred"

        # Extract evidence
        if "EVIDENCE:" in response:
            evidence_part = response.split("EVIDENCE:", 1)[1].strip()
            if evidence_part.lower() not in ["none", "none found", "n/a", ""]:
                evidence = evidence_part.split("\n")[0].strip()

        logger.debug(f"  [SeqDocVQA] {field_name} existence: {exists} (conf={confidence})")

        return ExistenceCheckResult(
            field_name=field_name,
            exists=exists,
            evidence=evidence,
            confidence=confidence
        )

    def _extract_detail(
        self,
        field_name: str,
        detail_config: dict,
        context: str
    ) -> DetailExtractionResult:
        """Extract a single detail field."""
        question = detail_config["question"]
        sub_field = detail_config["sub_field"]

        response = self._ask_llm(question, context)

        # Parse response
        value = None
        evidence = None
        grounded = False

        # Parse VALUE line
        if "VALUE:" in response:
            value_part = response.split("VALUE:", 1)[1].strip()
            value_line = value_part.split("\n")[0].strip()

            # Parse the value
            if value_line.upper() in ["NULL", "NONE", "N/A", ""]:
                value = None
            elif value_line.upper() in ["TRUE", "YES"]:
                value = True
            elif value_line.upper() in ["FALSE", "NO"]:
                value = False
            else:
                # Try to parse as number
                try:
                    if "." in value_line:
                        value = float(value_line)
                    else:
                        value = int(value_line.replace(",", ""))
                except ValueError:
                    value = value_line

        # Parse EVIDENCE line
        if "EVIDENCE:" in response:
            evidence_part = response.split("EVIDENCE:", 1)[1].strip()
            evidence_line = evidence_part.split("\n")[0].strip()
            if evidence_line.lower() not in ["none", "none found", "n/a", ""]:
                evidence = evidence_line
                grounded = True

        logger.debug(f"    [SeqDocVQA] {sub_field}: {value} (grounded={grounded})")

        return DetailExtractionResult(
            field_name=field_name,
            sub_field=sub_field,
            value=value,
            evidence=evidence,
            grounded=grounded
        )

    def _extract_share_classes(
        self,
        context: str
    ) -> list[dict]:
        """
        Special handling for share classes - multiple instances.
        """
        field_config = FIELD_CONFIGS["share_classes"]
        detail_config = field_config["detail_questions"][0]

        response = self._ask_llm(detail_config["question"], context)

        # Parse multiple class entries separated by ---
        classes = []
        entries = response.split("---")

        for entry in entries:
            if "CLASS_NAME:" not in entry:
                continue

            class_data = {
                "class_name": None,
                "minimum_initial_investment": None,
                "minimum_additional_investment": None,
                "sales_load_pct": 0,
                "distribution_servicing_fee_pct": 0,
            }

            for line in entry.strip().split("\n"):
                line = line.strip()
                if line.startswith("CLASS_NAME:"):
                    class_data["class_name"] = line.split(":", 1)[1].strip()
                elif line.startswith("MIN_INITIAL:"):
                    val = line.split(":", 1)[1].strip()
                    if val.upper() not in ["NULL", "NONE", ""]:
                        try:
                            class_data["minimum_initial_investment"] = float(val.replace(",", ""))
                        except ValueError:
                            pass
                elif line.startswith("MIN_ADDITIONAL:"):
                    val = line.split(":", 1)[1].strip()
                    if val.upper() not in ["NULL", "NONE", ""]:
                        try:
                            class_data["minimum_additional_investment"] = float(val.replace(",", ""))
                        except ValueError:
                            pass
                elif line.startswith("SALES_LOAD:"):
                    val = line.split(":", 1)[1].strip()
                    try:
                        class_data["sales_load_pct"] = float(val)
                    except ValueError:
                        pass
                elif line.startswith("DISTRIBUTION_FEE:"):
                    val = line.split(":", 1)[1].strip()
                    try:
                        class_data["distribution_servicing_fee_pct"] = float(val)
                    except ValueError:
                        pass

            if class_data["class_name"]:
                classes.append(class_data)

        return classes

    def _extract_allocation_targets(
        self,
        context: str
    ) -> list[dict]:
        """Special handling for allocation targets - multiple instances."""
        field_config = FIELD_CONFIGS["allocation_targets"]
        detail_config = field_config["detail_questions"][0]

        response = self._ask_llm(detail_config["question"], context)

        # Parse multiple allocation entries
        allocations = []
        entries = response.split("---")

        for entry in entries:
            if "TYPE:" not in entry:
                continue

            alloc_data = {
                "allocation_type": None,
                "min_pct": None,
                "max_pct": None,
            }

            for line in entry.strip().split("\n"):
                line = line.strip()
                if line.startswith("TYPE:"):
                    alloc_data["allocation_type"] = line.split(":", 1)[1].strip()
                elif line.startswith("MIN_PCT:"):
                    val = line.split(":", 1)[1].strip()
                    try:
                        alloc_data["min_pct"] = float(val)
                    except ValueError:
                        pass
                elif line.startswith("MAX_PCT:"):
                    val = line.split(":", 1)[1].strip()
                    try:
                        alloc_data["max_pct"] = float(val)
                    except ValueError:
                        pass

            if alloc_data["allocation_type"]:
                allocations.append(alloc_data)

        return allocations

    def _extract_concentration_limits(
        self,
        context: str
    ) -> list[dict]:
        """Special handling for concentration limits - multiple instances."""
        field_config = FIELD_CONFIGS["concentration_limits"]
        detail_config = field_config["detail_questions"][0]

        response = self._ask_llm(detail_config["question"], context)

        # Parse multiple limit entries
        limits = []
        entries = response.split("---")

        for entry in entries:
            if "LIMIT_TYPE:" not in entry:
                continue

            limit_data = {
                "limit_type": None,
                "max_pct": None,
            }

            for line in entry.strip().split("\n"):
                line = line.strip()
                if line.startswith("LIMIT_TYPE:"):
                    limit_data["limit_type"] = line.split(":", 1)[1].strip()
                elif line.startswith("MAX_PCT:"):
                    val = line.split(":", 1)[1].strip()
                    try:
                        limit_data["max_pct"] = float(val)
                    except ValueError:
                        pass

            if limit_data["limit_type"]:
                limits.append(limit_data)

        return limits

    def _synthesize_field(
        self,
        field_name: str,
        existence: ExistenceCheckResult,
        details: list[DetailExtractionResult],
        context: str
    ) -> dict:
        """
        Synthesize existence check and detail results into final extraction.

        Uses instructor schema validation when available.
        """
        schema_class = FIELD_SCHEMA_MAP.get(field_name)
        field_config = FIELD_CONFIGS.get(field_name, {})
        existence_indicator = field_config.get("existence_indicator", f"has_{field_name}")

        # Build base result from existence check and details
        result = {existence_indicator: existence.exists}

        for detail in details:
            result[detail.sub_field] = detail.value

        # Handle special multi-instance fields
        if field_name == "share_classes":
            result["share_classes"] = self._extract_share_classes(context) if existence.exists else []
        elif field_name == "allocation_targets":
            result["allocations"] = self._extract_allocation_targets(context) if existence.exists else []
        elif field_name == "concentration_limits":
            result["limits"] = self._extract_concentration_limits(context) if existence.exists else []

        # Try instructor validation if available
        if self.use_instructor and self.instructor_client and schema_class:
            try:
                return self._validate_with_instructor(field_name, schema_class, result)
            except Exception as e:
                logger.warning(f"Instructor validation failed for {field_name}: {e}")

        return result

    def _validate_with_instructor(
        self,
        field_name: str,
        schema_class: Type[BaseModel],
        raw_result: dict
    ) -> dict:
        """Validate and normalize result using instructor schema."""
        # Build prompt from raw result
        prompt = f"""Validate and normalize this extraction for {field_name}:

Raw data: {raw_result}

Return the validated data matching the expected schema.
Use null for missing values, not empty strings.
For percentages, use numeric values without % suffix.
"""

        if self.provider == "anthropic":
            result = self.instructor_client.messages.create(
                model=self.model,
                max_tokens=1500,
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
                    {"role": "system", "content": "Validate and normalize extraction data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
            )

        # Convert to dict, removing chain-of-thought fields
        result_dict = result.model_dump(exclude_none=False)
        result_dict.pop("reasoning", None)
        result_dict.pop("confidence", None)
        result_dict.pop("citation", None)

        return result_dict

    def extract_field(
        self,
        field_name: str,
        context: str
    ) -> SequentialDocVQAResult:
        """
        Extract a single field using sequential conditional approach.

        1. Check existence
        2. If exists, extract details
        3. Synthesize and validate
        """
        logger.info(f"  [SeqDocVQA] Extracting {field_name}")

        field_config = FIELD_CONFIGS.get(field_name)
        if not field_config:
            return SequentialDocVQAResult(
                field_name=field_name,
                existence_check=ExistenceCheckResult(
                    field_name=field_name,
                    exists=None,
                    evidence=None,
                    confidence="not_found"
                ),
                detail_results=[],
                synthesized_value=None,
                confidence="not_found",
                grounded=False,
                questions_asked=0,
                grounded_answers=0
            )

        # Step 1: Check existence
        existence = self._check_existence(field_name, context)
        questions_asked = 1
        grounded_answers = 1 if existence.evidence else 0

        # Step 2: Extract details if exists
        detail_results = []
        if existence.exists:
            for detail_config in field_config.get("detail_questions", []):
                # Skip multi-instance fields - handled in synthesis
                if detail_config["sub_field"] in ["class_details", "allocations", "limits"]:
                    questions_asked += 1
                    continue

                detail = self._extract_detail(field_name, detail_config, context)
                detail_results.append(detail)
                questions_asked += 1
                if detail.grounded:
                    grounded_answers += 1

        # Step 3: Synthesize
        synthesized = self._synthesize_field(
            field_name, existence, detail_results, context
        )

        # Determine overall confidence and grounding
        if existence.exists is None:
            confidence = "not_found"
        elif grounded_answers >= questions_asked / 2:
            confidence = "explicit"
        else:
            confidence = "inferred"

        grounded = grounded_answers > 0

        logger.info(
            f"  [SeqDocVQA] {field_name}: exists={existence.exists}, "
            f"{grounded_answers}/{questions_asked} grounded"
        )

        return SequentialDocVQAResult(
            field_name=field_name,
            existence_check=existence,
            detail_results=detail_results,
            synthesized_value=synthesized,
            confidence=confidence,
            grounded=grounded,
            questions_asked=questions_asked,
            grounded_answers=grounded_answers
        )

    def extract_all_fields(
        self,
        context: str,
        fields: Optional[list[str]] = None
    ) -> dict[str, SequentialDocVQAResult]:
        """
        Extract all fields using sequential conditional approach.

        Args:
            context: Document text
            fields: List of fields to extract (defaults to all)

        Returns:
            Dict mapping field_name -> SequentialDocVQAResult
        """
        if fields is None:
            fields = list(FIELD_CONFIGS.keys())

        results = {}
        for field_name in fields:
            logger.info(f"[SeqDocVQA] Processing {field_name}")
            results[field_name] = self.extract_field(field_name, context)

        return results


def convert_sequential_docvqa_to_extraction_format(
    results: dict[str, SequentialDocVQAResult]
) -> dict:
    """
    Convert SequentialDocVQA results to standard extraction format.

    This allows comparison with other extraction methods.
    """
    extraction = {}

    for field_name, result in results.items():
        if result.synthesized_value:
            extraction[field_name] = result.synthesized_value
        else:
            extraction[field_name] = None

    return extraction


# =============================================================================
# TIER 3 + SEQUENTIAL DocVQA HYBRID
# =============================================================================

# Import Tier3 keyword scoring
from .scoped_agentic import FIELD_KEYWORDS, score_section_for_field


@dataclass
class Tier3SequentialResult:
    """Result from Tier3+Sequential DocVQA extraction."""
    field_name: str
    sections_searched: int
    chunks_retrieved: int
    existence_check: ExistenceCheckResult
    detail_results: list[DetailExtractionResult]
    synthesized_value: Any
    confidence: str
    grounded: bool
    questions_asked: int
    grounded_answers: int


class Tier3SequentialDocVQAExtractor:
    """
    Tier 3 + Sequential DocVQA Hybrid Extractor.

    Combines:
    1. Tier3 keyword scoring for smart chunk selection
    2. Sequential existence-first questioning pattern

    Benefits:
    - Finds relevant sections in large documents (Tier3)
    - Reduces hallucinations via existence checks (Sequential)
    - Evidence-first format for better grounding
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
        self.top_k_sections = top_k_sections
        self.max_chunks_per_section = max_chunks_per_section
        self.delay = delay_between_calls
        self.api_key = api_key
        self.use_instructor = use_instructor

        # Create clients
        if provider == "anthropic":
            self.client = Anthropic(api_key=api_key)
            self.instructor_client = instructor.from_anthropic(
                Anthropic(api_key=api_key)
            ) if use_instructor else None
        else:
            self.client = OpenAI(api_key=api_key)
            self.instructor_client = instructor.from_openai(
                OpenAI(api_key=api_key)
            ) if use_instructor else None

    def _get_relevant_chunks(
        self,
        field_name: str,
        chunked_doc,
    ) -> tuple[str, int, int]:
        """
        Use Tier3 keyword scoring to find relevant chunks for a field.

        Returns:
            Tuple of (combined_text, sections_searched, chunks_retrieved)
        """
        # Score all sections
        section_scores = []
        for section in chunked_doc.chunked_sections:
            scored = score_section_for_field(section, field_name)
            if scored.score > 0:
                section_scores.append((scored.score, section))

        # Sort by score descending
        section_scores.sort(key=lambda x: x[0], reverse=True)

        # Take top-k sections
        top_sections = section_scores[:self.top_k_sections]

        if top_sections:
            logger.info(f"    Top {len(top_sections)} sections by keyword score:")
            for score, section in top_sections[:3]:
                logger.info(f"      - {section.section_title[:50]}... (score: {score})")

        # Collect chunks from top sections
        chunks = []
        for score, section in top_sections:
            section_chunks = section.chunks[:self.max_chunks_per_section]
            chunks.extend(section_chunks)

        # Combine chunk content
        combined_text = "\n\n---\n\n".join(
            f"[Section: {chunk.section_id}]\n{chunk.content}"
            for chunk in chunks
        )

        return combined_text, len(top_sections), len(chunks)

    def _ask_llm(self, prompt: str, context: str) -> str:
        """Ask a single question to the LLM."""
        full_prompt = f"""Based ONLY on the text below, answer the question.
If the information is not in the text, say so clearly.

TEXT:
{context[:30000]}

QUESTION:
{prompt}"""

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": full_prompt}],
                )
                return response.content[0].text
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Answer based only on the provided text. Quote evidence exactly."},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0,
                )
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"ERROR: {e}"
        finally:
            time.sleep(self.delay)

    def _check_existence(
        self,
        field_name: str,
        context: str
    ) -> ExistenceCheckResult:
        """Check if a feature exists using the retrieved chunks."""
        field_config = FIELD_CONFIGS.get(field_name)
        if not field_config:
            return ExistenceCheckResult(
                field_name=field_name,
                exists=None,
                evidence=None,
                confidence="not_found"
            )

        question = field_config["existence_question"]
        response = self._ask_llm(question, context)

        # Parse response
        exists = None
        evidence = None
        confidence = "not_found"

        response_upper = response.upper()
        if "EXISTS: YES" in response_upper or "EXISTS:YES" in response_upper:
            exists = True
            confidence = "explicit"
        elif "EXISTS: NO" in response_upper or "EXISTS:NO" in response_upper:
            exists = False
            confidence = "explicit"
        elif "EXISTS: UNKNOWN" in response_upper:
            exists = None
            confidence = "inferred"

        # Extract evidence
        if "EVIDENCE:" in response:
            evidence_part = response.split("EVIDENCE:", 1)[1].strip()
            if evidence_part.lower() not in ["none", "none found", "n/a", ""]:
                evidence = evidence_part.split("\n")[0].strip()

        return ExistenceCheckResult(
            field_name=field_name,
            exists=exists,
            evidence=evidence,
            confidence=confidence
        )

    def _extract_detail(
        self,
        field_name: str,
        detail_config: dict,
        context: str
    ) -> DetailExtractionResult:
        """Extract a single detail field."""
        question = detail_config["question"]
        sub_field = detail_config["sub_field"]

        response = self._ask_llm(question, context)

        # Parse response
        value = None
        evidence = None
        grounded = False

        if "VALUE:" in response:
            value_part = response.split("VALUE:", 1)[1].strip()
            value_line = value_part.split("\n")[0].strip()

            if value_line.upper() in ["NULL", "NONE", "N/A", ""]:
                value = None
            elif value_line.upper() in ["TRUE", "YES"]:
                value = True
            elif value_line.upper() in ["FALSE", "NO"]:
                value = False
            else:
                try:
                    if "." in value_line:
                        value = float(value_line)
                    else:
                        value = int(value_line.replace(",", ""))
                except ValueError:
                    value = value_line

        if "EVIDENCE:" in response:
            evidence_part = response.split("EVIDENCE:", 1)[1].strip()
            evidence_line = evidence_part.split("\n")[0].strip()
            if evidence_line.lower() not in ["none", "none found", "n/a", ""]:
                evidence = evidence_line
                grounded = True

        return DetailExtractionResult(
            field_name=field_name,
            sub_field=sub_field,
            value=value,
            evidence=evidence,
            grounded=grounded
        )

    def _extract_multi_instance(
        self,
        field_name: str,
        context: str
    ) -> list[dict]:
        """Extract multi-instance fields (share_classes, allocations, limits)."""
        field_config = FIELD_CONFIGS.get(field_name, {})
        detail_questions = field_config.get("detail_questions", [])

        if not detail_questions:
            return []

        detail_config = detail_questions[0]
        response = self._ask_llm(detail_config["question"], context)

        # Parse based on field type
        if field_name == "share_classes":
            return self._parse_share_classes(response)
        elif field_name == "allocation_targets":
            return self._parse_allocations(response)
        elif field_name == "concentration_limits":
            return self._parse_limits(response)

        return []

    def _parse_share_classes(self, response: str) -> list[dict]:
        """Parse share class response."""
        classes = []
        entries = response.split("---")

        for entry in entries:
            if "CLASS_NAME:" not in entry:
                continue

            class_data = {
                "class_name": None,
                "minimum_initial_investment": None,
                "minimum_additional_investment": None,
                "sales_load_pct": 0,
                "distribution_servicing_fee_pct": 0,
            }

            for line in entry.strip().split("\n"):
                line = line.strip()
                if line.startswith("CLASS_NAME:"):
                    class_data["class_name"] = line.split(":", 1)[1].strip()
                elif line.startswith("MIN_INITIAL:"):
                    val = line.split(":", 1)[1].strip()
                    if val.upper() not in ["NULL", "NONE", ""]:
                        try:
                            class_data["minimum_initial_investment"] = float(val.replace(",", ""))
                        except ValueError:
                            pass
                elif line.startswith("MIN_ADDITIONAL:"):
                    val = line.split(":", 1)[1].strip()
                    if val.upper() not in ["NULL", "NONE", ""]:
                        try:
                            class_data["minimum_additional_investment"] = float(val.replace(",", ""))
                        except ValueError:
                            pass
                elif line.startswith("SALES_LOAD:"):
                    try:
                        class_data["sales_load_pct"] = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith("DISTRIBUTION_FEE:"):
                    try:
                        class_data["distribution_servicing_fee_pct"] = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass

            if class_data["class_name"]:
                classes.append(class_data)

        return classes

    def _parse_allocations(self, response: str) -> list[dict]:
        """Parse allocation response."""
        allocations = []
        entries = response.split("---")

        for entry in entries:
            if "TYPE:" not in entry:
                continue

            alloc_data = {"allocation_type": None, "min_pct": None, "max_pct": None}

            for line in entry.strip().split("\n"):
                line = line.strip()
                if line.startswith("TYPE:"):
                    alloc_data["allocation_type"] = line.split(":", 1)[1].strip()
                elif line.startswith("MIN_PCT:"):
                    try:
                        alloc_data["min_pct"] = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith("MAX_PCT:"):
                    try:
                        alloc_data["max_pct"] = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass

            if alloc_data["allocation_type"]:
                allocations.append(alloc_data)

        return allocations

    def _parse_limits(self, response: str) -> list[dict]:
        """Parse concentration limits response."""
        limits = []
        entries = response.split("---")

        for entry in entries:
            if "LIMIT_TYPE:" not in entry:
                continue

            limit_data = {"limit_type": None, "max_pct": None}

            for line in entry.strip().split("\n"):
                line = line.strip()
                if line.startswith("LIMIT_TYPE:"):
                    limit_data["limit_type"] = line.split(":", 1)[1].strip()
                elif line.startswith("MAX_PCT:"):
                    try:
                        limit_data["max_pct"] = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass

            if limit_data["limit_type"]:
                limits.append(limit_data)

        return limits

    def _synthesize_field(
        self,
        field_name: str,
        existence: ExistenceCheckResult,
        details: list[DetailExtractionResult],
        context: str
    ) -> dict:
        """Synthesize extraction result."""
        field_config = FIELD_CONFIGS.get(field_name, {})
        existence_indicator = field_config.get("existence_indicator", f"has_{field_name}")

        result = {existence_indicator: existence.exists}

        for detail in details:
            result[detail.sub_field] = detail.value

        # Handle multi-instance fields
        if field_name == "share_classes" and existence.exists:
            result["share_classes"] = self._extract_multi_instance(field_name, context)
        elif field_name == "allocation_targets" and existence.exists:
            result["allocations"] = self._extract_multi_instance(field_name, context)
        elif field_name == "concentration_limits" and existence.exists:
            result["limits"] = self._extract_multi_instance(field_name, context)

        # Validate with instructor if available
        schema_class = FIELD_SCHEMA_MAP.get(field_name)
        if self.use_instructor and self.instructor_client and schema_class:
            try:
                return self._validate_with_instructor(field_name, schema_class, result)
            except Exception as e:
                logger.warning(f"Instructor validation failed for {field_name}: {e}")

        return result

    def _validate_with_instructor(
        self,
        field_name: str,
        schema_class: Type[BaseModel],
        raw_result: dict
    ) -> dict:
        """Validate with instructor schema."""
        prompt = f"""Validate and normalize this extraction for {field_name}:

Raw data: {raw_result}

Return the validated data matching the expected schema.
Use null for missing values.
For percentages, use numeric values without % suffix.
"""

        if self.provider == "anthropic":
            result = self.instructor_client.messages.create(
                model=self.model,
                max_tokens=1500,
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
                    {"role": "system", "content": "Validate and normalize extraction data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
            )

        result_dict = result.model_dump(exclude_none=False)
        result_dict.pop("reasoning", None)
        result_dict.pop("confidence", None)
        result_dict.pop("citation", None)

        return result_dict

    def extract_field(
        self,
        field_name: str,
        chunked_doc,
    ) -> Tier3SequentialResult:
        """
        Extract a single field using Tier3 + Sequential approach.

        1. Use Tier3 keyword scoring to find relevant chunks
        2. Check existence on those chunks
        3. If exists, extract details
        4. Synthesize and validate
        """
        logger.info(f"  [Tier3+SeqDocVQA] Extracting {field_name}")

        # Step 0: Get relevant chunks via Tier3 scoring
        context, sections_searched, chunks_retrieved = self._get_relevant_chunks(
            field_name, chunked_doc
        )
        logger.info(f"    Retrieved {chunks_retrieved} chunks from {sections_searched} sections")

        if not context:
            return Tier3SequentialResult(
                field_name=field_name,
                sections_searched=0,
                chunks_retrieved=0,
                existence_check=ExistenceCheckResult(
                    field_name=field_name,
                    exists=None,
                    evidence=None,
                    confidence="not_found"
                ),
                detail_results=[],
                synthesized_value=None,
                confidence="not_found",
                grounded=False,
                questions_asked=0,
                grounded_answers=0
            )

        field_config = FIELD_CONFIGS.get(field_name)
        if not field_config:
            return Tier3SequentialResult(
                field_name=field_name,
                sections_searched=sections_searched,
                chunks_retrieved=chunks_retrieved,
                existence_check=ExistenceCheckResult(
                    field_name=field_name,
                    exists=None,
                    evidence=None,
                    confidence="not_found"
                ),
                detail_results=[],
                synthesized_value=None,
                confidence="not_found",
                grounded=False,
                questions_asked=0,
                grounded_answers=0
            )

        # Step 1: Check existence
        existence = self._check_existence(field_name, context)
        questions_asked = 1
        grounded_answers = 1 if existence.evidence else 0

        logger.info(f"    Existence: {existence.exists} (grounded={existence.evidence is not None})")

        # Step 2: Extract details if exists
        detail_results = []
        if existence.exists:
            for detail_config in field_config.get("detail_questions", []):
                # Skip multi-instance fields
                if detail_config["sub_field"] in ["class_details", "allocations", "limits"]:
                    questions_asked += 1
                    continue

                detail = self._extract_detail(field_name, detail_config, context)
                detail_results.append(detail)
                questions_asked += 1
                if detail.grounded:
                    grounded_answers += 1

        # Step 3: Synthesize
        synthesized = self._synthesize_field(
            field_name, existence, detail_results, context
        )

        # Determine confidence
        if existence.exists is None:
            confidence = "not_found"
        elif grounded_answers >= questions_asked / 2:
            confidence = "explicit"
        else:
            confidence = "inferred"

        grounded = grounded_answers > 0

        logger.info(
            f"    Result: {grounded_answers}/{questions_asked} grounded, "
            f"confidence={confidence}"
        )

        return Tier3SequentialResult(
            field_name=field_name,
            sections_searched=sections_searched,
            chunks_retrieved=chunks_retrieved,
            existence_check=existence,
            detail_results=detail_results,
            synthesized_value=synthesized,
            confidence=confidence,
            grounded=grounded,
            questions_asked=questions_asked,
            grounded_answers=grounded_answers
        )

    def extract_all_fields(
        self,
        chunked_doc,
        fields: Optional[list[str]] = None
    ) -> dict[str, Tier3SequentialResult]:
        """
        Extract all fields using Tier3 + Sequential approach.

        Args:
            chunked_doc: ChunkedDocument with sections and chunks
            fields: List of fields to extract (defaults to all)

        Returns:
            Dict mapping field_name -> Tier3SequentialResult
        """
        if fields is None:
            fields = list(FIELD_CONFIGS.keys())

        results = {}
        for field_name in fields:
            logger.info(f"[Tier3+SeqDocVQA] Processing {field_name}")
            results[field_name] = self.extract_field(field_name, chunked_doc)

        return results


def convert_tier3_sequential_to_extraction_format(
    results: dict[str, Tier3SequentialResult]
) -> dict:
    """
    Convert Tier3+Sequential results to standard extraction format.
    """
    extraction = {}

    for field_name, result in results.items():
        if result.synthesized_value:
            extraction[field_name] = result.synthesized_value
        else:
            extraction[field_name] = None

    return extraction
