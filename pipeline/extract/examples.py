"""
Few-shot example library for LLM extraction.

This module provides:
- Structured example definitions for each field category
- Example loading from YAML files for easy editing
- Formatting utilities for prompt construction

Examples can be:
1. Defined inline in this file (for core examples)
2. Loaded from YAML files in data/examples/ (for easy editing)
3. Added programmatically from real extractions

Usage:
    from pipeline.extract.examples import get_examples_for_field, format_examples_for_prompt

    examples = get_examples_for_field("repurchase_terms")
    formatted = format_examples_for_prompt(examples)
"""

import json
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from enum import Enum


class FieldCategory(str, Enum):
    """Categories of extraction fields."""
    FEES = "fees"
    REPURCHASE = "repurchase"
    SHARE_CLASSES = "share_classes"
    ALLOCATION = "allocation"
    CONCENTRATION = "concentration"
    FUND_METADATA = "fund_metadata"


@dataclass
class ExtractionExample:
    """
    A single few-shot example for extraction.

    Attributes:
        source_text: The exact text from the filing (copy-paste from document)
        extraction: The correct extraction result as a dictionary
        field_category: Which category this example belongs to
        fund_name: Source fund (e.g., "StepStone", "Blackstone")
        filing_type: Filing type (e.g., "N-2", "N-CSR")
        section_title: Section where this text was found
        notes: Explanation of why this example is useful (edge cases, etc.)
        difficulty: How challenging this example is ("easy", "medium", "hard")
    """
    source_text: str
    extraction: dict[str, Any]
    field_category: FieldCategory
    fund_name: str = ""
    filing_type: str = "N-2"
    section_title: str = ""
    notes: str = ""
    difficulty: str = "medium"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d["field_category"] = self.field_category.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractionExample":
        """Create from dictionary."""
        if isinstance(data.get("field_category"), str):
            data["field_category"] = FieldCategory(data["field_category"])
        return cls(**data)

    def format_for_prompt(self) -> str:
        """Format this example for inclusion in a prompt."""
        extraction_json = json.dumps(self.extraction, indent=2, default=str)

        parts = [
            "--- EXAMPLE ---",
            f"Source text:\n\"\"\"\n{self.source_text}\n\"\"\"",
            f"\nCorrect extraction:\n```json\n{extraction_json}\n```",
        ]

        if self.notes:
            parts.append(f"\nNote: {self.notes}")

        parts.append("--- END EXAMPLE ---")

        return "\n".join(parts)


@dataclass
class ExampleLibrary:
    """
    Collection of examples organized by field category.

    Examples can be loaded from:
    1. Inline definitions (BUILTIN_EXAMPLES)
    2. YAML files in data/examples/
    3. Added programmatically
    """
    examples: dict[FieldCategory, list[ExtractionExample]] = field(default_factory=dict)

    def add_example(self, example: ExtractionExample) -> None:
        """Add an example to the library."""
        if example.field_category not in self.examples:
            self.examples[example.field_category] = []
        self.examples[example.field_category].append(example)

    def get_examples(
        self,
        category: FieldCategory,
        max_examples: int = 3,
        difficulty: Optional[str] = None,
    ) -> list[ExtractionExample]:
        """
        Get examples for a category.

        Args:
            category: Field category
            max_examples: Maximum number to return
            difficulty: Filter by difficulty if specified

        Returns:
            List of examples
        """
        examples = self.examples.get(category, [])

        if difficulty:
            examples = [e for e in examples if e.difficulty == difficulty]

        return examples[:max_examples]

    def save_to_yaml(self, path: Path) -> None:
        """Save all examples to a YAML file."""
        data = {}
        for category, examples in self.examples.items():
            data[category.value] = [e.to_dict() for e in examples]

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    @classmethod
    def load_from_yaml(cls, path: Path) -> "ExampleLibrary":
        """Load examples from a YAML file."""
        library = cls()

        if not path.exists():
            return library

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        for category_str, examples_data in data.items():
            category = FieldCategory(category_str)
            for example_data in examples_data:
                example = ExtractionExample.from_dict(example_data)
                library.add_example(example)

        return library

    def count_examples(self) -> dict[str, int]:
        """Count examples by category."""
        return {cat.value: len(exs) for cat, exs in self.examples.items()}


# =============================================================================
# BUILTIN EXAMPLES
# These are starter examples. Replace/augment with real filing extracts.
# =============================================================================

BUILTIN_EXAMPLES: list[ExtractionExample] = [
    # ─── REPURCHASE EXAMPLES ───
    ExtractionExample(
        field_category=FieldCategory.REPURCHASE,
        fund_name="StepStone",
        filing_type="N-2",
        section_title="REPURCHASES OF SHARES",
        source_text="""REPURCHASES OF SHARES

The Fund is an "interval fund," a type of fund that, in order to provide some
liquidity to shareholders, has adopted a fundamental policy to make quarterly
repurchase offers for between 5% and 25% of the Fund's outstanding Shares at
their NAV. Subject to applicable law and approval of the Board, for each
quarterly repurchase offer, the Fund currently expects to offer to repurchase
5% of the Fund's outstanding Shares at their NAV.

Shareholders will be notified in writing at least 21 days before each quarterly
repurchase offer. Payment for repurchased Shares will be made within seven days
after the repurchase pricing date.""",
        extraction={
            "fund_structure": "interval_fund",
            "repurchase_frequency": "quarterly",
            "repurchase_percentage_min": 5,
            "repurchase_percentage_max": 25,
            "repurchase_percentage_typical": 5,
            "notice_period_days": 21,
            "pricing_date_description": "repurchase pricing date",
            "confidence": "explicit",
            "citation": "quarterly repurchase offers for between 5% and 25%"
        },
        notes="Standard interval fund with explicit min/max range and typical offer",
        difficulty="easy",
    ),

    ExtractionExample(
        field_category=FieldCategory.REPURCHASE,
        fund_name="Blackstone",
        filing_type="N-2",
        section_title="REPURCHASES OF SHARES",
        source_text="""REPURCHASES OF SHARES

The Fund operates as an "interval fund" pursuant to Rule 23c-3 under the 1940 Act
and, as such, has adopted a fundamental policy to offer to repurchase at least 5%
of its outstanding Shares at their NAV on a quarterly basis.

A 2.00% early repurchase fee will be charged by the Fund with respect to any
repurchase of Shares from a shareholder at any time prior to the day immediately
preceding the one-year anniversary of a shareholder's purchase of Shares.

A shareholder who tenders for repurchase only a portion of the shareholder's
Shares in the Fund will be required to maintain a minimum account balance of $500.""",
        extraction={
            "fund_structure": "interval_fund",
            "repurchase_frequency": "quarterly",
            "repurchase_percentage_min": 5,
            "repurchase_percentage_max": None,
            "early_repurchase_fee": 2.0,
            "lock_up_period_days": 365,
            "minimum_balance_after_repurchase": 500,
            "confidence": "explicit",
            "citation": "offer to repurchase at least 5% of its outstanding Shares"
        },
        notes="Has early repurchase fee (lock-up proxy) and minimum balance requirement",
        difficulty="medium",
    ),

    ExtractionExample(
        field_category=FieldCategory.REPURCHASE,
        fund_name="Example Tender Fund",
        filing_type="N-2",
        section_title="TENDER OFFERS",
        source_text="""TENDER OFFERS

The Fund is a "tender offer fund." The Fund's Board may, from time to time and
in its sole discretion, cause the Fund to offer to repurchase Shares from
shareholders. The Fund is not required to conduct tender offers, and there is
no guarantee that shareholders will be able to sell any or all of their Shares.

When a tender offer is conducted, the Fund currently expects to offer to
repurchase Shares at their NAV, although it reserves the right to offer to
repurchase Shares at a different price.""",
        extraction={
            "fund_structure": "tender_offer_fund",
            "repurchase_frequency": "discretionary",
            "repurchase_percentage_min": None,
            "repurchase_percentage_max": None,
            "pricing_date_description": "at their NAV",
            "confidence": "explicit",
            "citation": "Board may, from time to time and in its sole discretion"
        },
        notes="Tender offer fund - discretionary repurchases, no guaranteed frequency",
        difficulty="medium",
    ),

    # ─── SHARE CLASS EXAMPLES ───
    ExtractionExample(
        field_category=FieldCategory.SHARE_CLASSES,
        fund_name="StepStone",
        filing_type="N-2",
        section_title="PLAN OF DISTRIBUTION",
        source_text="""PLAN OF DISTRIBUTION

Class S Shares
Class S shares are sold at NAV plus a maximum sales load of 3.50% of the offering
price. Class S shares are subject to an ongoing distribution fee of 0.85% per annum
of the Fund's average daily net assets attributable to Class S shares. The minimum
initial investment for Class S shares is $5,000, with a minimum subsequent
investment of $5,000.

Class D Shares
Class D shares are offered at NAV without any sales load. Class D shares are not
subject to an ongoing distribution fee. The minimum initial investment for Class D
shares is $5,000.

Class I Shares
Class I shares are offered at NAV without any sales charge and are not subject to
a distribution fee. The minimum initial investment in Class I shares is $1,000,000,
with a minimum subsequent investment of $100,000. Class I shares are generally
available for purchase only by eligible institutional investors.""",
        extraction={
            "share_classes": [
                {
                    "class_name": "Class S",
                    "sales_load_pct": 3.5,
                    "distribution_servicing_fee_pct": 0.85,
                    "minimum_initial_investment": 5000,
                    "minimum_additional_investment": 5000,
                    "investor_eligibility": None
                },
                {
                    "class_name": "Class D",
                    "sales_load_pct": 0,
                    "distribution_servicing_fee_pct": 0.25,  # Shareholder servicing fee
                    "minimum_initial_investment": 5000,
                    "minimum_additional_investment": None,
                    "investor_eligibility": None
                },
                {
                    "class_name": "Class I",
                    "sales_load_pct": 0,
                    "distribution_servicing_fee_pct": 0,
                    "minimum_initial_investment": 1000000,
                    "minimum_additional_investment": 100000,
                    "investor_eligibility": "eligible institutional investors"
                }
            ],
            "confidence": "explicit"
        },
        notes="Three share classes with combined distribution/servicing fees",
        difficulty="easy",
    ),

    # Example: combining distribution and servicing fees into one field
    ExtractionExample(
        field_category=FieldCategory.SHARE_CLASSES,
        fund_name="StepStone",
        filing_type="N-2",
        section_title="FEES Q&A",
        source_text="""What are the fees that investors pay with respect to the Shares?

There are two types of fees:
- First, for Class S Shares, there are shareholder transaction expenses that are a
  one-time upfront fee calculated as a percentage of the offering price. Class S
  Shares have a maximum selling commission of 3.50%.
- Second, for Class S Shares and Class D Shares, there are ongoing distribution and
  shareholder servicing fees that are calculated as a percentage of net asset value.
  The Class S Shares have annual distribution and shareholder servicing fees of 0.85%,
  and Class D Shares have annual shareholder servicing fees of 0.25%.

Class I Shares are not subject to any sales load, distribution fee, or shareholder
servicing fee.""",
        extraction={
            "share_classes": [
                {
                    "class_name": "Class S",
                    "sales_load_pct": 3.5,
                    "distribution_servicing_fee_pct": 0.85,  # Combined distribution + servicing
                    "minimum_initial_investment": None,
                },
                {
                    "class_name": "Class D",
                    "sales_load_pct": 0,
                    "distribution_servicing_fee_pct": 0.25,  # Shareholder servicing fee counts
                    "minimum_initial_investment": None,
                },
                {
                    "class_name": "Class I",
                    "sales_load_pct": 0,
                    "distribution_servicing_fee_pct": 0,
                    "minimum_initial_investment": None,
                }
            ],
            "confidence": "explicit"
        },
        notes=(
            "Both 'distribution fee' and 'shareholder servicing fee' go into distribution_servicing_fee_pct. "
            "Class S: 0.85% (combined). Class D: 0.25% (servicing only, but still counts). Class I: 0."
        ),
        difficulty="medium",
    ),

    ExtractionExample(
        field_category=FieldCategory.SHARE_CLASSES,
        fund_name="Blackstone",
        filing_type="N-2",
        section_title="PLAN OF DISTRIBUTION",
        source_text="""The Fund currently offers four classes of Shares: Class S, Class D, Class I
and Class I Advisory.

Class S shares: Maximum sales load of 3.5% of the offering price. Annual
distribution fee of 0.75% plus shareholder servicing fee of 0.10%. Minimum
initial investment of $2,500 and minimum subsequent investment of $500.

Class D shares: Maximum sales load of 1.5%. Annual distribution fee of 0.25%.
Minimum initial investment of $2,500 and minimum subsequent investment of $500.

Class I shares: No sales load. No distribution fee. Minimum initial investment
of $1,000,000. Minimum subsequent investment of $500.

Class I Advisory: No sales load. No distribution fee. Minimum initial investment
of $1,000,000. Available only through fee-based programs.""",
        extraction={
            "share_classes": [
                {
                    "class_name": "Class S",
                    "sales_load_pct": 3.5,
                    "distribution_servicing_fee_pct": 0.85,  # 0.75% dist + 0.10% servicing
                    "minimum_initial_investment": 2500,
                    "minimum_additional_investment": 500
                },
                {
                    "class_name": "Class D",
                    "sales_load_pct": 1.5,
                    "distribution_servicing_fee_pct": 0.25,
                    "minimum_initial_investment": 2500,
                    "minimum_additional_investment": 500
                },
                {
                    "class_name": "Class I",
                    "sales_load_pct": 0,
                    "distribution_servicing_fee_pct": 0,
                    "minimum_initial_investment": 1000000,
                    "minimum_additional_investment": 500
                },
                {
                    "class_name": "Class I Advisory",
                    "sales_load_pct": 0,
                    "distribution_servicing_fee_pct": 0,
                    "minimum_initial_investment": 1000000,
                    "distribution_channel": "fee-based programs"
                }
            ],
            "confidence": "explicit"
        },
        notes="Four share classes; Class S has 0.75% dist + 0.10% servicing = 0.85% combined",
        difficulty="medium",
    ),

    # Blue Owl-style: minimums delegated to intermediaries (CRITICAL: minimums should be null)
    ExtractionExample(
        field_category=FieldCategory.SHARE_CLASSES,
        fund_name="Blue Owl",
        filing_type="N-2",
        section_title="PLAN OF DISTRIBUTION",
        source_text="""The Fund offers three classes of Shares: Class I, Class S and Class U.

Class I Shares: Available to eligible institutional investors and certain other
investors. Class I Shares are not subject to any sales load, upfront placement fees,
or distribution and servicing fees.

Class S Shares: Available through various distribution channels. Class S Shares are
subject to a distribution and shareholder servicing fee of 0.85% per year. If Class S
Shares are purchased through certain financial intermediaries, those financial
intermediaries may directly charge transaction or other fees, including upfront
placement fees or brokerage commissions, in such amount as they may determine,
provided that the selling agents limit such charges to 3.50% of the net offering price.

Class U Shares: Available through various distribution channels. Class U Shares are
subject to a distribution and shareholder servicing fee of 0.75% per year. If Class U
Shares are purchased through certain financial intermediaries, those financial
intermediaries may charge placement fees up to 3.00% of the net offering price.

The Fund does not impose minimum investment requirements at the Fund level. Minimum
investment amounts, if any, will be determined by the financial intermediaries through
which investors purchase Shares. Investors should consult their financial intermediary
for information about applicable minimum investment requirements.""",
        extraction={
            "share_classes": [
                {
                    "class_name": "Class I",
                    "sales_load_pct": 0,
                    "distribution_servicing_fee_pct": 0,
                    # CRITICAL: null because no fund-level minimum specified
                    "minimum_initial_investment": None,
                    "minimum_additional_investment": None,
                    "investor_eligibility": "eligible institutional investors"
                },
                {
                    "class_name": "Class S",
                    "sales_load_pct": 3.5,  # Maximum intermediary fee
                    "distribution_servicing_fee_pct": 0.85,
                    # CRITICAL: null because minimums are "determined by financial intermediaries"
                    "minimum_initial_investment": None,
                    "minimum_additional_investment": None
                },
                {
                    "class_name": "Class U",
                    "sales_load_pct": 3.0,  # Maximum intermediary fee
                    "distribution_servicing_fee_pct": 0.75,
                    # CRITICAL: null because minimums are "determined by financial intermediaries"
                    "minimum_initial_investment": None,
                    "minimum_additional_investment": None
                }
            ],
            "confidence": "explicit"
        },
        notes=(
            "CRITICAL EXAMPLE: Fund does NOT impose minimum investments at fund level - "
            "they are 'determined by financial intermediaries'. ALL minimum_investment fields "
            "MUST be null, not typical values like 2500 or 1000000. Each fund is different!"
        ),
        difficulty="hard",
    ),

    # ─── ALLOCATION EXAMPLES ───
    ExtractionExample(
        field_category=FieldCategory.ALLOCATION,
        fund_name="StepStone",
        filing_type="N-2",
        section_title="INVESTMENT STRATEGY",
        source_text="""INVESTMENT STRATEGY

The Fund's investment strategy is to provide diversified exposure to private
markets through investments in:

Secondary Investments: The Fund expects to allocate approximately 40% to 70%
of its assets to secondary investments.

Direct Investments: The Fund expects to allocate approximately 20% to 50% of
its assets to direct co-investments alongside other investors.

Primary Fund Commitments: The Fund expects to allocate approximately 0% to 15%
of its assets to primary fund investments.

The allocation among investment types may vary over time based on market conditions.""",
        extraction={
            "allocations": [
                {
                    "asset_class": "Secondary Investments",
                    "range_min": 40,
                    "range_max": 70,
                    "target_percentage": None
                },
                {
                    "asset_class": "Direct Investments",
                    "range_min": 20,
                    "range_max": 50,
                    "target_percentage": None
                },
                {
                    "asset_class": "Primary Fund Commitments",
                    "range_min": 0,
                    "range_max": 15,
                    "target_percentage": None
                }
            ],
            "allocation_approach": "opportunistic",
            "confidence": "explicit",
            "citation": "40% to 70% of its assets to secondary investments"
        },
        notes="Explicit percentage ranges for investment type allocation",
        difficulty="easy",
    ),

    ExtractionExample(
        field_category=FieldCategory.ALLOCATION,
        fund_name="StepStone",
        filing_type="N-2",
        section_title="ASSET CLASS ALLOCATION",
        source_text="""The Fund targets the following asset class allocations:

Private Equity: 60-80% of the Fund's investments
Real Assets: 15-30% of the Fund's investments
Private Debt: 5-15% of the Fund's investments

The Fund's geographic focus is primarily on North America (70-80%), with
additional exposure to Europe (10-20%) and other regions (5-10%).""",
        extraction={
            "allocations": [
                {"asset_class": "Private Equity", "range_min": 60, "range_max": 80},
                {"asset_class": "Real Assets", "range_min": 15, "range_max": 30},
                {"asset_class": "Private Debt", "range_min": 5, "range_max": 15}
            ],
            "geography_allocations": [
                {"region": "North America", "range_min": 70, "range_max": 80},
                {"region": "Europe", "range_min": 10, "range_max": 20},
                {"region": "Other", "range_min": 5, "range_max": 10}
            ],
            "confidence": "explicit"
        },
        notes="Combined asset class and geography targets",
        difficulty="medium",
    ),

    ExtractionExample(
        field_category=FieldCategory.ALLOCATION,
        fund_name="Blackstone",
        filing_type="N-2",
        section_title="INVESTMENT OBJECTIVES",
        source_text="""The Fund may invest up to 100% of its Managed Assets in private credit
investments and up to 100% of its Managed Assets in liquid credit investments.
The Fund does not have a minimum allocation to either strategy.

The Fund may invest without limit in securities of issuers located outside the
United States, including emerging market securities.""",
        extraction={
            "allocations": [
                {"asset_class": "Private Credit", "range_min": 0, "range_max": 100},
                {"asset_class": "Liquid Credit", "range_min": 0, "range_max": 100}
            ],
            "geography_restrictions": "no limit on non-US or emerging markets",
            "confidence": "explicit",
            "citation": "may invest up to 100% of its Managed Assets"
        },
        notes="Wide 0-100% ranges indicate flexibility, not specific targets",
        difficulty="hard",
    ),

    # ─── CONCENTRATION EXAMPLES ───
    ExtractionExample(
        field_category=FieldCategory.CONCENTRATION,
        fund_name="StepStone",
        filing_type="N-2",
        section_title="INVESTMENT RESTRICTIONS",
        source_text="""INVESTMENT RESTRICTIONS

The Fund will not:
(1) Invest more than 25% of its net assets in any single portfolio company or
    co-investment opportunity;
(2) Invest more than 25% of its net assets in any single underlying fund;
(3) Invest more than 25% of its net assets in any single industry or sector;
(4) Concentrate its investments in any particular industry, except that this
    restriction does not apply to investments in private investment funds.

The Fund has no geographic concentration limits.""",
        extraction={
            "limits": [
                {"limit_type": "single_issuer", "limit_percentage": 25},
                {"limit_type": "single_fund", "limit_percentage": 25},
                {"limit_type": "single_sector", "limit_percentage": 25}
            ],
            "geographic_limit": None,
            "confidence": "explicit",
            "citation": "not invest more than 25% of its net assets in any single"
        },
        notes="Standard 25% limits across multiple categories",
        difficulty="easy",
    ),

    ExtractionExample(
        field_category=FieldCategory.CONCENTRATION,
        fund_name="Blackstone",
        filing_type="N-2",
        section_title="INVESTMENT POLICIES",
        source_text="""The Fund does not have any specific limits on concentration in any
single issuer, industry, or geographic region. The Adviser seeks to manage
concentration risk through diversification across multiple investments,
strategies, and sectors.

The Fund may use derivatives for hedging purposes and, to a limited extent,
for investment purposes.""",
        extraction={
            "limits": [],
            "diversification_policy": "diversification across multiple investments, strategies, and sectors",
            "has_concentration_limits": False,
            "derivatives_allowed": True,
            "derivative_use": "hedging and limited investment purposes",
            "confidence": "explicit",
            "citation": "does not have any specific limits on concentration"
        },
        notes="No explicit limits - extract the absence of limits as meaningful",
        difficulty="hard",
    ),

    # ─── FEES EXAMPLES ───
    ExtractionExample(
        field_category=FieldCategory.FEES,
        fund_name="Example Fund",
        filing_type="N-2",
        section_title="FEES AND EXPENSES",
        source_text="""Management Fee: The Fund pays the Adviser a management fee at an annual
rate of 1.25% of the Fund's average daily net assets.

Incentive Fee: The Adviser is also entitled to an incentive fee equal to 12.5%
of the Fund's net investment income for each calendar quarter, subject to a
hurdle rate equal to 1.5% per quarter (6% annualized) and a high water mark.

The Adviser has agreed to waive fees and reimburse expenses to the extent
necessary to limit total annual fund operating expenses to 2.00% of net assets
through March 31, 2026.""",
        extraction={
            "management_fee_rate": 1.25,
            "management_fee_basis": "average daily net assets",
            "has_incentive_fee": True,
            "incentive_fee_rate": 12.5,
            "hurdle_rate": 6.0,
            "hurdle_rate_basis": "annualized",
            "high_water_mark": True,
            "has_expense_cap": True,
            "expense_cap_rate": 2.0,
            "expense_cap_expiration": "March 31, 2026",
            "confidence": "explicit"
        },
        notes="Full fee structure with management, incentive, and expense cap",
        difficulty="medium",
    ),

    ExtractionExample(
        field_category=FieldCategory.FEES,
        fund_name="Fund-of-Funds Example",
        filing_type="N-2",
        section_title="FEES AND EXPENSES",
        source_text="""The Fund does not charge a performance-based fee at the fund level. However,
the underlying private funds in which the Fund invests typically charge
management fees ranging from 1.0% to 2.0% and incentive fees or carried
interest of approximately 15% to 20% of net profits.

Acquired Fund Fees and Expenses (AFFE): 2.79%

The Fund's management fee is 1.40% of net assets. There is no sales load for
direct purchases.""",
        extraction={
            "management_fee_rate": 1.4,
            "has_incentive_fee": False,
            "underlying_fund_management_range": "1.0% to 2.0%",
            "underlying_fund_incentive_range": "15% to 20%",
            "affe": 2.79,
            "confidence": "explicit",
            "citation": "does not charge a performance-based fee at the fund level"
        },
        notes="Fund-of-funds: distinguish fund-level vs underlying fund fees",
        difficulty="hard",
    ),
]


# =============================================================================
# GLOBAL EXAMPLE LIBRARY
# =============================================================================

_LIBRARY: Optional[ExampleLibrary] = None


def _get_library() -> ExampleLibrary:
    """Get or initialize the global example library."""
    global _LIBRARY

    if _LIBRARY is None:
        _LIBRARY = ExampleLibrary()

        # Add builtin examples
        for example in BUILTIN_EXAMPLES:
            _LIBRARY.add_example(example)

        # Try to load additional examples from YAML
        examples_dir = Path(__file__).parent.parent.parent / "data" / "examples"
        if examples_dir.exists():
            for yaml_file in examples_dir.glob("*.yaml"):
                try:
                    additional = ExampleLibrary.load_from_yaml(yaml_file)
                    for category, examples in additional.examples.items():
                        for example in examples:
                            _LIBRARY.add_example(example)
                except Exception as e:
                    print(f"Warning: Failed to load examples from {yaml_file}: {e}")

    return _LIBRARY


def get_examples_for_field(
    field_name: str,
    max_examples: int = 3,
    document_text: Optional[str] = None,
) -> list[ExtractionExample]:
    """
    Get relevant examples for a field, optionally using dynamic selection.

    Args:
        field_name: The field name (e.g., "repurchase_terms", "share_classes")
        max_examples: Maximum number of examples to return
        document_text: Optional document text for dynamic example selection

    Returns:
        List of relevant examples
    """
    # Map field names to categories
    field_to_category = {
        "incentive_fee": FieldCategory.FEES,
        "expense_cap": FieldCategory.FEES,
        "management_fee": FieldCategory.FEES,
        "repurchase_terms": FieldCategory.REPURCHASE,
        "allocation_targets": FieldCategory.ALLOCATION,
        "concentration_limits": FieldCategory.CONCENTRATION,
        "share_classes": FieldCategory.SHARE_CLASSES,
        "minimum_investment": FieldCategory.SHARE_CLASSES,
    }

    category = field_to_category.get(field_name)
    if category is None:
        return []

    library = _get_library()
    all_examples = library.get_examples(category, max_examples=100)  # Get all

    if document_text and all_examples:
        # Use dynamic selection based on document characteristics
        return select_examples_dynamically(all_examples, document_text, max_examples)

    return all_examples[:max_examples]


def select_examples_dynamically(
    examples: list[ExtractionExample],
    document_text: str,
    max_examples: int = 3,
) -> list[ExtractionExample]:
    """
    Select examples most relevant to the current document.

    Scores examples based on:
    1. Fund structure match (interval vs tender offer)
    2. Similar terminology/keywords
    3. Edge case relevance (e.g., "no limit" examples when doc has "no limit")

    Args:
        examples: All available examples for the field
        document_text: The document text to match against
        max_examples: Number of examples to return

    Returns:
        Sorted list of most relevant examples
    """
    if not examples:
        return []

    text_lower = document_text.lower()
    scored_examples = []

    for example in examples:
        score = 0.0
        example_text_lower = example.source_text.lower()

        # ─── Fund Structure Matching ───
        # Interval fund matching
        doc_is_interval = "interval fund" in text_lower or "rule 23c-3" in text_lower
        example_is_interval = "interval fund" in example_text_lower or "rule 23c-3" in example_text_lower

        # Tender offer matching
        doc_is_tender = "tender offer" in text_lower and "interval" not in text_lower
        example_is_tender = "tender offer" in example_text_lower and "interval" not in example_text_lower

        # Boost for matching fund structure
        if doc_is_interval and example_is_interval:
            score += 3.0
        elif doc_is_tender and example_is_tender:
            score += 3.0
        elif doc_is_interval and example_is_tender:
            score -= 1.0  # Penalize mismatch
        elif doc_is_tender and example_is_interval:
            score -= 1.0

        # ─── Edge Case Matching ───
        # "No limit" patterns
        no_limit_patterns = ["no limit", "no restriction", "unlimited", "without limit"]
        doc_has_no_limit = any(p in text_lower for p in no_limit_patterns)
        example_has_no_limit = any(p in example_text_lower for p in no_limit_patterns)

        if doc_has_no_limit and example_has_no_limit:
            score += 2.5  # Important to show how to handle "no limit"

        # "Does not" / negative patterns
        negative_patterns = ["does not", "will not", "not subject to", "no sales load"]
        doc_has_negative = any(p in text_lower for p in negative_patterns)
        example_has_negative = any(p in example_text_lower for p in negative_patterns)

        if doc_has_negative and example_has_negative:
            score += 1.5

        # ─── Fund-of-Funds Detection ───
        fof_patterns = ["underlying fund", "acquired fund", "affe", "fund-of-funds"]
        doc_is_fof = any(p in text_lower for p in fof_patterns)
        example_is_fof = any(p in example_text_lower for p in fof_patterns)

        if doc_is_fof and example_is_fof:
            score += 2.0

        # ─── Keyword Overlap ───
        # Common financial terms
        keywords = [
            "quarterly", "semi-annual", "annual", "monthly",
            "management fee", "incentive fee", "performance fee",
            "hurdle rate", "high water mark",
            "class s", "class d", "class i",
            "minimum investment", "sales load",
            "25%", "5%", "80%",
        ]
        overlap = sum(1 for kw in keywords if kw in text_lower and kw in example_text_lower)
        score += overlap * 0.3

        # ─── Difficulty Preference ───
        # Prefer medium difficulty as baseline, then easy for explicit, hard for ambiguous
        if example.difficulty == "medium":
            score += 0.5
        elif example.difficulty == "easy" and "explicit" in text_lower:
            score += 0.3
        elif example.difficulty == "hard" and doc_has_no_limit:
            score += 0.8  # Hard examples for tricky cases

        # ─── Source Fund Diversity ───
        # (Applied later to ensure variety)

        scored_examples.append((score, example))

    # Sort by score descending
    scored_examples.sort(key=lambda x: x[0], reverse=True)

    # Select top examples with diversity consideration
    selected = []
    seen_funds = set()

    for score, example in scored_examples:
        if len(selected) >= max_examples:
            break

        # Try to include examples from different funds for diversity
        if example.fund_name in seen_funds and len(selected) < max_examples - 1:
            # Skip if we've already seen this fund (unless we need to fill slots)
            continue

        selected.append(example)
        seen_funds.add(example.fund_name)

    # If we didn't get enough due to diversity constraints, fill remaining
    if len(selected) < max_examples:
        for score, example in scored_examples:
            if example not in selected:
                selected.append(example)
                if len(selected) >= max_examples:
                    break

    return selected


def detect_document_characteristics(text: str) -> dict:
    """
    Detect key characteristics of a document for example selection.

    Returns a dict with detected features:
    - fund_structure: "interval_fund", "tender_offer_fund", or "unknown"
    - is_fund_of_funds: bool
    - has_no_limit_language: bool
    - has_explicit_percentages: bool
    """
    text_lower = text.lower()

    return {
        "fund_structure": (
            "interval_fund" if ("interval fund" in text_lower or "rule 23c-3" in text_lower)
            else "tender_offer_fund" if "tender offer" in text_lower
            else "unknown"
        ),
        "is_fund_of_funds": any(p in text_lower for p in ["underlying fund", "acquired fund", "affe"]),
        "has_no_limit_language": any(p in text_lower for p in ["no limit", "no restriction", "unlimited"]),
        "has_explicit_percentages": bool(any(f"{i}%" in text for i in range(1, 100))),
    }


def format_examples_for_prompt(
    examples: list[ExtractionExample],
    include_notes: bool = True,
) -> str:
    """
    Format examples for inclusion in a prompt.

    Args:
        examples: List of examples to format
        include_notes: Whether to include explanatory notes

    Returns:
        Formatted string for prompt
    """
    if not examples:
        return ""

    parts = ["\n## Examples\n"]

    for i, example in enumerate(examples, 1):
        parts.append(f"\n### Example {i}")
        if example.fund_name:
            parts.append(f"(from {example.fund_name} {example.filing_type})")
        parts.append("\n")

        parts.append(f"**Source text:**\n```\n{example.source_text}\n```\n")

        extraction_json = json.dumps(example.extraction, indent=2, default=str)
        parts.append(f"**Correct extraction:**\n```json\n{extraction_json}\n```\n")

        if include_notes and example.notes:
            parts.append(f"**Note:** {example.notes}\n")

    return "\n".join(parts)


def add_example(example: ExtractionExample) -> None:
    """Add an example to the global library."""
    library = _get_library()
    library.add_example(example)


def save_examples_to_yaml(output_path: Optional[Path] = None) -> Path:
    """
    Save all examples to a YAML file for editing.

    Args:
        output_path: Output path (default: data/examples/all_examples.yaml)

    Returns:
        Path where examples were saved
    """
    if output_path is None:
        output_path = Path(__file__).parent.parent.parent / "data" / "examples" / "all_examples.yaml"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    library = _get_library()
    library.save_to_yaml(output_path)

    return output_path


def reload_examples() -> None:
    """Reload examples from files (call after editing YAML)."""
    global _LIBRARY
    _LIBRARY = None
    _get_library()  # Reinitialize


def get_example_counts() -> dict[str, int]:
    """Get count of examples by category."""
    library = _get_library()
    return library.count_examples()


# =============================================================================
# HELPER: Create example from real extraction
# =============================================================================

def create_example_from_extraction(
    source_text: str,
    extraction_result: dict,
    field_category: FieldCategory,
    fund_name: str = "",
    section_title: str = "",
    notes: str = "",
    difficulty: str = "medium",
) -> ExtractionExample:
    """
    Create a new example from a real extraction result.

    Use this after manually verifying an extraction is correct.

    Args:
        source_text: The source text that was extracted from
        extraction_result: The verified correct extraction
        field_category: Category of the field
        fund_name: Source fund name
        section_title: Section where text was found
        notes: Explanation of edge cases or important aspects
        difficulty: "easy", "medium", or "hard"

    Returns:
        ExtractionExample ready to add to library
    """
    return ExtractionExample(
        source_text=source_text,
        extraction=extraction_result,
        field_category=field_category,
        fund_name=fund_name,
        section_title=section_title,
        notes=notes,
        difficulty=difficulty,
    )
