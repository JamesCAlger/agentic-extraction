"""
Pydantic schemas for LLM extraction.

These schemas define the structured output format for each extraction type.
Used with the `instructor` library for reliable structured extraction.

Schema Version: 2.0 - Expanded for complete ground truth coverage
"""

from decimal import Decimal
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    """Confidence in the extracted value."""
    EXPLICIT = "explicit"      # Value directly stated
    INFERRED = "inferred"      # Calculated or derived from context
    NOT_FOUND = "not_found"    # Could not find in provided text


class Citation(BaseModel):
    """Source citation for an extracted value."""
    evidence_quote: str = Field(
        description="Verbatim quote from the text supporting this value (max 200 chars)"
    )
    section_title: Optional[str] = Field(
        default=None,
        description="Title of the section where this was found"
    )


class ReasoningStep(BaseModel):
    """A single step in chain-of-thought reasoning."""
    step: str = Field(description="What you're looking for or checking")
    observation: str = Field(description="What you found in the text")
    conclusion: str = Field(description="What this means for the extraction")


class ExtractionReasoning(BaseModel):
    """Chain-of-thought reasoning for an extraction."""
    search_strategy: Optional[str] = Field(
        default=None,
        description="Brief description of how you searched for this information"
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description="Key phrases or values found in the text (2-4 items)"
    )
    reasoning_steps: list[ReasoningStep] = Field(
        default_factory=list,
        description="Step-by-step reasoning process (1-3 steps)"
    )
    extraction_rationale: Optional[str] = Field(
        default=None,
        description="Brief explanation of why you extracted these specific values"
    )


# =============================================================================
# FUND METADATA SCHEMAS (NEW - v2.0)
# =============================================================================

class FundMetadataExtraction(BaseModel):
    """Extract basic fund metadata."""

    # Chain-of-thought reasoning
    reasoning: Optional[ExtractionReasoning] = Field(
        default=None,
        description="Step-by-step reasoning about fund metadata"
    )

    fund_name: Optional[str] = Field(
        default=None,
        description="Full legal name of the fund"
    )
    fund_manager: Optional[str] = Field(
        default=None,
        description="Name of the fund's investment manager/adviser"
    )
    sponsor: Optional[str] = Field(
        default=None,
        description="Name of the sponsor organization"
    )
    fund_currency: str = Field(
        default="USD",
        description="Primary currency of the fund (usually USD)"
    )
    fiscal_year_end: Optional[str] = Field(
        default=None,
        description="Fiscal year end date (e.g., 'March 31', 'December 31')"
    )
    fund_type: Optional[str] = Field(
        default=None,
        description="Fund type: 'interval_fund', 'tender_offer_fund', 'bdc', or 'other'"
    )
    number_of_share_classes: Optional[int] = Field(
        default=None,
        description="Total number of share classes offered"
    )

    confidence: Optional[ConfidenceLevel] = Field(
        default=None,
        description="Confidence level of extraction"
    )
    citation: Optional[Citation] = None


class LeverageLimitsExtraction(BaseModel):
    """Extract fund leverage/borrowing limits."""

    reasoning: Optional[ExtractionReasoning] = Field(
        default=None,
        description="Step-by-step reasoning about leverage limits"
    )

    uses_leverage: Optional[bool] = Field(
        default=None,
        description="Whether the fund uses leverage/borrowing"
    )
    max_leverage_pct: Optional[Decimal] = Field(
        default=None,
        description="Maximum leverage as percentage (e.g., 33 for 33% of total assets, 50 for 50% debt-to-equity)"
    )
    leverage_basis: Optional[str] = Field(
        default=None,
        description="Basis for leverage calculation: 'total_assets', 'net_assets', 'asset_coverage_ratio' (for 1940 Act funds reporting 300% coverage = 33% leverage), 'debt_to_equity', or description"
    )
    leverage_purpose: Optional[str] = Field(
        default=None,
        description="Stated purpose for leverage (e.g., 'bridge financing', 'investment purposes')"
    )
    credit_facility_size: Optional[Decimal] = Field(
        default=None,
        description="Size of credit facility in millions USD if mentioned"
    )

    confidence: Optional[ConfidenceLevel] = Field(
        default=None,
        description="Confidence level of extraction"
    )
    citation: Optional[Citation] = None


class DistributionTermsExtraction(BaseModel):
    """Extract distribution/dividend terms."""

    reasoning: Optional[ExtractionReasoning] = Field(
        default=None,
        description="Step-by-step reasoning about distribution terms"
    )

    distribution_frequency: Optional[str] = Field(
        default=None,
        description="How often distributions are made: 'monthly', 'quarterly', 'annual', 'variable'"
    )
    default_distribution_policy: Optional[str] = Field(
        default=None,
        description="Default handling: 'cash', 'reinvested', 'DRIP' (dividend reinvestment plan)"
    )
    distribution_source: Optional[str] = Field(
        default=None,
        description="Source of distributions: 'income', 'capital_gains', 'return_of_capital', 'mixed'"
    )
    target_distribution_rate: Optional[Decimal] = Field(
        default=None,
        description="Target annual distribution rate as percentage of NAV if stated"
    )

    confidence: Optional[ConfidenceLevel] = Field(
        default=None,
        description="Confidence level of extraction"
    )
    citation: Optional[Citation] = None


# =============================================================================
# FEE EXTRACTION SCHEMAS
# =============================================================================

class HurdleRateFrequency(str, Enum):
    """Frequency at which hurdle rate is measured."""
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    MONTHLY = "monthly"
    OTHER = "other"


class IncentiveFeeBasis(str, Enum):
    """Basis on which incentive fee is calculated."""
    NET_INVESTMENT_INCOME = "net_investment_income"  # Income only (Blue Owl style)
    TOTAL_RETURN = "total_return"  # Income + capital gains
    REALIZED_GAINS = "realized_gains"  # Capital gains only
    NAV_APPRECIATION = "nav_appreciation"  # Change in NAV
    OTHER = "other"


class IncentiveFeeExtraction(BaseModel):
    """
    Extract incentive/performance fee structure.

    Supports both simple structures (flat % above hurdle) and complex
    PE-style waterfalls with catch-up mechanisms.

    Example structures:
    - Simple: "20% of returns above 8% hurdle"
    - With catch-up: "100% catch-up to 10%, then 20% above" (2/20 structure)
    - Blue Owl style: "10% of net investment income above 1.5% quarterly hurdle,
      with 100% catch-up between 1.5% and 1.667%"
    """

    # Chain-of-thought reasoning
    reasoning: Optional[ExtractionReasoning] = Field(
        default=None,
        description="Step-by-step reasoning about the incentive fee structure"
    )

    # Core fields
    has_incentive_fee: Optional[bool] = Field(
        default=None,
        description="Whether the fund charges an incentive fee at the fund level"
    )
    incentive_fee_rate: Optional[Decimal] = Field(
        default=None,
        description="Incentive fee as percentage (e.g., 20 for 20%, 10 for 10%)"
    )

    # Hurdle rate details
    hurdle_rate: Optional[Decimal] = Field(
        default=None,
        description="Hurdle rate/preferred return before incentive applies. Use ANNUALIZED rate (e.g., 8 for 8%, 6 for 6%)"
    )
    hurdle_rate_as_stated: Optional[Decimal] = Field(
        default=None,
        description="JUST THE NUMBER of the periodic hurdle rate BEFORE annualization (e.g., 1.25 for '1.25% quarterly', 1.5 for '1.5% per quarter'). Null if only annualized rate given."
    )
    hurdle_rate_frequency: Optional[HurdleRateFrequency] = Field(
        default=None,
        description="Frequency at which hurdle is measured: quarterly, annual, monthly"
    )

    # High water mark
    high_water_mark: Optional[bool] = Field(
        default=None,
        description="Whether a high water mark applies (prevents paying incentive on same gains twice)"
    )

    # Catch-up mechanism (PE-style waterfalls)
    has_catch_up: Optional[bool] = Field(
        default=None,
        description="Whether there is a catch-up provision after hurdle is met"
    )
    catch_up_rate_pct: Optional[Decimal] = Field(
        default=None,
        description="Catch-up allocation to manager as percentage (typically 100 for 100%)"
    )
    catch_up_ceiling_pct: Optional[Decimal] = Field(
        default=None,
        description="Return level where catch-up ends, as stated (e.g., 1.667 for '1.667% quarterly'). Use same frequency as hurdle_rate_as_stated."
    )

    # Fee basis
    fee_basis: Optional[IncentiveFeeBasis] = Field(
        default=None,
        description="What returns the incentive fee is calculated on: net_investment_income, total_return, realized_gains, nav_appreciation"
    )

    # Crystallization
    crystallization_frequency: Optional[str] = Field(
        default=None,
        description="How often incentive fee is crystallized/paid: 'quarterly', 'annual', 'at redemption'"
    )

    # Free text for complex structures
    fee_structure_description: Optional[str] = Field(
        default=None,
        description="Brief description of fee structure if complex or doesn't fit above fields"
    )

    # For fund-of-funds: underlying fund fees
    underlying_fund_incentive_range: Optional[str] = Field(
        default=None,
        description="Range of incentive fees charged by underlying funds (e.g., '15%-20%')"
    )

    confidence: Optional[ConfidenceLevel] = Field(
        default=None,
        description="Confidence level of extraction"
    )
    citation: Optional[Citation] = None


class ExpenseCapExtraction(BaseModel):
    """Extract expense cap/fee waiver information."""

    has_expense_cap: Optional[bool] = Field(
        default=None,
        description="Whether there is an expense cap or fee waiver"
    )
    cap_rate: Optional[Decimal] = Field(
        default=None,
        description="Expense cap as percentage of net assets"
    )
    cap_expiration: Optional[str] = Field(
        default=None,
        description="When the expense cap expires (date or 'indefinite')"
    )
    waived_fees: Optional[list[str]] = Field(
        default=None,
        description="Which fees are subject to waiver"
    )
    recoupment_period: Optional[str] = Field(
        default=None,
        description="Period during which waived fees can be recouped"
    )

    confidence: Optional[ConfidenceLevel] = Field(
        default=None,
        description="Confidence level of extraction"
    )
    citation: Optional[Citation] = None


# =============================================================================
# REPURCHASE/LIQUIDITY SCHEMAS
# =============================================================================

class RepurchaseFrequency(str, Enum):
    """Repurchase offer frequency."""
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"
    MONTHLY = "monthly"
    DISCRETIONARY = "discretionary"  # Tender offer funds
    OTHER = "other"


class RepurchaseTermsExtraction(BaseModel):
    """Extract repurchase/tender offer terms."""

    # Chain-of-thought reasoning FIRST
    reasoning: Optional[ExtractionReasoning] = Field(
        default=None,
        description="Step-by-step reasoning about the fund's repurchase structure"
    )

    fund_structure: Optional[str] = Field(
        default=None,
        description="'interval_fund' (Rule 23c-3, mandatory offers) or 'tender_offer_fund' (discretionary)"
    )
    repurchase_frequency: Optional[RepurchaseFrequency] = Field(
        default=None,
        description="How often repurchase offers are made"
    )
    repurchase_percentage_min: Optional[Decimal] = Field(
        default=None,
        description="Minimum percentage of NAV offered for repurchase (e.g., 5)"
    )
    repurchase_percentage_max: Optional[Decimal] = Field(
        default=None,
        description="Maximum percentage of NAV offered for repurchase (e.g., 25)"
    )
    repurchase_percentage_typical: Optional[Decimal] = Field(
        default=None,
        description="Typical/target percentage if stated"
    )
    repurchase_basis: Optional[str] = Field(
        default=None,
        description="Basis for repurchase: 'nav', 'number_of_shares', 'net_assets'"
    )
    notice_period_days: Optional[int] = Field(
        default=None,
        description="Days of advance notice required for repurchase requests"
    )
    pricing_date_description: Optional[str] = Field(
        default=None,
        description="When repurchase price is determined"
    )

    # Lock-up and early redemption (EXPANDED v2.0)
    lock_up_period_years: Optional[Decimal] = Field(
        default=None,
        description="Lock-up period in YEARS before first repurchase eligibility (e.g., 1 for 1 year)"
    )
    lock_up_period_days: Optional[int] = Field(
        default=None,
        description="Lock-up period in DAYS if specified in days instead of years"
    )
    early_repurchase_fee_pct: Optional[Decimal] = Field(
        default=None,
        description="Early repurchase/redemption fee as percentage (e.g., 2 for 2%)"
    )
    early_repurchase_fee_period: Optional[str] = Field(
        default=None,
        description="Period during which early repurchase fee applies (e.g., 'within 1 year of purchase')"
    )
    minimum_repurchase_amount: Optional[Decimal] = Field(
        default=None,
        description="Minimum dollar amount for repurchase requests"
    )
    minimum_holding_after_repurchase: Optional[Decimal] = Field(
        default=None,
        description="Minimum balance that must remain after repurchase"
    )

    confidence: Optional[ConfidenceLevel] = Field(
        default=None,
        description="Confidence level of extraction"
    )
    citation: Optional[Citation] = None


# =============================================================================
# INVESTMENT STRATEGY SCHEMAS
# =============================================================================

class AllocationTarget(BaseModel):
    """Single allocation target."""
    asset_class: str = Field(description="Asset class or strategy name")
    target_percentage: Optional[str] = Field(
        default=None,
        description="Target allocation percentage (number or null if not found)"
    )
    range_min: Optional[str] = Field(
        default=None,
        description="Minimum allocation percentage (number or null if not found)"
    )
    range_max: Optional[str] = Field(
        default=None,
        description="Maximum allocation percentage (number or null if not found)"
    )


class AllocationTargetsExtraction(BaseModel):
    """Extract target asset allocation."""

    allocations: list[AllocationTarget] = Field(
        default_factory=list,
        description="List of target allocations by asset class"
    )
    allocation_approach: Optional[str] = Field(
        default=None,
        description="Description of allocation approach (e.g., 'opportunistic', 'strategic')"
    )
    rebalancing_frequency: Optional[str] = Field(
        default=None,
        description="How often allocations are rebalanced"
    )

    confidence: Optional[ConfidenceLevel] = Field(
        default=None,
        description="Confidence level of extraction"
    )
    citation: Optional[Citation] = None


class ConcentrationLimit(BaseModel):
    """Single concentration limit."""
    limit_type: str = Field(
        description="Type of limit (e.g., 'single_issuer', 'single_fund', 'industry', 'geography')"
    )
    limit_percentage: Optional[Decimal] = Field(
        default=None,
        description="Maximum percentage allowed. Use null if 'unlimited' or 'no limit'."
    )
    is_unlimited: bool = Field(
        default=False,
        description="True if the limit is explicitly stated as 'unlimited' or 'no limit'"
    )
    description: Optional[str] = Field(
        default=None,
        description="Additional details about the limit"
    )


class ConcentrationLimitsExtraction(BaseModel):
    """Extract investment concentration limits."""

    # Chain-of-thought reasoning FIRST (before extraction)
    reasoning: Optional[ExtractionReasoning] = Field(
        default=None,
        description="Step-by-step reasoning about what concentration limits are stated"
    )

    # Then the actual extraction
    has_concentration_limits: Optional[bool] = Field(
        default=None,
        description="Whether the fund has any concentration limits. False if 'no limits' stated."
    )
    limits: list[ConcentrationLimit] = Field(
        default_factory=list,
        description="List of concentration limits. Empty if no limits stated."
    )
    diversification_policy: Optional[str] = Field(
        default=None,
        description="General diversification policy description"
    )

    confidence: Optional[ConfidenceLevel] = Field(
        default=None,
        description="Confidence level of extraction"
    )
    citation: Optional[Citation] = None


# =============================================================================
# SHARE CLASS SCHEMAS
# =============================================================================


class ShareClassDiscovery(BaseModel):
    """
    First-pass: discover which share classes exist in the document.

    This schema is used for the discovery phase of two-pass share class extraction.
    The goal is to identify ALL share class names mentioned in the document before
    attempting to extract detailed field values for each class.
    """
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief reasoning about share classes found in the document"
    )
    share_class_names: list[str] = Field(
        default_factory=list,
        description=(
            "List of share class names found (e.g., 'Class I', 'Class S', 'Class R'). "
            "Normalize names: remove 'Shares' suffix, use 'Class X' format. "
            "Include ALL classes mentioned, even if only referenced once."
        )
    )
    confidence: Optional[ConfidenceLevel] = Field(
        default=None,
        description="Confidence level for the discovery"
    )


class ShareClassFieldsExtraction(BaseModel):
    """
    Second-pass: extract fields for a SINGLE specific share class.

    This schema is used after discovery to extract detailed field values
    for one share class at a time, with the class name known upfront.
    """
    class_name: str = Field(
        description="The share class name being extracted (provided upfront)"
    )

    reasoning: Optional[str] = Field(
        default=None,
        description="Brief reasoning about the values found for this class"
    )

    # Minimums
    minimum_initial_investment: Optional[Decimal] = Field(
        default=None,
        description="Minimum initial investment in dollars for this class"
    )
    minimum_additional_investment: Optional[Decimal] = Field(
        default=None,
        description="Minimum additional/subsequent investment in dollars for this class"
    )
    minimum_balance_for_repurchase: Optional[Decimal] = Field(
        default=None,
        description="Minimum balance to submit a repurchase request for this class"
    )

    # Eligibility
    investor_eligibility: Optional[str] = Field(
        default=None,
        description="Who can invest in this class (e.g., 'institutional investors')"
    )
    distribution_channel: Optional[str] = Field(
        default=None,
        description="How shares are distributed (e.g., 'fee-based advisory programs')"
    )

    # Fees
    sales_load_pct: Optional[Decimal] = Field(
        default=None,
        description="Sales load/charge as percentage (e.g., 3.5 for 3.5%)"
    )
    distribution_servicing_fee_pct: Optional[Decimal] = Field(
        default=None,
        description="Combined distribution and/or shareholder servicing fee as percentage"
    )

    # Offering price
    offering_price_basis: Optional[str] = Field(
        default=None,
        description="Basis for offering price: 'NAV', 'NAV plus sales load', etc."
    )

    # Evidence
    evidence_quote: Optional[str] = Field(
        default=None,
        description="Verbatim quote from the text supporting these values (max 300 chars)"
    )

    confidence: Optional[ConfidenceLevel] = Field(
        default=None,
        description="Confidence level for the extraction"
    )


class ShareClassDetails(BaseModel):
    """Details for a single share class."""
    class_name: str = Field(
        description=(
            "Share class name WITHOUT the word 'Shares'. "
            "Use 'Class I' not 'Class I Shares'. "
            "Examples: 'Class I', 'Class S', 'Class D', 'Class U'"
        )
    )

    # Minimums
    minimum_initial_investment: Optional[Decimal] = Field(
        default=None,
        description="Minimum initial investment in dollars"
    )
    minimum_additional_investment: Optional[Decimal] = Field(
        default=None,
        description="Minimum additional/subsequent investment in dollars"
    )
    minimum_balance_for_repurchase: Optional[Decimal] = Field(
        default=None,
        description="Minimum balance to submit a repurchase request (repurchase threshold)"
    )

    # Eligibility
    investor_eligibility: Optional[str] = Field(
        default=None,
        description="Who can invest in this class (e.g., 'institutional investors')"
    )
    distribution_channel: Optional[str] = Field(
        default=None,
        description="How shares are distributed (e.g., 'fee-based advisory programs')"
    )

    # Fees (if not in XBRL)
    sales_load_pct: Optional[Decimal] = Field(
        default=None,
        description="Sales load/charge as percentage (e.g., 3.5 for 3.5%)"
    )
    distribution_servicing_fee_pct: Optional[Decimal] = Field(
        default=None,
        description=(
            "Combined distribution and/or shareholder servicing fee as percentage. "
            "Include ANY ongoing annual fee labeled as 'distribution fee', '12b-1 fee', "
            "'shareholder servicing fee', or 'distribution and servicing fee'. "
            "These are economically equivalent to LPs. Set to 0 if no such fees."
        )
    )
    management_fee_pct: Optional[Decimal] = Field(
        default=None,
        description="Annual management/advisory fee as percentage of net assets (e.g., 1.25 for 1.25%)"
    )
    affe_pct: Optional[Decimal] = Field(
        default=None,
        description="Acquired fund fees and expenses as percentage (for fund-of-funds)"
    )
    interest_expense_pct: Optional[Decimal] = Field(
        default=None,
        description="Interest expenses on borrowings as percentage of net assets"
    )
    other_expenses_pct: Optional[Decimal] = Field(
        default=None,
        description="Other annual expenses as percentage of net assets"
    )
    total_expense_ratio_pct: Optional[Decimal] = Field(
        default=None,
        description="Total annual expenses before fee waivers as percentage of net assets"
    )
    net_expense_ratio_pct: Optional[Decimal] = Field(
        default=None,
        description="Net annual expenses after fee waivers as percentage of net assets"
    )
    fee_waiver_pct: Optional[Decimal] = Field(
        default=None,
        description="Fee waiver/reimbursement as percentage of net assets"
    )
    incentive_fee_xbrl_pct: Optional[Decimal] = Field(
        default=None,
        description=(
            "Incentive/performance fee as percentage of net assets from fee table. "
            "This is DISTINCT from the fund-level contractual incentive fee rate."
        )
    )

    # Offering price
    offering_price_basis: Optional[str] = Field(
        default=None,
        description="Basis for offering price: 'NAV', 'NAV plus sales load', etc."
    )


class ShareClassesExtraction(BaseModel):
    """Extract share class information."""

    # Chain-of-thought reasoning FIRST
    reasoning: Optional[ExtractionReasoning] = Field(
        default=None,
        description="Step-by-step reasoning about share class details"
    )

    share_classes: list[ShareClassDetails] = Field(
        default_factory=list,
        description="List of share classes offered"
    )

    confidence: Optional[ConfidenceLevel] = Field(
        default=None,
        description="Confidence level of extraction"
    )
    citation: Optional[Citation] = None


# =============================================================================
# COMBINED EXTRACTION RESULT
# =============================================================================

class DocumentExtractionResult(BaseModel):
    """Complete extraction result for a document."""

    filing_id: str
    cik: str
    fund_name: str

    # Fund metadata (NEW v2.0)
    fund_metadata: Optional[FundMetadataExtraction] = None
    leverage_limits: Optional[LeverageLimitsExtraction] = None
    distribution_terms: Optional[DistributionTermsExtraction] = None

    # Fee structure
    incentive_fee: Optional[IncentiveFeeExtraction] = None
    expense_cap: Optional[ExpenseCapExtraction] = None

    # Liquidity
    repurchase_terms: Optional[RepurchaseTermsExtraction] = None

    # Strategy
    allocation_targets: Optional[AllocationTargetsExtraction] = None
    concentration_limits: Optional[ConcentrationLimitsExtraction] = None

    # Share classes
    share_classes: Optional[ShareClassesExtraction] = None

    # Processing metadata
    chunks_processed: int
    extraction_errors: list[str] = Field(default_factory=list)
