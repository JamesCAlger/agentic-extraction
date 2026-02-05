"""
Per-Datapoint Extraction with Tier3-Style Prompts

This module tests whether per-datapoint extraction improves when using
Tier3-style extraction prompts instead of DocVQA-style questions.

Key differences from per_datapoint_extractor.py:
1. Uses "Extract X information" instead of "What is X?"
2. Uses full system prompt with principles
3. Uses Pydantic schemas via instructor
4. Includes negative guidance in prompts
5. Has retry logic on validation failure

This isolates the prompt style effect from the granularity effect.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Any, Type, TypeVar
from decimal import Decimal
import time

import instructor
from pydantic import BaseModel, Field

from .llm_provider import (
    create_instructor_client,
    RateLimitConfig,
    detect_provider,
    resolve_model_name,
)
from .per_datapoint_extractor import (
    DATAPOINT_KEYWORDS,
    get_top_chunks_for_datapoint,
    ScoredChunk,
)
from .prompts import SYSTEM_PROMPT  # Use the same system prompt as Tier3
from ..parse.models import ChunkedDocument

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# PYDANTIC SCHEMAS FOR EACH DATAPOINT (matching Tier3 style)
# =============================================================================

class DatapointExtraction(BaseModel):
    """Base schema for single datapoint extraction."""
    evidence_quote: Optional[str] = Field(
        default=None,
        description="Verbatim quote from text supporting this value (max 200 chars)"
    )
    confidence: Optional[str] = Field(
        default=None,
        description="Confidence level: 'explicit', 'inferred', or 'not_found'"
    )


class HasIncentiveFeeExtraction(DatapointExtraction):
    """Extract whether fund has incentive fee."""
    has_incentive_fee: Optional[bool] = Field(
        default=None,
        description="Whether the fund charges an incentive/performance fee at the FUND level"
    )


class IncentiveFeePctExtraction(DatapointExtraction):
    """Extract incentive fee percentage."""
    incentive_fee_pct: Optional[Decimal] = Field(
        default=None,
        description="Incentive fee percentage (e.g., 10 for 10%, 12.5 for 12.5%)"
    )


class HurdleRatePctExtraction(DatapointExtraction):
    """Extract hurdle rate percentage."""
    hurdle_rate_pct: Optional[Decimal] = Field(
        default=None,
        description="Hurdle rate as ANNUALIZED percentage (e.g., 5 for 5% annual)"
    )


class HurdleRateAsStatedExtraction(DatapointExtraction):
    """Extract hurdle rate as stated."""
    hurdle_rate_as_stated: Optional[str] = Field(
        default=None,
        description="JUST THE NUMBER of the periodic hurdle rate BEFORE annualization (e.g., '1.25' if doc says '1.25% quarterly', '1.5' if doc says '1.5% per quarter'). Return null if only annualized rate is given."
    )


class HurdleRateFrequencyExtraction(DatapointExtraction):
    """Extract hurdle rate frequency."""
    hurdle_rate_frequency: Optional[str] = Field(
        default=None,
        description="Frequency: 'quarterly', 'annual', 'monthly', or null"
    )


class HighWaterMarkExtraction(DatapointExtraction):
    """Extract high water mark."""
    high_water_mark: Optional[bool] = Field(
        default=None,
        description="Whether fund uses high water mark or loss recovery mechanism"
    )


class HasCatchUpExtraction(DatapointExtraction):
    """Extract catch-up provision."""
    has_catch_up: Optional[bool] = Field(
        default=None,
        description="Whether incentive fee has catch-up provision"
    )


class FeeBasisExtraction(DatapointExtraction):
    """Extract fee basis."""
    fee_basis: Optional[str] = Field(
        default=None,
        description="Basis: 'net_investment_income', 'net_profits', 'nav_appreciation', or null"
    )


class CrystallizationFrequencyExtraction(DatapointExtraction):
    """Extract crystallization frequency."""
    crystallization_frequency: Optional[str] = Field(
        default=None,
        description="How often fee crystallizes: 'quarterly', 'annual', or null"
    )


class HasExpenseCapExtraction(DatapointExtraction):
    """Extract expense cap existence."""
    has_expense_cap: Optional[bool] = Field(
        default=None,
        description="Whether fund has expense limitation or fee waiver"
    )


class ExpenseCapPctExtraction(DatapointExtraction):
    """Extract expense cap percentage."""
    expense_cap_pct: Optional[Decimal] = Field(
        default=None,
        description="Expense cap as percentage of net assets"
    )


class RepurchaseFrequencyExtraction(DatapointExtraction):
    """Extract repurchase frequency."""
    repurchase_frequency: Optional[str] = Field(
        default=None,
        description="How often repurchases occur: 'quarterly', 'semi-annual', 'annual'"
    )


class RepurchaseAmountPctExtraction(DatapointExtraction):
    """Extract repurchase amount percentage."""
    repurchase_amount_pct: Optional[Decimal] = Field(
        default=None,
        description="Minimum repurchase offer percentage (e.g., 5 for 5%)"
    )


class RepurchaseBasisExtraction(DatapointExtraction):
    """Extract repurchase basis."""
    repurchase_basis: Optional[str] = Field(
        default=None,
        description="Basis: 'outstanding_shares', 'net_assets', 'nav'"
    )


class LockUpPeriodExtraction(DatapointExtraction):
    """Extract lock-up period."""
    lock_up_period_years: Optional[Decimal] = Field(
        default=None,
        description="Lock-up period in years (e.g., 1 for one year)"
    )


class EarlyRepurchaseFeeExtraction(DatapointExtraction):
    """Extract early repurchase fee."""
    early_repurchase_fee_pct: Optional[Decimal] = Field(
        default=None,
        description="Early repurchase fee percentage (e.g., 2 for 2%)"
    )


class UsesLeverageExtraction(DatapointExtraction):
    """Extract leverage usage."""
    uses_leverage: Optional[bool] = Field(
        default=None,
        description="Whether fund uses leverage/borrowing"
    )


class MaxLeveragePctExtraction(DatapointExtraction):
    """Extract max leverage percentage."""
    max_leverage_pct: Optional[Decimal] = Field(
        default=None,
        description="Maximum leverage as percentage (e.g., 33 for 33% of assets)"
    )


class LeverageBasisExtraction(DatapointExtraction):
    """Extract leverage basis."""
    leverage_basis: Optional[str] = Field(
        default=None,
        description="Basis: 'total_assets', 'net_assets', 'managed_assets'"
    )


class DistributionFrequencyExtraction(DatapointExtraction):
    """Extract distribution frequency."""
    distribution_frequency: Optional[str] = Field(
        default=None,
        description="How often distributions paid: 'monthly', 'quarterly', 'annual'"
    )


class DefaultDistributionPolicyExtraction(DatapointExtraction):
    """Extract default distribution policy."""
    default_distribution_policy: Optional[str] = Field(
        default=None,
        description="Default policy: 'cash', 'reinvested'"
    )


class ShareClassInfo(BaseModel):
    """Single share class information."""
    class_name: Optional[str] = None
    minimum_initial_investment: Optional[int] = None
    distribution_servicing_fee_pct: Optional[Decimal] = None


class ShareClassesExtraction(DatapointExtraction):
    """Extract share classes."""
    share_classes: Optional[list[ShareClassInfo]] = Field(
        default=None,
        description="List of share classes with minimums and fees"
    )


# New schemas for additional fields
class CatchUpRatePctExtraction(DatapointExtraction):
    """Extract catch-up rate percentage."""
    catch_up_rate_pct: Optional[Decimal] = Field(
        default=None,
        description="Catch-up rate percentage (e.g., 100 for 100% catch-up)"
    )


class CatchUpCeilingPctExtraction(DatapointExtraction):
    """Extract catch-up ceiling percentage."""
    catch_up_ceiling_pct: Optional[Decimal] = Field(
        default=None,
        description="Catch-up ceiling as periodic percentage (e.g., 1.667 for 1.667% quarterly)"
    )


class UnderlyingFundIncentiveRangeExtraction(DatapointExtraction):
    """Extract underlying fund incentive fee range for fund-of-funds."""
    underlying_fund_incentive_range: Optional[str] = Field(
        default=None,
        description="Range of incentive fees charged by underlying funds (e.g., '15% to 20%')"
    )


class RepurchasePercentageMinExtraction(DatapointExtraction):
    """Extract minimum repurchase percentage."""
    repurchase_percentage_min: Optional[Decimal] = Field(
        default=None,
        description="Minimum repurchase offer percentage (e.g., 5 for 5%)"
    )


class RepurchasePercentageMaxExtraction(DatapointExtraction):
    """Extract maximum repurchase percentage."""
    repurchase_percentage_max: Optional[Decimal] = Field(
        default=None,
        description="Maximum repurchase offer percentage (e.g., 25 for 25%)"
    )


class SecondaryFundsMinPctExtraction(DatapointExtraction):
    """Extract minimum allocation to secondary funds."""
    secondary_funds_min_pct: Optional[Decimal] = Field(
        default=None,
        description="Minimum allocation target to secondary fund investments (%)"
    )


class SecondaryFundsMaxPctExtraction(DatapointExtraction):
    """Extract maximum allocation to secondary funds."""
    secondary_funds_max_pct: Optional[Decimal] = Field(
        default=None,
        description="Maximum allocation target to secondary fund investments (%)"
    )


class DirectInvestmentsMinPctExtraction(DatapointExtraction):
    """Extract minimum allocation to direct investments."""
    direct_investments_min_pct: Optional[Decimal] = Field(
        default=None,
        description="Minimum allocation target to direct/co-investments (%)"
    )


class DirectInvestmentsMaxPctExtraction(DatapointExtraction):
    """Extract maximum allocation to direct investments."""
    direct_investments_max_pct: Optional[Decimal] = Field(
        default=None,
        description="Maximum allocation target to direct/co-investments (%)"
    )


class SecondaryInvestmentsMinPctExtraction(DatapointExtraction):
    """Extract minimum allocation to secondary investments."""
    secondary_investments_min_pct: Optional[Decimal] = Field(
        default=None,
        description="Minimum allocation target to secondary investments (%)"
    )


class MaxSingleAssetPctExtraction(DatapointExtraction):
    """Extract maximum single asset concentration."""
    max_single_asset_pct: Optional[Decimal] = Field(
        default=None,
        description="Maximum allocation to any single asset/investment (%)"
    )


class MaxSingleFundPctExtraction(DatapointExtraction):
    """Extract maximum single fund concentration."""
    max_single_fund_pct: Optional[Decimal] = Field(
        default=None,
        description="Maximum allocation to any single underlying fund (%)"
    )


class MaxSingleSectorPctExtraction(DatapointExtraction):
    """Extract maximum single sector concentration."""
    max_single_sector_pct: Optional[Decimal] = Field(
        default=None,
        description="Maximum allocation to any single sector/industry (%)"
    )


# =============================================================================
# TIER3-STYLE EXTRACTION PROMPTS (not DocVQA questions)
# =============================================================================

TIER3_STYLE_PROMPTS = {
    "incentive_fee.has_incentive_fee": {
        "schema": HasIncentiveFeeExtraction,
        "prompt": """Extract whether this fund charges an incentive fee or performance fee.

Look for evidence of FUND-LEVEL incentive/performance fees:
- "incentive fee", "performance fee", "performance-based fee"
- "incentive allocation", "carried interest", "performance allocation"

IMPORTANT DISTINCTIONS:
- Only extract fees charged by THIS fund to its investors
- IGNORE underlying fund fees (AFFE, "underlying funds charge...")
- IGNORE interest expense on borrowings
- Management fees are NOT incentive fees

If the fund charges an incentive fee at the fund level, set has_incentive_fee=True.
If no fund-level incentive fee exists, set has_incentive_fee=False.

TEXT:
{text}"""
    },

    "incentive_fee.incentive_fee_pct": {
        "schema": IncentiveFeePctExtraction,
        "prompt": """Extract the incentive fee percentage charged by this fund.

Look for the FUND-LEVEL incentive fee rate:
- "incentive fee equal to X%"
- "performance fee of X%"
- "X% of net profits above the hurdle"
- "carried interest of X%"

Common values: 10%, 12.5%, 15%, 20%

IMPORTANT: Extract only the fund-level fee, NOT underlying fund fees.
If stated as a range (e.g., "10% to 20%"), extract the fund-level rate.

TEXT:
{text}"""
    },

    "incentive_fee.hurdle_rate_pct": {
        "schema": HurdleRatePctExtraction,
        "prompt": """Extract the hurdle rate for the incentive fee as an ANNUALIZED percentage.

Look for hurdle/preferred return language:
- "hurdle rate of X%", "X% annualized hurdle"
- "preferred return of X%"
- "X% quarterly" (multiply by 4 to annualize)
- "1.25% per quarter" = 5% annualized

CONVERSION RULES:
- If stated quarterly, multiply by 4 (e.g., 1.25% quarterly = 5% annual)
- If stated monthly, multiply by 12
- Report the ANNUALIZED rate

Common annualized values: 5%, 6%, 8%

TEXT:
{text}"""
    },

    "incentive_fee.hurdle_rate_as_stated": {
        "schema": HurdleRateAsStatedExtraction,
        "prompt": """Extract JUST THE NUMBER of the periodic hurdle rate BEFORE annualization.

IMPORTANT: We want the raw periodic rate, NOT the annualized equivalent.

Look for quarterly/monthly hurdle rates:
- "1.25% quarterly" -> return "1.25"
- "1.5% per quarter" -> return "1.5"
- "2% monthly" -> return "2"

If only an annualized rate is given (e.g., "5% annualized hurdle"), return null.
Do NOT extract the annualized rate - we have a separate field for that.

Return ONLY the number (e.g., "1.25"), not the full phrase.

TEXT:
{text}"""
    },

    "incentive_fee.hurdle_rate_frequency": {
        "schema": HurdleRateFrequencyExtraction,
        "prompt": """Extract the frequency at which the hurdle rate is measured.

Determine if the hurdle is stated on a:
- "quarterly" basis (e.g., "1.25% per quarter", "quarterly hurdle")
- "annual" basis (e.g., "5% annualized", "annual hurdle rate")
- "monthly" basis (rare)

Look for frequency indicators in the hurdle rate description.

TEXT:
{text}"""
    },

    "incentive_fee.high_water_mark": {
        "schema": HighWaterMarkExtraction,
        "prompt": """Extract whether the fund uses a high water mark or loss recovery mechanism.

Look for evidence of loss recovery provisions:
- "high water mark", "high-water mark", "HWM"
- "loss recovery account", "loss carryforward"
- "deficit recovery", "cumulative loss recovery"
- "prior losses must be recovered before incentive fee is paid"

Set high_water_mark=True if any such mechanism exists.
Set high_water_mark=False if explicitly stated there is no HWM.

TEXT:
{text}"""
    },

    "incentive_fee.has_catch_up": {
        "schema": HasCatchUpExtraction,
        "prompt": """Extract whether the incentive fee has a catch-up provision.

Look for catch-up language:
- "catch-up", "catch up provision"
- "full catch-up", "100% catch-up"
- "until the adviser receives its full incentive"

Set has_catch_up=True if a catch-up provision exists.

TEXT:
{text}"""
    },

    "incentive_fee.fee_basis": {
        "schema": FeeBasisExtraction,
        "prompt": """Extract the basis on which the incentive fee is calculated.

Determine the fee calculation basis:
- "net_investment_income": Fee on income only (interest, dividends)
- "net_profits": Fee on total return (income + capital gains)
- "nav_appreciation": Fee on NAV growth

Look for language like:
- "X% of pre-incentive fee net investment income" → net_investment_income
- "X% of net profits" → net_profits
- "X% of the appreciation in NAV" → nav_appreciation

TEXT:
{text}"""
    },

    "incentive_fee.crystallization_frequency": {
        "schema": CrystallizationFrequencyExtraction,
        "prompt": """Extract how often the incentive fee crystallizes (is calculated and paid).

Look for crystallization/payment timing:
- "quarterly in arrears", "calculated quarterly" → quarterly
- "annually", "annual crystallization" → annual
- "payable quarterly", "paid each quarter" → quarterly

TEXT:
{text}"""
    },

    "expense_cap.has_expense_cap": {
        "schema": HasExpenseCapExtraction,
        "prompt": """Extract whether the fund has an expense limitation or fee waiver agreement.

Look for expense cap evidence:
- "expense limitation agreement", "expense cap"
- "fee waiver", "agreed to waive fees"
- "reimburse expenses", "limit total annual operating expenses"
- "contractual cap", "voluntary waiver"

Set has_expense_cap=True if any such arrangement exists.

TEXT:
{text}"""
    },

    "expense_cap.expense_cap_pct": {
        "schema": ExpenseCapPctExtraction,
        "prompt": """Extract the expense cap percentage.

Look for the cap rate:
- "capped at X%", "limited to X%"
- "expense limitation of X%"
- "not to exceed X% of net assets"

Common values: 1.50%, 1.75%, 2.00%, 2.25%

TEXT:
{text}"""
    },

    "repurchase_terms.repurchase_frequency": {
        "schema": RepurchaseFrequencyExtraction,
        "prompt": """Extract how often the fund conducts repurchase or tender offers.

CRITICAL DISTINCTION:
- Repurchase frequency = how often fund BUYS BACK shares (liquidity events)
- This is NOT distribution/dividend frequency

Look for repurchase timing:
- "quarterly repurchase offers" → quarterly
- "semi-annual tender offers" → semi-annual
- "annual repurchase" → annual

For interval funds, this is typically quarterly (Rule 23c-3).

TEXT:
{text}"""
    },

    "repurchase_terms.repurchase_amount_pct": {
        "schema": RepurchaseAmountPctExtraction,
        "prompt": """Extract the percentage of shares the fund offers to repurchase.

Look for repurchase amounts:
- "5% to 25% of outstanding shares"
- "at least 5%", "minimum of 5%"
- "up to 25%"

Extract the MINIMUM percentage if a range is given.
For interval funds, minimum is typically 5% per Rule 23c-3.

TEXT:
{text}"""
    },

    "repurchase_terms.repurchase_basis": {
        "schema": RepurchaseBasisExtraction,
        "prompt": """Extract the basis for repurchase offer calculations.

Determine what the percentage is based on:
- "of outstanding shares" → outstanding_shares
- "of net assets" → net_assets
- "of NAV" → nav

Look for the denominator in repurchase offer descriptions.

TEXT:
{text}"""
    },

    "repurchase_terms.lock_up_period_years": {
        "schema": LockUpPeriodExtraction,
        "prompt": """Extract the lock-up period before shares can be repurchased.

Look for holding period requirements:
- "one year lock-up", "12-month lock-up" → 1
- "shares must be held for at least one year" → 1
- "within one year of purchase" (for early fee context) → implies 1 year lock-up
- "one-year anniversary" → 1
- "no lock-up period" → 0

The lock-up is the time before first repurchase eligibility.
Extract as years (e.g., 1 for one year, 0.5 for 6 months).

TEXT:
{text}"""
    },

    "repurchase_terms.early_repurchase_fee_pct": {
        "schema": EarlyRepurchaseFeeExtraction,
        "prompt": """Extract the early repurchase or early redemption fee percentage.

Look for early redemption fees:
- "2% early repurchase fee"
- "early withdrawal charge of 2%"
- "2% fee for shares held less than one year"
- "Early Repurchase Fee equal to 2%"

Common value is 2% for shares held less than 1 year.
If no early fee exists, return null (not 0).

TEXT:
{text}"""
    },

    "leverage_limits.uses_leverage": {
        "schema": UsesLeverageExtraction,
        "prompt": """Extract whether the fund uses or intends to use leverage/borrowing.

Look for leverage indicators:
- "may borrow", "will borrow", "intends to use leverage"
- "credit facility", "line of credit"
- "may use leverage for investment purposes"

Set uses_leverage=True if fund uses or may use borrowing.
Set uses_leverage=False if explicitly stated fund will not borrow.

TEXT:
{text}"""
    },

    "leverage_limits.max_leverage_pct": {
        "schema": MaxLeveragePctExtraction,
        "prompt": """Extract the maximum leverage as a percentage.

Look for leverage limits:
- "borrow up to 33-1/3%" → 33.33
- "leverage not to exceed 50%" → 50
- "asset coverage of 300%" → 33.33 (inverse: can borrow 1/3)
- "asset coverage of 150%" → 66.67 (inverse: can borrow 2/3)

CONVERSION for asset coverage:
- 300% coverage = max 33.33% leverage
- 200% coverage = max 50% leverage
- 150% coverage = max 66.67% leverage

Report as percentage of assets that can be borrowed.

TEXT:
{text}"""
    },

    "leverage_limits.leverage_basis": {
        "schema": LeverageBasisExtraction,
        "prompt": """Extract what the leverage limit is measured against.

Determine the denominator:
- "of total assets" → total_assets
- "of net assets" → net_assets
- "of managed assets" → managed_assets
- "asset coverage ratio" → asset_coverage

TEXT:
{text}"""
    },

    "distribution_terms.distribution_frequency": {
        "schema": DistributionFrequencyExtraction,
        "prompt": """Extract how often the fund pays distributions/dividends.

CRITICAL DISTINCTION - DO NOT CONFUSE:
1. DISTRIBUTION FREQUENCY = How often fund PAYS DIVIDENDS (extract this)
2. REPURCHASE FREQUENCY = How often fund buys back shares (IGNORE)

Look for dividend/distribution timing:
- "monthly distributions", "pay dividends monthly" → monthly
- "quarterly distributions" → quarterly
- "annual distributions" → annual

IGNORE any mentions of "quarterly repurchase offers" - that's redemption, not distribution.

TEXT:
{text}"""
    },

    "distribution_terms.default_distribution_policy": {
        "schema": DefaultDistributionPolicyExtraction,
        "prompt": """Extract the default distribution policy (cash or reinvested).

Look for default handling:
- "automatically reinvested", "reinvested unless otherwise elected" → reinvested
- "DRIP", "dividend reinvestment plan" → reinvested
- "paid in cash", "cash distributions" → cash

TEXT:
{text}"""
    },

    "share_classes.share_classes": {
        "schema": ShareClassesExtraction,
        "prompt": """Extract share class information including minimums and fees.

For EACH share class, extract:
1. Class name (e.g., "Class S", "Class I", "Class D")
2. Minimum initial investment (dollar amount, e.g., 2500, 1000000)
3. Distribution/servicing fee percentage (e.g., 0.85 for 0.85%)

Look in sections like:
- "PLAN OF DISTRIBUTION", "PURCHASE OF SHARES"
- Fee tables, share class descriptions

Common patterns:
- "Class S minimum investment of $2,500"
- "Class I requires $1,000,000 minimum"
- "Class S distribution fee of 0.85%"

TEXT:
{text}"""
    },

    # New prompts for additional fields
    "incentive_fee.catch_up_rate_pct": {
        "schema": CatchUpRatePctExtraction,
        "prompt": """Extract the catch-up rate percentage for the incentive fee.

The catch-up is the rate at which the adviser receives incentive fees between the hurdle rate and a ceiling.
Common values are 100% (full catch-up) meaning the adviser gets 100% of returns in that range.

Look for:
- "full catch-up" or "100% catch-up"
- "catch-up provision" with a percentage
- Incentive fee waterfall descriptions

Return null if no catch-up mechanism exists or rate is not specified.

TEXT:
{text}"""
    },

    "incentive_fee.catch_up_ceiling_pct": {
        "schema": CatchUpCeilingPctExtraction,
        "prompt": """Extract the catch-up ceiling percentage for the incentive fee.

This is the threshold above which the standard incentive split applies after catch-up.
Often expressed as a periodic rate (e.g., 1.667% quarterly).

Look for:
- "exceeds X%" in catch-up provisions
- The upper bound of the catch-up range
- "pre-incentive fee net investment income exceeds..."

Return null if no catch-up ceiling is specified.

TEXT:
{text}"""
    },

    "incentive_fee.underlying_fund_incentive_range": {
        "schema": UnderlyingFundIncentiveRangeExtraction,
        "prompt": """Extract the range of incentive fees charged by underlying funds.

This applies to fund-of-funds structures where the fund invests in other private funds.
Look for AFFE (Acquired Fund Fees and Expenses) sections.

IMPORTANT: Extract fees of UNDERLYING funds, not this fund's own fees.

Look for:
- "underlying private funds...charge...incentive fees of..."
- "carried interest of approximately X% to Y%"
- AFFE footnotes mentioning underlying fund fee ranges

Return as a string range like "15% to 20%" or null if not disclosed.

TEXT:
{text}"""
    },

    "repurchase_terms.repurchase_percentage_min": {
        "schema": RepurchasePercentageMinExtraction,
        "prompt": """Extract the minimum repurchase offer percentage.

For interval funds, this is the minimum percentage of shares the fund must offer to repurchase.
Usually 5% under SEC rules.

Look for:
- "offer to repurchase between X% and Y%"
- "minimum amount of X%"
- Interval fund repurchase policy

Return just the number (e.g., 5 for 5%), null if not specified.

TEXT:
{text}"""
    },

    "repurchase_terms.repurchase_percentage_max": {
        "schema": RepurchasePercentageMaxExtraction,
        "prompt": """Extract the maximum repurchase offer percentage.

For interval funds, this is the maximum percentage of shares the fund may offer to repurchase.
Usually up to 25% under SEC rules.

Look for:
- "offer to repurchase between X% and Y%"
- "up to X% of outstanding shares"
- Maximum repurchase limits

Return just the number (e.g., 25 for 25%), null if not specified.

TEXT:
{text}"""
    },

    "allocation_targets.secondary_funds_min_pct": {
        "schema": SecondaryFundsMinPctExtraction,
        "prompt": """Extract the minimum allocation target percentage for secondary fund investments.

This is the fund's target minimum allocation to investments in secondary private equity funds.

Look for:
- Investment allocation targets/guidelines
- "X% to Y% in secondary funds"
- Investment strategy constraints

Return just the number (e.g., 40 for 40%), null if not disclosed.

TEXT:
{text}"""
    },

    "allocation_targets.secondary_funds_max_pct": {
        "schema": SecondaryFundsMaxPctExtraction,
        "prompt": """Extract the maximum allocation target percentage for secondary fund investments.

This is the fund's target maximum allocation to investments in secondary private equity funds.

Look for:
- Investment allocation targets/guidelines
- "X% to Y% in secondary funds"
- Investment strategy constraints

Return just the number (e.g., 70 for 70%), null if not disclosed.

TEXT:
{text}"""
    },

    "allocation_targets.direct_investments_min_pct": {
        "schema": DirectInvestmentsMinPctExtraction,
        "prompt": """Extract the minimum allocation target percentage for direct/co-investments.

This is the fund's target minimum allocation to direct investments or co-investments.

Look for:
- Investment allocation targets/guidelines
- "X% to Y% in direct investments" or "co-investments"
- Investment strategy constraints

Return just the number (e.g., 20 for 20%), null if not disclosed.

TEXT:
{text}"""
    },

    "allocation_targets.direct_investments_max_pct": {
        "schema": DirectInvestmentsMaxPctExtraction,
        "prompt": """Extract the maximum allocation target percentage for direct/co-investments.

This is the fund's target maximum allocation to direct investments or co-investments.

Look for:
- Investment allocation targets/guidelines
- "X% to Y% in direct investments" or "co-investments"
- Investment strategy constraints

Return just the number (e.g., 50 for 50%), null if not disclosed.

TEXT:
{text}"""
    },

    "allocation_targets.secondary_investments_min_pct": {
        "schema": SecondaryInvestmentsMinPctExtraction,
        "prompt": """Extract the minimum allocation target percentage for secondary investments.

This is the fund's target minimum allocation to secondary investments (buying existing fund positions).

Look for:
- "at least X% of net assets in Secondary Investments"
- Investment policy 80% tests
- Investment allocation requirements

Return just the number (e.g., 80 for 80%), null if not disclosed.

TEXT:
{text}"""
    },

    "concentration_limits.max_single_asset_pct": {
        "schema": MaxSingleAssetPctExtraction,
        "prompt": """Extract the maximum single asset concentration limit.

This is the maximum percentage the fund can invest in any single asset or investment.

Look for:
- "no more than X% in any single investment"
- Concentration limits/guidelines
- Diversification requirements

Return just the number (e.g., 25 for 25%), null if not disclosed.

TEXT:
{text}"""
    },

    "concentration_limits.max_single_fund_pct": {
        "schema": MaxSingleFundPctExtraction,
        "prompt": """Extract the maximum single fund concentration limit.

This is the maximum percentage the fund can invest in any single underlying fund.

Look for:
- "no more than X% in any single fund"
- Concentration limits for underlying fund investments
- Diversification requirements

Return just the number (e.g., 25 for 25%), null if not disclosed.

TEXT:
{text}"""
    },

    "concentration_limits.max_single_sector_pct": {
        "schema": MaxSingleSectorPctExtraction,
        "prompt": """Extract the maximum single sector concentration limit.

This is the maximum percentage the fund can invest in any single sector or industry.

Look for:
- "no more than X% in any single sector"
- Industry concentration limits
- Diversification requirements

Return just the number (e.g., 25 for 25%), null if not disclosed.

TEXT:
{text}"""
    },
}


# =============================================================================
# EXTRACTION FUNCTION
# =============================================================================

@dataclass
class Tier3StyleDatapointResult:
    """Result from Tier3-style per-datapoint extraction."""
    datapoint_name: str
    value: Any
    evidence: Optional[str]
    confidence: Optional[str]
    chunks_searched: int
    top_chunk_score: int
    extraction_time_ms: int


@dataclass
class Tier3StyleExtractionTrace:
    """Full trace of Tier3-style per-datapoint extraction."""
    fund_name: str
    total_datapoints: int
    successful_extractions: int
    total_chunks_searched: int
    total_extraction_time_ms: int
    datapoint_results: dict[str, Tier3StyleDatapointResult]


def extract_single_datapoint_tier3_style(
    chunked_doc: ChunkedDocument,
    datapoint_name: str,
    client,
    model: str,
    provider: str,
    max_retries: int = 2,
    top_k_chunks: int = 10,
) -> Tier3StyleDatapointResult:
    """
    Extract a single datapoint using Tier3-style prompt and schema.
    """
    start_time = time.time()

    # Get config for this datapoint
    config = TIER3_STYLE_PROMPTS.get(datapoint_name)
    if not config:
        logger.warning(f"No Tier3-style config for {datapoint_name}")
        return Tier3StyleDatapointResult(
            datapoint_name=datapoint_name,
            value=None,
            evidence=None,
            confidence=None,
            chunks_searched=0,
            top_chunk_score=0,
            extraction_time_ms=int((time.time() - start_time) * 1000),
        )

    schema = config["schema"]
    prompt_template = config["prompt"]

    # Get top chunks using same keyword scoring as per-datapoint
    top_chunks = get_top_chunks_for_datapoint(chunked_doc, datapoint_name, top_k_chunks)

    if not top_chunks:
        return Tier3StyleDatapointResult(
            datapoint_name=datapoint_name,
            value=None,
            evidence=None,
            confidence="not_found",
            chunks_searched=0,
            top_chunk_score=0,
            extraction_time_ms=int((time.time() - start_time) * 1000),
        )

    # Combine chunk content (same as regular Tier3)
    combined_parts = []
    for sc in top_chunks:
        combined_parts.append(f"[Section: {sc.chunk.section_title}]\n{sc.chunk.content}")
    combined_text = "\n\n---\n\n".join(combined_parts)

    # Limit size
    if len(combined_text) > 12000:
        combined_text = combined_text[:12000]

    # Build prompt
    user_prompt = prompt_template.format(text=combined_text)

    try:
        # Use instructor for structured extraction (same as Tier3)
        create_kwargs = {
            "response_model": schema,
            "max_retries": max_retries,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }

        if provider != "google":
            create_kwargs["model"] = model

        if provider == "anthropic":
            create_kwargs["max_tokens"] = 4096

        result = client.chat.completions.create(**create_kwargs)

        # Extract the primary value from the schema
        value = None
        field_name = datapoint_name.split(".")[-1]

        if hasattr(result, field_name):
            value = getattr(result, field_name)
        else:
            # Try to find the value field
            for attr in dir(result):
                if not attr.startswith("_") and attr not in ["evidence_quote", "confidence", "model_fields", "model_config"]:
                    val = getattr(result, attr, None)
                    if val is not None and not callable(val):
                        value = val
                        break

        return Tier3StyleDatapointResult(
            datapoint_name=datapoint_name,
            value=value,
            evidence=getattr(result, "evidence_quote", None),
            confidence=getattr(result, "confidence", None),
            chunks_searched=len(top_chunks),
            top_chunk_score=top_chunks[0].score if top_chunks else 0,
            extraction_time_ms=int((time.time() - start_time) * 1000),
        )

    except Exception as e:
        logger.error(f"Failed to extract {datapoint_name}: {e}")
        return Tier3StyleDatapointResult(
            datapoint_name=datapoint_name,
            value=None,
            evidence=None,
            confidence="not_found",
            chunks_searched=len(top_chunks),
            top_chunk_score=top_chunks[0].score if top_chunks else 0,
            extraction_time_ms=int((time.time() - start_time) * 1000),
        )


class PerDatapointTier3StyleExtractor:
    """
    Per-datapoint extraction using Tier3-style prompts.

    Combines:
    - Per-datapoint keyword selection (granular chunk retrieval)
    - Tier3-style extraction prompts (not Q&A style)
    - Pydantic schema validation via instructor
    - Full system prompt with principles
    """

    DATAPOINTS_TO_EXTRACT = [
        "incentive_fee.has_incentive_fee",
        "incentive_fee.incentive_fee_pct",
        "incentive_fee.hurdle_rate_pct",
        "incentive_fee.hurdle_rate_as_stated",
        "incentive_fee.hurdle_rate_frequency",
        "incentive_fee.high_water_mark",
        "incentive_fee.has_catch_up",
        "incentive_fee.fee_basis",
        "incentive_fee.crystallization_frequency",
        "expense_cap.has_expense_cap",
        "expense_cap.expense_cap_pct",
        "repurchase_terms.repurchase_frequency",
        "repurchase_terms.repurchase_amount_pct",
        "repurchase_terms.repurchase_basis",
        "repurchase_terms.lock_up_period_years",
        "repurchase_terms.early_repurchase_fee_pct",
        "leverage_limits.uses_leverage",
        "leverage_limits.max_leverage_pct",
        "leverage_limits.leverage_basis",
        "distribution_terms.distribution_frequency",
        "distribution_terms.default_distribution_policy",
        "share_classes.share_classes",
    ]

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        delay_between_calls: float = 0.5,
        requests_per_minute: int = 60,
        top_k_chunks: int = 10,
        max_retries: int = 2,
    ):
        self.model = resolve_model_name(model)
        self.provider = provider or detect_provider(model).value
        self.api_key = api_key
        self.top_k_chunks = top_k_chunks
        self.max_retries = max_retries

        self.rate_limit = RateLimitConfig(
            delay_between_calls=delay_between_calls,
            requests_per_minute=requests_per_minute,
        )

        # Use instructor client (same as Tier3)
        self.client = create_instructor_client(
            provider=self.provider,
            model=self.model,
            api_key=api_key,
            rate_limit=self.rate_limit,
        )

    def extract_all(
        self,
        chunked_doc: ChunkedDocument,
        fund_name: str,
    ) -> tuple[dict, Tier3StyleExtractionTrace]:
        """Extract all datapoints using Tier3-style prompts."""
        logger.info(f"[Per-Datapoint Tier3-Style] Extracting {len(self.DATAPOINTS_TO_EXTRACT)} datapoints for {fund_name}")

        start_time = time.time()
        datapoint_results = {}
        extraction_result = {}
        successful = 0
        total_chunks = 0

        for datapoint_name in self.DATAPOINTS_TO_EXTRACT:
            # Skip if no config
            if datapoint_name not in TIER3_STYLE_PROMPTS:
                logger.warning(f"  Skipping {datapoint_name} (no config)")
                continue

            logger.info(f"  Extracting: {datapoint_name}")

            result = extract_single_datapoint_tier3_style(
                chunked_doc=chunked_doc,
                datapoint_name=datapoint_name,
                client=self.client,
                model=self.model,
                provider=self.provider,
                max_retries=self.max_retries,
                top_k_chunks=self.top_k_chunks,
            )

            datapoint_results[datapoint_name] = result
            total_chunks += result.chunks_searched

            if result.value is not None:
                successful += 1
                self._add_to_result(extraction_result, datapoint_name, result.value)

            logger.info(f"    -> {result.value} (score: {result.top_chunk_score}, conf: {result.confidence}, {result.extraction_time_ms}ms)")

        trace = Tier3StyleExtractionTrace(
            fund_name=fund_name,
            total_datapoints=len(self.DATAPOINTS_TO_EXTRACT),
            successful_extractions=successful,
            total_chunks_searched=total_chunks,
            total_extraction_time_ms=int((time.time() - start_time) * 1000),
            datapoint_results=datapoint_results,
        )

        logger.info(f"[Per-Datapoint Tier3-Style] Complete: {successful}/{len(self.DATAPOINTS_TO_EXTRACT)} in {trace.total_extraction_time_ms}ms")

        return extraction_result, trace

    def _add_to_result(self, result: dict, datapoint_name: str, value: Any):
        """Add extracted value to result dict."""
        parts = datapoint_name.split(".")
        if len(parts) == 2:
            field_name, subfield = parts
            if field_name not in result:
                result[field_name] = {}
            result[field_name][subfield] = value
        else:
            result[datapoint_name] = value
