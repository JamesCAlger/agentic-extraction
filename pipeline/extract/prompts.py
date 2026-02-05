"""
Extraction prompts for LLM-based field extraction.

Each prompt is designed to extract a specific field or set of related fields
from document sections. Prompts include:
- Clear instructions
- Expected format
- Examples of what to look for
- Guidance on confidence levels
"""

# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are a financial document analyst specializing in SEC filings for interval and tender offer funds. Your task is to extract specific structured data from fund prospectus sections.

Key principles:
1. ACCURACY: Only extract values explicitly stated or clearly calculable from the text
2. CITATIONS: Always provide verbatim quotes supporting your extraction
3. CONFIDENCE: Mark confidence as:
   - "explicit": Value directly stated in text
   - "inferred": Value derived from calculation or clear implication
   - "not_found": Value not present in provided text
4. COMPLETENESS: If partial information is available, extract what you can

For fund-of-funds, distinguish between:
- Fund-level fees (charged by the fund manager to investors)
- Underlying fund fees (charged by portfolio funds, passed through to investors)"""


# =============================================================================
# FIELD-SPECIFIC PROMPTS
# =============================================================================

# =============================================================================
# NEW v2.0 PROMPTS - Fund Metadata, Leverage, Distribution
# =============================================================================

FUND_METADATA_PROMPT = """Extract basic fund metadata from this text.

Look for:
- Full legal name of the fund
- Investment manager/adviser name
- Sponsor organization
- Fiscal year end (e.g., "March 31", "December 31")
- Fund type (interval fund, tender offer fund, BDC)
- Number of share classes offered

Common patterns:
- "[Fund Name] is a [type] fund"
- "managed by [Manager Name]"
- "fiscal year end is [date]"
- "We offer [X] classes of shares"

TEXT:
{text}"""


LEVERAGE_LIMITS_PROMPT = """Extract leverage and borrowing information from this text.

Look for:
- Whether the fund uses leverage/borrowing
- Maximum leverage percentage or ratio
- Basis for leverage (total assets, net assets, debt-to-equity, asset coverage)
- Credit facility details

=== LEVERAGE FORMAT NORMALIZATION ===

CRITICAL: The max_leverage_pct field should contain the NORMALIZED percentage of total assets
that can be borrowed, NOT the raw asset coverage ratio.

FORMAT RECOGNITION AND CONVERSION:

1. ASSET COVERAGE RATIO (MOST COMMON):
   Document says: "300% asset coverage requirement"
   CONVERT: max_leverage_pct = 100 / (300/100) = 33.33
   leverage_basis = "asset_coverage"

   Common conversions:
   - 300% coverage → max_leverage_pct = 33.33 (standard 1940 Act limit)
   - 200% coverage → max_leverage_pct = 50
   - 150% coverage → max_leverage_pct = 66.67

2. DEBT-TO-EQUITY RATIO:
   Document says: "50% debt-to-equity ratio"
   CONVERT: max_leverage_pct = 50 / (100 + 50) * 100 = 33.33
   leverage_basis = "debt_to_equity"

   Common conversions:
   - 50% D/E → max_leverage_pct = 33.33
   - 100% D/E → max_leverage_pct = 50
   - 33% D/E → max_leverage_pct = 25

3. PERCENTAGE OF ASSETS (DIRECT - NO CONVERSION):
   Document says: "may borrow up to 33% of total assets"
   max_leverage_pct = 33 (use as-is)
   leverage_basis = "total_assets" or "net_assets"

4. 1940 ACT REFERENCE:
   Document says: "in accordance with the 1940 Act"
   max_leverage_pct = 33.33 (statutory limit)
   leverage_basis = "asset_coverage"

CRITICAL OUTPUT FORMAT:
- max_leverage_pct should ALWAYS be the normalized "percentage of assets that can be borrowed"
- Typical values: 33, 33.33, 50, 25
- Do NOT return raw coverage ratios (300, 200) as max_leverage_pct
- Store the original format type in leverage_basis

Common 1940 Act patterns (look for these first):
- "asset coverage (as defined in the 1940 Act) of 300%" → 33.33%
- "maintain an asset coverage ratio of at least 300%" → 33.33%

=== WORKED EXAMPLES ===

EXAMPLE 1 - Asset Coverage (Most Common):
Text: "The Fund's borrowings may not exceed an amount equal to 33-1/3% of total assets. This complies with the 1940 Act requirement of 300% asset coverage."
CORRECT: max_leverage_pct = 33 (or 33.33), leverage_basis = "asset_coverage"
WRONG: max_leverage_pct = 300 (this is coverage ratio, not leverage %)

EXAMPLE 2 - Debt-to-Equity:
Text: "The Fund may maintain a debt-to-equity ratio of up to 50%."
CORRECT: max_leverage_pct = 33.33 (50/(100+50)*100), leverage_basis = "debt_to_equity"
WRONG: max_leverage_pct = 50 (this is D/E ratio, not % of total assets)

EXAMPLE 3 - BDC Higher Leverage:
Text: "As a BDC, the Fund may borrow up to 200% of net assets under the 150% asset coverage requirement."
CORRECT: max_leverage_pct = 66.67 (BDC: 100/1.5 = 66.67%), leverage_basis = "asset_coverage"

EXAMPLE 4 - Explicit % of Assets:
Text: "The Fund may borrow up to 15% of its gross assets for temporary purposes."
CORRECT: max_leverage_pct = 15, leverage_basis = "total_assets"
(No conversion needed - already expressed as % of assets)

TEXT:
{text}"""


DISTRIBUTION_TERMS_PROMPT = """Extract distribution/dividend policy from this text.

CRITICAL DISTINCTION - DO NOT CONFUSE THESE TWO DIFFERENT CONCEPTS:
1. DISTRIBUTION/DIVIDEND FREQUENCY = How often the fund PAYS DIVIDENDS to shareholders
   - Look for: "pay distributions monthly", "declare daily distributions", "monthly distributions"
   - This is about INCOME PAYMENTS to investors

2. REPURCHASE FREQUENCY = How often the fund offers to BUY BACK shares (liquidity events)
   - Look for: "quarterly repurchase offers", "tender offers"
   - This is about REDEMPTIONS, NOT distributions
   - IGNORE repurchase frequency when extracting distribution_frequency

Look for:
- Distribution/dividend frequency (monthly, quarterly, annual) - when dividends are PAID
- Default distribution policy (cash, reinvested/DRIP)
- Source of distributions
- Target distribution rate if stated (as a NUMBER, or null if not stated)

IMPORTANT: For numeric fields like target_distribution_rate:
- Return a number (e.g., 5 for 5%) if explicitly stated
- Return null if not found or not applicable
- NEVER return strings like "not_found" or "N/A" for numeric fields

Common patterns for DISTRIBUTION frequency (what you want):
- "pay distributions on a monthly basis"
- "declare daily distributions, pay monthly"
- "distributions will be paid [frequency]"
- "dividend reinvestment plan (DRIP)"

Patterns to IGNORE (these are repurchase, not distribution):
- "quarterly repurchase offers" - this is redemption frequency
- "tender offers on a quarterly basis" - this is redemption frequency

=== WORKED EXAMPLES ===

EXAMPLE 1 - Monthly Distribution:
Text: "The Fund intends to pay distributions monthly on or about the 15th of each month."
CORRECT: distribution_frequency = "monthly"

EXAMPLE 2 - Don't Confuse with Repurchase:
Text: "The Fund conducts quarterly repurchase offers. The Fund expects to pay distributions annually."
CORRECT: distribution_frequency = "annual" (ignore the quarterly repurchase)
WRONG: distribution_frequency = "quarterly" (this is repurchase, not distribution)

EXAMPLE 3 - No Distribution Frequency Stated:
Text: "The Fund operates an opt-out dividend reinvestment plan (DRIP)."
CORRECT: distribution_frequency = null (DRIP is default policy, not frequency)
CORRECT: default_distribution_policy = "DRIP"

EXAMPLE 4 - Common Mistake:
Text: "quarterly incentive fees are accrued and paid quarterly in arrears"
WRONG: distribution_frequency = "quarterly" (this is incentive fee, not dividend)
CORRECT: distribution_frequency = null (no distribution info here)

TEXT:
{text}"""


INCENTIVE_FEE_PROMPT = """Extract incentive/performance fee information from this text.

Look for:
- Performance-based fees, incentive fees, carried interest at the FUND LEVEL
- Hurdle rates, preferred returns, high water marks
- For fund-of-funds: underlying fund incentive fee ranges (often 15-20%)

=== SEMANTIC TAXONOMY FOR INCENTIVE FEE STRUCTURES ===

HIGH WATER MARK MECHANISMS (set high_water_mark = true for ALL of these):
1. Explicit "high water mark" or "HWM" language
2. "Loss Recovery Account" - tracks cumulative losses, prevents fees on recovered losses
3. "Loss Carryforward Account" - same as Loss Recovery Account
4. Any mechanism that prevents paying incentive on gains that merely recover prior losses

CRITICAL: "Loss Recovery Account" IS a high water mark mechanism.
Example: "Incentive Fee equal to 10% of net profits OVER the then balance of the Loss Recovery Account"
→ high_water_mark = true (the Loss Recovery Account IS a high water mark)

CATCH-UP MECHANISMS (set has_catch_up = true):
1. Explicit "catch-up" or "full catch-up" language
2. "100% of returns between X% and Y%"
3. Manager receives accelerated payment above hurdle until reaching full fee percentage

CRITICAL DISTINCTION:
- HIGH WATER MARK = prevents paying fees on recovered losses (Loss Recovery Account = HWM)
- CATCH-UP = accelerated fee payment between hurdle and ceiling
- These are DIFFERENT mechanisms. A fund can have both, one, or neither.
- Loss Recovery Account is NOT catch-up.

WHEN TO SET NULL vs FALSE:
- high_water_mark = null → Only if no incentive fee exists (has_incentive_fee = false)
- high_water_mark = false → Incentive fee exists but NO high water mark language found
- has_catch_up = null → Only if no incentive fee exists OR no clear indication

HURDLE RATE ANNUALIZATION:
- If stated as quarterly rate (e.g., "1.50% per quarter"), multiply by 4 for hurdle_rate_pct
  Example: "1.50% per quarter" → hurdle_rate_pct = 6 (annualized)
- hurdle_rate_frequency indicates how often measured (quarterly), not the annualized rate

IMPORTANT DISTINCTIONS:
- Interest expense on borrowings is NOT an incentive fee
- Management fees are NOT incentive fees
- Only extract fees tied to PERFORMANCE or PROFITS

If no fund-level incentive fee exists, set has_incentive_fee=False but still extract underlying fund fees if this is a fund-of-funds.

=== WORKED EXAMPLES FOR HURDLE RATE ===

EXAMPLE 1 - Quarterly Hurdle Stated, Needs Annualization:
Text: "subject to a 1.25% quarterly hurdle rate (5% annualized)"
CORRECT:
  hurdle_rate_pct = 5 (annualized)
  hurdle_rate_as_stated = "1.25" (quarterly, raw)
  hurdle_rate_frequency = "quarterly"

EXAMPLE 2 - Annual Hurdle:
Text: "subject to a 6% annual hurdle rate"
CORRECT:
  hurdle_rate_pct = 6
  hurdle_rate_as_stated = "6"
  hurdle_rate_frequency = "annual"

EXAMPLE 3 - No Hurdle Rate (Loss Recovery Account Fund):
Text: "Incentive Fee equal to 12.5% of net profits over the Loss Carryforward Account balance"
CORRECT:
  hurdle_rate_pct = null (no explicit hurdle - Loss Carryforward is NOT a hurdle)
  hurdle_rate_as_stated = null
  hurdle_rate_frequency = null
  high_water_mark = true (Loss Carryforward IS a high water mark)

EXAMPLE 4 - Fund-of-Funds (No Fund-Level Incentive):
Text: "The Fund does not charge an incentive fee. The underlying Investment Funds typically charge incentive fees of 15% to 20%."
CORRECT:
  has_incentive_fee = false
  incentive_fee_pct = null
  hurdle_rate_pct = null
  underlying_fund_incentive_range = "15% to 20%"

TEXT:
{text}"""


EXPENSE_CAP_PROMPT = """Extract expense cap or fee waiver information from this text.

Look for:
- Contractual expense limitations
- Fee waivers (voluntary or contractual)
- Expense reimbursement agreements
- Cap expiration dates
- Recoupment provisions

Common patterns:
- "agreed to waive fees...to the extent necessary to limit total annual operating expenses to X%"
- "expense cap of X% through [date]"
- "may recoup waived fees within X years"

TEXT:
{text}"""


REPURCHASE_TERMS_PROMPT = """Extract share repurchase or tender offer terms from this text.

This is critical for interval and tender offer funds. Look for:

1. FUND TYPE:
   - Interval fund: mandatory periodic repurchases (Rule 23c-3)
   - Tender offer fund: discretionary repurchases

2. FREQUENCY:
   - Quarterly, semi-annual, annual, or at board discretion

3. REPURCHASE AMOUNT:
   - Minimum percentage (often 5%)
   - Maximum percentage (often 25%)
   - Typical/target percentage if stated
   - Basis: "number of shares" or "net assets" or "NAV"

4. TIMING:
   - Notice period for requests (in days)
   - Pricing date (when NAV is determined)

5. LOCK-UP AND EARLY REDEMPTION (CRITICAL):
   - Lock-up period: Time before first repurchase eligibility
     - Often "1 year" or "12 months" from purchase
     - Extract as years (e.g., lock_up_period_years = 1)
   - Early repurchase/redemption fee: Fee charged for early redemption
     - Often "2%" for shares held less than 1 year
     - Extract as percentage (e.g., early_repurchase_fee_pct = 2)
   - Period for early fee: "within 1 year of purchase", "first 12 months"

6. MINIMUMS:
   - Minimum repurchase request amount
   - Minimum holding after repurchase

Common patterns:
- "repurchase offers...between 5% and 25% of outstanding shares"
- "quarterly repurchase offer for up to 5% of net assets"
- "2% early repurchase fee for shares held less than one year"
- "early withdrawal charge of 2%"
- "shareholders must hold shares for at least one year"
- "lock-up period of 12 months"
- "repurchase requests must be submitted X days before"
- "minimum repurchase amount of $500"

TEXT:
{text}"""


ALLOCATION_TARGETS_PROMPT = """Extract target asset allocation information from this text.

Look for:
- Target allocation percentages by asset class or strategy
- Allocation ranges (minimum/maximum)
- Types of investments (private equity, credit, real estate, etc.)

Common patterns:
- "target allocation of X% to [asset class]"
- "invest between X% and Y% in [strategy]"
- "approximately X% in private equity, Y% in credit"
- "allocate across multiple strategies including..."

Also extract:
- Allocation approach (opportunistic vs strategic)
- Rebalancing policy

TEXT:
{text}"""


CONCENTRATION_LIMITS_PROMPT = """Extract investment concentration limits from this text.

Look for limits on:
- Single issuer exposure
- Single fund allocation (for fund-of-funds)
- Industry/sector concentration
- Geographic concentration
- Illiquid securities

Common patterns:
- "no more than X% of net assets in any single issuer"
- "limit exposure to any single fund to X%"
- "at least X% of assets in diversified holdings"

TEXT:
{text}"""


SHARE_CLASSES_PROMPT = """Extract share class details from this text.

***CRITICAL ANTI-HALLUCINATION WARNING FOR MINIMUM INVESTMENTS***
MINIMUM INVESTMENTS VARY DRAMATICALLY BETWEEN FUNDS. DO NOT:
- Copy values from examples or similar funds
- Assume $2,500 or $1,000,000 are typical - many funds have NO fund-level minimums
- Invent values that are not EXPLICITLY stated in this document

ONLY extract a minimum investment if you can QUOTE the exact dollar amount from the text.
If no explicit amount is stated, return null. NULL IS CORRECT for many funds.

IMPORTANT - WHEN TO RETURN NULL FOR MINIMUMS:
Return null (not a number) for minimum investments when:
- The document says minimums are "determined by financial intermediaries" or "set by selling agents"
- The document says "no minimum investment at the fund level"
- The document says minimums are "subject to the policies of your financial intermediary"
- Minimums are NOT stated for a specific share class
- You cannot find an EXACT dollar amount in the text
- Only a fee table is present with NO minimum investment section
Each fund is DIFFERENT - do not assume minimums from other funds apply here.

IMPORTANT - NULL vs 0 HANDLING FOR NUMERIC FIELDS:
For ANY numeric field (minimums, fees, percentages), you MUST return:
- A NUMBER if the value is stated (e.g., 5000, 3.5, 0.25)
- 0 (zero) if the document explicitly states the fee does not apply or is waived
- null ONLY if the field is not mentioned at all in the text
- NEVER return strings like "Not specified", "N/A", "None", or "not found"

CRITICAL - When to return 0 vs null for FEES:
- Return 0 when: "Class I is not subject to any distribution fees" → 0
- Return 0 when: "No sales load" → 0
- Return 0 when: "No upfront sales charge" (but check for intermediary cap for sales_load_pct)
- Return null when: The fee is simply not mentioned anywhere in the text

For EACH share class mentioned, extract ALL of these fields:

1. Class name (e.g., "Class I", "Class S", "Class D", "Class I Advisory")

2. MINIMUM INVESTMENTS (return numbers or null):
   - minimum_initial_investment: Dollar amount for first purchase - ONLY if explicitly stated
   - minimum_additional_investment: Dollar amount for subsequent purchases - ONLY if explicitly stated
   - minimum_balance_for_repurchase: Minimum balance to request repurchase or null

3. Investor eligibility (institutional, retail, accredited, etc.) - string or null

4. Distribution channel (advisory programs, broker-dealers, fee-based programs, etc.) - string or null

5. FEES (return numbers or null, NEVER strings):
   - sales_load_pct: Front-end sales load/charge OR placement fee cap (e.g., 3.5 for 3.5%)
     IMPORTANT: If the fund itself charges no upfront sales load but allows intermediaries/
     selling agents to charge placement fees UP TO A MAXIMUM, use that maximum as sales_load_pct.
     Examples:
     - "No upfront sales charge but intermediaries may charge up to 3.5%" → sales_load_pct = 3.5
     - "Upfront placement fee of up to 1.5%" → sales_load_pct = 1.5
     - "Maximum commission of 3.0%" → sales_load_pct = 3.0
     Rationale: Investors will likely pay the maximum allowed by intermediaries.
   - distribution_servicing_fee_pct: Combined distribution AND/OR servicing fee (see below)
   - management_fee_pct: Annual management/advisory fee percentage
   - affe_pct: Acquired fund fees and expenses (for fund-of-funds)
   - interest_expense_pct: Interest expenses on borrowings
   - other_expenses_pct: Other annual expenses
   - total_expense_ratio_pct: Total annual expenses before fee waivers
   - net_expense_ratio_pct: Net annual expenses after fee waivers
   - fee_waiver_pct: Fee waiver/reimbursement percentage
   - incentive_fee_xbrl_pct: Incentive/performance fee as % of net assets from fee table

   WHERE TO FIND THESE FEES:
   Look in "Annual Fund Operating Expenses" tables, "Fee Table", "Summary of Fund Expenses",
   or "Fees and Expenses" sections. These are typically presented as percentages of net assets.

6. offering_price_basis: "NAV", "NAV plus sales load", etc. - string or null

CRITICAL - WHERE TO FIND MINIMUM INVESTMENTS:
Look for these patterns and extract ONLY if you find exact dollar amounts:
- "minimum initial investment of $X"
- "minimum investment for Class S is $X"
- "minimum subsequent investment of $X"
- Look in "PURCHASE OF SHARES", "PLAN OF DISTRIBUTION", "HOW TO PURCHASE" sections
If you search these sections and find NO dollar amounts for minimums, return null.

CRITICAL - distribution_servicing_fee_pct:
This is a SINGLE combined field for all ongoing annual fees related to distribution or servicing.
Include ANY fee labeled as:
- "distribution fee" or "12b-1 fee"
- "shareholder servicing fee" or "service fee"
- "distribution and servicing fee" (combined)

EXAMPLES:
- "Class S is subject to an annual distribution fee of 0.85%" → 0.85
- "Class D has an annual shareholder servicing fee of 0.25%" → 0.25
- "Class S distribution and servicing fee of 0.85%" → 0.85
- "Class I is not subject to any distribution or servicing fees" → 0

These fees are economically equivalent to investors - combine them into one field.

Common patterns:
- "Class I shares require a minimum investment of $1,000,000"
- "Class S minimum initial investment is $2,500"
- "minimum subsequent investment of $500"
- "Class S subject to distribution fee of 0.85%" → distribution_servicing_fee_pct = 0.85
- "Class D annual shareholder servicing fee of 0.25%" → distribution_servicing_fee_pct = 0.25

TEXT:
{text}"""


# =============================================================================
# DISCOVERY-FIRST SHARE CLASS EXTRACTION PROMPT
# =============================================================================

# This prompt is used when discovery has already identified which share classes
# exist in the document. It provides the LLM with the known class names upfront.

SHARE_CLASSES_WITH_DISCOVERY_PROMPT = """Extract share class details from this text.

***KNOWN SHARE CLASSES IN THIS DOCUMENT***
The following share classes have been identified in this document:
{discovered_classes}

You MUST extract details for EXACTLY these classes - no more, no less.
Do NOT extract any class not in this list. Do NOT invent additional classes.

***CRITICAL ANTI-HALLUCINATION WARNING FOR MINIMUM INVESTMENTS***
MINIMUM INVESTMENTS VARY DRAMATICALLY BETWEEN FUNDS. DO NOT:
- Copy values from examples or similar funds
- Assume $2,500 or $1,000,000 are typical - many funds have NO fund-level minimums
- Invent values that are not EXPLICITLY stated in this document

ONLY extract a minimum investment if you can QUOTE the exact dollar amount from the text.
If no explicit amount is stated, return null. NULL IS CORRECT for many funds.

IMPORTANT - WHEN TO RETURN NULL FOR MINIMUMS:
Return null (not a number) for minimum investments when:
- The document says minimums are "determined by financial intermediaries" or "set by selling agents"
- The document says "no minimum investment at the fund level"
- The document says minimums are "subject to the policies of your financial intermediary"
- Minimums are NOT stated for a specific share class
- You cannot find an EXACT dollar amount in the text
- Only a fee table is present with NO minimum investment section
Each fund is DIFFERENT - do not assume minimums from other funds apply here.

IMPORTANT - NULL vs 0 HANDLING FOR NUMERIC FIELDS:
For ANY numeric field (minimums, fees, percentages), you MUST return:
- A NUMBER if the value is stated (e.g., 5000, 3.5, 0.25)
- 0 (zero) if the document explicitly states the fee does not apply or is waived
- null ONLY if the field is not mentioned at all in the text
- NEVER return strings like "Not specified", "N/A", "None", or "not found"

CRITICAL - When to return 0 vs null for FEES:
- Return 0 when: "Class I is not subject to any distribution fees" → 0
- Return 0 when: "No sales load" → 0
- Return 0 when: "No upfront sales charge" (but check for intermediary cap for sales_load_pct)
- Return null when: The fee is simply not mentioned anywhere in the text

For EACH of the known share classes listed above, extract ALL of these fields:

1. Class name (must match one of: {discovered_classes})

2. MINIMUM INVESTMENTS (return numbers or null):
   - minimum_initial_investment: Dollar amount for first purchase - ONLY if explicitly stated
   - minimum_additional_investment: Dollar amount for subsequent purchases - ONLY if explicitly stated
   - minimum_balance_for_repurchase: Minimum balance to request repurchase or null

3. Investor eligibility (institutional, retail, accredited, etc.) - string or null

4. Distribution channel (advisory programs, broker-dealers, fee-based programs, etc.) - string or null

5. FEES (return numbers or null, NEVER strings):
   - sales_load_pct: Front-end sales load/charge OR placement fee cap (e.g., 3.5 for 3.5%)
     IMPORTANT: If the fund itself charges no upfront sales load but allows intermediaries/
     selling agents to charge placement fees UP TO A MAXIMUM, use that maximum as sales_load_pct.
   - distribution_servicing_fee_pct: Combined distribution AND/OR servicing fee
   - management_fee_pct: Annual management/advisory fee percentage
   - affe_pct: Acquired fund fees and expenses (for fund-of-funds)
   - interest_expense_pct: Interest expenses on borrowings
   - other_expenses_pct: Other annual expenses
   - total_expense_ratio_pct: Total annual expenses before fee waivers
   - net_expense_ratio_pct: Net annual expenses after fee waivers
   - fee_waiver_pct: Fee waiver/reimbursement percentage
   - incentive_fee_xbrl_pct: Incentive/performance fee as % of net assets from fee table

   WHERE TO FIND THESE FEES:
   Look in "Annual Fund Operating Expenses" tables, "Fee Table", "Summary of Fund Expenses",
   or "Fees and Expenses" sections. These are typically presented as percentages of net assets.

6. offering_price_basis: "NAV", "NAV plus sales load", etc. - string or null

CRITICAL - WHERE TO FIND MINIMUM INVESTMENTS:
Look for these patterns and extract ONLY if you find exact dollar amounts:
- "minimum initial investment of $X"
- "minimum investment for Class S is $X"
- "minimum subsequent investment of $X"
- Look in "PURCHASE OF SHARES", "PLAN OF DISTRIBUTION", "HOW TO PURCHASE" sections
If you search these sections and find NO dollar amounts for minimums, return null.

TEXT:
{text}"""


def format_share_classes_prompt_with_discovery(
    text: str,
    discovered_classes: list[str],
) -> str:
    """
    Format the share classes prompt with discovered class names.

    Args:
        text: The document text to extract from
        discovered_classes: List of share class names discovered in the document

    Returns:
        Formatted prompt string
    """
    classes_str = ", ".join(discovered_classes)
    return SHARE_CLASSES_WITH_DISCOVERY_PROMPT.format(
        discovered_classes=classes_str,
        text=text,
    )


# =============================================================================
# TWO-PASS SHARE CLASS EXTRACTION PROMPTS
# =============================================================================

SHARE_CLASS_DISCOVERY_PROMPT = """Identify ALL share classes mentioned in this SEC filing.

TASK: List every distinct share class name you find. Do NOT extract any values yet - just list the class names.

LOOK FOR patterns like:
- "Class I", "Class S", "Class D", "Class U", "Class R", "Class T", "Class A", "Class C"
- "Class I Shares", "Class S Shares" (normalize to "Class I", "Class S")
- "Institutional Class", "Advisor Class" (use exact name found if no "Class X" format)
- "I Shares", "S Shares", "D Shares" (normalize to "Class I", "Class S", "Class D")
- Any other share class designations

NORMALIZATION RULES:
1. Remove "Shares" suffix: "Class I Shares" → "Class I"
2. Standardize format: "I Shares" → "Class I"
3. Preserve unique names: "Institutional Class" stays as-is if no "Class X" equivalent
4. De-duplicate: If same class appears multiple times, list only once

CRITICAL:
- Include ALL classes mentioned, even if only referenced once
- Do NOT invent classes - only list what appears in the text
- If unsure whether something is a share class, include it
- Common share classes include: Class S, Class D, Class I, Class T, Class U, Class R, Class A, Class C

Return the list of share class names found in normalized format."""


SHARE_CLASS_FIELDS_PROMPT = """Extract details for {class_name} from this SEC filing.

You are extracting information for ONE specific share class: {class_name}

For {class_name}, find and extract:

1. MINIMUM INVESTMENTS (return numbers in dollars, or null if not stated):
   - minimum_initial_investment: Dollar amount for first purchase
   - minimum_additional_investment: Dollar amount for subsequent purchases
   - minimum_balance_for_repurchase: Minimum balance to request repurchase

2. INVESTOR ELIGIBILITY:
   - investor_eligibility: Who can invest (e.g., "institutional investors", "accredited investors")
   - distribution_channel: How shares are sold (e.g., "fee-based advisory programs", "broker-dealers")

3. FEES (return percentages as numbers, e.g., 3.5 for 3.5%):
   - sales_load_pct: Front-end sales load or maximum placement fee cap
   - distribution_servicing_fee_pct: Combined 12b-1/distribution/servicing fee
   - management_fee_pct: Annual management/advisory fee percentage
   - affe_pct: Acquired fund fees and expenses (for fund-of-funds)
   - interest_expense_pct: Interest expenses on borrowings
   - other_expenses_pct: Other annual expenses
   - total_expense_ratio_pct: Total annual expenses before fee waivers
   - net_expense_ratio_pct: Net annual expenses after fee waivers
   - fee_waiver_pct: Fee waiver/reimbursement percentage
   - incentive_fee_xbrl_pct: Incentive/performance fee as % of net assets from fee table

4. OFFERING PRICE:
   - offering_price_basis: "NAV", "NAV plus sales load", etc.

CRITICAL RULES:
- ONLY extract values explicitly associated with {class_name}
- Do NOT use values from other share classes
- Return null for any field without explicit {class_name} data
- Quote the exact text where you found each value in evidence_quote
- If the document says minimums are "set by financial intermediaries" or similar, return null

SEARCH PATTERNS for {class_name}:
- "{class_name} minimum investment"
- "{class_name} shares" + "minimum" + "$"
- Fee tables with {class_name} column/row
- "PLAN OF DISTRIBUTION" sections mentioning {class_name}
- "PURCHASE OF SHARES" sections with {class_name} details

TEXT:
{{text}}"""


# =============================================================================
# SECTION-TO-PROMPT MAPPING
# =============================================================================

SECTION_PROMPT_MAP = {
    # XBRL text blocks
    "cef:AcquiredFundFeesAndExpensesNote": [
        ("incentive_fee", INCENTIVE_FEE_PROMPT),
        ("expense_cap", EXPENSE_CAP_PROMPT),
    ],
    "cef:ManagementFeeNotBasedOnNetAssetsNote": [
        ("incentive_fee", INCENTIVE_FEE_PROMPT),
    ],
    "cef:OtherExpensesNote": [
        ("expense_cap", EXPENSE_CAP_PROMPT),
    ],
    "cef:InvestmentObjectivesAndPracticesTextBlock": [
        ("allocation_targets", ALLOCATION_TARGETS_PROMPT),
        ("concentration_limits", CONCENTRATION_LIMITS_PROMPT),
        ("leverage_limits", LEVERAGE_LIMITS_PROMPT),
    ],

    # Heading-based sections (normalized titles)
    "repurchases_of_shares": [
        ("repurchase_terms", REPURCHASE_TERMS_PROMPT),
    ],
    "repurchase_of_shares": [
        ("repurchase_terms", REPURCHASE_TERMS_PROMPT),
    ],
    "repurchase_offers": [
        ("repurchase_terms", REPURCHASE_TERMS_PROMPT),
    ],
    "periodic_repurchase_offers": [
        ("repurchase_terms", REPURCHASE_TERMS_PROMPT),
    ],
    "interval_fund_repurchases": [
        ("repurchase_terms", REPURCHASE_TERMS_PROMPT),
    ],
    "tender_offer": [
        ("repurchase_terms", REPURCHASE_TERMS_PROMPT),
    ],
    "tender_offers": [
        ("repurchase_terms", REPURCHASE_TERMS_PROMPT),
    ],
    "liquidity": [
        ("repurchase_terms", REPURCHASE_TERMS_PROMPT),
    ],
    "shareholder_liquidity": [
        ("repurchase_terms", REPURCHASE_TERMS_PROMPT),
    ],
    "plan_of_distribution": [
        ("share_classes", SHARE_CLASSES_PROMPT),
    ],
    "purchasing_shares": [
        ("share_classes", SHARE_CLASSES_PROMPT),
    ],
    "purchase_of_shares": [
        ("share_classes", SHARE_CLASSES_PROMPT),
    ],
    "how_to_purchase": [
        ("share_classes", SHARE_CLASSES_PROMPT),
    ],
    # Additional sections for share_classes (multi-section support)
    "fee_table": [
        ("share_classes", SHARE_CLASSES_PROMPT),
    ],
    "fees_and_expenses": [
        ("share_classes", SHARE_CLASSES_PROMPT),
    ],
    "summary_of_fund_expenses": [
        ("share_classes", SHARE_CLASSES_PROMPT),
    ],
    "distribution_arrangements": [
        ("share_classes", SHARE_CLASSES_PROMPT),
    ],
    "shareholder_information": [
        ("share_classes", SHARE_CLASSES_PROMPT),
    ],
    "investment_objectives": [
        ("allocation_targets", ALLOCATION_TARGETS_PROMPT),
    ],
    "investment_strategies": [
        ("allocation_targets", ALLOCATION_TARGETS_PROMPT),
        ("concentration_limits", CONCENTRATION_LIMITS_PROMPT),
    ],
    "investment_restrictions": [
        ("concentration_limits", CONCENTRATION_LIMITS_PROMPT),
    ],
    # NEW v2.0 sections
    "leverage": [
        ("leverage_limits", LEVERAGE_LIMITS_PROMPT),
    ],
    "borrowing": [
        ("leverage_limits", LEVERAGE_LIMITS_PROMPT),
    ],
    "use_of_leverage": [
        ("leverage_limits", LEVERAGE_LIMITS_PROMPT),
    ],
    "distributions": [
        ("distribution_terms", DISTRIBUTION_TERMS_PROMPT),
    ],
    "dividend_policy": [
        ("distribution_terms", DISTRIBUTION_TERMS_PROMPT),
    ],
    "distribution_policy": [
        ("distribution_terms", DISTRIBUTION_TERMS_PROMPT),
    ],
    "prospectus_summary": [
        ("fund_metadata", FUND_METADATA_PROMPT),
    ],
    "summary": [
        ("fund_metadata", FUND_METADATA_PROMPT),
    ],
}


def get_prompt_for_field(field_name: str) -> str:
    """Get the extraction prompt for a specific field (without examples)."""
    prompts = {
        # Original fields
        "incentive_fee": INCENTIVE_FEE_PROMPT,
        "expense_cap": EXPENSE_CAP_PROMPT,
        "repurchase_terms": REPURCHASE_TERMS_PROMPT,
        "allocation_targets": ALLOCATION_TARGETS_PROMPT,
        "concentration_limits": CONCENTRATION_LIMITS_PROMPT,
        "share_classes": SHARE_CLASSES_PROMPT,
        # NEW v2.0 fields
        "fund_metadata": FUND_METADATA_PROMPT,
        "leverage_limits": LEVERAGE_LIMITS_PROMPT,
        "distribution_terms": DISTRIBUTION_TERMS_PROMPT,
    }
    return prompts.get(field_name, "")


def get_prompt_with_examples(
    field_name: str,
    max_examples: int = 3,
    include_notes: bool = True,
    document_text: str = None,
) -> str:
    """
    Get the extraction prompt with few-shot examples.

    Args:
        field_name: The field to extract (e.g., "repurchase_terms")
        max_examples: Maximum number of examples to include
        include_notes: Whether to include explanatory notes
        document_text: Optional document text for dynamic example selection

    Returns:
        Complete prompt with examples
    """
    from .examples import get_examples_for_field, format_examples_for_prompt

    # Get base prompt
    base_prompt = get_prompt_for_field(field_name)
    if not base_prompt:
        return ""

    # Get examples - use dynamic selection if document_text is provided
    examples = get_examples_for_field(field_name, max_examples, document_text=document_text)

    if not examples:
        return base_prompt

    # Format examples
    examples_text = format_examples_for_prompt(examples, include_notes)

    # Escape curly braces in examples to avoid format() conflicts
    # The examples contain JSON with { and } which would be interpreted as placeholders
    examples_text_escaped = examples_text.replace("{", "{{").replace("}", "}}")

    # Insert examples before the TEXT placeholder
    # Find where {text} placeholder is
    if "{text}" in base_prompt:
        parts = base_prompt.rsplit("{text}", 1)
        prompt_with_examples = parts[0] + examples_text_escaped + "\n\nNow extract from this text:\n{text}"
        if len(parts) > 1:
            prompt_with_examples += parts[1]
        return prompt_with_examples
    else:
        return base_prompt + examples_text_escaped


def normalize_section_title(title: str) -> str:
    """Normalize section title for mapping lookup."""
    import re
    # Convert to lowercase, remove special chars, replace spaces with underscores
    normalized = title.lower()
    normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
    normalized = re.sub(r'\s+', '_', normalized.strip())
    return normalized
