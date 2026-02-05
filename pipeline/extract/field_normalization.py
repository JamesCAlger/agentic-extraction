"""
Field Normalization Utilities

This module provides normalization functions for fields that can be expressed
in multiple formats across different SEC filings. The goal is to convert
various document expressions to a consistent, comparable format.

Key normalizations:
1. Leverage - converts asset coverage, debt-to-equity, etc. to % of assets
2. Incentive fee semantics - maps Loss Recovery Account to high_water_mark, etc.
"""

import re
import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# LEVERAGE FORMAT NORMALIZATION
# =============================================================================

@dataclass
class LeverageNormalization:
    """
    Normalized leverage representation.

    All leverage limits are normalized to "percentage of total assets that can be borrowed".
    This is the most intuitive format for comparison.
    """
    raw_value: str                    # Original extracted value (e.g., "300%")
    raw_format: str                   # Detected format type
    normalized_pct: Optional[float]   # Normalized to % of total assets
    source_quote: str                 # Evidence from document
    confidence: float                 # 0-1 confidence in normalization
    notes: str                        # Explanation of conversion


# Leverage format types with conversion logic
LEVERAGE_FORMATS = {
    "asset_coverage": {
        "patterns": [
            r"(\d+(?:\.\d+)?)\s*%?\s*asset\s*coverage",
            r"asset\s*coverage\s*(?:of\s*)?(?:at\s*least\s*)?(\d+(?:\.\d+)?)\s*%",
            r"(\d+(?:\.\d+)?)\s*%\s*coverage\s*(?:ratio|requirement)",
        ],
        "description": "Asset coverage ratio (1940 Act style). 300% coverage = can borrow 33.3%",
        "convert": lambda x: 100 / (x / 100) if x > 0 else None,  # 300% -> 33.33%
    },
    "debt_to_equity": {
        "patterns": [
            r"debt[\s-]*to[\s-]*equity\s*(?:ratio\s*)?(?:of\s*)?(\d+(?:\.\d+)?)\s*%?",
            r"(\d+(?:\.\d+)?)\s*%?\s*debt[\s-]*to[\s-]*equity",
            r"borrow\s*(?:up\s*to\s*)?(\d+(?:\.\d+)?)\s*%?\s*of\s*(?:its\s*)?equity",
        ],
        "description": "Debt-to-equity ratio. 50% D/E = can borrow 33.3% of total assets",
        "convert": lambda x: (x / (100 + x)) * 100,  # 50% D/E -> 33.33% of assets
    },
    "pct_of_assets": {
        "patterns": [
            r"borrow\s*(?:up\s*to\s*)?(\d+(?:\.\d+)?)\s*%?\s*of\s*(?:total\s*)?assets",
            r"(\d+(?:\.\d+)?)\s*%\s*of\s*(?:total\s*)?assets",
            r"leverage\s*(?:of|up\s*to)\s*(\d+(?:\.\d+)?)\s*%",
            r"borrow\s*(?:up\s*to\s*)?(\d+(?:\.\d+)?)\s*%",
        ],
        "description": "Direct percentage of total assets",
        "convert": lambda x: x,  # Already in correct format
    },
    "pct_of_nav": {
        "patterns": [
            r"(\d+(?:\.\d+)?)\s*%\s*of\s*(?:net\s*)?(?:asset\s*)?value",
            r"(\d+(?:\.\d+)?)\s*%\s*of\s*NAV",
            r"borrow.*?(\d+(?:\.\d+)?)\s*%.*?net\s*assets",
        ],
        "description": "Percentage of NAV (approximately equal to % of assets for leverage)",
        "convert": lambda x: x,  # Treat as equivalent to % of assets
    },
    "one_third_language": {
        "patterns": [
            r"borrow\s*(?:up\s*to\s*)?(?:one[\s-]*third|1/3)",
            r"(?:one[\s-]*third|1/3)\s*of\s*(?:its\s*)?(?:total\s*)?assets",
        ],
        "description": "One-third of assets language (standard 1940 Act limit)",
        "convert": lambda x: 33.33,  # Fixed value
        "fixed_value": True,
    },
    "act_1940_reference": {
        "patterns": [
            r"1940\s*Act\s*(?:limit|requirement)",
            r"Investment\s*Company\s*Act\s*of\s*1940",
            r"statutory\s*(?:leverage\s*)?limit",
        ],
        "description": "Reference to 1940 Act standard (implies 33.33% limit)",
        "convert": lambda x: 33.33,  # Fixed value
        "fixed_value": True,
    },
}


def detect_leverage_format(text: str) -> Tuple[Optional[str], Optional[float], float]:
    """
    Detect the leverage format used in the text and extract the raw value.

    Args:
        text: Text containing leverage information

    Returns:
        Tuple of (format_type, raw_value, confidence)
    """
    text_lower = text.lower()

    for format_type, config in LEVERAGE_FORMATS.items():
        for pattern in config["patterns"]:
            match = re.search(pattern, text_lower)
            if match:
                if config.get("fixed_value"):
                    return format_type, None, 0.9
                try:
                    raw_value = float(match.group(1))
                    return format_type, raw_value, 0.95
                except (ValueError, IndexError):
                    continue

    return None, None, 0.0


def normalize_leverage(
    raw_value: Any,
    source_quote: str = "",
) -> LeverageNormalization:
    """
    Normalize leverage to percentage of total assets.

    Args:
        raw_value: Extracted value (could be "300%", "33%", "50% D/E", etc.)
        source_quote: Original text from document

    Returns:
        LeverageNormalization with standardized value
    """
    # Handle None/empty
    if raw_value is None or raw_value == "":
        return LeverageNormalization(
            raw_value="",
            raw_format="unknown",
            normalized_pct=None,
            source_quote=source_quote,
            confidence=0.0,
            notes="No value provided",
        )

    raw_str = str(raw_value).strip()

    # Try to detect format from source quote first (more context)
    if source_quote:
        format_type, detected_value, confidence = detect_leverage_format(source_quote)
        if format_type:
            config = LEVERAGE_FORMATS[format_type]

            if config.get("fixed_value"):
                normalized = config["convert"](None)
            else:
                # Use detected value if available, otherwise parse raw_value
                value_to_convert = detected_value
                if value_to_convert is None:
                    try:
                        value_to_convert = float(raw_str.replace("%", "").strip())
                    except ValueError:
                        value_to_convert = None

                if value_to_convert is not None:
                    normalized = config["convert"](value_to_convert)
                else:
                    normalized = None

            return LeverageNormalization(
                raw_value=raw_str,
                raw_format=format_type,
                normalized_pct=round(normalized, 2) if normalized else None,
                source_quote=source_quote,
                confidence=confidence,
                notes=config["description"],
            )

    # Fall back to analyzing raw value alone
    try:
        numeric_value = float(raw_str.replace("%", "").strip())
    except ValueError:
        return LeverageNormalization(
            raw_value=raw_str,
            raw_format="unknown",
            normalized_pct=None,
            source_quote=source_quote,
            confidence=0.0,
            notes="Could not parse numeric value",
        )

    # Heuristic: if value > 100, likely asset coverage ratio
    if numeric_value > 100:
        normalized = 100 / (numeric_value / 100)
        return LeverageNormalization(
            raw_value=raw_str,
            raw_format="asset_coverage",
            normalized_pct=round(normalized, 2),
            source_quote=source_quote,
            confidence=0.7,
            notes=f"Assumed asset coverage ratio ({numeric_value}% coverage = {normalized:.1f}% leverage)",
        )

    # If value <= 100, assume direct percentage
    return LeverageNormalization(
        raw_value=raw_str,
        raw_format="pct_of_assets",
        normalized_pct=round(numeric_value, 2),
        source_quote=source_quote,
        confidence=0.8,
        notes="Assumed direct percentage of assets",
    )


# =============================================================================
# SEMANTIC FIELD TAXONOMY
# =============================================================================

# Incentive fee structure concepts - helps LLMs distinguish related but different mechanisms

INCENTIVE_FEE_TAXONOMY = """
INCENTIVE FEE STRUCTURE TAXONOMY
================================

This taxonomy defines the key concepts in private fund incentive fee structures.
Use this to correctly identify and distinguish related but different mechanisms.

HIGH WATER MARK MECHANISMS
--------------------------
A high water mark prevents paying incentive fees on gains that merely recover prior losses.
The following ALL indicate high_water_mark = True:

1. EXPLICIT "high water mark" or "HWM" language
   Example: "subject to a high water mark provision"

2. LOSS RECOVERY ACCOUNT / LOSS CARRYFORWARD ACCOUNT
   Example: "Incentive Fee equal to 10% of net profits OVER the then balance of the Loss Recovery Account"
   This IS a high water mark because:
   - The account tracks cumulative losses
   - No fee is paid until losses are recovered
   - Functionally identical to HWM

3. CUMULATIVE LOSS provisions
   Example: "no incentive fee shall be payable until all prior losses are recovered"

4. NAV RESTORATION requirements
   Example: "fees payable only after NAV exceeds prior high"

CATCH-UP MECHANISMS
-------------------
A catch-up provision allows the manager to receive 100% of profits between the hurdle rate
and a ceiling rate, to "catch up" to their full incentive percentage.

Indicates has_catch_up = True:

1. EXPLICIT "catch-up" or "catch up" language
   Example: "with a full catch-up to the Adviser"

2. 100% ALLOCATION between hurdle and ceiling
   Example: "100% of returns between 1.50% and 1.667% per quarter"

3. CEILING/CAP percentage above hurdle
   Example: "until Pre-Incentive Fee Returns equal 1.667%"

CRITICAL DISTINCTIONS
---------------------
| Mechanism              | Purpose                           | Key Language               |
|------------------------|-----------------------------------|----------------------------|
| High Water Mark        | Protect LP from paying on         | "high water mark",         |
|                        | recovered losses                  | "Loss Recovery Account",   |
|                        |                                   | "Loss Carryforward"        |
| Catch-up               | Allow GP to earn full fee %       | "catch-up", "100% between",|
|                        | after passing hurdle              | "ceiling of X%"            |
| Hurdle Rate            | Minimum return before any fee     | "hurdle rate", "preferred  |
|                        |                                   | return", "X% threshold"    |

A fund can have:
- HIGH WATER MARK + CATCH-UP (both mechanisms)
- HIGH WATER MARK only (no catch-up)
- CATCH-UP only (no HWM)
- NEITHER mechanism

COMMON FUND PATTERNS
--------------------
1. CREDIT/DEBT FUNDS (e.g., Blue Owl, Blackstone Credit):
   - Typically: hurdle rate + catch-up, often NO high water mark
   - Fee based on: "Pre-Incentive Fee Net Investment Income"
   - Example: "10% of income above 6% hurdle with full catch-up"

2. FUND-OF-FUNDS (e.g., StepStone, Hamilton Lane):
   - Often: Loss Recovery Account (= HWM), no catch-up
   - Fee based on: "net profits" over Loss Recovery Account
   - Example: "10% of net profits over Loss Recovery Account balance"

3. PE/BUYOUT STYLE:
   - Typically: 20% carry, 8% preferred return
   - May have both HWM and catch-up
   - Fee based on: "total return" or "profits"

NULL VALUES
-----------
- If has_incentive_fee = False, then high_water_mark and has_catch_up should be null
- If no explicit language for a mechanism is found, return null (not False)
- Don't infer from fund type - extract what the document says
"""


LEVERAGE_TAXONOMY = """
LEVERAGE FORMAT TAXONOMY
========================

This taxonomy defines how leverage limits are expressed in SEC filings.
Different documents use different formats that must be normalized.

FORMAT TYPES
------------

1. ASSET COVERAGE RATIO (1940 Act style)
   Document says: "300% asset coverage requirement"
   Meaning: For every $1 borrowed, must have $3 in assets
   Normalized: 33.33% of total assets can be borrowed

   CONVERSION: leverage_pct = 100 / (coverage_pct / 100)
   - 300% coverage → 33.33% leverage
   - 200% coverage → 50% leverage
   - 150% coverage → 66.67% leverage

2. DEBT-TO-EQUITY RATIO
   Document says: "50% debt-to-equity ratio" or "0.5x D/E"
   Meaning: Debt can be 50% of equity (not total assets)
   Normalized: 33.33% of total assets

   CONVERSION: leverage_pct = (de_ratio / (100 + de_ratio)) * 100
   - 50% D/E → 33.33% of assets
   - 100% D/E → 50% of assets
   - 33% D/E → 25% of assets

3. PERCENTAGE OF TOTAL ASSETS (direct)
   Document says: "may borrow up to 33% of total assets"
   Normalized: 33% (no conversion needed)

4. PERCENTAGE OF NAV
   Document says: "leverage may not exceed 33% of NAV"
   Normalized: ~33% of total assets (approximately equivalent)

5. 1940 ACT REFERENCE
   Document says: "in accordance with the 1940 Act" or "statutory limits"
   Normalized: 33.33% (the standard 1940 Act limit)

EXTRACTION INSTRUCTIONS
-----------------------
When extracting max_leverage_pct:

1. IDENTIFY the format type from the language used
2. EXTRACT the raw percentage value
3. CONVERT to "percentage of total assets that can be borrowed"
4. RETURN the normalized value

Example extractions:
| Document Text                              | Raw Value | Format          | Normalized |
|--------------------------------------------|-----------|-----------------|------------|
| "300% asset coverage requirement"          | 300%      | asset_coverage  | 33.33%     |
| "asset coverage of at least 300%"          | 300%      | asset_coverage  | 33.33%     |
| "50% debt-to-equity ratio"                 | 50%       | debt_to_equity  | 33.33%     |
| "may borrow up to 33% of assets"           | 33%       | pct_of_assets   | 33%        |
| "one-third of total assets"                | 33.33%    | one_third       | 33.33%     |
| "subject to 1940 Act borrowing limits"     | 33.33%    | act_1940        | 33.33%     |

COMMON PATTERNS BY FUND TYPE
----------------------------
- Most registered funds: "300% asset coverage" → 33.33%
- Some credit funds: "50% debt-to-equity" → 33.33%
- Direct language: "may borrow up to X%" → X%

IMPORTANT: The final answer should ALWAYS be the normalized "percentage of total assets"
value (typically 33%, 50%, etc.), NOT the raw coverage ratio (300%, 200%, etc.).
"""


def get_field_taxonomy(field_name: str) -> Optional[str]:
    """
    Get the relevant taxonomy for a field.

    Args:
        field_name: Name of the field being extracted

    Returns:
        Taxonomy text if available, None otherwise
    """
    taxonomy_map = {
        "high_water_mark": INCENTIVE_FEE_TAXONOMY,
        "has_catch_up": INCENTIVE_FEE_TAXONOMY,
        "hurdle_rate_pct": INCENTIVE_FEE_TAXONOMY,
        "incentive_fee_pct": INCENTIVE_FEE_TAXONOMY,
        "has_incentive_fee": INCENTIVE_FEE_TAXONOMY,
        "max_leverage_pct": LEVERAGE_TAXONOMY,
        "uses_leverage": LEVERAGE_TAXONOMY,
    }
    return taxonomy_map.get(field_name)


# =============================================================================
# ENHANCED FIELD SPECS WITH TAXONOMY
# =============================================================================

# These are extraction hints enhanced with taxonomy understanding

ENHANCED_EXTRACTION_HINTS = {
    "high_water_mark": [
        "SEARCH: 'high water mark', 'loss recovery account', 'loss carryforward', 'cumulative loss'",
        "",
        "DECISION TREE:",
        "1. Found 'high water mark' or 'HWM' → TRUE",
        "2. Found 'Loss Recovery Account' or 'Loss Carryforward Account' → TRUE",
        "   (These track cumulative losses and prevent fees on recovered losses = HWM)",
        "3. Found 'cumulative loss' provision in fee context → TRUE",
        "4. Found ONLY 'catch-up' without above → Check further; may be FALSE",
        "5. No incentive fee at all (has_incentive_fee=False) → NULL",
        "6. Incentive fee exists but no HWM language found → NULL (not FALSE)",
        "",
        "CRITICAL: 'Loss Recovery Account' IS a high water mark mechanism.",
        "It prevents paying incentive fees on gains that merely recover prior losses.",
    ],

    "has_catch_up": [
        "SEARCH: 'catch-up', 'catch up', '100% of returns between', 'ceiling'",
        "",
        "DECISION TREE:",
        "1. Found 'catch-up' or 'catch up' in fee context → TRUE",
        "2. Found '100% of returns between X% and Y%' → TRUE",
        "3. Found ceiling percentage above hurdle → TRUE",
        "4. Found Loss Recovery Account WITHOUT catch-up language → NULL (not FALSE)",
        "5. No incentive fee at all (has_incentive_fee=False) → NULL",
        "6. Incentive fee exists but no catch-up language → NULL (not FALSE)",
        "",
        "CRITICAL: Loss Recovery Account is NOT catch-up. It's high water mark.",
        "Catch-up is about accelerated fee payment, not loss recovery.",
    ],

    "max_leverage_pct": [
        "SEARCH: 'leverage', 'borrowing', 'asset coverage', 'debt-to-equity', '1940 Act'",
        "",
        "FORMAT RECOGNITION (CRITICAL):",
        "1. '300% asset coverage' → Normalize to 33.33%",
        "   Formula: 100 / (coverage% / 100) = leverage%",
        "   300% coverage = can borrow 1/3 of assets = 33.33%",
        "",
        "2. '50% debt-to-equity' → Normalize to 33.33%",
        "   Formula: (D/E) / (1 + D/E) = leverage%",
        "   50% D/E = 0.5 / 1.5 = 33.33%",
        "",
        "3. 'may borrow up to 33% of assets' → 33% (already normalized)",
        "",
        "4. '1940 Act limits' or 'statutory limits' → 33.33% (standard limit)",
        "",
        "CRITICAL: Return the NORMALIZED percentage (33%, 50%), NOT the raw coverage (300%, 200%)",
        "Most registered funds are limited to 33.33% leverage (300% asset coverage).",
    ],

    "hurdle_rate_pct": [
        "SEARCH: 'hurdle rate', 'preferred return', 'threshold', 'minimum return'",
        "",
        "ANNUALIZATION:",
        "- If stated as quarterly rate (e.g., '1.25% per quarter'), multiply by 4",
        "  Example: 1.25% quarterly = 5% annualized",
        "- If stated as monthly rate, multiply by 12",
        "",
        "IMPORTANT: Hurdle rate is different from catch-up ceiling",
        "- Hurdle = minimum return before ANY fee is paid",
        "- Catch-up ceiling = rate at which manager has 'caught up' to full fee %",
        "",
        "If fund has NO incentive fee (has_incentive_fee=False), return NULL",
        "If incentive fee exists but no hurdle mentioned, the fund may have 0% hurdle - search carefully",
    ],
}


def get_enhanced_hints(field_name: str) -> list[str]:
    """
    Get enhanced extraction hints for a field.

    Args:
        field_name: Name of the field

    Returns:
        List of enhanced extraction hints
    """
    return ENHANCED_EXTRACTION_HINTS.get(field_name, [])
