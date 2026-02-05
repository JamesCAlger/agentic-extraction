"""
Section finder for extracting content from untagged document sections.

This module provides tiered fallback extraction when the document segmenter
doesn't identify key sections. Uses three tiers:

Tier 1: Exact section heading match (handled by document segmenter)
Tier 2: Regex pattern search with keyword scoring (this module)
Tier 3: Broad keyword search with context expansion

Each tier progressively relaxes matching requirements to find content
that may be embedded in non-standard document structures.
"""

import re
import logging
from typing import Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# Tier 2: Patterns to find key sections and their content boundaries
SECTION_PATTERNS = {
    "repurchase_terms": {
        "start_patterns": [
            r"REPURCHASES?\s+OF\s+SHARES",
            r"PERIODIC\s+REPURCHASE\s+OFFERS?",
            r"TENDER\s+OFFERS?\s+BY\s+THE\s+FUND",
            r"SHARE\s+REPURCHASE\s+PROGRAM",
            r"REPURCHASE\s+OF\s+COMMON\s+SHARES",
        ],
        "end_patterns": [
            r"DISTRIBUTION\s+POLICY",
            r"TAX\s+MATTERS",
            r"DIVIDEND\s+REINVESTMENT",
            r"CERTAIN\s+U\.?S\.?\s+FEDERAL",
            r"CUSTODIAN",
        ],
        "max_chars": 15000,
    },
    "share_classes": {
        "start_patterns": [
            r"DESCRIPTION\s+OF\s+(?:THE\s+)?SHARES",
            r"DESCRIPTION\s+OF\s+COMMON\s+SHARES",
            r"TYPES\s+OF\s+SHARES",
            r"CLASSES\s+OF\s+SHARES",
            r"PLAN\s+OF\s+DISTRIBUTION",
            r"PURCHASING\s+(?:COMMON\s+)?SHARES",
            r"HOW\s+TO\s+(?:BUY|PURCHASE)\s+SHARES",
            r"SHARE\s+CLASSES",
            r"CLASS\s+[SDITA]\s+SHARES",
        ],
        "end_patterns": [
            r"DIVIDEND\s+REINVESTMENT",
            r"DISTRIBUTION\s+POLICY",
            r"TAX\s+MATTERS",
            r"REPURCHASES?\s+OF\s+SHARES",
            r"CERTAIN\s+U\.?S\.?\s+FEDERAL",
        ],
        "max_chars": 25000,
    },
    "minimum_investment": {
        "start_patterns": [
            r"MINIMUM\s+(?:INITIAL\s+)?INVESTMENT",
            r"PURCHASING\s+(?:COMMON\s+)?SHARES",
            r"HOW\s+TO\s+(?:BUY|PURCHASE)",
            r"PURCHASE\s+TERMS",
        ],
        "end_patterns": [
            r"REPURCHASES?\s+OF\s+SHARES",
            r"DISTRIBUTION\s+POLICY",
        ],
        "max_chars": 15000,
    },
    "concentration_limits": {
        "start_patterns": [
            r"INVESTMENT\s+RESTRICTIONS",
            r"FUNDAMENTAL\s+(?:INVESTMENT\s+)?POLICIES",
            r"NON-FUNDAMENTAL\s+(?:INVESTMENT\s+)?POLICIES",
            r"INVESTMENT\s+LIMITATIONS",
            r"CONCENTRATION\s+LIMITS",
        ],
        "end_patterns": [
            r"MANAGEMENT\s+OF\s+THE\s+FUND",
            r"BOARD\s+OF\s+(?:DIRECTORS|TRUSTEES)",
            r"TAX\s+MATTERS",
            r"CERTAIN\s+U\.?S\.?\s+FEDERAL",
        ],
        "max_chars": 20000,
    },
    "expense_cap": {
        "start_patterns": [
            r"EXPENSE\s+(?:CAP|LIMIT(?:ATION)?)",
            r"FEE\s+WAIVER",
            r"EXPENSE\s+REIMBURSEMENT",
            r"CONTRACTUAL\s+FEE\s+WAIVER",
            r"ADVISER\s+HAS\s+(?:CONTRACTUALLY\s+)?AGREED\s+TO\s+WAIVE",
        ],
        "end_patterns": [
            r"BOARD\s+OF\s+(?:DIRECTORS|TRUSTEES)",
            r"MANAGEMENT\s+OF\s+THE\s+FUND",
            r"INVESTMENT\s+ADVISORY",
        ],
        "max_chars": 15000,
    },
    # NEW v2.0 fields
    "leverage_limits": {
        "start_patterns": [
            r"USE\s+OF\s+LEVERAGE",
            r"LEVERAGE\s+AND\s+BORROWING",
            r"BORROWING",
            r"LEVERAGE\s+RISK",
            r"WILL\s+YOU\s+USE\s+LEVERAGE",
            r"CREDIT\s+FACILITY",
            r"LEVERAGE",
        ],
        "end_patterns": [
            r"RISK\s+FACTORS",
            r"INVESTMENT\s+RESTRICTIONS",
            r"MANAGEMENT\s+OF\s+THE\s+FUND",
            r"TAX\s+MATTERS",
        ],
        "max_chars": 15000,
    },
    "distribution_terms": {
        "start_patterns": [
            r"DISTRIBUTION\s+POLICY",
            r"DIVIDEND\s+POLICY",
            r"DISTRIBUTIONS?\s+AND\s+DIVIDENDS?",
            r"DIVIDEND\s+REINVESTMENT",
            r"DISTRIBUTIONS?\s+TO\s+SHAREHOLDERS",
        ],
        "end_patterns": [
            r"TAX\s+MATTERS",
            r"CERTAIN\s+U\.?S\.?\s+FEDERAL",
            r"REPURCHASES?\s+OF\s+SHARES",
            r"CUSTODIAN",
        ],
        "max_chars": 12000,
    },
    # Additional Tier 2 patterns for complete coverage
    "fund_metadata": {
        "start_patterns": [
            r"PROSPECTUS\s+SUMMARY",
            r"SUMMARY\s+OF\s+THE\s+FUND",
            r"THE\s+FUND",
            r"ABOUT\s+THE\s+FUND",
            r"FUND\s+SUMMARY",
        ],
        "end_patterns": [
            r"SUMMARY\s+OF\s+FUND\s+EXPENSES",
            r"FEES\s+AND\s+EXPENSES",
            r"RISK\s+FACTORS",
            r"THE\s+OFFERING",
        ],
        "max_chars": 15000,
    },
    "incentive_fee": {
        "start_patterns": [
            r"INCENTIVE\s+FEE",
            r"PERFORMANCE\s+FEE",
            r"CARRIED\s+INTEREST",
            r"PERFORMANCE\s+ALLOCATION",
            r"MANAGEMENT\s+FEE.*INCENTIVE",
        ],
        "end_patterns": [
            r"EXPENSE\s+LIMITATION",
            r"EXPENSE\s+CAP",
            r"BOARD\s+OF\s+(?:DIRECTORS|TRUSTEES)",
            r"CUSTODIAN",
        ],
        "max_chars": 12000,
    },
    "allocation_targets": {
        "start_patterns": [
            r"INVESTMENT\s+OBJECTIVE",
            r"INVESTMENT\s+STRATEG(?:Y|IES)",
            r"ASSET\s+ALLOCATION",
            r"INVESTMENT\s+PROGRAM",
            r"PRINCIPAL\s+INVESTMENT\s+STRATEG",
        ],
        "end_patterns": [
            r"RISK\s+FACTORS",
            r"FEES\s+AND\s+EXPENSES",
            r"INVESTMENT\s+RESTRICTIONS",
            r"MANAGEMENT\s+OF\s+THE\s+FUND",
        ],
        "max_chars": 20000,
    },
}

# Tier 3: Broad keyword search - used when Tier 2 fails
TIER3_KEYWORDS = {
    "repurchase_terms": [
        "repurchase", "tender offer", "interval fund", "5%", "25%",
        "quarterly repurchase", "NAV", "repurchase request",
        "repurchase offer", "fundamental policy", "Rule 23c-3",
    ],
    "share_classes": [
        "class s", "class d", "class i", "sales load", "sales charge",
        "distribution fee", "minimum investment", "share class",
        "0.85%", "0.25%", "3.5%", "$5,000", "$1,000,000",
    ],
    "concentration_limits": [
        "25%", "not invest more than", "concentration", "diversification",
        "single issuer", "single fund", "industry", "sector limit",
        "investment restriction", "fundamental policy",
    ],
    "expense_cap": [
        "waive", "waiver", "expense cap", "expense limit", "reimbursement",
        "contractually agreed", "fee waiver", "operating expenses",
        "recoupment", "expense ratio",
    ],
    # NEW v2.0 keywords
    "leverage_limits": [
        "leverage", "borrowing", "borrow", "credit facility", "debt",
        "debt-to-equity", "total assets", "net assets", "asset coverage",
        "33-1/3%", "50%", "bank loan", "line of credit", "300%",
    ],
    "distribution_terms": [
        "distribution", "dividend", "reinvestment", "DRIP", "quarterly",
        "monthly", "annual", "cash", "reinvested", "dividend policy",
        "distribution rate", "income", "capital gains",
    ],
    # Additional Tier 3 keywords for complete coverage
    "fund_metadata": [
        "fund name", "investment manager", "adviser", "advisor", "sponsor",
        "fiscal year", "december 31", "march 31", "interval fund",
        "tender offer", "closed-end", "registered",
    ],
    "incentive_fee": [
        "incentive fee", "performance fee", "carried interest", "hurdle rate",
        "high water mark", "performance allocation", "20%", "15%",
        "profits", "net profits", "preferred return",
    ],
    "allocation_targets": [
        "target allocation", "asset allocation", "private equity", "private credit",
        "real estate", "real assets", "infrastructure", "secondaries",
        "co-investment", "primary", "diversified",
    ],
}


class SectionFinder:
    """
    Finds and extracts content from document sections by text pattern.

    Implements tiered fallback search:
    - Tier 2: Regex pattern search for section headings
    - Tier 3: Keyword-based search with context expansion
    """

    def __init__(self, html_content: str):
        """
        Initialize with document HTML.

        Args:
            html_content: Raw HTML content of the document
        """
        soup = BeautifulSoup(html_content, "html.parser")
        self.full_text = soup.get_text(separator="\n")
        self.html_content = html_content

    def find_section(self, field_name: str, tier: int = 2) -> Optional[str]:
        """
        Find and extract a section by field name.

        Args:
            field_name: Name of the field (e.g., "repurchase_terms")
            tier: Which tier to use (2=pattern, 3=keyword)

        Returns:
            Extracted section text or None if not found
        """
        if tier == 2:
            return self._find_section_tier2(field_name)
        elif tier == 3:
            return self._find_section_tier3(field_name)
        return None

    def _find_section_tier2(self, field_name: str) -> Optional[str]:
        """Tier 2: Pattern-based section search."""
        config = SECTION_PATTERNS.get(field_name)
        if not config:
            return None

        # Find section start
        start_pos = None
        for pattern in config["start_patterns"]:
            match = re.search(pattern, self.full_text, re.IGNORECASE | re.MULTILINE)
            if match:
                start_pos = match.start()
                break

        if start_pos is None:
            return None

        # Find section end
        end_pos = start_pos + config["max_chars"]
        search_text = self.full_text[start_pos:end_pos]

        for pattern in config["end_patterns"]:
            # Search for end pattern after some minimum content
            match = re.search(pattern, search_text[500:], re.IGNORECASE | re.MULTILINE)
            if match:
                end_pos = start_pos + 500 + match.start()
                break

        # Extract and clean section text
        section_text = self.full_text[start_pos:end_pos]
        section_text = self._clean_text(section_text)

        return section_text

    def _find_section_tier3(self, field_name: str) -> Optional[str]:
        """
        Tier 3: Keyword-based search with context expansion.

        Finds paragraphs/sections containing multiple relevant keywords
        and extracts surrounding context.
        """
        keywords = TIER3_KEYWORDS.get(field_name, [])
        if not keywords:
            return None

        text_lower = self.full_text.lower()

        # Find all keyword matches and score paragraphs
        paragraphs = self.full_text.split("\n\n")
        scored_paragraphs = []

        for i, para in enumerate(paragraphs):
            para_lower = para.lower()
            score = sum(1 for kw in keywords if kw.lower() in para_lower)
            if score > 0:
                scored_paragraphs.append((score, i, para))

        if not scored_paragraphs:
            return None

        # Sort by score descending
        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)

        # Extract top-scoring paragraphs with context
        best_indices = set()
        for score, idx, _ in scored_paragraphs[:5]:  # Top 5
            # Include surrounding paragraphs for context
            for j in range(max(0, idx - 1), min(len(paragraphs), idx + 3)):
                best_indices.add(j)

        # Build continuous text blocks
        sorted_indices = sorted(best_indices)
        extracted_parts = []
        current_block = []

        for i, idx in enumerate(sorted_indices):
            if i == 0 or idx == sorted_indices[i - 1] + 1:
                current_block.append(paragraphs[idx])
            else:
                if current_block:
                    extracted_parts.append("\n\n".join(current_block))
                current_block = [paragraphs[idx]]

        if current_block:
            extracted_parts.append("\n\n".join(current_block))

        # Combine all blocks
        result = "\n\n---\n\n".join(extracted_parts)

        # Limit size
        max_chars = 15000
        if len(result) > max_chars:
            result = result[:max_chars]

        return self._clean_text(result) if result else None

    def find_section_tiered(self, field_name: str) -> Optional[str]:
        """
        Find section using tiered approach (Tier 2, then Tier 3).

        Returns:
            Extracted section text or None
        """
        # Try Tier 2 first
        result = self._find_section_tier2(field_name)
        if result:
            logger.debug(f"Found {field_name} via Tier 2 (pattern search)")
            return result

        # Fall back to Tier 3
        result = self._find_section_tier3(field_name)
        if result:
            logger.debug(f"Found {field_name} via Tier 3 (keyword search)")
            return result

        return None

    def find_all_sections(self) -> dict[str, Optional[str]]:
        """
        Find all configured sections using tiered approach.

        Returns:
            Dict mapping field names to extracted text (or None)
        """
        return {
            field_name: self.find_section_tiered(field_name)
            for field_name in SECTION_PATTERNS
        }

    def _clean_text(self, text: str) -> str:
        """Clean up extracted text."""
        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()


def extract_missing_sections(
    html_content: str,
    missing_fields: list[str],
    use_tiered: bool = True,
) -> dict[str, str]:
    """
    Extract sections for fields that weren't found by the segmenter.

    Uses tiered fallback search:
    - Tier 2: Pattern-based section headings
    - Tier 3: Keyword-based search with context

    Args:
        html_content: Raw HTML document
        missing_fields: List of field names to search for
        use_tiered: Whether to use tiered search (Tier 2 -> Tier 3)

    Returns:
        Dict mapping field names to extracted text
    """
    finder = SectionFinder(html_content)
    results = {}

    for field in missing_fields:
        if use_tiered:
            text = finder.find_section_tiered(field)
        else:
            text = finder.find_section(field, tier=2)

        if text:
            results[field] = text
            logger.info(f"    Fallback found {field} ({len(text)} chars)")
        else:
            # Try Tier 3 if not already using tiered and Tier 2 failed
            if not use_tiered:
                text = finder.find_section(field, tier=3)
                if text:
                    results[field] = text
                    logger.info(f"    Tier 3 found {field} ({len(text)} chars)")

    return results
