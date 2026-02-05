"""
iXBRL Parser for SEC filings.

Extracts structured XBRL data from Inline XBRL (iXBRL) documents,
including numeric values, text blocks, and context information.

Supports observability logging for full extraction audit trail.
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Optional
from bs4 import BeautifulSoup, Tag
import tiktoken

from enum import Enum
from .models import (
    XBRLContext,
    XBRLNumericValue,
    XBRLTextBlock,
)

logger = logging.getLogger(__name__)


class FundType(str, Enum):
    """Fund structure classification based on SEC filing checkboxes."""
    INTERVAL_FUND = "interval_fund"
    TENDER_OFFER_FUND = "tender_offer_fund"
    BDC = "bdc"  # Business Development Company
    REGISTERED_CEF = "registered_cef"  # Other registered closed-end fund
    OTHER = "other"


class IXBRLParser:
    """Parser for Inline XBRL documents."""

    # XBRL namespaces we care about
    NAMESPACES = {
        "ix": "http://www.xbrl.org/2013/inlineXBRL",
        "cef": "http://xbrl.sec.gov/cef/",
        "dei": "http://xbrl.sec.gov/dei/",
    }

    # Tag prefixes for different data types
    NUMERIC_TAGS = ["ix:nonFraction", "ix:fraction"]
    TEXT_TAGS = ["ix:nonNumeric"]

    def __init__(self, estimate_tokens: bool = True):
        """
        Initialize parser.

        Args:
            estimate_tokens: Whether to estimate token counts (requires tiktoken)
        """
        self.estimate_tokens = estimate_tokens
        self._tokenizer = None
        if estimate_tokens:
            try:
                self._tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
            except Exception:
                self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        # Rough estimate: 1 token per 4 characters
        return len(text) // 4

    def parse(self, html_content: str) -> tuple[list[XBRLNumericValue], list[XBRLTextBlock], dict[str, XBRLContext]]:
        """
        Parse iXBRL document and extract all XBRL data.

        Args:
            html_content: Raw HTML content of the iXBRL document

        Returns:
            Tuple of (numeric_values, text_blocks, contexts)
        """
        # Use html.parser for better namespace handling, or lxml-xml for XML
        soup = BeautifulSoup(html_content, "html.parser")

        # First, extract all contexts
        contexts = self._extract_contexts(soup)

        # Extract numeric values
        numeric_values = self._extract_numeric_values(soup, contexts)

        # Extract text blocks
        text_blocks = self._extract_text_blocks(soup, contexts)

        return numeric_values, text_blocks, contexts

    def _extract_contexts(self, soup: BeautifulSoup) -> dict[str, XBRLContext]:
        """Extract XBRL context definitions."""
        contexts = {}

        # Find all context elements
        for ctx in soup.find_all(re.compile(r"xbrli?:context", re.IGNORECASE)):
            ctx_id = ctx.get("id", "")
            if not ctx_id:
                continue

            # Parse period
            period_start = None
            period_end = None
            instant = None

            period = ctx.find(re.compile(r"xbrli?:period", re.IGNORECASE))
            if period:
                start_elem = period.find(re.compile(r"xbrli?:startDate", re.IGNORECASE))
                end_elem = period.find(re.compile(r"xbrli?:endDate", re.IGNORECASE))
                instant_elem = period.find(re.compile(r"xbrli?:instant", re.IGNORECASE))

                if start_elem:
                    period_start = start_elem.get_text(strip=True)
                if end_elem:
                    period_end = end_elem.get_text(strip=True)
                if instant_elem:
                    instant = instant_elem.get_text(strip=True)

            # Parse share class from context ID or explicit member
            share_class = self._extract_share_class(ctx_id, ctx)

            contexts[ctx_id] = XBRLContext(
                context_id=ctx_id,
                share_class=share_class,
                period_start=period_start,
                period_end=period_end,
                instant=instant,
            )

        return contexts

    # =========================================================================
    # Share Class Pattern Configuration
    # =========================================================================

    # Patterns for extracting share class from contextRef or explicitMember
    # Ordered by specificity (most specific first)
    SHARE_CLASS_PATTERNS = [
        # Compound class names (must be before simple letter patterns)
        (r"Class([A-Z])Advisory(?:Shares?)?Member", lambda m: f"Class {m.group(1)} Advisory"),
        (r"Class([A-Z])Institutional(?:Shares?)?Member", lambda m: f"Class {m.group(1)} Institutional"),
        (r"Class([A-Z])Retail(?:Shares?)?Member", lambda m: f"Class {m.group(1)} Retail"),

        # Letter-based classes (Class I, Class S, Class D, etc.)
        (r"Class([A-Z])Shares?Member", lambda m: f"Class {m.group(1)}"),
        (r"Class([A-Z])CommonStock", lambda m: f"Class {m.group(1)}"),
        (r"Class_([A-Z])_", lambda m: f"Class {m.group(1)}"),
        (r"_Class([A-Z])_", lambda m: f"Class {m.group(1)}"),
        (r"_Class([A-Z])$", lambda m: f"Class {m.group(1)}"),

        # Named share classes
        (r"(Institutional)Shares?Member", lambda m: "Institutional"),
        (r"(Retail)Shares?Member", lambda m: "Retail"),
        (r"(Advisory)Shares?Member", lambda m: "Advisory"),
        (r"(Investor)Shares?Member", lambda m: "Investor"),

        # Share type indicators
        (r"(Common)Shares?Member", lambda m: "Common"),
        (r"(Preferred)Shares?Member", lambda m: "Preferred"),

        # Fund-specific patterns (Blackstone, etc.)
        (r"Class([A-Z])_?Advisory_?Member", lambda m: f"Class {m.group(1)} Advisory"),
        (r"(Premier|Select|Plus)Shares?Member", lambda m: m.group(1)),
    ]

    # Keywords that indicate this is NOT a share class (risk factors, etc.)
    NON_SHARE_CLASS_KEYWORDS = [
        "Risk", "Sector", "Investment", "Strategy", "Contact",
        "Repurchased",  # e.g., ClassISharesRepurchasedMember is not a share class
        "Concentration", "Tax", "Leverage", "Valuation",
        "Market", "Economic", "Geographic", "Currency",
    ]

    def _extract_share_class(self, context_id: str, context_elem: Tag) -> Optional[str]:
        """
        Extract share class from context ID or segment.

        Uses a comprehensive set of patterns to identify share classes from:
        1. contextRef attribute (e.g., "P08_19_2025_ClassISharesMember")
        2. xbrldi:explicitMember elements in the context segment

        Returns:
            Share class name (e.g., "Class I", "Class I Advisory", "Common")
            or None if no share class found
        """
        # Try extracting from context ID first
        share_class = self._match_share_class_patterns(context_id)
        if share_class:
            return share_class

        # Check explicit segment members
        segment = context_elem.find(re.compile(r"xbrli?:segment", re.IGNORECASE))
        if segment:
            for member in segment.find_all(re.compile(r"xbrldi:explicitMember", re.IGNORECASE)):
                member_text = member.get_text(strip=True)
                share_class = self._match_share_class_patterns(member_text)
                if share_class:
                    return share_class

        return None

    def _match_share_class_patterns(self, text: str) -> Optional[str]:
        """
        Match share class patterns against text.

        Args:
            text: Text to search (contextRef or explicitMember value)

        Returns:
            Share class name or None
        """
        # First check if this looks like a non-share-class member
        if self._is_non_share_class(text):
            return None

        # Try each pattern in order (most specific first)
        for pattern, formatter in self.SHARE_CLASS_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return formatter(match)

        return None

    def _is_non_share_class(self, text: str) -> bool:
        """
        Check if text contains keywords indicating it's not a share class.

        Filters out risk factors, sector concentrations, etc.
        """
        text_upper = text.upper()
        for keyword in self.NON_SHARE_CLASS_KEYWORDS:
            if keyword.upper() in text_upper:
                return True
        return False

    def _extract_numeric_values(
        self, soup: BeautifulSoup, contexts: dict[str, XBRLContext]
    ) -> list[XBRLNumericValue]:
        """Extract all numeric XBRL values."""
        values = []
        seen_ids = set()  # Track seen element IDs to avoid duplicates

        # Find ix:nonFraction elements (most common for numeric values)
        # BeautifulSoup normalizes tag names to lowercase
        for tag_name in ["ix:nonfraction", "ix:fraction"]:
            for elem in soup.find_all(tag_name):
                elem_id = elem.get("id", "")
                if elem_id and elem_id in seen_ids:
                    continue
                seen_ids.add(elem_id)

                value = self._parse_numeric_element(elem, contexts)
                if value:
                    values.append(value)

        return values

    def _parse_numeric_element(
        self, elem: Tag, contexts: dict[str, XBRLContext]
    ) -> Optional[XBRLNumericValue]:
        """Parse a single numeric XBRL element."""
        # Get tag name (e.g., "cef:ManagementFeesPercent")
        tag_name = elem.get("name", "")
        if not tag_name:
            return None

        # Get context (BeautifulSoup normalizes attribute names to lowercase)
        context_ref = elem.get("contextref", "")
        context = contexts.get(context_ref, XBRLContext(context_id=context_ref))

        # Get raw text value
        raw_text = elem.get_text(strip=True)

        # Parse numeric value
        try:
            # Handle special formats
            format_attr = elem.get("format", "")

            if "numwordsen" in format_attr.lower():
                # Text numbers like "None", "Zero"
                value = self._parse_text_number(raw_text)
            else:
                # Remove commas and parse
                clean_text = raw_text.replace(",", "").replace(" ", "")
                if clean_text.lower() in ["none", "-", "n/a", ""]:
                    value = Decimal("0")
                else:
                    value = Decimal(clean_text)

            # Apply scale
            scale = elem.get("scale")
            if scale:
                try:
                    scale_int = int(scale)
                    value = value * (Decimal("10") ** scale_int)
                except ValueError:
                    pass

        except (InvalidOperation, ValueError):
            return None

        return XBRLNumericValue(
            tag_name=tag_name,
            value=value,
            unit=elem.get("unitref"),
            decimals=self._safe_int(elem.get("decimals")),
            scale=self._safe_int(elem.get("scale")),
            context=context,
            element_id=elem.get("id"),
            raw_text=raw_text,
        )

    def _parse_text_number(self, text: str) -> Decimal:
        """Parse text representations of numbers."""
        text_lower = text.lower().strip()
        text_map = {
            "none": Decimal("0"),
            "zero": Decimal("0"),
            "one": Decimal("1"),
            "two": Decimal("2"),
            "three": Decimal("3"),
            "four": Decimal("4"),
            "five": Decimal("5"),
            "ten": Decimal("10"),
        }
        return text_map.get(text_lower, Decimal("0"))

    def _safe_int(self, value: Optional[str]) -> Optional[int]:
        """Safely convert string to int."""
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    def _extract_text_blocks(
        self, soup: BeautifulSoup, contexts: dict[str, XBRLContext]
    ) -> list[XBRLTextBlock]:
        """Extract all text block XBRL values."""
        blocks = []
        seen_ids = set()  # Track seen element IDs to avoid duplicates

        # Text blocks are ix:nonNumeric elements with TextBlock in the name
        # BeautifulSoup normalizes tag names to lowercase
        for elem in soup.find_all("ix:nonnumeric"):
            tag_name = elem.get("name", "")

            # Only include text blocks (not simple flags/values)
            if "TextBlock" not in tag_name and "Text" not in tag_name:
                continue

            elem_id = elem.get("id", "")
            if elem_id and elem_id in seen_ids:
                continue
            seen_ids.add(elem_id)

            block = self._parse_text_block(elem, contexts)
            if block:
                blocks.append(block)

        # Also extract fee-related footnotes as synthetic text blocks
        # These often contain incentive fee structures, expense caps, etc.
        footnote_blocks = self._extract_fee_footnotes(soup)
        blocks.extend(footnote_blocks)

        return blocks

    # Keywords that indicate a footnote contains fee structure information
    FEE_FOOTNOTE_KEYWORDS = [
        "incentive fee",
        "performance fee",
        "hurdle rate",
        "catch-up",
        "high water mark",
        "expense cap",
        "fee waiver",
        "pre-incentive fee",
        "net investment income",
    ]

    def _extract_fee_footnotes(self, soup: BeautifulSoup) -> list[XBRLTextBlock]:
        """
        Extract ix:footnote elements that contain fee structure information.

        Fee tables often reference footnotes for incentive fee details (e.g., "12.5%
        of Pre-Incentive Fee Net Investment Income"). These footnotes are defined
        in a hidden header section and not captured by normal text block extraction.

        Returns:
            List of synthetic XBRLTextBlock objects for fee-related footnotes
        """
        blocks = []

        # Find all ix:footnote elements
        for footnote in soup.find_all("ix:footnote"):
            footnote_id = footnote.get("id", "")
            content = footnote.get_text(separator=" ")
            content = re.sub(r"\s+", " ", content).strip()

            if not content:
                continue

            # Check if footnote contains fee-related keywords
            content_lower = content.lower()
            is_fee_related = any(
                keyword in content_lower for keyword in self.FEE_FOOTNOTE_KEYWORDS
            )

            if is_fee_related:
                # Create synthetic text block for this footnote
                block = XBRLTextBlock(
                    tag_name="ix:footnote:FeeStructure",  # Synthetic tag name
                    content=content,
                    content_html=str(footnote),
                    context=XBRLContext(context_id=footnote_id),
                    element_id=footnote_id,
                    char_count=len(content),
                    estimated_tokens=self._count_tokens(content),
                )
                blocks.append(block)

        return blocks

    def _parse_text_block(
        self, elem: Tag, contexts: dict[str, XBRLContext]
    ) -> Optional[XBRLTextBlock]:
        """Parse a single text block XBRL element."""
        tag_name = elem.get("name", "")
        if not tag_name:
            return None

        # Get context (BeautifulSoup normalizes attribute names to lowercase)
        context_ref = elem.get("contextref", "")
        context = contexts.get(context_ref, XBRLContext(context_id=context_ref))

        # Get HTML content
        content_html = str(elem)

        # Handle continued text blocks
        continued_at = elem.get("continuedat")
        if continued_at:
            # Find continuation element and append
            continuation = elem.find_parent().find(id=continued_at)
            if continuation:
                content_html += str(continuation)

        # Strip HTML to get plain text
        content = self._strip_html(content_html)

        if not content.strip():
            return None

        return XBRLTextBlock(
            tag_name=tag_name,
            content=content,
            content_html=content_html,
            context=context,
            element_id=elem.get("id"),
            char_count=len(content),
            estimated_tokens=self._count_tokens(content),
        )

    def _strip_html(self, html: str) -> str:
        """Strip HTML tags and clean up text."""
        soup = BeautifulSoup(html, "lxml")

        # Remove script and style elements
        for tag in soup(["script", "style"]):
            tag.decompose()

        # Get text
        text = soup.get_text(separator=" ")

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def extract_fund_type_flags(self, html_content: str) -> dict[str, bool]:
        """
        Extract fund type classification flags from iXBRL document.

        These are checkbox fields on the N-2 cover page that indicate
        whether the fund is an interval fund, BDC, etc.

        Args:
            html_content: Raw HTML content of the iXBRL document

        Returns:
            Dictionary of flag names to boolean values
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # Fund type flag tags to extract
        flag_tags = [
            "cef:RegisteredClosedEndFundFlag",
            "cef:BusinessDevelopmentCompanyFlag",
            "cef:IntervalFundFlag",
            "cef:PrimaryShelfFlag",
            "cef:PrimaryShelfQualifiedFlag",
            "cef:NewCefOrBdcRegistrantFlag",
        ]

        flags = {}

        for tag_name in flag_tags:
            # Search for ix:nonNumeric elements with this name
            # BeautifulSoup normalizes to lowercase
            for elem in soup.find_all("ix:nonnumeric"):
                if elem.get("name", "").lower() == tag_name.lower():
                    # Check if the ballot box is checked
                    content = elem.get_text()
                    # Unicode ballot box characters:
                    # ☒ (U+2612, &#9746;) = checked
                    # ☑ (U+2611, &#9745;) = checked
                    # ☐ (U+2610, &#9744;) = unchecked
                    # ASCII checkbox format (used by some filings):
                    # [X] or [x] = checked
                    # [  ] or [ ] or [&#160; ] = unchecked
                    is_checked = (
                        "☒" in content or
                        "☑" in content or
                        chr(9746) in content or
                        chr(9745) in content or
                        "[X]" in content or
                        "[x]" in content
                    )

                    # Also check format attribute for "fixed-true"
                    format_attr = elem.get("format", "")
                    if "fixed-true" in format_attr.lower():
                        is_checked = True

                    # Normalize key name (remove prefix)
                    key = tag_name.split(":")[-1]
                    flags[key] = is_checked
                    break

        return flags

    def classify_fund_type(self, flags: dict[str, bool]) -> FundType:
        """
        Classify fund type based on extracted flags.

        Logic:
        - If BusinessDevelopmentCompanyFlag is checked → BDC
        - If IntervalFundFlag is checked → Interval Fund
        - If RegisteredClosedEndFundFlag is checked (but not interval) → Tender Offer Fund
        - Otherwise → Other

        Args:
            flags: Dictionary of flag names to boolean values

        Returns:
            FundType enum value
        """
        # Check for BDC first
        if flags.get("BusinessDevelopmentCompanyFlag", False):
            return FundType.BDC

        # Check for interval fund
        if flags.get("IntervalFundFlag", False):
            return FundType.INTERVAL_FUND

        # Registered closed-end fund but not interval = tender offer
        if flags.get("RegisteredClosedEndFundFlag", False):
            return FundType.TENDER_OFFER_FUND

        return FundType.OTHER

    def extract_fund_type(self, html_content: str) -> tuple[FundType, dict[str, bool]]:
        """
        Extract fund type classification from iXBRL document.

        Convenience method that extracts flags and classifies in one call.

        Args:
            html_content: Raw HTML content of the iXBRL document

        Returns:
            Tuple of (FundType, flags_dict)
        """
        flags = self.extract_fund_type_flags(html_content)
        fund_type = self.classify_fund_type(flags)
        return fund_type, flags


@dataclass
class XBRLExtractionResult:
    """
    Result of XBRL extraction with observability metadata.

    Captures all extraction details for audit trail and debugging.
    """
    field_name: str
    value: Any
    found: bool
    share_class: Optional[str] = None
    tag_name: str = ""
    context_ref: str = ""
    raw_text: str = ""
    pattern_matched: Optional[str] = None  # Which share class pattern matched

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "field_name": self.field_name,
            "value": self.value if not isinstance(self.value, Decimal) else float(self.value),
            "found": self.found,
            "share_class": self.share_class,
            "tag_name": self.tag_name,
            "context_ref": self.context_ref,
            "raw_text": self.raw_text,
            "pattern_matched": self.pattern_matched,
        }


class XBRLValueExtractor:
    """
    High-level extractor for specific XBRL values.

    Maps XBRL tags to normalized field names for the database.
    Builds share-class-indexed fee tables for deterministic extraction.
    Supports observability logging for full extraction audit trail.
    """

    # Mapping of XBRL tags to normalized field names
    # Source: SEC CEF taxonomy + US-GAAP taxonomy
    # Full taxonomy: https://www.sec.gov/info/edgar/edgarfm-vol2-v59.pdf
    TAG_FIELD_MAP = {
        # =====================================================================
        # Fund Identification (dei: namespace)
        # =====================================================================
        "dei:EntityRegistrantName": "fund_name",
        "dei:EntityCentralIndexKey": "cik",

        # =====================================================================
        # Fund Type Flags (cef: namespace)
        # =====================================================================
        "cef:IntervalFundFlag": "is_interval_fund",
        "cef:RegisteredClosedEndFundFlag": "is_registered_cef",
        "cef:BusinessDevelopmentCompanyFlag": "is_bdc",
        "cef:PrimaryShelfFlag": "is_primary_shelf",
        "cef:PrimaryShelfQualifiedFlag": "is_primary_shelf_qualified",
        "cef:NewCefOrBdcRegistrantFlag": "is_new_registrant",

        # =====================================================================
        # Annual Fees & Expenses (by share class)
        # =====================================================================
        "cef:ManagementFeesPercent": "management_fee_pct",
        "cef:IncentiveFeesPercent": "incentive_fee_pct",  # Performance fee
        "cef:SalesLoadPercent": "sales_load_pct",
        "cef:DistributionServicingFeesPercent": "distribution_servicing_fee_pct",
        "cef:AcquiredFundFeesAndExpensesPercent": "affe_pct",
        "cef:InterestExpensesOnBorrowingsPercent": "interest_expense_pct",
        "cef:OtherAnnualExpensesPercent": "other_expenses_pct",
        "cef:OtherAnnualExpense1Percent": "other_expense_1_pct",
        "cef:OtherAnnualExpense2Percent": "other_expense_2_pct",
        "cef:TotalAnnualExpensesPercent": "total_expense_ratio_pct",
        "cef:NetExpenseOverAssetsPercent": "net_expense_ratio_pct",  # After waivers
        "cef:WaiversAndReimbursementsOfFeesPercent": "fee_waiver_pct",

        # =====================================================================
        # Transaction Fees (by share class)
        # =====================================================================
        "cef:OtherTransactionExpensesPercent": "other_transaction_fee_pct",
        "cef:OtherTransactionExpense1Percent": "early_repurchase_fee_pct",
        "cef:DividendReinvestmentAndCashPurchaseFees": "drip_fee",

        # =====================================================================
        # Expense Example (by share class)
        # =====================================================================
        "cef:ExpenseExampleYear01": "expense_example_1yr",
        "cef:ExpenseExampleYears1to3": "expense_example_3yr",
        "cef:ExpenseExampleYears1to5": "expense_example_5yr",
        "cef:ExpenseExampleYears1to10": "expense_example_10yr",

        # =====================================================================
        # NAV & Pricing (us-gaap: namespace)
        # =====================================================================
        "us-gaap:NetAssetValuePerShare": "nav_per_share",

        # =====================================================================
        # Shares Outstanding
        # =====================================================================
        "cef:OutstandingSecurityHeldShares": "shares_outstanding",
        "cef:OutstandingSecurityNotHeldShares": "shares_not_held",

        # =====================================================================
        # Leverage / Senior Securities
        # =====================================================================
        "cef:SeniorSecuritiesAmt": "leverage_amount",
        "cef:SeniorSecuritiesCvgPerUnit": "asset_coverage_ratio",
        "cef:AnnualCoverageReturnRatePercent": "annual_coverage_return_pct",

        # =====================================================================
        # Leverage Effects Table (hypothetical returns)
        # =====================================================================
        "cef:ReturnAtMinusTenPercent": "leverage_return_minus_10",
        "cef:ReturnAtMinusFivePercent": "leverage_return_minus_5",
        "cef:ReturnAtZeroPercent": "leverage_return_zero",
        "cef:ReturnAtPlusFivePercent": "leverage_return_plus_5",
        "cef:ReturnAtPlusTenPercent": "leverage_return_plus_10",
    }

    # Text blocks that contain extractable narrative content
    TEXT_BLOCK_FIELDS = {
        # =====================================================================
        # Risk Disclosures
        # =====================================================================
        "cef:RiskTextBlock": "risk_factors",
        "cef:RiskFactorsTableTextBlock": "risk_factors_table",

        # =====================================================================
        # Investment Strategy
        # =====================================================================
        "cef:InvestmentObjectivesAndPracticesTextBlock": "investment_objectives",

        # =====================================================================
        # Fee Tables & Notes
        # =====================================================================
        "cef:AnnualExpensesTableTextBlock": "annual_expenses_table",
        "cef:ShareholderTransactionExpensesTableTextBlock": "shareholder_fees_table",
        "cef:ExpenseExampleTableTextBlock": "expense_example_table",
        "cef:PurposeOfFeeTableNoteTextBlock": "fee_table_purpose_note",
        "cef:OtherExpensesNoteTextBlock": "other_expenses_note",
        "cef:OtherTransactionFeesNoteTextBlock": "repurchase_fee_note",
        "cef:ManagementFeeNotBasedOnNetAssetsNoteTextBlock": "mgmt_fee_methodology_note",
        "cef:AcquiredFundFeesAndExpensesNoteTextBlock": "affe_note",
        "cef:AcquiredFundFeesEstimatedNoteTextBlock": "affe_estimated_note",
        "cef:BasisOfTransactionFeesNoteTextBlock": "transaction_fees_basis_note",

        # =====================================================================
        # Leverage Disclosures
        # =====================================================================
        "cef:EffectsOfLeverageTextBlock": "leverage_effects",
        "cef:EffectsOfLeverageTableTextBlock": "leverage_effects_table",
        "cef:EffectsOfLeveragePurposeTextBlock": "leverage_purpose",
        "cef:SeniorSecuritiesAveragingMethodNoteTextBlock": "senior_securities_method_note",

        # =====================================================================
        # Share Class Information
        # =====================================================================
        "cef:CapitalStockTableTextBlock": "capital_stock_description",
        "cef:OutstandingSecuritiesTableTextBlock": "outstanding_shares_table",
        "cef:OutstandingSecurityTitleTextBlock": "share_class_title",
        "cef:SecurityTitleTextBlock": "security_title",

        # =====================================================================
        # Shareholder Rights
        # =====================================================================
        "cef:SecurityDividendsTextBlock": "dividend_policy",
        "cef:SecurityVotingRightsTextBlock": "voting_rights",
        "cef:SecurityLiquidationRightsTextBlock": "liquidation_rights",
        "cef:SecurityPreemptiveAndOtherRightsTextBlock": "preemptive_rights",
    }

    # Fee fields that should be grouped by share class
    SHARE_CLASS_FEE_FIELDS = [
        # Annual fees
        "management_fee_pct",
        "incentive_fee_pct",
        "sales_load_pct",
        "distribution_servicing_fee_pct",
        "affe_pct",
        "interest_expense_pct",
        "other_expenses_pct",
        "other_expense_1_pct",
        "other_expense_2_pct",
        "total_expense_ratio_pct",
        "net_expense_ratio_pct",
        "fee_waiver_pct",
        # Transaction fees
        "other_transaction_fee_pct",
        "early_repurchase_fee_pct",
        # Expense examples
        "expense_example_1yr",
        "expense_example_3yr",
        "expense_example_5yr",
        "expense_example_10yr",
        # NAV
        "nav_per_share",
    ]

    def __init__(
        self,
        parser: Optional[IXBRLParser] = None,
        enable_observability: bool = True,
    ):
        """
        Initialize with optional parser instance.

        Args:
            parser: Optional IXBRLParser instance
            enable_observability: Whether to log detailed extraction decisions
        """
        self.parser = parser or IXBRLParser()
        self.enable_observability = enable_observability
        # Cache for detected share classes
        self._detected_share_classes: list[str] = []
        # Observability: track all extraction results
        self._extraction_results: list[XBRLExtractionResult] = []
        # Observability: track pattern matches for debugging
        self._pattern_matches: list[dict] = []

    def extract_all(self, html_content: str) -> dict:
        """
        Extract all XBRL values and organize by field name.

        Returns:
            Dictionary with:
            - "fund_type": FundType classification (interval_fund, tender_offer_fund, bdc, etc.)
            - "fund_type_flags": dict of individual flag values
            - "numeric_fields": {field_name: value or {share_class: value}}
            - "text_blocks": {field_name: content}
            - "raw_values": list of all XBRLNumericValue objects
            - "raw_blocks": list of all XBRLTextBlock objects
            - "observability": dict with extraction trace (if enabled)
        """
        # Reset observability state
        self._extraction_results = []
        self._pattern_matches = []
        self._detected_share_classes = []

        numeric_values, text_blocks, contexts = self.parser.parse(html_content)

        # Extract fund type classification (deterministic from checkboxes)
        fund_type, fund_type_flags = self.parser.extract_fund_type(html_content)

        result = {
            "fund_type": fund_type.value,
            "fund_type_flags": fund_type_flags,
            "numeric_fields": {},
            "text_blocks": {},
            "raw_values": numeric_values,
            "raw_blocks": text_blocks,
        }

        # Process numeric values
        for value in numeric_values:
            field_name = self.TAG_FIELD_MAP.get(value.tag_name)
            if not field_name:
                continue

            # If there's a share class, organize by class
            if value.context.share_class:
                if field_name not in result["numeric_fields"]:
                    result["numeric_fields"][field_name] = {}
                result["numeric_fields"][field_name][value.context.share_class] = {
                    "value": float(value.value),
                    "raw_text": value.raw_text,
                }
            else:
                result["numeric_fields"][field_name] = {
                    "value": float(value.value),
                    "raw_text": value.raw_text,
                }

        # Process text blocks
        for block in text_blocks:
            field_name = self.TEXT_BLOCK_FIELDS.get(block.tag_name)
            if field_name:
                result["text_blocks"][field_name] = {
                    "content": block.content,
                    "char_count": block.char_count,
                    "estimated_tokens": block.estimated_tokens,
                }

        # Build share-class-indexed fee table
        result["share_class_fee_table"] = self.build_share_class_fee_table(numeric_values)
        result["detected_share_classes"] = self._detected_share_classes

        # Add observability data if enabled
        if self.enable_observability:
            result["observability"] = {
                "extraction_results": [r.to_dict() for r in self._extraction_results],
                "pattern_matches": self._pattern_matches,
                "total_numeric_values": len(numeric_values),
                "total_text_blocks": len(text_blocks),
                "total_contexts": len(contexts),
            }

        return result

    def build_share_class_fee_table(
        self, numeric_values: list[XBRLNumericValue]
    ) -> dict:
        """
        Build a share-class-indexed fee table for deterministic extraction.

        Groups XBRL fee values by share class, separating fund-level fees
        from share-class-specific fees.

        Args:
            numeric_values: List of parsed XBRL numeric values

        Returns:
            Dictionary with structure:
            {
                "fund_level": {field_name: value, ...},
                "by_class": {
                    "Class I": {field_name: value, ...},
                    "Class S": {field_name: value, ...},
                }
            }
        """
        fee_table = {
            "fund_level": {},
            "by_class": {},
        }

        # Track detected share classes
        detected_classes = set()

        for value in numeric_values:
            field_name = self.TAG_FIELD_MAP.get(value.tag_name)
            if not field_name:
                continue

            # Only process fee-related fields
            if field_name not in self.SHARE_CLASS_FEE_FIELDS:
                continue

            share_class = value.context.share_class

            if share_class:
                # Share-class-specific fee
                detected_classes.add(share_class)
                if share_class not in fee_table["by_class"]:
                    fee_table["by_class"][share_class] = {}

                fee_table["by_class"][share_class][field_name] = {
                    "value": float(value.value),
                    "raw_text": value.raw_text,
                    "context_ref": value.context.context_id,
                    "tag_name": value.tag_name,
                }

                # Log extraction for observability
                if self.enable_observability:
                    self._extraction_results.append(XBRLExtractionResult(
                        field_name=field_name,
                        value=float(value.value),
                        found=True,
                        share_class=share_class,
                        tag_name=value.tag_name,
                        context_ref=value.context.context_id,
                        raw_text=value.raw_text,
                    ))
            else:
                # Fund-level fee (no share class)
                fee_table["fund_level"][field_name] = {
                    "value": float(value.value),
                    "raw_text": value.raw_text,
                    "context_ref": value.context.context_id,
                    "tag_name": value.tag_name,
                }

                # Log extraction for observability
                if self.enable_observability:
                    self._extraction_results.append(XBRLExtractionResult(
                        field_name=field_name,
                        value=float(value.value),
                        found=True,
                        share_class=None,
                        tag_name=value.tag_name,
                        context_ref=value.context.context_id,
                        raw_text=value.raw_text,
                    ))

        # Update detected share classes cache
        self._detected_share_classes = sorted(detected_classes)

        return fee_table

    def get_detected_share_classes(self) -> list[str]:
        """
        Get list of share classes detected in the last extraction.

        Returns:
            Sorted list of share class names (e.g., ["Class D", "Class I", "Class S"])
        """
        return self._detected_share_classes

    def get_fee_for_share_class(
        self,
        fee_table: dict,
        field_name: str,
        share_class: str,
    ) -> Optional[dict]:
        """
        Get a specific fee value for a share class.

        Falls back to fund-level fee if not found for the specific class.

        Args:
            fee_table: Share class fee table from build_share_class_fee_table()
            field_name: Normalized field name (e.g., "management_fee_pct")
            share_class: Share class name (e.g., "Class I")

        Returns:
            Fee value dict with "value", "raw_text", etc., or None if not found
        """
        # Try share-class-specific first
        if share_class in fee_table.get("by_class", {}):
            class_fees = fee_table["by_class"][share_class]
            if field_name in class_fees:
                return class_fees[field_name]

        # Fall back to fund-level
        if field_name in fee_table.get("fund_level", {}):
            return fee_table["fund_level"][field_name]

        return None

    def get_extraction_results(self) -> list[XBRLExtractionResult]:
        """
        Get all extraction results from the last extraction.

        Returns:
            List of XBRLExtractionResult objects with full metadata
        """
        return self._extraction_results

    def get_observability_summary(self) -> dict:
        """
        Get a summary of observability data from the last extraction.

        Returns:
            Dictionary with summary statistics
        """
        if not self._extraction_results:
            return {"total_extractions": 0}

        by_field = {}
        by_share_class = {}

        for result in self._extraction_results:
            # Count by field
            by_field[result.field_name] = by_field.get(result.field_name, 0) + 1

            # Count by share class
            class_key = result.share_class or "fund_level"
            by_share_class[class_key] = by_share_class.get(class_key, 0) + 1

        return {
            "total_extractions": len(self._extraction_results),
            "detected_share_classes": self._detected_share_classes,
            "extractions_by_field": by_field,
            "extractions_by_share_class": by_share_class,
        }


def extract_xbrl_values(file_path: str) -> dict:
    """
    Convenience function to extract XBRL values from a file.

    Args:
        file_path: Path to the iXBRL HTML file

    Returns:
        Dictionary of extracted values organized by field name
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    extractor = XBRLValueExtractor()
    return extractor.extract_all(content)
