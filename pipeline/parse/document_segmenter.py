"""
Document Segmenter for SEC filings.

Identifies sections within N-2/N-CSR documents by analyzing
HTML structure (headings, formatting) and maps them to target fields.
"""

import hashlib
import logging
import re
from typing import Optional
from bs4 import BeautifulSoup, Tag, NavigableString
import tiktoken

from .models import (
    DocumentSection,
    DocumentMap,
    SectionType,
    ContentType,
    XBRLTextBlock,
    XBRLNumericValue,
)

logger = logging.getLogger(__name__)


class SectionFieldMapping:
    """
    Maps document sections to target extraction fields.

    Uses a two-tier approach:
    - Tier 1: XBRL text blocks (reliable, consistent across funds)
    - Tier 2: Known N-2 section names (for untagged content)
    """

    # ==========================================================================
    # TIER 1: XBRL TEXT BLOCKS (reliable, consistent)
    # ==========================================================================
    XBRL_BLOCK_FIELDS = {
        "cef:RiskTextBlock": ["key_risks"],
        "cef:RiskFactorsTableTextBlock": ["key_risks"],
        "cef:InvestmentObjectivesAndPracticesTextBlock": [
            "investment_objective",
            "strategy_description",
            "allocation_targets",  # Often contains allocation info
        ],
        "cef:AnnualExpensesTableTextBlock": ["fee_table", "share_classes", "incentive_fee"],  # Has per-class fees + incentive fee footnotes
        "cef:EffectsOfLeverageTextBlock": ["leverage_limits"],
        "cef:ShareholderTransactionExpensesTableTextBlock": ["transaction_fees", "share_classes"],  # Has per-class fees
        "cef:ExpenseExampleTableTextBlock": ["expense_example"],
        "cef:OutstandingSecuritiesTableTextBlock": ["share_classes"],
        # Additional blocks for share class data
        "cef:CapitalStockTableTextBlock": ["share_classes"],  # Has distribution fees per class
        "cef:OtherTransactionFeesNoteTextBlock": ["share_classes", "transaction_fees"],  # Has minimum investment
        "cef:ManagementFeeNotBasedOnNetAssetsNoteTextBlock": ["management_fee", "share_classes", "incentive_fee"],  # Has fee details including incentive fee structure
        # Fee-related note blocks that may contain incentive fee structure
        "cef:PurposeOfFeeTableNoteTextBlock": ["incentive_fee", "expense_cap"],  # Often describes fee structure
        "cef:OtherExpensesNoteTextBlock": ["incentive_fee", "expense_cap"],  # May contain fee cap/waiver info
        "cef:AcquiredFundFeesAndExpensesNoteTextBlock": ["incentive_fee"],  # Sometimes contains fund-level incentive fee details
        # Synthetic text block created from fee-related footnotes
        "ix:footnote:FeeStructure": ["incentive_fee", "expense_cap"],  # Contains incentive fee structure, hurdle rates, etc.
    }

    # Section heading patterns for untagged content
    HEADING_PATTERNS = {
        # Repurchase/Liquidity terms
        "repurchase_terms": {
            "patterns": [
                r"repurchase.*shares",
                r"repurchases\s+of\s+shares",
                r"periodic\s+repurchase",
                r"liquidity",
                r"tender\s+offer",
                r"repurchase\s+offer",
            ],
            "fields": [
                "repurchase_frequency",
                "repurchase_pct_nav",
                "repurchase_notice_days",
                "repurchase_pricing",
            ],
        },
        # Investment strategy and allocation targets
        "investment_strategy": {
            "patterns": [
                r"principal\s+investment\s+strateg",
                r"investment\s+objective",
                r"investment\s+policies",
                r"use\s+of\s+proceeds",
                r"allocation\s+target",           # Option D: allocation patterns
                r"asset\s+allocation",            # Option D
                r"portfolio\s+allocation",        # Option D
                r"target\s+allocation",           # Option D
                r"what\s+are.*allocation",        # Option D: Q&A format
            ],
            "fields": [
                "allocation_targets",
                "concentration_limits",
                "investment_restrictions",
            ],
        },
        # Distribution/Share classes
        "distribution": {
            "patterns": [
                r"plan\s+of\s+distribution",
                r"purchasing\s+shares",
                r"how\s+to\s+purchase",
                r"distribution\s+arrangements",
                r"minimum.*investment",           # Option D: minimum investment patterns
                r"initial\s+investment",          # Option D
                r"what\s+is.*minimum",            # Option D: Q&A format
                r"distribution.*fee",             # Option D: distribution fees
                r"service\s+fee",                 # Option D
                r"shareholder\s+servicing",       # Option D
                r"what\s+are.*fees.*investor",    # Q&A: "What are the fees that investors pay"
                r"fees.*shares?\s+purchase",      # Fee questions about share purchases
                r"sales\s+(load|charge)",         # Sales load patterns
            ],
            "fields": [
                "minimum_investment",
                "share_classes",
                "distribution_channels",
            ],
        },
        # Fee structure
        "fees": {
            "patterns": [
                r"fees?\s+and\s+expenses",
                r"fund\s+expenses",
                r"management\s+fee",
                r"advisory\s+fee",
                r"incentive\s+fee",
                r"fee\s+table",                    # Fee table sections
                r"summary\s+of\s+fund\s+expenses", # Summary fee tables
            ],
            "fields": [
                "management_fee",
                "incentive_fee",
                "performance_fee",
                "expense_cap",
                "share_classes",  # Fee sections contain per-class fee percentages
            ],
        },
        # Fund management
        "management": {
            "patterns": [
                r"management\s+of\s+the\s+fund",
                r"investment\s+advis[eo]r",
                r"board\s+of\s+(directors|trustees)",
                r"portfolio\s+manager",
            ],
            "fields": [
                "manager_name",
                "subadvisor",
                "gp_commitment",
            ],
        },
        # Risk factors
        "risks": {
            "patterns": [
                r"risk\s+factors",
                r"principal\s+risks",
                r"certain\s+risks",
            ],
            "fields": ["key_risks"],
        },
        # Leverage and borrowings
        "leverage": {
            "patterns": [
                r"leverage",
                r"borrowing",
                r"credit\s+facility",
                r"line\s+of\s+credit",
                r"debt\s+financing",
            ],
            "fields": ["leverage_limits"],
        },
        # Tax information
        "tax": {
            "patterns": [
                r"tax\s+considerations",
                r"federal\s+income\s+tax",
                r"tax\s+matters",
            ],
            "fields": ["tax_treatment"],
        },
        # Q&A Summary section - contains key fund terms in Q&A format
        # Located at ~3-10% of N-2 documents, before Fee Tables
        "qa_summary": {
            "patterns": [
                r"^is\s+there\s+any\s+minimum",          # Minimum investment Q
                r"^is\s+there\s+a\s+minimum",            # Alternative phrasing
                r"^what\s+is\s+the\s+minimum",           # Alternative phrasing
                r"^will\s+i\s+receive\s+distributions",  # Distribution frequency Q
                r"^how\s+often\s+will\s+.*distribut",    # Distribution frequency Q
                r"^can\s+i\s+request\s+.*repurchas",     # Repurchase terms Q
                r"^how\s+can\s+i\s+.*redeem",            # Redemption/repurchase Q
                r"^will\s+you\s+use\s+leverage",         # Leverage Q
                r"^what\s+is\s+.*fund",                  # Fund type Q (e.g., "What is the Fund?")
                r"^what\s+type\s+of\s+fund",             # Fund type Q
            ],
            "fields": [
                "share_classes",       # Minimum investment amounts
                "minimum_investment",  # Explicit minimum investment field
                "distribution_terms",  # Distribution frequency
                "repurchase_terms",    # Repurchase information
                "leverage_limits",     # Leverage usage
                "fund_type",           # Fund structure type
            ],
        },
    }

    # ==========================================================================
    # TIER 2: KNOWN N-2 SECTION NAMES (for untagged content)
    # These are searched directly in the document text, not relying on CSS styling
    #
    # Form N-2 has SEC-mandated sections (Part A Prospectus) plus fund-specific
    # discretionary sections. This list covers both.
    #
    # Sources:
    # - SEC Form N-2: https://www.sec.gov/files/formn-2.pdf
    # - 17 CFR 274.11a-1
    # ==========================================================================
    N2_STANDARD_SECTIONS = {
        # ======================================================================
        # SEC-MANDATED SECTIONS (Form N-2 Part A - Prospectus)
        # ======================================================================

        # Cover and navigation (usually not extracted, but needed for completeness)
        "table_of_contents": {
            "patterns": [
                r"TABLE\s+OF\s+CONTENTS",
                r"Table\s+of\s+Contents",
            ],
            "fields": [],  # Navigation only
            "priority": 3,
        },

        # Prospectus summary
        "prospectus_summary": {
            "patterns": [
                r"PROSPECTUS\s+SUMMARY",
                r"Prospectus\s+Summary",
                r"SUMMARY\s+OF\s+(?:THE\s+)?(?:FUND|PROSPECTUS)",
            ],
            "fields": ["fund_type", "investment_objective", "strategy_description"],
            "priority": 1,
        },

        # Fee table (SEC-mandated disclosure)
        "summary_of_fund_expenses": {
            "patterns": [
                r"SUMMARY\s+OF\s+FUND\s+EXPENSES",
                r"FEES?\s+AND\s+EXPENSES",
                r"Summary\s+of\s+Fund\s+Expenses",
                r"FEE\s+TABLE",
            ],
            "fields": ["management_fee", "incentive_fee", "expense_cap", "share_classes"],
            "priority": 1,
        },

        # Financial highlights (SEC-mandated)
        "financial_highlights": {
            "patterns": [
                r"FINANCIAL\s+HIGHLIGHTS",
                r"Financial\s+Highlights",
            ],
            "fields": ["nav_history", "expense_ratio", "performance"],
            "priority": 2,
        },

        # The Fund description
        "the_fund": {
            "patterns": [
                r"THE\s+FUND",
                r"The\s+Fund",
                r"ABOUT\s+THE\s+FUND",
            ],
            "fields": ["fund_type", "legal_structure", "fund_name"],
            "priority": 2,
        },

        # Use of proceeds
        "use_of_proceeds": {
            "patterns": [
                r"USE\s+OF\s+PROCEEDS",
                r"Use\s+of\s+Proceeds",
            ],
            "fields": ["use_of_proceeds"],
            "priority": 2,
        },

        # Investment objective (SEC-mandated)
        "investment_objective": {
            "patterns": [
                r"INVESTMENT\s+OBJECTIVE",
                r"Investment\s+Objective",
                r"INVESTMENT\s+GOAL",
            ],
            "fields": ["investment_objective"],
            "priority": 1,
        },

        # Investment strategies (SEC-mandated)
        "investment_strategies": {
            "patterns": [
                r"INVESTMENT\s+STRATEG(?:Y|IES)",
                r"PRINCIPAL\s+INVESTMENT\s+STRATEG",
                r"Investment\s+Strateg(?:y|ies)",
            ],
            "fields": ["strategy_description", "allocation_targets"],
            "priority": 1,
        },

        # Investment policies
        "investment_policies": {
            "patterns": [
                r"INVESTMENT\s+POLICIES",
                r"Investment\s+Policies",
                r"FUNDAMENTAL\s+POLICIES",
            ],
            "fields": ["investment_policies", "concentration_limits"],
            "priority": 2,
        },

        # Risk factors (SEC-mandated, often XBRL-tagged)
        "risk_factors": {
            "patterns": [
                r"RISK\s+FACTORS?",
                r"PRINCIPAL\s+RISKS?",
                r"Risk\s+Factors?",
                r"CERTAIN\s+RISKS",
            ],
            "fields": ["key_risks"],
            "priority": 1,
        },

        # Management of the fund
        "management_of_fund": {
            "patterns": [
                r"MANAGEMENT\s+OF\s+THE\s+FUND",
                r"Management\s+of\s+the\s+Fund",
                r"INVESTMENT\s+ADVISER",
                r"THE\s+ADVISER",
            ],
            "fields": ["manager_name", "subadvisor", "gp_commitment"],
            "priority": 1,
        },

        # Management and incentive fees
        "management_and_incentive_fees": {
            "patterns": [
                r"MANAGEMENT\s+AND\s+INCENTIVE\s+FEES?",
                r"INCENTIVE\s+FEE",
                r"PERFORMANCE\s+FEE",
                r"Management\s+and\s+Incentive\s+Fee",
                r"MANAGEMENT\s+FEE",
            ],
            "fields": ["management_fee", "incentive_fee", "hurdle_rate", "high_water_mark", "catch_up"],
            "priority": 1,
        },

        # Capital stock / Description of shares (SEC-mandated)
        "capital_stock": {
            "patterns": [
                r"CAPITAL\s+STOCK",
                r"Capital\s+Stock",
                r"DESCRIPTION\s+OF\s+(?:SHARES|SECURITIES|CAPITAL)",
                r"Description\s+of\s+(?:Shares|Securities)",
            ],
            "fields": ["share_classes", "voting_rights"],
            "priority": 1,
        },

        # Distribution/Purchase sections (contain minimum investment)
        "plan_of_distribution": {
            "patterns": [
                r"PLAN\s+OF\s+DISTRIBUTION",
                r"Plan\s+of\s+Distribution",
            ],
            "fields": ["minimum_investment", "share_classes", "distribution_channels"],
            "priority": 1,
        },

        "purchases_of_shares": {
            "patterns": [
                r"PURCHASES?\s+OF\s+SHARES",
                r"PURCHASING\s+SHARES",
                r"HOW\s+TO\s+PURCHASE",
                r"Purchase\s+of\s+Shares",
                r"Purchasing\s+Shares",
            ],
            "fields": ["minimum_investment", "share_classes", "purchase_terms"],
            "priority": 1,
        },

        # Repurchase sections
        "repurchase_of_shares": {
            "patterns": [
                r"REPURCHASES?\s+OF\s+SHARES",
                r"REPURCHASE\s+OFFERS?",
                r"TENDER\s+OFFERS?",
                r"Repurchase\s+of\s+Shares",
                r"Repurchases?\s+and\s+Transfers?\s+of\s+Shares",
                r"PERIODIC\s+REPURCHASE",
            ],
            "fields": ["repurchase_terms", "repurchase_frequency", "lock_up_period", "early_repurchase_fee"],
            "priority": 1,
        },

        # Distribution policy
        "distribution_policy": {
            "patterns": [
                r"DISTRIBUTION\s+POLICY",
                r"DIVIDEND\s+POLICY",
                r"Distribution\s+Policy",
                r"DISTRIBUTIONS",
            ],
            "fields": ["distribution_terms", "distribution_frequency"],
            "priority": 2,
        },

        # Dividend reinvestment
        "dividend_reinvestment": {
            "patterns": [
                r"DIVIDEND\s+REINVESTMENT",
                r"DISTRIBUTION\s+REINVESTMENT",
                r"Dividend\s+Reinvestment",
                r"DRIP",
            ],
            "fields": ["dividend_reinvestment"],
            "priority": 3,
        },

        # Tax considerations (SEC-mandated)
        "tax_considerations": {
            "patterns": [
                r"TAX\s+CONSIDERATIONS",
                r"FEDERAL\s+INCOME\s+TAX",
                r"Tax\s+Considerations",
                r"TAX\s+MATTERS",
                r"TAXATION",
            ],
            "fields": ["tax_treatment"],
            "priority": 2,
        },

        # Legal matters (SEC-mandated)
        "legal_matters": {
            "patterns": [
                r"LEGAL\s+MATTERS",
                r"Legal\s+Matters",
                r"LEGAL\s+COUNSEL",
            ],
            "fields": ["legal_counsel"],
            "priority": 3,
        },

        # Experts (SEC-mandated)
        "experts": {
            "patterns": [
                r"EXPERTS?",
                r"INDEPENDENT\s+(?:REGISTERED\s+)?PUBLIC\s+ACCOUNT",
                r"AUDITORS?",
            ],
            "fields": ["auditor"],
            "priority": 3,
        },

        # Additional information (SEC-mandated)
        "additional_information": {
            "patterns": [
                r"ADDITIONAL\s+INFORMATION",
                r"Additional\s+Information",
                r"WHERE\s+YOU\s+CAN\s+FIND",
                r"AVAILABLE\s+INFORMATION",
            ],
            "fields": [],
            "priority": 3,
        },

        # ======================================================================
        # DISCRETIONARY / FUND-SPECIFIC SECTIONS
        # ======================================================================

        # Eligible investors (common for interval/tender offer funds)
        "eligible_investors": {
            "patterns": [
                r"ELIGIBLE\s+INVESTORS?",
                r"Eligible\s+Investors?",
                r"WHO\s+MAY\s+INVEST",
                r"INVESTOR\s+SUITABILITY",
            ],
            "fields": ["minimum_investment", "investor_requirements"],
            "priority": 1,
        },

        # ERISA considerations (for pension/retirement investors)
        "erisa_considerations": {
            "patterns": [
                r"ERISA\s+CONSIDERATIONS",
                r"ERISA\s+AND\s+TAX",
                r"Erisa\s+Considerations",
                r"BENEFIT\s+PLAN\s+INVESTORS",
            ],
            "fields": ["erisa_eligible"],
            "priority": 2,
        },

        # Conflicts of interest
        "conflicts_of_interest": {
            "patterns": [
                r"CONFLICTS?\s+OF\s+INTEREST",
                r"Conflicts?\s+of\s+Interest",
                r"POTENTIAL\s+CONFLICTS",
            ],
            "fields": ["conflicts_of_interest"],
            "priority": 2,
        },

        # Valuation / NAV
        "valuation": {
            "patterns": [
                r"VALUATION",
                r"Valuation",
                r"NET\s+ASSET\s+VALUE",
                r"CALCULATION\s+OF\s+NAV",
                r"DETERMINATION\s+OF\s+NAV",
            ],
            "fields": ["valuation_methodology", "nav_frequency"],
            "priority": 2,
        },

        # Leverage
        "leverage": {
            "patterns": [
                r"LEVERAGE",
                r"USE\s+OF\s+LEVERAGE",
                r"BORROWINGS?",
                r"EFFECTS\s+OF\s+LEVERAGE",
            ],
            "fields": ["leverage_limits", "uses_leverage", "max_leverage"],
            "priority": 2,
        },

        # Expense cap/limitation
        "expense_limitation": {
            "patterns": [
                r"EXPENSE\s+CAP",
                r"EXPENSE\s+LIMITATION",
                r"FEE\s+WAIVER",
                r"Expense\s+Cap",
                r"Expense\s+Limitation",
                r"WAIVER\s+AND\s+REIMBURSEMENT",
            ],
            "fields": ["expense_cap", "expense_cap_expiration"],
            "priority": 2,
        },

        # Custodian
        "custodian": {
            "patterns": [
                r"CUSTODIAN",
                r"Custodian",
                r"CUSTODY\s+OF\s+ASSETS",
            ],
            "fields": ["custodian"],
            "priority": 3,
        },

        # Transfer agent
        "transfer_agent": {
            "patterns": [
                r"TRANSFER\s+AGENT",
                r"Transfer\s+Agent",
                r"REGISTRAR\s+AND\s+TRANSFER",
            ],
            "fields": ["transfer_agent"],
            "priority": 3,
        },

        # Anti-takeover provisions
        "anti_takeover": {
            "patterns": [
                r"ANTI-?TAKEOVER\s+PROVISIONS?",
                r"Anti-?Takeover\s+Provisions?",
                r"CERTAIN\s+PROVISIONS\s+OF.*CHARTER",
            ],
            "fields": ["anti_takeover_provisions"],
            "priority": 3,
        },

        # Privacy notice
        "privacy_notice": {
            "patterns": [
                r"PRIVACY\s+(?:POLICY|NOTICE)",
                r"Privacy\s+(?:Policy|Notice)",
            ],
            "fields": [],
            "priority": 3,
        },

        # Brokerage allocation
        "brokerage": {
            "patterns": [
                r"BROKERAGE\s+ALLOCATION",
                r"BROKERAGE\s+PRACTICES",
                r"Brokerage\s+Allocation",
            ],
            "fields": ["brokerage_practices"],
            "priority": 3,
        },

        # Closed-end fund structure
        "closed_end_structure": {
            "patterns": [
                r"CLOSED-?END\s+FUND\s+STRUCTURE",
                r"INTERVAL\s+FUND\s+RISKS?",
                r"LIMITED\s+LIQUIDITY",
            ],
            "fields": ["fund_type", "liquidity_terms"],
            "priority": 2,
        },

        # Portfolio managers
        "portfolio_managers": {
            "patterns": [
                r"PORTFOLIO\s+MANAGERS?",
                r"Portfolio\s+Managers?",
                r"INVESTMENT\s+PERSONNEL",
            ],
            "fields": ["portfolio_managers"],
            "priority": 2,
        },
    }

    @classmethod
    def get_fields_for_section(cls, section_title: str, xbrl_tag: Optional[str] = None) -> list[str]:
        """
        Determine target fields for a section.

        Args:
            section_title: The heading/title of the section
            xbrl_tag: Optional XBRL tag if this is a tagged text block

        Returns:
            List of field names to extract from this section
        """
        fields = []

        # Tier 1: Check XBRL tag first (most reliable)
        if xbrl_tag and xbrl_tag in cls.XBRL_BLOCK_FIELDS:
            fields.extend(cls.XBRL_BLOCK_FIELDS[xbrl_tag])

        # Tier 2: Check N-2 standard section patterns
        for section_name, config in cls.N2_STANDARD_SECTIONS.items():
            for pattern in config["patterns"]:
                if re.search(pattern, section_title, re.IGNORECASE):
                    fields.extend(config["fields"])
                    break

        # Legacy: Check heading patterns (for backwards compatibility)
        title_lower = section_title.lower()
        for section_type, config in cls.HEADING_PATTERNS.items():
            for pattern in config["patterns"]:
                if re.search(pattern, title_lower, re.IGNORECASE):
                    fields.extend(config["fields"])
                    break

        return list(set(fields))  # Deduplicate

    @classmethod
    def get_n2_section_name(cls, text: str) -> Optional[tuple[str, list[str]]]:
        """
        Check if text matches a known N-2 section name.

        Args:
            text: Text to check (usually first ~100 chars of a section)

        Returns:
            Tuple of (section_name, fields) if matched, None otherwise
        """
        for section_name, config in cls.N2_STANDARD_SECTIONS.items():
            for pattern in config["patterns"]:
                if re.search(pattern, text, re.IGNORECASE):
                    return (section_name, config["fields"])
        return None


class DocumentSegmenter:
    """
    Segments SEC filings into logical sections.

    Handles both iXBRL tagged content and untagged HTML sections.
    """

    # Heading tag priorities (lower = more important)
    HEADING_TAGS = {
        "h1": 1,
        "h2": 2,
        "h3": 3,
        "h4": 4,
        "h5": 5,
        "h6": 6,
    }

    # Style patterns that indicate headings
    HEADING_STYLES = [
        r"font-weight:\s*bold",
        r"font-weight:\s*700",
        r"text-transform:\s*uppercase",
    ]

    # Font size patterns for headings (larger = more important)
    FONT_SIZE_PATTERN = r"font-size:\s*(\d+(?:\.\d+)?)\s*pt"

    def __init__(self, min_section_chars: int = 100):
        """
        Initialize segmenter.

        Args:
            min_section_chars: Minimum characters for a valid section
        """
        self.min_section_chars = min_section_chars
        self._tokenizer = None
        try:
            self._tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        except Exception:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        return len(text) // 4

    def segment(
        self,
        html_content: str,
        xbrl_text_blocks: list[XBRLTextBlock],
        xbrl_numeric_values: list[XBRLNumericValue],
        filing_metadata: dict,
    ) -> DocumentMap:
        """
        Segment document into logical sections using two-tier approach.

        Tier 1: XBRL text blocks (reliable, consistent across funds)
        Tier 2: Known N-2 section names searched in document text

        Args:
            html_content: Raw HTML content
            xbrl_text_blocks: Pre-extracted XBRL text blocks
            xbrl_numeric_values: Pre-extracted XBRL numeric values
            filing_metadata: Dict with cik, accession_number, form_type, filing_date

        Returns:
            DocumentMap with all identified sections
        """
        soup = BeautifulSoup(html_content, "lxml")

        # Get full text for character positions
        full_text = soup.get_text()

        # =====================================================================
        # TIER 1: XBRL text blocks (most reliable)
        # =====================================================================
        xbrl_sections = self._create_xbrl_sections(xbrl_text_blocks)
        logger.info(f"    Tier 1: {len(xbrl_sections)} XBRL sections")

        # =====================================================================
        # TIER 2: Known N-2 section names (for untagged content)
        # =====================================================================
        n2_sections = self._find_n2_pattern_sections(full_text, html_content)
        logger.info(f"    Tier 2: {len(n2_sections)} N-2 pattern sections")

        # =====================================================================
        # LEGACY: CSS-based heading detection (for backwards compatibility)
        # =====================================================================
        headings = self._find_headings(soup)
        heading_sections = self._build_sections_from_headings(soup, headings, full_text)

        # Merge all sections (XBRL takes priority, then N-2, then headings)
        all_sections = self._merge_all_sections(xbrl_sections, n2_sections, heading_sections)

        # Determine which sections need extraction
        for section in all_sections:
            target_fields = SectionFieldMapping.get_fields_for_section(
                section.title, section.xbrl_tag
            )
            if target_fields:
                section.needs_extraction = True
                section.target_fields = target_fields

        # Create document map
        doc_map = DocumentMap(
            filing_id=f"{filing_metadata['cik']}_{filing_metadata['accession_number']}",
            cik=filing_metadata["cik"],
            accession_number=filing_metadata["accession_number"],
            form_type=filing_metadata["form_type"],
            filing_date=filing_metadata["filing_date"],
            total_chars=len(full_text),
            total_tokens=self._count_tokens(full_text),
            sections=all_sections,
            xbrl_numeric_values=xbrl_numeric_values,
            xbrl_text_blocks=xbrl_text_blocks,
        )

        return doc_map

    def _find_n2_pattern_sections(
        self,
        full_text: str,
        html_content: str,
    ) -> list[DocumentSection]:
        """
        Find sections by searching for known N-2 section names in the document.

        This is Tier 2 of the two-tier approach. It searches the full document
        text for standard N-2 section headings and creates sections from matches.

        Args:
            full_text: Plain text content of the document
            html_content: Original HTML content (for extracting HTML sections)

        Returns:
            List of DocumentSection objects for matched N-2 patterns
        """
        sections = []
        seen_sections = set()  # Avoid duplicates

        # Normalize text to handle embedded line breaks in section headings
        # Many SEC documents have headings split across multiple lines
        # e.g., "THE\n    FUND" should match "THE FUND"
        normalized_text = re.sub(r'\s+', ' ', full_text)

        for section_name, config in SectionFieldMapping.N2_STANDARD_SECTIONS.items():
            for pattern in config["patterns"]:
                for match in re.finditer(pattern, normalized_text, re.IGNORECASE | re.MULTILINE):
                    # Skip if we already found this section
                    if section_name in seen_sections:
                        continue

                    # Get matched text and find its position in original full_text
                    # We need to create a flexible pattern that allows whitespace variations
                    matched_text = match.group(0)
                    # Convert matched text to flexible pattern (allow any whitespace between words)
                    words = matched_text.split()
                    flexible_pattern = r'\s*'.join(re.escape(w) for w in words)

                    # Find in original full_text
                    original_match = re.search(flexible_pattern, full_text, re.IGNORECASE)
                    if not original_match:
                        continue

                    char_start = original_match.start()

                    # Find the next section heading to determine section end
                    # Look for the next match of ANY N-2 pattern in normalized text
                    char_end = len(normalized_text)
                    for other_name, other_config in SectionFieldMapping.N2_STANDARD_SECTIONS.items():
                        if other_name == section_name:
                            continue
                        for other_pattern in other_config["patterns"]:
                            other_match = re.search(
                                other_pattern,
                                normalized_text[match.end() + 50:],  # Skip past current match
                                re.IGNORECASE | re.MULTILINE
                            )
                            if other_match:
                                potential_end = match.end() + 50 + other_match.start()
                                if potential_end < char_end:
                                    char_end = potential_end

                    # Map normalized char_end back to full_text position (approximate)
                    # Since normalized text is shorter, we need to find a corresponding position
                    # Use a ratio-based approach
                    ratio = len(full_text) / len(normalized_text) if len(normalized_text) > 0 else 1.0
                    char_end_original = min(int(char_end * ratio), len(full_text))

                    # Limit section size to reasonable amount (50KB max)
                    char_end_original = min(char_end_original, char_start + 50000)

                    # Extract content from original full_text
                    content = full_text[char_start:char_end_original]

                    # Skip if too short
                    if len(content) < self.min_section_chars:
                        continue

                    # Create section
                    section_id = hashlib.md5(
                        f"n2_{section_name}_{char_start}".encode()
                    ).hexdigest()[:12]

                    # Use the matched text as the title (normalized)
                    title = matched_text.strip()

                    sections.append(DocumentSection(
                        section_id=section_id,
                        title=title,
                        section_type=SectionType.UNTAGGED,
                        content_type=ContentType.MIXED,
                        char_start=char_start,
                        char_end=char_end_original,
                        content=content,
                        content_html=content,  # Plain text for now
                        char_count=len(content),
                        estimated_tokens=self._count_tokens(content),
                        heading_level=2,
                        needs_extraction=True,  # Always extract N-2 pattern sections
                        target_fields=config["fields"],
                    ))

                    seen_sections.add(section_name)
                    break  # Only take first match per section type

        return sections

    def _merge_all_sections(
        self,
        xbrl_sections: list[DocumentSection],
        n2_sections: list[DocumentSection],
        heading_sections: list[DocumentSection],
    ) -> list[DocumentSection]:
        """
        Merge sections from all three sources, with XBRL taking priority.

        Priority order:
        1. XBRL sections (most reliable)
        2. N-2 pattern sections (Tier 2)
        3. CSS-based heading sections (legacy, least reliable)

        Sections that overlap with higher-priority sections are excluded.
        """
        all_sections = []

        # First, merge duplicate XBRL sections with same title
        merged_xbrl = self._merge_duplicate_xbrl_sections(xbrl_sections)
        xbrl_titles = set()
        xbrl_ranges = []

        for section in merged_xbrl:
            all_sections.append(section)
            xbrl_titles.add(section.title.lower())
            xbrl_ranges.append((section.char_start, section.char_end))

        # Add N-2 pattern sections that don't overlap with XBRL
        n2_titles = set()
        for section in n2_sections:
            # Check if title overlaps with XBRL
            title_lower = section.title.lower()
            is_duplicate = any(
                self._titles_similar(title_lower, xbrl_title)
                for xbrl_title in xbrl_titles
            )
            if not is_duplicate:
                all_sections.append(section)
                n2_titles.add(title_lower)

        # Add heading sections that don't overlap with XBRL or N-2
        for section in heading_sections:
            title_lower = section.title.lower()
            is_xbrl_duplicate = any(
                self._titles_similar(title_lower, xbrl_title)
                for xbrl_title in xbrl_titles
            )
            is_n2_duplicate = any(
                self._titles_similar(title_lower, n2_title)
                for n2_title in n2_titles
            )
            if not is_xbrl_duplicate and not is_n2_duplicate:
                all_sections.append(section)

        # Sort by character position
        all_sections.sort(key=lambda s: s.char_start)

        return all_sections

    # Q&A patterns commonly found in SEC prospectuses (Option C)
    QA_PATTERNS = [
        r"^Q:\s*",                    # Standard Q: prefix
        r"^Question:\s*",             # Full "Question:" prefix
        r"^What\s+(is|are)\s+",       # Questions starting with "What is/are"
        r"^How\s+(do|does|can)\s+",   # Questions starting with "How"
        r"^Who\s+(is|are)\s+",        # Questions starting with "Who"
        r"^Is\s+there\s+",            # Questions starting with "Is there" (e.g., minimum investment)
        r"^Will\s+(I|you|the)\s+",    # Questions starting with "Will" (e.g., distributions)
        r"^Can\s+(I|you|shareholders)\s+",  # Questions starting with "Can" (e.g., repurchases)
        r"^When\s+(will|can|do)\s+",  # Questions starting with "When"
        r"^Where\s+(can|will)\s+",    # Questions starting with "Where"
    ]

    # Cover page / document header patterns to skip
    COVER_PAGE_PATTERNS = [
        r"^REGISTRATION\s+STATEMENT",
        r"^SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION",
        r"^UNITED\s+STATES$",
        r"^FORM\s*N-",
        r"^TABLE\s+OF\s+CONTENTS$",
        r"^PROSPECTUS$",
        r"^STATEMENT\s+OF\s+ADDITIONAL\s+INFORMATION",
        r"^PART\s+[A-C]",
    ]

    def _is_cover_page_header(self, elem: Tag, text: str) -> bool:
        """
        Check if element is a document cover page header.

        Cover page headers are structural elements at the document start,
        not content section headers. They typically include:
        - SEC form type declarations
        - "TABLE OF CONTENTS" header
        - Registration statement declarations

        These should be skipped because their "content" would be
        the entire document.
        """
        text_upper = text.strip().upper()

        # Check against cover page patterns
        for pattern in self.COVER_PAGE_PATTERNS:
            if re.match(pattern, text_upper, re.IGNORECASE):
                return True

        # Check if this is a very early element (within first 5% of doc)
        # that's a direct child of body - likely a cover page element
        if elem.parent and elem.parent.name == "body":
            # Cover page elements are typically short titles
            if len(text) < 100 and text_upper.isupper():
                return True

        return False

    def _is_toc_entry(self, elem: Tag) -> bool:
        """
        Check if a heading element is inside a Table of Contents.

        TOC tables typically have:
        - Multiple rows (5+)
        - Each row has a heading-like cell + a page number cell
        - Page numbers are short numeric strings

        We want to skip these because they're links to sections, not
        the actual section starts.
        """
        # Must be inside a table
        parent_table = elem.find_parent("table")
        if not parent_table:
            return False

        rows = parent_table.find_all("tr", recursive=False)
        # Also check nested tbody
        tbody = parent_table.find("tbody")
        if tbody:
            rows = tbody.find_all("tr", recursive=False)

        # TOCs typically have many rows
        if len(rows) < 5:
            return False

        # Check for TOC pattern: multiple rows with heading + page number pattern
        page_number_count = 0
        for row in rows[:10]:  # Sample first 10 rows
            cells = row.find_all("td", recursive=False)
            if len(cells) >= 2:
                # Check if last cell looks like a page number
                last_cell_text = cells[-1].get_text(strip=True)
                # Page numbers are typically 1-4 digit numbers
                if re.match(r'^\d{1,4}$', last_cell_text):
                    page_number_count += 1

        # If >50% of sampled rows have page numbers, it's likely a TOC
        if page_number_count >= 3:
            return True

        # Alternative pattern: check if table has "TABLE OF CONTENTS" nearby
        # or contains many internal links
        table_text = parent_table.get_text(separator=" ", strip=True)[:500].upper()
        if "TABLE OF CONTENTS" in table_text or "CONTENTS" == table_text[:8]:
            return True

        # Check for many anchor links (common in TOCs)
        links = parent_table.find_all("a", href=True)
        internal_links = [l for l in links if l.get("href", "").startswith("#")]
        if len(internal_links) > 10:
            return True

        return False

    def _find_headings(self, soup: BeautifulSoup) -> list[dict]:
        """
        Find all heading elements in the document.

        Returns list of dicts with:
        - element: BeautifulSoup Tag
        - level: Heading level (1-6)
        - text: Heading text
        - char_pos: Character position in document
        """
        headings = []

        # Find standard heading tags
        for tag_name, level in self.HEADING_TAGS.items():
            for elem in soup.find_all(tag_name):
                text = elem.get_text(strip=True)
                if text and len(text) > 2:
                    headings.append({
                        "element": elem,
                        "level": level,
                        "text": text,
                        "tag": tag_name,
                    })

        # Find styled headings (bold, large font)
        for elem in soup.find_all(["p", "div", "span", "td"]):
            if self._is_styled_heading(elem):
                text = elem.get_text(strip=True)
                if text and len(text) > 2 and len(text) < 200:
                    # Skip TOC entries - they look like headings but are just links
                    if self._is_toc_entry(elem):
                        continue
                    # Skip cover page headers - structural, not content
                    if self._is_cover_page_header(elem, text):
                        continue
                    # Avoid duplicates
                    if not any(h["text"] == text for h in headings):
                        headings.append({
                            "element": elem,
                            "level": self._estimate_heading_level(elem),
                            "text": text,
                            "tag": elem.name,
                        })

        # Option C: Find Q&A format headings (common in SEC prospectus summaries)
        # These are often bold questions that introduce important content
        # Include "tr" elements since Q&A content is often in table rows
        for elem in soup.find_all(["p", "div", "span", "td", "tr"]):
            text = elem.get_text(strip=True)
            if text and len(text) > 10 and len(text) < 200:
                # Check if text matches Q&A pattern
                is_qa = any(re.match(p, text, re.IGNORECASE) for p in self.QA_PATTERNS)
                if is_qa:
                    # Skip TOC entries
                    if self._is_toc_entry(elem):
                        continue
                    # Skip cover page headers
                    if self._is_cover_page_header(elem, text):
                        continue
                    # Check if element has some styling (bold preferred but not required for Q:)
                    style = elem.get("style", "")
                    is_bold = any(re.search(p, style, re.IGNORECASE) for p in self.HEADING_STYLES[:2])

                    # Accept Q&A if bold, or if starts with "Q:" regardless of styling
                    if is_bold or text.startswith("Q:") or text.startswith("Question:"):
                        # Avoid duplicates
                        if not any(h["text"] == text for h in headings):
                            headings.append({
                                "element": elem,
                                "level": 4,  # Q&A headings are typically subsection level
                                "text": text,
                                "tag": elem.name,
                            })

        # Sort by document order
        headings.sort(key=lambda h: self._get_element_position(h["element"]))

        return headings

    def _is_styled_heading(self, elem: Tag) -> bool:
        """Check if element is styled like a heading."""
        style = elem.get("style", "")
        class_attr = " ".join(elem.get("class", []))

        # Check for bold
        is_bold = any(re.search(p, style, re.IGNORECASE) for p in self.HEADING_STYLES[:2])

        # Check for large font (Option A: lowered threshold from 12pt to 10pt)
        # SEC filings commonly use 10pt bold for headings
        font_match = re.search(self.FONT_SIZE_PATTERN, style)
        is_large = font_match and float(font_match.group(1)) >= 10

        # Check for uppercase text
        text = elem.get_text(strip=True)
        is_uppercase = text.isupper() and len(text) > 5

        # Check for bold child
        has_bold_child = elem.find(["b", "strong"]) is not None

        return (is_bold and is_large) or is_uppercase or (has_bold_child and len(text) < 150)

    def _estimate_heading_level(self, elem: Tag) -> int:
        """Estimate heading level from styling."""
        style = elem.get("style", "")

        # Check font size
        font_match = re.search(self.FONT_SIZE_PATTERN, style)
        if font_match:
            size = float(font_match.group(1))
            if size >= 16:
                return 1
            elif size >= 14:
                return 2
            elif size >= 12:
                return 3
            else:
                return 4

        # Check if uppercase (usually major heading)
        text = elem.get_text(strip=True)
        if text.isupper():
            return 2

        return 3

    def _get_element_position(self, elem: Tag) -> int:
        """Get approximate character position of element in document."""
        # Count characters of all preceding siblings and ancestors
        position = 0
        for prev in elem.find_all_previous(string=True):
            if isinstance(prev, NavigableString):
                position += len(str(prev))
        return position

    def _build_sections_from_headings(
        self, soup: BeautifulSoup, headings: list[dict], full_text: str
    ) -> list[DocumentSection]:
        """Build sections from identified headings."""
        sections = []

        for i, heading in enumerate(headings):
            # Find content between this heading and the next
            start_elem = heading["element"]

            # Determine end element
            if i + 1 < len(headings):
                end_elem = headings[i + 1]["element"]
            else:
                end_elem = None

            # Extract content between headings
            content_html, content = self._extract_section_content(start_elem, end_elem)

            if len(content) < self.min_section_chars:
                continue

            # Determine content type
            content_type = self._determine_content_type(content_html)

            # Find character positions
            char_start = full_text.find(heading["text"])
            char_end = char_start + len(content) if char_start >= 0 else 0

            section_id = hashlib.md5(
                f"{heading['text']}_{char_start}".encode()
            ).hexdigest()[:12]

            sections.append(DocumentSection(
                section_id=section_id,
                title=heading["text"],
                section_type=SectionType.UNTAGGED,
                content_type=content_type,
                char_start=max(0, char_start),
                char_end=char_end,
                content=content,
                content_html=content_html,
                char_count=len(content),
                estimated_tokens=self._count_tokens(content),
                heading_level=heading["level"],
            ))

        return sections

    def _extract_section_content(
        self, start_elem: Tag, end_elem: Optional[Tag]
    ) -> tuple[str, str]:
        """Extract HTML and text content between two elements.

        Handles table-structured Q&A content where the heading is in one table
        and the content is in sibling tables/elements.
        """
        content_parts = []
        html_parts = []

        # Collect siblings until end element
        current = start_elem.next_sibling
        while current:
            if end_elem and current == end_elem:
                break

            if isinstance(current, Tag):
                # Check if this contains the end element
                if end_elem and current.find(end_elem):
                    break
                html_parts.append(str(current))
                content_parts.append(current.get_text(separator=" "))
            elif isinstance(current, NavigableString):
                text = str(current).strip()
                if text:
                    html_parts.append(text)
                    content_parts.append(text)

            current = current.next_sibling

        # Include the heading itself
        heading_text = start_elem.get_text(separator=" ")
        content = heading_text + "\n\n" + " ".join(content_parts)
        content_html = str(start_elem) + "".join(html_parts)

        # TABLE-AWARE CONTENT EXTRACTION:
        # If content is too short and heading is inside a table cell,
        # the answer may be in sibling elements of the table (common Q&A format)
        if len(content) < self.min_section_chars:
            containing_table = start_elem.find_parent("table")
            if containing_table:
                # Try collecting siblings of the containing table
                table_siblings_parts = []
                table_siblings_html = []
                current = containing_table.next_sibling

                while current:
                    if isinstance(current, Tag):
                        # Stop if this element contains the next heading
                        if end_elem and (current == end_elem or current.find(end_elem)):
                            break
                        table_siblings_html.append(str(current))
                        table_siblings_parts.append(current.get_text(separator=" "))
                    elif isinstance(current, NavigableString):
                        text = str(current).strip()
                        if text:
                            table_siblings_html.append(text)
                            table_siblings_parts.append(text)

                    current = current.next_sibling

                # If we got more content this way, use it
                table_content = heading_text + "\n\n" + " ".join(table_siblings_parts)
                if len(table_content) > len(content):
                    content = table_content
                    content_html = str(containing_table) + "".join(table_siblings_html)

        # Clean up whitespace
        content = re.sub(r"\s+", " ", content).strip()

        return content_html, content

    def _determine_content_type(self, html: str) -> ContentType:
        """Determine the type of content in a section."""
        soup = BeautifulSoup(html, "lxml")

        has_table = soup.find("table") is not None
        has_list = soup.find(["ul", "ol"]) is not None

        if has_table and has_list:
            return ContentType.MIXED
        elif has_table:
            return ContentType.TABLE
        elif has_list:
            return ContentType.LIST
        else:
            return ContentType.TEXT

    # Minimum size for XBRL sections to be included
    # Tiny blocks (<100 chars) are typically just section headers, not content
    MIN_XBRL_SECTION_CHARS = 100

    def _create_xbrl_sections(
        self, text_blocks: list[XBRLTextBlock]
    ) -> list[DocumentSection]:
        """Create sections from XBRL text blocks.

        Filters out tiny blocks (<100 chars) which are typically just
        risk category headers like "Taxation", "SOFR Risk", etc.
        """
        sections = []
        skipped_tiny = 0

        for block in text_blocks:
            # Skip tiny XBRL blocks - they're typically just headers
            if block.char_count < self.MIN_XBRL_SECTION_CHARS:
                skipped_tiny += 1
                continue

            # Generate title from tag name
            tag_name = block.tag_name.split(":")[-1]
            title = re.sub(r"([A-Z])", r" \1", tag_name).strip()
            title = title.replace("Text Block", "").strip()

            section_id = hashlib.md5(
                f"{block.tag_name}_{block.element_id or ''}".encode()
            ).hexdigest()[:12]

            sections.append(DocumentSection(
                section_id=section_id,
                title=title,
                section_type=SectionType.XBRL_TEXT_BLOCK,
                content_type=ContentType.MIXED,
                char_start=0,  # XBRL blocks don't have absolute positions
                char_end=block.char_count,
                content=block.content,
                content_html=block.content_html,
                char_count=block.char_count,
                estimated_tokens=block.estimated_tokens,
                xbrl_tag=block.tag_name,
                xbrl_context=block.context,
            ))

        if skipped_tiny > 0:
            logger.debug(f"Skipped {skipped_tiny} tiny XBRL blocks (<{self.MIN_XBRL_SECTION_CHARS} chars)")

        return sections

    def _merge_sections(
        self,
        heading_sections: list[DocumentSection],
        xbrl_sections: list[DocumentSection],
    ) -> list[DocumentSection]:
        """Merge heading-based and XBRL sections, avoiding duplicates.

        Also merges XBRL sections with identical titles (e.g., multiple "Risk"
        sections) into single consolidated sections.
        """
        # First, merge duplicate XBRL sections with same title
        merged_xbrl = self._merge_duplicate_xbrl_sections(xbrl_sections)

        all_sections = []

        # Add merged XBRL sections (they have authoritative boundaries)
        xbrl_titles = set()
        for section in merged_xbrl:
            all_sections.append(section)
            xbrl_titles.add(section.title.lower())

        # Add heading sections that don't overlap with XBRL
        for section in heading_sections:
            # Skip if title is very similar to an XBRL section
            title_lower = section.title.lower()
            is_duplicate = any(
                self._titles_similar(title_lower, xbrl_title)
                for xbrl_title in xbrl_titles
            )
            if not is_duplicate:
                all_sections.append(section)

        # Sort by character position
        all_sections.sort(key=lambda s: s.char_start)

        return all_sections

    def _merge_duplicate_xbrl_sections(
        self,
        xbrl_sections: list[DocumentSection],
    ) -> list[DocumentSection]:
        """Merge XBRL sections with identical titles.

        Many SEC filings have multiple XBRL RiskTextBlock elements with
        the same "Risk" title. This consolidates them into single sections
        to reduce fragmentation.
        """
        from collections import defaultdict

        # Group by normalized title
        title_groups: dict[str, list[DocumentSection]] = defaultdict(list)
        for section in xbrl_sections:
            # Normalize title for grouping
            norm_title = re.sub(r"[^a-z]+", "", section.title.lower())
            title_groups[norm_title].append(section)

        merged_sections = []

        for norm_title, sections in title_groups.items():
            if len(sections) == 1:
                # No duplicates, keep as-is
                merged_sections.append(sections[0])
            else:
                # Multiple sections with same title - merge them
                merged = self._consolidate_sections(sections)
                merged_sections.append(merged)
                logger.debug(
                    f"Merged {len(sections)} '{sections[0].title}' sections "
                    f"into 1 ({merged.char_count} chars)"
                )

        return merged_sections

    def _consolidate_sections(
        self,
        sections: list[DocumentSection],
    ) -> DocumentSection:
        """Consolidate multiple sections into one.

        Concatenates content and HTML, combines metadata.
        """
        if len(sections) == 1:
            return sections[0]

        # Use first section as base
        base = sections[0]

        # Concatenate all content with separators
        all_content = []
        all_html = []
        total_chars = 0
        total_tokens = 0

        for section in sections:
            if section.content and section.content.strip():
                all_content.append(section.content.strip())
                total_chars += section.char_count
                total_tokens += section.estimated_tokens
            if section.content_html:
                all_html.append(section.content_html)

        # Join with double newline separator
        merged_content = "\n\n".join(all_content)
        merged_html = "\n".join(all_html) if all_html else None

        # Create merged section
        merged_id = hashlib.md5(
            f"merged_{base.title}_{len(sections)}".encode()
        ).hexdigest()[:12]

        return DocumentSection(
            section_id=merged_id,
            title=base.title,
            section_type=base.section_type,
            content_type=base.content_type,
            char_start=base.char_start,
            char_end=base.char_start + len(merged_content),
            content=merged_content,
            content_html=merged_html,
            char_count=len(merged_content),
            estimated_tokens=total_tokens,
            xbrl_tag=base.xbrl_tag,
            xbrl_context=base.xbrl_context,
        )

    def _titles_similar(self, title1: str, title2: str) -> bool:
        """Check if two titles refer to the same section."""
        # Normalize titles
        t1 = re.sub(r"[^a-z]+", "", title1.lower())
        t2 = re.sub(r"[^a-z]+", "", title2.lower())

        # Check if one contains the other
        return t1 in t2 or t2 in t1


def segment_document(
    file_path: str,
    xbrl_text_blocks: list[XBRLTextBlock],
    xbrl_numeric_values: list[XBRLNumericValue],
    filing_metadata: dict,
) -> DocumentMap:
    """
    Convenience function to segment a document file.

    Args:
        file_path: Path to the HTML file
        xbrl_text_blocks: Pre-extracted XBRL text blocks
        xbrl_numeric_values: Pre-extracted XBRL numeric values
        filing_metadata: Dict with cik, accession_number, form_type, filing_date

    Returns:
        DocumentMap with identified sections
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    segmenter = DocumentSegmenter()
    return segmenter.segment(content, xbrl_text_blocks, xbrl_numeric_values, filing_metadata)
