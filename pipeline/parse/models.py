"""
Pydantic models for document parsing and chunking.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    """Confidence level for extracted values."""
    EXPLICIT = "explicit"      # Value directly stated in text
    INFERRED = "inferred"      # Value derived/calculated from context
    NOT_FOUND = "not_found"    # Value not present in document
    CONFLICT = "conflict"      # Multiple conflicting values found


class SectionType(str, Enum):
    """Type of document section."""
    XBRL_NUMERIC = "xbrl_numeric"      # Tagged numeric value
    XBRL_TEXT_BLOCK = "xbrl_text_block" # Tagged text block
    UNTAGGED = "untagged"               # Identified by heading/structure


class ContentType(str, Enum):
    """Type of content within a section."""
    TEXT = "text"
    TABLE = "table"
    LIST = "list"
    MIXED = "mixed"


# =============================================================================
# XBRL Extraction Models
# =============================================================================

class XBRLContext(BaseModel):
    """XBRL context information (e.g., share class, time period)."""
    context_id: str
    share_class: Optional[str] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    instant: Optional[str] = None


class XBRLNumericValue(BaseModel):
    """Extracted XBRL numeric value."""
    tag_name: str                    # e.g., "cef:ManagementFeesPercent"
    value: Decimal
    unit: Optional[str] = None       # e.g., "pure", "USD"
    decimals: Optional[int] = None
    scale: Optional[int] = None      # e.g., -2 for percentage
    context: XBRLContext
    element_id: Optional[str] = None
    raw_text: str                    # Original text in document


class XBRLTextBlock(BaseModel):
    """Extracted XBRL text block."""
    tag_name: str                    # e.g., "cef:RiskTextBlock"
    content: str                     # Full text content (HTML stripped)
    content_html: str                # Original HTML content
    context: XBRLContext
    element_id: Optional[str] = None
    char_count: int
    estimated_tokens: int


# =============================================================================
# Document Structure Models
# =============================================================================

class DocumentSection(BaseModel):
    """A section of the document identified by heading or XBRL tag."""
    section_id: str
    title: str
    section_type: SectionType
    content_type: ContentType = ContentType.MIXED

    # Location in document
    char_start: int
    char_end: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    # Content
    content: str                     # Plain text content
    content_html: str                # Original HTML

    # Metrics
    char_count: int
    estimated_tokens: int

    # XBRL specific
    xbrl_tag: Optional[str] = None
    xbrl_context: Optional[XBRLContext] = None

    # Hierarchy
    heading_level: int = 0           # 0=root, 1=H1, 2=H2, etc.
    parent_section_id: Optional[str] = None

    # Processing flags
    needs_extraction: bool = False
    target_fields: list[str] = Field(default_factory=list)


class DocumentMap(BaseModel):
    """Complete map of document structure."""
    filing_id: str
    cik: str
    accession_number: str
    form_type: str
    filing_date: str

    # Document metadata
    total_chars: int
    total_tokens: int
    page_count: Optional[int] = None

    # Extracted structures
    sections: list[DocumentSection] = Field(default_factory=list)
    xbrl_numeric_values: list[XBRLNumericValue] = Field(default_factory=list)
    xbrl_text_blocks: list[XBRLTextBlock] = Field(default_factory=list)

    # Processing metadata
    parsed_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Chunking Models
# =============================================================================

class Chunk(BaseModel):
    """A chunk of text ready for LLM processing."""
    chunk_id: str
    section_id: str
    chunk_index: int                 # Order within section

    # Content
    content: str                     # Plain text
    content_html: Optional[str] = None

    # Location
    char_start: int                  # Relative to section
    char_end: int
    global_char_start: int           # Relative to full document
    global_char_end: int

    # Metrics
    char_count: int
    token_count: int

    # Context
    section_title: str
    preceding_context: Optional[str] = None  # Overlap from previous chunk

    # Processing
    content_hash: str                # For deduplication


class ChunkedSection(BaseModel):
    """A section that has been chunked for processing."""
    section_id: str
    section_title: str
    section_type: SectionType
    target_fields: list[str]

    chunks: list[Chunk]
    total_chunks: int
    total_tokens: int


class ChunkedDocument(BaseModel):
    """Document with all sections chunked and ready for extraction."""
    filing_id: str
    cik: str
    accession_number: str

    # Document-level XBRL values (no chunking needed)
    xbrl_numeric_values: list[XBRLNumericValue] = Field(default_factory=list)

    # Chunked sections requiring LLM extraction
    chunked_sections: list[ChunkedSection] = Field(default_factory=list)

    # Summary
    total_sections: int
    total_chunks: int
    total_tokens: int

    # Processing metadata
    chunked_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Extraction Models
# =============================================================================

class ExtractedValue(BaseModel):
    """A value extracted from a chunk."""
    field_name: str
    value: Optional[str | Decimal | int | float | bool | list] = None
    value_type: str = "string"       # string, decimal, integer, boolean, list

    # Citation
    chunk_id: str
    evidence_quote: Optional[str] = None

    # Confidence
    confidence: ConfidenceLevel
    confidence_score: Optional[float] = None  # 0.0 - 1.0

    # For conflict resolution
    alternatives: list[str] = Field(default_factory=list)
    needs_review: bool = False


class SectionExtractions(BaseModel):
    """All extractions from a section."""
    section_id: str
    section_title: str
    extractions: list[ExtractedValue]

    # Processing metadata
    chunks_processed: int
    extraction_time_ms: int


class DocumentExtractions(BaseModel):
    """All extractions from a document."""
    filing_id: str
    cik: str
    accession_number: str

    # XBRL values (deterministic)
    xbrl_extractions: list[XBRLNumericValue] = Field(default_factory=list)

    # LLM extractions
    section_extractions: list[SectionExtractions] = Field(default_factory=list)

    # Summary
    total_fields_extracted: int
    fields_needing_review: int

    # Processing metadata
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: int
