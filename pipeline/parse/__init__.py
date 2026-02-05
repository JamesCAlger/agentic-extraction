"""
Document parsing and chunking package.

This package provides tools for parsing SEC iXBRL filings,
segmenting documents into sections, and chunking for LLM extraction.
"""

from .models import (
    # Enums
    ConfidenceLevel,
    SectionType,
    ContentType,
    # XBRL models
    XBRLContext,
    XBRLNumericValue,
    XBRLTextBlock,
    # Document models
    DocumentSection,
    DocumentMap,
    # Chunking models
    Chunk,
    ChunkedSection,
    ChunkedDocument,
    # Extraction models
    ExtractedValue,
    SectionExtractions,
    DocumentExtractions,
)

from .ixbrl_parser import (
    FundType,
    IXBRLParser,
    XBRLValueExtractor,
    XBRLExtractionResult,
    extract_xbrl_values,
)

from .document_segmenter import (
    DocumentSegmenter,
    SectionFieldMapping,
    segment_document,
)

from .chunker import (
    ChunkingConfig,
    TextChunker,
    HTMLChunker,
    DocumentChunker,
    chunk_document,
)

from .processor import (
    DocumentProcessor,
    process_filing,
    print_processing_summary,
)

__all__ = [
    # Enums
    "ConfidenceLevel",
    "SectionType",
    "ContentType",
    "FundType",
    # XBRL models
    "XBRLContext",
    "XBRLNumericValue",
    "XBRLTextBlock",
    # Document models
    "DocumentSection",
    "DocumentMap",
    # Chunking models
    "Chunk",
    "ChunkedSection",
    "ChunkedDocument",
    # Extraction models
    "ExtractedValue",
    "SectionExtractions",
    "DocumentExtractions",
    # Parsers
    "IXBRLParser",
    "XBRLValueExtractor",
    "XBRLExtractionResult",
    "extract_xbrl_values",
    # Segmenter
    "DocumentSegmenter",
    "SectionFieldMapping",
    "segment_document",
    # Chunker
    "ChunkingConfig",
    "TextChunker",
    "HTMLChunker",
    "DocumentChunker",
    "chunk_document",
    # Processor
    "DocumentProcessor",
    "process_filing",
    "print_processing_summary",
]
