"""
Intelligent Chunking for LLM extraction.

Splits document sections into appropriately-sized chunks while
preserving semantic boundaries (tables, paragraphs, lists).
"""

import hashlib
import re
from typing import Optional
from bs4 import BeautifulSoup
import tiktoken

from .models import (
    DocumentSection,
    DocumentMap,
    Chunk,
    ChunkedSection,
    ChunkedDocument,
    ContentType,
    SectionType,
)


class ChunkingConfig:
    """Configuration for chunking behavior."""

    def __init__(
        self,
        max_tokens: int = 500,
        overlap_tokens: int = 200,
        min_chunk_tokens: int = 100,
        preserve_tables: bool = True,
        preserve_lists: bool = True,
        chunk_all_sections: bool = True,
    ):
        """
        Initialize chunking configuration.

        Args:
            max_tokens: Maximum tokens per chunk (leave room for prompt/response)
            overlap_tokens: Tokens of overlap between consecutive chunks
            min_chunk_tokens: Minimum tokens for a valid chunk
            preserve_tables: Keep tables intact (don't split)
            preserve_lists: Keep lists intact (don't split)
            chunk_all_sections: If True, chunk ALL sections regardless of needs_extraction.
                              If False, only chunk sections where needs_extraction=True.
                              Default is True for robust extraction across 200+ funds.
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.preserve_tables = preserve_tables
        self.preserve_lists = preserve_lists
        self.chunk_all_sections = chunk_all_sections


class TextChunker:
    """
    Chunks text content into appropriately-sized pieces.

    Uses a hierarchy of split points:
    1. Section breaks (double newlines)
    2. Paragraph breaks
    3. Sentence breaks
    4. Word breaks (last resort)
    """

    # Split point patterns in priority order
    SPLIT_PATTERNS = [
        (r"\n\n+", "paragraph"),           # Double newline / paragraph break
        (r"\n", "line"),                    # Single newline
        (r"(?<=[.!?])\s+", "sentence"),     # Sentence boundary
        (r"\s+", "word"),                   # Word boundary
    ]

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize chunker with configuration."""
        self.config = config or ChunkingConfig()
        self._tokenizer = None
        try:
            self._tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        except Exception:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        return len(text) // 4

    def chunk_text(self, text: str) -> list[dict]:
        """
        Chunk plain text into pieces.

        Returns list of dicts with:
        - content: The chunk text
        - char_start: Start position in original text
        - char_end: End position in original text
        - token_count: Number of tokens
        """
        if self.count_tokens(text) <= self.config.max_tokens:
            return [{
                "content": text,
                "char_start": 0,
                "char_end": len(text),
                "token_count": self.count_tokens(text),
            }]

        chunks = []
        remaining = text
        offset = 0

        while remaining:
            # Find the best split point
            chunk_text = self._find_chunk(remaining)

            if not chunk_text:
                break

            token_count = self.count_tokens(chunk_text)

            chunks.append({
                "content": chunk_text,
                "char_start": offset,
                "char_end": offset + len(chunk_text),
                "token_count": token_count,
            })

            # Move past the chunk, accounting for overlap
            advance = len(chunk_text)
            if self.config.overlap_tokens > 0 and len(remaining) > advance:
                # Calculate overlap in characters (rough estimate)
                overlap_chars = min(
                    self.config.overlap_tokens * 4,
                    len(chunk_text) // 2
                )
                advance = max(1, advance - overlap_chars)

            remaining = remaining[advance:]
            offset += advance

        return chunks

    def _find_chunk(self, text: str) -> str:
        """Find the best chunk from the start of text."""
        # Start with max size
        max_chars = self.config.max_tokens * 4  # Rough estimate

        if len(text) <= max_chars:
            return text

        # Try each split pattern
        for pattern, split_type in self.SPLIT_PATTERNS:
            chunk = self._find_split_point(text, pattern, max_chars)
            if chunk and self.count_tokens(chunk) <= self.config.max_tokens:
                return chunk

        # Last resort: hard cut at character limit
        return text[:max_chars]

    def _find_split_point(self, text: str, pattern: str, max_chars: int) -> Optional[str]:
        """Find the last occurrence of pattern before max_chars."""
        # Search in the relevant portion
        search_text = text[:max_chars + 100]

        matches = list(re.finditer(pattern, search_text))
        if not matches:
            return None

        # Find the last match that keeps us under the limit
        for match in reversed(matches):
            end_pos = match.start()
            candidate = text[:end_pos].strip()

            if candidate and self.count_tokens(candidate) <= self.config.max_tokens:
                return candidate

        return None


class HTMLChunker:
    """
    Chunks HTML content while preserving structure.

    Keeps tables and lists intact where possible.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize HTML chunker."""
        self.config = config or ChunkingConfig()
        self.text_chunker = TextChunker(config)

    def chunk_html(self, html: str) -> list[dict]:
        """
        Chunk HTML content into pieces.

        Returns list of dicts with:
        - content: Plain text content
        - content_html: HTML content (if preserved)
        - char_start: Start position in plain text
        - char_end: End position in plain text
        - token_count: Number of tokens
        - is_atomic: Whether this is an atomic unit (table/list)
        """
        soup = BeautifulSoup(html, "lxml")
        plain_text = soup.get_text(separator=" ")
        plain_text = re.sub(r"\s+", " ", plain_text).strip()

        # First, identify atomic units (tables, lists)
        atomic_units = self._extract_atomic_units(soup)

        # If no atomic units or they all fit, use simple text chunking
        if not atomic_units:
            text_chunks = self.text_chunker.chunk_text(plain_text)
            for chunk in text_chunks:
                chunk["content_html"] = None
                chunk["is_atomic"] = False
            return text_chunks

        # Build chunks around atomic units
        return self._chunk_with_atomic_units(soup, plain_text, atomic_units)

    def _extract_atomic_units(self, soup: BeautifulSoup) -> list[dict]:
        """Extract tables and lists that should be kept intact."""
        units = []

        if self.config.preserve_tables:
            for table in soup.find_all("table"):
                text = table.get_text(separator=" ")
                text = re.sub(r"\s+", " ", text).strip()
                token_count = self.text_chunker.count_tokens(text)

                # Only keep as atomic if it fits in a chunk
                if token_count <= self.config.max_tokens:
                    units.append({
                        "type": "table",
                        "element": table,
                        "text": text,
                        "html": str(table),
                        "token_count": token_count,
                    })

        if self.config.preserve_lists:
            for lst in soup.find_all(["ul", "ol"]):
                text = lst.get_text(separator=" ")
                text = re.sub(r"\s+", " ", text).strip()
                token_count = self.text_chunker.count_tokens(text)

                if token_count <= self.config.max_tokens:
                    units.append({
                        "type": "list",
                        "element": lst,
                        "text": text,
                        "html": str(lst),
                        "token_count": token_count,
                    })

        return units

    def _chunk_with_atomic_units(
        self,
        soup: BeautifulSoup,
        plain_text: str,
        atomic_units: list[dict],
    ) -> list[dict]:
        """Build chunks while preserving atomic units."""
        chunks = []

        # Sort atomic units by position in text
        for unit in atomic_units:
            pos = plain_text.find(unit["text"][:50])  # Find approximate position
            unit["position"] = pos if pos >= 0 else float("inf")

        atomic_units.sort(key=lambda u: u["position"])

        # Process text around atomic units
        current_pos = 0

        for unit in atomic_units:
            # Chunk text before this atomic unit
            if unit["position"] > current_pos:
                before_text = plain_text[current_pos:unit["position"]]
                if before_text.strip():
                    text_chunks = self.text_chunker.chunk_text(before_text.strip())
                    for chunk in text_chunks:
                        chunk["char_start"] += current_pos
                        chunk["char_end"] += current_pos
                        chunk["content_html"] = None
                        chunk["is_atomic"] = False
                        chunks.append(chunk)

            # Add atomic unit as its own chunk
            chunks.append({
                "content": unit["text"],
                "content_html": unit["html"],
                "char_start": unit["position"],
                "char_end": unit["position"] + len(unit["text"]),
                "token_count": unit["token_count"],
                "is_atomic": True,
            })

            current_pos = unit["position"] + len(unit["text"])

        # Chunk remaining text after last atomic unit
        if current_pos < len(plain_text):
            remaining = plain_text[current_pos:].strip()
            if remaining:
                text_chunks = self.text_chunker.chunk_text(remaining)
                for chunk in text_chunks:
                    chunk["char_start"] += current_pos
                    chunk["char_end"] += current_pos
                    chunk["content_html"] = None
                    chunk["is_atomic"] = False
                    chunks.append(chunk)

        return chunks


class DocumentChunker:
    """
    High-level chunker for document sections.

    Processes a DocumentMap and produces a ChunkedDocument
    ready for LLM extraction.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize document chunker."""
        self.config = config or ChunkingConfig()
        self.html_chunker = HTMLChunker(config)
        self.text_chunker = TextChunker(config)

    def chunk_document(self, doc_map: DocumentMap) -> ChunkedDocument:
        """
        Chunk document sections for LLM extraction.

        By default (chunk_all_sections=True), chunks ALL sections regardless
        of needs_extraction flag. This ensures Tier 3/4 agentic search has
        access to the full document content.

        Args:
            doc_map: Document map with identified sections

        Returns:
            ChunkedDocument ready for LLM processing
        """
        chunked_sections = []
        total_chunks = 0
        total_tokens = 0

        for section in doc_map.sections:
            # Skip sections only if chunk_all_sections=False AND needs_extraction=False
            if not self.config.chunk_all_sections and not section.needs_extraction:
                continue

            chunked = self._chunk_section(section, doc_map)
            if chunked.chunks:
                chunked_sections.append(chunked)
                total_chunks += chunked.total_chunks
                total_tokens += chunked.total_tokens

        return ChunkedDocument(
            filing_id=doc_map.filing_id,
            cik=doc_map.cik,
            accession_number=doc_map.accession_number,
            xbrl_numeric_values=doc_map.xbrl_numeric_values,
            chunked_sections=chunked_sections,
            total_sections=len(chunked_sections),
            total_chunks=total_chunks,
            total_tokens=total_tokens,
        )

    def _chunk_section(
        self, section: DocumentSection, doc_map: DocumentMap
    ) -> ChunkedSection:
        """Chunk a single section."""
        # Choose chunking strategy based on content type
        if section.content_type in [ContentType.TABLE, ContentType.MIXED]:
            raw_chunks = self.html_chunker.chunk_html(section.content_html)
        else:
            raw_chunks = self.text_chunker.chunk_text(section.content)
            for chunk in raw_chunks:
                chunk["content_html"] = None
                chunk["is_atomic"] = False

        # Convert to Chunk models
        chunks = []
        section_tokens = 0

        for i, raw in enumerate(raw_chunks):
            chunk_id = hashlib.md5(
                f"{section.section_id}_{i}_{raw['content'][:50]}".encode()
            ).hexdigest()[:16]

            # Calculate preceding context for non-first chunks
            preceding_context = None
            if i > 0 and self.config.overlap_tokens > 0:
                prev_content = raw_chunks[i - 1]["content"]
                # Get last N characters as context
                context_chars = self.config.overlap_tokens * 4
                preceding_context = prev_content[-context_chars:] if len(prev_content) > context_chars else prev_content

            chunk = Chunk(
                chunk_id=chunk_id,
                section_id=section.section_id,
                chunk_index=i,
                content=raw["content"],
                content_html=raw.get("content_html"),
                char_start=raw["char_start"],
                char_end=raw["char_end"],
                global_char_start=section.char_start + raw["char_start"],
                global_char_end=section.char_start + raw["char_end"],
                char_count=len(raw["content"]),
                token_count=raw["token_count"],
                section_title=section.title,
                preceding_context=preceding_context,
                content_hash=hashlib.md5(raw["content"].encode()).hexdigest(),
            )
            chunks.append(chunk)
            section_tokens += raw["token_count"]

        return ChunkedSection(
            section_id=section.section_id,
            section_title=section.title,
            section_type=section.section_type,
            target_fields=section.target_fields,
            chunks=chunks,
            total_chunks=len(chunks),
            total_tokens=section_tokens,
        )


def chunk_document(
    doc_map: DocumentMap,
    max_tokens: int = 4000,
    overlap_tokens: int = 200,
) -> ChunkedDocument:
    """
    Convenience function to chunk a document.

    Args:
        doc_map: Document map with identified sections
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap between chunks

    Returns:
        ChunkedDocument ready for LLM processing
    """
    config = ChunkingConfig(
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    chunker = DocumentChunker(config)
    return chunker.chunk_document(doc_map)
