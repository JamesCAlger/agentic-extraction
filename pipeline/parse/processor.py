"""
Document Processing Pipeline.

Orchestrates iXBRL parsing, section segmentation, and chunking
to prepare documents for LLM extraction.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import (
    DocumentMap,
    ChunkedDocument,
    XBRLNumericValue,
    XBRLTextBlock,
)
from .ixbrl_parser import IXBRLParser, XBRLValueExtractor
from .document_segmenter import DocumentSegmenter
from .chunker import DocumentChunker, ChunkingConfig
from .chunking_strategies import (
    BaseChunkingStrategy,
    ChunkingStrategyType,
    get_chunking_strategy,
    ChunkingConfig as StrategyChunkingConfig,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Main processor for SEC filings.

    Orchestrates the full pipeline:
    1. Parse iXBRL to extract tagged values
    2. Segment document into sections
    3. Chunk sections for LLM extraction
    """

    def __init__(
        self,
        max_chunk_tokens: int = 500,
        overlap_tokens: int = 200,
        output_dir: Optional[Path] = None,
        chunking_strategy: Optional[BaseChunkingStrategy] = None,
    ):
        """
        Initialize processor.

        Args:
            max_chunk_tokens: Maximum tokens per chunk for LLM
            overlap_tokens: Overlap between chunks for context
            output_dir: Directory to save processed outputs
            chunking_strategy: Optional custom chunking strategy. If None, uses standard.
        """
        self.ixbrl_parser = IXBRLParser()
        self.xbrl_extractor = XBRLValueExtractor(self.ixbrl_parser)
        self.segmenter = DocumentSegmenter()

        # Use custom strategy or default to standard chunker
        if chunking_strategy is not None:
            self.chunking_strategy = chunking_strategy
        else:
            self.chunking_strategy = DocumentChunker(
                ChunkingConfig(
                    max_tokens=max_chunk_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )
        self.output_dir = output_dir

    def process_file(
        self,
        file_path: Path,
        metadata: dict,
        save_outputs: bool = True,
    ) -> tuple[DocumentMap, ChunkedDocument, dict]:
        """
        Process a single SEC filing.

        Args:
            file_path: Path to the iXBRL HTML file
            metadata: Filing metadata (cik, accession_number, form_type, filing_date)
            save_outputs: Whether to save intermediate outputs

        Returns:
            Tuple of (document_map, chunked_document, xbrl_values)
        """
        logger.info(f"Processing {file_path}")
        start_time = datetime.now()

        # Read file
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Step 1: Parse XBRL
        logger.info("  Step 1: Parsing XBRL tags...")
        xbrl_values = self.xbrl_extractor.extract_all(html_content)
        numeric_values = xbrl_values["raw_values"]
        text_blocks = xbrl_values["raw_blocks"]

        logger.info(f"    Found {len(numeric_values)} numeric values")
        logger.info(f"    Found {len(text_blocks)} text blocks")

        # Step 2: Segment document
        logger.info("  Step 2: Segmenting document...")
        doc_map = self.segmenter.segment(
            html_content,
            text_blocks,
            numeric_values,
            metadata,
        )

        sections_needing_extraction = [s for s in doc_map.sections if s.needs_extraction]
        logger.info(f"    Found {len(doc_map.sections)} total sections")
        logger.info(f"    {len(sections_needing_extraction)} sections need LLM extraction")

        # Step 3: Chunk for LLM
        logger.info("  Step 3: Chunking for LLM extraction...")
        chunked_doc = self.chunking_strategy.chunk_document(doc_map)

        logger.info(f"    Created {chunked_doc.total_chunks} chunks")
        logger.info(f"    Total tokens: {chunked_doc.total_tokens:,}")

        # Calculate processing time
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"  Completed in {elapsed:.2f}s")

        # Save outputs if requested
        if save_outputs and self.output_dir:
            self._save_outputs(file_path, doc_map, chunked_doc, xbrl_values)

        return doc_map, chunked_doc, xbrl_values

    def process_filing(
        self,
        raw_dir: Path,
        save_outputs: bool = True,
    ) -> tuple[DocumentMap, ChunkedDocument, dict]:
        """
        Process a filing from the raw data directory.

        Expects directory structure:
        raw_dir/
          primary.html
          metadata.json

        Args:
            raw_dir: Path to the filing directory
            save_outputs: Whether to save intermediate outputs

        Returns:
            Tuple of (document_map, chunked_document, xbrl_values)
        """
        # Load metadata
        metadata_path = raw_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Find primary document
        primary_path = raw_dir / "primary.html"
        if not primary_path.exists():
            raise FileNotFoundError(f"Primary document not found: {primary_path}")

        return self.process_file(primary_path, metadata, save_outputs)

    def _save_outputs(
        self,
        source_path: Path,
        doc_map: DocumentMap,
        chunked_doc: ChunkedDocument,
        xbrl_values: dict,
    ):
        """Save processed outputs to files."""
        if not self.output_dir:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create output filename base
        base_name = f"{doc_map.cik}_{doc_map.accession_number}"

        # Save XBRL values (JSON-serializable subset)
        xbrl_output = {
            "numeric_fields": xbrl_values["numeric_fields"],
            "text_blocks": {
                k: {"char_count": v["char_count"], "estimated_tokens": v["estimated_tokens"]}
                for k, v in xbrl_values["text_blocks"].items()
            },
        }
        with open(self.output_dir / f"{base_name}_xbrl.json", "w") as f:
            json.dump(xbrl_output, f, indent=2, default=str)

        # Save document map (sections summary)
        sections_summary = []
        for section in doc_map.sections:
            sections_summary.append({
                "section_id": section.section_id,
                "title": section.title,
                "section_type": section.section_type.value,
                "char_count": section.char_count,
                "estimated_tokens": section.estimated_tokens,
                "needs_extraction": section.needs_extraction,
                "target_fields": section.target_fields,
                "xbrl_tag": section.xbrl_tag,
            })

        doc_map_output = {
            "filing_id": doc_map.filing_id,
            "cik": doc_map.cik,
            "accession_number": doc_map.accession_number,
            "form_type": doc_map.form_type,
            "total_chars": doc_map.total_chars,
            "total_tokens": doc_map.total_tokens,
            "section_count": len(doc_map.sections),
            "sections": sections_summary,
        }
        with open(self.output_dir / f"{base_name}_document_map.json", "w") as f:
            json.dump(doc_map_output, f, indent=2)

        # Save chunked document summary
        chunks_summary = []
        for section in chunked_doc.chunked_sections:
            for chunk in section.chunks:
                chunks_summary.append({
                    "chunk_id": chunk.chunk_id,
                    "section_id": chunk.section_id,
                    "section_title": chunk.section_title,
                    "chunk_index": chunk.chunk_index,
                    "char_count": chunk.char_count,
                    "token_count": chunk.token_count,
                    "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                })

        chunked_output = {
            "filing_id": chunked_doc.filing_id,
            "total_sections": chunked_doc.total_sections,
            "total_chunks": chunked_doc.total_chunks,
            "total_tokens": chunked_doc.total_tokens,
            "target_fields": list(set(
                field
                for section in chunked_doc.chunked_sections
                for field in section.target_fields
            )),
            "chunks": chunks_summary,
        }
        with open(self.output_dir / f"{base_name}_chunks.json", "w") as f:
            json.dump(chunked_output, f, indent=2)

        logger.info(f"  Saved outputs to {self.output_dir}")


def process_filing(
    raw_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    max_chunk_tokens: int = 500,
    overlap_tokens: int = 200,
    chunking_strategy: Optional[BaseChunkingStrategy] = None,
) -> tuple[DocumentMap, ChunkedDocument, dict]:
    """
    Convenience function to process a filing.

    Args:
        raw_dir: Path to the filing directory (contains primary.html and metadata.json)
        output_dir: Optional directory to save processed outputs
        max_chunk_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap between consecutive chunks
        chunking_strategy: Optional custom chunking strategy. If None, uses standard.

    Returns:
        Tuple of (document_map, chunked_document, xbrl_values)
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir) if output_dir else None

    processor = DocumentProcessor(
        max_chunk_tokens=max_chunk_tokens,
        overlap_tokens=overlap_tokens,
        output_dir=output_path,
        chunking_strategy=chunking_strategy,
    )
    return processor.process_filing(raw_path)


def print_processing_summary(
    doc_map: DocumentMap,
    chunked_doc: ChunkedDocument,
    xbrl_values: dict,
):
    """Print a summary of processing results."""
    print("\n" + "=" * 60)
    print(f"PROCESSING SUMMARY: {doc_map.filing_id}")
    print("=" * 60)

    print(f"\nDocument: {doc_map.form_type} filed {doc_map.filing_date}")
    print(f"Total size: {doc_map.total_chars:,} chars, ~{doc_map.total_tokens:,} tokens")

    print("\n--- XBRL Extracted Values ---")
    for field, value in xbrl_values["numeric_fields"].items():
        if isinstance(value, dict) and "value" not in value:
            # Multi-class value
            print(f"  {field}:")
            for share_class, class_value in value.items():
                print(f"    {share_class}: {class_value['value']}")
        else:
            print(f"  {field}: {value.get('value', value)}")

    print(f"\n--- Document Sections ({len(doc_map.sections)} total) ---")
    for section in doc_map.sections[:10]:  # First 10
        status = "[EXTRACT]" if section.needs_extraction else ""
        print(f"  {section.title[:50]:50} {section.estimated_tokens:>6} tok {status}")
    if len(doc_map.sections) > 10:
        print(f"  ... and {len(doc_map.sections) - 10} more sections")

    print(f"\n--- Chunks for LLM ({chunked_doc.total_chunks} total) ---")
    print(f"  Sections to process: {chunked_doc.total_sections}")
    print(f"  Total chunks: {chunked_doc.total_chunks}")
    print(f"  Total tokens: {chunked_doc.total_tokens:,}")

    target_fields = set()
    for section in chunked_doc.chunked_sections:
        target_fields.update(section.target_fields)
    print(f"\n  Target fields to extract:")
    for field in sorted(target_fields):
        print(f"    - {field}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python processor.py <raw_filing_dir> [output_dir]")
        sys.exit(1)

    raw_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    doc_map, chunked_doc, xbrl_values = process_filing(raw_dir, output_dir)
    print_processing_summary(doc_map, chunked_doc, xbrl_values)
