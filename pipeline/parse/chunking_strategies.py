"""
Chunking Strategies for Retrieval Experiments.

This module implements different chunking approaches to test their impact
on retrieval and extraction accuracy:

1. Standard: Current behavior (fill to max_tokens, paragraph-aware splits)
2. ParagraphAtomic: One paragraph per chunk, no combining
3. Hierarchical: Standard + section-level summaries for retrieval
4. RAPTOR: Bottom-up clustering with recursive summarization

Each strategy produces a ChunkedDocument that can be used by the extraction pipeline.
"""

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np

from .models import (
    DocumentSection,
    DocumentMap,
    Chunk,
    ChunkedSection,
    ChunkedDocument,
    ContentType,
)
from .chunker import ChunkingConfig, TextChunker, HTMLChunker

logger = logging.getLogger(__name__)


class ChunkingStrategyType(Enum):
    """Available chunking strategies."""
    STANDARD = "standard"
    PARAGRAPH_ATOMIC = "paragraph_atomic"
    HIERARCHICAL = "hierarchical"
    RAPTOR = "raptor"


@dataclass
class SectionSummary:
    """Summary of a section for hierarchical retrieval."""
    section_id: str
    section_title: str
    summary: str
    entities: list[dict] = field(default_factory=list)
    embedding: Optional[list[float]] = None


class ChunkedDocumentWithSummaries(ChunkedDocument):
    """Extended ChunkedDocument with section summaries for hierarchical retrieval."""
    # Note: Using list default directly is safe in Pydantic models (not mutable default issue)
    section_summaries: list[dict] = []  # Store as dicts, not SectionSummary to avoid pydantic issues
    document_summary: Optional[str] = None


class BaseChunkingStrategy(ABC):
    """Base class for chunking strategies."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.text_chunker = TextChunker(self.config)
        self.html_chunker = HTMLChunker(self.config)

    @abstractmethod
    def chunk_document(self, doc_map: DocumentMap) -> ChunkedDocument:
        """Chunk a document according to this strategy."""
        pass

    def _create_chunk(
        self,
        content: str,
        section: DocumentSection,
        chunk_index: int,
        char_start: int,
        char_end: int,
        content_html: Optional[str] = None,
        preceding_context: Optional[str] = None,
    ) -> Chunk:
        """Create a Chunk object with consistent ID generation."""
        chunk_id = hashlib.md5(
            f"{section.section_id}_{chunk_index}_{content[:50]}".encode()
        ).hexdigest()[:16]

        return Chunk(
            chunk_id=chunk_id,
            section_id=section.section_id,
            chunk_index=chunk_index,
            content=content,
            content_html=content_html,
            char_start=char_start,
            char_end=char_end,
            global_char_start=section.char_start + char_start,
            global_char_end=section.char_start + char_end,
            char_count=len(content),
            token_count=self.text_chunker.count_tokens(content),
            section_title=section.title,
            preceding_context=preceding_context,
            content_hash=hashlib.md5(content.encode()).hexdigest(),
        )


class StandardChunkingStrategy(BaseChunkingStrategy):
    """
    Standard chunking: fill to max_tokens with paragraph-aware splits.

    This is the current default behavior.
    """

    def chunk_document(self, doc_map: DocumentMap) -> ChunkedDocument:
        """Chunk using standard approach.

        By default (chunk_all_sections=True), chunks ALL sections regardless
        of needs_extraction flag for robust extraction across 200+ funds.
        """
        chunked_sections = []
        total_chunks = 0
        total_tokens = 0

        for section in doc_map.sections:
            # Skip sections only if chunk_all_sections=False AND needs_extraction=False
            if not self.config.chunk_all_sections and not section.needs_extraction:
                continue

            chunked = self._chunk_section(section)
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

    def _chunk_section(self, section: DocumentSection) -> ChunkedSection:
        """Chunk a single section using standard approach."""
        if section.content_type in [ContentType.TABLE, ContentType.MIXED]:
            raw_chunks = self.html_chunker.chunk_html(section.content_html)
        else:
            raw_chunks = self.text_chunker.chunk_text(section.content)
            for chunk in raw_chunks:
                chunk["content_html"] = None
                chunk["is_atomic"] = False

        chunks = []
        section_tokens = 0

        for i, raw in enumerate(raw_chunks):
            preceding_context = None
            if i > 0 and self.config.overlap_tokens > 0:
                prev_content = raw_chunks[i - 1]["content"]
                context_chars = self.config.overlap_tokens * 4
                preceding_context = prev_content[-context_chars:] if len(prev_content) > context_chars else prev_content

            chunk = self._create_chunk(
                content=raw["content"],
                section=section,
                chunk_index=i,
                char_start=raw["char_start"],
                char_end=raw["char_end"],
                content_html=raw.get("content_html"),
                preceding_context=preceding_context,
            )
            chunks.append(chunk)
            section_tokens += chunk.token_count

        return ChunkedSection(
            section_id=section.section_id,
            section_title=section.title,
            section_type=section.section_type,
            target_fields=section.target_fields,
            chunks=chunks,
            total_chunks=len(chunks),
            total_tokens=section_tokens,
        )


class ParagraphAtomicChunkingStrategy(BaseChunkingStrategy):
    """
    Paragraph-atomic chunking: one paragraph per chunk, no combining.

    Each paragraph becomes its own chunk regardless of size.
    Very small paragraphs (< min_tokens) are merged with the next.
    """

    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        min_paragraph_tokens: int = 30,
    ):
        super().__init__(config)
        self.min_paragraph_tokens = min_paragraph_tokens

    def chunk_document(self, doc_map: DocumentMap) -> ChunkedDocument:
        """Chunk with one paragraph per chunk.

        By default (chunk_all_sections=True), chunks ALL sections regardless
        of needs_extraction flag for robust extraction across 200+ funds.
        """
        chunked_sections = []
        total_chunks = 0
        total_tokens = 0

        for section in doc_map.sections:
            # Skip sections only if chunk_all_sections=False AND needs_extraction=False
            if not self.config.chunk_all_sections and not section.needs_extraction:
                continue

            chunked = self._chunk_section_by_paragraphs(section)
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

    def _chunk_section_by_paragraphs(self, section: DocumentSection) -> ChunkedSection:
        """Split section into paragraph-atomic chunks."""
        # Split on double newlines (paragraphs)
        text = section.content
        paragraphs = re.split(r'\n\n+', text)

        # Clean up and filter empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Merge very small paragraphs with the next
        merged_paragraphs = []
        buffer = ""
        buffer_start = 0
        current_pos = 0

        for i, para in enumerate(paragraphs):
            para_tokens = self.text_chunker.count_tokens(para)

            if para_tokens < self.min_paragraph_tokens and i < len(paragraphs) - 1:
                # Too small, buffer it
                if not buffer:
                    buffer_start = current_pos
                buffer += para + "\n\n"
            else:
                # Normal paragraph or last one
                if buffer:
                    # Flush buffer with this paragraph
                    combined = buffer + para
                    merged_paragraphs.append({
                        "content": combined.strip(),
                        "char_start": buffer_start,
                        "char_end": current_pos + len(para),
                    })
                    buffer = ""
                else:
                    merged_paragraphs.append({
                        "content": para,
                        "char_start": current_pos,
                        "char_end": current_pos + len(para),
                    })

            current_pos += len(para) + 2  # +2 for \n\n

        # Handle remaining buffer
        if buffer:
            merged_paragraphs.append({
                "content": buffer.strip(),
                "char_start": buffer_start,
                "char_end": current_pos,
            })

        # Create chunks
        chunks = []
        section_tokens = 0

        for i, para_info in enumerate(merged_paragraphs):
            # If paragraph is too large, fall back to standard chunking
            para_tokens = self.text_chunker.count_tokens(para_info["content"])

            if para_tokens > self.config.max_tokens:
                # Split large paragraph using standard chunker
                sub_chunks = self.text_chunker.chunk_text(para_info["content"])
                for j, sub in enumerate(sub_chunks):
                    chunk = self._create_chunk(
                        content=sub["content"],
                        section=section,
                        chunk_index=len(chunks),
                        char_start=para_info["char_start"] + sub["char_start"],
                        char_end=para_info["char_start"] + sub["char_end"],
                    )
                    chunks.append(chunk)
                    section_tokens += chunk.token_count
            else:
                chunk = self._create_chunk(
                    content=para_info["content"],
                    section=section,
                    chunk_index=i,
                    char_start=para_info["char_start"],
                    char_end=para_info["char_end"],
                )
                chunks.append(chunk)
                section_tokens += chunk.token_count

        return ChunkedSection(
            section_id=section.section_id,
            section_title=section.title,
            section_type=section.section_type,
            target_fields=section.target_fields,
            chunks=chunks,
            total_chunks=len(chunks),
            total_tokens=section_tokens,
        )


class HierarchicalChunkingStrategy(BaseChunkingStrategy):
    """
    Hierarchical chunking: standard chunks + section-level summaries.

    Adds an LLM-generated summary for each section that can be used
    for retrieval. The summary is designed to be more queryable than
    the raw chunks.
    """

    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        llm_client=None,
        summary_model: str = "gpt-4o-mini",
    ):
        super().__init__(config)
        self.llm_client = llm_client
        self.summary_model = summary_model
        self._standard_strategy = StandardChunkingStrategy(config)

    def chunk_document(self, doc_map: DocumentMap) -> ChunkedDocumentWithSummaries:
        """Chunk with standard approach, then add section summaries."""
        # First, do standard chunking
        base_doc = self._standard_strategy.chunk_document(doc_map)

        # Generate section summaries
        section_summaries = []
        for chunked_section in base_doc.chunked_sections:
            summary = self._generate_section_summary(chunked_section)
            if summary:
                section_summaries.append(summary)

        # Generate document summary from section summaries
        doc_summary = self._generate_document_summary(section_summaries)

        # Convert SectionSummary dataclasses to dicts for Pydantic
        summaries_as_dicts = [
            {
                "section_id": s.section_id,
                "section_title": s.section_title,
                "summary": s.summary,
                "entities": s.entities,
                "embedding": s.embedding,
            }
            for s in section_summaries
        ]

        return ChunkedDocumentWithSummaries(
            filing_id=base_doc.filing_id,
            cik=base_doc.cik,
            accession_number=base_doc.accession_number,
            xbrl_numeric_values=base_doc.xbrl_numeric_values,
            chunked_sections=base_doc.chunked_sections,
            total_sections=base_doc.total_sections,
            total_chunks=base_doc.total_chunks,
            total_tokens=base_doc.total_tokens,
            section_summaries=summaries_as_dicts,
            document_summary=doc_summary,
        )

    def _generate_section_summary(self, section: ChunkedSection) -> Optional[SectionSummary]:
        """Generate a summary for a section using LLM."""
        if not self.llm_client:
            logger.warning("No LLM client provided, skipping summary generation")
            return None

        # Concatenate chunk content (up to token limit)
        combined_text = ""
        for chunk in section.chunks[:10]:  # Limit to first 10 chunks
            combined_text += chunk.content + "\n\n"
            if len(combined_text) > 8000:  # ~2000 tokens
                break

        prompt = f"""You are summarizing a section from an SEC fund filing.
Section title: "{section.section_title}"
Target fields: {', '.join(section.target_fields) if section.target_fields else 'general'}

RULES:
1. Only state facts EXPLICITLY written in the text below
2. Quote exact dollar amounts, percentages, and class names
3. List any share classes mentioned with their key attributes
4. Do NOT infer or calculate values
5. Keep summary to 2-3 sentences max

Section content:
{combined_text[:6000]}

Output a factual summary listing the key data points found:"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.summary_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )
            summary_text = response.choices[0].message.content.strip()

            # Extract entities (simple pattern matching)
            entities = self._extract_entities(summary_text)

            return SectionSummary(
                section_id=section.section_id,
                section_title=section.section_title,
                summary=summary_text,
                entities=entities,
            )
        except Exception as e:
            logger.error(f"Failed to generate summary for {section.section_title}: {e}")
            return None

    def _generate_document_summary(self, section_summaries: list[SectionSummary]) -> Optional[str]:
        """Generate a document-level summary from section summaries."""
        if not self.llm_client or not section_summaries:
            return None

        summaries_text = "\n\n".join([
            f"**{s.section_title}**: {s.summary}"
            for s in section_summaries
        ])

        prompt = f"""Summarize this SEC fund filing in 2-3 sentences.
Include: fund type, share classes, key fees, and any notable terms.

Section summaries:
{summaries_text}

Document summary:"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.summary_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate document summary: {e}")
            return None

    def _extract_entities(self, text: str) -> list[dict]:
        """Extract entities from summary text."""
        entities = []

        # Dollar amounts
        for match in re.finditer(r'\$[\d,]+(?:\.\d+)?', text):
            entities.append({"type": "amount", "value": match.group()})

        # Percentages
        for match in re.finditer(r'[\d.]+%', text):
            entities.append({"type": "percentage", "value": match.group()})

        # Share classes
        for match in re.finditer(r'Class\s+[A-Z]\b', text):
            entities.append({"type": "share_class", "value": match.group()})

        return entities


class RAPTORChunkingStrategy(BaseChunkingStrategy):
    """
    RAPTOR-style chunking: bottom-up clustering with recursive summarization.

    1. Start with paragraph-atomic chunks
    2. Embed all chunks
    3. Cluster similar chunks
    4. Summarize each cluster
    5. Repeat recursively until single root
    """

    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        llm_client=None,
        embedding_model=None,
        summary_model: str = "gpt-4o-mini",
        cluster_size: int = 5,
        max_levels: int = 3,
    ):
        super().__init__(config)
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.summary_model = summary_model
        self.cluster_size = cluster_size
        self.max_levels = max_levels
        self._paragraph_strategy = ParagraphAtomicChunkingStrategy(config)

    def chunk_document(self, doc_map: DocumentMap) -> ChunkedDocumentWithSummaries:
        """Chunk using RAPTOR approach."""
        # Start with paragraph-atomic chunks
        base_doc = self._paragraph_strategy.chunk_document(doc_map)

        if not self.embedding_model or not self.llm_client:
            logger.warning("No embedding/LLM client, returning base chunks without RAPTOR tree")
            return ChunkedDocumentWithSummaries(
                filing_id=base_doc.filing_id,
                cik=base_doc.cik,
                accession_number=base_doc.accession_number,
                xbrl_numeric_values=base_doc.xbrl_numeric_values,
                chunked_sections=base_doc.chunked_sections,
                total_sections=base_doc.total_sections,
                total_chunks=base_doc.total_chunks,
                total_tokens=base_doc.total_tokens,
                section_summaries=[],
                document_summary=None,
            )

        # Collect all chunks across sections
        all_chunks = []
        for section in base_doc.chunked_sections:
            all_chunks.extend(section.chunks)

        # Build RAPTOR tree
        tree_summaries = self._build_raptor_tree(all_chunks)

        # Convert to section summaries format (as dicts for Pydantic)
        section_summaries = [
            {
                "section_id": f"raptor_level_{s['level']}_cluster_{s['cluster_id']}",
                "section_title": f"Cluster {s['cluster_id']} (Level {s['level']})",
                "summary": s['summary'],
                "entities": s.get('entities', []),
                "embedding": None,
            }
            for s in tree_summaries
        ]

        # Document summary is the root node
        doc_summary = tree_summaries[-1]['summary'] if tree_summaries else None

        return ChunkedDocumentWithSummaries(
            filing_id=base_doc.filing_id,
            cik=base_doc.cik,
            accession_number=base_doc.accession_number,
            xbrl_numeric_values=base_doc.xbrl_numeric_values,
            chunked_sections=base_doc.chunked_sections,
            total_sections=base_doc.total_sections,
            total_chunks=base_doc.total_chunks,
            total_tokens=base_doc.total_tokens,
            section_summaries=section_summaries,
            document_summary=doc_summary,
        )

    def _build_raptor_tree(self, chunks: list[Chunk]) -> list[dict]:
        """Build RAPTOR tree through recursive clustering and summarization."""
        all_summaries = []

        # Level 0: embed all chunks
        current_level_texts = [c.content for c in chunks]
        current_level_embeddings = self._embed_texts(current_level_texts)

        for level in range(self.max_levels):
            if len(current_level_texts) <= 1:
                break

            # Cluster current level
            clusters = self._cluster_texts(
                current_level_texts,
                current_level_embeddings,
            )

            # Summarize each cluster
            next_level_texts = []
            next_level_embeddings = []

            for cluster_id, cluster_texts in enumerate(clusters):
                summary = self._summarize_cluster(cluster_texts)

                all_summaries.append({
                    'level': level,
                    'cluster_id': cluster_id,
                    'summary': summary,
                    'source_count': len(cluster_texts),
                })

                next_level_texts.append(summary)

            # Embed summaries for next level
            if next_level_texts:
                next_level_embeddings = self._embed_texts(next_level_texts)

            current_level_texts = next_level_texts
            current_level_embeddings = next_level_embeddings

            logger.info(f"RAPTOR level {level}: {len(clusters)} clusters")

        return all_summaries

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the embedding model."""
        try:
            embeddings = self.embedding_model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return [[0.0] * 384 for _ in texts]  # Fallback

    def _cluster_texts(
        self,
        texts: list[str],
        embeddings: list[list[float]],
    ) -> list[list[str]]:
        """Cluster texts by embedding similarity."""
        if len(texts) <= self.cluster_size:
            return [texts]

        # Simple k-means style clustering
        embeddings_array = np.array(embeddings)
        n_clusters = max(1, len(texts) // self.cluster_size)

        # Initialize centroids randomly
        indices = np.random.choice(len(texts), n_clusters, replace=False)
        centroids = embeddings_array[indices]

        # Assign to nearest centroid
        clusters = [[] for _ in range(n_clusters)]
        for i, emb in enumerate(embeddings_array):
            distances = np.linalg.norm(centroids - emb, axis=1)
            nearest = np.argmin(distances)
            clusters[nearest].append(texts[i])

        # Filter empty clusters
        return [c for c in clusters if c]

    def _summarize_cluster(self, texts: list[str]) -> str:
        """Summarize a cluster of texts."""
        combined = "\n\n---\n\n".join(texts[:5])  # Limit input size

        prompt = f"""Summarize the following text excerpts from an SEC fund filing.
Extract and list the key facts, numbers, and terms found.
Be factual and quote exact values.

Texts:
{combined[:4000]}

Summary (2-3 sentences):"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.summary_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Cluster summarization failed: {e}")
            return "Summary generation failed."


def get_chunking_strategy(
    strategy_type: ChunkingStrategyType,
    config: Optional[ChunkingConfig] = None,
    llm_client=None,
    embedding_model=None,
    min_paragraph_tokens: int = 30,
) -> BaseChunkingStrategy:
    """
    Factory function to get a chunking strategy by type.

    Args:
        strategy_type: The type of chunking strategy to create
        config: Base chunking configuration (max_tokens, overlap_tokens, etc.)
        llm_client: LLM client for strategies that need summarization (hierarchical, RAPTOR)
        embedding_model: Embedding model for strategies that need embeddings (RAPTOR)
        min_paragraph_tokens: Minimum tokens for paragraph-atomic strategy (default 30)

    Returns:
        A chunking strategy instance
    """
    if strategy_type == ChunkingStrategyType.STANDARD:
        return StandardChunkingStrategy(config)
    elif strategy_type == ChunkingStrategyType.PARAGRAPH_ATOMIC:
        return ParagraphAtomicChunkingStrategy(config, min_paragraph_tokens=min_paragraph_tokens)
    elif strategy_type == ChunkingStrategyType.HIERARCHICAL:
        return HierarchicalChunkingStrategy(config, llm_client)
    elif strategy_type == ChunkingStrategyType.RAPTOR:
        return RAPTORChunkingStrategy(config, llm_client, embedding_model)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
