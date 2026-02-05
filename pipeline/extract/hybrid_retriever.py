"""
Hybrid Retriever: Union with Deduplication + RRF Ranking.

Combines keyword-based and dense (embedding) retrieval to capture
chunks found by either method, then ranks using Reciprocal Rank Fusion.

Based on failure analysis showing:
- 50.4% both methods correct
- 15.0% keyword-only correct
- 10.6% dense-only correct
- Hybrid ceiling: 76.1%

The union approach captures chunks that either method finds,
potentially improving recall from 65.5% to 70-75%.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from ..parse.models import Chunk, ChunkedDocument, ChunkedSection
from .embedding_retriever import (
    EmbeddingRetriever,
    EmbeddingConfig,
    RetrievedChunk,
)
from .scoped_agentic import (
    FIELD_KEYWORDS,
    score_section_for_field,
    find_relevant_chunks,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class HybridRetrievedChunk:
    """A chunk retrieved by hybrid retrieval with combined scoring."""

    chunk: Chunk
    keyword_rank: Optional[int]  # Rank in keyword results (1-indexed), None if not found
    dense_rank: Optional[int]  # Rank in dense results (1-indexed), None if not found
    keyword_score: Optional[float]  # Raw keyword score
    dense_score: Optional[float]  # Cosine similarity score
    rrf_score: float  # Combined RRF score
    retrieval_source: str  # "keyword_only", "dense_only", or "both"


@dataclass
class HybridConfig:
    """Configuration for hybrid retrieval."""

    # RRF constant (standard value is 60)
    rrf_k: int = 60

    # How many chunks to retrieve from each method before fusion
    keyword_top_k: int = 20
    dense_top_k: int = 20

    # Final number of chunks to return
    final_top_k: int = 10

    # Weights for score combination (alternative to pure RRF)
    # If provided, uses weighted scores instead of rank-based RRF
    keyword_weight: Optional[float] = None  # e.g., 0.5
    dense_weight: Optional[float] = None  # e.g., 0.5


# =============================================================================
# KEYWORD RETRIEVAL ADAPTER
# =============================================================================


def keyword_retrieve_chunks(
    chunked_doc: ChunkedDocument,
    field_name: str,
    top_k: int = 20,
) -> list[tuple[Chunk, int, float]]:
    """
    Retrieve chunks using keyword scoring.

    Adapts the scoped_agentic keyword scoring to return a flat list of chunks.

    Args:
        chunked_doc: The parsed document with sections and chunks
        field_name: Field being extracted (e.g., "incentive_fee")
        top_k: Number of chunks to return

    Returns:
        List of (chunk, rank, score) tuples sorted by score descending
    """
    # Map field to keyword config
    # Handle nested fields like "incentive_fee.hurdle_rate_pct"
    base_field = field_name.split(".")[0]

    # Score all sections
    scored_sections = []
    for section in chunked_doc.chunked_sections:
        if not section.chunks:
            continue
        scored = score_section_for_field(section, base_field)
        if scored.score > 0:
            scored_sections.append(scored)

    # Sort sections by score
    scored_sections.sort(key=lambda s: s.score, reverse=True)

    # Collect chunks from top sections
    chunk_scores: dict[str, tuple[Chunk, float]] = {}

    for scored_section in scored_sections:
        # Find relevant chunks in this section
        relevant_chunks = find_relevant_chunks(
            scored_section.section,
            base_field,
            max_chunks=10,
        )

        for chunk in relevant_chunks:
            # Use chunk content hash as ID for deduplication
            chunk_id = _get_chunk_id(chunk)
            if chunk_id not in chunk_scores:
                # Score combines section score and position
                chunk_scores[chunk_id] = (chunk, scored_section.score)

    # Sort by score and return top-k
    sorted_chunks = sorted(
        chunk_scores.values(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    # Add ranks
    result = []
    for rank, (chunk, score) in enumerate(sorted_chunks, start=1):
        result.append((chunk, rank, score))

    return result


def _get_chunk_id(chunk: Chunk) -> str:
    """Generate a unique ID for a chunk based on content and position."""
    # Use section title + start position + content hash
    content_hash = hash(chunk.content[:200]) if chunk.content else 0
    return f"{chunk.section_title}:{chunk.chunk_index}:{content_hash}"


# =============================================================================
# RECIPROCAL RANK FUSION
# =============================================================================


def reciprocal_rank_fusion(
    keyword_results: list[tuple[Chunk, int, float]],
    dense_results: list[RetrievedChunk],
    rrf_k: int = 60,
    final_top_k: int = 10,
) -> list[HybridRetrievedChunk]:
    """
    Combine keyword and dense retrieval results using Reciprocal Rank Fusion.

    RRF formula: score = 1/(k + rank_keyword) + 1/(k + rank_dense)

    Chunks appearing in only one result set get a penalty rank (1000).

    Args:
        keyword_results: List of (chunk, rank, score) from keyword retrieval
        dense_results: List of RetrievedChunk from dense retrieval
        rrf_k: RRF constant (typically 60)
        final_top_k: Number of chunks to return

    Returns:
        List of HybridRetrievedChunk sorted by RRF score
    """
    # Build lookup maps
    keyword_map: dict[str, tuple[int, float, Chunk]] = {}
    for chunk, rank, score in keyword_results:
        chunk_id = _get_chunk_id(chunk)
        keyword_map[chunk_id] = (rank, score, chunk)

    dense_map: dict[str, tuple[int, float, Chunk]] = {}
    for dense_result in dense_results:
        chunk_id = _get_chunk_id(dense_result.chunk)
        dense_map[chunk_id] = (
            dense_result.rank + 1,  # Convert 0-indexed to 1-indexed
            dense_result.similarity_score,
            dense_result.chunk,
        )

    # Get union of all chunk IDs
    all_chunk_ids = set(keyword_map.keys()) | set(dense_map.keys())

    # Calculate RRF scores
    hybrid_results = []
    penalty_rank = 1000  # Rank for missing results

    for chunk_id in all_chunk_ids:
        # Get keyword rank/score
        if chunk_id in keyword_map:
            kw_rank, kw_score, chunk = keyword_map[chunk_id]
        else:
            kw_rank, kw_score = penalty_rank, None
            chunk = dense_map[chunk_id][2]  # Get chunk from dense

        # Get dense rank/score
        if chunk_id in dense_map:
            dense_rank, dense_score, _ = dense_map[chunk_id]
        else:
            dense_rank, dense_score = penalty_rank, None

        # Calculate RRF score
        rrf_score = 1.0 / (rrf_k + kw_rank) + 1.0 / (rrf_k + dense_rank)

        # Determine source
        if kw_rank < penalty_rank and dense_rank < penalty_rank:
            source = "both"
        elif kw_rank < penalty_rank:
            source = "keyword_only"
        else:
            source = "dense_only"

        hybrid_results.append(HybridRetrievedChunk(
            chunk=chunk,
            keyword_rank=kw_rank if kw_rank < penalty_rank else None,
            dense_rank=dense_rank if dense_rank < penalty_rank else None,
            keyword_score=kw_score,
            dense_score=dense_score,
            rrf_score=rrf_score,
            retrieval_source=source,
        ))

    # Sort by RRF score (highest first) and return top-k
    hybrid_results.sort(key=lambda x: x.rrf_score, reverse=True)

    return hybrid_results[:final_top_k]


# =============================================================================
# HYBRID RETRIEVER CLASS
# =============================================================================


class HybridRetriever:
    """
    Hybrid retriever combining keyword and dense retrieval.

    Uses union with deduplication and RRF ranking to combine results
    from both retrieval methods.
    """

    def __init__(
        self,
        embedding_retriever: EmbeddingRetriever,
        config: Optional[HybridConfig] = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            embedding_retriever: Pre-configured embedding retriever for dense search
            config: Hybrid retrieval configuration
        """
        self.embedding_retriever = embedding_retriever
        self.config = config or HybridConfig()
        self._indexed_doc: Optional[ChunkedDocument] = None

    def index_document(self, chunked_doc: ChunkedDocument) -> int:
        """
        Index a document for retrieval.

        Args:
            chunked_doc: The parsed document

        Returns:
            Number of chunks indexed
        """
        self._indexed_doc = chunked_doc

        # Index for dense retrieval (EmbeddingRetriever takes ChunkedDocument directly)
        num_indexed = self.embedding_retriever.index_chunks(chunked_doc)

        logger.info(f"Indexed {num_indexed} chunks for hybrid retrieval")
        return num_indexed

    def retrieve(
        self,
        field_name: str,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[HybridRetrievedChunk]:
        """
        Retrieve chunks using hybrid keyword + dense retrieval.

        Args:
            field_name: Field being extracted (e.g., "incentive_fee")
            query: Optional custom query for dense retrieval
            top_k: Number of chunks to return (default: config.final_top_k)

        Returns:
            List of HybridRetrievedChunk sorted by RRF score
        """
        if self._indexed_doc is None:
            raise ValueError("No document indexed. Call index_document() first.")

        final_top_k = top_k or self.config.final_top_k

        # Get keyword results
        keyword_results = keyword_retrieve_chunks(
            self._indexed_doc,
            field_name,
            top_k=self.config.keyword_top_k,
        )

        # Get dense results
        dense_results = self.embedding_retriever.retrieve_for_field(
            field_name,
            custom_query=query,
            top_k=self.config.dense_top_k,
        )

        # Combine using RRF
        hybrid_results = reciprocal_rank_fusion(
            keyword_results,
            dense_results,
            rrf_k=self.config.rrf_k,
            final_top_k=final_top_k,
        )

        # Log retrieval stats
        keyword_only = sum(1 for r in hybrid_results if r.retrieval_source == "keyword_only")
        dense_only = sum(1 for r in hybrid_results if r.retrieval_source == "dense_only")
        both = sum(1 for r in hybrid_results if r.retrieval_source == "both")

        logger.info(
            f"Hybrid retrieval for '{field_name}': {len(hybrid_results)} chunks "
            f"(both: {both}, keyword_only: {keyword_only}, dense_only: {dense_only})"
        )

        return hybrid_results

    def get_stats(self) -> dict:
        """Return retriever statistics."""
        # Get document identifier if available
        indexed_doc_id = None
        if self._indexed_doc:
            indexed_doc_id = getattr(self._indexed_doc, 'filing_id', None) or \
                             getattr(self._indexed_doc, 'cik', 'indexed')

        return {
            "type": "hybrid",
            "config": {
                "rrf_k": self.config.rrf_k,
                "keyword_top_k": self.config.keyword_top_k,
                "dense_top_k": self.config.dense_top_k,
                "final_top_k": self.config.final_top_k,
            },
            "embedding_stats": self.embedding_retriever.get_stats(),
            "indexed_doc": indexed_doc_id,
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_hybrid_retriever(
    embedding_config: Optional[EmbeddingConfig] = None,
    hybrid_config: Optional[HybridConfig] = None,
    **kwargs,
) -> HybridRetriever:
    """
    Factory function to create a hybrid retriever.

    Args:
        embedding_config: Configuration for the embedding retriever
        hybrid_config: Configuration for hybrid retrieval
        **kwargs: Additional arguments passed to embedding retriever

    Returns:
        Configured HybridRetriever
    """
    from .embedding_retriever import create_embedding_retriever

    # Create embedding retriever
    if embedding_config:
        embedding_retriever = EmbeddingRetriever(embedding_config)
    else:
        embedding_retriever = create_embedding_retriever(**kwargs)

    return HybridRetriever(
        embedding_retriever=embedding_retriever,
        config=hybrid_config,
    )
