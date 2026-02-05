"""
Multi-Query Retriever for SEC Document Extraction.

Wraps existing keyword and/or embedding retrieval with multi-query expansion.
For each field, generates multiple query variations and fuses results with RRF.

This is the integration layer between query_expander.py and the Tier 3 retrieval.

Architecture:
    QueryExpander -> MultiQueryRetriever -> RRF Fusion -> Top-K Chunks -> LLM

Supports three retrieval strategies:
    1. keyword: Multi-query keyword retrieval only
    2. dense: Multi-query dense (embedding) retrieval only
    3. hybrid: Multi-query keyword + dense with two-level RRF fusion
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from ..parse.models import ChunkedDocument, Chunk
from .query_expander import (
    QueryExpander,
    QueryExpansionConfig,
    ExpandedQueries,
    create_query_expander,
)
from .embedding_retriever import EmbeddingRetriever, EmbeddingConfig, RetrievedChunk
from .scoped_agentic import (
    FIELD_KEYWORDS,
    score_section_for_field,
    find_relevant_chunks,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MultiQueryConfig:
    """Configuration for multi-query retrieval."""

    # Retrieval strategy: "keyword", "dense", or "hybrid"
    retrieval_strategy: str = "keyword"

    # Query expansion settings
    expansion_method: str = "programmatic"  # "programmatic", "llm", or "hybrid"
    expansion_model: str = "gpt-4o-mini"

    # RRF settings
    rrf_k: int = 60  # Standard RRF constant

    # Per-query retrieval settings
    per_query_top_k: int = 15  # Chunks per query before fusion
    final_top_k: int = 10  # Final chunks after RRF fusion

    # Embedding settings (only used if retrieval_strategy includes dense)
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"


@dataclass
class MultiQueryRetrievedChunk:
    """A chunk retrieved via multi-query retrieval."""

    chunk: Chunk
    rrf_score: float
    queries_found_by: list[str]  # Which queries found this chunk
    per_query_ranks: dict[str, int]  # query -> rank
    retrieval_strategy: str  # "keyword", "dense", or "hybrid"


# =============================================================================
# KEYWORD RETRIEVAL (using existing scoped_agentic functions)
# =============================================================================

def _keyword_retrieve_for_query(
    chunked_doc: ChunkedDocument,
    query: str,
    field_name: str,
    top_k: int = 15,
) -> list[tuple[Chunk, int, float]]:
    """
    Retrieve chunks using keyword matching for a specific query.

    Uses the existing score_section_for_field logic but also matches
    the query terms directly in chunk content.

    Args:
        chunked_doc: The parsed document
        query: The search query (from expansion)
        field_name: Original field name for fallback to FIELD_KEYWORDS
        top_k: Number of chunks to return

    Returns:
        List of (chunk, rank, score) tuples
    """
    # Score all chunks by query term matches
    query_terms = query.lower().split()
    chunk_scores: dict[str, tuple[Chunk, float]] = {}

    def get_chunk_id(chunk: Chunk) -> str:
        content_hash = hash(chunk.content[:200]) if chunk.content else 0
        return f"{chunk.section_title}:{chunk.chunk_index}:{content_hash}"

    for section in chunked_doc.chunked_sections:
        if not section.chunks:
            continue

        # Section-level score from existing logic
        section_score = score_section_for_field(section, field_name.split(".")[0])

        for chunk in section.chunks:
            chunk_id = get_chunk_id(chunk)
            content_lower = chunk.content.lower() if chunk.content else ""

            # Score by query term matches
            term_score = 0
            for term in query_terms:
                if term in content_lower:
                    # Exact phrase match bonus
                    if query.lower() in content_lower:
                        term_score += 5
                    else:
                        term_score += 1

            # Combine section score and term score
            combined_score = section_score.score * 0.3 + term_score * 0.7

            if combined_score > 0:
                if chunk_id not in chunk_scores or chunk_scores[chunk_id][1] < combined_score:
                    chunk_scores[chunk_id] = (chunk, combined_score)

    # Sort by score and return top-k
    sorted_chunks = sorted(
        chunk_scores.values(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    # Add ranks
    return [
        (chunk, rank, score)
        for rank, (chunk, score) in enumerate(sorted_chunks, start=1)
    ]


# =============================================================================
# MULTI-QUERY RETRIEVER CLASS
# =============================================================================

class MultiQueryRetriever:
    """
    Multi-query retriever with expansion and RRF fusion.

    Generates multiple query variations for each field, runs retrieval
    for each, and fuses results using Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        config: Optional[MultiQueryConfig] = None,
        query_expander: Optional[QueryExpander] = None,
        embedding_retriever: Optional[EmbeddingRetriever] = None,
    ):
        """
        Initialize multi-query retriever.

        Args:
            config: Configuration for retrieval behavior
            query_expander: Pre-configured expander (created if not provided)
            embedding_retriever: Pre-configured embedding retriever (for dense/hybrid)
        """
        self.config = config or MultiQueryConfig()
        self._indexed_doc: Optional[ChunkedDocument] = None
        self._retrieval_stats: dict = {}

        # Create or use provided query expander
        if query_expander:
            self.query_expander = query_expander
        else:
            self.query_expander = create_query_expander(
                expansion_method=self.config.expansion_method,
                model=self.config.expansion_model,
            )

        # Create or use provided embedding retriever (for dense/hybrid)
        if self.config.retrieval_strategy in ("dense", "hybrid"):
            if embedding_retriever:
                self.embedding_retriever = embedding_retriever
            else:
                self.embedding_retriever = EmbeddingRetriever(EmbeddingConfig(
                    provider=self.config.embedding_provider,
                    model=self.config.embedding_model,
                ))
        else:
            self.embedding_retriever = None

    def index_document(self, chunked_doc: ChunkedDocument) -> int:
        """
        Index a document for retrieval.

        Args:
            chunked_doc: The parsed document

        Returns:
            Number of chunks indexed
        """
        self._indexed_doc = chunked_doc
        num_chunks = sum(
            len(section.chunks)
            for section in chunked_doc.chunked_sections
            if section.chunks
        )

        # Index for embedding retrieval if needed
        if self.embedding_retriever:
            self.embedding_retriever.index_chunks(chunked_doc)

        logger.info(f"Indexed {num_chunks} chunks for multi-query retrieval")
        return num_chunks

    def retrieve(
        self,
        field_name: str,
        top_k: Optional[int] = None,
    ) -> list[MultiQueryRetrievedChunk]:
        """
        Retrieve chunks using multi-query expansion and RRF fusion.

        Args:
            field_name: Field being extracted (e.g., "incentive_fee")
            top_k: Number of chunks to return (default: config.final_top_k)

        Returns:
            List of MultiQueryRetrievedChunk sorted by RRF score
        """
        if self._indexed_doc is None:
            raise ValueError("No document indexed. Call index_document() first.")

        final_top_k = top_k or self.config.final_top_k

        # Get expanded queries
        expansion = self.query_expander.get_expanded_queries(field_name)
        queries = expansion.all_queries()

        logger.info(
            f"Multi-query retrieval for '{field_name}': {len(queries)} queries "
            f"(method: {expansion.source})"
        )

        # Track stats
        self._retrieval_stats = {
            "field": field_name,
            "num_queries": len(queries),
            "expansion_source": expansion.source,
            "retrieval_strategy": self.config.retrieval_strategy,
        }

        # Run retrieval per query based on strategy
        all_query_results: dict[str, list[tuple[Chunk, float]]] = {}

        for query in queries:
            if self.config.retrieval_strategy == "keyword":
                results = self._keyword_retrieve(query, field_name)
            elif self.config.retrieval_strategy == "dense":
                results = self._dense_retrieve(query)
            else:  # hybrid
                results = self._hybrid_retrieve(query, field_name)

            all_query_results[query] = results

        # Fuse results with RRF
        fused_results = self._rrf_fusion(all_query_results, final_top_k)

        # Log retrieval stats
        queries_contributed = len(set(
            q for r in fused_results for q in r.queries_found_by
        ))
        self._retrieval_stats["queries_contributed"] = queries_contributed
        self._retrieval_stats["chunks_returned"] = len(fused_results)

        logger.info(
            f"Multi-query retrieval: {len(fused_results)} chunks "
            f"(from {queries_contributed}/{len(queries)} queries)"
        )

        return fused_results

    def _keyword_retrieve(
        self,
        query: str,
        field_name: str,
    ) -> list[tuple[Chunk, float]]:
        """Run keyword retrieval for a single query."""
        results = _keyword_retrieve_for_query(
            self._indexed_doc,
            query,
            field_name,
            top_k=self.config.per_query_top_k,
        )
        return [(chunk, score) for chunk, rank, score in results]

    def _dense_retrieve(self, query: str) -> list[tuple[Chunk, float]]:
        """Run dense (embedding) retrieval for a single query."""
        if not self.embedding_retriever:
            return []

        results = self.embedding_retriever.retrieve(
            query,
            top_k=self.config.per_query_top_k,
        )
        return [(r.chunk, r.similarity_score) for r in results]

    def _hybrid_retrieve(
        self,
        query: str,
        field_name: str,
    ) -> list[tuple[Chunk, float]]:
        """Run hybrid (keyword + dense) retrieval for a single query."""
        # Get keyword results
        keyword_results = self._keyword_retrieve(query, field_name)

        # Get dense results
        dense_results = self._dense_retrieve(query)

        # Simple RRF fusion of the two
        def get_chunk_id(chunk: Chunk) -> str:
            content_hash = hash(chunk.content[:200]) if chunk.content else 0
            return f"{chunk.section_title}:{chunk.chunk_index}:{content_hash}"

        chunk_scores: dict[str, tuple[Chunk, float]] = {}
        penalty_rank = 1000

        # Add keyword results
        for rank, (chunk, score) in enumerate(keyword_results, start=1):
            chunk_id = get_chunk_id(chunk)
            rrf_score = 1.0 / (self.config.rrf_k + rank)
            chunk_scores[chunk_id] = (chunk, rrf_score)

        # Add/merge dense results
        for rank, (chunk, score) in enumerate(dense_results, start=1):
            chunk_id = get_chunk_id(chunk)
            rrf_score = 1.0 / (self.config.rrf_k + rank)

            if chunk_id in chunk_scores:
                # Chunk found by both - add scores
                existing_chunk, existing_score = chunk_scores[chunk_id]
                chunk_scores[chunk_id] = (existing_chunk, existing_score + rrf_score)
            else:
                chunk_scores[chunk_id] = (chunk, rrf_score)

        # Sort by combined score
        sorted_results = sorted(
            chunk_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )[:self.config.per_query_top_k]

        return sorted_results

    def _rrf_fusion(
        self,
        query_results: dict[str, list[tuple[Chunk, float]]],
        final_top_k: int,
    ) -> list[MultiQueryRetrievedChunk]:
        """
        Fuse results from multiple queries using RRF.

        Args:
            query_results: Dict mapping query -> list of (chunk, score)
            final_top_k: Number of results to return

        Returns:
            List of MultiQueryRetrievedChunk sorted by RRF score
        """
        def get_chunk_id(chunk: Chunk) -> str:
            content_hash = hash(chunk.content[:200]) if chunk.content else 0
            return f"{chunk.section_title}:{chunk.chunk_index}:{content_hash}"

        # Build chunk -> (chunk, queries, ranks) map
        chunk_data: dict[str, tuple[Chunk, dict[str, int]]] = {}

        for query, results in query_results.items():
            for rank, (chunk, score) in enumerate(results, start=1):
                chunk_id = get_chunk_id(chunk)

                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = (chunk, {})

                chunk_data[chunk_id][1][query] = rank

        # Calculate RRF scores
        fused_results = []

        for chunk_id, (chunk, query_ranks) in chunk_data.items():
            rrf_score = sum(
                1.0 / (self.config.rrf_k + rank)
                for rank in query_ranks.values()
            )

            fused_results.append(MultiQueryRetrievedChunk(
                chunk=chunk,
                rrf_score=rrf_score,
                queries_found_by=list(query_ranks.keys()),
                per_query_ranks=query_ranks,
                retrieval_strategy=self.config.retrieval_strategy,
            ))

        # Sort by RRF score
        fused_results.sort(key=lambda x: x.rrf_score, reverse=True)

        return fused_results[:final_top_k]

    def get_stats(self) -> dict:
        """Return retrieval statistics."""
        return {
            "config": {
                "retrieval_strategy": self.config.retrieval_strategy,
                "expansion_method": self.config.expansion_method,
                "rrf_k": self.config.rrf_k,
                "per_query_top_k": self.config.per_query_top_k,
                "final_top_k": self.config.final_top_k,
            },
            "expander_stats": self.query_expander.get_stats(),
            "last_retrieval": self._retrieval_stats,
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_multi_query_retriever(
    retrieval_strategy: str = "keyword",
    expansion_method: str = "programmatic",
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small",
    **kwargs,
) -> MultiQueryRetriever:
    """
    Factory function to create a multi-query retriever.

    Args:
        retrieval_strategy: "keyword", "dense", or "hybrid"
        expansion_method: "programmatic", "llm", or "hybrid"
        embedding_provider: For dense/hybrid (openai, voyage, etc.)
        embedding_model: Embedding model name
        **kwargs: Additional MultiQueryConfig options

    Returns:
        Configured MultiQueryRetriever

    Examples:
        # Keyword-only with programmatic expansion (fastest, free)
        retriever = create_multi_query_retriever(
            retrieval_strategy="keyword",
            expansion_method="programmatic",
        )

        # Hybrid retrieval with LLM expansion (best quality)
        retriever = create_multi_query_retriever(
            retrieval_strategy="hybrid",
            expansion_method="llm",
            embedding_provider="voyage",
            embedding_model="voyage-finance-2",
        )
    """
    config = MultiQueryConfig(
        retrieval_strategy=retrieval_strategy,
        expansion_method=expansion_method,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        **kwargs,
    )
    return MultiQueryRetriever(config)
