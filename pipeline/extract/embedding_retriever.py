"""
Embedding-based Dense Retrieval for SEC Document Extraction.

This module provides embedding-based retrieval as an alternative to
keyword-based Tier 3 retrieval. It supports multiple embedding providers:

1. OpenAI: text-embedding-3-small, text-embedding-3-large
2. Voyage AI: voyage-finance-2 (domain-specific), voyage-3 (general)
3. Local (sentence-transformers): bge-m3, nomic-embed-text-v1.5

Usage:
    retriever = EmbeddingRetriever(
        provider="openai",
        model="text-embedding-3-small",
    )
    retriever.index_chunks(chunked_doc)
    relevant_chunks = retriever.retrieve("minimum investment amount", top_k=10)
"""

import hashlib
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import json
import numpy as np

from ..parse.models import ChunkedDocument, ChunkedSection, Chunk

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for embedding-based retrieval."""

    # Provider: "openai", "voyage", "local"
    provider: str = "openai"

    # Model name (provider-specific)
    model: str = "text-embedding-3-small"

    # API key (if None, uses environment variable)
    api_key: Optional[str] = None

    # Embedding dimensions (for models that support it)
    dimensions: Optional[int] = None

    # Batch size for embedding calls
    batch_size: int = 100

    # Rate limiting
    requests_per_minute: int = 500
    delay_between_calls: float = 0.0

    # Retrieval settings
    top_k: int = 10

    # Cache settings
    cache_embeddings: bool = True
    cache_dir: str = "data/cache/embeddings"

    # Context prepending (adds section info to chunk before embedding)
    prepend_context: bool = False
    context_template: str = "Section: {section_title}\n\n{content}"


@dataclass
class RetrievedChunk:
    """A chunk with its similarity score."""
    chunk: Chunk
    similarity_score: float
    rank: int


# =============================================================================
# EMBEDDING PROVIDERS (Abstract Base)
# =============================================================================

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query text.

        Some providers have separate query vs document embeddings.

        Args:
            query: Query text to embed

        Returns:
            numpy array of shape (embedding_dim,)
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        pass


# =============================================================================
# OPENAI EMBEDDINGS
# =============================================================================

class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embedding provider."""

    # Model dimensions
    MODEL_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 100,
        delay_between_calls: float = 0.0,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.delay_between_calls = delay_between_calls
        self._client = None
        self._last_call_time = 0.0

        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")

    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    @property
    def embedding_dim(self) -> int:
        if self.dimensions:
            return self.dimensions
        return self.MODEL_DIMS.get(self.model, 1536)

    def _rate_limit(self):
        """Apply rate limiting."""
        if self.delay_between_calls > 0:
            elapsed = time.time() - self._last_call_time
            if elapsed < self.delay_between_calls:
                time.sleep(self.delay_between_calls - elapsed)
        self._last_call_time = time.time()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed texts in batches."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            self._rate_limit()

            kwargs = {"model": self.model, "input": batch}
            if self.dimensions and self.model.startswith("text-embedding-3"):
                kwargs["dimensions"] = self.dimensions

            response = self.client.embeddings.create(**kwargs)
            batch_embeddings = [e.embedding for e in response.data]
            all_embeddings.extend(batch_embeddings)

            logger.debug(f"Embedded batch {i//self.batch_size + 1}, {len(batch)} texts")

        return np.array(all_embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        result = self.embed_texts([query])
        return result[0]


# =============================================================================
# VOYAGE AI EMBEDDINGS
# =============================================================================

class VoyageEmbeddings(EmbeddingProvider):
    """Voyage AI embedding provider (includes domain-specific finance model)."""

    MODEL_DIMS = {
        "voyage-3": 1024,
        "voyage-3-lite": 512,
        "voyage-finance-2": 1024,
        "voyage-large-2": 1536,
        "voyage-code-2": 1536,
    }

    def __init__(
        self,
        model: str = "voyage-finance-2",
        api_key: Optional[str] = None,
        batch_size: int = 128,
        delay_between_calls: float = 0.0,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        self.batch_size = batch_size
        self.delay_between_calls = delay_between_calls
        self._client = None
        self._last_call_time = 0.0

        if not self.api_key:
            raise ValueError("Voyage API key required. Set VOYAGE_API_KEY env var.")

    @property
    def client(self):
        """Lazy-load Voyage client."""
        if self._client is None:
            import voyageai
            self._client = voyageai.Client(api_key=self.api_key)
        return self._client

    @property
    def embedding_dim(self) -> int:
        return self.MODEL_DIMS.get(self.model, 1024)

    def _rate_limit(self):
        """Apply rate limiting."""
        if self.delay_between_calls > 0:
            elapsed = time.time() - self._last_call_time
            if elapsed < self.delay_between_calls:
                time.sleep(self.delay_between_calls - elapsed)
        self._last_call_time = time.time()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed texts (as documents)."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            self._rate_limit()

            result = self.client.embed(
                texts=batch,
                model=self.model,
                input_type="document",
            )
            all_embeddings.extend(result.embeddings)

            logger.debug(f"Embedded batch {i//self.batch_size + 1}, {len(batch)} texts")

        return np.array(all_embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query (uses query input_type for asymmetric embedding)."""
        self._rate_limit()
        result = self.client.embed(
            texts=[query],
            model=self.model,
            input_type="query",
        )
        return np.array(result.embeddings[0])


# =============================================================================
# LOCAL EMBEDDINGS (sentence-transformers)
# =============================================================================

class LocalEmbeddings(EmbeddingProvider):
    """Local embedding provider using sentence-transformers."""

    MODEL_DIMS = {
        "BAAI/bge-m3": 1024,
        "BAAI/bge-large-en-v1.5": 1024,
        "nomic-ai/nomic-embed-text-v1.5": 768,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
    }

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        self.model_name = model
        self.batch_size = batch_size
        self._model = None
        self._device = device

    @property
    def model(self):
        """Lazy-load sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_name,
                device=self._device,
            )
            logger.info(f"Loaded local embedding model: {self.model_name}")
        return self._model

    @property
    def embedding_dim(self) -> int:
        return self.MODEL_DIMS.get(self.model_name, 768)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed texts using sentence-transformers."""
        # For BGE models, add instruction prefix for documents
        if "bge" in self.model_name.lower():
            # BGE doesn't need prefix for documents, only queries
            pass

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query."""
        # For BGE models, add instruction prefix
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        return embedding[0]


# =============================================================================
# EMBEDDING RETRIEVER
# =============================================================================

class EmbeddingRetriever:
    """
    Dense retrieval using embeddings.

    Indexes document chunks and retrieves by semantic similarity.
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize retriever with configuration.

        Args:
            config: EmbeddingConfig with provider, model, and settings
        """
        self.config = config
        self._provider: Optional[EmbeddingProvider] = None
        self._chunk_embeddings: Optional[np.ndarray] = None
        self._chunks: list[Chunk] = []
        self._indexed_doc_id: Optional[str] = None

        # Create cache directory if needed
        if config.cache_embeddings:
            Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

    @property
    def provider(self) -> EmbeddingProvider:
        """Lazy-load embedding provider."""
        if self._provider is None:
            self._provider = self._create_provider()
        return self._provider

    def _create_provider(self) -> EmbeddingProvider:
        """Create the appropriate embedding provider."""
        if self.config.provider == "openai":
            return OpenAIEmbeddings(
                model=self.config.model,
                api_key=self.config.api_key,
                dimensions=self.config.dimensions,
                batch_size=self.config.batch_size,
                delay_between_calls=self.config.delay_between_calls,
            )
        elif self.config.provider == "voyage":
            return VoyageEmbeddings(
                model=self.config.model,
                api_key=self.config.api_key,
                batch_size=self.config.batch_size,
                delay_between_calls=self.config.delay_between_calls,
            )
        elif self.config.provider == "local":
            return LocalEmbeddings(
                model=self.config.model,
                batch_size=self.config.batch_size,
            )
        else:
            raise ValueError(f"Unknown embedding provider: {self.config.provider}")

    def _get_cache_path(self, doc_id: str) -> Path:
        """Get cache file path for a document."""
        # Include model in cache key
        cache_key = f"{doc_id}_{self.config.provider}_{self.config.model}"
        if self.config.prepend_context:
            cache_key += "_ctx"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]
        return Path(self.config.cache_dir) / f"{cache_hash}.npz"

    def _prepare_chunk_text(self, chunk: Chunk) -> str:
        """Prepare chunk text for embedding (optionally with context)."""
        if self.config.prepend_context:
            return self.config.context_template.format(
                section_title=chunk.section_title or "",
                content=chunk.content,
            )
        return chunk.content

    def index_chunks(self, chunked_doc: ChunkedDocument) -> int:
        """
        Index all chunks from a document.

        Args:
            chunked_doc: ChunkedDocument to index

        Returns:
            Number of chunks indexed
        """
        doc_id = chunked_doc.filing_id

        # Check cache first
        cache_path = self._get_cache_path(doc_id)
        if self.config.cache_embeddings and cache_path.exists():
            logger.info(f"Loading cached embeddings from {cache_path}")
            cached = np.load(cache_path, allow_pickle=True)
            self._chunk_embeddings = cached["embeddings"]
            self._chunks = list(cached["chunks"])
            self._indexed_doc_id = doc_id
            logger.info(f"Loaded {len(self._chunks)} cached chunk embeddings")
            return len(self._chunks)

        # Collect all chunks
        self._chunks = []
        for section in chunked_doc.chunked_sections:
            self._chunks.extend(section.chunks)

        if not self._chunks:
            logger.warning("No chunks to index")
            return 0

        # Prepare texts for embedding
        texts = [self._prepare_chunk_text(chunk) for chunk in self._chunks]

        # Embed all chunks
        logger.info(f"Embedding {len(texts)} chunks with {self.config.provider}/{self.config.model}")
        start_time = time.time()
        self._chunk_embeddings = self.provider.embed_texts(texts)
        elapsed = time.time() - start_time
        logger.info(f"Embedded {len(texts)} chunks in {elapsed:.2f}s")

        # Cache embeddings
        if self.config.cache_embeddings:
            logger.info(f"Caching embeddings to {cache_path}")
            np.savez(
                cache_path,
                embeddings=self._chunk_embeddings,
                chunks=np.array(self._chunks, dtype=object),
            )

        self._indexed_doc_id = doc_id
        return len(self._chunks)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        field_name: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve most relevant chunks for a query.

        Args:
            query: Search query
            top_k: Number of chunks to return (default: config.top_k)
            field_name: Optional field name (for logging)

        Returns:
            List of RetrievedChunk sorted by similarity (highest first)
        """
        if self._chunk_embeddings is None:
            raise ValueError("No chunks indexed. Call index_chunks() first.")

        top_k = top_k or self.config.top_k

        # Embed query
        query_embedding = self.provider.embed_query(query)

        # Compute cosine similarities
        # Embeddings are already normalized for most providers
        similarities = np.dot(self._chunk_embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for rank, idx in enumerate(top_indices):
            results.append(RetrievedChunk(
                chunk=self._chunks[idx],
                similarity_score=float(similarities[idx]),
                rank=rank,
            ))

        if field_name:
            logger.info(
                f"Retrieved {len(results)} chunks for '{field_name}' "
                f"(top score: {results[0].similarity_score:.3f})"
            )

        return results

    def retrieve_for_field(
        self,
        field_name: str,
        custom_query: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve chunks relevant to a specific field.

        Uses predefined queries for known fields, or generates a query.

        Args:
            field_name: Field being extracted
            custom_query: Optional custom query
            top_k: Number of chunks to return

        Returns:
            List of RetrievedChunk
        """
        # Use custom query or generate from field name
        if custom_query:
            query = custom_query
        else:
            # Use the same queries as the reranker for consistency
            from .scoped_agentic import RERANKER_QUERIES
            query = RERANKER_QUERIES.get(
                field_name,
                f"What is the {field_name.replace('_', ' ')} for this fund?"
            )

        return self.retrieve(query, top_k=top_k, field_name=field_name)

    def get_stats(self) -> dict:
        """Return retriever statistics."""
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "indexed_chunks": len(self._chunks) if self._chunks else 0,
            "embedding_dim": self.provider.embedding_dim if self._provider else None,
            "prepend_context": self.config.prepend_context,
            "indexed_doc_id": self._indexed_doc_id,
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_embedding_retriever(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> EmbeddingRetriever:
    """
    Factory function to create an embedding retriever.

    Args:
        provider: "openai", "voyage", or "local"
        model: Model name (defaults to best for provider)
        api_key: API key (or uses env var)
        **kwargs: Additional EmbeddingConfig options

    Returns:
        Configured EmbeddingRetriever
    """
    # Default models per provider
    default_models = {
        "openai": "text-embedding-3-small",
        "voyage": "voyage-finance-2",
        "local": "BAAI/bge-m3",
    }

    config = EmbeddingConfig(
        provider=provider,
        model=model or default_models.get(provider, "text-embedding-3-small"),
        api_key=api_key,
        **kwargs,
    )

    return EmbeddingRetriever(config)
