"""Pre-RAG Explorer — parse, chunk, embed, and search documents."""

from .chunking import chunk_text
from .embeddings import generate_embeddings, generate_query_embedding
from .models import (
    Chunk,
    ChunkingMethod,
    ChunkParams,
    ChunkResult,
    ChunkStats,
    FileType,
    RetrievalMethod,
    SearchResult,
    VectorCollection,
)
from .parsing import parse_file
from .search import search, search_dense, search_hybrid, search_sparse
from .similarity import compute_bm25, cosine_similarity

__all__ = [
    "chunk_text",
    "compute_bm25",
    "cosine_similarity",
    "generate_embeddings",
    "generate_query_embedding",
    "parse_file",
    "search",
    "search_dense",
    "search_hybrid",
    "search_sparse",
    "Chunk",
    "ChunkingMethod",
    "ChunkParams",
    "ChunkResult",
    "ChunkStats",
    "FileType",
    "RetrievalMethod",
    "SearchResult",
    "VectorCollection",
]
