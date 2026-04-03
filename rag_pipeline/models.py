from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class FileType(StrEnum):
    TEXT = "text"
    CSV = "csv"
    PDF = "pdf"
    MARKDOWN = "markdown"


class ChunkingMethod(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    TOKEN = "token"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"


class RetrievalMethod(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class ChunkParams(BaseModel):
    chunk_size: int = 1000
    overlap: int | None = None
    token_count: int = 256
    sentence_count: int = 5
    similarity_threshold: float = 0.5


class ChunkStats(BaseModel):
    count: int
    avg_size: float
    avg_tokens: float


class ChunkResult(BaseModel):
    chunks: list[str]
    stats: ChunkStats


class Chunk(BaseModel):
    id: str
    text: str
    index: int
    source_file_id: str
    source_file_name: str
    chunk_method: ChunkingMethod
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorCollection(BaseModel):
    id: str
    name: str
    chunk_method: ChunkingMethod
    source_file_id: str
    source_file_name: str
    chunk_count: int
    params: ChunkParams
    created_at: str
    chunks: list[Chunk]
    vectors: list[list[float]]
    embedding_model: str | None = None


class SearchResult(BaseModel):
    chunk: Chunk
    score: float
    retrieval_method: RetrievalMethod
    collection_name: str
    collection_id: str
    embedding_model: str | None = None


class RetrievalAccuracy(BaseModel):
    """Retrieval quality vs labelled relevant chunk ids (same query)."""

    k: int = Field(description="Number of ranked results evaluated (list length).")
    precision_at_k: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of top-k results that are relevant.",
    )
    recall_at_k: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of labelled relevant chunks that appear in top-k.",
    )
    mean_reciprocal_rank: float = Field(
        ge=0.0,
        le=1.0,
        description="Reciprocal rank of the first relevant hit (0 if none).",
    )
    ndcg_at_k: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized discounted cumulative gain (binary relevance).",
    )
    hit_at_k: bool = Field(description="Whether at least one relevant chunk is in top-k.")
    relevant_in_top_k: int = Field(ge=0, description="Count of relevant chunks in top-k.")
    total_labelled_relevant: int = Field(
        ge=0,
        description="Number of chunk ids supplied as relevant for this query.",
    )


class SearchResponse(BaseModel):
    """Ranked hits plus optional accuracy when ground-truth chunk ids are provided."""

    results: list[SearchResult]
    accuracy: RetrievalAccuracy | None = Field(
        default=None,
        description="Set when `relevant_chunk_ids` is passed to `search`.",
    )
