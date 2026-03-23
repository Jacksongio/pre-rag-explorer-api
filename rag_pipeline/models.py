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
