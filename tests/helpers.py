from __future__ import annotations

from rag_pipeline.models import Chunk, ChunkingMethod, ChunkParams, VectorCollection


def make_vector_collection(
    *,
    chunks: list[Chunk],
    vectors: list[list[float]],
    created_at: str,
    coll_id: str = "col1",
    name: str = "test",
) -> VectorCollection:
    return VectorCollection(
        id=coll_id,
        name=name,
        chunk_method=ChunkingMethod.FIXED,
        source_file_id="file1",
        source_file_name="source.txt",
        chunk_count=len(chunks),
        params=ChunkParams(),
        created_at=created_at,
        chunks=chunks,
        vectors=vectors,
        embedding_model="test-model",
    )
