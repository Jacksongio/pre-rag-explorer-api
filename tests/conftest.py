from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rag_pipeline.models import Chunk, ChunkingMethod


@pytest.fixture
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            id="c0",
            text="alpha beta gamma",
            index=0,
            source_file_id="file1",
            source_file_name="source.txt",
            chunk_method=ChunkingMethod.FIXED,
        ),
        Chunk(
            id="c1",
            text="delta epsilon zeta",
            index=1,
            source_file_id="file1",
            source_file_name="source.txt",
            chunk_method=ChunkingMethod.FIXED,
        ),
    ]
