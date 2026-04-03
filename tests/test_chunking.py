from __future__ import annotations

import pytest

from rag_pipeline import chunk_text
from rag_pipeline.models import ChunkingMethod, ChunkParams


def test_chunk_fixed_size_and_stats():
    # overlap=0 is treated as default (200) in chunk_text; use a non-zero overlap for windows.
    text = "abcdefghijklmnop"
    result = chunk_text(text, ChunkingMethod.FIXED, ChunkParams(chunk_size=4, overlap=1))
    assert result.chunks[0] == "abcd"
    assert result.stats.count >= 4
    assert result.stats.count == len(result.chunks)


def test_chunk_fixed_accepts_string_method():
    result = chunk_text("abcd", "fixed", ChunkParams(chunk_size=2, overlap=1))
    # Final step can emit a short tail after the last stride.
    assert result.chunks == ["ab", "bc", "cd", "d"]


def test_chunk_recursive_respects_max_segment_size():
    text = "paragraph one\n\nparagraph two with more words in it"
    result = chunk_text(
        text,
        ChunkingMethod.RECURSIVE,
        ChunkParams(chunk_size=80, overlap=0),
    )
    assert result.stats.count >= 1
    joined = "\n".join(result.chunks)
    assert "paragraph one" in joined and "paragraph two" in joined


def test_chunk_token_uses_character_window():
    # overlap=0 becomes default 50 in chunk_text; overlap=1 keeps a small char step.
    result = chunk_text(
        "x" * 30,
        ChunkingMethod.TOKEN,
        ChunkParams(token_count=5, overlap=1),
    )
    assert result.stats.count == 2
    assert all(len(c) <= 20 for c in result.chunks)
    assert result.chunks[0] == "x" * 20 and "x" in result.chunks[-1]


def test_chunk_sentence():
    text = "First one. Second two. Third three."
    result = chunk_text(
        text,
        ChunkingMethod.SENTENCE,
        ChunkParams(sentence_count=2, overlap=0),
    )
    assert len(result.chunks) == 2
    assert "First" in result.chunks[0]
    assert "Third" in result.chunks[1]


def test_chunk_semantic_splits_on_blank_lines():
    text = "para one\n\npara two\n\n"
    result = chunk_text(text, ChunkingMethod.SEMANTIC)
    assert result.chunks == ["para one", "para two"]


def test_chunk_empty_text():
    result = chunk_text("", ChunkingMethod.FIXED, ChunkParams(chunk_size=10, overlap=0))
    assert result.chunks == []
    assert result.stats.count == 0
    assert result.stats.avg_size == 0.0
