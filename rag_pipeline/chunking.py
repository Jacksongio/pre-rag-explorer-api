from __future__ import annotations

import re

from .models import ChunkingMethod, ChunkParams, ChunkResult, ChunkStats


def chunk_text(
    text: str,
    method: ChunkingMethod | str,
    params: ChunkParams | None = None,
) -> ChunkResult:
    method = ChunkingMethod(method)
    params = params or ChunkParams()

    match method:
        case ChunkingMethod.FIXED:
            size = params.chunk_size or 1000
            overlap = params.overlap or 200
            chunks = _fixed_size_chunk(text, size, overlap)
        case ChunkingMethod.RECURSIVE:
            size = params.chunk_size or 1000
            overlap = params.overlap or 200
            chunks = _recursive_character_chunk(text, size, overlap)
        case ChunkingMethod.TOKEN:
            token_count = params.token_count or 256
            overlap = params.overlap or 50
            chunks = _token_based_chunk(text, token_count, overlap)
        case ChunkingMethod.SENTENCE:
            sentence_count = params.sentence_count or 5
            overlap = params.overlap or 1
            chunks = _sentence_based_chunk(text, sentence_count, overlap)
        case ChunkingMethod.SEMANTIC:
            chunks = _semantic_mock_chunk(text)

    avg_size = sum(len(c) for c in chunks) / len(chunks) if chunks else 0.0
    avg_tokens = avg_size / 4.0

    return ChunkResult(
        chunks=chunks,
        stats=ChunkStats(count=len(chunks), avg_size=avg_size, avg_tokens=avg_tokens),
    )


def _fixed_size_chunk(text: str, size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + size])
        i += size - overlap
        if i >= len(text) or size <= overlap:
            break
    return chunks


def _recursive_character_chunk(text: str, size: int, overlap: int) -> list[str]:
    separators = ["\n\n", "\n", ". ", " ", ""]

    def split(content: str, depth: int) -> list[str]:
        if len(content) <= size or depth >= len(separators):
            return [content]

        separator = separators[depth]
        parts = content.split(separator) if separator else list(content)
        result: list[str] = []
        current = ""

        for part in parts:
            candidate = (current + separator + part) if current else part
            if len(candidate) <= size:
                current = candidate
            else:
                if current:
                    result.append(current)
                current = part

        if current:
            result.append(current)

        final: list[str] = []
        for r in result:
            if len(r) > size:
                final.extend(split(r, depth + 1))
            else:
                final.append(r)
        return final

    return split(text, 0)


def _token_based_chunk(text: str, token_count: int, overlap: int) -> list[str]:
    char_size = token_count * 4
    char_overlap = overlap * 4
    return _fixed_size_chunk(text, char_size, char_overlap)


def _sentence_based_chunk(text: str, count: int, overlap: int) -> list[str]:
    sentences = re.findall(r"[^.!?]+[.!?]+", text)
    if not sentences:
        sentences = [text]

    chunks: list[str] = []
    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i : i + count]).strip()
        chunks.append(chunk)
        if i + count >= len(sentences):
            break
        i += count - overlap

    return chunks


def _semantic_mock_chunk(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]
