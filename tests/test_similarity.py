from __future__ import annotations

import math

from rag_pipeline import compute_bm25, cosine_similarity


def test_cosine_identical_unit_vectors():
    v = [0.6, 0.8]
    assert math.isclose(cosine_similarity(v, v), 1.0)


def test_cosine_orthogonal():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_bm25_empty_documents():
    assert compute_bm25("any query", []) == []


def test_bm25_prefers_matching_terms():
    scores = compute_bm25(
        "python asyncio",
        [
            "python asyncio tutorial",
            "cooking recipes",
        ],
    )
    assert len(scores) == 2
    assert scores[0] > scores[1]
