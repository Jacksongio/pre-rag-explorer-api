from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import rag_pipeline.embeddings as embeddings_mod
from rag_pipeline import generate_embeddings, generate_query_embedding


@pytest.fixture(autouse=True)
def clear_model_cache():
    embeddings_mod._models.clear()
    yield
    embeddings_mod._models.clear()


def test_generate_embeddings_empty_list():
    assert generate_embeddings([]) == []


def test_generate_query_embedding_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        generate_query_embedding("   ")


def test_generate_embeddings_replaces_blank_strings():
    model = MagicMock()
    model.encode.return_value = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.25, 0.75]],
        dtype=np.float64,
    )
    with patch.object(embeddings_mod, "_get_model", return_value=model):
        out = generate_embeddings(["real", "", "   "])
    assert len(out) == 3
    calls = model.encode.call_args[0][0]
    assert calls[1] == " " and calls[2] == " "
    assert out[0] == [1.0, 0.0]
    assert out[2] == [0.25, 0.75]


def test_generate_query_embedding_returns_list():
    model = MagicMock()
    model.encode.return_value = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    with patch.object(embeddings_mod, "_get_model", return_value=model):
        vec = generate_query_embedding("hello")
    assert vec == [0.0, 1.0, 0.0]
    model.encode.assert_called_once()
