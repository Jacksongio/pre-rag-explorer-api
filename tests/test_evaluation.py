from __future__ import annotations

import math

from rag_pipeline.evaluation import compute_retrieval_accuracy
from rag_pipeline.models import Chunk, ChunkingMethod, RetrievalMethod, SearchResult


def _result(chunk_id: str, score: float) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id=chunk_id,
            text="t",
            index=0,
            source_file_id="f",
            source_file_name="f.txt",
            chunk_method=ChunkingMethod.FIXED,
        ),
        score=score,
        retrieval_method=RetrievalMethod.DENSE,
        collection_name="c",
        collection_id="cid",
    )


def test_perfect_ranking_metrics():
    results = [_result("a", 1.0), _result("b", 0.5), _result("c", 0.25)]
    acc = compute_retrieval_accuracy(results, {"a"})
    assert math.isclose(acc.precision_at_k, 1 / 3)
    assert acc.recall_at_k == 1.0
    assert acc.mean_reciprocal_rank == 1.0
    assert acc.hit_at_k is True
    assert acc.ndcg_at_k == 1.0


def test_mrr_second_position():
    results = [_result("x", 1.0), _result("y", 0.9), _result("z", 0.8)]
    acc = compute_retrieval_accuracy(results, {"y"})
    assert acc.mean_reciprocal_rank == 0.5
    assert acc.hit_at_k is True


def test_miss_everything():
    results = [_result("a", 1.0), _result("b", 0.5)]
    acc = compute_retrieval_accuracy(results, {"z"})
    assert acc.hit_at_k is False
    assert acc.mean_reciprocal_rank == 0.0
    assert acc.recall_at_k == 0.0
    assert acc.precision_at_k == 0.0
    assert acc.ndcg_at_k == 0.0


def test_empty_results():
    acc = compute_retrieval_accuracy([], {"a"})
    assert acc.k == 0
    assert acc.hit_at_k is False
    assert acc.total_labelled_relevant == 1


def test_multi_relevant_recall():
    results = [_result("a", 1.0), _result("b", 0.5)]
    acc = compute_retrieval_accuracy(results, {"a", "b", "c"})
    assert acc.relevant_in_top_k == 2
    assert math.isclose(acc.recall_at_k, 2 / 3)
