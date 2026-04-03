from __future__ import annotations

from rag_pipeline import search, search_dense, search_hybrid, search_sparse
from rag_pipeline.models import RetrievalMethod

from tests.helpers import make_vector_collection


def test_search_dense_orders_by_cosine(utc_now, sample_chunks):
    q = [1.0, 0.0, 0.0]
    vecs = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    coll = make_vector_collection(chunks=sample_chunks, vectors=vecs, created_at=utc_now)
    results = search_dense(q, coll)
    assert results[0].score >= results[1].score
    assert results[0].chunk.id == "c0"
    assert all(r.retrieval_method == RetrievalMethod.DENSE for r in results)


def test_search_sparse_scores_documents(utc_now, sample_chunks):
    coll = make_vector_collection(
        chunks=sample_chunks,
        vectors=[[0.0], [0.0]],
        created_at=utc_now,
    )
    results = search_sparse("alpha gamma", coll)
    assert len(results) == 2
    assert results[0].chunk.id == "c0"
    assert results[0].score > results[1].score


def test_search_hybrid_combines_signals(utc_now, sample_chunks):
    q_emb = [1.0, 0.0, 0.0]
    vecs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    coll = make_vector_collection(chunks=sample_chunks, vectors=vecs, created_at=utc_now)
    results = search_hybrid("alpha beta", q_emb, coll, dense_weight=0.5)
    assert len(results) == 2
    assert all(r.retrieval_method == RetrievalMethod.HYBRID for r in results)


def test_search_merges_methods_and_respects_top_k(utc_now, sample_chunks):
    q_emb = [1.0, 0.0, 0.0]
    vecs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    coll = make_vector_collection(chunks=sample_chunks, vectors=vecs, created_at=utc_now)
    out = search(
        "alpha",
        q_emb,
        [coll],
        methods=["dense", "sparse"],
        top_k=1,
    )
    assert len(out.results) == 2
    assert out.accuracy is None


def test_search_includes_accuracy_when_labels_provided(utc_now, sample_chunks):
    q_emb = [1.0, 0.0, 0.0]
    vecs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    coll = make_vector_collection(chunks=sample_chunks, vectors=vecs, created_at=utc_now)
    out = search(
        "alpha",
        q_emb,
        [coll],
        methods=["dense"],
        top_k=5,
        relevant_chunk_ids={"c0"},
    )
    assert out.accuracy is not None
    assert out.accuracy.hit_at_k is True
    assert out.accuracy.mean_reciprocal_rank == 1.0
    assert out.accuracy.relevant_in_top_k >= 1
