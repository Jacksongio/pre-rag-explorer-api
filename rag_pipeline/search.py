from __future__ import annotations

from .models import RetrievalMethod, SearchResult, VectorCollection
from .similarity import compute_bm25, cosine_similarity


def search_dense(
    query_embedding: list[float],
    collection: VectorCollection,
) -> list[SearchResult]:
    results: list[SearchResult] = []
    for idx, vec in enumerate(collection.vectors):
        score = cosine_similarity(query_embedding, vec)
        results.append(
            SearchResult(
                chunk=collection.chunks[idx],
                score=score,
                retrieval_method=RetrievalMethod.DENSE,
                collection_name=collection.name,
                collection_id=collection.id,
                embedding_model=collection.embedding_model,
            )
        )
    return results


def search_sparse(
    query: str,
    collection: VectorCollection,
) -> list[SearchResult]:
    doc_texts = [c.text for c in collection.chunks]
    scores = compute_bm25(query, doc_texts)
    max_score = max(*scores, 1.0) if scores else 1.0

    results: list[SearchResult] = []
    for idx, raw_score in enumerate(scores):
        results.append(
            SearchResult(
                chunk=collection.chunks[idx],
                score=raw_score / max_score,
                retrieval_method=RetrievalMethod.SPARSE,
                collection_name=collection.name,
                collection_id=collection.id,
                embedding_model=collection.embedding_model,
            )
        )
    return results


def search_hybrid(
    query: str,
    query_embedding: list[float],
    collection: VectorCollection,
    dense_weight: float = 0.7,
) -> list[SearchResult]:
    sparse_weight = 1.0 - dense_weight

    doc_texts = [c.text for c in collection.chunks]
    dense_scores = [cosine_similarity(query_embedding, vec) for vec in collection.vectors]
    sparse_scores = compute_bm25(query, doc_texts)
    max_sparse = max(*sparse_scores, 1.0) if sparse_scores else 1.0

    results: list[SearchResult] = []
    for idx in range(len(collection.chunks)):
        ds = dense_scores[idx]
        ss = sparse_scores[idx] / max_sparse
        hybrid_score = (ds * dense_weight) + (ss * sparse_weight)
        results.append(
            SearchResult(
                chunk=collection.chunks[idx],
                score=hybrid_score,
                retrieval_method=RetrievalMethod.HYBRID,
                collection_name=collection.name,
                collection_id=collection.id,
                embedding_model=collection.embedding_model,
            )
        )
    return results


def search(
    query: str,
    query_embedding: list[float],
    collections: list[VectorCollection],
    methods: list[RetrievalMethod | str] | None = None,
    top_k: int = 5,
    dense_weight: float = 0.7,
) -> list[SearchResult]:
    """Run one or more retrieval methods across collections and return ranked results."""
    if methods is None:
        methods = [RetrievalMethod.DENSE]
    methods = [RetrievalMethod(m) for m in methods]

    all_results: list[SearchResult] = []

    for collection in collections:
        if RetrievalMethod.DENSE in methods:
            all_results.extend(search_dense(query_embedding, collection))
        if RetrievalMethod.SPARSE in methods:
            all_results.extend(search_sparse(query, collection))
        if RetrievalMethod.HYBRID in methods:
            all_results.extend(
                search_hybrid(query, query_embedding, collection, dense_weight)
            )

    all_results.sort(key=lambda r: r.score, reverse=True)
    return all_results[: top_k * len(methods)]
