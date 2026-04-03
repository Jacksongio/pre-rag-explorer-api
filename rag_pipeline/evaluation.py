from __future__ import annotations

import math
from collections.abc import Collection

from .models import RetrievalAccuracy, SearchResult


def compute_retrieval_accuracy(
    results: list[SearchResult],
    relevant_chunk_ids: Collection[str],
) -> RetrievalAccuracy:
    """Binary-relevance metrics over a ranked result list (higher score = better rank)."""
    relevant = {str(x) for x in relevant_chunk_ids}
    k = len(results)
    if k == 0:
        return RetrievalAccuracy(
            k=0,
            precision_at_k=0.0,
            recall_at_k=0.0,
            mean_reciprocal_rank=0.0,
            ndcg_at_k=0.0,
            hit_at_k=False,
            relevant_in_top_k=0,
            total_labelled_relevant=len(relevant),
        )

    retrieved_ids = [r.chunk.id for r in results]
    in_top = [rid in relevant for rid in retrieved_ids]
    relevant_in_top_k = sum(in_top)
    total_rel = len(relevant)

    precision_at_k = relevant_in_top_k / k
    recall_at_k = (
        relevant_in_top_k / total_rel if total_rel > 0 else 0.0
    )

    mrr = 0.0
    for i, ok in enumerate(in_top, start=1):
        if ok:
            mrr = 1.0 / i
            break

    # Binary nDCG@k: relevance 1 if chunk id in relevant set.
    dcg = 0.0
    for i, ok in enumerate(in_top, start=1):
        if ok:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(total_rel, k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    ndcg_at_k = (dcg / idcg) if idcg > 0 else 0.0

    return RetrievalAccuracy(
        k=k,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        mean_reciprocal_rank=mrr,
        ndcg_at_k=ndcg_at_k,
        hit_at_k=relevant_in_top_k > 0,
        relevant_in_top_k=relevant_in_top_k,
        total_labelled_relevant=total_rel,
    )
