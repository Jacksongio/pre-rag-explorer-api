from __future__ import annotations

import math
import re


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot_product += a * b
        norm_a += a * a
        norm_b += b * b
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0:
        return 0.0
    return dot_product / denom


def compute_bm25(
    query: str,
    documents: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    query_terms = [t for t in re.split(r"\W+", query.lower()) if t]
    doc_tokens = [
        [t for t in re.split(r"\W+", doc.lower()) if t] for doc in documents
    ]

    n = len(documents)
    if n == 0:
        return []

    total_tokens = sum(len(tokens) for tokens in doc_tokens)
    avg_doc_length = total_tokens / n if total_tokens > 0 else 1.0

    df: dict[str, int] = {}
    for term in query_terms:
        df[term] = sum(1 for tokens in doc_tokens if term in tokens)

    scores: list[float] = []
    for tokens in doc_tokens:
        tf_map: dict[str, int] = {}
        for t in tokens:
            tf_map[t] = tf_map.get(t, 0) + 1

        score = 0.0
        for term in query_terms:
            tf = tf_map.get(term, 0)
            doc_freq = df.get(term, 0)
            idf = math.log((n - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (len(tokens) / avg_doc_length))
            score += idf * numerator / denominator
        scores.append(score)

    return scores
