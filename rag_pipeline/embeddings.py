from __future__ import annotations

DEFAULT_MODEL = "all-MiniLM-L6-v2"

_models: dict[str, object] = {}


def _get_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for embeddings. "
            "Install it with: pip install sentence-transformers"
        )

    if model_name not in _models:
        _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]


def generate_embeddings(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
) -> list[list[float]]:
    if not texts:
        return []

    sanitized = [t if t and t.strip() else " " for t in texts]
    model = _get_model(model_name)
    embeddings = model.encode(sanitized, normalize_embeddings=True)
    return [vec.tolist() for vec in embeddings]


def generate_query_embedding(
    query: str,
    model_name: str = DEFAULT_MODEL,
) -> list[float]:
    if not query or not query.strip():
        raise ValueError("Query text must be a non-empty string")

    model = _get_model(model_name)
    embedding = model.encode(query, normalize_embeddings=True)
    return embedding.tolist()
