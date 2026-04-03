# Tests for `rag_pipeline`

These files are **pytest** tests for the Python package at `rag_pipeline/`. They assert inputs and outputs of parsing, chunking, similarity, retrieval, optional accuracy metrics, and embedding helpers (with the real model mocked).

## How to run

From the **repository root** (not from inside `tests/`):

```bash
cd /path/to/pre-rag-explorer-api
pip install -r requirements-dev.txt
pytest tests/ -v
```

Equivalent:

```bash
pytest
```

If your shell is already in `tests/`, use `pytest .` instead of `pytest tests/`.

Configuration lives in `pytest.ini` (`testpaths = tests`, `pythonpath = .` so `import rag_pipeline` resolves).

---

## Layout

| File | What it covers |
|------|----------------|
| `conftest.py` | Shared pytest fixtures (`utc_now`, `sample_chunks`). |
| `helpers.py` | `make_vector_collection()` — builds a minimal `VectorCollection` for search tests. |
| `test_parsing.py` | `parse_file()` for `.txt`, `.md`, unknown extensions, `.csv`, and `.pdf` (PDF via PyMuPDF). |
| `test_chunking.py` | `chunk_text()` for every `ChunkingMethod`, stats on `ChunkResult`, and empty input. |
| `test_similarity.py` | `cosine_similarity()` and `compute_bm25()`. |
| `test_search.py` | `search_dense`, `search_sparse`, `search_hybrid`, merged `search()`, and `SearchResponse.accuracy` when `relevant_chunk_ids` is set. |
| `test_evaluation.py` | `compute_retrieval_accuracy()` — precision/recall@k, MRR, nDCG, hits, empty rankings. |
| `test_embeddings.py` | `generate_embeddings` / `generate_query_embedding` with `_get_model` patched (no Hugging Face download). |

---

## What each module is checking

### `test_parsing.py`

- Text and markdown are returned unchanged (bytes vs str).
- Unknown file extensions fall back to plain-text handling.
- CSV rows are flattened into lines of joined column values.
- PDF: builds a tiny in-memory PDF with PyMuPDF, then checks extracted text contains the inserted phrase.

### `test_chunking.py`

- **Fixed** and **token** windows use **non-zero overlap** in assertions, because `chunk_text()` treats `overlap=0` as “use default” (`or 200` / `or 50`).
- **Recursive**, **sentence**, and **semantic** paths produce non-empty, plausible splits.
- Empty string yields empty chunk list and zero stats.

### `test_similarity.py`

- Cosine: identical vectors, orthogonal vectors, zero vector.
- BM25: empty corpus returns an empty score list; a query matching one document scores it above an unrelated document.

### `test_search.py`

- **Dense**: cosine ordering and which chunk ranks first with controlled 3-D vectors.
- **Sparse**: BM25 ranks the chunk that shares query terms higher.
- **Hybrid**: every result uses the hybrid retrieval method.
- **Merged `search`**: multiple methods combine, slice length is `top_k * len(methods)`, `accuracy` is `None` without labels.
- **Labels**: with `relevant_chunk_ids`, `accuracy` is present and reflects a correct top hit.

### `test_evaluation.py`

- Ideal list (relevant first) → full recall, MRR 1, nDCG 1 (for a single relevant id).
- Relevant at rank 2 → MRR 0.5.
- No relevant ids in the list → zero metrics, `hit_at_k` false.
- Empty `results` with non-empty labels → zero `k`, no hit.
- Multiple gold ids → recall is “found / total labelled”.

### `test_embeddings.py`

- Empty embedding list returns `[]`.
- Blank query raises `ValueError`.
- Blank strings in batch are sanitized to a single space before `encode`.
- Query path returns a plain `list[float]` from the mocked encoder.

---

## Related docs

- Manual CLI-style exercises: `rag_pipeline/CLI_TESTING.md`.
- Dev dependencies: `requirements-dev.txt`.
