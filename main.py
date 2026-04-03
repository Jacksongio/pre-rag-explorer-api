#!/usr/bin/env python3
"""RAG Pipeline CLI — upload documents, then query an LLM with retrieval-augmented context."""

from __future__ import annotations

import argparse
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from rag_pipeline.chunking import chunk_text
from rag_pipeline.embeddings import generate_embeddings, generate_query_embedding
from rag_pipeline.models import (
    Chunk,
    ChunkingMethod,
    ChunkParams,
    RetrievalMethod,
    VectorCollection,
)
from rag_pipeline.parsing import parse_file
from rag_pipeline.search import search


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def _ask_openai(prompt: str, model: str) -> str:
    import openai

    client = openai.OpenAI()  # uses OPENAI_API_KEY env var
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


def _ask_ollama(prompt: str, model: str) -> str:
    import urllib.request
    import json

    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    url = f"{base_url}/api/generate"
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read().decode())
    return body.get("response", "")


def ask_llm(prompt: str, *, backend: str, model: str) -> str:
    if backend == "openai":
        return _ask_openai(prompt, model)
    elif backend == "ollama":
        return _ask_ollama(prompt, model)
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")


# ---------------------------------------------------------------------------
# Automatic strategy selection
# ---------------------------------------------------------------------------

def choose_chunk_method(file_path: str, text: str) -> ChunkingMethod:
    """Pick a chunking strategy from basic document heuristics."""
    suffix = Path(file_path).suffix.lower()
    stripped = text.strip()
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    paragraph_count = len([p for p in re.split(r"\n\s*\n", text) if p.strip()])
    sentence_count = len(re.findall(r"[.!?]+", text))
    avg_line_length = (
        sum(len(line) for line in nonempty_lines) / len(nonempty_lines)
        if nonempty_lines
        else 0.0
    )

    if suffix == ".csv":
        return ChunkingMethod.FIXED
    if suffix in {".md", ".markdown"}:
        return ChunkingMethod.RECURSIVE
    if len(stripped) < 600:
        return ChunkingMethod.SENTENCE if sentence_count >= 3 else ChunkingMethod.FIXED
    if paragraph_count >= 4 or "#" in text or avg_line_length < 120:
        return ChunkingMethod.RECURSIVE
    if len(stripped) > 12000:
        return ChunkingMethod.TOKEN
    return ChunkingMethod.RECURSIVE


def choose_retrieval_method(
    query: str,
    collections: list[VectorCollection],
) -> RetrievalMethod:
    """Pick retrieval based on query style and collection size."""
    query_terms = [t for t in re.split(r"\W+", query.lower()) if t]
    total_chunks = sum(collection.chunk_count for collection in collections)
    has_exact_match_intent = (
        len(query_terms) <= 3
        or any(char.isdigit() for char in query)
        or '"' in query
        or ":" in query
    )

    if total_chunks <= 3:
        return RetrievalMethod.DENSE
    if has_exact_match_intent:
        return RetrievalMethod.HYBRID if total_chunks >= 8 else RetrievalMethod.SPARSE
    return RetrievalMethod.HYBRID if total_chunks >= 5 else RetrievalMethod.DENSE


# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------

def ingest_file(
    file_path: str,
    chunk_method: ChunkingMethod | str,
    chunk_params: ChunkParams,
    embedding_model: str,
) -> VectorCollection:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    content = path.read_bytes()
    text = parse_file(path.name, content)
    print(f"  Parsed {path.name} ({len(text)} chars)")

    selected_chunk_method = (
        choose_chunk_method(path.name, text)
        if str(chunk_method) == "auto"
        else ChunkingMethod(chunk_method)
    )
    if str(chunk_method) == "auto":
        print(f"  Auto-selected chunk method: {selected_chunk_method.value}")

    result = chunk_text(text, selected_chunk_method, chunk_params)
    print(f"  Chunked into {result.stats.count} chunks (avg {result.stats.avg_size:.0f} chars)")

    file_id = str(uuid.uuid4())
    collection_id = str(uuid.uuid4())
    chunks = [
        Chunk(
            id=str(uuid.uuid4()),
            text=chunk,
            index=i,
            source_file_id=file_id,
            source_file_name=path.name,
            chunk_method=selected_chunk_method,
        )
        for i, chunk in enumerate(result.chunks)
    ]

    print(f"  Generating embeddings with {embedding_model}...")
    vectors = generate_embeddings([c.text for c in chunks], model_name=embedding_model)
    print(f"  Generated {len(vectors)} embeddings")

    return VectorCollection(
        id=collection_id,
        name=path.stem,
        chunk_method=selected_chunk_method,
        source_file_id=file_id,
        source_file_name=path.name,
        chunk_count=len(chunks),
        params=chunk_params,
        created_at=datetime.now(timezone.utc).isoformat(),
        chunks=chunks,
        vectors=vectors,
        embedding_model=embedding_model,
    )


# ---------------------------------------------------------------------------
# Query loop
# ---------------------------------------------------------------------------

def build_prompt(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return (
        "Use the following context to answer the question. "
        "If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def run_query(
    query: str,
    collections: list[VectorCollection],
    *,
    embedding_model: str,
    retrieval_method: RetrievalMethod | str,
    top_k: int,
    llm_backend: str,
    llm_model: str,
) -> None:
    query_embedding = generate_query_embedding(query, model_name=embedding_model)
    selected_retrieval_method = (
        choose_retrieval_method(query, collections)
        if str(retrieval_method) == "auto"
        else RetrievalMethod(retrieval_method)
    )

    if str(retrieval_method) == "auto":
        print(f"  Auto-selected retrieval method: {selected_retrieval_method.value}")

    response = search(
        query=query,
        query_embedding=query_embedding,
        collections=collections,
        methods=[selected_retrieval_method],
        top_k=top_k,
    )

    if not response.results:
        print("No matching chunks found.")
        return

    print(f"\n  Retrieved {len(response.results)} chunks:")
    for i, r in enumerate(response.results, 1):
        preview = r.chunk.text[:80].replace("\n", " ")
        print(f"    [{i}] (score={r.score:.4f}) {preview}...")

    context_chunks = [r.chunk.text for r in response.results]
    prompt = build_prompt(query, context_chunks)

    print(f"\n  Asking {llm_backend}/{llm_model}...")
    answer = ask_llm(prompt, backend=llm_backend, model=llm_model)
    print(f"\n{answer}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RAG Pipeline CLI — upload documents and ask questions with "
            "retrieval-augmented generation. Defaults now auto-pick chunking "
            "per document and retrieval per query."
        ),
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Document files to ingest (pdf, txt, md, csv). You can also add files interactively.",
    )
    parser.add_argument(
        "--chunk-method",
        choices=["auto", *[m.value for m in ChunkingMethod]],
        default="auto",
        help="Chunking strategy (default: auto)",
    )
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in characters (default: 1000)")
    parser.add_argument("--overlap", type=int, default=200, help="Chunk overlap (default: 200)")
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model for embeddings (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--retrieval",
        choices=["auto", *[m.value for m in RetrievalMethod]],
        default="auto",
        help="Retrieval method (default: auto)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    parser.add_argument(
        "--llm-backend",
        choices=["openai", "ollama"],
        default="ollama",
        help="LLM backend (default: ollama)",
    )
    parser.add_argument(
        "--llm-model",
        default="llama3",
        help="LLM model name (default: llama3 for ollama, gpt-4o for openai)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Default model per backend
    if args.llm_model == "llama3" and args.llm_backend == "openai":
        args.llm_model = "gpt-4o"

    chunk_method: ChunkingMethod | str = args.chunk_method
    chunk_params = ChunkParams(chunk_size=args.chunk_size, overlap=args.overlap)
    retrieval_method: RetrievalMethod | str = args.retrieval

    collections: list[VectorCollection] = []

    # Ingest files from CLI args
    for fp in args.files:
        print(f"\nIngesting: {fp}")
        col = ingest_file(fp, chunk_method, chunk_params, args.embedding_model)
        collections.append(col)

    print("\n=== RAG Pipeline CLI ===")
    print("Commands:")
    print("  /add <file>   — Add a document")
    print("  /collections  — List loaded collections")
    print("  /quit         — Exit")
    print("  (anything else is treated as a query)\n")

    while True:
        try:
            user_input = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye!")
            break

        if user_input.lower().startswith("/add "):
            fp = user_input[5:].strip()
            if not fp:
                print("Usage: /add <file_path>")
                continue
            try:
                print(f"\nIngesting: {fp}")
                col = ingest_file(fp, chunk_method, chunk_params, args.embedding_model)
                collections.append(col)
                print(f"Added collection '{col.name}' ({col.chunk_count} chunks)\n")
            except Exception as e:
                print(f"Error ingesting file: {e}\n")
            continue

        if user_input.lower() == "/collections":
            if not collections:
                print("No collections loaded.\n")
            else:
                for c in collections:
                    print(f"  - {c.name} ({c.source_file_name}) — {c.chunk_count} chunks")
                print()
            continue

        # Treat as a query
        if not collections:
            print("No documents loaded. Use /add <file> or pass files as arguments.\n")
            continue

        try:
            run_query(
                user_input,
                collections,
                embedding_model=args.embedding_model,
                retrieval_method=retrieval_method,
                top_k=args.top_k,
                llm_backend=args.llm_backend,
                llm_model=args.llm_model,
            )
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
