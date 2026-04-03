#!/usr/bin/env python3
"""RAG Pipeline CLI — upload documents, then query an LLM with retrieval-augmented context."""

from __future__ import annotations

import argparse
import os
import sys
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
# Document ingestion
# ---------------------------------------------------------------------------

def ingest_file(
    file_path: str,
    chunk_method: ChunkingMethod,
    chunk_params: ChunkParams,
    embedding_model: str,
) -> VectorCollection:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    content = path.read_bytes()
    text = parse_file(path.name, content)
    print(f"  Parsed {path.name} ({len(text)} chars)")

    result = chunk_text(text, chunk_method, chunk_params)
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
            chunk_method=chunk_method,
        )
        for i, chunk in enumerate(result.chunks)
    ]

    print(f"  Generating embeddings with {embedding_model}...")
    vectors = generate_embeddings([c.text for c in chunks], model_name=embedding_model)
    print(f"  Generated {len(vectors)} embeddings")

    return VectorCollection(
        id=collection_id,
        name=path.stem,
        chunk_method=chunk_method,
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
    retrieval_method: RetrievalMethod,
    top_k: int,
    llm_backend: str,
    llm_model: str,
) -> None:
    query_embedding = generate_query_embedding(query, model_name=embedding_model)

    response = search(
        query=query,
        query_embedding=query_embedding,
        collections=collections,
        methods=[retrieval_method],
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
        description="RAG Pipeline CLI — upload documents and ask questions with retrieval-augmented generation.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Document files to ingest (pdf, txt, md, csv). You can also add files interactively.",
    )
    parser.add_argument(
        "--chunk-method",
        choices=[m.value for m in ChunkingMethod],
        default="recursive",
        help="Chunking strategy (default: recursive)",
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
        choices=[m.value for m in RetrievalMethod],
        default="dense",
        help="Retrieval method (default: dense)",
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

    chunk_method = ChunkingMethod(args.chunk_method)
    chunk_params = ChunkParams(chunk_size=args.chunk_size, overlap=args.overlap)
    retrieval_method = RetrievalMethod(args.retrieval)

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
