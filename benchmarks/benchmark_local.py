"""Lightweight local benchmarks for the RAG framework."""

import argparse
import json
import time
import tracemalloc
from statistics import mean

from rag import RAG, ChunkingConfig, HashingEmbeddingModel, TextChunker


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark local RAG operations.")
    parser.add_argument("--documents", type=int, default=200)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    documents = _make_documents(args.documents)
    queries = _make_queries(args.queries)

    embedding_model = HashingEmbeddingModel()
    chunker = TextChunker(ChunkingConfig(chunk_size=500, chunk_overlap=50))
    rag = RAG(chunker=chunker, embedding_model=embedding_model)

    tracemalloc.start()

    embedding_started = time.perf_counter()
    embedding_model.embed(documents)
    embedding_seconds = time.perf_counter() - embedding_started

    indexing_started = time.perf_counter()
    for index, document in enumerate(documents):
        rag.add_text(document, source=f"synthetic-{index}")
    indexing_seconds = time.perf_counter() - indexing_started

    retrieval_latencies: list[float] = []
    for query in queries:
        retrieval_started = time.perf_counter()
        rag.retrieve(query, limit=5)
        retrieval_latencies.append(time.perf_counter() - retrieval_started)

    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    payload = {
        "documents": args.documents,
        "queries": args.queries,
        "chunks": len(rag.chunks),
        "embedding_seconds": embedding_seconds,
        "indexing_seconds": indexing_seconds,
        "mean_retrieval_seconds": mean(retrieval_latencies),
        "max_retrieval_seconds": max(retrieval_latencies),
        "current_memory_bytes": current_memory,
        "peak_memory_bytes": peak_memory,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")

    return 0


def _make_documents(count: int) -> list[str]:
    topics = [
        "topic particles contrast grammar",
        "semantic search retrieval embeddings",
        "python packaging developer experience",
        "sqlite vector store persistence",
    ]
    return [
        (
            f"Document {index}. {topics[index % len(topics)]}. "
            "This synthetic text is used for repeatable local benchmarking. "
            "It intentionally avoids external services."
        )
        for index in range(count)
    ]


def _make_queries(count: int) -> list[str]:
    seeds = [
        "topic particles",
        "semantic retrieval",
        "python package",
        "sqlite persistence",
    ]
    return [seeds[index % len(seeds)] for index in range(count)]


if __name__ == "__main__":
    raise SystemExit(main())
