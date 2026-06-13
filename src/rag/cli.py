"""Command-line interface for local RAG operations."""

import argparse
import json
import sys
from pathlib import Path

from rag import RAG
from rag._version import get_version
from rag.documents.chunking import ChunkingConfig, TextChunker
from rag.exceptions import RAGError


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.handler(args))
    except RAGError as exc:
        print(f"rag: error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"rag: error: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Local utilities for the Python RAG framework.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version", help="Print the package version.")
    version_parser.set_defaults(handler=_handle_version)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Load a document and print basic metadata.",
    )
    inspect_parser.add_argument("path", type=Path)
    inspect_parser.add_argument(
        "--json", action="store_true", help="Print JSON output."
    )
    inspect_parser.set_defaults(handler=_handle_inspect)

    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Load a document and print generated chunks.",
    )
    _add_chunking_options(chunk_parser)
    chunk_parser.add_argument("path", type=Path)
    chunk_parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output.",
    )
    chunk_parser.set_defaults(handler=_handle_chunk)

    search_parser = subparsers.add_parser(
        "search",
        help="Index one document and run local semantic search.",
    )
    _add_chunking_options(search_parser)
    search_parser.add_argument("path", type=Path)
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=3)
    search_parser.add_argument("--json", action="store_true", help="Print JSON output.")
    search_parser.set_defaults(handler=_handle_search)

    return parser


def _add_chunking_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--chunk-size", type=int, default=1_000)
    parser.add_argument("--chunk-overlap", type=int, default=150)


def _handle_version(args: argparse.Namespace) -> int:
    print(get_version())
    return 0


def _handle_inspect(args: argparse.Namespace) -> int:
    rag = RAG()
    chunks = rag.add_path(args.path)
    document = rag.documents[0]
    payload = {
        "path": str(args.path),
        "document_id": document.id,
        "characters": len(document.text),
        "chunks": len(chunks),
        "metadata": document.metadata,
    }
    _print_payload(payload, json_output=bool(args.json))
    return 0


def _handle_chunk(args: argparse.Namespace) -> int:
    rag = _rag_with_chunker(args)
    chunks = rag.add_path(args.path)
    payload = [
        {
            "index": chunk.index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]
    _print_payload(payload, json_output=bool(args.json))
    return 0


def _handle_search(args: argparse.Namespace) -> int:
    rag = _rag_with_chunker(args)
    rag.add_path(args.path)
    results = rag.retrieve(args.query, limit=args.limit)
    payload = [
        {
            "score": result.score,
            "text": result.chunk.text,
            "metadata": result.chunk.metadata,
        }
        for result in results
    ]
    _print_payload(payload, json_output=bool(args.json))
    return 0


def _rag_with_chunker(args: argparse.Namespace) -> RAG:
    config = ChunkingConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    return RAG(chunker=TextChunker(config))


def _print_payload(payload: object, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if isinstance(payload, list):
        for item in payload:
            print(json.dumps(item, ensure_ascii=False))
        return

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
