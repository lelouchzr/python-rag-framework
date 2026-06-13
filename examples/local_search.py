"""Run local retrieval over a text file without an LLM."""

from pathlib import Path

from rag import RAG


def main() -> None:
    path = Path("examples/sample_notes.md")
    rag = RAG()
    rag.add_markdown(path)

    for result in rag.retrieve("What do topic particles mark?", limit=3):
        print(f"{result.score:.3f} {result.chunk.metadata['source']}")
        print(result.chunk.text)
        print()


if __name__ == "__main__":
    main()
