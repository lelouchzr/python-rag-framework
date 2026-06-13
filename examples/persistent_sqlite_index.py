"""Use SQLiteVectorStore for a persistent local index."""

from pathlib import Path

from rag import RAG, SQLiteVectorStore


def main() -> None:
    index_path = Path("examples/rag.sqlite")
    store = SQLiteVectorStore(index_path)
    rag = RAG(vector_store=store)
    rag.add_markdown("examples/sample_notes.md")

    for result in rag.retrieve("contrast particles", limit=2):
        print(f"{result.score:.3f}: {result.chunk.text}")

    store.close()


if __name__ == "__main__":
    main()
