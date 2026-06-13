"""Use the optional OpenAI adapter.

Install dependencies first:

    uv sync --extra openai

Set `OPENAI_API_KEY` in your environment before running the example.
"""

from rag import RAG, OpenAILLM


def main() -> None:
    rag = RAG(llm=OpenAILLM(model="gpt-4.1-mini"))
    rag.add_text(
        "RAG combines retrieval with generation so answers can be grounded.",
        source="inline-note",
    )

    answer = rag.ask("What does RAG combine?")
    print(answer.text)


if __name__ == "__main__":
    main()
