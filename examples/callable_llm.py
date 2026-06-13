"""Use a plain Python callable as the LLM integration."""

from collections.abc import Sequence

from rag import RAG, CallableLLM, Message


def generate(prompt: str, history: Sequence[Message]) -> str:
    """Tiny demo LLM used for local examples and tests."""

    if "topic particles" in prompt.lower():
        return "Topic particles often mark contrast or the topic of discussion."
    return "The provided context is insufficient."


def main() -> None:
    rag = RAG(llm=CallableLLM(generate))
    rag.add_text(
        "In Korean, 은/는 are topic particles and often introduce contrast.",
        source="inline-note",
    )

    answer = rag.ask("What do topic particles do?")
    print(answer.text)
    print(answer.citations[0].source)


if __name__ == "__main__":
    main()
