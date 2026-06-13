"""Prompt construction for retrieval-augmented answers."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from rag.retrieval.models import SearchResult


class PromptBuilder(BaseModel):
    """Build prompts from a question and retrieved chunks."""

    model_config = ConfigDict(frozen=True)

    system_instruction: str = (
        "You answer questions using only the provided context. "
        "If the context is insufficient, say that you do not know."
    )

    def build(self, question: str, results: Sequence[SearchResult]) -> str:
        context = self._format_context(results)
        return (
            f"{self.system_instruction}\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question.strip()}\n\n"
            "Answer with concise reasoning and cite sources using [source N]."
        )

    @staticmethod
    def _format_context(results: Sequence[SearchResult]) -> str:
        if not results:
            return "No relevant context was retrieved."

        sections: list[str] = []
        for index, result in enumerate(results, start=1):
            source = result.chunk.metadata.get("source", "unknown")
            sections.append(
                f"[source {index}] {source} "
                f"(chunk {result.chunk.index}, score {result.score:.3f})\n"
                f"{result.chunk.text}"
            )

        return "\n\n".join(sections)
