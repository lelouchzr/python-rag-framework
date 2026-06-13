"""LLM interfaces and adapters."""

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Protocol

from rag.generation.models import Message

GenerateCallable = Callable[[str, Sequence[Message]], str]
StreamCallable = Callable[[str, Sequence[Message]], Iterable[str]]


class LLM(Protocol):
    """Generates text from a prompt."""

    def generate(self, prompt: str, *, history: Sequence[Message]) -> str:
        """Generate a complete response."""

    def stream(self, prompt: str, *, history: Sequence[Message]) -> Iterable[str]:
        """Stream response chunks."""


class TokenStream(Iterator[str], Protocol):
    """Iterator returned by streaming LLM implementations."""


class CallableLLM:
    """Adapter that turns plain Python callables into an LLM integration."""

    def __init__(
        self,
        generate: GenerateCallable,
        *,
        stream: StreamCallable | None = None,
    ) -> None:
        self._generate = generate
        self._stream = stream

    def generate(self, prompt: str, *, history: Sequence[Message]) -> str:
        return self._generate(prompt, history)

    def stream(self, prompt: str, *, history: Sequence[Message]) -> Iterable[str]:
        if self._stream is not None:
            return self._stream(prompt, history)
        return iter([self.generate(prompt, history=history)])
