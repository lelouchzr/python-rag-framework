"""Provider-specific LLM adapters."""

import importlib
from collections.abc import Iterable, Sequence
from typing import Protocol, cast

from rag.exceptions import LLMNotConfiguredError
from rag.generation.models import Message


class _OpenAIResponseStream(Protocol):
    def __enter__(self) -> Iterable[object]: ...
    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None: ...


class _OpenAIResponsesClient(Protocol):
    def create(self, *, model: str, input: str) -> object: ...
    def stream(self, *, model: str, input: str) -> _OpenAIResponseStream: ...


class _OpenAIClient(Protocol):
    responses: _OpenAIResponsesClient


class OpenAILLM:
    """LLM adapter for OpenAI's Responses API."""

    def __init__(self, *, model: str, client: object | None = None) -> None:
        self.model = model
        self._client = cast(_OpenAIClient, client) if client else _load_openai_client()

    def generate(self, prompt: str, *, history: Sequence[Message]) -> str:
        response = self._client.responses.create(
            model=self.model,
            input=_prompt_with_history(prompt, history),
        )
        text = getattr(response, "output_text", None)
        if isinstance(text, str):
            return text
        raise LLMNotConfiguredError("OpenAI response did not include output_text.")

    def stream(self, prompt: str, *, history: Sequence[Message]) -> Iterable[str]:
        with self._client.responses.stream(
            model=self.model,
            input=_prompt_with_history(prompt, history),
        ) as stream:
            for event in stream:
                event_type = getattr(event, "type", None)
                delta = getattr(event, "delta", None)
                if event_type == "response.output_text.delta" and isinstance(
                    delta,
                    str,
                ):
                    yield delta


class _AnthropicMessageBlock(Protocol):
    text: str


class _AnthropicMessageResponse(Protocol):
    content: Sequence[_AnthropicMessageBlock]


class _AnthropicTextStream(Protocol):
    text_stream: Iterable[str]

    def __enter__(self) -> "_AnthropicTextStream": ...
    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None: ...


class _AnthropicMessagesClient(Protocol):
    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: Sequence[dict[str, str]],
    ) -> _AnthropicMessageResponse: ...

    def stream(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: Sequence[dict[str, str]],
    ) -> _AnthropicTextStream: ...


class _AnthropicClient(Protocol):
    messages: _AnthropicMessagesClient


class AnthropicLLM:
    """LLM adapter for Anthropic's Messages API."""

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int = 1_024,
        client: object | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = (
            cast(_AnthropicClient, client) if client else _load_anthropic_client()
        )

    def generate(self, prompt: str, *, history: Sequence[Message]) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system="Answer using the supplied RAG prompt.",
            messages=_anthropic_messages(prompt, history),
        )
        return "".join(block.text for block in response.content)

    def stream(self, prompt: str, *, history: Sequence[Message]) -> Iterable[str]:
        with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system="Answer using the supplied RAG prompt.",
            messages=_anthropic_messages(prompt, history),
        ) as stream:
            yield from stream.text_stream


def _prompt_with_history(prompt: str, history: Sequence[Message]) -> str:
    if not history:
        return prompt
    history_text = "\n".join(
        f"{message.role}: {message.content}" for message in history
    )
    return f"Conversation history:\n{history_text}\n\n{prompt}"


def _anthropic_messages(
    prompt: str,
    history: Sequence[Message],
) -> list[dict[str, str]]:
    messages = [
        {"role": message.role, "content": message.content}
        for message in history
        if message.role in {"user", "assistant"}
    ]
    messages.append({"role": "user", "content": prompt})
    return messages


def _load_openai_client() -> _OpenAIClient:
    try:
        openai_module = importlib.import_module("openai")
    except ImportError as exc:
        msg = "Install the 'openai' package to use OpenAILLM."
        raise LLMNotConfiguredError(msg) from exc

    client_factory = getattr(openai_module, "OpenAI", None)
    if client_factory is None:
        msg = "The installed openai package has no OpenAI client."
        raise LLMNotConfiguredError(msg)
    return cast(_OpenAIClient, client_factory())


def _load_anthropic_client() -> _AnthropicClient:
    try:
        anthropic_module = importlib.import_module("anthropic")
    except ImportError as exc:
        msg = "Install the 'anthropic' package to use AnthropicLLM."
        raise LLMNotConfiguredError(msg) from exc

    client_factory = getattr(anthropic_module, "Anthropic", None)
    if client_factory is None:
        msg = "The installed anthropic package has no Anthropic client."
        raise LLMNotConfiguredError(msg)
    return cast(_AnthropicClient, client_factory())
