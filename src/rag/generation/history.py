"""Conversation history state."""

from rag.generation.models import Message


class ConversationHistory:
    """Stores conversation messages."""

    def __init__(self) -> None:
        self._messages: list[Message] = []

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(self._messages)

    def append_user(self, content: str) -> None:
        self._messages.append(Message(role="user", content=content))

    def append_assistant(self, content: str) -> None:
        self._messages.append(Message(role="assistant", content=content))

    def clear(self) -> None:
        self._messages.clear()
