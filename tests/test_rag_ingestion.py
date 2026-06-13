from collections.abc import Sequence
from pathlib import Path

import pytest
from docx import Document as DocxWriter

from rag import (
    RAG,
    AnthropicLLM,
    CallableLLM,
    ChunkingConfig,
    HashingEmbeddingModel,
    LexicalReranker,
    NgramHashingEmbeddingModel,
    OpenAILLM,
    SQLiteVectorStore,
    TextChunker,
)
from rag.documents.loaders import OcrImageLoader
from rag.exceptions import (
    DocumentLoadError,
    EmptyDocumentError,
    LLMNotConfiguredError,
    UnsupportedFileTypeError,
)
from rag.models import Message


def test_add_text_stores_document_and_chunks() -> None:
    rag = RAG(chunker=TextChunker(ChunkingConfig(chunk_size=100, chunk_overlap=10)))

    chunks = rag.add_text(
        "Korean topic particles mark contrast. Subject particles mark new subjects.",
        source="notes",
        metadata={"language": "en"},
    )

    assert len(rag.documents) == 1
    assert len(chunks) == 1
    assert rag.chunks == tuple(chunks)
    assert chunks[0].metadata["source"] == "notes"
    assert chunks[0].metadata["language"] == "en"


def test_add_txt_loads_utf8_file(tmp_path: Path) -> None:
    file_path = tmp_path / "grammar.txt"
    file_path.write_text(
        "은/는 can introduce contrast.\n이/가 can mark focus.", encoding="utf-8"
    )

    rag = RAG(chunker=TextChunker(ChunkingConfig(chunk_size=100, chunk_overlap=10)))
    chunks = rag.add_txt(file_path)

    assert len(chunks) == 1
    assert chunks[0].text.startswith("은/는")
    assert chunks[0].metadata["filename"] == "grammar.txt"
    assert chunks[0].metadata["file_type"] == "txt"


def test_add_markdown_loads_markdown_file(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.md"
    file_path.write_text(
        "# Grammar\n\nUse examples when explaining particles.", encoding="utf-8"
    )

    rag = RAG(chunker=TextChunker(ChunkingConfig(chunk_size=100, chunk_overlap=10)))
    chunks = rag.add_markdown(file_path)

    assert len(chunks) == 1
    assert chunks[0].text.startswith("# Grammar")
    assert chunks[0].metadata["file_type"] == "md"


def test_add_docx_loads_word_document(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.docx"
    doc = DocxWriter()
    doc.add_paragraph("DOCX files are useful for source material.")
    doc.save(str(file_path))

    rag = RAG(chunker=TextChunker(ChunkingConfig(chunk_size=100, chunk_overlap=10)))
    chunks = rag.add_docx(file_path)

    assert len(chunks) == 1
    assert chunks[0].text == "DOCX files are useful for source material."
    assert chunks[0].metadata["file_type"] == "docx"


def test_add_pdf_loads_pdf_text(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.pdf"
    file_path.write_bytes(_minimal_pdf("PDF files can contain source text."))

    rag = RAG(chunker=TextChunker(ChunkingConfig(chunk_size=100, chunk_overlap=10)))
    chunks = rag.add_pdf(file_path)

    assert len(chunks) == 1
    assert "PDF files can contain source text." in chunks[0].text
    assert chunks[0].metadata["file_type"] == "pdf"


def test_chunker_creates_overlapping_chunks() -> None:
    text = " ".join(f"word{i}" for i in range(80))
    rag = RAG(chunker=TextChunker(ChunkingConfig(chunk_size=120, chunk_overlap=20)))

    chunks = rag.add_text(text)

    assert len(chunks) > 1
    assert chunks[0].end_char > chunks[1].start_char
    assert all(chunk.text for chunk in chunks)


def test_empty_text_is_rejected() -> None:
    rag = RAG()

    with pytest.raises(ValueError, match="Document text must not be empty"):
        rag.add_text("   ")


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text("\n", encoding="utf-8")
    rag = RAG()

    with pytest.raises(EmptyDocumentError, match="is empty"):
        rag.add_txt(file_path)


def test_wrong_loader_method_rejects_suffix(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.md"
    file_path.write_text("# Notes", encoding="utf-8")
    rag = RAG()

    with pytest.raises(UnsupportedFileTypeError, match="Expected one of .txt"):
        rag.add_txt(file_path)


def test_retrieve_returns_most_relevant_chunks() -> None:
    rag = RAG(chunker=TextChunker(ChunkingConfig(chunk_size=100, chunk_overlap=10)))
    rag.add_text(
        "Topic particles mark contrast in Korean grammar.",
        source="grammar",
    )
    rag.add_text(
        "Sourdough bread needs flour, water, and time.",
        source="bread",
    )

    results = rag.retrieve("Korean contrast particles", limit=1)

    assert len(results) == 1
    assert results[0].chunk.metadata["source"] == "grammar"
    assert results[0].score > 0


def test_retrieve_supports_metadata_filtering() -> None:
    rag = RAG(chunker=TextChunker(ChunkingConfig(chunk_size=100, chunk_overlap=10)))
    rag.add_text(
        "Topic particles mark contrast in Korean grammar.",
        source="public",
        metadata={"language": "en"},
    )
    rag.add_text(
        "Les particules coreennes demandent du contexte.",
        source="private",
        metadata={"language": "fr"},
    )

    results = rag.retrieve(
        "particules coreennes",
        limit=2,
        metadata_filter={"language": "fr"},
    )

    assert len(results) == 1
    assert results[0].chunk.metadata["source"] == "private"


def test_retrieve_supports_operator_metadata_filtering() -> None:
    rag = RAG(chunker=TextChunker(ChunkingConfig(chunk_size=100, chunk_overlap=10)))
    rag.add_text(
        "Topic particles mark contrast in Korean grammar.",
        source="old",
        metadata={"year": 2018, "language": "en", "title": "grammar notes"},
    )
    rag.add_text(
        "Topic particles can mark focus and contrast.",
        source="new",
        metadata={"year": 2024, "language": "en", "title": "advanced grammar"},
    )

    results = rag.retrieve(
        "topic particles contrast",
        limit=2,
        metadata_filter={
            "year": {"$gte": 2020},
            "language": {"$in": ["en", "fr"]},
            "title": {"$contains": "advanced"},
        },
    )

    assert len(results) == 1
    assert results[0].chunk.metadata["source"] == "new"


def test_ngram_embedding_model_returns_normalized_vectors() -> None:
    model = NgramHashingEmbeddingModel(dimensions=128)

    vector = model.embed(["contrastive topic particles"])[0]

    assert len(vector) == 128
    assert any(value != 0.0 for value in vector)


def test_lexical_reranker_can_prioritize_overlap() -> None:
    rag = RAG(
        chunker=TextChunker(ChunkingConfig(chunk_size=100, chunk_overlap=10)),
        reranker=LexicalReranker(vector_weight=0.0),
    )
    rag.add_text("alpha beta gamma", source="first")
    rag.add_text("topic particles contrast", source="second")

    results = rag.retrieve("topic contrast", limit=1)

    assert results[0].chunk.metadata["source"] == "second"


def test_sqlite_vector_store_persists_chunks(tmp_path: Path) -> None:
    store_path = tmp_path / "vectors.sqlite"
    store = SQLiteVectorStore(store_path)
    rag = RAG(
        chunker=TextChunker(ChunkingConfig(chunk_size=100, chunk_overlap=10)),
        vector_store=store,
    )
    rag.add_text("Topic particles mark contrast.", source="grammar")
    store.close()

    reopened = SQLiteVectorStore(store_path)
    query_embedding = HashingEmbeddingModel().embed(["Topic particles mark contrast."])[
        0
    ]
    results = reopened.search(query_embedding, limit=1)
    reopened.close()

    assert len(results) == 1
    assert results[0].chunk.metadata["source"] == "grammar"


def test_build_prompt_includes_retrieved_context() -> None:
    rag = RAG(chunker=TextChunker(ChunkingConfig(chunk_size=100, chunk_overlap=10)))
    rag.add_text("Topic particles mark contrast.", source="grammar.md")

    prompt = rag.build_prompt("What do topic particles do?")

    assert "Topic particles mark contrast." in prompt
    assert "What do topic particles do?" in prompt
    assert "[source 1]" in prompt


def test_ask_requires_configured_llm() -> None:
    rag = RAG()
    rag.add_text("Topic particles mark contrast.")

    with pytest.raises(LLMNotConfiguredError, match="No LLM is configured"):
        rag.ask("What do topic particles do?")


def test_ask_generates_answer_with_citations_and_history() -> None:
    prompts: list[str] = []

    def generate(prompt: str, history: Sequence[Message]) -> str:
        prompts.append(prompt)
        assert history == ()
        return "Topic particles can mark contrast."

    llm = CallableLLM(generate)
    rag = RAG(
        chunker=TextChunker(ChunkingConfig(chunk_size=100, chunk_overlap=10)),
        llm=llm,
    )
    rag.add_text("Topic particles mark contrast.", source="grammar.md")

    answer = rag.ask("What do topic particles do?")

    assert answer.text == "Topic particles can mark contrast."
    assert answer.citations[0].source == "grammar.md"
    assert "Topic particles mark contrast." in prompts[0]
    assert [message.role for message in rag.history] == ["user", "assistant"]


def test_stream_yields_parts_and_updates_history() -> None:
    def generate(prompt: str, history: Sequence[Message]) -> str:
        return "unused"

    def stream(prompt: str, history: Sequence[Message]) -> list[str]:
        assert "Topic particles mark contrast." in prompt
        assert history == ()
        return ["Topic ", "particles ", "contrast."]

    rag = RAG(
        chunker=TextChunker(ChunkingConfig(chunk_size=100, chunk_overlap=10)),
        llm=CallableLLM(generate, stream=stream),
    )
    rag.add_text("Topic particles mark contrast.", source="grammar.md")

    assert list(rag.stream("What do topic particles do?")) == [
        "Topic ",
        "particles ",
        "contrast.",
    ]
    assert rag.history[-1].content == "Topic particles contrast."


def test_clear_resets_corpus_and_history() -> None:
    rag = RAG(
        chunker=TextChunker(ChunkingConfig(chunk_size=100, chunk_overlap=10)),
        llm=CallableLLM(lambda prompt, history: "Answer."),
    )
    rag.add_text("Topic particles mark contrast.")
    rag.ask("What do topic particles do?")

    rag.clear()

    assert rag.documents == ()
    assert rag.chunks == ()
    assert rag.history == ()
    assert rag.retrieve("particles") == []


def test_openai_llm_adapter_uses_responses_api() -> None:
    client = _FakeOpenAIClient()
    llm = OpenAILLM(model="gpt-test", client=client)

    text = llm.generate("Prompt", history=[Message(role="user", content="Earlier")])

    assert text == "OpenAI answer"
    assert "Conversation history" in client.responses.last_input
    assert list(llm.stream("Prompt", history=[])) == ["Open", "AI"]


def test_anthropic_llm_adapter_uses_messages_api() -> None:
    client = _FakeAnthropicClient()
    llm = AnthropicLLM(model="claude-test", client=client)

    text = llm.generate(
        "Prompt",
        history=[Message(role="assistant", content="Earlier")],
    )

    assert text == "Anthropic answer"
    assert client.messages.last_messages[-1]["content"] == "Prompt"
    assert list(llm.stream("Prompt", history=[])) == ["Anthropic", " stream"]


def test_ocr_loader_reports_missing_or_failed_ocr_dependencies(tmp_path: Path) -> None:
    image_path = tmp_path / "scan.png"
    image_path.write_bytes(b"not a real image")

    with pytest.raises(DocumentLoadError):
        OcrImageLoader().load(image_path)


class _FakeOpenAIResponse:
    output_text = "OpenAI answer"


class _FakeOpenAIEvent:
    def __init__(self, delta: str) -> None:
        self.type = "response.output_text.delta"
        self.delta = delta


class _FakeOpenAIStream:
    def __enter__(self) -> list[_FakeOpenAIEvent]:
        return [_FakeOpenAIEvent("Open"), _FakeOpenAIEvent("AI")]

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        return None


class _FakeOpenAIResponses:
    def __init__(self) -> None:
        self.last_input = ""

    def create(self, *, model: str, input: str) -> _FakeOpenAIResponse:
        self.last_input = input
        return _FakeOpenAIResponse()

    def stream(self, *, model: str, input: str) -> _FakeOpenAIStream:
        self.last_input = input
        return _FakeOpenAIStream()


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = _FakeOpenAIResponses()


class _FakeAnthropicBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeAnthropicResponse:
    content = [_FakeAnthropicBlock("Anthropic answer")]


class _FakeAnthropicStream:
    text_stream = ["Anthropic", " stream"]

    def __enter__(self) -> "_FakeAnthropicStream":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        return None


class _FakeAnthropicMessages:
    def __init__(self) -> None:
        self.last_messages: Sequence[dict[str, str]] = []

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: Sequence[dict[str, str]],
    ) -> _FakeAnthropicResponse:
        self.last_messages = messages
        return _FakeAnthropicResponse()

    def stream(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: Sequence[dict[str, str]],
    ) -> _FakeAnthropicStream:
        self.last_messages = messages
        return _FakeAnthropicStream()


class _FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = _FakeAnthropicMessages()


def _minimal_pdf(text: str) -> bytes:
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            b"<< /Length "
            + str(len(f"BT /F1 18 Tf 72 720 Td ({escaped_text}) Tj ET")).encode()
            + b" >>\nstream\n"
            + f"BT /F1 18 Tf 72 720 Td ({escaped_text}) Tj ET".encode()
            + b"\nendstream"
        ),
    ]
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]

    for index, obj in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{index} 0 obj\n".encode())
        content.extend(obj)
        content.extend(b"\nendobj\n")

    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode())
    content.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return bytes(content)
