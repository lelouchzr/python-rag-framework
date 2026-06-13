"""High-level facade for building RAG applications."""

from collections.abc import Iterator
from pathlib import Path

from rag.documents.chunking import TextChunker
from rag.documents.corpus import Corpus
from rag.documents.loaders import FileLoader
from rag.documents.models import Chunk, Document, Metadata
from rag.embeddings import EmbeddingModel, HashingEmbeddingModel
from rag.exceptions import LLMNotConfiguredError, UnsupportedFileTypeError
from rag.generation.citations import CitationBuilder
from rag.generation.history import ConversationHistory
from rag.generation.llms import LLM
from rag.generation.models import Answer, Message
from rag.generation.prompts import PromptBuilder
from rag.indexes import InMemoryVectorStore, VectorStore
from rag.retrieval.filters import MetadataFilter
from rag.retrieval.models import SearchResult
from rag.retrieval.rerankers import Reranker
from rag.retrieval.retriever import Retriever


class RAG:
    """User-facing entry point for ingestion, retrieval, and answer generation."""

    def __init__(
        self,
        *,
        chunker: TextChunker | None = None,
        file_loader: FileLoader | None = None,
        embedding_model: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
        prompt_builder: PromptBuilder | None = None,
        llm: LLM | None = None,
    ) -> None:
        self._chunker = chunker or TextChunker()
        self._file_loader = file_loader or FileLoader()
        self._corpus = Corpus()
        self._retriever = Retriever(
            embedding_model=(
                embedding_model
                if embedding_model is not None
                else HashingEmbeddingModel()
            ),
            vector_store=vector_store
            if vector_store is not None
            else InMemoryVectorStore(),
            reranker=reranker,
        )
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._citation_builder = CitationBuilder()
        self._history = ConversationHistory()
        self._llm = llm

    @property
    def documents(self) -> tuple[Document, ...]:
        return self._corpus.documents

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return self._corpus.chunks

    @property
    def history(self) -> tuple[Message, ...]:
        return self._history.messages

    def add_text(
        self,
        text: str,
        *,
        source: str | None = None,
        metadata: Metadata | None = None,
    ) -> list[Chunk]:
        document_metadata = dict(metadata or {})
        if source is not None:
            document_metadata["source"] = source

        document = Document(text=text, metadata=document_metadata)
        return self._add_document(document)

    def add_txt(self, path: str | Path) -> list[Chunk]:
        return self._add_file(path, expected_suffixes={".txt"})

    def add_markdown(self, path: str | Path) -> list[Chunk]:
        return self._add_file(path, expected_suffixes={".md", ".markdown"})

    def add_pdf(self, path: str | Path) -> list[Chunk]:
        return self._add_file(path, expected_suffixes={".pdf"})

    def add_docx(self, path: str | Path) -> list[Chunk]:
        return self._add_file(path, expected_suffixes={".docx"})

    def add_image(self, path: str | Path) -> list[Chunk]:
        return self._add_file(
            path,
            expected_suffixes={".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"},
        )

    def add_path(self, path: str | Path) -> list[Chunk]:
        document = self._file_loader.load(path)
        return self._add_document(document)

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 4,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        return self._retriever.retrieve(
            query,
            limit=limit,
            metadata_filter=metadata_filter,
        )

    def build_prompt(
        self,
        question: str,
        *,
        limit: int = 4,
        metadata_filter: MetadataFilter | None = None,
    ) -> str:
        results = self.retrieve(
            question,
            limit=limit,
            metadata_filter=metadata_filter,
        )
        return self._prompt_builder.build(question, results)

    def ask(
        self,
        question: str,
        *,
        limit: int = 4,
        metadata_filter: MetadataFilter | None = None,
    ) -> Answer:
        llm = self._require_llm()
        results = self.retrieve(
            question,
            limit=limit,
            metadata_filter=metadata_filter,
        )
        prompt = self._prompt_builder.build(question, results)
        text = llm.generate(prompt, history=self.history)

        self._history.append_user(question)
        self._history.append_assistant(text)

        return Answer(
            text=text,
            question=question,
            citations=self._citation_builder.build(results),
            results=results,
        )

    def stream(
        self,
        question: str,
        *,
        limit: int = 4,
        metadata_filter: MetadataFilter | None = None,
    ) -> Iterator[str]:
        llm = self._require_llm()
        results = self.retrieve(
            question,
            limit=limit,
            metadata_filter=metadata_filter,
        )
        prompt = self._prompt_builder.build(question, results)

        parts: list[str] = []
        for part in llm.stream(prompt, history=self.history):
            parts.append(part)
            yield part

        answer_text = "".join(parts)
        self._history.append_user(question)
        self._history.append_assistant(answer_text)

    def clear_history(self) -> None:
        self._history.clear()

    def clear(self) -> None:
        self._corpus.clear()
        self._history.clear()
        self._retriever.clear()

    def _add_file(
        self, path: str | Path, *, expected_suffixes: set[str]
    ) -> list[Chunk]:
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix not in expected_suffixes:
            expected = ", ".join(sorted(expected_suffixes))
            msg = f"Expected one of {expected}, got '{suffix}'."
            raise UnsupportedFileTypeError(msg)

        document = self._file_loader.load(file_path)
        return self._add_document(document)

    def _add_document(self, document: Document) -> list[Chunk]:
        chunks = self._chunker.chunk(document)
        self._retriever.index(chunks)
        self._corpus.add(document, chunks)
        return chunks

    def _require_llm(self) -> LLM:
        if self._llm is None:
            msg = (
                "No LLM is configured. Pass an object implementing the LLM "
                "protocol to RAG(llm=...) before calling ask() or stream()."
            )
            raise LLMNotConfiguredError(msg)
        return self._llm
