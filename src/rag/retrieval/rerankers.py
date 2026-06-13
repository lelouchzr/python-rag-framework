"""Lightweight reranking utilities."""

import re
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from rag.retrieval.models import SearchResult

TOKEN_PATTERN = re.compile(r"[\w']+", flags=re.UNICODE)


class Reranker(Protocol):
    """Reorders retrieved results for a query."""

    def rerank(
        self,
        query: str,
        results: Sequence[SearchResult],
        *,
        limit: int,
    ) -> list[SearchResult]:
        """Return reranked results."""


class LexicalReranker(BaseModel):
    """Rerank results using token overlap blended with vector similarity."""

    model_config = ConfigDict(frozen=True)

    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)

    def rerank(
        self,
        query: str,
        results: Sequence[SearchResult],
        *,
        limit: int,
    ) -> list[SearchResult]:
        query_tokens = _tokens(query)
        lexical_weight = 1.0 - self.vector_weight

        reranked = [
            SearchResult(
                chunk=result.chunk,
                score=min(
                    1.0,
                    (result.score * self.vector_weight)
                    + (_overlap(query_tokens, result.chunk.text) * lexical_weight),
                ),
            )
            for result in results
        ]
        return sorted(reranked, key=lambda result: result.score, reverse=True)[:limit]


def _tokens(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(text.lower()))


def _overlap(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    text_tokens = _tokens(text)
    if not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)
