"""Local hashing embedding model."""

import re
from collections.abc import Sequence
from hashlib import blake2b
from math import sqrt

from pydantic import BaseModel, ConfigDict, Field

TOKEN_PATTERN = re.compile(r"[\w']+", flags=re.UNICODE)


class HashingEmbeddingModel(BaseModel):
    """A deterministic local embedding model based on feature hashing.

    It is intentionally simple and dependency-free. It gives the framework a
    useful default for local development and tests, while keeping the embedding
    model replaceable through the `EmbeddingModel` protocol.
    """

    model_config = ConfigDict(frozen=True)

    dimensions: int = Field(default=384, ge=32)
    lowercase: bool = True

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions

        for token in self._tokens(text):
            digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], byteorder="big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector

        return [value / norm for value in vector]

    def _tokens(self, text: str) -> list[str]:
        normalized = text.lower() if self.lowercase else text
        return TOKEN_PATTERN.findall(normalized)
