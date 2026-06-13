"""Local n-gram hashing embedding model."""

import re
from collections.abc import Sequence
from hashlib import blake2b
from math import sqrt

from pydantic import BaseModel, ConfigDict, Field

TOKEN_PATTERN = re.compile(r"[\w']+", flags=re.UNICODE)


class NgramHashingEmbeddingModel(BaseModel):
    """A stronger local embedding model using word and character n-grams."""

    model_config = ConfigDict(frozen=True)

    dimensions: int = Field(default=768, ge=64)
    min_char_ngram: int = Field(default=3, ge=2)
    max_char_ngram: int = Field(default=5, ge=2)
    lowercase: bool = True

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions

        for feature in self._features(text):
            digest = blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], byteorder="big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector

        return [value / norm for value in vector]

    def _features(self, text: str) -> list[str]:
        normalized = text.lower() if self.lowercase else text
        tokens = TOKEN_PATTERN.findall(normalized)
        features = [f"word:{token}" for token in tokens]

        for token in tokens:
            max_size = min(self.max_char_ngram, len(token))
            for size in range(self.min_char_ngram, max_size + 1):
                for start in range(0, len(token) - size + 1):
                    features.append(f"char:{token[start : start + size]}")

        return features
