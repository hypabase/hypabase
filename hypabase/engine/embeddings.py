"""Embedding providers for Hypabase vector search.

Abstract base class and implementations for text embedding generation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Default implementation calls embed() in a loop.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        return [self.embed(t) for t in texts]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors."""


class SentenceTransformerProvider(EmbeddingProvider):
    """Embedding provider using sentence-transformers.

    Default model: all-MiniLM-L6-v2 (384 dimensions).

    Requires: ``pip install sentence-transformers``
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerProvider. "
                "Install it with: pip install sentence-transformers"
            ) from e
        self._model = SentenceTransformer(model_name)
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise ValueError(
                f"Model '{model_name}' did not report embedding dimension. "
                f"This model may not support sentence embeddings."
            )
        self._dimension = dim

    def embed(self, text: str) -> list[float]:
        result: list[float] = self._model.encode(text).tolist()
        return result

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = self._model.encode(texts).tolist()
        return result

    @property
    def dimension(self) -> int:
        return int(self._dimension)


class FastEmbedProvider(EmbeddingProvider):
    """Embedding provider using FastEmbed (ONNX-based, lightweight).

    Default model: BAAI/bge-small-en-v1.5 (384 dimensions).
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ImportError(
                "fastembed is required for FastEmbedProvider. Install it with: pip install fastembed"
            ) from e
        self._model = TextEmbedding(model_name=model_name)
        self._model_name = model_name
        # Resolve dimension from model metadata or probe
        try:
            desc = getattr(self._model, "model_description", None)
            if isinstance(desc, dict) and "dim" in desc:
                self._dimension = int(desc["dim"])
            else:
                raise AttributeError
        except (AttributeError, KeyError, TypeError):
            probe = list(self._model.embed(["hello"]))
            self._dimension = len(probe[0])

    def embed(self, text: str) -> list[float]:
        results = list(self._model.embed([text]))
        result: list[float] = results[0].tolist()
        return result

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results = list(self._model.embed(texts))
        return [r.tolist() for r in results]

    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIProvider(EmbeddingProvider):
    """Embedding provider using OpenAI's API.

    Default model: text-embedding-3-small (1536 dimensions).

    Requires: ``pip install openai`` and ``OPENAI_API_KEY`` environment variable.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int = 1536,
    ) -> None:
        try:
            import openai  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError("openai is required for OpenAIProvider. Install it with: pip install openai") from e
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._dimension = dimensions

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(input=[text], model=self._model)
        result: list[float] = resp.data[0].embedding
        return result

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(input=texts, model=self._model)
        return [d.embedding for d in resp.data]

    @property
    def dimension(self) -> int:
        return self._dimension
