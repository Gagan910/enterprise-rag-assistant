import time

from sentence_transformers import SentenceTransformer

from src.config.settings import settings


class TextEmbedder:
    """Generate vector embeddings for text."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model

        model_start = time.perf_counter()

        self.model = SentenceTransformer(self.model_name)

        model_time = time.perf_counter() - model_start

        print(
            f"EMBEDDER MODEL LOAD={model_time:.2f}s",
            flush=True,
        )

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text cannot be empty")

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        if any(not isinstance(text, str) for text in texts):
            raise TypeError("all texts must be strings")

        if any(not text.strip() for text in texts):
            raise ValueError("texts cannot contain empty strings")

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Return embedding vector dimension."""
        return self.model.get_sentence_embedding_dimension()