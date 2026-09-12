from pathlib import Path

from src.ingestion.chunker import chunk_text
from src.ingestion.cleaner import clean_text
from src.ingestion.parser import parse_document
from src.retrieval.embedder import TextEmbedder
from src.retrieval.vector_store import VectorStore
from src.config.settings import settings


class IngestionPipeline:
    """Process documents and store their chunks in the vector database."""

    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: VectorStore,
    ):
        self.embedder = embedder
        self.vector_store = vector_store

    def ingest(self, file_path: str) -> int:
        """Parse, clean, chunk, embed, and store a document."""

        path = Path(file_path)

        raw_text = parse_document(str(path))
        cleaned_text = clean_text(raw_text)

        chunks = chunk_text(
            cleaned_text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)

        metadatas = [
            {
                "source": path.name,
                "file_path": str(path),
                "chunk_id": chunk.chunk_id,
            }
            for chunk in chunks
        ]

        ids = [
            f"{path.stem}-{chunk.chunk_id}"
            for chunk in chunks
        ]

        self.vector_store.add_chunks(
            chunks=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        return len(chunks)