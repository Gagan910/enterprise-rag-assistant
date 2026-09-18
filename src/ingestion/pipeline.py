import hashlib
from pathlib import Path

from src.ingestion.chunker import chunk_text
from src.ingestion.cleaner import clean_text
from src.ingestion.parser import parse_document
from src.config.settings import settings


class IngestionPipeline:
    """Process documents and store their chunks in the vector database."""

    def __init__(
        self,
        embedder,
        vector_store,
    ):
        self.embedder = embedder
        self.vector_store = vector_store

    @staticmethod
    def _document_id(path: Path) -> str:
        """Generate a deterministic document ID from document content."""
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    def delete_document(self, file_path: str) -> None:
        """Delete all stored chunks belonging to a document."""
        path = Path(file_path)
        document_id = self._document_id(path)

        results = self.vector_store.collection.get(
            where={"document_id": document_id}
        )

        ids = results.get("ids", [])

        if ids:
            self.vector_store.delete(ids)

    def ingest(self, file_path: str) -> int:
        """Parse, clean, chunk, embed, and store a document."""

        path = Path(file_path)

        raw_text = parse_document(str(path))
        document_id = self._document_id(path)
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
                "document_id": document_id,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in chunks
        ]

        ids = [
            f"{document_id}-{chunk.chunk_id}"
            for chunk in chunks
        ]

        if all(self.vector_store.exists([chunk_id]) for chunk_id in ids):
            return 0

        self.vector_store.add_chunks(
            chunks=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        return len(chunks)
    