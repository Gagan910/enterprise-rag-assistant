from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    chunk_id: int


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[DocumentChunk]:
    """
    Split text into overlapping word-based chunks.

    Args:
        text: Clean document text.
        chunk_size: Maximum number of words per chunk.
        chunk_overlap: Number of words shared between consecutive chunks.

    Returns:
        List of DocumentChunk objects.
    """
    if not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []

    step = chunk_size - chunk_overlap

    for chunk_id, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start : start + chunk_size]

        if not chunk_words:
            break

        chunks.append(
            DocumentChunk(
                text=" ".join(chunk_words),
                chunk_id=chunk_id,
            )
        )

        if start + chunk_size >= len(words):
            break

    return chunks