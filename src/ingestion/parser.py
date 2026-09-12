from pathlib import Path

from docx import Document
from pypdf import PdfReader


def parse_document(file_path: str) -> str:
    """
    Extract text from a PDF, DOCX, or TXT document.

    Args:
        file_path: Path to the document.

    Returns:
        Extracted text as a single string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is unsupported.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(path)

    if suffix == ".docx":
        return _parse_docx(path)

    if suffix == ".txt":
        return _parse_txt(path)

    raise ValueError(f"Unsupported document type: {suffix}")


def _parse_pdf(path: Path) -> str:
    reader = PdfReader(str(path))

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return "\n".join(pages).strip()


def _parse_docx(path: Path) -> str:
    document = Document(str(path))

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    return "\n".join(paragraphs).strip()


def _parse_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()