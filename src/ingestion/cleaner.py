import re


def clean_text(text: str) -> str:
    """
    Clean extracted document text while preserving meaningful content.

    Args:
        text: Raw text extracted from a document.

    Returns:
        Normalized and cleaned text.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove trailing whitespace from each line.
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Collapse runs of spaces/tabs into a single space.
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()