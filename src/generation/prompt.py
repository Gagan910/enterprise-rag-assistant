def build_rag_prompt(
    question: str,
    contexts: list[dict],
) -> str:
    """Build a grounded prompt using retrieved document chunks."""

    if not isinstance(question, str):
        raise TypeError("question must be a string")

    if not question.strip():
        raise ValueError("question cannot be empty")

    if not contexts:
        raise ValueError("contexts cannot be empty")

    context_blocks = []

    for index, context in enumerate(contexts, start=1):
        source = context.get("metadata", {}).get(
            "source",
            "Unknown source",
        )

        context_blocks.append(
            f"[Source {index}: {source}]\n"
            f"{context['text']}"
        )

    context_text = "\n\n".join(context_blocks)

    return f"""You are an enterprise document assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not use outside knowledge.
- If the answer is not supported by the context, say:
  "I could not find this information in the provided documents."
- Do not invent facts.
- Cite the relevant source(s) using [Source N].

Context:
{context_text}

Question:
{question}

Answer:
"""