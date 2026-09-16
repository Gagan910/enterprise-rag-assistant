from src.generation.llm import LLMClient
from src.generation.prompt import build_rag_prompt


class RAGGenerator:
    """Generate grounded answers from retrieved document context."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.provider_used = None

    def generate(
        self,
        question: str,
        contexts: list[dict],
    ) -> str:
        """Build a RAG prompt and generate an answer."""

        prompt = build_rag_prompt(
            question=question,
            contexts=contexts,
        )

        answer = self.llm_client.generate(prompt)

        self.provider_used = self.llm_client.provider_used

        return answer