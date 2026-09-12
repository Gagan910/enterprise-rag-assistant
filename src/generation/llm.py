from google import genai

from src.config.settings import settings


class LLMClient:
    """Client for generating answers with Gemini."""

    def __init__(self):
        self.model_name = settings.llm_model
        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options={"timeout": 30000},
        )

    def generate(self, prompt: str) -> str:
        """Generate a response from Gemini."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        return response.text or ""