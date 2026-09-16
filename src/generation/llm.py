import logging

import httpx
from google import genai
from google.genai.errors import ServerError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.settings import settings


logger = logging.getLogger(__name__)


def _log_retry(retry_state):
    logger.warning(
        "LLM RETRY attempt=%s/4 exception=%s",
        retry_state.attempt_number,
        type(retry_state.outcome.exception()).__name__,
    )


class LLMClient:
    """Client for generating answers with Gemini."""

    def __init__(self):
        self.model_name = settings.llm_model
        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options={"timeout": 120000},
        )

    @retry(
        retry=retry_if_exception_type(
            (
                httpx.ReadTimeout,
                ConnectionError,
                ServerError,
            )
        ),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(4),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _generate_with_retry(self, prompt: str):
        """Generate content with retry handling for transient failures."""
        return self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

    def generate(self, prompt: str) -> str:
        """Generate a response from Gemini."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        logger.info(
            "Generating response with Gemini model=%s",
            self.model_name,
        )

        response = self._generate_with_retry(prompt)

        return response.text or ""