import logging
import time

import httpx
from google import genai
from google.genai.errors import ClientError, ServerError
from groq import Groq
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.config.settings import settings


logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Raised when the configured LLM providers are unavailable."""


def _log_retry(retry_state):
    logger.warning(
        "LLM RETRY attempt=%s/4 exception=%s",
        retry_state.attempt_number,
        type(retry_state.outcome.exception()).__name__,
    )


class LLMClient:
    """LLM client with Gemini primary and Groq fallback."""

    def __init__(self):
        self.primary_provider = settings.llm_provider
        self.fallback_provider = settings.llm_fallback_provider
        self.model_name = settings.llm_model

        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options={"timeout": 120000},
        )

        self.groq_client = None

        if settings.groq_api_key:
            self.groq_client = Groq(
                api_key=settings.groq_api_key,
            )

        self.provider_used = None
        self.provider_attempts = []

        self._gemini_cooldown_until = 0.0
        self._gemini_cooldown_seconds = 300

    @retry(
        retry=retry_if_exception(
            lambda exc: (
                isinstance(
                    exc,
                    (
                        httpx.ReadTimeout,
                        ConnectionError,
                        ServerError,
                    ),
                )
                or (
                    isinstance(exc, ClientError)
                    and getattr(exc, "code", None) != 429
                )
            )
        ),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(4),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _generate_with_gemini(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        return response.text or ""

    def _generate_with_groq(self, prompt: str) -> str:
        if self.groq_client is None:
            raise RuntimeError(
                "Groq fallback is configured but GROQ_API_KEY is not available."
            )

        response = self.groq_client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.choices[0].message.content or ""

    def generate(self, prompt: str) -> str:
        """Generate a response using the primary provider with fallback."""

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        self.provider_used = None
        self.provider_attempts = []

        gemini_in_cooldown = (
            time.monotonic() < self._gemini_cooldown_until
        )

        if (
            self.primary_provider == "gemini"
            and gemini_in_cooldown
            and self.fallback_provider == "groq"
        ):
            logger.info(
                "Gemini cooldown active. Skipping Gemini and using Groq."
            )

            self.provider_attempts.append("groq")

            try:
                answer = self._generate_with_groq(prompt)
                self.provider_used = "groq"

                return answer

            except Exception as exc:
                logger.error(
                    "Fallback LLM unavailable provider=groq error=%s",
                    type(exc).__name__,
                )

                raise LLMUnavailableError(
                    "All configured LLM providers are unavailable."
                ) from exc

        if self.primary_provider == "gemini":
            self.provider_attempts.append("gemini")

            try:
                logger.info(
                    "Generating response with Gemini model=%s",
                    self.model_name,
                )

                answer = self._generate_with_gemini(prompt)
                self.provider_used = "gemini"

                return answer

            except (
                httpx.ReadTimeout,
                ConnectionError,
                ServerError,
            ) as exc:
                logger.warning(
                    "Primary LLM unavailable provider=gemini error=%s",
                    type(exc).__name__,
                )

                if self.fallback_provider != "groq":
                    raise LLMUnavailableError(
                        "Primary LLM provider is unavailable."
                    ) from exc

            except ClientError as exc:
                if getattr(exc, "code", None) != 429:
                    raise

                self._gemini_cooldown_until = (
                    time.monotonic() + self._gemini_cooldown_seconds
                )

                logger.warning(
                    "Primary LLM quota exhausted provider=gemini error=%s",
                    type(exc).__name__,
                )

                if self.fallback_provider != "groq":
                    raise LLMUnavailableError(
                        "Primary LLM provider quota is exhausted."
                    ) from exc

        if self.fallback_provider == "groq":
            self.provider_attempts.append("groq")

            logger.info(
                "Falling back to Groq model=%s",
                settings.groq_model,
            )

            try:
                answer = self._generate_with_groq(prompt)
                self.provider_used = "groq"

                return answer

            except Exception as exc:
                logger.error(
                    "Fallback LLM unavailable provider=groq error=%s",
                    type(exc).__name__,
                )

                raise LLMUnavailableError(
                    "All configured LLM providers are unavailable."
                ) from exc

        raise LLMUnavailableError(
            f"No supported LLM provider configured: "
            f"primary={self.primary_provider}, "
            f"fallback={self.fallback_provider}"
        )