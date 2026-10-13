import logging
import time

import httpx
from google import genai
from google.genai.errors import ClientError, ServerError
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    Groq,
    InternalServerError,
    RateLimitError,
)
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
    exc = retry_state.outcome.exception()

    logger.warning(
        "LLM RETRY attempt=%s/2 exception=%s details=%s",
        retry_state.attempt_number,
        type(exc).__name__,
        str(exc)[:200],
    )


def _is_retryable_gemini_error(exc: BaseException) -> bool:
    """Return True only for transient Gemini failures."""

    if isinstance(exc, (httpx.ReadTimeout, ConnectionError)):
        return True

    if isinstance(exc, ServerError):
        return True

    if isinstance(exc, ClientError):
        return getattr(exc, "code", None) not in {400, 401, 403, 404, 429}

    return False


def _is_retryable_groq_error(exc: BaseException) -> bool:
    """Return True only for transient Groq API failures."""

    if isinstance(
        exc,
        (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            httpx.TimeoutException,
            httpx.TransportError,
        ),
    ):
        return True

    if isinstance(exc, APIStatusError):
        status_code = getattr(exc, "status_code", None)
        return status_code == 429 or (
            status_code is not None and status_code >= 500
        )

    return False


class LLMClient:
    """LLM client with configurable primary and fallback providers."""

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
        retry=retry_if_exception(_is_retryable_gemini_error),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        stop=stop_after_attempt(2),
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
        """Generate using the primary provider, falling back on transient failures."""

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        self.provider_used = None
        self.provider_attempts = []

        providers = [self.primary_provider, self.fallback_provider]
        last_error = None

        for provider in providers:
            if (
                provider == "gemini"
                and time.monotonic() < self._gemini_cooldown_until
            ):
                logger.info("Gemini cooldown active; skipping Gemini.")
                continue

            if provider not in {"gemini", "groq"}:
                raise LLMUnavailableError(
                    f"Unsupported provider: {provider}"
                )

            self.provider_attempts.append(provider)

            try:
                if provider == "groq":
                    logger.info(
                        "Generating response with Groq model=%s",
                        settings.groq_model,
                    )
                    answer = self._generate_with_groq(prompt)
                else:
                    logger.info(
                        "Generating response with Gemini model=%s",
                        self.model_name,
                    )
                    answer = self._generate_with_gemini(prompt)

                self.provider_used = provider
                return answer

            except Exception as exc:
                # Determine whether this is a recognized availability failure.
                if provider == "groq":
                    is_transient = _is_retryable_groq_error(exc)
                else:
                    is_transient = _is_retryable_gemini_error(exc)

                    # Gemini quota errors should trigger fallback.
                    is_quota_error = (
                        isinstance(exc, ClientError)
                        and getattr(exc, "code", None) == 429
                    )
                    is_transient = is_transient or is_quota_error

                    if is_transient:
                        self._gemini_cooldown_until = (
                            time.monotonic()
                            + self._gemini_cooldown_seconds
                        )

                if not is_transient:
                    # Do not hide programming errors or permanent client errors.
                    raise

                last_error = exc
                logger.warning(
                    "Transient LLM provider failure provider=%s error=%s",
                    provider,
                    type(exc).__name__,
                )

        raise LLMUnavailableError(
            "All configured LLM providers are unavailable."
        ) from last_error