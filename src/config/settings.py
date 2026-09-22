from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str
    api_key: str | None = None
    api_keys: str | None = None
    groq_api_key: str | None = None

    llm_provider: str = "gemini"
    llm_fallback_provider: str = "groq"

    llm_model: str = "gemini-3.7-flash"
    groq_model: str = "llama-3.3-70b-versatile"

    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)

    retrieval_top_k: int = Field(default=5, gt=0)
    rerank_top_k: int = Field(default=3, gt=0)

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        supported_providers = {"gemini", "groq"}

        if self.llm_provider not in supported_providers:
            raise ValueError(
                f"Unsupported llm_provider: {self.llm_provider}"
            )

        if self.llm_fallback_provider not in supported_providers:
            raise ValueError(
                f"Unsupported llm_fallback_provider: "
                f"{self.llm_fallback_provider}"
            )

        if self.llm_provider == self.llm_fallback_provider:
            raise ValueError(
                "llm_provider and llm_fallback_provider must be different"
            )

        return self

    vector_store_path: str = "data/chroma"
    vector_collection_name: str = "documents"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
