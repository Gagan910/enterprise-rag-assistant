from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str

    llm_model: str = "gemini-3.7-flash"

    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_size: int = 500
    chunk_overlap: int = 100

    retrieval_top_k: int = 5
    rerank_top_k: int = 3

    vector_store_path: str = "data/chroma"
    vector_collection_name: str = "documents"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
settings = Settings()