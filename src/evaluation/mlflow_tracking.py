import os
import json

from dotenv import load_dotenv

load_dotenv()

import mlflow

from src.config.settings import settings


EXPERIMENT_NAME = "enterprise-rag-retrieval-evaluation"


def log_retrieval_evaluation(
    metrics: dict[str, float],
    top_k: int,
) -> None:
    """Log retrieval evaluation metrics and configuration to MLflow."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not configured.")

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{EXPERIMENT_NAME}' does not exist."
        )

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_metrics(
            {
                "hit_rate": metrics["hit_rate"],
                "mrr": metrics["mrr"],
            }
        )

        mlflow.log_params(
            {
                "top_k": top_k,
                "embedding_model": settings.embedding_model,
                "reranker_model": settings.reranker_model,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "retrieval_top_k": settings.retrieval_top_k,
                "rerank_top_k": settings.rerank_top_k,
            }
            
        )
        
        mlflow.log_params(
            {
                "top_k": top_k,
                "embedding_model": settings.embedding_model,
                "reranker_model": settings.reranker_model,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "retrieval_top_k": settings.retrieval_top_k,
                "rerank_top_k": settings.rerank_top_k,
            }
        )

        with open("retrieval_evaluation.json", "w", encoding="utf-8") as file:
            json.dump(metrics["results"], file, indent=2)

        mlflow.log_artifact("retrieval_evaluation.json")