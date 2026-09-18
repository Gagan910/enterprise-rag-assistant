from fastapi import FastAPI

from src.api.routes import components, router


app = FastAPI(
    title="Enterprise Document Intelligence & RAG Assistant",
    version="1.0.0",
)


@app.on_event("startup")
def initialize_rag_components():
    components._initialize()


app.include_router(router)