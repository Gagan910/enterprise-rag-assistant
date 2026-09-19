from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import components, router


@asynccontextmanager
async def lifespan(app: FastAPI):
    components._initialize()
    yield


app = FastAPI(
    title="Enterprise Document Intelligence & RAG Assistant",
    version="1.0.0",
    lifespan=lifespan,
)


app.include_router(router)