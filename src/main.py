import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import components, router

from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(components._initialize)
    yield


app = FastAPI(
    title="Enterprise Document Intelligence & RAG Assistant",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://enterprise-rag-frontend-g1bm.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)


app.include_router(router)