import time

from fastapi import FastAPI


startup_import_start = time.perf_counter()

from src.api.routes import components, router


startup_import_time = time.perf_counter() - startup_import_start

print(
    f"STARTUP IMPORT TIME={startup_import_time:.2f}s",
    flush=True,
)


app = FastAPI(
    title="Enterprise Document Intelligence & RAG Assistant",
    version="1.0.0",
)


@app.on_event("startup")
def initialize_rag_components():
    startup_start = time.perf_counter()

    components._initialize()

    startup_time = time.perf_counter() - startup_start

    print(
        f"RAG STARTUP INITIALIZATION={startup_time:.2f}s",
        flush=True,
    )


app.include_router(router)