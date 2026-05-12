"""FastAPI app for the ingestion service.

Lifespan: builds the Haystack pipeline once, ensures the Qdrant payload
indexes exist, then serves requests until shutdown.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import settings
from app.pipelines.indexing import init_pipeline
from app.routes.extract import router as extract_router
from app.routes.ingest import router as ingest_router
from app.services import qdrant_setup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
if settings.debug:
    logging.getLogger("app").setLevel(logging.DEBUG)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting ingestion service")
    log.info("Qdrant URI: %s (index=%s)", settings.qdrant_uri, settings.qdrant_index)
    log.info(
        "Extraction engine: %s (tika_url=%s)",
        settings.extraction_engine,
        settings.tika_url,
    )
    log.info(
        "Dense embedder: provider=%s model=%s dim=%s",
        settings.embedding_provider,
        settings.embedding_model,
        settings.embedding_dim,
    )
    log.info(
        "Sparse embeddings: %s (provider=%s model=%s)",
        settings.enable_sparse_embeddings,
        settings.sparse_embedding_provider,
        settings.sparse_embedding_model,
    )
    log.info(
        "Chunking: split_by=%s size=%d overlap=%d",
        settings.chunk_split_by,
        settings.chunk_size,
        settings.chunk_overlap,
    )

    init_pipeline()
    qdrant_setup.ensure_payload_indexes()

    yield

    log.info("Ingestion service shut down")


app = FastAPI(
    title="Ingestion Service",
    description="Document ingestion service for Open WebUI (Haystack v2)",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(ingest_router)
app.include_router(extract_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/ready")
async def health_ready():
    """Readiness probe — verifies Qdrant connectivity."""
    if qdrant_setup.health_check():
        return {"status": "ok"}
    return JSONResponse(
        status_code=503,
        content={"status": "error", "detail": "qdrant unreachable"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
