# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Document ingestion service for Open WebUI. A standalone FastAPI microservice that runs a Haystack v2 indexing pipeline (extract → chunk → embed → store) and writes Haystack-native documents (`content` + `meta`) to Qdrant. Sits next to `retrieval-agent/` in the parent repo and is called by Open WebUI's `EXTERNAL_INGESTION_ENGINE=external` patch via `PUT /api/v1/ingest`.

The pipeline is configurable end-to-end:

- **Extraction**: `EXTRACTION_ENGINE` selects `tika`, `pypdf`, `kreuzberg`, `docling`, `unstructured`, `vision-llm`, `hybrid-diagram`, or `auto`. Note the default is split-brained across files: `config.py` falls back to `tika`, `.env.example` ships `kreuzberg`, and `docker-compose.yml` falls back to `auto` (`${EXTRACTION_ENGINE:-auto}`) — so a `cp .env.example .env` Quick Start gives you `kreuzberg`, while an unset var under compose gives `auto`. `tika` and `kreuzberg` run as external HTTP sidecars (containers in the parent stack — `tika` at `TIKA_URL`, `goldziher/kreuzberg` at `KREUZBERG_URL`). `pypdf` is in-process. `tika`, `pypdf` and `kreuzberg` ship in the day-one image; `docling`/`unstructured` require optional deps. The factory raises a clear `ImportError` at startup if you select an engine whose dep is missing. The `kreuzberg` branch wraps the sidecar via a custom `KreuzbergRemoteConverter` Haystack component in `app/pipelines/kreuzberg_converter.py` — no third-party Haystack integration is used, so the dependency surface stays at `httpx` (already required). `vision-llm` renders document pages to images (office formats go through the bundled **Gotenberg** sidecar for office→PDF, PDF→PNG is local) and reconstructs layout-bound structure (flowcharts, diagrams, scans) into Markdown + a Mermaid graph via an OpenAI-compatible multimodal LLM (`VISION_LLM_*`). `hybrid-diagram` pairs native docx text (verbatim from the package XML) with a vision-inferred diagram, falling through to `vision-llm` for non-docx; it picks its vision profile per document via `docx_diagram_profile` — `diagram-topology` (Mermaid only) when the labels live in Word shapes/textboxes (a *vector* flowchart), or the `figure` profile when the diagram is a flattened *raster* PNG (labels are pixels, so the model reads them from the rendered image while the prose still comes from native text). `auto` is a routing *mode* (not a converter): docx with a vector flowchart (drawing/textbox text outweighs body text) **or** a large body raster image (a flattened diagram, zero textboxes) → `EXTRACTION_ROUTER_DIAGRAM_ENGINE` (default `hybrid-diagram`), everything else → `EXTRACTION_ROUTER_DEFAULT`; detection thresholds live in `EXTRACTION_ROUTER_*` (textbox: `_MIN_TEXTBOXES`/`_DRAWING_RATIO`; raster: `_MIN_BODY_IMAGES`/`_MIN_IMAGE_EMU` display-area floor + opt-in `_MIN_IMAGE_WORD_RATIO`) and the routing decision is logged when `DEBUG=true`. Header/footer logos never trigger the raster signal — detection reads only `word/document.xml`. The single source of truth for known engines is `KNOWN_EXTRACTION_ENGINES` in `app/config.py`.
- **Chunking**: factory in `app/pipelines/splitter.py` picks between Haystack's `DocumentSplitter` (`CHUNK_SPLIT_BY=word|sentence|passage`, counts in those units), the custom `HuggingFaceTokenizerSplitter` (`CHUNK_SPLIT_BY=token`, the default — wraps `langchain_text_splitters.RecursiveCharacterTextSplitter.from_huggingface_tokenizer` so chunk size is measured in the embedding model's actual tokens), and the structure-aware `MarkdownChunker` (`CHUNK_SPLIT_BY=markdown` — two-stage: split on `#`/`##`/`###` headings via `MarkdownHeaderTextSplitter`, then token-pack each section that exceeds `CHUNK_SIZE`. Preserves the heading hierarchy on each chunk as `meta.headers`). Token + markdown modes fall back to `EMBEDDING_MODEL` for the tokenizer unless `TOKENIZER_MODEL` is set; `transformers` + `langchain_text_splitters` are lazy-imported so word/sentence/passage users don't pay the cost. The tokenizer is `@lru_cache`d (first load ~1–3s, then free) and shared between the two HF-aware splitters.
- **Dense embedding**: `EMBEDDING_PROVIDER` picks `openai-compat` (the current `embed.itkdev.dk` path), `fastembed` (in-process), or `tei` (also OpenAI-compatible at the wire level). Required.
- **Sparse embedding**: `ENABLE_SPARSE_EMBEDDINGS=true` adds a `FastembedSparseDocumentEmbedder` stage so each Qdrant point holds both a dense and a sparse named vector. Optional; default off.

## Build & Run

The repo is **standalone** — its own `docker-compose.yml`, its own `.env`, run from the service root. The Dockerfile is multi-stage: `dev` target has test/lint tools (ruff, pytest), `prod` target is runtime-only. Compose defaults to `dev`. Python 3.12 in the container; `pyproject.toml` requires `>=3.11`.

The `frontend` Docker network is **external** — created by Traefik in the parent stack, or manually via `docker network create frontend` for standalone use. `task up` will refuse to start without it.

All `task` commands proxy through `docker compose exec ingestion` (`Taskfile.yml:11-12`). See `README.md` for the full task catalogue. Single test or test class:

```shell
docker compose exec ingestion pytest tests/test_ingest_endpoint.py::test_json_mode_happy_path -v
```

## Architecture

FastAPI app wired in `app/main.py` (lifespan, health probes, router include). Endpoints:

- **`PUT /api/v1/ingest`** — defined in `app/routes/ingest.py`. Bearer-token auth via `API_KEY` (`app/auth.py`). Single handler dispatches on `Content-Type`: `application/json` validates against `IngestRequestJSON` and fetches the file from S3, `multipart/form-data` streams the body to a tempfile. Both modes converge on `run_indexing_pipeline()`.
- **`GET /health`** — liveness probe (always 200 if the process is running).
- **`GET /health/ready`** — readiness probe. Returns 503 until `init_pipeline()` has finished (the sparse embedder downloads ~80 MB from HuggingFace on first boot) **and** Qdrant is reachable. The pipeline-warm gate is what keeps Docker / Kubernetes from routing traffic during cold start.

### Pipeline (`app/pipelines/indexing.py`)

Built once at lifespan startup, cached as module-level state — Haystack pipelines aren't cheap to construct. Three pieces of behaviour worth keeping here because they aren't obvious from any single file:

- **Idempotency via delete-then-write.** When `meta.overwrite=true` (default), `_delete_existing_by_file_id()` removes any existing Qdrant points where `meta.file_id == <file_id>` before the pipeline runs. Retries with the same `file_id` are safe; no duplicate vectors. Backfill via Open WebUI's reindex action depends on this contract.
- **All-or-nothing teardown.** On any pipeline exception, the same delete runs as cleanup. Partial writes don't reach Qdrant — points are either fully written (dense + sparse if enabled) or absent. The route layer (`app/routes/ingest.py`) catches the re-raised exception and maps it to the `IngestError.code` field.
- **`meta.collection_name` is the tenant key.** `QdrantDocumentStore` is configured with `hnsw_config={"m": 0, "payload_m": 16}` — multitenancy HNSW. Per-tenant subgraphs are keyed off the keyword payload index on `meta.collection_name` (created by `app/services/qdrant_setup.py` at startup). Mixing memories (short) and files (long) in one physical collection is fine because each `collection_name` gets its own subgraph.

### Vector Layout

With `ENABLE_SPARSE_EMBEDDINGS=true`, each Qdrant point carries two named vectors: a dense vector (used by the multitenancy HNSW) and a sparse vector (Qdrant's inverted index — no HNSW, no per-tenant subgraph; multitenancy is implicit because the same `meta.collection_name` filter applies at query time). Without sparse embeddings, only the dense named vector is written. The retrieval agent's hybrid-query path (Phase 3) detects which mode a collection is in at startup and chooses RRF fusion vs. dense-only + client-side BM25 accordingly.

### Schema

Each Qdrant point's payload carries:

```python
{
    "content": <chunk text>,
    "meta": {
        "file_id":         "<Open WebUI file UUID>",
        "collection_name": "<file-{uuid} | knowledge_id | user-memory-{uid} | ...>",
        "collection_type": "file | knowledge | memory | web-search | hash-based",
        "name":            "<filename>",
        "source":          "<filename>",
        "user_id":         "<user UUID>",
        "page":            <int, when the converter exposes it>,
        "headers":         <list[str], when CHUNK_SPLIT_BY=markdown — outermost-first breadcrumb of section headings, [] for chunks outside any heading>,
        "split_id":        <int, monotonic chunk index within the file>,
        # Optional document-level metadata (currently populated by the kreuzberg engine).
        # Request meta wins on collision; missing/empty values are omitted entirely.
        "title":           <str, document title from extractor>,
        "subject":         <str, PDF "subject" field>,
        "authors":         <list[str], from PDF author metadata>,
        "created_at":      <str, ISO 8601 timestamp from document properties>,
        "languages":       <list[str], ISO 639-1 codes from auto-detection; indexed in Qdrant as a KEYWORD payload index so future filter consumers can MatchAny>,
    }
}
```

`collection_name` preserves Open WebUI's existing naming so the UI / file model don't have to change. `collection_type` is a secondary indexed field for admin queries — not used for retrieval filtering.

### Testing

Tests use `pytest-asyncio` with `asyncio_mode = "auto"`. `tests/conftest.py` sets env vars **before any `app` imports** so tests don't accidentally hit real services. External dependencies (Qdrant, S3, the Haystack pipeline) are mocked via `patch()` against the route-layer symbols, not the underlying library functions — that keeps tests fast and avoids the fastembed model download path entirely.

## Configuration

All config via environment variables, loaded by pydantic-settings in `app/config.py`. See `.env.example` and `README.md` for the full list. The settings that matter beyond their docstrings — because they are **contracts with other systems**:

- `API_KEY` must equal Open WebUI's `EXTERNAL_INGESTION_API_KEY`. In the parent stack both are forked from a single deployer-facing `INGESTION_API_KEY` (see parent `docker-compose.yml` — the two consumers read `${INGESTION_API_KEY}` from the same source so they can never drift).
- `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `EMBEDDING_PREFIX_DOC` — must match what the retrieval agent uses at query time. Indexing-time prefix is `EMBEDDING_PREFIX_DOC` (e5 needs `passage: `, bge-m3 takes none, nomic uses `search_document: `). The retrieval agent applies `EMBEDDING_PREFIX_QUERY` on the query side; the two sides must use the same model + prefix or vector search returns garbage.
- `QDRANT_INDEX` is the physical Qdrant collection. Defaults to `ingestion_files` (distinct from Open WebUI's legacy multitenancy collections; the Phase 3 retrieval-agent rewrite will read from this collection exclusively).
- `S3_*` env vars match boto3 conventions. The service treats whatever endpoint it's pointed at (MinIO in dev, real AWS S3 in prod, etc.) as generic S3-compatible storage.

## Rules

- **Never read `.env` files.** They contain secrets (API keys, credentials). Use `.env.example` to understand available settings.
- **Always run Python commands inside the Docker container.** `pip install`, `pytest`, `ruff`, and any other project commands must be executed via `docker compose exec ingestion ...` from the repo root (or via the `task` wrapper). Never install or run Python tooling on the host.

## Failure-mode notes

- **Tika / Kreuzberg sidecar down** → `EXTRACTION_FAILED`. Open WebUI's file row goes to `failed`; the user sees the error and can retry once the sidecar is back.
- **Vision-LLM output exceeds `VISION_LLM_MAX_TOKENS`** → `EXTRACTION_FAILED`. The converter treats a `finish_reason=length` response as a hard failure (`vision_llm_converter.py`) rather than letting silently-truncated Markdown reach Qdrant. Remedy: raise `VISION_LLM_MAX_TOKENS` (staying under the served model's `--max-model-len` input headroom) or route the document to a text engine.
- **Embedding endpoint down** → `EMBEDDING_FAILED` (or `SPARSE_EMBEDDING_FAILED` for the sparse stage). The all-or-nothing teardown ensures no partial points reach Qdrant.
- **Qdrant write fails** → `QDRANT_WRITE_FAILED`. Tear-down still runs even if the original error came from a partial write — `_delete_existing_by_file_id()` swallows "collection not found" errors so the failure path is robust on cold starts.
- **S3 fetch 404 / auth** → `S3_FETCH_FAILED`. The route layer catches this before the pipeline runs.
- **Concurrent uploads from multiple users** are handled by FastAPI / uvicorn workers (configure via `WEB_CONCURRENCY`). Each request runs the pipeline in a per-request thread; the embedding endpoint is the bottleneck under load, not the pipeline itself. The autouse `reset_clients` test fixture catches state leakage between tests.
