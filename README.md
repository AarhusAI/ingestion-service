# Ingestion Service

Document ingestion microservice for [Open WebUI](https://github.com/open-webui/open-webui).
A standalone FastAPI service that runs a [Haystack v2](https://haystack.deepset.ai/)
indexing pipeline (extract → chunk → embed → store) and writes Haystack-native documents
to Qdrant. The pluggable extraction engine, dense embedder, and optional sparse embedder
are all driven by configuration — the service isn't bound to any specific embedding model.

Called by Open WebUI when `EXTERNAL_INGESTION_ENGINE=external`. The Open WebUI patch
delegates to `PUT /api/v1/ingest` with either an S3 reference (preferred) or a multipart
file upload.

## Requirements

- Docker and Docker Compose
- [Task](https://taskfile.dev/) (Go Task runner)
- A Qdrant instance (shared with the retrieval agent)
- An OpenAI-compatible embedding endpoint (e.g. the `embed.itkdev.dk` proxy)
- A Tika server reachable on the same network (when `EXTRACTION_ENGINE=tika`)

## Quick Start

### Local Development

```shell
cp .env.example .env
# Edit .env — at minimum set API_KEY and EMBEDDING_API_KEY. The other defaults
# in .env.example (EMBEDDING_API_BASE_URL=https://embed.itkdev.dk/v1, MinIO,
# Qdrant, Tika URLs) are the working Aarhus dev values; only override when
# pointing at something else.

# Generate a secure API key:
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Where the same secret has to land:
#   - standalone:    set as API_KEY in this service's .env.
#   - parent stack:  set as INGESTION_API_KEY in the parent .env. The parent
#                    docker-compose forks it into two places — this service's
#                    API_KEY and the openwebui container's
#                    EXTERNAL_INGESTION_API_KEY. You do not set the latter two
#                    by hand.

task setup          # starts container + installs dev deps (requires Traefik 'frontend' network)
task logs           # tail ingestion container logs
```

Common task commands:

```shell
task up             # start containers
task down           # stop containers
task shell          # open bash shell in the ingestion container
task install        # reinstall deps (pip install '.[dev]')
task lint           # run all linters (ruff check + format --check)
task lint:fix       # auto-fix lint issues
task test           # run all tests (pytest -v)
task test:coverage  # run tests with coverage report
task ci             # lint + test
```

Run a single test (or one test in a file):

```shell
docker compose exec ingestion pytest tests/test_ingest_endpoint.py -v
docker compose exec ingestion pytest tests/test_ingest_endpoint.py::test_json_mode_happy_path -v
```

### Production Image

```shell
task build:image              # build + push to ghcr.io/aarhusai/ingestion-service:latest
task build:image TAG=v1.0.0   # with specific tag
```

## Health Endpoints

- `GET /health` — liveness probe (always 200 if the process is running)
- `GET /health/ready` — readiness probe (verifies Qdrant connectivity, returns 503 if unreachable)

## API

### `PUT /api/v1/ingest`

Two transport modes share one endpoint and dispatch on `Content-Type`.

#### S3-reference mode (preferred)

```http
PUT /api/v1/ingest HTTP/1.1
Authorization: Bearer <API_KEY>
Content-Type: application/json

{
  "s3_bucket": "openwebui",
  "s3_key": "files/abc/report.pdf",
  "file_id": "abc",
  "filename": "report.pdf",
  "collection_name": "file-abc",
  "collection_type": "file",
  "user_id": "u-1",
  "overwrite": true
}
```

#### Multipart fallback

```http
PUT /api/v1/ingest HTTP/1.1
Authorization: Bearer <API_KEY>
Content-Type: multipart/form-data; boundary=...

# fields:
file=<binary>
file_id=abc
filename=report.pdf
collection_name=file-abc
collection_type=file
user_id=u-1
overwrite=true
```

#### Response

```json
{
  "status": true,
  "collection_name": "file-abc",
  "chunks_count": 42
}
```

#### Error

```json
{
  "status": false,
  "error": "human-readable message",
  "code": "EXTRACTION_FAILED"
}
```

Codes: `EXTRACTION_FAILED`, `EMBEDDING_FAILED`, `SPARSE_EMBEDDING_FAILED`,
`QDRANT_WRITE_FAILED`, `S3_FETCH_FAILED`, `INVALID_REQUEST`, `PIPELINE_FAILED`.

#### Idempotency

When `overwrite=true` (default), all existing Qdrant points with matching
`meta.file_id` are deleted before writing new chunks. Retries are safe — they
delete-and-rewrite, no duplicates. On any pipeline failure the same delete runs
as teardown, so partial writes never leak into Qdrant.

### `POST /api/v1/extract`

Developer-facing extraction probe: runs the configured Haystack converter
against an uploaded file and returns the raw extracted documents. No chunking,
no embedding, no Qdrant writes. Useful for sanity-checking how a given engine
sees a document before committing to a full ingest, and for comparing engines
side-by-side without restarting the container.

Multipart-only. Same Bearer-token auth as `/api/v1/ingest`.

Fields:

- `file` (required) — the document to extract.
- `engine` (optional) — one of `tika | pypdf | docling | unstructured`.
  Overrides `EXTRACTION_ENGINE` for this single request. When omitted, the
  configured default is used.

#### Example

```shell
curl -X POST \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@sample.pdf \
  -F engine=pypdf \
  http://localhost:8000/api/v1/extract
```

#### Response

```json
{
  "status": true,
  "engine": "pypdf",
  "documents": [
    {"content": "# Heading\n...", "meta": {"page": 1}},
    {"content": "...",            "meta": {"page": 2}}
  ]
}
```

Errors use the same `IngestError` shape as `/api/v1/ingest`. Codes returned:
`INVALID_REQUEST` (unknown engine, missing file, missing optional dependency
for `docling`/`unstructured`) and `EXTRACTION_FAILED` (converter raised at
runtime). Note that `engine=unstructured` is wired in but its converter uses
a `paths=` input socket Haystack's pipeline doesn't currently route to — the
ingest pipeline has the same limitation; this endpoint will surface it as
`EXTRACTION_FAILED`.

## Configuration

All config is via environment variables loaded by pydantic-settings. See
`.env.example` for the full list. The settings that matter beyond their docstrings
because they are **contracts with other services**:

- `API_KEY` must equal Open WebUI's `EXTERNAL_INGESTION_API_KEY` (and the
  retrieval agent's parallel value when querying the same data).
- `EMBEDDING_MODEL`, `EMBEDDING_DIM`, and `EMBEDDING_PREFIX_DOC` must match
  whatever the retrieval agent uses at query time. e5 needs `passage: ` on
  documents and `query: ` on queries; bge-m3 takes no prefix; nomic uses
  `search_document: ` / `search_query: `.
- `QDRANT_INDEX` is the physical Qdrant collection. Defaults to
  `ingestion_files` — distinct from Open WebUI's legacy multitenancy collections.
- `ENABLE_SPARSE_EMBEDDINGS=true` adds a sparse vector to each Qdrant point so
  the retrieval agent can use Qdrant's native hybrid query (RRF fusion) instead
  of the legacy client-side BM25.
- `CHUNK_SPLIT_BY` selects the chunking strategy. The default `token` mode
  measures `CHUNK_SIZE` / `CHUNK_OVERLAP` in the embedding model's actual
  HuggingFace tokens (via `RecursiveCharacterTextSplitter.from_huggingface_tokenizer`)
  so chunks respect the model's context window — important for e5-large's
  512-token cap once the `passage: ` prefix is prepended. `word`, `sentence`,
  and `passage` delegate to Haystack's built-in `DocumentSplitter` and count
  in those units instead. Token mode uses `TOKENIZER_MODEL` if set, otherwise
  falls back to `EMBEDDING_MODEL`.

## Supported Embedding Models

| Model | Dim | Doc / query prefix | Native sparse |
|---|---|---|---|
| `intfloat/multilingual-e5-large` | 1024 | `passage: ` / `query: ` | — |
| `BAAI/bge-m3` | 1024 | none / none | yes |
| `jinaai/jina-embeddings-v3` | 1024 | task-specific | — |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | `search_document: ` / `search_query: ` | — |

`EMBEDDING_PROVIDER`:
- `openai-compat` → `OpenAIDocumentEmbedder` (the current `embed.itkdev.dk` path)
- `fastembed` → `FastembedDocumentEmbedder` (in-process inference; supports BGE-M3 dense)
- `tei` → routes through `OpenAIDocumentEmbedder` (TEI exposes an OpenAI-compatible endpoint)

`SPARSE_EMBEDDING_PROVIDER`:
- `fastembed` → `FastembedSparseDocumentEmbedder` (BGE-M3 sparse, BM42, SPLADE family)
- `none` → no sparse stage, dense-only pipeline

## Extraction Engines

| `EXTRACTION_ENGINE` | Status | Notes |
|---|---|---|
| `tika` | day-one | Reuses the existing `tika` container in the parent stack |
| `pypdf` | day-one | Lightweight, PDF-only |
| `docling` | optional dep | Add `docling-haystack` to `pyproject.toml` and rebuild |
| `unstructured` | optional dep | Add `unstructured-fileconverter-haystack` to `pyproject.toml` and rebuild |
