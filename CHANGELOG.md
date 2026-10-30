# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog], and this project adheres to [Semantic Versioning].

## [Unreleased]

- Sparse vectors now use Qdrant's IDF modifier (required for BM42 scoring). New collections get it at creation;
  existing ones are migrated in place at startup (no reindex).
- The collection is created at startup, so payload indexes exist from first boot and `DELETE` before the first ingest
  returns 200 instead of 500.
- docx zip members over 50 MB (declared size) are refused with `EXTRACTION_FAILED` (zip-bomb guard); PDF page renders
  are capped at 16 MP.
- New `INGEST_MAX_CONCURRENCY` (default `4`, `0` = unlimited): pipeline runs in flight per worker.
- Auth: missing header → 401 with `WWW-Authenticate: Bearer`; non-ASCII tokens → 401 (was 500).
- Ingest: collection-binding and form-field checks run before the S3 fetch / temp-file spool (no orphaned temp files
  on 400/403); multipart bodies with a `Content-Length` over the cap get 413 before parsing.
- Kreuzberg non-JSON responses → `EXTRACTION_FAILED` (was `PIPELINE_FAILED`).

- `docker-compose.yml` fallbacks now match `app/config.py`: `EXTRACTION_ROUTER_DIAGRAM_ENGINE=hybrid-diagram` (was
  `vision-llm`), `CHUNK_SIZE=400` / `CHUNK_OVERLAP=80` (was `500` / `100`). Only affects a `.env` that doesn't set
  them. Compose now also passes through `KREUZBERG_MIN_TABLE_COLUMNS`, `EXTRACTION_ROUTER_MIN_BODY_IMAGES`,
  `EXTRACTION_ROUTER_MIN_IMAGE_EMU`, `EXTRACTION_ROUTER_MIN_IMAGE_WORD_RATIO`, `VISION_LLM_MAX_TOKENS` and
  `CHUNK_MIN_SIZE`.
- README: full configuration reference table.
- Removed the unused `HOST` / `PORT` settings; uvicorn's `0.0.0.0:8000` bind is set in the Dockerfile `CMD`.

- `ENABLE_SPARSE_EMBEDDINGS` now defaults to `true` (was `false`), matching the parent stack. **Breaking for existing
  standalone collections created dense-only:** startup fails with `QdrantStoreError` — set
  `ENABLE_SPARSE_EMBEDDINGS=false` to keep using them, or use a new `QDRANT_INDEX` and reindex. First boot downloads the
  ~80 MB BM42 model.

- `EXTRACTION_ENGINE` now defaults to `auto` (was `kreuzberg`) in `config.py` and `.env.example`, matching
  `docker-compose.yml` and the parent stack. Ordinary documents still go to `kreuzberg`; diagram-heavy docx route to
  `hybrid-diagram` and need `VISION_LLM_*` — without it they fail with `EXTRACTION_FAILED`. Pin
  `EXTRACTION_ENGINE=kreuzberg` to keep the old behaviour.

- Removed the `SPARSE_EMBEDDING_PROVIDER` setting; `ENABLE_SPARSE_EMBEDDINGS` is now the single sparse switch
  (fastembed is the only provider). `ENABLE_SPARSE_EMBEDDINGS=true` with `SPARSE_EMBEDDING_PROVIDER=none` now enables
  sparse embeddings — set `ENABLE_SPARSE_EMBEDDINGS=false` instead. A leftover `SPARSE_EMBEDDING_PROVIDER` is ignored.
- Removed the chunk-inspection endpoint (`GET /api/v1/documents/{file_id}/chunks`) and its `ENABLE_INSPECTION_API`
  setting; inspect stored chunks (and their `extraction_engine`/`extraction_route` meta) in the Qdrant UI instead.
- `CHUNK_SPLIT_BY` now defaults to `markdown` (was `token`), matching the parent stack. Changes chunk boundaries for
  deployments that relied on the old default — set `CHUNK_SPLIT_BY=token` to keep it, or reindex.
- Fixed stale `METRICS_ENABLED` comment: `/metrics` requires the `API_KEY` bearer token.

## [0.0.9] - 2026-10-06

- Added actions and linted code base.
- Removed duplicated helpers, unreachable fallbacks and unused error classes/settings from `app/` (~400 lines).
- Kreuzberg responses with an unrecognized shape now fail with `EXTRACTION_FAILED` instead of silently indexing zero
  chunks.
- Docker: prod venv and code are root-owned (read-only to `appuser`); image build no longer runs a recursive `chown`.
- New in-process `csv` extraction engine: each `.csv`/`.tsv` row becomes `column: value` lines, so every chunk keeps
  its column names. `auto` routes `.csv`/`.tsv` to it. Token and markdown splitters cut CSV documents only between
  rows, with no overlap. Handles Excel `sep=` lines, UTF-16 exports, and quoted delimiters in headers; cells past
  the header are kept as `column N`. The delimiter is picked by column-count consistency over the first 20 rows,
  preferring `;` over tab and `,` on ties, so Danish Excel exports with decimal commas parse correctly.

## [0.0.8] - 2026-09-24

### Fixed

- Ensure that files attached and knowledge bases do not override each other. Scoping by collection keeps a file's copies
in other collections (`file-<id>` plus each KB it belongs to) alive.

## [0.0.7] - 2026-07-30

### Added

- Added timeouts to embedding service calls.

## [0.0.6] - 2026-07-29

### Added

- Hard cap on embedding input tokens via new `EMBEDDING_MAX_TOKENS` setting (default 512). Chunks shrink so content, doc
  prefix, heading breadcrumb and special tokens all fit, preventing one oversized chunk from making the endpoint reject
  a whole batch with HTTP 400.
- New `CHUNK_MIN_SIZE` setting (default 100). In markdown mode, tiny adjacent sections are merged, never past
  `CHUNK_SIZE`. Set to 0 to disable.
- New `EMBED_HEADERS_BREADCRUMB` setting (default `true`). Embedders now see the full heading path (e.g. "Setup >
  Docker > Networking"), not just the leaf heading. Stored chunk content is unchanged.
- Every ingest run stamps a fresh `meta.ingest_version` on its chunks. New chunks are written alongside the old ones,
  and the old version is deleted only after the new one ingests successfully.
- Data flow documentation (`docs/data-flow.md`) with diagrams.

### Changed

- Docker Compose settings updated to match production.
- README, `CLAUDE.md`, Taskfile and `.env.example` updated for the new settings and tasks.
- Code reformatted with ruff.

### Fixed

- Raw-Qdrant helpers now use the pipeline's own settings rather than `global_settings`.
- Pipeline errors are unwrapped from Haystack's `PipelineRuntimeError`, so correct error codes are returned.
- Audit check fixed.
- Data flow diagram rendering fixed.

### Upgrade notes

- `CHUNK_MIN_SIZE`, `EMBED_HEADERS_BREADCRUMB` and `EMBEDDING_MAX_TOKENS` change chunk boundaries or vectors. Reindex
  existing collections for consistency.

## [0.0.5] - 2026-07-02

### Added

- New route to delete ingested data (`app/routes/delete.py`), with request/response models and tests.
- Logging on delete requests.

### Changed

- Reduced CPU usage during embedding.

### Removed

- Legacy Tika support.

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
[unreleased]: https://github.com/AarhusAI/ingestion-service/compare/v0.0.9...HEAD
[0.0.9]: https://github.com/AarhusAI/ingestion-service/releases/tag/0.0.9
[0.0.8]: https://github.com/AarhusAI/ingestion-service/releases/tag/0.0.8
[0.0.7]: https://github.com/AarhusAI/ingestion-service/releases/tag/0.0.7
[0.0.5]: https://github.com/AarhusAI/ingestion-service/releases/tag/0.0.5
