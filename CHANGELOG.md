# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog], and this project adheres to [Semantic Versioning].

## [Unreleased]

- Added actions and linted code base.

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
[unreleased]: https://github.com/AarhusAI/ingestion-service/compare/v0.0.5...HEAD
[0.0.8]: https://github.com/AarhusAI/ingestion-service/releases/tag/0.0.8
[0.0.7]: https://github.com/AarhusAI/ingestion-service/releases/tag/0.0.7
[0.0.5]: https://github.com/AarhusAI/ingestion-service/releases/tag/0.0.5
