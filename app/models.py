from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Error codes returned in IngestError.code
ErrorCode = Literal[
    "EXTRACTION_FAILED",
    "EMBEDDING_FAILED",
    "SPARSE_EMBEDDING_FAILED",
    "QDRANT_WRITE_FAILED",
    "S3_FETCH_FAILED",
    "INVALID_REQUEST",
    "PIPELINE_FAILED",
]


class IngestRequestJSON(BaseModel):
    """S3-reference mode body. Multipart mode uses raw form fields, not this model."""

    # extra="forbid" so a typo'd field name (s3_buckt) is a 400 instead of a
    # silently ignored no-op; strip whitespace so " " can't satisfy min_length.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    file_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    collection_name: str = Field(min_length=1)
    collection_type: str = "file"  # file | knowledge | memory | web-search | hash-based
    user_id: str = Field(min_length=1)
    overwrite: bool = True
    s3_bucket: str | None = None
    s3_key: str | None = None

    @model_validator(mode="after")
    def require_s3_reference(self):
        if not self.s3_bucket or not self.s3_key:
            raise ValueError(
                "s3_bucket and s3_key are required in JSON mode; "
                "use multipart/form-data to send a file directly."
            )
        return self


class ExtractionInfo(BaseModel):
    """How the document was extracted, surfaced on the ingest response.

    ``engine`` is the engine that ran. ``route`` carries the auto-router's
    signal + measured metrics (``{"signal": ..., "textboxes": ..., "ratio": ...}``)
    when ``EXTRACTION_ENGINE=auto``; it is ``None`` for a pinned engine, where
    no classification step runs.
    """

    engine: str
    route: dict | None = None


class IngestResponse(BaseModel):
    status: bool = True
    collection_name: str
    chunks_count: int
    # Omitted from the JSON when None (route sets response_model_exclude_none),
    # so the historical {status, collection_name, chunks_count} contract holds
    # for callers that don't care about extraction details.
    extraction: ExtractionInfo | None = None


class IngestError(BaseModel):
    status: bool = False
    error: str
    code: ErrorCode


class ChunkView(BaseModel):
    """One stored chunk, flattened for the inspection endpoint."""

    split_id: int | None = None
    content_length: int
    # Present per the ``include_content`` query param: full text, a preview, or
    # omitted entirely (None).
    content: str | None = None
    meta: dict = Field(default_factory=dict)


class DocumentChunkStats(BaseModel):
    """Document-level summary for the inspection endpoint.

    The doc-level fields (engine, route, languages, name) are identical across a
    document's chunks, so they're read from the first returned chunk.
    """

    total_chunks: int
    extraction_engine: str | None = None
    extraction_route: dict | None = None
    languages: list[str] | None = None
    name: str | None = None
    collection_name: str | None = None


class DocumentChunksResponse(BaseModel):
    """Response body of ``GET /api/v1/documents/{file_id}/chunks``."""

    status: bool = True
    file_id: str
    returned: int
    limit: int
    # Qdrant cursor for the next page; None when the document is exhausted.
    next_offset: str | None = None
    stats: DocumentChunkStats
    chunks: list[ChunkView]


class ExtractedDocument(BaseModel):
    """One Haystack ``Document`` flattened for JSON response.

    ``meta`` is whatever the converter attached (e.g. ``page`` for pypdf,
    file-level metadata for tika); no schema is imposed.
    """

    content: str
    meta: dict = Field(default_factory=dict)


class ExtractResponse(BaseModel):
    """Response body of ``POST /api/v1/extract`` — extraction-only probe."""

    status: bool = True
    engine: str
    # The vision-llm prompt profile used, when applicable; None for other engines.
    profile: str | None = None
    documents: list[ExtractedDocument]
