from typing import Literal

from pydantic import BaseModel, Field, model_validator

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


class IngestResponse(BaseModel):
    status: bool = True
    collection_name: str
    chunks_count: int


class IngestError(BaseModel):
    status: bool = False
    error: str
    code: ErrorCode


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
    documents: list[ExtractedDocument]
