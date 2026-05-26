from pydantic import field_validator
from pydantic_settings import BaseSettings

# Minimum length enforced on ``API_KEY``. The bearer is a shared static
# secret with no per-IP rate limiting; below ~32 chars of entropy it
# becomes feasible to brute-force given enough request budget. Kept here
# (not in Settings) so the value is referenced from the validator without
# a self-reference cycle.
API_KEY_MIN_LENGTH = 32


class Settings(BaseSettings):
    """Configuration loaded from environment variables (and .env when present)."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # ----- Auth -----
    api_key: str

    @field_validator("api_key")
    @classmethod
    def _api_key_min_length(cls, v: str) -> str:
        if len(v) < API_KEY_MIN_LENGTH:
            raise ValueError(
                f"API_KEY must be at least {API_KEY_MIN_LENGTH} characters. "
                f"Got {len(v)}. Generate a high-entropy token with: "
                "openssl rand -hex 32"
            )
        return v

    # ----- Shared URL scheme validator -----
    # Applied to every URL-typed setting so a typo'd or maliciously-pointed
    # value (`file://`, `gopher://`, a missing scheme) is rejected at
    # startup rather than discovered at first connect. Fields stay typed as
    # ``str`` so downstream consumers (httpx, boto3) don't need to coerce
    # from a ``HttpUrl`` object. Empty strings are accepted because
    # ``s3_endpoint_url`` and ``embedding_api_base_url`` use "" to mean
    # "use the SDK's default endpoint resolution".
    @field_validator(
        "tika_url",
        "kreuzberg_url",
        "qdrant_uri",
        "embedding_api_base_url",
        "s3_endpoint_url",
    )
    @classmethod
    def _validate_http_url(cls, v: str) -> str:
        if not v:
            return v
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(
                f"URL must start with http:// or https://; got {v!r}"
            )
        return v

    # ----- Server -----
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Per-request upload size cap, in bytes. Applies to the multipart file
    # part (enforced by stream_upload_to_tempfile) and to S3 fetches
    # (enforced by a head_object size check). Default 100 MB — large enough
    # for typical PDFs/DOCX, small enough that an authenticated caller can't
    # flood /tmp.
    max_upload_bytes: int = 100 * 1024 * 1024

    # ----- Extraction -----
    # tika | pypdf | docling | unstructured | kreuzberg
    # tika/kreuzberg run as external HTTP sidecars; the rest are in-process.
    # docling/unstructured require optional deps not bundled by default.
    extraction_engine: str = "tika"
    tika_url: str = "http://tika:9998"
    kreuzberg_url: str = "http://kreuzberg:8000"
    # Split connect vs read timeout for the Kreuzberg sidecar. A single
    # 60-second blanket value (the previous default) means a sidecar that
    # accepts the TCP handshake but never replies ties up a worker for the
    # full read window. Connect should fail fast; read can be long because
    # the actual extraction is CPU-bound.
    kreuzberg_connect_timeout: float = 5.0
    kreuzberg_read_timeout: float = 60.0
    # Explicit setting so a future operator pointing the sidecar at an
    # https:// URL with a self-signed cert can't quietly disable
    # verification with verify=False. Default True; set to false only with
    # full awareness.
    kreuzberg_tls_verify: bool = True

    # ----- Chunking -----
    # token mode measures chunk size with the embedding model's HuggingFace
    # tokenizer (xlm-r for e5, sentencepiece for bge-m3, etc.) so chunks
    # respect the model's context window. markdown mode splits on Markdown
    # headers first and token-packs each section, preserving a `meta.headers`
    # breadcrumb — useful when the converter emits Markdown (Docling natively).
    # word/sentence/passage delegate to Haystack's built-in DocumentSplitter
    # and count by approximate units.
    chunk_size: int = 400
    chunk_overlap: int = 80
    chunk_split_by: str = "token"  # token | markdown | word | sentence | passage
    # Optional override; empty falls back to embedding_model. Used in token
    # and markdown modes.
    tokenizer_model: str = ""
    # Pin the HuggingFace Hub revision (branch, tag, or commit SHA) used when
    # downloading the tokenizer. Empty = whatever HEAD resolves to at fetch
    # time. Pinning a commit SHA in production prevents an upstream model swap
    # or HF account compromise from silently changing tokenizer behaviour.
    tokenizer_revision: str = ""

    # ----- Dense embedder (required) -----
    # openai-compat | fastembed | tei
    embedding_provider: str = "openai-compat"
    embedding_api_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_dim: int = 1024
    # Required by the model card. e5: "passage: " on docs, "query: " on queries; bge-m3 takes none.
    embedding_prefix_doc: str = "passage: "
    # Not used at indexing time; kept here so the contract is documented in one
    # place and the retrieval agent's prefix can be sanity-checked against ours.
    embedding_prefix_query: str = "query: "

    # ----- Sparse embedder (optional) -----
    enable_sparse_embeddings: bool = False
    # fastembed | none
    sparse_embedding_provider: str = "fastembed"
    sparse_embedding_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"

    # ----- Qdrant -----
    qdrant_uri: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None
    qdrant_index: str = "ingestion_files"

    # ----- S3 (boto3-standard names) -----
    s3_endpoint_url: str = ""
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_region: str = "us-east-1"
    # Comma-separated allow-list of buckets the service is permitted to fetch
    # from. Empty (default) = no enforcement, log a startup warning. Set this
    # in production so a stolen API key or a future regression in the caller
    # can't be used to exfiltrate arbitrary objects from the configured S3
    # credentials' reach.
    s3_allowed_buckets: str = ""

    @property
    def allowed_buckets(self) -> set[str]:
        return {b.strip() for b in self.s3_allowed_buckets.split(",") if b.strip()}


settings = Settings()
