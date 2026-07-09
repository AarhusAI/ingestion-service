import os

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings

# Minimum length enforced on ``API_KEY``. The bearer is a shared static
# secret with no per-IP rate limiting; below ~32 chars of entropy it
# becomes feasible to brute-force given enough request budget. Kept here
# (not in Settings) so the value is referenced from the validator without
# a self-reference cycle.
API_KEY_MIN_LENGTH = 32

# Concrete extraction engines the factory (``app/pipelines/converters.py``)
# knows how to build. ``"auto"`` is NOT in here — it is a routing *mode*, not a
# converter, so the auto-mode validator below must resolve the router's
# default/diagram engines against this set. Kept here (not in Settings) so both
# the validator and the factory can reference one source of truth.
KNOWN_EXTRACTION_ENGINES = frozenset(
    {"pypdf", "docling", "unstructured", "kreuzberg", "vision-llm", "hybrid-diagram"}
)


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
        "kreuzberg_url",
        "vision_llm_api_base_url",
        "gotenberg_url",
        "qdrant_uri",
        "embedding_api_base_url",
        "s3_endpoint_url",
    )
    @classmethod
    def _validate_http_url(cls, v: str) -> str:
        if not v:
            return v
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(f"URL must start with http:// or https://; got {v!r}")
        return v

    # ----- Routing engine-name validator -----
    # When routing is on (extraction_engine="auto"), the router's default and
    # diagram engines are real converters that get built at startup — so a typo
    # like extraction_router_diagram_engine="vsion-llm" must fail fast here, not
    # at the first diagram document. The route layer's _SUPPORTED_ENGINES only
    # gates /extract overrides; this is the only place that validates the
    # routing engine names.
    @model_validator(mode="after")
    def _validate_routing_engines(self):
        if self.extraction_engine.lower() != "auto":
            return self
        for field in ("extraction_router_default", "extraction_router_diagram_engine"):
            value = getattr(self, field).lower()
            if value not in KNOWN_EXTRACTION_ENGINES:
                raise ValueError(
                    f"{field}={getattr(self, field)!r} is not a known extraction engine "
                    f"(one of: {' | '.join(sorted(KNOWN_EXTRACTION_ENGINES))})"
                )
        return self

    # ----- Server -----
    host: str = "0.0.0.0"  # nosec B104 - container-internal bind; Traefik fronts it on the frontend network
    port: int = 8000
    # Operator-triage switch (NOT a logging dial — see LOG_LEVEL_APP for that).
    # When true, ingest error responses reflect ``str(exc)`` instead of a fixed
    # safe message, so local debugging sees the real failure. Leave false in
    # production: the raw text can carry hostnames, paths, and AWS request IDs.
    debug: bool = False

    # ----- Logging / observability -----
    # LOG_LEVEL is the primary verbosity dial. DEBUG | INFO | WARNING | ERROR |
    # CRITICAL — applied to the root logger (so third-party libs follow it too).
    log_level: str = "INFO"
    # Per-namespace override for the service's own loggers (``app.*``), applied
    # on top of LOG_LEVEL. Lets you run verbose app logs (e.g. routing detector
    # metrics) without the third-party DEBUG flood (httpx/boto3/haystack).
    # Empty = inherit the root level. DEBUG | INFO | WARNING | ERROR | CRITICAL.
    log_level_app: str = ""
    # text = human-readable single line (the historical format); json = one
    # JSON object per line for Loki / a structured-log pipeline.
    log_format: str = "text"
    # Expose Prometheus metrics at ``GET /metrics``. Unauthenticated by design
    # (same trust model as /health — internal network only); when false the
    # endpoint returns 404. Instrumentation always runs regardless; this only
    # gates exposure.
    metrics_enabled: bool = True
    # Expose the read-only chunk-inspection endpoint
    # (``GET /api/v1/documents/{file_id}/chunks``). It returns chunk *content*,
    # so it's opt-in; when false the endpoint returns 404. Bearer-auth applies
    # either way.
    enable_inspection_api: bool = False

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)}; got {v!r}")
        return upper

    @field_validator("log_level_app")
    @classmethod
    def _validate_log_level_app(cls, v: str) -> str:
        if not v:
            return ""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"LOG_LEVEL_APP must be one of {sorted(allowed)} or empty; got {v!r}")
        return upper

    @field_validator("log_format")
    @classmethod
    def _validate_log_format(cls, v: str) -> str:
        lower = v.lower()
        if lower not in {"text", "json"}:
            raise ValueError(f"LOG_FORMAT must be 'text' or 'json'; got {v!r}")
        return lower

    # Per-request upload size cap, in bytes. Applies to the multipart file
    # part (enforced by stream_upload_to_tempfile) and to S3 fetches
    # (enforced by a head_object size check). Default 100 MB — large enough
    # for typical PDFs/DOCX, small enough that an authenticated caller can't
    # flood /tmp.
    max_upload_bytes: int = 100 * 1024 * 1024

    # ----- Extraction -----
    # pypdf | docling | unstructured | kreuzberg | vision-llm | auto
    # kreuzberg runs as an external HTTP sidecar; the rest are in-process.
    # docling/unstructured require optional deps not bundled by default.
    # vision-llm renders pages and reconstructs structure via a multimodal LLM.
    # "auto" enables per-document routing (see app/pipelines/detectors.py +
    # routing_converter.py): drawing-heavy docx go to the diagram engine, the
    # rest to the router default. Any other value pins that single engine
    # (current behaviour — routing OFF).
    extraction_engine: str = "kreuzberg"
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
    # Minimum columns a kreuzberg-detected table's rendered markdown must have to
    # be appended as a ## Tables section. The 4.0.x detector false-fires on
    # multi-column PROSE and emits 1-column line-dumps that just duplicate the body
    # (cells carry no row/col geometry, so even real tables can degenerate to one
    # column). Default 2 drops that duplication while keeping genuinely
    # column-segmented tables. Set to 1 to restore keep-all behaviour.
    kreuzberg_min_table_columns: int = 2

    # ----- Content-based routing (EXTRACTION_ENGINE=auto) -----
    # Only consulted when extraction_engine == "auto". Defaults keep the
    # cheap-default contract (kreuzberg for ordinary docs) while sending
    # drawing-heavy docx (swim-lane flowcharts etc.) to the vision engine.
    extraction_router_default: str = "kreuzberg"
    # hybrid-diagram = native docx text (authoritative, complete labels) + a
    # vision-inferred Mermaid graph. Preferred over bare vision-llm for the
    # diagram route because the body text is verbatim from the package XML
    # instead of OCR-guessed; vision-llm stays available for forced use.
    extraction_router_diagram_engine: str = "hybrid-diagram"
    # Vision-LLM profile the diagram route uses. Pinned separately from
    # vision_llm_profile (the engine's own default) so changing the engine
    # default for forced/explicit use can't alter what auto-routing sends for
    # flowcharts. Validated against the profile registry at startup.
    #
    # NOTE: with the default diagram engine (hybrid-diagram) this is mostly
    # moot — for a real (text-bearing) docx the hybrid converter pins its own
    # `diagram-topology` profile and ignores this value. It still applies in
    # two narrow cases: (a) the hybrid fallback for a diagram-detected docx with
    # ~no native text, and (b) when extraction_router_diagram_engine=vision-llm,
    # where it fully selects the auto-routed flowchart profile (diagram/general/
    # ocr). Kept for (b): dropping it would force that path onto vision_llm_profile
    # (default "general", wrong for flowcharts), reintroducing the coupling the
    # separate pin avoids.
    extraction_router_diagram_profile: str = "diagram"
    # Detection thresholds (tunable per deployment without a code change).
    # An absolute floor on the number of drawing/textbox text-bearing shapes
    # so a couple of callout boxes in an otherwise normal document can't
    # trigger the expensive engine.
    extraction_router_min_textboxes: int = 20
    # Drawing/textbox text must be at least this many times the body word
    # count for a docx to route to the diagram engine: ratio = drawing /
    # (body + 1).
    extraction_router_drawing_ratio: float = 2.0
    # Raster-image signal (a SECOND trigger for the diagram route). A docx whose
    # key content is a flattened raster PNG diagram (referenced via <a:blip> in
    # word/document.xml) has zero textboxes, so the signal above never fires and
    # the figure is dropped by plain-text engines. This catches such docs; the
    # diagram converter then uses the `figure` vision profile (native verbatim
    # prose + a vision pass scoped to the embedded figure). Header/footer logos
    # are excluded for free — detection reads only word/document.xml.
    #
    # Minimum number of embedded body images (DrawingML <a:blip> in
    # word/document.xml) before the raster signal is considered. The motivating
    # doc has exactly one content-bearing diagram, so the default is 1; the
    # display-area floor below — not the count — is what rejects decorative
    # images.
    extraction_router_min_body_images: int = 1
    # Minimum rendered display AREA (EMU², from <wp:extent cx cy>) of the LARGEST
    # body image for the raster signal to fire. 914400 EMU = 1 inch, so
    # 836_127_360_000 EMU² = 1 in². Default 1.5e12 ≈ 1.79 in² (a ~3.4 cm square):
    # a real process diagram (the motivating doc renders ~4.3 in²) clears it with
    # margin, while a 16 px icon (~0.03 in²) or a 1-inch logo (1 in²) is well
    # below it. Uses the max single extent, not the sum, so many small inline
    # icons can't add up to a false trigger.
    extraction_router_min_image_emu: int = 1_500_000_000_000
    # Optional ratio gate: max_image_area_emu / (body_words + 1) must be at least
    # this for the raster signal to fire. Guards against a single large
    # DECORATIVE photo in an otherwise prose-heavy report. DEFAULT 0 = DISABLED:
    # the motivating diagram doc (1536 words, ~2.3e9 ratio) and a hero-photo report
    # land too close to cleanly separate without real-sample calibration, so a
    # guessed threshold risks routing a real diagram to the default engine. Lower
    # stakes now, too — a decorative photo that slips through to the figure profile
    # degrades gracefully (the vision pass returns nothing → native body alone, see
    # docx_images / VisionLLMConverter allow_empty) rather than failing the ingest.
    # TODO: calibrate a non-zero default against a few real prose+photo docs;
    # operators seeing decorative-photo false positives can raise it meanwhile.
    extraction_router_min_image_word_ratio: float = 0.0

    # ----- Vision LLM extraction (EXTRACTION_ENGINE=vision-llm) -----
    # Renders document pages to images and asks an OpenAI-compatible
    # multimodal endpoint to reconstruct the structure as Markdown + Mermaid.
    # Model id is operator-configured so the code stays model-agnostic; the
    # served model today is a 4-bit (NVFP4) Gemma. Office->PDF rendering is
    # delegated to the Gotenberg sidecar (below); PDF->PNG is local.
    vision_llm_api_base_url: str = ""
    vision_llm_api_key: str = ""
    vision_llm_model: str = "gemma4-nvfp4"
    vision_llm_connect_timeout: float = 5.0
    # Long read window: a quantized VLM reasoning over several page images is
    # slow. Connect still fails fast (a stalled endpoint shouldn't tie up a
    # worker for the whole read window).
    vision_llm_read_timeout: float = 180.0
    # Render resolution and a hard page cap. Higher dpi = more legible but more
    # image tokens; max_pages bounds reconstruction-quality drift and cost on
    # long documents. NOTE: on the hybrid-diagram `figure` path the embedded figure
    # is sent at its native package resolution (app/pipelines/docx_images.py), so
    # dpi does NOT govern figure sharpness there — it applies to the full-page
    # profiles (diagram/general/ocr) and the topology page render.
    vision_llm_dpi: int = 150
    vision_llm_max_pages: int = 20
    # Max output tokens for the multimodal call. The served vLLM runs with
    # --max-model-len 32768 (TOTAL context = image-input + prompt + output); page
    # images cost only a few thousand input tokens, so 16384 for output is safe and
    # fits multi-page docs. Output exceeding this fails the extraction
    # (finish_reason=length) rather than silently truncating — raise it (up to the
    # input headroom under max-model-len) for very long documents.
    vision_llm_max_tokens: int = 16384
    vision_llm_tls_verify: bool = True
    # Injected into the system prompt so the model keeps the document's source
    # language verbatim instead of translating.
    vision_llm_language_hint: str = "Danish"
    # The engine's default prompt profile (diagram | general | ocr), used for
    # forced/explicit vision-llm use (EXTRACTION_ENGINE=vision-llm or
    # /extract?engine=vision-llm without ?profile=). NOT what auto-routing uses
    # for flowcharts — that is extraction_router_diagram_profile. Validated
    # against the profile registry at startup.
    vision_llm_profile: str = "general"

    # ----- Document rendering (Gotenberg sidecar) -----
    # External container that converts office formats (docx/odt/rtf/pptx/...)
    # to PDF via headless LibreOffice. Same deployment model as the tika /
    # kreuzberg sidecars — keeps LibreOffice (and its cold start / profile
    # locks) out of this service's image. Used by the vision-llm engine.
    gotenberg_url: str = "http://gotenberg:3000"
    gotenberg_connect_timeout: float = 5.0
    # LibreOffice conversion is the slow part, so the read window is generous.
    gotenberg_read_timeout: float = 120.0
    gotenberg_tls_verify: bool = True

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
    # markdown mode only: sections smaller than this many tokens merge into
    # adjacent ones (never past chunk_size), so heading-dense docs don't
    # produce tiny chunks that embed poorly and waste Qdrant points. 0
    # disables merging. Changing it changes chunk boundaries, so reindex for
    # consistency.
    chunk_min_size: int = 100

    @model_validator(mode="after")
    def _validate_chunk_sizes(self):
        if self.chunk_min_size < 0:
            raise ValueError(f"CHUNK_MIN_SIZE must be >= 0; got {self.chunk_min_size}")
        if self.chunk_min_size > self.chunk_size:
            raise ValueError(
                f"CHUNK_MIN_SIZE ({self.chunk_min_size}) must not exceed "
                f"CHUNK_SIZE ({self.chunk_size})"
            )
        return self

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
    # Prepend the section-heading breadcrumb (meta.headers_breadcrumb, e.g.
    # "Setup > Docker > Networking") to the text the embedders see — stored
    # chunk content is untouched. Only has an effect with
    # CHUNK_SPLIT_BY=markdown (the only mode that stamps the field). Flipping
    # this changes vectors, so reindex for consistency.
    embed_headers_breadcrumb: bool = True

    # ----- Sparse embedder (optional) -----
    enable_sparse_embeddings: bool = False
    # fastembed | none
    sparse_embedding_provider: str = "fastembed"
    sparse_embedding_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"

    # ----- In-process embedder CPU budget -----
    # Caps the ONNX thread pool used by the in-process fastembed embedders
    # (dense when EMBEDDING_PROVIDER=fastembed, and the sparse embedder). ONNX
    # otherwise grabs every core, saturating CPU and starving the uvicorn event
    # loop so /health stops responding mid-ingest. 0 = auto (leave 2 cores free
    # for the event loop); a positive value pins the thread count exactly.
    # NOTE: auto reads the *available* core count (sched_getaffinity), which
    # honors cpuset pinning but NOT a CFS quota (docker `cpus:` / `--cpus`). If
    # you cap the container with `cpus:`, set EMBEDDING_THREADS explicitly.
    embedding_threads: int = 0

    def resolved_embedding_threads(self) -> int:
        """Concrete thread cap for the fastembed embedders (see embedding_threads)."""
        if self.embedding_threads > 0:
            return self.embedding_threads
        try:
            available = len(os.sched_getaffinity(0))
        except AttributeError:  # not available on this platform
            available = os.cpu_count() or 1
        return max(1, available - 2)

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
