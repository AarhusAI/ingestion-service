from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration loaded from environment variables (and .env when present)."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # ----- Auth -----
    api_key: str

    # ----- Server -----
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # ----- Extraction -----
    # tika | pypdf | docling | unstructured
    # docling/unstructured require optional deps not bundled by default.
    extraction_engine: str = "tika"
    tika_url: str = "http://tika:9998"

    # ----- Chunking -----
    # token mode measures chunk size with the embedding model's HuggingFace
    # tokenizer (xlm-r for e5, sentencepiece for bge-m3, etc.) so chunks
    # respect the model's context window. word/sentence/passage delegate
    # to Haystack's built-in DocumentSplitter and count by approximate units.
    chunk_size: int = 400
    chunk_overlap: int = 80
    chunk_split_by: str = "token"  # token | word | sentence | passage
    # Optional override; empty falls back to embedding_model. Only used in
    # token mode.
    tokenizer_model: str = ""

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


settings = Settings()
