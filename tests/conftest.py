import os

# Override env vars BEFORE any app imports (Settings() runs at import time).
os.environ["API_KEY"] = "test-api-key"
os.environ["EMBEDDING_API_BASE_URL"] = "http://fake-embedding:8080"
os.environ["EMBEDDING_API_KEY"] = "fake-key"
os.environ["EMBEDDING_MODEL"] = "intfloat/multilingual-e5-large"
os.environ["QDRANT_URI"] = "http://fake-qdrant:6333"
os.environ["TIKA_URL"] = "http://fake-tika:9998"
os.environ["S3_ENDPOINT_URL"] = "http://fake-s3:9000"
os.environ["S3_ACCESS_KEY_ID"] = "fake-key"
os.environ["S3_SECRET_ACCESS_KEY"] = "fake-secret"

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.services import s3 as s3_service

# Force settings to match test env (in case .env file or container env overrode them)
settings.api_key = "test-api-key"


@pytest.fixture
def api_headers():
    return {"Authorization": "Bearer test-api-key"}


@pytest.fixture
async def client():
    # Imported lazily so individual tests can monkeypatch lifespan deps.
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def reset_clients():
    """Drop cached service clients between tests so settings overrides take effect."""
    yield
    s3_service.reset_client()
