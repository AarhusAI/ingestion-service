import os

# Override env vars BEFORE any app imports (Settings() runs at import time).
os.environ["API_KEY"] = "test-api-key-padded-to-min-length-x"
os.environ["EMBEDDING_API_BASE_URL"] = "http://fake-embedding:8080"
os.environ["EMBEDDING_API_KEY"] = "fake-key"
os.environ["EMBEDDING_MODEL"] = "intfloat/multilingual-e5-large"
os.environ["QDRANT_URI"] = "http://fake-qdrant:6333"
os.environ["KREUZBERG_URL"] = "http://fake-kreuzberg:8000"
# Pin the default extraction engine for tests so the suite is independent of
# whatever the live container is configured with (operators flipping
# EXTRACTION_ENGINE=auto shouldn't break tests that assert on defaults).
os.environ["EXTRACTION_ENGINE"] = "kreuzberg"
# Same reasoning for the routing knobs: a deployment that sets these in its .env
# (e.g. EXTRACTION_ROUTER_DIAGRAM_ENGINE=hybrid-diagram) must not leak into tests
# that build Settings(_env_file=None) — pydantic still reads os.environ. Pin them
# to the code defaults so routing/detector tests are deterministic.
os.environ["EXTRACTION_ROUTER_DEFAULT"] = "kreuzberg"
os.environ["EXTRACTION_ROUTER_DIAGRAM_ENGINE"] = "hybrid-diagram"
os.environ["EXTRACTION_ROUTER_DIAGRAM_PROFILE"] = "diagram"
os.environ["EXTRACTION_ROUTER_MIN_TEXTBOXES"] = "20"
os.environ["EXTRACTION_ROUTER_DRAWING_RATIO"] = "2.0"
os.environ["S3_ENDPOINT_URL"] = "http://fake-s3:9000"
os.environ["S3_ACCESS_KEY_ID"] = "fake-key"
os.environ["S3_SECRET_ACCESS_KEY"] = "fake-secret"
# Pin DEBUG off so error-redaction tests are independent of whatever the dev
# container's .env carries (operators flipping DEBUG=True locally shouldn't
# flip test expectations).
os.environ["DEBUG"] = "False"

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.services import s3 as s3_service

# Force settings to match test env (in case .env file or container env overrode them)
settings.api_key = "test-api-key-padded-to-min-length-x"


@pytest.fixture
def api_headers():
    return {"Authorization": "Bearer test-api-key-padded-to-min-length-x"}


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
