"""Bearer auth tests against the /api/v1/ingest endpoint."""


async def test_missing_authorization_header(client):
    response = await client.put(
        "/api/v1/ingest",
        json={"file_id": "x", "filename": "x", "collection_name": "x", "user_id": "x"},
    )
    # FastAPI's HTTPBearer returns 401 (modern) or 403 (older) — both are
    # valid "not authenticated" responses; we just want it rejected.
    assert response.status_code in (401, 403)


async def test_wrong_api_key(client):
    response = await client.put(
        "/api/v1/ingest",
        json={"file_id": "x", "filename": "x", "collection_name": "x", "user_id": "x"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"


def test_settings_rejects_short_api_key(monkeypatch):
    """A sub-threshold API_KEY must fail Settings() construction.

    Regression test for sec.md Finding 6: the bearer is a shared static
    secret with no rate limiting, so we refuse to start with a key short
    enough to brute-force."""
    from pydantic import ValidationError

    from app.config import Settings

    monkeypatch.setenv("API_KEY", "short")
    monkeypatch.setenv("EMBEDDING_API_BASE_URL", "http://x")
    monkeypatch.setenv("EMBEDDING_API_KEY", "x")
    try:
        Settings(_env_file=None)
    except ValidationError as exc:
        assert "API_KEY must be at least" in str(exc)
    else:
        raise AssertionError("Settings() should have refused a short API_KEY")


async def test_non_bearer_scheme(client):
    response = await client.put(
        "/api/v1/ingest",
        json={"file_id": "x", "filename": "x", "collection_name": "x", "user_id": "x"},
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
    )
    # FastAPI's HTTPBearer rejects non-Bearer schemes with 401 or 403 depending on version.
    assert response.status_code in (401, 403)
