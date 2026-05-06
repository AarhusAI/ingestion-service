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


async def test_non_bearer_scheme(client):
    response = await client.put(
        "/api/v1/ingest",
        json={"file_id": "x", "filename": "x", "collection_name": "x", "user_id": "x"},
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
    )
    # FastAPI's HTTPBearer rejects non-Bearer schemes with 401 or 403 depending on version.
    assert response.status_code in (401, 403)
