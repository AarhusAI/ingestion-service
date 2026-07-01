"""Chunk inspection endpoint (GET /api/v1/documents/{file_id}/chunks).

Qdrant scroll/count are patched at the route-layer symbols (same convention as
the ingest-endpoint tests) so no real Qdrant is touched.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _point(content: str, meta: dict):
    """A stand-in for a Qdrant scroll point: ``.payload`` with content + meta."""
    return SimpleNamespace(id="pt", payload={"content": content, "meta": meta})


@pytest.fixture
def enabled(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "enable_inspection_api", True)


async def test_disabled_returns_404(client, api_headers, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "enable_inspection_api", False)
    r = await client.get("/api/v1/documents/abc/chunks", headers=api_headers)
    assert r.status_code == 404


async def test_requires_auth(client, enabled):
    r = await client.get("/api/v1/documents/abc/chunks")
    assert r.status_code in (401, 403)


async def test_503_when_pipeline_not_ready(client, api_headers, enabled):
    with patch("app.routes.inspect.is_pipeline_ready", return_value=False):
        r = await client.get("/api/v1/documents/abc/chunks", headers=api_headers)
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "PIPELINE_FAILED"


async def test_invalid_include_content_400(client, api_headers, enabled):
    r = await client.get("/api/v1/documents/abc/chunks?include_content=bogus", headers=api_headers)
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "INVALID_REQUEST"


async def test_happy_path_returns_chunks_and_stats(client, api_headers, enabled):
    points = [
        _point(
            "Hello world",
            {
                "split_id": 0,
                "extraction_engine": "kreuzberg",
                "extraction_route": {"signal": "default"},
                "languages": ["da"],
                "name": "f.pdf",
                "collection_name": "file-abc",
            },
        ),
        _point("Second chunk", {"split_id": 1}),
    ]
    with (
        patch("app.routes.inspect.is_pipeline_ready", return_value=True),
        patch("app.routes.inspect.count_chunks_by_file_id", return_value=2),
        patch("app.routes.inspect.scroll_chunks_by_file_id", return_value=(points, None)),
    ):
        r = await client.get("/api/v1/documents/abc/chunks", headers=api_headers)

    assert r.status_code == 200
    body = r.json()
    assert body["file_id"] == "abc"
    assert body["returned"] == 2
    assert body["stats"]["total_chunks"] == 2
    assert body["stats"]["extraction_engine"] == "kreuzberg"
    assert body["stats"]["extraction_route"]["signal"] == "default"
    assert body["chunks"][0]["split_id"] == 0
    assert body["chunks"][0]["content_length"] == len("Hello world")
    # preview mode: short content returned whole
    assert body["chunks"][0]["content"] == "Hello world"


async def test_include_content_false_omits_text_but_keeps_length(client, api_headers, enabled):
    points = [_point("some text here", {"split_id": 0})]
    with (
        patch("app.routes.inspect.is_pipeline_ready", return_value=True),
        patch("app.routes.inspect.count_chunks_by_file_id", return_value=1),
        patch("app.routes.inspect.scroll_chunks_by_file_id", return_value=(points, None)),
    ):
        r = await client.get(
            "/api/v1/documents/abc/chunks?include_content=false", headers=api_headers
        )

    assert r.status_code == 200
    chunk = r.json()["chunks"][0]
    assert "content" not in chunk  # None → excluded
    assert chunk["content_length"] == len("some text here")


async def test_preview_truncates_long_content(client, api_headers, enabled):
    long_text = "x" * 500
    points = [_point(long_text, {"split_id": 0})]
    with (
        patch("app.routes.inspect.is_pipeline_ready", return_value=True),
        patch("app.routes.inspect.count_chunks_by_file_id", return_value=1),
        patch("app.routes.inspect.scroll_chunks_by_file_id", return_value=(points, None)),
    ):
        r = await client.get(
            "/api/v1/documents/abc/chunks?include_content=preview", headers=api_headers
        )

    chunk = r.json()["chunks"][0]
    assert chunk["content_length"] == 500
    assert chunk["content"].endswith("…")
    assert len(chunk["content"]) < 500
