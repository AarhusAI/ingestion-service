"""Delete endpoint (DELETE /api/v1/documents/{file_id}).

``delete_by_file_id`` is patched at the route-layer symbol (same convention as
the ingest / inspect endpoint tests) so no real Qdrant is touched.
"""

from unittest.mock import patch


async def test_requires_auth(client):
    r = await client.delete("/api/v1/documents/abc")
    assert r.status_code in (401, 403)


async def test_503_when_pipeline_not_ready(client, api_headers):
    with patch("app.routes.delete.is_pipeline_ready", return_value=False):
        r = await client.delete("/api/v1/documents/abc", headers=api_headers)
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "PIPELINE_FAILED"


async def test_happy_path_reports_deleted_count(client, api_headers):
    with (
        patch("app.routes.delete.is_pipeline_ready", return_value=True),
        patch("app.routes.delete.delete_by_file_id", return_value=3) as del_mock,
    ):
        r = await client.delete("/api/v1/documents/abc", headers=api_headers)

    assert r.status_code == 200
    body = r.json()
    assert body["status"] is True
    assert body["file_id"] == "abc"
    assert body["chunks_deleted"] == 3
    del_mock.assert_called_once_with("abc")


async def test_idempotent_unknown_file_id_returns_200_zero(client, api_headers):
    """Deleting an unknown / already-gone file_id is a no-op 200, not a 404."""
    with (
        patch("app.routes.delete.is_pipeline_ready", return_value=True),
        patch("app.routes.delete.delete_by_file_id", return_value=0),
    ):
        r = await client.delete("/api/v1/documents/does-not-exist", headers=api_headers)

    assert r.status_code == 200
    assert r.json()["chunks_deleted"] == 0


async def test_qdrant_failure_maps_to_delete_failed_500(client, api_headers):
    with (
        patch("app.routes.delete.is_pipeline_ready", return_value=True),
        patch(
            "app.routes.delete.delete_by_file_id",
            side_effect=RuntimeError("qdrant down"),
        ),
    ):
        r = await client.delete("/api/v1/documents/abc", headers=api_headers)

    assert r.status_code == 500
    assert r.json()["detail"]["code"] == "DELETE_FAILED"
