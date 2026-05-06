"""Endpoint behaviour: dispatch on Content-Type, error mapping, idempotency contract."""

from io import BytesIO
from unittest.mock import patch


async def test_json_mode_happy_path(client, api_headers, tmp_path):
    """JSON body fetches via S3, runs pipeline, returns chunks count."""
    fake_local = str(tmp_path / "fake.pdf")
    with open(fake_local, "wb") as fh:
        fh.write(b"%PDF-fake")

    with (
        patch(
            "app.routes.ingest.fetch_object_to_tempfile",
            return_value=fake_local,
        ),
        patch("app.routes.ingest.run_indexing_pipeline", return_value=42) as run_mock,
    ):
        response = await client.put(
            "/api/v1/ingest",
            json={
                "s3_bucket": "openwebui",
                "s3_key": "files/abc/report.pdf",
                "file_id": "abc",
                "filename": "report.pdf",
                "collection_name": "file-abc",
                "collection_type": "file",
                "user_id": "u-1",
                "overwrite": True,
            },
            headers=api_headers,
        )

    assert response.status_code == 200
    body = response.json()
    assert body == {"status": True, "collection_name": "file-abc", "chunks_count": 42}

    # Confirm the meta we passed through is what reaches the pipeline.
    args, _ = run_mock.call_args
    _path_arg, meta_arg = args
    assert meta_arg["file_id"] == "abc"
    assert meta_arg["collection_name"] == "file-abc"
    assert meta_arg["collection_type"] == "file"
    assert meta_arg["user_id"] == "u-1"
    assert meta_arg["overwrite"] is True
    assert meta_arg["name"] == "report.pdf"
    assert meta_arg["source"] == "report.pdf"


async def test_json_mode_missing_s3_reference(client, api_headers):
    """JSON without s3_bucket/s3_key returns 400 with INVALID_REQUEST code."""
    response = await client.put(
        "/api/v1/ingest",
        json={
            "file_id": "abc",
            "filename": "report.pdf",
            "collection_name": "file-abc",
            "user_id": "u-1",
        },
        headers=api_headers,
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "INVALID_REQUEST"


async def test_multipart_mode_happy_path(client, api_headers, tmp_path):
    """Multipart body streams to a tempfile and runs the pipeline."""
    files = {
        "file": ("report.pdf", BytesIO(b"%PDF-fake"), "application/pdf"),
    }
    data = {
        "file_id": "abc",
        "filename": "report.pdf",
        "collection_name": "file-abc",
        "collection_type": "file",
        "user_id": "u-1",
        "overwrite": "true",
    }

    with patch("app.routes.ingest.run_indexing_pipeline", return_value=7) as run_mock:
        response = await client.put("/api/v1/ingest", files=files, data=data, headers=api_headers)

    assert response.status_code == 200
    body = response.json()
    assert body == {"status": True, "collection_name": "file-abc", "chunks_count": 7}

    args, _ = run_mock.call_args
    path_arg, meta_arg = args
    # multipart writes to a real tempfile that exists at call time
    assert path_arg.endswith(".pdf")
    assert meta_arg["file_id"] == "abc"


async def test_unsupported_content_type(client, api_headers):
    response = await client.put(
        "/api/v1/ingest",
        content=b"raw bytes",
        headers={**api_headers, "Content-Type": "text/plain"},
    )
    assert response.status_code == 415
    detail = response.json()["detail"]
    assert detail["code"] == "INVALID_REQUEST"


async def test_s3_fetch_failure_maps_to_s3_fetch_failed(client, api_headers):
    with patch(
        "app.routes.ingest.fetch_object_to_tempfile",
        side_effect=RuntimeError("connection refused"),
    ):
        response = await client.put(
            "/api/v1/ingest",
            json={
                "s3_bucket": "openwebui",
                "s3_key": "missing.pdf",
                "file_id": "abc",
                "filename": "missing.pdf",
                "collection_name": "file-abc",
                "user_id": "u-1",
            },
            headers=api_headers,
        )
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["code"] == "S3_FETCH_FAILED"


async def test_pipeline_failure_maps_to_classified_error(client, api_headers, tmp_path):
    fake_local = str(tmp_path / "fake.pdf")
    with open(fake_local, "wb") as fh:
        fh.write(b"%PDF-fake")

    with (
        patch(
            "app.routes.ingest.fetch_object_to_tempfile",
            return_value=fake_local,
        ),
        patch(
            "app.routes.ingest.run_indexing_pipeline",
            side_effect=RuntimeError("Tika converter timeout"),
        ),
    ):
        response = await client.put(
            "/api/v1/ingest",
            json={
                "s3_bucket": "openwebui",
                "s3_key": "x.pdf",
                "file_id": "abc",
                "filename": "x.pdf",
                "collection_name": "file-abc",
                "user_id": "u-1",
            },
            headers=api_headers,
        )

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["code"] == "EXTRACTION_FAILED"
    assert detail["status"] is False
