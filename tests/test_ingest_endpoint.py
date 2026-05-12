"""Endpoint behaviour: dispatch on Content-Type, error mapping, idempotency contract."""

from io import BytesIO
from unittest.mock import patch

import pytest


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


async def test_multipart_mode_happy_path(client, api_headers):
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


# ---------------------------------------------------------------------------
# Helper unit tests — exercise the private helpers directly so we don't have
# to round-trip a full request to cover one-line branches.
# ---------------------------------------------------------------------------


def test_form_bool_passes_through_bool():
    """`_form_bool` accepts a real bool without re-parsing it as a string."""
    from app.routes.ingest import _form_bool

    assert _form_bool(True) is True
    assert _form_bool(False) is False


class _PyPDFError(Exception):
    """Stand-in for the pypdf converter exception (name-match branch)."""


class _OpenAIError(Exception):
    """Stand-in for an embedder exception with 'OpenAI' in the class name."""


@pytest.mark.parametrize(
    "exc, expected",
    [
        # Extraction: matched by message keyword OR exception class name.
        (RuntimeError("Tika converter timeout"), "EXTRACTION_FAILED"),
        (RuntimeError("could not extract pages"), "EXTRACTION_FAILED"),
        (_PyPDFError("malformed page tree"), "EXTRACTION_FAILED"),
        # Qdrant: 'qdrant' or 'vector store' (or 'vector' + 'write').
        (RuntimeError("qdrant write failed"), "QDRANT_WRITE_FAILED"),
        (RuntimeError("vector store unreachable"), "QDRANT_WRITE_FAILED"),
        # Sparse stage must be checked before the generic embed branch.
        (RuntimeError("sparse embed call failed"), "SPARSE_EMBEDDING_FAILED"),
        # Generic embed by message OR by OpenAI-named exception class.
        (RuntimeError("embed endpoint returned 500"), "EMBEDDING_FAILED"),
        (_OpenAIError("opaque server error"), "EMBEDDING_FAILED"),
        # Fallback.
        (RuntimeError("nothing recognizable here"), "PIPELINE_FAILED"),
    ],
)
def test_classify_pipeline_error(exc, expected):
    """Heuristic mapping from a raised exception → `IngestError.code`."""
    from app.routes.ingest import _classify_pipeline_error

    assert _classify_pipeline_error(exc) == expected
