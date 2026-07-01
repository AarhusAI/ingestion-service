"""POST /api/v1/extract — engine override, validation, error mapping."""

from io import BytesIO
from unittest.mock import MagicMock, patch


def _fake_doc(content: str = "hello", meta: dict | None = None):
    """Stand-in for a haystack ``Document`` — only ``.content`` and ``.meta`` are read."""
    d = MagicMock()
    d.content = content
    d.meta = meta or {}
    return d


async def test_extract_happy_path_uses_configured_engine(client, api_headers):
    """No ``engine`` field: configured default is used, ``engine_override`` stays None."""
    fake_converter = MagicMock()
    fake_converter.run.return_value = {
        "documents": [_fake_doc("# Heading\n...", {"page": 1})],
    }

    with patch("app.routes.extract.build_converter", return_value=fake_converter) as build_mock:
        response = await client.post(
            "/api/v1/extract",
            files={"file": ("report.pdf", BytesIO(b"%PDF-fake"), "application/pdf")},
            headers=api_headers,
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] is True
    assert body["engine"] == "kreuzberg"  # the default in app.config.Settings
    assert body["documents"] == [{"content": "# Heading\n...", "meta": {"page": 1}}]
    # engine_override stays None when client omits the field
    _, kwargs = build_mock.call_args
    assert kwargs.get("engine_override") is None
    # The converter was called with the standard sources/meta shape
    _, run_kwargs = fake_converter.run.call_args
    assert "sources" in run_kwargs and len(run_kwargs["sources"]) == 1
    assert run_kwargs["sources"][0].endswith(".pdf")


async def test_extract_engine_override_passed_to_build_converter(client, api_headers):
    fake_converter = MagicMock()
    fake_converter.run.return_value = {"documents": [_fake_doc("ok")]}

    with patch("app.routes.extract.build_converter", return_value=fake_converter) as build_mock:
        response = await client.post(
            "/api/v1/extract",
            files={"file": ("a.pdf", BytesIO(b"%PDF-fake"), "application/pdf")},
            data={"engine": "pypdf"},
            headers=api_headers,
        )

    assert response.status_code == 200
    assert response.json()["engine"] == "pypdf"
    _, kwargs = build_mock.call_args
    assert kwargs["engine_override"] == "pypdf"


async def test_extract_unknown_engine_returns_400(client, api_headers):
    response = await client.post(
        "/api/v1/extract",
        files={"file": ("a.pdf", BytesIO(b"%PDF-fake"), "application/pdf")},
        data={"engine": "banana"},
        headers=api_headers,
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "INVALID_REQUEST"
    assert "banana" in detail["error"]


async def test_extract_kreuzberg_engine_is_accepted(client, api_headers):
    """Regression guard: the route-layer ``_SUPPORTED_ENGINES`` whitelist must
    include ``kreuzberg``. Forgetting this returned 400 INVALID_REQUEST with
    ``Unknown engine='kreuzberg' (supported: docling | pypdf |
    unstructured)`` even though the factory was wired correctly."""
    fake_converter = MagicMock()
    fake_converter.run.return_value = {"documents": [_fake_doc("ok")]}

    with patch("app.routes.extract.build_converter", return_value=fake_converter) as build_mock:
        response = await client.post(
            "/api/v1/extract",
            files={"file": ("a.txt", BytesIO(b"hello"), "text/plain")},
            data={"engine": "kreuzberg"},
            headers=api_headers,
        )
    assert response.status_code == 200
    assert response.json()["engine"] == "kreuzberg"
    _, kwargs = build_mock.call_args
    assert kwargs["engine_override"] == "kreuzberg"


async def test_extract_vision_llm_engine_is_accepted(client, api_headers):
    """Regression guard: ``vision-llm`` must be in the route-layer whitelist
    (``_SUPPORTED_ENGINES``) — otherwise a correctly-wired factory still 400s."""
    fake_converter = MagicMock()
    fake_converter.run.return_value = {"documents": [_fake_doc("ok")]}

    with patch("app.routes.extract.build_converter", return_value=fake_converter) as build_mock:
        response = await client.post(
            "/api/v1/extract",
            files={"file": ("a.docx", BytesIO(b"PK\x03\x04"), "application/octet-stream")},
            data={"engine": "vision-llm"},
            headers=api_headers,
        )
    assert response.status_code == 200
    assert response.json()["engine"] == "vision-llm"
    _, kwargs = build_mock.call_args
    assert kwargs["engine_override"] == "vision-llm"


async def test_extract_auto_is_rejected(client, api_headers):
    """``auto`` is a routing mode for ingest, not a concrete engine — /extract
    must 400 it rather than try to build a non-existent 'auto' converter."""
    response = await client.post(
        "/api/v1/extract",
        files={"file": ("a.pdf", BytesIO(b"%PDF-fake"), "application/pdf")},
        data={"engine": "auto"},
        headers=api_headers,
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_REQUEST"


async def test_extract_profile_forwarded_for_vision_llm(client, api_headers):
    fake_converter = MagicMock()
    fake_converter.accepts_profile = True
    fake_converter.run.return_value = {
        "documents": [_fake_doc("# md", {"vision_profile": "ocr"})],
    }
    with patch("app.routes.extract.build_converter", return_value=fake_converter):
        response = await client.post(
            "/api/v1/extract",
            files={"file": ("a.pdf", BytesIO(b"%PDF-fake"), "application/pdf")},
            data={"engine": "vision-llm", "profile": "ocr"},
            headers=api_headers,
        )
    assert response.status_code == 200
    body = response.json()
    assert body["profile"] == "ocr"
    _, run_kwargs = fake_converter.run.call_args
    assert run_kwargs.get("profile") == "ocr"


async def test_extract_unknown_profile_returns_400(client, api_headers):
    response = await client.post(
        "/api/v1/extract",
        files={"file": ("a.pdf", BytesIO(b"%PDF-fake"), "application/pdf")},
        data={"engine": "vision-llm", "profile": "banana"},
        headers=api_headers,
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "INVALID_REQUEST"
    assert "banana" in detail["error"]


async def test_extract_profile_with_non_vision_engine_returns_400(client, api_headers):
    fake_converter = MagicMock()
    fake_converter.accepts_profile = False
    fake_converter.run.return_value = {"documents": [_fake_doc("ok")]}
    with patch("app.routes.extract.build_converter", return_value=fake_converter):
        response = await client.post(
            "/api/v1/extract",
            files={"file": ("a.pdf", BytesIO(b"%PDF-fake"), "application/pdf")},
            data={"engine": "pypdf", "profile": "ocr"},
            headers=api_headers,
        )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "INVALID_REQUEST"
    assert "vision-llm" in detail["error"]


async def test_extract_missing_file_returns_400(client, api_headers):
    # Multipart request body, but no ``file`` field — only ``engine``.
    response = await client.post(
        "/api/v1/extract",
        files={"not_file": ("a.pdf", BytesIO(b"x"), "application/pdf")},
        data={"engine": "pypdf"},
        headers=api_headers,
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_REQUEST"


async def test_extract_converter_failure_returns_500_extraction_failed(client, api_headers):
    fake_converter = MagicMock()
    fake_converter.run.side_effect = RuntimeError("kreuzberg unreachable")

    with patch("app.routes.extract.build_converter", return_value=fake_converter):
        response = await client.post(
            "/api/v1/extract",
            files={"file": ("a.pdf", BytesIO(b"%PDF-fake"), "application/pdf")},
            headers=api_headers,
        )
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "EXTRACTION_FAILED"


async def test_extract_missing_optional_dep_returns_400(client, api_headers):
    """ImportError from ``build_converter`` (optional dep missing) → 400 INVALID_REQUEST."""
    with patch(
        "app.routes.extract.build_converter",
        side_effect=ImportError(
            "EXTRACTION_ENGINE=docling requires the 'docling-haystack' package."
        ),
    ):
        response = await client.post(
            "/api/v1/extract",
            files={"file": ("a.pdf", BytesIO(b"%PDF-fake"), "application/pdf")},
            data={"engine": "docling"},
            headers=api_headers,
        )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "INVALID_REQUEST"
    assert "docling-haystack" in detail["error"]


async def test_extract_requires_auth(client):
    """No Authorization header → auth rejection (401/403, depending on FastAPI version)."""
    response = await client.post(
        "/api/v1/extract",
        files={"file": ("a.pdf", BytesIO(b"%PDF-fake"), "application/pdf")},
    )
    assert response.status_code in (401, 403)


async def test_extract_engine_field_is_case_insensitive(client, api_headers):
    """Mixed-case engine names are normalized (mirrors ``build_converter``)."""
    fake_converter = MagicMock()
    fake_converter.run.return_value = {"documents": [_fake_doc("ok")]}

    with patch("app.routes.extract.build_converter", return_value=fake_converter) as build_mock:
        response = await client.post(
            "/api/v1/extract",
            files={"file": ("a.pdf", BytesIO(b"%PDF-fake"), "application/pdf")},
            data={"engine": "PyPDF"},
            headers=api_headers,
        )

    assert response.status_code == 200
    assert response.json()["engine"] == "pypdf"
    _, kwargs = build_mock.call_args
    assert kwargs["engine_override"] == "pypdf"
