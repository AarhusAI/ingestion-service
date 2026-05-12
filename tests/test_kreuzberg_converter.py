"""Unit tests for ``KreuzbergRemoteConverter``.

Mocks the HTTP call to the kreuzberg sidecar via ``respx`` so the tests
don't need the container running. The component is exercised directly —
the route + pipeline tests live in ``test_ingest_endpoint.py`` and stay
engine-agnostic.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app.pipelines.kreuzberg_converter import KreuzbergRemoteConverter


@pytest.fixture
def tmp_source(tmp_path):
    p = tmp_path / "report.pdf"
    p.write_bytes(b"%PDF-fake")
    return str(p)


@respx.mock
def test_run_happy_path(tmp_source):
    """One source → one Document with content from the sidecar response."""
    respx.post("http://fake-kreuzberg:8000/extract").respond(
        200,
        json=[{"content": "hello world", "mime_type": "text/plain", "metadata": {}, "tables": []}],
    )

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    out = c.run(sources=[tmp_source], meta={"file_id": "abc"})["documents"]

    assert len(out) == 1
    assert out[0].content == "hello world"
    # Pipeline meta passes through verbatim; auto-extracted kreuzberg metadata
    # is intentionally dropped for v1.
    assert out[0].meta == {"file_id": "abc"}


@respx.mock
def test_run_per_source_meta_list(tmp_path):
    """A list of meta dicts is paired by index with each source."""
    a = tmp_path / "a.txt"
    a.write_text("a")
    b = tmp_path / "b.txt"
    b.write_text("b")

    route = respx.post("http://fake-kreuzberg:8000/extract")
    # Two requests; respx returns the same body each time which is fine —
    # we're checking pairing on the meta side.
    route.respond(
        200,
        json=[{"content": "extracted", "mime_type": "text/plain", "metadata": {}, "tables": []}],
    )

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    out = c.run(
        sources=[str(a), str(b)],
        meta=[{"file_id": "fa"}, {"file_id": "fb"}],
    )["documents"]

    assert [d.meta["file_id"] for d in out] == ["fa", "fb"]
    assert route.call_count == 2


@respx.mock
def test_run_meta_none(tmp_source):
    """Omitting meta is allowed; Documents get empty meta dicts."""
    respx.post("http://fake-kreuzberg:8000/extract").respond(
        200, json=[{"content": "x", "mime_type": "text/plain", "metadata": {}, "tables": []}]
    )

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    out = c.run(sources=[tmp_source])["documents"]

    assert out[0].meta == {}


@respx.mock
def test_run_object_payload_fallback(tmp_source):
    """Defensive against shape drift: a bare object (not an array) with a
    top-level ``content`` field is still accepted."""
    respx.post("http://fake-kreuzberg:8000/extract").respond(200, json={"content": "object-shape"})

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    out = c.run(sources=[tmp_source])["documents"]

    assert out[0].content == "object-shape"


@respx.mock
def test_run_sends_correct_mime_type_per_extension(tmp_path):
    """Kreuzberg dispatches on the multipart Content-Type, not file bytes.
    Sending ``application/octet-stream`` for a PDF trips ``UnsupportedFormatError``
    server-side (verified against goldziher/kreuzberg:4.0.7-core). Lock the
    contract: the per-part Content-Type must be derived from the extension."""
    pdf = tmp_path / "tmp_abc.pdf"
    pdf.write_bytes(b"%PDF-fake")
    docx = tmp_path / "report.docx"
    docx.write_bytes(b"PK\x03\x04docx-fake")

    route = respx.post("http://fake-kreuzberg:8000/extract").respond(200, json=[{"content": "x"}])

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    c.run(sources=[str(pdf)])
    c.run(sources=[str(docx)])

    pdf_body = route.calls[0].request.content.decode("utf-8", errors="replace")
    docx_body = route.calls[1].request.content.decode("utf-8", errors="replace")
    # The per-part Content-Type header lives inside the multipart body.
    assert "Content-Type: application/pdf" in pdf_body
    # mimetypes maps .docx → application/vnd.openxmlformats-officedocument.wordprocessingml.…
    assert "wordprocessingml" in docx_body
    # And the wrong header — the one that caused EXTRACTION_FAILED in prod — is gone.
    assert "Content-Type: application/octet-stream" not in pdf_body


@respx.mock
def test_run_uses_files_field_name(tmp_source):
    """The 4.0.x server requires the multipart field name to be ``files`` —
    the wrong name returns 400 'No files provided'. Lock the contract."""
    route = respx.post("http://fake-kreuzberg:8000/extract").respond(200, json=[{"content": "x"}])

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    c.run(sources=[tmp_source])

    body = route.calls[0].request.content.decode("utf-8", errors="replace")
    # Multipart body has ``name="files"`` for the file part.
    assert 'name="files"' in body


@respx.mock
def test_run_http_error_maps_to_runtime_error(tmp_source):
    """Connection refused (sidecar down) becomes a RuntimeError whose message
    contains "kreuzberg" / "extract" — that's what the route-layer error
    heuristic (``_classify_pipeline_error``) keys off to map to
    EXTRACTION_FAILED."""
    respx.post("http://fake-kreuzberg:8000/extract").mock(
        side_effect=httpx.ConnectError("connection refused")
    )

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    with pytest.raises(RuntimeError, match="kreuzberg extract failed"):
        c.run(sources=[tmp_source])


@respx.mock
def test_run_5xx_maps_to_runtime_error(tmp_source):
    respx.post("http://fake-kreuzberg:8000/extract").respond(500, text="boom")

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    with pytest.raises(RuntimeError, match="kreuzberg extract failed"):
        c.run(sources=[tmp_source])


def test_url_trailing_slash_normalised(tmp_path):
    """Constructor strips a trailing slash so the joined URL is always single-slashed."""
    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000/")
    assert c._url == "http://fake-kreuzberg:8000/extract"
