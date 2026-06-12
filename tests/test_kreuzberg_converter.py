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
def test_run_http_error_raises_extraction_error(tmp_source):
    """Connection refused (sidecar down) becomes an ``ExtractionError`` — the
    typed error the route-layer classifier dispatches on (sec.md Finding 5)."""
    from app.pipelines.errors import ExtractionError

    respx.post("http://fake-kreuzberg:8000/extract").mock(
        side_effect=httpx.ConnectError("connection refused")
    )

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    with pytest.raises(ExtractionError, match="kreuzberg extract failed"):
        c.run(sources=[tmp_source])


@respx.mock
def test_run_5xx_raises_extraction_error(tmp_source):
    from app.pipelines.errors import ExtractionError

    respx.post("http://fake-kreuzberg:8000/extract").respond(500, text="boom")

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    with pytest.raises(ExtractionError, match="kreuzberg extract failed"):
        c.run(sources=[tmp_source])


def test_url_trailing_slash_normalised(tmp_path):
    """Constructor strips a trailing slash so the joined URL is always single-slashed."""
    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000/")
    assert c._url == "http://fake-kreuzberg:8000/extract"


def test_constructor_splits_connect_and_read_timeouts():
    """Connect and read timeouts must land as separate fields on the httpx.Timeout.

    Regression test for sec.md Finding 9: a single blanket timeout meant a
    stalled sidecar tied up the worker for the full read window."""
    c = KreuzbergRemoteConverter(
        kreuzberg_url="http://x",
        connect_timeout=3.0,
        read_timeout=45.0,
    )
    assert c._timeout.connect == 3.0
    assert c._timeout.read == 45.0
    # Default verify=True.
    assert c._verify is True


def test_constructor_honours_verify_false():
    """An explicit verify=False is preserved (not silently re-enabled)."""
    c = KreuzbergRemoteConverter(kreuzberg_url="http://x", verify=False)
    assert c._verify is False


def test_factory_threads_timeout_and_verify_settings_through(monkeypatch):
    """build_converter must pass the kreuzberg_* settings into the converter."""
    from app.config import settings
    from app.pipelines.converters import build_converter

    monkeypatch.setattr(settings, "kreuzberg_connect_timeout", 7.5)
    monkeypatch.setattr(settings, "kreuzberg_read_timeout", 90.0)
    monkeypatch.setattr(settings, "kreuzberg_tls_verify", False)

    c = build_converter(settings, engine_override="kreuzberg")
    assert c._timeout.connect == 7.5
    assert c._timeout.read == 90.0
    assert c._verify is False


# ---------------------------------------------------------------------------
# Tables rendering — Kreuzberg returns tables as a separate structured array.
# We rescue them from the body-content flattening by appending as Markdown.
# ---------------------------------------------------------------------------


@respx.mock
def test_content_appends_tables_section(tmp_source):
    """Body + two tables → body, blank line, ``## Tables`` heading, each
    table under its own ``### Table N (page X)`` heading."""
    respx.post("http://fake-kreuzberg:8000/extract").respond(
        200,
        json=[
            {
                "content": "Body paragraph here.",
                "tables": [
                    {
                        "markdown": "| Region | Q1 |\n| --- | --- |\n| Nordic | 100 |\n",
                        "page_number": 1,
                    },
                    {
                        "markdown": "| Category | Amount |\n| --- | --- |\n| Salary | 500 |\n",
                        "page_number": 2,
                    },
                ],
            }
        ],
    )

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    out = c.run(sources=[tmp_source])["documents"]

    content = out[0].content
    assert content.startswith("Body paragraph here.\n\n## Tables\n\n")
    assert "### Table 1 (page 1)\n\n| Region | Q1 |" in content
    assert "### Table 2 (page 2)\n\n| Category | Amount |" in content


@respx.mock
def test_content_unchanged_when_tables_empty(tmp_source):
    """``"tables": []`` → content is just the body, no ``## Tables`` heading.
    Regression guard for the bulk of documents without tables."""
    respx.post("http://fake-kreuzberg:8000/extract").respond(
        200,
        json=[{"content": "plain body, no tables", "tables": []}],
    )

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    out = c.run(sources=[tmp_source])["documents"]

    assert out[0].content == "plain body, no tables"
    assert "## Tables" not in out[0].content


@respx.mock
def test_content_handles_missing_tables_key(tmp_source):
    """No ``tables`` key at all (older / variant builds) → safe fallback."""
    respx.post("http://fake-kreuzberg:8000/extract").respond(200, json=[{"content": "just body"}])

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    out = c.run(sources=[tmp_source])["documents"]

    assert out[0].content == "just body"


@respx.mock
def test_content_skips_table_without_markdown(tmp_source):
    """A table missing the ``markdown`` field is silently dropped (we don't
    fall back to rendering from ``cells`` ourselves). Numbering stays
    monotonic across **successful** renders — the surviving table is
    numbered 1, not 2."""
    respx.post("http://fake-kreuzberg:8000/extract").respond(
        200,
        json=[
            {
                "content": "body",
                "tables": [
                    {"cells": [["A", "B"]], "page_number": 1},  # no markdown → skip
                    {"markdown": "| X | Y |\n| --- | --- |\n| 1 | 2 |\n", "page_number": 5},
                ],
            }
        ],
    )

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    out = c.run(sources=[tmp_source])["documents"]

    content = out[0].content
    assert "### Table 1 (page 5)" in content
    # No "Table 2" — only one table was successfully rendered.
    assert "### Table 2" not in content


@respx.mock
def test_content_table_without_page_number(tmp_source):
    """Missing ``page_number`` → heading is ``### Table N`` with no page suffix."""
    respx.post("http://fake-kreuzberg:8000/extract").respond(
        200,
        json=[
            {
                "content": "body",
                "tables": [{"markdown": "| X | Y |\n| --- | --- |\n| 1 | 2 |\n"}],
            }
        ],
    )

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    out = c.run(sources=[tmp_source])["documents"]

    assert "### Table 1\n\n| X | Y |" in out[0].content
    assert "(page" not in out[0].content


@respx.mock
def test_content_no_tables_heading_when_all_invalid(tmp_source):
    """If every table fails to render, do not emit a bare ``## Tables`` heading."""
    respx.post("http://fake-kreuzberg:8000/extract").respond(
        200,
        json=[
            {
                "content": "body",
                "tables": [
                    {"cells": [["A"]]},  # no markdown
                    "not-even-a-dict",
                    {"markdown": ""},  # empty markdown
                ],
            }
        ],
    )

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    out = c.run(sources=[tmp_source])["documents"]

    assert out[0].content == "body"
    assert "## Tables" not in out[0].content


@respx.mock
def test_drops_single_column_table(tmp_source):
    """The 4.0.x failure mode: a multi-row 1-column line-dump (prose mis-detected
    as a table). Gated out by the default min_table_columns=2 → body only."""
    respx.post("http://fake-kreuzberg:8000/extract").respond(
        200,
        json=[
            {
                "content": "the real body text",
                "tables": [
                    {
                        "markdown": "| prose line one |\n| --- |\n| line two |\n| line three |\n",
                        "page_number": 2,
                    }
                ],
            }
        ],
    )

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    out = c.run(sources=[tmp_source])["documents"]

    assert out[0].content == "the real body text"
    assert "## Tables" not in out[0].content


@respx.mock
def test_drops_header_only_table(tmp_source):
    """Two columns but no data rows (e.g. a split title) → dropped."""
    respx.post("http://fake-kreuzberg:8000/extract").respond(
        200,
        json=[
            {
                "content": "body",
                "tables": [{"markdown": "| A | B | C |\n| --- | --- | --- |\n", "page_number": 1}],
            }
        ],
    )

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    out = c.run(sources=[tmp_source])["documents"]

    assert "## Tables" not in out[0].content


@respx.mock
def test_min_columns_one_keeps_single_column(tmp_source):
    """Escape hatch: min_table_columns=1 restores the pre-gate keep-all behaviour."""
    respx.post("http://fake-kreuzberg:8000/extract").respond(
        200,
        json=[
            {
                "content": "body",
                "tables": [
                    {"markdown": "| only column |\n| --- |\n| a |\n| b |\n", "page_number": 3}
                ],
            }
        ],
    )

    c = KreuzbergRemoteConverter(
        kreuzberg_url="http://fake-kreuzberg:8000", min_table_columns=1
    )
    out = c.run(sources=[tmp_source])["documents"]

    assert "## Tables" in out[0].content
    assert "### Table 1 (page 3)" in out[0].content


# ---------------------------------------------------------------------------
# Document-level metadata enrichment — title / subject / authors / created_at
# from ``result.metadata`` plus ``languages`` from top-level
# ``detected_languages``. Curated whitelist; everything else stays internal
# to Kreuzberg.
# ---------------------------------------------------------------------------


@respx.mock
def test_meta_includes_doc_metadata_whitelist(tmp_source):
    """Title / subject / authors / created_at / languages all land in Document.meta
    when the response carries them."""
    respx.post("http://fake-kreuzberg:8000/extract").respond(
        200,
        json=[
            {
                "content": "body",
                "metadata": {
                    "title": "Privacy by Design",
                    "subject": "Implementation Guidance",
                    "authors": ["Fred Carter"],
                    "created_at": "2010-11-02T15:06:47Z",
                    # Below this line: things we deliberately drop.
                    "created_by": "Writer",
                    "producer": "OpenOffice.org",
                    "page_count": 5,
                    "quality_score": 1.0,
                    "is_encrypted": False,
                    "width": 612,
                    "height": 792,
                },
                "detected_languages": ["en", "da"],
            }
        ],
    )

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    out = c.run(sources=[tmp_source], meta={"file_id": "f-1"})["documents"]

    m = out[0].meta
    assert m["title"] == "Privacy by Design"
    assert m["subject"] == "Implementation Guidance"
    assert m["authors"] == ["Fred Carter"]
    assert m["created_at"] == "2010-11-02T15:06:47Z"
    assert m["languages"] == ["en", "da"]
    # File-level request meta still authoritative.
    assert m["file_id"] == "f-1"
    # Blacklisted fields never leak into payload — they'd bloat Qdrant for nothing.
    for forbidden in (
        "created_by",
        "producer",
        "page_count",
        "quality_score",
        "is_encrypted",
        "width",
        "height",
    ):
        assert forbidden not in m


@respx.mock
def test_meta_omits_missing_or_empty_doc_metadata(tmp_source):
    """No metadata at all → only request meta. Empty/null values in metadata are
    dropped silently (no ``"subject": ""`` noise)."""
    respx.post("http://fake-kreuzberg:8000/extract").respond(
        200,
        json=[
            {
                "content": "body",
                "metadata": {
                    "title": "",  # empty string → dropped
                    "subject": None,  # null → dropped
                    "authors": [],  # empty list → dropped
                },
                "detected_languages": None,  # null → no `languages` key
            }
        ],
    )

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    out = c.run(sources=[tmp_source], meta={"file_id": "f-2"})["documents"]

    m = out[0].meta
    for absent in ("title", "subject", "authors", "created_at", "languages"):
        assert absent not in m, f"{absent} should not be in meta"
    # Request meta survives untouched.
    assert m == {"file_id": "f-2"}


@respx.mock
def test_meta_request_meta_wins_on_collision(tmp_source):
    """If the Kreuzberg metadata happens to use a key the request meta also
    uses (e.g. ``title``), the request value wins — the route layer is the
    contract authority."""
    respx.post("http://fake-kreuzberg:8000/extract").respond(
        200,
        json=[
            {
                "content": "body",
                "metadata": {"title": "Kreuzberg-detected Title"},
                "detected_languages": ["da"],
            }
        ],
    )

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    out = c.run(
        sources=[tmp_source],
        meta={"file_id": "f-3", "title": "Request-Override Title"},
    )["documents"]

    assert out[0].meta["title"] == "Request-Override Title"
    # Non-colliding doc-meta still passes through.
    assert out[0].meta["languages"] == ["da"]


@respx.mock
def test_meta_handles_missing_metadata_key(tmp_source):
    """Response with no ``metadata`` key at all (older / variant builds) doesn't crash."""
    respx.post("http://fake-kreuzberg:8000/extract").respond(
        200,
        json=[{"content": "body", "detected_languages": ["en"]}],
    )

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    out = c.run(sources=[tmp_source])["documents"]

    # `languages` still picked up from top level; no crash on missing `metadata`.
    assert out[0].meta == {"languages": ["en"]}


@respx.mock
def test_meta_languages_empty_list_dropped(tmp_source):
    """``detected_languages: []`` is treated as "nothing detected" and dropped —
    we never emit ``"languages": []``."""
    respx.post("http://fake-kreuzberg:8000/extract").respond(
        200,
        json=[{"content": "body", "metadata": {}, "detected_languages": []}],
    )

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    out = c.run(sources=[tmp_source])["documents"]

    assert "languages" not in out[0].meta


@respx.mock
def test_content_tables_only_no_body(tmp_source):
    """If body is empty but tables exist, the rendered section stands alone —
    no leading blank line, no stray whitespace."""
    respx.post("http://fake-kreuzberg:8000/extract").respond(
        200,
        json=[
            {
                "content": "",
                "tables": [
                    {"markdown": "| X | Y |\n| --- | --- |\n| 1 | 2 |\n", "page_number": 1}
                ],
            }
        ],
    )

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    out = c.run(sources=[tmp_source])["documents"]

    assert out[0].content.startswith("## Tables\n\n### Table 1 (page 1)")


@respx.mock
def test_unusable_payload_yields_empty_content_with_warning(tmp_source, caplog):
    """A response with no usable extraction result (shape drift) becomes an
    empty Document — and logs a warning, since the ingest will otherwise
    "succeed" with zero chunks and nothing else flags the drift."""
    import logging

    respx.post("http://fake-kreuzberg:8000/extract").respond(200, json=["not-a-dict"])

    c = KreuzbergRemoteConverter(kreuzberg_url="http://fake-kreuzberg:8000")
    with caplog.at_level(logging.WARNING, logger="app.pipelines.kreuzberg_converter"):
        out = c.run(sources=[tmp_source])["documents"]

    assert out[0].content == ""
    assert any("no usable extraction result" in r.message for r in caplog.records)
