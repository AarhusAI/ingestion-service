"""Rendering helper (app/pipelines/rendering.py).

Gotenberg HTTP and pypdfium2 are patched so no sidecar / PDF engine is needed.
"""

import io
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.pipelines import rendering
from app.pipelines.errors import ExtractionError

# -------------------- office_to_pdf --------------------


def test_office_to_pdf_returns_bytes(tmp_path):
    f = tmp_path / "a.docx"
    f.write_bytes(b"docx-bytes")
    resp = MagicMock()
    resp.content = b"%PDF-1.4 fake"
    resp.raise_for_status.return_value = None
    with patch("app.pipelines.rendering.httpx.post", return_value=resp) as post:
        out = rendering.office_to_pdf(str(f), gotenberg_url="http://gotenberg:3000")
    assert out == b"%PDF-1.4 fake"
    # Posts to the LibreOffice route as a multipart 'files' part.
    args, kwargs = post.call_args
    assert args[0].endswith("/forms/libreoffice/convert")
    assert "files" in kwargs


def test_office_to_pdf_http_error_raises_extraction_error(tmp_path):
    f = tmp_path / "a.docx"
    f.write_bytes(b"docx-bytes")
    with (
        patch("app.pipelines.rendering.httpx.post", side_effect=httpx.HTTPError("down")),
        pytest.raises(ExtractionError, match="gotenberg"),
    ):
        rendering.office_to_pdf(str(f), gotenberg_url="http://gotenberg:3000")


# -------------------- pdf_to_pngs --------------------


class _FakePage:
    def __init__(self):
        self.closed = False

    def render(self, scale):
        def _save(buf, format):
            buf.write(b"PNGDATA")

        pil = MagicMock()
        pil.save.side_effect = _save
        bitmap = MagicMock()
        bitmap.to_pil.return_value = pil
        return bitmap

    def close(self):
        self.closed = True


class _FakePdf:
    def __init__(self, n):
        self._pages = [_FakePage() for _ in range(n)]
        self.closed = False

    def __len__(self):
        return len(self._pages)

    def __getitem__(self, i):
        return self._pages[i]

    def close(self):
        self.closed = True


def test_pdf_to_pngs_renders_each_page():
    with patch("pypdfium2.PdfDocument", return_value=_FakePdf(3)):
        out = rendering.pdf_to_pngs(b"%PDF", dpi=150, max_pages=20)
    assert out == [b"PNGDATA", b"PNGDATA", b"PNGDATA"]


def test_pdf_to_pngs_respects_max_pages():
    with patch("pypdfium2.PdfDocument", return_value=_FakePdf(5)):
        out = rendering.pdf_to_pngs(b"%PDF", dpi=150, max_pages=2)
    assert out == [b"PNGDATA", b"PNGDATA"]


# -------------------- render_to_pngs orchestration --------------------


def test_render_pdf_skips_gotenberg(tmp_path):
    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF-bytes")
    with (
        patch("app.pipelines.rendering.office_to_pdf") as office,
        patch("app.pipelines.rendering.pdf_to_pngs", return_value=[b"x"]) as raster,
    ):
        out = rendering.render_to_pngs(str(f), gotenberg_url="http://g:3000")
    assert out == [b"x"]
    office.assert_not_called()
    # PDF bytes read straight off disk.
    assert raster.call_args[0][0] == b"%PDF-bytes"


def test_render_office_calls_gotenberg(tmp_path):
    f = tmp_path / "a.docx"
    f.write_bytes(b"docx")
    with (
        patch("app.pipelines.rendering.office_to_pdf", return_value=b"%PDF") as office,
        patch("app.pipelines.rendering.pdf_to_pngs", return_value=[b"x"]),
    ):
        out = rendering.render_to_pngs(str(f), gotenberg_url="http://g:3000")
    assert out == [b"x"]
    office.assert_called_once()


def test_render_unsupported_extension_raises(tmp_path):
    f = tmp_path / "a.xyz"
    f.write_bytes(b"data")
    with pytest.raises(ExtractionError):
        rendering.render_to_pngs(str(f), gotenberg_url="http://g:3000")


def test_render_zero_pages_raises(tmp_path):
    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF")
    with (
        patch("app.pipelines.rendering.pdf_to_pngs", return_value=[]),
        pytest.raises(ExtractionError, match="no pages"),
    ):
        rendering.render_to_pngs(str(f), gotenberg_url="http://g:3000")


def test_io_buffer_roundtrip_sanity():
    # Guards the BytesIO contract pdf_to_pngs relies on.
    buf = io.BytesIO()
    buf.write(b"PNGDATA")
    assert buf.getvalue() == b"PNGDATA"
