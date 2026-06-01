"""Content-based engine detection (app/pipelines/detectors.py).

Builds minimal in-memory ``.docx`` zips and asserts the routing decision. No
real Word, no network.
"""

import logging
import zipfile

from app.config import Settings
from app.pipelines.detectors import detect_engine


def _settings(**overrides) -> Settings:
    # Pin the diagram engine explicitly so these detection tests assert the
    # engine they configure, independent of the code/router default (which is
    # hybrid-diagram). Overridable per test.
    base = {"extraction_router_diagram_engine": "vision-llm"}
    base.update(overrides)
    return Settings(_env_file=None, api_key="a" * 32, **base)


def _write_docx(tmp_path, document_xml, *, app_words=None, name="f.docx"):
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("word/document.xml", document_xml)
        if app_words is not None:
            z.writestr(
                "docProps/app.xml",
                f"<Properties><Words>{app_words}</Words></Properties>",
            )
    return str(p)


def _textboxes(n, tag="w:txbxContent"):
    return f"<{tag}><w:t>label</w:t></{tag}>" * n


def test_diagram_heavy_docx_routes_to_vision(tmp_path):
    # 50 text boxes, 5 body words → 50 ≥ 20 and 50/6 ≈ 8.3 ≥ 2.0.
    xml = f"<w:document><w:body>{_textboxes(50)}</w:body></w:document>"
    src = _write_docx(tmp_path, xml, app_words=5)
    assert detect_engine(src, _settings()) == "vision-llm"


def test_diagram_engine_name_comes_from_settings(tmp_path):
    xml = f"<w:document><w:body>{_textboxes(50)}</w:body></w:document>"
    src = _write_docx(tmp_path, xml, app_words=5)
    s = _settings(extraction_router_diagram_engine="docling")
    assert detect_engine(src, s) == "docling"


def test_ordinary_prose_docx_passes_through(tmp_path):
    body = "<w:p><w:r><w:t>lots of normal prose</w:t></w:r></w:p>"
    xml = f"<w:document><w:body>{body}</w:body></w:document>"
    src = _write_docx(tmp_path, xml, app_words=4000)
    assert detect_engine(src, _settings()) is None


def test_few_textboxes_below_floor_passes_through(tmp_path):
    # 10 < EXTRACTION_ROUTER_MIN_TEXTBOXES (20) — never routes regardless of ratio.
    xml = f"<w:document><w:body>{_textboxes(10)}</w:body></w:document>"
    src = _write_docx(tmp_path, xml, app_words=1)
    assert detect_engine(src, _settings()) is None


def test_many_textboxes_but_large_body_passes_through(tmp_path):
    # 50 ≥ 20 but ratio 50/1001 ≈ 0.05 < 2.0 — a normal report with some diagrams.
    xml = f"<w:document><w:body>{_textboxes(50)}</w:body></w:document>"
    src = _write_docx(tmp_path, xml, app_words=1000)
    assert detect_engine(src, _settings()) is None


def test_vml_and_wps_textboxes_are_counted(tmp_path):
    # 15 legacy VML + 15 modern WPS = 30 ≥ 20, tiny body → routes.
    xml = (
        "<w:document><w:body>"
        + "<v:textbox><w:t>x</w:t></v:textbox>" * 15
        + "<wps:txbx><w:t>y</w:t></wps:txbx>" * 15
        + "</w:body></w:document>"
    )
    src = _write_docx(tmp_path, xml, app_words=2)
    assert detect_engine(src, _settings()) == "vision-llm"


def test_missing_app_xml_uses_body_word_fallback(tmp_path):
    # No docProps/app.xml → count <w:t> words after stripping text-box regions.
    # 30 boxes, 2 body words outside any box → 30/3 = 10 ≥ 2.0.
    xml = (
        "<w:document><w:body>"
        "<w:p><w:r><w:t>one two</w:t></w:r></w:p>"
        + _textboxes(30)
        + "</w:body></w:document>"
    )
    src = _write_docx(tmp_path, xml, app_words=None)
    assert detect_engine(src, _settings()) == "vision-llm"


def test_debug_logs_routing_signal(tmp_path, caplog):
    xml = f"<w:document><w:body>{_textboxes(50)}</w:body></w:document>"
    src = _write_docx(tmp_path, xml, app_words=5, name="Diagram.docx")
    with caplog.at_level(logging.DEBUG, logger="app.pipelines.detectors"):
        detect_engine(src, _settings())
    assert "docx routing Diagram.docx: textboxes=50 body_words=5 ratio=8.33" in caplog.text
    assert "-> vision-llm" in caplog.text


def test_malformed_zip_returns_none(tmp_path):
    p = tmp_path / "broken.docx"
    p.write_bytes(b"this is not a zip file")
    assert detect_engine(str(p), _settings()) is None


def test_non_docx_returns_none():
    assert detect_engine("/nonexistent/report.pdf", _settings()) is None
    assert detect_engine("notes.txt", _settings()) is None
