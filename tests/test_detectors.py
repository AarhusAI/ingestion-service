"""Content-based engine detection (app/pipelines/detectors.py).

Builds minimal in-memory ``.docx`` zips and asserts the routing decision. No
real Word, no network.
"""

import logging
import zipfile

from app.config import Settings
from app.pipelines.detectors import classify_engine, detect_engine, docx_diagram_profile


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


def _image_drawing(cx, cy):
    """A DrawingML floating picture displayed at cx by cy EMU, like Word emits."""
    return (
        f'<w:drawing><wp:anchor><wp:extent cx="{cx}" cy="{cy}"/>'
        '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        '<a:graphic><a:graphicData><pic:pic><pic:blipFill>'
        '<a:blip r:embed="rId1"/>'
        '</pic:blipFill></pic:pic></a:graphicData></a:graphic>'
        "</wp:anchor></w:drawing>"
    )


# Display extents (EMU) taken from the real motivating file: the body process
# diagram (~4.26 in²) and a footer logo banner (~0.087 in²).
_REAL_DIAGRAM = (1999615, 1780442)
_LOGO_BANNER = (1016673, 72000)


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


# ----- Raster-image signal (second trigger for the diagram route) -----


def test_large_body_raster_image_routes_to_diagram_engine(tmp_path):
    # One big figure, no textboxes, lots of prose → the raster signal fires.
    xml = f"<w:document><w:body>{_image_drawing(*_REAL_DIAGRAM)}</w:body></w:document>"
    src = _write_docx(tmp_path, xml, app_words=1536)
    assert detect_engine(src, _settings()) == "vision-llm"


def test_small_logo_image_passes_through(tmp_path):
    # A logo-sized image (~0.087 in²) is below the area floor → default.
    xml = f"<w:document><w:body>{_image_drawing(*_LOGO_BANNER)}</w:body></w:document>"
    src = _write_docx(tmp_path, xml, app_words=1536)
    assert detect_engine(src, _settings()) is None


def test_tiny_icon_below_area_floor_passes_through(tmp_path):
    xml = f"<w:document><w:body>{_image_drawing(150000, 150000)}</w:body></w:document>"
    src = _write_docx(tmp_path, xml, app_words=1536)
    assert detect_engine(src, _settings()) is None


def test_many_small_icons_do_not_sum(tmp_path):
    # Ten small icons: count ≥ 1 but each area tiny — we use max, not sum.
    body = _image_drawing(200000, 200000) * 10
    xml = f"<w:document><w:body>{body}</w:body></w:document>"
    src = _write_docx(tmp_path, xml, app_words=1536)
    assert detect_engine(src, _settings()) is None


def test_no_body_images_passes_through(tmp_path):
    body = "<w:p><w:r><w:t>just ordinary prose</w:t></w:r></w:p>"
    xml = f"<w:document><w:body>{body}</w:body></w:document>"
    src = _write_docx(tmp_path, xml, app_words=1536)
    assert detect_engine(src, _settings()) is None


def test_effect_extent_not_mistaken_for_real_extent(tmp_path):
    # The huge dimensions sit on <wp:effectExtent> (l/t/r/b); the real
    # <wp:extent cx/cy> is tiny, so the area gate must not be fooled.
    drawing = (
        '<w:drawing><wp:anchor><wp:extent cx="150000" cy="150000"/>'
        '<wp:effectExtent l="0" t="0" r="99999999" b="99999999"/>'
        '<a:graphic><a:graphicData><pic:pic><pic:blipFill>'
        '<a:blip r:embed="rId1"/>'
        "</pic:blipFill></pic:pic></a:graphicData></a:graphic></wp:anchor></w:drawing>"
    )
    xml = f"<w:document><w:body>{drawing}</w:body></w:document>"
    src = _write_docx(tmp_path, xml, app_words=10)
    assert detect_engine(src, _settings()) is None


def test_textbox_signal_takes_priority_over_raster(tmp_path):
    # Both signals present: routing still happens (same engine); the textbox
    # path is checked first. Profile choice is covered by docx_diagram_profile.
    body = _textboxes(50) + _image_drawing(*_REAL_DIAGRAM)
    xml = f"<w:document><w:body>{body}</w:body></w:document>"
    src = _write_docx(tmp_path, xml, app_words=5)
    assert detect_engine(src, _settings()) == "vision-llm"


def test_optional_word_ratio_gate_rejects_prose_heavy_image(tmp_path):
    # area 3.56e12 / (5000+1) ≈ 7.1e8: a gate above that rejects; default 0 keeps.
    xml = f"<w:document><w:body>{_image_drawing(*_REAL_DIAGRAM)}</w:body></w:document>"
    src = _write_docx(tmp_path, xml, app_words=5000)
    assert detect_engine(src, _settings()) == "vision-llm"  # gate disabled by default
    strict = _settings(extraction_router_min_image_word_ratio=1e9)
    assert detect_engine(src, strict) is None


def test_raster_debug_logs_routing_signal(tmp_path, caplog):
    xml = f"<w:document><w:body>{_image_drawing(*_REAL_DIAGRAM)}</w:body></w:document>"
    src = _write_docx(tmp_path, xml, app_words=1536, name="Wheel.docx")
    with caplog.at_level(logging.DEBUG, logger="app.pipelines.detectors"):
        detect_engine(src, _settings())
    assert "docx routing Wheel.docx: images=1" in caplog.text
    assert "-> vision-llm" in caplog.text


# ----- docx_diagram_profile (vector vs raster vision-pass profile) -----


def test_docx_diagram_profile_vector_is_topology(tmp_path):
    xml = f"<w:document><w:body>{_textboxes(30)}</w:body></w:document>"
    src = _write_docx(tmp_path, xml, app_words=5)
    assert docx_diagram_profile(src, _settings()) == "diagram-topology"


def test_docx_diagram_profile_raster_is_figure(tmp_path):
    xml = f"<w:document><w:body>{_image_drawing(*_REAL_DIAGRAM)}</w:body></w:document>"
    src = _write_docx(tmp_path, xml, app_words=1536)
    assert docx_diagram_profile(src, _settings()) == "figure"


def test_docx_diagram_profile_missing_file_defaults_to_topology():
    assert docx_diagram_profile("/nonexistent/x.docx", _settings()) == "diagram-topology"


# ----- classify_engine: full decision (engine + signal + metrics) -----


def test_classify_engine_textbox_signal_with_metrics(tmp_path):
    xml = f"<w:document><w:body>{_textboxes(50)}</w:body></w:document>"
    src = _write_docx(tmp_path, xml, app_words=5)
    decision = classify_engine(src, _settings())
    assert decision.engine == "vision-llm"
    assert decision.signal == "textbox"
    assert decision.metrics["textboxes"] == 50
    assert decision.metrics["body_words"] == 5
    assert decision.metrics["ratio"] > 2.0


def test_classify_engine_raster_signal_with_metrics(tmp_path):
    xml = f"<w:document><w:body>{_image_drawing(*_REAL_DIAGRAM)}</w:body></w:document>"
    src = _write_docx(tmp_path, xml, app_words=1536)
    decision = classify_engine(src, _settings())
    assert decision.engine == "vision-llm"
    assert decision.signal == "raster"
    assert decision.metrics["image_count"] == 1
    assert decision.metrics["max_image_emu"] > 0


def test_classify_engine_default_carries_metrics(tmp_path):
    # Enough textboxes to be measured, but the ratio is below the gate → default
    # decision, but the measured numbers are still surfaced ("why default").
    body = "<w:p><w:r><w:t>" + ("word " * 100) + "</w:t></w:r></w:p>" + _textboxes(20)
    xml = f"<w:document><w:body>{body}</w:body></w:document>"
    src = _write_docx(tmp_path, xml, app_words=100)
    decision = classify_engine(src, _settings())
    assert decision.engine is None
    assert decision.signal == "default"
    assert decision.metrics["textboxes"] == 20


def test_classify_engine_non_docx_is_default():
    decision = classify_engine("notes.txt", _settings())
    assert decision.engine is None
    assert decision.signal == "default"
    assert decision.metrics == {}
