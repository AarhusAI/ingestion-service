"""HybridDiagramConverter (app/pipelines/hybrid_diagram_converter.py).

The inner vision converter and the native docx extractor are patched at the
module symbol, so no rendering / VLM endpoint / real docx is touched.
"""

import logging
from unittest.mock import MagicMock, patch

from haystack import Document

from app.pipelines import hybrid_diagram_converter as hdc

_MERMAID = '```mermaid\nflowchart TD\n  n1["A"] --> n2["B"]\n```'


def _vision(content=_MERMAID, meta=None):
    """A mock vision converter returning one Document."""
    v = MagicMock(name="vision")
    doc_meta = meta if meta is not None else {"page_count": 3}
    v.run.return_value = {"documents": [Document(content=content, meta=doc_meta)]}
    return v


def _build(vision):
    """HybridDiagramConverter with ``build_converter`` patched to return ``vision``."""
    with patch.object(hdc, "build_converter", return_value=vision):
        return hdc.HybridDiagramConverter(settings=object())


def test_docx_merges_native_body_then_grounded_mermaid():
    vision = _vision()
    lines = ["Sagsvurdering", "- Statusattest", "Afgørelse"]
    with patch.object(hdc, "extract_docx_lines", return_value=lines):
        conv = _build(vision)
        out = conv.run(sources=["f.docx"], meta={"file_id": "f1"})["documents"]

    assert len(out) == 1
    doc = out[0]
    # native body is complete and verbatim …
    assert "Sagsvurdering" in doc.content
    assert "- Statusattest" in doc.content
    assert "Afgørelse" in doc.content
    # … and precedes the appended mermaid section
    assert "## Procesdiagram (Mermaid)" in doc.content
    assert "```mermaid" in doc.content
    assert doc.content.index("Sagsvurdering") < doc.content.index("## Procesdiagram")
    # meta describes the hybrid origin; request meta is preserved
    assert doc.meta["extractor"] == "hybrid-diagram"
    assert doc.meta["native_source"] == "docx-xml"
    assert doc.meta["vision_profile"] == "diagram-topology"
    assert doc.meta["page_count"] == 3
    assert doc.meta["file_id"] == "f1"


def test_docx_passes_topology_profile_and_native_grounding_to_vision():
    vision = _vision()
    lines = ["Sagsvurdering", "- Statusattest", "Afgørelse"]
    with patch.object(hdc, "extract_docx_lines", return_value=lines):
        _build(vision).run(sources=["f.docx"], meta=None)

    _, kwargs = vision.run.call_args
    assert kwargs["profile"] == "diagram-topology"
    # grounding is the native body, newline-joined and verbatim
    assert kwargs["grounding"] == "Sagsvurdering\n- Statusattest\nAfgørelse"
    # topology renders the page (labels inferred from the image); no blip override
    assert kwargs["images_override"] is None


def test_request_meta_wins_over_hybrid_meta():
    vision = _vision()
    with patch.object(hdc, "extract_docx_lines", return_value=["A", "B", "C"]):
        out = _build(vision).run(sources=["f.docx"], meta={"extractor": "override"})["documents"]
    assert out[0].meta["extractor"] == "override"


def test_mermaid_fence_extracted_from_noisy_vision_output():
    # The topology profile asks for the fence only, but a chatty model may wrap
    # it — we keep just the fence under our heading.
    noisy = 'Sure!\n```mermaid\nflowchart TD\n  n1["X"]\n```\nHope this helps'
    vision = _vision(content=noisy)
    with patch.object(hdc, "extract_docx_lines", return_value=["A", "B", "C"]):
        doc = _build(vision).run(sources=["f.docx"], meta=None)["documents"][0]
    assert "Sure!" not in doc.content
    assert "Hope this helps" not in doc.content
    assert doc.content.count("```") == 2


def test_no_mermaid_fence_ships_native_body_only():
    vision = _vision(content="I could not read the diagram", meta={"page_count": 1})
    with patch.object(hdc, "extract_docx_lines", return_value=["L1", "L2", "L3"]):
        doc = _build(vision).run(sources=["f.docx"], meta=None)["documents"][0]
    assert doc.content == "L1\nL2\nL3"
    assert "Procesdiagram" not in doc.content


def test_raster_docx_uses_figure_profile_and_keeps_full_output():
    # docx_diagram_profile says "figure" (raster diagram): the vision pass runs
    # the figure profile and its WHOLE output is kept (its own heading + Mermaid),
    # not just the fence — appended after the verbatim native prose.
    figure_out = (
        "## Procesoverblik\n\n"
        '```mermaid\nflowchart TD\n  n1["Sagsåbning"] --> n2["Opfølgning"]\n```'
    )
    vision = _vision(content=figure_out)
    lines = ["Støtte til køb af bil", "Retningslinje", "Indledning"]
    with (
        patch.object(hdc, "extract_docx_lines", return_value=lines),
        patch.object(hdc, "docx_diagram_profile", return_value="figure"),
        patch.object(hdc, "extract_docx_figure_images", return_value=[b"FIG-PNG"]),
    ):
        doc = _build(vision).run(sources=["f.docx"], meta=None)["documents"][0]

    assert "Støtte til køb af bil" in doc.content  # native prose preserved
    assert "## Procesoverblik" in doc.content  # figure's own heading kept
    assert "```mermaid" in doc.content
    assert doc.meta["vision_profile"] == "figure"
    _, kwargs = vision.run.call_args
    assert kwargs["profile"] == "figure"
    # figure path reads labels from the embedded image, so NO prose grounding …
    assert kwargs["grounding"] is None
    # … and the native-resolution blip bytes are sent instead of a page render.
    assert kwargs["images_override"] == [b"FIG-PNG"]


def test_figure_extraction_empty_falls_back_to_full_page_render():
    # No extractable blip (odd package) -> images_override omitted so the vision
    # converter renders the page as before. Still no grounding on the figure path.
    vision = _vision(content='## Fig\n```mermaid\nflowchart TD\n  n1["X"]\n```')
    with (
        patch.object(hdc, "extract_docx_lines", return_value=["A", "B", "C"]),
        patch.object(hdc, "docx_diagram_profile", return_value="figure"),
        patch.object(hdc, "extract_docx_figure_images", return_value=[]),
    ):
        _build(vision).run(sources=["f.docx"], meta=None)
    _, kwargs = vision.run.call_args
    assert kwargs["images_override"] is None
    assert kwargs["grounding"] is None


def test_raster_docx_with_empty_vision_ships_native_body_only():
    # The decorative-image case: figure pass returns nothing -> native body alone,
    # no diagram section, no exception.
    vision = _vision(content="   ", meta={})
    with (
        patch.object(hdc, "extract_docx_lines", return_value=["A", "B", "C"]),
        patch.object(hdc, "docx_diagram_profile", return_value="figure"),
        patch.object(hdc, "extract_docx_figure_images", return_value=[b"PHOTO"]),
    ):
        doc = _build(vision).run(sources=["f.docx"], meta=None)["documents"][0]
    assert doc.content == "A\nB\nC"


def test_non_docx_delegates_to_pure_vision():
    vision = _vision(content="full vision markdown", meta={"extractor": "vision-llm"})
    with patch.object(hdc, "extract_docx_lines") as ex:
        out = _build(vision).run(sources=["scan.pdf"], meta=None, profile="diagram")["documents"]

    ex.assert_not_called()  # native extraction is docx-only
    _, kwargs = vision.run.call_args
    assert kwargs["profile"] == "diagram"
    assert "grounding" not in kwargs
    assert out[0].content == "full vision markdown"


def test_docx_with_too_little_native_text_falls_back_to_full_vision():
    vision = _vision(content="full vision markdown", meta={})
    # One line is below the threshold → don't trust native text; let vision
    # transcribe the whole page (profile=diagram, no grounding).
    with patch.object(hdc, "extract_docx_lines", return_value=["only one line"]):
        out = _build(vision).run(sources=["f.docx"], meta=None)["documents"]

    _, kwargs = vision.run.call_args
    assert kwargs["profile"] == "diagram"
    assert "grounding" not in kwargs
    assert out[0].content == "full vision markdown"


def test_debug_logs_grounded_branch(caplog):
    vision = _vision()
    with (
        patch.object(hdc, "extract_docx_lines", return_value=["A", "B", "C"]),
        caplog.at_level(logging.DEBUG, logger="app.pipelines.hybrid_diagram_converter"),
    ):
        _build(vision).run(sources=["Diagram.docx"], meta=None)
    assert "hybrid Diagram.docx: native_lines=3 -> grounded diagram-topology" in caplog.text


def test_debug_logs_fallback_branch_for_non_docx(caplog):
    vision = _vision(content="x", meta={})
    with caplog.at_level(logging.DEBUG, logger="app.pipelines.hybrid_diagram_converter"):
        _build(vision).run(sources=["scan.pdf"], meta=None)
    assert "hybrid scan.pdf: non-docx -> delegate full-vision" in caplog.text


def test_warm_up_fans_out_to_inner_vision():
    vision = _vision()
    _build(vision).warm_up()
    assert vision.warm_up.called


def test_warm_up_skips_when_inner_has_no_warm_up():
    # A converter without a warm_up attribute must not break warm-up.
    _build(object()).warm_up()  # must not raise
