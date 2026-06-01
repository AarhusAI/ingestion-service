"""VisionLLMConverter (app/pipelines/vision_llm_converter.py).

The renderer and the chat-completions HTTP call are patched at the module symbol
so no Gotenberg / pypdfium2 / VLM endpoint is touched.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.pipelines.errors import ExtractionError
from app.pipelines.vision_llm_converter import VisionLLMConverter

_RENDER = "app.pipelines.vision_llm_converter.render_to_pngs"
_POST = "app.pipelines.vision_llm_converter.httpx.post"


def _conv(**overrides):
    base = dict(api_base_url="http://vlm/v1", api_key="SUPERSECRETKEY", model="m")
    base.update(overrides)
    return VisionLLMConverter(**base)


def _ok_response(content="## Konsulent\n- step"):
    r = MagicMock()
    r.json.return_value = {"choices": [{"message": {"content": content}}]}
    r.raise_for_status.return_value = None
    return r


def test_happy_path_returns_one_markdown_document():
    with (
        patch(_RENDER, return_value=[b"png1", b"png2"]),
        patch(_POST, return_value=_ok_response("## Lane\n- x")) as post,
    ):
        out = _conv().run(sources=["f.docx"], meta={"file_id": "f1"})["documents"]

    assert len(out) == 1
    assert out[0].content == "## Lane\n- x"
    assert out[0].meta["file_id"] == "f1"
    assert out[0].meta["extractor"] == "vision-llm"
    assert out[0].meta["page_count"] == 2
    # Bearer auth set; both page images sent.
    _, kwargs = post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer SUPERSECRETKEY"
    image_parts = [p for p in kwargs["json"]["messages"][1]["content"] if p["type"] == "image_url"]
    assert len(image_parts) == 2


def test_request_meta_wins_over_converter_meta():
    with (
        patch(_RENDER, return_value=[b"png"]),
        patch(_POST, return_value=_ok_response()),
    ):
        out = _conv().run(sources=["f.docx"], meta={"extractor": "override"})["documents"]
    assert out[0].meta["extractor"] == "override"


def test_meta_none_yields_only_converter_meta():
    with (
        patch(_RENDER, return_value=[b"png"]),
        patch(_POST, return_value=_ok_response()),
    ):
        out = _conv().run(sources=["f.docx"], meta=None)["documents"]
    assert out[0].meta == {"extractor": "vision-llm", "page_count": 1}


def test_http_error_raises_extraction_error():
    with (
        patch(_RENDER, return_value=[b"png"]),
        patch(_POST, side_effect=httpx.HTTPError("endpoint down")),
        pytest.raises(ExtractionError),
    ):
        _conv().run(sources=["f.docx"], meta=None)


def test_renderer_error_propagates():
    with (
        patch(_RENDER, side_effect=ExtractionError("gotenberg convert failed for f.docx")),
        pytest.raises(ExtractionError),
    ):
        _conv().run(sources=["f.docx"], meta=None)


def test_malformed_response_no_choices_raises():
    r = MagicMock()
    r.json.return_value = {"unexpected": "shape"}
    r.raise_for_status.return_value = None
    with (
        patch(_RENDER, return_value=[b"png"]),
        patch(_POST, return_value=r),
        pytest.raises(ExtractionError, match="empty content"),
    ):
        _conv().run(sources=["f.docx"], meta=None)


def test_non_json_body_raises():
    r = MagicMock()
    r.json.side_effect = ValueError("no json")
    r.raise_for_status.return_value = None
    with (
        patch(_RENDER, return_value=[b"png"]),
        patch(_POST, return_value=r),
        pytest.raises(ExtractionError, match="non-JSON"),
    ):
        _conv().run(sources=["f.docx"], meta=None)


def test_api_key_never_appears_in_error_message():
    with (
        patch(_RENDER, return_value=[b"png"]),
        patch(_POST, side_effect=httpx.HTTPError("connect failed")),
        pytest.raises(ExtractionError) as excinfo,
    ):
        _conv(api_key="SUPERSECRETKEY").run(sources=["f.docx"], meta=None)
    assert "SUPERSECRETKEY" not in str(excinfo.value)
