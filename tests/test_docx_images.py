"""Native docx figure extraction (app/pipelines/docx_images.py).

Fixtures craft a minimal ``word/document.xml`` + ``document.xml.rels`` + media
inside a real zip so the regex pairing (drawing extent <-> blip <-> rel target) is
exercised end-to-end without a sample binary.
"""

import zipfile
from types import SimpleNamespace

from app.pipelines.docx_images import _MAX_FIGURES, extract_docx_figure_images

_NS = 'xmlns:w="w" xmlns:wp="wp" xmlns:a="a" xmlns:r="r" xmlns:mc="mc"'
_FLOOR = 1_000_000


def _settings(min_emu: int = _FLOOR):
    return SimpleNamespace(extraction_router_min_image_emu=min_emu)


def _drawing(rid: str, cx: int, cy: int) -> str:
    """An inline picture: a drawing extent paired with a blip relationship id."""
    return (
        f'<w:drawing><wp:inline><wp:extent cx="{cx}" cy="{cy}"/>'
        f'<a:blip r:embed="{rid}"/></wp:inline></w:drawing>'
    )


def _rels(*pairs: tuple) -> str:
    """``pairs`` of (rid, target[, mode])."""
    body = ""
    for pair in pairs:
        rid, target = pair[0], pair[1]
        mode = pair[2] if len(pair) > 2 else None
        mode_attr = f' TargetMode="{mode}"' if mode else ""
        body += f'<Relationship Id="{rid}" Type="http://x/image" Target="{target}"{mode_attr}/>'
    return f"<Relationships>{body}</Relationships>"


def _docx(tmp_path, body: str, rels_pairs, media: dict, name: str = "d.docx") -> str:
    document_xml = f'<?xml version="1.0"?><w:document {_NS}><w:body>{body}</w:body></w:document>'
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/_rels/document.xml.rels", _rels(*rels_pairs))
        for member, data in media.items():
            zf.writestr(member, data)
    return str(path)


def test_extracts_referenced_body_figure_bytes(tmp_path):
    src = _docx(
        tmp_path,
        _drawing("rId1", 2000, 2000),  # area 4e6 >= floor
        [("rId1", "media/image1.png")],
        {"word/media/image1.png": b"FIGURE-BYTES"},
    )
    assert extract_docx_figure_images(src, _settings()) == [b"FIGURE-BYTES"]


def test_excludes_subfloor_image(tmp_path):
    # area 1e6-1 < floor -> not a content figure (icon-sized).
    src = _docx(
        tmp_path,
        _drawing("rId1", 999, 1000),
        [("rId1", "media/image1.png")],
        {"word/media/image1.png": b"TINY"},
    )
    assert extract_docx_figure_images(src, _settings()) == []


def test_unreferenced_media_excluded(tmp_path):
    # A logo sitting in word/media but referenced only from a footer's rels (not
    # document.xml.rels) is never resolved — body-only by construction.
    src = _docx(
        tmp_path,
        _drawing("rId1", 2000, 2000),
        [("rId1", "media/image1.png")],
        {"word/media/image1.png": b"FIGURE", "word/media/logo.png": b"LOGO"},
    )
    assert extract_docx_figure_images(src, _settings()) == [b"FIGURE"]


def test_sorted_largest_first(tmp_path):
    body = (
        _drawing("rId1", 2000, 1000) + _drawing("rId2", 3000, 3000) + _drawing("rId3", 2000, 2000)
    )
    src = _docx(
        tmp_path,
        body,
        [
            ("rId1", "media/a.png"),
            ("rId2", "media/b.png"),
            ("rId3", "media/c.png"),
        ],
        {"word/media/a.png": b"A", "word/media/b.png": b"B", "word/media/c.png": b"C"},
    )
    # areas: A=2e6, B=9e6, C=4e6 -> B, C, A
    assert extract_docx_figure_images(src, _settings()) == [b"B", b"C", b"A"]


def test_caps_at_max_figures(tmp_path):
    n = _MAX_FIGURES + 4
    drawings = "".join(_drawing(f"rId{i}", 2000, 1000 + i) for i in range(n))
    pairs = [(f"rId{i}", f"media/img{i}.png") for i in range(n)]
    media = {f"word/media/img{i}.png": f"IMG{i}".encode() for i in range(n)}
    out = extract_docx_figure_images(_docx(tmp_path, drawings, pairs, media), _settings())
    assert len(out) == _MAX_FIGURES
    # largest-first: highest cy (i = n-1) comes first.
    assert out[0] == f"IMG{n - 1}".encode()


def test_dedups_same_rid_reused_in_two_drawings(tmp_path):
    body = _drawing("rId1", 2000, 2000) + _drawing("rId1", 2000, 2000)
    src = _docx(
        tmp_path,
        body,
        [("rId1", "media/image1.png")],
        {"word/media/image1.png": b"ONCE"},
    )
    assert extract_docx_figure_images(src, _settings()) == [b"ONCE"]


def test_external_link_skipped(tmp_path):
    src = _docx(
        tmp_path,
        _drawing("rId1", 2000, 2000),
        [("rId1", "http://host/pic.png", "External")],
        {},
    )
    assert extract_docx_figure_images(src, _settings()) == []


def test_non_raster_target_skipped(tmp_path):
    # A vector part (emf/wmf) is the topology case, not a figure-profile raster.
    src = _docx(
        tmp_path,
        _drawing("rId1", 2000, 2000),
        [("rId1", "media/chart.emf")],
        {"word/media/chart.emf": b"VECTOR"},
    )
    assert extract_docx_figure_images(src, _settings()) == []


def test_referenced_media_missing_from_package_skipped(tmp_path):
    src = _docx(
        tmp_path,
        _drawing("rId1", 2000, 2000),
        [("rId1", "media/ghost.png")],
        {},  # the part the rel points at isn't in the zip
    )
    assert extract_docx_figure_images(src, _settings()) == []


def test_drawing_without_blip_ignored(tmp_path):
    # A textbox/shape drawing carries an extent but no embedded image.
    body = '<w:drawing><wp:inline><wp:extent cx="2000" cy="2000"/></wp:inline></w:drawing>'
    src = _docx(tmp_path, body, [("rId1", "media/image1.png")], {"word/media/image1.png": b"X"})
    assert extract_docx_figure_images(src, _settings()) == []


def test_blip_without_extent_ignored(tmp_path):
    body = '<w:drawing><a:blip r:embed="rId1"/></w:drawing>'
    src = _docx(tmp_path, body, [("rId1", "media/image1.png")], {"word/media/image1.png": b"X"})
    assert extract_docx_figure_images(src, _settings()) == []


def test_not_a_zip_returns_empty(tmp_path):
    path = tmp_path / "fake.docx"
    path.write_bytes(b"this is not a zip")
    assert extract_docx_figure_images(str(path), _settings()) == []


def test_missing_document_xml_returns_empty(tmp_path):
    path = tmp_path / "empty.docx"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("other.xml", "<x/>")
    assert extract_docx_figure_images(str(path), _settings()) == []


def test_missing_rels_returns_empty(tmp_path):
    path = tmp_path / "norels.docx"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "word/document.xml",
            f"<w:document {_NS}><w:body>{_drawing('rId1', 2000, 2000)}</w:body></w:document>",
        )
    assert extract_docx_figure_images(str(path), _settings()) == []
