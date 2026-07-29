"""Native docx text extraction (app/pipelines/docx_text.py).

Fixtures craft a minimal ``word/document.xml`` inside a real zip so the
substring/regex extraction is exercised end-to-end without a sample binary.
"""

import zipfile

from app.pipelines.docx_text import extract_docx_lines, extract_docx_text

_NS = 'xmlns:w="w" xmlns:mc="mc" xmlns:wps="wps" xmlns:v="v"'


def _run(text: str) -> str:
    return f"<w:r><w:t>{text}</w:t></w:r>"


def _para(*runs: str) -> str:
    return "<w:p>" + "".join(runs) + "</w:p>"


def _textbox(*paras: str) -> str:
    # An outer body <w:p> hosting a drawing whose <w:txbxContent> nests its own
    # <w:p> — the nesting that defeats naive </w:p> matching.
    return (
        "<w:p><w:r><w:drawing><wps:txbx><w:txbxContent>"
        + "".join(paras)
        + "</w:txbxContent></wps:txbx></w:drawing></w:r></w:p>"
    )


def _docx(tmp_path, body: str, name: str = "d.docx") -> str:
    document_xml = f'<?xml version="1.0"?><w:document {_NS}><w:body>{body}</w:body></w:document>'
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", document_xml)
    return str(path)


def test_joins_runs_within_a_paragraph_verbatim(tmp_path):
    # Word fragments a word across runs; per-paragraph join reconstructs it.
    src = _docx(tmp_path, _para(_run("udarbejde"), _run("s")))
    assert "udarbejdes" in extract_docx_lines(src)


def test_unescapes_entities(tmp_path):
    src = _docx(tmp_path, _para(_run("A"), _run("&amp;"), _run("B")))
    assert "A&B" in extract_docx_lines(src)


def test_line_break_splits_a_paragraph(tmp_path):
    src = _docx(tmp_path, _para(_run("Linje1"), "<w:br/>", _run("Linje2")))
    lines = extract_docx_lines(src)
    assert "Linje1" in lines
    assert "Linje2" in lines
    assert "Linje1Linje2" not in lines


def test_captures_both_body_and_textbox_text_in_order(tmp_path):
    body = _para(_run("Overskrift")) + _textbox(_para(_run("Boxlabel")))
    lines = extract_docx_lines(_docx(tmp_path, body))
    assert lines == ["Overskrift", "Boxlabel"]


def test_drops_empty_paragraphs(tmp_path):
    body = _para(_run("X")) + _para() + _para(_run("Y"))
    assert extract_docx_lines(_docx(tmp_path, body)) == ["X", "Y"]


def test_collapses_consecutive_duplicates_but_keeps_distant_repeats(tmp_path):
    body = (
        _para(_run("Gentag"))
        + _para(_run("Gentag"))  # consecutive → collapsed
        + _para(_run("Mellem"))
        + _para(_run("Gentag"))  # recurs in a different context → kept
    )
    lines = extract_docx_lines(_docx(tmp_path, body))
    assert lines == ["Gentag", "Mellem", "Gentag"]


def test_mc_fallback_duplicate_is_dropped(tmp_path):
    # Word mirrors each DrawingML shape as a legacy VML <mc:Fallback>; without
    # stripping it the text would be counted twice.
    body = (
        "<mc:AlternateContent><mc:Choice Requires='wps'>"
        + _textbox(_para(_run("Dobbelt")))
        + "</mc:Choice><mc:Fallback>"
        + "<v:textbox><w:txbxContent>"
        + _para(_run("Dobbelt"))
        + "</w:txbxContent></v:textbox>"
        + "</mc:Fallback></mc:AlternateContent>"
    )
    lines = extract_docx_lines(_docx(tmp_path, body))
    assert lines.count("Dobbelt") == 1


def test_lone_bullet_marker_rejoins_uppercase_item(tmp_path):
    body = (
        _para(_run("Sagsvurdering"))
        + _para(_run("-"))
        + _para(_run("Statusattest"))
        + _para(_run("-"))
        + _para(_run("Journaloplysninger"))
    )
    assert extract_docx_lines(_docx(tmp_path, body)) == [
        "Sagsvurdering",
        "- Statusattest",
        "- Journaloplysninger",
    ]


def test_hyphen_split_across_shapes_is_not_turned_into_bullets(tmp_path):
    # "3-hjulet el-køretøj" stored as fragments; the '-' here are word-internal
    # hyphens (next line lowercase) and must stay as-is, not become fake bullets.
    body = (
        _para(_run("f.eks. 3"))
        + _para(_run("-"))
        + _para(_run("hjulet el"))
        + _para(_run("-"))
        + _para(_run("køretøj"))
    )
    assert extract_docx_lines(_docx(tmp_path, body)) == [
        "f.eks. 3",
        "-",
        "hjulet el",
        "-",
        "køretøj",
    ]


def test_non_dash_bullet_glyph_is_normalised(tmp_path):
    body = _para(_run("•")) + _para(_run("Statusattest"))
    assert extract_docx_lines(_docx(tmp_path, body)) == ["- Statusattest"]


def test_trailing_lone_marker_is_dropped(tmp_path):
    body = _para(_run("Beslutning")) + _para(_run("-"))
    assert extract_docx_lines(_docx(tmp_path, body)) == ["Beslutning"]


def test_extract_docx_text_is_newline_joined(tmp_path):
    body = _para(_run("A")) + _para(_run("B")) + _para(_run("C"))
    assert extract_docx_text(_docx(tmp_path, body)) == "A\nB\nC"


def test_non_docx_file_returns_empty_list(tmp_path):
    p = tmp_path / "not.docx"
    p.write_text("just text, not a zip")
    assert extract_docx_lines(str(p)) == []


def test_zip_without_document_xml_returns_empty_list(tmp_path):
    p = tmp_path / "empty.docx"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("docProps/app.xml", "<x/>")
    assert extract_docx_lines(str(p)) == []
