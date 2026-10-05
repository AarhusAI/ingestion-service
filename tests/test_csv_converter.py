"""CsvConverter (app/pipelines/csv_converter.py): rows → ``column: value`` blocks."""

import pytest

from app.pipelines.csv_converter import CsvConverter
from app.pipelines.errors import ExtractionError


def _convert(tmp_path, data: bytes, name="t.csv", meta=None):
    p = tmp_path / name
    p.write_bytes(data)
    return CsvConverter().run(sources=[str(p)], meta=meta)["documents"][0]


def test_rows_become_column_value_blocks(tmp_path):
    doc = _convert(
        tmp_path, b"name,age,city\nAlice,30,\n,,\nBob,25,Aarhus\n", meta={"file_id": "f"}
    )
    # Empty cells omitted, blank row skipped, rows separated by a blank line.
    assert doc.content == "name: Alice\nage: 30\n\nname: Bob\nage: 25\ncity: Aarhus"
    assert doc.meta == {"file_id": "f", "extraction_engine": "csv"}


@pytest.mark.parametrize("data", [b"a;b\n1;2\n", b"a\tb\n1\t2\n", "\ufeffa,b\n1,2\n".encode()])
def test_delimiter_sniffed_and_bom_stripped(tmp_path, data):
    assert _convert(tmp_path, data).content == "a: 1\nb: 2"


def test_cp1252_fallback(tmp_path):
    assert _convert(tmp_path, "navn\nÆble\n".encode("cp1252")).content == "navn: Æble"


def test_empty_header_cell_gets_positional_name(tmp_path):
    assert _convert(tmp_path, b"a,,c\n1,2,3\n").content == "a: 1\ncolumn 2: 2\nc: 3"


def test_quoted_multiline_cell_is_flattened(tmp_path):
    # The cell's own blank line must not fake a row boundary for the splitter.
    assert _convert(tmp_path, b'a,b\n"x\n\ny",2\n').content == "a: x y\nb: 2"


@pytest.mark.parametrize(
    ("data", "name", "expected"),
    [
        (
            b"Navn, fuldt\tBy, postnr\nAnne\t8000 Aarhus\n",
            "t.tsv",
            "Navn, fuldt: Anne\nBy, postnr: 8000 Aarhus",
        ),
        (
            b'"Efternavn, fornavn, mellemnavn";By\n"Hansen, Anne, M";Aarhus\n',
            "t.csv",
            "Efternavn, fornavn, mellemnavn: Hansen, Anne, M\nBy: Aarhus",
        ),
        (b"sep=;\nnavn;alder\nAnne;30\n", "t.csv", "navn: Anne\nalder: 30"),
        ("navn;beløb, kr\nAnne;1,50\n".encode(), "t.csv", "navn: Anne\nbeløb, kr: 1,50"),
        ("a\tb\n1\t2\n".encode("utf-16"), "t.txt.csv", "a: 1\nb: 2"),
        (b"a\n\x81x\n", "t.csv", "a: \ufffdx"),
        (b"a,b\n1,2,3,4\n", "t.csv", "a: 1\nb: 2\ncolumn 3: 3\ncolumn 4: 4"),
    ],
)
def test_review_regressions(tmp_path, data, name, expected):
    assert _convert(tmp_path, data, name=name).content == expected


def test_missing_file_is_extraction_error(tmp_path):
    with pytest.raises(ExtractionError, match=r"gone\.csv"):
        CsvConverter().run(sources=[str(tmp_path / "gone.csv")])
