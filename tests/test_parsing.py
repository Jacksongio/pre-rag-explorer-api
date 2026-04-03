from __future__ import annotations

import pytest

from rag_pipeline import parse_file


def test_parse_txt_from_bytes():
    out = parse_file("note.txt", b"Hello\nWorld")
    assert out == "Hello\nWorld"


def test_parse_txt_from_str():
    out = parse_file("note.txt", "same content")
    assert out == "same content"


def test_parse_markdown():
    out = parse_file("README.md", b"# Title\n\nBody")
    assert out == "# Title\n\nBody"


def test_parse_unknown_extension_as_text():
    out = parse_file("data.xyz", b"treated as plain text")
    assert out == "treated as plain text"


def test_parse_csv_joins_rows():
    csv_content = "name,role\nAlice,dev\nBob,ops\n"
    out = parse_file("users.csv", csv_content)
    assert "Alice" in out and "dev" in out
    assert "Bob" in out and "ops" in out
    assert "\n" in out


def test_parse_pdf_extracts_text():
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Embedded PDF phrase")
    pdf_bytes = doc.tobytes()
    doc.close()

    out = parse_file("doc.pdf", pdf_bytes)
    assert "Embedded PDF phrase" in out
