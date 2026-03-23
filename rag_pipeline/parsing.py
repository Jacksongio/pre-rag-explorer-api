from __future__ import annotations

import csv
import io
from pathlib import Path

from .models import FileType


def parse_file(
    file_name: str,
    content: bytes | str,
) -> str:
    ext = Path(file_name).suffix.lower().lstrip(".")
    file_type = _extension_to_type(ext)

    match file_type:
        case FileType.PDF:
            return _parse_pdf(content if isinstance(content, bytes) else content.encode())
        case FileType.CSV:
            text = content if isinstance(content, str) else content.decode("utf-8")
            return _parse_csv(text)
        case FileType.MARKDOWN:
            return content if isinstance(content, str) else content.decode("utf-8")
        case FileType.TEXT:
            return content if isinstance(content, str) else content.decode("utf-8")


def _extension_to_type(ext: str) -> FileType:
    mapping = {
        "pdf": FileType.PDF,
        "csv": FileType.CSV,
        "md": FileType.MARKDOWN,
        "markdown": FileType.MARKDOWN,
        "txt": FileType.TEXT,
    }
    return mapping.get(ext, FileType.TEXT)


def _parse_pdf(data: bytes) -> str:
    try:
        import pymupdf
    except ImportError:
        raise ImportError(
            "pymupdf is required for PDF parsing. Install it with: pip install pymupdf"
        )

    doc = pymupdf.open(stream=data, filetype="pdf")
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def _parse_csv(text: str) -> str:
    reader = csv.DictReader(io.StringIO(text))
    rows: list[str] = []
    for row in reader:
        rows.append(" ".join(str(v) for v in row.values()))
    return "\n".join(rows)
