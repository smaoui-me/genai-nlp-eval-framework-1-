"""Turns an uploaded file (.txt or .pdf) into plain text we can annotate."""

from __future__ import annotations

import io
import re


def extract_text(uploaded_file) -> str:
    """Return the plain text of a file uploaded via st.file_uploader.
    `uploaded_file` has `.name` (original filename) and `.getvalue()` (raw bytes)."""
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith(".pdf"):
        return _extract_pdf_text(data)

    # errors="replace" swaps in a placeholder instead of crashing on odd bytes.
    return data.decode("utf-8", errors="replace")


def _extract_pdf_text(data: bytes) -> str:
    """Read all the text out of a PDF's raw bytes."""
    from pypdf import PdfReader  # imported here so a missing pypdf only breaks PDF uploads, not the whole app

    reader = PdfReader(io.BytesIO(data))  # BytesIO makes our bytes readable like a file, with no file on disk
    pages = [page.extract_text() or "" for page in reader.pages]  # "or \"\"" covers pages that fail to extract
    return _clean_pdf_text("\n".join(pages))


def _clean_pdf_text(text: str) -> str:
    """Tidy up common PDF text-extraction artifacts."""
    text = re.sub(r"-\n(?=\w)", "", text)  # join words split with a hyphen at a line break
    text = re.sub(r"[ \t]+", " ", text)  # collapse repeated spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse long runs of blank lines
    return text.strip()
