"""
file_reader.py
--------------
Extract plain text from uploaded PDF, DOCX, and TXT files.
All processing is local — no external API calls.
"""

from __future__ import annotations
import io


def read_uploaded_file(uploaded_file) -> str:
    """
    Extract text from a Streamlit UploadedFile object.

    Supported formats
    -----------------
    .txt   : Read directly as UTF-8 text.
    .pdf   : Extract text with PyMuPDF (fitz) — fallback to pdfplumber.
    .docx  : Extract paragraphs with python-docx.

    Returns
    -------
    str : Extracted text, stripped of excess whitespace.

    Raises
    ------
    ValueError : If the file type is unsupported or both PDF extractors fail.
    """
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()

    # ── TXT ──────────────────────────────────────────────────────────────────
    if name.endswith(".txt"):
        try:
            return raw_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            return raw_bytes.decode("latin-1").strip()

    # ── PDF ──────────────────────────────────────────────────────────────────
    if name.endswith(".pdf"):
        # Try PyMuPDF first (faster, better layout)
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            pages = [page.get_text() for page in doc]
            doc.close()
            text = "\n".join(pages).strip()
            if text:
                return text
        except ImportError:
            pass
        except Exception:
            pass

        # Fallback: pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            text = "\n".join(pages).strip()
            if text:
                return text
        except ImportError:
            pass
        except Exception:
            pass

        raise ValueError(
            "Could not extract text from PDF. "
            "Install PyMuPDF: pip install PyMuPDF"
        )

    # ── DOCX ─────────────────────────────────────────────────────────────────
    if name.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(raw_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs).strip()
        except ImportError:
            raise ValueError(
                "python-docx not installed. Run: pip install python-docx"
            )
        except Exception as e:
            raise ValueError(f"Could not read DOCX: {e}")

    raise ValueError(f"Unsupported file type: {uploaded_file.name}")
