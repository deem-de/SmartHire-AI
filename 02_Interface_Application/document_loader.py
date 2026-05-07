"""Load raw text from resume documents.

This module is separate from training. It only extracts text from files so the
text can be converted into the same JSON-like format used by the BERT pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path


def extract_text_from_document(file_path: str | Path) -> str:
    """Extract raw text from PDF, DOCX, TXT, or JSON files."""

    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _normalize_extracted_text(_extract_pdf_text(path))
    if suffix == ".docx":
        return _normalize_extracted_text(_extract_docx_text(path))
    if suffix in {".txt", ".json"}:
        return _normalize_extracted_text(path.read_text(encoding="utf-8", errors="ignore").strip())

    raise ValueError("Supported file types: PDF, DOCX, TXT, JSON")


def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF file.

    Real resumes come from many sources, so we try more than one extractor and
    keep the best-looking text.
    """

    candidates = []

    pypdf_text = _extract_pdf_text_with_pypdf(path)
    if pypdf_text:
        candidates.append(("pypdf", pypdf_text))

    pdfplumber_text = _extract_pdf_text_with_pdfplumber(path)
    if pdfplumber_text:
        candidates.append(("pdfplumber", pdfplumber_text))

    pymupdf_text = _extract_pdf_text_with_pymupdf(path)
    if pymupdf_text:
        candidates.append(("pymupdf", pymupdf_text))

    if not candidates:
        return ""

    best_name, best_text = max(candidates, key=lambda item: _score_extracted_text(item[1]))
    _ = best_name  # kept for future logging/debugging
    return best_text


def _extract_pdf_text_with_pypdf(path: Path) -> str:
    """Extract text using pypdf."""

    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise ImportError("Install PDF support with: python -m pip install pypdf") from error

    reader = PdfReader(str(path))
    pages = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)

    return "\n".join(pages).strip()


def _extract_pdf_text_with_pdfplumber(path: Path) -> str:
    """Try pdfplumber when available for layout-heavy resumes."""

    try:
        import pdfplumber
    except ImportError:
        return ""

    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            if page_text.strip():
                pages.append(page_text)

    return "\n".join(pages).strip()


def _extract_pdf_text_with_pymupdf(path: Path) -> str:
    """Use PyMuPDF block extraction for layout-heavy or multi-column resumes."""

    try:
        import fitz
    except ImportError:
        return ""

    blocks: list[str] = []
    with fitz.open(str(path)) as document:
        for page in document:
            for block in page.get_text("blocks", sort=True):
                block_text = (block[4] or "").strip()
                if block_text:
                    blocks.append(block_text)

    return "\n".join(blocks).strip()


def _extract_docx_text(path: Path) -> str:
    """Extract text from a DOCX file."""

    try:
        import docx
    except ImportError as error:
        raise ImportError("Install DOCX support with: python -m pip install python-docx") from error

    document = docx.Document(str(path))
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs]
    return "\n".join(paragraph for paragraph in paragraphs if paragraph).strip()


def _score_extracted_text(text: str) -> float:
    """Score text quality so we can choose the best extractor output."""

    if not text.strip():
        return 0.0

    normalized = _normalize_extracted_text(text)
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]

    score = 0.0
    score += min(len(normalized) / 120.0, 25.0)
    score += min(len(lines), 20)

    headers = ["SKILLS", "EXPERIENCE", "WORK EXPERIENCE", "EDUCATION", "PROJECTS"]
    score += sum(8 for header in headers if header in normalized.upper())

    if re.search(r"[\w.+-]+@[\w.-]+\.\w+", normalized):
        score += 12
    if re.search(r"\+?\d[\d\s()-]{7,}\d", normalized):
        score += 8

    weird_spacing_penalty = len(re.findall(r"(?:\b[A-Za-z]\b(?:\s+\b[A-Za-z]\b){4,})", text))
    score -= weird_spacing_penalty * 6

    artifact_penalty = 0
    artifact_penalty += len(re.findall(r"\(cid:\s*\d+\)", text, flags=re.I)) * 6
    artifact_penalty += len(re.findall(r"/[A-Za-z_]+", text)) * 3
    artifact_penalty += len(re.findall(r"\.[A-Za-z]{2,6}", text)) * 2
    artifact_penalty += len(re.findall(r"[A-Za-z]+gotthegoddamnshipback", normalized, flags=re.I)) * 3
    score -= artifact_penalty

    readable_header_bonus = 0
    for header in ["PROGRAMMING", "LANGUAGES", "DEGREES", "CERTIFICATES", "PUBLICATIONS", "TALKS"]:
        if re.search(rf"(?im)^\s*{re.escape(header)}\s*$", normalized):
            readable_header_bonus += 4
    score += readable_header_bonus
    return score


def _normalize_extracted_text(text: str) -> str:
    """Repair common PDF extraction noise before the text reaches the model."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\(cid:\s*\d+\)", " ", text, flags=re.I)
    text = text.replace("\t", " ")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2030", " ")
    text = text.replace("\u00bd", " ")
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)
    text = re.sub(r"[ ]{2,}", "  ", text)

    cleaned_lines: list[str] = []
    previous_non_empty = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        line = _repair_spaced_line(line)
        line = _repair_compound_headers(line)
        line = _normalize_contact_spacing(line)
        line = re.sub(r"\.(?=[A-Z])", ". ", line)
        line = line.replace("Aboutme", "About me")
        line = line.replace("Areasofspecialization", "Areas of specialization")
        line = line.replace("howtogetitback", "how to get it back")
        line = re.sub(r"\s+([,.;:])", r"\1", line)
        line = re.sub(r"([:])([^\s])", r"\1 \2", line)
        line = re.sub(r"\s{2,}", " ", line).strip()

        if line and line == previous_non_empty:
            continue

        cleaned_lines.append(line)
        previous_non_empty = line

    normalized = "\n".join(cleaned_lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _repair_spaced_line(line: str) -> str:
    """Collapse lines like 'S K I L L S' into 'SKILLS'."""

    groups = re.split(r"\s{2,}", line)
    repaired_groups = [_collapse_spaced_group(group) for group in groups if group.strip()]
    if repaired_groups:
        return " ".join(repaired_groups)
    return line.strip()


def _collapse_spaced_group(group: str) -> str:
    tokens = group.strip().split()
    if len(tokens) < 3:
        return group.strip()

    mergeable = 0
    for token in tokens:
        if len(token) == 1 and re.fullmatch(r"[\w@.+:/#&%_-]", token):
            mergeable += 1

    if mergeable / len(tokens) < 0.65:
        return group.strip()

    merged = "".join(tokens)
    merged = re.sub(r"(?i)^www(?=[A-Za-z0-9])", "www.", merged)
    merged = merged.replace("..", ".")
    return merged


def _normalize_contact_spacing(text: str) -> str:
    """Repair emails, URLs, and phones when spaces are inserted everywhere."""

    text = re.sub(r"\s*@\s*", "@", text)
    text = re.sub(r"\s*\.\s*", ".", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s*\(\s*", "(", text)
    text = re.sub(r"\s*\)\s*", ")", text)
    text = re.sub(r"(?<!\+)\+\s+", "+", text)
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)

    tokens = []
    for token in text.split():
        tokens.append(_collapse_doubled_token(token))
    return " ".join(tokens)


def _repair_compound_headers(line: str) -> str:
    replacements = {
        "DEGREES PROGRAMMING": "DEGREES\nPROGRAMMING",
        "LANGUAGES TALKS": "LANGUAGES\nTALKS",
        "CERTIFICATES & GRANTS PUBLICATIONS": "CERTIFICATES & GRANTS\nPUBLICATIONS",
    }
    updated = line
    for source, target in replacements.items():
        updated = updated.replace(source, target)
    return updated


def _collapse_doubled_token(token: str) -> str:
    """Repair noise like 'AAuugg' -> 'Aug' while leaving normal words alone."""

    if len(token) < 6:
        return token

    doubled_pairs = sum(1 for index in range(1, len(token)) if token[index] == token[index - 1])
    if doubled_pairs / len(token) < 0.25:
        return token

    collapsed = [token[0]]
    for character in token[1:]:
        if character == collapsed[-1]:
            continue
        collapsed.append(character)
    return "".join(collapsed)
