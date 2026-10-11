"""Extract selectable PDF text for this demo; no OCR or legal validation."""

from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
import logging
import re
import sys
from threading import get_ident
import warnings

from pypdf import PdfReader


# Technical demo limits, never rules for customer eligibility or allocation.
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_PAGES = 25
MAX_TEXT_CHARS = 60_000


class _PdfLogFilter(logging.Filter):
    """Keep parser messages out of logs for this extraction's thread only."""

    def __init__(self):
        super().__init__()
        self.thread_id = get_ident()
        self.warning_seen = False

    def filter(self, record):
        if record.thread != self.thread_id:
            return True
        if record.levelno >= logging.WARNING:
            self.warning_seen = True
        return False


@contextmanager
def _quiet_pdf_parser():
    # pypdf logs through its module names. Temporary, thread-specific filters
    # avoid exposing parser messages without changing application log levels.
    # They are removed even if parsing fails; other threads remain unaffected.
    pdf_filter = _PdfLogFilter()
    loggers = [logging.getLogger(name) for name in tuple(sys.modules)
               if name == "pypdf" or name.startswith("pypdf.")]
    for logger in loggers:
        logger.addFilter(pdf_filter)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", module=r"pypdf(?:\..*)?")
            yield pdf_filter
    finally:
        for logger in loggers:
            logger.removeFilter(pdf_filter)


def extract_document(filename: str, content: bytes) -> dict:
    """Return a complete extraction or a friendly, retryable failure result.

    Preserve the original upload separately. A failure contains no partial text
    so it cannot accidentally be used as a complete document for AI analysis.
    Text extraction cannot establish correctness or detect every mixed image /
    text page. Reviewers must check the extracted text against their synthetic PDF.
    """
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    basename = re.sub(r'[<>:"|?*\x00-\x1f]', "_", basename)
    result = {
        "filename": basename,
        "sha256": sha256(content).hexdigest(),
        "size_bytes": len(content),
        "status": "Needs attention",
        "text": "",
        "pages": [],
        "error": None,
    }

    def attention(message: str) -> dict:
        result["error"] = message
        return result

    if not basename.lower().endswith(".pdf"):
        return attention("Choose a PDF file ending in .pdf.")
    if not content:
        return attention("The uploaded PDF is empty. Choose a text-based PDF.")
    if len(content) > MAX_FILE_BYTES:
        return attention("This demo accepts PDFs up to 5 MB. Choose a smaller PDF.")
    if not content.startswith(b"%PDF-"):
        return attention("The file does not have a valid PDF header. Choose a PDF file.")

    try:
        with _quiet_pdf_parser() as parser_messages:
            reader = PdfReader(BytesIO(content), strict=True)
            if reader.is_encrypted:
                return attention("Encrypted PDFs are not supported. Choose an unencrypted PDF.")
            if not reader.pages:
                return attention("The PDF has no pages. Choose a text-based PDF.")
            if len(reader.pages) > MAX_PAGES:
                return attention("This demo accepts PDFs with up to 25 pages.")

            pages = []
            character_count = 0
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if not page_text or not page_text.strip():
                    return attention(
                        f"Page {page_number} has no extractable text. Scanned or blank "
                        "pages are not supported; choose a PDF with selectable text on every page."
                    )
                character_count += len(page_text) + (2 if pages else 0)
                if character_count > MAX_TEXT_CHARS:
                    return attention("The PDF exceeds this demo's 60,000-character text limit.")
                pages.append({"page": page_number, "text": page_text})
            if parser_messages.warning_seen:
                return attention("The PDF could not be read reliably. Choose a simpler text-based PDF.")
    except Exception:
        # PDFs are external inputs: parser errors must not lose the request or
        # expose document contents in a traceback. Never return raw exceptions.
        return attention("The PDF could not be read. Choose a valid, unencrypted text-based PDF.")

    result.update(status="Ready", text="\n\n".join(page["text"] for page in pages), pages=pages)
    return result
