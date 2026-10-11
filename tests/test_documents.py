from hashlib import sha256
from io import BytesIO
import logging
import unittest
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from uvdemo.documents import MAX_FILE_BYTES, MAX_PAGES, MAX_TEXT_CHARS, extract_document


def synthetic_pdf(page_texts: list[str | None], password: str | None = None) -> bytes:
    """Build tiny, in-memory test inputs only; None represents a blank page."""
    writer = PdfWriter()
    for text in page_texts:
        page = writer.add_blank_page(width=300, height=300)
        if text is not None:
            font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                     NameObject("/Subtype"): NameObject("/Type1"),
                                     NameObject("/BaseFont"): NameObject("/Helvetica")})
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
            stream = DecodedStreamObject()
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream.set_data(f"BT /F1 12 Tf 20 250 Td ({escaped}) Tj ET".encode("ascii"))
            page[NameObject("/Contents")] = stream
    if password is not None:
        writer.encrypt(password)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


class DocumentTests(unittest.TestCase):
    def assert_attention(self, document):
        self.assertEqual(document["status"], "Needs attention")
        self.assertEqual(document["text"], "")
        self.assertEqual(document["pages"], [])
        self.assertTrue(document["error"])

    def test_selectable_text_is_preserved_with_page_evidence_and_identity(self):
        content = synthetic_pdf(["Synthetic services agreement", "Government department"])
        result = extract_document(r"C:\fakepath\agreement.PDF", content)
        self.assertEqual(set(result), {"filename", "sha256", "size_bytes", "status",
                                      "text", "pages", "error"})
        self.assertEqual(result["filename"], "agreement.PDF")
        self.assertEqual(result["sha256"], sha256(content).hexdigest())
        self.assertEqual(result["size_bytes"], len(content))
        self.assertEqual(result["status"], "Ready")
        self.assertIsNone(result["error"])
        self.assertEqual([page["page"] for page in result["pages"]], [1, 2])
        self.assertEqual(result["pages"][1]["text"], "Government department")
        self.assertEqual(result["text"], "Synthetic services agreement\n\nGovernment department")

    def test_names_are_basename_only_and_control_characters_are_removed(self):
        result = extract_document("../../bad\x00:name.pdf", synthetic_pdf(["Synthetic text"]))
        self.assertEqual(result["filename"], "bad__name.pdf")
        self.assertEqual(result["status"], "Ready")

    def test_empty_wrong_type_bad_header_and_malformed_inputs_are_friendly(self):
        for filename, content in [("empty.pdf", b""), ("notes.txt", b"Synthetic"),
                                  ("wrong.pdf", b"Synthetic"),
                                  ("broken.pdf", b"%PDF-1.7\nSYNTHETIC PRIVATE CONTENT")]:
            with self.subTest(filename=filename):
                with self.assertNoLogs("pypdf", level="WARNING"):
                    result = extract_document(filename, content)
                self.assert_attention(result)
                self.assertNotIn("PRIVATE CONTENT", result["error"])

    def test_scanned_or_blank_page_blocks_partial_document_text(self):
        for page_texts in ([None], ["Selectable first page", None], ["   "]):
            with self.subTest(page_texts=page_texts):
                self.assert_attention(extract_document("blank.pdf", synthetic_pdf(page_texts)))

    def test_zero_pages_is_not_a_ready_document(self):
        self.assert_attention(extract_document("empty.pdf", synthetic_pdf([])))

    def test_encrypted_pdf_is_rejected_even_with_empty_password(self):
        for password in ("", "synthetic-password"):
            with self.subTest(password=password):
                self.assert_attention(extract_document("encrypted.pdf", synthetic_pdf(["Synthetic"], password)))

    def test_file_limit_prevents_parser_use(self):
        content = b"%PDF-" + b" " * MAX_FILE_BYTES
        with patch("uvdemo.documents.PdfReader") as reader:
            self.assert_attention(extract_document("large.pdf", content))
        reader.assert_not_called()

    def test_page_limit_accepts_boundary_and_rejects_excess(self):
        self.assertEqual(extract_document("allowed.pdf", synthetic_pdf(["Text"] * MAX_PAGES))["status"], "Ready")
        self.assert_attention(extract_document("large.pdf", synthetic_pdf(["Text"] * (MAX_PAGES + 1))))

    def test_text_limit_accepts_boundary_and_rejects_excess(self):
        self.assertEqual(extract_document("allowed.pdf", synthetic_pdf(["a" * MAX_TEXT_CHARS]))["status"], "Ready")
        self.assert_attention(extract_document("large.pdf", synthetic_pdf(["a" * (MAX_TEXT_CHARS + 1)])))

    def test_parser_error_has_no_partial_text_or_raw_exception(self):
        with patch("uvdemo.documents.PdfReader", side_effect=ValueError("SYNTHETIC PRIVATE CONTENT")):
            result = extract_document("bad.pdf", b"%PDF-1.7")
        self.assert_attention(result)
        self.assertNotIn("PRIVATE CONTENT", result["error"])

    def test_library_warning_is_private_and_logging_configuration_is_restored(self):
        logger = logging.getLogger("pypdf._reader")
        filters_before = list(logger.filters)
        level_before = logger.level

        def parser_warning(*args, **kwargs):
            logger.warning("SYNTHETIC PRIVATE CONTENT")
            raise ValueError("Unable to parse")

        with self.assertNoLogs("pypdf", level="WARNING"):
            with patch("uvdemo.documents.PdfReader", side_effect=parser_warning):
                self.assert_attention(extract_document("warning.pdf", b"%PDF-1.7"))
        self.assertEqual(logger.filters, filters_before)
        self.assertEqual(logger.level, level_before)


if __name__ == "__main__":
    unittest.main()
