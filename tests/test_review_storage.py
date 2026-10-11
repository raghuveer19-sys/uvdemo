"""Persistence tests use only temporary databases and synthetic document bytes."""

from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from test_documents import synthetic_pdf
from uvdemo.documents import extract_document
from uvdemo.storage import (
    claim_initial_analysis, get_case, get_case_history, get_document, get_latest_analysis, get_latest_review,
    initialize_database, list_cases, save_analysis, save_document_extraction,
    save_intake, save_review,
)


def sample_intake():
    return {"customer_name": "Fictional Acorn Services",
            "description": "A synthetic services agreement request.",
            "customer_segment": "Unknown", "handling_approach": "Unknown"}


def sample_review():
    return {"contract_type": "Services agreement", "customer_segment": "Government",
            "handling_approach": "Assisted/Custom", "digital_eligibility": "No",
            "facts_confirmed": True, "issues_resolved": True,
            "notes": "Reviewer confirmed the synthetic customer is a government department."}


def sample_document():
    # Storage treats bytes as opaque; PDF parsing has its own tests.
    return {"filename": "request.pdf", "content": b"%PDF-1.4\nstorage-only synthetic fixture"}


def sample_extraction(document, status="Ready"):
    text = "A synthetic services agreement." if status == "Ready" else ""
    return {"filename": document["filename"], "sha256": document["sha256"],
            "size_bytes": document["size_bytes"], "status": status, "text": text,
            "pages": [{"page": 1, "text": text}] if text else [],
            "error": None if text else "No extractable text."}


class ReviewStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db = Path(temporary.name) / "cases.sqlite3"
        initialize_database(self.db)

    def create_case(self, document=None):
        # Most review tests start with extracted evidence; passing an explicit
        # document leaves extraction pending for the document-focused tests.
        case = save_intake(self.db, "attempt-1", sample_intake(),
                           document=sample_document() if document is None else document)
        if document is None:
            saved = get_document(self.db, case["case_id"])
            save_document_extraction(self.db, case["case_id"], sample_extraction(saved))
        return case

    def save_mock(self, case_id):
        return save_analysis(self.db, case_id, "mock", "development-mock",
                             {"summary": "A synthetic request.", "contradictions": []}, None)

    def fail_history_writes(self):
        with closing(sqlite3.connect(self.db)) as connection:
            connection.execute("""CREATE TRIGGER fail_history BEFORE INSERT ON case_events
                BEGIN SELECT RAISE(ABORT, 'Simulated history failure'); END""")

    def test_increment_one_database_is_upgraded_without_changing_saved_cases(self):
        old_db = self.db.parent / "increment_one.sqlite3"
        with closing(sqlite3.connect(old_db)) as connection:
            connection.executescript("""
                CREATE TABLE cases (case_id TEXT PRIMARY KEY, submission_token TEXT NOT NULL UNIQUE,
                    original_input_json TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE case_events (event_id INTEGER PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(case_id), event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL, details_json TEXT NOT NULL);
            """)
            with connection:
                connection.execute("INSERT INTO cases VALUES (?, ?, ?, ?, ?)",
                                   ("old-case", "old-token", json.dumps(sample_intake(), sort_keys=True),
                                    "Awaiting review", "2026-10-09T01:00:00+00:00"))
                connection.execute("INSERT INTO case_events VALUES (1, ?, ?, ?, ?)",
                                   ("old-case", "Submitted", "2026-10-09T01:00:00+00:00",
                                    json.dumps({"actor": "submitter"})))
        initialize_database(old_db)
        initialize_database(old_db)
        self.assertEqual(get_case(old_db, "old-case")["original_input"], sample_intake())
        self.assertEqual(get_case(old_db, "old-case")["status"], "Awaiting review")
        self.assertEqual(len(get_case_history(old_db, "old-case")), 1)
        self.assertIsNone(get_document(old_db, "old-case"))
        self.assertIsNone(get_latest_analysis(old_db, "old-case"))
        self.assertIsNone(get_latest_review(old_db, "old-case"))
        retried = save_intake(old_db, "old-token", sample_intake())
        self.assertEqual(retried["case_id"], "old-case")
        self.assertEqual(len(get_case_history(old_db, "old-case")), 1)
        with self.assertRaisesRegex(ValueError, "different document"):
            save_intake(old_db, "old-token", sample_intake(), sample_document())
        self.assertIsNone(get_document(old_db, "old-case"))
        analysis = save_analysis(old_db, "old-case", "mock", "development-mock", {}, None)
        self.assertEqual(analysis["case_id"], "old-case")

    def test_raw_document_and_extraction_survive_reopening(self):
        upload = sample_document()
        upload["filename"] = r"C:\fakepath\request.pdf"
        case = self.create_case(upload)
        document = get_document(self.db, case["case_id"])
        self.assertEqual(document["filename"], "request.pdf")
        self.assertEqual(document["content"], upload["content"])
        self.assertEqual(document["sha256"], hashlib.sha256(upload["content"]).hexdigest())
        self.assertIsNone(document["extraction"])
        extraction = sample_extraction(document)
        self.assertEqual(save_document_extraction(self.db, case["case_id"], extraction), extraction)
        initialize_database(self.db)
        reopened = get_document(self.db, case["case_id"])
        self.assertEqual(reopened["content"], upload["content"])
        self.assertEqual(reopened["extraction"], extraction)
        self.assertEqual(get_case(self.db, case["case_id"])["original_input"], sample_intake())

    def test_filename_normalization_matches_between_storage_and_extraction(self):
        content = synthetic_pdf(["Synthetic services agreement"])
        case = self.create_case({"filename": "bad:name.pdf", "content": content})
        extraction = extract_document("bad:name.pdf", content)
        save_document_extraction(self.db, case["case_id"], extraction)
        document = get_document(self.db, case["case_id"])
        self.assertEqual(document["filename"], "bad_name.pdf")
        self.assertEqual(document["extraction"]["status"], "Ready")

    def test_attachment_retry_is_idempotent_and_cannot_replace_upload(self):
        case = self.create_case(sample_document())
        self.assertEqual(self.create_case(sample_document()), case)
        changed_bytes = {"filename": "request.pdf", "content": b"different bytes"}
        changed_name = {"filename": "other.pdf", "content": sample_document()["content"]}
        for changed in (None, changed_bytes, changed_name):
            with self.subTest(document=changed):
                with self.assertRaisesRegex(ValueError, "different document"):
                    save_intake(self.db, "attempt-1", sample_intake(), changed)
        self.assertEqual(get_document(self.db, case["case_id"])["content"],
                         sample_document()["content"])
        self.assertEqual(len(get_case_history(self.db, case["case_id"])), 1)

    def test_unreadable_pdf_is_preserved_with_extraction_failure(self):
        upload = {"filename": "request.PDF", "content": b"not a readable PDF"}
        case = self.create_case(upload)
        failed = extract_document(upload["filename"], upload["content"])
        self.assertEqual(failed["status"], "Needs attention")
        save_document_extraction(self.db, case["case_id"], failed)
        stored = get_document(self.db, case["case_id"])
        self.assertEqual(stored["content"], upload["content"])
        self.assertEqual(stored["extraction"], failed)
        self.assertEqual(get_case(self.db, case["case_id"])["status"], "Document needs attention")
        self.assertEqual([event["event_type"] for event in get_case_history(self.db, case["case_id"])],
                         ["Submitted", "Document extraction failed"])

    def test_attachment_and_case_roll_back_if_history_write_fails(self):
        self.fail_history_writes()
        with self.assertRaises(sqlite3.IntegrityError):
            self.create_case(sample_document())
        self.assertEqual(list_cases(self.db), [])
        with closing(sqlite3.connect(self.db)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM case_documents").fetchone()[0], 0)

    def test_oversized_attachment_is_rejected_without_saving_case(self):
        with self.assertRaisesRegex(ValueError, "5 MiB"):
            self.create_case({"filename": "large.pdf", "content": b"x" * (5 * 1024 * 1024 + 1)})
        self.assertEqual(list_cases(self.db), [])

    def test_extraction_must_match_saved_document_and_ready_result_is_fixed(self):
        case = self.create_case(sample_document())
        extraction = sample_extraction(get_document(self.db, case["case_id"]))
        changed_hash = {**extraction, "sha256": "wrong hash"}
        with self.assertRaisesRegex(ValueError, "does not match"):
            save_document_extraction(self.db, case["case_id"], changed_hash)
        self.assertIsNone(get_document(self.db, case["case_id"])["extraction"])
        save_document_extraction(self.db, case["case_id"], extraction)
        save_document_extraction(self.db, case["case_id"], dict(reversed(list(extraction.items()))))
        with self.assertRaisesRegex(ValueError, "already has a Ready extraction"):
            save_document_extraction(self.db, case["case_id"], {**extraction, "text": "Changed text"})
        self.assertEqual(len(get_case_history(self.db, case["case_id"])), 2)

    def test_failed_extraction_can_be_retried_without_losing_original(self):
        case = self.create_case(sample_document())
        document = get_document(self.db, case["case_id"])
        failed = sample_extraction(document, "Needs attention")
        save_document_extraction(self.db, case["case_id"], failed)
        save_document_extraction(self.db, case["case_id"], failed)
        ready = sample_extraction(document)
        save_document_extraction(self.db, case["case_id"], ready)
        self.assertEqual(get_document(self.db, case["case_id"])["extraction"], ready)
        self.assertEqual(get_document(self.db, case["case_id"])["content"], sample_document()["content"])
        self.assertEqual(len(get_case_history(self.db, case["case_id"])), 3)

    def test_analysis_failure_is_latest_attempt_and_preserves_case(self):
        case = self.create_case()
        first = self.save_mock(case["case_id"])
        self.assertEqual(get_case(self.db, case["case_id"])["status"], "Allocation pending")
        failed = save_analysis(self.db, case["case_id"], "live", "test-model", None,
                               "The model request failed.")
        self.assertGreater(failed["analysis_id"], first["analysis_id"])
        self.assertEqual(get_latest_analysis(self.db, case["case_id"]), failed)
        self.assertIsNone(failed["result"])
        self.assertEqual(get_case(self.db, case["case_id"])["original_input"], sample_intake())
        self.assertEqual(get_case(self.db, case["case_id"])["status"], "Analysis failed")

    def test_review_preserves_original_values_and_duplicate_confirmation_is_idempotent(self):
        case = self.create_case()
        analysis = self.save_mock(case["case_id"])
        review = save_review(self.db, case["case_id"], sample_review(), analysis["analysis_id"])
        repeated = save_review(self.db, case["case_id"], sample_review(), analysis["analysis_id"])
        self.assertEqual(repeated, review)
        self.assertEqual(get_latest_review(self.db, case["case_id"]), review)
        self.assertEqual(review["review"]["customer_segment"], "Government")
        self.assertEqual(get_case(self.db, case["case_id"])["original_input"]["customer_segment"], "Unknown")
        self.assertEqual(get_case(self.db, case["case_id"])["status"], "Reviewed - unassigned")
        self.assertEqual(len(get_case_history(self.db, case["case_id"])), 4)

    def test_new_analysis_invalidates_review_and_rejects_stale_confirmation(self):
        case = self.create_case()
        first = self.save_mock(case["case_id"])
        saved_review = save_review(self.db, case["case_id"], sample_review(), first["analysis_id"])
        failed = save_analysis(self.db, case["case_id"], "live", "test-model", None, "Service unavailable.")
        self.assertIsNone(get_latest_review(self.db, case["case_id"]))
        self.assertEqual(get_case(self.db, case["case_id"])["status"], "Analysis failed")
        with self.assertRaisesRegex(ValueError, "analysis changed"):
            save_review(self.db, case["case_id"], sample_review(), first["analysis_id"])
        with closing(sqlite3.connect(self.db)) as connection:
            preserved = connection.execute("SELECT review_json FROM case_reviews WHERE review_id = ?",
                                           (saved_review["review_id"],)).fetchone()
        self.assertEqual(json.loads(preserved[0]), sample_review())
        manual = save_review(self.db, case["case_id"], sample_review(), failed["analysis_id"])
        self.assertEqual(manual["analysis_id"], failed["analysis_id"])

    def test_pending_attachment_extraction_blocks_review(self):
        case = self.create_case(sample_document())
        with self.assertRaisesRegex(ValueError, "Extract the uploaded document"):
            save_review(self.db, case["case_id"], sample_review(), None)
        self.assertIsNone(get_latest_review(self.db, case["case_id"]))

    def test_changed_extraction_invalidates_review_but_identical_retry_does_not(self):
        case = self.create_case(sample_document())
        document = get_document(self.db, case["case_id"])
        failed = sample_extraction(document, "Needs attention")
        save_document_extraction(self.db, case["case_id"], failed)
        review = save_review(self.db, case["case_id"], sample_review(), None)
        self.assertEqual(review["document_extraction"], failed)
        save_document_extraction(self.db, case["case_id"], failed)
        self.assertEqual(get_latest_review(self.db, case["case_id"]), review)
        save_document_extraction(self.db, case["case_id"], sample_extraction(document))
        self.assertIsNone(get_latest_review(self.db, case["case_id"]))
        self.assertEqual(get_case(self.db, case["case_id"])["status"], "Analysis needed")
        with closing(sqlite3.connect(self.db)) as connection:
            snapshot = connection.execute(
                "SELECT document_extraction_json FROM case_reviews WHERE review_id = ?",
                (review["review_id"],),
            ).fetchone()[0]
        self.assertEqual(json.loads(snapshot), failed)

    def test_extraction_history_failure_preserves_current_review_and_text(self):
        case = self.create_case(sample_document())
        document = get_document(self.db, case["case_id"])
        failed = sample_extraction(document, "Needs attention")
        save_document_extraction(self.db, case["case_id"], failed)
        review = save_review(self.db, case["case_id"], sample_review(), None)
        self.fail_history_writes()
        with self.assertRaises(sqlite3.IntegrityError):
            save_document_extraction(self.db, case["case_id"], sample_extraction(document))
        self.assertEqual(get_document(self.db, case["case_id"])["extraction"], failed)
        self.assertEqual(get_latest_review(self.db, case["case_id"]), review)

    def test_review_validation_failure_does_not_change_status_or_history(self):
        case = self.create_case()
        invalid = {**sample_review(), "facts_confirmed": False}
        with self.assertRaises(ValueError):
            save_review(self.db, case["case_id"], invalid, None)
        self.assertEqual(get_case(self.db, case["case_id"])["status"], "Analysis needed")
        self.assertEqual(len(get_case_history(self.db, case["case_id"])), 2)

    def test_analysis_history_failure_rolls_back_attempt_and_preserves_current_review(self):
        case = self.create_case()
        review = save_review(self.db, case["case_id"], sample_review(), None)
        self.fail_history_writes()
        with self.assertRaises(sqlite3.IntegrityError):
            self.save_mock(case["case_id"])
        self.assertIsNone(get_latest_analysis(self.db, case["case_id"]))
        self.assertEqual(get_latest_review(self.db, case["case_id"]), review)

    def test_review_history_failure_rolls_back_confirmation_and_status(self):
        case = self.create_case()
        self.fail_history_writes()
        with self.assertRaises(sqlite3.IntegrityError):
            save_review(self.db, case["case_id"], sample_review(), None)
        self.assertIsNone(get_latest_review(self.db, case["case_id"]))
        self.assertEqual(get_case(self.db, case["case_id"])["status"], "Analysis needed")
        with closing(sqlite3.connect(self.db)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM case_reviews").fetchone()[0], 0)

    def test_list_cases_returns_latest_first_even_when_timestamps_match(self):
        first = self.create_case()
        second = save_intake(self.db, "attempt-2", sample_intake(), sample_document())
        with closing(sqlite3.connect(self.db)) as connection:
            with connection:
                connection.execute("UPDATE cases SET created_at = ?", (first["created_at"],))
        self.assertEqual([case["case_id"] for case in list_cases(self.db)],
                         [second["case_id"], first["case_id"]])

    def test_initial_analysis_claim_survives_reopening_and_allows_explicit_retry(self):
        case = self.create_case()
        case_id = case["case_id"]
        self.assertTrue(claim_initial_analysis(self.db, case_id, "live", "test-model"))
        initialize_database(self.db)
        self.assertFalse(claim_initial_analysis(self.db, case_id, "mock", "development-mock"))
        events = get_case_history(self.db, case_id)
        self.assertEqual(events[-1]["event_type"], "Initial AI analysis started")
        self.assertEqual(events[-1]["details"],
                         {"actor": "AI", "mode": "live", "model": "test-model"})
        self.assertEqual(len(events), 3)
        self.assertIsNone(get_latest_analysis(self.db, case_id))
        save_analysis(self.db, case_id, "live", "test-model", None, "Service unavailable.")
        self.assertFalse(claim_initial_analysis(self.db, case_id, "live", "test-model"))
        explicit_retry = self.save_mock(case_id)
        self.assertEqual(get_latest_analysis(self.db, case_id), explicit_retry)

    def test_concurrent_initial_analysis_claims_reserve_one_attempt(self):
        case_id = self.create_case()["case_id"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(claim_initial_analysis, self.db, case_id, "mock", "development-mock")
                       for _ in range(2)]
            self.assertEqual(sorted(future.result() for future in futures), [False, True])
        with closing(sqlite3.connect(self.db)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM initial_analysis_claims").fetchone()[0], 1)
        self.assertEqual([event["event_type"] for event in get_case_history(self.db, case_id)].count(
            "Initial AI analysis started"), 1)

    def test_saved_analysis_prevents_a_new_automatic_attempt(self):
        for index, (result, error) in enumerate((({}, None), (None, "Service unavailable."))):
            with self.subTest(error=error):
                case = save_intake(self.db, f"attempt-{index}", sample_intake(), sample_document())
                save_analysis(self.db, case["case_id"], "mock", "development-mock", result, error)
                before = get_case_history(self.db, case["case_id"])
                self.assertFalse(claim_initial_analysis(self.db, case["case_id"], "mock", "development-mock"))
                self.assertEqual(get_case_history(self.db, case["case_id"]), before)
        with closing(sqlite3.connect(self.db)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM initial_analysis_claims").fetchone()[0], 0)

    def test_claim_history_failure_rolls_back_reservation_and_allows_retry(self):
        case_id = self.create_case()["case_id"]
        self.fail_history_writes()
        with self.assertRaises(sqlite3.IntegrityError):
            claim_initial_analysis(self.db, case_id, "live", "test-model")
        with closing(sqlite3.connect(self.db)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM initial_analysis_claims").fetchone()[0], 0)
            connection.execute("DROP TRIGGER fail_history")
        self.assertEqual(len(get_case_history(self.db, case_id)), 2)
        self.assertTrue(claim_initial_analysis(self.db, case_id, "live", "test-model"))

    def test_claim_requires_an_existing_case_and_valid_mode_and_model(self):
        case_id = self.create_case()["case_id"]
        for arguments in (("missing", "mock", "development-mock"), (case_id, "other", "model"),
                          (case_id, "live", ""), (case_id, "live", None)):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    claim_initial_analysis(self.db, *arguments)
        self.assertEqual(len(get_case_history(self.db, case_id)), 2)
        with closing(sqlite3.connect(self.db)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM initial_analysis_claims").fetchone()[0], 0)

    def test_assigned_case_keeps_its_analysis_review_and_status_fixed(self):
        case_id = self.create_case()["case_id"]
        analysis = self.save_mock(case_id)
        save_review(self.db, case_id, sample_review(), analysis["analysis_id"])
        # Represent the allocation module's saved assignment without requiring
        # that module in these focused storage tests.
        with closing(sqlite3.connect(self.db)) as connection:
            with connection:
                connection.execute("CREATE TABLE case_assignments (case_id TEXT PRIMARY KEY)")
                connection.execute("INSERT INTO case_assignments VALUES (?)", (case_id,))
                connection.execute("UPDATE cases SET status = 'Assigned' WHERE case_id = ?", (case_id,))
        before = get_case_history(self.db, case_id)
        with self.assertRaisesRegex(ValueError, "already assigned"):
            self.save_mock(case_id)
        with self.assertRaisesRegex(ValueError, "already assigned"):
            save_review(self.db, case_id, sample_review(), analysis["analysis_id"])
        document = get_document(self.db, case_id)
        self.assertEqual(save_document_extraction(self.db, case_id, document["extraction"]),
                         document["extraction"])
        with self.assertRaisesRegex(ValueError, "already assigned"):
            save_document_extraction(self.db, case_id, {**document["extraction"], "text": "Changed text"})
        self.assertEqual(get_document(self.db, case_id)["extraction"], document["extraction"])
        self.assertEqual(get_latest_analysis(self.db, case_id), analysis)
        self.assertEqual(get_case(self.db, case_id)["status"], "Assigned")
        self.assertEqual(get_case_history(self.db, case_id), before)
        with closing(sqlite3.connect(self.db)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM case_reviews").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
