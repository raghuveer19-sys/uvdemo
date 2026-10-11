from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from uvdemo.intake import validate_intake
from uvdemo.storage import (
    get_case, get_case_by_submission, get_case_history, initialize_database, save_intake,
)


def sample_intake() -> dict:
    return {"customer_name": "Fictional Acorn Services", "description": "A synthetic request.",
            "customer_segment": "Unknown", "handling_approach": "Unknown"}


def sample_document() -> dict:
    return {"filename": "request.pdf", "content": b"%PDF-storage-only synthetic fixture"}


class IntakeStorageTests(unittest.TestCase):
    def test_explicit_digital_eligibility_is_validated_and_preserved(self):
        for value in ("Unknown", "Yes", "No"):
            intake = {**sample_intake(), "digital_eligibility": value}
            self.assertEqual(validate_intake(intake), {})
            case = save_intake(self.db, f"digital-{value}", intake, sample_document())
            self.assertEqual(get_case(self.db, case["case_id"])["original_input"], intake)
        for value in (None, True, "Maybe"):
            self.assertIn("digital_eligibility", validate_intake(
                {**sample_intake(), "digital_eligibility": value}))

    def test_missing_customer_segment(self):
        intake = sample_intake()
        del intake["customer_segment"]
        errors = validate_intake(intake)
        self.assertIn("customer_segment", errors)

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db = Path(temporary.name) / "cases.sqlite3"
        initialize_database(self.db)

    def counts(self):
        with closing(sqlite3.connect(self.db)) as connection:
            return (connection.execute("SELECT COUNT(*) FROM cases").fetchone()[0],
                    connection.execute("SELECT COUNT(*) FROM case_events").fetchone()[0])

    def test_blank_required_text_is_rejected_without_saving(self):
        for field in ("customer_name", "description"):
            for value in ("", " \n\t", None):
                with self.subTest(field=field, value=value):
                    intake = sample_intake()
                    intake[field] = value
                    self.assertIn(field, validate_intake(intake))
                    with self.assertRaises(ValueError):
                        save_intake(self.db, "attempt-1", intake, sample_document())
        self.assertEqual(self.counts(), (0, 0))

    def test_invalid_categories_are_rejected(self):
        for field in ("customer_segment", "handling_approach"):
            intake = sample_intake()
            intake[field] = "Invented category"
            self.assertIn(field, validate_intake(intake))
            with self.assertRaises(ValueError):
                save_intake(self.db, "attempt-1", intake, sample_document())
        self.assertEqual(self.counts(), (0, 0))

    def test_unknowns_are_saved_as_unconfirmed_needing_analysis(self):
        case = save_intake(self.db, "attempt-1", sample_intake(), sample_document())
        self.assertEqual(case["status"], "Analysis needed")
        self.assertEqual(case["original_input"]["customer_segment"], "Unknown")
        self.assertEqual(self.counts(), (1, 1))

    def test_original_text_survives_database_reopening(self):
        intake = sample_intake()
        intake["customer_name"] = "  Fictional O'Brien & Co  "
        intake["description"] = "Line one.\n  Line two: café.  "
        case = save_intake(self.db, "attempt-1", intake, sample_document())
        initialize_database(self.db)
        self.assertEqual(get_case(self.db, case["case_id"])["original_input"], intake)
        self.assertEqual(get_case_history(self.db, case["case_id"])[0]["event_type"], "Submitted")

    def test_segment_conflict_is_preserved_for_analysis(self):
        intake = sample_intake()
        intake["customer_segment"] = "Medium Business"
        intake["description"] = "The contracting customer is a fictional government department."
        case = save_intake(self.db, "attempt-1", intake, sample_document())
        self.assertEqual(case["original_input"], intake)
        self.assertEqual(case["status"], "Analysis needed")

    def test_retry_returns_same_case_and_does_not_repeat_history(self):
        first = save_intake(self.db, "attempt-1", sample_intake(), sample_document())
        reordered = dict(reversed(list(sample_intake().items())))
        second = save_intake(self.db, "attempt-1", reordered, sample_document())
        self.assertEqual(first, second)
        self.assertEqual(self.counts(), (1, 1))

    def test_changed_input_cannot_overwrite_saved_submission(self):
        case = save_intake(self.db, "attempt-1", sample_intake(), sample_document())
        changed = sample_intake()
        changed["description"] = "Different request"
        with self.assertRaisesRegex(ValueError, "different input"):
            save_intake(self.db, "attempt-1", changed, sample_document())
        self.assertEqual(get_case(self.db, case["case_id"])["original_input"], sample_intake())
        self.assertEqual(self.counts(), (1, 1))

    def test_intentional_new_request_can_have_identical_content(self):
        first = save_intake(self.db, "attempt-1", sample_intake(), sample_document())
        second = save_intake(self.db, "attempt-2", sample_intake(), sample_document())
        self.assertNotEqual(first["case_id"], second["case_id"])
        self.assertEqual(self.counts(), (2, 2))

    def test_concurrent_retries_create_one_case_and_one_event(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(save_intake, self.db, "attempt-1", sample_intake(), sample_document())
                       for _ in range(2)]
            results = [future.result() for future in futures]
        self.assertEqual(results[0]["case_id"], results[1]["case_id"])
        self.assertEqual(self.counts(), (1, 1))

    def test_history_failure_rolls_back_case_and_allows_retry(self):
        with closing(sqlite3.connect(self.db)) as connection:
            connection.execute("""CREATE TRIGGER fail_history BEFORE INSERT ON case_events
                BEGIN SELECT RAISE(ABORT, 'Simulated storage failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            save_intake(self.db, "attempt-1", sample_intake(), sample_document())
        self.assertEqual(self.counts(), (0, 0))
        with closing(sqlite3.connect(self.db)) as connection:
            connection.execute("DROP TRIGGER fail_history")
        save_intake(self.db, "attempt-1", sample_intake(), sample_document())
        self.assertEqual(self.counts(), (1, 1))

    def test_lookup_of_unknown_case_or_submission_returns_none(self):
        self.assertIsNone(get_case(self.db, "missing"))
        self.assertIsNone(get_case_by_submission(self.db, "missing"))

    def test_new_submission_requires_a_nonempty_pdf_without_partial_saves(self):
        for document in (None, {"filename": "request.pdf", "content": b""},
                         {"filename": "request.txt", "content": b"Some text"}):
            with self.subTest(document=document):
                with self.assertRaisesRegex(ValueError, "PDF"):
                    save_intake(self.db, "attempt-1", sample_intake(), document)
                self.assertEqual(self.counts(), (0, 0))
        # The same token can be corrected after validation failed.
        case = save_intake(self.db, "attempt-1", sample_intake(), sample_document())
        self.assertEqual(case["status"], "Analysis needed")
        self.assertEqual(self.counts(), (1, 1))


if __name__ == "__main__":
    unittest.main()
