"""Progress changes preserve allocation evidence and record one durable event."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import copy
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest

from uvdemo.allocation import allocate_case, initialize_allocation
from uvdemo.group_allocation import (
    allocate_group, get_group_assignment, get_group_decision, initialize_group_allocation,
)
from uvdemo.storage import (
    get_case, get_case_history, get_document, get_latest_analysis, initialize_database,
    save_analysis, save_document_extraction, save_intake, save_review,
)
from uvdemo.tracking import filter_tracker_rows, update_case_status


class TrackingTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db = Path(temporary.name) / "tracking.sqlite3"
        initialize_database(self.db)
        initialize_group_allocation(self.db)
        self.case_number = 0

    def execute(self, sql, parameters=()):
        with closing(sqlite3.connect(self.db)) as connection:
            with connection:
                return connection.execute(sql, parameters).fetchall()

    def create_case(self, *, allocated=True, conflict=False, legacy=False):
        self.case_number += 1
        case = save_intake(self.db, f"submission-{self.case_number}", {
            "customer_name": "Fictional Harbour Customer", "description": "A synthetic request.",
            "customer_segment": "Unknown", "handling_approach": "Unknown",
        }, {"filename": "synthetic.pdf", "content": b"%PDF-1.4 synthetic storage fixture"})
        case_id = case["case_id"]
        text = "Customer segment: Government"
        document = get_document(self.db, case_id)
        save_document_extraction(self.db, case_id, {
            "filename": document["filename"], "sha256": document["sha256"],
            "size_bytes": document["size_bytes"], "status": "Ready", "text": text,
            "pages": [{"page": 1, "text": text}], "error": None,
        })
        result = {
            "summary": "Synthetic request.", "contract_type": "Unknown",
            "customer_segment": "Government", "handling_approach": "Unknown",
            "digital_eligibility": "Unknown",
            "facts": [{"fact": "The document reports the customer segment.", "source": "document",
                       "page": 1, "quote": text}],
            "contradictions": ["Conflicting customer segments."] if conflict else [],
            "clarification_questions": [],
        }
        analysis = save_analysis(self.db, case_id, "mock", "mock-v1", result, None)
        if legacy:
            initialize_allocation(self.db)
            allocate_case(self.db, case_id, analysis["analysis_id"])
        elif allocated:
            allocate_group(self.db, case_id, analysis["analysis_id"])
        return case_id

    def progress_events(self, case_id):
        return [event for event in get_case_history(self.db, case_id)
                if event["event_type"] == "Case progress updated"]

    def test_progress_completion_and_reopening_persist_with_history(self):
        case_id = self.create_case()
        original = get_case(self.db, case_id)
        for previous, target in (("Allocated", "In progress"), ("In progress", "Completed"),
                                 ("Completed", "In progress")):
            saved = update_case_status(self.db, case_id, target, previous)
            self.assertEqual(saved["status"], target)
            self.assertEqual(saved, get_case(self.db, case_id))
            self.assertEqual({**saved, "status": original["status"]}, original)
            self.assertEqual(self.progress_events(case_id)[-1]["details"], {
                "actor": "demo user", "previous_status": previous,
                "status": target, "queue": "Government",
            })
        self.assertEqual(len(self.progress_events(case_id)), 3)

    def test_completed_allocation_and_evidence_remain_fixed(self):
        case_id = self.create_case()
        assignment = get_group_assignment(self.db, case_id)
        document = get_document(self.db, case_id)
        analysis = get_latest_analysis(self.db, case_id)
        for previous, target in (("Allocated", "In progress"), ("In progress", "Completed")):
            update_case_status(self.db, case_id, target, previous)
            self.assertEqual(get_group_assignment(self.db, case_id), assignment)
            self.assertEqual(get_group_decision(self.db, case_id), assignment)
        history = get_case_history(self.db, case_id)
        self.assertEqual(allocate_group(self.db, case_id, None), assignment)
        self.assertEqual(get_case(self.db, case_id)["status"], "Completed")
        with self.assertRaisesRegex(ValueError, "already allocated"):
            save_analysis(self.db, case_id, "mock", "mock-v1", analysis["result"], None)
        with self.assertRaisesRegex(ValueError, "already allocated"):
            save_review(self.db, case_id, {}, analysis["analysis_id"])
        with self.assertRaisesRegex(ValueError, "already allocated"):
            save_document_extraction(self.db, case_id, {**document["extraction"], "text": "Changed text."})
        self.assertEqual(get_document(self.db, case_id), document)
        self.assertEqual(get_latest_analysis(self.db, case_id), analysis)
        self.assertEqual(get_case_history(self.db, case_id), history)

    def test_duplicate_and_concurrent_updates_create_one_event(self):
        case_id = self.create_case()
        barrier = Barrier(2)

        def start(_):
            barrier.wait()
            return update_case_status(self.db, case_id, "In progress", "Allocated")

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(start, range(2)))
        self.assertEqual(results[0], results[1])
        self.assertEqual(update_case_status(self.db, case_id, "In progress", "Allocated"), results[0])
        self.assertEqual(len(self.progress_events(case_id)), 1)

    def test_stale_update_cannot_reopen_a_completed_case(self):
        case_id = self.create_case()
        update_case_status(self.db, case_id, "In progress", "Allocated")
        update_case_status(self.db, case_id, "Completed", "In progress")
        history = get_case_history(self.db, case_id)
        with self.assertRaisesRegex(ValueError, "status changed"):
            update_case_status(self.db, case_id, "In progress", "Allocated")
        self.assertEqual(get_case(self.db, case_id)["status"], "Completed")
        self.assertEqual(get_case_history(self.db, case_id), history)

    def test_invalid_statuses_and_skipped_or_backward_steps_are_rejected(self):
        case_id = self.create_case()
        history = get_case_history(self.db, case_id)
        for target, expected in (("Completed", "Allocated"), ("Rejected", "Allocated"),
                                 (None, "Allocated"), ([], "Allocated"),
                                 ("In progress", None), ("In progress", "invalid")):
            with self.subTest(target=target, expected=expected), self.assertRaises(ValueError):
                update_case_status(self.db, case_id, target, expected)
        self.assertEqual(get_case_history(self.db, case_id), history)
        update_case_status(self.db, case_id, "In progress", "Allocated")
        with self.assertRaisesRegex(ValueError, "Cannot change"):
            update_case_status(self.db, case_id, "Allocated", "In progress")
        self.assertEqual(get_case(self.db, case_id)["status"], "In progress")

    def test_unknown_unallocated_and_legacy_cases_cannot_progress(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            update_case_status(self.db, "missing", "In progress", "Allocated")
        for options in ({"allocated": False}, {"conflict": True}, {"legacy": True}):
            case_id = self.create_case(**options)
            before, history = get_case(self.db, case_id), get_case_history(self.db, case_id)
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "Only cases allocated"):
                update_case_status(self.db, case_id, "In progress", "Allocated")
            self.assertEqual(get_case(self.db, case_id), before)
            self.assertEqual(get_case_history(self.db, case_id), history)

    def test_database_without_group_tables_rejects_progress_cleanly(self):
        case_id = self.create_case(allocated=False)
        self.execute("DROP TABLE case_group_decisions")
        with self.assertRaisesRegex(ValueError, "Only cases allocated"):
            update_case_status(self.db, case_id, "In progress", "Allocated")

    def test_history_failure_rolls_back_status_and_allows_retry(self):
        case_id = self.create_case()
        history = get_case_history(self.db, case_id)
        self.execute("""CREATE TRIGGER fail_progress BEFORE INSERT ON case_events
                        WHEN NEW.event_type = 'Case progress updated'
                        BEGIN SELECT RAISE(ABORT, 'Simulated history failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            update_case_status(self.db, case_id, "In progress", "Allocated")
        self.assertEqual(get_case(self.db, case_id)["status"], "Allocated")
        self.assertEqual(get_case_history(self.db, case_id), history)
        self.execute("DROP TRIGGER fail_progress")
        self.assertEqual(update_case_status(self.db, case_id, "In progress", "Allocated")["status"], "In progress")
        self.assertEqual(len(self.progress_events(case_id)), 1)

    def test_allocation_display_still_hides_stale_evidence_or_unexpected_status(self):
        case_id = self.create_case()
        self.execute("UPDATE cases SET status = 'Unexpected' WHERE case_id = ?", (case_id,))
        self.assertIsNone(get_group_decision(self.db, case_id))
        with self.assertRaisesRegex(ValueError, "does not support"):
            update_case_status(self.db, case_id, "In progress", "Allocated")
        self.execute("UPDATE cases SET status = 'Completed' WHERE case_id = ?", (case_id,))
        self.assertIsNotNone(get_group_decision(self.db, case_id))
        self.execute("""INSERT INTO case_analyses(case_id, mode, model, result_json, error, created_at)
                        SELECT case_id, mode, model, result_json, error, created_at
                        FROM case_analyses WHERE case_id = ?""", (case_id,))
        self.assertIsNone(get_group_decision(self.db, case_id))


class TrackerFilterTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"Case ID": "Case-ABC", "Customer": "Acorn Council", "Status": "Allocated", "Group": "Government"},
            {"Case ID": "Case-DEF", "Customer": "Acorn Services", "Status": "Completed", "Group": "Enterprise"},
            {"Case ID": "Case-GHI", "Customer": "Birch Services", "Status": "In progress", "Group": "Government"},
            {"Case ID": "Case-JKL", "Customer": "Pending Customer", "Status": "Analysis needed", "Group": "Not allocated"},
        ]

    def test_search_matches_names_and_case_ids_ignoring_case_and_edge_spaces(self):
        self.assertEqual(filter_tracker_rows(self.rows, query="  aCoRn  "), self.rows[:2])
        self.assertEqual(filter_tracker_rows(self.rows, query=" CASE-def "), [self.rows[1]])
        self.assertEqual(filter_tracker_rows(self.rows, query="   "), self.rows)

    def test_status_group_and_search_combine_without_mutating_rows(self):
        original = copy.deepcopy(self.rows)
        self.assertEqual(filter_tracker_rows(self.rows, query="services", status="In progress",
                                             group="Government"), [self.rows[2]])
        self.assertEqual(filter_tracker_rows(self.rows, group="Government"), [self.rows[0], self.rows[2]])
        self.assertEqual(filter_tracker_rows(self.rows, status="Completed"), [self.rows[1]])
        self.assertEqual(filter_tracker_rows(self.rows, group="Not allocated"), [self.rows[3]])
        self.assertEqual(self.rows, original)

    def test_empty_and_nonmatching_filters_return_empty_lists(self):
        for options in ({"query": "missing"}, {"status": "allocated"}, {"group": "government"},
                        {"status": "Completed", "group": "Government"}):
            with self.subTest(options=options):
                self.assertEqual(filter_tracker_rows(self.rows, **options), [])
        self.assertEqual(filter_tracker_rows([]), [])


if __name__ == "__main__":
    unittest.main()
